"""Hybrid search: Chroma vectors + BM25, both ACL-filtered, merged with RRF.

    python -m app.search                                   # demo questions
    python -m app.search "what is an ICR?"
    python -m app.search --groups all,managers "when are salary reviews?"
"""
import re
import sys
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from app.config import RRF_K, SEARCH_K
from app.embeddings import embed_query
from app.vectordb import acl_filter, collection

STOPWORDS = set("a an and are as at be by do does for from how i in is it my of on or the to what when where who why with you your".split())


@dataclass
class Hit:
    id: str
    path: str
    title: str
    url: str
    heading: str
    text: str
    scores: dict = field(default_factory=dict)   # vector / keyword / rrf / rerank

    @property
    def label(self):
        return f"{self.title} > {self.heading}"


def _hit(id_, doc, meta):
    return Hit(id_, meta["path"], meta["title"], meta["url"], meta["heading"], doc)


def tokenize(text):
    # \w+ so albanian/macedonian words tokenize too
    return [w for w in re.findall(r"\w+", text.lower()) if len(w) > 1 and w not in STOPWORDS]


def vector_search(question, groups, k=SEARCH_K):
    res = collection().query(
        query_embeddings=[embed_query(question)], n_results=k,
        where=acl_filter(groups), include=["documents", "metadatas", "distances"],
    )
    hits = []
    for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        h = _hit(id_, doc, meta)
        h.scores["vector"] = round(1 - dist, 4)
        hits.append(h)
    return hits


def keyword_search(question, groups, k=SEARCH_K):
    # rebuilt on every call, fine for a few hundred chunks, cache it if the kb gets big
    data = collection().get(where=acl_filter(groups), include=["documents", "metadatas"])
    if not data["ids"]:
        return []
    corpus = [tokenize(f"{m['title']} {m['heading']} {d}") for d, m in zip(data["documents"], data["metadatas"])]
    scores = BM25Okapi(corpus).get_scores(tokenize(question))
    ranked = sorted(zip(scores, data["ids"], data["documents"], data["metadatas"]), key=lambda x: -x[0])
    hits = []
    for score, id_, doc, meta in ranked[:k]:
        if score <= 0:
            break
        h = _hit(id_, doc, meta)
        h.scores["keyword"] = round(float(score), 3)
        hits.append(h)
    return hits


def rrf(*lists, k=RRF_K):
    """Reciprocal rank fusion. Uses ranks only, so the vector and BM25 scores don't need to be comparable."""
    merged = {}
    for hits in lists:
        for rank, h in enumerate(hits, start=1):
            m = merged.setdefault(h.id, h)
            m.scores.update(h.scores)
            m.scores["rrf"] = m.scores.get("rrf", 0) + 1 / (k + rank)
    return sorted(merged.values(), key=lambda h: -h.scores["rrf"])


def hybrid_search(question, groups=("all",), k=SEARCH_K):
    vec = vector_search(question, list(groups), k)
    kw = keyword_search(question, list(groups), k)
    return rrf(vec, kw), vec, kw


def show(question, groups=("all",)):
    fused, vec, kw = hybrid_search(question, groups)
    print(f"\n=== {question!r}   groups={list(groups)}")
    for name, hits, key in [("vector", vec, "vector"), ("keyword (BM25)", kw, "keyword"), ("hybrid (RRF)", fused, "rrf")]:
        print(f"  -- {name}")
        if not hits:
            print("       (nothing)")
        for h in hits[:4]:
            print(f"     {h.scores[key]:>7.4f}  {h.label}")


if __name__ == "__main__":
    args, groups = sys.argv[1:], ["all"]
    if "--groups" in args:
        i = args.index("--groups")
        groups = args[i + 1].split(",")
        del args[i:i + 2]

    if args:
        show(" ".join(args), groups)
    else:
        show("how do I get on the company network from home?")      # paraphrase
        show("KST-PRN-2F")                                          # exact code
        show("what is an ICR?")                                     # acronym
        show("when are salary reviews?", ["all"])                   # acl: new joiner
        show("when are salary reviews?", ["all", "managers"])       # acl: manager
        show("Si mund të kërkoj pushim?")                           # albanian