"""Turn text into vectors with a model that runs locally on CPU.
The SAME model must embed both the chunks and the questions, or the numbers aren't comparable.

Most embedding models expect questions and documents in a specific format (a short instruction in front).
fastembed does NOT add it for you, so we apply the templates from config here.
"""
from functools import lru_cache

from fastembed import TextEmbedding

from app.config import EMBED_DOC_TEMPLATE, EMBED_MODEL, EMBED_QUERY_TEMPLATE, MODEL_CACHE


@lru_cache(maxsize=1)
def _model():
    return TextEmbedding(EMBED_MODEL, cache_dir=str(MODEL_CACHE))   # first call downloads the model


def embed_texts(texts):
    """For chunks."""
    inputs = [EMBED_DOC_TEMPLATE.replace("{text}", t) for t in texts]
    return [v.tolist() for v in _model().embed(inputs)]


def embed_query(question):
    """For questions."""
    return next(iter(_model().embed([EMBED_QUERY_TEMPLATE.replace("{text}", question)]))).tolist()


if __name__ == "__main__":
    import numpy as np

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    docs = [
        "Leave and Time Off Policy > How to book leave\nBook all leave in Kestrel People. Your line manager approves the request.",
        "IT Setup > Connecting to the VPN\nThe VPN portal address is vpn.kestrel.example.",
        "Office Logistics > Lunch and kitchen\nFruit and coffee are free.",
    ]
    vecs = embed_texts(docs)
    print(f"{EMBED_MODEL}: {len(vecs[0])} dimensions")
    for q in ["How do I request time off?",          # English
              "Si mund të kërkoj pushim?",           # Albanian
              "Како да побарам слободни денови?"]:   # Macedonian
        qv = embed_query(q)
        print(f"\n{q}")
        for d, v in sorted(zip(docs, vecs), key=lambda x: -cos(qv, x[1])):
            print(f"  {cos(qv, v):.3f}  {d.splitlines()[0]}")