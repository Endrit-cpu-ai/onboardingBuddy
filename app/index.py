"""Embed chunks and store them in Chroma. Skips unchanged docs, replaces changed ones, removes deleted ones.

    python -m app.index           # only what changed
    python -m app.index --force   # re-embed everything
"""
import sys
import time

from app.embeddings import embed_texts
from app.ingest import chunk_doc, load_docs
from app.vectordb import acl_flags, collection


def index(force=False):
    col = collection()
    docs = load_docs()
    stats = {"indexed": 0, "unchanged": 0, "removed": 0, "chunks": 0}

    # path -> hash of what's indexed now
    stored = {m["path"]: m["doc_hash"] for m in col.get(include=["metadatas"])["metadatas"]}

    for doc in docs:
        if stored.get(doc.path) == doc.hash and not force:
            stats["unchanged"] += 1
            continue

        chunks, quarantined = chunk_doc(doc)
        for q in quarantined:
            print(f"  ! quarantined {q}")

        col.delete(where={"path": doc.path})
        if chunks:
            col.add(
                ids=[c.id for c in chunks],
                embeddings=embed_texts(c.embed_text for c in chunks),
                documents=[c.text for c in chunks],
                metadatas=[{
                    "path": c.path, "title": c.title, "url": c.url, "heading": c.heading,
                    "doc_hash": c.doc_hash, "acl": ",".join(c.acl), **acl_flags(c.acl),
                } for c in chunks],
            )
        stats["indexed"] += 1
        stats["chunks"] += len(chunks)
        print(f"  + {doc.path}: {len(chunks)} chunks")

    for path in set(stored) - {d.path for d in docs}:     # docs that were deleted
        col.delete(where={"path": path})
        stats["removed"] += 1
        print(f"  - {path}: removed")

    return stats


if __name__ == "__main__":
    t = time.time()
    stats = index(force="--force" in sys.argv)
    print(f"\n{stats}   ({time.time() - t:.1f}s)")

    col = collection()
    managers = col.get(where={"acl_managers": True})["ids"]
    print(f"\nIn Chroma: {col.count()} chunks, {len(managers)} of them managers-only")

    s = col.get(ids=["it-setup-vpn.md#2"], include=["metadatas", "documents", "embeddings"])
    print(f"\nOne stored chunk ({s['ids'][0]}):")
    print(f"  text:     {s['documents'][0][:70]}...")
    print(f"  metadata: {s['metadatas'][0]}")
    print(f"  vector:   {len(s['embeddings'][0])} numbers, starting {[round(float(x), 3) for x in s['embeddings'][0][:4]]}")