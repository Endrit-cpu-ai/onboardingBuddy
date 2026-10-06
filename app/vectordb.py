"""Chroma collection + ACL helpers."""
from functools import lru_cache
import threading

import chromadb

from app.config import CHROMA_DIR, EMBED_MODEL

# one collection per embedding model, so switching models doesn't hit a dimension mismatch
COLLECTION = "onboarding-" + EMBED_MODEL.replace("/", "-")

_lock = threading.Lock()

@lru_cache(maxsize=1)
def collection():
    with _lock:                 # chroma fails if two threads create the client at once
        return _collection()


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

def acl_flags(acl):
    # chroma can't filter on lists, so each group becomes a flag: acl_all=True, acl_managers=True
    return {f"acl_{group}": True for group in acl}


def acl_filter(groups):
    """Match chunks that have at least one of the user's groups."""
    conditions = [{f"acl_{g}": True} for g in groups]
    return conditions[0] if len(conditions) == 1 else {"$or": conditions}