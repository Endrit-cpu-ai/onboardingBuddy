"""Everything about the Chroma database in one place: open it, and build the access filter."""
from functools import lru_cache
import threading

import chromadb

from app.config import CHROMA_DIR, EMBED_MODEL

# One collection per embedding model: vectors from different models can't be mixed,
# so switching models starts a fresh collection instead of crashing on a dimension mismatch.
COLLECTION = "onboarding-" + EMBED_MODEL.replace("/", "-")

_lock = threading.Lock()

@lru_cache(maxsize=1)
def collection():
    with _lock:                 # only one thread may create the Chroma client
        return _collection()


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

def acl_flags(acl):
    """Chroma can't filter on lists, so each group becomes its own flag: acl_all=True, acl_managers=True."""
    return {f"acl_{group}": True for group in acl}


def acl_filter(groups):
    """Chroma 'where' filter: the chunk must carry at least one of the user's groups."""
    conditions = [{f"acl_{g}": True} for g in groups]
    return conditions[0] if len(conditions) == 1 else {"$or": conditions}