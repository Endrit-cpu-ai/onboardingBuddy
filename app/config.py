"""All settings in one place.
Secrets come from .env (never committed). Everything else has a default here, safe to commit.

    python -m app.config      # print the current settings and check what's missing
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- Paths ---
KB_DIR = ROOT / "knowledge_base"
USERS_FILE = ROOT / "users.yaml"
GOLDEN_SET = ROOT / "golden_set.yaml"
DATA_DIR = Path(os.getenv("DATA_DIR", Path.home() / ".onboard-v2"))   # outside OneDrive: it locks database files
CHROMA_DIR = DATA_DIR / "chroma"
SQLITE_PATH = DATA_DIR / "onboard.db"
MODEL_CACHE = DATA_DIR / "models"

# --- Secrets (from .env) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")            # Socket Mode (local dev)
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")  # HTTP mode (FastAPI)
ESCALATION_CHANNEL = os.getenv("ESCALATION_CHANNEL", "")      # channel ID, e.g. C0123...
OFFICE_USERGROUP = os.getenv("OFFICE_USERGROUP", "")          # optional: only answer this Slack user group
API_KEY = os.getenv("API_KEY", "")                            # optional: required as X-API-Key header on the API

# --- Chunking (after changing: python -m app.index --force) ---
CHUNK_WORDS = 350
CHUNK_OVERLAP = 50

# --- Embeddings ---
# Changing the model creates a fresh Chroma collection automatically (see vectordb.py); then run python -m app.index
EMBED_MODEL = "google/embeddinggemma-300m"     # multilingual (Albanian, Macedonian, English...), 768 dims, ~1.2 GB
# The exact input format this model was trained with. {text} is replaced by the question / chunk.
EMBED_QUERY_TEMPLATE = "task: search result | query: {text}"
EMBED_DOC_TEMPLATE = "title: none | text: {text}"     # our chunk text already starts with "Title > Heading"

# --- Retrieval ---
SEARCH_K = 10          # candidates from vector search and from keyword search
RRF_K = 60             # Reciprocal Rank Fusion constant
RERANK_TOP_N = 4       # chunks the answer model sees

# --- Models ---
ROUTER_MODEL = "claude-haiku-4-5-20251001"
RERANK_MODEL = "claude-haiku-4-5-20251001"
GEN_MODEL = "claude-sonnet-5"
GEN_EFFORT = "low"  # how hard Sonnet thinks: low / medium / high. Lower = faster answers
LLM_TIMEOUT = 30       # seconds per call
LLM_RETRIES = 3

# --- Gates ---
SCORE_THRESHOLD = 7    # rerank score (0-10) below this = escalate without generating

DATA_DIR.mkdir(parents=True, exist_ok=True)


def _mask(value):
    return f"set ({value[:6]}…)" if value else "MISSING"


if __name__ == "__main__":
    print(f"Project folder:  {ROOT}")
    print(f"Docs folder:     {KB_DIR}  ({len(list(KB_DIR.glob('*.md')))} .md files)")
    print(f"Data folder:     {DATA_DIR}")
    print(f"Anthropic key:   {_mask(ANTHROPIC_API_KEY)}")
    print(f"Slack bot token: {_mask(SLACK_BOT_TOKEN)}")
    print(f"Slack app token: {_mask(SLACK_APP_TOKEN)}")
    print(f"Models:          router={ROUTER_MODEL}  rerank={RERANK_MODEL}  answer={GEN_MODEL}")