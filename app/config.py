"""App settings. Secrets come from .env, everything else lives here.

    python -m app.config    # print settings, check for missing secrets
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# paths
KB_DIR = ROOT / "knowledge_base"
USERS_FILE = ROOT / "users.yaml"
GOLDEN_SET = ROOT / "golden_set.yaml"
DATA_DIR = Path(os.getenv("DATA_DIR", Path.home() / ".onboard-v2"))   # keep out of OneDrive, it locks db files
CHROMA_DIR = DATA_DIR / "chroma"
SQLITE_PATH = DATA_DIR / "onboard.db"
MODEL_CACHE = DATA_DIR / "models"

# secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")            # socket mode
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")  # http mode
ESCALATION_CHANNEL = os.getenv("ESCALATION_CHANNEL", "")      # channel id (C...), not the name
OFFICE_USERGROUP = os.getenv("OFFICE_USERGROUP", "")          # optional, limits the bot to one user group
API_KEY = os.getenv("API_KEY", "")                            # optional, X-API-Key for the API

# chunking (re-index with --force after changing)
CHUNK_WORDS = 350
CHUNK_OVERLAP = 50

# embeddings (a new model gets its own collection, just run app.index)
EMBED_MODEL = "google/embeddinggemma-300m"     # multilingual, 768 dims
# prompt format gemma expects, fastembed doesn't add it
EMBED_QUERY_TEMPLATE = "task: search result | query: {text}"
EMBED_DOC_TEMPLATE = "title: none | text: {text}"     # title is already in the chunk text

# retrieval
SEARCH_K = 10          # candidates per search, also how many get reranked
RRF_K = 60
RERANK_TOP_N = 4       # chunks passed to the answer model

# models
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")    # openai | anthropic
if LLM_PROVIDER == "openai":
    ROUTER_MODEL = "gpt-6-luna"
    RERANK_MODEL = "gpt-6-luna"
    GEN_MODEL = "gpt-6-luna"
    FAST_EFFORT = "none"   # router + rerank, no reasoning needed
    GEN_EFFORT = "none"    # same 41/41 as sol, faster and ~20x cheaper
else:
    ROUTER_MODEL = "claude-haiku-4-5-20251001"
    RERANK_MODEL = "claude-haiku-4-5-20251001"
    GEN_MODEL = "claude-sonnet-5"
    FAST_EFFORT = None     # haiku has no effort setting
    GEN_EFFORT = "low"     # scored the same as medium on the eval, faster
LLM_TIMEOUT = 30       # seconds
LLM_RETRIES = 3

SCORE_THRESHOLD = 7    # escalate if the best rerank score (0-10) is below this

DATA_DIR.mkdir(parents=True, exist_ok=True)


def _mask(value):
    return f"set ({value[:6]}…)" if value else "MISSING"


if __name__ == "__main__":
    print(f"Project folder:  {ROOT}")
    print(f"Docs folder:     {KB_DIR}  ({len(list(KB_DIR.glob('*.md')))} .md files)")
    print(f"Data folder:     {DATA_DIR}")
    print(f"LLM provider:    {LLM_PROVIDER}")
    print(f"OpenAI key:      {_mask(OPENAI_API_KEY)}")
    print(f"Anthropic key:   {_mask(ANTHROPIC_API_KEY)}")
    print(f"Slack bot token: {_mask(SLACK_BOT_TOKEN)}")
    print(f"Slack app token: {_mask(SLACK_APP_TOKEN)}")
    print(f"Models:          router={ROUTER_MODEL}  rerank={RERANK_MODEL}  answer={GEN_MODEL}")