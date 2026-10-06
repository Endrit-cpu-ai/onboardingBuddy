"""Step 8: FASTAPI. The same handle() as the terminal, now as a web API that anything can call
(the Slack bot, a web page, Teams, a test script).

    uvicorn app.api:app --reload --port 8000
    then open http://localhost:8000/docs  (interactive API page)
"""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import (API_KEY, EMBED_MODEL, GEN_MODEL, RERANK_MODEL, ROUTER_MODEL, SLACK_BOT_TOKEN,
                        SLACK_SIGNING_SECRET)
from app.embeddings import embed_query
from app.handler import handle
from app.store import open_gaps, stats
from app.vectordb import collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app):
    # Load the embedding model and open the database ONCE at startup, so the first question isn't slow
    embed_query("warm up")
    logging.getLogger("api").info("ready: %d chunks indexed", collection().count())
    yield


app = FastAPI(title="Onboard Buddy API", version="2.0", lifespan=lifespan)


def check_key(x_api_key: str = Header(default="")):
    """If API_KEY is set in .env, every call must send it in the X-API-Key header."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="missing or wrong X-API-Key")


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["How do I connect to the VPN?"])
    user_key: str = Field(default="default", examples=["default", "manager"])


class Source(BaseModel):
    n: int
    label: str
    url: str


class AskOut(BaseModel):
    outcome: str          # answered | escalated | buddy | smalltalk
    text: str
    route: str
    reason: str
    language: str
    sources: list[Source]


# Plain `def` (not `async def`): handle() blocks while it waits for the AI,
# so FastAPI runs these in a thread pool and keeps serving other requests meanwhile.

@app.post("/ask", response_model=AskOut, dependencies=[Depends(check_key)])
def ask(body: AskIn):
    r = handle(body.question, body.user_key)
    return AskOut(outcome=r.outcome, text=r.text, route=r.route, reason=r.reason,
                  language=r.language, sources=r.sources)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": collection().count(), "embed_model": EMBED_MODEL,
            "router": ROUTER_MODEL, "reranker": RERANK_MODEL, "answer_model": GEN_MODEL}


@app.get("/stats", dependencies=[Depends(check_key)])
def get_stats():
    outcomes, langs, avg_ms = stats()
    total = sum(outcomes.values())
    return {"total": total, "outcomes": outcomes,
            "answer_rate": round(outcomes.get("answered", 0) / total, 2) if total else None,
            "languages": dict(langs), "avg_answer_seconds": round(avg_ms / 1000, 1) if avg_ms else None}


@app.get("/gaps", dependencies=[Depends(check_key)])
def get_gaps(limit: int = 30):
    return [{"ts": ts, "reason": reason, "question": q} for ts, reason, q in open_gaps(limit)]



# ---------- Slack over HTTP (production mode) ----------
# Only switched on when both Slack secrets are set. Slack then sends events to https://<your-server>/slack/events
if SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET:
    from slack_bolt.adapter.fastapi import SlackRequestHandler

    from app.slack_bot import bolt_app

    slack_handler = SlackRequestHandler(bolt_app)

    @app.post("/slack/events")
    async def slack_events(req: Request):
        return await slack_handler.handle(req)