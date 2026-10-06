"""Step 7b: HANDLE. The single entry point every front end calls (terminal, FastAPI, Slack):
   who's asking -> router -> (HR | Buddy | greeting | answer) -> log -> reply

    python -m app.handler "how do I book leave?"
    python -m app.handler --user manager "what should a manager do before a new joiner's first day?"
    python -m app.handler --stats        # answer rate, languages, speed
    python -m app.handler --gaps         # open escalations = missing docs
"""
import logging
import sys
import time
from dataclasses import dataclass, field

import yaml

from app.answer import answer
from app.config import USERS_FILE
from app.router import route
from app.store import log_escalation, log_question, open_gaps, stats


@dataclass
class Reply:
    outcome: str                  # answered | escalated | buddy | smalltalk
    text: str                     # what the user sees
    route: str = ""
    reason: str = ""
    language: str = ""
    sources: list = field(default_factory=list)   # [{"n": 1, "label": ..., "url": ...}]
    restricted: bool = False      # front ends must never forward the question text when True


def load_user(user_key):
    with open(USERS_FILE, encoding="utf-8") as f:
        users = yaml.safe_load(f) or {}
    return users.get(user_key) or users["default"]


def handle(question, user_key="default"):
    t0 = time.time()
    user = load_user(user_key)
    r = route(question)
    route_label = f"{r.category} ({r.how})"

    def done(reply, top_score=None):
        stored_q = f"[{r.topic} question, text withheld]" if reply.restricted else question
        stored_en = "" if reply.restricted else r.english
        log_question(user_key, stored_q, stored_en, r.language, route_label, reply.outcome, reply.reason,
                     top_score, int((time.time() - t0) * 1000))
        return reply

    if r.category == "restricted":
        log_escalation(user_key, f"[{r.topic} question, text withheld]", "restricted_topic")
        return done(Reply("escalated", (
            f"That one needs a person rather than me. {user['hr_contact']} in HR handles {r.topic} questions "
            f"and will treat it confidentially. I've let them know you'd like a word."),
            route_label, "restricted_topic", r.language, restricted=True))

    if r.category == "smalltalk":
        return done(Reply("smalltalk", (
            f"Hi! I'm Onboard Buddy. Ask me about tools, policies, acronyms or who to talk to, in English, "
            f"Albanian or Macedonian. For the human side of things, {user['buddy']} is your Buddy."),
            route_label, language=r.language))

    if r.category == "buddy":
        return done(Reply("buddy", (
            f"Good question, and a really common one. That's exactly what your Buddy is for: {user['buddy']} "
            f"knows the unwritten rules better than any document. Drop them a message."),
            route_label, language=r.language))

    result = answer(question, r.english, user["groups"])
    if result.status == "escalated":
        log_escalation(user_key, question, result.reason, result.draft,
                       [{"title": p.label, "score": p.scores.get("rerank")} for p in result.passages])
        return done(Reply("escalated", (
            "I don't have a reliable answer to that in the onboarding docs, so I won't guess. "
            "I've passed your question to the onboarding team and someone will get back to you."),
            route_label, result.reason, r.language), result.top_score)

    sources = [{"n": n, "label": p.label, "url": p.url} for n, p in result.sources()]
    return done(Reply("answered", result.text, route_label, language=r.language, sources=sources), result.top_score)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    args = sys.argv[1:]

    if "--stats" in args:
        outcomes, langs, avg = stats()
        total = sum(outcomes.values()) or 1
        print(f"{total} questions: " + ", ".join(f"{k} {v} ({100 * v // total}%)" for k, v in outcomes.items()))
        print("languages: " + ", ".join(f"{l} {n}" for l, n in langs))
        print(f"average time for an answered question: {avg / 1000:.1f}s" if avg else "")
        sys.exit()
    if "--gaps" in args:
        for ts, reason, q in open_gaps():
            print(f"{ts[:16]}  {reason[:38]:<38}  {q}")
        sys.exit()

    user_key = "default"
    if "--user" in args:
        i = args.index("--user")
        user_key = args[i + 1]
        del args[i:i + 2]

    reply = handle(" ".join(args) or "hi!", user_key)
    print(f"[{reply.outcome} | route: {reply.route} | {reply.language}{' | ' + reply.reason if reply.reason else ''}]\n")
    print(reply.text)
    for s in reply.sources:
        print(f"  [{s['n']}] {s['label']}  {s['url']}")