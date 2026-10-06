"""SQLite log. `questions` has every question and its outcome, `escalations` is what a human needs to pick up.
Restricted questions are stored as "[topic question, text withheld]"."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from app.config import SQLITE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id         INTEGER PRIMARY KEY,
    ts         TEXT NOT NULL,
    user_key   TEXT,
    question   TEXT,
    english    TEXT,
    language   TEXT,
    route      TEXT,
    outcome    TEXT,
    reason     TEXT,
    top_score  REAL,
    latency_ms INTEGER
);
CREATE TABLE IF NOT EXISTS escalations (
    id         INTEGER PRIMARY KEY,
    ts         TEXT NOT NULL,
    user_key   TEXT,
    question   TEXT,
    reason     TEXT,
    draft      TEXT,
    sources    TEXT,      -- json
    resolved   INTEGER DEFAULT 0
);
"""


def _conn():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.executescript(SCHEMA)
    return conn


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_question(user_key, question, english, language, route, outcome, reason="", top_score=None, latency_ms=None):
    with closing(_conn()) as c, c:
        c.execute("INSERT INTO questions (ts, user_key, question, english, language, route, outcome, reason, top_score, latency_ms) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (_now(), user_key, question, english, language, route, outcome, reason, top_score, latency_ms))


def log_escalation(user_key, question, reason, draft="", sources=()):
    with closing(_conn()) as c, c:
        c.execute("INSERT INTO escalations (ts, user_key, question, reason, draft, sources) VALUES (?,?,?,?,?,?)",
                  (_now(), user_key, question, reason, draft, json.dumps(list(sources))))


def open_gaps(limit=30):
    with closing(_conn()) as c:
        return c.execute("SELECT ts, reason, question FROM escalations WHERE resolved = 0 ORDER BY id DESC LIMIT ?",
                         (limit,)).fetchall()


def stats():
    with closing(_conn()) as c:
        rows = dict(c.execute("SELECT outcome, COUNT(*) FROM questions GROUP BY outcome").fetchall())
        langs = c.execute("SELECT language, COUNT(*) FROM questions GROUP BY language ORDER BY 2 DESC").fetchall()
        avg = c.execute("SELECT AVG(latency_ms) FROM questions WHERE outcome = 'answered'").fetchone()[0]
    return rows, langs, avg


def delete_user(user_key):
    """Remove everything stored for a user (leavers, test users)."""
    with closing(_conn()) as c, c:
        c.execute("DELETE FROM questions WHERE user_key = ?", (user_key,))
        c.execute("DELETE FROM escalations WHERE user_key = ?", (user_key,))