# Onboard Buddy V2

A Slack bot that answers new-joiner questions from company docs in **English, Albanian and Macedonian**, cites its sources, and hands anything sensitive or unknown to a human.

It runs on a fictional company (**Kestrel Digital**, 13 fake onboarding docs). V2 is a rebuild of V1 (`slackOnboarder`) on **Chroma + FastAPI**, with the lessons from V1 built in.

> **Note:** everything in `knowledge_base/` is made up. Two docs contain things planted on purpose to test the safety layers: fake credentials in `it-setup-vpn.md` (AWS's public example key and a printer password, both redacted at ingest) and a prompt-injection section in `team-wiki-misc.md` (quarantined at ingest). None of them are real.

**Status:** 41/41 on the golden set, two runs per case, 0 flaky. Runs on OpenAI `gpt-6-luna` for every LLM step, ~4.8s per answer. Anthropic (Claude) still works by changing one setting.

---

## How a question flows

```
Slack DM / POST /ask / terminal
        │
        ▼
handler.handle(question, user)           who's asking -> groups, Buddy, HR contact (users.yaml)
        │
        ▼
ROUTER   regex rules + 1 small-LLM call   category (kb / buddy / restricted / smalltalk),
        │                                 language, English translation
        ├── restricted ──► HR reply, question text never stored or posted
        ├── buddy ───────► "ask your Buddy"
        ├── smalltalk ───► greeting
        ▼ kb
SEARCH   on the ENGLISH version           Chroma vector search + BM25 keyword search,
        │                                 both ACL-filtered, merged with RRF
        ▼
RERANK   1 small-LLM call                 scores the top 10 candidates 0-10, keeps the best 4
        │
        ├── best score < 7 ──► escalate (Gate 2: don't even try)
        ▼
ANSWER   1 answer-model call              answers ONLY from the 4 sources, cites [n],
        │                                 replies in the ORIGINAL language
        ├── no / bad citations ──► escalate (Gate 3: grounding check)
        ▼
reply + sources + "Get a human" button    every question logged to SQLite
```

Ingest runs offline: `knowledge_base/*.md` → check metadata → redact secrets → split by `##` heading → quarantine prompt-injection sections → embed → Chroma.

---

## Stack

| Part | Choice | Runs |
|---|---|---|
| Vector index | Chroma (persistent, cosine) | Local folder |
| Keyword search | BM25 (`rank_bm25`), built per query over what the user may see | In process |
| Embeddings | `google/embeddinggemma-300m` via fastembed, 768 dims, multilingual | Laptop CPU |
| Router + reranker | OpenAI `gpt-6-luna`, effort `none` | OpenAI API |
| Answer model | OpenAI `gpt-6-luna`, effort `none` | OpenAI API |
| Log | SQLite (`questions`, `escalations`) | Local file |
| API | FastAPI + uvicorn | Laptop / server |
| Chat | Slack Bolt: Socket Mode (dev) or HTTP `/slack/events` (prod) | Laptop / server |
| Language | Python 3.12 | |

### LLM provider

Set with `LLM_PROVIDER` in `.env`. Only `app/llm.py` and the model block in `app/config.py` know which vendor is used.

| | `openai` (default) | `anthropic` |
|---|---|---|
| Router + reranker | `gpt-6-luna`, effort `none` | `claude-haiku-4-5-20251001` |
| Answers | `gpt-6-luna`, effort `none` | `claude-sonnet-5`, effort `low` |
| Key | `OPENAI_API_KEY` | `ANTHROPIC_API_KEY` |
| Eval | 41/41, 4.8s avg | 41/41, 4.4s avg |

Luna costs $0.10 / $0.50 per 1M input/output tokens, so a question is a fraction of a cent. Embeddings stay local either way, so switching provider doesn't need a re-index.

---

## What changed from V1

| V1 | V2 | Why |
|---|---|---|
| Postgres + pgvector on Neon | **Chroma** in a local folder | No DB server, no connection-string fights. One collection per embedding model, so switching models can't mix vector sizes |
| Postgres full-text search | **BM25** | Real keyword ranking; tokenizer handles Albanian/Macedonian (`\w+`, min length 2) |
| `bge-small-en` (English only) | **embeddinggemma-300m** + router **translates to English** | Gemma alone was weak cross-language; translating first made Albanian/Macedonian questions retrieve as well as English ones |
| Flat scripts (`ask.py`, `pipeline.py`, `bot.py`) | **`app/` package**, one `handler.handle()` used by terminal, API and Slack | One code path = one behaviour everywhere |
| Settings spread across files | **`app/config.py`** + secrets in `.env` | One place to tune |
| Raw API calls | **`app/llm.py`**: timeouts, SDK retries, safe JSON parsing | A bad JSON or network blip escalates instead of crashing |
| Anthropic only | **OpenAI or Anthropic**, one setting (`LLM_PROVIDER`) | The org provided an OpenAI key; the eval decides which models to use |
| Escalation table only | **Every question logged** (outcome, language, score, latency) | Answer rate, language mix and speed via `--stats` / `/stats` |
| No API | **FastAPI** `/ask`, `/health`, `/stats`, `/gaps` | Anything can call the bot (Teams, a web page, tests) |
| Socket Mode only | Socket Mode **or** HTTP mode on the same FastAPI app | HTTP is what a company server needs |
| Eval: 1 run per case | **2+ runs per case, flags FLAKY**, checks cited source, language, forbidden words; parallel | Catches router decisions that flip between runs |
| "Get a human" leaked restricted questions | Button only on answered / Buddy replies; restricted questions logged as `[topic question, text withheld]` | Privacy |
| Data next to the code (OneDrive) | Data in `~/.onboard-v2` | OneDrive locks database files |

---

## Project layout

```
slackOnboardBuddyV2/
├── .env                  secrets only (never commit)
├── requirements.txt
├── users.yaml            Slack member ID -> groups, Buddy, HR contact ("default" for everyone else)
├── golden_set.yaml       41 eval cases
├── knowledge_base/       the 13 docs (markdown + frontmatter: title, url, acl, owner)
└── app/
    ├── config.py         ALL settings + paths          python -m app.config
    │   INGEST (offline)
    ├── ingest.py         load, validate, redact secrets, chunk, quarantine injection
    ├── embeddings.py     text -> vectors (with the model's query/doc templates)
    ├── vectordb.py       Chroma collection (thread-safe), ACL flags + filter
    ├── index.py          incremental upsert: skip unchanged, replace changed, remove deleted
    │   QUERY (per question)
    ├── search.py         vector + BM25 + RRF
    ├── llm.py            every LLM call (OpenAI or Anthropic): timeouts, retries, JSON parsing
    ├── rerank.py         small LLM scores candidates 0-10
    ├── router.py         rules + small LLM: category, language, English translation
    ├── answer.py         score gate -> answer model -> grounding check
    ├── handler.py        THE entry point: route -> answer/escalate -> log -> Reply
    ├── store.py          SQLite log, stats, gaps, delete_user
    │   FRONT ENDS
    ├── api.py            FastAPI (+ Slack HTTP mode)
    ├── slack_bot.py      Slack DM bot (Socket Mode)
    │   QUALITY
    └── eval.py           golden set runner
```

Data (outside OneDrive): `C:\Users\<you>\.onboard-v2\` → `chroma\`, `onboard.db`, `models\`.

---

## Setup (Windows, PowerShell)

### 1. Python environment

```powershell
py -3.12 -m venv $HOME\.venvs\onboard-v2
& $HOME\.venvs\onboard-v2\Scripts\Activate.ps1        # prompt shows (onboard-v2)
pip install -r requirements.txt
```

`requirements.txt`:

```
openai>=3.0
anthropic>=0.70
chromadb>=1.5
fastembed>=0.7
rank_bm25>=0.2
fastapi>=0.115
uvicorn>=0.30
slack_bolt>=1.21
python-dotenv>=1.0
pyyaml>=6.0
```

pip installs many extra packages (Chroma pulls in a lot). That's normal.

### 2. Secrets: `.env`

Copy `.env.example` to `.env` and fill in:

```
LLM_PROVIDER=openai                 # or anthropic
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=                  # only for anthropic
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...            # Socket Mode
SLACK_SIGNING_SECRET=...            # only for HTTP mode
ESCALATION_CHANNEL=C0123456789      # channel ID of #onboarding-help, not the name
OFFICE_USERGROUP=                   # optional: Slack user group ID, only those members get answers (paid plans)
API_KEY=                            # optional: if set, API calls need header X-API-Key
```

Check it: `python -m app.config` shows every setting and flags anything `MISSING`.

### 3. Build the index

```powershell
python -m app.ingest      # report: 13 docs -> 50 chunks, 2 secrets redacted, 1 section quarantined
python -m app.index       # first run downloads embeddinggemma (~1.2 GB) into ~/.onboard-v2/models
```

### 4. Try it

```powershell
python -m app.handler "how do I connect to the VPN?"
python -m app.handler "Si mund të kërkoj pushim?"
python -m app.eval
```

### 5. Slack

Reuse the V1 Slack app (same tokens; stop the V1 bot first) or create one from this manifest:

```yaml
display_information:
  name: Onboard Buddy
features:
  app_home:
    messages_tab_enabled: true
    messages_tab_read_only_enabled: false
  bot_user:
    display_name: Onboard Buddy
    always_online: true
oauth_config:
  scopes:
    bot: [chat:write, chat:write.public, im:history, usergroups:read]
settings:
  event_subscriptions:
    bot_events: [message.im]
  interactivity:
    is_enabled: true
  socket_mode_enabled: true
```

```powershell
python -m app.slack_bot          # then DM "Onboard Buddy" from the Apps sidebar
```

**HTTP mode (server):** turn Socket Mode off in the Slack app, set the Event Subscriptions and Interactivity request URL to `https://<server>/slack/events`, set `SLACK_SIGNING_SECRET`, and run the API. It serves Slack too.

---

## Everyday commands

| Command | What it does |
|---|---|
| `python -m app.config` | Show settings, check secrets |
| `python -m app.ingest` | Docs → chunks report (secrets, quarantined sections) |
| `python -m app.index` | Re-index changed docs only |
| `python -m app.index --force` | Re-embed everything |
| `python -m app.embeddings` | Cross-language similarity test (EN / SQ / MK) |
| `python -m app.search "q" [--groups all,managers]` | Vector vs BM25 vs fused results |
| `python -m app.rerank "q" [--groups ...]` | Before/after reranking with 0-10 scores |
| `python -m app.answer [--groups ...] ["q"]` | Router + answer, demo questions if no question |
| `python -m app.handler "q" [--user manager]` | Full pipeline as a user, logged |
| `python -m app.handler --stats` | Answer rate, languages, average answer time |
| `python -m app.handler --gaps` | Open escalations = what the docs are missing |
| `uvicorn app.api:app --reload --port 8000` | API; interactive docs at http://localhost:8000/docs |
| `python -m app.slack_bot` | Slack bot, Socket Mode |
| `python -m app.eval` | Golden set, 2 runs per case |
| `python -m app.eval --only "vpn" --runs 5 -v` | Re-test a subset harder, print answers |

### API

| Endpoint | |
|---|---|
| `POST /ask` | `{"question": "...", "user_key": "default"}` → outcome, text, route, reason, language, sources |
| `GET /health` | Chunk count + models |
| `GET /stats` | Answer rate, languages, avg answer seconds |
| `GET /gaps` | Open escalations |
| `POST /slack/events` | Slack HTTP mode (only when Slack secrets are set) |

---

## Changing things safely

**Every change: change one thing → `python -m app.eval` → compare.** Keep it only if the score holds.

| You changed | Then run |
|---|---|
| A doc in `knowledge_base/` | `python -m app.index` |
| `CHUNK_WORDS` / `CHUNK_OVERLAP` | `python -m app.index --force` |
| `EMBED_MODEL` (+ its templates) | `python -m app.index` (a new collection is created automatically) |
| Thresholds, `SEARCH_K`, `RERANK_TOP_N`, models, effort, prompts | Nothing, just the eval |
| `users.yaml` | Nothing, re-read on every question |

**Adding a doc:** `knowledge_base/<name>.md` with frontmatter (`title`, `url`, `acl: [all]` or `[managers, hr]`, `owner`). Use `##` headings, since each section becomes a chunk. Missing frontmatter fails ingest with the file name.

**Manager access:** add the person's Slack member ID to `users.yaml` with `groups: [all, managers]`.

**Leaver / data deletion:** `store.delete_user("<slack id>")` removes their questions and escalations.

---

## Current settings and why

| Setting | Value | Evidence |
|---|---|---|
| `SEARCH_K` | 10 | 41/41, 4.4s avg / 6.6s slowest (vs 5.1s / 13.7s at 20). **Re-test at 10 vs 20 on the real corpus:** with 50 chunks, 10 is already 20% of everything |
| `RERANK_TOP_N` | 4 | Enough context, short prompt |
| `SCORE_THRESHOLD` | 7 | Answerable questions score 9-10; unanswerable ≤5 |
| Models (OpenAI) | `gpt-6-luna` everywhere | Luna for answers scored the same 41/41 as `gpt-6.1-sol`, faster (4.8s vs 7.7s avg) and ~20x cheaper |
| `GEN_EFFORT` | OpenAI `none`, Claude `low` | Answers are 1-4 sentences from 4 sources, no reasoning needed. Claude: `low` scored the same as `medium` |
| `CHUNK_WORDS` / overlap | 350 / 50 | Docs are short; most sections are one chunk |

### Eval history

| Run | Score | Change |
|---|---|---|
| 1 | 39/41 | Baseline. VPN crashed (Chroma opened by 2 threads at once); Albanian "Sa ditë pushim vjetor kam?" flaky (routed restricted once) |
| 2 | 40/41 | Lock around Chroma client; router prompt: "how a policy applies to you" is kb. Printer case was a bad test |
| 3 | 41/41 | Printer case now checks "password never leaks" instead of a fixed outcome |
| 4 | 41/41 | `GEN_EFFORT = "low"` |
| 5 | 41/41 | `SEARCH_K = 10` → 4.4s avg (Claude) |
| 6 | 40/41 | Switched to OpenAI: Luna router/rerank, Sol answers. "I'm sick today, what do I do?" flaky (routed as health once) |
| 7 | 41/41 | Router prompt: reporting a sick day is kb, "health" means diagnosis/condition/treatment. 7.7s avg, 38s slowest (one call hit the 30s timeout and retried) |
| 8 | 41/41 | Luna for answers too, effort `none` → 4.8s avg, 7.9s slowest |

Golden set: 15 English answerable, 4 Albanian, 4 Macedonian, 1 poisoned doc, 8 restricted (incl. SQ/MK), 3 not in docs, 2 access control, 2 Buddy, 2 smalltalk.

---

## Known gaps

- Canned replies (HR, Buddy, greeting, escalation) are English only.
- No conversation memory: follow-ups like "and for sick days?" lose context.
- Escalations can't be answered or resolved from Slack yet.
- `OFFICE_USERGROUP` filtering needs a paid Slack plan (Business+ has it), so it's untested on the free workspace.
- Embedding model not yet compared on the eval (gemma vs `BAAI/bge-base-en-v1.5`).
- Everything runs on one laptop; no container, no secret manager.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No module named fastembed` (or any package) | venv not active | `& $HOME\.venvs\onboard-v2\Scripts\Activate.ps1` |
| `metadata is missing ['url']` | Doc frontmatter incomplete | Add the field named in the error |
| `unexpected keyword argument 'temperature'` | Newer models don't take `temperature` | Removed; use `effort` (`GEN_EFFORT`) |
| `must contain the word 'json'` (OpenAI 400) | OpenAI JSON mode needs "json" in the input, not just the instructions | `llm.py` appends "Reply with JSON." in JSON mode |
| Fix "disappears" after it was saved to a file | The file was open in the editor and auto-save wrote the old version back | Close the tab without saving, reopen, then edit |
| `Could not connect to tenant default_tenant` | Two threads created the Chroma client at once | Lock in `vectordb.collection()` |
| `APIConnectionError: Connection error.` in a block of cases | Network dropped (Wi-Fi/VPN); SDK retried 3× | Re-run; try `--workers 2` |
| Albanian/Macedonian question answered badly | Retrieval runs on the router's English translation | `python -m app.handler "q"` shows the route and translation |
| Restricted reply for a general policy question | Router boundary | Add the case to `golden_set.yaml`, adjust the router prompt, re-run eval |
| Bot silent in Slack | Bot not running, V1 bot still running on the same tokens, or messaging in a channel | One bot at a time; DM via Apps |
| `channel_not_found` | Channel name instead of ID in `ESCALATION_CHANNEL` | Use the `C...` ID |
