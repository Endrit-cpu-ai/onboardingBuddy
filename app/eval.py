"""Step 10: EVAL. Run the golden set through the whole pipeline (router -> search -> rerank -> answer)
and score it. Every case runs twice: if the two runs disagree, the case is FLAKY.

    python -m app.eval                  # full run, 2 runs per case
    python -m app.eval --runs 3         # stricter flakiness check
    python -m app.eval -v               # also print every answer
    python -m app.eval --only "vpn"     # just the cases whose question contains "vpn"

Checks per case: outcome, cited source, required words, forbidden words, answer language.
Eval questions are logged under "eval:<user>" and deleted from the SQLite log at the end.
"""
import argparse
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

from app import handler
from app.config import EMBED_MODEL, GEN_EFFORT, GEN_MODEL, GOLDEN_SET, SCORE_THRESHOLD
from app.embeddings import embed_query
from app.ingest import load_docs
from app.store import delete_user

# eval users are "eval:default", "eval:manager"... -> same profile as users.yaml, but logged separately
_load_user = handler.load_user
handler.load_user = lambda key: _load_user(key.removeprefix("eval:"))

CYRILLIC = re.compile(r"[\u0400-\u04FF]")
ALBANIAN = re.compile(r"[ëç]|\b(dhe|në|për|që|është|një|të)\b", re.IGNORECASE)


def check(case, reply, urls):
    """Return a list of failures for one run (empty = pass)."""
    fails = []
    expected = case["expect"] if isinstance(case["expect"], list) else [case["expect"]]
    if reply.outcome not in expected:
        fails.append(f"expected {'/'.join(expected)}, got {reply.outcome}" + (f" ({reply.reason})" if reply.reason else ""))
        return fails
    if reply.outcome != "answered":
        return fails
    text = reply.text.lower()
    if case.get("source") and urls[case["source"]] not in {s["url"] for s in reply.sources}:
        fails.append(f"didn't cite {case['source']}")
    missing = [w for w in case.get("contains", []) if w.lower() not in text]
    if missing:
        fails.append(f"missing {missing}")
    found = [w for w in case.get("not", []) if w.lower() in text]
    if found:
        fails.append(f"contains forbidden {found}")
    lang = case.get("lang")
    if lang == "mk" and not CYRILLIC.search(reply.text):
        fails.append("answer not in Macedonian")
    if lang == "sq" and not ALBANIAN.search(reply.text):
        fails.append("answer not in Albanian")
    return fails


def run_case(case, urls):
    user_key = "eval:" + case.get("user", "default")
    t0 = time.time()
    reply = handler.handle(case["q"], user_key)
    return reply, check(case, reply, urls), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4, help="questions in parallel")
    ap.add_argument("--only", default="", help="only cases whose question contains this text")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the eval rows in the SQLite log")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    with open(GOLDEN_SET, encoding="utf-8") as f:
        cases = [c for c in yaml.safe_load(f) if args.only.lower() in c["q"].lower()]
    urls = {d.path: d.url for d in load_docs()}
    for c in cases:
        if c.get("source") and c["source"] not in urls:
            raise SystemExit(f"golden_set.yaml: unknown source {c['source']!r}")

    print(f"{len(cases)} cases x {args.runs} runs | embed {EMBED_MODEL} | answer {GEN_MODEL} ({GEN_EFFORT}) "
          f"| threshold {SCORE_THRESHOLD}\n")
    embed_query("warm up")          # load the model once before the threads start

    jobs = [(i, c) for i, c in enumerate(cases) for _ in range(args.runs)]
    results = {i: [] for i in range(len(cases))}
    t_start = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [(i, pool.submit(run_case, c, urls)) for i, c in jobs]
        for i, fut in futures:
            try:
                results[i].append(fut.result())
            except Exception as e:                      # a crash is a failed run, not a crashed eval
                results[i].append((None, [f"crashed: {type(e).__name__}: {e}"], 0))

    passed, flaky, failed, latencies = 0, [], [], []
    for i, c in enumerate(cases):
        runs = results[i]
        oks = [not fails for _, fails, _ in runs]
        latencies += [sec for r, _, sec in runs if r and r.outcome == "answered"]
        if all(oks):
            passed += 1
            mark = "ok   "
        elif any(oks):                                  # passed sometimes = flaky
            flaky.append(c["q"])
            mark = "FLAKY"
        else:
            failed.append(c["q"])
            mark = "FAIL "
        who = f" [{c['user']}]" if c.get("user") else ""
        exp = c["expect"] if isinstance(c["expect"], str) else "/".join(c["expect"])
        print(f"{mark} {exp:<10} {c['q']}{who}")
        for n, (reply, fails, _) in enumerate(runs, start=1):
            if fails:
                print(f"        run {n}: {'; '.join(fails)}")
            if args.verbose and reply:
                print(f"        run {n} -> {reply.text[:250]!r}")

    total = len(cases)
    print(f"\n=== {passed}/{total} passed ({100 * passed // max(total, 1)}%) | "
          f"{len(failed)} failed | {len(flaky)} flaky ===")
    if latencies:
        print(f"answered: avg {sum(latencies) / len(latencies):.1f}s, slowest {max(latencies):.1f}s "
              f"(with {args.workers} in parallel) | total {time.time() - t_start:.0f}s")

    if not args.keep:
        for user in {"eval:" + c.get("user", "default") for c in cases}:
            delete_user(user)


if __name__ == "__main__":
    main()