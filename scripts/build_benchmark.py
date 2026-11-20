"""Build the mixed benchmark: run the cheap and strong model on every query, grade, and cache.

    python -m scripts.build_benchmark                     # ~3,000 queries (see router/sources.py)
    python -m scripts.build_benchmark --scale 0.1         # 10% smoke test
    python -m scripts.build_benchmark --domains gsm8k mmlu --workers 8

Resumable: rows already in evaluation_cache are skipped; failed calls are NOT stored so the
next run retries them. Code questions execute model-written code: use Docker for isolation.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from router import db as dbm
from router import graders
from router.config import (DB_PATH, JUDGE_MODEL, JUDGE_PASS_SCORE, MAX_TOKENS, MODELS, STRONG_REASONING_EFFORT,
                           UPSTREAM_BASE_URL, get_api_key)
from router.sources import DEFAULT_COUNTS, LOADERS

_TPD_BLOCKED: set[str] = set()
_TPD_LOCK = threading.Lock()


def is_tpd_limit(err: Exception) -> bool:
    msg = str(err).lower()
    return "429" in msg and ("tokens per day" in msg or "(tpd)" in msg)


def make_client() -> OpenAI:
    key = get_api_key()
    if not key:
        raise SystemExit("Set GROQ_API_KEY (or XAI_API_KEY) first.")
    return OpenAI(api_key=key, base_url=UPSTREAM_BASE_URL, max_retries=0, timeout=120)


def with_retries(fn, attempts: int = 4):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - network/rate-limit errors are all treated alike
            if is_tpd_limit(e) or i == attempts - 1:
                raise
            time.sleep(min(30, 2 ** i + random.random()))


def call_model(client: OpenAI, tier: str, system: str | None, prompt: str) -> dict:
    spec = MODELS[tier]
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    extra = {"reasoning_effort": STRONG_REASONING_EFFORT} if "gpt-oss" in spec.name else {}

    def _go():
        t0 = time.perf_counter()
        r = client.chat.completions.create(model=spec.name, messages=messages, temperature=0.0,
                                           max_completion_tokens=MAX_TOKENS, extra_body=extra or None)
        return r, (time.perf_counter() - t0) * 1000

    r, latency = with_retries(_go)
    u = r.usage
    details = getattr(u, "completion_tokens_details", None) if u else None
    ch = r.choices[0]
    return {
        "text": ch.message.content or "", "latency_ms": latency, "finish_reason": ch.finish_reason,
        "in_tok": getattr(u, "prompt_tokens", 0) or 0, "out_tok": getattr(u, "completion_tokens", 0) or 0,
        "reasoning_tok": getattr(details, "reasoning_tokens", 0) or 0,
    }


def judge_call(client: OpenAI, messages: list[dict]) -> str:
    def _go():
        r = client.chat.completions.create(model=JUDGE_MODEL, messages=messages, temperature=0.0, max_completion_tokens=200)
        return r.choices[0].message.content or ""
    return with_retries(_go)


def work(client, q: dict, tier: str, judge_pass: int, skip_judge: bool):
    """Returns an evaluation_cache row tuple, or None if anything failed (=> retried next run)."""
    with _TPD_LOCK:
        if tier in _TPD_BLOCKED:
            return None
    try:
        meta = json.loads(q["meta"] or "{}")
        res = call_model(client, tier, meta.get("system"), q["query_text"])
        if res["out_tok"] == 0 and not res["text"]:
            return None
        if q["grader"] == "judge":
            if skip_judge:
                return None
            g = graders.grade_judge(lambda m: judge_call(client, m), q["query_text"], res["text"],
                                    q["reference"], judge_pass)
            if g is None:
                return None
        else:
            g = graders.grade(q["grader"], res["text"], q["reference"], meta)
        cost = MODELS[tier].cost(res["in_tok"], res["out_tok"])
        return (q["query_id"], MODELS[tier].name, res["text"], int(g.correct), g.score, cost, res["latency_ms"],
                res["in_tok"], res["out_tok"], res["reasoning_tok"], res["finish_reason"])
    except Exception as e:  # noqa: BLE001
        if is_tpd_limit(e):
            with _TPD_LOCK:
                if tier not in _TPD_BLOCKED:
                    _TPD_BLOCKED.add(tier)
                    tqdm.write(f"[{tier}] Daily TPD limit reached — skipping remaining '{tier}' calls: {str(e)[:180]}")
        else:
            tqdm.write(f"[{q['query_id']}/{tier}] failed: {type(e).__name__}: {str(e)[:120]}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--domains", nargs="*", choices=sorted(LOADERS), default=sorted(LOADERS))
    ap.add_argument("--scale", type=float, default=1.0, help="multiply default per-dataset counts")
    ap.add_argument("--workers", type=int, default=4, help="parallel API calls (mind your rate limits)")
    ap.add_argument("--judge-pass", type=int, default=JUDGE_PASS_SCORE)
    ap.add_argument("--skip-judge", action="store_true", help="skip chat/docqa (judge-graded) domains")
    a = ap.parse_args()

    conn = dbm.init_db(a.db)
    items = []
    for name in a.domains:
        n = max(1, int(DEFAULT_COUNTS[name] * a.scale))
        print(f"Loading {name} ({n})...")
        items += LOADERS[name](n)
    print(f"New queries added: {dbm.upsert_queries(conn, items)}")
    print("Splits:", dbm.assign_splits(conn))

    cols = ["query_id", "domain", "source_dataset", "query_text", "reference", "grader", "meta"]
    wanted = {i["query_id"] for i in items}
    queries = [dict(zip(cols, r)) for r in conn.execute(f"SELECT {','.join(cols)} FROM queries")]
    queries = [q for q in queries if q["query_id"] in wanted]
    done = set(conn.execute("SELECT query_id, model_name FROM evaluation_cache").fetchall())
    todo = [(q, t) for q in queries for t in ("cheap", "strong") if (q["query_id"], MODELS[t].name) not in done]
    print(f"{len(todo)} model calls to make ({len(queries) * 2 - len(todo)} cached)")
    if not todo:
        return

    client = make_client()
    insert = ("INSERT OR REPLACE INTO evaluation_cache (query_id, model_name, response_text, is_correct, score, "
              "cost_usd, latency_ms, input_tokens, output_tokens, reasoning_tokens, finish_reason) "
              "VALUES (?,?,?,?,?,?,?,?,?,?,?)")
    ok = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, client, q, t, a.judge_pass, a.skip_judge) for q, t in todo]
        for f in tqdm(as_completed(futs), total=len(futs), desc="Evaluating"):
            row = f.result()
            if row:
                conn.execute(insert, row)   # single writer thread
                conn.commit()
                ok += 1
    print(f"Stored {ok}/{len(todo)} new rows. Re-run to retry failures. Next: python -m scripts.audit")


if __name__ == "__main__":
    main()