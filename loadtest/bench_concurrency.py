"""Latency under concurrency: p50/p95/p99 at 1, 10, 50 concurrent clients.

    # routing decision only (no upstream call): isolates the router's own overhead
    python -m loadtest.bench_concurrency --target route
    # full path through the gateway vs. calling the models directly
    python -m loadtest.bench_concurrency --target gateway direct-cheap direct-strong --requests 200

Start the gateway first:  uvicorn gateway.app:app --port 8000
Writes results/latency.md and results/latency.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import httpx
import numpy as np

from router import db as dbm
from router.config import CHEAP, DB_PATH, RESULTS_DIR, STRONG, UPSTREAM_BASE_URL, get_api_key

FALLBACK = ["What is 17 * 23?", "Write a short poem about the sea.", "Explain how a hash map works.",
            "Translate 'good morning' to French.", "def is_prime(n): ... write this function."]


def load_prompts(n: int) -> list[str]:
    try:
        conn = dbm.connect(DB_PATH)
        rows = [r[0] for r in conn.execute("SELECT query_text FROM queries WHERE split='test' LIMIT 2000")]
        if rows:
            random.Random(0).shuffle(rows)
            return [r[:3000] for r in rows[:n]]
    except Exception:
        pass
    return [random.choice(FALLBACK) for _ in range(n)]


def target_spec(name: str, gateway: str):
    key = get_api_key()
    auth = {"Authorization": f"Bearer {key}"}
    if name == "route":
        return f"{gateway}/v1/route", lambda p: {"messages": [{"role": "user", "content": p}]}, {}
    if name == "gateway":
        return f"{gateway}/v1/chat/completions", lambda p: {"messages": [{"role": "user", "content": p}], "max_tokens": 128}, {}
    model = CHEAP.name if name == "direct-cheap" else STRONG.name
    return (f"{UPSTREAM_BASE_URL}/chat/completions",
            lambda p: {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": 128}, auth)


async def run_level(url, make, headers, prompts, concurrency):
    sem, lat, errors = asyncio.Semaphore(concurrency), [], 0
    limits = httpx.Limits(max_connections=concurrency + 5)
    async with httpx.AsyncClient(timeout=120, limits=limits) as client:
        async def one(p):
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post(url, json=make(p), headers=headers)
                    if r.status_code != 200:
                        errors += 1
                        return
                except httpx.HTTPError:
                    errors += 1
                    return
                lat.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        await asyncio.gather(*(one(p) for p in prompts))
        wall = time.perf_counter() - t0
    a = np.array(lat) if lat else np.array([np.nan])
    return {"concurrency": concurrency, "n": len(prompts), "errors": errors, "rps": len(lat) / wall,
            "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean())}


async def main_async(a):
    results = {}
    for name in a.target:
        url, make, headers = target_spec(name, a.gateway)
        results[name] = []
        for c in a.concurrency:
            n = max(a.requests, c * 5)
            r = await run_level(url, make, headers, load_prompts(n), c)
            results[name].append(r)
            print(f"{name:14s} c={c:<3d} p50={r['p50_ms']:8.1f} p95={r['p95_ms']:8.1f} p99={r['p99_ms']:8.1f} "
                  f"rps={r['rps']:6.1f} errors={r['errors']}")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "latency.json").write_text(json.dumps(results, indent=2))
    lines = ["| target | concurrency | n | errors | RPS | p50 ms | p95 ms | p99 ms |", "|---|---|---|---|---|---|---|---|"]
    for name, rows in results.items():
        for r in rows:
            lines.append(f"| {name} | {r['concurrency']} | {r['n']} | {r['errors']} | {r['rps']:.1f} | "
                         f"{r['p50_ms']:.1f} | {r['p95_ms']:.1f} | {r['p99_ms']:.1f} |")
    if "direct-cheap" in results and "direct-strong" in results and "route" in results:
        saved = results["direct-strong"][0]["p50_ms"] - results["direct-cheap"][0]["p50_ms"]
        ovh = results["route"][0]["p50_ms"]
        lines += ["", f"Latency budget at c=1: cheap model is {saved:.0f} ms faster (p50) than strong; "
                      f"routing costs {ovh:.0f} ms (p50) => net {'WIN' if saved > ovh else 'LOSS'} when routed cheap."]
    (RESULTS_DIR / "latency.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", nargs="+", default=["route"], choices=["route", "gateway", "direct-cheap", "direct-strong"])
    ap.add_argument("--gateway", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 10, 50])
    ap.add_argument("--requests", type=int, default=200)
    asyncio.run(main_async(ap.parse_args()))
