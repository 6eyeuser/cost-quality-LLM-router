"""Synthetic benchmark for tests, CI and offline demos. NOT real data: never report these numbers.

Difficulty correlates (noisily) with prompt length and hard-keywords, which is exactly the
structure a router can exploit, so the pipeline has something to learn.
"""
from __future__ import annotations

import json

import numpy as np

from . import db as dbm
from .config import CHEAP, STRONG

DOMAINS = {
    "math": (["apples", "train", "speed", "total", "each", "remaining", "ratio", "profit"], 0.55, "numeric"),
    "code": (["function", "python", "list", "return", "sort", "string", "def", "recursion"], 0.60, "code"),
    "knowledge": (["which", "following", "history", "capital", "theory", "chemistry", "law"], 0.45, "mcq"),
    "chat": (["write", "friendly", "email", "tips", "recipe", "story", "explain", "simple"], 0.15, "judge"),
}
HARD = ["prove", "derive", "integral", "asymptotic", "optimize", "concurrency", "theorem"]
FILLER = ["the", "a", "of", "and", "to", "in", "is", "for", "on", "with", "that", "it", "as", "by", "at"]


def populate(conn, n: int = 1200, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    names = list(DOMAINS)
    items, hardness = [], {}
    for i in range(n):
        dom = names[rng.integers(len(names))]
        vocab, base_hard, grader = DOMAINS[dom]
        d = float(np.clip(rng.beta(2, 2) * 0.6 + base_hard * 0.4, 0, 1))
        n_words = 4 + int(d * 45) + int(rng.integers(0, 6))
        words = [rng.choice(vocab) if rng.random() < 0.35 else rng.choice(FILLER) for _ in range(n_words)]
        words += [rng.choice(HARD) for _ in range(int(rng.poisson(d * 2.5)))]
        text = f"{dom} {' '.join(words)} ?"
        qid = f"syn_{i}"
        hardness[qid] = d
        items.append({"query_id": qid, "domain": dom, "source_dataset": "synthetic", "query_text": text,
                      "reference": "1", "grader": grader, "meta": json.dumps({})})
    dbm.upsert_queries(conn, items)
    dbm.assign_splits(conn)
    rows = []
    for it in items:
        d = hardness[it["query_id"]]
        for spec, p_ok, base_tok, lat in ((CHEAP, 0.97 - 0.85 * d, 120, 400), (STRONG, 0.97 - 0.18 * d, 500, 1500)):
            ok = int(rng.random() < p_ok)
            out_tok = int(base_tok * (0.6 + d) * rng.uniform(0.7, 1.3))
            in_tok = len(it["query_text"].split()) * 2 + 30
            finish = "length" if (spec is CHEAP and rng.random() < 0.02) else "stop"
            rows.append((it["query_id"], spec.name, "resp", ok, float(ok), spec.cost(in_tok, out_tok),
                         lat * rng.uniform(0.7, 1.4), in_tok, out_tok, 0, finish))
    conn.executemany("INSERT OR REPLACE INTO evaluation_cache (query_id, model_name, response_text, is_correct, score, "
                     "cost_usd, latency_ms, input_tokens, output_tokens, reasoning_tokens, finish_reason) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
