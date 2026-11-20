"""SQLite schema and helpers. One writer at a time; WAL mode so readers don't block."""
from __future__ import annotations

import random
import sqlite3
from pathlib import Path

from .config import DB_PATH, SEED, SPLIT_FRACTIONS

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    query_id       TEXT PRIMARY KEY,
    domain         TEXT NOT NULL,
    source_dataset TEXT,
    query_text     TEXT NOT NULL,   -- exactly what the router sees (no format instructions)
    reference      TEXT,            -- ground truth / reference answer / unit tests
    grader         TEXT NOT NULL,   -- numeric | mcq | code | judge
    meta           TEXT,            -- JSON: system prompt, test setup, etc.
    split          TEXT             -- train | val | test
);
CREATE TABLE IF NOT EXISTS evaluation_cache (
    query_id         TEXT,
    model_name       TEXT,
    response_text    TEXT,
    is_correct       INTEGER,
    score            REAL,
    cost_usd         REAL,
    latency_ms       REAL,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    reasoning_tokens INTEGER,
    finish_reason    TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (query_id, model_name)
);
CREATE INDEX IF NOT EXISTS idx_queries_domain ON queries(domain);
CREATE INDEX IF NOT EXISTS idx_queries_split ON queries(split);

CREATE TABLE IF NOT EXISTS request_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    request_id TEXT, selected_model TEXT, reason TEXT,
    p_cheap REAL, threshold REAL, overhead_ms REAL, upstream_ms REAL,
    status INTEGER, fallback INTEGER, escalated INTEGER, stream INTEGER,
    prompt_chars INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER,
    cost_usd REAL, est_saved_usd REAL
);
CREATE TABLE IF NOT EXISTS shadow_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    request_id TEXT, prompt TEXT, p_cheap REAL,
    cheap_response TEXT, strong_response TEXT,
    cheap_cost REAL, strong_cost REAL, agree REAL
);
"""


def connect(path: str | Path | None = None, check_same_thread: bool = True) -> sqlite3.Connection:
    path = Path(path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(path: str | Path | None = None) -> sqlite3.Connection:
    conn = connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def upsert_queries(conn: sqlite3.Connection, items: list[dict]) -> int:
    """Insert new queries; never overwrite an existing row (keeps its split stable)."""
    before = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
    conn.executemany(
        "INSERT OR IGNORE INTO queries (query_id, domain, source_dataset, query_text, reference, grader, meta) "
        "VALUES (:query_id, :domain, :source_dataset, :query_text, :reference, :grader, :meta)",
        items,
    )
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0] - before


def assign_splits(conn: sqlite3.Connection, seed: int = SEED, fractions=SPLIT_FRACTIONS) -> dict:
    """Stratified (per-domain) split. Only touches rows whose split is NULL, so queries
    added later never reshuffle queries that already have a split (no train/test leakage)."""
    rng = random.Random(seed)
    rows = conn.execute("SELECT query_id, domain FROM queries WHERE split IS NULL ORDER BY query_id").fetchall()
    by_domain: dict[str, list[str]] = {}
    for qid, dom in rows:
        by_domain.setdefault(dom, []).append(qid)
    for dom in sorted(by_domain):
        ids = by_domain[dom]
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * fractions[0]))
        n_val = int(round(n * fractions[1]))
        for i, qid in enumerate(ids):
            split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            conn.execute("UPDATE queries SET split=? WHERE query_id=?", (split, qid))
    conn.commit()
    return dict(conn.execute("SELECT split, COUNT(*) FROM queries GROUP BY split").fetchall())
