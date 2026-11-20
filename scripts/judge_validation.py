"""Validate the LLM judge against your own labels.

    python -m scripts.judge_validation export --n 100     # writes results/judge_validation.csv
    # open the CSV, fill the `human_pass` column with 1 (acceptable) or 0, save
    python -m scripts.judge_validation score              # agreement + Cohen's kappa
"""
import argparse

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from router import db as dbm
from router.config import RESULTS_DIR

CSV = RESULTS_DIR / "judge_validation.csv"


def export(n: int):
    conn = dbm.connect()
    df = pd.read_sql_query(
        "SELECT q.query_id, q.domain, q.query_text AS prompt, q.reference, e.model_name, e.response_text AS response, "
        "e.score AS judge_score, e.is_correct AS judge_pass FROM evaluation_cache e JOIN queries q USING(query_id) "
        "WHERE q.grader='judge'", conn)
    if df.empty:
        raise SystemExit("No judge-graded rows yet.")
    sample = df.sample(min(n, len(df)), random_state=0).copy()
    sample["human_pass"] = ""
    RESULTS_DIR.mkdir(exist_ok=True)
    sample.to_csv(CSV, index=False)
    print(f"Wrote {len(sample)} rows to {CSV}. Fill human_pass (1/0) and run `score`.")


def score():
    df = pd.read_csv(CSV)
    df = df[df["human_pass"].astype(str).isin(["0", "1", "0.0", "1.0"])]
    if df.empty:
        raise SystemExit("Fill the human_pass column first.")
    h, j = df["human_pass"].astype(float).astype(int), df["judge_pass"].astype(int)
    print(f"n={len(df)}  agreement={(h == j).mean():.1%}  Cohen's kappa={cohen_kappa_score(h, j):.3f}")
    print("kappa > 0.6 = substantial agreement. Below ~0.4, don't trust judge-graded domains.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "score"])
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    export(a.n) if a.cmd == "export" else score()
