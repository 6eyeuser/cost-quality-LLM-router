"""Summarise live shadow-mode data and check for drift (run against the gateway's log DB)."""
import json

import pandas as pd

from router import db as dbm
from router.config import ARTIFACT_PATH, DB_PATH


def main():
    conn = dbm.connect(DB_PATH)
    req = pd.read_sql_query("SELECT * FROM request_log", conn)
    sh = pd.read_sql_query("SELECT * FROM shadow_log", conn)
    if req.empty:
        print("No live traffic logged yet.")
        return
    import joblib
    art = joblib.load(ARTIFACT_PATH)
    print(f"Live requests: {len(req)} | routed cheap: {(req.selected_model == art['cheap_model']).mean():.1%} | "
          f"fallbacks: {int(req.fallback.sum())} | escalations: {int(req.escalated.sum())}")
    print(f"Estimated savings so far: ${req.est_saved_usd.sum():.4f} on ${req.cost_usd.sum():.4f} spent")
    print(f"Overhead ms: p50={req.overhead_ms.quantile(.5):.1f} p95={req.overhead_ms.quantile(.95):.1f}")
    ml = req[req.reason == "ml"]
    if len(ml):
        print(f"Mean live p(cheap)={ml.p_cheap.mean():.3f} vs training cheap-ok rate={art['train_cheap_ok_rate']:.3f} "
              "(a large gap suggests the traffic distribution has drifted)")
    if sh.empty:
        print("No shadow samples. Set SHADOW_RATE=0.05 on the gateway to collect them.")
        return
    a = sh.agree.dropna()
    print(f"Shadow samples: {len(sh)} | numeric-answer agreement (where comparable): {a.mean():.1%} over {len(a)}")
    print(f"Shadow cost ratio strong/cheap: {sh.strong_cost.sum() / max(sh.cheap_cost.sum(), 1e-12):.1f}x")


if __name__ == "__main__":
    main()
