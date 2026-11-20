"""Robustness report for the trained router (run after scripts.train_router).

    python -m scripts.robustness

Writes results/robustness.md (+ CSVs): perturbation stability, prompt-injection probes,
multilingual probe and leave-one-domain-out (out-of-distribution) performance.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import pandas as pd

from router import db as dbm
from router import evaluation as ev
from router.config import ARTIFACT_PATH, DB_PATH, RESULTS_DIR
from router.inference import LoadedRouter
from router.robustness import leave_one_domain_out, multilingual_probe, perturbation_report
from scripts.evaluate_router import md_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--artifact", default=str(ARTIFACT_PATH))
    ap.add_argument("--results", default=str(RESULTS_DIR))
    ap.add_argument("--target", type=float, default=0.95)
    a = ap.parse_args()
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)

    conn = dbm.connect(a.db)
    df = ev.load_frame(conn).reset_index(drop=True)
    conn.close()
    art = joblib.load(a.artifact)
    router = LoadedRouter(art)
    tau = router.threshold
    test = df[df.split == "test"].reset_index(drop=True)
    print(f"router={router.name} tau={tau} test_n={len(test)}")

    md = ["# Robustness report", ""]
    if math.isinf(tau):
        md += ["> The deployed threshold is infinite (target unreachable on validation), so every request already goes to the strong "
               "model and the perturbation tables below are trivial.", ""]
    pert = perturbation_report(test, router.p_cheap, tau)
    pert.to_csv(out / "robustness_perturbations.csv", index=False)
    md += ["## Benign rewrites and prompt-injection probes (test split)", "",
           "`flip_rate`: decisions that changed. `harmful_flip_rate`: strong->cheap flips on queries where the cheap model is worse "
           "than strong. For injections this should be ~0; a large value means the router can be talked into the cheap model.", "",
           md_table(pert), ""]
    worst = pert[pert.kind == "injection"]["harmful_flip_rate"].max()
    md += [f"**Worst injection harmful-flip rate: {100 * worst:.2f}% of queries.** "
           f"Worst benign flip rate: {100 * pert[pert.kind == 'benign']['flip_rate'].max():.1f}%.", ""]

    ml = multilingual_probe(router.p_cheap, tau)
    ml.to_csv(out / "robustness_multilingual.csv", index=False)
    summ = ml.groupby("lang").agg(mean_abs_dp=("abs_dp", "mean"), same_decision=("same_decision", "mean")).reset_index()
    md += ["## Multilingual probe (4 hand-written parallel questions x 4 languages; a stability check, not a benchmark)", "",
           md_table(summ), "",
           "all-MiniLM-L6-v2 is an English-centric embedder; large shifts here mean you should add multilingual data or route non-English "
           "prompts to the strong model by rule.", ""]

    emb = router.embedder.encode(df["query_text"].tolist()) if router.embedder is not None else None
    if emb is None:
        import numpy as np
        emb = np.zeros((len(df), 1), dtype="float32")
    lodo = leave_one_domain_out(df, emb, candidate=art["name"], target=a.target, safety=art.get("safety", 0.01))
    lodo.to_csv(out / "robustness_lodo.csv", index=False)
    md += ["## Leave-one-domain-out (out-of-distribution)", "",
           f"Trained WITHOUT the held-out domain, tau tuned on the other domains, scored on the whole held-out domain. "
           f"`ood_meets_target` shows whether quality retention stayed >= {a.target:.0%}.", "", md_table(lodo), ""]
    n_fail = int((~lodo["ood_meets_target"]).sum()) if len(lodo) else 0
    md += [f"**{n_fail} of {len(lodo)} held-out domains missed the quality target.** Missing the target on an unseen domain is "
           "the expected failure mode of a learned router; the mitigation is the uncertain band (`ROUTER_UNCERTAIN_BAND`), hard "
           "rules, shadow mode, and retraining when new traffic types appear.", ""]
    (out / "robustness.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:14]))
    print(f"\nFull report: {out / 'robustness.md'}")


if __name__ == "__main__":
    main()
