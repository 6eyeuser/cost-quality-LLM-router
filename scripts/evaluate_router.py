"""Final evaluation on the held-out TEST split. Run this once, after model selection.

    python -m scripts.evaluate_router

For every router (and baseline) the threshold is tuned on VAL, frozen, then applied to TEST.
Outputs results/summary.md, results/results.json and the plots used in the README.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from router import db as dbm
from router import evaluation as ev
from router.config import ARTIFACT_PATH, DB_PATH, RESULTS_DIR

TARGETS = (0.95, 0.99)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                                       else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def fmt_ci(point: float, ci) -> str:
    return f"{100 * point:.1f}% [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--results", default=str(RESULTS_DIR))
    ap.add_argument("--artifact", default=str(ARTIFACT_PATH))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--overhead-ms", type=float, default=0.0,
                    help="routing overhead (ms). 0 = use p50 from results/latency.json if present, else 0")
    ap.add_argument("--vcpu-usd-hour", type=float, default=0.04, help="cloud vCPU price used to cost the router itself")
    a = ap.parse_args()
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)

    ovh_ms = a.overhead_ms
    if ovh_ms <= 0 and (out / "latency.json").exists():
        try:
            ovh_ms = float(json.loads((out / "latency.json").read_text())["route"][0]["p50_ms"])
        except (KeyError, IndexError, ValueError):
            ovh_ms = 0.0
    ovh_usd = ev.router_overhead_usd(ovh_ms, a.vcpu_usd_hour)
    print(f"Router overhead used: {ovh_ms:.1f} ms/request = ${ovh_usd:.2e}/request (charged to routed strategies, not to always-strong)")

    conn = dbm.connect(a.db)
    df = ev.load_frame(conn).reset_index(drop=True)
    conn.close()
    cheap_name, strong_name = df.attrs["cheap"], df.attrs["strong"]
    preds = pd.read_csv(out / "predictions.csv")
    art = joblib.load(a.artifact)
    sel = art["name"]
    df = df.merge(preds.drop(columns=["domain", "split"]), on="query_id", how="inner")
    val = df[df.split == "val"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    print(f"val={len(val)} test={len(test)} | cheap={cheap_name} strong={strong_name}")

    cand_cols = [c for c in df.columns if c.startswith("p_")]
    train_max = art["train_length_ref_max"]

    # ---- score providers for each method ----------------------------------------------------
    methods: dict[str, tuple[np.ndarray, np.ndarray]] = {
        f"ML router ({sel})": (val[f"p_{sel}"].to_numpy(), test[f"p_{sel}"].to_numpy()),
        "length heuristic": (ev.length_scores(val, train_max), ev.length_scores(test, train_max)),
        "oracle": (ev.oracle_scores(val), ev.oracle_scores(test)),
    }
    ml_name = f"ML router ({sel})"

    results, rows = {}, []
    for target in TARGETS:
        for name, (pv, pt) in methods.items():
            o = 0.0 if name == "oracle" else ovh_usd      # the oracle is a bound, not a deployable router
            tau = ev.tune_tau(val, pv, target, o)
            m = ev.apply_tau(test, pt, tau, o)
            ci = ev.bootstrap_ci(test, pt >= tau, a.n_boot, overhead_usd=o)
            rows.append({"target": f"{target:.0%}", "method": name, "tau": tau, "pct_cheap": m["pct_cheap"],
                         "retention": fmt_ci(m["retention"], ci["retention_ci"]),
                         "savings": fmt_ci(m["savings"], ci["savings_ci"]),
                         "P(ret>=target)": float((ci["retention_samples"] >= target).mean())})
            results[f"{name}@{target}"] = {**m, "savings_ci": ci["savings_ci"], "retention_ci": ci["retention_ci"]}
        # random router: analytic expectation at the val-tuned cheap fraction
        f = float(ev.best_point(ev.random_curve(val), target)["pct_cheap"])
        ca, sa = test.correct_cheap.mean(), test.correct_strong.mean()
        cc, sc = test.cost_cheap.mean(), test.cost_strong.mean()
        acc, cost = f * ca + (1 - f) * sa, f * cc + (1 - f) * sc
        rows.append({"target": f"{target:.0%}", "method": "random (expected)", "tau": np.nan, "pct_cheap": f,
                     "retention": f"{100 * acc / sa:.1f}%", "savings": f"{100 * (1 - cost / sc):.1f}%",
                     "P(ret>=target)": np.nan})
    main_table = pd.DataFrame(rows)

    # ---- always-X anchors ---------------------------------------------------------------------
    anchors = pd.DataFrame([
        {"strategy": "always strong", "accuracy": test.correct_strong.mean(), "avg_cost_usd": test.cost_strong.mean(),
         "avg_latency_ms": test.lat_strong.mean()},
        {"strategy": "always cheap", "accuracy": test.correct_cheap.mean(), "avg_cost_usd": test.cost_cheap.mean(),
         "avg_latency_ms": test.lat_cheap.mean()},
    ])

    # ---- significance vs the length heuristic ------------------------------------------------
    pv_ml, pt_ml = methods[ml_name]
    tau_ml = ev.tune_tau(val, pv_ml, 0.95, ovh_usd)
    pv_len, pt_len = methods["length heuristic"]
    tau_len = ev.tune_tau(val, pv_len, 0.95, ovh_usd)
    paired = ev.paired_bootstrap_diff(test, pt_ml >= tau_ml, pt_len >= tau_len, a.n_boot, overhead_usd=ovh_usd)

    # ---- ablation (all candidates, val-tuned tau, test numbers) -------------------------------
    ab = []
    for c in cand_cols:
        tau = ev.tune_tau(val, val[c].to_numpy(), 0.95, ovh_usd)
        m = ev.apply_tau(test, test[c].to_numpy(), tau, ovh_usd)
        ab.append({"candidate": c[2:], "test_savings@95": m["savings"], "test_retention": m["retention"],
                   "pct_cheap": m["pct_cheap"], "selected_on_val": "<-- selected" if c == f"p_{sel}" else ""})
    ablation = pd.DataFrame(ab).sort_values("test_savings@95", ascending=False)

    # ---- deployed threshold, per-domain, cascade, calibration ---------------------------------
    tau_dep = float(art["threshold"])
    per_dom = ev.per_domain(test, pt_ml, tau_dep, ovh_usd)
    casc = []
    for label, tau in (("router only", tau_dep), ("router + safety net (cascade)", tau_dep), ("cheap-first cascade (all)", 0.0)):
        if label == "router only":
            mask = pt_ml >= tau
            m = ev.apply_tau(test, pt_ml, tau, ovh_usd)
            casc.append({"strategy": label, "savings": m["savings"], "retention": m["retention"],
                         "escalated": 0.0, "avg_latency_ms": ev.strategy_latency(test, mask, ovh_ms)})
        else:
            m = ev.cascade_metrics(test, pt_ml, tau, ovh_ms, ovh_usd)
            casc.append({"strategy": label, "savings": m["savings"], "retention": m["retention"],
                         "escalated": m["pct_escalated"], "avg_latency_ms": m["avg_latency_ms"]})
    casc.append({"strategy": "always strong", "savings": 0.0, "retention": 1.0, "escalated": 0.0,
                 "avg_latency_ms": float(test.lat_strong.mean())})
    cascade_tbl = pd.DataFrame(casc)

    y_test = ev.cheap_ok_label(test, art.get("label_mode", "as_good"))
    cal = {c[2:]: ev.calibration_stats(y_test, test[c].to_numpy()) for c in cand_cols
           if c[2:] in (sel, "lr_both", "hgb_both", "knn_emb")}

    # ---- plots --------------------------------------------------------------------------------
    curves = {ml_name: ev.sweep(test, pt_ml), "length heuristic": ev.sweep(test, pt_len),
              "random": ev.random_curve(test), "oracle": ev.sweep(test, methods["oracle"][1])}
    dep = ev.apply_tau(test, pt_ml, tau_dep, ovh_usd)
    ev.plot_pareto(curves, test, out / "pareto.png", points={"deployed tau": (dep["avg_cost"] / test.cost_strong.mean(), dep["accuracy"])})
    ev.plot_reliability(cal, out / "reliability.png")
    ev.plot_domains(per_dom, out / "per_domain.png")

    # ---- write report -------------------------------------------------------------------------
    h = main_table[(main_table.target == "95%") & (main_table.method == ml_name)].iloc[0]
    headline = (f"**{ml_name}: {h['savings']} cost reduction at {h['retention']} quality retention "
                f"(95% bootstrap CI) on {len(test)} held-out test queries.**")
    md = [f"# Results (held-out test split, n={len(test)})", "", headline, "",
          f"_Router overhead charged: {ovh_ms:.1f} ms/request = ${ovh_usd:.2e}/request (vCPU at ${a.vcpu_usd_hour}/h)._", "",
          "![pareto](pareto.png)", "", "## Cost saved at a quality target (tau tuned on val, frozen on test)", "",
          md_table(main_table), "", "## Anchors", "", md_table(anchors, "{:.6f}"), "",
          "## Is the ML router better than the length heuristic? (paired bootstrap, 95% target)", "",
          f"Savings difference = {100 * paired['diff_mean']:+.2f} pts, 95% CI [{100 * paired['ci'][0]:+.2f}, "
          f"{100 * paired['ci'][1]:+.2f}], P(ML better) = {paired['p_a_better']:.2f}. "
          "If the CI contains 0, the difference is not statistically distinguishable from noise.", "",
          "## Ablation: features x model (selection was made on VAL; test shown for reporting)", "", md_table(ablation), "",
          "## Per-domain behaviour at the deployed threshold", "", md_table(per_dom.reset_index()), "",
          "![domains](per_domain.png)", "",
          "## Pre-routing vs. cascade (latency from measured per-model latencies)", "", md_table(cascade_tbl, "{:.3f}"), "",
          "## Calibration (test)", "",
          *(f"- {k}: ECE={v['ece']:.3f}, Brier={v['brier']:.3f}" for k, v in cal.items()), "", "![reliability](reliability.png)"]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    main_table.to_csv(out / "main_results.csv", index=False)
    ablation.to_csv(out / "ablation.csv", index=False)
    per_dom.to_csv(out / "per_domain.csv")
    print("\n".join(md[:8]))
    print(f"\nFull report: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
