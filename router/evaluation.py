"""Split-aware routing evaluation.

Conventions
-----------
* p is P(cheap model is good enough). A query goes to the cheap model iff p >= tau.
* tau is ALWAYS chosen on the validation split and then applied, frozen, to the test split.
* Savings = 1 - avg_cost / avg_cost_of_always_strong.  Retention = acc / acc_of_always_strong.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .config import CHEAP, STRONG

INF = float("inf")


# --------------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------------
def detect_models(conn: sqlite3.Connection) -> tuple[str, str]:
    """(cheap, strong) model names. Uses config names if present, else mean cost."""
    names = [r[0] for r in conn.execute("SELECT DISTINCT model_name FROM evaluation_cache")]
    if CHEAP.name in names and STRONG.name in names:
        return CHEAP.name, STRONG.name
    cost = pd.read_sql_query("SELECT model_name, AVG(cost_usd) c FROM evaluation_cache GROUP BY model_name", conn)
    if len(cost) < 2:
        raise RuntimeError("Need evaluations from two models. Run scripts.build_benchmark first.")
    cost = cost.sort_values("c")
    return cost.iloc[0]["model_name"], cost.iloc[-1]["model_name"]


def load_frame(conn: sqlite3.Connection, cheap: str | None = None, strong: str | None = None) -> pd.DataFrame:
    """One row per query that has BOTH models evaluated."""
    if cheap is None or strong is None:
        cheap, strong = detect_models(conn)
    sql = """
    SELECT q.query_id, q.domain, q.split, q.query_text,
           c.is_correct AS correct_cheap,  s.is_correct AS correct_strong,
           c.cost_usd   AS cost_cheap,     s.cost_usd   AS cost_strong,
           c.latency_ms AS lat_cheap,      s.latency_ms AS lat_strong,
           c.finish_reason AS finish_cheap, c.response_text AS resp_cheap,
           c.output_tokens AS out_cheap,   s.output_tokens AS out_strong
    FROM queries q
    JOIN evaluation_cache c ON c.query_id = q.query_id AND c.model_name = ?
    JOIN evaluation_cache s ON s.query_id = q.query_id AND s.model_name = ?
    WHERE q.split IS NOT NULL
    """
    df = pd.read_sql_query(sql, conn, params=(cheap, strong))
    for col in ("correct_cheap", "correct_strong"):
        df[col] = df[col].astype(float)
    df.attrs["cheap"], df.attrs["strong"] = cheap, strong
    return df


def cheap_ok_label(df: pd.DataFrame, mode: str = "as_good") -> np.ndarray:
    """Training target.
    as_good       : cheap is at least as good as strong (routing it cheap loses nothing). Default,
                    because when BOTH models fail there is no reason to pay for the strong one.
    cheap_correct : cheap model got it right (the original behaviour)."""
    if mode == "cheap_correct":
        return df["correct_cheap"].to_numpy().astype(int)
    return (df["correct_cheap"].to_numpy() >= df["correct_strong"].to_numpy()).astype(int)


# --------------------------------------------------------------------------------------------
# Curves and operating points
# --------------------------------------------------------------------------------------------
def _metrics_for_mask(df: pd.DataFrame, cheap_mask: np.ndarray, overhead_usd: float = 0.0) -> dict:
    """overhead_usd = per-request cost of running the router itself (CPU time). It is charged to every
    routed request but NOT to the always-strong baseline, which has no router."""
    acc = np.where(cheap_mask, df["correct_cheap"], df["correct_strong"]).mean()
    cost = np.where(cheap_mask, df["cost_cheap"], df["cost_strong"]).mean() + overhead_usd
    s_acc, s_cost = df["correct_strong"].mean(), df["cost_strong"].mean()
    return {
        "pct_cheap": float(cheap_mask.mean()),
        "accuracy": float(acc),
        "avg_cost": float(cost),
        "retention": float(acc / s_acc) if s_acc > 0 else np.nan,
        "savings": float(1 - cost / s_cost) if s_cost > 0 else np.nan,
    }


def sweep(df: pd.DataFrame, p: np.ndarray, overhead_usd: float = 0.0) -> pd.DataFrame:
    """Exact threshold sweep: one row for every distinct routing the scores can produce."""
    p = np.asarray(p, dtype=float)
    taus = np.concatenate([np.unique(p), [INF]])
    rows = []
    for tau in taus:
        m = _metrics_for_mask(df, p >= tau, overhead_usd)
        m["tau"] = tau
        rows.append(m)
    return pd.DataFrame(rows)


def random_curve(df: pd.DataFrame, n: int = 101) -> pd.DataFrame:
    """Analytic expectation of a random router that sends a fraction f to the cheap model."""
    ca, sa = df["correct_cheap"].mean(), df["correct_strong"].mean()
    cc, sc = df["cost_cheap"].mean(), df["cost_strong"].mean()
    f = np.linspace(0, 1, n)
    acc = f * ca + (1 - f) * sa
    cost = f * cc + (1 - f) * sc
    return pd.DataFrame({"tau": np.nan, "pct_cheap": f, "accuracy": acc, "avg_cost": cost,
                         "retention": acc / sa, "savings": 1 - cost / sc})


def best_point(sw: pd.DataFrame, target_retention: float) -> pd.Series:
    """Highest-savings operating point with retention >= target (ties -> most conservative tau).
    Falls back to 'always strong' if nothing qualifies."""
    ok = sw[sw["retention"] >= target_retention - 1e-12]
    if ok.empty:
        return sw.sort_values("savings").iloc[0]
    best = ok["savings"].max()
    cand = ok[ok["savings"] >= best - 1e-12]
    if cand["tau"].isna().all():
        return cand.iloc[0]
    return cand.sort_values("tau", ascending=False).iloc[0]


def tune_tau(df_val: pd.DataFrame, p_val: np.ndarray, target_retention: float, overhead_usd: float = 0.0) -> float:
    """Pick tau on the validation split. Returns inf (=always strong) if the target is unreachable."""
    sw = sweep(df_val, p_val, overhead_usd)
    return float(best_point(sw, target_retention)["tau"])


def apply_tau(df: pd.DataFrame, p: np.ndarray, tau: float, overhead_usd: float = 0.0) -> dict:
    out = _metrics_for_mask(df, np.asarray(p) >= tau, overhead_usd)
    out["tau"] = tau
    return out


def oracle_scores(df: pd.DataFrame) -> np.ndarray:
    """Upper bound: cheap exactly where cheap is at least as good as strong."""
    return (df["correct_cheap"].to_numpy() >= df["correct_strong"].to_numpy()).astype(float)


def length_scores(df: pd.DataFrame, ref_max_chars: float | None = None) -> np.ndarray:
    """Length heuristic: shorter prompt => more likely cheap. ref_max_chars should come from TRAIN."""
    lens = df["query_text"].str.len().to_numpy(dtype=float)
    mx = ref_max_chars or max(lens.max(), 1.0)
    return 1.0 - np.clip(lens / mx, 0, 1)


# --------------------------------------------------------------------------------------------
# Uncertainty: bootstrap
# --------------------------------------------------------------------------------------------
def _per_query_arrays(df: pd.DataFrame, cheap_mask: np.ndarray, overhead_usd: float = 0.0):
    acc_r = np.where(cheap_mask, df["correct_cheap"], df["correct_strong"]).astype(float)
    cost_r = np.where(cheap_mask, df["cost_cheap"], df["cost_strong"]).astype(float) + overhead_usd
    return acc_r, cost_r, df["correct_strong"].to_numpy(float), df["cost_strong"].to_numpy(float)


def bootstrap_ci(df: pd.DataFrame, cheap_mask: np.ndarray, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05, overhead_usd: float = 0.0) -> dict:
    """Percentile CIs for savings & retention when the (frozen) routing decision is applied to
    resampled test queries."""
    rng = np.random.default_rng(seed)
    n = len(df)
    acc_r, cost_r, acc_s, cost_s = _per_query_arrays(df, cheap_mask, overhead_usd)
    idx = rng.integers(0, n, size=(n_boot, n))
    ret = acc_r[idx].mean(1) / np.clip(acc_s[idx].mean(1), 1e-12, None)
    sav = 1 - cost_r[idx].mean(1) / cost_s[idx].mean(1)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "savings_ci": (float(np.percentile(sav, lo)), float(np.percentile(sav, hi))),
        "retention_ci": (float(np.percentile(ret, lo)), float(np.percentile(ret, hi))),
        "retention_samples": ret,
        "savings_samples": sav,
    }


def paired_bootstrap_diff(df: pd.DataFrame, mask_a: np.ndarray, mask_b: np.ndarray,
                          n_boot: int = 2000, seed: int = 0, overhead_usd: float = 0.0) -> dict:
    """Is router A's savings better than B's, accounting for sampling noise? (paired resampling)"""
    rng = np.random.default_rng(seed)
    n = len(df)
    _, cost_a, _, cost_s = _per_query_arrays(df, mask_a, overhead_usd)
    _, cost_b, _, _ = _per_query_arrays(df, mask_b, overhead_usd)
    idx = rng.integers(0, n, size=(n_boot, n))
    sa = 1 - cost_a[idx].mean(1) / cost_s[idx].mean(1)
    sb = 1 - cost_b[idx].mean(1) / cost_s[idx].mean(1)
    d = sa - sb
    return {"diff_mean": float(d.mean()), "ci": (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
            "p_a_better": float((d > 0).mean())}


# --------------------------------------------------------------------------------------------
# Breakdowns
# --------------------------------------------------------------------------------------------
def per_domain(df: pd.DataFrame, p: np.ndarray, tau: float, overhead_usd: float = 0.0) -> pd.DataFrame:
    p = np.asarray(p)
    doms = df["domain"].to_numpy()
    rows = []
    for dom in sorted(set(doms)):
        pos = np.where(doms == dom)[0]
        g = df.iloc[pos]
        m = _metrics_for_mask(g, p[pos] >= tau, overhead_usd)
        m.update(domain=dom, n=len(g), strong_acc=float(g["correct_strong"].mean()),
                 cheap_acc=float(g["correct_cheap"].mean()))
        rows.append(m)
    return pd.DataFrame(rows).set_index("domain")


def cascade_metrics(df: pd.DataFrame, p: np.ndarray, tau: float, overhead_ms: float = 0.0,
                    overhead_usd: float = 0.0) -> dict:
    """Cheap-first with an automatic safety net: if the cheap answer is truncated or empty,
    escalate to the strong model (paying for both). Latency is the sum of calls made.
    No logprob/self-confidence signal is used, only checks available for free at serving time."""
    p = np.asarray(p)
    cheap_mask = p >= tau
    bad = (df["finish_cheap"].fillna("") == "length") | (df["resp_cheap"].fillna("").str.strip() == "")
    escalate = cheap_mask & bad.to_numpy()
    acc = np.where(cheap_mask & ~escalate, df["correct_cheap"], df["correct_strong"])
    cost = np.where(cheap_mask, df["cost_cheap"], df["cost_strong"]) + np.where(escalate, df["cost_strong"], 0) + overhead_usd
    lat = np.where(cheap_mask, df["lat_cheap"], df["lat_strong"]) + np.where(escalate, df["lat_strong"], 0) + overhead_ms
    s_acc, s_cost = df["correct_strong"].mean(), df["cost_strong"].mean()
    return {"pct_cheap_final": float((cheap_mask & ~escalate).mean()), "pct_escalated": float(escalate.mean()),
            "accuracy": float(acc.mean()), "retention": float(acc.mean() / s_acc),
            "savings": float(1 - cost.mean() / s_cost), "avg_latency_ms": float(lat.mean())}


def strategy_latency(df: pd.DataFrame, cheap_mask: np.ndarray, overhead_ms: float) -> float:
    """Mean end-to-end latency (ms) from measured per-model latencies + routing overhead."""
    lat = np.where(cheap_mask, df["lat_cheap"], df["lat_strong"])
    return float(lat.mean() + overhead_ms)


# --------------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------------
def router_overhead_usd(overhead_ms: float, vcpu_usd_per_hour: float = 0.04) -> float:
    """Approximate per-request compute cost of routing: CPU-seconds x cloud vCPU price."""
    return overhead_ms / 3_600_000.0 * vcpu_usd_per_hour


def calibration_stats(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece, curve = 0.0, []
    for b in range(bins):
        m = ids == b
        if m.any():
            conf, acc = p[m].mean(), y[m].mean()
            ece += m.mean() * abs(conf - acc)
            curve.append((float(conf), float(acc), int(m.sum())))
    return {"ece": float(ece), "brier": float(np.mean((p - y) ** 2)), "curve": curve}


# --------------------------------------------------------------------------------------------
# Plots (matplotlib imported lazily so the gateway never needs it)
# --------------------------------------------------------------------------------------------
def plot_pareto(curves: Mapping[str, pd.DataFrame], df: pd.DataFrame, path: str | Path,
                points: Mapping[str, tuple[float, float]] | None = None, title: str = "Cost vs. quality (test split)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s_cost = df["cost_strong"].mean()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    styles = {"oracle": ("k", ":"), "random": ("gray", "--"), "length heuristic": ("tab:orange", "-")}
    for name, c in curves.items():
        col, ls = styles.get(name, (None, "-"))
        c = c.sort_values("avg_cost")
        ax.plot(100 * c["avg_cost"] / s_cost, 100 * c["accuracy"], ls, color=col, label=name,
                lw=2.2 if name.startswith("ML") else 1.4)
    ax.scatter([100], [100 * df["correct_strong"].mean()], c="red", marker="*", s=140, zorder=5, label="always strong")
    ax.scatter([100 * df["cost_cheap"].mean() / s_cost], [100 * df["correct_cheap"].mean()], c="green",
               marker="s", s=70, zorder=5, label="always cheap")
    for name, (x, y) in (points or {}).items():
        ax.scatter([100 * x], [100 * y], marker="D", s=60, zorder=6, edgecolor="k", label=name)
    ax.set_xlabel("Average cost (% of always-strong)")
    ax.set_ylabel("Accuracy / pass rate (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(stats: Mapping[str, dict], path: str | Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, s in stats.items():
        if s["curve"]:
            x, y, _ = zip(*s["curve"])
            ax.plot(x, y, "o-", label=f"{name} (ECE {s['ece']:.3f})")
    ax.set_xlabel("Predicted P(cheap ok)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability (test)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_domains(table: pd.DataFrame, path: str | Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(table))
    ax.bar(x - 0.2, 100 * table["pct_cheap"], 0.4, label="% routed cheap")
    ax.bar(x + 0.2, 100 * table["retention"], 0.4, label="quality retention %")
    ax.axhline(95, color="red", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=30, ha="right")
    ax.set_title("Per-domain behaviour at the deployed threshold (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
