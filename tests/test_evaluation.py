import numpy as np
import pandas as pd

from router import evaluation as ev


def make_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    cc = (rng.random(n) < 0.6).astype(float)
    cs = (rng.random(n) < 0.9).astype(float)
    return pd.DataFrame({"query_id": range(n), "domain": rng.choice(["a", "b"], n), "split": "test",
                         "query_text": ["x" * int(l) for l in rng.integers(5, 200, n)],
                         "correct_cheap": cc, "correct_strong": cs, "cost_cheap": 1.0, "cost_strong": 10.0,
                         "lat_cheap": 100.0, "lat_strong": 900.0, "finish_cheap": "stop", "resp_cheap": "r"})


def test_sweep_endpoints():
    df = make_df()
    p = np.random.default_rng(1).random(len(df))
    sw = ev.sweep(df, p)
    last = sw.iloc[-1]                       # tau = inf => always strong
    assert last["pct_cheap"] == 0 and abs(last["savings"]) < 1e-12 and abs(last["retention"] - 1) < 1e-12
    first = sw.sort_values("tau").iloc[0]    # lowest tau => always cheap
    assert first["pct_cheap"] == 1.0
    assert abs(first["savings"] - 0.9) < 1e-9


def test_oracle_dominates_and_never_loses_quality():
    df = make_df()
    o = ev.oracle_scores(df)
    m = ev.apply_tau(df, o, 1.0)
    assert m["retention"] >= 1.0 and m["savings"] > 0.2
    rand = ev.random_curve(df)
    bp_o = ev.best_point(ev.sweep(df, o), 0.95)["savings"]
    bp_r = ev.best_point(rand, 0.95)["savings"]
    assert bp_o > bp_r


def test_random_curve_matches_simulation():
    df = make_df(4000, 3)
    curve = ev.random_curve(df)
    rng = np.random.default_rng(5)
    f = 0.3
    sims = []
    for _ in range(200):
        mask = rng.random(len(df)) < f
        sims.append(ev.apply_tau(df, mask.astype(float), 0.5)["savings"])
    expected = curve.iloc[30]["savings"]
    assert abs(np.mean(sims) - expected) < 0.01


def test_tune_on_val_applied_to_test_and_unreachable_target():
    df = make_df()
    p = np.random.default_rng(2).random(len(df))
    tau = ev.tune_tau(df, p, 0.9)
    assert ev.apply_tau(df, p, tau)["retention"] >= 0.9 - 1e-9
    assert ev.tune_tau(df, p, 5.0) == float("inf")      # unreachable => always strong


def test_bootstrap_ci_contains_point_estimate():
    df = make_df()
    p = ev.length_scores(df)
    mask = p >= 0.5
    ci = ev.bootstrap_ci(df, mask, n_boot=400)
    pt = ev.apply_tau(df, p, 0.5)
    assert ci["savings_ci"][0] <= pt["savings"] <= ci["savings_ci"][1]
    assert ci["retention_ci"][0] <= pt["retention"] <= ci["retention_ci"][1]


def test_paired_bootstrap_identical_routers_have_zero_diff():
    df = make_df()
    m = ev.length_scores(df) >= 0.5
    r = ev.paired_bootstrap_diff(df, m, m, n_boot=200)
    assert r["diff_mean"] == 0 and r["ci"] == (0.0, 0.0)


def test_cascade_escalates_truncated_answers():
    df = make_df()
    df.loc[:49, "finish_cheap"] = "length"
    p = np.ones(len(df))
    m = ev.cascade_metrics(df, p, 0.5)
    assert abs(m["pct_escalated"] - 50 / len(df)) < 1e-9
    assert m["savings"] < 0.9                      # paid for both on escalations


def test_per_domain_and_calibration():
    df = make_df()
    p = np.random.default_rng(0).random(len(df))
    t = ev.per_domain(df, p, 0.5)
    assert set(t.index) == {"a", "b"} and t["n"].sum() == len(df)
    y = (np.random.default_rng(1).random(5000) < 0.7).astype(int)
    assert ev.calibration_stats(y, np.full(5000, 0.7))["ece"] < 0.03
    assert ev.calibration_stats(y, np.full(5000, 0.1))["ece"] > 0.5
