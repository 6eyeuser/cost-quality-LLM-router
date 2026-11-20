import numpy as np
import pandas as pd
import pytest

from router import db as dbm
from router import evaluation as ev
from router import robustness as rb
from router.features import HashEmbedder
from router.synthetic import DOMAINS


def toy_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"query_id": range(n), "domain": "a", "split": "test",
                         "query_text": [f"question number {i} about things" for i in range(n)],
                         "correct_cheap": (rng.random(n) < 0.4).astype(float), "correct_strong": 1.0,
                         "cost_cheap": 1.0, "cost_strong": 10.0})


def test_perturbations_are_deterministic_and_change_text():
    t = "Please solve this very long problem about trains and apples."
    for name, fn in {**rb.BENIGN, **rb.INJECTION_ATTACKS}.items():
        assert fn(t) == fn(t), name
        assert fn(t) != t or name == "lowercase", name
    assert rb.BENIGN["lowercase"]("ABC") == "abc"


def test_gullible_router_is_caught_by_injection_probe():
    df = toy_df()
    gullible = lambda texts: np.array([0.9 if ("trivial" in t or "easy" in t or "simple" in t) else 0.1 for t in texts])
    rep = rb.perturbation_report(df, gullible, tau=0.5).set_index("perturbation")
    inj = rep.loc["inject_1"]
    assert inj["to_cheap_rate"] == 1.0
    assert inj["harmful_flip_rate"] == pytest.approx((df.correct_cheap < df.correct_strong).mean())
    assert inj["retention_after"] < 0.7                                    # quality collapses -> visible in the report
    assert rep.loc["lowercase", "flip_rate"] == 0.0                        # benign rewrite doesn't matter to this router


def test_robust_router_has_zero_harmful_flips():
    df = toy_df()
    robust = lambda texts: np.full(len(texts), 0.1)
    rep = rb.perturbation_report(df, robust, tau=0.5)
    assert (rep["harmful_flip_rate"] == 0).all() and (rep["flip_rate"] == 0).all()


def test_multilingual_probe_shape():
    out = rb.multilingual_probe(lambda texts: np.full(len(texts), 0.3), tau=0.5)
    assert len(out) == len(rb.MULTILINGUAL_PAIRS) * 4 and out["same_decision"].all() and (out["abs_dp"] == 0).all()


def test_leave_one_domain_out_runs_on_synthetic_data(synth_db):
    conn = dbm.connect(synth_db)
    df = ev.load_frame(conn).reset_index(drop=True)
    emb = HashEmbedder(64).encode(df["query_text"].tolist())
    out = rb.leave_one_domain_out(df, emb, candidate="hgb_rules")
    assert set(out["held_out_domain"]) == set(DOMAINS)
    assert np.isfinite(out[["ood_savings", "ood_retention", "indomain_savings"]].to_numpy()).all()
    assert (out["n_ood"] > 50).all()
