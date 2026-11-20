import json

import joblib
import pytest

from router import db as dbm
from router import evaluation as ev
from router.config import CHEAP, STRONG
from router.inference import load_router
from router.training import train


def test_splits_are_disjoint_stratified_and_stable(synth_db):
    conn = dbm.connect(synth_db)
    rows = conn.execute("SELECT domain, split, COUNT(*) FROM queries GROUP BY domain, split").fetchall()
    splits = {s for _, s, _ in rows}
    assert splits == {"train", "val", "test"}
    before = dict(conn.execute("SELECT query_id, split FROM queries").fetchall())
    dbm.assign_splits(conn)                      # re-running must not change anything
    assert before == dict(conn.execute("SELECT query_id, split FROM queries").fetchall())
    for dom in {r[0] for r in rows}:
        n = {s: c for d, s, c in rows if d == dom}
        assert n["train"] > n["val"] and n["train"] > n["test"]


@pytest.fixture(scope="module")
def trained(synth_db, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    res = train(synth_db, out_dir=out, artifact_path=out / "router.joblib", verbose=False,
                only=["lr_rules", "hgb_both", "knn_emb"])
    return res, out


def test_training_artifacts_and_selection_uses_val_only(trained):
    res, out = trained
    assert (out / "router.joblib").exists() and (out / "predictions.csv").exists()
    cfg = json.loads((out / "router_config.json").read_text())
    assert cfg["name"] in {"lr_rules", "hgb_both", "knn_emb"}
    preds = res["predictions"]
    assert set(preds["split"]) == {"val", "test"}          # no train rows leak into predictions
    sel = res["selection"]
    assert sel.iloc[0]["val_savings_at_target"] == sel["val_savings_at_target"].max()


def test_artifact_reloads_and_predicts_valid_probabilities(trained):
    _, out = trained
    r = load_router(out / "router.joblib")
    p = r.p_cheap(["What is 2+2?", "def f(x): return x", "Write a long essay about philosophy " * 20])
    assert p.shape == (3,) and ((p >= 0) & (p <= 1)).all()
    assert r.p_cheap_one("hello") == pytest.approx(r.p_cheap(["hello"])[0])


def test_serving_and_training_features_agree(trained, synth_db):
    """Skew guard: the artifact must reproduce the stored validation predictions."""
    res, out = trained
    art = joblib.load(out / "router.joblib")
    r = load_router(out / "router.joblib")
    preds = res["predictions"]
    conn = dbm.connect(synth_db)
    df = ev.load_frame(conn).set_index("query_id")
    sub = preds[preds.split == "val"].head(25)
    p_now = r.p_cheap(df.loc[sub.query_id, "query_text"].tolist())
    assert p_now == pytest.approx(sub[f"p_{art['name']}"].to_numpy(), abs=1e-5)


def test_label_modes_differ(synth_db):
    conn = dbm.connect(synth_db)
    df = ev.load_frame(conn)
    assert ev.cheap_ok_label(df, "as_good").sum() >= ev.cheap_ok_label(df, "cheap_correct").sum()
