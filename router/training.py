"""Train, compare and select routers. Selection uses the VALIDATION split only.

Produces
  results/model_selection.csv   one row per candidate (features x model), val metrics
  results/predictions.csv       p(cheap ok) for every val/test query, per candidate
  results/router_config.json    the selected router + deployment threshold
  gateway/models/router.joblib  the artifact the gateway loads
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import db as dbm
from .config import ARTIFACT_PATH, RESULTS_DIR, SEED
from .evaluation import calibration_stats, cheap_ok_label, load_frame, sweep, best_point
from .features import build_features, make_embedder, rule_matrix
from .inference import save_artifact
from .models import SimilarityWeightedRouter
from .perturb import ALL_PERTURBATIONS


def make_model(kind: str):
    if kind == "lr":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))])
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=200,
                                              class_weight="balanced", random_state=SEED)
    if kind == "knn":
        return KNeighborsClassifier(n_neighbors=15, weights="distance", metric="cosine", algorithm="brute")
    if kind == "sw":
        return SimilarityWeightedRouter(k=50, temperature=0.05)
    if kind == "mlp":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, early_stopping=True,
                                               max_iter=400, random_state=SEED))])
    raise ValueError(kind)


# (name, model kind, feature mode). kNN uses cosine on raw embeddings so it is emb-only.
CANDIDATES = [
    ("lr_emb", "lr", "emb"), ("lr_rules", "lr", "rules"), ("lr_both", "lr", "both"),
    ("hgb_emb", "hgb", "emb"), ("hgb_rules", "hgb", "rules"), ("hgb_both", "hgb", "both"),
    ("knn_emb", "knn", "emb"), ("sw_emb", "sw", "emb"), ("mlp_both", "mlp", "both"),
]


def _maybe_calibrate(model, y: np.ndarray, calibrate: bool, groups: np.ndarray | None = None):
    if calibrate and np.bincount(y, minlength=2).min() >= 6:
        if groups is None:
            groups = np.arange(len(y))
        sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        cv = list(sgkf.split(np.zeros(len(y)), y, groups))
        return CalibratedClassifierCV(model, method="sigmoid", cv=cv)
    return model


def train(db_path=None, label_mode: str = "as_good", target: float = 0.95, safety: float = 0.01,
          calibrate: bool = True, out_dir: Path | None = None, artifact_path: Path | None = None,
          embedder=None, only: list[str] | None = None, verbose: bool = True, augment: int = 0) -> dict:
    out_dir = Path(out_dir or RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = dbm.connect(db_path)
    df = load_frame(conn)
    conn.close()
    if df.empty:
        raise RuntimeError("No queries with both models evaluated. Run scripts.build_benchmark first.")
    cheap, strong = df.attrs["cheap"], df.attrs["strong"]
    df = df.reset_index(drop=True)
    y_all = cheap_ok_label(df, label_mode)
    split = df["split"].to_numpy()
    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    if min(len(tr), len(va), len(te)) == 0:
        raise RuntimeError(f"Empty split: train={len(tr)} val={len(va)} test={len(te)}")
    if len(np.unique(y_all[tr])) < 2:
        raise RuntimeError("Training labels have a single class; the cheap model is always/never good enough.")

    texts = df["query_text"].tolist()
    embedder = embedder or make_embedder()
    t0 = time.time()
    emb = embedder.encode(texts)
    if verbose:
        print(f"Embedded {len(texts)} queries in {time.time() - t0:.1f}s | train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    X = {m: build_features(texts, None, m, emb=emb) for m in ("emb", "rules", "both")}

    # Optional robustness augmentation: `augment` perturbed copies of every TRAIN prompt (benign rewrites + injection
    # strings), labels unchanged, so the router learns that padding/politeness/"this is easy" text carries no signal.
    # Validation and test stay untouched (and unperturbed) so selection and reporting remain honest.
    X_tr, y_tr = {m: X[m][tr] for m in X}, y_all[tr]
    if augment > 0:
        import random
        rng, fns = random.Random(SEED), list(ALL_PERTURBATIONS.values())
        aug_texts = [rng.choice(fns)(texts[i]) for _ in range(augment) for i in tr]
        aug_emb = embedder.encode(aug_texts)
        for m in X_tr:
            X_tr[m] = np.vstack([X_tr[m], build_features(aug_texts, None, m, emb=aug_emb)])
        y_tr = np.concatenate([y_tr] + [y_all[tr]] * augment)
        if verbose:
            print(f"Augmentation: +{len(aug_texts)} perturbed training rows (x{augment})")

    tgt = target + safety
    groups = np.tile(np.arange(len(tr)), augment + 1)
    rows, preds, fitted = [], df[["query_id", "domain", "split"]].iloc[np.r_[va, te]].copy(), {}
    preds["y_cheap_ok"] = y_all[np.r_[va, te]]
    for name, kind, mode in CANDIDATES:
        if only and name not in only:
            continue
        model = _maybe_calibrate(make_model(kind), y_tr, calibrate, groups=groups)
        model.fit(X_tr[mode], y_tr)
        p_va = model.predict_proba(X[mode][va])[:, 1]
        p_te = model.predict_proba(X[mode][te])[:, 1]
        preds[f"p_{name}"] = np.r_[p_va, p_te]
        sw = sweep(df.iloc[va], p_va)
        bp = best_point(sw, tgt)
        cal = calibration_stats(y_all[va], p_va)
        auc = roc_auc_score(y_all[va], p_va) if len(np.unique(y_all[va])) > 1 else np.nan
        rows.append({"candidate": name, "model": kind, "features": mode, "val_auc": auc, "val_ece": cal["ece"],
                     "val_brier": cal["brier"], "val_savings_at_target": bp["savings"],
                     "val_retention": bp["retention"], "val_pct_cheap": bp["pct_cheap"], "tau": bp["tau"]})
        fitted[name] = (model, mode, float(bp["tau"]))
        if verbose:
            print(f"  {name:10s} AUC={auc:.3f} ECE={cal['ece']:.3f} val-savings@{tgt:.0%}={bp['savings']:+.1%}")

    sel = pd.DataFrame(rows).sort_values(["val_savings_at_target", "val_auc"], ascending=False)
    sel.to_csv(out_dir / "model_selection.csv", index=False)
    preds.to_csv(out_dir / "predictions.csv", index=False)

    best = sel.iloc[0]
    model, mode, tau = fitted[best["candidate"]]
    train_len_max = float(df.iloc[tr]["query_text"].str.len().max())
    meta = dict(name=best["candidate"], feature_mode=mode, threshold=tau, target_retention=target, safety=safety,
                label_mode=label_mode, embedder=embedder.spec, cheap_model=cheap, strong_model=strong,
                train_cheap_ok_rate=float(y_all[tr].mean()), train_length_ref_max=train_len_max,
                n_train=int(len(tr)), calibrated=calibrate, augment=augment, trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                val_metrics={k: (None if pd.isna(v) else float(v)) for k, v in best.items()
                             if isinstance(v, (int, float, np.floating)) and k != "tau"})
    art_path = save_artifact(artifact_path or ARTIFACT_PATH, model=model, **meta)
    cfg = dict(meta)
    cfg["threshold"] = None if np.isinf(tau) else tau
    cfg["threshold_note"] = "null => target unreachable on validation; router sends everything to the strong model"
    (out_dir / "router_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    if verbose:
        print(f"\nSelected (by VALIDATION): {best['candidate']}  tau={tau:.4f}\nArtifact: {art_path}")
    return {"selection": sel, "predictions": preds, "meta": meta, "artifact": art_path}