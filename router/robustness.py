"""Robustness checks for a trained router.

1. Perturbation stability  - benign rewrites (case, typos, padding, politeness) should not flip decisions much.
2. Prompt-injection probes - text that tries to talk the router into the cheap model must not do so.
                             Reported as *harmful flips*: strong->cheap flips on queries where cheap actually fails.
3. Multilingual probe      - a tiny hand-written parallel set: same question, different language.
4. Leave-one-domain-out    - train without a domain, test on it: the honest measure of out-of-distribution behaviour.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import evaluation as ev
from .features import build_features
from .perturb import BENIGN, FILLER, INJECTION_ATTACKS, INJECTIONS  # noqa: F401  (re-exported)
from .training import CANDIDATES, _maybe_calibrate, make_model

# ---------------------------------------------------------------------------------------------
# 1 + 2. Perturbations
# ---------------------------------------------------------------------------------------------
def perturbation_report(df: pd.DataFrame, predict: Callable[[list[str]], np.ndarray], tau: float,
                        attacks: dict[str, Callable[[str], str]] | None = None) -> pd.DataFrame:
    """For each perturbation: how often does the routing decision change, and is the change harmful?

    harmful_flip_rate = share of queries that moved strong->cheap although the cheap model is WORSE than strong
    on that query (ground truth from the benchmark; assumes the perturbation does not change the right answer).
    """
    attacks = attacks or {**BENIGN, **INJECTION_ATTACKS}
    texts = df["query_text"].tolist()
    p0 = np.asarray(predict(texts))
    base_cheap = p0 >= tau
    worse = (df["correct_cheap"].to_numpy() < df["correct_strong"].to_numpy())
    rows = []
    for name, fn in attacks.items():
        p1 = np.asarray(predict([fn(t) for t in texts]))
        cheap = p1 >= tau
        to_cheap, to_strong = (~base_cheap & cheap), (base_cheap & ~cheap)
        m1 = ev.apply_tau(df, p1, tau)
        rows.append({"perturbation": name, "kind": "injection" if name.startswith("inject") else "benign",
                     "mean_abs_dp": float(np.abs(p1 - p0).mean()), "mean_dp": float((p1 - p0).mean()),
                     "flip_rate": float((cheap != base_cheap).mean()), "to_cheap_rate": float(to_cheap.mean()),
                     "to_strong_rate": float(to_strong.mean()),
                     "harmful_flip_rate": float((to_cheap & worse).mean()),
                     "retention_after": m1["retention"], "savings_after": m1["savings"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# 3. Multilingual probe (tiny, hand-written; stability check, NOT a benchmark)
# ---------------------------------------------------------------------------------------------
MULTILINGUAL_PAIRS: list[tuple[str, dict[str, str]]] = [
    ("What is the capital of France?", {"es": "¿Cuál es la capital de Francia?", "fr": "Quelle est la capitale de la France ?",
                                       "zh": "法国的首都是哪里？", "ar": "ما هي عاصمة فرنسا؟"}),
    ("Write a short poem about the sea.", {"es": "Escribe un poema corto sobre el mar.", "fr": "Écris un court poème sur la mer.",
                                           "zh": "写一首关于大海的短诗。", "ar": "اكتب قصيدة قصيرة عن البحر."}),
    ("A train travels 60 miles per hour for 3 hours. How far does it go?",
     {"es": "Un tren viaja a 60 millas por hora durante 3 horas. ¿Qué distancia recorre?",
      "fr": "Un train roule à 60 miles par heure pendant 3 heures. Quelle distance parcourt-il ?",
      "zh": "一列火车以每小时60英里的速度行驶3小时，它走了多远？", "ar": "يسير قطار بسرعة 60 ميلاً في الساعة لمدة 3 ساعات. ما المسافة التي يقطعها؟"}),
    ("Prove that the square root of 2 is irrational.", {"es": "Demuestra que la raíz cuadrada de 2 es irracional.",
                                                        "fr": "Prouve que la racine carrée de 2 est irrationnelle.",
                                                        "zh": "证明2的平方根是无理数。", "ar": "أثبت أن الجذر التربيعي للعدد 2 عدد غير نسبي."}),
]


def multilingual_probe(predict: Callable[[list[str]], np.ndarray], tau: float) -> pd.DataFrame:
    rows = []
    for en, variants in MULTILINGUAL_PAIRS:
        p_en = float(predict([en])[0])
        for lang, text in variants.items():
            p = float(predict([text])[0])
            rows.append({"english": en[:40], "lang": lang, "p_en": p_en, "p_lang": p, "abs_dp": abs(p - p_en),
                         "same_decision": (p >= tau) == (p_en >= tau)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# 4. Leave-one-domain-out
# ---------------------------------------------------------------------------------------------
def leave_one_domain_out(df: pd.DataFrame, emb: np.ndarray, candidate: str = "hgb_rules", target: float = 0.95,
                         safety: float = 0.01, label_mode: str = "as_good", calibrate: bool = True) -> pd.DataFrame:
    """Train WITHOUT domain d (train split of the others), tune tau on the others' val split, then score ALL of d.
    Compare with the in-domain protocol (trained on every domain, tau tuned on val, scored on d's test split)."""
    kind, mode = next((k, m) for n, k, m in CANDIDATES if n == candidate)
    df = df.reset_index(drop=True)
    y = ev.cheap_ok_label(df, label_mode)
    X = build_features(df["query_text"].tolist(), None, mode, emb=emb)
    split, dom = df["split"].to_numpy(), df["domain"].to_numpy()
    tgt = target + safety

    def fit(train_idx):
        m = _maybe_calibrate(make_model(kind), y[train_idx], calibrate)
        return m.fit(X[train_idx], y[train_idx])

    full = fit(np.where(split == "train")[0])
    p_full = full.predict_proba(X)[:, 1]
    rows = []
    for d in sorted(set(dom)):
        other = dom != d
        tr, va = np.where(other & (split == "train"))[0], np.where(other & (split == "val"))[0]
        if len(tr) == 0 or len(va) == 0 or len(np.unique(y[tr])) < 2:
            continue
        m = fit(tr)
        tau = ev.tune_tau(df.iloc[va], m.predict_proba(X[va])[:, 1], tgt)
        idx_all = np.where(dom == d)[0]
        ood = ev.apply_tau(df.iloc[idx_all], m.predict_proba(X[idx_all])[:, 1], tau)
        va_f, te_f = np.where(split == "val")[0], np.where((split == "test") & (dom == d))[0]
        tau_f = ev.tune_tau(df.iloc[va_f], p_full[va_f], tgt)
        idm = ev.apply_tau(df.iloc[te_f], p_full[te_f], tau_f) if len(te_f) else {"savings": np.nan, "retention": np.nan, "pct_cheap": np.nan}
        rows.append({"held_out_domain": d, "n_ood": len(idx_all), "ood_savings": ood["savings"], "ood_retention": ood["retention"],
                     "ood_pct_cheap": ood["pct_cheap"], "ood_meets_target": bool(ood["retention"] >= target),
                     "indomain_savings": idm["savings"], "indomain_retention": idm["retention"],
                     "indomain_pct_cheap": idm["pct_cheap"]})
    return pd.DataFrame(rows)
