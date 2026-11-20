import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from router.models import SimilarityWeightedRouter


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(+1, 0.5, (n // 2, 16)), rng.normal(-1, 0.5, (n // 2, 16))])
    y = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
    return X, y


def test_sw_router_separates_clusters_and_returns_valid_probabilities():
    X, y = _data()
    m = SimilarityWeightedRouter(k=20).fit(X, y)
    Xt, yt = _data(100, 1)
    p = m.predict_proba(Xt)
    assert p.shape == (100, 2) and np.allclose(p.sum(1), 1) and ((p >= 0) & (p <= 1)).all()
    assert (m.predict(Xt) == yt).mean() > 0.95


def test_sw_router_far_from_data_falls_back_toward_prior():
    X, y = _data()
    y = np.where(np.arange(len(y)) < 40, 1, 0)          # base rate 10%
    m = SimilarityWeightedRouter(k=10, temperature=5.0, prior_strength=50).fit(X, y)
    p = m.predict_proba(X[:5])[:, 1]
    assert np.all(np.abs(p - 0.1) < 0.1)


def test_sw_router_is_sklearn_compatible_and_picklable(tmp_path):
    X, y = _data()
    assert clone(SimilarityWeightedRouter(k=7)).k == 7
    cal = CalibratedClassifierCV(SimilarityWeightedRouter(k=15), method="sigmoid", cv=3).fit(X, y)
    joblib.dump(cal, tmp_path / "m.joblib")
    again = joblib.load(tmp_path / "m.joblib")
    assert np.allclose(cal.predict_proba(X[:20]), again.predict_proba(X[:20]))
