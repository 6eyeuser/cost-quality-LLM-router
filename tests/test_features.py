import numpy as np

from router.features import (RULE_FEATURE_NAMES, CachedEmbedder, HashEmbedder, build_features, rule_features,
                             rule_matrix)


def test_rule_feature_shape_and_flags():
    f = rule_features("def foo(x):\n    return x + 1")
    assert f.shape == (len(RULE_FEATURE_NAMES),)
    idx = RULE_FEATURE_NAMES.index
    assert f[idx("has_py_signature")] == 1
    assert f[idx("code_kw_count")] >= 2
    assert rule_features("What is the capital of France?")[idx("starts_wh")] == 1
    assert rule_features("A\n")[idx("has_py_signature")] == 0
    assert rule_features("Use math.sqrt here")[idx("has_math_module")] == 1
    assert np.isfinite(rule_features("")).all()


def test_build_features_modes():
    texts = ["hello world", "def f(): pass", "2 + 2 = ?"]
    emb = HashEmbedder(64)
    assert build_features(texts, emb, "emb").shape == (3, 64)
    assert build_features(texts, emb, "rules").shape == (3, len(RULE_FEATURE_NAMES))
    assert build_features(texts, emb, "both").shape == (3, 64 + len(RULE_FEATURE_NAMES))


def test_cache_returns_same_vectors_and_counts_hits():
    inner = HashEmbedder(32)
    c = CachedEmbedder(inner)
    a = c.encode(["x y z", "foo"])
    b = c.encode(["foo", "x y z", "new"])
    assert np.allclose(a[0], b[1]) and np.allclose(a[1], b[0])
    assert c.hits == 2 and c.misses == 3
    assert rule_matrix([]).shape[0] == 0
