"""The browser scores with the dumped trees and isotonic table. This is the reference the JS port copies."""
import bisect
import math

import numpy as np
import pandas as pd
import pytest

from sentinel.export.model import event_effects
from sentinel.models.fusion import Fusion


def leaf(node, x):
    while "leaf_value" not in node:
        assert node["decision_type"] == "<="
        v = x[node["split_feature"]]
        if math.isnan(v) and node["missing_type"] != "NaN":
            v = 0.0
        if (node["missing_type"] == "NaN" and math.isnan(v)) or (node["missing_type"] == "Zero" and abs(v) <= 1e-35):
            left = node["default_left"]
        else:
            left = v <= node["threshold"]
        node = node["left_child"] if left else node["right_child"]
    return node["leaf_value"]


def margin(trees, x):
    return sum(leaf(t["tree_structure"], x) for t in trees)


def isotonic(p, xs, ys):
    """sklearn IsotonicRegression(out_of_bounds="clip").predict: linear interpolation, clipped at the ends."""
    if p <= xs[0]:
        return ys[0]
    if p >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_left(xs, p)
    return ys[i - 1] + (ys[i] - ys[i - 1]) * (p - xs[i - 1]) / (xs[i] - xs[i - 1])


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    feats = ["robust_z", "payee_new", "since_email_change", "kw_n30_flag_1", "amt_log", "kw_mitigating_90d"]

    def make(n):
        X = pd.DataFrame({"robust_z": rng.normal(0, 3, n), "payee_new": rng.integers(0, 2, n).astype(float),
                          "since_email_change": rng.uniform(0, 400, n), "kw_n30_flag_1": rng.poisson(0.3, n).astype(float),
                          "amt_log": np.where(rng.random(n) < 0.3, 0.0, rng.normal(8, 2, n)), "kw_mitigating_90d": rng.poisson(0.2, n).astype(float)})
        X.loc[rng.random(n) < 0.05, "robust_z"] = np.nan  # exercise missing-value routing
        z = 0.6 * X["robust_z"].fillna(0) + 1.5 * X["payee_new"] - X["since_email_change"] / 100 + X["kw_n30_flag_1"] - 3
        return X, (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    (X, y), (Xv, yv), (Xt, _) = make(4000), make(2000), make(1500)
    return Fusion(feats).fit(X, y, Xv, yv), Xt


def test_tree_walk_matches_booster(fitted):
    m, X = fitted
    A = X[m.features].to_numpy(np.float32)
    trees = m.model.booster_.dump_model()["tree_info"]
    ours = np.array([margin(trees, [float(v) for v in row]) for row in A])
    assert len(ours) >= 1000
    assert np.abs(ours - m.model.booster_.predict(A, raw_score=True)).max() < 1e-9


def test_sigmoid_then_isotonic_matches_calibrated(fitted):
    m, X = fitted
    trees = m.model.booster_.dump_model()["tree_info"]
    xs, ys = m.iso.X_thresholds_.tolist(), m.iso.y_thresholds_.tolist()
    p = np.array([1 / (1 + math.exp(-margin(trees, [float(v) for v in row]))) for row in X[m.features].to_numpy(np.float32)])
    assert np.abs(p - m.raw(X)).max() < 1e-9
    cal = np.array([isotonic(v, xs, ys) for v in p])
    assert np.abs(cal - m.calibrated(m.raw(X))).max() < 1e-9


def test_event_effects_only_name_model_features():
    feats = ["robust_z", "amt_log", "cnt30_email_change", "since_email_change", "kw_n30_flag_2", "kw_notes_90d"]
    E = event_effects(feats)
    assert {e["feature"] for v in E.values() for e in v} <= set(feats)
    assert {"feature": "since_email_change", "op": "set", "value": 0} in E["email_change"]
    assert {"feature": "kw_n30_flag_2", "op": "add", "value": 1} in E["note_flag_2"]
