"""Fusion objectives and tuning selection (PLAN item 4)."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from sentinel.eval.tune import grid, select
from sentinel.models.fusion import PARAMS, Fusion, rank_groups

FEATS = ["amt_robust_z", "crypto_payee", "days_since_flag"]


def frame(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=FEATS)
    X["household_id"] = [f"HH-{i % 300:05d}" for i in range(n)]
    X["day"] = rng.uniform(0, 140, n)
    y = (X["amt_robust_z"] + X["crypto_payee"] - X["days_since_flag"] + rng.normal(size=n) > 2.5).astype(int)
    return X, y.to_numpy()


def test_default_fusion_unchanged():
    m = Fusion(FEATS, seed=3)
    assert isinstance(m.model, lgb.LGBMClassifier)
    got = m.model.get_params()
    assert all(got[k] == v for k, v in PARAMS.items()) and got["random_state"] == 3 and got["class_weight"] is None
    X, y = frame()
    a = Fusion(FEATS).fit(X, y, X, y).raw(X)
    assert ((a >= 0) & (a <= 1)).all()


def test_lambdarank_fusion_fits_and_explains():
    X, y = frame()
    m = Fusion(FEATS, params={"objective": "lambdarank", "n_estimators": 30, "class_weight": "balanced"}).fit(X, y, X, y)
    assert isinstance(m.model, lgb.LGBMRanker)
    r = m.raw(X)
    assert r.shape == (len(X),) and np.corrcoef(r, y)[0, 1] > 0.3
    p = m.calibrated(r)
    assert ((p >= 0) & (p <= 1)).all()
    c = m.contributions(X.iloc[:5])
    assert list(c.columns) == FEATS
    full = m.model.booster_.predict(X.iloc[:5][FEATS].to_numpy(np.float32), pred_contrib=True)
    assert np.allclose(full.sum(1), r[:5], atol=1e-6)


def test_rank_groups_split_big_weeks():
    X, _ = frame(6000)
    order, g = rank_groups(X, max_group=200)
    assert sum(g) == len(X) and max(g) <= 200 and sorted(order) == list(range(len(X)))
    wk = (X["day"].to_numpy()[order] // 7).astype(int)
    assert (np.diff(wk) >= 0).all()  # groups never mix weeks


def row(obj, d, recall, fa, n=300):
    return {"objective": obj, "params": {"max_depth": d, "n_estimators": n, "learning_rate": 0.05, "class_weight": None},
            "val": {"0.25%": {"recall": recall, "lookalike_false_alarm_rate": fa}}}


def test_select_max_val_recall_then_fa_then_simpler():
    assert select([row("binary", 3, 0.50, 0.1), row("lambdarank", 4, 0.60, 0.3), row("binary", 4, 0.55, 0.0)])["val"]["0.25%"]["recall"] == 0.60
    assert select([row("binary", 4, 0.6, 0.2), row("lambdarank", 4, 0.6, 0.1)])["objective"] == "lambdarank"
    assert select([row("lambdarank", 4, 0.6, 0.1), row("binary", 3, 0.6, 0.1)])["params"]["max_depth"] == 3


def test_grid_sizes():
    g = grid()
    assert len(g) == 24 and sum(c["objective"] == "lambdarank" for c in g) == 8
    assert all(c["params"]["class_weight"] is None for c in g if c["objective"] == "lambdarank")
    assert all(c["params"]["num_leaves"] == 2 ** c["params"]["max_depth"] for c in g)
