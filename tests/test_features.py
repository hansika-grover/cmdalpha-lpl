"""No label reaches the features, and no future record reaches a score."""
import inspect

import numpy as np
import pandas as pd
import pytest

import sentinel.features.build as build_mod
import sentinel.features.memory as memory_mod
import sentinel.features.money as money_mod
import sentinel.features.notes as notes_mod
import sentinel.features.sequences as seq_mod
from sentinel.data import load_world
from sentinel.features.build import build_features
from sentinel.generate.world import write_world
from sentinel.llm.keyword_reader import keyword_read
from tests.test_generator import LABEL_FIELDS


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("w")
    write_world(d, households=500)
    return d


def test_feature_code_never_loads_labels():
    for mod in (build_mod, money_mod, notes_mod, seq_mod):
        src = inspect.getsource(mod)
        assert "load_labels" not in src and "labels" not in src.replace("labels=False", ""), mod.__name__


def test_no_label_columns_in_features(world):
    W = load_world(world)
    F = build_features(W, {"kw": keyword_read(W["notes"])}, verbose=False)
    assert not set(F.columns) & LABEL_FIELDS


def test_features_ignore_the_future(world):
    """Scores at time t must not change when every record after t is deleted."""
    W = load_world(world)
    cut = 400.0
    F = build_features(W, {"kw": keyword_read(W["notes"])}, verbose=False)
    W2 = {k: v for k, v in W.items()}
    for k in ("money_movements", "account_events", "notes"):
        W2[k] = W[k][W[k]["day"] <= cut].reset_index(drop=True)
    F2 = build_features(W2, {"kw": keyword_read(W2["notes"])}, verbose=False)
    a = F[F["day"] <= cut].drop(columns=["ts"]).reset_index(drop=True)
    b = F2.drop(columns=["ts"]).reset_index(drop=True)
    assert (a["ref_id"].to_numpy() == b["ref_id"].to_numpy()).all()
    num = [c for c in a.columns if c not in ("kind", "ref_id", "household_id")]
    diff = np.abs(a[num].to_numpy(float) - b[num].to_numpy(float))
    assert np.nanmax(diff) < 1e-6, a[num].columns[np.nanargmax(diff.max(axis=0))]
    # household memory: engine counts too (the score history is tested on a tiny frame below)
    ma = memory_mod.engine_features(F, "kw")[(F["day"] <= cut).to_numpy()].to_numpy(float)
    mb = memory_mod.engine_features(F2, "kw").to_numpy(float)
    assert ma.shape == mb.shape and np.abs(ma - mb).max() < 1e-6
    assert ma[:, 0].max() > 0  # something fired


def test_score_history_uses_earlier_points_only():
    day = np.array([0.0, 5.0, 10.0, 20.0, 0.0, 1.0])
    F = pd.DataFrame({"household_id": ["B"] * 4 + ["A"] * 2,  # groups not in sorted order
                      "ts": pd.Timestamp("2024-10-01") + pd.to_timedelta(day, unit="D")})
    s = np.array([0.1, 0.9, 0.2, 0.3, 0.5, 0.4])
    H = memory_mod.score_history(F, s)
    assert H["peak_score_14d"].tolist() == [0.0, 0.1, 0.9, 0.2, 0.0, 0.5]  # day 20: day 5 is outside 14 days; own score never counts
    assert np.allclose(H["mean_score_30d"], [0.0, 0.1, 0.5, 0.4, 0.0, 0.5])
    s2 = s.copy()
    s2[3] = 99.0  # changing a point never changes its own row or any earlier one
    assert (memory_mod.score_history(F, s2) == H).all().all()


def test_oof_stage1_never_scores_a_household_it_trained_on(monkeypatch):
    class Rec:
        made = []

        def __init__(self, feats, seed=0):
            self.i = len(Rec.made)
            Rec.made.append(self)

        def fit(self, X, y, Xv, yv):
            self.seen = set(X["household_id"])
            return self

        def raw(self, X):
            return np.full(len(X), float(self.i))

    monkeypatch.setattr(memory_mod, "Fusion", Rec)
    rng = np.random.default_rng(0)
    hh = np.repeat([f"H{i}" for i in range(60)], 5)
    n = np.arange(len(hh))
    split = np.where(n < 200, "train", np.where(n < 250, "val", "test"))
    F = pd.DataFrame({"household_id": hh, "x": rng.random(len(hh))})
    y = (rng.random(len(hh)) < 0.1).astype(int)
    s, full = memory_mod.stage1_scores(F, y, np.zeros(len(hh), bool), split, ["x"])
    assert len(Rec.made) == 1 + memory_mod.FOLDS and full is Rec.made[0]
    for i in range(len(hh)):
        m = Rec.made[int(s[i])]
        if split[i] == "train":
            assert m is not full and hh[i] not in m.seen
        else:
            assert m is full
