"""No label reaches the features, and no future record reaches a score."""
import inspect

import numpy as np
import pandas as pd
import pytest

import sentinel.features.build as build_mod
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
