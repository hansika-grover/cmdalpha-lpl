"""Determinism and label-leakage tests for the synthetic world."""
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from sentinel.generate.world import DATA_FILES, LABEL_FILES, write_world

LABEL_FIELDS = {"case_id", "typology", "perpetrator_type", "start_ts", "first_outflow_ts", "last_outflow_ts",
                "total_loss", "episode_id", "related_record_ids", "planted_categories", "has_injection",
                "coached_cover_story"}


def digest(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*.csv"))}


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    a, b, c = (tmp_path_factory.mktemp(x) for x in "abc")
    write_world(a, households=600)
    write_world(b, households=600)
    write_world(c, households=600, seed=99)
    return a, b, c


def test_same_seed_identical_files(worlds):
    a, b, _ = worlds
    da, db = digest(a), digest(b)
    assert len(da) == len(DATA_FILES) + len(LABEL_FILES) + 1  # + note_truth
    assert da == db


def test_different_seed_differs(worlds):
    a, _, c = worlds
    assert digest(a)["money_movements.csv"] != digest(c)["money_movements.csv"]


def test_labels_only_in_labels_dir(worlds):
    a, _, _ = worlds
    for f in DATA_FILES:
        cols = set(pd.read_csv(a / f"{f}.csv", nrows=5).columns)
        assert not cols & LABEL_FIELDS, f"{f} exposes label fields {cols & LABEL_FIELDS}"
    assert {p.name for p in (a / "labels").glob("*.csv")} == {"cases.csv", "negatives.csv", "note_truth.csv"}


def test_ids_do_not_encode_labels(worlds):
    """Record IDs are assigned in time order, so case records are interleaved with everything else."""
    a, _, _ = worlds
    cases = pd.read_csv(a / "labels" / "cases.csv")
    mm = pd.read_csv(a / "money_movements.csv")
    case_tx = {r for s in cases["related_record_ids"] for r in str(s).split(";") if r.startswith("TX-")}
    pos = mm.index[mm["txn_id"].isin(case_tx)]
    assert pos.min() < len(mm) * 0.5 < pos.max()
    assert (mm["ts"].sort_values().to_numpy() == mm["ts"].to_numpy()).all()


def test_planted_rates(worlds):
    a, _, _ = worlds
    hh = pd.read_csv(a / "households.csv")
    cases = pd.read_csv(a / "labels" / "cases.csv")
    assert 0.012 <= len(cases) / len(hh) <= 0.025
    assert set(cases["perpetrator_type"]) <= {"stranger", "trusted"}
    assert (cases["total_loss"] > 0).all()
