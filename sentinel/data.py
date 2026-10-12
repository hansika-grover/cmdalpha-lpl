"""Loading the synthetic world.

`load_world` is the only entry point the feature and model code uses. It reads the
data files and never the labels directory, which is what the leakage test checks.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .generate.world import DATA_FILES, LABEL_FILES

START = pd.Timestamp("2024-10-01")


def _day(ts: pd.Series) -> pd.Series:
    return (ts - START).dt.total_seconds() / 86400.0


def load_world(d="data/world") -> dict[str, pd.DataFrame]:
    d = Path(d)
    W = {}
    for k in DATA_FILES:
        W[k] = pd.read_csv(d / f"{k}.csv", low_memory=False)
    for k in ("money_movements", "account_events", "notes"):
        W[k]["ts"] = pd.to_datetime(W[k]["ts"], format="%Y-%m-%d %H:%M")
        W[k]["day"] = _day(W[k]["ts"])
    W["payees"]["created_ts"] = pd.to_datetime(W["payees"]["created_ts"], format="%Y-%m-%d %H:%M")
    hh = W["households"]
    W["hh_index"] = pd.Series(range(len(hh)), index=hh["household_id"])
    return W


def load_labels(d="data/world") -> dict[str, pd.DataFrame]:
    """Ground truth. Only the evaluation harness (and supervised training targets) may call this."""
    d = Path(d) / "labels"
    L = {k: pd.read_csv(d / f"{k}.csv") for k in LABEL_FILES + ["note_truth"]}
    for c in ("start_ts", "first_outflow_ts", "last_outflow_ts"):
        L["cases"][c] = pd.to_datetime(L["cases"][c])
        L["cases"][c.replace("_ts", "_day")] = _day(L["cases"][c])
    L["negatives"]["start_ts"] = pd.to_datetime(L["negatives"]["start_ts"])
    return L
