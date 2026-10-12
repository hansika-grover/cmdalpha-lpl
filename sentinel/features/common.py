"""Scoring points and time-window helpers.

A household is re-scored at every event: each money-movement request (scored before
release), each account change and each note. Features at time t use only records with
timestamp <= t. All windows are computed on a global key = household_index * OFF + day,
so one searchsorted serves every household at once (days never exceed OFF - 365).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OFF = 10000.0
CAP = 400.0  # "days since" cap


class Stream:
    """Sorted event keys with cumulative values: O(log n) window sums for any query key."""

    def __init__(self, keys, values=None):
        keys = np.asarray(keys, dtype=float)
        order = np.argsort(keys, kind="stable")
        self.k = keys[order]
        v = np.ones(len(keys)) if values is None else np.asarray(values, dtype=float)[order]
        self.c = np.concatenate([[0.0], np.cumsum(v)])

    def window(self, q, w):
        """Sum of values with key in (q - w, q]."""
        hi = np.searchsorted(self.k, q, side="right")
        lo = np.searchsorted(self.k, q - w, side="right")
        return self.c[hi] - self.c[lo]

    def since(self, q, cap=CAP):
        """Days since the latest event at or before q in the same household (cap if none)."""
        if len(self.k) == 0:
            return np.full(len(q), cap)
        idx = np.searchsorted(self.k, q, side="right") - 1
        ok = idx >= 0
        prev = np.where(ok, self.k[np.clip(idx, 0, None)], -np.inf)
        same = np.floor(prev / OFF) == np.floor(q / OFF)
        return np.where(ok & same, np.minimum(q - prev, cap), cap)


def keyed(df: pd.DataFrame, hh_index: pd.Series) -> np.ndarray:
    return hh_index.loc[df["household_id"]].to_numpy() * OFF + df["day"].to_numpy()


def build_points(W) -> pd.DataFrame:
    parts = []
    for kind, src, idcol in [("mm", "money_movements", "txn_id"), ("ae", "account_events", "event_id"),
                             ("note", "notes", "note_id")]:
        df = W[src]
        parts.append(pd.DataFrame({"kind": kind, "ref_id": df[idcol].to_numpy(),
                                   "household_id": df["household_id"].to_numpy(), "day": df["day"].to_numpy(),
                                   "ts": df["ts"].to_numpy()}))
    P = pd.concat(parts, ignore_index=True)
    P["key"] = keyed(P, W["hh_index"])
    P = P.sort_values(["key", "kind"], kind="stable").reset_index(drop=True)
    return P
