"""Engine C features: aggregate any notes reader's per-note output over time.

A reader returns one row per note: note_id, flag_0..flag_7 (FinCEN-based categories),
mitigating, injection, and optionally risk (0-1). Injection is reported but never used
as a risk feature, so planted instructions cannot move a score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..generate.text import CATEGORIES
from .common import Stream, keyed

FLAG_COLS = [f"flag_{i}" for i in range(len(CATEGORIES))]
FEATURES = ([f"n30_{c}" for c in FLAG_COLS] + [f"n90_{c}" for c in FLAG_COLS] +
            ["flag_notes_30d", "flag_notes_90d", "distinct_cats_90d", "days_since_flag_note",
             "mitigating_90d", "notes_90d", "risk_max_note_30d", "risk_sum_90d"])


def notes_features(W, P: pd.DataFrame, reader_out: pd.DataFrame) -> pd.DataFrame:
    n = W["notes"][["note_id", "household_id", "day"]].merge(reader_out, on="note_id", how="left")
    for c in FLAG_COLS + ["mitigating"]:
        n[c] = n[c].fillna(0).astype(float)
    n["risk"] = n["risk"].fillna(0.0) if "risk" in n else 0.0
    n["key"] = keyed(n, W["hh_index"])
    q = P["key"].to_numpy()
    F = pd.DataFrame(index=P.index)
    k_note = n["key"].to_numpy()
    # a reader result can only be used once the note has been read (avail_day >= note day)
    if "avail_day" in n:
        n["avail_day"] = np.maximum(n["avail_day"].fillna(n["day"]), n["day"])
        k = (n["key"] - n["day"] + n["avail_day"]).to_numpy()
    else:
        k = k_note
    d90 = {}
    for c in FLAG_COLS:
        s = Stream(k, n[c])
        F[f"n30_{c}"] = s.window(q, 30)
        d90[c] = s.window(q, 90)
        F[f"n90_{c}"] = d90[c]
    anyflag = (n[FLAG_COLS].sum(axis=1) > 0).astype(float).to_numpy()
    s_any = Stream(k, anyflag)
    F["flag_notes_30d"] = s_any.window(q, 30)
    F["flag_notes_90d"] = s_any.window(q, 90)
    F["distinct_cats_90d"] = np.sum([d90[c] > 0 for c in FLAG_COLS], axis=0)
    F["days_since_flag_note"] = Stream(k[anyflag > 0]).since(q)
    F["mitigating_90d"] = Stream(k, n["mitigating"]).window(q, 90)
    F["notes_90d"] = Stream(k_note).window(q, 90)
    # max risk over 30d approximated by count of notes with risk >= 0.5 (window max is not cumulative)
    F["risk_max_note_30d"] = Stream(k, (n["risk"].to_numpy() >= 0.5).astype(float)).window(q, 30)
    F["risk_sum_90d"] = Stream(k, n["risk"].to_numpy()).window(q, 90)
    return F
