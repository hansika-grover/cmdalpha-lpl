"""Assemble the scoring-point feature table from the three engines."""
from __future__ import annotations

import time

import pandas as pd

from . import money, notes, sequences
from .common import build_points

GROUPS = {"money": money.FEATURES, "sequence": sequences.FEATURES}


def notes_group(prefix: str) -> list[str]:
    return [f"{prefix}_{c}" for c in notes.FEATURES]


def build_features(W, readers: dict[str, pd.DataFrame], verbose=True) -> pd.DataFrame:
    """One row per scoring point. `readers` maps a reader name (e.g. 'kw', 'llm') to its per-note output."""
    t = time.time()
    P = build_points(W)
    mm = money.enrich_txns(W)
    A = money.money_features(W, P, mm)
    B = sequences.sequence_features(W, P, mm)
    parts = [P, A, B]
    for name, out in readers.items():
        C = notes.notes_features(W, P, out)
        C.columns = [f"{name}_{c}" for c in C.columns]
        parts.append(C)
    F = pd.concat(parts, axis=1)
    if verbose:
        print(f"features: {len(F):,} scoring points x {F.shape[1]} columns in {time.time() - t:.0f}s")
    return F
