"""Household memory: slow-burn theft where each step looks small on its own.

Two kinds of feature, both using only scoring points at or before the current one:
* engine counts: how often each engine "fired" in the household in the last 30 days, from simple
  indicator rules on the feature table (no model):
    money    : withdrawal with robust z > 3, or >= $5k to a young payee, or a sub-$10k transfer
               with 2+ such in 60 days (structuring);
    sequence : any named chain at this point (ato_chain, access_then_new_payee, tc_change_then_out,
               poa_then_out, payee_is_*);
    notes    : a note the reader flagged (counted from when it was read, as in notes.py).
* score history: rolling max / mean of a stage-1 score (Fusion on money + sequence + notes, no
  memory) over the household's EARLIER points in the last 14 / 30 days (current point excluded, so the
  current event's risk is credited to its own engine, not to "history"; 0 when there is no earlier point).
  Training rows get out-of-fold stage-1 scores (5 folds by household within train); validation
  and test rows get the stage-1 model trained on every training row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.fusion import Fusion
from .common import CAP, Stream

ENGINE_FEATURES = ["engines_fired_30d", "money_fired_30d", "sequence_fired_30d", "notes_fired_30d", "days_since_any_fired"]
SCORE_FEATURES = ["peak_score_14d", "mean_score_30d"]
FEATURES = ENGINE_FEATURES + SCORE_FEATURES
SEQ_FIRES = ["ato_chain", "access_then_new_payee", "tc_change_then_out", "poa_then_out",
             "payee_is_trusted_contact", "payee_is_poa_or_joint", "payee_is_recent_poa"]
FOLDS = 5


def fired(F: pd.DataFrame) -> dict[str, np.ndarray]:
    """Point-level indicators for the money and sequence engines (notes come from the reader windows)."""
    a = np.expm1(F["amt_log"].to_numpy())
    out = F["is_outflow"].to_numpy() > 0
    struct = (a >= 8000) & (a < 10000) & (F["struct_cnt_60d"].to_numpy() >= 2)
    money = out & ((F["robust_z"].to_numpy() > 3) | ((F["young_payee"].to_numpy() > 0) & (a >= 5000)) | struct)
    seq = (F[SEQ_FIRES].to_numpy() > 0).any(axis=1)
    return {"money": money, "sequence": seq}


def engine_features(F: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """`prefix` is the notes reader whose columns to use (e.g. 'kw' or 'llm')."""
    q = F["key"].to_numpy()
    f = fired(F)
    M = pd.DataFrame(index=F.index)
    M["money_fired_30d"] = Stream(q[f["money"]]).window(q, 30)
    M["sequence_fired_30d"] = Stream(q[f["sequence"]]).window(q, 30)
    M["notes_fired_30d"] = F[f"{prefix}_flag_notes_30d"].to_numpy()
    M["engines_fired_30d"] = (M[["money_fired_30d", "sequence_fired_30d", "notes_fired_30d"]] > 0).sum(axis=1)
    ms = Stream(q[f["money"] | f["sequence"]]).since(q)
    M["days_since_any_fired"] = np.minimum(np.minimum(ms, F[f"{prefix}_days_since_flag_note"].to_numpy()), CAP)
    return M[ENGINE_FEATURES]


def fired_points(F: pd.DataFrame, reader_out: pd.DataFrame | None, notes: pd.DataFrame | None) -> pd.DataFrame:
    """Records that fired, for reason sentences: household_id, day (when usable), ref_id, engine."""
    f = fired(F)
    parts = [F.loc[f[e], ["household_id", "day", "ref_id"]].assign(engine=e) for e in ("money", "sequence")]
    if reader_out is not None and notes is not None:
        r = notes[["note_id", "household_id", "day"]].merge(reader_out, on="note_id")
        hit = r[[c for c in r.columns if c.startswith("flag_")]].fillna(0).sum(axis=1) > 0
        r = r[hit]
        d = np.maximum(r["avail_day"].fillna(r["day"]), r["day"]) if "avail_day" in r else r["day"]
        parts.append(pd.DataFrame({"household_id": r["household_id"], "day": d, "ref_id": r["note_id"], "engine": "notes"}))
    return pd.concat(parts, ignore_index=True)


def score_history(F: pd.DataFrame, s: np.ndarray) -> pd.DataFrame:
    """Rolling max over 14 days and mean over 30 days of `s` per household, over points strictly before this one.
    Within a household F must be in time order (as build_points leaves it)."""
    h = F["household_id"].to_numpy()
    d = pd.DataFrame({"h": h, "t": pd.to_datetime(F["ts"]).to_numpy(), "s": s})
    g = d.groupby("h", sort=True).rolling
    order = np.argsort(h, kind="stable")  # row order of the grouped result (groups sorted, rows kept in order)
    out = pd.DataFrame(index=F.index)
    for col, w, fn in (("peak_score_14d", "14D", "max"), ("mean_score_30d", "30D", "mean")):
        v = np.empty(len(d))
        v[order] = getattr(g(w, on="t", closed="left")["s"], fn)().to_numpy()
        out[col] = np.nan_to_num(v, nan=0.0)
    return out


def fold_of(households, y_hh: pd.Series, seed=0, k=FOLDS) -> pd.Series:
    """Train households -> fold 0..k-1, case households spread evenly across folds."""
    rng = np.random.default_rng(seed + 2000)
    out = {}
    for pos in (True, False):
        ids = np.array(sorted(h for h in households if bool(y_hh.get(h, 0)) == pos))
        ids = ids[rng.permutation(len(ids))]
        out.update({h: i % k for i, h in enumerate(ids)})
    return pd.Series(out)


def stage1_scores(F: pd.DataFrame, y, post, split, feats: list[str], seed=0, k=FOLDS):
    """Stage-1 raw scores: out-of-fold for train rows, full-train model for the rest. Returns (scores, full model).
    Labels are used only to fit on training rows (validation rows calibrate, as in the main model)."""
    y, post, split = np.asarray(y), np.asarray(post, bool), np.asarray(split)
    h = F["household_id"].to_numpy()
    tr, va = (split == "train") & ~post, (split == "val") & ~post
    trh = pd.unique(h[split == "train"])
    fold = fold_of(trh, pd.Series(y[tr], index=h[tr]).groupby(level=0).max(), seed, k)
    fr = pd.Series(h).map(fold).to_numpy()
    s = np.zeros(len(F))
    full = Fusion(feats, seed=seed).fit(F[tr], y[tr], F[va], y[va])
    rest = split != "train"
    s[rest] = full.raw(F[rest])
    for i in range(k):
        fit = tr & (fr != i)
        m = Fusion(feats, seed=seed).fit(F[fit], y[fit], F[va], y[va])
        sc = (split == "train") & (fr == i)
        s[sc] = m.raw(F[sc])
    return s, full


def attach(F: pd.DataFrame, stage1: Fusion, prefix: str, s=None) -> pd.DataFrame:
    """Add memory columns to F. Without precomputed `s`, the given stage-1 model scores every row (export, robustness)."""
    s = stage1.raw(F) if s is None else s
    return pd.concat([F, engine_features(F, prefix), score_history(F, s)], axis=1)
