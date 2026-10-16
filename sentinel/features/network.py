"""Money movement, firm-wide: the receiving account / wallet (payees.dest_ref) seen across every household.

Scam rings reuse mule accounts across victims; only a firm-wide view can see it. At a money-movement point
(household h, ref r, time t) we count OTHER households that paid r at or before t in the last 30/90 days,
how many of those paid r for the first time in the window, the days since r was first paid firm-wide (CAP if
already paid in the first 90 days of data, whose earlier history we don't see), and the 90-day count again but only while r is young (first paid < 60 days ago): new accounts many clients start paying.
Other points get 0 (counts) and CAP (age).

Exact distinct-household counts with one sorted array: each (ref, household) pair is "in the window" on the
union of [d, d + w) over its payment times d. Merging those intervals gives disjoint segments; +1 at each
segment start and -1 at its end, keyed by ref_index * OFF_MIN + minute, so a prefix sum at the query key is the
number of households in the window. Times are whole minutes (ts is minute-floored), so every boundary is exact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import CAP, Stream

FEATURES = ["payee_other_hh_30d", "payee_other_hh_90d", "payee_first_time_payers_90d", "payee_ref_age_days",
            "payee_young_ref_other_hh_90d"]
YOUNG_REF_DAYS = 60  # a receiving account first paid by any client this recently
ESTABLISHED_DAYS = 90  # data starts at day 0 but the firm's history doesn't: accounts paid in the first 90 days are
                       # treated as established (age = CAP), as money.PEER_DAYS treats the first 90 days as history
OFF_MIN = 4_000_000.0  # minutes per ref block (> 7 years); keys stay exact in float64
DAY = 1440


def txn_refs(W) -> pd.DataFrame:
    """txn_id, household_id, dest_ref, minute (sorted by ref, household, minute)."""
    mm = W["money_movements"][["txn_id", "household_id", "payee_id", "day"]]
    pay = W["payees"]
    ref = pay["dest_ref"] if "dest_ref" in pay else pay["payee_id"]  # worlds without the column: one ref per payee
    t = mm.merge(pd.DataFrame({"payee_id": pay["payee_id"], "dest_ref": ref.fillna(pay["payee_id"])}), on="payee_id", how="left")
    t["dest_ref"] = t["dest_ref"].fillna(t["payee_id"])
    t["minute"] = np.round(t["day"].to_numpy() * DAY)
    t["ref_i"] = pd.factorize(t["dest_ref"], sort=True)[0].astype(float)
    return t.sort_values(["ref_i", "household_id", "minute"], kind="stable").reset_index(drop=True)


def _distinct(t: pd.DataFrame, w: float) -> Stream:
    """+1/-1 stream whose prefix sum at ref*OFF_MIN + minute = households that paid ref in (minute - w, minute]."""
    m = t["minute"].to_numpy()
    pair = t.groupby(["ref_i", "household_id"], sort=False).ngroup().to_numpy()
    new_pair = np.r_[True, pair[1:] != pair[:-1]]
    new_seg = new_pair | np.r_[True, np.diff(m) > w]
    seg = np.cumsum(new_seg)
    g = pd.DataFrame({"seg": seg, "m": m, "r": t["ref_i"].to_numpy()}).groupby("seg").agg(a=("m", "min"), b=("m", "max"), r=("r", "first"))
    base = g["r"].to_numpy() * OFF_MIN
    keys = np.r_[base + g["a"].to_numpy(), base + g["b"].to_numpy() + w]
    vals = np.r_[np.ones(len(g)), -np.ones(len(g))]
    return Stream(keys, vals)


def txn_network(W) -> pd.DataFrame:
    """The four features for every transaction, indexed by txn_id (plus dest_ref)."""
    t = txn_refs(W)
    q = t["ref_i"].to_numpy() * OFF_MIN + t["minute"].to_numpy()
    whole = OFF_MIN / 2  # covers a ref's whole block and nothing of the previous one
    out = pd.DataFrame({"dest_ref": t["dest_ref"].to_numpy()}, index=t["txn_id"].to_numpy())
    for w, col in ((30, "payee_other_hh_30d"), (90, "payee_other_hh_90d")):
        out[col] = _distinct(t, w * DAY).window(q, whole) - 1  # the current payment keeps its own household in
    # first-time payers: each (ref, household) pair counted once, at its first payment
    first = t.groupby(["ref_i", "household_id"], sort=False)["minute"].transform("min").to_numpy()
    fp = ~t.duplicated(["ref_i", "household_id"]).to_numpy()
    own_first = (first > t["minute"].to_numpy() - 90 * DAY).astype(float)
    out["payee_first_time_payers_90d"] = Stream(q[fp]).window(q, 90 * DAY) - own_first
    ref_first = t.groupby("ref_i")["minute"].transform("min").to_numpy()
    age = np.minimum((t["minute"].to_numpy() - ref_first) / DAY, CAP)
    out["payee_ref_age_days"] = np.where(ref_first < ESTABLISHED_DAYS * DAY, CAP, age)
    # mule signature: several clients paying an account that is new to the firm (big legitimate payees are old)
    out["payee_young_ref_other_hh_90d"] = out["payee_other_hh_90d"] * (out["payee_ref_age_days"] < YOUNG_REF_DAYS)
    return out


def network_features(W, P: pd.DataFrame) -> pd.DataFrame:
    N = txn_network(W)
    F = pd.DataFrame(index=P.index)
    cur = N.reindex(P["ref_id"].where(P["kind"] == "mm"))
    for c in FEATURES:
        fill = CAP if c == "payee_ref_age_days" else 0.0
        F[c] = cur[c].fillna(fill).to_numpy()
    return F
