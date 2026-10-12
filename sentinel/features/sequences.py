"""Engine B: account-change sequences and named chains before money moves."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import CAP, Stream, keyed

EVENT_TYPES = ["password_reset", "new_device", "email_change", "phone_change", "address_change", "bank_link_added",
               "poa_added", "joint_owner_added", "trusted_contact_changed", "trusted_contact_removed"]
CHAIN_FEATURES = ["days_since_contact_change", "ato_chain", "ato_chain_young_payee", "access_then_new_payee",
                  "tc_change_then_out", "poa_then_out", "payee_is_trusted_contact", "payee_is_poa_or_joint",
                  "payee_is_recent_poa", "role_payee_sum_90d", "role_payee_cnt_90d"]
FEATURES = [f"cnt30_{t}" for t in EVENT_TYPES] + [f"since_{t}" for t in EVENT_TYPES] + CHAIN_FEATURES


def payee_roles(W, mm: pd.DataFrame) -> pd.DataFrame:
    """For each transaction to a person: is that person the trusted contact / POA / joint owner at request time?"""
    ae = W["account_events"]
    grants = ae[ae["type"].isin(["poa_added", "joint_owner_added", "trusted_contact_changed"]) & ae["related_person_id"].notna()]
    grants = grants[["household_id", "related_person_id", "type", "day"]].rename(columns={"related_person_id": "person_id", "day": "g_day"})
    init = W["households"][["household_id", "trusted_contact_id"]].dropna().rename(columns={"trusted_contact_id": "person_id"})
    init = init.assign(type="trusted_contact_changed", g_day=-1e9)
    grants = pd.concat([grants, init], ignore_index=True)
    tx = mm.loc[mm["person_id"].notna(), ["txn_id", "household_id", "person_id", "day"]]
    j = tx.merge(grants, on=["household_id", "person_id"], how="inner")
    j = j[j["g_day"] <= j["day"]]
    out = pd.DataFrame(index=mm["txn_id"])
    out["payee_is_trusted_contact"] = 0
    out["payee_is_poa_or_joint"] = 0
    out["payee_is_recent_poa"] = 0
    tc = j[j["type"] == "trusted_contact_changed"]["txn_id"].unique()
    poa = j[j["type"] != "trusted_contact_changed"]
    out.loc[tc, "payee_is_trusted_contact"] = 1
    out.loc[poa["txn_id"].unique(), "payee_is_poa_or_joint"] = 1
    out.loc[poa[poa["day"] - poa["g_day"] <= 365]["txn_id"].unique(), "payee_is_recent_poa"] = 1
    return out


def sequence_features(W, P: pd.DataFrame, mm: pd.DataFrame) -> pd.DataFrame:
    F = pd.DataFrame(index=P.index)
    ae = W["account_events"].copy()
    ae["key"] = keyed(ae, W["hh_index"])
    q = P["key"].to_numpy()
    since = {}
    for t in EVENT_TYPES:
        s = Stream(ae.loc[ae["type"] == t, "key"])
        F[f"cnt30_{t}"] = s.window(q, 30)
        since[t] = s.since(q)
        F[f"since_{t}"] = since[t]
    is_out = (P["kind"] == "mm").to_numpy()
    roles = payee_roles(W, mm)
    cur = P[["ref_id"]].merge(mm.set_index("txn_id")[["young_payee", "payee_is_new", "amount", "key"]], left_on="ref_id", right_index=True, how="left")
    cur = cur.join(roles, on="ref_id")
    young = (cur["young_payee"].fillna(False) | cur["payee_is_new"].fillna(False)).to_numpy()
    contact = np.minimum(since["email_change"], since["phone_change"])
    F["days_since_contact_change"] = contact
    F["ato_chain"] = (is_out & (contact <= 30) & (since["bank_link_added"] <= 30)).astype(int)
    F["ato_chain_young_payee"] = (F["ato_chain"].to_numpy() & young).astype(int)
    access = np.minimum(since["password_reset"], since["new_device"])
    F["access_then_new_payee"] = (is_out & (access <= 14) & young).astype(int)
    F["tc_change_then_out"] = (is_out & (np.minimum(since["trusted_contact_changed"], since["trusted_contact_removed"]) <= 30)).astype(int)
    F["poa_then_out"] = (is_out & (np.minimum(since["poa_added"], since["joint_owner_added"]) <= 90)).astype(int)
    for c in ("payee_is_trusted_contact", "payee_is_poa_or_joint", "payee_is_recent_poa"):
        F[c] = cur[c].fillna(0).astype(int).to_numpy()
    rp = roles.reindex(mm["txn_id"])
    role_amt = mm["amount"].to_numpy() * ((rp["payee_is_trusted_contact"] + rp["payee_is_poa_or_joint"]).to_numpy() > 0)
    F["role_payee_sum_90d"] = np.log1p(Stream(mm["key"], role_amt).window(q, 90))
    F["role_payee_cnt_90d"] = Stream(mm["key"], (role_amt > 0).astype(float)).window(q, 90)
    return F
