"""Today's methods: fixed withdrawal thresholds and a hand-written expert checklist (no notes)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def amount(F: pd.DataFrame) -> np.ndarray:
    return np.expm1(F["amt_log"].to_numpy())


def threshold_scores(F: pd.DataFrame, limit: float) -> np.ndarray:
    return ((F["is_outflow"].to_numpy() == 1) & (amount(F) >= limit)).astype(float)


CHECKLIST = {
    "New payee and $10,000 or more": lambda F, a: (F["young_payee"] == 1) & (a >= 10000),
    "Email or phone changed in the last 30 days, then $5,000 or more out": lambda F, a: (F["days_since_contact_change"] <= 30) & (a >= 5000),
    "Bank linked in the last 14 days, then money to a new payee": lambda F, a: (F["since_bank_link_added"] <= 14) & (F["young_payee"] == 1),
    "Money to a crypto exchange": lambda F, a: F["payee_crypto"] == 1,
    "Payee is the trusted contact or a power of attorney": lambda F, a: (F["payee_is_trusted_contact"] + F["payee_is_poa_or_joint"]) > 0,
    "20% or more of assets out at once": lambda F, a: F["share_assets"] >= 0.2,
    "Password reset or new device in the last 7 days, then a new payee": lambda F, a: (np.minimum(F["since_password_reset"], F["since_new_device"]) <= 7) & (F["young_payee"] == 1),
}


def checklist_scores(F: pd.DataFrame) -> np.ndarray:
    """Number of checklist rules fired on an outflow; a tiny amount term breaks ties for budgeting."""
    a = amount(F)
    out = F["is_outflow"].to_numpy() == 1
    n = np.zeros(len(F))
    for rule in CHECKLIST.values():
        n += np.asarray(rule(F, a), dtype=float)
    return np.where(out, n + np.log1p(a) / 100.0, 0.0)


def hard_rule(F: pd.DataFrame) -> np.ndarray:
    """Always escalate: money going repeatedly to the trusted contact or a new power of attorney."""
    role = (F["payee_is_trusted_contact"].to_numpy() + F["payee_is_recent_poa"].to_numpy()) > 0
    return (F["is_outflow"].to_numpy() == 1) & role & (F["role_payee_cnt_90d"].to_numpy() >= 2)
