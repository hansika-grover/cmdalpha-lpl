"""Alert policy: a household alerts when its score crosses a threshold, then is suppressed
while under review (30 days). The threshold is set on validation so that the weekly share of
households alerted matches a reviewer budget. Scores are evaluated at request time, so an
alert on a withdrawal request comes before the money is released."""
from __future__ import annotations

import numpy as np
import pandas as pd

SUPPRESS_DAYS = 30
WEEKS = 730 / 7


def simulate(hh: np.ndarray, day: np.ndarray, score: np.ndarray, tau: float, force=None) -> np.ndarray:
    """Indices of points that raise an alert. `force` marks hard-rule points that alert regardless of tau."""
    hit = score >= tau
    if force is not None:
        hit = hit | force
    idx = np.flatnonzero(hit)
    if len(idx) == 0:
        return idx
    order = idx[np.lexsort((day[idx], hh[idx]))]
    keep, last_h, last_d = [], None, -1e9
    for i in order:
        h, d = hh[i], day[i]
        if h != last_h:
            last_h, last_d = h, -1e9
        if d - last_d >= SUPPRESS_DAYS:
            keep.append(i)
            last_d = d
    return np.array(keep, dtype=int)


def weekly_rate(n_alerts: int, n_households: int) -> float:
    return n_alerts / n_households / WEEKS


def tau_for_budget(hh, day, score, n_households, budget, force=None) -> float:
    """Smallest threshold whose weekly alert share does not exceed the budget."""
    s = np.unique(np.quantile(score, np.linspace(0.5, 1.0, 4001)))
    lo, hi = 0, len(s) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if weekly_rate(len(simulate(hh, day, score, s[mid], force)), n_households) > budget:
            lo = mid + 1
        else:
            hi = mid
    return float(s[lo])


def alerts_frame(F: pd.DataFrame, idx: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    a = F.iloc[idx][["household_id", "day", "ref_id", "kind"]].copy()
    a["score"] = score[idx]
    return a.sort_values(["household_id", "day"]).reset_index(drop=True)
