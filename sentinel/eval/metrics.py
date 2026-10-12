"""Evaluation metrics over a split, with household-bootstrap confidence intervals.

All metrics are computed from per-household summaries, so a bootstrap replicate is just a
reweighting of households. Base-rate reweighting scales non-case households so cases are
0.5% of households instead of the planted 2%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .alerts import WEEKS

AGE_BANDS = [(55, 64), (65, 74), (75, 84), (85, 200)]
BAND_NAMES = ["55-64", "65-74", "75-84", "85+"]


class Truth:
    """Ground truth for one split, indexed for fast scoring of any alert set."""

    def __init__(self, W, L, households: list[str]):
        hs = set(households)
        self.households = pd.Index(sorted(hs))
        hh = W["households"].set_index("household_id").loc[self.households]
        self.age_band = np.select([(hh["age"] >= lo) & (hh["age"] <= hi) for lo, hi in AGE_BANDS], range(4), 0)
        ids = pd.concat([W["money_movements"][["txn_id", "day", "amount"]].rename(columns={"txn_id": "id"}),
                         W["account_events"][["event_id", "day"]].rename(columns={"event_id": "id"}),
                         W["notes"][["note_id", "day"]].rename(columns={"note_id": "id"})]).set_index("id")
        c = L["cases"][L["cases"]["household_id"].isin(hs)].copy()
        c["txns"] = [[r for r in str(s).split(";") if r.startswith("TX-")] for s in c["related_record_ids"]]
        self.cases = c.reset_index(drop=True)
        self.case_txn = [ids.loc[t, ["day", "amount"]].to_numpy() for t in c["txns"]]
        n = L["negatives"][L["negatives"]["household_id"].isin(hs)].copy()
        days = [ids.loc[[r for r in str(s).split(";") if r and r != "nan"], "day"].to_numpy() for s in n["related_record_ids"]]
        n["lo"] = [d.min() - 1 if len(d) else np.nan for d in days]
        n["hi"] = [d.max() + 7 if len(d) else np.nan for d in days]
        self.negs = n[n["lo"].notna()].reset_index(drop=True)
        self.is_case = self.households.isin(c["household_id"]).astype(int)
        self.hpos = pd.Series(np.arange(len(self.households)), index=self.households)


def summarize(alerts: pd.DataFrame, T: Truth) -> dict:
    """Per-household / per-case / per-episode arrays for one method's alerts on this split."""
    by_h = {h: g["day"].to_numpy() for h, g in alerts.groupby("household_id")}
    nc = len(T.cases)
    caught, before, warn, prot, loss = np.zeros(nc), np.zeros(nc), np.full(nc, np.nan), np.zeros(nc), np.zeros(nc)
    true_alerts = np.zeros(len(T.households))
    for i, r in T.cases.iterrows():
        tx = T.case_txn[i]
        loss[i] = tx[:, 1].sum() if len(tx) else r["total_loss"]
        d = by_h.get(r["household_id"], np.array([]))
        inwin = d[(d >= r["start_day"]) & (d <= r["last_outflow_day"] + 1e-6)]
        true_alerts[T.hpos[r["household_id"]]] += len(inwin)
        if len(inwin):
            fa = inwin.min()
            caught[i] = 1
            before[i] = float(fa <= r["first_outflow_day"] + 1e-6)
            warn[i] = r["first_outflow_day"] - fa
            prot[i] = tx[tx[:, 0] >= fa - 1e-6, 1].sum() if len(tx) else 0
    neg_flag = np.array([float(np.any((by_h.get(h, np.array([])) >= lo) & (by_h.get(h, np.array([])) <= hi)))
                         for h, lo, hi in zip(T.negs["household_id"], T.negs["lo"], T.negs["hi"])])
    n_alerts = np.zeros(len(T.households))
    cnt = alerts["household_id"].value_counts()
    cnt = cnt[cnt.index.isin(T.households)]
    n_alerts[T.hpos[cnt.index].to_numpy()] = cnt.to_numpy()
    return dict(caught=caught, before=before, warn=warn, prot=prot, loss=loss,
                case_h=T.hpos[T.cases["household_id"]].to_numpy(), typology=T.cases["typology"].to_numpy(),
                cover=T.cases["coached_cover_story"].to_numpy().astype(bool),
                neg_flag=neg_flag, neg_h=T.hpos[T.negs["household_id"]].to_numpy() if len(T.negs) else np.array([], int),
                neg_kind=T.negs["kind"].to_numpy(),
                n_alerts=n_alerts, true_alerts=true_alerts, is_case=T.is_case.astype(int), band=T.age_band)


def compute(S: dict, w: np.ndarray | None = None, realistic=0.005) -> dict:
    """Metrics from summaries; `w` is a household weight vector (bootstrap replicate)."""
    nh = len(S["n_alerts"])
    w = np.ones(nh) if w is None else w
    wc, wn = w[S["case_h"]], w[S["neg_h"]] if len(S["neg_h"]) else np.array([])
    ncase = wc.sum()
    out = {}
    out["recall"] = (wc * S["caught"]).sum() / ncase
    out["caught_before_money_left"] = (wc * S["before"]).sum() / ncase
    out["dollars_protected_share"] = (wc * S["prot"]).sum() / max((wc * S["loss"]).sum(), 1)
    m = S["caught"] > 0
    out["median_warning_days"] = weighted_median(S["warn"][m], wc[m]) if m.any() else np.nan
    out["lookalike_false_alarm_rate"] = (wn * S["neg_flag"]).sum() / wn.sum() if len(wn) else np.nan
    alerts, true = (w * S["n_alerts"]).sum(), (w * S["true_alerts"]).sum()
    out["alerts_per_1k_households_per_week"] = 1000 * alerts / w.sum() / WEEKS
    out["precision"] = true / alerts if alerts else np.nan
    noncase = 1 - S["is_case"]
    ncase_h, nnon = (w * S["is_case"]).sum(), (w * noncase).sum()
    k = ((1 - realistic) / realistic) / (nnon / max(ncase_h, 1))
    fp = alerts - true
    out["precision_at_0.5pct_base_rate"] = true / (true + fp * k) if alerts else np.nan
    alerted = (S["n_alerts"] > 0).astype(float)
    fa_h = (w * noncase * alerted).sum() / nnon
    out["false_alarm_households_share"] = fa_h
    bands = {}
    for b, name in enumerate(BAND_NAMES):
        mb = (S["band"] == b) & (noncase == 1)
        rate = (w[mb] * alerted[mb]).sum() / max(w[mb].sum(), 1e-9)
        bands[name] = {"rate": rate, "ratio_to_overall": rate / fa_h if fa_h else np.nan}
    out["fairness_by_age_band"] = bands
    return out


def weighted_median(x, w):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    if c[-1] == 0:
        return np.nan
    return float(x[np.searchsorted(c, c[-1] / 2)])


SCALARS = ["recall", "caught_before_money_left", "dollars_protected_share", "median_warning_days",
           "lookalike_false_alarm_rate", "alerts_per_1k_households_per_week", "precision",
           "precision_at_0.5pct_base_rate", "false_alarm_households_share"]


def with_ci(S: dict, B=1000, seed=0) -> dict:
    point = compute(S)
    rng = np.random.default_rng(seed)
    nh = len(S["n_alerts"])
    reps = {k: [] for k in SCALARS}
    for _ in range(B):
        w = np.bincount(rng.integers(0, nh, nh), minlength=nh).astype(float)
        r = compute(S, w)
        for k in SCALARS:
            reps[k].append(r[k])
    out = {}
    for k in SCALARS:
        v = np.array(reps[k], dtype=float)
        v = v[~np.isnan(v)]
        out[k] = {"value": _f(point[k]), "ci95": [_f(np.percentile(v, 2.5)), _f(np.percentile(v, 97.5))] if len(v) else None}
    out["fairness_by_age_band"] = {b: {kk: _f(vv) for kk, vv in d.items()} for b, d in point["fairness_by_age_band"].items()}
    out["n_cases"] = int(len(S["caught"]))
    out["n_lookalikes"] = int(len(S["neg_flag"]))
    out["n_households"] = int(nh)
    out["by_typology"] = {t: {"recall": _f(S["caught"][S["typology"] == t].mean()),
                              "caught_before_money_left": _f(S["before"][S["typology"] == t].mean()),
                              "n": int((S["typology"] == t).sum())} for t in sorted(set(S["typology"]))}
    out["coached_cover_story_recall"] = _f(S["caught"][S["cover"]].mean()) if S["cover"].any() else None
    out["lookalike_fa_by_kind"] = {k: _f(S["neg_flag"][S["neg_kind"] == k].mean()) for k in sorted(set(S["neg_kind"]))}
    m = S["caught"] > 0
    out["warning_days_distribution"] = [_f(x) for x in np.sort(S["warn"][m])]
    return out


def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)
