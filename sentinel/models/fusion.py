"""Fusion: a shallow, monotone gradient-boosted model over the three engines, calibrated on validation.

Ranking uses the raw score (no ties); the calibrated probability is what people see, so
0.7 means about 70% of similar points in the simulation were real cases.

objective="lambdarank" (chosen by eval/tune.py on validation, config/model.yaml) trains an LGBMRanker
grouped by week, since alerts are the top of a weekly ranking. LightGBM caps a query at 10,000 rows, so
a week is split into k household-hash buckets (smallest k that fits); each bucket is a random sample of
that week's households, so the ranking target is unchanged. Raw score is then the ranker margin.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from sklearn.isotonic import IsotonicRegression

DECREASING = ("since_", "days_since", "mitigating")
NEUTRAL = ("is_outflow", "amt_log", "thin_history", "notes_90d", "method_check", "method_acat", "method_journal",
           "out_cnt_7d", "out_cnt_30d")

PARAMS = dict(objective="binary", max_depth=3, num_leaves=8, learning_rate=0.05, n_estimators=400,
              min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              verbose=-1, n_jobs=-1)
MAX_GROUP = 10000  # LightGBM: "Number of rows ... exceeds upper limit of 10000 for a query"


def monotone(features: list[str]) -> list[int]:
    out = []
    for f in features:
        if f == "recurring" or any(p in f for p in DECREASING):
            out.append(-1)
        elif any(f == p or f.endswith(p) for p in NEUTRAL):
            out.append(0)
        else:
            out.append(1)
    return out


def rank_groups(X: pd.DataFrame, max_group: int = MAX_GROUP) -> tuple[np.ndarray, list[int]]:
    """Row order and group sizes for lambdarank: week = floor(day / 7), split into household-hash buckets
    until every group has at most `max_group` rows."""
    week = (X["day"].to_numpy() // 7).astype(np.int64)
    h = hash_pandas_object(X["household_id"], index=False).to_numpy()
    k = 1
    while True:
        key = week * k + (h % np.uint64(k)).astype(np.int64)
        _, cnt = np.unique(key, return_counts=True)
        if cnt.max() <= max_group:
            break
        k += 1
    order = np.argsort(key, kind="stable")
    return order, np.unique(key[order], return_counts=True)[1].tolist()


class Fusion:
    def __init__(self, features: list[str], seed: int = 0, params: dict | None = None):
        self.features = list(features)
        p = {**PARAMS, **(params or {})}
        self.params = p
        if p["objective"] == "lambdarank":
            p = {k: v for k, v in p.items() if k != "class_weight"}
            self.model = lgb.LGBMRanker(**p, random_state=seed, monotone_constraints=monotone(self.features))
        else:
            self.model = lgb.LGBMClassifier(**p, random_state=seed, monotone_constraints=monotone(self.features))
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)

    def fit(self, X: pd.DataFrame, y, Xv: pd.DataFrame, yv):
        y = np.asarray(y)
        if isinstance(self.model, lgb.LGBMRanker):
            order, group = rank_groups(X)
            self.model.fit(X[self.features].to_numpy(np.float32)[order], y[order].astype(int), group=group)
        else:
            self.model.fit(X[self.features].to_numpy(np.float32), y)
        self.iso.fit(self.raw(Xv), np.asarray(yv))
        return self

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        A = X[self.features].to_numpy(np.float32)
        if isinstance(self.model, lgb.LGBMRanker):
            return self.model.predict(A)
        return self.model.predict_proba(A)[:, 1]

    def calibrated(self, raw: np.ndarray) -> np.ndarray:
        return self.iso.predict(raw)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """TreeSHAP contributions in the model's margin (log-odds for binary; LightGBM pred_contrib); last column is the bias."""
        c = self.model.booster_.predict(X[self.features].to_numpy(np.float32), pred_contrib=True)
        return pd.DataFrame(c[:, :-1], columns=self.features, index=X.index)
