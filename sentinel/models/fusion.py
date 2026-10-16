"""Fusion: a shallow, monotone gradient-boosted model over the three engines, calibrated on validation.

Ranking uses the raw score (no ties); the calibrated probability is what people see, so
0.7 means about 70% of similar points in the simulation were real cases.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

DECREASING = ("since_", "days_since", "mitigating", "ref_age_days")
NEUTRAL = ("is_outflow", "amt_log", "thin_history", "notes_90d", "method_check", "method_acat", "method_journal",
           "out_cnt_7d", "out_cnt_30d")

PARAMS = dict(objective="binary", max_depth=3, num_leaves=8, learning_rate=0.05, n_estimators=400,
              min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              verbose=-1, n_jobs=-1)


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


class Fusion:
    def __init__(self, features: list[str], seed: int = 0):
        self.features = list(features)
        self.model = lgb.LGBMClassifier(**PARAMS, random_state=seed, monotone_constraints=monotone(self.features))
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)

    def fit(self, X: pd.DataFrame, y, Xv: pd.DataFrame, yv):
        self.model.fit(X[self.features].to_numpy(np.float32), np.asarray(y))
        self.iso.fit(self.raw(Xv), np.asarray(yv))
        return self

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features].to_numpy(np.float32))[:, 1]

    def calibrated(self, raw: np.ndarray) -> np.ndarray:
        return self.iso.predict(raw)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """TreeSHAP contributions in log-odds (LightGBM pred_contrib); last column is the bias."""
        c = self.model.booster_.predict(X[self.features].to_numpy(np.float32), pred_contrib=True)
        return pd.DataFrame(c[:, :-1], columns=self.features, index=X.index)
