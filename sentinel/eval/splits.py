"""Household-level splits and point labels. No household appears in two splits."""
from __future__ import annotations

import numpy as np
import pandas as pd


def household_split(households: pd.DataFrame, cases: pd.DataFrame, seed: int, frac=(0.6, 0.2, 0.2)) -> pd.Series:
    """Stratified by typology (non-cases form one stratum) so every split sees every scam type."""
    rng = np.random.default_rng(seed + 1000)
    strata = households[["household_id"]].merge(cases[["household_id", "typology"]], on="household_id", how="left")
    strata["typology"] = strata["typology"].fillna("none")
    out = {}
    for _, g in strata.groupby("typology", sort=True):
        ids = g["household_id"].to_numpy()[rng.permutation(len(g))]
        a, b = int(round(len(ids) * frac[0])), int(round(len(ids) * (frac[0] + frac[1])))
        out.update({h: "train" for h in ids[:a]})
        out.update({h: "val" for h in ids[a:b]})
        out.update({h: "test" for h in ids[b:]})
    return households["household_id"].map(out).set_axis(households["household_id"])


def point_labels(F: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """y = 1 while an exploitation case is under way (case start to last outflow).
    Points after a case ends are marked `post` and left out of training and calibration."""
    c = cases.set_index("household_id")
    j = F[["household_id", "day"]].join(c[["start_day", "last_outflow_day", "case_id"]], on="household_id")
    y = ((j["day"] >= j["start_day"]) & (j["day"] <= j["last_outflow_day"] + 1e-6)).astype(int)
    post = (j["day"] > j["last_outflow_day"] + 1e-6).fillna(False)
    return pd.DataFrame({"y": y.to_numpy(), "post": post.to_numpy(), "case_id": j["case_id"].to_numpy()}, index=F.index)


def note_labels(notes: pd.DataFrame, cases: pd.DataFrame) -> pd.Series:
    """Weak note-level target for the TF-IDF baseline: note written in a case household around the case."""
    c = cases.set_index("household_id")
    j = notes[["household_id", "day"]].join(c[["start_day", "last_outflow_day"]], on="household_id")
    return ((j["day"] >= j["start_day"] - 45) & (j["day"] <= j["last_outflow_day"] + 7)).astype(int)
