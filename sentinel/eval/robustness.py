"""Robustness: score perturbed copies of the world with the trained model and the same thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.build import build_features
from . import alerts as AL
from .baselines import hard_rule
from .metrics import summarize, with_ci

NOISE_TYPES = ["password_reset", "new_device", "email_change", "phone_change", "address_change", "bank_link_added"]
KEYS = ("recall", "caught_before_money_left", "lookalike_false_alarm_rate", "alerts_per_1k_households_per_week")


def perturb(W, kind: str, seed: int = 0):
    rng = np.random.default_rng(seed + 77)
    W2 = dict(W)
    if kind == "no_notes":
        W2["notes"] = W["notes"].iloc[:0].copy()
    elif kind == "half_notes":
        W2["notes"] = W["notes"][rng.random(len(W["notes"])) >= 0.5].reset_index(drop=True)
    elif kind == "noisy_account_events":
        hh = W["households"]["household_id"].to_numpy()
        n = len(W["account_events"])  # double the volume of routine account changes
        extra = pd.DataFrame({"event_id": [f"EV-NOISE-{i}" for i in range(n)], "household_id": rng.choice(hh, n),
                              "type": rng.choice(NOISE_TYPES, n), "channel": "web", "device_id": "Dnoise",
                              "related_person_id": None, "day": rng.uniform(0, 730, n)})
        extra["ts"] = pd.Timestamp("2024-10-01") + pd.to_timedelta(extra["day"], unit="D")
        W2["account_events"] = pd.concat([W["account_events"], extra], ignore_index=True).sort_values("day").reset_index(drop=True)
    return W2


def run(R, model, readers_fn, taus: dict, B=200):
    out = {}
    for kind in ("no_notes", "half_notes", "noisy_account_events"):
        W2 = perturb(R.W, kind, R.seed)
        F2 = build_features(W2, readers_fn(W2), verbose=False)
        F2["split"] = R.split.loc[F2["household_id"]].to_numpy()
        t = (F2["split"] == "test").to_numpy()
        hh = R.W["hh_index"].loc[F2["household_id"]].to_numpy()
        s = model.raw(F2)
        force = hard_rule(F2)
        out[kind] = {}
        for b, tau in taus.items():
            idx = AL.simulate(hh[t], F2["day"].to_numpy()[t], s[t], tau, force[t])
            S = summarize(AL.alerts_frame(F2[t], idx, s[t]), R.truth["test"])
            m = with_ci(S, B=B, seed=R.seed)
            out[kind][b] = {k: m[k] for k in KEYS}
    return out


def by_style_and_cover(R, scores, tau):
    """Recall by the writing style of the household's advisor, and on coached cover stories."""
    t = R.mask["test"]
    s, force = scores
    idx = AL.simulate(R.hh_code[t], R.day[t], s[t], tau, None if force is None else force[t])
    S = summarize(AL.alerts_frame(R.F[t], idx, s[t]), R.truth["test"])
    adv = R.W["advisors"].set_index("advisor_id")["style"]
    style = R.W["households"].set_index("household_id")["advisor_id"].map(adv)
    st = style.loc[R.truth["test"].cases["household_id"]].to_numpy()
    return ({k: {"recall": round(float(S["caught"][st == k].mean()), 4), "n": int((st == k).sum())} for k in sorted(set(st))},
            {"recall": round(float(S["caught"][S["cover"]].mean()), 4) if S["cover"].any() else None, "n": int(S["cover"].sum())})
