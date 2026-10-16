"""Export the trained model for the browser scenario builder (web/data/model.json and friends).

Scoring in JS (reference: tests/test_model_export.py):
  1. x = features in `features` order; apply Math.fround to each value (Python scores float32 inputs).
  2. margin = sum over `trees` of the leaf reached by walking `tree_structure`: at a split node read
     v = x[split_feature]; if v is NaN and missing_type != "NaN", v = 0; if (missing_type == "NaN" and v is NaN)
     or (missing_type == "Zero" and |v| <= 1e-35) go default_left, else go left iff v <= threshold
     (decision_type is always "<="). A node with `leaf_value` is a leaf.
  3. proba = 1 / (1 + exp(-margin)). This is Fusion.raw() (predict_proba) and the scale of the alert threshold.
  4. cal = isotonic(proba): linear interpolation in `isotonic.x` -> `isotonic.y`, clipped to the end values
     (sklearn IsotonicRegression(out_of_bounds="clip").predict). The isotonic table maps from proba, not margin.
  A point alerts when proba >= review_line.raw (equivalently cal >= review_line.cal); review_line.margin is the same
  line on the margin scale.

Presets: each showcase household's feature vector at its alert (or as-of) point, plus a `baseline` for a new
withdrawal at that moment (same formulas as features/money.py:enrich_txns): base_med and scale are the household's
12-month median and robust scale (peer band values when it has < 3 prior withdrawals; `thin` says which), prior_max is
the largest prior withdrawal (peer 95th percentile when none), typical is the monthly outflow over the prior 11 months.

event_effects: what one new event, at the same moment as the preset point, does to the feature vector. Each effect is
{feature, op, value}: "set" -> x[feature] = value; "add" -> x[feature] += value; "formula" -> x[feature] = value, a JS
expression over `amount` (the withdrawal), `f` (this feature's value), `x` (the whole vector before this event: all
effects of one event read the same x), `b` (the preset baseline), and helpers clip(v, lo, hi), log1p, expm1, min, max.
Apply events in order, each to the result of the previous one. Time does not pass between events.
  * account events (email_change, ...): cnt30_<type> += 1, since_<type> = 0, and the new scoring point is not a
    withdrawal, so per-withdrawal features (amount, payee, chain flags) are reset to 0; 30/90-day windows are kept.
  * note_plain / note_flag_<i> (a note whose reader output is category i) / note_extra_flag_<i> (a further category
    on the same note) / note_mitigating (mitigating context on the same note). Categories are in
    keyword_reader.categories; run the lexicon in JS (strip injection matches first) to choose the events.
  * withdrawal_new_payee: a withdrawal of `amount` to a payee created today; then optionally payee_kind_<kind> and/or
    method_wire for the same withdrawal. The payee is assumed not to be the trusted contact, POA or joint owner.
Only features in `features` are listed. The notes TF-IDF risk features (kw_risk_*) are never changed: the browser
cannot run that model. Household-memory features (engines fired, score history) and mule-network features (other
clients paying the same account) are also held at the preset's values, so a builder event that would add to them
under-scores slightly.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..features import money, sequences
from ..features.notes import FLAG_COLS
from ..generate.text import CATEGORIES
from ..llm import keyword_reader as KW
from ..llm import notes_reader as NR
from ..models.fusion import monotone
from ..models.reasons import family
from .build import WEB, Exporter, write

ACCOUNT_EVENTS = ["email_change", "phone_change", "bank_link_added", "new_device", "password_reset", "poa_added",
                  "joint_owner_added", "trusted_contact_changed"]
OUT_ONLY = ["ato_chain", "ato_chain_young_payee", "access_then_new_payee", "tc_change_then_out", "poa_then_out",
            "payee_is_trusted_contact", "payee_is_poa_or_joint", "payee_is_recent_poa"]
F32 = lambda v: float(np.float32(v))


def _plus(col):
    return f"log1p(expm1(x.{col}) + amount)"


def event_effects(features: list[str]) -> dict:
    p = next((f.split("_", 1)[0] for f in features if f.endswith("_notes_90d")), "kw")
    reset = [(f, "set", 0) for f in money.TXN_FEATURES + OUT_ONLY]
    E = {}
    for t in ACCOUNT_EVENTS:
        E[t] = reset + [(f"cnt30_{t}", "add", 1), (f"since_{t}", "set", 0)]
        if t in ("email_change", "phone_change"):
            E[t].append(("days_since_contact_change", "set", 0))
    note = reset + [(f"{p}_notes_90d", "add", 1)]
    E["note_plain"] = note
    for i, c in enumerate(FLAG_COLS):
        extra = [(f"{p}_n30_{c}", "add", 1), (f"{p}_n90_{c}", "add", 1),
                 (f"{p}_distinct_cats_90d", "formula", f"x.{p}_n90_{c} > 0 ? f : f + 1")]
        E[f"note_flag_{i}"] = note + extra + [(f"{p}_flag_notes_30d", "add", 1), (f"{p}_flag_notes_90d", "add", 1),
                                              (f"{p}_days_since_flag_note", "set", 0)]
        E[f"note_extra_flag_{i}"] = extra
    E["note_mitigating"] = [(f"{p}_mitigating_90d", "add", 1)]
    E["withdrawal_new_payee"] = reset + [
        ("is_outflow", "set", 1), ("amt_log", "formula", "log1p(amount)"),
        ("robust_z", "formula", "clip((amount - b.base_med) / b.scale, -5, 50)"),
        ("ratio_max", "formula", "clip(amount / b.prior_max, 0, 100)"), ("share_assets", "formula", "amount / b.assets"),
        ("payee_new", "set", 1), ("young_payee", "set", 1), ("thin_history", "formula", "b.thin"),
        ("out_cnt_7d", "add", 1), ("out_cnt_30d", "add", 1), ("out_sum_7d", "formula", _plus("out_sum_7d")),
        ("out_sum_30d", "formula", _plus("out_sum_30d")),
        ("out_30d_vs_typical", "formula", "clip((expm1(x.out_sum_30d) + amount) / max(b.typical, 500), 0, 200)"),
        ("young_payee_sum_30d", "formula", _plus("young_payee_sum_30d")),
        ("nonrecurring_sum_30d", "formula", _plus("nonrecurring_sum_30d")),
        ("struct_cnt_60d", "formula", "f + (amount >= 8000 && amount < 10000 ? 1 : 0)"),
        ("share_out_30d", "formula", "clip((expm1(x.out_sum_30d) + amount) / b.assets, 0, 5)"),
        ("ato_chain", "formula", "x.days_since_contact_change <= 30 && x.since_bank_link_added <= 30 ? 1 : 0"),
        ("ato_chain_young_payee", "formula", "x.days_since_contact_change <= 30 && x.since_bank_link_added <= 30 ? 1 : 0"),
        ("access_then_new_payee", "formula", "min(x.since_password_reset, x.since_new_device) <= 14 ? 1 : 0"),
        ("tc_change_then_out", "formula", "min(x.since_trusted_contact_changed, x.since_trusted_contact_removed) <= 30 ? 1 : 0"),
        ("poa_then_out", "formula", "min(x.since_poa_added, x.since_joint_owner_added) <= 90 ? 1 : 0")]
    E["payee_kind_crypto_exchange"] = [("payee_crypto", "set", 1), ("crypto_sum_30d", "formula", _plus("crypto_sum_30d"))]
    E["payee_kind_individual"] = [("payee_individual", "set", 1), ("individual_sum_30d", "formula", _plus("individual_sum_30d"))]
    E["payee_kind_external_bank"] = [("payee_ext_bank", "set", 1)]
    E["method_wire"] = [("method_wire", "set", 1)]
    have = set(features)
    ok = lambda f, v: f in have and all(r in have for r in re.findall(r"\bx\.(\w+)", str(v)))
    out = {}
    for k, effs in E.items():
        last = {f: (op, v) for f, op, v in effs if ok(f, v)}  # a later effect on the same feature wins (reset, then set)
        out[k] = [{"feature": f, "op": op, "value": v} for f, (op, v) in last.items()]
    return out


def baselines(W, h: str, ts) -> dict:
    """Baseline for a new withdrawal by household h at ts (records at or before ts are prior)."""
    mm = W["money_movements"]
    a = mm.loc[(mm["household_id"] == h) & (mm["ts"] <= ts)]
    yr = a.loc[a["ts"] > ts - pd.Timedelta(days=365), "amount"]
    hh = W["households"].set_index("household_id")
    assets = float(hh.loc[h, "assets"])
    early = mm[mm["day"] < money.PEER_DAYS]
    band = pd.cut(hh.loc[early["household_id"], "assets"], money.ASSET_BANDS, labels=False).to_numpy()
    peer = early["amount"][band == pd.cut([assets], money.ASSET_BANDS, labels=False)[0]]
    thin = len(yr) < 3
    s = peer if thin else yr
    med, iqr = float(s.median()), float(s.quantile(.75) - s.quantile(.25))
    old = a.loc[(a["ts"] > ts - pd.Timedelta(days=365)) & (a["ts"] <= ts - pd.Timedelta(days=30)), "amount"]
    return {"base_med": med, "scale": max(iqr / 1.349, 0.1 * med, 50.0),
            "prior_max": max(float(a["amount"].max()), 1.0) if len(a) else float(peer.quantile(.95)),
            "assets": assets, "typical": float(old.sum()) / 11, "thin": int(thin)}


def export_model(world="data/world", exporter=None):
    ex = exporter or Exporter(world)
    m, F = ex.model, ex.Fi
    feats = m.features
    booster = m.model.booster_
    dump = booster.dump_model()
    X = F[feats].to_numpy(np.float32)
    margin = booster.predict(X, raw_score=True)
    proba, cal = np.asarray(ex.raw), np.asarray(ex.cal)
    if not str(dump["objective"]).startswith("binary"):
        raise SystemExit(f"model.json assumes a binary model (proba = sigmoid(margin)); got {dump['objective']}")
    assert np.allclose(1 / (1 + np.exp(-margin)), proba, atol=1e-9)
    tau = float(ex.taus["0.25%"])
    vec = lambda i: {f: F32(v) for f, v in zip(feats, X[i])}
    presets, points = {}, []
    for key, v in ex.pick_showcase().items():
        if not v:
            continue
        h, d = v
        rows = ex.by_h[h][F.loc[ex.by_h[h], "day"].to_numpy() <= d + 1e-6]
        i = int(rows[-1])
        points.append(i)
        presets[key] = {"household_id": h, "ref_id": F.loc[i, "ref_id"], "day": float(F.loc[i, "day"]), "features": vec(i),
                        "baseline": baselines(ex.W, h, F.loc[i, "ts"]), "raw": float(proba[i]), "cal": float(cal[i])}
    model = {"version": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}-{hashlib.sha256(','.join(feats).encode()).hexdigest()[:8]}",
             "objective": dump["objective"], "doc": __doc__,
             "features": feats, "monotone": monotone(feats), "trees": dump["tree_info"],
             "isotonic": {"x": m.iso.X_thresholds_.tolist(), "y": m.iso.y_thresholds_.tolist()},
             "review_line": {"raw": tau, "cal": float(m.calibrated(np.array([tau]))[0]), "margin": float(np.log(tau / (1 - tau)))},
             "families": {f: family(f) for f in feats},
             "keyword_reader": {"categories": CATEGORIES, "lexicon": KW.LEXICON, "mitigating": KW.MITIGATING,
                                "injection": KW.INJECTION, "flags": "i",
                                "rules": "Remove injection matches from the text, then test each lexicon regex (index = category) and the mitigating regex."},
             "presets": presets, "event_effects": event_effects(feats)}
    # check rows: showcase points, the highest scores, and a random spread
    rng = np.random.default_rng(0)
    top = [i for i in np.argsort(-proba)[:40] if i not in points][:25]
    rest = [i for i in rng.permutation(len(F)) if i not in points and i not in top][:50 - len(points) - len(top)]
    check = {"note": "raw = booster margin (sum of leaf values); proba = sigmoid(raw) = Fusion.raw(); "
                     "cal = isotonic(proba). The isotonic table maps from proba, not raw. Match each to 1e-6.",
             "rows": [{"features": vec(i), "raw": float(margin[i]), "proba": float(proba[i]), "cal": float(cal[i])}
                      for i in points + top + rest]}
    prompt = {"model": NR.MODEL, "system": NR.SYSTEM, "schema": NR.SCHEMA, "categories": CATEGORIES}
    WEB.mkdir(parents=True, exist_ok=True)
    write(WEB / "model.json", model)
    write(WEB / "model_check.json", check)
    write(WEB / "reader_prompt.json", prompt)
    print(f"model.json: {len(feats)} features, {len(dump['tree_info'])} trees, presets={list(presets)}; model_check: {len(check['rows'])} rows")
    return model
