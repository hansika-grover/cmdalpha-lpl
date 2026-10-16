"""End-to-end evaluation (playbook section 11).

Train on the training split, calibrate and set alert thresholds on validation, report on test.
Nothing is tuned on the test split. Usage:
    python -m sentinel.eval.run_eval --world data/world --out results/metrics.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import load_labels, load_world
from ..features import money, sequences
from ..features.build import build_features, notes_group
from ..llm.keyword_reader import keyword_read, tfidf_risk
from ..models.fusion import Fusion
from . import alerts as AL
from .baselines import checklist_scores, hard_rule, threshold_scores
from .metrics import Truth, summarize, with_ci
from .splits import household_split, note_labels, point_labels

BUDGETS = {"0.05%": 0.0005, "0.1%": 0.001, "0.25%": 0.0025, "0.5%": 0.005, "1%": 0.01}
TYPOLOGIES = ["investment_romance_scam", "tech_support_impersonation", "government_impersonation",
              "silent_account_takeover", "grandparent_voice_clone", "trusted_person_theft"]


def kw_reader(W, L, split, world_dir) -> pd.DataFrame:
    cache = Path(world_dir) / "_kw_reader.pkl"
    if cache.exists() and "inj_quotes" in (kw := pd.read_pickle(cache)).columns:
        pass
    else:
        kw = keyword_read(W["notes"])
        kw.to_pickle(cache)
    notes = W["notes"].reset_index(drop=True)
    y = note_labels(notes, L["cases"])
    tr = notes["household_id"].map(split).eq("train").to_numpy()
    kw = kw.copy()
    kw["risk"] = tfidf_risk(notes, y[tr])
    return kw


# ---- model config (item 4): eval/tune.py writes config/model.yaml from validation; same params for every model spec
MODEL_CONFIG = "config/model.yaml"


def load_model_config(path=MODEL_CONFIG) -> dict | None:
    """Tuned model config, or None (then Fusion uses its defaults)."""
    if not Path(path).exists():
        return None
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_params(cfg: dict | None) -> dict | None:
    return None if cfg is None else {**(cfg.get("params") or {}), "objective": cfg["objective"]}
# ---- end model config


def sentinel_features(R) -> list[str]:
    """Sentinel's feature list; eval/tune.py tunes on exactly this."""
    C = R.feats["notes_llm"] or R.feats["notes_kw"]
    return R.feats["money"] + R.feats["sequence"] + C


def llm_reader(W, world_dir):
    """Claude notes reader output from the committed cache; None when it has not been run."""
    try:
        from ..llm.notes_reader import cached_outputs
    except ImportError:
        return None
    return cached_outputs(W)


class Run:
    def __init__(self, world_dir="data/world", seed=0, use_llm=True, verbose=True, params=None):
        t0 = time.time()
        self.world_dir, self.seed, self.verbose, self.params = world_dir, seed, verbose, params
        self.W, self.L = load_world(world_dir), load_labels(world_dir)
        self.split = household_split(self.W["households"], self.L["cases"], seed)
        readers = {"kw": kw_reader(self.W, self.L, self.split, world_dir)}
        llm = llm_reader(self.W, world_dir) if use_llm else None
        self.llm_coverage = 0.0
        if llm is not None and len(llm):
            readers["llm"] = llm
            self.llm_coverage = float(llm["note_id"].nunique() / len(self.W["notes"]))
        self.readers = readers
        tf = time.time()
        self.F = build_features(self.W, readers, verbose)
        self.feature_seconds = time.time() - tf
        self.F["split"] = self.split.loc[self.F["household_id"]].to_numpy()
        self.lab = point_labels(self.F, self.L["cases"])
        self.hh_code = self.W["hh_index"].loc[self.F["household_id"]].to_numpy()
        self.day = self.F["day"].to_numpy()
        self.mask = {s: (self.F["split"] == s).to_numpy() for s in ("train", "val", "test")}
        self.n_h = {s: int((self.split == s).sum()) for s in ("train", "val", "test")}
        self.truth = {s: Truth(self.W, self.L, list(self.split[self.split == s].index)) for s in ("val", "test")}
        self.reader = "llm" if "llm" in readers else "kw"
        self.feats = {"money": money.FEATURES, "sequence": sequences.FEATURES,
                      "notes_kw": notes_group("kw"), "notes_llm": notes_group("llm") if "llm" in readers else None}
        if verbose:
            print(f"prepared in {time.time() - t0:.0f}s; reader={self.reader}; households={self.n_h}")

    # -------------------------------------------------------------- models
    def train(self, feats, exclude_households=None, seed=0, params=None) -> Fusion:
        tr = self.mask["train"] & ~self.lab["post"].to_numpy()
        va = self.mask["val"] & ~self.lab["post"].to_numpy()
        if exclude_households is not None:
            keep = ~self.F["household_id"].isin(exclude_households).to_numpy()
            tr, va = tr & keep, va & keep
        y = self.lab["y"].to_numpy()
        return Fusion(feats, seed=seed, params=params or self.params).fit(self.F[tr], y[tr], self.F[va], y[va])

    def evaluate(self, score, force=None, budgets=BUDGETS, B=1000, exclude_val=None, split="test"):
        """Thresholds per budget from validation; metrics on `split`."""
        out = {}
        v, t = self.mask["val"], self.mask[split]
        if exclude_val is not None:
            v = v & ~self.F["household_id"].isin(exclude_val).to_numpy()
        fv = None if force is None else force[v]
        ft = None if force is None else force[t]
        n_val = self.n_h["val"] - (0 if exclude_val is None else len(set(exclude_val) & set(self.split[self.split == "val"].index)))
        for name, b in budgets.items():
            tau = AL.tau_for_budget(self.hh_code[v], self.day[v], score[v], n_val, b, fv)
            out[name] = self.metrics_at(score, tau, ft, B, split)
            out[name]["threshold"] = round(tau, 6)
        return out

    def metrics_at(self, score, tau, force_t=None, B=1000, split="test"):
        t = self.mask[split]
        idx = AL.simulate(self.hh_code[t], self.day[t], score[t], tau, force_t)
        A = AL.alerts_frame(self.F[t], idx, score[t])
        S = summarize(A, self.truth[split])
        return with_ci(S, B=B, seed=self.seed)

    def natural_rate(self, score, split="val"):
        m = self.mask[split]
        idx = AL.simulate(self.hh_code[m], self.day[m], score[m], 0.5)
        return AL.weekly_rate(len(idx), self.n_h[split])


def reliability(p, y, edges=(0, .02, .05, .1, .2, .3, .5, .7, .9, 1.0001)):
    bins = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        bins.append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()), "mean_predicted": round(float(p[m].mean()), 4),
                     "observed": round(float(y[m].mean()), 4)})
        ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return {"bins": bins, "ece": round(float(ece), 5)}


def main(world_dir="data/world", out="results/metrics.json", seed=0, B=1000, loto=True, use_llm=True, save_model=True, robust=True):
    t_all = time.time()
    cfg = load_model_config()
    R = Run(world_dir, seed, use_llm, params=config_params(cfg))
    res = {"meta": {"world": world_dir, "seed": seed, "households": R.n_h, "notes_reader": R.reader,
                    "llm_note_coverage": round(R.llm_coverage, 4),
                    "budgets_are": "weekly share of households sent to review",
                    "base_rate_planted": round(len(R.L["cases"]) / len(R.W["households"]), 4),
                    "model_config": cfg or "defaults (no config/model.yaml)"},
           "methods": {}}
    M = res["methods"]
    # ---- today's methods
    for lim in (100000, 50000, 25000):
        s = threshold_scores(R.F, lim)
        M[f"threshold_{lim // 1000}k"] = {"label": f"Threshold rule: any withdrawal of ${lim:,} or more",
                                          "natural": R.metrics_at(s, 0.5, None, B)}
    s_chk = checklist_scores(R.F)
    M["expert_checklist"] = {"label": "Expert rule checklist (no notes)", "budgets": R.evaluate(s_chk, B=B)}
    # ---- models
    C = R.feats["notes_llm"] or R.feats["notes_kw"]
    specs = {
        "money_only": ("Money movement only (engine A)", R.feats["money"], False),
        "rules_plus_keyword_notes": ("Engines A + B + keyword notes reader", R.feats["money"] + R.feats["sequence"] + R.feats["notes_kw"], True),
        "sentinel": (f"Sentinel: A + B + {'Claude' if R.reader == 'llm' else 'keyword (Claude not run)'} notes reader, fused and calibrated",
                     sentinel_features(R), True),
        "ablation_no_notes": ("Sentinel without notes", R.feats["money"] + R.feats["sequence"], True),
        "ablation_no_sequences": ("Sentinel without account-change sequences", R.feats["money"] + C, False),
        "ablation_no_money": ("Sentinel without money movement", R.feats["sequence"] + C, True),
    }
    if R.reader != "llm":
        specs.pop("rules_plus_keyword_notes")  # identical to sentinel when Claude has not been run
    models, scores = {}, {}
    for key, (label, feats, use_hard) in specs.items():
        t = time.time()
        m = R.train(feats, seed=seed)
        s = m.raw(R.F)
        force = hard_rule(R.F) if use_hard else None
        models[key], scores[key] = m, (s, force)
        M[key] = {"label": label, "budgets": R.evaluate(s, force, B=B)}
        if R.verbose:
            b = M[key]["budgets"]["0.25%"]
            print(f"{key:28s} recall@0.25%={b['recall']['value']:.3f} fa={b['lookalike_false_alarm_rate']['value']:.3f} ({time.time() - t:.0f}s)")
    # ---- same workload as the $100k threshold rule
    rate = R.natural_rate(threshold_scores(R.F, 100000))
    matched = {"weekly_rate": rate}
    for key in ("expert_checklist", "money_only", "sentinel"):
        s, force = (s_chk, None) if key == "expert_checklist" else scores[key]
        matched[key] = R.evaluate(s, force, budgets={"matched": rate}, B=B)["matched"]
    matched["threshold_100k"] = M["threshold_100k"]["natural"]
    res["matched_workload_100k"] = matched
    # ---- calibration and drivers
    sm = models["sentinel"]
    if save_model:
        run_dir = Path(world_dir) / "_run"
        run_dir.mkdir(exist_ok=True)
        with open(run_dir / "sentinel.pkl", "wb") as f:
            pickle.dump({"model": sm, "reader": R.reader, "split": R.split,
                         "taus": {b: M["sentinel"]["budgets"][b]["threshold"] for b in BUDGETS}}, f)
    t = R.mask["test"] & ~R.lab["post"].to_numpy()
    p = sm.calibrated(scores["sentinel"][0][t])
    res["calibration"] = reliability(p, R.lab["y"].to_numpy()[t])
    imp = pd.Series(sm.model.booster_.feature_importance("gain"), index=sm.features).sort_values(ascending=False)
    res["feature_importance_gain"] = {k: round(float(v), 1) for k, v in imp.head(25).items()}
    # ---- leave one typology out
    if loto:
        res["leave_one_typology_out"] = {}
        feats = specs["sentinel"][1]
        thr = threshold_scores(R.F, 100000)
        for typ in TYPOLOGIES:
            held = set(R.L["cases"].loc[R.L["cases"]["typology"] == typ, "household_id"])
            m = R.train(feats, exclude_households=held, seed=seed)
            s = m.raw(R.F)
            ev = R.evaluate(s, hard_rule(R.F), budgets={k: BUDGETS[k] for k in ("0.25%", "0.5%", "1%")}, B=200, exclude_val=held)
            base = R.metrics_at(thr, 0.5, None, 200)
            res["leave_one_typology_out"][typ] = {
                "n_test_cases": ev["0.5%"]["by_typology"].get(typ, {}).get("n", 0),
                "sentinel_recall": {b: ev[b]["by_typology"].get(typ, {}).get("recall") for b in ev},
                "sentinel_trained_on_all_recall": {b: M["sentinel"]["budgets"][b]["by_typology"].get(typ, {}).get("recall") for b in ("0.25%", "0.5%", "1%")},
                "threshold_100k_recall": base["by_typology"].get(typ, {}).get("recall"),
            }
            if R.verbose:
                print(f"LOTO {typ:28s} {res['leave_one_typology_out'][typ]['sentinel_recall']}")
    # ---- robustness (same trained model, same validation thresholds)
    if robust:
        from . import robustness
        def readers_fn(W2):
            keep = set(W2["notes"]["note_id"])
            return {k: v[v["note_id"].isin(keep)] for k, v in R.readers.items()}
        taus = {b: M["sentinel"]["budgets"][b]["threshold"] for b in ("0.25%", "1%")}
        rob = robustness.run(R, sm, readers_fn, taus)
        rob["baseline"] = {b: {k: M["sentinel"]["budgets"][b][k] for k in robustness.KEYS} for b in taus}
        rob["recall_by_advisor_style_at_0.25%"], rob["coached_cover_story_at_0.25%"] = \
            robustness.by_style_and_cover(R, scores["sentinel"], taus["0.25%"])
        res["robustness"] = rob
        if R.verbose:
            print("robustness recall@0.25%", {k: v["0.25%"]["recall"]["value"] for k, v in rob.items() if k in ("baseline", "no_notes", "half_notes", "noisy_account_events")})
    # ---- cost and speed (engines A-C, excluding the one-off notes reading)
    n_hh = len(R.W["households"])
    t = time.time()
    sm.raw(R.F)
    res["cost"] = {"feature_seconds_all_households": round(R.feature_seconds, 1),
                   "score_seconds_all_points": round(time.time() - t, 2),
                   "ms_per_household_scored": round(1000 * (R.feature_seconds + time.time() - t) / n_hh, 3),
                   "scoring_points": int(len(R.F))}
    res["meta"]["runtime_seconds"] = round(time.time() - t_all)
    tuning = Path(out).parent / "tuning.json"  # validation-only model selection, kept across re-runs
    if Path(out).name == "metrics.json" and tuning.exists():
        res["tuning"] = json.loads(tuning.read_text(encoding="utf-8"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {out} in {time.time() - t_all:.0f}s")
    return res, R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="data/world")
    ap.add_argument("--out", default="results/metrics.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--no-loto", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-robustness", action="store_true")
    a = ap.parse_args()
    main(a.world, a.out, a.seed, a.bootstrap, not a.no_loto, not a.no_llm, robust=not a.no_robustness)
