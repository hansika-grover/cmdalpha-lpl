"""Choose Sentinel's feature blocks and settings on VALIDATION, pooled over the 5 worlds. Never touches test.

Candidates, fixed before running: feature blocks {v1: money + sequences + notes, + memory, + network,
+ both} x settings {LightGBM defaults, the tuned config from eval/tune.py} = 8. Each is trained on a
world's train split; thresholds and metrics both come from its validation split. The winner has the
highest mean validation recall at the 0.25% weekly budget across worlds (ties: lower mean look-alike
false-alarm rate, then fewer blocks, then default settings). The winner is written to config/model.yaml,
which run_eval then uses; everything else is reported as an ablation.

Added on 16 Oct after a test run showed network + memory + tuned settings did not beat v1: the rule
above was declared first and is applied mechanically, so the choice is made on validation, not test.

    python -m sentinel.eval.select --world data/world --seed 0        # one world -> results/_select_0.json
    python -m sentinel.eval.select --summarize --seeds 0 1 2 3 4      # pool -> results/selection.json + config/model.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .baselines import hard_rule
from .run_eval import MODEL_CONFIG, Run, block_features

SELECTED_BY = "mean validation recall at 0.25% weekly budget over 5 worlds"
BUDGETS = {"0.25%": 0.0025, "1%": 0.01}
FEATURE_SETS = {"v1": ["money", "sequence", "notes"], "v1+memory": ["money", "sequence", "notes", "memory"],
                "v1+network": ["money", "network", "sequence", "notes"],
                "v1+network+memory": ["money", "network", "sequence", "notes", "memory"]}
KEYS = ("recall", "caught_before_money_left", "lookalike_false_alarm_rate", "dollars_protected_share")


def settings() -> dict:
    """Default LightGBM settings, and the config eval/tune.py chose (from results/tuning.json, not the live config)."""
    t = json.loads(Path("results/tuning.json").read_text(encoding="utf-8"))["chosen"]
    return {"default": None, "tuned": {**t["params"], "objective": t["objective"]}}


def world_dir(seed: int) -> str:
    return "data/world" if seed == 0 else f"data/world_s{seed}"


def run_world(seed: int, out: str | None = None) -> dict:
    t0 = time.time()
    R = Run(world_dir(seed), seed)
    force = hard_rule(R.F)
    rows = []
    for fs, blocks in FEATURE_SETS.items():
        for st, params in settings().items():
            t = time.time()
            m = R.train(block_features(R, blocks), seed=seed, params=params)
            ev = R.evaluate(m.raw(R.F), force, budgets=BUDGETS, B=0, split="val")
            rows.append({"features": fs, "settings": st, "val": {b: {k: ev[b][k]["value"] for k in KEYS} for b in ev},
                         "seconds": round(time.time() - t, 1)})
            print(f"world {seed} {fs:18s} {st:7s} val recall@0.25%={rows[-1]['val']['0.25%']['recall']:.3f} "
                  f"fa={rows[-1]['val']['0.25%']['lookalike_false_alarm_rate']}")
    res = {"seed": seed, "world": world_dir(seed), "rows": rows, "seconds": round(time.time() - t0)}
    Path(out or f"results/_select_{seed}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def pick(table: list[dict]) -> dict:
    def key(r):
        return (-r["recall_0.25%"], r["false_alarms_0.25%"], len(FEATURE_SETS[r["features"]]), r["settings"] != "default")
    return min(table, key=key)


def summarize(seeds, out="results/selection.json", config=MODEL_CONFIG) -> dict:
    runs = [json.loads(Path(f"results/_select_{s}.json").read_text(encoding="utf-8")) for s in seeds]
    table = []
    for fs in FEATURE_SETS:
        for st in ("default", "tuned"):
            v = [next(r["val"] for r in w["rows"] if r["features"] == fs and r["settings"] == st) for w in runs]
            m = lambda b, k: [x[b][k] for x in v]
            table.append({"features": fs, "settings": st, "recall_0.25%": round(float(np.mean(m("0.25%", "recall"))), 4),
                          "recall_0.25%_per_world": m("0.25%", "recall"),
                          "false_alarms_0.25%": round(float(np.mean(m("0.25%", "lookalike_false_alarm_rate"))), 4),
                          "before_money_0.25%": round(float(np.mean(m("0.25%", "caught_before_money_left"))), 4),
                          "recall_1%": round(float(np.mean(m("1%", "recall"))), 4)})
    best = pick(table)
    st = settings()[best["settings"]]
    chosen = {"objective": (st or {}).get("objective", "binary"),
              "params": {k: v for k, v in (st or {}).items() if k != "objective"},
              "features": FEATURE_SETS[best["features"]], "feature_set": best["features"], "settings": best["settings"],
              "selected_by": SELECTED_BY, "chosen_at": dt.datetime.now().isoformat(timespec="seconds")}
    res = {"selected_by": SELECTED_BY, "split_used": "val only (test untouched)", "worlds": list(seeds),
           "note": "Selection step added on 16 Oct after a test run; candidates and rule fixed before it ran.",
           "candidates": table, "chosen": {**chosen, "val": best}}
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    Path(config).write_text(yaml.safe_dump(chosen, sort_keys=False), encoding="utf-8")
    for r in sorted(table, key=lambda r: -r["recall_0.25%"]):
        print(f"{r['features']:18s} {r['settings']:7s} val recall@0.25% {r['recall_0.25%']:.4f} fa {r['false_alarms_0.25%']:.4f} "
              f"before {r['before_money_0.25%']:.4f} recall@1% {r['recall_1%']:.4f}")
    print(f"chosen: {best['features']} / {best['settings']} -> {config}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    summarize(a.seeds) if a.summarize else run_world(a.seed)
