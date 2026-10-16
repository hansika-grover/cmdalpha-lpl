"""Model selection on validation (PLAN item 4). Never touches the test split.

Grid: objective {binary, lambdarank} x depth {3, 4} x trees {300, 600} x learning rate {0.03, 0.05}
x class weight {None, balanced} (binary only) = 24 configs. Each is trained on train, thresholds and
metrics both come from validation; the winner is the highest validation recall at the 0.25% weekly
budget (ties: lower look-alike false-alarm rate, then simpler). Writes results/tuning.json and
config/model.yaml, which run_eval then uses for every model spec. Usage:
    python -m sentinel.eval.tune --world data/world [--quick]
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import time
from pathlib import Path

import yaml

from .baselines import hard_rule
from .run_eval import MODEL_CONFIG, Run, sentinel_features

SELECTED_BY = "validation recall at 0.25% weekly budget"
TUNE_BUDGETS = {"0.25%": 0.0025, "1%": 0.01}


def grid(quick=False) -> list[dict]:
    depths, trees, rates = ((3, 4), (300,), (0.05,)) if quick else ((3, 4), (300, 600), (0.03, 0.05))
    out = []
    for obj, d, n, lr in itertools.product(("binary", "lambdarank"), depths, trees, rates):
        for cw in ((None, "balanced") if obj == "binary" else (None,)):
            out.append({"objective": obj, "params": {"max_depth": d, "num_leaves": 2 ** d, "n_estimators": n,
                                                     "learning_rate": lr, "class_weight": cw}})
    return out


def simplicity(c: dict) -> tuple:
    p = c["params"]
    return (p["max_depth"], p["n_estimators"], c["objective"] != "binary", p.get("class_weight") is not None, -p["learning_rate"])


def select(rows: list[dict]) -> dict:
    """Max val recall@0.25%; ties: lower val look-alike false-alarm rate, then simpler."""
    def key(r):
        fa = r["val"]["0.25%"]["lookalike_false_alarm_rate"]
        return (-r["val"]["0.25%"]["recall"], 1.0 if fa is None else fa, simplicity(r))
    return min(rows, key=key)


def main(world_dir="data/world", quick=False, seed=0, B=0, out="results/tuning.json", config=MODEL_CONFIG):
    t_all = time.time()
    R = Run(world_dir, seed)
    feats = sentinel_features(R)
    force = hard_rule(R.F)
    rows = []
    for c in grid(quick):
        t = time.time()
        m = R.train(feats, seed=seed, params={**c["params"], "objective": c["objective"]})
        ev = R.evaluate(m.raw(R.F), force, budgets=TUNE_BUDGETS, B=B, split="val")
        val = {b: {k: ev[b][k]["value"] for k in ("recall", "caught_before_money_left", "lookalike_false_alarm_rate",
                                                   "precision", "dollars_protected_share")} for b in ev}
        rows.append({**c, "val": val, "seconds": round(time.time() - t, 1)})
        print(f"{c['objective']:10s} d={c['params']['max_depth']} n={c['params']['n_estimators']:3d} "
              f"lr={c['params']['learning_rate']} cw={c['params']['class_weight']}: "
              f"val recall@0.25%={val['0.25%']['recall']:.3f} fa={val['0.25%']['lookalike_false_alarm_rate']} "
              f"recall@1%={val['1%']['recall']:.3f} ({rows[-1]['seconds']}s)")
    best = select(rows)
    chosen = {"objective": best["objective"], "params": best["params"], "selected_by": SELECTED_BY,
              "chosen_at": dt.datetime.now().isoformat(timespec="seconds")}
    res = {"world": world_dir, "seed": seed, "split_used": "val (train fit, val thresholds and metrics; test untouched)",
           "selected_by": SELECTED_BY, "quick": quick, "features": feats, "notes_reader": R.reader,
           "configs": rows, "chosen": {**chosen, "val": best["val"]}, "runtime_seconds": round(time.time() - t_all)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    Path(config).parent.mkdir(parents=True, exist_ok=True)
    Path(config).write_text(yaml.safe_dump(chosen, sort_keys=False), encoding="utf-8")
    metrics = Path(out).parent / "metrics.json"
    if metrics.exists():
        M = json.loads(metrics.read_text(encoding="utf-8"))
        M["tuning"] = {k: res[k] for k in ("world", "selected_by", "quick", "chosen", "runtime_seconds")} | {
            "n_configs": len(rows), "table": [{k: r[k] for k in ("objective", "params", "val")} for r in rows]}
        metrics.write_text(json.dumps(M, indent=1), encoding="utf-8")
    print(f"chosen {chosen['objective']} {chosen['params']}; wrote {out} and {config} in {time.time() - t_all:.0f}s")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="data/world")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--out", default="results/tuning.json")
    ap.add_argument("--config", default=MODEL_CONFIG)
    a = ap.parse_args()
    main(a.world, a.quick, a.seed, a.bootstrap, a.out, a.config)
