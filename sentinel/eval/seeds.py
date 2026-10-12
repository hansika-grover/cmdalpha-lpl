"""Repeat the evaluation on independent worlds (different generator seeds) for stability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..generate.world import load_config, write_world
from .run_eval import main as run_eval

KEYS = ["recall", "caught_before_money_left", "dollars_protected_share", "lookalike_false_alarm_rate",
        "precision_at_0.5pct_base_rate"]


def main(seeds, households=None, out="results/seeds.json"):
    base = load_config()["seed"]
    runs = {}
    for s in seeds:
        d = "data/world" if s == 0 else f"data/world_s{s}"
        if not Path(d, "manifest.json").exists():
            write_world(d, seed=base + 1000 * s, households=households)
        p = "results/metrics.json" if s == 0 else f"results/seed_{s}.json"
        if s == 0 and Path(p).exists():
            runs[s] = json.loads(Path(p).read_text())
        else:
            runs[s] = run_eval(d, p, seed=s, B=200, loto=False, save_model=False, robust=False)[0]
    summary = {"seeds": list(seeds), "note": "seed 0 is data/world; seeds 1-4 are independent worlds", "methods": {}}
    for m in ("threshold_100k", "expert_checklist", "money_only", "sentinel"):
        summary["methods"][m] = {}
        for b in (("natural",) if m == "threshold_100k" else ("0.25%", "1%")):
            vals = {k: [] for k in KEYS}
            for r in runs.values():
                node = r["methods"][m]["natural"] if b == "natural" else r["methods"][m]["budgets"][b]
                for k in KEYS:
                    vals[k].append(np.nan if node[k]["value"] is None else node[k]["value"])
            summary["methods"][m][b] = {k: {"mean": round(float(np.nanmean(v)), 4), "sd": round(float(np.nanstd(v)), 4),
                                            "min": round(float(np.nanmin(v)), 4), "max": round(float(np.nanmax(v)), 4),
                                            "per_seed": [round(float(x), 4) for x in v]}
                                        for k, v in vals.items()}
    Path(out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({m: summary["methods"][m] for m in ("sentinel",)}, indent=1)[:3000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--households", type=int, default=None)
    a = ap.parse_args()
    main(a.seeds, a.households)
