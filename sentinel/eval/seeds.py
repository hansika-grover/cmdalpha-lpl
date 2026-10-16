"""Repeat the evaluation on independent worlds (different generator seeds) for stability.

Seed 0 is data/world (results/metrics.json); seed k is data/world_sk (results/seed_k.json). A seed's results file is
reused when it is newer than its world's manifest.json (results/metrics.json is always reused if present: it comes from
`make eval`), so worlds can be run one per process and summarized after:

    python -m sentinel.eval.seeds                       # run (or reuse) seeds 0-4, write results/seeds.json
    python -m sentinel.eval.seeds --one 2               # run seed 2 only; results/seeds.json is left alone
    python -m sentinel.eval.seeds --summarize           # aggregate existing files only; error if one is missing

results/seeds.json is written only by a full run of 2+ seeds or by --summarize, never by a single-seed run.
`make seeds-parallel` runs seeds 1-4 as 4 processes, then --summarize.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..generate.world import load_config, write_world
from .run_eval import TYPOLOGIES
from .run_eval import main as run_eval

KEYS = ["recall", "caught_before_money_left", "dollars_protected_share", "lookalike_false_alarm_rate",
        "precision_at_0.5pct_base_rate"]
METHODS = {"threshold_100k": ("natural",), "expert_checklist": ("0.25%", "1%"), "money_only": ("0.25%", "1%"),
           "sentinel": ("0.25%", "1%")}


def world_dir(s):
    return "data/world" if s == 0 else f"data/world_s{s}"


def result_path(s):
    return "results/metrics.json" if s == 0 else f"results/seed_{s}.json"


def fresh(s, d=None, p=None):
    """The seed's results file exists and is newer than its world and than config/model.yaml."""
    d, p = Path(d or world_dir(s)), Path(p or result_path(s))
    m, cfg = d / "manifest.json", Path("config/model.yaml")
    return (p.exists() and m.exists() and p.stat().st_mtime >= m.stat().st_mtime
            and (not cfg.exists() or p.stat().st_mtime >= cfg.stat().st_mtime))


def run_seed(s, households=None, d=None, p=None):
    d, p = d or world_dir(s), p or result_path(s)
    if fresh(s, d, p) or (s == 0 and Path(p).exists()):  # never overwrite the full metrics.json with a B=200 run
        if not fresh(s, d, p):
            print(f"warning: {p} is older than {d}/manifest.json; re-run `make eval`")
        return json.loads(Path(p).read_text(encoding="utf-8"))
    if not Path(d, "manifest.json").exists():
        write_world(d, seed=load_config()["seed"] + 1000 * s, households=households)
    return run_eval(d, p, seed=s, B=200, loto=False, save_model=False, robust=False)[0]


def load_seed(s):
    p = Path(result_path(s))
    if not p.exists():
        raise SystemExit(f"seeds --summarize: {p} is missing; run `python -m sentinel.eval.seeds --one {s}` first")
    if not fresh(s):
        print(f"warning: {p} is older than {world_dir(s)}/manifest.json")
    return json.loads(p.read_text(encoding="utf-8"))


def _node(r, m, b):
    return r["methods"][m]["natural"] if b == "natural" else r["methods"][m]["budgets"][b]


def _num(x):
    x = x.get("value") if isinstance(x, dict) else x
    return np.nan if x is None else float(x)


def _stats(v):
    v = np.array(v, dtype=float)
    ok = ~np.isnan(v)
    r = (lambda f: round(float(f(v[ok])), 4) if ok.any() else None)
    return {"mean": r(np.mean), "sd": r(np.std), "min": r(np.min), "max": r(np.max),
            "per_seed": [None if np.isnan(x) else round(float(x), 4) for x in v]}


def summarize(runs: dict) -> dict:
    """runs: {seed: run_eval result}. Headline metrics and per-typology recall, across seeds."""
    summary = {"seeds": list(runs), "note": "seed 0 is data/world; seeds 1-4 are independent worlds",
               "methods": {}, "by_typology": {}}
    for m, budgets in METHODS.items():
        summary["methods"][m], summary["by_typology"][m] = {}, {}
        for b in budgets:
            nodes = [_node(r, m, b) for r in runs.values()]
            summary["methods"][m][b] = {k: _stats([_num(n[k]) for n in nodes]) for k in KEYS}
            summary["by_typology"][m][b] = {
                t: {**_stats([_num(n.get("by_typology", {}).get(t, {}).get("recall")) for n in nodes]),
                    "n": [int(n.get("by_typology", {}).get(t, {}).get("n", 0)) for n in nodes]}
                for t in TYPOLOGIES}
    return summary


def main(seeds, households=None, out="results/seeds.json", summarize_only=False, write=True):
    runs = {s: load_seed(s) if summarize_only else run_seed(s, households) for s in seeds}
    if not write:
        return runs
    summary = summarize(runs)
    Path(out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({m: summary["methods"][m] for m in ("sentinel",)}, indent=1)[:3000])
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--one", type=int, default=None, help="run one seed only; do not write seeds.json")
    ap.add_argument("--summarize", action="store_true", help="aggregate existing results files only")
    ap.add_argument("--households", type=int, default=None)
    a = ap.parse_args()
    seeds = [a.one] if a.one is not None else a.seeds
    main(seeds, a.households, summarize_only=a.summarize, write=a.summarize or (a.one is None and len(seeds) > 1))
