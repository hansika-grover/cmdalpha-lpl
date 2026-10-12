import argparse
import json

from .world import write_world

ap = argparse.ArgumentParser(description="Generate the synthetic Sentinel world")
ap.add_argument("--out", default="data/world")
ap.add_argument("--config", default="config/world.yaml")
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--households", type=int, default=None)
a = ap.parse_args()
print(json.dumps(write_world(a.out, a.config, a.seed, a.households), indent=1))
