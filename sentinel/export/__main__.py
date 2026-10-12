import argparse

from .build import Exporter, evidence

ap = argparse.ArgumentParser(description="Write web/data/*.json for the dashboard")
ap.add_argument("--world", default="data/world")
a = ap.parse_args()
evidence()
Exporter(a.world).run()
