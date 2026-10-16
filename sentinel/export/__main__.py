import argparse

from .build import Exporter, evidence
from .model import export_model

ap = argparse.ArgumentParser(description="Write web/data/*.json for the dashboard")
ap.add_argument("--world", default="data/world")
a = ap.parse_args()
evidence()
ex = Exporter(a.world)
ex.run()
export_model(a.world, ex)
