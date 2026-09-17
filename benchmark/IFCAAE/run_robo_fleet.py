#!/usr/bin/env python3
"""
IFCAAE on robo_fleet -- thin runner, see `benchmark/IFCAAE/README.md` for what this
method actually is and where its code lives. Delegates to
`main_faithful_baseline("ifcaae", ...)` in `benchmark/run_robo_fleet_bck_federated.py`
(loaded as a module, same convention `benchmark/experiment_*.py` scripts
already use) -- does not duplicate or reimplement anything.

Usage:
    python3 benchmark/IFCAAE/run_robo_fleet.py [--num-clusters N]
"""
import argparse
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "robo_fleet_bck_fed", REPO_ROOT / "benchmark" / "run_robo_fleet_bck_federated.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-suffix", type=str, default=None)
    parser.add_argument("--num-clusters", type=int, default=2)

    args = parser.parse_args()
    m.main_faithful_baseline("ifcaae", out_suffix=args.out_suffix, num_clusters=args.num_clusters)
