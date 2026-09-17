#!/usr/bin/env python3
"""
FedPRO on me_ad -- thin runner, see `benchmark/FedPRO/README.md` for what this
method actually is and where its code lives. Delegates to
`main_fedpro_baseline(...)` in `benchmark/run_me_ad_bck_federated.py`
(loaded as a module, same convention `benchmark/experiment_*.py` scripts
already use) -- does not duplicate or reimplement anything.

Usage:
    python3 benchmark/FedPRO/run_me_ad.py
"""
import argparse
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "me_ad_bck_fed", REPO_ROOT / "benchmark" / "run_me_ad_bck_federated.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-suffix", type=str, default=None)

    args = parser.parse_args()
    m.main_fedpro_baseline(out_suffix=args.out_suffix)
