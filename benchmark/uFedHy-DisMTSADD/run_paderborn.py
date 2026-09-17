#!/usr/bin/env python3
"""
uFedHy-DisMTSADD on paderborn -- thin runner, see `benchmark/uFedHy-DisMTSADD/README.md` for what this
method actually is and where its code lives. Delegates to
`main_ufedhy_baseline(...)` in `benchmark/run_paderborn_bck_federated.py`
(loaded as a module, same convention `benchmark/experiment_*.py` scripts
already use) -- does not duplicate or reimplement anything.

Usage:
    python3 benchmark/uFedHy-DisMTSADD/run_paderborn.py
"""
import argparse
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "paderborn_bck_fed", REPO_ROOT / "benchmark" / "run_paderborn_bck_federated.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-suffix", type=str, default=None)

    args = parser.parse_args()
    m.main_ufedhy_baseline(out_suffix=args.out_suffix)
