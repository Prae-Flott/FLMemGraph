#!/usr/bin/env python3
"""
Grid search NUM_PROTOTYPES in [2, 20] on robo_fleet, comparing the global
vs. per-prototype ("proto") FD/PD/FD+JD/FD+PD/FD+JD+PD signals at each value -- follow-up
to the single NUM_PROTOTYPES=2 result showing per-prototype conditioning
slightly HURT on robo_fleet (unlike Paderborn's clear win), hypothesized to
be because too few prototypes collapse federated alignment onto one shared
slot (`num_shared_prototypes=1` in the alignment log), making "per-prototype"
statistically indistinguishable from "global, but with fewer samples and
more noise." Sweeping NUM_PROTOTYPES up tests whether more prototype
capacity gives per-prototype conditioning room to actually differentiate
operating regimes the way it does on Paderborn.

Calls `run_robo_fleet_bck_federated.main(num_prototypes=M, ...)` in-process
for each M (no subprocess overhead), writing the running results to
`checkpoints/robo_fleet/num_prototypes_sweep.json` after every value so a
partial sweep is inspectable if interrupted.

Usage:
    python3 sweep_robo_fleet_num_prototypes.py [--lo 2] [--hi 20]
"""
import argparse
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "checkpoints" / "robo_fleet" / "num_prototypes_sweep.json"

spec = importlib.util.spec_from_file_location("robo_fleet_bck_fed", Path(__file__).resolve().parent / "run_robo_fleet_bck_federated.py")
v2f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2f)

SIGNALS = ["FD_max", "FD_max_proto", "PD_max", "PD_max_proto",
           "FD_JD_max", "FD_JD_proto_max", "FD_PD_max", "FD_PD_proto_max", "FD_JD_PD_max", "FD_JD_PD_proto_max"]


def main(lo=2, hi=20):
    results = []
    for m in range(lo, hi + 1):
        print(f"\n{'='*70}\nNUM_PROTOTYPES = {m}\n{'='*70}")
        report = v2f.main(num_prototypes=m, out_suffix=f"sweep_m{m}")
        summary = report["summary_mean_auroc_overall"]
        row = {"num_prototypes": m,
               "num_shared_prototypes_final_round": report["alignment_log"][-1]["num_shared_prototypes"],
               "signals": {sig: summary[sig] for sig in SIGNALS if sig in summary}}
        results.append(row)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[sweep] M={m} done, num_shared_prototypes={row['num_shared_prototypes_final_round']} "
              f"B_auroc={row['signals'].get('FD_max', {}).get('auroc')} "
              f"B_proto_auroc={row['signals'].get('FD_max_proto', {}).get('auroc')}")

    print(f"\n{'M':<5}{'shared':<8}" + "".join(f"{s:<14}" for s in SIGNALS))
    for row in results:
        vals = "".join(f"{row['signals'].get(s, {}).get('auroc', float('nan')):<14.3f}" for s in SIGNALS)
        print(f"{row['num_prototypes']:<5}{row['num_shared_prototypes_final_round']:<8}{vals}")

    print(f"\nsaved -> {OUT_PATH}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lo", type=int, default=2)
    parser.add_argument("--hi", type=int, default=20)
    args = parser.parse_args()
    main(lo=args.lo, hi=args.hi)
