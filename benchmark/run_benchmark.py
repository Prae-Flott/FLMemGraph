#!/usr/bin/env python3
"""
Runs every `status: "implemented"` baseline (`baselines/registry.py`)
against robo3er (`datasets/robo3er_adapter.py`, pooled/centralized mode)
and reports AUROC per fault type -- the harness's own smoke test, and the
first row of what should grow into a full cross-dataset x cross-method
table as more datasets/baselines from the registries get implemented
(most are currently `status: "planned"`/`"external"`, see the registry
files and `benchmark/README.md`).

Usage:
    python3 run_benchmark.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets.robo3er_adapter import load as load_robo3er_bench  # noqa: E402
from baselines.traditional import PCABaseline, IsolationForestBaseline  # noqa: E402
from baselines.gdn_baseline import GDNBaseline, GDNTunedBaseline  # noqa: E402
from metrics import auroc  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def main():
    fit_w, calib_w, test_normal_w, faults, cols = load_robo3er_bench(pooled=True)
    print(f"robo3er (pooled): fit={len(fit_w)} calib={len(calib_w)} test_normal={len(test_normal_w)} "
          f"faults={ {k: len(v) for k,v in faults.items()} }")

    methods = {
        "PCA": PCABaseline(),
        "IsolationForest": IsolationForestBaseline(),
        "GDN": GDNBaseline(),
        "GDN-tuned": GDNTunedBaseline(),
    }

    results = {}
    for name, model in methods.items():
        print(f"\nfitting {name}...")
        if name in ("GDN", "GDN-tuned"):
            model.fit(fit_w, calib_w)
        else:
            model.fit(fit_w)
        normal_score = model.score(test_normal_w)
        results[name] = {}
        for fault_name, fault_w in faults.items():
            fault_score = model.score(fault_w)
            labels = [0] * len(normal_score) + [1] * len(fault_score)
            scores = list(normal_score) + list(fault_score)
            results[name][fault_name] = auroc(labels, scores)

    fault_names = list(faults.keys())
    header = f"{'method':<18}" + "".join(f"{n:>16}" for n in fault_names)
    print("\n" + header)
    for name in methods:
        row = f"{name:<18}" + "".join(f"{results[name][f]:>16.3f}" for f in fault_names)
        print(row)

    with open(OUT_DIR / "benchmark_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {OUT_DIR / 'benchmark_report.json'}")


if __name__ == "__main__":
    main()
