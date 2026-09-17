#!/usr/bin/env python3
"""
PROJECT MAINLINE ENTRY POINT (finalized 2026-08-30).

Single documented way to train + evaluate the federated JointPrototype
family across all three datasets, for both tasks (detection AUROC and
node-level localization hit-rate). This module does not reimplement any
data loading, training, or scoring logic -- it is a thin dispatcher over
the existing `benchmark/run_*_bck_federated.py` (detection) and
`benchmark/diagnose_*_localization_federated.py` (localization) scripts,
loaded the same way those diagnose scripts already load their own
`run_*_bck_federated.py` base (`importlib.util.spec_from_file_location`),
so there is exactly one place that knows "which dataset uses which
script" and one CLI for all of them.

Mainline defaults (both overridable per call, never removed as options):
  - combined score: BK = smooth_max(B_node_max, K_forecast_max)
  - clustering: linkage="single" (BFS connected components,
    `federated_memory.align_and_split`'s original behavior)
  - forecast_head (K) IS FedAvg'd whenever the dataset syncs
    encoder/decoder (paderborn) -- see
    `src/federated/federated_train_eval.py`'s module docstring. Sielaff keeps its
    original memory-only exchange (it never syncs encoder/decoder, so this
    is inert for it either way).

Every dataset reports ONLY B/K/BK as of the 2026-09-02 BK-only trim (see
`federated_train_eval.py`'s module docstring) -- the earlier C/E/F/H/I/J/CK/HK/BCK
signals are retired from every detection report (`paderborn`'s
full ablation scoring code is snapshotted verbatim in
`archive/src/federated/legacy_scores.py`; `sielaff`/`alfa` were trimmed
in place, same convention). `--signal` selects which column this module
prints as the headline number (default `BK`, the only non-B/K signal left);
`linkage="complete"` remains an opt-in alternative to the default
single-linkage BFS.

Report formats are NOT unified across datasets (deliberately -- Sielaff's
scoring/labeling is a genuinely different algorithm, see
`federated_train_eval.py`'s module docstring) -- `headline_score()` below best-
effort extracts the requested signal's overall number from whichever
summary key each dataset's script actually writes; consult the saved
JSON report directly for anything beyond the headline number.

Usage:
    python3 src/federated/pipeline.py --dataset paderborn
    python3 src/federated/pipeline.py --dataset paderborn --mode localization
    python3 src/federated/pipeline.py --dataset alfa --signal K --linkage complete
    python3 src/federated/pipeline.py --dataset paderborn --mode both --out-suffix my_run
"""
import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "benchmark"

# dataset -> {"detection": script, "localization": script or None}.
# Sielaff is deliberately NOT registered here: report/040_experiment.tex
# describes the paper's generalization datasets as "two other PUBLIC
# datasets" alongside the own robot fleet, and Sielaff (a private,
# self-collected reverse-vending-machine dataset -- see
# `benchmark/datasets/sielaff_physics.md`) doesn't meet that bar, unlike
# Paderborn (public bearing benchmark) and ALFA (public UAV fault dataset).
# Its scripts/data/checkpoints are untouched on disk, just not dispatched
# from this pipeline entry point -- see `run_sielaff_red_bck_federated.py`
# / `diagnose_sielaff_red_localization_federated.py` if reinstating it.
DATASET_SCRIPTS = {
    "paderborn": {
        "detection": "run_paderborn_bck_federated.py",
        "localization": "diagnose_paderborn_localization_federated.py",
    },
    "alfa": {
        "detection": "run_alfa_bck_federated.py",
        "localization": "diagnose_alfa_localization_federated.py",
    },
}

# Best-effort: which top-level report key holds the overall summary dict,
# tried in order (differs per dataset's script -- see module docstring).
_SUMMARY_KEYS = ("summary_mean_auroc_overall", "summary_mean_auroc")
_SIGNAL_TO_REPORT_KEY = {
    "B": "B_node_max", "K": "K_forecast_max", "BK": "BK_max",
}


def _load_script(filename):
    path = BENCHMARK_DIR / filename
    mod_name = f"_pipeline_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def headline_score(report: dict, signal: str = "BK"):
    """Best-effort pull of `signal`'s overall mean AUROC out of a detection
    report -- returns None (not an exception) if the report's summary
    doesn't have that key, since exact summary shape is dataset-specific
    (see module docstring)."""
    report_key = _SIGNAL_TO_REPORT_KEY.get(signal, signal)
    for key in _SUMMARY_KEYS:
        summary = report.get(key)
        if summary and report_key in summary:
            return summary[report_key]
    return None


def run_detection(dataset: str, signal: str = "BK", linkage: str = "single", **kwargs):
    """Runs the dataset's federated detection script and returns its full
    report dict. `signal` only controls which number this function prints
    as the headline -- the report itself always contains every signal."""
    scripts = DATASET_SCRIPTS[dataset]
    module = _load_script(scripts["detection"])
    report = module.main(linkage=linkage, **kwargs)
    score = headline_score(report, signal)
    if score is not None:
        print(f"\n[pipeline] {dataset} detection, headline signal {signal} "
              f"(linkage={linkage}): {score:.4f}")
    else:
        print(f"\n[pipeline] {dataset} detection: could not extract headline "
              f"signal {signal!r} from this report's summary -- inspect the "
              f"returned dict directly.")
    return report


def run_localization(dataset: str, signal: str = "BK", linkage: str = "single", **kwargs):
    """Runs the dataset's federated localization (root-cause diagnosis)
    script and returns its full report dict. Per-fault-type hit rates for
    every signal are always present; this only prints `signal`'s numbers."""
    scripts = DATASET_SCRIPTS[dataset]
    if scripts["localization"] is None:
        raise ValueError(f"{dataset} has no localization script registered")
    module = _load_script(scripts["localization"])
    report = module.main(linkage=linkage, **kwargs)
    print(f"\n[pipeline] {dataset} localization, headline signal {signal} "
          f"(linkage={linkage}):")
    for key, val in report.items():
        if key.startswith("_"):
            continue
        row = val.get(signal) if isinstance(val, dict) else None
        if isinstance(row, dict) and "direct_or_indirect_pct" in row:
            print(f"    {key:<16}{row['direct_or_indirect_pct']}%")
    return report


def run(dataset: str, mode: str = "detection", signal: str = "BK", linkage: str = "single", **kwargs):
    """`mode`: "detection", "localization", or "both". Returns a dict with
    whichever of {"detection", "localization"} keys were run."""
    if dataset not in DATASET_SCRIPTS:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {list(DATASET_SCRIPTS)}")
    out = {}
    if mode in ("detection", "both"):
        out["detection"] = run_detection(dataset, signal=signal, linkage=linkage, **kwargs)
    if mode in ("localization", "both"):
        out["localization"] = run_localization(dataset, signal=signal, linkage=linkage, **kwargs)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASET_SCRIPTS))
    parser.add_argument("--mode", default="detection", choices=["detection", "localization", "both"])
    parser.add_argument("--signal", default="BK", choices=list(_SIGNAL_TO_REPORT_KEY))
    parser.add_argument("--linkage", default="single", choices=["single", "complete"])
    parser.add_argument("--horizon-mult", type=int, default=None,
                         help="passthrough to the underlying script; omit to use its own default")
    parser.add_argument("--out-suffix", type=str, default=None)
    args = parser.parse_args()

    extra = {"out_suffix": args.out_suffix}
    if args.horizon_mult is not None:
        extra["horizon_mult"] = args.horizon_mult

    run(args.dataset, mode=args.mode, signal=args.signal, linkage=args.linkage, **extra)
