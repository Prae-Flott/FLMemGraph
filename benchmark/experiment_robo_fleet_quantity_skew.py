#!/usr/bin/env python3
"""
EXPERIMENT (not mainline): quantity-skew non-IID sweep for the FD+JD+PD
mainline on robo_fleet.

`run_robo_fleet_bck_federated.py`'s 4 real per-robot clients already come
out ALMOST perfectly balanced (fit sizes 518/548/555/564, ~5% spread) --
there is no existing knob anywhere in this codebase that controls how
IID/non-IID a federation is (no Dirichlet-alpha label-skew partitioner
exists; every dataset's clients are physically fixed: per-robot,
per-bearing, per-fault-type). This script adds one FOR THIS EXPERIMENT
ONLY, without touching `run_robo_fleet_bck_federated.py`: a "quantity
skew" resample of each client's own TRAINING (fit_idx) pool, following
the standard Dirichlet(alpha) quantity-skew convention (Li et al.,
"Federated Learning on Non-IID Data Silos") -- small alpha => a few
clients get most of the training budget and others get very little;
large alpha => proportions converge back to ~uniform (the dataset's own
near-IID default). calib_idx/test_normal_idx and every fault-type
client's evaluation split are left EXACTLY as `build_clients()` produces
them, so only the volume of NORMAL data each client trains on changes --
evaluation stays apples-to-apples across every alpha in the sweep.

Mechanism per alpha:
  1. total_fit = sum(len(c.fit_idx) for c in the real clients) -- the
     Dirichlet draw redistributes this SAME total budget across clients
     instead of inflating/shrinking it.
  2. proportions ~ Dirichlet([alpha] * num_clients), target_n_c =
     round(proportions_c * total_fit), clipped to
     [MIN_FIT_SAMPLES, client's own available fit pool] -- a client can
     use LESS of its own pool than it has, never more (data physically
     lives on that robot).
  3. new fit_idx = a random (seeded, reproducible per alpha) subsample of
     that size from the client's original fit_idx, sorted.

Does NOT modify `run_robo_fleet_bck_federated.py`/`federated_train_eval.py` --
imports the former as a module and monkeypatches its `build_clients` for
the duration of each run only.

Usage:
    python3 benchmark/experiment_robo_fleet_quantity_skew.py \
        [--alphas 1000 5 1 0.5 0.1] [--min-fit 40]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("robo_fleet_bck_fed", REPO_ROOT / "run_robo_fleet_bck_federated.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

REAL_BUILD_CLIENTS = m.build_clients
SIGNALS = ["FD_max", "JD_mahal", "FD_JD_max", "PD_max", "FD_PD_max", "FD_JD_PD_max"]


def make_skewed_build_clients(alpha, min_fit, seed):
    def skewed_build_clients():
        data, targets, label_map, clients, cols, window_robot = REAL_BUILD_CLIENTS()
        rng = np.random.default_rng(seed)
        pool_sizes = np.array([len(c.fit_idx) for c in clients])
        total_fit = int(pool_sizes.sum())
        proportions = rng.dirichlet([alpha] * len(clients))
        target_n = np.round(proportions * total_fit).astype(int)
        target_n = np.clip(target_n, min_fit, pool_sizes)
        for c, n_c in zip(clients, target_n):
            n_c = int(n_c)
            if n_c >= len(c.fit_idx):
                continue  # keep full pool (also covers n_c == pool size)
            # A random (independent-per-window) subsample would shred the
            # runs of horizon_mult consecutive window indices `build_pairs`
            # requires for forecast-head (K) pairing, driving every
            # skewed-down client's usable fit set to ~0 pairs. Take a
            # contiguous slice instead -- it preserves those runs (fit_idx
            # is itself a contiguous early-session prefix of normal_idx)
            # while still cutting the client down to a smaller, still-non-
            # contiguous-with-full-pool subset of its own data.
            start = rng.integers(0, len(c.fit_idx) - n_c + 1)
            c.fit_idx = c.fit_idx[start:start + n_c]
        return data, targets, label_map, clients, cols, window_robot

    return skewed_build_clients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", type=float, nargs="+", default=[1000.0, 5.0, 1.0, 0.5, 0.1])
    parser.add_argument("--min-fit", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
    # alpha=None run first: the dataset's own natural (already near-balanced) split.
    for alpha in [None] + list(args.alphas):
        # Recompute the actual fit sizes used (for reporting) with the same seed logic.
        m.build_clients = make_skewed_build_clients(alpha, args.min_fit, args.seed) if alpha is not None else REAL_BUILD_CLIENTS
        _, _, _, probe_clients, _, _ = m.build_clients()
        fit_sizes = [len(c.fit_idx) for c in probe_clients]

        label = "natural" if alpha is None else f"alpha={alpha}"
        print(f"\n{'=' * 70}\nquantity-skew run: {label}  fit_sizes={fit_sizes}\n{'=' * 70}")
        report = m.main(out_suffix=("qskew_natural" if alpha is None else f"qskew_alpha{alpha}"))
        summary = report["summary_mean_auroc_overall"]
        row = {"alpha": label, "fit_sizes": fit_sizes,
               "fit_spread_max_over_min": max(fit_sizes) / max(min(fit_sizes), 1)}
        for sig in SIGNALS:
            row[sig] = summary.get(sig, {}).get("auroc")
        rows.append(row)

    m.build_clients = REAL_BUILD_CLIENTS  # restore

    print(f"\n{'alpha':<14}{'max/min':>9}" + "".join(f"{s.replace('_max', '').replace('_node', '').replace('_cov_mahal', '').replace('_forecast', ''):>9}" for s in SIGNALS))
    for row in rows:
        vals = "".join(f"{row[s]:>9.3f}" if row[s] is not None else f"{'--':>9}" for s in SIGNALS)
        print(f"{row['alpha']:<14}{row['fit_spread_max_over_min']:>9.2f}{vals}")

    out_path = REPO_ROOT.parent / "checkpoints" / "robo_fleet" / "quantity_skew_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
