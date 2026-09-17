#!/usr/bin/env python3
"""
EXPERIMENT (not mainline): SCENARIO-skew (indoor/outdoor) non-IID sweep
for the BHK mainline on robo_fleet, complementing
`experiment_robo_fleet_quantity_skew.py`'s data-VOLUME skew with a
data-COMPOSITION skew: instead of controlling how MUCH normal training
data each client gets, this controls what MIX of scenarios it comes
from.

robo_fleet's raw session layout genuinely has two recording
environments per robot -- confirmed in `data/robo_fleet/
window_session_names.json` (e.g. `rob_00/normal/map_inside_20min` vs.
`rob_00/normal/map_outside_20min`) -- so "indoor vs. outdoor" is real
metadata here, not synthesized. Each client's TRAINING (fit_idx) pool is
naturally ~68-74% indoor / ~26-32% outdoor already (indoor sessions were
recorded first within each robot's timeline, and `build_clients()`'s
fit split is the first 70% of each robot's time-ordered normal indices)
-- see the module-level `NATURAL_INSIDE_FRACTION` printout at runtime.

Mechanism per alpha (drawn INDEPENDENTLY per client, unlike the quantity-
skew script's shared-budget Dirichlet draw -- this is the point: small
alpha should let different robots land on OPPOSITE scenario extremes,
not just each be skewed in the same direction):
  1. p_c ~ Beta(alpha, alpha) = that client's target indoor FRACTION of
     its own (unchanged-size) fit_idx pool. alpha>>1 concentrates p_c
     near 0.5 (a balanced indoor/outdoor mix, close to but not identical
     to the natural ~70/30 split); alpha<<1 is bimodal near 0 or 1 (each
     client ends up almost entirely one scenario, and independently-drawn
     clients often land on DIFFERENT extremes).
  2. n_indoor = round(p_c * total_fit_c) and n_outdoor =
     round((1-p_c) * total_fit_c), EACH capped independently against its
     own scenario's available pool (both pools are SUBSETS OF THE
     CLIENT'S OWN ORIGINAL fit_idx -- calib_idx/test_normal_idx/fault
     splits are untouched, exactly like the quantity-skew script, so
     eval stays fixed across every alpha). These two counts are NOT
     coupled by `n_outdoor = total - n_indoor`: since indoor_pool +
     outdoor_pool == total_fit_c exactly (every fit_idx window is one or
     the other), a coupled formula has zero headroom and collapses
     n_indoor to one fixed value regardless of alpha. Instead, a
     client's REALIZED total fit size is allowed to shrink at extreme
     skew -- wanting ~100% one scenario means using only that scenario's
     available pool, less than the natural total. A scenario count below
     20 is zeroed out entirely (too few for `build_pairs`'s forecast-
     pairing chains to matter).
  3. new fit_idx = the client's own indoor sub-pool's indices (a
     contiguous slice, chosen the same way as the quantity-skew script,
     for the same `build_pairs`-needs-consecutive-indices reason) unioned
     with its outdoor sub-pool's indices, sorted.

Because indoor dominates each client's fit_idx pool by construction
(indoor_pool ~= 68-74% of total, outdoor_pool the rest), max ~100%-indoor
purity uses close to the natural total fit size, while max outdoor
purity is capped much lower (only ~134-181 outdoor windows exist per
client) -- an asymmetric achievable range that's a real dataset
property, not a script bug.

Does NOT modify `run_robo_fleet_bck_federated.py`/`federated_train_eval.py`.

Usage:
    python3 benchmark/experiment_robo_fleet_scenario_skew.py \
        [--alphas 50 5 1 0.3 0.05] [--seed 0]
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
SIGNALS = ["B_node_max", "H_cov_mahal", "BH_max", "K_forecast_max", "BK_max", "BHK_max"]
SESSION_NAMES = json.load(open(REPO_ROOT.parent / "data" / "robo_fleet" / "window_session_names.json"))
MIN_SCENARIO = 20


def scenario_of(idx):
    name = SESSION_NAMES[idx]
    if "map_inside" in name:
        return "indoor"
    if "map_outside" in name:
        return "outdoor"
    return "unknown"


def contiguous_slice(rng, pool, n):
    n = min(n, len(pool))
    if n <= 0:
        return np.zeros(0, dtype=int)
    start = rng.integers(0, len(pool) - n + 1)
    return pool[start:start + n]


def make_skewed_build_clients(alpha, seed):
    def skewed_build_clients():
        data, targets, label_map, clients, cols, window_robot = REAL_BUILD_CLIENTS()
        rng = np.random.default_rng(seed)
        for c in clients:
            scen = np.array([scenario_of(i) for i in c.fit_idx])
            indoor_pool = c.fit_idx[scen == "indoor"]
            outdoor_pool = c.fit_idx[scen == "outdoor"]
            total = len(c.fit_idx)
            p_c = rng.beta(alpha, alpha)
            # indoor_pool + outdoor_pool == total exactly (every fit_idx window
            # is one or the other), so a `total - n_indoor` coupled formula
            # always collapses n_indoor to a single fixed value regardless of
            # p_c/alpha -- there is no headroom to trade one category for the
            # other while holding total fixed. Instead each count is capped
            # independently against its own pool and the client's realized
            # total fit size is allowed to shrink at extreme skew (a client
            # that wants to be ~100% one scenario simply has less data,
            # exactly the physical situation this experiment models).
            n_indoor = int(np.clip(round(p_c * total), 0, len(indoor_pool)))
            n_outdoor = int(np.clip(round((1 - p_c) * total), 0, len(outdoor_pool)))
            if 0 < n_outdoor < MIN_SCENARIO:
                n_outdoor = 0
            if 0 < n_indoor < MIN_SCENARIO:
                n_indoor = 0
            new_indoor = contiguous_slice(rng, indoor_pool, n_indoor)
            new_outdoor = contiguous_slice(rng, outdoor_pool, n_outdoor)
            c.fit_idx = np.sort(np.concatenate([new_indoor, new_outdoor]))
        return data, targets, label_map, clients, cols, window_robot

    return skewed_build_clients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", type=float, nargs="+", default=[50.0, 5.0, 1.0, 0.3, 0.05])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for alpha in [None] + list(args.alphas):
        m.build_clients = make_skewed_build_clients(alpha, args.seed) if alpha is not None else REAL_BUILD_CLIENTS
        _, _, _, probe_clients, _, _ = m.build_clients()
        composition = []
        for c in probe_clients:
            scen = np.array([scenario_of(i) for i in c.fit_idx])
            composition.append(f"{c.robot_name}:indoor={int((scen == 'indoor').sum())}/outdoor={int((scen == 'outdoor').sum())}")

        label = "natural" if alpha is None else f"alpha={alpha}"
        print(f"\n{'=' * 70}\nscenario-skew run: {label}\n  " + "  ".join(composition) + f"\n{'=' * 70}")
        report = m.main(out_suffix=("sskew_natural" if alpha is None else f"sskew_alpha{alpha}"))
        summary = report["summary_mean_auroc_overall"]
        row = {"alpha": label, "composition": composition}
        for sig in SIGNALS:
            row[sig] = summary.get(sig, {}).get("auroc")
        rows.append(row)

    m.build_clients = REAL_BUILD_CLIENTS

    print(f"\n{'alpha':<14}" + "".join(f"{s.replace('_max', '').replace('_node', '').replace('_cov_mahal', '').replace('_forecast', ''):>9}" for s in SIGNALS))
    for row in rows:
        vals = "".join(f"{row[s]:>9.3f}" if row[s] is not None else f"{'--':>9}" for s in SIGNALS)
        print(f"{row['alpha']:<14}{vals}")

    out_path = REPO_ROOT.parent / "checkpoints" / "robo_fleet" / "scenario_skew_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
