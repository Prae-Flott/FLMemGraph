#!/usr/bin/env python3
"""
Unified torque-balance physics residual for Paderborn, replacing
`run_paderborn_physics_gdn.py`'s current-based residual (which hurt
detection -- see `paderborn_physics.md`) with one that connects directly
to the bearing instead of routing through the motor.

## The physical relation (see src/paderborn_physics.py.fit_torque_model
## for the full writeup)

The torque sensor sits between the drive motor and the bearing/flywheel/
load-motor stack, so measured torque is a steady-state balance of
bearing friction (driven by the independently, externally-applied radial
`force`) plus the commanded load:

    predicted_torque = a*force + b*speed + c

`a` stands in for the load-dependent (Coulomb-like) bearing friction
coefficient (structurally `0.5*mu*d_m`, `mu` not precisely known for this
rig so it's fit, not assumed -- same "fix the structure, fit the
constant" logic as kinematics.py). `b` stands in for the speed-dependent
(viscous) friction component -- `speed`'s LEVEL, not `dω/dt`, because
shaft speed is essentially constant within each 4s recording (checked
directly: std/mean < 0.03%), so there's no meaningful inertial dynamics
to capture, only the viscous-friction mechanism.

residual = measured_torque - predicted_torque, fed to GDN as a 4th node
ALONGSIDE the three raw channels (vibration_1, phase_current_1,
phase_current_2) -- unlike the earlier current-residual experiment,
which REPLACED the current channels, this ADDS a new signal, since the
earlier finding was that raw current already outperforms residualized
current -- no reason to give that up while testing whether torque_residual
contributes something additional.

Two runs, same seed/architecture, only the node set differs:
  A. baseline: vibration_1 + phase_current_1/2 (3 nodes, matches
     run_paderborn_physics_gdn.py's method A)
  B. +torque_residual: the same 3 channels + torque_residual (4 nodes)

Usage:
    python3 run_paderborn_torque_residual_gdn.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gdn_model import GDN  # noqa: E402
from paderborn_physics import fit_torque_model, apply_torque_residual, fit_torque_model_fixed_mu  # noqa: E402
from datasets.paderborn_adapter import (  # noqa: E402
    load_bearing_with_physics, HEALTHY_CODES, ALL_DAMAGED_CODES, category_of, damage_origin_of,
)

OUT_DIR = REPO_ROOT / "checkpoints"

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
SEED = 42
EMBED_DIM = 64
TOP_K = 15
HISTORY_LEN = 64
DENSE_STRIDE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_CHANNELS = ["vibration_1", "phase_current_1", "phase_current_2"]


def stack_nodes(d, keys):
    return np.stack([d[k] for k in keys], axis=-1).astype(np.float32)


def make_dense_pairs(arr, history_len=HISTORY_LEN, stride=DENSE_STRIDE):
    n, t, f = arr.shape
    positions = list(range(history_len, t, stride))
    history = np.empty((n, len(positions), history_len, f), dtype=np.float32)
    target = np.empty((n, len(positions), f), dtype=np.float32)
    for j, pos in enumerate(positions):
        history[:, j] = arr[:, pos - history_len : pos]
        target[:, j] = arr[:, pos]
    file_id = np.repeat(np.arange(n), len(positions))
    return history.reshape(-1, history_len, f), target.reshape(-1, f), file_id


@torch.no_grad()
def per_sample_error(model, history, target, batch_size=BATCH_SIZE):
    model.eval()
    errors = []
    for i in range(0, len(history), batch_size):
        h = torch.from_numpy(history[i : i + batch_size]).to(DEVICE)
        t = torch.from_numpy(target[i : i + batch_size]).to(DEVICE)
        pred = model(h)
        errors.append(((pred - t) ** 2).cpu().numpy())
    return np.concatenate(errors, axis=0)


def per_file_error(model, arr, batch_size=BATCH_SIZE):
    history, target, file_id = make_dense_pairs(arr)
    err = per_sample_error(model, history, target, batch_size)
    n = len(arr)
    out = np.zeros((n, err.shape[1]), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    np.add.at(out, file_id, err)
    np.add.at(counts, file_id, 1)
    return (out / counts[:, None]).astype(np.float32)


def gdn_score(per_file_err, median, iqr):
    return ((per_file_err - median) / np.maximum(iqr, 1e-8)).max(axis=1)


def train_gdn(fit_arr, calib_arr, num_nodes):
    torch.manual_seed(SEED)
    fit_history, fit_target, _ = make_dense_pairs(fit_arr)
    model = GDN(num_nodes=num_nodes, window_size=HISTORY_LEN, embed_dim=EMBED_DIM, top_k=TOP_K).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fit_history), torch.from_numpy(fit_target)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    best_calib_mse, best_state = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        fit_loss = 0.0
        for h, t in loader:
            h, t = h.to(DEVICE), t.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(h), t)
            loss.backward()
            optimizer.step()
            fit_loss += loss.item() * h.size(0)
        fit_loss /= len(fit_history)
        calib_mse = float(per_file_error(model, calib_arr).mean())
        print(f"    epoch {epoch:2d}  fit_mse={fit_loss:.6f}  calib_mse={calib_mse:.6f}")
        if calib_mse < best_calib_mse:
            best_calib_mse = calib_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_calib_mse


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("loading healthy bearings (K001-K006) with mechanical channels ...")
    healthy_data = {code: load_bearing_with_physics(code) for code in HEALTHY_CODES}

    fit_slices, calib_slices, test_slices = {}, {}, {}
    for code, d in healthy_data.items():
        n = len(d["vibration_1"])
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        fit_slices[code] = slice(0, n_fit)
        calib_slices[code] = slice(n_fit, n_fit + n_calib)
        test_slices[code] = slice(n_fit + n_calib, n)
        print(f"  {code}: n={n} fit={n_fit} calib={n_calib} test_normal={n - n_fit - n_calib}")

    def pool(d_dict, slices, key):
        return np.concatenate([d_dict[c][key][slices[c]] for c in d_dict], axis=0)

    print("\nfitting torque-balance physics model (force, speed -> torque; FIT split only) ...")
    fit_force = pool(healthy_data, fit_slices, "force")
    fit_speed = pool(healthy_data, fit_slices, "speed")
    fit_torque = pool(healthy_data, fit_slices, "torque")
    torque_params = fit_torque_model(fit_torque, fit_force, fit_speed)
    print(f"  [free fit]  torque ~= {torque_params['a_force']:.6f}*force + {torque_params['b_speed']:.6f}*speed "
          f"+ {torque_params['c_intercept']:.4f}   (R^2={torque_params['r2']:.3f})")

    torque_params_fixed = fit_torque_model_fixed_mu(fit_torque, fit_force, fit_speed)
    print(f"  [fixed mu={torque_params_fixed['mu_used']}]  torque ~= {torque_params_fixed['a_force']:.8f}*force "
          f"(FIXED) + {torque_params_fixed['b_speed']:.6f}*speed + {torque_params_fixed['c_intercept']:.4f}   "
          f"(R^2={torque_params_fixed['r2']:.3f})")

    def with_residual(d, params):
        out = dict(d)
        out["torque_residual"] = apply_torque_residual(d["torque"], d["force"], d["speed"], params)
        return out

    healthy_data_r = {code: with_residual(d, torque_params) for code, d in healthy_data.items()}
    healthy_data_rf = {code: with_residual(d, torque_params_fixed) for code, d in healthy_data.items()}

    results = {}
    for label, node_keys in [("A_baseline", BASE_CHANNELS),
                               ("B_plus_torque_residual_freefit", BASE_CHANNELS + ["torque_residual"]),
                               ("C_plus_torque_residual_fixed_mu", BASE_CHANNELS + ["torque_residual"])]:
        print(f"\n{'=' * 70}\nMETHOD {label}  (nodes={node_keys})\n{'=' * 70}")
        if label == "A_baseline":
            src = healthy_data
        elif label == "B_plus_torque_residual_freefit":
            src = healthy_data_r
        else:
            src = healthy_data_rf
        params_for_fault = torque_params if label == "B_plus_torque_residual_freefit" else torque_params_fixed
        num_nodes = len(node_keys)

        fit_arr = np.concatenate([stack_nodes(src[c], node_keys)[fit_slices[c]] for c in src], axis=0)
        calib_arr = np.concatenate([stack_nodes(src[c], node_keys)[calib_slices[c]] for c in src], axis=0)
        test_normal_arr = np.concatenate([stack_nodes(src[c], node_keys)[test_slices[c]] for c in src], axis=0)

        scaler = StandardScaler().fit(fit_arr.reshape(-1, num_nodes))

        def scale(a):
            n, t, f = a.shape
            return scaler.transform(a.reshape(-1, f)).reshape(n, t, f).astype(np.float32)

        fit_scaled, calib_scaled, test_normal_scaled = scale(fit_arr), scale(calib_arr), scale(test_normal_arr)

        print(f"  training GDN (nodes={num_nodes}) ...")
        model, calib_mse = train_gdn(fit_scaled, calib_scaled, num_nodes)

        calib_err = per_file_error(model, calib_scaled)
        median = np.median(calib_err, axis=0)
        q75, q25 = np.percentile(calib_err, [75, 25], axis=0)
        iqr = np.maximum(q75 - q25, 1e-8)
        normal_err = per_file_error(model, test_normal_scaled)
        normal_score = gdn_score(normal_err, median, iqr)

        rows = []
        for code in ALL_DAMAGED_CODES:
            d = load_bearing_with_physics(code)
            if "torque_residual" in node_keys:
                d = with_residual(d, params_for_fault)
            fault_arr = scale(stack_nodes(d, node_keys))
            fault_err = per_file_error(model, fault_arr)
            fault_score = gdn_score(fault_err, median, iqr)
            labels = np.concatenate([np.zeros(len(normal_score)), np.ones(len(fault_score))])
            scores = np.concatenate([normal_score, fault_score])
            auroc = float(sk_metrics.roc_auc_score(labels, scores))
            rows.append((code, category_of(code), damage_origin_of(code), auroc))
            print(f"    {code:<6}{category_of(code):<12}{damage_origin_of(code):<11}AUROC={auroc:.3f}")

        cat_summary = {}
        for cat in ["outer_ring", "inner_ring", "combined"]:
            subset = [r[3] for r in rows if r[1] == cat]
            if subset:
                cat_summary[cat] = {"n": len(subset), "mean_auroc": float(np.mean(subset))}
        print(f"  category summary: {cat_summary}")
        results[label] = {"calib_mse": calib_mse, "rows": rows, "category_summary": cat_summary}

    print(f"\n{'=' * 70}\nCOMPARISON: baseline vs. +torque_residual (free-fit) vs. +torque_residual (fixed mu)\n{'=' * 70}")
    print(f"{'bearing':<8}{'category':<12}{'baseline':>10}{'freefit':>10}{'fixed_mu':>10}{'Δfreefit':>10}{'Δfixed_mu':>10}")
    rows_a = {r[0]: r[3] for r in results["A_baseline"]["rows"]}
    rows_b = {r[0]: r[3] for r in results["B_plus_torque_residual_freefit"]["rows"]}
    rows_c = {r[0]: r[3] for r in results["C_plus_torque_residual_fixed_mu"]["rows"]}
    for code in ALL_DAMAGED_CODES:
        a, b, c = rows_a[code], rows_b[code], rows_c[code]
        print(f"{code:<8}{category_of(code):<12}{a:>10.3f}{b:>10.3f}{c:>10.3f}{b - a:>+10.3f}{c - a:>+10.3f}")

    summary = {}
    for label, rows in [("B_freefit", rows_b), ("C_fixed_mu", rows_c)]:
        n_better = sum(1 for cd in ALL_DAMAGED_CODES if rows[cd] > rows_a[cd])
        n_worse = sum(1 for cd in ALL_DAMAGED_CODES if rows[cd] < rows_a[cd])
        n_same = len(ALL_DAMAGED_CODES) - n_better - n_worse
        summary[label] = {"n_improved": n_better, "n_worse": n_worse, "n_unchanged": n_same}
        print(f"\n{label}: {n_better} improved, {n_worse} got worse, {n_same} unchanged (of {len(ALL_DAMAGED_CODES)})")

    report = {"torque_model_freefit": torque_params, "torque_model_fixed_mu": torque_params_fixed,
               "results": results, "summary": summary}
    with open(OUT_DIR / "paderborn_torque_residual_gdn_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {OUT_DIR / 'paderborn_torque_residual_gdn_report.json'}")


if __name__ == "__main__":
    main()
