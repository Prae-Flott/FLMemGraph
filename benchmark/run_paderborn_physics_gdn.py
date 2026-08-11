#!/usr/bin/env python3
"""
Physics-residual GDN for Paderborn, same method as robo3er's
`src/train_gdn_physics.py`: fit a physically-motivated relation between
two independently-measured signals on FIT-split (healthy) data only,
replace the raw signal with its residual (measured minus predicted)
before GDN ever sees it, train plain GDN (forecasting, no memory --
matches the robo3er script this mirrors, memory came LATER in that
project's own history too), evaluate AUROC per damaged bearing.

## The physical relation (see src/paderborn_physics.py for the full
## writeup, and benchmark/datasets/paderborn_physics.md for the general
## reference-formula documentation)

robo3er: differential-drive kinematics, wheel velocities -> predicted
odometry, residual replaces `odom_odo_lintw_x`/`odom_odo_angtw_z`.

Paderborn: basic motor physics, measured torque + speed -> predicted
current envelope (RMS), residual replaces `phase_current_1`/
`phase_current_2`. `vibration_1` stays raw (RMS... actually mean-decimated,
see adapter) throughout, same as robo3er keeps its other 36
kinematic_core features untouched.

## Two runs in this script, for a clean single-variable comparison

  A. GDN-raw-current: plain RMS-decimated current (no physics residual)
     + vibration_1 -- the "no prior" baseline, methodologically identical
     to `run_paderborn_fl_model.py`'s channel set except forecasting
     instead of reconstruction (matching robo3er's `train_gdn.py`).
  B. GDN-physics-residual: torque/speed-predicted current residual +
     vibration_1 -- the physics-prior arm under test (matching robo3er's
     `train_gdn_physics.py`).

Usage:
    python3 run_paderborn_physics_gdn.py
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
from paderborn_physics import fit_current_model, apply_current_residual  # noqa: E402
from datasets.paderborn_adapter import (  # noqa: E402
    load_bearing_with_physics, HEALTHY_CODES, ALL_DAMAGED_CODES, category_of, damage_origin_of, CHANNELS,
)

OUT_DIR = REPO_ROOT / "checkpoints" / "paderborn"

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

NODE_ORDER = ["vibration_1", "phase_current_1", "phase_current_2"]


def stack_nodes(d, current_keys):
    """d: dict from load_bearing_with_physics (or with residual current
    substituted). current_keys: which dict keys to use for the two
    current slots (lets the caller pass in either raw or residual
    arrays under the same names). Returns [n, T, 3] float32."""
    return np.stack([d["vibration_1"], d[current_keys[0]], d[current_keys[1]]], axis=-1).astype(np.float32)


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
    num_nodes = len(NODE_ORDER)

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

    print("\nfitting torque/speed -> current-envelope physics model (FIT split only) ...")
    fit_torque = pool(healthy_data, fit_slices, "torque")
    fit_speed = pool(healthy_data, fit_slices, "speed")
    current_models = {}
    for ch in ["phase_current_1", "phase_current_2"]:
        fit_current = pool(healthy_data, fit_slices, ch)
        params = fit_current_model(fit_current, fit_torque, fit_speed)
        current_models[ch] = params
        print(f"  {ch}: current ~= {params['a_torque']:.4f}*torque + {params['b_speed']:.6f}*speed "
              f"+ {params['c_intercept']:.4f}   (R^2={params['r2']:.3f})")

    def residual_dict(d):
        out = dict(d)
        for ch, params in current_models.items():
            out[ch + "_residual"] = apply_current_residual(d[ch], d["torque"], d["speed"], params)
        return out

    healthy_data_r = {code: residual_dict(d) for code, d in healthy_data.items()}

    results = {}
    for label, current_keys in [("A_raw_current", ("phase_current_1", "phase_current_2")),
                                  ("B_physics_residual", ("phase_current_1_residual", "phase_current_2_residual"))]:
        print(f"\n{'=' * 70}\nMETHOD {label}\n{'=' * 70}")
        src = healthy_data if label == "A_raw_current" else healthy_data_r

        fit_arr = np.concatenate([stack_nodes(src[c], current_keys)[fit_slices[c]] for c in src], axis=0)
        calib_arr = np.concatenate([stack_nodes(src[c], current_keys)[calib_slices[c]] for c in src], axis=0)
        test_normal_arr = np.concatenate([stack_nodes(src[c], current_keys)[test_slices[c]] for c in src], axis=0)

        scaler = StandardScaler().fit(fit_arr.reshape(-1, num_nodes))

        def scale(a):
            n, t, f = a.shape
            return scaler.transform(a.reshape(-1, f)).reshape(n, t, f).astype(np.float32)

        fit_scaled, calib_scaled, test_normal_scaled = scale(fit_arr), scale(calib_arr), scale(test_normal_arr)

        print(f"  training GDN (nodes={num_nodes}, current={'residual' if 'residual' in label else 'raw'}) ...")
        model, calib_mse = train_gdn(fit_scaled, calib_scaled, num_nodes)

        calib_err = per_file_error(model, calib_scaled)
        median = np.median(calib_err, axis=0)
        q75, q25 = np.percentile(calib_err, [75, 25], axis=0)
        iqr = np.maximum(q75 - q25, 1e-8)
        normal_err = per_file_error(model, test_normal_scaled)
        normal_score = gdn_score(normal_err, median, iqr)

        rows = []
        for code in ALL_DAMAGED_CODES:
            d = load_bearing_with_physics(code) if label == "A_raw_current" else residual_dict(load_bearing_with_physics(code))
            fault_arr = scale(stack_nodes(d, current_keys))
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

    print(f"\n{'=' * 70}\nCOMPARISON: raw current vs. physics-residual current\n{'=' * 70}")
    print(f"{'bearing':<8}{'category':<12}{'raw AUROC':>12}{'residual AUROC':>16}{'delta':>10}")
    rows_a = {r[0]: r[3] for r in results["A_raw_current"]["rows"]}
    rows_b = {r[0]: r[3] for r in results["B_physics_residual"]["rows"]}
    for code in ALL_DAMAGED_CODES:
        a, b = rows_a[code], rows_b[code]
        print(f"{code:<8}{category_of(code):<12}{a:>12.3f}{b:>16.3f}{b - a:>+10.3f}")

    report = {"current_models": current_models, "results": results}
    with open(OUT_DIR / "paderborn_physics_gdn_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {OUT_DIR / 'paderborn_physics_gdn_report.json'}")


if __name__ == "__main__":
    main()
