#!/usr/bin/env python3
"""
Fair 3-way comparison on Paderborn, all using the SAME 6-channel node set
(vibration_1, phase_current_1, phase_current_2, force, speed, torque) and
the SAME fit/calib/test_normal split as
`run_paderborn_joint_prototype.py`, so the only variable is the model
architecture:

  A. GDN       -- plain forecasting (src.gdn_model.GDN), no memory, no
                  physics prior at all -- the "no structure beyond
                  learned graph attention" baseline.
  B. AE        -- plain whole-window reconstruction
                  (src.conv_autoencoder.ConvAutoEncoder), no graph, no
                  memory -- the "no relational structure at all" baseline.
  C. Joint Prototype (full, D from the ablation study) -- reused from
     `checkpoints/paderborn_joint_prototype_report.json` (deterministic,
     same seed/data/split, no need to retrain) --
     `src.joint_prototype_model.JointPrototypeGDN`.

Every prior Paderborn script (run_paderborn_fl_model.py,
run_paderborn_physics_gdn.py, run_paderborn_torque_residual_gdn.py) used
only 3-4 channels; this is the first GDN/AE baseline run on the full
6-channel set the joint-prototype model uses, so the comparison in
`memory/paderborn-joint-prototype.md` (vs. the 3-channel GDN-raw
baseline) is not perfectly apples-to-apples on channel count -- this
script fixes that.

Usage:
    python3 run_paderborn_6ch_comparison.py
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
from conv_autoencoder import ConvAutoEncoder  # noqa: E402
from datasets.paderborn_adapter import (  # noqa: E402
    load_bearing_with_physics, HEALTHY_CODES, ALL_DAMAGED_CODES, category_of, damage_origin_of,
)

OUT_DIR = REPO_ROOT / "checkpoints"

NODE_NAMES = ["vibration_1", "phase_current_1", "phase_current_2", "force", "speed", "torque"]
FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15
BATCH_SIZE = 256
EPOCHS = 12
LR = 1e-3
SEED = 42
EMBED_DIM = 64
TOP_K = 15
HISTORY_LEN = 64      # for GDN forecasting
DENSE_STRIDE = 8
WINDOW_LEN = 64        # for AE reconstruction (matches joint-prototype's window)
WINDOW_STRIDE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stack_nodes(d):
    return np.stack([d[name] for name in NODE_NAMES], axis=-1).astype(np.float32)


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


def make_windows(arr, window_len=WINDOW_LEN, stride=WINDOW_STRIDE):
    n, t, f = arr.shape
    positions = list(range(0, t - window_len + 1, stride))
    windows = np.empty((n, len(positions), window_len, f), dtype=np.float32)
    for j, pos in enumerate(positions):
        windows[:, j] = arr[:, pos : pos + window_len]
    file_id = np.repeat(np.arange(n), len(positions))
    return windows.reshape(-1, window_len, f), file_id


@torch.no_grad()
def per_sample_error_gdn(model, history, target, batch_size=BATCH_SIZE):
    model.eval()
    errors = []
    for i in range(0, len(history), batch_size):
        h = torch.from_numpy(history[i : i + batch_size]).to(DEVICE)
        t = torch.from_numpy(target[i : i + batch_size]).to(DEVICE)
        pred = model(h)
        errors.append(((pred - t) ** 2).cpu().numpy())
    return np.concatenate(errors, axis=0)


def per_file_error_gdn(model, arr, batch_size=BATCH_SIZE):
    history, target, file_id = make_dense_pairs(arr)
    err = per_sample_error_gdn(model, history, target, batch_size)
    n = len(arr)
    out = np.zeros((n, err.shape[1]), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    np.add.at(out, file_id, err)
    np.add.at(counts, file_id, 1)
    return (out / counts[:, None]).astype(np.float32)


@torch.no_grad()
def per_file_error_ae(model, arr, batch_size=BATCH_SIZE):
    windows, file_id = make_windows(arr)
    model.eval()
    errs = []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(DEVICE)
        recon = model(batch)
        errs.append(((recon - batch) ** 2).mean(dim=1).cpu().numpy())
    err = np.concatenate(errs, axis=0)
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
        calib_mse = float(per_file_error_gdn(model, calib_arr).mean())
        print(f"    epoch {epoch:2d}  fit_mse={fit_loss:.6f}  calib_mse={calib_mse:.6f}")
        if calib_mse < best_calib_mse:
            best_calib_mse = calib_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def train_ae(fit_arr, calib_arr, num_nodes, window_size=WINDOW_LEN):
    torch.manual_seed(SEED)
    model = ConvAutoEncoder(num_nodes, window_size).to(DEVICE)
    fit_windows, _ = make_windows(fit_arr)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fit_windows)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    best_calib_mse, best_state = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        fit_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss = torch.nn.functional.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            fit_loss += loss.item() * batch.size(0)
        fit_loss /= len(fit_windows)
        calib_mse = float(per_file_error_ae(model, calib_arr).mean())
        print(f"    epoch {epoch:2d}  fit_mse={fit_loss:.6f}  calib_mse={calib_mse:.6f}")
        if calib_mse < best_calib_mse:
            best_calib_mse = calib_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def evaluate(name, per_file_error_fn, model, scale_fn, calib_arr, test_normal_arr):
    calib_err = per_file_error_fn(model, calib_arr)
    median = np.median(calib_err, axis=0)
    q75, q25 = np.percentile(calib_err, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    normal_err = per_file_error_fn(model, test_normal_arr)
    normal_score = gdn_score(normal_err, median, iqr)

    rows = []
    for code in ALL_DAMAGED_CODES:
        d = load_bearing_with_physics(code)
        fault_arr = scale_fn(stack_nodes(d))
        fault_err = per_file_error_fn(model, fault_arr)
        fault_score = gdn_score(fault_err, median, iqr)
        labels = np.concatenate([np.zeros(len(normal_score)), np.ones(len(fault_score))])
        scores = np.concatenate([normal_score, fault_score])
        auroc = float(sk_metrics.roc_auc_score(labels, scores))
        rows.append((code, category_of(code), damage_origin_of(code), auroc))
        print(f"    {code:<6}{category_of(code):<12}{damage_origin_of(code):<11}AUROC={auroc:.3f}")
    return rows


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    num_nodes = len(NODE_NAMES)

    print("loading healthy bearings (K001-K006), 6 channels ...")
    healthy_data = {code: load_bearing_with_physics(code) for code in HEALTHY_CODES}

    fit_list, calib_list, test_list = [], [], []
    for code, d in healthy_data.items():
        n = len(d["vibration_1"])
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        arr = stack_nodes(d)
        fit_list.append(arr[:n_fit])
        calib_list.append(arr[n_fit : n_fit + n_calib])
        test_list.append(arr[n_fit + n_calib :])
        print(f"  {code}: n={n} fit={n_fit} calib={n_calib} test_normal={n - n_fit - n_calib}")

    fit_arr = np.concatenate(fit_list, axis=0)
    calib_arr = np.concatenate(calib_list, axis=0)
    test_normal_arr = np.concatenate(test_list, axis=0)

    scaler = StandardScaler().fit(fit_arr.reshape(-1, num_nodes))

    def scale(a):
        n, t, f = a.shape
        return scaler.transform(a.reshape(-1, f)).reshape(n, t, f).astype(np.float32)

    fit_scaled, calib_scaled, test_normal_scaled = scale(fit_arr), scale(calib_arr), scale(test_normal_arr)

    results = {}

    print("\n" + "=" * 70 + "\nMETHOD A: plain GDN (forecasting, no memory, no physics prior)\n" + "=" * 70)
    model_gdn = train_gdn(fit_scaled, calib_scaled, num_nodes)
    rows_gdn = evaluate("GDN", per_file_error_gdn, model_gdn, scale, calib_scaled, test_normal_scaled)
    results["A_GDN"] = rows_gdn

    print("\n" + "=" * 70 + "\nMETHOD B: plain AE (reconstruction, no graph, no memory)\n" + "=" * 70)
    model_ae = train_ae(fit_scaled, calib_scaled, num_nodes)
    rows_ae = evaluate("AE", per_file_error_ae, model_ae, scale, calib_scaled, test_normal_scaled)
    results["B_AE"] = rows_ae

    print("\n" + "=" * 70 + "\nMETHOD C: Joint Prototype (full, D) -- reused from prior run\n" + "=" * 70)
    with open(OUT_DIR / "paderborn_joint_prototype_report.json") as f:
        jp_report = json.load(f)
    rows_jp = [(code, v["category"], v["origin"], v["D_full"]) for code, v in jp_report["bearings"].items()]

    print(f"\n{'bearing':<8}{'category':<12}{'A: GDN':>10}{'B: AE':>10}{'C: JointProto':>14}")
    all_rows = {"A_GDN": {r[0]: r for r in rows_gdn}, "B_AE": {r[0]: r for r in rows_ae},
                 "C_JointProto": {r[0]: r for r in rows_jp}}
    for code in ALL_DAMAGED_CODES:
        a = all_rows["A_GDN"][code][3]
        b = all_rows["B_AE"][code][3]
        c = all_rows["C_JointProto"][code][3]
        cat = all_rows["A_GDN"][code][1]
        print(f"{code:<8}{cat:<12}{a:>10.3f}{b:>10.3f}{c:>14.3f}")

    print(f"\n{'method':<16}{'mean (all 26)':>16}{'outer_ring':>12}{'inner_ring':>12}{'combined':>12}")
    summary = {}
    for key, label in [("A_GDN", "GDN"), ("B_AE", "AE"), ("C_JointProto", "JointProto")]:
        rows = list(all_rows[key].values())
        mean_all = float(np.mean([r[3] for r in rows]))
        cat_means = {}
        for cat in ["outer_ring", "inner_ring", "combined"]:
            subset = [r[3] for r in rows if r[1] == cat]
            cat_means[cat] = float(np.mean(subset)) if subset else None
        summary[key] = {"mean_all": mean_all, **cat_means}
        print(f"{label:<16}{mean_all:>16.3f}" + "".join(f"{cat_means[c]:>12.3f}" for c in ["outer_ring", "inner_ring", "combined"]))

    report = {"nodes": NODE_NAMES, "results": {k: [list(r) for r in v.values()] for k, v in all_rows.items()},
               "summary": summary}
    with open(OUT_DIR / "paderborn_6ch_comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {OUT_DIR / 'paderborn_6ch_comparison_report.json'}")


if __name__ == "__main__":
    main()
