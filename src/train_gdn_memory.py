#!/usr/bin/env python3
"""
Adds a discrete prototypical memory (FedPM, arXiv:2604.04475 -- see
`gdn_memory_model.py` for the full adaptation writeup) to the physics-
residual, kinematic-core-only GDN pipeline from `train_gdn_physics.py`.
Same data, same feature set (ACTIVE_FEATURE_GROUPS=["kinematic_core"], 38
features), same physics residual, same dense-sliding/history-30 protocol,
same fit/calib/test_normal split -- the ONLY change is GDN -> GDNMemory
(adds a VQ-VAE-style codebook + two commitment loss terms) and an
additional "quantization distance" anomaly signal computed alongside
GDN's existing prediction-error signal. This isolates "does adding memory
help" as a single-variable ablation against
`gdn_physics_separability_report.json` (the no-memory run already on disk
for this exact feature set).

Three anomaly scores are computed and reported side by side, all through
the SAME GDN-style (per-feature median/IQR normalize, then max) rule:
  - "pred"      : prediction squared error only (identical to
                  train_gdn_physics.py's score -- the no-memory baseline)
  - "quant"     : quantization distance only (||z - nearest prototype||^2
                  per node) -- a purely new signal from the memory module
  - "pred+quant": both concatenated into one 2N-dim per-window error
                  vector before the same median/IQR-normalize-then-max
                  rule, so whichever signal is more anomalous for a given
                  fault type dominates the max automatically, no manual
                  weighting.

Usage:
    python3 train_gdn_memory.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gdn_memory_model import GDNMemory  # noqa: E402
from kinematics import (  # noqa: E402
    fit_kinematic_params,
    apply_kinematic_residual,
    residual_feature_names,
)
from dataset import (  # noqa: E402
    load_robo3er,
    split_normal,
    fit_scaler,
    scale,
    gdn_score,
)
import feature_groups  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "robo3er"  # local copy, see dataset.py's comment
OUT_DIR = REPO_ROOT / "checkpoints"
NO_MEMORY_REPORT = OUT_DIR / "gdn_physics_separability_report.json"

ACTIVE_FEATURE_GROUPS = ["kinematic_core"]  # matches train_gdn_physics.py's current setting

BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
SEED = 42
EMBED_DIM = 64
TOP_K = 15
HISTORY_LEN = 30
NUM_PROTOTYPES = 256  # FedPM's own M
BETA = 0.25  # weight on L_ME (encoder commitment); L_MC always weight 1, matching the paper
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_split():
    data, targets, cols, label_map, _ = load_robo3er(drop_dead=True)
    if ACTIVE_FEATURE_GROUPS is not None:
        data, cols = feature_groups.select_columns(data, cols, ACTIVE_FEATURE_GROUPS)
    fit_idx, calib_idx, test_normal_idx = split_normal(targets)
    return data, targets, cols, label_map, fit_idx, calib_idx, test_normal_idx


def make_dense_pairs(scaled_windows, history_len=HISTORY_LEN):
    n, t, f = scaled_windows.shape
    num_targets = t - history_len
    history = np.empty((n, num_targets, history_len, f), dtype=np.float32)
    target = np.empty((n, num_targets, f), dtype=np.float32)
    for i in range(num_targets):
        history[:, i] = scaled_windows[:, i : i + history_len]
        target[:, i] = scaled_windows[:, i + history_len]
    window_id = np.repeat(np.arange(n), num_targets)
    return (
        history.reshape(n * num_targets, history_len, f),
        target.reshape(n * num_targets, f),
        window_id,
    )


def make_loader(history, target, batch_size, shuffle):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(history), torch.from_numpy(target))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def per_sample_error(model, history, target, batch_size=BATCH_SIZE):
    """Returns (pred_sq_err [M,N], quant_dist [M,N])."""
    model.eval()
    pred_errs, quant_dists = [], []
    for i in range(0, len(history), batch_size):
        h = torch.from_numpy(history[i : i + batch_size]).to(DEVICE)
        t = torch.from_numpy(target[i : i + batch_size]).to(DEVICE)
        pred, z, z_q, quant_dist = model(h, return_memory_diagnostics=True)
        pred_errs.append(((pred - t) ** 2).cpu().numpy())
        quant_dists.append(quant_dist.cpu().numpy())
    return np.concatenate(pred_errs, axis=0), np.concatenate(quant_dists, axis=0)


def aggregate_by_window(per_sample_err, window_id, num_windows):
    out = np.zeros((num_windows, per_sample_err.shape[1]), dtype=np.float64)
    counts = np.zeros(num_windows, dtype=np.int64)
    np.add.at(out, window_id, per_sample_err)
    np.add.at(counts, window_id, 1)
    return (out / counts[:, None]).astype(np.float32)


def per_window_error(model, scaled_windows, batch_size=BATCH_SIZE):
    """Returns (pred_err [W,N], quant_err [W,N])."""
    history, target, window_id = make_dense_pairs(scaled_windows)
    pred_err, quant_dist = per_sample_error(model, history, target, batch_size)
    num_windows = len(scaled_windows)
    return (
        aggregate_by_window(pred_err, window_id, num_windows),
        aggregate_by_window(quant_dist, window_id, num_windows),
    )


def train(model, fit_history, fit_target, calib_scaled):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    loader = make_loader(fit_history, fit_target, BATCH_SIZE, shuffle=True)

    best_calib_mse = float("inf")
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        fit_loss = fit_pred_loss = fit_me_loss = fit_mc_loss = 0.0
        for h, t in loader:
            h, t = h.to(DEVICE), t.to(DEVICE)
            optimizer.zero_grad()
            pred, z, z_q, _ = model(h, return_memory_diagnostics=True)
            l_pred = criterion(pred, t)
            l_me, l_mc = model.memory.commitment_losses(z, z_q)
            loss = l_pred + BETA * l_me + l_mc
            loss.backward()
            optimizer.step()
            bs = h.size(0)
            fit_loss += loss.item() * bs
            fit_pred_loss += l_pred.item() * bs
            fit_me_loss += l_me.item() * bs
            fit_mc_loss += l_mc.item() * bs
        n = len(fit_history)
        fit_loss, fit_pred_loss, fit_me_loss, fit_mc_loss = (
            fit_loss / n, fit_pred_loss / n, fit_me_loss / n, fit_mc_loss / n
        )

        calib_pred_err, _ = per_window_error(model, calib_scaled)
        calib_mse = float(calib_pred_err.mean())
        util = model.memory.codebook_utilization()
        print(
            f"epoch {epoch:3d}  loss={fit_loss:.4f} (pred={fit_pred_loss:.4f} "
            f"L_ME={fit_me_loss:.4f} L_MC={fit_mc_loss:.4f})  calib_pred_mse={calib_mse:.6f}  "
            f"codebook_util={util:.1%}"
        )
        if calib_mse < best_calib_mse:
            best_calib_mse = calib_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_calib_mse


def combined_score(pred_err, quant_err, median, iqr):
    """Concatenate pred error and quant distance into one 2N-dim vector
    per window, then apply the same GDN-style median/IQR-normalize-then-
    max rule -- whichever signal is more anomalous wins the max."""
    combined = np.concatenate([pred_err, quant_err], axis=1)
    return gdn_score(combined, median, iqr)


def main():
    data, targets, cols, label_map, fit_idx, calib_idx, test_normal_idx = load_and_split()
    num_nodes = data.shape[2]

    kin_params = fit_kinematic_params(data, fit_idx, cols)
    data = apply_kinematic_residual(data, cols, kin_params)
    cols_renamed = residual_feature_names(cols)

    print(
        f"normal windows -> fit={len(fit_idx)} calib={len(calib_idx)} "
        f"test={len(test_normal_idx)}   nodes(sensors)={num_nodes}  "
        f"history_len={HISTORY_LEN}  num_prototypes={NUM_PROTOTYPES}  beta={BETA}"
    )

    scaler = fit_scaler(data, fit_idx)
    fit_scaled = scale(data[fit_idx], scaler)
    calib_scaled = scale(data[calib_idx], scaler)
    fit_history, fit_target, _ = make_dense_pairs(fit_scaled)
    print(f"dense (history, target) pairs for training: {len(fit_history)}")

    model = GDNMemory(num_nodes=num_nodes, window_size=HISTORY_LEN, embed_dim=EMBED_DIM,
                       top_k=TOP_K, num_prototypes=NUM_PROTOTYPES).to(DEVICE)
    print(f"\ntraining GDN+memory (embed_dim={EMBED_DIM}, top_k={TOP_K}, M={NUM_PROTOTYPES})...")
    model, best_calib_mse = train(model, fit_history, fit_target, calib_scaled)
    print(f"\nbest calib pred MSE: {best_calib_mse:.6f}")
    print(f"final codebook utilization: {model.memory.codebook_utilization():.1%} "
          f"of {NUM_PROTOTYPES} prototypes used at least once during training")

    calib_pred_err, calib_quant_err = per_window_error(model, calib_scaled)
    pred_median, pred_iqr = np.median(calib_pred_err, axis=0), None
    q75, q25 = np.percentile(calib_pred_err, [75, 25], axis=0)
    pred_iqr = np.maximum(q75 - q25, 1e-8)

    quant_median = np.median(calib_quant_err, axis=0)
    q75, q25 = np.percentile(calib_quant_err, [75, 25], axis=0)
    quant_iqr = np.maximum(q75 - q25, 1e-8)

    combined_median = np.concatenate([pred_median, quant_median])
    combined_iqr = np.concatenate([pred_iqr, quant_iqr])

    test_normal_scaled = scale(data[test_normal_idx], scaler)
    normal_pred_err, normal_quant_err = per_window_error(model, test_normal_scaled)
    scores_normal = {
        "pred": gdn_score(normal_pred_err, pred_median, pred_iqr),
        "quant": gdn_score(normal_quant_err, quant_median, quant_iqr),
        "pred+quant": combined_score(normal_pred_err, normal_quant_err, combined_median, combined_iqr),
    }
    thresholds = {k: float(np.percentile(v, 95)) for k, v in scores_normal.items()}

    guard_band_path = DATA_DIR / "stuck_severity_split.json"
    guard_band_excluded = set()
    if guard_band_path.exists():
        with open(guard_band_path) as f:
            guard_band_excluded = set(json.load(f)["transition_excluded_idx"])

    report = {
        "config": {"embed_dim": EMBED_DIM, "top_k": TOP_K, "epochs": EPOCHS, "history_len": HISTORY_LEN,
                    "num_nodes": num_nodes, "num_prototypes": NUM_PROTOTYPES, "beta": BETA,
                    "active_feature_groups": ACTIVE_FEATURE_GROUPS,
                    "codebook_utilization": model.memory.codebook_utilization()},
        "fault_types": {},
    }

    all_rows = {k: [] for k in scores_normal}
    for label_id_str, name in label_map.items():
        label_id = int(label_id_str)
        if label_id == 0:
            continue
        fault_idx = np.where(targets == label_id)[0]
        if guard_band_excluded:
            keep_mask = np.array([idx not in guard_band_excluded for idx in fault_idx])
            fault_idx = fault_idx[keep_mask]
        if len(fault_idx) == 0:
            continue
        fault_scaled = scale(data[fault_idx], scaler)
        fault_pred_err, fault_quant_err = per_window_error(model, fault_scaled)

        fault_scores = {
            "pred": gdn_score(fault_pred_err, pred_median, pred_iqr),
            "quant": gdn_score(fault_quant_err, quant_median, quant_iqr),
            "pred+quant": combined_score(fault_pred_err, fault_quant_err, combined_median, combined_iqr),
        }

        report["fault_types"][name] = {"label_id": label_id, "n": int(len(fault_idx))}
        for score_name, fault_score in fault_scores.items():
            normal_score = scores_normal[score_name]
            labels = np.concatenate([np.zeros(len(normal_score)), np.ones(len(fault_score))])
            auroc = float(sk_metrics.roc_auc_score(labels, np.concatenate([normal_score, fault_score])))
            recall = float((fault_score > thresholds[score_name]).mean())
            report["fault_types"][name][f"auroc_{score_name}"] = auroc
            report["fault_types"][name][f"recall_at_p95_{score_name}"] = recall
            all_rows[score_name].append((name, auroc))

    print(f"\n{'fault type':<16}{'AUROC (pred only)':>20}{'AUROC (quant only)':>21}{'AUROC (pred+quant)':>21}")
    for name in report["fault_types"]:
        p = report["fault_types"][name]["auroc_pred"]
        q = report["fault_types"][name]["auroc_quant"]
        c = report["fault_types"][name]["auroc_pred+quant"]
        print(f"{name:<16}{p:>20.3f}{q:>21.3f}{c:>21.3f}")

    if NO_MEMORY_REPORT.exists():
        with open(NO_MEMORY_REPORT) as f:
            no_mem = json.load(f)
        print(f"\n{'fault type':<16}{'no-memory AUROC':>17}{'pred+quant AUROC':>18}{'delta':>9}")
        for name in report["fault_types"]:
            if name in no_mem["fault_types"]:
                b = no_mem["fault_types"][name]["auroc"]
                c = report["fault_types"][name]["auroc_pred+quant"]
                print(f"{name:<16}{b:>17.3f}{c:>18.3f}{c-b:>9.3f}")

    torch.save({"model_state_dict": model.state_dict(), "eval_report": report},
               OUT_DIR / "gdn_memory_robo3er.pth")
    with open(OUT_DIR / "gdn_memory_separability_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {OUT_DIR / 'gdn_memory_robo3er.pth'}")
    print(f"saved -> {OUT_DIR / 'gdn_memory_separability_report.json'}")


if __name__ == "__main__":
    main()
