#!/usr/bin/env python3
"""
GDN (Deng & Hooi, AAAI 2021) forecaster for Sielaff -- a second real,
independent dataset (10 reverse-vending machines, `data/sielaff/`, derived
from the raw logs already checked into `data/sielaff_data/` via FL-bench's
`data/generate_sielaff_data.py`, not re-run here -- the already-generated
`data.npy`/`targets.npy`/`metadata.json`/`partition.pkl` were copied over,
same as `data/robo3er/` was), migrated from `~/Projects/FL-bench`'s
`test_gdn/train_sielaff_gdn.py`. Mirrors `train_gdn.py`'s (this project's
tuned robo3er GDN, see `baselines/gdn_baseline.GDNTunedBaseline`) protocol
but on Sielaff's 39-feature schema with an 8-step window instead of
robo3er's 60, using THIS project's own `src/gdn_model.GDN` (verified
byte-identical to FL-bench's `test_gdn/gdn_model.py`) rather than
duplicating the model.

GDN predicts the next step of every sensor from a learned graph-attention
hop over its most-similar sensors, scored via the same GDN convention used
everywhere else in this project: per-feature median/IQR normalize from a
held-out normal calibration set, then max over features.

Dense sliding pairs: window_size=8, HISTORY_LEN=4 leaves 4 target
positions per window (~4x the gradient signal per window vs. predicting
only the last step) -- same trick `train_gdn.py` uses for robo3er's
60-step windows, scaled down to fit Sielaff's much shorter window.

Per-machine chronological 70/15/15 split: `data.npy` concatenates all 10
machines' windows back-to-back, not one continuous stream, so fit/calib/
test-normal boundaries are computed per machine and then pooled (matches
robo3er's `dataset.py` convention of a pooled/centralized split).

Reliability masking (`RELIABILITY_RATIO`): unlike robo3er, several Sielaff
sensors (e.g. RingCamera-derived features) are only reported by a subset
of the 10 machines -- their calib-split IQR is then near-zero for the
machines that never report them, and dividing by that near-zero IQR would
let a trivial fluctuation dominate the max()-over-features GDN score.
Features whose calib IQR falls below `RELIABILITY_RATIO` of the median IQR
are excluded from the max(), not scored as -inf-worthy noise.

Usage:
    python3 run_sielaff_gdn.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
from gdn_model import GDN  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "sielaff"
OUT_DIR = REPO_ROOT / "checkpoints" / "sielaff"

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
SEED = 42
EMBED_DIM = 64
TOP_K = 15
HISTORY_LEN = 4  # window_size=8, leaves 4 dense targets per window
RELIABILITY_RATIO = 0.05  # features w/ calib IQR < 5% of median IQR excluded from max()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_machine_index_sets():
    with open(DATA_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    machine_indices = []
    for client_id in range(partition["separation"]["total"]):
        d = partition["data_indices"][client_id]
        idx = np.concatenate([d["train"], d["val"], d["test"]]).astype(np.int64)
        machine_indices.append(np.sort(idx))
    return machine_indices


def load_and_split():
    data = np.load(DATA_DIR / "data.npy")
    targets = np.load(DATA_DIR / "targets.npy")
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    cols = meta["feature_columns"]
    label_map = meta["class_names"]

    machine_indices = load_machine_index_sets()
    fit_idx, calib_idx, test_normal_idx = [], [], []
    for idx in machine_indices:
        normal_idx = idx[targets[idx] == 0]
        n = len(normal_idx)
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        fit_idx.append(normal_idx[:n_fit])
        calib_idx.append(normal_idx[n_fit : n_fit + n_calib])
        test_normal_idx.append(normal_idx[n_fit + n_calib :])

    fit_idx = np.concatenate(fit_idx)
    calib_idx = np.concatenate(calib_idx)
    test_normal_idx = np.concatenate(test_normal_idx)
    return data, targets, cols, label_map, fit_idx, calib_idx, test_normal_idx


def fit_scaler(data, fit_idx):
    scaler = StandardScaler()
    scaler.fit(data[fit_idx].reshape(-1, data.shape[-1]))
    return scaler


def scale(windows, scaler):
    n, t, f = windows.shape
    return scaler.transform(windows.reshape(-1, f)).reshape(n, t, f).astype(np.float32)


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
    model.eval()
    errors = []
    for i in range(0, len(history), batch_size):
        h = torch.from_numpy(history[i : i + batch_size]).to(DEVICE)
        t = torch.from_numpy(target[i : i + batch_size]).to(DEVICE)
        pred = model(h)
        err = (pred - t) ** 2
        errors.append(err.cpu().numpy())
    return np.concatenate(errors, axis=0)


def aggregate_by_window(per_sample_err, window_id, num_windows):
    out = np.zeros((num_windows, per_sample_err.shape[1]), dtype=np.float64)
    counts = np.zeros(num_windows, dtype=np.int64)
    np.add.at(out, window_id, per_sample_err)
    np.add.at(counts, window_id, 1)
    return (out / counts[:, None]).astype(np.float32)


def per_window_error(model, scaled_windows, batch_size=BATCH_SIZE):
    history, target, window_id = make_dense_pairs(scaled_windows)
    err = per_sample_error(model, history, target, batch_size)
    return aggregate_by_window(err, window_id, len(scaled_windows))


def train(model, fit_history, fit_target, calib_scaled_windows):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    loader = make_loader(fit_history, fit_target, BATCH_SIZE, shuffle=True)

    best_calib_mse = float("inf")
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        fit_loss = 0.0
        for h, t in loader:
            h, t = h.to(DEVICE), t.to(DEVICE)
            optimizer.zero_grad()
            pred = model(h)
            loss = criterion(pred, t)
            loss.backward()
            optimizer.step()
            fit_loss += loss.item() * h.size(0)
        fit_loss /= len(fit_history)

        calib_mse = float(per_window_error(model, calib_scaled_windows).mean())
        print(f"epoch {epoch:3d}  fit_mse={fit_loss:.6f}  calib_mse={calib_mse:.6f}")
        if calib_mse < best_calib_mse:
            best_calib_mse = calib_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_calib_mse


def reliable_feature_mask(iqr):
    threshold = RELIABILITY_RATIO * np.median(iqr)
    return iqr > threshold


def gdn_score(per_window_err, median, iqr, reliable_mask=None):
    normalized = (per_window_err - median) / iqr
    if reliable_mask is not None:
        normalized = np.where(reliable_mask[None, :], normalized, -np.inf)
    return normalized.max(axis=1)


def main():
    data, targets, cols, label_map, fit_idx, calib_idx, test_normal_idx = load_and_split()
    num_nodes = data.shape[2]
    print(
        f"normal windows -> fit={len(fit_idx)} calib={len(calib_idx)} "
        f"test={len(test_normal_idx)}   nodes(sensors)={num_nodes}  "
        f"history_len={HISTORY_LEN}  dense_targets_per_window={data.shape[1]-HISTORY_LEN}"
    )

    scaler = fit_scaler(data, fit_idx)
    fit_scaled = scale(data[fit_idx], scaler)
    calib_scaled = scale(data[calib_idx], scaler)
    fit_history, fit_target, _ = make_dense_pairs(fit_scaled)
    print(f"dense (history, target) pairs for training: {len(fit_history)}")

    model = GDN(num_nodes=num_nodes, window_size=HISTORY_LEN, embed_dim=EMBED_DIM, top_k=TOP_K).to(DEVICE)
    print(f"\ntraining GDN (embed_dim={EMBED_DIM}, top_k={TOP_K}, dense sliding, stride=1 within each window)...")
    model, best_calib_mse = train(model, fit_history, fit_target, calib_scaled)
    print(f"\nbest calib MSE: {best_calib_mse:.6f}")

    focus = [c for c in ["Helligkeit_Flaschenerkennung", "journal_count", "reject_count"] if c in cols]
    if focus:
        print("\nlearned graph neighborhoods (top-5 by cosine similarity of embeddings):")
        print(model.graph_summary(cols, focus, k=5))

    calib_err = per_window_error(model, calib_scaled)
    median = np.median(calib_err, axis=0)
    q75, q25 = np.percentile(calib_err, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    reliable_mask = reliable_feature_mask(iqr)
    excluded_features = [cols[i] for i in np.where(~reliable_mask)[0]]
    print(f"\nexcluded {len(excluded_features)}/{len(iqr)} features from GDN max() "
          f"(calib IQR < {RELIABILITY_RATIO:.0%} of median IQR): {excluded_features}")

    test_normal_scaled = scale(data[test_normal_idx], scaler)
    normal_err = per_window_error(model, test_normal_scaled)
    normal_score = gdn_score(normal_err, median, iqr, reliable_mask)
    threshold = float(np.percentile(normal_score, 95))
    fpr = float((normal_score > threshold).mean())

    report = {
        "config": {"embed_dim": EMBED_DIM, "top_k": TOP_K, "epochs": EPOCHS, "history_len": HISTORY_LEN, "dense": True},
        "n_features": int(num_nodes),
        "reliability_ratio": RELIABILITY_RATIO,
        "excluded_features": excluded_features,
        "threshold_p95": threshold,
        "normal": {"n": int(len(test_normal_idx)), "false_positive_rate": fpr},
        "fault_types": {},
    }

    rows = []
    full_z_by_fault = {}
    for label_id_str, name in label_map.items():
        label_id = int(label_id_str)
        if label_id == 0:
            continue
        fault_idx = np.where(targets == label_id)[0]
        if len(fault_idx) == 0:
            continue
        fault_scaled = scale(data[fault_idx], scaler)
        fault_err = per_window_error(model, fault_scaled)
        fault_score = gdn_score(fault_err, median, iqr, reliable_mask)

        labels = np.concatenate([np.zeros(len(normal_score)), np.ones(len(fault_score))])
        auroc = float(sk_metrics.roc_auc_score(labels, np.concatenate([normal_score, fault_score])))
        recall = float((fault_score > threshold).mean())

        z = (fault_err.mean(axis=0) - median) / iqr
        z_reliable_only = np.where(reliable_mask, z, -np.inf)
        top_sensors = [cols[i] for i in np.argsort(-z_reliable_only)[:3]]
        full_z_by_fault[name] = z

        report["fault_types"][name] = {
            "label_id": label_id,
            "n": int(len(fault_idx)),
            "auroc": auroc,
            "recall_at_p95": recall,
            "top_sensors_by_z": top_sensors,
        }
        rows.append((name, len(fault_idx), auroc, recall, top_sensors))

    rows.sort(key=lambda r: r[2], reverse=True)
    print(f"\nheld-out normal: n={len(test_normal_idx)}  FPR@p95={fpr:.3f}")
    print(f"{'fault type':<20}{'n':>6}{'AUROC':>9}{'recall@p95':>12}  top sensors (by z)")
    for name, n, auroc, recall, top_sensors in rows:
        print(f"{name:<20}{n:>6}{auroc:>9.3f}{recall:>12.3f}  {top_sensors}")

    torch.save({"model_state_dict": model.state_dict(), "eval_report": report}, OUT_DIR / "gdn_sielaff.pth")
    with open(OUT_DIR / "gdn_sielaff_report.json", "w") as f:
        json.dump(report, f, indent=2)

    import csv
    with open(OUT_DIR / "gdn_sielaff_full_z_per_class.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "feature", "z_score", "reliable"])
        for fault_name, z_vec in full_z_by_fault.items():
            for feat_idx, z_val in enumerate(z_vec):
                writer.writerow([fault_name, cols[feat_idx], float(z_val), bool(reliable_mask[feat_idx])])

    print(f"\nsaved -> {OUT_DIR / 'gdn_sielaff.pth'}")
    print(f"saved -> {OUT_DIR / 'gdn_sielaff_report.json'}")
    print(f"saved -> {OUT_DIR / 'gdn_sielaff_full_z_per_class.csv'}")


if __name__ == "__main__":
    main()
