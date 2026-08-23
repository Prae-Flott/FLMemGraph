"""
Shared dataloader for src/: one place for loading Sielaff (10 reverse-
vending machines), instead of each benchmark script re-implementing (and
risking silently diverging from) the same `load_and_split`/scaling logic
-- mirrors `src/robo3er/dataset.py`'s role for robo3er.

The raw per-machine CSV/xlsx processing (statistic_values.csv pivoting,
error-code -> group mapping, windowing) is NOT reproduced here -- that
pipeline lives in `~/Projects/FL-bench/data/generate_sielaff_data.py`
(a sibling project this dataset was originally migrated from, see
`memory/fl-bench-migrations.md`) and already produced the on-disk
`data/sielaff/{data.npy,targets.npy,metadata.json,partition.pkl}` this
module reads. Only the LOADING/SPLITTING side (the part every benchmark
script here actually needs) is consolidated into src/, matching this
project's "migrate as real standalone code for what we actually use, not
a framework pointer" convention.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "sielaff"

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15


def load_sielaff():
    """Returns (data [N, window_size, F], targets [N], feature_columns,
    class_names {label_id_str: name}, machine_ids [10])."""
    data = np.load(DATA_DIR / "data.npy")
    targets = np.load(DATA_DIR / "targets.npy")
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    cols = meta["feature_columns"]
    label_map = meta["class_names"]
    machine_ids = meta.get("machine_ids", [])
    return data, targets, cols, label_map, machine_ids


def load_machine_index_sets():
    """Returns a list of 10 sorted index arrays into `data`/`targets`, one
    per machine (train+val+test pooled -- this project's convention is its
    own fit/calib/test_normal chronological split per machine, not the
    upstream partition's own train/val/test split, see `split_normal`)."""
    with open(DATA_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    machine_indices = []
    for client_id in range(partition["separation"]["total"]):
        d = partition["data_indices"][client_id]
        idx = np.concatenate([d["train"], d["val"], d["test"]]).astype(np.int64)
        machine_indices.append(np.sort(idx))
    return machine_indices


def split_normal(machine_indices, targets, fit_fraction=FIT_FRACTION, calib_fraction=CALIB_FRACTION):
    """Per-machine chronological fit/calib/test_normal split (each
    machine's own normal-label indices split in on-disk order, not
    shuffled), pooled across all machines -- matches every centralized
    Sielaff script's convention. Returns (fit_idx, calib_idx,
    test_normal_idx), each a pooled, sorted-within-machine index array."""
    fit_idx, calib_idx, test_normal_idx = [], [], []
    for idx in machine_indices:
        normal_idx = idx[targets[idx] == 0]
        n = len(normal_idx)
        n_fit = int(n * fit_fraction)
        n_calib = int(n * calib_fraction)
        fit_idx.append(normal_idx[:n_fit])
        calib_idx.append(normal_idx[n_fit : n_fit + n_calib])
        test_normal_idx.append(normal_idx[n_fit + n_calib :])
    return np.concatenate(fit_idx), np.concatenate(calib_idx), np.concatenate(test_normal_idx)


def fit_scaler(data, fit_idx):
    scaler = StandardScaler()
    scaler.fit(data[fit_idx].reshape(-1, data.shape[-1]))
    return scaler


def scale(windows, scaler):
    n, t, f = windows.shape
    return scaler.transform(windows.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
