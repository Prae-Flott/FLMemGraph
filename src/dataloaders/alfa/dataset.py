"""
Shared dataloader for src/: one place for loading ALFA (UAV fault/anomaly
flight data), mirroring `src/dataloaders/sielaff/dataset.py`'s role.

Unlike robo3er (5 robots) or Sielaff (10 machines), ALFA is a single
aircraft flown 47 times, so there is no natural per-unit client split --
`data/alfa/` groups windows into 6 clients by FAULT TYPE instead (engine,
aileron, rudder, elevator, aileron_rudder_combo, no_failure), each pooling
every flight that shares that fault. See `src/dataloaders/alfa/build_alfa.py`
for how `data/alfa/{data.npy,targets.npy,metadata.json,partition.pkl}` was
built from the raw per-flight CSVs in `data/alfa_data/`.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "alfa"

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15


def load_alfa():
    """Returns (data [N, window_size, F], targets [N], feature_columns,
    class_names {label_id_str: name}, client_names [6])."""
    data = np.load(DATA_DIR / "data.npy")
    targets = np.load(DATA_DIR / "targets.npy")
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    cols = meta["feature_columns"]
    label_map = meta["class_names"]
    client_names = meta.get("client_names", [])
    return data, targets, cols, label_map, client_names


def load_client_index_sets():
    """Returns a list of 6 sorted index arrays into `data`/`targets`, one per
    fault-type client (train+val+test pooled -- same convention as
    `sielaff.dataset.load_machine_index_sets`, val is always empty here since
    `build_alfa.py` only splits train/test)."""
    with open(DATA_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    client_indices = []
    for client_id in range(partition["separation"]["total"]):
        d = partition["data_indices"][client_id]
        idx = np.concatenate([d["train"], d["val"], d["test"]]).astype(np.int64)
        client_indices.append(np.sort(idx))
    return client_indices


def split_normal(client_indices, targets, fit_fraction=FIT_FRACTION, calib_fraction=CALIB_FRACTION):
    """Per-client chronological fit/calib/test_normal split (each client's
    own normal-label indices split in on-disk order, not shuffled), pooled
    across all clients -- matches `sielaff.dataset.split_normal`. Returns
    (fit_idx, calib_idx, test_normal_idx), each a pooled, sorted-within-client
    index array."""
    fit_idx, calib_idx, test_normal_idx = [], [], []
    for idx in client_indices:
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
