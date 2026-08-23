"""
Shared dataloader for src/: Sielaff's RED-SEVERITY fault variant
(`data/sielaff_red/`), mirroring `dataset.py`'s role for the coarse
5-error-group `data/sielaff/` variant -- kept as a SEPARATE module rather
than a parameterized DATA_DIR argument because the two are genuinely
different datasets (different labels, different windows, built by a
different pipeline), not just two configs of the same loader.

`data/sielaff_red/` is built by `benchmark/datasets/build_sielaff_red.py`
directly from the raw per-machine log tables (NOT derived from
`data/sielaff/`), labeling each 4h sliding window (8 x 30min buckets,
stride 2 -- same window geometry as `data/sielaff/`, see `dataset.py`'s
docstring) against the 47 RED-severity error IDs in
`data/sielaff_data/sielaff_ground_truth.md`'s severity scheme:
  0 = no_error        (no logged error of any severity in the window)
  1 = red              (>=1 RED-severity error ID present in the window)
  2 = non_red_error    (only orange/green errors -- a contrast class,
                        excluded from the normal fit/calib reference set
                        the same way any other non-zero label is)

Verified against `benchmark/run_sielaff_red_v2_1_federated.py`'s
`build_clients()` (the canonical RED-severity federated script) -- same
`partition.pkl` schema (10 machines, `separation.total`/`data_indices`),
same fit/calib/test_normal split formula.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "sielaff_red"

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15


def load_sielaff_red():
    """Returns (data [N, window_size, F], targets [N], feature_columns,
    class_names {'0': 'no_error', '1': 'red', '2': 'non_red_error'},
    machine_ids [10], red_id_names {error_id_str: name})."""
    data = np.load(DATA_DIR / "data.npy")
    targets = np.load(DATA_DIR / "targets.npy")
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    cols = meta["feature_columns"]
    label_map = meta["class_names"]
    machine_ids = meta.get("machine_ids", [])
    red_id_names = meta.get("red_id_names", {})
    return data, targets, cols, label_map, machine_ids, red_id_names


def load_machine_index_sets():
    """Same partition.pkl schema/logic as `dataset.load_machine_index_sets`,
    just pointed at `data/sielaff_red/partition.pkl`."""
    with open(DATA_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    machine_indices = []
    for client_id in range(partition["separation"]["total"]):
        d = partition["data_indices"][client_id]
        idx = np.concatenate([d["train"], d["val"], d["test"]]).astype(np.int64)
        machine_indices.append(np.sort(idx))
    return machine_indices


def split_normal(machine_indices, targets, fit_fraction=FIT_FRACTION, calib_fraction=CALIB_FRACTION):
    """Identical logic to `dataset.split_normal` -- per-machine chronological
    fit/calib/test_normal split of label==0 ("no_error") windows, pooled
    across machines. `non_red_error` (label 2) is a contrast class, NOT
    "normal" -- it is correctly excluded here the same way any fault label
    is, matching `run_sielaff_red_v2_1_federated.py`'s own `targets[idx]==0`
    filter."""
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
