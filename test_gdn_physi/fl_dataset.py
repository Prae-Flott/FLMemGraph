"""
Multi-source client split for the federated system: robo3er's 5 robots as
5 real, non-IID federated clients -- the actual "多来源" (multi-source)
data this project has, as opposed to a synthetic non-IID partition.

Uses `data/robo3er/partition.pkl`'s existing `data_indices` (the same
per-robot partition the rest of FL-bench's framework, e.g. IFCAAE, already
trains on -- reused here rather than re-deriving it, so this federated
system's client boundaries match the project's established convention).
Verified client composition (2026-08, current on-disk data.npy):

    client 0 (robot00): n=378,  normal=279, Broken Pipe=25, Low battery=74
    client 1 (robot01): n=354,  normal=148, cable trapped=206
    client 2 (robot02): n=178,  normal=178 (no faults at all)
    client 3 (robot03): n=367,  normal=333, Broken Pipe=34
    client 4 (robot04): n=4766, normal=4617, stuck=149

This is a genuinely severe non-IID split (robot04 alone is 79% of all
data and the ONLY source of `stuck`; robot02 contributes no fault
evaluation data at all, only normal-regime diversity for the shared
memory). Exactly the kind of heterogeneity the memory-only federation
(vs. weight/gradient averaging) is meant to handle without one dominant
client's data distribution swamping the others -- see `fl_model.py`'s
module docstring: encoders/structure heads never leave the client, only
the memory codebook is exchanged.
"""
import json
import pickle
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_robo3er  # noqa: E402
from kinematics import fit_kinematic_params, apply_kinematic_residual, residual_feature_names  # noqa: E402
import feature_groups  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "robo3er"  # local copy, see dataset.py's comment

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15


class Client:
    def __init__(self, client_id: int, robot_name: str, fit_idx, calib_idx, test_normal_idx, fault_idx_by_label):
        self.client_id = client_id
        self.robot_name = robot_name
        self.fit_idx = fit_idx
        self.calib_idx = calib_idx
        self.test_normal_idx = test_normal_idx
        self.fault_idx_by_label = fault_idx_by_label  # {label_id: idx array}


def load_fl_clients(active_feature_groups=("kinematic_core",)):
    """Returns (data [N,T,F], targets [N], cols, label_map, clients: list[Client]).
    `data` already has the kinematic residual applied PER CLIENT (each
    robot fits its own k_v/k_w from its own fit split -- see kinematics.py
    -- so a robot-specific wheel-radius/track-width difference, e.g. from
    wear, shows up as a different fitted constant per client rather than
    forcing one global fit across robots with different physical
    calibration)."""
    data, targets, cols, label_map, robot_map = load_robo3er(drop_dead=True)
    if active_feature_groups is not None:
        data, cols = feature_groups.select_columns(data, cols, list(active_feature_groups))

    with open(DATA_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    client_indices = partition["data_indices"]

    clients = []
    data_out = data.copy()
    for client_id, idx_dict in enumerate(client_indices):
        all_idx = np.array(idx_dict["train"] + idx_dict["val"] + idx_dict["test"], dtype=int)
        normal_idx = np.sort(all_idx[targets[all_idx] == 0])

        n = len(normal_idx)
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        fit_idx = normal_idx[:n_fit]
        calib_idx = normal_idx[n_fit : n_fit + n_calib]
        test_normal_idx = normal_idx[n_fit + n_calib :]

        fault_idx_by_label = {}
        for label_id in sorted(set(targets[all_idx].tolist())):
            if label_id == 0:
                continue
            fault_idx = all_idx[targets[all_idx] == label_id]
            if len(fault_idx) > 0:
                fault_idx_by_label[label_id] = fault_idx

        if len(fit_idx) >= 10:  # only fit per-client kinematics if there's enough data to fit robustly
            kin_params = fit_kinematic_params(data, fit_idx, cols)
            data_out[all_idx] = apply_kinematic_residual(data[all_idx], cols, kin_params)

        clients.append(Client(
            client_id=client_id,
            robot_name=robot_map.get(str(client_id), f"robot{client_id:02d}"),
            fit_idx=fit_idx, calib_idx=calib_idx, test_normal_idx=test_normal_idx,
            fault_idx_by_label=fault_idx_by_label,
        ))

    cols_renamed = residual_feature_names(cols)
    return data_out, targets, cols_renamed, label_map, clients
