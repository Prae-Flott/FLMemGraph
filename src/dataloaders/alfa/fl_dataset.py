"""
Fault-type-client split for ALFA -- mirrors
`src/dataloaders/sielaff/red_fl_dataset.py`'s role, pointed at `data/alfa/`.
Each `Client` here is a FAULT TYPE (engine/aileron/rudder/elevator/
aileron_rudder_combo/no_failure), not a physical unit -- see
`src/dataloaders/alfa/dataset.py`'s module docstring for why.
"""
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_alfa, load_client_index_sets, FIT_FRACTION, CALIB_FRACTION  # noqa: E402


class Client:
    def __init__(self, client_id: int, client_name: str, all_idx, fit_idx, calib_idx, test_normal_idx, fault_idx_by_label):
        self.client_id = client_id
        self.client_name = client_name
        self.all_idx = all_idx
        self.fit_idx = fit_idx
        self.calib_idx = calib_idx
        self.test_normal_idx = test_normal_idx
        self.fault_idx_by_label = fault_idx_by_label  # {label_id: idx}, empty for the no_failure client


def load_fl_clients(fit_fraction=FIT_FRACTION, calib_fraction=CALIB_FRACTION):
    """Returns (data [N, window_size, F], targets [N], cols, label_map,
    client_names, clients: list[Client]), one Client per fault-type group."""
    data, targets, cols, label_map, client_names = load_alfa()
    client_indices = load_client_index_sets()

    clients = []
    for client_id, all_idx in enumerate(client_indices):
        normal_idx = all_idx[targets[all_idx] == 0]
        n = len(normal_idx)
        n_fit = int(n * fit_fraction)
        n_calib = int(n * calib_fraction)
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

        client_name = client_names[client_id] if client_id < len(client_names) else f"client{client_id}"
        clients.append(Client(
            client_id=client_id, client_name=client_name, all_idx=all_idx,
            fit_idx=fit_idx, calib_idx=calib_idx, test_normal_idx=test_normal_idx,
            fault_idx_by_label=fault_idx_by_label,
        ))

    return data, targets, cols, label_map, client_names, clients
