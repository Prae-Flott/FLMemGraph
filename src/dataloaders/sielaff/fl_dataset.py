"""
Multi-source client split for the federated system: Sielaff's 10 real
reverse-vending machines as 10 real federated clients -- mirrors
`src/robo3er/fl_dataset.py`'s role, reusing `data/sielaff/partition.pkl`'s
existing per-machine index sets (`dataset.load_machine_index_sets`) rather
than re-deriving anything.

Unlike robo3er, Sielaff has no per-client feature transform (no kinematic
residual) -- machines are the same hardware model, so no per-machine
physical-parameter refit is needed here.
"""
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_sielaff, load_machine_index_sets, FIT_FRACTION, CALIB_FRACTION  # noqa: E402


class Client:
    def __init__(self, client_id: int, machine_id: str, fit_idx, calib_idx, test_normal_idx, fault_idx_by_label):
        self.client_id = client_id
        self.machine_id = machine_id
        self.fit_idx = fit_idx
        self.calib_idx = calib_idx
        self.test_normal_idx = test_normal_idx
        self.fault_idx_by_label = fault_idx_by_label  # {label_id: idx array}


def load_fl_clients(fit_fraction=FIT_FRACTION, calib_fraction=CALIB_FRACTION):
    """Returns (data [N, window_size, F], targets [N], cols, label_map,
    clients: list[Client]), one Client per machine."""
    data, targets, cols, label_map, machine_ids = load_sielaff()
    machine_indices = load_machine_index_sets()

    clients = []
    for client_id, all_idx in enumerate(machine_indices):
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

        machine_id = machine_ids[client_id] if client_id < len(machine_ids) else f"machine{client_id}"
        clients.append(Client(
            client_id=client_id, machine_id=machine_id,
            fit_idx=fit_idx, calib_idx=calib_idx, test_normal_idx=test_normal_idx,
            fault_idx_by_label=fault_idx_by_label,
        ))

    return data, targets, cols, label_map, clients
