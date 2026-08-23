"""
Multi-source client split for Sielaff's RED-SEVERITY variant -- mirrors
`fl_dataset.py`'s role for the coarse `data/sielaff/` variant, pointed at
`data/sielaff_red/` instead. Verified against
`benchmark/run_sielaff_red_v2_1_federated.py`'s `build_clients()`
(client_id/all_idx/fit_idx/calib_idx/test_normal_idx construction is the
same formula) and `benchmark/diagnose_sielaff_red_localization_federated.py`
(same base split, that script additionally tracks a `red_idx` field this
module generalizes into `fault_idx_by_label` -- covers BOTH non-zero
labels, "red" (1) and the "non_red_error" contrast class (2), not just
red alone).
"""
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from red_dataset import load_sielaff_red, load_machine_index_sets, FIT_FRACTION, CALIB_FRACTION  # noqa: E402


class Client:
    def __init__(self, client_id: int, machine_id: str, all_idx, fit_idx, calib_idx, test_normal_idx, fault_idx_by_label):
        self.client_id = client_id
        self.machine_id = machine_id
        self.all_idx = all_idx
        self.fit_idx = fit_idx
        self.calib_idx = calib_idx
        self.test_normal_idx = test_normal_idx
        self.fault_idx_by_label = fault_idx_by_label  # {1: red_idx, 2: non_red_error_idx}


def load_fl_clients(fit_fraction=FIT_FRACTION, calib_fraction=CALIB_FRACTION):
    """Returns (data [N, window_size, F], targets [N], cols, label_map,
    machine_ids, red_id_names, clients: list[Client]), one Client per machine."""
    data, targets, cols, label_map, machine_ids, red_id_names = load_sielaff_red()
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
            client_id=client_id, machine_id=machine_id, all_idx=all_idx,
            fit_idx=fit_idx, calib_idx=calib_idx, test_normal_idx=test_normal_idx,
            fault_idx_by_label=fault_idx_by_label,
        ))

    return data, targets, cols, label_map, machine_ids, red_id_names, clients
