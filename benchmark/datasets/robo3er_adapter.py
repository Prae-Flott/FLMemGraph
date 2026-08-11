"""
Adapter presenting robo3er through the benchmark's common interface:
`load()` -> (fit_windows, calib_windows, test_normal_windows,
{fault_name: fault_windows}), all [N, T, F] float32 arrays, normal-only
splits already scaled by a scaler fit on the fit split. Wraps
`src/robo3er/dataset.py` (already this project's own canonical robo3er
loader) rather than re-deriving anything.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "robo3er"))
from dataset import load_robo3er, split_normal, fit_scaler, scale  # noqa: E402


def load(pooled: bool = True):
    """pooled=True: all 5 robots' data pooled (matches
    `auto_encoder/train_centralized_ae.py`'s convention) -- the right mode
    for comparing centralized baselines. For a federated/per-client
    comparison use `test_gdn_physi/fl_dataset.py` directly instead."""
    data, targets, cols, label_map, _ = load_robo3er(drop_dead=True)
    fit_idx, calib_idx, test_normal_idx = split_normal(targets)
    scaler = fit_scaler(data, fit_idx)

    fit_w = scale(data[fit_idx], scaler)
    calib_w = scale(data[calib_idx], scaler)
    test_normal_w = scale(data[test_normal_idx], scaler)

    faults = {}
    for label_id_str, name in label_map.items():
        label_id = int(label_id_str)
        if label_id == 0:
            continue
        fault_idx = np.where(targets == label_id)[0]
        if len(fault_idx) == 0:
            continue
        faults[name] = scale(data[fault_idx], scaler)

    return fit_w, calib_w, test_normal_w, faults, cols
