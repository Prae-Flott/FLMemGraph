#!/usr/bin/env python3
"""
Does robo3er's fault data show feature-to-feature correlations that
differ noticeably from normal data -- and specifically, do the physically
meaningful pairs (wheel velocity vs. chassis twist, wheel PWM vs. current,
wheel ticks vs. wheel velocity, odometry vs. IMU) decouple during a fault?
This is the empirical question behind every physics-prior experiment in
this folder (kinematics.py, feature_groups.py) -- this script measures it
directly rather than assuming it.

For each fault type: correlation matrix on that fault's windows vs. the
correlation matrix on FIT-split normal windows (same reference used
everywhere else in this project), reported as (a) overall structural
shift (Frobenius norm of the difference, an at-a-glance "how much does
this fault's correlation structure move overall") and (b) the specific
physically-meaningful pairs this project's physics-residual work already
cares about.

Usage:
    python3 correlation_shift_analysis.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_robo3er, split_normal  # noqa: E402

PHYSICALLY_MEANINGFUL_PAIRS = [
    ("wheel_vels_velocity_left", "odom_odo_lintw_x", "wheel v_l vs chassis linear twist (kinematics.py's k_v relation)"),
    ("wheel_vels_velocity_right", "odom_odo_angtw_z", "wheel v_r vs chassis angular twist (kinematics.py's k_w relation)"),
    ("odom_odo_angtw_z", "imu_imu_angvel_z", "odom (wheel-derived) vs IMU (independent) yaw rate -- cross-estimator check, NOT yet implemented"),
    ("wheel_ticks_ticks_left", "wheel_vels_velocity_left", "wheel position (ticks) vs wheel velocity -- integration identity"),
    ("wheel_ticks_ticks_right", "wheel_vels_velocity_right", "wheel position (ticks) vs wheel velocity -- integration identity"),
    ("wheel_status_pwm_left", "wheel_vels_velocity_left", "commanded PWM vs actual velocity -- motor dynamics, not rigid kinematics"),
    ("wheel_status_pwm_left", "wheel_status_current_ma_left", "PWM vs current draw -- motor load relationship"),
    ("wheel_status_current_ma_left", "wheel_status_current_ma_right", "left/right motor current coupling (should track together absent an asymmetric fault)"),
]


def corr_matrix(data, idx, cols):
    flat = data[idx].reshape(-1, data.shape[-1])
    return np.corrcoef(flat.T), cols


def main():
    data, targets, cols, label_map, _ = load_robo3er(drop_dead=True)
    col_idx = {c: i for i, c in enumerate(cols)}
    fit_idx, calib_idx, test_normal_idx = split_normal(targets)

    normal_corr, _ = corr_matrix(data, fit_idx, cols)

    print("=== Physically meaningful pairs: normal vs each fault type ===\n")
    header = f"{'pair':<70}{'normal':>8}"
    for label_id_str, name in label_map.items():
        if int(label_id_str) == 0:
            continue
        header += f"{name:>16}"
    print(header)

    fault_corrs = {}
    for label_id_str, name in label_map.items():
        label_id = int(label_id_str)
        if label_id == 0:
            continue
        fault_idx = np.where(targets == label_id)[0]
        fault_corr, _ = corr_matrix(data, fault_idx, cols)
        fault_corrs[name] = fault_corr

    for a, b, desc in PHYSICALLY_MEANINGFUL_PAIRS:
        i, j = col_idx[a], col_idx[b]
        row = f"{a} vs {b:<30}"[:70].ljust(70)
        row += f"{normal_corr[i,j]:>8.3f}"
        for name in fault_corrs:
            row += f"{fault_corrs[name][i,j]:>16.3f}"
        print(row)
        print(f"    ({desc})")

    print("\n=== Overall correlation-structure shift per fault type ===")
    print("(Frobenius norm of (fault_corr - normal_corr), off-diagonal only, NaN pairs excluded"
          " -- these come from a near-constant feature within a small fault sample -- "
          "bigger = more relational change)\n")
    n = len(cols)
    mask = ~np.eye(n, dtype=bool)
    rows = []
    for name, fc in fault_corrs.items():
        d = (fc - normal_corr)[mask]
        shift = np.linalg.norm(d[~np.isnan(d)])
        n_nan = int(np.isnan(d).sum())
        rows.append((name, shift, n_nan))
    name_to_label = {v: int(k) for k, v in label_map.items()}
    rows.sort(key=lambda r: -r[1])
    for name, shift, n_nan in rows:
        n_windows = int((targets == name_to_label[name]).sum())
        print(f"{name:<16} {shift:>8.2f}   (n_windows={n_windows}, nan_pairs_excluded={n_nan})")

    print("\n=== Top-10 individual pairs with largest |corr| shift, per fault type ===")
    for name, fc in fault_corrs.items():
        diff = np.abs(fc - normal_corr)
        diff[~mask] = 0
        flat_idx = np.argsort(-diff, axis=None)[: 40]  # oversample, dedupe symmetric pairs below
        seen = set()
        print(f"\n-- {name} --")
        count = 0
        for fi in flat_idx:
            i, j = np.unravel_index(fi, diff.shape)
            if i > j:
                continue
            if (i, j) in seen:
                continue
            seen.add((i, j))
            print(f"  {cols[i]:<38} vs {cols[j]:<38}  normal={normal_corr[i,j]:>6.3f}  fault={fc[i,j]:>6.3f}  |delta|={diff[i,j]:.3f}")
            count += 1
            if count >= 10:
                break


if __name__ == "__main__":
    main()
