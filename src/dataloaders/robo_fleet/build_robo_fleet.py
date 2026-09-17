#!/usr/bin/env python3
"""
Build the windowed `robo_fleet` dataset from the raw per-robot session
folders in `data/robo_fleet/` (extracted from `data/robo_pdm/` by
`build_robo_fleet.py`'s sibling extraction step -- see that folder's
`merged.csv` files, already reduced to the same 26 kinematic_core+
actuation feature columns as `data/robo_pdm_test/metadata.json`).

Unlike `build_robo_pdm_test.py` (fault-type clients, no natural physical-
unit grouping), `robo_fleet`'s raw layout IS grouped by physical robot
(`rob_00`..`rob_03`, mirroring `data/robo_pdm/data/`), so clients here are
the 4 physical robots -- same convention as `data/robo3er/` (client =
physical unit, not fault type). Each robot contributes one `normal`
session pair and 4 fault-type subtrees (`fault_1.Caster Wheel Jam`,
`fault_2.Added Load_2.5kg`, `fault_3.Drive Wheel Cable`,
`fault_4.Thumbtack Fault`), each fault subtree containing multiple short
map_inside/map_outside session recordings (different sub-location, e.g.
left/right wheel, or session group). Every session's `merged.csv` is
windowed independently (never bridged across sessions), matching
`build_robo_pdm_test.py`'s convention; the per-window session id is saved
to `window_session_names.json`.

Source layout (raw, extracted, not windowed):
    data/robo_fleet/rob_XX/normal/<session>/merged.csv
    data/robo_fleet/rob_XX/fault_N.<name>/[<sub_location>/]map_{inside,outside}[...]/<session>/merged.csv
Each `merged.csv` has `time_s`, `timestamp_ms`, and the 26 feature columns
(raw `<group>__<field>` naming, e.g. `wheel_vels__velocity_left`) -- see
the extraction step for the full mapping. Column mapping to robo3er/
robo_pdm_test's `feature_columns` naming (`wheel_vels_velocity_left`,
single underscore) is 1:1, same `MAPPING` as `build_robo_pdm_test.py`
restricted to these 26.

Window scheme: `window_size=60`, `stride=32` -- matches
`data/robo3er/metadata.json` / `data/robo_pdm_test/metadata.json`.

Usage:
    python3 build_robo_fleet.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "robo_fleet"
OUT_DIR = RAW_DIR  # build in place, like robo3er/robo_pdm_test's data/<name>/ convention

WINDOW_SIZE = 60
STRIDE = 32
TRAIN_FRAC = 0.70
CALIB_FRAC = 0.15

ROBOT_NAMES = ["rob_00", "rob_01", "rob_02", "rob_03"]

FAULT_LABELS = {
    "normal": 0,
    "fault_1.Caster Wheel Jam": 1,
    "fault_2.Added Load_2.5kg": 2,
    "fault_3.Drive Wheel Cable": 3,
    "fault_4.Thumbtack Fault": 4,
}
CLASS_NAMES = {
    0: "normal",
    1: "caster_wheel_jam",
    2: "added_load_2.5kg",
    3: "drive_wheel_cable",
    4: "thumbtack_fault",
}

# robo3er/robo_pdm_test feature_columns name -> robo_fleet merged.csv column name
MAPPING = {
    "wheel_vels_velocity_left": "wheel_vels__velocity_left",
    "wheel_vels_velocity_right": "wheel_vels__velocity_right",
    "odom_odo_orient_x": "odom__orient_x",
    "odom_odo_orient_y": "odom__orient_y",
    "odom_odo_orient_z": "odom__orient_z",
    "odom_odo_orient_w": "odom__orient_w",
    "odom_odo_lintw_x": "odom__lin_x",
    "odom_odo_lintw_y": "odom__lin_y",
    "odom_odo_lintw_z": "odom__lin_z",
    "odom_odo_angtw_x": "odom__ang_x",
    "odom_odo_angtw_y": "odom__ang_y",
    "odom_odo_angtw_z": "odom__ang_z",
    "imu_imu_orient_x": "imu__orient_x",
    "imu_imu_orient_y": "imu__orient_y",
    "imu_imu_orient_z": "imu__orient_z",
    "imu_imu_orient_w": "imu__orient_w",
    "imu_imu_angvel_x": "imu__angvel_x",
    "imu_imu_angvel_y": "imu__angvel_y",
    "imu_imu_angvel_z": "imu__angvel_z",
    "imu_imu_acc_x": "imu__acc_x",
    "imu_imu_acc_y": "imu__acc_y",
    "imu_imu_acc_z": "imu__acc_z",
    "wheel_status_current_ma_left": "wheel_status__current_ma_left",
    "wheel_status_current_ma_right": "wheel_status__current_ma_right",
    "wheel_status_pwm_left": "wheel_status__pwm_left",
    "wheel_status_pwm_right": "wheel_status__pwm_right",
}
FEATURE_COLUMNS = list(MAPPING.keys())


def normalize_numeric_column(series: pd.Series) -> np.ndarray:
    """Matches build_robo_pdm_test.py / FL-bench's generate_robo_data.py."""
    if pd.api.types.is_bool_dtype(series):
        numeric = series.astype(np.float32)
    else:
        numeric = pd.to_numeric(series, errors="coerce")
        text_values = series.astype(str).str.strip().str.lower()
        bool_values = pd.Series(np.nan, index=series.index, dtype=np.float32)
        bool_values[text_values == "true"] = 1.0
        bool_values[text_values == "false"] = 0.0
        numeric = numeric.where(~numeric.isna(), bool_values)

    numeric = numeric.astype(np.float32)
    if numeric.isna().all():
        return np.zeros(len(series), dtype=np.float32)
    numeric = numeric.ffill().bfill().fillna(0.0)
    return numeric.to_numpy(dtype=np.float32)


def load_session(csv_path: Path):
    df = pd.read_csv(csv_path)
    missing = [MAPPING[c] for c in FEATURE_COLUMNS if MAPPING[c] not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing expected columns {missing}")
    arrs = [normalize_numeric_column(df[MAPPING[c]]) for c in FEATURE_COLUMNS]
    return np.stack(arrs, axis=1).astype(np.float32)  # [T, F]


def window_session(arr: np.ndarray, window_size: int, stride: int):
    n = len(arr)
    if n < window_size:
        return np.empty((0, window_size, arr.shape[1]), dtype=np.float32)
    starts = range(0, n - window_size + 1, stride)
    return np.stack([arr[s : s + window_size] for s in starts]).astype(np.float32)


def fault_label_for(robot_dir: Path, merged_csv: Path) -> int:
    """The fault folder is always the first path component under the robot
    dir (`normal` or `fault_N.<name>`), regardless of how many sub-location/
    map_inside/map_outside/session levels follow it."""
    rel = merged_csv.relative_to(robot_dir)
    fault_folder = rel.parts[0]
    if fault_folder not in FAULT_LABELS:
        raise ValueError(f"{merged_csv}: unrecognized fault folder {fault_folder!r}")
    return FAULT_LABELS[fault_folder]


def main():
    all_windows, all_targets, all_client, all_session = [], [], [], []
    session_meta = {}

    for client_id, robot_name in enumerate(ROBOT_NAMES):
        robot_dir = RAW_DIR / robot_name
        merged_files = sorted(
            p for p in robot_dir.rglob("merged.csv") if not p.name.startswith("._")
        )
        for csv_path in merged_files:
            session_id = f"{robot_name}/{csv_path.relative_to(robot_dir).parent.as_posix()}"
            label = fault_label_for(robot_dir, csv_path)
            arr = load_session(csv_path)
            windows = window_session(arr, WINDOW_SIZE, STRIDE)
            n = len(windows)
            if n == 0:
                print(f"skip (too short for one window): {session_id} ({len(arr)} rows)")
                continue
            targets = np.full(n, label, dtype=np.int64)
            all_windows.append(windows)
            all_targets.append(targets)
            all_client.append(np.full(n, client_id, dtype=np.int64))
            all_session.extend([session_id] * n)
            session_meta[session_id] = {"client": robot_name, "label": int(label),
                                         "n_rows": int(len(arr)), "n_windows": int(n)}
            print(f"  {session_id}: label={label} rows={len(arr)} windows={n}")

    data = np.concatenate(all_windows, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    client_of = np.concatenate(all_client, axis=0)

    print(f"\ntotal windows: {len(targets)}, feature dim: {data.shape[-1]}")
    print("class balance:", {int(k): int(v) for k, v in zip(*np.unique(targets, return_counts=True))})
    print("client sizes:", {ROBOT_NAMES[c]: int((client_of == c).sum()) for c in range(len(ROBOT_NAMES))})

    # Re-order so each client's windows are contiguous (robo3er/robo_pdm_test convention).
    order = np.argsort(client_of, kind="stable")
    data, targets, client_of = data[order], targets[order], client_of[order]
    all_session = [all_session[i] for i in order]

    # Global standardization, matching robo3er/robo_pdm_test's "standardized": true convention.
    flat = data.reshape(-1, data.shape[-1])
    feature_mean = flat.mean(axis=0)
    feature_std = flat.std(axis=0)
    safe_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    data = ((data - feature_mean) / safe_std).astype(np.float32)

    partition_indices = []
    running = 0
    for c in range(len(ROBOT_NAMES)):
        n = int((client_of == c).sum())
        idx = np.arange(running, running + n).astype(np.int64)
        running += n
        n_train = int(n * TRAIN_FRAC)
        n_calib = int(n * CALIB_FRAC)
        partition_indices.append({
            # plain python lists, not np arrays -- data/robo3er/partition.pkl's actual convention
            # (load_fl_clients concatenates train+val+test with "+", which silently does elementwise
            # addition instead of concatenation if these are numpy arrays).
            "train": idx[:n_train].tolist(),
            "val": idx[n_train : n_train + n_calib].tolist(),
            "test": idx[n_train + n_calib :].tolist(),
        })

    np.save(OUT_DIR / "data.npy", data)
    np.save(OUT_DIR / "targets.npy", targets)
    with open(OUT_DIR / "partition.pkl", "wb") as f:
        pickle.dump({
            "separation": {"train": list(range(len(ROBOT_NAMES))), "val": list(range(len(ROBOT_NAMES))),
                            "test": list(range(len(ROBOT_NAMES))), "total": len(ROBOT_NAMES)},
            "data_indices": partition_indices,
        }, f)
    with open(OUT_DIR / "window_session_names.json", "w") as f:
        json.dump(all_session, f)

    metadata = {
        "window_mode": "raw_10hz",
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "windows_per_label": {str(k): int(v) for k, v in zip(*np.unique(targets, return_counts=True))},
        "label_source": "session_folder",
        "label_map": {str(k): v for k, v in CLASS_NAMES.items()},
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "client_names": ROBOT_NAMES,
        "robot_map": {str(i): name for i, name in enumerate(ROBOT_NAMES)},
        "feature_columns": FEATURE_COLUMNS,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "standardized": True,
        "num_features": data.shape[-1],
        "num_classes": len(CLASS_NAMES),
        "num_samples": int(len(targets)),
        "num_clients": len(ROBOT_NAMES),
        "num_agents": len(ROBOT_NAMES),
        "session_meta": session_meta,
        "source_note": "Same robot/sensor family as data/robo3er/ and data/robo_pdm_test/, extracted from "
                        "data/robo_pdm/ raw session CSVs (see extraction step) down to the same 26 "
                        "kinematic_core+actuation feature columns as data/robo_pdm_test/metadata.json, "
                        "then windowed by build_robo_fleet.py. Clients are the 4 physical robots "
                        "(rob_00..rob_03), like robo3er -- NOT fault-type groups like robo_pdm_test/ALFA, "
                        "since robo_fleet's raw layout is naturally robot-grouped.",
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nsaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
