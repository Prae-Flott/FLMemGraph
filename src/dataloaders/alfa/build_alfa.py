#!/usr/bin/env python3
"""
Build the windowed ALFA (AIR Lab Failure and Anomaly) dataset from the raw
per-flight, per-topic CSVs in `data/alfa_data/` -- mirrors `build_sielaff_red.py`'s
role for Sielaff: derive features AND targets together from raw tables, then
write the common `data/<name>/{data.npy,targets.npy,metadata.json,partition.pkl}`
layout every dataloader/benchmark script in this repo expects.

Source: Keipour, Mousaei, Scherer, "ALFA: A Dataset for UAV Fault and Anomaly
Detection" (IJRR 2021, arXiv:1907.06268). 47 autonomous fixed-wing flights,
CC0, downloaded from https://kilthub.cmu.edu/articles/dataset/12707963
(`processed.zip`, `-> processed/<flight_name>/<flight_name>-<topic>.csv`).
Only the CSVs were kept locally (bag/mat dropped, not needed here).

Client definition: unlike robo3er (5 robots) or Sielaff (10 machines), ALFA
is ONE physical aircraft flown 47 times, so there is no natural per-unit
client split. Per user direction, clients here are FAULT-TYPE GROUPS instead
(pooling every flight that shares a fault, e.g. all engine-failure flights
into one client) -- 6 clients: engine, aileron, rudder, elevator,
aileron_rudder_combo (the one flight with both), and no_failure (flights with
no fault at all, contributing only extra normal reference windows).

Fault onset & type come directly from the `-failure_status-<surface>.csv`
files each flight directory carries (not just the directory name, which is
inconsistent/abbreviated) -- these files exist ONLY for the fault's active
duration (verified: their [min, max] timestamp range sits inside the
flight's overall time range and ends near the flight's last recorded
sample, i.e. the flight is cut short shortly after the fault, consistent
with the paper's emergency-landing framing). So: fault onset = min
timestamp across whichever `-failure_status-*.csv` files a flight has;
every window entirely before onset is `normal` (0), every window
overlapping/after onset takes the flight's fault-type label. Flights with
no `-failure_status-*.csv` at all are `no_failure` (all-normal).

Window scheme: most topics here are logged at only ~2.5-20Hz (much slower
than robo3er/Sielaff's finer-grained signals), and total post-fault flight
time across all 47 flights is only ~13 minutes (paper's own number) -- so
window size/stride are kept small to not waste the little post-fault signal
there is. Per-topic series are resampled to a common 5Hz grid (ffill/bfill
across gaps, matching `build_sielaff_red.py`'s bucketize+fill convention),
then windowed at 15 samples (3s) / stride 8 (1.6s).

Usage:
    python3 build_alfa.py
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "alfa_data"
OUT_DIR = REPO_ROOT / "data" / "alfa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESAMPLE = "200ms"  # 5Hz
WINDOW_LEN = 15     # 15 * 200ms = 3s
STRIDE = 8          # 8 * 200ms = 1.6s
TRAIN_FRAC = 0.75

# fault-type client id -> name; "no_failure" is a baseline-only client (all its
# windows are label 0). Determined from which `-failure_status-<surface>.csv`
# files a flight has (surface set -> client), not the free-text directory name.
CLIENT_NAMES = ["engine", "aileron", "rudder", "elevator", "aileron_rudder_combo", "no_failure"]
SURFACES_TO_CLIENT = {
    frozenset(["engines"]): "engine",
    frozenset(["aileron"]): "aileron",
    frozenset(["rudder"]): "rudder",
    frozenset(["elevator"]): "elevator",
    frozenset(["aileron", "rudder"]): "aileron_rudder_combo",
}
# target label id -> name; label 0 is shared normal, 1..5 are fault-active windows
CLASS_NAMES = {
    0: "normal",
    1: "engine_failure",
    2: "aileron_failure",
    3: "rudder_failure",
    4: "elevator_failure",
    5: "aileron_rudder_combo_failure",
}
CLIENT_TO_FAULT_LABEL = {"engine": 1, "aileron": 2, "rudder": 3, "elevator": 4,
                          "aileron_rudder_combo": 5, "no_failure": None}

# topic-file-suffix -> {output_column_name: source_field_column}
TOPIC_COLUMNS = {
    "mavros-imu-data": {
        "ang_vel_x": "field.angular_velocity.x", "ang_vel_y": "field.angular_velocity.y",
        "ang_vel_z": "field.angular_velocity.z", "lin_acc_x": "field.linear_acceleration.x",
        "lin_acc_y": "field.linear_acceleration.y", "lin_acc_z": "field.linear_acceleration.z",
    },
    "mavros-nav_info-roll": {"roll": "field.measured"},
    "mavros-nav_info-pitch": {"pitch": "field.measured"},
    "mavros-nav_info-yaw": {"yaw": "field.measured"},
    "mavros-vfr_hud": {
        "airspeed": "field.airspeed", "groundspeed": "field.groundspeed",
        "heading": "field.heading", "throttle": "field.throttle",
        "altitude": "field.altitude", "climb": "field.climb",
    },
    "mavros-battery": {"battery_voltage": "field.voltage", "battery_current": "field.current"},
    "mavros-rc-out": {f"rc_out_{i}": f"field.channels{i}" for i in range(8)},
    "mavctrl-rpy": {"ctrl_cmd_x": "field.x", "ctrl_cmd_y": "field.y", "ctrl_cmd_z": "field.z"},
    "mavctrl-path_dev": {"path_dev_x": "field.x", "path_dev_y": "field.y", "path_dev_z": "field.z"},
    "mavros-nav_info-errors": {
        "alt_error": "field.alt_error", "aspd_error": "field.aspd_error",
        "xtrack_error": "field.xtrack_error",
    },
    "mavros-local_position-velocity": {
        "vel_lin_x": "field.twist.linear.x", "vel_lin_y": "field.twist.linear.y",
        "vel_lin_z": "field.twist.linear.z", "vel_ang_x": "field.twist.angular.x",
        "vel_ang_y": "field.twist.angular.y", "vel_ang_z": "field.twist.angular.z",
    },
    "mavros-global_position-local": {
        "pos_x": "field.pose.pose.position.x", "pos_y": "field.pose.pose.position.y",
        "pos_z": "field.pose.pose.position.z",
    },
}
FAILURE_STATUS_RE = re.compile(r"-failure_status-([a-z]+)\.csv$")


def read_topic(flight_dir, flight_name, suffix):
    path = flight_dir / f"{flight_name}-{suffix}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["_t"] = pd.to_datetime(df["%time"], unit="ns")
    return df


def flight_surfaces(flight_dir, flight_name):
    surfaces, onset_ns = set(), []
    for f in flight_dir.glob(f"{flight_name}-failure_status-*.csv"):
        m = FAILURE_STATUS_RE.search(f.name)
        if not m:
            continue
        surfaces.add(m.group(1))
        t = pd.read_csv(f)["%time"]
        onset_ns.append(int(t.min()))
    return surfaces, (min(onset_ns) if onset_ns else None)


def build_flight_features(flight_dir, flight_name, full_index):
    """Return a (len(full_index), F) DataFrame of raw (unscaled) features."""
    cols = {}
    for suffix, colmap in TOPIC_COLUMNS.items():
        df = read_topic(flight_dir, flight_name, suffix)
        if df is None:
            continue
        s = df.set_index("_t")
        for out_name, src_col in colmap.items():
            if src_col not in s.columns:
                continue
            cols[out_name] = s[src_col].resample(RESAMPLE, origin=full_index[0]).mean()
    feat = pd.DataFrame(cols).reindex(full_index)
    feat = feat.ffill().bfill()
    return feat


def main():
    flight_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())
    feature_columns = None
    all_windows, all_targets, all_client, all_flight_name = [], [], [], []
    client_window_ranges = {name: [] for name in CLIENT_NAMES}
    flight_meta = {}

    for flight_dir in flight_dirs:
        flight_name = flight_dir.name
        if "no_ground_truth" in flight_name:
            print(f"skip (unlabeled): {flight_name}")
            continue

        surfaces, onset_ns = flight_surfaces(flight_dir, flight_name)
        if surfaces:
            client = SURFACES_TO_CLIENT.get(frozenset(surfaces))
            if client is None:
                print(f"skip (unrecognized surface set {surfaces}): {flight_name}")
                continue
        else:
            client = "no_failure"

        imu = read_topic(flight_dir, flight_name, "mavros-imu-data")
        if imu is None or len(imu) == 0:
            print(f"skip (no imu data): {flight_name}")
            continue
        t_min, t_max = imu["_t"].min(), imu["_t"].max()
        full_index = pd.date_range(t_min, t_max, freq=RESAMPLE)

        feat = build_flight_features(flight_dir, flight_name, full_index)
        if feature_columns is None:
            feature_columns = list(feat.columns)
        else:
            feat = feat.reindex(columns=feature_columns)
        all_nan_cols = feat.columns[feat.isna().all()].tolist()
        if all_nan_cols:
            print(f"  WARNING {flight_name}: columns entirely NaN before fillna(0.0) -- "
                  f"resample/reindex likely misaligned, values silently zeroed: {all_nan_cols}")
        feat = feat.fillna(0.0)
        arr = feat.to_numpy(dtype=np.float32)

        is_fault = np.zeros(len(full_index), dtype=bool)
        if onset_ns is not None:
            onset_t = pd.Timestamp(onset_ns, unit="ns")
            is_fault = np.asarray(full_index >= onset_t)

        n_steps = len(full_index)
        starts = list(range(0, n_steps - WINDOW_LEN + 1, STRIDE))
        fault_label = CLIENT_TO_FAULT_LABEL[client]
        flight_windows, flight_targets = [], []
        for s in starts:
            e = s + WINDOW_LEN
            label = fault_label if (fault_label is not None and is_fault[s:e].any()) else 0
            flight_windows.append(arr[s:e])
            flight_targets.append(label)

        if not flight_windows:
            print(f"skip (flight too short for one window): {flight_name}")
            continue

        flight_windows = np.stack(flight_windows)
        flight_targets = np.array(flight_targets, dtype=np.int64)
        n = len(flight_targets)
        all_windows.append(flight_windows)
        all_targets.append(flight_targets)
        all_client.append(np.full(n, CLIENT_NAMES.index(client), dtype=np.int64))
        all_flight_name.extend([flight_name] * n)
        client_window_ranges[client].append(n)
        flight_meta[flight_name] = {
            "client": client, "surfaces": sorted(surfaces),
            "n_windows": int(n), "n_fault_windows": int((flight_targets != 0).sum()),
            "duration_s": float((t_max - t_min).total_seconds()),
            "post_fault_s": float((t_max - pd.Timestamp(onset_ns, unit="ns")).total_seconds())
                             if onset_ns is not None else 0.0,
        }
        print(f"  {flight_name}: client={client} n={n} fault_windows={flight_meta[flight_name]['n_fault_windows']}")

    data = np.concatenate(all_windows, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    client_of = np.concatenate(all_client, axis=0)

    print(f"\ntotal windows: {len(targets)}, feature dim: {data.shape[-1]}")
    print("class balance:", {int(k): int(v) for k, v in zip(*np.unique(targets, return_counts=True))})
    print("client sizes:", {CLIENT_NAMES[c]: int((client_of == c).sum()) for c in range(len(CLIENT_NAMES))})

    # Re-order data/targets so each client's windows are contiguous (matches
    # robo3er/Sielaff's convention of index-contiguous-per-client partitions;
    # windows were appended in flight order above, not client order).
    order = np.argsort(client_of, kind="stable")
    data, targets, client_of = data[order], targets[order], client_of[order]
    all_flight_name = [all_flight_name[i] for i in order]

    partition_indices = []
    running = 0
    for c in range(len(CLIENT_NAMES)):
        n = int((client_of == c).sum())
        idx = np.arange(running, running + n).astype(np.int64)
        running += n
        n_train = int(n * TRAIN_FRAC)
        partition_indices.append({
            "train": idx[:n_train], "val": np.array([], dtype=np.int64), "test": idx[n_train:],
        })

    np.save(OUT_DIR / "data.npy", data)
    np.save(OUT_DIR / "targets.npy", targets)
    with open(OUT_DIR / "partition.pkl", "wb") as f:
        pickle.dump({
            "separation": {"train": list(range(len(CLIENT_NAMES))), "val": list(range(len(CLIENT_NAMES))),
                            "test": list(range(len(CLIENT_NAMES))), "total": len(CLIENT_NAMES)},
            "data_indices": partition_indices,
        }, f)
    with open(OUT_DIR / "window_flight_names.json", "w") as f:
        json.dump(all_flight_name, f)

    metadata = {
        "window_mode": "time",
        "resample": RESAMPLE,
        "window_size": WINDOW_LEN,
        "stride": STRIDE,
        "windows_per_label": {str(k): int(v) for k, v in zip(*np.unique(targets, return_counts=True))},
        "label_source": "failure_status_csv_onset",
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "client_names": CLIENT_NAMES,
        "feature_columns": feature_columns,
        "feature_mean": data.reshape(-1, data.shape[-1]).mean(axis=0).tolist(),
        "feature_std": data.reshape(-1, data.shape[-1]).std(axis=0).tolist(),
        "num_features": data.shape[-1],
        "num_classes": len(CLASS_NAMES),
        "num_samples": int(len(targets)),
        "num_clients": len(CLIENT_NAMES),
        "flight_meta": flight_meta,
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nsaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
