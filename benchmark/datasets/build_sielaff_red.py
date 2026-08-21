#!/usr/bin/env python3
"""
Build a RED-severity-only windowed dataset for Sielaff, from the raw log
tables in `data/sielaff_data/data_sielaff.zip`.

Why a separate build (not a relabel of the existing `data/sielaff/` arrays):
the existing dataset's targets are `dominant_error_group_in_window` over 6
coarse error-type GROUPS (bottle_recognition, mechanical_jam, ...) -- see
`data/sielaff/metadata.json`. Severity (red/orange/green) is a per-error-ID
attribute (`data/sielaff_data/sielaff_ground_truth.md`) that cuts ACROSS
those groups, so it cannot be recovered from the group-labeled arrays alone;
the raw per-event error IDs are required. This script re-derives features
AND labels together from the raw tables so they stay aligned.

Fault definition (degradation window): Sielaff logs are discrete events, not
a continuously drifting sensor signal, so "degradation" is operationalized
the same way the existing V2/V3 pipelines treat it -- a fixed-length window
of aggregated 30-min buckets that CONTAINS the anomalous event(s), scored
against windows containing no logged error at all. This matches the
"Sliding Window Level (4h windows)" scheme already benchmarked in
sielaff_ground_truth.md (severity_4, N=18757) -- we use the same 8 buckets x
30min = 4h window, stride 2 buckets (1h).

Label scheme (3-class, red is the sole target of interest):
  0 = no_error       -- zero logged errors of ANY severity in the window
  1 = red            -- >=1 RED-severity error in the window (all 47 IDs from
                        sielaff_ground_truth.md: 31 A_red_classifiable +
                        16 D_red_unclassifiable)
  2 = non_red_error   -- >=1 orange/green error but NO red error (kept out of
                        the "normal" reference set so it doesn't contaminate
                        calibration; reported as a contrast class, not scored
                        as the target)

Usage:
    python3 build_sielaff_red.py
"""
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ZIP_PATH = REPO_ROOT / "data" / "sielaff_data" / "data_sielaff.zip"
OUT_DIR = REPO_ROOT / "data" / "sielaff_red"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKET = "30min"
WINDOW_LEN = 8   # 8 * 30min = 4h, matches severity_4 sliding-window scheme
STRIDE = 2       # 2 * 30min = 1h, matches windows_per_label stride in data/sielaff/metadata.json
TRAIN_FRAC = 0.75
TEST_FRAC = 0.25

# From data/sielaff_data/sielaff_ground_truth.md (id -> name), red severity only
RED_ID_NAMES = {
    203: "status_bottle_out_of_range", 236: "status_slow_label_movement",
    403: "status_bottle_collision", 237: "status_backward_label_movement",
    461: "compactor1_door_open", 471: "compactor1_door_close",
    420: "status_safety_circuit_off", 308: "status_crate_bottle_wrong",
    202: "status_bottle_direction_wrong", 417: "status_dooropen",
    618: "crate_tailback_ok", 462: "compactor2_door_open", 472: "compactor2_door_close",
    235: "status_bottle_compacted", 801: "status_printer_ok", 603: "crate_background_ok",
    901: "status_misc_sw_restart", 620: "crate_up", 463: "compactor3_door_open",
    473: "compactor3_door_close", 250: "action_cleaning_start", 481: "compactor1_flap_open",
    252: "action_cleaning_daily", 464: "compactor4_door_open", 465: "compactor5_door_open",
    491: "compactor1_flap_close", 221: "status_last_ls_passed", 466: "compactor6_door_open",
    474: "compactor4_door_close", 475: "compactor5_door_close", 476: "compactor6_door_close",
    303: "status_crate_not_in_assortment", 482: "compactor2_flap_open",
    492: "compactor2_flap_close", 806: "status_receipt_retracted",
    483: "compactor3_flap_open", 207: "status_bottle_fraud", 493: "compactor3_flap_close",
    309: "status_crate_diagonally", 900: "status_misc_pc_reboot", 307: "status_crate_fraud",
    255: "action_cleaning_week", 486: "compactor6_flap_open", 485: "compactor5_flap_open",
    484: "compactor4_flap_open", 467: "compactor7_door_open", 477: "compactor7_door_close",
}
RED_IDS = set(RED_ID_NAMES)
assert len(RED_IDS) == 47, f"expected 47 red IDs, got {len(RED_IDS)}"

MACHINE_IDS = ["90384920", "90424719", "90433175", "90470204", "90470289",
               "90470674", "90481323", "90511222", "90512472", "90518171"]


def read_csv(zf, name):
    df = pd.read_csv(zf.open(name))
    for col in ("timestamp", "timestamptz"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def bucketize_mean(df, value_col, ts_col="timestamptz"):
    return df.set_index(ts_col)[value_col].resample(BUCKET).mean()


def bucketize_sum(df, value_col, ts_col="timestamptz"):
    return df.set_index(ts_col)[value_col].resample(BUCKET).sum()


def bucketize_count(df, ts_col="timestamptz"):
    return df.set_index(ts_col).resample(BUCKET).size()


def bucketize_nunique(df, value_col, ts_col="timestamptz"):
    return df.set_index(ts_col)[value_col].resample(BUCKET).nunique()


def build_machine_features(tables, serial, full_index):
    """Return a (len(full_index), 39) DataFrame of raw (unscaled) features for one machine."""
    sv = tables["statistic_values"]
    sv_m = sv[sv["serial_no"] == serial]
    jr = tables["journal"][tables["journal"]["serial_no"] == serial]
    rc = tables["receipts"][tables["receipts"]["serial_no"] == serial]
    rj = tables["reject"][tables["reject"]["serial_no"] == serial]
    cl = tables["cleaning"][tables["cleaning"]["serial_no"] == serial]

    cols = {}
    cols["Helligkeit_Flaschenerkennung"] = bucketize_mean(
        sv_m[(sv_m["action"] == "Helligkeit") & (sv_m["name"] == "Flaschenerkennung")], "value")
    cols["journal_count"] = bucketize_count(jr)
    cols["total_quantity"] = bucketize_sum(jr, "quantity")
    cols["total_deposit"] = bucketize_sum(jr, "deposit")
    cols["total_weight"] = bucketize_sum(jr, "weight")
    cols["unique_categories"] = bucketize_nunique(jr, "category")
    cols["reject_count"] = bucketize_count(rj)
    cols["has_reject"] = (bucketize_count(rj) > 0).astype(float)
    cols["cleaning_count"] = bucketize_count(cl)
    cols["cleaning_duration"] = bucketize_sum(cl, "duration")
    cols["cleaning_begin_pixels"] = bucketize_mean(cl, "begin_pixels")
    cols["cleaning_end_pixels"] = bucketize_mean(cl, "end_pixels")
    cols["receipt_count"] = bucketize_count(rc)

    for n in range(1, 8):
        cols[f"schließen_Weiche {n}"] = bucketize_mean(
            sv_m[(sv_m["action"] == "schließen") & (sv_m["name"] == f"Weiche {n}")], "value")
        cols[f"öffnen_Weiche {n}"] = bucketize_mean(
            sv_m[(sv_m["action"] == "öffnen") & (sv_m["name"] == f"Weiche {n}")], "value")
    for n in range(1, 7):
        cols[f"read SM_RingCamera {n}"] = bucketize_mean(
            sv_m[(sv_m["action"] == "read SM") & (sv_m["name"] == f"RingCamera {n}")], "value")
        cols[f"read barcode_RingCamera {n}"] = bucketize_mean(
            sv_m[(sv_m["action"] == "read barcode") & (sv_m["name"] == f"RingCamera {n}")], "value")

    feat = pd.DataFrame(cols).reindex(full_index)
    count_cols = ["journal_count", "total_quantity", "total_deposit", "total_weight",
                  "unique_categories", "reject_count", "has_reject", "cleaning_count",
                  "cleaning_duration", "cleaning_begin_pixels", "cleaning_end_pixels",
                  "receipt_count"]
    for c in count_cols:
        feat[c] = feat[c].fillna(0.0)
    value_cols = [c for c in feat.columns if c not in count_cols]
    feat[value_cols] = feat[value_cols].ffill().bfill()
    feat[value_cols] = feat[value_cols].fillna(feat[value_cols].mean()).fillna(0.0)
    return feat


def main():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        tables = {name: read_csv(zf, f"{name}.csv") for name in
                  ["statistic_values", "journal", "receipts", "reject", "cleaning", "error_logs"]}

    err = tables["error_logs"]
    print(f"error_logs total rows: {len(err)}")
    red_err = err[err["error_id"].isin(RED_IDS)]
    print(f"red-severity events: {len(red_err)} (ground_truth.md reports 5,593 across 47 IDs)")

    t_min = min(df["timestamptz"].min() for df in tables.values() if "timestamptz" in df.columns)
    t_max = max(df["timestamptz"].max() for df in tables.values() if "timestamptz" in df.columns)
    full_index = pd.date_range(t_min.floor(BUCKET), t_max.ceil(BUCKET), freq=BUCKET, inclusive="left")
    print(f"time range {t_min} .. {t_max}, {len(full_index)} buckets @ {BUCKET}")

    feature_columns = None
    all_windows, all_targets, all_machine, all_red_ids = [], [], [], []
    partition_indices = []
    running = 0

    for m_i, serial in enumerate(MACHINE_IDS):
        serial_int = int(serial)
        feat = build_machine_features(tables, serial_int, full_index)
        if feature_columns is None:
            feature_columns = list(feat.columns)
        arr = feat[feature_columns].to_numpy(dtype=np.float32)

        err_m = err[err["serial_no"] == serial_int]
        red_err_m = err_m[err_m["error_id"].isin(RED_IDS)].copy()
        red_err_m["bucket"] = red_err_m["timestamptz"].dt.floor(BUCKET)
        bucket_to_red_ids = red_err_m.groupby("bucket")["error_id"].apply(list).to_dict()
        bucket_has_red = red_err_m.set_index("timestamptz").resample(BUCKET).size() \
            .reindex(full_index, fill_value=0) > 0
        bucket_has_any = err_m.set_index("timestamptz").resample(BUCKET).size() \
            .reindex(full_index, fill_value=0) > 0
        has_red = bucket_has_red.to_numpy()
        has_any = bucket_has_any.to_numpy()

        n_buckets = len(full_index)
        starts = list(range(0, n_buckets - WINDOW_LEN + 1, STRIDE))
        client_windows, client_targets, client_red_ids = [], [], []
        for s in starts:
            e = s + WINDOW_LEN
            if has_red[s:e].any():
                label = 1
                window_red_ids = sorted({rid for b in full_index[s:e]
                                          for rid in bucket_to_red_ids.get(b, [])})
            elif has_any[s:e].any():
                label = 2
                window_red_ids = []
            else:
                label = 0
                window_red_ids = []
            client_windows.append(arr[s:e])
            client_targets.append(label)
            client_red_ids.append(window_red_ids)

        client_windows = np.stack(client_windows)
        client_targets = np.array(client_targets, dtype=np.int64)
        n = len(client_targets)
        n_train = int(n * TRAIN_FRAC)
        idx = np.arange(running, running + n)
        partition_indices.append({
            "train": idx[:n_train].astype(np.int64),
            "val": np.array([], dtype=np.int64),
            "test": idx[n_train:].astype(np.int64),
        })
        running += n

        all_windows.append(client_windows)
        all_targets.append(client_targets)
        all_machine.append(np.full(n, m_i, dtype=np.int64))
        all_red_ids.extend(client_red_ids)
        counts = {k: int((client_targets == k).sum()) for k in (0, 1, 2)}
        print(f"  machine {serial}: {n} windows, no_error={counts[0]} red={counts[1]} non_red={counts[2]}")

    data = np.concatenate(all_windows, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    machine_of = np.concatenate(all_machine, axis=0)
    assert len(all_red_ids) == len(targets)

    print(f"\ntotal windows: {len(targets)}, feature dim: {data.shape[-1]}")
    print("class balance:", {int(k): int(v) for k, v in zip(*np.unique(targets, return_counts=True))})

    np.save(OUT_DIR / "data.npy", data)
    np.save(OUT_DIR / "targets.npy", targets)
    with open(OUT_DIR / "partition.pkl", "wb") as f:
        pickle.dump({"separation": {"train": list(range(10)), "val": list(range(10)),
                                     "test": list(range(10)), "total": 10},
                     "data_indices": partition_indices}, f)
    with open(OUT_DIR / "window_red_ids.json", "w") as f:
        json.dump(all_red_ids, f)

    metadata = {
        "window_mode": "time",
        "bucket": BUCKET,
        "window_size": WINDOW_LEN,
        "stride": STRIDE,
        "windows_per_label": {str(k): int(v) for k, v in zip(*np.unique(targets, return_counts=True))},
        "label_source": "red_severity_v_ground_truth_md",
        "class_names": {"0": "no_error", "1": "red", "2": "non_red_error"},
        "red_ids": sorted(RED_IDS),
        "red_id_names": {str(k): v for k, v in RED_ID_NAMES.items()},
        "feature_columns": feature_columns,
        "machine_ids": MACHINE_IDS,
        "feature_mean": data.reshape(-1, data.shape[-1]).mean(axis=0).tolist(),
        "feature_std": data.reshape(-1, data.shape[-1]).std(axis=0).tolist(),
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nsaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
