"""
voraus-AD adapter -- reads `data/voraus_ad/voraus-ad-dataset-100hz.parquet`
(downloaded from `https://media.vorausrobotik.com/voraus-ad-dataset-100hz.parquet`,
official host, no registration; reference paper: Brockmann, Rudolph,
Rosenhahn, Wandt, IEEE T-RO 2023, arXiv:2311.04765, `docs/voraus_ad_paper.pdf`).
See `memory/voraus-ad-dataset.md` for the full verified structure.

## Node set (18 nodes: 3 signals x 6 joints)

Per joint i in 1..6: `motor_iq_i` (motor current, drives torque),
`motor_torque_i` (motor-side torque), `torque_sensor_a_i` (independent
joint-side torque sensor -- the dataset has a redundant `_b_i` too, not
used here to keep the graph at a comparable size to Paderborn/robo3er's
prior 6-7 node graphs).

No cross-joint edges are declared -- the arm's actual kinematic coupling
(DH parameters/link geometry) isn't in the parquet and hasn't been
empirically characterized yet (see `memory/voraus-ad-dataset.md`'s "not
done yet" list). Only within-joint edges are used, replicated identically
across all 6 joints:
  - `motor_iq_i -> motor_torque_i`: "proportional" (motor current ~
    torque via the motor's torque constant Kt -- the same textbook
    relation `docs/joint_prototype_physics_gdn_anomaly_attention_prompt.md`
    Sec 9 uses as its own "proportional" example).
  - `motor_torque_i -> torque_sensor_a_i`: "nonlinear" (motor-side torque
    vs. the joint-side sensor differs through link/gearbox dynamics,
    friction and compliance -- not assumed linear, especially under a
    fault).

## Official train/test split (from the reference repo's `voraus_ad.py`)

`variant == PRE_A` (`setting == 72`, 948 samples) is a DEDICATED
pure-normal training split. Every other variant (the remaining ~1,174
samples) is test -- a mix of normal operation under other variants plus
every anomalous sample. This adapter further splits PRE_A into FIT/CALIB
(matching this project's Paderborn/robo3er convention of a held-out calib
split for z-score thresholding) and uses the 419 non-PRE_A NORMAL_OPERATION
samples as `test_normal` -- a genuinely held-out normal set, distinct in
distribution from the pure PRE_A training variant.

## Variable-length samples

Each sample (one pick-and-place cycle) has a different row count
(986-1164 at 100Hz). Samples are scaled first (StandardScaler fit on the
FIT split's pooled rows, matching Paderborn/robo3er's convention), THEN
zero-padded to the dataset-wide max length (1164) -- same order of
operations as the reference repo's own `load_pandas_dataframes(..., pad=True)`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PARQUET_PATH = REPO_ROOT / "data" / "voraus_ad" / "voraus-ad-dataset-100hz.parquet"

PRE_A_SETTING = 72
NORMAL_CATEGORY = 12
CATEGORY_NAMES = {
    0: "axis_friction", 1: "axis_weight", 2: "collision_foam", 3: "collision_cable",
    4: "collision_carton", 5: "miss_can", 6: "lose_can", 7: "can_weight",
    8: "entangled", 9: "invalid_position", 10: "motor_commutation", 11: "wobbling_station",
    12: "normal_operation",
}

NUM_JOINTS = 6
NODE_NAMES = []
for _i in range(1, NUM_JOINTS + 1):
    NODE_NAMES += [f"motor_iq_{_i}", f"motor_torque_{_i}", f"torque_sensor_a_{_i}"]
NODE_IDX = {name: i for i, name in enumerate(NODE_NAMES)}

EDGES_NAMED = []
EDGE_TYPES = []
for _i in range(1, NUM_JOINTS + 1):
    EDGES_NAMED.append((f"motor_iq_{_i}", f"motor_torque_{_i}"))
    EDGE_TYPES.append("proportional")
    EDGES_NAMED.append((f"motor_torque_{_i}", f"torque_sensor_a_{_i}"))
    EDGE_TYPES.append("nonlinear")
PRIOR_EDGES = [(NODE_IDX[s], NODE_IDX[d]) for s, d in EDGES_NAMED]

FIXED_LEN = 1164  # dataset-wide max row count per sample, verified in memory/voraus-ad-dataset.md
FIT_FRACTION = 0.80  # of the 948 PRE_A (normal-only) samples; remainder is CALIB


def _extract_samples(df, sample_ids):
    """df: pandas DataFrame with NODE_NAMES + meta columns. Returns dict
    sample_id -> [T, num_nodes] float32 array (T varies per sample, NOT
    yet padded)."""
    out = {}
    grouped = df[df["sample"].isin(sample_ids)].groupby("sample")
    for sid, g in grouped:
        out[sid] = g[NODE_NAMES].to_numpy(dtype=np.float32)
    return out


def _pad(arrays, fixed_len=FIXED_LEN):
    n = len(arrays)
    out = np.zeros((n, fixed_len, len(NODE_NAMES)), dtype=np.float32)
    for i, arr in enumerate(arrays):
        t = min(len(arr), fixed_len)
        out[i, :t] = arr[:t]
    return out


def load(seed: int = 42):
    """Returns (fit_arr, calib_arr, test_normal_arr, faults, node_names) --
    faults is {category_name: [N, FIXED_LEN, 18] array}, everything else
    is a plain [N, FIXED_LEN, 18] float32 array, scaled by a StandardScaler
    fit on the FIT split only."""
    cols = NODE_NAMES + ["sample", "anomaly", "category", "setting"]
    df = pd.read_parquet(PARQUET_PATH, columns=cols)
    meta = df.groupby("sample").first()[["category", "setting"]]

    rng = np.random.default_rng(seed)
    pre_a_ids = meta.index[meta["setting"] == PRE_A_SETTING].to_numpy()
    pre_a_ids = rng.permutation(pre_a_ids)
    n_fit = int(len(pre_a_ids) * FIT_FRACTION)
    fit_ids, calib_ids = pre_a_ids[:n_fit], pre_a_ids[n_fit:]

    test_normal_ids = meta.index[(meta["setting"] != PRE_A_SETTING) & (meta["category"] == NORMAL_CATEGORY)].to_numpy()

    fault_ids = {}
    for cat_id, name in CATEGORY_NAMES.items():
        if cat_id == NORMAL_CATEGORY:
            continue
        ids = meta.index[meta["category"] == cat_id].to_numpy()
        if len(ids) > 0:
            fault_ids[name] = ids

    fit_raw = _extract_samples(df, fit_ids)
    calib_raw = _extract_samples(df, calib_ids)
    test_normal_raw = _extract_samples(df, test_normal_ids)
    fault_raw = {name: _extract_samples(df, ids) for name, ids in fault_ids.items()}

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(np.concatenate(list(fit_raw.values()), axis=0))

    def scale_all(raw_dict):
        return [scaler.transform(arr).astype(np.float32) for arr in raw_dict.values()]

    fit_arr = _pad(scale_all(fit_raw))
    calib_arr = _pad(scale_all(calib_raw))
    test_normal_arr = _pad(scale_all(test_normal_raw))
    faults = {name: _pad(scale_all(raw)) for name, raw in fault_raw.items()}

    return fit_arr, calib_arr, test_normal_arr, faults, NODE_NAMES


if __name__ == "__main__":
    fit_arr, calib_arr, test_normal_arr, faults, node_names = load()
    print(f"fit={fit_arr.shape} calib={calib_arr.shape} test_normal={test_normal_arr.shape}")
    for name, arr in faults.items():
        print(f"  fault[{name}]={arr.shape}")
