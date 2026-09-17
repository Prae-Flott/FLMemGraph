"""
Shared dataloader for the robo3er/robo_fleet family of datasets (same 71
kinematic_core+actuation feature columns, same windowing convention): one
place for loading a client's `data.npy`/`targets.npy`, dropping
genuinely-zero-information columns, and producing the fit/calib/test_normal
split every script in this folder uses, instead of each training script
re-implementing (and risking silently diverging from) the same logic.

Originally written for robo3er (hence `load_robo3er`'s name and the
column-purification analysis below, done on robo3er's own data); robo3er
itself has since been removed from this project, but this module now
lives under `src/dataloaders/robo_fleet/` because `run_robo_fleet_bck_federated.py`
is its only remaining caller -- always pass `data_dir` explicitly (see
`load_robo3er`'s docstring), there is no longer a default dataset here.

## Feature purification decision

Only 3 of robo3er's 71 raw columns are dropped:
`tf_footprint_base_footprint_trans_z`, `..._rot_x`, `..._rot_y` -- verified
exactly zero-variance (1 unique value) across ALL 6043 windows, both
normal AND every one of the 4 fault types, on the current on-disk data
(re-checked here, not assumed from stale memory notes -- the dataset was
regenerated 2026-07-23 and old per-feature stats aren't guaranteed to
still hold). These carry zero information under any circumstance recorded
so far, so dropping them has no possible downside.

Everything else in the 71 is KEPT, on purpose, for two different reasons
depending on the feature:

1. **Statistically redundant continuous features are still kept.**
   Re-running the correlation audit on the current data found the same
   redundant clusters `memory/robo3er-anomaly-detection-approaches.md`
   already documented on the old data: cliff sensors + `ir_opcode_opcode`
   collapse to ~1 signal, pose is triplicated across
   `odom_odo_pos_*`/`tf_link_base_link_*`/`tf_footprint_base_footprint_*`,
   battery voltage/temperature/current move together. A prior experiment
   (`auto_encoder/train_reduced_feature_ae.py`, dropping 19 such features,
   71->52) already tested removing these and found 3 of 4 fault types
   unaffected or slightly better, but `stuck` -- the hardest fault type --
   got WORSE (0.796 -> 0.717 AUROC), because even a "redundant" duplicate
   feature can serve as conditioning context that sharpens another
   feature's own reconstruction/prediction baseline (law of total
   variance: less conditioning context can only inflate a feature's
   baseline noise, which directly shrinks its normalized anomaly score for
   the same absolute fault-induced deviation). GDN's graph-attention
   architecture is if anything MORE exposed to this than the flat
   autoencoder that finding came from: a neighbor's raw feature values are
   used directly (via the learned attention graph) to help predict another
   node's value, so removing a node also removes it from every other
   node's neighbor candidate pool, not just from its own encoder input.
   **Conclusion: do not drop statistically-redundant features here either.**

2. **Near-constant discrete status flags are also kept, but handled
   differently -- at the scoring layer, not by removal.** Features like
   `slip_status_is_slipping`, `dock_status_is_docked`,
   `stop_status_is_stopped`, `kidnap_status_is_kidnapped`,
   `ir_opcode_sensor`, `dock_status_dock_visible` are binary/near-constant
   (2 unique values), which is exactly what makes GDN's robust z-score
   `(err - median) / IQR` blow up on them (documented in
   `memory/robo3er-anomaly-detection-approaches.md`: raw z-scores in the
   thousands, e.g. `dock_status_dock_visible`=4032, because a near-zero
   calibration IQR makes any tiny absolute error explode). But these are
   simultaneously robo3er's clearest fault SIGNATURES
   (`memory/deployment-architecture.md`: `stop_status_is_stopped`->Broken
   Pipe, `slip_status_is_slipping`->cable trapped, etc.) -- dropping them
   would remove real detection signal for 3 of 4 fault types. The fix
   applied here is `gdn_score`'s IQR floor (see below), not exclusion.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15

# Confirmed exactly zero-variance across every window, normal and fault
# alike, on the current on-disk data.npy -- see module docstring.
DEAD_COLUMNS = [
    "tf_footprint_base_footprint_trans_z",
    "tf_footprint_base_footprint_rot_x",
    "tf_footprint_base_footprint_rot_y",
]


def load_robo3er(data_dir, drop_dead=True):
    """Returns (data [N,60,F], targets [N], feature_columns, label_map,
    robot_map). F is 68 if drop_dead else 71.

    `data_dir` is required (e.g. `data/robo_fleet/`, which is built with
    the same 71 `feature_columns` names/order and the same DEAD_COLUMNS
    convention as robo3er's original data) -- this function was originally
    robo3er-only with a hardcoded default; robo3er has since been removed
    from this project, so there is no longer a valid default to fall back
    to. Every caller must pass `data_dir` explicitly."""
    data_dir = Path(data_dir)
    data = np.load(data_dir / "data.npy")
    targets = np.load(data_dir / "targets.npy")
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    cols = meta["feature_columns"]
    label_map = meta["label_map"]
    robot_map = meta.get("robot_map", {})

    if drop_dead:
        keep_idx = [i for i, c in enumerate(cols) if c not in DEAD_COLUMNS]
        data = data[:, :, keep_idx]
        cols = [cols[i] for i in keep_idx]

    return data, targets, cols, label_map, robot_map


def split_normal(targets, fit_fraction=FIT_FRACTION, calib_fraction=CALIB_FRACTION):
    """Chronological split of normal (label==0) window indices into
    fit / calib / test_normal, matching every other script in this
    project's convention (sorted by index, not shuffled)."""
    normal_idx = np.sort(np.where(targets == 0)[0])
    n = len(normal_idx)
    n_fit = int(n * fit_fraction)
    n_calib = int(n * calib_fraction)
    fit_idx = normal_idx[:n_fit]
    calib_idx = normal_idx[n_fit : n_fit + n_calib]
    test_normal_idx = normal_idx[n_fit + n_calib :]
    return fit_idx, calib_idx, test_normal_idx


def fit_scaler(data, fit_idx):
    scaler = StandardScaler()
    scaler.fit(data[fit_idx].reshape(-1, data.shape[-1]))
    return scaler


def scale(windows, scaler):
    n, t, f = windows.shape
    return scaler.transform(windows.reshape(-1, f)).reshape(n, t, f).astype(np.float32)


def gdn_score(per_window_err, median, iqr, iqr_floor_frac=None):
    """GDN-style anomaly score: per-feature robust z-score, then max over
    features. Default (`iqr_floor_frac=None`) matches
    `test_gdn/train_gdn.py` exactly -- floor at 1e-8, i.e. essentially no
    floor -- so the feature-pruning ablation in this file stays a clean
    single-variable comparison against the existing baseline report.

    Passing `iqr_floor_frac` (e.g. 0.1) switches to a MEDIAN-IQR-relative
    floor intended to fix the documented z-in-the-thousands blowup on
    near-constant status flags (module docstring, point 2) for
    *attribution* stability. Measured trade-off on the current data: it
    reduced cable-trapped AUROC by 0.041 and Low-battery by 0.028 (both
    fault types whose true signature IS one of those near-constant flags,
    e.g. `slip_status_is_slipping` for cable trapped -- so the "blown up"
    z-score, despite being statistically unstable, happened to already be
    pointing at the right feature for THOSE fault types specifically), for
    a modest stuck gain. This is a real detection-AUROC-vs-attribution-
    stability trade-off, not a strict improvement -- do not enable it by
    default without deciding which side of that trade-off the deployment
    use case needs."""
    floor = np.maximum(iqr, 1e-8)
    if iqr_floor_frac is not None:
        informative = iqr[iqr > 1e-6]
        ref_scale = float(np.median(informative)) if len(informative) else 1.0
        floor = np.maximum(floor, iqr_floor_frac * ref_scale)
    return ((per_window_err - median) / floor).max(axis=1)
