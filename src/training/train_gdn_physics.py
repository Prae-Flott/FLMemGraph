#!/usr/bin/env python3
"""
Physics-residual GDN: same architecture, training loop, dense-sliding
protocol, and GDN-style scoring as `test_gdn/train_gdn.py`, with exactly
ONE change -- `odom_odo_lintw_x` and `odom_odo_angtw_z` are replaced by
their differential-drive KINEMATIC RESIDUAL (see `kinematics.py`) before
anything else happens. Every other of robo3er's 71 features is untouched,
raw. Everything downstream (scaler, GDN model, dense pairs, training,
fit/calib/test-normal split, per-fault AUROC) is byte-identical to
`test_gdn/train_gdn.py` so the two runs are a clean, single-variable
ablation -- this file does not re-derive any of that protocol, it composes
it with one physics-informed preprocessing step.

Two things this buys, beyond whatever AUROC delta shows up:

1. Interpretability: the kinematic parameters (k_v ~ wheel_radius/2,
   k_w ~ wheel_radius/track_width) are physical constants a domain expert
   can sanity-check against the robot's actual spec sheet -- printed and
   saved in `physics_report.json` alongside the R^2 each equation explains
   on held-out normal data (i.e. "how much of this signal is rigid-body
   kinematics vs. something GDN still has to learn").
2. A slip-relevant residual: a wheel spinning without the chassis actually
   moving accordingly is the textbook definition of wheel slip, so this
   residual is expected to specifically help `cable trapped` (robo3er's
   slip-labeled fault) beyond whatever the raw-feature GDN already found.

Usage:
    python3 train_gdn_physics.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "robo3er"))
from gdn_model import GDN  # noqa: E402
from kinematics import (  # noqa: E402
    fit_kinematic_params,
    apply_kinematic_residual,
    residual_feature_names,
)
from dataset import (  # noqa: E402
    load_robo3er,
    split_normal,
    fit_scaler,
    scale,
    gdn_score,
    DEAD_COLUMNS,
)
import feature_groups  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "robo3er"
OUT_DIR = REPO_ROOT / "checkpoints" / "robo3er"
BASELINE_CKPT = OUT_DIR / "gdn_baseline_robo3er.pth"

# Minimal-validation-first: start with ONLY the kinematics-equation
# features (displacement/velocity/acceleration, see feature_groups.py),
# add "actuation", "status_flags", "environment" one at a time once this
# tier is validated. None = all 68 (no group filtering) -- the setting
# used for every earlier run in this folder.
ACTIVE_FEATURE_GROUPS = ["kinematic_core"]

BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
SEED = 42
EMBED_DIM = 64
TOP_K = 15
HISTORY_LEN = 30  # matches test_gdn/train_gdn.py so the baseline checkpoint is comparable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FOCUS_SENSORS = [
    "wheel_status_current_ma_left",
    "wheel_status_current_ma_right",
    "stop_status_is_stopped",
    "slip_status_is_slipping",
    "dock_status_is_docked",
    "odom_odo_lintw_x__kin_residual",
    "odom_odo_angtw_z__kin_residual",
]


def load_and_split():
    data, targets, cols, label_map, _ = load_robo3er(drop_dead=True)
    if ACTIVE_FEATURE_GROUPS is not None:
        data, cols = feature_groups.select_columns(data, cols, ACTIVE_FEATURE_GROUPS)
    fit_idx, calib_idx, test_normal_idx = split_normal(targets)
    return data, targets, cols, label_map, fit_idx, calib_idx, test_normal_idx


def make_dense_pairs(scaled_windows, history_len=HISTORY_LEN):
    n, t, f = scaled_windows.shape
    num_targets = t - history_len
    history = np.empty((n, num_targets, history_len, f), dtype=np.float32)
    target = np.empty((n, num_targets, f), dtype=np.float32)
    for i in range(num_targets):
        history[:, i] = scaled_windows[:, i : i + history_len]
        target[:, i] = scaled_windows[:, i + history_len]
    window_id = np.repeat(np.arange(n), num_targets)
    return (
        history.reshape(n * num_targets, history_len, f),
        target.reshape(n * num_targets, f),
        window_id,
    )


def make_loader(history, target, batch_size, shuffle):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(history), torch.from_numpy(target))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def per_sample_error(model, history, target, batch_size=BATCH_SIZE):
    model.eval()
    errors = []
    for i in range(0, len(history), batch_size):
        h = torch.from_numpy(history[i : i + batch_size]).to(DEVICE)
        t = torch.from_numpy(target[i : i + batch_size]).to(DEVICE)
        pred = model(h)
        err = (pred - t) ** 2
        errors.append(err.cpu().numpy())
    return np.concatenate(errors, axis=0)


def aggregate_by_window(per_sample_err, window_id, num_windows):
    out = np.zeros((num_windows, per_sample_err.shape[1]), dtype=np.float64)
    counts = np.zeros(num_windows, dtype=np.int64)
    np.add.at(out, window_id, per_sample_err)
    np.add.at(counts, window_id, 1)
    return (out / counts[:, None]).astype(np.float32)


def per_window_error(model, scaled_windows, batch_size=BATCH_SIZE):
    history, target, window_id = make_dense_pairs(scaled_windows)
    err = per_sample_error(model, history, target, batch_size)
    return aggregate_by_window(err, window_id, len(scaled_windows))


def train(model, fit_history, fit_target, calib_scaled):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    loader = make_loader(fit_history, fit_target, BATCH_SIZE, shuffle=True)

    best_calib_mse = float("inf")
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        fit_loss = 0.0
        for h, t in loader:
            h, t = h.to(DEVICE), t.to(DEVICE)
            optimizer.zero_grad()
            pred = model(h)
            loss = criterion(pred, t)
            loss.backward()
            optimizer.step()
            fit_loss += loss.item() * h.size(0)
        fit_loss /= len(fit_history)

        calib_mse = float(per_window_error(model, calib_scaled).mean())
        print(f"epoch {epoch:3d}  fit_mse={fit_loss:.6f}  calib_mse={calib_mse:.6f}")
        if calib_mse < best_calib_mse:
            best_calib_mse = calib_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_calib_mse


def compare_graph_to_baseline(model, cols_renamed):
    """If a baseline (pure-data, no physics residual, full 71 raw features
    including the 3 dead columns this project drops) checkpoint exists,
    load it into a fresh GDN and print its learned neighbors for the same
    focus sensors side by side -- shows whether stripping out the
    kinematically-explained part of the odom signal changes what GDN
    considers that node's relevant neighbors. Skipped entirely when
    ACTIVE_FEATURE_GROUPS restricts the node set below all 68 -- the
    baseline's node universe (all 68) is no longer comparable one-to-one."""
    if ACTIVE_FEATURE_GROUPS is not None:
        print(
            f"\n(ACTIVE_FEATURE_GROUPS={ACTIVE_FEATURE_GROUPS} restricts the node set -- "
            "skipping baseline graph comparison, node universes aren't comparable)"
        )
        return
    if not BASELINE_CKPT.exists():
        print(f"\n(no baseline checkpoint at {BASELINE_CKPT}, skipping graph comparison)")
        return
    with open(DATA_DIR / "metadata.json") as f:
        raw_cols = json.load(f)["feature_columns"]  # full 71, baseline's own node set

    ckpt = torch.load(BASELINE_CKPT, map_location=DEVICE, weights_only=False)
    baseline = GDN(num_nodes=len(raw_cols), window_size=HISTORY_LEN, embed_dim=EMBED_DIM, top_k=TOP_K).to(DEVICE)
    baseline.load_state_dict(ckpt["model_state_dict"])

    baseline_focus = [n for n in FOCUS_SENSORS if not n.endswith("__kin_residual")]
    baseline_focus += ["odom_odo_lintw_x", "odom_odo_angtw_z"]  # raw names in the baseline model

    print("\nbaseline (pure data-driven) top-5 neighbors:")
    print(baseline.graph_summary(raw_cols, baseline_focus, k=5))
    print("\nphysics-residual model top-5 neighbors:")
    focus = [n for n in FOCUS_SENSORS if n in cols_renamed]
    print(model.graph_summary(cols_renamed, focus, k=5))


def main():
    data, targets, cols, label_map, fit_idx, calib_idx, test_normal_idx = load_and_split()
    num_nodes = data.shape[2]

    kin_params = fit_kinematic_params(data, fit_idx, cols)
    print("fitted kinematic params (differential-drive prior, fit split only):")
    print(
        f"  linear:  v_lin = {kin_params['k_v']:.6f} * (v_l+v_r) + {kin_params['b_v']:.6f}"
        f"   R^2={kin_params['r2_lin']:.4f}"
    )
    print(
        f"  angular: v_ang = {kin_params['k_w']:.6f} * (v_r-v_l) + {kin_params['b_w']:.6f}"
        f"   R^2={kin_params['r2_ang']:.4f}"
    )
    print(
        "  (k_v ~ wheel_radius/2, k_w ~ wheel_radius/track_width -- compare "
        "against this robot's spec sheet as a bidirectional sanity check)"
    )

    data = apply_kinematic_residual(data, cols, kin_params)
    cols_renamed = residual_feature_names(cols)

    print(
        f"\nnormal windows -> fit={len(fit_idx)} calib={len(calib_idx)} "
        f"test={len(test_normal_idx)}   nodes(sensors)={num_nodes}  "
        f"history_len={HISTORY_LEN}  dense_targets_per_window={data.shape[1]-HISTORY_LEN}"
    )

    scaler = fit_scaler(data, fit_idx)
    fit_scaled = scale(data[fit_idx], scaler)
    calib_scaled = scale(data[calib_idx], scaler)
    fit_history, fit_target, _ = make_dense_pairs(fit_scaled)
    print(f"dense (history, target) pairs for training: {len(fit_history)}")

    model = GDN(num_nodes=num_nodes, window_size=HISTORY_LEN, embed_dim=EMBED_DIM, top_k=TOP_K).to(DEVICE)
    print(f"\ntraining physics-residual GDN (embed_dim={EMBED_DIM}, top_k={TOP_K})...")
    model, best_calib_mse = train(model, fit_history, fit_target, calib_scaled)
    print(f"\nbest calib MSE: {best_calib_mse:.6f}")

    compare_graph_to_baseline(model, cols_renamed)

    calib_err = per_window_error(model, calib_scaled)
    median = np.median(calib_err, axis=0)
    q75, q25 = np.percentile(calib_err, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)

    test_normal_scaled = scale(data[test_normal_idx], scaler)
    normal_err = per_window_error(model, test_normal_scaled)
    normal_score = gdn_score(normal_err, median, iqr)
    threshold = float(np.percentile(normal_score, 95))
    fpr = float((normal_score > threshold).mean())

    report = {
        "config": {"embed_dim": EMBED_DIM, "top_k": TOP_K, "epochs": EPOCHS,
                    "history_len": HISTORY_LEN, "dense": True, "physics_residual": True,
                    "num_nodes": num_nodes, "dropped_dead_columns": DEAD_COLUMNS},
        "kinematic_params": kin_params,
        "threshold_p95": threshold,
        "normal": {"n": int(len(test_normal_idx)), "false_positive_rate": fpr},
        "fault_types": {},
    }

    guard_band_path = DATA_DIR / "stuck_severity_split.json"
    guard_band_excluded = set()
    if guard_band_path.exists():
        with open(guard_band_path) as f:
            guard_band_excluded = set(json.load(f)["transition_excluded_idx"])

    rows = []
    full_z_by_fault = {}
    for label_id_str, name in label_map.items():
        label_id = int(label_id_str)
        if label_id == 0:
            continue
        fault_idx = np.where(targets == label_id)[0]
        n_excluded = 0
        if guard_band_excluded:
            keep_mask = np.array([idx not in guard_band_excluded for idx in fault_idx])
            n_excluded = int((~keep_mask).sum())
            fault_idx = fault_idx[keep_mask]
        if len(fault_idx) == 0:
            continue
        fault_scaled = scale(data[fault_idx], scaler)
        fault_err = per_window_error(model, fault_scaled)
        fault_score = gdn_score(fault_err, median, iqr)

        labels = np.concatenate([np.zeros(len(normal_score)), np.ones(len(fault_score))])
        auroc = float(
            sk_metrics.roc_auc_score(labels, np.concatenate([normal_score, fault_score]))
        )
        recall = float((fault_score > threshold).mean())

        z = (fault_err.mean(axis=0) - median) / iqr
        top_sensors = [cols_renamed[i] for i in np.argsort(-z)[:3]]
        full_z_by_fault[name] = z

        report["fault_types"][name] = {
            "label_id": label_id,
            "n": int(len(fault_idx)),
            "n_guard_band_excluded": n_excluded,
            "auroc": auroc,
            "recall_at_p95": recall,
            "top_sensors_by_z": top_sensors,
        }
        rows.append((name, len(fault_idx), n_excluded, auroc, recall, top_sensors))

    rows.sort(key=lambda r: r[3], reverse=True)
    print(f"\nheld-out normal: n={len(test_normal_idx)}  FPR@p95={fpr:.3f}")
    print(f"{'fault type':<16}{'n':>6}{'excl':>6}{'AUROC':>9}{'recall@p95':>12}  top sensors (by z)")
    for name, n, n_excluded, auroc, recall, top_sensors in rows:
        print(f"{name:<16}{n:>6}{n_excluded:>6}{auroc:>9.3f}{recall:>12.3f}  {top_sensors}")

    baseline_report_path = OUT_DIR / "gdn_baseline_separability_report.json"
    if baseline_report_path.exists():
        with open(baseline_report_path) as f:
            baseline_report = json.load(f)
        print(f"\n{'fault type':<16}{'baseline AUROC':>16}{'physics-residual AUROC':>24}{'delta':>9}")
        for name in report["fault_types"]:
            if name in baseline_report["fault_types"]:
                b = baseline_report["fault_types"][name]["auroc"]
                p = report["fault_types"][name]["auroc"]
                print(f"{name:<16}{b:>16.3f}{p:>24.3f}{p-b:>9.3f}")

    torch.save({"model_state_dict": model.state_dict(), "eval_report": report},
               OUT_DIR / "gdn_physics_robo3er.pth")
    with open(OUT_DIR / "gdn_physics_separability_report.json", "w") as f:
        json.dump(report, f, indent=2)

    import csv
    with open(OUT_DIR / "gdn_physics_full_z_per_class.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "feature", "z_score"])
        for fault_name, z_vec in full_z_by_fault.items():
            for feat_idx, z_val in enumerate(z_vec):
                writer.writerow([fault_name, cols_renamed[feat_idx], float(z_val)])

    print(f"\nsaved -> {OUT_DIR / 'gdn_physics_robo3er.pth'}")
    print(f"saved -> {OUT_DIR / 'gdn_physics_separability_report.json'}")
    print(f"saved -> {OUT_DIR / 'gdn_physics_full_z_per_class.csv'}")


if __name__ == "__main__":
    main()
