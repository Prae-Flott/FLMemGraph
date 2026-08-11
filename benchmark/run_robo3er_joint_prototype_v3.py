#!/usr/bin/env python3
"""
Joint Prototype Memory + Physics-Relation GDN + Anomaly Attention (v3) on
robo3er, per `docs/joint_prototype_physics_gdn_anomaly_attention_prompt.md`.
First run of `JointPrototypeGDNv3` (`src/joint_prototype_model.py`) outside
Paderborn -- reuses the model unchanged, only the node set / declared
physics edges / relation types are robo3er-specific, built from
`benchmark/datasets/robo3er_physics.md` / `src/feature_groups.py`'s
kinematic chain rather than bearing physics.

7 nodes, chosen as the smallest set that covers robo3er's ONE validated
physical relation (differential-drive kinematics, `src/kinematics.py`)
plus its natural extensions already documented but not yet exploited
(IMU cross-check, actuation current):
  - wheel_vels_velocity_left / _right   -- the two independently driven wheels
  - odom_odo_lintw_x                     -- chassis linear velocity (odometry)
  - odom_odo_angtw_z                     -- chassis angular velocity (odometry)
  - imu_imu_angvel_z                     -- chassis yaw rate, INDEPENDENT estimator
  - wheel_status_current_ma_left/_right  -- per-wheel motor current (actuation)

Declared edges (7), typed per Sec 8/9 of the design doc:
  - wheel_l/r -> odom_lin, wheel_l/r -> odom_ang: "proportional" (textbook
    differential-drive kinematics, v_lin = k_v*(v_l+v_r), v_ang = k_w*(v_r-v_l)
    -- the same relation `kinematics.py`'s residual is built from, just
    expressed as a learnable typed edge instead of an OLS pre-fit).
  - odom_ang -> imu_angvel_z: "proportional" -- two independent sensors
    estimating the SAME physical quantity (chassis yaw rate); a real gap
    here is a stronger slip signal than wheel-vs-odom per
    `feature_groups.py`'s docstring, and was never implemented before now.
  - current_l -> wheel_l, current_r -> wheel_r: "nonlinear" -- motor
    current under load vs. resulting wheel velocity is a genuinely
    nonlinear actuation relationship, not assumed linear.

Same fit/calib/test_normal/fault-type split as every other robo3er
baseline here (`benchmark/datasets/robo3er_adapter.py`, pooled across all
5 robots). Unlike Paderborn's per-file scripts, robo3er's unit is already
a fixed T=60 window -- no additional windowing/aggregation needed, AUROC
is computed per-window per fault type, matching
`run_ifcaae_baseline.py`/`run_fedavg_baseline.py`'s convention.

Usage:
    python3 run_robo3er_joint_prototype_v3.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_prototype_model import JointPrototypeGDNv3  # noqa: E402
from datasets.robo3er_adapter import load  # noqa: E402

OUT_DIR = REPO_ROOT / "checkpoints" / "robo3er"

NODE_NAMES = [
    "wheel_vels_velocity_left", "wheel_vels_velocity_right",
    "odom_odo_lintw_x", "odom_odo_angtw_z", "imu_imu_angvel_z",
    "wheel_status_current_ma_left", "wheel_status_current_ma_right",
]
NODE_IDX = {name: i for i, name in enumerate(NODE_NAMES)}
EDGES_NAMED = [
    ("wheel_vels_velocity_left", "odom_odo_lintw_x"),
    ("wheel_vels_velocity_right", "odom_odo_lintw_x"),
    ("wheel_vels_velocity_left", "odom_odo_angtw_z"),
    ("wheel_vels_velocity_right", "odom_odo_angtw_z"),
    ("odom_odo_angtw_z", "imu_imu_angvel_z"),
    ("wheel_status_current_ma_left", "wheel_vels_velocity_left"),
    ("wheel_status_current_ma_right", "wheel_vels_velocity_right"),
]
EDGE_TYPES = [
    "proportional", "proportional", "proportional", "proportional",
    "proportional",
    "nonlinear", "nonlinear",
]
PRIOR_EDGES = [(NODE_IDX[s], NODE_IDX[d]) for s, d in EDGES_NAMED]

WINDOW_LEN = 60  # robo3er windows are already fixed T=60, no extra windowing
BATCH_SIZE = 256
EPOCHS = 12
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 16
TOP_K = 5
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_TYPED = 0.5
MIN_PROTO_SAMPLES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_nodes(windows_68, all_cols):
    idx = [all_cols.index(n) for n in NODE_NAMES]
    return windows_68[:, :, idx].astype(np.float32)


@torch.no_grad()
def per_sample_scores(model, windows, batch_size=BATCH_SIZE):
    model.eval()
    d_G_all, s_node_all, s_edge_all, s_typed_all, idx_all, r_all = [], [], [], [], [], []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(DEVICE)
        out = model(batch, training_mode=False)
        d_G_all.append(out["d_G"].cpu().numpy())
        s_node_all.append(out["s_node"].cpu().numpy())
        s_edge_all.append(out["s_edge"].cpu().numpy())
        s_typed_all.append(out["s_node_typed"].cpu().numpy())
        idx_all.append(out["idx"].cpu().numpy())
        r_all.append(out["r"].cpu().numpy())
    return (np.concatenate(d_G_all), np.concatenate(s_node_all), np.concatenate(s_edge_all),
            np.concatenate(s_typed_all), np.concatenate(idx_all), np.concatenate(r_all))


def train(model, fit_arr, calib_arr):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fit_arr)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    best_calib_mse, best_state = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        fit_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch, training_mode=True)
            l_pred = torch.nn.functional.mse_loss(out["x_hat"], batch)
            l_me, l_mc = model.memory.commitment_losses(out["z"], out["p_star"])
            l_edge = out["s_edge"].mean()
            l_typed = out["r"].mean()
            loss = l_pred + BETA * l_me + l_mc + LAMBDA_EDGE * l_edge + LAMBDA_TYPED * l_typed
            loss.backward()
            optimizer.step()
            fit_loss += loss.item() * batch.size(0)
        fit_loss /= len(fit_arr)

        d_G, _, _, _, _, _ = per_sample_scores(model, calib_arr)
        calib_mse = float(d_G.mean())
        print(f"  epoch {epoch:2d}  fit_loss={fit_loss:.6f}  calib_dG_mean={calib_mse:.6f}  "
              f"codebook_util={model.memory.codebook_utilization():.2f}  "
              f"prior_bias_strength={model.edge_head.prior_bias_strength.item():.3f}  "
              f"typed_temp={model.typed_head.log_temperature.exp().item():.3f}")
        if calib_mse < best_calib_mse:
            best_calib_mse = calib_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def zscore(x, calib_x):
    median = np.median(calib_x, axis=0)
    q75, q25 = np.percentile(calib_x, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    return (x - median) / iqr


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    num_nodes = len(NODE_NAMES)

    print("loading robo3er (pooled across 5 robots) ...")
    fit_w68, calib_w68, test_normal_w68, faults68, cols = load(pooled=True)
    fit_arr = select_nodes(fit_w68, cols)
    calib_arr = select_nodes(calib_w68, cols)
    test_normal_arr = select_nodes(test_normal_w68, cols)
    faults = {name: select_nodes(w, cols) for name, w in faults68.items()}
    print(f"  fit={fit_arr.shape} calib={calib_arr.shape} test_normal={test_normal_arr.shape}")
    for name, arr in faults.items():
        print(f"  fault[{name}]={arr.shape}")

    print(f"\ntraining JointPrototypeGDNv3 (nodes={num_nodes}, prior_edges={len(PRIOR_EDGES)}, "
          f"edge_types={EDGE_TYPES}, top_k={TOP_K}, M={NUM_PROTOTYPES}, embed_dim={EMBED_DIM}) ...")
    model = JointPrototypeGDNv3(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=EMBED_DIM,
                                  num_prototypes=NUM_PROTOTYPES, prior_edges=PRIOR_EDGES,
                                  edge_types=EDGE_TYPES, top_k=TOP_K).to(DEVICE)
    model = train(model, fit_arr, calib_arr)

    print("\nStage B: estimating prototype-conditioned edge-residual statistics from calib split ...")
    _, _, _, _, calib_idx, calib_r = per_sample_scores(model, calib_arr)
    model.typed_head.set_calibration(calib_r, calib_idx, NUM_PROTOTYPES, min_samples=MIN_PROTO_SAMPLES)
    n_valid_proto = int(model.typed_head.calib_valid_proto.sum())
    print(f"  {n_valid_proto}/{NUM_PROTOTYPES} prototypes had >= {MIN_PROTO_SAMPLES} calib windows "
          f"(rest fall back to global edge stats)")

    print("\ncalibrating from calib split ...")
    d_G_calib, s_node_calib, s_edge_calib, s_typed_calib, _, _ = per_sample_scores(model, calib_arr)
    d_G_normal, s_node_normal, s_edge_normal, s_typed_normal, _, _ = per_sample_scores(model, test_normal_arr)

    z_dG_normal = zscore(d_G_normal, d_G_calib)
    z_node_normal = zscore(s_node_normal, s_node_calib)
    z_edge_normal = zscore(s_edge_normal, s_edge_calib)
    z_typed_normal = zscore(s_typed_normal, s_typed_calib)

    def ablation_scores(z_dG, z_node, z_edge, z_typed):
        return {
            "A_prototype_only": z_dG,
            "B_prototype_plus_node": np.maximum(z_dG, z_node.max(axis=1)),
            "C_edge_only": z_edge.max(axis=1),
            "D_full_v2": np.maximum(np.maximum(z_dG, z_node.max(axis=1)), z_edge.max(axis=1)),
            "E_typed_edge_only": z_typed.max(axis=1),
            "F_full_v3": np.max(np.stack([z_dG, z_node.max(axis=1), z_edge.max(axis=1),
                                            z_typed.max(axis=1)], axis=1), axis=1),
        }

    scores_normal = ablation_scores(z_dG_normal, z_node_normal, z_edge_normal, z_typed_normal)

    print("\nscoring fault types ...")
    report = {"config": {"embed_dim": EMBED_DIM, "num_prototypes": NUM_PROTOTYPES, "top_k": TOP_K,
                           "window_len": WINDOW_LEN, "epochs": EPOCHS, "prior_edges": EDGES_NAMED,
                           "edge_types": EDGE_TYPES, "min_proto_samples": MIN_PROTO_SAMPLES},
               "nodes": NODE_NAMES, "final_prior_bias_strength": model.edge_head.prior_bias_strength.item(),
               "final_typed_temperature": model.typed_head.log_temperature.exp().item(),
               "n_valid_prototypes_for_typed_calib": n_valid_proto,
               "fault_types": {}}
    rows = {k: [] for k in scores_normal}
    for name, arr in faults.items():
        d_G_f, s_node_f, s_edge_f, s_typed_f, _, _ = per_sample_scores(model, arr)
        z_dG_f = zscore(d_G_f, d_G_calib)
        z_node_f = zscore(s_node_f, s_node_calib)
        z_edge_f = zscore(s_edge_f, s_edge_calib)
        z_typed_f = zscore(s_typed_f, s_typed_calib)
        scores_fault = ablation_scores(z_dG_f, z_node_f, z_edge_f, z_typed_f)

        aurocs = {}
        for key in scores_normal:
            labels = np.concatenate([np.zeros(len(scores_normal[key])), np.ones(len(scores_fault[key]))])
            scores = np.concatenate([scores_normal[key], scores_fault[key]])
            auroc = float(sk_metrics.roc_auc_score(labels, scores))
            aurocs[key] = auroc
            rows[key].append((name, auroc))
        report["fault_types"][name] = {"n": int(len(arr)), **aurocs}
        print(f"  {name:<16}n={len(arr):<5}" + "  ".join(f"{k}={v:.3f}" for k, v in aurocs.items()))

    print(f"\n{'method':<24}{'mean AUROC (4 faults)':>22}")
    summary = {}
    for key in scores_normal:
        mean_auroc = float(np.mean([r[1] for r in rows[key]]))
        summary[key] = mean_auroc
        print(f"{key:<24}{mean_auroc:>22.3f}")

    print(f"\nfinal prior_bias_strength: {model.edge_head.prior_bias_strength.item():.4f}")
    print(f"final typed-edge attention temperature: {model.typed_head.log_temperature.exp().item():.4f}")

    report["summary_mean_auroc"] = summary
    with open(OUT_DIR / "robo3er_joint_prototype_v3_report.json", "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "report": report},
                OUT_DIR / "robo3er_joint_prototype_v3.pth")
    print(f"\nsaved -> {OUT_DIR / 'robo3er_joint_prototype_v3_report.json'}")


if __name__ == "__main__":
    main()
