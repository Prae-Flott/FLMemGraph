#!/usr/bin/env python3
"""
Joint Prototype Memory + Physics-Relation GDN + Anomaly Attention (v3) on
robo3er, per `docs/joint_prototype_physics_gdn_anomaly_attention_prompt.md`.
First run of `JointPrototypeV31` (`src/models/joint_prototype_model.py`) outside
Paderborn -- reuses the model unchanged, only the node set / declared
physics edges / relation types are robo3er-specific, built from
`benchmark/datasets/robo3er_physics.md` / `src/robo3er/feature_groups.py`'s
kinematic chain rather than bearing physics.

ALL 68 kept features (`src/robo3er/dataset.py`'s `drop_dead=True` set) are
used as nodes, not just a hand-picked 7 -- matching every other dataset in
this project (Paderborn's 6 nodes, Sielaff's 39, voraus-AD's 66 are each
close to that dataset's FULL available channel set). The original 7-node
version undersold robo3er relative to every other dataset here: V2's
node-deviation signal (`d_node`) needs no declared physics relation at
all to be useful (see signal B in `memory/scoring-signals-B-C-E-H.md`), so there was
no real reason to drop odometry position/orientation, wheel ticks, IMU
acceleration, PWM, or the battery/IR/cliff/status channels just because
they don't have a DECLARED edge -- they still contribute a per-node
deviation score. Only the (still small) declared-edge subgraph below
needs an actual verified physical relation. (See
`memory/scoring-signals-B-C-E-H.md` for current signal B numbers.)

Declared edges (7, same physical relations as before, now embedded in the
68-node graph -- nodes with no declared incoming edge simply get 0 from
`resid_phys`, per `TypedRelationAnomalyHead`'s docstring, but still
score via `d_node`/`resid_struct`), typed per Sec 8/9 of the design doc:
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
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_prototype_model import JointPrototypeV31  # noqa: E402
from datasets.robo3er_adapter import load  # noqa: E402

OUT_DIR = REPO_ROOT / "checkpoints" / "robo3er"

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

WINDOW_LEN = 60  # robo3er windows are already fixed T=60, no extra windowing
BATCH_SIZE = 256
EPOCHS = 12
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 16
TOP_K = 8  # raised from 5 -- 68 nodes vs. the original 7 means each node now has far
           # more OTHER nodes to potentially attend to, TOP_K=5 would cover under 10%
           # of them (vs. ~70% at the old 7-node scale)
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_TYPED = 0.5
MIN_PROTO_SAMPLES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def per_sample_scores(model, windows, batch_size=BATCH_SIZE):
    model.eval()
    d_proto_all, d_node_all, resid_struct_all, resid_phys_all, idx_all, r_edge_all, d_mahal_all = [], [], [], [], [], [], []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(DEVICE)
        out = model(batch, training_mode=False)
        d_proto_all.append(out["d_proto"].cpu().numpy())
        d_node_all.append(out["d_node"].cpu().numpy())
        resid_struct_all.append(out["resid_struct"].cpu().numpy())
        resid_phys_all.append(out["resid_phys"].cpu().numpy())
        idx_all.append(out["idx"].cpu().numpy())
        r_edge_all.append(out["r_edge"].cpu().numpy())
        d_mahal_all.append(out["d_mahal"].cpu().numpy())
    return (np.concatenate(d_proto_all), np.concatenate(d_node_all), np.concatenate(resid_struct_all),
            np.concatenate(resid_phys_all), np.concatenate(idx_all), np.concatenate(r_edge_all),
            np.concatenate(d_mahal_all))


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
            l_edge = out["resid_struct"].mean()
            l_typed = out["r_edge"].mean()
            loss = l_pred + BETA * l_me + l_mc + LAMBDA_EDGE * l_edge + LAMBDA_TYPED * l_typed
            loss.backward()
            optimizer.step()
            fit_loss += loss.item() * batch.size(0)
        fit_loss /= len(fit_arr)

        d_proto, _, _, _, _, _, _ = per_sample_scores(model, calib_arr)
        calib_mse = float(d_proto.mean())
        print(f"  epoch {epoch:2d}  fit_loss={fit_loss:.6f}  calib_d_proto_mean={calib_mse:.6f}  "
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

    print("loading robo3er (pooled across 5 robots), using ALL 68 kept features as nodes ...")
    fit_arr, calib_arr, test_normal_arr, faults, cols = load(pooled=True)
    num_nodes = len(cols)
    node_idx = {name: i for i, name in enumerate(cols)}
    prior_edges = [(node_idx[s], node_idx[d]) for s, d in EDGES_NAMED]
    print(f"  fit={fit_arr.shape} calib={calib_arr.shape} test_normal={test_normal_arr.shape}")
    for name, arr in faults.items():
        print(f"  fault[{name}]={arr.shape}")

    print(f"\ntraining JointPrototypeV31 (nodes={num_nodes}, prior_edges={len(prior_edges)}, "
          f"edge_types={EDGE_TYPES}, top_k={TOP_K}, M={NUM_PROTOTYPES}, embed_dim={EMBED_DIM}) ...")
    model = JointPrototypeV31(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=EMBED_DIM,
                                  num_prototypes=NUM_PROTOTYPES, prior_edges=prior_edges,
                                  edge_types=EDGE_TYPES, top_k=TOP_K).to(DEVICE)
    model = train(model, fit_arr, calib_arr)

    print("\nStage B: calibrating typed-head and covariance-head from calib split ...")
    _, d_node_calib_b, _, _, calib_idx, calib_r, _ = per_sample_scores(model, calib_arr)
    model.typed_head.set_calibration(calib_r, calib_idx, NUM_PROTOTYPES, min_samples=MIN_PROTO_SAMPLES)
    cov_min = max(MIN_PROTO_SAMPLES, num_nodes + 1)
    model.cov_head.set_calibration(d_node_calib_b, calib_idx, min_samples=cov_min)
    n_valid_proto = int(model.typed_head.calib_valid_proto.sum())
    n_valid_cov = int(model.cov_head.calib_valid.sum())
    print(f"  typed_head: {n_valid_proto}/{NUM_PROTOTYPES} prototypes with >= {MIN_PROTO_SAMPLES} calib windows")
    print(f"  cov_head:   {n_valid_cov}/{NUM_PROTOTYPES} prototypes with >= {cov_min} calib windows")

    print("\ncalibrating from calib split ...")
    _, d_node_calib, resid_struct_calib, resid_phys_calib, _, _, d_mahal_calib = per_sample_scores(model, calib_arr)
    _, d_node_normal, resid_struct_normal, resid_phys_normal, _, _, d_mahal_normal = per_sample_scores(model, test_normal_arr)

    z_node_normal = zscore(d_node_normal, d_node_calib)
    z_struct_normal = zscore(resid_struct_normal, resid_struct_calib)
    z_phys_normal = zscore(resid_phys_normal, resid_phys_calib)
    z_mahal_normal = zscore(d_mahal_normal, d_mahal_calib)

    def ablation_scores(z_node, z_struct, z_phys, z_mahal):
        b = z_node.max(axis=1)
        return {
            "B_node_max": b,
            "C_struct_max": z_struct.max(axis=1),
            "E_phys_max": z_phys.max(axis=1),
            "F_v3_max": np.max(np.stack([z_node.max(axis=1), z_struct.max(axis=1), z_phys.max(axis=1)], axis=1), axis=1),
            "H_cov_mahal": z_mahal,
            "I_node_cov_max": np.maximum(b, z_mahal),
            "J_v3_cov_max": np.max(np.stack([z_node.max(axis=1), z_struct.max(axis=1), z_phys.max(axis=1), z_mahal], axis=1), axis=1),
        }

    scores_normal = ablation_scores(z_node_normal, z_struct_normal, z_phys_normal, z_mahal_normal)

    print("\nscoring fault types ...")
    report = {"config": {"embed_dim": EMBED_DIM, "num_prototypes": NUM_PROTOTYPES, "top_k": TOP_K,
                           "window_len": WINDOW_LEN, "epochs": EPOCHS, "prior_edges": EDGES_NAMED,
                           "edge_types": EDGE_TYPES, "min_proto_samples": MIN_PROTO_SAMPLES},
               "nodes": cols, "final_prior_bias_strength": model.edge_head.prior_bias_strength.item(),
               "final_typed_temperature": model.typed_head.log_temperature.exp().item(),
               "n_valid_prototypes_for_typed_calib": n_valid_proto,
               "n_valid_prototypes_for_cov_calib": n_valid_cov,
               "fault_types": {}}
    rows = {k: [] for k in scores_normal}
    for name, arr in faults.items():
        _, d_node_f, resid_struct_f, resid_phys_f, _, _, d_mahal_f = per_sample_scores(model, arr)
        z_node_f = zscore(d_node_f, d_node_calib)
        z_struct_f = zscore(resid_struct_f, resid_struct_calib)
        z_phys_f = zscore(resid_phys_f, resid_phys_calib)
        z_mahal_f = zscore(d_mahal_f, d_mahal_calib)
        scores_fault = ablation_scores(z_node_f, z_struct_f, z_phys_f, z_mahal_f)

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
    with open(OUT_DIR / "robo3er_v3_1_report.json", "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "report": report},
                OUT_DIR / "robo3er_v3_1.pth")
    print(f"\nsaved -> {OUT_DIR / 'robo3er_v3_1_report.json'}")


if __name__ == "__main__":
    main()
