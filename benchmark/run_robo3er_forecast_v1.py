#!/usr/bin/env python3
"""
Joint Prototype Memory V3.1 + ForecastHead (new signal K) on robo3er, per
`shared_forecast_head_proposal.md`. First run of `JointPrototypeV31Forecast`
(`src/models/joint_prototype_model.py`) -- adds a GDN-style cross-node
attention forecast head, trained to predict the LAST `FORECAST_H` timesteps
of each window from its first `WINDOW_LEN - FORECAST_H` timesteps, in raw
feature space (not z/deviation space -- see `ForecastHead`'s docstring for
why). This is the proposal's "step 1" only: the shared backbone is left as
the existing `SharedEncoder` (Linear), unchanged from `run_robo3er_v3_1.py`
-- B/C/E/H are computed exactly as in V31 and are not expected to move.
The "step 0" backbone-upgrade axis (UNet/Transformer) is NOT part of this
run; see the proposal doc's two-independent-risk-axes section for why they
must be validated separately.

Also note the proposal's forecast target is a genuinely separate FUTURE
window; this run instead forecasts a SUFFIX of the SAME window from its
PREFIX (in-window split), because `benchmark/datasets/robo3er_adapter.py`
hands back already-extracted `[N, T, F]` windows with no cross-window
temporal linkage. See `ForecastHead`'s docstring.

Same fit/calib/test_normal/fault-type split, same 68-node graph and 7
declared physics edges as `run_robo3er_v3_1.py`. New signal:
  K_forecast_max = max_i zscore(k_resid_i)  (per-node squared forecast
  error, raw feature space, z-scored against the calib split like B/C/E/H)
and the combination L_full_plus_forecast = max(B, C, E, H, K).

Usage:
    python3 run_robo3er_forecast_v1.py
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
from joint_prototype_model import JointPrototypeV31Forecast  # noqa: E402
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
FORECAST_H = 12  # last 20% of the window is the forecast target (in-window split)
BATCH_SIZE = 256
EPOCHS = 12
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 16
TOP_K = 8
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_TYPED = 0.5
LAMBDA_FORECAST = 0.5
MIN_PROTO_SAMPLES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def per_sample_scores(model, windows, batch_size=BATCH_SIZE):
    model.eval()
    d_proto_all, d_node_all, resid_struct_all, resid_phys_all = [], [], [], []
    idx_all, r_edge_all, d_mahal_all, k_resid_all = [], [], [], []
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
        k_resid_all.append(out["k_resid"].cpu().numpy())
    return (np.concatenate(d_proto_all), np.concatenate(d_node_all), np.concatenate(resid_struct_all),
            np.concatenate(resid_phys_all), np.concatenate(idx_all), np.concatenate(r_edge_all),
            np.concatenate(d_mahal_all), np.concatenate(k_resid_all))


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
            l_forecast = out["k_resid"].mean()
            loss = (l_pred + BETA * l_me + l_mc + LAMBDA_EDGE * l_edge
                    + LAMBDA_TYPED * l_typed + LAMBDA_FORECAST * l_forecast)
            loss.backward()
            optimizer.step()
            fit_loss += loss.item() * batch.size(0)
        fit_loss /= len(fit_arr)

        d_proto, _, _, _, _, _, _, _ = per_sample_scores(model, calib_arr)
        calib_mse = float(d_proto.mean())
        print(f"  epoch {epoch:2d}  fit_loss={fit_loss:.6f}  calib_d_proto_mean={calib_mse:.6f}  "
              f"codebook_util={model.memory.codebook_utilization():.2f}  "
              f"prior_bias_strength={model.edge_head.prior_bias_strength.item():.3f}  "
              f"typed_temp={model.typed_head.log_temperature.exp().item():.3f}  "
              f"forecast_prior_bias={model.forecast_head.prior_bias_strength.item():.3f}")
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

    print(f"\ntraining JointPrototypeV31Forecast (nodes={num_nodes}, prior_edges={len(prior_edges)}, "
          f"edge_types={EDGE_TYPES}, top_k={TOP_K}, M={NUM_PROTOTYPES}, embed_dim={EMBED_DIM}, "
          f"forecast_h={FORECAST_H}) ...")
    model = JointPrototypeV31Forecast(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=EMBED_DIM,
                                       num_prototypes=NUM_PROTOTYPES, prior_edges=prior_edges,
                                       edge_types=EDGE_TYPES, forecast_h=FORECAST_H, top_k=TOP_K).to(DEVICE)
    model = train(model, fit_arr, calib_arr)

    print("\nStage B: calibrating typed-head and covariance-head from calib split ...")
    _, d_node_calib_b, _, _, calib_idx, calib_r, _, _ = per_sample_scores(model, calib_arr)
    model.typed_head.set_calibration(calib_r, calib_idx, NUM_PROTOTYPES, min_samples=MIN_PROTO_SAMPLES)
    cov_min = max(MIN_PROTO_SAMPLES, num_nodes + 1)
    model.cov_head.set_calibration(d_node_calib_b, calib_idx, min_samples=cov_min)
    n_valid_proto = int(model.typed_head.calib_valid_proto.sum())
    n_valid_cov = int(model.cov_head.calib_valid.sum())
    print(f"  typed_head: {n_valid_proto}/{NUM_PROTOTYPES} prototypes with >= {MIN_PROTO_SAMPLES} calib windows")
    print(f"  cov_head:   {n_valid_cov}/{NUM_PROTOTYPES} prototypes with >= {cov_min} calib windows")

    print("\ncalibrating from calib split ...")
    (_, d_node_calib, resid_struct_calib, resid_phys_calib, _, _,
     d_mahal_calib, k_resid_calib) = per_sample_scores(model, calib_arr)
    (_, d_node_normal, resid_struct_normal, resid_phys_normal, _, _,
     d_mahal_normal, k_resid_normal) = per_sample_scores(model, test_normal_arr)

    z_node_normal = zscore(d_node_normal, d_node_calib)
    z_struct_normal = zscore(resid_struct_normal, resid_struct_calib)
    z_phys_normal = zscore(resid_phys_normal, resid_phys_calib)
    z_mahal_normal = zscore(d_mahal_normal, d_mahal_calib)
    z_forecast_normal = zscore(k_resid_normal, k_resid_calib)

    def ablation_scores(z_node, z_struct, z_phys, z_mahal, z_forecast):
        b = z_node.max(axis=1)
        c = z_struct.max(axis=1)
        e = z_phys.max(axis=1)
        h = z_mahal
        k = z_forecast.max(axis=1)
        return {
            "B_node_max": b,
            "C_struct_max": c,
            "E_phys_max": e,
            "F_v3_max": np.max(np.stack([b, c, e], axis=1), axis=1),
            "H_cov_mahal": h,
            "I_node_cov_max": np.maximum(b, h),
            "J_v3_cov_max": np.max(np.stack([b, c, e, h], axis=1), axis=1),
            "K_forecast_max": k,
            "L_full_plus_forecast": np.max(np.stack([b, c, e, h, k], axis=1), axis=1),
        }

    scores_normal = ablation_scores(z_node_normal, z_struct_normal, z_phys_normal, z_mahal_normal, z_forecast_normal)

    print("\nscoring fault types ...")
    report = {"config": {"embed_dim": EMBED_DIM, "num_prototypes": NUM_PROTOTYPES, "top_k": TOP_K,
                          "window_len": WINDOW_LEN, "forecast_h": FORECAST_H, "epochs": EPOCHS,
                          "prior_edges": EDGES_NAMED, "edge_types": EDGE_TYPES,
                          "min_proto_samples": MIN_PROTO_SAMPLES, "lambda_forecast": LAMBDA_FORECAST},
              "nodes": cols, "final_prior_bias_strength": model.edge_head.prior_bias_strength.item(),
              "final_typed_temperature": model.typed_head.log_temperature.exp().item(),
              "final_forecast_prior_bias_strength": model.forecast_head.prior_bias_strength.item(),
              "n_valid_prototypes_for_typed_calib": n_valid_proto,
              "n_valid_prototypes_for_cov_calib": n_valid_cov,
              "fault_types": {}}
    rows = {k: [] for k in scores_normal}
    for name, arr in faults.items():
        (_, d_node_f, resid_struct_f, resid_phys_f, _, _,
         d_mahal_f, k_resid_f) = per_sample_scores(model, arr)
        z_node_f = zscore(d_node_f, d_node_calib)
        z_struct_f = zscore(resid_struct_f, resid_struct_calib)
        z_phys_f = zscore(resid_phys_f, resid_phys_calib)
        z_mahal_f = zscore(d_mahal_f, d_mahal_calib)
        z_forecast_f = zscore(k_resid_f, k_resid_calib)
        scores_fault = ablation_scores(z_node_f, z_struct_f, z_phys_f, z_mahal_f, z_forecast_f)

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
    print(f"final forecast-head prior_bias_strength: {model.forecast_head.prior_bias_strength.item():.4f}")

    report["summary_mean_auroc"] = summary
    with open(OUT_DIR / "robo3er_forecast_v1_report.json", "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "report": report},
               OUT_DIR / "robo3er_forecast_v1.pth")
    print(f"\nsaved -> {OUT_DIR / 'robo3er_forecast_v1_report.json'}")


if __name__ == "__main__":
    main()
