#!/usr/bin/env python3
"""
Joint Prototype Memory V3.1 + ForecastHead (signal K), v2, on robo3er, per
`docs/shared_forecast_head_proposal.md`'s revised technical route (2026-08-17
update). Supersedes `run_robo3er_forecast_v1.py`'s in-window prefix/suffix
split, which the proposal's author flagged as leaking most of the forecast
target through window-to-window overlap (`window_size=60`, `stride=32` means
adjacent windows share 28 of 60 steps -- a naive "predict the window's own
tail" task is 47% trivially copyable from the input's own tail).

**Data split: reuses the existing fit/calib/test_normal split unchanged --
no new split logic.** B/C/E/H's windowing/split is untouched. The only new
thing is, WITHIN this split, pairing each window index `i` with its
immediate successor `i+1` to build (input, future-target) pairs:

- `i+1` must belong to the SAME robot as `i` (`data/robo3er/partition.pkl`'s
  per-client index ranges -- robo3er's 5 robots' raw CSVs were windowed
  independently, so window index adjacency does NOT imply temporal
  adjacency across a robot boundary).
- For the three NORMAL splits (fit/calib/test_normal), `i+1` must also
  fall in the SAME split and be normal-labeled (`targets[i+1]==0`) --
  training/calibration must only see genuine normal-to-normal
  continuations. Orphan windows with no valid successor (robot/fault/split
  boundary) are dropped, not force-paired. Verified counts on current
  data: fit 3940->3929, calib 844->843, test_normal 845->839.
- For each FAULT type's evaluation windows, `i+1` only needs to exist and
  share the same robot (may itself be fault-labeled) -- evaluation is
  "does the forecast from this window's own trajectory match what
  actually happened next," which is meaningful whether or not the window
  is itself already anomalous. Verified: cable trapped 206->206 (no
  boundary loss), stuck 149->148 (1 orphan at the fault segment's tail).

**Forecast target is the non-overlapping RAW-TIME segment only:** window
`i` covers raw steps `[stride*i, stride*i+window_size)`; window `i+1`
covers `[stride*(i+1), stride*(i+1)+window_size)`. The genuinely-new
segment relative to window `i` is `[stride*i+window_size,
stride*(i+1)+window_size)`, length `stride` -- which is exactly window
`i+1`'s LAST `stride` raw timesteps. So `x_hat_future` has shape
`[B, stride, N]`, not `[B, window_size, N]`; forecasting the whole next
window would trivially reproduce the 28-step overlap already visible in
the input.

Same 68-node graph, 7 declared physics edges, fit/calib/test_normal
convention as `run_robo3er_v3_1.py`. New signal:
  K_forecast_max = max_i zscore(k_resid_i)
and L_full_plus_forecast = max(B, C, E, H, K).

**2026-08-17 addendum, per `forecast_head_k_assessment.md`'s improvement-1
and improvement-2 experiments (both cheap, run before any backbone work):**

- `--no-forecast-prior`: ablates the forecast head's declared-edge prior
  bias specifically (`forecast_prior_edges=None`), leaving B/C/E/H's
  `edge_head`/`typed_head` untouched, to answer "is K's AUROC coming from
  the declared physics edges, or from unbiased learned top-k attention
  alone?" Also reports per-declared-edge-target-node calib-set forecast
  MSE (with vs. without the bias) as a mechanism-level check independent
  of the downstream AUROC.
- `--horizon-mult M` (default 1): forecast horizon = `M * stride` raw
  timesteps, built by chaining M consecutive non-overlapping tail
  segments (windows `i+1..i+M`, each contributing its own `stride`-length
  tail) rather than repeating one window's overlapping content -- tests
  whether a longer forecast horizon accumulates enough drift from
  `stuck`'s steady electrical current to make the forecast error more
  anomaly-sensitive (open question per the proposal's validation step 2).

Usage:
    python3 run_robo3er_forecast_v2.py [--no-forecast-prior] [--horizon-mult M] [--out-suffix NAME]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "robo3er"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_prototype_model import JointPrototypeV31Forecast  # noqa: E402
from dataset import load_robo3er, split_normal, fit_scaler, scale  # noqa: E402

OUT_DIR = REPO_ROOT / "checkpoints" / "robo3er"
DATA_DIR = REPO_ROOT / "data" / "robo3er"

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

WINDOW_LEN = 60
STRIDE = 32       # from data/robo3er/metadata.json -- forecast_h = stride
                   # (the genuinely non-overlapping segment of the next window)
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


def build_window_robot_map(num_windows):
    with open(DATA_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    window_robot = np.full(num_windows, -1, dtype=int)
    for client_id, idx_dict in enumerate(partition["data_indices"]):
        idx = np.array(idx_dict["train"] + idx_dict["val"] + idx_dict["test"], dtype=int)
        window_robot[idx] = client_id
    assert (window_robot >= 0).all(), "every window must be assigned to a robot"
    return window_robot


def build_pairs(idx_arr, targets, window_robot, horizon_mult, require_next_normal, require_same_split=False):
    """Returns (valid_i, chains, mask) -- see module docstring for the
    three validity conditions, now checked over the WHOLE chain
    `[i+1, ..., i+horizon_mult]` (all `horizon_mult` successors must be
    valid, not just the first). `chains` is a list of length-`horizon_mult`
    index lists, one per entry in `valid_i`. `mask` is a boolean array
    over `idx_arr` (same order), True where that position has a fully
    valid chain -- lets the caller subset any other per-`idx_arr`-position
    array (e.g. already-computed B/C/E/H z-scores) down to the same
    paired subset used for K. `require_same_split=True` additionally
    requires every chain member in idx_arr itself (used for the three
    normal splits, matching the proposal's "同一段" condition)."""
    N = len(targets)
    idx_set = set(idx_arr.tolist()) if require_same_split else None
    vi, chains, mask = [], [], []
    for i in idx_arr:
        chain = [i + k for k in range(1, horizon_mult + 1)]
        valid = True
        for j in chain:
            if (j >= N or window_robot[j] != window_robot[i]
                    or (require_next_normal and targets[j] != 0)
                    or (require_same_split and j not in idx_set)):
                valid = False
                break
        mask.append(valid)
        if valid:
            vi.append(i)
            chains.append(chain)
    return np.array(vi, dtype=int), chains, np.array(mask, dtype=bool)


def gather_future(data_scaled, chains, stride):
    """chains: list of length-M index lists. Returns [len(chains), M*stride, N]
    -- each chain's target is the concatenation of its M members' own
    non-overlapping `stride`-length tail segments, in order (NOT M full
    overlapping windows -- see module docstring)."""
    if len(chains) == 0:
        return np.zeros((0, 0, data_scaled.shape[-1]), dtype=data_scaled.dtype)
    segments = [data_scaled[[c[k] for c in chains]][:, -stride:, :] for k in range(len(chains[0]))]
    return np.concatenate(segments, axis=1)


@torch.no_grad()
def per_sample_scores(model, windows, batch_size=BATCH_SIZE):
    model.eval()
    d_proto_all, d_node_all, resid_struct_all, resid_phys_all = [], [], [], []
    idx_all, r_edge_all, d_mahal_all = [], [], []
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


@torch.no_grad()
def forecast_scores(model, x_in, x_future, batch_size=BATCH_SIZE):
    model.eval()
    k_resid_all = []
    for i in range(0, len(x_in), batch_size):
        xb = torch.from_numpy(x_in[i : i + batch_size]).to(DEVICE)
        fb = torch.from_numpy(x_future[i : i + batch_size]).to(DEVICE)
        out = model(xb, training_mode=False, x_future=fb)
        k_resid_all.append(out["k_resid"].cpu().numpy())
    return np.concatenate(k_resid_all)


@torch.no_grad()
def forecast_mse_per_node(model, x_in, x_future, batch_size=BATCH_SIZE):
    """Mean squared forecast error PER NODE, averaged over the whole
    dataset and forecast horizon (not per-window, not max-aggregated) --
    a mechanism-level check ("does the declared edge help THIS target
    node's forecast get more accurate") independent of the downstream
    max-aggregated anomaly-detection AUROC."""
    model.eval()
    se_sum, count = None, 0
    for i in range(0, len(x_in), batch_size):
        xb = torch.from_numpy(x_in[i : i + batch_size]).to(DEVICE)
        fb = torch.from_numpy(x_future[i : i + batch_size]).to(DEVICE)
        out = model(xb, training_mode=False, x_future=fb)
        se = (out["x_hat_future"] - fb).pow(2).sum(dim=1)  # [B, N], summed over horizon
        se_sum = se.sum(dim=0) if se_sum is None else se_sum + se.sum(dim=0)
        count += xb.shape[0] * fb.shape[1]  # windows * horizon length
    return (se_sum / count).cpu().numpy()  # [N]


def train(model, fit_arr, fit_future, calib_arr):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fit_arr), torch.from_numpy(fit_future)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    best_calib_mse, best_state = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        fit_loss = 0.0
        for batch, future in loader:
            batch, future = batch.to(DEVICE), future.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch, training_mode=True, x_future=future)
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

        d_proto, _, _, _, _, _, _ = per_sample_scores(model, calib_arr)
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


def main(use_forecast_prior=True, horizon_mult=1, out_suffix=None):
    forecast_h = STRIDE * horizon_mult
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("loading robo3er (pooled across 5 robots), using ALL 68 kept features as nodes ...")
    data, targets, cols, label_map, _ = load_robo3er(drop_dead=True)
    num_nodes = len(cols)
    node_idx = {name: i for i, name in enumerate(cols)}
    prior_edges = [(node_idx[s], node_idx[d]) for s, d in EDGES_NAMED]
    forecast_prior_edges = prior_edges if use_forecast_prior else None
    edge_target_nodes = sorted({d for _s, d in prior_edges})  # declared edges' target nodes

    window_robot = build_window_robot_map(len(targets))
    fit_idx, calib_idx, test_normal_idx = split_normal(targets)
    scaler = fit_scaler(data, fit_idx)
    data_scaled = scale(data, scaler)  # scale ALL windows once, up front, consistently

    def make_pairs(idx_arr, require_next_normal, require_same_split):
        vi, chains, mask = build_pairs(idx_arr, targets, window_robot, horizon_mult,
                                        require_next_normal, require_same_split)
        x_in = data_scaled[vi]
        x_future = gather_future(data_scaled, chains, STRIDE)
        return x_in, x_future, mask, len(idx_arr), len(vi)

    fit_in, fit_future, _, n0, n1 = make_pairs(fit_idx, True, True)
    print(f"  fit   pairs: {n0} -> {n1}")
    calib_in, calib_future, _, n0, n1 = make_pairs(calib_idx, True, True)
    print(f"  calib pairs: {n0} -> {n1}")
    test_normal_in, test_normal_future, test_normal_mask, n0, n1 = make_pairs(test_normal_idx, True, True)
    print(f"  test_normal pairs: {n0} -> {n1}")

    fault_pairs = {}
    fault_masks = {}
    for label_id_str, name in label_map.items():
        label_id = int(label_id_str)
        if label_id == 0:
            continue
        fault_idx = np.where(targets == label_id)[0]
        if len(fault_idx) == 0:
            continue
        f_in, f_future, f_mask, n0, n1 = make_pairs(fault_idx, False, False)
        fault_pairs[name] = (f_in, f_future)
        fault_masks[name] = f_mask
        print(f"  fault[{name}] pairs: {n0} -> {n1}")

    # B/C/E/H still score on the FULL (unpaired) normal/fault window arrays,
    # matching run_robo3er_v3_1.py exactly -- only the forecast-loss training
    # batch and the K-signal scoring use the paired subset.
    fit_full = data_scaled[fit_idx]
    calib_full = data_scaled[calib_idx]
    test_normal_full = data_scaled[test_normal_idx]
    fault_full = {name: data_scaled[np.where(targets == int(lid))[0]]
                  for lid, name in label_map.items() if int(lid) != 0}

    print(f"\ntraining JointPrototypeV31Forecast (nodes={num_nodes}, prior_edges={len(prior_edges)}, "
          f"edge_types={EDGE_TYPES}, top_k={TOP_K}, M={NUM_PROTOTYPES}, embed_dim={EMBED_DIM}, "
          f"forecast_h={forecast_h}, use_forecast_prior={use_forecast_prior}) ...")
    model = JointPrototypeV31Forecast(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=EMBED_DIM,
                                       num_prototypes=NUM_PROTOTYPES, prior_edges=prior_edges,
                                       edge_types=EDGE_TYPES, forecast_h=forecast_h, top_k=TOP_K,
                                       forecast_prior_edges=forecast_prior_edges).to(DEVICE)
    model = train(model, fit_in, fit_future, calib_full)

    print("\nStage B: calibrating typed-head and covariance-head from calib split ...")
    _, d_node_calib_b, _, _, calib_idx_b, calib_r, _ = per_sample_scores(model, calib_full)
    model.typed_head.set_calibration(calib_r, calib_idx_b, NUM_PROTOTYPES, min_samples=MIN_PROTO_SAMPLES)
    cov_min = max(MIN_PROTO_SAMPLES, num_nodes + 1)
    model.cov_head.set_calibration(d_node_calib_b, calib_idx_b, min_samples=cov_min)
    n_valid_proto = int(model.typed_head.calib_valid_proto.sum())
    n_valid_cov = int(model.cov_head.calib_valid.sum())
    print(f"  typed_head: {n_valid_proto}/{NUM_PROTOTYPES} prototypes with >= {MIN_PROTO_SAMPLES} calib windows")
    print(f"  cov_head:   {n_valid_cov}/{NUM_PROTOTYPES} prototypes with >= {cov_min} calib windows")

    print("\ncalibrating from calib split ...")
    _, d_node_calib, resid_struct_calib, resid_phys_calib, _, _, d_mahal_calib = per_sample_scores(model, calib_full)
    _, d_node_normal, resid_struct_normal, resid_phys_normal, _, _, d_mahal_normal = per_sample_scores(model, test_normal_full)
    k_resid_calib = forecast_scores(model, calib_in, calib_future)
    k_resid_normal = forecast_scores(model, test_normal_in, test_normal_future)

    mse_per_node = forecast_mse_per_node(model, calib_in, calib_future)
    edge_target_mse = {cols[n]: float(mse_per_node[n]) for n in edge_target_nodes}
    print(f"\ndeclared-edge target-node calib forecast MSE (use_forecast_prior={use_forecast_prior}):")
    for name, mse in edge_target_mse.items():
        print(f"  {name:<32}{mse:.4f}")
    print(f"  (all-node mean: {float(mse_per_node.mean()):.4f}, "
          f"declared-target mean: {float(np.mean(list(edge_target_mse.values()))):.4f})")

    z_node_normal = zscore(d_node_normal, d_node_calib)
    z_struct_normal = zscore(resid_struct_normal, resid_struct_calib)
    z_phys_normal = zscore(resid_phys_normal, resid_phys_calib)
    z_mahal_normal = zscore(d_mahal_normal, d_mahal_calib)
    z_forecast_normal = zscore(k_resid_normal, k_resid_calib)

    def base_scores(z_node, z_struct, z_phys, z_mahal):
        b = z_node.max(axis=1)
        c = z_struct.max(axis=1)
        e = z_phys.max(axis=1)
        h = z_mahal
        return {
            "B_node_max": b,
            "C_struct_max": c,
            "E_phys_max": e,
            "F_v3_max": np.max(np.stack([b, c, e], axis=1), axis=1),
            "H_cov_mahal": h,
            "I_node_cov_max": np.maximum(b, h),
            "J_v3_cov_max": np.max(np.stack([b, c, e, h], axis=1), axis=1),
        }

    def add_forecast_scores(base, z_forecast, mask):
        """K/L/BK/CK/HK are computed on the PAIRED subset (mask over
        base's own index order) -- a few orphan windows without a valid
        next-window pair are dropped from these specifically, while
        B/C/E/F/H/I/J keep the full window count. BK/CK/HK are the three
        pairwise max-combinations requested alongside the full L
        combination, to see whether pairing K with just ONE existing
        signal avoids the ranking-corruption L suffers from (see
        memory/forecast-head-signal-k.md's combination-rules discussion)."""
        k = z_forecast.max(axis=1)
        b, c, e, h = (base["B_node_max"][mask], base["C_struct_max"][mask],
                      base["E_phys_max"][mask], base["H_cov_mahal"][mask])
        out = dict(base)
        out["K_forecast_max"] = k
        out["BK_max"] = np.maximum(b, k)
        out["CK_max"] = np.maximum(c, k)
        out["HK_max"] = np.maximum(h, k)
        out["L_full_plus_forecast"] = np.max(np.stack([b, c, e, h, k], axis=1), axis=1)
        return out

    scores_normal_base = base_scores(z_node_normal, z_struct_normal, z_phys_normal, z_mahal_normal)
    scores_normal = add_forecast_scores(scores_normal_base, z_forecast_normal, test_normal_mask)

    print("\nscoring fault types ...")
    report = {"config": {"embed_dim": EMBED_DIM, "num_prototypes": NUM_PROTOTYPES, "top_k": TOP_K,
                          "window_len": WINDOW_LEN, "stride": STRIDE, "horizon_mult": horizon_mult,
                          "forecast_h": forecast_h, "epochs": EPOCHS,
                          "prior_edges": EDGES_NAMED, "edge_types": EDGE_TYPES,
                          "use_forecast_prior": use_forecast_prior,
                          "min_proto_samples": MIN_PROTO_SAMPLES, "lambda_forecast": LAMBDA_FORECAST},
              "nodes": cols, "final_prior_bias_strength": model.edge_head.prior_bias_strength.item(),
              "final_typed_temperature": model.typed_head.log_temperature.exp().item(),
              "final_forecast_prior_bias_strength": model.forecast_head.prior_bias_strength.item(),
              "n_valid_prototypes_for_typed_calib": n_valid_proto,
              "n_valid_prototypes_for_cov_calib": n_valid_cov,
              "pair_counts": {"fit": [len(fit_idx), len(fit_in)], "calib": [len(calib_idx), len(calib_in)],
                              "test_normal": [len(test_normal_idx), len(test_normal_in)]},
              "declared_edge_target_node_calib_mse": edge_target_mse,
              "all_node_calib_mse_mean": float(mse_per_node.mean()),
              "fault_types": {}}
    rows = {k: [] for k in scores_normal}
    for name, arr in fault_full.items():
        f_in, f_future = fault_pairs[name]
        _, d_node_f, resid_struct_f, resid_phys_f, _, _, d_mahal_f = per_sample_scores(model, arr)
        k_resid_f = forecast_scores(model, f_in, f_future)
        z_node_f = zscore(d_node_f, d_node_calib)
        z_struct_f = zscore(resid_struct_f, resid_struct_calib)
        z_phys_f = zscore(resid_phys_f, resid_phys_calib)
        z_mahal_f = zscore(d_mahal_f, d_mahal_calib)
        z_forecast_f = zscore(k_resid_f, k_resid_calib)
        scores_fault_base = base_scores(z_node_f, z_struct_f, z_phys_f, z_mahal_f)
        scores_fault = add_forecast_scores(scores_fault_base, z_forecast_f, fault_masks[name])

        aurocs = {}
        for key in scores_normal:
            # K/L use the PAIRED subset (a few orphan windows dropped) on both
            # sides -- still a valid AUROC, just over a very slightly smaller
            # (<1% for normal splits, exact for cable trapped, -1 for stuck)
            # sample than B/C/E/F/H/I/J.
            labels = np.concatenate([np.zeros(len(scores_normal[key])), np.ones(len(scores_fault[key]))])
            scores = np.concatenate([scores_normal[key], scores_fault[key]])
            auroc = float(sk_metrics.roc_auc_score(labels, scores))
            aurocs[key] = auroc
            rows[key].append((name, auroc))
        report["fault_types"][name] = {"n": int(len(arr)), "n_paired": int(len(f_in)), **aurocs}
        print(f"  {name:<16}n={len(arr):<5}" + "  ".join(f"{k}={v:.3f}" for k, v in aurocs.items()))

    print(f"\n{'method':<24}{'mean AUROC (2 faults)':>22}")
    summary = {}
    for key in scores_normal:
        mean_auroc = float(np.mean([r[1] for r in rows[key]]))
        summary[key] = mean_auroc
        print(f"{key:<24}{mean_auroc:>22.3f}")

    print(f"\nfinal prior_bias_strength: {model.edge_head.prior_bias_strength.item():.4f}")
    print(f"final typed-edge attention temperature: {model.typed_head.log_temperature.exp().item():.4f}")
    print(f"final forecast-head prior_bias_strength: {model.forecast_head.prior_bias_strength.item():.4f}")

    report["summary_mean_auroc"] = summary
    suffix = out_suffix if out_suffix is not None else (
        "v2" if (use_forecast_prior and horizon_mult == 1) else
        f"v2_{'prior' if use_forecast_prior else 'noprior'}_h{horizon_mult}"
    )
    report_path = OUT_DIR / f"robo3er_forecast_{suffix}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "report": report},
               OUT_DIR / f"robo3er_forecast_{suffix}.pth")
    print(f"\nsaved -> {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-forecast-prior", action="store_true",
                        help="ablate the forecast head's declared-edge prior bias (forecast_prior_edges=None)")
    parser.add_argument("--horizon-mult", type=int, default=1,
                        help="forecast horizon = horizon_mult * stride, chained over that many future windows")
    parser.add_argument("--out-suffix", type=str, default=None,
                        help="override the output report/checkpoint filename suffix")
    args = parser.parse_args()
    main(use_forecast_prior=not args.no_forecast_prior, horizon_mult=args.horizon_mult,
         out_suffix=args.out_suffix)
