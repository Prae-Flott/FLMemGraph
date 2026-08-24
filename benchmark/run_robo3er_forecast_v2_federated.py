#!/usr/bin/env python3
"""
FEDERATED JointPrototypeV31Forecast (V3.1 + ForecastHead, signal K) on
robo3er's 5 real robots as 5 federated clients. Per
`memory/benchmark-policy-federated-only.md`: the federated variant of
`run_robo3er_forecast_v2.py`'s cross-window pairing route (same pairing
rules -- see that script's docstring), grafted onto
`run_robo3er_v3_1_federated.py`'s federated protocol (per-client local
encoder/edge_head/typed_head/forecast_head, ONLY the `JointPrototypeMemory`
codebook exchanged via `align_and_split`, optional FedAvg'd encoder/decoder
via `SYNC_ENCODER_DECODER`). `forecast_head` is NOT synced even when
`SYNC_ENCODER_DECODER=True` (its own `forecast_head.encoder` doesn't match
the `("encoder.", "decoder.")` FedAvg prefix filter) -- fully local, same
treatment as `edge_head`/`typed_head`.

Same 42-node feature-group-scaled node set, `NUM_PROTOTYPES=2`, per-client
pairing (same robot, boundary-aware) as the centralized v2 script -- see
both `run_robo3er_v3_1_federated.py`'s and `run_robo3er_forecast_v2.py`'s
module docstrings for why each of those choices exists.

Adds `K_forecast_max`/`BK_max`/`CK_max`/`HK_max` (the pairwise
combinations from `memory/forecast-head-signal-k.md`'s combination test)
on top of the existing B/C/E/F/H/I/J ablation columns, using the SAME
`two_stage_group_score` (top-k-mean + re-zscore) aggregation as every
other federated group score here, for consistency.

Usage:
    python3 run_robo3er_forecast_v2_federated.py [--horizon-mult M] [--no-forecast-prior]
        [--out-suffix NAME] [--calib-mode {global,per_prototype,ema,shrinkage}] [--alpha 20]

`--calib-mode` (default `global`, UNCHANGED behavior): see
`run_robo3er_v3_1_federated.py`'s module docstring and
`memory/calib-in-prototype-ab.md` for the full Path A/B/shrinkage
explanation. Applies identically here to signal K (`k_resid`) on top of
B/C/E/H -- `per_prototype` fits `model.score_calib_k` from the calib
split's PAIRED k_resid array (via `set_k_calibration`); `ema`/`shrinkage`
modes have no online/federated hook for K in this script (matches C's
fallback) and stay global for K.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "robo3er"))
sys.path.insert(0, str(REPO_ROOT / "src" / "federated"))
from joint_prototype_model import JointPrototypeV31Forecast  # noqa: E402
from dataset import load_robo3er  # noqa: E402
import feature_groups  # noqa: E402
from federated_memory import (align_and_split, fedavg_state_dict,  # noqa: E402
                               compute_prototype_dev_stats, compute_shrinkage_stats)

SYNC_ENCODER_DECODER = True

DATA_DIR = REPO_ROOT / "data" / "robo3er"
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

WINDOW_LEN = 60
STRIDE = 32
BATCH_SIZE = 256
ROUNDS = 5
LOCAL_EPOCHS = 12
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 2
TOP_K = 8
FEATURE_GROUPS = ["kinematic_core", "actuation"]
EXCLUDE_NODES_PREFIXES = ("tf_link_base_link_", "tf_footprint_base_footprint_",
                           "odom_odo_pos_", "wheel_ticks_")
INCLUDE_EXTRA_NODES = []
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_TYPED = 0.5
LAMBDA_FORECAST = 0.5
MIN_PROTO_SAMPLES = 5
TOP_K_AGG = 3
GAMMA = 0.5
DELTA = 0.5
FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Client:
    def __init__(self, client_id, robot_name, fit_idx, calib_idx, test_normal_idx, fault_idx_by_label):
        self.client_id = client_id
        self.robot_name = robot_name
        self.fit_idx = fit_idx
        self.calib_idx = calib_idx
        self.test_normal_idx = test_normal_idx
        self.fault_idx_by_label = fault_idx_by_label


def build_clients():
    data_full, targets, cols_full, label_map, robot_map = load_robo3er(drop_dead=True)
    selected = set(feature_groups.active_feature_names(FEATURE_GROUPS))
    selected -= {c for c in cols_full if c.startswith(EXCLUDE_NODES_PREFIXES)}
    selected |= set(INCLUDE_EXTRA_NODES)
    keep_idx = [i for i, c in enumerate(cols_full) if c in selected]
    data, cols = data_full[:, :, keep_idx], [cols_full[i] for i in keep_idx]
    data = data.astype(np.float32)

    with open(DATA_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    client_indices = partition["data_indices"]

    window_robot = np.full(len(targets), -1, dtype=int)
    clients = []
    for client_id, idx_dict in enumerate(client_indices):
        all_idx = np.array(idx_dict["train"] + idx_dict["val"] + idx_dict["test"], dtype=int)
        window_robot[all_idx] = client_id
        normal_idx = np.sort(all_idx[targets[all_idx] == 0])
        n = len(normal_idx)
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        fit_idx = normal_idx[:n_fit]
        calib_idx = normal_idx[n_fit : n_fit + n_calib]
        test_normal_idx = normal_idx[n_fit + n_calib :]

        fault_idx_by_label = {}
        for label_id in sorted(set(targets[all_idx].tolist())):
            if label_id == 0:
                continue
            fault_idx = all_idx[targets[all_idx] == label_id]
            if len(fault_idx) > 0:
                fault_idx_by_label[label_id] = fault_idx

        clients.append(Client(
            client_id=client_id,
            robot_name=robot_map.get(str(client_id), f"robot{client_id:02d}"),
            fit_idx=fit_idx, calib_idx=calib_idx, test_normal_idx=test_normal_idx,
            fault_idx_by_label=fault_idx_by_label,
        ))
    return data, targets, label_map, clients, cols, window_robot


def build_pairs(idx_arr, targets, window_robot, horizon_mult, require_next_normal, require_same_split=False):
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
    if len(chains) == 0:
        return np.zeros((0, 0, data_scaled.shape[-1]), dtype=data_scaled.dtype)
    segments = [data_scaled[[c[k] for c in chains]][:, -stride:, :] for k in range(len(chains[0]))]
    return np.concatenate(segments, axis=1)


def scale_client(data, scaler, idx):
    n, t, f = data[idx].shape
    return scaler.transform(data[idx].reshape(-1, f)).reshape(n, t, f).astype(np.float32)


def fit_client_scaler(data, fit_idx):
    scaler = StandardScaler()
    scaler.fit(data[fit_idx].reshape(-1, data.shape[-1]))
    return scaler


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


def train_local(model, fit_in, fit_future, epochs, calib_mode="global"):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fit_in), torch.from_numpy(fit_future)),
        batch_size=min(BATCH_SIZE, len(fit_in)), shuffle=True,
    )
    model.train()
    for _ in range(epochs):
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
            if calib_mode == "ema":
                model.memory.update_ema(out["d_node"].detach(), out["idx"].detach())
    return model


def zscore(x, calib_x):
    median = np.median(calib_x, axis=0)
    q75, q25 = np.percentile(calib_x, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    return (x - median) / iqr


def topk_mean(x, k):
    k = min(k, x.shape[1])
    return np.sort(x, axis=1)[:, -k:].mean(axis=1)


def two_stage_group_score(raw_calib, raw_x, k=TOP_K_AGG, score_head=None, idx_calib=None, idx_x=None,
                          ema_memory=None):
    if score_head is not None:
        z_calib = score_head.zscore(raw_calib, idx_calib)
        z_x = score_head.zscore(raw_x, idx_x)
    elif ema_memory is not None:
        z_calib = ema_memory.ema_zscore(raw_calib, idx_calib)
        z_x = ema_memory.ema_zscore(raw_x, idx_x)
    else:
        z_calib = zscore(raw_calib, raw_calib)
        z_x = zscore(raw_x, raw_calib)
    stat_calib = topk_mean(z_calib, k)
    stat_x = topk_mean(z_x, k)
    median = np.median(stat_calib)
    iqr = max(np.percentile(stat_calib, 75) - np.percentile(stat_calib, 25), 1e-8)
    return (stat_x - median) / iqr


def base_scores(node_score, edge_score, typed_score, cov_score):
    b = node_score
    return {
        "B_node_max": b,
        "C_struct_max": edge_score,
        "E_phys_max": typed_score,
        "F_v3_max": np.max(np.stack([node_score, edge_score, typed_score], axis=1), axis=1),
        "H_cov_mahal": cov_score,
        "I_node_cov_max": np.maximum(b, cov_score),
        "J_v3_cov_max": np.max(np.stack([node_score, edge_score, typed_score, cov_score], axis=1), axis=1),
    }


def add_forecast_scores(base, forecast_score, mask):
    """B/C/E/F/H/I/J stay FULL-length (unmasked) in the output -- only
    K/BK/CK/HK use the forecast-pairing-masked subset (a few orphan
    windows without a valid horizon_mult-chain dropped), matching the
    centralized `run_robo3er_forecast_v2.py`'s convention. `base` here is
    the FULL, unmasked score dict; `mask` (over the same index order) is
    applied to LOCAL copies only, for combining with `forecast_score`
    (already at the masked/paired length) -- never written back onto
    the full-length B/C/E/F/H/I/J entries themselves."""
    b, c, h = base["B_node_max"][mask], base["C_struct_max"][mask], base["H_cov_mahal"][mask]
    out = dict(base)
    out["K_forecast_max"] = forecast_score
    out["BK_max"] = np.maximum(b, forecast_score)
    out["CK_max"] = np.maximum(c, forecast_score)
    out["HK_max"] = np.maximum(h, forecast_score)
    return out


def main(horizon_mult=10, use_forecast_prior=True, out_suffix=None, calib_mode="global", alpha=20.0):
    assert calib_mode in ("global", "per_prototype", "ema", "shrinkage")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    forecast_h = STRIDE * horizon_mult

    print(f"loading robo3er, splitting into 5 real per-robot federated clients, "
          f"{FEATURE_GROUPS} as nodes, forecast head horizon_mult={horizon_mult} ...")
    data, targets, label_map, clients, cols, window_robot = build_clients()
    num_nodes = len(cols)
    node_idx = {name: i for i, name in enumerate(cols)}
    prior_edges = [(node_idx[s], node_idx[d]) for s, d in EDGES_NAMED]
    forecast_prior_edges = prior_edges if use_forecast_prior else None
    for c in clients:
        print(f"  client {c.client_id} ({c.robot_name}): fit={len(c.fit_idx)} calib={len(c.calib_idx)} "
              f"test_normal={len(c.test_normal_idx)} faults={ {label_map[str(k)]: len(v) for k, v in c.fault_idx_by_label.items()} }")

    scalers = [fit_client_scaler(data, c.fit_idx) for c in clients]
    models = [JointPrototypeV31Forecast(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=EMBED_DIM,
                                         num_prototypes=NUM_PROTOTYPES, prior_edges=prior_edges,
                                         edge_types=EDGE_TYPES, forecast_h=forecast_h, top_k=TOP_K,
                                         forecast_prior_edges=forecast_prior_edges).to(DEVICE)
              for _ in clients]

    shared_init = {k: v.clone() for k, v in models[0].state_dict().items() if not k.startswith("memory.")}
    for model in models[1:]:
        model.load_state_dict(shared_init, strict=False)

    # Pre-build each client's fit-split pairs once (pairing doesn't depend on the model).
    fit_pairs = []
    for c in clients:
        vi, chains, _ = build_pairs(c.fit_idx, targets, window_robot, horizon_mult, True, True)
        x_in = data[vi]
        x_future = gather_future(data, chains, STRIDE)
        fit_pairs.append((vi, x_in, x_future))

    print(f"\nfederated training: {ROUNDS} rounds x {LOCAL_EPOCHS} local epochs, "
          f"memory exchange (gamma={GAMMA}, delta={DELTA})"
          + (", + FedAvg'd encoder/decoder" if SYNC_ENCODER_DECODER else " only"))
    diagnostics_log = []
    for rnd in range(1, ROUNDS + 1):
        fit_counts = []
        for c, model, scaler, (vi, x_in_raw, x_future_raw) in zip(clients, models, scalers, fit_pairs):
            n, t, f = x_in_raw.shape
            x_in = scaler.transform(x_in_raw.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
            n2, t2, f2 = x_future_raw.shape
            x_future = (scaler.transform(x_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                        if n2 > 0 else x_future_raw)
            train_local(model, x_in, x_future, LOCAL_EPOCHS, calib_mode=calib_mode)
            fit_counts.append(len(x_in))

        codebooks = [m.memory.codebook.detach().clone() for m in models]
        usage_counts = [m.memory.usage_count.detach().clone() for m in models]
        P_G, diag = align_and_split(codebooks, usage_counts, gamma=GAMMA, delta=DELTA)
        diag = {k: v for k, v in diag.items() if k != "per_cluster_per_node_agreement"}
        diag["round"] = rnd
        diagnostics_log.append(diag)
        print(f"  round {rnd}: {diag}")

        for model, p_g in zip(models, P_G):
            model.memory.load_memory(p_g)

        if calib_mode == "shrinkage":
            # one-shot local pass over each client's OWN fit pairs, using this
            # round's just-received aligned codebook -- see
            # federated_memory.compute_shrinkage_stats' docstring for why
            # this must happen AFTER load_memory().
            client_dev_stats = []
            for c, model, scaler, (vi, x_in_raw, x_future_raw) in zip(clients, models, scalers, fit_pairs):
                n, t, f = x_in_raw.shape
                x_in = scaler.transform(x_in_raw.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
                _, d_node_fit, _, _, idx_fit, _, _ = per_sample_scores(model, x_in)
                client_dev_stats.append(compute_prototype_dev_stats(d_node_fit, idx_fit, NUM_PROTOTYPES))
            num_shared = diag["num_shared_prototypes"]
            shrink_stats, alpha_used = compute_shrinkage_stats(client_dev_stats, num_shared, alpha)
            diag["shrinkage_alpha"] = alpha_used.tolist()
            for model, (mean_s, var_s, valid_s) in zip(models, shrink_stats):
                model.memory.load_shrinkage_stats(mean_s, var_s, valid_s)

        if SYNC_ENCODER_DECODER:
            avg = fedavg_state_dict([m.state_dict() for m in models], fit_counts,
                                     prefixes=("encoder.", "decoder."))
            for model in models:
                model.load_state_dict(avg, strict=False)

    report = {"config": {"rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "num_prototypes": NUM_PROTOTYPES,
                          "embed_dim": EMBED_DIM, "top_k": TOP_K, "gamma": GAMMA, "delta": DELTA,
                          "prior_edges": EDGES_NAMED, "edge_types": EDGE_TYPES,
                          "min_proto_samples": MIN_PROTO_SAMPLES, "top_k_agg": TOP_K_AGG,
                          "sync_encoder_decoder": SYNC_ENCODER_DECODER, "horizon_mult": horizon_mult,
                          "forecast_h": forecast_h, "use_forecast_prior": use_forecast_prior,
                          "feature_groups": FEATURE_GROUPS, "num_nodes": num_nodes},
              "alignment_log": diagnostics_log, "clients": {}}

    calib_data = {}
    cov_min = max(MIN_PROTO_SAMPLES, num_nodes + 1)
    for c, model, scaler in zip(clients, models, scalers):
        print(f"\nStage B ({c.robot_name}): typed-head + covariance-head calibration ...")
        calib_w = scale_client(data, scaler, c.calib_idx)
        _, d_node_calib_b, _, _, calib_idx_w, calib_r_w, _ = per_sample_scores(model, calib_w)
        model.typed_head.set_calibration(calib_r_w, calib_idx_w, NUM_PROTOTYPES, min_samples=MIN_PROTO_SAMPLES)
        model.cov_head.set_calibration(d_node_calib_b, calib_idx_w, min_samples=cov_min)

        _, d_node_calib, resid_struct_calib, resid_phys_calib, calib_idx_w, _, d_mahal_calib = per_sample_scores(model, calib_w)
        if calib_mode == "per_prototype":
            model.set_score_calibration(d_node_calib, resid_struct_calib, calib_idx_w,
                                        min_samples=MIN_PROTO_SAMPLES)

        vi, chains, _ = build_pairs(c.calib_idx, targets, window_robot, horizon_mult, True, True)
        calib_in = scale_client(data, scaler, vi) if len(vi) else np.zeros((0, WINDOW_LEN, num_nodes), dtype=np.float32)
        calib_future_raw = gather_future(data, chains, STRIDE)
        if len(vi):
            n2, t2, f2 = calib_future_raw.shape
            calib_future = scaler.transform(calib_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
        else:
            calib_future = calib_future_raw
        k_resid_calib = forecast_scores(model, calib_in, calib_future) if len(vi) else np.zeros((0, num_nodes), dtype=np.float32)
        if len(vi):
            _, _, _, _, k_idx_calib, _, _ = per_sample_scores(model, calib_in)
        else:
            k_idx_calib = np.zeros((0,), dtype=int)
        if calib_mode == "per_prototype" and len(vi):
            model.set_k_calibration(k_resid_calib, k_idx_calib, min_samples=MIN_PROTO_SAMPLES)

        calib_data[c.client_id] = (d_node_calib, resid_struct_calib, resid_phys_calib, d_mahal_calib,
                                    k_resid_calib, calib_idx_w, k_idx_calib)

    all_pairs = {k: [] for k in ["B_node_max", "C_struct_max", "E_phys_max", "F_v3_max",
                                   "H_cov_mahal", "I_node_cov_max", "J_v3_cov_max",
                                   "K_forecast_max", "BK_max", "CK_max", "HK_max"]}
    for c, model, scaler in zip(clients, models, scalers):
        (d_node_calib, resid_struct_calib, resid_phys_calib, d_mahal_calib,
         k_resid_calib, calib_idx_w, k_idx_calib) = calib_data[c.client_id]
        report["clients"].setdefault(c.robot_name, {})

        if not c.fault_idx_by_label:
            print(f"client {c.client_id} ({c.robot_name}): no fault windows, skipping evaluation")
            continue

        node_head = model.score_calib_node if calib_mode == "per_prototype" else None
        struct_head = model.score_calib_struct if calib_mode == "per_prototype" else None
        k_head = model.score_calib_k if calib_mode == "per_prototype" else None
        ema_mem = model.memory if calib_mode in ("ema", "shrinkage") else None  # C/K have no
        # online/federated hook -> global fallback; ema_zscore() reused as-is for shrinkage.

        test_w = scale_client(data, scaler, c.test_normal_idx)
        _, d_node_normal, resid_struct_normal, resid_phys_normal, idx_normal, _, d_mahal_normal = per_sample_scores(model, test_w)
        node_score_normal = two_stage_group_score(d_node_calib, d_node_normal, score_head=node_head,
                                                   idx_calib=calib_idx_w, idx_x=idx_normal, ema_memory=ema_mem)
        edge_score_normal = two_stage_group_score(resid_struct_calib, resid_struct_normal, score_head=struct_head,
                                                   idx_calib=calib_idx_w, idx_x=idx_normal)
        typed_score_normal = two_stage_group_score(resid_phys_calib, resid_phys_normal)
        cov_score_normal = zscore(d_mahal_normal, d_mahal_calib)
        base_normal = base_scores(node_score_normal, edge_score_normal, typed_score_normal, cov_score_normal)

        vi_n, chains_n, mask_n = build_pairs(c.test_normal_idx, targets, window_robot, horizon_mult, True, True)
        test_normal_in = scale_client(data, scaler, vi_n) if len(vi_n) else np.zeros((0, WINDOW_LEN, num_nodes), dtype=np.float32)
        future_n_raw = gather_future(data, chains_n, STRIDE)
        if len(vi_n):
            n2, t2, f2 = future_n_raw.shape
            test_normal_future = scaler.transform(future_n_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
            k_resid_normal = forecast_scores(model, test_normal_in, test_normal_future)
            if len(k_resid_calib):
                _, _, _, _, k_idx_normal, _, _ = per_sample_scores(model, test_normal_in)
                forecast_score_normal = two_stage_group_score(k_resid_calib, k_resid_normal, score_head=k_head,
                                                               idx_calib=k_idx_calib, idx_x=k_idx_normal)
            else:
                forecast_score_normal = np.zeros(len(vi_n))
        else:
            forecast_score_normal = np.zeros(0)
        scores_normal = add_forecast_scores(base_normal, forecast_score_normal, mask_n)

        client_report = {"robot_name": c.robot_name, "fault_types": {}}
        rows = {k: [] for k in scores_normal}
        for label_id, fault_idx in c.fault_idx_by_label.items():
            name = label_map[str(label_id)]
            fault_w = scale_client(data, scaler, fault_idx)
            _, d_node_f, resid_struct_f, resid_phys_f, idx_f, _, d_mahal_f = per_sample_scores(model, fault_w)
            node_score_f = two_stage_group_score(d_node_calib, d_node_f, score_head=node_head,
                                                  idx_calib=calib_idx_w, idx_x=idx_f, ema_memory=ema_mem)
            edge_score_f = two_stage_group_score(resid_struct_calib, resid_struct_f, score_head=struct_head,
                                                  idx_calib=calib_idx_w, idx_x=idx_f)
            typed_score_f = two_stage_group_score(resid_phys_calib, resid_phys_f)
            cov_score_f = zscore(d_mahal_f, d_mahal_calib)
            base_f = base_scores(node_score_f, edge_score_f, typed_score_f, cov_score_f)

            vi_f, chains_f, mask_f = build_pairs(fault_idx, targets, window_robot, horizon_mult, False, False)
            if len(vi_f):
                fault_in = scale_client(data, scaler, vi_f)
                future_f_raw = gather_future(data, chains_f, STRIDE)
                n2, t2, f2 = future_f_raw.shape
                fault_future = scaler.transform(future_f_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                k_resid_f = forecast_scores(model, fault_in, fault_future)
                if len(k_resid_calib):
                    _, _, _, _, k_idx_f, _, _ = per_sample_scores(model, fault_in)
                    forecast_score_f = two_stage_group_score(k_resid_calib, k_resid_f, score_head=k_head,
                                                             idx_calib=k_idx_calib, idx_x=k_idx_f)
                else:
                    forecast_score_f = np.zeros(len(vi_f))
            else:
                forecast_score_f = np.zeros(0)
            scores_fault = add_forecast_scores(base_f, forecast_score_f, mask_f)

            aurocs = {}
            for key in scores_normal:
                if len(scores_normal[key]) == 0 or len(scores_fault[key]) == 0:
                    continue
                labels = np.concatenate([np.zeros(len(scores_normal[key])), np.ones(len(scores_fault[key]))])
                s = np.concatenate([scores_normal[key], scores_fault[key]])
                auroc = float(sk_metrics.roc_auc_score(labels, s))
                aurocs[key] = auroc
                rows[key].append(auroc)
                all_pairs[key].append(auroc)
            client_report["fault_types"][name] = {"n": int(len(fault_idx)), **aurocs}
            print(f"  {c.robot_name:<10}{name:<16}n={len(fault_idx):<5}"
                  + "  ".join(f"{k}={v:.3f}" for k, v in aurocs.items()))

        client_report["client_mean_auroc"] = {k: float(np.mean(v)) for k, v in rows.items() if v}
        report["clients"][c.robot_name] = client_report

    summary_overall = {k: float(np.mean(v)) for k, v in all_pairs.items() if v}
    print(f"\n{'method':<24}{'mean AUROC (all client-fault pairs)':>36}")
    for key, v in summary_overall.items():
        print(f"{key:<24}{v:>36.3f}")

    report["config"]["calib_mode"] = calib_mode
    if calib_mode == "shrinkage":
        report["config"]["alpha"] = "auto" if alpha is None else alpha
    report["summary_mean_auroc_overall"] = summary_overall
    if calib_mode == "shrinkage":
        mode_suffix = f"_calibmode_shrinkage_alpha{'auto' if alpha is None else int(alpha)}"
    elif calib_mode != "global":
        mode_suffix = f"_calibmode_{calib_mode}"
    else:
        mode_suffix = ""
    suffix = out_suffix if out_suffix is not None else (
        f"forecast_v2_federated_h{horizon_mult}{'_noprior' if not use_forecast_prior else ''}"
        + ("_encdec_synced" if SYNC_ENCODER_DECODER else "") + mode_suffix)
    out_json = OUT_DIR / f"robo3er_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
               OUT_DIR / f"robo3er_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-mult", type=int, default=10)
    parser.add_argument("--no-forecast-prior", action="store_true")
    parser.add_argument("--out-suffix", type=str, default=None)
    parser.add_argument("--calib-mode", choices=["global", "per_prototype", "ema", "shrinkage"], default="global")
    parser.add_argument("--alpha", type=str, default="20",
                        help="shrinkage strength for --calib-mode shrinkage, or 'auto' for data-driven per-slot alpha")
    args = parser.parse_args()
    args.alpha = None if args.alpha == "auto" else float(args.alpha)
    main(horizon_mult=args.horizon_mult, use_forecast_prior=not args.no_forecast_prior,
         out_suffix=args.out_suffix, calib_mode=args.calib_mode, alpha=args.alpha)
