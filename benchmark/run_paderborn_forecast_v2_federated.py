#!/usr/bin/env python3
"""
FEDERATED JointPrototypeV31Forecast (V3.1 + ForecastHead, signal K) on
Paderborn, treating K001-K006 as 6 federated clients. Per
`memory/benchmark-policy-federated-only.md`: the federated variant of
`run_paderborn_forecast_v2.py`'s per-file cross-window pairing route
(same pairing rules -- see that script's docstring), grafted onto
`run_paderborn_v3_1_federated.py`'s federated protocol (per-client local
encoder/edge_head/typed_head/forecast_head, ONLY the memory codebook
exchanged, optional FedAvg'd encoder/decoder). `forecast_head` stays
local even when `SYNC_ENCODER_DECODER=True`, same as `edge_head`/
`typed_head` (see `run_robo3er_forecast_v2_federated.py`'s docstring for
why the FedAvg prefix filter doesn't touch it).

Adds `K_forecast_max`/`BK_max`/`CK_max`/`HK_max` on top of B/C/E/F/H/I/J.

**WINDOW-level scoring + `--calib-mode`** (ported from
`run_paderborn_v3_1_federated.py`, same motivation and same non-
regression-safe caveat -- see that script's module docstring): uses the
`two_stage_group_score` (top-k-mean + re-zscore) aggregation, matching
`run_robo3er_forecast_v2_federated.py`'s own convention, instead of the
original file-level-aggregate + plain `.max(axis=1)` design. `shrinkage`
has no hook for K (same scope limit as C) and stays `global` for it.

Usage:
    python3 run_paderborn_forecast_v2_federated.py [--horizon-mult M] [--no-forecast-prior]
        [--out-suffix NAME] [--calib-mode {global,shrinkage}] [--alpha 20|auto]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "federated"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_prototype_model import JointPrototypeV31Forecast  # noqa: E402
from federated_memory import (align_and_split, confidence_weights_from_losses, fedavg_state_dict,  # noqa: E402
                               compute_prototype_dev_stats, compute_shrinkage_stats)

SYNC_ENCODER_DECODER = True
from datasets.paderborn_adapter import (  # noqa: E402
    load_bearing_with_physics, HEALTHY_CODES, ALL_DAMAGED_CODES, category_of, damage_origin_of,
)

OUT_DIR = REPO_ROOT / "checkpoints" / "paderborn"

NODE_NAMES = ["vibration_1", "phase_current_1", "phase_current_2", "force", "speed", "torque"]
NODE_IDX = {name: i for i, name in enumerate(NODE_NAMES)}
EDGES_NAMED = [
    ("force", "vibration_1"), ("force", "torque"),
    ("speed", "vibration_1"), ("speed", "torque"),
    ("speed", "phase_current_1"), ("speed", "phase_current_2"),
    ("torque", "phase_current_1"), ("torque", "phase_current_2"),
]
EDGE_TYPES = [
    "nonlinear", "nonlinear", "nonlinear", "nonlinear",
    "proportional", "proportional", "proportional", "proportional",
]
PRIOR_EDGES = [(NODE_IDX[s], NODE_IDX[d]) for s, d in EDGES_NAMED]

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15
WINDOW_LEN = 64
WINDOW_STRIDE = 32
BATCH_SIZE = 256
ROUNDS = 5
LOCAL_EPOCHS = 12
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 16
TOP_K = 5
TOP_K_AGG = 3  # see run_paderborn_v3_1_federated.py's two_stage_group_score
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_TYPED = 0.5
LAMBDA_FORECAST = 0.5
MIN_PROTO_SAMPLES = 5
GAMMA = 0.5
DELTA = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stack_nodes(d):
    return np.stack([d[name] for name in NODE_NAMES], axis=-1).astype(np.float32)


def make_windows(arr, window_len=WINDOW_LEN, stride=WINDOW_STRIDE):
    n, t, f = arr.shape
    positions = list(range(0, t - window_len + 1, stride))
    windows = np.empty((n, len(positions), window_len, f), dtype=np.float32)
    for j, pos in enumerate(positions):
        windows[:, j] = arr[:, pos : pos + window_len]
    file_id = np.repeat(np.arange(n), len(positions))
    return windows.reshape(-1, window_len, f), file_id, len(positions)


def build_chains(n_files, n_positions, horizon_mult):
    vi, chains = [], []
    for file_idx in range(n_files):
        base = file_idx * n_positions
        for local_pos in range(n_positions - horizon_mult):
            gi = base + local_pos
            vi.append(gi)
            chains.append([gi + k for k in range(1, horizon_mult + 1)])
    return np.array(vi, dtype=int), chains


def gather_future(windows, chains, stride):
    if len(chains) == 0:
        return np.zeros((0, 0, windows.shape[-1]), dtype=windows.dtype)
    segments = [windows[[c[k] for c in chains]][:, -stride:, :] for k in range(len(chains[0]))]
    return np.concatenate(segments, axis=1)


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


def train_local(model, fit_windows, fit_future, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fit_windows), torch.from_numpy(fit_future)),
        batch_size=min(BATCH_SIZE, len(fit_windows)), shuffle=True,
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
    return model


def zscore(x, calib_x):
    median = np.median(calib_x, axis=0)
    q75, q25 = np.percentile(calib_x, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    return (x - median) / iqr


def topk_mean(x, k):
    k = min(k, x.shape[1])
    return np.sort(x, axis=1)[:, -k:].mean(axis=1)


def two_stage_group_score(raw_calib, raw_x, k=TOP_K_AGG, idx_calib=None, idx_x=None,
                          dev_memory=None):
    if dev_memory is not None:
        z_calib = dev_memory.dev_zscore(raw_calib, idx_calib)
        z_x = dev_memory.dev_zscore(raw_x, idx_x)
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
    """B/C/E/F/H/I/J stay full-length; K/BK/CK/HK use the forecast-pairing-
    masked subset -- same convention as run_robo3er_forecast_v2_federated.py."""
    b, c, h = base["B_node_max"][mask], base["C_struct_max"][mask], base["H_cov_mahal"][mask]
    out = dict(base)
    out["K_forecast_max"] = forecast_score
    out["BK_max"] = np.maximum(b, forecast_score)
    out["CK_max"] = np.maximum(c, forecast_score)
    out["HK_max"] = np.maximum(h, forecast_score)
    return out


def main(horizon_mult=10, use_forecast_prior=True, out_suffix=None, calib_mode="global", alpha=20.0):
    assert calib_mode in ("global", "shrinkage")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    num_nodes = len(NODE_NAMES)
    forecast_h = WINDOW_STRIDE * horizon_mult
    forecast_prior_edges = PRIOR_EDGES if use_forecast_prior else None

    print("loading healthy bearings (K001-K006) as 6 federated clients ...")
    clients = []
    for code in HEALTHY_CODES:
        d = load_bearing_with_physics(code)
        arr = stack_nodes(d)
        n = len(arr)
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        fit_raw, calib_raw, test_raw = arr[:n_fit], arr[n_fit : n_fit + n_calib], arr[n_fit + n_calib :]
        scaler = StandardScaler().fit(fit_raw.reshape(-1, num_nodes))

        def scale(a, scaler=scaler):
            n_, t_, f_ = a.shape
            return scaler.transform(a.reshape(-1, f_)).reshape(n_, t_, f_).astype(np.float32)

        clients.append({"code": code, "fit": scale(fit_raw), "calib": scale(calib_raw),
                          "test_normal": scale(test_raw), "scaler": scaler})
        print(f"  {code}: n={n} fit={n_fit} calib={n_calib} test_normal={n - n_fit - n_calib}")

    models = [JointPrototypeV31Forecast(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=EMBED_DIM,
                                         num_prototypes=NUM_PROTOTYPES, prior_edges=PRIOR_EDGES,
                                         edge_types=EDGE_TYPES, forecast_h=forecast_h, top_k=TOP_K,
                                         forecast_prior_edges=forecast_prior_edges).to(DEVICE)
              for _ in clients]

    shared_init = {k: v.clone() for k, v in models[0].state_dict().items() if not k.startswith("memory.")}
    for model in models[1:]:
        model.load_state_dict(shared_init, strict=False)

    print(f"\nfederated training: {ROUNDS} rounds x {LOCAL_EPOCHS} local epochs, "
          f"memory exchange (gamma={GAMMA}, delta={DELTA}), horizon_mult={horizon_mult}"
          + (", + FedAvg'd encoder/decoder" if SYNC_ENCODER_DECODER else " only"))
    diagnostics_log = []
    for rnd in range(1, ROUNDS + 1):
        fit_counts = []
        for c, model in zip(clients, models):
            fit_windows, fit_file_id, fit_n_pos = make_windows(c["fit"])
            fit_vi, fit_chains = build_chains(len(c["fit"]), fit_n_pos, horizon_mult)
            fit_in = fit_windows[fit_vi]
            fit_future = gather_future(fit_windows, fit_chains, WINDOW_STRIDE)
            train_local(model, fit_in, fit_future, LOCAL_EPOCHS)
            fit_counts.append(len(fit_in))

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
            # one-shot local pass over each client's OWN fit windows (all
            # windows, not just the forecast-paired subset -- matches
            # run_paderborn_v3_1_federated.py's placement, AFTER load_memory()).
            client_dev_stats = []
            for c, model in zip(clients, models):
                fit_windows, _, _ = make_windows(c["fit"])
                _, d_node_fit, _, _, idx_fit, _, _ = per_sample_scores(model, fit_windows)
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

    print("\nStage B + calibration per client ...")
    calib_data = {}  # code -> (d_node_calib, resid_struct_calib, resid_phys_calib, d_mahal_calib,
                      #          k_resid_calib, calib_idx_w, k_idx_calib)
    component_losses = {"proto": [], "edge": [], "typed": []}
    cov_min = max(MIN_PROTO_SAMPLES, num_nodes + 1)
    n_valid_cov_per_client = []
    for c, model in zip(clients, models):
        calib_windows, _, calib_n_pos = make_windows(c["calib"])
        d_proto_w, d_node_calib, resid_struct_calib, resid_phys_calib, calib_idx_w, calib_r_edge_w, d_mahal_calib = \
            per_sample_scores(model, calib_windows)
        model.typed_head.set_calibration(calib_r_edge_w, calib_idx_w, NUM_PROTOTYPES, min_samples=MIN_PROTO_SAMPLES)
        model.cov_head.set_calibration(d_node_calib, calib_idx_w, min_samples=cov_min)

        calib_vi, calib_chains = build_chains(len(c["calib"]), calib_n_pos, horizon_mult)
        if len(calib_vi):
            calib_in = calib_windows[calib_vi]
            calib_future = gather_future(calib_windows, calib_chains, WINDOW_STRIDE)
            k_resid_calib = forecast_scores(model, calib_in, calib_future)
            _, _, _, _, k_idx_calib, _, _ = per_sample_scores(model, calib_in)
        else:
            k_resid_calib = np.zeros((0, num_nodes), dtype=np.float32)
            k_idx_calib = np.zeros((0,), dtype=int)
        calib_data[c["code"]] = (d_node_calib, resid_struct_calib, resid_phys_calib, d_mahal_calib,
                                  k_resid_calib, calib_idx_w, k_idx_calib)
        n_valid_cov_per_client.append(int(model.cov_head.calib_valid.sum()))
        component_losses["proto"].append(float(d_proto_w.mean()))
        component_losses["edge"].append(float(resid_struct_calib.mean()))
        component_losses["typed"].append(float(calib_r_edge_w.mean()))

    client_weights = confidence_weights_from_losses(component_losses, temperature=1.0)
    for c, w in zip(clients, client_weights):
        print(f"  {c['code']} confidence weights: proto={w['proto']:.3f} edge={w['edge']:.3f} typed={w['typed']:.3f}")

    print("\nscoring damaged bearings against every client's personalized model ...")
    report = {"config": {"embed_dim": EMBED_DIM, "num_prototypes": NUM_PROTOTYPES, "top_k": TOP_K,
                          "window_len": WINDOW_LEN, "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS,
                          "gamma": GAMMA, "delta": DELTA, "prior_edges": EDGES_NAMED, "edge_types": EDGE_TYPES,
                          "min_proto_samples": MIN_PROTO_SAMPLES, "clients": HEALTHY_CODES,
                          "sync_encoder_decoder": SYNC_ENCODER_DECODER, "horizon_mult": horizon_mult,
                          "forecast_h": forecast_h, "use_forecast_prior": use_forecast_prior,
                          "top_k_agg": TOP_K_AGG, "calib_mode": calib_mode},
              "nodes": NODE_NAMES, "alignment_log": diagnostics_log,
              "client_confidence_weights": {c["code"]: w for c, w in zip(clients, client_weights)},
              "bearings": {}}
    if calib_mode == "shrinkage":
        report["config"]["alpha"] = "auto" if alpha is None else alpha

    normal_scores_per_client = []
    for c, model in zip(clients, models):
        (d_node_calib, resid_struct_calib, resid_phys_calib, d_mahal_calib,
         k_resid_calib, calib_idx_w, k_idx_calib) = calib_data[c["code"]]
        dev_mem = model.memory if calib_mode == "shrinkage" else None  # C/K have no
        # shrinkage hook -> global fallback; dev_zscore() reads whatever
        # load_shrinkage_stats() populated this round.

        test_normal_windows, _, test_n_pos = make_windows(c["test_normal"])
        _, d_node_n, resid_struct_n, resid_phys_n, idx_n, _, d_mahal_n = per_sample_scores(model, test_normal_windows)
        node_score_n = two_stage_group_score(d_node_calib, d_node_n,
                                              idx_calib=calib_idx_w, idx_x=idx_n, dev_memory=dev_mem)
        edge_score_n = two_stage_group_score(resid_struct_calib, resid_struct_n,
                                              idx_calib=calib_idx_w, idx_x=idx_n)
        typed_score_n = two_stage_group_score(resid_phys_calib, resid_phys_n)
        cov_score_n = zscore(d_mahal_n, d_mahal_calib)
        base_n = base_scores(node_score_n, edge_score_n, typed_score_n, cov_score_n)

        vi_n, chains_n = build_chains(len(c["test_normal"]), test_n_pos, horizon_mult)
        mask_n = np.zeros(len(test_normal_windows), dtype=bool)
        mask_n[vi_n] = True
        if len(vi_n):
            in_n = test_normal_windows[vi_n]
            future_n = gather_future(test_normal_windows, chains_n, WINDOW_STRIDE)
            k_resid_n = forecast_scores(model, in_n, future_n)
            if len(k_resid_calib):
                _, _, _, _, k_idx_n, _, _ = per_sample_scores(model, in_n)
                forecast_score_n = two_stage_group_score(k_resid_calib, k_resid_n,
                                                          idx_calib=k_idx_calib, idx_x=k_idx_n)
            else:
                forecast_score_n = np.zeros(len(vi_n))
        else:
            forecast_score_n = np.zeros(0)
        normal_scores_per_client.append(add_forecast_scores(base_n, forecast_score_n, mask_n))

    rows = {k: [] for k in normal_scores_per_client[0]}
    for code in ALL_DAMAGED_CODES:
        d = load_bearing_with_physics(code)
        arr = stack_nodes(d)
        cat, origin = category_of(code), damage_origin_of(code)

        per_client_auroc = {k: [] for k in rows}
        for c, model, scores_normal in zip(clients, models, normal_scores_per_client):
            (d_node_calib, resid_struct_calib, resid_phys_calib, d_mahal_calib,
             k_resid_calib, calib_idx_w, k_idx_calib) = calib_data[c["code"]]
            dev_mem = model.memory if calib_mode == "shrinkage" else None

            arr_scaled = c["scaler"].transform(arr.reshape(-1, num_nodes)).reshape(arr.shape).astype(np.float32)
            fault_windows, _, fault_n_pos = make_windows(arr_scaled)
            _, d_node_f, resid_struct_f, resid_phys_f, idx_f, _, d_mahal_f = per_sample_scores(model, fault_windows)
            node_score_f = two_stage_group_score(d_node_calib, d_node_f,
                                                  idx_calib=calib_idx_w, idx_x=idx_f, dev_memory=dev_mem)
            edge_score_f = two_stage_group_score(resid_struct_calib, resid_struct_f,
                                                  idx_calib=calib_idx_w, idx_x=idx_f)
            typed_score_f = two_stage_group_score(resid_phys_calib, resid_phys_f)
            cov_score_f = zscore(d_mahal_f, d_mahal_calib)
            base_f = base_scores(node_score_f, edge_score_f, typed_score_f, cov_score_f)

            vi_f, chains_f = build_chains(len(arr_scaled), fault_n_pos, horizon_mult)
            mask_f = np.zeros(len(fault_windows), dtype=bool)
            mask_f[vi_f] = True
            if len(vi_f):
                in_f = fault_windows[vi_f]
                future_f = gather_future(fault_windows, chains_f, WINDOW_STRIDE)
                k_resid_f = forecast_scores(model, in_f, future_f)
                if len(k_resid_calib):
                    _, _, _, _, k_idx_f, _, _ = per_sample_scores(model, in_f)
                    forecast_score_f = two_stage_group_score(k_resid_calib, k_resid_f,
                                                             idx_calib=k_idx_calib, idx_x=k_idx_f)
                else:
                    forecast_score_f = np.zeros(len(vi_f))
            else:
                forecast_score_f = np.zeros(0)
            scores_fault = add_forecast_scores(base_f, forecast_score_f, mask_f)

            for key in rows:
                if len(scores_normal[key]) == 0 or len(scores_fault[key]) == 0:
                    continue
                labels = np.concatenate([np.zeros(len(scores_normal[key])), np.ones(len(scores_fault[key]))])
                s = np.concatenate([scores_normal[key], scores_fault[key]])
                per_client_auroc[key].append(float(sk_metrics.roc_auc_score(labels, s)))

        mean_across_clients = {k: float(np.mean(v)) for k, v in per_client_auroc.items() if v}
        for key in mean_across_clients:
            rows[key].append(mean_across_clients[key])
        report["bearings"][code] = {"category": cat, "origin": origin, "n": len(arr),
                                      "per_client_auroc": {clients[i]["code"]: {k: v[i] for k, v in per_client_auroc.items() if i < len(v)}
                                                             for i in range(len(clients))},
                                      "mean_across_clients": mean_across_clients}
        print(f"  {code:<6}{cat:<12}{origin:<11}" + "  ".join(f"{k}={v:.3f}" for k, v in mean_across_clients.items()))

    print(f"\n{'method':<24}{'mean AUROC (all 26, mean over clients)':>40}")
    summary = {}
    for key in rows:
        if not rows[key]:
            continue
        mean_auroc = float(np.mean(rows[key]))
        summary[key] = mean_auroc
        print(f"{key:<24}{mean_auroc:>40.3f}")
    print(f"\n{'method':<24}{'outer_ring':>12}{'inner_ring':>12}{'combined':>12}")
    category_summary = {}
    for key in rows:
        cat_means = {}
        for cat in ["outer_ring", "inner_ring", "combined"]:
            subset = [report["bearings"][code]["mean_across_clients"][key] for code in ALL_DAMAGED_CODES
                      if category_of(code) == cat and key in report["bearings"][code]["mean_across_clients"]]
            cat_means[cat] = float(np.mean(subset)) if subset else None
        category_summary[key] = cat_means
        print(f"{key:<24}" + "".join(f"{(cat_means[c] if cat_means[c] is not None else float('nan')):>12.3f}"
                                      for c in ["outer_ring", "inner_ring", "combined"]))

    report["summary_mean_auroc"] = summary
    report["category_summary"] = category_summary
    report["n_valid_prototypes_for_cov_calib"] = n_valid_cov_per_client
    if calib_mode == "shrinkage":
        mode_suffix = f"_calibmode_shrinkage_alpha{'auto' if alpha is None else int(alpha)}"
    else:
        mode_suffix = ""
    suffix = out_suffix if out_suffix is not None else (
        f"forecast_v2_federated_h{horizon_mult}{'_noprior' if not use_forecast_prior else ''}"
        + ("_encdec_synced" if SYNC_ENCODER_DECODER else "") + mode_suffix)
    out_json = OUT_DIR / f"paderborn_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
               OUT_DIR / f"paderborn_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-mult", type=int, default=10)
    parser.add_argument("--no-forecast-prior", action="store_true")
    parser.add_argument("--out-suffix", type=str, default=None)
    parser.add_argument("--calib-mode", choices=["global", "shrinkage"], default="global")
    parser.add_argument("--alpha", type=str, default="20",
                        help="shrinkage strength for --calib-mode shrinkage, or 'auto' for data-driven per-slot alpha")
    args = parser.parse_args()
    args.alpha = None if args.alpha == "auto" else float(args.alpha)
    main(horizon_mult=args.horizon_mult, use_forecast_prior=not args.no_forecast_prior,
         out_suffix=args.out_suffix, calib_mode=args.calib_mode, alpha=args.alpha)
