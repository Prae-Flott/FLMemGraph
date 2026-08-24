#!/usr/bin/env python3
"""
FEDERATED JointPrototypeV21Forecast (V2.1 + ForecastHead, signal K) on
Sielaff's 10 real machines as 10 federated clients. Per
`memory/benchmark-policy-federated-only.md`: the federated variant of
`run_sielaff_forecast_v2.py`'s per-machine cross-window pairing route
(same pairing rules -- see that script's docstring), grafted onto
`run_sielaff_v2_1_federated.py`'s federated protocol (per-client local
encoder/edge_head/cov_head/forecast_head, ONLY the memory codebook
exchanged via `align_and_split`, no encoder/decoder FedAvg on this
dataset -- matches `run_sielaff_v2_1_federated.py`'s "memory-only
exchange"). `forecast_head` stays local, same as `edge_head`/`cov_head`.

Adds `K_forecast_max`/`BK_max`/`CK_max`/`HK_max`, reusing
`RELIABILITY_RATIO` sparse-feature masking for K same as the centralized
`run_sielaff_forecast_v2.py`.

Usage:
    python3 run_sielaff_forecast_v2_federated.py [--horizon-mult M] [--no-forecast-prior]
        [--out-suffix NAME] [--calib-mode {global,shrinkage}]
        [--num-prototypes N] [--alpha 20|auto]

`--calib-mode` (default `global`, unchanged behavior): see
`run_robo3er_v3_1_federated.py`'s docstring and
`memory/calib-in-prototype-ab.md`. Applies to B's per-node z-score (the
reliability mask stays global in every mode, same simplification as
`run_sielaff_v2_1_federated.py`). `shrinkage` has no hook for C/K, falling
back to `global` for those (same scope limit as elsewhere).

`--num-prototypes` (default `NUM_PROTOTYPES` below, unchanged behavior):
see `run_sielaff_v2_1_federated.py`'s docstring and
`memory/sielaff-num-prototypes-sweep.md` for the grid-search this flag
was added for.
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
sys.path.insert(0, str(REPO_ROOT / "src" / "federated"))
from joint_prototype_model import JointPrototypeV21Forecast  # noqa: E402
from federated_memory import (align_and_split,  # noqa: E402
                               compute_prototype_dev_stats, compute_shrinkage_stats)

DATA_DIR = REPO_ROOT / "data" / "sielaff"
OUT_DIR = REPO_ROOT / "checkpoints" / "sielaff"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15
BATCH_SIZE = 256
ROUNDS = 5
LOCAL_EPOCHS = 30
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 16
WINDOW_LEN = 8
STRIDE = 2
TOP_K = 8
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_FORECAST = 0.5
GAMMA = 0.5
DELTA = 0.5
RELIABILITY_RATIO = 0.05
MIN_PROTO_SAMPLES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Client:
    def __init__(self, client_id, all_idx, fit_idx, calib_idx, test_normal_idx):
        self.client_id = client_id
        self.all_idx = all_idx
        self.fit_idx = fit_idx
        self.calib_idx = calib_idx
        self.test_normal_idx = test_normal_idx


def build_clients():
    data = np.load(DATA_DIR / "data.npy")
    targets = np.load(DATA_DIR / "targets.npy")
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    cols = meta["feature_columns"]
    label_map = meta["class_names"]

    with open(DATA_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)

    window_machine = np.full(len(targets), -1, dtype=int)
    clients = []
    for client_id in range(partition["separation"]["total"]):
        d = partition["data_indices"][client_id]
        idx = np.sort(np.concatenate([d["train"], d["val"], d["test"]]).astype(np.int64))
        window_machine[idx] = client_id
        normal_idx = idx[targets[idx] == 0]
        n = len(normal_idx)
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        clients.append(Client(
            client_id=client_id,
            all_idx=idx,
            fit_idx=normal_idx[:n_fit],
            calib_idx=normal_idx[n_fit : n_fit + n_calib],
            test_normal_idx=normal_idx[n_fit + n_calib :],
        ))
    return data, targets, cols, label_map, clients, window_machine


def build_pairs(idx_arr, targets, window_machine, horizon_mult, require_next_normal, require_same_split=False):
    N = len(targets)
    idx_set = set(idx_arr.tolist()) if require_same_split else None
    vi, chains, mask = [], [], []
    for i in idx_arr:
        chain = [i + k for k in range(1, horizon_mult + 1)]
        valid = True
        for j in chain:
            if (j >= N or window_machine[j] != window_machine[i]
                    or (require_next_normal and targets[j] != 0)
                    or (require_same_split and j not in idx_set)):
                valid = False
                break
        mask.append(valid)
        if valid:
            vi.append(i)
            chains.append(chain)
    return np.array(vi, dtype=int), chains, np.array(mask, dtype=bool)


def gather_future(data_arr, chains, stride):
    if len(chains) == 0:
        return np.zeros((0, 0, data_arr.shape[-1]), dtype=data_arr.dtype)
    segments = [data_arr[[c[k] for c in chains]][:, -stride:, :] for k in range(len(chains[0]))]
    return np.concatenate(segments, axis=1)


def fit_client_scaler(data, fit_idx):
    scaler = StandardScaler()
    scaler.fit(data[fit_idx].reshape(-1, data.shape[-1]))
    return scaler


def scale_client(data, scaler, idx):
    if len(idx) == 0:
        return np.zeros((0,) + data.shape[1:], dtype=np.float32)
    n, t, f = data[idx].shape
    return scaler.transform(data[idx].reshape(-1, f)).reshape(n, t, f).astype(np.float32)


@torch.no_grad()
def per_sample_scores(model, windows, batch_size=BATCH_SIZE):
    model.eval()
    d_proto_all, d_node_all, resid_struct_all, idx_all, d_mahal_all = [], [], [], [], []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(DEVICE)
        out = model(batch, training_mode=False)
        d_proto_all.append(out["d_proto"].cpu().numpy())
        d_node_all.append(out["d_node"].cpu().numpy())
        resid_struct_all.append(out["resid_struct"].cpu().numpy())
        idx_all.append(out["idx"].cpu().numpy())
        d_mahal_all.append(out["d_mahal"].cpu().numpy())
    return (np.concatenate(d_proto_all), np.concatenate(d_node_all),
            np.concatenate(resid_struct_all), np.concatenate(idx_all),
            np.concatenate(d_mahal_all))


@torch.no_grad()
def forecast_scores(model, x_in, x_future, batch_size=BATCH_SIZE):
    model.eval()
    if len(x_in) == 0:
        return np.zeros((0, model.num_nodes), dtype=np.float32)
    k_resid_all = []
    for i in range(0, len(x_in), batch_size):
        xb = torch.from_numpy(x_in[i : i + batch_size]).to(DEVICE)
        fb = torch.from_numpy(x_future[i : i + batch_size]).to(DEVICE)
        out = model(xb, training_mode=False, x_future=fb)
        k_resid_all.append(out["k_resid"].cpu().numpy())
    return np.concatenate(k_resid_all)


def train_local(model, fit_in, fit_future, epochs):
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
            l_forecast = out["k_resid"].mean()
            loss = l_pred + BETA * l_me + l_mc + LAMBDA_EDGE * l_edge + LAMBDA_FORECAST * l_forecast
            loss.backward()
            optimizer.step()
    return model


def zscore(x, calib_x):
    median = np.median(calib_x, axis=0)
    q75, q25 = np.percentile(calib_x, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    return (x - median) / iqr, iqr


def zscore_mode(x, idx_x, calib_x, idx_calib, dev_memory=None):
    if dev_memory is not None:
        return dev_memory.dev_zscore(x, idx_x)
    z, _ = zscore(x, calib_x)
    return z


def main(horizon_mult=10, use_forecast_prior=True, out_suffix=None, calib_mode="global",
         num_prototypes=NUM_PROTOTYPES, alpha=20.0):
    assert calib_mode in ("global", "shrinkage")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    forecast_h = STRIDE * horizon_mult

    print(f"loading Sielaff, splitting into 10 real per-machine federated clients, "
          f"horizon_mult={horizon_mult} ...")
    data, targets, cols, label_map, clients, window_machine = build_clients()
    num_nodes = len(cols)
    for c in clients:
        print(f"  client {c.client_id}: fit={len(c.fit_idx)} calib={len(c.calib_idx)} "
              f"test_normal={len(c.test_normal_idx)}")

    scalers = [fit_client_scaler(data, c.fit_idx) for c in clients]
    models = [JointPrototypeV21Forecast(num_nodes=num_nodes, window_size=WINDOW_LEN,
                                         embed_dim=EMBED_DIM, num_prototypes=num_prototypes,
                                         forecast_h=forecast_h, prior_edges=None, top_k=TOP_K).to(DEVICE)
              for _ in clients]

    shared_init = {k: v.clone() for k, v in models[0].state_dict().items() if not k.startswith("memory.")}
    for model in models[1:]:
        model.load_state_dict(shared_init, strict=False)

    fit_pairs = []
    for c in clients:
        vi, chains, _ = build_pairs(c.fit_idx, targets, window_machine, horizon_mult, True, True)
        fit_pairs.append((vi, data[vi], gather_future(data, chains, STRIDE)))

    print(f"\nfederated training: {ROUNDS} rounds x {LOCAL_EPOCHS} local epochs, "
          f"memory-only exchange (gamma={GAMMA}, delta={DELTA})")
    diagnostics_log = []
    for rnd in range(1, ROUNDS + 1):
        for c, model, scaler, (vi, x_in_raw, x_future_raw) in zip(clients, models, scalers, fit_pairs):
            n, t, f = x_in_raw.shape
            x_in = scaler.transform(x_in_raw.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
            n2, t2, f2 = x_future_raw.shape
            x_future = (scaler.transform(x_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                        if n2 > 0 else x_future_raw)
            train_local(model, x_in, x_future, LOCAL_EPOCHS)

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
                _, d_node_fit, _, idx_fit, _ = per_sample_scores(model, x_in)
                client_dev_stats.append(compute_prototype_dev_stats(d_node_fit, idx_fit, num_prototypes))
            num_shared = diag["num_shared_prototypes"]
            shrink_stats, alpha_used = compute_shrinkage_stats(client_dev_stats, num_shared, alpha)
            diag["shrinkage_alpha"] = alpha_used.tolist()
            for model, (mean_s, var_s, valid_s) in zip(models, shrink_stats):
                model.memory.load_shrinkage_stats(mean_s, var_s, valid_s)

    report = {"config": {"rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "num_prototypes": num_prototypes,
                          "embed_dim": EMBED_DIM, "window_len": WINDOW_LEN, "gamma": GAMMA, "delta": DELTA,
                          "reliability_ratio": RELIABILITY_RATIO, "horizon_mult": horizon_mult,
                          "forecast_h": forecast_h, "use_forecast_prior": use_forecast_prior},
              "nodes": cols, "alignment_log": diagnostics_log, "clients": {}}

    cov_min = max(5, num_nodes + 1)
    all_pairs = {"B_node_max": [], "C_struct_max": [], "H_cov_mahal": [], "I_node_cov_max": [],
                 "K_forecast_max": [], "BK_max": [], "CK_max": [], "HK_max": []}
    for c, model, scaler in zip(clients, models, scalers):
        calib_arr = scale_client(data, scaler, c.calib_idx)
        test_normal_arr = scale_client(data, scaler, c.test_normal_idx)

        _, d_node_b, _, calib_idx_b, _ = per_sample_scores(model, calib_arr)
        model.cov_head.set_calibration(d_node_b, calib_idx_b, min_samples=cov_min)

        d_proto_calib, d_node_calib, resid_struct_calib, calib_idx_w, d_mahal_calib = per_sample_scores(model, calib_arr)
        d_proto_normal, d_node_normal, resid_struct_normal, idx_normal, d_mahal_normal = per_sample_scores(model, test_normal_arr)

        dev_mem = model.memory if calib_mode == "shrinkage" else None  # C/K fall back
        # to global (no shrinkage hook); dev_zscore() reads whatever
        # load_shrinkage_stats() populated this round.

        vi_c, chains_c, _ = build_pairs(c.calib_idx, targets, window_machine, horizon_mult, True, True)
        calib_in = scale_client(data, scaler, vi_c)
        future_c_raw = gather_future(data, chains_c, STRIDE)
        if len(vi_c):
            n2, t2, f2 = future_c_raw.shape
            calib_future = scaler.transform(future_c_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
        else:
            calib_future = future_c_raw
        k_resid_calib = forecast_scores(model, calib_in, calib_future)
        if len(vi_c):
            _, _, _, k_idx_calib, _ = per_sample_scores(model, calib_in)
        else:
            k_idx_calib = np.zeros((0,), dtype=int)
        vi_n, chains_n, mask_n = build_pairs(c.test_normal_idx, targets, window_machine, horizon_mult, True, True)
        test_normal_in = scale_client(data, scaler, vi_n)
        future_n_raw = gather_future(data, chains_n, STRIDE)
        if len(vi_n):
            n2, t2, f2 = future_n_raw.shape
            test_normal_future = scaler.transform(future_n_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
        else:
            test_normal_future = future_n_raw
        k_resid_normal = forecast_scores(model, test_normal_in, test_normal_future)
        if len(vi_n):
            _, _, _, k_idx_normal, _ = per_sample_scores(model, test_normal_in)
        else:
            k_idx_normal = np.zeros((0,), dtype=int)

        # Some tiny clients (e.g. Sielaff machine 0, calib=6) have too few
        # windows for even one valid horizon_mult=10 chain -- k_resid_calib
        # ends up empty and there is no baseline to z-score against, so the
        # forecast signal (and its BK/CK/HK combinations) is disabled
        # entirely for this client rather than padded with mismatched shapes.
        has_forecast = len(k_resid_calib) > 0

        _, node_iqr = zscore(d_node_normal, d_node_calib)
        reliable_mask = node_iqr >= RELIABILITY_RATIO * np.median(node_iqr)
        if has_forecast:
            _, forecast_iqr = zscore(k_resid_normal, k_resid_calib)
            forecast_mask = forecast_iqr >= RELIABILITY_RATIO * np.median(forecast_iqr)
        else:
            forecast_mask = np.ones(num_nodes, dtype=bool)

        z_node_normal = zscore_mode(d_node_normal, idx_normal, d_node_calib, calib_idx_w, dev_memory=dev_mem)
        z_struct_normal = zscore_mode(resid_struct_normal, idx_normal, resid_struct_calib, calib_idx_w)
        z_mahal_normal, _ = zscore(d_mahal_normal, d_mahal_calib)
        b_normal_full = z_node_normal[:, reliable_mask].max(axis=1)
        c_normal_full = z_struct_normal.max(axis=1)
        h_normal_full = z_mahal_normal
        base_normal = {"B_node_max": b_normal_full, "C_struct_max": c_normal_full,
                       "H_cov_mahal": h_normal_full, "I_node_cov_max": np.maximum(b_normal_full, h_normal_full)}

        if has_forecast:
            # B/C/H/I stay FULL-length in scores_normal -- only K/BK/CK/HK use
            # the forecast-pairing-masked subset (local copies only, never
            # written back onto the full-length entries). Matches the
            # centralized `run_sielaff_forecast_v2.py`'s convention.
            b_masked, c_masked, h_masked = (base_normal["B_node_max"][mask_n], base_normal["C_struct_max"][mask_n],
                                             base_normal["H_cov_mahal"][mask_n])
            z_forecast_normal = zscore_mode(k_resid_normal, k_idx_normal, k_resid_calib, k_idx_calib)
            k_normal = z_forecast_normal[:, forecast_mask].max(axis=1)
            scores_normal = dict(base_normal)
            scores_normal["K_forecast_max"] = k_normal
            scores_normal["BK_max"] = np.maximum(b_masked, k_normal)
            scores_normal["CK_max"] = np.maximum(c_masked, k_normal)
            scores_normal["HK_max"] = np.maximum(h_masked, k_normal)
        else:
            scores_normal = dict(base_normal)

        client_report = {"excluded_nodes_node": [cols[i] for i in range(num_nodes) if not reliable_mask[i]],
                          "excluded_nodes_forecast": [cols[i] for i in range(num_nodes) if not forecast_mask[i]],
                          "n_valid_cov_prototypes": int(model.cov_head.calib_valid.sum()),
                          "num_prototypes": num_prototypes,
                          "has_forecast": has_forecast,
                          "fault_types": {}}
        rows = {k: [] for k in scores_normal}
        for label_id_str, name in label_map.items():
            label_id = int(label_id_str)
            if label_id == 0:
                continue
            fault_idx = np.where(targets == label_id)[0]
            fault_idx = np.intersect1d(fault_idx, c.all_idx)
            if len(fault_idx) == 0:
                continue
            fault_arr = scale_client(data, scaler, fault_idx)
            d_proto_f, d_node_f, resid_struct_f, idx_f, d_mahal_f = per_sample_scores(model, fault_arr)
            z_node_f = zscore_mode(d_node_f, idx_f, d_node_calib, calib_idx_w, dev_memory=dev_mem)
            z_struct_f = zscore_mode(resid_struct_f, idx_f, resid_struct_calib, calib_idx_w)
            z_mahal_f, _ = zscore(d_mahal_f, d_mahal_calib)
            b_f_full = z_node_f[:, reliable_mask].max(axis=1)
            c_f_full = z_struct_f.max(axis=1)
            h_f_full = z_mahal_f
            base_f = {"B_node_max": b_f_full, "C_struct_max": c_f_full,
                      "H_cov_mahal": h_f_full, "I_node_cov_max": np.maximum(b_f_full, h_f_full)}

            if has_forecast:
                vi_f, chains_f, mask_f = build_pairs(fault_idx, targets, window_machine, horizon_mult, False, False)
                fault_in = scale_client(data, scaler, vi_f)
                future_f_raw = gather_future(data, chains_f, STRIDE)
                if len(vi_f):
                    n2, t2, f2 = future_f_raw.shape
                    fault_future = scaler.transform(future_f_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                else:
                    fault_future = future_f_raw
                k_resid_f = forecast_scores(model, fault_in, fault_future)
                b_f_masked, c_f_masked, h_f_masked = (base_f["B_node_max"][mask_f], base_f["C_struct_max"][mask_f],
                                                       base_f["H_cov_mahal"][mask_f])
                if len(k_resid_f):
                    _, _, _, k_idx_f, _ = per_sample_scores(model, fault_in)
                    z_forecast_f = zscore_mode(k_resid_f, k_idx_f, k_resid_calib, k_idx_calib)
                    k_f = z_forecast_f[:, forecast_mask].max(axis=1)
                else:
                    k_f = np.zeros(0)
                scores_fault = dict(base_f)
                scores_fault["K_forecast_max"] = k_f
                scores_fault["BK_max"] = np.maximum(b_f_masked, k_f)
                scores_fault["CK_max"] = np.maximum(c_f_masked, k_f)
                scores_fault["HK_max"] = np.maximum(h_f_masked, k_f)
            else:
                scores_fault = dict(base_f)

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
            print(f"  machine{c.client_id:<3}{name:<20}n={len(fault_idx):<5}"
                  + "  ".join(f"{k}={v:.3f}" for k, v in aurocs.items()))

        if rows["B_node_max"]:
            client_report["client_mean_auroc"] = {k: float(np.mean(v)) for k, v in rows.items() if v}
        report["clients"][f"machine{c.client_id}"] = client_report

    summary_overall = {k: float(np.mean(v)) for k, v in all_pairs.items() if v}
    print(f"\n{'method':<24}{'mean AUROC (all machine-fault pairs)':>36}")
    for key, v in summary_overall.items():
        print(f"{key:<24}{v:>36.3f}")

    report["config"]["calib_mode"] = calib_mode
    if calib_mode == "shrinkage":
        report["config"]["alpha"] = "auto" if alpha is None else alpha
    report["summary_mean_auroc_overall"] = summary_overall
    if calib_mode == "shrinkage":
        mode_suffix = f"_calibmode_shrinkage_alpha{'auto' if alpha is None else int(alpha)}"
    else:
        mode_suffix = ""
    proto_suffix = f"_m{num_prototypes}" if num_prototypes != NUM_PROTOTYPES else ""
    suffix = out_suffix if out_suffix is not None else (
        f"forecast_v2_federated_h{horizon_mult}{'_noprior' if not use_forecast_prior else ''}{mode_suffix}{proto_suffix}")
    out_json = OUT_DIR / f"sielaff_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
               OUT_DIR / f"sielaff_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-mult", type=int, default=10)
    parser.add_argument("--no-forecast-prior", action="store_true")
    parser.add_argument("--out-suffix", type=str, default=None)
    parser.add_argument("--calib-mode", choices=["global", "shrinkage"], default="global")
    parser.add_argument("--num-prototypes", type=int, default=NUM_PROTOTYPES)
    parser.add_argument("--alpha", type=str, default="20",
                        help="shrinkage strength for --calib-mode shrinkage, or 'auto' for data-driven per-slot alpha")
    args = parser.parse_args()
    args.alpha = None if args.alpha == "auto" else float(args.alpha)
    main(horizon_mult=args.horizon_mult, use_forecast_prior=not args.no_forecast_prior,
         out_suffix=args.out_suffix, calib_mode=args.calib_mode, num_prototypes=args.num_prototypes,
         alpha=args.alpha)
