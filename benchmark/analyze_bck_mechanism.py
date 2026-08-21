"""
Diagnostic analysis: why do B (node amplitude) and C (attention-structural
residual) agree so strongly on argmax, and what role does C's cross-node
attention structure play vs K (forecast residual)?

Reuses the exact same federated training/calibration path as
diagnose_robo3er_localization_federated.py, but additionally dumps:
  - per-sample argmax agreement rate between B and C (and both vs K)
  - ||d|| (raw per-node deviation magnitude that feeds BOTH B and C)
  - ||d_hat|| (C's attention-predicted deviation) relative to ||d||
    -> if d_hat is small relative to d, resid_struct ~ ||d||^2 ~ d_node,
       explaining the B/C agreement structurally, not coincidentally.
  - k_resid vs d_node correlation (K's forecast-residual signal vs B's
    prototype-deviation signal) to characterize how K differs mechanistically.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "models"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "robo3er"))

import numpy as np
import torch
import json

import run_robo3er_forecast_v2_federated as v2f

torch.manual_seed(v2f.SEED)
np.random.seed(v2f.SEED)

horizon_mult = 10
forecast_h = v2f.STRIDE * horizon_mult
rounds = v2f.ROUNDS
local_epochs = v2f.LOCAL_EPOCHS

print("loading robo3er 26-node federated clients ...")
data, targets, label_map, clients, cols, window_robot = v2f.build_clients()
num_nodes = len(cols)
node_idx = {name: i for i, name in enumerate(cols)}
prior_edges = [(node_idx[s], node_idx[d]) for s, d in v2f.EDGES_NAMED
               if s in node_idx and d in node_idx]
edge_types = [t for (s, d), t in zip(v2f.EDGES_NAMED, v2f.EDGE_TYPES) if s in node_idx and d in node_idx]

scalers = [v2f.fit_client_scaler(data, c.fit_idx) for c in clients]
models = [v2f.JointPrototypeV31Forecast(
    num_nodes=num_nodes, window_size=v2f.WINDOW_LEN, embed_dim=v2f.EMBED_DIM,
    num_prototypes=v2f.NUM_PROTOTYPES, prior_edges=prior_edges, edge_types=edge_types,
    forecast_h=forecast_h, top_k=v2f.TOP_K, forecast_prior_edges=prior_edges,
).to(v2f.DEVICE) for _ in clients]

shared_init = {k: v.clone() for k, v in models[0].state_dict().items() if not k.startswith("memory.")}
for model in models[1:]:
    model.load_state_dict(shared_init, strict=False)

fit_pairs = []
for c in clients:
    vi, chains, _ = v2f.build_pairs(c.fit_idx, targets, window_robot, horizon_mult, True, True)
    x_in = data[vi]
    x_future = v2f.gather_future(data, chains, v2f.STRIDE)
    fit_pairs.append((vi, x_in, x_future))

print(f"federated training: {rounds} rounds x {local_epochs} local epochs ...")
for rnd in range(1, rounds + 1):
    fit_counts = []
    for c, model, scaler, (vi, x_in_raw, x_future_raw) in zip(clients, models, scalers, fit_pairs):
        n, t, f = x_in_raw.shape
        x_in = scaler.transform(x_in_raw.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
        n2, t2, f2 = x_future_raw.shape
        x_future = (scaler.transform(x_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                    if n2 > 0 else x_future_raw)
        v2f.train_local(model, x_in, x_future, local_epochs)
        fit_counts.append(len(x_in))
    codebooks = [m.memory.codebook.detach().clone() for m in models]
    usage_counts = [m.memory.usage_count.detach().clone() for m in models]
    P_G, diag = v2f.align_and_split(codebooks, usage_counts, gamma=v2f.GAMMA, delta=v2f.DELTA)
    for model, p_g in zip(models, P_G):
        model.memory.load_memory(p_g)
    if v2f.SYNC_ENCODER_DECODER:
        avg = v2f.fedavg_state_dict([m.state_dict() for m in models], fit_counts, prefixes=("encoder.", "decoder."))
        for model in models:
            model.load_state_dict(avg, strict=False)

print("calibrating ...")
calib_stats = {}
cov_min = max(v2f.MIN_PROTO_SAMPLES, num_nodes + 1)
for c, model, scaler in zip(clients, models, scalers):
    calib_w = v2f.scale_client(data, scaler, c.calib_idx)
    _, d_node_calib_b, _, _, calib_idx_w, calib_r_w, _ = v2f.per_sample_scores(model, calib_w)
    model.typed_head.set_calibration(calib_r_w, calib_idx_w, v2f.NUM_PROTOTYPES, min_samples=v2f.MIN_PROTO_SAMPLES)
    model.cov_head.set_calibration(d_node_calib_b, calib_idx_w, min_samples=cov_min)
    _, d_node_calib, resid_struct_calib, _, calib_idx_b2, _, d_mahal_calib = v2f.per_sample_scores(model, calib_w)
    vi, chains, _ = v2f.build_pairs(c.calib_idx, targets, window_robot, horizon_mult, True, True)
    if len(vi):
        calib_in = v2f.scale_client(data, scaler, vi)
        calib_future_raw = v2f.gather_future(data, chains, v2f.STRIDE)
        n2, t2, f2 = calib_future_raw.shape
        calib_future = scaler.transform(calib_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
        k_resid_calib = v2f.forecast_scores(model, calib_in, calib_future)
    else:
        k_resid_calib = np.zeros((0, num_nodes), dtype=np.float32)
    node_median = np.median(d_node_calib, axis=0)
    node_iqr = np.maximum(np.percentile(d_node_calib, 75, axis=0) - np.percentile(d_node_calib, 25, axis=0), 1e-8)
    struct_median = np.median(resid_struct_calib, axis=0)
    struct_iqr = np.maximum(np.percentile(resid_struct_calib, 75, axis=0) - np.percentile(resid_struct_calib, 25, axis=0), 1e-8)
    if len(k_resid_calib):
        k_median = np.median(k_resid_calib, axis=0)
        k_iqr = np.maximum(np.percentile(k_resid_calib, 75, axis=0) - np.percentile(k_resid_calib, 25, axis=0), 1e-8)
    else:
        k_median, k_iqr = np.zeros(num_nodes), np.ones(num_nodes)
    node_reliable = node_iqr >= 0.05 * np.median(node_iqr)
    struct_reliable = struct_iqr >= 0.05 * np.median(struct_iqr)
    k_reliable = k_iqr >= 0.05 * np.median(k_iqr)
    calib_stats[c.client_id] = dict(node_median=node_median, node_iqr=node_iqr, node_reliable=node_reliable,
                                     struct_median=struct_median, struct_iqr=struct_iqr, struct_reliable=struct_reliable,
                                     k_median=k_median, k_iqr=k_iqr, k_reliable=k_reliable)

# ---- Now the actual mechanism analysis on fault windows ----
results = {"per_client": [], "cols": cols}
all_d_norm, all_dhat_norm, all_resid_norm = [], [], []
agree_bc, agree_bk, agree_ck, total, total_paired = 0, 0, 0, 0, 0
pearson_bc_scores, pearson_bk_scores = [], []

for c, model, scaler in zip(clients, models, scalers):
    if not c.fault_idx_by_label:
        continue
    st = calib_stats[c.client_id]
    for label_id, fault_idx in c.fault_idx_by_label.items():
        fault_name = label_map[str(label_id)]
        fault_w = v2f.scale_client(data, scaler, fault_idx)

        model.eval()
        with torch.no_grad():
            batch = torch.from_numpy(fault_w).to(v2f.DEVICE)
            out = model(batch, training_mode=False)
            z = out["z"]
            p_star = out["p_star"]
            d = (z - p_star).cpu().numpy()  # [B, N, D] raw deviation feeding BOTH B and C
            d_hat, resid_struct_t = model.edge_head(z, p_star)
            d_hat = d_hat.cpu().numpy()
            d_node_f = out["d_node"].cpu().numpy()
            resid_struct_f = resid_struct_t.cpu().numpy()

        vi, chains, mask_f = v2f.build_pairs(fault_idx, targets, window_robot, horizon_mult, False, False)
        if len(vi):
            fin = v2f.scale_client(data, scaler, vi)
            ffut_raw = v2f.gather_future(data, chains, v2f.STRIDE)
            n2, t2, f2 = ffut_raw.shape
            ffut = scaler.transform(ffut_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
            k_resid_f = v2f.forecast_scores(model, fin, ffut)
        else:
            k_resid_f = np.zeros((0, num_nodes), dtype=np.float32)

        d_norm = np.linalg.norm(d, axis=-1)          # [B, N] ||d_i||
        resid_norm = np.sqrt(resid_struct_f)          # [B, N] ||d_i - d_hat_i||
        # d_hat contribution magnitude per node
        dhat_norm = np.linalg.norm(d_hat, axis=-1)

        all_d_norm.append(d_norm.ravel())
        all_dhat_norm.append(dhat_norm.ravel())
        all_resid_norm.append(resid_norm.ravel())

        z_b = (d_node_f - st["node_median"]) / st["node_iqr"]
        z_c = (resid_struct_f - st["struct_median"]) / st["struct_iqr"]
        rel_b = st["node_reliable"]
        rel_c = st["struct_reliable"]
        argmax_b = np.argmax(np.where(rel_b[None, :], z_b, -np.inf), axis=1)
        argmax_c = np.argmax(np.where(rel_c[None, :], z_c, -np.inf), axis=1)
        agree_bc += (argmax_b == argmax_c).sum()
        total += len(argmax_b)

        if len(k_resid_f):
            z_k = (k_resid_f - st["k_median"]) / st["k_iqr"]
            rel_k = st["k_reliable"]
            argmax_k = np.argmax(np.where(rel_k[None, :], z_k, -np.inf), axis=1)
            # align B/C argmax (defined over ALL fault_idx samples) to the subset
            # of samples that had a valid horizon-mult forward chain (mask_f)
            argmax_b_paired = argmax_b[mask_f]
            argmax_c_paired = argmax_c[mask_f]
            agree_bk += (argmax_b_paired == argmax_k).sum()
            agree_ck += (argmax_c_paired == argmax_k).sum()
            total_paired += len(argmax_k)

            # per-window Pearson correlation between K's raw residual and B's
            # raw d_node, over the SAME paired samples
            pearson_bk_scores.append(np.corrcoef(d_node_f[mask_f].ravel(), k_resid_f.ravel())[0, 1])

        # per-window Pearson correlation between B's and C's raw score vectors (flattened)
        pearson_bc_scores.append(np.corrcoef(d_node_f.ravel(), resid_struct_f.ravel())[0, 1])

        results["per_client"].append(dict(
            client=c.client_id, fault=fault_name, n=len(fault_idx),
            mean_d_norm=float(d_norm.mean()), mean_dhat_norm=float(dhat_norm.mean()),
            mean_resid_norm=float(resid_norm.mean()),
            frac_dhat_over_d=float((dhat_norm.mean()) / (d_norm.mean() + 1e-12)),
        ))

all_d_norm = np.concatenate(all_d_norm)
all_dhat_norm = np.concatenate(all_dhat_norm)
all_resid_norm = np.concatenate(all_resid_norm)

summary = dict(
    bc_argmax_agreement_rate=float(agree_bc / total) if total else None,
    bk_argmax_agreement_rate=float(agree_bk / total_paired) if total_paired else None,
    ck_argmax_agreement_rate=float(agree_ck / total_paired) if total_paired else None,
    total_samples=int(total),
    total_paired_samples=int(total_paired),
    mean_d_norm_over_all=float(all_d_norm.mean()),
    mean_dhat_norm_over_all=float(all_dhat_norm.mean()),
    mean_resid_norm_over_all=float(all_resid_norm.mean()),
    dhat_to_d_ratio=float(all_dhat_norm.mean() / (all_d_norm.mean() + 1e-12)),
    resid_to_d_ratio=float(all_resid_norm.mean() / (all_d_norm.mean() + 1e-12)),
    mean_pearson_score_bc=float(np.nanmean(pearson_bc_scores)),
    mean_pearson_score_bk=float(np.nanmean(pearson_bk_scores)) if pearson_bk_scores else None,
    per_client=results["per_client"],
)

out_path = "/home/roboserver/Projects/FLMemGraph/checkpoints/robo3er/bck_mechanism_analysis.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "per_client"}, indent=1))
print("saved:", out_path)
