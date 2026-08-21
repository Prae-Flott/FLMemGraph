"""Sample-level overlap between the BK and CK combo-argmax rules, to check
whether they are truly redundant (same samples correct/incorrect) or
complementary (different failure sets) -- not just tied on aggregate rate."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, torch, json
import run_robo3er_forecast_v2_federated as v2f

torch.manual_seed(v2f.SEED); np.random.seed(v2f.SEED)
horizon_mult = 10
forecast_h = v2f.STRIDE * horizon_mult

data, targets, label_map, clients, cols, window_robot = v2f.build_clients()
num_nodes = len(cols)
node_idx = {name: i for i, name in enumerate(cols)}
prior_edges = [(node_idx[s], node_idx[d]) for s, d in v2f.EDGES_NAMED if s in node_idx and d in node_idx]
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
    x_in = data[vi]; x_future = v2f.gather_future(data, chains, v2f.STRIDE)
    fit_pairs.append((vi, x_in, x_future))

for rnd in range(1, v2f.ROUNDS + 1):
    fit_counts = []
    for c, model, scaler, (vi, x_in_raw, x_future_raw) in zip(clients, models, scalers, fit_pairs):
        n, t, f = x_in_raw.shape
        x_in = scaler.transform(x_in_raw.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
        n2, t2, f2 = x_future_raw.shape
        x_future = (scaler.transform(x_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                    if n2 > 0 else x_future_raw)
        v2f.train_local(model, x_in, x_future, v2f.LOCAL_EPOCHS)
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

def zn(x, med, iqr):
    return (x - med) / iqr

def masked_argmax(z, reliable):
    return np.argmax(np.where(reliable[None, :], z, -np.inf), axis=1)

cov_min = max(v2f.MIN_PROTO_SAMPLES, num_nodes + 1)
calib_stats = {}
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

DIRECT_DOMAIN = {
    "stuck": lambda name: name.startswith("wheel_"),
    "cable trapped": lambda name: name == "slip_status_is_slipping",
}
INDIRECT_DOMAIN = {
    "stuck": lambda name: False,
    "cable trapped": lambda name: (
        name.startswith("wheel_vels_velocity")
        or name.startswith("odom_odo_lintw_") or name.startswith("odom_odo_angtw_")
        or name.startswith("imu_imu_angvel_")
    ),
}
def classify(name, fault):
    if DIRECT_DOMAIN[fault](name):
        return "direct"
    if INDIRECT_DOMAIN[fault](name):
        return "indirect"
    return "other"

overlap = {}
for c, model, scaler in zip(clients, models, scalers):
    if not c.fault_idx_by_label:
        continue
    st = calib_stats[c.client_id]
    for label_id, fault_idx in c.fault_idx_by_label.items():
        fault_name = label_map[str(label_id)]
        fault_w = v2f.scale_client(data, scaler, fault_idx)
        _, d_node_f, resid_struct_f, _, idx_f, _, d_mahal_f = v2f.per_sample_scores(model, fault_w)
        z_node_f = zn(d_node_f, st["node_median"], st["node_iqr"])
        z_struct_f = zn(resid_struct_f, st["struct_median"], st["struct_iqr"])
        argmax_b = masked_argmax(z_node_f, st["node_reliable"])
        argmax_c = masked_argmax(z_struct_f, st["struct_reliable"])

        vi_f, chains_f, mask_f = v2f.build_pairs(fault_idx, targets, window_robot, horizon_mult, False, False)
        if len(vi_f):
            fault_in = v2f.scale_client(data, scaler, vi_f)
            future_f_raw = v2f.gather_future(data, chains_f, v2f.STRIDE)
            n2, t2, f2 = future_f_raw.shape
            fault_future = scaler.transform(future_f_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
            k_resid_f = v2f.forecast_scores(model, fault_in, fault_future)
            z_k_f = zn(k_resid_f, st["k_median"], st["k_iqr"])
            argmax_k_paired = masked_argmax(z_k_f, st["k_reliable"])
        else:
            z_k_f = np.zeros((0, num_nodes))
            argmax_k_paired = np.zeros(0, dtype=int)

        b_paired = z_node_f.max(axis=1)[mask_f]
        c_paired = z_struct_f.max(axis=1)[mask_f]
        argmax_b_paired = argmax_b[mask_f]
        argmax_c_paired = argmax_c[mask_f]
        k_scalar_paired = z_k_f.max(axis=1) if len(z_k_f) else np.zeros(0)

        def combo_argmax(scalar_a, argmax_a, scalar_b, argmax_b_):
            if len(scalar_a) == 0:
                return np.zeros(0, dtype=int)
            a_wins = scalar_a >= scalar_b
            return np.where(a_wins, argmax_a, argmax_b_)

        argmax_bk = combo_argmax(b_paired, argmax_b_paired, k_scalar_paired, argmax_k_paired)
        argmax_ck = combo_argmax(c_paired, argmax_c_paired, k_scalar_paired, argmax_k_paired)

        hit_bk = np.array([classify(cols[i], fault_name) != "other" for i in argmax_bk])
        hit_ck = np.array([classify(cols[i], fault_name) != "other" for i in argmax_ck])
        same_argmax = (argmax_bk == argmax_ck)

        rec = overlap.setdefault(fault_name, dict(n=0, both_hit=0, both_miss=0, bk_only=0, ck_only=0,
                                                     same_argmax=0, b_wins_over_k_frac=[], c_wins_over_k_frac=[]))
        rec["n"] += len(hit_bk)
        rec["both_hit"] += int((hit_bk & hit_ck).sum())
        rec["both_miss"] += int((~hit_bk & ~hit_ck).sum())
        rec["bk_only"] += int((hit_bk & ~hit_ck).sum())
        rec["ck_only"] += int((~hit_bk & hit_ck).sum())
        rec["same_argmax"] += int(same_argmax.sum())
        if len(b_paired):
            rec["b_wins_over_k_frac"].append(float((b_paired >= k_scalar_paired).mean()))
        if len(c_paired):
            rec["c_wins_over_k_frac"].append(float((c_paired >= k_scalar_paired).mean()))

for fault, rec in overlap.items():
    rec["b_wins_over_k_frac"] = float(np.mean(rec["b_wins_over_k_frac"])) if rec["b_wins_over_k_frac"] else None
    rec["c_wins_over_k_frac"] = float(np.mean(rec["c_wins_over_k_frac"])) if rec["c_wins_over_k_frac"] else None

print(json.dumps(overlap, indent=1))
with open("/home/roboserver/Projects/FLMemGraph/checkpoints/robo3er/bk_ck_overlap.json", "w") as f:
    json.dump(overlap, f, indent=1)
