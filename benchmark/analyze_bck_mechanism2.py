"""Follow-up: per-fault-type B/C/K correlation + learned attention topology
for the specific nodes that dominate stuck/cable-trapped argmax."""
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

# ---- inspect learned top-k neighbor graph for the stuck-fault client (id=4) ----
stuck_client = next(c for c in clients if c.client_id == 4)
model_stuck = models[clients.index(stuck_client)]
neighbor_idx = model_stuck.edge_head._topk_neighbors().cpu().numpy()  # [N, top_k]
wheel_l = node_idx["wheel_status_current_ma_left"]
wheel_r = node_idx["wheel_status_current_ma_right"]
print("stuck-client learned top-k attention neighbors of wheel_status_current_ma_left:",
      [cols[i] for i in neighbor_idx[wheel_l]])
print("stuck-client learned top-k attention neighbors of wheel_status_current_ma_right:",
      [cols[i] for i in neighbor_idx[wheel_r]])
print("is wheel_right in wheel_left's neighbor set?", wheel_r in neighbor_idx[wheel_l])
print("is wheel_left in wheel_right's neighbor set?", wheel_l in neighbor_idx[wheel_r])

prior_mask = model_stuck.edge_head.prior_mask.cpu().numpy()
print("declared physics edge wheel_left<-wheel_right in prior_mask:", prior_mask[wheel_l, wheel_r])
print("declared physics edge wheel_right<-wheel_left in prior_mask:", prior_mask[wheel_r, wheel_l])
print("prior_bias_strength (learned scalar):", float(model_stuck.edge_head.prior_bias_strength.item()))

# ---- per-fault-type correlation between B (d_node) and C (resid_struct), and B vs K ----
out = {}
for c, model, scaler in zip(clients, models, scalers):
    if not c.fault_idx_by_label:
        continue
    for label_id, fault_idx in c.fault_idx_by_label.items():
        fault_name = label_map[str(label_id)]
        fault_w = v2f.scale_client(data, scaler, fault_idx)
        model.eval()
        with torch.no_grad():
            batch = torch.from_numpy(fault_w).to(v2f.DEVICE)
            o = model(batch, training_mode=False)
            d_node_f = o["d_node"].cpu().numpy()
            resid_struct_f = o["resid_struct"].cpu().numpy()
        vi, chains, mask_f = v2f.build_pairs(fault_idx, targets, window_robot, horizon_mult, False, False)
        if len(vi):
            fin = v2f.scale_client(data, scaler, vi)
            ffut_raw = v2f.gather_future(data, chains, v2f.STRIDE)
            n2, t2, f2 = ffut_raw.shape
            ffut = scaler.transform(ffut_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
            k_resid_f = v2f.forecast_scores(model, fin, ffut)
        else:
            k_resid_f = np.zeros((0, num_nodes), dtype=np.float32)
        pear_bc = float(np.corrcoef(d_node_f.ravel(), resid_struct_f.ravel())[0, 1])
        pear_bk = float(np.corrcoef(d_node_f[mask_f].ravel(), k_resid_f.ravel())[0, 1]) if len(k_resid_f) else None
        # per-node correlation just on the wheel_left/right columns (the argmax winners for stuck)
        pear_bc_wheel = float(np.corrcoef(d_node_f[:, wheel_l], resid_struct_f[:, wheel_l])[0, 1])
        out[fault_name] = dict(pearson_bc_all=pear_bc, pearson_bk_all=pear_bk,
                                pearson_bc_wheel_left=pear_bc_wheel, n=len(fault_idx))

print(json.dumps(out, indent=1))
with open("/home/roboserver/Projects/FLMemGraph/checkpoints/robo3er/bck_mechanism_analysis2.json", "w") as f:
    json.dump(dict(per_fault=out,
                    stuck_neighbors_of_wheel_left=[cols[i] for i in neighbor_idx[wheel_l]],
                    stuck_neighbors_of_wheel_right=[cols[i] for i in neighbor_idx[wheel_r]],
                    wheel_pair_mutual_neighbor=bool(wheel_r in neighbor_idx[wheel_l] and wheel_l in neighbor_idx[wheel_r]),
                    prior_bias_strength=float(model_stuck.edge_head.prior_bias_strength.item())), f, indent=1)
