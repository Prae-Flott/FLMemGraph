#!/usr/bin/env python3
"""
FEDERATED JointPrototypeV31 (V3, and V2 read off its B_node_max
ablation column) on Paderborn, treating the 6 healthy reference bearings
(K001-K006) as 6 federated clients.

## Why K001-K006 as clients (synthetic, not a real deployment fleet)

Paderborn is a single test rig, not a fleet of independently-deployed
machines like robo3er's 5 robots or Sielaff's 10 vending machines -- there
is no real multi-source structure to federate over. K001-K006 ARE,
however, 6 physically distinct bearing units (each with its own full
80-file measurement set, `benchmark/datasets/paderborn_adapter.py`), the
closest thing this dataset has to independent "sources" -- unlike a purely
arbitrary split (e.g. by operating condition or file index), this one
groups measurements that share a genuinely distinct physical rotating
part. It is still labeled synthetic/non-deployment here per the user's
explicit direction, since one test rig sequentially measuring 6 bearings
is not the same claim as 6 independently operating machines.

## Evaluation

Damaged bearings (KA*/KI*/KB*) are separate physical units not owned by
any one client, so each damaged bearing is scored against EVERY client's
final personalized model (own calib-split z-scoring per
`deployment-architecture.md`'s "share weights, calibrate locally"); the
per-bearing number reported is the MEAN AUROC across all 6 clients'
models, and the top-line summary is the mean of that over all 26 damaged
bearings -- directly comparable in form to the centralized script's
"mean AUROC (all 26)".

Otherwise identical to `run_paderborn_joint_prototype_v3.py`: same node
set, declared physics edges, edge types, core hyperparameters.

Usage:
    python3 run_paderborn_v3_1_federated.py
"""
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
from joint_prototype_model import JointPrototypeV31  # noqa: E402
from federated_memory import align_and_split, confidence_weights_from_losses, fedavg_state_dict  # noqa: E402

SYNC_ENCODER_DECODER = True  # opt-in departure from "only exchange memory" -- see
                              # federated_memory.fedavg_state_dict's docstring and
                              # memory/joint-prototype-federated-results.md
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
LOCAL_EPOCHS = 12  # == centralized EPOCHS
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 16
TOP_K = 5
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_TYPED = 0.5
MIN_PROTO_SAMPLES = 5
GAMMA = 0.5
DELTA = 0.5  # matches the other two federated scripts' tuned value for small,
             # non-IID per-client fit sets (each client here is ONE bearing's
             # 80 files, much smaller than the pooled 6-bearing centralized fit set)
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
    return windows.reshape(-1, window_len, f), file_id


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


def per_file_scores(model, arr, batch_size=BATCH_SIZE):
    windows, file_id = make_windows(arr)
    d_proto, d_node, resid_struct, resid_phys, _idx, _r, d_mahal = per_sample_scores(model, windows, batch_size)
    n = len(arr)

    def agg(x):
        out = np.zeros((n,) + x.shape[1:], dtype=np.float64)
        counts = np.zeros(n, dtype=np.int64)
        np.add.at(out, file_id, x)
        np.add.at(counts, file_id, 1)
        shape = (n,) + (1,) * (x.ndim - 1)
        return (out / counts.reshape(shape)).astype(np.float32)

    return agg(d_proto), agg(d_node), agg(resid_struct), agg(resid_phys), agg(d_mahal)


def train_local(model, fit_windows, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fit_windows)),
        batch_size=min(BATCH_SIZE, len(fit_windows)), shuffle=True,
    )
    model.train()
    for _ in range(epochs):
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
    return model


def zscore(x, calib_x):
    median = np.median(calib_x, axis=0)
    q75, q25 = np.percentile(calib_x, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    return (x - median) / iqr


def ablation_scores(z_node, z_struct, z_phys, z_mahal):
    b = z_node.max(axis=1)
    return {
        "B_node_max": b,
        "C_struct_max": z_struct.max(axis=1),
        "E_phys_max": z_phys.max(axis=1),
        "F_v3_max": np.max(np.stack([z_node.max(axis=1), z_struct.max(axis=1),
                                      z_phys.max(axis=1)], axis=1), axis=1),
        "H_cov_mahal": z_mahal,
        "I_node_cov_max": np.maximum(b, z_mahal),
        "J_v3_cov_max": np.max(np.stack([z_node.max(axis=1), z_struct.max(axis=1),
                                          z_phys.max(axis=1), z_mahal], axis=1), axis=1),
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    num_nodes = len(NODE_NAMES)

    print("loading healthy bearings (K001-K006) as 6 federated clients ...")
    clients = []  # list of dicts: code, fit_arr, calib_arr, test_normal_arr, scaler
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

    models = [JointPrototypeV31(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=EMBED_DIM,
                                    num_prototypes=NUM_PROTOTYPES, prior_edges=PRIOR_EDGES,
                                    edge_types=EDGE_TYPES, top_k=TOP_K).to(DEVICE)
              for _ in clients]

    shared_init = {k: v.clone() for k, v in models[0].state_dict().items() if not k.startswith("memory.")}
    for model in models[1:]:
        model.load_state_dict(shared_init, strict=False)

    print(f"\nfederated training: {ROUNDS} rounds x {LOCAL_EPOCHS} local epochs, "
          f"memory exchange (gamma={GAMMA}, delta={DELTA})"
          + (", + FedAvg'd encoder/decoder" if SYNC_ENCODER_DECODER else " only"))
    diagnostics_log = []
    for rnd in range(1, ROUNDS + 1):
        fit_counts = []
        for c, model in zip(clients, models):
            fit_windows, _ = make_windows(c["fit"])
            train_local(model, fit_windows, LOCAL_EPOCHS)
            fit_counts.append(len(fit_windows))

        codebooks = [m.memory.codebook.detach().clone() for m in models]
        usage_counts = [m.memory.usage_count.detach().clone() for m in models]
        P_G, diag = align_and_split(codebooks, usage_counts, gamma=GAMMA, delta=DELTA)
        diag = {k: v for k, v in diag.items() if k != "per_cluster_per_node_agreement"}
        diag["round"] = rnd
        diagnostics_log.append(diag)
        print(f"  round {rnd}: {diag}")

        for model, p_g in zip(models, P_G):
            model.memory.load_memory(p_g)

        if SYNC_ENCODER_DECODER:
            avg = fedavg_state_dict([m.state_dict() for m in models], fit_counts,
                                     prefixes=("encoder.", "decoder."))
            for model in models:
                model.load_state_dict(avg, strict=False)

    print("\nStage B + calibration per client ...")
    calib_stats = []  # per client: (d_proto_calib, d_node_calib, resid_struct_calib, resid_phys_calib, d_mahal_calib)
    component_losses = {"proto": [], "edge": [], "typed": []}
    cov_min = max(MIN_PROTO_SAMPLES, num_nodes + 1)
    n_valid_cov_per_client = []
    for c, model in zip(clients, models):
        calib_windows, _ = make_windows(c["calib"])
        d_proto_w, d_node_b, resid_struct_w, _, calib_idx_w, calib_r_edge_w, _ = per_sample_scores(model, calib_windows)
        model.typed_head.set_calibration(calib_r_edge_w, calib_idx_w, NUM_PROTOTYPES, min_samples=MIN_PROTO_SAMPLES)
        model.cov_head.set_calibration(d_node_b, calib_idx_w, min_samples=cov_min)
        calib_stats.append(per_file_scores(model, c["calib"]))
        n_valid_cov_per_client.append(int(model.cov_head.calib_valid.sum()))
        # calib-time mean of each component's OWN raw training-objective quantity -- see
        # federated_memory.confidence_weights_from_losses's docstring.
        component_losses["proto"].append(float(d_proto_w.mean()))
        component_losses["edge"].append(float(resid_struct_w.mean()))
        component_losses["typed"].append(float(calib_r_edge_w.mean()))

    client_weights = confidence_weights_from_losses(component_losses, temperature=1.0)
    for c, w in zip(clients, client_weights):
        print(f"  {c['code']} confidence weights: proto={w['proto']:.3f} edge={w['edge']:.3f} typed={w['typed']:.3f}")

    print("\nscoring damaged bearings against every client's personalized model ...")
    report = {"config": {"embed_dim": EMBED_DIM, "num_prototypes": NUM_PROTOTYPES, "top_k": TOP_K,
                           "window_len": WINDOW_LEN, "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS,
                           "gamma": GAMMA, "delta": DELTA, "prior_edges": EDGES_NAMED, "edge_types": EDGE_TYPES,
                           "min_proto_samples": MIN_PROTO_SAMPLES, "clients": HEALTHY_CODES,
                           "sync_encoder_decoder": SYNC_ENCODER_DECODER},
               "nodes": NODE_NAMES, "alignment_log": diagnostics_log,
               "client_confidence_weights": {c["code"]: w for c, w in zip(clients, client_weights)},
               "bearings": {}}

    # normal-side (calib-held-out test_normal, per client, scored on that SAME client)
    normal_scores_per_client = []
    for c, model, calib in zip(clients, models, calib_stats):
        d_proto_calib, d_node_calib, resid_struct_calib, resid_phys_calib, d_mahal_calib = calib
        d_proto_n, d_node_n, resid_struct_n, resid_phys_n, d_mahal_n = per_file_scores(model, c["test_normal"])
        z = ablation_scores(zscore(d_node_n, d_node_calib),
                              zscore(resid_struct_n, resid_struct_calib),
                              zscore(resid_phys_n, resid_phys_calib),
                              zscore(d_mahal_n, d_mahal_calib))
        normal_scores_per_client.append(z)

    rows = {k: [] for k in normal_scores_per_client[0]}
    for code in ALL_DAMAGED_CODES:
        d = load_bearing_with_physics(code)
        arr = stack_nodes(d)
        cat, origin = category_of(code), damage_origin_of(code)

        per_client_auroc = {k: [] for k in rows}
        for c, model, calib, scores_normal in zip(clients, models, calib_stats, normal_scores_per_client):
            arr_scaled = c["scaler"].transform(arr.reshape(-1, num_nodes)).reshape(arr.shape).astype(np.float32)
            d_proto_calib, d_node_calib, resid_struct_calib, resid_phys_calib, d_mahal_calib = calib
            d_proto_f, d_node_f, resid_struct_f, resid_phys_f, d_mahal_f = per_file_scores(model, arr_scaled)
            scores_fault = ablation_scores(zscore(d_node_f, d_node_calib),
                                             zscore(resid_struct_f, resid_struct_calib),
                                             zscore(resid_phys_f, resid_phys_calib),
                                             zscore(d_mahal_f, d_mahal_calib))
            for key in rows:
                labels = np.concatenate([np.zeros(len(scores_normal[key])), np.ones(len(scores_fault[key]))])
                s = np.concatenate([scores_normal[key], scores_fault[key]])
                per_client_auroc[key].append(float(sk_metrics.roc_auc_score(labels, s)))

        mean_across_clients = {k: float(np.mean(v)) for k, v in per_client_auroc.items()}
        for key in rows:
            rows[key].append(mean_across_clients[key])
        report["bearings"][code] = {"category": cat, "origin": origin, "n": len(arr),
                                      "per_client_auroc": {clients[i]["code"]: {k: v[i] for k, v in per_client_auroc.items()}
                                                             for i in range(len(clients))},
                                      "mean_across_clients": mean_across_clients}
        print(f"  {code:<6}{cat:<12}{origin:<11}" + "  ".join(f"{k}={v:.3f}" for k, v in mean_across_clients.items()))

    print(f"\n{'method':<24}{'mean AUROC (all 26, mean over clients)':>40}")
    summary = {}
    for key in rows:
        mean_auroc = float(np.mean(rows[key]))
        summary[key] = mean_auroc
        print(f"{key:<24}{mean_auroc:>40.3f}")
    print(f"\n{'method':<24}{'outer_ring':>12}{'inner_ring':>12}{'combined':>12}")
    category_summary = {}
    for key in rows:
        cat_means = {}
        for cat in ["outer_ring", "inner_ring", "combined"]:
            subset = [report["bearings"][code]["mean_across_clients"][key] for code in ALL_DAMAGED_CODES
                      if category_of(code) == cat]
            cat_means[cat] = float(np.mean(subset)) if subset else None
        category_summary[key] = cat_means
        print(f"{key:<24}" + "".join(f"{cat_means[c]:>12.3f}" for c in ["outer_ring", "inner_ring", "combined"]))

    report["summary_mean_auroc"] = summary
    report["category_summary"] = category_summary
    report["n_valid_prototypes_for_cov_calib"] = n_valid_cov_per_client
    suffix = "_encdec_synced" if SYNC_ENCODER_DECODER else ""
    out_json = OUT_DIR / f"paderborn_v3_1_federated{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
                OUT_DIR / f"paderborn_v3_1_federated{suffix}.pth")
    print(f"\nsaved -> {out_json}")


if __name__ == "__main__":
    main()
