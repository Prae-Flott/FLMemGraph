#!/usr/bin/env python3
"""
Joint Prototype Memory + Physics-Relation GDN + Anomaly Attention (v3), per
`docs/joint_prototype_physics_gdn_anomaly_attention_prompt.md`.

Adds, on top of v2's generic learned-attention edge head
(`TrendGraphAttentionHead`, kept unchanged and still scored as its own
ablation column below):

- `TypedRelationAnomalyHead` -- the same 8 declared physics edges as v1/v2,
  but each edge now uses a relation-TYPE-appropriate message function
  (Sec 9): "proportional" edges (current <- speed/torque, the doc's own
  textbook example) get a linear map; "nonlinear"/"dynamic" edges
  (vibration/torque <- force/speed) get a small MLP.
- Prototype-conditioned edge-residual standardization (Sec 12): after
  training, a calibration pass computes per-(prototype, edge) mean/std of
  the raw residual, falling back to a global mean/std for sparsely
  populated prototypes.
- An unsupervised anomaly attention (Sec 16) over each node's incoming
  declared edges, producing `s_node_typed` -- a node-level "which
  relationship broke" score, structurally separate from v2's
  `s_edge` ("who matters normally" attention).

The self-supervised relation-breaking-augmentation training (Sec 17) and
full 3-stage curriculum (Sec 22) are NOT implemented here -- see
`TypedRelationAnomalyHead`'s docstring for why this is a deliberate,
documented scoping decision, not an oversight.

Same 6 nodes, same fit/calib/split, same core hyperparameters as v1/v2 for
direct comparability. Ablations A-D reproduce v2 exactly (same signals);
E and F are new (typed-edge-only, and the full v3 combination).

Usage:
    python3 run_paderborn_joint_prototype_v3.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_prototype_model import JointPrototypeGDNv3  # noqa: E402
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
# relation type per edge, in the same order as EDGES_NAMED -- domain-knowledge
# assignment (Sec 8 "Level 2: relation-type knowledge"): motor current relates to
# torque/speed close to linearly (the design doc's own "Current -> Torque,
# proportional" example); vibration and torque response to load/speed are genuinely
# nonlinear (contact mechanics, induction-motor torque-speed curve).
EDGE_TYPES = [
    "nonlinear", "nonlinear",       # force -> vibration_1, force -> torque
    "nonlinear", "nonlinear",       # speed -> vibration_1, speed -> torque
    "proportional", "proportional", # speed -> phase_current_1/2
    "proportional", "proportional", # torque -> phase_current_1/2
]
PRIOR_EDGES = [(NODE_IDX[s], NODE_IDX[d]) for s, d in EDGES_NAMED]

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15
WINDOW_LEN = 64
WINDOW_STRIDE = 32
BATCH_SIZE = 256
EPOCHS = 12
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 16
TOP_K = 5
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_TYPED = 0.5
MIN_PROTO_SAMPLES = 5
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
    d_G_all, s_node_all, s_edge_all, s_typed_all, idx_all, r_all = [], [], [], [], [], []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(DEVICE)
        out = model(batch, training_mode=False)
        d_G_all.append(out["d_G"].cpu().numpy())
        s_node_all.append(out["s_node"].cpu().numpy())
        s_edge_all.append(out["s_edge"].cpu().numpy())
        s_typed_all.append(out["s_node_typed"].cpu().numpy())
        idx_all.append(out["idx"].cpu().numpy())
        r_all.append(out["r"].cpu().numpy())
    return (np.concatenate(d_G_all), np.concatenate(s_node_all), np.concatenate(s_edge_all),
            np.concatenate(s_typed_all), np.concatenate(idx_all), np.concatenate(r_all))


def per_file_scores(model, arr, batch_size=BATCH_SIZE):
    windows, file_id = make_windows(arr)
    d_G, s_node, s_edge, s_typed, _idx, _r = per_sample_scores(model, windows, batch_size)
    n = len(arr)

    def agg(x):
        out = np.zeros((n,) + x.shape[1:], dtype=np.float64)
        counts = np.zeros(n, dtype=np.int64)
        np.add.at(out, file_id, x)
        np.add.at(counts, file_id, 1)
        shape = (n,) + (1,) * (x.ndim - 1)
        return (out / counts.reshape(shape)).astype(np.float32)

    return agg(d_G), agg(s_node), agg(s_edge), agg(s_typed)


def train(model, fit_arr, calib_arr):
    torch.manual_seed(SEED)
    fit_windows, _ = make_windows(fit_arr)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fit_windows)),
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
            l_edge = out["s_edge"].mean()
            l_typed = out["r"].mean()
            loss = l_pred + BETA * l_me + l_mc + LAMBDA_EDGE * l_edge + LAMBDA_TYPED * l_typed
            loss.backward()
            optimizer.step()
            fit_loss += loss.item() * batch.size(0)
        fit_loss /= len(fit_windows)

        d_G, _, _, _ = per_file_scores(model, calib_arr)
        calib_mse = float(d_G.mean())
        print(f"  epoch {epoch:2d}  fit_loss={fit_loss:.6f}  calib_dG_mean={calib_mse:.6f}  "
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
    num_nodes = len(NODE_NAMES)

    print("loading healthy bearings (K001-K006), 6 channels ...")
    healthy_data = {code: load_bearing_with_physics(code) for code in HEALTHY_CODES}

    fit_list, calib_list, test_list = [], [], []
    for code, d in healthy_data.items():
        n = len(d["vibration_1"])
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        arr = stack_nodes(d)
        fit_list.append(arr[:n_fit])
        calib_list.append(arr[n_fit : n_fit + n_calib])
        test_list.append(arr[n_fit + n_calib :])
        print(f"  {code}: n={n} fit={n_fit} calib={n_calib} test_normal={n - n_fit - n_calib}")

    fit_arr = np.concatenate(fit_list, axis=0)
    calib_arr = np.concatenate(calib_list, axis=0)
    test_normal_arr = np.concatenate(test_list, axis=0)

    scaler = StandardScaler().fit(fit_arr.reshape(-1, num_nodes))

    def scale(a):
        n, t, f = a.shape
        return scaler.transform(a.reshape(-1, f)).reshape(n, t, f).astype(np.float32)

    fit_scaled, calib_scaled, test_normal_scaled = scale(fit_arr), scale(calib_arr), scale(test_normal_arr)

    print(f"\ntraining JointPrototypeGDNv3 (nodes={num_nodes}, prior_edges={len(PRIOR_EDGES)}, "
          f"edge_types={EDGE_TYPES}, top_k={TOP_K}, M={NUM_PROTOTYPES}, embed_dim={EMBED_DIM}) ...")
    model = JointPrototypeGDNv3(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=EMBED_DIM,
                                  num_prototypes=NUM_PROTOTYPES, prior_edges=PRIOR_EDGES,
                                  edge_types=EDGE_TYPES, top_k=TOP_K).to(DEVICE)
    model = train(model, fit_scaled, calib_scaled)

    print("\nStage B: estimating prototype-conditioned edge-residual statistics from calib split ...")
    calib_windows, _ = make_windows(calib_scaled)
    _, _, _, _, calib_idx_per_window, calib_r_per_window = per_sample_scores(model, calib_windows)
    model.typed_head.set_calibration(calib_r_per_window, calib_idx_per_window, NUM_PROTOTYPES,
                                      min_samples=MIN_PROTO_SAMPLES)
    n_valid_proto = int(model.typed_head.calib_valid_proto.sum())
    print(f"  {n_valid_proto}/{NUM_PROTOTYPES} prototypes had >= {MIN_PROTO_SAMPLES} calib windows "
          f"(rest fall back to global edge stats)")

    print("\ncalibrating from healthy calib split ...")
    d_G_calib, s_node_calib, s_edge_calib, s_typed_calib = per_file_scores(model, calib_scaled)
    d_G_normal, s_node_normal, s_edge_normal, s_typed_normal = per_file_scores(model, test_normal_scaled)

    z_dG_normal = zscore(d_G_normal, d_G_calib)
    z_node_normal = zscore(s_node_normal, s_node_calib)
    z_edge_normal = zscore(s_edge_normal, s_edge_calib)
    z_typed_normal = zscore(s_typed_normal, s_typed_calib)

    def ablation_scores(z_dG, z_node, z_edge, z_typed):
        return {
            "A_prototype_only": z_dG,
            "B_prototype_plus_node": np.maximum(z_dG, z_node.max(axis=1)),
            "C_edge_only": z_edge.max(axis=1),
            "D_full_v2": np.maximum(np.maximum(z_dG, z_node.max(axis=1)), z_edge.max(axis=1)),
            "E_typed_edge_only": z_typed.max(axis=1),
            "F_full_v3": np.max(np.stack([z_dG, z_node.max(axis=1), z_edge.max(axis=1),
                                            z_typed.max(axis=1)], axis=1), axis=1),
        }

    scores_normal = ablation_scores(z_dG_normal, z_node_normal, z_edge_normal, z_typed_normal)

    print("\nscoring damaged bearings ...")
    report = {"config": {"embed_dim": EMBED_DIM, "num_prototypes": NUM_PROTOTYPES, "top_k": TOP_K,
                           "window_len": WINDOW_LEN, "epochs": EPOCHS, "prior_edges": EDGES_NAMED,
                           "edge_types": EDGE_TYPES, "min_proto_samples": MIN_PROTO_SAMPLES},
               "nodes": NODE_NAMES, "final_prior_bias_strength": model.edge_head.prior_bias_strength.item(),
               "final_typed_temperature": model.typed_head.log_temperature.exp().item(),
               "n_valid_prototypes_for_typed_calib": n_valid_proto,
               "bearings": {}}
    rows = {k: [] for k in scores_normal}
    for code in ALL_DAMAGED_CODES:
        d = load_bearing_with_physics(code)
        arr = scale(stack_nodes(d))
        d_G_f, s_node_f, s_edge_f, s_typed_f = per_file_scores(model, arr)
        z_dG_f = zscore(d_G_f, d_G_calib)
        z_node_f = zscore(s_node_f, s_node_calib)
        z_edge_f = zscore(s_edge_f, s_edge_calib)
        z_typed_f = zscore(s_typed_f, s_typed_calib)
        scores_fault = ablation_scores(z_dG_f, z_node_f, z_edge_f, z_typed_f)

        cat, origin = category_of(code), damage_origin_of(code)
        aurocs = {}
        for key in scores_normal:
            labels = np.concatenate([np.zeros(len(scores_normal[key])), np.ones(len(scores_fault[key]))])
            scores = np.concatenate([scores_normal[key], scores_fault[key]])
            auroc = float(sk_metrics.roc_auc_score(labels, scores))
            aurocs[key] = auroc
            rows[key].append((code, cat, origin, auroc))
        report["bearings"][code] = {"category": cat, "origin": origin, "n": len(arr), **aurocs}
        print(f"  {code:<6}{cat:<12}{origin:<11}" + "  ".join(f"{k}={v:.3f}" for k, v in aurocs.items()))

    print(f"\n{'method':<24}{'mean AUROC (all 26)':>22}")
    summary = {}
    for key in scores_normal:
        mean_auroc = float(np.mean([r[3] for r in rows[key]]))
        summary[key] = mean_auroc
        print(f"{key:<24}{mean_auroc:>22.3f}")
    print(f"\n{'method':<24}{'outer_ring':>12}{'inner_ring':>12}{'combined':>12}")
    category_summary = {}
    for key in scores_normal:
        cat_means = {}
        for cat in ["outer_ring", "inner_ring", "combined"]:
            subset = [r[3] for r in rows[key] if r[1] == cat]
            cat_means[cat] = float(np.mean(subset)) if subset else None
        category_summary[key] = cat_means
        print(f"{key:<24}" + "".join(f"{cat_means[c]:>12.3f}" for c in ["outer_ring", "inner_ring", "combined"]))

    print(f"\nfinal prior_bias_strength: {model.edge_head.prior_bias_strength.item():.4f}")
    print(f"final typed-edge attention temperature: {model.typed_head.log_temperature.exp().item():.4f}")

    report["summary_mean_auroc"] = summary
    report["category_summary"] = category_summary
    with open(OUT_DIR / "paderborn_joint_prototype_v3_report.json", "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "report": report},
                OUT_DIR / "paderborn_joint_prototype_v3.pth")
    print(f"\nsaved -> {OUT_DIR / 'paderborn_joint_prototype_v3_report.json'}")


if __name__ == "__main__":
    main()
