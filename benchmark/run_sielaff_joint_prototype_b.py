#!/usr/bin/env python3
"""
Joint Prototype Memory on Sielaff -- "path B" ONLY (prototype + per-node
deviation, `A_prototype_only` + `B_prototype_plus_node` from the
Paderborn/robo3er/voraus-AD JointPrototypeGDNv3 ablation naming), with NO
edge/relation head at all.

This is a deliberate, well-motivated scope, not a simplification for its
own sake: `benchmark/datasets/sielaff_physics.md` documents that this
dataset has NO verified physical prior -- no declared edges exist to feed
`TrendGraphAttentionHead`/`TypedRelationAnomalyHead` (v2/v3's edge
mechanisms), so running the full `JointPrototypeGDNv2`/`v3` would mean
either leaving `prior_edges=None` (pure GDN-style generic attention, not
actually testing anything about Joint Prototype's node/edge distinction)
or inventing edges with no physical justification, which this project
has consistently avoided doing elsewhere. Path B needs NEITHER of those
-- it only needs `SharedEncoder` + `JointPrototypeMemory`, reused directly
from `src/joint_prototype_model.py` (no new model code). This directly
answers the cross-dataset finding from voraus-AD/robo3er
(`memory/joint-prototype-scheme-v3.md`, which also documents Scheme B's
generalization back to `memory/joint-prototype-scheme-b.md`):
path B (node-level deviation from the matched joint prototype) wins or
ties on the vast majority of fault categories on BOTH those datasets
EVEN WHEN edges/relations were available -- so testing it here, where no
edges exist at all, is the most direct apples-to-apples way to see
whether Joint Prototype Memory's core idea (has this multi-sensor joint
state been seen before?) adds anything over Sielaff's existing plain-GDN
baseline (`run_sielaff_gdn.py`, `checkpoints/sielaff/gdn_sielaff_report.json`),
without needing a physics prior this dataset doesn't have.

Same data loading, per-machine fit/calib/test-normal split, and
RELIABILITY_RATIO sparse-feature masking as `run_sielaff_gdn.py` (several
Sielaff sensors, e.g. RingCamera features, are only reported by a subset
of the 10 machines -- their calib-split IQR is then near-zero, and
including them in a max() over nodes would let trivial fluctuations
dominate). Window is the full [8, 39] sample (no dense-pairing needed --
Joint Prototype reconstructs the whole window per node, it doesn't
forecast the next step the way GDN does).

Usage:
    python3 run_sielaff_joint_prototype_b.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics as sk_metrics
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from joint_prototype_model import SharedEncoder, JointPrototypeMemory  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "sielaff"
OUT_DIR = REPO_ROOT / "checkpoints" / "sielaff"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 16
WINDOW_LEN = 8
BETA = 0.25
RELIABILITY_RATIO = 0.05  # matches run_sielaff_gdn.py -- features w/ calib IQR < 5% of
                          # median IQR are excluded from the node-level max()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JointPrototypeMemoryOnly(nn.Module):
    """Encoder + Joint Prototype Memory + a training-only decoder -- NO
    edge head. Produces exactly A (`d_G`) and the per-node deviation
    (`s_node`) that path B combines with A, nothing else."""

    def __init__(self, num_nodes: int, window_size: int, embed_dim: int, num_prototypes: int):
        super().__init__()
        self.encoder = SharedEncoder(window_size, embed_dim)
        self.memory = JointPrototypeMemory(num_prototypes, num_nodes, embed_dim)
        self.decoder = nn.Linear(embed_dim, window_size)

    def forward(self, x, training_mode=False):
        z = self.encoder(x)
        out = self.memory(z)
        out["z"] = z
        if training_mode:
            out["x_hat"] = self.decoder(z).transpose(1, 2)
        return out


def load_machine_index_sets():
    with open(DATA_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    machine_indices = []
    for client_id in range(partition["separation"]["total"]):
        d = partition["data_indices"][client_id]
        idx = np.concatenate([d["train"], d["val"], d["test"]]).astype(np.int64)
        machine_indices.append(np.sort(idx))
    return machine_indices


def load_and_split():
    data = np.load(DATA_DIR / "data.npy")
    targets = np.load(DATA_DIR / "targets.npy")
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    cols = meta["feature_columns"]
    label_map = meta["class_names"]

    machine_indices = load_machine_index_sets()
    fit_idx, calib_idx, test_normal_idx = [], [], []
    for idx in machine_indices:
        normal_idx = idx[targets[idx] == 0]
        n = len(normal_idx)
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        fit_idx.append(normal_idx[:n_fit])
        calib_idx.append(normal_idx[n_fit : n_fit + n_calib])
        test_normal_idx.append(normal_idx[n_fit + n_calib :])

    fit_idx = np.concatenate(fit_idx)
    calib_idx = np.concatenate(calib_idx)
    test_normal_idx = np.concatenate(test_normal_idx)
    return data, targets, cols, label_map, fit_idx, calib_idx, test_normal_idx


def fit_scaler(data, fit_idx):
    scaler = StandardScaler()
    scaler.fit(data[fit_idx].reshape(-1, data.shape[-1]))
    return scaler


def scale(windows, scaler):
    n, t, f = windows.shape
    return scaler.transform(windows.reshape(-1, f)).reshape(n, t, f).astype(np.float32)


@torch.no_grad()
def per_sample_scores(model, windows, batch_size=BATCH_SIZE):
    model.eval()
    d_G_all, s_node_all = [], []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(DEVICE)
        out = model(batch, training_mode=False)
        d_G_all.append(out["d_G"].cpu().numpy())
        s_node_all.append(out["s_node"].cpu().numpy())
    return np.concatenate(d_G_all), np.concatenate(s_node_all)


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
            loss = l_pred + BETA * l_me + l_mc
            loss.backward()
            optimizer.step()
            fit_loss += loss.item() * batch.size(0)
        fit_loss /= len(fit_arr)

        d_G, _ = per_sample_scores(model, calib_arr)
        calib_mse = float(d_G.mean())
        print(f"  epoch {epoch:2d}  fit_loss={fit_loss:.6f}  calib_dG_mean={calib_mse:.6f}  "
              f"codebook_util={model.memory.codebook_utilization():.2f}")
        if calib_mse < best_calib_mse:
            best_calib_mse = calib_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def zscore(x, calib_x):
    median = np.median(calib_x, axis=0)
    q75, q25 = np.percentile(calib_x, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    return (x - median) / iqr, iqr


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("loading Sielaff ...")
    data, targets, cols, label_map, fit_idx, calib_idx, test_normal_idx = load_and_split()
    num_nodes = len(cols)
    scaler = fit_scaler(data, fit_idx)
    fit_arr = scale(data[fit_idx], scaler)
    calib_arr = scale(data[calib_idx], scaler)
    test_normal_arr = scale(data[test_normal_idx], scaler)
    print(f"  fit={fit_arr.shape} calib={calib_arr.shape} test_normal={test_normal_arr.shape} "
          f"nodes(sensors)={num_nodes}")

    print(f"\ntraining JointPrototypeMemoryOnly (nodes={num_nodes}, M={NUM_PROTOTYPES}, "
          f"embed_dim={EMBED_DIM}, window_len={WINDOW_LEN}) -- path B only, no edges ...")
    model = JointPrototypeMemoryOnly(num_nodes=num_nodes, window_size=WINDOW_LEN,
                                       embed_dim=EMBED_DIM, num_prototypes=NUM_PROTOTYPES).to(DEVICE)
    model = train(model, fit_arr, calib_arr)

    print("\ncalibrating from calib split ...")
    d_G_calib, s_node_calib = per_sample_scores(model, calib_arr)
    d_G_normal, s_node_normal = per_sample_scores(model, test_normal_arr)

    _, node_iqr = zscore(s_node_normal, s_node_calib)
    median_iqr = np.median(node_iqr)
    reliable_mask = node_iqr >= RELIABILITY_RATIO * median_iqr
    excluded = [cols[i] for i in range(num_nodes) if not reliable_mask[i]]
    print(f"excluded {len(excluded)}/{num_nodes} sparsely-reported nodes from the max() "
          f"(calib IQR < {RELIABILITY_RATIO:.0%} of median IQR): {excluded}")

    z_dG_normal, _ = zscore(d_G_normal, d_G_calib)
    z_node_normal, _ = zscore(s_node_normal, s_node_calib)
    z_node_normal_masked = z_node_normal[:, reliable_mask]

    def scores(z_dG, z_node_masked):
        return {
            "A_prototype_only": z_dG,
            "B_prototype_plus_node": np.maximum(z_dG, z_node_masked.max(axis=1)),
        }

    scores_normal = scores(z_dG_normal, z_node_normal_masked)

    print("\nscoring fault types ...")
    report = {"config": {"embed_dim": EMBED_DIM, "num_prototypes": NUM_PROTOTYPES,
                           "window_len": WINDOW_LEN, "epochs": EPOCHS,
                           "reliability_ratio": RELIABILITY_RATIO},
               "nodes": cols, "excluded_nodes": excluded, "fault_types": {}}
    rows = {k: [] for k in scores_normal}
    for label_id_str, name in label_map.items():
        label_id = int(label_id_str)
        if label_id == 0:
            continue
        fault_idx = np.where(targets == label_id)[0]
        if len(fault_idx) == 0:
            continue
        arr = scale(data[fault_idx], scaler)
        d_G_f, s_node_f = per_sample_scores(model, arr)
        z_dG_f, _ = zscore(d_G_f, d_G_calib)
        z_node_f, _ = zscore(s_node_f, s_node_calib)
        z_node_f_masked = z_node_f[:, reliable_mask]
        scores_fault = scores(z_dG_f, z_node_f_masked)

        aurocs = {}
        for key in scores_normal:
            labels = np.concatenate([np.zeros(len(scores_normal[key])), np.ones(len(scores_fault[key]))])
            s = np.concatenate([scores_normal[key], scores_fault[key]])
            auroc = float(sk_metrics.roc_auc_score(labels, s))
            aurocs[key] = auroc
            rows[key].append((name, auroc))
        report["fault_types"][name] = {"n": int(len(arr)), **aurocs}
        print(f"  {name:<20}n={len(arr):<5}" + "  ".join(f"{k}={v:.3f}" for k, v in aurocs.items()))

    print(f"\n{'method':<24}{'mean AUROC (5 fault types)':>28}")
    summary = {}
    for key in scores_normal:
        mean_auroc = float(np.mean([r[1] for r in rows[key]]))
        summary[key] = mean_auroc
        print(f"{key:<24}{mean_auroc:>28.3f}")

    report["summary_mean_auroc"] = summary
    with open(OUT_DIR / "sielaff_joint_prototype_b_report.json", "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "report": report},
                OUT_DIR / "sielaff_joint_prototype_b.pth")
    print(f"\nsaved -> {OUT_DIR / 'sielaff_joint_prototype_b_report.json'}")


if __name__ == "__main__":
    main()
