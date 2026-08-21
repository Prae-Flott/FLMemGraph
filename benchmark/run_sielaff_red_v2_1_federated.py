#!/usr/bin/env python3
"""
FEDERATED Joint Prototype Memory ("V2" -- no edge head, see signal B in
`memory/scoring-signals-B-C-E-H.md`) on Sielaff's 10 real reverse-vending
machines as 10 federated clients, RED-SEVERITY faults only.

Same protocol as `run_sielaff_joint_prototype_v2_federated.py` (encoder
stays fully local per client, ONLY the `JointPrototypeMemory` codebook is
exchanged each round via `federated_memory.align_and_split`), but trained
and evaluated on `data/sielaff_red/` -- built by
`benchmark/datasets/build_sielaff_red.py` directly from the raw log tables,
labeling each 4h sliding window (8 x 30min buckets, matching the
severity_4 scheme in `data/sielaff_data/sielaff_ground_truth.md`) as:
  0 = no_error (no logged error of any severity in the window)
  1 = red (>=1 of the 47 RED-severity error IDs in the window)
  2 = non_red_error (only orange/green errors -- reported as a contrast
      class, excluded from the "normal" fit/calib reference set)

V2-only (no V3 run exists for Sielaff): `benchmark/datasets/sielaff_physics.md`
documents that Sielaff has NO verified physical prior, so there are no
declared edges to feed a typed-relation head.

Client evaluation: each of the 10 machines calibrates its OWN z-scores
from its OWN calib split after the final round (share weights, calibrate
locally), then scores its OWN fault windows. `summary_mean_auroc_overall`
is the unweighted mean across all (machine, fault_type) pairs actually
evaluated (both "red" and the "non_red_error" contrast class).

Usage:
    python3 build_sielaff_red.py   # once, to build data/sielaff_red/
    python3 run_sielaff_red_joint_prototype_v2_federated.py
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
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "federated"))
from joint_prototype_model import SharedEncoder, JointPrototypeMemory  # noqa: E402
from federated_memory import align_and_split  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "sielaff_red"
OUT_DIR = REPO_ROOT / "checkpoints" / "sielaff"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15
BATCH_SIZE = 256
ROUNDS = 5
LOCAL_EPOCHS = 30  # == centralized EPOCHS
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 16
WINDOW_LEN = 8
BETA = 0.25
GAMMA = 0.5
DELTA = 0.5  # same as robo3er's federated setting -- small per-client normal-fit
             # sets here too (each of 10 machines, not one pooled 5000+ set)
RELIABILITY_RATIO = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JointPrototypeMemoryOnly(nn.Module):
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

    clients = []
    for client_id in range(partition["separation"]["total"]):
        d = partition["data_indices"][client_id]
        idx = np.sort(np.concatenate([d["train"], d["val"], d["test"]]).astype(np.int64))
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
    return data, targets, cols, label_map, clients


def fit_client_scaler(data, fit_idx):
    scaler = StandardScaler()
    scaler.fit(data[fit_idx].reshape(-1, data.shape[-1]))
    return scaler


def scale_client(data, scaler, idx):
    n, t, f = data[idx].shape
    return scaler.transform(data[idx].reshape(-1, f)).reshape(n, t, f).astype(np.float32)


@torch.no_grad()
def per_sample_scores(model, windows, batch_size=BATCH_SIZE):
    model.eval()
    d_proto_all, d_node_all = [], []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(DEVICE)
        out = model(batch, training_mode=False)
        d_proto_all.append(out["d_proto"].cpu().numpy())
        d_node_all.append(out["d_node"].cpu().numpy())
    return np.concatenate(d_proto_all), np.concatenate(d_node_all)


def train_local(model, fit_arr, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fit_arr)),
        batch_size=min(BATCH_SIZE, len(fit_arr)), shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch, training_mode=True)
            l_pred = torch.nn.functional.mse_loss(out["x_hat"], batch)
            l_me, l_mc = model.memory.commitment_losses(out["z"], out["p_star"])
            loss = l_pred + BETA * l_me + l_mc
            loss.backward()
            optimizer.step()
    return model


def zscore(x, calib_x):
    median = np.median(calib_x, axis=0)
    q75, q25 = np.percentile(calib_x, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    return (x - median) / iqr, iqr


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("loading Sielaff, splitting into 10 real per-machine federated clients ...")
    data, targets, cols, label_map, clients = build_clients()
    num_nodes = len(cols)
    for c in clients:
        print(f"  client {c.client_id}: fit={len(c.fit_idx)} calib={len(c.calib_idx)} "
              f"test_normal={len(c.test_normal_idx)}")

    scalers = [fit_client_scaler(data, c.fit_idx) for c in clients]
    models = [JointPrototypeMemoryOnly(num_nodes=num_nodes, window_size=WINDOW_LEN,
                                         embed_dim=EMBED_DIM, num_prototypes=NUM_PROTOTYPES).to(DEVICE)
              for _ in clients]

    shared_init = {k: v.clone() for k, v in models[0].state_dict().items() if not k.startswith("memory.")}
    for model in models[1:]:
        model.load_state_dict(shared_init, strict=False)

    print(f"\nfederated training: {ROUNDS} rounds x {LOCAL_EPOCHS} local epochs, "
          f"memory-only exchange (gamma={GAMMA}, delta={DELTA})")
    diagnostics_log = []
    for rnd in range(1, ROUNDS + 1):
        for c, model, scaler in zip(clients, models, scalers):
            fit_w = scale_client(data, scaler, c.fit_idx)
            train_local(model, fit_w, LOCAL_EPOCHS)

        codebooks = [m.memory.codebook.detach().clone() for m in models]
        usage_counts = [m.memory.usage_count.detach().clone() for m in models]
        P_G, diag = align_and_split(codebooks, usage_counts, gamma=GAMMA, delta=DELTA)
        diag = {k: v for k, v in diag.items() if k != "per_cluster_per_node_agreement"}
        diag["round"] = rnd
        diagnostics_log.append(diag)
        print(f"  round {rnd}: {diag}")

        for model, p_g in zip(models, P_G):
            model.memory.load_memory(p_g)

    report = {"config": {"rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "num_prototypes": NUM_PROTOTYPES,
                          "embed_dim": EMBED_DIM, "window_len": WINDOW_LEN, "gamma": GAMMA, "delta": DELTA,
                          "reliability_ratio": RELIABILITY_RATIO},
              "nodes": cols, "alignment_log": diagnostics_log, "clients": {}}

    all_pairs = {"B_node_max": []}
    all_pairs_by_name = {}
    for c, model, scaler in zip(clients, models, scalers):
        calib_arr = scale_client(data, scaler, c.calib_idx)
        test_normal_arr = scale_client(data, scaler, c.test_normal_idx)
        d_proto_calib, d_node_calib = per_sample_scores(model, calib_arr)
        d_proto_normal, d_node_normal = per_sample_scores(model, test_normal_arr)

        _, node_iqr = zscore(d_node_normal, d_node_calib)
        median_iqr = np.median(node_iqr)
        reliable_mask = node_iqr >= RELIABILITY_RATIO * median_iqr

        z_node_normal, _ = zscore(d_node_normal, d_node_calib)
        z_node_normal_masked = z_node_normal[:, reliable_mask]
        scores_normal = {
            "B_node_max": z_node_normal_masked.max(axis=1),
        }

        client_report = {"excluded_nodes": [cols[i] for i in range(num_nodes) if not reliable_mask[i]],
                          "fault_types": {}}
        rows = {k: [] for k in scores_normal}
        for label_id_str, name in label_map.items():
            label_id = int(label_id_str)
            if label_id == 0:
                continue
            fault_idx = np.where(targets == label_id)[0]
            # keep only this machine's own faults, matching this client's own partition
            fault_idx = np.intersect1d(fault_idx, c.all_idx)
            if len(fault_idx) == 0:
                continue
            fault_arr = scale_client(data, scaler, fault_idx)
            d_proto_f, d_node_f = per_sample_scores(model, fault_arr)
            z_node_f, _ = zscore(d_node_f, d_node_calib)
            z_node_f_masked = z_node_f[:, reliable_mask]
            scores_fault = {
                "B_node_max": z_node_f_masked.max(axis=1),
            }

            aurocs = {}
            for key in scores_normal:
                labels = np.concatenate([np.zeros(len(scores_normal[key])), np.ones(len(scores_fault[key]))])
                s = np.concatenate([scores_normal[key], scores_fault[key]])
                auroc = float(sk_metrics.roc_auc_score(labels, s))
                aurocs[key] = auroc
                rows[key].append(auroc)
                all_pairs[key].append(auroc)
                all_pairs_by_name.setdefault(name, {"B_node_max": []})
                all_pairs_by_name[name][key].append(auroc)
            client_report["fault_types"][name] = {"n": int(len(fault_idx)), **aurocs}
            print(f"  machine{c.client_id:<3}{name:<20}n={len(fault_idx):<5}"
                  + "  ".join(f"{k}={v:.3f}" for k, v in aurocs.items()))

        if rows["B_node_max"]:
            client_report["client_mean_auroc"] = {k: float(np.mean(v)) for k, v in rows.items()}
        report["clients"][f"machine{c.client_id}"] = client_report

    summary_overall = {k: float(np.mean(v)) for k, v in all_pairs.items()}
    summary_by_name = {name: {k: float(np.mean(v)) for k, v in d.items()}
                        for name, d in all_pairs_by_name.items()}
    print(f"\n{'method':<24}{'mean AUROC (all machine-fault pairs)':>36}")
    for key, v in summary_overall.items():
        print(f"{key:<24}{v:>36.3f}")
    print(f"\n{'fault_type':<16}{'method':<24}{'mean AUROC':>12}")
    for name, d in summary_by_name.items():
        for key, v in d.items():
            print(f"{name:<16}{key:<24}{v:>12.3f}")

    report["summary_mean_auroc_overall"] = summary_overall
    report["summary_mean_auroc_by_fault_type"] = summary_by_name
    with open(OUT_DIR / "sielaff_red_v2_1_federated_report.json", "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
                OUT_DIR / "sielaff_red_v2_1_federated.pth")
    print(f"\nsaved -> {OUT_DIR / 'sielaff_red_v2_1_federated_report.json'}")


if __name__ == "__main__":
    main()
