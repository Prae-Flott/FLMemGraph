#!/usr/bin/env python3
"""
IFCAAE: unsupervised, autoencoder-based adaptation of IFCA (Ghosh, Chung,
Yin, Ramchandran, "An Efficient Framework for Clustered Federated
Learning", IEEE Trans. Information Theory 2022), migrated from
`~/Projects/FL-bench`'s `src/server|client/ifcaae.py` -- see
`memory/ifcaae-federated-clustering.md` there for the original build notes
and the clustering-collapse bug this reimplementation carries the fix for.

Standard IFCA maintains `num_clusters` independent global models and
hard-assigns each client to whichever gives the lowest label-supervised
loss. IFCAAE removes the label dependency: the model is a reconstruction
autoencoder (`src/conv_autoencoder.ConvAutoEncoder`), clients train and
pick clusters using only reconstruction MSE on windows they already know
are normal (no labels used for training or clustering); fault labels are
used exclusively at evaluation time.

This is category (4)'s other "generic FL, not memory/structure-aware"
comparison point alongside `run_fedavg_baseline.py` -- but unlike FedAvg's
single global model, IFCAAE tests whether just letting clients self-
organize into a SMALL number of shared clusters (no per-node discrete
memory, no relational/physical consistency signal) is enough to handle
robo3er's severe non-IID split. `memory/ifcaae-federated-clustering.md`'s
original FL-bench run already found num_clusters=2 converges to a
physically sensible split (robot04 alone vs. the other four) -- reused as
the default here.

## Two fixes carried over from the original FL-bench build (both needed --
## verified there on the same 5-client robo3er setting this reuses)

1. **Warmup before splitting.** Starting every cluster beyond cluster 0
   from independent random init makes its round-1 reconstruction error
   arbitrary, so a badly-initialized cluster can lose every client's
   argmin from round 1 onward and never train again -- the clustered
   federation silently degenerates into plain FedAvg. Fix: `warmup_rounds`
   trains ONE shared model first, then splits it into `num_clusters`
   perturbed copies (Gaussian noise scaled by each parameter's own std).
2. **Respawn idle clusters.** A cluster nobody picks for `respawn_patience`
   consecutive rounds is revived from the busiest cluster's current
   weights plus a fresh perturbation, instead of being left dead at stale
   weights for the rest of training.

Usage:
    python3 run_ifcaae_baseline.py
"""
import json
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conv_autoencoder import ConvAutoEncoder  # noqa: E402
from fl_dataset import load_fl_clients  # noqa: E402
from dataset import gdn_score  # noqa: E402
from baselines.fedavg_baseline import federated_average  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "robo3er"
OUT_DIR = REPO_ROOT / "checkpoints" / "robo3er"

BATCH_SIZE = 32
ROUNDS = 15
LOCAL_EPOCHS = 20
LR = 1e-3
SEED = 42
NUM_CLUSTERS = 2         # memory/ifcaae-federated-clustering.md: converges to
                          # robot04-alone vs. the-other-four with this setting
WARMUP_ROUNDS = 5
PERTURB_STD = 0.05
RESPAWN_PATIENCE = 3
IQR_FLOOR_FRAC = 0.1     # matches train_fl_memory_gdn.py / run_fedavg_baseline.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scale_client(data, scaler, idx):
    n, t, f = data[idx].shape
    return scaler.transform(data[idx].reshape(-1, f)).reshape(n, t, f).astype(np.float32)


def fit_client_scaler(data, fit_idx):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(data[fit_idx].reshape(-1, data.shape[-1]))
    return scaler


def perturb(state_dict, std):
    """Gaussian-perturb every floating-point tensor (weights/buffers like
    BatchNorm running stats); integer buffers (e.g. BatchNorm's
    `num_batches_tracked`) are copied unperturbed -- `randn_like` has no
    meaning on an integer count."""
    out = OrderedDict()
    for k, v in state_dict.items():
        if torch.is_floating_point(v):
            out[k] = v + torch.randn_like(v) * std * v.float().std(unbiased=False)
        else:
            out[k] = v.clone()
    return out


@torch.no_grad()
def mean_recon_error(model, windows, batch_size=BATCH_SIZE):
    """Scalar mean reconstruction MSE -- used only for cluster ASSIGNMENT
    (argmin across clusters), not for the final per-feature anomaly score
    (see per_window_error below)."""
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(DEVICE)
        recon = model(batch)
        err = ((recon - batch) ** 2).mean(dim=(1, 2))
        total += err.sum().item()
        count += err.numel()
    return total / count if count else float("inf")


@torch.no_grad()
def per_window_error(model, windows, batch_size=BATCH_SIZE):
    """[N, num_nodes] per-feature squared reconstruction error, mean over
    time -- same convention as train_gdn.py's per_window_error, feeding the
    same GDN-style per-feature median/IQR score used everywhere else in
    this project (dataset.gdn_score)."""
    model.eval()
    errs = []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(DEVICE)
        recon = model(batch)
        errs.append(((recon - batch) ** 2).mean(dim=1).cpu().numpy())  # mean over T -> [B, N]
    return np.concatenate(errs, axis=0)


def train_local(model, fit_windows, epochs=LOCAL_EPOCHS):
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
            recon = model(batch)
            loss = torch.nn.functional.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
    return model


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data, targets, cols, label_map, clients = load_fl_clients(active_feature_groups=["kinematic_core"])
    num_nodes = data.shape[2]
    window_size = data.shape[1]
    print(f"num_nodes={num_nodes}  window_size={window_size}  num_clients={len(clients)}  "
          f"num_clusters={NUM_CLUSTERS}  warmup_rounds={WARMUP_ROUNDS}")
    for c in clients:
        print(f"  client {c.client_id} ({c.robot_name}): fit={len(c.fit_idx)} calib={len(c.calib_idx)} "
              f"test_normal={len(c.test_normal_idx)} faults={ {label_map[str(k)]: len(v) for k, v in c.fault_idx_by_label.items()} }")

    scalers = [fit_client_scaler(data, c.fit_idx) for c in clients]
    fit_data = [scale_client(data, s, c.fit_idx) for c, s in zip(clients, scalers)]

    proto = ConvAutoEncoder(num_nodes, window_size).to(DEVICE)
    cluster_params = [deepcopy(proto.state_dict()) for _ in range(NUM_CLUSTERS)]
    rounds_unused = [0] * NUM_CLUSTERS
    client_cluster_ids = [0] * len(clients)

    print(f"\nIFCAAE: {ROUNDS} rounds x {LOCAL_EPOCHS} local epochs")
    for rnd in range(1, ROUNDS + 1):
        in_warmup = rnd <= WARMUP_ROUNDS
        packages_by_cluster = {}  # cluster_id -> list[(state_dict, weight)]

        for ci, (c, fit_w) in enumerate(zip(clients, fit_data)):
            if in_warmup:
                cid = 0
            else:
                model = ConvAutoEncoder(num_nodes, window_size).to(DEVICE)
                losses = []
                for params in cluster_params:
                    model.load_state_dict(params)
                    losses.append(mean_recon_error(model, fit_w))
                cid = int(np.argmin(losses))
            client_cluster_ids[ci] = cid

            model = ConvAutoEncoder(num_nodes, window_size).to(DEVICE)
            model.load_state_dict(cluster_params[cid])
            train_local(model, fit_w, epochs=LOCAL_EPOCHS)

            packages_by_cluster.setdefault(cid, []).append((model.state_dict(), len(c.fit_idx)))

        for cid, packages in packages_by_cluster.items():
            state_dicts = [p[0] for p in packages]
            weights = [p[1] for p in packages]
            cluster_params[cid] = federated_average(state_dicts, weights)

        assignment = ", ".join(f"client{i}->cluster{cid}" for i, cid in enumerate(client_cluster_ids))
        print(f"  round {rnd}{' [warmup]' if in_warmup else ''}: {assignment}")

        if rnd == WARMUP_ROUNDS:
            base = cluster_params[0]
            for i in range(1, NUM_CLUSTERS):
                cluster_params[i] = perturb(base, PERTURB_STD)
            rounds_unused = [0] * NUM_CLUSTERS
            print(f"  -> warmup done, split cluster 0 into {NUM_CLUSTERS} perturbed clusters "
                  f"(perturb_std={PERTURB_STD})")
        elif rnd > WARMUP_ROUNDS:
            picked = set(packages_by_cluster.keys())
            busiest = max(packages_by_cluster, key=lambda cid: len(packages_by_cluster[cid]))
            for cid in range(NUM_CLUSTERS):
                if cid in picked:
                    rounds_unused[cid] = 0
                    continue
                rounds_unused[cid] += 1
                if rounds_unused[cid] >= RESPAWN_PATIENCE:
                    cluster_params[cid] = perturb(cluster_params[busiest], PERTURB_STD)
                    rounds_unused[cid] = 0
                    print(f"  -> cluster {cid} unpicked for {RESPAWN_PATIENCE} rounds, "
                          f"respawned from cluster {busiest}")

    # Final per-client model = whichever cluster it last picked.
    models = []
    for ci in range(len(clients)):
        model = ConvAutoEncoder(num_nodes, window_size).to(DEVICE)
        model.load_state_dict(cluster_params[client_cluster_ids[ci]])
        models.append(model)

    guard_band_path = DATA_DIR / "stuck_severity_split.json"
    guard_band_excluded = set()
    if guard_band_path.exists():
        with open(guard_band_path) as f:
            guard_band_excluded = set(json.load(f)["transition_excluded_idx"])

    report = {"config": {"rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS,
                          "num_clusters": NUM_CLUSTERS, "warmup_rounds": WARMUP_ROUNDS,
                          "perturb_std": PERTURB_STD, "respawn_patience": RESPAWN_PATIENCE},
              "final_cluster_assignment": {c.robot_name: client_cluster_ids[i] for i, c in enumerate(clients)},
              "clients": {}}
    print(f"\nfinal cluster assignment: " +
          ", ".join(f"{c.robot_name}->cluster{client_cluster_ids[i]}" for i, c in enumerate(clients)))

    for c, model, scaler in zip(clients, models, scalers):
        if not c.fault_idx_by_label:
            print(f"\nclient {c.client_id} ({c.robot_name}): no fault windows, skipping evaluation")
            continue

        calib_w = scale_client(data, scaler, c.calib_idx)
        calib_err = per_window_error(model, calib_w)
        median = np.median(calib_err, axis=0)
        q75, q25 = np.percentile(calib_err, [75, 25], axis=0)
        iqr = np.maximum(q75 - q25, 1e-8)

        test_w = scale_client(data, scaler, c.test_normal_idx)
        normal_err = per_window_error(model, test_w)
        normal_score = gdn_score(normal_err, median, iqr, iqr_floor_frac=IQR_FLOOR_FRAC)

        client_report = {"robot_name": c.robot_name, "cluster_id": client_cluster_ids[c.client_id], "fault_types": {}}
        print(f"\nclient {c.client_id} ({c.robot_name}, cluster {client_cluster_ids[c.client_id]}):")
        print(f"  {'fault type':<16}{'AUROC':>10}")
        for label_id, fault_idx in c.fault_idx_by_label.items():
            name = label_map[str(label_id)]
            if guard_band_excluded:
                keep = np.array([i not in guard_band_excluded for i in fault_idx])
                fault_idx = fault_idx[keep]
            if len(fault_idx) == 0:
                continue
            fault_w = scale_client(data, scaler, fault_idx)
            fault_err = per_window_error(model, fault_w)
            fault_score = gdn_score(fault_err, median, iqr, iqr_floor_frac=IQR_FLOOR_FRAC)

            labels = np.concatenate([np.zeros(len(normal_score)), np.ones(len(fault_score))])
            auroc = float(sk_metrics.roc_auc_score(labels, np.concatenate([normal_score, fault_score])))
            client_report["fault_types"][name] = {"n": int(len(fault_idx)), "auroc": auroc}
            print(f"  {name:<16}{auroc:>10.3f}")

        report["clients"][c.robot_name] = client_report

    torch.save({f"client_{c.client_id}_state_dict": m.state_dict() for c, m in zip(clients, models)},
               OUT_DIR / "ifcaae_baseline_robo3er.pth")
    with open(OUT_DIR / "ifcaae_baseline_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {OUT_DIR / 'ifcaae_baseline_robo3er.pth'}")
    print(f"saved -> {OUT_DIR / 'ifcaae_baseline_report.json'}")


if __name__ == "__main__":
    main()
