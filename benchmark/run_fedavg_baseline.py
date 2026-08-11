#!/usr/bin/env python3
"""
FedAvg baseline over the identical model/data/schedule as
`src/training/train_fl_memory_gdn.py`, differing ONLY in the aggregation rule (see
`baselines/fedavg_baseline.py`'s module docstring for why this is the right
apples-to-apples comparison): standard size-weighted full-parameter
averaging every round instead of memory-only codebook alignment.

Per round:
  1. Each client loads the current global model (round 0: the shared
     init), trains locally for LOCAL_EPOCHS on its own normal fit windows
     (whole-window reconstruction, same as train_fl_memory_gdn.py).
  2. Server averages every client's FULL state dict (encoder + memory +
     structure head + decoder), weighted by client training-set size, and
     broadcasts the single resulting global model back to all clients --
     no personalization at all (unlike our own system, where encoder +
     structure head are permanently local per client).

After ROUNDS rounds, evaluated exactly like train_fl_memory_gdn.py: same
GDN-style per-node median/IQR-normalize-then-max scoring rule, calibrated
from each client's OWN calib split (only the model is now global/shared,
not the calibration reference -- matches this project's "share weights,
calibrate locally" convention).

Usage:
    python3 run_fedavg_baseline.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "robo3er"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fl_model import FLGDNMemory  # noqa: E402
from fl_dataset import load_fl_clients  # noqa: E402
from dataset import gdn_score  # noqa: E402
from baselines.fedavg_baseline import federated_average  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "robo3er"
OUT_DIR = REPO_ROOT / "checkpoints" / "robo3er"

BATCH_SIZE = 32
ROUNDS = 5
LOCAL_EPOCHS = 20
LR = 1e-3
SEED = 42
EMBED_DIM = 64
TOP_K = 15
NUM_PROTOTYPES = 256
BETA = 0.25   # L_ME weight
LAMBDA = 0.5  # L_struct weight
IQR_FLOOR_FRAC = 0.1  # matches train_fl_memory_gdn.py -- small per-client
                       # calib sets need this floor regardless of aggregation rule
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scale_client(data, scaler, idx):
    n, t, f = data[idx].shape
    return scaler.transform(data[idx].reshape(-1, f)).reshape(n, t, f).astype(np.float32)


def fit_client_scaler(data, fit_idx):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(data[fit_idx].reshape(-1, data.shape[-1]))
    return scaler


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
            out = model(batch, training_mode=True)
            l_pred = torch.nn.functional.mse_loss(out["x_hat"], batch)
            l_me, l_mc = model.memory.commitment_losses(out["z"], out["z_q"])
            l_struct = torch.nn.functional.mse_loss(out["z"].detach(), out["z_hat"])
            loss = l_pred + BETA * l_me + l_mc + LAMBDA * l_struct
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def per_window_dr(model, windows, batch_size=BATCH_SIZE):
    model.eval()
    d_all, r_all = [], []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(DEVICE)
        out = model(batch, training_mode=False)
        d_all.append(out["d"].cpu().numpy())
        r_all.append(out["r"].cpu().numpy())
    return np.concatenate(d_all, axis=0), np.concatenate(r_all, axis=0)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data, targets, cols, label_map, clients = load_fl_clients(active_feature_groups=["kinematic_core"])
    num_nodes = data.shape[2]
    window_size = data.shape[1]
    print(f"num_nodes={num_nodes}  window_size={window_size}  num_clients={len(clients)}")
    for c in clients:
        print(f"  client {c.client_id} ({c.robot_name}): fit={len(c.fit_idx)} calib={len(c.calib_idx)} "
              f"test_normal={len(c.test_normal_idx)} faults={ {label_map[str(k)]: len(v) for k, v in c.fault_idx_by_label.items()} }")

    scalers = [fit_client_scaler(data, c.fit_idx) for c in clients]
    models = [FLGDNMemory(num_nodes, window_size, EMBED_DIM, TOP_K, NUM_PROTOTYPES).to(DEVICE) for _ in clients]

    # Same shared-init rationale as train_fl_memory_gdn.py, but here it
    # covers the WHOLE model (not just encoder+structure) since FedAvg
    # averages every parameter -- starting from independently-random full
    # models and averaging them round 0 would produce a meaningless
    # "average of unrelated random networks" first global model.
    shared_init = models[0].state_dict()
    for model in models[1:]:
        model.load_state_dict(shared_init)

    print(f"\nFedAvg baseline: {ROUNDS} rounds x {LOCAL_EPOCHS} local epochs, "
          f"full-parameter weighted averaging (McMahan et al. 2017)")
    for rnd in range(1, ROUNDS + 1):
        for c, model, scaler in zip(clients, models, scalers):
            fit_w = scale_client(data, scaler, c.fit_idx)
            train_local(model, fit_w, epochs=LOCAL_EPOCHS)

        state_dicts = [m.state_dict() for m in models]
        client_weights = [len(c.fit_idx) for c in clients]
        global_state = federated_average(state_dicts, client_weights)
        print(f"  round {rnd}: aggregated {len(clients)} clients "
              f"(weights={client_weights}) -> single global model")

        for model in models:
            model.load_state_dict(global_state)

    guard_band_path = DATA_DIR / "stuck_severity_split.json"
    guard_band_excluded = set()
    if guard_band_path.exists():
        with open(guard_band_path) as f:
            guard_band_excluded = set(json.load(f)["transition_excluded_idx"])

    report = {"config": {"rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS,
                          "num_prototypes": NUM_PROTOTYPES, "beta": BETA, "lambda": LAMBDA,
                          "aggregation": "fedavg_full_parameter"},
              "clients": {}}

    for c, model, scaler in zip(clients, models, scalers):
        if not c.fault_idx_by_label:
            print(f"\nclient {c.client_id} ({c.robot_name}): no fault windows, skipping evaluation")
            continue

        calib_w = scale_client(data, scaler, c.calib_idx)
        d_calib, r_calib = per_window_dr(model, calib_w)
        d_median, r_median = np.median(d_calib, axis=0), np.median(r_calib, axis=0)
        d_q75, d_q25 = np.percentile(d_calib, [75, 25], axis=0)
        r_q75, r_q25 = np.percentile(r_calib, [75, 25], axis=0)
        d_iqr = np.maximum(d_q75 - d_q25, 1e-8)
        r_iqr = np.maximum(r_q75 - r_q25, 1e-8)
        dr_median = np.concatenate([d_median, r_median])
        dr_iqr = np.concatenate([d_iqr, r_iqr])

        test_w = scale_client(data, scaler, c.test_normal_idx)
        d_normal, r_normal = per_window_dr(model, test_w)
        scores_normal = {
            "d": gdn_score(d_normal, d_median, d_iqr, iqr_floor_frac=IQR_FLOOR_FRAC),
            "r": gdn_score(r_normal, r_median, r_iqr, iqr_floor_frac=IQR_FLOOR_FRAC),
            "d+r": gdn_score(np.concatenate([d_normal, r_normal], axis=1), dr_median, dr_iqr, iqr_floor_frac=IQR_FLOOR_FRAC),
        }

        client_report = {"robot_name": c.robot_name, "fault_types": {}}
        print(f"\nclient {c.client_id} ({c.robot_name}):")
        print(f"  {'fault type':<16}{'AUROC(d)':>10}{'AUROC(r)':>10}{'AUROC(d+r)':>12}")
        for label_id, fault_idx in c.fault_idx_by_label.items():
            name = label_map[str(label_id)]
            if guard_band_excluded:
                keep = np.array([i not in guard_band_excluded for i in fault_idx])
                fault_idx = fault_idx[keep]
            if len(fault_idx) == 0:
                continue
            fault_w = scale_client(data, scaler, fault_idx)
            d_fault, r_fault = per_window_dr(model, fault_w)
            scores_fault = {
                "d": gdn_score(d_fault, d_median, d_iqr, iqr_floor_frac=IQR_FLOOR_FRAC),
                "r": gdn_score(r_fault, r_median, r_iqr, iqr_floor_frac=IQR_FLOOR_FRAC),
                "d+r": gdn_score(np.concatenate([d_fault, r_fault], axis=1), dr_median, dr_iqr, iqr_floor_frac=IQR_FLOOR_FRAC),
            }
            aurocs = {}
            for key in scores_normal:
                labels = np.concatenate([np.zeros(len(scores_normal[key])), np.ones(len(scores_fault[key]))])
                auroc = float(sk_metrics.roc_auc_score(labels, np.concatenate([scores_normal[key], scores_fault[key]])))
                aurocs[key] = auroc
            client_report["fault_types"][name] = {"n": int(len(fault_idx)), **{f"auroc_{k}": v for k, v in aurocs.items()}}
            print(f"  {name:<16}{aurocs['d']:>10.3f}{aurocs['r']:>10.3f}{aurocs['d+r']:>12.3f}")

        report["clients"][c.robot_name] = client_report

    torch.save({f"client_{c.client_id}_state_dict": m.state_dict() for c, m in zip(clients, models)},
               OUT_DIR / "fedavg_baseline_robo3er.pth")
    with open(OUT_DIR / "fedavg_baseline_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {OUT_DIR / 'fedavg_baseline_robo3er.pth'}")
    print(f"saved -> {OUT_DIR / 'fedavg_baseline_report.json'}")


if __name__ == "__main__":
    main()
