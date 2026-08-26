#!/usr/bin/env python3
"""
Federated multi-source fault detection over robo3er's 5 real robots
(`fl_dataset.py`'s client split), implementing the design in
`mem_phys_prompt_zh.md`: one FLGDNMemory (`fl_model.py`) per client,
fully local encoder + structure head (never leaves the client), and ONLY
the discrete memory codebook exchanged each round via server-side
alignment (`federated_memory.py`).

Per round:
  1. Each client loads the current global memory P_G,n (round 0: its own
     random init), trains locally for LOCAL_EPOCHS on its own normal fit
     windows only (whole-window reconstruction, see fl_model.py -- no
     dense sliding needed since there's no forecasting target).
  2. Server collects every client's (codebook, usage_count), aligns them
     (`align_and_split`), broadcasts P_G,n back.

After ROUNDS rounds, each client with fault data is evaluated with three
scores (all via the same GDN-style per-node median/IQR-normalize-then-max
rule, calibrated from THAT CLIENT'S OWN calib split -- matches
`deployment-architecture.md`'s "share weights, calibrate locally"
recommendation):
  - "d"      : memory distance only
  - "r"      : structure residual only
  - "d+r"    : both concatenated (2N-dim) before the same max rule

Usage:
    python3 train_fl_memory_gdn.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "dataloaders" / "robo3er"))
sys.path.insert(0, str(REPO_ROOT / "src" / "federated"))
from fl_model import FLGDNMemory  # noqa: E402
from fl_dataset import load_fl_clients  # noqa: E402
from federated_memory import align_and_split  # noqa: E402
from dataset import gdn_score  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "robo3er"  # local copy, see dataset.py's comment
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
GAMMA = 0.5   # shared-prototype fraction
DELTA = 0.5   # cross-client alignment cosine-similarity threshold (lowered from
              # FedPM's own value after the shared-init fix still showed
              # cross-client similarities mostly in the 0.3-0.6 range for this
              # much smaller, much more non-IID 5-client setting than FedPM's
              # own benchmarks -- see fl_memory_gdn.log for the diagnostics
              # this was tuned against)
IQR_FLOOR_FRAC = 0.1  # per-client calib sets are small (22-692 windows) --
                       # without a floor, a single near-zero-IQR dimension can
                       # blow the combined score up by orders of magnitude
                       # (empirically observed: scores in the millions on one
                       # client's calib set). See dataset.gdn_score's docstring
                       # for the detection-AUROC-vs-stability trade-off this
                       # implies -- accepted here because the alternative
                       # (unfloored) was not "slightly worse," it was broken.
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
    """[N, num_nodes] d and r matrices for a window set."""
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

    # Share ONE initial encoder+structure-head state across all clients before
    # any local training. Without this, each client's 64-dim latent space is an
    # independently-random rotation of the "same" underlying physical manifold
    # -- cosine similarity between two different clients' codebook prototypes
    # is then meaningless (empirically verified: max cross-client cosine sim
    # was 0.50 with independent inits, never exceeding any reasonable delta,
    # so align_and_split found ZERO shared clusters every round). Sharing the
    # starting point doesn't violate "encoder stays local" -- each client still
    # trains its encoder fully independently afterward, only the ROUND-0 seed
    # is common, giving correlated-enough embedding geometry for the alignment
    # step's cosine similarity to mean something.
    shared_init = {k: v.clone() for k, v in models[0].state_dict().items()
                   if k.startswith("encoder.") or k.startswith("structure.")}
    for model in models[1:]:
        model.load_state_dict(shared_init, strict=False)

    print(f"\nfederated training: {ROUNDS} rounds x {LOCAL_EPOCHS} local epochs, "
          f"memory-only exchange (gamma={GAMMA}, delta={DELTA})")
    diagnostics_log = []
    for rnd in range(1, ROUNDS + 1):
        for c, model, scaler in zip(clients, models, scalers):
            fit_w = scale_client(data, scaler, c.fit_idx)
            train_local(model, fit_w, epochs=LOCAL_EPOCHS)

        codebooks = [m.memory.codebook.detach().clone() for m in models]
        usage_counts = [m.memory.usage_count.detach().clone() for m in models]
        P_G, diag = align_and_split(codebooks, usage_counts, gamma=GAMMA, delta=DELTA)
        diag["round"] = rnd
        diagnostics_log.append(diag)
        print(f"  round {rnd}: {diag}")

        for model, p_g in zip(models, P_G):
            model.load_memory(p_g)

    guard_band_path = DATA_DIR / "stuck_severity_split.json"
    guard_band_excluded = set()
    if guard_band_path.exists():
        with open(guard_band_path) as f:
            guard_band_excluded = set(json.load(f)["transition_excluded_idx"])

    report = {"config": {"rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "num_prototypes": NUM_PROTOTYPES,
                          "beta": BETA, "lambda": LAMBDA, "gamma": GAMMA, "delta": DELTA},
              "alignment_log": diagnostics_log, "clients": {}}

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
               OUT_DIR / "fl_memory_gdn_robo3er.pth")
    with open(OUT_DIR / "fl_memory_gdn_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {OUT_DIR / 'fl_memory_gdn_robo3er.pth'}")
    print(f"saved -> {OUT_DIR / 'fl_memory_gdn_report.json'}")


if __name__ == "__main__":
    main()
