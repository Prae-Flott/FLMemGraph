#!/usr/bin/env python3
"""
FEDERATED Joint Prototype Memory V2.1 (with covariance anomaly head, no
physics prior -- see `memory/scoring-signals-B-C-E-H.md`) on Sielaff's 10
real reverse-vending machines as 10 federated clients.

Upgraded from pure V2 (JointPrototypeMemoryOnly) to V2.1
(JointPrototypeV21): adds GDN-style learned attention (TrendGraphAttentionHead)
and a prototype-conditioned covariance anomaly head (DeviationCovarianceHead)
alongside the original prototype+node signals. Sielaff has NO verified physical
prior (`benchmark/datasets/sielaff_physics.md`), so prior_edges=None -- the
attention graph is fully learned from data. V3's TypedRelationAnomalyHead is
NOT added (needs declared edge types).

Ablation columns A (prototype only) and B (prototype+node) remain directly
comparable to the existing centralized `run_sielaff_joint_prototype_v2.py`.
New columns: C (attention edge), H (covariance), I (V2+cov).

Federated protocol: encoder/edge_head/cov_head stay local; ONLY the
JointPrototypeMemory codebook is exchanged each round via align_and_split.

Usage:
    python3 run_sielaff_v2_1_federated.py [--calib-mode {global,per_prototype,ema}]

`--calib-mode` (default `global`, unchanged behavior): see
`run_robo3er_v3_1_federated.py`'s docstring and
`memory/calib-in-prototype-ab.md` for the full Path A/B explanation.
Only B (`d_node`)/C (`resid_struct`)'s per-node z-score (stage before the
`.max(axis=1)`) is affected; the reliability mask (`RELIABILITY_RATIO`)
stays computed from the GLOBAL calib IQR in every mode -- it is a coarse
near-constant-node filter, not part of this A/B test's scope.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "federated"))
from joint_prototype_model import JointPrototypeV21  # noqa: E402
from federated_memory import align_and_split  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "sielaff"
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
TOP_K = 8
BETA = 0.25
LAMBDA_EDGE = 0.5
GAMMA = 0.5
DELTA = 0.5  # same as robo3er's federated setting -- small per-client normal-fit
             # sets here too (each of 10 machines, not one pooled 5000+ set)
RELIABILITY_RATIO = 0.05
MIN_PROTO_SAMPLES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    d_proto_all, d_node_all, resid_struct_all, idx_all, d_mahal_all = [], [], [], [], []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(DEVICE)
        out = model(batch, training_mode=False)
        d_proto_all.append(out["d_proto"].cpu().numpy())
        d_node_all.append(out["d_node"].cpu().numpy())
        resid_struct_all.append(out["resid_struct"].cpu().numpy())
        idx_all.append(out["idx"].cpu().numpy())
        d_mahal_all.append(out["d_mahal"].cpu().numpy())
    return (np.concatenate(d_proto_all), np.concatenate(d_node_all),
            np.concatenate(resid_struct_all), np.concatenate(idx_all),
            np.concatenate(d_mahal_all))


def train_local(model, fit_arr, epochs, calib_mode="global"):
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
            l_edge = out["resid_struct"].mean()
            loss = l_pred + BETA * l_me + l_mc + LAMBDA_EDGE * l_edge
            loss.backward()
            optimizer.step()
            if calib_mode == "ema":
                model.memory.update_ema(out["d_node"].detach(), out["idx"].detach())
    return model


def zscore(x, calib_x):
    median = np.median(calib_x, axis=0)
    q75, q25 = np.percentile(calib_x, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    return (x - median) / iqr, iqr


def zscore_mode(x, idx_x, calib_x, idx_calib, score_head=None, ema_memory=None):
    """Like `zscore()` but with an optional per-prototype (`score_head`,
    Path B) or EMA (`ema_memory`, Path A) override -- default (both None)
    is bit-identical to `zscore(x, calib_x)`'s first return value."""
    if score_head is not None:
        return score_head.zscore(x, idx_x)
    if ema_memory is not None:
        return ema_memory.ema_zscore(x, idx_x)
    z, _ = zscore(x, calib_x)
    return z


def main(calib_mode="global"):
    assert calib_mode in ("global", "per_prototype", "ema")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("loading Sielaff, splitting into 10 real per-machine federated clients ...")
    data, targets, cols, label_map, clients = build_clients()
    num_nodes = len(cols)
    for c in clients:
        print(f"  client {c.client_id}: fit={len(c.fit_idx)} calib={len(c.calib_idx)} "
              f"test_normal={len(c.test_normal_idx)}")

    scalers = [fit_client_scaler(data, c.fit_idx) for c in clients]
    models = [JointPrototypeV21(num_nodes=num_nodes, window_size=WINDOW_LEN,
                                    embed_dim=EMBED_DIM, num_prototypes=NUM_PROTOTYPES,
                                    prior_edges=None, top_k=TOP_K).to(DEVICE)
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
            train_local(model, fit_w, LOCAL_EPOCHS, calib_mode=calib_mode)

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

    cov_min = max(5, num_nodes + 1)
    all_pairs = {"B_node_max": [], "C_struct_max": [], "H_cov_mahal": [], "I_node_cov_max": []}
    for c, model, scaler in zip(clients, models, scalers):
        calib_arr = scale_client(data, scaler, c.calib_idx)
        test_normal_arr = scale_client(data, scaler, c.test_normal_idx)

        # Stage B: calibrate covariance head from calib set
        _, d_node_b, _, calib_idx_b, _ = per_sample_scores(model, calib_arr)
        model.cov_head.set_calibration(d_node_b, calib_idx_b, min_samples=cov_min)

        d_proto_calib, d_node_calib, resid_struct_calib, calib_idx_w, d_mahal_calib = per_sample_scores(model, calib_arr)
        d_proto_normal, d_node_normal, resid_struct_normal, idx_normal, d_mahal_normal = per_sample_scores(model, test_normal_arr)

        if calib_mode == "per_prototype":
            model.set_score_calibration(d_node_calib, resid_struct_calib, calib_idx_w, min_samples=MIN_PROTO_SAMPLES)
        node_head = model.score_calib_node if calib_mode == "per_prototype" else None
        struct_head = model.score_calib_struct if calib_mode == "per_prototype" else None
        ema_mem = model.memory if calib_mode == "ema" else None  # C has no online EMA hook -> global fallback

        _, node_iqr = zscore(d_node_normal, d_node_calib)  # reliability mask always global, see module docstring
        median_iqr = np.median(node_iqr)
        reliable_mask = node_iqr >= RELIABILITY_RATIO * median_iqr

        z_node_normal = zscore_mode(d_node_normal, idx_normal, d_node_calib, calib_idx_w, score_head=node_head, ema_memory=ema_mem)
        z_struct_normal = zscore_mode(resid_struct_normal, idx_normal, resid_struct_calib, calib_idx_w, score_head=struct_head)
        z_mahal_normal, _ = zscore(d_mahal_normal, d_mahal_calib)
        z_node_normal_masked = z_node_normal[:, reliable_mask]
        b_normal = z_node_normal_masked.max(axis=1)
        scores_normal = {
            "B_node_max": b_normal,
            "C_struct_max": z_struct_normal.max(axis=1),
            "H_cov_mahal": z_mahal_normal,
            "I_node_cov_max": np.maximum(b_normal, z_mahal_normal),
        }

        client_report = {"excluded_nodes": [cols[i] for i in range(num_nodes) if not reliable_mask[i]],
                          "n_valid_cov_prototypes": int(model.cov_head.calib_valid.sum()),
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
            d_proto_f, d_node_f, resid_struct_f, idx_f, d_mahal_f = per_sample_scores(model, fault_arr)
            z_node_f = zscore_mode(d_node_f, idx_f, d_node_calib, calib_idx_w, score_head=node_head, ema_memory=ema_mem)
            z_struct_f = zscore_mode(resid_struct_f, idx_f, resid_struct_calib, calib_idx_w, score_head=struct_head)
            z_mahal_f, _ = zscore(d_mahal_f, d_mahal_calib)
            z_node_f_masked = z_node_f[:, reliable_mask]
            b_f = z_node_f_masked.max(axis=1)
            scores_fault = {
                "B_node_max": b_f,
                "C_struct_max": z_struct_f.max(axis=1),
                "H_cov_mahal": z_mahal_f,
                "I_node_cov_max": np.maximum(b_f, z_mahal_f),
            }

            aurocs = {}
            for key in scores_normal:
                labels = np.concatenate([np.zeros(len(scores_normal[key])), np.ones(len(scores_fault[key]))])
                s = np.concatenate([scores_normal[key], scores_fault[key]])
                auroc = float(sk_metrics.roc_auc_score(labels, s))
                aurocs[key] = auroc
                rows[key].append(auroc)
                all_pairs[key].append(auroc)
            client_report["fault_types"][name] = {"n": int(len(fault_idx)), **aurocs}
            print(f"  machine{c.client_id:<3}{name:<20}n={len(fault_idx):<5}"
                  + "  ".join(f"{k}={v:.3f}" for k, v in aurocs.items()))

        if rows["B_node_max"]:
            client_report["client_mean_auroc"] = {k: float(np.mean(v)) for k, v in rows.items()}
        report["clients"][f"machine{c.client_id}"] = client_report

    summary_overall = {k: float(np.mean(v)) for k, v in all_pairs.items()}
    print(f"\n{'method':<24}{'mean AUROC (all machine-fault pairs)':>36}")
    for key, v in summary_overall.items():
        print(f"{key:<24}{v:>36.3f}")

    report["config"]["calib_mode"] = calib_mode
    report["summary_mean_auroc_overall"] = summary_overall
    mode_suffix = f"_calibmode_{calib_mode}" if calib_mode != "global" else ""
    out_name = f"sielaff_v2_1_federated{mode_suffix}_report.json"
    with open(OUT_DIR / out_name, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
                OUT_DIR / f"sielaff_v2_1_federated{mode_suffix}.pth")
    print(f"\nsaved -> {OUT_DIR / out_name}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-mode", choices=["global", "per_prototype", "ema"], default="global")
    args = parser.parse_args()
    main(calib_mode=args.calib_mode)
