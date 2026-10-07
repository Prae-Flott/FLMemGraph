#!/usr/bin/env python3
"""
FEDERATED JointPrototypeV21Forecast (V2.1 + ForecastHead, signal PD) on
ALFA's 6 fault-type clients (engine/aileron/rudder/elevator/
aileron_rudder_combo/no_failure -- see `src/dataloaders/alfa/dataset.py`'s
module docstring for why clients are fault-type groups here and not
physical units), trained/evaluated on `data/alfa/` (built by
`src/dataloaders/alfa/build_alfa.py` -- run that first). Window labels:
0=normal, 1..5=fault type (see `data/alfa/metadata.json`'s `class_names`).

PROJECT MAINLINE (FD+PD-only trim 2026-09-02, see
`src/federated/federated_train_eval.py`'s module docstring): reports ONLY
`FD_max`/`PD_max`/`FD_PD_max` (FD+PD via `smooth_max`,
`SMOOTH_MAX_T`) -- the earlier C/H/BCK columns are retired from this
detection report; `edge_head`/`cov_head` are still trained but no longer
scored here (`diagnose_alfa_localization_federated.py` calibrates and uses
them independently for node-level localization).

Structurally this is `run_sielaff_red_bck_federated.py` minus the
per-red-ID multi-label breakdown (ALFA windows are single-label, like
robo3er/Paderborn, not multi-label like Sielaff's red windows) -- same
`JointPrototypeV21Forecast` (no declared physics edges: ALFA has no
`*_physics.md`/edge module yet, matching Sielaff's "no declared edges"
choice, not robo3er/Paderborn's typed-edge V31Forecast), same memory-only
federated exchange (no encoder/decoder FedAvg), same single-stage
z-score+max B/K scoring (not V31's two-stage top-k-mean+re-zscore).

**Multi-flight-per-client continuity fix (the one real correctness
difference from every existing `_bck_federated.py` script)**: every other
dataset's clients are ONE continuous physical unit's own time-ordered log,
so adjacent indices within a client are always temporally adjacent. ALFA's
clients pool MULTIPLE DISCONTINUOUS FLIGHTS (e.g. all 23 engine-failure
flights' windows, concatenated flight-by-flight) -- index i+1 is only
guaranteed temporally adjacent to index i if they're the SAME flight, not
just the same client. `build_pairs` here checks `window_flight` (from
`data/alfa/window_flight_names.json`, one flight name per window,
factorized to ints), not just `window_client`, before accepting a forecast
chain -- otherwise K's forecast-horizon chains would silently bridge two
unrelated flights (e.g. one engine-failure flight's last window predicting
a DIFFERENT engine-failure flight's first window).

horizon_mult defaults to 1 (not robo3er/Paderborn's 10): ALFA's smallest
client (aileron_rudder_combo, 1 flight) has only 88 total windows / 49 fit
windows -- thin, matching Sielaff-red's same default-1 choice for the same
reason (see that script's docstring).

Usage:
    python3 src/dataloaders/alfa/build_alfa.py   # once, to build data/alfa/
    python3 run_alfa_bck_federated.py [--horizon-mult M] [--out-suffix NAME]
        [--num-prototypes N]
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
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "FedPRO"))
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "FedCPG"))
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "uFedHy-DisMTSADD"))
from joint_prototype_model import JointPrototypeV21Forecast  # noqa: E402
from baseline_models import (ReconOnlyModel, ExemplarOnlyModel, FedPROModel,  # noqa: E402
                              FedCPGModel, SCNorTransformerModel, Hypernetwork)
from federated_train_eval import (run_federated_rounds, smooth_max, detection_metrics,  # noqa: E402
                           fit_prototype_zscore_stats, zscore_prototype, loss_weighted_combo,
                           gdn_score, train_local_recon, train_local_exemplar,
                           per_node_recon_error, per_node_exemplar_deviation)
from fedpro import (train_local_classifier, embed_all, build_initial_prototypes_kmeans,  # noqa: E402
                     refine_prototypes_margin, retrieve_and_vote, confidence_ensemble)
from fedcpg import train_local_fedcpg, aggregate_global_prototypes, predict_anomaly_score  # noqa: E402
from ufedhy import hypernet_round_update  # noqa: E402
from federated_memory import confidence_weights_from_losses  # noqa: E402
from comm_cost import round_comm_record  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "alfa"
OUT_DIR = REPO_ROOT / "checkpoints" / "alfa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15
BATCH_SIZE = 256
ROUNDS = 5
LOCAL_EPOCHS = 30
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 16
WINDOW_LEN = 15
STRIDE = 8
TOP_K = 8
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_FORECAST = 0.5
GAMMA = 0.5
DELTA = 0.5
RELIABILITY_RATIO = 0.05
MIN_PROTO_SAMPLES = 5
SMOOTH_MAX_T = 1.0  # temperature for smooth_max's cross-signal (B/C/K) combination
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Client:
    def __init__(self, client_id, client_name, all_idx, fit_idx, calib_idx, test_normal_idx):
        self.client_id = client_id
        self.client_name = client_name
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
    client_names = meta["client_names"]
    with open(DATA_DIR / "window_flight_names.json") as f:
        flight_names = json.load(f)
    flight_to_id = {name: i for i, name in enumerate(sorted(set(flight_names)))}
    window_flight = np.array([flight_to_id[n] for n in flight_names], dtype=int)

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
            client_id=client_id, client_name=client_names[client_id], all_idx=idx,
            fit_idx=normal_idx[:n_fit],
            calib_idx=normal_idx[n_fit : n_fit + n_calib],
            test_normal_idx=normal_idx[n_fit + n_calib :],
        ))
    return data, targets, cols, label_map, client_names, clients, window_flight


def build_pairs(idx_arr, targets, window_flight, horizon_mult, require_next_normal, require_same_split=False):
    N = len(targets)
    idx_set = set(idx_arr.tolist()) if require_same_split else None
    vi, chains, mask = [], [], []
    for i in idx_arr:
        chain = [i + k for k in range(1, horizon_mult + 1)]
        valid = True
        for j in chain:
            if (j >= N or window_flight[j] != window_flight[i]
                    or (require_next_normal and targets[j] != 0)
                    or (require_same_split and j not in idx_set)):
                valid = False
                break
        mask.append(valid)
        if valid:
            vi.append(i)
            chains.append(chain)
    return np.array(vi, dtype=int), chains, np.array(mask, dtype=bool)


def gather_future(data_arr, chains, stride):
    if len(chains) == 0:
        return np.zeros((0, 0, data_arr.shape[-1]), dtype=data_arr.dtype)
    segments = [data_arr[[c[k] for c in chains]][:, -stride:, :] for k in range(len(chains[0]))]
    return np.concatenate(segments, axis=1)


def fit_client_scaler(data, fit_idx):
    scaler = StandardScaler()
    scaler.fit(data[fit_idx].reshape(-1, data.shape[-1]))
    return scaler


def scale_client(data, scaler, idx):
    if len(idx) == 0:
        return np.zeros((0,) + data.shape[1:], dtype=np.float32)
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


@torch.no_grad()
def forecast_scores(model, x_in, x_future, batch_size=BATCH_SIZE):
    model.eval()
    if len(x_in) == 0:
        return np.zeros((0, model.num_nodes), dtype=np.float32)
    k_resid_all = []
    for i in range(0, len(x_in), batch_size):
        xb = torch.from_numpy(x_in[i : i + batch_size]).to(DEVICE)
        fb = torch.from_numpy(x_future[i : i + batch_size]).to(DEVICE)
        out = model(xb, training_mode=False, x_future=fb)
        k_resid_all.append(out["k_resid"].cpu().numpy())
    return np.concatenate(k_resid_all)


def train_local(model, fit_in, fit_future, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fit_in), torch.from_numpy(fit_future)),
        batch_size=min(BATCH_SIZE, len(fit_in)), shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for batch, future in loader:
            batch, future = batch.to(DEVICE), future.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch, training_mode=True, x_future=future)
            l_pred = torch.nn.functional.mse_loss(out["x_hat"], batch)
            l_me, l_mc = model.memory.commitment_losses(out["z"], out["p_star"])
            l_edge = out["resid_struct"].mean()
            l_forecast = out["k_resid"].mean()
            loss = l_pred + BETA * l_me + l_mc + LAMBDA_EDGE * l_edge + LAMBDA_FORECAST * l_forecast
            loss.backward()
            optimizer.step()
    return model


def zscore(x, calib_x):
    median = np.median(calib_x, axis=0)
    q75, q25 = np.percentile(calib_x, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    return (x - median) / iqr, iqr


def metrics_dict(scores_normal, scores_fault):
    out = {}
    for key in scores_normal:
        if key not in scores_fault:
            continue
        m = detection_metrics(scores_normal[key], scores_fault[key])
        if m is not None:
            out[key] = m
    return out


def main_fedpro_baseline(out_suffix=None, k_per_class=3, top_k=3, tau=0.1, embed_dim=EMBED_DIM,
                          fedavg_rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, proto_epochs=50, proto_lr=1e-3):
    """FedPRO (Zhou et al., IEEE TC 2026) on ALFA -- a deliberate exception
    to fit-on-healthy: supervised multi-class classification (no_failure +
    ALFA's 5 fault types = 6 classes), trained on ALL labeled data per
    client (each client already IS one fault-type group, so most clients
    see only 2 of the 6 classes -- normal + their own fault type -- a
    natural non-IID split for federated classification, no artificial
    per-client repartitioning needed here unlike Paderborn). See
    `benchmark/FedPRO/fedpro.py`'s module docstring for the method itself."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print("loading ALFA, splitting into 6 fault-type federated clients, baseline=fedpro "
          "(supervised, all labels, NOT fit-on-healthy) ...")
    data, targets, cols, label_map, client_names, clients, window_flight = build_clients()
    num_nodes = len(cols)
    num_classes = len(label_map)

    client_splits = []
    for c in clients:
        idx = np.asarray(c.all_idx)
        y = targets[idx]
        perm = rng.permutation(len(idx))
        n_train = max(1, int(0.7 * len(idx)))
        tr, te = perm[:n_train], perm[n_train:]
        client_splits.append({
            "train_idx": idx[tr], "train_y": y[tr].astype(np.int64),
            "test_idx": idx[te], "test_y": y[te].astype(np.int64),
        })
        print(f"  client {c.client_id} ({c.client_name}): train={len(tr)} test={len(te)} "
              f"classes={sorted(set(y.tolist()))}")

    scalers = [StandardScaler().fit(data[s["train_idx"]].reshape(-1, num_nodes)) for s in client_splits]
    models = [FedPROModel(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=embed_dim,
                           num_classes=num_classes).to(DEVICE) for _ in clients]

    def scale(idx, scaler):
        n, t, f = data[idx].shape
        return scaler.transform(data[idx].reshape(-1, f)).reshape(n, t, f).astype(np.float32)

    def local_train_step(c, model):
        s, scaler = client_splits[c.client_id], scalers[c.client_id]
        x_in = scale(s["train_idx"], scaler)
        train_local_classifier(model, x_in, s["train_y"], local_epochs, DEVICE, lr=LR, batch_size=BATCH_SIZE)
        return len(x_in)

    print(f"\nfederated training (off-the-shelf FedAvg classifier): {fedavg_rounds} rounds x "
          f"{local_epochs} local epochs, baseline=fedpro")
    run_federated_rounds(clients, models, local_train_step, fedavg_rounds, GAMMA, DELTA, mode="fedavg")

    print("\nbuilding + refining per-client prototypes (encoder frozen) ...")
    all_protos, all_proto_labels = [], []
    for c, model in zip(clients, models):
        for p in model.parameters():
            p.requires_grad_(False)
        s, scaler = client_splits[c.client_id], scalers[c.client_id]
        x_train = scale(s["train_idx"], scaler)
        z_train = embed_all(model, x_train, DEVICE)
        protos, proto_labels = build_initial_prototypes_kmeans(z_train, s["train_y"], num_classes, k_per_class)
        protos = refine_prototypes_margin(protos, proto_labels, z_train, s["train_y"],
                                           epochs=proto_epochs, lr=proto_lr, tau=tau)
        all_protos.append(protos)
        all_proto_labels.append(proto_labels)
        print(f"  client {c.client_id} ({c.client_name}): {len(protos)} prototypes "
              f"across {len(set(proto_labels.tolist()))} classes")
    bank_protos = np.concatenate(all_protos, axis=0)
    bank_labels = np.concatenate(all_proto_labels, axis=0)
    print(f"  global prototype bank: {len(bank_protos)} prototypes from {len(clients)} clients")

    print("\nprototype retrieval-augmented inference on each client's held-out test split ...")
    report = {"config": {"baseline": "fedpro", "fedavg_rounds": fedavg_rounds, "local_epochs": local_epochs,
                          "embed_dim": embed_dim, "num_classes": num_classes, "k_per_class": k_per_class,
                          "top_k": top_k, "tau": tau, "proto_epochs": proto_epochs,
                          "split": "stratified_70_30_all_labels_NOT_fit_on_healthy",
                          "note": "FedPRO is a supervised multi-class classifier wrapper (Zhou et al., "
                                  "IEEE TC 2026) -- see benchmark/FedPRO/fedpro.py"},
              "clients": {}}

    all_pairs = []
    for c, model in zip(clients, models):
        s, scaler = client_splits[c.client_id], scalers[c.client_id]
        x_test = scale(s["test_idx"], scaler)
        z_test = embed_all(model, x_test, DEVICE)
        with torch.no_grad():
            model.eval()
            logits = model(torch.from_numpy(x_test).to(DEVICE))
            y_wi = torch.softmax(logits, dim=1).cpu().numpy()
        y_p, alpha = retrieve_and_vote(z_test, bank_protos, bank_labels, num_classes, top_k=top_k)
        y_en = confidence_ensemble(y_p, y_wi, alpha)
        pred = np.argmax(y_en, axis=1)
        accuracy = float((pred == s["test_y"]).mean())
        anomaly_score = 1.0 - y_en[:, 0]

        client_report = {"client_name": c.client_name, "accuracy": accuracy, "fault_types": {}}
        normal_mask = s["test_y"] == 0
        scores_normal = anomaly_score[normal_mask]
        for label_id in sorted(set(s["test_y"].tolist()) - {0}):
            fault_mask = s["test_y"] == label_id
            scores_fault = anomaly_score[fault_mask]
            m = detection_metrics(scores_normal, scores_fault)
            if m is None:
                continue
            name = label_map[str(label_id)]
            client_report["fault_types"][name] = {"n": int(fault_mask.sum()), "fedpro_score": m}
            all_pairs.append(m)
            print(f"  {c.client_name:<22}{name:<28}n={int(fault_mask.sum()):<5}"
                  f"accuracy={accuracy:.3f}  fedpro_score=auroc:{m['auroc']:.3f}")
        report["clients"][c.client_name] = client_report

    mean_accuracy = float(np.mean([v["accuracy"] for v in report["clients"].values()]))
    summary_overall = {
        metric: float(np.mean([row[metric] for row in all_pairs])) for metric in ("auroc", "auprc", "precision", "f1")
    } if all_pairs else {}
    print(f"\nmean classification accuracy: {mean_accuracy:.3f}")
    print(f"{'fedpro_score':<16}{'AUROC':>10}{'AUPRC':>10}{'Precision':>12}{'F1':>10}")
    if summary_overall:
        print(f"{'fedpro_score':<16}{summary_overall['auroc']:>10.3f}{summary_overall['auprc']:>10.3f}"
              f"{summary_overall['precision']:>12.3f}{summary_overall['f1']:>10.3f}")

    report["mean_accuracy"] = mean_accuracy
    report["summary_mean_auroc_overall"] = {"fedpro_score": summary_overall}
    suffix = out_suffix if out_suffix is not None else "bck_federated_h1_fedpro"
    out_json = OUT_DIR / f"alfa_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "bank_protos": bank_protos,
                "bank_labels": bank_labels, "report": report}, OUT_DIR / f"alfa_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


def main_fedcpg_baseline(out_suffix=None, embed_dim=EMBED_DIM, proj_dim=32, fedcpg_rounds=ROUNDS,
                          local_epochs=LOCAL_EPOCHS, alpha=0.5, beta=0.2, tau=0.1):
    """FedCPG (Li et al., *Computers in Industry* 2025) on ALFA -- same
    per-client-owns-all-its-own-labels split FedPRO already established
    (`main_fedpro_baseline`'s docstring): each client already IS one
    fault-type group, a natural non-IID split, no repartitioning needed.
    See `benchmark/FedCPG/fedcpg.py` for the algorithm and
    `docs/1-s2.0-S0166361524001088-main.pdf` for the paper."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print("loading ALFA, splitting into 6 fault-type federated clients, baseline=fedcpg "
          "(supervised, all labels, NOT fit-on-healthy) ...")
    data, targets, cols, label_map, client_names, clients, window_flight = build_clients()
    num_nodes = len(cols)
    num_classes = len(label_map)

    client_splits = []
    for c in clients:
        idx = np.asarray(c.all_idx)
        y = targets[idx]
        perm = rng.permutation(len(idx))
        n_train = max(1, int(0.7 * len(idx)))
        tr, te = perm[:n_train], perm[n_train:]
        client_splits.append({
            "train_idx": idx[tr], "train_y": y[tr].astype(np.int64),
            "test_idx": idx[te], "test_y": y[te].astype(np.int64),
        })
        print(f"  client {c.client_id} ({c.client_name}): train={len(tr)} test={len(te)} "
              f"classes={sorted(set(y.tolist()))}")

    scalers = [StandardScaler().fit(data[s["train_idx"]].reshape(-1, num_nodes)) for s in client_splits]
    models = [FedCPGModel(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=embed_dim,
                           proj_dim=proj_dim, num_classes=num_classes).to(DEVICE) for _ in clients]

    def scale(idx, scaler):
        n, t, f = data[idx].shape
        return scaler.transform(data[idx].reshape(-1, f)).reshape(n, t, f).astype(np.float32)

    print(f"\nfederated training (backbone FedAvg + personalized head + class-prototype contrast): "
          f"{fedcpg_rounds} rounds x {local_epochs} local epochs, baseline=fedcpg")
    global_protos = torch.zeros(num_classes, proj_dim)
    for rnd in range(1, fedcpg_rounds + 1):
        client_protos, client_counts, fit_counts = [], [], []
        for c, model in zip(clients, models):
            s, scaler = client_splits[c.client_id], scalers[c.client_id]
            x_in = scale(s["train_idx"], scaler)
            local_protos, local_counts = train_local_fedcpg(
                model, x_in, s["train_y"], local_epochs, DEVICE, global_protos, num_classes,
                lr=LR, batch_size=BATCH_SIZE, alpha=alpha, beta=beta, tau=tau)
            client_protos.append(local_protos); client_counts.append(local_counts)
            fit_counts.append(len(x_in))
        state_dicts = [{k: v for k, v in m.state_dict().items() if k.startswith(("encoder.", "proj."))}
                       for m in models]
        total = sum(fit_counts)
        backbone_avg = {k: sum(sd[k] * (n / total) for sd, n in zip(state_dicts, fit_counts))
                        for k in state_dicts[0]}
        for m in models:
            m.load_state_dict(backbone_avg, strict=False)
        global_protos = aggregate_global_prototypes(client_protos, client_counts)
        print(f"  round {rnd}: aggregated backbone + {int((global_protos.abs().sum(dim=1) > 0).sum())}"
              f"/{num_classes} global class prototypes")

    print("\nevaluating personalized heads on each client's held-out test split ...")
    report = {"config": {"baseline": "fedcpg", "fedcpg_rounds": fedcpg_rounds, "local_epochs": local_epochs,
                          "embed_dim": embed_dim, "proj_dim": proj_dim, "num_classes": num_classes,
                          "alpha": alpha, "beta": beta, "tau": tau,
                          "split": "stratified_70_30_all_labels_NOT_fit_on_healthy",
                          "note": "FedCPG (Li et al., Computers in Industry 2025) -- "
                                  "see benchmark/FedCPG/fedcpg.py"},
              "clients": {}}

    all_pairs = []
    for c, model in zip(clients, models):
        s, scaler = client_splits[c.client_id], scalers[c.client_id]
        x_test = scale(s["test_idx"], scaler)
        with torch.no_grad():
            model.eval()
            logits = model(torch.from_numpy(x_test).to(DEVICE))
            pred = logits.argmax(dim=1).cpu().numpy()
        accuracy = float((pred == s["test_y"]).mean())
        anomaly_score = predict_anomaly_score(model, x_test, DEVICE, batch_size=BATCH_SIZE)

        client_report = {"client_name": c.client_name, "accuracy": accuracy, "fault_types": {}}
        normal_mask = s["test_y"] == 0
        scores_normal = anomaly_score[normal_mask]
        for label_id in sorted(set(s["test_y"].tolist()) - {0}):
            fault_mask = s["test_y"] == label_id
            scores_fault = anomaly_score[fault_mask]
            m = detection_metrics(scores_normal, scores_fault)
            if m is None:
                continue
            name = label_map[str(label_id)]
            client_report["fault_types"][name] = {"n": int(fault_mask.sum()), "fedcpg_score": m}
            all_pairs.append(m)
            print(f"  {c.client_name:<22}{name:<28}n={int(fault_mask.sum()):<5}"
                  f"accuracy={accuracy:.3f}  fedcpg_score=auroc:{m['auroc']:.3f}")
        report["clients"][c.client_name] = client_report

    mean_accuracy = float(np.mean([v["accuracy"] for v in report["clients"].values()]))
    summary_overall = {
        metric: float(np.mean([row[metric] for row in all_pairs])) for metric in ("auroc", "auprc", "precision", "f1")
    } if all_pairs else {}
    print(f"\nmean classification accuracy: {mean_accuracy:.3f}")
    print(f"{'fedcpg_score':<16}{'AUROC':>10}{'AUPRC':>10}{'Precision':>12}{'F1':>10}")
    if summary_overall:
        print(f"{'fedcpg_score':<16}{summary_overall['auroc']:>10.3f}{summary_overall['auprc']:>10.3f}"
              f"{summary_overall['precision']:>12.3f}{summary_overall['f1']:>10.3f}")

    report["mean_accuracy"] = mean_accuracy
    report["summary_mean_auroc_overall"] = {"fedcpg_score": summary_overall}
    suffix = out_suffix if out_suffix is not None else "fedcpg"
    out_json = OUT_DIR / f"alfa_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
               OUT_DIR / f"alfa_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


def main_ufedhy_baseline(out_suffix=None, hyper_embed_dim=32, hyper_hidden_dim=64,
                          transformer_embed_dim=16, ufedhy_rounds=ROUNDS, local_epochs=LOCAL_EPOCHS,
                          hyper_lr=1e-3):
    """uFedHy-DisMTSADD (Hao et al., *Information Processing & Management*
    2025) on ALFA -- fit-on-healthy, runs like FedAvg/IFCAAE/Fed-ExDNN.
    See `benchmark/uFedHy-DisMTSADD/ufedhy.py` for the hypernetwork update rule and
    `docs/1-s2.0-S0306457325000494-main.pdf` for the paper."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("loading ALFA, splitting into 6 fault-type federated clients, baseline=ufedhy ...")
    data, targets, cols, label_map, client_names, clients, window_flight = build_clients()
    num_nodes = len(cols)
    for c in clients:
        print(f"  client {c.client_id} ({c.client_name}): fit={len(c.fit_idx)} calib={len(c.calib_idx)} "
              f"test_normal={len(c.test_normal_idx)}")

    scalers = [fit_client_scaler(data, c.fit_idx) for c in clients]
    template = SCNorTransformerModel(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=transformer_embed_dim)
    target_shapes = {name: p.shape for name, p in template.state_dict().items()}
    hypernet = Hypernetwork(num_clients=len(clients), target_shapes=target_shapes,
                             embed_dim=hyper_embed_dim, hidden_dim=hyper_hidden_dim).to(DEVICE)
    hyper_optimizer = torch.optim.Adam(hypernet.parameters(), lr=hyper_lr)
    models = [SCNorTransformerModel(num_nodes=num_nodes, window_size=WINDOW_LEN,
                                     embed_dim=transformer_embed_dim).to(DEVICE) for _ in clients]

    def get_fit_windows(c):
        return scale_client(data, scalers[c.client_id], c.fit_idx)

    print(f"\nfederated training (hypernetwork-generated weights + local SGD + MSE distillation update): "
          f"{ufedhy_rounds} rounds x {local_epochs} local epochs, baseline=ufedhy")
    diagnostics_log = []
    for rnd in range(1, ufedhy_rounds + 1):
        hyper_losses = []
        bytes_up = bytes_down = 0
        for c, model in zip(clients, models):
            x_in = get_fit_windows(c)
            hyper_loss, n, bd, bu = hypernet_round_update(hypernet, hyper_optimizer, c.client_id, model, x_in,
                                                            local_epochs, DEVICE, lr=LR, batch_size=BATCH_SIZE)
            hyper_losses.append(hyper_loss)
            bytes_down += bd
            bytes_up += bu
        diag = {"round": rnd, "mode": "ufedhy", "mean_hypernet_distill_loss": float(np.mean(hyper_losses)),
                **round_comm_record(bytes_up, bytes_down)}
        diagnostics_log.append(diag)
        print(f"  round {rnd}: {diag}")

    report = {"config": {"rounds": ufedhy_rounds, "local_epochs": local_epochs, "baseline": "ufedhy",
                          "transformer_embed_dim": transformer_embed_dim, "hyper_embed_dim": hyper_embed_dim,
                          "hyper_hidden_dim": hyper_hidden_dim, "window_len": WINDOW_LEN, "num_nodes": num_nodes,
                          "score_name": "recon_score",
                          "note": "localization uses this project's own per-node argmax convention, NOT "
                                  "the paper's own PC-algorithm+PageRank causal diagnosis -- see "
                                  "benchmark/uFedHy-DisMTSADD/ufedhy.py's module docstring"},
              "nodes": cols, "alignment_log": diagnostics_log, "clients": {}}

    all_pairs = []
    for c, model, scaler in zip(clients, models, scalers):
        calib_w = scale_client(data, scaler, c.calib_idx)
        calib_raw = per_node_recon_error(model, calib_w, DEVICE)

        test_w = scale_client(data, scaler, c.test_normal_idx)
        test_raw = per_node_recon_error(model, test_w, DEVICE)
        scores_normal = gdn_score(calib_raw, test_raw)

        client_report = {"client_name": c.client_name, "fault_types": {}}
        rows = []
        for label_id_str, name in label_map.items():
            label_id = int(label_id_str)
            if label_id == 0:
                continue
            fault_idx = np.where(targets == label_id)[0]
            fault_idx = np.intersect1d(fault_idx, c.all_idx)
            if len(fault_idx) == 0:
                continue
            fault_w = scale_client(data, scaler, fault_idx)
            fault_raw = per_node_recon_error(model, fault_w, DEVICE)
            scores_fault = gdn_score(calib_raw, fault_raw)

            m = detection_metrics(scores_normal, scores_fault)
            if m is None:
                continue
            client_report["fault_types"][name] = {"n": int(len(fault_idx)), "recon_score": m}
            rows.append(m)
            all_pairs.append(m)
            print(f"  {c.client_name:<22}{name:<28}n={len(fault_idx):<5}recon_score=auroc:{m['auroc']:.3f}")

        if rows:
            client_report["client_mean_metrics"] = {
                metric: float(np.mean([row[metric] for row in rows])) for metric in ("auroc", "auprc", "precision", "f1")
            }
        report["clients"][c.client_name] = client_report

    summary_overall = {
        metric: float(np.mean([row[metric] for row in all_pairs])) for metric in ("auroc", "auprc", "precision", "f1")
    } if all_pairs else {}
    print(f"\n{'recon_score':<16}{'AUROC':>10}{'AUPRC':>10}{'Precision':>12}{'F1':>10}")
    if summary_overall:
        print(f"{'recon_score':<16}{summary_overall['auroc']:>10.3f}{summary_overall['auprc']:>10.3f}"
              f"{summary_overall['precision']:>12.3f}{summary_overall['f1']:>10.3f}")

    report["summary_mean_auroc_overall"] = {"recon_score": summary_overall}
    suffix = out_suffix if out_suffix is not None else "ufedhy"
    out_json = OUT_DIR / f"alfa_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models],
                "hypernet_state_dict": hypernet.state_dict(), "report": report},
               OUT_DIR / f"alfa_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


def main_faithful_baseline(baseline, out_suffix=None, num_clusters=2, num_prototypes=NUM_PROTOTYPES):
    """FedAvg / IFCAAE / Fed-ExDNN, each with its OWN minimal architecture
    and own single anomaly score (no FD/JD/PD/FD+PD/FD+JD+PD) -- see
    `src/models/baseline_models.py`'s module docstring. No forecast
    pairing/chains needed (none of these three baselines forecast): plain
    per-client fit/calib/fault windows only."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"loading ALFA, splitting into 6 fault-type federated clients, baseline={baseline} ...")
    data, targets, cols, label_map, client_names, clients, window_flight = build_clients()
    num_nodes = len(cols)
    for c in clients:
        print(f"  client {c.client_id} ({c.client_name}): fit={len(c.fit_idx)} calib={len(c.calib_idx)} "
              f"test_normal={len(c.test_normal_idx)}")

    scalers = [fit_client_scaler(data, c.fit_idx) for c in clients]

    def make_model():
        if baseline == "fedexdnn":
            return ExemplarOnlyModel(num_nodes=num_nodes, window_size=WINDOW_LEN,
                                      embed_dim=EMBED_DIM, num_prototypes=num_prototypes).to(DEVICE)
        return ReconOnlyModel(num_nodes=num_nodes, window_size=WINDOW_LEN).to(DEVICE)

    models = [make_model() for _ in clients]

    def get_fit_windows(c):
        return scale_client(data, scalers[c.client_id], c.fit_idx)

    def local_train_step(c, model):
        x_in = get_fit_windows(c)
        if baseline == "fedexdnn":
            train_local_exemplar(model, x_in, LOCAL_EPOCHS, DEVICE, lr=LR, beta=BETA, batch_size=BATCH_SIZE)
        else:
            train_local_recon(model, x_in, LOCAL_EPOCHS, DEVICE, lr=LR, batch_size=BATCH_SIZE)
        return len(x_in)

    print(f"\nfederated training: {ROUNDS} rounds x {LOCAL_EPOCHS} local epochs, baseline={baseline}")
    diagnostics_log = run_federated_rounds(
        clients, models, local_train_step, ROUNDS, GAMMA, DELTA,
        mode=baseline, num_clusters=num_clusters, get_fit_windows=get_fit_windows, device=DEVICE)

    score_fn = per_node_exemplar_deviation if baseline == "fedexdnn" else per_node_recon_error
    score_name = "exemplar_score" if baseline == "fedexdnn" else "recon_score"

    report = {"config": {"rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "baseline": baseline,
                          "num_clusters": num_clusters if baseline == "ifcaae" else None,
                          "num_prototypes": num_prototypes if baseline == "fedexdnn" else None,
                          "embed_dim": EMBED_DIM if baseline == "fedexdnn" else None,
                          "window_len": WINDOW_LEN, "num_nodes": num_nodes, "score_name": score_name},
              "nodes": cols, "alignment_log": diagnostics_log, "clients": {}}

    all_pairs = []
    for c, model, scaler in zip(clients, models, scalers):
        calib_w = scale_client(data, scaler, c.calib_idx)
        calib_raw = score_fn(model, calib_w, DEVICE)

        test_w = scale_client(data, scaler, c.test_normal_idx)
        test_raw = score_fn(model, test_w, DEVICE)
        scores_normal = gdn_score(calib_raw, test_raw)

        client_report = {"client_name": c.client_name, "fault_types": {}}
        rows = []
        for label_id_str, name in label_map.items():
            label_id = int(label_id_str)
            if label_id == 0:
                continue
            fault_idx = np.where(targets == label_id)[0]
            fault_idx = np.intersect1d(fault_idx, c.all_idx)
            if len(fault_idx) == 0:
                continue
            fault_w = scale_client(data, scaler, fault_idx)
            fault_raw = score_fn(model, fault_w, DEVICE)
            scores_fault = gdn_score(calib_raw, fault_raw)

            m = detection_metrics(scores_normal, scores_fault)
            if m is None:
                continue
            client_report["fault_types"][name] = {"n": int(len(fault_idx)), score_name: m}
            rows.append(m)
            all_pairs.append(m)
            print(f"  {c.client_name:<22}{name:<28}n={len(fault_idx):<5}{score_name}=auroc:{m['auroc']:.3f}")

        if rows:
            client_report["client_mean_metrics"] = {
                metric: float(np.mean([row[metric] for row in rows])) for metric in ("auroc", "auprc", "precision", "f1")
            }
        report["clients"][c.client_name] = client_report

    summary_overall = {
        metric: float(np.mean([row[metric] for row in all_pairs])) for metric in ("auroc", "auprc", "precision", "f1")
    } if all_pairs else {}
    print(f"\n{score_name:<16}{'AUROC':>10}{'AUPRC':>10}{'Precision':>12}{'F1':>10}")
    if summary_overall:
        print(f"{score_name:<16}{summary_overall['auroc']:>10.3f}{summary_overall['auprc']:>10.3f}"
              f"{summary_overall['precision']:>12.3f}{summary_overall['f1']:>10.3f}")

    report["summary_mean_auroc_overall"] = {score_name: summary_overall}
    suffix = out_suffix if out_suffix is not None else f"bck_federated_h1_{baseline}"
    out_json = OUT_DIR / f"alfa_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
               OUT_DIR / f"alfa_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


def main(horizon_mult=1, out_suffix=None, num_prototypes=NUM_PROTOTYPES, linkage="single",
         metric="cosine", edge_quantile=None, delta=None):
    delta = DELTA if delta is None else delta
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    forecast_h = STRIDE * horizon_mult

    print(f"loading ALFA, splitting into 6 fault-type federated clients, "
          f"horizon_mult={horizon_mult} ...")
    data, targets, cols, label_map, client_names, clients, window_flight = build_clients()
    num_nodes = len(cols)
    for c in clients:
        print(f"  client {c.client_id} ({c.client_name}): fit={len(c.fit_idx)} calib={len(c.calib_idx)} "
              f"test_normal={len(c.test_normal_idx)}")

    scalers = [fit_client_scaler(data, c.fit_idx) for c in clients]
    models = [JointPrototypeV21Forecast(num_nodes=num_nodes, window_size=WINDOW_LEN,
                                         embed_dim=EMBED_DIM, num_prototypes=num_prototypes,
                                         forecast_h=forecast_h, prior_edges=None, top_k=TOP_K).to(DEVICE)
              for _ in clients]

    shared_init = {k: v.clone() for k, v in models[0].state_dict().items() if not k.startswith("memory.")}
    for model in models[1:]:
        model.load_state_dict(shared_init, strict=False)

    fit_pairs = []
    for c in clients:
        vi, chains, _ = build_pairs(c.fit_idx, targets, window_flight, horizon_mult, True, True)
        fit_pairs.append((vi, data[vi], gather_future(data, chains, STRIDE)))

    def local_train_step(c, model):
        idx = c.client_id
        scaler = scalers[idx]
        vi, x_in_raw, x_future_raw = fit_pairs[idx]
        n, t, f = x_in_raw.shape
        x_in = scaler.transform(x_in_raw.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
        n2, t2, f2 = x_future_raw.shape
        x_future = (scaler.transform(x_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                    if n2 > 0 else x_future_raw)
        train_local(model, x_in, x_future, LOCAL_EPOCHS)
        return len(x_in)

    print(f"\nfederated training: {ROUNDS} rounds x {LOCAL_EPOCHS} local epochs, "
          f"memory-only exchange (gamma={GAMMA}, delta={delta})")
    diagnostics_log = run_federated_rounds(
        clients, models, local_train_step, ROUNDS, GAMMA, delta,
        sync_encoder_decoder=False, linkage=linkage,
        metric=metric, edge_quantile=edge_quantile)

    report = {"config": {"rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "num_prototypes": num_prototypes,
                          "embed_dim": EMBED_DIM, "window_len": WINDOW_LEN, "gamma": GAMMA, "delta": delta,
                          "linkage": linkage, "metric": metric, "edge_quantile": edge_quantile,
                          "reliability_ratio": RELIABILITY_RATIO, "horizon_mult": horizon_mult,
                          "forecast_h": forecast_h, "smooth_max_t": SMOOTH_MAX_T, "baseline": "ours"},
              "nodes": cols, "alignment_log": diagnostics_log, "clients": {}}

    # Pre-scan pass: calibrate each client's cov_head and collect calib-loss-
    # quality proxies (B/H/K), needed BEFORE any client's scoring since
    # `confidence_weights_from_losses` z-scores each signal ACROSS ALL
    # CLIENTS -- see federated_train_eval.loss_weighted_combo's docstring. Repeats
    # `cov_head.set_calibration` (idempotent, cheap at ALFA's scale) rather
    # than restructuring the single scoring loop below into two passes.
    b_loss, h_loss, k_loss = [], [], []
    for c, model, scaler in zip(clients, models, scalers):
        calib_arr = scale_client(data, scaler, c.calib_idx)
        _, d_node_calib_pre, _, idx_calib_pre, _ = per_sample_scores(model, calib_arr)
        cov_min = max(MIN_PROTO_SAMPLES, num_nodes + 1)
        model.cov_head.set_calibration(d_node_calib_pre, idx_calib_pre, min_samples=cov_min)
        _, d_node_calib_pre, _, _, d_mahal_calib_pre = per_sample_scores(model, calib_arr)
        b_loss.append(float(d_node_calib_pre.mean()))
        h_loss.append(float(np.median(d_mahal_calib_pre)))
        vi_c_pre, chains_c_pre, _ = build_pairs(c.calib_idx, targets, window_flight, horizon_mult, True, True)
        if len(vi_c_pre):
            calib_in_pre = scale_client(data, scaler, vi_c_pre)
            future_c_raw_pre = gather_future(data, chains_c_pre, STRIDE)
            n2, t2, f2 = future_c_raw_pre.shape
            calib_future_pre = scaler.transform(future_c_raw_pre.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
            k_resid_calib_pre = forecast_scores(model, calib_in_pre, calib_future_pre)
            k_loss.append(float(k_resid_calib_pre.mean()) if len(k_resid_calib_pre) else float("nan"))
        else:
            k_loss.append(float("nan"))
    lw_ready = all(not np.isnan(v) for v in k_loss)
    client_lw_weights = confidence_weights_from_losses({"FD": b_loss, "JD": h_loss, "PD": k_loss}) if lw_ready else None
    if client_lw_weights is None:
        print("  [loss-weighted fusion] skipped: at least one client has no forecast-pairing calib windows")
    else:
        for c, w in zip(clients, client_lw_weights):
            print(f"  [loss-weighted fusion] {c.client_name}: {w}")

    all_pairs = {}
    for ci, (c, model, scaler) in enumerate(zip(clients, models, scalers)):
        calib_arr = scale_client(data, scaler, c.calib_idx)
        test_normal_arr = scale_client(data, scaler, c.test_normal_idx)

        _, d_node_calib, _, idx_calib, _ = per_sample_scores(model, calib_arr)
        cov_min = max(MIN_PROTO_SAMPLES, num_nodes + 1)
        model.cov_head.set_calibration(d_node_calib, idx_calib, min_samples=cov_min)
        _, d_node_calib, _, idx_calib, d_mahal_calib = per_sample_scores(model, calib_arr)
        d_mahal_med = float(np.median(d_mahal_calib))
        d_mahal_iqr = max(float(np.percentile(d_mahal_calib, 75) - np.percentile(d_mahal_calib, 25)), 1e-8)
        node_proto_stats = fit_prototype_zscore_stats(d_node_calib, idx_calib, num_prototypes, MIN_PROTO_SAMPLES)

        _, d_node_normal, _, idx_normal, d_mahal_normal = per_sample_scores(model, test_normal_arr)

        vi_c, chains_c, mask_c = build_pairs(c.calib_idx, targets, window_flight, horizon_mult, True, True)
        calib_in = scale_client(data, scaler, vi_c)
        future_c_raw = gather_future(data, chains_c, STRIDE)
        if len(vi_c):
            n2, t2, f2 = future_c_raw.shape
            calib_future = scaler.transform(future_c_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
        else:
            calib_future = future_c_raw
        k_resid_calib = forecast_scores(model, calib_in, calib_future)
        idx_k_calib = idx_calib[mask_c]
        k_proto_stats = (fit_prototype_zscore_stats(k_resid_calib, idx_k_calib, num_prototypes, MIN_PROTO_SAMPLES)
                          if len(k_resid_calib) else None)
        vi_n, chains_n, mask_n = build_pairs(c.test_normal_idx, targets, window_flight, horizon_mult, True, True)
        test_normal_in = scale_client(data, scaler, vi_n)
        future_n_raw = gather_future(data, chains_n, STRIDE)
        if len(vi_n):
            n2, t2, f2 = future_n_raw.shape
            test_normal_future = scaler.transform(future_n_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
        else:
            test_normal_future = future_n_raw
        k_resid_normal = forecast_scores(model, test_normal_in, test_normal_future)

        has_forecast = len(k_resid_calib) > 0

        _, node_iqr = zscore(d_node_normal, d_node_calib)
        reliable_mask = node_iqr >= RELIABILITY_RATIO * np.median(node_iqr)
        if has_forecast:
            _, forecast_iqr = zscore(k_resid_normal, k_resid_calib)
            forecast_mask = forecast_iqr >= RELIABILITY_RATIO * np.median(forecast_iqr)
        else:
            forecast_mask = np.ones(num_nodes, dtype=bool)

        # B is now the per-prototype ("by operating regime") z-score, not the
        # global-calib version -- see memory/robo-fleet-num-prototypes-sweep.md.
        z_node_normal_proto = zscore_prototype(d_node_normal, idx_normal, node_proto_stats)
        b_normal_full = z_node_normal_proto[:, reliable_mask].max(axis=1)
        h_normal_full = (d_mahal_normal - d_mahal_med) / d_mahal_iqr
        base_normal = {"FD_max": b_normal_full, "JD_mahal": h_normal_full,
                        "FD_JD_max": smooth_max(np.stack([b_normal_full, h_normal_full], axis=1), SMOOTH_MAX_T)}

        if has_forecast:
            b_masked, h_masked = base_normal["FD_max"][mask_n], base_normal["JD_mahal"][mask_n]
            # K is also now the per-prototype version.
            z_forecast_normal_proto = zscore_prototype(k_resid_normal, idx_normal[mask_n], k_proto_stats)
            k_normal = z_forecast_normal_proto[:, forecast_mask].max(axis=1)
            scores_normal = dict(base_normal)
            scores_normal["PD_max"] = k_normal
            scores_normal["FD_PD_max"] = smooth_max(np.stack([b_masked, k_normal], axis=1), SMOOTH_MAX_T)
            scores_normal["FD_JD_PD_max"] = smooth_max(np.stack([b_masked, h_masked, k_normal], axis=1), SMOOTH_MAX_T)
            if client_lw_weights is not None:
                scores_normal["FD_JD_PD_lw"] = loss_weighted_combo(
                    {"FD": b_masked, "JD": h_masked, "PD": k_normal}, client_lw_weights[ci])
        else:
            scores_normal = dict(base_normal)

        client_report = {"client_name": c.client_name,
                          "excluded_nodes_node": [cols[i] for i in range(num_nodes) if not reliable_mask[i]],
                          "excluded_nodes_forecast": [cols[i] for i in range(num_nodes) if not forecast_mask[i]],
                          "num_prototypes": num_prototypes,
                          "has_forecast": has_forecast,
                          "fault_types": {}}
        rows = {k: [] for k in scores_normal}
        for label_id_str, name in label_map.items():
            label_id = int(label_id_str)
            if label_id == 0:
                continue
            fault_idx = np.where(targets == label_id)[0]
            fault_idx = np.intersect1d(fault_idx, c.all_idx)
            if len(fault_idx) == 0:
                continue
            fault_arr = scale_client(data, scaler, fault_idx)
            _, d_node_f, _, idx_f, d_mahal_f = per_sample_scores(model, fault_arr)
            z_node_f_proto = zscore_prototype(d_node_f, idx_f, node_proto_stats)
            b_f_full = z_node_f_proto[:, reliable_mask].max(axis=1)
            h_f_full = (d_mahal_f - d_mahal_med) / d_mahal_iqr
            base_f = {"FD_max": b_f_full, "JD_mahal": h_f_full,
                      "FD_JD_max": smooth_max(np.stack([b_f_full, h_f_full], axis=1), SMOOTH_MAX_T)}

            mask_f = np.zeros(len(fault_idx), dtype=bool)
            if has_forecast:
                vi_f, chains_f, mask_f = build_pairs(fault_idx, targets, window_flight, horizon_mult, False, False)
                fault_in = scale_client(data, scaler, vi_f)
                future_f_raw = gather_future(data, chains_f, STRIDE)
                if len(vi_f):
                    n2, t2, f2 = future_f_raw.shape
                    fault_future = scaler.transform(future_f_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                else:
                    fault_future = future_f_raw
                k_resid_f = forecast_scores(model, fault_in, fault_future)
                b_f_masked, h_f_masked = base_f["FD_max"][mask_f], base_f["JD_mahal"][mask_f]
                if len(k_resid_f):
                    z_forecast_f_proto = zscore_prototype(k_resid_f, idx_f[mask_f], k_proto_stats)
                    k_f = z_forecast_f_proto[:, forecast_mask].max(axis=1)
                else:
                    k_f = np.zeros(0)
                scores_fault = dict(base_f)
                scores_fault["PD_max"] = k_f
                scores_fault["FD_PD_max"] = (smooth_max(np.stack([b_f_masked, k_f], axis=1), SMOOTH_MAX_T)
                                           if len(k_f) else np.zeros(0))
                scores_fault["FD_JD_PD_max"] = (smooth_max(np.stack([b_f_masked, h_f_masked, k_f], axis=1), SMOOTH_MAX_T)
                                            if len(k_f) else np.zeros(0))
                if client_lw_weights is not None and len(k_f):
                    scores_fault["FD_JD_PD_lw"] = loss_weighted_combo(
                        {"FD": b_f_masked, "JD": h_f_masked, "PD": k_f}, client_lw_weights[ci])
            else:
                scores_fault = dict(base_f)

            metrics = metrics_dict(scores_normal, scores_fault)
            for key, v in metrics.items():
                rows[key].append(v)
                all_pairs.setdefault(key, []).append(v)
            client_report["fault_types"][name] = {"n": int(len(fault_idx)), **metrics}
            print(f"  {c.client_name:<22}{name:<28}n={len(fault_idx):<5}"
                  + "  ".join(f"{k}=auroc:{v['auroc']:.3f}" for k, v in metrics.items()))

        if rows["FD_max"]:
            client_report["client_mean_metrics"] = {
                k: {m: float(np.mean([row[m] for row in v])) for m in ("auroc", "auprc", "precision", "f1")}
                for k, v in rows.items() if v
            }
        report["clients"][c.client_name] = client_report

    summary_overall = {
        k: {m: float(np.mean([row[m] for row in v])) for m in ("auroc", "auprc", "precision", "f1")}
        for k, v in all_pairs.items() if v
    }
    print(f"\n{'method':<16}{'AUROC':>10}{'AUPRC':>10}{'Precision':>12}{'F1':>10}")
    for key, m in summary_overall.items():
        print(f"{key:<16}{m['auroc']:>10.3f}{m['auprc']:>10.3f}{m['precision']:>12.3f}{m['f1']:>10.3f}")

    report["summary_mean_auroc_overall"] = summary_overall
    proto_suffix = f"_m{num_prototypes}" if num_prototypes != NUM_PROTOTYPES else ""
    metric_suffix = f"_metric_{metric}" if metric != "cosine" else ""
    suffix = out_suffix if out_suffix is not None else f"bck_federated_h{horizon_mult}{proto_suffix}{metric_suffix}"
    out_json = OUT_DIR / f"alfa_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
               OUT_DIR / f"alfa_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-mult", type=int, default=1)
    parser.add_argument("--out-suffix", type=str, default=None)
    parser.add_argument("--num-prototypes", type=int, default=NUM_PROTOTYPES)
    parser.add_argument("--linkage", type=str, default="single", choices=["single", "complete"])
    parser.add_argument("--baseline", type=str, default="ours",
                         choices=["ours", "fedavg", "ifcaae", "fedexdnn", "fedpro", "fedcpg", "ufedhy"],
                         help="server-aggregation strategy -- see run_federated_rounds' docstring "
                              "('fedpro'/'fedcpg' are supervised -- see fedpro.py/fedcpg.py; "
                              "'ufedhy' is fit-on-healthy -- see ufedhy.py)")
    parser.add_argument("--num-clusters", type=int, default=2,
                         help="IFCAAE only: number of global model clusters")
    parser.add_argument("--delta", type=float, default=None,
                         help="override DELTA (default module constant, 0.5) -- cross-client "
                              "cosine-similarity edge threshold for align_and_split")
    args = parser.parse_args()
    if args.baseline == "ours":
        main(horizon_mult=args.horizon_mult, out_suffix=args.out_suffix, num_prototypes=args.num_prototypes,
             linkage=args.linkage, delta=args.delta)
    elif args.baseline == "fedpro":
        main_fedpro_baseline(out_suffix=args.out_suffix)
    elif args.baseline == "fedcpg":
        main_fedcpg_baseline(out_suffix=args.out_suffix)
    elif args.baseline == "ufedhy":
        main_ufedhy_baseline(out_suffix=args.out_suffix)
    else:
        main_faithful_baseline(args.baseline, out_suffix=args.out_suffix, num_clusters=args.num_clusters,
                                num_prototypes=args.num_prototypes)
