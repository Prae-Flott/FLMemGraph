#!/usr/bin/env python3
"""
FEDERATED JointPrototypeV21Forecast (V2.1 + ForecastHead, signal K) on
robo_fleet's 4 real robots (`rob_00`..`rob_03`) as 4 federated clients --
the own-fleet dataset from `report/040_experiment.tex` (4 iRobot Create3
units, 26 recorded ROS2-topic features, 4 induced mechanical faults),
built by `data/robo_fleet/` + `src/dataloaders/robo_fleet/build_robo_fleet.py`
from the raw `data/robo_pdm/` session CSVs. Near-identical script to
`run_robo3er_bck_federated.py` (same `load_robo3er`/`load_fl_clients`
loader family, `data_dir` pointed at `data/robo_fleet` instead -- see
`src/dataloaders/robo_fleet/dataset.py`'s `load_robo3er` docstring for why
that loader is dataset-directory-agnostic), differing only in client
count (4 vs. 5) and DATA_DIR/OUT_DIR.

Reports `B_node_max`/`H_cov_mahal`/`BH_max`/`K_forecast_max`/`BK_max`/
`BHK_max` -- H (`cov_head`'s Mahalanobis prototype-deviation signal) is
calibrated and scored here (not just in the localization diagnose
script), matching the BHK-mainline promotion in
`src/federated/federated_train_eval.py`. `forecast_head` (K) IS FedAvg'd whenever
`SYNC_ENCODER_DECODER=True` (the shipped default here) -- its own
parameters (`attn_w`/`embeddings`/`attn_a`/`out_proj`) match the
`("encoder.", "decoder.", "forecast_head.")` FedAvg prefix filter.
`edge_head` (generic cross-node attention residual, signal C) is still
trained (its loss term still regularizes the shared encoder) but not
scored in this detection report. **2026-09-12: switched from
`JointPrototypeV31Forecast` to `V21Forecast`, dropping the declared-
physics-edge `typed_head`/signal-E machinery project-wide** -- a
controlled ablation (`lambda_edge=lambda_typed=0` vs. the shipped
`0.5/0.5`, same V31 model) found it worth only ~0.005 BHK_max here, and
the same simpler architecture ALFA/SMD already used (no declared edges)
is now the one mainline across every dataset -- see
`memory/v21-mainline-switch.md`.

Same 26-node feature-group-scaled node set, `NUM_PROTOTYPES=2`, per-client
pairing (same robot, boundary-aware) as robo3er -- see
`run_robo3er_bck.py`'s module docstring for why each of those choices
exists.

The training+eval mainline (round loop, scoring primitives, `train_local`)
lives in `src/federated/federated_train_eval.py`, shared verbatim with
`run_paderborn_bck_federated.py`/`run_me_ad_bck_federated.py` (all
`JointPrototypeV21Forecast` as of 2026-09-12) -- this script only supplies
robo_fleet-specific data loading/client construction/pairing and CLI
plumbing.

Usage:
    python3 run_robo_fleet_bck_federated.py [--horizon-mult M] [--no-forecast-prior]
        [--out-suffix NAME]

All z-score reference stats (B's `two_stage_group_score`/`gdn_score`, H's
median/IQR, K's `federated_train_eval.gdn_score`) use the per-client held-out
`calib_idx` split (global median/IQR, no per-prototype conditioning for
B/K; H's `cov_head` IS per-prototype-conditioned, with a global fallback
for sparsely-populated prototypes -- see `DeviationCovarianceHead`).
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
sys.path.insert(0, str(REPO_ROOT / "src" / "dataloaders" / "robo_fleet"))
sys.path.insert(0, str(REPO_ROOT / "src" / "federated"))
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "FedPRO"))
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "FedCPG"))
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "uFedHy-DisMTSADD"))
from joint_prototype_model import JointPrototypeV21Forecast  # noqa: E402
from baseline_models import (ReconOnlyModel, ExemplarOnlyModel, FedPROModel,  # noqa: E402
                              FedCPGModel, SCNorTransformerModel, Hypernetwork)
from dataset import load_robo3er  # noqa: E402
import feature_groups  # noqa: E402
from federated_train_eval import (two_stage_group_score, base_scores,  # noqa: E402
                           add_forecast_scores, add_h_score, per_sample_scores, forecast_scores,
                           train_local, run_federated_rounds, gdn_score, detection_metrics,
                           two_stage_group_score_proto, gdn_score_proto, smooth_max,
                           loss_weighted_combo, train_local_recon, train_local_exemplar,
                           per_node_recon_error, per_node_exemplar_deviation)
from fedpro import (train_local_classifier, embed_all, build_initial_prototypes_kmeans,  # noqa: E402
                     refine_prototypes_margin, retrieve_and_vote, confidence_ensemble)
from fedcpg import train_local_fedcpg, aggregate_global_prototypes, predict_anomaly_score  # noqa: E402
from ufedhy import hypernet_round_update  # noqa: E402
from federated_memory import confidence_weights_from_losses  # noqa: E402
from comm_cost import tensor_bytes, state_dict_bytes, round_comm_record  # noqa: E402

SYNC_ENCODER_DECODER = True

DATA_DIR = REPO_ROOT / "data" / "robo_fleet"
OUT_DIR = REPO_ROOT / "checkpoints" / "robo_fleet"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGES_NAMED = [
    ("wheel_vels_velocity_left", "odom_odo_lintw_x"),
    ("wheel_vels_velocity_right", "odom_odo_lintw_x"),
    ("wheel_vels_velocity_left", "odom_odo_angtw_z"),
    ("wheel_vels_velocity_right", "odom_odo_angtw_z"),
    ("odom_odo_angtw_z", "imu_imu_angvel_z"),
    ("wheel_status_current_ma_left", "wheel_vels_velocity_left"),
    ("wheel_status_current_ma_right", "wheel_vels_velocity_right"),
]
# EDGE_TYPES (per-edge "proportional"/"nonlinear" typing) removed 2026-09-12
# along with the switch to JointPrototypeV21Forecast -- see that class's
# docstring and memory/v21-mainline-switch.md. EDGES_NAMED is still used:
# `JointPrototypeV21`'s edge_head keeps declared edges as a soft, LEARNED-
# STRENGTH additive bias on its generic attention (not a hard requirement),
# it just no longer feeds a separate typed-relation head.

WINDOW_LEN = 60
STRIDE = 32
BATCH_SIZE = 256
ROUNDS = 5
LOCAL_EPOCHS = 12
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 2
TOP_K = 8
FEATURE_GROUPS = ["kinematic_core", "actuation"]
EXCLUDE_NODES_PREFIXES = ("tf_link_base_link_", "tf_footprint_base_footprint_",
                           "odom_odo_pos_", "wheel_ticks_")
INCLUDE_EXTRA_NODES = []
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_FORECAST = 0.5
MIN_PROTO_SAMPLES = 5
TOP_K_AGG = 3
GAMMA = 0.5
DELTA = 0.5
FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Client:
    def __init__(self, client_id, robot_name, fit_idx, calib_idx, test_normal_idx, fault_idx_by_label):
        self.client_id = client_id
        self.robot_name = robot_name
        self.fit_idx = fit_idx
        self.calib_idx = calib_idx
        self.test_normal_idx = test_normal_idx
        self.fault_idx_by_label = fault_idx_by_label


def build_clients():
    data_full, targets, cols_full, label_map, robot_map = load_robo3er(drop_dead=True, data_dir=DATA_DIR)
    selected = set(feature_groups.active_feature_names(FEATURE_GROUPS))
    selected -= {c for c in cols_full if c.startswith(EXCLUDE_NODES_PREFIXES)}
    selected |= set(INCLUDE_EXTRA_NODES)
    keep_idx = [i for i, c in enumerate(cols_full) if c in selected]
    data, cols = data_full[:, :, keep_idx], [cols_full[i] for i in keep_idx]
    data = data.astype(np.float32)

    with open(DATA_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    client_indices = partition["data_indices"]

    window_robot = np.full(len(targets), -1, dtype=int)
    clients = []
    for client_id, idx_dict in enumerate(client_indices):
        all_idx = np.array(idx_dict["train"] + idx_dict["val"] + idx_dict["test"], dtype=int)
        window_robot[all_idx] = client_id
        normal_idx = np.sort(all_idx[targets[all_idx] == 0])
        n = len(normal_idx)
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        fit_idx = normal_idx[:n_fit]
        calib_idx = normal_idx[n_fit : n_fit + n_calib]
        test_normal_idx = normal_idx[n_fit + n_calib :]

        fault_idx_by_label = {}
        for label_id in sorted(set(targets[all_idx].tolist())):
            if label_id == 0:
                continue
            fault_idx = all_idx[targets[all_idx] == label_id]
            if len(fault_idx) > 0:
                fault_idx_by_label[label_id] = fault_idx

        clients.append(Client(
            client_id=client_id,
            robot_name=robot_map.get(str(client_id), f"robot{client_id:02d}"),
            fit_idx=fit_idx, calib_idx=calib_idx, test_normal_idx=test_normal_idx,
            fault_idx_by_label=fault_idx_by_label,
        ))
    return data, targets, label_map, clients, cols, window_robot


def build_pairs(idx_arr, targets, window_robot, horizon_mult, require_next_normal, require_same_split=False):
    N = len(targets)
    idx_set = set(idx_arr.tolist()) if require_same_split else None
    vi, chains, mask = [], [], []
    for i in idx_arr:
        chain = [i + k for k in range(1, horizon_mult + 1)]
        valid = True
        for j in chain:
            if (j >= N or window_robot[j] != window_robot[i]
                    or (require_next_normal and targets[j] != 0)
                    or (require_same_split and j not in idx_set)):
                valid = False
                break
        mask.append(valid)
        if valid:
            vi.append(i)
            chains.append(chain)
    return np.array(vi, dtype=int), chains, np.array(mask, dtype=bool)


def gather_future(data_scaled, chains, stride):
    if len(chains) == 0:
        return np.zeros((0, 0, data_scaled.shape[-1]), dtype=data_scaled.dtype)
    segments = [data_scaled[[c[k] for c in chains]][:, -stride:, :] for k in range(len(chains[0]))]
    return np.concatenate(segments, axis=1)


def scale_client(data, scaler, idx):
    n, t, f = data[idx].shape
    return scaler.transform(data[idx].reshape(-1, f)).reshape(n, t, f).astype(np.float32)


def fit_client_scaler(data, fit_idx):
    scaler = StandardScaler()
    scaler.fit(data[fit_idx].reshape(-1, data.shape[-1]))
    return scaler


def main_fedpro_baseline(out_suffix=None, k_per_class=3, top_k=3, tau=0.1, embed_dim=EMBED_DIM,
                          fedavg_rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, proto_epochs=50, proto_lr=1e-3):
    """FedPRO (Zhou et al., IEEE TC 2026) -- robo_fleet ONLY, a deliberate
    exception to the fit-on-healthy convention: supervised multi-class
    classification (Normal + robo_fleet's 4 fault types = 5 classes),
    trained on ALL labeled data per client, not just normal. See
    `benchmark/FedPRO/fedpro.py`'s module docstring for the full rationale
    and `docs/Prototype_Retrieval-Augmented_Federated_Learning_System_for_Robust_Intrusion_Detection.pdf`
    for the method itself."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print("loading robo_fleet, splitting into 4 real per-robot federated clients, baseline=fedpro "
          "(supervised, all labels, NOT fit-on-healthy) ...")
    data, targets, label_map, clients, cols, window_robot = build_clients()
    num_nodes = len(cols)
    num_classes = len(label_map)

    # FedPRO-specific per-client stratified 70/30 train/test split over
    # ALL labeled indices (normal + every fault type this client has) --
    # genuinely different from every other baseline's fit-on-healthy split.
    client_splits = []
    for c in clients:
        idx_by_label = {0: np.concatenate([c.fit_idx, c.calib_idx, c.test_normal_idx])}
        idx_by_label.update(c.fault_idx_by_label)
        train_idx, train_y, test_idx, test_y = [], [], [], []
        for label_id, idx in idx_by_label.items():
            idx = np.asarray(idx)
            perm = rng.permutation(len(idx))
            n_train = max(1, int(0.7 * len(idx)))
            tr, te = idx[perm[:n_train]], idx[perm[n_train:]]
            train_idx.append(tr); train_y.append(np.full(len(tr), label_id, dtype=np.int64))
            test_idx.append(te); test_y.append(np.full(len(te), label_id, dtype=np.int64))
        client_splits.append({
            "train_idx": np.concatenate(train_idx), "train_y": np.concatenate(train_y),
            "test_idx": np.concatenate(test_idx), "test_y": np.concatenate(test_y),
        })
        print(f"  client {c.client_id} ({c.robot_name}): train={len(client_splits[-1]['train_idx'])} "
              f"test={len(client_splits[-1]['test_idx'])} classes={sorted(idx_by_label.keys())}")

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
    comm_log = run_federated_rounds(clients, models, local_train_step, fedavg_rounds, GAMMA, DELTA, mode="fedavg")

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
        print(f"  client {c.client_id} ({c.robot_name}): {len(protos)} prototypes "
              f"across {len(set(proto_labels.tolist()))} classes")
    bank_protos = np.concatenate(all_protos, axis=0)
    bank_labels = np.concatenate(all_proto_labels, axis=0)
    print(f"  global prototype bank: {len(bank_protos)} prototypes from {len(clients)} clients")
    # comm cost, stage 2 (one-shot, not per-round): each client uploads its
    # own refined prototypes once; every client then downloads the FULL
    # concatenated bank for retrieval-augmented inference (no cross-client
    # alignment happens server-side, unlike "ours"/fedexdnn -- see module
    # docstring's "the server just concatenates").
    stage2_bytes_up = sum(tensor_bytes(p) + tensor_bytes(pl) for p, pl in zip(all_protos, all_proto_labels))
    stage2_bytes_down = len(clients) * (tensor_bytes(bank_protos) + tensor_bytes(bank_labels))
    comm_log.append({"round": "stage2_prototype_bank_exchange",
                      **round_comm_record(stage2_bytes_up, stage2_bytes_down)})

    print("\nprototype retrieval-augmented inference on each client's held-out test split ...")
    report = {"config": {"baseline": "fedpro", "fedavg_rounds": fedavg_rounds, "local_epochs": local_epochs,
                          "embed_dim": embed_dim, "num_classes": num_classes, "k_per_class": k_per_class,
                          "top_k": top_k, "tau": tau, "proto_epochs": proto_epochs,
                          "split": "stratified_70_30_all_labels_NOT_fit_on_healthy",
                          "note": "FedPRO is a supervised multi-class classifier wrapper (Zhou et al., "
                                  "IEEE TC 2026) -- robo_fleet-only exception to the fit-on-healthy policy, "
                                  "see benchmark/FedPRO/fedpro.py"},
              "communication_log": comm_log,
              "communication_total_bytes": {
                  "up": sum(d["comm_bytes_up"] for d in comm_log),
                  "down": sum(d["comm_bytes_down"] for d in comm_log),
              },
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
        anomaly_score = 1.0 - y_en[:, 0]  # bridged AUROC score: P(not-normal)

        client_report = {"robot_name": c.robot_name, "accuracy": accuracy, "fault_types": {}}
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
            print(f"  {c.robot_name:<10}{name:<16}n={int(fault_mask.sum()):<5}"
                  f"accuracy={accuracy:.3f}  fedpro_score=auroc:{m['auroc']:.3f}")
        report["clients"][c.robot_name] = client_report

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
    suffix = out_suffix if out_suffix is not None else "forecast_v2_federated_h10_encdec_synced_fedpro"
    out_json = OUT_DIR / f"robo_fleet_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "bank_protos": bank_protos,
                "bank_labels": bank_labels, "report": report}, OUT_DIR / f"robo_fleet_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


def main_fedcpg_baseline(out_suffix=None, embed_dim=EMBED_DIM, proj_dim=32, fedcpg_rounds=ROUNDS,
                          local_epochs=LOCAL_EPOCHS, alpha=0.5, beta=0.2, tau=0.1):
    """FedCPG (Li et al., *Computers in Industry* 2025) -- robo_fleet ONLY,
    same fit-on-ALL-labels exception as FedPRO (see `fedpro.py`'s module
    docstring): class-prototype contrast needs multiple known classes at
    training time. See `benchmark/FedCPG/fedcpg.py` for the algorithm and
    `docs/1-s2.0-S0166361524001088-main.pdf` for the paper."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print("loading robo_fleet, splitting into 4 real per-robot federated clients, baseline=fedcpg "
          "(supervised, all labels, NOT fit-on-healthy) ...")
    data, targets, label_map, clients, cols, window_robot = build_clients()
    num_nodes = len(cols)
    num_classes = len(label_map)

    client_splits = []
    for c in clients:
        idx_by_label = {0: np.concatenate([c.fit_idx, c.calib_idx, c.test_normal_idx])}
        idx_by_label.update(c.fault_idx_by_label)
        train_idx, train_y, test_idx, test_y = [], [], [], []
        for label_id, idx in idx_by_label.items():
            idx = np.asarray(idx)
            perm = rng.permutation(len(idx))
            n_train = max(1, int(0.7 * len(idx)))
            tr, te = idx[perm[:n_train]], idx[perm[n_train:]]
            train_idx.append(tr); train_y.append(np.full(len(tr), label_id, dtype=np.int64))
            test_idx.append(te); test_y.append(np.full(len(te), label_id, dtype=np.int64))
        client_splits.append({
            "train_idx": np.concatenate(train_idx), "train_y": np.concatenate(train_y),
            "test_idx": np.concatenate(test_idx), "test_y": np.concatenate(test_y),
        })
        print(f"  client {c.client_id} ({c.robot_name}): train={len(client_splits[-1]['train_idx'])} "
              f"test={len(client_splits[-1]['test_idx'])} classes={sorted(idx_by_label.keys())}")

    scalers = [StandardScaler().fit(data[s["train_idx"]].reshape(-1, num_nodes)) for s in client_splits]
    models = [FedCPGModel(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=embed_dim,
                           proj_dim=proj_dim, num_classes=num_classes).to(DEVICE) for _ in clients]

    def scale(idx, scaler):
        n, t, f = data[idx].shape
        return scaler.transform(data[idx].reshape(-1, f)).reshape(n, t, f).astype(np.float32)

    print(f"\nfederated training (backbone FedAvg + personalized head + class-prototype contrast): "
          f"{fedcpg_rounds} rounds x {local_epochs} local epochs, baseline=fedcpg")
    global_protos = torch.zeros(num_classes, proj_dim)
    comm_log = []
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
        # backbone-only FedAvg (paper Eq. (2)) -- head stays local/personalized, never averaged.
        state_dicts = [{k: v for k, v in m.state_dict().items() if k.startswith(("encoder.", "proj."))}
                       for m in models]
        total = sum(fit_counts)
        backbone_avg = {k: sum(sd[k] * (n / total) for sd, n in zip(state_dicts, fit_counts))
                        for k in state_dicts[0]}
        # comm cost: each client uploads its backbone subset + its local
        # class prototypes/counts; server broadcasts back the averaged
        # backbone + updated global prototypes to every client.
        bytes_up = sum(state_dict_bytes(sd) for sd in state_dicts)
        bytes_up += sum(tensor_bytes(p) + tensor_bytes(n) for p, n in zip(client_protos, client_counts))
        bytes_down = len(models) * (state_dict_bytes(backbone_avg) + tensor_bytes(global_protos))
        for m in models:
            m.load_state_dict(backbone_avg, strict=False)
        global_protos = aggregate_global_prototypes(client_protos, client_counts)
        comm_log.append({"round": rnd, **round_comm_record(bytes_up, bytes_down)})
        print(f"  round {rnd}: aggregated backbone + {int((global_protos.abs().sum(dim=1) > 0).sum())}"
              f"/{num_classes} global class prototypes")

    print("\nevaluating personalized heads on each client's held-out test split ...")
    report = {"config": {"baseline": "fedcpg", "fedcpg_rounds": fedcpg_rounds, "local_epochs": local_epochs,
                          "embed_dim": embed_dim, "proj_dim": proj_dim, "num_classes": num_classes,
                          "alpha": alpha, "beta": beta, "tau": tau,
                          "split": "stratified_70_30_all_labels_NOT_fit_on_healthy",
                          "note": "FedCPG (Li et al., Computers in Industry 2025) -- robo_fleet-only "
                                  "exception to the fit-on-healthy policy, see benchmark/FedCPG/fedcpg.py"},
              "communication_log": comm_log,
              "communication_total_bytes": {
                  "up": sum(d["comm_bytes_up"] for d in comm_log),
                  "down": sum(d["comm_bytes_down"] for d in comm_log),
              },
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

        client_report = {"robot_name": c.robot_name, "accuracy": accuracy, "fault_types": {}}
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
            print(f"  {c.robot_name:<10}{name:<16}n={int(fault_mask.sum()):<5}"
                  f"accuracy={accuracy:.3f}  fedcpg_score=auroc:{m['auroc']:.3f}")
        report["clients"][c.robot_name] = client_report

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
    suffix = out_suffix if out_suffix is not None else "forecast_v2_federated_h10_encdec_synced_fedcpg"
    out_json = OUT_DIR / f"robo_fleet_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
               OUT_DIR / f"robo_fleet_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


def main_ufedhy_baseline(out_suffix=None, hyper_embed_dim=32, hyper_hidden_dim=64,
                          transformer_embed_dim=16, ufedhy_rounds=ROUNDS, local_epochs=LOCAL_EPOCHS,
                          hyper_lr=1e-3):
    """uFedHy-DisMTSADD (Hao et al., *Information Processing & Management*
    2025) -- fit-on-healthy, runs like FedAvg/IFCAAE/Fed-ExDNN (all
    datasets, not robo_fleet-only). See `benchmark/uFedHy-DisMTSADD/ufedhy.py` for the
    hypernetwork update rule and `docs/1-s2.0-S0306457325000494-main.pdf`
    for the paper."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"loading robo_fleet, splitting into 4 real per-robot federated clients, baseline=ufedhy ...")
    data, targets, label_map, clients, cols, window_robot = build_clients()
    num_nodes = len(cols)
    for c in clients:
        print(f"  client {c.client_id} ({c.robot_name}): fit={len(c.fit_idx)} calib={len(c.calib_idx)} "
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
                          "feature_groups": FEATURE_GROUPS, "score_name": "recon_score",
                          "note": "localization uses this project's own per-node argmax convention, NOT "
                                  "the paper's own PC-algorithm+PageRank causal diagnosis -- see "
                                  "benchmark/uFedHy-DisMTSADD/ufedhy.py's module docstring"},
              "alignment_log": diagnostics_log, "clients": {}}

    all_pairs = []
    for c, model, scaler in zip(clients, models, scalers):
        calib_w = scale_client(data, scaler, c.calib_idx)
        calib_raw = per_node_recon_error(model, calib_w, DEVICE)

        report["clients"].setdefault(c.robot_name, {"fault_types": {}})
        if not c.fault_idx_by_label:
            print(f"client {c.client_id} ({c.robot_name}): no fault windows, skipping evaluation")
            continue

        test_w = scale_client(data, scaler, c.test_normal_idx)
        test_raw = per_node_recon_error(model, test_w, DEVICE)
        scores_normal = gdn_score(calib_raw, test_raw)

        for label_id, fault_idx in c.fault_idx_by_label.items():
            name = label_map[str(label_id)]
            fault_w = scale_client(data, scaler, fault_idx)
            fault_raw = per_node_recon_error(model, fault_w, DEVICE)
            scores_fault = gdn_score(calib_raw, fault_raw)

            m = detection_metrics(scores_normal, scores_fault)
            if m is None:
                continue
            report["clients"][c.robot_name]["fault_types"][name] = {"n": int(len(fault_idx)), "recon_score": m}
            all_pairs.append(m)
            print(f"  {c.robot_name:<10}{name:<16}n={len(fault_idx):<5}recon_score=auroc:{m['auroc']:.3f}")

    summary_overall = {
        m: float(np.mean([row[m] for row in all_pairs])) for m in ("auroc", "auprc", "precision", "f1")
    } if all_pairs else {}
    print(f"\n{'recon_score':<16}{'AUROC':>10}{'AUPRC':>10}{'Precision':>12}{'F1':>10}")
    if summary_overall:
        print(f"{'recon_score':<16}{summary_overall['auroc']:>10.3f}{summary_overall['auprc']:>10.3f}"
              f"{summary_overall['precision']:>12.3f}{summary_overall['f1']:>10.3f}")

    report["summary_mean_auroc_overall"] = {"recon_score": summary_overall}
    suffix = out_suffix if out_suffix is not None else "forecast_v2_federated_h10_encdec_synced_ufedhy"
    out_json = OUT_DIR / f"robo_fleet_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models],
                "hypernet_state_dict": hypernet.state_dict(), "report": report},
               OUT_DIR / f"robo_fleet_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


def main_faithful_baseline(baseline, out_suffix=None, num_clusters=2, num_prototypes=NUM_PROTOTYPES):
    """FedAvg / IFCAAE / Fed-ExDNN, each with its OWN minimal architecture
    and own single anomaly score (no B/H/K/BK/BHK) -- see
    `src/models/baseline_models.py`'s module docstring and
    `benchmark/baselines/registry.py`. No forecast pairing/chains needed
    (none of these three baselines forecast): plain per-client fit/calib/
    fault windows only."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"loading robo_fleet, splitting into 4 real per-robot federated clients, "
          f"{FEATURE_GROUPS} as nodes, baseline={baseline} ...")
    data, targets, label_map, clients, cols, window_robot = build_clients()
    num_nodes = len(cols)
    for c in clients:
        print(f"  client {c.client_id} ({c.robot_name}): fit={len(c.fit_idx)} calib={len(c.calib_idx)} "
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
                          "window_len": WINDOW_LEN, "num_nodes": num_nodes,
                          "feature_groups": FEATURE_GROUPS, "score_name": score_name},
              "alignment_log": diagnostics_log, "clients": {}}

    all_pairs = []
    for c, model, scaler in zip(clients, models, scalers):
        calib_w = scale_client(data, scaler, c.calib_idx)
        calib_raw = score_fn(model, calib_w, DEVICE)

        report["clients"].setdefault(c.robot_name, {"fault_types": {}})
        if not c.fault_idx_by_label:
            print(f"client {c.client_id} ({c.robot_name}): no fault windows, skipping evaluation")
            continue

        test_w = scale_client(data, scaler, c.test_normal_idx)
        test_raw = score_fn(model, test_w, DEVICE)
        scores_normal = gdn_score(calib_raw, test_raw)

        for label_id, fault_idx in c.fault_idx_by_label.items():
            name = label_map[str(label_id)]
            fault_w = scale_client(data, scaler, fault_idx)
            fault_raw = score_fn(model, fault_w, DEVICE)
            scores_fault = gdn_score(calib_raw, fault_raw)

            m = detection_metrics(scores_normal, scores_fault)
            if m is None:
                continue
            report["clients"][c.robot_name]["fault_types"][name] = {"n": int(len(fault_idx)), score_name: m}
            all_pairs.append(m)
            print(f"  {c.robot_name:<10}{name:<16}n={len(fault_idx):<5}{score_name}=auroc:{m['auroc']:.3f}")

    summary_overall = {
        m: float(np.mean([row[m] for row in all_pairs])) for m in ("auroc", "auprc", "precision", "f1")
    } if all_pairs else {}
    print(f"\n{score_name:<16}{'AUROC':>10}{'AUPRC':>10}{'Precision':>12}{'F1':>10}")
    if summary_overall:
        print(f"{score_name:<16}{summary_overall['auroc']:>10.3f}{summary_overall['auprc']:>10.3f}"
              f"{summary_overall['precision']:>12.3f}{summary_overall['f1']:>10.3f}")

    report["summary_mean_auroc_overall"] = {score_name: summary_overall}
    suffix = out_suffix if out_suffix is not None else f"forecast_v2_federated_h10_encdec_synced_{baseline}"
    out_json = OUT_DIR / f"robo_fleet_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
               OUT_DIR / f"robo_fleet_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


def main(horizon_mult=10, use_forecast_prior=True, out_suffix=None, linkage="single", node_agg="topk",
         num_prototypes=None, metric="cosine", edge_quantile=None, delta=None):
    num_prototypes = NUM_PROTOTYPES if num_prototypes is None else num_prototypes
    delta = DELTA if delta is None else delta
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    forecast_h = STRIDE * horizon_mult

    print(f"loading robo3er, splitting into 5 real per-robot federated clients, "
          f"{FEATURE_GROUPS} as nodes, forecast head horizon_mult={horizon_mult} ...")
    data, targets, label_map, clients, cols, window_robot = build_clients()
    num_nodes = len(cols)
    node_idx = {name: i for i, name in enumerate(cols)}
    prior_edges = [(node_idx[s], node_idx[d]) for s, d in EDGES_NAMED]
    forecast_prior_edges = prior_edges if use_forecast_prior else None
    for c in clients:
        print(f"  client {c.client_id} ({c.robot_name}): fit={len(c.fit_idx)} calib={len(c.calib_idx)} "
              f"test_normal={len(c.test_normal_idx)} faults={ {label_map[str(k)]: len(v) for k, v in c.fault_idx_by_label.items()} }")

    scalers = [fit_client_scaler(data, c.fit_idx) for c in clients]
    models = [JointPrototypeV21Forecast(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=EMBED_DIM,
                                         num_prototypes=num_prototypes, prior_edges=prior_edges,
                                         forecast_h=forecast_h, top_k=TOP_K,
                                         forecast_prior_edges=forecast_prior_edges).to(DEVICE)
              for _ in clients]

    shared_init = {k: v.clone() for k, v in models[0].state_dict().items() if not k.startswith("memory.")}
    for model in models[1:]:
        model.load_state_dict(shared_init, strict=False)

    # Pre-build each client's fit-split pairs once (pairing doesn't depend on the model).
    fit_pairs = []
    for c in clients:
        vi, chains, _ = build_pairs(c.fit_idx, targets, window_robot, horizon_mult, True, True)
        x_in = data[vi]
        x_future = gather_future(data, chains, STRIDE)
        fit_pairs.append((vi, x_in, x_future))

    def local_train_step(c, model):
        idx = c.client_id
        vi, x_in_raw, x_future_raw = fit_pairs[idx]
        scaler = scalers[idx]
        n, t, f = x_in_raw.shape
        x_in = scaler.transform(x_in_raw.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
        n2, t2, f2 = x_future_raw.shape
        x_future = (scaler.transform(x_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                    if n2 > 0 else x_future_raw)
        train_local(model, x_in, x_future, LOCAL_EPOCHS, DEVICE, lr=LR, beta=BETA,
                    lambda_edge=LAMBDA_EDGE, lambda_forecast=LAMBDA_FORECAST, batch_size=BATCH_SIZE)
        return len(x_in)

    print(f"\nfederated training: {ROUNDS} rounds x {LOCAL_EPOCHS} local epochs, "
          f"memory exchange (gamma={GAMMA}, delta={delta})"
          + (", + FedAvg'd encoder/decoder" if SYNC_ENCODER_DECODER else " only"))
    diagnostics_log = run_federated_rounds(
        clients, models, local_train_step, ROUNDS, GAMMA, delta,
        sync_encoder_decoder=SYNC_ENCODER_DECODER, linkage=linkage,
        metric=metric, edge_quantile=edge_quantile)

    report = {"config": {"rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "num_prototypes": num_prototypes,
                          "embed_dim": EMBED_DIM, "top_k": TOP_K, "gamma": GAMMA, "delta": delta,
                          "metric": metric, "edge_quantile": edge_quantile,
                          "linkage": linkage,
                          "prior_edges": EDGES_NAMED,
                          "min_proto_samples": MIN_PROTO_SAMPLES, "top_k_agg": TOP_K_AGG,
                          "sync_encoder_decoder": SYNC_ENCODER_DECODER, "horizon_mult": horizon_mult,
                          "forecast_h": forecast_h, "use_forecast_prior": use_forecast_prior,
                          "feature_groups": FEATURE_GROUPS, "num_nodes": num_nodes, "node_agg": node_agg,
                          "baseline": "ours"},
              "alignment_log": diagnostics_log, "clients": {}}

    calib_data = {}
    for c, model, scaler in zip(clients, models, scalers):
        print(f"\ncalibration ({c.robot_name}): K/H reference stats ...")
        calib_w = scale_client(data, scaler, c.calib_idx)
        _, d_node_calib, _, _, idx_calib, _, _ = per_sample_scores(model, calib_w, DEVICE, batch_size=BATCH_SIZE)
        cov_min = max(MIN_PROTO_SAMPLES, num_nodes + 1)
        model.cov_head.set_calibration(d_node_calib, idx_calib, min_samples=cov_min)
        # d_mahal is 0 (identity-cov, zero-mean) before set_calibration -- re-run to get real values.
        _, d_node_calib, _, _, idx_calib, _, d_mahal_calib = per_sample_scores(model, calib_w, DEVICE, batch_size=BATCH_SIZE)
        d_mahal_med = float(np.median(d_mahal_calib))
        d_mahal_iqr = max(float(np.percentile(d_mahal_calib, 75) - np.percentile(d_mahal_calib, 25)), 1e-8)

        vi, chains, mask_c = build_pairs(c.calib_idx, targets, window_robot, horizon_mult, True, True)
        calib_in = scale_client(data, scaler, vi) if len(vi) else np.zeros((0, WINDOW_LEN, num_nodes), dtype=np.float32)
        calib_future_raw = gather_future(data, chains, STRIDE)
        if len(vi):
            n2, t2, f2 = calib_future_raw.shape
            calib_future = scaler.transform(calib_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
        else:
            calib_future = calib_future_raw
        k_resid_calib = forecast_scores(model, calib_in, calib_future, DEVICE, batch_size=BATCH_SIZE) if len(vi) else np.zeros((0, num_nodes), dtype=np.float32)
        idx_k_calib = idx_calib[mask_c]
        calib_data[c.client_id] = (d_node_calib, k_resid_calib, d_mahal_med, d_mahal_iqr, idx_calib, idx_k_calib)

    # Per-client calib-loss-quality confidence weights for the loss-weighted
    # fusion alternative to smooth_max -- see federated_train_eval.loss_weighted_combo's
    # docstring. B's own loss proxy = mean d_node (joint retrieval distance);
    # H's = mean d_mahal (a poorly-conditioned per-prototype covariance
    # inflates this even on calib's OWN data); K's = mean k_resid (forecast
    # error). All computed AFTER calibration (so d_mahal reflects the fitted
    # cov_head, not the pre-calibration identity fallback).
    b_loss = [float(calib_data[c.client_id][0].mean()) for c in clients]
    h_loss = []
    k_loss = []
    for c in clients:
        d_node_calib, k_resid_calib, d_mahal_med, d_mahal_iqr, idx_calib, idx_k_calib = calib_data[c.client_id]
        h_loss.append(d_mahal_med)  # median d_mahal on calib = H's own fit quality
        k_loss.append(float(k_resid_calib.mean()) if len(k_resid_calib) else float("nan"))
    lw_ready = all(not np.isnan(v) for v in k_loss)
    client_lw_weights = (confidence_weights_from_losses({"B": b_loss, "H": h_loss, "K": k_loss})
                          if lw_ready else None)
    if client_lw_weights is None:
        print("  [loss-weighted fusion] skipped: at least one client has no forecast-pairing calib windows")
    else:
        for c, w in zip(clients, client_lw_weights):
            print(f"  [loss-weighted fusion] {c.robot_name}: {w}")

    all_pairs = {}
    for c, model, scaler in zip(clients, models, scalers):
        d_node_calib, k_resid_calib, d_mahal_med, d_mahal_iqr, idx_calib, idx_k_calib = calib_data[c.client_id]
        report["clients"].setdefault(c.robot_name, {})

        if not c.fault_idx_by_label:
            print(f"client {c.client_id} ({c.robot_name}): no fault windows, skipping evaluation")
            continue

        test_w = scale_client(data, scaler, c.test_normal_idx)
        _, d_node_normal, _, _, idx_normal, _, d_mahal_normal = per_sample_scores(model, test_w, DEVICE, batch_size=BATCH_SIZE)
        # B is now DEFINED as the per-prototype ("by operating regime") z-score,
        # not the global-calib version -- see memory/robo-fleet-num-prototypes-sweep.md.
        node_score_normal = (gdn_score_proto(d_node_calib, idx_calib, d_node_normal, idx_normal, num_prototypes, MIN_PROTO_SAMPLES)
                              if node_agg == "max" else
                              two_stage_group_score_proto(d_node_calib, idx_calib, d_node_normal, idx_normal,
                                                           TOP_K_AGG, num_prototypes, MIN_PROTO_SAMPLES))
        h_score_normal = (d_mahal_normal - d_mahal_med) / d_mahal_iqr
        base_normal = add_h_score(base_scores(node_score_normal), h_score_normal)

        vi_n, chains_n, mask_n = build_pairs(c.test_normal_idx, targets, window_robot, horizon_mult, True, True)
        test_normal_in = scale_client(data, scaler, vi_n) if len(vi_n) else np.zeros((0, WINDOW_LEN, num_nodes), dtype=np.float32)
        future_n_raw = gather_future(data, chains_n, STRIDE)
        if len(vi_n):
            n2, t2, f2 = future_n_raw.shape
            test_normal_future = scaler.transform(future_n_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
            k_resid_normal = forecast_scores(model, test_normal_in, test_normal_future, DEVICE, batch_size=BATCH_SIZE)
            # K is also now the per-prototype version.
            forecast_score_normal = (gdn_score_proto(k_resid_calib, idx_k_calib, k_resid_normal, idx_normal[mask_n],
                                                       num_prototypes, MIN_PROTO_SAMPLES)
                                      if len(k_resid_calib) else np.zeros(len(vi_n)))
        else:
            forecast_score_normal = np.zeros(0)
        scores_normal = add_forecast_scores(base_normal, forecast_score_normal, mask_n)
        if client_lw_weights is not None and len(forecast_score_normal):
            b_lw_n, h_lw_n = base_normal["B_node_max"][mask_n], base_normal["H_cov_mahal"][mask_n]
            scores_normal["BHK_lw"] = loss_weighted_combo(
                {"B": b_lw_n, "H": h_lw_n, "K": forecast_score_normal}, client_lw_weights[c.client_id])

        client_report = {"robot_name": c.robot_name, "fault_types": {}}
        rows = {k: [] for k in scores_normal}
        for key in scores_normal:
            all_pairs.setdefault(key, [])
        for label_id, fault_idx in c.fault_idx_by_label.items():
            name = label_map[str(label_id)]
            fault_w = scale_client(data, scaler, fault_idx)
            _, d_node_f, _, _, idx_f, _, d_mahal_f = per_sample_scores(model, fault_w, DEVICE, batch_size=BATCH_SIZE)
            node_score_f = (gdn_score_proto(d_node_calib, idx_calib, d_node_f, idx_f, num_prototypes, MIN_PROTO_SAMPLES)
                             if node_agg == "max" else
                             two_stage_group_score_proto(d_node_calib, idx_calib, d_node_f, idx_f,
                                                          TOP_K_AGG, num_prototypes, MIN_PROTO_SAMPLES))
            h_score_f = (d_mahal_f - d_mahal_med) / d_mahal_iqr
            base_f = add_h_score(base_scores(node_score_f), h_score_f)

            vi_f, chains_f, mask_f = build_pairs(fault_idx, targets, window_robot, horizon_mult, False, False)
            if len(vi_f):
                fault_in = scale_client(data, scaler, vi_f)
                future_f_raw = gather_future(data, chains_f, STRIDE)
                n2, t2, f2 = future_f_raw.shape
                fault_future = scaler.transform(future_f_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                k_resid_f = forecast_scores(model, fault_in, fault_future, DEVICE, batch_size=BATCH_SIZE)
                forecast_score_f = (gdn_score_proto(k_resid_calib, idx_k_calib, k_resid_f, idx_f[mask_f],
                                                      num_prototypes, MIN_PROTO_SAMPLES)
                                     if len(k_resid_calib) else np.zeros(len(vi_f)))
            else:
                forecast_score_f = np.zeros(0)
            scores_fault = add_forecast_scores(base_f, forecast_score_f, mask_f)
            if client_lw_weights is not None and len(forecast_score_f):
                b_lw_f, h_lw_f = base_f["B_node_max"][mask_f], base_f["H_cov_mahal"][mask_f]
                scores_fault["BHK_lw"] = loss_weighted_combo(
                    {"B": b_lw_f, "H": h_lw_f, "K": forecast_score_f}, client_lw_weights[c.client_id])

            metrics = {}
            for key in scores_normal:
                m = detection_metrics(scores_normal[key], scores_fault.get(key, np.zeros(0)))
                if m is None:
                    continue
                metrics[key] = m
                rows[key].append(m)
                all_pairs[key].append(m)
            client_report["fault_types"][name] = {"n": int(len(fault_idx)), **metrics}
            print(f"  {c.robot_name:<10}{name:<16}n={len(fault_idx):<5}"
                  + "  ".join(f"{k}=auroc:{v['auroc']:.3f}" for k, v in metrics.items()))

        client_report["client_mean_metrics"] = {
            k: {m: float(np.mean([row[m] for row in v])) for m in ("auroc", "auprc", "precision", "f1")}
            for k, v in rows.items() if v
        }
        report["clients"][c.robot_name] = client_report

    summary_overall = {
        k: {m: float(np.mean([row[m] for row in v])) for m in ("auroc", "auprc", "precision", "f1")}
        for k, v in all_pairs.items() if v
    }
    print(f"\n{'method':<16}{'AUROC':>10}{'AUPRC':>10}{'Precision':>12}{'F1':>10}")
    for key, m in summary_overall.items():
        print(f"{key:<16}{m['auroc']:>10.3f}{m['auprc']:>10.3f}{m['precision']:>12.3f}{m['f1']:>10.3f}")

    report["summary_mean_auroc_overall"] = summary_overall
    suffix = out_suffix if out_suffix is not None else (
        f"forecast_v2_federated_h{horizon_mult}{'_noprior' if not use_forecast_prior else ''}"
        + ("_encdec_synced" if SYNC_ENCODER_DECODER else "")
        + ("_bmax" if node_agg == "max" else "")
        + (f"_metric_{metric}" if metric != "cosine" else ""))
    out_json = OUT_DIR / f"robo_fleet_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
               OUT_DIR / f"robo_fleet_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-mult", type=int, default=10)
    parser.add_argument("--no-forecast-prior", action="store_true")
    parser.add_argument("--out-suffix", type=str, default=None)
    parser.add_argument("--linkage", type=str, default="single", choices=["single", "complete"])
    parser.add_argument("--node-agg", type=str, default="topk", choices=["topk", "max"],
                         help="B (node_score) aggregation: 'topk' (default, two-stage top-k-mean+"
                              "re-zscore) or 'max' (single-stage max Z-score, matches K's gdn_score)")
    parser.add_argument("--num-prototypes", type=int, default=None,
                         help="override NUM_PROTOTYPES (default module constant, 2)")
    parser.add_argument("--baseline", type=str, default="ours",
                         choices=["ours", "fedavg", "ifcaae", "fedexdnn", "fedpro", "fedcpg", "ufedhy"],
                         help="server-aggregation strategy -- see run_federated_rounds' docstring "
                              "('fedpro'/'fedcpg' are robo_fleet-only, supervised -- see fedpro.py/"
                              "fedcpg.py; 'ufedhy' is fit-on-healthy like fedavg/ifcaae/fedexdnn -- "
                              "see ufedhy.py)")
    parser.add_argument("--num-clusters", type=int, default=2,
                         help="IFCAAE only: number of global model clusters")
    parser.add_argument("--delta", type=float, default=None,
                         help="override DELTA (default module constant, 0.5) -- cross-client "
                              "cosine-similarity edge threshold for align_and_split")
    args = parser.parse_args()
    if args.baseline == "ours":
        main(horizon_mult=args.horizon_mult, use_forecast_prior=not args.no_forecast_prior,
             out_suffix=args.out_suffix, linkage=args.linkage, node_agg=args.node_agg,
             num_prototypes=args.num_prototypes, delta=args.delta)
    elif args.baseline == "fedpro":
        main_fedpro_baseline(out_suffix=args.out_suffix)
    elif args.baseline == "fedcpg":
        main_fedcpg_baseline(out_suffix=args.out_suffix)
    elif args.baseline == "ufedhy":
        main_ufedhy_baseline(out_suffix=args.out_suffix)
    else:
        main_faithful_baseline(args.baseline, out_suffix=args.out_suffix, num_clusters=args.num_clusters,
                                num_prototypes=args.num_prototypes or NUM_PROTOTYPES)
