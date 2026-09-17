#!/usr/bin/env python3
"""
FEDERATED JointPrototypeV21Forecast (V2.1 + ForecastHead, signal K) on
Paderborn, treating K001-K006 as 6 federated clients. Per
`memory/benchmark-policy-federated-only.md`: the federated variant of
`run_paderborn_bck.py`'s per-file cross-window pairing route
(same pairing rules -- see that script's docstring), with a federated
protocol (`JointPrototypeMemory` codebook exchanged via `align_and_split`,
FedAvg'd encoder/decoder AND `forecast_head` via `SYNC_ENCODER_DECODER`).

PROJECT MAINLINE (finalized 2026-08-30, BHK computed since always despite
an earlier docstring's stale "BK-only" claim -- see `federated_train_eval.py`'s
module docstring; **2026-09-12: switched from `JointPrototypeV31Forecast`
to `V21Forecast`, dropping the declared-physics-edge `typed_head`/signal-E
machinery project-wide** -- a controlled ablation on robo_fleet found it
worth only ~0.005 BHK_max, and the same simpler no-declared-edges
architecture ALFA/SMD already used is now the one mainline everywhere --
see `memory/v21-mainline-switch.md`): this script reports
`B_node_max`/`H_cov_mahal`/`BH_max`/`K_forecast_max`/`BK_max`/`BHK_max`.
`edge_head` (generic cross-node attention residual, signal C) and
`cov_head` are still trained/scored; `cov_head` also remains available on
the saved model for `diagnose_paderborn_localization_federated.py`
(node-level fault localization, a separate task from this script's
detection AUROC) -- that diagnose script still constructs its own
`V31Forecast` model independently and has NOT been switched over yet, a
documented follow-up gap, not an inconsistency introduced silently.

**WINDOW-level scoring**: uses the `two_stage_group_score` (top-k-mean +
re-zscore) aggregation by default for B (`--node-agg topk`), or a
single-stage max Z-score (`--node-agg max`, matching K's own `gdn_score`)
-- see `two_stage_group_score`/`gdn_score` in `src/federated/federated_train_eval.py`.
All z-score reference stats use the per-client held-out `calib` split
(global median/IQR, no per-prototype conditioning) -- see
`memory/calib-in-prototype-ab.md` for the calibration-architecture history;
`per_prototype`/`ema`/`shrinkage` were all tried and dropped.

The training+eval mainline (round loop, scoring primitives, `train_local`)
lives in `src/federated/federated_train_eval.py`, shared verbatim with
`run_robo_fleet_bck_federated.py`/`run_me_ad_bck_federated.py` (all
`JointPrototypeV21Forecast` as of 2026-09-12) -- this script only
supplies Paderborn-specific data loading (per-file windowing/chain-
building, since each bearing recording is a variable-length file rather
than a pre-fixed window array) and CLI plumbing.

Usage:
    python3 run_paderborn_bck_federated.py [--horizon-mult M] [--no-forecast-prior]
        [--out-suffix NAME]
"""
import argparse
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
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "FedPRO"))
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "FedCPG"))
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "uFedHy-DisMTSADD"))
sys.path.insert(0, str(REPO_ROOT / "src" / "dataloaders" / "paderborn"))
from joint_prototype_model import JointPrototypeV21Forecast  # noqa: E402
from baseline_models import (ReconOnlyModel, ExemplarOnlyModel, FedPROModel,  # noqa: E402
                              FedCPGModel, SCNorTransformerModel, Hypernetwork)
from federated_train_eval import (two_stage_group_score, base_scores,  # noqa: E402
                           add_forecast_scores, add_h_score, per_sample_scores, forecast_scores,
                           train_local, run_federated_rounds, gdn_score, detection_metrics,
                           two_stage_group_score_proto, gdn_score_proto, smooth_max,
                           loss_weighted_combo, train_local_recon, train_local_exemplar,
                           per_node_recon_error, per_node_exemplar_deviation)
from federated_memory import confidence_weights_from_losses  # noqa: E402
from comm_cost import round_comm_record  # noqa: E402
from fedpro import (train_local_classifier, embed_all, build_initial_prototypes_kmeans,  # noqa: E402
                     refine_prototypes_margin, retrieve_and_vote, confidence_ensemble)
from fedcpg import train_local_fedcpg, aggregate_global_prototypes, predict_anomaly_score  # noqa: E402
from ufedhy import hypernet_round_update  # noqa: E402

SYNC_ENCODER_DECODER = True
from paderborn_adapter import (  # noqa: E402
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
# EDGE_TYPES (per-edge "proportional"/"nonlinear" typing) removed 2026-09-12
# along with the switch to JointPrototypeV21Forecast -- see that class's
# docstring and memory/v21-mainline-switch.md. EDGES_NAMED/PRIOR_EDGES are
# still used as `edge_head`'s soft, LEARNED-STRENGTH attention bias.
PRIOR_EDGES = [(NODE_IDX[s], NODE_IDX[d]) for s, d in EDGES_NAMED]

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15
WINDOW_LEN = 64
WINDOW_STRIDE = 32
BATCH_SIZE = 256
ROUNDS = 5
LOCAL_EPOCHS = 12
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 16
TOP_K = 5
TOP_K_AGG = 3  # see two_stage_group_score in src/federated/federated_train_eval.py
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_FORECAST = 0.5
MIN_PROTO_SAMPLES = 5
GAMMA = 0.5
DELTA = 0.5
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
    return windows.reshape(-1, window_len, f), file_id, len(positions)


def build_chains(n_files, n_positions, horizon_mult):
    vi, chains = [], []
    for file_idx in range(n_files):
        base = file_idx * n_positions
        for local_pos in range(n_positions - horizon_mult):
            gi = base + local_pos
            vi.append(gi)
            chains.append([gi + k for k in range(1, horizon_mult + 1)])
    return np.array(vi, dtype=int), chains


def gather_future(windows, chains, stride):
    if len(chains) == 0:
        return np.zeros((0, 0, windows.shape[-1]), dtype=windows.dtype)
    segments = [windows[[c[k] for c in chains]][:, -stride:, :] for k in range(len(chains[0]))]
    return np.concatenate(segments, axis=1)


def main_fedpro_baseline(out_suffix=None, k_per_class=3, top_k=3, tau=0.1, embed_dim=EMBED_DIM,
                          fedavg_rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, proto_epochs=50, proto_lr=1e-3):
    """FedPRO (Zhou et al., IEEE TC 2026) on Paderborn -- a deliberate
    exception to fit-on-healthy: supervised 4-class classification
    (healthy + outer_ring/inner_ring/combined damage categories). Unlike
    robo_fleet/ALFA, Paderborn's federation clients (K001-K006 healthy
    bearings) have NO fault data of their own by construction -- damaged
    bearings are separate FILES, not client-owned occurrences. This
    baseline round-robin partitions `ALL_DAMAGED_CODES`'s 26 damaged-
    bearing files across the 6 healthy-bearing clients (client i gets
    `ALL_DAMAGED_CODES[i::6]`) so each client has a supervised mix of
    categories to discriminate -- a documented, non-standard
    reinterpretation of "client" needed ONLY for this supervised
    baseline (every other baseline/`ours` keeps Paderborn's original
    fit-on-healthy client definition unchanged). See
    `benchmark/FedPRO/fedpro.py`'s module docstring for the method itself."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    num_nodes = len(NODE_NAMES)
    category_labels = {"outer_ring": 1, "inner_ring": 2, "combined": 3}
    label_names = {0: "healthy", 1: "outer_ring", 2: "inner_ring", 3: "combined"}
    num_classes = 1 + len(category_labels)

    print("loading Paderborn, splitting into 6 healthy-bearing clients, baseline=fedpro "
          "(supervised, healthy+damage-category labels, NOT fit-on-healthy) ...")
    damaged_groups = [ALL_DAMAGED_CODES[i::len(HEALTHY_CODES)] for i in range(len(HEALTHY_CODES))]

    client_data = []
    for ci, code in enumerate(HEALTHY_CODES):
        d = load_bearing_with_physics(code)
        arr = stack_nodes(d)
        scaler = StandardScaler().fit(arr.reshape(-1, num_nodes))

        def scale(a, scaler=scaler):
            n_, t_, f_ = a.shape
            return scaler.transform(a.reshape(-1, f_)).reshape(n_, t_, f_).astype(np.float32)

        windows_h, _, _ = make_windows(scale(arr))
        x_list, y_list = [windows_h], [np.zeros(len(windows_h), dtype=np.int64)]
        for dcode in damaged_groups[ci]:
            dd = load_bearing_with_physics(dcode)
            darr = stack_nodes(dd)
            dwindows, _, _ = make_windows(scale(darr))
            label = category_labels[category_of(dcode)]
            x_list.append(dwindows)
            y_list.append(np.full(len(dwindows), label, dtype=np.int64))
        x_all = np.concatenate(x_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)

        perm = rng.permutation(len(x_all))
        n_train = max(1, int(0.7 * len(x_all)))
        train_idx, test_idx = perm[:n_train], perm[n_train:]
        client_data.append({"code": code, "x_train": x_all[train_idx], "y_train": y_all[train_idx],
                              "x_test": x_all[test_idx], "y_test": y_all[test_idx],
                              "damaged_codes": damaged_groups[ci]})
        print(f"  {code}: train={len(train_idx)} test={len(test_idx)} "
              f"damaged_codes={damaged_groups[ci]} classes={sorted(set(y_all.tolist()))}")

    models = [FedPROModel(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=embed_dim,
                           num_classes=num_classes).to(DEVICE) for _ in client_data]

    def local_train_step(c, model):
        train_local_classifier(model, c["x_train"], c["y_train"], local_epochs, DEVICE, lr=LR, batch_size=BATCH_SIZE)
        return len(c["x_train"])

    print(f"\nfederated training (off-the-shelf FedAvg classifier): {fedavg_rounds} rounds x "
          f"{local_epochs} local epochs, baseline=fedpro")
    run_federated_rounds(client_data, models, local_train_step, fedavg_rounds, GAMMA, DELTA, mode="fedavg")

    print("\nbuilding + refining per-client prototypes (encoder frozen) ...")
    all_protos, all_proto_labels = [], []
    for c, model in zip(client_data, models):
        for p in model.parameters():
            p.requires_grad_(False)
        z_train = embed_all(model, c["x_train"], DEVICE)
        protos, proto_labels = build_initial_prototypes_kmeans(z_train, c["y_train"], num_classes, k_per_class)
        protos = refine_prototypes_margin(protos, proto_labels, z_train, c["y_train"],
                                           epochs=proto_epochs, lr=proto_lr, tau=tau)
        all_protos.append(protos)
        all_proto_labels.append(proto_labels)
        print(f"  {c['code']}: {len(protos)} prototypes across {len(set(proto_labels.tolist()))} classes")
    bank_protos = np.concatenate(all_protos, axis=0)
    bank_labels = np.concatenate(all_proto_labels, axis=0)
    print(f"  global prototype bank: {len(bank_protos)} prototypes from {len(client_data)} clients")

    print("\nprototype retrieval-augmented inference on each client's held-out test split ...")
    report = {"config": {"baseline": "fedpro", "fedavg_rounds": fedavg_rounds, "local_epochs": local_epochs,
                          "embed_dim": embed_dim, "num_classes": num_classes, "k_per_class": k_per_class,
                          "top_k": top_k, "tau": tau, "proto_epochs": proto_epochs,
                          "split": "stratified_70_30_all_labels_NOT_fit_on_healthy",
                          "damaged_code_partition": damaged_groups,
                          "note": "FedPRO is a supervised multi-class classifier wrapper (Zhou et al., "
                                  "IEEE TC 2026) on healthy+damage-CATEGORY labels -- Paderborn's clients "
                                  "have no fault data of their own by construction, so damaged bearings "
                                  "were round-robin partitioned across clients, see this function's "
                                  "docstring. See benchmark/FedPRO/fedpro.py"},
              "clients": {}}

    all_pairs = []
    for c, model in zip(client_data, models):
        x_test, y_test = c["x_test"], c["y_test"]
        z_test = embed_all(model, x_test, DEVICE)
        with torch.no_grad():
            model.eval()
            logits = model(torch.from_numpy(x_test).to(DEVICE))
            y_wi = torch.softmax(logits, dim=1).cpu().numpy()
        y_p, alpha = retrieve_and_vote(z_test, bank_protos, bank_labels, num_classes, top_k=top_k)
        y_en = confidence_ensemble(y_p, y_wi, alpha)
        pred = np.argmax(y_en, axis=1)
        accuracy = float((pred == y_test).mean())
        anomaly_score = 1.0 - y_en[:, 0]

        client_report = {"code": c["code"], "accuracy": accuracy, "categories": {}}
        normal_mask = y_test == 0
        scores_normal = anomaly_score[normal_mask]
        for label_id in sorted(set(y_test.tolist()) - {0}):
            fault_mask = y_test == label_id
            scores_fault = anomaly_score[fault_mask]
            m = detection_metrics(scores_normal, scores_fault)
            if m is None:
                continue
            name = label_names[label_id]
            client_report["categories"][name] = {"n": int(fault_mask.sum()), "fedpro_score": m}
            all_pairs.append((name, m))
            print(f"  {c['code']:<6}{name:<12}n={int(fault_mask.sum()):<5}"
                  f"accuracy={accuracy:.3f}  fedpro_score=auroc:{m['auroc']:.3f}")
        report["clients"][c["code"]] = client_report

    mean_accuracy = float(np.mean([v["accuracy"] for v in report["clients"].values()]))
    summary_overall = {
        metric: float(np.mean([row[metric] for _, row in all_pairs])) for metric in ("auroc", "auprc", "precision", "f1")
    } if all_pairs else {}
    category_summary = {}
    for name in category_labels:
        subset = [row["auroc"] for cat, row in all_pairs if cat == name]
        category_summary[name] = float(np.mean(subset)) if subset else None
    print(f"\nmean classification accuracy: {mean_accuracy:.3f}")
    print(f"{'fedpro_score':<16}{'AUROC':>10}{'AUPRC':>10}{'Precision':>12}{'F1':>10}")
    if summary_overall:
        print(f"{'fedpro_score':<16}{summary_overall['auroc']:>10.3f}{summary_overall['auprc']:>10.3f}"
              f"{summary_overall['precision']:>12.3f}{summary_overall['f1']:>10.3f}")

    report["mean_accuracy"] = mean_accuracy
    report["summary_mean_auroc_overall"] = {"fedpro_score": summary_overall}
    report["category_summary"] = {"fedpro_score": category_summary}
    suffix = out_suffix if out_suffix is not None else "forecast_v2_federated_h10_encdec_synced_fedpro"
    out_json = OUT_DIR / f"paderborn_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "bank_protos": bank_protos,
                "bank_labels": bank_labels, "report": report}, OUT_DIR / f"paderborn_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


def main_fedcpg_baseline(out_suffix=None, embed_dim=EMBED_DIM, proj_dim=32, fedcpg_rounds=ROUNDS,
                          local_epochs=LOCAL_EPOCHS, alpha=0.5, beta=0.2, tau=0.1):
    """FedCPG (Li et al., *Computers in Industry* 2025) on Paderborn --
    same round-robin damaged-bearing repartition FedPRO already
    established (`main_fedpro_baseline`'s docstring) since Paderborn's
    healthy-bearing clients have no fault data of their own by
    construction. See `benchmark/FedCPG/fedcpg.py` for the algorithm and
    `docs/1-s2.0-S0166361524001088-main.pdf` for the paper."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    num_nodes = len(NODE_NAMES)
    category_labels = {"outer_ring": 1, "inner_ring": 2, "combined": 3}
    label_names = {0: "healthy", 1: "outer_ring", 2: "inner_ring", 3: "combined"}
    num_classes = 1 + len(category_labels)

    print("loading Paderborn, splitting into 6 healthy-bearing clients, baseline=fedcpg "
          "(supervised, healthy+damage-category labels, NOT fit-on-healthy) ...")
    damaged_groups = [ALL_DAMAGED_CODES[i::len(HEALTHY_CODES)] for i in range(len(HEALTHY_CODES))]

    client_data = []
    for ci, code in enumerate(HEALTHY_CODES):
        d = load_bearing_with_physics(code)
        arr = stack_nodes(d)
        scaler = StandardScaler().fit(arr.reshape(-1, num_nodes))

        def scale(a, scaler=scaler):
            n_, t_, f_ = a.shape
            return scaler.transform(a.reshape(-1, f_)).reshape(n_, t_, f_).astype(np.float32)

        windows_h, _, _ = make_windows(scale(arr))
        x_list, y_list = [windows_h], [np.zeros(len(windows_h), dtype=np.int64)]
        for dcode in damaged_groups[ci]:
            dd = load_bearing_with_physics(dcode)
            darr = stack_nodes(dd)
            dwindows, _, _ = make_windows(scale(darr))
            label = category_labels[category_of(dcode)]
            x_list.append(dwindows)
            y_list.append(np.full(len(dwindows), label, dtype=np.int64))
        x_all = np.concatenate(x_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)

        perm = rng.permutation(len(x_all))
        n_train = max(1, int(0.7 * len(x_all)))
        train_idx, test_idx = perm[:n_train], perm[n_train:]
        client_data.append({"code": code, "x_train": x_all[train_idx], "y_train": y_all[train_idx],
                              "x_test": x_all[test_idx], "y_test": y_all[test_idx]})
        print(f"  {code}: train={len(train_idx)} test={len(test_idx)} "
              f"damaged_codes={damaged_groups[ci]} classes={sorted(set(y_all.tolist()))}")

    models = [FedCPGModel(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=embed_dim,
                           proj_dim=proj_dim, num_classes=num_classes).to(DEVICE) for _ in client_data]

    print(f"\nfederated training (backbone FedAvg + personalized head + class-prototype contrast): "
          f"{fedcpg_rounds} rounds x {local_epochs} local epochs, baseline=fedcpg")
    global_protos = torch.zeros(num_classes, proj_dim)
    for rnd in range(1, fedcpg_rounds + 1):
        client_protos, client_counts, fit_counts = [], [], []
        for c, model in zip(client_data, models):
            local_protos, local_counts = train_local_fedcpg(
                model, c["x_train"], c["y_train"], local_epochs, DEVICE, global_protos, num_classes,
                lr=LR, batch_size=BATCH_SIZE, alpha=alpha, beta=beta, tau=tau)
            client_protos.append(local_protos); client_counts.append(local_counts)
            fit_counts.append(len(c["x_train"]))
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
                          "damaged_code_partition": damaged_groups,
                          "note": "FedCPG (Li et al., Computers in Industry 2025) on healthy+damage-"
                                  "CATEGORY labels -- see main_fedpro_baseline's docstring for the "
                                  "round-robin damaged-code partition, benchmark/FedCPG/fedcpg.py for "
                                  "the method"},
              "clients": {}}

    all_pairs = []
    for c, model in zip(client_data, models):
        x_test, y_test = c["x_test"], c["y_test"]
        with torch.no_grad():
            model.eval()
            logits = model(torch.from_numpy(x_test).to(DEVICE))
            pred = logits.argmax(dim=1).cpu().numpy()
        accuracy = float((pred == y_test).mean())
        anomaly_score = predict_anomaly_score(model, x_test, DEVICE, batch_size=BATCH_SIZE)

        client_report = {"code": c["code"], "accuracy": accuracy, "categories": {}}
        normal_mask = y_test == 0
        scores_normal = anomaly_score[normal_mask]
        for label_id in sorted(set(y_test.tolist()) - {0}):
            fault_mask = y_test == label_id
            scores_fault = anomaly_score[fault_mask]
            m = detection_metrics(scores_normal, scores_fault)
            if m is None:
                continue
            name = label_names[label_id]
            client_report["categories"][name] = {"n": int(fault_mask.sum()), "fedcpg_score": m}
            all_pairs.append((name, m))
            print(f"  {c['code']:<6}{name:<12}n={int(fault_mask.sum()):<5}"
                  f"accuracy={accuracy:.3f}  fedcpg_score=auroc:{m['auroc']:.3f}")
        report["clients"][c["code"]] = client_report

    mean_accuracy = float(np.mean([v["accuracy"] for v in report["clients"].values()]))
    summary_overall = {
        metric: float(np.mean([row[metric] for _, row in all_pairs])) for metric in ("auroc", "auprc", "precision", "f1")
    } if all_pairs else {}
    category_summary = {}
    for name in category_labels:
        subset = [row["auroc"] for cat, row in all_pairs if cat == name]
        category_summary[name] = float(np.mean(subset)) if subset else None
    print(f"\nmean classification accuracy: {mean_accuracy:.3f}")
    print(f"{'fedcpg_score':<16}{'AUROC':>10}{'AUPRC':>10}{'Precision':>12}{'F1':>10}")
    if summary_overall:
        print(f"{'fedcpg_score':<16}{summary_overall['auroc']:>10.3f}{summary_overall['auprc']:>10.3f}"
              f"{summary_overall['precision']:>12.3f}{summary_overall['f1']:>10.3f}")

    report["mean_accuracy"] = mean_accuracy
    report["summary_mean_auroc_overall"] = {"fedcpg_score": summary_overall}
    report["category_summary"] = {"fedcpg_score": category_summary}
    suffix = out_suffix if out_suffix is not None else "fedcpg"
    out_json = OUT_DIR / f"paderborn_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
               OUT_DIR / f"paderborn_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


def main_ufedhy_baseline(out_suffix=None, hyper_embed_dim=32, hyper_hidden_dim=64,
                          transformer_embed_dim=16, ufedhy_rounds=ROUNDS, local_epochs=LOCAL_EPOCHS,
                          hyper_lr=1e-3):
    """uFedHy-DisMTSADD (Hao et al., *Information Processing & Management*
    2025) on Paderborn -- fit-on-healthy, runs like FedAvg/IFCAAE/
    Fed-ExDNN. See `benchmark/uFedHy-DisMTSADD/ufedhy.py` for the hypernetwork update
    rule and `docs/1-s2.0-S0306457325000494-main.pdf` for the paper."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    num_nodes = len(NODE_NAMES)

    print("loading healthy bearings (K001-K006) as 6 federated clients, baseline=ufedhy ...")
    clients = []
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

    template = SCNorTransformerModel(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=transformer_embed_dim)
    target_shapes = {name: p.shape for name, p in template.state_dict().items()}
    hypernet = Hypernetwork(num_clients=len(clients), target_shapes=target_shapes,
                             embed_dim=hyper_embed_dim, hidden_dim=hyper_hidden_dim).to(DEVICE)
    hyper_optimizer = torch.optim.Adam(hypernet.parameters(), lr=hyper_lr)
    models = [SCNorTransformerModel(num_nodes=num_nodes, window_size=WINDOW_LEN,
                                     embed_dim=transformer_embed_dim).to(DEVICE) for _ in clients]

    def get_fit_windows(c):
        fit_windows, _, _ = make_windows(c["fit"])
        return fit_windows

    print(f"\nfederated training (hypernetwork-generated weights + local SGD + MSE distillation update): "
          f"{ufedhy_rounds} rounds x {local_epochs} local epochs, baseline=ufedhy")
    diagnostics_log = []
    for rnd in range(1, ufedhy_rounds + 1):
        hyper_losses = []
        bytes_up = bytes_down = 0
        for client_id, (c, model) in enumerate(zip(clients, models)):
            x_in = get_fit_windows(c)
            hyper_loss, n, bd, bu = hypernet_round_update(hypernet, hyper_optimizer, client_id, model, x_in,
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
                          "clients": HEALTHY_CODES, "score_name": "recon_score",
                          "note": "localization uses this project's own per-node argmax convention, NOT "
                                  "the paper's own PC-algorithm+PageRank causal diagnosis -- see "
                                  "benchmark/uFedHy-DisMTSADD/ufedhy.py's module docstring"},
              "nodes": NODE_NAMES, "alignment_log": diagnostics_log, "bearings": {}}

    calib_raw_per_client, normal_scores_per_client = [], []
    for c, model in zip(clients, models):
        calib_windows, _, _ = make_windows(c["calib"])
        calib_raw = per_node_recon_error(model, calib_windows, DEVICE)
        calib_raw_per_client.append(calib_raw)
        test_normal_windows, _, _ = make_windows(c["test_normal"])
        test_raw = per_node_recon_error(model, test_normal_windows, DEVICE)
        normal_scores_per_client.append(gdn_score(calib_raw, test_raw))

    per_bearing_rows = []
    for code in ALL_DAMAGED_CODES:
        d = load_bearing_with_physics(code)
        arr = stack_nodes(d)
        cat, origin = category_of(code), damage_origin_of(code)

        per_client_metrics = []
        for c, model, calib_raw, scores_normal in zip(clients, models, calib_raw_per_client, normal_scores_per_client):
            arr_scaled = c["scaler"].transform(arr.reshape(-1, num_nodes)).reshape(arr.shape).astype(np.float32)
            fault_windows, _, _ = make_windows(arr_scaled)
            fault_raw = per_node_recon_error(model, fault_windows, DEVICE)
            scores_fault = gdn_score(calib_raw, fault_raw)
            m = detection_metrics(scores_normal, scores_fault)
            if m is not None:
                per_client_metrics.append(m)

        if not per_client_metrics:
            continue
        mean_across_clients = {
            metric: float(np.mean([row[metric] for row in per_client_metrics]))
            for metric in ("auroc", "auprc", "precision", "f1")
        }
        per_bearing_rows.append(mean_across_clients)
        report["bearings"][code] = {"category": cat, "origin": origin, "n": len(arr),
                                      "recon_score": mean_across_clients}
        print(f"  {code:<6}{cat:<12}{origin:<11}recon_score=auroc:{mean_across_clients['auroc']:.3f}")

    summary = {
        metric: float(np.mean([row[metric] for row in per_bearing_rows]))
        for metric in ("auroc", "auprc", "precision", "f1")
    } if per_bearing_rows else {}
    print(f"\n{'recon_score':<24}{'AUROC':>10}{'AUPRC':>10}{'Precision':>12}{'F1':>10}")
    if summary:
        print(f"{'recon_score':<24}{summary['auroc']:>10.3f}{summary['auprc']:>10.3f}"
              f"{summary['precision']:>12.3f}{summary['f1']:>10.3f}")

    category_summary = {}
    for cat in ["outer_ring", "inner_ring", "combined"]:
        subset = [report["bearings"][code]["recon_score"]["auroc"] for code in ALL_DAMAGED_CODES
                  if code in report["bearings"] and category_of(code) == cat]
        category_summary[cat] = float(np.mean(subset)) if subset else None

    report["summary_mean_auroc_overall"] = {"recon_score": summary}
    report["category_summary"] = {"recon_score": category_summary}
    suffix = out_suffix if out_suffix is not None else "ufedhy"
    out_json = OUT_DIR / f"paderborn_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models],
                "hypernet_state_dict": hypernet.state_dict(), "report": report},
               OUT_DIR / f"paderborn_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


def main_faithful_baseline(baseline, out_suffix=None, num_clusters=2, num_prototypes=NUM_PROTOTYPES):
    """FedAvg / IFCAAE / Fed-ExDNN, each with its OWN minimal architecture
    and own single anomaly score (no B/H/K/BK/BHK) -- see
    `src/models/baseline_models.py`'s module docstring. No forecast
    chains needed (none of these three baselines forecast): plain
    per-file windows only."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    num_nodes = len(NODE_NAMES)

    print(f"loading healthy bearings (K001-K006) as 6 federated clients, baseline={baseline} ...")
    clients = []
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

    def make_model():
        if baseline == "fedexdnn":
            return ExemplarOnlyModel(num_nodes=num_nodes, window_size=WINDOW_LEN,
                                      embed_dim=EMBED_DIM, num_prototypes=num_prototypes).to(DEVICE)
        return ReconOnlyModel(num_nodes=num_nodes, window_size=WINDOW_LEN).to(DEVICE)

    models = [make_model() for _ in clients]

    def get_fit_windows(c):
        fit_windows, _, _ = make_windows(c["fit"])
        return fit_windows

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
                          "clients": HEALTHY_CODES, "score_name": score_name},
              "nodes": NODE_NAMES, "alignment_log": diagnostics_log, "bearings": {}}

    normal_scores_per_client, calib_raw_per_client = [], []
    for c, model in zip(clients, models):
        calib_windows, _, _ = make_windows(c["calib"])
        calib_raw = score_fn(model, calib_windows, DEVICE)
        calib_raw_per_client.append(calib_raw)
        test_normal_windows, _, _ = make_windows(c["test_normal"])
        test_raw = score_fn(model, test_normal_windows, DEVICE)
        normal_scores_per_client.append(gdn_score(calib_raw, test_raw))

    per_bearing_rows = []
    for code in ALL_DAMAGED_CODES:
        d = load_bearing_with_physics(code)
        arr = stack_nodes(d)
        cat, origin = category_of(code), damage_origin_of(code)

        per_client_metrics = []
        for c, model, calib_raw, scores_normal in zip(clients, models, calib_raw_per_client, normal_scores_per_client):
            arr_scaled = c["scaler"].transform(arr.reshape(-1, num_nodes)).reshape(arr.shape).astype(np.float32)
            fault_windows, _, _ = make_windows(arr_scaled)
            fault_raw = score_fn(model, fault_windows, DEVICE)
            scores_fault = gdn_score(calib_raw, fault_raw)
            m = detection_metrics(scores_normal, scores_fault)
            if m is not None:
                per_client_metrics.append(m)

        if not per_client_metrics:
            continue
        mean_across_clients = {
            metric: float(np.mean([row[metric] for row in per_client_metrics]))
            for metric in ("auroc", "auprc", "precision", "f1")
        }
        per_bearing_rows.append(mean_across_clients)
        report["bearings"][code] = {"category": cat, "origin": origin, "n": len(arr),
                                      score_name: mean_across_clients}
        print(f"  {code:<6}{cat:<12}{origin:<11}{score_name}=auroc:{mean_across_clients['auroc']:.3f}")

    summary = {
        metric: float(np.mean([row[metric] for row in per_bearing_rows]))
        for metric in ("auroc", "auprc", "precision", "f1")
    } if per_bearing_rows else {}
    print(f"\n{score_name:<24}{'AUROC':>10}{'AUPRC':>10}{'Precision':>12}{'F1':>10}")
    if summary:
        print(f"{score_name:<24}{summary['auroc']:>10.3f}{summary['auprc']:>10.3f}"
              f"{summary['precision']:>12.3f}{summary['f1']:>10.3f}")

    category_summary = {}
    for cat in ["outer_ring", "inner_ring", "combined"]:
        subset = [report["bearings"][code][score_name]["auroc"] for code in ALL_DAMAGED_CODES
                  if code in report["bearings"] and category_of(code) == cat]
        category_summary[cat] = float(np.mean(subset)) if subset else None

    report["summary_mean_auroc_overall"] = {score_name: summary}
    report["category_summary"] = {score_name: category_summary}
    suffix = out_suffix if out_suffix is not None else f"forecast_v2_federated_h10_encdec_synced_{baseline}"
    out_json = OUT_DIR / f"paderborn_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
               OUT_DIR / f"paderborn_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


def main(horizon_mult=10, use_forecast_prior=True, out_suffix=None, linkage="single", node_agg="topk",
         num_prototypes=None, metric="cosine", edge_quantile=None, delta=None):
    num_prototypes = NUM_PROTOTYPES if num_prototypes is None else num_prototypes
    delta = DELTA if delta is None else delta
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    num_nodes = len(NODE_NAMES)
    forecast_h = WINDOW_STRIDE * horizon_mult
    forecast_prior_edges = PRIOR_EDGES if use_forecast_prior else None

    print("loading healthy bearings (K001-K006) as 6 federated clients ...")
    clients = []
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

    models = [JointPrototypeV21Forecast(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=EMBED_DIM,
                                         num_prototypes=num_prototypes, prior_edges=PRIOR_EDGES,
                                         forecast_h=forecast_h, top_k=TOP_K,
                                         forecast_prior_edges=forecast_prior_edges).to(DEVICE)
              for _ in clients]

    shared_init = {k: v.clone() for k, v in models[0].state_dict().items() if not k.startswith("memory.")}
    for model in models[1:]:
        model.load_state_dict(shared_init, strict=False)

    def local_train_step(c, model):
        fit_windows, fit_file_id, fit_n_pos = make_windows(c["fit"])
        fit_vi, fit_chains = build_chains(len(c["fit"]), fit_n_pos, horizon_mult)
        fit_in = fit_windows[fit_vi]
        fit_future = gather_future(fit_windows, fit_chains, WINDOW_STRIDE)
        train_local(model, fit_in, fit_future, LOCAL_EPOCHS, DEVICE, lr=LR, beta=BETA,
                    lambda_edge=LAMBDA_EDGE, lambda_forecast=LAMBDA_FORECAST, batch_size=BATCH_SIZE)
        return len(fit_in)

    print(f"\nfederated training: {ROUNDS} rounds x {LOCAL_EPOCHS} local epochs, "
          f"memory exchange (gamma={GAMMA}, delta={delta}), horizon_mult={horizon_mult}"
          + (", + FedAvg'd encoder/decoder" if SYNC_ENCODER_DECODER else " only"))
    diagnostics_log = run_federated_rounds(
        clients, models, local_train_step, ROUNDS, GAMMA, delta,
        sync_encoder_decoder=SYNC_ENCODER_DECODER, linkage=linkage,
        metric=metric, edge_quantile=edge_quantile)

    print("\ncalibration per client (K/H reference stats) ...")
    calib_data = {}  # code -> (d_node_calib, k_resid_calib, d_mahal_med, d_mahal_iqr)
    for c, model in zip(clients, models):
        calib_windows, _, calib_n_pos = make_windows(c["calib"])
        _, d_node_calib, _, _, idx_calib, _, _ = per_sample_scores(model, calib_windows, DEVICE, batch_size=BATCH_SIZE)
        cov_min = max(MIN_PROTO_SAMPLES, num_nodes + 1)
        model.cov_head.set_calibration(d_node_calib, idx_calib, min_samples=cov_min)
        _, d_node_calib, _, _, idx_calib, _, d_mahal_calib = per_sample_scores(model, calib_windows, DEVICE, batch_size=BATCH_SIZE)
        d_mahal_med = float(np.median(d_mahal_calib))
        d_mahal_iqr = max(float(np.percentile(d_mahal_calib, 75) - np.percentile(d_mahal_calib, 25)), 1e-8)

        calib_vi, calib_chains = build_chains(len(c["calib"]), calib_n_pos, horizon_mult)
        if len(calib_vi):
            calib_in = calib_windows[calib_vi]
            calib_future = gather_future(calib_windows, calib_chains, WINDOW_STRIDE)
            k_resid_calib = forecast_scores(model, calib_in, calib_future, DEVICE, batch_size=BATCH_SIZE)
        else:
            k_resid_calib = np.zeros((0, num_nodes), dtype=np.float32)
        idx_k_calib = idx_calib[calib_vi]
        calib_data[c["code"]] = (d_node_calib, k_resid_calib, d_mahal_med, d_mahal_iqr, idx_calib, idx_k_calib)

    # Per-client calib-loss-quality confidence weights for the loss-weighted
    # fusion alternative to smooth_max -- see federated_train_eval.loss_weighted_combo.
    b_loss = [float(calib_data[c["code"]][0].mean()) for c in clients]
    h_loss = [calib_data[c["code"]][2] for c in clients]  # d_mahal_med
    k_loss = [float(calib_data[c["code"]][1].mean()) if len(calib_data[c["code"]][1]) else float("nan") for c in clients]
    lw_ready = all(not np.isnan(v) for v in k_loss)
    client_lw_weights = (confidence_weights_from_losses({"B": b_loss, "H": h_loss, "K": k_loss})
                          if lw_ready else None)
    if client_lw_weights is None:
        print("  [loss-weighted fusion] skipped: at least one client has no forecast-pairing calib windows")
    else:
        for c, w in zip(clients, client_lw_weights):
            print(f"  [loss-weighted fusion] {c['code']}: {w}")

    print("\nscoring damaged bearings against every client's personalized model ...")
    report = {"config": {"embed_dim": EMBED_DIM, "num_prototypes": num_prototypes, "top_k": TOP_K,
                          "window_len": WINDOW_LEN, "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS,
                          "gamma": GAMMA, "delta": delta, "linkage": linkage,
                          "metric": metric, "edge_quantile": edge_quantile,
                          "prior_edges": EDGES_NAMED,
                          "min_proto_samples": MIN_PROTO_SAMPLES, "clients": HEALTHY_CODES,
                          "sync_encoder_decoder": SYNC_ENCODER_DECODER, "horizon_mult": horizon_mult,
                          "forecast_h": forecast_h, "use_forecast_prior": use_forecast_prior,
                          "top_k_agg": TOP_K_AGG, "node_agg": node_agg, "baseline": "ours"},
              "nodes": NODE_NAMES, "alignment_log": diagnostics_log,
              "bearings": {}}

    normal_scores_per_client = []
    for ci, (c, model) in enumerate(zip(clients, models)):
        d_node_calib, k_resid_calib, d_mahal_med, d_mahal_iqr, idx_calib, idx_k_calib = calib_data[c["code"]]

        test_normal_windows, _, test_n_pos = make_windows(c["test_normal"])
        _, d_node_n, _, _, idx_n, _, d_mahal_n = per_sample_scores(model, test_normal_windows, DEVICE, batch_size=BATCH_SIZE)
        node_score_n = (gdn_score_proto(d_node_calib, idx_calib, d_node_n, idx_n, num_prototypes, MIN_PROTO_SAMPLES)
                         if node_agg == "max" else
                         two_stage_group_score_proto(d_node_calib, idx_calib, d_node_n, idx_n,
                                                      TOP_K_AGG, num_prototypes, MIN_PROTO_SAMPLES))
        h_score_n = (d_mahal_n - d_mahal_med) / d_mahal_iqr
        base_n = add_h_score(base_scores(node_score_n), h_score_n)

        vi_n, chains_n = build_chains(len(c["test_normal"]), test_n_pos, horizon_mult)
        mask_n = np.zeros(len(test_normal_windows), dtype=bool)
        mask_n[vi_n] = True
        if len(vi_n):
            in_n = test_normal_windows[vi_n]
            future_n = gather_future(test_normal_windows, chains_n, WINDOW_STRIDE)
            k_resid_n = forecast_scores(model, in_n, future_n, DEVICE, batch_size=BATCH_SIZE)
            forecast_score_n = (gdn_score_proto(k_resid_calib, idx_k_calib, k_resid_n, idx_n[mask_n],
                                                 num_prototypes, MIN_PROTO_SAMPLES)
                                 if len(k_resid_calib) else np.zeros(len(vi_n)))
        else:
            forecast_score_n = np.zeros(0)
        scores_n = add_forecast_scores(base_n, forecast_score_n, mask_n)
        if client_lw_weights is not None and len(forecast_score_n):
            b_lw_n, h_lw_n = base_n["B_node_max"][mask_n], base_n["H_cov_mahal"][mask_n]
            scores_n["BHK_lw"] = loss_weighted_combo({"B": b_lw_n, "H": h_lw_n, "K": forecast_score_n},
                                                       client_lw_weights[ci])
        normal_scores_per_client.append(scores_n)

    rows = {k: [] for k in normal_scores_per_client[0]}
    for code in ALL_DAMAGED_CODES:
        d = load_bearing_with_physics(code)
        arr = stack_nodes(d)
        cat, origin = category_of(code), damage_origin_of(code)

        per_client_metrics = {k: [] for k in rows}
        for ci, (c, model, scores_normal) in enumerate(zip(clients, models, normal_scores_per_client)):
            d_node_calib, k_resid_calib, d_mahal_med, d_mahal_iqr, idx_calib, idx_k_calib = calib_data[c["code"]]

            arr_scaled = c["scaler"].transform(arr.reshape(-1, num_nodes)).reshape(arr.shape).astype(np.float32)
            fault_windows, _, fault_n_pos = make_windows(arr_scaled)
            _, d_node_f, _, _, idx_f, _, d_mahal_f = per_sample_scores(model, fault_windows, DEVICE, batch_size=BATCH_SIZE)
            node_score_f = (gdn_score_proto(d_node_calib, idx_calib, d_node_f, idx_f, num_prototypes, MIN_PROTO_SAMPLES)
                             if node_agg == "max" else
                             two_stage_group_score_proto(d_node_calib, idx_calib, d_node_f, idx_f,
                                                          TOP_K_AGG, num_prototypes, MIN_PROTO_SAMPLES))
            h_score_f = (d_mahal_f - d_mahal_med) / d_mahal_iqr
            base_f = add_h_score(base_scores(node_score_f), h_score_f)

            vi_f, chains_f = build_chains(len(arr_scaled), fault_n_pos, horizon_mult)
            mask_f = np.zeros(len(fault_windows), dtype=bool)
            mask_f[vi_f] = True
            if len(vi_f):
                in_f = fault_windows[vi_f]
                future_f = gather_future(fault_windows, chains_f, WINDOW_STRIDE)
                k_resid_f = forecast_scores(model, in_f, future_f, DEVICE, batch_size=BATCH_SIZE)
                forecast_score_f = (gdn_score_proto(k_resid_calib, idx_k_calib, k_resid_f, idx_f[mask_f],
                                                      num_prototypes, MIN_PROTO_SAMPLES)
                                     if len(k_resid_calib) else np.zeros(len(vi_f)))
            else:
                forecast_score_f = np.zeros(0)
            scores_fault = add_forecast_scores(base_f, forecast_score_f, mask_f)
            if client_lw_weights is not None and len(forecast_score_f):
                b_lw_f, h_lw_f = base_f["B_node_max"][mask_f], base_f["H_cov_mahal"][mask_f]
                scores_fault["BHK_lw"] = loss_weighted_combo({"B": b_lw_f, "H": h_lw_f, "K": forecast_score_f},
                                                                client_lw_weights[ci])

            for key in rows:
                m = detection_metrics(scores_normal.get(key, np.zeros(0)), scores_fault.get(key, np.zeros(0)))
                if m is None:
                    continue
                per_client_metrics[key].append(m)

        mean_across_clients = {
            k: {metric: float(np.mean([row[metric] for row in v])) for metric in ("auroc", "auprc", "precision", "f1")}
            for k, v in per_client_metrics.items() if v
        }
        for key in mean_across_clients:
            rows[key].append(mean_across_clients[key])
        report["bearings"][code] = {"category": cat, "origin": origin, "n": len(arr),
                                      "per_client_metrics": {clients[i]["code"]: {k: v[i] for k, v in per_client_metrics.items() if i < len(v)}
                                                             for i in range(len(clients))},
                                      "mean_across_clients": mean_across_clients}
        print(f"  {code:<6}{cat:<12}{origin:<11}"
              + "  ".join(f"{k}=auroc:{v['auroc']:.3f}" for k, v in mean_across_clients.items()))

    print(f"\n{'method':<16}{'AUROC':>10}{'AUPRC':>10}{'Precision':>12}{'F1':>10}")
    summary = {}
    for key in rows:
        if not rows[key]:
            continue
        mean_metrics = {m: float(np.mean([row[m] for row in rows[key]])) for m in ("auroc", "auprc", "precision", "f1")}
        summary[key] = mean_metrics
        print(f"{key:<16}{mean_metrics['auroc']:>10.3f}{mean_metrics['auprc']:>10.3f}"
              f"{mean_metrics['precision']:>12.3f}{mean_metrics['f1']:>10.3f}")
    print(f"\n{'method':<24}{'outer_ring':>12}{'inner_ring':>12}{'combined':>12}")
    category_summary = {}
    for key in rows:
        cat_means = {}
        for cat in ["outer_ring", "inner_ring", "combined"]:
            subset = [report["bearings"][code]["mean_across_clients"][key]["auroc"] for code in ALL_DAMAGED_CODES
                      if category_of(code) == cat and key in report["bearings"][code]["mean_across_clients"]]
            cat_means[cat] = float(np.mean(subset)) if subset else None
        category_summary[key] = cat_means
        print(f"{key:<24}" + "".join(f"{(cat_means[c] if cat_means[c] is not None else float('nan')):>12.3f}"
                                      for c in ["outer_ring", "inner_ring", "combined"]))

    report["summary_mean_auroc_overall"] = summary
    report["category_summary"] = category_summary
    suffix = out_suffix if out_suffix is not None else (
        f"forecast_v2_federated_h{horizon_mult}{'_noprior' if not use_forecast_prior else ''}"
        + ("_encdec_synced" if SYNC_ENCODER_DECODER else "")
        + ("_bmax" if node_agg == "max" else "")
        + (f"_metric_{metric}" if metric != "cosine" else ""))
    out_json = OUT_DIR / f"paderborn_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
               OUT_DIR / f"paderborn_{suffix}.pth")
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
        main(horizon_mult=args.horizon_mult, use_forecast_prior=not args.no_forecast_prior,
             out_suffix=args.out_suffix, linkage=args.linkage, node_agg=args.node_agg,
             delta=args.delta)
    elif args.baseline == "fedpro":
        main_fedpro_baseline(out_suffix=args.out_suffix)
    elif args.baseline == "fedcpg":
        main_fedcpg_baseline(out_suffix=args.out_suffix)
    elif args.baseline == "ufedhy":
        main_ufedhy_baseline(out_suffix=args.out_suffix)
    else:
        main_faithful_baseline(args.baseline, out_suffix=args.out_suffix, num_clusters=args.num_clusters)
