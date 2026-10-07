#!/usr/bin/env python3
"""
FEDERATED JointPrototypeV21Forecast (V2.1 + ForecastHead, signal PD) on
ME-AD, treating each of its 16 operation codes as a federated client --
see `src/dataloaders/me_ad/me_ad_adapter.py`'s module docstring for the full
dataset description, client-definition rationale (real, independently-
recorded motion programs, same distinction Paderborn draws between
bearing codes), and the important caveat that ALL 16 clients share the
SAME underlying physical fault (progressive joint-3 actuator wear on one
shared robot), unlike robo_fleet/Paderborn's independent per-client
faults.

Healthy/faulty split is ME-AD's OWN "Suggested AD Tasks" convention,
RE-VERIFIED empirically (a literal cycles-[0:70]/[70:120] reading gives
near-chance AUROC everywhere -- the fault only becomes visible at the
TRUE END of each code's ~1600-2000-cycle run): first 70 cycles = healthy
pool, LAST 50 cycles of the full run = faulty pool, per operation code --
see `me_ad_adapter.py`'s module docstring for the empirical check. Each client's
healthy pool is further split fit/calib/test_normal
(`FIT_FRACTION`/`CALIB_FRACTION`, same convention as every other
dataset); the faulty pool is this client's single fault set (label
"joint3_degradation" -- ME-AD has no fault SUBtypes to enumerate, unlike
robo_fleet's 4 named faults), scored against that SAME client's own
model (robo_fleet/ALFA's per-client-owns-its-fault convention, not
Paderborn's cross-client-scores-an-external-bearing convention -- there
is no external held-out unit here, every client has its own fault
occurrence).

Declares a kinematic-chain physics prior: adjacent joints' filtered
torque channels (`tau_filt_i -- tau_filt_{i+1}`) -- see
`me_ad_adapter.EDGES_NAMED` -- used as `edge_head`'s soft, LEARNED-
STRENGTH attention bias (a per-edge `"nonlinear"` TYPING of these edges,
`me_ad_adapter.EDGE_TYPES`, is still defined there but no longer imported
or used here as of 2026-09-12's switch away from the declared-edge
`typed_head`, see below).

**2026-09-12: switched from `JointPrototypeV31Forecast` to `V21Forecast`,
dropping the declared-physics-edge `typed_head`/signal-E machinery
project-wide** -- a controlled ablation on robo_fleet found it worth only
~0.005 FD_JD_PD_max, and the same simpler no-declared-edges architecture
ALFA/SMD already used is now the one mainline everywhere -- see
`memory/v21-mainline-switch.md`.

The training+eval mainline (round loop, scoring primitives, `train_local`)
lives in `src/federated/federated_train_eval.py`, shared verbatim with
`run_robo_fleet_bck_federated.py`/`run_paderborn_bck_federated.py` (all
`JointPrototypeV21Forecast` as of 2026-09-12) -- this script only supplies
ME-AD-specific data loading (per-cycle windowing, since each pick-and-
place cycle is a variable-length recording, same pattern as Paderborn's
per-file windowing) and CLI plumbing.

Usage:
    python3 run_me_ad_bck_federated.py [--horizon-mult M] [--no-forecast-prior]
        [--out-suffix NAME] [--baseline {ours,fedavg,ifcaae,fedexdnn,fedpro}]

Requires `data/me_ad_raw/ME-AD.zip` (10.79GB, Zenodo DOI
10.5281/zenodo.20817530, CC-BY-SA-4.0) to already be downloaded -- not
committed to the repo, no separate "build" step needed (the adapter reads
directly from the zip's per-cycle members, no full unzip required).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "federated"))
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "FedPRO"))
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "FedCPG"))
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "uFedHy-DisMTSADD"))
sys.path.insert(0, str(REPO_ROOT / "src" / "dataloaders" / "me_ad"))
from joint_prototype_model import JointPrototypeV21Forecast  # noqa: E402
from baseline_models import (ReconOnlyModel, ExemplarOnlyModel, FedPROModel,  # noqa: E402
                              FedCPGModel, SCNorTransformerModel, Hypernetwork)
from federated_train_eval import (two_stage_group_score, base_scores,  # noqa: E402
                           add_forecast_scores, add_h_score, per_sample_scores, forecast_scores,
                           train_local, run_federated_rounds, gdn_score, detection_metrics,
                           two_stage_group_score_proto, gdn_score_proto, smooth_max,
                           train_local_recon, train_local_exemplar,
                           per_node_recon_error, per_node_exemplar_deviation)
from fedpro import (train_local_classifier, embed_all, build_initial_prototypes_kmeans,  # noqa: E402
                     refine_prototypes_margin, retrieve_and_vote, confidence_ensemble)
from fedcpg import train_local_fedcpg, aggregate_global_prototypes, predict_anomaly_score  # noqa: E402
from ufedhy import hypernet_round_update  # noqa: E402
from comm_cost import round_comm_record  # noqa: E402

SYNC_ENCODER_DECODER = True
from me_ad_adapter import (  # noqa: E402
    load_operation_code, OPERATION_CODES, motion_family, NODE_NAMES, EDGES_NAMED,
    NUM_HEALTHY_CYCLES, NUM_FAULTY_CYCLES,
)
# EDGE_TYPES (per-edge "proportional"/"nonlinear" typing, still defined in
# me_ad_adapter.py) is no longer imported here as of 2026-09-12's switch to
# JointPrototypeV21Forecast -- see that class's docstring and
# memory/v21-mainline-switch.md. EDGES_NAMED/PRIOR_EDGES are still used as
# `edge_head`'s soft, LEARNED-STRENGTH attention bias.

OUT_DIR = REPO_ROOT / "checkpoints" / "me_ad"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NODE_IDX = {name: i for i, name in enumerate(NODE_NAMES)}
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
NUM_PROTOTYPES = 8
TOP_K = 5
TOP_K_AGG = 3
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_FORECAST = 0.5
MIN_PROTO_SAMPLES = 5
GAMMA = 0.5
DELTA = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def build_clients():
    """Returns `{code: {"healthy_cycles": [...], "faulty_cycles": [...]}}`
    -- raw (unscaled) per-cycle arrays, one dict entry per operation code
    (client). `healthy_cycles` = first `NUM_HEALTHY_CYCLES` of the run,
    `faulty_cycles` = LAST `NUM_FAULTY_CYCLES` of the run (NOT cycles
    `NUM_HEALTHY_CYCLES:NUM_HEALTHY_CYCLES+NUM_FAULTY_CYCLES` -- see
    `me_ad_adapter.py`'s module docstring for why: the fault only becomes
    visible at the true end of each code's ~1600-2000-cycle run, verified
    empirically). Cycles are NOT concatenated here (each cycle stays a
    separate `[T_i, 24]` array, like Paderborn's single-file convention
    but per-cycle instead of per-bearing) so `make_windows` never bridges
    two different cycles into one window."""
    print(f"loading ME-AD, splitting into {len(OPERATION_CODES)} operation-code federated clients ...")
    clients = {}
    for code in OPERATION_CODES:
        healthy_cycles, faulty_cycles = load_operation_code(code, num_healthy=NUM_HEALTHY_CYCLES,
                                                              num_faulty=NUM_FAULTY_CYCLES)
        clients[code] = {"healthy_cycles": healthy_cycles, "faulty_cycles": faulty_cycles}
        print(f"  {code} ({motion_family(code)}): {len(healthy_cycles)} healthy cycles (start of run), "
              f"{len(faulty_cycles)} faulty cycles (end of run)")
    return clients


def _pad_stack(cycles):
    """Stack variable-length `[T_i, N]` cycle arrays into one `[n_cycles,
    T_min, N]` array by truncating every cycle to the shortest one's
    length -- `make_windows` (ported from Paderborn) needs a single
    fixed-length axis, and cycles vary by a handful of samples around
    ~1300 (or ~380 for the F3 family) due to per-cycle corruption
    trimming, never enough to matter for window-level scoring."""
    t_min = min(c.shape[0] for c in cycles)
    return np.stack([c[:t_min] for c in cycles], axis=0)


def main(horizon_mult=4, use_forecast_prior=True, out_suffix=None, linkage="single", node_agg="topk",
         num_prototypes=None, metric="cosine", edge_quantile=None, delta=None):
    num_prototypes = NUM_PROTOTYPES if num_prototypes is None else num_prototypes
    delta = DELTA if delta is None else delta
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    num_nodes = len(NODE_NAMES)
    forecast_h = WINDOW_STRIDE * horizon_mult
    forecast_prior_edges = PRIOR_EDGES if use_forecast_prior else None

    raw_clients = build_clients()
    clients = []
    for code in OPERATION_CODES:
        healthy_arr = _pad_stack(raw_clients[code]["healthy_cycles"])
        faulty_arr = _pad_stack(raw_clients[code]["faulty_cycles"])
        n = len(healthy_arr)
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        fit_raw, calib_raw, test_raw = healthy_arr[:n_fit], healthy_arr[n_fit : n_fit + n_calib], healthy_arr[n_fit + n_calib :]
        scaler = StandardScaler().fit(fit_raw.reshape(-1, num_nodes))

        def scale(a, scaler=scaler):
            n_, t_, f_ = a.shape
            return scaler.transform(a.reshape(-1, f_)).reshape(n_, t_, f_).astype(np.float32)

        clients.append({"code": code, "fit": scale(fit_raw), "calib": scale(calib_raw),
                          "test_normal": scale(test_raw), "fault": scale(faulty_arr), "scaler": scaler})

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

    report = {"config": {"embed_dim": EMBED_DIM, "num_prototypes": num_prototypes, "top_k": TOP_K,
                          "window_len": WINDOW_LEN, "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS,
                          "gamma": GAMMA, "delta": delta, "linkage": linkage,
                          "metric": metric, "edge_quantile": edge_quantile,
                          "prior_edges": EDGES_NAMED,
                          "min_proto_samples": MIN_PROTO_SAMPLES, "clients": OPERATION_CODES,
                          "sync_encoder_decoder": SYNC_ENCODER_DECODER, "horizon_mult": horizon_mult,
                          "forecast_h": forecast_h, "use_forecast_prior": use_forecast_prior,
                          "top_k_agg": TOP_K_AGG, "node_agg": node_agg, "baseline": "ours",
                          "note": "all 16 clients share ONE physical robot's progressive joint-3 fault "
                                  "-- see me_ad_adapter.py's module docstring caveat"},
              "nodes": NODE_NAMES, "alignment_log": diagnostics_log, "codes": {}}

    all_pairs = {}
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
        scores_normal = add_forecast_scores(base_n, forecast_score_n, mask_n)

        fault_windows, _, fault_n_pos = make_windows(c["fault"])
        _, d_node_f, _, _, idx_f, _, d_mahal_f = per_sample_scores(model, fault_windows, DEVICE, batch_size=BATCH_SIZE)
        node_score_f = (gdn_score_proto(d_node_calib, idx_calib, d_node_f, idx_f, num_prototypes, MIN_PROTO_SAMPLES)
                         if node_agg == "max" else
                         two_stage_group_score_proto(d_node_calib, idx_calib, d_node_f, idx_f,
                                                      TOP_K_AGG, num_prototypes, MIN_PROTO_SAMPLES))
        h_score_f = (d_mahal_f - d_mahal_med) / d_mahal_iqr
        base_f = add_h_score(base_scores(node_score_f), h_score_f)

        vi_f, chains_f = build_chains(len(c["fault"]), fault_n_pos, horizon_mult)
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

        metrics = {}
        for key in scores_normal:
            m = detection_metrics(scores_normal[key], scores_fault.get(key, np.zeros(0)))
            if m is None:
                continue
            metrics[key] = m
            all_pairs.setdefault(key, []).append(m)
        report["codes"][c["code"]] = {"motion_family": motion_family(c["code"]),
                                       "fault_types": {"joint3_degradation": {"n": len(c["fault"]), **metrics}}}
        print(f"  {c['code']:<6}{motion_family(c['code']):<4}"
              + "  ".join(f"{k}=auroc:{v['auroc']:.3f}" for k, v in metrics.items()))

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
    out_json = OUT_DIR / f"me_ad_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
               OUT_DIR / f"me_ad_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


def main_fedpro_baseline(out_suffix=None, k_per_class=3, top_k=3, tau=0.1, embed_dim=EMBED_DIM,
                          fedavg_rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, proto_epochs=50, proto_lr=1e-3):
    """FedPRO (Zhou et al., IEEE TC 2026) on ME-AD -- a deliberate
    exception to fit-on-healthy: supervised BINARY classification
    (healthy=0 vs joint3_degradation=1). Unlike Paderborn (whose clients
    have no fault data of their own by construction, needing a
    round-robin partition workaround), ME-AD's clients already own BOTH
    classes exactly like robo_fleet/ALFA (each operation code has its
    own healthy pool AND its own end-of-run faulty pool -- see
    `me_ad_adapter.py`), so no reinterpretation of "client" is needed
    here: same per-client-owns-both-classes design as robo_fleet's
    original FedPRO baseline, just with 2 classes instead of 5 (ME-AD has
    no fault SUBtypes to enumerate). See `benchmark/FedPRO/fedpro.py`'s
    module docstring for the method itself."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    num_nodes = len(NODE_NAMES)
    num_classes = 2

    print("loading ME-AD, splitting into 16 operation-code federated clients, baseline=fedpro "
          "(supervised, healthy vs joint3_degradation, NOT fit-on-healthy) ...")
    raw_clients = build_clients()
    client_data = []
    for code in OPERATION_CODES:
        healthy_arr = _pad_stack(raw_clients[code]["healthy_cycles"])
        faulty_arr = _pad_stack(raw_clients[code]["faulty_cycles"])
        # healthy_arr/faulty_arr have INDEPENDENTLY-computed time-axis lengths
        # (`_pad_stack` truncates each group to its own shortest cycle) --
        # reshape to 2D (flatten cycle+time) BEFORE concatenating so the
        # mismatched time axis never needs to match between the two groups.
        scaler = StandardScaler().fit(
            np.concatenate([healthy_arr.reshape(-1, num_nodes), faulty_arr.reshape(-1, num_nodes)], axis=0))

        def scale(a, scaler=scaler):
            n_, t_, f_ = a.shape
            return scaler.transform(a.reshape(-1, f_)).reshape(n_, t_, f_).astype(np.float32)

        windows_h, _, _ = make_windows(scale(healthy_arr))
        windows_f, _, _ = make_windows(scale(faulty_arr))
        x_all = np.concatenate([windows_h, windows_f], axis=0)
        y_all = np.concatenate([np.zeros(len(windows_h), dtype=np.int64),
                                 np.ones(len(windows_f), dtype=np.int64)])

        perm = rng.permutation(len(x_all))
        n_train = max(1, int(0.7 * len(x_all)))
        train_idx, test_idx = perm[:n_train], perm[n_train:]
        client_data.append({"code": code, "x_train": x_all[train_idx], "y_train": y_all[train_idx],
                              "x_test": x_all[test_idx], "y_test": y_all[test_idx]})
        print(f"  {code} ({motion_family(code)}): train={len(train_idx)} test={len(test_idx)}")

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
                          "note": "FedPRO is a supervised binary classifier wrapper (Zhou et al., "
                                  "IEEE TC 2026) on healthy vs joint3_degradation labels -- ME-AD's "
                                  "16 operation-code clients each own both classes by construction "
                                  "(unlike Paderborn), no client-repartitioning needed. See "
                                  "benchmark/FedPRO/fedpro.py"},
              "codes": {}}

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

        normal_mask = y_test == 0
        fault_mask = y_test == 1
        m = detection_metrics(anomaly_score[normal_mask], anomaly_score[fault_mask])
        report["codes"][c["code"]] = {"motion_family": motion_family(c["code"]), "accuracy": accuracy,
                                       "fault_types": {"joint3_degradation": {"n": int(fault_mask.sum()),
                                                                                "fedpro_score": m}} if m else {}}
        if m is not None:
            all_pairs.append(m)
        print(f"  {c['code']:<6}{motion_family(c['code']):<4}accuracy={accuracy:.3f}"
              + (f"  fedpro_score=auroc:{m['auroc']:.3f}" if m else "  [skip: no fault windows]"))

    mean_accuracy = float(np.mean([v["accuracy"] for v in report["codes"].values()]))
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
    suffix = out_suffix if out_suffix is not None else "forecast_v2_federated_h4_encdec_synced_fedpro"
    out_json = OUT_DIR / f"me_ad_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "bank_protos": bank_protos,
                "bank_labels": bank_labels, "report": report}, OUT_DIR / f"me_ad_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


def main_fedcpg_baseline(out_suffix=None, embed_dim=EMBED_DIM, proj_dim=32, fedcpg_rounds=ROUNDS,
                          local_epochs=LOCAL_EPOCHS, alpha=0.5, beta=0.2, tau=0.1):
    """FedCPG (Li et al., *Computers in Industry* 2025) on ME-AD -- same
    per-client-owns-both-classes split FedPRO already established
    (`main_fedpro_baseline`'s docstring): binary healthy vs
    joint3_degradation, no client-repartitioning needed (unlike
    Paderborn). See `benchmark/FedCPG/fedcpg.py` for the algorithm and
    `docs/1-s2.0-S0166361524001088-main.pdf` for the paper."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    num_nodes = len(NODE_NAMES)
    num_classes = 2

    print("loading ME-AD, splitting into 16 operation-code federated clients, baseline=fedcpg "
          "(supervised, healthy vs joint3_degradation, NOT fit-on-healthy) ...")
    raw_clients = build_clients()
    client_data = []
    for code in OPERATION_CODES:
        healthy_arr = _pad_stack(raw_clients[code]["healthy_cycles"])
        faulty_arr = _pad_stack(raw_clients[code]["faulty_cycles"])
        scaler = StandardScaler().fit(
            np.concatenate([healthy_arr.reshape(-1, num_nodes), faulty_arr.reshape(-1, num_nodes)], axis=0))

        def scale(a, scaler=scaler):
            n_, t_, f_ = a.shape
            return scaler.transform(a.reshape(-1, f_)).reshape(n_, t_, f_).astype(np.float32)

        windows_h, _, _ = make_windows(scale(healthy_arr))
        windows_f, _, _ = make_windows(scale(faulty_arr))
        x_all = np.concatenate([windows_h, windows_f], axis=0)
        y_all = np.concatenate([np.zeros(len(windows_h), dtype=np.int64),
                                 np.ones(len(windows_f), dtype=np.int64)])

        perm = rng.permutation(len(x_all))
        n_train = max(1, int(0.7 * len(x_all)))
        train_idx, test_idx = perm[:n_train], perm[n_train:]
        client_data.append({"code": code, "x_train": x_all[train_idx], "y_train": y_all[train_idx],
                              "x_test": x_all[test_idx], "y_test": y_all[test_idx]})
        print(f"  {code} ({motion_family(code)}): train={len(train_idx)} test={len(test_idx)}")

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
                          "note": "FedCPG (Li et al., Computers in Industry 2025) on healthy vs "
                                  "joint3_degradation labels -- see benchmark/FedCPG/fedcpg.py"},
              "codes": {}}

    all_pairs = []
    for c, model in zip(client_data, models):
        x_test, y_test = c["x_test"], c["y_test"]
        with torch.no_grad():
            model.eval()
            logits = model(torch.from_numpy(x_test).to(DEVICE))
            pred = logits.argmax(dim=1).cpu().numpy()
        accuracy = float((pred == y_test).mean())
        anomaly_score = predict_anomaly_score(model, x_test, DEVICE, batch_size=BATCH_SIZE)

        normal_mask = y_test == 0
        fault_mask = y_test == 1
        m = detection_metrics(anomaly_score[normal_mask], anomaly_score[fault_mask])
        report["codes"][c["code"]] = {"motion_family": motion_family(c["code"]), "accuracy": accuracy,
                                       "fault_types": {"joint3_degradation": {"n": int(fault_mask.sum()),
                                                                                "fedcpg_score": m}} if m else {}}
        if m is not None:
            all_pairs.append(m)
        print(f"  {c['code']:<6}{motion_family(c['code']):<4}accuracy={accuracy:.3f}"
              + (f"  fedcpg_score=auroc:{m['auroc']:.3f}" if m else "  [skip: no fault windows]"))

    mean_accuracy = float(np.mean([v["accuracy"] for v in report["codes"].values()]))
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
    out_json = OUT_DIR / f"me_ad_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
               OUT_DIR / f"me_ad_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


def main_ufedhy_baseline(out_suffix=None, hyper_embed_dim=32, hyper_hidden_dim=64,
                          transformer_embed_dim=16, ufedhy_rounds=ROUNDS, local_epochs=LOCAL_EPOCHS,
                          hyper_lr=1e-3):
    """uFedHy-DisMTSADD (Hao et al., *Information Processing & Management*
    2025) on ME-AD -- fit-on-healthy, runs like FedAvg/IFCAAE/Fed-ExDNN.
    See `benchmark/uFedHy-DisMTSADD/ufedhy.py` for the hypernetwork update rule and
    `docs/1-s2.0-S0306457325000494-main.pdf` for the paper."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    num_nodes = len(NODE_NAMES)

    print("loading ME-AD, splitting into 16 operation-code federated clients, baseline=ufedhy ...")
    raw_clients = build_clients()
    clients = []
    for code in OPERATION_CODES:
        healthy_arr = _pad_stack(raw_clients[code]["healthy_cycles"])
        faulty_arr = _pad_stack(raw_clients[code]["faulty_cycles"])
        n = len(healthy_arr)
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        fit_raw, calib_raw, test_raw = healthy_arr[:n_fit], healthy_arr[n_fit : n_fit + n_calib], healthy_arr[n_fit + n_calib :]
        scaler = StandardScaler().fit(fit_raw.reshape(-1, num_nodes))

        def scale(a, scaler=scaler):
            n_, t_, f_ = a.shape
            return scaler.transform(a.reshape(-1, f_)).reshape(n_, t_, f_).astype(np.float32)

        clients.append({"code": code, "fit": scale(fit_raw), "calib": scale(calib_raw),
                          "test_normal": scale(test_raw), "fault": scale(faulty_arr), "scaler": scaler})

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
                          "clients": OPERATION_CODES, "score_name": "recon_score",
                          "note": "localization uses this project's own per-node argmax convention, NOT "
                                  "the paper's own PC-algorithm+PageRank causal diagnosis -- see "
                                  "benchmark/uFedHy-DisMTSADD/ufedhy.py's module docstring"},
              "nodes": NODE_NAMES, "alignment_log": diagnostics_log, "codes": {}}

    all_pairs = []
    for c, model in zip(clients, models):
        calib_windows, _, _ = make_windows(c["calib"])
        calib_raw = per_node_recon_error(model, calib_windows, DEVICE)
        test_normal_windows, _, _ = make_windows(c["test_normal"])
        test_raw = per_node_recon_error(model, test_normal_windows, DEVICE)
        scores_normal = gdn_score(calib_raw, test_raw)

        fault_windows, _, _ = make_windows(c["fault"])
        fault_raw = per_node_recon_error(model, fault_windows, DEVICE)
        scores_fault = gdn_score(calib_raw, fault_raw)

        m = detection_metrics(scores_normal, scores_fault)
        if m is None:
            continue
        report["codes"][c["code"]] = {"motion_family": motion_family(c["code"]),
                                       "n": len(c["fault"]), "recon_score": m}
        all_pairs.append(m)
        print(f"  {c['code']:<6}{motion_family(c['code']):<4}recon_score=auroc:{m['auroc']:.3f}")

    summary = {
        metric: float(np.mean([row[metric] for row in all_pairs])) for metric in ("auroc", "auprc", "precision", "f1")
    } if all_pairs else {}
    print(f"\n{'recon_score':<16}{'AUROC':>10}{'AUPRC':>10}{'Precision':>12}{'F1':>10}")
    if summary:
        print(f"{'recon_score':<16}{summary['auroc']:>10.3f}{summary['auprc']:>10.3f}"
              f"{summary['precision']:>12.3f}{summary['f1']:>10.3f}")

    report["summary_mean_auroc_overall"] = {"recon_score": summary}
    suffix = out_suffix if out_suffix is not None else "ufedhy"
    out_json = OUT_DIR / f"me_ad_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models],
                "hypernet_state_dict": hypernet.state_dict(), "report": report},
               OUT_DIR / f"me_ad_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


def main_faithful_baseline(baseline, out_suffix=None, num_clusters=2, num_prototypes=NUM_PROTOTYPES,
                            horizon_mult=4):
    """FedAvg / IFCAAE / Fed-ExDNN, each with its OWN minimal architecture
    and own single anomaly score (no FD/JD/PD/FD+PD/FD+JD+PD) -- see
    `src/models/baseline_models.py`'s module docstring. No forecast
    chains needed (none of these three baselines forecast): plain
    per-cycle windows only."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    num_nodes = len(NODE_NAMES)

    raw_clients = build_clients()
    clients = []
    for code in OPERATION_CODES:
        healthy_arr = _pad_stack(raw_clients[code]["healthy_cycles"])
        faulty_arr = _pad_stack(raw_clients[code]["faulty_cycles"])
        n = len(healthy_arr)
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        fit_raw, calib_raw, test_raw = healthy_arr[:n_fit], healthy_arr[n_fit : n_fit + n_calib], healthy_arr[n_fit + n_calib :]
        scaler = StandardScaler().fit(fit_raw.reshape(-1, num_nodes))

        def scale(a, scaler=scaler):
            n_, t_, f_ = a.shape
            return scaler.transform(a.reshape(-1, f_)).reshape(n_, t_, f_).astype(np.float32)

        clients.append({"code": code, "fit": scale(fit_raw), "calib": scale(calib_raw),
                          "test_normal": scale(test_raw), "fault": scale(faulty_arr), "scaler": scaler})

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
                          "clients": OPERATION_CODES, "score_name": score_name},
              "nodes": NODE_NAMES, "alignment_log": diagnostics_log, "codes": {}}

    all_pairs = []
    for c, model in zip(clients, models):
        calib_windows, _, _ = make_windows(c["calib"])
        calib_raw = score_fn(model, calib_windows, DEVICE)
        test_normal_windows, _, _ = make_windows(c["test_normal"])
        test_raw = score_fn(model, test_normal_windows, DEVICE)
        scores_normal = gdn_score(calib_raw, test_raw)

        fault_windows, _, _ = make_windows(c["fault"])
        fault_raw = score_fn(model, fault_windows, DEVICE)
        scores_fault = gdn_score(calib_raw, fault_raw)

        m = detection_metrics(scores_normal, scores_fault)
        if m is None:
            continue
        report["codes"][c["code"]] = {"motion_family": motion_family(c["code"]),
                                       "n": len(c["fault"]), score_name: m}
        all_pairs.append(m)
        print(f"  {c['code']:<6}{motion_family(c['code']):<4}{score_name}=auroc:{m['auroc']:.3f}")

    summary = {
        metric: float(np.mean([row[metric] for row in all_pairs])) for metric in ("auroc", "auprc", "precision", "f1")
    } if all_pairs else {}
    print(f"\n{score_name:<16}{'AUROC':>10}{'AUPRC':>10}{'Precision':>12}{'F1':>10}")
    if summary:
        print(f"{score_name:<16}{summary['auroc']:>10.3f}{summary['auprc']:>10.3f}"
              f"{summary['precision']:>12.3f}{summary['f1']:>10.3f}")

    report["summary_mean_auroc_overall"] = {score_name: summary}
    suffix = out_suffix if out_suffix is not None else f"forecast_v2_federated_h{horizon_mult}_encdec_synced_{baseline}"
    out_json = OUT_DIR / f"me_ad_{suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
               OUT_DIR / f"me_ad_{suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-mult", type=int, default=4)
    parser.add_argument("--no-forecast-prior", action="store_true")
    parser.add_argument("--out-suffix", type=str, default=None)
    parser.add_argument("--linkage", type=str, default="single", choices=["single", "complete"])
    parser.add_argument("--node-agg", type=str, default="topk", choices=["topk", "max"])
    parser.add_argument("--baseline", type=str, default="ours",
                         choices=["ours", "fedavg", "ifcaae", "fedexdnn", "fedpro", "fedcpg", "ufedhy"],
                         help="server-aggregation strategy -- see run_federated_rounds' docstring "
                              "('fedpro'/'fedcpg' are supervised binary classification, healthy vs "
                              "joint3_degradation -- see fedpro.py/fedcpg.py; 'ufedhy' is "
                              "fit-on-healthy -- see ufedhy.py)")
    parser.add_argument("--num-clusters", type=int, default=2)
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
        main_faithful_baseline(args.baseline, out_suffix=args.out_suffix, num_clusters=args.num_clusters,
                                horizon_mult=args.horizon_mult)
