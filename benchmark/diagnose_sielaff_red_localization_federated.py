#!/usr/bin/env python3
"""
Node-level DIAGNOSIS (root-cause localization) test for B/C/H/K and their
pairwise combinations BK/CK/HK on Sielaff, federated, against the
CORRECT fault-type ground truth: the 47 company/domain-expert
RED-severity error IDs in `data/sielaff_data/sielaff_ground_truth.md`
(via `data/sielaff_red/`, built by `benchmark/datasets/build_sielaff_red.py`),
per [[sielaff-red-fault-federated]] and per the user's explicit
correction: "所有公司文件中提供的红色的error 都要被视作故障类型" (every
red-severity error listed in the company files must be treated as a
fault type). This REPLACES `diagnose_sielaff_localization_federated.py`,
which incorrectly treated `data/sielaff/`'s 5 coarse
`dominant_error_group_in_window` groups as if they were the real fault
types -- see `memory/three-dataset-bck-comparison.md`'s retraction note.

Architecture: `JointPrototypeV21Forecast` (same as the retracted script
-- Sielaff has no declared physics edges, see
`benchmark/datasets/sielaff_physics.md`), 10 real machines as 10
federated clients, memory-only exchange (no encoder/decoder FedAvg, same
as `run_sielaff_red_v2_1_federated.py`/`run_sielaff_bck_federated.py`).
`horizon_mult=10`, `WINDOW_LEN=8`/`STRIDE=2` (same window geometry as the
red AUROC script -- `data/sielaff_red/` uses the identical 4h/1h-stride
windowing as `data/sielaff/`).

**Multi-label windows**: a 4h red window can contain UP TO 23 concurrent
red error IDs at once (verified: `window_red_ids.json`). Per
`analyze_sielaff_red_feature_attribution.py`'s established convention,
each red window's single argmax vote is attributed to EVERY red ID
present in that window (not split or deduplicated) -- a window's true
root cause is genuinely ambiguous among its co-occurring IDs, so this is
reported as designed, not an artifact.

**Pooled calibration reference, per-client scoring** (also borrowed from
`analyze_sielaff_red_feature_attribution.py`, with the same justification):
each client's calib+test_normal split is only ~6-9 windows here (too few
for a stable per-node IQR -- verified in that script to floor-degenerate
per-client). Z-scoring MEDIAN/IQR reference is POOLED across all 10
clients' calib+test_normal windows; each red window is still scored by
its OWN client's own trained model. This is a deliberate, documented
departure from the federated policy's per-client-calibrates-locally
convention, appropriate here for the same reason the original
B-only attribution script departed from it.

Ground truth: `feature_domain()` / `FAULT_EXPECTED_DOMAIN` /
`NO_SENSOR_KEYWORDS` are copied verbatim from
`analyze_sielaff_red_feature_attribution.py` (same 39 raw columns, same
physical-semantics mapping) -- reused, not redefined, so the domain
knowledge stays a single source of truth. Red IDs whose name matches no
keyword rule are `unmapped`; red IDs matching a `NO_SENSOR_KEYWORDS`
keyword (compactor/crate_/safety_circuit/dooropen/misc_*) have no
directly-instrumented sensor in this column set and are reported as
`no_direct_sensor`, not silently scored against a domain that doesn't
exist.

Usage:
    python3 diagnose_sielaff_red_localization_federated.py [--horizon-mult M] [--rounds R] [--local-epochs E] [--out-suffix NAME]
"""
import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "federated"))
from joint_prototype_model import JointPrototypeV21Forecast  # noqa: E402
from federated_memory import align_and_split  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "sielaff_red"
OUT_DIR = REPO_ROOT / "checkpoints" / "sielaff"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hardware-heterogeneous columns: verified empirically (per-machine zero-fraction
# check, not assumed) to be permanently 0.0 for the large majority of the 10 real
# Sielaff machines because those machines simply don't have that piece of hardware
# (fewer sorting gates, no ring-camera module) -- not because the event is rare.
# `read SM/barcode_RingCamera *`: only 1/10 machines has any nonzero reading at
# all. `schließen/öffnen_Weiche 3-7`: only 3-4/10 machines have gates beyond the
# first two.
#
# TESTED (2026-08-19) and found to HURT, not help: dropping these 22 columns
# (`--drop-heterogeneous-hardware`) roughly HALVES K/BK/CK's confound-excluded
# aggregate direct% (26.3/26.0/25.9 -> 13.7/14.0/13.7), while B/C/H barely move.
# Cause: the removed columns (`öffnen_Weiche 2`'s neighbors and the ring-camera
# reads) were carrying REAL signal for the few equipped machines specifically on
# the ring_camera_read/sorting_gate_switch-domain fault types (bottle_collision/
# bottle_out_of_range/etc.) -- with them gone, argmax on those fault types falls
# through entirely to the SAME residual "generic activity" confound
# (`receipt_count`/`reject_count`/`cleaning_duration`) that already dominates
# elsewhere, since nothing else is left to compete with it. A blanket column
# drop is the wrong fix for hardware heterogeneity here -- it throws away a
# minority of clients' real signal without addressing the actual dominant
# problem (the activity-proxy confound, which persists with or without these
# columns). Per-client node masking (letting each client score only its own
# equipped hardware, without corrupting the pooled calibration reference with
# other clients' permanent zeros) was the other candidate fix discussed but is
# NOT implemented -- this result suggests it's not obviously worth building
# either, since the columns' problem was never really "noise from dead
# channels," it was that a DIFFERENT confound wins in their absence regardless.
# See memory/sielaff-diagnosis-localization.md for the full writeup.
#
# Default is OFF (keep all 39 columns) given this result. `--drop-heterogeneous-hardware`
# is kept as an opt-in flag for reproducing the negative finding, not recommended.
HETEROGENEOUS_HARDWARE_COLS = (
    [f"read SM_RingCamera {n}" for n in range(1, 7)]
    + [f"read barcode_RingCamera {n}" for n in range(1, 7)]
    + [f"schließen_Weiche {n}" for n in range(3, 8)]
    + [f"öffnen_Weiche {n}" for n in range(3, 8)]
)
assert len(HETEROGENEOUS_HARDWARE_COLS) == 22

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15
BATCH_SIZE = 256
ROUNDS = 5
LOCAL_EPOCHS = 30
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 16
WINDOW_LEN = 8
STRIDE = 2
TOP_K = 8
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_FORECAST = 0.5
GAMMA = 0.5
DELTA = 0.5
RELIABILITY_RATIO = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- ground truth --
# Verbatim from analyze_sielaff_red_feature_attribution.py.
def feature_domain(feat):
    if feat == "Helligkeit_Flaschenerkennung":
        return "bottle_recognition_sensor"
    if feat.startswith("read SM_RingCamera") or feat.startswith("read barcode_RingCamera"):
        return "ring_camera_read"
    if feat.startswith("schließen_Weiche") or feat.startswith("öffnen_Weiche"):
        return "sorting_gate_switch"
    if feat.startswith("cleaning_"):
        return "cleaning_cycle"
    if feat in ("journal_count", "total_quantity", "total_deposit", "total_weight", "unique_categories"):
        return "transaction_throughput"
    if feat in ("reject_count", "has_reject"):
        return "reject_stream"
    if feat == "receipt_count":
        return "receipt_printing"
    return "other"


FAULT_EXPECTED_DOMAIN = [
    (("bottle_out_of_range", "bottle_direction_wrong", "bottle_collision", "last_ls_passed"),
     {"ring_camera_read", "bottle_recognition_sensor", "sorting_gate_switch"}),
    (("label_movement",), {"ring_camera_read"}),
    (("crate_bottle_wrong",), {"transaction_throughput", "reject_stream"}),
    (("printer", "receipt"), {"receipt_printing"}),
    (("cleaning",), {"cleaning_cycle"}),
    (("bottle_compacted",), {"transaction_throughput", "sorting_gate_switch"}),
]
NO_SENSOR_KEYWORDS = ("compactor", "crate_", "safety_circuit", "dooropen", "misc_sw_restart",
                      "misc_pc_reboot", "bottle_fraud", "crate_fraud")


def expected_domain_for(red_name):
    """Returns (expected_domain_set_or_None, status) where status is one of
    'mapped' (has an expected-domain rule), 'no_direct_sensor', 'unmapped'."""
    for keywords, expected in FAULT_EXPECTED_DOMAIN:
        if any(k in red_name for k in keywords):
            return expected, "mapped"
    if any(k in red_name for k in NO_SENSOR_KEYWORDS):
        return None, "no_direct_sensor"
    return None, "unmapped"


# --------------------------------------------------------------------- data --
class Client:
    def __init__(self, client_id, all_idx, fit_idx, calib_idx, test_normal_idx, red_idx):
        self.client_id = client_id
        self.all_idx = all_idx
        self.fit_idx = fit_idx
        self.calib_idx = calib_idx
        self.test_normal_idx = test_normal_idx
        self.red_idx = red_idx


def build_clients(drop_heterogeneous_hardware=False):
    data = np.load(DATA_DIR / "data.npy")
    targets = np.load(DATA_DIR / "targets.npy")
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    cols = meta["feature_columns"]
    if drop_heterogeneous_hardware:
        keep_idx = [i for i, c in enumerate(cols) if c not in HETEROGENEOUS_HARDWARE_COLS]
        data = data[:, :, keep_idx]
        cols = [cols[i] for i in keep_idx]
    red_id_names = {int(k): v for k, v in meta["red_id_names"].items()}
    with open(DATA_DIR / "window_red_ids.json") as f:
        window_red_ids = json.load(f)
    with open(DATA_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)

    window_machine = np.full(len(targets), -1, dtype=int)
    clients = []
    for client_id in range(partition["separation"]["total"]):
        d = partition["data_indices"][client_id]
        idx = np.sort(np.concatenate([d["train"], d["val"], d["test"]]).astype(np.int64))
        window_machine[idx] = client_id
        normal_idx = idx[targets[idx] == 0]
        n = len(normal_idx)
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        red_idx = idx[targets[idx] == 1]
        clients.append(Client(
            client_id=client_id, all_idx=idx,
            fit_idx=normal_idx[:n_fit], calib_idx=normal_idx[n_fit : n_fit + n_calib],
            test_normal_idx=normal_idx[n_fit + n_calib :], red_idx=red_idx,
        ))
    return data, targets, cols, red_id_names, window_red_ids, clients, window_machine


def build_pairs(idx_arr, targets, window_machine, horizon_mult, require_next_normal, require_same_split=False):
    N = len(targets)
    idx_set = set(idx_arr.tolist()) if require_same_split else None
    vi, chains, mask = [], [], []
    for i in idx_arr:
        chain = [i + k for k in range(1, horizon_mult + 1)]
        valid = True
        for j in chain:
            if (j >= N or window_machine[j] != window_machine[i]
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


def h_contributions(model, d_node, idx):
    cov_head = model.cov_head
    valid = cov_head.calib_valid.cpu().numpy()[idx]
    mu = np.where(valid[:, None], cov_head.calib_mu.cpu().numpy()[idx],
                  cov_head.global_mu.cpu().numpy()[None, :])
    cov_inv = np.where(valid[:, None, None], cov_head.calib_cov_inv.cpu().numpy()[idx],
                        cov_head.global_cov_inv.cpu().numpy()[None, :, :])
    diff = d_node - mu
    weighted = np.einsum("bij,bj->bi", cov_inv, diff)
    return diff * weighted


def zn(x, med, iqr):
    return (x - med) / iqr


def masked_argmax(z, reliable):
    z_masked = np.where(reliable[None, :], z, -np.inf)
    return np.argmax(z_masked, axis=1)


# Activity-level normalization: the confound-exclusion fixes above (excluding
# `transaction_throughput`, then discovering `receipt_count`/`reject_count`/
# `cleaning_duration` dominate in its absence) only ever excluded specific
# feature DOMAINS one at a time. The underlying problem is structural: Sielaff's
# 30-min-bucketed event-log features are all partially driven by a shared
# "how busy was the machine this window" latent factor -- ANY window with real
# transaction activity elevates journal/receipt/reject/cleaning counts together,
# regardless of which specific fault occurred. Rather than keep excluding
# confound domains one at a time, condition each node's deviation on an
# explicit activity-level covariate and score the RESIDUAL -- the same
# "explained-by-formula, score what's left" idea `kinematics.py` uses for
# robo3er's wheel/odometry residual, applied post-hoc to d_node/resid_struct/
# k_resid instead of pre-hoc to raw features (avoids retraining the encoder).
# Scoped to B/C/K only -- H's Mahalanobis mechanism has its own, different,
# covariance-based defense against correlated confounds (inverse-covariance
# reweighting suppresses shared-variance directions) and is left on raw d_node
# as a natural comparison: does explicit normalization actually beat H's
# built-in mechanism, or is H already doing this implicitly?
ACTIVITY_PROXY_COLS = ["journal_count", "reject_count", "cleaning_count", "receipt_count"]


def compute_activity_level(raw_windows, cols):
    """raw_windows: [N, T, num_nodes] RAW (unscaled) window data, same row
    order as the scaled/scored array it will be paired with. Returns [N]
    scalar activity level per window (total logged-event count across the
    window, summed over the 4 core event-count columns)."""
    idx = [cols.index(c) for c in ACTIVITY_PROXY_COLS if c in cols]
    if not idx or len(raw_windows) == 0:
        return np.zeros(len(raw_windows), dtype=np.float32)
    return raw_windows[:, :, idx].sum(axis=(1, 2)).astype(np.float32)


def fit_activity_regression(signal, activity_level):
    """signal: [N, D]. Per-dimension OLS signal_d ~ a_d*activity + b_d.
    Returns coef [2, D] (slope, intercept)."""
    A = np.stack([activity_level, np.ones_like(activity_level)], axis=1)
    coef, *_ = np.linalg.lstsq(A, signal, rcond=None)
    return coef


def residualize(signal, activity_level, coef):
    pred = coef[0][None, :] * activity_level[:, None] + coef[1][None, :]
    return signal - pred


# `total_weight`/`total_deposit`/`journal_count`/etc. ("transaction_throughput"
# domain) dominate the RAW argmax for almost every single red ID regardless of
# that ID's true mechanism -- verified empirically in this run (see the
# script's own printed top_nodes: `total_weight` wins >90% of votes on faults
# whose expected domain is ring_camera/bottle_recognition/sorting_gate, which
# don't even contain it). Same activity/exposure confound
# `analyze_sielaff_red_feature_attribution.py` already documented and handled
# via a confound-excluded ranking: a genuinely no_error window is
# disproportionately an IDLE window (no transactions -> no throughput), so ANY
# window with real activity looks anomalous on these features regardless of
# WHICH fault occurred. Excluding this domain from the argmax candidate pool
# (on top of the existing reliability mask) gives the more mechanism-
# informative number; the raw (confound-INCLUDED) number is still reported
# alongside it, not discarded.
CONFOUND_DOMAIN = "transaction_throughput"


def exclude_confound(reliable, cols):
    non_confound = np.array([feature_domain(c) != CONFOUND_DOMAIN for c in cols])
    return reliable & non_confound


def main(horizon_mult=10, rounds=None, local_epochs=None, out_suffix=None, drop_heterogeneous_hardware=False,
         activity_normalize=True):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    forecast_h = STRIDE * horizon_mult
    rounds = rounds if rounds is not None else ROUNDS
    local_epochs = local_epochs if local_epochs is not None else LOCAL_EPOCHS

    print(f"loading Sielaff RED (47 red-severity error IDs = the fault types), "
          f"10 per-machine federated clients, horizon_mult={horizon_mult}, "
          f"rounds={rounds}, local_epochs={local_epochs}, "
          f"drop_heterogeneous_hardware={drop_heterogeneous_hardware}, "
          f"activity_normalize={activity_normalize} ...")
    data, targets, cols, red_id_names, window_red_ids, clients, window_machine = build_clients(
        drop_heterogeneous_hardware=drop_heterogeneous_hardware)
    num_nodes = len(cols)
    print(f"  num_nodes={num_nodes}: {cols}")
    for c in clients:
        print(f"  client {c.client_id}: fit={len(c.fit_idx)} calib={len(c.calib_idx)} "
              f"test_normal={len(c.test_normal_idx)} red={len(c.red_idx)}")

    scalers = [fit_client_scaler(data, c.fit_idx) for c in clients]
    models = [JointPrototypeV21Forecast(num_nodes=num_nodes, window_size=WINDOW_LEN,
                                         embed_dim=EMBED_DIM, num_prototypes=NUM_PROTOTYPES,
                                         forecast_h=forecast_h, prior_edges=None, top_k=TOP_K).to(DEVICE)
              for _ in clients]

    shared_init = {k: v.clone() for k, v in models[0].state_dict().items() if not k.startswith("memory.")}
    for model in models[1:]:
        model.load_state_dict(shared_init, strict=False)

    fit_pairs = []
    for c in clients:
        vi, chains, _ = build_pairs(c.fit_idx, targets, window_machine, horizon_mult, True, True)
        fit_pairs.append((vi, data[vi], gather_future(data, chains, STRIDE)))

    print(f"\nfederated training: {rounds} rounds x {local_epochs} local epochs, memory-only exchange ...")
    for rnd in range(1, rounds + 1):
        for c, model, scaler, (vi, x_in_raw, x_future_raw) in zip(clients, models, scalers, fit_pairs):
            n, t, f = x_in_raw.shape
            x_in = scaler.transform(x_in_raw.reshape(-1, f)).reshape(n, t, f).astype(np.float32) if n else x_in_raw
            n2, t2, f2 = x_future_raw.shape
            x_future = (scaler.transform(x_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                        if n2 > 0 else x_future_raw)
            if n:
                train_local(model, x_in, x_future, local_epochs)

        codebooks = [m.memory.codebook.detach().clone() for m in models]
        usage_counts = [m.memory.usage_count.detach().clone() for m in models]
        P_G, diag = align_and_split(codebooks, usage_counts, gamma=GAMMA, delta=DELTA)
        print(f"  round {rnd}: shared_clusters={diag.get('num_multi_client_clusters')}")
        for model, p_g in zip(models, P_G):
            model.memory.load_memory(p_g)

    print("\ncalibrating cov_head per client, then POOLING z-score reference across all clients "
          "(per-client calib is only ~6-9 windows here -- see script docstring) ...")
    cov_min = max(NUM_PROTOTYPES, num_nodes + 1)
    per_client = []
    d_node_pool, resid_struct_pool, d_mahal_pool, k_resid_pool = [], [], [], []
    activity_node_pool, activity_k_pool = [], []
    for c, model, scaler in zip(clients, models, scalers):
        calib_w = scale_client(data, scaler, c.calib_idx)
        normal_w = scale_client(data, scaler, c.test_normal_idx)
        pool_w = np.concatenate([calib_w, normal_w], axis=0) if len(calib_w) or len(normal_w) else calib_w
        if len(pool_w) == 0:
            per_client.append((c, model, scaler, None))
            continue
        _, d_node_pw, resid_struct_pw, idx_pw, _ = per_sample_scores(model, pool_w)
        model.cov_head.set_calibration(d_node_pw, idx_pw, min_samples=cov_min)
        _, d_node_c, resid_struct_c, idx_c, d_mahal_c = per_sample_scores(model, pool_w)

        pool_idx = np.concatenate([c.calib_idx, c.test_normal_idx])
        activity_node_pool.append(compute_activity_level(data[pool_idx], cols))
        vi, chains, _ = build_pairs(pool_idx, targets, window_machine, horizon_mult, True, False)
        if len(vi):
            k_in = scale_client(data, scaler, vi)
            k_future_raw = gather_future(data, chains, STRIDE)
            n2, t2, f2 = k_future_raw.shape
            k_future = scaler.transform(k_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
            k_resid_c = forecast_scores(model, k_in, k_future)
            activity_k_pool.append(compute_activity_level(data[vi], cols))
        else:
            k_resid_c = np.zeros((0, num_nodes), dtype=np.float32)

        d_node_pool.append(d_node_c)
        resid_struct_pool.append(resid_struct_c)
        d_mahal_pool.append(d_mahal_c)
        if len(k_resid_c):
            k_resid_pool.append(k_resid_c)
        per_client.append((c, model, scaler, True))

    d_node_calib = np.concatenate(d_node_pool, axis=0)
    resid_struct_calib = np.concatenate(resid_struct_pool, axis=0)
    d_mahal_calib = np.concatenate(d_mahal_pool, axis=0)
    k_resid_calib = np.concatenate(k_resid_pool, axis=0) if k_resid_pool else np.zeros((0, num_nodes))
    activity_node_calib = np.concatenate(activity_node_pool, axis=0)
    activity_k_calib = np.concatenate(activity_k_pool, axis=0) if activity_k_pool else np.zeros(0, dtype=np.float32)

    if activity_normalize:
        print(f"  activity-normalizing B/C/K against {ACTIVITY_PROXY_COLS} (H left on raw d_node)")
        coef_node = fit_activity_regression(d_node_calib, activity_node_calib)
        coef_struct = fit_activity_regression(resid_struct_calib, activity_node_calib)
        d_node_calib = residualize(d_node_calib, activity_node_calib, coef_node)
        resid_struct_calib = residualize(resid_struct_calib, activity_node_calib, coef_struct)
        if len(k_resid_calib):
            coef_k = fit_activity_regression(k_resid_calib, activity_k_calib)
            k_resid_calib = residualize(k_resid_calib, activity_k_calib, coef_k)
        else:
            coef_k = None
    else:
        coef_node = coef_struct = coef_k = None

    node_median = np.median(d_node_calib, axis=0)
    node_q75, node_q25 = np.percentile(d_node_calib, [75, 25], axis=0)
    node_iqr = np.maximum(node_q75 - node_q25, 1e-8)
    struct_median = np.median(resid_struct_calib, axis=0)
    struct_q75, struct_q25 = np.percentile(resid_struct_calib, [75, 25], axis=0)
    struct_iqr = np.maximum(struct_q75 - struct_q25, 1e-8)
    if len(k_resid_calib):
        k_median = np.median(k_resid_calib, axis=0)
        k_q75, k_q25 = np.percentile(k_resid_calib, [75, 25], axis=0)
        k_iqr = np.maximum(k_q75 - k_q25, 1e-8)
    else:
        k_median, k_iqr = np.zeros(num_nodes), np.ones(num_nodes)
    node_reliable = node_iqr >= RELIABILITY_RATIO * np.median(node_iqr)
    struct_reliable = struct_iqr >= RELIABILITY_RATIO * np.median(struct_iqr)
    k_reliable = k_iqr >= RELIABILITY_RATIO * np.median(k_iqr)
    node_reliable_excl = exclude_confound(node_reliable, cols)
    struct_reliable_excl = exclude_confound(struct_reliable, cols)
    k_reliable_excl = exclude_confound(k_reliable, cols)
    d_mahal_med = np.median(d_mahal_calib)
    d_mahal_iqr = max(np.percentile(d_mahal_calib, 75) - np.percentile(d_mahal_calib, 25), 1e-8)
    print(f"  pooled calib n={len(d_node_calib)}, reliable B/H={node_reliable.sum()}/{num_nodes}  "
          f"C={struct_reliable.sum()}/{num_nodes}  K={k_reliable.sum()}/{num_nodes}  "
          f"(confound-excluded: B/H={node_reliable_excl.sum()} C={struct_reliable_excl.sum()} K={k_reliable_excl.sum()})")

    print("\nscoring red windows, attributing each window's argmax to every red ID it contains "
          "(RAW and confound-excluded variants) ...")
    votes = {}       # signal -> red_id -> Counter(node_name), RAW (confound-included)
    votes_excl = {}  # same, with transaction_throughput excluded from the argmax pool
    n_windows_per_id = Counter()
    for key in ("B", "C", "H", "K", "BK", "CK", "HK"):
        votes[key] = {}
        votes_excl[key] = {}

    for c, model, scaler, ok in per_client:
        if not ok or len(c.red_idx) == 0:
            continue
        red_w = scale_client(data, scaler, c.red_idx)
        _, d_node_f, resid_struct_f, idx_f, d_mahal_f = per_sample_scores(model, red_w)
        contrib_h_f = h_contributions(model, d_node_f, idx_f)  # H: always on RAW d_node
        z_mahal_f = zn(d_mahal_f, d_mahal_med, d_mahal_iqr)

        if activity_normalize:
            activity_f = compute_activity_level(data[c.red_idx], cols)
            d_node_f_scored = residualize(d_node_f, activity_f, coef_node)
            resid_struct_f_scored = residualize(resid_struct_f, activity_f, coef_struct)
        else:
            d_node_f_scored, resid_struct_f_scored = d_node_f, resid_struct_f
        z_node_f = zn(d_node_f_scored, node_median, node_iqr)
        z_struct_f = zn(resid_struct_f_scored, struct_median, struct_iqr)

        argmax_b = masked_argmax(z_node_f, node_reliable)
        argmax_c = masked_argmax(z_struct_f, struct_reliable)
        argmax_h = masked_argmax(contrib_h_f, node_reliable)
        argmax_b_excl = masked_argmax(z_node_f, node_reliable_excl)
        argmax_c_excl = masked_argmax(z_struct_f, struct_reliable_excl)
        argmax_h_excl = masked_argmax(contrib_h_f, node_reliable_excl)

        vi_f, chains_f, mask_f = build_pairs(c.red_idx, targets, window_machine, horizon_mult, False, False)
        if len(vi_f):
            fault_in = scale_client(data, scaler, vi_f)
            future_f_raw = gather_future(data, chains_f, STRIDE)
            n2, t2, f2 = future_f_raw.shape
            fault_future = scaler.transform(future_f_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
            k_resid_f = forecast_scores(model, fault_in, fault_future)
            if activity_normalize and coef_k is not None:
                activity_k_f = compute_activity_level(data[vi_f], cols)
                k_resid_f = residualize(k_resid_f, activity_k_f, coef_k)
            z_k_f = zn(k_resid_f, k_median, k_iqr)
            argmax_k = masked_argmax(z_k_f, k_reliable)
            argmax_k_excl = masked_argmax(z_k_f, k_reliable_excl)
        else:
            z_k_f = np.zeros((0, num_nodes))
            argmax_k = np.zeros(0, dtype=int)
            argmax_k_excl = np.zeros(0, dtype=int)

        b_paired = z_node_f.max(axis=1)[mask_f]
        c_paired = z_struct_f.max(axis=1)[mask_f]
        h_paired = z_mahal_f[mask_f]
        k_scalar = z_k_f.max(axis=1) if len(z_k_f) else np.zeros(0)
        argmax_b_p, argmax_c_p, argmax_h_p = argmax_b[mask_f], argmax_c[mask_f], argmax_h[mask_f]
        argmax_b_p_excl, argmax_c_p_excl, argmax_h_p_excl = (
            argmax_b_excl[mask_f], argmax_c_excl[mask_f], argmax_h_excl[mask_f])

        def combo(scalar_a, argmax_a, scalar_k, argmax_k_):
            if len(scalar_a) == 0:
                return np.zeros(0, dtype=int)
            a_wins = scalar_a >= scalar_k
            return np.where(a_wins, argmax_a, argmax_k_)

        argmax_bk = combo(b_paired, argmax_b_p, k_scalar, argmax_k)
        argmax_ck = combo(c_paired, argmax_c_p, k_scalar, argmax_k)
        argmax_hk = combo(h_paired, argmax_h_p, k_scalar, argmax_k)
        argmax_bk_excl = combo(b_paired, argmax_b_p_excl, k_scalar, argmax_k_excl)
        argmax_ck_excl = combo(c_paired, argmax_c_p_excl, k_scalar, argmax_k_excl)
        argmax_hk_excl = combo(h_paired, argmax_h_p_excl, k_scalar, argmax_k_excl)

        per_win = {"B": argmax_b, "C": argmax_c, "H": argmax_h}
        per_win_paired = {"K": argmax_k, "BK": argmax_bk, "CK": argmax_ck, "HK": argmax_hk}
        per_win_excl = {"B": argmax_b_excl, "C": argmax_c_excl, "H": argmax_h_excl}
        per_win_paired_excl = {"K": argmax_k_excl, "BK": argmax_bk_excl, "CK": argmax_ck_excl, "HK": argmax_hk_excl}

        for local_row, global_win_idx in enumerate(c.red_idx):
            red_ids_here = window_red_ids[global_win_idx]
            if not red_ids_here:
                continue
            for rid in red_ids_here:
                n_windows_per_id[rid] += 1
                for key, arr in per_win.items():
                    votes[key].setdefault(rid, Counter())[cols[arr[local_row]]] += 1
                for key, arr in per_win_excl.items():
                    votes_excl[key].setdefault(rid, Counter())[cols[arr[local_row]]] += 1

        # paired (K-derived) votes: mask_f tells us which red_idx rows survived pairing
        paired_global_idx = c.red_idx[mask_f]
        for local_row, global_win_idx in enumerate(paired_global_idx):
            red_ids_here = window_red_ids[global_win_idx]
            for rid in red_ids_here:
                for key, arr in per_win_paired.items():
                    votes[key].setdefault(rid, Counter())[cols[arr[local_row]]] += 1
                for key, arr in per_win_paired_excl.items():
                    votes_excl[key].setdefault(rid, Counter())[cols[arr[local_row]]] += 1

    def direct_pct_for(counter, expected):
        total = sum(counter.values())
        if not total:
            return total, None
        direct = sum(v for feat, v in counter.items() if feature_domain(feat) in expected)
        return total, round(100.0 * direct / total, 1)

    results = {}
    print(f"\n{'red_id':<8}{'name':<32}{'status':<16}{'n':<6}"
          + "".join(f"{k+'_raw%':<12}{k+'_excl%':<12}" for k in ("B", "C", "H", "K", "BK", "CK", "HK")))
    for rid in sorted(n_windows_per_id, key=lambda r: -n_windows_per_id[r]):
        name = red_id_names.get(rid, f"id_{rid}")
        expected, status = expected_domain_for(name)
        n = n_windows_per_id[rid]
        row = {"name": name, "status": status, "n": n, "signals": {}}
        line = f"{rid:<8}{name:<32}{status:<16}{n:<6}"
        for key in ("B", "C", "H", "K", "BK", "CK", "HK"):
            counter = votes[key].get(rid, Counter())
            counter_excl = votes_excl[key].get(rid, Counter())
            if status == "mapped":
                total, direct_pct = direct_pct_for(counter, expected)
                total_excl, direct_pct_excl = direct_pct_for(counter_excl, expected)
            else:
                total, direct_pct = sum(counter.values()), None
                total_excl, direct_pct_excl = sum(counter_excl.values()), None
            row["signals"][key] = {
                "n_votes": total, "direct_pct": direct_pct, "top_nodes": counter.most_common(3),
                "direct_pct_confound_excluded": direct_pct_excl,
                "top_nodes_confound_excluded": counter_excl.most_common(3),
            }
            line += (f"{(str(direct_pct) if direct_pct is not None else '-'):<12}"
                      f"{(str(direct_pct_excl) if direct_pct_excl is not None else '-'):<12}")
        print(line)
        results[str(rid)] = row

    # Aggregate: pooled direct% across only the "mapped" (diagnosable) red IDs, per signal
    def aggregate_for(vote_dict):
        aggregate = {}
        for key in ("B", "C", "H", "K", "BK", "CK", "HK"):
            total_votes, total_direct = 0, 0
            for rid, row in results.items():
                if row["status"] != "mapped":
                    continue
                expected, _ = expected_domain_for(row["name"])
                counter = vote_dict[key].get(int(rid), Counter())
                for feat, v in counter.items():
                    total_votes += v
                    if feature_domain(feat) in expected:
                        total_direct += v
            aggregate[key] = {
                "n_votes": total_votes,
                "direct_pct": round(100.0 * total_direct / total_votes, 1) if total_votes else None,
            }
        return aggregate

    aggregate = aggregate_for(votes)
    aggregate_excl = aggregate_for(votes_excl)
    print("\naggregate direct% across all 'mapped' (diagnosable) red IDs, pooled votes -- RAW vs. confound-excluded:")
    for key in aggregate:
        print(f"  {key:<6}raw: n={aggregate[key]['n_votes']:<6}direct%={aggregate[key]['direct_pct']:<8}"
              f"confound-excluded: n={aggregate_excl[key]['n_votes']:<6}direct%={aggregate_excl[key]['direct_pct']}")

    out = {
        "_config": {"num_nodes": num_nodes, "cols": cols, "horizon_mult": horizon_mult,
                    "rounds": rounds, "local_epochs": local_epochs, "reliability_ratio": RELIABILITY_RATIO,
                    "pooled_calib_n": len(d_node_calib), "confound_domain_excluded": CONFOUND_DOMAIN,
                    "drop_heterogeneous_hardware": drop_heterogeneous_hardware,
                    "activity_normalize": activity_normalize, "activity_proxy_cols": ACTIVITY_PROXY_COLS},
        "per_red_id": results,
        "aggregate_over_mapped_ids_raw": aggregate,
        "aggregate_over_mapped_ids_confound_excluded": aggregate_excl,
    }
    default_suffix = (f"h{horizon_mult}" + ("_17node" if drop_heterogeneous_hardware else "_39node")
                       + ("_actnorm" if activity_normalize else ""))
    suffix = out_suffix if out_suffix is not None else default_suffix
    out_path = OUT_DIR / f"sielaff_red_diagnosis_localization_federated_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-mult", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--out-suffix", type=str, default=None)
    parser.add_argument("--drop-heterogeneous-hardware", action="store_true",
                         help="drop the 22 hardware-heterogeneous columns (RingCamera/gates 3-7) -- "
                              "tested and found to HURT K/BK/CK's diagnosis accuracy, opt-in only, "
                              "kept for reproducing that negative finding, not recommended")
    parser.add_argument("--no-activity-normalize", action="store_true",
                         help="disable activity-level residualization of B/C/K (on by default) -- "
                              "pass this to reproduce the pre-normalization confounded numbers")
    args = parser.parse_args()
    main(horizon_mult=args.horizon_mult, rounds=args.rounds, local_epochs=args.local_epochs,
         out_suffix=args.out_suffix, drop_heterogeneous_hardware=args.drop_heterogeneous_hardware,
         activity_normalize=not args.no_activity_normalize)
