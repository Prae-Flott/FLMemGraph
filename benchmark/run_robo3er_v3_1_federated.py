#!/usr/bin/env python3
"""
FEDERATED JointPrototypeV31 (V3, and V2 read off its B_node_max
ablation column -- see `memory/scoring-signals-B-C-E-H.md`) on robo3er's 5
real robots as 5 federated clients, following the exact per-round protocol
already validated for the older, since-deleted `FLGDNMemory` architecture:
encoder + edge_head + typed_head stay
fully local per client, ONLY the `JointPrototypeMemory` codebook is exchanged
each round via `federated_memory.align_and_split` (now node-aware -- see that
module's docstring -- rather than flattening the [M, N, D] joint-prototype
codebook before computing similarity).

Declared physics edges/types and the client split are shared with the
centralized `run_robo3er_joint_prototype_v3.py` (robo3er's 5 robots,
`data/robo3er/partition.pkl`, loaded directly from `load_robo3er()`, NOT
through `fl_dataset.load_fl_clients` since that helper also applies a
per-client KINEMATIC RESIDUAL transform V3 doesn't want). Node count and
prototype count DELIBERATELY DIVERGE from the centralized script, though
-- see `memory/joint-prototype-federated-results.md`'s node-count-vs-
federated-capacity follow-up: the centralized script's full 68-node/16-
prototype config gives smallest-client robot01 (103 fit windows) a
~1194-parameters-per-fit-sample ratio for the memory codebook alone (vs.
~38 for the largest client/pool), which measurably hurt federated results
(round-5 shared-cluster count dropped from 5 to 1 vs. the original
7-node config, `B_node_max` dropped 0.686->0.431 with encoder sync).
This script instead uses `KINEMATIC_CORE + ACTUATION` (42 nodes -- still
6x the original 7, keeps all 7 declared edges' endpoints, drops only
`STATUS_FLAGS`/`ENVIRONMENT`, physically unrelated to the robot's own
equations of motion per `feature_groups.py`'s own docstring) and
`NUM_PROTOTYPES=8` (halved from 16), bringing robot01's ratio down to
~711 -- a federated-capacity-aware compromise, not a claim that 42/8 is
somehow "more correct" than 68/16 in general; the centralized script
keeps 68/16 since centralized pooling doesn't have this problem.

Per-client evaluation (this dataset's fault types are inherently
per-robot -- robot02 has none, robot04 alone has 100% of `stuck`, see
`fl_dataset.py`'s module docstring): each client's calib split calibrates
its OWN z-scores after the final round (matches
`deployment-architecture.md`'s "share weights, calibrate locally"), then
scores its own fault windows. `summary_mean_auroc_overall` averages across
ALL (client, fault_type) pairs (unweighted), the federated-setting analogue
of the centralized script's flat mean-over-fault-types number.

## Fault ground truth (see `benchmark/datasets/robo3er_physics.md`)

- `stuck` (robot04, 149 windows): cable slowly wraps around ONE wheel
  axle → that wheel's current GRADUALLY increases, opposite wheel stays
  normal → wheel locks, robot stops. Key signal: left/right current
  ASYMMETRY; the covariance head (H) detects this best (0.946) because
  the asymmetric joint-deviation pattern is unlike any normal regime's
  co-deviation structure, even though no single node is obviously extreme.
- `Broken Pipe` (robot00/01, ~59 windows): ROS2 clock-sync failure with
  the robot base → ALL base-published topics freeze simultaneously →
  recover when clock re-syncs. Detected by d_node global freeze pattern.
- `cable_trapped` (robot01, 206 windows): cables pinched under chassis
  → robot spins/rocks continuously. High yaw rate variance; kinematic
  relations (wheel→odom) are systematically violated. Most signals detect
  this well.
- `Low battery` (robot00, 74 windows): battery < 30% → motor efficiency
  drops, may trigger soft speed limiting. Direct battery channels
  (battery_state_*) EXCLUDED from feature set (cumulative/non-stationary);
  indirect signal via elevated wheel current only when actively driving.

Usage:
    python3 run_robo3er_v3_1_federated.py [--calib-mode {global,per_prototype,ema,shrinkage}] [--alpha 20]

`--calib-mode` (default `global`, UNCHANGED behavior, regression-safe):
per `memory/calib-in-prototype-ab.md`'s branch experiment closing the gap
between how H/E (per-prototype calib, model buffers) and B/C (global
median/IQR, script-level) are calibrated.
- `global`: exactly today's behavior -- one script-level median/IQR over
  the whole calib split, not tied to prototypes.
- `per_prototype`: Path B -- `JointPrototypeV31.score_calib_node`/
  `score_calib_struct` (new `ScoreCalibrationHead`s in
  `joint_prototype_model.py`), fit from the SAME calib split as
  `cov_head`/`typed_head`, same min_samples-fallback-to-global pattern.
- `ema`: Path A -- `JointPrototypeMemory.update_ema()` accumulates
  per-prototype mean/var of `d_node` ONLINE during local training
  (VQ-VAE-codebook-EMA style, with a warm-up gate), used instead of an
  offline calib pass. `resid_struct` (signal C) has no online EMA hook in
  this script (only `d_node`/B, per the task's stated priority) --
  `ema` mode falls back to `global` for C.
- `shrinkage`: empirical-Bayes per-client, per-prototype shrinkage, the
  successor to both plain `ema` (pure local stats -- robo3er's small clients
  never clear warm-up within a round) and the reverted `federated_ema`
  experiment (one fused stat shared identically by every client in a
  cluster, which let big clients like robot04 overwrite small ones' real
  statistics -- see `memory/calib-in-prototype-ab.md`'s "Why federated_ema
  failed" section). Each round, AFTER `align_and_split`+`load_memory()`,
  every client does a one-shot local pass over its own fit split under
  that round's just-received aligned codebook
  (`federated_memory.compute_prototype_dev_stats`); the server pools these
  fleet-wide per shared prototype slot (`compute_shrinkage_stats`) and
  blends each client's own local statistic with the fleet prior via
  `lambda_{c,k} = n_{c,k} / (n_{c,k} + alpha)` -- large clients (`n` large)
  keep almost all of their own statistic, small clients borrow heavily from
  the fleet but are never fully overwritten. `--alpha` controls shrinkage
  strength (larger = more pooling); swept across scripts/datasets, not
  tuned per-script. `resid_struct`/C has no shrinkage hook either (same
  scope limit as `ema`), falls back to `global`.
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
sys.path.insert(0, str(REPO_ROOT / "src" / "robo3er"))
sys.path.insert(0, str(REPO_ROOT / "src" / "federated"))
from joint_prototype_model import JointPrototypeV31  # noqa: E402
from dataset import load_robo3er  # noqa: E402
import feature_groups  # noqa: E402
from federated_memory import (align_and_split, fedavg_state_dict,  # noqa: E402
                               compute_prototype_dev_stats, compute_shrinkage_stats)

SYNC_ENCODER_DECODER = True  # opt-in departure from "only exchange memory" -- see
                              # federated_memory.fedavg_state_dict's docstring and
                              # memory/joint-prototype-federated-results.md

DATA_DIR = REPO_ROOT / "data" / "robo3er"
OUT_DIR = REPO_ROOT / "checkpoints" / "robo3er"

# kinematic_core + actuation (42 nodes) used as nodes -- see
# run_robo3er_joint_prototype_v3.py's module docstring for why; node_idx is
# built from the live `cols` list in build_clients() below, not a fixed name list.
EDGES_NAMED = [
    ("wheel_vels_velocity_left", "odom_odo_lintw_x"),
    ("wheel_vels_velocity_right", "odom_odo_lintw_x"),
    ("wheel_vels_velocity_left", "odom_odo_angtw_z"),
    ("wheel_vels_velocity_right", "odom_odo_angtw_z"),
    ("odom_odo_angtw_z", "imu_imu_angvel_z"),
    ("wheel_status_current_ma_left", "wheel_vels_velocity_left"),
    ("wheel_status_current_ma_right", "wheel_vels_velocity_right"),
]
EDGE_TYPES = [
    "proportional", "proportional", "proportional", "proportional",
    "proportional",
    "nonlinear", "nonlinear",
]

WINDOW_LEN = 60
BATCH_SIZE = 256
ROUNDS = 5
LOCAL_EPOCHS = 12  # == centralized EPOCHS, so total local training exposure per client
                    # is comparable dataset-pass-count-wise to the centralized run
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 2  # see module docstring -- best of {1,2,4,8} tested at 42 nodes
TOP_K = 8
FEATURE_GROUPS = ["kinematic_core", "actuation"]
# 11 tf_link_base_link_*/tf_footprint_base_footprint_* columns dropped from that 42 -- measured
# (50k-row sample, pooled across all 5 robots) near-duplicates of odom_odo_pos/orient already
# kept: e.g. r=0.998 tf_link_rot_z<->tf_footprint_rot_z, r=0.995 odom_orient_x<->tf_link_rot_x,
# r=0.98 odom_orient_w<->tf_link/footprint_rot_w -- matches feature_groups.py's own docstring,
# which already calls these "a software-broadcast re-expression of the SAME odometry estimate
# in a different frame... not an independent sensor... not a physical fault signal." Declared
# edges/cross-check pairs (current<->wheel, wheel<->odom, odom_ang<->imu_angvel) are NOT touched
# by this -- those are also highly correlated under normal operation, but that correlation IS the
# fault signal (should stay correlated; a fault is exactly when it stops), unlike tf-vs-odom which
# is pure redundancy with no independent information content either way.
EXCLUDE_NODES_PREFIXES = ("tf_link_base_link_", "tf_footprint_base_footprint_",  # redundant w/ odom, see above
                           "odom_odo_pos_", "wheel_ticks_")  # -> 26 nodes. These two are CUMULATIVE/
# UNBOUNDED quantities (chassis position, wheel tick count), unlike every other kept node
# (velocity/orientation/current/PWM), which are roughly stationary/bounded. A joint-prototype
# memory matches windows to a small fixed set of "operating regimes" -- position doesn't have
# regimes in that sense (two different rooms of the building aren't different physical states,
# just different places), so including it adds a slowly-drifting dimension prototype matching
# can never stably cover, only ever chase. Measured effect (encoder+decoder synced): B/C/F all
# improve over the 31-node config (e.g. B_node_max 0.638->0.750); ONLY F_v3_max gets worse
# (0.712->0.673), because F's plain max() still folds in the already-known-weak typed-edge signal
# (E), which the position/ticks removal doesn't touch -- B_node_max (excludes E entirely) is the
# more honest "how good is V2 now" number. NOT a universal win, though: the SAME removal, WITHOUT
# encoder/decoder sync, makes B worse (0.598 at 31 nodes -> 0.503 at 26) -- this
# specific reduction is beneficial conditional on encoder sync being on (`SYNC_ENCODER_DECODER`,
# the recommended default), not an unconditional improvement.
INCLUDE_EXTRA_NODES = [
    # battery: the direct diagnostic source for the "Low battery" fault type -- dropping it
    # purely because it's outside KINEMATIC_CORE/ACTUATION (a physics-motivated boundary, not a
    # "these can't matter" claim) was too blunt; put back explicitly rather than assumed-safe-to-omit.
    "battery_state_voltage", "battery_state_current", "battery_state_charge",
    "battery_state_capacity", "battery_state_temperature", "battery_state_percentage",
    # slip_status_is_slipping: the robot's own firmware slip DETECTION flag -- feature_groups.py
    # calls it a candidate "silver label" for validating a slip signal, not literally an input the
    # kinematics equations need, but that's exactly why it's worth keeping as a MONITORED node (a
    # slip event should make other kinematic nodes look anomalous too; this flag directly says when
    # that's happening) rather than treating "not a kinematics input" as "irrelevant."
    "slip_status_is_slipping",
    # stop_status_is_stopped: plausibly relevant to the "stuck" fault type specifically (robot04's
    # fault, 149 windows) -- dropped earlier along with the rest of STATUS_FLAGS without checking
    # whether it correlates with the one fault type it's most likely to matter for.
    "stop_status_is_stopped",
]  # NOT re-including dock_status_*/kidnap_status_*/ir_opcode_sensor -- no specific fault-relevance
   # argument was made for those the way there is for battery/slip/stop, so they stay out rather
   # than re-adding everything status_flags/environment ever had.
#
# TESTED and REVERTED: actually including INCLUDE_EXTRA_NODES made things clearly worse
# (B_node_max 0.742->0.521, F_v3_max 0.750->0.520, even "Low battery"'s OWN AUROC 0.402->0.183) -- not because
# these signals are irrelevant, but because each hits a DIFFERENT already-diagnosed failure mode
# feeding it through the same pipeline as velocity/current: battery_state_* is ANOTHER cumulative/
# non-stationary quantity (same class of problem as odom position/wheel ticks, see
# EXCLUDE_NODES_PREFIXES above -- battery drains ~monotonically over a session, doesn't cluster
# into "regimes"); slip_status_is_slipping/stop_status_is_stopped are near-constant binary flags
# (99.9%/92.3% one value) where z-score/top-k-mean normalization degenerates (near-zero IQR on
# normal data, then a huge z-score spike on the rare nonzero value) -- the same "tiny-client IQR
# blowup" failure mode found on the older, since-deleted FLGDNMemory architecture (its fix,
# iqr_floor_frac, is documented in `memory/feature-purification-audit.md`). Left EMPTY
# (not deleted) since the underlying concern (battery/slip/stop plausibly matter for Low
# battery/stuck specifically) is legitimate -- the right fix is a DIFFERENT encoding (e.g. a
# within-window battery RATE OF CHANGE instead of raw level; scoring slip/stop status separately
# as a rule/label rather than through the continuous aggregation), not raw inclusion as a node.
INCLUDE_EXTRA_NODES_UNUSED = list(INCLUDE_EXTRA_NODES)
INCLUDE_EXTRA_NODES = []
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_TYPED = 0.5
MIN_PROTO_SAMPLES = 5
TOP_K_AGG = 3  # top-k-mean aggregation for the node/edge/typed-edge GROUPS -- see
               # two_stage_group_score()'s docstring; same value as voraus-AD's fix, not
               # separately tuned for robo3er's node count
GAMMA = 0.5
DELTA = 0.5  # matches train_fl_memory_gdn.py's tuned value for this same 5-client,
             # heavily non-IID robo3er setting
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
    data_full, targets, cols_full, label_map, robot_map = load_robo3er(drop_dead=True)
    selected = set(feature_groups.active_feature_names(FEATURE_GROUPS))
    selected -= {c for c in cols_full if c.startswith(EXCLUDE_NODES_PREFIXES)}
    selected |= set(INCLUDE_EXTRA_NODES)
    keep_idx = [i for i, c in enumerate(cols_full) if c in selected]  # preserves cols_full's order
    data, cols = data_full[:, :, keep_idx], [cols_full[i] for i in keep_idx]
    data = data.astype(np.float32)

    with open(DATA_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    client_indices = partition["data_indices"]

    clients = []
    for client_id, idx_dict in enumerate(client_indices):
        all_idx = np.array(idx_dict["train"] + idx_dict["val"] + idx_dict["test"], dtype=int)
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
    return data, label_map, clients, cols


def scale_client(data, scaler, idx):
    n, t, f = data[idx].shape
    return scaler.transform(data[idx].reshape(-1, f)).reshape(n, t, f).astype(np.float32)


def fit_client_scaler(data, fit_idx):
    scaler = StandardScaler()
    scaler.fit(data[fit_idx].reshape(-1, data.shape[-1]))
    return scaler


@torch.no_grad()
def per_sample_scores(model, windows, batch_size=BATCH_SIZE):
    model.eval()
    d_proto_all, d_node_all, resid_struct_all, resid_phys_all, idx_all, r_edge_all, d_mahal_all = [], [], [], [], [], [], []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(DEVICE)
        out = model(batch, training_mode=False)
        d_proto_all.append(out["d_proto"].cpu().numpy())
        d_node_all.append(out["d_node"].cpu().numpy())
        resid_struct_all.append(out["resid_struct"].cpu().numpy())
        resid_phys_all.append(out["resid_phys"].cpu().numpy())
        idx_all.append(out["idx"].cpu().numpy())
        r_edge_all.append(out["r_edge"].cpu().numpy())
        d_mahal_all.append(out["d_mahal"].cpu().numpy())
    return (np.concatenate(d_proto_all), np.concatenate(d_node_all), np.concatenate(resid_struct_all),
            np.concatenate(resid_phys_all), np.concatenate(idx_all), np.concatenate(r_edge_all),
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
            l_typed = out["r_edge"].mean()
            loss = l_pred + BETA * l_me + l_mc + LAMBDA_EDGE * l_edge + LAMBDA_TYPED * l_typed
            loss.backward()
            optimizer.step()
            if calib_mode == "ema":
                # Path A: online EMA update of d_node's per-prototype mean/var,
                # AFTER the optimizer step (uses this step's post-update idx/d_node
                # on the next forward would be ideal, but re-forwarding every step
                # is wasteful; using this step's own out["d_node"]/out["idx"] -- from
                # just before the update -- is the standard VQ-VAE-EMA convention:
                # the codebook/encoder move slowly per step, so a one-step lag is
                # negligible next to ema_decay's much longer effective averaging window.
                model.memory.update_ema(out["d_node"].detach(), out["idx"].detach())
    return model


def zscore(x, calib_x):
    median = np.median(calib_x, axis=0)
    q75, q25 = np.percentile(calib_x, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    return (x - median) / iqr


def topk_mean(x, k):
    """x: [B, D]. Mean of the top-k values along axis 1 (k clipped to D)."""
    k = min(k, x.shape[1])
    return np.sort(x, axis=1)[:, -k:].mean(axis=1)


def two_stage_group_score(raw_calib, raw_x, k=TOP_K_AGG, score_head=None, idx_calib=None, idx_x=None,
                          ema_memory=None):
    """Replaces a raw `.max(axis=1)` over a many-dimension group
    (d_node/resid_struct/resid_phys) with a top-k-mean, THEN a second
    calibration stage re-z-scoring that aggregate statistic against its
    OWN calib-split distribution -- fixes the max-aggregation noise-floor
    problem (more dimensions -> more chances for one to spike by noise
    alone -> inflated false-positive tail on normal data), same fix
    already validated on voraus-AD (`run_voraus_ad_joint_prototype_v3.py`,
    `memory/joint-prototype-scheme-v3.md`). See that script's module
    docstring for the full mechanism explanation.

    `score_head` (a `ScoreCalibrationHead`, Path B) or `ema_memory` (a
    `JointPrototypeMemory` with EMA stats populated, Path A) OPTIONALLY
    replace stage 1's GLOBAL per-dimension z-score with a per-prototype
    one -- default (`score_head=None`, `ema_memory=None`) is bit-identical
    to the original global behavior. Stage 2 (the final aggregate-scalar
    z-score) stays global in every mode, matching signal H's own
    convention (`d_mahal` is already per-prototype-normalized internally,
    then z-scored globally against calib at script level)."""
    if score_head is not None:
        z_calib = score_head.zscore(raw_calib, idx_calib)
        z_x = score_head.zscore(raw_x, idx_x)
    elif ema_memory is not None:
        z_calib = ema_memory.ema_zscore(raw_calib, idx_calib)
        z_x = ema_memory.ema_zscore(raw_x, idx_x)
    else:
        z_calib = zscore(raw_calib, raw_calib)
        z_x = zscore(raw_x, raw_calib)
    stat_calib = topk_mean(z_calib, k)
    stat_x = topk_mean(z_x, k)
    median = np.median(stat_calib)
    iqr = max(np.percentile(stat_calib, 75) - np.percentile(stat_calib, 25), 1e-8)
    return (stat_x - median) / iqr


def ablation_scores(node_score, edge_score, typed_score, cov_score):
    b = node_score
    return {
        "B_node_max": b,
        "C_struct_max": edge_score,
        "E_phys_max": typed_score,
        "F_v3_max": np.max(np.stack([node_score, edge_score, typed_score], axis=1), axis=1),
        "H_cov_mahal": cov_score,
        "I_node_cov_max": np.maximum(b, cov_score),
        "J_v3_cov_max": np.max(np.stack([node_score, edge_score, typed_score, cov_score], axis=1), axis=1),
    }


def main(calib_mode="global", alpha=20.0):
    assert calib_mode in ("global", "per_prototype", "ema", "shrinkage")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"loading robo3er, splitting into 5 real per-robot federated clients, "
          f"{FEATURE_GROUPS} as nodes (federated-capacity-scaled, see module docstring) ...")
    data, label_map, clients, cols = build_clients()
    num_nodes = len(cols)
    node_idx = {name: i for i, name in enumerate(cols)}
    prior_edges = [(node_idx[s], node_idx[d]) for s, d in EDGES_NAMED]
    for c in clients:
        print(f"  client {c.client_id} ({c.robot_name}): fit={len(c.fit_idx)} calib={len(c.calib_idx)} "
              f"test_normal={len(c.test_normal_idx)} faults={ {label_map[str(k)]: len(v) for k, v in c.fault_idx_by_label.items()} }")

    scalers = [fit_client_scaler(data, c.fit_idx) for c in clients]
    models = [JointPrototypeV31(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=EMBED_DIM,
                                    num_prototypes=NUM_PROTOTYPES, prior_edges=prior_edges,
                                    edge_types=EDGE_TYPES, top_k=TOP_K).to(DEVICE)
              for _ in clients]

    # shared round-0 init for every non-memory module, same reasoning as
    # train_fl_memory_gdn.py: independently-random encoders make cross-client
    # cosine similarity between codebook prototypes meaningless.
    shared_init = {k: v.clone() for k, v in models[0].state_dict().items() if not k.startswith("memory.")}
    for model in models[1:]:
        model.load_state_dict(shared_init, strict=False)

    print(f"\nfederated training: {ROUNDS} rounds x {LOCAL_EPOCHS} local epochs, "
          f"memory exchange (gamma={GAMMA}, delta={DELTA})"
          + (", + FedAvg'd encoder/decoder" if SYNC_ENCODER_DECODER else " only"))
    diagnostics_log = []
    for rnd in range(1, ROUNDS + 1):
        fit_counts = []
        for c, model, scaler in zip(clients, models, scalers):
            fit_w = scale_client(data, scaler, c.fit_idx)
            train_local(model, fit_w, LOCAL_EPOCHS, calib_mode=calib_mode)
            fit_counts.append(len(fit_w))

        codebooks = [m.memory.codebook.detach().clone() for m in models]
        usage_counts = [m.memory.usage_count.detach().clone() for m in models]
        P_G, diag = align_and_split(codebooks, usage_counts, gamma=GAMMA, delta=DELTA)
        diag = {k: v for k, v in diag.items() if k != "per_cluster_per_node_agreement"}
        diag["round"] = rnd
        diagnostics_log.append(diag)
        print(f"  round {rnd}: {diag}")

        for model, p_g in zip(models, P_G):
            model.memory.load_memory(p_g)

        if calib_mode == "shrinkage":
            # one-shot local pass over each client's OWN fit split, using this
            # round's just-received aligned codebook (compute_shrinkage_stats'
            # docstring explains why placement AFTER load_memory() matters --
            # index k means the same shared prototype for every client only
            # once everyone has the same aligned codebook).
            client_dev_stats = []
            for c, model, scaler in zip(clients, models, scalers):
                fit_w = scale_client(data, scaler, c.fit_idx)
                _, d_node_fit, _, _, idx_fit, _, _ = per_sample_scores(model, fit_w)
                client_dev_stats.append(compute_prototype_dev_stats(d_node_fit, idx_fit, NUM_PROTOTYPES))
            num_shared = diag["num_shared_prototypes"]
            shrink_stats, alpha_used = compute_shrinkage_stats(client_dev_stats, num_shared, alpha)
            diag["shrinkage_alpha"] = alpha_used.tolist()
            for model, (mean_s, var_s, valid_s) in zip(models, shrink_stats):
                model.memory.load_shrinkage_stats(mean_s, var_s, valid_s)

        if SYNC_ENCODER_DECODER:
            # sample-size-weighted FedAvg over encoder+decoder ONLY -- memory (just
            # personalized above), edge_head, typed_head stay untouched/fully local.
            avg = fedavg_state_dict([m.state_dict() for m in models], fit_counts,
                                     prefixes=("encoder.", "decoder."))
            for model in models:
                model.load_state_dict(avg, strict=False)

    report = {"config": {"rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "num_prototypes": NUM_PROTOTYPES,
                          "embed_dim": EMBED_DIM, "top_k": TOP_K, "gamma": GAMMA, "delta": DELTA,
                          "prior_edges": EDGES_NAMED, "edge_types": EDGE_TYPES,
                          "min_proto_samples": MIN_PROTO_SAMPLES, "top_k_agg": TOP_K_AGG,
                          "sync_encoder_decoder": SYNC_ENCODER_DECODER,
                          "feature_groups": FEATURE_GROUPS, "num_nodes": num_nodes},
              "alignment_log": diagnostics_log, "clients": {}}

    # Pass 1 (ALL clients, including fault-free ones like robot02): Stage B
    # calibration -- set typed-head and covariance-head from each client's
    # held-out calib split, then store calib arrays for per-client z-scoring.
    calib_data = {}
    cov_min = max(MIN_PROTO_SAMPLES, num_nodes + 1)
    for c, model, scaler in zip(clients, models, scalers):
        print(f"\nStage B ({c.robot_name}): typed-head + covariance-head calibration ...")
        calib_w = scale_client(data, scaler, c.calib_idx)
        _, d_node_calib_b, _, _, calib_idx_w, calib_r_w, _ = per_sample_scores(model, calib_w)
        model.typed_head.set_calibration(calib_r_w, calib_idx_w, NUM_PROTOTYPES, min_samples=MIN_PROTO_SAMPLES)
        model.cov_head.set_calibration(d_node_calib_b, calib_idx_w, min_samples=cov_min)

        _, d_node_calib, resid_struct_calib, resid_phys_calib, _, _, d_mahal_calib = per_sample_scores(model, calib_w)
        if calib_mode == "per_prototype":
            model.set_score_calibration(d_node_calib, resid_struct_calib, calib_idx_w,
                                        min_samples=MIN_PROTO_SAMPLES)
        calib_data[c.client_id] = (d_node_calib, resid_struct_calib, resid_phys_calib, d_mahal_calib, calib_idx_w)

    all_pairs = {k: [] for k in ["B_node_max", "C_struct_max", "E_phys_max", "F_v3_max",
                                   "H_cov_mahal", "I_node_cov_max", "J_v3_cov_max"]}
    for c, model, scaler in zip(clients, models, scalers):
        d_node_calib, resid_struct_calib, resid_phys_calib, d_mahal_calib, calib_idx_w = calib_data[c.client_id]
        report["clients"].setdefault(c.robot_name, {})

        if not c.fault_idx_by_label:
            print(f"client {c.client_id} ({c.robot_name}): no fault windows, skipping evaluation")
            continue

        node_head = model.score_calib_node if calib_mode == "per_prototype" else None
        struct_head = model.score_calib_struct if calib_mode == "per_prototype" else None
        ema_mem = model.memory if calib_mode in ("ema", "shrinkage") else None  # C has no
        # online EMA hook (or shrinkage hook) -> falls back to global; ema_zscore() is
        # reused as-is for shrinkage since load_shrinkage_stats() populates the SAME buffers.

        test_w = scale_client(data, scaler, c.test_normal_idx)
        _, d_node_normal, resid_struct_normal, resid_phys_normal, idx_normal, _, d_mahal_normal = per_sample_scores(model, test_w)
        node_score_normal = two_stage_group_score(d_node_calib, d_node_normal, score_head=node_head,
                                                   idx_calib=calib_idx_w, idx_x=idx_normal, ema_memory=ema_mem)
        edge_score_normal = two_stage_group_score(resid_struct_calib, resid_struct_normal, score_head=struct_head,
                                                   idx_calib=calib_idx_w, idx_x=idx_normal)
        typed_score_normal = two_stage_group_score(resid_phys_calib, resid_phys_normal)
        cov_score_normal = zscore(d_mahal_normal, d_mahal_calib)
        scores_normal = ablation_scores(node_score_normal, edge_score_normal,
                                         typed_score_normal, cov_score_normal)

        client_report = {"robot_name": c.robot_name, "fault_types": {}}
        rows = {k: [] for k in scores_normal}
        for label_id, fault_idx in c.fault_idx_by_label.items():
            name = label_map[str(label_id)]
            fault_w = scale_client(data, scaler, fault_idx)
            _, d_node_f, resid_struct_f, resid_phys_f, idx_f, _, d_mahal_f = per_sample_scores(model, fault_w)
            node_score_f = two_stage_group_score(d_node_calib, d_node_f, score_head=node_head,
                                                  idx_calib=calib_idx_w, idx_x=idx_f, ema_memory=ema_mem)
            edge_score_f = two_stage_group_score(resid_struct_calib, resid_struct_f, score_head=struct_head,
                                                  idx_calib=calib_idx_w, idx_x=idx_f)
            typed_score_f = two_stage_group_score(resid_phys_calib, resid_phys_f)
            cov_score_f = zscore(d_mahal_f, d_mahal_calib)
            scores_fault = ablation_scores(node_score_f, edge_score_f,
                                            typed_score_f, cov_score_f)

            aurocs = {}
            for key in scores_normal:
                labels = np.concatenate([np.zeros(len(scores_normal[key])), np.ones(len(scores_fault[key]))])
                s = np.concatenate([scores_normal[key], scores_fault[key]])
                auroc = float(sk_metrics.roc_auc_score(labels, s))
                aurocs[key] = auroc
                rows[key].append(auroc)
                all_pairs[key].append(auroc)
            client_report["fault_types"][name] = {"n": int(len(fault_idx)), **aurocs}
            print(f"  {c.robot_name:<10}{name:<16}n={len(fault_idx):<5}"
                  + "  ".join(f"{k}={v:.3f}" for k, v in aurocs.items()))

        client_report["client_mean_auroc"] = {k: float(np.mean(v)) for k, v in rows.items()}
        report["clients"][c.robot_name] = client_report

    summary_overall = {k: float(np.mean(v)) for k, v in all_pairs.items()}
    print(f"\n{'method':<24}{'mean AUROC (all client-fault pairs)':>36}")
    for key, v in summary_overall.items():
        print(f"{key:<24}{v:>36.3f}")

    report["config"]["calib_mode"] = calib_mode
    if calib_mode == "shrinkage":
        report["config"]["alpha"] = "auto" if alpha is None else alpha
    report["summary_mean_auroc_overall"] = summary_overall
    suffix = "_encdec_synced" if SYNC_ENCODER_DECODER else ""
    if calib_mode == "shrinkage":
        mode_suffix = f"_calibmode_shrinkage_alpha{'auto' if alpha is None else int(alpha)}"
    elif calib_mode != "global":
        mode_suffix = f"_calibmode_{calib_mode}"
    else:
        mode_suffix = ""
    out_json = OUT_DIR / f"robo3er_v3_1_federated{suffix}{mode_suffix}_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dicts": [m.state_dict() for m in models], "report": report},
                OUT_DIR / f"robo3er_v3_1_federated{suffix}{mode_suffix}.pth")
    print(f"\nsaved -> {out_json}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-mode", choices=["global", "per_prototype", "ema", "shrinkage"], default="global")
    parser.add_argument("--alpha", type=str, default="20",
                        help="shrinkage strength for --calib-mode shrinkage, or 'auto' for data-driven per-slot alpha")
    args = parser.parse_args()
    args.alpha = None if args.alpha == "auto" else float(args.alpha)
    main(calib_mode=args.calib_mode, alpha=args.alpha)
