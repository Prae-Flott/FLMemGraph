#!/usr/bin/env python3
"""
For every RED-severity fault window flagged by
`run_sielaff_red_joint_prototype_v2_federated.py`, find which FEATURE
(node) deviated furthest from its prototype (argmax reliable per-node
z-score, `s_node`), then aggregate by fault ID/name (from
`data/sielaff_red/window_red_ids.json`, produced alongside the window
labels by `build_sielaff_red.py`) to answer: "when fault X happens, which
sensor moves the most, and does that make physical sense?"

Retrains the exact same federated V2 model (same hyperparameters/seed as
`run_sielaff_red_joint_prototype_v2_federated.py`) rather than reloading
its checkpoint, so the calibration stats (median/IQR per node, reliable-
node mask) are recomputed consistently in the same script.

Plausibility check: a hand-built FEATURE_DOMAIN / FAULT_EXPECTED_DOMAIN
mapping (see bottom of file) -- NOT learned, just the physical semantics
of the 39 raw feature columns vs. what each red fault ID actually is.
Many red faults (compactor doors/flaps, crate handling, safety circuit)
have NO directly-instrumented feature in this 39-column set (no compactor/
crate/safety sensors were logged), so their top feature is necessarily an
INDIRECT correlate, not a direct sensor match -- flagged as such rather
than silently claimed as "relevant."

Usage:
    python3 analyze_sielaff_red_feature_attribution.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "sielaff_red_v2fed", REPO_ROOT / "run_sielaff_red_joint_prototype_v2_federated.py")
v2fed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2fed)

DATA_DIR = REPO_ROOT.parent / "data" / "sielaff_red"
OUT_DIR = REPO_ROOT.parent / "checkpoints" / "sielaff"


def main():
    torch.manual_seed(v2fed.SEED)
    np.random.seed(v2fed.SEED)

    print("retraining the federated model identically to the AUROC run ...")
    data, targets, cols, label_map, clients = v2fed.build_clients()
    num_nodes = len(cols)
    scalers = [v2fed.fit_client_scaler(data, c.fit_idx) for c in clients]
    models = [v2fed.JointPrototypeMemoryOnly(num_nodes=num_nodes, window_size=v2fed.WINDOW_LEN,
                                              embed_dim=v2fed.EMBED_DIM,
                                              num_prototypes=v2fed.NUM_PROTOTYPES).to(v2fed.DEVICE)
              for _ in clients]
    shared_init = {k: v.clone() for k, v in models[0].state_dict().items() if not k.startswith("memory.")}
    for model in models[1:]:
        model.load_state_dict(shared_init, strict=False)

    for rnd in range(1, v2fed.ROUNDS + 1):
        for c, model, scaler in zip(clients, models, scalers):
            fit_w = v2fed.scale_client(data, scaler, c.fit_idx)
            v2fed.train_local(model, fit_w, v2fed.LOCAL_EPOCHS)
        codebooks = [m.memory.codebook.detach().clone() for m in models]
        usage_counts = [m.memory.usage_count.detach().clone() for m in models]
        P_G, diag = v2fed.align_and_split(codebooks, usage_counts, gamma=v2fed.GAMMA, delta=v2fed.DELTA)
        print(f"  round {rnd}: shared_clusters={diag['num_multi_client_clusters']}")
        for model, p_g in zip(models, P_G):
            model.memory.load_memory(p_g)

    with open(DATA_DIR / "window_red_ids.json") as f:
        window_red_ids = json.load(f)
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    red_id_names = {int(k): v for k, v in meta["red_id_names"].items()}

    # POOLED calibration across all 10 clients, not per-client: per-client calib is only
    # 6-9 windows here (this dataset's clean windows are scarce, see
    # memory/sielaff-red-fault-federated.md), so per-client node IQR collapses to the
    # zscore() floor (1e-8) for nearly every feature -- verified empirically, that made
    # "total_weight" win almost every fault type purely because it's the one feature
    # that's exactly 0 in nearly all clean windows, not because it's actually the most
    # fault-relevant. Pooling calib/test_normal scores across clients (63/77 windows
    # total instead of 6-9) gives real per-node IQR estimates for attribution. Each
    # window is still scored by its OWN client's model -- only the z-scoring reference
    # distribution is pooled. This is a deliberate departure from the federated AUROC
    # run's per-client-calibrates-locally convention, appropriate here because this
    # script answers a different question (which node, not is-it-anomalous) that needs
    # more resolution than 10 way smaller per-client calib sets can support.
    s_node_calib_all, s_node_normal_all = [], []
    per_client_scores = []
    for c, model, scaler in zip(clients, models, scalers):
        calib_arr = v2fed.scale_client(data, scaler, c.calib_idx)
        test_normal_arr = v2fed.scale_client(data, scaler, c.test_normal_idx)
        _, s_node_calib = v2fed.per_sample_scores(model, calib_arr)
        _, s_node_normal = v2fed.per_sample_scores(model, test_normal_arr)
        s_node_calib_all.append(s_node_calib)
        s_node_normal_all.append(s_node_normal)
        per_client_scores.append((c, model, scaler))

    s_node_calib_pool = np.concatenate(s_node_calib_all, axis=0)
    s_node_normal_pool = np.concatenate(s_node_normal_all, axis=0)
    _, node_iqr = v2fed.zscore(s_node_normal_pool, s_node_calib_pool)
    median_iqr = np.median(node_iqr)
    reliable_mask = node_iqr >= v2fed.RELIABILITY_RATIO * median_iqr
    reliable_idx = np.where(reliable_mask)[0]
    print(f"\npooled calib n={len(s_node_calib_pool)}, test_normal n={len(s_node_normal_pool)}, "
          f"{reliable_mask.sum()}/{num_nodes} nodes reliable "
          f"(node_iqr floor-degenerate before pooling: verified separately)")

    calib_median = np.median(s_node_calib_pool, axis=0)
    q75, q25 = np.percentile(s_node_calib_pool, [75, 25], axis=0)
    calib_iqr = np.maximum(q75 - q25, 1e-8)

    # `total_weight`/`total_deposit`/`journal_count`/etc. ("transaction_throughput" domain)
    # turned out to dominate the top-1 feature for almost EVERY fault type (0.5-1.0 vote
    # share) once calibration was fixed above -- this is a genuine activity/exposure
    # confound, not noise: a strictly no_error window is disproportionately an IDLE
    # window (no transactions -> no chance of any error, of any severity), so ANY window
    # with real throughput looks anomalous relative to that idle baseline regardless of
    # WHICH fault actually occurred. Report top-1 as-is (raw), but also compute a
    # confound-excluded ranking (masking out the throughput domain from the argmax) as
    # the more mechanism-informative signal for plausibility judgment.
    reliable_names = [cols[i] for i in reliable_idx]
    non_confound_mask = np.array([feature_domain(n) != "transaction_throughput" for n in reliable_names])
    non_confound_idx = np.where(non_confound_mask)[0]

    votes = defaultdict(Counter)
    votes_excl_confound = defaultdict(Counter)
    top1_z = defaultdict(list)
    n_windows_per_id = Counter()

    for c, model, scaler in per_client_scores:
        red_idx = np.array([i for i in c.all_idx if targets[i] == 1])
        if len(red_idx) == 0:
            continue
        red_arr = v2fed.scale_client(data, scaler, red_idx)
        _, s_node_f = v2fed.per_sample_scores(model, red_arr)
        z_node_f = (s_node_f - calib_median) / calib_iqr
        z_node_f_reliable = z_node_f[:, reliable_idx]

        top1_local = np.argmax(z_node_f_reliable, axis=1)
        top1_local_excl = non_confound_idx[np.argmax(z_node_f_reliable[:, non_confound_idx], axis=1)]
        for row, global_win_idx in enumerate(red_idx):
            feat_name = cols[reliable_idx[top1_local[row]]]
            feat_name_excl = cols[reliable_idx[top1_local_excl[row]]]
            z = float(z_node_f_reliable[row, top1_local[row]])
            for rid in window_red_ids[global_win_idx]:
                votes[rid][feat_name] += 1
                votes_excl_confound[rid][feat_name_excl] += 1
                top1_z[rid].append(z)
                n_windows_per_id[rid] += 1

    rows = []
    for rid in sorted(votes, key=lambda r: -sum(votes[r].values())):
        name = red_id_names.get(rid, f"id_{rid}")
        n = n_windows_per_id[rid]
        top2_raw = votes[rid].most_common(2)
        top2_excl = votes_excl_confound[rid].most_common(2)
        raw_feat, raw_count = top2_raw[0]
        excl_feat, excl_count = top2_excl[0]
        mean_z = float(np.mean(top1_z[rid]))
        domain = classify_plausibility(name, excl_feat)
        rows.append({
            "error_id": rid, "name": name, "n_windows": n,
            "raw_top_feature": raw_feat, "raw_top_feature_share": round(raw_count / n, 3),
            "top_feature_excl_throughput_confound": excl_feat,
            "top_feature_excl_confound_share": round(excl_count / n, 3),
            "second_feature_excl_confound": top2_excl[1][0] if len(top2_excl) > 1 else None,
            "second_feature_excl_confound_share": round(top2_excl[1][1] / n, 3) if len(top2_excl) > 1 else None,
            "mean_raw_top_feature_zscore": round(mean_z, 2),
            "plausibility": domain[0], "rationale": domain[1],
        })

    print(f"\n{'id':<5}{'name':<32}{'n':<5}{'top_feature (throughput confound excluded)':<32}{'share':<7}{'plausibility':<16}")
    for r in rows:
        print(f"{r['error_id']:<5}{r['name']:<32}{r['n_windows']:<5}"
              f"{r['top_feature_excl_throughput_confound']:<32}"
              f"{r['top_feature_excl_confound_share']:<7}{r['plausibility']:<16}")

    with open(OUT_DIR / "sielaff_red_feature_attribution.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {OUT_DIR / 'sielaff_red_feature_attribution.json'}")


# --- plausibility mapping: physical semantics of the 39 raw feature columns ---
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


# fault-name keyword -> expected feature domain(s); faults matching none of these
# keywords have NO directly-instrumented sensor in this 39-column feature set
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


def classify_plausibility(fault_name, top_feature):
    got = feature_domain(top_feature)
    for keywords, expected in FAULT_EXPECTED_DOMAIN:
        if any(k in fault_name for k in keywords):
            if got in expected:
                return ("plausible", f"{fault_name} is physically tied to {sorted(expected)}, "
                                      f"top feature '{top_feature}' ({got}) matches")
            return ("indirect", f"{fault_name} expected {sorted(expected)}, but top feature "
                                 f"'{top_feature}' ({got}) is an indirect correlate, not the "
                                 f"expected sensor family")
    if any(k in fault_name for k in NO_SENSOR_KEYWORDS):
        return ("no_direct_sensor", f"{fault_name} has NO directly-instrumented feature in this "
                                     f"39-column set (no compactor/crate/safety-circuit sensor was "
                                     f"logged) -- '{top_feature}' ({got}) is at best a proxy via "
                                     f"correlated downstream effects, not a direct measurement")
    return ("unmapped", f"no expected-domain rule defined for {fault_name}; top feature "
                         f"'{top_feature}' ({got}) reported without a plausibility judgment")


if __name__ == "__main__":
    main()
