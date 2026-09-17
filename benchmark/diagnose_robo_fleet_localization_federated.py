#!/usr/bin/env python3
"""
FEDERATED node-level DIAGNOSIS (root-cause localization) test for
B/C/H/K/BK/BCK on robo_fleet, scored as TOP-K ACCURACY per fault type
against the mechanical-domain ground truth in
`data/robo_fleet/metadata.json`'s `fault_localization_domains` field
(flat per-fault feature set, no direct/indirect split -- the same
convention as `data/robo_pdm_test/metadata.json`, whose
`diagnose_robo_pdm_test_localization_federated.py` this script's top-k
mechanism (`masked_topk`/`combo_topk`/`topk_baseline_pct`) is ported
from verbatim).

Built directly on `run_robo_fleet_bck_federated.py` (imported as a
module, exactly as `diagnose_robo3er_localization_federated.py` imports
`run_robo3er_bck_federated.py`) -- same JointPrototypeV31Forecast +
`federated_train_eval` plumbing, NOT the older V21/archive stack
`diagnose_robo_pdm_test_localization_federated.py` is built on.

Per-signal localization mechanism (identical readout to
`diagnose_robo3er_localization_federated.py`, generalized from argmax to
top-k):
  B: top-k_i zscore(d_node)_i                         (native)
  C: top-k_i zscore(resid_struct)_i                    (native)
  K: top-k_i zscore(k_resid)_i                         (native)
  H: top-k_i contribution_i  (Hotelling-T^2 per-node contribution
     decomposition of the Mahalanobis quadratic form, read off
     DeviationCovarianceHead's calib_mu/calib_cov_inv/global_mu/
     global_cov_inv buffers -- no model changes)
  BK/BHK/BCK: per window, a confidence-weighted (softmax) blend of the
     contributing signals' per-node z-scores supplies the top-k node set
     (`combo_topk`, mirrors `combo_conf`'s softmax-weighting semantics
     but keeps the whole top-k set instead of collapsing to one argmax).
     BHK (B+H+K) is the actual shipped project mainline combo
     (`BHK_max = smooth_max(B, H, K)` in `run_robo_fleet_bck_federated.py`'s
     own detection report / `federated_train_eval.add_forecast_scores`) -- BCK
     (B+C+K, swapping in C for H) is kept alongside it for comparison
     only, it is NOT the mainline signal.
A fault window counts as a HIT for a signal if ANY of its top-k reliable
nodes intersects the fault's ground-truth domain set.

Ground-truth NAME FIX: `fault_localization_domains.domains` in
`data/robo_fleet/metadata.json` was copied from `robo_pdm_test`'s and
keeps that dataset's fault names for 2 of 4 faults
(`right_side_added_load_2.5kg`, `right_drive_wheel_cable`), which do NOT
match robo_fleet's own `label_map` names (`added_load_2.5kg`,
`drive_wheel_cable`, confirmed via `label_map`/`class_names` in the same
metadata file). `ALIAS` below remaps the domain dict's keys onto
robo_fleet's actual fault names before use -- without this, top-k
hit-rate for those two faults would silently read as an always-empty
domain (0% for every signal, indistinguishable from a real localization
failure).

FedPRO is NOT included in this comparison: it is a supervised classifier
(Normal + 4 fault types, `main_fedpro_baseline` in
`run_robo_fleet_bck_federated.py`) with no native per-node anomaly score,
so it cannot produce a top-k node ranking without an added attribution
mechanism (e.g. input-gradient saliency) that this run does not
implement, per explicit user direction. Its own per-fault classification
accuracy/AUROC (from the existing `--baseline fedpro` report) is reported
alongside this table as reference context only, NOT as a top-k
localization competitor -- it answers a different question (which fault
class, not which node).

Per-client models (4 real robots), each evaluated on its OWN fault
windows using its OWN calibration stats -- results pooled across all
clients into one aggregate top-k hit-rate table per signal per fault
type.

Usage:
    python3 benchmark/diagnose_robo_fleet_localization_federated.py [--horizon-mult M] [--topk K] [--out-suffix NAME]
"""
import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT.parent / "checkpoints" / "robo_fleet"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = REPO_ROOT.parent / "data" / "robo_fleet"

spec = importlib.util.spec_from_file_location("robo_fleet_bck_fed", REPO_ROOT / "run_robo_fleet_bck_federated.py")
v2f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2f)

RELIABILITY_RATIO = 0.05

# robo_pdm_test -> robo_fleet fault-name aliasing, see module docstring.
ALIAS = {
    "right_side_added_load_2.5kg": "added_load_2.5kg",
    "right_drive_wheel_cable": "drive_wheel_cable",
}


def load_domain():
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    raw = meta["fault_localization_domains"]["domains"]
    domain = {ALIAS.get(k, k): set(v) for k, v in raw.items()}
    return domain


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


def masked_topk(z, reliable, k):
    """Per window, the indices of the top-`k` RELIABLE nodes by z-score
    (descending) -- ported verbatim from
    `diagnose_robo_pdm_test_localization_federated.py`."""
    if len(z) == 0:
        return np.zeros((0, 0), dtype=int)
    z_masked = np.where(reliable[None, :], z, -np.inf)
    k_eff = max(1, min(k, int(reliable.sum())))
    return np.argsort(-z_masked, axis=1)[:, :k_eff]


def combo_topk(scalars, per_node_arrays, reliable_masks, k, T=1.0):
    if len(scalars[0]) == 0:
        return np.zeros((0, 0), dtype=int)
    stacked_scalar = np.stack(scalars, axis=1)
    w = np.exp((stacked_scalar - stacked_scalar.max(axis=1, keepdims=True)) / T)
    w /= w.sum(axis=1, keepdims=True)
    stacked_pernode = np.stack(per_node_arrays, axis=0)
    combined = np.einsum("nk,knd->nd", w, stacked_pernode)
    reliable = np.all(np.stack(reliable_masks, axis=0), axis=0)
    return masked_topk(combined, reliable, k)


def topk_baseline_pct(domain_size, num_nodes, k):
    """Hypergeometric random-guess baseline for a top-k hit-rate (P(at
    least one of k uniform-without-replacement picks lands in the
    domain)) -- ported verbatim from
    `diagnose_robo_pdm_test_localization_federated.py`."""
    k_eff = min(k, num_nodes)
    if domain_size == 0:
        return 0.0
    miss = (math.comb(num_nodes - domain_size, k_eff) / math.comb(num_nodes, k_eff)
            if num_nodes - domain_size >= k_eff else 0.0)
    return round(100.0 * (1.0 - miss), 1)


def main(horizon_mult=10, out_suffix=None, linkage="single", topk=3, num_prototypes=None,
         rounds=None, local_epochs=None):
    torch.manual_seed(v2f.SEED)
    np.random.seed(v2f.SEED)
    forecast_h = v2f.STRIDE * horizon_mult
    rounds = rounds if rounds is not None else v2f.ROUNDS
    local_epochs = local_epochs if local_epochs is not None else v2f.LOCAL_EPOCHS
    num_prototypes = num_prototypes if num_prototypes is not None else v2f.NUM_PROTOTYPES

    domain = load_domain()

    print(f"loading robo_fleet, {v2f.FEATURE_GROUPS} as nodes, 4 federated clients, "
          f"forecast head horizon_mult={horizon_mult}, top-{topk} localization ...")
    data, targets, label_map, clients, cols, window_robot = v2f.build_clients()
    num_nodes = len(cols)
    print(f"num_nodes={num_nodes}")
    print(f"cols={cols}")
    node_idx = {name: i for i, name in enumerate(cols)}
    prior_edges = [(node_idx[s], node_idx[d]) for s, d in v2f.EDGES_NAMED
                   if s in node_idx and d in node_idx]
    edge_types = [t for (s, d), t in zip(v2f.EDGES_NAMED, v2f.EDGE_TYPES) if s in node_idx and d in node_idx]

    missing_domain_nodes = {fault: sorted(names - set(cols)) for fault, names in domain.items()}
    missing_domain_nodes = {f: n for f, n in missing_domain_nodes.items() if n}
    if missing_domain_nodes:
        print(f"WARNING: ground-truth domain nodes outside the {num_nodes}-node set, dropped: {missing_domain_nodes}")
        domain = {f: {n for n in names if n in cols} for f, names in domain.items()}

    scalers = [v2f.fit_client_scaler(data, c.fit_idx) for c in clients]
    models = [v2f.JointPrototypeV31Forecast(
        num_nodes=num_nodes, window_size=v2f.WINDOW_LEN, embed_dim=v2f.EMBED_DIM,
        num_prototypes=num_prototypes, prior_edges=prior_edges, edge_types=edge_types,
        forecast_h=forecast_h, top_k=v2f.TOP_K, forecast_prior_edges=prior_edges,
    ).to(v2f.DEVICE) for _ in clients]

    shared_init = {k: v.clone() for k, v in models[0].state_dict().items() if not k.startswith("memory.")}
    for model in models[1:]:
        model.load_state_dict(shared_init, strict=False)

    fit_pairs = []
    for c in clients:
        vi, chains, _ = v2f.build_pairs(c.fit_idx, targets, window_robot, horizon_mult, True, True)
        x_in = data[vi]
        x_future = v2f.gather_future(data, chains, v2f.STRIDE)
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
        v2f.train_local(model, x_in, x_future, local_epochs, v2f.DEVICE, lr=v2f.LR, beta=v2f.BETA,
                         lambda_edge=v2f.LAMBDA_EDGE, lambda_typed=v2f.LAMBDA_TYPED,
                         lambda_forecast=v2f.LAMBDA_FORECAST, batch_size=v2f.BATCH_SIZE)
        return len(x_in)

    print(f"\nfederated training: {rounds} rounds x {local_epochs} local epochs, "
          f"memory exchange + FedAvg'd encoder/decoder ...")
    v2f.run_federated_rounds(
        clients, models, local_train_step, rounds, v2f.GAMMA, v2f.DELTA,
        sync_encoder_decoder=v2f.SYNC_ENCODER_DECODER, linkage=linkage)

    print("\ncalibrating cov_head per client ...")
    calib_stats = {}
    cov_min = max(v2f.MIN_PROTO_SAMPLES, num_nodes + 1)
    for c, model, scaler in zip(clients, models, scalers):
        calib_w = v2f.scale_client(data, scaler, c.calib_idx)
        _, d_node_calib_b, _, _, calib_idx_w, _, _ = v2f.per_sample_scores(model, calib_w, v2f.DEVICE)
        model.cov_head.set_calibration(d_node_calib_b, calib_idx_w, min_samples=cov_min)

        _, d_node_calib, resid_struct_calib, _, calib_idx_b2, _, d_mahal_calib = v2f.per_sample_scores(model, calib_w, v2f.DEVICE)
        d_mahal_med = float(np.median(d_mahal_calib))
        d_mahal_iqr = max(float(np.percentile(d_mahal_calib, 75) - np.percentile(d_mahal_calib, 25)), 1e-8)

        vi, chains, _ = v2f.build_pairs(c.calib_idx, targets, window_robot, horizon_mult, True, True)
        if len(vi):
            calib_in = v2f.scale_client(data, scaler, vi)
            calib_future_raw = v2f.gather_future(data, chains, v2f.STRIDE)
            n2, t2, f2 = calib_future_raw.shape
            calib_future = scaler.transform(calib_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
            k_resid_calib = v2f.forecast_scores(model, calib_in, calib_future, v2f.DEVICE)
        else:
            k_resid_calib = np.zeros((0, num_nodes), dtype=np.float32)

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

        calib_stats[c.client_id] = dict(
            node_median=node_median, node_iqr=node_iqr, node_reliable=node_reliable,
            struct_median=struct_median, struct_iqr=struct_iqr, struct_reliable=struct_reliable,
            k_median=k_median, k_iqr=k_iqr, k_reliable=k_reliable,
            d_mahal_med=d_mahal_med, d_mahal_iqr=d_mahal_iqr,
        )
        print(f"  {c.robot_name}: reliable nodes B/H={node_reliable.sum()}/{num_nodes}  "
              f"C={struct_reliable.sum()}/{num_nodes}  K={k_reliable.sum()}/{num_nodes}")

    print(f"\ntop-{topk} hit rate: a fault window counts as a HIT if ANY of its top-{topk} "
          f"reliable nodes is in the fault's ground-truth domain set")
    print(f"\n{'fault':<20}{'signal':<8}{'n':<6}{'domain%':<10}")
    pooled = {}

    def add(fault_name, key, idx_matrix):
        entry = pooled.setdefault(fault_name, {}).setdefault(key, {"hits": 0, "n": 0, "node_counter": Counter()})
        dom = domain.get(fault_name, set())
        for row in idx_matrix:
            names = [cols[i] for i in row]
            entry["n"] += 1
            entry["node_counter"].update(names)
            if any(name in dom for name in names):
                entry["hits"] += 1

    for c, model, scaler in zip(clients, models, scalers):
        if not c.fault_idx_by_label:
            continue
        st = calib_stats[c.client_id]
        for label_id, fault_idx in c.fault_idx_by_label.items():
            fault_name = label_map[str(label_id)]
            fault_w = v2f.scale_client(data, scaler, fault_idx)
            _, d_node_f, resid_struct_f, _, idx_f, _, d_mahal_f = v2f.per_sample_scores(model, fault_w, v2f.DEVICE)

            z_node_f = zn(d_node_f, st["node_median"], st["node_iqr"])
            z_struct_f = zn(resid_struct_f, st["struct_median"], st["struct_iqr"])
            contrib_h_f = h_contributions(model, d_node_f, idx_f)

            topk_b = masked_topk(z_node_f, st["node_reliable"], topk)
            topk_c = masked_topk(z_struct_f, st["struct_reliable"], topk)
            topk_h = masked_topk(contrib_h_f, st["node_reliable"], topk)

            vi_f, chains_f, mask_f = v2f.build_pairs(fault_idx, targets, window_robot, horizon_mult, False, False)
            if len(vi_f):
                fault_in = v2f.scale_client(data, scaler, vi_f)
                future_f_raw = v2f.gather_future(data, chains_f, v2f.STRIDE)
                n2, t2, f2 = future_f_raw.shape
                fault_future = scaler.transform(future_f_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                k_resid_f = v2f.forecast_scores(model, fault_in, fault_future, v2f.DEVICE)
                z_k_f = zn(k_resid_f, st["k_median"], st["k_iqr"])
                topk_k_paired = masked_topk(z_k_f, st["k_reliable"], topk)
            else:
                z_k_f = np.zeros((0, num_nodes))
                topk_k_paired = np.zeros((0, 0), dtype=int)

            add(fault_name, "B", topk_b)
            add(fault_name, "C", topk_c)
            add(fault_name, "H", topk_h)
            add(fault_name, "K", topk_k_paired)

            z_mahal_f = zn(d_mahal_f, st["d_mahal_med"], st["d_mahal_iqr"])
            b_paired = z_node_f.max(axis=1)[mask_f]
            c_paired = z_struct_f.max(axis=1)[mask_f]
            h_paired = z_mahal_f[mask_f]
            k_scalar_paired = z_k_f.max(axis=1) if len(z_k_f) else np.zeros(0)
            b_pernode_paired = z_node_f[mask_f]
            c_pernode_paired = z_struct_f[mask_f]
            h_pernode_paired = contrib_h_f[mask_f]

            add(fault_name, "BK", combo_topk([b_paired, k_scalar_paired], [b_pernode_paired, z_k_f],
                                              [st["node_reliable"], st["k_reliable"]], topk))
            # BHK: the actual shipped mainline signal (`BHK_max = smooth_max(B, H, K)`
            # in `federated_train_eval.add_forecast_scores`) -- NOT BCK (which swaps in C for
            # H and was never the mainline combo), see fl-baseline-comparison.md.
            add(fault_name, "BHK", combo_topk([b_paired, h_paired, k_scalar_paired],
                                               [b_pernode_paired, h_pernode_paired, z_k_f],
                                               [st["node_reliable"], st["node_reliable"], st["k_reliable"]], topk))
            add(fault_name, "BCK", combo_topk([b_paired, c_paired, k_scalar_paired],
                                               [b_pernode_paired, c_pernode_paired, z_k_f],
                                               [st["node_reliable"], st["struct_reliable"], st["k_reliable"]], topk))

    results = {}
    for fault_name, sigs in pooled.items():
        for key, entry in sigs.items():
            n = entry["n"]
            domain_pct = 100.0 * entry["hits"] / n if n else 0.0
            print(f"{fault_name:<20}{key:<8}{n:<6}{domain_pct:<10.1f}")
            results.setdefault(fault_name, {})[key] = {
                "n": n, "domain_pct": round(domain_pct, 1),
                "top_nodes": entry["node_counter"].most_common(5),
            }

    baseline = {}
    for fault_name in domain:
        baseline[fault_name] = {"domain_pct": topk_baseline_pct(len(domain[fault_name]), num_nodes, topk)}
    print(f"\nrandom-guess baseline (top-{topk} domain%, {num_nodes} nodes):")
    for fault_name, b in baseline.items():
        print(f"  {fault_name:<20}{b['domain_pct']}")

    results["_baseline"] = baseline
    results["_domain"] = {k: sorted(v) for k, v in domain.items()}
    results["_config"] = {
        "num_nodes": num_nodes, "cols": cols, "feature_groups": v2f.FEATURE_GROUPS, "horizon_mult": horizon_mult,
        "rounds": rounds, "local_epochs": local_epochs, "reliability_ratio": RELIABILITY_RATIO,
        "linkage": linkage, "num_prototypes": num_prototypes, "topk": topk,
        "note": "FedPRO excluded (supervised classifier, no native per-node score) -- see module docstring "
                "and run_robo_fleet_bck_federated.py --baseline fedpro for its own classification accuracy.",
    }
    suffix = out_suffix if out_suffix is not None else f"top{topk}_h{horizon_mult}"
    out_path = OUT_DIR / f"robo_fleet_diagnosis_localization_federated_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-mult", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--num-prototypes", type=int, default=None)
    parser.add_argument("--out-suffix", type=str, default=None)
    parser.add_argument("--linkage", type=str, default="single", choices=["single", "complete"])
    parser.add_argument("--topk", type=int, default=3,
                         help="a fault window counts as a hit if ANY of its top-k reliable nodes "
                              "is in the ground-truth domain (default 3)")
    args = parser.parse_args()
    main(horizon_mult=args.horizon_mult, rounds=args.rounds, local_epochs=args.local_epochs,
         num_prototypes=args.num_prototypes, out_suffix=args.out_suffix, linkage=args.linkage, topk=args.topk)
