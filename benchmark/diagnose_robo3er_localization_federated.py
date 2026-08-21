#!/usr/bin/env python3
"""
FEDERATED, 26-NODE re-run of `diagnose_robo3er_localization.py` --
node-level DIAGNOSIS (root-cause localization) test for B/C/H/K and their
pairwise combinations BK/CK/HK on robo3er, per
`memory/benchmark-policy-federated-only.md` (all benchmark results must
be the federated variant going forward) and the 26-node
`FEATURE_GROUPS=["kinematic_core","actuation"]` reduction documented in
`memory/joint-prototype-federated-results.md` ("Follow-up: removing
cumulative/unbounded-quantity nodes (31 -> 26)").

Built directly on `run_robo3er_forecast_v2_federated.py` (imported as a
module, exactly as the centralized predecessor imports
`run_robo3er_forecast_v2.py`) rather than the centralized script --
signal K/the forecast head has never been trained federated before this
run (`memory/forecast-head-signal-k.md`'s "K has never been run
federated" caveat); this script's federated training loop is that first
run, not just a diagnosis-side change.

Per-signal localization mechanism is UNCHANGED from the centralized
script (see that file's docstring for the full derivation):
  B: argmax_i zscore(d_node)_i                         (native)
  C: argmax_i zscore(resid_struct)_i                    (native)
  K: argmax_i zscore(k_resid)_i                         (native)
  H: argmax_i contribution_i  (Hotelling-T^2 per-node contribution
     decomposition of the Mahalanobis quadratic form, read off
     DeviationCovarianceHead's calib_mu/calib_cov_inv/global_mu/
     global_cov_inv buffers -- no model changes)
  BK/CK/HK: per window, whichever component (B vs K, C vs K, H vs K) has
     the larger own z-scored scalar supplies the argmax node -- the combo
     only ROUTES to a component's own mechanism, per the review doc.

Ground-truth domains are RECOMPUTED for the 26-node set, not copied
verbatim from the 68-node script, because two of the four prefixes/names
the centralized script's domains reference behave differently here:
  - `wheel_ticks_*` is entirely EXCLUDED under `EXCLUDE_NODES_PREFIXES`
    (federated script) -- dropped from cable trapped's indirect domain.
  - `slip_status_is_slipping` (cable trapped's ONLY direct-domain member
    in the centralized script) lives in `feature_groups.STATUS_FLAGS`,
    which is NOT one of this run's active groups (`kinematic_core`,
    `actuation`) -- cable trapped's direct-domain candidate pool is
    THE EMPTY SET at 26 nodes. Every signal's direct-hit rate for cable
    trapped is therefore structurally 0% here, not a mechanism failure --
    reported as such, not hidden.
  stuck's direct domain (any `wheel_*` column) is unaffected: its
  wheel_* members present at 26 nodes (wheel_vels_velocity_left/right,
  wheel_status_current_ma_left/right, wheel_status_pwm_left/right)
  survive the reduction; only wheel_ticks_* is dropped, and that was
  already double-counted as both direct (wheel_*) and part of cable
  trapped's indirect domain at 68 nodes.

Per-client models (5 real robots), each evaluated on its OWN fault
windows using its OWN calibration stats -- results pooled across all
clients into one aggregate hit-rate table per signal per fault type,
matching the centralized script's single-pool aggregation.

Usage:
    python3 diagnose_robo3er_localization_federated.py [--horizon-mult M] [--rounds R] [--local-epochs E] [--out-suffix NAME]
"""
import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT.parent / "checkpoints" / "robo3er"

spec = importlib.util.spec_from_file_location("robo3er_fc_v2_fed", REPO_ROOT / "run_robo3er_forecast_v2_federated.py")
v2f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2f)

# Same "near-constant discrete flag blows up z-score" exclusion as the
# centralized script's RELIABILITY_RATIO convention.
RELIABILITY_RATIO = 0.05

DIRECT_DOMAIN = {
    "stuck": lambda name: name.startswith("wheel_"),
    "cable trapped": lambda name: name == "slip_status_is_slipping",  # not in 26-node set -> always False
}
INDIRECT_DOMAIN = {
    "stuck": lambda name: False,
    "cable trapped": lambda name: (
        name.startswith("wheel_vels_velocity")
        or name.startswith("odom_odo_lintw_") or name.startswith("odom_odo_angtw_")
        or name.startswith("imu_imu_angvel_")
    ),  # wheel_ticks_* dropped: excluded from the 26-node set entirely
}


def classify(name, fault):
    if DIRECT_DOMAIN[fault](name):
        return "direct"
    if INDIRECT_DOMAIN[fault](name):
        return "indirect"
    return "other"


def h_contributions(model, d_node, idx):
    """[Nwin, N] per-node Mahalanobis contribution -- identical readout to
    the centralized script's h_contributions, against this client's own
    DeviationCovarianceHead calibration buffers."""
    cov_head = model.cov_head
    valid = cov_head.calib_valid.cpu().numpy()[idx]
    mu = np.where(valid[:, None], cov_head.calib_mu.cpu().numpy()[idx],
                  cov_head.global_mu.cpu().numpy()[None, :])
    cov_inv = np.where(valid[:, None, None], cov_head.calib_cov_inv.cpu().numpy()[idx],
                        cov_head.global_cov_inv.cpu().numpy()[None, :, :])
    diff = d_node - mu
    weighted = np.einsum("bij,bj->bi", cov_inv, diff)
    contrib = diff * weighted
    return contrib


def zn(x, med, iqr):
    return (x - med) / iqr


def masked_argmax(z, reliable):
    z_masked = np.where(reliable[None, :], z, -np.inf)
    return np.argmax(z_masked, axis=1)


def main(horizon_mult=10, use_forecast_prior=True, rounds=None, local_epochs=None, out_suffix=None):
    torch.manual_seed(v2f.SEED)
    np.random.seed(v2f.SEED)
    forecast_h = v2f.STRIDE * horizon_mult
    rounds = rounds if rounds is not None else v2f.ROUNDS
    local_epochs = local_epochs if local_epochs is not None else v2f.LOCAL_EPOCHS

    print(f"loading robo3er, {v2f.FEATURE_GROUPS} as nodes (26-node set), "
          f"5 federated clients, forecast head horizon_mult={horizon_mult}, "
          f"rounds={rounds}, local_epochs={local_epochs} ...")
    data, targets, label_map, clients, cols, window_robot = v2f.build_clients()
    num_nodes = len(cols)
    print(f"num_nodes={num_nodes}")
    print(f"cols={cols}")
    node_idx = {name: i for i, name in enumerate(cols)}
    prior_edges = [(node_idx[s], node_idx[d]) for s, d in v2f.EDGES_NAMED
                   if s in node_idx and d in node_idx]
    edge_types = [t for (s, d), t in zip(v2f.EDGES_NAMED, v2f.EDGE_TYPES) if s in node_idx and d in node_idx]
    dropped_edges = [(s, d) for s, d in v2f.EDGES_NAMED if s not in node_idx or d not in node_idx]
    if dropped_edges:
        print(f"WARNING: prior edges referencing nodes outside the 26-node set, dropped: {dropped_edges}")
    forecast_prior_edges = prior_edges if use_forecast_prior else None

    scalers = [v2f.fit_client_scaler(data, c.fit_idx) for c in clients]
    models = [v2f.JointPrototypeV31Forecast(
        num_nodes=num_nodes, window_size=v2f.WINDOW_LEN, embed_dim=v2f.EMBED_DIM,
        num_prototypes=v2f.NUM_PROTOTYPES, prior_edges=prior_edges, edge_types=edge_types,
        forecast_h=forecast_h, top_k=v2f.TOP_K, forecast_prior_edges=forecast_prior_edges,
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

    print(f"\nfederated training: {rounds} rounds x {local_epochs} local epochs "
          f"(memory exchange + FedAvg'd encoder/decoder) -- signal K's first-ever federated run")
    for rnd in range(1, rounds + 1):
        fit_counts = []
        for c, model, scaler, (vi, x_in_raw, x_future_raw) in zip(clients, models, scalers, fit_pairs):
            n, t, f = x_in_raw.shape
            x_in = scaler.transform(x_in_raw.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
            n2, t2, f2 = x_future_raw.shape
            x_future = (scaler.transform(x_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                        if n2 > 0 else x_future_raw)
            v2f.train_local(model, x_in, x_future, local_epochs)
            fit_counts.append(len(x_in))

        codebooks = [m.memory.codebook.detach().clone() for m in models]
        usage_counts = [m.memory.usage_count.detach().clone() for m in models]
        P_G, diag = v2f.align_and_split(codebooks, usage_counts, gamma=v2f.GAMMA, delta=v2f.DELTA)
        print(f"  round {rnd}: cluster agreement / usage diag keys={list(diag.keys())}")
        for model, p_g in zip(models, P_G):
            model.memory.load_memory(p_g)
        if v2f.SYNC_ENCODER_DECODER:
            avg = v2f.fedavg_state_dict([m.state_dict() for m in models], fit_counts, prefixes=("encoder.", "decoder."))
            for model in models:
                model.load_state_dict(avg, strict=False)

    print("\ncalibrating typed-head + covariance-head per client ...")
    calib_stats = {}
    cov_min = max(v2f.MIN_PROTO_SAMPLES, num_nodes + 1)
    for c, model, scaler in zip(clients, models, scalers):
        calib_w = v2f.scale_client(data, scaler, c.calib_idx)
        _, d_node_calib_b, _, _, calib_idx_w, calib_r_w, _ = v2f.per_sample_scores(model, calib_w)
        model.typed_head.set_calibration(calib_r_w, calib_idx_w, v2f.NUM_PROTOTYPES, min_samples=v2f.MIN_PROTO_SAMPLES)
        model.cov_head.set_calibration(d_node_calib_b, calib_idx_w, min_samples=cov_min)

        _, d_node_calib, resid_struct_calib, _, calib_idx_b2, _, d_mahal_calib = v2f.per_sample_scores(model, calib_w)

        vi, chains, _ = v2f.build_pairs(c.calib_idx, targets, window_robot, horizon_mult, True, True)
        if len(vi):
            calib_in = v2f.scale_client(data, scaler, vi)
            calib_future_raw = v2f.gather_future(data, chains, v2f.STRIDE)
            n2, t2, f2 = calib_future_raw.shape
            calib_future = scaler.transform(calib_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
            k_resid_calib = v2f.forecast_scores(model, calib_in, calib_future)
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
        d_mahal_med = np.median(d_mahal_calib)
        d_mahal_iqr = max(np.percentile(d_mahal_calib, 75) - np.percentile(d_mahal_calib, 25), 1e-8)

        calib_stats[c.client_id] = dict(
            node_median=node_median, node_iqr=node_iqr, node_reliable=node_reliable,
            struct_median=struct_median, struct_iqr=struct_iqr, struct_reliable=struct_reliable,
            k_median=k_median, k_iqr=k_iqr, k_reliable=k_reliable,
            d_mahal_med=d_mahal_med, d_mahal_iqr=d_mahal_iqr,
        )
        print(f"  {c.robot_name}: reliable nodes B/H={node_reliable.sum()}/{num_nodes}  "
              f"C={struct_reliable.sum()}/{num_nodes}  K={k_reliable.sum()}/{num_nodes}")

    print(f"\n{'fault':<16}{'signal':<8}{'n':<6}{'direct%':<10}{'direct_or_indirect%':<20}")
    pooled = {}  # fault_name -> signal -> list of argmax node names
    for c, model, scaler in zip(clients, models, scalers):
        if not c.fault_idx_by_label:
            continue
        st = calib_stats[c.client_id]
        for label_id, fault_idx in c.fault_idx_by_label.items():
            fault_name = label_map[str(label_id)]
            fault_w = v2f.scale_client(data, scaler, fault_idx)
            _, d_node_f, resid_struct_f, _, idx_f, _, d_mahal_f = v2f.per_sample_scores(model, fault_w)

            z_node_f = zn(d_node_f, st["node_median"], st["node_iqr"])
            z_struct_f = zn(resid_struct_f, st["struct_median"], st["struct_iqr"])
            contrib_h_f = h_contributions(model, d_node_f, idx_f)
            z_mahal_f = zn(d_mahal_f, st["d_mahal_med"], st["d_mahal_iqr"])

            argmax_b = masked_argmax(z_node_f, st["node_reliable"])
            argmax_c = masked_argmax(z_struct_f, st["struct_reliable"])
            argmax_h = masked_argmax(contrib_h_f, st["node_reliable"])

            vi_f, chains_f, mask_f = v2f.build_pairs(fault_idx, targets, window_robot, horizon_mult, False, False)
            if len(vi_f):
                fault_in = v2f.scale_client(data, scaler, vi_f)
                future_f_raw = v2f.gather_future(data, chains_f, v2f.STRIDE)
                n2, t2, f2 = future_f_raw.shape
                fault_future = scaler.transform(future_f_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                k_resid_f = v2f.forecast_scores(model, fault_in, fault_future)
                z_k_f = zn(k_resid_f, st["k_median"], st["k_iqr"])
                argmax_k_paired = masked_argmax(z_k_f, st["k_reliable"])
            else:
                z_k_f = np.zeros((0, num_nodes))
                argmax_k_paired = np.zeros(0, dtype=int)

            def add(fault_name, key, argmax_idx):
                pooled.setdefault(fault_name, {}).setdefault(key, []).extend(cols[i] for i in argmax_idx)

            add(fault_name, "B", argmax_b)
            add(fault_name, "C", argmax_c)
            add(fault_name, "H", argmax_h)
            add(fault_name, "K", argmax_k_paired)

            b_paired = z_node_f.max(axis=1)[mask_f]
            c_paired = z_struct_f.max(axis=1)[mask_f]
            h_paired = z_mahal_f[mask_f]
            k_scalar_paired = z_k_f.max(axis=1) if len(z_k_f) else np.zeros(0)
            argmax_b_paired = argmax_b[mask_f]
            argmax_c_paired = argmax_c[mask_f]
            argmax_h_paired = argmax_h[mask_f]

            def combo_argmax(scalar_a, argmax_a, scalar_b, argmax_b_):
                if len(scalar_a) == 0:
                    return np.zeros(0, dtype=int)
                a_wins = scalar_a >= scalar_b
                return np.where(a_wins, argmax_a, argmax_b_)

            add(fault_name, "BK", combo_argmax(b_paired, argmax_b_paired, k_scalar_paired, argmax_k_paired))
            add(fault_name, "CK", combo_argmax(c_paired, argmax_c_paired, k_scalar_paired, argmax_k_paired))
            add(fault_name, "HK", combo_argmax(h_paired, argmax_h_paired, k_scalar_paired, argmax_k_paired))

    results = {}
    for fault_name, sigs in pooled.items():
        for key, names in sigs.items():
            n = len(names)
            classes = [classify(name, fault_name) for name in names]
            c = Counter(classes)
            direct_pct = 100.0 * c["direct"] / n if n else 0.0
            combined_pct = 100.0 * (c["direct"] + c["indirect"]) / n if n else 0.0
            print(f"{fault_name:<16}{key:<8}{n:<6}{direct_pct:<10.1f}{combined_pct:<20.1f}")
            results.setdefault(fault_name, {})[key] = {
                "n": n, "direct_pct": round(direct_pct, 1), "direct_or_indirect_pct": round(combined_pct, 1),
                "top_nodes": Counter(names).most_common(5),
            }

    results["_config"] = {
        "num_nodes": num_nodes, "cols": cols, "feature_groups": v2f.FEATURE_GROUPS,
        "exclude_nodes_prefixes": v2f.EXCLUDE_NODES_PREFIXES, "horizon_mult": horizon_mult,
        "rounds": rounds, "local_epochs": local_epochs, "sync_encoder_decoder": v2f.SYNC_ENCODER_DECODER,
        "reliability_ratio": RELIABILITY_RATIO,
    }
    suffix = out_suffix if out_suffix is not None else f"h{horizon_mult}"
    out_path = OUT_DIR / f"robo3er_diagnosis_localization_federated26_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-mult", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--out-suffix", type=str, default=None)
    args = parser.parse_args()
    main(horizon_mult=args.horizon_mult, rounds=args.rounds, local_epochs=args.local_epochs, out_suffix=args.out_suffix)
