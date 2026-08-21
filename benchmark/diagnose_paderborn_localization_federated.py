#!/usr/bin/env python3
"""
Node-level DIAGNOSIS (root-cause localization) test for B/C/H/K and their
pairwise combinations BK/CK/HK on Paderborn, federated (per
`memory/benchmark-policy-federated-only.md`), completing the 3-dataset
B/C/K/BK/CK comparison started by `diagnose_robo3er_localization_federated.py`
and `diagnose_sielaff_localization_federated.py`.

Built on `run_paderborn_forecast_v2_federated.py` (imported as a module):
K001-K006 healthy bearings as 6 federated clients, `JointPrototypeV31Forecast`
with the 8 declared torque/speed/force<->vibration/current physics edges,
FedAvg'd encoder/decoder. Every damaged bearing code is evaluated against
EVERY client's personalized model (same cross-client convention as the
detection-AUROC script), at WINDOW level (not the AUROC script's per-file
aggregation -- diagnosis needs the node-level array per sample, which
per-file averaging would blur).

Only 6 nodes exist here (`vibration_1`, `phase_current_1`, `phase_current_2`,
`force`, `speed`, `torque`) -- far fewer than robo3er/Sielaff, so this is
structurally a much easier localization task (random-guess baseline
1/6=16.7% direct). Same per-signal argmax mechanism as the other two
scripts (B/C/K native per-node arrays, H's Hotelling-T^2 contribution
decomposition, BK/CK/HK route to whichever component's own z-scored
scalar wins per window) -- see `diagnose_robo3er_localization_federated.py`
for the full derivation.

Ground truth (all three damage categories -- outer_ring/inner_ring/combined
-- share the SAME domain, since this 6-channel set cannot distinguish
damage location, only that vibration is the affected channel; see
`memory/robo3er-explicit-physics-scoring.md`'s Paderborn section: "bearing
faults manifest as impulsive vibration... does NOT change the mean-value
relationships between torque, speed, and current"):
  direct   = {"vibration_1"}
  indirect = {"phase_current_1", "phase_current_2"}  (motor current
             signature analysis literature documents bearing-fault energy
             leaking into the current spectrum via load fluctuation --
             a real but indirect correlate, not what bearing vibration
             physically is)
  other    = {"force", "speed", "torque"}  (drive-side operating-point
             parameters, not bearing-damage sensors)

Usage:
    python3 diagnose_paderborn_localization_federated.py [--horizon-mult M] [--out-suffix NAME]
"""
import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT.parent / "checkpoints" / "paderborn"

spec = importlib.util.spec_from_file_location("paderborn_fc_v2_fed", REPO_ROOT / "run_paderborn_forecast_v2_federated.py")
v2f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2f)

RELIABILITY_RATIO = 0.05

DIRECT_DOMAIN = {"vibration_1"}
INDIRECT_DOMAIN = {"phase_current_1", "phase_current_2"}


def classify(name):
    if name in DIRECT_DOMAIN:
        return "direct"
    if name in INDIRECT_DOMAIN:
        return "indirect"
    return "other"


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


def window_forecast(model, arr_scaled, horizon_mult, stride):
    windows, file_id, n_positions = v2f.make_windows(arr_scaled)
    n_files = len(arr_scaled)
    vi, chains = v2f.build_chains(n_files, n_positions, horizon_mult)
    if len(vi) == 0:
        return np.zeros((0, model.num_nodes), dtype=np.float32), np.zeros(0, dtype=int)
    x_in = windows[vi]
    x_future = v2f.gather_future(windows, chains, stride)
    k_resid = v2f.forecast_scores(model, x_in, x_future)
    return k_resid, file_id[vi]


def main(horizon_mult=10, use_forecast_prior=True, out_suffix=None):
    torch.manual_seed(v2f.SEED)
    np.random.seed(v2f.SEED)
    num_nodes = len(v2f.NODE_NAMES)
    forecast_h = v2f.WINDOW_STRIDE * horizon_mult
    forecast_prior_edges = v2f.PRIOR_EDGES if use_forecast_prior else None

    print("loading healthy bearings (K001-K006) as 6 federated clients ...")
    clients = []
    for code in v2f.HEALTHY_CODES:
        d = v2f.load_bearing_with_physics(code)
        arr = v2f.stack_nodes(d)
        n = len(arr)
        n_fit = int(n * v2f.FIT_FRACTION)
        n_calib = int(n * v2f.CALIB_FRACTION)
        fit_raw, calib_raw, test_raw = arr[:n_fit], arr[n_fit : n_fit + n_calib], arr[n_fit + n_calib :]
        scaler = StandardScaler().fit(fit_raw.reshape(-1, num_nodes))

        def scale(a, scaler=scaler):
            n_, t_, f_ = a.shape
            return scaler.transform(a.reshape(-1, f_)).reshape(n_, t_, f_).astype(np.float32)

        clients.append({"code": code, "fit": scale(fit_raw), "calib": scale(calib_raw),
                         "test_normal": scale(test_raw), "scaler": scaler})
        print(f"  {code}: n={n} fit={n_fit} calib={n_calib} test_normal={n - n_fit - n_calib}")

    models = [v2f.JointPrototypeV31Forecast(
        num_nodes=num_nodes, window_size=v2f.WINDOW_LEN, embed_dim=v2f.EMBED_DIM,
        num_prototypes=v2f.NUM_PROTOTYPES, prior_edges=v2f.PRIOR_EDGES, edge_types=v2f.EDGE_TYPES,
        forecast_h=forecast_h, top_k=v2f.TOP_K, forecast_prior_edges=forecast_prior_edges,
    ).to(v2f.DEVICE) for _ in clients]

    shared_init = {k: v.clone() for k, v in models[0].state_dict().items() if not k.startswith("memory.")}
    for model in models[1:]:
        model.load_state_dict(shared_init, strict=False)

    print(f"\nfederated training: {v2f.ROUNDS} rounds x {v2f.LOCAL_EPOCHS} local epochs, "
          f"horizon_mult={horizon_mult} ...")
    for rnd in range(1, v2f.ROUNDS + 1):
        fit_counts = []
        for c, model in zip(clients, models):
            fit_windows, fit_file_id, fit_n_pos = v2f.make_windows(c["fit"])
            fit_vi, fit_chains = v2f.build_chains(len(c["fit"]), fit_n_pos, horizon_mult)
            fit_in = fit_windows[fit_vi]
            fit_future = v2f.gather_future(fit_windows, fit_chains, v2f.WINDOW_STRIDE)
            v2f.train_local(model, fit_in, fit_future, v2f.LOCAL_EPOCHS)
            fit_counts.append(len(fit_in))

        codebooks = [m.memory.codebook.detach().clone() for m in models]
        usage_counts = [m.memory.usage_count.detach().clone() for m in models]
        P_G, diag = v2f.align_and_split(codebooks, usage_counts, gamma=v2f.GAMMA, delta=v2f.DELTA)
        print(f"  round {rnd}: shared_clusters={diag.get('num_multi_client_clusters')}")
        for model, p_g in zip(models, P_G):
            model.memory.load_memory(p_g)
        if v2f.SYNC_ENCODER_DECODER:
            avg = v2f.fedavg_state_dict([m.state_dict() for m in models], fit_counts, prefixes=("encoder.", "decoder."))
            for model in models:
                model.load_state_dict(avg, strict=False)

    print("\ncalibrating cov_head per client ...")
    calib_stats = []
    cov_min = max(v2f.MIN_PROTO_SAMPLES, num_nodes + 1)
    for c, model in zip(clients, models):
        calib_windows, _, _ = v2f.make_windows(c["calib"])
        _, d_node_b, resid_struct_b, _, calib_idx_w, _, d_mahal_b = v2f.per_sample_scores(model, calib_windows)
        model.cov_head.set_calibration(d_node_b, calib_idx_w, min_samples=cov_min)
        # recompute d_mahal now that calibration is set
        _, d_node_calib, resid_struct_calib, _, calib_idx_c, _, d_mahal_calib = v2f.per_sample_scores(model, calib_windows)

        k_resid_calib, _ = window_forecast(model, c["calib"], horizon_mult, v2f.WINDOW_STRIDE)

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

        calib_stats.append(dict(
            node_median=node_median, node_iqr=node_iqr, node_reliable=node_reliable,
            struct_median=struct_median, struct_iqr=struct_iqr, struct_reliable=struct_reliable,
            k_median=k_median, k_iqr=k_iqr, k_reliable=k_reliable,
            d_mahal_med=d_mahal_med, d_mahal_iqr=d_mahal_iqr,
        ))
        print(f"  {c['code']}: reliable B/H={node_reliable.sum()}/{num_nodes}  "
              f"C={struct_reliable.sum()}/{num_nodes}  K={k_reliable.sum()}/{num_nodes}")

    print(f"\n{'category':<14}{'signal':<8}{'n':<8}{'direct%':<10}{'direct_or_indirect%':<20}")
    pooled = {}  # category -> signal -> list of node names
    for code in v2f.ALL_DAMAGED_CODES:
        d = v2f.load_bearing_with_physics(code)
        arr = v2f.stack_nodes(d)
        cat = v2f.category_of(code)

        for c, model, st in zip(clients, models, calib_stats):
            arr_scaled = c["scaler"].transform(arr.reshape(-1, num_nodes)).reshape(arr.shape).astype(np.float32)
            windows, file_id, _ = v2f.make_windows(arr_scaled)
            _, d_node_f, resid_struct_f, _, idx_f, _, d_mahal_f = v2f.per_sample_scores(model, windows)

            z_node_f = zn(d_node_f, st["node_median"], st["node_iqr"])
            z_struct_f = zn(resid_struct_f, st["struct_median"], st["struct_iqr"])
            contrib_h_f = h_contributions(model, d_node_f, idx_f)
            z_mahal_win = zn(d_mahal_f, st["d_mahal_med"], st["d_mahal_iqr"])

            argmax_b = masked_argmax(z_node_f, st["node_reliable"])
            argmax_c = masked_argmax(z_struct_f, st["struct_reliable"])
            argmax_h = masked_argmax(contrib_h_f, st["node_reliable"])

            k_resid_f, k_file_id = window_forecast(model, arr_scaled, horizon_mult, v2f.WINDOW_STRIDE)
            if len(k_resid_f):
                z_k_f = zn(k_resid_f, st["k_median"], st["k_iqr"])
                argmax_k = masked_argmax(z_k_f, st["k_reliable"])
            else:
                z_k_f = np.zeros((0, num_nodes))
                argmax_k = np.zeros(0, dtype=int)

            def add(key, argmax_idx):
                pooled.setdefault(cat, {}).setdefault(key, []).extend(v2f.NODE_NAMES[i] for i in argmax_idx)

            add("B", argmax_b)
            add("C", argmax_c)
            add("H", argmax_h)
            add("K", argmax_k)

            # BK/CK/HK: align B/C/H's per-window scalars to the SAME windows K covers
            # (window i has a valid forecast iff i is not in the last horizon_mult
            # positions of its file -- k_file_id gives us which original window index
            # each k_resid row corresponds to via the chain's starting position).
            if len(k_resid_f):
                # k rows correspond to file-relative "vi" positions; recover the same
                # global window indices used for B/C/H by re-deriving vi via build_chains.
                windows_all, file_id_all, n_positions_all = v2f.make_windows(arr_scaled)
                vi_k, _ = v2f.build_chains(len(arr_scaled), n_positions_all, horizon_mult)
                b_at_k = z_node_f.max(axis=1)[vi_k]
                c_at_k = z_struct_f.max(axis=1)[vi_k]
                h_at_k = z_mahal_win[vi_k]
                argmax_b_at_k = argmax_b[vi_k]
                argmax_c_at_k = argmax_c[vi_k]
                argmax_h_at_k = argmax_h[vi_k]
                k_scalar = z_k_f.max(axis=1)

                def combo(scalar_a, argmax_a, scalar_k, argmax_k_):
                    a_wins = scalar_a >= scalar_k
                    return np.where(a_wins, argmax_a, argmax_k_)

                add("BK", combo(b_at_k, argmax_b_at_k, k_scalar, argmax_k))
                add("CK", combo(c_at_k, argmax_c_at_k, k_scalar, argmax_k))
                add("HK", combo(h_at_k, argmax_h_at_k, k_scalar, argmax_k))

    results = {}
    for cat, sigs in pooled.items():
        for key, names in sigs.items():
            n = len(names)
            if n == 0:
                continue
            classes = [classify(name) for name in names]
            c = Counter(classes)
            direct_pct = 100.0 * c["direct"] / n
            combined_pct = 100.0 * (c["direct"] + c["indirect"]) / n
            print(f"{cat:<14}{key:<8}{n:<8}{direct_pct:<10.1f}{combined_pct:<20.1f}")
            results.setdefault(cat, {})[key] = {
                "n": n, "direct_pct": round(direct_pct, 1), "direct_or_indirect_pct": round(combined_pct, 1),
                "top_nodes": Counter(names).most_common(6),
            }

    results["_config"] = {
        "num_nodes": num_nodes, "node_names": v2f.NODE_NAMES, "horizon_mult": horizon_mult,
        "rounds": v2f.ROUNDS, "local_epochs": v2f.LOCAL_EPOCHS, "reliability_ratio": RELIABILITY_RATIO,
        "sync_encoder_decoder": v2f.SYNC_ENCODER_DECODER,
    }
    suffix = out_suffix if out_suffix is not None else f"h{horizon_mult}"
    out_path = OUT_DIR / f"paderborn_diagnosis_localization_federated_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-mult", type=int, default=10)
    parser.add_argument("--out-suffix", type=str, default=None)
    args = parser.parse_args()
    main(horizon_mult=args.horizon_mult, out_suffix=args.out_suffix)
