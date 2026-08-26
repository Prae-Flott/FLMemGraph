#!/usr/bin/env python3
"""
Node-level DIAGNOSIS (root-cause localization) test for B/C/H/K and their
pairwise combinations BK/CK/HK on robo3er -- distinct from the existing
DETECTION AUROC results in memory/forecast-head-signal-k.md's "Pairwise
combination test" section. Those numbers answer "how well does each
score separate normal from fault windows"; this script answers "when a
fault window IS flagged, does the signal's own argmax mechanism point at
the physically correct node," per
`diagnosis_interpretability_review.md`'s section 2 methodology table.

Retrains JointPrototypeV31Forecast identically to
`run_robo3er_bck.py --horizon-mult 10` (the horizon at which HK
peaked, 0.901 mean AUROC) rather than reloading a checkpoint, so the
calibration stats (per-node median/IQR, per-prototype mu/cov_inv) are
recomputed consistently in this script.

Per-signal localization mechanism (argmax over nodes of a per-window
node-level array), exactly as diagnosis_interpretability_review.md
section 2 lays out -- NEW CODE ONLY for H, which has no native per-node
output:
  B: argmax_i zscore(d_node)_i                         (native)
  C: argmax_i zscore(resid_struct)_i                    (native)
  K: argmax_i zscore(k_resid)_i                         (native)
  H: argmax_i contribution_i, where
       diff = d_node - mu*(idx)                         [N]
       contribution_i = diff_i * sum_j cov_inv[i,j]*diff_j
     i.e. the per-node term-wise expansion of the Mahalanobis quadratic
     form diff @ cov_inv @ diff = sum_i contribution_i (standard
     Hotelling-T^2 contribution-plot decomposition; see review doc
     section 3, method 1). Read-only math on existing DeviationCovarianceHead
     buffers (calib_mu/calib_cov_inv/global_mu/global_cov_inv) -- no model
     changes.

Combination diagnosis (BK/CK/HK): per window, compare the two components'
OWN z-scored scalar (B vs K, C vs K, H vs K); report whichever component
supplied the max's own argmax node, per the review doc's "the combo
doesn't invent new diagnostic info, it only routes to the winning
component's own mechanism" conclusion.

Ground truth: robo3er's 2 fault types have direct-sensor domains from
src/robo3er/kinematics.py + src/robo3er/dataset.py's module docstrings:
  stuck        -> locked wheel: any wheel_* column (direct)
  cable trapped -> slip_status_is_slipping is the clearest direct
                   signature (memory/feature-purification-audit.md); the
                   wheel<->odom/imu kinematic residual (wheel_ticks,
                   odom_odo_lintw_*/angtw_*, imu_imu_angvel_*,
                   wheel_vels_velocity_*) is the INDIRECT/kinematic
                   domain per kinematics.py's own stated rationale for
                   why this fault breaks that specific relationship.
Reported as top-1 direct-hit rate and top-1 direct-or-indirect rate,
analogous to analyze_sielaff_red_feature_attribution.py's plausible/
indirect split.

Usage:
    python3 diagnose_robo3er_localization.py [--horizon-mult M]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import importlib.util
spec = importlib.util.spec_from_file_location("robo3er_bck", REPO_ROOT / "run_robo3er_bck.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

OUT_DIR = REPO_ROOT.parent / "checkpoints" / "robo3er"

# Nodes whose calib-split IQR collapses toward the zscore() floor (near-
# constant discrete flags like dock_status_dock_visible, ir_opcode_sensor)
# blow up to huge z-scores from tiny absolute noise and dominate every
# signal's argmax without carrying real localization information -- the
# exact "z-in-the-thousands" failure mode documented in
# src/robo3er/dataset.py's module docstring point 2. Excluding them from
# the argmax pool (not from detection scoring, which is untouched) is the
# same RELIABILITY_RATIO convention benchmark/run_sielaff_red_v2_1_federated.py
# already uses for the same reason.
RELIABILITY_RATIO = 0.05

DIRECT_DOMAIN = {
    "stuck": lambda name: name.startswith("wheel_"),
    "cable trapped": lambda name: name == "slip_status_is_slipping",
}
INDIRECT_DOMAIN = {
    "stuck": lambda name: False,  # wheel_* already covers the direct mechanism fully
    "cable trapped": lambda name: (
        name.startswith("wheel_ticks_") or name.startswith("wheel_vels_velocity")
        or name.startswith("odom_odo_lintw_") or name.startswith("odom_odo_angtw_")
        or name.startswith("imu_imu_angvel_")
    ),
}


def classify(name, fault):
    if DIRECT_DOMAIN[fault](name):
        return "direct"
    if INDIRECT_DOMAIN[fault](name):
        return "indirect"
    return "other"


def h_contributions(model, d_node, idx):
    """[Nwin, N] per-node Mahalanobis contribution, term-wise expansion of
    diff @ cov_inv @ diff. Pure numpy readout of DeviationCovarianceHead's
    existing calibration buffers -- no model modification."""
    cov_head = model.cov_head
    valid = cov_head.calib_valid.cpu().numpy()[idx]  # [Nwin]
    mu = np.where(valid[:, None], cov_head.calib_mu.cpu().numpy()[idx],
                  cov_head.global_mu.cpu().numpy()[None, :])
    cov_inv = np.where(valid[:, None, None], cov_head.calib_cov_inv.cpu().numpy()[idx],
                        cov_head.global_cov_inv.cpu().numpy()[None, :, :])
    diff = d_node - mu  # [Nwin, N]
    weighted = np.einsum("bij,bj->bi", cov_inv, diff)  # [Nwin, N] = sum_j cov_inv_ij diff_j
    contrib = diff * weighted  # [Nwin, N], sums to the scalar Mahalanobis distance
    return contrib


def main(horizon_mult=10):
    forecast_h = v2.STRIDE * horizon_mult
    torch.manual_seed(v2.SEED)
    np.random.seed(v2.SEED)

    print(f"loading robo3er, horizon_mult={horizon_mult} (HK's detection-AUROC peak) ...")
    data, targets, cols, label_map, _ = v2.load_robo3er(drop_dead=True)
    num_nodes = len(cols)
    node_idx = {name: i for i, name in enumerate(cols)}
    prior_edges = [(node_idx[s], node_idx[d]) for s, d in v2.EDGES_NAMED]

    window_robot = v2.build_window_robot_map(len(targets))
    fit_idx, calib_idx, test_normal_idx = v2.split_normal(targets)
    scaler = v2.fit_scaler(data, fit_idx)
    data_scaled = v2.scale(data, scaler)

    def make_pairs(idx_arr, require_next_normal, require_same_split):
        vi, chains, mask = v2.build_pairs(idx_arr, targets, window_robot, horizon_mult,
                                           require_next_normal, require_same_split)
        x_in = data_scaled[vi]
        x_future = v2.gather_future(data_scaled, chains, v2.STRIDE)
        return x_in, x_future, mask, vi

    fit_in, fit_future, _, _ = make_pairs(fit_idx, True, True)
    calib_in, calib_future, _, _ = make_pairs(calib_idx, True, True)

    fault_pairs = {}
    for label_id_str, name in label_map.items():
        label_id = int(label_id_str)
        if label_id == 0:
            continue
        fault_idx = np.where(targets == label_id)[0]
        if len(fault_idx) == 0:
            continue
        f_in, f_future, f_mask, _ = make_pairs(fault_idx, False, False)
        fault_pairs[name] = (f_in, f_future, f_mask)

    fit_full = data_scaled[fit_idx]
    calib_full = data_scaled[calib_idx]
    fault_full = {name: data_scaled[np.where(targets == int(lid))[0]]
                  for lid, name in label_map.items() if int(lid) != 0}

    print("training JointPrototypeV31Forecast (identical to run_robo3er_bck.py) ...")
    model = v2.JointPrototypeV31Forecast(
        num_nodes=num_nodes, window_size=v2.WINDOW_LEN, embed_dim=v2.EMBED_DIM,
        num_prototypes=v2.NUM_PROTOTYPES, prior_edges=prior_edges, edge_types=v2.EDGE_TYPES,
        forecast_h=forecast_h, top_k=v2.TOP_K, forecast_prior_edges=prior_edges,
    ).to(v2.DEVICE)
    model = v2.train(model, fit_in, fit_future, calib_full)

    print("calibrating typed-head and covariance-head from calib split ...")
    _, d_node_calib_b, _, _, calib_idx_b, calib_r, _ = v2.per_sample_scores(model, calib_full)
    model.typed_head.set_calibration(calib_r, calib_idx_b, v2.NUM_PROTOTYPES, min_samples=v2.MIN_PROTO_SAMPLES)
    cov_min = max(v2.MIN_PROTO_SAMPLES, num_nodes + 1)
    model.cov_head.set_calibration(d_node_calib_b, calib_idx_b, min_samples=cov_min)

    _, d_node_calib, resid_struct_calib, _, _, _, _ = v2.per_sample_scores(model, calib_full)
    k_resid_calib = v2.forecast_scores(model, calib_in, calib_future)

    node_median = np.median(d_node_calib, axis=0)
    node_q75, node_q25 = np.percentile(d_node_calib, [75, 25], axis=0)
    node_iqr = np.maximum(node_q75 - node_q25, 1e-8)
    struct_median = np.median(resid_struct_calib, axis=0)
    struct_q75, struct_q25 = np.percentile(resid_struct_calib, [75, 25], axis=0)
    struct_iqr = np.maximum(struct_q75 - struct_q25, 1e-8)
    k_median = np.median(k_resid_calib, axis=0)
    k_q75, k_q25 = np.percentile(k_resid_calib, [75, 25], axis=0)
    k_iqr = np.maximum(k_q75 - k_q25, 1e-8)

    node_reliable = node_iqr >= RELIABILITY_RATIO * np.median(node_iqr)
    struct_reliable = struct_iqr >= RELIABILITY_RATIO * np.median(struct_iqr)
    k_reliable = k_iqr >= RELIABILITY_RATIO * np.median(k_iqr)
    print(f"reliable nodes: B/H={node_reliable.sum()}/{num_nodes}  "
          f"C={struct_reliable.sum()}/{num_nodes}  K={k_reliable.sum()}/{num_nodes}")

    def masked_argmax(z, reliable):
        z_masked = np.where(reliable[None, :], z, -np.inf)
        return np.argmax(z_masked, axis=1)

    # scalar calib refs for H/B/C/K's window-level z-score (combo winner selection)
    _, _, _, _, _, _, d_mahal_calib = v2.per_sample_scores(model, calib_full)

    def zn(x, med, iqr):
        return (x - med) / iqr

    results = {}
    print(f"\n{'fault':<16}{'signal':<8}{'n':<6}{'direct%':<10}{'direct_or_indirect%':<20}")
    for fault_name, arr in fault_full.items():
        f_in, f_future, f_mask = fault_pairs[fault_name]
        d_proto_f, d_node_f, resid_struct_f, resid_phys_f, idx_f, r_edge_f, d_mahal_f = \
            v2.per_sample_scores(model, arr)
        k_resid_f = v2.forecast_scores(model, f_in, f_future)  # [n_paired, N]

        z_node_f = zn(d_node_f, node_median, node_iqr)              # [n, N]
        z_struct_f = zn(resid_struct_f, struct_median, struct_iqr)  # [n, N]
        z_k_f = zn(k_resid_f, k_median, k_iqr)                      # [n_paired, N]
        contrib_h_f = h_contributions(model, d_node_f, idx_f)       # [n, N]

        z_mahal_f = zn(d_mahal_f, np.median(d_mahal_calib), max(
            np.percentile(d_mahal_calib, 75) - np.percentile(d_mahal_calib, 25), 1e-8))

        argmax_b = masked_argmax(z_node_f, node_reliable)
        argmax_c = masked_argmax(z_struct_f, struct_reliable)
        argmax_h = masked_argmax(contrib_h_f, node_reliable)  # H reads d_node, same reliability pool as B
        argmax_k_paired = masked_argmax(z_k_f, k_reliable)    # aligned to f_mask==True rows only

        def score_signal(key, argmax_idx, n):
            classes = [classify(cols[i], fault_name) for i in argmax_idx]
            c = Counter(classes)
            direct_pct = 100.0 * c["direct"] / n
            combined_pct = 100.0 * (c["direct"] + c["indirect"]) / n
            print(f"{fault_name:<16}{key:<8}{n:<6}{direct_pct:<10.1f}{combined_pct:<20.1f}")
            results.setdefault(fault_name, {})[key] = {
                "n": n, "direct_pct": round(direct_pct, 1), "direct_or_indirect_pct": round(combined_pct, 1),
                "top_nodes": Counter(cols[i] for i in argmax_idx).most_common(5),
            }

        score_signal("B", argmax_b, len(argmax_b))
        score_signal("C", argmax_c, len(argmax_c))
        score_signal("H", argmax_h, len(argmax_h))

        # K and any combo involving K only cover the paired subset (f_mask)
        n_paired = int(f_mask.sum())
        score_signal("K", argmax_k_paired, n_paired)

        b_paired = z_node_f.max(axis=1)[f_mask]
        c_paired = z_struct_f.max(axis=1)[f_mask]
        h_paired = z_mahal_f[f_mask]
        k_scalar_paired = z_k_f.max(axis=1)
        argmax_b_paired = argmax_b[f_mask]
        argmax_c_paired = argmax_c[f_mask]
        argmax_h_paired = argmax_h[f_mask]

        def combo_argmax(scalar_a, argmax_a, scalar_b, argmax_b_):
            a_wins = scalar_a >= scalar_b
            return np.where(a_wins, argmax_a, argmax_b_)

        score_signal("BK", combo_argmax(b_paired, argmax_b_paired, k_scalar_paired, argmax_k_paired), n_paired)
        score_signal("CK", combo_argmax(c_paired, argmax_c_paired, k_scalar_paired, argmax_k_paired), n_paired)
        score_signal("HK", combo_argmax(h_paired, argmax_h_paired, k_scalar_paired, argmax_k_paired), n_paired)

    out_path = OUT_DIR / f"robo3er_diagnosis_localization_h{horizon_mult}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-mult", type=int, default=10)
    args = parser.parse_args()
    main(horizon_mult=args.horizon_mult)
