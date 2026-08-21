#!/usr/bin/env python3
"""
Joint Prototype Memory V3.1 + ForecastHead (signal K), on Paderborn, per
`docs/shared_forecast_head_proposal.md`'s v2 cross-window pairing route
(same convention as `run_robo3er_forecast_v2.py` -- see that script's
docstring for the full rationale) plus the pairwise `BK`/`CK`/`HK`
combinations from `memory/forecast-head-signal-k.md`'s combination test.

**Pairing is per-FILE, not per-robot** -- Paderborn has no robot-map
equivalent; `run_paderborn_v3_1.py`'s `make_windows()` already slides
non-overlapping-by-`WINDOW_STRIDE` windows WITHIN each measurement file
independently (`file_id` output), so "does window i+1..i+M exist within
the SAME file" is the only pairing condition needed -- no cross-file
leakage risk, and no separate normal/fault-boundary check like robo3er's
`targets[i+1]==0`, since every FILE here is entirely one class (healthy
fit/calib/test_normal files vs. one specific damaged-bearing code's
files) by construction of `run_paderborn_v3_1.py`'s file-level (not
window-level) train/calib/test split.

Forecast target is, as in robo3er v2, the non-overlapping tail segment
of the M-th successor window (or the concatenation of M non-overlapping
`WINDOW_STRIDE`-length tail segments for `horizon_mult > 1`), NOT a
repeat of overlapping window content.

Scoring stays PER-FILE (not per-window), matching `run_paderborn_v3_1.py`'s
`per_file_scores` convention -- K is averaged across all a file's valid
paired windows (a handful of tail positions per file, `horizon_mult` of
`n_positions_per_file`, are dropped for lack of a valid pair; every file
still has plenty of others to average over: `n_positions_per_file - horizon_mult`
positions).

Usage:
    python3 run_paderborn_forecast_v2.py [--horizon-mult M] [--no-forecast-prior] [--out-suffix NAME]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_prototype_model import JointPrototypeV31Forecast  # noqa: E402
from datasets.paderborn_adapter import (  # noqa: E402
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
EDGE_TYPES = [
    "nonlinear", "nonlinear",
    "nonlinear", "nonlinear",
    "proportional", "proportional",
    "proportional", "proportional",
]
PRIOR_EDGES = [(NODE_IDX[s], NODE_IDX[d]) for s, d in EDGES_NAMED]

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15
WINDOW_LEN = 64
WINDOW_STRIDE = 32
BATCH_SIZE = 256
EPOCHS = 12
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 16
TOP_K = 5
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_TYPED = 0.5
LAMBDA_FORECAST = 0.5
MIN_PROTO_SAMPLES = 5
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
    """File-major, position-minor window ordering (matches `make_windows`):
    global index gi -> file gi//n_positions, local position gi%n_positions.
    Returns (valid_i, chains) -- valid_i global indices with room for a
    full `horizon_mult`-step chain WITHIN the same file; chains[k] is the
    list of the `horizon_mult` successor global indices for valid_i[k]."""
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


def agg_by_file(x, file_id, n_files):
    out = np.zeros((n_files,) + x.shape[1:], dtype=np.float64)
    counts = np.zeros(n_files, dtype=np.int64)
    np.add.at(out, file_id, x)
    np.add.at(counts, file_id, 1)
    shape = (n_files,) + (1,) * (x.ndim - 1)
    return (out / counts.reshape(shape)).astype(np.float32)


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


def per_file_scores(model, arr, batch_size=BATCH_SIZE):
    windows, file_id, _ = make_windows(arr)
    d_proto, d_node, resid_struct, resid_phys, _idx, _r, d_mahal = per_sample_scores(model, windows, batch_size)
    n = len(arr)
    return (agg_by_file(d_proto, file_id, n), agg_by_file(d_node, file_id, n),
            agg_by_file(resid_struct, file_id, n), agg_by_file(resid_phys, file_id, n),
            agg_by_file(d_mahal, file_id, n))


@torch.no_grad()
def forecast_scores(model, x_in, x_future, batch_size=BATCH_SIZE):
    model.eval()
    k_resid_all = []
    for i in range(0, len(x_in), batch_size):
        xb = torch.from_numpy(x_in[i : i + batch_size]).to(DEVICE)
        fb = torch.from_numpy(x_future[i : i + batch_size]).to(DEVICE)
        out = model(xb, training_mode=False, x_future=fb)
        k_resid_all.append(out["k_resid"].cpu().numpy())
    return np.concatenate(k_resid_all)


def per_file_forecast_scores(model, arr, horizon_mult, stride, batch_size=BATCH_SIZE):
    windows, file_id, n_positions = make_windows(arr)
    n_files = len(arr)
    vi, chains = build_chains(n_files, n_positions, horizon_mult)
    x_in = windows[vi]
    x_future = gather_future(windows, chains, stride)
    k_resid = forecast_scores(model, x_in, x_future, batch_size)  # [len(vi), N]
    return agg_by_file(k_resid, file_id[vi], n_files)


def train(model, fit_arr, calib_arr, horizon_mult):
    torch.manual_seed(SEED)
    fit_windows, fit_file_id, fit_n_pos = make_windows(fit_arr)
    fit_vi, fit_chains = build_chains(len(fit_arr), fit_n_pos, horizon_mult)
    fit_in = fit_windows[fit_vi]
    fit_future = gather_future(fit_windows, fit_chains, WINDOW_STRIDE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fit_in), torch.from_numpy(fit_future)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    best_calib_mse, best_state = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        fit_loss = 0.0
        for batch, future in loader:
            batch, future = batch.to(DEVICE), future.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch, training_mode=True, x_future=future)
            l_pred = torch.nn.functional.mse_loss(out["x_hat"], batch)
            l_me, l_mc = model.memory.commitment_losses(out["z"], out["p_star"])
            l_edge = out["resid_struct"].mean()
            l_typed = out["r_edge"].mean()
            l_forecast = out["k_resid"].mean()
            loss = (l_pred + BETA * l_me + l_mc + LAMBDA_EDGE * l_edge
                    + LAMBDA_TYPED * l_typed + LAMBDA_FORECAST * l_forecast)
            loss.backward()
            optimizer.step()
            fit_loss += loss.item() * batch.size(0)
        fit_loss /= len(fit_in)

        d_proto, _, _, _, _ = per_file_scores(model, calib_arr)
        calib_mse = float(d_proto.mean())
        print(f"  epoch {epoch:2d}  fit_loss={fit_loss:.6f}  calib_d_proto_mean={calib_mse:.6f}  "
              f"codebook_util={model.memory.codebook_utilization():.2f}  "
              f"prior_bias_strength={model.edge_head.prior_bias_strength.item():.3f}  "
              f"typed_temp={model.typed_head.log_temperature.exp().item():.3f}  "
              f"forecast_prior_bias={model.forecast_head.prior_bias_strength.item():.3f}")
        if calib_mse < best_calib_mse:
            best_calib_mse = calib_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def zscore(x, calib_x):
    median = np.median(calib_x, axis=0)
    q75, q25 = np.percentile(calib_x, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    return (x - median) / iqr


def main(horizon_mult=10, use_forecast_prior=True, out_suffix=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    num_nodes = len(NODE_NAMES)
    forecast_prior_edges = PRIOR_EDGES if use_forecast_prior else None

    print("loading healthy bearings (K001-K006), 6 channels ...")
    healthy_data = {code: load_bearing_with_physics(code) for code in HEALTHY_CODES}

    fit_list, calib_list, test_list = [], [], []
    for code, d in healthy_data.items():
        n = len(d["vibration_1"])
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        arr = stack_nodes(d)
        fit_list.append(arr[:n_fit])
        calib_list.append(arr[n_fit : n_fit + n_calib])
        test_list.append(arr[n_fit + n_calib :])
        print(f"  {code}: n={n} fit={n_fit} calib={n_calib} test_normal={n - n_fit - n_calib}")

    fit_arr = np.concatenate(fit_list, axis=0)
    calib_arr = np.concatenate(calib_list, axis=0)
    test_normal_arr = np.concatenate(test_list, axis=0)

    scaler = StandardScaler().fit(fit_arr.reshape(-1, num_nodes))

    def scale(a):
        n, t, f = a.shape
        return scaler.transform(a.reshape(-1, f)).reshape(n, t, f).astype(np.float32)

    fit_scaled, calib_scaled, test_normal_scaled = scale(fit_arr), scale(calib_arr), scale(test_normal_arr)

    forecast_h = WINDOW_STRIDE * horizon_mult
    print(f"\ntraining JointPrototypeV31Forecast (nodes={num_nodes}, prior_edges={len(PRIOR_EDGES)}, "
          f"edge_types={EDGE_TYPES}, top_k={TOP_K}, M={NUM_PROTOTYPES}, embed_dim={EMBED_DIM}, "
          f"horizon_mult={horizon_mult}, forecast_h={forecast_h}, use_forecast_prior={use_forecast_prior}) ...")
    model = JointPrototypeV31Forecast(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=EMBED_DIM,
                                       num_prototypes=NUM_PROTOTYPES, prior_edges=PRIOR_EDGES,
                                       edge_types=EDGE_TYPES, forecast_h=forecast_h, top_k=TOP_K,
                                       forecast_prior_edges=forecast_prior_edges).to(DEVICE)
    model = train(model, fit_scaled, calib_scaled, horizon_mult)

    print("\nStage B: estimating prototype-conditioned statistics from calib split ...")
    calib_windows, _, _ = make_windows(calib_scaled)
    _, d_node_calib_b, _, _, calib_idx_per_window, calib_r_edge_per_window, _ = per_sample_scores(model, calib_windows)
    model.typed_head.set_calibration(calib_r_edge_per_window, calib_idx_per_window, NUM_PROTOTYPES,
                                      min_samples=MIN_PROTO_SAMPLES)
    n_valid_proto = int(model.typed_head.calib_valid_proto.sum())
    print(f"  typed_head: {n_valid_proto}/{NUM_PROTOTYPES} prototypes had >= {MIN_PROTO_SAMPLES} calib windows")
    cov_min = max(MIN_PROTO_SAMPLES, num_nodes + 1)
    model.cov_head.set_calibration(d_node_calib_b, calib_idx_per_window, min_samples=cov_min)
    n_valid_cov = int(model.cov_head.calib_valid.sum())
    print(f"  cov_head:   {n_valid_cov}/{NUM_PROTOTYPES} prototypes with >= {cov_min} calib windows")

    print("\ncalibrating from healthy calib split ...")
    d_proto_calib, d_node_calib, resid_struct_calib, resid_phys_calib, d_mahal_calib = per_file_scores(model, calib_scaled)
    d_proto_normal, d_node_normal, resid_struct_normal, resid_phys_normal, d_mahal_normal = per_file_scores(model, test_normal_scaled)
    k_resid_calib = per_file_forecast_scores(model, calib_scaled, horizon_mult, WINDOW_STRIDE)
    k_resid_normal = per_file_forecast_scores(model, test_normal_scaled, horizon_mult, WINDOW_STRIDE)

    z_node_normal = zscore(d_node_normal, d_node_calib)
    z_struct_normal = zscore(resid_struct_normal, resid_struct_calib)
    z_phys_normal = zscore(resid_phys_normal, resid_phys_calib)
    z_mahal_normal = zscore(d_mahal_normal[:, None], d_mahal_calib[:, None])[:, 0]
    z_forecast_normal = zscore(k_resid_normal, k_resid_calib)

    def base_scores(z_node, z_struct, z_phys, z_mahal):
        b = z_node.max(axis=1)
        c = z_struct.max(axis=1)
        e = z_phys.max(axis=1)
        h = z_mahal
        return {
            "B_node_max": b, "C_struct_max": c, "E_phys_max": e,
            "F_v3_max": np.max(np.stack([b, c, e], axis=1), axis=1),
            "H_cov_mahal": h,
            "I_node_cov_max": np.maximum(b, h),
            "J_v3_cov_max": np.max(np.stack([b, c, e, h], axis=1), axis=1),
        }

    def add_forecast_scores(base, z_forecast):
        k = z_forecast.max(axis=1)
        b, c, h = base["B_node_max"], base["C_struct_max"], base["H_cov_mahal"]
        out = dict(base)
        out["K_forecast_max"] = k
        out["BK_max"] = np.maximum(b, k)
        out["CK_max"] = np.maximum(c, k)
        out["HK_max"] = np.maximum(h, k)
        return out

    scores_normal_base = base_scores(z_node_normal, z_struct_normal, z_phys_normal, z_mahal_normal)
    scores_normal = add_forecast_scores(scores_normal_base, z_forecast_normal)

    print("\nscoring damaged bearings ...")
    report = {"config": {"embed_dim": EMBED_DIM, "num_prototypes": NUM_PROTOTYPES, "top_k": TOP_K,
                          "window_len": WINDOW_LEN, "stride": WINDOW_STRIDE, "horizon_mult": horizon_mult,
                          "forecast_h": forecast_h, "epochs": EPOCHS, "prior_edges": EDGES_NAMED,
                          "edge_types": EDGE_TYPES, "use_forecast_prior": use_forecast_prior,
                          "min_proto_samples": MIN_PROTO_SAMPLES},
              "nodes": NODE_NAMES, "final_prior_bias_strength": model.edge_head.prior_bias_strength.item(),
              "final_typed_temperature": model.typed_head.log_temperature.exp().item(),
              "final_forecast_prior_bias_strength": model.forecast_head.prior_bias_strength.item(),
              "n_valid_prototypes_for_typed_calib": n_valid_proto,
              "bearings": {}}
    rows = {k: [] for k in scores_normal}
    for code in ALL_DAMAGED_CODES:
        d = load_bearing_with_physics(code)
        arr = scale(stack_nodes(d))
        d_proto_f, d_node_f, resid_struct_f, resid_phys_f, d_mahal_f = per_file_scores(model, arr)
        k_resid_f = per_file_forecast_scores(model, arr, horizon_mult, WINDOW_STRIDE)
        z_node_f = zscore(d_node_f, d_node_calib)
        z_struct_f = zscore(resid_struct_f, resid_struct_calib)
        z_phys_f = zscore(resid_phys_f, resid_phys_calib)
        z_mahal_f = zscore(d_mahal_f[:, None], d_mahal_calib[:, None])[:, 0]
        z_forecast_f = zscore(k_resid_f, k_resid_calib)
        scores_fault_base = base_scores(z_node_f, z_struct_f, z_phys_f, z_mahal_f)
        scores_fault = add_forecast_scores(scores_fault_base, z_forecast_f)

        cat, origin = category_of(code), damage_origin_of(code)
        aurocs = {}
        for key in scores_normal:
            labels = np.concatenate([np.zeros(len(scores_normal[key])), np.ones(len(scores_fault[key]))])
            scores = np.concatenate([scores_normal[key], scores_fault[key]])
            auroc = float(sk_metrics.roc_auc_score(labels, scores))
            aurocs[key] = auroc
            rows[key].append((code, cat, origin, auroc))
        report["bearings"][code] = {"category": cat, "origin": origin, "n": len(arr), **aurocs}
        print(f"  {code:<6}{cat:<12}{origin:<11}" + "  ".join(f"{k}={v:.3f}" for k, v in aurocs.items()))

    print(f"\n{'method':<24}{'mean AUROC (all 26)':>22}")
    summary = {}
    for key in scores_normal:
        mean_auroc = float(np.mean([r[3] for r in rows[key]]))
        summary[key] = mean_auroc
        print(f"{key:<24}{mean_auroc:>22.3f}")
    print(f"\n{'method':<24}{'outer_ring':>12}{'inner_ring':>12}{'combined':>12}")
    category_summary = {}
    for key in scores_normal:
        cat_means = {}
        for cat in ["outer_ring", "inner_ring", "combined"]:
            subset = [r[3] for r in rows[key] if r[1] == cat]
            cat_means[cat] = float(np.mean(subset)) if subset else None
        category_summary[key] = cat_means
        print(f"{key:<24}" + "".join(f"{cat_means[c]:>12.3f}" for c in ["outer_ring", "inner_ring", "combined"]))

    report["summary_mean_auroc"] = summary
    report["category_summary"] = category_summary
    suffix = out_suffix if out_suffix is not None else f"forecast_v2_h{horizon_mult}{'_noprior' if not use_forecast_prior else ''}"
    report_path = OUT_DIR / f"paderborn_{suffix}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "report": report},
               OUT_DIR / f"paderborn_{suffix}.pth")
    print(f"\nsaved -> {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-mult", type=int, default=10)
    parser.add_argument("--no-forecast-prior", action="store_true")
    parser.add_argument("--out-suffix", type=str, default=None)
    args = parser.parse_args()
    main(horizon_mult=args.horizon_mult, use_forecast_prior=not args.no_forecast_prior,
         out_suffix=args.out_suffix)
