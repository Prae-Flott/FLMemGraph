#!/usr/bin/env python3
"""
Joint Prototype Memory V2.1 + ForecastHead (signal K), on Sielaff, per
`docs/shared_forecast_head_proposal.md`'s v2 cross-window pairing route
(same convention as `run_robo3er_forecast_v2.py` -- see that script's
docstring for the full pairing rationale) plus the pairwise `BK`/`CK`/`HK`
combinations from `memory/forecast-head-signal-k.md`'s combination test.

Uses `JointPrototypeV21Forecast` (no declared physics edges/typed head --
Sielaff has none, matching `run_sielaff_v2_1.py`'s `prior_edges=None`).

**Pairing is per-MACHINE**, via `data/sielaff/partition.pkl`'s
`data_indices` (10 machines) -- the exact same mechanism as robo3er's
`partition.pkl`-based `robot_map` in `run_robo3er_forecast_v2.py`. Same
three validity conditions: same machine, next window normal-labeled +
same split (for fit/calib/test_normal), or just same machine (for fault
evaluation).

`window_size=8`, `stride=2` (from `data/sielaff/metadata.json`) -- much
smaller than robo3er/Paderborn's windows, so `horizon_mult=10` means
`forecast_h=20`, 2.5x the input window's own length. Reuses
`run_sielaff_v2_1.py`'s `RELIABILITY_RATIO` sparse-feature masking,
extended to the forecast signal too (a sparsely-reported node's near-zero
calib IQR would blow up K's z-score the same way it does for B/C).

Usage:
    python3 run_sielaff_forecast_v2.py [--horizon-mult M] [--no-forecast-prior] [--out-suffix NAME]
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
from joint_prototype_model import JointPrototypeV21Forecast  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "sielaff"
OUT_DIR = REPO_ROOT / "checkpoints" / "sielaff"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIT_FRACTION = 0.70
CALIB_FRACTION = 0.15
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 16
WINDOW_LEN = 8
STRIDE = 2
TOP_K = 10
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_FORECAST = 0.5
RELIABILITY_RATIO = 0.05
MIN_PROTO_SAMPLES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_machine_index_sets():
    with open(DATA_DIR / "partition.pkl", "rb") as f:
        partition = pickle.load(f)
    machine_indices = []
    for client_id in range(partition["separation"]["total"]):
        d = partition["data_indices"][client_id]
        idx = np.concatenate([d["train"], d["val"], d["test"]]).astype(np.int64)
        machine_indices.append(np.sort(idx))
    return machine_indices


def load_and_split():
    data = np.load(DATA_DIR / "data.npy")
    targets = np.load(DATA_DIR / "targets.npy")
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    cols = meta["feature_columns"]
    label_map = meta["class_names"]

    machine_indices = load_machine_index_sets()
    window_machine = np.full(len(targets), -1, dtype=int)
    for m, idx in enumerate(machine_indices):
        window_machine[idx] = m

    fit_idx, calib_idx, test_normal_idx = [], [], []
    for idx in machine_indices:
        normal_idx = idx[targets[idx] == 0]
        n = len(normal_idx)
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        fit_idx.append(normal_idx[:n_fit])
        calib_idx.append(normal_idx[n_fit : n_fit + n_calib])
        test_normal_idx.append(normal_idx[n_fit + n_calib :])

    fit_idx = np.concatenate(fit_idx)
    calib_idx = np.concatenate(calib_idx)
    test_normal_idx = np.concatenate(test_normal_idx)
    return data, targets, cols, label_map, fit_idx, calib_idx, test_normal_idx, window_machine


def fit_scaler(data, fit_idx):
    scaler = StandardScaler()
    scaler.fit(data[fit_idx].reshape(-1, data.shape[-1]))
    return scaler


def scale(windows, scaler):
    n, t, f = windows.shape
    return scaler.transform(windows.reshape(-1, f)).reshape(n, t, f).astype(np.float32)


def build_pairs(idx_arr, targets, window_machine, horizon_mult, require_next_normal, require_same_split=False):
    """Same three validity conditions as `run_robo3er_forecast_v2.py`'s
    `build_pairs` -- see that function's docstring."""
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


def gather_future(data_scaled, chains, stride):
    if len(chains) == 0:
        return np.zeros((0, 0, data_scaled.shape[-1]), dtype=data_scaled.dtype)
    segments = [data_scaled[[c[k] for c in chains]][:, -stride:, :] for k in range(len(chains[0]))]
    return np.concatenate(segments, axis=1)


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
    k_resid_all = []
    for i in range(0, len(x_in), batch_size):
        xb = torch.from_numpy(x_in[i : i + batch_size]).to(DEVICE)
        fb = torch.from_numpy(x_future[i : i + batch_size]).to(DEVICE)
        out = model(xb, training_mode=False, x_future=fb)
        k_resid_all.append(out["k_resid"].cpu().numpy())
    return np.concatenate(k_resid_all)


def train(model, fit_in, fit_future, calib_arr):
    torch.manual_seed(SEED)
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
            l_forecast = out["k_resid"].mean()
            loss = l_pred + BETA * l_me + l_mc + LAMBDA_EDGE * l_edge + LAMBDA_FORECAST * l_forecast
            loss.backward()
            optimizer.step()
            fit_loss += loss.item() * batch.size(0)
        fit_loss /= len(fit_in)

        d_proto, _, _, _, _ = per_sample_scores(model, calib_arr)
        calib_mse = float(d_proto.mean())
        print(f"  epoch {epoch:2d}  fit_loss={fit_loss:.6f}  calib_d_proto_mean={calib_mse:.6f}  "
              f"codebook_util={model.memory.codebook_utilization():.2f}  "
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
    return (x - median) / iqr, iqr


def main(horizon_mult=10, use_forecast_prior=True, out_suffix=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    forecast_h = STRIDE * horizon_mult

    print("loading Sielaff ...")
    data, targets, cols, label_map, fit_idx, calib_idx, test_normal_idx, window_machine = load_and_split()
    num_nodes = len(cols)
    scaler = fit_scaler(data, fit_idx)
    data_scaled = scale(data, scaler)

    def make_pairs(idx_arr, require_next_normal, require_same_split):
        vi, chains, mask = build_pairs(idx_arr, targets, window_machine, horizon_mult,
                                        require_next_normal, require_same_split)
        x_in = data_scaled[vi]
        x_future = gather_future(data_scaled, chains, STRIDE)
        return x_in, x_future, mask, len(idx_arr), len(vi)

    fit_in, fit_future, _, n0, n1 = make_pairs(fit_idx, True, True)
    print(f"  fit   pairs: {n0} -> {n1}")
    calib_in, calib_future, _, n0, n1 = make_pairs(calib_idx, True, True)
    print(f"  calib pairs: {n0} -> {n1}")
    test_normal_in, test_normal_future, test_normal_mask, n0, n1 = make_pairs(test_normal_idx, True, True)
    print(f"  test_normal pairs: {n0} -> {n1}")

    fault_pairs, fault_masks = {}, {}
    for label_id_str, name in label_map.items():
        label_id = int(label_id_str)
        if label_id == 0:
            continue
        fault_idx = np.where(targets == label_id)[0]
        if len(fault_idx) == 0:
            continue
        f_in, f_future, f_mask, n0, n1 = make_pairs(fault_idx, False, False)
        fault_pairs[name] = (f_in, f_future)
        fault_masks[name] = f_mask
        print(f"  fault[{name}] pairs: {n0} -> {n1}")

    fit_arr = data_scaled[fit_idx]
    calib_arr = data_scaled[calib_idx]
    test_normal_arr = data_scaled[test_normal_idx]
    fault_full = {name: data_scaled[np.where(targets == int(lid))[0]]
                  for lid, name in label_map.items() if int(lid) != 0}
    print(f"  fit={fit_arr.shape} calib={calib_arr.shape} test_normal={test_normal_arr.shape} nodes={num_nodes}")

    print(f"\ntraining JointPrototypeV21Forecast (nodes={num_nodes}, top_k={TOP_K}, M={NUM_PROTOTYPES}, "
          f"embed_dim={EMBED_DIM}, window_len={WINDOW_LEN}, horizon_mult={horizon_mult}, "
          f"forecast_h={forecast_h}, use_forecast_prior={use_forecast_prior}) -- V2.1 + forecast head, "
          f"no declared physics edges ...")
    model = JointPrototypeV21Forecast(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=EMBED_DIM,
                                       num_prototypes=NUM_PROTOTYPES, forecast_h=forecast_h,
                                       prior_edges=None, top_k=TOP_K).to(DEVICE)
    model = train(model, fit_in, fit_future, calib_arr)

    print("\nStage B: calibrating covariance head from calib split ...")
    _, d_node_calib_b, _, calib_idx_b, _ = per_sample_scores(model, calib_arr)
    cov_min = max(MIN_PROTO_SAMPLES, num_nodes + 1)
    model.cov_head.set_calibration(d_node_calib_b, calib_idx_b, min_samples=cov_min)
    n_valid_cov = int(model.cov_head.calib_valid.sum())
    print(f"  cov_head: {n_valid_cov}/{NUM_PROTOTYPES} prototypes with >= {cov_min} calib windows")

    print("\ncalibrating from calib split ...")
    d_proto_calib, d_node_calib, resid_struct_calib, _, d_mahal_calib = per_sample_scores(model, calib_arr)
    d_proto_normal, d_node_normal, resid_struct_normal, _, d_mahal_normal = per_sample_scores(model, test_normal_arr)
    k_resid_calib = forecast_scores(model, calib_in, calib_future)
    k_resid_normal = forecast_scores(model, test_normal_in, test_normal_future)

    _, node_iqr = zscore(d_node_normal, d_node_calib)
    _, edge_iqr = zscore(resid_struct_normal, resid_struct_calib)
    _, forecast_iqr = zscore(k_resid_normal, k_resid_calib)
    node_mask = node_iqr >= RELIABILITY_RATIO * np.median(node_iqr)
    edge_mask = edge_iqr >= RELIABILITY_RATIO * np.median(edge_iqr)
    forecast_mask = forecast_iqr >= RELIABILITY_RATIO * np.median(forecast_iqr)
    print(f"excluded {int((~node_mask).sum())}/{num_nodes} nodes from node max()")
    print(f"excluded {int((~edge_mask).sum())}/{num_nodes} nodes from edge max()")
    print(f"excluded {int((~forecast_mask).sum())}/{num_nodes} nodes from forecast max()")

    z_node_normal, _ = zscore(d_node_normal, d_node_calib)
    z_struct_normal, _ = zscore(resid_struct_normal, resid_struct_calib)
    z_mahal_normal, _ = zscore(d_mahal_normal[:, None], d_mahal_calib[:, None])
    z_mahal_normal = z_mahal_normal[:, 0]
    z_forecast_normal, _ = zscore(k_resid_normal, k_resid_calib)

    def base_scores(z_node, z_struct, z_mahal):
        b = z_node[:, node_mask].max(axis=1)
        c = z_struct[:, edge_mask].max(axis=1)
        h = z_mahal
        return {"B_node_max": b, "C_struct_max": c, "H_cov_mahal": h, "I_node_cov_max": np.maximum(b, h)}

    def add_forecast_scores(base, z_forecast, mask):
        k = z_forecast[:, forecast_mask].max(axis=1)
        b, c, h = base["B_node_max"][mask], base["C_struct_max"][mask], base["H_cov_mahal"][mask]
        out = dict(base)
        out["K_forecast_max"] = k
        out["BK_max"] = np.maximum(b, k)
        out["CK_max"] = np.maximum(c, k)
        out["HK_max"] = np.maximum(h, k)
        return out

    scores_normal_base = base_scores(z_node_normal, z_struct_normal, z_mahal_normal)
    scores_normal = add_forecast_scores(scores_normal_base, z_forecast_normal, test_normal_mask)

    print("\nscoring fault types ...")
    report = {"config": {"embed_dim": EMBED_DIM, "num_prototypes": NUM_PROTOTYPES, "top_k": TOP_K,
                          "window_len": WINDOW_LEN, "stride": STRIDE, "horizon_mult": horizon_mult,
                          "forecast_h": forecast_h, "epochs": EPOCHS, "reliability_ratio": RELIABILITY_RATIO,
                          "use_forecast_prior": use_forecast_prior},
              "nodes": cols, "fault_types": {}}
    rows = {k: [] for k in scores_normal}
    for name, arr in fault_full.items():
        f_in, f_future = fault_pairs[name]
        _, d_node_f, resid_struct_f, _, d_mahal_f = per_sample_scores(model, arr)
        k_resid_f = forecast_scores(model, f_in, f_future)
        z_node_f, _ = zscore(d_node_f, d_node_calib)
        z_struct_f, _ = zscore(resid_struct_f, resid_struct_calib)
        z_mahal_f, _ = zscore(d_mahal_f[:, None], d_mahal_calib[:, None])
        z_mahal_f = z_mahal_f[:, 0]
        z_forecast_f, _ = zscore(k_resid_f, k_resid_calib)
        scores_fault_base = base_scores(z_node_f, z_struct_f, z_mahal_f)
        scores_fault = add_forecast_scores(scores_fault_base, z_forecast_f, fault_masks[name])

        aurocs = {}
        for key in scores_normal:
            labels = np.concatenate([np.zeros(len(scores_normal[key])), np.ones(len(scores_fault[key]))])
            s = np.concatenate([scores_normal[key], scores_fault[key]])
            auroc = float(sk_metrics.roc_auc_score(labels, s))
            aurocs[key] = auroc
            rows[key].append((name, auroc))
        report["fault_types"][name] = {"n": int(len(arr)), "n_paired": int(len(f_in)), **aurocs}
        print(f"  {name:<20}n={len(arr):<5}" + "  ".join(f"{k}={v:.3f}" for k, v in aurocs.items()))

    print(f"\n{'method':<24}{'mean AUROC (5 fault types)':>28}")
    summary = {}
    for key in scores_normal:
        mean_auroc = float(np.mean([r[1] for r in rows[key]]))
        summary[key] = mean_auroc
        print(f"{key:<24}{mean_auroc:>28.3f}")

    report["summary_mean_auroc"] = summary
    suffix = out_suffix if out_suffix is not None else f"forecast_v2_h{horizon_mult}{'_noprior' if not use_forecast_prior else ''}"
    report_path = OUT_DIR / f"sielaff_{suffix}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "report": report},
               OUT_DIR / f"sielaff_{suffix}.pth")
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
