#!/usr/bin/env python3
"""
Single-machine (no federation) run of this project's full "记忆检索 +
物理验证" pipeline on the Paderborn KAt bearing dataset -- every step
except cross-device federation, reusing `src/fl_model.FLGDNMemory`
UNCHANGED (the Sec 2A/2B-compliant architecture: one shared encoder,
two independent latent heads -- memory distance `d` and structure
residual `r` -- decoder training-only and prunable at inference), exactly
as established for the (now-superseded) IMS bearing exploration.

## Why this dataset doesn't need IMS's trend/ranking workaround

Paderborn has DEFINITE per-bearing damage-class ground truth (see
`benchmark/datasets/paderborn_adapter.py` and
`memory/paderborn-bearing-dataset.md` for the verified taxonomy: K*=
healthy, KA*=outer ring, KI*=inner ring, KB*=combined), so this script
uses this project's NORMAL convention: fit + calibrate on healthy
bearings only, evaluate AUROC per damaged bearing code -- same as
robo3er/Sielaff, not IMS's forced "no ground truth, rank by trend"
substitute.

## What's fed to the model

Only the three ~64kHz channels (`vibration_1`, `phase_current_1`,
`phase_current_2`) -- see `paderborn_adapter.py`'s docstring for why the
4kHz mechanical channels and 1Hz temperature aren't included yet (a
stated simplification, not a claim they're uninformative). Each 4-second
measurement file is box-average decimated 256,000 -> 2048 samples, then
`FLGDNMemory` reconstructs short overlapping sub-windows of that
decimated sequence (`WINDOW_LEN` samples, stride `WINDOW_STRIDE`) -- the
same whole-window-reconstruction task established for the bearing
architecture-gap fix, not GDN's forecasting task.

Scores (matching `train_fl_memory_gdn.py`'s own d/r/d+r convention):
  - "d": memory quantization distance (novelty vs. learned healthy regimes)
  - "r": structure-head residual (cross-channel -- vibration vs. current
         -- consistency)
  - "d+r": both concatenated raw per-node before ONE shared per-node
           median/IQR normalize-then-max

Usage:
    python3 run_paderborn_fl_model.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fl_model import FLGDNMemory  # noqa: E402
from datasets.paderborn_adapter import (  # noqa: E402
    load_bearing, HEALTHY_CODES, ALL_DAMAGED_CODES, category_of, damage_origin_of, CHANNELS,
)

OUT_DIR = REPO_ROOT / "checkpoints" / "paderborn"

DECIMATE = 125           # 256000 -> 2048 samples/file (~512Hz effective, see adapter docstring)
FIT_FRACTION = 0.70      # per healthy bearing, of its 80 files
CALIB_FRACTION = 0.15    # remainder (0.15) -> test_normal
WINDOW_LEN = 64
WINDOW_STRIDE = 32
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
SEED = 42
EMBED_DIM = 64
TOP_K = 15  # auto-capped to num_nodes-1=2 inside FLGDNMemory (only 3 channels)
NUM_PROTOTYPES = 32
BETA = 0.25
LAMBDA = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_windows(arr, window_len, stride):
    """arr: [num_files, T, F] -> (windows [M, window_len, F], file_id [M])."""
    n, t, f = arr.shape
    positions = list(range(0, t - window_len + 1, stride))
    windows = np.empty((n, len(positions), window_len, f), dtype=np.float32)
    for j, pos in enumerate(positions):
        windows[:, j] = arr[:, pos : pos + window_len]
    file_id = np.repeat(np.arange(n), len(positions))
    return windows.reshape(-1, window_len, f), file_id


@torch.no_grad()
def per_sample_dr(model, windows, batch_size=BATCH_SIZE):
    model.eval()
    d_all, r_all = [], []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(DEVICE)
        out = model(batch, training_mode=False)
        d_all.append(out["d"].cpu().numpy())
        r_all.append(out["r"].cpu().numpy())
    return np.concatenate(d_all, axis=0), np.concatenate(r_all, axis=0)


def per_file_dr(model, arr, window_len=WINDOW_LEN, stride=WINDOW_STRIDE, batch_size=BATCH_SIZE):
    windows, file_id = make_windows(arr, window_len, stride)
    d, r = per_sample_dr(model, windows, batch_size)
    n = len(arr)
    out_d = np.zeros((n, d.shape[1]), dtype=np.float64)
    out_r = np.zeros((n, r.shape[1]), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    np.add.at(out_d, file_id, d)
    np.add.at(out_r, file_id, r)
    np.add.at(counts, file_id, 1)
    return (out_d / counts[:, None]).astype(np.float32), (out_r / counts[:, None]).astype(np.float32)


def gdn_score(per_file_err, median, iqr):
    return ((per_file_err - median) / np.maximum(iqr, 1e-8)).max(axis=1)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    num_nodes = len(CHANNELS)

    print("loading healthy bearings (K001-K006) ...")
    fit_list, calib_list, test_normal_list = [], [], []
    for code in HEALTHY_CODES:
        w, _, _ = load_bearing(code, decimate=DECIMATE)
        n = len(w)
        n_fit = int(n * FIT_FRACTION)
        n_calib = int(n * CALIB_FRACTION)
        fit_list.append(w[:n_fit])
        calib_list.append(w[n_fit : n_fit + n_calib])
        test_normal_list.append(w[n_fit + n_calib :])
        print(f"  {code}: n={n} fit={n_fit} calib={n_calib} test_normal={n - n_fit - n_calib}")
    fit_raw = np.concatenate(fit_list, axis=0)
    calib_raw = np.concatenate(calib_list, axis=0)
    test_normal_raw = np.concatenate(test_normal_list, axis=0)
    print(f"pooled healthy: fit={len(fit_raw)} calib={len(calib_raw)} test_normal={len(test_normal_raw)}")

    scaler = StandardScaler().fit(fit_raw.reshape(-1, num_nodes))

    def scale(arr):
        n, t, f = arr.shape
        return scaler.transform(arr.reshape(-1, f)).reshape(n, t, f).astype(np.float32)

    fit_scaled, calib_scaled, test_normal_scaled = scale(fit_raw), scale(calib_raw), scale(test_normal_raw)

    print(f"\ntraining FLGDNMemory (nodes={num_nodes}, window_len={WINDOW_LEN}, "
          f"M={NUM_PROTOTYPES}, embed_dim={EMBED_DIM}) ...")
    fit_windows, _ = make_windows(fit_scaled, WINDOW_LEN, WINDOW_STRIDE)
    print(f"training windows: {len(fit_windows)}")
    model = FLGDNMemory(num_nodes=num_nodes, window_size=WINDOW_LEN, embed_dim=EMBED_DIM,
                          top_k=TOP_K, num_prototypes=NUM_PROTOTYPES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fit_windows)),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    best_calib_mse, best_state = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        fit_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch, training_mode=True)
            l_pred = torch.nn.functional.mse_loss(out["x_hat"], batch)
            l_me, l_mc = model.memory.commitment_losses(out["z"], out["z_q"])
            l_struct = torch.nn.functional.mse_loss(out["z"].detach(), out["z_hat"])
            loss = l_pred + BETA * l_me + l_mc + LAMBDA * l_struct
            loss.backward()
            optimizer.step()
            fit_loss += loss.item() * batch.size(0)
        fit_loss /= len(fit_windows)

        d_calib, r_calib = per_file_dr(model, calib_scaled)
        calib_mse = float(r_calib.mean())
        print(f"  epoch {epoch:2d}  fit_loss={fit_loss:.6f}  calib_r_mse={calib_mse:.6f}  "
              f"codebook_util={model.memory.codebook_utilization():.2f}")
        if calib_mse < best_calib_mse:
            best_calib_mse = calib_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    print("\ncalibrating scoring thresholds from healthy calib split ...")
    d_calib, r_calib = per_file_dr(model, calib_scaled)
    d_median, r_median = np.median(d_calib, axis=0), np.median(r_calib, axis=0)
    d_q75, d_q25 = np.percentile(d_calib, [75, 25], axis=0)
    r_q75, r_q25 = np.percentile(r_calib, [75, 25], axis=0)
    d_iqr = np.maximum(d_q75 - d_q25, 1e-8)
    r_iqr = np.maximum(r_q75 - r_q25, 1e-8)
    dr_median = np.concatenate([d_median, r_median])
    dr_iqr = np.concatenate([d_iqr, r_iqr])

    d_normal, r_normal = per_file_dr(model, test_normal_scaled)
    scores_normal = {
        "d": gdn_score(d_normal, d_median, d_iqr),
        "r": gdn_score(r_normal, r_median, r_iqr),
        "d+r": gdn_score(np.concatenate([d_normal, r_normal], axis=1), dr_median, dr_iqr),
    }

    print("\nscoring damaged bearings ...")
    report = {"config": {"decimate": DECIMATE, "window_len": WINDOW_LEN, "window_stride": WINDOW_STRIDE,
                           "num_prototypes": NUM_PROTOTYPES, "beta": BETA, "lambda": LAMBDA, "epochs": EPOCHS},
               "channels": CHANNELS, "healthy_bearings": HEALTHY_CODES,
               "test_normal_n": len(test_normal_raw), "bearings": {}}
    rows = []
    for code in ALL_DAMAGED_CODES:
        w, _, _ = load_bearing(code, decimate=DECIMATE)
        w_scaled = scale(w)
        d_fault, r_fault = per_file_dr(model, w_scaled)
        scores_fault = {
            "d": gdn_score(d_fault, d_median, d_iqr),
            "r": gdn_score(r_fault, r_median, r_iqr),
            "d+r": gdn_score(np.concatenate([d_fault, r_fault], axis=1), dr_median, dr_iqr),
        }
        aurocs = {}
        for key in scores_normal:
            labels = np.concatenate([np.zeros(len(scores_normal[key])), np.ones(len(scores_fault[key]))])
            scores = np.concatenate([scores_normal[key], scores_fault[key]])
            aurocs[key] = float(sk_metrics.roc_auc_score(labels, scores))
        cat, origin = category_of(code), damage_origin_of(code)
        report["bearings"][code] = {"category": cat, "origin": origin, "n": len(w),
                                       **{f"auroc_{k}": v for k, v in aurocs.items()}}
        rows.append((code, cat, origin, aurocs["d"], aurocs["r"], aurocs["d+r"]))
        print(f"  {code:<6}{cat:<12}{origin:<11}"
              f"AUROC(d)={aurocs['d']:.3f}  AUROC(r)={aurocs['r']:.3f}  AUROC(d+r)={aurocs['d+r']:.3f}")

    print(f"\n{'code':<6}{'category':<12}{'origin':<11}{'AUROC(d)':>10}{'AUROC(r)':>10}{'AUROC(d+r)':>12}")
    for code, cat, origin, ad, ar, adr in rows:
        print(f"{code:<6}{cat:<12}{origin:<11}{ad:>10.3f}{ar:>10.3f}{adr:>12.3f}")

    print(f"\n{'category':<12}{'n_bearings':>12}{'mean AUROC(d)':>16}{'mean AUROC(r)':>16}{'mean AUROC(d+r)':>18}")
    category_summary = {}
    for cat in ["outer_ring", "inner_ring", "combined"]:
        subset = [r for r in rows if r[1] == cat]
        if not subset:
            continue
        mean_d = float(np.mean([r[3] for r in subset]))
        mean_r = float(np.mean([r[4] for r in subset]))
        mean_dr = float(np.mean([r[5] for r in subset]))
        category_summary[cat] = {"n_bearings": len(subset), "mean_auroc_d": mean_d,
                                    "mean_auroc_r": mean_r, "mean_auroc_dr": mean_dr}
        print(f"{cat:<12}{len(subset):>12}{mean_d:>16.3f}{mean_r:>16.3f}{mean_dr:>18.3f}")
    report["category_summary"] = category_summary

    with open(OUT_DIR / "paderborn_fl_model_report.json", "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "report": report},
                OUT_DIR / "paderborn_fl_model.pth")
    print(f"\nsaved -> {OUT_DIR / 'paderborn_fl_model_report.json'}")
    print(f"saved -> {OUT_DIR / 'paderborn_fl_model.pth'}")


if __name__ == "__main__":
    main()
