#!/usr/bin/env python3
"""
robo_fleet analog of `plot_robo_pdm_test_embeddings_pca.py`: for each of
robo_fleet's 4 real robot clients (`rob_00`..`rob_03`), train the same
federated `JointPrototypeV31Forecast` setup as
`run_robo_fleet_bck_federated.py`, then draw two PCA(2D) panels per robot:

  left  = MODEL EMBEDDING SPACE: each window's memory-embedding `z`
          ([num_nodes, embed_dim], flattened) for that robot's own
          normal (test_normal_idx) and ALL 4 of ITS fault-type windows,
          PCA-fit on the pooled (normal+4 faults) set, with the client's
          codebook prototypes projected into the SAME PCA space
          (transformed, not re-fit).
  right = RAW FEATURE SPACE: each window's raw standardized 26-dim
          feature vector (time-mean over the window), PCA-fit the same
          way, with the top-|loading| features per axis annotated.

Unlike robo_pdm_test (clients ARE fault-type groups, so that script's
panels are one-fault-vs-pooled-normal per row), robo_fleet's clients are
PHYSICAL ROBOTS -- every client already has normal + all 4 fault types --
so each row here is a genuine 5-class scatter (normal + 4 faults) for
ONE robot's own model, not a binary one-fault-vs-normal split. This
directly answers "are this robot's own fault types separable in its own
embedding/raw space," the multi-class analog of robo_pdm_test's question.

`--method kpca` swaps linear PCA for RBF Kernel PCA in both panels, same
protocol as the robo_pdm_test script (see that script's docstring for
why: lets a fault that only separates along a curved manifold show a
cleaner split than linear PCA can).

Usage:
    python3 benchmark/plot_robo_fleet_embeddings_pca.py [--out PATH] [--method {pca,kpca}] [--gamma G]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA, KernelPCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"

import importlib.util
spec = importlib.util.spec_from_file_location("robo_fleet_bck_fed", BENCHMARK_DIR / "run_robo_fleet_bck_federated.py")
v2f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2f)

FAULT_COLORS = {
    "caster_wheel_jam": "crimson",
    "added_load_2.5kg": "darkorange",
    "drive_wheel_cable": "seagreen",
    "thumbtack_fault": "purple",
}

FAULT_DISPLAY_NAMES = {
    "caster_wheel_jam": "Caster wheel jam",
    "added_load_2.5kg": "Drag load",
    "drive_wheel_cable": "Drive wheel jam",
    "thumbtack_fault": "Thumbtack",
}


@torch.no_grad()
def get_z_and_windowmean(model, windows, batch_size=v2f.BATCH_SIZE):
    """z: [n, num_nodes, embed_dim] flattened to [n, num_nodes*embed_dim].
    window_mean: [n, num_nodes] time-mean of the (already scaler-
    transformed) raw window, for the interpretable raw-feature-space
    panel."""
    model.eval()
    if len(windows) == 0:
        return (np.zeros((0, model.num_nodes * model.memory.codebook.shape[-1]), dtype=np.float32),
                np.zeros((0, model.num_nodes), dtype=np.float32))
    z_all = []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i:i + batch_size]).to(v2f.DEVICE)
        out = model(batch, training_mode=False)
        z_all.append(out["z"].reshape(out["z"].shape[0], -1).cpu().numpy())
    z_all = np.concatenate(z_all)
    window_mean = windows.mean(axis=1)
    return z_all, window_mean


def fit_reducer(X, method, gamma=None):
    if method == "pca":
        return PCA(n_components=2, random_state=0).fit(X)
    return KernelPCA(n_components=2, kernel="rbf", gamma=gamma, random_state=0,
                      fit_inverse_transform=False).fit(X)


def explained_var_ratio(reducer, method, X=None, gamma=None):
    if method == "pca":
        return reducer.explained_variance_ratio_
    ev2 = reducer.eigenvalues_[:2]
    full = KernelPCA(n_components=None, kernel="rbf", gamma=gamma, random_state=0,
                      fit_inverse_transform=False).fit(X)
    total = full.eigenvalues_.sum()
    return ev2 / total if total > 0 else np.zeros(2)


def feature_relevance(reducer, method, X, cols):
    """PCA: literal per-axis loading vector (components_). KPCA: no linear
    loading exists, so fall back to Pearson correlation of each raw
    feature against each projected axis over the pooled window set."""
    if method == "pca":
        return reducer.components_  # [2, F]
    proj = reducer.transform(X)  # [n, 2]
    F = X.shape[1]
    corr = np.zeros((2, F))
    for axis in range(2):
        for fi in range(F):
            c = np.corrcoef(X[:, fi], proj[:, axis])[0, 1]
            corr[axis, fi] = 0.0 if np.isnan(c) else c
    return corr


def main(out_path=None, horizon_mult=10, num_prototypes=None, linkage="single",
         method="pca", gamma=None, use_forecast_prior=True):
    torch.manual_seed(v2f.SEED)
    np.random.seed(v2f.SEED)
    num_prototypes = num_prototypes if num_prototypes is not None else v2f.NUM_PROTOTYPES
    forecast_h = v2f.STRIDE * horizon_mult

    print("loading robo_fleet, 4 real per-robot federated clients ...")
    data, targets, label_map, clients, cols, window_robot = v2f.build_clients()
    num_nodes = len(cols)
    node_idx = {name: i for i, name in enumerate(cols)}
    prior_edges = [(node_idx[s], node_idx[d]) for s, d in v2f.EDGES_NAMED]
    forecast_prior_edges = prior_edges if use_forecast_prior else None
    for c in clients:
        print(f"  client {c.client_id} ({c.robot_name}): fit={len(c.fit_idx)} calib={len(c.calib_idx)} "
              f"test_normal={len(c.test_normal_idx)} faults={ {label_map[str(k)]: len(v) for k, v in c.fault_idx_by_label.items()} }")

    scalers = [v2f.fit_client_scaler(data, c.fit_idx) for c in clients]
    models = [v2f.JointPrototypeV21Forecast(num_nodes=num_nodes, window_size=v2f.WINDOW_LEN, embed_dim=v2f.EMBED_DIM,
                                             num_prototypes=num_prototypes, prior_edges=prior_edges,
                                             forecast_h=forecast_h, top_k=v2f.TOP_K,
                                             forecast_prior_edges=forecast_prior_edges).to(v2f.DEVICE)
              for _ in clients]
    shared_init = {k: val.clone() for k, val in models[0].state_dict().items() if not k.startswith("memory.")}
    for model in models[1:]:
        model.load_state_dict(shared_init, strict=False)

    fit_pairs = []
    for c in clients:
        vi, chains, _ = v2f.build_pairs(c.fit_idx, targets, window_robot, horizon_mult, True, True)
        fit_pairs.append((vi, data[vi], v2f.gather_future(data, chains, v2f.STRIDE)))

    def local_train_step(c, model):
        idx = c.client_id
        scaler = scalers[idx]
        vi, x_in_raw, x_future_raw = fit_pairs[idx]
        n, t, f = x_in_raw.shape
        x_in = scaler.transform(x_in_raw.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
        n2, t2, f2 = x_future_raw.shape
        x_future = (scaler.transform(x_future_raw.reshape(-1, f2)).reshape(n2, t2, f2).astype(np.float32)
                    if n2 > 0 else x_future_raw)
        v2f.train_local(model, x_in, x_future, v2f.LOCAL_EPOCHS, v2f.DEVICE, lr=v2f.LR, beta=v2f.BETA,
                         lambda_edge=v2f.LAMBDA_EDGE, lambda_forecast=v2f.LAMBDA_FORECAST,
                         batch_size=v2f.BATCH_SIZE)
        return len(x_in)

    print(f"\nfederated training: {v2f.ROUNDS} rounds x {v2f.LOCAL_EPOCHS} local epochs ...")
    v2f.run_federated_rounds(clients, models, local_train_step, v2f.ROUNDS, v2f.GAMMA, v2f.DELTA,
                              sync_encoder_decoder=v2f.SYNC_ENCODER_DECODER, linkage=linkage)

    n_clients = len(clients)
    fig, axes = plt.subplots(n_clients, 2, figsize=(15, 5 * n_clients))
    method_label = "PCA" if method == "pca" else "Kernel PCA (RBF)"

    for row, (c, model, scaler) in enumerate(zip(clients, models, scalers)):
        normal_w = v2f.scale_client(data, scaler, c.test_normal_idx)
        z_normal, wm_normal = get_z_and_windowmean(model, normal_w)

        fault_names, fault_z, fault_wm = [], {}, {}
        for label_id, fault_idx in c.fault_idx_by_label.items():
            name = label_map[str(label_id)]
            fault_w = v2f.scale_client(data, scaler, fault_idx)
            z_f, wm_f = get_z_and_windowmean(model, fault_w)
            fault_names.append(name)
            fault_z[name] = z_f
            fault_wm[name] = wm_f

        # --- left panel: model embedding space, pooled fit across normal + every fault ---
        z_all = np.concatenate([z_normal] + [fault_z[n] for n in fault_names], axis=0)
        reducer_z = fit_reducer(z_all, method, gamma)
        z_2d = reducer_z.transform(z_all)
        ev_z = explained_var_ratio(reducer_z, method, X=z_all, gamma=gamma)

        proto = model.memory.codebook.detach().cpu().numpy()  # [M, N, D]
        proto_2d = reducer_z.transform(proto.reshape(proto.shape[0], -1))

        ax = axes[row, 0] if n_clients > 1 else axes[0]
        n0 = len(z_normal)
        ax.scatter(z_2d[:n0, 0], z_2d[:n0, 1], c="steelblue", alpha=0.5, s=14, label="normal")
        cursor = n0
        for name in fault_names:
            n_f = len(fault_z[name])
            ax.scatter(z_2d[cursor:cursor + n_f, 0], z_2d[cursor:cursor + n_f, 1],
                       c=FAULT_COLORS.get(name, "gray"), alpha=0.5, s=14, label=FAULT_DISPLAY_NAMES.get(name, name))
            cursor += n_f
        ax.scatter(proto_2d[:, 0], proto_2d[:, 1], c="black", marker="x", s=90, linewidths=2, label="prototypes")
        ax.set_title(f"{c.robot_name}: model embedding z, {method_label} 2D "
                     f"(explained var {ev_z.sum():.2f})", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7, loc="best")

        # --- standalone, large-font single-panel version of the SAME embedding-space
        # plot above (identical data/PCA fit, just its own figure) -- sized and
        # font-scaled for legibility once dropped into an IEEE double-column
        # template at single-column width (~3.5in): oversized fonts here survive
        # LaTeX's proportional downscaling instead of shrinking to unreadable.
        fig_solo, ax_solo = plt.subplots(figsize=(7, 7))
        ax_solo.scatter(z_2d[:n0, 0], z_2d[:n0, 1], c="steelblue", alpha=0.55, s=30, label="normal")
        cursor_solo = n0
        for name in fault_names:
            n_f = len(fault_z[name])
            ax_solo.scatter(z_2d[cursor_solo:cursor_solo + n_f, 0], z_2d[cursor_solo:cursor_solo + n_f, 1],
                            c=FAULT_COLORS.get(name, "gray"), alpha=0.55, s=30, label=FAULT_DISPLAY_NAMES.get(name, name))
            cursor_solo += n_f
        ax_solo.scatter(proto_2d[:, 0], proto_2d[:, 1], c="gold", marker="X", s=260,
                        edgecolors="black", linewidths=1.5, zorder=5, label="prototypes")
        ax_solo.set_xlabel("PC1", fontsize=17)
        ax_solo.set_ylabel("PC2", fontsize=17)
        ax_solo.tick_params(axis="both", labelsize=14)
        ax_solo.legend(fontsize=17, loc="upper right", markerscale=1.6, frameon=True)
        fig_solo.tight_layout()
        solo_path = str(PROJECT_ROOT / "checkpoints" / "robo_fleet"
                        / f"robo_fleet_embeddings_pca_{method}_{c.robot_name}.png")
        fig_solo.savefig(solo_path, dpi=220, bbox_inches="tight")
        plt.close(fig_solo)
        print(f"  saved standalone embedding panel -> {solo_path}")

        # --- right panel: raw (interpretable) feature space ---
        wm_all = np.concatenate([wm_normal] + [fault_wm[n] for n in fault_names], axis=0)
        reducer_raw = fit_reducer(wm_all, method, gamma)
        wm_2d = reducer_raw.transform(wm_all)
        ev_raw = explained_var_ratio(reducer_raw, method, X=wm_all, gamma=gamma)

        ax2 = axes[row, 1] if n_clients > 1 else axes[1]
        ax2.scatter(wm_2d[:n0, 0], wm_2d[:n0, 1], c="steelblue", alpha=0.5, s=14, label="normal")
        cursor = n0
        for name in fault_names:
            n_f = len(fault_wm[name])
            ax2.scatter(wm_2d[cursor:cursor + n_f, 0], wm_2d[cursor:cursor + n_f, 1],
                        c=FAULT_COLORS.get(name, "gray"), alpha=0.5, s=14, label=FAULT_DISPLAY_NAMES.get(name, name))
            cursor += n_f
        ax2.set_title(f"{c.robot_name}: raw 26-feature space, {method_label} 2D "
                      f"(explained var {ev_raw.sum():.2f})", fontsize=10)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.legend(fontsize=7, loc="best")

        loadings = feature_relevance(reducer_raw, method, wm_all, cols)  # [2, num_nodes]
        stat_name = "loading" if method == "pca" else "corr"
        top_pc1 = sorted(zip(cols, loadings[0]), key=lambda t: -abs(t[1]))[:5]
        top_pc2 = sorted(zip(cols, loadings[1]), key=lambda t: -abs(t[1]))[:5]
        print(f"\n=== {c.robot_name}: raw-feature-space {method_label} {stat_name}s ===")
        print(f"  PC1 (explains {ev_raw[0]:.2f}): " + ", ".join(f"{n}={w:+.2f}" for n, w in top_pc1))
        print(f"  PC2 (explains {ev_raw[1]:.2f}): " + ", ".join(f"{n}={w:+.2f}" for n, w in top_pc2))
        note = "PC1 top: " + ", ".join(n for n, _ in top_pc1[:3])
        ax2.text(0.02, 0.02, note, transform=ax2.transAxes, fontsize=7, va="bottom",
                  bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    default_name = f"robo_fleet_embeddings_{method}.png"
    out_path = out_path or str(PROJECT_ROOT / "checkpoints" / "robo_fleet" / default_name)
    plt.savefig(out_path, dpi=130)
    print(f"\nsaved -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--horizon-mult", type=int, default=10)
    parser.add_argument("--num-prototypes", type=int, default=None)
    parser.add_argument("--linkage", type=str, default="single", choices=["single", "complete"])
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "kpca"])
    parser.add_argument("--gamma", type=float, default=None, help="RBF kernel gamma for --method kpca "
                         "(default: sklearn's 1/n_features)")
    args = parser.parse_args()
    main(out_path=args.out, horizon_mult=args.horizon_mult, num_prototypes=args.num_prototypes,
         linkage=args.linkage, method=args.method, gamma=args.gamma)
