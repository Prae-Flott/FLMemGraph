#!/usr/bin/env python3
"""
Joint Prototype Memory + Physics-Relation GDN + Anomaly Attention (v3) on
voraus-AD, per `docs/joint_prototype_physics_gdn_anomaly_attention_prompt.md`.
Third dataset for `JointPrototypeGDNv3` (`src/joint_prototype_model.py`,
after Paderborn and robo3er) -- model reused unchanged, only the node set/
declared physics edges/relation types are voraus-AD-specific
(`benchmark/datasets/voraus_ad_adapter.py`).

REVISED (v2 of this script): 66 nodes (11 signals x 6 joints -- the full
target/motor/joint tracking chain, both torque sensors, and the
torque-forming/magnetizing current pair), 54 declared WITHIN-joint edges.
The original 18-node version was scoped down to match Paderborn/robo3er's
much smaller graphs, not because of any real data or model constraint,
and missed exactly the signals (`joint_velocity`, the tracking chain)
`benchmark/datasets/voraus_ad_physics.md` identified as most diagnostic
-- see `benchmark/datasets/voraus_ad_adapter.py`'s module docstring for
the full per-edge rationale. No cross-joint edges declared -- the arm's
kinematic coupling isn't characterized yet, see `memory/voraus-ad-dataset.md`.

Official train/test convention (`voraus_ad.py`'s `variant==PRE_A` split):
FIT/CALIB come from the 948 PRE_A (pure-normal) samples, TEST_NORMAL is
the 419 held-out NORMAL_OPERATION samples from other variants, and each
of the 12 named fault categories is scored separately -- same per-sample
AUROC convention as `run_robo3er_joint_prototype_v3.py` (one voraus-AD
"sample" = one pick-and-place cycle, already a fixed-length [1164, 66]
padded array, no further windowing needed).

## Aggregation fix (v3 of this script)

The 66-node run (see `memory/voraus-ad-joint-prototype-v3.md`'s update
note) found the typed-edge signal got WORSE than the 18-node version
despite adding the friction edge it was specifically missing -- diagnosed
as the known max-aggregation noise-floor problem
(`memory/paderborn-joint-prototype.md`), sharply worse here because
`s_node`/`s_edge`/`s_node_typed` each grew to 66 dimensions: taking a
raw `.max(axis=1)` over more dimensions systematically inflates the
right tail of the NORMAL test set's score distribution (more chances for
one dimension to spike by noise alone), which hurts AUROC even when a
genuinely informative dimension (the new friction edge) is among them.

Fix: `topk_mean()` + `two_stage_group_score()` replace the raw
per-dimension z-score + `.max(axis=1)` for the node/edge/typed-edge
GROUPS (which have many dimensions) with a top-`TOP_K_AGG`-mean, THEN
re-calibrate that aggregate statistic AGAIN against its own distribution
on the calib split (not just each dimension independently) -- directly
correcting for "the null distribution of top-k-mean shifts as the group
grows" rather than assuming per-dimension normalization alone accounts
for it. The top-level combination across the 4 signal GROUPS (dG, node,
edge, typed -- only 4 items) is left as a plain max, since 4 items carries
far less of this multiple-comparisons risk than 66. `EPOCHS` is also
increased (12 -> 40): the 66-node run's `calib_dG_mean` was still
decreasing steadily at epoch 12 (unlike the smaller 18-node graph, which
had largely plateaued by then), so more training capacity was needed
regardless of the aggregation fix -- both changes are applied together
per the user's request, not isolated as a controlled ablation.

Usage:
    python3 run_voraus_ad_joint_prototype_v3.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics as sk_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_prototype_model import JointPrototypeGDNv3  # noqa: E402
from datasets.voraus_ad_adapter import load, NODE_NAMES, PRIOR_EDGES, EDGE_TYPES, FIXED_LEN  # noqa: E402

OUT_DIR = REPO_ROOT / "checkpoints" / "voraus_ad"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64  # smaller than Paderborn/robo3er -- windows are much longer (1164 vs 60-64)
EPOCHS = 100  # bumped again from 40 -- at epoch 40, calib_dG_mean was still decreasing
              # (~0.01/epoch, down from ~0.034/epoch around epoch 20-25 -- slowing but not
              # flat), so pushed further to see where it actually plateaus. Best-checkpoint
              # selection by calib_dG_mean (see train()) already guards against picking an
              # overfit late epoch if performance turns over before training ends.
LR = 1e-3
SEED = 42
EMBED_DIM = 64
NUM_PROTOTYPES = 16
TOP_K = 10  # bumped from 5 (the 18-node version's setting) -- with 66 nodes, top_k=5 would
            # attend over under 8% of other nodes; the generic attention head still needs
            # enough room to find useful cross-signal pairs beyond the declared typed edges
BETA = 0.25
LAMBDA_EDGE = 0.5
LAMBDA_TYPED = 0.5
MIN_PROTO_SAMPLES = 5
TOP_K_AGG = 3  # for the node/edge/typed-edge GROUP aggregation (see module docstring's
               # "Aggregation fix" section) -- top-3-mean instead of a raw max over all
               # 66 node/edge dimensions, re-calibrated against its own calib distribution
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def per_sample_scores(model, windows, batch_size=BATCH_SIZE):
    model.eval()
    d_G_all, s_node_all, s_edge_all, s_typed_all, idx_all, r_all = [], [], [], [], [], []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(DEVICE)
        out = model(batch, training_mode=False)
        d_G_all.append(out["d_G"].cpu().numpy())
        s_node_all.append(out["s_node"].cpu().numpy())
        s_edge_all.append(out["s_edge"].cpu().numpy())
        s_typed_all.append(out["s_node_typed"].cpu().numpy())
        idx_all.append(out["idx"].cpu().numpy())
        r_all.append(out["r"].cpu().numpy())
    return (np.concatenate(d_G_all), np.concatenate(s_node_all), np.concatenate(s_edge_all),
            np.concatenate(s_typed_all), np.concatenate(idx_all), np.concatenate(r_all))


def train(model, fit_arr, calib_arr):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fit_arr)),
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
            l_me, l_mc = model.memory.commitment_losses(out["z"], out["p_star"])
            l_edge = out["s_edge"].mean()
            l_typed = out["r"].mean()
            loss = l_pred + BETA * l_me + l_mc + LAMBDA_EDGE * l_edge + LAMBDA_TYPED * l_typed
            loss.backward()
            optimizer.step()
            fit_loss += loss.item() * batch.size(0)
        fit_loss /= len(fit_arr)

        d_G, _, _, _, _, _ = per_sample_scores(model, calib_arr)
        calib_mse = float(d_G.mean())
        print(f"  epoch {epoch:2d}  fit_loss={fit_loss:.6f}  calib_dG_mean={calib_mse:.6f}  "
              f"codebook_util={model.memory.codebook_utilization():.2f}  "
              f"prior_bias_strength={model.edge_head.prior_bias_strength.item():.3f}  "
              f"typed_temp={model.typed_head.log_temperature.exp().item():.3f}")
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


def topk_mean(x, k):
    """x: [B, D]. Mean of the top-k values along axis 1 (k clipped to D)."""
    k = min(k, x.shape[1])
    return np.sort(x, axis=1)[:, -k:].mean(axis=1)


def two_stage_group_score(raw_calib, raw_x, k=TOP_K_AGG):
    """raw_calib/raw_x: [N, D] raw (unstandardized) per-dimension signal
    (D = num_nodes, e.g. s_node/s_edge/s_node_typed). Two calibration
    stages: (1) per-dimension z-score against calib, as before; (2)
    top-k-mean of those z-scores, THEN re-z-score that scalar statistic
    against its OWN distribution on the calib split. Stage 2 is what a
    raw `.max(axis=1)` (the D->infinity limit of top-1) skips -- without
    it, the aggregate statistic's null distribution silently shifts as D
    grows, inflating false positives on the normal test set. See module
    docstring's "Aggregation fix" section."""
    z_calib = zscore(raw_calib, raw_calib)
    z_x = zscore(raw_x, raw_calib)
    stat_calib = topk_mean(z_calib, k)
    stat_x = topk_mean(z_x, k)
    median = np.median(stat_calib)
    iqr = max(np.percentile(stat_calib, 75) - np.percentile(stat_calib, 25), 1e-8)
    return (stat_x - median) / iqr


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    num_nodes = len(NODE_NAMES)

    print("loading voraus-AD ...")
    fit_arr, calib_arr, test_normal_arr, faults, node_names = load(seed=SEED)
    print(f"  fit={fit_arr.shape} calib={calib_arr.shape} test_normal={test_normal_arr.shape}")
    for name, arr in faults.items():
        print(f"  fault[{name}]={arr.shape}")

    print(f"\ntraining JointPrototypeGDNv3 (nodes={num_nodes}, prior_edges={len(PRIOR_EDGES)}, "
          f"edge_types={EDGE_TYPES}, top_k={TOP_K}, M={NUM_PROTOTYPES}, embed_dim={EMBED_DIM}, "
          f"window_len={FIXED_LEN}) ...")
    model = JointPrototypeGDNv3(num_nodes=num_nodes, window_size=FIXED_LEN, embed_dim=EMBED_DIM,
                                  num_prototypes=NUM_PROTOTYPES, prior_edges=PRIOR_EDGES,
                                  edge_types=EDGE_TYPES, top_k=TOP_K).to(DEVICE)
    model = train(model, fit_arr, calib_arr)

    print("\nStage B: estimating prototype-conditioned edge-residual statistics from calib split ...")
    _, _, _, _, calib_idx, calib_r = per_sample_scores(model, calib_arr)
    model.typed_head.set_calibration(calib_r, calib_idx, NUM_PROTOTYPES, min_samples=MIN_PROTO_SAMPLES)
    n_valid_proto = int(model.typed_head.calib_valid_proto.sum())
    print(f"  {n_valid_proto}/{NUM_PROTOTYPES} prototypes had >= {MIN_PROTO_SAMPLES} calib windows "
          f"(rest fall back to global edge stats)")

    print("\ncalibrating from calib split ...")
    d_G_calib, s_node_calib, s_edge_calib, s_typed_calib, _, _ = per_sample_scores(model, calib_arr)
    d_G_normal, s_node_normal, s_edge_normal, s_typed_normal, _, _ = per_sample_scores(model, test_normal_arr)

    z_dG_normal = zscore(d_G_normal, d_G_calib)
    node_score_normal = two_stage_group_score(s_node_calib, s_node_normal)
    edge_score_normal = two_stage_group_score(s_edge_calib, s_edge_normal)
    typed_score_normal = two_stage_group_score(s_typed_calib, s_typed_normal)

    def ablation_scores(z_dG, node_score, edge_score, typed_score):
        return {
            "A_prototype_only": z_dG,
            "B_prototype_plus_node": np.maximum(z_dG, node_score),
            "C_edge_only": edge_score,
            "D_full_v2": np.maximum(np.maximum(z_dG, node_score), edge_score),
            "E_typed_edge_only": typed_score,
            "F_full_v3": np.max(np.stack([z_dG, node_score, edge_score, typed_score], axis=1), axis=1),
        }

    scores_normal = ablation_scores(z_dG_normal, node_score_normal, edge_score_normal, typed_score_normal)

    print("\nscoring fault categories ...")
    report = {"config": {"embed_dim": EMBED_DIM, "num_prototypes": NUM_PROTOTYPES, "top_k": TOP_K,
                           "top_k_agg": TOP_K_AGG, "window_len": FIXED_LEN, "epochs": EPOCHS,
                           "prior_edges": PRIOR_EDGES, "edge_types": EDGE_TYPES,
                           "min_proto_samples": MIN_PROTO_SAMPLES},
               "nodes": node_names, "final_prior_bias_strength": model.edge_head.prior_bias_strength.item(),
               "final_typed_temperature": model.typed_head.log_temperature.exp().item(),
               "n_valid_prototypes_for_typed_calib": n_valid_proto,
               "fault_categories": {}}
    rows = {k: [] for k in scores_normal}
    for name, arr in faults.items():
        d_G_f, s_node_f, s_edge_f, s_typed_f, _, _ = per_sample_scores(model, arr)
        z_dG_f = zscore(d_G_f, d_G_calib)
        node_score_f = two_stage_group_score(s_node_calib, s_node_f)
        edge_score_f = two_stage_group_score(s_edge_calib, s_edge_f)
        typed_score_f = two_stage_group_score(s_typed_calib, s_typed_f)
        scores_fault = ablation_scores(z_dG_f, node_score_f, edge_score_f, typed_score_f)

        aurocs = {}
        for key in scores_normal:
            labels = np.concatenate([np.zeros(len(scores_normal[key])), np.ones(len(scores_fault[key]))])
            scores = np.concatenate([scores_normal[key], scores_fault[key]])
            auroc = float(sk_metrics.roc_auc_score(labels, scores))
            aurocs[key] = auroc
            rows[key].append((name, auroc))
        report["fault_categories"][name] = {"n": int(len(arr)), **aurocs}
        print(f"  {name:<20}n={len(arr):<5}" + "  ".join(f"{k}={v:.3f}" for k, v in aurocs.items()))

    print(f"\n{'method':<24}{'mean AUROC (12 categories)':>28}")
    summary = {}
    for key in scores_normal:
        mean_auroc = float(np.mean([r[1] for r in rows[key]]))
        summary[key] = mean_auroc
        print(f"{key:<24}{mean_auroc:>28.3f}")

    print(f"\nfinal prior_bias_strength: {model.edge_head.prior_bias_strength.item():.4f}")
    print(f"final typed-edge attention temperature: {model.typed_head.log_temperature.exp().item():.4f}")

    report["summary_mean_auroc"] = summary
    with open(OUT_DIR / "voraus_ad_joint_prototype_v3_report.json", "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "report": report},
                OUT_DIR / "voraus_ad_joint_prototype_v3.pth")
    print(f"\nsaved -> {OUT_DIR / 'voraus_ad_joint_prototype_v3_report.json'}")


if __name__ == "__main__":
    main()
