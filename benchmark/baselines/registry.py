"""
Baseline method registry, transcribed from `mem_phys_prompt_zh.md` Sec 8.2.
Four categories:
  ① traditional unsupervised (statistical/shallow) -- cheap sanity floor
  ② deep reconstruction/predictive UAD -- "one signal" competitors, exactly
     what Sec 2B's design argues is insufficient on its own
  ③ Transformer/graph-based deep UAD -- GDN itself lives here, MUST be run
     as the "no memory, no federation" ablation baseline
  ④ federated methods -- the real competitive set (FedKO/FedKAD/FedDyMem/
     DP-DGAD/FEDPM)

`status`: "implemented" (runnable in this repo right now), "planned" (not
yet implemented, no code here), "external" (would need the original
authors' code, not reimplemented here).
"""

BASELINES = {
    # ------------------------------------------------------------- ① --
    "PCA": {"category": "①", "type": "traditional", "status": "implemented",
             "module": "benchmark.baselines.traditional.PCABaseline"},
    "IsolationForest": {"category": "①", "type": "traditional", "status": "implemented",
                          "module": "benchmark.baselines.traditional.IsolationForestBaseline"},
    "KNN": {"category": "①", "type": "traditional", "status": "planned"},
    "OCSVM": {"category": "①", "type": "traditional", "status": "planned"},
    "DAGMM": {"category": "①", "type": "traditional", "status": "planned",
               "paper": "Zong et al., ICLR 2018"},
    # ------------------------------------------------------------- ② --
    "AE": {"category": "②", "type": "deep_reconstruction", "status": "implemented",
            "module": "auto_encoder.train_centralized_ae (this project's own pooled ConvAutoEncoder)"},
    "LSTM-VAE": {"category": "②", "type": "deep_reconstruction", "status": "planned"},
    "USAD": {"category": "②", "type": "deep_reconstruction", "status": "planned",
              "paper": "Audibert et al., KDD 2020"},
    "OmniAnomaly": {"category": "②", "type": "deep_reconstruction", "status": "planned",
                      "paper": "Su et al., KDD 2019", "code": "https://github.com/NetManAIOps/OmniAnomaly"},
    "MSCRED": {"category": "②", "type": "deep_reconstruction", "status": "planned"},
    "MAD-GAN": {"category": "②", "type": "deep_reconstruction", "status": "planned"},
    "DeepSVDD": {"category": "②", "type": "deep_reconstruction", "status": "planned"},
    "THOC": {"category": "②", "type": "deep_reconstruction", "status": "planned"},
    "InterFusion": {"category": "②", "type": "deep_reconstruction", "status": "planned"},
    # ------------------------------------------------------------- ③ --
    "GDN": {"category": "③", "type": "graph_structure", "status": "implemented",
             "module": "src.gdn_model.GDN",
             "note": "This project's own graph-attention forecaster -- run as the mandatory "
                      "'no memory, no federation' structure-only ablation baseline."},
    "Paderborn-fl-model-single-machine": {"category": "④", "type": "federated_uad", "status": "implemented",
                                "module": "benchmark/run_paderborn_fl_model.py, reuses src.fl_model.FLGDNMemory unchanged",
                                "note": "This project's full memory+structure pipeline minus federation, "
                                         "on the Paderborn KAt bearing dataset (datasets/registry.py's "
                                         "Paderborn-KAt-bearing entry). Unlike the deleted IMS bearing "
                                         "exploration, this dataset has definite per-bearing damage "
                                         "ground truth, so evaluation uses this project's normal "
                                         "fit-on-healthy/AUROC-per-fault-class convention (not a trend/"
                                         "ranking substitute). Result: strong category-mean AUROC "
                                         "(combined 1.000, inner_ring 0.750, outer_ring 0.641) but a real "
                                         "bimodal split -- 15/26 damaged bearings near-perfect, 10/26 "
                                         "BELOW 0.5 AUROC (score direction inverted, mostly artificial-"
                                         "damage bearings), leading hypothesis: crude box-average "
                                         "decimation (125x) washes out the high-frequency impulsive "
                                         "signature artificial single-point defects rely on, while real "
                                         "fatigue damage's broader signature survives -- untested. See "
                                         "memory/paderborn-fl-model-run.md for the full per-bearing table."},
    "Paderborn-physics-residual-gdn": {"category": "③", "type": "physics_prior", "status": "implemented",
                                "module": "benchmark/run_paderborn_physics_gdn.py, src/paderborn_physics.py",
                                "note": "Same method as robo3er's src.train_gdn_physics -- fit a physical "
                                         "relation (motor current envelope ~= a*torque + b*speed + c) from "
                                         "healthy FIT-split data only, replace phase_current_1/2 with the "
                                         "residual, train plain GDN (forecasting, no memory yet), single-"
                                         "variable comparison against raw current on the same architecture/"
                                         "seed. Result: an honest NEGATIVE, opposite of robo3er's kinematics "
                                         "residual -- 23/26 damaged bearings got WORSE (outer_ring mean AUROC "
                                         "0.697->0.598, inner_ring 0.819->0.751). Root cause: the fitted "
                                         "relation's R^2 is only ~0.25 (torque/speed explain barely a "
                                         "quarter of current-envelope variance normally), unlike "
                                         "kinematics.py's near-deterministic wheel-velocity->odometry "
                                         "relation -- residualizing against a WEAK prediction injects "
                                         "regression noise instead of removing genuine explained variance. "
                                         "See memory/paderborn-physics-residual-gdn.md and "
                                         "benchmark/datasets/paderborn_physics.md for the full analysis."},
    "Paderborn-torque-residual-gdn": {"category": "③", "type": "physics_prior", "status": "implemented",
                                "module": "benchmark/run_paderborn_torque_residual_gdn.py, src/paderborn_physics.py",
                                "note": "Unified torque-balance physics residual, per user design direction: "
                                         "predicted_torque = a*force + b*speed + c (fit on healthy data), "
                                         "residual added as a 4th node alongside vibration_1+phase_current_1/2. "
                                         "Result: ALSO a negative (10/26 bearings worse, 3 improved, 13 "
                                         "unchanged at ceiling) -- the fitted force coefficient came out "
                                         "NEGATIVE (physically implausible), traced to the dataset's 4 "
                                         "discrete operating conditions NOT varying force/speed/torque "
                                         "independently (force takes only 2 values, confounded with the "
                                         "other two) -- a 2-predictor OLS fit across 4 confounded design "
                                         "points can't reliably separate the physical effects, the same root "
                                         "cause behind the earlier current-residual's weak R^2. Third "
                                         "consecutive negative physics-prior result on this dataset -- see "
                                         "memory/paderborn-torque-residual-gdn.md and "
                                         "benchmark/datasets/paderborn_physics.md Section 3. Follow-up "
                                         "(memory/paderborn-fixed-mu-torque-residual.md): re-fit with the "
                                         "force coefficient FIXED at a published catalog friction "
                                         "coefficient (mu=0.0014) instead of freely fit -- confirms the "
                                         "collinearity diagnosis (every bearing that got worse got LESS "
                                         "worse with the fixed, non-confounded coefficient) but still net "
                                         "negative (1 improved/12 worse vs free-fit's 3/10) -- R^2 dropped "
                                         "further to 0.089, showing bearing friction is genuinely a small "
                                         "fraction of this rig's total shaft torque, not just a "
                                         "collinearity artifact."},
    "Paderborn-joint-prototype": {"category": "③", "type": "physics_prior", "status": "implemented",
                                "module": "benchmark/run_paderborn_joint_prototype.py, "
                                           "src.joint_prototype_model.JointPrototypeGDN",
                                "note": "Pivots from parameter-level (exact-magnitude) physics residuals "
                                         "-- 3 consecutive negative results, see Paderborn-torque-"
                                         "residual-gdn -- to TREND-based edge relations, per "
                                         "docs/joint_prototype_three_level_anomaly_prompt.md. "
                                         "JointPrototypeMemory: M prototypes are FULL [N,D] joint-state "
                                         "snapshots (not per-feature codebooks). TrendEdgeHead predicts "
                                         "each edge target's DEVIATION from its own matched-prototype "
                                         "baseline from the source's deviation from ITS baseline -- "
                                         "relative, not absolute magnitude, sidestepping the confounded-"
                                         "4-operating-condition collinearity that sank every prior "
                                         "residual attempt. Initial result (vs. a 3-channel GDN baseline, "
                                         "not apples-to-apples): beat that baseline on outer_ring/"
                                         "inner_ring. CORRECTED by a fair same-6-channel rerun "
                                         "(benchmark/run_paderborn_6ch_comparison.py, "
                                         "memory/paderborn-6ch-fair-comparison.md): plain GDN on the "
                                         "SAME 6 channels actually beats JointPrototype on every "
                                         "category (mean AUROC 0.819 vs 0.802) -- the added complexity "
                                         "doesn't clearly pay for itself once compared fairly. Its real "
                                         "value proposition is node/edge-level localization and "
                                         "interpretability, not raw detection AUROC. See "
                                         "memory/paderborn-joint-prototype.md for the ablation table "
                                         "and memory/paderborn-6ch-fair-comparison.md for the corrected "
                                         "comparison."},
    "Paderborn-joint-prototype-v2-attention": {"category": "③", "type": "physics_prior", "status": "implemented",
                                "module": "benchmark/run_paderborn_joint_prototype_v2.py, "
                                           "src.joint_prototype_model.JointPrototypeGDNv2",
                                "note": "Design correction to Paderborn-joint-prototype (v1): edge/"
                                         "structural anomaly now comes from GDN-STYLE LEARNED ATTENTION "
                                         "over neighbors (TrendGraphAttentionHead, same mechanism as "
                                         "gdn_model.GDN/fl_model.StructureHead), not a fixed 8-edge list "
                                         "with one linear map each. Declared physics edges bias attention "
                                         "logits (one learned scalar) but do NOT restrict which "
                                         "relationships can be learned -- undeclared edges remain fully "
                                         "learnable. Still operates on deviations from the matched joint "
                                         "prototype (not raw magnitudes), keeping v1's sound response to "
                                         "the confounded-4-operating-condition collinearity problem. "
                                         "Result: FIRST CLEAR, SUBSTANTIAL WIN in this dataset's entire "
                                         "physics-prior exploration -- beats the fair 6-channel GDN "
                                         "baseline on every category (mean AUROC 0.873 vs 0.819, "
                                         "outer_ring 0.829 vs 0.757, inner_ring 0.888 vs 0.838), 11/26 "
                                         "bearings improved vs. only 2 with tiny regressions. See "
                                         "memory/paderborn-joint-prototype-v2-attention.md for the full "
                                         "table and design rationale."},
    "GDN-tuned": {"category": "③", "type": "graph_structure", "status": "implemented",
                   "module": "benchmark.baselines.gdn_baseline.GDNTunedBaseline",
                   "note": "Migrated from ~/Projects/FL-bench's test_gdn/train_gdn.py -- the actual "
                            "tuned hyperparameters (history_len=30, calib-MSE best-checkpoint "
                            "selection) that produced this project's best-effort GDN-alone number. "
                            "Previously only the model class + a pre-trained checkpoint were carried "
                            "into this repo, not the training script that produced them -- fixed."},
    "AnomalyTransformer": {"category": "③", "type": "transformer", "status": "planned",
                             "paper": "Xu et al., ICLR 2022"},
    "TranAD": {"category": "③", "type": "transformer", "status": "planned",
                "paper": "Tuli et al., VLDB 2022"},
    "ModernTCN": {"category": "③", "type": "transformer", "status": "planned"},
    "MTAD-GAT": {"category": "③", "type": "graph_structure", "status": "planned",
                  "paper": "Zhao et al., ICDM 2020",
                  "note": "Feature-oriented + time-oriented dual graph attention -- close cousin of "
                           "this project's own Sec 2A discussion of adding a temporal self-attention branch."},
    "GTA": {"category": "③", "type": "graph_structure", "status": "planned"},
    # ------------------------------------------------------------- ④ --
    "FedAvg": {"category": "④", "type": "federated_generic", "status": "implemented",
                "module": "benchmark.baselines.fedavg_baseline.federated_average, "
                          "driver benchmark/run_fedavg_baseline.py",
                "note": "Standalone reimplementation (McMahan et al., AISTATS 2017) over this "
                         "project's own FLGDNMemory architecture and 5-robot client split -- "
                         "identical to src.train_fl_memory_gdn except full-parameter weighted "
                         "averaging replaces memory-only codebook alignment. Previously listed as "
                         "'implemented' pointing at ~/Projects/FL-bench's src.server.fedavg, which "
                         "was never actually runnable from this repo -- that was a stale claim, "
                         "now fixed with a real in-repo baseline."},
    "FedProx": {"category": "④", "type": "federated_generic", "status": "planned",
                 "note": "Not migrated -- FL-bench's src.server.fedprox is entangled with the same "
                          "hydra/ray/classification framework FedAvg was; would need the same "
                          "standalone-reimplementation treatment as fedavg_baseline.py."},
    "IFCAAE": {"category": "④", "type": "federated_clustering", "status": "implemented",
                "module": "benchmark/run_ifcaae_baseline.py, model src.conv_autoencoder.ConvAutoEncoder",
                "paper": "Ghosh, Chung, Yin, Ramchandran (IFCA), IEEE Trans. Information Theory 2022",
                "note": "Standalone reimplementation of FL-bench's src/server|client/ifcaae.py -- "
                         "unsupervised, reconstruction-based IFCA adaptation (label-free clustering + "
                         "training). Not in the original Sec 8.2 taxonomy but the closest existing "
                         "clustering-based FL competitor to this project's codebook-alignment "
                         "approach: tests whether letting clients self-organize into a small number "
                         "of shared clusters (no per-node discrete memory, no relational/physical "
                         "consistency signal) is enough on robo3er's non-IID split. Carries over the "
                         "two collapse-prevention fixes from the original build (warmup-before-split, "
                         "idle-cluster respawn) -- both verified still necessary here."},
    "FedKO": {"category": "④", "type": "federated_uad", "status": "external",
               "paper": "arXiv:2503.11255"},
    "FedKAD": {"category": "④", "type": "federated_uad", "status": "external",
                "paper": "arXiv:2607.08978"},
    "FedDyMem": {"category": "④", "type": "federated_uad_vision", "status": "external",
                  "paper": "arXiv:2502.21012"},
    "DP-DGAD": {"category": "④", "type": "federated_dynamic_graph", "status": "external",
                 "paper": "arXiv:2508.00664"},
    "FEDPM": {"category": "④", "type": "federated_forecast_memory", "status": "external",
               "paper": "arXiv:2604.04475",
               "note": "The paper this project's memory module is adapted from -- see "
                        "src/gdn_memory_model.py."},
    "ours_gdn_memory": {"category": "④", "type": "federated_uad", "status": "implemented",
                          "module": "src.train_fl_memory_gdn",
                          "note": "This project's own memory+structure+federation system."},
}


def by_category(cat):
    return {k: v for k, v in BASELINES.items() if v["category"] == cat}


def implemented():
    return {k: v for k, v in BASELINES.items() if v["status"] == "implemented"}


if __name__ == "__main__":
    for cat in ["①", "②", "③", "④"]:
        print(f"\n=== category {cat} ===")
        for name, info in by_category(cat).items():
            print(f"  {name:<20} status={info['status']}")
