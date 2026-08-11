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
             "module": "src.models.gdn_model.GDN",
             "note": "This project's own graph-attention forecaster -- run as the mandatory "
                      "'no memory, no federation' structure-only ablation baseline."},
    "Paderborn-fl-model-single-machine": {"category": "④", "type": "federated_uad", "status": "implemented",
                                "module": "benchmark/run_paderborn_fl_model.py, reuses src.models.fl_model.FLGDNMemory unchanged",
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
    "joint-prototype-scheme-v2": {"category": "③", "type": "physics_prior", "status": "implemented",
                                "module": "src.models.joint_prototype_model.SharedEncoder + JointPrototypeMemory "
                                           "(no edge head). Sielaff: benchmark/run_sielaff_joint_prototype_v2.py. "
                                           "Also present as the A/B ablation columns inside each dataset's "
                                           "Scheme V3 run script (Paderborn/robo3er/voraus-AD).",
                                "note": "V2: Joint Prototype Memory only (device-level + per-node "
                                         "deviation from the matched joint prototype), NO edges/relations at "
                                         "all -- mechanism-agnostic, needs no physics prior. Best or "
                                         "near-best signal on 3 of 4 datasets tested: Sielaff (0.978 mean "
                                         "AUROC, beats the existing tuned-GDN baseline by +0.091 with ZERO "
                                         "physics prior -- this dataset has none), robo3er (0.942, clear "
                                         "best), voraus-AD (0.756, wins/ties 11 of 12 fault categories). "
                                         "Loses clearly only on Paderborn, the one dataset with literature-"
                                         "verified physical relations a real fault mechanism actually breaks "
                                         "-- see joint-prototype-scheme-v3. Recommended default: run V2 "
                                         "first on any new dataset before investing in physics-relation "
                                         "analysis. See memory/joint-prototype-scheme-v2.md for the full "
                                         "cross-dataset table and per-category pattern."},
    "joint-prototype-scheme-v3": {"category": "③", "type": "physics_prior", "status": "implemented",
                                "module": "src.models.joint_prototype_model.JointPrototypeGDNv3 (TrendGraphAttentionHead "
                                           "+ TypedRelationAnomalyHead). Paderborn: "
                                           "benchmark/run_paderborn_joint_prototype_v3.py. robo3er: "
                                           "benchmark/run_robo3er_joint_prototype_v3.py. voraus-AD: "
                                           "benchmark/run_voraus_ad_joint_prototype_v3.py.",
                                "note": "Scheme V3: V2 + GDN-style learned attention over declared-edge-"
                                         "biased neighbors + relation-specific typed message functions + "
                                         "prototype-conditioned edge-residual standardization + anomaly "
                                         "attention. Needs domain knowledge: a verified physical relation "
                                         "between two signals must be declared as an edge (proportional/"
                                         "nonlinear) before this adds anything V2 doesn't already give. "
                                         "Lineage (intermediate code removed, conclusions preserved): a "
                                         "physics-residual precursor on Paderborn (OLS-fit current/torque "
                                         "residual, 3 consecutive negative results, root cause: Paderborn's 4 "
                                         "operating conditions confound the predictors) motivated operating on "
                                         "DEVIATIONS from a matched joint prototype instead of raw magnitudes "
                                         "-> v1 (fixed edge list, one linear map/edge) did NOT beat a fair "
                                         "GDN baseline (0.802 vs 0.819) -> v2 (learned attention, physics "
                                         "edges as an attention-logit bias not a restriction) was the first "
                                         "clear win (0.873 vs 0.819) -> v3 (this version) adds typed relations "
                                         "on top. RESULT: v3 wins clearly ONLY on Paderborn (0.904 edge-signal "
                                         "vs 0.819 GDN baseline) -- the one dataset with textbook physical "
                                         "relations (bearing vibration/force/torque/current coupling) a real "
                                         "fault mechanism breaks. On robo3er and voraus-AD, V2 (no edges) "
                                         "wins instead -- see joint-prototype-scheme-v2. Also documents the "
                                         "max-aggregation noise-floor problem (expanding voraus-AD's graph "
                                         "18->66 nodes made results WORSE under raw max() despite adding a "
                                         "physically correct friction edge) and its fix (top-k-mean + a second "
                                         "calibration stage against the calib split's own distribution, "
                                         "recovering and exceeding the original prediction). See "
                                         "memory/joint-prototype-scheme-v3.md for the full cross-dataset table, "
                                         "lineage detail, and the aggregation fix."},
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
                         "identical to src.training.train_fl_memory_gdn except full-parameter weighted "
                         "averaging replaces memory-only codebook alignment. Previously listed as "
                         "'implemented' pointing at ~/Projects/FL-bench's src.server.fedavg, which "
                         "was never actually runnable from this repo -- that was a stale claim, "
                         "now fixed with a real in-repo baseline."},
    "FedProx": {"category": "④", "type": "federated_generic", "status": "planned",
                 "note": "Not migrated -- FL-bench's src.server.fedprox is entangled with the same "
                          "hydra/ray/classification framework FedAvg was; would need the same "
                          "standalone-reimplementation treatment as fedavg_baseline.py."},
    "IFCAAE": {"category": "④", "type": "federated_clustering", "status": "implemented",
                "module": "benchmark/run_ifcaae_baseline.py, model src.models.conv_autoencoder.ConvAutoEncoder",
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
                        "src/models/gdn_memory_model.py."},
    "ours_gdn_memory": {"category": "④", "type": "federated_uad", "status": "implemented",
                          "module": "src.training.train_fl_memory_gdn",
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
