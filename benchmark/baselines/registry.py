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
             "module": "test_gdn.gdn_model.GDN",
             "note": "This project's own graph-attention forecaster -- run as the mandatory "
                      "'no memory, no federation' structure-only ablation baseline."},
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
                "module": "src.server.fedavg (this project's own FL-bench framework)"},
    "FedProx": {"category": "④", "type": "federated_generic", "status": "implemented",
                 "module": "src.server.fedprox"},
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
                        "test_gdn_physi/gdn_memory_model.py."},
    "ours_gdn_memory": {"category": "④", "type": "federated_uad", "status": "implemented",
                          "module": "test_gdn_physi.train_fl_memory_gdn",
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
