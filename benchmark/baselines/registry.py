"""
FL baseline registry for the current (V3.1/FD+PD-mainline) pipeline -- see
`docs/baseline_selection_for_benchmark.md` for the selection rationale.
Supersedes `archive/benchmark/baselines/registry.py` (pre-V3.1, described
now-deleted model classes).

Each baseline gets its OWN minimal architecture and OWN single anomaly
score (see `src/models/baseline_models.py`'s module docstring) --
deliberately NOT this project's B/H/K-producing `JointPrototypeV31Forecast`/
`V21Forecast` backbone, so no baseline can accidentally inherit a
relational/memory/forecast signal its own algorithm was never designed to
have. Compare each baseline's single AUROC directly against `ours`'s
`FD_JD_PD_max`/`FD_JD_PD_lw` (the project mainline) -- see
`benchmark/compare_baselines.py`.

`status`: "implemented" (runnable now, in this repo) or "planned".
Run each via `--baseline {fedavg,ifcaae,fedexdnn,fedpro}` on any of
`benchmark/run_{robo_fleet,paderborn,alfa,me_ad}_bck_federated.py`
(dispatches to `main_faithful_baseline`, or `main_fedpro_baseline` for
`fedpro`, in each script -- NOT the "ours" `main()`). FedPRO runs on all
4 datasets (originally robo_fleet-only, extended to Paderborn/ALFA/ME-AD
with dataset-specific class definitions -- see its entry below), but
its results are supervised multi-class accuracy bridged into an AUROC,
NOT directly comparable to the other three baselines' fit-on-healthy
AUROC. Reproducible via `benchmark/compare_baselines.py --dataset
{robo_fleet,paderborn,alfa,me_ad}`.

Two more recent (2025) SOTA methods were added 2026-09-10, robo_fleet
first then extended to Paderborn/ALFA/ME-AD the same day -- all 4 datasets
now have `--baseline {fedcpg,ufedhy}` support (each
`run_*_bck_federated.py`):
  - **FedCPG** (supervised, personalized, model-decoupled -- same
    fit-on-ALL-labels category as FedPRO, see its entry below).
  - **uFedHy-DisMTSADD** (unsupervised federated hypernetwork -- fit-on-
    healthy, same directly-comparable-AUROC category as FedAvg/IFCAAE/
    Fed-ExDNN, see its entry below).
Both were built from the actual paper PDFs (`docs/1-s2.0-*-main.pdf`,
read in full), not abstracts -- see each entry for its own documented
scope cuts.
"""

BASELINES = {
    "FedAvg": {
        "category": "generic_fl", "status": "implemented",
        "module": "src.models.baseline_models.ReconOnlyModel (plain ConvAutoEncoder, "
                  "no memory/relational/forecast) + "
                  "src.federated.federated_memory.fedavg_state_dict_full, "
                  "driven via run_federated_rounds(mode='fedavg')",
        "note": "Standard FedAvg (McMahan et al., AISTATS 2017) over a PLAIN reconstruction "
                "model only -- encoder+decoder, no memory/prototypes, no relational/edge "
                "signal, no forecast head. One single global model every round, no "
                "personalization at all. Score: single GDN-style reconstruction-error AUROC "
                "(`federated_train_eval.gdn_score` on `per_node_recon_error`). Uses the SAME model + "
                "training loss as IFCAAE below -- the only difference between the two is "
                "aggregation topology (one global model vs. `num_clusters` clustered models), "
                "isolating 'does clustering help over plain FedAvg' as the single variable.",
        "results": {"robo_fleet": 0.885, "paderborn": 0.839, "alfa": 0.866, "me_ad": 0.617},
    },
    "IFCAAE": {
        "category": "federated_clustering", "status": "implemented",
        "module": "src.models.baseline_models.ReconOnlyModel (same as FedAvg) + "
                  "src.federated.federated_train_eval.recon_eval_loss, "
                  "driven via run_federated_rounds(mode='ifcaae')",
        "paper": "Ghosh, Chung, Yin, Ramchandran (IFCA), IEEE Trans. Information Theory 2022",
        "note": "Unsupervised, reconstruction-based IFCA adaptation over the SAME plain "
                "ConvAutoEncoder as FedAvg (no memory, no relational, no forecast): maintains "
                "`num_clusters` full global model replicas, hard-assigns each client per round "
                "to whichever replica reconstructs the client's own fit windows best (pure "
                "reconstruction MSE via `recon_eval_loss`), then FedAvg's within the assigned "
                "cluster. Score: single GDN-style reconstruction-error AUROC, same as FedAvg. "
                "Carries over the warmup + idle-cluster-respawn collapse-prevention fixes from "
                "the deleted `run_ifcaae_baseline.py` (`git show 9cc70fe`), rescaled for this "
                "project's 5-round budget (vs. the original 15-round robo3er-only run).",
        "results": {"robo_fleet": 0.880, "paderborn": 0.838, "alfa": 0.925, "me_ad": 0.631},
    },
    "Fed-ExDNN": {
        "category": "federated_memory", "status": "implemented",
        "module": "src.models.baseline_models.ExemplarOnlyModel (SharedEncoder + "
                  "JointPrototypeMemory + plain decoder, no relational/forecast) + "
                  "src.federated.federated_memory.align_and_split_fedcc, "
                  "driven via run_federated_rounds(mode='fedexdnn')",
        "note": "Related-work method judged 'most similar' to this project's own mechanism "
                "(local exemplar learning + server-side FedCC constrained clustering) -- see "
                "docs/baseline_selection_for_benchmark.md Sec 3.1. NO PUBLIC CODE EXISTS for "
                "Fed-ExDNN, so this is a DOCUMENTED APPROXIMATION: the local model is ONLY "
                "encoder + discrete VQ memory (this project's own `JointPrototypeMemory`, "
                "reused unchanged -- a local exemplar dictionary IS what that submodule "
                "already is) + a plain linear decoder for the reconstruction training signal, "
                "with NO relational/edge head and NO forecast head (unlike `ours`'s full "
                "architecture). Server-side alignment swaps `align_and_split`'s cosine-"
                "threshold + BFS single-linkage graph for k-means clustering with a hard "
                "per-cluster size cap (the 'constrained' part of FedCC) -- see "
                "`align_and_split_fedcc`'s docstring for the exact algorithm. Score: single "
                "GDN-style exemplar-deviation AUROC (`federated_train_eval.gdn_score` on "
                "`per_node_exemplar_deviation`'s `d_node`), NOT this project's own tuned "
                "`two_stage_group_score`/prototype-conditioned calibration. Treat this "
                "baseline's numbers as an approximation of Fed-ExDNN's mechanism, not a "
                "faithful reproduction of the original paper.",
        "results": {"robo_fleet": 0.801, "paderborn": 0.842, "alfa": 0.828, "me_ad": 0.597},
    },
    "FedPRO": {
        "category": "supervised_test_time_wrapper", "status": "implemented",
        "module": "src.models.baseline_models.FedPROModel + benchmark.FedPRO.fedpro, "
                  "driven via each run_*_bck_federated.py --baseline fedpro "
                  "(main_fedpro_baseline, all 3 datasets)",
        "paper": "Zhou, Yan, Nian, Liu, Wang, Chen, Theodoropoulos, Cheng, \"Prototype "
                 "Retrieval-Augmented Federated Learning System for Robust Intrusion "
                 "Detection,\" IEEE Trans. Computers, vol. 75, no. 8, Aug 2026. PDF: "
                 "docs/Prototype_Retrieval-Augmented_Federated_Learning_System_for_Robust_Intrusion_Detection.pdf. "
                 "Official code: github.com/zza234s/FedPRO",
        "note": "FUNDAMENTALLY DIFFERENT from the other three baselines: FedPRO is a "
                "test-time, plug-and-play wrapper around an ALREADY-TRAINED off-the-shelf FL "
                "classifier (here: plain FedAvg, cross-entropy), not a standalone trainable "
                "FL algorithm, and it is inherently SUPERVISED multi-class classification -- "
                "its margin-based prototype optimization (paper Eq. 5) requires multiple "
                "known classes to discriminate between, which collapses to nothing under "
                "this project's usual fit-on-healthy convention (only 'normal' ever seen "
                "during training). DELIBERATE, DOCUMENTED EXCEPTION to the federated-only/"
                "fit-on-healthy benchmark policy, run on all 3 datasets with dataset-specific "
                "class definitions: robo_fleet uses its 4 induced fault types (Normal + 4 = 5 "
                "classes) since each client (physical robot) owns its own fault occurrences, "
                "as does ALFA (Normal + 5 fault types = 6 classes, each client already IS one "
                "fault-type group so most clients see only 2 of 6 classes -- a natural non-IID "
                "split, no repartitioning needed). Paderborn's clients (6 healthy bearings) "
                "have NO fault data of their own by construction (unlike robo_fleet/ALFA) -- "
                "its 26 damaged-bearing FILES were round-robin partitioned across the 6 "
                "clients (client i gets `ALL_DAMAGED_CODES[i::6]`) and labeled by damage "
                "CATEGORY (healthy + outer_ring/inner_ring/combined = 4 classes), a documented "
                "non-standard reinterpretation of 'client' needed only for this supervised "
                "baseline -- every other baseline/`ours` keeps Paderborn's original client "
                "definition. All three use a per-client stratified 70/30 train/test split, "
                "not the usual fit/calib/test_normal/fault split. "
                "Mechanism: (1) plain FedAvg trains a shared encoder+classifier; (2) each "
                "client clusters its own frozen-encoder training embeddings per class into "
                "initial prototypes -- paper uses FINCH (parameter-free), we substitute a "
                "small K-means (`fedpro.build_initial_prototypes_kmeans`), justified by the "
                "paper's OWN ablation (Table V) finding K-means 'performs competitively' "
                "with FINCH; (3) margin-based hinge-loss refinement of ONLY the prototype "
                "tensor, encoder frozen (`fedpro.refine_prototypes_margin`, Eq. 5); (4) "
                "server just CONCATENATES all clients' prototypes into one global bank, no "
                "cross-client alignment (unlike `ours`'s `align_and_split` or Fed-ExDNN's "
                "FedCC); (5) test-time retrieval + similarity-weighted voting + confidence-"
                "weighted ensemble with the trained classifier's own prediction "
                "(`fedpro.retrieve_and_vote`/`confidence_ensemble`, Eq. 8-10). Reports BOTH "
                "the native multi-class `accuracy` metric (paper's own metric) AND a bridged "
                "binary anomaly score `1 - y_en[normal_class]` scored per-(client, fault-type) "
                "AUROC for comparability with `ours`'s FD+JD+PD -- but this AUROC is NOT "
                "apples-to-apples with the other three baselines' fit-on-healthy AUROC: "
                "FedPRO sees every fault type as a labeled training class, an easier task.",
        "results": {"robo_fleet": "0.976 (accuracy 0.834)", "paderborn": "0.965 (accuracy 0.617)", "alfa": "0.681 (accuracy 0.829)", "me_ad": "0.616 (accuracy 0.575)"},
    },
    "FedCPG": {
        "category": "supervised_personalized_fl", "status": "implemented (all 4 datasets)",
        "module": "src.models.baseline_models.FedCPGModel + benchmark.FedCPG.fedcpg, "
                  "driven via each run_*_bck_federated.py --baseline fedcpg "
                  "(main_fedcpg_baseline, all 4 datasets)",
        "paper": "Li, Wang, Cao, Li, Yi, Huang, \"FedCPG: A class prototype guided "
                 "personalized lightweight federated learning framework for cross-factory "
                 "fault detection,\" Computers in Industry, vol. 164, 104180, 2025. PDF: "
                 "docs/1-s2.0-S0166361524001088-main.pdf. No public code found.",
        "note": "Same category as FedPRO: SUPERVISED multi-class personalized FL, not "
                "fit-on-healthy, robo_fleet-only for the same reason (class-prototype "
                "contrast needs multiple known classes at training time). Model-decoupled: "
                "a backbone (`encoder`+`proj`, FedAvg'd every round, paper Eq. 2) + a "
                "personalized `head` classifier NEVER aggregated (stays local per client, "
                "paper Sec 3.2 Step 3). Local objective (Eq. 14) is CE + alpha*l_g + "
                "beta*l_p: `l_g` (Eq. 12) is a supervised-contrastive loss pulling each "
                "sample toward the GLOBAL class prototype of its label (server-aggregated "
                "across clients weighted by class count, Eq. 3-4) and away from other "
                "classes' global prototypes; `l_p` (Eq. 13) is the same contrastive form "
                "against that client's OWN local class prototypes (recomputed once per local "
                "epoch, frozen for that epoch), preventing the personalized head from "
                "drifting entirely to the global model's class geometry. OWN-MINIMAL-"
                "ARCHITECTURE substitution: the paper's own backbone is a wide-kernel-CNN + "
                "lightweight-MLP ('light-WMLP', Fig. 4) purpose-built for raw single-channel "
                "vibration signal; substituted with this project's own flatten-then-MLP "
                "encoder (same one FedPROModel uses) since robo_fleet's windows are already "
                "multivariate feature-group tensors -- the paper's OWN mechanism (backbone/"
                "head split + dual class-prototype contrastive losses) is reproduced "
                "faithfully, only the backbone's internal layers differ. Reports BOTH native "
                "multi-class `accuracy` (paper's own metric) AND a bridged binary anomaly "
                "score `1 - P(normal)` per-(client, fault-type) AUROC, same convention and "
                "same NOT-apples-to-apples caveat as FedPRO's bridged AUROC.",
        "results": {"robo_fleet": "0.983 (accuracy 0.838)", "paderborn": "0.973 (accuracy 0.710)",
                    "alfa": "0.705 (accuracy 0.854)", "me_ad": "0.713 (accuracy 0.663)"},
    },
    "uFedHy-DisMTSADD": {
        "category": "federated_hypernetwork", "status": "implemented (all 4 datasets)",
        "module": "src.models.baseline_models.SCNorTransformerModel + Hypernetwork, "
                  "benchmark.uFedHy_DisMTSADD.ufedhy, driven via each run_*_bck_federated.py "
                  "--baseline ufedhy (main_ufedhy_baseline, all 4 datasets)",
        "paper": "Hao, Chen, Chen, Li, \"Effectively detecting and diagnosing distributed "
                 "multivariate time series anomalies via Unsupervised Federated "
                 "Hypernetwork,\" Information Processing & Management, vol. 62, 104107, "
                 "2025. PDF: docs/1-s2.0-S0306457325000494-main.pdf. Official code: "
                 "github.com/Hjfyoyo/uFedHy-DisMTSADD (not consulted -- built from the paper "
                 "text alone).",
        "note": "Unsupervised, fit-on-healthy -- same category as FedAvg/IFCAAE/Fed-ExDNN "
                "(unlike FedPRO/FedCPG), so this is the directly-comparable-AUROC group. Two "
                "halves: (1) client-side SC Nor-Transformer (`SCNorTransformerModel`) -- "
                "per-window series normalization (Eq. 7-9, z-score over the time axis, undone "
                "at the output, Eq. 10) + series CONVERSION embedding (Eq. 11: each channel's "
                "whole window becomes ONE token, i.e. variate-as-token, unlike a vanilla "
                "Transformer's per-timestep tokens) + an encoder-only Transformer block "
                "(self-attention + LayerNorm + feed-forward) + a projection back to a "
                "per-channel reconstruction; (2) a federated hypernetwork (`Hypernetwork`, "
                "pFedHN-style, Shamsian et al. ICML 2021 -- the paper's own cited foundation) "
                "that generates a client's ENTIRE target-network weights from a per-client "
                "learnable embedding each round, trained via the paper's own simplified "
                "update rule (Eq. 5-6: after local SGD from the generated weights, ONE more "
                "hypernetwork forward pass with gradient minimizes MSE against the post-"
                "local-training weights -- not full second-order unrolled backprop through "
                "the local steps). Score: per-node reconstruction-error AUROC, same GDN-style "
                "convention as FedAvg/IFCAAE/Fed-ExDNN (`per_node_recon_error`+`gdn_score`). "
                "TWO DOCUMENTED SCOPE CUTS: (a) 1 encoder layer, not the paper's 3 (the layer "
                "count multiplies the hypernetwork's own per-tensor output-head count, kept "
                "small enough to train in this project's usual compute budget); (b) "
                "localization uses this project's own per-node argmax convention, NOT the "
                "paper's PC-algorithm-causal-graph + PageRank root-cause ranking (a materially "
                "different, non-differentiable module, out of scope here).",
        "results": {"robo_fleet": 0.728, "paderborn": 0.845, "alfa": 0.752, "me_ad": 0.610},
    },
}


def implemented():
    return {k: v for k, v in BASELINES.items() if v["status"] == "implemented"}


if __name__ == "__main__":
    for name, info in BASELINES.items():
        print(f"{name:<12} status={info['status']:<12} category={info['category']}")
