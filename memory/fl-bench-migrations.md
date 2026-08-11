# Migrations from `~/Projects/FL-bench` (2026-08-06)

FL-bench is the sibling repo this project was originally split out of
(`docs/PROJECT_OVERVIEW.md`'s top note, 2026-08-05). It still holds ~40 FL
algorithms and several pieces of code/data this project doesn't have a
copy of. This session migrated four of them into FLMemGraph as real,
standalone, in-repo runnable pieces -- not pointers back into FL-bench.

## Why "standalone reimplementation," not a direct port

FL-bench's `src/server|client/*.py` methods are entangled with its own
framework: hydra config, ray parallelism, and (for FedAvg/FedProx/etc.)
`CrossEntropyLoss`-classification assumptions baked into `FedAvgClient.fit()`.
None of that fits this project's unsupervised, whole-window-reconstruction
anomaly setting. Every migration below re-derives just the *algorithm*
(the actual aggregation/clustering rule) as a small standalone module, then
wires it into this project's own model architecture and data loader
(`fl_model.FLGDNMemory`, `fl_dataset.load_fl_clients`) instead of dragging
the framework over.

## What was migrated

1. **FedAvg** (`benchmark/baselines/fedavg_baseline.py`'s
   `federated_average`, driver `benchmark/run_fedavg_baseline.py`).
   `benchmark/baselines/registry.py` previously listed FedAvg as
   `status: "implemented"` pointing at FL-bench's `src.server.fedavg` --
   that was never actually runnable from inside this repo, a stale/false
   claim, now fixed. Runs the IDENTICAL model + 5-robot client split as
   `src/training/train_fl_memory_gdn.py`, only the aggregation rule changes: full-
   parameter size-weighted averaging (one shared global model) instead of
   memory-only codebook alignment. Result: FedAvg loses 0.16+ AUROC on
   `stuck` (robot04's fault, but robot04 is 79% of all data) vs. our
   memory-only alignment -- consistent with the design doc's hypothesis
   that full-parameter averaging lets a dominant client's local optimum
   swamp the others, which memory-only alignment structurally avoids
   (encoder + structure head stay personalized per client).

2. **GDN-tuned** (`benchmark/baselines/gdn_baseline.GDNTunedBaseline`).
   `benchmark/baselines/gdn_baseline.GDNBaseline` (pre-existing) was
   always explicitly a "lightweight" variant; the actual tuned
   hyperparameters (history_len=30 not 10, calib-MSE best-checkpoint
   selection across epochs) that produced this project's best-effort
   GDN-alone number lived only in FL-bench's `test_gdn/train_gdn.py`,
   never carried over (only the model class + a pre-trained checkpoint
   were). Now wired into `benchmark/run_benchmark.py`. Beats the
   lightweight variant on 3/4 fault types (cable trapped +0.056, Low
   battery +0.050, stuck +0.069; Broken Pipe -0.010).

3. **IFCAAE** (`benchmark/run_ifcaae_baseline.py`, model
   `src/conv_autoencoder.ConvAutoEncoder`). Unsupervised, reconstruction-
   based adaptation of IFCA (Ghosh et al. 2022) -- clients hard-assign to
   whichever of `num_clusters` global models reconstructs their own
   normal data best, no labels used for training or clustering. Carries
   over both collapse-prevention fixes documented in FL-bench's
   `memory/ifcaae-federated-clustering.md`: warmup rounds before
   splitting into perturbed clusters (starting every cluster beyond
   cluster 0 from independent random init makes round-1 assignment
   arbitrary and can strand a cluster forever), and respawning clusters
   nobody picks for `respawn_patience` rounds. Verified this
   reimplementation reproduces the ORIGINAL run's finding exactly:
   `num_clusters=2` converges to robot04 alone in one cluster, the other
   four robots together in the other -- not a coincidence, robot04's data
   consistently shows a different operating regime in every diagnostic
   run through this whole project's history (also the 79%-of-data
   dominant client). AUROC is weaker than both our system and FedAvg on
   every fault type except `stuck` (0.632 -- between FedAvg's 0.553 and
   ours' 0.717), consistent with IFCAAE having neither a discrete-memory
   novelty signal nor a structure-consistency signal, only reconstruction
   error under a coarse cluster split.

4. **Sielaff dataset + tuned GDN** (`data/sielaff/` -- windowed data
   copied from FL-bench, generated there by `data/generate_sielaff_data.py`
   from the raw CSVs already in this repo's `data/sielaff_data/`, not
   regenerated here; `benchmark/run_sielaff_gdn.py`, migrated from
   FL-bench's `test_gdn/train_sielaff_gdn.py`). This project's second
   real, independent dataset (10 reverse-vending machines, 39 features, 5
   fault classes) -- until now `data/sielaff_data/` only had raw CSVs with
   no windowed/trainable form anywhere in this repo. Reuses this
   project's own `src/gdn_model.GDN` (verified byte-identical to
   FL-bench's copy) rather than duplicating it; adds a Sielaff-specific
   reliability mask (features some machines never report, e.g.
   RingCamera columns, get excluded from the max()-over-features score
   instead of letting their near-zero calib IQR dominate it). AUROC
   0.76-0.93 across all 5 fault classes on first run, all with n in the
   hundreds-to-thousands -- a much larger, more stable per-class sample
   than robo3er's per-client federated splits ever have.

## What was deliberately NOT migrated this session

FedProx and the other ~35 FL-bench algorithms remain untouched --
`benchmark/baselines/registry.py`'s FedProx entry documents it would need
the same standalone-reimplementation treatment as `fedavg_baseline.py`.
SHAP interpretability scripts (`scripts/shap_robo3er.py`,
`compare_shap_gdn.py`) and Sielaff's hierarchical/multilabel classifiers
(`scripts/train_sielaff_hierarchical.py`, `train_sielaff_multilabel.py`)
were identified as candidates but not migrated -- flagged for a future
session if attribution-stability analysis or Sielaff's richer label
schemes become relevant here.
