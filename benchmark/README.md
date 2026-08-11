# benchmark/

Baseline + public-dataset scaffold for evaluating `src/`'s
"记忆检索 + 物理验证" federated fault-detection system
(`mem_phys_prompt_zh.md`), built out per that design doc's Sec 8 plan.

## What's actually implemented vs. scaffolded

**Implemented and runnable right now** (`python3 run_benchmark.py`):
- Dataset: robo3er (pooled/centralized), via `datasets/robo3er_adapter.py`
- Baselines: PCA reconstruction (`baselines/traditional.py`),
  IsolationForest (`baselines/traditional.py`), GDN
  (`baselines/gdn_baseline.GDNBaseline`, lightweight variant for fast
  smoke runs) and **GDN-tuned** (`baselines/gdn_baseline.GDNTunedBaseline`,
  migrated from FL-bench's `test_gdn/train_gdn.py` -- history_len=30 +
  calib-MSE best-checkpoint selection, the actual tuned hyperparameters;
  previously only the model class + a pre-trained checkpoint were carried
  into this repo, not the training procedure that produced them). Sample
  run (pooled robo3er, AUROC):

  | method | Broken Pipe | cable trapped | Low battery | stuck |
  |---|---|---|---|---|
  | PCA | 0.581 | 0.997 | 0.995 | 0.686 |
  | IsolationForest | 0.871 | 0.992 | 0.991 | 0.581 |
  | GDN (lightweight) | 0.939 | 0.894 | 0.926 | 0.531 |
  | GDN-tuned | 0.929 | 0.950 | 0.976 | 0.600 |

- This project's own federated memory+structure system:
  `src/train_fl_memory_gdn.py` (not wired into
  `run_benchmark.py` yet since it needs the per-client, not pooled, data
  loader -- run it separately)
- **FedAvg baseline** (`python3 run_fedavg_baseline.py`): standalone
  reimplementation of standard FedAvg (McMahan et al., AISTATS 2017,
  `baselines/fedavg_baseline.py`'s `federated_average`), run over the
  IDENTICAL model (`fl_model.FLGDNMemory`) and 5-robot client split as
  `train_fl_memory_gdn.py` -- the only variable that changes is the
  aggregation rule: full-parameter size-weighted averaging every round
  (one shared global model, no personalization) instead of memory-only
  codebook alignment (encoder + structure head stay local/personalized).
  This is a real in-repo baseline, not a pointer into
  `~/Projects/FL-bench`'s `src/server/fedavg.py` -- that module is
  entangled with FL-bench's hydra/ray/classification-loss framework and
  was never actually runnable from this repo (the registry previously
  claimed "implemented" against it, which was false -- fixed here).
  First-run comparison (same 5 rounds x 20 local epochs, kinematic_core
  feature set, AUROC(d+r)):

  | fault | ours (memory-only fed.) | FedAvg (full-param avg.) |
  |---|---|---|
  | cable trapped (robot01) | 0.922 | 0.940 |
  | stuck (robot04) | **0.717** | 0.553 |
  | Broken Pipe (robot03) | 0.520 | 0.494 |
  | Broken Pipe (robot00) | 0.000 | 0.163 |
  | Low battery (robot00) | 0.344 | 0.430 |

  On `stuck` -- robot04's fault, but robot04 is 79% of all data by
  volume -- FedAvg loses over 0.16 AUROC vs. our memory-only alignment,
  consistent with the design doc's Sec 8.4 hypothesis: full-parameter
  averaging lets the dominant client's local optimum swamp the smaller
  clients' encoders/structure heads (which FedAvg forces to be shared,
  unlike our design), while memory-only alignment leaves each client's
  encoder/structure head fully personalized and only shares the
  discrete-regime vocabulary. robot00 (smallest client, 195 fit windows)
  moves in both directions depending on fault type -- not enough signal
  yet to call a consistent winner there, matches the already-documented
  small-client instability (`memory/fl-memory-physics-system.md`).

- **IFCAAE baseline** (`python3 run_ifcaae_baseline.py`): standalone
  reimplementation of FL-bench's `src/server|client/ifcaae.py` -- an
  unsupervised, reconstruction-based adaptation of IFCA (Ghosh et al.,
  2022) that maintains `num_clusters` independent global models and
  hard-assigns each client to whichever reconstructs its own normal data
  best, no labels used for training or clustering. Not in the original
  Sec 8.2 taxonomy but the closest existing clustering-based FL
  competitor to this project's codebook-alignment approach -- tests
  whether letting clients self-organize into a small number of shared
  clusters (no per-node discrete memory, no relational/physical
  consistency signal) is enough on robo3er's non-IID split. Carries over
  the two collapse-prevention fixes from the original FL-bench build
  (`memory/ifcaae-federated-clustering.md` there): warmup rounds before
  splitting into clusters (starting every cluster beyond cluster 0 from
  independent random init makes round-1 assignment arbitrary and can
  strand a cluster forever), and respawning clusters nobody picks for
  `respawn_patience` rounds. Reproduces that original run's finding: with
  `num_clusters=2`, converges to robot04 alone in one cluster and the
  other four robots together in the other (robot04's data consistently
  shows a different operating regime -- see `fl_dataset.py`'s docstring,
  it's also the 79%-of-data dominant client). AUROC per client (single
  reconstruction-error signal, no d/r split since IFCAAE has no memory or
  structure head):

  | client | fault | AUROC |
  |---|---|---|
  | robot01 | cable trapped | 0.891 |
  | robot04 | stuck | 0.632 |
  | robot03 | Broken Pipe | 0.621 |
  | robot00 | Broken Pipe | 0.000 |
  | robot00 | Low battery | 0.248 |

  Weaker than both our system and FedAvg on every fault type except
  `stuck` (0.632, between FedAvg's 0.553 and ours' 0.717) -- consistent
  with IFCAAE having neither a discrete-memory novelty signal nor a
  structure-consistency signal, only plain reconstruction error under a
  coarse 2-cluster split.

## Second dataset: Sielaff (`python3 run_sielaff_gdn.py`)

10 reverse-vending-machine reject/journal/error logs (`data/sielaff_data/`
raw CSVs already lived in this repo; the derived windowed dataset --
`data/sielaff/{data.npy,targets.npy,metadata.json,partition.pkl}`,
39 features, 8-step windows, 5 fault classes -- was copied over from
FL-bench, where it was generated by `data/generate_sielaff_data.py`, not
re-run here). Not one of Sec 8.1's registry datasets (this project's own
second real dataset, alongside robo3er), so it isn't in
`datasets/registry.py` -- `run_sielaff_gdn.py` is a standalone script,
not wired into `run_benchmark.py`'s pooled-dataset loop.

Migrated from FL-bench's `test_gdn/train_sielaff_gdn.py`, reusing THIS
project's own `src/gdn_model.GDN` (verified byte-identical to FL-bench's
copy) rather than duplicating the model -- same tuned-GDN protocol as
`GDNTunedBaseline` (dense sliding, calib-MSE checkpoint selection) but
with Sielaff's own history_len=4/window_size=8, plus a Sielaff-specific
reliability mask (`RELIABILITY_RATIO`) that excludes features some
machines never report (e.g. RingCamera columns) from the max()-over-
features score, so their near-zero calib IQR can't dominate it. Sample
run (AUROC, all 5 fault classes have `n` in the hundreds-to-thousands,
enough for a stable pooled estimate unlike robo3er's per-client federated
splits):

| fault | n | AUROC |
|---|---|---|
| bottle_recognition | 3812 | 0.926 |
| mechanical_jam | 116 | 0.916 |
| compactor_status | 414 | 0.902 |
| other | 274 | 0.841 |
| crate_recognition | 138 | 0.763 |

## Fourth dataset: Paderborn KAt bearing

`data/paderborn_bearing_data/{K001..,KA..,KB..,KI..}/` -- 32 type-6203
ball bearing experiments (~21GB, gitignored like all of `data/`), fetched
from the official KAt DataCenter mirror
(`https://groups.uni-paderborn.de/kat/BearingDataCenter/`, no
registration). Unlike a run-to-failure dataset, this one has DEFINITE
per-bearing damage-class ground truth: 6 healthy, 12 artificial-damage,
14 real accelerated-lifetime-damage bearings, each with a verified damage
code (KA=outer-ring, KI=inner-ring, KB=combined) -- structurally close to
robo3er/Sielaff's fit-on-normal/per-class-AUROC convention. Also includes
motor current channels alongside vibration, a new sensing modality for
this project. Reference paper + all 64 per-bearing PDF fact
sheets/measuring logs are in `docs/paderborn_bearing_KAt2016.pdf` /
`docs/paderborn_bearing_facts/`. Full verified damage-code taxonomy and
channel/sampling-rate details: `memory/paderborn-bearing-dataset.md`.

**Single-machine full pipeline** (`python3 run_paderborn_fl_model.py`):
this project's memory+structure design minus federation, reusing
`src/fl_model.FLGDNMemory` unchanged, fit+calibrated on the 6 healthy
bearings only (pooled), evaluated as AUROC per damaged bearing code --
this project's NORMAL convention (definite ground truth, no trend/ranking
substitute needed). Category-mean AUROC looks strong (combined 1.000,
inner_ring 0.750, outer_ring 0.641) but hides a real bimodal split: 15 of
26 damaged bearings near-perfect (AUROC(d+r) >= 0.999), 10 BELOW 0.5
(score direction inverted -- when flipped, most recover to 0.64-0.76),
overwhelmingly artificial-damage bearings (8 of the 10). Leading
hypothesis: the 125x box-average decimation used here washes out the
high-frequency impulsive signature artificial single-point defects rely
on, while real fatigue damage's broader signature survives -- a real,
physically-grounded hypothesis but untested this session. Two data bugs
found+fixed while building `paderborn_adapter.py` (non-constant per-file
sample counts across the 2560 `.mat` files; one file that fails to parse
with scipy). Full per-bearing table and follow-ups:
`memory/paderborn-fl-model-run.md`.

**Joint Prototype Memory: Scheme B and Scheme V3** -- this project's main
physics-informed anomaly detection line, now consolidated into two final
versions after several intermediate iterations (fixed-edge lists,
parameter-level physics residuals) were superseded and removed (code +
checkpoints deleted, conclusions preserved in memory):

- **Scheme B** (`src/joint_prototype_model.SharedEncoder` +
  `JointPrototypeMemory`, no edges/relations at all -- mechanism-
  agnostic, needs no physics prior): best or near-best signal on 3 of 4
  datasets tested -- Sielaff (`run_sielaff_joint_prototype_b.py`, 0.978
  mean AUROC, beats the dataset's existing tuned-GDN baseline by +0.091
  with ZERO physics prior, since none exists for this dataset), robo3er
  (0.942), voraus-AD (0.756, wins/ties 11 of 12 fault categories).
  Recommended default for any new dataset before investing in physics-
  relation analysis. Full cross-dataset table:
  `memory/joint-prototype-scheme-b.md`.
- **Scheme V3** (`JointPrototypeGDNv3`: Scheme B + GDN-style learned
  attention over declared-edge-biased neighbors + typed relation-
  specific message functions + prototype-conditioned edge-residual
  standardization + anomaly attention): wins clearly ONLY on Paderborn
  (`run_paderborn_joint_prototype_v3.py`, 0.904 mean AUROC on the
  edge-attention signal vs. a fair 0.819 GDN baseline) -- the one
  dataset with literature-verified physical relations (bearing
  vibration/force/torque/current coupling) a real fault mechanism
  actually breaks. Also run on robo3er (`run_robo3er_joint_prototype_v3.py`)
  and voraus-AD (`run_voraus_ad_joint_prototype_v3.py`), where Scheme B
  wins instead. Needs domain knowledge: a verified physical relation
  must be declared as an edge before this adds anything Scheme B
  doesn't already give -- per-dataset physical-prior reference docs
  (formulas + what's verified vs. untried, including datasets with no
  hard prior found) are in `benchmark/datasets/robo3er_physics.md`,
  `benchmark/datasets/sielaff_physics.md`,
  `benchmark/datasets/paderborn_physics.md`,
  `benchmark/datasets/voraus_ad_physics.md`. Full cross-dataset table,
  condensed lineage, and the max-aggregation noise-floor fix found while
  expanding voraus-AD's graph: `memory/joint-prototype-scheme-v3.md`.

**Scaffolded, not implemented** (registries document what's needed):
- `datasets/registry.py`: all of Sec 8.1's public datasets (SWaT, WADI,
  SMD, SMAP, MSL, PSM, GECCO, MVTec-AD, MPDD, VisA, Bitcoin-Alpha,
  Wikipedia, MOOC, ETTh1, Electricity, Weather, Exchange) with category,
  source paper, and how to obtain each one. NONE of these are downloaded
  into this repo yet -- SWaT/WADI need manual iTrust registration
  (cannot be automated), MVTec-AD/MPDD/VisA are multi-GB, the rest
  (SMD/SMAP/MSL/PSM/GECCO/the graph datasets/the ETT family) are freely
  downloadable but were not fetched this session. Run
  `python3 datasets/registry.py` for the full table with status per
  dataset.
- `baselines/registry.py`: every baseline from Sec 8.2 (traditional:
  KNN/OCSVM/DAGMM; deep reconstruction: LSTM-VAE/USAD/OmniAnomaly/MSCRED/
  MAD-GAN/DeepSVDD/THOC/InterFusion; transformer/graph: AnomalyTransformer/
  TranAD/ModernTCN/MTAD-GAT/GTA; federated: FedKO/FedKAD/FedDyMem/
  DP-DGAD/FEDPM) with status (`implemented`/`planned`/`external`). Most
  are `planned` or `external` -- these would need either reimplementation
  or the original authors' code, neither done here. Run
  `python3 baselines/registry.py` for the full table.

## Why this scope, not the full suite

Implementing ~20 baseline methods and downloading 8+ external benchmark
datasets (several gated behind manual registration or multi-GB) is a
multi-week effort, not a single session's work. This scaffold prioritizes:
1. A **working, runnable harness** end-to-end (dataset -> baseline ->
   metrics -> report), proven against the one dataset already on disk
   (robo3er), so extending it is "add one adapter/wrapper," not "design
   the harness."
2. **Complete, accurate registries** of what Sec 8 asked for, so nothing
   from the design doc is silently dropped -- every dataset and method is
   at least named, sourced, and status-tracked, even where not
   implemented.
3. The **mandatory ablations** (Sec 8.4) that don't need new external
   data: GDN alone (no memory, no federation) is implemented as a
   baseline; the memory+structure ablations (d-only / r-only / d+r) are
   already built into `src/train_fl_memory_gdn.py`'s own
   evaluation, not duplicated here.

## Extending this

- **New dataset**: add a `datasets/<name>_adapter.py` exposing the same
  `load() -> (fit, calib, test_normal, {fault: windows}, cols)` interface
  as `robo3er_adapter.py`, flip its `registry.py` entry's status to
  `"downloaded"` once fetched.
- **New baseline**: implement a `fit(...)`/`score(...)` class (see
  `baselines/traditional.py` for the minimal interface), register it in
  `baselines/registry.py`, add it to `run_benchmark.py`'s `methods` dict.
- **Metrics**: `metrics.py` already implements F1/PA-F1/AUROC/AUPRC per
  Sec 8.3 -- PA-F1 is computed but flagged as not the primary number (see
  its docstring for why: point-adjustment is known-inflatable on
  segment-heavy datasets like SWaT).
