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

**Physics-residual GDN** (`python3 run_paderborn_physics_gdn.py`): same
method as robo3er's `src/train_gdn_physics.py` -- fit motor
current-envelope ~= a·torque + b·speed + c from healthy data only
(`src/paderborn_physics.py`), replace `phase_current_1/2` with the
residual, train plain `GDN` (forecasting, no memory), single-variable
comparison against raw current on the same architecture/seed. Result:
an honest NEGATIVE, opposite of robo3er's result -- 23/26 damaged
bearings got WORSE (outer_ring mean AUROC 0.697->0.598, inner_ring
0.819->0.751). Root cause: the fitted relation's R^2 is only ~0.25
(torque/speed explain barely a quarter of current-envelope variance
normally) vs. `kinematics.py`'s near-deterministic wheel-velocity->
odometry relation -- residualizing against a WEAK prediction injects
regression noise rather than removing genuine explained variance, a
generalizable lesson about when this residual pattern helps.

**Unified torque-balance residual** (`python3 run_paderborn_torque_residual_gdn.py`):
follow-up per a design conversation that speed/torque/friction/load
torque should be ONE equation, not separate chains -- `predicted_torque
= a·force + b·speed + c` (force = independently, externally-applied
radial load; `speed`'s level used, not `dω/dt`, since shaft speed is
essentially constant within each 4s recording -- checked directly, std/
mean < 0.03%), residual added as a 4th node alongside the 3 raw
channels. ALSO negative: 10/26 bearings got worse, only 3 improved.
Root cause traced further this time: **the fitted force coefficient came
out NEGATIVE** (physically implausible -- more load should mean more
friction, not less), because Paderborn's 4 operating conditions don't
vary force/speed/torque independently (force only takes 2 values,
confounded with the other two across just 4 discrete design points) --
a 2-predictor OLS fit can't reliably separate the physical effects from
which-condition-this-is. This explains BOTH physics-residual failures on
this dataset as one shared methodological root cause. Per-dataset
physical-prior reference docs (formulas + what's verified vs. untried)
for every dataset in this project, including ones with no hard prior
found: `benchmark/datasets/robo3er_physics.md`,
`benchmark/datasets/sielaff_physics.md`,
`benchmark/datasets/paderborn_physics.md`. Full analysis:
`memory/paderborn-physics-residual-gdn.md`,
`memory/paderborn-torque-residual-gdn.md`.

**Joint Prototype Memory + trend-based edges** (`python3 run_paderborn_joint_prototype.py`):
after 3 consecutive negative parameter-level-residual results, pivoted to
a redesign -- `docs/joint_prototype_three_level_anomaly_prompt.md` (new
design doc). `src/joint_prototype_model.py`'s `JointPrototypeMemory`
(prototypes are FULL `[N,D]` joint-state snapshots -- "has the whole
device been in this operating regime before," not per-feature novelty)
+ `TrendEdgeHead` (predicts a node's DEVIATION from its own
matched-prototype baseline from another node's deviation, via a learned
linear map -- relative co-movement, not an absolute-magnitude
regression). All 6 real-time channels used as nodes (most of any
Paderborn script so far), 8 physics-skeleton edges. Runs the design
doc's 4 mandatory ablations (prototype-only / +node / edge-only / full).
**Initial result** (vs. a 3-channel GDN baseline): beat that baseline on
outer_ring (0.697->0.736) and inner_ring (0.819->0.841), though not
uniformly (7/26 bearings improved, 6/26 worse). Full ablation table and
per-bearing breakdown: `memory/paderborn-joint-prototype.md`.

**CORRECTED: fair 6-channel comparison** (`python3 run_paderborn_6ch_comparison.py`):
the comparison above wasn't apples-to-apples -- JointPrototype had 3 more
input channels (force/speed/torque) than the GDN baseline it beat.
Retrained plain `GDN` and `ConvAutoEncoder` on the SAME 6 channels.
**Plain GDN actually beats Joint Prototype on every category** (mean
AUROC 0.819 vs. 0.802; outer_ring 0.757 vs. 0.736; inner_ring 0.838 vs.
0.823) -- the added complexity doesn't clearly pay for itself in raw
detection accuracy once compared fairly, reported honestly rather than
reframed. `TrendEdgeHead`'s mechanism (predicting deviations, not
magnitudes) remains a methodologically sound response to the confounded-
operating-conditions problem, but its real value on this dataset is
node/edge-level localization/interpretability, not AUROC. Also found:
AE flipped from strongest (in earlier 3-channel comparisons) to weakest
(0.679 mean) with 6 channels -- an open question why. Full comparison:
`memory/paderborn-6ch-fair-comparison.md`.

**v2: GDN-style attention edge head -- the first clear win** (`python3 run_paderborn_joint_prototype_v2.py`):
design correction per user feedback -- edge/structural anomaly should
come from GDN-STYLE LEARNED ATTENTION over neighbors
(`TrendGraphAttentionHead`, same mechanism as
`gdn_model.GDN`/`fl_model.StructureHead`), not v1's fixed 8-edge list
with one linear map each. Physics-known edges bias attention logits (one
learned scalar) but do NOT restrict which relationships can be learned --
undeclared relationships stay fully learnable ("对于物理先验没有表示的边，
GDN也可以学习他们之间的关系"). Still operates on deviations from the
matched joint prototype, keeping v1's sound response to the confounded-
operating-conditions collinearity problem, now paired with a mechanism
that can actually exploit it. **Result: beats the fair 6-channel GDN
baseline on every category** (mean AUROC 0.873 vs. 0.819, outer_ring
0.829 vs. 0.757, inner_ring 0.888 vs. 0.838) -- 11/26 bearings improved,
only 2 with tiny regressions (-0.031, -0.021), 13 unchanged at ceiling.
The `edge-only` ablation alone (0.861) already beats the GDN baseline
before adding memory/node signals. First unambiguous win in this
dataset's entire physics-prior exploration. Full table and design
rationale: `memory/paderborn-joint-prototype-v2-attention.md`.

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
