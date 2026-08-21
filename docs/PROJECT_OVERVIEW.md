# FLMemGraph

Federated, physics-informed, memory-augmented graph anomaly detection for
robot fleets. Standalone project, split out of `FL-bench` (the sibling
repo at `~/Projects/FL-bench`) on 2026-08-05. Built for robo3er, a
5-robot iRobot Create3 fleet with 4 labeled fault types (Broken Pipe,
cable trapped, Low battery, stuck).

See `docs/mem_phys_prompt_zh.md` for the original design spec (robo3er,
per-feature memory + latent structure head), and
`docs/joint_prototype_physics_gdn_anomaly_attention_prompt.md` for the
Joint Prototype Memory design this project's main anomaly-detection line
now implements as two consolidated final versions -- **V2**
(prototype + per-node deviation, no edges, mechanism-agnostic) and
**Scheme V3** (adds typed relation-specific edges, pays off only when a
verified physical relation exists) -- see `memory/scoring-signals-B-C-E-H.md`
for the current cross-dataset results, `memory/joint-prototype-scheme-v3.md`
for architecture rationale, and `memory/` generally for the rest of the
development history.

## Layout

```
FLMemGraph/
├── data/robo3er/        robo3er dataset (data.npy, targets.npy, metadata.json, partition.pkl)
├── data/sielaff/         second real dataset: 10 reverse-vending machines, windowed
│                         (data.npy, targets.npy, metadata.json, partition.pkl) -- generated
│                         upstream in FL-bench, copied here, raw CSVs in data/sielaff_data/
├── data/paderborn_bearing_data/  third real dataset: Paderborn KAt Bearing DataCenter,
│                         32 bearings (healthy/outer-ring/inner-ring/combined damage),
│                         motor current + vibration + mechanical channels, ~21GB
├── data/voraus_ad/       fourth real dataset: vorausrobotik 6-DOF pick-and-place arm,
│                         2122 samples, 12 named fault categories, 130 machine-data
│                         signals -- see memory/voraus-ad-dataset.md
├── src/                  all project code, organized by role:
│   ├── models/           gdn_model.py (GDN, no memory/federation), gdn_memory_model.py
│   │                     (discrete prototypical memory + memory-augmented GDN),
│   │                     conv_autoencoder.py (reconstruction AE, used by IFCAAE baseline),
│   │                     fl_model.py (federated memory+structure head), joint_prototype_model.py
│   │                     (V2 + Scheme V3 -- SharedEncoder, JointPrototypeMemory,
│   │                     TrendGraphAttentionHead, TypedRelationAnomalyHead, JointPrototypeGDNv3)
│   ├── robo3er/          dataset.py (loader), feature_groups.py, kinematics.py (physics
│   │                     residual), fl_dataset.py (per-client federated split),
│   │                     correlation_shift_analysis.py
│   ├── federated/        federated_memory.py (cross-robot codebook alignment)
│   ├── training/         train_fl_memory_gdn.py, train_gdn_memory.py, train_gdn_physics.py
│   │                     -- every train_*.py entry point
│   └── decision_logic.py decision logic (incl. three_level_decision), general-purpose,
│                         not tied to one dataset/model group
├── benchmark/            baseline methods + public-dataset registry, including real FedAvg,
│                         IFCAAE, GDN-tuned, V2/V3 runs per dataset, and a
│                         Sielaff-dataset GDN run (see benchmark/README.md for what's
│                         implemented vs. scaffolded)
├── checkpoints/          trained model weights + evaluation reports, one subfolder per
│                         dataset (checkpoints/robo3er/, checkpoints/paderborn/,
│                         checkpoints/sielaff/, checkpoints/voraus_ad/)
├── docs/                 design spec + this file
└── memory/               project history: what was tried, what worked, what broke and why
```

## Where things stand (2026-08-05)

- **Single-robot, physics-residual GDN** (`src/training/train_gdn_physics.py`):
  best `stuck` result in the project's history (AUROC ~0.72-0.74) via a
  differential-drive kinematic residual that sidesteps the shrinkage bias
  every prior forecasting-based method hit on that fault type.
- **Discrete prototypical memory** (`gdn_memory_model.py`,
  `train_gdn_memory.py`): adapted from FedPM (arXiv:2604.04475). Adding
  the memory bottleneck improved prediction-based detection on Broken
  Pipe (+0.15 AUROC) and stuck (+0.10) even without using the
  quantization-distance signal directly — an implicit regularization
  effect, not the originally-intended mechanism.
- **Federated multi-robot system** (superseded, `fl_model.py`/
  `train_fl_memory_gdn.py` since deleted 2026-08-16 in favor of
  `JointPrototypeV31`): unified shared encoder feeding a memory head
  (novelty) and a latent-space structure head (relational consistency) —
  the two-signal design from `docs/mem_phys_prompt_zh.md`. Federated over
  robo3er's real 5-robot non-IID split. cable trapped (0.92) and stuck
  (0.72) AUROC were strong; robot00 (smallest client, 195 fit windows) was
  a known weak point — the small-client instability it surfaced is
  discussed in `memory/joint-prototype-federated-results.md`.

## Running things

```bash
# single-robot physics-residual GDN
python3 src/training/train_gdn_physics.py

# + discrete memory
python3 src/training/train_gdn_memory.py

# federated multi-robot system (this project's own memory+structure design)
python3 src/training/train_fl_memory_gdn.py

# federated baselines, same model/data/schedule as train_fl_memory_gdn.py,
# only the aggregation rule differs -- direct ablation comparisons
python3 benchmark/run_fedavg_baseline.py     # full-parameter weighted averaging
python3 benchmark/run_ifcaae_baseline.py     # unsupervised clustered FL (IFCA adaptation)

# pooled/centralized baseline comparison harness (robo3er)
python3 benchmark/run_benchmark.py           # PCA / IsolationForest / GDN / GDN-tuned

# second real dataset (Sielaff reverse-vending machines)
python3 benchmark/run_sielaff_gdn.py
```

All training scripts write checkpoints + JSON evaluation reports to
`checkpoints/<dataset>/` (e.g. `checkpoints/robo3er/`, `checkpoints/paderborn/`,
`checkpoints/sielaff/`).
