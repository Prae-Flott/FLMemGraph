# FLMemGraph

Federated, physics-informed, memory-augmented graph anomaly detection for
robot fleets. Standalone project, split out of `FL-bench` (the sibling
repo at `~/Projects/FL-bench`) on 2026-08-05. Built for robo3er, a
5-robot iRobot Create3 fleet with 4 labeled fault types (Broken Pipe,
cable trapped, Low battery, stuck).

See `docs/mem_phys_prompt_zh.md` for the full design spec this project
implements, and `memory/` for the development history, bugs found/fixed,
and results at each stage.

## Layout

```
FLMemGraph/
├── data/robo3er/        robo3er dataset (data.npy, targets.npy, metadata.json, partition.pkl)
├── src/                  all project code: gdn_model.py (GDN class, no memory/federation),
│                         kinematics residual, feature groups, discrete prototypical memory,
│                         latent-space structure head, federated cross-robot memory alignment,
│                         decision logic, and every train_*.py entry point
├── benchmark/            baseline methods + public-dataset registry (mostly scaffolded,
│                         see benchmark/README.md for what's actually implemented)
├── checkpoints/          trained model weights + evaluation reports from every run
├── docs/                 design spec + this file
└── memory/               project history: what was tried, what worked, what broke and why
```

## Where things stand (2026-08-05)

- **Single-robot, physics-residual GDN** (`src/train_gdn_physics.py`):
  best `stuck` result in the project's history (AUROC ~0.72-0.74) via a
  differential-drive kinematic residual that sidesteps the shrinkage bias
  every prior forecasting-based method hit on that fault type.
- **Discrete prototypical memory** (`gdn_memory_model.py`,
  `train_gdn_memory.py`): adapted from FedPM (arXiv:2604.04475). Adding
  the memory bottleneck improved prediction-based detection on Broken
  Pipe (+0.15 AUROC) and stuck (+0.10) even without using the
  quantization-distance signal directly — an implicit regularization
  effect, not the originally-intended mechanism.
- **Federated multi-robot system** (`fl_model.py`, `federated_memory.py`,
  `decision_logic.py`, `train_fl_memory_gdn.py`): unified shared encoder
  feeding a memory head (novelty) and a latent-space structure head
  (relational consistency) — the two-signal design from
  `docs/mem_phys_prompt_zh.md`. Federated over robo3er's real 5-robot
  non-IID split. cable trapped (0.92) and stuck (0.72) AUROC are strong;
  robot00 (smallest client, 195 fit windows) is a known weak point —
  see `memory/fl-memory-physics-system.md` for the full bug history.

## Running things

```bash
# single-robot physics-residual GDN
python3 src/train_gdn_physics.py

# + discrete memory
python3 src/train_gdn_memory.py

# federated multi-robot system
python3 src/train_fl_memory_gdn.py

# baseline comparison harness
python3 benchmark/run_benchmark.py
```

All training scripts write checkpoints + JSON evaluation reports to
`checkpoints/`.
