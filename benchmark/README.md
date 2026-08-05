# benchmark/

Baseline + public-dataset scaffold for evaluating `src/`'s
"记忆检索 + 物理验证" federated fault-detection system
(`mem_phys_prompt_zh.md`), built out per that design doc's Sec 8 plan.

## What's actually implemented vs. scaffolded

**Implemented and runnable right now** (`python3 run_benchmark.py`):
- Dataset: robo3er (pooled/centralized), via `datasets/robo3er_adapter.py`
- Baselines: PCA reconstruction (`baselines/traditional.py`),
  IsolationForest (`baselines/traditional.py`), GDN
  (`baselines/gdn_baseline.py`, lightweight variant -- use
  FL-bench's `test_gdn/train_gdn.py` directly for the tuned number, not
  carried into this repo, only `gdn_model.py` and its checkpoint were)
- This project's own federated memory+structure system:
  `src/train_fl_memory_gdn.py` (not wired into
  `run_benchmark.py` yet since it needs the per-client, not pooled, data
  loader -- run it separately)

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
