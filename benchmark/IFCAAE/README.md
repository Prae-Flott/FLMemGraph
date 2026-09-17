# IFCAAE

Unsupervised, reconstruction-based IFCA adaptation (Ghosh, Chung, Yin,
Ramchandran, IEEE Trans. Information Theory 2022) -- see
`benchmark/baselines/registry.py`'s `"IFCAAE"` entry for the full
description and per-dataset AUROC results.

**Where the actual code lives**:
- Model: `src/models/baseline_models.py::ReconOnlyModel` (same as FedAvg).
- Aggregation: `src/federated/federated_train_eval.py::run_federated_rounds(mode="ifcaae")`
  (clustering logic + warmup/respawn fixes live there).
- Per-dataset entry point: `main_faithful_baseline("ifcaae", ...)` inside
  each `benchmark/run_<dataset>_bck_federated.py`.

## Usage
```
python3 benchmark/IFCAAE/run_robo_fleet.py [--num-clusters N]
python3 benchmark/IFCAAE/run_paderborn.py [--num-clusters N]
python3 benchmark/IFCAAE/run_alfa.py [--num-clusters N]
python3 benchmark/IFCAAE/run_me_ad.py [--num-clusters N]
```
