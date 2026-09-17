# Fed-ExDNN

Local-exemplar-memory + FedCC-style constrained-clustering baseline
(documented approximation, no public code exists for Fed-ExDNN itself) --
see `benchmark/baselines/registry.py`'s `"Fed-ExDNN"` entry for the full
description and per-dataset AUROC results.

**Where the actual code lives**:
- Model: `src/models/baseline_models.py::ExemplarOnlyModel` (`SharedEncoder`
  + this project's own `JointPrototypeMemory` + a plain linear decoder).
- Aggregation: `src/federated/federated_memory.py::align_and_split_fedcc`,
  driven via `src/federated/federated_train_eval.py::run_federated_rounds(mode="fedexdnn")`.
- Per-dataset entry point: `main_faithful_baseline("fedexdnn", ...)` inside
  each `benchmark/run_<dataset>_bck_federated.py`.

## Usage
```
python3 benchmark/Fed-ExDNN/run_robo_fleet.py [--num-prototypes N]
python3 benchmark/Fed-ExDNN/run_paderborn.py [--num-prototypes N]
python3 benchmark/Fed-ExDNN/run_alfa.py [--num-prototypes N]
python3 benchmark/Fed-ExDNN/run_me_ad.py [--num-prototypes N]
```
