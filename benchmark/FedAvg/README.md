# FedAvg

Standard FedAvg (McMahan et al., AISTATS 2017) baseline -- see
`benchmark/baselines/registry.py`'s `"FedAvg"` entry for the full
description and per-dataset AUROC results.

**Where the actual code lives** (these runners are thin delegators, not
reimplementations):
- Model: `src/models/baseline_models.py::ReconOnlyModel` (plain
  `ConvAutoEncoder`, no memory/relational/forecast).
- Aggregation: `src/federated/federated_memory.py::fedavg_state_dict_full`,
  driven via `src/federated/federated_train_eval.py::run_federated_rounds(mode="fedavg")`.
- Per-dataset entry point: `main_faithful_baseline("fedavg", ...)` inside
  each `benchmark/run_<dataset>_bck_federated.py`.

## Usage
```
python3 benchmark/FedAvg/run_robo_fleet.py
python3 benchmark/FedAvg/run_paderborn.py
python3 benchmark/FedAvg/run_alfa.py
python3 benchmark/FedAvg/run_me_ad.py
```
Each accepts `--out-suffix NAME` (report filename suffix) and
`--num-clusters N` (unused by FedAvg itself, accepted for CLI parity with
IFCAAE's runner).
