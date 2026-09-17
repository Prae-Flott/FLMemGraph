# FedPRO

Prototype Retrieval-Augmented Federated Learning (Zhou et al., IEEE Trans.
Computers, Aug 2026) -- test-time wrapper around an off-the-shelf FedAvg
classifier. Supervised multi-class, NOT fit-on-healthy, its AUROC is a
bridged score, not apples-to-apples with FedAvg/IFCAAE/Fed-ExDNN/
uFedHy-DisMTSADD. See `benchmark/baselines/registry.py`'s `"FedPRO"` entry
for the full description, per-dataset class definitions, and results;
official code: github.com/zza234s/FedPRO; paper PDF:
`docs/Prototype_Retrieval-Augmented_Federated_Learning_System_for_Robust_Intrusion_Detection.pdf`.

**Where the actual code lives**:
- Model: `src/models/baseline_models.py::FedPROModel`.
- Algorithm: `benchmark/FedPRO/fedpro.py` (K-means initial prototypes +
  margin refinement + retrieval-augmented ensemble).
- Per-dataset entry point: `main_fedpro_baseline(...)` inside each
  `benchmark/run_<dataset>_bck_federated.py` (each with its own dataset-
  specific supervised class-definition adaptation -- see that function's
  docstring in each script).

## Usage
```
python3 benchmark/FedPRO/run_robo_fleet.py
python3 benchmark/FedPRO/run_paderborn.py
python3 benchmark/FedPRO/run_alfa.py
python3 benchmark/FedPRO/run_me_ad.py
```
