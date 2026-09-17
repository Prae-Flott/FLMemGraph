# FedCPG

Class prototype guided personalized lightweight federated learning
(Li, Wang, Cao, Li, Yi, Huang, *Computers in Industry* 164, 2025, 104180)
-- model-decoupled backbone/head + dual global/local class-prototype
contrastive losses. Supervised multi-class, NOT fit-on-healthy, same
bridged-AUROC caveat as FedPRO. See `benchmark/baselines/registry.py`'s
`"FedCPG"` entry and `memory/fedcpg-ufedhy-sota-baselines.md` for the full
mechanism, per-dataset results, and fidelity notes; paper PDF:
`docs/1-s2.0-S0166361524001088-main.pdf` (no public code found).

**Where the actual code lives**:
- Model: `src/models/baseline_models.py::FedCPGModel`.
- Algorithm: `benchmark/FedCPG/fedcpg.py` (global/local class-prototype
  computation, aggregation, and the dual contrastive loss).
- Per-dataset entry point: `main_fedcpg_baseline(...)` inside each
  `benchmark/run_<dataset>_bck_federated.py` (reuses FedPRO's own
  per-dataset supervised class-definition split).

## Usage
```
python3 benchmark/FedCPG/run_robo_fleet.py
python3 benchmark/FedCPG/run_paderborn.py
python3 benchmark/FedCPG/run_alfa.py
python3 benchmark/FedCPG/run_me_ad.py
```
