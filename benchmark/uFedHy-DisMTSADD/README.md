# uFedHy-DisMTSADD

Unsupervised Federated Hypernetwork Method for Distributed Multivariate
Time Series Anomaly Detection and Diagnosis (Hao, Chen, Chen, Li,
*Information Processing & Management* 62, 2025, 104107) -- a per-client
learnable embedding + shared hypernetwork generates each client's entire
SC Nor-Transformer weights every round (not FedAvg). Unsupervised,
fit-on-healthy, directly comparable to FedAvg/IFCAAE/Fed-ExDNN's AUROC.
See `benchmark/baselines/registry.py`'s `"uFedHy-DisMTSADD"` entry and
`memory/fedcpg-ufedhy-sota-baselines.md` for the full mechanism,
per-dataset results, and documented scope cuts; paper PDF:
`docs/1-s2.0-S0306457325000494-main.pdf` (official code claimed at
github.com/Hjfyoyo/uFedHy-DisMTSADD, not consulted).

**Where the actual code lives**:
- Models: `src/models/baseline_models.py::SCNorTransformerModel` (client
  target network) + `Hypernetwork` (server-side weight generator).
- Algorithm: `benchmark/uFedHy-DisMTSADD/ufedhy.py` (the paper's own
  simplified generate -> local-SGD -> MSE-distillation hypernetwork update rule).
- Per-dataset entry point: `main_ufedhy_baseline(...)` inside each
  `benchmark/run_<dataset>_bck_federated.py`.

## Usage
```
python3 benchmark/uFedHy-DisMTSADD/run_robo_fleet.py
python3 benchmark/uFedHy-DisMTSADD/run_paderborn.py
python3 benchmark/uFedHy-DisMTSADD/run_alfa.py
python3 benchmark/uFedHy-DisMTSADD/run_me_ad.py
```
