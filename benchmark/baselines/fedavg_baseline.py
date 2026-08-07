"""
Standard FedAvg (McMahan et al., AISTATS 2017), as the generic-FL baseline
for `mem_phys_prompt_zh.md` Sec 8.2 category (4) ("FedAvg/FedProx/SCAFFOLD/
MOON -- 通用FL基线"). This is a real, standalone reimplementation, NOT a
pointer into `~/Projects/FL-bench`'s `src/server/fedavg.py` -- that
implementation is entangled with FL-bench's hydra config, ray parallelism,
and CrossEntropyLoss-classification assumptions (`FedAvgClient.fit()` calls
`self.criterion(logit, y)` on hard labels), none of which fit this
project's unsupervised, whole-window-reconstruction anomaly setting.
`benchmark/baselines/registry.py` previously listed "FedAvg" as
`status: "implemented"` pointing at that sibling-repo module, which was
never actually runnable from inside this repo -- that was a stale/false
claim, fixed by this file.

The FedAvg *algorithm* itself is exactly the two-line thing McMahan's paper
describes: after each client trains locally, the server produces the new
global model as clients' parameter-wise weighted average
(weight = client dataset size), then broadcasts it back for the next
round's local training to start from. See `federated_average` below.

## Why this is the right FedAvg baseline for THIS project specifically

`src/train_fl_memory_gdn.py` (this project's own system) already uses the
exact same architecture (`fl_model.FLGDNMemory`: shared encoder -> memory
head + structure head) and the exact same 5-robot non-IID client split
(`fl_dataset.load_fl_clients`) -- the ONLY thing that differs is what gets
aggregated across clients each round:

  - ours:   ONLY the discrete memory codebook, via cosine-similarity
            clustering + utility-diversity ranking (`federated_memory.
            align_and_split`) -- encoder and structure head stay
            permanently local/personalized per client.
  - FedAvg: EVERY parameter (encoder + memory codebook + structure head +
            decoder), via plain size-weighted averaging -- one single
            global model shared by all clients after each round, no
            personalization at all.

Running both over the identical model/data/rounds/local-epochs isolates
"communication strategy" as the single variable, which is exactly what
Sec 8.4's ablation table asks for ("参数平均(FedAvg on encoder) vs 只传字典").
"""
from collections import OrderedDict
from typing import Sequence

import torch


def federated_average(
    state_dicts: Sequence[OrderedDict], weights: Sequence[float]
) -> OrderedDict:
    """The FedAvg aggregation rule: parameter-wise weighted mean across
    clients' full model state dicts. `weights` is typically each client's
    local training-set size (McMahan et al.'s n_k / n convention) -- larger
    clients pull the global model further toward their own local optimum,
    which is precisely the "one dominant client can swamp the others"
    failure mode this project's memory-only federation (Sec 3/4) is
    designed to avoid on severely non-IID splits like robo3er's (robot04
    alone is 79% of all data).

    All state dicts must have identical keys/shapes (same model
    architecture across clients, standard FedAvg assumption -- no
    personalization layers)."""
    total = float(sum(weights))
    norm_weights = [w / total for w in weights]

    avg = OrderedDict()
    for key in state_dicts[0].keys():
        stacked = torch.stack(
            [sd[key].float() * w for sd, w in zip(state_dicts, norm_weights)], dim=0
        )
        avg[key] = stacked.sum(dim=0).to(state_dicts[0][key].dtype)
    return avg
