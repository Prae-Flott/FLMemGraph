---
name: benchmark-policy-federated-only
description: Standing policy (2026-08-17) -- all benchmark runs in this project must use the federated (FL) variant, never centralized, going forward
metadata:
  type: project
---

# Policy: benchmarks are FL-only from 2026-08-17 onward

**All future benchmark runs in this project must use the FEDERATED
script variant** (`*_federated.py`), not the centralized one. This
project's name (FLMemGraph) and its actual research question (federated
memory exchange, `src/federated/federated_memory.py`) are about the
federated setting -- centralized runs are a debugging/ablation tool, not
the thing being measured, and should not be reported as the project's
result going forward.

**Why this matters concretely:** centralized and federated numbers on
this project's datasets are NOT interchangeable and have previously
diverged sharply -- e.g. robo3er `stuck` AUROC for signal H was 0.946
federated vs. 0.869 centralized (`memory/scoring-signals-B-C-E-H.md`);
`memory/joint-prototype-federated-results.md` documents cases going the
other way too (voraus-AD collapses toward chance federated). Any new
signal (e.g. the forecast head / signal K work in
`memory/forecast-head-signal-k.md`) that was only validated centralized
has NOT been validated for this project's actual purpose until it's
re-run federated.

**How to apply:** when asked to "run the benchmark" / "test X on
[dataset]" without the user specifying centralized explicitly, default
to writing/running the `_federated.py` variant (or extending the
existing one) rather than `run_<dataset>_v*.py`. If a centralized script
doesn't exist yet for something new, build the federated version
directly rather than centralized-first -- do not treat centralized as a
stepping stone to be later "also" federated unless the user asks for a
centralized-only sanity check for a specific structural reason.

See `memory/joint-prototype-federated-results.md` for the established
federated protocol (per-client local encoder/edge_head/typed_head, only
`JointPrototypeMemory` codebook exchanged via `align_and_split`, optional
FedAvg'd encoder/decoder via `SYNC_ENCODER_DECODER`) and
`memory/forecast-head-signal-k.md` for signal K, whose centralized-only
results (as of 2026-08-17) are pending a federated re-run under this
policy.
