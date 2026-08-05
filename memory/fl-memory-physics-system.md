# Federated memory+structure fault detection system

This is FLMemGraph's own copy of this history (originally written while
this project still lived inside `~/Projects/FL-bench`, see that repo's
`memory/fl-memory-physics-system.md` for the move note). Data
(`data/robo3er/`) and the GDN baseline (`test_gdn/`) were copied, not
symlinked, from FL-bench -- this project has zero runtime dependency on
FL-bench being present.

Built from a detailed design doc (`mem_phys_prompt_zh.md`, pasted into
conversation 2026-08-05) proposing a 3-layer architecture: one shared
per-node encoder feeding a discrete-prototype memory head (novelty vs.
"known normal") and a latent-space graph-relation structure head
(physical/relational consistency) -- the two-signal combination resolves
what neither signal alone can: "doesn't match anything memorized" is
ambiguous between "genuinely new but still valid work mode" and "real
fault," and only the relation-consistency signal (holds vs. broken) tells
them apart.

## Files

- `fl_model.py` -- `SharedEncoder` (Linear window->D, channel-independent),
  `StructureHead` (TopK graph attention over OTHER nodes' latents, no
  self-loop -- see its docstring for why self-loop would make the
  structure residual a trivial always-zero shortcut, unlike GDN's own
  forecasting variant where self-loop is fine because of the temporal
  offset), `FLGDNMemory` (wires encoder + memory head [reused from
  `gdn_memory_model.DiscretePrototypicalMemory`] + structure head +
  training-only decode head).
- `federated_memory.py` -- server-side cross-client codebook alignment:
  cosine-similarity graph (cross-client pairs only) + BFS connected
  components as candidate shared clusters, top-K by cluster size ->
  mean-pooled shared prototypes P_S, remaining prototypes ranked by
  utility-diversity (Freq - max cross-client similarity) -> personalized
  P_p,n, broadcast P_G,n = [P_S; P_p,n] per client.
- `decision_logic.py` -- stateful `StreamMonitor`: per-window (d, r) ->
  KNOWN_NORMAL / HARD_ALARM / SOFT_ALERT / PENDING_LABEL (persistence-
  based escalation), `grow_codebook` for confirmed-normal pending items
  (LRU-by-usage-count eviction into the fixed-capacity codebook).
- `fl_dataset.py` -- multi-source clients = robo3er's REAL 5 robots, via
  `data/robo3er/partition.pkl`'s existing `data_indices` (same partition
  IFCAAE etc. already use). Verified severe non-IID: robot04 alone is 79%
  of all data and the only stuck source; robot02 has zero fault windows;
  Broken Pipe splits across robot00+robot03; Low battery only robot00;
  cable trapped only robot01. Per-client kinematic residual (each robot
  fits its own k_v/k_w).
- `train_fl_memory_gdn.py` -- driver: R rounds x local-epochs per client
  (whole-window reconstruction, no dense sliding needed since there's no
  forecasting target), server alignment between rounds, final per-client
  evaluation with d-only/r-only/d+r AUROC (GDN-style score, calibrated
  from each client's OWN calib split).

## Two real bugs found and fixed during first run -- both instructive

1. **Independent random encoder init across clients makes cross-client
   cosine similarity meaningless.** First run: `align_and_split` found
   ZERO shared clusters every round (max cross-client cosine sim was
   0.50, delta was 0.8). Diagnosed by directly checking the cosine-sim
   distribution between two clients' trained codebooks. Fix: broadcast
   ONE shared random init for encoder+structure-head weights to all
   clients before round 1 (memory codebook still starts independent --
   that's what's meant to diverge and get aligned). This doesn't violate
   "encoder stays local" (each client still trains fully independently
   after round 0), it just gives the embedding geometry a common
   reference frame so cosine similarity between different clients'
   prototypes is comparable at all. Also lowered delta 0.8->0.5 for this
   much smaller (5 clients, not 28+) more severely non-IID setting.
   After the fix: 57-67 shared clusters emerge and GROW over rounds
   (57->67 across 5 rounds) -- genuine convergence, not noise.
2. **Small per-client calib sets (22-692 windows) blow up the IQR-
   normalized score by orders of magnitude** (observed: scores in the
   millions) on whichever dimension happens to have a near-zero IQR in
   that tiny sample. Same root cause as the GDN z-score blowup documented
   in `robo3er-anomaly-detection-approaches.md`, but far more severe here
   because 22-49 calib windows is much less stable than the pooled 833.
   Fix: `dataset.gdn_score(..., iqr_floor_frac=0.1)` (previously
   documented as an OPTIONAL trade-off knob, off by default for
   detection-AUROC reasons) -- here it's not optional, the unfloored
   version was numerically broken, not just "slightly worse."

## Results (5 rounds x 20 local epochs, kinematic_core 38-feature set)

| client (robot) | fault | AUROC(d) | AUROC(r) | AUROC(d+r) |
|---|---|---|---|---|
| robot01 | cable trapped | 0.936 | 0.922 | 0.922 |
| robot04 | stuck | 0.568 | **0.712** | **0.717** |
| robot03 | Broken Pipe | 0.461 | 0.506 | 0.520 |
| robot00 | Broken Pipe | 0.000 | 0.000 | 0.000 |
| robot00 | Low battery | 0.401 | 0.341 | 0.344 |

cable trapped and stuck are strong and land close to this project's best
centralized numbers (stuck's structure-residual AUROC 0.712-0.717 nearly
matches `test_gdn_physi/train_gdn_physics.py`'s kinematic-residual result
on POOLED data, 0.717-0.742 -- genuinely encouraging, the federated
per-client version isn't losing much by staying local).

robot00's results are weak/inverted specifically because it's the
smallest client (195 fit / 41 calib windows) -- verified this is NOT a
bug: raw memory-distance values for Broken Pipe fault windows are
uniformly ~10-100x SMALLER than calib-normal windows across nearly every
node (checked directly, not inferred from AUROC alone). Plausible
explanation: Broken Pipe means the robot is stopped, and a
near-constant/near-zero kinematic state may be easier for a
barely-trained (195-window) encoder+memory to represent well than the
more variable normal driving state -- but this needs more data to confirm
rather than assert. **Known limitation, not yet resolved**: tiny-client
calibration instability. A next step worth trying: borrow/shrink a small
client's calib median/IQR toward the federated pool's aggregate stats
(the memory codebook is already shared; the calibration reference isn't).

## benchmark/ folder (scaffold, per mem_phys_prompt_zh.md Sec 8)

`benchmark/datasets/registry.py` and `benchmark/baselines/registry.py`
transcribe the FULL Sec 8.1/8.2 dataset and method taxonomy (A/B/C/D
dataset categories, ①②③④ baseline categories) with per-entry status.
**Only robo3er + PCA/IsolationForest/GDN(lightweight) are actually
implemented and runnable** (`python3 benchmark/run_benchmark.py`) -- the
other ~15 datasets (SWaT/WADI need iTrust registration, MVTec-AD family
is multi-GB, SMD/SMAP/MSL/PSM/GECCO/the graph and ETT datasets are freely
downloadable but not fetched this session) and ~15 baseline methods are
registered/documented but not implemented. See `benchmark/README.md` for
the explicit scope decision and how to extend each registry entry from
"planned" to "implemented."
