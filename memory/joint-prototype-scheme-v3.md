---
name: joint-prototype-scheme-v3
description: Final conclusions for Joint Prototype Memory "Scheme V3" (typed relation-specific edges + prototype-conditioned standardization + anomaly attention) across all datasets tested
---

# Scheme V3: typed-relation Joint Prototype GDN -- final cross-dataset
# conclusions (consolidated 2026-08-11)

## Naming note

This document's "V3" is the current, final version. "V2" (formerly "Scheme
B", now folded into `scoring-signals-B-C-E-H.md`'s signal B) is the
no-edges-at-all baseline: `SharedEncoder` + `JointPrototypeMemory` only, no
edge/relation head. Do not confuse V2 with the historical
`JointPrototypeGDNv2` class discussed in the Lineage section below (which
had generic edge attention but no typed relations, and was deleted in an
earlier cleanup pass) -- that class predates and is unrelated to the
current "V2" naming; it's referred to below as "the interim
attention-only design" to avoid the collision.

## What it is

`src/models/joint_prototype_model.JointPrototypeGDNv3` = V2's encoder +
`JointPrototypeMemory` (no edges, signal B), PLUS:
- `TrendGraphAttentionHead`: GDN-style learned attention over each node's
  TopK-similar neighbors: declared physics edges bias attention logits
  (one learned scalar) but do NOT restrict which relationships can be
  learned. Ablation signal **C** (edge-only).
- `TypedRelationAnomalyHead`: relation-specific message functions
  (linear for "proportional" edges, small MLP for "nonlinear" edges) +
  prototype-conditioned edge-residual standardization + an unsupervised
  anomaly attention, covering ONLY the declared physics edges. Ablation
  signal **E** (typed-edge-only).
- **D** = max(A, B, C); **F** = max(A, B, C, E) -- the "everything"
  combination.

Needs domain knowledge: a verified physical relation between two signals
(what kind -- proportional or nonlinear) must be declared as an edge
before this adds anything V2/signal-B doesn't already provide. See
`scoring-signals-B-C-E-H.md`'s combination-rules table for when to prefer
the simpler, prior-free alternative instead.

## Lineage (condensed -- intermediate code removed, conclusions kept)

Three iterations preceded this final version, all now removed from the
repo (code + checkpoints), their lessons folded in here:
1. **Physics-residual precursor** (Paderborn): fit an OLS regression
   predicting motor current or torque from raw operating-condition
   values (`predicted = a*torque + b*speed + c`), residual fed to plain
   GDN. THREE consecutive negative results (current residual, torque
   residual free-fit, torque residual fixed-mu) -- root cause: Paderborn's
   4 discrete operating conditions don't vary force/speed/torque
   independently, so a regression across pooled raw magnitudes can't
   separate physical effects from which-condition-this-is. This is WHY
   Joint Prototype Memory operates on DEVIATIONS from a matched
   prototype (`d_i = z_i - p_i*`) rather than raw magnitudes -- deviation
   space sidesteps the confound, magnitude space doesn't.
2. **Fixed edge list** (one linear map per declared edge, an early
   `JointPrototypeGDN` design): did NOT beat a fair same-footing
   plain-GDN baseline on Paderborn (0.802 vs. 0.819) -- the fixed-edge
   skeleton and single linear map per edge had less capacity than GDN's
   own learned attention, and covered only the declared pairs.
3. **The interim attention-only design** (historically called
   `JointPrototypeGDNv2` -- NOT the same as this project's current "V2"
   naming, see the note above): learned attention over all pairs,
   physics edges as an attention-logit bias instead of a hard
   restriction -- the first clear, substantial win on Paderborn (0.873
   vs. the 0.819 baseline). Kept deviation-space edges from the fixed-edge
   design, replaced the capacity-limited fixed edge list with GDN-style
   attention. This became `TrendGraphAttentionHead`, still used unchanged
   inside V3 today.
4. **V3** (this version): adds typed relation-specific message functions
   + prototype-conditioned standardization + anomaly attention on top of
   the interim design's attention head, per
   `docs/joint_prototype_physics_gdn_anomaly_attention_prompt.md`.

## Cross-dataset results: superseded, see scoring-signals-B-C-E-H.md

The numbers originally here (a first, centralized, non-encdec-synced V3
run, plus voraus-AD -- a dataset since deleted from the repo) are stale
and partly conflict with the current federated+sync numbers on the same
signal letters. **[[scoring-signals-B-C-E-H]] is the authoritative,
up-to-date source for per-dataset signal performance and recommended
combinations** -- read that file for current numbers, not this one.

The one qualitative finding that still holds and is worth keeping here:
**which signal wins is a property of the dataset's fault mechanisms, not
of model capability.** Paderborn's bearing damage genuinely breaks
cross-channel physical coupling (vibration/force/torque/current), so its
edge-based signals (C/E/J) win; robo3er's and Sielaff's faults are
better explained by node-amplitude or covariance-pattern signals (B/H)
or generic structural attention (C) -- see scoring-signals-B-C-E-H.md's
per-dataset recommendation table for the current picture.

## Key mechanism lesson: max-aggregation noise floor, and its fix

Expanding voraus-AD's node/edge count (18->66 nodes, 12->54 edges, adding
a physically well-motivated friction edge) made results WORSE at first
(`axis_friction` regressed 0.566->0.489) under the original raw
`.max(axis=1)` aggregation -- more aggregated dimensions systematically
inflates the false-positive tail of the normal-data score distribution,
independent of whether the new dimensions are individually informative.
Fix: replace raw max with a top-k-mean, THEN a second calibration stage
that re-normalizes that aggregate statistic against its own distribution
on the calib split (not just each dimension independently) -- this
recovered and then exceeded the original prediction (`axis_friction`
D: 0.566 -> 0.771 -> 0.789 with further training). Any future node/edge
expansion on any dataset should use this two-stage aggregation from the
start, not raw max.

## Where results/code live

- `src/models/joint_prototype_model.py`: `JointPrototypeV31` (current,
  only model class in the codebase; renamed from `JointPrototypeGDNv3`).
- Paderborn: `benchmark/run_paderborn_v3_1.py` (centralized),
  `benchmark/run_paderborn_v3_1_federated.py` (federated+sync),
  `checkpoints/paderborn/paderborn_v3_1*_report.json`.
- robo3er: `benchmark/run_robo3er_v3_1.py` (centralized),
  `benchmark/run_robo3er_v3_1_federated.py` (federated+sync),
  `checkpoints/robo3er/robo3er_v3_1*_report.json`.
- Sielaff: no declared edges (`prior_edges=[]`, equivalent to V2.1) --
  `benchmark/run_sielaff_v2_1.py`, `benchmark/run_sielaff_v2_1_federated.py`.
- voraus-AD: dataset and all related scripts/checkpoints deleted from the
  repo (2026-08-16) -- no longer runnable, kept only as a historical
  data point in the qualitative finding above.
- Physics references per dataset: `benchmark/datasets/paderborn_physics.md`,
  `benchmark/datasets/robo3er_physics.md` (Sielaff has no verified physics
  prior -- `benchmark/datasets/sielaff_physics.md` documents that explicitly).

## TypedRelationAnomalyHead aggregation: softmax → max (2026-08-16)

Changed `TypedRelationAnomalyHead.forward()` aggregation from
`softmax(temperature · r_tilde) · r_tilde` to `max(r_tilde.clamp(min=0))`
(one-line change, `src/models/joint_prototype_model.py:480-483`).

Results after retraining V3.1 on both datasets:

| dataset | E_phys_max (softmax) | E_phys_max (max) | Δ | J_v3_cov_max (both) |
|---|---|---|---|---|
| Paderborn | 0.816 | **0.821** | +0.005 | 0.894 |
| robo3er | 0.759 | 0.760 | +0.001 | 0.725 |

Paderborn: all 4 target nodes have exactly 2 incoming edges
(vibration_1 ← force+speed; torque ← force+speed; current_1/2 ← speed+torque),
so softmax vs max is structurally meaningful -- but the improvement is tiny (+0.005).
Weak individual bearings KA07/KA08 gained +0.013/+0.012 on E_phys_max.

Robo3er: `stuck`-critical edges (`current → wheel_vel`) have 1 incoming edge each
so softmax=max identically. The `odom` target nodes do have 2 incoming edges, but
those aren't the dominant signal for `stuck`. No change on any metric.

**`C_struct_max` and `J_v3_cov_max` are unaffected on both datasets** -- the
aggregation change only affects `resid_phys` (E signal).

## What's NOT done

- No self-supervised relation-breaking augmentation training for the
  anomaly attention (design doc Sec 17).
- No cross-joint/cross-node edges on voraus-AD (needs the arm's DH
  parameters or an empirical coupling check).
- No adaptive per-sample weighting between B and edge-based signals
  (discussed but not implemented) -- combination is still a fixed
  max()/top-k-mean, not learned per sample or per operating regime.
- No controlled multi-seed ablation anywhere to separate "retrain
  variance" from "mechanism contribution" precisely.
