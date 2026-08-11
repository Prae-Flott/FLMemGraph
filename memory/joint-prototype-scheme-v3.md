---
name: joint-prototype-scheme-v3
description: Final conclusions for Joint Prototype Memory "Scheme V3" (typed relation-specific edges + prototype-conditioned standardization + anomaly attention) across all datasets tested
---

# Scheme V3: typed-relation Joint Prototype GDN -- final cross-dataset
# conclusions (consolidated 2026-08-11)

## What it is

`src/joint_prototype_model.JointPrototypeGDNv3` = Scheme B's encoder +
`JointPrototypeMemory` (`memory/joint-prototype-scheme-b.md`), PLUS:
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
before this adds anything Scheme B doesn't already provide. See
`memory/joint-prototype-scheme-b.md` for when to prefer the simpler,
prior-free alternative instead.

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
2. **v1** (fixed edge list, one linear map per declared edge): did NOT
   beat a fair same-footing plain-GDN baseline on Paderborn (0.802 vs.
   0.819) -- the fixed-edge skeleton and single linear map per edge had
   less capacity than GDN's own learned attention, and covered only the
   declared pairs.
3. **v2** (learned attention over all pairs, physics edges as an
   attention-logit bias instead of a hard restriction): the first clear,
   substantial win on Paderborn (0.873 vs. the 0.819 baseline) -- kept
   deviation-space edges from v1, replaced the capacity-limited fixed
   edge list with GDN-style attention. This became `TrendGraphAttentionHead`,
   still used unchanged inside v3 today.
4. **v3** (this version): adds typed relation-specific message functions
   + prototype-conditioned standardization + anomaly attention on top of
   v2's attention head, per `docs/joint_prototype_physics_gdn_anomaly_attention_prompt.md`.

## Cross-dataset results (final numbers, mean AUROC per dataset)

| dataset | A | B | C | D | E | F | winner |
|---|---|---|---|---|---|---|---|
| **Paderborn** (6 nodes, 26 bearings) | 0.750 | ~0.79 | **0.904** | 0.889 | 0.816 | 0.898 | C (edge attention) |
| **robo3er** (7 nodes, 4 fault types) | 0.893 | **0.942** | 0.826 | 0.893 | 0.791 | 0.861 | B (no edges) |
| **voraus-AD** (66 nodes, 12 fault categories) | 0.642 | **0.756** | 0.729 | 0.730 | 0.647 | 0.687 | B (no edges) |
| fair baselines (GDN / AE, Paderborn) | 0.819 / 0.679 | | | | | | V3 beats both |

**Only Paderborn's fault mechanism is dominated by the declared edges.**
Both other datasets' best path is B (Scheme B, no edges at all) -- the
typed-edge mechanism (E) never wins outright anywhere, and the "kitchen
sink" combination (F) never beats the single best individual signal on
any dataset. This is the central, repeatedly-confirmed finding: **which
signal wins is a property of the dataset's fault mechanisms, not of
model capability** -- Paderborn's bearing damage genuinely breaks
cross-channel physical coupling (vibration/force/torque/current); most
of robo3er's and voraus-AD's faults look like a single value drifting,
which Scheme B already captures without needing any declared relation.

The ONE consistent exception within otherwise B-dominated datasets:
voraus-AD's `motor_commutation` fault (current stops predicting torque
proportionally, a textbook relation break) -- C/D beat B specifically on
this one category (0.841/0.840 vs. 0.818), nowhere else in that
dataset's 12 categories.

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

- `src/models/joint_prototype_model.py`: `JointPrototypeGDNv3` and its
  components (current, only version in the codebase).
- Paderborn: `benchmark/run_paderborn_joint_prototype_v3.py`,
  `checkpoints/paderborn/paderborn_joint_prototype_v3_report.json`.
- robo3er: `benchmark/run_robo3er_joint_prototype_v3.py`,
  `checkpoints/robo3er/robo3er_joint_prototype_v3_report.json`.
- voraus-AD: `benchmark/run_voraus_ad_joint_prototype_v3.py`,
  `benchmark/datasets/voraus_ad_adapter.py`,
  `checkpoints/voraus_ad/voraus_ad_joint_prototype_v3_report.json`.
- Physics references per dataset: `benchmark/datasets/paderborn_physics.md`,
  `benchmark/datasets/robo3er_physics.md`, `benchmark/datasets/voraus_ad_physics.md`
  (Sielaff has no verified physics prior -- `benchmark/datasets/sielaff_physics.md`
  documents that explicitly; only Scheme B was run there, see
  `memory/joint-prototype-scheme-b.md`).

## What's NOT done

- No self-supervised relation-breaking augmentation training for the
  anomaly attention (design doc Sec 17) -- still the unsupervised
  softmax-over-magnitude version.
- No cross-joint/cross-node edges on voraus-AD (needs the arm's DH
  parameters or an empirical coupling check).
- No adaptive per-sample weighting between B and edge-based signals
  (discussed but not implemented) -- combination is still a fixed
  max()/top-k-mean, not learned per sample or per operating regime.
- No controlled multi-seed ablation anywhere to separate "retrain
  variance" from "mechanism contribution" precisely.
