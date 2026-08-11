# v3 on robo3er: node/prototype signal dominates, typed-edge mechanism
# HURTS overall -- opposite pattern from Paderborn (2026-08-11)

## What was run

First run of `JointPrototypeGDNv3` (`src/joint_prototype_model.py`,
built for Paderborn) outside the bearing dataset --
`benchmark/run_robo3er_joint_prototype_v3.py`, reusing the model
unchanged. 7 nodes chosen from robo3er's kinematic chain
(`src/feature_groups.py`, `benchmark/datasets/robo3er_physics.md`):
`wheel_vels_velocity_left/right`, `odom_odo_lintw_x`, `odom_odo_angtw_z`,
`imu_imu_angvel_z`, `wheel_status_current_ma_left/right`. 7 declared
edges: the 4 differential-drive kinematics relations (wheel velocities ->
chassis lin/ang velocity) + 1 new one never exploited before (odom yaw
rate -> IMU yaw rate, an independent cross-check
`feature_groups.py`'s docstring flagged as a stronger slip signal than
wheel-vs-odom but never implemented) + 2 actuation edges (motor current
-> wheel velocity). First 5 edges typed "proportional" (textbook linear
kinematics / sensor cross-check), last 2 "nonlinear" (motor dynamics
under load). Pooled across all 5 robots, same fit/calib/test_normal/
fault-type split as every other robo3er baseline
(`benchmark/datasets/robo3er_adapter.py`).

## Result: opposite pattern from Paderborn

| method | mean AUROC (4 faults) | Broken Pipe | cable trapped | Low battery | stuck |
|---|---|---|---|---|---|
| A: prototype only | 0.893 | 0.764 | 0.997 | 0.989 | 0.822 |
| **B: prototype + node** | **0.942** | **0.981** | 1.000 | 0.997 | 0.789 |
| C: edge only (v2 generic attention) | 0.826 | 0.591 | 0.994 | 0.979 | 0.741 |
| D: full v2 (A+B+C) | 0.893 | 0.873 | 0.994 | 0.985 | 0.720 |
| E: typed-edge only (v3, new) | 0.791 | 0.548 | 0.999 | 0.985 | 0.630 |
| F: full v3 (all signals) | 0.861 | 0.795 | 0.999 | 0.991 | 0.659 |

On Paderborn, edge/attention-based signals were the strongest component
and typed-edge combined (F) beat v2 (D). On robo3er, it's the reverse:
**`B` (prototype + node, no edge signal at all) is the single best
combination (0.942), beating `F_full_v3` (0.861) by 0.081.** The new
typed-edge signal (`E`, 0.791) is the WEAKEST signal in the whole table,
and adding it to the combination makes `F` worse than `D` (0.861 <
0.893), which is itself no better than `A` alone.

The effect is starkest on the two "hard" fault types:
- **Broken Pipe**: node signal (`B`=0.981) vs. edge signals (`C`=0.591,
  `E`=0.548) -- a ~0.4 AUROC gap.
- **stuck**: node signal (`A`=0.822) again clearly beats every
  edge-inclusive combination (`D`=0.720, `F`=0.659) -- notable because
  `stuck` is this project's historically hardest fault type (previously
  best-ever result was AUROC 0.53->0.72-0.74 via `kinematics.py`'s
  residual + GDN forecasting, see `memory/physics-informed-gdn.md`);
  JointPrototype's edge mechanisms make it WORSE here, not better.

`cable trapped` and `Low battery` are near-ceiling (0.98-1.00) under
every signal combination and don't discriminate between methods.

## Why this is the opposite of Paderborn, not a contradiction

The two datasets' fault mechanisms are structurally different:
- Paderborn's bearing damage manifests as a **cross-channel relationship**
  being broken (vibration/force/torque/current losing their normal
  coupling) -- edge/attention signals are the natural fit, which is why
  they dominated there.
- robo3er's faults (`stuck`, `broken pipe`) look more like a **single
  node's value** going out of its normal operating range (a wheel simply
  not moving, a sensor reading pinned) -- the prototype-match failure
  (`d_G`) and per-node deviation (`s_node`) already capture most of that,
  and the extra edge-based max-aggregated dimensions (7 declared edges on
  a 7-node graph -- proportionally MORE edges relative to nodes than
  Paderborn's 8-edges-on-6-nodes) mostly just add false-positive-prone
  noise on top, via the same max-aggregation noise-floor mechanism
  flagged in `memory/paderborn-joint-prototype.md`'s root-cause analysis
  -- except here the effect is strong enough to be a net negative rather
  than a modest one.

This is consistent with, not contradictory to, v3's design: the model
correctly learned SOMETHING on the declared edges (codebook utilization
100%, `prior_bias_strength` grew from 1.0 to 1.086 at peak), it's just
that on THIS dataset's fault types, that structural signal is a weaker
diagnostic than the simpler node-level one, and combining them via hard
max lets the weaker one drag down the stronger one.

## Implication for the "make v3 more mature" plan

This result sharpens item 4 (aggregation strategy) from the maturation
plan discussed after the Paderborn v3 write-up: a fixed
`max(dG, node, edge, typed)` combination cannot be dataset-optimal for
both datasets at once -- Paderborn wants edge signals weighted up,
robo3er wants them weighted down or excluded. A calibration-split-learned
weighted combination (or even just per-dataset ablation selection, e.g.
using `B` instead of `F` when edges don't help) would fix this
concretely instead of always taking the max over every available signal.
It also means the "9/16 prototypes had enough calib samples" and "typed
temperature stuck at 1.0000 (never moved from init)" details are worth
double-checking -- an anomaly-attention temperature that never leaves its
initial value across BOTH datasets now is a specific, checkable sign
that the unsupervised Sec 16 attention isn't learning anything
data-dependent, not just an under-exercised mechanism on sparse data.

## What's NOT done

- Same scoping-down list as `memory/paderborn-joint-prototype-v3-typed-attention.md`
  applies here (2 relation types only, unsupervised anomaly attention,
  no self-supervised augmentation training, no 3-stage curriculum).
- No federated/per-robot run -- this is the pooled-centralized
  comparison point only, matching this project's other robo3er
  baselines' convention.
- No controlled multi-seed ablation here either (same caveat as
  Paderborn's v3 write-up about not yet being able to cleanly separate
  "retrain variance" from "mechanism contribution").

Full report: `checkpoints/robo3er_joint_prototype_v3_report.json`, model
weights: `checkpoints/robo3er_joint_prototype_v3.pth`.
