# v3 on voraus-AD: first pass, weak and unvalidated -- no baseline to
# compare against yet (2026-08-11)

## What was run

First test of `JointPrototypeGDNv3` on voraus-AD
(`benchmark/run_voraus_ad_joint_prototype_v3.py`,
`benchmark/datasets/voraus_ad_adapter.py`), the third dataset after
Paderborn and robo3er. 18 nodes (3 signals x 6 joints: `motor_iq_i`,
`motor_torque_i`, `torque_sensor_a_i`), 12 declared edges replicated
identically within each joint (`motor_iq_i -> motor_torque_i`
"proportional", `motor_torque_i -> torque_sensor_a_i` "nonlinear") --
NO cross-joint edges declared, since the arm's actual kinematic coupling
isn't characterized yet (see `memory/voraus-ad-dataset.md`). Official
train/test convention: FIT/CALIB from the 948 `variant==PRE_A`
pure-normal samples (758/190 split), TEST_NORMAL = 419 held-out
`NORMAL_OPERATION` samples from other variants, 12 named fault
categories scored separately. Each sample is a full pick-and-place cycle,
zero-padded to the dataset-wide max length (1164 timesteps @ 100Hz,
actual lengths range 986-1164).

## Result: weak across the board, no ceiling categories except one

| method | mean AUROC (12 categories) |
|---|---|
| A: prototype only | 0.660 |
| B: prototype + node | 0.679 |
| C: edge only (v2 generic attention) | 0.679 |
| D: full v2 (A+B+C) | 0.680 |
| E: typed-edge only (v3, new) | 0.666 |
| F: full v3 (all signals) | 0.666 |

Per-category (`D_full_v2` / `F_full_v3`):

| category | n | D (v2) | F (v3) |
|---|---|---|---|
| miss_can | 11 | 0.983 | 0.985 |
| wobbling_station | 37 | 0.819 | 0.794 |
| invalid_position | 12 | 0.772 | 0.775 |
| motor_commutation | 89 | 0.764 | 0.704 |
| collision_carton | 22 | 0.707 | 0.724 |
| axis_weight | 156 | 0.655 | 0.634 |
| collision_foam | 72 | 0.654 | 0.640 |
| collision_cable | 48 | 0.638 | 0.634 |
| lose_can | 74 | 0.586 | 0.594 |
| axis_friction | 144 | 0.566 | 0.540 |
| can_weight | 80 | 0.536 | 0.515 |
| entangled | 10 | 0.482 | 0.451 |

Compared to Paderborn (mean 0.87-0.90) and robo3er (mean 0.86-0.94), this
is a MUCH weaker result -- only `miss_can` is near ceiling, most
categories sit in the 0.55-0.72 range (barely-informative to modest), and
`entangled` (n=10) is at or below 0.5 (essentially chance/inverted) for
every signal combination.

**This is NOT a validated negative result the way the robo3er comparison
was** -- unlike Paderborn/robo3er, THERE IS NO FAIR BASELINE (plain GDN/
AE on the same 18 nodes) run on this dataset yet to compare against. It's
entirely possible 0.66-0.68 mean AUROC is actually competitive or even
good for this dataset/node-set/split, or that it's genuinely weak
relative to a simpler baseline -- unknown until that comparison exists.
Treat these numbers as a first-pass sanity check only, not a claim about
v3's typed-edge mechanism's value on this dataset.

## UPDATE: the reference paper directly explains most of this (2026-08-11)

After reading `docs/voraus_ad_paper.pdf` in full and writing
`benchmark/datasets/voraus_ad_physics.md`, hypothesis #1 below is no
longer the leading explanation -- the paper's OWN ablation (Fig. 13a)
found mean AUROC by signal subset: **electrical alone ~65%, mechanical
alone ~92%, all signals ~93%**. This run's 18-node graph
(`motor_iq`/`motor_torque`/`torque_sensor_a` x 6 joints) is built almost
entirely from what the paper calls "electrical" (current) plus one
narrow mechanical signal -- it has ZERO position/velocity nodes, the
exact category the paper says matters most. **This run's overall mean
(0.66-0.68) landing almost exactly at the paper's own "electrical alone"
figure (~65%) is very unlikely to be a coincidence.**

The paper also gives a precise, dataset-verified mechanism for why
`axis_friction` scored worst among the larger-sample categories
(0.540-0.566): friction is explicitly defined as *"a higher torque of
the motor is needed for the same movement"* -- i.e. it breaks the
target/velocity-vs-torque relation, NOT the current-vs-torque relation
(#2 in `voraus_ad_physics.md`) this graph actually has. The one relation
friction breaks isn't declared as an edge anywhere in this run.

**Revised next step**: adding `joint_velocity_i`/`joint_position_i` (or
the full target->motor->joint tracking chain from `voraus_ad_physics.md`
#1) as nodes is now the best-evidenced single change to try before
anything else on this list -- ahead of cross-joint edges, hyperparameter
tuning, or the padding/windowing choice.

## Plausible reasons this run is weak (untested hypotheses, not diagnosed)

1. **No cross-joint edges.** Several fault categories (collision_*,
   wobbling_station, invalid_position) plausibly perturb the WHOLE arm's
   coordinated motion, not a single joint's local current/torque
   relationship -- exactly the kind of signal only a cross-joint edge
   could catch, and none are declared here.
2. **Zero-padding dilution.** Real sample lengths range 986-1164 (~15%
   variation); every sample is padded to the dataset-wide max (1164)
   with post-scaling zeros before being fed through ONE
   `nn.Linear(1164, embed_dim)` per node. Padded zeros are learnable
   signal to the encoder the same as real data, and the fraction of
   padding differs per sample -- this is a meaningfully different, more
   dilutive regime than Paderborn/robo3er's fixed, fully-real windows.
3. **Untuned hyperparameters.** `EPOCHS=12`, `EMBED_DIM=64`,
   `NUM_PROTOTYPES=16` etc. were carried over unchanged from Paderborn/
   robo3er's much smaller graphs (6-7 nodes, 60-64 timesteps) to this
   dataset's 18 nodes x 1164 timesteps -- no reason to expect the same
   settings are well-matched to a ~3x larger node count and ~19x longer
   window.
4. **Small-sample AUROC noise.** `entangled` (n=10), `miss_can` (n=11),
   `invalid_position` (n=12) have too few test samples for their AUROC
   to be a stable estimate -- `entangled`'s near/below-0.5 score in
   particular shouldn't be read as "the model gets this backwards,"
   just as "this estimate is unreliable."
5. **Only 2/16 prototypes** had >= 5 calib windows (calib split is only
   190 samples spread across 16 slots) -- the prototype-conditioned
   standardization is even more under-exercised here than on Paderborn
   (5/16) or robo3er (9/16).

## What's NOT done

- **No fair baseline (plain GDN, plain AE) on the same 18-node graph** --
  the single most important missing piece before any AUROC number here
  can be interpreted as good/bad/better-than-v2. This should be the very
  next step, mirroring `run_paderborn_6ch_comparison.py`'s convention.
- No cross-joint edges -- would need either the robot's actual DH
  parameters (not in the parquet) or an empirical coupling check.
- No hyperparameter tuning for this dataset's much larger scale.
- No windowing alternative to zero-padding tried (e.g. truncating to the
  shortest sample's length, or padding with the last real value instead
  of zero).
- The dual redundant torque sensor (`torque_sensor_b_i`) and the
  target/joint position-velocity chain per joint are documented in
  `memory/voraus-ad-dataset.md` but not used as nodes/edges here -- this
  run intentionally used the smallest node set that still tests the
  proportional/nonlinear typed-edge distinction, not the richest
  possible graph.

Full report: `checkpoints/voraus_ad/voraus_ad_joint_prototype_v3_report.json`,
model weights: `checkpoints/voraus_ad/voraus_ad_joint_prototype_v3.pth`.
