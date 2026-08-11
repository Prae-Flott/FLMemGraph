# voraus-AD dataset: downloaded and structure verified (2026-08-11)

## What was fetched

Per the user's request to pull https://github.com/vorausrobotik/voraus-ad-dataset
as a comparison dataset. Downloaded (not yet adapted/piped into a training
script -- this is a download-and-document-only step):

- `data/voraus_ad/voraus-ad-dataset-100hz.parquet` (~1.1GB) -- the 100Hz
  variant, matching what the reference repo's `train.py` uses by default.
  A 500Hz/~5.3GB variant also exists at the same host but was not fetched
  (no current need for 5x the resolution).
- `docs/voraus_ad_paper.pdf` -- the reference paper (Brockmann, Rudolph,
  Rosenhahn, Wandt, "The voraus-AD Dataset for Anomaly Detection in Robot
  Applications", IEEE T-RO 2023, arXiv:2311.04765).

Both direct downloads, no registration required. Dataset license: CC
BY-NC-SA 4.0 (non-commercial). Reference repo code: MIT.

`pyarrow` was installed (`pip install --break-system-packages pyarrow`,
matching this project's existing no-venv, system-wide-packages convention)
since pandas needs it to read `.parquet`.

## Verified structure

Loaded and inspected directly (not just read from the reference repo's
`voraus_ad.py`, which was also fetched/read for the column-name reference
-- see below):

- **2,321,690 rows, 2,122 unique samples** (one "sample" = one full
  pick-and-place cycle). Rows per sample: min 986, max 1164, mean 1094
  (at 100Hz that's roughly 10-11.6 seconds/cycle).
- **112 machine-data signals**: 4 robot-level electrical (robot voltage/
  current, IO current, system current) + 6 joints x 18 signals each
  (target/motor/joint position, velocity, acceleration where applicable,
  target torque, computed inertia, computed torque, motor torque, TWO
  independent torque sensors A/B, motor Iq/Id current, electrical/
  mechanical/load power, motor/supply/brake voltage).
- **13 category values** (0-12), matching `Category` enum in the
  reference repo's `voraus_ad.py` exactly:

  | id | category | # samples |
  |---|---|---|
  | 0 | AXIS_FRICTION | 144 |
  | 1 | AXIS_WEIGHT | 156 |
  | 2 | COLLISION_FOAM | 72 |
  | 3 | COLLISION_CABLE | 48 |
  | 4 | COLLISION_CARTON | 22 |
  | 5 | MISS_CAN | 11 |
  | 6 | LOSE_CAN | 74 |
  | 7 | CAN_WEIGHT | 80 |
  | 8 | ENTANGLED | 10 |
  | 9 | INVALID_POSITION | 12 |
  | 10 | MOTOR_COMMUTATION | 89 |
  | 11 | WOBBLING_STATION | 37 |
  | 12 | NORMAL_OPERATION | 1367 |

  755 samples are anomalous (categories 0-11), 1367 normal (category 12).
- **77 named "variant"/"setting" values** (0-76) -- finer-grained than
  category (e.g. category AXIS_WEIGHT splits into variants A1_115G/
  A1_231G/A1_500G by added weight, category COLLISION_* splits by which
  obstacle material). One variant, `PRE_A` (id 72, `Variant.PRE_A`), is a
  DEDICATED pure-normal training split -- **948 of the 2,122 samples**
  have `setting==72`.

## Official train/test split convention (from the reference repo)

`voraus_ad.py`'s `load_pandas_dataframes()`: **train = every sample with
`variant == PRE_A` (948 samples, all normal)**; **test = every sample with
`variant != PRE_A`** (the remaining ~1,174 samples -- a mix of normal
operation under the OTHER variants plus every anomalous sample). This is
the same "fit on normal, evaluate on a mix of held-out normal + every
fault type" convention this project already uses for robo3er/Paderborn/
Sielaff, so no new evaluation methodology needs inventing -- AUROC per
category against the same per-sample framework should drop in directly.

Per-sample zero-padding to the training split's max length is the
reference repo's convention for fixed-size batching; this project's own
per-window (Paderborn: fixed T=64 sliding window) or per-sample
(robo3er: fixed T=60) convention will need its own choice here given
variable-length samples (986-1164 rows) -- not yet decided.

## Why this dataset is a strong next target for JointPrototypeGDNv3

Unlike robo3er (2 wheels, no real kinematic chain beyond simple
differential drive) or Paderborn (single bearing, no multi-node
mechanical structure at all), voraus-AD's 6-DOF arm has a genuine,
well-documented **kinematic + electromechanical chain per joint**:
target position/velocity/acceleration -> motor position/velocity ->
joint position/velocity, target torque -> computed torque -> motor
torque -> two independent torque sensors (A/B, a built-in redundant
cross-check exactly like robo3er's odom-vs-IMU idea, but native to this
dataset rather than something to newly wire up), and motor
current(Iq/Id)/voltage -> electrical/mechanical power. This gives a much
richer, more physically motivated graph than anything tried on this
project so far, both WITHIN each joint (many candidate typed edges: the
target->motor->joint chain is naturally "proportional"/near-identity in
steady state, current->torque is the same Kt-style "proportional"
relation `robo3er_physics.md` already uses, torque-sensor-A-vs-B is a
same-quantity redundant cross-check) and ACROSS joints (kinematic
coupling along the arm -- not yet characterized here, would need the
robot's actual DH parameters or an empirical coupling check).

12 named, physically distinct fault categories (vs. Paderborn's 3-way
damage-origin taxonomy or robo3er's 4 fault types) also gives much finer
per-category comparison resolution for testing whether V3's
node-vs-edge-vs-typed-edge signal split localizes differently per fault
mechanism (a collision should show up very differently from a motor
commutation fault or an added can weight) -- directly extending the
"does the diagnosis signal differ by fault type" question this project
already investigated on Paderborn and robo3er -- see
`memory/joint-prototype-scheme-v3.md` and `memory/joint-prototype-scheme-b.md`
for the consolidated cross-dataset conclusions, including this dataset's
own results.

## What's NOT done yet

(Note: `benchmark/datasets/voraus_ad_adapter.py` and a full Scheme V3 run
now exist -- see `memory/joint-prototype-scheme-v3.md` -- the item below
describes this file's original download-only scope, kept for history.)
- Originally, no adapter existed -- this file covered a download +
  structure-verification step only, per the literal request ("拉取这篇
  数据集作为对比" -- pull this dataset for comparison), not yet wired
  into this project's adapter/registry-runnable convention the way
  Paderborn's `paderborn_adapter.py` is.
- No `benchmark/datasets/voraus_ad_physics.md` yet (the per-dataset
  physics-prior reference doc convention used for robo3er/Paderborn/
  Sielaff) -- would need the robot's actual joint count/DH parameters
  (not in the parquet itself) or a data-driven characterization of the
  inter-joint coupling before declaring cross-joint edges with any
  confidence; the within-joint edges (target->motor->joint chain,
  torque-sensor A/B cross-check, current->torque) can be declared from
  the column semantics alone without needing robot geometry.
- No decision yet on windowing convention for the variable-length
  (986-1164 row) samples -- fixed-length zero-padding (matching the
  reference repo) vs. this project's usual fixed sliding-window
  approach are both viable, not yet chosen.
- No baseline run (GDN/AE/JointPrototype) on this dataset yet.
