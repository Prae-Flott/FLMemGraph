# robo3er: physical prior relationships & formulas

Full implementation: `src/robo3er/kinematics.py`. This is the reference
implementation every other dataset's physics-prior work in this project
is modeled after ("expert-knowledge-informed structure, data-fit
parameters").

## Platform

robo3er is a 5-robot fleet of iRobot Create3-class **differential-drive**
mobile robots. Each robot has two independently-driven wheels; steering
is achieved by commanding different left/right wheel velocities, not a
separate steering mechanism.

## The physical relation

Textbook forward kinematics for a differential-drive robot relates the
two wheel velocities `(v_l, v_r)` to the chassis's linear and angular
velocity:

```
v_lin = k_v * (v_l + v_r)     where k_v = r/2
v_ang = k_w * (v_r - v_l)     where k_w = r/L
```

`r` = effective wheel radius, `L` = effective track width (distance
between the two wheels' contact points) -- both physical constants of
the specific robot.

## Why the constants are FIT, not hard-coded

`r` and `L` are physically real, but their exact values in the units the
logged topics use are unknown (could be wheel surface speed in m/s, raw
encoder rate, affected by calibration drift or gearbox slack). Rather
than assume a spec-sheet number, `k_v` and `k_w` are fit by ordinary
least squares from each robot's own held-out NORMAL (fit-split) data --
keeping the physically-motivated two-parameter structure of the equation,
not a free 4-coefficient linear regression. An intercept term (`b_v`,
`b_w`) is fit alongside each slope to absorb constant sensor
offset/calibration bias -- physically that term shouldn't exist for an
ideal robot at rest, so a fitted intercept far from 0 is itself a
diagnostic flag worth reporting, not silently discarded.

In the federated system (`src/robo3er/fl_dataset.py`), this fit is done
**per client** (per robot) -- each robot may have slightly different
wheel radius/track width (e.g. from wear), so a per-robot fitted constant
captures that rather than forcing one global fit across robots with
different physical calibration.

## The residual (what actually gets fed to the model)

```
residual_lin = measured(odom_odo_lintw_x) - predicted(v_l, v_r; k_v, b_v)
residual_ang = measured(odom_odo_angtw_z) - predicted(v_l, v_r; k_w, b_w)
```

These two residuals REPLACE the raw `odom_odo_lintw_x` /
`odom_odo_angtw_z` columns before GDN (or any downstream model) sees the
data -- every other feature in the active feature group is untouched.
Whatever variance the kinematic equation predicts is treated as
"explained by rigid-body kinematics" and stripped out; what's left is
fed forward.

## Why this residual is diagnostic

A wheel that is commanded/measured to spin but whose motion does NOT show
up in the chassis's actual odometry twist -- i.e. a large kinematic
residual -- is the textbook definition of wheel slip. This residual
carries direct signal for robo3er's `cable trapped` (slip) fault type,
on top of whatever raw-feature signal GDN could already find on its own.

## Measured impact (see `memory/physics-informed-gdn.md` for the full
## writeup)

This residual, combined with GDN's own forecasting mechanism
(`src/training/train_gdn_physics.py`), produced this project's best-ever `stuck`
fault result (AUROC 0.53 -> 0.72-0.74) -- the single largest improvement
found anywhere in this project's history, via a mechanism orthogonal to
every other approach tried on that fault type (memory, feature selection,
architecture changes).

## What's NOT modeled

- Only the linear/angular VELOCITY relation is used. The chain continues
  further (velocity -> displacement via integration, and there's a
  second independent estimator of chassis yaw rate via the IMU,
  `imu_imu_angvel_z`, that `src/robo3er/feature_groups.py`'s docstring flags as
  "arguably a MORE rigorous slip detector than wheel-vs-odom, since both
  sides of THAT comparison are wheel-derived" -- not yet implemented).
- Wheel radius/track width are assumed constant per robot per run (no
  time-varying wear model).
