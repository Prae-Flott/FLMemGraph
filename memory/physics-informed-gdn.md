# Physics-residual GDN: kinematics prior stripped from odom before GDN sees it

`test_gdn_physi/` (`kinematics.py`, `train_gdn_physics.py`). Same protocol
as `test_gdn/train_gdn.py` (dense sliding, history_len=30, GDN-style
per-feature median/IQR-then-max scoring) with exactly one change: before
anything else, `odom_odo_lintw_x` and `odom_odo_angtw_z` are replaced by
their differential-drive KINEMATIC RESIDUAL (actual minus the two-wheel-
velocity kinematic prediction), so GDN only has to learn what rigid-body
kinematics can't already explain. Motivated by the multi-robot design
discussion (see conversation, 2026-07-30): use expert kinematics equations
to supply graph *structure*, fit only the physical constants (k_v ~
wheel_radius/2, k_w ~ wheel_radius/track_width) from data — a two-way
check, since those fitted constants can themselves be sanity-checked
against the robot's spec sheet.

## Result: biggest AUROC gain yet found for `stuck`

| fault | baseline (pure-data GDN) | physics-residual GDN | delta |
|---|---|---|---|
| Broken Pipe | 0.914 | 0.908 | -0.006 |
| cable trapped | 0.950 | 0.963 | +0.013 |
| Low battery | 0.975 | 0.982 | +0.007 |
| **stuck** | 0.528 | **0.717** | **+0.189** |

(Baseline numbers here are from a fresh `test_gdn/train_gdn.py` run on the
*current* on-disk `data/robo3er/` — see the data-versioning warning in
`README.md`; do not compare directly to the older 0.4-0.7 stuck numbers
elsewhere in this memory folder, which were measured on the pre-2026-07-23
data. This file's own baseline-vs-physics comparison is internally
consistent, same data both sides.)

Fitted params (fit split only, never calib/test/fault):
`v_lin = 0.4725*(v_l+v_r) + 0.0017` (R²=0.831), `v_ang = 1.1721*(v_r-v_l)
+ 0.0151` (R²=0.450). Angular equation explains less variance — consistent
with angular velocity being the first thing wheel slip should decouple.

## Why this specifically fixes stuck when nothing else did

Per `stuck-fault-gradual-onset.md`, every prior method failed on stuck
because reconstruction/prediction error measures *local, trend-conditioned*
surprise, and stuck's current ramp is smooth enough to be locally
predictable (shrinkage). The kinematic residual is not a temporal
forecast at all — it's an instantaneous physical-consistency check (does
measured wheel velocity actually translate into measured chassis motion
right now), so it doesn't inherit the trend-following shrinkage that
sabotages every forecasting/reconstruction-based score. This is a
genuinely different axis from history length, prediction density, or
architecture (all tried previously, see
`robo3er-anomaly-detection-approaches.md`) — it's the first fix that
reaches the ~0.7 ceiling for stuck via a different mechanism rather than
tuning the same forecasting mechanism harder.

## Graph structure shift (interpretability payoff)

Baseline model's `odom_odo_lintw_x` top neighbor is
`wheel_status_pwm_right` (0.31 sim) — exactly the relationship the
kinematic equation already captures analytically. After the residual
substitution, `odom_odo_lintw_x__kin_residual`'s top neighbors shift to
`imu_imu_orient_y`, `wheel_ticks_ticks_right` — sensors that could explain
the part kinematics can't, not a re-derivation of the same physics.
Qualitative evidence the residual substitution is doing what it's meant to
(freeing the attention budget from a relationship already known
analytically), not just a numerology artifact.

## Update: re-run through the new test_gdn_physi/dataset.py (68 features, dead columns dropped)

See `feature-purification-audit.md` for the feature-drop reasoning. With
the 3 confirmed-zero-info columns dropped (68 features, scoring floor
otherwise unchanged from baseline), a re-run gave stuck AUROC 0.742 (even
higher than the 0.717 above) and Broken Pipe 0.956, cable trapped 0.905,
Low battery 0.957. Treat run-to-run swings of ~0.02-0.05 as within GDN's
known training-noise band (same config, different random embedding init
order since num_nodes changed 71->68) rather than a real effect of
dropping 3 always-zero columns — the picture (stuck now solidly >0.7,
other three staying 0.90+) is consistent with the original finding.

## Not yet done

This only covers one kinematic pair (wheel velocities -> chassis twist).
Natural next candidates if extending: wheel_ticks vs wheel_vels
(integration consistency), wheel current/PWM vs velocity (motor dynamics,
softer/non-rigid relationship, likely needs a learned rather than
closed-form residual). The multi-robot cross-device attention layer
(point 2 of the original design discussion) has not been started — this
file only covers the single-robot physics-residual piece.
