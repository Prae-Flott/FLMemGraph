# Paderborn: unified torque-balance residual (2026-08-07)

## What was built, per the user's design direction

Following a design conversation converging on "speed, torque, friction,
and load torque should be unified into ONE torque-balance equation"
rather than treating current-vs-torque and force-vs-friction as separate
disconnected chains: `src/paderborn_physics.py.fit_torque_model()` /
`apply_torque_residual()` -- `predicted_torque = a*force + b*speed + c`,
fit on healthy FIT-split data, residual = measured torque - predicted.

**Checked before implementing** (per the design conversation): shaft
speed is essentially CONSTANT within each 4-second recording (std/mean
< 0.03% in every file sampled) -- these are 4 discrete steady-state
operating points, not a continuous transient, so an inertial `J*(dω/dt)`
term would be ~0 and dominated by differentiation noise. Used `speed`'s
raw LEVEL instead, justified by the viscous (speed-dependent) component
of rolling-bearing friction, not inertial dynamics.

`benchmark/run_paderborn_torque_residual_gdn.py`: trains plain GDN twice,
same seed/architecture -- (A) baseline 3 nodes (vibration_1 +
phase_current_1/2, raw, matching the earlier physics-residual script's
method A) vs. (B) the SAME 3 nodes PLUS `torque_residual` as a 4th node
(added, not substituted -- since raw current already outperformed
residualized current in the prior experiment, no reason to give that up).

## Result: adding the residual made things worse for most bearings --
## a third consecutive negative finding for this dataset's physics-prior
## attempts

| outcome | count (of 26 damaged bearings) |
|---|---|
| improved | 3 |
| got worse | 10 |
| unchanged | 13 (mostly already at the 1.000 ceiling) |

Category means: outer_ring 0.697 -> 0.627 (-0.070), inner_ring 0.819 ->
0.753 (-0.066), combined 0.998 -> 1.000 (+0.001, already near-ceiling).
The bearings that got WORSE are overwhelmingly the ones that were already
the hardest cases (KA05/06/08/09/30, KI04/05/07/08 -- several dropped by
0.15-0.22 AUROC). Full per-bearing table:
`checkpoints/paderborn/paderborn_torque_residual_gdn_report.json`.

## Root cause found: NOT just "the physical relation is weak" -- this
## dataset's experimental design doesn't have enough independent
## variation in the predictors to support this kind of regression at all

The fitted model: `torque ~= -0.000493*force + -0.000443*speed + 2.116`
(R^2=0.310). **The force coefficient is NEGATIVE** -- physically
implausible (higher radial load should require MORE friction torque to
overcome, not less). This is a red flag, checked further:

Paderborn's 4 operating conditions are `N15_M07_F10`, `N09_M07_F10`,
`N15_M01_F10`, `N15_M07_F04` -- verified directly (20 files each, exactly
4 unique condition strings per bearing). `force` only takes 2 values
(1000N in 3 of 4 conditions, 400N in exactly 1), and that 400N condition
ONLY co-occurs with one specific (speed, torque) combination
(`N15_M07`). **`force`, `speed`, and the commanded torque level are NOT
independently varied in this dataset's design -- they're locked together
across just 4 discrete points.** A 2-predictor OLS regression across
pooled healthy data is therefore not estimating true physical
sensitivities (`∂torque/∂force` holding speed fixed, etc.) -- it's
effectively fitting a plane through 4 cluster centroids where each
predictor is confounded with "which of the 4 conditions this file is
from." The negative force coefficient and the earlier current model's
weak R^2 (~0.25) are both symptoms of this SAME underlying limitation,
not two unrelated weak-signal findings.

**This reframes both physics-residual attempts on this dataset**: the
current-based residual (`run_paderborn_physics_gdn.py`) and this
torque-based one both tried to fit a 2-variable linear model across data
where the 2 variables are collinear by experimental design, not just
noisy. Neither result should be read as "torque/speed/force don't relate
to current/friction physically" -- they should be read as "this
regression approach cannot reliably separate those effects from THIS
dataset's 4-point design," a methodological limitation, not (necessarily)
a physical one.

## What would actually work better, if this direction is revisited

1. **Don't fit at all -- use the catalog friction coefficient directly.**
   `src/paderborn_physics.py` already has the bearing's pitch diameter
   (`d_m` = 28.55mm); a published rolling-bearing friction coefficient
   range (mu ~ 0.0010-0.0018 for ball bearings) could compute
   `predicted_torque = 0.5*mu*force*d_m + c` with `mu` FIXED (not fit),
   only the intercept `c` (baseline/no-load torque) fit from data --
   sidesteps the collinearity problem entirely since it's a 1-predictor
   fit, not 2, and the slope comes from a hardware/materials constant
   the same way the motor's Kt did.
2. **Fit within each of the 4 conditions separately**, or only ever
   compare a bearing's residual against the SAME condition's healthy
   baseline (matches `gdn_score`'s per-node calibration idea, but applied
   at the condition level too) -- avoids needing one pooled regression to
   generalize across conditions it can't actually distinguish.
3. **More operating conditions with independently-varied force/speed/
   torque would fix this at the data level**, but that's not available --
   this dataset's design fixed 4 specific combinations, not a full
   factorial grid.

## Consistent pattern across all three physics-prior attempts on this
## dataset (current-residual, torque-residual, and now the collinearity
## diagnosis)

Every attempted physics-residual on Paderborn so far has made detection
WORSE, not better -- a real, repeated, honestly-reported finding, unlike
robo3er where the equivalent residual was the single biggest improvement
found anywhere in the project. The dataset's own reference paper's own
diagnostic method (motor current signature analysis) still works via
spectral/envelope analysis at the bearing's characteristic defect
frequencies (`characteristic_frequencies()`, implemented, never
exercised as a feature) -- unlike a linear force/speed/torque regression,
that approach doesn't depend on this dataset's confounded experimental
design at all, and remains the most promising untried direction if this
dataset's physics-prior work continues.

Full report: `checkpoints/paderborn/paderborn_torque_residual_gdn_report.json`.
