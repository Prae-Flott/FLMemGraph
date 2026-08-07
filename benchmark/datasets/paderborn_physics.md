# Paderborn KAt bearing: physical prior relationships & formulas

Full implementation: `src/paderborn_physics.py` (formulas + the fitted
residual actually used), `benchmark/run_paderborn_physics_gdn.py` (the
`kinematics.py`-style training script this was built for). See
`robo3er_physics.md` for the reference pattern this follows.

## Platform

Type 6203 deep-groove ball bearing under a driven shaft, KAt lifetime
test rig (see `docs/paderborn_bearing_KAt2016.pdf`). Two independent
physical relations are documented here: (1) bearing geometry -> defect
frequencies (reference formulas, not yet wired into a feature pipeline),
and (2) motor torque/speed -> current envelope (the actual residual used
by `run_paderborn_physics_gdn.py` -- with an honest negative result, see
below).

## 1. Bearing geometry -> characteristic defect frequencies (reference only)

Bearing 6203 geometry, verified against the paper's Table 1: 8 rolling
elements, 6.75mm roller diameter, 28.55mm pitch diameter, 0 degree
contact angle (deep-groove, not angular-contact). Standard formulas
(same structure used for the deleted IMS bearing work's ZA-2115
bearing, different geometry numbers):

```
BPFO = (n/2)*f_r*(1 - (d/D)*cos(theta))      outer race defect frequency
BPFI = (n/2)*f_r*(1 + (d/D)*cos(theta))      inner race defect frequency
BSF  = (D/2d)*f_r*(1 - (d/D)^2*cos^2(theta)) rolling element defect frequency
FTF  = (f_r/2)*(1 - (d/D)*cos(theta))        cage/train frequency
```

Unlike IMS (fixed nominal 2000 RPM assumed), Paderborn's `speed` channel
is directly measured per file, so `f_r` here can be computed from actual
data rather than assumed constant -- a strictly harder/more precise prior
than IMS's version, though this precision hasn't been exploited yet
(these formulas are implemented in `src/paderborn_physics.py.
characteristic_frequencies()` but not yet used as a feature -- e.g. no
envelope-spectrum band-energy extraction has been built for this dataset,
unlike the (deleted) IMS exploration that tried this).

At the nominal 1500 RPM condition: BPFO=76.4Hz, BPFI=123.6Hz,
BSF=49.9Hz (2xBSF=99.8Hz), FTF=9.5Hz.

## 2. Motor torque/speed -> current envelope (the residual actually used)

**The physical idea** (same "expert-knowledge-informed structure,
data-fit parameters" pattern as `kinematics.py`): basic motor physics --
current magnitude scales with load torque (and to a lesser extent,
speed). A bearing defect injects extra torque ripple onto the shaft that
isn't explained by the smooth baseline torque/speed relationship alone --
analogous to `kinematics.py`'s "wheel motion not showing up in odometry
== slip."

```
predicted_current_envelope = a*torque + b*speed + c   (ordinary least squares,
                                                          fit on healthy FIT
                                                          split only)
residual = measured_current_envelope - predicted_current_envelope
```

Fit from `benchmark/run_paderborn_physics_gdn.py`'s run (pooled healthy
K001-K006 fit split, `src/paderborn_physics.py.fit_current_model`):

```
phase_current_1 ~= 1.174*torque + 0.000078*speed + 0.065   (R^2=0.251)
phase_current_2 ~= 1.170*torque + 0.000072*speed + 0.085   (R^2=0.245)
```

The current envelope is extracted via RMS-per-block decimation (not
mean) from the raw ~64kHz AC signal -- see `paderborn_adapter.py`'s
`load_bearing_with_physics()` docstring for why mean-decimation is the
wrong representation for an oscillating current signal.

## Result: the residual made detection WORSE, not better -- an honest
## negative, unlike robo3er's kinematics residual

Same-seed, same-architecture, single-variable comparison (plain
`src/gdn_model.GDN`, forecasting task, `vibration_1` + raw-vs-residual
current, evaluated per damaged bearing):

| category | raw current mean AUROC | physics-residual mean AUROC | delta |
|---|---|---|---|
| outer_ring | 0.697 | 0.598 | **-0.100** |
| inner_ring | 0.819 | 0.751 | **-0.068** |
| combined | 0.998 | 0.970 | -0.029 |

**23 of 26 damaged bearings got WORSE** with the physics residual (2 flat
at the 1.000 ceiling, only KI01 improved, +0.025). This is the opposite
of robo3er's kinematics residual, which was this project's single
biggest improvement anywhere.

**Why, most likely**: the fitted relation's R^2 is only ~0.25 -- torque
and speed together explain barely a quarter of the current envelope's
variance under NORMAL operation. Contrast with `kinematics.py`'s
differential-drive equation, which is a near-deterministic relation under
normal conditions (wheel velocity essentially DOES determine odometry
absent slip) -- subtracting a high-fidelity prediction cleanly isolates
the anomalous remainder. Subtracting a WEAK prediction (R^2=0.25) instead
mostly subtracts noise, and what's left is dominated by that
regression's own residual noise (three-quarters of the original signal's
variance, restructured through an imperfect linear model) rather than a
clean "explained vs. anomalous" split -- actively harder for GDN to
learn a stable normal baseline from than the raw current envelope was.

**Secondary finding, not the main point of this comparison but worth
recording**: this run's "raw current" arm (plain `GDN` forecasting +
RMS-decimated current) scored HIGHER than the earlier
`run_paderborn_fl_model.py` result (mean-decimated current +
`FLGDNMemory` reconstruction) on both categories measured in both runs
(outer_ring 0.697 vs 0.641, inner_ring 0.819 vs 0.750) -- plausibly the
RMS-vs-mean decimation fix for current, though the task type
(forecasting vs. reconstruction) also changed between the two runs, so
this isn't an isolated single-variable result either.

## 3. Unified torque-balance residual (force+speed -> torque) -- also negative

Follow-up design idea: instead of routing through the motor (Section 2),
connect directly to the bearing. The torque sensor sits between the
drive motor and the bearing/flywheel/load-motor stack, so measured
torque is a steady-state balance of bearing friction (driven by the
independently, externally-applied radial `force`) plus the commanded
load -- residualizing torque against `force` (and `speed`, for the
viscous-friction component) should isolate "torque beyond what basic
bearing-friction physics predicts," a more direct bearing-health signal
than current.

```
predicted_torque = a*force + b*speed + c
residual = measured_torque - predicted_torque
```

`speed`'s LEVEL is used, not `dω/dt` -- checked directly, shaft speed is
essentially constant within each 4s recording (std/mean < 0.03%), so
there's no meaningful inertial dynamics to capture; the speed term
instead represents the viscous (speed-dependent) friction component.

Fit from `benchmark/run_paderborn_torque_residual_gdn.py`'s run:

```
torque ~= -0.000493*force + -0.000443*speed + 2.116   (R^2=0.310)
```

**Result: also made detection worse.** Added as a 4th node alongside the
3 raw channels (not a replacement, since raw current already beat
residualized current in Section 2): of 26 damaged bearings, 3 improved,
10 got worse, 13 unchanged (mostly already at the AUROC=1.000 ceiling).
Category means: outer_ring 0.697->0.627, inner_ring 0.819->0.753.

## Root cause diagnosis: NOT just "weak relation" -- this dataset's
## experimental design confounds the predictors

**The fitted force coefficient is NEGATIVE** -- physically implausible
(more radial load should require MORE friction torque, not less). Traced
to why: Paderborn's 4 operating conditions (`N15_M07_F10`, `N09_M07_F10`,
`N15_M01_F10`, `N15_M07_F04`) do NOT vary `force`/`speed`/`torque`
independently -- `force` only takes 2 values, and the 400N one ONLY
co-occurs with one specific (speed, torque) combination. A 2-predictor
OLS regression across these 4 discrete, confounded design points cannot
reliably separate a true `∂torque/∂force` sensitivity from "which of the
4 conditions this came from" -- it's fitting a plane through 4 cluster
centroids, not estimating physical sensitivities. This is the same root
cause behind Section 2's weak R^2 (~0.25) too, not a separate,
unrelated weak-signal finding -- both regressions were asked to do
something this dataset's controlled design doesn't actually support.

**Tried the fix -- confirms the diagnosis, doesn't rescue the result.**
`src/paderborn_physics.fit_torque_model_fixed_mu()`: FIXES the force
coefficient from the published catalog range (`mu`=0.0014 midpoint,
`a=0.5*mu*d_m`), only fits `b`(speed) and `c`(intercept). Two findings:

1. **R^2 dropped further, to 0.089** (vs. free-fit's 0.310) -- with `mu`
   fixed, `a*force` only predicts ~0.005-0.024 Nm of friction torque
   against a measured range of 0.45-1.46 Nm. Bearing friction is
   genuinely a small fraction of this rig's total shaft torque (most
   comes from the commanded load motor, not the test bearing) -- a real
   property of the rig, not a modeling error.
2. **GDN result confirms the collinearity diagnosis, partially**: every
   bearing that degraded with the free-fit residual degraded LESS with
   the fixed, non-confounded coefficient (e.g. KA30: -0.197 -> -0.095,
   KI05: -0.217 -> -0.086) -- but it's still net negative overall (1
   improved / 12 worse vs. free-fit's 3 improved / 10 worse). Full
   writeup: `memory/paderborn-fixed-mu-torque-residual.md`.

`force`/`torque` measure physically DIFFERENT things at different points
in the drivetrain (`force` = radial load applied directly to the bearing
housing by an independent spring-screw mechanism; `torque` = measured
between the drive motor and the bearing/flywheel/load-motor stack,
closer to "motor output torque" than "bearing friction torque" itself) --
connected only through the bearing's friction physics, which this rig's
torque budget shows is a small contributor to the total.

## Takeaway across all four physics-residual attempts on this dataset
## (current; torque free-fit; torque fixed-mu)

Every one made detection WORSE, not better -- consistent, not a fluke.
Section 2's success criterion (`kinematics.py`'s near-deterministic
relation under normal operation) doesn't hold here for any of them.
Section 3's diagnosis explains part of why (4-point experimental design
confounds force/speed/torque, destabilizing a free multi-predictor OLS
fit) -- fixing that with a published catalog constant made the damage
consistently SMALLER (confirming the diagnosis was real) but not
positive, because the deeper issue is that bearing friction is simply a
small fraction of this rig's total torque budget (R^2=0.089 even with a
correctly-specified, non-confounded model). Verified no bugs/leakage in
any of these fits (FIT-split-only, no calib/test/fault contamination).
The unused bearing-geometry defect-frequency formulas (Section 1) remain
the most promising avenue if this dataset is revisited -- unlike a linear
force/speed/torque regression, envelope-spectrum band-energy extraction
at the bearing's characteristic frequencies doesn't depend on this
confounded experimental design or on friction being a large share of the
torque budget.

Full reports: `checkpoints/paderborn_physics_gdn_report.json`,
`checkpoints/paderborn_torque_residual_gdn_report.json`. Full analysis:
`memory/paderborn-physics-residual-gdn.md`,
`memory/paderborn-torque-residual-gdn.md`.
