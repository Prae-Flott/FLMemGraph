# Paderborn: fixed-mu torque residual -- confirms the collinearity
# diagnosis, but doesn't rescue the residual approach (2026-08-07)

## What was built, and why

Follow-up to `memory/paderborn-torque-residual-gdn.md`'s diagnosis: the
free-fit torque model's negative force coefficient was traced to
Paderborn's 4 operating conditions not varying force/speed/torque
independently (force only takes 2 values, confounded with the other
two), so a 2-predictor OLS fit can't reliably separate the effects.

Per a design clarification (`force` = radial load applied directly to
the bearing housing by an independent spring-screw mechanism; `torque` =
measured by a sensor between the drive motor and the bearing/flywheel/
load-motor stack, closer to "motor output torque" than "bearing friction
torque" itself -- the two are physically DIFFERENT quantities at DIFFERENT
points in the drivetrain, connected only through the bearing's friction
physics), added `src/paderborn_physics.fit_torque_model_fixed_mu()`:
FIXES the force coefficient from a published rolling-bearing friction
coefficient (`mu` = 0.0014, midpoint of the catalog range 0.0010-0.0018)
via the textbook formula `a = 0.5*mu*d_m` (d_m = 28.55mm pitch diameter,
converted to meters) -- only `b` (speed/viscous term) and `c` (intercept)
are still fit. Removes one of the two confounded free parameters,
sidestepping the collinearity problem for the FORCE term specifically.

**A first honest finding, before even running GDN**: with `mu` fixed,
R^2 dropped further, to 0.089 (vs. the free-fit version's 0.310). This
by itself is informative -- the physically-grounded friction contribution
(`a*force`) is TINY relative to the total measured torque range
(`a≈2e-5 Nm/N`, times force~270-1200N, gives only ~0.005-0.024 Nm of
predicted friction torque, against a measured torque range of
0.45-1.46 Nm) -- i.e. even with a correct, non-confounded physical
constant, bearing friction alone explains only a small slice of this
rig's total shaft torque; most of it comes from the commanded load
(load motor), not the test bearing's own friction. This is a genuine
property of the test rig, not a modeling mistake.

## GDN result: three-way comparison

| method | improved | worse | unchanged (of 26) |
|---|---|---|---|
| B: free-fit torque residual | 3 | 10 | 13 |
| C: fixed-mu torque residual | 1 | 12 | 13 |

Neither beats plain baseline overall. But comparing B vs. C's MAGNITUDE
of damage on the hardest bearings (the ones both variants hurt) shows the
fixed-mu version is consistently LESS harmful:

| bearing | B (free-fit) delta | C (fixed-mu) delta |
|---|---|---|
| KA05 | -0.146 | **-0.106** |
| KA09 | -0.209 | **-0.195** |
| KA30 | -0.197 | **-0.095** |
| KI05 | -0.217 | **-0.086** |
| KI07 | -0.195 | **-0.099** |
| KI08 | -0.171 | **-0.090** |
| KI04 | -0.153 | **-0.113** |

Every one of these got LESS worse with the fixed, non-confounded
coefficient -- a real, measurable confirmation that the earlier
collinearity diagnosis was correct and meaningful, not just a plausible-
sounding story. But it's a partial rescue, not a full one: C has MORE
bearings showing some degradation than B (12 vs 10, though smaller in
magnitude), and it slightly hurt two bearings B had been neutral-to-
positive on (KI01: +0.010 -> -0.061, KB23: +0.004 -> -0.015).

## Honest overall verdict

Three physics-residual attempts on Paderborn now (current-based,
free-fit torque-based, fixed-mu torque-based) -- NONE net-improve
detection. The fixed-mu version's smaller damage magnitude is a genuine,
useful confirmation that removing collinearity helps (validates the
diagnosis), but the underlying issue restated by the R^2=0.089 finding is
more fundamental: **the friction-torque signal this residual is trying to
isolate is just a small fraction of this rig's total shaft torque
budget**, so even a perfectly-specified, non-confounded linear model of
it has little explanatory power to residualize against productively.
This doesn't rule out the bearing-geometry defect-frequency approach
(`characteristic_frequencies()`, still unused) -- that one measures
something structurally different (spectral content at specific
frequencies, not a linear DC-ish torque balance) and doesn't share this
same "friction is a small fraction of the measured quantity" problem.

Full report: `checkpoints/paderborn_torque_residual_gdn_report.json`
(now contains both the free-fit and fixed-mu models' full results).
