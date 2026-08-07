"""
Physics prior for the Paderborn KAt bearing dataset, playing the same
architectural role `kinematics.py` plays for robo3er: a physically-
motivated relation between two INDEPENDENTLY MEASURED signals, fit from
normal (healthy) data only, whose residual (measured minus predicted) is
fed to GDN instead of the raw signal -- "expert-knowledge-informed
structure, data-fit parameters," per this project's established pattern.

## The three things bundled here

1. `characteristic_frequencies()`: bearing 6203's BPFO/BPFI/BSF/FTF
   defect frequencies, computed from KNOWN geometry (verified against
   `docs/paderborn_bearing_KAt2016.pdf`'s Table 1) and MEASURED shaft
   speed (this dataset's `speed` channel -- unlike the deleted IMS
   bearing work, which had to assume a fixed nominal RPM, here the actual
   instantaneous speed is a real measured signal, so this is a strictly
   harder/more precise prior than IMS's version). Reference-only in this
   module -- not yet wired into a residual/feature pipeline, documented
   for `benchmark/datasets/paderborn_physics.md` and future work.

2. `fit_current_model()` / `apply_current_residual()`: an earlier
   `kinematics.py`-analogous RESIDUAL (motor current vs. torque/speed via
   the motor's torque constant), tried in `run_paderborn_physics_gdn.py`
   and found to hurt detection (see that script's/`paderborn_physics.md`'s
   results) -- kept for reference/comparison, superseded by (3) below as
   the primary physics-residual signal.

3. `fit_torque_model()` / `apply_torque_residual()`: the CURRENT primary
   residual, used by `benchmark/run_paderborn_torque_residual_gdn.py`.
   Unlike (2), which sits one physical step removed from the bearing
   (torque -> current, via the MOTOR's electromagnetic relation), this
   one connects directly to the bearing: the test rig's torque sensor
   sits between the drive motor and the bearing/flywheel/load-motor
   stack (`docs/paderborn_bearing_KAt2016.pdf` Figure 4), so measured
   torque is a steady-state balance of bearing friction torque (driven
   by the INDEPENDENTLY, externally-applied radial `force`) plus the
   commanded load. Residualizing torque against `force` (and `speed`,
   for the speed-dependent/viscous friction component -- see below)
   isolates "torque beyond what basic bearing-friction physics predicts
   from the applied load," a more direct bearing-health signal than
   current ever was.

## Why a 2-variable linear fit, not a full contact-mechanics model

A first-principles bearing friction model (EHL lubrication theory,
Hertzian contact, etc.) or a full motor circuit model (rotor equations,
slip, electrical frequency) is a much bigger undertaking than this
project's "expert-knowledge-informed structure, data-fit parameters"
convention calls for. Matching `kinematics.py`'s own choice (a
2-parameter differential-drive equation, not a full rigid-body dynamics
model), this module fits the SIMPLEST physically-motivated relation --
linear in the driving quantity/quantities -- and treats it as a baseline
to residualize against, not a claim of mechanistic completeness.

## Why `speed`, not `dω/dt` (angular acceleration), in the torque model

Checked directly (see `memory/paderborn-torque-residual-gdn.md`): shaft
speed is essentially CONSTANT within each 4-second recording (std/mean
< 0.03% in every file checked) -- these are 4 discrete steady-state
operating points (900/1500 RPM x two torque/force levels), not a
continuous transient. An inertial term `J*(dω/dt)` would therefore be
~0 for nearly the entire window and dominated by numerical-differentiation
noise if computed anyway. `speed`'s absolute LEVEL is kept as a linear
predictor instead, justified by a different, still-legitimate physical
mechanism: rolling-bearing friction has both a load-dependent (Coulomb-
like) component and a speed-dependent (viscous, lubricant-drag) component
-- `a*force` captures the former, `b*speed` the latter.
"""
import numpy as np


def _fit_linear_2var(y, x1, x2):
    """Fit y = a*x1 + b*x2 + c by ordinary least squares. Shared by
    fit_current_model and fit_torque_model -- same 2-predictor-plus-
    intercept structure, different physical quantities plugged in.
    Returns (a, b, c, r2)."""
    A = np.stack([x1.reshape(-1), x2.reshape(-1), np.ones(x1.size)], axis=1)
    yy = y.reshape(-1)
    (a, b, c), _, _, _ = np.linalg.lstsq(A, yy, rcond=None)
    y_hat = A @ np.array([a, b, c])
    ss_res = np.sum((yy - y_hat) ** 2)
    ss_tot = np.sum((yy - yy.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(a), float(b), float(c), r2

# Bearing type 6203 (per docs/paderborn_bearing_KAt2016.pdf Table 1,
# "manufacturer specific information about bearing" -- verified, not
# re-derived from a generic spec sheet).
NUM_ROLLING_ELEMENTS = 8
ROLLER_DIAMETER_MM = 6.75
PITCH_DIAMETER_MM = 28.55
CONTACT_ANGLE_DEG = 0.0  # deep-groove ball bearing, no angular contact


def characteristic_frequencies(rpm, n=NUM_ROLLING_ELEMENTS, d=ROLLER_DIAMETER_MM,
                                 D=PITCH_DIAMETER_MM, contact_angle_deg=CONTACT_ANGLE_DEG):
    """Returns {name: frequency_hz} for BPFO, BPFI, BSF, 2xBSF, FTF, and
    the shaft rate f_r itself, given an RPM (can be a scalar or an array
    -- e.g. the mean of a file's measured `speed` channel)."""
    fr = np.asarray(rpm) / 60.0
    theta = np.radians(contact_angle_deg)
    ratio = (d / D) * np.cos(theta)

    bpfo = (n / 2) * fr * (1 - ratio)
    bpfi = (n / 2) * fr * (1 + ratio)
    bsf = (D / (2 * d)) * fr * (1 - (d / D) ** 2 * np.cos(theta) ** 2)
    ftf = (fr / 2) * (1 - ratio)

    return {"shaft_rate": fr, "BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "2xBSF": 2 * bsf, "FTF": ftf}


def fit_current_model(current_envelope, torque, speed):
    """Fit predicted_current = a*torque + b*speed + c by ordinary least
    squares, from healthy FIT-split data only (caller's responsibility --
    matches kinematics.py's fit_kinematic_params, which is likewise only
    ever called on the fit split, never calib/test/fault).

    `current_envelope`, `torque`, `speed`: same-shape 1D (or flattened)
    arrays, one value per aligned time sample.

    Returns dict of fitted params + R^2 (a fitted intercept far from 0,
    or a low R^2, is itself a diagnostic worth reporting -- same
    convention as kinematics.py's b_v/b_w intercept note). See module
    docstring item 2 -- this is the SUPERSEDED current-based residual,
    kept for reference/comparison, not the primary one anymore."""
    a, b, c, r2 = _fit_linear_2var(current_envelope, torque, speed)
    return {"a_torque": a, "b_speed": b, "c_intercept": c, "r2": r2}


def apply_current_residual(current_envelope, torque, speed, params):
    """Return current_envelope - predicted(torque, speed; params), same
    shape as current_envelope. Applied identically to fit/calib/
    test_normal/fault windows using the SAME fit-derived params --
    matches kinematics.py's apply_kinematic_residual leakage discipline."""
    predicted = params["a_torque"] * torque + params["b_speed"] * speed + params["c_intercept"]
    return current_envelope - predicted


def fit_torque_model(torque, force, speed):
    """Fit predicted_torque = a*force + b*speed + c by ordinary least
    squares, from healthy FIT-split data only. `a` structurally stands in
    for the load-dependent (Coulomb-like) friction coefficient term
    (`0.5*mu*d_m` in the textbook rolling-bearing-friction formula --
    `mu`, the actual friction coefficient, isn't precisely known for this
    rig's specific lubrication state, so it's fit rather than assumed,
    same "fix the structure, fit the constant" logic as kinematics.py's
    k_v/k_w). `b` stands in for the speed-dependent (viscous) friction
    component. See module docstring for why `speed`'s LEVEL is used, not
    its derivative.

    `torque`, `force`, `speed`: same-shape 1D (or flattened) arrays, one
    value per aligned time sample.

    Returns dict of fitted params + R^2."""
    a, b, c, r2 = _fit_linear_2var(torque, force, speed)
    return {"a_force": a, "b_speed": b, "c_intercept": c, "r2": r2}


def apply_torque_residual(torque, force, speed, params):
    """Return torque - predicted(force, speed; params), same shape as
    torque. Applied identically to fit/calib/test_normal/fault windows
    using the SAME fit-derived params -- matches kinematics.py's
    apply_kinematic_residual leakage discipline."""
    predicted = params["a_force"] * force + params["b_speed"] * speed + params["c_intercept"]
    return torque - predicted


# Published rolling-element-bearing friction coefficient range for ball bearings
# (textbook/catalog value -- e.g. SKF/Palmgren simplified friction model
# M = 0.5*mu*F*d_m -- NOT specific to this exact rig's lubricant/wear state, a
# generic engineering figure). Midpoint used as the FIXED slope below.
CATALOG_MU_BALL_BEARING = (0.0010, 0.0018)
CATALOG_MU_MIDPOINT = sum(CATALOG_MU_BALL_BEARING) / 2  # 0.0014


def fit_torque_model_fixed_mu(torque, force, speed, mu=CATALOG_MU_MIDPOINT):
    """Alternative to fit_torque_model(): FIXES the force coefficient
    from the published catalog friction coefficient instead of fitting it
    -- sidesteps the collinearity problem found in fit_torque_model()
    (Paderborn's 4 operating conditions don't vary force/speed/torque
    independently, so a free 2-predictor OLS fit can't reliably separate
    their effects -- see memory/paderborn-torque-residual-gdn.md). Here
    `a` = 0.5*mu*d_m is a FIXED physical constant (mu from the catalog
    range, d_m = PITCH_DIAMETER_MM converted to meters); only `b` (speed/
    viscous-friction coefficient) and `c` (intercept) are fit -- one
    fewer free parameter fighting the same confounded 4-point design.

    `torque` in Nm, `force` in N (native units of this dataset's
    channels); `d_m` converted mm->m so `a` comes out in Nm/N."""
    d_m_meters = PITCH_DIAMETER_MM / 1000.0
    a = 0.5 * mu * d_m_meters
    residual_after_friction = torque.reshape(-1) - a * force.reshape(-1)
    # fit residual_after_friction = b*speed + c (1 remaining free physical term + intercept)
    A = np.stack([speed.reshape(-1), np.ones(speed.size)], axis=1)
    (b, c), _, _, _ = np.linalg.lstsq(A, residual_after_friction, rcond=None)
    y_hat = a * force.reshape(-1) + b * speed.reshape(-1) + c
    yy = torque.reshape(-1)
    ss_res = np.sum((yy - y_hat) ** 2)
    ss_tot = np.sum((yy - yy.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"a_force": float(a), "a_force_is_fixed_not_fit": True, "mu_used": float(mu),
            "b_speed": float(b), "c_intercept": float(c), "r2": r2}


if __name__ == "__main__":
    freqs = characteristic_frequencies(1500)
    print("6203 bearing characteristic frequencies at 1500 RPM:")
    for name, hz in freqs.items():
        print(f"  {name:<12}{hz:>8.2f} Hz")
