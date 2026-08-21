"""
Differential-drive kinematics prior for robo3er.

robo3er's 5 robots are iRobot Create3-class differential-drive platforms.
Textbook forward kinematics for a differential-drive robot relates the two
wheel velocities (v_l, v_r) to the chassis's linear and angular velocity:

    v_lin   = r/2 * (v_l + v_r)
    v_ang   = r/L * (v_r - v_l)

where r is the effective wheel radius and L the effective track width.
Both are physical constants of one specific robot -- unknown here in the
exact units these logged topics use (could be wheel surface speed in m/s,
could be raw encoder rate, calibration drift, gearbox slack, etc.), so
rather than hard-coding a spec-sheet number, this module *fits* the two
scalar constants (k_v = r/2, k_w = r/L) from held-out normal data by least
squares, keeping the physically-motivated two-parameter structure of the
equation instead of a free 4-coefficient linear regression. That's the
"expert-knowledge-informed structure, data-fit parameters" split described
in the design discussion this folder implements.

This choice is independently confirmed, not just convenient: the official
Create 3 docs state the `/odom` topic is a sensor-fusion dead-reckoning
estimate (wheel encoders + IMU + optical-flow mouse), not a pure wheel-
kinematics integration -- see
`docs/robo3er_physics_constants_audit.md`. Plugging the manufacturer's own
URDF wheel radius/track width into r/2, r/L predicts k_v/k_w that are off
from the real per-robot fitted values by DIFFERENT ratios (26.5x vs 7.6x),
which rules out a simple unit-conversion explanation and confirms k_v/k_w
absorb the fusion filter's own gain/bias, not just wheel geometry -- so
hard-coding a spec-sheet r, L would be a strictly worse model than fitting.

Whatever variance in `odom_odo_lintw_x` / `odom_odo_angtw_z` this equation
predicts is treated as "explained by rigid-body kinematics" and stripped
out; what's left (the residual) is fed to GDN instead of the raw signal.
Concretely, a wheel that is commanded/measured to spin but whose motion
does NOT show up in the chassis's actual odometry twist -- i.e. a large
kinematic residual -- is the textbook definition of wheel slip, so this
residual is expected to carry directly relevant signal for robo3er's
"cable trapped" (slip) fault type specifically, on top of whatever
raw-feature signal GDN could already find.

An intercept term (b_v, b_w) is fit alongside each slope to absorb any
constant sensor offset/calibration bias -- physically that term shouldn't
exist for an ideal robot at rest, so a fitted intercept far from 0 is
itself a diagnostic flag (worth reporting to a domain expert), not
something to silently discard.

## Regime-conditioned formulas (rule-based, not learned)

Real-data analysis (`docs/robo3er_regime_kinematics.md`) established that
the wheel<->odometry relationship is NOT state-independent:

  * `sum(v_l, v_r) -> odom_lin` and `diff(v_r, v_l) -> odom_ang` (the two
    relations above) hold up reasonably across both TRANSLATING (two
    wheels turning the same direction) and ROTATING_IN_PLACE (wheels
    turning opposite directions) regimes -- fit them on ALL non-stationary
    data, no gating needed.
  * Each SINGLE-wheel edge (`left/right -> odom_lin`, `left/right ->
    odom_ang`), by contrast, is only a valid approximation while
    TRANSLATING: its R^2 collapses outside that regime, and
    `right -> odom_ang` even flips sign between regimes (a collinearity
    artifact of v_l and v_r moving together while translating, not a real
    physical relationship) -- so these edges must be GATED OFF (not
    scored) outside TRANSLATING, rather than scored with an unreliable
    regime-specific coefficient. This is a rule (a `sign(v_l)==sign(v_r)`
    check), not a learned parameter -- exactly the "keep formulas
    conservative, decide by rule which formula/edge applies" split the
    project settled on.

The regime classifier and per-edge, per-regime fit machinery below are
written to be reusable by the new explicit-formula scoring head (see
project plan step 6), not just by the legacy `apply_kinematic_residual`
substitution this module originally shipped with.
"""
import numpy as np

WHEEL_LEFT = "wheel_vels_velocity_left"
WHEEL_RIGHT = "wheel_vels_velocity_right"
ODOM_LIN_X = "odom_odo_lintw_x"
ODOM_ANG_Z = "odom_odo_angtw_z"

# Regime labels, in the same vocabulary used throughout the project's
# analysis docs and prior discussion.
STATIONARY = "STATIONARY"
TRANSLATING = "TRANSLATING"
ROTATING_IN_PLACE = "ROTATING_IN_PLACE"
REGIMES = (STATIONARY, TRANSLATING, ROTATING_IN_PLACE)

# Below this wheel-speed magnitude (rad/s) on BOTH wheels, the robot is
# classified STATIONARY regardless of sign -- avoids classifying encoder
# noise around zero as a same-/opposite-sign regime.
STATIONARY_THRESH = 0.02

# A regime needs at least this many fit-split samples before its
# per-regime fit is trusted; below this, callers fall back to the
# TRANSLATING (majority-regime) coefficients. Chosen well below the
# smallest regime actually observed in fit-split data (ROTATING_IN_PLACE,
# ~2470 of 236400 frames, i.e. ~1%) so it only fires for a regime that is
# nearly absent, not merely a minority.
MIN_REGIME_SAMPLES = 30

# Edge topology: which single input feature(s) predict which target, how
# they combine, and which regimes the relation is trustworthy in. This is
# the "keep the topology decision a RULE, keep coefficients DATA-FIT" split
# -- active_regimes is fixed by the analysis in
# docs/robo3er_regime_kinematics.md, not learned.
EDGE_SPECS = {
    "sum_to_lin": {
        "inputs": (WHEEL_LEFT, WHEEL_RIGHT),
        "target": ODOM_LIN_X,
        "combine": lambda wl, wr: wl + wr,
        "active_regimes": (TRANSLATING, ROTATING_IN_PLACE),
    },
    "diff_to_ang": {
        "inputs": (WHEEL_LEFT, WHEEL_RIGHT),
        "target": ODOM_ANG_Z,
        "combine": lambda wl, wr: wr - wl,
        "active_regimes": (TRANSLATING, ROTATING_IN_PLACE),
    },
    "left_to_lin": {
        "inputs": (WHEEL_LEFT,),
        "target": ODOM_LIN_X,
        "combine": lambda wl: wl,
        "active_regimes": (TRANSLATING,),
    },
    "right_to_lin": {
        "inputs": (WHEEL_RIGHT,),
        "target": ODOM_LIN_X,
        "combine": lambda wr: wr,
        "active_regimes": (TRANSLATING,),
    },
    "left_to_ang": {
        "inputs": (WHEEL_LEFT,),
        "target": ODOM_ANG_Z,
        "combine": lambda wl: wl,
        "active_regimes": (TRANSLATING,),
    },
    "right_to_ang": {
        "inputs": (WHEEL_RIGHT,),
        "target": ODOM_ANG_Z,
        "combine": lambda wr: wr,
        "active_regimes": (TRANSLATING,),
    },
}


def classify_regime(v_l, v_r, thresh=STATIONARY_THRESH):
    """Rule-based (non-learned) per-frame regime classification from the
    two raw wheel velocities. `v_l`, `v_r` are same-shape arrays (any
    shape). Returns an object-dtype array of the same shape with values in
    {STATIONARY, TRANSLATING, ROTATING_IN_PLACE}.

    Rule (fixed by the differential-drive sign analysis, not fit from
    data): both wheels below `thresh` in magnitude -> STATIONARY; same
    sign -> TRANSLATING (co-rotating: straight/curved driving); opposite
    sign -> ROTATING_IN_PLACE.
    """
    v_l = np.asarray(v_l)
    v_r = np.asarray(v_r)
    stationary = (np.abs(v_l) < thresh) & (np.abs(v_r) < thresh)
    same_sign = (np.sign(v_l) == np.sign(v_r)) & ~stationary
    regime = np.full(v_l.shape, ROTATING_IN_PLACE, dtype=object)
    regime[same_sign] = TRANSLATING
    regime[stationary] = STATIONARY
    return regime


def _lstsq_line(x, y):
    """Fit y = k*x + b by ordinary least squares. Returns (k, b, r2, n)."""
    n = int(x.shape[0])
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    A = np.stack([x, np.ones_like(x)], axis=1)
    (k, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_hat = k * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(k), float(b), r2, n


def fit_kinematic_params(data, fit_idx, cols, min_regime_samples=MIN_REGIME_SAMPLES):
    """Fit k_v, b_v (linear velocity) and k_w, b_w (angular velocity) from
    the FIT split only (never calib/test/fault) to avoid leakage. `data` is
    the raw (unscaled) [N, T, F] array, `fit_idx` the normal-fit window
    indices, `cols` the feature name list.

    Backward-compatible return keys (k_v, b_v, r2_lin, k_w, b_w, r2_ang)
    are the GLOBAL fit across all non-stationary regimes pooled together,
    unchanged from the original single-formula version of this module --
    existing callers that only read these keys see identical values and
    behavior. Two new keys are added on top:

      * "by_regime": {regime_name: {k_v, b_v, r2_lin, k_w, b_w, r2_ang, n}}
        for TRANSLATING and ROTATING_IN_PLACE (STATIONARY is handled
        separately via `stationary_noise_stats`, not a linear fit -- see
        module docstring). A regime with fewer than `min_regime_samples`
        fit-split samples is OMITTED here and callers should fall back to
        the global fit or to TRANSLATING (see `apply_kinematic_residual`
        and `fit_edge_params`).
      * "regime_counts": {regime_name: n} over the fit split, for
        diagnostics/reporting.
    """
    idx = {c: i for i, c in enumerate(cols)}
    wl_all = data[fit_idx][:, :, idx[WHEEL_LEFT]].reshape(-1)
    wr_all = data[fit_idx][:, :, idx[WHEEL_RIGHT]].reshape(-1)
    lin_all = data[fit_idx][:, :, idx[ODOM_LIN_X]].reshape(-1)
    ang_all = data[fit_idx][:, :, idx[ODOM_ANG_Z]].reshape(-1)

    k_v, b_v, r2_lin, _ = _lstsq_line(wl_all + wr_all, lin_all)
    k_w, b_w, r2_ang, _ = _lstsq_line(wr_all - wl_all, ang_all)

    regime = classify_regime(wl_all, wr_all)
    regime_counts = {r: int((regime == r).sum()) for r in REGIMES}

    by_regime = {}
    for r in (TRANSLATING, ROTATING_IN_PLACE):
        mask = regime == r
        n = int(mask.sum())
        if n < min_regime_samples:
            continue
        kv_r, bv_r, r2lin_r, _ = _lstsq_line(wl_all[mask] + wr_all[mask], lin_all[mask])
        kw_r, bw_r, r2ang_r, _ = _lstsq_line(wr_all[mask] - wl_all[mask], ang_all[mask])
        by_regime[r] = {
            "k_v": kv_r, "b_v": bv_r, "r2_lin": r2lin_r,
            "k_w": kw_r, "b_w": bw_r, "r2_ang": r2ang_r,
            "n": n,
        }

    return {
        "k_v": k_v, "b_v": b_v, "r2_lin": r2_lin,
        "k_w": k_w, "b_w": b_w, "r2_ang": r2_ang,
        "by_regime": by_regime,
        "regime_counts": regime_counts,
    }


def stationary_noise_stats(data, fit_idx, cols, thresh=STATIONARY_THRESH, min_samples=200):
    """Compute STATIONARY-regime noise statistics for `odom_odo_lintw_x`
    and `odom_odo_angtw_z`, for use as an absolute-threshold anomaly rule
    (rather than a regression) when the robot is not moving -- see
    docs/robo3er_regime_kinematics.md section 4. Ideally called against the
    CALIB split (fit-split STATIONARY coverage was found to be too sparse,
    ~4 frames, for a reliable std estimate).

    Returns None (with the sample count still reported under "n") when
    fewer than `min_samples` STATIONARY frames are available -- callers
    should treat that as "insufficient data, do not gate on this regime
    yet" rather than silently trusting a noisy estimate.
    """
    idx = {c: i for i, c in enumerate(cols)}
    wl = data[fit_idx][:, :, idx[WHEEL_LEFT]].reshape(-1)
    wr = data[fit_idx][:, :, idx[WHEEL_RIGHT]].reshape(-1)
    lin = data[fit_idx][:, :, idx[ODOM_LIN_X]].reshape(-1)
    ang = data[fit_idx][:, :, idx[ODOM_ANG_Z]].reshape(-1)

    mask = (np.abs(wl) < thresh) & (np.abs(wr) < thresh)
    n = int(mask.sum())
    if n < min_samples:
        return {"n": n, "sufficient": False}

    return {
        "n": n,
        "sufficient": True,
        "odom_lin_std": float(lin[mask].std()),
        "odom_ang_std": float(ang[mask].std()),
        "odom_lin_mean_abs": float(np.abs(lin[mask]).mean()),
        "odom_ang_mean_abs": float(np.abs(ang[mask]).mean()),
    }


def fit_edge_params(data, fit_idx, cols, edge_specs=EDGE_SPECS, min_regime_samples=MIN_REGIME_SAMPLES):
    """Fit one scalar linear relation (k, b) per (edge, active regime) in
    `edge_specs`, for use by the explicit-formula scoring head (project
    plan step 6). Unlike `fit_kinematic_params` (which only covers the two
    sum/diff hyper-edges for the legacy residual-substitution path), this
    covers every edge in `edge_specs`, including the four single-wheel
    edges that are only fit/active in TRANSLATING.

    Returns {edge_name: {regime_name: {k, b, r2, n}}}. A regime absent from
    an edge's own "active_regimes" is never fit or returned for that edge
    -- e.g. "left_to_lin" (active only in TRANSLATING) never has a
    "ROTATING_IN_PLACE" key, by design (gated off, not scored with an
    unreliable coefficient; see module docstring). A regime that IS in
    "active_regimes" but has fewer than `min_regime_samples` fit-split
    samples is also omitted -- callers must handle a missing regime key
    themselves (typically: don't score that edge on frames in that
    regime), not silently substitute another regime's coefficients.
    """
    idx = {c: i for i, c in enumerate(cols)}
    wl_all = data[fit_idx][:, :, idx[WHEEL_LEFT]].reshape(-1)
    wr_all = data[fit_idx][:, :, idx[WHEEL_RIGHT]].reshape(-1)
    regime = classify_regime(wl_all, wr_all)

    feature_cache = {}

    def get_feature(name):
        if name not in feature_cache:
            feature_cache[name] = data[fit_idx][:, :, idx[name]].reshape(-1)
        return feature_cache[name]

    results = {}
    for edge_name, spec in edge_specs.items():
        target = get_feature(spec["target"])
        input_arrays = [get_feature(name) for name in spec["inputs"]]
        combined = spec["combine"](*input_arrays)

        per_regime = {}
        for r in spec["active_regimes"]:
            mask = regime == r
            n = int(mask.sum())
            if n < min_regime_samples:
                continue
            k, b, r2, n = _lstsq_line(combined[mask], target[mask])
            per_regime[r] = {"k": k, "b": b, "r2": r2, "n": n}
        results[edge_name] = per_regime

    return results


def apply_kinematic_residual(data, cols, params, regime_gating=False):
    """Return a COPY of `data` with `odom_odo_lintw_x` and
    `odom_odo_angtw_z` replaced by their kinematic residual (actual minus
    the differential-drive prediction from the two wheel velocities). Every
    other of the 71 columns is untouched. Applied identically to
    fit/calib/test_normal/fault windows using the SAME fit-derived
    params -- the model never sees fault or held-out data during fitting.

    `regime_gating=False` (default, UNCHANGED from the original version of
    this module): always uses the single global `params["k_v"]/["b_v"]`,
    `params["k_w"]/["b_w"]` pair, exactly reproducing prior behavior and
    all previously-recorded benchmark numbers.

    `regime_gating=True`: classifies each frame's regime from its own
    (v_l, v_r) and looks up `params["by_regime"][regime]` for that frame's
    prediction; STATIONARY frames and any regime missing from
    `params["by_regime"]` (too few fit-split samples) fall back to the
    global coefficients, since sum/diff formulas were found to remain
    reasonably valid across regimes (see module docstring) and a
    near-zero-wheel-speed frame predicts a near-zero residual under either
    the global or the TRANSLATING-regime coefficients. `params` must come
    from `fit_kinematic_params` for this mode (needs "by_regime").
    """
    idx = {c: i for i, c in enumerate(cols)}
    out = data.copy()

    wl = data[:, :, idx[WHEEL_LEFT]]
    wr = data[:, :, idx[WHEEL_RIGHT]]

    if not regime_gating:
        lin_pred = params["k_v"] * (wl + wr) + params["b_v"]
        ang_pred = params["k_w"] * (wr - wl) + params["b_w"]
    else:
        by_regime = params.get("by_regime", {})
        regime = classify_regime(wl, wr)

        k_v = np.full(wl.shape, params["k_v"], dtype=float)
        b_v = np.full(wl.shape, params["b_v"], dtype=float)
        k_w = np.full(wl.shape, params["k_w"], dtype=float)
        b_w = np.full(wl.shape, params["b_w"], dtype=float)

        for r in (TRANSLATING, ROTATING_IN_PLACE):
            if r not in by_regime:
                continue
            mask = regime == r
            k_v[mask] = by_regime[r]["k_v"]
            b_v[mask] = by_regime[r]["b_v"]
            k_w[mask] = by_regime[r]["k_w"]
            b_w[mask] = by_regime[r]["b_w"]
        # STATIONARY frames intentionally keep the global coefficients
        # (initial fill above) -- see docstring.

        lin_pred = k_v * (wl + wr) + b_v
        ang_pred = k_w * (wr - wl) + b_w

    out[:, :, idx[ODOM_LIN_X]] = data[:, :, idx[ODOM_LIN_X]] - lin_pred
    out[:, :, idx[ODOM_ANG_Z]] = data[:, :, idx[ODOM_ANG_Z]] - ang_pred
    return out


def residual_feature_names(cols):
    """Column names with the two kinematics-covered features marked, for
    readable graph/report output. Order and count are unchanged."""
    renamed = list(cols)
    idx = {c: i for i, c in enumerate(cols)}
    renamed[idx[ODOM_LIN_X]] = ODOM_LIN_X + "__kin_residual"
    renamed[idx[ODOM_ANG_Z]] = ODOM_ANG_Z + "__kin_residual"
    return renamed
