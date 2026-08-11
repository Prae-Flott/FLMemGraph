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
"""
import numpy as np

WHEEL_LEFT = "wheel_vels_velocity_left"
WHEEL_RIGHT = "wheel_vels_velocity_right"
ODOM_LIN_X = "odom_odo_lintw_x"
ODOM_ANG_Z = "odom_odo_angtw_z"


def _lstsq_line(x, y):
    """Fit y = k*x + b by ordinary least squares. Returns (k, b, r2)."""
    A = np.stack([x, np.ones_like(x)], axis=1)
    (k, b), _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_hat = k * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(k), float(b), r2


def fit_kinematic_params(data, fit_idx, cols):
    """Fit k_v, b_v (linear velocity) and k_w, b_w (angular velocity) from
    the FIT split only (never calib/test/fault) to avoid leakage. `data` is
    the raw (unscaled) [N, T, F] array, `fit_idx` the normal-fit window
    indices, `cols` the feature name list."""
    idx = {c: i for i, c in enumerate(cols)}
    wl = data[fit_idx][:, :, idx[WHEEL_LEFT]].reshape(-1)
    wr = data[fit_idx][:, :, idx[WHEEL_RIGHT]].reshape(-1)
    lin = data[fit_idx][:, :, idx[ODOM_LIN_X]].reshape(-1)
    ang = data[fit_idx][:, :, idx[ODOM_ANG_Z]].reshape(-1)

    k_v, b_v, r2_lin = _lstsq_line(wl + wr, lin)
    k_w, b_w, r2_ang = _lstsq_line(wr - wl, ang)

    return {
        "k_v": k_v, "b_v": b_v, "r2_lin": r2_lin,
        "k_w": k_w, "b_w": b_w, "r2_ang": r2_ang,
    }


def apply_kinematic_residual(data, cols, params):
    """Return a COPY of `data` with `odom_odo_lintw_x` and
    `odom_odo_angtw_z` replaced by their kinematic residual (actual minus
    the differential-drive prediction from the two wheel velocities). Every
    other of the 71 columns is untouched. Applied identically to
    fit/calib/test_normal/fault windows using the SAME fit-derived
    params -- the model never sees fault or held-out data during fitting."""
    idx = {c: i for i, c in enumerate(cols)}
    out = data.copy()

    wl = data[:, :, idx[WHEEL_LEFT]]
    wr = data[:, :, idx[WHEEL_RIGHT]]

    lin_pred = params["k_v"] * (wl + wr) + params["b_v"]
    ang_pred = params["k_w"] * (wr - wl) + params["b_w"]

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
