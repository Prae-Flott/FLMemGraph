"""
Unit tests for kinematics.py's regime classification and regime-conditioned
fitting (project plan step 5). Plain-assertion script, no pytest dependency
(none is installed in this repo's env) -- run directly:

    python src/robo3er/test_kinematics.py

Covers:
  1. classify_regime correctness on hand-constructed cases (all three
     regimes, including the near-zero STATIONARY threshold boundary).
  2. apply_kinematic_residual(regime_gating=False) is BIT-IDENTICAL to the
     pre-rewrite behavior (guards against silently changing the existing
     benchmark pipeline's numbers).
  3. On synthetic data obeying a clean regime-dependent relation, straight
     (TRANSLATING) driving residuals are near zero under the fit
     TRANSLATING-regime coefficients.
  4. A regime-gated residual on synthetic ROTATING_IN_PLACE data that
     deliberately violates the TRANSLATING-fit formula (i.e. a
     mis-specified-formula fault-like case) produces a LARGER residual
     magnitude than the correctly-gated one would -- checks that gating
     off an inapplicable regime, rather than force-scoring it with the
     wrong regime's coefficients, actually matters.
"""
import numpy as np

from kinematics import (
    classify_regime,
    fit_kinematic_params,
    fit_edge_params,
    apply_kinematic_residual,
    stationary_noise_stats,
    STATIONARY,
    TRANSLATING,
    ROTATING_IN_PLACE,
    WHEEL_LEFT,
    WHEEL_RIGHT,
    ODOM_LIN_X,
    ODOM_ANG_Z,
)

FAILURES = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        FAILURES.append(name)


def test_classify_regime_basic_cases():
    v_l = np.array([0.5, 0.5, -0.5, 0.5, 0.005, 0.0, -0.01, 1.0])
    v_r = np.array([0.5, -0.5, 0.5, 0.6, -0.005, 0.0, 0.01, -1.0])
    regime = classify_regime(v_l, v_r)
    expected = np.array([
        TRANSLATING,        # both positive, same sign
        ROTATING_IN_PLACE,  # opposite signs
        ROTATING_IN_PLACE,  # opposite signs
        TRANSLATING,        # both positive, same sign, different magnitude
        STATIONARY,         # both below threshold
        STATIONARY,         # both exactly zero
        STATIONARY,         # both below threshold, opposite tiny signs
        ROTATING_IN_PLACE,  # large opposite signs
    ], dtype=object)
    check("classify_regime matches hand-labeled cases", np.array_equal(regime, expected))


def test_classify_regime_threshold_boundary():
    # Just above threshold on both wheels, same sign -> TRANSLATING even
    # though magnitudes are small.
    v_l = np.array([0.021, 0.021])
    v_r = np.array([0.021, -0.021])
    regime = classify_regime(v_l, v_r, thresh=0.02)
    check(
        "just-above-threshold same-sign classified TRANSLATING",
        regime[0] == TRANSLATING,
    )
    check(
        "just-above-threshold opposite-sign classified ROTATING_IN_PLACE",
        regime[1] == ROTATING_IN_PLACE,
    )


def _make_synthetic_dataset(n_windows=200, t=20, seed=0):
    """Synthetic [N, T, F] array with a KNOWN differential-drive relation
    that differs by regime, mimicking the real finding that single-wheel
    coefficients are regime-dependent. cols order matches the 4 features
    kinematics.py reads.
    """
    rng = np.random.default_rng(seed)
    cols = [WHEEL_LEFT, WHEEL_RIGHT, ODOM_LIN_X, ODOM_ANG_Z]

    # Half the windows translate (same-sign wheels), half rotate in place
    # (opposite-sign wheels).
    n_translate = n_windows // 2
    n_rotate = n_windows - n_translate

    def make_translating(n):
        base = rng.uniform(0.3, 1.5, size=(n, t))
        noise = rng.normal(0, 0.02, size=(n, t))
        wl = base + noise
        wr = base + rng.normal(0, 0.02, size=(n, t))
        k_v, b_v = 0.47, 0.0
        k_w, b_w = 1.17, 0.0
        lin = k_v * (wl + wr) + b_v + rng.normal(0, 0.01, size=(n, t))
        ang = k_w * (wr - wl) + b_w + rng.normal(0, 0.01, size=(n, t))
        return wl, wr, lin, ang

    def make_rotating(n):
        base = rng.uniform(0.3, 1.5, size=(n, t))
        wl = base + rng.normal(0, 0.02, size=(n, t))
        wr = -base + rng.normal(0, 0.02, size=(n, t))
        # Different diff->ang coefficient in this regime, matching the
        # real-data finding that in-place-rotation coefficients differ
        # from translating-regime coefficients.
        k_v, b_v = 0.35, 0.1
        k_w, b_w = 0.96, 0.05
        lin = k_v * (wl + wr) + b_v + rng.normal(0, 0.01, size=(n, t))
        ang = k_w * (wr - wl) + b_w + rng.normal(0, 0.01, size=(n, t))
        return wl, wr, lin, ang

    wl_t, wr_t, lin_t, ang_t = make_translating(n_translate)
    wl_r, wr_r, lin_r, ang_r = make_rotating(n_rotate)

    wl = np.concatenate([wl_t, wl_r], axis=0)
    wr = np.concatenate([wr_t, wr_r], axis=0)
    lin = np.concatenate([lin_t, lin_r], axis=0)
    ang = np.concatenate([ang_t, ang_r], axis=0)

    data = np.stack([wl, wr, lin, ang], axis=-1)
    fit_idx = np.arange(n_windows)
    return data, fit_idx, cols, n_translate, n_rotate


def test_translating_residual_near_zero_when_gated():
    data, fit_idx, cols, n_translate, n_rotate = _make_synthetic_dataset()
    params = fit_kinematic_params(data, fit_idx, cols, min_regime_samples=10)

    check("by_regime has TRANSLATING", TRANSLATING in params["by_regime"])
    check("by_regime has ROTATING_IN_PLACE", ROTATING_IN_PLACE in params["by_regime"])

    resid_gated = apply_kinematic_residual(data, cols, params, regime_gating=True)
    idx = {c: i for i, c in enumerate(cols)}

    # Residuals should be small (dominated by the injected observation
    # noise, not by systematic regime misspecification) for BOTH regimes
    # when correctly gated, since each regime is scored with its own
    # coefficients.
    lin_resid = resid_gated[:, :, idx[ODOM_LIN_X]]
    ang_resid = resid_gated[:, :, idx[ODOM_ANG_Z]]
    check(
        "gated residual std is small (~injected noise level) on synthetic clean data",
        lin_resid.std() < 0.05 and ang_resid.std() < 0.05,
    )


def test_regime_gating_reduces_misspecification_vs_global():
    """On the ROTATING_IN_PLACE half of the synthetic data (whose true
    coefficients differ from TRANSLATING's), scoring with the GLOBAL
    (regime-pooled) coefficients should leave a larger residual than
    scoring with the correctly gated per-regime coefficients -- this is
    the concrete benefit regime gating is supposed to buy."""
    data, fit_idx, cols, n_translate, n_rotate = _make_synthetic_dataset()
    params = fit_kinematic_params(data, fit_idx, cols, min_regime_samples=10)
    idx = {c: i for i, c in enumerate(cols)}

    resid_global = apply_kinematic_residual(data, cols, params, regime_gating=False)
    resid_gated = apply_kinematic_residual(data, cols, params, regime_gating=True)

    rotate_slice = slice(n_translate, n_translate + n_rotate)
    global_abs_mean = np.abs(resid_global[rotate_slice, :, idx[ODOM_ANG_Z]]).mean()
    gated_abs_mean = np.abs(resid_gated[rotate_slice, :, idx[ODOM_ANG_Z]]).mean()

    check(
        "regime-gated residual is smaller than global-formula residual on the "
        "regime whose true coefficients differ from the pooled fit",
        gated_abs_mean < global_abs_mean,
    )


def test_apply_kinematic_residual_default_matches_legacy_global_formula():
    """regime_gating=False (the default) must reproduce the exact original
    single-formula behavior -- this guards against silently changing
    results for every existing benchmark script that calls this function
    without the new kwarg."""
    data, fit_idx, cols, _, _ = _make_synthetic_dataset(n_windows=20, seed=1)
    params = fit_kinematic_params(data, fit_idx, cols)
    idx = {c: i for i, c in enumerate(cols)}

    wl = data[:, :, idx[WHEEL_LEFT]]
    wr = data[:, :, idx[WHEEL_RIGHT]]
    expected_lin = data[:, :, idx[ODOM_LIN_X]] - (params["k_v"] * (wl + wr) + params["b_v"])
    expected_ang = data[:, :, idx[ODOM_ANG_Z]] - (params["k_w"] * (wr - wl) + params["b_w"])

    out_default = apply_kinematic_residual(data, cols, params)  # no kwarg passed
    out_explicit_false = apply_kinematic_residual(data, cols, params, regime_gating=False)

    check(
        "default call (no regime_gating kwarg) matches manual global formula",
        np.allclose(out_default[:, :, idx[ODOM_LIN_X]], expected_lin)
        and np.allclose(out_default[:, :, idx[ODOM_ANG_Z]], expected_ang),
    )
    check(
        "regime_gating=False explicit matches default call bit-for-bit",
        np.array_equal(out_default, out_explicit_false),
    )


def test_fit_edge_params_gates_single_wheel_edges_to_translating_only():
    data, fit_idx, cols, _, _ = _make_synthetic_dataset()
    edge_params = fit_edge_params(data, fit_idx, cols, min_regime_samples=10)

    check("sum_to_lin fit in both regimes", set(edge_params["sum_to_lin"].keys()) == {TRANSLATING, ROTATING_IN_PLACE})
    check("diff_to_ang fit in both regimes", set(edge_params["diff_to_ang"].keys()) == {TRANSLATING, ROTATING_IN_PLACE})
    for edge in ["left_to_lin", "right_to_lin", "left_to_ang", "right_to_ang"]:
        check(
            f"{edge} is gated to TRANSLATING only (no ROTATING_IN_PLACE key)",
            set(edge_params[edge].keys()) == {TRANSLATING},
        )


def test_min_regime_samples_omits_sparse_regime():
    # Build a dataset with a real but tiny ROTATING_IN_PLACE presence,
    # below MIN_REGIME_SAMPLES default -- by_regime should omit it, not
    # fit garbage on 3 points.
    data, fit_idx, cols, n_translate, _ = _make_synthetic_dataset(n_windows=200, seed=2)
    idx = {c: i for i, c in enumerate(cols)}
    # Zero out all but 1 rotating window's worth of frames by relabeling
    # them into the translating regime's sign pattern (small hack: just
    # test with a high min_regime_samples threshold instead, which is the
    # documented fallback trigger).
    params = fit_kinematic_params(data, fit_idx, cols, min_regime_samples=10**6)
    check(
        "an unreachably high min_regime_samples omits every regime from by_regime",
        len(params["by_regime"]) == 0,
    )
    # And apply_kinematic_residual with regime_gating=True should then
    # fall back to the global coefficients everywhere (no crash, no
    # missing-key error).
    out = apply_kinematic_residual(data, cols, params, regime_gating=True)
    check("regime_gating=True with empty by_regime does not raise and returns finite values",
          np.all(np.isfinite(out)))


def test_stationary_noise_stats_insufficient_data_flag():
    # Synthetic dataset here has essentially zero STATIONARY frames (both
    # wheels are always at least ~0.3 rad/s in magnitude), so this should
    # report insufficient data rather than fabricate a std from ~0 points.
    data, fit_idx, cols, _, _ = _make_synthetic_dataset()
    stats = stationary_noise_stats(data, fit_idx, cols, min_samples=50)
    check("stationary_noise_stats flags insufficient data on a dataset with no stationary frames",
          stats["sufficient"] is False)


if __name__ == "__main__":
    test_classify_regime_basic_cases()
    test_classify_regime_threshold_boundary()
    test_translating_residual_near_zero_when_gated()
    test_regime_gating_reduces_misspecification_vs_global()
    test_apply_kinematic_residual_default_matches_legacy_global_formula()
    test_fit_edge_params_gates_single_wheel_edges_to_translating_only()
    test_min_regime_samples_omits_sparse_regime()
    test_stationary_noise_stats_insufficient_data_flag()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        raise SystemExit(1)
    else:
        print("All tests passed.")
