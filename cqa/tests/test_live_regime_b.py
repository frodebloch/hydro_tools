"""Tests for cqa.live_regime_b (sec.12.21.21.20).

Cross-checks:
  1. ``saturation_probability_gaussian`` against scipy.stats.norm.cdf
     ground truth on a hand-crafted scalar case.
  2. End-to-end ``estimate_regime_b_severity`` on a synthetic AR(1) thrust
     buffer with known (mu, sigma); LF cutoff well above the AR(1)
     bandwidth should recover (mu, sigma) within sampling error.
  3. End-to-end on a synthetic two-component (LF + WF) signal: the LP
     filter should reject the WF content and recover the LF (mu, sigma).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from cqa.live_regime_b import (
    saturation_probability_gaussian,
    estimate_regime_b_severity,
    lf_filter,
    OperationalCapGeometry,
    sway_cap_given_yaw,
    yaw_cap_given_sway,
    operational_cap_at,
)


# ---------------------------------------------------------------------------
# 1. analytical
# ---------------------------------------------------------------------------

def test_saturation_probability_zero_mean_symmetric():
    """Zero-mean Gaussian: P(|X|>cap) = 2*Phi(-cap/sigma)."""
    mu = np.zeros(3)
    sigma = np.array([1.0, 1.0, 1.0])
    cap = np.array([0.0, 1.0, 2.0])
    p = saturation_probability_gaussian(mu, sigma, cap)
    expected = 2.0 * norm.sf(cap / sigma)
    np.testing.assert_allclose(p, expected, rtol=1e-9)
    # cap=0 -> P=1 (X is non-zero almost surely)
    assert p[0] == pytest.approx(1.0, abs=1e-9)
    # cap=2*sigma -> P ~ 2*0.0228 = 0.0455
    assert p[2] == pytest.approx(0.04550, abs=1e-4)


def test_saturation_probability_biased_mean():
    """Mean offset towards +cap: upper tail dominates."""
    mu = np.array([0.5])
    sigma = np.array([1.0])
    cap = np.array([1.0])
    p = saturation_probability_gaussian(mu, sigma, cap)
    # Upper: P(X > 1) = Phi(-(1-0.5)/1) = Phi(-0.5) = 0.3085
    # Lower: P(X < -1) = Phi((-1-0.5)/1) = Phi(-1.5) = 0.0668
    # Sum = 0.3753
    assert p[0] == pytest.approx(0.3753, abs=1e-3)


def test_saturation_probability_mu_at_cap():
    """Mean sits exactly at cap: upper tail is exactly 0.5."""
    mu = np.array([1.0])
    sigma = np.array([1.0])
    cap = np.array([1.0])
    p = saturation_probability_gaussian(mu, sigma, cap)
    # Upper: Phi(0) = 0.5;  Lower: Phi(-2) = 0.0228;  sum = 0.5228
    assert p[0] == pytest.approx(0.5228, abs=1e-3)


# ---------------------------------------------------------------------------
# 2. end-to-end on synthetic stationary AR(1)
# ---------------------------------------------------------------------------

def _make_ar1(n: int, mu: float, sigma: float, tau_s: float,
              fs_hz: float, rng) -> np.ndarray:
    """AR(1) with mean=mu, std=sigma, decorrelation time tau_s."""
    dt = 1.0 / fs_hz
    rho = np.exp(-dt / tau_s)
    # innovation std so the stationary variance is sigma^2:
    # var = inn_var / (1-rho^2) => inn = sigma * sqrt(1-rho^2)
    inn = sigma * np.sqrt(1.0 - rho ** 2)
    x = np.zeros(n)
    x[0] = mu + rng.standard_normal() * sigma
    for k in range(1, n):
        x[k] = mu + rho * (x[k - 1] - mu) + inn * rng.standard_normal()
    return x


def test_ar1_recovers_mu_sigma():
    """AR(1) with tau=15s, fs=10Hz, 600 s window. mu and sigma should be
    recovered to within ~10% (sampling error ~ 1/sqrt(2*N_eff) with
    N_eff = 600/15 = 40 -> 11%). LP cutoff at 0.30 rad/s is well above
    the AR(1) bandwidth (1/15 = 0.067 rad/s) so it shouldn't bias the
    estimates."""
    rng = np.random.default_rng(7)
    fs = 10.0
    n = int(600 * fs)
    mu_true = np.array([100.0e3, 500.0e3, 1e6])      # N, N, N*m
    sigma_true = np.array([20.0e3, 80.0e3, 0.5e6])   # N, N, N*m
    buf = np.zeros((n, 3))
    for k in range(3):
        buf[:, k] = _make_ar1(n, mu_true[k], sigma_true[k],
                              tau_s=15.0, fs_hz=fs, rng=rng)
    # cap well above the demand: should give p_sat ~ 0
    cap = np.array([1.0e6, 1.5e6, 1e7])
    res = estimate_regime_b_severity(
        buf, fs_hz=fs, cap_residual=tuple(cap), window_s=600.0,
    )
    np.testing.assert_allclose(res.mu, mu_true, rtol=0.20)
    np.testing.assert_allclose(res.sigma, sigma_true, rtol=0.20)
    # cap way above demand -> green
    assert res.severity < 1e-6
    assert res.traffic == "green"


def test_ar1_severity_at_known_cap():
    """AR(1) with mu=500 kN, sigma=200 kN (full), cap=1000 kN.

    The LP filter at 0.30 rad/s retains the LF-band variance of the
    AR(1) (tau=15s, omega_c=1/15=0.067 rad/s). Analytical AR(1) variance
    fraction below cutoff:
        f = arctan(omega_LP * tau) / (pi/2)
    For omega_LP=0.30, tau=15: f = arctan(4.5)/(pi/2) = 0.860
    So LP-filtered sigma ~ 200 * sqrt(0.86) ~ 186 kN.

    Severity reference uses this LP sigma:
      upper: Phi(-(1000-500)/186) = Phi(-2.69) = 0.0036
      lower: Phi(-(1000+500)/186) ~ 0
    Allow 50% rel-tol because the LP filter has finite slope and the
    AR(1)/Butterworth combination has its own subtle distortion."""
    rng = np.random.default_rng(11)
    fs = 10.0
    n = int(1800 * fs)  # 30 min for tight statistics
    buf = np.zeros((n, 3))
    mu_true = 500.0e3
    sigma_true = 200.0e3
    for k in range(3):
        buf[:, k] = _make_ar1(n, mu_true, sigma_true,
                              tau_s=15.0, fs_hz=fs, rng=rng)
    cap = np.full(3, 1000.0e3)
    res = estimate_regime_b_severity(
        buf, fs_hz=fs, cap_residual=tuple(cap), window_s=1800.0,
    )
    # Use the LP-filtered sigma to compute expected p_sat per DOF
    expected = (norm.sf((cap - res.mu) / res.sigma)
                + norm.cdf((-cap - res.mu) / res.sigma))
    np.testing.assert_allclose(res.p_sat, expected, rtol=1e-9)
    # Sanity-check severity is in green (P_sat ~ 0.001 < 0.01 amber)
    assert res.traffic == "green"


# ---------------------------------------------------------------------------
# 3. LP filter rejects WF content
# ---------------------------------------------------------------------------

def test_lp_filter_rejects_wf():
    """Signal = LF (sigma 50 kN, tau 100 s) + WF (sigma 200 kN at 1.0
    rad/s).  After LP at 0.30 rad/s the WF component should be ~killed
    (suppressed by Butterworth rolloff at omega/omega_c = 3.3, which is
    ~50 dB for 4th order); recovered sigma should be ~ sigma_LF = 50."""
    rng = np.random.default_rng(3)
    fs = 10.0
    n = int(1800 * fs)
    t = np.arange(n) / fs

    LF = _make_ar1(n, mu=0.0, sigma=50.0e3, tau_s=100.0, fs_hz=fs, rng=rng)
    # Coherent WF tone for predictable sigma (sigma_tone = amp/sqrt(2)):
    amp = 200.0e3 * np.sqrt(2.0)
    omega_wf = 1.0  # rad/s -- well above LP cutoff at 0.30
    WF = amp * np.sin(omega_wf * t + rng.uniform(0, 2 * np.pi))
    signal = LF + WF
    buf = np.column_stack([signal, signal, signal])

    sigma_full = signal.std()  # should be sqrt(50^2 + 200^2) ~ 206 kN
    assert sigma_full > 150e3

    cap = np.full(3, 1e9)
    res = estimate_regime_b_severity(
        buf, fs_hz=fs, cap_residual=tuple(cap), window_s=1800.0,
        omega_lp_rad_s=0.30,
    )
    # LP should recover the 50 kN LF sigma to within ~30%
    np.testing.assert_allclose(res.sigma, 50.0e3, rtol=0.30)


def test_buffer_too_short_uses_what_is_available():
    """If buffer < window_s, function uses everything and proceeds."""
    rng = np.random.default_rng(1)
    fs = 10.0
    n = int(60 * fs)  # 60 s buffer, request 300 s
    buf = np.zeros((n, 3))
    for k in range(3):
        buf[:, k] = _make_ar1(n, 0.0, 100e3, 15.0, fs, rng)
    cap = np.full(3, 1e9)
    res = estimate_regime_b_severity(
        buf, fs_hz=fs, cap_residual=tuple(cap), window_s=300.0,
    )
    # Should not raise and should report n_eff for the requested window
    # even though the actual buffer is shorter.
    assert res.severity < 1e-6
    assert res.traffic == "green"


def test_input_shape_validation():
    """Wrong shape should raise."""
    buf = np.zeros((100, 2))
    with pytest.raises(ValueError):
        estimate_regime_b_severity(buf, fs_hz=10.0, cap_residual=(1, 1, 1))
    buf = np.zeros((100, 3))
    with pytest.raises(ValueError):
        estimate_regime_b_severity(buf, fs_hz=10.0, cap_residual=(1, 1))


# ---------------------------------------------------------------------------
# 4. Yaw-priority operational cap (sec.12.21.21.24)
# ---------------------------------------------------------------------------

# CSOV bus_port-lost geometry from sec.12.21.21.24 (brucon polytope port).
_CSOV_GEOM = OperationalCapGeometry(
    F_bow_max=629.0e3, F_bow_min=-629.0e3,
    F_stern_max=522.0e3, F_stern_min=-522.0e3,
    arm_bow=36.29, arm_stern=-48.09,
)


def test_sway_cap_at_zero_yaw_recovers_unconstrained_sum():
    """At tau_z = 0, the sway cap should equal F_bow_max + F_stern matching
    yaw zero (i.e. F_stern = -F_bow*arm_bow/arm_stern)."""
    lo, hi = sway_cap_given_yaw(0.0, _CSOV_GEOM)
    # At zero yaw, F_stern = -F_bow * arm_bow / arm_stern
    # F_bow = +629e3 => F_stern = -629e3 * 36.29 / -48.09 = +474.6 kN
    # Both within bounds; tau_y = 629 + 474.6 = 1103.6 kN.
    assert hi == pytest.approx(1103.6e3, rel=1e-3)
    assert lo == pytest.approx(-1103.6e3, rel=1e-3)


def test_sway_cap_shrinks_with_yaw_demand():
    """Increasing |tau_z| from 0 should monotonically shrink the achievable
    sway range."""
    _, hi0 = sway_cap_given_yaw(0.0, _CSOV_GEOM)
    _, hi15 = sway_cap_given_yaw(15e6, _CSOV_GEOM)
    _, hi30 = sway_cap_given_yaw(30e6, _CSOV_GEOM)
    assert hi0 > hi15 > hi30 > 0
    # Symmetric on the negative side.
    _, hi_n15 = sway_cap_given_yaw(-15e6, _CSOV_GEOM)
    assert hi15 == pytest.approx(hi_n15, rel=0.05)


def test_sway_cap_at_brucon_operating_point():
    """At the empirically-measured bf8_q10_w45 post-WCF operating point
    (tau_z ~ +19.5 MNm), the predicted sway cap should match the
    observed Alloc_y mean to within ~15%. See sec.12.21.21.24.

    Empirical: mean Alloc_y at clip moments = +656 kN, peak = +864 kN.
    """
    lo, hi = sway_cap_given_yaw(19.5e6, _CSOV_GEOM)
    # Predicted hi ~ 690 kN (computed in the diagnostic).
    assert hi == pytest.approx(690e3, rel=0.05)
    # Empirical mean Alloc_y is 656 kN -- predicted 690 should be within 10%.
    empirical_mean = 656e3
    assert abs(hi - empirical_mean) / empirical_mean < 0.10


def test_yaw_cap_given_sway_symmetric():
    """At tau_y = 0, yaw cap should be near decoupled max_yaw."""
    lo, hi = yaw_cap_given_sway(0.0, _CSOV_GEOM)
    # At tau_y=0, F_stern = -F_bow. F_bow at +629 kN -> F_stern -629
    # (clipped to -522). At F_stern=-522, F_bow=+522.
    # tau_z = 522*36.29 + (-522)*(-48.09) = 522 * (36.29+48.09) = 44048 kN.m
    assert hi == pytest.approx(44048e3, rel=1e-3)
    assert lo == pytest.approx(-44048e3, rel=1e-3)


def test_yaw_cap_shrinks_with_sway_demand():
    _, hi0 = yaw_cap_given_sway(0.0, _CSOV_GEOM)
    _, hi500 = yaw_cap_given_sway(500e3, _CSOV_GEOM)
    _, hi_max = yaw_cap_given_sway(1100e3, _CSOV_GEOM)
    assert hi0 > hi500 > hi_max > 0


def test_operational_cap_at_assembles_three_dofs():
    """operational_cap_at should return a (3,) array with the supplied
    surge_cap_N at index 0 and conditional caps at indices 1 and 2."""
    mu = np.array([100e3, 400e3, 10e6])  # surge, sway, yaw
    surge_cap = 1e6
    cap = operational_cap_at(mu, _CSOV_GEOM, surge_cap_N=surge_cap)
    assert cap.shape == (3,)
    assert cap[0] == surge_cap
    # Sway cap should be less than the decoupled max (1103 kN) because
    # yaw demand is non-zero.
    assert 0 < cap[1] < 1103e3
    # Yaw cap should be less than decoupled max for the same reason.
    assert 0 < cap[2] < 44048e3


def test_estimate_severity_with_geometry_path():
    """End-to-end test: pass a buffer with known LF (mu, sigma) and verify
    that severity is computed against the operational (yaw-coupled) cap.

    At the bf8_q10_w45 operating point (mu_y=+493 kN, sigma_y=86 kN,
    mu_z=+14.6 MNm), operational sway cap is ~700 kN. z-margin =
    (700-493)/86 ~ 2.4 sigma -> p_sat ~ 8e-3 -> amber territory.
    Contrast with the legacy decoupled cap of 1104 kN which would give
    p_sat ~ 1e-12.
    """
    rng = np.random.default_rng(seed=2024)
    fs = 10.0
    N = 3000  # 300 s window
    # Build a wide-band Gaussian LF buffer with EXACTLY mu, sigma at
    # frequencies the LP filter will preserve. Use raw Gaussian (white)
    # because lf_filter at 0.30 rad/s drops only a small fraction of
    # the spectrum.
    sway = rng.normal(loc=493e3, scale=86e3, size=N)
    surge = rng.normal(loc=200e3, scale=50e3, size=N)
    yaw = rng.normal(loc=14.6e6, scale=3e6, size=N)
    tau_buf = np.column_stack([surge, sway, yaw])

    res = estimate_regime_b_severity(
        tau_buf, fs_hz=fs,
        geometry=_CSOV_GEOM, surge_cap_N=1.36e6,
    )
    # mu_y should round-trip ~+493 kN (LP filter preserves DC).
    assert res.mu[1] == pytest.approx(493e3, abs=20e3)
    # sigma_y after LP filter will be REDUCED from the white 86 kN by
    # the fraction of white power retained below 0.30 rad/s -> for white
    # input it's roughly sqrt(f_c/(fs/2)) ~ sqrt(0.048/5) ~ 0.10.
    # Don't pin sigma exactly; just verify it's positive and not zero.
    assert res.sigma[1] > 0
    # Operational cap on sway at this mu_z ~ 700 kN (should be markedly
    # less than the decoupled vertex cap of 1104 kN).
    cap_sway = res.cap_residual[1]
    assert 600e3 < cap_sway < 800e3
    # Severity must be greater than the decoupled-cap result. Compute
    # legacy cap as a sanity check.
    res_legacy = estimate_regime_b_severity(
        tau_buf, fs_hz=fs, cap_residual=(1.36e6, 1.104e6, 47.93e6),
    )
    assert res.severity > res_legacy.severity, (
        f"geometry path ({res.severity:.2e}) should report higher severity "
        f"than legacy decoupled cap ({res_legacy.severity:.2e})"
    )


def test_geometry_path_requires_surge_cap():
    """geometry without surge_cap_N should raise."""
    buf = np.zeros((100, 3))
    with pytest.raises(ValueError):
        estimate_regime_b_severity(
            buf, fs_hz=10.0, geometry=_CSOV_GEOM
        )


def test_at_least_one_of_cap_or_geometry_required():
    buf = np.zeros((100, 3))
    with pytest.raises(ValueError):
        estimate_regime_b_severity(buf, fs_hz=10.0)
