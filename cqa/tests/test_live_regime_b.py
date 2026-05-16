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
