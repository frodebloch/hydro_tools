"""Tests for cqa.sea_spreading (cos-2s and wrapped-Gaussian)."""

from __future__ import annotations

import numpy as np
import pytest

from cqa import SeaSpreading, cos_2s_norm_const, spreading_quadrature


# ---------------------------------------------------------------------------
# Quadrature weights
# ---------------------------------------------------------------------------


def test_default_spreading_is_cos_2s_s15():
    s = SeaSpreading()
    assert s.kind == "cos_2s"
    assert s.s == 15.0
    assert s.n_dir == 31


def test_quadrature_weights_sum_to_one_cos_2s():
    angles, w = spreading_quadrature(SeaSpreading(kind="cos_2s", s=15.0, n_dir=31))
    assert angles.shape == (31,)
    assert w.shape == (31,)
    assert np.isclose(w.sum(), 1.0)


def test_quadrature_weights_sum_to_one_gaussian():
    angles, w = spreading_quadrature(
        SeaSpreading(kind="gaussian", sigma_deg=20.0, n_dir=31)
    )
    assert np.isclose(w.sum(), 1.0)


def test_long_crested_returns_single_sample():
    angles, w = spreading_quadrature(SeaSpreading.long_crested(), theta_bar_rad=0.7)
    assert angles.shape == (1,)
    assert np.isclose(angles[0], 0.7)
    assert np.isclose(w[0], 1.0)


def test_long_crested_zero_sigma_also_collapses():
    """Even with n_dir > 1, sigma_deg=0 should give a single sample."""
    angles, w = spreading_quadrature(
        SeaSpreading(kind="gaussian", sigma_deg=0.0, n_dir=31), theta_bar_rad=1.2
    )
    assert angles.shape == (1,)
    assert np.isclose(angles[0], 1.2)


def test_quadrature_centered_on_theta_bar():
    """Sample mean should be ~theta_bar for symmetric spreadings."""
    theta_bar = 0.4
    for spread in (
        SeaSpreading(kind="cos_2s", s=15.0, n_dir=51),
        SeaSpreading(kind="gaussian", sigma_deg=20.0, n_dir=51),
    ):
        angles, w = spreading_quadrature(spread, theta_bar_rad=theta_bar)
        mean = float(np.sum(w * angles))
        assert np.isclose(mean, theta_bar, atol=1e-9)


# ---------------------------------------------------------------------------
# cos-2s normalisation
# ---------------------------------------------------------------------------


def test_cos_2s_norm_constant_integrates_to_one():
    """Integral of D(phi; s) over [-pi, pi] must be 1."""
    for s in (1.0, 5.0, 15.0, 50.0):
        c = cos_2s_norm_const(s)
        phi = np.linspace(-np.pi, np.pi, 2001)
        D = c * np.cos(phi / 2.0) ** (2.0 * s)
        integral = float(np.trapezoid(D, phi))
        assert np.isclose(integral, 1.0, rtol=1e-4)


def test_cos_2s_narrows_as_s_increases():
    """Higher s => more peaked at phi=0; D(0;s2) > D(0;s1) for s2>s1."""
    s1, s2 = 5.0, 50.0
    assert cos_2s_norm_const(s2) > cos_2s_norm_const(s1)


# ---------------------------------------------------------------------------
# Effective spread sanity
# ---------------------------------------------------------------------------


def test_cos_2s_s15_effective_sigma():
    """Compute the second moment from our quadrature. For cos-2s with
    s=15 the analytic one-sigma is about 20.6 deg (closer to swell
    than wind-sea). DNV-RP-C205 Table 3-9 maps s=15 to wind-sea-ish
    spread; matching brucon's CSOV defaults is more important than the
    exact number."""
    spread = SeaSpreading(kind="cos_2s", s=15.0, n_dir=201)
    angles, w = spreading_quadrature(spread, theta_bar_rad=0.0)
    var = float(np.sum(w * angles ** 2))
    sigma_deg = float(np.degrees(np.sqrt(var)))
    assert 18.0 < sigma_deg < 24.0


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        spreading_quadrature(SeaSpreading(kind="bogus", n_dir=11))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# brucon-style cos^n spreading
# ---------------------------------------------------------------------------


def test_cos_n_classmethod_constructs_kind_cos_n():
    s = SeaSpreading.cos_n(2.0)
    assert s.kind == "cos_n"
    assert s.n == 2.0


def test_cos_n_quadrature_weights_sum_to_one():
    angles, w = spreading_quadrature(SeaSpreading.cos_n(2.0, n_dir=51))
    assert np.isclose(w.sum(), 1.0)
    # Support is +/- pi/2 (brucon convention).
    assert np.isclose(angles.min(), -np.pi / 2.0)
    assert np.isclose(angles.max(),  np.pi / 2.0)


def test_cos_n_n2_matches_cos2s_s4_in_one_sigma():
    """Gaussian-limit equivalence: cos^n is approximately cos_2s with
    s ~= 2 n in the narrow-spread limit. brucon's default n=2 thus
    approximates cqa cos_2s s=4 (broader than the cqa default s=15).

    Verify by comparing the second moments from both quadratures
    against each other, not against any external reference -- that
    locks the equivalence relation we documented in the docstring.
    """
    a_n, w_n = spreading_quadrature(SeaSpreading.cos_n(2.0, n_dir=201))
    var_n = float(np.sum(w_n * a_n ** 2))
    sigma_n_deg = float(np.degrees(np.sqrt(var_n)))

    a_s, w_s = spreading_quadrature(SeaSpreading(kind="cos_2s", s=4.0, n_dir=201))
    var_s = float(np.sum(w_s * a_s ** 2))
    sigma_s_deg = float(np.degrees(np.sqrt(var_s)))

    # Within 20% of each other (Gaussian-limit equivalence is asymptotic;
    # at n=2 the support of cos_n is +/- pi/2 while cos_2s is +/- pi,
    # so neither is in the strict narrow-spread regime where the
    # equivalence is tight; observed sigma_cos_n=2 ~= 32.5 deg vs
    # sigma_cos_2s_s4 ~= 38.1 deg, ~15% off).
    assert abs(sigma_n_deg - sigma_s_deg) / sigma_s_deg < 0.20


# ---------------------------------------------------------------------------
# Bretschneider / wave_elevation_psd dispatcher
# ---------------------------------------------------------------------------


def test_bretschneider_psd_equals_jonswap_gamma_one():
    """Bretschneider is the gamma=1 limit of the JONSWAP form -- the
    ``A_gamma = 1 - 0.287 ln(gamma)`` normalisation collapses to 1
    and ``gamma^r`` is identically 1."""
    from cqa.psd import bretschneider_psd, jonswap_psd
    omega = np.linspace(0.05, 4.0, 4000)
    Hs, Tp = 2.0, 8.0
    S_bret = bretschneider_psd(omega, Hs, Tp)
    S_jon1 = jonswap_psd(omega, Hs, Tp, gamma=1.0)
    # Same code path via the wrapper -- bit-for-bit identical.
    assert np.array_equal(S_bret, S_jon1)


def test_bretschneider_zero_moment_matches_Hs_squared_over_16():
    """For Bretschneider, m_0 = integral S(omega) d omega = Hs^2 / 16
    (DNV-RP-C205 Table 3-1)."""
    from cqa.psd import bretschneider_psd
    omega = np.linspace(0.05, 4.0, 8000)
    Hs, Tp = 3.0, 9.0
    m0 = float(np.trapezoid(bretschneider_psd(omega, Hs, Tp), omega))
    # ~1.5% slack to account for low-omega truncation at omega = 0.05.
    assert abs(m0 - Hs ** 2 / 16.0) / (Hs ** 2 / 16.0) < 0.02


def test_wave_elevation_psd_dispatcher_matches_named_calls():
    from cqa.psd import wave_elevation_psd, jonswap_psd, bretschneider_psd
    omega = np.linspace(0.05, 4.0, 1024)
    Hs, Tp = 2.5, 7.5
    assert np.array_equal(
        wave_elevation_psd(omega, Hs, Tp, kind="bretschneider"),
        bretschneider_psd(omega, Hs, Tp),
    )
    assert np.array_equal(
        wave_elevation_psd(omega, Hs, Tp, kind="jonswap", gamma=3.3),
        jonswap_psd(omega, Hs, Tp, gamma=3.3),
    )
    with pytest.raises(ValueError):
        wave_elevation_psd(omega, Hs, Tp, kind="bogus")  # type: ignore[arg-type]
