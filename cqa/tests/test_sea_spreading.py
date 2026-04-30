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
