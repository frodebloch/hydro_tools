"""Tests for the 1-DOF roll vessel."""
from __future__ import annotations

import numpy as np
import pytest

from roll_reduction_tanks.vessel import RollVessel, RollVesselConfig


def _csov_like_config(GM=1.787, damping_ratio=0.05) -> RollVesselConfig:
    """A CSOV-scale vessel (numbers from data/csov/README.md)."""
    I44 = 8.929e8
    a44 = 0.20 * I44
    rho = 1025.0
    g = 9.81
    displacement = 10848.0
    c44 = rho * g * displacement * GM
    b44 = damping_ratio * 2.0 * np.sqrt(c44 * (I44 + a44))
    return RollVesselConfig(
        I44=I44, a44=a44, b44_lin=b44, GM=GM, displacement=displacement,
        rho=rho, g=g,
    )


def test_natural_frequency_and_period_consistent():
    cfg = _csov_like_config()
    assert cfg.natural_frequency == pytest.approx(np.sqrt(cfg.c44 / (cfg.I44 + cfg.a44)))
    assert cfg.natural_period == pytest.approx(2 * np.pi / cfg.natural_frequency)


def test_csov_natural_period_in_expected_range():
    """CSOV at GM=1.787 m, a44=0.2*I44 gives T_roll ~ 15 s."""
    cfg = _csov_like_config()
    assert 13.0 < cfg.natural_period < 17.0


def test_zero_excitation_at_rest_stays_at_rest():
    cfg = _csov_like_config()
    v = RollVessel(cfg)
    out = v.integrate(lambda t: 0.0, dt=0.05, t_end=20.0)
    assert np.max(np.abs(out["phi"])) < 1e-12
    assert np.max(np.abs(out["phi_dot"])) < 1e-12


def test_free_decay_period_matches_analytic():
    """Damped natural period = T_n / sqrt(1 - zeta^2)."""
    zeta = 0.03
    cfg = _csov_like_config(damping_ratio=zeta)
    v = RollVessel(cfg, phi0=np.deg2rad(5.0))
    dt = 0.02
    t_end = 5.0 * cfg.natural_period
    out = v.integrate(lambda t: 0.0, dt=dt, t_end=t_end)

    # Find zero crossings (positive-going) to estimate period.
    phi = out["phi"]
    t = out["t"]
    crossings = []
    for i in range(1, len(phi)):
        if phi[i - 1] < 0 and phi[i] >= 0:
            # Linear interpolation
            frac = -phi[i - 1] / (phi[i] - phi[i - 1])
            crossings.append(t[i - 1] + frac * dt)
    assert len(crossings) >= 3
    periods = np.diff(crossings)
    measured = float(np.mean(periods))
    expected = cfg.natural_period / np.sqrt(1.0 - zeta**2)
    assert measured == pytest.approx(expected, rel=2e-3)


def test_free_decay_log_decrement_matches_analytic():
    """Logarithmic decrement: delta = 2*pi*zeta / sqrt(1 - zeta^2)."""
    zeta = 0.05
    cfg = _csov_like_config(damping_ratio=zeta)
    v = RollVessel(cfg, phi0=np.deg2rad(5.0))
    dt = 0.02
    t_end = 5.0 * cfg.natural_period
    out = v.integrate(lambda t: 0.0, dt=dt, t_end=t_end)
    phi = out["phi"]

    # Find local maxima.
    peaks = []
    for i in range(1, len(phi) - 1):
        if phi[i - 1] < phi[i] > phi[i + 1] and phi[i] > 0:
            peaks.append(phi[i])
    assert len(peaks) >= 3
    log_decrements = np.log(np.array(peaks[:-1]) / np.array(peaks[1:]))
    measured = float(np.mean(log_decrements))
    expected = 2 * np.pi * zeta / np.sqrt(1 - zeta**2)
    assert measured == pytest.approx(expected, rel=5e-2)


def test_steady_state_amplitude_matches_linear_rao():
    """For a regular external moment M0*cos(w t), steady-state |phi| = |M0/H(iw)|.

    H(iw) = -(I+a) w^2 + i b w + c
    """
    cfg = _csov_like_config(damping_ratio=0.05)
    w = cfg.natural_frequency * 1.3  # Off-resonance.
    M0 = 1.0e7  # N*m
    v = RollVessel(cfg)
    dt = 0.02
    # Settle long enough that transients decay below 1%.
    t_settle = 8 * cfg.natural_period / cfg.damping_ratio
    out = v.integrate(lambda t: M0 * np.cos(w * t), dt=dt, t_end=t_settle + 5 * 2 * np.pi / w)

    # Take last 4 periods to estimate amplitude
    n_window = int(4 * 2 * np.pi / w / dt)
    phi_window = out["phi"][-n_window:]
    measured_amp = (phi_window.max() - phi_window.min()) / 2

    H = -(cfg.I44 + cfg.a44) * w**2 + 1j * cfg.b44_lin * w + cfg.c44
    expected_amp = M0 / abs(H)

    assert measured_amp == pytest.approx(expected_amp, rel=5e-3)
