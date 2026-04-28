"""Tests for waves.py: round-trip and GM-decoupling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from roll_reduction_tanks.pdstrip_io import load_csov
from roll_reduction_tanks.vessel import RollVessel, RollVesselConfig
from roll_reduction_tanks.waves import (
    RegularWave,
    encounter_frequency,
    roll_moment_complex_amplitude,
    roll_moment_from_pdstrip,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "csov"


@pytest.fixture(scope="module")
def csov_rao():
    return load_csov(
        DATA / "csov_pdstrip.dat",
        DATA / "csov_pdstrip.inp",
        GM_pdstrip=1.787,
    )


def _vessel_at_gm(rao, GM, b44_lin=None):
    a44 = rao.a44_assumed
    b44_lin = rao.b44_assumed if b44_lin is None else b44_lin
    return RollVesselConfig(
        I44=rao.I44, a44=a44, b44_lin=b44_lin,
        GM=GM, displacement=rao.displacement,
        rho=rao.rho, g=rao.g,
    )


def test_encounter_frequency_zero_speed_unchanged():
    assert encounter_frequency(0.6, -90.0, 0.0) == pytest.approx(0.6)
    assert encounter_frequency(0.6, 180.0, 0.0) == pytest.approx(0.6)


def test_encounter_frequency_head_seas_increases():
    """Head seas (beta=180) at positive speed: omega_e > omega."""
    omega = 0.6
    omega_e = encounter_frequency(omega, 180.0, 5.0)
    assert omega_e > omega


def test_encounter_frequency_following_seas_decreases():
    omega = 0.6
    omega_e = encounter_frequency(omega, 0.0, 5.0)
    assert omega_e < omega


def test_complex_amplitude_consistent_with_callable(csov_rao):
    """The closure returned by roll_moment_from_pdstrip should agree with
    the complex amplitude form at every time."""
    wave = RegularWave(omega=0.6, heading_deg=-90.0, amplitude=2.0, speed=0.0)
    M_func = roll_moment_from_pdstrip(wave, csov_rao)
    Me, omega_e = roll_moment_complex_amplitude(wave, csov_rao)
    for t in np.linspace(0.0, 30.0, 17):
        expected = (Me * np.exp(1j * omega_e * t)).real
        assert M_func(t) == pytest.approx(expected, rel=1e-12, abs=1e-9)


def test_roundtrip_at_pdstrip_gm(csov_rao):
    """Drive a vessel-only model (no tank) at the same GM and damping as
    pdstrip used; the steady-state roll amplitude must equal the pdstrip
    RAO magnitude (to within numerical / settling tolerance)."""
    wave = RegularWave(omega=0.5, heading_deg=-90.0, amplitude=1.0, speed=0.0)
    cfg = _vessel_at_gm(csov_rao, GM=csov_rao.GM_pdstrip)

    M_func = roll_moment_from_pdstrip(wave, csov_rao)
    v = RollVessel(cfg)
    dt = 0.05
    # Long enough for transients to decay; ~10 time-constants = 10/(zeta*wn)
    zeta = cfg.damping_ratio
    t_settle = 10.0 / (zeta * cfg.natural_frequency)
    out = v.integrate(M_func, dt=dt, t_end=t_settle + 6 * 2 * np.pi / wave.omega)

    n_window = int(4 * 2 * np.pi / wave.omega / dt)
    phi_window = out["phi"][-n_window:]
    measured_amp = (phi_window.max() - phi_window.min()) / 2

    # Expected: |Phi_pdstrip| at this (omega, beta, U), times wave amplitude
    expected_amp = abs(csov_rao.get_roll_rao(wave.omega, wave.heading_deg, wave.speed)) * wave.amplitude
    assert measured_amp == pytest.approx(expected_amp, rel=2e-3)


def test_gm_decoupling(csov_rao):
    """Same wave-exciting moment, different simulator GM → response amplitude
    must match the closed-form linear RAO at the new c44.

    This is the core decoupling property: a44, b44 cancel in the back-out
    and only c44 changes.
    """
    wave = RegularWave(omega=0.5, heading_deg=-90.0, amplitude=1.0, speed=0.0)
    Me, omega_e = roll_moment_complex_amplitude(wave, csov_rao)
    M_func = roll_moment_from_pdstrip(wave, csov_rao)

    GM_new = 0.8
    cfg_new = _vessel_at_gm(csov_rao, GM=GM_new)
    v = RollVessel(cfg_new)

    dt = 0.05
    zeta = cfg_new.damping_ratio
    t_settle = 10.0 / max(zeta * cfg_new.natural_frequency, 1e-3)
    out = v.integrate(M_func, dt=dt, t_end=t_settle + 6 * 2 * np.pi / omega_e)

    n_window = int(4 * 2 * np.pi / omega_e / dt)
    phi_window = out["phi"][-n_window:]
    measured_amp = (phi_window.max() - phi_window.min()) / 2

    # Closed-form: |Phi_new| = |Me| / |H_new|
    H_new = -(cfg_new.I44 + cfg_new.a44) * omega_e**2 + 1j * cfg_new.b44_lin * omega_e + cfg_new.c44
    expected_amp = abs(Me) / abs(H_new)
    assert measured_amp == pytest.approx(expected_amp, rel=5e-3)


def test_back_out_self_consistency_off_grid(csov_rao):
    """Round-trip should also work at a frequency that is *between* grid
    points (exercising the trilinear interpolation in the loader)."""
    omega_a = csov_rao.omega[10]
    omega_b = csov_rao.omega[11]
    omega_mid = 0.5 * (omega_a + omega_b)
    wave = RegularWave(omega=omega_mid, heading_deg=-90.0, amplitude=1.0, speed=0.0)

    cfg = _vessel_at_gm(csov_rao, GM=csov_rao.GM_pdstrip)
    M_func = roll_moment_from_pdstrip(wave, csov_rao)
    v = RollVessel(cfg)
    dt = 0.05
    zeta = cfg.damping_ratio
    t_settle = 10.0 / (zeta * cfg.natural_frequency)
    out = v.integrate(M_func, dt=dt, t_end=t_settle + 6 * 2 * np.pi / wave.omega)

    n_window = int(4 * 2 * np.pi / wave.omega / dt)
    phi_window = out["phi"][-n_window:]
    measured_amp = (phi_window.max() - phi_window.min()) / 2
    expected_amp = abs(csov_rao.get_roll_rao(wave.omega, wave.heading_deg, wave.speed)) * wave.amplitude
    assert measured_amp == pytest.approx(expected_amp, rel=3e-3)
