"""Tests for the Luenberger wave-moment observer and the
ResonatorObserverRDTController that uses it.
"""
from __future__ import annotations

import numpy as np
import pytest

from roll_reduction_tanks.controllers.luenberger_wave_observer import (
    LuenbergerWaveObserver,
    LuenbergerWaveObserverConfig,
)
from roll_reduction_tanks.coupling import CoupledSystem
from roll_reduction_tanks.simulation import run_simulation
from roll_reduction_tanks.tanks.utube_open import OpenUtubeConfig, OpenUtubeTank
from roll_reduction_tanks.tanks.utube_rdt import (
    InverseDynamicsRDTController,
    RDTUtubeConfig,
    RDTUtubeTank,
    ResonatorObserverRDTController,
    StateFeedbackRDTController,
)
from roll_reduction_tanks.vessel import RollVessel, RollVesselConfig


# ---------------------------------------------------------- fixtures


def _csov_vessel():
    return RollVessel(RollVesselConfig(
        I44=2.6e8, a44=4.2e7, b44_lin=2.0e7,
        GM=3.0, displacement=10848.0, rho=1025.0, g=9.81,
    ))


def _utube_geom_kw():
    return dict(
        duct_below_waterline=6.5,
        undisturbed_fluid_height=2.5,
        utube_duct_height=0.6,
        resevoir_duct_width=2.0,
        utube_duct_width=16.0,
        tank_thickness=5.0,
        tank_to_xcog=0.0,
        tank_wall_friction_coef=0.05,
        tank_height=5.0,
    )


def _vessel_params_for_observer():
    """Return (I_total, b44, c44) consistent with _csov_vessel()."""
    v = _csov_vessel()
    return v.config.total_inertia, v.config.b44_lin, v.config.c44


# ====================================================================
# Observer tests (no tank, just the linear augmented plant)
# ====================================================================


def _simulate_bare_vessel_phi(omega: float, M_amp: float, dt: float, T: float):
    """Numerically integrate the bare vessel under sinusoidal M_wave;
    return (t, phi, phi_dot, M_wave) arrays."""
    I, b, c = _vessel_params_for_observer()
    n = int(round(T / dt)) + 1
    t = np.arange(n) * dt
    M_wave = M_amp * np.sin(omega * t)
    phi = np.zeros(n)
    phi_dot = np.zeros(n)

    def deriv(state, M):
        ph, pd = state
        return np.array([pd, (M - b * pd - c * ph) / I])

    x = np.zeros(2)
    for k in range(n - 1):
        Mk = M_wave[k]
        Mk2 = M_amp * np.sin(omega * (t[k] + 0.5 * dt))
        Mk3 = M_wave[k + 1]
        k1 = deriv(x, Mk)
        k2 = deriv(x + 0.5 * dt * k1, Mk2)
        k3 = deriv(x + 0.5 * dt * k2, Mk2)
        k4 = deriv(x + dt * k3, Mk3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        phi[k + 1] = x[0]
        phi_dot[k + 1] = x[1]
    return t, phi, phi_dot, M_wave


def test_observer_constructs_and_places_all_poles():
    """Default pole placement must succeed and place all 5 poles."""
    I, b, c = _vessel_params_for_observer()
    cfg = LuenbergerWaveObserverConfig(
        I_total=I, b44=b, c44=c, omega_e=2 * np.pi / 8.5,
    )
    obs = LuenbergerWaveObserver(cfg)
    assert obs.L.shape == (5, 1)
    assert len(obs.placed_poles) == 5
    # All closed-loop poles strictly stable
    assert all(p.real < 0.0 for p in obs.placed_poles)


def test_observer_tracks_M_wave_at_resonator_frequency():
    """Drive the bare vessel with sinusoidal M_wave at omega_e = omega.
    The observer's M_wave_hat must converge in amplitude and phase to
    the true sinusoid within a few wave periods."""
    omega = 2 * np.pi / 9.0
    M_amp = 5.0e6
    dt = 0.05
    T = 200.0
    t, phi, phi_dot, M_true = _simulate_bare_vessel_phi(omega, M_amp, dt, T)

    I, b, c = _vessel_params_for_observer()
    cfg = LuenbergerWaveObserverConfig(
        I_total=I, b44=b, c44=c, omega_e=omega,
    )
    obs = LuenbergerWaveObserver(cfg)

    M_hat = np.zeros_like(t)
    for k in range(len(t)):
        # No tank in this test: M_tank_known = 0
        obs.update(phi[k], M_tank_known=0.0, dt=dt)
        M_hat[k] = obs.M_wave_hat

    # Use the last 60 s as steady state. Compare amplitude and RMS
    # error normalised by signal amplitude.
    n = int(60.0 / dt)
    err = M_hat[-n:] - M_true[-n:]
    rms_err = float(np.sqrt(np.mean(err ** 2)))
    rms_signal = float(np.sqrt(np.mean(M_true[-n:] ** 2)))
    rel_rms = rms_err / rms_signal
    assert rel_rms < 0.15, (
        f"Observer M_wave_hat RMS error = {rel_rms*100:.1f}% of signal "
        f"(expected < 15%)"
    )

    # Amplitude check: peak of |M_hat| should be within +-15% of M_amp
    amp_hat = float(np.max(np.abs(M_hat[-n:])))
    assert 0.85 * M_amp < amp_hat < 1.15 * M_amp, (
        f"Reconstructed amplitude {amp_hat:.2e} vs true {M_amp:.2e}"
    )


def test_observer_tracks_dc_bias():
    """Constant M_wave (pure bias) must be absorbed into M_bias_hat."""
    M_const = 2.0e6
    dt = 0.05
    T = 600.0  # bias state has ~20 s time constant; settle for many tau.
    n = int(T / dt) + 1
    t = np.arange(n) * dt

    # Static heel under constant moment: phi = M_const / c, phi_dot = 0.
    I, b, c = _vessel_params_for_observer()
    phi_static = M_const / c
    phi = np.full(n, phi_static)

    cfg = LuenbergerWaveObserverConfig(
        I_total=I, b44=b, c44=c, omega_e=2 * np.pi / 9.0,
    )
    obs = LuenbergerWaveObserver(cfg)
    for k in range(n):
        obs.update(phi[k], M_tank_known=0.0, dt=dt)

    M_hat = obs.M_wave_hat
    rel_err = abs(M_hat - M_const) / M_const
    assert rel_err < 0.10, (
        f"DC bias not absorbed: M_wave_hat={M_hat:.2e} vs true={M_const:.2e}"
    )


def test_observer_robust_to_mistuned_omega_e():
    """Detune omega_e by +/-15% and verify the observer still tracks
    most of the wave-moment energy (degraded but not catastrophically)."""
    omega_true = 2 * np.pi / 9.0
    M_amp = 5.0e6
    dt = 0.05
    T = 300.0
    t, phi, _, M_true = _simulate_bare_vessel_phi(omega_true, M_amp, dt, T)

    I, b, c = _vessel_params_for_observer()
    for detune in (0.85, 1.15):
        cfg = LuenbergerWaveObserverConfig(
            I_total=I, b44=b, c44=c, omega_e=omega_true * detune,
        )
        obs = LuenbergerWaveObserver(cfg)
        M_hat = np.zeros_like(t)
        for k in range(len(t)):
            obs.update(phi[k], M_tank_known=0.0, dt=dt)
            M_hat[k] = obs.M_wave_hat

        n = int(80.0 / dt)
        err = M_hat[-n:] - M_true[-n:]
        rms_err = float(np.sqrt(np.mean(err ** 2)))
        rms_signal = float(np.sqrt(np.mean(M_true[-n:] ** 2)))
        rel_rms = rms_err / rms_signal
        # Tolerate degraded performance: 15% mistune should still keep
        # error well below the signal level.
        assert rel_rms < 0.6, (
            f"Detuned (x{detune}) observer RMS error = {rel_rms*100:.1f}% "
            f"(expected < 60%)"
        )


# ====================================================================
# Closed-loop tests with ResonatorObserverRDTController + RDT tank
# ====================================================================


def test_resonator_controller_requires_attach():
    """Bare controller without attached tank must raise on thrust()."""
    I, b, c = _vessel_params_for_observer()
    obs = LuenbergerWaveObserver(LuenbergerWaveObserverConfig(
        I_total=I, b44=b, c44=c, omega_e=2 * np.pi / 9.0,
    ))
    ctrl = ResonatorObserverRDTController(observer=obs)
    with pytest.raises(RuntimeError):
        ctrl.thrust({"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0}, 0.0)


def test_resonator_controller_beats_passive_in_regular_seas():
    """Closed-loop with the observer-based controller must reduce
    resonant roll vs the corresponding passive open-tube tank."""
    omega = 2 * np.pi / 11.4
    M_amp = 5.0e6

    def M_wave(t):
        return M_amp * np.sin(omega * t)

    I, b, c = _vessel_params_for_observer()
    obs = LuenbergerWaveObserver(LuenbergerWaveObserverConfig(
        I_total=I, b44=b, c44=c, omega_e=omega,
    ))
    cfg_a = RDTUtubeConfig(F_max=300_000.0, **_utube_geom_kw())
    ctrl = ResonatorObserverRDTController(observer=obs)
    tank_a = RDTUtubeTank(cfg_a, controller=ctrl)
    sys_a = CoupledSystem(_csov_vessel(), tanks=[tank_a], M_wave_func=M_wave)
    res_a = run_simulation(sys_a, dt=0.05, t_end=600.0)

    tank_p = OpenUtubeTank(OpenUtubeConfig(**_utube_geom_kw()))
    sys_p = CoupledSystem(_csov_vessel(), tanks=[tank_p], M_wave_func=M_wave)
    res_p = run_simulation(sys_p, dt=0.05, t_end=600.0)

    n = int(100 / 0.05)
    amp_a = float(np.max(np.abs(res_a.phi[-n:])))
    amp_p = float(np.max(np.abs(res_p.phi[-n:])))
    assert amp_a < amp_p, (
        f"Observer-active ({np.rad2deg(amp_a):.2f} deg) did not beat "
        f"passive ({np.rad2deg(amp_p):.2f} deg)."
    )


def test_resonator_controller_brackets_pd_and_ideal():
    """Performance ordering at resonance:
       passive >  PD-state-feedback  >  Resonator+observer  >  ideal-inverse-dyn.
    The observer-based controller should land BETWEEN PD and ideal."""
    omega = 2 * np.pi / 11.4
    M_amp = 5.0e6

    def M_wave(t):
        return M_amp * np.sin(omega * t)

    I, b, c = _vessel_params_for_observer()
    geom = _utube_geom_kw()

    # Ideal inverse-dynamics
    ctrl_id = InverseDynamicsRDTController(M_wave_func=M_wave)
    tank_id = RDTUtubeTank(RDTUtubeConfig(F_max=300_000.0, **geom),
                           controller=ctrl_id)
    sys_id = CoupledSystem(_csov_vessel(), tanks=[tank_id], M_wave_func=M_wave)
    res_id = run_simulation(sys_id, dt=0.05, t_end=600.0)

    # Resonator + observer
    obs = LuenbergerWaveObserver(LuenbergerWaveObserverConfig(
        I_total=I, b44=b, c44=c, omega_e=omega,
    ))
    ctrl_ob = ResonatorObserverRDTController(observer=obs)
    tank_ob = RDTUtubeTank(RDTUtubeConfig(F_max=300_000.0, **geom),
                           controller=ctrl_ob)
    sys_ob = CoupledSystem(_csov_vessel(), tanks=[tank_ob], M_wave_func=M_wave)
    res_ob = run_simulation(sys_ob, dt=0.05, t_end=600.0)

    # PD
    ctrl_pd = StateFeedbackRDTController(K_phi=0.0, K_phidot=5e7)
    tank_pd = RDTUtubeTank(RDTUtubeConfig(F_max=300_000.0, **geom),
                           controller=ctrl_pd)
    sys_pd = CoupledSystem(_csov_vessel(), tanks=[tank_pd], M_wave_func=M_wave)
    res_pd = run_simulation(sys_pd, dt=0.05, t_end=600.0)

    n = int(100 / 0.05)
    amp_id = float(np.max(np.abs(res_id.phi[-n:])))
    amp_ob = float(np.max(np.abs(res_ob.phi[-n:])))
    amp_pd = float(np.max(np.abs(res_pd.phi[-n:])))

    # Ideal must be best (smallest amplitude)
    assert amp_id <= amp_ob + 1e-6, (
        f"Ideal ({np.rad2deg(amp_id):.3f}) not <= observer "
        f"({np.rad2deg(amp_ob):.3f})"
    )
    # Observer must beat (or at least match within margin) PD
    assert amp_ob <= 1.10 * amp_pd, (
        f"Observer ({np.rad2deg(amp_ob):.3f}) much worse than PD "
        f"({np.rad2deg(amp_pd):.3f})"
    )
