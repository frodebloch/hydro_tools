"""Tests for the active rim-driven-thruster (RDT) U-tube tank."""
from __future__ import annotations

import numpy as np

from roll_reduction_tanks.coupling import CoupledSystem
from roll_reduction_tanks.simulation import run_simulation
from roll_reduction_tanks.tanks.utube_open import OpenUtubeConfig, OpenUtubeTank
from roll_reduction_tanks.tanks.utube_rdt import (
    RDTUtubeConfig,
    RDTUtubeTank,
    InverseDynamicsRDTController,
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


# ---------------------------------------------------------- behaviour


def test_zero_thrust_matches_passive_open_tube():
    """RDT tank with controller commanding zero force must reproduce
    the passive OpenUtubeTank time history exactly."""

    class ZeroController(StateFeedbackRDTController):
        def thrust(self, vessel_kin, t):  # noqa
            return 0.0

    rdt = RDTUtubeTank(RDTUtubeConfig(**_utube_geom_kw()), controller=ZeroController())
    passive = OpenUtubeTank(OpenUtubeConfig(**_utube_geom_kw()))

    def M_wave(t):
        return 5e6 * np.sin(2 * np.pi / 11.0 * t)

    sys_a = CoupledSystem(_csov_vessel(), tanks=[rdt],     M_wave_func=M_wave)
    sys_b = CoupledSystem(_csov_vessel(), tanks=[passive], M_wave_func=M_wave)

    res_a = run_simulation(sys_a, dt=0.05, t_end=120.0)
    res_b = run_simulation(sys_b, dt=0.05, t_end=120.0)

    np.testing.assert_allclose(res_a.phi, res_b.phi, atol=1e-9, rtol=1e-9)


def test_thrust_saturation_is_respected():
    """Saturated controller commanding 10x F_max must clip to F_max."""

    class HardController(StateFeedbackRDTController):
        def thrust(self, vessel_kin, t):  # noqa
            return 10 * 200_000.0

    cfg = RDTUtubeConfig(F_max=200_000.0, **_utube_geom_kw())
    tank = RDTUtubeTank(cfg, controller=HardController())

    def M_wave(t):
        return 0.0

    sys = CoupledSystem(_csov_vessel(), tanks=[tank], M_wave_func=M_wave)
    run_simulation(sys, dt=0.05, t_end=2.0)
    assert tank.last_F_cmd == 10 * 200_000.0
    assert tank.last_F_applied == 200_000.0


def test_inverse_dynamics_beats_passive_in_regular_seas():
    """The ideal active controller, given perfect M_wave knowledge,
    must produce smaller steady-state roll than the corresponding
    passive open-tube tank at the vessel resonance."""
    omega = 2 * np.pi / 11.4   # vessel T_n approx
    M_amp = 5e6

    def M_wave(t):
        return M_amp * np.sin(omega * t)

    # Ideal active
    cfg_a = RDTUtubeConfig(F_max=300_000.0, **_utube_geom_kw())
    ctrl = InverseDynamicsRDTController(M_wave_func=M_wave)
    tank_a = RDTUtubeTank(cfg_a, controller=ctrl)
    sys_a = CoupledSystem(_csov_vessel(), tanks=[tank_a], M_wave_func=M_wave)
    res_a = run_simulation(sys_a, dt=0.05, t_end=600.0)

    # Passive
    tank_p = OpenUtubeTank(OpenUtubeConfig(**_utube_geom_kw()))
    sys_p = CoupledSystem(_csov_vessel(), tanks=[tank_p], M_wave_func=M_wave)
    res_p = run_simulation(sys_p, dt=0.05, t_end=600.0)

    # Compare the last-100-second amplitude
    n = int(100 / 0.05)
    amp_a = float(np.max(np.abs(res_a.phi[-n:])))
    amp_p = float(np.max(np.abs(res_p.phi[-n:])))
    assert amp_a < amp_p, (
        f"Active ({np.rad2deg(amp_a):.2f} deg) did not beat passive "
        f"({np.rad2deg(amp_p):.2f} deg)."
    )
    # And by a substantial margin (factor of 2+).
    assert amp_a < 0.5 * amp_p, (
        f"Active reduction smaller than expected: active "
        f"{np.rad2deg(amp_a):.2f} deg vs passive {np.rad2deg(amp_p):.2f} deg "
        f"(ratio = {amp_a/amp_p:.2f})."
    )


def test_state_feedback_pd_reduces_roll_at_resonance():
    """A pure rate-feedback controller should reduce resonant roll
    relative to the passive tank, even without wave knowledge."""
    omega_n = np.sqrt(_csov_vessel().config.c44
                      / _csov_vessel().config.total_inertia)
    M_amp = 5e6

    def M_wave(t):
        return M_amp * np.sin(omega_n * t)

    K_phidot = 5e7   # tuned by inspection; rate gain in N / (rad/s)
    cfg = RDTUtubeConfig(F_max=300_000.0, **_utube_geom_kw())
    ctrl = StateFeedbackRDTController(K_phi=0.0, K_phidot=K_phidot)
    tank_a = RDTUtubeTank(cfg, controller=ctrl)
    sys_a = CoupledSystem(_csov_vessel(), tanks=[tank_a], M_wave_func=M_wave)
    res_a = run_simulation(sys_a, dt=0.05, t_end=600.0)

    tank_p = OpenUtubeTank(OpenUtubeConfig(**_utube_geom_kw()))
    sys_p = CoupledSystem(_csov_vessel(), tanks=[tank_p], M_wave_func=M_wave)
    res_p = run_simulation(sys_p, dt=0.05, t_end=600.0)

    n = int(100 / 0.05)
    amp_a = float(np.max(np.abs(res_a.phi[-n:])))
    amp_p = float(np.max(np.abs(res_p.phi[-n:])))
    assert amp_a < amp_p, (
        f"State-feedback active ({np.rad2deg(amp_a):.2f} deg) did not "
        f"beat passive ({np.rad2deg(amp_p):.2f} deg)."
    )


def test_inverse_dynamics_controller_requires_attach():
    """An InverseDynamicsRDTController used without attach_tank() must
    raise (RDTUtubeTank.__init__ should attach automatically; the
    raise guards against direct misuse)."""
    ctrl = InverseDynamicsRDTController(M_wave_func=lambda t: 0.0)
    import pytest
    with pytest.raises(RuntimeError):
        ctrl.thrust({"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0}, 0.0)
