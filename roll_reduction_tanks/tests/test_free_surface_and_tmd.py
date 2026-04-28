"""Tests for the free-surface tank and the tuned-mass-damper baseline."""
from __future__ import annotations

import numpy as np
import pytest

from roll_reduction_tanks.tanks.free_surface import (
    FreeSurfaceConfig,
    FreeSurfaceTank,
)
from roll_reduction_tanks.tanks.tuned_mass_damper import (
    TunedMassDamperConfig,
    TunedMassDamperTank,
    den_hartog_optimal,
)


# --------------------------------------------------------------------- free surface


def _free_surface_config(**overrides) -> FreeSurfaceConfig:
    base = dict(
        length=20.0,
        width=8.0,
        fluid_depth=2.0,
        z_tank=4.0,
        z_cog=2.5,
        damping_ratio=0.05,
    )
    base.update(overrides)
    return FreeSurfaceConfig(**base)


def test_free_surface_natural_frequency_lloyd_formula():
    """omega_n^2 = pi g / L * tanh(pi h / L)."""
    cfg = _free_surface_config()
    tank = FreeSurfaceTank(cfg)
    kL = np.pi / cfg.length
    expected = np.sqrt(cfg.g * kL * np.tanh(kL * cfg.fluid_depth))
    assert tank.natural_frequency == pytest.approx(expected)


def test_free_surface_zero_state_zero_moment():
    tank = FreeSurfaceTank(_free_surface_config())
    f = tank.forces({"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0})
    assert f["roll"] == pytest.approx(0.0)


def test_free_surface_decay_period_matches_natural():
    cfg = _free_surface_config(damping_ratio=0.01)
    tank = FreeSurfaceTank(cfg, q0=0.02)

    def kin(t):
        return {"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0}

    dt = 0.01
    n = 6000
    q = np.empty(n)
    for i in range(n):
        q[i] = tank.state[0]
        tank.step_rk4(kin, i * dt, dt)

    s = np.sign(q)
    crossings = np.where((s[:-1] <= 0) & (s[1:] > 0))[0]
    assert len(crossings) >= 3
    measured_T = float(np.mean(np.diff(crossings))) * dt
    assert measured_T == pytest.approx(tank.natural_period, rel=0.05)


def test_free_surface_static_gm_loss():
    """Steady roll ``phi`` with ``phi_dot = phi_ddot = 0`` should drive the
    sloshing mode to ``q_ss = g phi / omega_n^2`` and produce a hull
    moment combining the dynamic-mode static gravity term and the extra
    static surface-tilt stiffness, recovering the *classical* free-surface
    GM loss in the static limit::

        M = m_eq g q - dc44_extra phi
          = -dc44_classical * phi    (since at static eq m_eq g q = dc44_dyn phi)

    where dc44_classical = rho_t g W L^3 / 12.
    """
    cfg = _free_surface_config(damping_ratio=0.50)  # heavy damping for fast settling
    tank = FreeSurfaceTank(cfg)

    phi = 0.05  # rad, ~2.9 deg
    def kin(t):
        return {"phi": phi, "phi_dot": 0.0, "phi_ddot": 0.0}

    dt = 0.01
    n = 4000
    for i in range(n):
        tank.step_rk4(kin, i * dt, dt)

    q_expected = cfg.g * phi / tank.omega_n ** 2
    assert tank.state[0] == pytest.approx(q_expected, rel=0.02)
    assert tank.state[1] == pytest.approx(0.0, abs=5e-3)

    M = tank.forces(kin(n * dt))["roll"]
    # Classical hydrostatic free-surface moment (destabilising: +ve, in
    # the same sense as +phi).
    dc44_classical = cfg.rho * cfg.g * cfg.width * cfg.length ** 3 / 12.0
    M_expected = dc44_classical * phi
    assert M == pytest.approx(M_expected, rel=0.02)


def test_free_surface_static_extra_stiffness_decomposition():
    """dc44_extra + dc44_dynamic should equal dc44_classical."""
    tank = FreeSurfaceTank(_free_surface_config())
    dc44_classical = tank.config.rho * tank.config.g * tank.config.width \
                   * tank.config.length ** 3 / 12.0
    assert tank.dc44_classical == pytest.approx(dc44_classical)
    assert tank.dc44_dynamic + tank.dc44_extra == pytest.approx(dc44_classical)
    # And dc44_dynamic = m_eq g^2 / omega_n^2.
    assert tank.dc44_dynamic == pytest.approx(
        tank.m_eq * tank.config.g ** 2 / tank.omega_n ** 2
    )


def test_free_surface_wall_elevation_and_fill_ratio():
    """beta_1 = sqrt(k_eq / K_1) * q, fill_ratio = |beta_1|/h."""
    cfg = _free_surface_config()
    tank = FreeSurfaceTank(cfg, q0=0.5)
    K_1 = cfg.rho * cfg.g * cfg.width * cfg.length / 2.0
    expected_beta = np.sqrt(tank.k_eq / K_1) * 0.5
    assert tank.wall_elevation() == pytest.approx(expected_beta)
    assert tank.fill_ratio() == pytest.approx(abs(expected_beta) / cfg.fluid_depth)


def test_free_surface_run_dry_warning():
    """Once |beta_1|/h exceeds warn_fill_ratio, a one-shot UserWarning fires."""
    cfg = _free_surface_config(warn_fill_ratio=0.1)
    tank = FreeSurfaceTank(cfg)
    # Drive hard at resonance to push fill ratio up fast.
    omega = tank.omega_n
    Phi0 = 0.05
    def kin(t):
        return {
            "phi": Phi0 * np.sin(omega * t),
            "phi_dot": Phi0 * omega * np.cos(omega * t),
            "phi_ddot": -Phi0 * omega ** 2 * np.sin(omega * t),
        }
    dt = 0.05
    with pytest.warns(UserWarning, match="fill ratio"):
        for i in range(2000):
            tank.step_rk4(kin, i * dt, dt)


# --------------------------------------------------------------------- TMD


def _tmd_config(**overrides) -> TunedMassDamperConfig:
    base = dict(
        mass=1e5,                     # 100 t
        natural_frequency=2 * np.pi / 11.4,
        z_mount=8.5,
        z_cog=2.5,                    # h_arm = 6.0 m
        damping_ratio=0.05,
    )
    base.update(overrides)
    return TunedMassDamperConfig(**base)


def test_tmd_coefficients_match_definitions():
    cfg = _tmd_config(mass=2e5, natural_frequency=0.5, damping_ratio=0.1)
    tank = TunedMassDamperTank(cfg)
    assert tank.k_t == pytest.approx(2e5 * 0.25)
    assert tank.b_t == pytest.approx(2 * 0.1 * 2e5 * 0.5)
    assert tank.h_arm == pytest.approx(cfg.z_mount - cfg.z_cog)
    assert tank.natural_period == pytest.approx(2 * np.pi / 0.5)


def test_tmd_zero_state_zero_moment():
    """Zero state + zero phi_ddot must give zero roll moment."""
    tank = TunedMassDamperTank(_tmd_config())
    f = tank.forces({"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0})
    assert f["roll"] == pytest.approx(0.0)


def test_tmd_no_gravity_coupling():
    """With phi_ddot = 0 and any phi != 0, the EOM RHS depends only on
    state (no g*phi forcing). This distinguishes the TMD from a gravity
    pendulum, which would have an additional ``-m_p L_p g phi`` term."""
    tank = TunedMassDamperTank(_tmd_config())
    deriv_phi0 = tank.derivatives(np.array([0.0, 0.0]),
                                  {"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0},
                                  t=0.0)
    deriv_phi_large = tank.derivatives(np.array([0.0, 0.0]),
                                       {"phi": 0.5, "phi_dot": 0.0, "phi_ddot": 0.0},
                                       t=0.0)
    assert np.allclose(deriv_phi0, deriv_phi_large)


def test_tmd_decay_period_matches_natural():
    cfg = _tmd_config(damping_ratio=0.01)
    tank = TunedMassDamperTank(cfg, x0=0.5)

    def kin(t):
        return {"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0}

    dt = 0.01
    n = 6000
    x = np.empty(n)
    for i in range(n):
        x[i] = tank.state[0]
        tank.step_rk4(kin, i * dt, dt)

    s = np.sign(x)
    crossings = np.where((s[:-1] <= 0) & (s[1:] > 0))[0]
    assert len(crossings) >= 3
    measured_T = float(np.mean(np.diff(crossings))) * dt
    assert measured_T == pytest.approx(tank.natural_period, rel=0.02)


def test_tmd_forced_response_amplitude():
    """At the TMD's own natural frequency, with light damping, the
    steady-state amplitude of x should approach ``-h_arm Phi / (2 zeta)``
    where Phi is the phi_ddot amplitude divided by omega_n^2 (the
    classic SDOF amplification factor of 1/(2 zeta) at resonance)."""
    omega_n = 1.0
    cfg = _tmd_config(natural_frequency=omega_n, damping_ratio=0.05,
                      mass=1e5, z_mount=8.5, z_cog=2.5)
    tank = TunedMassDamperTank(cfg)

    # Drive with phi(t) = Phi0 sin(omega_n t) so phi_ddot = -Phi0 omega_n^2 sin(omega_n t).
    Phi0 = 0.1  # rad
    def kin(t):
        return {
            "phi": Phi0 * np.sin(omega_n * t),
            "phi_dot": Phi0 * omega_n * np.cos(omega_n * t),
            "phi_ddot": -Phi0 * omega_n ** 2 * np.sin(omega_n * t),
        }

    dt = 0.01
    n = 8000   # long enough to settle
    x = np.empty(n)
    for i in range(n):
        x[i] = tank.state[0]
        tank.step_rk4(kin, i * dt, dt)

    # Steady-state amplitude in the latter half.
    amp_x = float(np.max(np.abs(x[n // 2:])))
    # Expected: |x| = (h_arm Phi0) / (2 zeta) for forcing of amplitude
    # m_t h_arm Phi0 omega_n^2 on a system with k = m_t omega_n^2.
    expected = tank.h_arm * Phi0 / (2 * cfg.damping_ratio)
    assert amp_x == pytest.approx(expected, rel=0.05)


# --------------------------------------------------------------------- Den Hartog


def test_den_hartog_zero_mass_ratio():
    """At mu -> 0 the optimal frequency ratio -> 1 and zeta -> 0."""
    omega_t, zeta_t = den_hartog_optimal(
        mass=1.0, h_arm=1.0, I44_total=1e9, omega_p=1.0,
    )
    mu = 1.0 / 1e9
    assert omega_t == pytest.approx(1.0 / (1 + mu))
    assert zeta_t == pytest.approx(np.sqrt(3 * mu / (8 * (1 + mu) ** 3)))
    assert zeta_t < 1e-4   # essentially zero


def test_den_hartog_known_value():
    """For mu = 0.05 (Den Hartog 1956 eq. 3.32-3.34, undamped primary):
        f_opt   = 1 / (1 + 0.05)            = 0.9524
        zeta_opt = sqrt(3*0.05 / (8*1.05^3)) = 0.1273
    """
    I44 = 1e8
    h_arm = 5.0
    m = 0.05 * I44 / h_arm ** 2
    omega_p = 1.0
    omega_t, zeta_t = den_hartog_optimal(m, h_arm, I44, omega_p)
    assert omega_t == pytest.approx(1.0 / 1.05, rel=1e-3)
    assert zeta_t == pytest.approx(0.12733, rel=1e-3)


def test_den_hartog_invalid_inputs():
    with pytest.raises(ValueError):
        den_hartog_optimal(mass=-1.0, h_arm=1.0, I44_total=1.0, omega_p=1.0)
    with pytest.raises(ValueError):
        den_hartog_optimal(mass=1.0, h_arm=0.0, I44_total=1.0, omega_p=1.0)
    with pytest.raises(ValueError):
        den_hartog_optimal(mass=1.0, h_arm=1.0, I44_total=-1.0, omega_p=1.0)
