"""Tests for the open-top passive U-tube tank.

The cross-coupling and self coefficients follow Holden, Perez & Fossen
(2011), validated against 44 model-scale experiments. See
:mod:`tanks.utube_open` and ``docs/utube_derivation.md`` Part I.

Vertical convention: ``duct_below_waterline`` is z-down (positive = duct
below waterline). Brucon's variable name ``utube_datum_to_cog`` was
ambiguous; we resolve by re-referencing to the waterline. Numerically,
brucon's coefficient algebra uses the same formula
``a_phi = Q*(z_d + h_0)``, so the legacy test value below now passes
when interpreted as a pure algebraic check.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from roll_reduction_tanks.tanks.utube_open import OpenUtubeConfig, OpenUtubeTank


# Geometry from brucon's tank_roll_model_test.cpp::FlowRateAndGeometry
# (numeric values reproduced as-is; the variable was named
# `utube_datum_to_cog` in brucon).
WINDEN_TEST = OpenUtubeConfig(
    duct_below_waterline=-13.48,
    undisturbed_fluid_height=2.5,
    utube_duct_height=0.125,
    resevoir_duct_width=2.0,
    utube_duct_width=25.0,
    tank_thickness=5.0,
    tank_to_xcog=0.0,
    tank_wall_friction_coef=0.1,
    tank_height=5.0,
    rho=1025.0,
    g=9.81,
)


@pytest.fixture
def tank():
    return OpenUtubeTank(WINDEN_TEST)


# ============================================================ algebraic checks against brucon's numeric outputs


def test_brucon_flow_rate(tank):
    """Q = rho * b_r * W^2 * t / 2 -- unchanged from brucon."""
    assert tank.Q == pytest.approx(3736125.0, abs=0.0)


def test_brucon_a_phi_formula(tank):
    """a_phi = Q * (z_d + h_0) -- Holden eq. 14, matches brucon's formula."""
    expected = WINDEN_TEST.duct_below_waterline + WINDEN_TEST.undisturbed_fluid_height
    assert tank.a_phi == pytest.approx(tank.Q * expected, abs=0.0)
    # And matches brucon's pinned numerical output for the Winden geometry.
    assert tank.a_phi == pytest.approx(-41022652.5, abs=0.0)


def test_brucon_remaining_cross_coupling_coefficients(tank):
    """a_y, c_phi and a_psi match brucon. Sign of c_phi*phi on the tank
    EOM RHS is opposite to brucon (Holden gives +, brucon writes -),
    but the *magnitude* stored as `c_phi` here equals brucon's."""
    assert tank.a_y == pytest.approx(-3736125.0, abs=0.0)
    assert tank.c_phi == pytest.approx(36651386.25, abs=0.0)
    assert tank.a_psi == pytest.approx(0.0, abs=0.0)


def test_brucon_self_coefficients(tank):
    """Self coefficients (a_tau, b_tau, c_tau) match Bertram (4.123) and brucon."""
    assert tank.a_tau == pytest.approx(816343312.5, abs=0.0)
    assert tank.b_tau == pytest.approx(646536431.25, abs=0.0)
    assert tank.c_tau == pytest.approx(36651386.25, abs=0.0)


def test_brucon_natural_frequency(tank):
    """omega_tau = sqrt(c_tau / a_tau) -- Bertram (4.123)."""
    assert tank.natural_frequency == pytest.approx(0.21188918, abs=1e-2)


# ============================================================ Holden / Bertram identities


def test_a_tau_matches_bertram_4123():
    """Bertram eq. 4.123: omega_tau^2 = g / (h_0 + W*b_r/(2*h_d))."""
    cfg = WINDEN_TEST
    W = cfg.utube_duct_width + cfg.resevoir_duct_width
    expected_omega2 = cfg.g / (
        cfg.undisturbed_fluid_height + W * cfg.resevoir_duct_width / (2.0 * cfg.utube_duct_height)
    )
    t_obj = OpenUtubeTank(cfg)
    assert t_obj.natural_frequency ** 2 == pytest.approx(expected_omega2, rel=1e-12)


def test_c_phi_equals_Q_g(tank):
    """Bertram eq. 4.122: c_phi = Q * g."""
    assert tank.c_phi == pytest.approx(tank.Q * tank.config.g, abs=0.0)


def test_a_phi_changes_sign_at_dead_zone():
    """a_phi vanishes at z_d = -h_0 (one fluid-depth above the WL) and
    flips sign across it. Below the dead zone (z_d > -h_0, including all
    in-hull placements) a_phi > 0; above it a_phi < 0."""
    cfg_below = replace(WINDEN_TEST, duct_below_waterline=0.0)        # z_d > -h_0 (h_0 = 2.5)
    cfg_above = replace(WINDEN_TEST, duct_below_waterline=-10.0)      # z_d < -h_0
    cfg_dead  = replace(WINDEN_TEST, duct_below_waterline=-2.5)       # z_d == -h_0
    assert OpenUtubeTank(cfg_below).a_phi > 0
    assert OpenUtubeTank(cfg_above).a_phi < 0
    assert OpenUtubeTank(cfg_dead).a_phi == pytest.approx(0.0, abs=1e-9)


# ============================================================ dynamics / reciprocity


def test_static_equilibrium_free_surface_effect():
    """Quasi-static: a slow constant hull tilt phi should produce a
    fluid tilt tau in the *same* sign (free-surface effect destabilises).

    Drive the tank with phi = 0.01 rad held constant, phi_ddot = 0; let
    the fluid relax for many natural periods. Steady state:
    c_tau*tau = +c_phi*phi (both Q*g, so tau -> phi).
    """
    cfg = replace(WINDEN_TEST, tank_wall_friction_coef=0.5)  # heavy damping
    t_obj = OpenUtubeTank(cfg)
    phi = 0.01

    def kin(t):
        return {"phi": phi, "phi_dot": 0.0, "phi_ddot": 0.0}

    dt = 0.05
    for i in range(int(50 * t_obj.natural_period / dt)):
        t_obj.step_rk4(kin, i * dt, dt)

    tau_ss = t_obj.state[0]
    # tau / phi should equal c_phi / c_tau = 1 in equilibrium.
    assert tau_ss == pytest.approx(phi, rel=0.05)
    assert tau_ss > 0  # SAME sign as phi


def test_lagrangian_reciprocity_via_force_balance():
    """The cross coefficient that multiplies phi_ddot on the tank RHS
    must equal the cross coefficient that multiplies tau_ddot in the
    moment back on the hull (a_phi)."""
    cfg = replace(WINDEN_TEST, duct_below_waterline=5.0)  # arbitrary in-hull placement
    t_obj = OpenUtubeTank(cfg)

    # Probe the vessel-side moment by setting tau_ddot via _last_tau_ddot
    # directly and reading off the roll moment with tau = 0.
    t_obj.state = np.array([0.0, 0.0])
    t_obj._last_tau_ddot = 1.0
    M = t_obj.forces({"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0})["roll"]
    # M = +a_phi*tau_ddot + c_phi*tau = +a_phi * 1.0
    assert M == pytest.approx(t_obj.a_phi, abs=0.0)


def test_zero_state_zero_moment(tank):
    """At zero state and zero vessel kinematics, all forces should be zero."""
    f = tank.forces({"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0})
    for k, v in f.items():
        assert v == 0.0, f"{k} not zero"


def test_clamp_to_physical_max(tank):
    """If we drive the state past tau_max, the integrator must clamp it."""
    tank_init = OpenUtubeTank(WINDEN_TEST)
    tau_max = tank_init.tau_max
    tank_init.state = np.array([0.99 * tau_max, 1.0])
    def kin(t):
        return {"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0}
    for _ in range(100):
        tank_init.step_rk4(kin, 0.0, 0.05)
    assert abs(tank_init.state[0]) <= tau_max + 1e-12


def _underdamped_config():
    """Same geometry as the Winden test but with smaller friction so
    fluid is underdamped and oscillates."""
    return OpenUtubeConfig(
        duct_below_waterline=-13.48,
        undisturbed_fluid_height=2.5,
        utube_duct_height=0.125,
        resevoir_duct_width=2.0,
        utube_duct_width=25.0,
        tank_thickness=5.0,
        tank_to_xcog=0.0,
        tank_wall_friction_coef=0.005,
        tank_height=5.0,
    )


def test_unforced_decay_is_oscillatory_and_decaying():
    cfg = _underdamped_config()
    t_obj = OpenUtubeTank(cfg, tau0=np.deg2rad(2.0))

    def kin(t):
        return {"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0}

    dt = 0.1
    n = 1000
    tau = np.empty(n)
    tau[0] = t_obj.state[0]
    for i in range(1, n):
        t_obj.step_rk4(kin, i * dt, dt)
        tau[i] = t_obj.state[0]

    peaks = []
    for i in range(1, n - 1):
        if tau[i - 1] < tau[i] > tau[i + 1] and tau[i] > 0:
            peaks.append((i * dt, tau[i]))
    assert len(peaks) >= 3
    peak_amps = [p[1] for p in peaks]
    for a, b in zip(peak_amps, peak_amps[1:]):
        assert b < a


def test_natural_period_matches_decay_period():
    cfg = WINDEN_TEST
    t_obj = OpenUtubeTank(cfg, tau0=np.deg2rad(2.0))

    def kin(t):
        return {"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0}

    dt = 0.05
    n = int(8 * t_obj.natural_period / dt)
    t_arr = np.empty(n)
    tau = np.empty(n)
    tau[0] = t_obj.state[0]
    t_arr[0] = 0.0
    for i in range(1, n):
        t_obj.step_rk4(kin, i * dt, dt)
        tau[i] = t_obj.state[0]
        t_arr[i] = i * dt

    crossings = []
    for i in range(1, n):
        if tau[i - 1] < 0 and tau[i] >= 0:
            frac = -tau[i - 1] / (tau[i] - tau[i - 1])
            crossings.append(t_arr[i - 1] + frac * dt)
    if len(crossings) < 3:
        pytest.skip("Damping too high to measure period; skip.")
    measured = float(np.mean(np.diff(crossings)))
    expected = t_obj.natural_period / np.sqrt(max(1 - t_obj.damping_ratio**2, 1e-9))
    assert measured == pytest.approx(expected, rel=0.02)


# ============================================================ nonlinear (quadratic) damping (Holden eq. 22)


def test_quad_damping_default_off():
    """Default config has quad_damping_coef = 0; behaviour identical to
    pure linear model."""
    tank = OpenUtubeTank(WINDEN_TEST)
    assert tank.b_quad == 0.0
    # Force evaluation reproduces linear-only RHS:
    deriv = tank.derivatives(np.array([0.01, 0.05]),
                             {"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0},
                             t=0.0)
    expected_tau_ddot = (
        - tank.b_tau * 0.05 - tank.c_tau * 0.01
    ) / tank.a_tau
    assert deriv[1] == pytest.approx(expected_tau_ddot, rel=1e-12)


def test_quad_damping_increases_dissipation_at_large_amplitude():
    """At large tau_dot, quadratic damping bleeds energy faster than the
    same-b_tau linear-only model. Use a lightly-damped baseline so the
    quadratic term is visibly different.
    """
    base = replace(WINDEN_TEST, tank_wall_friction_coef=0.005)  # very light linear damping
    cfg_lin = replace(base, quad_damping_coef=0.0)
    cfg_nl  = replace(base, quad_damping_coef=1.0e10)

    tau0 = np.deg2rad(8.0)
    tank_lin = OpenUtubeTank(cfg_lin, tau0=tau0)
    tank_nl  = OpenUtubeTank(cfg_nl,  tau0=tau0)

    def kin(t):
        return {"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0}

    dt = 0.05
    # Run long enough that tau_dot has been at peak for a while -- one
    # full period.
    n = int(tank_lin.natural_period / dt)
    for i in range(n):
        tank_lin.step_rk4(kin, i * dt, dt)
        tank_nl.step_rk4(kin, i * dt, dt)

    def E(tk):
        return (0.5 * tk.a_tau * tk.state[1] ** 2
                + 0.5 * tk.c_tau * tk.state[0] ** 2)

    assert E(tank_nl) < E(tank_lin), (
        f"Quadratic damping should bleed more energy at high amplitude "
        f"(got E_nl={E(tank_nl):.3e} vs E_lin={E(tank_lin):.3e})"
    )


def test_quad_damping_amplitude_dependent_decay_rate():
    """Hallmark of quadratic damping: decay rate per cycle is
    amplitude-dependent (decays roughly hyperbolically), unlike the
    constant log-decrement of linear damping. Test that envelope of
    a quad-damped decay drops faster between cycles 1-2 than between
    cycles 4-5."""
    cfg = replace(WINDEN_TEST,
                  tank_wall_friction_coef=0.0,    # remove linear damping
                  quad_damping_coef=2.0e10)
    tau0 = np.deg2rad(10.0)
    tank = OpenUtubeTank(cfg, tau0=tau0)

    def kin(t):
        return {"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0}

    dt = 0.05
    n = int(8 * tank.natural_period / dt)
    tau = np.empty(n)
    for i in range(n):
        tau[i] = tank.state[0]
        tank.step_rk4(kin, i * dt, dt)

    # Find positive peaks (local maxima of tau).
    peaks = []
    for i in range(1, n - 1):
        if tau[i - 1] < tau[i] >= tau[i + 1] and tau[i] > 0:
            peaks.append(tau[i])
    assert len(peaks) >= 5, f"Need >=5 peaks, got {len(peaks)}"

    # Linear damping: peaks[k+1]/peaks[k] = const. Quadratic damping:
    # ratio increases (decays slower) as amplitude shrinks.
    ratio_early = peaks[1] / peaks[0]
    ratio_late  = peaks[4] / peaks[3]
    assert ratio_late > ratio_early, (
        f"Expected later-cycle ratio > earlier-cycle ratio for quadratic "
        f"damping (got early={ratio_early:.3f}, late={ratio_late:.3f})"
    )


def test_estimate_quad_damping_from_loss_units_and_scaling():
    """The estimator should scale linearly in K_loss and in rho, and as
    1/h_d^2."""
    tank = OpenUtubeTank(WINDEN_TEST)
    bq_1 = tank.estimate_quad_damping_from_loss(K_loss=1.0)
    bq_2 = tank.estimate_quad_damping_from_loss(K_loss=2.0)
    assert bq_2 == pytest.approx(2.0 * bq_1, rel=1e-12)

    cfg_thin_duct = replace(WINDEN_TEST, utube_duct_height=WINDEN_TEST.utube_duct_height / 2)
    bq_thin = OpenUtubeTank(cfg_thin_duct).estimate_quad_damping_from_loss(K_loss=1.0)
    assert bq_thin == pytest.approx(4.0 * bq_1, rel=1e-12)

    # And the value is positive, finite, and order of magnitude consistent
    # with Holden's experimental b_quad ~= 280 kg/m * (W/2) for his small
    # tank. For our much larger CSOV-scale tank we expect a much larger
    # value -- just test it's positive and finite.
    assert bq_1 > 0
    assert np.isfinite(bq_1)
