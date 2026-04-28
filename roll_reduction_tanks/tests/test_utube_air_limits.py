"""Limit tests for the air-valve U-tube tank.

* Fully open valve: behaves like an open-top U-tube (matching natural
  frequency, matching free-decay period).
* Fully closed valve: tank natural frequency rises predictably above the
  open-tube value.
"""
from __future__ import annotations

import numpy as np
import pytest

from roll_reduction_tanks.controllers.constant import (
    FullyClosedValve,
    FullyOpenValve,
)
from roll_reduction_tanks.tanks.utube_air import (
    AirValveUtubeConfig,
    AirValveUtubeTank,
)
from roll_reduction_tanks.tanks.utube_open import (
    OpenUtubeConfig,
    OpenUtubeTank,
)


def _shared_geometry_kwargs() -> dict:
    """Underdamped Winden-style geometry shared by open and air tanks."""
    return dict(
        duct_below_waterline=-13.48,
        undisturbed_fluid_height=2.5,
        utube_duct_height=0.508,
        resevoir_duct_width=2.0,
        utube_duct_width=25.0,
        tank_thickness=5.0,
        tank_to_xcog=0.0,
        tank_wall_friction_coef=0.005,
        tank_height=5.0,
    )


def test_air_open_valve_matches_open_tube_natural_frequency():
    open_cfg = OpenUtubeConfig(**_shared_geometry_kwargs())
    open_tank = OpenUtubeTank(open_cfg)

    air_cfg = AirValveUtubeConfig(
        chamber_volume_each=50.0,
        valve_area_max=10.0,  # very large -> no significant pressure drop
        **_shared_geometry_kwargs(),
    )
    air_tank = AirValveUtubeTank(air_cfg, controller=FullyOpenValve())

    # Coefficients should match exactly (geometry algebra is identical).
    assert open_tank.a_tau == pytest.approx(air_tank.a_tau)
    assert open_tank.b_tau == pytest.approx(air_tank.b_tau)
    assert open_tank.c_tau == pytest.approx(air_tank.c_tau)
    # And the air tank's "open valve" analytic prediction matches.
    assert air_tank.open_valve_natural_frequency == pytest.approx(
        open_tank.natural_frequency
    )


def test_air_closed_valve_natural_frequency_higher_than_open():
    air_cfg = AirValveUtubeConfig(
        chamber_volume_each=10.0,  # smaller chamber -> stiffer spring
        valve_area_max=0.0,
        **_shared_geometry_kwargs(),
    )
    air_tank = AirValveUtubeTank(air_cfg, controller=FullyClosedValve())
    open_w = air_tank.open_valve_natural_frequency
    closed_w = air_tank.closed_valve_natural_frequency
    assert closed_w > open_w
    # And the gas spring should significantly stiffen the response — at least
    # 10 % above the open value with these parameters.
    assert closed_w > 1.10 * open_w


def test_air_closed_valve_decay_period_matches_analytic():
    """Free decay with valve shut: tau oscillates near closed_valve_natural_frequency."""
    air_cfg = AirValveUtubeConfig(
        chamber_volume_each=10.0,
        valve_area_max=0.0,
        **_shared_geometry_kwargs(),
    )
    # Use a small amplitude so the linearised stiffness is accurate
    # (closed-valve gas compression is strongly nonlinear at large tau).
    tank = AirValveUtubeTank(air_cfg, controller=FullyClosedValve(), tau0=0.001)

    def kin(t):
        return {"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0}

    dt = 0.005
    n = 8000
    tau = np.empty(n)
    for i in range(n):
        tau[i] = tank.state[0]
        tank.step_rk4(kin, i * dt, dt)

    # Find positive-going zero crossings of tau.
    s = np.sign(tau)
    crossings = np.where((s[:-1] <= 0) & (s[1:] > 0))[0]
    assert len(crossings) >= 3, "expected at least 3 oscillations"
    # Period from successive zero crossings.
    periods = np.diff(crossings) * dt
    measured_T = float(np.mean(periods))
    expected_T = 2 * np.pi / tank.closed_valve_natural_frequency
    assert measured_T == pytest.approx(expected_T, rel=0.05)


def test_air_open_valve_decay_period_matches_open_tube():
    """Free decay with valve wide open: tau period ~ open-tube period."""
    air_cfg = AirValveUtubeConfig(
        chamber_volume_each=50.0,
        valve_area_max=10.0,
        **_shared_geometry_kwargs(),
    )
    tank = AirValveUtubeTank(air_cfg, controller=FullyOpenValve(), tau0=0.05)

    def kin(t):
        return {"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0}

    dt = 0.01
    n = 8000
    tau = np.empty(n)
    for i in range(n):
        tau[i] = tank.state[0]
        tank.step_rk4(kin, i * dt, dt)

    s = np.sign(tau)
    crossings = np.where((s[:-1] <= 0) & (s[1:] > 0))[0]
    assert len(crossings) >= 3
    periods = np.diff(crossings) * dt
    measured_T = float(np.mean(periods))
    expected_T = 2 * np.pi / tank.open_valve_natural_frequency
    assert measured_T == pytest.approx(expected_T, rel=0.05)


def test_air_zero_state_zero_pressure_difference_zero_moment():
    air_cfg = AirValveUtubeConfig(
        chamber_volume_each=50.0,
        valve_area_max=0.5,
        **_shared_geometry_kwargs(),
    )
    tank = AirValveUtubeTank(air_cfg)
    p1, p2 = tank.chamber_pressures()
    assert p1 == pytest.approx(p2)
    f = tank.forces({"phi": 0.0, "phi_dot": 0.0, "phi_ddot": 0.0})
    assert f["roll"] == pytest.approx(0.0)
