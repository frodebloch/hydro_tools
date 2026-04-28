"""Tests for FrequencyTrackingController.

Covers:
  * static mapping ``_target_opening`` (saturation + linear interpolation)
  * period estimation on a synthetic sinusoidal phi_dot trace
  * convergence of the smoothed opening towards the target
  * controller does not crash before any zero-crossing has occurred
    (returns the configured initial opening)
"""
from __future__ import annotations

import math

import pytest

from roll_reduction_tanks.controllers.frequency_tracking import (
    FrequencyTrackingController,
)


def test_target_opening_saturates_at_T_closed():
    c = FrequencyTrackingController(T_closed=11.4, T_open=14.0)
    assert c._target_opening(8.0) == 0.0
    assert c._target_opening(11.4) == 0.0


def test_target_opening_saturates_at_T_open():
    c = FrequencyTrackingController(T_closed=11.4, T_open=14.0)
    assert c._target_opening(14.0) == 1.0
    assert c._target_opening(20.0) == 1.0


def test_target_opening_linear_midpoint():
    c = FrequencyTrackingController(T_closed=11.4, T_open=14.0)
    T_mid = 0.5 * (11.4 + 14.0)
    assert c._target_opening(T_mid) == pytest.approx(0.5)


def test_target_opening_linear_general():
    c = FrequencyTrackingController(T_closed=10.0, T_open=20.0)
    # 25% from T_closed
    assert c._target_opening(12.5) == pytest.approx(0.25)
    # 75% from T_closed
    assert c._target_opening(17.5) == pytest.approx(0.75)


def test_initial_opening_before_any_crossing():
    c = FrequencyTrackingController(initial_opening=0.7,
                                    smoothing_tau=0.0)
    # No phi_dot history yet; should return initial.
    u = c.opening({"phi_dot": 0.0}, t=0.0)
    assert u == pytest.approx(0.7)


def test_period_estimation_on_sine():
    """Drive the controller with phi_dot = cos(2*pi*t/T) for T = 12 s
    and check that ``estimated_period`` converges to 12 s.
    """
    T = 12.0
    omega = 2 * math.pi / T
    c = FrequencyTrackingController(T_closed=11.4, T_open=14.0,
                                    smoothing_tau=0.0)
    dt = 0.05
    t = 0.0
    # Run several full periods so multiple zero-crossings are seen.
    for _ in range(int(5 * T / dt)):
        phi_dot = math.cos(omega * t)
        c.opening({"phi_dot": phi_dot}, t=t)
        t += dt
    assert c.estimated_period is not None
    assert c.estimated_period == pytest.approx(T, rel=0.02)


def test_smoothed_opening_converges_to_target():
    """With a short tau and many samples at constant target, the
    smoothed opening should match the target closely.
    """
    T = 12.0  # midpoint of [11.4, 14.0]? actually 12 -> u=0.6/2.6=0.2308
    omega = 2 * math.pi / T
    c = FrequencyTrackingController(T_closed=11.4, T_open=14.0,
                                    smoothing_tau=1.0,
                                    initial_opening=0.0)
    dt = 0.05
    t = 0.0
    for _ in range(int(20 * T / dt)):
        phi_dot = math.cos(omega * t)
        u = c.opening({"phi_dot": phi_dot}, t=t)
        t += dt
    expected = (T - 11.4) / (14.0 - 11.4)
    assert u == pytest.approx(expected, abs=0.02)


def test_opening_is_clamped_to_unit_interval():
    """Even with a wild initial value, the public output is in [0,1]."""
    c = FrequencyTrackingController(initial_opening=2.5)
    u = c.opening({"phi_dot": 0.0}, t=0.0)
    assert 0.0 <= u <= 1.0
