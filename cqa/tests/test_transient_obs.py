"""Unit tests for cqa.transient_obs.pulse_response_saturated (sec.12.21.21.6 Option A).

Coverage:

* ``test_implicit_tau_cmd_recovers_controller_law``: helper accessor
  returns ``-Kp eta_hat - Kd nu_hat - b_hat - Ki I`` consistently with
  the build_observer_augmented_system_full ``tau_thr_dot`` row.
* ``test_pulse_response_saturated_infinite_cap_matches_pulse_response``:
  with a no-op clip, the RK4 saturated integrator reproduces the
  expm/trapezoidal :func:`pulse_response` trajectory to integrator
  order. Validates the formulation.
* ``test_pulse_response_saturated_zero_forcing_zero_ic``: no
  ``tau_lost``, no IC -> state stays at zero (no spurious clip
  injection).
* ``test_pulse_response_saturated_tight_sway_cap_winds_integrator_up``:
  with a sway cap below the steady-state ``b_hat`` magnitude the
  controller's PI integrator builds without bound while truth ``nu_y``
  drifts away -- the empirical signature of the bf8_q10_w45 spiral.
* ``test_pulse_response_saturated_per_axis_independence``: clipping
  only the sway axis leaves surge and yaw trajectories identical to
  the unclipped ones (within RK4 tolerance).
"""
from __future__ import annotations

import numpy as np
import pytest

from cqa.config import csov_default_config
from cqa.vessel import LinearVesselModel
from cqa.controller import LinearDpController
from cqa.transient_obs import (
    build_observer_augmented_system_full,
    pulse_response,
    pulse_response_saturated,
    implicit_tau_cmd,
    IDX_ETA, IDX_NU, IDX_ETA_HAT, IDX_NU_HAT, IDX_B_HAT,
    IDX_TAU_THR, IDX_INT,
    N_STATE,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _build_aug():
    """CSOV vessel + brucon regulator, same as p7_brucon_validation scripts."""
    cfg = csov_default_config()
    vessel = LinearVesselModel.from_config(cfg.vessel)
    omega_n = np.array([0.060, 0.080, 0.120])
    zeta = np.array([0.95, 0.95, 0.95])
    ctrl = LinearDpController.from_bandwidth(
        vessel.M, vessel.D, omega_n=omega_n, zeta=zeta
    )
    return build_observer_augmented_system_full(vessel, ctrl, T_thr=5.0)


# --------------------------------------------------------------------------- #
# Helper assertions                                                           #
# --------------------------------------------------------------------------- #
def _identity_clip(tau_raw: np.ndarray) -> np.ndarray:
    return np.asarray(tau_raw, dtype=float)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_implicit_tau_cmd_recovers_controller_law():
    """`implicit_tau_cmd(aug, x)` reproduces `-Kp eta_hat - Kd nu_hat - b_hat - Ki I`."""
    aug = _build_aug()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N_STATE)
    expected = (
        -aug.Kp @ x[IDX_ETA_HAT]
        - aug.Kd @ x[IDX_NU_HAT]
        - x[IDX_B_HAT]
        - aug.Ki @ x[IDX_INT]
    )
    actual = implicit_tau_cmd(aug, x)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_implicit_tau_cmd_matches_tau_thr_row_of_A():
    """Cross-check: (A @ x)[IDX_TAU_THR] = (1/T_thr) (tau_cmd_raw - tau_thr)."""
    aug = _build_aug()
    rng = np.random.default_rng(1)
    x = rng.standard_normal(N_STATE)
    expected_row = (1.0 / aug.T_thr) * (implicit_tau_cmd(aug, x) - x[IDX_TAU_THR])
    actual_row = (aug.A @ x)[IDX_TAU_THR]
    np.testing.assert_allclose(actual_row, expected_row, atol=1e-12)


def test_pulse_response_saturated_zero_forcing_zero_ic():
    """No forcing, no IC -> state stays at zero (no spurious clip injection)."""
    aug = _build_aug()
    t = np.linspace(0.0, 30.0, 301)
    tau_lost = np.zeros((len(t), 3))
    X = pulse_response_saturated(aug, t, tau_lost, _identity_clip)
    assert np.max(np.abs(X)) < 1e-12


def test_pulse_response_saturated_infinite_cap_matches_pulse_response():
    """Identity clip -> RK4 trajectory matches expm/trapezoidal pulse_response.

    Validates the (A + clip correction) decomposition: with a no-op clip
    the correction term must vanish, and RK4 of A x is an accurate
    integrator of x_dot = A x.
    """
    aug = _build_aug()
    t = np.linspace(0.0, 60.0, 1201)  # dt = 0.05 s for tight RK4 accuracy
    tau_env = np.array([3.0e5, -2.0e5, 1.0e6])  # generic kN-scale forcing
    # Build a step-like tau_lost pulse that decays exponentially -- matches
    # the wcfdi_scenario.build_pulse_inputs default.
    T_realloc = 5.0
    tau_lost_mag = -tau_env
    tau_lost = tau_lost_mag[None, :] * np.exp(-t / T_realloc)[:, None]
    x0 = np.zeros(N_STATE)
    x0[IDX_TAU_THR] = tau_lost_mag

    X_linear = pulse_response(aug, t, tau_lost, x0=x0)
    X_sat = pulse_response_saturated(aug, t, tau_lost, _identity_clip, x0=x0)

    # Compare eta (m), nu (m/s), eta_hat, nu_hat. These are the operator-
    # facing channels. Tolerance is loose because the two integrators
    # differ at O(dt^2) for the expm scheme and O(dt^4) for RK4 -- the
    # difference scales like dt^3 here (~1e-4).
    for sl, atol in [(IDX_ETA, 1e-4), (IDX_NU, 1e-4),
                     (IDX_ETA_HAT, 1e-4), (IDX_NU_HAT, 1e-4)]:
        np.testing.assert_allclose(
            X_sat[:, sl], X_linear[:, sl], atol=atol, rtol=1e-3,
            err_msg=f"Slice {sl} drifted under identity clip",
        )


def test_pulse_response_saturated_per_axis_independence():
    """Sway-only clip leaves surge and yaw trajectories unchanged.

    Construct a state evolution where the controller raw command on
    surge and yaw never hits the loose surge/yaw caps; clip only sway
    tightly. Trajectory rows for surge/yaw should match the un-clipped
    baseline; sway should diverge.
    """
    aug = _build_aug()
    t = np.linspace(0.0, 40.0, 801)
    # Sway-only env force, large enough to drive a tight sway cap.
    tau_env = np.array([0.0, -4.0e5, 0.0])
    tau_lost_mag = -tau_env
    tau_lost = tau_lost_mag[None, :] * np.exp(-t / 5.0)[:, None]
    x0 = np.zeros(N_STATE)
    x0[IDX_TAU_THR] = tau_lost_mag

    def clip_sway_only(tau_raw):
        c = np.asarray(tau_raw, dtype=float).copy()
        # Tight sway cap at 100 kN, loose elsewhere.
        c[1] = np.clip(c[1], -1.0e5, +1.0e5)
        return c

    X_baseline = pulse_response_saturated(aug, t, tau_lost, _identity_clip, x0=x0)
    X_clipped = pulse_response_saturated(aug, t, tau_lost, clip_sway_only, x0=x0)

    # Surge (idx 0) and yaw (idx 2) of eta should be ~unchanged.
    # Sway (idx 1) should diverge (clip is active).
    surge_diff = np.max(np.abs(X_clipped[:, 0] - X_baseline[:, 0]))
    yaw_diff = np.max(np.abs(X_clipped[:, 2] - X_baseline[:, 2]))
    sway_diff = np.max(np.abs(X_clipped[:, 1] - X_baseline[:, 1]))

    # Off-axis coupling is non-zero in the augmented system (yaw and surge
    # do leak into sway via the closed-loop matrix), but the *direct*
    # effect on surge/yaw of clipping sway is small. Allow 5 cm of leakage
    # on surge/yaw; require sway divergence > 50 cm.
    assert surge_diff < 0.05, f"sway clip leaked {surge_diff:.3f} m into surge"
    assert yaw_diff < 0.05, f"sway clip leaked {yaw_diff:.3f} rad into yaw"
    assert sway_diff > 0.5, f"sway clip had no effect on sway: {sway_diff:.3f} m"


def test_pulse_response_saturated_tight_sway_cap_winds_integrator_up():
    """Tight sway cap drives the spiral: PI integrator grows without bound,
    truth sway drifts toward port, mismatch between order and delivery
    persists. This is the bf8_q10_w45 mechanism (sec.12.21.21.6) in
    miniature.

    Construction: persistent sway env force of -500 kN, no WCFDI event
    (``tau_lost = 0``), tight sway cap at +/-300 kN. Initial condition
    is the intact SS that *would* hold against the env force if the cap
    were loose; the cap is what triggers the divergence.
    """
    aug = _build_aug()
    from cqa.transient_obs import intact_mean_steady_state_obs
    t = np.linspace(0.0, 120.0, 2401)
    tau_env = np.array([0.0, -5.0e5, 0.0])
    tau_lost = np.zeros((len(t), 3))
    # IC: where the intact closed loop would settle with no clip. Truth
    # SS is eta=nu=0; the truth row needs an extra env-force forcing to
    # close the balance (pulse_response_saturated does not add tau_env
    # internally; we encode it by setting tau_thr_ss such that the truth
    # force balance is satisfied only if tau_thr = -tau_env -- but since
    # we are not adding tau_env to nu_dot in this fixture, the cleanest
    # IC is x0 = 0 and we inject tau_env via tau_lost with opposite sign:
    # treat tau_env as a "constant disturbance" channel.
    #
    # Use B_lost (= +Minv on truth nu) to inject the env force as a
    # constant tau_lost, then set IC = 0. Cap on sway truncates the
    # controller's response.
    tau_lost = np.tile(tau_env[None, :], (len(t), 1))
    x0 = np.zeros(N_STATE)

    def tight_sway_cap(tau_raw):
        c = np.asarray(tau_raw, dtype=float).copy()
        # Cap sway thrust at 300 kN < 500 kN required for force balance.
        c[1] = np.clip(c[1], -3.0e5, +3.0e5)
        return c

    X = pulse_response_saturated(aug, t, tau_lost, tight_sway_cap, x0=x0)

    eta_y = X[:, 1]
    int_y = X[:, IDX_INT.start + 1]
    tau_thr_y = X[:, IDX_TAU_THR.start + 1]

    # Late-time drift: |eta_y(t=120)| >> |eta_y(t=30)|.
    assert abs(eta_y[-1]) > 2.0 * abs(eta_y[600]), (
        f"sway did not continue to drift: |eta_y(30s)|={abs(eta_y[600]):.2f}, "
        f"|eta_y(120s)|={abs(eta_y[-1]):.2f}"
    )

    # Drift direction matches env force (port-going env -> port drift).
    assert eta_y[-1] < 0.0, f"eta_y(t=120s) = {eta_y[-1]:.2f}, expected port-drift"

    # Delivered sway thrust is capped at +/-300 kN (with small RK4 noise).
    assert tau_thr_y.max() <= 3.0e5 + 1.0e3, "tau_thr exceeded sway cap"
    assert tau_thr_y.min() >= -3.0e5 - 1.0e3, "tau_thr exceeded sway cap"

    # PI integrator winds up under sustained saturation.
    assert abs(int_y[-1]) > abs(int_y[600]), (
        "integrator did not continue to wind up under saturation"
    )


def test_pulse_response_saturated_validates_inputs():
    """Shape mismatches and non-uniform grids are rejected."""
    aug = _build_aug()
    t = np.linspace(0.0, 10.0, 101)

    # tau_lost wrong shape
    with pytest.raises(ValueError, match="tau_lost_t must be"):
        pulse_response_saturated(aug, t, np.zeros((50, 3)), _identity_clip)

    # Non-uniform grid
    t_bad = np.array([0.0, 1.0, 1.5, 3.0])
    with pytest.raises(ValueError, match="uniform"):
        pulse_response_saturated(aug, t_bad, np.zeros((4, 3)), _identity_clip)

    # Clip returns wrong shape
    def bad_clip(_):
        return np.zeros(2)

    with pytest.raises(ValueError, match="must return"):
        pulse_response_saturated(aug, t, np.zeros((101, 3)), bad_clip)
