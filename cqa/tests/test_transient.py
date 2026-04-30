"""Tests for the WCFDI transient predictor."""

from __future__ import annotations

import numpy as np
import pytest

from cqa import csov_default_config, wcfdi_transient, WcfdiScenario
from cqa.transient import (
    build_augmented_system,
    intact_mean_steady_state,
)
from cqa.vessel import LinearVesselModel
from cqa.controller import LinearDpController


def test_intact_steady_state_is_zero_eta():
    """In intact steady state with PD + bias FF, eta = nu = 0 and tau_thr = -tau_env."""
    cfg = csov_default_config()
    vessel = LinearVesselModel.from_config(cfg.vessel)
    controller = LinearDpController.from_bandwidth(vessel.M, vessel.D)
    aug = build_augmented_system(vessel, controller)
    tau_env = np.array([1.0e5, -2.0e5, 3.0e6])
    x_ss = intact_mean_steady_state(aug, tau_env)
    eta = x_ss[0:3]
    nu = x_ss[3:6]
    b_hat = x_ss[6:9]
    tau_thr = x_ss[9:12]
    assert np.allclose(eta, 0.0, atol=1e-6), f"eta should be 0, got {eta}"
    assert np.allclose(nu, 0.0, atol=1e-6), f"nu should be 0, got {nu}"
    assert np.allclose(b_hat, tau_env, atol=1e-3), f"b_hat should equal tau_env, got {b_hat}"
    assert np.allclose(tau_thr, -tau_env, atol=1e-3), (
        f"tau_thr should equal -tau_env, got {tau_thr}"
    )


def test_no_failure_means_no_transient():
    """alpha = 1.0 (no capability lost) should give zero deviation throughout."""
    cfg = csov_default_config()
    res = wcfdi_transient(
        cfg,
        Vw_mean=10.0,
        Hs=2.0,
        Tp=8.0,
        Vc=0.3,
        theta_rel=np.pi / 2.0,
        scenario=WcfdiScenario(alpha=(1.0, 1.0, 1.0)),
        t_end=60.0,
        n_t=61,
    )
    # eta_mean stays at zero throughout (mean steady state).
    assert np.max(np.abs(res.eta_mean)) < 1e-3, (
        f"max |eta_mean| = {np.max(np.abs(res.eta_mean))} should be ~0"
    )


def test_cqa_violated_flag_set_for_overload():
    """If env force exceeds post-failure cap, info should flag the violation."""
    cfg = csov_default_config()
    res = wcfdi_transient(
        cfg,
        Vw_mean=14.0,
        Hs=2.5,
        Tp=8.5,
        Vc=0.5,
        theta_rel=np.pi / 2.0,
        scenario=WcfdiScenario(alpha=(0.3, 0.3, 0.3)),
        t_end=60.0,
        n_t=61,
    )
    assert res.info["cqa_precondition_violated"][1], (
        "Expected sway DOF to violate CQA precondition for this scenario, "
        f"got {res.info['cqa_precondition_violated']}"
    )


def test_mean_trajectory_bounded_when_cqa_satisfied():
    """When the CQA precondition holds, the mean trajectory must remain
    bounded and decay back to (near) zero — never drift off."""
    cfg = csov_default_config()
    res = wcfdi_transient(
        cfg,
        Vw_mean=10.0,
        Hs=2.0,
        Tp=8.0,
        Vc=0.3,
        theta_rel=np.pi / 2.0,
        scenario=WcfdiScenario(alpha=(0.6, 0.6, 0.6)),
        t_end=600.0,
        n_t=601,
    )
    # Sanity: precondition holds.
    assert not np.any(res.info["cqa_precondition_violated"]), (
        f"Test precondition: CQA must hold; got "
        f"{res.info['cqa_precondition_violated']}, tau_env={res.info['tau_env']}, "
        f"cap_post={res.info['tau_cap_post']}"
    )
    # Mean trajectory must stay bounded over 10 minutes.
    max_eta_e = float(np.max(np.abs(res.eta_mean[:, 1])))
    assert max_eta_e < 5.0, (
        f"Mean sway excursion {max_eta_e:.2f} m too large for CQA-guarded scenario"
    )
    # Late-time mean should have decayed back near zero.
    late_eta_e = float(np.abs(res.eta_mean[-1, 1]))
    assert late_eta_e < 0.5 * max(max_eta_e, 1e-3) or late_eta_e < 0.05, (
        f"Mean trajectory did not decay: peak={max_eta_e:.3f}, late={late_eta_e:.3f}"
    )


def test_covariance_bounded_when_cqa_satisfied():
    """Under the CQA precondition the variance envelope must stay bounded."""
    cfg = csov_default_config()
    res = wcfdi_transient(
        cfg,
        Vw_mean=10.0,
        Hs=2.0,
        Tp=8.0,
        Vc=0.3,
        theta_rel=np.pi / 2.0,
        scenario=WcfdiScenario(alpha=(0.6, 0.6, 0.6)),
        t_end=600.0,
        n_t=301,
    )
    assert not np.any(res.info["cqa_precondition_violated"])
    sigma_e_init = res.eta_std[0, 1]
    sigma_e_max = float(np.max(res.eta_std[:, 1]))
    # Covariance can change between the 6-state and 12-state models, but
    # must stay finite and within an order of magnitude of the initial.
    assert sigma_e_max < 10.0 * max(sigma_e_init, 0.05), (
        f"Variance grew unbounded: max sigma_e={sigma_e_max:.3f} vs init {sigma_e_init:.3f}"
    )
