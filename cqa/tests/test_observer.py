"""Tests for the observer-augmented closed-loop linearisation."""

from __future__ import annotations

import numpy as np
import pytest

from cqa import csov_default_config
from cqa.config import ObserverParams, wave_filter_zeta_n
from cqa.controller import LinearDpController
from cqa.observer import (
    build_observer_augmented_system,
    position_state_indices,
)
from cqa.closed_loop import state_covariance_freqdomain_general


_PDSTRIP_PATH = "/home/blofro/src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"


def _has_pdstrip() -> bool:
    from pathlib import Path
    return Path(_PDSTRIP_PATH).is_file()


def test_zeta_n_schedule_matches_sandbox_endpoints():
    # Sandbox at Tp=10.22 reports ζ_n = 0.107 (see sandbox print).
    assert abs(wave_filter_zeta_n(10.22) - 0.1073) < 1e-3
    # Endpoints clamp to (0.25, 0.10).
    assert wave_filter_zeta_n(20.0) == 0.25
    assert wave_filter_zeta_n(8.0) == 0.10


def test_observer_aug_state_layout_24_or_27():
    cfg = csov_default_config()
    from cqa.vessel import LinearVesselModel
    v = LinearVesselModel.from_config(cfg.vessel)
    ctrl = LinearDpController.from_bandwidth(
        v.M, v.D, omega_n=cfg.controller.omega_n, zeta=cfg.controller.zeta,
    )
    aug = build_observer_augmented_system(
        v, ctrl, cfg.observer, Tp=10.0, T_thr=5.0, include_integrator=False,
    )
    assert aug.n_state == 24
    assert aug.A.shape == (24, 24)
    assert aug.B_w.shape == (24, 3)
    assert aug.B_wf.shape == (24, 3)

    aug2 = build_observer_augmented_system(
        v, ctrl, cfg.observer, Tp=10.0, T_thr=5.0, include_integrator=True,
    )
    assert aug2.n_state == 27
    assert aug2.A.shape == (27, 27)


def test_observer_aug_is_stable_at_csov_defaults():
    cfg = csov_default_config()
    from cqa.vessel import LinearVesselModel
    v = LinearVesselModel.from_config(cfg.vessel)
    ctrl = LinearDpController.from_bandwidth(
        v.M, v.D, omega_n=cfg.controller.omega_n, zeta=cfg.controller.zeta,
    )
    for Tp in (6.0, 10.0, 14.0, 18.0):
        aug = build_observer_augmented_system(
            v, ctrl, cfg.observer, Tp=Tp,
            T_thr=cfg.controller.thruster_time_constant_s,
            include_integrator=True,
        )
        eig = np.linalg.eigvals(aug.A)
        # All eigenvalues must have non-positive real part (allowing
        # the integrator pole to sit at 0 if Ki=0; here Ki>0 so the
        # integrator pole moves into the LHP via the closed-loop coupling).
        assert eig.real.max() < 1e-6, (
            f"Unstable observer-aug at Tp={Tp}: "
            f"max real eigvalue = {eig.real.max():.3e}"
        )


def test_block_diagonal_in_dof_for_zero_off_diagonal_gains():
    """When Kp, Kd, M, D, K_a1, K_b1 are all diagonal (the brucon case),
    the observer-aug should split into 3 independent DOF blocks. We test
    this by checking that surge-state perturbations don't propagate into
    sway/yaw rows of A (and vice versa), modulo the integrator block
    which is intentionally diagonal in DOF.
    """
    cfg = csov_default_config()
    from cqa.vessel import LinearVesselModel
    v = LinearVesselModel.from_config(cfg.vessel)
    ctrl = LinearDpController.from_bandwidth(
        v.M, v.D, omega_n=cfg.controller.omega_n, zeta=cfg.controller.zeta,
    )
    aug = build_observer_augmented_system(
        v, ctrl, cfg.observer, Tp=10.0, T_thr=5.0, include_integrator=True,
    )
    A = aug.A
    # For each DOF k in {0, 1, 2}, build the index list of all states
    # belonging to that DOF: every state-block stride 3 starting at i+k.
    n_state = aug.n_state
    blocks_per_dof = n_state // 3
    for k in range(3):
        own_idx = np.array([3 * b + k for b in range(blocks_per_dof)])
        other_idx = np.setdiff1d(np.arange(n_state), own_idx)
        # Off-block rows of A (i.e. rows belonging to DOF != k) should
        # have zero coupling from columns belonging to DOF k.
        cross = A[np.ix_(other_idx, own_idx)]
        assert np.max(np.abs(cross)) < 1e-12, (
            f"DOF {k} couples to DOFs others through {np.max(np.abs(cross))}"
        )


@pytest.mark.skipif(not _has_pdstrip(), reason="pdstrip dat not available")
def test_sigma_y_matches_brucon_at_p7_test_sea_state():
    """Cross-check: cqa observer-aug σ_eta_e at HS=4.20 m, Tp=10.22 s,
    beam-on must reproduce brucon empirical σ_y = 0.69 m to within
    ±20 % at csov_default tuning, and to within ±10 % when the thrust
    lag is calibrated to the sandbox's σ-closure value (T_thr=2 s).

    Sandbox numbers for cross-reference (from
    `scripts/p7_brucon_validation/sandbox_passive_observer.py`):
      perfect-FB no-integrator   0.254 m
      observer-only              0.561 m
      observer + WF              0.621 m
      observer + WF + integrator 0.638 m
      + thrust lag T_thr=2 s     0.673 m  ← best match
      + thrust lag T_thr=5 s     0.798 m  (cqa default; pessimistic)

    Brucon empirical median (30 long-run seeds, late window): 0.69 m.

    This test guards against future regressions in any of:
    - drift PSD normalisation (one-sided rad/s, no /π)
    - observer state layout / sign conventions
    - integrator coupling
    - thrust-lag block.
    """
    from cqa.vessel import LinearVesselModel
    from cqa.drift import slow_drift_force_psd_newman_pdstrip
    from cqa.rao import load_pdstrip_rao

    cfg = csov_default_config()
    v = LinearVesselModel.from_config(cfg.vessel)
    ctrl = LinearDpController.from_bandwidth(
        v.M, v.D, omega_n=cfg.controller.omega_n, zeta=cfg.controller.zeta,
    )
    rao = load_pdstrip_rao(_PDSTRIP_PATH)
    S_drift = slow_drift_force_psd_newman_pdstrip(
        rao_table=rao, Hs=4.2, Tp=10.22, theta_wave_rel=np.pi / 2,
    )
    BRUCON_TARGET = 0.69

    # csov_default tuning: ±20% (T_thr=5 is conservative).
    aug_default = build_observer_augmented_system(
        v, ctrl, cfg.observer, Tp=10.22,
        T_thr=cfg.controller.thruster_time_constant_s,
        include_integrator=True,
    )
    P = state_covariance_freqdomain_general(
        aug_default.A, aug_default.B_w, [S_drift],
        omega_lo=1e-4, omega_hi=1.0, n_points=2048,
    )
    sigma_default = float(np.sqrt(P[1, 1]))
    rel_err_default = abs(sigma_default - BRUCON_TARGET) / BRUCON_TARGET
    assert rel_err_default < 0.20, (
        f"csov_default σ_eta_e = {sigma_default:.3f} m, "
        f"brucon = {BRUCON_TARGET:.3f} m, rel_err = {rel_err_default*100:.1f}% (limit 20%)"
    )

    # Sandbox-calibrated thrust lag: ±10%.
    aug_calib = build_observer_augmented_system(
        v, ctrl, cfg.observer, Tp=10.22,
        T_thr=2.0,
        include_integrator=True,
    )
    P2 = state_covariance_freqdomain_general(
        aug_calib.A, aug_calib.B_w, [S_drift],
        omega_lo=1e-4, omega_hi=1.0, n_points=2048,
    )
    sigma_calib = float(np.sqrt(P2[1, 1]))
    rel_err_calib = abs(sigma_calib - BRUCON_TARGET) / BRUCON_TARGET
    assert rel_err_calib < 0.10, (
        f"calibrated (T_thr=2s) σ_eta_e = {sigma_calib:.3f} m, "
        f"brucon = {BRUCON_TARGET:.3f} m, rel_err = {rel_err_calib*100:.1f}% (limit 10%)"
    )


def test_position_state_indices_layout():
    idx = position_state_indices()
    assert idx["eta"] == slice(0, 3)
    assert idx["nu"] == slice(3, 6)
    assert idx["b_hat"] == slice(6, 9)
    assert idx["tau_thr"] == slice(9, 12)
    assert idx["eta_hat_LF"] == slice(12, 15)
    assert idx["nu_hat"] == slice(15, 18)
    assert idx["xi_w"] == slice(18, 21)
    assert idx["eta_hat_w"] == slice(21, 24)
    assert idx["I"] == slice(24, 27)


def test_position_state_indices_omits_integrator_when_absent():
    cfg = csov_default_config()
    from cqa.vessel import LinearVesselModel
    v = LinearVesselModel.from_config(cfg.vessel)
    ctrl = LinearDpController.from_bandwidth(
        v.M, v.D, omega_n=cfg.controller.omega_n, zeta=cfg.controller.zeta,
    )
    aug = build_observer_augmented_system(
        v, ctrl, cfg.observer, Tp=10.0, T_thr=5.0, include_integrator=False,
    )
    idx = position_state_indices(aug)
    assert "I" not in idx
    assert idx["eta_hat_w"] == slice(21, 24)


def test_zero_drift_gives_zero_steady_state_sigma():
    """Sanity: zero disturbance ⇒ σ = 0 (numerical noise only)."""
    cfg = csov_default_config()
    from cqa.vessel import LinearVesselModel
    v = LinearVesselModel.from_config(cfg.vessel)
    ctrl = LinearDpController.from_bandwidth(
        v.M, v.D, omega_n=cfg.controller.omega_n, zeta=cfg.controller.zeta,
    )
    aug = build_observer_augmented_system(
        v, ctrl, cfg.observer, Tp=10.0, T_thr=5.0,
        include_integrator=True,
    )
    omega = np.logspace(-4, 0, 256)

    def S_zero(_w):
        return np.zeros((3, 3))

    P = state_covariance_freqdomain_general(
        aug.A, aug.B_w, [S_zero],
        omega_lo=1e-4, omega_hi=1.0, n_points=256,
    )
    assert np.max(np.abs(P)) < 1e-20
