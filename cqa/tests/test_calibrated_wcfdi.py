"""Unit tests for cqa.calibrated_wcfdi.

Coverage:
  - rescale_covariance_diagonal: PSD preservation, diagonal target,
    correlation preservation, edge cases.
  - build_calibrated_context: input validation, shape contracts,
    measured-tau_env propagation.
  - wcfdi_transient_calibrated and wcfdi_mc_calibrated: sanity check
    that calibration changes the right things and leaves the model
    invariants intact when sigma_measured == sigma_model.
"""

from __future__ import annotations

import numpy as np
import pytest

from cqa.calibrated_wcfdi import (
    rescale_covariance_diagonal,
    build_calibrated_context,
    wcfdi_transient_calibrated,
    wcfdi_mc_calibrated,
)
from cqa.config import csov_default_config
from cqa.transient import WcfdiScenario
from cqa.gangway import GangwayJointState


# ---------------------------------------------------------------------------
# rescale_covariance_diagonal
# ---------------------------------------------------------------------------


class TestRescaleCovarianceDiagonal:
    """Analytical checks for the diagonal-rescale helper."""

    def test_target_diagonals_match_exactly(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((6, 6))
        P = A @ A.T + 0.01 * np.eye(6)  # PSD
        sigma_target = np.array([0.5, 1.5, 2.0])
        P_cal, d = rescale_covariance_diagonal(
            P, sigma_target, indices=(0, 1, 2),
        )
        np.testing.assert_allclose(np.sqrt(np.diag(P_cal)[:3]), sigma_target, rtol=1e-12)

    def test_unrescaled_diagonals_unchanged(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((6, 6))
        P = A @ A.T + 0.01 * np.eye(6)
        sigma_target = np.array([0.5, 1.5])
        P_cal, _ = rescale_covariance_diagonal(P, sigma_target, indices=(0, 1))
        # Indices 2..5 keep their model diagonal (d=1 there)
        np.testing.assert_allclose(np.diag(P_cal)[2:], np.diag(P)[2:], rtol=1e-12)

    def test_psd_preserved(self):
        """D · P · D is PSD whenever P is PSD and D is positive diagonal."""
        rng = np.random.default_rng(1)
        A = rng.standard_normal((6, 6))
        P = A @ A.T  # PSD
        sigma_target = np.array([3.0, 0.1, 5.0])  # extreme rescale
        P_cal, _ = rescale_covariance_diagonal(P, sigma_target, indices=(0, 1, 2))
        eigs = np.linalg.eigvalsh(P_cal)
        assert eigs.min() >= -1e-10  # FP-tolerance

    def test_symmetry_preserved(self):
        rng = np.random.default_rng(2)
        A = rng.standard_normal((6, 6))
        P = A @ A.T + 0.01 * np.eye(6)
        sigma_target = np.array([0.7, 1.3, 2.1])
        P_cal, _ = rescale_covariance_diagonal(P, sigma_target, indices=(0, 1, 2))
        assert np.allclose(P_cal, P_cal.T, atol=1e-14)

    def test_correlation_matrix_preserved_within_rescaled_block(self):
        """For pairs (i, j) both in `indices`, the correlation coefficient
        rho_ij = P_ij / sqrt(P_ii P_jj) is invariant under D · P · D.
        """
        rng = np.random.default_rng(3)
        A = rng.standard_normal((4, 4))
        P = A @ A.T + 0.01 * np.eye(4)
        sigma_target = np.array([2.0, 0.5])
        P_cal, _ = rescale_covariance_diagonal(P, sigma_target, indices=(0, 1))

        rho_orig = P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
        rho_cal = P_cal[0, 1] / np.sqrt(P_cal[0, 0] * P_cal[1, 1])
        assert abs(rho_orig - rho_cal) < 1e-12

    def test_identity_when_target_equals_model(self):
        rng = np.random.default_rng(4)
        A = rng.standard_normal((6, 6))
        P = A @ A.T + 0.01 * np.eye(6)
        sigma_model_eta = np.sqrt(np.diag(P)[:3])
        P_cal, d = rescale_covariance_diagonal(
            P, sigma_model_eta, indices=(0, 1, 2),
        )
        np.testing.assert_allclose(P_cal, P, atol=1e-12)
        np.testing.assert_allclose(d, np.ones(6), atol=1e-12)

    def test_rejects_nonsquare(self):
        with pytest.raises(ValueError, match="square"):
            rescale_covariance_diagonal(
                np.zeros((3, 4)), np.array([1.0]), indices=(0,),
            )

    def test_rejects_size_mismatch(self):
        with pytest.raises(ValueError, match="indices"):
            rescale_covariance_diagonal(
                np.eye(3), np.array([1.0, 2.0]), indices=(0,),
            )

    def test_rejects_negative_sigma(self):
        with pytest.raises(ValueError, match="non-negative"):
            rescale_covariance_diagonal(
                np.eye(3), np.array([-0.5]), indices=(0,),
            )

    def test_rejects_out_of_range_index(self):
        with pytest.raises(ValueError, match="out-of-range"):
            rescale_covariance_diagonal(
                np.eye(3), np.array([1.0]), indices=(7,),
            )

    def test_zero_model_diagonal_handled(self):
        """If a model diagonal is zero, the rescale just sets the
        target diagonal directly (cross-terms in that row/col stay
        zero, which matches the model's belief)."""
        P = np.diag([0.0, 4.0, 9.0])
        sigma_target = np.array([2.0])
        P_cal, _ = rescale_covariance_diagonal(P, sigma_target, indices=(0,))
        assert np.isclose(P_cal[0, 0], 4.0)
        assert np.isclose(P_cal[0, 1], 0.0)
        assert np.isclose(P_cal[1, 1], 4.0)
        assert np.isclose(P_cal[2, 2], 9.0)


# ---------------------------------------------------------------------------
# build_calibrated_context
# ---------------------------------------------------------------------------


class TestBuildCalibratedContext:
    """Shape, validation, and semantics of the calibrated context builder."""

    def test_shapes_and_substitution(self):
        cfg = csov_default_config()
        sigma_meas = np.array([0.4, 0.9, np.deg2rad(0.3)])
        tau_env_meas = np.array([10000.0, -200000.0, 50000.0])
        ctx = build_calibrated_context(
            cfg,
            sigma_measured_lf_body=sigma_meas,
            tau_env_measured=tau_env_meas,
            Vw_mean=10.0, Hs=4.0, Tp=10.0, Vc=0.5,
            theta_rel=np.deg2rad(45.0),
        )
        # tau_env was substituted exactly
        np.testing.assert_array_equal(ctx.tau_env_used, tau_env_meas)
        # P6 diagonals at η match measured
        np.testing.assert_allclose(
            np.sqrt(np.diag(ctx.P6_calibrated)[:3]), sigma_meas, rtol=1e-10,
        )
        # P12 diagonals at η also match
        np.testing.assert_allclose(
            np.sqrt(np.diag(ctx.P12_calibrated)[:3]), sigma_meas, rtol=1e-10,
        )
        # Velocity block in P6 is unchanged from model
        np.testing.assert_allclose(
            np.diag(ctx.P6_calibrated)[3:], np.diag(ctx.P6_model)[3:], rtol=1e-10,
        )
        # Sanity on shapes
        assert ctx.aug.A.shape == (12, 12)
        assert ctx.x_ss_intact.shape == (12,)
        assert ctx.P6_calibrated.shape == (6, 6)
        assert ctx.P12_calibrated.shape == (12, 12)

    def test_input_validation(self):
        cfg = csov_default_config()
        with pytest.raises(ValueError, match="sigma_measured_lf_body"):
            build_calibrated_context(
                cfg, sigma_measured_lf_body=[0.1, 0.2],  # wrong shape
                tau_env_measured=[0.0, 0.0, 0.0],
            )
        with pytest.raises(ValueError, match="tau_env_measured"):
            build_calibrated_context(
                cfg, sigma_measured_lf_body=[0.1, 0.2, 0.3],
                tau_env_measured=[0.0, 0.0],  # wrong shape
            )
        with pytest.raises(ValueError, match="non-negative"):
            build_calibrated_context(
                cfg, sigma_measured_lf_body=[-0.1, 0.2, 0.3],
                tau_env_measured=[0.0, 0.0, 0.0],
            )

    def test_x_ss_intact_responds_to_tau_env(self):
        """Different tau_env -> different intact mean steady state."""
        cfg = csov_default_config()
        sigma_meas = [0.3, 0.6, np.deg2rad(0.3)]
        ctx_a = build_calibrated_context(
            cfg, sigma_measured_lf_body=sigma_meas,
            tau_env_measured=[0.0, 0.0, 0.0],
        )
        ctx_b = build_calibrated_context(
            cfg, sigma_measured_lf_body=sigma_meas,
            tau_env_measured=[0.0, -300000.0, 0.0],
        )
        # The integrator state b_hat[7] (sway) absorbs the steady force
        # under intact closed loop -> different x_ss_intact in that
        # component at minimum.
        assert not np.allclose(ctx_a.x_ss_intact, ctx_b.x_ss_intact)


# ---------------------------------------------------------------------------
# wcfdi_transient_calibrated and wcfdi_mc_calibrated
# ---------------------------------------------------------------------------


class TestCalibratedTransientPipeline:
    """End-to-end smoke + invariance tests."""

    @pytest.fixture
    def cfg(self):
        return csov_default_config()

    @pytest.fixture
    def scenario(self):
        # Aggressive-enough scenario: half thrust ceiling, immediate at 30%
        return WcfdiScenario(
            alpha=(0.5, 0.5, 0.5),
            gamma_immediate=0.3,
            T_realloc=30.0,
        )

    @pytest.fixture
    def joint(self, cfg):
        L0 = 0.5 * (cfg.gangway.telescope_min + cfg.gangway.telescope_max)
        return GangwayJointState(h=15.0, alpha_g=0.0, beta_g=0.0, L=L0)

    def test_calibrated_transient_runs_and_returns_valid_result(self, cfg, scenario):
        sigma_meas = np.array([0.4, 0.9, np.deg2rad(0.3)])
        tau_env_meas = np.array([5000.0, -200000.0, 30000.0])
        ctx = build_calibrated_context(
            cfg,
            sigma_measured_lf_body=sigma_meas,
            tau_env_measured=tau_env_meas,
            Vw_mean=10.0, Hs=4.0, Tp=10.0, Vc=0.5,
            theta_rel=np.deg2rad(45.0),
        )
        res = wcfdi_transient_calibrated(
            cfg, scenario, ctx, t_end=120.0, n_t=121,
        )
        assert res.t.shape == (121,)
        assert res.eta_mean.shape == (121, 3)
        assert res.eta_std.shape == (121, 3)
        # Initial sigma envelope matches the calibrated P6 diagonals
        np.testing.assert_allclose(res.eta_std[0], sigma_meas, rtol=1e-8)
        # Calibration block is in info
        assert res.info["tau_env_source"] == "measured"
        assert "calibration" in res.info
        np.testing.assert_array_equal(
            res.info["calibration"]["sigma_measured_lf_body"], sigma_meas,
        )

    def test_calibrated_mc_runs_and_pos_peak_responds_to_sigma(self, cfg, scenario, joint):
        """Doubling the measured sigma_y should increase pos_peak."""
        tau_env_meas = np.array([0.0, -100000.0, 0.0])
        ctx_low = build_calibrated_context(
            cfg,
            sigma_measured_lf_body=[0.2, 0.3, np.deg2rad(0.2)],
            tau_env_measured=tau_env_meas,
            Vw_mean=10.0, Hs=4.0, Tp=10.0, Vc=0.5,
            theta_rel=np.deg2rad(90.0),
        )
        ctx_hi = build_calibrated_context(
            cfg,
            sigma_measured_lf_body=[0.2, 0.6, np.deg2rad(0.2)],
            tau_env_measured=tau_env_meas,
            Vw_mean=10.0, Hs=4.0, Tp=10.0, Vc=0.5,
            theta_rel=np.deg2rad(90.0),
        )
        res_low = wcfdi_mc_calibrated(
            cfg, scenario, joint, ctx_low,
            n_samples=80, t_end=120.0, n_t=121, rng_seed=42,
        )
        res_hi = wcfdi_mc_calibrated(
            cfg, scenario, joint, ctx_hi,
            n_samples=80, t_end=120.0, n_t=121, rng_seed=42,
        )
        # Higher injected sway sigma -> larger pos_peak distribution.
        # Compare medians across the MC ensembles.
        med_low = float(np.nanmedian(res_low.pos_peak))
        med_hi = float(np.nanmedian(res_hi.pos_peak))
        assert med_hi > med_low, (
            f"Expected pos_peak median to grow with measured sigma_y, "
            f"got {med_low:.3f} -> {med_hi:.3f}"
        )

    def test_calibrated_mc_reduces_to_baseline_when_meas_eq_model(self, cfg, scenario, joint):
        """If we feed the model's own sigma + model's own tau_env through
        the calibrated path, the MC ensemble peaks should match the raw
        wcfdi_mc within MC noise (same rng_seed)."""
        from cqa.wcfdi_mc import wcfdi_mc, _build_operating_context

        # Pick a sea state, compute model context, then feed its own
        # (sigma_lf, tau_env) into the calibrated pipeline. Result should
        # match wcfdi_mc on the same seed.
        Vw, Hs, Tp, Vc, theta = 10.0, 4.0, 10.0, 0.5, np.deg2rad(45.0)
        model_ctx = _build_operating_context(
            cfg, Vw, Hs, Tp, Vc, theta,
        )
        sigma_model = np.sqrt(np.diag(model_ctx["P6_intact"])[:3])
        tau_env_model = model_ctx["tau_env"]

        ctx_cal = build_calibrated_context(
            cfg,
            sigma_measured_lf_body=sigma_model,
            tau_env_measured=tau_env_model,
            Vw_mean=Vw, Hs=Hs, Tp=Tp, Vc=Vc, theta_rel=theta,
        )
        # Sanity: P6_calibrated should equal P6_model exactly
        np.testing.assert_allclose(
            ctx_cal.P6_calibrated, ctx_cal.P6_model, rtol=1e-10,
        )

        # Run both pipelines with the same seed.
        res_raw = wcfdi_mc(
            cfg, Vw, Hs, Tp, Vc, theta, scenario, joint,
            n_samples=64, t_end=100.0, n_t=101, rng_seed=7,
        )
        res_cal = wcfdi_mc_calibrated(
            cfg, scenario, joint, ctx_cal,
            n_samples=64, t_end=100.0, n_t=101, rng_seed=7,
        )

        # pos_peak distributions should match (same starting states,
        # same propagation). Drop NaNs from both consistently.
        valid = (~np.isnan(res_raw.pos_peak)) & (~np.isnan(res_cal.pos_peak))
        np.testing.assert_allclose(
            res_cal.pos_peak[valid], res_raw.pos_peak[valid], rtol=1e-6, atol=1e-6,
        )
