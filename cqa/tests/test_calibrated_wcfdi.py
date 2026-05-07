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


# ---------------------------------------------------------------------------
# tau_lost pulse plumbing (G2 step 4 / sec.12.21.7)
# ---------------------------------------------------------------------------


class TestTauLostPulse:
    """Unit tests for the measured tau_lost(t) injection in the calibrated
    transient/MC paths."""

    @pytest.fixture
    def cfg(self):
        return csov_default_config()

    @pytest.fixture
    def scenario(self):
        return WcfdiScenario(
            alpha=(0.5, 0.5, 0.5),
            gamma_immediate=0.3,
            T_realloc=10.0,
        )

    @pytest.fixture
    def joint(self, cfg):
        L0 = 0.5 * (cfg.gangway.telescope_min + cfg.gangway.telescope_max)
        return GangwayJointState(h=15.0, alpha_g=0.0, beta_g=0.0, L=L0)

    def _build_ctx(self, cfg, **overrides):
        kwargs = dict(
            sigma_measured_lf_body=[0.3, 0.5, np.deg2rad(0.3)],
            tau_env_measured=[0.0, -100000.0, 0.0],
            Vw_mean=10.0, Hs=4.0, Tp=10.0, Vc=0.5,
            theta_rel=np.deg2rad(90.0),
        )
        kwargs.update(overrides)
        return build_calibrated_context(cfg, **kwargs)

    def test_default_tau_lost_zero_preserves_baseline_mean(self, cfg, scenario):
        """With tau_lost_pre_wcf left at default (zeros), the calibrated
        transient mean should be identical to the no-pulse path. The
        cqa fixture uses tau_env reachable post-WCF, so the mean is
        the intact fixed point (flat trajectory)."""
        ctx = self._build_ctx(cfg)  # default tau_lost_pre_wcf = zeros
        res = wcfdi_transient_calibrated(cfg, scenario, ctx, t_end=60.0, n_t=61)
        # Should be flat at the intact fixed point: eta_mean[0] ~ eta_mean[t]
        # for all t (within ODE tolerance).
        np.testing.assert_allclose(
            res.eta_mean - res.eta_mean[0:1, :], 0.0, atol=1e-3,
        )
        # info.calibration block should report zero pulse and the configured shape
        cal = res.info["calibration"]
        np.testing.assert_array_equal(cal["tau_lost_pre_wcf"], np.zeros(3))
        assert cal["tau_lost_pulse_shape"] == "linear_decay"
        assert cal["tau_lost_duration_s"] == 5.0

    def test_nonzero_tau_lost_drives_mean_response(self, cfg, scenario):
        """A 100 kN sway tau_lost pulse should produce a clearly non-zero
        sway transient (more than a few cm)."""
        ctx_zero = self._build_ctx(cfg)
        ctx_pulse = self._build_ctx(
            cfg,
            tau_lost_pre_wcf=[0.0, 100_000.0, 0.0],
            tau_lost_pulse_shape="square",
            tau_lost_duration_s=10.0,
        )
        res_zero = wcfdi_transient_calibrated(cfg, scenario, ctx_zero, t_end=60.0, n_t=61)
        res_pulse = wcfdi_transient_calibrated(cfg, scenario, ctx_pulse, t_end=60.0, n_t=61)
        sway_pulse = res_pulse.eta_mean[:, 1] - res_pulse.eta_mean[0, 1]
        sway_zero = res_zero.eta_mean[:, 1] - res_zero.eta_mean[0, 1]
        # Pulse-driven sway transient should be at least 10 cm peak.
        assert np.max(np.abs(sway_pulse)) > 0.1, (
            f"100 kN sway pulse over 10 s should drive a > 10 cm sway transient, "
            f"got max |sway| = {np.max(np.abs(sway_pulse)):.4f} m"
        )
        # And it should be much larger than the no-pulse case.
        assert np.max(np.abs(sway_pulse)) > 5 * np.max(np.abs(sway_zero) + 1e-6)
        # Sign: positive tau_lost_y subtracts from effective force, so
        # vessel drifts toward NEGATIVE sway (-Minv @ tau_lost on nu).
        idx_peak = int(np.argmax(np.abs(sway_pulse)))
        assert sway_pulse[idx_peak] < 0, (
            f"Expected negative sway peak from positive tau_lost_y, got {sway_pulse[idx_peak]}"
        )

    def test_square_vs_linear_decay_amplitude_ordering(self, cfg, scenario):
        """For the same peak amplitude and duration, a square pulse delivers
        2x the impulse of a linear-decay pulse, so the position response
        should be larger."""
        ctx_sq = self._build_ctx(
            cfg,
            tau_lost_pre_wcf=[0.0, 100_000.0, 0.0],
            tau_lost_pulse_shape="square",
            tau_lost_duration_s=10.0,
        )
        ctx_ld = self._build_ctx(
            cfg,
            tau_lost_pre_wcf=[0.0, 100_000.0, 0.0],
            tau_lost_pulse_shape="linear_decay",
            tau_lost_duration_s=10.0,
        )
        res_sq = wcfdi_transient_calibrated(cfg, scenario, ctx_sq, t_end=60.0, n_t=61)
        res_ld = wcfdi_transient_calibrated(cfg, scenario, ctx_ld, t_end=60.0, n_t=61)
        peak_sq = np.max(np.abs(res_sq.eta_mean[:, 1] - res_sq.eta_mean[0, 1]))
        peak_ld = np.max(np.abs(res_ld.eta_mean[:, 1] - res_ld.eta_mean[0, 1]))
        # Square should be ~2x linear-decay (impulse ratio is exactly 2).
        # Allow some slack for closed-loop damping/response shape differences.
        assert peak_sq > 1.4 * peak_ld, (
            f"square pulse peak {peak_sq:.3f} m should be > 1.4x linear_decay {peak_ld:.3f} m"
        )

    def test_tau_lost_fn_method(self, cfg):
        """CalibratedContext.tau_lost_fn returns the right values at sample times."""
        ctx_sq = self._build_ctx(
            cfg,
            tau_lost_pre_wcf=[10.0, 20.0, 30.0],
            tau_lost_pulse_shape="square",
            tau_lost_duration_s=5.0,
        )
        # Inside window: full amplitude; outside: zero
        np.testing.assert_array_equal(ctx_sq.tau_lost_fn(0.0), [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(ctx_sq.tau_lost_fn(2.5), [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(ctx_sq.tau_lost_fn(4.999), [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(ctx_sq.tau_lost_fn(5.0), [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(ctx_sq.tau_lost_fn(6.0), [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(ctx_sq.tau_lost_fn(-1.0), [0.0, 0.0, 0.0])

        ctx_ld = self._build_ctx(
            cfg,
            tau_lost_pre_wcf=[10.0, 20.0, 30.0],
            tau_lost_pulse_shape="linear_decay",
            tau_lost_duration_s=10.0,
        )
        # At t=0: full amplitude; at t=5: half; at t=10: zero (boundary -> zero)
        np.testing.assert_allclose(ctx_ld.tau_lost_fn(0.0), [10.0, 20.0, 30.0])
        np.testing.assert_allclose(ctx_ld.tau_lost_fn(5.0), [5.0, 10.0, 15.0])
        np.testing.assert_allclose(ctx_ld.tau_lost_fn(10.0), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(ctx_ld.tau_lost_fn(11.0), [0.0, 0.0, 0.0])

    def test_invalid_pulse_shape_rejected(self, cfg):
        with pytest.raises(ValueError, match="tau_lost_pulse_shape"):
            self._build_ctx(cfg, tau_lost_pulse_shape="exponential")

    def test_invalid_tau_lost_shape_rejected(self, cfg):
        with pytest.raises(ValueError, match="tau_lost_pre_wcf"):
            self._build_ctx(cfg, tau_lost_pre_wcf=[10.0, 20.0])  # wrong shape

    def test_invalid_duration_rejected(self, cfg):
        with pytest.raises(ValueError, match="tau_lost_duration_s"):
            self._build_ctx(cfg, tau_lost_duration_s=-1.0)

    def test_mc_with_tau_lost_propagates_to_pos_peak(self, cfg, scenario, joint):
        """MC with a strong tau_lost pulse should yield larger pos_peak
        than MC with zero pulse."""
        ctx_zero = self._build_ctx(cfg)
        ctx_pulse = self._build_ctx(
            cfg,
            tau_lost_pre_wcf=[0.0, 200_000.0, 0.0],
            tau_lost_pulse_shape="square",
            tau_lost_duration_s=10.0,
        )
        res_zero = wcfdi_mc_calibrated(
            cfg, scenario, joint, ctx_zero,
            n_samples=64, t_end=80.0, n_t=81, rng_seed=11,
        )
        res_pulse = wcfdi_mc_calibrated(
            cfg, scenario, joint, ctx_pulse,
            n_samples=64, t_end=80.0, n_t=81, rng_seed=11,
        )
        med_zero = float(np.nanmedian(res_zero.pos_peak))
        med_pulse = float(np.nanmedian(res_pulse.pos_peak))
        assert med_pulse > 1.5 * med_zero, (
            f"Expected pulse-driven pos_peak median to be much larger than zero-pulse case, "
            f"got {med_zero:.3f} -> {med_pulse:.3f}"
        )


class TestPosCgFields:
    """The pos_cg_* fields expose body-frame CG radial deviation, which is
    what the brucon p7 validation harness compares against (sqrt(SurgeDev^2
    + SwayDev^2)).  pos_cg_* and pos_base_* differ by a heading-induced
    offset because the gangway base is offset from CG."""

    @pytest.fixture
    def cfg(self):
        return csov_default_config()

    @pytest.fixture
    def scenario(self):
        return WcfdiScenario(
            alpha=(0.5, 0.5, 0.5),
            gamma_immediate=0.3,
            T_realloc=30.0,
        )

    @pytest.fixture
    def joint(self, cfg):
        L0 = 0.5 * (cfg.gangway.telescope_min + cfg.gangway.telescope_max)
        return GangwayJointState(h=15.0, alpha_g=0.0, beta_g=0.0, L=L0)

    def test_pos_cg_fields_present_and_finite(self, cfg, scenario, joint):
        """Both fields are populated with finite values and have the
        expected shapes."""
        ctx = build_calibrated_context(
            cfg,
            sigma_measured_lf_body=[0.3, 0.5, np.deg2rad(0.2)],
            tau_env_measured=np.array([0.0, -100_000.0, 0.0]),
            Vw_mean=10.0, Hs=4.0, Tp=10.0, Vc=0.5,
            theta_rel=np.deg2rad(90.0),
        )
        n_samples, n_t = 32, 81
        res = wcfdi_mc_calibrated(
            cfg, scenario, joint, ctx,
            n_samples=n_samples, t_end=80.0, n_t=n_t, rng_seed=7,
        )
        assert res.pos_cg_traj.shape == (n_samples, n_t)
        assert res.pos_cg_peak.shape == (n_samples,)
        finite_mask = np.isfinite(res.pos_cg_peak)
        assert finite_mask.sum() > 0.9 * n_samples
        # Non-negative magnitudes
        assert np.all(res.pos_cg_traj[finite_mask] >= 0.0)
        assert np.all(res.pos_cg_peak[finite_mask] >= 0.0)
        # Peak >= any time-series value (consistency)
        assert np.all(
            res.pos_cg_peak[finite_mask]
            >= res.pos_cg_traj[finite_mask].max(axis=1) - 1e-12
        )

    def test_pos_cg_starts_at_zero(self, cfg, scenario, joint):
        """pos_cg_traj is delta-from-WCF radial, so column 0 must be exactly
        zero per realisation."""
        ctx = build_calibrated_context(
            cfg,
            sigma_measured_lf_body=[0.4, 0.6, np.deg2rad(0.2)],
            tau_env_measured=np.array([0.0, -50_000.0, 0.0]),
            Vw_mean=10.0, Hs=4.0, Tp=10.0, Vc=0.5,
            theta_rel=np.deg2rad(90.0),
        )
        res = wcfdi_mc_calibrated(
            cfg, scenario, joint, ctx,
            n_samples=32, t_end=80.0, n_t=81, rng_seed=13,
        )
        finite = np.isfinite(res.pos_cg_peak)
        assert np.all(res.pos_cg_traj[finite, 0] == 0.0), (
            "pos_cg_traj must start at zero per realisation"
        )

    def test_pos_cg_responds_to_tau_lost_pulse(self, cfg, scenario, joint):
        """A strong tau_lost pulse should drive pos_cg_peak distribution
        upward (delta-from-WCF radial grows when the system is kicked)."""
        ctx_zero = build_calibrated_context(
            cfg,
            sigma_measured_lf_body=[0.3, 0.4, np.deg2rad(0.2)],
            tau_env_measured=np.array([0.0, -50_000.0, 0.0]),
            Vw_mean=10.0, Hs=4.0, Tp=10.0, Vc=0.5,
            theta_rel=np.deg2rad(90.0),
            tau_lost_pre_wcf=np.zeros(3),
        )
        ctx_pulse = build_calibrated_context(
            cfg,
            sigma_measured_lf_body=[0.3, 0.4, np.deg2rad(0.2)],
            tau_env_measured=np.array([0.0, -50_000.0, 0.0]),
            Vw_mean=10.0, Hs=4.0, Tp=10.0, Vc=0.5,
            theta_rel=np.deg2rad(90.0),
            tau_lost_pre_wcf=[0.0, 200_000.0, 0.0],
            tau_lost_pulse_shape="square",
            tau_lost_duration_s=10.0,
        )
        res_zero = wcfdi_mc_calibrated(
            cfg, scenario, joint, ctx_zero,
            n_samples=64, t_end=80.0, n_t=81, rng_seed=23,
        )
        res_pulse = wcfdi_mc_calibrated(
            cfg, scenario, joint, ctx_pulse,
            n_samples=64, t_end=80.0, n_t=81, rng_seed=23,
        )
        med_zero = float(np.nanmedian(res_zero.pos_cg_peak))
        med_pulse = float(np.nanmedian(res_pulse.pos_cg_peak))
        assert med_pulse > 1.5 * med_zero, (
            f"Expected pulse-driven pos_cg_peak median to be much larger than "
            f"zero-pulse case, got {med_zero:.3f} -> {med_pulse:.3f}"
        )


class TestPostWcfReInit:
    """Phase 2 plumbing: tau_thr_post_init_delta and T_thr_post_override_s
    on CalibratedContext.  These let the user inject a closed-loop
    re-initialisation (subtract a delta from the post-WCF initial thrust
    state) and override the post-WCF thruster lag time constant."""

    @pytest.fixture
    def cfg(self):
        return csov_default_config()

    @pytest.fixture
    def scenario(self):
        return WcfdiScenario(
            alpha=(0.5, 0.5, 0.5),
            gamma_immediate=0.3,
            T_realloc=30.0,
            T_thr_post=5.0,
        )

    @pytest.fixture
    def joint(self, cfg):
        L0 = 0.5 * (cfg.gangway.telescope_min + cfg.gangway.telescope_max)
        return GangwayJointState(h=15.0, alpha_g=0.0, beta_g=0.0, L=L0)

    def _build_ctx(self, cfg, **overrides):
        kw = dict(
            sigma_measured_lf_body=[0.2, 0.3, np.deg2rad(0.2)],
            tau_env_measured=np.array([0.0, -100_000.0, 0.0]),
            Vw_mean=10.0, Hs=4.0, Tp=10.0, Vc=0.5,
            theta_rel=np.deg2rad(90.0),
        )
        kw.update(overrides)
        return build_calibrated_context(cfg, **kw)

    def test_default_delta_is_zero(self, cfg):
        """Default tau_thr_post_init_delta is the zero vector."""
        ctx = self._build_ctx(cfg)
        assert np.allclose(ctx.tau_thr_post_init_delta, np.zeros(3))
        assert ctx.T_thr_post_override_s is None

    def test_init_delta_validates_shape(self, cfg):
        with pytest.raises((ValueError, TypeError, AssertionError)):
            self._build_ctx(cfg, tau_thr_post_init_delta=[1.0, 2.0])  # length 2

    def test_init_delta_drives_larger_pos_cg_peak(self, cfg, scenario, joint):
        """A non-zero tau_thr_post_init_delta perturbs the post-WCF thrust
        starting state and should yield larger CG peaks."""
        ctx_zero = self._build_ctx(cfg)
        ctx_kick = self._build_ctx(
            cfg,
            tau_thr_post_init_delta=np.array([0.0, 200_000.0, 0.0]),
        )
        res_zero = wcfdi_mc_calibrated(
            cfg, scenario, joint, ctx_zero,
            n_samples=48, t_end=80.0, n_t=81, rng_seed=31,
        )
        res_kick = wcfdi_mc_calibrated(
            cfg, scenario, joint, ctx_kick,
            n_samples=48, t_end=80.0, n_t=81, rng_seed=31,
        )
        med_zero = float(np.nanmedian(res_zero.pos_cg_peak))
        med_kick = float(np.nanmedian(res_kick.pos_cg_peak))
        assert med_kick > 1.3 * med_zero, (
            f"Non-zero tau_thr_post_init_delta should increase pos_cg_peak; "
            f"got {med_zero:.3f} -> {med_kick:.3f}"
        )

    def test_T_thr_post_override_takes_precedence(self, cfg, scenario, joint):
        """An explicit ctx.T_thr_post_override_s overrides
        scenario.T_thr_post.  Larger T_thr_post = slower thruster
        recovery = larger position excursion under the same kick."""
        kick = np.array([0.0, 200_000.0, 0.0])
        ctx_short = self._build_ctx(
            cfg,
            tau_thr_post_init_delta=kick,
            T_thr_post_override_s=2.0,
        )
        ctx_long = self._build_ctx(
            cfg,
            tau_thr_post_init_delta=kick,
            T_thr_post_override_s=20.0,
        )
        res_short = wcfdi_mc_calibrated(
            cfg, scenario, joint, ctx_short,
            n_samples=48, t_end=80.0, n_t=81, rng_seed=37,
        )
        res_long = wcfdi_mc_calibrated(
            cfg, scenario, joint, ctx_long,
            n_samples=48, t_end=80.0, n_t=81, rng_seed=37,
        )
        med_short = float(np.nanmedian(res_short.pos_cg_peak))
        med_long = float(np.nanmedian(res_long.pos_cg_peak))
        assert med_long > med_short, (
            f"Larger T_thr_post should yield larger pos_cg_peak; "
            f"got T=2s: {med_short:.3f} m, T=20s: {med_long:.3f} m"
        )
