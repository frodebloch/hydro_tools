"""Tests for cqa.online_estimator.

Covers:
* `closed_loop_decorrelation_time` for surge / sway / yaw / position.
* `BayesianSigmaEstimator`: empty-window prior collapse, posterior
  concentration on synthetic Gaussian / AR(1) data, effective-sample-size
  behaviour under perfect correlation, ring-buffer eviction, reset,
  credible-interval shrinkage, zero-mean vs sample-mean modes, validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cqa.config import ControllerParams, csov_default_config
from cqa.online_estimator import (
    BayesianSigmaEstimator,
    PosteriorHealth,
    RadialPosterior,
    SigmaPosterior,
    closed_loop_decorrelation_time,
    combine_radial_posterior,
)


# ---------------------------------------------------------------------------
# closed_loop_decorrelation_time
# ---------------------------------------------------------------------------


def test_decorr_surge_basic():
    c = ControllerParams(omega_n_surge=0.05, zeta_surge=0.8)
    T = closed_loop_decorrelation_time(c, "surge")
    assert T == pytest.approx(1.0 / (0.05 * 0.8))


def test_decorr_sway_yaw_basic():
    c = ControllerParams(
        omega_n_surge=0.06,
        omega_n_sway=0.04,
        omega_n_yaw=0.03,
        zeta_surge=0.9,
        zeta_sway=0.7,
        zeta_yaw=0.5,
    )
    assert closed_loop_decorrelation_time(c, "sway") == pytest.approx(
        1.0 / (0.04 * 0.7)
    )
    assert closed_loop_decorrelation_time(c, "yaw") == pytest.approx(
        1.0 / (0.03 * 0.5)
    )


def test_decorr_position_takes_max_of_xy():
    """The radial position channel is dominated by the slowest of
    surge / sway, so we want the LARGER decorrelation time."""
    c = ControllerParams(
        omega_n_surge=0.10,  # fast: T_x = 1/(0.10*0.8) = 12.5 s
        omega_n_sway=0.02,  # slow: T_y = 1/(0.02*0.8) = 62.5 s
        zeta_surge=0.8,
        zeta_sway=0.8,
    )
    T_pos = closed_loop_decorrelation_time(c, "position")
    assert T_pos == pytest.approx(1.0 / (0.02 * 0.8))


def test_decorr_position_csov_default():
    """Sanity check against the CSOV default config."""
    cfg = csov_default_config()
    T = closed_loop_decorrelation_time(cfg.controller, "position")
    # CSOV defaults: omega_n=(0.06,0.06,0.05), zeta=(0.9,0.9,0.9).
    # Both surge and sway give 1/(0.06*0.9) = 18.52 s.
    assert T == pytest.approx(1.0 / (0.06 * 0.9))


def test_decorr_invalid_axis_raises():
    c = ControllerParams()
    with pytest.raises(ValueError, match="axis must be one of"):
        closed_loop_decorrelation_time(c, "heave")


# ---------------------------------------------------------------------------
# BayesianSigmaEstimator: construction & validation
# ---------------------------------------------------------------------------


def _make_default_estimator(**kwargs) -> BayesianSigmaEstimator:
    """Default-configured estimator with prior_sigma2=1.0, T_decorr=10 s,
    dt=1 s, window=60 s. Override fields via kwargs."""
    defaults = dict(
        prior_sigma2=1.0,
        T_decorr_s=10.0,
        dt_s=1.0,
        prior_strength_n0=2.0,
        window_s=60.0,
        assume_zero_mean=True,
    )
    defaults.update(kwargs)
    return BayesianSigmaEstimator(**defaults)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(prior_sigma2=0.0),
        dict(prior_sigma2=-1.0),
        dict(T_decorr_s=0.0),
        dict(T_decorr_s=-1.0),
        dict(dt_s=0.0),
        dict(dt_s=-0.5),
        dict(prior_strength_n0=0.0),
        dict(prior_strength_n0=-2.0),
        dict(window_s=0.0),
    ],
)
def test_invalid_construction_raises(kwargs):
    with pytest.raises(ValueError):
        _make_default_estimator(**kwargs)


def test_window_smaller_than_dt_raises():
    with pytest.raises(ValueError, match="capacity"):
        BayesianSigmaEstimator(
            prior_sigma2=1.0,
            T_decorr_s=10.0,
            dt_s=2.0,
            window_s=1.0,
        )


def test_capacity_floor():
    est = _make_default_estimator(window_s=60.0, dt_s=1.0)
    assert est.n_raw_capacity == 60
    est2 = _make_default_estimator(window_s=60.0, dt_s=0.4)
    assert est2.n_raw_capacity == 150


# ---------------------------------------------------------------------------
# BayesianSigmaEstimator: empty window / prior collapse
# ---------------------------------------------------------------------------


def test_empty_window_posterior_equals_prior_mean():
    """With no samples, the posterior on sigma^2 should equal the prior
    mean (sigma2_mean ~= prior_sigma2)."""
    prior = 1.7
    est = _make_default_estimator(prior_sigma2=prior, prior_strength_n0=10.0)
    post = est.posterior()
    assert post.n_raw == 0
    assert post.n_eff == pytest.approx(0.0)
    # InvGamma(alpha_0=5, beta_0=prior*4) -> mean = prior.
    assert post.sigma2_mean == pytest.approx(prior)
    assert post.sigma_mean == pytest.approx(np.sqrt(prior))
    assert post.prior_sigma2 == pytest.approx(prior)


def test_default_n0_softclip_finite_prior_mean():
    """Default prior_strength_n0=2 corresponds to alpha_0=1 exactly,
    which has improper (infinite) mean. We soft-clip to alpha_0=1+eps
    so the prior mean is finite and ~= prior_sigma2 (within a few %)."""
    est = _make_default_estimator(prior_sigma2=2.0, prior_strength_n0=2.0)
    post = est.posterior()
    assert np.isfinite(post.sigma2_mean)
    # alpha_0 = 1 + 1e-6 -> mean = beta_0/eps -> very large unless beta_0
    # is tiny. We built beta_0 = prior_sigma2 * (alpha_0 - 1) = 2*1e-6,
    # so mean = 2e-6 / 1e-6 = 2.0.
    assert post.sigma2_mean == pytest.approx(2.0, rel=1e-3)


# ---------------------------------------------------------------------------
# BayesianSigmaEstimator: convergence on synthetic data
# ---------------------------------------------------------------------------


def test_posterior_concentrates_on_iid_gaussian():
    """Feed many IID N(0, sigma^2) samples with T_decorr ~ dt so that
    n_eff ~ n_raw. With a weak prior the posterior median should be
    close to the true sigma^2."""
    rng = np.random.default_rng(0xC0FFEE)
    true_sigma = 0.7
    true_sigma2 = true_sigma ** 2
    # Decorrelation time = dt -> n_eff = n_raw (no Bartlett discount).
    est = BayesianSigmaEstimator(
        prior_sigma2=10.0,  # deliberately wrong
        T_decorr_s=1.0,
        dt_s=1.0,
        prior_strength_n0=2.0,  # weak
        window_s=2000.0,  # 2000 IID samples
    )
    for _ in range(2000):
        est.update(rng.normal(0.0, true_sigma))
    post = est.posterior(credible=0.95)
    # With ~2000 IID samples the posterior should be tight around truth.
    assert post.n_eff == pytest.approx(2000.0)
    assert post.sigma2_median == pytest.approx(true_sigma2, rel=0.10)
    assert post.sigma_median == pytest.approx(true_sigma, rel=0.05)
    # Credible interval contains the truth.
    assert post.sigma2_lo < true_sigma2 < post.sigma2_hi


def test_perfectly_correlated_samples_n_eff_at_floor():
    """If all samples are identical (perfectly correlated), the Bartlett
    correction should keep n_eff at ~ window/T_decorr regardless of how
    many raw samples we push."""
    est = _make_default_estimator(
        prior_sigma2=1.0,
        T_decorr_s=20.0,
        dt_s=1.0,
        window_s=60.0,
    )
    for _ in range(60):  # fill the window
        est.update(0.5)
    # n_eff = n_raw * dt / max(dt, T_decorr) = 60 * 1 / 20 = 3.0.
    assert est.n_raw == 60
    assert est.n_eff == pytest.approx(3.0)


def test_credible_interval_shrinks_with_more_data():
    """As n_eff grows the equal-tail credible interval on sigma^2 should
    shrink (in absolute width)."""
    rng = np.random.default_rng(0xFEED)
    est = BayesianSigmaEstimator(
        prior_sigma2=1.0,
        T_decorr_s=1.0,
        dt_s=1.0,
        prior_strength_n0=2.0,
        window_s=5000.0,
    )
    widths = []
    for k in range(5000):
        est.update(rng.normal(0.0, 1.0))
        if k + 1 in (50, 500, 4000):
            p = est.posterior(credible=0.90)
            widths.append(p.sigma2_hi - p.sigma2_lo)
    assert widths[0] > widths[1] > widths[2]


# ---------------------------------------------------------------------------
# BayesianSigmaEstimator: ring buffer & reset
# ---------------------------------------------------------------------------


def test_ring_buffer_eviction_keeps_sum_sq_correct():
    """Push more samples than the capacity; the incremental sum-of-squares
    should match a brute-force recompute of the in-window samples."""
    est = _make_default_estimator(window_s=10.0, dt_s=1.0, T_decorr_s=1.0)
    samples = [0.1, -0.2, 0.5, 0.3, -0.4, 0.6, 0.7, -0.1, 0.2, -0.3,
               0.8, -0.9, 1.0, 0.05]  # 14 samples, capacity 10
    for x in samples:
        est.update(x)
    in_window = samples[-10:]  # last 10
    expected_sum_sq = sum(x * x for x in in_window)
    post = est.posterior()
    # S = sum_sq, S_eff = S * n_eff/n_raw, beta = beta_0 + S_eff/2.
    # Check sum_sq indirectly via the posterior beta.
    n_eff = post.n_eff
    n_raw = post.n_raw
    assert n_raw == 10
    S_eff_expected = expected_sum_sq * (n_eff / n_raw)
    # beta_0 from the prior: n0=2 -> alpha_0=1+eps, beta_0=prior*eps.
    alpha_0 = 1.0 + 1e-6
    beta_0 = 1.0 * (alpha_0 - 1.0)
    beta_expected = beta_0 + 0.5 * S_eff_expected
    assert post.beta == pytest.approx(beta_expected, rel=1e-9)


def test_reset_collapses_to_prior():
    rng = np.random.default_rng(7)
    est = _make_default_estimator(prior_sigma2=4.0, prior_strength_n0=10.0)
    for _ in range(100):
        est.update(rng.normal(0.0, 2.0))
    est.reset()
    assert est.n_raw == 0
    post = est.posterior()
    assert post.sigma2_mean == pytest.approx(4.0)
    assert post.n_eff == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BayesianSigmaEstimator: assume_zero_mean=False path
# ---------------------------------------------------------------------------


def test_sample_mean_subtraction_with_offset():
    """If the observed signal has a non-zero mean, the zero-mean
    sufficient statistic over-estimates sigma^2. With assume_zero_mean
    =False the sample mean is subtracted and the estimate recovers."""
    rng = np.random.default_rng(123)
    true_sigma = 0.5
    offset = 3.0
    n = 1000

    samples = offset + rng.normal(0.0, true_sigma, size=n)

    est_zero = BayesianSigmaEstimator(
        prior_sigma2=1.0,
        T_decorr_s=1.0,
        dt_s=1.0,
        prior_strength_n0=2.0,
        window_s=float(n),
        assume_zero_mean=True,
    )
    est_dem = BayesianSigmaEstimator(
        prior_sigma2=1.0,
        T_decorr_s=1.0,
        dt_s=1.0,
        prior_strength_n0=2.0,
        window_s=float(n),
        assume_zero_mean=False,
    )
    for x in samples:
        est_zero.update(x)
        est_dem.update(x)

    post_zero = est_zero.posterior()
    post_dem = est_dem.posterior()
    # Zero-mean estimator picks up the offset^2 -> ~9.25 m^2.
    assert post_zero.sigma2_median > 5.0
    # Demeaned estimator recovers ~0.25 m^2.
    assert post_dem.sigma2_median == pytest.approx(true_sigma ** 2, rel=0.10)


# ---------------------------------------------------------------------------
# BayesianSigmaEstimator: is_warm
# ---------------------------------------------------------------------------


def test_is_warm_threshold_behaviour():
    est = _make_default_estimator(
        prior_sigma2=1.0,
        T_decorr_s=20.0,
        dt_s=1.0,
        window_s=60.0,
    )
    # n_eff = n_raw * 1 / 20. Need n_raw >= 20 for n_eff >= 1.
    for _ in range(19):
        est.update(0.0)
    assert not est.is_warm()
    est.update(0.0)  # 20th sample -> n_eff = 1.0
    assert est.is_warm()
    # Custom threshold
    assert not est.is_warm(n_eff_threshold=2.0)


# ---------------------------------------------------------------------------
# Posterior properties
# ---------------------------------------------------------------------------


def test_credible_interval_ordering_and_level_validation():
    rng = np.random.default_rng(11)
    est = _make_default_estimator(window_s=200.0, T_decorr_s=1.0)
    for _ in range(200):
        est.update(rng.normal(0.0, 1.0))
    post = est.posterior(credible=0.90)
    assert post.sigma2_lo < post.sigma2_median < post.sigma2_hi
    assert post.sigma_lo < post.sigma_median < post.sigma_hi

    with pytest.raises(ValueError, match="credible"):
        est.posterior(credible=0.0)
    with pytest.raises(ValueError, match="credible"):
        est.posterior(credible=1.0)
    with pytest.raises(ValueError, match="credible"):
        est.posterior(credible=-0.1)


# ---------------------------------------------------------------------------
# PosteriorHealth diagnostics
# ---------------------------------------------------------------------------


def test_health_empty_window_returns_safe_defaults():
    """Empty window: posterior == prior. is_warm=False; ratios are NaN."""
    est = _make_default_estimator()
    h = est.health()
    assert isinstance(h, PosteriorHealth)
    assert h.n_raw == 0
    assert h.n_eff == 0.0
    assert h.is_warm is False
    # By convention prior IS in the prior CI (posterior == prior).
    assert h.prior_in_credible_interval is True
    # All sample-derived primitives are undefined.
    assert np.isnan(h.sample_mean)
    assert np.isnan(h.sample_mean_over_sigma)
    assert np.isnan(h.kurtosis_excess)
    assert np.isnan(h.halves_sigma_ratio)


def test_health_zero_mean_gaussian_after_warmup_is_clean():
    """Stationary zero-mean Gaussian: A2 ratio small, kurtosis ~0,
    halves ratio ~1, prior inside CI."""
    rng = np.random.default_rng(42)
    est = _make_default_estimator(
        prior_sigma2=1.0,
        T_decorr_s=1.0,
        dt_s=1.0,
        window_s=400.0,
    )
    for _ in range(400):
        est.update(rng.normal(0.0, 1.0))
    h = est.health()
    assert h.is_warm is True
    # Sample mean of 400 N(0,1) draws: |mean|/sigma ~ 1/sqrt(400) = 0.05.
    # Allow slack: well below the 0.3 "unsettled" threshold.
    assert h.sample_mean_over_sigma < 0.15
    # Prior sigma=1 should sit inside the posterior 90% CI.
    assert h.prior_in_credible_interval is True
    # Excess kurtosis: variance ~24/n = 0.06 -> ~0.25 sd.
    assert abs(h.kurtosis_excess) < 1.0
    # Halves ratio: variance of log-ratio ~ 2/n; ratio in [0.7, 1.4] easily.
    assert 0.7 < h.halves_sigma_ratio < 1.4


def test_health_static_offset_flags_a2_violation():
    """Sin + DC offset: A2 (zero mean) fails, sample_mean_over_sigma > 0.3.

    Models DP integral term still ramping / observer bias not converged.
    """
    est = _make_default_estimator(
        prior_sigma2=1.0,
        T_decorr_s=1.0,
        dt_s=1.0,
        window_s=400.0,
    )
    # DC offset = 0.5 m, sinusoid amplitude 0.3 m -> sigma_zm ~ sqrt(0.5^2 +
    # 0.3^2/2) ~ 0.55 m. |mean|/sigma ~ 0.5/0.55 ~ 0.9 -> "unsettled".
    rng = np.random.default_rng(7)
    for k in range(400):
        t = k * 1.0
        est.update(0.5 + 0.3 * np.sin(2 * np.pi * t / 30.0) + 0.05 * rng.normal())
    h = est.health()
    assert h.is_warm is True
    assert h.sample_mean_over_sigma > 0.3, (
        f"Expected A2 indicator > 0.3, got {h.sample_mean_over_sigma:.3f}"
    )
    assert abs(h.sample_mean - 0.5) < 0.1


def test_health_heavy_tails_flag_a3_violation():
    """Student-t (df=4) draws: kurtosis_excess clearly > 1."""
    rng = np.random.default_rng(13)
    est = _make_default_estimator(
        prior_sigma2=2.0,  # var of t_4 = 4/(4-2) = 2
        T_decorr_s=1.0,
        dt_s=1.0,
        window_s=2000.0,
    )
    samples = rng.standard_t(df=4, size=2000)
    for s in samples:
        est.update(float(s))
    h = est.health()
    # True excess kurtosis of t_df is 6/(df-4) -> infinite for df=4; for
    # df=5 it would be 6. Empirically for t_4 with n=2000 it is large
    # (samples include rare big draws). Require > 1 to be robustly above
    # Gaussian noise floor (~0.25 sd at this n).
    assert h.kurtosis_excess > 1.0, (
        f"Expected heavy tails to give excess kurtosis > 1, "
        f"got {h.kurtosis_excess:.3f}"
    )


def test_health_ramp_flags_a1_nonstationarity():
    """Variance ramp from sigma=0.5 to sigma=2.0 across the window:
    halves_sigma_ratio significantly > 1."""
    rng = np.random.default_rng(21)
    est = _make_default_estimator(
        prior_sigma2=1.0,
        T_decorr_s=1.0,
        dt_s=1.0,
        window_s=400.0,
    )
    n = 400
    for k in range(n):
        # Linearly ramp sigma from 0.5 to 2.0.
        sigma_k = 0.5 + (2.0 - 0.5) * (k / (n - 1))
        est.update(sigma_k * rng.normal())
    h = est.health()
    # Second-half RMS expected ~1.6, first-half ~0.8 -> ratio ~2.
    assert h.halves_sigma_ratio > 1.5, (
        f"Expected halves ratio > 1.5 for ramp, got {h.halves_sigma_ratio:.3f}"
    )


def test_health_prior_data_tension_flag():
    """Data with sigma=3 against a prior with sigma2=1: posterior CI should
    NOT contain the prior sigma=1 once the window is well-populated."""
    rng = np.random.default_rng(33)
    est = _make_default_estimator(
        prior_sigma2=1.0,
        prior_strength_n0=2.0,  # weak prior
        T_decorr_s=1.0,
        dt_s=1.0,
        window_s=400.0,
    )
    for _ in range(400):
        est.update(3.0 * rng.normal())
    h = est.health()
    assert h.is_warm is True
    assert h.prior_in_credible_interval is False
    # And the consistent zero-mean Gaussian flags should remain clean.
    assert h.sample_mean_over_sigma < 0.2


def test_health_does_not_mutate_state():
    """health() must be a pure function of the buffer."""
    rng = np.random.default_rng(0)
    est = _make_default_estimator(window_s=100.0, T_decorr_s=1.0)
    for _ in range(100):
        est.update(rng.normal())
    n_before = est.n_raw
    sumsq_before = est._sum_sq
    sum_before = est._sum
    _ = est.health()
    _ = est.health()
    assert est.n_raw == n_before
    assert est._sum_sq == sumsq_before
    assert est._sum == sum_before


def test_health_kurtosis_undefined_below_4_samples():
    est = _make_default_estimator()
    est.update(0.1)
    est.update(-0.1)
    est.update(0.05)
    h = est.health()
    assert h.n_raw == 3
    assert np.isnan(h.kurtosis_excess)
    assert np.isnan(h.halves_sigma_ratio)
    # But sample_mean and the A2 ratio ARE defined.
    assert not np.isnan(h.sample_mean)
    assert not np.isnan(h.sample_mean_over_sigma)


# ---------------------------------------------------------------------------
# combine_radial_posterior
# ---------------------------------------------------------------------------


def _build_warm_estimator(
    sigma_true: float,
    n_samples: int = 600,
    prior_sigma2: float = 1.0,
    seed: int = 0,
) -> BayesianSigmaEstimator:
    """Helper: build an estimator and feed it n_samples of N(0, sigma_true)."""
    est = _make_default_estimator(
        prior_sigma2=prior_sigma2,
        T_decorr_s=1.0,         # all samples independent
        dt_s=1.0,
        window_s=float(n_samples),
        prior_strength_n0=2.0,  # weak prior, posterior driven by data
    )
    rng = np.random.default_rng(seed)
    for _ in range(n_samples):
        est.update(rng.normal(0.0, sigma_true))
    return est


def test_combine_radial_equal_variance_recovers_rayleigh_identities():
    """sigma_x = sigma_y = sigma => sigma_R = sqrt(2)*sigma and
    E[R] = sigma * sqrt(pi/2) = sigma_R * sqrt(pi)/2.

    Both estimators data-conditioned at sigma=1 so the posterior medians
    are tightly concentrated around sigma_x^2 = sigma_y^2 = 1.
    """
    sigma_true = 1.0
    est_x = _build_warm_estimator(sigma_true=sigma_true, seed=11)
    est_y = _build_warm_estimator(sigma_true=sigma_true, seed=22)
    post_x = est_x.posterior()
    post_y = est_y.posterior()

    rng = np.random.default_rng(42)
    rad = combine_radial_posterior(
        post_x, post_y, credible=0.90, n_mc=5000, rng=rng,
    )

    # sigma_R analytical: sqrt(sigma_x^2 + sigma_y^2) = sqrt(2)*sigma.
    expected_sigma_R = np.sqrt(2.0) * sigma_true
    assert rad.sigma_R_median == pytest.approx(expected_sigma_R, rel=0.05)
    assert rad.sigma_R_mean == pytest.approx(expected_sigma_R, rel=0.05)

    # E[R] analytical: sigma * sqrt(pi/2) = sigma_R * sqrt(pi)/2.
    expected_ER = sigma_true * np.sqrt(np.pi / 2.0)
    assert rad.expected_R_median == pytest.approx(expected_ER, rel=0.05)
    # And the equivalent identity using sigma_R:
    assert rad.expected_R_median == pytest.approx(
        rad.sigma_R_median * np.sqrt(np.pi) / 2.0, rel=0.05
    )

    # CI ordering and the median falls inside.
    assert rad.sigma_R_lo < rad.sigma_R_median < rad.sigma_R_hi
    assert rad.expected_R_lo < rad.expected_R_median < rad.expected_R_hi


def test_combine_radial_unequal_variance_uses_trace():
    """sigma_x = 0.6, sigma_y = 1.0 => sigma_R = sqrt(0.36 + 1.0) ~ 1.166.

    Verifies the sqrt(sigma_x^2 + sigma_y^2) construction works for the
    Hoyt (unequal variance) case.
    """
    est_x = _build_warm_estimator(sigma_true=0.6, seed=101)
    est_y = _build_warm_estimator(sigma_true=1.0, seed=202)
    post_x = est_x.posterior()
    post_y = est_y.posterior()

    rng = np.random.default_rng(7)
    rad = combine_radial_posterior(post_x, post_y, n_mc=5000, rng=rng)

    expected_sigma_R = np.sqrt(0.6 ** 2 + 1.0 ** 2)
    assert rad.sigma_R_median == pytest.approx(expected_sigma_R, rel=0.05)
    assert rad.sigma_R_mean == pytest.approx(expected_sigma_R, rel=0.05)


def test_combine_radial_warmth_composition():
    """is_warm AND of the two axes; n_eff_min = min."""
    # x: warm
    est_x = _make_default_estimator(
        prior_sigma2=1.0, T_decorr_s=1.0, dt_s=1.0, window_s=100.0,
    )
    rng = np.random.default_rng(0)
    for _ in range(100):
        est_x.update(rng.normal())
    # y: cold (only 10 samples with T_decorr=1 -> n_eff=10) actually warm
    est_y = _make_default_estimator(
        prior_sigma2=1.0, T_decorr_s=1.0, dt_s=1.0, window_s=100.0,
    )
    for _ in range(10):
        est_y.update(rng.normal())

    rad = combine_radial_posterior(
        est_x.posterior(), est_y.posterior(), n_mc=500,
    )
    assert rad.is_warm is True  # both >= 1
    assert rad.n_eff_min == pytest.approx(min(est_x.n_eff, est_y.n_eff))
    assert rad.n_eff_x == pytest.approx(est_x.n_eff)
    assert rad.n_eff_y == pytest.approx(est_y.n_eff)

    # Now make y cold (zero samples).
    est_cold = _make_default_estimator(
        prior_sigma2=1.0, T_decorr_s=1.0, dt_s=1.0, window_s=100.0,
    )
    rad2 = combine_radial_posterior(
        est_x.posterior(), est_cold.posterior(), n_mc=500,
    )
    assert rad2.is_warm is False
    assert rad2.n_eff_min == 0.0


def test_combine_radial_rejects_invalid_params():
    est = _build_warm_estimator(sigma_true=1.0, seed=1)
    post = est.posterior()
    with pytest.raises(ValueError, match="credible"):
        combine_radial_posterior(post, post, credible=0.0)
    with pytest.raises(ValueError, match="credible"):
        combine_radial_posterior(post, post, credible=1.0)
    with pytest.raises(ValueError, match="n_mc"):
        combine_radial_posterior(post, post, n_mc=10)


def test_combine_radial_against_time_domain_realisation():
    """Tier A integration check: predicted sigma_R and E[R] from the
    posteriors match the empirical sqrt(mean(r^2)) and mean(r) from a
    fresh independent (X, Y) realisation drawn at the posterior mean
    sigmas.

    Uses sigma_x = 0.7, sigma_y = 1.1 (Hoyt asymmetry ratio ~1.57, in
    the regime where the equal-variance E[R] approximation is
    measurably biased).
    """
    sigma_x_true = 0.7
    sigma_y_true = 1.1

    # Build well-converged posteriors.
    est_x = _build_warm_estimator(sigma_true=sigma_x_true, n_samples=2000, seed=1)
    est_y = _build_warm_estimator(sigma_true=sigma_y_true, n_samples=2000, seed=2)

    rng = np.random.default_rng(99)
    rad = combine_radial_posterior(
        est_x.posterior(), est_y.posterior(),
        n_mc=5000, rng=rng,
    )

    # Empirical: simulate r(t) on a long independent series at the TRUE
    # sigmas and compute moments.
    rng2 = np.random.default_rng(303)
    n_emp = 100_000
    X = rng2.normal(0.0, sigma_x_true, size=n_emp)
    Y = rng2.normal(0.0, sigma_y_true, size=n_emp)
    R = np.sqrt(X * X + Y * Y)

    sigma_R_emp = float(np.sqrt(np.mean(R * R)))   # E[R^2] = sigma_x^2 + sigma_y^2
    ER_emp = float(np.mean(R))

    # Predicted matches empirical (5% tolerance covers MC + posterior bias).
    assert rad.sigma_R_median == pytest.approx(sigma_R_emp, rel=0.05)
    # Empirical sigma_R should sit inside the posterior CI on sigma_R.
    assert rad.sigma_R_lo <= sigma_R_emp <= rad.sigma_R_hi

    # E[R]: the MC-derived estimate accounts for Hoyt asymmetry exactly,
    # so this should match within MC noise (~1% at n_mc=5000).
    assert rad.expected_R_median == pytest.approx(ER_emp, rel=0.03)


def test_combine_radial_mc_reproducibility_with_seeded_rng():
    est_x = _build_warm_estimator(sigma_true=1.0, seed=5)
    est_y = _build_warm_estimator(sigma_true=1.5, seed=6)
    post_x = est_x.posterior()
    post_y = est_y.posterior()

    rad_a = combine_radial_posterior(
        post_x, post_y, n_mc=1000,
        rng=np.random.default_rng(123),
    )
    rad_b = combine_radial_posterior(
        post_x, post_y, n_mc=1000,
        rng=np.random.default_rng(123),
    )
    assert rad_a.sigma_R_median == rad_b.sigma_R_median
    assert rad_a.sigma_R_lo == rad_b.sigma_R_lo
    assert rad_a.sigma_R_hi == rad_b.sigma_R_hi
    assert rad_a.expected_R_median == rad_b.expected_R_median
