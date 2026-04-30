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
    SigmaPosterior,
    closed_loop_decorrelation_time,
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
