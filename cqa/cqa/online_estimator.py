"""Online Bayesian estimator of the position-deviation variance sigma^2.

Goal
----
At runtime, while the DP is on station, observe the realised scalar
deviation X(t) (e.g. the radial gangway-base position deviation, or
the telescope-length deviation around the setpoint) and continuously
update a posterior distribution of its variance sigma^2. Feed the
posterior sigma into the same Rice / Cartwright-Longuet-Higgins
formula used by `summarise_intact_prior`, replacing the model-based
prior sigma with a data-conditioned posterior.

Why a Bayesian update (and not just a windowed sample variance)?
----------------------------------------------------------------
Three reasons, in order of importance:

1. Operator confidence in the model. The prior we use today is
   built from the linearised closed-loop spectrum + nominal
   disturbance PSDs. It can be wrong by a factor of 2 (current
   bias errors, sea-state misclassification, controller retune).
   A posterior that visibly tracks the data tells the operator
   whether the prior model is being supported or contradicted.

2. Quantified uncertainty at small effective sample sizes. A
   60 s window of a 100 s decorrelation-time signal contains <1
   independent draw. A frequentist sample variance would be
   meaningless; an inverse-gamma posterior degrades gracefully
   to the prior with the right credible interval.

3. Composability with the Rice formula. The prior layer already
   exposes a `sigma_override` hook (see `p_exceed_from_psd`):
   the spectral SHAPE (and thus nu_0+ and the Vanmarcke q) is
   kept from the prior model, only the LEVEL (sigma) is updated
   from data. This is the textbook "posterior on the variance,
   prior on the spectral shape" decomposition for stationary
   Gaussian channels.

Conjugate inverse-gamma model
-----------------------------
For a zero-mean stationary Gaussian process X with unknown
variance sigma^2, the conjugate prior is the inverse gamma:

    sigma^2 ~ InvGamma(alpha_0, beta_0).

The posterior after observing N independent samples
{x_1, ..., x_N} with sufficient statistic S = sum_i x_i^2 is

    sigma^2 | data ~ InvGamma(alpha_0 + N/2, beta_0 + S/2).

We don't observe N independent samples; we observe N_raw correlated
samples at dt. Using an effective sample size correction

    N_eff = N_raw / max(1, T_decorr / dt)

(Bartlett-style) and rescaling the sufficient statistic to match,

    S_eff = S * (N_eff / N_raw),

we get an inverse-gamma posterior whose credible interval reflects
the *number of independent draws* in the window, not the raw sample
count. This is the standard treatment for stationary Gaussian
channels (Bayesian Data Analysis, Gelman et al., ch. 2.6).

Prior parameterisation
----------------------
We parameterise the InvGamma prior by

    prior_sigma2  = E[sigma^2] under the prior   = beta_0 / (alpha_0 - 1)
    prior_strength_n0  = "effective prior sample count" n_0
                       so that alpha_0 = n_0 / 2 and
                                beta_0  = prior_sigma2 * (alpha_0 - 1).

Default ``prior_strength_n0 = 2.0``: very weak (alpha_0 = 1, posterior
reduces to N-driven for any meaningful window). Increase to anchor
more strongly to the prior; useful when the linearised spectrum is
trusted (e.g. dock trials).

Decorrelation time T_decorr
---------------------------
The Bartlett effective-sample-size correction needs the
*variance-estimator* decorrelation time, defined as

    T_var = pi * integral_0^infty S_X(omega)^2 d omega / m0^2

(see `cqa.extreme_value.variance_decorrelation_time_from_psd` for the
derivation, which uses the Isserlis identity for Gaussian 4th moments
and Plancherel on the autocovariance). For broadband forcing this
recovers ``~1/(zeta*omega_n)``; for narrowband disturbance forcing it
can be 3-5x larger -- the realistic case for DP-class operations
(slow-drift + current variability with multi-minute time scales).

The legacy fallback

    T_decorr = 1 / (zeta * omega_n),

implemented in ``closed_loop_decorrelation_time``, ignores the
disturbance shape and is preserved only for diagnostics. Production
should pass ``T_var`` from the prior summary.

Production / prototype boundary
-------------------------------
Pure numpy + scipy. Pure-Python ring buffer for the sliding window
(prototype scale; the C++ port should use a fixed-size circular
queue with O(1) push and incremental sum-of-squares).

References
----------
* Gelman, A. et al. (2013), "Bayesian Data Analysis", 3rd ed.,
  ch. 2.6 (normal model with unknown variance).
* Bartlett, M.S. (1946), "On the theoretical specification and
  sampling properties of autocorrelated time-series",
  J. R. Stat. Soc. Suppl. 8, 27-41 (effective sample size for
  autocorrelated Gaussian data).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import ControllerParams


# ---------------------------------------------------------------------------
# Posterior summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SigmaPosterior:
    """Inverse-gamma posterior summary on sigma^2.

    Means and credible intervals are computed in BOTH variance space
    (sigma^2 [m^2]) and standard-deviation space (sigma [m]). For the
    Rice formula downstream, sigma is the natural unit -- but the
    posterior is naturally inverse-gamma on sigma^2, so we report both.

    Fields
    ------
    sigma2_mean   : E[sigma^2 | data] = beta / (alpha - 1)   for alpha > 1.
    sigma2_median : median of InvGamma(alpha, beta).
    sigma2_lo, sigma2_hi : equal-tail credible interval bounds on sigma^2
                           (default 90 %, i.e. 5th and 95th percentiles).
    sigma_*       : sqrt(sigma2_*). For the median / quantiles this is
                    exact (sqrt is monotone). sigma_mean is sqrt(E[sigma^2]),
                    NOT E[sigma]; use sigma_median when you need a true
                    point estimate of sigma.
    n_raw         : number of raw samples in the window.
    n_eff         : effective independent sample count after Bartlett
                    correction.
    alpha, beta   : posterior inverse-gamma shape and scale.
    prior_sigma2  : the prior mean of sigma^2.
    prior_strength_n0 : the "effective prior sample count" n_0.
    credible      : the credible level used for sigma*_lo / *_hi (e.g. 0.90).
    """

    sigma2_mean: float
    sigma2_median: float
    sigma2_lo: float
    sigma2_hi: float
    sigma_mean: float
    sigma_median: float
    sigma_lo: float
    sigma_hi: float
    n_raw: int
    n_eff: float
    alpha: float
    beta: float
    prior_sigma2: float
    prior_strength_n0: float
    credible: float


# ---------------------------------------------------------------------------
# Posterior health diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PosteriorHealth:
    """Cheap runtime primitives diagnosing the assumptions baked into
    ``BayesianSigmaEstimator.posterior()``.

    Design philosophy
    -----------------
    Expose primitives, do not decide. The operator panel (or the C++
    runtime) composes a WARMING / OK / UNSETTLED / INVALID badge from
    these scalars using site-tuned thresholds. We only flag the cheapest
    assumption-failure modes here; richer goodness-of-fit checks belong
    in the offline regression suite, not the online cycle.

    Assumption coverage
    -------------------
    A1 stationarity            -- ``halves_sigma_ratio``
    A2 zero mean (DP regulates)-- ``sample_mean_over_sigma``  (KEY)
    A3 Gaussian marginals      -- ``kurtosis_excess``
    A4 Bartlett ESS captures   -- ``is_warm`` (n_eff >= threshold)
       autocorrelation
    A5 prior shape correct,    -- ``prior_in_credible_interval``
       only LEVEL data-driven

    The primary signal is ``sample_mean_over_sigma``: at the start of
    an operation the DP integral term and observer bias estimator are
    still settling (~2-5 min and ~1-2 min respectively); a non-zero
    sample mean during that transient inflates the sufficient statistic
    ``S = sum x_i^2`` because ``E[X^2] = sigma^2 + mu^2``. A ratio of
    0.3 inflates the variance estimate by ~9 %; a ratio of 1.0 inflates
    it by 2x. Suggested thresholds:

        |mean|/sigma < 0.1   "settled"
        0.1 <= ratio < 0.3   "warming"
        0.3 <= ratio < 1.0   "unsettled"
        ratio >= 1.0         "invalid" (likely setpoint drift / bias)

    Fields
    ------
    n_raw : int. Raw samples currently in the window.
    n_eff : float. Effective independent sample count after Bartlett.
    is_warm : bool. ``n_eff >= n_eff_threshold``.
    n_eff_threshold : float. Threshold used for ``is_warm``.
    sample_mean : float. Windowed sample mean of X. NaN if window empty.
    sample_mean_over_sigma : float. ``|sample_mean| / sigma_post.sigma_median``.
        NaN if window empty or posterior median is non-positive. **Primary
        A2 indicator** -- catches DP integral / observer bias not yet
        converged, persistent low-frequency disturbance, setpoint drift.
    prior_in_credible_interval : bool. True if the prior sigma
        ``sqrt(prior_sigma2)`` falls inside the posterior equal-tail
        credible interval ``[sigma_lo, sigma_hi]``. False signals
        model-data tension (A4/A5).
    kurtosis_excess : float. Sample excess kurtosis (kurtosis - 3.0)
        of the in-window samples. Zero for Gaussian; positive for
        heavy-tailed (slamming, saturation). NaN if n_raw < 4.
        Note: variance of the sample kurtosis is ~24/n_raw for Gaussian
        data, so this is noisy below ~50 samples; use with caution.
    halves_sigma_ratio : float. ``std(second_half) / std(first_half)``
        of the windowed samples (zero-mean assumption: divides by
        sqrt(sum_sq/n) of each half). Ratios far from 1.0 indicate
        non-stationarity within the window. NaN if n_raw < 4.
    """

    n_raw: int
    n_eff: float
    is_warm: bool
    n_eff_threshold: float
    sample_mean: float
    sample_mean_over_sigma: float
    prior_in_credible_interval: bool
    kurtosis_excess: float
    halves_sigma_ratio: float


# ---------------------------------------------------------------------------
# Radial posterior (combination of per-axis x, y posteriors)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadialPosterior:
    """Combined radial posterior built from two independent per-axis
    InvGamma posteriors on sigma_x^2 and sigma_y^2.

    Why this exists
    ---------------
    The DP regulates the body-x and body-y position to setpoint
    independently, so the natural channels for online estimation are
    ``dx(t)`` and ``dy(t)`` (each zero-mean Gaussian with its own
    variance). The operator, however, cares about the **radial
    distance** ``R(t) = sqrt(dx^2 + dy^2)`` — a single scalar with
    obvious geometric meaning ("how far are we from setpoint?").

    This class produces operator-facing radial scalars from the two
    per-axis posteriors:

    * ``sigma_R = sqrt(sigma_x^2 + sigma_y^2)`` — the Rice-formula
      radial scale parameter. Equal to ``sqrt(2)*sigma`` in the
      equal-variance Rayleigh limit; equal to ``sqrt(trace(Sigma))``
      in general (axis-rotation invariant).

    * ``E[R]`` — the typical radial deviation, marginalised over both
      the posterior uncertainty in (sigma_x, sigma_y) AND the in-window
      Hoyt asymmetry. Computed by Monte Carlo: sample
      sigma_x^2 ~ InvGamma, sigma_y^2 ~ InvGamma, then X ~ N(0, sigma_x),
      Y ~ N(0, sigma_y), R = sqrt(X^2+Y^2), report mean(R). In the
      equal-variance limit this reduces to ``sigma_R * sqrt(pi)/2``
      (i.e. the Rayleigh ``sigma * sqrt(pi/2)``).

    * 90% CI on sigma_R from MC samples (the sum of two independent
      InvGamma RVs is NOT InvGamma — no closed form, MC is the right
      tool).

    Independence assumption
    -----------------------
    Built assuming ``cov(sigma_x^2, sigma_y^2) = 0`` (independent
    posteriors). For decoupled surge/sway controllers the per-axis
    estimators ARE independent (separate sufficient statistics on
    independent channels). Cross-axis coupling at oblique forcing
    (cov(dx, dy) ≠ 0) is a second-order effect, deferred until
    empirical evidence shows it matters at production operating points.

    Fields
    ------
    sigma_R_median, sigma_R_mean : sqrt(sigma_x^2+sigma_y^2) point
        estimates [m]. ``median`` uses MC sample median (matches the
        per-axis posterior medians for equal-variance, slightly
        different otherwise); ``mean = sqrt(E[sigma_x^2]+E[sigma_y^2])``
        is exact from the InvGamma means (only finite when both
        posteriors have alpha > 1).
    sigma_R_lo, sigma_R_hi : equal-tail credible-interval bounds [m]
        at the level ``credible``. From MC.
    expected_R_median : E[R] in metres, marginalised over the joint
        posterior + Hoyt asymmetry, computed by MC.
    expected_R_lo, expected_R_hi : equal-tail CI on E[R] [m]. NB this
        is the CI on the *expected* radial distance under the
        posterior, NOT the CI of the actual radial distance (which is
        a much wider Rayleigh-type spread).
    n_mc : number of Monte Carlo samples used.
    n_eff_min : min(n_eff_x, n_eff_y) — composite "warmth" indicator.
        The radial summary is only as warm as its slowest axis.
    n_eff_x, n_eff_y : per-axis effective sample counts.
    is_warm : True iff both per-axis estimators are warm.
    credible : credible level used for the CIs.
    radial_mean_offset_m : float. Magnitude of the 2D sample-mean vector
        ``|(mean(dx), mean(dy))|`` in metres. Physically: the
        time-averaged vessel offset from setpoint inside the window.
        ``nan`` if ``sample_mean_x`` / ``sample_mean_y`` were not
        supplied to ``combine_radial_posterior``.
    radial_mean_offset_over_sigma : float. The dimensionless ratio
        ``radial_mean_offset_m / sigma_R_median``. **Principled radial
        analogue of the per-axis A2 indicator** ``sample_mean_over_sigma``:
        catches systematic 2D drift off-station regardless of which
        cardinal axis it acts along. Same threshold semantics as the
        per-axis primitive (``< 0.1`` settled, ``< 0.3`` warming,
        ``< 1.0`` UNSETTLED, ``>= 1.0`` INVALID — variance estimate
        inflated by ``mu^2`` contamination of the sufficient statistic
        on each axis). Strictly cleaner than worst-of-x,y on the
        per-axis ratios because it is rotation-invariant in the body
        frame: a 30 degree heading change does not flip the badge.
        ``nan`` if sample means were not supplied.
    """

    sigma_R_median: float
    sigma_R_mean: float
    sigma_R_lo: float
    sigma_R_hi: float
    expected_R_median: float
    expected_R_lo: float
    expected_R_hi: float
    n_mc: int
    n_eff_min: float
    n_eff_x: float
    n_eff_y: float
    is_warm: bool
    credible: float
    radial_mean_offset_m: float
    radial_mean_offset_over_sigma: float


def combine_radial_posterior(
    posterior_x: SigmaPosterior,
    posterior_y: SigmaPosterior,
    *,
    credible: float = 0.90,
    n_mc: int = 2000,
    rng: Optional[np.random.Generator] = None,
    sample_mean_x: Optional[float] = None,
    sample_mean_y: Optional[float] = None,
) -> RadialPosterior:
    """Combine two per-axis InvGamma posteriors into a radial summary.

    Pure function; no state.

    Parameters
    ----------
    posterior_x, posterior_y : SigmaPosterior
        Per-axis posteriors, typically from two
        ``BayesianSigmaEstimator`` instances watching ``dx(t)`` and
        ``dy(t)`` from ``base_position_xy_time_series``.
    credible : float in (0, 1), default 0.90.
        Equal-tail credible level for the CI bounds.
    n_mc : int, default 2000.
        Monte Carlo sample count. 2000 gives ~1 % stderr on the 5 %
        and 95 % quantiles of sigma_R. Cheap (one scipy.invgamma.rvs
        per axis + one normal draw per axis + one sqrt).
    rng : optional numpy Generator. Defaults to a fresh
        ``default_rng()`` seeded from the OS each call (so successive
        calls give jittered but consistent estimates). Pass an explicit
        Generator to make demos / tests reproducible.
    sample_mean_x, sample_mean_y : optional float, default None.
        Windowed sample means of dx, dy in metres. When both are
        supplied, the radial composite A2 indicator
        ``radial_mean_offset_over_sigma`` is computed and exposed on
        the returned ``RadialPosterior``. Pass them as
        ``estimator_x.health(...).sample_mean`` (and similarly for y),
        which is the same source the per-axis A2 primitive uses.
        When either is None, both fields are set to ``nan``.

    Returns
    -------
    RadialPosterior. See the dataclass docstring for field semantics.
    """
    from scipy.stats import invgamma

    if not (0.0 < credible < 1.0):
        raise ValueError(f"credible must be in (0, 1), got {credible}")
    if n_mc < 100:
        raise ValueError(
            f"n_mc must be >= 100 for usable quantiles, got {n_mc}"
        )

    if rng is None:
        rng = np.random.default_rng()

    # Sample sigma_x^2, sigma_y^2 from their InvGamma posteriors. Use
    # scipy's parameterisation (a=alpha, scale=beta) consistently with
    # SigmaPosterior.posterior().
    sigma2_x_samples = invgamma.rvs(
        a=posterior_x.alpha, scale=posterior_x.beta,
        size=n_mc, random_state=rng,
    )
    sigma2_y_samples = invgamma.rvs(
        a=posterior_y.alpha, scale=posterior_y.beta,
        size=n_mc, random_state=rng,
    )
    sigma_R_samples = np.sqrt(sigma2_x_samples + sigma2_y_samples)

    # E[R | sigma_x, sigma_y]: for each MC draw of (sigma_x, sigma_y),
    # the conditional Hoyt mean is itself an integral. Cheapest exact
    # marginalisation: one extra (X, Y) draw per MC sample, then
    # mean(sqrt(X^2+Y^2)) is an unbiased estimator of E[R]. With
    # n_mc=2000 the SE on E[R] is ~0.5 %.
    sigma_x_samples = np.sqrt(sigma2_x_samples)
    sigma_y_samples = np.sqrt(sigma2_y_samples)
    X = rng.standard_normal(n_mc) * sigma_x_samples
    Y = rng.standard_normal(n_mc) * sigma_y_samples
    R_samples = np.sqrt(X * X + Y * Y)

    # Quantiles.
    lo_q = 0.5 * (1.0 - credible)
    hi_q = 1.0 - lo_q
    sigma_R_median = float(np.median(sigma_R_samples))
    sigma_R_lo = float(np.quantile(sigma_R_samples, lo_q))
    sigma_R_hi = float(np.quantile(sigma_R_samples, hi_q))

    # Closed-form mean (exact, no MC noise) when both alphas > 1.
    if posterior_x.alpha > 1.0 and posterior_y.alpha > 1.0:
        sigma_R_mean = float(np.sqrt(
            posterior_x.beta / (posterior_x.alpha - 1.0)
            + posterior_y.beta / (posterior_y.alpha - 1.0)
        ))
    else:
        sigma_R_mean = float("inf")

    # E[R]: report MC mean as the point estimate (exact under both
    # posterior uncertainty AND Hoyt asymmetry of the in-window R |
    # sigma_x, sigma_y). For the CI on E[R], we report quantiles of
    # the per-sample conditional mean E[R | sigma_x, sigma_y]. We use
    # the equal-variance approximation
    #   E[R | sigma_x, sigma_y] ~ sqrt(sigma_x^2+sigma_y^2) * sqrt(pi)/2
    # whose worst-case bias for sigma_y/sigma_x in [0.5, 2.0] is ~3 %;
    # exact only at sigma_y/sigma_x = 1 (Rayleigh). Documented caveat.
    cond_R_means = sigma_R_samples * (np.sqrt(np.pi) / 2.0)
    expected_R_median = float(np.mean(R_samples))
    expected_R_lo = float(np.quantile(cond_R_means, lo_q))
    expected_R_hi = float(np.quantile(cond_R_means, hi_q))

    n_eff_x = float(posterior_x.n_eff)
    n_eff_y = float(posterior_y.n_eff)
    n_eff_min = float(min(n_eff_x, n_eff_y))
    is_warm = (n_eff_x >= 1.0) and (n_eff_y >= 1.0)

    # Radial 2D mean-offset metric. Rotation-invariant in the body
    # frame: a heading change rotates (mean_x, mean_y) but preserves
    # its magnitude. Strictly cleaner A2 indicator on the radial
    # channel than worst-of-x,y on the per-axis ratios. Denominator is
    # the radial *median* sigma so the metric is comparable to the
    # per-axis sample_mean_over_sigma (which uses the same per-axis
    # sigma_median).
    if sample_mean_x is not None and sample_mean_y is not None:
        mean_offset_m = float(np.hypot(float(sample_mean_x),
                                       float(sample_mean_y)))
        if sigma_R_median > 0.0 and np.isfinite(sigma_R_median):
            mean_offset_over_sigma = float(mean_offset_m / sigma_R_median)
        else:
            mean_offset_over_sigma = float("nan")
    else:
        mean_offset_m = float("nan")
        mean_offset_over_sigma = float("nan")

    return RadialPosterior(
        sigma_R_median=sigma_R_median,
        sigma_R_mean=sigma_R_mean,
        sigma_R_lo=sigma_R_lo,
        sigma_R_hi=sigma_R_hi,
        expected_R_median=expected_R_median,
        expected_R_lo=expected_R_lo,
        expected_R_hi=expected_R_hi,
        n_mc=int(n_mc),
        n_eff_min=n_eff_min,
        n_eff_x=n_eff_x,
        n_eff_y=n_eff_y,
        is_warm=bool(is_warm),
        credible=float(credible),
        radial_mean_offset_m=mean_offset_m,
        radial_mean_offset_over_sigma=mean_offset_over_sigma,
    )


# ---------------------------------------------------------------------------
# Validity badge: compose A1-A5 health primitives into a single verdict
# ---------------------------------------------------------------------------


# Ordered worst-to-best for max() comparison: a numeric rank works fine
# but using strings + a dict keeps the public API human-readable.
_VALIDITY_RANK = {"OK": 0, "WARMING": 1, "UNSETTLED": 2, "INVALID": 3}
_VALIDITY_LEVELS = ("OK", "WARMING", "UNSETTLED", "INVALID")


def _worst_validity(*levels: str) -> str:
    """Return the highest-severity validity level among the args."""
    return max(levels, key=lambda v: _VALIDITY_RANK[v])


@dataclass(frozen=True)
class ValidityBadge:
    """Composite operator-facing verdict on a single posterior channel.

    Built by ``compose_validity_badge`` from a ``PosteriorHealth`` plus
    optional thresholds. The ``level`` is a single 4-state badge
    (OK / WARMING / UNSETTLED / INVALID) computed as the worst over
    five per-assumption verdicts; ``reasons`` is a human-readable list
    of strings naming every primitive that contributed *at or above*
    WARMING. Empty for a fully OK channel.

    Per-assumption verdicts
    -----------------------
    A4 (n_eff): below ``n_eff_min`` -> INVALID, below ``n_eff_warm`` ->
        WARMING, otherwise OK. Rationale: ``n_eff < 2`` means the
        posterior is the prior + noise; we cannot trust *any* of the
        other primitives if there are not at least a couple of
        independent draws in the window.
    A2 (|mean|/sigma): the four-band ``[0.1, 0.3, 1.0]`` ladder
        documented on ``PosteriorHealth.sample_mean_over_sigma``.
        NaN -> INVALID (window empty or sigma_median <= 0).
    A1 (halves_sigma_ratio): inside ``[1/1.5, 1.5]`` OK, inside
        ``[1/2, 2]`` WARMING, outside UNSETTLED. NaN (n_raw < 4) ->
        skipped (treated as OK; A4 already governs the cold case).
        Rationale: stationarity violations inflate the variance
        estimate but do not make it nonsensical -- they degrade
        gracefully.
    A3 (|kurtosis_excess|): inside 0.5 OK, inside 1.5 WARMING,
        outside UNSETTLED. NaN (n_raw < 4) -> skipped. Caveat:
        sample kurtosis has variance ~24/n_raw under H0=Gaussian, so
        the WARMING band fires routinely below ~50 raw samples and
        should be interpreted loosely there.
    A5 (prior_in_credible_interval): False -> WARMING only.
        Rationale: posterior contradicting prior is informational
        ("model and data disagree, investigate") but does not
        invalidate the data-driven posterior; the data wins.

    Fields
    ------
    level : str. One of ``"OK"``, ``"WARMING"``, ``"UNSETTLED"``,
        ``"INVALID"``. Worst over the per-assumption verdicts.
    reasons : tuple[str, ...]. Human-readable strings for every
        primitive that was at or above WARMING. Empty when level=OK.
        Each string starts with the assumption tag (``"A2: ..."``)
        and ends with the threshold band that fired.
    """

    level: str
    reasons: tuple


def compose_validity_badge(
    health: PosteriorHealth,
    *,
    n_eff_min: float = 2.0,
    n_eff_warm: float = 5.0,
    a2_warming: float = 0.1,
    a2_unsettled: float = 0.3,
    a2_invalid: float = 1.0,
    a1_ok_ratio: float = 1.5,
    a1_warm_ratio: float = 2.0,
    a3_ok: float = 0.5,
    a3_warm: float = 1.5,
) -> ValidityBadge:
    """Compose A1-A5 health primitives into a single per-channel
    validity badge.

    Pure function; no state. Defaults match the operator-band
    suggestions documented on ``PosteriorHealth``. All thresholds are
    overridable per call so the C++ panel layer can tune per site /
    per channel.

    Parameters
    ----------
    health : PosteriorHealth
        From ``BayesianSigmaEstimator.health(...)``.
    n_eff_min : float, default 2.0.
        Below this the posterior is the prior + noise -> INVALID.
    n_eff_warm : float, default 5.0.
        Below this but at least ``n_eff_min`` -> WARMING.
    a2_warming, a2_unsettled, a2_invalid : float, defaults
        ``0.1, 0.3, 1.0``. ``|sample_mean| / sigma_median`` thresholds
        for the A2 ladder.
    a1_ok_ratio, a1_warm_ratio : float, defaults ``1.5, 2.0``.
        ``halves_sigma_ratio`` is OK if it lies in
        ``[1/a1_ok_ratio, a1_ok_ratio]``, WARMING if in
        ``[1/a1_warm_ratio, a1_warm_ratio]``, else UNSETTLED.
    a3_ok, a3_warm : float, defaults ``0.5, 1.5``.
        ``|kurtosis_excess|`` thresholds.

    Returns
    -------
    ValidityBadge.
    """
    if not (n_eff_min > 0.0 and n_eff_warm >= n_eff_min):
        raise ValueError(
            f"require n_eff_warm >= n_eff_min > 0; got "
            f"n_eff_min={n_eff_min}, n_eff_warm={n_eff_warm}"
        )
    if not (0.0 < a2_warming < a2_unsettled < a2_invalid):
        raise ValueError(
            f"A2 thresholds must satisfy 0 < a2_warming < a2_unsettled < "
            f"a2_invalid; got {a2_warming}, {a2_unsettled}, {a2_invalid}"
        )
    if not (1.0 < a1_ok_ratio <= a1_warm_ratio):
        raise ValueError(
            f"A1 thresholds must satisfy 1 < a1_ok_ratio <= a1_warm_ratio; "
            f"got {a1_ok_ratio}, {a1_warm_ratio}"
        )
    if not (0.0 < a3_ok < a3_warm):
        raise ValueError(
            f"A3 thresholds must satisfy 0 < a3_ok < a3_warm; "
            f"got {a3_ok}, {a3_warm}"
        )

    reasons = []

    # A4 warmth (governs everything else; checked first because if
    # there are too few independent draws, the other primitives are
    # noise).
    if health.n_eff < n_eff_min:
        a4 = "INVALID"
        reasons.append(
            f"A4: n_eff={health.n_eff:.1f} below INVALID threshold "
            f"{n_eff_min:.1f} (posterior is essentially the prior)"
        )
    elif health.n_eff < n_eff_warm:
        a4 = "WARMING"
        reasons.append(
            f"A4: n_eff={health.n_eff:.1f} below warm threshold "
            f"{n_eff_warm:.1f}"
        )
    else:
        a4 = "OK"

    # A2 zero-mean (the operationally most important one).
    r = health.sample_mean_over_sigma
    if not np.isfinite(r):
        a2 = "INVALID"
        reasons.append(
            "A2: sample_mean_over_sigma is NaN (window empty or "
            "sigma_median <= 0)"
        )
    elif r >= a2_invalid:
        a2 = "INVALID"
        reasons.append(
            f"A2: |mean|/sigma={r:.2f} >= {a2_invalid:.2f} "
            f"(variance estimate inflated >=2x; likely setpoint drift "
            f"or unmodeled DC bias)"
        )
    elif r >= a2_unsettled:
        a2 = "UNSETTLED"
        reasons.append(
            f"A2: |mean|/sigma={r:.2f} in [{a2_unsettled:.2f}, "
            f"{a2_invalid:.2f}) (variance inflated 9-100%; DP integral "
            f"or observer bias likely still settling)"
        )
    elif r >= a2_warming:
        a2 = "WARMING"
        reasons.append(
            f"A2: |mean|/sigma={r:.2f} in [{a2_warming:.2f}, "
            f"{a2_unsettled:.2f})"
        )
    else:
        a2 = "OK"

    # A1 stationarity. NaN (window too short) -> skip; A4 already
    # handles the cold case.
    h = health.halves_sigma_ratio
    if not np.isfinite(h):
        a1 = "OK"
    else:
        # Symmetric in log-ratio: ratio and 1/ratio should land in the
        # same band.
        h_eff = max(h, 1.0 / h) if h > 0.0 else float("inf")
        if h_eff <= a1_ok_ratio:
            a1 = "OK"
        elif h_eff <= a1_warm_ratio:
            a1 = "WARMING"
            reasons.append(
                f"A1: halves_sigma_ratio={h:.2f} in warming band "
                f"(|log| <= log({a1_warm_ratio:.2f}))"
            )
        else:
            a1 = "UNSETTLED"
            reasons.append(
                f"A1: halves_sigma_ratio={h:.2f} outside "
                f"[{1.0/a1_warm_ratio:.2f}, {a1_warm_ratio:.2f}] "
                f"(non-stationarity within window: sea-state ramp, "
                f"controller retune, or transient settling)"
            )

    # A3 Gaussianity. NaN -> skip.
    k = health.kurtosis_excess
    if not np.isfinite(k):
        a3 = "OK"
    else:
        ka = abs(k)
        if ka <= a3_ok:
            a3 = "OK"
        elif ka <= a3_warm:
            a3 = "WARMING"
            reasons.append(
                f"A3: |kurtosis_excess|={ka:.2f} in warming band "
                f"(noisy below ~50 raw samples; interpret loosely)"
            )
        else:
            a3 = "UNSETTLED"
            reasons.append(
                f"A3: |kurtosis_excess|={ka:.2f} > {a3_warm:.2f} "
                f"(heavy tails: thruster saturation, slamming, or "
                f"non-Gaussian residuals)"
            )

    # A5 prior-in-CI. False -> WARMING only (informational, data wins).
    if health.prior_in_credible_interval:
        a5 = "OK"
    else:
        a5 = "WARMING"
        reasons.append(
            "A5: prior sigma falls outside the posterior CI "
            "(model-data tension; investigate sea-state, controller "
            "retune, or post-WCFDI state)"
        )

    level = _worst_validity(a4, a2, a1, a3, a5)
    return ValidityBadge(level=level, reasons=tuple(reasons))


# ---------------------------------------------------------------------------
# Decorrelation-time helper
# ---------------------------------------------------------------------------


def closed_loop_decorrelation_time(
    controller: ControllerParams,
    axis: str,
) -> float:
    """Coarse decorrelation-time fallback from the controller bandwidth.

    .. warning::
       This is a *fallback* heuristic for use when the closed-loop
       output PSD is not available. For the production pipeline,
       prefer the per-channel ``pos_T_decorr_var_s`` and
       ``gw_T_decorr_var_s`` fields exposed by
       ``IntactPriorSummary`` (computed via
       ``cqa.extreme_value.variance_decorrelation_time_from_psd``):
       these are the right Bartlett scale for variance estimation,
       integrating the *actual* output spectral shape including any
       narrowband disturbance content (slow-drift, current variability
       with long memory).

       For the canonical CSOV operating point the PSD-derived
       ``T_var`` is ~94 s, vs ~18.5 s from this heuristic -- a 5x
       factor that propagates directly into the posterior credible
       interval on sigma. The heuristic systematically OVER-states
       n_eff for narrowband-driven channels.

    For a critically/over-damped 2nd-order closed loop ``s^2 + 2 zeta
    omega_n s + omega_n^2 = 0`` the dominant pole real part is
    ``zeta * omega_n``, giving an exponential autocovariance scale

        T_decorr = 1 / (zeta * omega_n).

    This is exact only when the closed-loop output PSD is dominated by
    its own broadband response to white-noise input -- not the case in
    DP applications where the disturbance forcing is itself narrowband
    (slow-drift, current variability).

    For the radial position channel, pass ``axis="position"`` to get
    the conservative (largest) of the surge / sway decorrelation
    times -- the radial process is dominated by the slowest contributor.

    Parameters
    ----------
    controller : ControllerParams (from CqaConfig).
    axis : one of {"surge", "sway", "yaw", "position"}.

    Returns
    -------
    T_decorr in seconds.
    """
    axis = axis.lower()
    if axis == "surge":
        return 1.0 / (controller.zeta_surge * controller.omega_n_surge)
    if axis == "sway":
        return 1.0 / (controller.zeta_sway * controller.omega_n_sway)
    if axis == "yaw":
        return 1.0 / (controller.zeta_yaw * controller.omega_n_yaw)
    if axis == "position":
        T_x = 1.0 / (controller.zeta_surge * controller.omega_n_surge)
        T_y = 1.0 / (controller.zeta_sway * controller.omega_n_sway)
        return float(max(T_x, T_y))
    raise ValueError(
        f'axis must be one of "surge","sway","yaw","position"; got {axis!r}'
    )


# ---------------------------------------------------------------------------
# Bayesian estimator
# ---------------------------------------------------------------------------


class BayesianSigmaEstimator:
    """Online Bayesian estimator of sigma^2 over a sliding time window.

    Conjugate inverse-gamma model with effective-sample-size correction
    for autocorrelation. See module docstring for the full derivation.

    Parameters
    ----------
    prior_sigma2 : float. Prior mean of sigma^2 (typically the model-based
        variance from the closed-loop spectrum). [m^2 if X is a position;
        any consistent unit^2 otherwise.]
    T_decorr_s : float. Variance-estimator decorrelation time [s] of
        the channel being observed. The Bartlett ESS uses
        ``n_eff = n_raw * dt / max(dt, T_decorr_s)``.
        Recommended source:
        ``IntactPriorSummary.pos_T_decorr_var_s`` (or
        ``gw_T_decorr_var_s``) computed via
        ``variance_decorrelation_time_from_psd`` from the same
        closed-loop output PSD that built the prior.
        ``closed_loop_decorrelation_time(cfg.controller, axis)`` is a
        *coarse fallback* (the bandwidth-only proxy
        ``1/(zeta*omega_n)``) that significantly under-estimates
        T_decorr when the disturbance spectrum is narrowband relative
        to the closed loop.
    dt_s : float. Sample period of the observed channel [s]. The sliding
        window will hold ``floor(window_s / dt_s)`` raw samples.
    prior_strength_n0 : float, default 2.0. "Effective prior sample
        count" parameterising the inverse-gamma prior shape. The
        InvGamma prior is built so that ``alpha_0 = prior_strength_n0/2``
        (must be > 1 for a finite prior mean) and
        ``beta_0 = prior_sigma2 * (alpha_0 - 1)``. Default 2.0 yields
        alpha_0 = 1 (improper-mean limit) which we softly clip to
        alpha_0 = 1 + 1e-6 internally so the prior mean is finite.
        Increase to anchor more strongly to the prior.
    window_s : float, default 60.0. Length of the sliding window [s].
    assume_zero_mean : bool, default True. The DP regulates X to its
        setpoint, so X ~ zero-mean is the right model. Set False to
        subtract the windowed sample mean before forming the sufficient
        statistic (loses 1 dof, almost always not worth it for DP-class
        signals).

    Attributes
    ----------
    n_raw_capacity : int. Maximum raw samples held in the window.
    """

    def __init__(
        self,
        prior_sigma2: float,
        T_decorr_s: float,
        dt_s: float,
        prior_strength_n0: float = 2.0,
        window_s: float = 60.0,
        assume_zero_mean: bool = True,
    ):
        if prior_sigma2 <= 0.0:
            raise ValueError(f"prior_sigma2 must be > 0, got {prior_sigma2}")
        if T_decorr_s <= 0.0:
            raise ValueError(f"T_decorr_s must be > 0, got {T_decorr_s}")
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be > 0, got {dt_s}")
        if prior_strength_n0 <= 0.0:
            raise ValueError(
                f"prior_strength_n0 must be > 0, got {prior_strength_n0}"
            )
        if window_s <= 0.0:
            raise ValueError(f"window_s must be > 0, got {window_s}")

        self.prior_sigma2 = float(prior_sigma2)
        self.prior_strength_n0 = float(prior_strength_n0)
        self.T_decorr_s = float(T_decorr_s)
        self.dt_s = float(dt_s)
        self.window_s = float(window_s)
        self.assume_zero_mean = bool(assume_zero_mean)

        self.n_raw_capacity = int(np.floor(self.window_s / self.dt_s))
        if self.n_raw_capacity < 1:
            raise ValueError(
                f"window_s/dt_s = {self.window_s}/{self.dt_s} gives "
                f"capacity {self.n_raw_capacity} < 1"
            )

        # Inverse-gamma prior parameters. Soft-clip alpha_0 to >1 so the
        # prior mean is finite (default n0=2 hits alpha_0=1 exactly).
        alpha_0 = 0.5 * self.prior_strength_n0
        if alpha_0 <= 1.0:
            alpha_0 = 1.0 + 1e-6
        self._alpha_0 = float(alpha_0)
        self._beta_0 = float(self.prior_sigma2 * (self._alpha_0 - 1.0))

        # Ring buffer of raw samples. Maintain incremental sum of squares
        # for O(1) update: when a sample falls off the back we subtract
        # its squared value; when a new sample comes in we add its
        # squared value.
        self._buffer: deque[float] = deque(maxlen=self.n_raw_capacity)
        self._sum_sq: float = 0.0
        # If we choose to subtract a sample mean (non-default), we also
        # need a running sum:
        self._sum: float = 0.0

    # ----- mutators -----

    def update(self, sample: float) -> None:
        """Push one new observation into the sliding window."""
        x = float(sample)
        if len(self._buffer) == self.n_raw_capacity:
            # Buffer full -> evict the oldest sample.
            x_old = self._buffer[0]
            self._sum_sq -= x_old * x_old
            self._sum -= x_old
        self._buffer.append(x)
        self._sum_sq += x * x
        self._sum += x

    def reset(self) -> None:
        """Clear the sliding window. Posterior collapses to the prior."""
        self._buffer.clear()
        self._sum_sq = 0.0
        self._sum = 0.0

    # ----- introspection -----

    @property
    def n_raw(self) -> int:
        """Current number of raw samples in the window."""
        return len(self._buffer)

    @property
    def n_eff(self) -> float:
        """Effective independent sample count after Bartlett correction.

        ``n_eff = n_raw * dt / max(dt, T_decorr)``. With dt=1 s and
        T_decorr=20 s this gives n_eff = n_raw / 20.
        """
        return self.n_raw * self.dt_s / max(self.dt_s, self.T_decorr_s)

    def is_warm(self, n_eff_threshold: float = 1.0) -> bool:
        """True once the window holds at least one effectively-
        independent sample. Below this threshold the posterior is
        essentially the prior and reporting it as a "data-driven" sigma
        would be misleading.

        Parameters
        ----------
        n_eff_threshold : float, default 1.0. Effective sample count
            above which the posterior is considered "warm".
        """
        return self.n_eff >= n_eff_threshold

    # ----- posterior -----

    def posterior(self, credible: float = 0.90) -> SigmaPosterior:
        """Return the current inverse-gamma posterior on sigma^2.

        Parameters
        ----------
        credible : float in (0, 1). Credible level for the equal-tail
            interval reported as (sigma2_lo, sigma2_hi). Default 0.90
            (5th and 95th percentiles).

        Notes
        -----
        * If the window is empty, the posterior equals the prior.
        * The sufficient statistic is
              S      = sum_i x_i^2  (zero-mean assumption), or
              S      = sum_i (x_i - x_bar)^2  if assume_zero_mean=False
          rescaled to the effective sample count:
              S_eff  = S * n_eff / n_raw.
        * Equal-tail interval uses the inverse CDF of InvGamma(alpha,
          beta).
        """
        from scipy.stats import invgamma

        if not (0.0 < credible < 1.0):
            raise ValueError(f"credible must be in (0, 1), got {credible}")

        n_raw = self.n_raw
        n_eff = self.n_eff

        if n_raw == 0:
            S = 0.0
            S_eff = 0.0
        else:
            if self.assume_zero_mean:
                S = float(self._sum_sq)
            else:
                # Sample-mean-corrected sufficient statistic; loses
                # 1 dof.
                x_bar = self._sum / n_raw
                S = float(self._sum_sq - n_raw * x_bar * x_bar)
            # Rescale to match the effective sample count.
            if n_raw > 0:
                S_eff = S * (n_eff / n_raw)
            else:
                S_eff = 0.0

        # Posterior parameters.
        alpha = self._alpha_0 + 0.5 * n_eff
        beta = self._beta_0 + 0.5 * S_eff

        # Posterior mean and median on sigma^2.
        if alpha > 1.0:
            sigma2_mean = float(beta / (alpha - 1.0))
        else:
            sigma2_mean = float("inf")

        # scipy invgamma is parameterised by shape `a` and scale; for
        # InvGamma(alpha, beta) we pass a=alpha, scale=beta.
        rv = invgamma(a=alpha, scale=beta)
        sigma2_median = float(rv.median())
        lo_q = 0.5 * (1.0 - credible)
        hi_q = 1.0 - lo_q
        sigma2_lo = float(rv.ppf(lo_q))
        sigma2_hi = float(rv.ppf(hi_q))

        sigma_mean = float(np.sqrt(max(sigma2_mean, 0.0)))
        sigma_median = float(np.sqrt(max(sigma2_median, 0.0)))
        sigma_lo = float(np.sqrt(max(sigma2_lo, 0.0)))
        sigma_hi = float(np.sqrt(max(sigma2_hi, 0.0)))

        return SigmaPosterior(
            sigma2_mean=sigma2_mean,
            sigma2_median=sigma2_median,
            sigma2_lo=sigma2_lo,
            sigma2_hi=sigma2_hi,
            sigma_mean=sigma_mean,
            sigma_median=sigma_median,
            sigma_lo=sigma_lo,
            sigma_hi=sigma_hi,
            n_raw=int(n_raw),
            n_eff=float(n_eff),
            alpha=float(alpha),
            beta=float(beta),
            prior_sigma2=self.prior_sigma2,
            prior_strength_n0=self.prior_strength_n0,
            credible=float(credible),
        )

    # ----- diagnostics -----

    def health(
        self,
        credible: float = 0.90,
        n_eff_threshold: float = 1.0,
    ) -> PosteriorHealth:
        """Cheap runtime diagnostics on the posterior assumptions.

        Pure function over the buffer; does not mutate state.

        Parameters
        ----------
        credible : float in (0, 1). Credible level used to compute the
            interval against which ``prior_in_credible_interval`` is
            tested. Default 0.90 (matches the default in ``posterior``).
        n_eff_threshold : float. Threshold for the ``is_warm`` flag,
            forwarded to ``is_warm()``. Default 1.0.

        Returns
        -------
        PosteriorHealth. See that dataclass for field semantics. NaN is
        used for any primitive that is undefined at the current sample
        count (e.g. kurtosis with n_raw < 4).
        """
        n_raw = self.n_raw
        n_eff = self.n_eff
        warm = self.is_warm(n_eff_threshold=n_eff_threshold)

        if n_raw == 0:
            nan = float("nan")
            return PosteriorHealth(
                n_raw=0,
                n_eff=float(n_eff),
                is_warm=warm,
                n_eff_threshold=float(n_eff_threshold),
                sample_mean=nan,
                sample_mean_over_sigma=nan,
                prior_in_credible_interval=True,  # posterior == prior
                kurtosis_excess=nan,
                halves_sigma_ratio=nan,
            )

        # Sample mean. Cheap from the running sum.
        sample_mean = float(self._sum / n_raw)

        # Posterior median sigma -- needed for the A2 ratio and for the
        # prior-in-CI test. Reuse posterior() so we get the same Bartlett
        # ESS and the same credible interval semantics.
        post = self.posterior(credible=credible)

        if post.sigma_median > 0.0:
            sample_mean_over_sigma = float(abs(sample_mean) / post.sigma_median)
        else:
            sample_mean_over_sigma = float("nan")

        prior_sigma = float(np.sqrt(self.prior_sigma2))
        prior_in_ci = bool(post.sigma_lo <= prior_sigma <= post.sigma_hi)

        # Sample kurtosis (excess). Use the zero-mean form to be
        # consistent with the estimator's working assumption; this also
        # avoids confounding A2 (mean offset) with A3 (heavy tails).
        if n_raw >= 4:
            x = np.fromiter(self._buffer, dtype=float, count=n_raw)
            if self.assume_zero_mean:
                m2 = float(np.mean(x * x))
                m4 = float(np.mean(x * x * x * x))
            else:
                xc = x - np.mean(x)
                m2 = float(np.mean(xc * xc))
                m4 = float(np.mean(xc * xc * xc * xc))
            if m2 > 0.0:
                kurtosis_excess = float(m4 / (m2 * m2) - 3.0)
            else:
                kurtosis_excess = float("nan")

            # Halves split for stationarity. Zero-mean RMS of each half.
            half = n_raw // 2
            x1 = x[:half]
            x2 = x[-half:]
            if self.assume_zero_mean:
                s1 = float(np.sqrt(np.mean(x1 * x1)))
                s2 = float(np.sqrt(np.mean(x2 * x2)))
            else:
                s1 = float(np.std(x1, ddof=0))
                s2 = float(np.std(x2, ddof=0))
            if s1 > 0.0:
                halves_sigma_ratio = float(s2 / s1)
            else:
                halves_sigma_ratio = float("nan")
        else:
            kurtosis_excess = float("nan")
            halves_sigma_ratio = float("nan")

        return PosteriorHealth(
            n_raw=int(n_raw),
            n_eff=float(n_eff),
            is_warm=warm,
            n_eff_threshold=float(n_eff_threshold),
            sample_mean=sample_mean,
            sample_mean_over_sigma=sample_mean_over_sigma,
            prior_in_credible_interval=prior_in_ci,
            kurtosis_excess=kurtosis_excess,
            halves_sigma_ratio=halves_sigma_ratio,
        )


__all__ = [
    "SigmaPosterior",
    "PosteriorHealth",
    "RadialPosterior",
    "BayesianSigmaEstimator",
    "combine_radial_posterior",
    "closed_loop_decorrelation_time",
]
