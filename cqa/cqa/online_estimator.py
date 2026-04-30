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
For a 2nd-order closed loop with natural frequency omega_n and
damping zeta, the dominant pole real part is ``zeta * omega_n``.
The exponential autocovariance scale is therefore

    T_decorr = 1 / (zeta * omega_n).

For the radial gangway-base position channel, surge and sway
contribute via the linear combination
``c_base_x = [1, 0, -base_y]``, ``c_base_y = [0, 1, base_x]``.
We take the slowest (max) decorrelation time across the two
horizontal axes as the conservative scale.

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
# Decorrelation-time helper
# ---------------------------------------------------------------------------


def closed_loop_decorrelation_time(
    controller: ControllerParams,
    axis: str,
) -> float:
    """Decorrelation time of one closed-loop axis.

    For a critically/over-damped 2nd-order closed loop ``s^2 + 2 zeta
    omega_n s + omega_n^2 = 0`` the dominant pole real part is
    ``zeta * omega_n``, giving an exponential autocovariance scale

        T_decorr = 1 / (zeta * omega_n).

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
    T_decorr_s : float. Closed-loop decorrelation time [s]. Use
        `closed_loop_decorrelation_time(cfg.controller, axis)` to derive
        from the controller tuning.
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


__all__ = [
    "SigmaPosterior",
    "BayesianSigmaEstimator",
    "closed_loop_decorrelation_time",
]
