"""Extreme-value statistics for the operator-facing intact-prior layer.

Goal
----
Answer the operator question "what is the probability that this scalar
process X(t) breaches level a at any point during the next T_op
seconds?" in closed form, given:

  * the steady-state std dev sigma of X(t) (the position-axis or
    telescope-length output of the linearised closed loop), and
  * the spectrum S_X(omega) of X(t) (used to compute the
    zero-up-crossing rate).

Theory (Rice / Cartwright-Longuet-Higgins, narrowband Gaussian)
---------------------------------------------------------------
For a stationary Gaussian process X(t) with one-sided spectrum
S_X(omega), the spectral moments are

    lambda_k = integral_0^inf omega^k S_X(omega) d omega.

Variance is sigma^2 = lambda_0. The mean rate of zero up-crossings is

    nu_0+ = (1 / 2 pi) * sqrt(lambda_2 / lambda_0).            (1)

For a stationary Gaussian process with high (relative to sigma) level
a, the mean rate of up-crossings of the level a is

    nu_a+ = nu_0+ * exp(-a^2 / (2 sigma^2)).                   (2)

Using the standard Poisson approximation for rare exceedances of a
Gaussian process, the number of up-crossings of +a in [0, T] is
approximately Poisson with rate nu_a+. The probability of NO up-crossing
in T is therefore

    P(no up-cross of +a in T) = exp(-nu_a+ * T).               (3)

For the bilateral case (excursion of |X| above a), we double the rate
to count up-crossings of +a AND down-crossings of -a:

    P(|X(t)| <= a for all t in [0, T]) ~ exp(-2 nu_a+ T).      (4)

So:

    P_breach(T, a) = 1 - exp(-2 nu_0+ T * exp(-a^2 / (2 sigma^2))). (5)

This formula is valid for **rare** exceedances (a/sigma >~ 2). For
small a/sigma the Poisson approximation breaks down (clumping of
crossings); we clamp the formula's domain accordingly and emit a
diagnostic for low ratios.

Bandwidth correction (Vanmarcke 1975)
-------------------------------------
The plain Poisson assumption (eq. 3) treats up-crossings as
independent point events. For a NARROWBAND process this is wrong:
each "envelope swell" produces several closely-spaced up-crossings,
so the true probability of NO crossing in [0, T] is HIGHER than
exp(-nu_a+ T) -- equivalently the true P_breach is LOWER. Rice/Poisson
is conservative for narrowband.

Vanmarcke (1975) gives a closed-form correction interpolating between
the wideband (Poisson) limit and the narrowband (envelope) limit:

    P(no |X| crossing of a in [0, T])
        ~ exp(-nu_0+ * T * (1 - exp(-sqrt(pi/2) * q * a/sigma))
                          / (1 - exp(-a^2/(2 sigma^2))) * exp(-a^2/(2 sigma^2)))

where q is the Vanmarcke spectral bandwidth parameter

    q = sqrt(1 - lambda_1^2 / (lambda_0 lambda_2)),    q in [0, 1]    (7)

q -> 0 (narrowband):   matches the Rayleigh envelope rate.
q -> 1 (wideband):     reduces to the Rice/Poisson rate.

Practical regime guide (DNV-RP-C205, Naess & Moan):
    q < 0.3   narrowband -- Poisson over-estimates by factor 2-5
    0.3-0.6  transition  -- Poisson over-estimates by factor 1.5-3
    q > 0.6   wideband   -- Poisson is fine

Default in this module: clustering="vanmarcke" everywhere we have a
PSD (auto-computes q). For cases where only (sigma, nu0+) are
available we fall back to Poisson and emit a diagnostic.

Multi-band processes (e.g. slow-drift + wave-frequency telescope length)
------------------------------------------------------------------------
When X is the sum of two spectrally well-separated Gaussian processes
X_1 (slow, ~0.01 Hz) and X_2 (wave, ~0.1 Hz), the up-crossings of |X|
above level a are dominated by the contribution from each band
independently in the rare-exceedance regime. Each band contributes
its own Rice rate, and the combined no-crossing probability factorises:

    P_breach_combined(T, a) ~ 1 - exp(-T * sum_i 2 nu_a+,i).    (6)

This is conservative when the bands have comparable variance (we
ignore the fact that 'a' should be measured against the joint sigma
sqrt(sum sigma_i^2), not each band's sigma separately) but it's the
right structure for two genuinely separated bands like ours, and it
admits a clean per-band attribution for the operator (slow vs wave
contribution to the breach probability).

Production / prototype boundary
-------------------------------
Pure numpy. Inputs are sigma (m) and either nu_0+ (Hz) directly or a
PSD callable that we numerically integrate. Outputs are scalars in the
range [0, 1] (probabilities). No vessel/controller knowledge needed
here -- the spectral moments are produced upstream by the closed-loop
machinery in cqa/closed_loop.py.

References
----------
* Naess, A. & Moan, T. (2013), "Stochastic Dynamics of Marine
  Structures", ch. 6 (Cartwright-Longuet-Higgins, Rice formula).
* Madsen, P.H., Krenk, S., Lind, N.C. (1986), "Methods of Structural
  Safety", ch. 9 (Poisson approximation for excursions of Gaussian
  processes).
* DNV-RP-C205 sec. 3.5.6 (extreme-value distribution of narrowband
  responses).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import numpy as np


# ---------------------------------------------------------------------------
# Spectral moments
# ---------------------------------------------------------------------------


def spectral_moments(
    S_X: np.ndarray,
    omega: np.ndarray,
    orders: Sequence[int] = (0, 2),
) -> dict[int, float]:
    """Compute spectral moments lambda_k = int omega^k S_X(omega) d omega.

    Parameters
    ----------
    S_X : (n,) array of one-sided PSD values [unit_X^2 / (rad/s)].
    omega : (n,) array of angular frequencies [rad/s], strictly increasing.
    orders : iterable of integer moment orders. Default (0, 2) is
        sufficient for the zero-up-crossing rate.

    Returns
    -------
    dict mapping order -> moment value.
    """
    S_X = np.asarray(S_X, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if S_X.shape != omega.shape:
        raise ValueError(
            f"S_X and omega must have the same shape, got "
            f"{S_X.shape} vs {omega.shape}"
        )
    out: dict[int, float] = {}
    for k in orders:
        integrand = (omega ** k) * S_X
        out[int(k)] = float(np.trapezoid(integrand, omega))
    return out


def zero_upcrossing_rate(
    S_X: np.ndarray,
    omega: np.ndarray,
) -> float:
    """Mean zero-up-crossing rate nu_0+ [Hz] of a stationary Gaussian
    process with one-sided spectrum S_X(omega).

        nu_0+ = (1 / 2 pi) * sqrt(lambda_2 / lambda_0).

    Returns 0.0 if lambda_0 is zero (degenerate spectrum).
    """
    m = spectral_moments(S_X, omega, orders=(0, 2))
    lam0, lam2 = m[0], m[2]
    if lam0 <= 0.0:
        return 0.0
    if lam2 <= 0.0:
        return 0.0
    return float(np.sqrt(lam2 / lam0) / (2.0 * np.pi))


def vanmarcke_bandwidth_q(
    S_X: np.ndarray,
    omega: np.ndarray,
) -> float:
    """Vanmarcke (1975) spectral bandwidth parameter q in [0, 1].

        q = sqrt(1 - lambda_1^2 / (lambda_0 * lambda_2)).

    Interpretation:
      * q -> 0   process is narrowband (single sharp peak)
      * q -> 1   process is wideband (white noise on a finite band)

    Used by the Vanmarcke correction in p_exceed_rice to interpolate
    between the wideband Poisson limit and the narrowband envelope
    limit. Numerically clamped to [0, 1] (small negative values from
    quadrature noise are returned as 0).

    Returns 1.0 (wideband / no clustering) if lambda_0*lambda_2 is
    zero.
    """
    m = spectral_moments(S_X, omega, orders=(0, 1, 2))
    lam0, lam1, lam2 = m[0], m[1], m[2]
    denom = lam0 * lam2
    if denom <= 0.0:
        return 1.0
    arg = 1.0 - (lam1 * lam1) / denom
    # Clamp small negative quadrature error to 0 and >1 to 1.
    arg = max(0.0, min(1.0, arg))
    return float(np.sqrt(arg))


def clh_epsilon(
    S_X: np.ndarray,
    omega: np.ndarray,
) -> float:
    """Cartwright-Longuet-Higgins (1956) bandwidth parameter
    epsilon in [0, 1].

        epsilon = sqrt(1 - lambda_2^2 / (lambda_0 * lambda_4)).

    Older bandwidth measure; uses lambda_4 which is dominated by the
    high-frequency tail and therefore quite sensitive to spectral
    cutoff. We prefer Vanmarcke's q for the Poisson-vs-envelope
    interpolation, but expose epsilon as a diagnostic.

    Returns 1.0 if lambda_0*lambda_4 is zero or lambda_4 is not
    finite (e.g. a process whose spectrum doesn't decay fast enough).
    """
    m = spectral_moments(S_X, omega, orders=(0, 2, 4))
    lam0, lam2, lam4 = m[0], m[2], m[4]
    if lam0 <= 0.0 or lam4 <= 0.0:
        return 1.0
    arg = 1.0 - (lam2 * lam2) / (lam0 * lam4)
    arg = max(0.0, min(1.0, arg))
    return float(np.sqrt(arg))


# ---------------------------------------------------------------------------
# Rice exceedance probability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiceExceedanceResult:
    """Detailed result of a Rice-formula exceedance computation."""

    p_breach: float          # in [0, 1], probability of |X| > a anywhere in [0, T]
    expected_count: float    # E[number of |X| > a excursions in [0, T]] (Poisson rate * T)
    nu_0_plus: float         # zero-up-crossing rate [Hz]
    nu_a_plus: float         # mean rate of up-crossings of +a [Hz]
    sigma: float             # std dev of X
    threshold: float         # a
    T: float                 # operation duration [s]
    rarity_ratio: float      # a / sigma; meaningful >= ~2 for the Poisson approx
    valid: bool              # True if rarity_ratio >= rarity_min
    clustering: str          # "poisson" | "vanmarcke"
    q: float                 # Vanmarcke bandwidth parameter (1.0 if unknown / Poisson)


def p_exceed_rice(
    sigma: float,
    nu_0_plus: float,
    threshold: float,
    T: float,
    bilateral: bool = True,
    rarity_min: float = 2.0,
    clustering: str = "vanmarcke",
    q: Optional[float] = None,
) -> RiceExceedanceResult:
    """Probability of |X| breaching a level over a duration T (Rice + Vanmarcke).

    For a stationary Gaussian X(t) with std sigma and zero-up-crossing
    rate nu_0+, with threshold a = `threshold`:

        nu_a+ = nu_0+ * exp(-a^2 / (2 sigma^2))         (Rice, eq. 2)

    The number of upward crossings of level +a in [0, T] is approximated
    as a Poisson process with rate nu_a+, but with a CLUSTERING
    CORRECTION when the process is narrowband. Two clustering models:

      "poisson"  : assume independent crossings (the textbook Rice
                   formula). Exact for white noise; over-estimates
                   P_breach for narrowband processes.
      "vanmarcke": Vanmarcke (1975) interpolation between Poisson and
                   the narrowband envelope-peak limit. Requires the
                   spectral bandwidth parameter q in [0, 1]; q=1 ->
                   matches Poisson, q=0 -> matches Rayleigh envelope.
                   Default. Falls back to Poisson with a diagnostic
                   if q is not provided.

    Bilateral (default) counts excursions of |X| above a, doubling the
    expected count.

    Parameters
    ----------
    sigma : std dev of the process [unit_X].
    nu_0_plus : zero-up-crossing rate [Hz].
    threshold : level a (>0) [unit_X].
    T : exposure duration [s].
    bilateral : if True, count both upward and downward excursions.
    rarity_min : a/sigma below which the Poisson approximation breaks
        down. Result.valid is False below this.
    clustering : "vanmarcke" (default) or "poisson".
    q : Vanmarcke bandwidth parameter. Required if clustering=
        "vanmarcke" and you want the correction; otherwise the function
        falls back to Poisson with a warning recorded as
        clustering="poisson" in the result.

    Notes
    -----
    * sigma == 0 or nu_0_plus == 0 => P_breach = 0.
    * threshold == 0 => P_breach = 1.
    * The expected count returned is the *Poisson* expected count
      (2*nu_a+*T or nu_a+*T): per-band attribution and multi-band
      combination compose cleanly with this. The Vanmarcke correction
      is applied AFTER summing per-band Poisson counts (see
      p_exceed_rice_multiband).
    """
    sigma = float(sigma)
    nu_0_plus = float(nu_0_plus)
    threshold = float(threshold)
    T = float(T)
    if T < 0.0:
        raise ValueError(f"T must be >= 0, got {T}")
    if threshold < 0.0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if nu_0_plus < 0.0:
        raise ValueError(f"nu_0_plus must be >= 0, got {nu_0_plus}")
    if clustering not in ("vanmarcke", "poisson"):
        raise ValueError(
            f'clustering must be "vanmarcke" or "poisson", got {clustering!r}'
        )

    if sigma == 0.0 or nu_0_plus == 0.0 or T == 0.0:
        return RiceExceedanceResult(
            p_breach=0.0, expected_count=0.0,
            nu_0_plus=nu_0_plus, nu_a_plus=0.0, sigma=sigma,
            threshold=threshold, T=T,
            rarity_ratio=(threshold / sigma) if sigma > 0 else float("inf"),
            valid=True, clustering=clustering,
            q=float(q) if q is not None else 1.0,
        )
    if threshold == 0.0:
        return RiceExceedanceResult(
            p_breach=1.0, expected_count=float("inf"),
            nu_0_plus=nu_0_plus, nu_a_plus=nu_0_plus, sigma=sigma,
            threshold=threshold, T=T,
            rarity_ratio=0.0, valid=False, clustering=clustering,
            q=float(q) if q is not None else 1.0,
        )

    rarity = threshold / sigma
    nu_a = nu_0_plus * np.exp(-(threshold ** 2) / (2.0 * sigma ** 2))
    bilateral_factor = 2.0 if bilateral else 1.0
    expected_count_poisson = bilateral_factor * nu_a * T

    # Vanmarcke clustering correction.
    # Effective Poisson exponent is multiplied by the Vanmarcke factor
    # phi(a, sigma, q) = (1 - exp(-sqrt(pi/2) * q * a/sigma))
    #                  / (1 - exp(-a^2/(2 sigma^2)))
    # which is in (0, 1] and -> 1 as q -> 1.
    if clustering == "vanmarcke" and q is not None and q < 1.0:
        q_eff = max(0.0, min(1.0, float(q)))
        ratio = threshold / sigma
        num = 1.0 - np.exp(-np.sqrt(np.pi / 2.0) * q_eff * ratio)
        den = 1.0 - np.exp(-(ratio ** 2) / 2.0)
        # den is bounded away from zero whenever ratio > 0; guard for
        # extreme small-ratio underflow.
        if den > 1e-300:
            phi = num / den
        else:
            phi = 1.0
        phi = max(0.0, min(1.0, float(phi)))
        effective_count = expected_count_poisson * phi
        clustering_used = "vanmarcke"
        q_used = q_eff
    else:
        # Either clustering=poisson, or vanmarcke requested without q.
        effective_count = expected_count_poisson
        clustering_used = "poisson"
        q_used = float(q) if q is not None else 1.0

    p_breach = float(1.0 - np.exp(-effective_count))
    return RiceExceedanceResult(
        p_breach=p_breach,
        expected_count=float(expected_count_poisson),
        nu_0_plus=nu_0_plus,
        nu_a_plus=float(nu_a),
        sigma=sigma,
        threshold=threshold,
        T=T,
        rarity_ratio=float(rarity),
        valid=bool(rarity >= rarity_min),
        clustering=clustering_used,
        q=q_used,
    )


def p_exceed_rice_multiband(
    bands: Sequence[tuple],
    threshold: float,
    T: float,
    bilateral: bool = True,
    clustering: str = "vanmarcke",
) -> dict[str, float]:
    """Combine independent spectral bands into one P_breach.

    Each band entry is either:
      * (sigma_i, nu_0_plus_i)            -- legacy 2-tuple, q assumed
                                             1 (Poisson) for that band.
      * (sigma_i, nu_0_plus_i, q_i)       -- with Vanmarcke bandwidth.

    The Vanmarcke clustering correction is applied PER BAND (each band
    has its own bandwidth q_i), then the corrected expected counts add
    (independent-Poisson product of no-crossing probabilities):

        P_breach = 1 - exp( - sum_i corrected_count_i ).

    The threshold is applied PER BAND (same level a tested against each
    band's sigma). This is the right structure for the "slow-drift +
    wave-frequency telescope" combination, where the slow band's sigma
    governs the slow excursions and the wave band's sigma governs the
    wave-frequency peaks; both have to stay inside the same end-stop
    a = L_max - L_setpoint.

    Returns
    -------
    dict with keys:
        "p_breach": combined exceedance probability in [0, 1]
        "expected_count_total": sum of per-band CORRECTED expected counts
        "expected_count_per_band": list[float], per-band corrected count
        "p_breach_per_band": list[float], per-band marginal P_breach
            (useful for attribution / "which band drove the alarm")
        "q_per_band": list[float]
    """
    per_band_counts: list[float] = []
    per_band_p: list[float] = []
    per_band_q: list[float] = []
    for entry in bands:
        if len(entry) == 2:
            sigma_i, nu0_i = entry
            q_i = None
        elif len(entry) == 3:
            sigma_i, nu0_i, q_i = entry
        else:
            raise ValueError(
                f"band entries must be (sigma, nu0) or (sigma, nu0, q); "
                f"got length {len(entry)}"
            )
        r = p_exceed_rice(
            sigma=sigma_i, nu_0_plus=nu0_i,
            threshold=threshold, T=T, bilateral=bilateral,
            clustering=clustering, q=q_i,
        )
        # The "corrected count" is -log(1 - p_breach) since
        # p_breach = 1 - exp(-corrected_count) per band. This composes
        # cleanly across independent bands.
        if r.p_breach >= 1.0 - 1e-300:
            corrected_count = float("inf")
        else:
            corrected_count = float(-np.log(max(1.0 - r.p_breach, 1e-300)))
        per_band_counts.append(corrected_count)
        per_band_p.append(r.p_breach)
        per_band_q.append(r.q)
    total_count = float(sum(per_band_counts))
    p_combined = float(1.0 - np.exp(-total_count))
    return {
        "p_breach": p_combined,
        "expected_count_total": total_count,
        "expected_count_per_band": per_band_counts,
        "p_breach_per_band": per_band_p,
        "q_per_band": per_band_q,
    }


# ---------------------------------------------------------------------------
# Convenience wrappers from PSD
# ---------------------------------------------------------------------------


def p_exceed_from_psd(
    S_X: np.ndarray,
    omega: np.ndarray,
    threshold: float,
    T: float,
    bilateral: bool = True,
    sigma_override: Optional[float] = None,
    clustering: str = "vanmarcke",
) -> RiceExceedanceResult:
    """End-to-end: take a one-sided PSD S_X(omega), threshold, duration,
    and return a Rice exceedance result with Vanmarcke clustering
    correction (auto-computed q from the PSD).

    sigma is derived from sqrt(lambda_0) of the same PSD unless
    `sigma_override` is given (used to inject a Bayesian-updated
    sigma later: prior PSD shape, posterior variance).

    The Vanmarcke bandwidth parameter q is always computed from the
    full PSD shape, even when sigma_override is provided -- this is
    deliberate: a Bayesian-updated variance does not change the spectral
    SHAPE of the closed-loop response, only its level.
    """
    nu0 = zero_upcrossing_rate(S_X, omega)
    q = vanmarcke_bandwidth_q(S_X, omega) if clustering == "vanmarcke" else None
    if sigma_override is None:
        m = spectral_moments(S_X, omega, orders=(0,))
        sigma = float(np.sqrt(max(m[0], 0.0)))
    else:
        sigma = float(sigma_override)
    return p_exceed_rice(
        sigma=sigma, nu_0_plus=nu0, threshold=threshold, T=T,
        bilateral=bilateral, clustering=clustering, q=q,
    )


# ---------------------------------------------------------------------------
# Inverse Rice: solve for the level a such that P_breach(T, a) = p
# ---------------------------------------------------------------------------


def inverse_rice(
    p: float,
    sigma: float,
    nu_0_plus: float,
    T: float,
    bilateral: bool = True,
    clustering: str = "vanmarcke",
    q: Optional[float] = None,
    a_max_sigma: float = 8.0,
    tol: float = 1e-9,
    max_iter: int = 80,
) -> float:
    """Inverse of `p_exceed_rice`: find level a such that P_breach = p.

    The forward map a -> P_breach is monotone DECREASING (a higher
    threshold is less likely to be breached). We invert to obtain the
    p-quantile of the running maximum of |X(t)| over [0, T]:

        a(p) = level such that P( max_{t in [0,T]} |X(t)| > a ) = p.

    For `clustering="poisson"` (or `q is None`) there is a closed form
    derived from the bilateral Rice / Poisson formula
        P_breach = 1 - exp( -2 nu_0+ T exp(-a^2 / (2 sigma^2)) )
    which solves to
        a(p) = sigma * sqrt( -2 ln( -ln(1 - p) / (b * nu_0+ * T) ) )
    with b = 2 (bilateral) or 1 (unilateral). When the argument of the
    outer log is non-positive (p so small the corresponding count is
    < the rate * T offset), we return 0.

    For `clustering="vanmarcke"` we run a 1-D bisection on
        f(a) = p_exceed_rice(a) - p
    over a in [0, a_max_sigma * sigma]. f is continuous and monotone
    decreasing in a, so bisection is safe. Typically converges in
    20-40 iterations to tol=1e-9.

    Edge cases
    ----------
    sigma == 0 or nu_0+ == 0 or T == 0  -> P_breach = 0 for any a > 0.
        We return 0 (any a >= 0 satisfies the equation in the limit).
    p == 0   -> a = +inf in principle. Return a_max_sigma * sigma as a
        finite practical bound.
    p == 1   -> a = 0.
    p outside (0, 1)  -> ValueError.

    Returns
    -------
    a : float, the inverse-Rice quantile in the same units as sigma.
    """
    p = float(p)
    sigma = float(sigma)
    nu_0_plus = float(nu_0_plus)
    T = float(T)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if nu_0_plus < 0.0:
        raise ValueError(f"nu_0_plus must be >= 0, got {nu_0_plus}")
    if T < 0.0:
        raise ValueError(f"T must be >= 0, got {T}")

    if sigma == 0.0 or nu_0_plus == 0.0 or T == 0.0:
        return 0.0
    if p == 1.0:
        return 0.0
    if p == 0.0:
        return float(a_max_sigma * sigma)

    if clustering not in ("vanmarcke", "poisson"):
        raise ValueError(
            f'clustering must be "vanmarcke" or "poisson", got {clustering!r}'
        )

    use_vanmarcke = (clustering == "vanmarcke" and q is not None and q < 1.0)

    if not use_vanmarcke:
        # Closed-form Poisson inverse.
        b = 2.0 if bilateral else 1.0
        # P_breach = 1 - exp(-b * nu_0+ * T * exp(-a^2/(2 sigma^2)))
        #   => -ln(1 - p) = b * nu_0+ * T * exp(-a^2/(2 sigma^2))
        #   => a^2 = -2 sigma^2 ln( -ln(1-p) / (b * nu_0+ * T) )
        log1mp = -np.log(1.0 - p)            # > 0 since p in (0,1)
        rate_T = b * nu_0_plus * T
        if rate_T <= 0.0:
            return float(a_max_sigma * sigma)
        ratio = log1mp / rate_T
        if ratio >= 1.0:
            # The mean total count over T at threshold 0 is rate_T;
            # if log1mp >= rate_T, the requested probability is so high
            # that even a = 0 doesn't reach it. Return 0.
            return 0.0
        # outer log: ln(ratio) is negative; -2 ln(ratio) > 0
        a2 = -2.0 * sigma * sigma * np.log(ratio)
        return float(np.sqrt(max(a2, 0.0)))

    # Vanmarcke branch: bisection.
    a_lo = 0.0
    a_hi = float(a_max_sigma * sigma)

    def f(a: float) -> float:
        r = p_exceed_rice(
            sigma=sigma, nu_0_plus=nu_0_plus, threshold=a,
            T=T, bilateral=bilateral, clustering="vanmarcke", q=q,
        )
        return r.p_breach - p

    f_lo = f(a_lo)
    f_hi = f(a_hi)
    # f(0) = 1 - p > 0; f(a_hi) typically -p < 0. If both same sign,
    # extend a_hi (rare; means a_max_sigma was too small).
    grow_iter = 0
    while f_hi > 0.0 and grow_iter < 6:
        a_hi *= 2.0
        f_hi = f(a_hi)
        grow_iter += 1
    if f_hi > 0.0:
        # Even at a very large threshold the exceedance probability is
        # above p; return the upper bound as a finite practical answer.
        return float(a_hi)

    for _ in range(max_iter):
        a_mid = 0.5 * (a_lo + a_hi)
        f_mid = f(a_mid)
        if f_mid > 0.0:
            a_lo = a_mid
            f_lo = f_mid
        else:
            a_hi = a_mid
            f_hi = f_mid
        if (a_hi - a_lo) < tol * max(1.0, a_hi):
            break
    return float(0.5 * (a_lo + a_hi))


def inverse_rice_multiband(
    p: float,
    bands: Sequence[tuple],
    T: float,
    bilateral: bool = True,
    clustering: str = "vanmarcke",
    a_max_sigma: float = 8.0,
    tol: float = 1e-9,
    max_iter: int = 80,
) -> float:
    """Inverse of `p_exceed_rice_multiband`: find a such that combined
    P_breach across the independent bands equals p.

    Same monotone bisection approach as `inverse_rice`. Upper bracket
    is taken as `a_max_sigma * max(sigma_i)` over the bands.
    """
    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    sigmas = [float(b[0]) for b in bands]
    if not sigmas:
        raise ValueError("bands must not be empty")
    sigma_max = max(sigmas)
    if sigma_max == 0.0 or T <= 0.0:
        return 0.0
    if p == 1.0:
        return 0.0
    if p == 0.0:
        return float(a_max_sigma * sigma_max)

    a_lo = 0.0
    a_hi = float(a_max_sigma * sigma_max)

    def f(a: float) -> float:
        d = p_exceed_rice_multiband(
            bands=bands, threshold=a, T=T,
            bilateral=bilateral, clustering=clustering,
        )
        return d["p_breach"] - p

    f_hi = f(a_hi)
    grow_iter = 0
    while f_hi > 0.0 and grow_iter < 6:
        a_hi *= 2.0
        f_hi = f(a_hi)
        grow_iter += 1
    if f_hi > 0.0:
        return float(a_hi)

    for _ in range(max_iter):
        a_mid = 0.5 * (a_lo + a_hi)
        f_mid = f(a_mid)
        if f_mid > 0.0:
            a_lo = a_mid
        else:
            a_hi = a_mid
        if (a_hi - a_lo) < tol * max(1.0, a_hi):
            break
    return float(0.5 * (a_lo + a_hi))


__all__ = [
    "RiceExceedanceResult",
    "spectral_moments",
    "zero_upcrossing_rate",
    "vanmarcke_bandwidth_q",
    "clh_epsilon",
    "p_exceed_rice",
    "p_exceed_rice_multiband",
    "p_exceed_from_psd",
    "inverse_rice",
    "inverse_rice_multiband",
]
