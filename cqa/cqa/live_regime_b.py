"""Live regime-B saturation severity estimator (sec.12.21.21.20).

Predicts post-WCF sustained-saturation severity from purely *observed*
quantities: a recent buffer of delivered controller thrust ``Tx/Ty/Tz``
(or equivalently the ``FbTauSurge/Sway/Yaw`` feedback channel) and the
brucon-derived residual thrust polytope of the surviving thruster set.

Rationale
---------
The forecast pipeline ``cqa.transient.wcfdi_transient`` computes the
analogous severity from a *weather model* (``Vw, Hs, Tp, Vc, theta_rel``)
plus a linearised closed-loop model. Both can be wrong: the weather
model has direction-dependent biases on oblique sea states (cqa-vs-brucon
gap of ~4x on mean sway demand for bf8_q10_w45 cells, sec.12.21.21.17),
and the linearised covariance underestimates ``sigma_tau_cmd`` by ~4x
because it uses intact-A_cl closed-loop covariance for what is really
the residual post-WCF system.

Both problems vanish if we read the **delivered thrust history** directly:
the controller has already done the closed-loop propagation for us, the
delivered thrust IS the env-load summary in force units, and post-WCF
the polytope cap is set by the surviving thruster geometry alone (no
weather model needed).

Why the bias estimator ``b_hat`` is the wrong source
----------------------------------------------------
The bias estimator's ``tau_b = 1000 s`` time constant pulls ``b_hat``
to ~91% of the true LF mean (analysis.md sec.12.21.13) and absorbs
**none** of the LF wave-drift fluctuation. Empirically pre-WCF on
bf8_q10_w45 seed 1027:

    Order_y mean = +573 kN, std = +79 kN
    b_hat_y mean = -488 kN, std =  +7 kN  (~9% of std_Order)

So ``b_hat`` carries ~85% of the mean signal but only ~1% of the
variance (in variance units; ~9% in std). Using ``b_hat`` would
under-predict ``sigma_tau_cmd`` by an order of magnitude.

Live estimator design (v1)
--------------------------
For each rolling step (or once per CQA evaluation):

    1. Read a buffer of the last ``window_s`` seconds of delivered
       thrust ``(Tx, Ty, Tz)`` from the brucon controller (or in the
       live cqa, from the observer feed of the same channel).
    2. Low-pass at ``omega_lp_rad_s`` (~ 0.30 rad/s by default,
       2-3x the typical wave period). The LF-filtered series isolates
       the slow-varying demand component the polytope cap actually
       governs; the WF tail above 0.3 rad/s is allocator-internal
       and not regime-B relevant.
    3. Compute ``(mu, sigma)`` of the LF series per DOF.
    4. For each DOF, compute the saturation probability
       ``P(|N(mu, sigma)| > cap_residual)`` analytically:
       ``P = Phi(-(cap-mu)/sigma) + Phi(-(cap+mu)/sigma)``
       which represents the instantaneous fraction of time the
       LF demand exceeds the residual polytope cap in steady state.
    5. Combine per-DOF probabilities into a scalar severity score
       (max across DOFs) and an IMCA traffic-light state.

Stationarity assumption
-----------------------
The 300 s default window is comfortably inside the conventional 20-min
sea-state stationarity assumption (DNV-ST-N001, IMCA M254). With
``tau_decorr_LF ~ 15 s`` (analysis.md heuristic), 300 s gives
``N_eff ~ 20`` and sigma sampling error of ~16%. Long enough to track
the LF wave-drift variance, short enough to be operator-current.

Future extensions (NOT in v1)
-----------------------------
  * Wind feed-forward subtraction: ``tau_LF -> tau_LF - F_wind(Vw_obs)``
    before estimating ``(mu, sigma)``, with Vw measured. Handles wind
    shifts faster than the natural buffer timescale.
  * Multi-window envelope: parallel short (60 s) and long (300 s)
    estimators with worst-case severity. Discussed and rejected for
    v1 -- the long-window sigma captures local-mean drift implicitly.
  * Closed-loop bandwidth inflation factor: deemed unnecessary
    (sec.12.21.21.19 -- post-WCF dynamics for sub-cap fluctuations are
    essentially the same as pre-WCF).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import norm

from .extreme_value import inverse_rice, zero_upcrossing_rate, vanmarcke_bandwidth_q
from .transient_obs import (
    AugmentedSystemObs,
    IDX_ETA,
)


# ----------------------------------------------------------------------
# Defaults (sec.12.21.21.19 / .20)
# ----------------------------------------------------------------------
DEFAULT_WINDOW_S = 300.0
DEFAULT_OMEGA_LP_RAD_S = 0.30
# IMCA M254-aligned thresholds (sec.12.21.21.20). Operator-facing
# semantics:
#   green : severity < 0.01   (P(sat) < 1% of steady-state time)
#   amber : 0.01 <= severity < 0.10
#   red   : severity >= 0.10
DEFAULT_AMBER = 0.01
DEFAULT_RED = 0.10


# ----------------------------------------------------------------------
# Yaw-priority operational cap (sec.12.21.21.24)
# ----------------------------------------------------------------------
# Empirical analysis of brucon bf8_q10_w45 post-WCF Order vs Alloc data
# (sec.12.21.21.24) shows that the *decoupled* per-DOF residual polytope
# value (e.g. max_sway = 1104 kN at the vertex where yaw = surge = 0) is
# unreachable in practice: when yaw is being demanded simultaneously, the
# allocator must consume part of the bow/stern force budget for yaw
# moment, leaving less for sway.
#
# At brucon's typical post-WCF operating point on bf8_q10_w45 the order
# channel asks for ~+866 kN sway WITH ~+19.5 MNm yaw simultaneously. The
# allocator delivers the yaw demand exactly (yaw-priority allocation,
# per brucon BasicAllocator) and the *remaining* sway capability is
# ~+656 kN mean / +864 kN peak. This is the operational cap.
#
# Closed-form derivation for a bow+stern-decomposed thruster set:
#   F_bow + F_stern = tau_y         (sway balance)
#   F_bow * arm_bow + F_stern * arm_stern = tau_z   (yaw balance)
# Given tau_z fixed (yaw demand met), the (F_bow, F_stern) pair lies on
# a 1D line and tau_y = F_bow + F_stern is bounded by the (F_bow, F_stern)
# box. Two candidate corners:
#   * F_bow saturated: F_stern = (tau_z - F_bow_max * arm_bow) / arm_stern
#   * F_stern saturated: F_bow = (tau_z - F_stern_max * arm_stern) / arm_bow
# Pick whichever yields larger tau_y (subject to the other bound staying
# valid). Symmetric construction gives the yaw cap conditional on sway.
#
# This neglects surge coupling (azimuth thrusters splitting capability
# between surge and sway). Deferred to v2.1 -- surge demand is typically
# small relative to the polytope's surge budget for the CSOV cases of
# interest, and the second-order effect is < 15% on the empirical match.


@dataclass(frozen=True)
class OperationalCapGeometry:
    """Bow/stern sway-force breakdown for the yaw-priority operational cap.

    Parameters
    ----------
    F_bow_max, F_bow_min : float
        Maximum positive / negative sway force the bow thruster group
        can produce when fully committed to sway [N]. Note: brucon's
        ``max_sway_bow`` and ``min_sway_bow`` directly.
    F_stern_max, F_stern_min : float
        Same for the stern thruster group [N].
    arm_bow : float
        Force-weighted x-position of the bow group [m] (positive forward).
    arm_stern : float
        Force-weighted x-position of the stern group [m] (negative
        for stern-mounted thrusters).

    Notes
    -----
    Construct via :func:`geometry_from_residual_polytope` from a brucon
    :class:`saturation_screening.MaxTau` instance.
    """

    F_bow_max: float
    F_bow_min: float
    F_stern_max: float
    F_stern_min: float
    arm_bow: float
    arm_stern: float


def sway_cap_given_yaw(
    tau_z: float, geom: OperationalCapGeometry
) -> Tuple[Optional[float], Optional[float]]:
    """Operational sway cap conditional on a yaw demand.

    Returns ``(tau_y_min, tau_y_max)`` -- the achievable sway range when
    the allocator delivers exactly ``tau_z`` of yaw moment using the
    bow + stern force decomposition. Returns ``(None, None)`` if no
    valid (F_bow, F_stern) pair satisfies the yaw demand within bounds
    (i.e. the yaw itself is infeasible).

    With ``tau_z`` fixed, (F_bow, F_stern) is constrained to a 1-D line.
    The feasible segment is the intersection with the box ``[F_bow_min,
    F_bow_max] x [F_stern_min, F_stern_max]``. The endpoints of that
    segment maximise/minimise ``tau_y = F_bow + F_stern``. We collect
    every box-intersection candidate and take the min/max at the end
    (the four corner labels do NOT correspond to min/max of tau_y in
    general, because the linear functional flips sign as you walk the
    box).
    """
    ab, asn = geom.arm_bow, geom.arm_stern
    Fb_max, Fb_min = geom.F_bow_max, geom.F_bow_min
    Fs_max, Fs_min = geom.F_stern_max, geom.F_stern_min

    EPS = 1e-9
    if abs(ab) < EPS or abs(asn) < EPS:
        return None, None

    cands = []
    # Corner: F_bow saturated -> F_stern determined by yaw balance.
    for Fb in (Fb_max, Fb_min):
        Fs = (tau_z - Fb * ab) / asn
        if Fs_min - 1e-6 <= Fs <= Fs_max + 1e-6:
            cands.append(Fb + Fs)
    # Corner: F_stern saturated -> F_bow determined by yaw balance.
    for Fs in (Fs_max, Fs_min):
        Fb = (tau_z - Fs * asn) / ab
        if Fb_min - 1e-6 <= Fb <= Fb_max + 1e-6:
            cands.append(Fb + Fs)

    if not cands:
        return None, None
    return min(cands), max(cands)


def yaw_cap_given_sway(
    tau_y: float, geom: OperationalCapGeometry
) -> Tuple[Optional[float], Optional[float]]:
    """Operational yaw cap conditional on a sway demand.

    Returns ``(tau_z_min, tau_z_max)``. Symmetric construction to
    :func:`sway_cap_given_yaw`: given the sway balance ``F_bow + F_stern
    = tau_y`` fixed, walk the (F_bow, F_stern) line to find the yaw
    extremes ``tau_z = F_bow * arm_bow + F_stern * arm_stern``.
    """
    ab, asn = geom.arm_bow, geom.arm_stern
    Fb_max, Fb_min = geom.F_bow_max, geom.F_bow_min
    Fs_max, Fs_min = geom.F_stern_max, geom.F_stern_min

    cands = []
    # Corner: F_bow saturated -> F_stern = tau_y - F_bow.
    for Fb in (Fb_max, Fb_min):
        Fs = tau_y - Fb
        if Fs_min - 1e-6 <= Fs <= Fs_max + 1e-6:
            cands.append(Fb * ab + Fs * asn)
    # Corner: F_stern saturated -> F_bow = tau_y - F_stern.
    for Fs in (Fs_max, Fs_min):
        Fb = tau_y - Fs
        if Fb_min - 1e-6 <= Fb <= Fb_max + 1e-6:
            cands.append(Fb * ab + Fs * asn)

    if not cands:
        return None, None
    return min(cands), max(cands)


def operational_cap_at(
    mu_tau_N_Nm: np.ndarray, geom: OperationalCapGeometry,
    surge_cap_N: float,
) -> np.ndarray:
    """Per-DOF symmetric operational cap conditional on mean demand.

    Given an LF mean demand ``mu_tau = (mu_surge, mu_sway, mu_yaw)``,
    return the symmetric operational cap ``cap = (c_surge, c_sway,
    c_yaw)`` where:
        c_sway = min(|tau_y_max(mu_yaw)|, |tau_y_min(mu_yaw)|)
        c_yaw  = min(|tau_z_max(mu_sway)|, |tau_z_min(mu_sway)|)
        c_surge = surge_cap_N (passed through; surge coupling deferred)

    If a DOF cap is infeasible (yaw or sway demand outside the polytope),
    returns ``0.0`` for that DOF -- the controller would already be
    failing to maintain station.
    """
    mu = np.asarray(mu_tau_N_Nm, dtype=float)
    mu_y, mu_z = mu[1], mu[2]
    ty_lo, ty_hi = sway_cap_given_yaw(mu_z, geom)
    tz_lo, tz_hi = yaw_cap_given_sway(mu_y, geom)
    if ty_lo is None or ty_hi is None:
        c_sway = 0.0
    else:
        c_sway = min(abs(ty_lo), abs(ty_hi))
    if tz_lo is None or tz_hi is None:
        c_yaw = 0.0
    else:
        c_yaw = min(abs(tz_lo), abs(tz_hi))
    return np.array([surge_cap_N, c_sway, c_yaw], dtype=float)


@dataclass
class RegimeBSeverity:
    """Result of a regime-B saturation severity evaluation.

    Attributes
    ----------
    mu : (3,) np.ndarray
        LF-filtered mean of delivered thrust per DOF [N, N, N*m].
    sigma : (3,) np.ndarray
        LF-filtered std of delivered thrust per DOF [N, N, N*m].
    cap_residual : (3,) np.ndarray
        Residual polytope cap per DOF [N, N, N*m].
    p_sat : (3,) np.ndarray
        Per-DOF probability of |tau_LF| exceeding cap_residual in the
        steady-state Gaussian approximation.
    severity : float
        Headline scalar = max(p_sat). Operator-facing.
    traffic : str
        One of ``"green"``, ``"amber"``, ``"red"`` per the IMCA M254
        scheme.
    n_eff : float
        Effective independent-sample count used to estimate sigma
        (window / tau_decorr_LF). Sigma sampling rel-error ~ 1/sqrt(2*n_eff).
    """

    mu: np.ndarray
    sigma: np.ndarray
    cap_residual: np.ndarray
    p_sat: np.ndarray
    severity: float
    traffic: str
    n_eff: float


def lf_filter(
    tau: np.ndarray,
    fs_hz: float,
    omega_lp_rad_s: float = DEFAULT_OMEGA_LP_RAD_S,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass of a (N, 3) thrust history.

    Parameters
    ----------
    tau : (N,) or (N, 3) np.ndarray
        Delivered thrust history; columns are DOFs (surge, sway, yaw).
    fs_hz : float
        Sample rate of the input.
    omega_lp_rad_s : float
        LF cutoff frequency in rad/s.
    order : int
        Butterworth order (default 4 -> 24 dB/oct rolloff).

    Returns
    -------
    np.ndarray
        Low-pass-filtered tau, same shape as input.
    """
    f_c = omega_lp_rad_s / (2.0 * np.pi)
    b, a = butter(order, f_c / (fs_hz / 2.0), btype="low")
    if tau.ndim == 1:
        return filtfilt(b, a, tau)
    out = np.empty_like(tau)
    for k in range(tau.shape[1]):
        out[:, k] = filtfilt(b, a, tau[:, k])
    return out


def saturation_probability_gaussian(
    mu: np.ndarray, sigma: np.ndarray, cap: np.ndarray
) -> np.ndarray:
    """Per-DOF P(|N(mu, sigma)| > cap) under the steady-state Gaussian
    assumption.

    The cap is symmetric (-cap, +cap). The two-sided probability is
    ``Phi(-(cap-mu)/sigma) + Phi(-(cap+mu)/sigma)``.

    All inputs broadcast.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cap = np.asarray(cap, dtype=float)
    sig = np.maximum(sigma, 1e-9)
    upper = norm.sf((cap - mu) / sig)   # P(X > +cap)
    lower = norm.cdf((-cap - mu) / sig) # P(X < -cap)
    return upper + lower


def _traffic(severity: float, amber: float, red: float) -> str:
    if severity >= red:
        return "red"
    if severity >= amber:
        return "amber"
    return "green"


def estimate_regime_b_severity(
    tau_buffer: np.ndarray,
    fs_hz: float,
    cap_residual: Tuple[float, float, float] | np.ndarray | None = None,
    window_s: float = DEFAULT_WINDOW_S,
    omega_lp_rad_s: float = DEFAULT_OMEGA_LP_RAD_S,
    tau_decorr_lf_s: float = 15.0,
    amber: float = DEFAULT_AMBER,
    red: float = DEFAULT_RED,
    geometry: Optional[OperationalCapGeometry] = None,
    surge_cap_N: Optional[float] = None,
) -> RegimeBSeverity:
    """Estimate regime-B saturation severity from a delivered-thrust buffer.

    Two cap modes are supported (mutually exclusive):

    1. **Legacy decoupled cap** (``cap_residual`` given, ``geometry`` not):
       Compare LF demand against a fixed per-DOF cap (vertex polytope
       value). Tends to under-report severity because the vertex cap is
       unreachable when multiple DOFs are demanded simultaneously.
       Kept for backward compatibility and for cases where geometry is
       unavailable.

    2. **Yaw-priority operational cap** (``geometry`` and ``surge_cap_N``
       given): the per-DOF cap is computed at each evaluation from the
       LF mean demand of the *other* DOFs (sec.12.21.21.24). This is
       the cap the brucon allocator actually experiences and produces
       severity estimates that match the empirical Alloc-vs-Order
       clipping behaviour on bf8_q10_w45 within ~10%.

    Parameters
    ----------
    tau_buffer : (N, 3) np.ndarray
        Recent history of delivered thrust (or equivalently FbTauSurge/
        Sway/Yaw) per DOF, in [N, N, N*m]. Samples are uniformly spaced
        at ``fs_hz``. Should be at least ``window_s`` long, but the
        function uses the last ``window_s`` regardless.
    fs_hz : float
        Sample rate of the buffer.
    cap_residual : (3,) sequence, optional
        Legacy decoupled per-DOF residual cap. Required if ``geometry``
        is not given.
    window_s : float
        Buffer window length to use for (mu, sigma) estimation.
    omega_lp_rad_s : float
        LF cutoff for the low-pass filter applied to the buffer before
        statistics are computed.
    tau_decorr_lf_s : float
        Heuristic LF decorrelation time, used only to report ``n_eff``.
    amber, red : float
        Severity thresholds for the IMCA traffic-light gating.
    geometry : OperationalCapGeometry, optional
        Bow/stern force-decomposition geometry. When given (with
        ``surge_cap_N``), enables yaw-priority operational cap.
    surge_cap_N : float, optional
        Decoupled surge cap [N]. Required when ``geometry`` is given.
        Surge coupling is deferred to v2.1.

    Returns
    -------
    RegimeBSeverity
    """
    tau = np.asarray(tau_buffer, dtype=float)
    if tau.ndim != 2 or tau.shape[1] != 3:
        raise ValueError(
            f"tau_buffer must be (N, 3); got shape {tau.shape}"
        )
    if geometry is not None and surge_cap_N is None:
        raise ValueError("surge_cap_N is required when geometry is given.")
    if geometry is None and cap_residual is None:
        raise ValueError("Provide either cap_residual or geometry+surge_cap_N.")

    n_samples_window = int(round(window_s * fs_hz))
    if tau.shape[0] < n_samples_window:
        n_samples_window = tau.shape[0]
    tau_w = tau[-n_samples_window:]

    tau_lf = lf_filter(tau_w, fs_hz=fs_hz, omega_lp_rad_s=omega_lp_rad_s)
    mu = tau_lf.mean(axis=0)
    sigma = tau_lf.std(axis=0)

    if geometry is not None:
        cap = operational_cap_at(mu, geometry, surge_cap_N=surge_cap_N)
    else:
        cap = np.asarray(cap_residual, dtype=float)
        if cap.shape != (3,):
            raise ValueError(f"cap_residual must be (3,); got shape {cap.shape}")

    p_sat = saturation_probability_gaussian(mu, sigma, cap)
    severity = float(p_sat.max())

    n_eff = float(window_s / max(tau_decorr_lf_s, 1e-9))

    return RegimeBSeverity(
        mu=mu,
        sigma=sigma,
        cap_residual=cap,
        p_sat=p_sat,
        severity=severity,
        traffic=_traffic(severity, amber, red),
        n_eff=n_eff,
    )


# ======================================================================
# Option 2: post-WCF excursion distribution machinery (sec.12.21.21.29)
# ======================================================================
#
# The estimator above returns a steady-state saturation *probability*: at
# any instant in the stationary post-WCF regime, what fraction of time is
# tau_LF outside the residual polytope cap? It is a static, scale-free
# severity score.
#
# Option 2 returns the *distribution of position excursion magnitudes*
# expected over a finite horizon (e.g. 200 s) post-WCF. Each saturation
# event injects a delta_tau pulse into the closed loop, the linear
# augmented system convolves these into a position-deviation history,
# and the maximum of |eta_post-WCF| over the horizon is a Rice-type
# extreme-value statistic of a Gaussian process.
#
# Pipeline (per DOF where saturation is active):
#   1. Take (mu, sigma) from estimate_regime_b_severity. The pre-WCF
#      delivered-thrust LF process is N(mu, sigma) (stationarity at the
#      sea-state level over the 300 s window).
#   2. After WCF the controller will still demand tau_demand ~ N(mu, sigma)
#      but only +-cap is delivered. The "lost" thrust is
#         delta_tau = max(0, X - cap) + min(0, X + cap)  for X ~ N(mu, sigma).
#      This is the spillover past the symmetric +-cap. The mean and
#      variance of delta_tau are given by truncated-normal formulas
#      (clipped_gaussian_moments).
#   3. delta_tau enters the linear augmented system as the same B_lost
#      forcing used by pulse_response. The resulting eta_post-WCF(t) is
#      a stationary stochastic process with PSD
#         S_eta(omega) = |H_{eta<-delta_tau}(omega)|^2 * S_dtau(omega).
#      The variance integrates to sigma_eta^2 (per DOF).
#   4. The peak |eta| over a horizon T is approximated by Rice's
#      Poisson-exceedance formula
#         P(max_T |eta| > r) ~ 1 - exp(-2 nu_0 T exp(-r^2 / (2 sigma_eta^2)))
#      with nu_0 the mean zero-up-crossing rate of eta.
#
# A deterministic mean offset E[eta] = H_DC * E[delta_tau] is added on
# top of the random fluctuation; the headline "P95 post-WCF excursion"
# is the 95th percentile of (E[eta] + max_T |eta_random|).
#
# Assumptions for v1:
# - Stationary post-WCF demand statistics (the controller's demand
#   process inherits the pre-WCF LF statistics). Valid for the 200 s
#   horizon of interest, much shorter than the sea-state stationarity
#   (20 min).
# - First-order Gauss-Markov approximation for the LF process spectrum.
#   For autocovariance R(tau) = sigma^2 exp(-|tau|/tau_corr), the one-
#   sided rad/s-native PSD (no /pi factor in the integral) is
#     S_X(omega) = (2 sigma^2 tau_corr / pi) / (1 + (omega tau_corr)^2)
#   so that int_0^inf S_X(omega) domega = sigma^2 (matches the cqa PSD
#   convention -- see cqa/closed_loop.py header). tau_corr ~ tau_decorr_lf_s.
# - The saturation nonlinearity reshapes S_dtau slightly (generates
#   higher harmonics) but for the first cut we approximate delta_tau as
#   sharing the LF spectral shape with the clipped variance. Validation
#   against direct MC in tests will quantify the error.
# - Independence across DOFs in the headline. The full radial xy
#   excursion E[R] = sqrt(E[eta_x^2] + E[eta_y^2]) is computed assuming
#   surge and sway are independent (cross-coupling deferred).


def clipped_gaussian_moments(
    mu: np.ndarray, sigma: np.ndarray, cap: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and std of the saturation "spillover"
    ``delta_tau = (X - cap) 1{X > cap} + (X + cap) 1{X < -cap}``
    for ``X ~ N(mu, sigma)`` and symmetric cap +-cap, per DOF.

    Derivation. Let ``f(x), F(x)`` be the standard normal PDF and CDF.
    With ``a = (cap - mu)/sigma``, ``b = (-cap - mu)/sigma`` (note
    ``b < a`` since cap > 0), the upper-tail spillover ``Y_+ = X - cap``
    on event ``{X > cap}`` has:

        P(X > cap)              = 1 - F(a)
        E[X 1{X > cap}]         = mu (1 - F(a)) + sigma f(a)
        E[X^2 1{X > cap}]       = (mu^2 + sigma^2)(1 - F(a))
                                  + sigma (mu + cap) f(a)       (see note)

    so
        E[Y_+] = E[X 1{>}] - cap P(X > cap)
               = (mu - cap)(1 - F(a)) + sigma f(a)
        E[Y_+^2] = E[(X - cap)^2 1{>}]
                 = (sigma^2 + (mu - cap)^2)(1 - F(a))
                   + sigma (mu + cap - 2 cap) f(a)
                 = (sigma^2 + (mu - cap)^2)(1 - F(a))
                   + sigma (mu - cap) f(a).

    By symmetry of the negative tail (replace ``cap -> -cap`` and use
    the lower-tail moments), the lower spillover ``Y_- = X + cap`` on
    ``{X < -cap}`` has:

        E[Y_-] = (mu + cap) F(b) - sigma f(b)
        E[Y_-^2] = (sigma^2 + (mu + cap)^2) F(b)
                   - sigma (mu + cap) f(b).

    The full spillover ``delta_tau = Y_+ + Y_-`` (disjoint events) has
    mean ``E[Y_+] + E[Y_-]`` and second moment ``E[Y_+^2] + E[Y_-^2]``;
    its variance is ``E[delta_tau^2] - E[delta_tau]^2``.

    Sanity limits (verified in tests):

    * ``cap -> infinity``: P(X > cap) -> 0 and f(a) -> 0 exponentially
      fast, so both moments vanish (no clipping, no spillover).
    * ``cap -> 0``: delta_tau == X, so ``E[delta_tau] == mu`` and
      ``Var[delta_tau] == sigma^2``.

    Parameters
    ----------
    mu, sigma : (3,) array-like
        Per-DOF mean and std of the LF demand process [N, N, N*m].
        ``sigma >= 0`` (zero allowed; degenerate point mass).
    cap : (3,) array-like
        Symmetric polytope cap per DOF [N, N, N*m]. ``cap >= 0``.

    Returns
    -------
    mu_dtau, sigma_dtau : (3,) np.ndarray
        Mean and std of the per-DOF saturation spillover.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cap = np.asarray(cap, dtype=float)
    if not (mu.shape == sigma.shape == cap.shape):
        raise ValueError(
            f"mu, sigma, cap must share shape; got {mu.shape}, "
            f"{sigma.shape}, {cap.shape}"
        )
    if np.any(sigma < 0):
        raise ValueError("sigma must be non-negative")
    if np.any(cap < 0):
        raise ValueError("cap must be non-negative")

    # Degenerate sigma -> point-mass response. delta_tau is the
    # spillover of a deterministic mu.
    sig = np.where(sigma > 0, sigma, 1.0)  # safe divisor
    a = (cap - mu) / sig
    b = (-cap - mu) / sig
    f_a = norm.pdf(a)
    f_b = norm.pdf(b)
    F_a = norm.cdf(a)
    F_b = norm.cdf(b)
    one_minus_Fa = 1.0 - F_a

    E_Yp = (mu - cap) * one_minus_Fa + sig * f_a
    E_Yp2 = (sig * sig + (mu - cap) ** 2) * one_minus_Fa + sig * (mu - cap) * f_a
    E_Ym = (mu + cap) * F_b - sig * f_b
    E_Ym2 = (sig * sig + (mu + cap) ** 2) * F_b - sig * (mu + cap) * f_b

    # Where sigma == 0 the formulas above are NaN / wrong because we
    # substituted sig=1. Patch directly: delta_tau is a deterministic
    # spillover of mu.
    deg = (sigma <= 0)
    if np.any(deg):
        mu_d = np.where(mu > cap, mu - cap, np.where(mu < -cap, mu + cap, 0.0))
        E_Yp = np.where(deg, np.where(mu > cap, mu - cap, 0.0), E_Yp)
        E_Yp2 = np.where(deg, np.where(mu > cap, (mu - cap) ** 2, 0.0), E_Yp2)
        E_Ym = np.where(deg, np.where(mu < -cap, mu + cap, 0.0), E_Ym)
        E_Ym2 = np.where(deg, np.where(mu < -cap, (mu + cap) ** 2, 0.0), E_Ym2)
        del mu_d  # unused; structure check only

    mu_dtau = E_Yp + E_Ym
    second_moment = E_Yp2 + E_Ym2
    var_dtau = np.maximum(second_moment - mu_dtau * mu_dtau, 0.0)
    sigma_dtau = np.sqrt(var_dtau)
    return mu_dtau, sigma_dtau


def gauss_markov_psd(
    sigma: float, tau_corr: float, omega: np.ndarray,
) -> np.ndarray:
    """First-order Gauss-Markov one-sided PSD, rad/s-native, no /pi factor.

    For autocovariance R(tau) = sigma^2 * exp(-|tau| / tau_corr), the
    one-sided PSD satisfying int_0^inf S(omega) domega = sigma^2 is

        S(omega) = (2 sigma^2 tau_corr / pi) / (1 + (omega tau_corr)^2).

    Derivation: the two-sided PSD in the Hz convention is
    S_2(f) = 4 sigma^2 tau / (1 + (2 pi f tau)^2); converting to one-
    sided rad/s with no /pi (the cqa convention, see closed_loop.py
    header lines 9-26) gives the formula above. Sanity:

        int_0^inf (2 sigma^2 tau / pi) / (1 + (omega tau)^2) domega
            = (2 sigma^2 tau / pi) * (1/tau) * (pi/2) = sigma^2.    OK.

    Parameters
    ----------
    sigma : float
        Process standard deviation [signal units]. sigma >= 0.
    tau_corr : float
        Decorrelation time [s]. tau_corr > 0.
    omega : (n,) np.ndarray
        Angular frequencies [rad/s], omega >= 0.

    Returns
    -------
    S : (n,) np.ndarray, units [signal^2 / (rad/s)].
    """
    sigma = float(sigma)
    tau_corr = float(tau_corr)
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if tau_corr <= 0.0:
        raise ValueError(f"tau_corr must be > 0, got {tau_corr}")
    omega = np.asarray(omega, dtype=float)
    if np.any(omega < 0):
        raise ValueError("omega must be non-negative")
    num = 2.0 * sigma * sigma * tau_corr / np.pi
    return num / (1.0 + (omega * tau_corr) ** 2)


def eta_frequency_response(
    aug: AugmentedSystemObs,
    omega: np.ndarray,
    input_dof: int,
    output_dof: int,
) -> np.ndarray:
    """Closed-loop frequency response H(jw) from B_lost input to the
    truth-eta output of the 21-state augmented system.

    Mathematically, for the LTI state-space (A, B_lost, C_eta) where
    C_eta selects ``eta[output_dof]`` from the 21-state vector,

        H(jw) = C_eta @ (jw I - A)^{-1} @ B_lost[:, input_dof].

    This is the steady-state amplitude/phase response of eta_{output_dof}
    to a unit-amplitude sinusoidal "lost-thrust" forcing on DOF
    ``input_dof``. Magnitude-squared |H|^2 maps an input PSD to an
    output PSD.

    Implementation: solve (jw I - A) x = b once per frequency (LU per
    omega; (21x21) is small so this is cheap). Returns complex (n,).

    DC limit (omega = 0): H(0) = -C_eta @ A^{-1} @ B_lost[:, input_dof]
    is the steady-state position gain of a step in delta_tau, used to
    compute the deterministic mean offset E[eta] = H(0) * E[delta_tau].

    Parameters
    ----------
    aug : AugmentedSystemObs
        21-state augmented system.
    omega : (n,) np.ndarray
        Angular frequencies [rad/s], non-negative.
    input_dof : int in {0, 1, 2}
        DOF index of the lost-thrust forcing column.
    output_dof : int in {0, 1, 2}
        DOF index of the eta output (0=surge, 1=sway, 2=yaw).

    Returns
    -------
    H : (n,) complex np.ndarray, units [m/N] (surge/sway) or [rad/N*m]
        (yaw).
    """
    if input_dof not in (0, 1, 2):
        raise ValueError(f"input_dof must be in {{0,1,2}}, got {input_dof}")
    if output_dof not in (0, 1, 2):
        raise ValueError(f"output_dof must be in {{0,1,2}}, got {output_dof}")
    A = aug.A
    n = A.shape[0]
    b = aug.B_lost[:, input_dof]
    eta_idx = IDX_ETA.start + output_dof
    omega = np.asarray(omega, dtype=float)
    H = np.zeros(omega.shape, dtype=complex)
    I = np.eye(n)
    for k, w in enumerate(omega):
        M = (1j * w) * I - A
        x = np.linalg.solve(M, b)
        H[k] = x[eta_idx]
    return H


def eta_psd_from_dtau(
    aug: AugmentedSystemObs,
    sigma_dtau: np.ndarray,
    tau_corr: float,
    omega: np.ndarray,
) -> np.ndarray:
    """Per-DOF closed-loop eta PSD driven by per-DOF independent
    Gauss-Markov delta_tau processes.

    Approximation: each DOF of delta_tau is modelled as a first-order
    Gauss-Markov process with std ``sigma_dtau[i]`` and decorrelation
    ``tau_corr`` (same for all DOFs). DOFs are assumed independent --
    no cross-spectral terms -- so the eta-axis PSD is the diagonal sum

        S_eta[i](omega) = sum_j |H_{eta_i <- dtau_j}(omega)|^2
                              * S_dtau[j](omega).

    Returns
    -------
    S_eta : (3, n_omega) np.ndarray, units [m^2 / (rad/s)] for surge/
        sway rows, [rad^2 / (rad/s)] for yaw.
    """
    sigma_dtau = np.asarray(sigma_dtau, dtype=float)
    if sigma_dtau.shape != (3,):
        raise ValueError(f"sigma_dtau must have shape (3,), got {sigma_dtau.shape}")
    omega = np.asarray(omega, dtype=float)
    S_eta = np.zeros((3, omega.size), dtype=float)
    for i in range(3):  # output dof
        for j in range(3):  # input dof
            if sigma_dtau[j] <= 0.0:
                continue
            H = eta_frequency_response(aug, omega, input_dof=j, output_dof=i)
            S_in = gauss_markov_psd(sigma_dtau[j], tau_corr, omega)
            S_eta[i] += (H.real ** 2 + H.imag ** 2) * S_in
    return S_eta


@dataclass(frozen=True)
class PostWcfExcursionDistribution:
    """Result bundle for `estimate_post_wcf_excursion_distribution`.

    Fields per DOF (surge, sway, yaw):

    mu_dtau     : (3,) saturation spillover mean       [N, N, N*m]
    sigma_dtau  : (3,) saturation spillover std        [N, N, N*m]
    mu_eta      : (3,) deterministic mean eta offset   [m, m, rad]
    sigma_eta   : (3,) random eta std                  [m, m, rad]
    nu_0_plus   : (3,) zero-up-crossing rate of eta    [Hz]
    q_vanmarcke : (3,) Vanmarcke bandwidth parameter   [-]
    eta_p50     : (3,) median running-max |eta - mu_eta| over horizon
    eta_p95     : (3,) 95th percentile running-max |eta - mu_eta|
    eta_p95_with_offset : (3,) |mu_eta| + eta_p95 (operator headline)
    eta_xy_p95_with_offset : float, sqrt(eta_x^2 + eta_y^2) headline for
        the (surge, sway) plane using the P95-with-offset per axis added
        in quadrature (conservative bound; ignores cross-coupling).
    """
    mu_dtau: np.ndarray
    sigma_dtau: np.ndarray
    mu_eta: np.ndarray
    sigma_eta: np.ndarray
    nu_0_plus: np.ndarray
    q_vanmarcke: np.ndarray
    eta_p50: np.ndarray
    eta_p95: np.ndarray
    eta_p95_with_offset: np.ndarray
    eta_xy_p95_with_offset: float


def estimate_post_wcf_excursion_distribution(
    aug: AugmentedSystemObs,
    mu: np.ndarray,
    sigma: np.ndarray,
    cap_residual: np.ndarray,
    tau_decorr_lf_s: float = 15.0,
    t_horizon_s: float = 200.0,
    omega_max_rad_s: float = 3.0,
    n_omega: int = 512,
    clustering: str = "vanmarcke",
) -> PostWcfExcursionDistribution:
    """End-to-end post-WCF excursion-distribution headline.

    Inputs are exactly the steady-state outputs of
    `estimate_regime_b_severity` (mu, sigma of the LF demand process)
    plus the residual polytope cap from the surviving-thruster set and
    the augmented closed-loop system aug.

    Pipeline (sec.12.21.21.29 in analysis.md):

      1. clipped_gaussian_moments -> (mu_dtau, sigma_dtau).
      2. gauss_markov_psd at sigma_dtau, tau_decorr_lf_s -> S_dtau.
      3. eta_frequency_response on aug -> H(omega) per (in, out).
      4. S_eta = sum_j |H_ij|^2 S_dtau[j].
      5. spectral moments -> sigma_eta, nu_0+, q_vanmarcke.
      6. inverse_rice (Vanmarcke clustering by default) at p=0.5, 0.05
         -> p50, p95 running-max of |eta - mu_eta|.
      7. Add deterministic mean E[eta] = H(0) * mu_dtau on top.
      8. Radial xy: |mu_eta_xy| + p95_xy (per-axis quadrature, conservative).

    Parameters
    ----------
    aug : 21-state augmented system.
    mu, sigma : (3,) per-DOF mean and std of the LF demand process
        [N, N, N*m].
    cap_residual : (3,) symmetric polytope cap [N, N, N*m].
    tau_decorr_lf_s : LF decorrelation time [s], default 15 (cqa
        heuristic, sec.12.21.13).
    t_horizon_s : extreme-value horizon [s], default 200 (post-WCF
        operator window).
    omega_max_rad_s : upper integration frequency [rad/s]. The Gauss-
        Markov PSD is monotone decreasing in omega; default 3.0 captures
        > 99% of the variance for tau_corr ~ 15 s.
    n_omega : number of frequency samples (linear grid from 0 to
        omega_max_rad_s). 512 is the sweet spot for the (21x21) solve.
    clustering : "vanmarcke" or "poisson", passed to inverse_rice.

    Returns
    -------
    PostWcfExcursionDistribution
    """
    mu_dtau, sigma_dtau = clipped_gaussian_moments(mu, sigma, cap_residual)
    omega = np.linspace(0.0, omega_max_rad_s, n_omega)

    # Deterministic mean offset eta = -A^{-1} B_lost mu_dtau, projected.
    Ainv_B_mu = np.linalg.solve(aug.A, aug.B_lost @ mu_dtau)
    # x_ss solves A x = -B_lost * mu_dtau (since A x + B_lost mu = 0 in ss)
    mu_eta = -Ainv_B_mu[IDX_ETA]

    # Per-DOF eta PSD.
    S_eta = eta_psd_from_dtau(aug, sigma_dtau, tau_decorr_lf_s, omega)

    sigma_eta = np.zeros(3)
    nu0 = np.zeros(3)
    q_vm = np.zeros(3)
    p50 = np.zeros(3)
    p95 = np.zeros(3)
    for i in range(3):
        S_i = S_eta[i]
        var_i = float(np.trapezoid(S_i, omega))
        sigma_eta[i] = np.sqrt(max(var_i, 0.0))
        if sigma_eta[i] <= 0.0:
            continue
        nu0[i] = zero_upcrossing_rate(S_i, omega)
        q_vm[i] = vanmarcke_bandwidth_q(S_i, omega)
        if nu0[i] <= 0.0:
            continue
        p50[i] = inverse_rice(
            p=0.5, sigma=sigma_eta[i], nu_0_plus=nu0[i],
            T=t_horizon_s, bilateral=True, clustering=clustering, q=q_vm[i],
        )
        p95[i] = inverse_rice(
            p=0.05, sigma=sigma_eta[i], nu_0_plus=nu0[i],
            T=t_horizon_s, bilateral=True, clustering=clustering, q=q_vm[i],
        )

    eta_p95_with_offset = np.abs(mu_eta) + p95
    eta_xy_p95_with_offset = float(np.hypot(
        eta_p95_with_offset[0], eta_p95_with_offset[1]
    ))
    return PostWcfExcursionDistribution(
        mu_dtau=mu_dtau,
        sigma_dtau=sigma_dtau,
        mu_eta=mu_eta,
        sigma_eta=sigma_eta,
        nu_0_plus=nu0,
        q_vanmarcke=q_vm,
        eta_p50=p50,
        eta_p95=p95,
        eta_p95_with_offset=eta_p95_with_offset,
        eta_xy_p95_with_offset=eta_xy_p95_with_offset,
    )
