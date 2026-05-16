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
