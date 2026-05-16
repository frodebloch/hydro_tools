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
from typing import Tuple

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
    cap_residual: Tuple[float, float, float] | np.ndarray,
    window_s: float = DEFAULT_WINDOW_S,
    omega_lp_rad_s: float = DEFAULT_OMEGA_LP_RAD_S,
    tau_decorr_lf_s: float = 15.0,
    amber: float = DEFAULT_AMBER,
    red: float = DEFAULT_RED,
) -> RegimeBSeverity:
    """Estimate regime-B saturation severity from a delivered-thrust buffer.

    Parameters
    ----------
    tau_buffer : (N, 3) np.ndarray
        Recent history of delivered thrust (or equivalently FbTauSurge/
        Sway/Yaw) per DOF, in [N, N, N*m]. Samples are uniformly spaced
        at ``fs_hz``. Should be at least ``window_s`` long, but the
        function uses the last ``window_s`` regardless.
    fs_hz : float
        Sample rate of the buffer.
    cap_residual : (3,) sequence
        Per-DOF residual polytope cap = ``min(|tau_max+|, |tau_max-|)``
        of the surviving thruster set, in [N, N, N*m]. Compute via
        ``saturation_screening.compute_residual_polytope``.
    window_s : float
        Buffer window length to use for (mu, sigma) estimation.
    omega_lp_rad_s : float
        LF cutoff for the low-pass filter applied to the buffer before
        statistics are computed.
    tau_decorr_lf_s : float
        Heuristic LF decorrelation time, used only to report ``n_eff``.
        Does not affect the severity calculation.
    amber, red : float
        Severity thresholds for the IMCA traffic-light gating.

    Returns
    -------
    RegimeBSeverity
    """
    tau = np.asarray(tau_buffer, dtype=float)
    if tau.ndim != 2 or tau.shape[1] != 3:
        raise ValueError(
            f"tau_buffer must be (N, 3); got shape {tau.shape}"
        )
    cap = np.asarray(cap_residual, dtype=float)
    if cap.shape != (3,):
        raise ValueError(f"cap_residual must be (3,); got shape {cap.shape}")

    n_samples_window = int(round(window_s * fs_hz))
    if tau.shape[0] < n_samples_window:
        # Use what we have, but report; the caller can act on n_eff.
        n_samples_window = tau.shape[0]
    tau_w = tau[-n_samples_window:]

    tau_lf = lf_filter(tau_w, fs_hz=fs_hz, omega_lp_rad_s=omega_lp_rad_s)
    mu = tau_lf.mean(axis=0)
    sigma = tau_lf.std(axis=0)

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
