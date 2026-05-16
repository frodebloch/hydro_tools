"""Operator-facing summary for the LIVE WCFDI decision cell.

Companion to ``cqa.operator_view`` but driven from the LIVE pipeline
(``cqa.live_decision.evaluate_decision_cell_live``) instead of the
forecast/MC pipeline.

The operator panel renders two side-by-side bars on the same metric
axis (radial position deviation, in metres) so a non-engineer can
compare:

  RIGHT NOW (intact)    : where is the boat sitting against setpoint
                          right now?  P50 / P95 of |R(t)| from the
                          live observer offset and Bayesian sigma
                          posteriors.

  IF WCF NOW (post-fault): how big does the excursion get if a
                           worst-case thrust failure occurs from the
                           current state?  P50 / P95 of the post-WCF
                           peak of |eta_hat + delta_eta_mean(t)|, plus
                           the time-to-peak.

Both axes use the IMCA M254 Rev.1 Figure 8 thresholds (2 m amber,
4 m red, configurable via ``OperationalLimits``) and the same
green/amber/red traffic light rule. Operator reads are uniform
across both states.

Distribution model
------------------
The radial position is modelled as ``R = |eta_offset + nu|`` with
``eta_offset`` deterministic (live LF position for intact, live LF +
mean transient at each t for WCF) and ``nu`` zero-mean bivariate
Gaussian with per-axis std (sigma_x, sigma_y).

The per-axis std differs between the two axes:

  * **Intact** halo: LF-channel posteriors only. The intact bar shows
    the spread of the LF position estimate around its mean. WF noise
    lives on the wave-frequency observer channel ``eta_wave``, NOT
    on ``eta_hat_LF``; folding it in here would inflate the intact
    halo by ~2x at typical Bf6+ states (verified on pwq30: 60-s pre-
    WCF window of brucon SurgeDev/SwayDev has sigma ~0.4 m, while
    LF + WF + b_hat-radial in quadrature gives ~0.6 m). The b_hat-
    realisation halo is also a property of the post-WCF transient
    (spread of delta_eta_mean(t) due to b_hat snapshot uncertainty),
    not the steady-state intact position.

  * **WCF** halo: LF + WF + b_hat-radial in quadrature. WF and b_hat
    legitimately spread the post-WCF peak: the WF channel adds wave-
    band oscillation that is in phase / out of phase with the
    transient peak depending on realisation, and b_hat carries the
    snapshot uncertainty in the assumed tau_env that drives the
    transient. This matches the sigma envelope used inside
    ``evaluate_decision_cell_live.wcfdi_pos_peak_m``.

This is the **offset Hoyt distribution** (offset Rice when sigma_x ==
sigma_y, which it nearly is for the CSOV at most directions).
Quantiles are computed by a small per-call Monte Carlo (n_mc samples
in each axis, default 2000) -- cheap enough at 1 Hz refresh and
avoids the special-function machinery a closed-form Rice CDF would
need.

Time-to-peak
------------
For the WCF axis we report the time at which the deterministic
``|eta_hat + delta_eta_mean(t)|`` reaches its maximum. This is the
"how much time do I have" cue an operator naturally wants.

Validation
----------
See ``scripts/p7_brucon_validation/validate_live_operator_panel.py``
and the 12-cell roll-up
``scripts/p7_brucon_validation/roll_up_live_operator_panel.py``.

Intact axis (12 cells, 30 brucon seeds each, Bf4 -> Bf8 +/- 10 deg
spread, head + 45 deg windsea + the two propeller-walk cells): the
panel reports the actual offset+noise distance from the DP setpoint
(|eta_hat_LF + nu|), and is compared against the un-demeaned brucon
LF radius hypot(SurgeDev, SwayDev). Resulting bias on the P95 is
small and slightly conservative across the matrix (~+0 to +20 %),
with single-snapshot coverage 50-70 % -- the latter is genuine
sampling variability of single-realisation P95 vs a calibrated
halo, not over-conservatism.

WCF axis: the previous per-instant Gaussian halo at the
deterministic peak under-predicted the per-seed brucon
single-realisation P95 by 11-59 % across the matrix (worst on
energetic cells where the LF channel has the largest natural
slow drift). Diagnosis (scripts/p7_brucon_validation/
diagnose_lf_transient_shape.py, analysis.md sec.12.21.9):

  * cqa pred ensemble-mean Δη(t) on the b̂-loaded DOF matches
    truth ensemble-mean to within ~30 % on head-on cells (e.g.
    bf6_h0 surge -0.39 m truth vs -0.44 m pred at t≈32 s).
    The deterministic transient on the loaded DOF is correct.
  * The per-seed P95 gap is dominated by NATURAL LF DRIFT in
    the brucon LF channel: a stationary correlated process with
    sigma_R_LF ~ 0.3-0.9 m and tau_decorr ~15 s wandering over
    the 60 s post-WCF window. This adds ~a_q(N_eff_LF) *
    sigma_R_LF to the per-seed window-max, where the Gumbel/
    Rice peak factor a_95 ~ 3.5 at N_eff = 4. The previous
    per-instant Gaussian halo gave only ~1.96 * sigma -- about
    half what the window-max statistic actually produces.
  * Cross-coupled DOFs (sway/yaw under head-on b̂) DO show a
    real model gap: cqa pred ensemble-mean is essentially zero
    while truth has +0.5 m sway / +0.04 rad yaw transient.
    Tracked separately as a candidate WCFDI asymmetry-physics
    investigation; not addressed here. The natural-drift
    fix below partially absorbs it through the LF Gumbel
    contribution (which is direction-isotropic in 2D and so
    enters every radial direction equally).

Fix in this version: the WCF P95 / P50 are computed by
``_radial_window_max_quantiles``, which:

  * adds an LF Gumbel contribution with a_50 / sqrt(pi/2)
    scaling on a 2D Gaussian sample (preserves random
    direction; magnitude calibrated to give the right MEDIAN
    over the window), with N_eff_LF = t_horizon / 15 s;
  * adds a WF Gumbel contribution the same way with N_eff_WF
    = t_horizon / 5 s;
  * adds the b̂ snapshot uncertainty as a per-instant 2D
    Gaussian halo (deterministic-mean uncertainty -- no
    correlation knob to exploit);
  * computes the radial quantile of the sum by MC.

12-cell roll-up of the WCF P95 bias before / after this fix:

  cell             before      after     coverage_after
  bf4_c1_h0          +26 %      -16 %        47 %
  bf4_c1_q10         +12 %      -19 %        40 %
  bf6_h0             -23 %       -8 %        87 %
  bf6_q10            -23 %      -10 %        87 %
  bf6_h0_w45         -17 %      -18 %        87 %
  bf6_q10_w45        -13 %      -21 %        67 %
  bf8_h0             -32 %       +1 %        90 %
  bf8_q10            -40 %      -22 %        73 %
  bf8_h0_w45         -41 %      -28 %        60 %
  bf8_q10_w45        -35 %      -15 %        83 %
  pwo                -59 %      -25 %        63 %
  pwq30              -42 %       -9 %        87 %

Big wins on the energetic / cross-coupled cells (pwo -34 pp,
pwq30 -33 pp, bf8_h0 -33 pp, bf8_q10_w45 -20 pp). Small
regressions on bf4 calm cells (~30 pp) where the previous
slight over-prediction had been useful slack -- absolute
magnitudes there are ~0.1 m, well below operator-relevant
scales. P50 is now consistently -20..-30 %; the operationally
important quantile is P95.

Residual P95 gap on bf6/bf8 (-8..-28 %) is the cross-coupled
DOF model gap noted above: even with the right LF Gumbel
envelope, the deterministic mean trajectory under-predicts
sway/yaw transients on cells where the WCFDI thrust loss is
asymmetric in physical thruster geometry. Resolving that
requires extending WcfdiScenario to carry an asymmetric loss
vector matched to the brucon thruster bus assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import CqaConfig
from .gangway import (
    GangwayJointState,
    telescope_sensitivity,
    telescope_sensitivity_6dof,
)
from .live_decision import (
    LiveObserverState,
    LiveSigmaPosterior,
    _build_aug_for_live,
)
from .transient_obs import (
    pulse_response,
    pulse_response_with_lift_coupling,
    N_STATE,
    IDX_TAU_THR,
)
from .transient import WcfdiScenario
from .decision_matrix import _imca_traffic, _worst
from .live_regime_b import (
    estimate_regime_b_severity,
    RegimeBSeverity,
    OperationalCapGeometry,
    DEFAULT_AMBER as _REGB_AMBER,
    DEFAULT_RED as _REGB_RED,
)


# ---------------------------------------------------------------------------
# Summary dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveOperatorSummary:
    """Operator-facing two-bar summary of the LIVE decision cell.

    All radial values in metres, body-frame radial distance from
    setpoint. Probabilities and quantiles are conditional on the
    live observer state and Bayesian sigma posteriors at the call
    instant. The WCF-axis values are additionally conditional on
    "a WCF event happening right now from this state".

    Intact axis (right now)
    -----------------------
    intact_R_p50, intact_R_p95 : float, m
        Median and 95th percentile of the live radial deviation
        ``|eta_hat_LF + nu|`` under the joint LF + WF + b_hat noise
        posterior.
    intact_traffic : "green" / "amber" / "red"
        IMCA M254 Fig.8 colour driven by ``intact_R_p95`` vs the
        warning / alarm radii.

    WCF axis (if WCF now)
    ---------------------
    wcf_R_p50, wcf_R_p95 : float, m
        Median and 95th percentile of the post-WCF peak radial
        deviation. The peak is taken over ``[0, t_horizon]`` with
        ``t_horizon`` set by the caller (default 60 s).
    wcf_t_peak_s : float, s
        Time after the (notional) WCF instant at which the
        deterministic mean trajectory ``|eta_hat + delta_eta_mean(t)|``
        peaks. Useful operator cue for time-to-react.
    wcf_traffic : "green" / "amber" / "red"
        IMCA colour driven by ``wcf_R_p95``.

    Combined
    --------
    overall_traffic : worst-of(intact_traffic, wcf_traffic).
    pos_warning_radius_m, pos_alarm_radius_m : the IMCA thresholds
        used (defaults from cfg.operational_limits).

    Diagnostics
    -----------
    sigma_R_intact_m : float, m
        Quadrature sum of the per-axis LF sigmas (the noise scale on
        the intact halo).
    sigma_R_wcf_m : float, m
        Quadrature sum of LF + WF + b_hat-radial per-axis sigmas (the
        noise scale on the WCF halo). Matches the sigma envelope used
        by ``evaluate_decision_cell_live.wcfdi_pos_peak_m``.
    intact_R_offset_m : float, m
        ``|eta_hat_LF|`` at the call instant (the deterministic part
        of the intact axis).
    wcf_R_offset_at_peak_m : float, m
        ``|eta_hat_LF + delta_eta_mean(t_peak)|`` (the deterministic
        part of the WCF axis at its peak).
    """

    # Intact axis
    intact_R_p50: float
    intact_R_p95: float
    intact_R_offset_m: float
    intact_traffic: str

    # WCF axis
    wcf_R_p50: float
    wcf_R_p95: float
    wcf_R_offset_at_peak_m: float
    wcf_t_peak_s: float
    wcf_traffic: str

    # Shared
    pos_warning_radius_m: float
    pos_alarm_radius_m: float
    sigma_R_intact_m: float
    sigma_R_wcf_m: float
    overall_traffic: str

    # ----- Gangway telescope axis (optional, present iff a joint is given) -----
    # The telescope bar reports |dL| -- the magnitude of the
    # telescope-length deviation required to keep the tip on the
    # world-fixed landing point as the vessel moves. Mirrors the
    # position bars (single non-negative axis with a single worst-
    # margin threshold). Operationally simpler and matches the
    # ``evaluate_decision_cell_live`` |c_L @ eta| envelope.
    #
    # The threshold (``gangway_stroke_m``) is the worst end-stop
    # margin ``min(L_max - L0, L0 - L_min)``, with IMCA-style
    # 60 % amber / 80 % red of stroke.
    #
    # An earlier prototype used SIGNED dL with separate extend/
    # retract thresholds; brucon validation showed the signed
    # response is bimodal under WCFDI (extend phase early, retract
    # phase late) and a single-instant Gaussian halo at the
    # deterministic dL peak under-predicts the magnitude P95 by
    # 40-50 % in bf6/bf8 because it cannot reach the opposite-sign
    # tail. Switching to |dL| recovers the unimodal-positive
    # statistics the panel's halo model assumes.
    #
    # 12-cell brucon roll-up of the |dL| bar (forward gangway,
    # h=15 m, L0=25 m; see scripts/p7_brucon_validation/roll_up_
    # gangway_bar.py). Bias evolution as comparator + posterior
    # plumbing matured (P95 bias / coverage; cells grouped):
    #
    #               horiz_3dof   6dof+folded   6dof+excursion   6dof+LF+Gumbel
    #               (initial)    (asym fix)    (subtract bug)   (window-max)
    #   bf4         P95 -31..-32%  -6..-17%      +3..-11%        +12..+26%  90..97%
    #   bf6         P95 -46..-53% -15..-23%     -12..-19%        -13..-23%  60..80%
    #   bf8         P95 -45..-51% -19..-32%     -17..-29%        -32..-41%  47..60%
    #   pwo         P95 -78%      -57%          -52%             -59%       17%
    #   pwq30       P95 -61%      -38%          -34%             -42%       57%
    #
    # The 6dof+LF+Gumbel column uses the new window-max model:
    # pred_pq = |c.delta_eta_mean(t_peak)| + a_q(N_eff) *
    # sigma_dL_wf_measured, with N_eff = tau_LF / T_zc_dL_wf_measured
    # and a_q from the Gumbel/Rice extreme-value formula. tau_LF is
    # the duration the |LF transient| stays above 0.8*peak. The
    # WF-only sigma comes from a windowed-demean of the gangway
    # WF channel (synthesized in brucon validation as c6 .
    # eta_wf_full over the same pre-WCF window the per-DOF
    # posteriors use). All quantities are observer-state +
    # posterior driven; no Tp / Hs / sea-state lookup.
    #
    # P50 improved on every cell (e.g. bf6_h0 -52% -> -42%, bf8_h0
    # -47% -> -41%). P95 got somewhat worse on bf8/pwo because the
    # window-max model removes the spurious noise width that the
    # folded-normal halo had been using to mask the LF transient
    # under-prediction. The residual on bf6/bf8/pwo is now cleanly
    # attributable to two distinct causes (per the position-bar
    # diagnosis in analysis.md sec.12.21.9 and
    # diagnose_lf_transient_shape.py):
    #
    #   (i)  natural slow LF drift in the brucon LF channel
    #        (sigma_R_LF ~ 0.3-0.9 m, tau_decorr ~15 s) which adds
    #        a Gumbel/Rice window-max contribution the position bar
    #        now models but the gangway bar's c3 . eta_LF projection
    #        does NOT yet model on top of |c3 . delta_eta_mean(t_peak)|;
    #
    #   (ii) the same cross-coupling model gap the position bar has --
    #        cqa pred ensemble-mean is essentially zero on the
    #        un-loaded DOFs (sway / yaw under head-on b̂) while truth
    #        has substantial transients. WCFDI thrust loss is
    #        asymmetric in physical thruster geometry but cqa models
    #        it as symmetric per-DOF alpha-cap reduction.
    #
    # Adding an LF Gumbel contribution to the gangway excursion would
    # mirror the position-bar fix; deferred until per-DOF brucon LF
    # sigmas can be projected through c6 the same way the WF posteriors
    # already are. The gangway comparator itself is now apples-to-apples.
    #
    # Note: when sigma_dL_wf_measured / T_zc_dL_wf_measured are not
    # supplied on LiveSigmaPosterior, the panel falls back to the
    # previous folded-normal halo at the deterministic peak (per-
    # instant statistic). This is structurally low for the window-
    # max truth comparator, but matches the panel's previous
    # behaviour when the upstream estimator has no gangway-channel
    # measurement to feed in.
    #
    # WF projection coverage:
    #   "horizontal_3dof" : only surge/sway/yaw WF posteriors used
    #                       (roll/pitch/heave not provided -- the
    #                       displayed sigma_dL is a LOWER BOUND).
    #   "full_6dof"       : all six WF DOFs included.
    # The position bars are unaffected by this choice.
    gangway_present: bool = False
    gangway_dL_p50: float = 0.0     # P50 of |dL|, m
    gangway_dL_p95: float = 0.0     # P95 of |dL|, m
    gangway_dL_intact_offset: float = 0.0  # |dL| live deterministic offset, m
    gangway_dL_wcf_offset_at_peak: float = 0.0  # |dL| at deterministic WCF peak, m
    # Excursion-only quantiles: folded-normal of the WCFDI-induced *change*
    # in dL relative to the live LF baseline, i.e. |c . delta_eta_mean(t_peak)
    # + WF_noise + b_hat_noise| with the eta_hat_lf baseline removed from
    # the deterministic mean. These are the apples-to-apples companions to a
    # truth statistic of max|dL_truth(t) - dL_pre|. The operator-facing
    # gangway_dL_p50/p95 above (which include the live LF baseline) remain
    # the right thing for "will the telescope hit the end-stop"; the
    # excursion fields are what the brucon validation comparator must use.
    gangway_dL_excursion_p50: float = 0.0
    gangway_dL_excursion_p95: float = 0.0
    gangway_stroke_m: float = 0.0   # min(L_max - L0, L0 - L_min), m
    gangway_sigma_dL_intact_m: float = 0.0
    gangway_sigma_dL_wcf_m: float = 0.0
    gangway_t_peak_s: float = 0.0
    gangway_traffic: str = "green"
    gangway_wf_coverage: str = "horizontal_3dof"

    # ----- Regime-B saturation severity (optional) -----
    # Present iff the caller supplied both ``obs_state.tau_buffer`` /
    # ``tau_buffer_fs_hz`` and the ``cap_residual_N_Nm`` argument to
    # ``summarise_for_operator_live``. Estimates the post-WCF
    # sustained-saturation severity from the rolling buffer of
    # delivered thrust against the residual polytope of the surviving
    # thruster set. See cqa.live_regime_b and analysis.md
    # sec.12.21.21.16-21.
    #
    # severity = max(p_sat) across DOFs, where p_sat[i] = P(|tau_LF_i|
    # > cap_residual_i) in the steady-state Gaussian approximation.
    # Traffic light: green if severity < 0.01, amber [0.01, 0.10),
    # red >= 0.10. The headline ``overall_traffic`` is the worst-of
    # intact / wcf / gangway / regime-B.
    regime_b_present: bool = False
    regime_b_severity: float = 0.0
    regime_b_p_sat: Optional[np.ndarray] = None        # (3,)
    regime_b_mu_N_Nm: Optional[np.ndarray] = None      # (3,)
    regime_b_sigma_N_Nm: Optional[np.ndarray] = None   # (3,)
    regime_b_cap_residual_N_Nm: Optional[np.ndarray] = None  # (3,)
    regime_b_traffic: str = "green"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Decorrelation timescales for the WCF window-max envelope
# ---------------------------------------------------------------------------
#
# These are the same heuristic decorrelation times used in the live
# brucon validation harness (scripts/p7_brucon_validation/
# live_cell_per_seed_pwq30.py) for the BayesianSigmaEstimator
# windows. Treating them here as module-level constants documents
# them in one place; the WCF window-max envelope uses them to size
# the Gumbel/Rice peak factor a_q(N_eff) over the WCF horizon.
#
#   T_DECORR_LF (surge / sway): ~ 1 / omega_pid ~ 12-17 s on the
#     CSOV Medium tuning -> 15 s.
#   T_DECORR_WF (all axes): ~ Tp / 2 ~ 5 s for typical Bf6+ states.
#
# Yaw is not used in the radial-position WCF envelope (the bar is
# 2D xy magnitude); the gangway bar uses its own tau_LF measured
# from the LF peak shape and T_zc_dL_wf_measured for N_eff and so
# is independent of these constants.
T_DECORR_LF_S = 15.0
T_DECORR_WF_S = 5.0


def _gumbel_peak_factor(N_eff: float, q: float) -> float:
    """Gumbel/Rice extreme-value peak factor at quantile ``q``.

    Returns ``a_q`` such that for a stationary zero-mean Gaussian
    with ``N_eff`` independent crests in the observation window,
    the q-quantile of ``max_t |x(t)|`` is approximately
    ``a_q * sigma``. This is the same formula used by the gangway
    bar (Gumbel/Rice extreme-value model); centralised here so the
    position-bar window-max envelope and the gangway window-max
    excursion share one implementation.

    Floors ``N_eff`` at 1.001 to keep ``ln`` defined for the
    "single crest" limit (a_50 -> 0.95, a_95 -> 4.4 at N=1.001
    -- which is conservative; the q=0.95 quantile of |x| with one
    crest is ~1.96 sigma, not 4.4 sigma, so callers that want
    "fall back to per-instant Gaussian" should detect N_eff <= 1
    explicitly and use the folded-normal/MC path instead).

    Conventions match analysis.md sec.12.6:
      a_50 = sqrt(2 ln N) - euler_gamma / sqrt(2 ln N)
      a_95 = sqrt(2 ln N) - ln(-ln 0.95) / sqrt(2 ln N)
    where euler_gamma ~= 0.5772156649 (Euler-Mascheroni constant).
    """
    N = max(float(N_eff), 1.001)
    ln_term = float(np.sqrt(2.0 * np.log(N)))
    if q == 0.50:
        return ln_term - 0.5772156649 / ln_term
    if q == 0.95:
        return ln_term - float(np.log(-np.log(0.95))) / ln_term
    # General q: Gumbel CDF F(a) = exp(-exp(-(a*ln_term - ln N)))
    # Inverting: a = ln_term - ln(-ln q) / ln_term.
    return ln_term - float(np.log(-np.log(q))) / ln_term


def _radial_quantiles(
    offset_xy: np.ndarray,
    sigma_x: float,
    sigma_y: float,
    n_mc: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Return (P50, P95) of ``R = |offset_xy + nu|`` with
    ``nu ~ N(0, diag(sigma_x^2, sigma_y^2))``.

    Pure function. Per-call MC; no state. ``offset_xy`` is a (2,)
    deterministic body-frame mean offset.

    This is a per-instant statistic. For window-max statistics over
    a finite horizon (e.g. the WCF post-fault peak), use
    ``_radial_window_max_quantiles`` instead, which combines the
    per-instant 2D Gaussian halo on b_hat uncertainty with scalar
    Gumbel/Rice contributions for the LF and WF correlated noise
    processes.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    nu_x = rng.standard_normal(n_mc) * sigma_x
    nu_y = rng.standard_normal(n_mc) * sigma_y
    R = np.hypot(offset_xy[0] + nu_x, offset_xy[1] + nu_y)
    return float(np.quantile(R, 0.50)), float(np.quantile(R, 0.95))


def _radial_window_max_quantiles(
    offset_xy_at_peak: np.ndarray,
    sigma_lf_x: float,
    sigma_lf_y: float,
    sigma_wf_x: float,
    sigma_wf_y: float,
    sigma_b_hat_axis: float,
    *,
    t_horizon_s: float,
    T_decorr_lf_s: float = T_DECORR_LF_S,
    T_decorr_wf_s: float = T_DECORR_WF_S,
    n_mc: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """WCF post-fault P50, P95 of ``max_{t in [0, t_horizon]} |R(t)|``.

    Three independent noise contributions, each modelled with the
    correlation structure that fits its physics:

      1. LF residual: stationary correlated Gaussian process with
         per-axis sigma (``sigma_lf_x``, ``sigma_lf_y``) and
         decorrelation time ``T_decorr_lf_s``. Over the WCF horizon
         it sees ``N_eff_lf = t_horizon / T_decorr_lf`` independent
         crests; the q-quantile of ``max_t |x_lf|`` is
         ``a_q(N_eff_lf) * sigma_R_lf`` in **magnitude** with
         **random direction in 2D** (independent of the radial axis
         of ``offset_xy_at_peak``). Modelled here as an isotropic
         2D vector with Rayleigh-magnitude scaling.

      2. WF residual: same as LF but with ``T_decorr_wf_s`` and
         per-axis (``sigma_wf_x``, ``sigma_wf_y``). Independent of
         the LF residual.

      3. b_hat snapshot uncertainty: a deterministic-mean
         uncertainty (no time-correlation knob), modelled per-instant
         as 2D Gaussian halo with per-axis sigma
         ``sigma_b_hat_axis``.

    All three vectors add in xy:
        nu_total = nu_lf + nu_wf + nu_bhat
    and the WCF radial deviation is ``R = |offset_xy_at_peak +
    nu_total|``. Per-call MC; n_mc samples.

    Why 2D random-direction (not scalar add): adding ``a_q * sigma``
    as a scalar to ``R`` would double-count. The radial direction
    at peak ``R_det = |eta_hat + delta_eta_mean(t_peak)|`` is fixed,
    and the LF/WF residuals are NOT preferentially aligned with it.
    Worst-case alignment gives the scalar-add answer (max bound
    ~3.5 sigma per process); typical alignment gives the
    quadrature-sum answer (~2 sigma per process). The 2D MC
    captures the actual distribution.

    Falls back gracefully: when ``T_decorr_*`` exceeds
    ``t_horizon_s`` (N_eff < 1) the corresponding Gumbel scaling
    drops to 1.0 -- the residual is treated as quasi-static within
    the window with per-instant Gaussian magnitude. (For LF this
    means the live snapshot of eta_hat already captures the
    relevant offset and adding extra spread would be redundant.
    For WF it means the wave forcing has not changed within the
    window, so a single Gaussian sample suffices.)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Gumbel scaling per process: max-amplitude scales as
    # a_q(N_eff) * sigma instead of ~1 * sigma. Implemented as a
    # multiplier on the per-instant Gaussian sample so that the
    # 2D random direction is preserved.
    N_eff_lf = max(float(t_horizon_s) / float(T_decorr_lf_s), 1.0)
    N_eff_wf = max(float(t_horizon_s) / float(T_decorr_wf_s), 1.0)
    # Use the median Gumbel peak factor a_50 as the magnitude
    # multiplier on the per-instant Gaussian: the per-instant
    # Gaussian's expected magnitude is ~1.25 sigma (Rayleigh mean
    # for sigma_x = sigma_y); the Gumbel-max expected magnitude is
    # ~a_50 sigma. Scaling by a_50 / 1.25 inflates the per-instant
    # 2D Gaussian to have the right MEDIAN magnitude over the
    # window, while preserving the 2D direction distribution. The
    # MC then yields the correct radial P50 / P95 by composition
    # with the deterministic peak offset and the b_hat halo.
    rayleigh_mean_factor = 1.2533141373155001  # = sqrt(pi/2)
    if N_eff_lf > 1.0:
        scale_lf = _gumbel_peak_factor(N_eff_lf, 0.50) / rayleigh_mean_factor
    else:
        scale_lf = 1.0
    if N_eff_wf > 1.0:
        scale_wf = _gumbel_peak_factor(N_eff_wf, 0.50) / rayleigh_mean_factor
    else:
        scale_wf = 1.0

    nu_lf_x = rng.standard_normal(n_mc) * sigma_lf_x * scale_lf
    nu_lf_y = rng.standard_normal(n_mc) * sigma_lf_y * scale_lf
    nu_wf_x = rng.standard_normal(n_mc) * sigma_wf_x * scale_wf
    nu_wf_y = rng.standard_normal(n_mc) * sigma_wf_y * scale_wf
    nu_bh_x = rng.standard_normal(n_mc) * sigma_b_hat_axis
    nu_bh_y = rng.standard_normal(n_mc) * sigma_b_hat_axis

    R = np.hypot(
        offset_xy_at_peak[0] + nu_lf_x + nu_wf_x + nu_bh_x,
        offset_xy_at_peak[1] + nu_lf_y + nu_wf_y + nu_bh_y,
    )
    return float(np.quantile(R, 0.50)), float(np.quantile(R, 0.95))


def _sigmas_intact_axis(sigma_post: LiveSigmaPosterior) -> tuple[float, float]:
    """Per-axis position-noise sigma for the **intact** axis.

    Only the LF channel contributes -- WF noise lives on the wave-
    frequency observer estimate, not on the LF position estimate, and
    the b_hat-radial halo is a property of the post-WCF transient.
    """
    sig_lf_x = float(sigma_post.posterior_lf_x.sigma_median)
    sig_lf_y = float(sigma_post.posterior_lf_y.sigma_median)
    return sig_lf_x, sig_lf_y


def _sigmas_wcf_axis(sigma_post: LiveSigmaPosterior) -> tuple[float, float]:
    """Per-axis position-noise sigma for the **WCF** axis.

    LF + WF + b_hat-radial in quadrature, the same composition used
    by ``evaluate_decision_cell_live.wcfdi_pos_peak_m``. The b_hat
    radial std is split equally between the two axes
    (sigma_b_hat_axis = sigma_R_b_hat_m / sqrt(2)).
    """
    sig_lf_x = float(sigma_post.posterior_lf_x.sigma_median)
    sig_lf_y = float(sigma_post.posterior_lf_y.sigma_median)
    sig_wf_x = float(sigma_post.posterior_wf_x.sigma_median)
    sig_wf_y = float(sigma_post.posterior_wf_y.sigma_median)
    sig_bh_axis = float(sigma_post.sigma_R_b_hat_m) / float(np.sqrt(2.0))
    sigma_x = float(np.sqrt(sig_lf_x ** 2 + sig_wf_x ** 2 + sig_bh_axis ** 2))
    sigma_y = float(np.sqrt(sig_lf_y ** 2 + sig_wf_y ** 2 + sig_bh_axis ** 2))
    return sigma_x, sigma_y


# ---------------------------------------------------------------------------
# Telescope-axis sigma helpers (gangway bar)
# ---------------------------------------------------------------------------


def _wf_6dof_sigma_vector(
    sigma_post: LiveSigmaPosterior,
) -> tuple[np.ndarray, str]:
    """Build the 6-vector of WF per-DOF sigmas for a 6-DOF projection.

    Order: (surge, sway, heave, roll, pitch, yaw) matching
    ``cqa.gangway.telescope_sensitivity_6dof``.

    Returns ``(sig6, coverage)`` where ``coverage`` is

      * ``"full_6dof"``        if ``posterior_wf_heave``,
                               ``posterior_wf_roll`` and
                               ``posterior_wf_pitch`` are all present.
      * ``"horizontal_3dof"``  if any of the three is None. In this
                               case heave/roll/pitch entries of
                               ``sig6`` are set to 0.0 and the caller
                               should display a LOWER-BOUND warning.

    The horizontal-only fallback is operator-honest: it is better to
    show "you are missing roll/pitch monitoring, this is a lower
    bound" than to silently inject a prior we have not earned. The
    user has agreed roll is the dominant out-of-plane contributor at
    realistic gangway base/height geometry, so the warning is
    important to surface in the plot title.
    """
    sig_x = float(sigma_post.posterior_wf_x.sigma_median)
    sig_y = float(sigma_post.posterior_wf_y.sigma_median)
    sig_yaw = float(sigma_post.posterior_wf_yaw.sigma_median)
    pwf_h = sigma_post.posterior_wf_heave
    pwf_r = sigma_post.posterior_wf_roll
    pwf_p = sigma_post.posterior_wf_pitch
    if pwf_h is not None and pwf_r is not None and pwf_p is not None:
        sig_h = float(pwf_h.sigma_median)
        sig_r = float(pwf_r.sigma_median)
        sig_p = float(pwf_p.sigma_median)
        coverage = "full_6dof"
    else:
        sig_h = sig_r = sig_p = 0.0
        coverage = "horizontal_3dof"
    sig6 = np.array([sig_x, sig_y, sig_h, sig_r, sig_p, sig_yaw], dtype=float)
    return sig6, coverage


def _sigma_dL_intact(
    joint: GangwayJointState,
    cfg: CqaConfig,
    sigma_post: LiveSigmaPosterior,
) -> tuple[float, str]:
    """sigma of dL for the intact (steady-state) gangway bar.

    LF horizontal channel only -- consistent with the intact position
    bar, which also uses LF only (the WF noise lives on
    ``eta_wave``, not on the LF position estimate). The b_hat-radial
    halo is a property of the post-WCF transient and is NOT included
    here.

    Coverage tag returned alongside is always ``"horizontal_3dof"``
    for the intact bar -- LF is body-frame surge/sway/yaw only by
    construction. We surface it as a uniform tag so the operator-
    panel title can render a single coverage line for the gangway
    bar.
    """
    c3 = telescope_sensitivity(joint, cfg.gangway)
    sig_lf = np.array([
        sigma_post.posterior_lf_x.sigma_median,
        sigma_post.posterior_lf_y.sigma_median,
        sigma_post.posterior_lf_yaw.sigma_median,
    ], dtype=float)
    var = float(np.sum((c3 * sig_lf) ** 2))
    return float(np.sqrt(max(var, 0.0))), "horizontal_3dof"


def _sigma_dL_wcf(
    joint: GangwayJointState,
    cfg: CqaConfig,
    sigma_post: LiveSigmaPosterior,
) -> tuple[float, str]:
    """sigma of dL for the WCF (post-fault peak) gangway bar.

    LF horizontal (3-DOF, c) + WF (6-DOF when full coverage available,
    otherwise horizontal-only fallback with c6 entries for heave/
    roll/pitch zeroed) + b_hat-radial split equally onto the two
    horizontal axes (mirrors ``_sigmas_wcf_axis``).

    Returns (sigma_dL, coverage_tag).
    """
    c3 = telescope_sensitivity(joint, cfg.gangway)
    c6 = telescope_sensitivity_6dof(joint, cfg.gangway)

    # LF (horizontal 3-DOF only).
    sig_lf = np.array([
        sigma_post.posterior_lf_x.sigma_median,
        sigma_post.posterior_lf_y.sigma_median,
        sigma_post.posterior_lf_yaw.sigma_median,
    ], dtype=float)
    var_lf = float(np.sum((c3 * sig_lf) ** 2))

    # WF (3 or 6 DOF, depending on coverage).
    sig_wf6, coverage = _wf_6dof_sigma_vector(sigma_post)
    var_wf = float(np.sum((c6 * sig_wf6) ** 2))

    # b_hat-radial: split equally across body-x / body-y, like
    # _sigmas_wcf_axis. Yaw component is treated as zero (b_hat
    # radial is a position-equivalent magnitude only). Project
    # through the horizontal entries of c6.
    sig_bh_axis = float(sigma_post.sigma_R_b_hat_m) / float(np.sqrt(2.0))
    var_bh = (c6[0] * sig_bh_axis) ** 2 + (c6[1] * sig_bh_axis) ** 2

    var = var_lf + var_wf + var_bh
    return float(np.sqrt(max(var, 0.0))), coverage


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def summarise_for_operator_live(
    cfg: CqaConfig,
    obs_state: LiveObserverState,
    sigma_post: LiveSigmaPosterior,
    *,
    scenario: Optional[WcfdiScenario] = None,
    t_horizon_s: float = 60.0,
    n_t: int = 121,
    Tp_obs_s: float = 10.0,
    n_mc: int = 2000,
    rng: Optional[np.random.Generator] = None,
    joint: Optional[GangwayJointState] = None,
    cap_residual_N_Nm: Optional[tuple] = None,
    regime_b_geometry: Optional["OperationalCapGeometry"] = None,
    regime_b_surge_cap_N: Optional[float] = None,
) -> LiveOperatorSummary:
    """Build the operator-facing two- or three-bar summary from the live state.

    Parameters
    ----------
    cfg, obs_state, sigma_post
        Same contract as ``evaluate_decision_cell_live``.
    scenario
        Optional WCFDI failure shape. Default = the same canonical
        failure used by ``evaluate_decision_cell_live``.
    t_horizon_s : float, default 60 s.
        Forward-looking horizon for the post-WCF peak. 60 s captures
        the full transient at CSOV scales (mean trajectory has settled
        to its quasi-steady offset by then).
    n_t : int, default 121.
        Number of integration steps over the horizon (dt = 0.5 s by
        default; tight enough to localise the peak time within ~0.5 s).
    Tp_obs_s, n_mc, rng : tuning knobs (defaults are reasonable).
    joint : GangwayJointState, optional.
        If provided, a third "gangway telescope" bar is computed:
        magnitude of telescope-length deviation ``|dL| =
        |L_required - L0|`` with a single worst-margin red threshold
        ``min(L_max - L0, L0 - L_min)`` and matching 60 % / 80 %
        amber/red rule (mirrors the position-bar style and the
        ``evaluate_decision_cell_live`` |c_L @ eta| envelope).
        WF roll/pitch/heave use the optional posteriors from
        ``sigma_post`` when present (full 6-DOF projection); when
        absent, the bar falls back to a horizontal-only projection
        and the coverage tag flags the displayed sigma_dL as a
        lower bound. The position bars are unaffected by ``joint``.

    Returns
    -------
    LiveOperatorSummary. See the class docstring.
    """
    if scenario is None:
        scenario = WcfdiScenario(
            alpha=(2.0 / 3.0,) * 3,
            gamma_immediate=0.5,
            T_realloc=10.0,
        )
    if rng is None:
        rng = np.random.default_rng(0)

    sigma_x_intact, sigma_y_intact = _sigmas_intact_axis(sigma_post)
    sigma_x_wcf, sigma_y_wcf = _sigmas_wcf_axis(sigma_post)
    sigma_R_intact = float(np.hypot(sigma_x_intact, sigma_y_intact))
    sigma_R_wcf = float(np.hypot(sigma_x_wcf, sigma_y_wcf))

    pos_warn = float(cfg.operational_limits.position_warning_radius_m)
    pos_alarm = float(cfg.operational_limits.position_alarm_radius_m)

    # ---- Intact axis ----
    eta_hat_lf = np.asarray(obs_state.eta_hat, dtype=float)
    intact_offset_xy = eta_hat_lf[:2]
    intact_offset_m = float(np.hypot(*intact_offset_xy))
    intact_p50, intact_p95 = _radial_quantiles(
        intact_offset_xy, sigma_x_intact, sigma_y_intact, n_mc=n_mc, rng=rng,
    )
    intact_traffic = _imca_traffic(intact_p95, pos_warn, pos_alarm)

    # ---- WCF axis ----
    aug = _build_aug_for_live(cfg, Tp_obs_s=Tp_obs_s)
    t_grid = np.linspace(0.0, t_horizon_s, n_t)
    # Apply b_hat steady-state bias correction; see analysis.md §12.21.13 and
    # VesselParticulars.b_hat_bias_correction_factor.
    b_corr = float(cfg.vessel.b_hat_bias_correction_factor)
    tau_env = b_corr * np.asarray(obs_state.b_hat, dtype=float)
    # Default: parametric placeholder
    #     tau_lost(t) = -(1 - beta(t)) * tau_env, x0 = 0.
    # When `scenario.tau_lost_pre_wcf` is set (sec.12.21.20):
    #     tau_lost(t) = -tau_lost_pre_wcf * exp(-t / T_realloc_lost)
    #     x0[IDX_TAU_THR] = -tau_lost_pre_wcf
    # See WcfdiScenario.build_pulse_inputs and analysis.md sec.12.21.20.
    tau_lost, x0_pulse = scenario.build_pulse_inputs(
        t_grid=t_grid, tau_env=tau_env,
        n_state=N_STATE, idx_tau_thr=IDX_TAU_THR,
    )

    K_lift = float(getattr(cfg.vessel, "lift_coupling_K_per_rad", 0.0))
    if K_lift > 0.0:
        X = pulse_response_with_lift_coupling(
            aug, t_grid, tau_lost,
            b_hat0=tau_env, K_lift=K_lift,
            x0=x0_pulse,
        )
    else:
        X = pulse_response(aug, t_grid, tau_lost, x0=x0_pulse)
    delta_eta_mean = X[:, 0:3]

    # Deterministic radial trajectory: |eta_hat_LF + delta_eta_mean(t)| in xy.
    eta_xy_t = eta_hat_lf[None, 0:2] + delta_eta_mean[:, 0:2]
    R_det_t = np.hypot(eta_xy_t[:, 0], eta_xy_t[:, 1])
    k_peak = int(np.argmax(R_det_t))
    wcf_t_peak = float(t_grid[k_peak])
    wcf_offset_at_peak = float(R_det_t[k_peak])
    # Window-max P50/P95 (LF + WF Gumbel contributions over t_horizon_s,
    # plus per-instant b_hat halo at the deterministic peak instant).
    # See diagnose_lf_transient_shape.py and analysis.md sec.12.21.9 for
    # why the previous per-instant Gaussian on the LF channel
    # under-predicted the per-seed WCF P95: the LF channel has
    # natural slow drift with sigma_R_LF ~ 0.3-0.9 m and tau_decorr
    # ~15 s, giving N_eff ~ 4 over a 60 s horizon and a Gumbel peak
    # factor a_95 ~ 3.5x sigma -- much larger than the ~1.96x sigma
    # a per-instant Gaussian gives.
    sig_lf_x_w, sig_lf_y_w = _sigmas_intact_axis(sigma_post)  # LF only
    sig_wf_x_w = float(sigma_post.posterior_wf_x.sigma_median)
    sig_wf_y_w = float(sigma_post.posterior_wf_y.sigma_median)
    sig_bh_axis_w = float(sigma_post.sigma_R_b_hat_m) / float(np.sqrt(2.0))
    wcf_p50, wcf_p95 = _radial_window_max_quantiles(
        eta_xy_t[k_peak],
        sig_lf_x_w, sig_lf_y_w,
        sig_wf_x_w, sig_wf_y_w,
        sig_bh_axis_w,
        t_horizon_s=t_horizon_s,
        n_mc=n_mc, rng=rng,
    )
    wcf_traffic = _imca_traffic(wcf_p95, pos_warn, pos_alarm)

    overall = _worst(intact_traffic, wcf_traffic)

    # ---- Gangway telescope bar (optional) ----
    gw_present = False
    gw_dL_p50 = gw_dL_p95 = 0.0
    gw_dL_intact = gw_dL_wcf_at_peak = 0.0
    gw_dL_exc_p50 = gw_dL_exc_p95 = 0.0
    gw_stroke = 0.0
    gw_sigma_dL_intact = gw_sigma_dL_wcf = 0.0
    gw_t_peak = 0.0
    gw_traffic = "green"
    gw_coverage = "horizontal_3dof"
    if joint is not None:
        gw_present = True
        gw_cfg = cfg.gangway
        L0 = float(joint.L)
        L_min = float(gw_cfg.telescope_min)
        L_max = float(gw_cfg.telescope_max)
        gw_stroke = max(min(L_max - L0, L0 - L_min), 0.0)

        c3 = telescope_sensitivity(joint, gw_cfg)
        gw_sigma_dL_intact, _ = _sigma_dL_intact(joint, cfg, sigma_post)
        gw_sigma_dL_wcf, gw_coverage = _sigma_dL_wcf(joint, cfg, sigma_post)

        # Intact deterministic |dL|: magnitude of LF eta_hat projected
        # onto the telescope axis.
        gw_dL_intact = float(np.abs(c3 @ eta_hat_lf))

        # WCF deterministic |dL| trajectory: |dL(t)| = |c3 . (eta_hat_LF
        # + delta_eta_mean(t))|. Take time of maximum |dL|.
        dL_t_signed = (eta_hat_lf[None, :] + delta_eta_mean) @ c3
        dL_t = np.abs(dL_t_signed)
        k_peak_dL = int(np.argmax(dL_t))
        gw_t_peak = float(t_grid[k_peak_dL])
        gw_dL_wcf_at_peak = float(dL_t[k_peak_dL])

        # Quantiles of |dL| at the peak instant. Underlying signed dL
        # is N(mu, sigma_dL_wcf) with mu = dL_t_signed[k_peak_dL]; the
        # magnitude follows a folded normal. Per-call MC for
        # consistency with the position bars (no special functions).
        mu_signed = float(dL_t_signed[k_peak_dL])
        nu = (rng.standard_normal(n_mc) if gw_sigma_dL_wcf > 0.0
              else np.zeros(n_mc))
        dL_abs_samples = np.abs(mu_signed + gw_sigma_dL_wcf * nu)
        gw_dL_p50 = float(np.quantile(dL_abs_samples, 0.50))
        gw_dL_p95 = float(np.quantile(dL_abs_samples, 0.95))

        # Excursion-only quantiles: the WCFDI-induced *change* in dL
        # relative to the live LF baseline. The apples-to-apples
        # comparator quantity for max|dL_truth(t) - dL_pre| over the
        # post-WCF window. Two regimes:
        #
        # (a) "measured" -- when sigma_post.sigma_dL_wf_measured and
        #     T_zc_dL_wf_measured are both provided (the upstream
        #     sigma-estimator is monitoring the gangway WF channel
        #     directly): combine the LF deterministic peak with a
        #     Gumbel/Rice extreme-value WF contribution sized to the
        #     near-peak duration of the LF transient. Formally:
        #
        #         pred_pq = |c3 . delta_eta_mean(t_peak)|
        #                   + a_q(N_eff) * sigma_dL_wf_measured
        #         N_eff   = tau_LF / T_zc_dL_wf_measured
        #         tau_LF  = duration over which |c3 . delta_eta_mean(t)|
        #                   stays >= 0.8 * peak
        #         a_50    = sqrt(2 ln N_eff) - 0.5772 / sqrt(2 ln N_eff)
        #         a_95    = sqrt(2 ln N_eff) - ln(-ln 0.95)
        #                                       / sqrt(2 ln N_eff)
        #
        #     The LF-peak and worst WF crest are added linearly. This
        #     is conservative: in reality the LF transient sits at
        #     its max only briefly and the worst WF crest will not
        #     coincide with it, but assuming coincidence gives a
        #     defensible upper bound on the worst-case window-max
        #     telescope excursion. Strictly observer-state +
        #     Bayesian-posterior driven (no Tp / Hs / sea-state
        #     lookup).
        #
        # (b) "folded-normal fallback" -- when the measured fields
        #     are absent: collapse to a single-instant folded-normal
        #     halo at the deterministic peak instant with sigma =
        #     sigma_dL_wcf (per-DOF posterior projection through c6,
        #     plus b_hat-axial). This is what the previous version
        #     of this panel did. Reported P50 is structurally low
        #     vs window-max truth (single-instant statistic vs
        #     extreme-value statistic) but cheap and self-contained.
        mu_signed_exc = float(delta_eta_mean[k_peak_dL] @ c3)
        dL_LF_peak = float(abs(mu_signed_exc))
        s_meas = sigma_post.sigma_dL_wf_measured
        T_zc_meas = sigma_post.T_zc_dL_wf_measured
        if (s_meas is not None and T_zc_meas is not None
                and s_meas > 0.0 and T_zc_meas > 0.0):
            # tau_LF: duration the |LF transient| stays above 0.8*peak.
            dL_LF_t_abs = np.abs(delta_eta_mean @ c3)
            peak_abs = float(dL_LF_t_abs.max())
            if peak_abs > 0.0:
                near_peak = dL_LF_t_abs >= 0.8 * peak_abs
                # Contiguous-block duration around the peak (use the
                # longest contiguous true-block to ignore minor
                # secondary lobes).
                if near_peak.any():
                    starts = np.where(np.diff(near_peak.astype(int))
                                      == 1)[0] + 1
                    ends = np.where(np.diff(near_peak.astype(int))
                                    == -1)[0] + 1
                    if near_peak[0]:
                        starts = np.r_[0, starts]
                    if near_peak[-1]:
                        ends = np.r_[ends, near_peak.size]
                    blocks = list(zip(starts, ends))
                    # Pick the block containing the peak instant.
                    k = int(np.argmax(dL_LF_t_abs))
                    tau_LF_s = 0.0
                    for s_i, e_i in blocks:
                        if s_i <= k < e_i:
                            tau_LF_s = float(t_grid[e_i - 1]
                                             - t_grid[s_i])
                            break
                    if tau_LF_s <= 0.0:
                        tau_LF_s = float(t_grid[1] - t_grid[0])
                else:
                    tau_LF_s = float(t_grid[1] - t_grid[0])
            else:
                tau_LF_s = T_zc_meas
            # Gumbel peak factors. Floor N_eff at 1 (single crest)
            # to keep ln defined; if tau_LF < T_zc the WF gets one
            # half-cycle within the LF peak duration -- treat as 1.
            N_eff = max(tau_LF_s / float(T_zc_meas), 1.0)
            ln_term = float(np.sqrt(2.0 * np.log(max(N_eff, 1.001))))
            # 0.5772... = Euler-Mascheroni constant; Gumbel mean.
            # ln(-ln(0.95)) ~= -2.9702 -> negative => +2.9702/ln_term.
            a_50 = ln_term - 0.5772156649 / ln_term
            a_95 = ln_term - float(np.log(-np.log(0.95))) / ln_term
            gw_dL_exc_p50 = dL_LF_peak + a_50 * float(s_meas)
            gw_dL_exc_p95 = dL_LF_peak + a_95 * float(s_meas)
        else:
            # Fallback: per-instant folded-normal at deterministic peak.
            dL_exc_samples = np.abs(mu_signed_exc + gw_sigma_dL_wcf * nu)
            gw_dL_exc_p50 = float(np.quantile(dL_exc_samples, 0.50))
            gw_dL_exc_p95 = float(np.quantile(dL_exc_samples, 0.95))

        # Traffic light: same IMCA convention as the position bars and
        # ``evaluate_decision_cell_live`` -- amber at 60% of stroke,
        # red at 80% of stroke. Driven by P95 of |dL|.
        if gw_stroke > 0.0:
            gw_traffic = _imca_traffic(
                gw_dL_p95, 0.60 * gw_stroke, 0.80 * gw_stroke,
            )
        else:
            gw_traffic = "green"
        overall = _worst(overall, gw_traffic)

    # ---- Regime-B saturation severity (optional) ----
    # Driven by the rolling delivered-thrust buffer on the observer
    # state and the residual polytope cap of the surviving thruster
    # set. See cqa.live_regime_b and analysis.md sec.12.21.21.16-21.
    regime_b_present = False
    regime_b_severity = 0.0
    regime_b_p_sat = None
    regime_b_mu = None
    regime_b_sigma = None
    regime_b_cap = None
    regime_b_traffic = "green"
    if (
        obs_state.tau_buffer is not None
        and obs_state.tau_buffer_fs_hz is not None
        and (cap_residual_N_Nm is not None or regime_b_geometry is not None)
    ):
        if regime_b_geometry is not None:
            if regime_b_surge_cap_N is None:
                raise ValueError(
                    "regime_b_geometry requires regime_b_surge_cap_N."
                )
            rb: RegimeBSeverity = estimate_regime_b_severity(
                tau_buffer=obs_state.tau_buffer,
                fs_hz=float(obs_state.tau_buffer_fs_hz),
                geometry=regime_b_geometry,
                surge_cap_N=float(regime_b_surge_cap_N),
            )
        else:
            rb = estimate_regime_b_severity(
                tau_buffer=obs_state.tau_buffer,
                fs_hz=float(obs_state.tau_buffer_fs_hz),
                cap_residual=cap_residual_N_Nm,
            )
        regime_b_present = True
        regime_b_severity = rb.severity
        regime_b_p_sat = rb.p_sat
        regime_b_mu = rb.mu
        regime_b_sigma = rb.sigma
        regime_b_cap = rb.cap_residual
        regime_b_traffic = rb.traffic
        overall = _worst(overall, regime_b_traffic)

    return LiveOperatorSummary(
        intact_R_p50=intact_p50,
        intact_R_p95=intact_p95,
        intact_R_offset_m=intact_offset_m,
        intact_traffic=intact_traffic,
        wcf_R_p50=wcf_p50,
        wcf_R_p95=wcf_p95,
        wcf_R_offset_at_peak_m=wcf_offset_at_peak,
        wcf_t_peak_s=wcf_t_peak,
        wcf_traffic=wcf_traffic,
        pos_warning_radius_m=pos_warn,
        pos_alarm_radius_m=pos_alarm,
        sigma_R_intact_m=sigma_R_intact,
        sigma_R_wcf_m=sigma_R_wcf,
        overall_traffic=overall,
        gangway_present=gw_present,
        gangway_dL_p50=gw_dL_p50,
        gangway_dL_p95=gw_dL_p95,
        gangway_dL_intact_offset=gw_dL_intact,
        gangway_dL_wcf_offset_at_peak=gw_dL_wcf_at_peak,
        gangway_dL_excursion_p50=gw_dL_exc_p50,
        gangway_dL_excursion_p95=gw_dL_exc_p95,
        gangway_stroke_m=gw_stroke,
        gangway_sigma_dL_intact_m=gw_sigma_dL_intact,
        gangway_sigma_dL_wcf_m=gw_sigma_dL_wcf,
        gangway_t_peak_s=gw_t_peak,
        gangway_traffic=gw_traffic,
        gangway_wf_coverage=gw_coverage,
        regime_b_present=regime_b_present,
        regime_b_severity=regime_b_severity,
        regime_b_p_sat=regime_b_p_sat,
        regime_b_mu_N_Nm=regime_b_mu,
        regime_b_sigma_N_Nm=regime_b_sigma,
        regime_b_cap_residual_N_Nm=regime_b_cap,
        regime_b_traffic=regime_b_traffic,
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


_TRAFFIC_COLOURS = {
    "green": "#2ca02c",
    "amber": "#ff9900",
    "red": "#d62728",
}


def plot_live_operator_summary(summary: LiveOperatorSummary, fig=None):
    """Render the operator panel.

    Two bars when ``summary.gangway_present`` is False:
      * Top: "RIGHT NOW (intact)"
      * Bot: "IF WCF NOW"
    Three bars when ``summary.gangway_present`` is True; the third
    "GANGWAY (telescope)" bar uses an unsigned |dL| axis with a single
    worst-margin red threshold (80 % of stroke) and matching amber
    (60 % of stroke), mirroring the position-bar style.

    P50 shown as light open marker, P95 shown as filled diamond. Bar
    tint = traffic-light colour.

    Title prints both/all numbers in plain language plus the time-
    to-peak for the WCF row.
    """
    import matplotlib.pyplot as plt

    n_rows = 3 if summary.gangway_present else 2
    if fig is None:
        fig, axes = plt.subplots(n_rows, 1, figsize=(11, 2.6 * n_rows + 0.2),
                                 gridspec_kw=dict(hspace=1.4))
    else:
        axes = fig.subplots(n_rows, 1, gridspec_kw=dict(hspace=1.4))
    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.07, right=0.97)

    bar_max = max(
        summary.pos_alarm_radius_m * 1.3,
        summary.intact_R_p95 * 1.2,
        summary.wcf_R_p95 * 1.2,
        summary.pos_alarm_radius_m + 0.5,
    )

    def _draw_bar(ax, p50, p95, traffic, title_main, title_sub):
        col = _TRAFFIC_COLOURS[traffic]
        ax.barh(0, bar_max, height=0.5, color=col, alpha=0.25,
                edgecolor=col, linewidth=2)
        ax.plot(p50, 0, marker="o", markersize=11,
                markerfacecolor="white", markeredgecolor="#333333",
                markeredgewidth=2, label=f"P50 = {p50:.2f} m")
        ax.plot(p95, 0, marker="D", markersize=12,
                markerfacecolor=col, markeredgecolor="black",
                markeredgewidth=2, label=f"P95 = {p95:.2f} m")
        ax.axvline(summary.pos_warning_radius_m, color="#ff9900",
                   ls="--", lw=2,
                   label=f"amber > {summary.pos_warning_radius_m:.0f} m")
        ax.axvline(summary.pos_alarm_radius_m, color="#d62728",
                   ls="--", lw=2,
                   label=f"red > {summary.pos_alarm_radius_m:.0f} m")
        ax.set_xlim(0, bar_max)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("radial distance from setpoint [m]")
        ax.set_title(f"{title_main}   [{traffic.upper()}]\n{title_sub}",
                     fontsize=11, loc="left")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
        ax.grid(True, axis="x", alpha=0.3)

    # Intact (top)
    _draw_bar(
        axes[0],
        summary.intact_R_p50, summary.intact_R_p95, summary.intact_traffic,
        "RIGHT NOW (intact)",
        f"P50 = {summary.intact_R_p50:.2f} m   "
        f"P95 = {summary.intact_R_p95:.2f} m   "
        f"(live offset {summary.intact_R_offset_m:.2f} m, "
        f"sigma_R = {summary.sigma_R_intact_m:.2f} m)",
    )

    # WCF (middle if gangway present, else bottom)
    _draw_bar(
        axes[1],
        summary.wcf_R_p50, summary.wcf_R_p95, summary.wcf_traffic,
        "IF WCF NOW",
        f"P50 = {summary.wcf_R_p50:.2f} m   "
        f"P95 = {summary.wcf_R_p95:.2f} m   "
        f"(peak in ~{summary.wcf_t_peak_s:.0f} s, "
        f"deterministic offset at peak {summary.wcf_R_offset_at_peak_m:.2f} m, "
        f"sigma_R = {summary.sigma_R_wcf_m:.2f} m)",
    )

    # Gangway (bottom, optional)
    if summary.gangway_present:
        _draw_abs_dL_bar(axes[2], summary)

    fig.suptitle(
        f"Live operator panel  -  overall: {summary.overall_traffic.upper()}",
        fontsize=13, y=0.95,
    )

    return fig


def _draw_abs_dL_bar(ax, summary: LiveOperatorSummary):
    """Render the |dL| telescope bar (mirrors the position-bar style).

    x = |dL| in metres, single non-negative axis. P50 (open circle)
    + P95 (filled diamond), 60%-of-stroke amber dashed and 80%-of-
    stroke red dashed thresholds, traffic-light tint matching
    ``summary.gangway_traffic``.
    """
    col = _TRAFFIC_COLOURS[summary.gangway_traffic]
    stroke = summary.gangway_stroke_m
    bar_max = max(stroke * 1.05, summary.gangway_dL_p95 * 1.2, 0.5)

    ax.barh(0, bar_max, height=0.5, color=col, alpha=0.25,
            edgecolor=col, linewidth=2)
    ax.plot(summary.gangway_dL_p50, 0, marker="o", markersize=11,
            markerfacecolor="white", markeredgecolor="#333333",
            markeredgewidth=2,
            label=f"P50 = {summary.gangway_dL_p50:.2f} m")
    ax.plot(summary.gangway_dL_p95, 0, marker="D", markersize=12,
            markerfacecolor=col, markeredgecolor="black",
            markeredgewidth=2,
            label=f"P95 = {summary.gangway_dL_p95:.2f} m")
    if stroke > 0.0:
        ax.axvline(0.60 * stroke, color="#ff9900", ls="--", lw=2,
                   label=f"amber > {0.60 * stroke:.2f} m")
        ax.axvline(0.80 * stroke, color="#d62728", ls="--", lw=2,
                   label=f"red > {0.80 * stroke:.2f} m")
    ax.set_xlim(0, bar_max)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("|telescope length deviation| from nominal  [m]")

    coverage_note = ("" if summary.gangway_wf_coverage == "full_6dof"
                     else "  (LOWER BOUND: roll/pitch/heave WF posteriors not provided)")
    ax.set_title(
        f"GANGWAY (telescope)   [{summary.gangway_traffic.upper()}]\n"
        f"P50 = {summary.gangway_dL_p50:.2f} m   "
        f"P95 = {summary.gangway_dL_p95:.2f} m   "
        f"(WCF peak in ~{summary.gangway_t_peak_s:.0f} s, "
        f"stroke = {stroke:.2f} m, "
        f"sigma_dL = {summary.gangway_sigma_dL_wcf_m:.2f} m){coverage_note}",
        fontsize=11, loc="left",
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.grid(True, axis="x", alpha=0.3)
