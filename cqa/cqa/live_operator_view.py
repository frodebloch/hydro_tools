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

WCF axis: under-predicts the realised single-realisation post-WCF
peak by ~15-20 % across most of the matrix. This was extensively
diagnosed (see scripts/p7_brucon_validation):

  * Adding nu_hat as IC (x0[3:6] = x0[9:12] = nu_hat) changes the
    peak by +5 % only -- LF velocity at the WCF instant carries
    little energy.
  * Adding b_hat as IC (x0[12:15] = b_hat) explodes the prediction
    3-4x, confirming the existing x0 = 0 IC is correct: the linear
    augmented system is set up as a *delta around intact steady-
    state*, with tau_env re-entering through tau_lost(t) as the
    additional perturbation only.
  * The deterministic mean trajectory itself (no noise, no halo) is
    accurate to within +/-10 % of the brucon ensemble-mean post-WCF
    transient peak across the matrix.
  * The remaining 15-20 % gap is therefore a comparator-statistic
    effect, not a model bug: the panel reports P95 of |R| at the
    deterministic peak time under a single Gaussian halo at that
    instant, while the truth statistic is the single-realisation
    max over a 60-s window of a *correlated* noise process. Even
    with strong correlation (LF tau_decorr ~60 s, comparable to the
    transient horizon), max-over-window samples ~0.3-0.5 sigma above
    the deterministic peak, which at typical sigma_R ~0.5-0.7 m and
    peaks ~2 m gives the observed ~10-15 % inflation.
  * iid-per-timestep noise gives ~+80 % over-prediction (confirms
    correlation matters and any fix must respect LF/WF decorrelation
    timescales).

Known residual physics gap: pwo (beam-on, current = +20 deg) under-
predicts by ~50 %, the documented missing dF_x/dpsi mirror term
(analysis.md sec.12). NOT a comparator effect; tracked separately.
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
)
from .transient import WcfdiScenario
from .decision_matrix import _imca_traffic, _worst


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
    # 12-cell brucon roll-up of the |dL| bar in horizontal-3DOF
    # mode (forward gangway, h=15 m, L0=25 m; see scripts/p7_brucon_
    # validation/roll_up_gangway_bar.py):
    #
    #          P50 bias    P95 bias    coverage
    #   bf4    -38..-45%   -31..-32%   43..57%
    #   bf6    -64..-72%   -46..-53%   10..20%
    #   bf8    -68..-69%   -45..-51%    7..23%
    #   pwo    -93%        -78%         0%
    #   pwq30  -84%        -61%         3%
    #
    # Roughly 2x the position-bar gap. This is the EXPECTED
    # outcome of the deferred 6-DOF roll/pitch/heave posterior
    # work: c6[4] ~ +23 m/rad pitch lever and c6[3] ~ 20 m/rad
    # roll lever mean even ~1 deg pitch RMS at bf6 alone
    # contributes ~0.40 m to sigma_dL, comparable to the entire
    # horizontal-only sigma_dL ~0.34 m we currently predict.
    # Folded through the P95 statistic, true sigma_dL ~2x
    # predicted -> P95 ~2x under-predicted, matching the table.
    # The pwo/pwq30 cells additionally inherit the documented
    # missing dF_x/dpsi mirror term (analysis.md sec.12).
    # Operator-panel title and ``gangway_wf_coverage`` field
    # surface this as "LOWER BOUND" until the 6-DOF posteriors
    # are wired.
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
    gangway_stroke_m: float = 0.0   # min(L_max - L0, L0 - L_min), m
    gangway_sigma_dL_intact_m: float = 0.0
    gangway_sigma_dL_wcf_m: float = 0.0
    gangway_t_peak_s: float = 0.0
    gangway_traffic: str = "green"
    gangway_wf_coverage: str = "horizontal_3dof"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
    """
    if rng is None:
        rng = np.random.default_rng(0)
    nu_x = rng.standard_normal(n_mc) * sigma_x
    nu_y = rng.standard_normal(n_mc) * sigma_y
    R = np.hypot(offset_xy[0] + nu_x, offset_xy[1] + nu_y)
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
    tau_env = np.asarray(obs_state.b_hat, dtype=float)
    gamma_imm = float(scenario.gamma_immediate)
    T_realloc = float(scenario.T_realloc) if scenario.T_realloc > 0 else 1e-9
    beta_t = 1.0 + (gamma_imm - 1.0) * np.exp(-t_grid / T_realloc)
    tau_lost = (beta_t[:, None] - 1.0) * (-tau_env[None, :])

    K_lift = float(getattr(cfg.vessel, "lift_coupling_K_per_rad", 0.0))
    if K_lift > 0.0:
        X = pulse_response_with_lift_coupling(
            aug, t_grid, tau_lost,
            b_hat0=tau_env, K_lift=K_lift,
            x0=np.zeros(N_STATE),
        )
    else:
        X = pulse_response(aug, t_grid, tau_lost, x0=np.zeros(N_STATE))
    delta_eta_mean = X[:, 0:3]

    # Deterministic radial trajectory: |eta_hat_LF + delta_eta_mean(t)| in xy.
    eta_xy_t = eta_hat_lf[None, 0:2] + delta_eta_mean[:, 0:2]
    R_det_t = np.hypot(eta_xy_t[:, 0], eta_xy_t[:, 1])
    k_peak = int(np.argmax(R_det_t))
    wcf_t_peak = float(t_grid[k_peak])
    wcf_offset_at_peak = float(R_det_t[k_peak])
    wcf_p50, wcf_p95 = _radial_quantiles(
        eta_xy_t[k_peak], sigma_x_wcf, sigma_y_wcf, n_mc=n_mc, rng=rng,
    )
    wcf_traffic = _imca_traffic(wcf_p95, pos_warn, pos_alarm)

    overall = _worst(intact_traffic, wcf_traffic)

    # ---- Gangway telescope bar (optional) ----
    gw_present = False
    gw_dL_p50 = gw_dL_p95 = 0.0
    gw_dL_intact = gw_dL_wcf_at_peak = 0.0
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
        gangway_stroke_m=gw_stroke,
        gangway_sigma_dL_intact_m=gw_sigma_dL_intact,
        gangway_sigma_dL_wcf_m=gw_sigma_dL_wcf,
        gangway_t_peak_s=gw_t_peak,
        gangway_traffic=gw_traffic,
        gangway_wf_coverage=gw_coverage,
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
