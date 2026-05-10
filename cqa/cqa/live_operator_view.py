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
from .gangway import GangwayJointState
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
) -> LiveOperatorSummary:
    """Build the operator-facing two-bar summary from the live state.

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
    """Render the two-bar operator panel.

    Top row "RIGHT NOW (intact)" and bottom row "IF WCF NOW", same
    radial-distance axis [0, max(P95, alarm) * 1.2]. P50 shown as
    light open marker, P95 shown as filled diamond. IMCA warning
    (amber) and alarm (red) thresholds drawn as dashed verticals.
    Bar tint = traffic-light colour driven by P95 vs the thresholds.

    Title prints both numbers in plain language plus the time-to-peak
    for the WCF row.
    """
    import matplotlib.pyplot as plt

    if fig is None:
        fig, axes = plt.subplots(2, 1, figsize=(11, 5.5),
                                 gridspec_kw=dict(hspace=1.4))
    else:
        axes = fig.subplots(2, 1, gridspec_kw=dict(hspace=1.4))
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.07, right=0.97)

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

    # WCF (bottom)
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

    fig.suptitle(
        f"Live operator panel  -  overall: {summary.overall_traffic.upper()}",
        fontsize=13, y=0.95,
    )

    return fig
