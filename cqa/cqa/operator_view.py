"""Operator-facing summary of the WCFDI MC result.

Goal: boil the result of a Monte-Carlo over WCF starting states down to
the smallest set of numbers that an operator can act on. Two axes, two
traffic lights, one figure -- following IMCA M254 Rev.1 Figure 8
"Decision matrix":

  - VESSEL CAPABILITY (footprint at the gangway base, vessel body frame):
        green : footprint < 2 m    -> approach installation for connection
        amber : footprint > 2 m    -> approach and check at connection distance
        red   : footprint > 4 m    -> Master to determine whether to approach

  - GANGWAY CAPABILITY (utilisation of the gangway limit instrumentation):
        green : util < 60 %        -> approach installation for connection
        amber : util > 60 %        -> approach and check connection distance
        red   : util > 80 %        -> Master to determine; while connected,
                                      abort and return gangway to vessel.

For the cqa prototype:
  * "Vessel footprint" is the P95 of the post-WCFDI peak position
    deviation at the gangway base (Monte-Carlo over WCF starting states),
    in metres. Reported alongside the conditional exceedance
    probabilities P(footprint > 2 m) and P(footprint > 4 m).
  * "Gangway utilisation" is the worst-side combined demand on the
    telescope stroke:
        util = max( (gw_p95 + k_wave * sigma_L_wave) / stroke_lower,
                    (gw_p95 + k_wave * sigma_L_wave) / stroke_upper )
    where stroke_lower = L0 - L_min, stroke_upper = L_max - L0, gw_p95
    is the 95th-percentile of the slow-content peak |dL| from the WCFDI
    MC, and k_wave * sigma_L_wave is the 1st-order wave-frequency
    contribution from cqa.wave_response.sigma_L_wave. The end-stop hit
    probability P(slow+wave hits L_min/L_max) is reported as a
    complementary diagnostic.

The IMCA M254 Rev.1 thresholds are the defaults; both axes accept
overrides for vessel-/operator-specific tuning.

Important: the conditional exceedance probabilities reported alongside
the traffic light are CONDITIONAL on a WCF happening from the current
operating state. They do NOT include the underlying WCF event rate
(which would come from a class-society fault-tree/FMEA analysis and is
out of scope for the online tool). The operator-facing label therefore
reads "if a WCF occurs now: ..." to avoid being misread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .config import CqaConfig, OperationalLimits
from .wcfdi_mc import WcfdiMcResult


# ---------------------------------------------------------------------------
# Summary dataclass
# ---------------------------------------------------------------------------


@dataclass
class OperatorSummary:
    """Two-axis IMCA M254-style operator summary derived from a WcfdiMcResult.

    All probabilities are CONDITIONAL on a WCF happening at t=0 from the
    current operating state.

    Vessel-capability axis (footprint at the gangway base, vessel frame):
      pos_p_warning : P(peak |dp_base| > 2 m)
      pos_p_alarm   : P(peak |dp_base| > 4 m)
      pos_p50       : 50th percentile of peak |dp_base| [m]
      pos_p95       : 95th percentile of peak |dp_base| [m]      <- "footprint"
      pos_warning_radius_m, pos_alarm_radius_m : the IMCA M254 limits used [m]
      pos_traffic   : "green" | "amber" | "red" per IMCA M254 Fig.8
                      (driven by pos_p95 vs warning/alarm radii).

    Gangway-capability axis (telescope utilisation):
      gw_util_imca  : worst-side combined slow+wave demand divided by
                      worst-side stroke. 1.0 = at end-stop, 0.6 = amber
                      threshold, 0.8 = red threshold.
      gw_imca_warning_frac, gw_imca_alarm_frac : the thresholds used
                                                 (defaults 0.60, 0.80).
      gw_traffic    : "green" | "amber" | "red" per IMCA M254 Fig.8
                      (driven by gw_util_imca vs the two fractions).
      gw_p_alarm    : P(telescope leaves [L_min, L_max] at any t),
                      ALREADY INCLUDING the wave-frequency contribution
                      via the margin-shrink rule below if sigma_L_wave
                      was provided. Reported alongside util as a
                      complementary spillover diagnostic.
      gw_p50, gw_p95: percentiles of peak |dL_slow| [m] (DP slow content
                      only; wave content shown separately as
                      gw_sigma_L_wave_m and gw_k_sigma_L_wave_m).
      gw_lower_margin_p50 / p95 : how close the trajectory got to L_min
                      (after wave-margin-shrink if applicable).
      gw_upper_margin_p50 / p95 : same for L_max.
      gw_telescope_min_m, gw_telescope_max_m, gw_L0_m : limits and
                      nominal length used.
      gw_sigma_L_wave_m   : 1-sigma wave-frequency telescope deviation
                            (NaN if wave channel not used).
      gw_k_sigma_L_wave_m : k_wave * sigma_L_wave -- the deterministic
                            margin-shrink applied (NaN if not used).
      gw_k_wave           : the peak factor used (1.96 default; matches
                            evaluate_operability k_sigma).
    """

    # Conditioning context (so the headline label can be honest)
    n_samples: int
    sample_mode: str
    weather_summary: str  # e.g. "Vw=14 m/s, Hs=2.8 m, Vc=0.5 m/s, beam"

    # --- Vessel-capability axis ---
    pos_p_warning: float
    pos_p_alarm: float
    pos_p50: float
    pos_p95: float
    pos_warning_radius_m: float
    pos_alarm_radius_m: float
    pos_traffic: str

    # --- Gangway-capability axis ---
    gw_util_imca: float
    gw_imca_warning_frac: float
    gw_imca_alarm_frac: float
    gw_traffic: str
    gw_p_alarm: float
    gw_p50: float
    gw_p95: float
    gw_lower_margin_p50: float
    gw_lower_margin_p95: float
    gw_upper_margin_p50: float
    gw_upper_margin_p95: float
    gw_telescope_min_m: float
    gw_telescope_max_m: float
    gw_L0_m: float
    gw_sigma_L_wave_m: float
    gw_k_sigma_L_wave_m: float
    gw_k_wave: float


# ---------------------------------------------------------------------------
# Build summary from MC result
# ---------------------------------------------------------------------------


def summarise_for_operator(
    res: WcfdiMcResult,
    cfg: CqaConfig,
    weather_summary: str = "",
    sigma_L_wave: float = 0.0,
    k_wave: float = 1.96,
    gw_imca_warning_frac: float = 0.60,
    gw_imca_alarm_frac: float = 0.80,
) -> OperatorSummary:
    """Reduce a WcfdiMcResult to the IMCA M254 Fig.8 two-axis summary.

    Pulls the position warning/alarm radii from `cfg.operational_limits`
    (default 2 m / 4 m, matching IMCA M254) and the telescope end-stops
    from `cfg.gangway`.

    IMCA M254 Rev.1 Figure 8 thresholds
    -----------------------------------
    Vessel capability (footprint = pos_p95):
        green if pos_p95 < pos_warning_radius_m  (default 2 m)
        amber if pos_p95 > pos_warning_radius_m  (default 2 m)
        red   if pos_p95 > pos_alarm_radius_m    (default 4 m)
    Gangway capability (utilisation of limit instrumentation):
        util = max( (gw_p95 + k_wave * sigma_L_wave) / (L0 - L_min),
                    (gw_p95 + k_wave * sigma_L_wave) / (L_max - L0) )
        green if util < gw_imca_warning_frac     (default 0.60)
        amber if util > gw_imca_warning_frac     (default 0.60)
        red   if util > gw_imca_alarm_frac       (default 0.80)

    Wave-frequency telescope content
    --------------------------------
    The DP MC samples the **slow** (DP closed-loop + WCFDI transient)
    telescope content. The 1st-order wave-frequency content is an
    independent, zero-mean, Gaussian process with one-sigma magnitude
    ``sigma_L_wave`` (computed by ``cqa.wave_response.sigma_L_wave`` from
    the pdstrip RAOs and the operator-set sea state). To produce a
    combined operability statement, each per-sample telescope margin
    (margin_low, margin_high) is shrunk by ``k_wave * sigma_L_wave``,
    representing the expected k_wave-sigma reach of the wave-frequency
    component over the response window. Default ``k_wave = 1.96`` matches
    the ``k_sigma`` used by ``evaluate_operability`` for the static
    operability check, giving consistent semantics between online and
    static analyses.

    Pass ``sigma_L_wave = 0.0`` (default) to disable the wave channel
    and reproduce the old behaviour exactly.
    """
    limits = cfg.operational_limits

    valid_pos = ~np.isnan(res.pos_peak)
    pos_peak = res.pos_peak[valid_pos]
    pos_p_warning = float(np.mean(pos_peak > limits.position_warning_radius_m))
    pos_p_alarm = float(np.mean(pos_peak > limits.position_alarm_radius_m))
    pos_p50 = float(np.percentile(pos_peak, 50))
    pos_p95 = float(np.percentile(pos_peak, 95))

    # IMCA M254 vessel-capability colour: footprint = pos_p95 vs
    # warning/alarm radii (2 m / 4 m by default).
    pos_traffic = _imca_traffic(
        pos_p95,
        limits.position_warning_radius_m,
        limits.position_alarm_radius_m,
    )

    valid_gw = ~np.isnan(res.dL_peak_abs)
    dL = res.dL_peak_abs[valid_gw]
    gw_p50 = float(np.percentile(dL, 50))
    gw_p95 = float(np.percentile(dL, 95))

    # Wave-margin shrink. If sigma_L_wave == 0 this is a no-op and we
    # exactly reproduce the prior behaviour.
    delta_wave = float(k_wave * sigma_L_wave)
    margin_low = res.margin_low[valid_gw] - delta_wave
    margin_high = res.margin_high[valid_gw] - delta_wave
    operable_combined = (margin_low > 0.0) & (margin_high > 0.0)
    gw_p_alarm = float(1.0 - np.mean(operable_combined))

    # Margin percentiles (P50 = typical, P95 = the worst 5% of cases ->
    # use the 5th percentile because lower margin = closer to limit).
    gw_lower_margin_p50 = float(np.percentile(margin_low, 50))
    gw_lower_margin_p95 = float(np.percentile(margin_low, 5))
    gw_upper_margin_p50 = float(np.percentile(margin_high, 50))
    gw_upper_margin_p95 = float(np.percentile(margin_high, 5))

    # IMCA M254 gangway-capability utilisation: worst-side combined
    # demand / available stroke. Treats the slow P95 and the
    # k_wave-sigma wave reach as additive (deterministic envelope), the
    # same convention used to shrink the per-sample margins above.
    info = res.info
    L0 = float(info.get("L0", info["joint"].L))
    L_min = float(cfg.gangway.telescope_min)
    L_max = float(cfg.gangway.telescope_max)
    stroke_lower = max(L0 - L_min, 1e-9)
    stroke_upper = max(L_max - L0, 1e-9)
    demand_combined = gw_p95 + delta_wave
    util_lower = demand_combined / stroke_lower
    util_upper = demand_combined / stroke_upper
    gw_util_imca = float(max(util_lower, util_upper))
    gw_traffic = _imca_traffic(
        gw_util_imca, gw_imca_warning_frac, gw_imca_alarm_frac
    )

    return OperatorSummary(
        n_samples=int(info.get("n_samples", len(res.dL_peak))),
        sample_mode=str(info.get("sample_mode", "?")),
        weather_summary=weather_summary,
        pos_p_warning=pos_p_warning,
        pos_p_alarm=pos_p_alarm,
        pos_p50=pos_p50,
        pos_p95=pos_p95,
        pos_warning_radius_m=limits.position_warning_radius_m,
        pos_alarm_radius_m=limits.position_alarm_radius_m,
        pos_traffic=pos_traffic,
        gw_util_imca=gw_util_imca,
        gw_imca_warning_frac=float(gw_imca_warning_frac),
        gw_imca_alarm_frac=float(gw_imca_alarm_frac),
        gw_traffic=gw_traffic,
        gw_p_alarm=gw_p_alarm,
        gw_p50=gw_p50,
        gw_p95=gw_p95,
        gw_lower_margin_p50=gw_lower_margin_p50,
        gw_lower_margin_p95=gw_lower_margin_p95,
        gw_upper_margin_p50=gw_upper_margin_p50,
        gw_upper_margin_p95=gw_upper_margin_p95,
        gw_telescope_min_m=L_min,
        gw_telescope_max_m=L_max,
        gw_L0_m=L0,
        gw_sigma_L_wave_m=float(sigma_L_wave),
        gw_k_sigma_L_wave_m=float(delta_wave),
        gw_k_wave=float(k_wave),
    )


# ---------------------------------------------------------------------------
# Operator-facing plot
# ---------------------------------------------------------------------------


_TRAFFIC_COLOURS = {
    "green": "#2ca02c",
    "amber": "#ff9900",
    "red": "#d62728",
}


def _imca_traffic(value: float, warn_threshold: float, alarm_threshold: float) -> str:
    """Generic IMCA M254 Fig.8-style traffic-light classifier.

    Parameters
    ----------
    value : the metric (e.g. footprint in m, or utilisation as a fraction).
    warn_threshold : amber-onset threshold (e.g. 2.0 m, or 0.60).
    alarm_threshold : red-onset threshold (e.g. 4.0 m, or 0.80).

    Both axes in IMCA M254 Fig.8 use a strict "greater-than" rule:
        green if value < warn,
        amber if warn <= value < alarm,
        red   if value >= alarm.
    """
    if value >= alarm_threshold:
        return "red"
    if value >= warn_threshold:
        return "amber"
    return "green"


def _traffic_light_colour(p_exceed: float) -> str:
    """Probability-based green/amber/red colour (legacy spillover diagnostic).

    Used for the conditional end-stop-hit probability readout, which
    sits ALONGSIDE the IMCA M254 utilisation-based gangway-axis colour
    (computed by ``_imca_traffic``). Thresholds are PLACEHOLDERS chosen
    to give a useful operator signal at the prototype stage.
    """
    if p_exceed < 0.01:
        return _TRAFFIC_COLOURS["green"]
    if p_exceed < 0.10:
        return _TRAFFIC_COLOURS["amber"]
    return _TRAFFIC_COLOURS["red"]


def plot_operator_summary(summary: OperatorSummary, fig=None):
    """Render the two-bar operator-facing figure.

    Two horizontal bars stacked vertically:
      - Top bar: vessel position (warning radius and alarm radius marked).
      - Bottom bar: gangway telescope (lower and upper end-stop marked).

    Each bar shows P50 (light marker) and P95 (heavy marker) of the
    realised peak excursion. The bar is coloured by the highest-severity
    exceedance probability (warning -> amber, alarm -> red).

    The title prints the conditional probabilities in plain language.
    """
    import matplotlib.pyplot as plt

    if fig is None:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7.0),
                                 gridspec_kw=dict(hspace=1.1))
    else:
        axes = fig.subplots(2, 1, gridspec_kw=dict(hspace=1.1))
    # Reserve top margin for the suptitle so it does not overlap the
    # upper-axes left-aligned title.
    fig.subplots_adjust(top=0.82, bottom=0.10, left=0.07, right=0.97)

    # --- Vessel position bar ---
    ax = axes[0]
    bar_max = max(summary.pos_alarm_radius_m * 1.3,
                  summary.pos_p95 * 1.2,
                  summary.pos_alarm_radius_m + 0.5)
    # IMCA M254 Fig.8 vessel-capability colour: footprint = pos_p95 vs
    # warning/alarm radii (2 m / 4 m by default). Driven directly by
    # summary.pos_traffic so the rule is identical to summarise_for_operator.
    bar_colour = _TRAFFIC_COLOURS[summary.pos_traffic]

    ax.barh(0, bar_max, height=0.6, color=bar_colour, alpha=0.25,
            edgecolor=bar_colour, linewidth=2)
    # P50 and P95 markers
    ax.plot(summary.pos_p50, 0, marker="o", markersize=10,
            color="#333333", label=f"P50 peak = {summary.pos_p50:.2f} m",
            markerfacecolor="white", markeredgewidth=2)
    ax.plot(summary.pos_p95, 0, marker="D", markersize=12,
            color="black",
            label=f"P95 peak (footprint) = {summary.pos_p95:.2f} m",
            markerfacecolor=bar_colour, markeredgewidth=2)
    # Limit lines (IMCA M254 Fig.8 thresholds)
    ax.axvline(summary.pos_warning_radius_m, color="#ff9900", ls="--", lw=2,
               label=f"IMCA amber = {summary.pos_warning_radius_m:.1f} m")
    ax.axvline(summary.pos_alarm_radius_m, color="#d62728", ls="--", lw=2,
               label=f"IMCA red = {summary.pos_alarm_radius_m:.1f} m")
    ax.set_xlim(0, bar_max)
    ax.set_yticks([])
    ax.set_xlabel("Vessel footprint (P95 of post-WCF |dp_base|) [m]")
    ax.set_title(
        f"VESSEL CAPABILITY  [{summary.pos_traffic.upper()}]   "
        f"footprint = {summary.pos_p95:.2f} m   "
        f"(IMCA M254 Fig.8: amber > {summary.pos_warning_radius_m:.0f} m, "
        f"red > {summary.pos_alarm_radius_m:.0f} m)\n"
        f"if WCF now: P(footprint > {summary.pos_warning_radius_m:.0f} m) "
        f"= {summary.pos_p_warning*100:.1f} %     "
        f"P(footprint > {summary.pos_alarm_radius_m:.0f} m) "
        f"= {summary.pos_p_alarm*100:.1f} %",
        fontsize=10, loc="left",
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.grid(True, axis="x", alpha=0.3)

    # --- Gangway bar ---
    # Show telescope L on its own axis: L_min ..... L0 ..... L_max,
    # with peak |dL| represented as a "reach to either side" of L0.
    # If a wave-frequency contribution is present, the "envelope" of the
    # combined slow + wave reach is drawn as a translucent band on each
    # side of the slow-content marker.
    ax = axes[1]
    L0 = summary.gw_L0_m
    L_min = summary.gw_telescope_min_m
    L_max = summary.gw_telescope_max_m
    span = L_max - L_min
    if span <= 0:
        span = 2 * max(summary.gw_p95, 1.0)
        L_min = L0 - span / 2
        L_max = L0 + span / 2
    bar_colour_gw = _TRAFFIC_COLOURS[summary.gw_traffic]
    ax.barh(0, L_max - L_min, left=L_min, height=0.6,
            color=bar_colour_gw, alpha=0.25,
            edgecolor=bar_colour_gw, linewidth=2)
    ax.plot(L0, 0, marker="|", markersize=20, color="black",
            markeredgewidth=2, label=f"L_setpoint = {L0:.1f} m")

    delta_wave = summary.gw_k_sigma_L_wave_m
    use_wave = delta_wave > 0.0

    # Slow-content P50 markers
    p50_lo = L0 - summary.gw_p50
    p50_hi = L0 + summary.gw_p50
    ax.plot([p50_lo, p50_hi], [0, 0],
            marker="o", markersize=8, color="#333333",
            markerfacecolor="white", markeredgewidth=2,
            ls="", label=f"P50 slow |dL| = {summary.gw_p50:.2f} m")

    # Slow-content P95 markers
    p95_lo = L0 - summary.gw_p95
    p95_hi = L0 + summary.gw_p95
    ax.plot([p95_lo, p95_hi], [0, 0],
            marker="D", markersize=10, color="black",
            markerfacecolor=bar_colour_gw, markeredgewidth=2,
            ls="", label=f"P95 slow |dL| = {summary.gw_p95:.2f} m")

    # Wave-frequency envelope around the P95 slow markers (combined reach)
    if use_wave:
        for centre in (p95_lo, p95_hi):
            sign = -1 if centre < L0 else +1
            band_lo = centre + 0 if sign > 0 else centre - delta_wave
            band_hi = centre + delta_wave if sign > 0 else centre - 0
            # Span from the slow marker outward by k*sigma_L_wave
            ax.barh(0, abs(delta_wave), left=min(band_lo, band_hi),
                    height=0.35,
                    color="#1f77b4", alpha=0.35, edgecolor="#1f77b4",
                    linewidth=1.5,
                    label=("wave-frequency reach "
                           f"(\u00b1{summary.gw_k_wave:.2f}\u00b7\u03c3_L_wave "
                           f"= \u00b1{delta_wave:.2f} m)"
                           if centre == p95_lo else None))

    # IMCA M254 Fig.8 utilisation thresholds, drawn at L0 +/- warn*stroke
    # and L0 +/- alarm*stroke on each side. These are the "amber" and
    # "red" gangway-capability lines an operator should watch.
    stroke_lo = L0 - L_min
    stroke_hi = L_max - L0
    ax.axvline(L0 - summary.gw_imca_warning_frac * stroke_lo,
               color="#ff9900", ls="--", lw=2.5,
               label=f"IMCA amber ({summary.gw_imca_warning_frac*100:.0f} % util)")
    ax.axvline(L0 + summary.gw_imca_warning_frac * stroke_hi,
               color="#ff9900", ls="--", lw=2.5)
    ax.axvline(L0 - summary.gw_imca_alarm_frac * stroke_lo,
               color="#d62728", ls="--", lw=2.5,
               label=f"IMCA red ({summary.gw_imca_alarm_frac*100:.0f} % util)")
    ax.axvline(L0 + summary.gw_imca_alarm_frac * stroke_hi,
               color="#d62728", ls="--", lw=2.5)
    ax.axvline(L_min, color="#8b0000", ls="-", lw=2.5,
               label=f"L_min = {L_min:.1f} m")
    ax.axvline(L_max, color="#8b0000", ls="-", lw=2.5,
               label=f"L_max = {L_max:.1f} m")
    ax.set_xlim(L_min - 0.5, L_max + 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Telescope length L [m]")
    title_extra = (
        f"   \u03c3_L_wave = {summary.gw_sigma_L_wave_m:.2f} m"
        if use_wave else ""
    )
    ax.set_title(
        f"GANGWAY CAPABILITY  [{summary.gw_traffic.upper()}]   "
        f"util = {summary.gw_util_imca*100:.0f} %   "
        f"(IMCA M254 Fig.8: amber > {summary.gw_imca_warning_frac*100:.0f} %, "
        f"red > {summary.gw_imca_alarm_frac*100:.0f} %){title_extra}\n"
        f"if WCF now: P(end-stop hit, slow+wave) = {summary.gw_p_alarm*100:.1f} %     "
        f"min margin P95 = "
        f"{min(summary.gw_lower_margin_p95, summary.gw_upper_margin_p95):.2f} m",
        fontsize=10, loc="left",
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95, ncol=2)
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        f"WCFDI consequence summary for current operating state\n"
        f"({summary.weather_summary}, N_MC={summary.n_samples}, sample_mode={summary.sample_mode})",
        fontsize=12, y=0.96,
    )
    return fig


# ---------------------------------------------------------------------------
# Intact-prior (no-WCF) extreme-value layer
# ---------------------------------------------------------------------------
#
# This is the "Question 1" answer of the operator framework:
#   "Are we able to maintain position within the limits over the planned
#    operation duration T_op (typically 20-30 min)?"
#
# We compute, for the same two axes (vessel-base footprint, telescope
# length), the probability that the limit is breached AT ANY POINT in
# [0, T_op] under steady DP, with no WCF. The Rice / Cartwright-
# Longuet-Higgins formula gives this in closed form once we have:
#     - the steady-state std dev sigma of each axis,
#     - the zero-up-crossing rate nu_0+ of each axis,
# both derived from the same closed-loop machinery used by P1.
#
# Two separate axes => two separate exceedance probabilities. The
# operator gets two new traffic lights, one per axis, with thresholds
# expressed as P_breach(T_op) thresholds (default amber 1 %, red 10 %).
# These are PRIOR probabilities (model-based). When the online
# estimator is wired in (next step), the same dataclass will be reused
# with sigma replaced by a posterior estimate and nu_0+ kept from the
# prior shape (variance-only Bayesian update).
#
# This panel sits ABOVE the existing IMCA M254 Fig.8 WCFDI bars in the
# operator figure: the first question to answer is whether intact
# operation is acceptable in the chosen window; THEN, conditional on a
# WCF, the post-failure transient is the second answer.

from typing import Callable as _Callable, Sequence as _Sequence
from .gangway import GangwayJointState, telescope_sensitivity


@dataclass
class IntactPriorSummary:
    """Intact-condition extreme-value summary for a planned operation.

    Length-scale presentation (IMCA M254 Fig.8 lifted to quantiles of
    the running max).
    -----------------------------------------------------------------
    For each axis (vessel-base footprint, telescope length deviation)
    the panel reports two quantiles of the running maximum of |X(t)|
    over [0, T_op]:
        a_P50 : the median peak excursion (50% chance the peak exceeds
                this value during T_op).
        a_P90 : the P90 peak excursion (10% chance the peak exceeds
                this value during T_op).
    Both are obtained by inverting the Rice / Cartwright-Longuet-Higgins
    formula (with Vanmarcke clustering correction) from the closed-loop
    spectrum.

    Traffic-light rule (drives `*_traffic_prior`):
        green = a_P90 < warn_radius
        amber = warn_radius <= a_P90 < alarm_radius
        red   = a_P90 >= alarm_radius
    Same semantics as the WCFDI panel: "predicted distance vs IMCA
    radii", just different "predicted distance" sources. The two
    quantiles (P50, P90) are hardcoded for the prototype; an
    operator-selectable quantile is deferred to the C++ implementation.

    Vessel-capability axis:
      pos_sigma_m   : sqrt(sigma_x^2+sigma_y^2) of (dp_base_x, dp_base_y) [m].
      pos_nu0_max   : max(nu_0+) across the two horizontal axes [Hz].
      pos_q         : max(q) across axes (Vanmarcke spectral bandwidth).
      pos_a_p50, pos_a_p90 : P50 / P90 peak |dp_base| over T_op [m].
      pos_p_breach  : P(footprint > pos_warning_radius_m) over T_op
                      (kept as diagnostic).
      pos_p_breach_alarm : P(footprint > pos_alarm_radius_m) over T_op.
      pos_traffic_prior : "green"|"amber"|"red" by a_p90 vs warn/alarm.

    Gangway-capability axis (telescope length):
      gw_sigma_slow_m, gw_nu0_slow, gw_q_slow : DP slow band.
      gw_sigma_wave_m, gw_nu0_wave, gw_q_wave : 1st-order wave band.
      gw_threshold_to_lower_m, gw_threshold_to_upper_m : per-side strokes.
      gw_threshold_used_m : worst-side stroke (min of the two).
      gw_warn_frac, gw_alarm_frac : fractions of stroke for warn/alarm.
      gw_warn_m, gw_alarm_m       : the same as absolute distances.
      gw_a_p50, gw_a_p90 : P50 / P90 peak |dL| over T_op [m].
      gw_p_breach_warn  : P(|dL| > gw_warn_m) over T_op (slow+wave bands).
      gw_p_breach_alarm : P(|dL| > gw_alarm_m) over T_op.
      gw_p_breach_per_band : per-band attribution at the warn threshold.
      gw_traffic_prior  : "green"|"amber"|"red" by a_p90 vs warn/alarm.

    Operation context:
      T_op_s    : exposure duration used [s].
      quantiles : tuple of the two quantiles used (default (0.50, 0.90)).
    """

    T_op_s: float
    quantiles: tuple[float, float]

    pos_sigma_m: float
    pos_nu0_max: float
    pos_q: float
    pos_a_p50: float
    pos_a_p90: float
    pos_p_breach: float
    pos_p_breach_alarm: float
    pos_traffic_prior: str
    pos_warning_radius_m: float
    pos_alarm_radius_m: float

    gw_sigma_slow_m: float
    gw_nu0_slow: float
    gw_q_slow: float
    gw_sigma_wave_m: float
    gw_nu0_wave: float
    gw_q_wave: float
    gw_threshold_to_lower_m: float
    gw_threshold_to_upper_m: float
    gw_threshold_used_m: float
    gw_warn_frac: float
    gw_alarm_frac: float
    gw_warn_m: float
    gw_alarm_m: float
    gw_a_p50: float
    gw_a_p90: float
    gw_p_breach_warn: float
    gw_p_breach_alarm: float
    gw_p_breach_per_band: tuple[float, float]
    gw_traffic_prior: str


def _imca_traffic_prior(a_p90: float, warn_radius: float,
                        alarm_radius: float) -> str:
    """Intact-prior IMCA M254 Fig.8-style traffic light, distance-driven.

    Driven by the P90 quantile of the running maximum of |X(t)| over
    [0, T_op] (a length, in metres) against the two IMCA radii:

      green = a_p90 < warn_radius
      amber = warn_radius <= a_p90 < alarm_radius
      red   = a_p90 >= alarm_radius

    Identical semantics to the WCFDI panel rule (which uses the P95
    of the post-failure peak): "predicted distance vs IMCA radii".
    """
    if a_p90 >= alarm_radius:
        return "red"
    if a_p90 >= warn_radius:
        return "amber"
    return "green"


def summarise_intact_prior(
    cl,
    S_F_funcs: list,
    cfg: CqaConfig,
    joint: GangwayJointState,
    T_op_s: float = 20.0 * 60.0,
    sigma_L_wave: float = 0.0,
    Tp_wave_s: float = 8.0,
    quantiles: tuple[float, float] = (0.50, 0.90),
    gw_warn_frac: float = 0.60,
    gw_alarm_frac: float = 0.80,
    omega_grid: Optional[np.ndarray] = None,
    posterior_sigma_radial_m: Optional[float] = None,
    posterior_sigma_telescope_slow_m: Optional[float] = None,
) -> IntactPriorSummary:
    """Build the intact-condition extreme-value summary for one operation.

    Length-scale presentation
    -------------------------
    For each axis we report two quantiles (default P50, P90) of the
    running maximum of |X(t)| over [0, T_op], inverted from the Rice
    formula with Vanmarcke clustering correction. The traffic light
    is driven by the P90 quantile vs the IMCA radii:

      green = a_p90 < warn_radius
      amber = warn_radius <= a_p90 < alarm_radius
      red   = a_p90 >= alarm_radius

    Same rule as the WCFDI panel (which uses P95 of the post-failure
    peak), so the two panels read with identical semantics: "predicted
    distance vs IMCA radii", just different sources.

    Vessel: warn_radius = cfg.operational_limits.position_warning_radius_m
            alarm_radius = cfg.operational_limits.position_alarm_radius_m
            (defaults 2 m / 4 m, IMCA M254).

    Gangway: warn = gw_warn_frac * worst_side_stroke
             alarm = gw_alarm_frac * worst_side_stroke
             (defaults 60% / 80%, IMCA M254 utilisation).

    Parameters
    ----------
    cl : ClosedLoop. Built from vessel + controller.
    S_F_funcs : list of disturbance force PSD callables.
    cfg : full CqaConfig.
    joint : current gangway joint state.
    T_op_s : planned operation duration [s]. Default 20 min.
    sigma_L_wave : 1-sigma wave-frequency telescope deviation [m].
    Tp_wave_s : wave peak period [s].
    quantiles : (lower, upper) pair of probabilities in (0,1) used to
        report the two markers. Default (0.50, 0.90). Operator-
        selectable quantile is deferred to the C++ implementation.
    gw_warn_frac, gw_alarm_frac : gangway warn / alarm thresholds as
        fraction of worst-side stroke. Defaults 0.60 / 0.80.
    omega_grid : optional integration grid [rad/s].
    posterior_sigma_radial_m : optional float. If provided, overrides
        the model-based ``sigma_radial`` in the position-axis Rice /
        inverse-Rice calls with a data-conditioned posterior sigma
        (e.g. from `BayesianSigmaEstimator.posterior().sigma_median`).
        The spectral SHAPE (nu_0+, Vanmarcke q) is kept from the
        prior closed-loop spectrum: only the LEVEL is updated.
    posterior_sigma_telescope_slow_m : optional float. Same idea for
        the slow-drift telescope band. The wave-frequency band stays
        as prescribed by ``sigma_L_wave`` (driven by the wave RAO,
        not the DP closed loop).

    Returns
    -------
    IntactPriorSummary.
    """
    from .closed_loop import axis_psd
    from .extreme_value import (
        spectral_moments,
        zero_upcrossing_rate,
        vanmarcke_bandwidth_q,
        p_exceed_rice,
        p_exceed_rice_multiband,
        inverse_rice,
        inverse_rice_multiband,
    )

    if omega_grid is None:
        omega_grid = np.logspace(-4, 0, 1024)
    omega_grid = np.asarray(omega_grid, dtype=float)

    p_lo, p_hi = float(quantiles[0]), float(quantiles[1])
    if not (0.0 < p_lo < 1.0) or not (0.0 < p_hi < 1.0):
        raise ValueError(f"quantiles must be in (0,1), got {quantiles}")
    if p_lo >= p_hi:
        raise ValueError(f"quantiles must be ordered (lo<hi), got {quantiles}")

    # --- Vessel base-position axis ---
    base_x_b, base_y_b, _ = cfg.gangway.base_position_body
    c_base_x = np.array([1.0, 0.0, -base_y_b])
    c_base_y = np.array([0.0, 1.0,  base_x_b])

    S_base_x = axis_psd(cl, S_F_funcs, c_base_x, omega_grid)
    S_base_y = axis_psd(cl, S_F_funcs, c_base_y, omega_grid)
    sigma_x = float(np.sqrt(spectral_moments(S_base_x, omega_grid, [0])[0]))
    sigma_y = float(np.sqrt(spectral_moments(S_base_y, omega_grid, [0])[0]))
    nu0_x = zero_upcrossing_rate(S_base_x, omega_grid)
    nu0_y = zero_upcrossing_rate(S_base_y, omega_grid)
    q_x = vanmarcke_bandwidth_q(S_base_x, omega_grid)
    q_y = vanmarcke_bandwidth_q(S_base_y, omega_grid)

    sigma_radial = float(np.sqrt(sigma_x ** 2 + sigma_y ** 2))
    nu0_pos = float(max(nu0_x, nu0_y))
    q_pos = float(max(q_x, q_y))

    # Posterior override: keep nu_0+ and q from the model spectrum
    # ("shape"), substitute the data-conditioned variance ("level").
    sigma_radial_prior = sigma_radial
    if posterior_sigma_radial_m is not None:
        if posterior_sigma_radial_m <= 0.0:
            raise ValueError(
                f"posterior_sigma_radial_m must be > 0, "
                f"got {posterior_sigma_radial_m}"
            )
        sigma_radial = float(posterior_sigma_radial_m)

    limits = cfg.operational_limits
    a_warn = float(limits.position_warning_radius_m)
    a_alarm = float(limits.position_alarm_radius_m)

    # P50 / P90 of running max of |position| over T_op (inverse Rice).
    # Note inverse_rice's "p" is P_breach = P(running max > a). So the
    # P90 quantile of the running max corresponds to P_breach = 0.10.
    pos_a_p50 = inverse_rice(
        p=1.0 - p_lo, sigma=sigma_radial, nu_0_plus=nu0_pos,
        T=T_op_s, bilateral=True, clustering="vanmarcke", q=q_pos,
    )
    pos_a_p90 = inverse_rice(
        p=1.0 - p_hi, sigma=sigma_radial, nu_0_plus=nu0_pos,
        T=T_op_s, bilateral=True, clustering="vanmarcke", q=q_pos,
    )

    # Diagnostic P_breach at the IMCA radii (kept for completeness).
    res_pos_warn = p_exceed_rice(
        sigma=sigma_radial, nu_0_plus=nu0_pos,
        threshold=a_warn, T=T_op_s, bilateral=True,
        clustering="vanmarcke", q=q_pos,
    )
    res_pos_alarm = p_exceed_rice(
        sigma=sigma_radial, nu_0_plus=nu0_pos,
        threshold=a_alarm, T=T_op_s, bilateral=True,
        clustering="vanmarcke", q=q_pos,
    )
    pos_traffic_prior = _imca_traffic_prior(pos_a_p90, a_warn, a_alarm)

    # --- Gangway telescope axis ---
    c_L = telescope_sensitivity(joint, cfg.gangway)
    S_dL_slow = axis_psd(cl, S_F_funcs, c_L, omega_grid)
    sigma_slow = float(np.sqrt(spectral_moments(S_dL_slow, omega_grid, [0])[0]))
    nu0_slow = zero_upcrossing_rate(S_dL_slow, omega_grid)
    q_slow = vanmarcke_bandwidth_q(S_dL_slow, omega_grid)

    sigma_slow_prior = sigma_slow
    if posterior_sigma_telescope_slow_m is not None:
        if posterior_sigma_telescope_slow_m <= 0.0:
            raise ValueError(
                f"posterior_sigma_telescope_slow_m must be > 0, "
                f"got {posterior_sigma_telescope_slow_m}"
            )
        sigma_slow = float(posterior_sigma_telescope_slow_m)

    sigma_wave = float(sigma_L_wave)
    nu0_wave = float(1.0 / Tp_wave_s) if (Tp_wave_s > 0 and sigma_wave > 0) else 0.0
    q_wave = 0.3  # JONSWAP-typical narrowband bandwidth proxy

    L0 = float(joint.L)
    L_min = float(cfg.gangway.telescope_min)
    L_max = float(cfg.gangway.telescope_max)
    stroke_lo = max(L0 - L_min, 0.0)
    stroke_hi = max(L_max - L0, 0.0)
    stroke = min(stroke_lo, stroke_hi)
    gw_warn_m = float(gw_warn_frac) * stroke
    gw_alarm_m = float(gw_alarm_frac) * stroke

    # Multi-band combination, evaluated at BOTH the warn and alarm
    # gangway thresholds (diagnostic).
    bands = [(sigma_slow, nu0_slow, q_slow)]
    if sigma_wave > 0.0 and nu0_wave > 0.0:
        bands.append((sigma_wave, nu0_wave, q_wave))
    mb_warn = p_exceed_rice_multiband(
        bands=bands, threshold=gw_warn_m, T=T_op_s, bilateral=True,
        clustering="vanmarcke",
    )
    mb_alarm = p_exceed_rice_multiband(
        bands=bands, threshold=gw_alarm_m, T=T_op_s, bilateral=True,
        clustering="vanmarcke",
    )
    gw_p_warn = float(mb_warn["p_breach"])
    gw_p_alarm = float(mb_alarm["p_breach"])
    per_band_warn = mb_warn["p_breach_per_band"]
    if len(per_band_warn) == 1:
        per_band_tuple = (float(per_band_warn[0]), 0.0)
    else:
        per_band_tuple = (float(per_band_warn[0]), float(per_band_warn[1]))

    # P50 / P90 of running max of |dL| over T_op (multi-band inverse).
    gw_a_p50 = inverse_rice_multiband(
        p=1.0 - p_lo, bands=bands, T=T_op_s,
        bilateral=True, clustering="vanmarcke",
    )
    gw_a_p90 = inverse_rice_multiband(
        p=1.0 - p_hi, bands=bands, T=T_op_s,
        bilateral=True, clustering="vanmarcke",
    )
    gw_traffic_prior = _imca_traffic_prior(gw_a_p90, gw_warn_m, gw_alarm_m)

    return IntactPriorSummary(
        T_op_s=float(T_op_s),
        quantiles=(p_lo, p_hi),
        pos_sigma_m=sigma_radial,
        pos_nu0_max=nu0_pos,
        pos_q=q_pos,
        pos_a_p50=float(pos_a_p50),
        pos_a_p90=float(pos_a_p90),
        pos_p_breach=float(res_pos_warn.p_breach),
        pos_p_breach_alarm=float(res_pos_alarm.p_breach),
        pos_traffic_prior=pos_traffic_prior,
        pos_warning_radius_m=a_warn,
        pos_alarm_radius_m=a_alarm,
        gw_sigma_slow_m=sigma_slow,
        gw_nu0_slow=nu0_slow,
        gw_q_slow=q_slow,
        gw_sigma_wave_m=sigma_wave,
        gw_nu0_wave=nu0_wave,
        gw_q_wave=q_wave if (sigma_wave > 0.0 and nu0_wave > 0.0) else 1.0,
        gw_threshold_to_lower_m=stroke_lo,
        gw_threshold_to_upper_m=stroke_hi,
        gw_threshold_used_m=float(stroke),
        gw_warn_frac=float(gw_warn_frac),
        gw_alarm_frac=float(gw_alarm_frac),
        gw_warn_m=float(gw_warn_m),
        gw_alarm_m=float(gw_alarm_m),
        gw_a_p50=float(gw_a_p50),
        gw_a_p90=float(gw_a_p90),
        gw_p_breach_warn=gw_p_warn,
        gw_p_breach_alarm=gw_p_alarm,
        gw_p_breach_per_band=per_band_tuple,
        gw_traffic_prior=gw_traffic_prior,
    )


def plot_intact_prior(prior: IntactPriorSummary, fig=None):
    """Render the intact-prior length-scale panel.

    Two horizontal distance bars: one for the vessel-base footprint,
    one for the telescope deviation. Each bar is drawn as a coloured
    background showing the IMCA bands (green / amber / red) along the
    distance axis, with two marker dots at the P50 and P90 quantiles
    of the running max of |X(t)| over T_op. The bar's title gives the
    two quantile values in metres and the IMCA traffic colour driven
    by P90 vs the alarm / warn radii.
    """
    import matplotlib.pyplot as plt

    if fig is None:
        fig, axes = plt.subplots(2, 1, figsize=(11, 4.0),
                                 gridspec_kw=dict(hspace=1.4))
    else:
        axes = fig.subplots(2, 1, gridspec_kw=dict(hspace=1.4))
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.07, right=0.97)

    T_min = prior.T_op_s / 60.0
    p_lo, p_hi = prior.quantiles
    p_lo_pct = p_lo * 100.0
    p_hi_pct = p_hi * 100.0

    def _bar(ax, a_p50, a_p90, warn_m, alarm_m, label, traffic, extra_text=""):
        x_max = max(1.2 * alarm_m, a_p90 * 1.2, alarm_m + 0.5)
        # Coloured IMCA bands as background.
        ax.barh(0, warn_m, left=0.0, height=0.55,
                color=_TRAFFIC_COLOURS["green"], alpha=0.18,
                edgecolor=_TRAFFIC_COLOURS["green"], linewidth=1.2)
        ax.barh(0, alarm_m - warn_m, left=warn_m, height=0.55,
                color=_TRAFFIC_COLOURS["amber"], alpha=0.18,
                edgecolor=_TRAFFIC_COLOURS["amber"], linewidth=1.2)
        ax.barh(0, x_max - alarm_m, left=alarm_m, height=0.55,
                color=_TRAFFIC_COLOURS["red"], alpha=0.18,
                edgecolor=_TRAFFIC_COLOURS["red"], linewidth=1.2)
        # IMCA radius lines.
        ax.axvline(warn_m, color="#ff9900", ls="--", lw=2,
                   label=f"IMCA amber = {warn_m:.2f} m")
        ax.axvline(alarm_m, color="#d62728", ls="--", lw=2,
                   label=f"IMCA red = {alarm_m:.2f} m")
        # P50 (light) and P90 (heavy) markers.
        ax.plot([a_p50], [0.0], marker="o", markersize=11, color="#333333",
                markerfacecolor="white", markeredgewidth=2,
                label=f"P{p_lo_pct:.0f} peak = {a_p50:.2f} m",
                zorder=4, ls="")
        bar_colour = _TRAFFIC_COLOURS[traffic]
        ax.plot([a_p90], [0.0], marker="D", markersize=13, color="black",
                markerfacecolor=bar_colour, markeredgewidth=2,
                label=f"P{p_hi_pct:.0f} peak = {a_p90:.2f} m",
                zorder=4, ls="")
        ax.set_xlim(0, x_max)
        ax.set_ylim(-0.4, 0.4)
        ax.set_yticks([])
        ax.set_xlabel("Predicted peak |excursion| over T_op [m]")
        ax.set_title(
            f"{label}  [{traffic.upper()}]   "
            f"P{p_lo_pct:.0f}={a_p50:.2f} m, P{p_hi_pct:.0f}={a_p90:.2f} m"
            f"{extra_text}",
            fontsize=10, loc="left",
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.95, ncol=1)
        ax.grid(True, axis="x", alpha=0.3)

    _bar(
        axes[0],
        prior.pos_a_p50, prior.pos_a_p90,
        prior.pos_warning_radius_m, prior.pos_alarm_radius_m,
        "VESSEL CAPABILITY (intact, footprint at gangway base)",
        prior.pos_traffic_prior,
        extra_text=(f"   \u03c3_radial = {prior.pos_sigma_m:.2f} m   "
                    f"\u03bd\u2080\u207a = {prior.pos_nu0_max*1000:.1f} mHz"),
    )
    extra_gw = (
        f"   \u03c3_slow={prior.gw_sigma_slow_m:.2f} m"
        + (f"   \u03c3_wave={prior.gw_sigma_wave_m:.2f} m" if prior.gw_sigma_wave_m > 0 else "")
        + f"   stroke={prior.gw_threshold_used_m:.1f} m"
    )
    _bar(
        axes[1],
        prior.gw_a_p50, prior.gw_a_p90,
        prior.gw_warn_m, prior.gw_alarm_m,
        "GANGWAY CAPABILITY (intact, telescope vs end-stop)",
        prior.gw_traffic_prior,
        extra_text=extra_gw,
    )

    fig.suptitle(
        f"Intact-condition prior for the next {T_min:.0f} min "
        f"(no WCF, model-based, stationary Gaussian; markers = quantiles of running max)",
        fontsize=12, y=0.95,
    )
    return fig
