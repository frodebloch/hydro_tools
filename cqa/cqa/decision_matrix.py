"""Forecast-case WCFDI decision matrix (roadmap item 4b).

Per forecast time-slot and chosen vessel heading, evaluate the same
intact-prior and post-WCFDI metrics that the operability polar
(``operability_polar`` / ``wcfdi_operability_overlay``) computes
against a *swept* Pierson-Moskowitz environment, but instead at the
*forecast* sea state ``(V_w, H_s, T_p, V_c)``. The output is a
``WcfdiDecisionMatrix``: a ``(slots, headings)`` grid of per-cell
traffic lights (intact, WCFDI, overall) plus the underlying numerical
metrics for inspection.

Audience and scope
------------------
This is an **operationally-facing** workstream. Where the operability
polar is a design / table-top tool that sweeps a synthetic environment
to delineate the feasible region, the decision matrix consumes a
*specific* forecast (a planned operation window) and answers the
operator-facing question per heading: "for the forecast at this slot,
what is the predicted footprint and the consequence of a WCFDI now?"

Same engines are reused: ``summarise_intact_prior`` (intact P90 of the
running maximum -> traffic light against the IMCA M254 radii) and
``wcfdi_transient`` (post-failure mean + covariance envelope ->
traffic light against the same radii) plus the bistability gate from
``wcfdi_transient.info["bistability_risk_score"]``.

Direction model
---------------
v1 honours the polar's collinear convention: per slot a single
``theta_env_compass`` carries wind, wave and current. Realistic for
wind-driven North-Sea seas where the three are usually co-aligned.
The evaluator computes ``theta_rel = theta_env_compass - heading_compass``
(positive into the vessel) and feeds it to the PSD assemblers and the
WCFDI transient. Independent wind / wave / current directions are a
deferred extension; see ``analysis.md`` §12.15.

Heading convention
------------------
``heading_compass`` is the vessel's compass heading (the bow direction).
``theta_env_compass`` is the meteorological "from" direction (the
direction the wind / wave / current is coming from). Therefore the
relative direction *into the vessel* is

    theta_rel = wrap_to_minus_pi_pi(theta_env_compass - heading_compass)

with theta_rel = 0 -> head-on, pi/2 -> beam from port (right-hand rule
about the vertical axis), as elsewhere in cqa.

Traffic-light combination
-------------------------
``overall`` per cell is the worst (red > amber > green) of:

    intact_traffic, wcfdi_traffic

This matches the IMCA M254 Fig.8 "decision matrix" semantics: any axis
in red flips the cell to red; any axis in amber and none in red flips
to amber.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import CqaConfig
from .gangway import GangwayJointState, telescope_sensitivity
from .transient import WcfdiScenario, wcfdi_transient


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastSlot:
    """One forecast time-slot.

    Attributes
    ----------
    label : opaque identifier (e.g. ISO8601 timestamp). The engine does
        not parse it; the demo plot uses it as the column tick label.
    Vw : 10 m mean wind speed [m/s].
    Hs : significant wave height [m].
    Tp : peak wave period [s].
    Vc : current speed [m/s].
    theta_env_compass : meteorological "from" direction of the
        co-aligned wind/wave/current [rad], compass-frame
        (0 = from north). Operator UI is responsible for any deg-rad
        and N-vs-E conversions before constructing the slot.
    """

    label: str
    Vw: float
    Hs: float
    Tp: float
    Vc: float
    theta_env_compass: float


@dataclass(frozen=True)
class DecisionCell:
    """One per-cell entry in the decision matrix."""

    slot_index: int
    heading_index: int
    heading_compass: float
    theta_rel: float

    # Intact axis
    intact_pos_a_p90_m: float
    intact_gw_a_p90_m: float
    intact_pos_traffic: str
    intact_gw_traffic: str
    intact_traffic: str

    # WCFDI axis
    wcfdi_pos_peak_m: float
    wcfdi_gw_peak_m: float
    wcfdi_pos_traffic: str
    wcfdi_gw_traffic: str
    wcfdi_bistability_score: float
    wcfdi_cqa_violated: bool
    wcfdi_traffic: str

    # Combined
    overall_traffic: str


@dataclass(frozen=True)
class WcfdiDecisionMatrix:
    """Forecast-case decision matrix: (n_slots x n_headings) of cells."""

    slots: tuple
    headings_compass: np.ndarray
    cells: tuple
    pos_warn_radius_m: float
    pos_alarm_radius_m: float
    gw_warn_m: float
    gw_alarm_m: float
    bistability_alarm: float
    k_sigma: float
    t_end_wcfdi_s: float
    scenario_alpha: tuple
    scenario_T_realloc: float

    def cell(self, slot_index: int, heading_index: int) -> DecisionCell:
        n_h = self.headings_compass.size
        return self.cells[slot_index * n_h + heading_index]

    def overall_grid(self) -> np.ndarray:
        """Return an ``(n_slots, n_headings)`` array of overall-traffic
        strings (``"green"|"amber"|"red"``)."""
        n_h = self.headings_compass.size
        n_s = len(self.slots)
        out = np.empty((n_s, n_h), dtype=object)
        for s in range(n_s):
            for h in range(n_h):
                out[s, h] = self.cells[s * n_h + h].overall_traffic
        return out

    def intact_grid(self) -> np.ndarray:
        n_h = self.headings_compass.size
        n_s = len(self.slots)
        out = np.empty((n_s, n_h), dtype=object)
        for s in range(n_s):
            for h in range(n_h):
                out[s, h] = self.cells[s * n_h + h].intact_traffic
        return out

    def wcfdi_grid(self) -> np.ndarray:
        n_h = self.headings_compass.size
        n_s = len(self.slots)
        out = np.empty((n_s, n_h), dtype=object)
        for s in range(n_s):
            for h in range(n_h):
                out[s, h] = self.cells[s * n_h + h].wcfdi_traffic
        return out


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _wrap_pi(x: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return float(np.mod(x + np.pi, 2.0 * np.pi) - np.pi)


def _worst(*labels: str) -> str:
    """Worst of a set of green/amber/red labels."""
    order = {"green": 0, "amber": 1, "red": 2}
    return max(labels, key=lambda s: order[s])


def _imca_traffic(value_m: float, warn_m: float, alarm_m: float) -> str:
    if not np.isfinite(value_m) or value_m >= alarm_m:
        return "red"
    if value_m >= warn_m:
        return "amber"
    return "green"


# ---------------------------------------------------------------------------
# Per-slot intact and WCFDI evaluation
# ---------------------------------------------------------------------------


def _build_intact_prior_at_forecast(
    cfg: CqaConfig,
    joint: GangwayJointState,
    Vw: float,
    Hs: float,
    Tp: float,
    Vc: float,
    theta_rel: float,
    *,
    rao_table=None,
    sigma_Vc: float,
    tau_Vc: float,
    T_op_s: float,
    quantile_p: float,
    omega_grid: Optional[np.ndarray],
    use_pm_for_drift: bool,
):
    """Forecast-Hs/Tp counterpart of operability_polar._evaluate_intact_prior_at.

    Same closed-loop / PSD assembly, but Hs and Tp are taken from the
    forecast slot rather than derived from V_w via Pierson-Moskowitz.
    """
    from .vessel import LinearVesselModel, CurrentForceModel
    from .controller import LinearDpController
    from .closed_loop import ClosedLoop
    from .psd import (
        npd_wind_gust_force_psd,
        slow_drift_force_psd_newman,
        current_variability_force_psd,
        WindForceModel,
    )
    from .operator_view import summarise_intact_prior
    from .wave_response import sigma_L_wave as _sigma_L_wave_fn
    from .drift import slow_drift_force_psd_newman_pdstrip

    vp = cfg.vessel
    wp = cfg.wind
    cp = cfg.current
    wd = cfg.wave_drift
    cp_ctrl = cfg.controller

    vessel = LinearVesselModel.from_config(vp)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D, omega_n=cp_ctrl.omega_n, zeta=cp_ctrl.zeta,
    )
    cl = ClosedLoop.build(vessel, controller)

    wind_model = WindForceModel(wp=wp, loa=vp.loa)
    S_wind = npd_wind_gust_force_psd(wind_model, Vw, theta_rel)

    if rao_table is not None:
        S_drift = slow_drift_force_psd_newman_pdstrip(
            rao_table=rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel,
        )
    elif use_pm_for_drift:
        S_drift = slow_drift_force_psd_newman(
            (wd.drift_x_amp, wd.drift_y_amp, wd.drift_n_amp),
            Hs, Tp, theta_rel,
        )
    else:
        def S_drift(_w):
            return np.zeros((3, 3))

    current_model = CurrentForceModel(
        cp=cp,
        lateral_area_underwater=vp.lpp * vp.draft,
        frontal_area_underwater=vp.beam * vp.draft,
        loa=vp.loa,
    )
    if Vc > 1e-9:
        F0 = current_model.force(Vc, theta_rel)
        dFdVc = 2.0 * F0 / Vc
    else:
        dFdVc = np.zeros(3)
    S_curr = current_variability_force_psd(dFdVc, sigma_Vc=sigma_Vc, tau=tau_Vc)

    if rao_table is not None and Tp > 0 and Hs > 0:
        wave = _sigma_L_wave_fn(joint, cfg, rao_table, Hs=Hs, Tp=Tp,
                                theta_wave_rel=theta_rel)
        sigma_L_wave_m = wave.sigma_L_wave
    else:
        sigma_L_wave_m = 0.0

    return summarise_intact_prior(
        cl, [S_wind, S_drift, S_curr], cfg, joint,
        T_op_s=T_op_s,
        sigma_L_wave=sigma_L_wave_m,
        Tp_wave_s=Tp if Tp > 0 else 8.0,
        quantiles=(0.50, quantile_p),
        omega_grid=omega_grid,
    )


def _wcfdi_peak_at_forecast(
    cfg: CqaConfig,
    joint: GangwayJointState,
    Vw: float,
    Hs: float,
    Tp: float,
    Vc: float,
    theta_rel: float,
    *,
    scenario: WcfdiScenario,
    k_sigma: float,
    t_end: float,
    sigma_Vc: float,
    tau_Vc: float,
    c_L: np.ndarray,
):
    """Forecast-Hs/Tp counterpart of operability_polar._wcfdi_peak_metrics.

    Returns ``(peak_pos_m, peak_dL_m, bistability_score, cqa_violated)``.

    ``cqa_violated`` is True if either (a) the underlying linear ODE
    raises (rare; happens only when the solver gives up) OR (b) the
    transient reports any per-DOF CQA precondition violation in
    ``info["cqa_precondition_violated"]`` (the routine, robust signal:
    post-failure thrust insufficient to hold the steady-state
    environmental load in at least one DOF). On either flavour we
    return ``(inf, inf)`` for the peaks so the caller's IMCA traffic
    rule yields red.
    """
    try:
        res = wcfdi_transient(
            cfg,
            Vw_mean=Vw, Hs=Hs, Tp=Tp, Vc=Vc, theta_rel=theta_rel,
            scenario=scenario,
            sigma_Vc=sigma_Vc, tau_Vc=tau_Vc, t_end=t_end,
        )
    except Exception:
        return float("inf"), float("inf"), 0.0, True

    cqa_viol_arr = res.info.get("cqa_precondition_violated", None)
    if cqa_viol_arr is not None and bool(np.any(cqa_viol_arr)):
        return (
            float("inf"), float("inf"),
            float(res.info.get("bistability_risk_score", 0.0)),
            True,
        )

    eta = res.eta_mean
    P_eta = res.P[:, 0:3, 0:3]
    pos_mean_r = np.sqrt(eta[:, 0] ** 2 + eta[:, 1] ** 2)
    sigma_R_t = np.sqrt(np.maximum(P_eta[:, 0, 0] + P_eta[:, 1, 1], 0.0))
    pos_envelope = pos_mean_r + k_sigma * sigma_R_t

    dL_mean = eta @ c_L
    sigma_dL = np.sqrt(np.maximum(
        np.einsum("i,nij,j->n", c_L, P_eta, c_L), 0.0,
    ))
    dL_envelope = np.abs(dL_mean) + k_sigma * sigma_dL

    return (
        float(np.max(pos_envelope)),
        float(np.max(dL_envelope)),
        float(res.info.get("bistability_risk_score", 0.0)),
        False,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_decision_cell(
    cfg: CqaConfig,
    joint: GangwayJointState,
    slot: ForecastSlot,
    heading_compass: float,
    *,
    slot_index: int = 0,
    heading_index: int = 0,
    scenario: Optional[WcfdiScenario] = None,
    rao_table=None,
    sigma_Vc: float = 0.1,
    tau_Vc: float = 600.0,
    T_op_s: float = 20.0 * 60.0,
    quantile_p: float = 0.90,
    omega_grid: Optional[np.ndarray] = None,
    use_pm_for_drift: bool = True,
    k_sigma: float = 0.674,
    t_end_wcfdi: float = 200.0,
    bistability_alarm: float = 1.5,
) -> DecisionCell:
    """Evaluate the decision-matrix cell at one (slot, heading) pair.

    Parameters
    ----------
    cfg, joint : same as the operability polar.
    slot : ForecastSlot.
    heading_compass : vessel bow direction [rad].
    slot_index, heading_index : recorded on the cell for caller
        bookkeeping when this function is used standalone.
    scenario : WCFDI scenario. Default = single thruster group lost
        (alpha = 2/3, gamma_immediate = 0.5, T_realloc = 10 s).
    All other kwargs match the operability polar defaults so the
    decision-matrix cell at the forecast point is directly comparable
    to the polar at the corresponding swept point.

    Returns
    -------
    DecisionCell.
    """
    if scenario is None:
        scenario = WcfdiScenario(
            alpha=(2.0 / 3.0,) * 3,
            gamma_immediate=0.5,
            T_realloc=10.0,
        )

    theta_rel = _wrap_pi(slot.theta_env_compass - heading_compass)

    # ---- Intact axis ----
    prior = _build_intact_prior_at_forecast(
        cfg, joint,
        Vw=slot.Vw, Hs=slot.Hs, Tp=slot.Tp, Vc=slot.Vc,
        theta_rel=theta_rel,
        rao_table=rao_table,
        sigma_Vc=sigma_Vc, tau_Vc=tau_Vc,
        T_op_s=T_op_s, quantile_p=quantile_p,
        omega_grid=omega_grid,
        use_pm_for_drift=use_pm_for_drift,
    )
    intact_pos_traffic = prior.pos_traffic_prior
    intact_gw_traffic = prior.gw_traffic_prior
    intact_traffic = _worst(intact_pos_traffic, intact_gw_traffic)

    # ---- WCFDI axis ----
    pos_warn_r = float(cfg.operational_limits.position_warning_radius_m)
    pos_alarm_r = float(cfg.operational_limits.position_alarm_radius_m)
    L0 = float(joint.L)
    L_min = float(cfg.gangway.telescope_min)
    L_max = float(cfg.gangway.telescope_max)
    stroke = min(max(L0 - L_min, 0.0), max(L_max - L0, 0.0))
    gw_warn_m = 0.60 * stroke
    gw_alarm_m = 0.80 * stroke

    c_L = telescope_sensitivity(joint, cfg.gangway)

    pos_peak, dL_peak, bist_score, cqa_violated = _wcfdi_peak_at_forecast(
        cfg, joint,
        Vw=slot.Vw, Hs=slot.Hs, Tp=slot.Tp, Vc=slot.Vc,
        theta_rel=theta_rel,
        scenario=scenario, k_sigma=k_sigma, t_end=t_end_wcfdi,
        sigma_Vc=sigma_Vc, tau_Vc=tau_Vc, c_L=c_L,
    )
    if bist_score > bistability_alarm and not cqa_violated:
        # Bistability gate: deterministic predictor in the meta-stable
        # regime; treat both axes as alarm. See analysis.md §12.14.
        wcfdi_pos_traffic = "red"
        wcfdi_gw_traffic = "red" if stroke > 0.0 else "green"
    else:
        wcfdi_pos_traffic = _imca_traffic(pos_peak, pos_warn_r, pos_alarm_r)
        if stroke > 0.0:
            wcfdi_gw_traffic = _imca_traffic(dL_peak, gw_warn_m, gw_alarm_m)
        else:
            # Telescope at an end-stop: no admissible slack on this side.
            # The WCFDI peak in dL is meaningless; degrade gracefully.
            wcfdi_gw_traffic = "green"

    wcfdi_traffic = _worst(wcfdi_pos_traffic, wcfdi_gw_traffic)

    overall_traffic = _worst(intact_traffic, wcfdi_traffic)

    return DecisionCell(
        slot_index=slot_index,
        heading_index=heading_index,
        heading_compass=float(heading_compass),
        theta_rel=float(theta_rel),
        intact_pos_a_p90_m=float(prior.pos_a_p90),
        intact_gw_a_p90_m=float(prior.gw_a_p90),
        intact_pos_traffic=intact_pos_traffic,
        intact_gw_traffic=intact_gw_traffic,
        intact_traffic=intact_traffic,
        wcfdi_pos_peak_m=float(pos_peak),
        wcfdi_gw_peak_m=float(dL_peak),
        wcfdi_pos_traffic=wcfdi_pos_traffic,
        wcfdi_gw_traffic=wcfdi_gw_traffic,
        wcfdi_bistability_score=float(bist_score),
        wcfdi_cqa_violated=bool(cqa_violated),
        wcfdi_traffic=wcfdi_traffic,
        overall_traffic=overall_traffic,
    )


def wcfdi_decision_matrix(
    cfg: CqaConfig,
    joint: GangwayJointState,
    slots: list,
    headings_compass: np.ndarray,
    *,
    scenario: Optional[WcfdiScenario] = None,
    rao_table=None,
    sigma_Vc: float = 0.1,
    tau_Vc: float = 600.0,
    T_op_s: float = 20.0 * 60.0,
    quantile_p: float = 0.90,
    omega_grid: Optional[np.ndarray] = None,
    use_pm_for_drift: bool = True,
    k_sigma: float = 0.674,
    t_end_wcfdi: float = 200.0,
    bistability_alarm: float = 1.5,
    progress_cb=None,
) -> WcfdiDecisionMatrix:
    """Build the full forecast-case decision matrix (slots x headings).

    Parameters
    ----------
    cfg, joint : same as the operability polar.
    slots : sequence of ``ForecastSlot``.
    headings_compass : (n_h,) array of vessel headings [rad].
    scenario : WcfdiScenario. Default = single thruster group lost.
    All other kwargs documented on ``evaluate_decision_cell``; the
    defaults are matched to ``operability_polar`` /
    ``wcfdi_operability_overlay`` so polar and matrix are directly
    comparable.
    progress_cb : optional callable ``(k, n, label)`` invoked once per
        cell.

    Returns
    -------
    WcfdiDecisionMatrix.
    """
    if scenario is None:
        scenario = WcfdiScenario(
            alpha=(2.0 / 3.0,) * 3,
            gamma_immediate=0.5,
            T_realloc=10.0,
        )
    headings_arr = np.asarray(headings_compass, dtype=float)
    n_s = len(slots)
    n_h = headings_arr.size
    n_total = n_s * n_h

    cells: list = []
    k = 0
    for s_i, slot in enumerate(slots):
        for h_i, heading in enumerate(headings_arr):
            k += 1
            if progress_cb is not None:
                progress_cb(
                    k, n_total,
                    f"slot {s_i+1}/{n_s} '{slot.label}', "
                    f"heading {np.degrees(heading):.0f} deg",
                )
            cell = evaluate_decision_cell(
                cfg, joint, slot, float(heading),
                slot_index=s_i, heading_index=h_i,
                scenario=scenario, rao_table=rao_table,
                sigma_Vc=sigma_Vc, tau_Vc=tau_Vc,
                T_op_s=T_op_s, quantile_p=quantile_p,
                omega_grid=omega_grid,
                use_pm_for_drift=use_pm_for_drift,
                k_sigma=k_sigma, t_end_wcfdi=t_end_wcfdi,
                bistability_alarm=bistability_alarm,
            )
            cells.append(cell)

    # Resolve thresholds for the result (same logic as evaluate cell).
    pos_warn_r = float(cfg.operational_limits.position_warning_radius_m)
    pos_alarm_r = float(cfg.operational_limits.position_alarm_radius_m)
    L0 = float(joint.L)
    L_min = float(cfg.gangway.telescope_min)
    L_max = float(cfg.gangway.telescope_max)
    stroke = min(max(L0 - L_min, 0.0), max(L_max - L0, 0.0))
    gw_warn_m = 0.60 * stroke
    gw_alarm_m = 0.80 * stroke

    return WcfdiDecisionMatrix(
        slots=tuple(slots),
        headings_compass=headings_arr,
        cells=tuple(cells),
        pos_warn_radius_m=pos_warn_r,
        pos_alarm_radius_m=pos_alarm_r,
        gw_warn_m=float(gw_warn_m),
        gw_alarm_m=float(gw_alarm_m),
        bistability_alarm=float(bistability_alarm),
        k_sigma=float(k_sigma),
        t_end_wcfdi_s=float(t_end_wcfdi),
        scenario_alpha=tuple(scenario.alpha),
        scenario_T_realloc=float(scenario.T_realloc),
    )


__all__ = [
    "ForecastSlot",
    "DecisionCell",
    "WcfdiDecisionMatrix",
    "evaluate_decision_cell",
    "wcfdi_decision_matrix",
]
