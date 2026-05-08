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
    if Vw > 1e-9:
        S_wind = npd_wind_gust_force_psd(wind_model, Vw, theta_rel)
    else:
        # No mean wind => NPD spectrum is undefined (Vw_mean appears
        # with a negative power) and the gust force PSD is identically
        # zero. Use a constant zero PSD callable so the Lyapunov sum
        # stays well-defined.
        def S_wind(_w):
            return np.zeros((3, 3))

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


def _wcfdi_peak_at_forecast_obs(
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
    rao_table=None,
    n_t: int = 401,
):
    """27-state observer-augmented counterpart of _wcfdi_peak_at_forecast.

    Drives the explicit-observer model in ``cqa.transient_obs`` (validated
    against brucon truth at pwq30 to ~25 percent on LF mean peak; the
    legacy 15-state pipeline is ~4x low). Returns the same tuple shape:
    ``(peak_pos_m, peak_dL_m, bistability_score, cqa_violated)``.

    Mean trajectory: linear ODE on the 27-state aug system driven by
    ``tau_lost(t)`` from the WcfdiScenario thrust-cap recovery (no
    saturation in the linear sense; saturation lives entirely in the
    construction of tau_lost(t)). Initial state is the exact intact SS
    via ``A x_ss = -B_d tau_env``.

    Covariance trajectory: Lyapunov ODE
        dP/dt = A P + P A^T + B_w W_eq B_w^T
    on the same 27-state intact A, with W_eq matched to the intact
    closed-loop 6-DOF P6 (truth (eta, nu) covariance from the
    frequency-domain integral over wind/drift/current PSDs), exactly the
    same W_eq pattern as in cqa.transient. P0 lifts P6 into the
    27-state augmented covariance with observer/wave-filter blocks zero.

    bistability_score is left at 0 in v1; the saturation gate from the
    15-state model does not directly translate (no clipping in the linear
    27-state).
    """
    from .vessel import LinearVesselModel, CurrentForceModel
    from .controller import LinearDpController
    from .closed_loop import ClosedLoop, state_covariance_freqdomain
    from .psd import (
        npd_wind_gust_force_psd,
        slow_drift_force_psd_newman,
        current_variability_force_psd,
        WindForceModel,
    )
    from .transient_obs import (
        build_observer_augmented_system_full,
        pulse_response,
        csov_observer_gains,
        N_STATE as N_STATE_OBS,
    )
    from .transient import lift_intact_cov_to_augmented
    from scipy.integrate import solve_ivp

    vp = cfg.vessel
    wp = cfg.wind
    cp = cfg.current
    wd = cfg.wave_drift
    cp_ctrl = cfg.controller

    vessel = LinearVesselModel.from_config(vp)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D,
        omega_n=cp_ctrl.omega_n, zeta=cp_ctrl.zeta,
    )

    # Wave-filter peak frequency from forecast Tp (the brucon observer
    # locks omega_p to the wave-period estimate at runtime).
    obs_gains = csov_observer_gains(Tp_s=Tp if Tp > 0 else 10.0)
    aug = build_observer_augmented_system_full(
        vessel, controller, obs_gains=obs_gains,
        T_thr=cp_ctrl.thruster_time_constant_s,
    )

    # ---- Mean environmental force (wind + current + mean drift) ----
    wind_model = WindForceModel(wp=wp, loa=vp.loa)
    current_model = CurrentForceModel(
        cp=cp,
        lateral_area_underwater=vp.lpp * vp.draft,
        frontal_area_underwater=vp.beam * vp.draft,
        loa=vp.loa,
    )
    F_wind = wind_model.force(Vw, theta_rel)
    F_curr = current_model.force(Vc, theta_rel)
    if rao_table is not None:
        from .drift import mean_drift_force_pdstrip
        F_drift = mean_drift_force_pdstrip(
            rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel,
        )
    else:
        F_drift = np.array([
            wd.drift_x_amp * Hs ** 2 * np.cos(theta_rel),
            wd.drift_y_amp * Hs ** 2 * np.sin(theta_rel),
            wd.drift_n_amp * Hs ** 2 * np.sin(2.0 * theta_rel),
        ])
    tau_env = F_wind + F_curr + F_drift

    cap_post = scenario.resolved_cap_post(cfg)
    cqa_violated_arr = np.abs(tau_env) > cap_post
    if bool(np.any(cqa_violated_arr)):
        return float("inf"), float("inf"), 0.0, True

    # ---- Initial perturbation: zero (we propagate the deviation from the
    # intact SS). The validated direct cqa-27 path against brucon truth uses
    # x0 = zeros for the same reason: pulse_response solves
    #     d delta_x / dt = A delta_x + B_lost tau_lost(t)
    # i.e. it omits the persistent B_d tau_env term, so an "absolute" SS
    # initialisation would relax under unforced A and produce a spurious
    # transient. The footprint metric of operational interest is the
    # deviation from the operator's commanded set-point (= intact SS) anyway.
    x0 = np.zeros(N_STATE_OBS)

    # ---- tau_lost(t) on uniform grid ----
    # The "deliverable thrust" model used here is the brucon-physics-faithful
    # one: at t=0+ half the thrusters drop out, the surviving alloc cannot
    # immediately reproduce the pre-WCF thrust direction, and the actually
    # delivered thrust drops to a fraction of T_pre = -tau_env. It then
    # recovers toward the steady-state achievable thrust over T_realloc.
    #
    # Empirically (cqa/scripts/p7_brucon_validation/check_obs_perseed_taulost.py
    # ensemble at pwq30, sway DOF):
    #   T_pre        = +108 kN
    #   T(t+1s)      = -29 kN     (delivered thrust drops by ~137 kN)
    #   tau_lost(0+) = T(0+) - T_pre = -137 kN
    # The previous "cap-based" formulation tau_lost = tau_env - clip(tau_env,
    # +-cap) gives identically zero whenever |tau_env| <= cap_immediate, which
    # is the operationally interesting regime (CQA precondition holds with
    # margin). It systematically misses the dominant transient, which is
    # set by the allocator's inability to reproduce the pre-WCF thrust
    # direction immediately, NOT by capacity saturation per DOF.
    #
    # Parameterisation:
    #   T_post(t) = beta(t) * T_pre        (pre-WCF balanced thrust scaled)
    #   beta(0+) = gamma_immediate         (default 0.5: brucon-empirical match)
    #   beta(inf) = 1                      (controller recovers to balance)
    #   beta(t) = 1 + (gamma_immediate - 1) * exp(-t / T_realloc)
    # tau_lost(t) injected via B_lost (B_lost = +Minv) is then:
    #   tau_lost(t) = T_post(t) - T_pre = (beta(t) - 1) * T_pre
    #               = -(1 - beta(t)) * tau_env
    #               = -(1 - gamma_immediate) * tau_env * exp(-t / T_realloc)
    # (T_pre = -tau_env at intact SS).
    #
    # Note: scenario.alpha encodes the steady-state surviving-fraction cap;
    # not used here because we model recovery as beta(inf)=1 (full reach back
    # to the demanded thrust). If alpha < |tau_env|/cap_intact in any DOF,
    # the CQA precondition has already failed and we returned (inf, inf)
    # above. Inside the precondition envelope, the steady-state achievable
    # thrust always equals the demand.
    t_grid = np.linspace(0.0, t_end, n_t)
    gamma_imm = float(scenario.gamma_immediate)
    T_realloc = float(scenario.T_realloc) if scenario.T_realloc > 0 else 1e-9
    beta_t = 1.0 + (gamma_imm - 1.0) * np.exp(-t_grid / T_realloc)  # (n_t,)
    tau_lost = (beta_t[:, None] - 1.0) * (-tau_env[None, :])
    # i.e. tau_lost(t, dof) = -(1 - beta(t)) * tau_env[dof]
    # At t=0+ : tau_lost = -(1 - gamma_imm) * tau_env
    # At t=inf: tau_lost = 0

    # ---- Mean trajectory ----
    try:
        X = pulse_response(aug, t_grid, tau_lost, x0=x0)
    except Exception:
        return float("inf"), float("inf"), 0.0, True

    eta_mean = X[:, 0:3]  # truth body deviation (m, m, rad)

    # ---- Covariance trajectory (Lyapunov ODE on intact A) ----
    cl_intact = ClosedLoop.build(vessel, controller)
    if Vw > 1e-9:
        S_wind = npd_wind_gust_force_psd(wind_model, Vw, theta_rel)
    else:
        def S_wind(_w):
            return np.zeros((3, 3))
    if rao_table is not None:
        from .drift import slow_drift_force_psd_newman_pdstrip
        S_drift = slow_drift_force_psd_newman_pdstrip(
            rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel,
        )
    else:
        S_drift = slow_drift_force_psd_newman(
            (wd.drift_x_amp, wd.drift_y_amp, wd.drift_n_amp),
            Hs, Tp, theta_rel,
        )
    if Vc > 1e-9:
        dFdVc = 2.0 * F_curr / Vc
    else:
        dFdVc = np.zeros(3)
    S_curr = current_variability_force_psd(dFdVc, sigma_Vc=sigma_Vc, tau=tau_Vc)
    P6 = state_covariance_freqdomain(cl_intact, [S_wind, S_drift, S_curr])
    P0 = lift_intact_cov_to_augmented(P6, n_state=N_STATE_OBS)

    # Match W_eq to intact 6-DOF P6 via Lyapunov on the 6-state closed loop.
    A_cl6 = cl_intact.A_cl
    B_w6 = cl_intact.B_w
    Q6 = -(A_cl6 @ P6 + P6 @ A_cl6.T)
    Bp = np.linalg.pinv(B_w6)
    W_eq = Bp @ Q6 @ Bp.T
    W_eq = 0.5 * (W_eq + W_eq.T)
    eigs, V = np.linalg.eigh(W_eq)
    eigs = np.maximum(eigs, 0.0)
    W_eq = V @ np.diag(eigs) @ V.T
    BWBT_aug = aug.B_w @ W_eq @ aug.B_w.T

    n_aug = aug.n_state
    def rhs_P(t, P_flat):
        P = P_flat.reshape(n_aug, n_aug)
        return (aug.A @ P + P @ aug.A.T + BWBT_aug).flatten()

    sol_P = solve_ivp(
        fun=rhs_P,
        t_span=(0.0, t_end),
        y0=P0.flatten(),
        t_eval=t_grid,
        method="RK45",
        rtol=1e-5,
        atol=1e-9,
    )
    if not sol_P.success:
        return float("inf"), float("inf"), 0.0, True
    P_t = sol_P.y.T.reshape(n_t, n_aug, n_aug)
    P_t = 0.5 * (P_t + P_t.transpose(0, 2, 1))
    P_eta = P_t[:, 0:3, 0:3]

    # ---- Envelopes ----
    pos_mean_r = np.sqrt(eta_mean[:, 0] ** 2 + eta_mean[:, 1] ** 2)
    sigma_R_t = np.sqrt(np.maximum(P_eta[:, 0, 0] + P_eta[:, 1, 1], 0.0))
    pos_envelope = pos_mean_r + k_sigma * sigma_R_t

    dL_mean = eta_mean @ c_L
    sigma_dL = np.sqrt(np.maximum(
        np.einsum("i,nij,j->n", c_L, P_eta, c_L), 0.0,
    ))
    dL_envelope = np.abs(dL_mean) + k_sigma * sigma_dL

    return (
        float(np.max(pos_envelope)),
        float(np.max(dL_envelope)),
        0.0,  # bistability_score: not directly defined for the linear 27-state
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
    use_obs_transient: bool = False,
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

    if use_obs_transient:
        pos_peak, dL_peak, bist_score, cqa_violated = _wcfdi_peak_at_forecast_obs(
            cfg, joint,
            Vw=slot.Vw, Hs=slot.Hs, Tp=slot.Tp, Vc=slot.Vc,
            theta_rel=theta_rel,
            scenario=scenario, k_sigma=k_sigma, t_end=t_end_wcfdi,
            sigma_Vc=sigma_Vc, tau_Vc=tau_Vc, c_L=c_L,
            rao_table=rao_table,
        )
    else:
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
    use_obs_transient: bool = False,
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
                use_obs_transient=use_obs_transient,
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
