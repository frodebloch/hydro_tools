"""Live operational CQA decision-matrix cell, fully measurement-driven.

Background
----------
The ``cqa.decision_matrix`` module ('forecast/planning' pipeline) answers
the question "Will I be able to do this operation 3 hours from now,
given a forecast Hs/Tp/theta?" using parametric env-force PSDs and
spectral covariance integration.

This module is the LIVE pipeline: "Right now, given what the observer
is telling me about the vessel's current state, what is the operability
traffic-light, and how would it change if a WCF happened in the next
second?"

Inputs (must come from the running plant, NOT from any forecast service):
    * Live observer snapshot (eta_hat_LF, nu_hat_LF, b_hat, eta_wave,
      heading), mapped onto the cqa-27 augmented state layout from
      ``cqa.transient_obs``.
    * Bayesian sigma posteriors (LF + WF) from
      ``cqa.online_estimator.BayesianSigmaEstimator`` running on the
      observer's body-frame channels (``SurgeDev`` / ``SwayDev`` /
      ``HeadingDev`` for LF; ``xHf`` / ``yHf`` / ``headingHf`` for WF).
    * Validity badges from ``compose_validity_badge`` (A1-A5 health
      ladder) -- gates the cell to AMBER/RED when the posterior is not
      yet warm or shows zero-mean / consistency violations.
    * Vessel + controller + observer parameter blocks (design-time,
      fixed in ``cfg``).
    * WCFDI scenario knobs (alpha, gamma_immediate, T_realloc) --
      design-time configuration of the failure to evaluate.

Forbidden inputs at runtime:
    Any forecast (Vw, Hs, Tp, Vc, theta) value. The mean environmental
    force is recovered from the observer's bias estimate b_hat (which
    by construction integrates the unmodelled disturbance), and the
    observable position statistics come from the BayesianSigmaEstimator
    posteriors -- not from PSD integration over a forecast spectrum.

WCFDI mean trajectory (deterministic, linear)::

    delta_eta_dev(t) = pulse_response(aug_27, t_grid, tau_lost(t), x0=0)[:, 0:3]
    tau_env := +b_hat                              (see derivation below)
    tau_lost(t) = -(1 - beta(t)) * tau_env         (scenario formula, same as forecast)
    beta(t) = 1 + (gamma_imm - 1) * exp(-t / T_realloc)

Sign convention for tau_env (verified against brucon source):
    Brucon NPO uses M*nu_hat_dot = D(nu_hat) + tau_thr + b_hat + ...
    (libs/dp/dp_estimator/observer_vessel_model.cpp:99-105). At intact
    steady state nu_hat = 0, D(0) = 0, so b_hat = -tau_thr_ss = -OrderTau_pre.
    The mean environmental force the vessel actually experiences is
    -tau_thr_ss = +b_hat (thrusters cancel the env load).
    Verified at pwq30: Order_pre,sway = +110 kN -> b_hat = -110 kN
    -> tau_env = -110 kN -> vessel drifts to -y when thrust is lost,
    matching brucon ensemble-mean (delta_eta_y peak = -0.683 m).

WCFDI position envelope (operator-facing radius)::

    pos_envelope(t) = | eta_hat_LF + delta_eta_mean(t) |
                    + k_sigma * sqrt(sigma_R_LF^2 + sigma_R_WF^2)

The sigma component is TIME-INVARIANT in the linear deterministic
model: the WCFDI transient adds no extra randomness on top of the live
baseline (which already captures all the stationary noise via the
Bayes posterior). Only the mean position moves during the transient.

Intact axis::

    intact_pos_envelope = | eta_hat_LF | + k_sigma * sqrt(sigma_R_LF^2 + sigma_R_WF^2)

Mapping to brucon NPO state (for C++ port)
------------------------------------------
The ``LiveObserverState`` fields map 1:1 to the cqa-27 layout (see
``cqa.transient_obs`` module docstring):

    eta_hat[0..2]   -> idx  6..8   (observer LF position, body)   [m, m, rad]
    nu_hat[0..2]    -> idx  9..11  (observer LF velocity, body)   [m/s, m/s, rad/s]
    b_hat[0..2]     -> idx 12..14  (observer bias, force-units)   [N, N, Nm]
    eta_wave[0..2]  -> idx 24..26  (wave-filter output, body)     [m, m, rad]

Brucon log columns (verified against scripts/p7_brucon_validation/):

    eta_hat[0]    : SurgeDev                   (body LF surge, m)
    eta_hat[1]    : SwayDev                    (body LF sway, m)
    eta_hat[2]    : HeadingDev                 (body LF yaw, rad)
    nu_hat[0]     : SurgeSpeed                 (body LF, m/s)
    nu_hat[1]     : SwaySpeed                  (body LF, m/s)
    nu_hat[2]     : RateOfTurn                 (deg/min -> rad/s via deg2rad/60)
    b_hat[0..2]   : -OrderTau{Surge,Sway,Yaw}  (kN/kNm -> N/Nm via *1e3)
                    (intact-SS approximation: at SS, b_hat = -OrderTau,
                    consistent with brucon's observer nu_dot equation
                    M*nu_hat_dot = D(nu_hat) + tau_thr + b_hat + ...
                    at libs/dp/dp_estimator/observer_vessel_model.cpp:99.
                    For a more faithful live signal, export b_hat
                    directly from the brucon NonlinearPassiveObserver.)
    eta_wave[0]   : xHf                        (m)
    eta_wave[1]   : yHf                        (m)
    eta_wave[2]   : headingHf                  (rad)
    heading_compass : heading                  (deg -> rad via deg2rad)

The truth states (idx 0..2 = eta, idx 3..5 = nu) and observer-internal
states (idx 15..27: tau_thr, integrator, wave-filter integrator) are
not consumed by the live cell; pulse_response works in deviation
coordinates (x0 = zeros) so their absolute values don't matter.

Validation
----------
``scripts/p7_brucon_validation/live_cell_per_seed_pwq30.py`` builds a
LiveObserverState from each pwq30 seed's pre-WCF observer log,
computes the live cell, and overlays the predicted WCFDI envelope
against the realised post-WCF trajectory. Per-seed agreement is
expected to be much better than the forecast-pipeline ensemble
agreement, because the live cell conditions on the actual realised
mean force b_hat instead of an ensemble-average parametric drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import CqaConfig
from .vessel import LinearVesselModel
from .controller import LinearDpController
from .gangway import GangwayJointState, telescope_sensitivity
from .online_estimator import (
    SigmaPosterior, RadialPosterior, ValidityBadge,
)
from .transient import WcfdiScenario
from .transient_obs import (
    AugmentedSystemObs,
    build_observer_augmented_system_full,
    csov_observer_gains,
    pulse_response,
    N_STATE,
    IDX_ETA_HAT, IDX_NU_HAT, IDX_B_HAT, IDX_ETA_W,
)
from .decision_matrix import (
    DecisionCell,
    _imca_traffic, _worst,
)


# ---------------------------------------------------------------------------
# Live observer-state contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveObserverState:
    """Snapshot of the running observer, mapped to cqa-27 layout.

    All quantities are in body-frame (the brucon NPO operates in body
    frame; LF positions are body-fixed deviations from the DP
    setpoint). Heading is in compass radians (true north = 0, increasing
    clockwise looking down).

    See module docstring for the brucon log column mapping.
    """
    eta_hat: np.ndarray         # (3,) body LF position dev [m, m, rad]
    nu_hat: np.ndarray          # (3,) body LF velocity     [m/s, m/s, rad/s]
    b_hat: np.ndarray           # (3,) body bias estimate   [N, N, Nm]
    eta_wave: np.ndarray        # (3,) body WF position     [m, m, rad]
    heading_compass: float      # rad

    def __post_init__(self):
        for name in ("eta_hat", "nu_hat", "b_hat", "eta_wave"):
            v = np.asarray(getattr(self, name), dtype=float)
            if v.shape != (3,):
                raise ValueError(f"{name} must have shape (3,), got {v.shape}")
            object.__setattr__(self, name, v)


@dataclass(frozen=True)
class LiveSigmaPosterior:
    """Bayesian sigma posteriors for both LF and WF, body frame.

    LF is built from BayesianSigmaEstimator on observer's eta_hat
    channels (SurgeDev, SwayDev, HeadingDev). WF from estimators on
    the wave-filter output (xHf, yHf, headingHf).

    The radial sub-fields are derived via combine_radial_posterior from
    the per-axis posteriors.

    The validity badges (one per band) come from compose_validity_badge
    on the per-axis health objects, then aggregated worst-of-three.
    They gate the cell to AMBER/RED when the posterior is not yet warm
    or A1-A5 ladder shows degraded indicators.
    """
    # LF (low-frequency, observer eta_hat band)
    posterior_lf_x: SigmaPosterior
    posterior_lf_y: SigmaPosterior
    posterior_lf_yaw: SigmaPosterior
    radial_lf: RadialPosterior
    validity_lf: ValidityBadge

    # WF (wave-frequency, observer eta_wave band)
    posterior_wf_x: SigmaPosterior
    posterior_wf_y: SigmaPosterior
    posterior_wf_yaw: SigmaPosterior
    radial_wf: RadialPosterior
    validity_wf: ValidityBadge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validity_to_traffic(badge: ValidityBadge) -> str:
    """Worst-of-A1..A5 -> green/amber/red.

    OK -> green; WARMING/UNSETTLED -> amber; INVALID -> red.
    """
    level = getattr(badge, "level", "OK")
    return {"OK": "green", "WARMING": "amber",
            "UNSETTLED": "amber", "INVALID": "red"}.get(level, "amber")


def _build_aug_for_live(cfg: CqaConfig, Tp_obs_s: float = 10.0) -> AugmentedSystemObs:
    """Assemble the cqa-27 augmented system for the live cell.

    The wave-filter peak frequency Tp_obs_s is the observer's current
    estimate (brucon's NPO has its own wave-period estimator that
    sets omega_p at runtime). Default 10 s matches the CSOV scenario
    typical Tp; for production, plumb the live Tp estimator output
    through here.
    """
    vp = cfg.vessel
    cp_ctrl = cfg.controller
    vessel = LinearVesselModel.from_config(vp)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D,
        omega_n=cp_ctrl.omega_n, zeta=cp_ctrl.zeta,
    )
    obs_gains = csov_observer_gains(Tp_s=Tp_obs_s if Tp_obs_s > 0 else 10.0)
    return build_observer_augmented_system_full(
        vessel, controller, obs_gains=obs_gains,
        T_thr=cp_ctrl.thruster_time_constant_s,
    )


# ---------------------------------------------------------------------------
# Live decision cell
# ---------------------------------------------------------------------------


def evaluate_decision_cell_live(
    cfg: CqaConfig,
    joint: GangwayJointState,
    obs_state: LiveObserverState,
    sigma_post: LiveSigmaPosterior,
    *,
    scenario: Optional[WcfdiScenario] = None,
    k_sigma: float = 0.674,
    t_end_wcfdi: float = 200.0,
    n_t: int = 401,
    Tp_obs_s: float = 10.0,
) -> DecisionCell:
    """Evaluate one live operational CQA decision cell.

    Parameters
    ----------
    cfg : CqaConfig
        Vessel, controller, observer, gangway, operational-limits
        parameters. Design-time, fixed.
    joint : GangwayJointState
        Current gangway joint state (live).
    obs_state : LiveObserverState
        Live observer snapshot. See class docstring for the brucon
        column mapping.
    sigma_post : LiveSigmaPosterior
        Live BayesianSigmaEstimator posteriors (LF + WF, body-frame
        per axis, plus combined radial and validity badges).
    scenario : WcfdiScenario, optional
        WCFDI failure to evaluate. Default = single thruster group lost
        with gamma_immediate=0.5, T_realloc=10 s, alpha=2/3.
    k_sigma : float, default 0.674 (= p75 envelope, decision-matrix default).
    t_end_wcfdi : float, default 200 s. Forward horizon for the
        post-WCF transient.
    n_t : int, default 401. Number of integration steps.
    Tp_obs_s : float, default 10 s. Observer's current wave-period
        estimate (drives the wave-filter peak frequency in the model).
        For production, plumb through the live brucon Tp estimator
        output here.

    Returns
    -------
    DecisionCell. ``slot_index`` and ``heading_index`` are set to -1
    (the live cell is a one-off snapshot, not part of a forecast grid).
    ``intact_pos_a_p90_m`` carries the sigma-envelope value (mean
    offset + k_sigma * sigma_R_total), NOT a p90 quantile (the live
    pipeline drops the operational-quantile semantics in favour of a
    pure sigma envelope, per the design discussion).

    Notes
    -----
    The WCFDI mean trajectory is propagated in deviation coordinates
    via ``pulse_response(x0=0)``, then added to the live offset
    ``eta_hat_LF`` to get the predicted absolute footprint. The sigma
    envelope is time-invariant (no extra randomness from the linear
    deterministic transient on top of the live baseline).
    """
    if scenario is None:
        scenario = WcfdiScenario(
            alpha=(2.0 / 3.0,) * 3,
            gamma_immediate=0.5,
            T_realloc=10.0,
        )

    # ---- Mean environmental force from the observer's bias estimate ----
    # The brucon NonlinearPassiveObserver models the observer's nu_dot as
    #   M * nu_hat_dot = D(nu_hat) + tau_thr + b_hat + ...
    # (libs/dp/dp_estimator/observer_vessel_model.cpp:99-105). At the
    # observer's quasi-SS with nu_hat=0 and D(0)=0, this gives
    # b_hat = -tau_thr_ss = -OrderTau_pre. Physically the observer's bias
    # estimate IS the mean unmodelled force on the vessel, i.e.
    #     tau_env := +b_hat                       (NOT -b_hat)
    # Verified at pwq30: Order_pre_sway = +110 kN -> b_hat_sway = -110 kN
    # -> tau_env_sway = -110 kN (env pushes vessel in -y), consistent
    # with brucon truth (vessel drifts -y post-WCF).
    tau_env = np.asarray(obs_state.b_hat, dtype=float)

    # ---- Build the cqa-27 augmented system at the observer's Tp ----
    aug = _build_aug_for_live(cfg, Tp_obs_s=Tp_obs_s)

    # ---- WCFDI scenario tau_lost(t), same formula as forecast pipeline ----
    # See cqa.decision_matrix._wcfdi_peak_at_forecast_obs for derivation.
    # tau_lost(t) = -(1 - beta(t)) * tau_env, injected via B_lost = +Minv.
    t_grid = np.linspace(0.0, t_end_wcfdi, n_t)
    gamma_imm = float(scenario.gamma_immediate)
    T_realloc = float(scenario.T_realloc) if scenario.T_realloc > 0 else 1e-9
    beta_t = 1.0 + (gamma_imm - 1.0) * np.exp(-t_grid / T_realloc)
    tau_lost = (beta_t[:, None] - 1.0) * (-tau_env[None, :])

    # ---- Mean deviation trajectory ----
    X = pulse_response(aug, t_grid, tau_lost, x0=np.zeros(N_STATE))
    delta_eta_mean = X[:, 0:3]   # body, m/m/rad

    # ---- Sigma envelope (LF + WF, time-invariant in linear deterministic model) ----
    sigma_R_lf = float(sigma_post.radial_lf.sigma_R_median)
    sigma_R_wf = float(sigma_post.radial_wf.sigma_R_median)
    sigma_R_total = float(np.sqrt(sigma_R_lf ** 2 + sigma_R_wf ** 2))

    # Per-axis sigmas for gangway-tip projection.
    sig_lf = np.array([sigma_post.posterior_lf_x.sigma_median,
                       sigma_post.posterior_lf_y.sigma_median,
                       sigma_post.posterior_lf_yaw.sigma_median])
    sig_wf = np.array([sigma_post.posterior_wf_x.sigma_median,
                       sigma_post.posterior_wf_y.sigma_median,
                       sigma_post.posterior_wf_yaw.sigma_median])

    # Telescope sensitivity (3-DoF body): c_L @ delta_eta = delta_L.
    c_L = telescope_sensitivity(joint, cfg.gangway)

    # ---- Intact axis (live observer offset + sigma envelope) ----
    eta_hat_lf = np.asarray(obs_state.eta_hat, dtype=float)
    intact_pos_offset = float(np.hypot(eta_hat_lf[0], eta_hat_lf[1]))
    intact_pos_envelope = intact_pos_offset + k_sigma * sigma_R_total

    intact_dL_offset = float(np.abs(c_L @ eta_hat_lf))
    sigma_dL_lf = float(np.sqrt(np.sum((c_L * sig_lf) ** 2)))
    sigma_dL_wf = float(np.sqrt(np.sum((c_L * sig_wf) ** 2)))
    sigma_dL_total = float(np.sqrt(sigma_dL_lf ** 2 + sigma_dL_wf ** 2))
    intact_dL_envelope = intact_dL_offset + k_sigma * sigma_dL_total

    # IMCA traffic for intact.
    pos_warn_r = float(cfg.operational_limits.position_warning_radius_m)
    pos_alarm_r = float(cfg.operational_limits.position_alarm_radius_m)
    L0 = float(joint.L)
    L_min = float(cfg.gangway.telescope_min)
    L_max = float(cfg.gangway.telescope_max)
    stroke = min(max(L0 - L_min, 0.0), max(L_max - L0, 0.0))
    gw_warn_m = 0.60 * stroke
    gw_alarm_m = 0.80 * stroke

    intact_pos_traffic = _imca_traffic(intact_pos_envelope, pos_warn_r, pos_alarm_r)
    if stroke > 0.0:
        intact_gw_traffic = _imca_traffic(intact_dL_envelope, gw_warn_m, gw_alarm_m)
    else:
        intact_gw_traffic = "green"
    intact_traffic = _worst(intact_pos_traffic, intact_gw_traffic)

    # ---- WCFDI axis (live offset + transient mean + sigma envelope) ----
    # Absolute footprint at each t: |eta_hat_LF + delta_eta_mean(t)| in xy.
    eta_xy_t = eta_hat_lf[None, 0:2] + delta_eta_mean[:, 0:2]
    pos_t = np.sqrt(np.sum(eta_xy_t ** 2, axis=1))
    pos_envelope_t = pos_t + k_sigma * sigma_R_total
    wcfdi_pos_peak = float(pos_envelope_t.max())

    dL_t = (eta_hat_lf[None, :] + delta_eta_mean) @ c_L
    dL_envelope_t = np.abs(dL_t) + k_sigma * sigma_dL_total
    wcfdi_dL_peak = float(dL_envelope_t.max())

    wcfdi_pos_traffic = _imca_traffic(wcfdi_pos_peak, pos_warn_r, pos_alarm_r)
    if stroke > 0.0:
        wcfdi_gw_traffic = _imca_traffic(wcfdi_dL_peak, gw_warn_m, gw_alarm_m)
    else:
        wcfdi_gw_traffic = "green"
    wcfdi_traffic = _worst(wcfdi_pos_traffic, wcfdi_gw_traffic)

    # ---- Validity-badge gate ----
    # The validity badges from the BayesianSigmaEstimator chain go on
    # top of the IMCA traffic: if the posterior is INVALID the cell
    # cannot be trusted -> red; WARMING/UNSETTLED -> at least amber.
    badge_traffic_lf = _validity_to_traffic(sigma_post.validity_lf)
    badge_traffic_wf = _validity_to_traffic(sigma_post.validity_wf)
    badge_traffic = _worst(badge_traffic_lf, badge_traffic_wf)
    intact_traffic = _worst(intact_traffic, badge_traffic)
    wcfdi_traffic = _worst(wcfdi_traffic, badge_traffic)
    overall_traffic = _worst(intact_traffic, wcfdi_traffic)

    return DecisionCell(
        slot_index=-1,
        heading_index=-1,
        heading_compass=float(obs_state.heading_compass),
        theta_rel=float("nan"),  # not defined for the live cell
        intact_pos_a_p90_m=intact_pos_envelope,
        intact_gw_a_p90_m=intact_dL_envelope,
        intact_pos_traffic=intact_pos_traffic,
        intact_gw_traffic=intact_gw_traffic,
        intact_traffic=intact_traffic,
        wcfdi_pos_peak_m=wcfdi_pos_peak,
        wcfdi_gw_peak_m=wcfdi_dL_peak,
        wcfdi_pos_traffic=wcfdi_pos_traffic,
        wcfdi_gw_traffic=wcfdi_gw_traffic,
        wcfdi_bistability_score=0.0,  # not defined in linear 27-state
        wcfdi_cqa_violated=False,     # not enforced in the live cell v1
        wcfdi_traffic=wcfdi_traffic,
        overall_traffic=overall_traffic,
    )
