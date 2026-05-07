"""Measurement-calibrated WCFDI counter-factual estimation.

Phase 1 of the live operability nowcast (analysis.md §12.21.6): inject
two measurement-derived inputs into ``wcfdi_transient`` /
``wcfdi_mc`` so the post-WCFDI excursion estimate reflects the
vessel's current state instead of the operator-set sea state alone.

The two injected inputs (per §12.21.6 design discussion):

  1. **Intact P6 diagonals** (LF σ_x_body, σ_y_body, σ_ψ_body) from
     :class:`cqa.online_estimator.BayesianSigmaEstimator`. The model's
     correlation structure on (η, ν) is preserved, only the position
     diagonals are rescaled to the measured magnitudes::

         P6_cal = D · P6_model · D
         D = diag(σ_measured / σ_model, 1, 1, 1)   # σ on η only

     ``D · P · D`` preserves PSD-ness when ``D`` is diagonal-positive
     and ``P`` is PSD, and preserves the correlation matrix
     ``C = D_model^{-1} · P · D_model^{-1}`` exactly. Velocities
     (rows/columns 3..5) are left at model values; closed-loop coupling
     keeps them roughly proportional to the position σ but the strict
     measurement-only path for velocity estimation is Phase 2 work.

  2. **Mean environmental force** ``tau_env_measured = (Fx, Fy, Mz)``
     from the controller's bias-estimator integrator (`b_hat`), which by
     construction tracks the slowly-varying environmental load. Replaces
     the model's ``F_wind + F_curr + F_drift`` computation entirely. The
     intact mean steady state ``x_ss_intact`` is then derived from this
     measured force, so the post-WCF deterministic trajectory starts from
     the right operating point.

The rest of the wcfdi propagation logic (augmented system A, controller
gains, post-failure cap schedule, MC sampling, gangway projection) is
re-used unchanged from :mod:`cqa.wcfdi_mc` and :mod:`cqa.transient` --
this module is pure plumbing that swaps the two spectral-derived inputs.

Validation against brucon WCF event in ``pwo`` ensemble: see
``scripts/p7_brucon_validation/calibrated_wcfdi_brucon_validation.py``
and ``analysis.md §12.21.7``.

Future work (Phase 2, see §12.21.6):
  - Architecture C: rescale at ``operator_view.summarise_for_operator``
    to support real-time wave-radar / buoy-derived spectrum inputs for
    the intact-state nowcast.
  - Full re-engineering of ``wcfdi_mc`` to take ``(σ_intact_measured,
    tau_env_measured)`` as primary inputs, building P6 from a
    model-derived correlation matrix scaled to measured diagonals
    consistently across positions and velocities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
from scipy.integrate import solve_ivp

from .config import CqaConfig
from .vessel import LinearVesselModel
from .controller import LinearDpController
from .closed_loop import (
    ClosedLoop,
    state_covariance_freqdomain,
    state_covariance_freqdomain_general,
)
from .psd import (
    npd_wind_gust_force_psd,
    slow_drift_force_psd_newman,
    current_variability_force_psd,
)
from .vessel import WindForceModel, CurrentForceModel
from .transient import (
    AugmentedSystem,
    build_augmented_system,
    intact_mean_steady_state,
    lift_intact_cov_to_augmented,
    WcfdiScenario,
    TransientResult,
    _augmented_rhs_post,
    _clip_per_dof,
)
from .gangway import GangwayJointState, telescope_sensitivity
from .wcfdi_mc import WcfdiMcResult


# ---------------------------------------------------------------------------
# Diagonal rescale of a covariance to measured sigmas (PSD-preserving)
# ---------------------------------------------------------------------------


def rescale_covariance_diagonal(
    P: np.ndarray,
    sigma_target: np.ndarray,
    indices: Sequence[int],
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale a PSD covariance matrix's diagonals at selected indices.

    Builds D = diag(d_i) where d_i = sigma_target_i / sqrt(P[i, i]) for
    i in ``indices`` and d_i = 1 otherwise, then returns ``D @ P @ D``.

    Properties (analytical, exact):
      - Result is symmetric and PSD whenever ``P`` is symmetric PSD
        (D-conjugation by a positive diagonal preserves PSD-ness).
      - ``diag(result)[indices] == sigma_target ** 2`` (exact).
      - The correlation matrix ``C = D_orig^{-1} · P · D_orig^{-1}`` is
        unchanged for any pair (i, j) where both i and j are in
        ``indices``, and proportionally rescaled for cross-terms.

    Parameters
    ----------
    P : array (n, n)
        Symmetric PSD covariance to rescale. Not modified.
    sigma_target : array
        Per-index target sigma (one entry per index in ``indices``).
        Must be the same length as ``indices``.
    indices : sequence of int
        Positions in ``P`` whose diagonals (and their associated rows /
        columns) should be rescaled.
    eps : float
        Floor for the model sigma to avoid divide-by-zero on degenerate
        diagonals.

    Returns
    -------
    P_rescaled : array (n, n)
        ``D · P · D`` with the constructed diagonal D.
    d : array (n,)
        The diagonal entries of D (per-row scaling factors). Useful for
        diagnostics and follow-up rescales.

    Raises
    ------
    ValueError
        On size mismatch, non-square P, or negative sigma_target.
    """
    P = np.asarray(P, dtype=float)
    sigma_target = np.asarray(sigma_target, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P must be square; got shape {P.shape}")
    if len(indices) != len(sigma_target):
        raise ValueError(
            f"len(indices)={len(indices)} != len(sigma_target)={len(sigma_target)}"
        )
    if np.any(sigma_target < 0):
        raise ValueError("sigma_target entries must be non-negative")
    n = P.shape[0]
    d = np.ones(n)
    for k, idx in enumerate(indices):
        if not (0 <= idx < n):
            raise ValueError(f"indices contains out-of-range value {idx} (n={n})")
        diag_model = float(P[idx, idx])
        sigma_model = np.sqrt(max(diag_model, 0.0))
        if sigma_model < eps:
            # Model thinks this DOF has no variance; we can't preserve a
            # non-trivial correlation structure, so just inject the
            # measured magnitude on the diagonal (cross-terms remain
            # zero, which is the model's belief).
            d[idx] = 1.0
            P_diag_target = float(sigma_target[k]) ** 2
            P = P.copy()
            P[idx, idx] = P_diag_target
        else:
            d[idx] = float(sigma_target[k]) / sigma_model
    P_rescaled = (d[:, None] * P) * d[None, :]
    # Clean tiny asymmetry from FP arithmetic
    P_rescaled = 0.5 * (P_rescaled + P_rescaled.T)
    return P_rescaled, d


# ---------------------------------------------------------------------------
# Calibrated operating-point context
# ---------------------------------------------------------------------------


@dataclass
class CalibratedContext:
    """Operating-point context with measurement-substituted tau_env and P6.

    Used by both :func:`wcfdi_transient_calibrated` and
    :func:`wcfdi_mc_calibrated` so the substitution is identical
    across the linearised and MC pipelines.

    Attributes
    ----------
    aug : AugmentedSystem
        12-state augmented system (vessel + controller + bias-estimator
        + thruster lag). Same A, B, B_w as the model build at this
        operating point; the calibration is in tau_env / P6 / P12 only.
    tau_env_used : (3,) array
        The tau_env that was used to build x_ss_intact and (in the
        linearised path) the mean trajectory. Equal to the supplied
        tau_env_measured when given, else recomputed from cfg + spectra.
    x_ss_intact : (12,) array
        Intact mean steady state, derived from ``tau_env_used``.
    P6_calibrated : (6, 6) array
        Intact (eta, nu) covariance with η-diagonals rescaled to the
        measured sigmas. ν block left at model values.
    P12_calibrated : (12, 12) array
        Same rescale, lifted to the augmented state.
    P6_model : (6, 6) array
        Original model P6 before rescale (kept for diagnostics / the
        per-DOF ratio reporting).
    sigma_measured_lf_body : (3,) array
        The σ values that were injected on the η diagonals.
    sigma_model_lf_body : (3,) array
        The corresponding model σ values (sqrt of P6_model diagonal),
        useful for reporting the per-DOF calibration ratio.
    """

    aug: AugmentedSystem
    tau_env_used: np.ndarray
    x_ss_intact: np.ndarray
    P6_calibrated: np.ndarray
    P12_calibrated: np.ndarray
    P6_model: np.ndarray
    sigma_measured_lf_body: np.ndarray
    sigma_model_lf_body: np.ndarray
    cl_intact: ClosedLoop
    vessel: LinearVesselModel
    controller: LinearDpController


def build_calibrated_context(
    cfg: CqaConfig,
    *,
    sigma_measured_lf_body: Sequence[float],
    tau_env_measured: Sequence[float],
    Vw_mean: float = 0.0,
    Hs: float = 0.0,
    Tp: float = 8.0,
    Vc: float = 0.0,
    theta_rel: float = 0.0,
    omega_n: Optional[tuple[float, float, float]] = None,
    zeta: Optional[tuple[float, float, float]] = None,
    T_b: Optional[float] = None,
    T_thr: Optional[float] = None,
    sigma_Vc: float = 0.1,
    tau_Vc: float = 600.0,
    rao_table=None,
) -> CalibratedContext:
    """Build the calibrated operating-point context.

    Parameters
    ----------
    cfg : CqaConfig
        Vessel + controller + gangway configuration. The vessel
        mechanics, controller bandwidth/damping, observer time
        constants, and post-failure scenario all stay model-driven --
        only ``tau_env`` and the η-diagonal of ``P6`` get replaced.
    sigma_measured_lf_body : (3,) array-like
        ``(σ_x_body, σ_y_body, σ_ψ_body)`` from the live
        :class:`BayesianSigmaEstimator` posterior, body-frame, low-pass
        filtered to the controller-relevant band. ``σ_ψ_body`` in
        radians.
    tau_env_measured : (3,) array-like
        ``(Fx_body, Fy_body, Mz_body)`` mean environmental force from
        the controller's bias-estimator integrator at the WCF instant.
        Replaces ``F_wind + F_curr + F_drift`` from the spectral path.
    Vw_mean, Hs, Tp, Vc, theta_rel
        Sea state used to build the model's wind/drift/current PSDs
        ``S_wind``, ``S_drift``, ``S_curr`` for the unmeasured (η, ν)
        cross-correlation structure and the ν diagonal. These should
        be the operator's best estimate of the *current* sea state;
        if unavailable, pass any reasonable defaults -- the η diagonal
        is fully measurement-driven and dominates the post-WCF peak,
        so the unmeasured-block sensitivity is Phase 2 follow-up.
    omega_n, zeta, T_b, T_thr, sigma_Vc, tau_Vc, rao_table
        Same semantics as :func:`cqa.wcfdi_mc._build_operating_context`.

    Returns
    -------
    CalibratedContext
    """
    sigma_meas = np.asarray(sigma_measured_lf_body, dtype=float)
    tau_env_meas = np.asarray(tau_env_measured, dtype=float)
    if sigma_meas.shape != (3,):
        raise ValueError(
            f"sigma_measured_lf_body must have shape (3,), got {sigma_meas.shape}"
        )
    if tau_env_meas.shape != (3,):
        raise ValueError(
            f"tau_env_measured must have shape (3,), got {tau_env_meas.shape}"
        )
    if np.any(sigma_meas < 0):
        raise ValueError("sigma_measured_lf_body entries must be non-negative")

    vp = cfg.vessel
    wp = cfg.wind
    cp_p = cfg.current
    wd = cfg.wave_drift

    cp_ctrl = cfg.controller
    if omega_n is None:
        omega_n = cp_ctrl.omega_n
    if zeta is None:
        zeta = cp_ctrl.zeta
    if T_b is None:
        T_b = cp_ctrl.bias_time_constant_s
    if T_thr is None:
        T_thr = cp_ctrl.thruster_time_constant_s

    vessel = LinearVesselModel.from_config(vp)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D, omega_n=omega_n, zeta=zeta
    )
    aug = build_augmented_system(vessel, controller, T_b=T_b, T_thr=T_thr)

    # --- intact mean steady state from MEASURED tau_env ---
    x_ss_intact = intact_mean_steady_state(aug, tau_env_meas)

    # --- model P6 from spectra (used only for correlation structure +
    #     unmeasured ν block) ---
    cl_intact = ClosedLoop.build(vessel, controller)

    # Build the wind/drift/current PSDs at the operator-supplied sea
    # state. These are required by the closed-loop covariance integral
    # because the correlation structure on (η, ν) depends on which
    # disturbance dominates -- but the η magnitudes will be over-written
    # by sigma_meas immediately afterwards, so the calibration is robust
    # to errors in (Hs, Tp, β).
    if Vw_mean > 1e-9:
        wind_model = WindForceModel(wp=wp, loa=vp.loa)
        S_wind = npd_wind_gust_force_psd(wind_model, Vw_mean, theta_rel)
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
            (wd.drift_x_amp, wd.drift_y_amp, wd.drift_n_amp), Hs, Tp, theta_rel,
        )

    lateral_uw = vp.lpp * vp.draft
    frontal_uw = vp.beam * vp.draft
    current_model = CurrentForceModel(
        cp=cp_p,
        lateral_area_underwater=lateral_uw,
        frontal_area_underwater=frontal_uw,
        loa=vp.loa,
    )
    F_curr = current_model.force(Vc, theta_rel)
    if Vc > 1e-9:
        dFdVc = 2.0 * F_curr / Vc
    else:
        dFdVc = np.zeros(3)
    S_curr = current_variability_force_psd(dFdVc, sigma_Vc=sigma_Vc, tau=tau_Vc)

    P6_model = state_covariance_freqdomain(cl_intact, [S_wind, S_drift, S_curr])
    sigma_model_lf_body = np.sqrt(np.maximum(np.diag(P6_model[:3, :3]), 0.0))

    # --- diagonal rescale of η block to measured sigmas ---
    # Indices 0, 1, 2 in the 6-state (η, ν) ordering are surge, sway, yaw
    # position; 3, 4, 5 are the corresponding velocities (left at model
    # values).
    P6_cal, _d6 = rescale_covariance_diagonal(
        P6_model, sigma_meas, indices=(0, 1, 2),
    )

    # --- 12-state augmented covariance: same rescale logic ---
    # Build the model P12 first, then rescale its η diagonals.
    P12_model = state_covariance_freqdomain_general(
        aug.A, aug.B_w, [S_wind, S_drift, S_curr]
    )
    P12_cal, _d12 = rescale_covariance_diagonal(
        P12_model, sigma_meas, indices=(0, 1, 2),
    )

    return CalibratedContext(
        aug=aug,
        tau_env_used=tau_env_meas.copy(),
        x_ss_intact=x_ss_intact,
        P6_calibrated=P6_cal,
        P12_calibrated=P12_cal,
        P6_model=P6_model,
        sigma_measured_lf_body=sigma_meas.copy(),
        sigma_model_lf_body=sigma_model_lf_body,
        cl_intact=cl_intact,
        vessel=vessel,
        controller=controller,
    )


# ---------------------------------------------------------------------------
# Calibrated linearised transient (mean trajectory + variance ODE)
# ---------------------------------------------------------------------------


def wcfdi_transient_calibrated(
    cfg: CqaConfig,
    scenario: WcfdiScenario,
    ctx: CalibratedContext,
    *,
    t_end: float = 200.0,
    n_t: int = 401,
) -> TransientResult:
    """Linearised post-WCFDI propagation with measurement-calibrated inputs.

    Equivalent to :func:`cqa.transient.wcfdi_transient` but takes a
    pre-built :class:`CalibratedContext` instead of recomputing the
    sea-state-derived ``tau_env`` and ``P6``. The post-failure dynamics,
    immediate-cap clipping logic, mean-trajectory ODE, and covariance ODE
    are re-implemented here (rather than calling ``wcfdi_transient``
    directly) because the upstream API is wired through the spectral
    pipeline.
    """
    aug = ctx.aug
    tau_env = ctx.tau_env_used
    x0 = ctx.x_ss_intact
    P6 = ctx.P6_calibrated
    cl_intact = ctx.cl_intact

    # Apply T_thr_post override if requested
    if scenario.T_thr_post is not None:
        T_b = cfg.controller.bias_time_constant_s
        aug = build_augmented_system(
            ctx.vessel, ctx.controller, T_b=T_b, T_thr=scenario.T_thr_post
        )

    cap_post = scenario.resolved_cap_post(cfg)
    cap_immediate = scenario.resolved_cap_immediate(cfg)

    cqa_violated = np.abs(tau_env) > cap_post

    # Step force imbalance: clip thruster output to immediate cap.
    x0_post = x0.copy()
    x0_post[9:12] = _clip_per_dof(x0[9:12], cap_immediate)
    delta_tau = x0_post[9:12] - x0[9:12]

    cap_fn = lambda t: scenario.cap_at_time(t, cfg)

    # Mean trajectory ODE
    t_eval = np.linspace(0.0, t_end, n_t)
    sol = solve_ivp(
        fun=lambda t, x: _augmented_rhs_post(t, x, aug, tau_env, cap_fn),
        t_span=(0.0, t_end),
        y0=x0_post,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
        max_step=min(2.0, scenario.T_realloc / 4.0) if scenario.T_realloc > 0 else 2.0,
    )
    if not sol.success:
        raise RuntimeError(f"Mean trajectory ODE failed: {sol.message}")
    x_mean = sol.y.T

    # Variance ODE: same equivalent-W as wcfdi_transient, but built from
    # the *calibrated* P6 so the propagated envelope inherits the measured
    # magnitudes.
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

    P0 = lift_intact_cov_to_augmented(P6)

    def rhs_P(t, P_flat):
        P = P_flat.reshape(12, 12)
        Pdot = aug.A @ P + P @ aug.A.T + BWBT_aug
        return Pdot.flatten()

    sol_P = solve_ivp(
        fun=rhs_P,
        t_span=(0.0, t_end),
        y0=P0.flatten(),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-5,
        atol=1e-9,
    )
    if not sol_P.success:
        raise RuntimeError(f"Covariance ODE failed: {sol_P.message}")
    P_t = sol_P.y.T.reshape(n_t, 12, 12)
    P_t = 0.5 * (P_t + P_t.transpose(0, 2, 1))

    eta_mean = x_mean[:, 0:3]
    eta_std = np.sqrt(np.maximum(np.diagonal(P_t[:, 0:3, 0:3], axis1=1, axis2=2), 0.0))

    # Bistability score (same recipe as wcfdi_transient)
    K_tau = np.zeros((3, 12))
    K_tau[:, 0:3] = -aug.Kp
    K_tau[:, 3:6] = -aug.Kd
    K_tau[:, 6:9] = -np.eye(3)
    tau_cmd_mean = (K_tau @ x_mean.T).T
    tau_cmd_var = np.einsum("ij,tjk,lk->til", K_tau, P_t, K_tau)
    sigma_tau_cmd = np.sqrt(
        np.maximum(np.diagonal(tau_cmd_var, axis1=1, axis2=2), 0.0)
    )
    cap_t = np.array([scenario.cap_at_time(float(tt), cfg) for tt in t_eval])
    headroom = cap_t - np.abs(tau_cmd_mean)
    severity_t = np.maximum(0.0, -headroom) / np.maximum(sigma_tau_cmd, 1e-9)
    bistability_per_dof = severity_t.max(axis=0)
    bistability_risk_score = float(bistability_per_dof.max())

    info = {
        "tau_env": tau_env,
        "tau_env_source": "measured",
        "x0_intact": x0,
        "x0_post": x0_post,
        "delta_tau_step": delta_tau,
        "tau_cap_post": cap_post,
        "tau_cap_immediate": cap_immediate,
        "T_realloc": scenario.T_realloc,
        "P0_eta_diag": np.sqrt(np.maximum(np.diag(P6), 0.0)),
        "cqa_precondition_violated": cqa_violated,
        "bistability_per_dof": bistability_per_dof,
        "bistability_risk_score": bistability_risk_score,
        "tau_cmd_mean": tau_cmd_mean,
        "sigma_tau_cmd": sigma_tau_cmd,
        "cap_t": cap_t,
        "calibration": {
            "sigma_measured_lf_body": ctx.sigma_measured_lf_body,
            "sigma_model_lf_body": ctx.sigma_model_lf_body,
            "ratio_per_dof": (
                ctx.sigma_measured_lf_body
                / np.where(ctx.sigma_model_lf_body > 0, ctx.sigma_model_lf_body, 1.0)
            ),
        },
    }
    return TransientResult(
        t=t_eval,
        x_mean=x_mean,
        P=P_t,
        eta_mean=eta_mean,
        eta_std=eta_std,
        info=info,
    )


# ---------------------------------------------------------------------------
# Calibrated MC over starting states
# ---------------------------------------------------------------------------


def wcfdi_mc_calibrated(
    cfg: CqaConfig,
    scenario: WcfdiScenario,
    joint: GangwayJointState,
    ctx: CalibratedContext,
    *,
    n_samples: int = 500,
    t_end: float = 200.0,
    n_t: int = 201,
    rng_seed: Optional[int] = 0,
    sample_mode: str = "eta_nu",
) -> WcfdiMcResult:
    """Monte-Carlo over WCF starting state with measurement-calibrated context.

    Mirrors :func:`cqa.wcfdi_mc.wcfdi_mc` but draws starting states from
    the calibrated ``P6`` / ``P12`` and uses the calibrated ``tau_env``
    and ``x_ss_intact`` from ``ctx``.

    Parameters
    ----------
    cfg, scenario, joint
        Same semantics as :func:`wcfdi_mc`.
    ctx : CalibratedContext
        Pre-built measurement-calibrated context (see
        :func:`build_calibrated_context`).
    n_samples, t_end, n_t, rng_seed, sample_mode
        Same semantics as :func:`wcfdi_mc`.
    """
    if sample_mode not in ("eta_nu", "full12"):
        raise ValueError(f"Unknown sample_mode: {sample_mode!r}")

    aug = ctx.aug
    tau_env = ctx.tau_env_used
    x_ss_intact = ctx.x_ss_intact
    P6 = ctx.P6_calibrated
    P12 = ctx.P12_calibrated

    if scenario.T_thr_post is not None:
        T_b = cfg.controller.bias_time_constant_s
        aug = build_augmented_system(
            ctx.vessel, ctx.controller, T_b=T_b, T_thr=scenario.T_thr_post
        )

    cap_post = scenario.resolved_cap_post(cfg)
    cap_immediate = scenario.resolved_cap_immediate(cfg)
    cqa_violated = np.abs(tau_env) > cap_post

    c_L = telescope_sensitivity(joint, cfg.gangway)
    L0 = joint.L
    cap_fn = lambda t: scenario.cap_at_time(t, cfg)

    rng = np.random.default_rng(rng_seed)
    if sample_mode == "eta_nu":
        eigvals, eigvecs = np.linalg.eigh(P6)
        eigvals = np.maximum(eigvals, 0.0)
        L6 = eigvecs @ np.diag(np.sqrt(eigvals))
        z = rng.standard_normal((n_samples, 6))
        delta_eta_nu = z @ L6.T
        delta_x = np.zeros((n_samples, 12))
        delta_x[:, 0:6] = delta_eta_nu
    else:  # full12
        eigvals, eigvecs = np.linalg.eigh(P12)
        eigvals = np.maximum(eigvals, 0.0)
        L12 = eigvecs @ np.diag(np.sqrt(eigvals))
        z = rng.standard_normal((n_samples, 12))
        delta_x = z @ L12.T

    t_eval = np.linspace(0.0, t_end, n_t)
    L_traj = np.zeros((n_samples, n_t))
    dL_peak = np.zeros(n_samples)
    dL_peak_abs = np.zeros(n_samples)
    x0_samples = np.zeros((n_samples, 12))
    margin_low = np.zeros(n_samples)
    margin_high = np.zeros(n_samples)
    operable = np.zeros(n_samples, dtype=bool)
    pos_base_traj = np.zeros((n_samples, n_t))
    pos_peak = np.zeros(n_samples)

    L_min = cfg.gangway.telescope_min
    L_max = cfg.gangway.telescope_max
    base_x_b, base_y_b, _ = cfg.gangway.base_position_body

    n_failed = 0
    for i in range(n_samples):
        x0_post = x_ss_intact.copy() + delta_x[i]
        x0_post[9:12] = _clip_per_dof(x0_post[9:12], cap_immediate)
        x0_samples[i] = delta_x[i]

        sol = solve_ivp(
            fun=lambda t, x: _augmented_rhs_post(t, x, aug, tau_env, cap_fn),
            t_span=(0.0, t_end),
            y0=x0_post,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
            max_step=min(2.0, scenario.T_realloc / 4.0) if scenario.T_realloc > 0 else 2.0,
        )
        if not sol.success:
            n_failed += 1
            L_traj[i] = np.nan
            dL_peak[i] = np.nan
            dL_peak_abs[i] = np.nan
            pos_base_traj[i] = np.nan
            pos_peak[i] = np.nan
            continue

        eta_t = sol.y[0:3, :].T
        dL_t = eta_t @ c_L
        L_t = L0 + dL_t
        L_traj[i] = L_t

        idx_peak = int(np.argmax(np.abs(dL_t)))
        dL_peak[i] = float(dL_t[idx_peak])
        dL_peak_abs[i] = float(np.abs(dL_t[idx_peak]))

        margin_low[i] = float(np.min(L_t - L_min))
        margin_high[i] = float(np.min(L_max - L_t))
        operable[i] = (margin_low[i] > 0.0) and (margin_high[i] > 0.0)

        dp_n = eta_t[:, 0] - eta_t[:, 2] * base_y_b
        dp_e = eta_t[:, 1] + eta_t[:, 2] * base_x_b
        pos_t = np.sqrt(dp_n ** 2 + dp_e ** 2)
        pos_base_traj[i] = pos_t
        pos_peak[i] = float(np.max(pos_t))

    # Linearised baseline -- now using the calibrated transient.
    lin = wcfdi_transient_calibrated(
        cfg, scenario, ctx, t_end=t_end, n_t=n_t,
    )
    dL_lin_mean = lin.eta_mean @ c_L
    dL_lin_std = np.sqrt(np.maximum(np.einsum("i,nij,j->n", c_L, lin.P[:, 0:3, 0:3], c_L), 0.0))
    L_mean_linear = L0 + dL_lin_mean
    L_std_linear = dL_lin_std

    info = {
        "n_samples": n_samples,
        "n_failed": n_failed,
        "tau_env": tau_env,
        "tau_env_source": "measured",
        "tau_cap_post": cap_post,
        "tau_cap_immediate": cap_immediate,
        "T_realloc": scenario.T_realloc,
        "cqa_precondition_violated": cqa_violated,
        "P6_intact": P6,
        "P12_intact": P12,
        "c_L": c_L,
        "L0": L0,
        "joint": joint,
        "sample_mode": sample_mode,
        "calibration": {
            "sigma_measured_lf_body": ctx.sigma_measured_lf_body,
            "sigma_model_lf_body": ctx.sigma_model_lf_body,
            "ratio_per_dof": (
                ctx.sigma_measured_lf_body
                / np.where(ctx.sigma_model_lf_body > 0, ctx.sigma_model_lf_body, 1.0)
            ),
        },
    }

    return WcfdiMcResult(
        t=t_eval,
        L_traj=L_traj,
        dL_peak=dL_peak,
        dL_peak_abs=dL_peak_abs,
        x0_samples=x0_samples,
        margin_low=margin_low,
        margin_high=margin_high,
        operable=operable,
        pos_base_traj=pos_base_traj,
        pos_peak=pos_peak,
        L_mean_linear=L_mean_linear,
        L_std_linear=L_std_linear,
        info=info,
    )


__all__ = [
    "rescale_covariance_diagonal",
    "CalibratedContext",
    "build_calibrated_context",
    "wcfdi_transient_calibrated",
    "wcfdi_mc_calibrated",
]
