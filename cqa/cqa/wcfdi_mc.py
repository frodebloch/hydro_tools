"""Monte-Carlo over WCF starting state.

Goal: quantify how the realised post-WCFDI peak excursion depends on the
vessel's intact-state starting point at the WCF instant. The static CQA
(and our linearised P2 transient) implicitly assume the WCF hits when the
vessel is exactly at the DP setpoint with zero velocity. In reality the
vessel is somewhere inside the intact 95 % footprint with some velocity,
and the worst-case realisation of the post-WCF peak depends strongly on
the realised (eta, nu) at t=0-.

This first version samples ONLY (eta, nu) at t=0- from the intact 6x6
position+velocity covariance and propagates the deterministic post-WCF
mean ODE for each draw. The bias-estimate b_hat and the thruster output
tau_thr are kept at their intact deterministic steady-state values
(b_hat = +tau_env, tau_thr = -tau_env). This is the cleanest first cut:

  - decouples the "where am I, where am I going" effect from
    "what is my integrator doing" and "what is my thrust output";
  - matches how an external observer or a class-society reviewer would
    typically frame the question;
  - leaves the cross-effects (sampling b_hat, tau_thr, or injecting
    slow disturbances during recovery) as documented next-steps that can
    be added one at a time and quantified against this baseline.

Outputs (per operating point):
  - empirical CDF of peak |Delta_L| over the recovery window;
  - per-time percentile bands on L(t) (50, 95, 99 %);
  - sensitivity table: regression of peak |Delta_L| on the whitened
    components of x(0-), giving per-sigma contribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
from scipy.integrate import solve_ivp

from .config import CqaConfig
from .vessel import LinearVesselModel, WindForceModel, CurrentForceModel
from .controller import LinearDpController
from .closed_loop import ClosedLoop, state_covariance_freqdomain, state_covariance_freqdomain_general
from .psd import (
    npd_wind_gust_force_psd,
    slow_drift_force_psd_newman,
    current_variability_force_psd,
)
from .transient import (
    AugmentedSystem,
    build_augmented_system,
    intact_mean_steady_state,
    WcfdiScenario,
    _augmented_rhs_post,
    _clip_per_dof,
)
from .gangway import GangwayJointState, telescope_sensitivity, GangwayConfig


# ---------------------------------------------------------------------------
# MC result container
# ---------------------------------------------------------------------------


@dataclass
class WcfdiMcResult:
    """Result of a Monte-Carlo over WCF starting states at one operating point."""

    t: np.ndarray                  # (N_t,) time grid [s]
    L_traj: np.ndarray             # (N_samples, N_t) telescope length L(t) per sample [m]
    dL_peak: np.ndarray            # (N_samples,) signed peak Delta_L per sample [m]
    dL_peak_abs: np.ndarray        # (N_samples,) abs peak Delta_L per sample [m]
    x0_samples: np.ndarray         # (N_samples, 12) starting-state perturbation
    #                                in augmented order (eta_n,e,psi, u,v,r,
    #                                b_x,b_y,b_n, tau_thr_x,thr_y,thr_n).
    #                                Columns 6..11 are zero when sample_mode='eta_nu'.

    # Operability gating outputs:
    margin_low: np.ndarray         # (N_samples,) min over time of L - L_min
    margin_high: np.ndarray        # (N_samples,) min over time of L_max - L
    operable: np.ndarray           # (N_samples,) bool: stayed inside [L_min, L_max] all the time

    # Vessel position deviation at the gangway base point (horizontal,
    # vessel-body frame; surge/sway components combined into a magnitude).
    # Used by the operator-view "vessel position" P_exceed number,
    # separate from the gangway telescope number.
    pos_base_traj: np.ndarray      # (N_samples, N_t) |Delta_p_base|(t) [m]
    pos_peak: np.ndarray           # (N_samples,) max over time of |Delta_p_base| [m]

    # Linearised baseline for comparison:
    L_mean_linear: np.ndarray      # (N_t,) linearised mean L(t) (deterministic, x0=0)
    L_std_linear: np.ndarray       # (N_t,) linearised 1-sigma envelope width

    info: dict = field(default_factory=dict)

    # Convenience
    def percentile_bands_L(self, qs: tuple[float, ...] = (5, 25, 50, 75, 95, 99)) -> dict:
        return {q: np.percentile(self.L_traj, q, axis=0) for q in qs}

    def cdf_peak_abs(self) -> tuple[np.ndarray, np.ndarray]:
        s = np.sort(self.dL_peak_abs)
        F = np.arange(1, len(s) + 1) / len(s)
        return s, F

    def operable_fraction(self) -> float:
        return float(np.mean(self.operable))


# ---------------------------------------------------------------------------
# Core: build the operating-point context (vessel, controller, env, P0)
# ---------------------------------------------------------------------------


def _build_operating_context(
    cfg: CqaConfig,
    Vw_mean: float,
    Hs: float,
    Tp: float,
    Vc: float,
    theta_rel: float,
    omega_n: Optional[tuple[float, float, float]] = None,
    zeta: Optional[tuple[float, float, float]] = None,
    T_b: Optional[float] = None,
    T_thr: Optional[float] = None,
    sigma_Vc: float = 0.1,
    tau_Vc: float = 600.0,
    rao_table=None,
):
    """Build the linear vessel/controller, augmented system, environment force,
    intact P0 covariance, and intact mean steady-state augmented vector.

    Returns a dict; same construction logic as `wcfdi_transient` but
    refactored so we can reuse it for the MC sweep without duplicating.

    Controller / observer tuning (`omega_n`, `zeta`, `T_b`, `T_thr`)
    defaults to ``cfg.controller`` (single source of truth shared with
    the intact-prior pipeline). Pass explicit overrides only when you
    really mean to deviate from the configured tuning.

    If ``rao_table`` is supplied, the parametric mean-drift force and
    parametric Newman slow-drift PSD are replaced with the pdstrip
    diagonal-QTF versions
    (:func:`cqa.drift.mean_drift_force_pdstrip` and
    :func:`cqa.drift.slow_drift_force_psd_newman_pdstrip`). The
    parametric ``cfg.wave_drift`` parameters are then unused.
    """
    vp = cfg.vessel
    wp = cfg.wind
    cp = cfg.current
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

    # Mean environmental force at this direction
    wind_model = WindForceModel(wp=wp, loa=vp.loa)
    lateral_uw = vp.lpp * vp.draft
    frontal_uw = vp.beam * vp.draft
    current_model = CurrentForceModel(
        cp=cp,
        lateral_area_underwater=lateral_uw,
        frontal_area_underwater=frontal_uw,
        loa=vp.loa,
    )
    F_wind = wind_model.force(Vw_mean, theta_rel)
    F_curr = current_model.force(Vc, theta_rel)
    if rao_table is not None:
        # pdstrip-driven Pinkster mean drift (and PSD below).
        from .drift import (
            mean_drift_force_pdstrip,
            slow_drift_force_psd_newman_pdstrip,
        )
        F_drift = mean_drift_force_pdstrip(
            rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel
        )
    else:
        F_drift = np.array(
            [
                wd.drift_x_amp * Hs ** 2 * np.cos(theta_rel),
                wd.drift_y_amp * Hs ** 2 * np.sin(theta_rel),
                wd.drift_n_amp * Hs ** 2 * np.sin(2.0 * theta_rel),
            ]
        )
    tau_env = F_wind + F_curr + F_drift

    # Intact mean steady-state augmented vector
    x_ss_intact = intact_mean_steady_state(aug, tau_env)

    # Intact 6x6 closed-loop covariance (eta, nu)
    cl_intact = ClosedLoop.build(vessel, controller)
    S_wind = npd_wind_gust_force_psd(wind_model, Vw_mean, theta_rel)
    if rao_table is not None:
        S_drift = slow_drift_force_psd_newman_pdstrip(
            rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel
        )
    else:
        S_drift = slow_drift_force_psd_newman(
            (wd.drift_x_amp, wd.drift_y_amp, wd.drift_n_amp), Hs, Tp, theta_rel
        )
    if Vc > 1e-9:
        dFdVc = 2.0 * F_curr / Vc
    else:
        dFdVc = np.zeros(3)
    S_curr = current_variability_force_psd(dFdVc, sigma_Vc=sigma_Vc, tau=tau_Vc)
    P6 = state_covariance_freqdomain(cl_intact, [S_wind, S_drift, S_curr])

    # 12x12 intact augmented covariance: same disturbances, but lifted into
    # the augmented system. Wind/drift/current force noise enters via the
    # nu channel only; b_hat and tau_thr have no direct stochastic input
    # but acquire variance through closed-loop coupling with eta. This
    # is what gives the WCF starting-state MC a non-trivial spread on the
    # bias estimate and the thruster output at t=0-.
    P12 = state_covariance_freqdomain_general(
        aug.A, aug.B_w, [S_wind, S_drift, S_curr]
    )

    return {
        "vessel": vessel,
        "controller": controller,
        "aug": aug,
        "tau_env": tau_env,
        "x_ss_intact": x_ss_intact,
        "P6_intact": P6,
        "P12_intact": P12,
        "cl_intact": cl_intact,
    }


# ---------------------------------------------------------------------------
# MC entry point
# ---------------------------------------------------------------------------


def wcfdi_mc(
    cfg: CqaConfig,
    Vw_mean: float,
    Hs: float,
    Tp: float,
    Vc: float,
    theta_rel: float,
    scenario: WcfdiScenario,
    joint: GangwayJointState,
    n_samples: int = 500,
    t_end: float = 200.0,
    n_t: int = 201,
    rng_seed: Optional[int] = 0,
    omega_n: Optional[tuple[float, float, float]] = None,
    zeta: Optional[tuple[float, float, float]] = None,
    T_b: Optional[float] = None,
    T_thr: Optional[float] = None,
    sigma_Vc: float = 0.1,
    tau_Vc: float = 600.0,
    sample_mode: str = "eta_nu",
    rao_table=None,
) -> WcfdiMcResult:
    """Monte-Carlo over WCF starting state.

    `sample_mode` controls which components of the augmented intact state
    are randomised at t=0-:

      - ``"eta_nu"``  : sample only (eta, nu) from the 6x6 intact P6;
        b_hat = +tau_env, tau_thr = -tau_env (deterministic). This is the
        cleanest first-cut baseline (decouples 'where am I' from 'what is
        my integrator/thrust doing').

      - ``"full12"``  : sample the entire 12-state intact (eta, nu, b_hat,
        tau_thr) jointly from the 12x12 augmented covariance P12. Centred
        on the deterministic intact mean steady state. This captures the
        fact that the bias estimator and thruster outputs at the WCF
        instant are themselves stochastic (driven by the same wind/drift/
        current process that drives eta and nu), and adds the variance
        contribution from those components to the post-WCF peak.

    For each MC sample:
      - draw the starting-state perturbation per `sample_mode`;
      - clip tau_thr to the immediate post-WCF cap to model the t=0+ step;
      - propagate the deterministic post-WCF augmented ODE for t in [0, t_end];
      - record L(t) = L0 + c^T eta(t).

    The result also contains the linearised baseline (deterministic mean
    trajectory and 1-sigma envelope), computed once, for comparison.
    """
    if sample_mode not in ("eta_nu", "full12"):
        raise ValueError(f"Unknown sample_mode: {sample_mode!r}")

    # Resolve controller/observer time constants once so the T_thr_post
    # override path below uses the same values as the intact build.
    cp_ctrl = cfg.controller
    if T_b is None:
        T_b = cp_ctrl.bias_time_constant_s
    if T_thr is None:
        T_thr = cp_ctrl.thruster_time_constant_s

    ctx = _build_operating_context(
        cfg, Vw_mean, Hs, Tp, Vc, theta_rel,
        omega_n=omega_n, zeta=zeta, T_b=T_b, T_thr=T_thr,
        sigma_Vc=sigma_Vc, tau_Vc=tau_Vc, rao_table=rao_table,
    )
    aug: AugmentedSystem = ctx["aug"]
    tau_env: np.ndarray = ctx["tau_env"]
    x_ss_intact: np.ndarray = ctx["x_ss_intact"]
    P6: np.ndarray = ctx["P6_intact"]
    P12: np.ndarray = ctx["P12_intact"]

    # Apply T_thr_post override if requested
    if scenario.T_thr_post is not None:
        aug = build_augmented_system(
            ctx["vessel"], ctx["controller"], T_b=T_b, T_thr=scenario.T_thr_post
        )

    cap_post = scenario.resolved_cap_post(cfg)
    cap_immediate = scenario.resolved_cap_immediate(cfg)
    cqa_violated = np.abs(tau_env) > cap_post

    # Telescope sensitivity (3-vector)
    c_L = telescope_sensitivity(joint, cfg.gangway)
    L0 = joint.L

    # Time-varying cap
    cap_fn: Callable[[float], np.ndarray] = lambda t: scenario.cap_at_time(t, cfg)

    # MC sampling. Build a 12-D perturbation vector per sample.
    rng = np.random.default_rng(rng_seed)
    if sample_mode == "eta_nu":
        # Cholesky-style factorisation of P6
        eigvals, eigvecs = np.linalg.eigh(P6)
        eigvals = np.maximum(eigvals, 0.0)
        L6 = eigvecs @ np.diag(np.sqrt(eigvals))
        z = rng.standard_normal((n_samples, 6))
        delta_eta_nu = z @ L6.T  # (n_samples, 6)
        delta_x = np.zeros((n_samples, 12))
        delta_x[:, 0:6] = delta_eta_nu
    else:  # full12
        eigvals, eigvecs = np.linalg.eigh(P12)
        eigvals = np.maximum(eigvals, 0.0)
        L12 = eigvecs @ np.diag(np.sqrt(eigvals))
        z = rng.standard_normal((n_samples, 12))
        delta_x = z @ L12.T  # (n_samples, 12)

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

    # Body-frame horizontal coordinates of the gangway base point. The
    # vessel-frame deviation of the base, for small heading psi, is
    #     Dp_base_surge = eta_n - psi * y_b
    #     Dp_base_sway  = eta_e + psi * x_b
    # (eta_n, eta_e are the setpoint-aligned surge/sway offsets under the
    # small-heading linearisation; "north" here is the setpoint surge
    # direction, "east" the setpoint sway direction.)
    base_x_b, base_y_b, _ = cfg.gangway.base_position_body

    n_failed = 0
    for i in range(n_samples):
        x0_post = x_ss_intact.copy() + delta_x[i]
        # Step thrust clipping at t=0+ (immediate cap, before realloc).
        # This applies regardless of sample_mode: even if we sampled a
        # tau_thr(0-) inside the intact cap, post-WCF we clip to the
        # smaller immediate cap.
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

        eta_t = sol.y[0:3, :].T  # (n_t, 3)
        dL_t = eta_t @ c_L
        L_t = L0 + dL_t
        L_traj[i] = L_t

        # signed peak = the dL with the largest absolute value
        idx_peak = int(np.argmax(np.abs(dL_t)))
        dL_peak[i] = float(dL_t[idx_peak])
        dL_peak_abs[i] = float(np.abs(dL_t[idx_peak]))

        margin_low[i] = float(np.min(L_t - L_min))
        margin_high[i] = float(np.min(L_max - L_t))
        operable[i] = (margin_low[i] > 0.0) and (margin_high[i] > 0.0)

        # Earth-frame horizontal deviation of the gangway base point
        dp_n = eta_t[:, 0] - eta_t[:, 2] * base_y_b
        dp_e = eta_t[:, 1] + eta_t[:, 2] * base_x_b
        pos_t = np.sqrt(dp_n ** 2 + dp_e ** 2)
        pos_base_traj[i] = pos_t
        pos_peak[i] = float(np.max(pos_t))

    # Linearised baseline: deterministic mean trajectory + 1-sigma envelope
    # (re-use existing linearised analysis; we just want it for the plot)
    from .transient import wcfdi_transient
    lin = wcfdi_transient(
        cfg, Vw_mean, Hs, Tp, Vc, theta_rel, scenario,
        omega_n=omega_n, zeta=zeta, T_b=T_b, T_thr=T_thr,
        sigma_Vc=sigma_Vc, tau_Vc=tau_Vc,
        t_end=t_end, n_t=n_t,
    )
    dL_lin_mean = lin.eta_mean @ c_L
    dL_lin_std = np.sqrt(np.maximum(np.einsum("i,nij,j->n", c_L, lin.P[:, 0:3, 0:3], c_L), 0.0))
    L_mean_linear = L0 + dL_lin_mean
    L_std_linear = dL_lin_std

    info = {
        "n_samples": n_samples,
        "n_failed": n_failed,
        "tau_env": tau_env,
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


# ---------------------------------------------------------------------------
# Sensitivity analysis: which starting-state component drives the peak?
# ---------------------------------------------------------------------------


def starting_state_sensitivity(
    res: WcfdiMcResult,
) -> dict:
    """Linear regression of dL_peak (signed) on whitened starting-state components.

    Operates on the active components of `res.x0_samples`: when
    `sample_mode='eta_nu'`, columns 6..11 are zero and are dropped from
    the regression (they would otherwise produce a singular design).

    Returns a dict with:
      - 'labels': list of component labels in augmented order, restricted
        to the active subset.
      - 'sigmas': starting-state per-component sigmas (active subset).
      - 'beta': regression coefficients in original units (m peak per unit
        of each component).
      - 'beta_per_sigma': = beta * sigmas, units of m peak per 1-sigma
        input. The operator-friendly diagnostic.
      - 'r2': coefficient of determination of the linear fit.

    Note: a linear fit ignores the saturation nonlinearity in the post-WCF
    propagation. The regression coefficients are an *average* sensitivity
    across the MC sample.
    """
    all_labels = [
        "surge offset [m]", "sway offset [m]", "heading offset [rad]",
        "surge vel [m/s]", "sway vel [m/s]", "yaw rate [rad/s]",
        "bias surge [N]", "bias sway [N]", "bias yaw [Nm]",
        "thrust surge [N]", "thrust sway [N]", "thrust yaw [Nm]",
    ]
    valid = ~np.isnan(res.dL_peak)
    X_all = res.x0_samples[valid]  # (N_valid, 12)
    y = res.dL_peak[valid]

    # Active columns: those with non-zero variance across samples
    col_var = np.var(X_all, axis=0)
    active = col_var > 1e-30
    X = X_all[:, active]
    labels = [lbl for lbl, a in zip(all_labels, active) if a]

    # Per-component sigma from the underlying covariance (P12 if available,
    # else P6 padded with zeros). Use the empirical sigma as fallback.
    P12 = res.info.get("P12_intact")
    P6 = res.info.get("P6_intact")
    sigmas_full = np.zeros(12)
    if P12 is not None:
        sigmas_full = np.sqrt(np.maximum(np.diag(P12), 0.0))
    elif P6 is not None:
        sigmas_full[:6] = np.sqrt(np.maximum(np.diag(P6), 0.0))
    # Override with empirical sigma where the analytical value is zero but
    # we did sample (defensive)
    emp = np.sqrt(col_var)
    sigmas_full = np.where(sigmas_full > 0, sigmas_full, emp)
    sigmas = sigmas_full[active]

    # Whitened design matrix for the per-sigma diagnostic
    X_white = X / np.where(sigmas > 0, sigmas, 1.0)

    # Linear least squares (no intercept; deterministic mean drift is
    # already captured by the linearised baseline)
    beta_white, *_ = np.linalg.lstsq(X_white, y, rcond=None)
    beta = beta_white / np.where(sigmas > 0, sigmas, 1.0)
    y_hat = X_white @ beta_white
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "labels": labels,
        "sigmas": sigmas,
        "beta": beta,
        "beta_per_sigma": beta_white,
        "r2": r2,
    }
