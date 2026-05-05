"""Time-domain self-MC validator for the linearised WCFDI predictor.

Validates `cqa.transient.wcfdi_transient` (the augmented-state linear
predictor of post-WCFDI mean and covariance trajectories) against a
stochastic time-domain Monte-Carlo built from the *same* augmented
linear matrices, using Shinozuka realisations of the disturbance PSDs.

What this validates
-------------------
1. The augmented 12-state structure (eta, nu, b_hat, tau_thr).
2. The deterministic mean-trajectory ODE under the time-varying per-DOF
   thrust cap (`scenario.cap_at_time`).
3. The covariance-trajectory ODE
   ``dP/dt = A P + P A^T + B_w W_eq B_w^T``
   matches the empirical covariance over independent realisations.
4. The pre-failure intact steady-state statistics
   (zero-mean eta, nu; deterministic b_hat, tau_thr) used as initial
   condition at t=0.

What this does NOT validate
---------------------------
The MC and the linear predictor share the same augmented A matrices and
the same constant-intensity equivalent white-noise approximation
(W_eq). It is therefore *not* a check against a higher-fidelity
nonlinear vessel simulator. That cross-check should be done against
the brucon `vessel_simulator` library (e.g. driven by
`dp_runfast_simulator` Lua scripts -- the Lua harness wraps the same
C++ engine, so both paths produce the same physics) and is a separate
work-stream.

Strategy
--------
Per scenario:
  1. Pre-realise a single long disturbance force time series ``F(t)``
     of total length ``t_warm + t_end`` using
     ``realise_vector_force_time_series`` summed over the wind, slow-
     drift, and current-variability PSDs at the operating point.
  2. Integrate the *intact* augmented ODE (no clipping) from
     ``x = 0`` for the warm-up window ``[-t_warm, 0]``. By the end of
     the warm-up the state distribution has converged to the intact
     stationary distribution.
  3. At ``t=0`` apply the WCFDI step: clip ``tau_thr`` to the
     immediate cap ``gamma_immediate * cap_intact`` (consistent with
     `wcfdi_transient`).
  4. Integrate the *post-failure* augmented ODE (with the time-varying
     per-DOF cap) from ``t=0`` to ``t=t_end``, recording the augmented
     state on the linearised predictor's time grid.
  5. Repeat M times with independent random seeds; compute empirical
     mean and std of ``eta(t)`` per DOF and compare to the linearised
     prediction.

Calibration metrics
-------------------
For each of the 3 eta DOFs we report:
  * ``mean_rms``: RMS over t of ``|eta_mean_emp - eta_mean_lin|``,
    normalised by ``max_t |eta_mean_lin|``.
  * ``std_rms``: same for the time-varying std.
  * ``mean_max_abs``: max over t of the same residual.

A cell is considered "calibrated" when both relative residuals fall
below a tolerance (default 0.15 for both, allowing ~15 % discrepancy
attributable to W_eq approximation and finite M).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from .config import CqaConfig
from .vessel import LinearVesselModel, WindForceModel, CurrentForceModel
from .controller import LinearDpController
from .closed_loop import ClosedLoop, state_covariance_freqdomain
from .psd import (
    npd_wind_gust_force_psd,
    slow_drift_force_psd_newman,
    current_variability_force_psd,
)
from .time_series_realisation import realise_vector_force_time_series
from .transient import (
    AugmentedSystem,
    build_augmented_system,
    intact_mean_steady_state,
    WcfdiScenario,
    _clip_per_dof,
    wcfdi_transient,
    TransientResult,
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class WcfdiSelfMcResult:
    """Per-cell self-MC validation result.

    Attributes
    ----------
    t : (N_t,) post-failure time grid [s], identical to the linear
        predictor's grid.
    eta_mean_emp : (N_t, 3) empirical mean trajectory.
    eta_std_emp  : (N_t, 3) empirical standard deviation.
    eta_mean_lin : (N_t, 3) linearised predictor mean.
    eta_std_lin  : (N_t, 3) linearised predictor std.
    n_realisations : number of MC seeds actually used.
    operating_point : dict echoing (Vw_mean, Hs, Tp, Vc, theta_rel,
        alpha).
    metrics : dict per-DOF calibration metrics (see module docstring).
    info : misc diagnostic info (warm-up params, dt, etc).
    """

    t: np.ndarray
    eta_mean_emp: np.ndarray
    eta_std_emp: np.ndarray
    eta_mean_lin: np.ndarray
    eta_std_lin: np.ndarray
    n_realisations: int
    operating_point: dict
    metrics: dict
    info: dict = field(default_factory=dict)
    eta_realisations: Optional[np.ndarray] = None
    """(n_realisations, n_t, 3) per-realisation eta trajectories, or
    None unless the call passed ``return_realisations=True``. Memory:
    ``8 * n_realisations * n_t * 3`` bytes (e.g. ~1.2 MB for M=128,
    n_t=401)."""


# ---------------------------------------------------------------------------
# Internal: build the disturbance PSD callables
# ---------------------------------------------------------------------------


def _build_disturbance_psd_funcs(
    cfg: CqaConfig,
    Vw_mean: float,
    Hs: float,
    Tp: float,
    Vc: float,
    theta_rel: float,
    sigma_Vc: float = 0.1,
    tau_Vc: float = 600.0,
) -> tuple[list[Callable[[float], np.ndarray]], np.ndarray]:
    """Return (S_F_funcs, tau_env) at the operating point.

    Mirrors the construction in `wcfdi_transient` so the self-MC
    drives the ODE with the same PSDs and the same mean force as the
    linearised predictor.
    """
    vp = cfg.vessel
    wp = cfg.wind
    cp = cfg.current
    wd = cfg.wave_drift

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
    F_drift = np.array(
        [
            wd.drift_x_amp * Hs ** 2 * np.cos(theta_rel),
            wd.drift_y_amp * Hs ** 2 * np.sin(theta_rel),
            wd.drift_n_amp * Hs ** 2 * np.sin(2.0 * theta_rel),
        ]
    )
    tau_env = F_wind + F_curr + F_drift

    S_wind = npd_wind_gust_force_psd(wind_model, Vw_mean, theta_rel)
    S_drift = slow_drift_force_psd_newman(
        (wd.drift_x_amp, wd.drift_y_amp, wd.drift_n_amp), Hs, Tp, theta_rel
    )
    if Vc > 1e-9:
        dFdVc = 2.0 * F_curr / Vc
    else:
        dFdVc = np.zeros(3)
    S_curr = current_variability_force_psd(dFdVc, sigma_Vc=sigma_Vc, tau=tau_Vc)
    return [S_wind, S_drift, S_curr], tau_env


# ---------------------------------------------------------------------------
# Internal: ZOH integration of the augmented ODE driven by a force series
# ---------------------------------------------------------------------------


def _integrate_augmented_intact_zoh(
    aug: AugmentedSystem,
    F_t: np.ndarray,
    tau_env: np.ndarray,
    t: np.ndarray,
    x0: np.ndarray,
) -> np.ndarray:
    """Integrate the intact augmented ODE under a ZOH input force.

    State: x in R^12 with layout [eta(3), nu(3), b_hat(3), tau_thr(3)].
    ODE:   x_dot = A x + B_d tau_env + B_w (F(t)).

    Used for the intact warm-up phase. The intact closed loop is fully
    linear (no clipping) at the CQA-feasible operating points.

    Returns: (12, N_t) state trajectory.
    """
    A = aug.A
    Bw = aug.B_w
    Bd = aug.B_d
    N_t = t.size
    dt = float(t[1] - t[0])
    Phi = expm(A * dt)
    # ZOH input gain for a piecewise-constant input u_k:
    # x_{k+1} = Phi x_k + Gamma_w F_k + Gamma_d tau_env
    I = np.eye(A.shape[0])
    # Use solve(A, .) for stability when A is well-conditioned.
    try:
        Gamma_w = np.linalg.solve(A, (Phi - I) @ Bw)
        Gamma_d_tau = np.linalg.solve(A, (Phi - I) @ Bd) @ tau_env
    except np.linalg.LinAlgError:
        # Fallback: explicit integral via series expansion (rarely needed).
        Gamma_w = dt * Bw
        Gamma_d_tau = dt * Bd @ tau_env

    x = np.zeros((12, N_t))
    x[:, 0] = x0
    x_k = x0.copy()
    for k in range(N_t - 1):
        x_k = Phi @ x_k + Gamma_w @ F_t[:, k] + Gamma_d_tau
        x[:, k + 1] = x_k
    return x


def _integrate_augmented_post_clipped(
    aug: AugmentedSystem,
    F_t: np.ndarray,
    tau_env: np.ndarray,
    t: np.ndarray,
    x0: np.ndarray,
    cap_fn: Callable[[float], np.ndarray],
) -> np.ndarray:
    """Integrate the post-WCFDI augmented ODE with per-DOF thrust clipping.

    Uses RK4 (fixed step) for compactness and speed. The disturbance
    force time series is treated as ZOH between samples.

    State layout per row of `aug.A`:
        eta(0:3), nu(3:6), b_hat(6:9), tau_thr(9:12)

    Returns: (12, N_t) state trajectory.
    """
    Minv_D = aug.A[3:6, 3:6]   # = -M^-1 D
    Minv = aug.B_w[3:6, :]     # = M^-1
    Kp = aug.Kp
    Kd = aug.Kd
    T_b = aug.T_b
    T_thr = aug.T_thr
    N_t = t.size
    dt = float(t[1] - t[0])

    def rhs(t_now: float, x: np.ndarray, F_now: np.ndarray) -> np.ndarray:
        eta = x[0:3]
        nu = x[3:6]
        b_hat = x[6:9]
        tau_thr = x[9:12]
        cap_now = cap_fn(t_now)
        eta_dot = nu
        nu_dot = Minv_D @ nu + Minv @ tau_thr + Minv @ (tau_env + F_now)
        b_hat_dot = (1.0 / T_b) * (Kp @ eta)
        tau_cmd = -Kp @ eta - Kd @ nu - b_hat
        tau_cmd_clipped = _clip_per_dof(tau_cmd, cap_now)
        tau_thr_dot = (1.0 / T_thr) * (tau_cmd_clipped - tau_thr)
        out = np.empty(12)
        out[0:3] = eta_dot
        out[3:6] = nu_dot
        out[6:9] = b_hat_dot
        out[9:12] = tau_thr_dot
        return out

    x = np.zeros((12, N_t))
    x[:, 0] = x0
    x_k = x0.copy()
    for k in range(N_t - 1):
        F_k = F_t[:, k]  # ZOH on the force input
        t_k = t[k]
        # RK4
        k1 = rhs(t_k, x_k, F_k)
        k2 = rhs(t_k + 0.5 * dt, x_k + 0.5 * dt * k1, F_k)
        k3 = rhs(t_k + 0.5 * dt, x_k + 0.5 * dt * k2, F_k)
        k4 = rhs(t_k + dt, x_k + dt * k3, F_k)
        x_k = x_k + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x[:, k + 1] = x_k
    return x


# ---------------------------------------------------------------------------
# Public API: per-cell self-MC
# ---------------------------------------------------------------------------


def wcfdi_self_mc(
    cfg: CqaConfig,
    Vw_mean: float,
    Hs: float,
    Tp: float,
    Vc: float,
    theta_rel: float,
    scenario: WcfdiScenario,
    *,
    n_realisations: int = 64,
    t_end: float = 200.0,
    n_t: int = 401,
    t_warm: float = 600.0,
    omega_grid: Optional[np.ndarray] = None,
    sigma_Vc: float = 0.1,
    tau_Vc: float = 600.0,
    seed=None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    return_realisations: bool = False,
) -> WcfdiSelfMcResult:
    """Run the time-domain self-MC at a single operating point.

    Parameters
    ----------
    cfg, Vw_mean, Hs, Tp, Vc, theta_rel, scenario : passed straight
        through to `wcfdi_transient` for the linearised reference.
    n_realisations : number of independent MC seeds.
    t_end, n_t : post-failure time grid (must match the linearised
        predictor's grid; default 200 s / 401 samples = 0.5 s).
    t_warm : intact warm-up duration [s] before t=0. Long enough for
        the slow-drift PSD's correlation time (~few hundred seconds).
    omega_grid : optional Shinozuka frequency grid. Default: log-spaced
        from 1e-3 to 1.0 rad/s, 256 points (covers slow-drift band).
    sigma_Vc, tau_Vc : current-variability process parameters
        (matched to `wcfdi_transient` defaults).
    seed : base seed; per-realisation seeds are derived as
        `(seed, k)` via `np.random.SeedSequence`.
    progress_cb : optional callable(k, n) called once per realisation.
    return_realisations : if True, the returned
        ``WcfdiSelfMcResult.eta_realisations`` holds the full
        ``(n_realisations, n_t, 3)`` ensemble (default False to keep
        the result lightweight).

    Returns
    -------
    WcfdiSelfMcResult.
    """
    # ----- Linearised reference -----
    lin = wcfdi_transient(
        cfg, Vw_mean=Vw_mean, Hs=Hs, Tp=Tp, Vc=Vc, theta_rel=theta_rel,
        scenario=scenario, t_end=t_end, n_t=n_t,
        sigma_Vc=sigma_Vc, tau_Vc=tau_Vc,
    )
    eta_mean_lin = lin.eta_mean.copy()           # (n_t, 3)
    eta_std_lin = lin.eta_std.copy()             # (n_t, 3)
    t_post = lin.t                                # (n_t,)

    # ----- Augmented system + PSD callables -----
    cp_ctrl = cfg.controller
    vessel = LinearVesselModel.from_config(cfg.vessel)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D, omega_n=cp_ctrl.omega_n, zeta=cp_ctrl.zeta,
    )
    aug = build_augmented_system(
        vessel, controller,
        T_b=cp_ctrl.bias_time_constant_s,
        T_thr=cp_ctrl.thruster_time_constant_s,
    )
    if scenario.T_thr_post is not None:
        aug_post = build_augmented_system(
            vessel, controller,
            T_b=cp_ctrl.bias_time_constant_s,
            T_thr=scenario.T_thr_post,
        )
    else:
        aug_post = aug

    S_F_funcs, tau_env = _build_disturbance_psd_funcs(
        cfg, Vw_mean, Hs, Tp, Vc, theta_rel,
        sigma_Vc=sigma_Vc, tau_Vc=tau_Vc,
    )

    # Intact deterministic steady state (warm-up target).
    x_ss = intact_mean_steady_state(aug, tau_env)

    if omega_grid is None:
        omega_grid = np.geomspace(1e-3, 1.0, 256)

    # ----- Time grids -----
    dt = float(t_post[1] - t_post[0])
    if not np.allclose(np.diff(t_post), dt):
        raise ValueError("Linearised predictor returned non-uniform t grid")
    n_warm = int(np.ceil(t_warm / dt))
    t_warm_grid = np.arange(-n_warm, 0) * dt   # (n_warm,) up to (but not incl) 0
    t_full = np.concatenate([t_warm_grid, t_post])  # (n_warm + n_t,)
    N_full = t_full.size

    # ----- MC loop -----
    cap_fn = lambda tt: scenario.cap_at_time(tt, cfg)
    cap_immediate = scenario.resolved_cap_immediate(cfg)

    eta_post_all = np.zeros((n_realisations, n_t, 3))

    seedseq = np.random.SeedSequence(seed)
    child_seeds = seedseq.spawn(n_realisations)

    for k in range(n_realisations):
        rng = np.random.default_rng(child_seeds[k])
        # Single long realisation covering both warm-up and post phases.
        F_full = realise_vector_force_time_series(
            S_F_funcs, omega_grid, t_full, rng,
        )  # (3, N_full)

        # --- Intact warm-up: integrate from x = x_ss (which is the
        # deterministic equilibrium under tau_env). The disturbance F
        # then drives the stochastic part.
        F_warm = F_full[:, :n_warm + 1]   # need one extra sample for ZOH at end
        # Build a contiguous warm-up time grid that includes t=0 as the
        # final sample, so the final state at t=0 is what we hand off
        # to the post-failure phase.
        t_warm_full = np.arange(0, n_warm + 1) * dt   # 0, dt, ..., n_warm*dt
        # Slice F so its length matches t_warm_full.
        if F_warm.shape[1] < t_warm_full.size:
            # Should not happen given construction.
            raise RuntimeError("Disturbance realisation too short for warm-up")
        x_warm = _integrate_augmented_intact_zoh(
            aug, F_warm[:, :t_warm_full.size], tau_env, t_warm_full, x_ss,
        )
        x_at_failure = x_warm[:, -1].copy()

        # --- Apply WCFDI step at t=0: clip tau_thr to immediate cap. ---
        x_at_failure[9:12] = _clip_per_dof(x_at_failure[9:12], cap_immediate)

        # --- Post-failure integration on t_post grid. ---
        F_post = F_full[:, n_warm:n_warm + n_t]
        x_post = _integrate_augmented_post_clipped(
            aug_post, F_post, tau_env, t_post, x_at_failure, cap_fn,
        )
        eta_post_all[k] = x_post[0:3, :].T   # (n_t, 3)

        if progress_cb is not None:
            progress_cb(k + 1, n_realisations)

    # ----- Empirical statistics -----
    eta_mean_emp = eta_post_all.mean(axis=0)              # (n_t, 3)
    eta_std_emp = eta_post_all.std(axis=0, ddof=1)         # (n_t, 3)

    # ----- Calibration metrics (per DOF) -----
    metrics = {}
    dof_names = ("surge", "sway", "yaw")
    for d, name in enumerate(dof_names):
        mean_residual = eta_mean_emp[:, d] - eta_mean_lin[:, d]
        std_residual = eta_std_emp[:, d] - eta_std_lin[:, d]
        scale_mean = max(np.max(np.abs(eta_mean_lin[:, d])), 1e-12)
        scale_std = max(np.max(np.abs(eta_std_lin[:, d])), 1e-12)
        metrics[name] = {
            "mean_rms_rel": float(np.sqrt(np.mean(mean_residual ** 2)) / scale_mean),
            "mean_max_rel": float(np.max(np.abs(mean_residual)) / scale_mean),
            "std_rms_rel": float(np.sqrt(np.mean(std_residual ** 2)) / scale_std),
            "std_max_rel": float(np.max(np.abs(std_residual)) / scale_std),
            "scale_mean": float(scale_mean),
            "scale_std": float(scale_std),
        }

    operating_point = {
        "Vw_mean": Vw_mean, "Hs": Hs, "Tp": Tp, "Vc": Vc,
        "theta_rel": theta_rel, "alpha": tuple(scenario.alpha),
        "gamma_immediate": scenario.gamma_immediate,
        "T_realloc": scenario.T_realloc,
    }
    info = {
        "t_warm": t_warm,
        "n_warm": n_warm,
        "dt": dt,
        "omega_grid_n": int(omega_grid.size),
        "tau_env": tau_env,
        "cqa_precondition_violated": lin.info.get(
            "cqa_precondition_violated", np.zeros(3, bool)
        ),
    }

    return WcfdiSelfMcResult(
        t=t_post,
        eta_mean_emp=eta_mean_emp,
        eta_std_emp=eta_std_emp,
        eta_mean_lin=eta_mean_lin,
        eta_std_lin=eta_std_lin,
        n_realisations=n_realisations,
        operating_point=operating_point,
        metrics=metrics,
        info=info,
        eta_realisations=eta_post_all if return_realisations else None,
    )


# ---------------------------------------------------------------------------
# Validation matrix
# ---------------------------------------------------------------------------


@dataclass
class WcfdiSelfMcMatrixCell:
    """One cell in the validation grid."""

    Vw_mean: float
    theta_rel: float
    alpha: float
    result: WcfdiSelfMcResult


@dataclass
class WcfdiSelfMcMatrix:
    """Output of `wcfdi_self_mc_matrix`.

    Attributes
    ----------
    cells : list of WcfdiSelfMcMatrixCell.
    Vw_grid, theta_grid, alpha_grid : the swept axes.
    n_realisations : per-cell MC seeds.
    """

    cells: list
    Vw_grid: np.ndarray
    theta_grid: np.ndarray
    alpha_grid: np.ndarray
    n_realisations: int


def wcfdi_self_mc_matrix(
    cfg: CqaConfig,
    *,
    Vw_grid: np.ndarray,
    theta_grid: np.ndarray,
    alpha_grid: np.ndarray,
    Hs_of_Vw: Callable[[float], float] = lambda V: 0.21 * V,
    Tp_of_Hs: Callable[[float], float] = lambda H: max(4.0, 4.0 * np.sqrt(H)),
    Vc: float = 0.3,
    n_realisations: int = 64,
    t_end: float = 200.0,
    n_t: int = 401,
    t_warm: float = 600.0,
    seed: Optional[int] = 0,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> WcfdiSelfMcMatrix:
    """Run the validation matrix Vw_grid x theta_grid x alpha_grid.

    Default sea-state mapping is a Pierson-Moskowitz-style
    ``Hs = 0.21 * Vw`` and ``Tp = 4 sqrt(Hs)``, matched to the
    operability polar so the grid points are physically consistent
    with the design tool.
    """
    cells = []
    n_total = Vw_grid.size * theta_grid.size * alpha_grid.size
    k = 0
    for Vw in Vw_grid:
        Hs = float(Hs_of_Vw(Vw))
        Tp = float(Tp_of_Hs(Hs))
        for th in theta_grid:
            for a in alpha_grid:
                k += 1
                if progress_cb is not None:
                    progress_cb(
                        k, n_total,
                        f"Vw={Vw:.1f} m/s, theta={np.degrees(th):.0f} deg, "
                        f"alpha={a:.2f}",
                    )
                scenario = WcfdiScenario(
                    alpha=(a, a, a), gamma_immediate=0.5, T_realloc=10.0,
                )
                res = wcfdi_self_mc(
                    cfg, Vw_mean=float(Vw), Hs=Hs, Tp=Tp, Vc=Vc,
                    theta_rel=float(th), scenario=scenario,
                    n_realisations=n_realisations,
                    t_end=t_end, n_t=n_t, t_warm=t_warm,
                    seed=(seed, k) if seed is not None else None,
                )
                cells.append(WcfdiSelfMcMatrixCell(
                    Vw_mean=float(Vw), theta_rel=float(th), alpha=float(a),
                    result=res,
                ))
    return WcfdiSelfMcMatrix(
        cells=cells,
        Vw_grid=np.asarray(Vw_grid, dtype=float),
        theta_grid=np.asarray(theta_grid, dtype=float),
        alpha_grid=np.asarray(alpha_grid, dtype=float),
        n_realisations=n_realisations,
    )


__all__ = [
    "WcfdiSelfMcResult",
    "WcfdiSelfMcMatrix",
    "WcfdiSelfMcMatrixCell",
    "wcfdi_self_mc",
    "wcfdi_self_mc_matrix",
]
