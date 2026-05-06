"""WCFDI transient analysis.

Predicts the time-evolution of the vessel pose around the DP setpoint in the
seconds immediately following a worst-case failure (loss of one thruster
group / power group), assuming the failure happens *now* at the given
operating point.

State (12-vector):
    x = [eta(3), nu(3), b_hat(3), tau_thr(3)]
where
    eta      : position/heading deviation from setpoint (NED-aligned, small
               heading approximation)
    nu       : body velocity (surge, sway, yaw rate)
    b_hat    : controller bias estimate (low-frequency env. force estimate)
    tau_thr  : actually-produced thruster force (first-order lag from
               command)

Continuous-time augmented dynamics (linear, no saturation):

    eta_dot     = nu
    M nu_dot    = -D nu + tau_thr + tau_env_const + w(t)
    b_hat_dot   = (1/T_b) (tau_env_est_residual)         # see below
    tau_thr_dot = (1/T_thr) (tau_cmd - tau_thr)
    tau_cmd     = -Kp eta - Kd nu - b_hat

Bias estimator: the brucon dp_estimator (Fossen passive observer, ch. 12)
corrects the bias from the *position innovation*, not from a magic
knowledge of the true environmental force. The linearised form is

    b_hat_dot = (1/T_b) Kp eta

so that b_hat slowly integrates the position residual and asymptotically
equals tau_env when the controller has driven eta to zero. Compared to
the naive form `b_hat_dot = (1/T_b)(tau_env - b_hat)`, this has

  - identical deterministic steady-state mean (eta=0, b_hat=tau_env),
  - non-zero stochastic variance on b_hat, inherited from eta's
    closed-loop variance through the same wind/drift/current process
    that drives eta. This is the variance the WCFDI starting-state MC
    needs in order to sample b_hat(0-) realistically.

Saturation: the commanded thrust is clipped against a per-DOF capability
cap (post-WCFDI for the failure scenario). Saturation only affects the
deterministic mean; the stochastic part is integrated against the
unsaturated linear system (conservative: tends to keep variance from
collapsing artificially during deep saturation).

WCFDI event at t=0:
    intact:   tau_capability = tau_cap_intact
    post:     tau_capability = tau_cap_post = alpha * tau_cap_intact (per DOF)
    The pre-failure tau_thr cannot exceed tau_cap_post immediately after
    the event, so a step force imbalance arises:

        Delta_tau(0+) = tau_thr(0-) - clip(tau_thr(0-), -tau_cap_post, tau_cap_post)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov

from .config import CqaConfig
from .vessel import LinearVesselModel, WindForceModel, CurrentForceModel
from .controller import LinearDpController
from .closed_loop import ClosedLoop, state_covariance_freqdomain
from .psd import (
    npd_wind_gust_force_psd,
    slow_drift_force_psd_newman,
    current_variability_force_psd,
)


# ---------------------------------------------------------------------------
# Augmented-state assembly
# ---------------------------------------------------------------------------


@dataclass
class AugmentedSystem:
    """Augmented linear system used for transient analysis.

    State layout:
        idx  0..2 : eta (n,e,psi)
        idx  3..5 : nu  (u,v,r)
        idx  6..8 : b_hat (3-DOF bias estimate, force units)
        idx  9..11: tau_thr (3-DOF thruster output, force units)
    """

    A: np.ndarray  # 12x12
    B_w: np.ndarray  # 12x3 stochastic disturbance into nu via M^{-1}
    B_d: np.ndarray  # 12x3 deterministic env. force bias (drives b_hat tracker
    #                       and acts on nu via M^{-1})
    M: np.ndarray  # 3x3 (kept for projections)
    Kp: np.ndarray  # 3x3 controller gains (kept for clipping logic)
    Kd: np.ndarray
    T_b: float  # bias estimator time constant [s]
    T_thr: float  # thruster lag time constant [s]


def build_augmented_system(
    vessel: LinearVesselModel,
    controller: LinearDpController,
    T_b: float = 100.0,
    T_thr: float = 5.0,
) -> AugmentedSystem:
    """Build the 12x12 A matrix and disturbance/input maps."""
    Minv = np.linalg.inv(vessel.M)
    Kp, Kd = controller.feedback()

    A = np.zeros((12, 12))

    # eta_dot = nu
    A[0:3, 3:6] = np.eye(3)

    # nu_dot = -M^-1 D nu + M^-1 tau_thr   (no environment in homogeneous part;
    # the constant bias enters via B_d below)
    A[3:6, 3:6] = -Minv @ vessel.D
    A[3:6, 9:12] = Minv

    # b_hat_dot = (1/T_b) Kp eta   (passive-observer form, eta-driven)
    A[6:9, 0:3] = (1.0 / T_b) * Kp

    # tau_thr_dot = (1/T_thr) (tau_cmd - tau_thr)
    # tau_cmd = -Kp eta - Kd nu - b_hat
    A[9:12, 0:3] = -(1.0 / T_thr) * Kp
    A[9:12, 3:6] = -(1.0 / T_thr) * Kd
    A[9:12, 6:9] = -(1.0 / T_thr) * np.eye(3)
    A[9:12, 9:12] = -(1.0 / T_thr) * np.eye(3)

    # B_w : stochastic force enters nu via M^-1
    B_w = np.zeros((12, 3))
    B_w[3:6, :] = Minv

    # B_d : deterministic env. force enters nu via M^-1 (mean force balance).
    # The bias estimator no longer takes tau_env as a direct input (it is
    # driven by eta), so the b_hat row of B_d is zero.
    B_d = np.zeros((12, 3))
    B_d[3:6, :] = Minv

    return AugmentedSystem(
        A=A,
        B_w=B_w,
        B_d=B_d,
        M=vessel.M,
        Kp=Kp,
        Kd=Kd,
        T_b=T_b,
        T_thr=T_thr,
    )


# ---------------------------------------------------------------------------
# Pre-failure steady state (intact, mean part)
# ---------------------------------------------------------------------------


def intact_mean_steady_state(
    aug: AugmentedSystem, tau_env: np.ndarray
) -> np.ndarray:
    """Steady-state mean augmented state under intact closed loop.

    Analytic result for a well-tuned DP loop with bias FF:
        eta_ss     = 0
        nu_ss      = 0
        b_hat_ss   = +tau_env   (bias estimator absorbs the slow load)
        tau_thr_ss = -tau_env   (thrusters balance the load)

    We use the closed form rather than `np.linalg.solve(A, ...)` because
    the augmented A mixes wildly different unit scales (position rows,
    force rows, and the small `1/T_b * Kp` coupling on the b_hat block)
    and is numerically near-singular even though every eigenvalue is in
    the open LHP. The analytic answer above is exact and unit-safe.
    """
    x_ss = np.zeros(12)
    x_ss[6:9] = tau_env
    x_ss[9:12] = -tau_env
    return x_ss


# ---------------------------------------------------------------------------
# Pre-failure stochastic steady state (covariance) -- delegates to P1 path.
# ---------------------------------------------------------------------------


def intact_state_covariance_freqdomain(
    cl_intact: ClosedLoop,
    S_F_funcs: list[Callable[[np.ndarray], np.ndarray]],
    omega_lo: float = 1e-4,
    omega_hi: float = 1.5,
    n_points: int = 512,
) -> np.ndarray:
    """6x6 state covariance (eta, nu) under intact closed loop. Re-uses P1 path.

    Returned matrix is upgraded to 12x12 by the caller (zeros for b, tau_thr
    stochastic init), since the transient analysis assumes the bias and
    thruster states have already settled to deterministic values at t=0-.
    """
    return state_covariance_freqdomain(cl_intact, S_F_funcs, omega_lo, omega_hi, n_points)


def lift_intact_cov_to_augmented(P6: np.ndarray) -> np.ndarray:
    """Lift 6x6 (eta, nu) covariance into 12x12 augmented covariance.

    Bias and thruster states are taken as deterministic at t=0- (they have
    already converged to the intact steady-state values), so their covariance
    is zero at the failure instant.
    """
    P12 = np.zeros((12, 12))
    P12[:6, :6] = P6
    return P12


# ---------------------------------------------------------------------------
# Transient solvers
# ---------------------------------------------------------------------------


@dataclass
class WcfdiScenario:
    """Defines the WCFDI failure to evaluate.

    Simplified transient model:
      - At t=0 (the WCF event), per-DOF available cap drops instantly to
        `gamma_immediate * tau_cap_intact` (default 0.5: half of intact
        capability is available immediately, regardless of what eventually
        survives).
      - The available cap then recovers exponentially with time constant
        `T_realloc` to the post-failure steady-state cap
        `alpha * tau_cap_intact`.
      - During this recovery window the DP regulator may saturate against
        the time-varying cap (PD + bias terms responding to the growing
        excursion can demand more thrust than is currently available).
        The peak excursion is governed by this saturation episode.

    NOTE (future work): this parametric (gamma_immediate, T_realloc)
    first-order ramp is a deliberate placeholder. Once the overall
    approach is validated, it should be replaced by querying the real
    brucon `BasicAllocator` over the surviving thruster set, with actual
    azimuth slew rates and per-thruster force/moment limits, to compute
    the true time-varying per-DOF achievable cap.

    Fields:
      `alpha`: per-DOF surviving fraction of the intact capability *after*
        reallocation has completed (steady-state post-failure cap).
      `tau_cap_intact`: per-DOF intact capability ceiling [N, N, Nm]. If
        `None`, `wcfdi_transient` substitutes `cfg.thrust_capability`.
      `T_thr_post`: optional override for thruster lag after the event.
      `gamma_immediate`: per-DOF fraction of the intact capability
        available immediately at t=0+ (default 0.5).
      `T_realloc`: first-order time constant of thrust reallocation [s]
        (default 10 s, representative of azimuth slew + DP allocator
        bandwidth for a CSOV).
    """

    alpha: tuple[float, float, float] = (0.5, 0.5, 0.5)
    tau_cap_intact: Optional[tuple[float, float, float]] = None
    T_thr_post: Optional[float] = None
    gamma_immediate: float = 0.5
    T_realloc: float = 10.0

    def resolved_cap_intact(self, cfg: CqaConfig) -> np.ndarray:
        if self.tau_cap_intact is None:
            return cfg.thrust_capability.as_array()
        return np.array(self.tau_cap_intact)

    def resolved_cap_post(self, cfg: CqaConfig) -> np.ndarray:
        return np.array(self.alpha) * self.resolved_cap_intact(cfg)

    def resolved_cap_immediate(self, cfg: CqaConfig) -> np.ndarray:
        return self.gamma_immediate * self.resolved_cap_intact(cfg)

    def cap_at_time(self, t: float, cfg: CqaConfig) -> np.ndarray:
        """Per-DOF available cap at time t after the WCF event.

        First-order ramp from `gamma_immediate * cap_intact` (at t=0+) to
        `alpha * cap_intact` (as t -> infinity), with time constant
        `T_realloc`.
        """
        cap_post = self.resolved_cap_post(cfg)
        cap_imm = self.resolved_cap_immediate(cfg)
        if self.T_realloc <= 0.0:
            return cap_post
        return cap_post + (cap_imm - cap_post) * np.exp(-t / self.T_realloc)


def _clip_per_dof(tau: np.ndarray, cap: np.ndarray) -> np.ndarray:
    return np.clip(tau, -cap, cap)


def _augmented_rhs_post(
    t: float,
    x: np.ndarray,
    aug: AugmentedSystem,
    tau_env: np.ndarray,
    cap_fn: Callable[[float], np.ndarray],
) -> np.ndarray:
    """RHS for the deterministic post-failure mean trajectory with thrust clipping.

    `cap_fn(t)` returns the per-DOF available cap at time t (modelling
    the thrust-reallocation ramp from the immediate-post-failure value
    up to the steady-state post-failure cap).
    """
    eta = x[0:3]
    nu = x[3:6]
    b_hat = x[6:9]
    tau_thr = x[9:12]

    Minv_D = aug.A[3:6, 3:6]  # = -M^-1 D
    Minv = aug.B_w[3:6, :]  # = M^-1

    cap_now = cap_fn(t)
    eta_dot = nu
    nu_dot = Minv_D @ nu + Minv @ tau_thr + Minv @ tau_env
    b_hat_dot = (1.0 / aug.T_b) * (aug.Kp @ eta)
    tau_cmd = -aug.Kp @ eta - aug.Kd @ nu - b_hat
    tau_cmd_clipped = _clip_per_dof(tau_cmd, cap_now)
    tau_thr_dot = (1.0 / aug.T_thr) * (tau_cmd_clipped - tau_thr)

    out = np.empty(12)
    out[0:3] = eta_dot
    out[3:6] = nu_dot
    out[6:9] = b_hat_dot
    out[9:12] = tau_thr_dot
    return out


@dataclass
class TransientResult:
    """Result of a WCFDI transient analysis at a fixed operating point."""

    t: np.ndarray  # (N,) time grid [s]
    x_mean: np.ndarray  # (N, 12) deterministic mean trajectory
    P: np.ndarray  # (N, 12, 12) covariance at each time
    eta_mean: np.ndarray  # (N, 3) shorthand for x_mean[:, 0:3]
    eta_std: np.ndarray  # (N, 3) sqrt of P[:, 0:3, 0:3] diagonal
    info: dict = field(default_factory=dict)

    def env_lower_upper(self, k: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
        lo = self.eta_mean - k * self.eta_std
        hi = self.eta_mean + k * self.eta_std
        return lo, hi


def wcfdi_transient(
    cfg: CqaConfig,
    Vw_mean: float,
    Hs: float,
    Tp: float,
    Vc: float,
    theta_rel: float,
    scenario: WcfdiScenario,
    omega_n: Optional[tuple[float, float, float]] = None,
    zeta: Optional[tuple[float, float, float]] = None,
    T_b: Optional[float] = None,
    T_thr: Optional[float] = None,
    sigma_Vc: float = 0.1,
    tau_Vc: float = 600.0,
    t_end: float = 200.0,
    n_t: int = 401,
    rao_table: Optional["RaoTable"] = None,
) -> TransientResult:
    """Predict the post-WCFDI mean trajectory + covariance envelope.

    The relative weather direction `theta_rel` (rad) is the heading of the
    weather *into* the vessel, with 0 = head-on. All disturbances are
    assumed to come from this direction.

    Controller / observer tuning (`omega_n`, `zeta`, `T_b`, `T_thr`)
    defaults to ``cfg.controller`` (single source of truth shared with
    the WCFDI MC and the intact-prior pipeline).

    Drift force model
    -----------------
    When ``rao_table`` is supplied, the **mean drift force** and the
    **slow-drift force PSD** are evaluated by integrating the
    pdstrip-resolved diagonal QTF against the Bretschneider
    wave-elevation spectrum (the spectral path -- this is what
    ``mean_drift_force_pdstrip`` and
    ``slow_drift_force_psd_newman_pdstrip`` provide, and what brucon's
    ``MeanDriftForces()`` matches to ~1 % at force level; see analysis
    section 12.16). When ``rao_table is None`` (default for
    backwards compatibility), the legacy parametric path is used:
    ``F_drift_i = drift_i_amp * Hs^2 * trig_i(theta)`` with the
    coefficients from ``cfg.wave_drift``. The parametric coefficients
    are *frequency-integrated* and ignore Tp dependence; they can be
    seriously wrong (wrong sign, factor-of-2 magnitude) for sea states
    far from their implicit calibration point. **Pass ``rao_table`` for
    any cqa-vs-brucon validation work or any forecast-grid evaluation
    that sweeps Tp.** The legacy path remains available for offline
    analyses where no pdstrip data is loaded.

    The result's ``info`` dict carries a ``bistability_risk_score``
    (per-DOF and scalar): the maximum over time of
    ``max(0, |tau_cmd_mean(t)| - cap(t)) / sigma_tau_cmd(t)``. Empirically
    against ``cqa.wcfdi_self_mc``, scores below ~1 sigma correspond to
    deterministic + stochastic recovery in essentially every realisation;
    scores above ~1 sigma correspond to a meta-stable saturated regime
    where a fraction of stochastic realisations runs away even though
    the deterministic mean ODE recovers. Operability gating should
    therefore treat large bistability scores as alarm regardless of the
    nominal mean+std excursion prediction.

    Returns a TransientResult with t, mean and covariance trajectories.
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

    # Mean environmental force at this direction (wind + current; mean wave
    # drift is small slow component, captured by drift PSD; for the mean
    # transient we add the steady drift force too).
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
    # Mean drift force.
    if rao_table is not None:
        # Spectral path: integrate pdstrip diagonal QTF against
        # Bretschneider wave-elevation spectrum (matches brucon
        # MeanDriftForces() to ~1 % at force level).
        from .drift import mean_drift_force_pdstrip
        F_drift = mean_drift_force_pdstrip(
            rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel,
        )
    else:
        # Parametric fallback: F_drift_i = drift_i_amp * Hs^2 * trig_i(theta).
        # Frequency-integrated coefficients; ignores Tp dependence and may
        # be seriously wrong for sea states far from the implicit
        # calibration point. See docstring above.
        F_drift = np.array(
            [
                wd.drift_x_amp * Hs ** 2 * np.cos(theta_rel),
                wd.drift_y_amp * Hs ** 2 * np.sin(theta_rel),
                wd.drift_n_amp * Hs ** 2 * np.sin(2.0 * theta_rel),
            ]
        )
    tau_env = F_wind + F_curr + F_drift

    # --- Pre-failure mean steady state at t = 0- ---
    x0 = intact_mean_steady_state(aug, tau_env)

    # --- Pre-failure covariance at t = 0- ---
    cl_intact = ClosedLoop.build(vessel, controller)
    S_wind = npd_wind_gust_force_psd(wind_model, Vw_mean, theta_rel)
    if rao_table is not None:
        from .drift import slow_drift_force_psd_newman_pdstrip
        S_drift = slow_drift_force_psd_newman_pdstrip(
            rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel,
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
    P0 = lift_intact_cov_to_augmented(P6)

    # --- Post-failure dynamics ---
    if scenario.T_thr_post is not None:
        aug = build_augmented_system(vessel, controller, T_b=T_b, T_thr=scenario.T_thr_post)
    cap_post = scenario.resolved_cap_post(cfg)
    cap_immediate = scenario.resolved_cap_immediate(cfg)

    # CQA precondition: the static CQA upstream is responsible for ensuring
    # |tau_env| <= cap_post in every DOF at the chosen operating point. We
    # check it here as a sanity guard and surface it via `info`; if it is
    # violated the transient analysis is still run but the result is
    # physically a drift-off, not a recoverable transient.
    cqa_violated = np.abs(tau_env) > cap_post  # (3,) bool

    # Step force imbalance: at t=0+ the achievable thruster output is
    # clipped to the *immediate* (pre-reallocation) cap. This is the
    # dominant deterministic transient source for a CQA-guarded operating
    # point: the surviving thrusters must ramp up to take over the load.
    x0_post = x0.copy()
    x0_post[9:12] = _clip_per_dof(x0[9:12], cap_immediate)
    delta_tau = x0_post[9:12] - x0[9:12]  # diagnostic

    cap_fn = lambda t: scenario.cap_at_time(t, cfg)

    # --- Solve mean trajectory (nonlinear because of clipping) ---
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
    x_mean = sol.y.T  # (N, 12)

    # --- Solve covariance trajectory: dP/dt = A P + P A^T + B_w S_w_eq B_w^T ---
    # Use a constant-intensity W consistent with the closed-loop band, taken
    # as the W that produces the *intact* steady-state covariance P6 via the
    # Lyapunov equation:
    #     A_cl_intact P6 + P6 A_cl_intact^T + B_w_6 W_eq B_w_6^T = 0
    # Solve for W_eq.
    A_cl6 = cl_intact.A_cl
    B_w6 = cl_intact.B_w  # 6x3
    Q6 = -(A_cl6 @ P6 + P6 @ A_cl6.T)  # = B_w6 W_eq B_w6^T (3x3 inner)
    # Solve W_eq from least-squares: find W minimising ||B_w6 W B_w6^T - Q6||.
    # Since B_w6 = [0; M^-1]^T-like, we can pseudo-invert.
    Bp = np.linalg.pinv(B_w6)
    W_eq = Bp @ Q6 @ Bp.T
    W_eq = 0.5 * (W_eq + W_eq.T)
    # Symmetric PSD project (clip negative eigenvalues to zero):
    eigs, V = np.linalg.eigh(W_eq)
    eigs = np.maximum(eigs, 0.0)
    W_eq = V @ np.diag(eigs) @ V.T

    BWBT_aug = aug.B_w @ W_eq @ aug.B_w.T  # 12x12

    # The variance ODE uses the *intact* augmented A (full feedback gains).
    # Rationale: under the CQA precondition |tau_env| <= cap_post, the
    # controller is not steady-state-saturated; the temporary saturation
    # affecting the *mean* trajectory does not break the linearised
    # feedback for *fluctuations* around it. So the variance dynamics are
    # governed by the same closed-loop A as in the intact case (with the
    # added thrust-lag and bias-estimator states).
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
    # Symmetrise:
    P_t = 0.5 * (P_t + P_t.transpose(0, 2, 1))

    eta_mean = x_mean[:, 0:3]
    eta_std = np.sqrt(np.maximum(np.diagonal(P_t[:, 0:3, 0:3], axis1=1, axis2=2), 0.0))

    # ----- Bistability severity score -----
    # Diagnoses the regime where the deterministic mean ODE finds the
    # recovering branch of the saturated dynamics, but a non-trivial
    # fraction of stochastic realisations falls onto the runaway branch.
    # Validated empirically against the wcfdi_self_mc engine: when the
    # commanded thrust mean exceeds the cap by more than ~1 std of the
    # commanded-thrust fluctuation, MC recovery rate drops below ~90%.
    #
    # Per DOF, per time:
    #   mu_tau   = K_tau @ x_mean        (deterministic commanded thrust)
    #   sigma_tau= sqrt(diag(K_tau @ P @ K_tau^T))
    #   severity = max(0, (|mu_tau| - cap) / sigma_tau)
    #
    # The reported score is the maximum over time, per DOF, plus the
    # overall scalar maximum (the headline "bistability_risk_score").
    K_tau = np.zeros((3, 12))
    K_tau[:, 0:3] = -aug.Kp
    K_tau[:, 3:6] = -aug.Kd
    K_tau[:, 6:9] = -np.eye(3)
    tau_cmd_mean = (K_tau @ x_mean.T).T            # (n_t, 3)
    # tau_cmd variance: (n_t, 3, 3) = K @ P @ K^T per time
    tau_cmd_var = np.einsum("ij,tjk,lk->til", K_tau, P_t, K_tau)
    sigma_tau_cmd = np.sqrt(
        np.maximum(np.diagonal(tau_cmd_var, axis1=1, axis2=2), 0.0)
    )                                               # (n_t, 3)
    cap_t = np.array([scenario.cap_at_time(float(tt), cfg) for tt in t_eval])
    headroom = cap_t - np.abs(tau_cmd_mean)        # (n_t, 3)
    severity_t = np.maximum(0.0, -headroom) / np.maximum(sigma_tau_cmd, 1e-9)
    bistability_per_dof = severity_t.max(axis=0)   # (3,)
    bistability_risk_score = float(bistability_per_dof.max())

    info = {
        "tau_env": tau_env,
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
    }
    return TransientResult(
        t=t_eval,
        x_mean=x_mean,
        P=P_t,
        eta_mean=eta_mean,
        eta_std=eta_std,
        info=info,
    )