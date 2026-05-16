"""WCFDI transient analysis with EXPLICIT OBSERVER STATES (27-state).

Background
----------
The 15-state model in ``cqa.transient`` collapses the brucon Fossen
passive observer into a single frozen-b̂ feed-forward term, with the
controller acting on truth η, ν. This is correct for intact
station-keeping but **systematically underpredicts the post-WCF
excursion** because:

1.  Brucon's controller actually acts on the observer's η̂, ν̂ — not on
    truth.
2.  The observer's velocity-estimate ODE is fed ``τ_obs = τ_cmd``
    (orders, with ``use_tau_feedback=false`` default), which does **not**
    drop on the WCFDI event. Truth ν drops, observer ν̂ doesn't, so an
    innovation builds up. This is the "dual-injection" asymmetry.
3.  The observer's bias estimate b̂ slowly absorbs the persistent
    innovation (T_b = 1000 s) and contaminates η̂, distorting the
    recovery shape relative to truth.
4.  Brucon's observer has a 2nd-order wave filter on the innovation,
    designed to keep WF (ω≈ω_p≈0.6 rad/s) energy out of η̂. The filter
    has non-trivial **phase response at LF** (ω≈0.05–0.1 rad/s), which
    was found (by direct comparison against the brucon truth ensemble)
    to be the dominant cause of η̂ lagging truth and hence delaying
    the controller's recovery action.

Diagnostic ``brucon_etahat_vs_etatruth.py`` (committed in this branch)
shows the truth − η̂ error peaks at 0.21 m at t = +45 s post-WCF.

State (27-vector)
-----------------
::

    idx  0..2  : eta        (truth body deviation)        [m, m, rad]
    idx  3..5  : nu         (truth body velocity)         [m/s, m/s, rad/s]
    idx  6..8  : eta_hat    (observer LF position)        [m, m, rad]
    idx  9..11 : nu_hat     (observer LF velocity)        [m/s, m/s, rad/s]
    idx 12..14 : b_hat      (observer bias, force units)  [N, N, Nm]
    idx 15..17 : tau_thr    (1st-order thrust lag)        [N, N, Nm]
    idx 18..20 : I          (PI integrator on eta_hat)    [m·s, m·s, rad·s]
    idx 21..23 : xi         (wave filter integrator)      [m·s,...]
    idx 24..26 : eta_wave   (wave filter output)          [m, m, rad]

Continuous-time linear dynamics, innovation
``e = eta - eta_hat - eta_wave``::

    eta_dot      = nu                                                       (truth kinematics)
    M·nu_dot     = -D·nu + tau_thr + tau_env_const + w(t) + tau_lost(t)     (truth, dual-inject)
    eta_hat_dot  = nu_hat + omega_c·e                                       (observer η̂)
    M·nu_hat_dot = -D·nu_hat + b_hat + tau_cmd + M·K_a_pos·e                (observer ν̂; consumes tau_cmd)
    b_hat_dot    = -(1/T_b)·b_hat + K_b_pos·e                               (observer bias, dynamic)
    tau_thr_dot  = (1/T_thr)·(tau_cmd - tau_thr)                            (thrust lag)
    I_dot        = eta_hat                                                  (controller integrates η̂)
    xi_dot       = eta_wave + k1·e                                          (wave filter integrator)
    eta_wave_dot = -omega_p²·xi - 2 zeta_w·omega_p·eta_wave + k2·e          (wave filter output)

    tau_cmd      = -Kp·eta_hat - Kd·nu_hat - b_hat - Ki·I

Wave filter constants (per DOF, brucon ``SecondOrderWaveFilter``)::

    k1 = -2 (1 - zeta_w) omega_c / omega_p
    k2 =  2 omega_p (1 - zeta_w)

Observer gains (per DOF, diag) come from the brucon CSOV
``observer.prototxt``::

    surge: K_b_pos = 0.0012,  K_a_pos = 0.12,  ω_c = 1.04 rad/s,  T_b = 1000 s
    sway : K_b_pos = 0.0012,  K_a_pos = 0.12,  ω_c = 1.04 rad/s,  T_b = 1000 s
    yaw  : K_b_pos = 0.002 ,  K_a_pos = 0.20,  ω_c = 1.04 rad/s,  T_b = 1000 s

Wave-filter peak frequency ω_p locks to the brucon wave-period estimator
(default Tp ≈ 10 s -> ω_p ≈ 0.628 rad/s); ζ_w defaults to 0.1.

Initial conditions
------------------
Use ``np.linalg.solve(A, -B_d @ tau_env)`` to obtain the exact intact
SS (the closed-form approximation has a small residual on the b_hat
row; the system is type-1 so the exact SS is well-defined).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .vessel import LinearVesselModel
from .controller import LinearDpController


# ---------------------------------------------------------------------------
# State-vector index map (27 states)
# ---------------------------------------------------------------------------
IDX_ETA = slice(0, 3)
IDX_NU = slice(3, 6)
IDX_ETA_HAT = slice(6, 9)
IDX_NU_HAT = slice(9, 12)
IDX_B_HAT = slice(12, 15)
IDX_TAU_THR = slice(15, 18)
IDX_INT = slice(18, 21)
IDX_XI = slice(21, 24)         # wave filter integrator
IDX_ETA_W = slice(24, 27)      # wave filter output (η_w)
N_STATE = 27


# ---------------------------------------------------------------------------
# Default observer gains (CSOV config_csov/observer.prototxt)
# ---------------------------------------------------------------------------
@dataclass
class ObserverGains:
    """Per-DOF diagonal observer gains (surge, sway, yaw)."""
    K_b_pos: np.ndarray  # (3,) bias-from-position gain [1/s]
    K_a_pos: np.ndarray  # (3,) acceleration-from-position gain [1/s^2]
    omega_c: np.ndarray  # (3,) wave-filter cutoff / position-innov gain [1/s]
    T_b: np.ndarray      # (3,) bias time constant [s]
    omega_p: np.ndarray  # (3,) wave-filter peak frequency [rad/s] (≈ 2π/Tp)
    zeta_w: np.ndarray   # (3,) wave-filter relative damping ratio [-]


def csov_observer_gains(Tp_s: float = 10.0, zeta_w: float = 0.1) -> ObserverGains:
    """Match build/bin/config_csov/observer.prototxt verbatim.

    The wave-filter peak frequency is set from ``Tp_s`` (default 10 s,
    matching the brucon CSOV scenario Tp ≈ 10.224 s). The damping
    ratio defaults to brucon's 0.1 (set in
    ``second_order_wave_filter.cpp:64`` as ``ScaleGainLinear`` between
    0.1 and 0.25 depending on peak frequency; for Tp=10 s the brucon
    selector lands at 0.1).
    """
    omega_p = 2.0 * np.pi / Tp_s
    return ObserverGains(
        K_b_pos=np.array([0.0012, 0.0012, 0.002]),
        K_a_pos=np.array([0.12, 0.12, 0.20]),
        omega_c=np.array([1.04, 1.04, 1.04]),
        T_b=np.array([1000.0, 1000.0, 1000.0]),
        omega_p=np.array([omega_p, omega_p, omega_p]),
        zeta_w=np.array([zeta_w, zeta_w, zeta_w]),
    )


# ---------------------------------------------------------------------------
# Augmented system (21 states)
# ---------------------------------------------------------------------------
@dataclass
class AugmentedSystemObs:
    """21-state augmented closed-loop system with explicit observer."""
    A: np.ndarray         # (21, 21) state matrix
    B_w: np.ndarray       # (21, 3)  stochastic disturbance into truth ν
    B_d: np.ndarray       # (21, 3)  deterministic env. force into truth ν
    B_lost: np.ndarray    # (21, 3)  τ_lost(t) into truth ν (DUAL-INJECT)
    M: np.ndarray         # (3, 3)
    D: np.ndarray         # (3, 3)
    Kp: np.ndarray        # (3, 3)
    Kd: np.ndarray
    Ki: np.ndarray
    obs_gains: ObserverGains
    T_thr: float
    n_state: int = N_STATE


def build_observer_augmented_system_full(
    vessel: LinearVesselModel,
    controller: LinearDpController,
    obs_gains: ObserverGains | None = None,
    T_thr: float = 5.0,
    Ki_factor: float = 0.1,
) -> AugmentedSystemObs:
    """Assemble the 21-state augmented system A, B_w, B_d, B_lost.

    Parameters
    ----------
    vessel, controller : per ``cqa.vessel`` / ``cqa.controller``.
    obs_gains : observer gain bundle. Defaults to CSOV config.
    T_thr : 1st-order thrust lag time constant [s].
    Ki_factor : per-DOF Ki = ``Ki_factor * omega_n_per_dof * Kp_diag``,
        with ``omega_n_per_dof[i] = sqrt(Kp[i,i] / M[i,i])``. Brucon
        convention is 0.1.
    """
    if obs_gains is None:
        obs_gains = csov_observer_gains()

    M = vessel.M
    D = vessel.D
    Minv = np.linalg.inv(M)
    Kp, Kd = controller.feedback()

    # PI integrator gain (diag), brucon convention.
    M_diag = np.array([M[i, i] for i in range(3)])
    Kp_diag = np.array([Kp[i, i] for i in range(3)])
    omega_n_per_dof = np.sqrt(np.maximum(Kp_diag / np.maximum(M_diag, 1e-12), 0.0))
    Ki_vec = Ki_factor * omega_n_per_dof * Kp_diag
    Ki = np.diag(Ki_vec)

    # Diagonal observer-gain matrices.
    K_b_pos = np.diag(obs_gains.K_b_pos)
    K_a_pos = np.diag(obs_gains.K_a_pos)
    Omega_c = np.diag(obs_gains.omega_c)
    Tb_inv = np.diag(1.0 / obs_gains.T_b)

    # Wave-filter constants (per DOF, see SecondOrderWaveFilter::SetPeakFrequency
    # at libs/common/math/second_order_wave_filter.cpp:66-67):
    #   k1 = -2 (1 - zeta_w) omega_c / omega_p
    #   k2 =  2 omega_p (1 - zeta_w)
    one_m_zeta = 1.0 - obs_gains.zeta_w
    k1_wf = -2.0 * one_m_zeta * obs_gains.omega_c / obs_gains.omega_p   # (3,)
    k2_wf = 2.0 * obs_gains.omega_p * one_m_zeta                         # (3,)
    K1_WF = np.diag(k1_wf)
    K2_WF = np.diag(k2_wf)
    Wp2 = np.diag(obs_gains.omega_p ** 2)
    TwoZetaWp = np.diag(2.0 * obs_gains.zeta_w * obs_gains.omega_p)

    A = np.zeros((N_STATE, N_STATE))

    # Innovation: e = eta - eta_hat - eta_wave
    # Helper to add "G @ e" to a row block.
    def add_G_times_e(row_slice: slice, G: np.ndarray) -> None:
        A[row_slice, IDX_ETA] += G
        A[row_slice, IDX_ETA_HAT] += -G
        A[row_slice, IDX_ETA_W] += -G

    # --- Row 0..2 : eta_dot = nu ---
    A[IDX_ETA, IDX_NU] = np.eye(3)

    # --- Row 3..5 : M nu_dot = -D nu + tau_thr  ->
    #                nu_dot = -Minv D nu + Minv tau_thr ---
    A[IDX_NU, IDX_NU] = -Minv @ D
    A[IDX_NU, IDX_TAU_THR] = Minv

    # --- Row 6..8 : eta_hat_dot = nu_hat + omega_c · e ---
    A[IDX_ETA_HAT, IDX_NU_HAT] = np.eye(3)
    add_G_times_e(IDX_ETA_HAT, Omega_c)

    # --- Row 9..11 : M nu_hat_dot = -D nu_hat + b_hat + tau_cmd + M K_a_pos · e
    #                 nu_hat_dot = -Minv D nu_hat + Minv b_hat + Minv tau_cmd + K_a_pos · e
    A[IDX_NU_HAT, IDX_NU_HAT] = -Minv @ D
    A[IDX_NU_HAT, IDX_B_HAT] = Minv
    # tau_cmd = -Kp eta_hat - Kd nu_hat - b_hat - Ki I
    A[IDX_NU_HAT, IDX_ETA_HAT] += -Minv @ Kp
    A[IDX_NU_HAT, IDX_NU_HAT] += -Minv @ Kd
    A[IDX_NU_HAT, IDX_B_HAT] += -Minv  # cancels the Minv from the observer's own b_hat term
    A[IDX_NU_HAT, IDX_INT] += -Minv @ Ki
    add_G_times_e(IDX_NU_HAT, K_a_pos)

    # NOTE on the b_hat cancellation: the observer's ν̂_dot model includes
    # +b̂ ("the bias force I think is acting"), and the controller's τ_cmd
    # includes -b̂ ("FF compensation for the env force I'm modelling").
    # When the controller acts on observer states, those two b̂ contributions
    # in ν̂_dot cancel exactly. b̂ still affects truth via the τ_cmd → τ_thr
    # chain (which is the actual physical channel where the bias FF
    # rejects the env force on the vessel).

    # --- Row 12..14 : b_hat_dot = -(1/T_b) b_hat + K_b_pos · e ---
    A[IDX_B_HAT, IDX_B_HAT] = -Tb_inv
    add_G_times_e(IDX_B_HAT, K_b_pos)

    # --- Row 15..17 : tau_thr_dot = (1/T_thr) (tau_cmd - tau_thr) ---
    inv_T_thr = 1.0 / T_thr
    A[IDX_TAU_THR, IDX_ETA_HAT] = -inv_T_thr * Kp
    A[IDX_TAU_THR, IDX_NU_HAT] = -inv_T_thr * Kd
    A[IDX_TAU_THR, IDX_B_HAT] = -inv_T_thr * np.eye(3)
    A[IDX_TAU_THR, IDX_INT] = -inv_T_thr * Ki
    A[IDX_TAU_THR, IDX_TAU_THR] = -inv_T_thr * np.eye(3)

    # --- Row 18..20 : I_dot = eta_hat ---
    A[IDX_INT, IDX_ETA_HAT] = np.eye(3)

    # --- Row 21..23 : xi_dot = eta_wave + k1 · e ---
    A[IDX_XI, IDX_ETA_W] = np.eye(3)
    add_G_times_e(IDX_XI, K1_WF)

    # --- Row 24..26 : eta_wave_dot = -omega_p² xi - 2 zeta_w omega_p eta_wave + k2 · e ---
    A[IDX_ETA_W, IDX_XI] = -Wp2
    A[IDX_ETA_W, IDX_ETA_W] = -TwoZetaWp
    add_G_times_e(IDX_ETA_W, K2_WF)

    # --- B_w : stochastic env force into truth ν via Minv ---
    B_w = np.zeros((N_STATE, 3))
    B_w[IDX_NU, :] = Minv

    # --- B_d : deterministic env force into truth ν via Minv ---
    B_d = np.zeros((N_STATE, 3))
    B_d[IDX_NU, :] = Minv

    # --- B_lost : τ_lost(t) into truth ν via Minv (DUAL-INJECT) ---
    B_lost = np.zeros((N_STATE, 3))
    B_lost[IDX_NU, :] = Minv

    return AugmentedSystemObs(
        A=A, B_w=B_w, B_d=B_d, B_lost=B_lost,
        M=M, D=D, Kp=Kp, Kd=Kd, Ki=Ki,
        obs_gains=obs_gains, T_thr=T_thr, n_state=N_STATE,
    )


def intact_mean_steady_state_obs(aug: AugmentedSystemObs, tau_env: np.ndarray) -> np.ndarray:
    """Steady-state mean state under intact closed loop (analytic).

    With the observer fully converged and the integrator active, the SS
    is::

        eta = nu = eta_hat = nu_hat = 0
        b_hat   = 0                              (decays without innov)
        I       = Ki^{-1} · tau_env              (integrator carries load)
        tau_thr = -tau_env                       (lagged tau_cmd at SS)

    Note this differs from the 15-state ``intact_mean_steady_state`` in
    ``cqa.transient`` where b̂ is frozen at +tau_env. Here b̂ is dynamic
    and decays via -(1/T_b)·b̂ when innovation is zero. The PI integrator
    picks up the load instead, consistent with the "type-1" closed loop
    that has zero SS error against a constant disturbance.

    Audit
    -----
    Substituting into the dynamics::

        b_hat_dot = -b̂/T_b + K_b_pos·(η - η̂)        = 0 + 0 = 0  ✓
        eta_dot   = ν                                = 0           ✓
        M ν_dot   = -D·0 + τ_thr_ss + τ_env          = -τ_env + τ_env = 0  ✓
        eta_hat_dot = ν̂ + ω_c·(η - η̂)               = 0 + 0 = 0  ✓
        M ν̂_dot   = -D·0 + 0 + τ_cmd_ss + 0
                  = τ_cmd_ss = -(b̂ + Ki·I_ss) = -τ_env
                  WAIT — this should be 0 in SS, but we get -τ_env.

    The conflict: in the observer model, ν̂_dot at SS receives τ_cmd as
    forcing (no τ_env on the observer side, since the observer doesn't
    know about the env force directly). For ν̂_dot = 0 at SS we need
    τ_cmd_ss = 0, which would mean b̂_ss + Ki·I_ss = 0 — contradicting
    the truth force balance b̂_ss + Ki·I_ss = τ_env.

    Resolution: at intact SS, the observer's ν̂_dot is *not* zero unless
    we add the missing knowledge that τ_env is acting on truth. In a
    "real" passive observer this manifests as a small steady-state
    position offset: the controller's η̂ is biased so that K_a_pos·e
    provides the missing force on ν̂_dot. The exact analytic SS is then
    not purely closed-form in η̂.

    For the purpose of *transient* analysis (small perturbations around
    a quasi-SS), we initialise from the closed-form approximation above
    and let the dynamics relax over a brief settling window before
    injecting the WCF event. The diagnostic harness should run a long
    enough pre-WCF settling phase to remove this initial transient.
    """
    x_ss = np.zeros(aug.n_state)
    Ki_diag = np.array([aug.Ki[i, i] for i in range(3)])
    if np.any(np.abs(Ki_diag) < 1e-12):
        raise ValueError("Ki has zero diagonal; SS undefined without integrator")
    x_ss[IDX_INT] = tau_env / Ki_diag
    x_ss[IDX_TAU_THR] = -tau_env
    return x_ss


# ---------------------------------------------------------------------------
# Pulse-response driver
# ---------------------------------------------------------------------------


def pulse_response(
    aug: AugmentedSystemObs,
    t_grid: np.ndarray,
    tau_lost_t: np.ndarray,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Time-step x_dot = A x + B_lost · tau_lost(t)  on uniform t_grid.

    Uses ``expm(A·dt)`` precomputed for stability against the b_hat
    block's slow eigenvalues, with trapezoidal integration of the
    forcing — same approach as the 15-state diagnostic.

    Parameters
    ----------
    aug : 21-state augmented system.
    t_grid : (N,) uniform time grid [s].
    tau_lost_t : (N, 3) τ_lost(t) per DOF [N, N, Nm].
    x0 : (21,) initial perturbation around intact steady state. Defaults
        to zero (i.e. the result is the response of the perturbation
        from intact SS to a τ_lost pulse).

    Returns
    -------
    X : (N, 21) state trajectory.
    """
    from scipy.linalg import expm
    n = aug.n_state
    N = len(t_grid)
    if x0 is None:
        x0 = np.zeros(n)
    dt = float(t_grid[1] - t_grid[0])
    if not np.allclose(np.diff(t_grid), dt, rtol=1e-8):
        raise ValueError("t_grid must be uniform")
    Phi = expm(aug.A * dt)

    X = np.zeros((N, n))
    X[0] = x0
    for k in range(1, N):
        u_k = aug.B_lost @ tau_lost_t[k]
        u_km1 = aug.B_lost @ tau_lost_t[k - 1]
        # Trapezoidal forcing: integral of Phi(dt - s) u(s) ds approximated as
        # 0.5 dt (Phi u_{k-1} + u_k). For small dt this is accurate; we keep
        # this consistent with pulse_response_diagnostic.py for the 15-state
        # comparison.
        X[k] = Phi @ X[k - 1] + 0.5 * dt * (Phi @ u_km1 + u_k)
    return X


def pulse_response_with_lift_coupling(
    aug: AugmentedSystemObs,
    t_grid: np.ndarray,
    tau_lost_t: np.ndarray,
    b_hat0: np.ndarray,
    K_lift: float,
    x0: np.ndarray | None = None,
    n_iter: int = 2,
) -> np.ndarray:
    """pulse_response with a slender-body lift-coupling correction.

    The constant-body-frame b_hat assumption in pulse_response misses
    the post-WCF rotation of the wave/wind/current force vector as the
    vessel yaws by dpsi(t). Slender-body approximation: at small alpha
    the lateral force scales as F_y ~ -q*A*C_Y*sin(alpha), so
        dF_y/dpsi = -dF_y/dalpha = +q*A*C_Y = -F_x * (C_Y/C_D0) = -F_x * K
    where K is calibrated offline (cqa.config or scripts/calibrate_lift_coupling.py).

    Implementation: Picard iteration.
      iter 1: integrate without coupling (delta_b = 0), extract dpsi(t).
      iter 2: integrate with extra forcing (0, -F_x0 * K * dpsi(t), 0)
              injected through B_d, derived from iter 1's dpsi.
      iter 3+: refine using the previous iteration's dpsi.

    Default n_iter=2 applies the coupling once. n_iter=1 returns the
    UN-coupled result (for diagnostic / sanity comparison).

    Parameters
    ----------
    aug : 27-state augmented system.
    t_grid : (N,) uniform time grid.
    tau_lost_t : (N, 3) thruster-loss force time series.
    b_hat0 : (3,) intact-DP body-frame disturbance estimate
        (b_hat at t=0; treated constant baseline).
    K_lift : C_Y / C_D0, the lift-coupling scalar [per rad].
    x0 : (n_state,) initial perturbation. Defaults to zero.
    n_iter : Picard iterations (default 2; n_iter=1 returns the
        UN-coupled result, n_iter>=2 applies the lift coupling).

    Returns
    -------
    X : (N, n_state) state trajectory with coupling applied.
    """
    from scipy.linalg import expm
    n = aug.n_state
    N = len(t_grid)
    if x0 is None:
        x0 = np.zeros(n)
    dt = float(t_grid[1] - t_grid[0])
    if not np.allclose(np.diff(t_grid), dt, rtol=1e-8):
        raise ValueError("t_grid must be uniform")
    Phi = expm(aug.A * dt)

    Fx0 = float(b_hat0[0])
    coupling_y = -Fx0 * K_lift   # dF_y/dpsi  [N/rad]

    # Yaw deviation index in IDX_ETA: surge=0, sway=1, yaw=2.
    yaw_idx = IDX_ETA.start + 2

    # Start with zero coupling correction.
    delta_b_t = np.zeros((N, 3))
    X = np.zeros((N, n))

    for it in range(max(1, n_iter)):
        X[0] = x0
        for k in range(1, N):
            u_k = aug.B_lost @ tau_lost_t[k] + aug.B_d @ delta_b_t[k]
            u_km1 = aug.B_lost @ tau_lost_t[k - 1] + aug.B_d @ delta_b_t[k - 1]
            X[k] = Phi @ X[k - 1] + 0.5 * dt * (Phi @ u_km1 + u_k)
        # Update coupling forcing from this pass's dpsi(t).
        dpsi_t = X[:, yaw_idx]
        delta_b_t = np.zeros((N, 3))
        delta_b_t[:, 1] = coupling_y * dpsi_t

    return X


# ---------------------------------------------------------------------------
# Saturation-aware pulse response (sec.12.21.21.6 Option A)
# ---------------------------------------------------------------------------


def implicit_tau_cmd(aug: AugmentedSystemObs, x: np.ndarray) -> np.ndarray:
    """Recover the controller command ``tau_cmd`` from the state vector.

    The 27-state augmented model encodes the controller law

        tau_cmd = -Kp eta_hat - Kd nu_hat - b_hat - Ki I

    inside the ``tau_thr_dot`` row of ``A`` (see
    ``build_observer_augmented_system_full`` lines 255-261). For
    saturation-aware integration we need the *raw* command value at each
    integration step, prior to clipping; this helper exposes it.
    """
    eta_hat = x[IDX_ETA_HAT]
    nu_hat = x[IDX_NU_HAT]
    b_hat = x[IDX_B_HAT]
    I_int = x[IDX_INT]
    return -aug.Kp @ eta_hat - aug.Kd @ nu_hat - b_hat - aug.Ki @ I_int


def pulse_response_saturated(
    aug: AugmentedSystemObs,
    t_grid: np.ndarray,
    tau_lost_t: np.ndarray,
    clip_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Pulse response with allocator saturation on the controller command.

    Per sec.12.21.21.6 Option A: model the dominant non-linearity that
    drives the bf8_q10_w45 late-time spiral by clipping the implicit
    ``tau_cmd`` produced by the controller against a residual thrust
    polytope at every integration step. The clipped command feeds the
    physical thrust-lag block (and thence the truth ``nu_dot``), while
    the observer continues to see the un-clipped orders (matching the
    brucon CSOV ``use_tau_feedback = false`` default).

    The saturated dynamics are::

        x_dot = A x + B_lost tau_lost(t)
              + e_tau_thr  *  (1/T_thr) * (tau_cmd_clip(x) - tau_cmd_raw(x))

    where the bracketed correction subtracts the contribution of the
    *un-clipped* command (already baked into ``A @ x``) and re-injects
    the clipped command into the ``tau_thr_dot`` row. With
    ``tau_cmd_clip == tau_cmd_raw`` the correction vanishes and the
    trajectory matches :func:`pulse_response` to integrator order.

    Integration uses explicit RK4 with the uniform grid ``t_grid`` and
    linear interpolation of ``tau_lost(t)`` at the half-step; this
    handles the piecewise-linear clip cleanly while preserving the
    linear-case accuracy.

    Parameters
    ----------
    aug : AugmentedSystemObs
        21- (or 27-) state augmented system.
    t_grid : (N,) np.ndarray
        Uniform time grid [s].
    tau_lost_t : (N, 3) np.ndarray
        :math:`\\tau_{\\mathrm{lost}}(t)` forcing per DOF [N, N, N*m].
    clip_fn : callable
        Closure mapping a raw 3-vector command ``(cx, cy, cz)`` to a
        clipped 3-vector. Caller supplies the projection geometry (e.g.
        yaw-priority operational cap from
        :mod:`cqa.live_regime_b`). Must return a finite ``(3,)`` array.
    x0 : (n_state,) np.ndarray, optional
        Initial perturbation around intact SS. Defaults to zero.

    Returns
    -------
    X : (N, n_state) np.ndarray
        State trajectory.

    Notes
    -----
    The observer state (``eta_hat``, ``nu_hat``, ``b_hat``) is fed the
    un-clipped command, consistent with the brucon observer wiring. This
    is what drives the post-WCF integrator wind-up that the deterministic
    spiral mechanism (sec.12.21.21.6) requires: the controller keeps
    demanding more thrust than is delivered, the integrator keeps
    growing, and the observer's bias estimate keeps absorbing the
    innovation -- all while truth ``nu`` continues to drift because
    ``tau_thr`` is capped.
    """
    n = aug.n_state
    N = len(t_grid)
    if x0 is None:
        x0 = np.zeros(n)
    if tau_lost_t.shape != (N, 3):
        raise ValueError(
            f"tau_lost_t must be ({N}, 3); got {tau_lost_t.shape}"
        )
    dt = float(t_grid[1] - t_grid[0])
    if not np.allclose(np.diff(t_grid), dt, rtol=1e-8):
        raise ValueError("t_grid must be uniform")

    inv_T_thr = 1.0 / aug.T_thr
    A = aug.A
    B_lost = aug.B_lost

    # Pre-compute the (n_state, 3) projector that pumps a 3-vector
    # correction into the IDX_TAU_THR row.
    E_thr = np.zeros((n, 3))
    E_thr[IDX_TAU_THR, :] = np.eye(3)

    def f(x: np.ndarray, tau_lost_k: np.ndarray) -> np.ndarray:
        # Raw linear dynamics (un-clipped command already baked into A).
        x_dot = A @ x + B_lost @ tau_lost_k
        # Clip correction on the tau_thr_dot row only.
        tau_raw = implicit_tau_cmd(aug, x)
        tau_clip = np.asarray(clip_fn(tau_raw), dtype=float)
        if tau_clip.shape != (3,):
            raise ValueError(
                f"clip_fn must return a (3,) array; got {tau_clip.shape}"
            )
        delta = tau_raw - tau_clip
        x_dot += E_thr @ (-inv_T_thr * delta)
        return x_dot

    X = np.zeros((N, n))
    X[0] = x0
    for k in range(1, N):
        x_k = X[k - 1]
        tl_a = tau_lost_t[k - 1]
        tl_b = tau_lost_t[k]
        tl_mid = 0.5 * (tl_a + tl_b)
        k1 = f(x_k, tl_a)
        k2 = f(x_k + 0.5 * dt * k1, tl_mid)
        k3 = f(x_k + 0.5 * dt * k2, tl_mid)
        k4 = f(x_k + dt * k3, tl_b)
        X[k] = x_k + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return X
