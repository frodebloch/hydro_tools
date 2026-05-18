"""Linearity reconstruction diagnostic for bf8_q10_w45 post-WCF spiral.

Question (sec.12.21.21.27 follow-up): can the brucon post-WCF
position transient be reconstructed from a linear model driven by
brucon-extracted thrust signals?

Three reconstruction tests:

  Test A (closed-loop saturation forcing):
    Inject delta_tau_brucon = OrderTau - T_delivered on the tau_thr
    channel; let the linear closed loop (controller + observer +
    integrator + wave filter) respond. x0 = 0, tau_lost = 0. Tests
    whether linear closed-loop dynamics + saturation forcing alone
    reproduce the spiral.

  Test B (open-loop plant response to brucon T_delivered):
    Drive tau_thr toward (T_delivered_brucon - T_pre_WCF) directly,
    bypassing the linear controller entirely. Tests whether the
    linear plant (mass, damping) reproduces the rigid-body response
    to brucon's actual delivered thrust history. If Test B matches
    brucon eta well, the linear plant is fine and any gap in Test A
    is in the controller/observer.

  Test C (open-loop plant response to brucon OrderTau):
    Drive tau_thr toward (OrderTau_brucon - OrderTau_pre_WCF). This
    is what brucon's vessel would have moved like if no saturation
    had occurred. Compared to brucon truth, the difference
    (Test C eta - brucon eta) is the saturation-attributable
    excursion. Compared to Test B, gives the "what saturation cost
    us" decomposition.

If YES on both outlier and calm seeds -> linearity holds; the spiral
is a saturation-feedback phenomenon. Option 2 (analytical excursion
distribution from a distribution of saturation events) becomes
viable: the per-event impulse response H_eta-from-delta-tau computed
from the augmented A-matrix can be convolved with a distribution of
delta_tau events to predict the distribution of added excursion.

If NO -> closed-loop nonlinearity dominates and the analytical
superposition fails. Option 2 needs more structure (e.g. recovering
the operating-point linearisation per realisation) or we fall back to
Option 1 (stochastic MC).

Construction
------------
Common: extract per-seed time series on [T_WCF, T_WCF + 180 s]:

  OrderTau(t):    cols 34/35/36 -> N, N, N*m (x1e3 from kN, kN*m)
  T_delivered(t): cols 7/8/9    -> N, N, N*m
  delta_tau(t):   OrderTau - T_delivered

Pre-WCF references (mean over [T_WCF - 30 s, T_WCF - 5 s], same
units):
  OrderTau_pre, T_pre

Test A integrates::

  x_dot = A @ x + E_tau_thr @ (-1/T_thr * delta_tau_brucon(t))

with x0 = 0 and tau_lost = 0. The implicit controller in A handles
its own demand response; only the saturation gap is injected
externally.

Test B and C integrate::

  x_dot = A @ x + E_tau_thr @ (1/T_thr * (target(t) - tau_cmd_implicit(x)))

with target(t) = (T_delivered_brucon(t) - T_pre) for Test B, or
(OrderTau_brucon(t) - OrderTau_pre) for Test C. The correction
overrides the implicit linear controller demand and drives tau_thr
toward the brucon-extracted target. RK4, dt = 0.1 s. x0 = 0.

Seeds tested
------------
- 1012 (outlier, P95 sway ~9 m, the worst-case seed identified in
  sec.12.21.21.9).
- 1001 (calm, expected P95 < 4 m).

Run:
  PYTHONPATH=. .venv/bin/python scripts/p7_brucon_validation/linearity_reconstruction_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
ROOT = THIS.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqa.live_decision import _build_aug_for_live  # noqa: E402
from cqa.config import csov_default_config  # noqa: E402
from cqa.transient_obs import IDX_ETA, IDX_TAU_THR, IDX_INT, IDX_B_HAT, IDX_ETA_HAT, IDX_NU_HAT, implicit_tau_cmd  # noqa: E402

T_WCF = 1560.0
T_HORIZON = 180.0
DT_INT = 0.1  # 10 Hz, matches brucon log fs

# Column indices in brucon .out (0-indexed)
COL_T = 0
COL_TX, COL_TY, COL_TZ = 7, 8, 9
COL_DRIFT_X, COL_DRIFT_Y, COL_DRIFT_MZ = 10, 11, 12
COL_WIND_X, COL_WIND_Y, COL_WIND_MZ = 14, 15, 16
COL_CUR_X, COL_CUR_Y, COL_CUR_MZ = 18, 19, 20
COL_SURGEDEV, COL_SWAYDEV, COL_HEADINGDEV = 24, 25, 22
COL_ORDER_SURGE, COL_ORDER_SWAY, COL_ORDER_YAW = 34, 35, 36

SEEDS = {
    "outlier_1012": 1012,
    "calm_1001": 1001,
}


def load_seed(seed: int) -> np.ndarray:
    f = THIS / "work" / f"bf8_q10_w45_seed{seed}" / f"bf8_q10_w45_seed{seed}.out"
    return np.loadtxt(f, skiprows=1)


def extract_delta_tau(data: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Extract delta_tau_brucon = OrderTau - T_delivered on the target grid.

    Returns (N, 3) array in N, N, N*m. brucon stores forces in kN and
    moments in kN*m -> multiply by 1e3.
    """
    t_b = data[:, COL_T]
    t_abs = T_WCF + t_grid  # absolute brucon time

    def interp(col_o, col_d):
        order = data[:, col_o]
        deliv = data[:, col_d]
        delta = order - deliv  # in kN or kN*m
        return np.interp(t_abs, t_b, delta) * 1e3  # -> N or N*m

    dtau = np.column_stack([
        interp(COL_ORDER_SURGE, COL_TX),
        interp(COL_ORDER_SWAY, COL_TY),
        interp(COL_ORDER_YAW, COL_TZ),
    ])
    return dtau


def extract_brucon_sway(data: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """SwayDev(t) - SwayDev(T_WCF) on the target grid. Units: m."""
    t_b = data[:, COL_T]
    sway_full = data[:, COL_SWAYDEV]
    t_abs = T_WCF + t_grid
    sway_at_wcf = np.interp(T_WCF, t_b, sway_full)
    sway = np.interp(t_abs, t_b, sway_full)
    return sway - sway_at_wcf


def extract_brucon_surge(data: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """SurgeDev(t) - SurgeDev(T_WCF) on the target grid. Units: m."""
    t_b = data[:, COL_T]
    surge_full = data[:, COL_SURGEDEV]
    t_abs = T_WCF + t_grid
    surge_at_wcf = np.interp(T_WCF, t_b, surge_full)
    surge = np.interp(t_abs, t_b, surge_full)
    return surge - surge_at_wcf


def extract_target_for_test_BC(
    data: np.ndarray, t_grid: np.ndarray, mode: str
) -> np.ndarray:
    """Extract pre-WCF-demeaned target signal for Test B or C.

    mode == "delivered" -> T_delivered(t) - T_pre  (Test B)
    mode == "order"     -> OrderTau(t) - OrderTau_pre  (Test C)

    Returns (N, 3) array in N, N, N*m (converted from kN).
    """
    t_b = data[:, COL_T]
    pre_mask = (t_b >= T_WCF - 30.0) & (t_b <= T_WCF - 5.0)

    if mode == "delivered":
        cols = [COL_TX, COL_TY, COL_TZ]
    elif mode == "order":
        cols = [COL_ORDER_SURGE, COL_ORDER_SWAY, COL_ORDER_YAW]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    t_abs = T_WCF + t_grid
    out = np.zeros((len(t_grid), 3))
    for i, c in enumerate(cols):
        sig = data[:, c]
        sig_pre = float(sig[pre_mask].mean())
        sig_on_grid = np.interp(t_abs, t_b, sig)
        out[:, i] = (sig_on_grid - sig_pre) * 1e3  # kN -> N (or kN*m -> N*m)
    return out


def reconstruct_linear(aug, t_grid: np.ndarray, dtau_t: np.ndarray) -> np.ndarray:
    """Test A: x_dot = A @ x + E_thr @ (-1/T_thr * dtau(t)) by RK4.

    Closed-loop linear response to brucon delta_tau injected on the
    tau_thr channel. Linear controller, observer, integrator, wave
    filter all active and responding via A. x0 = 0.
    """
    n = aug.n_state
    N = len(t_grid)
    A = aug.A
    inv_T_thr = 1.0 / aug.T_thr

    E_thr = np.zeros((n, 3))
    E_thr[IDX_TAU_THR, :] = np.eye(3)

    dt = float(t_grid[1] - t_grid[0])

    def forcing(k):
        return -inv_T_thr * dtau_t[k]

    def forcing_mid(k):
        return -inv_T_thr * 0.5 * (dtau_t[k] + dtau_t[k + 1])

    def f(x, force_vec):
        return A @ x + E_thr @ force_vec

    X = np.zeros((N, n))
    for k in range(1, N):
        x_k = X[k - 1]
        fa = forcing(k - 1)
        fm = forcing_mid(k - 1)
        fb = forcing(k)
        k1 = f(x_k, fa)
        k2 = f(x_k + 0.5 * dt * k1, fm)
        k3 = f(x_k + 0.5 * dt * k2, fm)
        k4 = f(x_k + dt * k3, fb)
        X[k] = x_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return X


def reconstruct_linear_with_lift(
    aug, t_grid: np.ndarray, dtau_t: np.ndarray,
    b_hat0: np.ndarray, K_lift: float, n_iter: int = 3,
) -> np.ndarray:
    """Test A + K_lift: closed-loop linear + delta_tau + slender-body
    lift coupling via Picard iteration.

    Coupling forcing per pulse_response_with_lift_coupling:
        coupling_y(t) = -Fx0 * K_lift * dpsi(t)   [N]
    where Fx0 = b_hat0[0] (intact-DP surge bias estimate in N) and
    dpsi(t) is the yaw deviation from the previous Picard iterate.
    Injected via B_d (the body-frame disturbance input matrix).
    """
    n = aug.n_state
    N = len(t_grid)
    A = aug.A
    B_d = aug.B_d
    inv_T_thr = 1.0 / aug.T_thr

    E_thr = np.zeros((n, 3))
    E_thr[IDX_TAU_THR, :] = np.eye(3)

    dt = float(t_grid[1] - t_grid[0])
    Fx0 = float(b_hat0[0])
    coupling_y_per_rad = -Fx0 * K_lift  # N/rad
    yaw_idx = IDX_ETA.start + 2

    delta_b_t = np.zeros((N, 3))
    X = np.zeros((N, n))

    def forcing_dtau(k):
        return -inv_T_thr * dtau_t[k]

    def forcing_dtau_mid(k):
        return -inv_T_thr * 0.5 * (dtau_t[k] + dtau_t[k + 1])

    def f(x, force_thr, delta_b):
        return A @ x + E_thr @ force_thr + B_d @ delta_b

    for it in range(max(1, n_iter)):
        X[0] = np.zeros(n)
        for k in range(1, N):
            x_k = X[k - 1]
            fa = forcing_dtau(k - 1)
            fm = forcing_dtau_mid(k - 1)
            fb = forcing_dtau(k)
            db_a = delta_b_t[k - 1]
            db_m = 0.5 * (delta_b_t[k - 1] + delta_b_t[k])
            db_b = delta_b_t[k]
            k1 = f(x_k, fa, db_a)
            k2 = f(x_k + 0.5 * dt * k1, fm, db_m)
            k3 = f(x_k + 0.5 * dt * k2, fm, db_m)
            k4 = f(x_k + dt * k3, fb, db_b)
            X[k] = x_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Update coupling forcing from this iterate's dpsi(t).
        dpsi_t = X[:, yaw_idx]
        delta_b_t = np.zeros((N, 3))
        delta_b_t[:, 1] = coupling_y_per_rad * dpsi_t

    return X


def reconstruct_open_loop(
    aug, t_grid: np.ndarray, target_t: np.ndarray
) -> np.ndarray:
    """Test B/C: override the linear controller, drive tau_thr toward
    target(t) directly.

    Implementation: the A-matrix bakes
        tau_thr_dot_row(x) = (1/T_thr) * (tau_cmd_implicit(x) - tau_thr)
    where tau_cmd_implicit = -Kp eta_hat - Kd nu_hat - bhat - Ki I.
    To override with brucon target(t), inject a correction that
    cancels tau_cmd_implicit and adds target(t):
        correction = E_thr @ (1/T_thr * (target(t) - tau_cmd_implicit(x)))
    Net effect on tau_thr_dot:
        tau_thr_dot = (1/T_thr) * (target(t) - tau_thr)
    The plant response (eta, nu) and observer (eta_hat, nu_hat, bhat)
    all still evolve via A from the (now externally driven) tau_thr.
    """
    n = aug.n_state
    N = len(t_grid)
    A = aug.A
    inv_T_thr = 1.0 / aug.T_thr

    E_thr = np.zeros((n, 3))
    E_thr[IDX_TAU_THR, :] = np.eye(3)

    dt = float(t_grid[1] - t_grid[0])

    def f(x, target_vec):
        tau_imp = implicit_tau_cmd(aug, x)
        correction = inv_T_thr * (target_vec - tau_imp)
        return A @ x + E_thr @ correction

    X = np.zeros((N, n))
    for k in range(1, N):
        x_k = X[k - 1]
        ta = target_t[k - 1]
        tm = 0.5 * (target_t[k - 1] + target_t[k])
        tb = target_t[k]
        k1 = f(x_k, ta)
        k2 = f(x_k + 0.5 * dt * k1, tm)
        k3 = f(x_k + 0.5 * dt * k2, tm)
        k4 = f(x_k + dt * k3, tb)
        X[k] = x_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return X


def extract_b_hat_pre_wcf(seed: int) -> np.ndarray:
    """Extract the brucon NPO bias estimate at T_WCF (mean over pre-WCF
    window [T_WCF - 30, T_WCF - 5]). Returns (3,) in N, N, N*m.
    """
    f = THIS / "work" / f"bf8_q10_w45_seed{seed}" / f"bf8_q10_w45_seed{seed}_estimator.out"
    d = np.genfromtxt(f, names=True, delimiter='\t')
    m = (d["Time"] >= T_WCF - 30.0) & (d["Time"] <= T_WCF - 5.0)
    return np.array([
        d["EstBiasSurge"][m].mean() * 1e3,
        d["EstBiasSway"][m].mean() * 1e3,
        d["EstBiasYaw"][m].mean() * 1e3,
    ])


def extract_observer_state_at_wcf(seed: int) -> dict:
    """Extract observer (eta_hat, nu_hat, b_hat) at t=T_WCF as a mean
    over [T_WCF - 5, T_WCF + 0.5] s. Returns dict with arrays of shape (3,)
    in deviation coords (subtract pre-WCF mean for eta_hat); nu_hat in
    body frame; b_hat in N / N*m.

    Note: eta_hat is returned as the deviation from its pre-WCF mean,
    matching the deviation-coordinate framing used by extract_brucon_sway.
    """
    f = THIS / "work" / f"bf8_q10_w45_seed{seed}" / f"bf8_q10_w45_seed{seed}_estimator.out"
    d = np.genfromtxt(f, names=True, delimiter='\t')
    t = d["Time"]
    pre = (t >= T_WCF - 30.0) & (t <= T_WCF - 5.0)
    at = (t >= T_WCF - 1.0) & (t <= T_WCF + 1.0)
    # eta_hat in cqa convention: [surge_dev, sway_dev, heading_dev]
    # Brucon estimator logs EstPosX, EstPosY (NED), EstHeading. Use the
    # deviations directly (mean(at) - mean(pre)) which collapses
    # NED->body for small heading dev.
    eta_hat_dev = np.array([
        d["LongEst"][at].mean() - d["LongEst"][pre].mean(),
        d["LatEst"][at].mean() - d["LatEst"][pre].mean(),
        d["EstHeading"][at].mean() - d["EstHeading"][pre].mean(),
    ])
    nu_hat = np.array([
        d["EstVelSurge"][at].mean(),
        d["EstVelSway"][at].mean(),
        d["EstRot"][at].mean(),
    ])
    b_hat = np.array([
        d["EstBiasSurge"][at].mean() * 1e3,
        d["EstBiasSway"][at].mean() * 1e3,
        d["EstBiasYaw"][at].mean() * 1e3,
    ])
    return {"eta_hat": eta_hat_dev, "nu_hat": nu_hat, "b_hat": b_hat}


def extract_order_tau_at_wcf(data: np.ndarray) -> np.ndarray:
    """OrderTau averaged over [T_WCF - 1, T_WCF + 1] s in N, N, N*m.

    Deviation-from-pre form: mean(at) - mean(pre). Pre-WCF the
    controller demand is whatever the closed loop has settled to;
    cqa's tau_cmd in deviation coords is the *deviation* from this.
    """
    t = data[:, COL_T]
    pre = (t >= T_WCF - 30.0) & (t <= T_WCF - 5.0)
    at = (t >= T_WCF - 1.0) & (t <= T_WCF + 1.0)
    out = np.zeros(3)
    for i, c in enumerate([COL_ORDER_SURGE, COL_ORDER_SWAY, COL_ORDER_YAW]):
        out[i] = (data[at, c].mean() - data[pre, c].mean()) * 1e3
    return out


def reconstruct_integrator_ic(
    seed: int, data: np.ndarray, Kp: np.ndarray, Kd: np.ndarray, Ki: np.ndarray,
) -> np.ndarray:
    """Solve for I(T_WCF) from the PI controller law in deviation coords:

        OrderTau_dev = -Kp @ eta_hat_dev - Kd @ nu_hat_dev - b_hat_dev - Ki @ I

    => I = -(OrderTau_dev + Kp @ eta_hat_dev + Kd @ nu_hat_dev + b_hat_dev) / Ki

    All right-hand quantities are deviations from their pre-WCF mean.
    b_hat_dev = b_hat(T_WCF) - b_hat_pre (the pre-WCF bias is what the
    cqa SS would converge to; deviation coords zero it).

    Notes:
    - nu_hat_dev: pre-WCF mean is close to 0 (vessel station-keeping)
      so we just use nu_hat(T_WCF).
    - This reconstruction depends on the Kd convention used (cqa-default
      vs brucon). Both are computed and printed for cross-check.
    """
    pre = (data[:, COL_T] >= T_WCF - 30.0) & (data[:, COL_T] <= T_WCF - 5.0)
    obs = extract_observer_state_at_wcf(seed)
    b_hat_pre = extract_b_hat_pre_wcf(seed)
    b_hat_dev = obs["b_hat"] - b_hat_pre
    order_tau_dev = extract_order_tau_at_wcf(data)
    I = -(order_tau_dev + Kp @ obs["eta_hat"] + Kd @ obs["nu_hat"] + b_hat_dev) / np.diag(Ki)
    return I


def extract_env_force_minus_bhat(
    data: np.ndarray, t_grid: np.ndarray, b_hat_pre: np.ndarray,
) -> np.ndarray:
    """HYP 5 variant: F_env(t) - b_hat_pre, NOT demeaned.

    Rationale: brucon's intact-state controller balances steady mean
    env force via the bias feedforward path. The b_hat_pre estimate is
    the long-time average (NPO time constant 1000 s) of that mean
    force. The previous demeaning [T_WCF-30, T_WCF-5] used a 25 s
    window of the LF process, which is a noisy sample, not the
    expectation. After WCF the thrusters can't sustain that mean force
    fully (alpha < 1), so the un-cancelled component drives drift.

    Sign convention: pre-WCF Newton balance gives T_pre + F_env_pre = 0,
    and the NPO converges b_hat -> -T_pre = F_env_pre. So b_hat_pre is
    the long-time mean of F_env, and (F_env(t) - b_hat_pre) is the
    zero-mean perturbation (cf. demean variant). For HYP 5 we want the
    DC component PRESERVED: the relevant un-cancelled DC after WCF is
    NOT zero because the thrusters at alpha < 1 cannot sustain the
    full mean. So we keep the absolute F_env (NO subtraction of
    b_hat_pre) and let the closed loop's bias-FF cancel what it can.
    """
    t_b = data[:, COL_T]
    fx_full = data[:, COL_DRIFT_X] + data[:, COL_WIND_X] + data[:, COL_CUR_X]
    fy_full = data[:, COL_DRIFT_Y] + data[:, COL_WIND_Y] + data[:, COL_CUR_Y]
    fmz_full = data[:, COL_DRIFT_MZ] + data[:, COL_WIND_MZ] + data[:, COL_CUR_MZ]
    t_abs = T_WCF + t_grid
    out = np.zeros((len(t_grid), 3))
    # Return absolute env force in N / N*m (NOT demeaned, NOT minus b_hat).
    out[:, 0] = np.interp(t_abs, t_b, fx_full) * 1e3
    out[:, 1] = np.interp(t_abs, t_b, fy_full) * 1e3
    out[:, 2] = np.interp(t_abs, t_b, fmz_full) * 1e3
    return out


def extract_env_force_perturbation(
    data: np.ndarray, t_grid: np.ndarray,
) -> np.ndarray:
    """Extract total env force perturbation F_env(t) - F_env_pre.

    F_env = Drift + Wind + Cur, body-frame, in kN/kN*m. Returns (N, 3)
    in N, N, N*m, demeaned by pre-WCF mean over [T_WCF - 30, T_WCF - 5].

    Sanity-checked against pre-WCF Newton balance:
    T_pre + F_env_pre ~= 0 (verified on bf8_q10_w45 seed 1012:
    residuals 29 / -54 / -610 kN/kN/kN*m, within wave-PSD variance).
    """
    t_b = data[:, COL_T]
    pre = (t_b >= T_WCF - 30.0) & (t_b <= T_WCF - 5.0)
    fx_full = data[:, COL_DRIFT_X] + data[:, COL_WIND_X] + data[:, COL_CUR_X]
    fy_full = data[:, COL_DRIFT_Y] + data[:, COL_WIND_Y] + data[:, COL_CUR_Y]
    fmz_full = data[:, COL_DRIFT_MZ] + data[:, COL_WIND_MZ] + data[:, COL_CUR_MZ]
    fx_pre = float(fx_full[pre].mean())
    fy_pre = float(fy_full[pre].mean())
    fmz_pre = float(fmz_full[pre].mean())
    t_abs = T_WCF + t_grid
    out = np.zeros((len(t_grid), 3))
    out[:, 0] = (np.interp(t_abs, t_b, fx_full) - fx_pre) * 1e3
    out[:, 1] = (np.interp(t_abs, t_b, fy_full) - fy_pre) * 1e3
    out[:, 2] = (np.interp(t_abs, t_b, fmz_full) - fmz_pre) * 1e3
    return out
    """Extract total env force perturbation F_env(t) - F_env_pre.

    F_env = Drift + Wind + Cur, body-frame, in kN/kN*m. Returns (N, 3)
    in N, N, N*m, demeaned by pre-WCF mean over [T_WCF - 30, T_WCF - 5].

    Sanity-checked against pre-WCF Newton balance:
    T_pre + F_env_pre ~= 0 (verified on bf8_q10_w45 seed 1012:
    residuals 29 / -54 / -610 kN/kN/kN*m, within wave-PSD variance).
    """
    t_b = data[:, COL_T]
    pre = (t_b >= T_WCF - 30.0) & (t_b <= T_WCF - 5.0)

    def channel(col_drift, col_wind, col_cur):
        total = data[:, col_drift] + data[:, col_wind] + data[:, col_cur]
        return total

    fx_full = channel(COL_DRIFT_X, COL_WIND_X, COL_CUR_X)
    fy_full = channel(COL_DRIFT_Y, COL_WIND_Y, COL_CUR_Y)
    fmz_full = channel(COL_DRIFT_MZ, COL_WIND_MZ, COL_CUR_MZ)

    fx_pre = float(fx_full[pre].mean())
    fy_pre = float(fy_full[pre].mean())
    fmz_pre = float(fmz_full[pre].mean())

    t_abs = T_WCF + t_grid
    out = np.zeros((len(t_grid), 3))
    out[:, 0] = (np.interp(t_abs, t_b, fx_full) - fx_pre) * 1e3
    out[:, 1] = (np.interp(t_abs, t_b, fy_full) - fy_pre) * 1e3
    out[:, 2] = (np.interp(t_abs, t_b, fmz_full) - fmz_pre) * 1e3
    return out


def reconstruct_with_env_and_lift(
    aug, t_grid: np.ndarray, dtau_t: np.ndarray, df_env_t: np.ndarray,
    b_hat0: np.ndarray, K_lift: float, n_iter: int = 3,
) -> np.ndarray:
    """Test F: Test E + slender-body lift coupling via Picard iteration.

    Adds coupling_y(t) = -Fx0 * K_lift * dpsi(t) to the sway channel of
    B_d on top of (F_env(t) - F_env_pre).  Otherwise identical to
    reconstruct_with_env.  Picard iterates dpsi from the previous run.
    """
    n = aug.n_state
    N = len(t_grid)
    A = aug.A
    B_d = aug.B_d
    inv_T_thr = 1.0 / aug.T_thr

    E_thr = np.zeros((n, 3))
    E_thr[IDX_TAU_THR, :] = np.eye(3)

    dt = float(t_grid[1] - t_grid[0])
    Fx0 = float(b_hat0[0])
    coupling_y_per_rad = -Fx0 * K_lift
    yaw_idx = IDX_ETA.start + 2

    delta_b_t = np.zeros((N, 3))  # Picard iterate of lift coupling
    X = np.zeros((N, n))

    def f(x, df, dtau, db):
        return A @ x + B_d @ (df + db) + E_thr @ (-inv_T_thr * dtau)

    for it in range(max(1, n_iter)):
        X[0] = np.zeros(n)
        for k in range(1, N):
            x_k = X[k - 1]
            df_a = df_env_t[k - 1]; df_b = df_env_t[k]; df_m = 0.5 * (df_a + df_b)
            dt_a = dtau_t[k - 1]; dt_b = dtau_t[k]; dt_m = 0.5 * (dt_a + dt_b)
            db_a = delta_b_t[k - 1]; db_b = delta_b_t[k]; db_m = 0.5 * (db_a + db_b)
            k1 = f(x_k, df_a, dt_a, db_a)
            k2 = f(x_k + 0.5 * dt * k1, df_m, dt_m, db_m)
            k3 = f(x_k + 0.5 * dt * k2, df_m, dt_m, db_m)
            k4 = f(x_k + dt * k3, df_b, dt_b, db_b)
            X[k] = x_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        dpsi_t = X[:, yaw_idx]
        delta_b_t = np.zeros((N, 3))
        delta_b_t[:, 1] = coupling_y_per_rad * dpsi_t

    return X


def extract_env_force_pre_wcf(data: np.ndarray) -> np.ndarray:
    """Mean total env force over [T_WCF-30, T_WCF-5]. Returns (3,) in N, N, N*m.

    Used to compute the intact pre-WCF steady state via
    ``np.linalg.solve(aug.A, -aug.B_d @ tau_env_pre)`` as the initial
    condition for reconstructions (sec.12.21.21.28b).
    """
    t_b = data[:, COL_T]
    pre = (t_b >= T_WCF - 30.0) & (t_b <= T_WCF - 5.0)
    fx = (data[:, COL_DRIFT_X] + data[:, COL_WIND_X] + data[:, COL_CUR_X])[pre].mean()
    fy = (data[:, COL_DRIFT_Y] + data[:, COL_WIND_Y] + data[:, COL_CUR_Y])[pre].mean()
    fmz = (data[:, COL_DRIFT_MZ] + data[:, COL_WIND_MZ] + data[:, COL_CUR_MZ])[pre].mean()
    return np.array([fx, fy, fmz]) * 1e3  # kN -> N


def intact_steady_state(aug, tau_env_pre: np.ndarray) -> np.ndarray:
    """Solve A x = -B_d @ tau_env_pre for the intact pre-WCF steady state.

    Equivalent to the brucon scenario evolved long enough for the
    observer, integrator, and bias estimator to settle before
    T_WCF. Required as the IC for post-WCF linear reconstructions
    so that b_hat(0) carries the intact env-force compensation
    rather than being zero.
    """
    rhs = -aug.B_d @ tau_env_pre
    return np.linalg.solve(aug.A, rhs)


def reconstruct_with_env_antiwindup(
    aug, t_grid: np.ndarray, dtau_t: np.ndarray, df_env_t: np.ndarray,
    F_max: np.ndarray, Ki_diag: np.ndarray, x0: np.ndarray | None = None,
) -> np.ndarray:
    """Test E2: Test E with brucon-style integrator anti-windup.

    Per brucon ``libs/common/regulators/pid.cpp:23`` and
    ``include/brucon/regulators/pid.h:31-32``, brucon's PI integrator
    clamps ``integral_term_`` to ``+-0.75 * max_saturation_limit_``
    per axis. In our state-space form ``I_dot = eta_hat`` and the
    contribution to tau_cmd is ``Ki * I``, so the equivalent state
    clamp is ``|I| <= 0.75 * F_max / Ki`` (per-axis).

    Implementation: after each RK4 step, project ``X[k, IDX_INT]``
    onto the box. This is a naive state projection (does NOT
    implement the brucon "freeze integration at limit" detail) but
    captures the wind-up bound which is the dominant effect.
    """
    n = aug.n_state
    N = len(t_grid)
    A = aug.A
    B_d = aug.B_d
    inv_T_thr = 1.0 / aug.T_thr

    E_thr = np.zeros((n, 3))
    E_thr[IDX_TAU_THR, :] = np.eye(3)

    dt = float(t_grid[1] - t_grid[0])

    # Per-axis integrator-state clamp (m.s or rad.s)
    I_max = np.array([
        0.75 * F_max[i] / max(Ki_diag[i], 1e-12)
        for i in range(3)
    ])

    def f(x, df, dtau):
        return A @ x + B_d @ df + E_thr @ (-inv_T_thr * dtau)

    X = np.zeros((N, n))
    if x0 is not None:
        X[0] = x0
    for k in range(1, N):
        x_k = X[k - 1]
        df_a = df_env_t[k - 1]; df_b = df_env_t[k]; df_m = 0.5 * (df_a + df_b)
        dt_a = dtau_t[k - 1]; dt_b = dtau_t[k]; dt_m = 0.5 * (dt_a + dt_b)
        k1 = f(x_k, df_a, dt_a)
        k2 = f(x_k + 0.5 * dt * k1, df_m, dt_m)
        k3 = f(x_k + 0.5 * dt * k2, df_m, dt_m)
        k4 = f(x_k + dt * k3, df_b, dt_b)
        X[k] = x_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Anti-windup: project integrator state onto the box.
        for i in range(3):
            X[k, IDX_INT.start + i] = np.clip(
                X[k, IDX_INT.start + i], -I_max[i], I_max[i]
            )
    return X


def reconstruct_with_env(
    aug, t_grid: np.ndarray, dtau_t: np.ndarray, df_env_t: np.ndarray,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Test E: closed-loop response to delta_tau + env-force perturbation.

    Integrates::
        x_dot = A @ x
              + B_d @ (F_env(t) - F_env_pre)
              + E_thr @ (-1/T_thr * delta_tau_brucon(t))

    by RK4. The env-force perturbation drives the rigid body via B_d
    (-> Minv @ F into nu_dot); the linear closed loop generates its
    own controller demand via A; the brucon-extracted saturation gap
    is layered on the tau_thr channel.
    """
    n = aug.n_state
    N = len(t_grid)
    A = aug.A
    B_d = aug.B_d
    inv_T_thr = 1.0 / aug.T_thr

    E_thr = np.zeros((n, 3))
    E_thr[IDX_TAU_THR, :] = np.eye(3)

    dt = float(t_grid[1] - t_grid[0])

    def f(x, df, dtau):
        return A @ x + B_d @ df + E_thr @ (-inv_T_thr * dtau)

    X = np.zeros((N, n))
    for k in range(1, N):
        x_k = X[k - 1]
        df_a = df_env_t[k - 1]
        df_b = df_env_t[k]
        df_m = 0.5 * (df_a + df_b)
        dt_a = dtau_t[k - 1]
        dt_b = dtau_t[k]
        dt_m = 0.5 * (dt_a + dt_b)
        k1 = f(x_k, df_a, dt_a)
        k2 = f(x_k + 0.5 * dt * k1, df_m, dt_m)
        k3 = f(x_k + 0.5 * dt * k2, df_m, dt_m)
        k4 = f(x_k + dt * k3, df_b, dt_b)
        X[k] = x_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return X


def _build_aug_brucon_kd(cfg) -> "AugmentedSystemObs":
    """Build aug with brucon Kd convention (subtract_open_loop_damping=False).

    Hypothesis 3 test: brucon's PID does NOT subtract open-loop D when
    setting Kd = 2 zeta M omega; cqa default subtracts D. Result: cqa
    runs less-damped than brucon and over-predicts P95.
    """
    from cqa.vessel import LinearVesselModel
    from cqa.controller import LinearDpController
    from cqa.transient_obs import build_observer_augmented_system_full, csov_observer_gains
    vp = cfg.vessel
    cp_ctrl = cfg.controller
    vessel = LinearVesselModel.from_config(vp)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D,
        omega_n=cp_ctrl.omega_n, zeta=cp_ctrl.zeta,
        subtract_open_loop_damping=False,
    )
    obs_gains = csov_observer_gains(Tp_s=10.0)
    return build_observer_augmented_system_full(
        vessel, controller, obs_gains=obs_gains,
        T_thr=cp_ctrl.thruster_time_constant_s,
    )


def run() -> None:
    cfg = csov_default_config()
    aug = _build_aug_for_live(cfg, Tp_obs_s=10.0)
    aug_brucon_kd = _build_aug_brucon_kd(cfg)
    print(
        "Kd convention comparison [sway]:\n"
        f"  cqa default  (subtract_open_loop_damping=True ): Kd_y = {aug.Kd[1,1]:.3e} N/(m/s)\n"
        f"  brucon (subtract_open_loop_damping=False):       Kd_y = {aug_brucon_kd.Kd[1,1]:.3e} N/(m/s)\n"
        f"  ratio brucon/cqa = {aug_brucon_kd.Kd[1,1]/aug.Kd[1,1]:.3f}"
    )
    K_lift = float(getattr(cfg.vessel, "lift_coupling_K_per_rad", 0.0))
    print(f"Built aug system: n_state = {aug.n_state}, T_thr = {aug.T_thr:.3f} s")
    print(f"K_lift from cfg.vessel.lift_coupling_K_per_rad = {K_lift:.3f} per rad")

    # Brucon intact_cap (from roll-up): surge 1.36 MN, sway 1.70 MN, yaw 86.4 MN.m
    F_max = np.array([1.36e6, 1.70e6, 86.4e6])
    Ki_diag = np.array([
        aug.A[IDX_TAU_THR.start + i, IDX_INT.start + i] * -aug.T_thr
        for i in range(3)
    ])
    I_max = 0.75 * F_max / Ki_diag
    print(f"Anti-windup: F_max = {F_max/1e3} kN/kN/kN.m")
    print(f"             Ki = {Ki_diag} N/(m.s)/...")
    print(f"             I_max = {I_max} m.s/m.s/rad.s")

    t_grid = np.arange(0.0, T_HORIZON + DT_INT * 0.5, DT_INT)
    N = len(t_grid)
    print(f"Integration grid: N = {N}, dt = {DT_INT} s, horizon = {T_HORIZON} s")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    for col, (label, seed) in enumerate(SEEDS.items()):
        data = load_seed(seed)
        dtau = extract_delta_tau(data, t_grid)
        df_env = extract_env_force_perturbation(data, t_grid)
        brucon_sway = extract_brucon_sway(data, t_grid)
        b_hat0 = extract_b_hat_pre_wcf(seed)

        # Test A: closed-loop response to delta_tau alone
        X_A = reconstruct_linear(aug, t_grid, dtau)
        sway_A = X_A[:, IDX_ETA][:, 1]
        yaw_A = X_A[:, IDX_ETA][:, 2]

        # Test E: Test A + brucon env-force perturbation via B_d
        X_E = reconstruct_with_env(aug, t_grid, dtau, df_env)
        sway_E = X_E[:, IDX_ETA][:, 1]
        yaw_E = X_E[:, IDX_ETA][:, 2]

        # Test E2: Test E with brucon-style integrator anti-windup
        X_E2 = reconstruct_with_env_antiwindup(aug, t_grid, dtau, df_env, F_max, Ki_diag)
        sway_E2 = X_E2[:, IDX_ETA][:, 1]
        yaw_E2 = X_E2[:, IDX_ETA][:, 2]
        I_E_peak = np.abs(X_E[:, IDX_INT][:, 1]).max()
        I_E2_peak = np.abs(X_E2[:, IDX_INT][:, 1]).max()

        # Test E2 with brucon Kd convention (Hypothesis 3): less damping
        # subtraction -> stronger Kd -> smaller overshoot.
        X_E2b = reconstruct_with_env_antiwindup(aug_brucon_kd, t_grid, dtau, df_env, F_max, Ki_diag)
        sway_E2b = X_E2b[:, IDX_ETA][:, 1]
        yaw_E2b = X_E2b[:, IDX_ETA][:, 2]

        # --- HYP 4: Initial-condition reconstruction ---
        # User observation: cqa Test E2 picks up the brucon trace well but
        # accumulates a DC offset over the first ~20 s. Signature of a
        # missing IC. We reconstruct the brucon integrator state at T_WCF
        # from the PI controller law (deviation coords) and seed x0.
        I_ic_cqa = reconstruct_integrator_ic(seed, data, aug.Kp, aug.Kd, aug.Ki)
        I_ic_bru = reconstruct_integrator_ic(seed, data, aug_brucon_kd.Kp, aug_brucon_kd.Kd, aug_brucon_kd.Ki)
        x0_ic = np.zeros(aug.n_state)
        x0_ic[IDX_B_HAT.start:IDX_B_HAT.stop] = 0.0  # b_hat is in deviation coords (pre-WCF mean is zero)
        x0_ic[IDX_INT.start:IDX_INT.stop] = I_ic_cqa
        x0_ic_bru = np.zeros(aug_brucon_kd.n_state)
        x0_ic_bru[IDX_INT.start:IDX_INT.stop] = I_ic_bru

        X_E2_ic = reconstruct_with_env_antiwindup(
            aug, t_grid, dtau, df_env, F_max, Ki_diag, x0=x0_ic,
        )
        sway_E2_ic = X_E2_ic[:, IDX_ETA][:, 1]
        X_E2b_ic = reconstruct_with_env_antiwindup(
            aug_brucon_kd, t_grid, dtau, df_env, F_max, Ki_diag, x0=x0_ic_bru,
        )
        sway_E2b_ic = X_E2b_ic[:, IDX_ETA][:, 1]

        # --- HYP 5: env force NOT demeaned, absolute F_env injected ---
        # See extract_env_force_minus_bhat docstring. The intact-state
        # 25 s pre-window mean was a noisy LF sample; removing it
        # subtracted the very mean wave drift that drives the post-WCF
        # excursion (alpha < 1 -> un-cancelled fraction).
        df_env_abs = extract_env_force_minus_bhat(data, t_grid, b_hat0)
        X_H5 = reconstruct_with_env_antiwindup(
            aug, t_grid, dtau, df_env_abs, F_max, Ki_diag,
        )
        sway_H5 = X_H5[:, IDX_ETA][:, 1]
        # Also with I-IC for full reconstruction.
        X_H5_ic = reconstruct_with_env_antiwindup(
            aug, t_grid, dtau, df_env_abs, F_max, Ki_diag, x0=x0_ic,
        )
        sway_H5_ic = X_H5_ic[:, IDX_ETA][:, 1]

        # Test F: Test E + K_lift Picard
        X_F = reconstruct_with_env_and_lift(aug, t_grid, dtau, df_env, b_hat0, K_lift, n_iter=3)
        sway_F = X_F[:, IDX_ETA][:, 1]
        yaw_F = X_F[:, IDX_ETA][:, 2]

        # Test E_env_only: env force forcing alone (no delta_tau) - diagnostic
        X_Eo = reconstruct_with_env(aug, t_grid, np.zeros_like(dtau), df_env)
        sway_Eo = X_Eo[:, IDX_ETA][:, 1]

        def p95(x):
            return float(np.percentile(np.abs(x), 95))

        # Env-force perturbation magnitudes
        df_y_p95 = p95(df_env[:, 1])
        df_y_peak = np.abs(df_env[:, 1]).max()

        print(f"\n=== {label} (seed {seed}) ===")
        print(f"  b_hat0 [kN, kN, kN.m]: ({b_hat0[0]/1e3:.1f}, {b_hat0[1]/1e3:.1f}, {b_hat0[2]/1e3:.1f})")
        print(f"  dF_env_y peak/P95: {df_y_peak/1e3:.1f} / {df_y_p95/1e3:.1f} kN")
        print(f"  coupling_y = -Fx0 * K = {-b_hat0[0]*K_lift/1e3:.1f} kN/rad")
        print(f"  brucon sway peak/P95:           {np.abs(brucon_sway).max():.2f} / {p95(brucon_sway):.2f} m")
        print(f"  Test A (dtau only)         peak/P95: {np.abs(sway_A).max():.2f} / {p95(sway_A):.2f} m  -> ratio = {p95(sway_A)/p95(brucon_sway):.2f}")
        print(f"  Test E (dtau + env)        peak/P95: {np.abs(sway_E).max():.2f} / {p95(sway_E):.2f} m  -> ratio = {p95(sway_E)/p95(brucon_sway):.2f}  (I_y peak {I_E_peak:.0f} m.s)")
        print(f"  Test E2 (E + anti-windup)  peak/P95: {np.abs(sway_E2).max():.2f} / {p95(sway_E2):.2f} m  -> ratio = {p95(sway_E2)/p95(brucon_sway):.2f}  (I_y peak {I_E2_peak:.1f} m.s, I_max_y = {I_max[1]:.1f} m.s)")
        print(f"  Test E2b (E2 + brucon-Kd)  peak/P95: {np.abs(sway_E2b).max():.2f} / {p95(sway_E2b):.2f} m  -> ratio = {p95(sway_E2b)/p95(brucon_sway):.2f}  [HYP 3]")
        print(f"  I_ic (cqa-Kd) [m.s, m.s, rad.s]: ({I_ic_cqa[0]:.1f}, {I_ic_cqa[1]:.1f}, {I_ic_cqa[2]:.3f})")
        print(f"  I_ic (bru-Kd) [m.s, m.s, rad.s]: ({I_ic_bru[0]:.1f}, {I_ic_bru[1]:.1f}, {I_ic_bru[2]:.3f})  [HYP 4]")
        print(f"  Test E2_ic  (E2 + I-IC)   peak/P95: {np.abs(sway_E2_ic).max():.2f} / {p95(sway_E2_ic):.2f} m  -> ratio = {p95(sway_E2_ic)/p95(brucon_sway):.2f}  [HYP 4]")
        print(f"  Test E2b_ic (E2b+ I-IC)   peak/P95: {np.abs(sway_E2b_ic).max():.2f} / {p95(sway_E2b_ic):.2f} m  -> ratio = {p95(sway_E2b_ic)/p95(brucon_sway):.2f}  [HYP 3+4]")
        _pre = (data[:, COL_T] >= T_WCF - 30) & (data[:, COL_T] <= T_WCF - 5)
        _F_env_pre = np.array([
            (data[_pre, COL_DRIFT_X] + data[_pre, COL_WIND_X] + data[_pre, COL_CUR_X]).mean() * 1e3,
            (data[_pre, COL_DRIFT_Y] + data[_pre, COL_WIND_Y] + data[_pre, COL_CUR_Y]).mean() * 1e3,
            (data[_pre, COL_DRIFT_MZ] + data[_pre, COL_WIND_MZ] + data[_pre, COL_CUR_MZ]).mean() * 1e3,
        ])
        print(f"  F_env_pre  [kN, kN, kN.m]: ({_F_env_pre[0]/1e3:.1f}, {_F_env_pre[1]/1e3:.1f}, {_F_env_pre[2]/1e3:.1f})")
        print(f"  -b_hat_pre [kN, kN, kN.m]: ({-b_hat0[0]/1e3:.1f}, {-b_hat0[1]/1e3:.1f}, {-b_hat0[2]/1e3:.1f})  (expect ~ +F_env_pre by intact Newton balance)")
        print(f"  Test H5   (abs env, no demean)         peak/P95: {np.abs(sway_H5).max():.2f} / {p95(sway_H5):.2f} m  -> ratio = {p95(sway_H5)/p95(brucon_sway):.2f}  [HYP 5]")
        print(f"  Test H5_ic (abs env + I-IC)            peak/P95: {np.abs(sway_H5_ic).max():.2f} / {p95(sway_H5_ic):.2f} m  -> ratio = {p95(sway_H5_ic)/p95(brucon_sway):.2f}  [HYP 5+4]")
        print(f"  Test F (E + K_lift)        peak/P95: {np.abs(sway_F).max():.2f} / {p95(sway_F):.2f} m  -> ratio = {p95(sway_F)/p95(brucon_sway):.2f}")
        print(f"  Test E env-only            peak/P95: {np.abs(sway_Eo).max():.2f} / {p95(sway_Eo):.2f} m  (delta_tau=0)")
        print(f"  Yaw peak: A = {np.degrees(np.abs(yaw_A).max()):.2f} deg, E = {np.degrees(np.abs(yaw_E).max()):.2f} deg, E2 = {np.degrees(np.abs(yaw_E2).max()):.2f} deg, F = {np.degrees(np.abs(yaw_F).max()):.2f} deg")

        # Top row: sway truth vs Test A vs Test E vs Test E2
        ax = axes[0, col]
        ax.plot(t_grid, brucon_sway, color="C0", lw=1.8, label="brucon truth")
        ax.plot(t_grid, sway_A, color="C3", lw=1.0, ls="--", alpha=0.7, label="Test A: dtau only")
        ax.plot(t_grid, sway_E, color="C2", lw=1.0, ls=":", alpha=0.7, label="Test E: dtau + env")
        ax.plot(t_grid, sway_E2, color="C1", lw=1.4, ls="-", label="Test E2: E + anti-windup")
        ax.plot(t_grid, sway_E2_ic, color="C4", lw=1.6, ls="-", label="Test E2_ic: + I-IC [HYP 4]")
        ax.axhline(0, color="black", lw=0.4)
        ax.set_title(f"{label}: linearity reconstruction")
        ax.set_ylabel("sway [m]")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        # Bottom row: forcings (delta_tau_y and env-force_y)
        ax = axes[1, col]
        ax.plot(t_grid, dtau[:, 1] / 1e3, color="C3", lw=0.9, label="delta_tau_y [kN]")
        ax.plot(t_grid, df_env[:, 1] / 1e3, color="C2", lw=0.9, label="dF_env_y [kN]")
        ax.axhline(0, color="black", lw=0.4)
        ax.set_ylabel("forcing [kN]")
        ax.set_xlabel("t - T_WCF [s]")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "bf8_q10_w45: linearity diagnostic (A: dtau; E: dtau+env; E2: E+anti-windup; F: E+K_lift)\n"
        "Anti-windup: |I| <= 0.75 * F_max / Ki per-axis, matching brucon PID::Step clamp",
        fontsize=11,
    )
    fig.tight_layout()
    out = THIS / "linearity_reconstruction_diagnostic.png"
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    run()
