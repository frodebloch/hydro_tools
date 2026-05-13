"""LF/WF decomposition diagnostic on representative pwq30 seeds.

Question: where does the ~50% unexplained variance in regime-B peak |R|
come from?

Two-timescale structure of the brucon sim:
    eta_total(t) = eta_LF(t) + eta_W(t)
where eta_W is an additive WF process driven by the wave-elevation
realisation, and eta_LF responds to the LF channel (tau_thr - tau_env).

Diagnostic strategy:
  1. Pick 3 representative regime-B seeds: low / median / high peak_R.
  2. For each, snapshot the FULL brucon state at t = t_WCF, populated
     into a cqa-27 IC (eta, nu, eta_hat, nu_hat, b_hat, tau_thr,
     int, eta_w, xi).
  3. Use brucon's REALISED tau_lost(t) = OrderTau_pre - tau_thr_brucon(t)
     as the forcing (NOT the parametric WcfdiScenario form, which we
     already showed is inadequate).
  4. Forward-integrate cqa-27 from t_WCF to t_WCF+60s.
  5. Compare per-channel:
       - LF channels: cqa-27 forward eta_hat_LF vs brucon truth's
         (x,y,h) - (xHf,yHf,headingHf)  (= reconstructed LF).
       - WF channels: cqa-27 forward eta_w (rings down from IC, no
         exogenous wave forcing) vs brucon truth's (xHf, yHf, headingHf).
       - Combined |R(t)|: cqa-27 vs truth.

Interpretation:
  - If cqa-27 LF tracks truth LF well, the linear model is adequate and
    the residual peak |R| variance is wave-realisation-driven (the WF
    channel gets re-excited by post-WCF waves which the homogeneous
    forward sim cannot see).
  - If cqa-27 LF DIVERGES from truth LF, the linear model is missing
    physics (allocator nonlinearity, RPM spool-down, etc.) and the
    residual is unmodelled-LF.

Output: peak_R_regime_split_traces_pwq30.png with one row per seed,
columns = (LF surge/sway, WF radial, R total).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))

from cqa.config import CqaConfig, csov_default_config
from cqa.transient_obs import (
    build_observer_augmented_system_full,
    csov_observer_gains,
    pulse_response,
    IDX_ETA, IDX_NU, IDX_ETA_HAT, IDX_NU_HAT, IDX_B_HAT,
    IDX_TAU_THR, IDX_INT, IDX_XI, IDX_ETA_W, N_STATE,
)
from cqa.controller import LinearDpController
from cqa.vessel import LinearVesselModel

WORK_ROOT = THIS / "work"
TAG = "pwq30"
SEEDS = list(range(1000, 1030))
T_WCF = 560.0
T_HORIZON = 60.0
DT_FWD = 0.05   # forward-sim grid step


def _load_seed(seed):
    seed_dir = WORK_ROOT / f"{TAG}_seed{seed:04d}"
    if not seed_dir.exists():
        return None
    main_p = next((p for p in seed_dir.glob("*.out") if "estimator" not in p.name), None)
    est_p = seed_dir / f"{TAG}_seed{seed:04d}_estimator.out"
    if main_p is None or not est_p.exists():
        return None
    with open(main_p) as f:
        hdr_m = f.readline().strip().split("\t")
    M = {h: data for h, data in zip(hdr_m, np.loadtxt(main_p, skiprows=1, delimiter="\t").T)}
    with open(est_p) as f:
        hdr_e = f.readline().strip().split("\t")
    E = {h: data for h, data in zip(hdr_e, np.loadtxt(est_p, skiprows=1, delimiter="\t").T)}
    return M, E


def _get_pre_baselines(M, E):
    """Pre-WCF body-frame mean of (x, y) and steady-state OrderTau."""
    t = M["t"]
    pre = (t >= T_WCF - 30.0) & (t <= T_WCF - 5.0)
    h_pre = np.deg2rad(M["heading"][pre])
    s_b = np.cos(h_pre) * M["x"][pre] + np.sin(h_pre) * M["y"][pre]
    w_b = -np.sin(h_pre) * M["x"][pre] + np.cos(h_pre) * M["y"][pre]
    # Circular mean of pre-WCF heading (avoid +/-180 wrap bias).
    psi_pre = float(np.angle(np.mean(np.exp(1j * h_pre))))
    s_pre = float(s_b.mean())
    w_pre = float(w_b.mean())
    # SS pre-WCF OrderTau (commanded thrust at SS)
    tau_pre = np.array([
        float(M["OrderTauSurge"][pre].mean()),
        float(M["OrderTauSway"][pre].mean()),
        float(M["OrderTauYaw"][pre].mean()),
    ]) * 1e3
    return psi_pre, s_pre, w_pre, tau_pre


def _build_seed_traces(M, E):
    """Return time-series in body frame of: t, eta_LF_truth, eta_W_truth,
    R_truth, tau_lost_truth, all on the brucon main grid for t in
    [T_WCF, T_WCF+T_HORIZON]."""
    t = M["t"]
    psi_pre, s_pre, w_pre, tau_pre = _get_pre_baselines(M, E)
    mask = (t >= T_WCF) & (t <= T_WCF + T_HORIZON)
    t_post = t[mask]
    h = np.deg2rad(M["heading"][mask])

    # Body-frame (x,y,h) total motion, deviation from pre-WCF mean.
    s_b = np.cos(h) * M["x"][mask] + np.sin(h) * M["y"][mask] - s_pre
    w_b = -np.sin(h) * M["x"][mask] + np.cos(h) * M["y"][mask] - w_pre
    # Heading deviation: properly unwrap to avoid +/-180 boundary jumps.
    psi_b = np.angle(np.exp(1j * (h - psi_pre)))
    # WF channels are already in body frame in brucon (xHf, yHf, headingHf).
    s_W = M["xHf"][mask]
    w_W = M["yHf"][mask]
    psi_W = M["headingHf"][mask]
    # LF = total - WF
    s_LF = s_b - s_W
    w_LF = w_b - w_W
    psi_LF = np.angle(np.exp(1j * (psi_b - psi_W)))

    R_truth = np.hypot(s_b, w_b)
    R_LF_truth = np.hypot(s_LF, w_LF)
    R_W_truth = np.hypot(s_W, w_W)

    # Realised tau_lost(t) = T_post - T_pre = tau_thr - tau_pre
    # (authoritative convention; sec.12.21.17 sign fix).
    tau_thr_brucon = np.column_stack([
        M["Tx"][mask], M["Ty"][mask], M["Tz"][mask],
    ]) * 1e3
    tau_lost_truth = tau_thr_brucon - tau_pre[None, :]

    return dict(
        t_post=t_post, t0=T_WCF,
        s_b=s_b, w_b=w_b, psi_b=psi_b,
        s_LF=s_LF, w_LF=w_LF, psi_LF=psi_LF,
        s_W=s_W, w_W=w_W, psi_W=psi_W,
        R_truth=R_truth, R_LF_truth=R_LF_truth, R_W_truth=R_W_truth,
        tau_lost_truth=tau_lost_truth,
        tau_pre=tau_pre,
        psi_pre=psi_pre, s_pre=s_pre, w_pre=w_pre,
    )


def _build_ic(M, E, traces):
    """Populate cqa-27 initial state from brucon snapshot at t_WCF."""
    t = M["t"]
    tE = E["Time"]
    t0 = T_WCF

    def at(ta, va, tt):
        return float(np.interp(tt, ta, va))

    psi_pre = traces["psi_pre"]
    s_pre = traces["s_pre"]
    w_pre = traces["w_pre"]

    # eta (truth, body, deviation): the cqa-27 augmented system uses
    # eta = eta_hat + eta_w + innovation_residual. To make the IC a
    # self-consistent SS of the model, set eta := eta_hat + eta_w so
    # the innovation e = 0 at t=0. This avoids polluting all observer
    # state derivatives at t=0 by a residual that brucon happens to
    # have but the linear model treats as a forcing term.
    eta_hat_t0 = np.array([
        at(t, M["SurgeDev"], t0),
        at(t, M["SwayDev"], t0),
        at(t, M["HeadingDev"], t0),
    ])
    eta_w_t0 = np.array([
        at(t, M["xHf"], t0),
        at(t, M["yHf"], t0),
        at(t, M["headingHf"], t0),
    ])
    eta = eta_hat_t0 + eta_w_t0

    # nu (truth body velocity): SurgeSpeed, SwaySpeed, RateOfTurn (deg/min -> rad/s)
    nu = np.array([
        at(t, M["SurgeSpeed"], t0),
        at(t, M["SwaySpeed"], t0),
        np.deg2rad(at(t, M["RateOfTurn"], t0)) / 60.0,
    ])

    # eta_hat (LF estimate, body): SurgeDev, SwayDev, HeadingDev (rad).
    # NOTE these are the controller's tracked-deviation, expected to match LF channel.
    eta_hat = eta_hat_t0
    # nu_hat: not logged. Approximate as nu (observer tracks at SS).
    nu_hat = nu.copy()

    # b_hat (Nm-scaled): EstBias{Surge,Sway,Yaw} * 1e3
    b_hat = np.array([
        at(tE, E["EstBiasSurge"], t0),
        at(tE, E["EstBiasSway"], t0),
        at(tE, E["EstBiasYaw"], t0),
    ]) * 1e3

    # tau_thr (SS): use OrderTau at t_WCF (just before failure).
    tau_thr = np.array([
        at(t, M["OrderTauSurge"], t0),
        at(t, M["OrderTauSway"], t0),
        at(t, M["OrderTauYaw"], t0),
    ]) * 1e3

    # int (PI integrator state): SS approximation from tau_env / Ki_diag.
    # Defer; build with config Ki below.

    # eta_w (WF state, body): xHf, yHf, headingHf at t_WCF.
    eta_w = eta_w_t0

    # xi (WF integrator state): from wave-filter eqn at SS,
    #   eta_w_dot = -omega_p^2 xi - 2 zeta omega_p eta_w + k2*e
    # Innovation e is small. For narrow-band eta_w(t) ~ A*sin(omega_p*t),
    # eta_w_dot ~ A*omega_p*cos = (omega_p)*eta_w_orthogonal. We don't have
    # the quadrature signal logged, so set xi = 0 and accept a small initial
    # transient (decays in O(2*pi/omega_p) ~ Tp seconds).
    xi = np.zeros(3)

    return eta, nu, eta_hat, nu_hat, b_hat, tau_thr, eta_w, xi


def _forward_sim(cfg: CqaConfig, ic, traces, Tp_obs_s=10.0):
    """Run cqa-27 forward sim with brucon-realised tau_lost on a uniform
    grid in [t_WCF, t_WCF + T_HORIZON]."""
    eta, nu, eta_hat, nu_hat, b_hat, tau_thr, eta_w, xi = ic

    # Build augmented system.
    vp = cfg.vessel
    cp_ctrl = cfg.controller
    vessel = LinearVesselModel.from_config(vp)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D,
        omega_n=cp_ctrl.omega_n, zeta=cp_ctrl.zeta,
    )
    obs_gains = csov_observer_gains(Tp_s=Tp_obs_s)
    aug = build_observer_augmented_system_full(
        vessel, controller, obs_gains=obs_gains,
        T_thr=cp_ctrl.thruster_time_constant_s,
    )

    # int state: SS such that Ki * int = -tau_env (so that tau_cmd has
    # FF cancellation of tau_env). tau_env = +b_hat (live cell convention).
    # tau_cmd = -Kp eta_hat - Kd nu_hat - b_hat - Ki int.
    # At SS pre-WCF, tau_cmd = -tau_env -> Ki int = +tau_env - Kp*eta_hat - Kd*nu_hat - b_hat
    # but eta_hat ~ small offset, nu_hat ~ 0; with b_hat = -tau_env this gives
    # Ki int = +tau_env - (-tau_env) = 0?  No, work it through carefully:
    # tau_cmd_SS = -tau_env (drives tau_thr to -tau_env, which cancels env on vessel).
    # So -Kp*eta_hat - Kd*nu_hat - b_hat - Ki*int = -tau_env
    # With b_hat = -tau_env: Ki*int = tau_env - Kp*eta_hat - Kd*nu_hat - b_hat
    #                              = tau_env + tau_env - Kp*eta_hat - Kd*nu_hat
    # Use exact form:
    Kp = controller.Kp
    Kd = controller.Kd
    Ki = aug.Ki
    Ki_diag = np.array([Ki[i, i] for i in range(3)])
    # Robustly: since we have b_hat and tau_thr from brucon, set int such that
    # the controller equation is consistent at t_WCF:
    #   tau_cmd_SS = tau_thr (in SS, they match) = -Kp eta_hat - Kd nu_hat - b_hat - Ki int
    # so Ki int = -Kp eta_hat - Kd nu_hat - b_hat - tau_thr
    rhs = -(Kp @ eta_hat) - (Kd @ nu_hat) - b_hat - tau_thr
    int_state = rhs / Ki_diag

    # Diagnostic: verify the IC is a SS of the cqa-27 dynamics by computing
    # the time-derivative x_dot = A @ x0 with all 27 states populated, and
    # printing the magnitude per channel. A consistent IC should have
    # x_dot near zero in all channels.
    # We do this AFTER packing x0 below.

    # Pack x0.
    x0 = np.zeros(N_STATE)
    x0[IDX_ETA] = eta
    x0[IDX_NU] = nu
    x0[IDX_ETA_HAT] = eta_hat
    x0[IDX_NU_HAT] = nu_hat
    x0[IDX_B_HAT] = b_hat
    x0[IDX_TAU_THR] = tau_thr
    x0[IDX_INT] = int_state
    x0[IDX_XI] = xi
    x0[IDX_ETA_W] = eta_w

    # Forward grid.
    t_fwd = np.arange(0.0, T_HORIZON + 1e-9, DT_FWD)
    # Resample brucon-realised tau_lost onto this grid.
    t_brucon_local = traces["t_post"] - T_WCF
    tau_lost_brucon = traces["tau_lost_truth"]
    tau_lost_grid = np.column_stack([
        np.interp(t_fwd, t_brucon_local, tau_lost_brucon[:, k])
        for k in range(3)
    ])

    X = pulse_response(aug, t_fwd, tau_lost_grid, x0=x0)
    # Experiment A: same IC, ZERO exogenous tau_lost. If the cqa-27 system
    # is given the brucon IC and no disturbance, does it sit near the IC
    # for 60 s (kinematic + dynamic SS holds) or does it drift (observer
    # innovation imbalance / kinematic inconsistency)?
    X_zero = pulse_response(aug, t_fwd, np.zeros_like(tau_lost_grid), x0=x0)

    # IC-consistency diagnostic: compute x_dot = A @ x0 and report the
    # per-channel magnitude. A consistent SS IC should have x_dot ~ 0
    # in every channel.
    x_dot = aug.A @ x0
    # Innovation: e = eta - eta_hat - eta_w. Should be ~zero at SS.
    e = x0[IDX_ETA] - x0[IDX_ETA_HAT] - x0[IDX_ETA_W]
    print(f"    IC consistency check: |x_dot|_max per channel block:")
    print(f"      innovation e   = {e}  [m, m, rad]    |e| = {np.linalg.norm(e):.3e}")
    print(f"      eta            = {x0[IDX_ETA]}")
    print(f"      eta_hat        = {x0[IDX_ETA_HAT]}")
    print(f"      eta_w          = {x0[IDX_ETA_W]}")
    print(f"      d/dt eta      : {np.max(np.abs(x_dot[IDX_ETA])):.3e}  [m/s, m/s, rad/s]")
    print(f"      d/dt nu       : {np.max(np.abs(x_dot[IDX_NU])):.3e}  [m/s2, m/s2, rad/s2]")
    print(f"      d/dt eta_hat  : {np.max(np.abs(x_dot[IDX_ETA_HAT])):.3e}")
    print(f"      d/dt nu_hat   : {np.max(np.abs(x_dot[IDX_NU_HAT])):.3e}")
    print(f"      d/dt b_hat    : {np.max(np.abs(x_dot[IDX_B_HAT])):.3e}")
    print(f"      d/dt tau_thr  : {np.max(np.abs(x_dot[IDX_TAU_THR])):.3e}  [N/s, N/s, Nm/s]")
    print(f"      d/dt int      : {np.max(np.abs(x_dot[IDX_INT])):.3e}")
    print(f"      d/dt xi       : {np.max(np.abs(x_dot[IDX_XI])):.3e}")
    print(f"      d/dt eta_w    : {np.max(np.abs(x_dot[IDX_ETA_W])):.3e}")

    return t_fwd, X, X_zero


def _peak_R_seeds(rows):
    """From a previous regression, pull the peak_R values to pick reps."""
    pks = []
    for s in SEEDS:
        out = _load_seed(s)
        if out is None:
            continue
        traces = _build_seed_traces(*out)
        pks.append((s, float(traces["R_truth"].max()),
                    float(traces["t_post"][int(np.argmax(traces["R_truth"]))] - T_WCF)))
    return pks


def main():
    pks = _peak_R_seeds([])
    pks_sorted = sorted(pks, key=lambda r: r[1])
    # Pick low / median / high among regime-B seeds (t_peak_R >= 15s).
    regime_B = [p for p in pks_sorted if p[2] >= 15.0]
    if len(regime_B) < 3:
        sys.exit("not enough regime-B seeds")
    low = regime_B[0]
    high = regime_B[-1]
    median = regime_B[len(regime_B) // 2]
    chosen = [low, median, high]
    print("Selected representative seeds:")
    for tag, (s, pk, tpk) in zip(["LOW", "MED", "HIGH"], chosen):
        print(f"  {tag}: seed {s}  peak_R = {pk:.3f} m  t_peak = {tpk:.1f} s")

    cfg = csov_default_config()

    fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharey="col")
    col_titles = ["LF surge (m)", "LF sway (m)", "WF radial (m)", "Total |R| (m)"]

    for row_i, (tag, (seed, pk, tpk)) in enumerate(zip(["LOW", "MED", "HIGH"], chosen)):
        out = _load_seed(seed)
        traces = _build_seed_traces(*out)
        ic = _build_ic(*out, traces)
        # Use mean Tp from the seed's estimator output.
        tE = out[1]["Time"]
        Tp_est = out[1].get("EstWavePeriodSurge")
        Tp_obs = (float(np.interp(T_WCF, tE, Tp_est))
                  if Tp_est is not None else 10.0)
        if not (3.0 < Tp_obs < 30.0):
            Tp_obs = 10.0
        t_fwd, X, X_zero = _forward_sim(cfg, ic, traces, Tp_obs_s=Tp_obs)

        # Forward predictions in body frame.
        s_LF_pred = X[:, IDX_ETA_HAT][:, 0]
        w_LF_pred = X[:, IDX_ETA_HAT][:, 1]
        s_W_pred = X[:, IDX_ETA_W][:, 0]
        w_W_pred = X[:, IDX_ETA_W][:, 1]
        # Total "predicted" |R|: use TRUTH eta state (IDX_ETA) which the
        # cqa-27 model evolves coherently.
        s_total_pred = X[:, IDX_ETA][:, 0]
        w_total_pred = X[:, IDX_ETA][:, 1]
        R_pred = np.hypot(s_total_pred, w_total_pred)
        R_W_pred = np.hypot(s_W_pred, w_W_pred)

        # Experiment A overlay: zero exogenous tau_lost.
        s_LF_zero = X_zero[:, IDX_ETA_HAT][:, 0]
        w_LF_zero = X_zero[:, IDX_ETA_HAT][:, 1]
        R_zero = np.hypot(X_zero[:, IDX_ETA][:, 0], X_zero[:, IDX_ETA][:, 1])

        t_truth = traces["t_post"] - T_WCF

        # --- Col 0: LF surge ---
        ax = axes[row_i, 0]
        ax.plot(t_truth, traces["s_LF"], color="k", lw=1.4, label="brucon LF (truth)")
        ax.plot(t_fwd, s_LF_pred, color="C0", lw=1.0, label="cqa-27 eta_hat (with tau_lost)")
        ax.plot(t_fwd, s_LF_zero, color="C0", lw=1.0, ls="--", label="cqa-27 eta_hat (tau_lost=0)")
        ax.set_ylabel(f"{tag}\nseed {seed}\npeak_R={pk:.2f} m")
        if row_i == 0:
            ax.set_title(col_titles[0])
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # --- Col 1: LF sway ---
        ax = axes[row_i, 1]
        ax.plot(t_truth, traces["w_LF"], color="k", lw=1.4)
        ax.plot(t_fwd, w_LF_pred, color="C0", lw=1.0)
        ax.plot(t_fwd, w_LF_zero, color="C0", lw=1.0, ls="--")
        if row_i == 0:
            ax.set_title(col_titles[1])
        ax.grid(alpha=0.3)

        # --- Col 2: WF radial ---
        ax = axes[row_i, 2]
        ax.plot(t_truth, traces["R_W_truth"], color="k", lw=1.4, label="brucon WF (truth)")
        ax.plot(t_fwd, R_W_pred, color="C3", lw=1.0, label="cqa-27 eta_w (decays)")
        if row_i == 0:
            ax.set_title(col_titles[2])
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # --- Col 3: total |R| ---
        ax = axes[row_i, 3]
        ax.plot(t_truth, traces["R_truth"], color="k", lw=1.4, label="brucon truth")
        ax.plot(t_truth, traces["R_LF_truth"], color="grey", lw=1.0, ls=":",
                label="brucon LF only")
        ax.plot(t_fwd, R_pred, color="C2", lw=1.2, label="cqa-27 (with tau_lost)")
        ax.plot(t_fwd, R_zero, color="C2", lw=1.0, ls="--", label="cqa-27 (tau_lost=0)")
        ax.axhline(pk, color="k", lw=0.5, ls="--")
        if row_i == 0:
            ax.set_title(col_titles[3])
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Print scalar summary
        peak_pred = float(R_pred.max())
        peak_zero = float(R_zero.max())
        peak_LF_truth = float(traces["R_LF_truth"].max())
        print(f"  {tag} seed {seed}: peak_R_truth={pk:.3f}  "
              f"peak_R_LF_truth={peak_LF_truth:.3f}  "
              f"peak_R_cqa27={peak_pred:.3f}  "
              f"peak_R_cqa27_ZERO={peak_zero:.3f}  Tp_obs={Tp_obs:.2f}")

    for ax in axes[-1, :]:
        ax.set_xlabel("t - t_WCF [s]")

    plt.suptitle(f"LF/WF decomposition: cqa-27 vs brucon truth ({TAG}, "
                 f"3 representative seeds, brucon-realised tau_lost)",
                 fontsize=11)
    plt.tight_layout()
    out_p = THIS / "peak_R_regime_split_traces_pwq30.png"
    plt.savefig(out_p, dpi=120)
    print(f"\nsaved {out_p}")


if __name__ == "__main__":
    main()
