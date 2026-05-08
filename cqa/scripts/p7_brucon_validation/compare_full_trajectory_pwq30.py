"""Full trajectory overlay: cqa-27 forecast pipeline vs brucon ensemble at pwq30.

Direct successor to ``compare_pipeline_vs_brucon_pwq30.py`` (which only
compared scalar peaks). This script overlays the full time histories of:

  - tau_lost(t) per DOF (surge, sway, yaw)
  - eta_x(t), eta_y(t)  body-frame deviation
  - sigma_R(t)         radial 1-sigma envelope from Lyapunov

between:
  (A) the brucon ensemble (30 seeds, mean +/- 5..95 percentile) -- the
      observable ground truth of the W2W operability prediction problem.
  (B) the cqa-27 observer-augmented forecast pipeline run with the
      decision-matrix scenario knobs (alpha=2/3, gamma_immediate=0.5,
      T_realloc=10 s), i.e. EXACTLY the model the operator sees through
      ``evaluate_decision_cell(use_obs_transient=True)``.

This is the diagnostic the user requested to expose:
  - shape of scenario-derived tau_lost vs brucon's actual deficit
  - full transient eta(t) match (not just peak)
  - timing offset of cqa peak (~28 s) vs brucon mean peak (~34 s)
  - recovery decay rate
  - surge vs sway component-level bias
  - sigma_R(t) ensemble brucon spread vs cqa-27 Lyapunov envelope

Run::
    PYTHONPATH=. .venv/bin/python scripts/p7_brucon_validation/compare_full_trajectory_pwq30.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

THIS = Path(__file__).resolve().parent
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))

from cqa import csov_default_config, GangwayJointState, ForecastSlot  # noqa: E402
from cqa.vessel import LinearVesselModel, CurrentForceModel  # noqa: E402
from cqa.controller import LinearDpController  # noqa: E402
from cqa.closed_loop import ClosedLoop, state_covariance_freqdomain  # noqa: E402
from cqa.psd import (  # noqa: E402
    npd_wind_gust_force_psd,
    slow_drift_force_psd_newman,
    current_variability_force_psd,
    WindForceModel,
)
from cqa.transient import WcfdiScenario, lift_intact_cov_to_augmented  # noqa: E402
from cqa.transient_obs import (  # noqa: E402
    build_observer_augmented_system_full,
    pulse_response,
    csov_observer_gains,
    N_STATE as N_STATE_OBS,
)


# ----- pwq30 conditions (same as compare_pipeline_vs_brucon_pwq30.py) -----
HS = 4.19571865443425
TP = 10.22443464601827
THETA_ENV_COMPASS = np.deg2rad(210.0)
HEADING_COMPASS = np.deg2rad(180.0)  # theta_rel = +30 deg

WORK_ROOT = THIS / "work"
TAG = "pwq30"
SEEDS = list(range(1000, 1030))
T_WCF = 560.0
BASELINE_START = 5.0
BASELINE_END = 30.0
T_PRE = 30.0
T_POST = 120.0
DT = 0.1


# ---- Brucon ensemble loader (same conventions as check_obs_perseed_taulost.py) ----


def project_ned_to_body(north, east, heading_deg):
    h = np.deg2rad(heading_deg)
    surge = np.cos(h) * north + np.sin(h) * east
    sway = -np.sin(h) * north + np.cos(h) * east
    return surge, sway


def load_main(seed_dir: Path) -> dict[str, np.ndarray]:
    main = next(p for p in seed_dir.glob("*.out") if "estimator" not in p.name)
    with open(main) as f:
        header = f.readline().strip().split("\t")
    data = np.loadtxt(main, skiprows=1, delimiter="\t")
    return {h: data[:, i] for i, h in enumerate(header)}


def load_seed(seed: int):
    seed_dir = WORK_ROOT / f"{TAG}_seed{seed:04d}"
    if not seed_dir.exists():
        return None
    try:
        cols = load_main(seed_dir)
    except (StopIteration, OSError):
        return None
    t = cols["t"]
    if t[-1] < T_WCF + 30.0:
        return None

    # Body-frame deviations vs pre-WCF time-mean baseline (per check_obs_perseed_taulost.py).
    s_b, w_b = project_ned_to_body(cols["x"], cols["y"], cols["heading"])
    base_mask = (t >= T_WCF - BASELINE_END) & (t <= T_WCF - BASELINE_START)
    s_b -= s_b[base_mask].mean()
    w_b -= w_b[base_mask].mean()

    # tau_lost = T - Order, in [N], zero before WCF.
    deficit_surge = (cols["Tx"] - cols["OrderTauSurge"]) * 1e3
    deficit_sway = (cols["Ty"] - cols["OrderTauSway"]) * 1e3
    deficit_yaw = (cols["Tz"] - cols["OrderTauYaw"]) * 1e3
    pre = t < T_WCF
    deficit_surge[pre] = 0.0
    deficit_sway[pre] = 0.0
    deficit_yaw[pre] = 0.0

    t_rel = t - T_WCF
    t_grid = np.arange(-T_PRE, T_POST + DT / 2, DT)
    return dict(
        t_grid=t_grid,
        eta_x=np.interp(t_grid, t_rel, s_b),
        eta_y=np.interp(t_grid, t_rel, w_b),
        tau_lost=np.column_stack([
            np.interp(t_grid, t_rel, deficit_surge),
            np.interp(t_grid, t_rel, deficit_sway),
            np.interp(t_grid, t_rel, deficit_yaw),
        ]),
    )


# ---- cqa-27 forecast trajectory (mirrors _wcfdi_peak_at_forecast_obs) ----


def cqa27_forecast_trajectory(
    cfg, slot: ForecastSlot, theta_rel: float,
    scenario: WcfdiScenario, t_grid: np.ndarray,
):
    """Reproduce the trajectories that ``_wcfdi_peak_at_forecast_obs`` peaks over.

    Returns dict with keys: t (s, t=0 is WCF), tau_lost (Nt, 3) [N], eta_mean (Nt, 3)
    body-frame [m, m, rad], P_eta (Nt, 3, 3) covariance, sigma_R (Nt,).
    """
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
    obs_gains = csov_observer_gains(Tp_s=slot.Tp if slot.Tp > 0 else 10.0)
    aug = build_observer_augmented_system_full(
        vessel, controller, obs_gains=obs_gains,
        T_thr=cp_ctrl.thruster_time_constant_s,
    )

    wind_model = WindForceModel(wp=wp, loa=vp.loa)
    current_model = CurrentForceModel(
        cp=cp,
        lateral_area_underwater=vp.lpp * vp.draft,
        frontal_area_underwater=vp.beam * vp.draft,
        loa=vp.loa,
    )
    F_wind = wind_model.force(slot.Vw, theta_rel)
    F_curr = current_model.force(slot.Vc, theta_rel)
    F_drift = np.array([
        wd.drift_x_amp * slot.Hs ** 2 * np.cos(theta_rel),
        wd.drift_y_amp * slot.Hs ** 2 * np.sin(theta_rel),
        wd.drift_n_amp * slot.Hs ** 2 * np.sin(2.0 * theta_rel),
    ])
    tau_env = F_wind + F_curr + F_drift

    # Scenario tau_lost(t) (only positive t matters).
    gamma_imm = float(scenario.gamma_immediate)
    T_realloc = float(scenario.T_realloc) if scenario.T_realloc > 0 else 1e-9
    pos = t_grid >= 0
    t_pos = t_grid[pos]
    beta_t = 1.0 + (gamma_imm - 1.0) * np.exp(-t_pos / T_realloc)
    tau_lost_pos = (beta_t[:, None] - 1.0) * (-tau_env[None, :])
    tau_lost = np.zeros((len(t_grid), 3))
    tau_lost[pos] = tau_lost_pos

    # Mean trajectory: integrate from x0=0 (deviation from intact SS) over t_pos.
    X_pos = pulse_response(aug, t_pos - t_pos[0], tau_lost_pos, x0=np.zeros(N_STATE_OBS))
    X = np.zeros((len(t_grid), N_STATE_OBS))
    X[pos] = X_pos
    eta_mean = X[:, 0:3]

    # Covariance: Lyapunov ODE on aug.A driven by W_eq matched to intact 6-DOF P6.
    cl_intact = ClosedLoop.build(vessel, controller)
    if slot.Vw > 1e-9:
        S_wind = npd_wind_gust_force_psd(wind_model, slot.Vw, theta_rel)
    else:
        def S_wind(_w):
            return np.zeros((3, 3))
    S_drift = slow_drift_force_psd_newman(
        (wd.drift_x_amp, wd.drift_y_amp, wd.drift_n_amp),
        slot.Hs, slot.Tp, theta_rel,
    )
    if slot.Vc > 1e-9:
        dFdVc = 2.0 * F_curr / slot.Vc
    else:
        dFdVc = np.zeros(3)
    S_curr = current_variability_force_psd(dFdVc, sigma_Vc=0.1, tau=600.0)
    P6 = state_covariance_freqdomain(cl_intact, [S_wind, S_drift, S_curr])
    P0 = lift_intact_cov_to_augmented(P6, n_state=N_STATE_OBS)

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
        t_span=(0.0, float(t_pos[-1] - t_pos[0])),
        y0=P0.flatten(),
        t_eval=t_pos - t_pos[0],
        method="RK45", rtol=1e-5, atol=1e-9,
    )
    P_t_pos = sol_P.y.T.reshape(len(t_pos), n_aug, n_aug)
    P_t_pos = 0.5 * (P_t_pos + P_t_pos.transpose(0, 2, 1))
    P_eta_pos = P_t_pos[:, 0:3, 0:3]
    # Pre-WCF: hold at intact stationary covariance (lift of P6, eta block).
    P_eta_intact = P0[0:3, 0:3]
    P_eta = np.zeros((len(t_grid), 3, 3))
    P_eta[~pos] = P_eta_intact
    P_eta[pos] = P_eta_pos

    sigma_R = np.sqrt(np.maximum(P_eta[:, 0, 0] + P_eta[:, 1, 1], 0.0))

    return dict(
        t=t_grid,
        tau_lost=tau_lost,
        eta_mean=eta_mean,
        P_eta=P_eta,
        sigma_R=sigma_R,
        tau_env=tau_env,
    )


def main():
    cfg = csov_default_config()
    slot = ForecastSlot(
        label="pwq30", Vw=14.0, Hs=HS, Tp=TP, Vc=0.0,
        theta_env_compass=THETA_ENV_COMPASS,
    )
    theta_rel = float(THETA_ENV_COMPASS - HEADING_COMPASS)
    scenario = WcfdiScenario(
        alpha=(2.0 / 3.0,) * 3,
        gamma_immediate=0.5,
        T_realloc=10.0,
    )

    # ---- Brucon ensemble ----
    seed_data = []
    for seed in SEEDS:
        d = load_seed(seed)
        if d is not None:
            seed_data.append((seed, d))
    print(f"Loaded {len(seed_data)}/{len(SEEDS)} seeds")
    if not seed_data:
        sys.exit("No seeds loaded.")
    t_grid = seed_data[0][1]["t_grid"]

    bru_eta_x = np.array([d["eta_x"] for _, d in seed_data])
    bru_eta_y = np.array([d["eta_y"] for _, d in seed_data])
    bru_tau_lost = np.array([d["tau_lost"] for _, d in seed_data])  # (Ns, Nt, 3) [N]

    bru_eta_x_m = bru_eta_x.mean(0)
    bru_eta_y_m = bru_eta_y.mean(0)
    bru_eta_x_p5 = np.percentile(bru_eta_x, 5, 0)
    bru_eta_x_p95 = np.percentile(bru_eta_x, 95, 0)
    bru_eta_y_p5 = np.percentile(bru_eta_y, 5, 0)
    bru_eta_y_p95 = np.percentile(bru_eta_y, 95, 0)

    bru_tl_m = bru_tau_lost.mean(0) * 1e-3  # [kN]
    bru_tl_p5 = np.percentile(bru_tau_lost, 5, 0) * 1e-3
    bru_tl_p95 = np.percentile(bru_tau_lost, 95, 0) * 1e-3

    # Brucon ensemble radial std at each t (cross-seed std of position vector magnitude
    # is NOT the same thing as a Lyapunov sigma_R; the comparable quantity is the
    # std of x and y treated as components, then sigma_R^2 = var(x) + var(y)).
    bru_sigma_R = np.sqrt(bru_eta_x.var(0) + bru_eta_y.var(0))

    # ---- cqa-27 forecast pipeline ----
    cqa = cqa27_forecast_trajectory(cfg, slot, theta_rel, scenario, t_grid)

    sigma_eta_x = np.sqrt(np.maximum(cqa["P_eta"][:, 0, 0], 0.0))
    sigma_eta_y = np.sqrt(np.maximum(cqa["P_eta"][:, 1, 1], 0.0))

    # ---- Headlines ----
    print()
    print(f"theta_rel        = {np.rad2deg(theta_rel):.1f} deg")
    print(f"tau_env (cqa)    = ({cqa['tau_env'][0]*1e-3:+.1f}, "
          f"{cqa['tau_env'][1]*1e-3:+.1f}, {cqa['tau_env'][2]*1e-3:+.1f}) (kN, kN, kNm)")
    print(f"scenario         = gamma_imm={scenario.gamma_immediate}, "
          f"T_realloc={scenario.T_realloc} s, alpha={scenario.alpha}")
    print()
    print("--- tau_lost peaks (kN, post-WCF) ---")
    post = t_grid > 0
    for k, lab in enumerate(("surge", "sway", "yaw")):
        scale = 1e-3 if k < 2 else 1e-3
        unit = "kN" if k < 2 else "kNm"
        bm = bru_tl_m[post, k]
        cm = cqa["tau_lost"][post, k] * 1e-3
        i_b = np.argmax(np.abs(bm))
        i_c = np.argmax(np.abs(cm))
        print(f"  {lab:5s}  brucon mean: {bm[i_b]:+.1f} {unit} at t={t_grid[post][i_b]:+.1f} s   "
              f"cqa-27: {cm[i_c]:+.1f} {unit} at t={t_grid[post][i_c]:+.1f} s")
    print()
    print("--- eta peaks (m) ---")
    for k, (lab, bru, cm) in enumerate([
        ("eta_x", bru_eta_x_m, cqa["eta_mean"][:, 0]),
        ("eta_y", bru_eta_y_m, cqa["eta_mean"][:, 1]),
    ]):
        i_b = np.argmax(np.abs(bru[post]))
        i_c = np.argmax(np.abs(cm[post]))
        print(f"  {lab}  brucon mean: {bru[post][i_b]:+.3f} m at t={t_grid[post][i_b]:+.1f} s   "
              f"cqa-27 mean: {cm[post][i_c]:+.3f} m at t={t_grid[post][i_c]:+.1f} s")
    print()
    print("--- sigma_R (m) at end of horizon (t=+120 s) ---")
    print(f"  brucon ensemble std: {bru_sigma_R[-1]:.3f} m")
    print(f"  cqa-27 Lyapunov     : {cqa['sigma_R'][-1]:.3f} m")
    print(f"  brucon intact (t<0) : {bru_sigma_R[t_grid < 0].mean():.3f} m")
    print(f"  cqa-27 intact (t<0) : {cqa['sigma_R'][t_grid < 0].mean():.3f} m")

    # ---- Plot ----
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    dof_labels = ("surge [kN]", "sway [kN]", "yaw [kNm]")
    for k in range(3):
        ax = axes[0, k]
        ax.fill_between(t_grid, bru_tl_p5[:, k], bru_tl_p95[:, k],
                        alpha=0.18, color="C0", label="brucon 5..95%")
        ax.plot(t_grid, bru_tl_m[:, k], color="C0", lw=2, label=f"brucon mean ({len(seed_data)})")
        ax.plot(t_grid, cqa["tau_lost"][:, k] * 1e-3, color="C3", lw=2, ls="--",
                label="cqa-27 scenario")
        ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.3)
        ax.set_title(f"tau_lost {('surge', 'sway', 'yaw')[k]}")
        ax.set_ylabel(dof_labels[k])
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, loc="best")

    # eta_x, eta_y, then a placeholder; merge bottom row for sigma_R + radial mean.
    for k, (ax, bru_m, bru_p5, bru_p95, cqa_m, sig, lab) in enumerate([
        (axes[1, 0], bru_eta_x_m, bru_eta_x_p5, bru_eta_x_p95,
         cqa["eta_mean"][:, 0], sigma_eta_x, "eta_x [m] (surge)"),
        (axes[1, 1], bru_eta_y_m, bru_eta_y_p5, bru_eta_y_p95,
         cqa["eta_mean"][:, 1], sigma_eta_y, "eta_y [m] (sway)"),
    ]):
        ax.fill_between(t_grid, bru_p5, bru_p95, alpha=0.18, color="C0", label="brucon 5..95%")
        ax.plot(t_grid, bru_m, color="C0", lw=2, label="brucon mean")
        ax.plot(t_grid, cqa_m, color="C3", lw=2, ls="--", label="cqa-27 mean")
        # k_sigma=0.674 -> p75 envelope (the decision-matrix default).
        ax.plot(t_grid, cqa_m + 0.674 * sig, color="C3", lw=1, ls=":",
                label="cqa-27 mean +/- 0.674 sigma")
        ax.plot(t_grid, cqa_m - 0.674 * sig, color="C3", lw=1, ls=":")
        ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.3)
        ax.set_title(lab)
        ax.set_ylabel("[m]")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, loc="best")
    # Radial mean R(t) in axes[1, 2].
    ax = axes[1, 2]
    bru_R = np.sqrt(bru_eta_x ** 2 + bru_eta_y ** 2)
    bru_R_m = bru_R.mean(0)
    bru_R_p5 = np.percentile(bru_R, 5, 0)
    bru_R_p95 = np.percentile(bru_R, 95, 0)
    cqa_R_m = np.sqrt(cqa["eta_mean"][:, 0] ** 2 + cqa["eta_mean"][:, 1] ** 2)
    ax.fill_between(t_grid, bru_R_p5, bru_R_p95, alpha=0.18, color="C0")
    ax.plot(t_grid, bru_R_m, color="C0", lw=2, label="brucon ensemble |R|")
    ax.plot(t_grid, cqa_R_m, color="C3", lw=2, ls="--", label="cqa-27 mean |R|")
    ax.plot(t_grid, cqa_R_m + 0.674 * cqa["sigma_R"], color="C3", lw=1, ls=":",
            label="cqa-27 mean + 0.674 sigma_R (decision envelope)")
    ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.3)
    ax.set_title("|R(t)| body-frame radial deviation")
    ax.set_ylabel("[m]")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # Bottom row: sigma_R(t), per-component sigmas, and an annotation panel.
    ax = axes[2, 0]
    ax.plot(t_grid, bru_sigma_R, color="C0", lw=2, label="brucon ensemble std (= sqrt(var_x + var_y))")
    ax.plot(t_grid, cqa["sigma_R"], color="C3", lw=2, ls="--", label="cqa-27 Lyapunov sigma_R")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title("sigma_R(t)")
    ax.set_xlabel("t - t_WCF [s]")
    ax.set_ylabel("[m]")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    bru_sx = bru_eta_x.std(0)
    bru_sy = bru_eta_y.std(0)
    ax.plot(t_grid, bru_sx, color="C0", lw=2, label="brucon std(eta_x)")
    ax.plot(t_grid, bru_sy, color="C2", lw=2, label="brucon std(eta_y)")
    ax.plot(t_grid, sigma_eta_x, color="C0", lw=1.5, ls="--", label="cqa sigma_x")
    ax.plot(t_grid, sigma_eta_y, color="C2", lw=1.5, ls="--", label="cqa sigma_y")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title("Per-component sigmas")
    ax.set_xlabel("t - t_WCF [s]")
    ax.set_ylabel("[m]")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[2, 2]
    ax.axis("off")
    ax.text(
        0.0, 1.0,
        "pwq30 conditions\n"
        f"  Hs = {HS:.2f} m,  Tp = {TP:.2f} s\n"
        f"  Vw = {slot.Vw:.1f} m/s, Vc = {slot.Vc:.1f} m/s\n"
        f"  theta_rel = +30 deg (bow-quarter port)\n"
        f"  vessel: CSOV (Lpp = {cfg.vessel.lpp} m)\n\n"
        f"Scenario (decision-matrix default):\n"
        f"  alpha = {scenario.alpha[0]:.3f}\n"
        f"  gamma_imm = {scenario.gamma_immediate}\n"
        f"  T_realloc = {scenario.T_realloc} s\n\n"
        f"tau_env (cqa, kN/kNm):\n"
        f"  surge = {cqa['tau_env'][0]*1e-3:+.1f}\n"
        f"  sway  = {cqa['tau_env'][1]*1e-3:+.1f}\n"
        f"  yaw   = {cqa['tau_env'][2]*1e-3:+.1f}\n\n"
        f"Ensemble: {len(seed_data)} brucon seeds\n"
        f"  cqa peak |R|: {cqa_R_m.max():.3f} m at t={t_grid[np.argmax(cqa_R_m)]:+.1f} s\n"
        f"  brucon peak |R|: {bru_R_m.max():.3f} m at t={t_grid[np.argmax(bru_R_m)]:+.1f} s",
        ha="left", va="top", fontsize=9, family="monospace",
        transform=ax.transAxes,
    )

    fig.suptitle(
        f"Full trajectory overlay at pwq30 -- cqa-27 forecast pipeline vs brucon ensemble "
        f"({len(seed_data)} seeds)",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = THIS / "compare_full_trajectory_pwq30.png"
    plt.savefig(out_png, dpi=120)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
