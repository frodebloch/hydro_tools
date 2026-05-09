"""Calibrate sigma_R_tau_lost: per-seed peak |delta_eta_LF| spread due to
tau_lost realisation variability across the pwq30 ensemble.

Rationale (from the design discussion):

  The live cell uses pulse_response(x0=0) to predict the deterministic
  WCFDI deviation trajectory delta_eta_LF(t), based on a fixed average
  tau_lost profile. The total peak-|R| prediction is then

      |eta_hat_obs + delta_eta_LF(t)| + k_sigma * sigma_R_total

  with sigma_R_total = sqrt(sigma_R_LF^2 + sigma_R_WF^2). Both LF and
  WF sigmas come from the BayesianSigmaEstimator, which captures the
  *steady-state* WF and LF stochastic content. Neither captures the
  spread in the deterministic post-WCF trajectory caused by seed-to-seed
  variability in the realised tau_lost(t) profile.

  This script calibrates that missing variance term (sigma_R_tau_lost)
  empirically from the brucon ensemble, so it can be added in quadrature
  to sigma_R_total.

Procedure:

  1. For each pwq30 seed, compute realised tau_lost(t) over [0, 60] s
     post-WCF: tau_lost_seed(t) = tau_pre_seed - tau_thr_brucon(t).
     tau_pre is the seed's own steady-state OrderTau (it varies slightly
     seed-to-seed because the pre-WCF station-keeping equilibrium is
     not exactly identical).

  2. Ensemble-mean tau_lost_avg(t) and per-axis +/-1 sigma band.

  3. Run pulse_response(x0=0) with tau_lost_avg -> delta_eta_LF_avg(t):
     this is what the live cell uses today.

  4. For each seed, pulse_response(x0=0) with that seed's tau_lost ->
     delta_eta_LF_seed(t). Per-seed peak |delta_eta_LF_seed|.

  5. sigma_R_tau_lost = std across seeds of peak |delta_eta_LF_seed|
     (taken in body frame, surge-sway radial).

Outputs:
  - peak_R_tau_lost_sigma_pwq30.png: 2x2 plots
    (mean+/-1sigma tau_lost per axis, per-seed delta_eta_LF traces,
    histogram of per-seed peak, summary bar of the sigma terms).
  - printed sigma_R_tau_lost in metres for adding to the live cell.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))

from cqa.config import csov_default_config
from cqa.controller import LinearDpController
from cqa.transient_obs import (
    build_observer_augmented_system_full,
    csov_observer_gains,
    pulse_response,
    IDX_ETA_HAT, N_STATE,
)
from cqa.vessel import LinearVesselModel

WORK_ROOT = THIS / "work"
TAG = "pwq30"
SEEDS = list(range(1000, 1030))
T_WCF = 560.0
T_HORIZON = 60.0
DT = 0.05

# Pre-WCF window for tau_pre estimation per seed.
T_PRE_LO = T_WCF - 30.0
T_PRE_HI = T_WCF - 5.0


def _load_seed(seed):
    seed_dir = WORK_ROOT / f"{TAG}_seed{seed:04d}"
    if not seed_dir.exists():
        return None
    main_p = next((p for p in seed_dir.glob("*.out") if "estimator" not in p.name), None)
    if main_p is None:
        return None
    with open(main_p) as f:
        hdr = f.readline().strip().split("\t")
    M = {h: data for h, data in zip(hdr, np.loadtxt(main_p, skiprows=1, delimiter="\t").T)}
    if M["t"][-1] < T_WCF + T_HORIZON:
        return None
    return M


def _per_seed_tau_lost(M, t_grid):
    """Return (tau_lost_seed[N,3], tau_pre_seed[3]) for the seed.
    tau_lost(t) = tau_pre - tau_thr_brucon(t), interpolated on t_grid."""
    t = M["t"]
    pre = (t >= T_PRE_LO) & (t <= T_PRE_HI)
    tau_pre = np.array([
        float(M["OrderTauSurge"][pre].mean()),
        float(M["OrderTauSway"][pre].mean()),
        float(M["OrderTauYaw"][pre].mean()),
    ]) * 1e3
    post = (t >= T_WCF) & (t <= T_WCF + T_HORIZON + 1.0)
    t_post = t[post]
    tau_thr_post = np.column_stack([
        M["Tx"][post], M["Ty"][post], M["Tz"][post]
    ]) * 1e3
    # Resample onto uniform forward grid.
    tau_thr_grid = np.column_stack([
        np.interp(t_grid + T_WCF, t_post, tau_thr_post[:, k])
        for k in range(3)
    ])
    tau_lost_grid = tau_pre[None, :] - tau_thr_grid
    return tau_lost_grid, tau_pre


def main():
    cfg = csov_default_config()
    vessel = LinearVesselModel.from_config(cfg.vessel)
    cp = cfg.controller
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D, omega_n=cp.omega_n, zeta=cp.zeta,
    )
    obs_gains = csov_observer_gains(Tp_s=10.0)
    aug = build_observer_augmented_system_full(
        vessel, controller, obs_gains=obs_gains, T_thr=cp.thruster_time_constant_s,
    )

    t_grid = np.arange(0.0, T_HORIZON + 1e-9, DT)
    N = len(t_grid)

    # Collect per-seed tau_lost.
    per_seed_tau_lost = []
    per_seed_tau_pre = []
    seed_ids = []
    for s in SEEDS:
        M = _load_seed(s)
        if M is None:
            continue
        tau_lost_seed, tau_pre = _per_seed_tau_lost(M, t_grid)
        per_seed_tau_lost.append(tau_lost_seed)
        per_seed_tau_pre.append(tau_pre)
        seed_ids.append(s)
    n_seeds = len(per_seed_tau_lost)
    print(f"Loaded {n_seeds} seeds")

    tau_lost_arr = np.stack(per_seed_tau_lost, axis=0)   # [n_seeds, N, 3]
    tau_pre_arr = np.stack(per_seed_tau_pre, axis=0)      # [n_seeds, 3]

    tau_lost_mean = tau_lost_arr.mean(axis=0)             # [N, 3]
    tau_lost_std = tau_lost_arr.std(axis=0)               # [N, 3]

    print(f"\ntau_pre per seed (kN, kN, kNm):")
    print(f"  mean = {tau_pre_arr.mean(0) / 1e3}")
    print(f"  std  = {tau_pre_arr.std(0) / 1e3}")

    print(f"\ntau_lost(t) max per axis (kN, kN, kNm) over [0, {T_HORIZON:.0f}] s:")
    print(f"  ensemble mean: {np.max(np.abs(tau_lost_mean), axis=0) / 1e3}")
    print(f"  per-axis 1-sigma at t=peak-of-mean:")
    for k in range(3):
        kpk = int(np.argmax(np.abs(tau_lost_mean[:, k])))
        print(f"    axis {k}: t_peak={t_grid[kpk]:.1f}s "
              f"mean={tau_lost_mean[kpk, k] / 1e3:+.1f}  "
              f"sigma={tau_lost_std[kpk, k] / 1e3:.1f}  (kN or kNm)")

    # Forward-sim with the ensemble-mean tau_lost.
    X_mean = pulse_response(aug, t_grid, tau_lost_mean, x0=np.zeros(N_STATE))
    deta_LF_mean = X_mean[:, IDX_ETA_HAT][:, 0:2]   # surge/sway only
    R_mean = np.hypot(deta_LF_mean[:, 0], deta_LF_mean[:, 1])
    peak_R_mean = float(R_mean.max())

    # Per-seed forward sim with seed's tau_lost.
    peak_R_seeds = np.zeros(n_seeds)
    deta_LF_seeds = np.zeros((n_seeds, N, 2))
    for i, tl in enumerate(per_seed_tau_lost):
        X = pulse_response(aug, t_grid, tl, x0=np.zeros(N_STATE))
        d = X[:, IDX_ETA_HAT][:, 0:2]
        deta_LF_seeds[i] = d
        peak_R_seeds[i] = np.hypot(d[:, 0], d[:, 1]).max()

    sigma_R_tau_lost = float(peak_R_seeds.std())
    print(f"\n--- Result ---")
    print(f"Mean tau_lost prediction peak |delta_eta_LF| = {peak_R_mean:.3f} m")
    print(f"Per-seed peak |delta_eta_LF|:")
    print(f"  mean = {peak_R_seeds.mean():.3f} m")
    print(f"  std  = {peak_R_seeds.std():.3f} m   <-- sigma_R_tau_lost")
    print(f"  range = [{peak_R_seeds.min():.3f}, {peak_R_seeds.max():.3f}] m")

    # Plot.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) tau_lost mean +/-1 sigma per axis.
    ax = axes[0, 0]
    labels = ["surge [kN]", "sway [kN]", "yaw [kNm]"]
    colors = ["C0", "C1", "C2"]
    for k in range(3):
        scale = 1e3
        ax.plot(t_grid, tau_lost_mean[:, k] / scale, color=colors[k], lw=1.4,
                label=f"mean {labels[k]}")
        ax.fill_between(t_grid,
                        (tau_lost_mean[:, k] - tau_lost_std[:, k]) / scale,
                        (tau_lost_mean[:, k] + tau_lost_std[:, k]) / scale,
                        color=colors[k], alpha=0.2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("t - t_WCF [s]")
    ax.set_ylabel("tau_lost  (kN or kNm)")
    ax.set_title("Realised tau_lost: ensemble mean +/-1 sigma (n=" + str(n_seeds) + ")")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (0,1) per-seed delta_eta_LF radial trajectories with mean overlaid.
    ax = axes[0, 1]
    R_seeds = np.hypot(deta_LF_seeds[:, :, 0], deta_LF_seeds[:, :, 1])
    for i in range(n_seeds):
        ax.plot(t_grid, R_seeds[i], color="grey", lw=0.5, alpha=0.5)
    ax.plot(t_grid, R_mean, color="k", lw=2.0, label="mean tau_lost prediction")
    R_seeds_mean = R_seeds.mean(axis=0)
    R_seeds_std = R_seeds.std(axis=0)
    ax.plot(t_grid, R_seeds_mean, color="C3", lw=1.2, ls="--",
            label="ensemble mean of per-seed |delta_eta_LF|")
    ax.fill_between(t_grid, R_seeds_mean - R_seeds_std, R_seeds_mean + R_seeds_std,
                    color="C3", alpha=0.2, label="+/-1 sigma envelope")
    ax.set_xlabel("t - t_WCF [s]")
    ax.set_ylabel("|delta_eta_LF| [m]")
    ax.set_title("pulse_response(x0=0) with per-seed tau_lost")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,0) histogram of per-seed peak.
    ax = axes[1, 0]
    ax.hist(peak_R_seeds, bins=10, color="C0", alpha=0.7)
    ax.axvline(peak_R_mean, color="k", lw=1.5, label=f"mean-tau_lost peak = {peak_R_mean:.2f} m")
    ax.axvline(peak_R_seeds.mean(), color="C3", lw=1.5, ls="--",
               label=f"per-seed mean = {peak_R_seeds.mean():.2f} m")
    ax.axvspan(peak_R_seeds.mean() - sigma_R_tau_lost,
               peak_R_seeds.mean() + sigma_R_tau_lost,
               color="C3", alpha=0.15, label=f"+/-1 sigma = {sigma_R_tau_lost:.2f} m")
    ax.set_xlabel("peak |delta_eta_LF| [m]")
    ax.set_ylabel("count")
    ax.set_title(f"Per-seed peak distribution (sigma_R_tau_lost = {sigma_R_tau_lost:.3f} m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,1) summary bar: contributions to total sigma_R.
    # (Reference values: pwq30 LF sigma_R at W=60s ~ 0.27 m,
    #  WF sigma_R ~ 0.61 m, total quadrature ~ 0.67 m)
    sigma_R_LF_typ = 0.27
    sigma_R_WF_typ = 0.61
    sigma_R_total_old = float(np.hypot(sigma_R_LF_typ, sigma_R_WF_typ))
    sigma_R_total_new = float(np.sqrt(sigma_R_LF_typ ** 2 + sigma_R_WF_typ ** 2 + sigma_R_tau_lost ** 2))
    ax = axes[1, 1]
    bars = ["sigma_R_LF\n(W=60s)", "sigma_R_WF\n(W=60s)", "sigma_R_tau_lost\n(this study)",
            "total old\n(LF+WF only)", "total new\n(+ tau_lost)"]
    vals = [sigma_R_LF_typ, sigma_R_WF_typ, sigma_R_tau_lost,
            sigma_R_total_old, sigma_R_total_new]
    colors2 = ["C0", "C1", "C3", "grey", "C2"]
    ax.bar(bars, vals, color=colors2)
    for b, v in zip(bars, vals):
        ax.text(b, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("sigma_R [m]")
    ax.set_title("Sigma envelope contributions (typical pwq30 values)")
    ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8)

    plt.suptitle(f"sigma_R_tau_lost calibration at {TAG} (n={n_seeds})", fontsize=12)
    plt.tight_layout()
    out = THIS / "peak_R_tau_lost_sigma_pwq30.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
