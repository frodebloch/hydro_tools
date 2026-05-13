"""Calibrate sigma_R_b_hat: peak-|R| spread due to b_hat measurement
uncertainty propagated through the cqa-27 augmented system.

Rationale (from the design discussion, see where_we_are_now_pwq30.py):

  The live cell uses pulse_response(x0=0) to predict the deterministic
  WCFDI deviation trajectory delta_eta_LF(t), based on tau_lost(t) :=
  T_post(t) - T_pre, with T_pre := -b_hat (the live observer's bias
  estimate at the snapshot moment). The "tau_lost" name here matches
  the authoritative convention in cqa.decision_matrix._wcfdi_peak_at_
  forecast_obs (lines 519-527): it is the hull-felt thrust DEVIATION
  from intact-equilibrium thrust. Injected through B_lost = +Minv,
  positive tau_lost drives positive acceleration; negative tau_lost
  (i.e. T_post < T_pre, "thrust dropped") drives negative acceleration.

  Sign-convention note (sec.12.21.17): an earlier version of this
  script used tau_lost := T_pre - T_post (the opposite sign), which
  resulted in sign-flipped delta_eta_mean in every calibration npz
  and infected the brucon-validation harness. The fix is applied at
  _per_seed_tau_lost and at the MC step that perturbs b_hat.

  An earlier version used the controller's commanded thrust OrderTau,
  averaged over a 25-s pre-WCF window, as the source of tau_pre. That
  was a noisy estimator (std of 12-25 kN across seeds) because the
  short-window mean still contains wave-frequency residual. Using
  brucon's NPO bias estimate b_hat directly is much cleaner: b_hat
  has built-in T_b ~ 1000 s integration, so its snapshot value is
  well-converged to the true LF environmental force. Per-seed
  std(b_hat at t=T_WCF-5) drops to ~3-4 kN in surge/sway, ~84 kNm
  in yaw -- 5x to 8x more stable than OrderTau-based tau_pre.

  Propagating the residual b_hat measurement noise through cqa-27
  (Monte Carlo) gives the corresponding peak-|delta_eta_LF| spread
  contribution: sigma_R_b_hat. This is what should be added in
  quadrature with sigma_R_LF and sigma_R_WF in the live envelope.

Procedure:

  1. Per seed, read b_hat snapshot at t=T_WCF-5 from the brucon NPO
     log (EstBiasSurge/Sway/Yaw in kN/kNm). Compute mean and std
     across seeds.

  2. Run pulse_response(x0=0) with each seed's tau_lost(t) :=
     tau_thr_brucon_seed(t) - tau_pre  -> per-seed
     delta_eta_LF_seed(t). Per-seed peak |delta_eta_LF_seed|.

  3. Ensemble-mean delta_eta_LF -> the deterministic predictor used
     by the live cell.

  4. Monte Carlo over b_hat noise (with the ensemble-mean tau_thr(t)
     as the deterministic post-WCF thrust trajectory): perturbed
     b_hat samples -> peak |R| spread -> sigma_R_b_hat.

Outputs:
  - peak_R_b_hat_sigma_pwq30.png: diagnostic panels
    (b_hat distribution, per-seed delta_eta_LF traces, MC peak
    distribution from b_hat noise, sigma envelope decomposition).
  - scenario_pwq30_calibration.npz with:
      t_grid, delta_eta_mean, delta_eta_seeds, peak_R_seeds,
      sigma_R_b_hat_m, b_hat_mean, b_hat_std, seed_ids, tag.
  - printed sigma_R_b_hat in metres for adding to the live cell.
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
# Cell parameters (defaults match pwq30; override on CLI for other cells).
TAG = "pwq30"
SEEDS = list(range(1000, 1030))
T_WCF = 560.0
T_HORIZON = 60.0
DT = 0.05

# b_hat snapshot moment (live cell convention: t = T_WCF - 5).
B_HAT_SNAPSHOT_T = T_WCF - 5.0


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default=TAG,
                   help=f"Cell tag, also work-dir prefix (default: {TAG})")
    p.add_argument("--t-wcf", type=float, default=T_WCF,
                   help=f"WCF injection time in seconds (default: {T_WCF})")
    p.add_argument("--seeds", default=f"{SEEDS[0]}-{SEEDS[-1]+1}",
                   help="Seed range as 'lo-hi' (Python-style half-open) "
                        f"(default: {SEEDS[0]}-{SEEDS[-1]+1})")
    return p.parse_args()


def _apply_args(args):
    """Mutate module-level cell parameters from parsed args."""
    global TAG, T_WCF, B_HAT_SNAPSHOT_T, SEEDS
    TAG = args.tag
    T_WCF = float(args.t_wcf)
    B_HAT_SNAPSHOT_T = T_WCF - 5.0
    lo, hi = args.seeds.split("-")
    SEEDS = list(range(int(lo), int(hi)))


def _load_seed(seed):
    """Return (M, E) main + estimator dicts for the seed, or None."""
    seed_dir = WORK_ROOT / f"{TAG}_seed{seed:04d}"
    if not seed_dir.exists():
        return None
    main_p = next((p for p in seed_dir.glob("*.out") if "estimator" not in p.name), None)
    est_p = next(seed_dir.glob("*estimator*.out"), None)
    if main_p is None or est_p is None:
        return None
    with open(main_p) as f:
        hdr = f.readline().strip().split("\t")
    M = {h: data for h, data in zip(hdr, np.loadtxt(main_p, skiprows=1, delimiter="\t").T)}
    if M["t"][-1] < T_WCF + T_HORIZON:
        return None
    with open(est_p) as f:
        hdr_e = f.readline().strip().split("\t")
    E = {h: data for h, data in zip(hdr_e, np.loadtxt(est_p, skiprows=1, delimiter="\t").T)}
    return M, E


def _per_seed_tau_lost(M, E, t_grid):
    """Return (tau_lost_seed[N,3], tau_pre_seed[3]) for the seed.

    tau_pre := -b_hat at t = T_WCF - 5  (live cell convention; brucon
    NPO bias estimate, which is well-converged via T_b ~ 1000 s).

    tau_lost(t) := T_post(t) - T_pre, the post-WCF thrust DEVIATION
    from the intact-equilibrium thrust. This matches the authoritative
    convention used by cqa.decision_matrix._wcfdi_peak_at_forecast_obs
    (lines 519-527) and cqa.live_decision.summarise_for_operator_live
    (line 452). In the pulse_response framework with B_lost = +Minv,
    the hull-felt deviation force IS tau_lost (no extra sign flip).

    Earlier version of this script (pre-2026-05-12) used the OPPOSITE
    sign tau_lost = tau_pre - T_post, which propagated as a sign-flipped
    delta_eta_mean in every calibration npz and infected the
    brucon-validation harness. See analysis.md sec.12.21.17.
    """
    # b_hat snapshot from brucon NPO (kN/kNm) -> N/Nm via *1e3.
    k_bh = int(np.argmin(np.abs(E["Time"] - B_HAT_SNAPSHOT_T)))
    b_hat = np.array([
        float(E["EstBiasSurge"][k_bh]),
        float(E["EstBiasSway"][k_bh]),
        float(E["EstBiasYaw"][k_bh]),
    ]) * 1e3
    # tau_env := +b_hat (env force on vessel)  -> tau_pre := -b_hat.
    tau_pre = -b_hat

    # Brucon-realised post-WCF thrust on the truth vessel.
    t = M["t"]
    post = (t >= T_WCF) & (t <= T_WCF + T_HORIZON + 1.0)
    t_post = t[post]
    tau_thr_post = np.column_stack([
        M["Tx"][post], M["Ty"][post], M["Tz"][post]
    ]) * 1e3
    tau_thr_grid = np.column_stack([
        np.interp(t_grid + T_WCF, t_post, tau_thr_post[:, k])
        for k in range(3)
    ])
    # tau_lost = T_post - T_pre  (authoritative convention; sec.12.21.17)
    tau_lost_grid = tau_thr_grid - tau_pre[None, :]
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

    # ---- Collect per-seed b_hat snapshot, tau_thr trajectory, derived tau_lost ----
    per_seed_tau_lost = []
    per_seed_tau_pre = []
    per_seed_b_hat = []
    per_seed_tau_thr = []
    seed_ids = []
    for s in SEEDS:
        out = _load_seed(s)
        if out is None:
            continue
        M, E = out
        tau_lost_seed, tau_pre_seed = _per_seed_tau_lost(M, E, t_grid)
        per_seed_tau_lost.append(tau_lost_seed)
        per_seed_tau_pre.append(tau_pre_seed)
        per_seed_b_hat.append(-tau_pre_seed)        # b_hat = -tau_pre
        # Reconstruct per-seed tau_thr(t) for the MC step.
        # New convention (sec.12.21.17): tau_lost = T_post - T_pre
        # => T_post = T_pre + tau_lost
        per_seed_tau_thr.append(tau_pre_seed[None, :] + tau_lost_seed)
        seed_ids.append(s)
    n_seeds = len(per_seed_tau_lost)
    print(f"Loaded {n_seeds} seeds")

    tau_lost_arr = np.stack(per_seed_tau_lost, axis=0)   # [n_seeds, N, 3]
    tau_pre_arr = np.stack(per_seed_tau_pre, axis=0)      # [n_seeds, 3]
    b_hat_arr = np.stack(per_seed_b_hat, axis=0)           # [n_seeds, 3]
    tau_thr_arr = np.stack(per_seed_tau_thr, axis=0)        # [n_seeds, N, 3]

    b_hat_mean = b_hat_arr.mean(axis=0)
    b_hat_std = b_hat_arr.std(axis=0)
    tau_thr_mean = tau_thr_arr.mean(axis=0)              # [N, 3]
    tau_lost_mean = tau_lost_arr.mean(axis=0)            # [N, 3]
    tau_lost_std = tau_lost_arr.std(axis=0)              # [N, 3]

    print(f"\nb_hat snapshot at t=T_WCF-5 (kN, kN, kNm) across {n_seeds} seeds:")
    print(f"  mean = {b_hat_mean / 1e3}")
    print(f"  std  = {b_hat_std / 1e3}    <-- live observer noise")

    print(f"\ntau_lost(t) max per axis (kN, kN, kNm) over [0, {T_HORIZON:.0f}] s:")
    print(f"  ensemble mean: {np.max(np.abs(tau_lost_mean), axis=0) / 1e3}")

    # ---- Forward-sim with the ensemble-mean tau_lost (deterministic) ----
    X_mean = pulse_response(aug, t_grid, tau_lost_mean, x0=np.zeros(N_STATE))
    deta_LF_mean = X_mean[:, IDX_ETA_HAT][:, 0:2]
    R_mean = np.hypot(deta_LF_mean[:, 0], deta_LF_mean[:, 1])
    peak_R_mean = float(R_mean.max())

    # ---- Per-seed forward sim with seed's tau_lost ----
    peak_R_seeds = np.zeros(n_seeds)
    deta_LF_seeds = np.zeros((n_seeds, N, 3))
    for i, tl in enumerate(per_seed_tau_lost):
        X = pulse_response(aug, t_grid, tl, x0=np.zeros(N_STATE))
        deta_LF_seeds[i] = X[:, IDX_ETA_HAT]
        peak_R_seeds[i] = np.hypot(deta_LF_seeds[i, :, 0], deta_LF_seeds[i, :, 1]).max()

    # ---- Monte Carlo: propagate b_hat noise through cqa-27 ----
    # Hold tau_thr at the ensemble-mean trajectory and perturb b_hat (-> tau_pre)
    # by Gaussian noise calibrated by b_hat_std. The resulting peak |R| spread
    # is sigma_R_b_hat: the live-cell envelope contribution from observer noise.
    rng = np.random.default_rng(0)
    n_mc = 500
    peaks_mc = np.zeros(n_mc)
    for k in range(n_mc):
        db = rng.normal(0.0, b_hat_std)
        tau_pre_k = -(b_hat_mean + db)
        # New convention (sec.12.21.17): tau_lost = T_post - T_pre
        tau_lost_k = tau_thr_mean - tau_pre_k[None, :]
        Xk = pulse_response(aug, t_grid, tau_lost_k, x0=np.zeros(N_STATE))
        de = Xk[:, IDX_ETA_HAT][:, 0:2]
        peaks_mc[k] = np.hypot(de[:, 0], de[:, 1]).max()
    sigma_R_b_hat = float(peaks_mc.std())

    print(f"\n--- Result ---")
    print(f"Peak |delta_eta_LF| of mean-tau_lost prediction: {peak_R_mean:.3f} m")
    print(f"Per-seed peak |delta_eta_LF|: mean={peak_R_seeds.mean():.3f}, "
          f"std={peak_R_seeds.std():.3f} m, "
          f"range=[{peak_R_seeds.min():.3f}, {peak_R_seeds.max():.3f}]")
    print(f"MC peak |R| (b_hat noise propagated, n_mc={n_mc}):")
    print(f"  mean={peaks_mc.mean():.3f} m, std={peaks_mc.std():.3f} m")
    print(f"  -> sigma_R_b_hat = {sigma_R_b_hat:.3f} m")
    print()
    print(f"For reference, the previous OrderTau-based estimator gave "
          f"sigma_R_tau_lost ~ 0.46 m (~6x larger).")

    # ---- Plot ----
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
    ax.set_title(f"Realised tau_lost: ensemble mean +/-1 sigma (n={n_seeds})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (0,1) per-seed delta_eta_LF radial trajectories.
    ax = axes[0, 1]
    R_seeds = np.hypot(deta_LF_seeds[:, :, 0], deta_LF_seeds[:, :, 1])
    for i in range(n_seeds):
        ax.plot(t_grid, R_seeds[i], color="grey", lw=0.5, alpha=0.5)
    ax.plot(t_grid, R_mean, color="k", lw=2.0, label="ensemble-mean tau_lost prediction")
    ax.set_xlabel("t - t_WCF [s]")
    ax.set_ylabel("|delta_eta_LF| [m]")
    ax.set_title("pulse_response(x0=0) with per-seed tau_lost")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,0) MC histogram of peak |R| from b_hat noise propagation.
    ax = axes[1, 0]
    ax.hist(peaks_mc, bins=20, color="C2", alpha=0.7,
            label=f"MC (n={n_mc}) b_hat noise -> peak |R|")
    ax.axvline(peaks_mc.mean(), color="k", lw=1.5,
               label=f"MC mean = {peaks_mc.mean():.2f} m")
    ax.axvspan(peaks_mc.mean() - sigma_R_b_hat,
               peaks_mc.mean() + sigma_R_b_hat,
               color="C2", alpha=0.15,
               label=f"+/-1 sigma_R_b_hat = {sigma_R_b_hat:.3f} m")
    ax.set_xlabel("MC peak |delta_eta_LF| [m]")
    ax.set_ylabel("count")
    ax.set_title(f"sigma_R_b_hat from MC b_hat propagation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,1) summary bar: contributions to total sigma_R.
    sigma_R_LF_typ = 0.36         # typical from validation script
    sigma_R_WF_typ = 0.53
    sigma_R_total_old = float(np.sqrt(sigma_R_LF_typ**2 + sigma_R_WF_typ**2 + 0.46**2))
    sigma_R_total_new = float(np.sqrt(sigma_R_LF_typ**2 + sigma_R_WF_typ**2 + sigma_R_b_hat**2))
    ax = axes[1, 1]
    bars = ["sigma_R_LF\n(W=60s)", "sigma_R_WF\n(W=60s)",
            "sigma_R_b_hat\n(MC, this study)",
            "total\n(old: tau_lost)", "total\n(new: b_hat)"]
    vals = [sigma_R_LF_typ, sigma_R_WF_typ, sigma_R_b_hat,
            sigma_R_total_old, sigma_R_total_new]
    colors2 = ["C0", "C1", "C3", "grey", "C2"]
    ax.bar(bars, vals, color=colors2)
    for b, v in zip(bars, vals):
        ax.text(b, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("sigma_R [m]")
    ax.set_title(f"Sigma envelope contributions (typical {TAG} values)")
    ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8)

    plt.suptitle(f"sigma_R_b_hat calibration at {TAG} (n={n_seeds})", fontsize=12)
    plt.tight_layout()
    out = THIS / f"peak_R_b_hat_sigma_{TAG}.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}")

    # ---- Save calibration artefact for the live cell ----
    # delta_eta_mean = pulse_response(ensemble-mean tau_lost) -- the
    # deterministic predictor the live cell will use.
    # sigma_R_b_hat = MC propagation of b_hat noise through cqa-27.
    artefact = THIS / f"scenario_{TAG}_calibration.npz"
    np.savez(
        artefact,
        t_grid=t_grid,
        delta_eta_mean=X_mean[:, IDX_ETA_HAT],          # [N, 3] surge/sway/yaw
        delta_eta_seeds=deta_LF_seeds,                  # [n_seeds, N, 3]
        peak_R_seeds=peak_R_seeds,
        b_hat_mean=b_hat_mean,                          # N, N, Nm
        b_hat_std=b_hat_std,
        sigma_R_b_hat_m=np.array(sigma_R_b_hat),
        peaks_mc=peaks_mc,
        seed_ids=np.array(seed_ids),
        n_seeds=np.array(n_seeds),
        tag=np.array(TAG),
        convention=np.array(
            "delta_eta_mean = pulse_response(ensemble-mean tau_lost) "
            "with tau_lost := T_post - T_pre (sec.12.21.17 sign fix); "
            "sigma_R_b_hat from MC over b_hat measurement noise"
        ),
    )
    print(f"saved {artefact}")


if __name__ == "__main__":
    _apply_args(_parse_args())
    main()
