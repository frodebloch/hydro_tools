"""'Where we are now' diagnostic for the live-cell WCFDI envelope.

Per-seed comparison at pwq30 (n=30) showing:

  Panel (0,0): per-seed scatter cqa-prediction-peak vs brucon-truth-peak,
               under three predictor variants:
               - V1 OrderTau-based per-seed tau_pre (original calibration)
               - V2 b_hat-based per-seed tau_pre  (b_hat per seed at T_WCF-5)
               - V3 b_hat-based ENSEMBLE-MEAN tau_pre (live-cell-equivalent)

  Panel (0,1): histogram + bootstrap CIs of peak |R| distributions for
               brucon truth and the three cqa variants.

  Panel (1,0): cqa peak |R| stack-up showing each component contribution:
               eta_hat IC, delta_eta deterministic, k*sigma_R envelope.
               Plotted against brucon-truth peak per seed.

  Panel (1,1): bias and prediction-error std table summary.

Also reports:
  - sigma_R_b_hat (b_hat-based propagation through cqa-27)
  - vs sigma_R_tau_lost (OrderTau-based, original)

Run:
    PYTHONPATH=. .venv/bin/python \\
        scripts/p7_brucon_validation/where_we_are_now_pwq30.py
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
    IDX_ETA_HAT,
    N_STATE,
)
from cqa.vessel import LinearVesselModel

WORK_ROOT = THIS / "work"
# Cell parameters (defaults match pwq30; override on CLI for other cells).
TAG = "pwq30"
SEEDS = list(range(1000, 1030))
T_WCF = 560.0
T_HORIZON = 60.0
DT = 0.05
T_PRE_LO = T_WCF - 30.0
T_PRE_HI = T_WCF - 5.0
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
    global TAG, T_WCF, T_PRE_LO, T_PRE_HI, B_HAT_SNAPSHOT_T, SEEDS
    TAG = args.tag
    T_WCF = float(args.t_wcf)
    T_PRE_LO = T_WCF - 30.0
    T_PRE_HI = T_WCF - 5.0
    B_HAT_SNAPSHOT_T = T_WCF - 5.0
    lo, hi = args.seeds.split("-")
    SEEDS = list(range(int(lo), int(hi)))


def _load_seed(seed):
    seed_dir = WORK_ROOT / f"{TAG}_seed{seed:04d}"
    main_p = next((p for p in seed_dir.glob("*.out") if "estimator" not in p.name), None)
    est_p = next(seed_dir.glob("*estimator*.out"), None)
    if main_p is None or est_p is None:
        return None
    with open(main_p) as f:
        hdr = f.readline().strip().split("\t")
    M = {h: c for h, c in zip(hdr, np.loadtxt(main_p, skiprows=1, delimiter="\t").T)}
    with open(est_p) as f:
        hdr_e = f.readline().strip().split("\t")
    E = {h: c for h, c in zip(hdr_e, np.loadtxt(est_p, skiprows=1, delimiter="\t").T)}
    if M["t"][-1] < T_WCF + T_HORIZON:
        return None
    return M, E


def _bootstrap_ci(x, q="mean", B=5000, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    if q == "mean":
        s = x[idx].mean(axis=1)
    else:
        s = np.percentile(x[idx], q, axis=1)
    return np.percentile(s, [2.5, 50, 97.5])


def main():
    cfg = csov_default_config()
    vessel = LinearVesselModel.from_config(cfg.vessel)
    cp = cfg.controller
    ctrl = LinearDpController.from_bandwidth(
        vessel.M, vessel.D, omega_n=cp.omega_n, zeta=cp.zeta
    )
    obs_g = csov_observer_gains(Tp_s=10.0)
    aug = build_observer_augmented_system_full(
        vessel, ctrl, obs_gains=obs_g, T_thr=cp.thruster_time_constant_s
    )
    t_grid = np.arange(0.0, T_HORIZON + 1e-9, DT)

    # ---- per-seed inputs ----
    n_seeds = 0
    eta_lf_seeds = []         # [n,2] LF position at t=T_WCF
    tau_pre_OT_seeds = []     # [n,3] from OrderTau, kN
    tau_pre_bh_seeds = []     # [n,3] from -b_hat at T_WCF-5, kN (sign-corrected)
    tau_thr_seeds = []        # [n,N,3] kN
    truth_peak_abs = []       # [n] brucon |R| peak (vs DP setpoint)

    for seed in SEEDS:
        out = _load_seed(seed)
        if out is None:
            continue
        M, E = out
        pre = (M["t"] >= T_PRE_LO) & (M["t"] <= T_PRE_HI)
        post = (M["t"] >= T_WCF) & (M["t"] <= T_WCF + T_HORIZON + 1)
        t_post = M["t"][post]

        eta_lf_seeds.append([M["SurgeDev"][post][0], M["SwayDev"][post][0]])
        tau_pre_OT_seeds.append([
            float(M["OrderTauSurge"][pre].mean()),
            float(M["OrderTauSway"][pre].mean()),
            float(M["OrderTauYaw"][pre].mean()),
        ])
        k_bh = int(np.argmin(np.abs(E["Time"] - B_HAT_SNAPSHOT_T)))
        # Live cell convention: tau_env := +b_hat -> tau_pre := -b_hat
        tau_pre_bh_seeds.append([
            -float(E["EstBiasSurge"][k_bh]),
            -float(E["EstBiasSway"][k_bh]),
            -float(E["EstBiasYaw"][k_bh]),
        ])
        tau_thr_seeds.append(np.column_stack([
            np.interp(t_grid + T_WCF, t_post, M["Tx"][post]),
            np.interp(t_grid + T_WCF, t_post, M["Ty"][post]),
            np.interp(t_grid + T_WCF, t_post, M["Tz"][post]),
        ]))
        truth_peak_abs.append(
            float(np.hypot(M["SurgeDev"][post], M["SwayDev"][post]).max())
        )
        n_seeds += 1

    eta_lf_seeds = np.asarray(eta_lf_seeds)
    tau_pre_OT_seeds = np.asarray(tau_pre_OT_seeds) * 1e3   # N, Nm
    tau_pre_bh_seeds = np.asarray(tau_pre_bh_seeds) * 1e3
    tau_thr_seeds = np.asarray(tau_thr_seeds) * 1e3
    truth_peak_abs = np.asarray(truth_peak_abs)

    tau_pre_bh_ensemble = tau_pre_bh_seeds.mean(axis=0)

    # ---- run cqa under three predictors ----
    def _peak_abs(tau_pre_per_seed, eta_lf_per_seed):
        peaks = np.zeros(n_seeds)
        deta_at_peak = np.zeros((n_seeds, 2))
        for i in range(n_seeds):
            # tau_lost = T_post - T_pre (authoritative; sec.12.21.17 sign fix)
            tau_lost = tau_thr_seeds[i] - tau_pre_per_seed[i][None, :]
            X = pulse_response(aug, t_grid, tau_lost, x0=np.zeros(N_STATE))
            de = X[:, IDX_ETA_HAT][:, 0:2]
            R = np.hypot(eta_lf_per_seed[i, 0] + de[:, 0],
                         eta_lf_per_seed[i, 1] + de[:, 1])
            peaks[i] = R.max()
            kpk = int(np.argmax(R))
            deta_at_peak[i] = de[kpk]
        return peaks, deta_at_peak

    # V1: OrderTau per seed
    peak_V1, _ = _peak_abs(tau_pre_OT_seeds, eta_lf_seeds)
    # V2: b_hat per seed
    peak_V2, _ = _peak_abs(tau_pre_bh_seeds, eta_lf_seeds)
    # V3: b_hat ensemble mean (live-cell-equivalent)
    peak_V3, deta_pk_V3 = _peak_abs(
        np.tile(tau_pre_bh_ensemble, (n_seeds, 1)), eta_lf_seeds
    )

    # Component breakdown for V3 (live cell)
    eta_hat_mag = np.hypot(eta_lf_seeds[:, 0], eta_lf_seeds[:, 1])
    deta_mag_at_peak_V3 = np.hypot(deta_pk_V3[:, 0], deta_pk_V3[:, 1])

    # Sigma envelope contribution.
    #   sigma_R_b_hat: cell-specific, from offline calibration npz produced
    #     by peak_R_b_hat_sigma_pwq30.py (script-name fossilised, multi-cell).
    #   sigma_R_LF, sigma_R_WF: cell-specific in principle (depend on sea
    #     state and observer Tp). Computed here from the brucon ensemble
    #     pre-WCF window as a stand-in for the per-seed Bayesian posterior
    #     medians (which the live cell uses). Acceptable for offline
    #     diagnostics; the live cell itself uses BayesianSigmaEstimator.
    calib_npz = THIS / f"scenario_{TAG}_calibration.npz"
    if calib_npz.exists():
        SIGMA_R_BHAT = float(np.load(calib_npz)["sigma_R_b_hat_m"])
    else:
        SIGMA_R_BHAT = 0.073  # pwq30 fallback

    # Pre-WCF ensemble-std proxy for LF and WF radial sigmas.
    pre_lf_dx = []; pre_lf_dy = []
    pre_wf_dx = []; pre_wf_dy = []
    for seed in SEEDS:
        out = _load_seed(seed)
        if out is None:
            continue
        M, E = out
        pre = (M["t"] >= T_PRE_LO) & (M["t"] <= T_PRE_HI)
        # LF residual: SurgeDev/SwayDev minus their 60-s mean.
        sx = M["SurgeDev"][pre]; sy = M["SwayDev"][pre]
        pre_lf_dx.append(sx - sx.mean()); pre_lf_dy.append(sy - sy.mean())
        # WF residual: HfPosX/Y from estimator log.
        pre_e = (E["Time"] >= T_PRE_LO) & (E["Time"] <= T_PRE_HI)
        wx = E["HfPosX"][pre_e]; wy = E["HfPosY"][pre_e]
        pre_wf_dx.append(wx - wx.mean()); pre_wf_dy.append(wy - wy.mean())
    sig_lf_x = float(np.std(np.concatenate(pre_lf_dx)))
    sig_lf_y = float(np.std(np.concatenate(pre_lf_dy)))
    sig_wf_x = float(np.std(np.concatenate(pre_wf_dx)))
    sig_wf_y = float(np.std(np.concatenate(pre_wf_dy)))
    SIGMA_R_LF = float(np.hypot(sig_lf_x, sig_lf_y))
    SIGMA_R_WF = float(np.hypot(sig_wf_x, sig_wf_y))

    SIGMA_R_TOTAL = float(np.sqrt(SIGMA_R_LF**2 + SIGMA_R_WF**2 + SIGMA_R_BHAT**2))
    K_SIGMA = 0.674
    sigma_halo = K_SIGMA * SIGMA_R_TOTAL

    envelope_V3 = peak_V3 + sigma_halo

    # ---- stats ----
    rng = np.random.default_rng(42)

    def stats(x):
        m = x.mean(); s = x.std()
        p = np.percentile(x, [5, 25, 50, 75, 95])
        return m, s, p

    def _print(label, x):
        m, s, p = stats(x)
        print(f"  {label:30s} mean={m:.3f}  std={s:.3f}  P5={p[0]:.2f} "
              f"P50={p[2]:.2f} P75={p[3]:.2f} P95={p[4]:.2f}")

    print(f"=== Predictors (peak abs |R|, with eta_hat IC, n={n_seeds}) ===")
    _print("V1: OrderTau per-seed",   peak_V1)
    _print("V2: b_hat per-seed",      peak_V2)
    _print("V3: b_hat ensemble (live)", peak_V3)
    _print("brucon truth absolute",   truth_peak_abs)
    print()
    print(f"=== Live-cell envelope (V3 + k_sigma * sigma_R_total) ===")
    _print("V3 + 0.674*sigma_total", envelope_V3)
    print(f"  sigma_R_LF={SIGMA_R_LF}, sigma_R_WF={SIGMA_R_WF}, sigma_R_b_hat={SIGMA_R_BHAT}")
    print(f"  sigma_R_total = {SIGMA_R_TOTAL:.3f}, halo = {sigma_halo:.3f} m")
    print()

    print(f"=== Bootstrap 95% CI on means ===")
    for x, l in [(peak_V1, "V1"), (peak_V2, "V2"), (peak_V3, "V3"),
                 (envelope_V3, "V3+halo"), (truth_peak_abs, "truth")]:
        lo, m, hi = _bootstrap_ci(x, "mean", rng=rng)
        print(f"  {l:10s}  [{lo:.2f}, {m:.2f}, {hi:.2f}]")
    print()

    # Coverage of truth peak by V3 envelope
    cov = (envelope_V3 >= truth_peak_abs).mean()
    print(f"=== Operational coverage ===")
    print(f"  envelope >= truth:  {cov*100:.0f}% (n={n_seeds})")
    cov_lo, cov_med, cov_hi = _bootstrap_ci((envelope_V3 >= truth_peak_abs).astype(float), "mean", rng=rng)
    print(f"  bootstrap 95% CI:   [{cov_lo*100:.0f}%, {cov_med*100:.0f}%, {cov_hi*100:.0f}%]")

    # ---- plot ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (0,0) per-seed scatter
    ax = axes[0, 0]
    lim = 1.05 * max(peak_V1.max(), peak_V2.max(), peak_V3.max(),
                     truth_peak_abs.max(), envelope_V3.max())
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y=x (perfect)")
    ax.scatter(truth_peak_abs, peak_V1, s=30, c="C0", marker="o", alpha=0.7,
               label=f"V1 OrderTau per-seed  (bias={peak_V1.mean()-truth_peak_abs.mean():+.2f})")
    ax.scatter(truth_peak_abs, peak_V2, s=30, c="C1", marker="s", alpha=0.7,
               label=f"V2 b_hat per-seed     (bias={peak_V2.mean()-truth_peak_abs.mean():+.2f})")
    ax.scatter(truth_peak_abs, peak_V3, s=30, c="C2", marker="^", alpha=0.7,
               label=f"V3 b_hat ensemble    (bias={peak_V3.mean()-truth_peak_abs.mean():+.2f})")
    ax.scatter(truth_peak_abs, envelope_V3, s=22, c="C3", marker="x",
               label=f"V3 + k*sigma envelope ({cov*100:.0f}% coverage)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("brucon truth peak |R| [m]")
    ax.set_ylabel("cqa-prediction peak |R| [m]")
    ax.set_title("Per-seed scatter: cqa prediction vs brucon truth")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # (0,1) histogram comparison
    ax = axes[0, 1]
    bins = np.linspace(0, lim, 18)
    ax.hist(peak_V1, bins=bins, alpha=0.45, color="C0", label="V1 OrderTau", density=True)
    ax.hist(peak_V3, bins=bins, alpha=0.45, color="C2", label="V3 b_hat ens", density=True)
    ax.hist(envelope_V3, bins=bins, alpha=0.45, color="C3",
            label="V3+halo envelope", density=True)
    ax.hist(truth_peak_abs, bins=bins, alpha=0.45, color="k",
            label="brucon truth", density=True)
    for x, c, ls in [(peak_V1, "C0", "-"), (peak_V3, "C2", "-"),
                     (envelope_V3, "C3", "-"), (truth_peak_abs, "k", "-")]:
        ax.axvline(x.mean(), color=c, ls=ls, lw=1.2, alpha=0.9)
    ax.set_xlabel("peak |R| [m]")
    ax.set_ylabel("density")
    ax.set_title("Distribution comparison (vertical lines = means)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (1,0) component stack-up bar chart per seed (sorted by truth)
    ax = axes[1, 0]
    order = np.argsort(truth_peak_abs)
    x_ind = np.arange(n_seeds)
    eh = eta_hat_mag[order]
    dd = peak_V3[order] - eh                           # delta_eta contribution
    sh = np.full(n_seeds, sigma_halo)
    truth_sorted = truth_peak_abs[order]
    ax.bar(x_ind, eh, color="C0", alpha=0.85, label=f"|eta_hat_LF| (mean {eta_hat_mag.mean():.2f})")
    ax.bar(x_ind, dd, bottom=eh, color="C2", alpha=0.85,
           label=f"|...+delta_eta_LF|-|eta_hat| (mean {dd.mean():.2f})")
    ax.bar(x_ind, sh, bottom=eh+dd, color="C3", alpha=0.85,
           label=f"k*sigma_R_total halo ({sigma_halo:.2f} m)")
    ax.plot(x_ind, truth_sorted, "k_", ms=10, mew=2, label="brucon truth peak")
    ax.set_xlabel("seed (sorted by brucon truth peak)")
    ax.set_ylabel("peak |R| [m]")
    ax.set_title("V3 envelope stack-up vs truth (per seed)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # (1,1) summary text panel
    ax = axes[1, 1]; ax.axis("off")
    err_V3 = truth_peak_abs - peak_V3
    rmse_V3 = float(np.sqrt((err_V3**2).mean()))
    err_envV3 = truth_peak_abs - envelope_V3
    summary = [
        f"=== sigma envelope decomposition (was tau_lost-based, now b_hat-based) ===",
        "",
        f"  sigma_R_LF       = {SIGMA_R_LF:.2f} m   (Bayesian posterior, 60s window)",
        f"  sigma_R_WF       = {SIGMA_R_WF:.2f} m   (Bayesian posterior, 60s window)",
        f"  sigma_R_b_hat    = {SIGMA_R_BHAT:.2f} m   (b_hat noise propagated through cqa-27)",
        f"  sigma_R_total    = {SIGMA_R_TOTAL:.2f} m   (quadrature)",
        f"  k_sigma          = {K_SIGMA}        (P75 single-tail)",
        f"  halo             = {sigma_halo:.2f} m",
        "",
        f"  PREVIOUS (sigma_R_tau_lost = 0.46 m)  -> halo = {K_SIGMA*np.sqrt(SIGMA_R_LF**2+SIGMA_R_WF**2+0.46**2):.2f} m",
        f"  CURRENT  (sigma_R_b_hat    = 0.07 m)  -> halo = {sigma_halo:.2f} m",
        "",
        f"=== Predictor performance vs brucon truth (n={n_seeds}) ===",
        "",
        f"  V1 OrderTau per-seed:  bias = {peak_V1.mean()-truth_peak_abs.mean():+.3f} m, std(err) = {(truth_peak_abs-peak_V1).std():.3f} m",
        f"  V2 b_hat per-seed:     bias = {peak_V2.mean()-truth_peak_abs.mean():+.3f} m, std(err) = {(truth_peak_abs-peak_V2).std():.3f} m",
        f"  V3 b_hat ensemble:     bias = {peak_V3.mean()-truth_peak_abs.mean():+.3f} m, std(err) = {err_V3.std():.3f} m",
        f"  V3 + halo:             bias = {envelope_V3.mean()-truth_peak_abs.mean():+.3f} m, RMSE = {rmse_V3:.3f} m",
        "",
        f"=== Operational coverage of truth peak by V3 envelope ===",
        f"  envelope >= truth: {cov*100:.0f}%  (target: ~75% for k_sigma=0.674)",
        f"  bootstrap 95% CI: [{cov_lo*100:.0f}%, {cov_med*100:.0f}%, {cov_hi*100:.0f}%]",
        "",
        f"=== Direction of sign convention check ===",
        f"  b_hat_mean (kN/kNm)   = {tau_pre_bh_ensemble[0]/-1e3:+.1f}, {tau_pre_bh_ensemble[1]/-1e3:+.1f}, {tau_pre_bh_ensemble[2]/-1e3:+.1f}",
        f"  -> tau_env=+b_hat (env pushes vessel in this direction)",
        f"  -> tau_pre=-b_hat = {tau_pre_bh_ensemble[0]/1e3:+.1f}, {tau_pre_bh_ensemble[1]/1e3:+.1f}, {tau_pre_bh_ensemble[2]/1e3:+.1f} (counter-thrust)",
    ]
    ax.text(0.0, 0.98, "\n".join(summary), va="top", ha="left",
            fontfamily="monospace", fontsize=8.5)

    plt.suptitle(
        f"Where we are now: live-cell WCFDI envelope vs brucon truth ({TAG}, n={n_seeds})",
        fontsize=12,
    )
    plt.tight_layout()
    out = THIS / f"where_we_are_now_{TAG}.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    _apply_args(_parse_args())
    main()
