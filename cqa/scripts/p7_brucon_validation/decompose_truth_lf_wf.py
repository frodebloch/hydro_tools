"""LF vs WF decomposition of brucon truth (analysis.md sec.12.21.13).

Splits the per-seed truth R_peak into its LF (SurgeDev/SwayDev, the DP
LF estimator output) and WF (xHf/yHf, the DP wave-filter output)
constituents:

  R_LF_peak  = max |[SurgeDev - sd_pre, SwayDev - wd_pre]| in post-WCF
               window (this is what the b_hat-driven scenario IRF
               should predict deterministically)

  R_WF_peak  = max |[xHf, yHf]| in post-WCF window (this is what cqa's
               sigma_WF Gumbel envelope is supposed to capture
               statistically)

  R_TOT_peak = max |LF + WF| (the combined position excursion, what
               we've been calling "truth")

Then compares per-seed:
  * marginal P50/P95 of R_LF and R_WF distributions across seeds
  * to cqa predictions: variant A R_det (LF prediction) and Gumbel σ_WF
  * per-seed correlation between cqa A_R and brucon R_LF (this is the
    proper LF-only correlation test; previous truth-vs-A_R used
    R_TOT which is LF+WF mixed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))
sys.path.insert(0, str(THIS))

import live_cell_per_seed_pwq30 as live_cell  # noqa: E402
from live_cell_per_seed_pwq30 import (         # noqa: E402
    _load_tsv, load_seed, build_live_sigma_posterior,
)
from harness import parse_output                # noqa: E402

from cqa.config import csov_default_config     # noqa: E402
from cqa.live_decision import _build_aug_for_live  # noqa: E402
from cqa.transient_obs import (                # noqa: E402
    pulse_response_with_lift_coupling, pulse_response, N_STATE,
)
from cqa.transient import WcfdiScenario        # noqa: E402

TAG = "bf8_q10_w45"
T_WCF = 560.0
T_HORIZON_S = 60.0
N_T = 121
TP_OBS_S = 10.0
SEED_RANGE = (1000, 1030)
POST_TRUTH_WIN = (5.0, 120.0)
PRE_LF_WIN = (-20.0, -1.0)
# Settled pre-WCF window for sigma calibration (vessel needs ~7 min to
# settle from initial transient; use last ~55 s before WCF)
PRE_SIGMA_WIN = (-60.0, -5.0)


def main():
    # Plumb live_cell module-level globals
    live_cell.TAG = TAG
    live_cell.T_WCF = T_WCF
    live_cell.T_EVAL = T_WCF - 5.0
    live_cell.WIN_END = T_WCF - 1.0
    live_cell.WIN_START = live_cell.WIN_END - live_cell.WIN_S
    live_cell.SEEDS = list(range(SEED_RANGE[0], SEED_RANGE[1]))
    live_cell.CALIB_NPZ = THIS / f"scenario_{TAG}_calibration.npz"

    calib = np.load(live_cell.CALIB_NPZ, allow_pickle=True)
    sigma_R_b_hat_m = float(calib["sigma_R_b_hat_m"])

    cfg = csov_default_config()
    K_lift = float(getattr(cfg.vessel, "lift_coupling_K_per_rad", 0.0))
    scenario = WcfdiScenario(
        alpha=(2.0 / 3.0,) * 3, gamma_immediate=0.5, T_realloc=10.0,
    )
    aug = _build_aug_for_live(cfg, Tp_obs_s=TP_OBS_S)
    t_grid = np.linspace(0.0, T_HORIZON_S, N_T)

    rows = []
    for seed in range(SEED_RANGE[0], SEED_RANGE[1]):
        d = load_seed(seed)
        if d is None:
            continue
        sigma_post = build_live_sigma_posterior(d, sigma_R_b_hat_m=sigma_R_b_hat_m)
        eta_hat_lf = np.asarray(d["eta_hat"], dtype=float)
        b_hat = np.asarray(d["b_hat"], dtype=float)

        # cqa variant A R_det
        gamma_imm = float(scenario.gamma_immediate)
        T_realloc = float(scenario.T_realloc)
        beta_t = 1.0 + (gamma_imm - 1.0) * np.exp(-t_grid / T_realloc)
        tau_lost = (beta_t[:, None] - 1.0) * (-b_hat[None, :])
        X = pulse_response_with_lift_coupling(
            aug, t_grid, tau_lost, b_hat0=b_hat, K_lift=K_lift,
            x0=np.zeros(N_STATE),
        )
        delta_eta_mean = X[:, 0:3]
        eta_xy_t = eta_hat_lf[None, 0:2] + delta_eta_mean[:, 0:2]
        R_det_t = np.hypot(eta_xy_t[:, 0], eta_xy_t[:, 1])
        A_R = float(np.max(R_det_t))

        # Brucon truth, decomposed
        seed_dir = live_cell.WORK_ROOT / f"{TAG}_seed{seed:04d}"
        main_p = next((p for p in seed_dir.glob("*.out") if "estimator" not in p.name), None)
        M = _load_tsv(main_p)
        t = M["t"]
        pre_lf = (t >= T_WCF + PRE_LF_WIN[0]) & (t <= T_WCF + PRE_LF_WIN[1])
        sd_pre = float(M["SurgeDev"][pre_lf].mean())
        wd_pre = float(M["SwayDev"][pre_lf].mean())
        post = (t >= T_WCF + POST_TRUTH_WIN[0]) & (t <= T_WCF + POST_TRUTH_WIN[1])

        LF_x = M["SurgeDev"] - sd_pre
        LF_y = M["SwayDev"] - wd_pre
        WF_x = M["xHf"]
        WF_y = M["yHf"]
        TOT_x = LF_x + WF_x
        TOT_y = LF_y + WF_y

        R_LF = np.hypot(LF_x, LF_y)
        R_WF = np.hypot(WF_x, WF_y)
        R_TOT = np.hypot(TOT_x, TOT_y)

        R_LF_peak = float(np.max(R_LF[post]))
        R_WF_peak = float(np.max(R_WF[post]))
        R_TOT_peak = float(np.max(R_TOT[post]))

        # Also: LF and WF channel sigmas pre-WCF for sigma calibration check
        # Use SETTLED pre-WCF window only (last 55 s before WCF), since the
        # brucon sim has a ~7 min initial settling transient on Bf8 cells.
        settled = (t >= T_WCF + PRE_SIGMA_WIN[0]) & (t <= T_WCF + PRE_SIGMA_WIN[1])
        sig_LF_x_pre = float(np.std(LF_x[settled]))
        sig_LF_y_pre = float(np.std(LF_y[settled]))
        sig_WF_x_pre = float(np.std(WF_x[settled]))
        sig_WF_y_pre = float(np.std(WF_y[settled]))
        sig_LF_pre = float(np.std(R_LF[settled]))
        sig_WF_pre = float(np.std(R_WF[settled]))

        rows.append((seed, A_R, R_LF_peak, R_WF_peak, R_TOT_peak,
                     sig_LF_pre, sig_WF_pre,
                     float(sigma_post.sigma_R_b_hat_m),
                     sig_LF_x_pre, sig_LF_y_pre, sig_WF_x_pre, sig_WF_y_pre))

    arr = np.array(rows)
    A_R = arr[:, 1]
    R_LF = arr[:, 2]
    R_WF = arr[:, 3]
    R_TOT = arr[:, 4]
    sig_LF_pre = arr[:, 5]
    sig_WF_pre = arr[:, 6]
    sig_bh = arr[:, 7]
    sig_LF_x_pre = arr[:, 8]
    sig_LF_y_pre = arr[:, 9]
    sig_WF_x_pre = arr[:, 10]
    sig_WF_y_pre = arr[:, 11]

    print(f"\nn = {len(rows)} seeds, {TAG}")
    print(f"\nPer-seed table (first 10):")
    print(f"{'seed':>5} {'A_R':>6} {'R_LF':>6} {'R_WF':>6} {'R_TOT':>6} {'sigLF':>6} {'sigWF':>6}")
    for r in rows[:10]:
        print(f"{int(r[0]):>5} {r[1]:>6.3f} {r[2]:>6.3f} {r[3]:>6.3f} {r[4]:>6.3f} {r[5]:>6.3f} {r[6]:>6.3f}")

    print(f"\nMarginal P50 / P95 across seeds:")
    print(f"{'channel':<10} {'P50':>7} {'P95':>7} {'mean':>7} {'std':>7}")
    for name, x in [("A_R (cqa)", A_R), ("R_LF", R_LF), ("R_WF", R_WF), ("R_TOT", R_TOT)]:
        print(f"{name:<10} {np.quantile(x,0.5):>7.3f} {np.quantile(x,0.95):>7.3f} "
              f"{x.mean():>7.3f} {x.std():>7.3f}")

    print(f"\nPre-WCF sigma calibration (SETTLED window [T_WCF{PRE_SIGMA_WIN[0]:+.0f}, T_WCF{PRE_SIGMA_WIN[1]:+.0f}] s):")
    print(f"  brucon sigma_LF_x (SurgeDev-demeaned): mean={sig_LF_x_pre.mean():.3f}  std={sig_LF_x_pre.std():.3f}")
    print(f"  brucon sigma_LF_y (SwayDev-demeaned):  mean={sig_LF_y_pre.mean():.3f}  std={sig_LF_y_pre.std():.3f}")
    print(f"  brucon sigma_WF_x (xHf):               mean={sig_WF_x_pre.mean():.3f}")
    print(f"  brucon sigma_WF_y (yHf):               mean={sig_WF_y_pre.mean():.3f}")
    print(f"  brucon sigma_R_LF (radial):            mean={sig_LF_pre.mean():.3f}")
    print(f"  brucon sigma_R_WF (radial):            mean={sig_WF_pre.mean():.3f}")
    print(f"  cqa sigma_R_b_hat_m (posterior):       mean={sig_bh.mean():.3f}")

    print(f"\nPer-seed correlations:")
    for name_x, x in [("R_LF", R_LF), ("R_WF", R_WF), ("R_TOT", R_TOT)]:
        p_r, p_p = pearsonr(A_R, x)
        s_r, s_p = spearmanr(A_R, x)
        print(f"  A_R vs {name_x:<6}  Pearson r={p_r:+.3f} (p={p_p:.3f})  "
              f"Spearman ρ={s_r:+.3f} (p={s_p:.3f})")
    p_r, p_p = pearsonr(R_LF, R_WF)
    print(f"  R_LF vs R_WF   Pearson r={p_r:+.3f} (p={p_p:.3f})")


if __name__ == "__main__":
    main()
