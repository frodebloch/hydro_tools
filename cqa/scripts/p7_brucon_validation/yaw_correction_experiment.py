"""Hybrid tau_lost experiment for bf8_q10_w45.

Tests whether correcting the YAW component of tau_lost (replacing
cqa's `(1-beta(t))*b_hat_yaw` with brucon's ensemble-mean
`Tz(t) - OrderTauYaw(t)`) is sufficient to close the cqa WCF P95
prediction gap on bf8_q10_w45 (currently -15%, see analysis.md
sec.12.21.9 roll-up).

Three variants compared, per seed, then ensemble:

  A baseline   : cqa formula `tau_lost = (1-beta(t)) * b_hat` for all 3 DOFs
  B yaw-only   : cqa formula for surge/sway; brucon ensemble-mean
                 `Tz(t) - OrderTauYaw(t)` (kN/kNm -> N/Nm) for yaw
  C fully-bru  : brucon ensemble-mean `(T - Order)` for all 3 DOFs

For each variant we compute the live operator panel's WCF P95 (same
window-max formula as production: deterministic
|eta_hat_LF + delta_eta_mean(t_peak)| + Gumbel LF/WF window-max +
b_hat halo). Brucon truth is the per-seed realised peak of demeaned
LF radial in [t_eval+5, t_eval+120] s.

Decision rule for the user's question (2026-05):
  * If P95_B closes the gap (matches P95_truth within ~5%), the
    structural cause of the bf8-oblique under-prediction IS the
    yaw-sign error in WcfdiScenario, and the fix is to extend it.
  * If P95_B is still under-predicted but P95_C closes the gap, the
    yaw is necessary but not sufficient; surge/sway magnitudes also
    matter.
  * If even P95_C under-predicts, there's a deeper model problem
    (e.g. observer / closed-loop dynamics) outside the tau_lost
    representation.

Usage:
  .venv/bin/python scripts/p7_brucon_validation/yaw_correction_experiment.py \\
      --tag bf8_q10_w45
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))  # cqa pkg
sys.path.insert(0, str(THIS))                # live_cell_per_seed_pwq30

import live_cell_per_seed_pwq30 as live_cell                  # noqa: E402
from live_cell_per_seed_pwq30 import (                         # noqa: E402
    _load_tsv, load_seed, build_live_sigma_posterior,
)
from harness import parse_output                               # noqa: E402

from cqa.config import csov_default_config                    # noqa: E402
from cqa.live_decision import LiveObserverState, _build_aug_for_live  # noqa: E402
from cqa.live_operator_view import (                          # noqa: E402
    _radial_window_max_quantiles, _sigmas_intact_axis, _sigmas_wcf_axis,
)
from cqa.transient_obs import (                               # noqa: E402
    pulse_response, pulse_response_with_lift_coupling, N_STATE,
)
from cqa.transient import WcfdiScenario                       # noqa: E402


from _constants import T_WCF_S as T_WCF  # noqa: E402  # active script: refresh sec.12.21.16
T_HORIZON_S = 60.0
N_T = 121
TP_OBS_S = 10.0
N_MC = 2000
SEED_RANGE = (1000, 1030)


def load_brucon_taulost_ensemble(tag: str, t_grid: np.ndarray) -> np.ndarray:
    """Return ensemble-mean tau_lost(t) = (Tx,Ty,Tz)_post - (Tx,Ty,Tz)_pre_mean
    aligned to t_grid (t=0 = T_WCF), shape (n_t, 3) in N/Nm.

    NOTE: tau_lost truth is now defined as the hull-experienced thrust
    DEVIATION from pre-WCF intact mean, i.e. the unbalanced force the
    closed-loop system must absorb. This is independent of the DP
    feedback path's transient (FbTau) and of the controller's order
    response (OrderTau), both of which are post-event reactions, not
    causes. See analysis.md sec.12.21.13.

    Sign: tau_lost = T_post - T_pre, so a thrust LOSS gives NEGATIVE
    tau_lost (hull thrust dropped), and the position response is driven
    by env_load - T_post = (env_load - T_pre) - (T_post - T_pre) =
    -tau_lost (since env_load = T_pre at intact equilibrium). So in the
    cqa pulse-response framework where positive tau_lost drives
    positive position offset, we want tau_lost = -(T_post - T_pre).

    But for direct comparison with cqa formula which produces
    (1-beta)*b_hat = positive in direction of env_load, we need brucon
    truth in the same convention: tau_lost_for_cqa = (T_pre - T_post).
    """
    work = THIS / "work"
    pre_window = (-20.0, -1.0)  # 19 s pre-WCF baseline
    rows = []
    for seed in range(SEED_RANGE[0], SEED_RANGE[1]):
        seed_dir = work / f"{tag}_seed{seed:04d}"
        outp = seed_dir / f"{tag}_seed{seed:04d}.out"
        if not outp.exists():
            continue
        m = parse_output(outp)
        t = m.columns["t"] - T_WCF
        if t[-1] < t_grid[-1]:
            continue
        pre_mask = (t >= pre_window[0]) & (t <= pre_window[1])
        Tx_pre = float(m.columns["Tx"][pre_mask].mean())
        Ty_pre = float(m.columns["Ty"][pre_mask].mean())
        Tz_pre = float(m.columns["Tz"][pre_mask].mean())
        rows.append(np.column_stack([
            Tx_pre - np.interp(t_grid, t, m.columns["Tx"]),
            Ty_pre - np.interp(t_grid, t, m.columns["Ty"]),
            Tz_pre - np.interp(t_grid, t, m.columns["Tz"]),
        ]))
    arr = np.array(rows)  # (n, n_t, 3) in kN/kNm, sign matches cqa formula
    return arr.mean(axis=0) * 1e3  # to N/Nm


def run_variant(tau_lost_t: np.ndarray, aug, t_grid: np.ndarray,
                eta_hat_lf: np.ndarray, b_hat: np.ndarray,
                K_lift: float, sigma_post,
                rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Run pulse-response + window-max for one tau_lost(t) candidate.
    Returns (R_det_peak, t_peak, wcf_P50, wcf_P95)."""
    if K_lift > 0.0:
        X = pulse_response_with_lift_coupling(
            aug, t_grid, tau_lost_t, b_hat0=b_hat, K_lift=K_lift,
            x0=np.zeros(N_STATE),
        )
    else:
        X = pulse_response(aug, t_grid, tau_lost_t, x0=np.zeros(N_STATE))
    delta_eta_mean = X[:, 0:3]
    eta_xy_t = eta_hat_lf[None, 0:2] + delta_eta_mean[:, 0:2]
    R_det_t = np.hypot(eta_xy_t[:, 0], eta_xy_t[:, 1])
    k_peak = int(np.argmax(R_det_t))

    sig_lf_x, sig_lf_y = _sigmas_intact_axis(sigma_post)
    sig_wf_x = float(sigma_post.posterior_wf_x.sigma_median)
    sig_wf_y = float(sigma_post.posterior_wf_y.sigma_median)
    sig_bh = float(sigma_post.sigma_R_b_hat_m) / float(np.sqrt(2.0))
    p50, p95 = _radial_window_max_quantiles(
        eta_xy_t[k_peak], sig_lf_x, sig_lf_y, sig_wf_x, sig_wf_y, sig_bh,
        t_horizon_s=T_HORIZON_S, n_mc=N_MC, rng=rng,
    )
    return float(R_det_t[k_peak]), float(t_grid[k_peak]), p50, p95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bf8_q10_w45")
    args = ap.parse_args()
    tag = args.tag

    # Plumb live_cell module-level globals
    live_cell.TAG = tag
    live_cell.T_WCF = T_WCF
    live_cell.T_EVAL = T_WCF - 5.0
    live_cell.WIN_END = T_WCF - 1.0
    live_cell.WIN_START = live_cell.WIN_END - live_cell.WIN_S
    live_cell.SEEDS = list(range(SEED_RANGE[0], SEED_RANGE[1]))
    live_cell.CALIB_NPZ = THIS / f"scenario_{tag}_calibration.npz"

    calib = np.load(live_cell.CALIB_NPZ, allow_pickle=True)
    sigma_R_b_hat_m = float(calib["sigma_R_b_hat_m"])

    cfg = csov_default_config()
    K_lift = float(getattr(cfg.vessel, "lift_coupling_K_per_rad", 0.0))

    # Canonical WcfdiScenario (matches summarise_for_operator_live default)
    scenario = WcfdiScenario(
        alpha=(2.0 / 3.0,) * 3, gamma_immediate=0.5, T_realloc=10.0,
    )

    # Pre-compute aug + t_grid + brucon ensemble-mean tau_lost(t)
    aug = _build_aug_for_live(cfg, Tp_obs_s=TP_OBS_S)
    t_grid = np.linspace(0.0, T_HORIZON_S, N_T)
    tau_lost_brucon = load_brucon_taulost_ensemble(tag, t_grid)  # (n_t, 3) N/Nm

    print(f"\nbrucon ensemble-mean tau_lost(t) at t=0.5/5/10/30 s, [{tag}]:")
    for t0 in (0.5, 5.0, 10.0, 30.0):
        i = int(np.argmin(np.abs(t_grid - t0)))
        print(f"  t={t0:4.1f}s : "
              f"S={tau_lost_brucon[i,0]/1e3:+7.1f} kN  "
              f"W={tau_lost_brucon[i,1]/1e3:+7.1f} kN  "
              f"Y={tau_lost_brucon[i,2]/1e3:+8.1f} kNm")

    # Per-seed loop
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(SEED_RANGE[0], SEED_RANGE[1]):
        d = load_seed(seed)
        if d is None:
            continue
        sigma_post = build_live_sigma_posterior(d, sigma_R_b_hat_m=sigma_R_b_hat_m)
        eta_hat_lf = np.asarray(d["eta_hat"], dtype=float)
        # Apply b_hat steady-state bias correction to match production
        # cqa.live_decision (analysis.md sec.12.21.13).
        b_corr = float(cfg.vessel.b_hat_bias_correction_factor)
        b_hat = b_corr * np.asarray(d["b_hat"], dtype=float)

        # Variant A: cqa baseline
        gamma_imm = float(scenario.gamma_immediate)
        T_realloc = float(scenario.T_realloc) if scenario.T_realloc > 0 else 1e-9
        beta_t = 1.0 + (gamma_imm - 1.0) * np.exp(-t_grid / T_realloc)
        tau_lost_A = (beta_t[:, None] - 1.0) * (-b_hat[None, :])

        # Variant B: yaw replaced with brucon ensemble-mean
        tau_lost_B = tau_lost_A.copy()
        tau_lost_B[:, 2] = tau_lost_brucon[:, 2]

        # Variant C: all three from brucon ensemble-mean
        tau_lost_C = tau_lost_brucon.copy()

        # Variant D: cqa surge/sway, ZERO yaw
        tau_lost_D = tau_lost_A.copy()
        tau_lost_D[:, 2] = 0.0

        # Brucon truth
        seed_dir = live_cell.WORK_ROOT / f"{tag}_seed{seed:04d}"
        main_p = next((p for p in seed_dir.glob("*.out") if "estimator" not in p.name), None)
        M = _load_tsv(main_p)
        win_m = ((M["t"] >= live_cell.WIN_START) & (M["t"] <= live_cell.WIN_END))
        sd_pre = float(M["SurgeDev"][win_m].mean())
        wd_pre = float(M["SwayDev"][win_m].mean())
        R_lf_wcf = np.hypot(M["SurgeDev"] - sd_pre, M["SwayDev"] - wd_pre)
        post = (M["t"] > live_cell.T_EVAL + 5.0) & (M["t"] <= live_cell.T_EVAL + 120.0)
        wcf_peak_truth = float(np.max(R_lf_wcf[post]))

        rng_A = np.random.default_rng(seed)  # per-seed deterministic MC
        rng_B = np.random.default_rng(seed)
        rng_C = np.random.default_rng(seed)
        rng_D = np.random.default_rng(seed)
        R_A, t_A, p50_A, p95_A = run_variant(tau_lost_A, aug, t_grid, eta_hat_lf, b_hat, K_lift, sigma_post, rng_A)
        R_B, t_B, p50_B, p95_B = run_variant(tau_lost_B, aug, t_grid, eta_hat_lf, b_hat, K_lift, sigma_post, rng_B)
        R_C, t_C, p50_C, p95_C = run_variant(tau_lost_C, aug, t_grid, eta_hat_lf, b_hat, K_lift, sigma_post, rng_C)
        R_D, t_D, p50_D, p95_D = run_variant(tau_lost_D, aug, t_grid, eta_hat_lf, b_hat, K_lift, sigma_post, rng_D)

        rows.append(dict(
            seed=seed,
            truth=wcf_peak_truth,
            A_R=R_A, A_p50=p50_A, A_p95=p95_A,
            B_R=R_B, B_p50=p50_B, B_p95=p95_B,
            C_R=R_C, C_p50=p50_C, C_p95=p95_C,
            D_R=R_D, D_p50=p50_D, D_p95=p95_D,
        ))

    truth = np.array([r["truth"] for r in rows])
    n = len(rows)
    print(f"\nn = {n} seeds")
    print(f"\n{'metric':<12} {'baseline A':>11} {'yaw-fix B':>11} {'fully bru C':>13} {'zero-yaw D':>13} {'brucon truth':>14}")
    print("-" * 80)
    def stats(key_R, key_p50, key_p95):
        Rs = np.array([r[key_R] for r in rows])
        p50s = np.array([r[key_p50] for r in rows])
        p95s = np.array([r[key_p95] for r in rows])
        return Rs.mean(), p50s.mean(), p95s.mean()
    A_R, A_p50, A_p95 = stats("A_R", "A_p50", "A_p95")
    B_R, B_p50, B_p95 = stats("B_R", "B_p50", "B_p95")
    C_R, C_p50, C_p95 = stats("C_R", "C_p50", "C_p95")
    D_R, D_p50, D_p95 = stats("D_R", "D_p50", "D_p95")
    truth_p50 = float(np.quantile(truth, 0.50))
    truth_p95 = float(np.quantile(truth, 0.95))
    print(f"{'R_det (mean)':<12} {A_R:>11.3f} {B_R:>11.3f} {C_R:>13.3f} {D_R:>13.3f} {'-':>14}")
    print(f"{'P50 (mean)':<12} {A_p50:>11.3f} {B_p50:>11.3f} {C_p50:>13.3f} {D_p50:>13.3f} {truth_p50:>14.3f}")
    print(f"{'P95 (mean)':<12} {A_p95:>11.3f} {B_p95:>11.3f} {C_p95:>13.3f} {D_p95:>13.3f} {truth_p95:>14.3f}")
    print()
    print(f"P95 bias vs truth:")
    print(f"  baseline A     (cqa  S/W/Y):  {100*(A_p95 - truth_p95)/truth_p95:+.1f}%")
    print(f"  yaw-fix  B     (cqa  S/W,  bru Y):  {100*(B_p95 - truth_p95)/truth_p95:+.1f}%")
    print(f"  fully bru C    (bru  S/W/Y):  {100*(C_p95 - truth_p95)/truth_p95:+.1f}%")
    print(f"  zero-yaw D     (cqa  S/W,  zero Y): {100*(D_p95 - truth_p95)/truth_p95:+.1f}%")

    # Decomposition: how much of the predicted P95 is R_det vs sigma?
    print()
    print(f"P95 decomposition (mean across seeds):")
    print(f"  variant A: R_det={A_R:.2f} + sigma_contribs={A_p95-A_R:.2f} = P95={A_p95:.2f}")
    print(f"  variant B: R_det={B_R:.2f} + sigma_contribs={B_p95-B_R:.2f} = P95={B_p95:.2f}")
    print(f"  variant C: R_det={C_R:.2f} + sigma_contribs={C_p95-C_R:.2f} = P95={C_p95:.2f}")
    print(f"  variant D: R_det={D_R:.2f} + sigma_contribs={D_p95-D_R:.2f} = P95={D_p95:.2f}")
    print(f"  brucon truth peak (per seed): P50={truth_p50:.2f}, P95={truth_p95:.2f}")
    print()
    print("Per-seed table (first 10):")
    print(f"{'seed':>5} {'truth':>7} {'A_Rdet':>7} {'A_P95':>7} {'B_Rdet':>7} {'B_P95':>7} {'C_Rdet':>7} {'C_P95':>7} {'D_Rdet':>7} {'D_P95':>7}")
    for r in rows[:10]:
        print(f"{r['seed']:>5} {r['truth']:>7.3f} {r['A_R']:>7.3f} {r['A_p95']:>7.3f} "
              f"{r['B_R']:>7.3f} {r['B_p95']:>7.3f} {r['C_R']:>7.3f} {r['C_p95']:>7.3f} "
              f"{r['D_R']:>7.3f} {r['D_p95']:>7.3f}")


if __name__ == "__main__":
    main()
