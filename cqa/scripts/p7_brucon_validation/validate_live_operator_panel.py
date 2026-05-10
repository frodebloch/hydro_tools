"""Validate the LIVE operator-panel summary against brucon truth.

Cell-agnostic: pass ``--tag <cell_tag>`` to run on any cell that has a
calibration artefact (``scenario_<tag>_calibration.npz``) and brucon
work-dirs (``work/<tag>_seed*/``). All 12 cells in the brucon
validation matrix share T_WCF = 560 s (see run_validation_matrix.py).

Apples-to-apples comparison
---------------------------
The live operator summary predicts the LF body-frame radial deviation
|eta_hat_LF + delta_eta_mean(t) + nu|, where:
  - eta_hat_LF, delta_eta_mean(t) are LF (no waves)
  - nu is small-time noise (LF + WF + b_hat sigmas in quadrature on
    the WCF axis; LF only on the intact axis)
The "WF" component of nu is the WAVE FREQUENCY noise on the LF observer
estimate, NOT the wave-induced position swing of the vessel.

So the brucon comparator MUST be the LF body-frame truth from the
brucon DP estimator (SurgeDev, SwayDev), NOT the raw NED position
(which contains the full ~1 m WF swing on top).

This script computes:
  - LF truth   : R_LF(t)    = hypot(SurgeDev(t)  - SurgeDev_pre_mean,
                                     SwayDev(t)   - SwayDev_pre_mean)
  - TOTAL truth: R_tot(t)   = hypot(x_body(t),  y_body(t))   (NED rotated to body)
  - intact axis: P50/P95 of R_LF in the pre-WCF window
  - WCF axis  : max_t R_LF over the post-WCF window  (per seed)

Both LF and total are reported so we can see how much of any apparent
gap is "predictor under-predicts LF" vs "predictor doesn't include WF
position swing" (the latter is by design -- LF/WF separation is the
whole point of the observer).

Run with::

    PYTHONPATH=. .venv/bin/python \\
        scripts/p7_brucon_validation/validate_live_operator_panel.py --tag pwq30
    PYTHONPATH=. .venv/bin/python \\
        scripts/p7_brucon_validation/validate_live_operator_panel.py --tag bf6_h0
    ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))
sys.path.insert(0, str(THIS))

# Import the per-seed helpers from the pwq30 module. They are cell-
# agnostic apart from a few module-level globals that we override below
# via the same ``_apply_args`` path the main pwq30 script uses.
import live_cell_per_seed_pwq30 as live_cell                     # noqa: E402
from live_cell_per_seed_pwq30 import (                            # noqa: E402
    _load_tsv,
    load_seed,
    build_live_sigma_posterior,
)

from cqa.config import csov_default_config                       # noqa: E402
from cqa.live_decision import LiveObserverState                  # noqa: E402
from cqa.live_operator_view import summarise_for_operator_live   # noqa: E402


PRE_WCF_T_LO = -30.0   # rel. T_EVAL
PRE_WCF_T_HI = 5.0     # rel. T_EVAL  (T_WCF is at +5 s)
POST_WCF_T_LO = 5.0
POST_WCF_T_HI = 120.0


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tag", default="pwq30",
                   help="Cell tag (work-dir prefix and calibration npz "
                        "stem). Default: pwq30.")
    p.add_argument("--t-wcf", type=float, default=560.0,
                   help="WCF injection time in seconds (default: 560.0).")
    p.add_argument("--seeds", default="1000-1030",
                   help="Seed range as 'lo-hi' (Python-style half-open) "
                        "(default: 1000-1030).")
    return p.parse_args()


def _apply_args(args) -> Path:
    """Override globals on live_cell_per_seed_pwq30 so its load_seed /
    build_live_sigma_posterior pick up the requested cell. Returns the
    path to the calibration npz."""
    live_cell.TAG = args.tag
    live_cell.T_WCF = float(args.t_wcf)
    live_cell.T_EVAL = live_cell.T_WCF - 5.0
    live_cell.WIN_END = live_cell.T_WCF - 1.0
    live_cell.WIN_START = live_cell.WIN_END - live_cell.WIN_S
    lo, hi = (int(s) for s in args.seeds.split("-"))
    live_cell.SEEDS = list(range(lo, hi))
    calib = THIS / f"scenario_{args.tag}_calibration.npz"
    live_cell.CALIB_NPZ = calib
    return calib


def main() -> int:
    args = _parse_args()
    calib_npz = _apply_args(args)
    if not calib_npz.exists():
        print(f"Missing calibration: {calib_npz}", file=sys.stderr)
        return 1
    sigma_R_b_hat_m = float(np.load(calib_npz, allow_pickle=True)["sigma_R_b_hat_m"])

    cfg = csov_default_config()

    rows = []  # one per seed
    for seed in live_cell.SEEDS:
        d = load_seed(seed)
        if d is None:
            continue

        sigma_post = build_live_sigma_posterior(d, sigma_R_b_hat_m=sigma_R_b_hat_m)
        obs = LiveObserverState(
            eta_hat=d["eta_hat"], nu_hat=d["nu_hat"], b_hat=d["b_hat"],
            eta_wave=d["eta_wave"],
            heading_compass=float(d.get("heading_compass", 0.0)),
        )
        s = summarise_for_operator_live(cfg, obs, sigma_post)

        # Re-load main .out to get LF truth (SurgeDev / SwayDev).
        seed_dir = live_cell.WORK_ROOT / f"{live_cell.TAG}_seed{seed:04d}"
        main_p = next((p for p in seed_dir.glob("*.out")
                       if "estimator" not in p.name), None)
        M = _load_tsv(main_p)
        t_main = M["t"]
        win_m = (t_main >= live_cell.WIN_START) & (t_main <= live_cell.WIN_END)
        sd_pre_mean = float(M["SurgeDev"][win_m].mean())
        wd_pre_mean = float(M["SwayDev"][win_m].mean())
        surge_lf = M["SurgeDev"] - sd_pre_mean
        sway_lf = M["SwayDev"] - wd_pre_mean
        R_lf = np.hypot(surge_lf, sway_lf)

        # Pre-WCF window (intact): 60 s pre-WCF.
        intact_mask = win_m
        # Post-WCF window: t > T_WCF.
        post_mask = ((t_main > live_cell.T_EVAL + 5.0)
                     & (t_main <= live_cell.T_EVAL + 120.0))

        intact_R_lf = R_lf[intact_mask]
        post_R_lf = R_lf[post_mask]

        # Total (LF + WF) truth from raw NED, for context only.
        post_R_tot = d["truth_R"][(d["t_plot_grid"] >= POST_WCF_T_LO)
                                  & (d["t_plot_grid"] <= POST_WCF_T_HI)]

        intact_p50_truth = float(np.quantile(intact_R_lf, 0.50))
        intact_p95_truth = float(np.quantile(intact_R_lf, 0.95))
        wcf_peak_truth_lf = float(np.max(post_R_lf))
        wcf_peak_truth_tot = float(np.max(post_R_tot))

        rows.append(dict(
            seed=seed,
            intact_p50_pred=s.intact_R_p50, intact_p95_pred=s.intact_R_p95,
            intact_p50_truth=intact_p50_truth, intact_p95_truth=intact_p95_truth,
            wcf_p50_pred=s.wcf_R_p50, wcf_p95_pred=s.wcf_R_p95,
            wcf_peak_truth_lf=wcf_peak_truth_lf,
            wcf_peak_truth_tot=wcf_peak_truth_tot,
            wcf_t_peak_pred=s.wcf_t_peak_s,
        ))

    if not rows:
        print("No seeds loaded.", file=sys.stderr)
        return 1

    seeds = np.array([r["seed"] for r in rows])
    int_p50_pred = np.array([r["intact_p50_pred"] for r in rows])
    int_p95_pred = np.array([r["intact_p95_pred"] for r in rows])
    int_p50_tr = np.array([r["intact_p50_truth"] for r in rows])
    int_p95_tr = np.array([r["intact_p95_truth"] for r in rows])
    wcf_p50_pred = np.array([r["wcf_p50_pred"] for r in rows])
    wcf_p95_pred = np.array([r["wcf_p95_pred"] for r in rows])
    wcf_truth_lf = np.array([r["wcf_peak_truth_lf"] for r in rows])
    wcf_truth_tot = np.array([r["wcf_peak_truth_tot"] for r in rows])

    # ------------- intact: per-seed pred vs per-seed LF truth -------------
    print("\n=== INTACT AXIS  (predicted LF |R| vs brucon LF |R| from SurgeDev/SwayDev) ===")
    print(f"  N seeds = {len(rows)}")
    print(f"  P50:  pred mean = {int_p50_pred.mean():.3f} m   "
          f"truth mean = {int_p50_tr.mean():.3f} m   "
          f"bias = {(int_p50_pred - int_p50_tr).mean():+.3f} m   "
          f"|err|/truth = {np.mean(np.abs(int_p50_pred - int_p50_tr) / int_p50_tr):.1%}")
    print(f"  P95:  pred mean = {int_p95_pred.mean():.3f} m   "
          f"truth mean = {int_p95_tr.mean():.3f} m   "
          f"bias = {(int_p95_pred - int_p95_tr).mean():+.3f} m   "
          f"|err|/truth = {np.mean(np.abs(int_p95_pred - int_p95_tr) / int_p95_tr):.1%}")
    cov95 = float(np.mean(int_p95_tr <= int_p95_pred))
    print(f"  Coverage of P95: brucon LF P95 <= pred P95 in "
          f"{cov95:.0%} of seeds  (target: ~95%)")

    # ------------- WCF: per-seed pred vs realised LF peak -------------
    print("\n=== WCF AXIS  (predicted LF peak vs brucon LF peak from SurgeDev/SwayDev) ===")
    print(f"  Realised post-WCF LF peak ensemble (N={len(rows)}):")
    print(f"    mean = {wcf_truth_lf.mean():.3f} m   "
          f"std = {wcf_truth_lf.std(ddof=1):.3f} m   "
          f"P50 = {np.quantile(wcf_truth_lf, 0.50):.3f} m   "
          f"P95 = {np.quantile(wcf_truth_lf, 0.95):.3f} m   "
          f"max = {wcf_truth_lf.max():.3f} m")
    print(f"  Per-seed live predictions (mean across seeds):")
    print(f"    P50 pred = {wcf_p50_pred.mean():.3f} m   "
          f"P95 pred = {wcf_p95_pred.mean():.3f} m")
    cov_wcf_p95 = float(np.mean(wcf_truth_lf <= wcf_p95_pred))
    cov_wcf_p50 = float(np.mean(wcf_truth_lf <= wcf_p50_pred))
    print(f"  Per-seed coverage: realised LF peak <= pred P95 in "
          f"{cov_wcf_p95:.0%} of seeds  (target: ~95%)")
    print(f"  Per-seed coverage: realised LF peak <= pred P50 in "
          f"{cov_wcf_p50:.0%} of seeds  (target: ~50%)")
    bias_wcf = (wcf_p50_pred - wcf_truth_lf).mean()
    print(f"  Bias of pred P50 vs realised LF peak: "
          f"{bias_wcf:+.3f} m  ({100 * bias_wcf / wcf_truth_lf.mean():+.1f}% of mean truth)")

    print("\n  --- For context: realised TOTAL peak (LF + WF, raw NED) ensemble:")
    print(f"    mean = {wcf_truth_tot.mean():.3f} m   "
          f"P50 = {np.quantile(wcf_truth_tot, 0.50):.3f} m   "
          f"P95 = {np.quantile(wcf_truth_tot, 0.95):.3f} m   "
          f"max = {wcf_truth_tot.max():.3f} m")
    print(f"    (Operator panel does NOT predict this -- WF position swing is")
    print(f"     filtered out by the LF observer by design.)")

    # ------------- Plot -------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    ax = axes[0]
    ax.plot(seeds, int_p50_tr, "o", color="#1f77b4", label="brucon P50 (intact)")
    ax.plot(seeds, int_p50_pred, "x", color="#1f77b4", label="pred  P50 (intact)")
    ax.plot(seeds, int_p95_tr, "o", color="#d62728", label="brucon P95 (intact)")
    ax.plot(seeds, int_p95_pred, "x", color="#d62728", label="pred  P95 (intact)")
    ax.set_xlabel("seed")
    ax.set_ylabel("radial distance [m]")
    ax.set_title("Intact axis: per-seed pred vs brucon empirical")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    order = np.argsort(wcf_truth_lf)
    ax.plot(np.arange(len(rows)), wcf_truth_lf[order], "o-",
            color="#444444", label="brucon LF realised peak (sorted)")
    ax.plot(np.arange(len(rows)), wcf_truth_tot[order], "s",
            color="#bbbbbb", label="brucon TOTAL realised peak (LF+WF)")
    ax.plot(np.arange(len(rows)), wcf_p50_pred[order], "x",
            color="#1f77b4", label="pred P50 (per-seed)")
    ax.plot(np.arange(len(rows)), wcf_p95_pred[order], "x",
            color="#d62728", label="pred P95 (per-seed)")
    ax.set_xlabel("seed (sorted by realised LF peak)")
    ax.set_ylabel("post-WCF radial peak [m]")
    ax.set_title("WCF axis: per-seed pred vs brucon realised peak")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Live operator panel vs brucon  -  {args.tag} "
                 f"({len(rows)} seeds)", fontsize=12)
    fig.tight_layout()
    out = THIS / f"validate_live_operator_panel_{args.tag}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
