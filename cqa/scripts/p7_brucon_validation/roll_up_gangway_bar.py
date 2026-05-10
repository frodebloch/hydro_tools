"""Brucon roll-up of the gangway-telescope bar in the live operator panel.

|dL| comparator (mirrors the WCF position-bar comparator)
---------------------------------------------------------
After an initial signed-dL prototype showed structural under-
prediction in bf6/bf8, the live operator panel was reverted to the
|dL| metric (folded-normal halo about the deterministic |dL| peak
under WCFDI) with a single worst-margin red threshold
``min(L_max - L0, L0 - L_min)``. This script validates that
|dL| metric against brucon truth in the same apples-to-apples
fashion as ``roll_up_live_operator_panel.py`` does for the WCF
radial position bar:

    pred  : panel ``gangway_dL_p50`` / ``gangway_dL_p95`` (folded-
            normal halo about the deterministic |dL| peak instant
            under the LF + WF + b_hat-radial sigma envelope, all
            referenced to L0).

    truth : per seed, ``s_max_abs = max |dL_truth(t) - dL_pre|``
            over the post-WCF window, where
            ``dL_truth(t) = c3 . eta_full(t)`` and ``eta_full =
            (SurgeDev + xHf, SwayDev + yHf, HeadingDev + headingHf)``
            and ``dL_pre`` is the pre-WCF window mean of
            ``dL_truth``. Quantiles **across seeds** of single-
            realisation s_max_abs give the truth P50 / P95.

The pred is similarly demeaned via
``pred_p* - gangway_dL_intact_offset`` so both pred and truth
report the *change* in |dL| relative to the pre-WCF mean position
(apples-to-apples; cancels any non-zero LF offset that the live
state happens to carry).

Note: ``gangway_dL_*`` quantiles are non-negative magnitudes and so
is the deterministic offset, but the demean is a magnitude-of-
magnitudes subtraction so the result can in principle be negative
when the WCF peak is *smaller* than the live offset (rare; would
indicate the WCFDI transient pulls back toward L0 rather than
away). We clip to zero in the print only; the raw value is kept
for the bias.

Apples-to-apples scope (3-DOF horizontal projection only)
--------------------------------------------------------
We do NOT have brucon roll/pitch/heave WF posteriors plumbed into
``LiveSigmaPosterior`` yet, so this validation deliberately runs
the operator panel in its **horizontal-only fallback**
(``gangway_wf_coverage = "horizontal_3dof"``) and compares against
a brucon "truth" trajectory built from the **full horizontal
motion** (LF + WF) projected through the same 3-DOF telescope
sensitivity. Roll/pitch/heave contributions to dL remain
unvalidated and are deliberately omitted from both pred and truth
so the comparison is apples-to-apples. The forthcoming roll/pitch
posterior work (separate session) will re-run this validation in
full 6-DOF mode against a truth that includes brucon's Roll/Pitch/
Heave channels (which DO exist in the log, just not yet wired into
the posterior).

Joint geometry (held fixed across all 12 cells)
----------------------------------------------
Forward-pointing horizontal gangway, mid-stroke length::

    h        = 15.0  m  (rotation centre 15 m above the gangway base)
    alpha_g  = 0.0   rad
    beta_g   = 0.0   rad
    L0       = 25.0  m  (stroke = min(7, 7) = 7 m)

This is a "hypothetical landing setpoint" -- the brucon scenarios
are sea-trial DP runs without an actual W2W landing target, so the
joint state was chosen to expose the in-plane variance (forward
gangway -> surge dominates dL for head/quartering seas, contributes
significantly via yaw lever-arm c3[2] = -9 m/rad for all headings).
A single fixed geometry across cells gives a directly-comparable
bias matrix.

Run with::

    PYTHONPATH=. .venv/bin/python \\
        scripts/p7_brucon_validation/roll_up_gangway_bar.py

The expected residual is the same comparator-statistic effect
diagnosed for the position bars (single-realisation max over a
post-WCF window of correlated noise vs panel quantile at one
deterministic peak instant): ~10-20 % under-prediction in
magnitude.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))
sys.path.insert(0, str(THIS))

import live_cell_per_seed_pwq30 as live_cell                     # noqa: E402
from live_cell_per_seed_pwq30 import (                            # noqa: E402
    _load_tsv,
    load_seed,
    build_live_sigma_posterior,
)

from cqa.config import csov_default_config                       # noqa: E402
from cqa.live_decision import LiveObserverState                  # noqa: E402
from cqa.live_operator_view import summarise_for_operator_live   # noqa: E402
from cqa.gangway import GangwayJointState, telescope_sensitivity # noqa: E402


CELLS = [
    "bf4_c1_h0", "bf4_c1_q10",
    "bf6_h0",    "bf6_q10",    "bf6_h0_w45", "bf6_q10_w45",
    "bf8_h0",    "bf8_q10",    "bf8_h0_w45", "bf8_q10_w45",
    "pwo",       "pwq30",
]
T_WCF = 560.0
SEED_LO, SEED_HI = 1000, 1030

# Fixed joint geometry (see module docstring).
JOINT = GangwayJointState(h=15.0, alpha_g=0.0, beta_g=0.0, L=25.0)


def _set_cell(tag: str) -> None:
    live_cell.TAG = tag
    live_cell.T_WCF = T_WCF
    live_cell.T_EVAL = T_WCF - 5.0
    live_cell.WIN_END = T_WCF - 1.0
    live_cell.WIN_START = live_cell.WIN_END - live_cell.WIN_S
    live_cell.SEEDS = list(range(SEED_LO, SEED_HI))
    live_cell.CALIB_NPZ = THIS / f"scenario_{tag}_calibration.npz"


def _validate_cell(tag: str, cfg, c3: np.ndarray) -> dict | None:
    _set_cell(tag)
    calib_npz = live_cell.CALIB_NPZ
    if not calib_npz.exists():
        return None
    sigma_R_b_hat_m = float(np.load(calib_npz, allow_pickle=True)["sigma_R_b_hat_m"])

    rows = []
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
        s = summarise_for_operator_live(cfg, obs, sigma_post, joint=JOINT)
        # Sanity: we expect horizontal-only coverage everywhere here
        # (no roll/pitch/heave posteriors plumbed yet).
        assert s.gangway_wf_coverage == "horizontal_3dof"

        # Reload the brucon main log to project full-motion (LF + WF)
        # onto the telescope axis.
        seed_dir = live_cell.WORK_ROOT / f"{tag}_seed{seed:04d}"
        main_p = next((p for p in seed_dir.glob("*.out")
                       if "estimator" not in p.name), None)
        M = _load_tsv(main_p)
        t_main = M["t"]

        # Truth dL(t) = c3 . eta_full(t) where eta_full = LF + WF
        # horizontal channels only (matches the panel's coverage).
        # HeadingDev / headingHf are exported by brucon in DEGREES
        # (see live_cell_per_seed_pwq30.load_seed for the full unit
        # discussion); convert to radians before projecting through
        # the c3[2] = -9 m/rad lever arm.
        eta_full = np.column_stack([
            M["SurgeDev"] + M["xHf"],
            M["SwayDev"] + M["yHf"],
            np.deg2rad(M["HeadingDev"] + M["headingHf"]),
        ])
        dL_truth = eta_full @ c3

        win_m = (t_main >= live_cell.WIN_START) & (t_main <= live_cell.WIN_END)
        post_mask = ((t_main > live_cell.T_EVAL + 5.0)
                     & (t_main <= live_cell.T_EVAL + 120.0))
        dL_pre = float(dL_truth[win_m].mean())
        # |dL| max-over-window of vector-demeaned dL_truth; analogous
        # to the WCF position-bar truth statistic
        # max |hypot(SurgeDev-sd_pre, SwayDev-wd_pre)| over post-WCF.
        s_max_abs = float(np.max(np.abs(dL_truth[post_mask] - dL_pre)))

        # Panel pred is referenced to L0 (an absolute magnitude). To
        # get apples-to-apples deltas-around-pre, subtract the live
        # "now" |dL| offset from the predicted P50/P95.
        pred_p50 = s.gangway_dL_p50 - s.gangway_dL_intact_offset
        pred_p95 = s.gangway_dL_p95 - s.gangway_dL_intact_offset
        pred_det = (s.gangway_dL_wcf_offset_at_peak
                    - s.gangway_dL_intact_offset)

        rows.append(dict(
            seed=seed,
            pred_p50=pred_p50,
            pred_p95=pred_p95,
            pred_det=pred_det,
            sigma_dL_wcf=s.gangway_sigma_dL_wcf_m,
            t_peak=s.gangway_t_peak_s,
            traffic=s.gangway_traffic,
            s_max_abs=s_max_abs,
        ))
    if not rows:
        return None

    p50_pred = np.array([r["pred_p50"] for r in rows])
    p95_pred = np.array([r["pred_p95"] for r in rows])
    s_abs = np.array([r["s_max_abs"] for r in rows])

    # Truth quantiles across the seed ensemble: distribution of
    # single-realisation max-|dL| post-WCF excursion.
    truth_p50 = float(np.quantile(s_abs, 0.50))
    truth_p95 = float(np.quantile(s_abs, 0.95))

    pred_p50_mean = float(p50_pred.mean())
    pred_p95_mean = float(p95_pred.mean())

    # Coverage of truth |dL| by the predicted P95 envelope, per seed.
    cov_p95 = float(np.mean(s_abs <= p95_pred))

    return dict(
        tag=tag,
        n_seeds=len(rows),
        pred_p50_mean=pred_p50_mean,
        pred_p95_mean=pred_p95_mean,
        truth_p50=truth_p50,
        truth_p95=truth_p95,
        # Bias on the matched magnitude quantiles. Use a denominator
        # floor (0.05 m, ~5 cm) to keep near-zero cells from blowing
        # up the percent.
        bias_p50_pct=100 * (pred_p50_mean - truth_p50)
                          / max(truth_p50, 0.05),
        bias_p95_pct=100 * (pred_p95_mean - truth_p95)
                          / max(truth_p95, 0.05),
        cov_p95=cov_p95,
        sigma_dL_wcf_mean=float(np.mean([r["sigma_dL_wcf"] for r in rows])),
        n_red=sum(r["traffic"] == "red" for r in rows),
        n_amber=sum(r["traffic"] == "amber" for r in rows),
        n_green=sum(r["traffic"] == "green" for r in rows),
    )


def main() -> int:
    cfg = csov_default_config()
    c3 = telescope_sensitivity(JOINT, cfg.gangway)

    print(f"Joint:      h={JOINT.h:.1f} m, alpha={JOINT.alpha_g:.2f} rad, "
          f"beta={JOINT.beta_g:.2f} rad, L0={JOINT.L:.1f} m")
    print(f"c3:         {c3}  (m/m, m/m, m/rad)")
    print(f"Coverage:   horizontal_3dof "
          f"(roll/pitch/heave WF posteriors not yet wired)\n")

    results = []
    for tag in CELLS:
        m = _validate_cell(tag, cfg, c3)
        if m is None:
            print(f"  {tag}: skipped (no data)")
            continue
        results.append(m)
        print(f"  {tag}: validated {m['n_seeds']} seeds")
    print()

    print("Gangway-telescope bar vs brucon truth (3-DOF horizontal projection)")
    print("Pred is |dL|_p* relative to the live LF \"now\" |dL| offset.")
    print("Truth is per-seed max |dL_truth - dL_pre| over post-WCF window,")
    print("then quantiles across seeds.\n")

    hdr = (f"{'cell':<14} {'N':>3}  "
           f"{'p50.pr':>7} {'p50.tr':>7} {'b50%':>6}  "
           f"{'p95.pr':>7} {'p95.tr':>7} {'b95%':>6} {'cov%':>5}  "
           f"{'sig_dL':>7}  {'g/a/r':>9}")
    print(hdr)
    print("-" * len(hdr))
    for m in results:
        print(f"{m['tag']:<14} {m['n_seeds']:>3}  "
              f"{m['pred_p50_mean']:>7.3f} {m['truth_p50']:>7.3f} "
              f"{m['bias_p50_pct']:>+5.0f}%  "
              f"{m['pred_p95_mean']:>7.3f} {m['truth_p95']:>7.3f} "
              f"{m['bias_p95_pct']:>+5.0f}% {100*m['cov_p95']:>4.0f}%  "
              f"{m['sigma_dL_wcf_mean']:>7.3f}  "
              f"{m['n_green']}/{m['n_amber']}/{m['n_red']:<5}")

    # ---- Plot ----
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.0))
    tags = [m["tag"] for m in results]
    x = np.arange(len(tags))
    width = 0.20

    ax.bar(x - 1.5 * width, [m["pred_p50_mean"] for m in results],
           width=width, color="#1f77b4", alpha=0.85, label="pred P50")
    ax.bar(x - 0.5 * width, [m["truth_p50"] for m in results],
           width=width, color="#1f77b4", alpha=0.45, edgecolor="#1f77b4",
           hatch="//", label="brucon P50")
    ax.bar(x + 0.5 * width, [m["pred_p95_mean"] for m in results],
           width=width, color="#d62728", alpha=0.85, label="pred P95")
    ax.bar(x + 1.5 * width, [m["truth_p95"] for m in results],
           width=width, color="#d62728", alpha=0.45, edgecolor="#d62728",
           hatch="//", label="brucon P95")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|dL| relative to pre-WCF mean [m]")
    ax.set_title(
        f"Gangway-telescope bar  -  12-cell brucon roll-up  "
        f"|dL| pred vs truth\n"
        f"(joint forward, h=15 m, L0=25 m; horizontal-3DOF only)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = THIS / "roll_up_gangway_bar.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
