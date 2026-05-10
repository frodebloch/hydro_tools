"""Roll up the live operator panel validation across all brucon cells.

Runs ``validate_live_operator_panel.py --tag <cell>`` for every cell
that has a calibration npz and brucon work-dirs, captures the key
quality metrics (intact P95 coverage, |err|/truth on intact P95, WCF
P95 coverage, WCF P95 bias vs realised LF peak), and prints a summary
table that lets us compare across sea states.

Run with::

    PYTHONPATH=. .venv/bin/python \\
        scripts/p7_brucon_validation/roll_up_live_operator_panel.py
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


CELLS = [
    "bf4_c1_h0", "bf4_c1_q10",
    "bf6_h0",    "bf6_q10",    "bf6_h0_w45", "bf6_q10_w45",
    "bf8_h0",    "bf8_q10",    "bf8_h0_w45", "bf8_q10_w45",
    "pwo",       "pwq30",
]
T_WCF = 560.0
SEED_LO, SEED_HI = 1000, 1030


def _set_cell(tag: str) -> None:
    live_cell.TAG = tag
    live_cell.T_WCF = T_WCF
    live_cell.T_EVAL = T_WCF - 5.0
    live_cell.WIN_END = T_WCF - 1.0
    live_cell.WIN_START = live_cell.WIN_END - live_cell.WIN_S
    live_cell.SEEDS = list(range(SEED_LO, SEED_HI))
    live_cell.CALIB_NPZ = THIS / f"scenario_{tag}_calibration.npz"


def _validate_cell(tag: str) -> dict | None:
    """Run validation for one cell. Returns a metrics dict, or None
    if the cell can't be loaded."""
    _set_cell(tag)
    calib_npz = live_cell.CALIB_NPZ
    if not calib_npz.exists():
        return None
    sigma_R_b_hat_m = float(np.load(calib_npz, allow_pickle=True)["sigma_R_b_hat_m"])

    cfg = csov_default_config()

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
        s = summarise_for_operator_live(cfg, obs, sigma_post)

        seed_dir = live_cell.WORK_ROOT / f"{tag}_seed{seed:04d}"
        main_p = next((p for p in seed_dir.glob("*.out")
                       if "estimator" not in p.name), None)
        M = _load_tsv(main_p)
        t_main = M["t"]
        win_m = (t_main >= live_cell.WIN_START) & (t_main <= live_cell.WIN_END)
        # Intact-axis truth: actual radial distance from setpoint (do NOT
        # demean). The operator panel reports |eta_hat_LF + nu|, i.e.
        # the actual offset+noise distance from the DP setpoint, so the
        # apples-to-apples truth is |SurgeDev, SwayDev| directly --
        # SurgeDev/SwayDev are already setpoint-relative LF deviations
        # in body frame.
        # WCF-axis truth still uses the demeaned LF radius because the
        # WCF transient is measured *relative to the pre-WCF mean
        # position* (the deterministic part of the WCF prediction is
        # eta_hat + delta_eta_mean(t), and we already include eta_hat
        # in the pred -- the realised peak is the post-WCF excursion
        # *from the same eta_hat reference*).
        sd_pre = float(M["SurgeDev"][win_m].mean())
        wd_pre = float(M["SwayDev"][win_m].mean())
        R_lf_intact_truth = np.hypot(M["SurgeDev"], M["SwayDev"])
        R_lf_wcf = np.hypot(M["SurgeDev"] - sd_pre, M["SwayDev"] - wd_pre)
        post_mask = ((t_main > live_cell.T_EVAL + 5.0)
                     & (t_main <= live_cell.T_EVAL + 120.0))
        intact_R = R_lf_intact_truth[win_m]
        post_R = R_lf_wcf[post_mask]

        rows.append(dict(
            intact_p95_pred=s.intact_R_p95,
            intact_p95_truth=float(np.quantile(intact_R, 0.95)),
            intact_p50_pred=s.intact_R_p50,
            intact_p50_truth=float(np.quantile(intact_R, 0.50)),
            wcf_p95_pred=s.wcf_R_p95,
            wcf_p50_pred=s.wcf_R_p50,
            wcf_peak_truth=float(np.max(post_R)),
            intact_traffic=s.intact_traffic,
            wcf_traffic=s.wcf_traffic,
            overall=s.overall_traffic,
        ))
    if not rows:
        return None

    int_p95_pred = np.array([r["intact_p95_pred"] for r in rows])
    int_p95_tr = np.array([r["intact_p95_truth"] for r in rows])
    int_p50_pred = np.array([r["intact_p50_pred"] for r in rows])
    int_p50_tr = np.array([r["intact_p50_truth"] for r in rows])
    wcf_p95_pred = np.array([r["wcf_p95_pred"] for r in rows])
    wcf_p50_pred = np.array([r["wcf_p50_pred"] for r in rows])
    wcf_truth = np.array([r["wcf_peak_truth"] for r in rows])

    return dict(
        tag=tag,
        n_seeds=len(rows),
        intact_p95_pred_mean=int_p95_pred.mean(),
        intact_p95_truth_mean=int_p95_tr.mean(),
        intact_p95_cov=float(np.mean(int_p95_tr <= int_p95_pred)),
        intact_p95_err_pct=100 * np.mean(np.abs(int_p95_pred - int_p95_tr) / int_p95_tr),
        # Signed bias: positive = pred over truth (conservative), negative = under truth.
        intact_p95_bias_pct=100 * (int_p95_pred - int_p95_tr).mean() / int_p95_tr.mean(),
        intact_p50_bias=float((int_p50_pred - int_p50_tr).mean()),
        intact_p50_bias_pct=100 * (int_p50_pred - int_p50_tr).mean() / int_p50_tr.mean(),
        wcf_p95_pred_mean=wcf_p95_pred.mean(),
        wcf_truth_mean=wcf_truth.mean(),
        wcf_truth_p95=float(np.quantile(wcf_truth, 0.95)),
        wcf_p95_cov=float(np.mean(wcf_truth <= wcf_p95_pred)),
        wcf_p50_bias_pct=100 * (wcf_p50_pred - wcf_truth).mean() / wcf_truth.mean(),
        # Composition of per-seed traffic-light verdicts (how often green/amber/red).
        n_red=sum(r["overall"] == "red" for r in rows),
        n_amber=sum(r["overall"] == "amber" for r in rows),
        n_green=sum(r["overall"] == "green" for r in rows),
    )


def main() -> int:
    results = []
    for tag in CELLS:
        m = _validate_cell(tag)
        if m is None:
            print(f"  {tag}: skipped (no data)")
            continue
        results.append(m)
        print(f"  {tag}: validated {m['n_seeds']} seeds")
    print()

    # ---- Table ----
    print("Live operator panel vs brucon LF truth -- 12-cell roll-up\n")
    hdr = (f"{'cell':<14} {'N':>3}   "
           f"{'intP95.pr':>9} {'intP95.tr':>9} {'cov%':>5} {'|err|%':>6} {'bias%':>6} {'P50bs%':>6}   "
           f"{'wcfP95.pr':>9} {'wcfPK.tr':>9} {'wcfPK.P95':>9} {'cov%':>5} {'P50bias%':>9}   "
           f"{'g/a/r':>9}")
    print(hdr)
    print("-" * len(hdr))
    for m in results:
        print(f"{m['tag']:<14} {m['n_seeds']:>3}   "
              f"{m['intact_p95_pred_mean']:>9.3f} {m['intact_p95_truth_mean']:>9.3f} "
              f"{100*m['intact_p95_cov']:>4.0f}% {m['intact_p95_err_pct']:>5.0f}% "
              f"{m['intact_p95_bias_pct']:>+5.0f}% {m['intact_p50_bias_pct']:>+5.0f}%   "
              f"{m['wcf_p95_pred_mean']:>9.3f} {m['wcf_truth_mean']:>9.3f} "
              f"{m['wcf_truth_p95']:>9.3f} {100*m['wcf_p95_cov']:>4.0f}% "
              f"{m['wcf_p50_bias_pct']:>+8.1f}%   "
              f"{m['n_green']}/{m['n_amber']}/{m['n_red']:<5}")

    # ---- Plot: intact and WCF coverage across cells ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tags = [m["tag"] for m in results]
    x = np.arange(len(tags))

    ax = axes[0]
    ax.bar(x - 0.20, [m["intact_p95_pred_mean"] for m in results],
           width=0.4, color="#1f77b4", alpha=0.8, label="pred P95")
    ax.bar(x + 0.20, [m["intact_p95_truth_mean"] for m in results],
           width=0.4, color="#444444", alpha=0.8, label="brucon P95")
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.axhline(2.0, color="#ff9900", ls="--", lw=1, label="IMCA amber 2 m")
    ax.axhline(4.0, color="#d62728", ls="--", lw=1, label="IMCA red 4 m")
    ax.set_ylabel("intact P95 of |R_LF| [m]")
    ax.set_title("INTACT axis  -  pred vs brucon LF P95 across cells")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - 0.20, [m["wcf_p95_pred_mean"] for m in results],
           width=0.4, color="#1f77b4", alpha=0.8, label="pred P95 (mean)")
    ax.bar(x + 0.20, [m["wcf_truth_p95"] for m in results],
           width=0.4, color="#444444", alpha=0.8, label="brucon LF peak P95")
    ax.plot(x, [m["wcf_truth_mean"] for m in results], "o",
            color="#888888", label="brucon LF peak mean")
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.axhline(2.0, color="#ff9900", ls="--", lw=1, label="IMCA amber 2 m")
    ax.axhline(4.0, color="#d62728", ls="--", lw=1, label="IMCA red 4 m")
    ax.set_ylabel("WCF post-fault peak [m]")
    ax.set_title("WCF axis  -  pred vs brucon LF peak across cells")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Live operator panel  -  12-cell brucon roll-up", fontsize=12)
    fig.tight_layout()
    out = THIS / "roll_up_live_operator_panel.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
