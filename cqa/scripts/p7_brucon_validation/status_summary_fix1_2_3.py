"""Status summary for §12.21.8.1: Fix 1+2 (no Ki) vs Fix 1+2+3 (Ki on).

Reads two npz files produced by `calibrated_wcfdi_brucon_validation.py`
with environment variable `CQA_INTEGRATOR=0` and `CQA_INTEGRATOR=1`,
and produces a 4-panel status plot:

    [ Pooled CDF overlay     ] [ Per-quantile bar chart      ]
    [ Ensemble-mean Δsurge    ] [ Ensemble-mean Δsway          ]

Each npz contains:
  t_cqa, t_truth, truth_dsurge_mean, truth_dsway_mean,
  cal_dsurge_mean, cal_dsway_mean, truth_peaks_60, cal_pooled_peaks,
  raw_pos_peak_p50.

Usage
-----
    CQA_INTEGRATOR=0 CQA_STATUS_NPZ=/tmp/status_no_ki.npz \\
        python calibrated_wcfdi_brucon_validation.py
    CQA_INTEGRATOR=1 CQA_STATUS_NPZ=/tmp/status_ki_on.npz \\
        python calibrated_wcfdi_brucon_validation.py
    python status_summary_fix1_2_3.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent

NPZ_NO_KI = Path("/tmp/status_no_ki.npz")
NPZ_KI_ON = Path("/tmp/status_ki_on.npz")
OUT_PNG = THIS / "status_fix1_2_3.png"


def _load(path: Path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def main() -> None:
    if not NPZ_NO_KI.exists() or not NPZ_KI_ON.exists():
        raise SystemExit(
            f"Missing one of {NPZ_NO_KI} or {NPZ_KI_ON}. "
            f"Run calibrated_wcfdi_brucon_validation.py twice with "
            f"CQA_INTEGRATOR=0 and CQA_INTEGRATOR=1, both with "
            f"CQA_STATUS_NPZ set."
        )

    nki = _load(NPZ_NO_KI)
    ki = _load(NPZ_KI_ON)

    # truth arrays are identical between runs; pull from one
    t_truth = nki["t_truth"]
    truth_dsurge = nki["truth_dsurge_mean"]
    truth_dsway = nki["truth_dsway_mean"]
    truth_peaks_60 = nki["truth_peaks_60"]

    t_cqa = nki["t_cqa"]
    nki_dsu = nki["cal_dsurge_mean"]
    nki_dsw = nki["cal_dsway_mean"]
    nki_pool = nki["cal_pooled_peaks"]

    ki_dsu = ki["cal_dsurge_mean"]
    ki_dsw = ki["cal_dsway_mean"]
    ki_pool = ki["cal_pooled_peaks"]

    # ---- summary stats ----
    truth_p50 = float(np.median(truth_peaks_60))
    truth_p95 = float(np.quantile(truth_peaks_60, 0.95))
    truth_max = float(truth_peaks_60.max())
    nki_p50 = float(np.median(nki_pool))
    nki_p95 = float(np.quantile(nki_pool, 0.95))
    nki_p99 = float(np.quantile(nki_pool, 0.99))
    nki_max = float(nki_pool.max())
    ki_p50 = float(np.median(ki_pool))
    ki_p95 = float(np.quantile(ki_pool, 0.95))
    ki_p99 = float(np.quantile(ki_pool, 0.99))
    ki_max = float(ki_pool.max())

    print("--- pooled CDF stats ---")
    print(f"  truth (LF, N={len(truth_peaks_60)}): "
          f"P50={truth_p50:.2f} P95={truth_p95:.2f} max={truth_max:.2f} m")
    print(f"  Fix 1+2 (no Ki, n={len(nki_pool)}):   "
          f"P50={nki_p50:.2f} P95={nki_p95:.2f} P99={nki_p99:.2f} max={nki_max:.2f} m")
    print(f"  Fix 1+2+3 (Ki on, n={len(ki_pool)}):  "
          f"P50={ki_p50:.2f} P95={ki_p95:.2f} P99={ki_p99:.2f} max={ki_max:.2f} m")
    print(f"  P95/truth_P95: no-Ki={nki_p95/truth_p95:.2f}  "
          f"Ki-on={ki_p95/truth_p95:.2f}")

    # ---- plot ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "G2 status: Fix 1+2 (no Ki) → Fix 1+2+3 (Ki on)\n"
        "bow-quartering 30°, bus_port WCF, pwq30 ensemble (N=30 seeds)",
        fontsize=12,
    )

    # --- Panel 1: pooled CDF overlay ---
    ax = axes[0, 0]
    sorted_truth = np.sort(truth_peaks_60)
    cdf_truth = np.arange(1, len(sorted_truth) + 1) / len(sorted_truth)
    ax.plot(sorted_truth, cdf_truth, "-o", color="black", lw=1.6, ms=4,
            label=f"brucon truth LF (N={len(sorted_truth)})")

    sorted_nki = np.sort(nki_pool)
    cdf_nki = np.arange(1, len(sorted_nki) + 1) / len(sorted_nki)
    ax.plot(sorted_nki, cdf_nki, color="tab:orange", lw=2.0,
            label=f"cal Fix 1+2 (n={len(sorted_nki)})")

    sorted_ki = np.sort(ki_pool)
    cdf_ki = np.arange(1, len(sorted_ki) + 1) / len(sorted_ki)
    ax.plot(sorted_ki, cdf_ki, color="tab:blue", lw=2.0,
            label=f"cal Fix 1+2+3 (n={len(sorted_ki)})")

    ax.axhline(0.5, ls=":", color="grey", lw=0.8)
    ax.axhline(0.95, ls=":", color="grey", lw=0.8)
    ax.set_xlabel("|Δradial|_peak (CG, body, [0,60] s) [m]")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Per-realisation peak excursion CDF")
    ax.set_xlim(0, max(truth_max, ki_max) * 1.05)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # --- Panel 2: per-quantile bar chart ---
    ax = axes[0, 1]
    quantiles = ["P50", "P95", "P99", "max"]
    truth_vals = [truth_p50, truth_p95, np.nan, truth_max]  # truth has no P99 (N=30)
    nki_vals = [nki_p50, nki_p95, nki_p99, nki_max]
    ki_vals = [ki_p50, ki_p95, ki_p99, ki_max]

    x = np.arange(len(quantiles))
    width = 0.27
    truth_vals_plot = [v if np.isfinite(v) else 0.0 for v in truth_vals]
    bars0 = ax.bar(x - width, truth_vals_plot, width, color="black",
                   alpha=0.75, label="brucon truth (LF)")
    bars1 = ax.bar(x, nki_vals, width, color="tab:orange",
                   alpha=0.85, label="cal Fix 1+2")
    bars2 = ax.bar(x + width, ki_vals, width, color="tab:blue",
                   alpha=0.85, label="cal Fix 1+2+3")

    for bars, vals in [(bars0, truth_vals), (bars1, nki_vals), (bars2, ki_vals)]:
        for bar, v in zip(bars, vals):
            if not np.isfinite(v):
                continue
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(quantiles)
    ax.set_ylabel("|Δradial|_peak [m]")
    ax.set_title("Per-quantile comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # annotation: the underprediction gap
    gap_no_ki = (truth_p95 - nki_p95) / truth_p95 * 100
    gap_ki = (truth_p95 - ki_p95) / truth_p95 * 100
    ax.text(
        0.98, 0.50,
        f"P95 underprediction:\n"
        f"  Fix 1+2:   −{gap_no_ki:.0f} %\n"
        f"  Fix 1+2+3: −{gap_ki:.0f} %",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                  edgecolor="grey"),
    )

    # --- Panel 3: ensemble-mean Δsurge ---
    ax = axes[1, 0]
    ax.plot(t_truth, truth_dsurge, color="black", lw=2.0,
            label=f"brucon mean (N=30) peak {truth_dsurge[np.argmax(np.abs(truth_dsurge))]:+.2f} m")
    ax.plot(t_cqa, nki_dsu, color="tab:orange", lw=1.8,
            label=f"cal Fix 1+2 peak {nki_dsu[np.argmax(np.abs(nki_dsu))]:+.2f} m")
    ax.plot(t_cqa, ki_dsu, color="tab:blue", lw=1.8,
            label=f"cal Fix 1+2+3 peak {ki_dsu[np.argmax(np.abs(ki_dsu))]:+.2f} m")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlim(0, t_truth[-1])
    ax.set_xlabel("time since WCF [s]")
    ax.set_ylabel("Δsurge [m]")
    ax.set_title("Ensemble-mean Δsurge since WCF")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # --- Panel 4: ensemble-mean Δsway ---
    ax = axes[1, 1]
    ax.plot(t_truth, truth_dsway, color="black", lw=2.0,
            label=f"brucon mean (N=30) peak {truth_dsway[np.argmax(np.abs(truth_dsway))]:+.2f} m")
    ax.plot(t_cqa, nki_dsw, color="tab:orange", lw=1.8,
            label=f"cal Fix 1+2 peak {nki_dsw[np.argmax(np.abs(nki_dsw))]:+.2f} m")
    ax.plot(t_cqa, ki_dsw, color="tab:blue", lw=1.8,
            label=f"cal Fix 1+2+3 peak {ki_dsw[np.argmax(np.abs(ki_dsw))]:+.2f} m")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlim(0, t_truth[-1])
    ax.set_xlabel("time since WCF [s]")
    ax.set_ylabel("Δsway [m]")
    ax.set_title("Ensemble-mean Δsway since WCF")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
