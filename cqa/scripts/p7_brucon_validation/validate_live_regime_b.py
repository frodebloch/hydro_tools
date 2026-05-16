"""Ensemble validation of the live regime-B severity estimator
(sec.12.21.21.20).

For each seed of each brucon cell, do two things in parallel:

  1. PREDICT:  using only the pre-WCF window of (Tx, Ty, Tz), estimate
              the regime-B severity P(saturation) per DOF via the live
              estimator in ``cqa.live_regime_b``. Uses the brucon
              residual polytope (bus_port lost) as the cap.
  2. MEASURE: using the post-WCF window of (Tx, Ty, Tz), measure the
              actual fraction of time the demand exceeded the residual
              polytope cap per DOF.

If the framework is correct, predicted severity and measured exceedance
should agree across seeds and cells -- low for safe cells, high for the
known regime-B cells (bf8_h0_w45 mild, bf8_q10_w45 severe).

Output:
  * Cell-level summary table (mean predicted vs mean observed per cell).
  * Scatter plot of predicted vs observed (one point per seed).
  * Traffic-light breakdown per cell.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))
sys.path.insert(0, str(THIS))

from cqa.live_regime_b import (                                # noqa: E402
    estimate_regime_b_severity,
    saturation_probability_gaussian,
)
from saturation_screening import (                             # noqa: E402
    CSOV_THRUSTERS, CSOV_BUS_PORT_LOST, compute_residual_polytope,
)

T_WCF_S = 1560.0
SIM_DT = 0.1
FS_HZ = 1.0 / SIM_DT

# Pre-WCF window for live estimator
PRE_WIN_S = 300.0
PRE_T_END = T_WCF_S - 5.0
PRE_T_START = PRE_T_END - PRE_WIN_S

# Post-WCF window for empirical measurement
POST_T_START = T_WCF_S + 30.0   # skip regime-A transient
POST_T_END = T_WCF_S + 200.0

# Column indices (0-based)
COL_T = 0
COL_TX = 7    # Tx [kN]
COL_TY = 8    # Ty [kN]
COL_TZ = 9    # Tz [kN*m]

CELLS = [
    "bf4_c1_h0", "bf4_c1_q10",
    "bf6_h0", "bf6_q10", "bf6_h0_w45", "bf6_q10_w45",
    "bf8_h0", "bf8_q10", "bf8_h0_w45", "bf8_q10_w45",
    "pwq30",
]

# Residual polytope: brucon ports forces in kN, but cqa.live_regime_b
# wants N / Nm. Convert when feeding into the estimator.
_RES = compute_residual_polytope(
    CSOV_THRUSTERS,
    surviving_indices=tuple(
        i for i in range(len(CSOV_THRUSTERS)) if i not in CSOV_BUS_PORT_LOST
    ),
)
CAP_RESIDUAL_KN = np.array([
    min(abs(_RES.max_surge), abs(_RES.min_surge)),
    min(abs(_RES.max_sway),  abs(_RES.min_sway)),
    min(abs(_RES.max_yaw),   abs(_RES.min_yaw)),
])
# Convert kN/kNm -> N/Nm for the estimator API
CAP_RESIDUAL_N = CAP_RESIDUAL_KN * 1e3


def load_seed(cell: str, seed: int) -> dict | None:
    p = THIS / "work" / f"{cell}_seed{seed}" / f"{cell}_seed{seed}.out"
    if not p.exists():
        return None
    try:
        d = np.loadtxt(p, skiprows=1, usecols=[COL_T, COL_TX, COL_TY, COL_TZ])
    except Exception:
        return None
    return {"t": d[:, 0], "Tx": d[:, 1], "Ty": d[:, 2], "Tz": d[:, 3]}


def evaluate_seed(d: dict) -> dict | None:
    """Predict and measure for one seed. Returns None if the seed lacks
    the pre or post window."""
    t = d["t"]
    pre_mask = (t >= PRE_T_START) & (t <= PRE_T_END)
    post_mask = (t >= POST_T_START) & (t <= POST_T_END)
    if pre_mask.sum() < int(0.5 * PRE_WIN_S * FS_HZ):
        return None
    if post_mask.sum() < int(0.5 * (POST_T_END - POST_T_START) * FS_HZ):
        return None

    # Pre-WCF buffer in N / N*m
    buf_kN = np.column_stack([
        d["Tx"][pre_mask], d["Ty"][pre_mask], d["Tz"][pre_mask],
    ])
    buf_N = buf_kN * 1e3  # kN -> N (yaw kN*m -> N*m: same factor 1e3)

    res = estimate_regime_b_severity(
        buf_N, fs_hz=FS_HZ, cap_residual=tuple(CAP_RESIDUAL_N),
        window_s=PRE_WIN_S,
    )

    # Post-WCF observed exceedance per DOF
    post_kN = np.column_stack([
        d["Tx"][post_mask], d["Ty"][post_mask], d["Tz"][post_mask],
    ])
    post_N = post_kN * 1e3
    exc = (np.abs(post_N) > CAP_RESIDUAL_N[None, :]).mean(axis=0)

    return {
        "mu_kN":   res.mu / 1e3,
        "sig_kN":  res.sigma / 1e3,
        "p_sat":   res.p_sat,
        "severity": res.severity,
        "traffic":  res.traffic,
        "post_exc": exc,
    }


def main() -> None:
    print(f"[polytope] residual cap (kN, kN, kNm) = "
          f"({CAP_RESIDUAL_KN[0]:.0f}, {CAP_RESIDUAL_KN[1]:.0f}, "
          f"{CAP_RESIDUAL_KN[2]:.0f})")
    print(f"[window] pre-WCF: [{PRE_T_START:.0f}, {PRE_T_END:.0f}] s "
          f"({PRE_WIN_S:.0f} s)")
    print(f"[window] post-WCF: [{POST_T_START:.0f}, {POST_T_END:.0f}] s "
          f"(skip first 30 s for regime A)")
    print()

    all_results = {}
    for cell in CELLS:
        seeds = []
        for seed in range(1000, 1051):
            d = load_seed(cell, seed)
            if d is None:
                continue
            r = evaluate_seed(d)
            if r is not None:
                seeds.append(r)
        if not seeds:
            print(f"  {cell}: no usable seeds")
            continue
        all_results[cell] = seeds

    # ===== Cell-level summary table =====
    print("=" * 122)
    print("PREDICTED severity (pre-WCF) vs OBSERVED exceedance "
          "(post-WCF, regime B)")
    print("=" * 122)
    print(f"{'cell':<14s} {'n':>3s} | "
          f"{'pred_p_S%':>10s} {'pred_p_Y%':>10s} {'pred_p_Z%':>10s} "
          f"{'p_max%':>7s} | "
          f"{'obs_e_S%':>9s} {'obs_e_Y%':>9s} {'obs_e_Z%':>9s} | "
          f"{'G/A/R':>7s}")
    print("-" * 122)
    summary_rows = []
    for cell, rows in all_results.items():
        n = len(rows)
        p = np.array([r["p_sat"] for r in rows])
        e = np.array([r["post_exc"] for r in rows])
        traffic = [r["traffic"] for r in rows]
        n_g = sum(1 for t in traffic if t == "green")
        n_a = sum(1 for t in traffic if t == "amber")
        n_r = sum(1 for t in traffic if t == "red")
        sev = np.array([r["severity"] for r in rows])
        print(
            f"{cell:<14s} {n:>3d} | "
            f"{p[:,0].mean()*100:10.3f} {p[:,1].mean()*100:10.3f} "
            f"{p[:,2].mean()*100:10.3f} "
            f"{sev.mean()*100:7.2f} | "
            f"{e[:,0].mean()*100:9.2f} {e[:,1].mean()*100:9.2f} "
            f"{e[:,2].mean()*100:9.2f} | "
            f"{n_g}/{n_a}/{n_r}"
        )
        summary_rows.append({
            "cell": cell, "n": n, "pred_max_mean": float(sev.mean()),
            "obs_y_mean": float(e[:, 1].mean()),
            "obs_y_max": float(e[:, 1].max()),
            "pred_y_max": float(p[:, 1].max()),
        })

    # ===== Scatter: predicted severity vs observed exceedance, sway DOF =====
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    for cell, rows in all_results.items():
        p_y = np.array([r["p_sat"][1] for r in rows])
        e_y = np.array([r["post_exc"][1] for r in rows])
        ax.scatter(p_y + 1e-6, e_y + 1e-6,
                   label=cell, s=24, alpha=0.7)
    # 1:1 line
    lims = (1e-6, 1.0)
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5, label="1:1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.axvline(0.01, color="orange", ls=":", lw=0.7, alpha=0.6)
    ax.axvline(0.10, color="red", ls=":", lw=0.7, alpha=0.6)
    ax.set_xlabel("PREDICTED P(saturation) sway, from pre-WCF Ty")
    ax.set_ylabel("OBSERVED post-WCF exceedance fraction sway")
    ax.set_title("Per-seed: predicted vs observed (sway DOF)")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(alpha=0.3, which="both")

    # Cell-level summary bar chart
    ax = axes[1]
    cells_sorted = sorted(summary_rows, key=lambda r: -r["obs_y_mean"])
    cell_names = [r["cell"] for r in cells_sorted]
    pred = [r["pred_max_mean"] * 100 for r in cells_sorted]
    obs = [r["obs_y_mean"] * 100 for r in cells_sorted]
    x = np.arange(len(cell_names))
    ax.bar(x - 0.2, pred, 0.4, label="pred severity (max DOF) %", color="C0")
    ax.bar(x + 0.2, obs, 0.4, label="obs sway exc %", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(cell_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("%")
    ax.set_title("Cell-level mean: pred severity vs obs sway exceedance")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = THIS / "live_regime_b_ensemble_validation.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
