"""Diagnostic: Order_y absolute level on worst-clipping bf8_q10_w45 seed
(sec.12.21.21.18).

Goal: determine whether regime-B clipping in the brucon ensemble means
``Order_y reaches the residual polytope cap of 1104 kN`` or instead means
``Order_y is well below the polytope but the allocator output Alloc_y is
still clipped`` (which would point at allocator-internal feasibility
constraints -- per-thruster max, azimuth slew rate, infeasible thrust
direction -- rather than the {tau_surge, tau_sway, tau_yaw} box).

This decides whether the residual-polytope framework (sec.12.21.21.15)
is the right abstraction at all.

For seeds previously flagged as outliers (clip-fraction 17-48 %, max|sway|
3.6-9.4 m), plot Order_y(t), Alloc_y(t), the residual polytope ±1104 kN,
and the empirical regime-B window (30..200 s post-WCF).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))
sys.path.insert(0, str(THIS))

from saturation_regime_scan import load_seed                 # noqa: E402

T_WCF_S = 1560.0
T_POST_WIN = 200.0

# Per the context: outlier seeds from saturation_regime_scan that have
# the largest regime-B clip-fractions in bf8_q10_w45.
OUTLIER_SEEDS = [1027, 1023, 1012, 1008, 1005, 1002, 1000]

# Residual polytope (kN), bus_port lost, sway DOF (sec.12.21.21.15).
CAP_Y_KN = 1103.66


def main() -> None:
    tag = "bf8_q10_w45"
    fig, axes = plt.subplots(len(OUTLIER_SEEDS), 1,
                              figsize=(11, 2.0 * len(OUTLIER_SEEDS)),
                              sharex=True)
    for ax, seed in zip(axes, OUTLIER_SEEDS):
        d = load_seed(tag, seed)
        if d is None:
            ax.text(0.5, 0.5, f"seed {seed}: no data",
                    transform=ax.transAxes, ha="center")
            continue
        t = d["t"] - T_WCF_S
        m = (t >= 0) & (t <= T_POST_WIN)
        t_p = t[m]
        O_y = d["OrderSway"][m] / 1e3
        A_y = d["AllocSway"][m] / 1e3

        # Compute summary stats for the regime-B window (30..200 s).
        mB = (t_p > 30) & (t_p <= T_POST_WIN)
        order_max = float(np.abs(O_y[mB]).max()) if mB.any() else np.nan
        order_mean = float(O_y[mB].mean()) if mB.any() else np.nan
        order_std = float(O_y[mB].std()) if mB.any() else np.nan
        clip = np.abs(O_y) - np.abs(A_y)
        clip_frac = float((clip[mB] > 0.01 * np.abs(O_y).max()).mean()) \
            if mB.any() else 0.0
        polytope_breach = float((np.abs(O_y[mB]) > CAP_Y_KN).mean()) \
            if mB.any() else 0.0

        ax.plot(t_p, O_y, "C0-", lw=1.0, label="OrderTauSway")
        ax.plot(t_p, A_y, "C1-", lw=1.0, alpha=0.8, label="AllocTauSway")
        ax.axhline(+CAP_Y_KN, color="r", ls="--", lw=1.0,
                   label=f"residual polytope ±{CAP_Y_KN:.0f} kN")
        ax.axhline(-CAP_Y_KN, color="r", ls="--", lw=1.0)
        ax.axvspan(30, T_POST_WIN, color="0.85", alpha=0.4, zorder=0)
        ax.set_ylabel(f"seed {seed}\n[kN]")
        ax.grid(alpha=0.3)
        ax.text(
            0.99, 0.95,
            f"regime-B: |O_y| max={order_max:.0f} kN  "
            f"mean={order_mean:+.0f} std={order_std:.0f}\n"
            f"clip_frac={clip_frac:.2f}  "
            f"polytope_breach_frac={polytope_breach:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, family="monospace",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.5")
        )
        if ax is axes[0]:
            ax.legend(loc="lower right", fontsize=8, ncol=3)
            ax.set_title(f"{tag}: Order_y vs Alloc_y on outlier seeds, "
                         f"residual polytope cap")

    axes[-1].set_xlabel("t - t_WCF [s] (shaded = regime-B window 30..200 s)")
    out = THIS / "bf8_q10_w45_outlier_order_y_vs_polytope.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
