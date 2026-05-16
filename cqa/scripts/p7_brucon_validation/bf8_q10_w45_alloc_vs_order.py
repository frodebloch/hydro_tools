"""bf8_q10_w45 saturation diagnostic: Order vs Alloc vs Delivered thrust.

Hypothesis (sec.12.21.21.11): per-thruster saturation, masked in DOF aggregates
by allocator rebalancing, manifests as Order > Alloc clipping when the order
exceeds the feasible thrust polytope.

The brucon log writes:
  Order*  = controller demand (raw, unclipped PID output)
  Alloc*  = allocator output (after thruster-feasibility QP)
  T*      = delivered thrust (after thruster dynamics)

Order != Alloc => infeasibility => per-thruster saturation.
Alloc != T     => thruster slew/dynamics lag (expected on transients).

Plot: 6-panel grid (3 DOF x 2 rows). Top row = position deviations vs t for
ensemble + outlier seeds. Bottom row = Order (dashed), Alloc (solid),
Delivered (thin) thrust traces.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

T_WCF_S = 1560.0
HERE = Path(__file__).resolve().parent
WORK = HERE / "work"

# Outlier seeds identified in sec.12.21.21.9 (post-WCF peaks 5.77-10.49 m).
OUTLIER_SEEDS = [1000, 1002, 1005, 1008, 1012, 1023, 1027]
SEED_RANGE = range(1000, 1051)  # 51 seeds total

# Column indices in .out file (0-indexed). We use the controller-frame, DP-filtered
# (LF) deviations SurgeDev/SwayDev/HeadingDev rather than NED x/y -- those are the
# signals the controller actually sees and acts on (no HF wave content).
COL = {
    "t": 0,
    "SurgeDev": 24,
    "SwayDev": 25,
    "HeadingDev": 22,
    "Tx": 7,
    "Ty": 8,
    "Tz": 9,
    "OrderTauSurge": 34,
    "OrderTauSway": 35,
    "OrderTauYaw": 36,
    "AllocTauSurge": 37,
    "AllocTauSway": 38,
    "AllocTauYaw": 39,
}


def load_seed(seed: int) -> dict | None:
    f = WORK / f"bf8_q10_w45_seed{seed}" / f"bf8_q10_w45_seed{seed}.out"
    if not f.exists():
        return None
    d = np.loadtxt(f, skiprows=1)
    return {
        "t": d[:, COL["t"]],
        "SurgeDev": d[:, COL["SurgeDev"]],
        "SwayDev": d[:, COL["SwayDev"]],
        "HeadingDev": d[:, COL["HeadingDev"]],
        "Tx": d[:, COL["Tx"]],
        "Ty": d[:, COL["Ty"]],
        "Tz": d[:, COL["Tz"]],
        "OrderSurge": d[:, COL["OrderTauSurge"]],
        "OrderSway": d[:, COL["OrderTauSway"]],
        "OrderYaw": d[:, COL["OrderTauYaw"]],
        "AllocSurge": d[:, COL["AllocTauSurge"]],
        "AllocSway": d[:, COL["AllocTauSway"]],
        "AllocYaw": d[:, COL["AllocTauYaw"]],
    }


def main() -> None:
    seeds_data: dict[int, dict] = {}
    for s in SEED_RANGE:
        d = load_seed(s)
        if d is not None:
            seeds_data[s] = d
    print(f"Loaded {len(seeds_data)} seeds")

    # Reference window: per-seed pre-WCF mean (t in [T_WCF-300, T_WCF-10]) for x,y,psi.
    # Define window relative to WCF.
    t_lo, t_hi = T_WCF_S - 30.0, T_WCF_S + 200.0

    # Common time grid (use seed 1012 grid as canonical -- all should match).
    t_ref = seeds_data[1012]["t"]
    mask = (t_ref >= t_lo) & (t_ref <= t_hi)
    ts = t_ref[mask] - T_WCF_S

    # Stack per-seed deviations.
    def collect(field: str) -> np.ndarray:
        arr = np.zeros((len(seeds_data), mask.sum()))
        for i, (s, d) in enumerate(sorted(seeds_data.items())):
            ref_mask = (d["t"] >= T_WCF_S - 300.0) & (d["t"] <= T_WCF_S - 10.0)
            ref = d[field][ref_mask].mean()
            arr[i] = d[field][mask] - ref
        return arr

    # Use the controller-frame DP-filtered deviations directly. SurgeDev/SwayDev
    # are body-frame, LF only (no HF wave content) -- matches what the sim console
    # displays in real time. Subtract per-seed pre-WCF mean to remove any residual
    # bias offset.
    surge = collect("SurgeDev")
    sway = collect("SwayDev")
    psi = collect("HeadingDev")

    # Thrust collections (no per-seed reference; absolute values).
    def collect_raw(field: str) -> np.ndarray:
        arr = np.zeros((len(seeds_data), mask.sum()))
        for i, (s, d) in enumerate(sorted(seeds_data.items())):
            arr[i] = d[field][mask]
        return arr

    Tx = collect_raw("Tx")
    Ty = collect_raw("Ty")
    Tz = collect_raw("Tz")
    OrderS = collect_raw("OrderSurge")
    OrderY = collect_raw("OrderSway")
    OrderZ = collect_raw("OrderYaw")
    AllocS = collect_raw("AllocSurge")
    AllocY = collect_raw("AllocSway")
    AllocZ = collect_raw("AllocYaw")

    seed_list = sorted(seeds_data.keys())
    outlier_idx = [seed_list.index(s) for s in OUTLIER_SEEDS if s in seed_list]

    # Saturation metric: clipping ratio = (|Order| - |Alloc|) when Order and Alloc
    # have the same sign and |Order| > |Alloc|, else 0.
    def clip_amount(O: np.ndarray, A: np.ndarray) -> np.ndarray:
        same_sign = np.sign(O) == np.sign(A)
        clipped = np.maximum(np.abs(O) - np.abs(A), 0.0)
        return np.where(same_sign, clipped, np.abs(O) - np.abs(A))

    clipS = clip_amount(OrderS, AllocS)
    clipY = clip_amount(OrderY, AllocY)
    clipZ = clip_amount(OrderZ, AllocZ)

    fig, axes = plt.subplots(3, 3, figsize=(18, 11), sharex=True)

    dofs = [
        ("Surge", surge, Tx, OrderS, AllocS, clipS, "kN"),
        ("Sway", sway, Ty, OrderY, AllocY, clipY, "kN"),
        ("Yaw [deg]", psi, Tz, OrderZ, AllocZ, clipZ, "kN.m"),
    ]

    for col, (label, dev, deliv, order, alloc, clip, unit) in enumerate(dofs):
        # Row 0: deviations
        ax = axes[0, col]
        ax.plot(ts, dev.T, color="lightgray", lw=0.5, alpha=0.6)
        ax.plot(ts, dev.mean(axis=0), color="black", lw=1.8, label="mean")
        ax.fill_between(ts, np.percentile(dev, 5, axis=0), np.percentile(dev, 95, axis=0),
                        color="black", alpha=0.15, label="5-95%")
        for k, idx in enumerate(outlier_idx):
            ax.plot(ts, dev[idx], lw=0.9, alpha=0.85,
                    label=f"seed {OUTLIER_SEEDS[k]}" if col == 1 else None)
        ax.axvline(0, color="red", ls=":", lw=0.8)
        ax.set_title(f"{label} deviation")
        ax.set_ylabel("[m]" if "Yaw" not in label else "[deg]")
        ax.grid(alpha=0.3)
        if col == 1:
            ax.legend(loc="upper left", fontsize=7, ncol=2)

        # Row 1: Order vs Alloc vs Delivered (ensemble mean only, with per-seed envelope)
        ax = axes[1, col]
        ax.plot(ts, order.mean(axis=0), color="C3", lw=1.5, ls="--", label="Order (cmd)")
        ax.plot(ts, alloc.mean(axis=0), color="C0", lw=1.5, label="Alloc (clipped)")
        ax.plot(ts, deliv.mean(axis=0), color="C2", lw=1.0, alpha=0.8, label="Delivered")
        ax.fill_between(ts,
                        np.percentile(order, 5, axis=0),
                        np.percentile(order, 95, axis=0),
                        color="C3", alpha=0.10)
        ax.fill_between(ts,
                        np.percentile(alloc, 5, axis=0),
                        np.percentile(alloc, 95, axis=0),
                        color="C0", alpha=0.10)
        ax.axvline(0, color="red", ls=":", lw=0.8)
        ax.axhline(0, color="black", lw=0.4)
        ax.set_ylabel(f"thrust [{unit}]")
        ax.grid(alpha=0.3)
        ax.set_title(f"{label.split()[0]} thrust: Order vs Alloc vs Delivered")
        if col == 0:
            ax.legend(loc="best", fontsize=8)

        # Row 2: clipping (Order - Alloc) -- saturation indicator
        ax = axes[2, col]
        # Per-seed clipping traces, gray
        ax.plot(ts, clip.T, color="lightgray", lw=0.4, alpha=0.5)
        ax.plot(ts, clip.mean(axis=0), color="black", lw=1.5, label="ensemble mean")
        ax.plot(ts, np.percentile(clip, 95, axis=0), color="C1", lw=1.2,
                label="95th pct")
        for k, idx in enumerate(outlier_idx):
            ax.plot(ts, clip[idx], lw=0.7, alpha=0.6, color=f"C{(k % 7) + 2}")
        ax.axvline(0, color="red", ls=":", lw=0.8)
        ax.axhline(0, color="black", lw=0.4)
        ax.set_xlabel("t - t_WCF [s]")
        ax.set_ylabel(f"|Order|-|Alloc| [{unit}]")
        ax.set_title(f"{label.split()[0]}: clipping (saturation)")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "bf8_q10_w45 (CSOV, Hs=5.7m Tp=10s wave_rel=45deg) — saturation diagnostic via Order vs Alloc",
        fontsize=12,
    )
    fig.tight_layout()
    out = HERE / "bf8_q10_w45_alloc_vs_order.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")

    # Numerical summary: clipping incidence and magnitude.
    print("\n=== Saturation summary (Order vs Alloc, t in [-20, +200]s) ===")
    for label, clip, unit in [("Surge", clipS, "kN"),
                              ("Sway", clipY, "kN"),
                              ("Yaw", clipZ, "kN.m")]:
        # Fraction of (seed, time) samples where clipping > 1% of typical order magnitude
        thresh = 0.01 * np.abs({"Surge": OrderS, "Sway": OrderY, "Yaw": OrderZ}[label]).max()
        frac = (clip > thresh).mean()
        max_seed_pct = (clip > thresh).mean(axis=1).max()  # worst seed
        print(f"  {label:6s}: max clip = {clip.max():8.1f} {unit:5s}  "
              f"| frac samples clipped (>{thresh:.1f}) = {frac:.3f}  "
              f"| worst-seed clipped time fraction = {max_seed_pct:.3f}")

    # Per-outlier seed clipping severity
    print("\n=== Outlier seeds: time-fraction with active clipping per DOF ===")
    print(f"{'seed':>6s} | {'surge%':>7s} {'sway%':>7s} {'yaw%':>7s} | {'max|y dev|':>10s}")
    for k, idx in enumerate(outlier_idx):
        s = OUTLIER_SEEDS[k]
        thr_s = 0.01 * np.abs(OrderS).max()
        thr_y = 0.01 * np.abs(OrderY).max()
        thr_z = 0.01 * np.abs(OrderZ).max()
        fs = (clipS[idx] > thr_s).mean() * 100
        fy = (clipY[idx] > thr_y).mean() * 100
        fz = (clipZ[idx] > thr_z).mean() * 100
        print(f"{s:>6d} | {fs:7.2f} {fy:7.2f} {fz:7.2f} | {np.abs(sway[idx]).max():10.2f}")


if __name__ == "__main__":
    main()
