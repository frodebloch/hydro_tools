"""Per-seed, per-cell saturation regime classifier across the brucon validation set.

Two regimes per the conceptual model (sec.12.21.21.12):

  Regime A -- transient saturation, t in (0, T_A] post-WCF, driven by the deficit
              pulse + integrator wind-up + lost-bus geometry.
  Regime B -- sustained saturation, t in (T_A, T_END], driven by slow-varying
              environmental load exceeding the residual polytope per-DOF capacity.

For each seed and each DOF, we compute:
  - clip_frac_A : fraction of samples in regime-A window with |Order|>|Alloc|+thresh
  - clip_frac_B : fraction of samples in regime-B window with |Order|>|Alloc|+thresh
  - max_clip    : peak absolute clipping magnitude in each window
We also record post-WCF max |SwayDev|, |SurgeDev|, |HeadingDev| for correlation.

Output: tabular summary per cell (mean across seeds), per-cell histogram, plus
a master CSV-like print of (cell, seed, clip metrics, excursion metrics) so
we can read off the empirical A/B/AandB/none classification.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

T_WCF_S = 1560.0
T_A_END_S = 30.0   # regime A window: (0, 30] s post-WCF
T_B_END_S = 200.0  # regime B window: (30, 200] s post-WCF

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"

# Same column indices as bf8_q10_w45_alloc_vs_order.py (verified consistent
# across all bf*/pwq30 cells; pwq30 has extras after index 44 that we ignore).
COL = {
    "t": 0,
    "SurgeDev": 24,
    "SwayDev": 25,
    "HeadingDev": 22,
    "OrderTauSurge": 34,
    "OrderTauSway": 35,
    "OrderTauYaw": 36,
    "AllocTauSurge": 37,
    "AllocTauSway": 38,
    "AllocTauYaw": 39,
}

# Cells to scan: all CSOV cells that have ensembles with WCF.
CELLS = [
    "bf4_c1_h0",
    "bf4_c1_q10",
    "bf6_h0",
    "bf6_h0_w45",
    "bf6_q10",
    "bf6_q10_w45",
    "bf8_h0",
    "bf8_h0_w45",
    "bf8_q10",
    "bf8_q10_w45",
    "pwq30",
]

# Threshold for "clipping": |Order|-|Alloc| must exceed 1% of typical order
# magnitude in that DOF for the cell. Computed per cell + DOF below.


def load_seed(cell: str, seed: int) -> dict | None:
    f = WORK / f"{cell}_seed{seed}" / f"{cell}_seed{seed}.out"
    if not f.exists():
        return None
    try:
        d = np.loadtxt(f, skiprows=1, usecols=list(COL.values()))
    except Exception:
        return None
    return {
        "t": d[:, 0],
        "SurgeDev": d[:, 1],
        "SwayDev": d[:, 2],
        "HeadingDev": d[:, 3],
        "OrderSurge": d[:, 4],
        "OrderSway": d[:, 5],
        "OrderYaw": d[:, 6],
        "AllocSurge": d[:, 7],
        "AllocSway": d[:, 8],
        "AllocYaw": d[:, 9],
    }


def _clip_amount(O: np.ndarray, A: np.ndarray) -> np.ndarray:
    """|Order| - |Alloc| if same sign, else |Order|-|Alloc|.  Negative values
    indicate Alloc > Order which shouldn't happen (sign of bug); zero clipping
    when feasible."""
    same_sign = np.sign(O) == np.sign(A)
    clipped = np.maximum(np.abs(O) - np.abs(A), 0.0)
    return np.where(same_sign, clipped, np.abs(O) - np.abs(A))


def classify_seed(d: dict, thr_S: float, thr_Y: float, thr_Z: float) -> dict:
    t = d["t"] - T_WCF_S
    mA = (t > 0) & (t <= T_A_END_S)
    mB = (t > T_A_END_S) & (t <= T_B_END_S)
    mPre = (t >= -300) & (t <= -10)

    out = {}
    for dof, O, A, thr in [
        ("S", d["OrderSurge"], d["AllocSurge"], thr_S),
        ("Y", d["OrderSway"], d["AllocSway"], thr_Y),
        ("Z", d["OrderYaw"], d["AllocYaw"], thr_Z),
    ]:
        clip = _clip_amount(O, A)
        out[f"fA_{dof}"] = (clip[mA] > thr).mean() if mA.any() else 0.0
        out[f"fB_{dof}"] = (clip[mB] > thr).mean() if mB.any() else 0.0
        out[f"maxA_{dof}"] = clip[mA].max() if mA.any() else 0.0
        out[f"maxB_{dof}"] = clip[mB].max() if mB.any() else 0.0

    # Excursion metrics: max |dev| post-WCF, referenced to pre-WCF mean
    post = (t > 0) & (t <= T_B_END_S)
    for label, sig in [("surge", "SurgeDev"), ("sway", "SwayDev"),
                       ("yaw", "HeadingDev")]:
        ref = d[sig][mPre].mean() if mPre.any() else 0.0
        excursion = np.abs(d[sig][post] - ref)
        out[f"max_{label}"] = excursion.max() if post.any() else 0.0
    return out


def scan_cell(cell: str, seeds: range = range(1000, 1051)) -> list[dict]:
    """Load all available seeds, compute thresholds from the cell-aggregate,
    classify each seed."""
    raw = []
    for s in seeds:
        d = load_seed(cell, s)
        if d is not None:
            raw.append((s, d))
    if not raw:
        return []
    # Per-cell threshold = 1% of max |Order| across all seeds in this cell
    all_OS = np.concatenate([d["OrderSurge"] for _, d in raw])
    all_OY = np.concatenate([d["OrderSway"] for _, d in raw])
    all_OZ = np.concatenate([d["OrderYaw"] for _, d in raw])
    thr_S = 0.01 * np.abs(all_OS).max()
    thr_Y = 0.01 * np.abs(all_OY).max()
    thr_Z = 0.01 * np.abs(all_OZ).max()

    rows = []
    for s, d in raw:
        c = classify_seed(d, thr_S, thr_Y, thr_Z)
        c["cell"] = cell
        c["seed"] = s
        c["thr_S"] = thr_S
        c["thr_Y"] = thr_Y
        c["thr_Z"] = thr_Z
        rows.append(c)
    return rows


def main() -> None:
    all_rows = []
    for cell in CELLS:
        rows = scan_cell(cell)
        if not rows:
            print(f"  {cell}: no data")
            continue
        all_rows.extend(rows)

    # ===== Cell-level summary =====
    print("\n" + "=" * 110)
    print("CELL-LEVEL SUMMARY (mean across seeds)")
    print("=" * 110)
    hdr = (f"{'cell':<14s} {'n':>3s} | "
           f"{'fA_S%':>5s} {'fA_Y%':>5s} {'fA_Z%':>5s} | "
           f"{'fB_S%':>5s} {'fB_Y%':>5s} {'fB_Z%':>5s} | "
           f"{'P50|y|':>6s} {'P95|y|':>6s} {'max|y|':>6s}")
    print(hdr)
    print("-" * 110)
    for cell in CELLS:
        rows = [r for r in all_rows if r["cell"] == cell]
        if not rows:
            continue
        n = len(rows)

        def m(k: str) -> float:
            return float(np.mean([r[k] for r in rows]))

        max_y_arr = np.array([r["max_sway"] for r in rows])
        print(f"{cell:<14s} {n:>3d} | "
              f"{m('fA_S')*100:5.1f} {m('fA_Y')*100:5.1f} {m('fA_Z')*100:5.1f} | "
              f"{m('fB_S')*100:5.1f} {m('fB_Y')*100:5.1f} {m('fB_Z')*100:5.1f} | "
              f"{np.percentile(max_y_arr, 50):6.2f} "
              f"{np.percentile(max_y_arr, 95):6.2f} "
              f"{max_y_arr.max():6.2f}")

    # ===== Per-seed dump for outlier cells =====
    print("\n" + "=" * 110)
    print("PER-SEED DETAIL: cells with max|sway|>3m anywhere")
    print("=" * 110)
    print(f"{'cell':<14s} {'seed':>5s} | "
          f"{'fA_S%':>5s} {'fA_Y%':>5s} {'fA_Z%':>5s} | "
          f"{'fB_S%':>5s} {'fB_Y%':>5s} {'fB_Z%':>5s} | "
          f"{'mxA_Y':>6s} {'mxB_Y':>6s} | "
          f"{'mx|x|':>5s} {'mx|y|':>5s} {'mx|psi|':>7s}")
    print("-" * 110)
    for cell in CELLS:
        rows = [r for r in all_rows if r["cell"] == cell]
        if not rows:
            continue
        if max(r["max_sway"] for r in rows) < 3.0:
            continue
        for r in sorted(rows, key=lambda r: -r["max_sway"]):
            if r["max_sway"] < 3.0:
                continue
            print(f"{r['cell']:<14s} {r['seed']:>5d} | "
                  f"{r['fA_S']*100:5.1f} {r['fA_Y']*100:5.1f} {r['fA_Z']*100:5.1f} | "
                  f"{r['fB_S']*100:5.1f} {r['fB_Y']*100:5.1f} {r['fB_Z']*100:5.1f} | "
                  f"{r['maxA_Y']:6.1f} {r['maxB_Y']:6.1f} | "
                  f"{r['max_surge']:5.2f} {r['max_sway']:5.2f} {r['max_yaw']:7.2f}")

    # ===== Cross-tabulation: regime classification vs excursion =====
    print("\n" + "=" * 110)
    print("REGIME CLASSIFICATION (per seed) vs excursion")
    print("Threshold for 'active' = >5% of window with clipping in any DOF")
    print("=" * 110)
    bins = {"none": [], "A_only": [], "B_only": [], "A_and_B": []}
    for r in all_rows:
        a = max(r["fA_S"], r["fA_Y"], r["fA_Z"]) > 0.05
        b = max(r["fB_S"], r["fB_Y"], r["fB_Z"]) > 0.05
        key = ("A_and_B" if a and b else
               "A_only" if a else
               "B_only" if b else "none")
        bins[key].append(r)
    for key, rows in bins.items():
        if not rows:
            print(f"  {key:<10s}: 0 seeds")
            continue
        max_y = np.array([r["max_sway"] for r in rows])
        print(f"  {key:<10s}: {len(rows):3d} seeds | "
              f"max|sway| P50={np.percentile(max_y,50):4.2f} "
              f"P95={np.percentile(max_y,95):4.2f} "
              f"max={max_y.max():5.2f}")

    # ===== Per-cell A vs B counts =====
    print("\n" + "=" * 110)
    print("PER-CELL regime distribution (count of seeds in each class)")
    print("=" * 110)
    print(f"{'cell':<14s} {'n':>3s} | {'none':>5s} {'Aonly':>5s} {'Bonly':>5s} {'AandB':>5s}")
    print("-" * 60)
    for cell in CELLS:
        rows = [r for r in all_rows if r["cell"] == cell]
        if not rows:
            continue
        n_none = n_a = n_b = n_ab = 0
        for r in rows:
            a = max(r["fA_S"], r["fA_Y"], r["fA_Z"]) > 0.05
            b = max(r["fB_S"], r["fB_Y"], r["fB_Z"]) > 0.05
            if a and b: n_ab += 1
            elif a: n_a += 1
            elif b: n_b += 1
            else: n_none += 1
        print(f"{cell:<14s} {len(rows):>3d} | {n_none:>5d} {n_a:>5d} {n_b:>5d} {n_ab:>5d}")


if __name__ == "__main__":
    main()
