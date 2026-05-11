"""Cross-cell b_hat snapshot vs true env-force ratio (analysis.md sec.12.21.13).

Tests the hypothesis that the DP bias estimator's settling behavior
produces a systematic ~10% under-estimation of the true env force
on the hull, owing to the steady-state coupling between the bias
estimator and the controller's PI integrator (with tau_b=1000 s,
K_p=0.0012 on surge/sway, K_p=0.002 on yaw).

For each of 12 validation cells, computes per-seed:
  * b_hat snapshot from estimator output (EstBias*, kN/kNm) at
    t=T_WCF-5s -- exactly what cqa sees as the live observer state.
  * True env force mean: F_env = Wind + Drift + Cur, averaged over the
    SETTLED pre-WCF window [T_WCF-60, T_WCF-5] s.
  * Ratio b_hat / F_env per DOF.

Cross-cell rollup reveals whether the under-estimation is:
  * Universal (~ const factor across all cells) -> static scalar
    correction in cqa.
  * Magnitude-dependent (bigger gap for higher F_env) -> coupling
    analysis needed.
  * Heading-dependent (different on oblique vs head) -> hints at
    nonlinearity (e.g. crab angle interactions).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from harness import parse_output  # noqa: E402

CELLS = [
    "bf4_c1_h0", "bf4_c1_q10",
    "bf6_h0", "bf6_q10", "bf6_h0_w45", "bf6_q10_w45",
    "bf8_h0", "bf8_q10", "bf8_h0_w45", "bf8_q10_w45",
    "pwo", "pwq30",
]
T_WCF = 560.0
T_EVAL_DT = -5.0
PRE_SIGMA_WIN = (-60.0, -5.0)
SEED_RANGE = (1000, 1030)


def cell_stats(tag: str) -> dict:
    work = THIS / "work"
    ratios_x, ratios_y, ratios_z = [], [], []
    bhat_x, bhat_y, bhat_z = [], [], []
    true_x, true_y, true_z = [], [], []
    for seed in range(SEED_RANGE[0], SEED_RANGE[1]):
        seed_dir = work / f"{tag}_seed{seed:04d}"
        main_p = seed_dir / f"{tag}_seed{seed:04d}.out"
        est_p = seed_dir / f"{tag}_seed{seed:04d}_estimator.out"
        if not (main_p.exists() and est_p.exists()):
            continue
        m = parse_output(main_p)
        e = parse_output(est_p)
        t_e = e.columns["Time"]
        t_m = m.columns["t"]
        if t_m[-1] < T_WCF or t_e[-1] < T_WCF + T_EVAL_DT:
            continue

        # b_hat snapshot at t_eval
        i_eval = int(np.argmin(np.abs(t_e - (T_WCF + T_EVAL_DT))))
        bx = float(e.columns["EstBiasSurge"][i_eval])
        by = float(e.columns["EstBiasSway"][i_eval])
        bz = float(e.columns["EstBiasYaw"][i_eval])

        # True F_env mean over settled pre-WCF window (kN, kNm)
        settled = (t_m >= T_WCF + PRE_SIGMA_WIN[0]) & (t_m <= T_WCF + PRE_SIGMA_WIN[1])
        if settled.sum() < 100:
            continue
        Fx = float((m.columns["WindX"] + m.columns["DriftX"] + m.columns["CurX"])[settled].mean())
        Fy = float((m.columns["WindY"] + m.columns["DriftY"] + m.columns["CurY"])[settled].mean())
        Mz = float((m.columns["WindMz"] + m.columns["DriftMz"] + m.columns["CurMz"])[settled].mean())

        bhat_x.append(bx); bhat_y.append(by); bhat_z.append(bz)
        true_x.append(Fx); true_y.append(Fy); true_z.append(Mz)
        # Per-seed ratios only meaningful when |F| not near 0.
        if abs(Fx) > 10.0:
            ratios_x.append(bx / Fx)
        if abs(Fy) > 10.0:
            ratios_y.append(by / Fy)
        if abs(Fz := Mz) > 100.0:
            ratios_z.append(bz / Mz)
    return dict(
        n=len(bhat_x),
        bx=np.array(bhat_x), by=np.array(bhat_y), bz=np.array(bhat_z),
        tx=np.array(true_x), ty=np.array(true_y), tz=np.array(true_z),
        rx=np.array(ratios_x), ry=np.array(ratios_y), rz=np.array(ratios_z),
    )


def main():
    print(f"\nb_hat snapshot vs F_env true mean, per cell:\n")
    hdr = (f"{'cell':<14} {'n':>3} | "
           f"{'F_x_true':>9} {'b_x mean':>9} {'r_x':>6} | "
           f"{'F_y_true':>9} {'b_y mean':>9} {'r_y':>6} | "
           f"{'F_z_true':>10} {'b_z mean':>10} {'r_z':>6}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for tag in CELLS:
        s = cell_stats(tag)
        if s["n"] == 0:
            print(f"{tag:<14} (no data)")
            continue
        rx = s["rx"].mean() if len(s["rx"]) else float("nan")
        ry = s["ry"].mean() if len(s["ry"]) else float("nan")
        rz = s["rz"].mean() if len(s["rz"]) else float("nan")
        rows.append((tag, s, rx, ry, rz))
        print(f"{tag:<14} {s['n']:>3} | "
              f"{s['tx'].mean():>+9.0f} {s['bx'].mean():>+9.0f} {rx:>6.3f} | "
              f"{s['ty'].mean():>+9.0f} {s['by'].mean():>+9.0f} {ry:>6.3f} | "
              f"{s['tz'].mean():>+10.0f} {s['bz'].mean():>+10.0f} {rz:>6.3f}")

    # Per-axis aggregate (only cells where |F| meaningfully non-zero):
    all_rx, all_ry, all_rz = [], [], []
    for _, s, _, _, _ in rows:
        all_rx.extend(s["rx"].tolist())
        all_ry.extend(s["ry"].tolist())
        all_rz.extend(s["rz"].tolist())
    print()
    print(f"Aggregate b_hat/F_env ratios (per-seed pooled, |F|>threshold cells only):")
    print(f"  Surge (Fx): n={len(all_rx):>4}  mean={np.mean(all_rx):.3f}  std={np.std(all_rx):.3f}")
    print(f"  Sway  (Fy): n={len(all_ry):>4}  mean={np.mean(all_ry):.3f}  std={np.std(all_ry):.3f}")
    print(f"  Yaw   (Mz): n={len(all_rz):>4}  mean={np.mean(all_rz):.3f}  std={np.std(all_rz):.3f}")


if __name__ == "__main__":
    main()
