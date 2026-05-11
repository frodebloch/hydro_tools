"""Per-seed MF-window correlation test (analysis.md sec.12.21.13 test 3).

Hypothesis: the per-seed WCF position peak in brucon is driven by MF
band env-force peaks that arrive during the post-WCF window. If true,
per-seed correlation between (band-passed MF peak in [T_WCF, T_WCF+60]s)
and (truth R_LF peak in [T_WCF+5, T_WCF+120]s) should be positive and
significant, while baseline cqa A_R (which only sees b_hat snapshot) is
uncorrelated with truth (we already showed Pearson ~ 0).

Method, per seed (bf8_q10_w45, 30 seeds):
  1. Load Fy and Mz total env force from brucon main .out.
  2. Band-pass filter into MF band (20-200 s).
  3. Compute |F_MF| peak in post-WCF window [T_WCF, T_WCF+60] s.
  4. Compute truth: demeaned LF radial peak in post-WCF window
     [T_WCF+5, T_WCF+120] s.
  5. Correlate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import pearsonr, spearmanr

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from harness import parse_output  # noqa: E402

TAG = "bf8_q10_w45"
T_WCF = 560.0
SEED_RANGE = (1000, 1030)
POST_FORCE_WIN = (0.0, 60.0)     # for MF peak (relative to T_WCF)
POST_TRUTH_WIN = (5.0, 120.0)    # for truth R peak
PRE_LF_WIN = (-20.0, -1.0)       # for LF demean baseline

# MF band edges (Hz) — match sec.12.21.13 definition
MF_LO_HZ = 1.0 / 200.0
MF_HI_HZ = 1.0 / 20.0


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def main():
    rows = []
    work = THIS / "work"
    for seed in range(SEED_RANGE[0], SEED_RANGE[1]):
        seed_dir = work / f"{TAG}_seed{seed:04d}"
        outp = seed_dir / f"{TAG}_seed{seed:04d}.out"
        if not outp.exists():
            continue
        m = parse_output(outp)
        t = m.columns["t"]
        dt = float(np.median(np.diff(t)))
        fs = 1.0 / dt

        Fx = m.columns["WindX"] + m.columns["DriftX"] + m.columns["CurX"]
        Fy = m.columns["WindY"] + m.columns["DriftY"] + m.columns["CurY"]
        Mz = m.columns["WindMz"] + m.columns["DriftMz"] + m.columns["CurMz"]

        # Band-pass entire signal (need long context for filter)
        Fx_mf = bandpass(Fx, fs, MF_LO_HZ, MF_HI_HZ)
        Fy_mf = bandpass(Fy, fs, MF_LO_HZ, MF_HI_HZ)
        Mz_mf = bandpass(Mz, fs, MF_LO_HZ, MF_HI_HZ)

        # MF peak in post-WCF window
        post_force = (t >= T_WCF + POST_FORCE_WIN[0]) & (t <= T_WCF + POST_FORCE_WIN[1])
        Fx_peak = float(np.max(np.abs(Fx_mf[post_force])))
        Fy_peak = float(np.max(np.abs(Fy_mf[post_force])))
        Mz_peak = float(np.max(np.abs(Mz_mf[post_force])))
        # Also the "radial" MF force peak (combining surge+sway in body-NED)
        # but in NED here, OK as 2D vector magnitude.
        F2D_peak = float(np.max(np.hypot(Fx_mf[post_force], Fy_mf[post_force])))

        # Truth: demeaned LF radial peak (SurgeDev/SwayDev pre-WCF demean)
        pre_lf = (t >= T_WCF + PRE_LF_WIN[0]) & (t <= T_WCF + PRE_LF_WIN[1])
        sd_pre = float(m.columns["SurgeDev"][pre_lf].mean())
        wd_pre = float(m.columns["SwayDev"][pre_lf].mean())
        R_lf = np.hypot(m.columns["SurgeDev"] - sd_pre,
                        m.columns["SwayDev"] - wd_pre)
        post_truth = (t >= T_WCF + POST_TRUTH_WIN[0]) & (t <= T_WCF + POST_TRUTH_WIN[1])
        truth = float(np.max(R_lf[post_truth]))

        rows.append((seed, truth, Fx_peak, Fy_peak, Mz_peak, F2D_peak))

    arr = np.array(rows)
    seeds = arr[:, 0].astype(int)
    truth = arr[:, 1]
    Fx_peak = arr[:, 2]
    Fy_peak = arr[:, 3]
    Mz_peak = arr[:, 4]
    F2D_peak = arr[:, 5]

    print(f"\nn = {len(rows)} seeds, {TAG}")
    print(f"\nPer-seed table (first 10):")
    print(f"{'seed':>5} {'truth':>7} {'|Fx_MF|':>9} {'|Fy_MF|':>9} {'|Mz_MF|':>9} {'|F2D|':>8}")
    for r in rows[:10]:
        print(f"{r[0]:>5} {r[1]:>7.3f} {r[2]:>9.1f} {r[3]:>9.1f} {r[4]:>9.0f} {r[5]:>8.1f}")

    print(f"\nCorrelations (truth vs post-WCF MF peak):")
    for name, x in [("|Fx_MF|", Fx_peak), ("|Fy_MF|", Fy_peak),
                    ("|Mz_MF|", Mz_peak), ("|F2D_MF|", F2D_peak)]:
        p_r, p_p = pearsonr(truth, x)
        s_r, s_p = spearmanr(truth, x)
        print(f"  {name:<10} Pearson r={p_r:+.3f} (p={p_p:.3f})  "
              f"Spearman ρ={s_r:+.3f} (p={s_p:.3f})")

    print(f"\nMarginal stats:")
    print(f"  truth   : mean={truth.mean():.2f}  std={truth.std():.2f}  "
          f"min={truth.min():.2f}  max={truth.max():.2f}")
    print(f"  |Fx_MF| : mean={Fx_peak.mean():.1f}  std={Fx_peak.std():.1f}  kN")
    print(f"  |Fy_MF| : mean={Fy_peak.mean():.1f}  std={Fy_peak.std():.1f}  kN")
    print(f"  |Mz_MF| : mean={Mz_peak.mean():.0f}  std={Mz_peak.std():.0f}  kNm")
    print(f"  |F2D|   : mean={F2D_peak.mean():.1f}  std={F2D_peak.std():.1f}  kN")


if __name__ == "__main__":
    main()
