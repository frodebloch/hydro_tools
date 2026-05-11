"""Tests 1+2 from the MF-band hypothesis (analysis.md sec.12.21.13).

Hypothesis: cqa under-predicts WCF P95 on bf8-oblique cells because the
b_hat snapshot only captures the LF (>~200 s) mean of the env force on
the hull, while the env force has a substantial MF (~20-200 s) band that
is not in b_hat (filtered out by the slow bias estimator) AND not in WF
(too slow for the wave filter). In post-WCF operation, MF peaks act as
unmodelled forcing during the realloc transient, fattening the tail of
the position excursion distribution.

Test 1: PSD of total env force F_env(t) = Wind+Drift+Cur on hull, vs
b_hat snapshot value, vs WF band. Look for energy in 0.005-0.05 Hz
(20-200 s) period band.

Test 2: Time-domain sigma of the MF-band-pass-filtered env force,
compared to the b_hat snapshot uncertainty (sigma_R_b_hat_m).

One seed (bf8_q10_w45 seed 1000), all 3 DOFs (X, Y, Mz).
Outputs PNG.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from harness import parse_output  # noqa: E402


TAG = "bf8_q10_w45"
SEED = 1000
T_WCF = 560.0
T_PRE_START = 60.0   # skip startup transient
T_PRE_END = T_WCF - 5.0  # pre-WCF window for env-force statistics


def main():
    seed_dir = THIS / "work" / f"{TAG}_seed{SEED:04d}"
    m = parse_output(seed_dir / f"{TAG}_seed{SEED:04d}.out")
    t = m.columns["t"]
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt

    # Total env force on hull, NED, kN/kNm
    Fx = m.columns["WindX"] + m.columns["DriftX"] + m.columns["CurX"]
    Fy = m.columns["WindY"] + m.columns["DriftY"] + m.columns["CurY"]
    Mz = m.columns["WindMz"] + m.columns["DriftMz"] + m.columns["CurMz"]

    # Restrict to pre-WCF window (intact, stationary)
    mask = (t >= T_PRE_START) & (t <= T_PRE_END)
    Fx_w = Fx[mask]
    Fy_w = Fy[mask]
    Mz_w = Mz[mask]
    t_w = t[mask]
    print(f"window: t = [{T_PRE_START}, {T_PRE_END}] s, n = {len(t_w)} samples, "
          f"fs = {fs:.3f} Hz, dt = {dt:.2f} s")

    # ---- Test 1: PSD ----
    from scipy.signal import welch
    n_per = min(2048, len(Fx_w) // 4)
    fxx, Sxx = welch(Fx_w - Fx_w.mean(), fs=fs, nperseg=n_per)
    fyy, Syy = welch(Fy_w - Fy_w.mean(), fs=fs, nperseg=n_per)
    fmm, Smm = welch(Mz_w - Mz_w.mean(), fs=fs, nperseg=n_per)

    # Define bands (Hz)
    BANDS = {
        "MF (20-200 s)": (1/200.0, 1/20.0),
        "WF (5-20 s)":   (1/20.0, 1/5.0),
        "VHF (>5 s)":    (1/5.0, fs/2),
    }

    def band_sigma(f, S, lo, hi):
        m = (f >= lo) & (f <= hi)
        if not m.any():
            return 0.0
        return float(np.sqrt(np.trapezoid(S[m], f[m])))

    print(f"\nEnv-force band sigmas (pre-WCF window):")
    print(f"{'band':<18} {'sigma_Fx [kN]':>14} {'sigma_Fy [kN]':>14} {'sigma_Mz [kNm]':>16}")
    for name, (lo, hi) in BANDS.items():
        sx = band_sigma(fxx, Sxx, lo, hi)
        sy = band_sigma(fyy, Syy, lo, hi)
        sm = band_sigma(fmm, Smm, lo, hi)
        print(f"{name:<18} {sx:>14.1f} {sy:>14.1f} {sm:>16.0f}")

    # Total sigmas (full demean)
    sx_tot = float(Fx_w.std())
    sy_tot = float(Fy_w.std())
    sm_tot = float(Mz_w.std())
    print(f"{'TOTAL (time-dom)':<18} {sx_tot:>14.1f} {sy_tot:>14.1f} {sm_tot:>16.0f}")
    print(f"{'MEAN (b_hat-true)':<18} {Fx_w.mean():>+14.1f} {Fy_w.mean():>+14.1f} {Mz_w.mean():>+16.0f}")

    # ---- Compare to estimator b_hat ----
    # Load estimator output (LiveObserverState would have b_hat from end of pre-WCF)
    est_p = seed_dir / f"{TAG}_seed{SEED:04d}_estimator.out"
    est = parse_output(est_p)
    print(f"\nestimator cols (sample): {sorted(est.columns.keys())[:30]}")

    plt.figure(figsize=(11, 7))
    for i, (name, f, S, lab) in enumerate(
        [("Fx", fxx, Sxx, "Fx [kN²/Hz]"),
         ("Fy", fyy, Syy, "Fy [kN²/Hz]"),
         ("Mz", fmm, Smm, "Mz [kNm²/Hz]")], start=1):
        ax = plt.subplot(3, 1, i)
        ax.loglog(f, S, "k-", lw=1)
        for bname, (lo, hi) in BANDS.items():
            ax.axvspan(lo, hi, alpha=0.12, label=bname)
        ax.set_ylabel(lab)
        ax.grid(True, which="both", alpha=0.3)
        if i == 1:
            ax.legend(loc="lower left", fontsize=8)
            ax.set_title(f"Env-force PSD on hull, {TAG} seed{SEED:04d}, "
                         f"pre-WCF window [{T_PRE_START},{T_PRE_END}] s")
    plt.xlabel("frequency [Hz]")
    plt.tight_layout()
    out = THIS / f"diagnose_env_force_mf_band_{TAG}_seed{SEED:04d}.png"
    plt.savefig(out, dpi=120)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
