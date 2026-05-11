"""Cross-cell MF-band env-force PSD comparison.

Tests the hypothesis (analysis.md sec.12.21.13) that the bf8-oblique WCF
P95 under-prediction in cqa is driven by a missing MF (~20-200 s) band
of env-force fluctuation that is captured neither by the b_hat snapshot
(LF only) nor by the WF wave filter (~5-20 s).

Method: for each validation cell, for each seed, compute the band-sigma
of the total env force F_env = Wind + Drift + Cur on the hull in three
bands (MF: 20-200 s, WF: 5-20 s, VHF: <5 s). Average across seeds. Show
which cells have large MF energy vs which do not.

Cross-reference with the WCF P95 bias rollup (sec.12.21.9) to test the
hypothesis: cells with high MF on Fy / Mz should also be the cells with
the largest under-prediction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

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
T_PRE_START = 60.0
T_PRE_END = T_WCF - 5.0
SEED_RANGE = (1000, 1030)

BANDS = {
    "MF": (1/200.0, 1/20.0),
    "WF": (1/20.0, 1/5.0),
    "VHF": (1/5.0, 5.0),
}


def band_sigma(f, S, lo, hi):
    m = (f >= lo) & (f <= hi)
    if not m.any():
        return 0.0
    return float(np.sqrt(np.trapezoid(S[m], f[m])))


def cell_stats(tag: str) -> dict:
    work = THIS / "work"
    bands_per_seed = {b: {"Fx": [], "Fy": [], "Mz": []} for b in BANDS}
    means = {"Fx": [], "Fy": [], "Mz": []}
    for seed in range(SEED_RANGE[0], SEED_RANGE[1]):
        outp = work / f"{tag}_seed{seed:04d}" / f"{tag}_seed{seed:04d}.out"
        if not outp.exists():
            continue
        m = parse_output(outp)
        t = m.columns["t"]
        dt = float(np.median(np.diff(t)))
        fs = 1.0 / dt
        Fx = m.columns["WindX"] + m.columns["DriftX"] + m.columns["CurX"]
        Fy = m.columns["WindY"] + m.columns["DriftY"] + m.columns["CurY"]
        Mz = m.columns["WindMz"] + m.columns["DriftMz"] + m.columns["CurMz"]
        mask = (t >= T_PRE_START) & (t <= T_PRE_END)
        if mask.sum() < 1024:
            continue
        for name, sig in [("Fx", Fx[mask]), ("Fy", Fy[mask]), ("Mz", Mz[mask])]:
            means[name].append(float(sig.mean()))
            n_per = min(2048, len(sig) // 4)
            f, S = welch(sig - sig.mean(), fs=fs, nperseg=n_per)
            for b, (lo, hi) in BANDS.items():
                bands_per_seed[b][name].append(band_sigma(f, S, lo, hi))
    out = {"n_seeds": len(means["Fx"])}
    for name in ("Fx", "Fy", "Mz"):
        out[f"mean_{name}"] = float(np.mean(means[name])) if means[name] else 0.0
        for b in BANDS:
            arr = bands_per_seed[b][name]
            out[f"sig_{b}_{name}"] = float(np.mean(arr)) if arr else 0.0
    return out


def main():
    print(f"\nEnv-force band sigmas, ensemble-mean across seeds, pre-WCF window:\n")
    hdr = f"{'cell':<14} {'n':>3} | {'mean_Fy':>9} {'MF_Fy':>7} {'WF_Fy':>7} | "
    hdr += f"{'mean_Mz':>10} {'MF_Mz':>8} {'WF_Mz':>8} | {'MF/mean Fy':>10}"
    print(hdr)
    print("-" * len(hdr))
    for tag in CELLS:
        s = cell_stats(tag)
        if s["n_seeds"] == 0:
            print(f"{tag:<14} (no data)")
            continue
        ratio_Fy = abs(s["sig_MF_Fy"] / s["mean_Fy"]) if s["mean_Fy"] else 0.0
        print(f"{tag:<14} {s['n_seeds']:>3} | "
              f"{s['mean_Fy']:>+9.0f} {s['sig_MF_Fy']:>7.1f} {s['sig_WF_Fy']:>7.1f} | "
              f"{s['mean_Mz']:>+10.0f} {s['sig_MF_Mz']:>8.0f} {s['sig_WF_Mz']:>8.0f} | "
              f"{ratio_Fy:>10.2%}")


if __name__ == "__main__":
    main()
