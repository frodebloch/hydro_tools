"""How does sigma_y depend on the live BayesianSigmaEstimator window?

Background
----------
``live_cell_per_seed_pwq30.py`` reports sigma_R total (median posterior)
= 0.668 m per seed, vs brucon ensemble truth = 0.840 m, a 20% under-
prediction. The current setup feeds the BayesianSigmaEstimator the
pre-WCF samples MEAN-DETRENDED (per seed) over a 60 s window, with
``assume_zero_mean=True``.

This script decomposes that 20% gap by sweeping the window length
W in {30, 60, 120, 300, 500} s and tracking three sigma estimators
on the same brucon data:

  (a) RAW samples, sample-variance (no mean removed):
        sigma^2 = mean(x^2)
      includes setpoint offset + long-period DP wander + WF + noise.

  (b) MEAN-DETRENDED samples, sample-variance (windowed mean removed):
        sigma^2 = mean((x - x_bar_W)^2)
      drops the in-window mean (= setpoint offset + part of LF wander
      with period > W). Matches the current live-cell behaviour.

  (c) FULL-RECORD sample-variance (single mean over the entire pre-WCF
      stretch removed once, then taken across W non-overlapping windows):
        the "ground truth" the brucon ensemble report (0.840 m) is
      conceptually closest to.

The curves are computed for both LF channels (SurgeDev, SwayDev) and
WF channels (xHf, yHf), per seed and ensemble-averaged. Output: a
4x2 figure with sigma_axis vs W, mean-detrended (red) vs raw (blue)
vs full-record reference (black dashed), per channel.

Run with::

    PYTHONPATH=. .venv/bin/python \\
        scripts/p7_brucon_validation/sigma_window_sweep_pwq30.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))


WORK_ROOT = THIS / "work"
TAG = "pwq30"
SEEDS = list(range(1000, 1030))

T_WCF = 560.0
PRE_START = 60.0     # discard the first 60 s (DP transient at sim start)
PRE_END = 560.0      # right up to the WCF
WINDOWS_S = [30.0, 60.0, 120.0, 300.0, 500.0]

CHANNELS = [
    ("LF surge",  "SurgeDev"),
    ("LF sway",   "SwayDev"),
    ("WF surge",  "xHf"),
    ("WF sway",   "yHf"),
]


def _load_main(seed: int):
    seed_dir = WORK_ROOT / f"{TAG}_seed{seed:04d}"
    if not seed_dir.exists():
        return None
    main_p = next((p for p in seed_dir.glob("*.out") if "estimator" not in p.name), None)
    if main_p is None:
        return None
    with open(main_p) as f:
        hdr = f.readline().strip().split("\t")
    data = np.loadtxt(main_p, skiprows=1, delimiter="\t")
    return {h: data[:, i] for i, h in enumerate(hdr)}


def windowed_sigma(x: np.ndarray, t: np.ndarray, W: float, *, detrend: bool):
    """Return the average windowed sample-std over non-overlapping windows
    of length W within [PRE_START, PRE_END]. ``detrend=True`` subtracts
    the per-window mean; ``False`` uses raw squares.
    """
    pre_mask = (t >= PRE_START) & (t <= PRE_END)
    tw = t[pre_mask]
    xw = x[pre_mask]
    dt = float(tw[1] - tw[0])
    n_per = max(1, int(round(W / dt)))
    n_full = len(xw) // n_per
    if n_full < 1:
        return float("nan")
    sigmas = []
    for k in range(n_full):
        seg = xw[k * n_per:(k + 1) * n_per]
        if detrend:
            seg = seg - seg.mean()
            sigmas.append(float(np.sqrt(np.mean(seg * seg))))
        else:
            sigmas.append(float(np.sqrt(np.mean(seg * seg))))
    # Average sigma across windows.
    return float(np.mean(sigmas))


def full_record_sigma(x: np.ndarray, t: np.ndarray):
    """Sample-std over the entire pre-WCF stretch, with the single full
    mean removed. This is the reference the brucon ensemble truth uses.
    """
    pre_mask = (t >= PRE_START) & (t <= PRE_END)
    xw = x[pre_mask]
    xw = xw - xw.mean()
    return float(np.sqrt(np.mean(xw * xw)))


def main():
    seed_data = []
    for seed in SEEDS:
        cols = _load_main(seed)
        if cols is None:
            continue
        seed_data.append((seed, cols))
    print(f"Loaded {len(seed_data)} seeds")

    # results[(channel, variant, W)] = list of per-seed sigmas
    results: dict[tuple[str, str, float], list[float]] = {}
    full_record: dict[str, list[float]] = {ch: [] for _, ch in CHANNELS}
    for _, cols in seed_data:
        t = cols["t"]
        for _, ch in CHANNELS:
            x = cols[ch]
            full_record[ch].append(full_record_sigma(x, t))
            for W in WINDOWS_S:
                results.setdefault((ch, "raw", W), []).append(
                    windowed_sigma(x, t, W, detrend=False))
                results.setdefault((ch, "detr", W), []).append(
                    windowed_sigma(x, t, W, detrend=True))

    # Ensemble mean (and 5-95 across seeds) per (channel, variant, W).
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (label, ch) in zip(axes.flat, CHANNELS):
        sig_full = np.array(full_record[ch])
        full_med = float(np.median(sig_full))
        sig_raw = np.array([results[(ch, "raw", W)] for W in WINDOWS_S])
        sig_det = np.array([results[(ch, "detr", W)] for W in WINDOWS_S])
        # axis 0 is W, axis 1 is seed
        raw_med = np.median(sig_raw, axis=1)
        raw_lo = np.percentile(sig_raw, 5, axis=1)
        raw_hi = np.percentile(sig_raw, 95, axis=1)
        det_med = np.median(sig_det, axis=1)
        det_lo = np.percentile(sig_det, 5, axis=1)
        det_hi = np.percentile(sig_det, 95, axis=1)

        ax.fill_between(WINDOWS_S, raw_lo, raw_hi, alpha=0.15, color="C0")
        ax.plot(WINDOWS_S, raw_med, color="C0", marker="o", label="raw (no detrend)")
        ax.fill_between(WINDOWS_S, det_lo, det_hi, alpha=0.15, color="C3")
        ax.plot(WINDOWS_S, det_med, color="C3", marker="s",
                label="windowed mean detrend (current live cell)")
        ax.axhline(full_med, color="k", lw=1.0, ls="--",
                   label=f"full pre-WCF stretch ref = {full_med:.3f} m")

        ax.set_xscale("log")
        ax.set_xlabel("window length W [s]")
        ax.set_ylabel("sigma [m]")
        ax.set_title(f"{label}  ({ch})")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, which="both", alpha=0.3)

    # Overall: total sigma_R = sqrt(sigma_x^2 + sigma_y^2) for LF and WF.
    print("\n--- ensemble medians (m) ---")
    for label, ch in CHANNELS:
        sig_full_med = float(np.median(full_record[ch]))
        print(f"  {label:10s} ({ch:9s}) full-record ref : {sig_full_med:.3f}")
        for W in WINDOWS_S:
            r = float(np.median(results[(ch, "raw", W)]))
            d = float(np.median(results[(ch, "detr", W)]))
            print(f"      W={W:5.0f}s  raw={r:.3f}   detr={d:.3f}   "
                  f"detr/full={d/sig_full_med:5.2f}")

    # Total radial sigmas at each window length:
    print("\n--- LF radial sigma_R = sqrt(sigma_x^2 + sigma_y^2), median over seeds ---")
    sig_full_x = np.median(full_record["SurgeDev"])
    sig_full_y = np.median(full_record["SwayDev"])
    full_R_lf = float(np.hypot(sig_full_x, sig_full_y))
    print(f"  full-record ref : {full_R_lf:.3f}")
    for W in WINDOWS_S:
        sx = np.median(results[("SurgeDev", "detr", W)])
        sy = np.median(results[("SwayDev", "detr", W)])
        R = float(np.hypot(sx, sy))
        print(f"  W={W:5.0f}s detr : {R:.3f}   ({R/full_R_lf:.2f}x ref)")

    print("\n--- WF radial sigma_R, median over seeds ---")
    sig_full_x = np.median(full_record["xHf"])
    sig_full_y = np.median(full_record["yHf"])
    full_R_wf = float(np.hypot(sig_full_x, sig_full_y))
    print(f"  full-record ref : {full_R_wf:.3f}")
    for W in WINDOWS_S:
        sx = np.median(results[("xHf", "detr", W)])
        sy = np.median(results[("yHf", "detr", W)])
        R = float(np.hypot(sx, sy))
        print(f"  W={W:5.0f}s detr : {R:.3f}   ({R/full_R_wf:.2f}x ref)")

    print("\n--- Total sigma_R = sqrt(R_LF^2 + R_WF^2), median over seeds ---")
    full_total = float(np.hypot(full_R_lf, full_R_wf))
    print(f"  full-record ref : {full_total:.3f}  (cf brucon ensemble truth 0.840)")
    for W in WINDOWS_S:
        sx_lf = np.median(results[("SurgeDev", "detr", W)])
        sy_lf = np.median(results[("SwayDev", "detr", W)])
        sx_wf = np.median(results[("xHf", "detr", W)])
        sy_wf = np.median(results[("yHf", "detr", W)])
        R_lf = np.hypot(sx_lf, sy_lf)
        R_wf = np.hypot(sx_wf, sy_wf)
        R = float(np.hypot(R_lf, R_wf))
        print(f"  W={W:5.0f}s detr : {R:.3f}  ({R/full_total:.2f}x full-record, "
              f"{R/0.840:.2f}x brucon-ensemble-truth)")

    plt.suptitle("Per-channel sigma vs sliding-window length, pwq30 ensemble", fontsize=12)
    plt.tight_layout()
    out = THIS / "sigma_window_sweep_pwq30.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
