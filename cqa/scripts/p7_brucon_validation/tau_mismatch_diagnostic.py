"""Candidate (2): does observer-vs-vessel tau mismatch poison y_LF?

Background
----------
With `use_tau_feedback: false` (CSOV default) the observer integrates
`allocated_tau` (post-allocator commanded force) into its velocity model,
NOT `feedback_tau` (the actual force after thruster rate limits, saturation,
allocator optimisation slack). If

    delta_tau(t) := allocated_tau(t) - feedback_tau(t)

has a non-zero mean or sub-mHz structure, the observer's velocity model
slowly accumulates an error which propagates into y_LF (the controller's
estimated position) but NOT into the vessel's true motion. This would
manifest as a low-frequency wander in sigma_y_LF that the LTI sandbox
(which assumes commanded == actual) cannot reproduce.

Test (data-only, no new sims)
----------------------------
For each of the 30 long-run seeds in `work_long/`, in the late window
[1500, 3000] s, extract:

  - AllocTauSway   [kN]   (commanded by allocator)
  - FbTauSway      [kN]   (sum of actual thruster YForceFeedback, post-rate-limit)
  - Ty             [kN]   (force on vessel from simulator -- usually == FbTau
                            modulo numerical subtleties)
  - DriftY         [kN]   (the slow-drift forcing this is supposed to balance)
  - sway_LF (body) [m]    (the output we care about)

Compute per-seed:
  - mean(AllocTau - FbTau)         <- DC observer/vessel mismatch
  - std(AllocTau - FbTau)
  - sigma_y_LF
  - corr(mean(delta_tau), sigma_y_LF) across seeds

Compute ensemble:
  - PSD of delta_tau
  - is there structure in f < 0.002 Hz?
  - coherence between delta_tau and sway_LF

Decision
--------
  - if mean(delta) > 5 kN typical and correlates with sigma_y_LF: candidate
    (2) is the mechanism; the fix is to either turn use_tau_feedback on or
    model the rate-limit transfer in the sandbox observer.
  - if mean(delta) ~ 0 kN AND PSD of delta has no sub-mHz structure: refute
    candidate (2); move on to (1) or (3).

Outputs
-------
  - PNG: tau_mismatch_diagnostic.png  (gitignored)
  - Console: per-seed table + ensemble correlation
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.signal  # noqa: E402

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from validate_sandbox_timeseries import (  # noqa: E402
    load_seed_csv,
    project_ned_to_body,
)

WORK_DIR = THIS / "work_long"
TAG = "pwo_long"
SEEDS = list(range(1000, 1030))
WINDOW = (1500.0, 3000.0)
DT = 0.1
NPERSEG = 5000          # 500 s segment
OUT_PNG = THIS / "tau_mismatch_diagnostic.png"


def get_seed_traces(seed: int) -> dict[str, np.ndarray]:
    """Return relevant brucon channels for one seed in WINDOW, demeaned where useful.

    Tau channels are in kN per analysis.md sect. 12.20.2 (Channel semantics
    correction). We KEEP them in kN for readability and convert to N only when
    feeding into LTI predictions.
    """
    seed_dir = WORK_DIR / f"{TAG}_seed{seed}"
    cols = load_seed_csv(seed_dir)
    t = cols["t"]
    mask = (t >= WINDOW[0]) & (t <= WINDOW[1])
    out: dict[str, np.ndarray] = {"t": t[mask] - t[mask][0]}
    for k in ("AllocTauSway", "FbTauSway", "Ty", "DriftY",
              "OrderTauSway"):
        out[k] = cols[k][mask]
    # body-frame sway_LF
    _, sway_total = project_ned_to_body(cols["x"][mask], cols["y"][mask],
                                        cols["heading"][mask])
    sway_lf = sway_total - cols["yHf"][mask]
    sway_lf -= sway_lf.mean()
    out["sway_lf"] = sway_lf
    out["delta_tau"] = out["AllocTauSway"] - out["FbTauSway"]
    return out


def main() -> None:
    print("=" * 78)
    print("Tau mismatch diagnostic: AllocTauSway vs FbTauSway in the long window")
    print(f"  window: {WINDOW} s,  seeds: {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})")
    print("=" * 78)
    print()

    rows = []
    delta_sums = None
    sway_sums = None
    csd_sums = None
    f_grid = None
    fs = 1.0 / DT
    for s in SEEDS:
        try:
            d = get_seed_traces(s)
        except Exception as e:
            print(f"  seed {s}: skip ({e})")
            continue

        mean_delta = float(d["delta_tau"].mean())
        std_delta = float(d["delta_tau"].std(ddof=1))
        mean_alloc = float(d["AllocTauSway"].mean())
        mean_fb = float(d["FbTauSway"].mean())
        mean_drift = float(d["DriftY"].mean())
        sigma_y = float(d["sway_lf"].std(ddof=1))

        rows.append(dict(
            seed=s, mean_alloc=mean_alloc, mean_fb=mean_fb, mean_drift=mean_drift,
            mean_delta=mean_delta, std_delta=std_delta, sigma_y=sigma_y,
        ))

        # PSD accumulation (delta_tau in kN, sway_lf in m)
        f, S_dd = scipy.signal.welch(d["delta_tau"], fs=fs, nperseg=NPERSEG,
                                     scaling="density")
        _, S_yy = scipy.signal.welch(d["sway_lf"], fs=fs, nperseg=NPERSEG,
                                     scaling="density")
        _, S_dy = scipy.signal.csd(d["delta_tau"], d["sway_lf"], fs=fs,
                                   nperseg=NPERSEG, scaling="density")
        if delta_sums is None:
            delta_sums = np.zeros_like(S_dd)
            sway_sums = np.zeros_like(S_yy)
            csd_sums = np.zeros_like(S_dy, dtype=complex)
            f_grid = f
        delta_sums += S_dd
        sway_sums += S_yy
        csd_sums += S_dy

    n = len(rows)
    if n == 0:
        raise RuntimeError("no seeds processed")
    S_dd_avg = delta_sums / n
    S_yy_avg = sway_sums / n
    S_dy_avg = csd_sums / n
    coherence = (np.abs(S_dy_avg) ** 2) / (S_dd_avg * S_yy_avg + 1e-30)

    # --- per-seed table ---
    print(f"per-seed stats over {WINDOW} s ({n} seeds):")
    hdr = f"{'seed':>5} {'mean_alloc':>11} {'mean_fb':>11} {'mean_drift':>11} {'mean_delta':>11} {'std_delta':>11} {'sigma_y':>9}"
    print(hdr)
    print(f"{'':>5} {'[kN]':>11} {'[kN]':>11} {'[kN]':>11} {'[kN]':>11} {'[kN]':>11} {'[m]':>9}")
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['seed']:>5} {r['mean_alloc']:>11.2f} {r['mean_fb']:>11.2f} "
              f"{r['mean_drift']:>11.2f} {r['mean_delta']:>11.3f} {r['std_delta']:>11.2f} "
              f"{r['sigma_y']:>9.3f}")

    # --- aggregates ---
    means_alloc = np.array([r["mean_alloc"] for r in rows])
    means_fb = np.array([r["mean_fb"] for r in rows])
    means_drift = np.array([r["mean_drift"] for r in rows])
    means_delta = np.array([r["mean_delta"] for r in rows])
    stds_delta = np.array([r["std_delta"] for r in rows])
    sigmas_y = np.array([r["sigma_y"] for r in rows])

    print()
    print(f"ensemble across {n} seeds (mean +/- std-of-means):")
    print(f"  mean_alloc      = {means_alloc.mean():+8.2f}  +/- {means_alloc.std():.2f} kN")
    print(f"  mean_fb         = {means_fb.mean():+8.2f}  +/- {means_fb.std():.2f} kN")
    print(f"  mean_drift      = {means_drift.mean():+8.2f}  +/- {means_drift.std():.2f} kN")
    print(f"  mean_alloc-fb   = {means_delta.mean():+8.3f}  +/- {means_delta.std():.3f} kN")
    print(f"  std(alloc-fb)   = {stds_delta.mean():8.2f}  +/- {stds_delta.std():.2f} kN")
    print(f"  sigma_y_LF      = {sigmas_y.mean():8.3f}  +/- {sigmas_y.std():.3f} m")

    # --- correlations across seeds ---
    print()

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")

    print("seed-to-seed correlations:")
    print(f"  corr(mean_delta,  sigma_y) = {corr(means_delta, sigmas_y):+.3f}")
    print(f"  corr(|mean_delta|, sigma_y) = {corr(np.abs(means_delta), sigmas_y):+.3f}")
    print(f"  corr(std_delta,    sigma_y) = {corr(stds_delta, sigmas_y):+.3f}")
    print(f"  corr(|mean_drift|, sigma_y) = {corr(np.abs(means_drift), sigmas_y):+.3f}")
    print(f"  corr(mean_alloc,   sigma_y) = {corr(means_alloc, sigmas_y):+.3f}")

    # --- band-localised diagnostics ---
    print()
    print("band-localised PSD of delta_tau:")
    print(f"  {'band [Hz]':<14} {'sqrt(int S_dd) [kN]':>22} {'sqrt(int S_yy) [m]':>22} {'mean coh':>10}")
    bands = [(1e-4, 0.002), (0.002, 0.005), (0.005, 0.01), (0.01, 0.02),
             (0.02, 0.05), (0.05, 0.5)]
    for lo, hi in bands:
        m = (f_grid >= lo) & (f_grid <= hi)
        if not m.any():
            continue
        sig_d_band = float(np.sqrt(np.trapezoid(S_dd_avg[m], f_grid[m])))
        sig_y_band = float(np.sqrt(np.trapezoid(S_yy_avg[m], f_grid[m])))
        mean_coh = float(coherence[m].mean())
        print(f"  [{lo:.4f}-{hi:.3f}] {sig_d_band:>22.3f} {sig_y_band:>22.4f} {mean_coh:>10.3f}")

    # --- plot ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # (1) per-seed scatter
    ax = axes[0]
    sc = ax.scatter(means_delta, sigmas_y, c=means_drift, cmap="coolwarm",
                    s=60, edgecolor="k")
    fig.colorbar(sc, ax=ax, label="mean DriftY [kN]")
    ax.set_xlabel("mean(AllocTauSway - FbTauSway) [kN]")
    ax.set_ylabel("sigma_y_LF [m]")
    ax.set_title(f"Per-seed (n={n}) tau-mismatch DC vs y_LF spread\n"
                 f"corr(mean_delta, sigma_y) = {corr(means_delta, sigmas_y):+.3f}")
    ax.grid(alpha=0.3)
    for r in rows:
        ax.annotate(f"{r['seed']}", (r['mean_delta'], r['sigma_y']),
                    fontsize=6, alpha=0.6)

    # (2) ensemble PSDs
    ax = axes[1]
    f_plot = f_grid.copy(); f_plot[0] = f_plot[1]
    ax.loglog(f_plot, S_dd_avg, "C0-", lw=1.2,
              label="S_dd: PSD of (Alloc - Fb) [kN^2/Hz]")
    ax2 = ax.twinx()
    ax2.loglog(f_plot, S_yy_avg, "C3-", lw=1.2,
               label="S_yy: PSD of sway_LF [m^2/Hz]")
    ax.set_xlabel("frequency f [Hz]")
    ax.set_ylabel("S_dd [kN^2/Hz]", color="C0")
    ax2.set_ylabel("S_yy [m^2/Hz]", color="C3")
    ax.set_title("Ensemble PSDs of tau mismatch and y_LF")
    ax.axvspan(1e-4, 0.002, color="orange", alpha=0.15,
               label="sub-mHz band where most sigma_y_LF lives")
    ax.legend(loc="upper right", fontsize=8)
    ax2.legend(loc="lower left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1e-4, 0.5)

    # (3) coherence
    ax = axes[2]
    ax.semilogx(f_plot, coherence, "k-", lw=1.5)
    ax.axvspan(1e-4, 0.002, color="orange", alpha=0.15)
    ax.set_xlabel("frequency f [Hz]")
    ax.set_ylabel("coherence gamma^2(delta_tau, sway_LF)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1e-4, 0.5)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Coherence between (Alloc-Fb) and sway_LF\n"
                 "high coherence at sub-mHz => candidate (2) confirmed")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120)
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
