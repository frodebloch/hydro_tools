"""Practical repeat period analysis — what actually matters.

The pairwise rational approximation test is too strict. What matters for
operability analysis is: how long until the ENTIRE wave field approximately
repeats? This requires ALL phase angles ωᵢ·T to simultaneously be near
multiples of 2π.

For geometric spacing ωᵢ = ω₀·r^i, the wave field repeats when r^i·T
is near-integer (in units of 2π/ω₀) for all i. Since r is irrational,
the orbit (r^0, r^1, ..., r^{N-1}) on the N-torus is equidistributed
(Weyl's theorem), meaning there IS no short repeat — the field is
ergodic.

For spectral-optimal: the frequencies are arbitrary reals. The question
is whether the vector (ω₁, ω₂, ..., ωN) is "linearly independent over Q"
in a practical sense.

The RIGHT test: simulate the actual wave field and measure autocorrelation.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent))
from drift_force_resolution import (
    jonswap, geometric_frequencies, frequency_steps,
    PI, BEAUFORT, W_MIN, W_MAX,
)
from optimization_analysis import optimal_drift_frequencies
from envelope_weighted_test import (
    parse_all_drift_coefficients, compute_envelope_importance,
    envelope_weighted_optimal,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PDSTRIP_PATH = Path("/home/blofro/src/brucon/libs/simulator/vessel_model/test/config/csov_pdstrip.dat")


def wave_autocorrelation(freqs, hs, tp, lags, seed=42):
    """Compute the autocorrelation of a wave elevation time series.

    η(t) = Σ aᵢ cos(ωᵢ t + φᵢ)

    R(τ) = <η(t)·η(t+τ)> / <η²>

    For independent random phases, the theoretical autocorrelation is:
        R(τ) = Σ aᵢ² cos(ωᵢ τ) / Σ aᵢ²

    This is deterministic (no phase dependence) and gives the envelope
    of how "repeated" the signal looks at each lag.
    """
    dw = frequency_steps(freqs)
    S = jonswap(freqs, hs, tp)
    a_sq = 2.0 * S * dw

    total_energy = np.sum(a_sq)
    if total_energy < 1e-30:
        return np.zeros_like(lags)

    R = np.zeros(len(lags))
    for k, tau in enumerate(lags):
        R[k] = np.sum(a_sq * np.cos(freqs * tau)) / total_energy

    return R


def find_near_repeats(R, lags, threshold=0.9):
    """Find lag times where autocorrelation exceeds threshold."""
    # Skip the zero-lag peak
    idx = np.where((R[1:] > threshold) & (lags[1:] > 60))[0] + 1
    return lags[idx], R[idx]


def main():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    omega_n = 0.06

    # Time lags: scan from 0 to 6 hours at 1-second resolution
    T_max = 6 * 3600  # 6 hours
    dt = 1.0
    lags = np.arange(0, T_max, dt)

    ss = BEAUFORT[7]

    # Load PdStrip data for envelope-weighted grids
    pd_freqs, pd_dirs, surge_d, sway_d, yaw_d = parse_all_drift_coefficients(PDSTRIP_PATH)
    D2_envelope, _, _, _ = compute_envelope_importance(pd_freqs, pd_dirs, surge_d, sway_d, yaw_d)
    D2_interp = interp1d(pd_freqs[::-1], D2_envelope[::-1], kind='linear',
                         bounds_error=False, fill_value=0.0)

    grids = {
        "Geometric N=35": geometric_frequencies(35),
        "Geometric N=70": geometric_frequencies(70),
        "Spectral-optimal N=35": optimal_drift_frequencies(
            35, W_MIN, W_MAX, ss["hs"], ss["tp"], omega_n, 0.95),
        "Spectral-optimal N=70": optimal_drift_frequencies(
            70, W_MIN, W_MAX, ss["hs"], ss["tp"], omega_n, 0.95),
        "Envelope-weighted N=35": envelope_weighted_optimal(
            35, W_MIN, W_MAX, D2_interp, ss["hs"], ss["tp"], omega_n),
        "Envelope-weighted N=70": envelope_weighted_optimal(
            70, W_MIN, W_MAX, D2_interp, ss["hs"], ss["tp"], omega_n),
    }

    fig, axes = plt.subplots(len(grids), 1, figsize=(16, 3.5 * len(grids)),
                             sharex=True)

    print("=" * 80)
    print(f"AUTOCORRELATION ANALYSIS — BF7 (Hs={ss['hs']}m, Tp={ss['tp']}s)")
    print(f"Scanning 0 to {T_max/3600:.0f} hours at {dt}s resolution")
    print("=" * 80)

    for ax_idx, (name, freqs) in enumerate(grids.items()):
        print(f"\n  Computing {name}...")
        R = wave_autocorrelation(freqs, ss["hs"], ss["tp"], lags)

        ax = axes[ax_idx]
        ax.plot(lags / 3600, R, lw=0.3, color="C0")
        ax.set_ylabel("R(τ)")
        ax.set_title(name)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-1.05, 1.05)

        # Find peaks
        t_near, r_near = find_near_repeats(R, lags, threshold=0.5)
        if len(t_near) > 0:
            print(f"    Near-repeats (R > 0.5): {len(t_near)} found")
            print(f"    First at T = {t_near[0]:.0f}s ({t_near[0]/3600:.2f}h), R = {r_near[0]:.3f}")
            ax.axhline(0.5, color="red", ls="--", lw=0.5, alpha=0.5)
            for t, r in zip(t_near[:5], r_near[:5]):
                ax.annotate(f"{t/3600:.2f}h\nR={r:.2f}",
                           (t/3600, r), fontsize=6, ha="center")
        else:
            print(f"    No near-repeats (R > 0.5) in {T_max/3600:.0f} hours")

        # Max autocorrelation after 1 minute
        R_after_1min = R[lags > 60]
        lags_after_1min = lags[lags > 60]
        max_R = np.max(np.abs(R_after_1min))
        max_T = lags_after_1min[np.argmax(np.abs(R_after_1min))]
        print(f"    Max |R(τ)| for τ > 60s: {max_R:.4f} at T = {max_T:.0f}s ({max_T/3600:.2f}h)")

        # 3-hour mark
        ax.axvline(3.0, color="green", ls="--", lw=1, alpha=0.5, label="3h mark")

    axes[-1].set_xlabel("Lag τ [hours]")
    fig.tight_layout()
    fig.savefig(output_dir / "13_autocorrelation.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {output_dir / '13_autocorrelation.png'}")

    # ----- Zoomed view of first 30 minutes -----
    fig2, axes2 = plt.subplots(len(grids), 1, figsize=(16, 3.5 * len(grids)),
                                sharex=True)
    lags_short = np.arange(0, 1800, 0.5)

    for ax_idx, (name, freqs) in enumerate(grids.items()):
        R = wave_autocorrelation(freqs, ss["hs"], ss["tp"], lags_short)
        ax = axes2[ax_idx]
        ax.plot(lags_short / 60, R, lw=0.5, color="C0")
        ax.set_ylabel("R(τ)")
        ax.set_title(f"{name} — first 30 min")
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="k", lw=0.3)

    axes2[-1].set_xlabel("Lag τ [minutes]")
    fig2.tight_layout()
    fig2.savefig(output_dir / "14_autocorrelation_zoom.png", dpi=150)
    plt.close(fig2)
    print(f"Plot saved: {output_dir / '14_autocorrelation_zoom.png'}")


if __name__ == "__main__":
    main()
