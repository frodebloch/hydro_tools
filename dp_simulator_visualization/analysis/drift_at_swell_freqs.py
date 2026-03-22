"""Check drift coefficients at swell frequencies.

The key question: at the low-frequency swell range (ω ≈ 0.3-0.6 rad/s),
are the surge drift coefficients significant or negligible?
"""

import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent))
from drift_force_resolution import (
    jonswap, geometric_frequencies, frequency_steps,
    parse_drift_coefficients,
    PI, BEAUFORT, W_MIN, W_MAX,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    pdstrip_path = "/home/blofro/src/brucon/libs/simulator/vessel_model/test/config/csov_pdstrip.dat"
    output_dir = Path(__file__).parent / "output"

    ref_freqs, ref_drift = parse_drift_coefficients(pdstrip_path)

    # Print the actual drift coefficients
    print("=" * 60)
    print("Surge drift coefficients D(ω) — head seas, speed=0")
    print("=" * 60)
    print(f"  {'ω [rad/s]':>10s}  {'T [s]':>7s}  {'D [N/m²]':>12s}")
    print("-" * 40)
    for w, d in zip(ref_freqs, ref_drift):
        T = 2 * PI / w
        print(f"  {w:10.4f}  {T:7.1f}  {d:12.2f}")

    # Key swell peak frequencies for our test cases
    print("\n\nSwell peak frequencies and drift coefficients:")
    drift_interp = interp1d(
        ref_freqs[::-1], ref_drift[::-1],
        kind="cubic", fill_value="extrapolate"
    )

    swell_cases = [
        ("B: Tp=12s", 12.0),
        ("C: Tp=14s", 14.0),
        ("D: Tp=16s", 16.0),
        ("F: Tp=18s", 18.0),
    ]
    for label, tp in swell_cases:
        wp = 2 * PI / (tp * 0.834)
        D_at_peak = drift_interp(wp)
        # Compare to drift at typical wind sea peak
        D_at_wind = drift_interp(1.0)  # ~6s wind sea
        print(f"  {label}: ωp = {wp:.3f} rad/s, D = {D_at_peak:.1f} N/m², "
              f"ratio to D(1.0) = {abs(D_at_peak/D_at_wind):.2f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Top: drift coefficient vs frequency
    w_fine = np.linspace(W_MIN, W_MAX, 500)
    D_fine = drift_interp(w_fine)

    ax1.plot(w_fine, D_fine, "b-", lw=2, label="D(ω) surge drift coeff")
    ax1.plot(ref_freqs, ref_drift, "ko", ms=4, label="PdStrip data points")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("ω [rad/s]")
    ax1.set_ylabel("D(ω) [N/m²]")
    ax1.set_title("Surge drift coefficient — head seas")
    ax1.grid(True, alpha=0.3)

    # Mark swell peaks
    colors = ["C1", "C2", "C3", "C4"]
    for (label, tp), color in zip(swell_cases, colors):
        wp = 2 * PI / (tp * 0.834)
        D_val = drift_interp(wp)
        ax1.axvline(wp, color=color, ls="--", lw=1, alpha=0.7)
        ax1.annotate(f"{label}\nω={wp:.2f}\nD={D_val:.0f}",
                     (wp, D_val), fontsize=7, color=color,
                     xytext=(10, 20), textcoords="offset points")
    ax1.legend(fontsize=8)

    # Bottom: drift coefficient × spectrum for each dual case
    dual_cases = {
        "D: BF4 + Swell 3m/16s": {"hs_w": 1.5, "tp_w": 6.5, "hs_s": 3.0, "tp_s": 16.0},
        "C: BF5 + Swell 2.5m/14s": {"hs_w": 2.1, "tp_w": 7.5, "hs_s": 2.5, "tp_s": 14.0},
        "B: BF5 + Swell 1.5m/12s": {"hs_w": 2.1, "tp_w": 7.5, "hs_s": 1.5, "tp_s": 12.0},
    }

    for case_name, ss in dual_cases.items():
        S_wind = jonswap(w_fine, ss["hs_w"], ss["tp_w"])
        S_swell = jonswap(w_fine, ss["hs_s"], ss["tp_s"])
        S_total = S_wind + S_swell

        # D(ω)² × S(ω) is proportional to the mean drift contribution per unit bandwidth
        # D(ω)² × S(ω)² is proportional to SV drift contribution (diagonal-like)
        integrand = D_fine**2 * S_total
        ax2.plot(w_fine, integrand / integrand.max(), lw=1.5, label=case_name)

    ax2.set_xlabel("ω [rad/s]")
    ax2.set_ylabel("D(ω)² · S(ω) [normalized]")
    ax2.set_title("Drift force 'importance' — where should frequencies go?")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "16_drift_at_swell_freqs.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {output_dir / '16_drift_at_swell_freqs.png'}")


if __name__ == "__main__":
    main()
