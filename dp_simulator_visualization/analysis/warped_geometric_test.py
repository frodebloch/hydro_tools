"""Compare warped-geometric vs spectral-optimal vs standard geometric.

Warped-geometric preserves the incommensurate frequency ratio property
(guaranteeing non-repeating time series) while concentrating frequencies
near the spectral peak.

The idea: ω_i = ω_min · R^(g(i/(N-1)))  where g maps [0,1] -> [0,1]
monotonically but nonlinearly, spending more "index budget" near the
spectral peak.  All frequency ratios ω_{i+1}/ω_i = R^(g' * Δ) remain
irrational (non-repeating guaranteed).
"""

import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent))
from drift_force_resolution import (
    jonswap, geometric_frequencies, frequency_steps,
    parse_drift_coefficients, compute_sv_drift_variance_spectrum,
    vessel_transfer_sq,
    PI, G, BEAUFORT, ZETA, MU_BINS, MU_MAX, W_MIN, W_MAX,
)
from optimization_analysis import (
    optimal_drift_frequencies, compute_filtered_sigma,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Warped geometric: concentrate near spectral peak while keeping
# all ratios incommensurate
# ---------------------------------------------------------------------------

def warped_geometric_frequencies(n, w_min, w_max, w_center, sharpness=4.0):
    """Geometric frequencies with nonlinear index warping.

    Standard geometric: ω_i = w_min * R^(i/(N-1)),  R = w_max/w_min
    Warped:             ω_i = w_min * R^(g(i/(N-1)))

    where g(u) is a monotonic [0,1]->[0,1] map that spends more of its
    range near the normalized position of w_center.

    g(u) = F(u) / F(1)  where F(u) = integral_0^u  p(t) dt
    and p(t) is a peaked density centered at t_c = log(w_center/w_min)/log(R).

    Using p(t) = 1 + sharpness * exp(-(t - t_c)^2 / (2*sigma^2)):
    - sharpness=0 recovers standard geometric
    - larger sharpness concentrates more indices near w_center

    Parameters
    ----------
    n : int
        Number of frequencies.
    w_min, w_max : float
        Range bounds.
    w_center : float
        Frequency to concentrate near (typically spectral peak).
    sharpness : float
        How strongly to concentrate (0 = standard geometric).
    """
    R = w_max / w_min
    # Position of w_center in log-frequency space [0, 1]
    t_c = np.log(w_center / w_min) / np.log(R)
    t_c = np.clip(t_c, 0.05, 0.95)

    # Width of the concentration region (in normalized log-freq space)
    sigma = 0.15  # ~15% of the log-frequency range

    # Build the warping function g(u) by numerical integration
    n_fine = 10000
    u_fine = np.linspace(0, 1, n_fine)
    # Density: uniform base + Gaussian peak at t_c
    density = 1.0 + sharpness * np.exp(-0.5 * ((u_fine - t_c) / sigma) ** 2)
    # CDF (unnormalized)
    cdf = np.cumsum(density)
    cdf = cdf / cdf[-1]  # normalize to [0, 1]

    # Evaluate g at the N uniform index positions
    u_indices = np.linspace(0, 1, n)
    g_values = np.interp(u_indices, np.linspace(0, 1, n_fine), cdf)

    # Map to frequencies
    freqs = w_min * R ** g_values
    return np.sort(freqs)[::-1]  # descending


def warped_geometric_auto(n, w_min, w_max, hs, tp, sharpness=4.0):
    """Warped geometric with automatic centering on JONSWAP peak."""
    # JONSWAP peak frequency
    t1 = np.clip(tp * 0.834, 1.0, 50.0)
    w_peak = 2.0 * PI / t1  # approximate peak
    # Refine: find actual peak from spectrum
    w_fine = np.linspace(w_min, w_max, 5000)
    S = jonswap(w_fine, hs, tp)
    w_peak = w_fine[np.argmax(S)]
    return warped_geometric_frequencies(n, w_min, w_max, w_peak, sharpness)


# ---------------------------------------------------------------------------
# Test: non-repeating property
# ---------------------------------------------------------------------------

def check_incommensurate(freqs, label):
    """Check consecutive frequency ratios — are they all different?"""
    ratios = freqs[:-1] / freqs[1:]  # descending, so ratio > 1
    unique_ratios = len(set(np.round(ratios, 12)))
    all_unique = unique_ratios == len(ratios)
    ratio_range = (ratios.min(), ratios.max())
    print(f"  {label}: {len(ratios)} ratios, {unique_ratios} unique, "
          f"range [{ratio_range[0]:.6f}, {ratio_range[1]:.6f}], "
          f"all different: {all_unique}")
    return ratios


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main():
    pdstrip_path = "/home/blofro/src/brucon/libs/simulator/vessel_model/test/config/csov_pdstrip.dat"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse drift coefficients
    ref_freqs, ref_drift = parse_drift_coefficients(pdstrip_path)
    drift_interp = interp1d(
        ref_freqs[::-1], ref_drift[::-1],
        kind="cubic", fill_value="extrapolate"
    )

    omega_n = 0.06
    N = 35

    # Reference
    sigma_ref = {}
    for bf, ss in BEAUFORT.items():
        freqs_ref = geometric_frequencies(2000)
        sigma_ref[bf], _ = compute_filtered_sigma(
            freqs_ref, drift_interp, ss["hs"], ss["tp"], omega_n, ZETA
        )

    # Strategies to compare
    print("=" * 80)
    print(f"COMPARISON: N={N}, omega_n={omega_n}, zeta={ZETA}")
    print("=" * 80)

    # 1. Check ratio properties
    print("\nFrequency ratio analysis:")
    bf7 = BEAUFORT[7]

    freqs_geo = geometric_frequencies(N)
    ratios_geo = check_incommensurate(freqs_geo, "Geometric (standard)")

    for sharp in [2, 4, 6, 8, 12]:
        freqs_wg = warped_geometric_auto(N, W_MIN, W_MAX, bf7["hs"], bf7["tp"], sharp)
        check_incommensurate(freqs_wg, f"Warped-geo (s={sharp})")

    freqs_opt = optimal_drift_frequencies(N, W_MIN, W_MAX, bf7["hs"], bf7["tp"], omega_n, ZETA)
    check_incommensurate(freqs_opt, "Spectral-optimal")

    # 2. Error comparison across sea states and sharpness values
    print(f"\n{'Strategy':<35s}", end="")
    for bf in BEAUFORT:
        print(f"  BF{bf}", end="")
    print("    Mean|err|")

    # Standard geometric
    errors = []
    print(f"{'Geometric (current)':<35s}", end="")
    for bf, ss in BEAUFORT.items():
        sig, _ = compute_filtered_sigma(
            geometric_frequencies(N), drift_interp, ss["hs"], ss["tp"], omega_n, ZETA
        )
        err = (sig / sigma_ref[bf] - 1.0) * 100.0
        errors.append(abs(err))
        print(f" {err:+5.1f}%", end="")
    print(f"   {np.mean(errors):5.1f}%")

    # Warped geometric at various sharpness
    for sharp in [2, 3, 4, 6, 8, 12]:
        errors = []
        print(f"{'Warped-geo (s=' + str(sharp) + ')':<35s}", end="")
        for bf, ss in BEAUFORT.items():
            freqs = warped_geometric_auto(N, W_MIN, W_MAX, ss["hs"], ss["tp"], sharp)
            sig, _ = compute_filtered_sigma(
                freqs, drift_interp, ss["hs"], ss["tp"], omega_n, ZETA
            )
            err = (sig / sigma_ref[bf] - 1.0) * 100.0
            errors.append(abs(err))
            print(f" {err:+5.1f}%", end="")
        print(f"   {np.mean(errors):5.1f}%")

    # Spectral-optimal (the best we found before)
    errors = []
    print(f"{'Spectral-optimal':<35s}", end="")
    for bf, ss in BEAUFORT.items():
        freqs = optimal_drift_frequencies(N, W_MIN, W_MAX, ss["hs"], ss["tp"], omega_n, ZETA)
        sig, _ = compute_filtered_sigma(
            freqs, drift_interp, ss["hs"], ss["tp"], omega_n, ZETA
        )
        err = (sig / sigma_ref[bf] - 1.0) * 100.0
        errors.append(abs(err))
        print(f" {err:+5.1f}%", end="")
    print(f"   {np.mean(errors):5.1f}%")

    # 3. Also test with more frequencies
    print(f"\n\nSame comparison at N=50:")
    N2 = 50
    print(f"{'Strategy':<35s}", end="")
    for bf in BEAUFORT:
        print(f"  BF{bf}", end="")
    print("    Mean|err|")

    print(f"{'Geometric':<35s}", end="")
    errors = []
    for bf, ss in BEAUFORT.items():
        sig, _ = compute_filtered_sigma(
            geometric_frequencies(N2), drift_interp, ss["hs"], ss["tp"], omega_n, ZETA
        )
        err = (sig / sigma_ref[bf] - 1.0) * 100.0
        errors.append(abs(err))
        print(f" {err:+5.1f}%", end="")
    print(f"   {np.mean(errors):5.1f}%")

    for sharp in [4, 6, 8]:
        errors = []
        print(f"{'Warped-geo (s=' + str(sharp) + ')':<35s}", end="")
        for bf, ss in BEAUFORT.items():
            freqs = warped_geometric_auto(N2, W_MIN, W_MAX, ss["hs"], ss["tp"], sharp)
            sig, _ = compute_filtered_sigma(
                freqs, drift_interp, ss["hs"], ss["tp"], omega_n, ZETA
            )
            err = (sig / sigma_ref[bf] - 1.0) * 100.0
            errors.append(abs(err))
            print(f" {err:+5.1f}%", end="")
        print(f"   {np.mean(errors):5.1f}%")

    errors = []
    print(f"{'Spectral-optimal':<35s}", end="")
    for bf, ss in BEAUFORT.items():
        freqs = optimal_drift_frequencies(N2, W_MIN, W_MAX, ss["hs"], ss["tp"], omega_n, ZETA)
        sig, _ = compute_filtered_sigma(
            freqs, drift_interp, ss["hs"], ss["tp"], omega_n, ZETA
        )
        err = (sig / sigma_ref[bf] - 1.0) * 100.0
        errors.append(abs(err))
        print(f" {err:+5.1f}%", end="")
    print(f"   {np.mean(errors):5.1f}%")

    # 4. Plot frequency distributions for visual comparison
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    ss = BEAUFORT[7]
    w_fine = np.linspace(W_MIN, W_MAX, 2000)
    S_fine = jonswap(w_fine, ss["hs"], ss["tp"])
    S_norm = S_fine / S_fine.max()

    for ax_idx, (N_plot, title) in enumerate([(35, "N=35"), (50, "N=50"), (70, "N=70")]):
        ax = axes[ax_idx]

        # Spectrum background
        ax.fill_between(w_fine, S_norm, alpha=0.15, color="blue", label="JONSWAP (normalized)")

        # Frequency placements
        configs = [
            ("Geometric", geometric_frequencies(N_plot), "C0", "|", 15),
            ("Warped-geo (s=6)", warped_geometric_auto(N_plot, W_MIN, W_MAX, ss["hs"], ss["tp"], 6), "C1", "|", 15),
            ("Spectral-optimal", optimal_drift_frequencies(N_plot, W_MIN, W_MAX, ss["hs"], ss["tp"], omega_n, ZETA), "C3", "|", 15),
        ]

        y_offsets = [0.85, 0.55, 0.25]
        for (label, freqs, color, marker, ms), y_off in zip(configs, y_offsets):
            ax.plot(freqs, np.full_like(freqs, y_off), marker, color=color,
                    ms=ms, mew=1.5, label=label)
            # Show local spacing
            dw = np.diff(freqs[::-1])  # ascending order, positive dw
            ax.text(W_MAX + 0.02, y_off, f"Δω: {dw.min():.4f}–{dw.max():.4f}",
                    fontsize=7, va="center")

        ax.set_xlim(W_MIN - 0.05, W_MAX + 0.3)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("ω [rad/s]")
        ax.set_title(f"{title} — BF7 (Hs={ss['hs']}m, Tp={ss['tp']}s)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "10_warped_geometric_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {output_dir / '10_warped_geometric_comparison.png'}")

    # 5. Repeat period analysis
    print("\n\nRepeat period analysis:")
    print("(Time until wave elevation repeats to within 1% correlation)")
    print("For geometric: T_repeat = LCM of all periods — effectively infinite")
    print("For warped-geometric: same — all ratios are irrational")
    print("For spectral-optimal from CDF: ratios are numerically arbitrary,")
    print("  effectively incommensurate, but not guaranteed by construction")


if __name__ == "__main__":
    main()
