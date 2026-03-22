"""Dual-spectrum analysis: wind sea + swell.

How does spectral-optimal placement handle combined sea states?
Compare single JONSWAP vs wind+swell combination.
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
    PI, BEAUFORT, ZETA, MU_BINS, MU_MAX, W_MIN, W_MAX,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def jonswap_dual(w, hs_wind, tp_wind, hs_swell, tp_swell, gamma_wind=3.3, gamma_swell=5.0):
    """Combined wind sea + swell spectrum (sum of two JONSWAP).

    Swell typically has higher gamma (narrower peak).
    Returns total S(ω) = S_wind(ω) + S_swell(ω).
    """
    S_w = jonswap(w, hs_wind, tp_wind, gamma_wind)
    S_s = jonswap(w, hs_swell, tp_swell, gamma_swell)
    return S_w + S_s


def optimal_drift_frequencies_dual(n, w_min, w_max, hs_w, tp_w, hs_s, tp_s,
                                    omega_n, gamma_w=3.3, gamma_s=5.0):
    """Spectral-optimal placement for dual-peaked spectrum."""
    w_fine = np.linspace(w_min, w_max, 10000)
    S = jonswap_dual(w_fine, hs_w, tp_w, hs_s, tp_s, gamma_w, gamma_s)

    S_shifted = jonswap_dual(w_fine + omega_n, hs_w, tp_w, hs_s, tp_s, gamma_w, gamma_s)
    weight = np.sqrt(S * S_shifted + S * S)

    cdf = np.cumsum(weight)
    cdf = cdf / cdf[-1]

    quantiles = np.linspace(0.0, 1.0, n + 2)[1:-1]
    w_opt = np.interp(quantiles, cdf, w_fine)
    w_opt[0] = max(w_opt[0], w_min)
    w_opt[-1] = min(w_opt[-1], w_max)

    return np.sort(w_opt)[::-1]


def compute_sv_variance_dual(freqs, drift_interp, hs_w, tp_w, hs_s, tp_s,
                              gamma_w=3.3, gamma_s=5.0):
    """Compute drift force variance for dual-peaked spectrum."""
    n = len(freqs)
    dw = frequency_steps(freqs)
    S = jonswap_dual(freqs, hs_w, tp_w, hs_s, tp_s, gamma_w, gamma_s)
    a_sq = 2.0 * S * dw
    D = drift_interp(freqs)

    d_mu = MU_MAX / MU_BINS
    mu_centers = np.linspace(d_mu/2, MU_MAX - d_mu/2, MU_BINS)
    S_F = np.zeros(MU_BINS)
    mean_drift = np.sum(D * a_sq)

    total_var = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            mu = abs(freqs[j] - freqs[i])
            T_ij = 0.5 * (D[i] + D[j])
            contrib = 2.0 * T_ij**2 * a_sq[i] * a_sq[j]
            total_var += contrib
            bin_idx = int(mu / d_mu)
            if 0 <= bin_idx < MU_BINS:
                S_F[bin_idx] += contrib

    S_F /= d_mu
    return mu_centers, S_F, total_var, mean_drift


def compute_filtered_sigma_dual(freqs, drift_interp, hs_w, tp_w, hs_s, tp_s,
                                 omega_n, zeta, gamma_w=3.3, gamma_s=5.0):
    """Vessel-filtered RMS slowly-varying drift for dual spectrum."""
    mu_c, S_F, _, mean_drift = compute_sv_variance_dual(
        freqs, drift_interp, hs_w, tp_w, hs_s, tp_s, gamma_w, gamma_s
    )
    H2 = vessel_transfer_sq(mu_c, omega_n, zeta)
    d_mu = MU_MAX / MU_BINS
    filt_var = np.sum(S_F * H2) * d_mu
    return np.sqrt(filt_var), mean_drift


def main():
    pdstrip_path = "/home/blofro/src/brucon/libs/simulator/vessel_model/test/config/csov_pdstrip.dat"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_freqs, ref_drift = parse_drift_coefficients(pdstrip_path)
    drift_interp = interp1d(
        ref_freqs[::-1], ref_drift[::-1],
        kind="cubic", fill_value="extrapolate"
    )

    omega_n = 0.06

    # ===================================================================
    # Define combined sea states
    # ===================================================================
    DUAL_SEAS = {
        "A: Wind only BF6": {
            "hs_w": 3.1, "tp_w": 8.5, "hs_s": 0.0, "tp_s": 12.0,
        },
        "B: Wind BF5 + Swell 1.5m/12s": {
            "hs_w": 2.1, "tp_w": 7.5, "hs_s": 1.5, "tp_s": 12.0,
        },
        "C: Wind BF5 + Swell 2.5m/14s": {
            "hs_w": 2.1, "tp_w": 7.5, "hs_s": 2.5, "tp_s": 14.0,
        },
        "D: Wind BF4 + Swell 3.0m/16s": {
            "hs_w": 1.5, "tp_w": 6.5, "hs_s": 3.0, "tp_s": 16.0,
        },
        "E: Wind BF6 + Swell 2.0m/12s": {
            "hs_w": 3.1, "tp_w": 8.5, "hs_s": 2.0, "tp_s": 12.0,
        },
        "F: Wind BF3 + Swell 1.0m/18s": {
            "hs_w": 0.8, "tp_w": 5.5, "hs_s": 1.0, "tp_s": 18.0,
        },
    }

    # ===================================================================
    # Analysis
    # ===================================================================
    print("=" * 80)
    print("DUAL SPECTRUM ANALYSIS: Wind sea + swell")
    print(f"ω_n = {omega_n} rad/s, ζ = {ZETA}")
    print("=" * 80)

    N_values = [35, 50, 70]

    # Results storage for summary
    results = {}

    for case_name, ss in DUAL_SEAS.items():
        hs_w, tp_w = ss["hs_w"], ss["tp_w"]
        hs_s, tp_s = ss["hs_s"], ss["tp_s"]

        # Combined Hs
        hs_comb = np.sqrt(hs_w**2 + hs_s**2)

        # Peak frequencies
        wp_wind = 2 * PI / (tp_w * 0.834) if hs_w > 0 else 0
        wp_swell = 2 * PI / (tp_s * 0.834) if hs_s > 0 else 0
        sep = abs(wp_wind - wp_swell) if hs_s > 0 else 0

        print(f"\n  {case_name}")
        print(f"    Wind: Hs={hs_w}m, Tp={tp_w}s (ωp={wp_wind:.3f})")
        if hs_s > 0:
            print(f"    Swell: Hs={hs_s}m, Tp={tp_s}s (ωp={wp_swell:.3f})")
            print(f"    Peak separation: {sep:.3f} rad/s")
        print(f"    Combined Hs: {hs_comb:.2f}m")

        # Reference: N=2000 geometric
        sig_ref, mean_ref = compute_filtered_sigma_dual(
            geometric_frequencies(2000), drift_interp,
            hs_w, tp_w, hs_s, tp_s, omega_n, ZETA
        )

        results[case_name] = {"ref": sig_ref}

        print(f"    Reference σ (N=2000): {sig_ref:.2f} kN")

        for N in N_values:
            # Geometric
            sig_geo, _ = compute_filtered_sigma_dual(
                geometric_frequencies(N), drift_interp,
                hs_w, tp_w, hs_s, tp_s, omega_n, ZETA
            )
            err_geo = (sig_geo / sig_ref - 1.0) * 100.0

            # Spectral-optimal (single JONSWAP — uses only wind sea params)
            # This is wrong for dual spectra but shows what happens if you
            # only optimize for one peak
            if hs_w > 0:
                from optimization_analysis import optimal_drift_frequencies
                sig_opt_single, _ = compute_filtered_sigma_dual(
                    optimal_drift_frequencies(N, W_MIN, W_MAX, hs_w, tp_w, omega_n, ZETA),
                    drift_interp, hs_w, tp_w, hs_s, tp_s, omega_n, ZETA
                )
                err_opt_single = (sig_opt_single / sig_ref - 1.0) * 100.0
            else:
                err_opt_single = float('nan')

            # Spectral-optimal (dual spectrum aware)
            sig_opt_dual, _ = compute_filtered_sigma_dual(
                optimal_drift_frequencies_dual(N, W_MIN, W_MAX,
                                               hs_w, tp_w, hs_s, tp_s, omega_n),
                drift_interp, hs_w, tp_w, hs_s, tp_s, omega_n, ZETA
            )
            err_opt_dual = (sig_opt_dual / sig_ref - 1.0) * 100.0

            print(f"    N={N:3d}: geo {err_geo:+6.1f}%  "
                  f"opt-wind-only {err_opt_single:+6.1f}%  "
                  f"opt-dual {err_opt_dual:+6.1f}%")

    # ===================================================================
    # Plot: spectra and frequency placements for key cases
    # ===================================================================
    plot_cases = ["B: Wind BF5 + Swell 1.5m/12s",
                  "C: Wind BF5 + Swell 2.5m/14s",
                  "D: Wind BF4 + Swell 3.0m/16s",
                  "F: Wind BF3 + Swell 1.0m/18s"]

    fig, axes = plt.subplots(len(plot_cases), 1, figsize=(14, 3.5 * len(plot_cases)))

    w_fine = np.linspace(W_MIN, W_MAX, 2000)
    N_plot = 35

    for ax_idx, case_name in enumerate(plot_cases):
        ss = DUAL_SEAS[case_name]
        ax = axes[ax_idx]

        # Spectrum
        S_total = jonswap_dual(w_fine, ss["hs_w"], ss["tp_w"],
                               ss["hs_s"], ss["tp_s"])
        S_wind = jonswap(w_fine, ss["hs_w"], ss["tp_w"])
        S_swell = jonswap(w_fine, ss["hs_s"], ss["tp_s"]) if ss["hs_s"] > 0 else np.zeros_like(w_fine)

        S_max = S_total.max()
        ax.fill_between(w_fine, S_total / S_max, alpha=0.15, color="blue",
                        label="Total spectrum")
        ax.plot(w_fine, S_wind / S_max, "b--", lw=0.8, alpha=0.5, label="Wind sea")
        if ss["hs_s"] > 0:
            ax.plot(w_fine, S_swell / S_max, "r--", lw=0.8, alpha=0.5, label="Swell")

        # Frequency placements
        f_geo = geometric_frequencies(N_plot)
        f_opt_dual = optimal_drift_frequencies_dual(
            N_plot, W_MIN, W_MAX,
            ss["hs_w"], ss["tp_w"], ss["hs_s"], ss["tp_s"], omega_n
        )

        ax.plot(f_geo, np.full_like(f_geo, 0.85), "|", color="C0",
                ms=12, mew=1.5, label="Geometric")
        ax.plot(f_opt_dual, np.full_like(f_opt_dual, 0.65), "|", color="C3",
                ms=12, mew=1.5, label="Spectral-optimal (dual)")

        ax.set_xlabel("ω [rad/s]")
        ax.set_title(case_name)
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 1.05)

    fig.suptitle(f"Dual Spectrum: Frequency Placement (N={N_plot})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "15_dual_spectrum.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {output_dir / '15_dual_spectrum.png'}")

    # ===================================================================
    # Key question: how many freqs does each peak "get"?
    # ===================================================================
    print("\n" + "=" * 80)
    print("FREQUENCY BUDGET ALLOCATION PER PEAK (N=35, spectral-optimal dual)")
    print("=" * 80)

    for case_name, ss in DUAL_SEAS.items():
        if ss["hs_s"] == 0:
            continue

        freqs = optimal_drift_frequencies_dual(
            35, W_MIN, W_MAX,
            ss["hs_w"], ss["tp_w"], ss["hs_s"], ss["tp_s"], omega_n
        )

        wp_wind = 2 * PI / (ss["tp_w"] * 0.834)
        wp_swell = 2 * PI / (ss["tp_s"] * 0.834)
        # Midpoint between peaks
        w_mid = 0.5 * (wp_wind + wp_swell)

        n_swell = np.sum(freqs < w_mid)
        n_wind = np.sum(freqs >= w_mid)

        # Energy fractions
        S_at_freqs = jonswap_dual(freqs, ss["hs_w"], ss["tp_w"],
                                   ss["hs_s"], ss["tp_s"])
        S_wind_at = jonswap(freqs, ss["hs_w"], ss["tp_w"])
        S_swell_at = jonswap(freqs, ss["hs_s"], ss["tp_s"]) if ss["hs_s"] > 0 else 0

        total_hs_sq = ss["hs_w"]**2 + ss["hs_s"]**2
        wind_frac = ss["hs_w"]**2 / total_hs_sq * 100
        swell_frac = ss["hs_s"]**2 / total_hs_sq * 100

        print(f"  {case_name}")
        print(f"    Wind peak:  {n_wind:2d} freqs ({n_wind/35*100:.0f}%), "
              f"energy fraction: {wind_frac:.0f}%")
        print(f"    Swell peak: {n_swell:2d} freqs ({n_swell/35*100:.0f}%), "
              f"energy fraction: {swell_frac:.0f}%")


if __name__ == "__main__":
    main()
