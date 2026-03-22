"""Optimization analysis: frequency placement and direction/frequency trade-offs.

Extends the baseline drift force resolution analysis to explore:
1. Sea-state-tailored frequency placement vs geometric spacing
2. Angular resolution vs frequency resolution trade-off
3. Optimal N_freq × N_dir at fixed computational budget

Usage:
    python optimization_analysis.py [--pdstrip PATH]
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import interp1d

# Import shared functions from baseline analysis
import sys
sys.path.insert(0, str(Path(__file__).parent))
from drift_force_resolution import (
    jonswap, geometric_frequencies, frequency_steps,
    parse_drift_coefficients, parse_drift_coefficients_all_directions,
    compute_sv_drift_variance_spectrum, compute_sv_drift_short_crested,
    vessel_transfer_sq, peak_factor,
    PI, G, BEAUFORT, ZETA, MU_BINS, MU_MAX, W_MIN, W_MAX,
    T_SIM, T_SIM_HOURS,
)


# ---------------------------------------------------------------------------
# Alternative frequency grid strategies
# ---------------------------------------------------------------------------

def spectral_peak_frequency(tp):
    """Peak frequency of JONSWAP spectrum."""
    t1 = np.clip(tp * 0.834, 1.0, 50.0)
    return 2.0 * PI / t1  # simplified; exact peak varies with gamma


def linear_frequencies(n, w_min=W_MIN, w_max=W_MAX):
    """Uniform linear spacing, descending."""
    return np.linspace(w_max, w_min, n)


def cosine_frequencies(n, w_min=W_MIN, w_max=W_MAX):
    """Cosine spacing — denser at both ends, sparser in middle."""
    theta = np.linspace(0, PI, n)
    w = 0.5 * (w_max + w_min) + 0.5 * (w_max - w_min) * np.cos(theta)
    return w  # already descending


def peak_concentrated_frequencies(n, w_min, w_max, w_peak, concentration=0.5):
    """Concentrate frequencies near the spectral peak.

    Uses a mapping that places `concentration` fraction of the frequencies
    within ±0.3 rad/s of the spectral peak, distributing the rest linearly.

    Parameters
    ----------
    n : int
        Number of frequencies.
    w_min, w_max : float
        Range bounds.
    w_peak : float
        Target spectral peak frequency.
    concentration : float
        Fraction of frequencies to place near peak (0-1).
    """
    # Use a tanh-based warping that concentrates points near w_peak
    # Map uniform u ∈ [0,1] to w ∈ [w_min, w_max] with concentration near w_peak
    u = np.linspace(0, 1, n)

    # Normalized peak location
    p = (w_peak - w_min) / (w_max - w_min)

    # Warping: stretch the region near p
    # Use a smooth mapping: w = w_min + (w_max-w_min) * g(u)
    # where g(u) maps [0,1] -> [0,1] with derivative concentrated near p
    alpha = 1.0 + concentration * 8.0  # controls concentration strength
    # Sigmoid-like warping centered at p
    def warp(u_val):
        # Beta function-like warping
        # Shift and scale a tanh to concentrate near p
        s = np.tanh(alpha * (u_val - 0.5)) / np.tanh(alpha * 0.5)
        return 0.5 * (1.0 + s)

    g = warp(u)
    # Rescale g to [0,1]
    g = (g - g.min()) / (g.max() - g.min())

    # Now shift the concentration center from 0.5 to p
    # Use a second transformation
    # Simple approach: use cumulative distribution of a beta-like function
    # peaked at p
    from scipy.stats import beta as beta_dist
    # Beta distribution parameters: mode at p, concentration controlled by alpha
    if 0.05 < p < 0.95:
        # Set alpha_b, beta_b so mode = p and concentration is high
        kappa = 2.0 + concentration * 20.0  # total concentration
        alpha_b = 1.0 + kappa * p
        beta_b = 1.0 + kappa * (1.0 - p)
        u_uniform = np.linspace(0, 1, n)
        w_normalized = beta_dist.ppf(
            u_uniform * 0.998 + 0.001,  # avoid 0 and 1
            alpha_b, beta_b
        )
    else:
        w_normalized = u  # fallback to uniform

    w = w_min + (w_max - w_min) * w_normalized
    return np.sort(w)[::-1]  # descending


def optimal_drift_frequencies(n, w_min, w_max, hs, tp, omega_n, zeta):
    """Frequency placement optimized for drift force variance accuracy.

    Strategy: distribute frequencies so that the density of frequency PAIRS
    at each difference frequency μ is proportional to the vessel-filtered
    drift force spectral density at μ.

    This is equivalent to concentrating frequencies where the product
    S(ω)·S(ω+μ_peak) is large — i.e., near the spectral peak, with
    spacing ~μ_peak (the vessel natural frequency) or smaller.

    Implementation: use the JONSWAP spectrum as a probability density
    to distribute frequency points. Frequencies are placed at quantiles
    of S(ω), ensuring dense placement where the spectrum has most energy.
    """
    # Fine reference grid for CDF computation
    w_fine = np.linspace(w_min, w_max, 10000)
    S = jonswap(w_fine, hs, tp)

    # Weight by S(ω) — this concentrates pairs near the spectral peak
    # where most drift force energy originates
    weight = S

    # Optional: additionally weight by proximity to producing pairs
    # at the vessel natural frequency μ = ω_n
    # For a pair (ω, ω+μ), both need spectral energy, and μ needs to be
    # near ω_n. So weight by S(ω) * S(ω + ω_n) approximately.
    S_shifted = jonswap(w_fine + omega_n, hs, tp)
    weight = np.sqrt(S * S_shifted + S * S)  # geometric mean + self-pair

    # Compute CDF and invert
    cdf = np.cumsum(weight)
    cdf = cdf / cdf[-1]

    # Place frequencies at uniform quantiles of this CDF
    quantiles = np.linspace(0.0, 1.0, n + 2)[1:-1]  # exclude endpoints
    w_opt = np.interp(quantiles, cdf, w_fine)

    # Ensure we cover the full range
    w_opt[0] = max(w_opt[0], w_min)
    w_opt[-1] = min(w_opt[-1], w_max)

    return np.sort(w_opt)[::-1]  # descending


# ---------------------------------------------------------------------------
# Compute variance for arbitrary frequency grid (reused utility)
# ---------------------------------------------------------------------------

def compute_filtered_sigma(freqs, drift_interp, hs, tp, omega_n, zeta):
    """Compute vessel-filtered RMS slowly-varying drift force for given grid."""
    D = drift_interp(freqs)
    mu_c, S_F, total_var, mean_drift = compute_sv_drift_variance_spectrum(
        freqs, D, hs, tp, MU_BINS, MU_MAX
    )
    H2 = vessel_transfer_sq(mu_c, omega_n, zeta)
    d_mu = MU_MAX / MU_BINS
    filt_var = np.sum(S_F * H2) * d_mu
    return np.sqrt(filt_var), mean_drift


def compute_filtered_sigma_sc(freqs, drift_interp_2d, pdstrip_dirs,
                               hs, tp, spreading, wave_dir, omega_n, zeta):
    """Compute vessel-filtered RMS for short-crested seas."""
    n_dir = len(pdstrip_dirs)
    drift_2d = np.zeros((len(freqs), n_dir))
    for j, d in enumerate(pdstrip_dirs):
        drift_2d[:, j] = drift_interp_2d[d](freqs)

    mu_c, S_F, total_var, mean_drift = compute_sv_drift_short_crested(
        freqs, drift_2d, pdstrip_dirs, hs, tp, spreading, wave_dir,
        MU_BINS, MU_MAX
    )
    H2 = vessel_transfer_sq(mu_c, omega_n, zeta)
    d_mu = MU_MAX / MU_BINS
    filt_var = np.sum(S_F * H2) * d_mu
    return np.sqrt(filt_var), mean_drift


# ---------------------------------------------------------------------------
# Direction subsampling
# ---------------------------------------------------------------------------

def subsample_directions(pdstrip_dirs, stride):
    """Subsample direction grid by stride (e.g. stride=2: 10° → 20°)."""
    return pdstrip_dirs[::stride]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_optimization_analysis(pdstrip_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse drift coefficients
    ref_freqs, ref_drift = parse_drift_coefficients(pdstrip_path)
    ref_freqs_all, pdstrip_dirs, ref_drift_2d = \
        parse_drift_coefficients_all_directions(pdstrip_path)

    drift_interp = interp1d(
        ref_freqs[::-1], ref_drift[::-1],
        kind="cubic", fill_value="extrapolate"
    )
    drift_interp_2d = {}
    for j, d in enumerate(pdstrip_dirs):
        drift_interp_2d[d] = interp1d(
            ref_freqs_all[::-1], ref_drift_2d[::-1, j],
            kind="cubic", fill_value="extrapolate"
        )

    omega_n = 0.06  # Representative DP bandwidth

    # Reference: N=2000 geometric grid
    sigma_ref = {}
    for bf, ss in BEAUFORT.items():
        freqs_ref = geometric_frequencies(2000)
        sigma_ref[bf], _ = compute_filtered_sigma(
            freqs_ref, drift_interp, ss["hs"], ss["tp"], omega_n, ZETA
        )

    # ===================================================================
    # ANALYSIS 1: Frequency placement strategies at fixed N
    # ===================================================================
    print("=" * 80)
    print("ANALYSIS 1: Frequency placement strategies")
    print("=" * 80)

    N_test = [35, 50, 70, 100]
    strategies = {
        "Geometric (current)": geometric_frequencies,
        "Linear": linear_frequencies,
        "Cosine": cosine_frequencies,
    }

    # For peak-concentrated and optimal, we need sea-state parameters
    bf_focus = 7  # BF 7 as representative
    ss = BEAUFORT[bf_focus]
    w_peak = spectral_peak_frequency(ss["tp"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax_idx, N in enumerate(N_test):
        ax = axes[ax_idx // 2][ax_idx % 2]

        for name, gen_func in strategies.items():
            errors = []
            for bf, ss_i in BEAUFORT.items():
                freqs = gen_func(N)
                sig, _ = compute_filtered_sigma(
                    freqs, drift_interp, ss_i["hs"], ss_i["tp"], omega_n, ZETA
                )
                err = (sig / sigma_ref[bf] - 1.0) * 100.0
                errors.append(err)
            ax.plot(list(BEAUFORT.keys()), errors, "o-", ms=4, label=name)

        # Peak-concentrated (tailored to each sea state)
        errors_peak = []
        for bf, ss_i in BEAUFORT.items():
            wp = spectral_peak_frequency(ss_i["tp"])
            freqs = peak_concentrated_frequencies(N, W_MIN, W_MAX, wp, 0.5)
            sig, _ = compute_filtered_sigma(
                freqs, drift_interp, ss_i["hs"], ss_i["tp"], omega_n, ZETA
            )
            err = (sig / sigma_ref[bf] - 1.0) * 100.0
            errors_peak.append(err)
        ax.plot(list(BEAUFORT.keys()), errors_peak, "s-", ms=5,
                label="Peak-concentrated (tailored)", color="red")

        # Optimal drift (tailored to each sea state)
        errors_opt = []
        for bf, ss_i in BEAUFORT.items():
            freqs = optimal_drift_frequencies(
                N, W_MIN, W_MAX, ss_i["hs"], ss_i["tp"], omega_n, ZETA
            )
            sig, _ = compute_filtered_sigma(
                freqs, drift_interp, ss_i["hs"], ss_i["tp"], omega_n, ZETA
            )
            err = (sig / sigma_ref[bf] - 1.0) * 100.0
            errors_opt.append(err)
        ax.plot(list(BEAUFORT.keys()), errors_opt, "D-", ms=5,
                label="Spectral-optimal (tailored)", color="darkred")

        ax.set_xlabel("Beaufort number")
        ax.set_ylabel("RMS force error vs reference [%]")
        ax.set_title(f"N = {N} frequencies")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", lw=0.5)

    fig.suptitle(f"Frequency Placement Strategies — ω_n={omega_n} rad/s, ζ={ZETA}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "08_placement_strategies.png", dpi=150)
    plt.close(fig)
    print("  Plot 8: placement strategies")

    # Print summary table
    print(f"\n  N=35, ω_n={omega_n} rad/s:")
    print(f"  {'Strategy':<30s}", end="")
    for bf in BEAUFORT:
        print(f"  BF{bf:d}", end="")
    print()
    for name, gen_func in strategies.items():
        print(f"  {name:<30s}", end="")
        for bf, ss_i in BEAUFORT.items():
            freqs = gen_func(35)
            sig, _ = compute_filtered_sigma(
                freqs, drift_interp, ss_i["hs"], ss_i["tp"], omega_n, ZETA
            )
            err = (sig / sigma_ref[bf] - 1.0) * 100.0
            print(f" {err:+5.1f}%", end="")
        print()
    # Peak-concentrated
    print(f"  {'Peak-concentrated':<30s}", end="")
    for bf, ss_i in BEAUFORT.items():
        wp = spectral_peak_frequency(ss_i["tp"])
        freqs = peak_concentrated_frequencies(35, W_MIN, W_MAX, wp, 0.5)
        sig, _ = compute_filtered_sigma(
            freqs, drift_interp, ss_i["hs"], ss_i["tp"], omega_n, ZETA
        )
        err = (sig / sigma_ref[bf] - 1.0) * 100.0
        print(f" {err:+5.1f}%", end="")
    print()
    # Optimal
    print(f"  {'Spectral-optimal':<30s}", end="")
    for bf, ss_i in BEAUFORT.items():
        freqs = optimal_drift_frequencies(
            35, W_MIN, W_MAX, ss_i["hs"], ss_i["tp"], omega_n, ZETA
        )
        sig, _ = compute_filtered_sigma(
            freqs, drift_interp, ss_i["hs"], ss_i["tp"], omega_n, ZETA
        )
        err = (sig / sigma_ref[bf] - 1.0) * 100.0
        print(f" {err:+5.1f}%", end="")
    print()

    # ===================================================================
    # ANALYSIS 2: N_freq × N_dir trade-off at fixed compute budget
    # ===================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Frequency vs direction trade-off at fixed compute budget")
    print("=" * 80)

    # Fixed budget: N_freq^2 * N_dir = C
    # Current: 35^2 * 36 = 44100
    # Budget levels to test
    BUDGETS = {
        "1× (current)": 35**2 * 36,
        "2×": 2 * 35**2 * 36,
        "4×": 4 * 35**2 * 36,
    }

    # For each budget, try different N_dir values and compute optimal N_freq
    dir_options = [6, 9, 12, 18, 24, 36]  # must divide 360/10 = 36

    SPREADING = 2.0
    WAVE_DIR = 180.0

    # Reference for short-crested: N=2000 freq, 36 dirs
    sigma_ref_sc = {}
    for bf, ss_i in BEAUFORT.items():
        freqs_ref = geometric_frequencies(2000)
        sig, _ = compute_filtered_sigma_sc(
            freqs_ref, drift_interp_2d, pdstrip_dirs,
            ss_i["hs"], ss_i["tp"], SPREADING, WAVE_DIR, omega_n, ZETA
        )
        sigma_ref_sc[bf] = sig

    bf_plot_list = [5, 6, 7]  # Representative sea states

    fig, axes = plt.subplots(len(BUDGETS), len(bf_plot_list),
                             figsize=(5 * len(bf_plot_list), 4 * len(BUDGETS)))

    for budget_idx, (budget_name, budget) in enumerate(BUDGETS.items()):
        for bf_idx, bf in enumerate(bf_plot_list):
            ss_i = BEAUFORT[bf]
            ax = axes[budget_idx][bf_idx]

            n_freqs_list = []
            errors_geo = []
            errors_opt = []

            for n_dir in dir_options:
                n_freq = int(np.sqrt(budget / n_dir))
                if n_freq < 10:
                    continue
                n_freqs_list.append(n_freq)

                # Direction stride for subsampling
                stride = 36 // n_dir
                sub_dirs = subsample_directions(pdstrip_dirs, stride)

                # Rebuild interpolators for subsampled directions
                sub_interp_2d = {}
                for d in sub_dirs:
                    j_orig = np.argmin(np.abs(pdstrip_dirs - d))
                    sub_interp_2d[d] = drift_interp_2d[pdstrip_dirs[j_orig]]

                # Geometric frequencies
                freqs = geometric_frequencies(n_freq)
                sig, _ = compute_filtered_sigma_sc(
                    freqs, sub_interp_2d, sub_dirs,
                    ss_i["hs"], ss_i["tp"], SPREADING, WAVE_DIR, omega_n, ZETA
                )
                errors_geo.append((sig / sigma_ref_sc[bf] - 1.0) * 100.0)

                # Spectral-optimal frequencies (tailored)
                freqs_opt = optimal_drift_frequencies(
                    n_freq, W_MIN, W_MAX, ss_i["hs"], ss_i["tp"], omega_n, ZETA
                )
                sig_opt, _ = compute_filtered_sigma_sc(
                    freqs_opt, sub_interp_2d, sub_dirs,
                    ss_i["hs"], ss_i["tp"], SPREADING, WAVE_DIR, omega_n, ZETA
                )
                errors_opt.append((sig_opt / sigma_ref_sc[bf] - 1.0) * 100.0)

            # Plot
            x_labels = [f"{nf}f×{nd}d" for nf, nd in
                       zip(n_freqs_list, [d for d in dir_options if int(np.sqrt(budget/d)) >= 10])]
            x = range(len(n_freqs_list))
            ax.bar([i - 0.15 for i in x], errors_geo, 0.3, label="Geometric freq",
                   color="steelblue", alpha=0.8)
            ax.bar([i + 0.15 for i in x], errors_opt, 0.3, label="Optimal freq",
                   color="darkred", alpha=0.8)
            ax.set_xticks(list(x))
            ax.set_xticklabels(x_labels, fontsize=7, rotation=45)
            ax.set_ylabel("σ error [%]")
            ax.axhline(0, color="k", lw=0.5)
            ax.grid(True, alpha=0.3, axis="y")
            if budget_idx == 0:
                ax.set_title(f"BF {bf} (Hs={ss_i['hs']}m)")
            if bf_idx == 0:
                ax.set_ylabel(f"{budget_name}\nσ error [%]")
            if budget_idx == 0 and bf_idx == 0:
                ax.legend(fontsize=7)

    fig.suptitle(f"N_freq × N_dir Trade-off — Short-crested (s=2), ω_n={omega_n}, ζ={ZETA}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "09_freq_dir_tradeoff.png", dpi=150)
    plt.close(fig)
    print("  Plot 9: freq/dir trade-off")

    # ===================================================================
    # ANALYSIS 3: Summary — what's the best practical configuration?
    # ===================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Best configurations at each budget level")
    print("=" * 80)

    for budget_name, budget in BUDGETS.items():
        print(f"\n  Budget: {budget_name} ({budget:,d} ops)")
        print(f"  {'Config':<20s}", end="")
        for bf in bf_plot_list:
            print(f"  {'BF'+str(bf)+' geo':>10s} {'BF'+str(bf)+' opt':>10s}", end="")
        print()

        for n_dir in dir_options:
            n_freq = int(np.sqrt(budget / n_dir))
            if n_freq < 10 or n_freq > 2000:
                continue

            stride = 36 // n_dir
            sub_dirs = subsample_directions(pdstrip_dirs, stride)
            sub_interp_2d = {}
            for d in sub_dirs:
                j_orig = np.argmin(np.abs(pdstrip_dirs - d))
                sub_interp_2d[d] = drift_interp_2d[pdstrip_dirs[j_orig]]

            config = f"{n_freq}f × {n_dir}d"
            print(f"  {config:<20s}", end="")

            for bf in bf_plot_list:
                ss_i = BEAUFORT[bf]

                freqs = geometric_frequencies(n_freq)
                sig, _ = compute_filtered_sigma_sc(
                    freqs, sub_interp_2d, sub_dirs,
                    ss_i["hs"], ss_i["tp"], SPREADING, WAVE_DIR, omega_n, ZETA
                )
                err_geo = (sig / sigma_ref_sc[bf] - 1.0) * 100.0

                freqs_opt = optimal_drift_frequencies(
                    n_freq, W_MIN, W_MAX, ss_i["hs"], ss_i["tp"], omega_n, ZETA
                )
                sig_opt, _ = compute_filtered_sigma_sc(
                    freqs_opt, sub_interp_2d, sub_dirs,
                    ss_i["hs"], ss_i["tp"], SPREADING, WAVE_DIR, omega_n, ZETA
                )
                err_opt = (sig_opt / sigma_ref_sc[bf] - 1.0) * 100.0

                print(f"  {err_geo:+10.1f}% {err_opt:+10.1f}%", end="")
            print()

    # ===================================================================
    # ANALYSIS 4: Direction resolution effect on mean drift (not just SV)
    # ===================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 4: Direction resolution effect on mean drift force")
    print("=" * 80)

    # Mean drift converges fast with N_freq but may be sensitive to N_dir
    # for oblique seas. Check for head seas (should be insensitive).
    print(f"\n  Head seas, spreading=2, N_freq=2000 (converged):")
    print(f"  {'N_dir':<10s}", end="")
    for bf in BEAUFORT:
        print(f"  {'BF'+str(bf):>8s}", end="")
    print()

    freqs_ref = geometric_frequencies(2000)
    mean_ref = {}
    for bf, ss_i in BEAUFORT.items():
        _, mean_ref[bf] = compute_filtered_sigma_sc(
            freqs_ref, drift_interp_2d, pdstrip_dirs,
            ss_i["hs"], ss_i["tp"], SPREADING, WAVE_DIR, omega_n, ZETA
        )

    for n_dir in [6, 9, 12, 18, 24, 36]:
        stride = 36 // n_dir
        sub_dirs = subsample_directions(pdstrip_dirs, stride)
        sub_interp_2d = {}
        for d in sub_dirs:
            j_orig = np.argmin(np.abs(pdstrip_dirs - d))
            sub_interp_2d[d] = drift_interp_2d[pdstrip_dirs[j_orig]]

        print(f"  {n_dir:<10d}", end="")
        for bf, ss_i in BEAUFORT.items():
            _, mean_d = compute_filtered_sigma_sc(
                freqs_ref, sub_interp_2d, sub_dirs,
                ss_i["hs"], ss_i["tp"], SPREADING, WAVE_DIR, omega_n, ZETA
            )
            err = (mean_d / mean_ref[bf] - 1.0) * 100.0 if abs(mean_ref[bf]) > 1.0 else 0.0
            print(f"  {err:+8.1f}%", end="")
        print()

    print(f"\nPlots saved to: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdstrip",
        default="/home/blofro/src/brucon/libs/simulator/vessel_model/test/config/csov_pdstrip.dat",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "output"),
    )
    args = parser.parse_args()
    run_optimization_analysis(args.pdstrip, args.output)


if __name__ == "__main__":
    main()
