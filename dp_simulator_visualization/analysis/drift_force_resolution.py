"""Drift force variance vs frequency resolution analysis.

Computes the slowly-varying wave drift force variance spectrum S_F(μ) using
Newman's approximation (arithmetic mean form, matching the C++ dp_simulator),
for different frequency grid resolutions.  Shows how the 35-frequency
geometric grid used in the simulator compares to finer reference grids,
both with and without vessel DP transfer function filtering.

Covers both long-crested (head seas) and short-crested (spreading=2, 10° resolution)
cases to verify that directional spreading does not improve convergence.

Usage:
    python drift_force_resolution.py [--pdstrip PATH]
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PI = np.pi
G = 9.81

# Beaufort sea states (DNV ST-0111)
BEAUFORT = {
    3: {"hs": 0.8, "tp": 5.5},
    5: {"hs": 2.1, "tp": 7.5},
    6: {"hs": 3.1, "tp": 8.5},
    7: {"hs": 4.2, "tp": 9.0},
    8: {"hs": 5.7, "tp": 10.0},
}

# Frequency grid resolutions to compare
N_VALUES = [35, 70, 140, 280, 560, 2000]

# Frequency range (matching simulator)
W_MIN = 0.300  # rad/s
W_MAX = 1.800  # rad/s

# DP controller parameters (surge, delivered range)
OMEGA_N_VALUES = [0.05, 0.06, 0.07]  # rad/s
ZETA = 0.95

# Simulation duration for peak factor estimation
T_SIM_HOURS = 3.0
T_SIM = T_SIM_HOURS * 3600.0  # seconds

# Binning for difference frequency spectrum
MU_MAX = 0.20   # rad/s — well above DP bandwidth
MU_BINS = 200   # number of bins


# ---------------------------------------------------------------------------
# JONSWAP spectrum (matching C++ and Python wave_model.py exactly)
# ---------------------------------------------------------------------------
def jonswap(w, hs, tp, gamma=3.3):
    """JONSWAP spectral density S(ω) [m²·s/rad]."""
    w = np.clip(np.asarray(w, dtype=float), 1e-10, 1000.0)
    t1 = np.clip(tp * 0.834, 1.0, 50.0)
    sigma = np.where(w > 5.24, 0.09, 0.07)
    exponent = -((0.191 * w * t1 - 1.0) / (np.sqrt(2.0) * sigma)) ** 2
    Y = np.exp(exponent)
    S = (155.0 * hs**2 / (t1**4 * w**5)
         * np.exp(-944.0 / (t1**4 * w**4))
         * gamma**Y)
    return S


# ---------------------------------------------------------------------------
# Geometric frequency grid (matching simulator convention)
# ---------------------------------------------------------------------------
def geometric_frequencies(n, w_min=W_MIN, w_max=W_MAX):
    """Descending geometric frequency grid from w_max to w_min."""
    ratio = (w_max / w_min) ** (1.0 / (n - 1))
    # Descending: w_max, w_max/ratio, w_max/ratio^2, ...
    return w_max / ratio ** np.arange(n)


def frequency_steps(freqs):
    """Frequency step Δω for each component (descending order, matching C++)."""
    n = len(freqs)
    dw = np.zeros(n)
    if n >= 2:
        dw[0] = freqs[0] - freqs[1]
        dw[-1] = freqs[-2] - freqs[-1]
        if n > 2:
            dw[1:-1] = (freqs[:-2] - freqs[2:]) / 2.0
    elif n == 1:
        dw[0] = 1.0
    return dw


# ---------------------------------------------------------------------------
# Parse PdStrip drift coefficients
# ---------------------------------------------------------------------------
def parse_drift_coefficients(filepath):
    """Extract surge drift coefficients D(ω) for speed=0, head seas (angle=180°).

    Returns (frequencies, drift_coeffs) both as arrays, frequencies descending.
    """
    freqs = []
    drift = []
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            freq = float(row[0])
            angle = float(row[2])
            speed = float(row[3])
            if abs(speed) < 0.001 and abs(angle - 180.0) < 0.1:
                freqs.append(freq)
                drift.append(float(row[16]))  # surge_d
    return np.array(freqs), np.array(drift)


def parse_drift_coefficients_all_directions(filepath):
    """Extract surge drift coefficients D(ω, θ) for speed=0, all directions.

    Returns
    -------
    frequencies : array (n_freq,), descending
    directions : array (n_dir,), PdStrip convention (0=aft, 180=head)
    drift_coeffs : array (n_freq, n_dir), surge drift D(ω, θ)
    """
    # First pass: find unique frequencies and directions at speed=0
    data = {}  # (freq, angle) -> surge_d
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            freq = float(row[0])
            angle = float(row[2])
            speed = float(row[3])
            if abs(speed) < 0.001:
                data[(freq, angle)] = float(row[16])

    freqs = sorted(set(k[0] for k in data), reverse=True)
    dirs = sorted(set(k[1] for k in data))

    drift = np.zeros((len(freqs), len(dirs)))
    for i, f in enumerate(freqs):
        for j, d in enumerate(dirs):
            drift[i, j] = data.get((f, d), 0.0)

    return np.array(freqs), np.array(dirs), drift


# ---------------------------------------------------------------------------
# Slowly-varying drift force variance computation
# ---------------------------------------------------------------------------
def compute_sv_drift_variance_spectrum(freqs, drift_coeffs, hs, tp, mu_bins, mu_max):
    """Compute the slowly-varying drift force variance spectrum S_F(μ).

    Uses Newman's approximation (arithmetic mean):
        T(ωᵢ, ωⱼ) = ½[D(ωᵢ) + D(ωⱼ)]

    The variance contribution from each off-diagonal pair (i,j) at
    difference frequency μ = |ωⱼ - ωᵢ| is:
        δσ² = 2 · [½(Dᵢ + Dⱼ)]² · aᵢ² · aⱼ²

    where aᵢ² = 2·S(ωᵢ)·Δωᵢ, and the factor 2 accounts for the symmetric
    pair (j,i).  We bin these contributions by μ to get a spectral density.

    Returns
    -------
    mu_centers : array, shape (mu_bins,)
        Difference frequency bin centers [rad/s].
    S_F : array, shape (mu_bins,)
        Variance spectral density [(N/m²)²·s/rad] (before scaling by ½).
        Note: actual units depend on drift coefficient units.
    total_variance : float
        Total slowly-varying variance (sum of all off-diagonal contributions).
    mean_drift : float
        Mean drift force [same units as D·a²].
    """
    n = len(freqs)
    dw = frequency_steps(freqs)
    S = jonswap(freqs, hs, tp)
    a_sq = 2.0 * S * dw  # amplitude squared for each component

    # Bin setup
    d_mu = mu_max / mu_bins
    mu_edges = np.linspace(0, mu_max, mu_bins + 1)
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    S_F = np.zeros(mu_bins)

    # Mean drift force (diagonal terms)
    mean_drift = np.sum(drift_coeffs * a_sq)

    # Off-diagonal terms: all pairs (i, j) with j > i
    total_var = 0.0
    D = drift_coeffs
    for i in range(n):
        for j in range(i + 1, n):
            mu = abs(freqs[j] - freqs[i])
            # Newman: T_ij = 0.5 * (D_i + D_j)
            T_ij = 0.5 * (D[i] + D[j])
            # Variance contribution from this pair (×2 for symmetry i↔j)
            contrib = 2.0 * T_ij**2 * a_sq[i] * a_sq[j]
            total_var += contrib

            # Bin into spectrum
            bin_idx = int(mu / d_mu)
            if 0 <= bin_idx < mu_bins:
                S_F[bin_idx] += contrib

    # Convert binned variance to spectral density
    S_F /= d_mu

    return mu_centers, S_F, total_var, mean_drift


def compute_sv_drift_short_crested(
    freqs, drift_2d, pdstrip_dirs, hs, tp, spreading, wave_dir_deg,
    mu_bins, mu_max
):
    """Compute slowly-varying drift force variance for short-crested seas.

    Matches the C++ implementation: each direction is computed independently
    (no cross-direction terms), with independent phases per direction.
    The total variance is the sum of per-direction variances.

    Parameters
    ----------
    freqs : array (n_freq,), descending
        Frequency grid [rad/s].
    drift_2d : array (n_freq, n_dir)
        Surge drift coefficients D(ω, θ) — interpolated onto this freq grid.
    pdstrip_dirs : array (n_dir,)
        PdStrip direction bins [deg].
    hs, tp : float
        JONSWAP spectrum parameters.
    spreading : float
        Cosine-power spreading exponent (e.g. 2.0).
    wave_dir_deg : float
        Dominant wave direction in PdStrip convention (180 = head seas).
    mu_bins, mu_max : int, float
        Binning parameters for difference frequency.

    Returns
    -------
    mu_centers, S_F, total_var, mean_drift — same as long-crested version.
    """
    n_freq = len(freqs)
    n_dir = len(pdstrip_dirs)
    dw = frequency_steps(freqs)
    S_1d = jonswap(freqs, hs, tp)
    dir_step_deg = pdstrip_dirs[1] - pdstrip_dirs[0]  # 10°
    dir_step_rad = np.deg2rad(dir_step_deg)

    # Directional spreading weights: cos^s(θ - θ_dom) / scale
    # Compute scale by Simpson integration matching C++
    n_simp = 21
    theta_simp = np.linspace(-PI / 2.0, PI / 2.0, n_simp)
    integrand = np.power(np.cos(theta_simp), spreading)
    step_simp = PI / (n_simp - 1.0)
    s_val = integrand[0] + integrand[-1]
    s_val += 4.0 * np.sum(integrand[1:-1:2])
    s_val += 2.0 * np.sum(integrand[2:-2:2])
    spread_scale = max(s_val * step_simp / 3.0, 1.0)

    # Per-direction weights
    dir_weights = np.zeros(n_dir)
    for j in range(n_dir):
        diff = ((wave_dir_deg - pdstrip_dirs[j]) + 180) % 360 - 180
        if abs(diff) < 90.0:
            dir_weights[j] = np.cos(np.deg2rad(diff)) ** spreading / spread_scale
        else:
            dir_weights[j] = 0.0

    # Bin setup
    d_mu = mu_max / mu_bins
    mu_edges = np.linspace(0, mu_max, mu_bins + 1)
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    S_F = np.zeros(mu_bins)
    total_var = 0.0
    mean_drift = 0.0

    # Sum over active directions — each direction independently
    for j in range(n_dir):
        if dir_weights[j] < 1e-10:
            continue

        # S_2d(ω, θ_j) = S_1d(ω) · D(θ_j) · Δθ
        S_2d = S_1d * dir_weights[j] * dir_step_rad
        a_sq = 2.0 * S_2d * dw  # amplitude squared per component
        D = drift_2d[:, j]  # drift coefficients at this direction

        # Mean drift (diagonal)
        mean_drift += np.sum(D * a_sq)

        # Off-diagonal variance
        for i in range(n_freq):
            for k in range(i + 1, n_freq):
                mu = abs(freqs[k] - freqs[i])
                T_ik = 0.5 * (D[i] + D[k])
                contrib = 2.0 * T_ik**2 * a_sq[i] * a_sq[k]
                total_var += contrib
                bin_idx = int(mu / d_mu)
                if 0 <= bin_idx < mu_bins:
                    S_F[bin_idx] += contrib

    S_F /= d_mu
    return mu_centers, S_F, total_var, mean_drift


# ---------------------------------------------------------------------------
# Vessel DP transfer function
# ---------------------------------------------------------------------------
def vessel_transfer_sq(mu, omega_n, zeta):
    """Squared magnitude of 2nd-order vessel response transfer function.

    |H(μ)|² = 1 / [(1 - (μ/ω_n)²)² + (2ζμ/ω_n)²]

    This represents the DP closed-loop response: the vessel acts as a
    low-pass filter on drift forces, with natural frequency ω_n and
    damping ratio ζ.
    """
    r = mu / omega_n
    return 1.0 / ((1.0 - r**2)**2 + (2.0 * zeta * r)**2)


# ---------------------------------------------------------------------------
# Peak factor (extreme value estimate)
# ---------------------------------------------------------------------------
def peak_factor(omega_n, T_sim):
    """Expected peak factor for a narrow-band process.

    For a narrow-band process with dominant frequency near ω_n,
    the number of cycles in T_sim is approximately n = T_sim * ω_n / (2π).
    The expected maximum of a Rayleigh-distributed envelope over n cycles
    is approximately √(2·ln(n)) for large n.

    For a chi-squared (quadratic) process like slowly-varying drift,
    the peak factor is somewhat different, but √(2·ln(n)) provides a
    reasonable engineering estimate for the peak-to-RMS ratio.
    """
    n_cycles = max(T_sim * omega_n / (2.0 * PI), 1.0)
    return np.sqrt(2.0 * np.log(n_cycles))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis(pdstrip_path, output_dir):
    """Run the full parametric drift force resolution analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse reference drift coefficients
    ref_freqs, ref_drift = parse_drift_coefficients(pdstrip_path)
    print(f"Loaded {len(ref_freqs)} drift coefficients from {pdstrip_path}")
    print(f"  Frequency range: {ref_freqs[-1]:.3f} – {ref_freqs[0]:.3f} rad/s")
    print(f"  Surge drift range: {ref_drift.min():.0f} – {ref_drift.max():.0f} N/m²")

    # Create interpolator for drift coefficients (log-space for geometric grid)
    # Reverse to ascending for interp1d
    drift_interp = interp1d(
        ref_freqs[::-1], ref_drift[::-1],
        kind="cubic", fill_value="extrapolate"
    )

    # -----------------------------------------------------------------------
    # Pre-compute results for all (N, BF) combinations
    # -----------------------------------------------------------------------
    # results[N][BF] = {mu_centers, S_F, total_var, mean_drift, freqs, drift}
    results = {}
    for N in N_VALUES:
        results[N] = {}
        freqs = geometric_frequencies(N)
        D = drift_interp(freqs)

        for bf, ss in BEAUFORT.items():
            mu_c, S_F, total_var, mean_drift = compute_sv_drift_variance_spectrum(
                freqs, D, ss["hs"], ss["tp"], MU_BINS, MU_MAX
            )
            results[N][bf] = {
                "mu": mu_c,
                "S_F": S_F,
                "total_var": total_var,
                "mean_drift": mean_drift,
                "freqs": freqs,
                "drift": D,
            }

        print(f"  N={N:5d}: computed {len(BEAUFORT)} sea states")

    # Reference N for relative errors
    N_ref = N_VALUES[-1]

    # -----------------------------------------------------------------------
    # Plot 1: Drift coefficient D(ω) and interpolation check
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    w_fine = np.linspace(W_MIN, W_MAX, 500)
    ax.plot(w_fine, drift_interp(w_fine) / 1e3, "k-", lw=1, label="Interpolated")
    ax.plot(ref_freqs, ref_drift / 1e3, "ro", ms=5, label="PdStrip (N=35)")
    for N in [70, 140, 280]:
        f = geometric_frequencies(N)
        ax.plot(f, drift_interp(f) / 1e3, ".", ms=2, alpha=0.5)
    ax.set_xlabel("Frequency ω [rad/s]")
    ax.set_ylabel("Surge drift coefficient D(ω) [kN/m²]")
    ax.set_title("CSOV Surge Drift Coefficient — Head Seas (PdStrip)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "01_drift_coefficient.png", dpi=150)
    plt.close(fig)
    print("  Plot 1: drift coefficient")

    # -----------------------------------------------------------------------
    # Plot 2: S_F(μ) for different N at BF 6 (representative)
    # -----------------------------------------------------------------------
    bf_plot = 6
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for N in N_VALUES:
        r = results[N][bf_plot]
        label = f"N={N}"
        lw = 2.0 if N == 35 else (1.5 if N == N_ref else 0.8)
        ls = "-" if N in (35, N_ref) else "--"
        ax.plot(r["mu"], r["S_F"] / 1e12, ls, lw=lw, label=label)
    ax.set_xlabel("Difference frequency μ [rad/s]")
    ax.set_ylabel("S_F(μ) [10¹² N²·s/rad]")
    ax.set_title(f"Slowly-Varying Drift Force Spectrum — BF {bf_plot} "
                 f"(Hs={BEAUFORT[bf_plot]['hs']}m, Tp={BEAUFORT[bf_plot]['tp']}s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MU_MAX)

    # Same but vessel-filtered (ω_n = 0.06)
    ax = axes[1]
    omega_n = 0.06
    for N in N_VALUES:
        r = results[N][bf_plot]
        H2 = vessel_transfer_sq(r["mu"], omega_n, ZETA)
        label = f"N={N}"
        lw = 2.0 if N == 35 else (1.5 if N == N_ref else 0.8)
        ls = "-" if N in (35, N_ref) else "--"
        ax.plot(r["mu"], r["S_F"] * H2 / 1e12, ls, lw=lw, label=label)
    ax.set_xlabel("Difference frequency μ [rad/s]")
    ax.set_ylabel("|H(μ)|²·S_F(μ) [10¹² N²·s/rad]")
    ax.set_title(f"Vessel-Filtered Response — ω_n={omega_n} rad/s, ζ={ZETA}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MU_MAX)

    fig.tight_layout()
    fig.savefig(output_dir / "02_spectrum_comparison.png", dpi=150)
    plt.close(fig)
    print("  Plot 2: spectrum comparison")

    # -----------------------------------------------------------------------
    # Plot 3: Total variance vs N — convergence curves
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Unfiltered
    ax = axes[0]
    for bf in BEAUFORT:
        vars_by_n = [results[N][bf]["total_var"] for N in N_VALUES]
        label = f"BF {bf} (Hs={BEAUFORT[bf]['hs']}m)"
        ax.plot(N_VALUES, np.array(vars_by_n) / 1e9, "o-", ms=4, label=label)
    ax.set_xlabel("Number of frequencies N")
    ax.set_ylabel("σ²_F [10⁹ N²]")
    ax.set_title("Total Slowly-Varying Drift Force Variance (Unfiltered)")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(N_VALUES)

    # Vessel-filtered (ω_n = 0.06)
    ax = axes[1]
    omega_n = 0.06
    for bf in BEAUFORT:
        filtered_vars = []
        for N in N_VALUES:
            r = results[N][bf]
            H2 = vessel_transfer_sq(r["mu"], omega_n, ZETA)
            d_mu = MU_MAX / MU_BINS
            filt_var = np.sum(r["S_F"] * H2) * d_mu
            filtered_vars.append(filt_var)
        label = f"BF {bf} (Hs={BEAUFORT[bf]['hs']}m)"
        ax.plot(N_VALUES, np.array(filtered_vars) / 1e9, "o-", ms=4, label=label)
    ax.set_xlabel("Number of frequencies N")
    ax.set_ylabel("σ²_resp [10⁹ N²]")
    ax.set_title(f"Vessel-Filtered Variance — ω_n={omega_n}, ζ={ZETA}")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(N_VALUES)

    fig.tight_layout()
    fig.savefig(output_dir / "03_variance_convergence.png", dpi=150)
    plt.close(fig)
    print("  Plot 3: variance convergence")

    # -----------------------------------------------------------------------
    # Plot 4: Relative error vs N (% relative to N_ref)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, omega_n in enumerate(OMEGA_N_VALUES):
        ax = axes[ax_idx]
        for bf in BEAUFORT:
            # Reference: vessel-filtered variance at N_ref
            r_ref = results[N_ref][bf]
            H2_ref = vessel_transfer_sq(r_ref["mu"], omega_n, ZETA)
            d_mu = MU_MAX / MU_BINS
            var_ref = np.sum(r_ref["S_F"] * H2_ref) * d_mu

            errors = []
            for N in N_VALUES:
                r = results[N][bf]
                H2 = vessel_transfer_sq(r["mu"], omega_n, ZETA)
                var_n = np.sum(r["S_F"] * H2) * d_mu
                err_pct = (var_n / var_ref - 1.0) * 100.0 if var_ref > 0 else 0.0
                errors.append(err_pct)
            label = f"BF {bf}"
            ax.plot(N_VALUES, errors, "o-", ms=4, label=label)

        ax.set_xlabel("Number of frequencies N")
        ax.set_ylabel("Variance error vs N=2000 [%]")
        ax.set_title(f"ω_n = {omega_n} rad/s (T_n = {2*PI/omega_n:.0f}s)")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", lw=0.5)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks(N_VALUES)

    fig.suptitle(f"Relative Error in Vessel-Filtered Drift Force Variance (ζ={ZETA})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "04_relative_error.png", dpi=150)
    plt.close(fig)
    print("  Plot 4: relative error")

    # -----------------------------------------------------------------------
    # Plot 5: Expected peak drift force vs N (3-hour simulation)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, omega_n in enumerate(OMEGA_N_VALUES):
        ax = axes[ax_idx]
        pf = peak_factor(omega_n, T_SIM)

        for bf in BEAUFORT:
            peak_forces = []
            for N in N_VALUES:
                r = results[N][bf]
                H2 = vessel_transfer_sq(r["mu"], omega_n, ZETA)
                d_mu = MU_MAX / MU_BINS
                filt_var = np.sum(r["S_F"] * H2) * d_mu
                sigma = np.sqrt(filt_var)
                # Expected peak = mean drift + peak_factor * sigma
                # (We report sigma * pf here; mean drift is separate)
                peak_forces.append(pf * sigma / 1e3)  # kN

            label = f"BF {bf} (Hs={BEAUFORT[bf]['hs']}m)"
            ax.plot(N_VALUES, peak_forces, "o-", ms=4, label=label)

        ax.set_xlabel("Number of frequencies N")
        ax.set_ylabel("Expected peak slowly-varying force [kN]")
        ax.set_title(f"ω_n = {omega_n} rad/s, peak factor = {pf:.2f}")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks(N_VALUES)

    fig.suptitle(f"Expected Peak Slowly-Varying Surge Drift Force — {T_SIM_HOURS:.0f}h Simulation, ζ={ZETA}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "05_peak_force.png", dpi=150)
    plt.close(fig)
    print("  Plot 5: peak force")

    # -----------------------------------------------------------------------
    # Plot 6: RMS force error (σ error, not variance error)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Show for the middle ω_n = 0.06
    omega_n = 0.06
    d_mu = MU_MAX / MU_BINS

    for bf in BEAUFORT:
        r_ref = results[N_ref][bf]
        H2_ref = vessel_transfer_sq(r_ref["mu"], omega_n, ZETA)
        sigma_ref = np.sqrt(np.sum(r_ref["S_F"] * H2_ref) * d_mu)

        sigma_errors = []
        for N in N_VALUES:
            r = results[N][bf]
            H2 = vessel_transfer_sq(r["mu"], omega_n, ZETA)
            sigma_n = np.sqrt(np.sum(r["S_F"] * H2) * d_mu)
            err_pct = (sigma_n / sigma_ref - 1.0) * 100.0 if sigma_ref > 0 else 0.0
            sigma_errors.append(err_pct)

        label = f"BF {bf} (Hs={BEAUFORT[bf]['hs']}m, Tp={BEAUFORT[bf]['tp']}s)"
        ax.plot(N_VALUES, sigma_errors, "o-", ms=5, lw=1.5, label=label)

    ax.set_xlabel("Number of frequencies N")
    ax.set_ylabel("RMS force error vs N=2000 [%]")
    ax.set_title(f"Error in RMS Slowly-Varying Surge Drift Force — "
                 f"ω_n={omega_n} rad/s, ζ={ZETA}")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(N_VALUES)
    fig.tight_layout()
    fig.savefig(output_dir / "06_rms_error.png", dpi=150)
    plt.close(fig)
    print("  Plot 6: RMS error")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUMMARY — Vessel-filtered slowly-varying drift force variance")
    print("          Head seas, long-crested JONSWAP (γ=3.3), surge, ζ=0.95")
    print("=" * 90)

    for omega_n in OMEGA_N_VALUES:
        T_n = 2 * PI / omega_n
        print(f"\n  ω_n = {omega_n} rad/s  (T_n = {T_n:.0f} s)")
        print(f"  {'BF':>4s} {'Hs':>5s} {'Tp':>5s} ", end="")
        print(f"{'Mean':>10s} ", end="")
        for N in N_VALUES:
            print(f"{'N='+str(N):>10s} ", end="")
        print(f" {'Err35':>7s}")
        print(f"  {'':>4s} {'(m)':>5s} {'(s)':>5s} ", end="")
        print(f"{'(kN)':>10s} ", end="")
        for _ in N_VALUES:
            print(f"{'σ (kN)':>10s} ", end="")
        print(f" {'(%)':>7s}")
        print("  " + "-" * 86)

        for bf in BEAUFORT:
            ss = BEAUFORT[bf]
            mean_drift = results[N_VALUES[0]][bf]["mean_drift"]

            sigmas = []
            for N in N_VALUES:
                r = results[N][bf]
                H2 = vessel_transfer_sq(r["mu"], omega_n, ZETA)
                filt_var = np.sum(r["S_F"] * H2) * d_mu
                sigmas.append(np.sqrt(filt_var))

            sigma_ref = sigmas[-1]
            err = (sigmas[0] / sigma_ref - 1.0) * 100.0 if sigma_ref > 0 else 0.0

            print(f"  {bf:4d} {ss['hs']:5.1f} {ss['tp']:5.1f} ", end="")
            print(f"{mean_drift/1e3:10.1f} ", end="")
            for s in sigmas:
                print(f"{s/1e3:10.2f} ", end="")
            print(f" {err:+7.1f}")

    print("\n" + "=" * 90)
    print(f"Peak factors (√(2·ln(n_cycles)), {T_SIM_HOURS:.0f}h simulation):")
    for omega_n in OMEGA_N_VALUES:
        pf = peak_factor(omega_n, T_SIM)
        n_cyc = T_SIM * omega_n / (2 * PI)
        print(f"  ω_n={omega_n}: {n_cyc:.0f} cycles, peak factor = {pf:.2f}")
    print("=" * 90)

    # ===================================================================
    # SHORT-CRESTED COMPARISON
    # ===================================================================
    # Parse drift coefficients for ALL directions (speed=0)
    print("\n\nShort-crested analysis (spreading=2, 10° resolution, head seas)...")
    ref_freqs_all, pdstrip_dirs, ref_drift_2d = \
        parse_drift_coefficients_all_directions(pdstrip_path)
    print(f"  Loaded drift coefficients: {ref_drift_2d.shape} (freq × dir)")
    print(f"  Directions: {pdstrip_dirs[0]:.0f}° to {pdstrip_dirs[-1]:.0f}° "
          f"({len(pdstrip_dirs)} bins, Δθ={pdstrip_dirs[1]-pdstrip_dirs[0]:.0f}°)")

    # Create 2D interpolators: one per direction
    drift_interp_2d = {}
    for j, d in enumerate(pdstrip_dirs):
        drift_interp_2d[d] = interp1d(
            ref_freqs_all[::-1], ref_drift_2d[::-1, j],
            kind="cubic", fill_value="extrapolate"
        )

    SPREADING = 2.0
    WAVE_DIR = 180.0  # PdStrip: 180 = head seas

    # Only compute for a subset of N values (short-crested is slower)
    N_SC = [35, 70, 140, 280, 560, 2000]

    sc_results = {}
    for N in N_SC:
        sc_results[N] = {}
        freqs = geometric_frequencies(N)

        # Interpolate drift coefficients onto this frequency grid for all dirs
        drift_2d = np.zeros((len(freqs), len(pdstrip_dirs)))
        for j, d in enumerate(pdstrip_dirs):
            drift_2d[:, j] = drift_interp_2d[d](freqs)

        for bf, ss in BEAUFORT.items():
            mu_c, S_F, total_var, mean_drift = compute_sv_drift_short_crested(
                freqs, drift_2d, pdstrip_dirs, ss["hs"], ss["tp"],
                SPREADING, WAVE_DIR, MU_BINS, MU_MAX
            )
            sc_results[N][bf] = {
                "mu": mu_c,
                "S_F": S_F,
                "total_var": total_var,
                "mean_drift": mean_drift,
            }
        print(f"  N={N:5d}: computed {len(BEAUFORT)} sea states (short-crested)")

    # -----------------------------------------------------------------------
    # Plot 7: Long-crested vs short-crested RMS error comparison
    # -----------------------------------------------------------------------
    omega_n = 0.06
    d_mu = MU_MAX / MU_BINS

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Long-crested
    ax = axes[0]
    for bf in BEAUFORT:
        r_ref = results[N_ref][bf]
        H2_ref = vessel_transfer_sq(r_ref["mu"], omega_n, ZETA)
        sigma_ref = np.sqrt(np.sum(r_ref["S_F"] * H2_ref) * d_mu)
        errs = []
        for N in N_VALUES:
            r = results[N][bf]
            H2 = vessel_transfer_sq(r["mu"], omega_n, ZETA)
            sigma_n = np.sqrt(np.sum(r["S_F"] * H2) * d_mu)
            errs.append((sigma_n / sigma_ref - 1.0) * 100.0 if sigma_ref > 0 else 0.0)
        ax.plot(N_VALUES, errs, "o-", ms=4, label=f"BF {bf}")
    ax.set_xlabel("Number of frequencies N")
    ax.set_ylabel("RMS force error vs N=2000 [%]")
    ax.set_title("Long-Crested (spreading=∞)")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(N_VALUES)

    # Right: Short-crested
    ax = axes[1]
    N_ref_sc = N_SC[-1]
    for bf in BEAUFORT:
        r_ref = sc_results[N_ref_sc][bf]
        H2_ref = vessel_transfer_sq(r_ref["mu"], omega_n, ZETA)
        sigma_ref = np.sqrt(np.sum(r_ref["S_F"] * H2_ref) * d_mu)
        errs = []
        for N in N_SC:
            r = sc_results[N][bf]
            H2 = vessel_transfer_sq(r["mu"], omega_n, ZETA)
            sigma_n = np.sqrt(np.sum(r["S_F"] * H2) * d_mu)
            errs.append((sigma_n / sigma_ref - 1.0) * 100.0 if sigma_ref > 0 else 0.0)
        ax.plot(N_SC, errs, "o-", ms=4, label=f"BF {bf}")
    ax.set_xlabel("Number of frequencies N")
    ax.set_ylabel("RMS force error vs N=2000 [%]")
    ax.set_title("Short-Crested (spreading=2, 10° resolution)")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(N_SC)

    fig.suptitle(f"Effect of Directional Spreading on Frequency Resolution Error — "
                 f"ω_n={omega_n} rad/s, ζ={ZETA}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "07_long_vs_short_crested.png", dpi=150)
    plt.close(fig)
    print("  Plot 7: long vs short crested comparison")

    # Print short-crested summary
    print("\n" + "=" * 90)
    print("SHORT-CRESTED COMPARISON — spreading=2, head seas, 10° resolution")
    print(f"  ω_n = {omega_n} rad/s  (T_n = {2*PI/omega_n:.0f} s)")
    print("=" * 90)
    print(f"  {'BF':>4s} {'Hs':>5s}  {'Long-crested':>20s}  {'Short-crested':>20s}")
    print(f"  {'':>4s} {'(m)':>5s}  {'σ_35/σ_2000 err%':>20s}  {'σ_35/σ_2000 err%':>20s}")
    print("  " + "-" * 60)

    for bf in BEAUFORT:
        ss = BEAUFORT[bf]
        # Long-crested error
        r_ref = results[N_ref][bf]
        H2 = vessel_transfer_sq(r_ref["mu"], omega_n, ZETA)
        sigma_ref_lc = np.sqrt(np.sum(r_ref["S_F"] * H2) * d_mu)
        r35 = results[35][bf]
        sigma_35_lc = np.sqrt(np.sum(r35["S_F"] * vessel_transfer_sq(r35["mu"], omega_n, ZETA)) * d_mu)
        err_lc = (sigma_35_lc / sigma_ref_lc - 1.0) * 100.0 if sigma_ref_lc > 0 else 0.0

        # Short-crested error
        r_ref = sc_results[N_ref_sc][bf]
        sigma_ref_sc = np.sqrt(np.sum(r_ref["S_F"] * vessel_transfer_sq(r_ref["mu"], omega_n, ZETA)) * d_mu)
        r35 = sc_results[35][bf]
        sigma_35_sc = np.sqrt(np.sum(r35["S_F"] * vessel_transfer_sq(r35["mu"], omega_n, ZETA)) * d_mu)
        err_sc = (sigma_35_sc / sigma_ref_sc - 1.0) * 100.0 if sigma_ref_sc > 0 else 0.0

        print(f"  {bf:4d} {ss['hs']:5.1f}  {err_lc:+20.1f}  {err_sc:+20.1f}")

    print("=" * 90)

    print(f"\nPlots saved to: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdstrip",
        default="/home/blofro/src/brucon/libs/simulator/vessel_model/test/config/csov_pdstrip.dat",
        help="Path to csov_pdstrip.dat",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "output"),
        help="Output directory for plots",
    )
    args = parser.parse_args()
    run_analysis(args.pdstrip, args.output)


if __name__ == "__main__":
    main()
