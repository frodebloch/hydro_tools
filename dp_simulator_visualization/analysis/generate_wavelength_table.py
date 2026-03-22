#!/usr/bin/env python3
"""Generate an optimized wavelength table for PdStrip.

Produces frequency placements that minimize slowly-varying drift force
estimation error, then converts to deep-water wavelengths for PdStrip input.

Two modes:
  1. Spectral-optimal (no existing pdstrip data needed): weights by S(omega)
  2. Envelope-weighted (requires existing pdstrip.dat): weights by D2_envelope * S(omega)
     where D2_envelope = max over all DOFs and headings of drift coefficient squared.

Output: wavelengths (metres), space-separated on one line, ascending order.

Usage examples:
    # Spectral-optimal for BF7:
    python generate_wavelength_table.py --hs 4.2 --tp 9.0 --n 70

    # Envelope-weighted using existing drift data:
    python generate_wavelength_table.py --hs 4.2 --tp 9.0 --n 70 \\
        --pdstrip-file /path/to/csov_pdstrip.dat

    # Compare against current geometric grid:
    python generate_wavelength_table.py --hs 4.2 --tp 9.0 --n 70 --compare

    # Generate a dense/wide grid for future runtime interpolation:
    python generate_wavelength_table.py --hs 4.2 --tp 9.0 --n 200 \\
        --w-min 0.15 --w-max 3.0 --pdstrip-file /path/to/csov_pdstrip.dat
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent))
from drift_force_resolution import (
    jonswap, geometric_frequencies, frequency_steps,
    vessel_transfer_sq, compute_sv_drift_variance_spectrum,
    PI, G, BEAUFORT, ZETA, MU_BINS, MU_MAX,
)
from optimization_analysis import compute_filtered_sigma


# ---------------------------------------------------------------------------
# Frequency placement algorithms
# ---------------------------------------------------------------------------

def spectral_optimal_frequencies(n, w_min, w_max, hs, tp, omega_n,
                                  gamma=3.3):
    """CDF-quantile placement weighted by S(omega) * S(omega+omega_n).

    Concentrates frequencies where the spectrum has most energy and where
    frequency pairs separated by ~omega_n both have energy.
    """
    w_fine = np.linspace(w_min, w_max, 10000)
    S = jonswap(w_fine, hs, tp, gamma)
    S_shifted = jonswap(w_fine + omega_n, hs, tp, gamma)
    weight = np.sqrt(S * S_shifted + S * S)
    weight = np.maximum(weight, 1e-30)

    cdf = np.cumsum(weight)
    cdf = cdf / cdf[-1]

    quantiles = np.linspace(0.0, 1.0, n + 2)[1:-1]
    w_opt = np.interp(quantiles, cdf, w_fine)
    w_opt[0] = max(w_opt[0], w_min)
    w_opt[-1] = min(w_opt[-1], w_max)

    return np.sort(w_opt)


def envelope_weighted_frequencies(n, w_min, w_max, D2_interp, hs, tp,
                                   omega_n, gamma=3.3):
    """CDF-quantile placement weighted by D2_envelope(omega) * S(omega).

    D2_envelope captures where the hull generates drift forces, making the
    placement robust across all DOFs and headings.  Prevents wasting
    frequencies at swell peaks where QTFs are negligible.
    """
    w_fine = np.linspace(w_min, w_max, 10000)
    S = jonswap(w_fine, hs, tp, gamma)
    D2 = np.maximum(D2_interp(w_fine), 0.0)

    S_shifted = jonswap(w_fine + omega_n, hs, tp, gamma)
    D2_shifted = np.maximum(
        D2_interp(np.clip(w_fine + omega_n, w_min, w_max)), 0.0
    )

    weight = np.sqrt(D2 * S * D2_shifted * S_shifted + D2**2 * S**2)
    weight = np.maximum(weight, 1e-30)

    cdf = np.cumsum(weight)
    cdf = cdf / cdf[-1]

    quantiles = np.linspace(0.0, 1.0, n + 2)[1:-1]
    w_opt = np.interp(quantiles, cdf, w_fine)
    w_opt[0] = max(w_opt[0], w_min)
    w_opt[-1] = min(w_opt[-1], w_max)

    return np.sort(w_opt)


# ---------------------------------------------------------------------------
# PdStrip data loading (drift coefficients)
# ---------------------------------------------------------------------------

def load_drift_envelope(filepath):
    """Load drift coefficients and compute D2 envelope over all DOFs/headings.

    Returns an interpolation function D2_envelope(omega).
    """
    data = {}
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            freq = float(row[0])
            speed = float(row[3])
            if abs(speed) < 0.001:  # zero speed only
                surge_d = float(row[16])
                sway_d = float(row[17])
                yaw_d = float(row[18])
                if freq not in data:
                    data[freq] = []
                data[freq].append((surge_d, sway_d, yaw_d))

    freqs = sorted(data.keys())
    D2_envelope = np.zeros(len(freqs))
    for i, f in enumerate(freqs):
        coeffs = data[f]
        max_D2 = 0.0
        for surge_d, sway_d, yaw_d in coeffs:
            D2 = max(surge_d**2, sway_d**2, yaw_d**2)
            max_D2 = max(max_D2, D2)
        D2_envelope[i] = max_D2

    return interp1d(freqs, D2_envelope, kind="linear",
                    bounds_error=False, fill_value=0.0)


def load_surge_drift_head_seas(filepath):
    """Load surge drift coefficient for head seas (180 deg), speed=0.

    Returns an interpolation function D_surge(omega).
    """
    freqs_pd = []
    surge_d_head = []
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            freq = float(row[0])
            angle = float(row[2])
            speed = float(row[3])
            if abs(speed) < 0.001 and abs(angle - 180.0) < 0.5:
                freqs_pd.append(freq)
                surge_d_head.append(float(row[16]))
    return interp1d(freqs_pd, surge_d_head, kind="linear",
                    bounds_error=False, fill_value=0.0)


# ---------------------------------------------------------------------------
# Wavelength conversion
# ---------------------------------------------------------------------------

def omega_to_wavelength(omega):
    """Convert circular frequency (rad/s) to deep-water wavelength (m).

    lambda = 2*pi*g / omega^2
    """
    return 2.0 * PI * G / omega**2


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_grid_stats(freqs, label):
    """Print frequency grid statistics."""
    dw = np.abs(frequency_steps(freqs))
    print(f"\n  {label}:")
    print(f"    N = {len(freqs)}")
    print(f"    omega range: [{freqs.min():.4f}, {freqs.max():.4f}] rad/s")
    print(f"    lambda range: [{omega_to_wavelength(freqs.max()):.1f}, "
          f"{omega_to_wavelength(freqs.min()):.1f}] m")
    print(f"    d_omega range: [{dw.min():.5f}, {dw.max():.5f}] rad/s")
    print(f"    d_omega mean:  {dw.mean():.5f} rad/s")


def print_drift_accuracy(freqs_desc, hs, tp, omega_n, drift_interp, label):
    """Compute and print drift force accuracy for a given grid.

    freqs_desc: frequencies in descending order (as used by the simulator).
    Returns sigma.
    """
    sigma, mean_d = compute_filtered_sigma(
        freqs_desc, drift_interp, hs, tp, omega_n, ZETA
    )
    print(f"    sigma_drift (filtered at omega_n={omega_n}): {sigma:.2f} kN")
    print(f"    Mean drift: {mean_d:.2f} kN")
    return sigma


def compute_reference_sigma(hs, tp, omega_n, drift_interp, w_min, w_max):
    """Compute reference sigma at N=2000 for comparison."""
    freqs_ref = np.linspace(w_max, w_min, 2000)  # descending
    sigma_ref, _ = compute_filtered_sigma(
        freqs_ref, drift_interp, hs, tp, omega_n, ZETA
    )
    return sigma_ref


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate optimized wavelength table for PdStrip"
    )
    parser.add_argument("--n", type=int, default=70,
                        help="Number of frequencies (default: 70)")
    parser.add_argument("--hs", type=float, required=True,
                        help="Significant wave height [m]")
    parser.add_argument("--tp", type=float, required=True,
                        help="Peak period [s]")
    parser.add_argument("--omega-n", type=float, default=0.06,
                        help="DP natural frequency [rad/s] (default: 0.06)")
    parser.add_argument("--w-min", type=float, default=0.20,
                        help="Min circular frequency [rad/s] (default: 0.20)")
    parser.add_argument("--w-max", type=float, default=2.50,
                        help="Max circular frequency [rad/s] (default: 2.50)")
    parser.add_argument("--pdstrip-file", type=str, default=None,
                        help="Path to existing pdstrip.dat for D2 envelope "
                             "weighting")
    parser.add_argument("--gamma", type=float, default=3.3,
                        help="JONSWAP peak enhancement factor (default: 3.3)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare against current geometric N=35 grid")

    args = parser.parse_args()

    print("=" * 72)
    print("PdStrip Wavelength Table Generator")
    print("=" * 72)
    print(f"  Sea state:  Hs = {args.hs} m, Tp = {args.tp} s, gamma = {args.gamma}")
    print(f"  DP:         omega_n = {args.omega_n} rad/s, zeta = {ZETA}")
    print(f"  Grid:       N = {args.n}, omega in [{args.w_min}, {args.w_max}] rad/s")

    # Determine method and generate frequencies
    D2_interp = None
    drift_interp = None

    if args.pdstrip_file:
        print(f"  Method:     Envelope-weighted (D2*S)")
        print(f"  PdStrip:    {args.pdstrip_file}")

        D2_interp = load_drift_envelope(args.pdstrip_file)
        drift_interp = load_surge_drift_head_seas(args.pdstrip_file)

        freqs_opt = envelope_weighted_frequencies(
            args.n, args.w_min, args.w_max, D2_interp,
            args.hs, args.tp, args.omega_n, args.gamma
        )
    else:
        print(f"  Method:     Spectral-optimal (S only)")
        freqs_opt = spectral_optimal_frequencies(
            args.n, args.w_min, args.w_max,
            args.hs, args.tp, args.omega_n, args.gamma
        )

    # Convert to wavelengths
    # freqs_opt is ascending in omega -> descending in wavelength
    # PdStrip wants ascending wavelengths
    wavelengths = omega_to_wavelength(freqs_opt)[::-1]

    # --- Output wavelength table ---
    print("\n" + "=" * 72)
    print("WAVELENGTH TABLE (ascending, metres) — paste into pdstrip.inp")
    print("=" * 72)
    wl_str = " ".join(f"{wl:.3f}" for wl in wavelengths)
    print(wl_str)

    print(f"\n  Count: {len(wavelengths)}")
    print(f"  Range: {wavelengths[0]:.1f} m  to  {wavelengths[-1]:.1f} m")

    # Corresponding frequencies (descending, for reference)
    freqs_desc = freqs_opt[::-1]
    print(f"\n  Frequencies (rad/s, descending):")
    for i in range(0, len(freqs_desc), 10):
        chunk = freqs_desc[i:i + 10]
        print("    " + "  ".join(f"{w:.4f}" for w in chunk))

    # --- Grid statistics ---
    print_grid_stats(freqs_opt, "Optimal grid")

    # --- Drift force accuracy diagnostics ---
    if drift_interp is not None:
        print("\n" + "-" * 72)
        print("DRIFT FORCE ACCURACY ESTIMATE (surge, head seas)")
        print("-" * 72)

        # Use wider range for reference to be fair
        w_min_ref = min(args.w_min, 0.20)
        w_max_ref = max(args.w_max, 2.50)

        sigma_opt = print_drift_accuracy(
            freqs_desc, args.hs, args.tp, args.omega_n,
            drift_interp, "Optimal"
        )

        sigma_ref = compute_reference_sigma(
            args.hs, args.tp, args.omega_n, drift_interp,
            w_min_ref, w_max_ref
        )
        print(f"\n  Reference (N=2000 linear, [{w_min_ref},{w_max_ref}]): "
              f"sigma = {sigma_ref:.2f} kN")

        if sigma_opt is not None and sigma_ref > 0:
            err = (sigma_opt - sigma_ref) / sigma_ref * 100
            print(f"  Optimal grid error: {err:+.1f}%")

        if args.compare:
            print()
            freqs_geo = geometric_frequencies(35)
            print_grid_stats(freqs_geo, "Current geometric N=35")
            sigma_geo = print_drift_accuracy(
                freqs_geo, args.hs, args.tp, args.omega_n,
                drift_interp, "Geometric N=35"
            )
            if sigma_geo is not None and sigma_ref > 0:
                err_geo = (sigma_geo - sigma_ref) / sigma_ref * 100
                print(f"  Geometric N=35 error: {err_geo:+.1f}%")

            print(f"\n  Improvement: {err:+.1f}% vs {err_geo:+.1f}%")

    elif args.compare:
        print("\n  (Drift accuracy comparison requires --pdstrip-file)")

    # --- Newman cost estimate ---
    n_dir = 36  # typical direction count
    ops_new = args.n * args.n * n_dir
    ops_old = 35 * 35 * n_dir
    print(f"\n  Newman cost: {ops_new:,} ops/step "
          f"({ops_new / ops_old:.1f}x vs current N=35)")

    print()


if __name__ == "__main__":
    main()
