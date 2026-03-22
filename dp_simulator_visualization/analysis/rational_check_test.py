"""Verify spectral-optimal frequencies are incommensurate, fix if not.

Pragmatic approach: generate optimal grid, check all pairs for near-rational
ratios, nudge any offenders by a tiny epsilon.

For operability analysis: a 3-hour simulation with wave periods 3-20s means
we need the repeat period to exceed ~10800s. If the closest rational
approximation p/q to any ratio ω_i/ω_j has q ≤ Q, the approximate repeat
period for that pair is T ≈ 2π·q/ω_j. We need T >> 3 hours for all pairs.

With Q_max = 1000 and ω_min = 0.3 rad/s:
    T_min = 2π·1000/0.3 ≈ 20,944 s ≈ 5.8 hours — safe for 3h simulations.

With Q_max = 500:
    T_min = 2π·500/0.3 ≈ 10,472 s ≈ 2.9 hours — marginal.

So Q_max = 1000 is a reasonable threshold.
"""

import sys
from pathlib import Path
import numpy as np
from fractions import Fraction
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent))
from drift_force_resolution import (
    jonswap, geometric_frequencies, frequency_steps,
    parse_drift_coefficients, compute_sv_drift_variance_spectrum,
    vessel_transfer_sq,
    PI, BEAUFORT, ZETA, MU_BINS, MU_MAX, W_MIN, W_MAX,
)
from optimization_analysis import optimal_drift_frequencies, compute_filtered_sigma


def check_and_fix_rationals(freqs, q_max=1000, max_nudges=100):
    """Check all frequency pairs for near-rational ratios and fix them.

    For each pair (i, j), compute ω_i/ω_j and find best rational p/q
    with q ≤ q_max. If such an approximation exists (meaning the ratio
    is "too close" to rational), nudge ω_j by a small amount.

    Parameters
    ----------
    freqs : array, descending
        Frequency grid.
    q_max : int
        Maximum denominator. Pairs with best-rational denominator ≤ q_max
        are flagged as "too rational."
    max_nudges : int
        Safety limit on number of nudge iterations.

    Returns
    -------
    freqs_fixed : array, descending
        Fixed frequency grid.
    n_fixes : int
        Number of frequencies nudged.
    """
    freqs = freqs.copy()
    n = len(freqs)
    n_fixes = 0

    for iteration in range(max_nudges):
        # Find worst (smallest q) pair
        worst_q = q_max + 1
        worst_pair = None

        for i in range(n):
            for j in range(i + 1, n):
                ratio = float(freqs[i] / freqs[j])
                frac = Fraction(ratio).limit_denominator(q_max)
                q = frac.denominator
                approx_error = abs(ratio - frac.numerator / frac.denominator)

                # Check if this is a "dangerously good" approximation
                # The approximation error should be < 1/(2·q²) for it to be
                # the convergent of the continued fraction (i.e., genuinely close)
                threshold = 1.0 / (2.0 * q * q)
                if approx_error < threshold and q <= worst_q:
                    worst_q = q
                    worst_pair = (i, j, frac, approx_error)

        if worst_pair is None:
            break

        i, j, frac, err = worst_pair
        # Nudge freq[j] by a small amount — 0.1% of local spacing
        local_dw = abs(freqs[max(0, j-1)] - freqs[min(n-1, j+1)]) / 2
        if local_dw == 0:
            local_dw = abs(freqs[0] - freqs[-1]) / n
        nudge = local_dw * 0.001 * (1 + iteration * 0.1)  # grow if repeated
        freqs[j] += nudge  # shift up slightly
        n_fixes += 1

        if n_fixes <= 5:
            print(f"    Fix {n_fixes}: ω[{i}]/ω[{j}] ≈ {frac} (q={frac.denominator}, "
                  f"err={err:.2e}), nudged ω[{j}] by {nudge:.6f} rad/s")

    # Re-sort descending
    freqs = np.sort(freqs)[::-1]
    return freqs, n_fixes


def analyze_rationality(freqs, label, q_max=1000):
    """Report the rationality properties of a frequency grid."""
    n = len(freqs)
    min_q = float('inf')
    worst_pair = None
    near_rational_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            ratio = float(freqs[i] / freqs[j])
            frac = Fraction(ratio).limit_denominator(q_max)
            q = frac.denominator
            approx_error = abs(ratio - frac.numerator / frac.denominator)
            threshold = 1.0 / (2.0 * q * q)

            if approx_error < threshold:
                near_rational_count += 1
                if q < min_q:
                    min_q = q
                    worst_pair = (i, j, frac, approx_error)

    # Minimum repeat period estimate
    if worst_pair:
        _, j, frac, _ = worst_pair
        T_min = 2 * PI * frac.denominator / freqs[j]
    else:
        T_min = float('inf')

    print(f"\n  {label}:")
    print(f"    Near-rational pairs (q ≤ {q_max}): {near_rational_count} of {n*(n-1)//2}")
    if worst_pair:
        i, j, frac, err = worst_pair
        print(f"    Worst: ω[{i}]/ω[{j}] = {freqs[i]:.6f}/{freqs[j]:.6f} "
              f"≈ {frac} (q={frac.denominator}, err={err:.2e})")
        print(f"    Min approx repeat period: {T_min:.0f} s = {T_min/3600:.1f} hours")
    else:
        print(f"    No near-rational pairs found → repeat period > {T_min} hours")
    return T_min


def main():
    pdstrip_path = "/home/blofro/src/brucon/libs/simulator/vessel_model/test/config/csov_pdstrip.dat"

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

    print("=" * 80)
    print("INCOMMENSURABILITY CHECK AND FIX")
    print("=" * 80)

    for bf in [5, 7]:
        ss = BEAUFORT[bf]
        print(f"\n{'='*60}")
        print(f"BF{bf}: Hs={ss['hs']}m, Tp={ss['tp']}s")
        print(f"{'='*60}")

        # 1. Standard geometric — baseline
        freqs_geo = geometric_frequencies(N)
        analyze_rationality(freqs_geo, "Geometric (current)")

        # 2. Spectral-optimal — before fix
        freqs_opt = optimal_drift_frequencies(
            N, W_MIN, W_MAX, ss["hs"], ss["tp"], omega_n, ZETA
        )
        analyze_rationality(freqs_opt, "Spectral-optimal (raw)")

        # 3. Spectral-optimal — after fix
        print(f"\n  Applying rational-ratio fix (q_max=1000):")
        freqs_fixed, n_fixes = check_and_fix_rationals(freqs_opt, q_max=1000)
        print(f"    Total fixes: {n_fixes}")
        analyze_rationality(freqs_fixed, "Spectral-optimal (fixed)")

        # 4. Compare accuracy before/after fix
        sig_geo, _ = compute_filtered_sigma(
            freqs_geo, drift_interp, ss["hs"], ss["tp"], omega_n, ZETA
        )
        sig_opt, _ = compute_filtered_sigma(
            freqs_opt, drift_interp, ss["hs"], ss["tp"], omega_n, ZETA
        )
        sig_fix, _ = compute_filtered_sigma(
            freqs_fixed, drift_interp, ss["hs"], ss["tp"], omega_n, ZETA
        )

        err_geo = (sig_geo / sigma_ref[bf] - 1.0) * 100.0
        err_opt = (sig_opt / sigma_ref[bf] - 1.0) * 100.0
        err_fix = (sig_fix / sigma_ref[bf] - 1.0) * 100.0

        print(f"\n  Drift force accuracy:")
        print(f"    Geometric:             {err_geo:+.1f}%")
        print(f"    Spectral-optimal:      {err_opt:+.1f}%")
        print(f"    Spectral-optimal+fix:  {err_fix:+.1f}%")
        print(f"    Accuracy loss from fix: {abs(err_fix - err_opt):.2f}%")

    # Also test with more frequencies
    print(f"\n\n{'='*80}")
    print("SAME ANALYSIS AT N=50 and N=70")
    print("=" * 80)

    for N_test in [50, 70]:
        ss = BEAUFORT[7]
        print(f"\n  N={N_test}, BF7:")

        freqs_opt = optimal_drift_frequencies(
            N_test, W_MIN, W_MAX, ss["hs"], ss["tp"], omega_n, ZETA
        )
        T_before = analyze_rationality(freqs_opt, f"N={N_test} spectral-optimal (raw)")

        freqs_fixed, n_fixes = check_and_fix_rationals(freqs_opt, q_max=1000)
        print(f"    Fixes applied: {n_fixes}")
        T_after = analyze_rationality(freqs_fixed, f"N={N_test} spectral-optimal (fixed)")

        sig_opt, _ = compute_filtered_sigma(
            freqs_opt, drift_interp, ss["hs"], ss["tp"], omega_n, ZETA
        )
        sig_fix, _ = compute_filtered_sigma(
            freqs_fixed, drift_interp, ss["hs"], ss["tp"], omega_n, ZETA
        )
        err_opt = (sig_opt / sigma_ref[7] - 1.0) * 100.0
        err_fix = (sig_fix / sigma_ref[7] - 1.0) * 100.0
        print(f"    Accuracy: raw {err_opt:+.1f}%, fixed {err_fix:+.1f}%, "
              f"loss {abs(err_fix - err_opt):.2f}%")


if __name__ == "__main__":
    main()
