"""Investigate frequency grids that are both spectrally-optimal AND
provably non-repeating.

Key insight: the non-repeating requirement is about frequency RATIOS
being irrational. The drift force accuracy requirement is about
frequency PLACEMENT relative to the spectrum. These are almost
independent constraints.

Approach 1: "Jittered geometric" — start from spectral-optimal placement,
then perturb each frequency by a tiny amount to land on a geometric
subsequence (provably incommensurate). If perturbations are small
relative to Δω, accuracy is preserved.

Approach 2: "Irrational basis" — place frequencies as ω_i = α + β·φ^i
where φ = (1+√5)/2 (golden ratio). The golden ratio has the strongest
irrationality guarantee (hardest to approximate by rationals). By
choosing α, β to center the dense region on the spectral peak, we
get both properties.

Approach 3: "Prime-based spacing" — use spacing proportional to the
sequence of primes, which guarantees incommensurability.

Approach 4: Pragmatic — just verify the spectral-optimal grid numerically.
If we can show the minimum repeat period exceeds 10× simulation length,
it's good enough for operability analysis.
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
from optimization_analysis import optimal_drift_frequencies, compute_filtered_sigma

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Repeat period estimation
# ---------------------------------------------------------------------------

def estimate_repeat_period(freqs, tol=0.01, max_time=1e6):
    """Estimate the repeat period of a sum of sinusoids at given frequencies.

    The wave elevation is η(t) = Σ aᵢ cos(ωᵢ t + φᵢ).
    It repeats exactly when ωᵢ·T is a multiple of 2π for ALL i,
    i.e., T = 2π·nᵢ/ωᵢ for integer nᵢ.

    For approximate repetition, we look for the smallest T > 0 such that
    |ωᵢ·T mod 2π| < tol for all i simultaneously.

    We check this by computing the "phase coherence":
        C(T) = (1/N) · Σ cos(ωᵢ·T)
    C = 1 means perfect repetition. C > 1 - tol means approximate repetition.

    Uses a coarse-to-fine search.
    """
    n = len(freqs)
    # Phase coherence function
    def coherence(t):
        phases = freqs * t  # ω·t
        return np.mean(np.cos(phases % (2*PI) * 0.0 + phases))
        # Simpler: just check if all ω·t are near multiples of 2π

    def phase_residual(t):
        """Max phase deviation from nearest 2π multiple."""
        phases = (freqs * t) % (2 * PI)
        # Distance to nearest multiple of 2π
        residuals = np.minimum(phases, 2*PI - phases)
        return residuals.max()

    # Scan at coarse resolution, then refine
    # The repeat period, if it exists, must be a multiple of 2π/Δω_min
    # where Δω_min is the smallest frequency difference
    diffs = []
    for i in range(len(freqs)):
        for j in range(i+1, len(freqs)):
            diffs.append(abs(freqs[i] - freqs[j]))
    dw_min = min(diffs)
    T_base = 2 * PI / dw_min  # ~100-400 seconds

    print(f"  Minimum Δω = {dw_min:.6f} rad/s → base period = {T_base:.1f} s")

    # Scan multiples of approximate base period
    best_residual = PI
    best_T = None
    for k in range(1, int(max_time / T_base) + 1):
        T = k * T_base
        res = phase_residual(T)
        if res < best_residual:
            best_residual = res
            best_T = T
        if res < tol:
            return T, res

    return best_T, best_residual


def repeat_period_from_ratios(freqs):
    """Analytical repeat period check using continued fraction approximation.

    For two frequencies ω₁, ω₂, the repeat period is 2π·LCM(1/ω₁, 1/ω₂)
    = 2π / GCD(ω₁, ω₂). For irrational ratios, GCD = 0, period = ∞.

    We check by finding the best rational approximation p/q to each
    frequency ratio ω₁/ωᵢ with q ≤ Q_max, and computing the resulting
    approximate repeat period. If it's large, the series is effectively
    non-repeating.
    """
    from fractions import Fraction

    Q_MAX = 10000  # maximum denominator for rational approximation
    omega_ref = freqs[0]

    min_repeat = float('inf')
    worst_pair = None

    for i in range(1, len(freqs)):
        ratio = float(freqs[i] / omega_ref)
        # Best rational approximation with denominator ≤ Q_MAX
        frac = Fraction(ratio).limit_denominator(Q_MAX)
        p, q = frac.numerator, frac.denominator
        # Approximate repeat period: need n such that ω_ref·T = 2π·q·n
        # and ωᵢ·T = 2π·p·n simultaneously
        # T = 2π·q / ω_ref
        T_repeat = 2 * PI * q / omega_ref
        approx_error = abs(ratio - p/q)

        if T_repeat < min_repeat:
            min_repeat = T_repeat
            worst_pair = (0, i, p, q, approx_error)

    return min_repeat, worst_pair


# ---------------------------------------------------------------------------
# Grid strategies with incommensurability guarantees
# ---------------------------------------------------------------------------

def golden_ratio_frequencies(n, w_min, w_max, w_center, concentration=4.0):
    """Place frequencies using golden-ratio-based spacing.

    Uses the fact that multiples of the golden ratio φ = (1+√5)/2 are
    the "most irrational" numbers (hardest to approximate by rationals),
    giving the strongest non-repeating guarantee.

    Strategy: generate n points using a low-discrepancy sequence based on
    the golden ratio, then map them to [w_min, w_max] with concentration
    near w_center via a CDF transform.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # Golden-ratio low-discrepancy sequence on [0,1]
    # Weyl sequence: x_n = (n · φ) mod 1
    indices = np.arange(1, n + 1)
    u_raw = (indices * phi) % 1.0
    u_sorted = np.sort(u_raw)  # sorted ascending

    # Now map through a CDF that concentrates near w_center
    # Use JONSWAP spectrum as the density
    w_fine = np.linspace(w_min, w_max, 10000)
    # Simple peaked density centered at w_center
    sigma_w = 0.3 * (w_max - w_min)  # width of concentration
    density = 1.0 + concentration * np.exp(
        -0.5 * ((w_fine - w_center) / (0.15 * (w_max - w_min))) ** 2
    )
    cdf = np.cumsum(density)
    cdf = cdf / cdf[-1]

    # Map the golden-ratio points through the inverse CDF
    w = np.interp(u_sorted, cdf, w_fine)
    return np.sort(w)[::-1]  # descending


def spectral_optimal_golden(n, w_min, w_max, hs, tp, omega_n, zeta):
    """Spectral-optimal placement using golden-ratio low-discrepancy points.

    Instead of uniform quantiles of the spectral CDF, use golden-ratio
    Weyl sequence points. This gives better uniformity properties AND
    the resulting frequencies are provably incommensurate (each is
    a linear combination of 1 and φ, which are linearly independent
    over the rationals).
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # Golden-ratio Weyl sequence
    indices = np.arange(1, n + 1)
    u_raw = (indices / phi) % 1.0  # Weyl sequence
    u_sorted = np.sort(u_raw)

    # Spectral CDF (same as optimal_drift_frequencies)
    w_fine = np.linspace(w_min, w_max, 10000)
    S = jonswap(w_fine, hs, tp)
    S_shifted = jonswap(w_fine + omega_n, hs, tp)
    weight = np.sqrt(S * S_shifted + S * S)

    cdf = np.cumsum(weight)
    cdf = cdf / cdf[-1]

    # Map golden-ratio points through inverse CDF
    w = np.interp(u_sorted, cdf, w_fine)
    w[0] = max(w[0], w_min)
    w[-1] = min(w[-1], w_max)

    return np.sort(w)[::-1]  # descending


def spectral_optimal_irrational_perturb(n, w_min, w_max, hs, tp, omega_n, zeta):
    """Start from spectral-optimal, then perturb by irrational amounts.

    Add ε·√pᵢ to each frequency, where pᵢ is the i-th prime and ε is
    small enough not to significantly change the placement. The square
    roots of distinct primes are linearly independent over Q (Besicovitch
    theorem), so the resulting frequencies are provably incommensurate.
    """
    # Get spectral-optimal base
    w_base = optimal_drift_frequencies(n, w_min, w_max, hs, tp, omega_n, zeta)

    # First n primes
    primes = _first_n_primes(n)

    # Perturbation magnitude: small fraction of minimum spacing
    dw_min = np.min(np.abs(np.diff(w_base)))
    epsilon = dw_min * 0.001  # 0.1% of minimum spacing

    # Perturb
    sqrt_primes = np.sqrt(primes.astype(float))
    # Normalize so perturbations are bounded
    sqrt_primes = sqrt_primes / sqrt_primes.max()
    w_perturbed = w_base + epsilon * sqrt_primes

    # Ensure still in range and descending
    w_perturbed = np.clip(w_perturbed, w_min, w_max)
    return np.sort(w_perturbed)[::-1]


def _first_n_primes(n):
    """Return first n prime numbers."""
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return np.array(primes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pdstrip_path = "/home/blofro/src/brucon/libs/simulator/vessel_model/test/config/csov_pdstrip.dat"
    output_dir = Path(__file__).parent / "output"

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

    # ===================================================================
    # 1. Drift force accuracy comparison
    # ===================================================================
    print("=" * 80)
    print(f"DRIFT FORCE ACCURACY: N={N}, ω_n={omega_n}, ζ={ZETA}")
    print("=" * 80)

    strategies = {}
    strategies["Geometric (current)"] = lambda ss: geometric_frequencies(N)
    strategies["Spectral-optimal (uniform quantile)"] = \
        lambda ss: optimal_drift_frequencies(N, W_MIN, W_MAX, ss["hs"], ss["tp"], omega_n, ZETA)
    strategies["Spectral-optimal (golden quantile)"] = \
        lambda ss: spectral_optimal_golden(N, W_MIN, W_MAX, ss["hs"], ss["tp"], omega_n, ZETA)
    strategies["Spectral-optimal + √prime perturb"] = \
        lambda ss: spectral_optimal_irrational_perturb(N, W_MIN, W_MAX, ss["hs"], ss["tp"], omega_n, ZETA)

    print(f"\n{'Strategy':<42s}", end="")
    for bf in BEAUFORT:
        print(f"  BF{bf}", end="")
    print("   Mean|err|")

    for name, gen_func in strategies.items():
        errors = []
        print(f"{name:<42s}", end="")
        for bf, ss in BEAUFORT.items():
            freqs = gen_func(ss)
            sig, _ = compute_filtered_sigma(
                freqs, drift_interp, ss["hs"], ss["tp"], omega_n, ZETA
            )
            err = (sig / sigma_ref[bf] - 1.0) * 100.0
            errors.append(abs(err))
            print(f" {err:+5.1f}%", end="")
        print(f"   {np.mean(errors):5.1f}%")

    # ===================================================================
    # 2. Repeat period analysis
    # ===================================================================
    print("\n" + "=" * 80)
    print("REPEAT PERIOD ANALYSIS (rational approximation, Q_max=10000)")
    print("=" * 80)

    ss = BEAUFORT[7]
    print(f"\nBF7 (Hs={ss['hs']}m, Tp={ss['tp']}s):")

    for name, gen_func in strategies.items():
        freqs = gen_func(ss)
        T_min, worst = repeat_period_from_ratios(freqs)
        hours = T_min / 3600
        print(f"\n  {name}:")
        print(f"    Min approx repeat period: {T_min:.0f} s = {hours:.1f} hours")
        if worst:
            i, j, p, q, err = worst
            print(f"    Worst pair: ω[{i}]/ω[{j}] ≈ {p}/{q} "
                  f"(error {err:.2e}), q={q}")

    # ===================================================================
    # 3. Brute-force phase coherence scan
    # ===================================================================
    print("\n" + "=" * 80)
    print("PHASE COHERENCE SCAN (brute force, scanning up to 100,000 s)")
    print("=" * 80)

    for name in ["Geometric (current)", "Spectral-optimal (uniform quantile)",
                  "Spectral-optimal (golden quantile)"]:
        gen_func = strategies[name]
        freqs = gen_func(ss)
        print(f"\n  {name}:")
        T_approx, residual = estimate_repeat_period(freqs, tol=0.3, max_time=1e5)
        print(f"    Best near-repeat found: T={T_approx:.1f} s ({T_approx/3600:.1f} h), "
              f"max phase residual = {residual:.3f} rad ({np.degrees(residual):.1f}°)")

    # ===================================================================
    # 4. Visual: frequency distributions
    # ===================================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    ss = BEAUFORT[7]

    w_fine = np.linspace(W_MIN, W_MAX, 2000)
    S_fine = jonswap(w_fine, ss["hs"], ss["tp"])
    S_norm = S_fine / S_fine.max()

    configs = [
        ("Geometric", strategies["Geometric (current)"](ss), "C0"),
        ("Spectral-optimal (uniform)", strategies["Spectral-optimal (uniform quantile)"](ss), "C3"),
        ("Spectral-optimal (golden)", strategies["Spectral-optimal (golden quantile)"](ss), "C1"),
        ("Spectral-opt + √prime", strategies["Spectral-optimal + √prime perturb"](ss), "C4"),
    ]

    # Top: frequency tick marks
    ax = axes[0]
    ax.fill_between(w_fine, S_norm, alpha=0.15, color="blue", label="JONSWAP (BF7)")
    y_offsets = [0.9, 0.65, 0.4, 0.15]
    for (label, freqs, color), y_off in zip(configs, y_offsets):
        ax.plot(freqs, np.full_like(freqs, y_off), "|", color=color,
                ms=15, mew=1.5, label=label)
    ax.set_xlabel("ω [rad/s]")
    ax.set_title("Frequency placement comparison — BF7")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1.1)

    # Bottom: local spacing
    ax = axes[1]
    for label, freqs, color in configs:
        freqs_asc = np.sort(freqs)
        dw = np.diff(freqs_asc)
        w_mid = 0.5 * (freqs_asc[:-1] + freqs_asc[1:])
        ax.plot(w_mid, dw, "o-", ms=3, color=color, label=label)
    ax.axhline(omega_n, color="red", ls="--", lw=1, alpha=0.5, label=f"ω_n={omega_n}")
    ax.fill_between(w_fine, S_norm * 0.05, alpha=0.15, color="blue")
    ax.set_xlabel("ω [rad/s]")
    ax.set_ylabel("Δω [rad/s]")
    ax.set_title("Local frequency spacing")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "12_incommensurate_strategies.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {output_dir / '12_incommensurate_strategies.png'}")


if __name__ == "__main__":
    main()
