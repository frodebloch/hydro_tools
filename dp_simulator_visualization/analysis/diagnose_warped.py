"""Diagnose why warped-geometric performs worse than uniform geometric.

Hypothesis: the slowly-varying drift force variance comes from frequency
PAIRS at small difference frequency mu. When we concentrate frequencies
near the spectral peak, we get lots of pairs with VERY small mu (nearly
zero), but lose pairs at the crucial mu ~ omega_n range because we've
depleted the tails.

The key insight: we need pairs (omega_i, omega_j) where:
  1. Both omega_i and omega_j have significant spectral energy
  2. |omega_i - omega_j| is near the vessel natural frequency omega_n

Concentrating frequencies near the peak gives many pairs satisfying (1)
but with mu << omega_n, not mu ~ omega_n. We need SPREAD to get
pairs at the right difference frequency.
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
from optimization_analysis import optimal_drift_frequencies
from warped_geometric_test import warped_geometric_auto

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    ss = BEAUFORT[7]  # BF7

    # Generate the three grids
    grids = {
        "Geometric": geometric_frequencies(N),
        "Warped-geo (s=6)": warped_geometric_auto(N, W_MIN, W_MAX, ss["hs"], ss["tp"], 6),
        "Spectral-optimal": optimal_drift_frequencies(N, W_MIN, W_MAX, ss["hs"], ss["tp"], omega_n, ZETA),
    }

    # Reference
    freqs_ref = geometric_frequencies(2000)
    D_ref = drift_interp(freqs_ref)
    mu_ref, SF_ref, _, _ = compute_sv_drift_variance_spectrum(
        freqs_ref, D_ref, ss["hs"], ss["tp"], MU_BINS, MU_MAX
    )
    H2_ref = vessel_transfer_sq(mu_ref, omega_n, ZETA)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, (name, freqs) in enumerate(grids.items()):
        D = drift_interp(freqs)
        dw = frequency_steps(freqs)
        S = jonswap(freqs, ss["hs"], ss["tp"])

        mu_c, S_F, total_var, mean_drift = compute_sv_drift_variance_spectrum(
            freqs, D, ss["hs"], ss["tp"], MU_BINS, MU_MAX
        )

        # Top row: difference frequency histogram + S_F spectrum
        ax = axes[0, col]
        # Count frequency pairs in each mu bin
        pair_counts = np.zeros(MU_BINS)
        d_mu = MU_MAX / MU_BINS
        n = len(freqs)
        for i in range(n):
            for j in range(i+1, n):
                mu = abs(freqs[j] - freqs[i])
                idx = int(mu / d_mu)
                if 0 <= idx < MU_BINS:
                    pair_counts[idx] += 1

        ax.bar(mu_c, pair_counts, width=d_mu*0.9, alpha=0.4, color="C0",
               label="# freq pairs")
        ax.set_ylabel("# frequency pairs per bin", color="C0")
        ax.set_xlabel("μ = |ωj - ωi| [rad/s]")
        ax.set_title(name)

        ax2 = ax.twinx()
        ax2.plot(mu_ref, SF_ref / SF_ref.max(), "k-", alpha=0.3, lw=1,
                 label="S_F ref (N=2000)")
        ax2.plot(mu_c, S_F / max(SF_ref.max(), 1e-30), "C1-", lw=2,
                 label=f"S_F (N={N})")
        ax2.axvline(omega_n, color="red", ls="--", lw=1, alpha=0.5,
                    label=f"ω_n={omega_n}")
        ax2.set_ylabel("S_F (normalized)", color="C1")
        if col == 0:
            ax2.legend(fontsize=7, loc="upper right")

        # Bottom row: frequency spacing analysis
        ax = axes[1, col]
        # Show local spacing dw vs frequency
        freqs_asc = freqs[::-1]
        dw_local = np.diff(freqs_asc)
        w_mid = 0.5 * (freqs_asc[:-1] + freqs_asc[1:])

        ax.plot(w_mid, dw_local, "o-", ms=3, label="Δω(ω)")
        ax.axhline(omega_n, color="red", ls="--", lw=1, alpha=0.5,
                   label=f"ω_n = {omega_n}")

        # Show JONSWAP for context
        w_fine = np.linspace(W_MIN, W_MAX, 500)
        S_fine = jonswap(w_fine, ss["hs"], ss["tp"])
        S_scale = dw_local.max() / S_fine.max() * 0.5
        ax.fill_between(w_fine, S_fine * S_scale, alpha=0.15, color="blue")

        ax.set_xlabel("ω [rad/s]")
        ax.set_ylabel("Δω [rad/s]")
        ax.set_title(f"{name}: local spacing")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

        # Print statistics
        print(f"\n{name}:")
        print(f"  Δω range: {dw_local.min():.5f} – {dw_local.max():.5f} rad/s")
        print(f"  # pairs with |μ - ω_n| < 0.01: "
              f"{sum(1 for i in range(len(freqs)) for j in range(i+1,len(freqs)) if abs(abs(freqs[j]-freqs[i]) - omega_n) < 0.01)}")
        print(f"  # pairs with μ < 0.01: "
              f"{sum(1 for i in range(len(freqs)) for j in range(i+1,len(freqs)) if abs(freqs[j]-freqs[i]) < 0.01)}")

        # Weighted pair quality: sum of S(wi)*S(wj)*H(mu) for all pairs
        weighted_quality = 0.0
        for i in range(len(freqs)):
            for j in range(i+1, len(freqs)):
                mu = abs(freqs[j] - freqs[i])
                weighted_quality += S[i] * dw[i] * S[j] * dw[j] * vessel_transfer_sq(np.array([mu]), omega_n, ZETA)[0]
        print(f"  Weighted pair quality (S·dw·S·dw·H²): {weighted_quality:.6e}")

    fig.tight_layout()
    fig.savefig(output_dir / "11_diagnostic_pair_analysis.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {output_dir / '11_diagnostic_pair_analysis.png'}")


if __name__ == "__main__":
    main()
