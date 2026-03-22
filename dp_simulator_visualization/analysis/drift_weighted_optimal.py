"""Drift-weighted spectral-optimal placement.

The correct importance function for drift force variance is NOT S(ω)
but D(ω)²·S(ω) — the product of drift coefficient squared and spectral
energy. This naturally avoids wasting frequencies at swell peaks where
the QTFs are negligible.
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
from dual_spectrum_test import (
    jonswap_dual, compute_filtered_sigma_dual,
    optimal_drift_frequencies_dual,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def drift_weighted_optimal(n, w_min, w_max, drift_interp, hs, tp, omega_n,
                           gamma=3.3):
    """Spectral-optimal weighted by D(ω)² — single spectrum."""
    w_fine = np.linspace(w_min, w_max, 10000)
    S = jonswap(w_fine, hs, tp, gamma)
    D = drift_interp(w_fine)

    # Weight: D²·S captures where drift force energy actually is
    S_shifted = jonswap(w_fine + omega_n, hs, tp, gamma)
    D_shifted = drift_interp(np.clip(w_fine + omega_n, w_min, w_max))

    weight = np.sqrt(D**2 * S * D_shifted**2 * S_shifted + D**4 * S**2)
    weight = np.maximum(weight, 1e-30)

    cdf = np.cumsum(weight)
    cdf = cdf / cdf[-1]

    quantiles = np.linspace(0.0, 1.0, n + 2)[1:-1]
    w_opt = np.interp(quantiles, cdf, w_fine)
    w_opt[0] = max(w_opt[0], w_min)
    w_opt[-1] = min(w_opt[-1], w_max)

    return np.sort(w_opt)[::-1]


def drift_weighted_optimal_dual(n, w_min, w_max, drift_interp,
                                 hs_w, tp_w, hs_s, tp_s, omega_n,
                                 gamma_w=3.3, gamma_s=5.0):
    """Spectral-optimal weighted by D(ω)² — dual spectrum."""
    w_fine = np.linspace(w_min, w_max, 10000)
    S = jonswap_dual(w_fine, hs_w, tp_w, hs_s, tp_s, gamma_w, gamma_s)
    D = drift_interp(w_fine)

    S_shifted = jonswap_dual(w_fine + omega_n, hs_w, tp_w, hs_s, tp_s,
                             gamma_w, gamma_s)
    D_shifted = drift_interp(np.clip(w_fine + omega_n, w_min, w_max))

    weight = np.sqrt(D**2 * S * D_shifted**2 * S_shifted + D**4 * S**2)
    weight = np.maximum(weight, 1e-30)

    cdf = np.cumsum(weight)
    cdf = cdf / cdf[-1]

    quantiles = np.linspace(0.0, 1.0, n + 2)[1:-1]
    w_opt = np.interp(quantiles, cdf, w_fine)
    w_opt[0] = max(w_opt[0], w_min)
    w_opt[-1] = min(w_opt[-1], w_max)

    return np.sort(w_opt)[::-1]


def main():
    pdstrip_path = "/home/blofro/src/brucon/libs/simulator/vessel_model/test/config/csov_pdstrip.dat"
    output_dir = Path(__file__).parent / "output"

    ref_freqs, ref_drift = parse_drift_coefficients(pdstrip_path)
    drift_interp = interp1d(
        ref_freqs[::-1], ref_drift[::-1],
        kind="cubic", fill_value="extrapolate"
    )

    omega_n = 0.06

    # ===================================================================
    # Single-spectrum: verify drift-weighted is at least as good
    # ===================================================================
    print("=" * 80)
    print("SINGLE SPECTRUM — Drift-weighted vs energy-weighted optimal")
    print("=" * 80)

    sigma_ref = {}
    for bf, ss in BEAUFORT.items():
        freqs_ref = geometric_frequencies(2000)
        sigma_ref[bf], _ = compute_filtered_sigma(
            freqs_ref, drift_interp, ss["hs"], ss["tp"], omega_n, ZETA
        )

    N = 35
    print(f"\n  N={N}, ω_n={omega_n}:")
    print(f"  {'Strategy':<40s}", end="")
    for bf in BEAUFORT:
        print(f"  BF{bf}", end="")
    print()

    for label, gen_func in [
        ("Geometric", lambda ss: geometric_frequencies(N)),
        ("Energy-weighted optimal", lambda ss: optimal_drift_frequencies(
            N, W_MIN, W_MAX, ss["hs"], ss["tp"], omega_n, ZETA)),
        ("Drift-weighted optimal", lambda ss: drift_weighted_optimal(
            N, W_MIN, W_MAX, drift_interp, ss["hs"], ss["tp"], omega_n)),
    ]:
        print(f"  {label:<40s}", end="")
        for bf, ss in BEAUFORT.items():
            freqs = gen_func(ss)
            sig, _ = compute_filtered_sigma(
                freqs, drift_interp, ss["hs"], ss["tp"], omega_n, ZETA
            )
            err = (sig / sigma_ref[bf] - 1.0) * 100.0
            print(f" {err:+5.1f}%", end="")
        print()

    # ===================================================================
    # Dual spectrum
    # ===================================================================
    print("\n" + "=" * 80)
    print("DUAL SPECTRUM — Drift-weighted vs energy-weighted optimal")
    print("=" * 80)

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

    for N in [35, 50, 70]:
        print(f"\n  N = {N}")
        print(f"  {'Case':<35s} {'Geometric':>10s} {'E-opt dual':>10s} "
              f"{'D-opt dual':>10s} {'D-opt single':>12s}")

        for case_name, ss in DUAL_SEAS.items():
            hs_w, tp_w = ss["hs_w"], ss["tp_w"]
            hs_s, tp_s = ss["hs_s"], ss["tp_s"]

            # Reference
            sig_ref, _ = compute_filtered_sigma_dual(
                geometric_frequencies(2000), drift_interp,
                hs_w, tp_w, hs_s, tp_s, omega_n, ZETA
            )

            # Geometric
            sig_geo, _ = compute_filtered_sigma_dual(
                geometric_frequencies(N), drift_interp,
                hs_w, tp_w, hs_s, tp_s, omega_n, ZETA
            )

            # Energy-weighted dual
            sig_e_dual, _ = compute_filtered_sigma_dual(
                optimal_drift_frequencies_dual(
                    N, W_MIN, W_MAX, hs_w, tp_w, hs_s, tp_s, omega_n),
                drift_interp, hs_w, tp_w, hs_s, tp_s, omega_n, ZETA
            )

            # Drift-weighted dual
            sig_d_dual, _ = compute_filtered_sigma_dual(
                drift_weighted_optimal_dual(
                    N, W_MIN, W_MAX, drift_interp,
                    hs_w, tp_w, hs_s, tp_s, omega_n),
                drift_interp, hs_w, tp_w, hs_s, tp_s, omega_n, ZETA
            )

            # Drift-weighted single (wind only — ignore swell)
            if hs_w > 0:
                sig_d_single, _ = compute_filtered_sigma_dual(
                    drift_weighted_optimal(
                        N, W_MIN, W_MAX, drift_interp,
                        hs_w, tp_w, omega_n),
                    drift_interp, hs_w, tp_w, hs_s, tp_s, omega_n, ZETA
                )
                err_d_single = (sig_d_single / sig_ref - 1.0) * 100.0
            else:
                err_d_single = float('nan')

            err_geo = (sig_geo / sig_ref - 1.0) * 100.0
            err_e_dual = (sig_e_dual / sig_ref - 1.0) * 100.0
            err_d_dual = (sig_d_dual / sig_ref - 1.0) * 100.0

            print(f"  {case_name:<35s} {err_geo:+10.1f}% {err_e_dual:+10.1f}% "
                  f"{err_d_dual:+10.1f}% {err_d_single:+12.1f}%")

    # ===================================================================
    # Plot: frequency placement comparison for Case D
    # ===================================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    ss = DUAL_SEAS["D: Wind BF4 + Swell 3.0m/16s"]
    w_fine = np.linspace(W_MIN, W_MAX, 2000)

    S_total = jonswap_dual(w_fine, ss["hs_w"], ss["tp_w"], ss["hs_s"], ss["tp_s"])
    D_fine = drift_interp(w_fine)
    importance = D_fine**2 * S_total

    ax = axes[0]
    ax.fill_between(w_fine, S_total / S_total.max(), alpha=0.2, color="blue",
                    label="Wave spectrum S(ω)")
    ax.plot(w_fine, importance / importance.max(), "r-", lw=1.5,
            label="D(ω)²·S(ω) — drift importance")

    f_geo = geometric_frequencies(35)
    f_e_opt = optimal_drift_frequencies_dual(
        35, W_MIN, W_MAX, ss["hs_w"], ss["tp_w"], ss["hs_s"], ss["tp_s"], omega_n)
    f_d_opt = drift_weighted_optimal_dual(
        35, W_MIN, W_MAX, drift_interp,
        ss["hs_w"], ss["tp_w"], ss["hs_s"], ss["tp_s"], omega_n)

    ax.plot(f_geo, np.full_like(f_geo, 0.90), "|", color="C0", ms=12, mew=1.5,
            label="Geometric")
    ax.plot(f_e_opt, np.full_like(f_e_opt, 0.70), "|", color="C2", ms=12, mew=1.5,
            label="Energy-weighted optimal")
    ax.plot(f_d_opt, np.full_like(f_d_opt, 0.50), "|", color="C3", ms=12, mew=1.5,
            label="Drift-weighted optimal")

    ax.set_title("Case D: Wind BF4 + Swell 3.0m/16s — Frequency Placement")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("ω [rad/s]")

    # Count freqs in swell vs wind region
    w_mid = 0.7
    for label, freqs, color in [("Geometric", f_geo, "C0"),
                                  ("E-weighted", f_e_opt, "C2"),
                                  ("D-weighted", f_d_opt, "C3")]:
        n_low = np.sum(freqs < w_mid)
        n_high = np.sum(freqs >= w_mid)
        ax.annotate(f"{label}: {n_low} low / {n_high} high",
                    xy=(0.02, 0.95 - 0.06 * ["Geometric", "E-weighted", "D-weighted"].index(label)),
                    xycoords="axes fraction", fontsize=7, color=color)

    # Bottom: same for Case E (closer peaks)
    ax = axes[1]
    ss2 = DUAL_SEAS["E: Wind BF6 + Swell 2.0m/12s"]
    S_total2 = jonswap_dual(w_fine, ss2["hs_w"], ss2["tp_w"], ss2["hs_s"], ss2["tp_s"])
    importance2 = D_fine**2 * S_total2

    ax.fill_between(w_fine, S_total2 / S_total2.max(), alpha=0.2, color="blue",
                    label="Wave spectrum S(ω)")
    ax.plot(w_fine, importance2 / importance2.max(), "r-", lw=1.5,
            label="D(ω)²·S(ω)")

    f_d_opt2 = drift_weighted_optimal_dual(
        35, W_MIN, W_MAX, drift_interp,
        ss2["hs_w"], ss2["tp_w"], ss2["hs_s"], ss2["tp_s"], omega_n)
    f_geo2 = geometric_frequencies(35)

    ax.plot(f_geo2, np.full_like(f_geo2, 0.90), "|", color="C0", ms=12, mew=1.5,
            label="Geometric")
    ax.plot(f_d_opt2, np.full_like(f_d_opt2, 0.60), "|", color="C3", ms=12, mew=1.5,
            label="Drift-weighted optimal")

    ax.set_title("Case E: Wind BF6 + Swell 2.0m/12s — Frequency Placement")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("ω [rad/s]")

    fig.tight_layout()
    fig.savefig(output_dir / "17_drift_weighted_placement.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {output_dir / '17_drift_weighted_placement.png'}")


if __name__ == "__main__":
    main()
