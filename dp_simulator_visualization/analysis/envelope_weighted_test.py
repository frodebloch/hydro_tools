"""All-DOF, all-heading drift coefficient analysis.

Check whether D(ω) has a similar "important frequency band" across
all DOFs and headings. If so, a single drift-weighted importance
function can work robustly for dynamic operations.
"""

import sys
from pathlib import Path
import csv
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent))
from drift_force_resolution import (
    jonswap, geometric_frequencies, frequency_steps,
    vessel_transfer_sq, compute_sv_drift_variance_spectrum,
    PI, BEAUFORT, ZETA, MU_BINS, MU_MAX, W_MIN, W_MAX,
)
from optimization_analysis import compute_filtered_sigma

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_all_drift_coefficients(filepath):
    """Parse surge, sway, yaw drift coefficients for all headings, speed=0.

    Returns
    -------
    freqs : array (n_freq,) descending
    directions : array (n_dir,)
    surge_d : array (n_freq, n_dir)
    sway_d : array (n_freq, n_dir)
    yaw_d : array (n_freq, n_dir)
    """
    data = {}
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for row in reader:
            freq = float(row[0])
            angle = float(row[2])
            speed = float(row[3])
            if abs(speed) < 0.001:
                surge_d = float(row[16])
                sway_d = float(row[17])
                yaw_d = float(row[18])
                data[(freq, angle)] = (surge_d, sway_d, yaw_d)

    freqs = sorted(set(k[0] for k in data), reverse=True)
    dirs = sorted(set(k[1] for k in data))

    n_f, n_d = len(freqs), len(dirs)
    surge = np.zeros((n_f, n_d))
    sway = np.zeros((n_f, n_d))
    yaw = np.zeros((n_f, n_d))

    for i, f in enumerate(freqs):
        for j, d in enumerate(dirs):
            if (f, d) in data:
                surge[i, j], sway[i, j], yaw[i, j] = data[(f, d)]

    return np.array(freqs), np.array(dirs), surge, sway, yaw


def compute_envelope_importance(freqs, dirs, surge_d, sway_d, yaw_d):
    """Compute D²(ω) envelope — max across all DOFs and headings.

    This represents the worst-case importance of each frequency
    regardless of heading or which DOF dominates.
    """
    # For each frequency, take max D² across all DOFs and all headings
    D2_surge = np.max(surge_d**2, axis=1)  # max over headings
    D2_sway = np.max(sway_d**2, axis=1)
    D2_yaw = np.max(yaw_d**2, axis=1)

    # Envelope: max across DOFs
    D2_envelope = np.maximum(np.maximum(D2_surge, D2_sway), D2_yaw)
    return D2_envelope, D2_surge, D2_sway, D2_yaw


def envelope_weighted_optimal(n, w_min, w_max, D2_envelope_interp,
                               hs, tp, omega_n, gamma=3.3):
    """Spectral-optimal weighted by D²_envelope(ω) — robust across DOFs/headings."""
    w_fine = np.linspace(w_min, w_max, 10000)
    S = jonswap(w_fine, hs, tp, gamma)
    D2 = D2_envelope_interp(w_fine)
    D2 = np.maximum(D2, 0)  # ensure non-negative after interpolation

    S_shifted = jonswap(w_fine + omega_n, hs, tp, gamma)
    D2_shifted = D2_envelope_interp(np.clip(w_fine + omega_n, w_min, w_max))
    D2_shifted = np.maximum(D2_shifted, 0)

    weight = np.sqrt(D2 * S * D2_shifted * S_shifted + D2**2 * S**2)
    weight = np.maximum(weight, 1e-30)

    cdf = np.cumsum(weight)
    cdf = cdf / cdf[-1]

    quantiles = np.linspace(0.0, 1.0, n + 2)[1:-1]
    w_opt = np.interp(quantiles, cdf, w_fine)
    w_opt[0] = max(w_opt[0], w_min)
    w_opt[-1] = min(w_opt[-1], w_max)

    return np.sort(w_opt)[::-1]


def envelope_weighted_optimal_dual(n, w_min, w_max, D2_envelope_interp,
                                    hs_w, tp_w, hs_s, tp_s, omega_n,
                                    gamma_w=3.3, gamma_s=5.0):
    """Envelope-weighted optimal for dual spectrum."""
    w_fine = np.linspace(w_min, w_max, 10000)

    S_w = jonswap(w_fine, hs_w, tp_w, gamma_w)
    S_s = jonswap(w_fine, hs_s, tp_s, gamma_s) if hs_s > 0 else np.zeros_like(w_fine)
    S = S_w + S_s

    D2 = np.maximum(D2_envelope_interp(w_fine), 0)

    S_shifted_w = jonswap(w_fine + omega_n, hs_w, tp_w, gamma_w)
    S_shifted_s = jonswap(w_fine + omega_n, hs_s, tp_s, gamma_s) if hs_s > 0 else np.zeros_like(w_fine)
    S_shifted = S_shifted_w + S_shifted_s

    D2_shifted = np.maximum(D2_envelope_interp(np.clip(w_fine + omega_n, w_min, w_max)), 0)

    weight = np.sqrt(D2 * S * D2_shifted * S_shifted + D2**2 * S**2)
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
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse all DOFs
    freqs, dirs, surge_d, sway_d, yaw_d = parse_all_drift_coefficients(pdstrip_path)
    D2_env, D2_surge, D2_sway, D2_yaw = compute_envelope_importance(
        freqs, dirs, surge_d, sway_d, yaw_d
    )

    # Also parse single-heading surge for comparison
    from drift_force_resolution import parse_drift_coefficients
    ref_freqs, ref_drift = parse_drift_coefficients(pdstrip_path)
    drift_interp_surge = interp1d(
        ref_freqs[::-1], ref_drift[::-1],
        kind="cubic", fill_value="extrapolate"
    )

    # Envelope interpolator
    D2_env_interp = interp1d(
        freqs[::-1], D2_env[::-1],
        kind="cubic", fill_value="extrapolate"
    )

    # Per-DOF interpolators (max over headings)
    D2_surge_interp = interp1d(freqs[::-1], D2_surge[::-1], kind="cubic", fill_value="extrapolate")
    D2_sway_interp = interp1d(freqs[::-1], D2_sway[::-1], kind="cubic", fill_value="extrapolate")
    D2_yaw_interp = interp1d(freqs[::-1], D2_yaw[::-1], kind="cubic", fill_value="extrapolate")

    omega_n = 0.06

    # ===================================================================
    # 1. Plot D(ω) for all DOFs and headings
    # ===================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: surge D(ω) for all headings
    ax = axes[0, 0]
    for j, d in enumerate(dirs):
        alpha = 0.3 if abs(d - 180) > 30 else 0.8
        lw = 1.5 if abs(d - 180) < 5 else 0.5
        ax.plot(freqs, surge_d[:, j], lw=lw, alpha=alpha)
    ax.set_title("Surge drift D(ω) — all headings")
    ax.set_xlabel("ω [rad/s]")
    ax.set_ylabel("D [N/m²]")
    ax.grid(True, alpha=0.2)

    # Top-right: sway
    ax = axes[0, 1]
    for j, d in enumerate(dirs):
        alpha = 0.3
        ax.plot(freqs, sway_d[:, j], lw=0.5, alpha=alpha)
    ax.set_title("Sway drift D(ω) — all headings")
    ax.set_xlabel("ω [rad/s]")
    ax.grid(True, alpha=0.2)

    # Bottom-left: yaw
    ax = axes[1, 0]
    for j, d in enumerate(dirs):
        ax.plot(freqs, yaw_d[:, j], lw=0.5, alpha=0.3)
    ax.set_title("Yaw drift D(ω) — all headings")
    ax.set_xlabel("ω [rad/s]")
    ax.set_ylabel("D [Nm/m²]")
    ax.grid(True, alpha=0.2)

    # Bottom-right: D² envelope comparison
    ax = axes[1, 1]
    w_fine = np.linspace(W_MIN, W_MAX, 500)
    ax.plot(freqs, np.sqrt(D2_surge), "b-", lw=1.5, label="|D|_surge max over headings")
    ax.plot(freqs, np.sqrt(D2_sway), "r-", lw=1.5, label="|D|_sway max over headings")
    ax.plot(freqs, np.sqrt(D2_yaw), "g-", lw=1.5, label="|D|_yaw max over headings")
    ax.plot(freqs, np.sqrt(D2_env), "k--", lw=2, label="Envelope (max all DOFs)")
    ax.set_title("D² envelope — max across DOFs and headings")
    ax.set_xlabel("ω [rad/s]")
    ax.set_ylabel("|D|_max")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "18_all_dof_drift.png", dpi=150)
    plt.close(fig)
    print(f"Plot saved: 18_all_dof_drift.png")

    # ===================================================================
    # 2. Quantify: where does the envelope concentrate?
    # ===================================================================
    print("\n" + "=" * 80)
    print("D² ENVELOPE ANALYSIS")
    print("=" * 80)

    # What fraction of total D² is below various frequency thresholds?
    D2_total = np.trapz(D2_env, freqs[::-1])
    for w_thresh in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]:
        mask = freqs[::-1] <= w_thresh
        D2_below = np.trapz(D2_env[::-1][mask], freqs[::-1][mask])
        print(f"  ω ≤ {w_thresh:.1f}: {D2_below/D2_total*100:.1f}% of total D² envelope")

    # ===================================================================
    # 3. Compare envelope-weighted vs surge-only vs geometric — single spectrum
    # ===================================================================
    print("\n" + "=" * 80)
    print("SINGLE SPECTRUM: Envelope-weighted vs surge-only-weighted vs geometric")
    print(f"N=35, ω_n={omega_n}, ζ={ZETA}")
    print("(Errors are for SURGE HEAD SEAS drift only — apples-to-apples with baseline)")
    print("=" * 80)

    # Reference
    sigma_ref = {}
    for bf, ss in BEAUFORT.items():
        freqs_ref = geometric_frequencies(2000)
        sigma_ref[bf], _ = compute_filtered_sigma(
            freqs_ref, drift_interp_surge, ss["hs"], ss["tp"], omega_n, ZETA
        )

    N = 35
    print(f"\n  {'Strategy':<40s}", end="")
    for bf in BEAUFORT:
        print(f"  BF{bf}", end="")
    print("   Mean|err|")

    from drift_weighted_optimal import drift_weighted_optimal

    for label, gen_func in [
        ("Geometric", lambda ss: geometric_frequencies(N)),
        ("Surge-weighted optimal", lambda ss: drift_weighted_optimal(
            N, W_MIN, W_MAX, drift_interp_surge, ss["hs"], ss["tp"], omega_n)),
        ("Envelope-weighted optimal", lambda ss: envelope_weighted_optimal(
            N, W_MIN, W_MAX, D2_env_interp, ss["hs"], ss["tp"], omega_n)),
    ]:
        errors = []
        print(f"  {label:<40s}", end="")
        for bf, ss in BEAUFORT.items():
            freqs_test = gen_func(ss)
            sig, _ = compute_filtered_sigma(
                freqs_test, drift_interp_surge, ss["hs"], ss["tp"], omega_n, ZETA
            )
            err = (sig / sigma_ref[bf] - 1.0) * 100.0
            errors.append(abs(err))
            print(f" {err:+5.1f}%", end="")
        print(f"   {np.mean(errors):5.1f}%")

    # ===================================================================
    # 4. Test: does envelope-weighted hold up for SWAY at beam seas?
    # ===================================================================
    print("\n" + "=" * 80)
    print("SWAY AT BEAM SEAS (angle=90°) — does envelope-weighted still work?")
    print("=" * 80)

    # Parse sway drift at beam seas
    sway_beam_idx = np.argmin(np.abs(dirs - 90.0))
    sway_beam = sway_d[:, sway_beam_idx]
    drift_interp_sway_beam = interp1d(
        freqs[::-1], sway_beam[::-1],
        kind="cubic", fill_value="extrapolate"
    )

    # Reference for sway beam seas
    sigma_ref_sway = {}
    for bf, ss in BEAUFORT.items():
        freqs_ref = geometric_frequencies(2000)
        D_ref = drift_interp_sway_beam(freqs_ref)
        mu_c, S_F, _, _ = compute_sv_drift_variance_spectrum(
            freqs_ref, D_ref, ss["hs"], ss["tp"], MU_BINS, MU_MAX
        )
        H2 = vessel_transfer_sq(mu_c, omega_n, ZETA)
        d_mu = MU_MAX / MU_BINS
        sigma_ref_sway[bf] = np.sqrt(np.sum(S_F * H2) * d_mu)

    print(f"\n  {'Strategy':<40s}", end="")
    for bf in BEAUFORT:
        print(f"  BF{bf}", end="")
    print("   Mean|err|")

    for label, gen_func in [
        ("Geometric", lambda ss: geometric_frequencies(N)),
        ("Envelope-weighted optimal", lambda ss: envelope_weighted_optimal(
            N, W_MIN, W_MAX, D2_env_interp, ss["hs"], ss["tp"], omega_n)),
    ]:
        errors = []
        print(f"  {label:<40s}", end="")
        for bf, ss in BEAUFORT.items():
            freqs_test = gen_func(ss)
            D_test = drift_interp_sway_beam(freqs_test)
            mu_c, S_F, _, _ = compute_sv_drift_variance_spectrum(
                freqs_test, D_test, ss["hs"], ss["tp"], MU_BINS, MU_MAX
            )
            H2 = vessel_transfer_sq(mu_c, omega_n, ZETA)
            d_mu = MU_MAX / MU_BINS
            sig = np.sqrt(np.sum(S_F * H2) * d_mu)
            err = (sig / sigma_ref_sway[bf] - 1.0) * 100.0
            errors.append(abs(err))
            print(f" {err:+5.1f}%", end="")
        print(f"   {np.mean(errors):5.1f}%")

    # ===================================================================
    # 5. Test: multiple headings simultaneously
    # ===================================================================
    print("\n" + "=" * 80)
    print("ALL HEADINGS — worst-case error across headings (surge DOF)")
    print("=" * 80)

    # Build per-heading surge drift interpolators
    surge_interps = {}
    for j, d in enumerate(dirs):
        surge_interps[d] = interp1d(
            freqs[::-1], surge_d[::-1, j],
            kind="cubic", fill_value="extrapolate"
        )

    test_headings = [0, 30, 60, 90, 120, 150, 180, 210, 240]

    for label, gen_func in [
        ("Geometric N=35", lambda ss: geometric_frequencies(35)),
        ("Envelope-weighted N=35", lambda ss: envelope_weighted_optimal(
            35, W_MIN, W_MAX, D2_env_interp, ss["hs"], ss["tp"], omega_n)),
        ("Geometric N=70", lambda ss: geometric_frequencies(70)),
        ("Envelope-weighted N=70", lambda ss: envelope_weighted_optimal(
            70, W_MIN, W_MAX, D2_env_interp, ss["hs"], ss["tp"], omega_n)),
    ]:
        print(f"\n  {label}:")
        for bf in [5, 7]:
            ss = BEAUFORT[bf]
            worst_err = 0
            worst_heading = 0
            best_err = -100
            best_heading = 0

            for heading in test_headings:
                h_idx = np.argmin(np.abs(dirs - heading))
                d_interp = surge_interps[dirs[h_idx]]

                # Reference
                freqs_ref = geometric_frequencies(2000)
                D_ref = d_interp(freqs_ref)
                mu_c, S_F, _, _ = compute_sv_drift_variance_spectrum(
                    freqs_ref, D_ref, ss["hs"], ss["tp"], MU_BINS, MU_MAX
                )
                H2 = vessel_transfer_sq(mu_c, omega_n, ZETA)
                d_mu = MU_MAX / MU_BINS
                sig_ref = np.sqrt(np.sum(S_F * H2) * d_mu)

                if sig_ref < 1.0:  # skip negligible cases
                    continue

                # Test
                freqs_test = gen_func(ss)
                D_test = d_interp(freqs_test)
                mu_c, S_F, _, _ = compute_sv_drift_variance_spectrum(
                    freqs_test, D_test, ss["hs"], ss["tp"], MU_BINS, MU_MAX
                )
                sig_test = np.sqrt(np.sum(S_F * H2) * d_mu)
                err = (sig_test / sig_ref - 1.0) * 100.0

                if err < worst_err:
                    worst_err = err
                    worst_heading = heading
                if err > best_err:
                    best_err = err
                    best_heading = heading

            print(f"    BF{bf}: worst {worst_err:+.1f}% at {worst_heading}°, "
                  f"best {best_err:+.1f}% at {best_heading}°")

    # ===================================================================
    # 6. Dual spectrum with envelope weighting
    # ===================================================================
    print("\n" + "=" * 80)
    print("DUAL SPECTRUM: Envelope-weighted")
    print("=" * 80)

    from dual_spectrum_test import compute_filtered_sigma_dual

    DUAL_SEAS = {
        "A: Wind only BF6": {"hs_w": 3.1, "tp_w": 8.5, "hs_s": 0.0, "tp_s": 12.0},
        "B: Wind BF5 + Swell 1.5m/12s": {"hs_w": 2.1, "tp_w": 7.5, "hs_s": 1.5, "tp_s": 12.0},
        "C: Wind BF5 + Swell 2.5m/14s": {"hs_w": 2.1, "tp_w": 7.5, "hs_s": 2.5, "tp_s": 14.0},
        "D: Wind BF4 + Swell 3.0m/16s": {"hs_w": 1.5, "tp_w": 6.5, "hs_s": 3.0, "tp_s": 16.0},
        "E: Wind BF6 + Swell 2.0m/12s": {"hs_w": 3.1, "tp_w": 8.5, "hs_s": 2.0, "tp_s": 12.0},
        "F: Wind BF3 + Swell 1.0m/18s": {"hs_w": 0.8, "tp_w": 5.5, "hs_s": 1.0, "tp_s": 18.0},
    }

    for N_test in [35, 50, 70]:
        print(f"\n  N = {N_test}")
        print(f"  {'Case':<35s} {'Geometric':>10s} {'Env-weighted':>12s}")

        for case_name, ss in DUAL_SEAS.items():
            hs_w, tp_w = ss["hs_w"], ss["tp_w"]
            hs_s, tp_s = ss["hs_s"], ss["tp_s"]

            sig_ref, _ = compute_filtered_sigma_dual(
                geometric_frequencies(2000), drift_interp_surge,
                hs_w, tp_w, hs_s, tp_s, omega_n, ZETA
            )

            sig_geo, _ = compute_filtered_sigma_dual(
                geometric_frequencies(N_test), drift_interp_surge,
                hs_w, tp_w, hs_s, tp_s, omega_n, ZETA
            )

            sig_env, _ = compute_filtered_sigma_dual(
                envelope_weighted_optimal_dual(
                    N_test, W_MIN, W_MAX, D2_env_interp,
                    hs_w, tp_w, hs_s, tp_s, omega_n),
                drift_interp_surge,
                hs_w, tp_w, hs_s, tp_s, omega_n, ZETA
            )

            err_geo = (sig_geo / sig_ref - 1.0) * 100.0
            err_env = (sig_env / sig_ref - 1.0) * 100.0

            print(f"  {case_name:<35s} {err_geo:+10.1f}% {err_env:+12.1f}%")

    # ===================================================================
    # 7. Frequency distribution plot
    # ===================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ss = BEAUFORT[7]
    w_fine = np.linspace(W_MIN, W_MAX, 2000)
    S_fine = jonswap(w_fine, ss["hs"], ss["tp"])
    D2_fine = np.maximum(D2_env_interp(w_fine), 0)

    # Importance function
    importance = D2_fine * S_fine

    ax1.fill_between(w_fine, S_fine / S_fine.max(), alpha=0.2, color="blue",
                     label="S(ω) normalized")
    ax1.plot(w_fine, np.sqrt(D2_fine) / np.sqrt(D2_fine).max(), "r-", lw=1,
             label="|D|_envelope normalized")
    ax1.plot(w_fine, importance / importance.max(), "k-", lw=2,
             label="D²·S (importance)")

    f_geo = geometric_frequencies(35)
    f_env = envelope_weighted_optimal(35, W_MIN, W_MAX, D2_env_interp,
                                       ss["hs"], ss["tp"], omega_n)

    ax1.plot(f_geo, np.full_like(f_geo, 0.85), "|", color="C0", ms=12, mew=1.5,
             label="Geometric N=35")
    ax1.plot(f_env, np.full_like(f_env, 0.65), "|", color="C3", ms=12, mew=1.5,
             label="Envelope-weighted N=35")

    ax1.set_title(f"BF7 — Envelope D² importance (max across all DOFs and headings)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlabel("ω [rad/s]")

    # Dual spectrum case
    ss_d = {"hs_w": 1.5, "tp_w": 6.5, "hs_s": 3.0, "tp_s": 16.0}
    S_wind = jonswap(w_fine, ss_d["hs_w"], ss_d["tp_w"])
    S_swell = jonswap(w_fine, ss_d["hs_s"], ss_d["tp_s"])
    S_dual = S_wind + S_swell
    imp_dual = D2_fine * S_dual

    ax2.fill_between(w_fine, S_dual / S_dual.max(), alpha=0.2, color="blue",
                     label="S_total normalized")
    ax2.plot(w_fine, S_wind / S_dual.max(), "b--", lw=0.8, alpha=0.5, label="Wind")
    ax2.plot(w_fine, S_swell / S_dual.max(), "r--", lw=0.8, alpha=0.5, label="Swell")
    ax2.plot(w_fine, imp_dual / imp_dual.max(), "k-", lw=2,
             label="D²·S_total (importance)")

    f_env_dual = envelope_weighted_optimal_dual(
        35, W_MIN, W_MAX, D2_env_interp,
        ss_d["hs_w"], ss_d["tp_w"], ss_d["hs_s"], ss_d["tp_s"], omega_n)

    ax2.plot(f_geo, np.full_like(f_geo, 0.85), "|", color="C0", ms=12, mew=1.5,
             label="Geometric N=35")
    ax2.plot(f_env_dual, np.full_like(f_env_dual, 0.65), "|", color="C3", ms=12, mew=1.5,
             label="Envelope-weighted N=35")

    ax2.set_title("Case D (Wind BF4 + Swell 3m/16s) — D² suppresses swell frequencies")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlabel("ω [rad/s]")

    fig.tight_layout()
    fig.savefig(output_dir / "19_envelope_weighted.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: 19_envelope_weighted.png")


if __name__ == "__main__":
    main()
