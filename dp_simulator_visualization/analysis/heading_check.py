"""Quick check: drift coefficients and optimization performance at specific headings.

Focus on 30° off bow (PdStrip 150°) and beam seas (PdStrip 90°) as
representative operational conditions.
"""

import sys
from pathlib import Path
import csv
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent))
from drift_force_resolution import (
    jonswap, geometric_frequencies, frequency_steps,
    parse_drift_coefficients, compute_sv_drift_variance_spectrum,
    vessel_transfer_sq,
    PI, BEAUFORT, ZETA, MU_BINS, MU_MAX, W_MIN, W_MAX,
)
from optimization_analysis import compute_filtered_sigma

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_drift_all_dofs_at_heading(filepath, target_angle):
    """Parse surge, sway, yaw drift at a specific heading, speed=0."""
    freqs = []
    surge = []
    sway = []
    yaw = []
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            freq = float(row[0])
            angle = float(row[2])
            speed = float(row[3])
            if abs(speed) < 0.001 and abs(angle - target_angle) < 0.1:
                freqs.append(freq)
                surge.append(float(row[16]))
                sway.append(float(row[17]))
                yaw.append(float(row[18]))
    return np.array(freqs), np.array(surge), np.array(sway), np.array(yaw)


def compute_filtered_sigma_generic(freqs, D, hs, tp, omega_n, zeta):
    """Compute vessel-filtered RMS drift for arbitrary D(ω) array."""
    mu_c, S_F, total_var, mean_drift = compute_sv_drift_variance_spectrum(
        freqs, D, hs, tp, MU_BINS, MU_MAX
    )
    H2 = vessel_transfer_sq(mu_c, omega_n, zeta)
    d_mu = MU_MAX / MU_BINS
    filt_var = np.sum(S_F * H2) * d_mu
    return np.sqrt(filt_var), mean_drift


def envelope_weighted_optimal(n, w_min, w_max, D2_env_interp,
                               hs, tp, omega_n, gamma=3.3):
    """Envelope-weighted optimal placement."""
    w_fine = np.linspace(w_min, w_max, 10000)
    S = jonswap(w_fine, hs, tp, gamma)
    D2 = np.maximum(D2_env_interp(w_fine), 0)

    S_shifted = jonswap(w_fine + omega_n, hs, tp, gamma)
    D2_shifted = np.maximum(D2_env_interp(np.clip(w_fine + omega_n, w_min, w_max)), 0)

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

    omega_n = 0.06

    # Parse drift at specific headings
    test_headings = {
        "Head seas (180°)": 180.0,
        "30° off bow (150°)": 150.0,
        "60° off bow (120°)": 120.0,
        "Beam seas (90°)": 90.0,
        "Quartering (210°)": 210.0,
    }

    heading_data = {}
    for label, angle in test_headings.items():
        freqs, surge, sway, yaw = parse_drift_all_dofs_at_heading(pdstrip_path, angle)
        heading_data[label] = {
            "freqs": freqs, "surge": surge, "sway": sway, "yaw": yaw,
            "angle": angle,
        }

    # ===================================================================
    # 1. Print D(ω) at key frequencies for each heading and DOF
    # ===================================================================
    print("=" * 80)
    print("DRIFT COEFFICIENTS AT SPECIFIC HEADINGS")
    print("=" * 80)

    for label, data in heading_data.items():
        print(f"\n  {label}:")
        print(f"  {'ω':>8s} {'T':>6s} {'Surge':>12s} {'Sway':>12s} {'Yaw':>12s}")
        for i in range(0, len(data["freqs"]), 5):  # every 5th freq
            w = data["freqs"][i]
            T = 2 * PI / w
            print(f"  {w:8.3f} {T:6.1f} {data['surge'][i]:12.0f} "
                  f"{data['sway'][i]:12.0f} {data['yaw'][i]:12.0f}")

    # ===================================================================
    # 2. Build envelope D² (max over all headings and DOFs)
    # ===================================================================
    # Collect all D² values
    all_D2 = []
    ref_freqs = heading_data["Head seas (180°)"]["freqs"]
    for label, data in heading_data.items():
        all_D2.append(data["surge"]**2)
        all_D2.append(data["sway"]**2)
        all_D2.append(data["yaw"]**2)

    # Also add remaining headings
    for extra_angle in [0, 30, 60, 240, 260]:
        f, su, sw, ya = parse_drift_all_dofs_at_heading(pdstrip_path, extra_angle)
        all_D2.append(su**2)
        all_D2.append(sw**2)
        all_D2.append(ya**2)

    D2_envelope = np.max(np.array(all_D2), axis=0)
    D2_env_interp = interp1d(
        ref_freqs[::-1], D2_envelope[::-1],
        kind="cubic", fill_value="extrapolate"
    )

    # ===================================================================
    # 3. Performance comparison per heading per DOF
    # ===================================================================
    print("\n" + "=" * 80)
    print(f"DRIFT FORCE ACCURACY BY HEADING AND DOF")
    print(f"N=35, ω_n={omega_n}, ζ={ZETA}")
    print("=" * 80)

    for bf in [5, 7]:
        ss = BEAUFORT[bf]
        print(f"\n  BF{bf} (Hs={ss['hs']}m, Tp={ss['tp']}s):")
        print(f"  {'Heading + DOF':<35s} {'σ_ref':>10s} {'Geo err':>10s} {'Env err':>10s}")

        f_geo = geometric_frequencies(35)
        f_env = envelope_weighted_optimal(35, W_MIN, W_MAX, D2_env_interp,
                                           ss["hs"], ss["tp"], omega_n)

        for label, data in heading_data.items():
            freqs_raw = data["freqs"]

            for dof_name, dof_data in [("surge", data["surge"]),
                                         ("sway", data["sway"]),
                                         ("yaw", data["yaw"])]:
                # Interpolator for this heading+DOF
                d_interp = interp1d(
                    freqs_raw[::-1], dof_data[::-1],
                    kind="cubic", fill_value="extrapolate"
                )

                # Reference
                freqs_ref = geometric_frequencies(2000)
                sig_ref, _ = compute_filtered_sigma_generic(
                    freqs_ref, d_interp(freqs_ref),
                    ss["hs"], ss["tp"], omega_n, ZETA
                )

                if sig_ref < 10:  # skip negligible
                    continue

                # Geometric
                sig_geo, _ = compute_filtered_sigma_generic(
                    f_geo, d_interp(f_geo),
                    ss["hs"], ss["tp"], omega_n, ZETA
                )
                err_geo = (sig_geo / sig_ref - 1.0) * 100.0

                # Envelope-weighted
                sig_env, _ = compute_filtered_sigma_generic(
                    f_env, d_interp(f_env),
                    ss["hs"], ss["tp"], omega_n, ZETA
                )
                err_env = (sig_env / sig_ref - 1.0) * 100.0

                tag = f"{label} — {dof_name}"
                print(f"  {tag:<35s} {sig_ref:10.0f} {err_geo:+10.1f}% {err_env:+10.1f}%")

    # ===================================================================
    # 4. Plot D(ω) for 30° off bow (150°) — all DOFs
    # ===================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    data_150 = heading_data["30° off bow (150°)"]
    data_180 = heading_data["Head seas (180°)"]

    for ax_idx, (dof, dof_label) in enumerate([("surge", "Surge"), ("sway", "Sway"), ("yaw", "Yaw")]):
        ax = axes[ax_idx]
        ax.plot(data_180["freqs"], data_180[dof], "b-", lw=1.5, label="Head seas (180°)")
        ax.plot(data_150["freqs"], data_150[dof], "r-", lw=1.5, label="30° off bow (150°)")
        ax.plot(heading_data["Beam seas (90°)"]["freqs"],
                heading_data["Beam seas (90°)"][dof], "g-", lw=1.5, label="Beam seas (90°)")
        ax.axhline(0, color="k", lw=0.3)
        ax.set_xlabel("ω [rad/s]")
        ax.set_ylabel(f"D_{dof_label} [N/m² or Nm/m²]")
        ax.set_title(f"{dof_label} drift coefficient")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Drift coefficients at key headings", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "20_drift_by_heading.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: 20_drift_by_heading.png")


if __name__ == "__main__":
    main()
