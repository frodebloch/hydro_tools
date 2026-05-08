"""Heading-coupled mean-drift force re-balance — diagnose late-time +0.2 m offset.

Mechanism under test
--------------------
Post-WCF, brucon's vessel heading deviates from setpoint as the closed
loop recovers. The mean wave-drift force is a strong function of the
relative wave heading (rad/s-native QTF integral). As psi(t) shifts,
F_drift on body sway and surge change too — providing a slowly-evolving
disturbance on top of the (essentially-constant pre-WCF) tau_env.

cqa's frozen-tau_env assumption (and the linearised-wave-drift around
intact heading) misses this entirely. If the implied Delta F_drift_y(t)
is O(20-50 kN) over the [0, 90 s] window, that explains both:
  - the prolonged sway peak (drift force keeps pushing through the
    deficit window);
  - the late-time +0.2 m sway offset (new equilibrium drift force is
    different from the intact one).

Method
------
1. Read brucon ensemble heading(t) and SurgeDev / SwayDev over [-30, +120] s
   around WCF.
2. For each seed, compute relative wave heading theta_rel(t) =
   theta_wave_relative_intact + (psi(t) - psi(t=0-)).
3. Use cqa.drift.mean_drift_force_pdstrip with the same Hs, Tp as the
   pwq30 ensemble (Hs=4.196 m, Tp=10.224 s) to compute F_drift body-frame
   at each theta_rel(t).
4. Plot Delta F_drift = F_drift(theta_rel(t)) - F_drift(theta_rel(t=0-))
   per seed + ensemble mean for surge / sway / yaw.
5. Compare ensemble-mean Delta F_drift_sway(t) to the ensemble-mean
   Delta sway truth excursion shape — if shapes match, mechanism
   confirmed; if not, rejected.
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
_REPO_ROOT = str(THIS.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from harness import parse_output  # noqa: E402
from cqa.rao import load_pdstrip_rao  # noqa: E402
from cqa.drift import mean_drift_force_pdstrip  # noqa: E402

# Match calibrated_wcfdi_brucon_validation.py:
ENSEMBLE_DIR = THIS / "work"
TAG = "pwq30"
SEEDS = list(range(1000, 1030))
T_WCF_S = 560.0
T_PRE_WIN = 30.0
T_POST_WIN = 120.0

# Sea state matches pwq30:
HS = 4.19571865443425
TP = 10.22443464601827
# Bow-quartering 30° port: relative wave direction into vessel = +30°
# (cqa convention: 0 = head, +pi/2 = port beam).
THETA_REL_INTACT = np.deg2rad(30.0)

PDSTRIP_PATH = "/home/blofro/src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"


def load_seed_heading(seed: int):
    seed_dir = ENSEMBLE_DIR / f"{TAG}_seed{seed:04d}"
    out_path = seed_dir / f"{TAG}_seed{seed:04d}.out"
    if not out_path.exists():
        return None
    res = parse_output(out_path)
    t = res.columns["t"]
    heading_dev = res.columns["HeadingDev"]  # deg
    surge_dev = res.columns["SurgeDev"]
    sway_dev = res.columns["SwayDev"]
    mask = (t >= T_WCF_S - T_PRE_WIN) & (t <= T_WCF_S + T_POST_WIN)
    t_rel = t[mask] - T_WCF_S
    return (
        t_rel,
        np.deg2rad(heading_dev[mask]),  # rad, deviation from setpoint
        surge_dev[mask],
        sway_dev[mask],
    )


def main():
    seeds_data = []
    for s in SEEDS:
        d = load_seed_heading(s)
        if d is not None:
            seeds_data.append(d)
    print(f"Loaded {len(seeds_data)}/{len(SEEDS)} seeds")
    if not seeds_data:
        raise RuntimeError("no seed data found")

    t = seeds_data[0][0]
    n_t = len(t)
    psi_dev = np.array([d[1][:n_t] for d in seeds_data])     # rad
    surge_dev = np.array([d[2][:n_t] for d in seeds_data])
    sway_dev = np.array([d[3][:n_t] for d in seeds_data])

    # Reference psi at t=0- (last sample before WCF)
    idx_pre = int(np.argmin(np.abs(t + 0.05)))
    psi0 = psi_dev[:, idx_pre:idx_pre + 1]               # (n_seeds, 1)
    psi_change = psi_dev - psi0                           # (n_seeds, n_t)

    # Compute theta_rel(t) = THETA_REL_INTACT + (psi(t) - psi(0))
    # Sign: as vessel turns toward port (psi positive in NED), the wave
    # arrives at a smaller relative angle (rotates from port-quartering
    # toward head-on). cqa convention: theta_rel positive = port beam.
    # Brucon NED psi increases for clockwise turn (compass convention),
    # so a +psi means vessel rotates clockwise i.e. wave (which was
    # coming from compass 210°) approaches relatively from a more-port
    # direction => theta_rel increases. So sign is +.
    theta_rel = THETA_REL_INTACT + psi_change             # (n_seeds, n_t)

    # Pre-compute mean drift force over a fine theta_rel grid, then
    # interp per (seed, time). Drift is expensive (256-omega quadrature
    # per call), so building a 1D LUT is critical.
    print(f"loading pdstrip RAO from {PDSTRIP_PATH}")
    rao = load_pdstrip_rao(PDSTRIP_PATH)

    # Find theta_rel range across all seeds + a little padding
    th_min = float(theta_rel.min()) - np.deg2rad(2.0)
    th_max = float(theta_rel.max()) + np.deg2rad(2.0)
    theta_grid = np.linspace(th_min, th_max, 41)
    print(f"building drift LUT over theta_rel = [{np.rad2deg(th_min):.2f}, "
          f"{np.rad2deg(th_max):.2f}] deg, n={len(theta_grid)}")
    F_lut = np.array([
        mean_drift_force_pdstrip(rao, Hs=HS, Tp=TP, theta_wave_rel=th)
        for th in theta_grid
    ])  # (n_theta, 3)
    # Sanity: print F_drift at intact heading
    F_intact = mean_drift_force_pdstrip(rao, Hs=HS, Tp=TP,
                                        theta_wave_rel=THETA_REL_INTACT)
    print(f"F_drift @ intact theta_rel = 30deg : "
          f"surge={F_intact[0]/1e3:+.2f} kN, sway={F_intact[1]/1e3:+.2f} kN, "
          f"yaw={F_intact[2]/1e3:+.2f} kNm")

    # Interpolate F_drift(theta_rel) per seed/time
    F_per = np.zeros((len(seeds_data), n_t, 3))
    for k in range(len(seeds_data)):
        for c in range(3):
            F_per[k, :, c] = np.interp(theta_rel[k], theta_grid, F_lut[:, c])

    # Delta F = F(t) - F(t=0-)
    F0 = F_per[:, idx_pre:idx_pre + 1, :]                 # (n_seeds, 1, 3)
    dF = F_per - F0                                       # (n_seeds, n_t, 3)
    dF_mean = dF.mean(axis=0)                             # (n_t, 3)

    # Brucon ensemble-mean Δsway for context
    sway_dev_demeaned = sway_dev - sway_dev[:, idx_pre:idx_pre + 1]
    surge_dev_demeaned = surge_dev - surge_dev[:, idx_pre:idx_pre + 1]
    psi_change_mean = psi_change.mean(axis=0)
    sway_dev_mean = sway_dev_demeaned.mean(axis=0)
    surge_dev_mean = surge_dev_demeaned.mean(axis=0)

    # --- plot ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # Panel 1: heading deviation
    ax = axes[0]
    for p in psi_change:
        ax.plot(t, np.rad2deg(p), color="grey", alpha=0.25, lw=0.6)
    ax.plot(t, np.rad2deg(psi_change_mean), "C0-", lw=2,
            label=f"ensemble mean (n={len(seeds_data)})")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4, alpha=0.5)
    ax.set_title("Brucon heading deviation Δψ(t) post-WCF")
    ax.set_ylabel("Δψ [deg]")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Panel 2: implied ΔF_drift_sway and ΔF_drift_surge
    ax = axes[1]
    for k in range(len(seeds_data)):
        ax.plot(t, dF[k, :, 1] / 1e3, color="grey", alpha=0.2, lw=0.5)
    ax.plot(t, dF_mean[:, 1] / 1e3, "C2-", lw=2,
            label=f"ΔF_drift_sway ensemble mean (n={len(seeds_data)})")
    ax.plot(t, dF_mean[:, 0] / 1e3, "C3-", lw=2,
            label=f"ΔF_drift_surge ensemble mean")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4, alpha=0.5)
    ax.set_title("Implied Δ mean-drift force from heading deviation")
    ax.set_ylabel("ΔF_drift [kN]")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Panel 3: brucon truth Δsway alongside scaled ΔF_drift_sway, for shape comparison
    ax = axes[2]
    ax.plot(t, sway_dev_mean, "C0-", lw=2,
            label="brucon ensemble-mean Δsway [m]")
    # scale ΔF_drift_sway to compare shapes (peak-norm)
    if np.any(np.abs(dF_mean[:, 1]) > 1.0):
        peak_dFsway = float(dF_mean[np.argmax(np.abs(dF_mean[:, 1])), 1])
        peak_dsway = float(sway_dev_mean[np.argmax(np.abs(sway_dev_mean))])
        scale = peak_dsway / peak_dFsway if peak_dFsway != 0 else 0
        ax.plot(t, dF_mean[:, 1] * scale, "C2--", lw=1.5,
                label=f"ΔF_drift_sway · ({scale*1e3:+.3f} m/kN) [scaled to match peak]")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4, alpha=0.5)
    ax.set_title("Brucon Δsway vs implied heading-coupled ΔF_drift_sway shape")
    ax.set_xlabel("t since WCF [s]")
    ax.set_ylabel("Δsway [m]")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("Heading-coupled mean-drift force re-balance test", y=1.00)
    plt.tight_layout()
    out_png = THIS / "brucon_drift_heading_coupling.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"saved {out_png}")

    # --- summary ---
    print()
    print("--- summary ---")
    psi_peak_deg = np.rad2deg(np.max(np.abs(psi_change_mean)))
    t_psi_peak = t[np.argmax(np.abs(psi_change_mean))]
    psi_late_deg = np.rad2deg(psi_change_mean[-1])
    print(f"  Δψ peak (ensemble mean): {psi_peak_deg:.2f} deg at t={t_psi_peak:.1f} s")
    print(f"  Δψ late-time (t=120s):   {psi_late_deg:.2f} deg")
    print()
    dF_sway_peak = dF_mean[np.argmax(np.abs(dF_mean[:, 1])), 1] / 1e3
    t_dF_sway_peak = t[np.argmax(np.abs(dF_mean[:, 1]))]
    dF_sway_late = dF_mean[-1, 1] / 1e3
    print(f"  ΔF_drift_sway peak (ensemble mean): {dF_sway_peak:+.2f} kN at t={t_dF_sway_peak:.1f} s")
    print(f"  ΔF_drift_sway late-time (t=120s):    {dF_sway_late:+.2f} kN")
    print(f"  vs F_drift_intact_sway = {F_intact[1]/1e3:+.2f} kN  "
          f"(=> {abs(dF_sway_peak/(F_intact[1]/1e3))*100:.0f}% peak relative magnitude)")
    print()
    dF_surge_peak = dF_mean[np.argmax(np.abs(dF_mean[:, 0])), 0] / 1e3
    print(f"  ΔF_drift_surge peak (ensemble mean): {dF_surge_peak:+.2f} kN")
    print(f"  vs F_drift_intact_surge = {F_intact[0]/1e3:+.2f} kN")
    print()
    print("  decision: if |ΔF_drift_sway peak| O(20-50 kN) and shape matches "
          "Δsway envelope, mechanism is confirmed.")


if __name__ == "__main__":
    main()
