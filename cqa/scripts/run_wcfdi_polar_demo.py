"""WCFDI operability polar demo: overlay the post-failure peak boundary
on top of the intact-prior P90 operability polar.

Two-panel polar (vessel base | gangway telescope) with:

  * intact P90 operability rings (green / amber / red shaded), AND
  * post-WCFDI peak-excursion no-go boundary (single dashed line per
    panel, set at the IMCA M254 alarm radius).

Default scenario (per session direction): one of three thruster groups
lost (alpha = 2/3 per DOF), gamma_immediate = 0.5, T_realloc = 10 s,
k_sigma = 0.674 (P75 of the post-failure peak distribution).

Rationale for the modest k_sigma: WCFDI itself is a rare event
(annual frequency ~ 1e-3 for a single thruster group). Conditioning
on a high-quantile peak excursion *given* the failure compounds two
rare events and inflates the design margin against a joint event
whose probability is already small. P75 sits between the deterministic
P50 (k = 0) and the intact polar's P90 (k = 1.282) convention.

The script also generates a Tier A diagnostic figure that sweeps V_w
over a fine grid for one direction (the worst-case heading found by
the polar) and plots the post-WCFDI peak metric vs V_w. This shows
the steepness of the peak in V_w near the boundary and validates the
~ V_w^4 scaling used to motivate the single-line plot convention.

Run:
    python scripts/run_wcfdi_polar_demo.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cqa import (
    csov_default_config,
    GangwayJointState,
    operability_polar,
    plot_operability_polar,
    wcfdi_operability_overlay,
    WcfdiScenario,
)
from cqa.operability_polar import _wcfdi_peak_metrics
from cqa.gangway import telescope_sensitivity


def main() -> None:
    cfg = csov_default_config()
    gw = cfg.gangway
    L0 = 0.5 * (gw.telescope_min + gw.telescope_max)
    joint = GangwayJointState(
        h=gw.rotation_centre_height_above_base,
        alpha_g=-np.pi / 2.0,  # port
        beta_g=0.0,
        L=L0,
    )

    print("Sweeping intact operability polar (36 directions)...")
    t0 = time.perf_counter()
    polar = operability_polar(
        cfg, joint,
        n_directions=36,
        Vw_min=2.0,
        Vw_max=30.0,
        Vc_m_s=0.5,
        rao_table=None,           # parametric drift PSD only (fast)
        T_op_s=20.0 * 60.0,
        quantile_p=0.90,
        bisect_tol_m_s=0.25,
    )
    dt_intact = time.perf_counter() - t0
    print(f"  intact polar done in {dt_intact:.1f} s")

    # Post-WCFDI scenario: one of three thruster groups lost
    # (alpha = 2/3 per DOF, gamma_immediate = 0.5, T_realloc = 10 s).
    scenario = WcfdiScenario(
        alpha=(2.0 / 3.0,) * 3,
        gamma_immediate=0.5,
        T_realloc=10.0,
    )
    k_sigma = 0.674  # P75 of the conditional peak

    print("\nSweeping post-WCFDI overlay (36 directions, ~30 ms / call)...")
    t0 = time.perf_counter()
    overlay = wcfdi_operability_overlay(
        cfg, joint,
        scenario=scenario,
        n_directions=36,
        Vw_min=2.0,
        Vw_max=30.0,
        Vc_m_s=0.5,
        k_sigma=k_sigma,
        bisect_tol_m_s=0.25,
        t_end=200.0,
    )
    dt_wcfdi = time.perf_counter() - t0
    print(f"  WCFDI overlay done in {dt_wcfdi:.1f} s")

    # Console summary.
    def _worst_dir_deg(arr: np.ndarray) -> tuple[float, float]:
        i = int(np.argmin(arr))
        return float(np.degrees(overlay.theta_rel_rad[i])), float(arr[i])

    th_palarm, vw_palarm = _worst_dir_deg(polar.pos_alarm_Vw)
    th_palarm_w, vw_palarm_w = _worst_dir_deg(overlay.pos_alarm_Vw)
    th_galarm, vw_galarm = _worst_dir_deg(polar.gw_alarm_Vw)
    th_galarm_w, vw_galarm_w = _worst_dir_deg(overlay.gw_alarm_Vw)

    print()
    print("=== Operability polar summary ===")
    print(f"  Intact:  T_op={polar.T_op_s/60:.0f} min, "
          f"P{polar.quantile_p*100:.0f} envelope, "
          f"Vc={polar.Vc_m_s:.1f} m/s")
    print(f"  WCFDI:   alpha={overlay.alpha}, "
          f"T_realloc={overlay.T_realloc:.0f} s, "
          f"k_sigma={overlay.k_sigma:.3f} (P75 conditional peak), "
          f"t_end={overlay.t_end:.0f} s")
    print()
    print("Vessel base position "
          f"(alarm {polar.pos_alarm_radius_m:.1f} m IMCA M254)")
    print(f"  intact worst-direction alarm V_w: {vw_palarm:5.1f} m/s "
          f"@ {th_palarm:5.0f} deg")
    print(f"  WCFDI  worst-direction alarm V_w: {vw_palarm_w:5.1f} m/s "
          f"@ {th_palarm_w:5.0f} deg")
    print(f"  reduction: {vw_palarm - vw_palarm_w:5.1f} m/s "
          f"({(1.0 - vw_palarm_w / vw_palarm) * 100:4.1f} %)")
    print()
    print("Gangway telescope "
          f"(alarm {polar.gw_alarm_m:.2f} m, 80% of stroke)")
    print(f"  intact worst-direction alarm V_w: {vw_galarm:5.1f} m/s "
          f"@ {th_galarm:5.0f} deg")
    print(f"  WCFDI  worst-direction alarm V_w: {vw_galarm_w:5.1f} m/s "
          f"@ {th_galarm_w:5.0f} deg")
    print(f"  reduction: {vw_galarm - vw_galarm_w:5.1f} m/s "
          f"({(1.0 - vw_galarm_w / vw_galarm) * 100:4.1f} %)")

    if np.any(overlay.pos_alarm_capped_low):
        n = int(np.sum(overlay.pos_alarm_capped_low))
        print(f"\n  NOTE: WCFDI vessel alarm already breached at Vw_min "
              f"in {n} directions (saturated low; treat as 'no-go at any "
              f"wind speed in this heading post-failure').")

    fig = plot_operability_polar(polar, overlay=overlay)
    out = os.path.join(HERE, "csov_wcfdi_operability_polar.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # -----------------------------------------------------------------
    # Tier A diagnostic: peak metric vs V_w at the worst-case direction.
    # Validates the steepness argument used to justify drawing only the
    # alarm dashed line (warn and alarm crossings collapse onto a
    # near-identical V_w because the peak ~ V_w^4 near the boundary).
    # -----------------------------------------------------------------
    theta_diag = np.radians(th_palarm_w)
    print(f"\nTier A diagnostic: peak metric vs V_w "
          f"at theta = {th_palarm_w:.0f} deg (worst-case heading)...")

    Vw_grid = np.linspace(2.0, max(15.0, vw_palarm + 2.0), 40)
    pos_peaks = np.zeros_like(Vw_grid)
    dL_peaks = np.zeros_like(Vw_grid)
    c_L = telescope_sensitivity(joint, cfg.gangway)
    t1 = time.perf_counter()
    for j, Vw in enumerate(Vw_grid):
        try:
            pos_peaks[j], dL_peaks[j] = _wcfdi_peak_metrics(
                cfg, joint, Vw, 0.5, theta_diag,
                scenario=scenario, k_sigma=k_sigma, t_end=200.0,
                sigma_Vc=0.1, tau_Vc=600.0, c_L=c_L,
            )
        except Exception:
            pos_peaks[j] = np.nan
            dL_peaks[j] = np.nan
    print(f"  {len(Vw_grid)} V_w grid points done in "
          f"{time.perf_counter() - t1:.1f} s")

    fig2, ax_pair = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_pos, ax_gw = ax_pair

    # Vessel-base panel.
    ax_pos.plot(Vw_grid, pos_peaks, "k-", lw=1.8,
                label=f"post-WCFDI peak (k_sigma={k_sigma:.3f})")
    ax_pos.axhline(polar.pos_warn_radius_m, color="#cc8800",
                   ls=":", lw=1.5,
                   label=f"warn ({polar.pos_warn_radius_m:.1f} m)")
    ax_pos.axhline(polar.pos_alarm_radius_m, color="#990000",
                   ls=":", lw=1.5,
                   label=f"alarm ({polar.pos_alarm_radius_m:.1f} m)")
    ax_pos.axvline(vw_palarm_w, color="#1a1a1a", ls="--", lw=1.0,
                   label=f"V_w (alarm) = {vw_palarm_w:.1f} m/s")
    ax_pos.set_xlabel("V_w [m/s]")
    ax_pos.set_ylabel("Vessel base peak excursion [m]")
    ax_pos.set_title(f"Vessel base @ theta={th_palarm_w:.0f} deg")
    ax_pos.set_ylim(0.0, max(8.0, polar.pos_alarm_radius_m * 2.0))
    ax_pos.grid(True, alpha=0.3)
    ax_pos.legend(loc="upper left", fontsize=9)

    # Telescope panel.
    ax_gw.plot(Vw_grid, dL_peaks, "k-", lw=1.8,
               label=f"post-WCFDI peak (k_sigma={k_sigma:.3f})")
    ax_gw.axhline(polar.gw_warn_m, color="#cc8800", ls=":", lw=1.5,
                  label=f"warn ({polar.gw_warn_m:.2f} m)")
    ax_gw.axhline(polar.gw_alarm_m, color="#990000", ls=":", lw=1.5,
                  label=f"alarm ({polar.gw_alarm_m:.2f} m)")
    ax_gw.axvline(vw_galarm_w, color="#1a1a1a", ls="--", lw=1.0,
                  label=f"V_w (alarm) = {vw_galarm_w:.1f} m/s")
    ax_gw.set_xlabel("V_w [m/s]")
    ax_gw.set_ylabel("Telescope |dL| peak [m]")
    ax_gw.set_title(f"Telescope @ theta={th_palarm_w:.0f} deg")
    ax_gw.set_ylim(0.0, max(polar.gw_alarm_m * 2.0, 8.0))
    ax_gw.grid(True, alpha=0.3)
    ax_gw.legend(loc="upper left", fontsize=9)

    fig2.suptitle(
        "Tier A: post-WCFDI peak metric vs V_w "
        f"(alpha=({overlay.alpha[0]:.2f},...), T_realloc={overlay.T_realloc:.0f} s)\n"
        "warn/alarm thresholds collapse to nearly the same V_w "
        "because the peak grows like V_w^4 near the boundary",
        fontsize=11,
    )
    fig2.tight_layout(rect=(0, 0, 1, 0.94))
    out2 = os.path.join(HERE, "csov_wcfdi_peak_vs_Vw.png")
    fig2.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {out2}")

    # Quick numerical check of the V_w^4 claim: compare slope of
    # log(peak) vs log(V_w) on the upper half of the operable range.
    half = len(Vw_grid) // 2
    upper = slice(half, len(Vw_grid))
    finite = np.isfinite(pos_peaks[upper]) & (pos_peaks[upper] > 1e-3)
    if np.sum(finite) >= 4:
        log_Vw = np.log(Vw_grid[upper][finite])
        log_pk = np.log(pos_peaks[upper][finite])
        slope = float(np.polyfit(log_Vw, log_pk, 1)[0])
        print(f"\n  Tier A check: log-log slope d(log peak_pos)/d(log V_w) "
              f"= {slope:.2f} on upper half of grid "
              f"(expected ~ 2-4 for the conjectured V_w^4 mechanism)")


if __name__ == "__main__":
    main()
