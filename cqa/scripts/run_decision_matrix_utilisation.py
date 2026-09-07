"""Forecast-case WCFDI decision matrix as a three-panel utilisation heatmap.

Motivated by the MTS DP 2026 paper (paper.org, §3.7): make the added
benefit of the motion-based CQA framework visually explicit by
comparing three side-by-side utilisation heatmaps for the same synthetic
24 h storm and the same twelve headings:

  1. Classical CQA utilisation:
       |tau_env| / tau_residual_cap, max across DOFs.
     Follows the reference DP system caution/warning convention:
       green   : util < 0.80  (comfortable)
       yellow  : 0.80 <= util < 0.90  (caution)
       red     : util >= 0.90  (warning)

  2. Motion-based intact utilisation:
       intact P90 running-max footprint / alarm radius.
     Thresholds:
       green   : util < warn/alarm ratio  (P90 < warn)
       yellow  : warn/alarm <= util < 1.0 (warn <= P90 < alarm)
       red     : util >= 1.0              (P90 >= alarm)

  3. Motion-based post-WCFDI utilisation:
       post-WCFDI peak envelope / alarm radius,
     same threshold structure as (2).

All three panels use the same green -> yellow -> red gradient colour
map so the operator can directly compare "stress level" across the
three CQA perspectives. The point the figure lands is: the classical
CQA remains comfortably green through the storm, but the motion-based
post-WCFDI panel identifies the actual operational constraint.

Run:
    python scripts/run_decision_matrix_utilisation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

from cqa import (
    csov_default_config, GangwayJointState,
    ForecastSlot, wcfdi_decision_matrix,
)
from cqa.sea_state_relations import pm_hs_from_vw, pm_tp_from_vw


# ---------------------------------------------------------------------------
# Synthetic forecast (kept identical to run_decision_matrix_demo.py so the
# two figures show exactly the same storm).
# ---------------------------------------------------------------------------

def synthetic_storm_forecast(n_slots: int = 24):
    """Return a list of ForecastSlot for a synthetic 24 h passing storm.

    Storm profile chosen deliberately in the operationally interesting
    regime: peak Vw ~ 12 m/s, so the classical CQA stays comfortably
    within the residual (post-WCFDI) capacity throughout, while the
    motion-based post-WCFDI utilisation still identifies where and
    when the P90 excursion crosses the alarm threshold. This is
    where the motion-based framework adds operational content beyond
    the classical CQA.
    """
    t_h = np.arange(n_slots)
    # Storm profile: peak Vw ~ 12 m/s at 12:00, chosen so the classical
    # CQA stays comfortably within the residual capacity through most of
    # the storm (peak ~80% utilisation in beam sectors, at the caution
    # threshold), while the motion-based post-WCFDI utilisation identifies
    # a materially larger red band around the storm peak in the beam
    # headings. This is the operationally interesting regime where the
    # motion-based analysis adds new information beyond the classical
    # CQA verdict.
    Vw = 5.0 + 7.5 * (1.0 - np.abs(t_h - 12.0) / 12.0)  # 5 -> 12.5 -> 5 m/s
    theta_deg = 315.0 + (360.0 - 315.0) * (t_h / (n_slots - 1))
    theta_deg = np.mod(theta_deg, 360.0)
    theta_rad = np.radians(theta_deg)
    Vc = 0.5

    slots = []
    for k in range(n_slots):
        Hs = float(pm_hs_from_vw(Vw[k]))
        Tp = float(pm_tp_from_vw(Vw[k]))
        slots.append(ForecastSlot(
            label=f"{k:02d}:00",
            Vw=float(Vw[k]),
            Hs=Hs, Tp=Tp, Vc=Vc,
            theta_env_compass=float(theta_rad[k]),
        ))
    return slots


# ---------------------------------------------------------------------------
# Classical CQA utilisation
#
# For each (slot, heading) cell, evaluate the mean environmental force
# vector (wind + current + mean drift) in body frame and divide per
# DOF by the residual (post-WCFDI) capability. Utilisation = max ratio
# across DOFs, clipped for plotting to [0, 1.2].
# ---------------------------------------------------------------------------

def classical_ca_utilisation(cfg, slot: ForecastSlot,
                              heading_compass: float,
                              alpha: float = 2.0 / 3.0) -> float:
    """Classical CQA utilisation ratio at one (slot, heading) cell.

    Returns max_DOF(|tau_env_DOF| / tau_residual_cap_DOF).
    """
    from cqa.psd import WindForceModel
    from cqa.vessel import CurrentForceModel

    def _wrap_pi(x):
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    vp = cfg.vessel
    wp = cfg.wind
    cp = cfg.current
    wd = cfg.wave_drift

    theta_rel = _wrap_pi(heading_compass - slot.theta_env_compass)

    wind_model = WindForceModel(wp=wp, loa=vp.loa)
    current_model = CurrentForceModel(
        cp=cp,
        lateral_area_underwater=vp.lpp * vp.draft,
        frontal_area_underwater=vp.beam * vp.draft,
        loa=vp.loa,
    )

    F_wind = wind_model.force(slot.Vw, theta_rel)
    F_curr = current_model.force(slot.Vc, theta_rel)
    F_drift = np.array([
        wd.drift_x_amp * slot.Hs ** 2 * np.cos(theta_rel),
        wd.drift_y_amp * slot.Hs ** 2 * np.sin(theta_rel),
        wd.drift_n_amp * slot.Hs ** 2 * np.sin(2.0 * theta_rel),
    ])
    tau_env = F_wind + F_curr + F_drift

    cap = cfg.thrust_capability.as_array()
    cap_post = alpha * cap
    util_per_dof = np.abs(tau_env) / np.maximum(cap_post, 1e-9)
    return float(np.max(util_per_dof))


# ---------------------------------------------------------------------------
# Gradient colour map: green up to caution, yellow through warning, red beyond.
# ---------------------------------------------------------------------------

def _make_gradient_cmap(caution: float, warning: float, max_val: float):
    """Piecewise-linear green -> yellow -> red centred on (caution, warning).

    Returns (cmap, norm) where norm maps [0, max_val] into the cmap.
    Colours: green (0..caution) -> yellow (caution..warning) -> red (warning..max_val).
    """
    # Anchor colours in RGB.
    green = np.array([0.16, 0.55, 0.16])   # #2a8a2a
    yellow = np.array([0.86, 0.64, 0.16])  # #dba32a
    red = np.array([0.80, 0.20, 0.20])     # #cc3333
    dark_red = np.array([0.55, 0.05, 0.05])

    def _rgb(t):
        # Piecewise interpolation in the space (0..caution..warning..1..max_val)
        t = np.clip(t, 0.0, max_val)
        if t <= caution:
            frac = t / max(caution, 1e-9)
            return green + (yellow - green) * frac
        elif t <= warning:
            frac = (t - caution) / max(warning - caution, 1e-9)
            return yellow + (red - yellow) * frac
        elif t <= 1.0:
            frac = (t - warning) / max(1.0 - warning, 1e-9)
            return red + (dark_red - red) * frac * 0.5
        else:
            # Beyond the alarm: darker red.
            return dark_red

    N = 256
    lut = np.zeros((N, 4))
    for i in range(N):
        t = i / (N - 1) * max_val
        rgb = _rgb(t)
        lut[i, :3] = rgb
        lut[i, 3] = 1.0
    cmap = LinearSegmentedColormap.from_list("green_yellow_red", lut, N=N)
    return cmap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = csov_default_config()
    joint = GangwayJointState(h=15.0, alpha_g=0.0, beta_g=0.0, L=25.0)

    slots = synthetic_storm_forecast(n_slots=24)
    headings_deg = np.arange(0.0, 360.0, 30.0)  # 12 headings
    headings = np.radians(headings_deg)
    n_s = len(slots)
    n_h = len(headings)

    # ----- Motion-based matrix (intact P90 and post-WCFDI peak) -----
    print(f"Building motion-based decision matrix: {n_s} slots x {n_h} headings")
    mx = wcfdi_decision_matrix(
        cfg, joint, slots, headings,
        progress_cb=lambda k, n, label: print(f"  cell {k}/{n}", end="\r"),
    )
    print()

    intact_p90 = np.array([
        [mx.cell(s, h).intact_pos_a_p90_m for h in range(n_h)]
        for s in range(n_s)
    ])
    wcfdi_peak = np.array([
        [mx.cell(s, h).wcfdi_pos_peak_m for h in range(n_h)]
        for s in range(n_s)
    ])

    # ----- Classical CQA utilisation matrix (built independently) -----
    print(f"Building classical CQA utilisation matrix: {n_s * n_h} cells")
    classical_util = np.array([
        [classical_ca_utilisation(cfg, slots[s], headings[h])
         for h in range(n_h)]
        for s in range(n_s)
    ])

    # ----- Utilisation ratios -----
    pos_warn = mx.pos_warn_radius_m
    pos_alarm = mx.pos_alarm_radius_m
    warn_ratio = pos_warn / pos_alarm  # caution threshold for motion axes

    intact_util = intact_p90 / pos_alarm
    # Post-WCFDI peak can be huge in "lost station" regime; clip for
    # colour-map readability. The colour saturates at util = 1.2 (20%
    # past alarm) which is enough to communicate "well past".
    wcfdi_util = np.minimum(wcfdi_peak / pos_alarm, 1.2)

    # Classical CQA colour scale: reference DP system: 80% caution, 90% warning.
    classical_util_plot = np.minimum(classical_util, 1.2)

    # ----- Colour maps -----
    # Motion panels: caution at warn/alarm = 0.5, warning at alarm = 1.0.
    cmap_motion = _make_gradient_cmap(caution=warn_ratio, warning=1.0,
                                       max_val=1.2)
    # Classical CQA panel: reference DP system 80/90.
    cmap_classical = _make_gradient_cmap(caution=0.80, warning=0.90,
                                          max_val=1.2)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    slot_labels = [s.label for s in slots]

    panels = [
        (axes[0], classical_util_plot, cmap_classical,
         "Classical CQA utilisation "
         "(environmental forces / residual capability; "
         "caution 80%, warning 90%)"),
        (axes[1], intact_util, cmap_motion,
         "Motion-based intact utilisation "
         "(P90 footprint / alarm radius)"),
        (axes[2], wcfdi_util, cmap_motion,
         "Motion-based post-WCFDI utilisation "
         "(post-WCFDI peak / alarm radius)"),
    ]

    ims = []
    for ax, grid, cmap, title in panels:
        im = ax.imshow(
            grid.T, aspect="auto", interpolation="nearest",
            cmap=cmap, vmin=0.0, vmax=1.2,
            extent=(-0.5, n_s - 0.5, -0.5, n_h - 0.5),
            origin="lower",
        )
        ims.append(im)
        ax.set_title(title, fontsize=11)
        ax.set_yticks(np.arange(n_h))
        ax.set_yticklabels([f"{int(d):03d}" for d in headings_deg])
        ax.set_ylabel("vessel heading [deg]")
        ax.set_xticks(np.arange(n_s))
        ax.set_xticklabels(slot_labels, rotation=45, ha="right", fontsize=8)
        ax.grid(False)

    axes[-1].set_xlabel("forecast slot (hour of day)")

    # Shared colourbar across the three panels (they use the same vmin/vmax
    # by construction). Reference lines at the operational thresholds.
    cbar = fig.colorbar(ims[0], ax=axes, orientation="vertical",
                         pad=0.02, fraction=0.03, aspect=40)
    cbar.set_label("utilisation ratio", fontsize=10)
    cbar.set_ticks([0.0, 0.5, 0.8, 0.9, 1.0, 1.2])
    cbar.set_ticklabels(["0.0", "0.5", "0.8", "0.9", "1.0", ">1.2"])

    fig.suptitle(
        "CSOV WCFDI forecast decision matrix -- utilisation view "
        "(synthetic 24 h storm)",
        fontsize=12,
    )

    fig.tight_layout(rect=(0, 0, 0.90, 0.96))
    out = Path(__file__).resolve().parent / "csov_wcfdi_decision_matrix_utilisation.png"
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")

    # ---- Text summary -----
    print()
    print("Peak utilisation across the 24-h storm:")
    print(f"  classical CQA:              {classical_util.max():.2f} "
          f"(at slot {np.unravel_index(classical_util.argmax(), classical_util.shape)})")
    print(f"  motion-based intact:       {intact_util.max():.2f} "
          f"(at slot {np.unravel_index(intact_util.argmax(), intact_util.shape)})")
    print(f"  motion-based post-WCFDI:   {wcfdi_util.max():.2f} "
          f"(at slot {np.unravel_index(wcfdi_util.argmax(), wcfdi_util.shape)})")
    print()

    frac_red_classical = float((classical_util >= 0.90).mean())
    frac_red_intact = float((intact_util >= 1.0).mean())
    frac_red_wcfdi = float((wcfdi_util >= 1.0).mean())
    print(f"Fraction of cells in red band:")
    print(f"  classical CQA (>= 90%):     {frac_red_classical:.1%}")
    print(f"  motion intact (>= alarm):  {frac_red_intact:.1%}")
    print(f"  motion WCFDI (>= alarm):   {frac_red_wcfdi:.1%}")


if __name__ == "__main__":
    main()
