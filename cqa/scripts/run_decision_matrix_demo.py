"""Forecast-case WCFDI decision matrix demo (analysis.md §12.15).

Generates a synthetic 24 h CSOV forecast (a passing storm: V_w ramps
from ~7 m/s up to ~16 m/s and back down, direction slowly veers from
NW to N), then evaluates the decision matrix at every (slot, heading)
and renders three side-by-side heatmaps:

    intact traffic | WCFDI traffic | overall traffic

Each row is a vessel heading, each column is a forecast time slot.
Green / amber / red cells use the IMCA M254 Fig. 8 colour convention.

The synthetic forecast follows Pierson-Moskowitz fully-developed
wind-wave law (matched to the operability polar's sea-state law) so
the matrix at the worst cell is directly comparable to the polar's
boundary V_w at that direction.

Run:
    python scripts/run_decision_matrix_demo.py
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
from matplotlib.colors import ListedColormap, BoundaryNorm

from cqa import (
    csov_default_config, GangwayJointState,
    ForecastSlot, wcfdi_decision_matrix,
)
from cqa.sea_state_relations import pm_hs_from_vw, pm_tp_from_vw


def synthetic_storm_forecast(n_slots: int = 24):
    """Return a list of ForecastSlot for a synthetic 24 h passing storm."""
    t_h = np.arange(n_slots)
    # V_w: triangular ramp 7 -> 16 -> 7 m/s, peak at hour 12.
    Vw = 7.0 + 9.0 * (1.0 - np.abs(t_h - 12.0) / 12.0)
    # Direction veers from NW (315 deg = 5.50 rad) to N (0 rad) over
    # the day; "from" direction.
    theta_deg = 315.0 + (360.0 - 315.0) * (t_h / (n_slots - 1))
    theta_deg = np.mod(theta_deg, 360.0)
    theta_rad = np.radians(theta_deg)
    Vc = 0.5  # constant moderate current

    slots = []
    for k in range(n_slots):
        Hs = float(pm_hs_from_vw(Vw[k]))
        Tp = float(pm_tp_from_vw(Vw[k]))
        slots.append(ForecastSlot(
            label=f"{k:02d}:00",
            Vw=float(Vw[k]),
            Hs=Hs, Tp=Tp,
            Vc=Vc,
            theta_env_compass=float(theta_rad[k]),
        ))
    return slots


def traffic_to_int(grid):
    """Map (n_s, n_h) of strings to (n_s, n_h) ints (green=0, amber=1, red=2)."""
    m = {"green": 0, "amber": 1, "red": 2}
    out = np.empty(grid.shape, dtype=int)
    for s in range(grid.shape[0]):
        for h in range(grid.shape[1]):
            out[s, h] = m[grid[s, h]]
    return out


def main() -> None:
    cfg = csov_default_config()
    joint = GangwayJointState(h=15.0, alpha_g=0.0, beta_g=0.0, L=25.0)

    slots = synthetic_storm_forecast(n_slots=24)
    headings_deg = np.arange(0.0, 360.0, 30.0)  # 12 headings, every 30 deg
    headings = np.radians(headings_deg)

    print(
        f"Building decision matrix: {len(slots)} slots x "
        f"{len(headings)} headings = {len(slots) * len(headings)} cells"
    )
    mx = wcfdi_decision_matrix(
        cfg, joint, slots, headings,
        progress_cb=lambda k, n, label: print(f"  cell {k}/{n}", end="\r"),
    )
    print()

    # ---- Heatmaps ----
    cmap = ListedColormap(["#2a8a2a", "#dba32a", "#cc3333"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    intact_int = traffic_to_int(mx.intact_grid())
    wcfdi_int = traffic_to_int(mx.wcfdi_grid())
    overall_int = traffic_to_int(mx.overall_grid())

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    titles = ["Intact (no failure)", "Post-WCFDI", "Overall (worst-of)"]
    grids = [intact_int, wcfdi_int, overall_int]

    n_s = len(slots)
    n_h = len(headings)
    slot_labels = [s.label for s in slots]

    for ax, title, grid in zip(axes, titles, grids):
        # Transpose so x=time, y=heading
        ax.imshow(
            grid.T, aspect="auto", interpolation="nearest",
            cmap=cmap, norm=norm,
            extent=(-0.5, n_s - 0.5, -0.5, n_h - 0.5),
            origin="lower",
        )
        ax.set_title(title, fontsize=11)
        ax.set_yticks(np.arange(n_h))
        ax.set_yticklabels([f"{int(d):03d}" for d in headings_deg])
        ax.set_ylabel("vessel heading [deg]")
        ax.set_xticks(np.arange(n_s))
        ax.set_xticklabels(slot_labels, rotation=45, ha="right", fontsize=8)
        ax.grid(False)

    axes[-1].set_xlabel("forecast slot (hour of day)")

    # Forecast strip across the top of fig: V_w and theta_env
    fig.suptitle(
        f"CSOV WCFDI forecast decision matrix  "
        f"(synthetic 24 h storm; alpha=2/3; bistability gate=1.5)",
        fontsize=12,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(__file__).resolve().parent / "csov_wcfdi_decision_matrix.png"
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")

    # Also a small text summary
    print()
    print("Forecast summary:")
    for s in slots:
        print(
            f"  {s.label}  V_w={s.Vw:5.2f} m/s  Hs={s.Hs:4.2f} m  "
            f"Tp={s.Tp:4.2f} s  theta_env={np.degrees(s.theta_env_compass):5.1f} deg"
        )

    n_red_overall = int((overall_int == 2).sum())
    n_amber_overall = int((overall_int == 1).sum())
    n_green_overall = int((overall_int == 0).sum())
    print()
    print(
        f"Overall traffic-light distribution: "
        f"{n_green_overall} green, {n_amber_overall} amber, "
        f"{n_red_overall} red (out of {n_s * n_h} cells)"
    )

    # ---- Per-slot polars at chosen transition hours ----
    print()
    print("Rendering per-slot polars at hours 04, 09, 12, 15 ...")
    for hour in (4, 9, 12, 15):
        render_slot_polar(cfg, joint, slots[hour], hour)


def render_slot_polar(cfg, joint, slot, hour: int) -> None:
    """Two-panel polar (intact P90 footprint | post-WCFDI peak) for one slot.

    The two panels are deliberately rendered side-by-side rather than
    overlaid: the underlying metrics measure conceptually different
    quantities (the intact P90 is a quantile of the running-max of a
    stationary process over T_op; the WCFDI peak is the maximum of a
    single transient mean-plus-envelope after a one-shot failure) and
    overlaying them on the same ring would suggest they are directly
    comparable. The shared radial scale and shared IMCA reference
    rings are still useful for "is this number bigger than the alarm
    radius" reading per panel.
    """
    headings_deg = np.arange(0.0, 360.0, 10.0)  # 36 headings
    headings = np.radians(headings_deg)

    print(f"  hour {hour:02d}: evaluating 36 headings")
    mx = wcfdi_decision_matrix(
        cfg, joint, [slot], headings,
    )

    intact_r = np.array([mx.cell(0, h).intact_pos_a_p90_m
                         for h in range(headings.size)])
    wcfdi_r = np.array([mx.cell(0, h).wcfdi_pos_peak_m
                        for h in range(headings.size)])
    wcfdi_traffic = np.array([mx.cell(0, h).wcfdi_traffic
                              for h in range(headings.size)])

    pos_warn = mx.pos_warn_radius_m
    pos_alarm = mx.pos_alarm_radius_m

    # Cap WCFDI radii for plotting (they can be huge in red sectors).
    # The classification (red ring) carries the information; the visual
    # only needs to show "well past the alarm circle".
    r_max = max(pos_alarm * 1.6, np.percentile(intact_r, 95) * 1.4)
    wcfdi_r_plot = np.minimum(wcfdi_r, r_max * 1.2)

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(13, 6.5), subplot_kw={"projection": "polar"},
    )

    for ax, r, traffic_per_h, title in [
        (ax_l, intact_r, None,
         "Intact P90 footprint\nP90 of running-max |position| over T_op (no failure)"),
        (ax_r, wcfdi_r_plot, wcfdi_traffic,
         "Post-WCFDI peak excursion\nmax_t |eta_mean(t)| + 0.674 sigma(t) "
         "(single thruster group lost)"),
    ]:
        # Convert "compass-from" to standard polar (vessel-frame relative
        # heading): we plot per *vessel heading* with N up. theta_zero =
        # north-up, clockwise.
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # IMCA reference rings (warn + alarm). Drawn first so they sit
        # behind the data line.
        theta_full = np.linspace(0, 2 * np.pi, 360)
        ax.plot(theta_full, np.full_like(theta_full, pos_warn),
                color="#dba32a", lw=1.5, ls="--",
                label=f"warn = {pos_warn:.1f} m")
        ax.plot(theta_full, np.full_like(theta_full, pos_alarm),
                color="#cc3333", lw=1.5, ls="--",
                label=f"alarm = {pos_alarm:.1f} m")

        # Close the ring by repeating the first sample.
        h_closed = np.append(headings, headings[0])
        r_closed = np.append(r, r[0])
        ax.plot(h_closed, r_closed, color="black", lw=2.0, marker="o",
                markersize=3.5, label="cell value")

        # Colour-code the markers by per-cell traffic if supplied.
        if traffic_per_h is not None:
            colours = []
            for s in traffic_per_h:
                colours.append({"green": "#2a8a2a",
                                "amber": "#dba32a",
                                "red": "#cc3333"}[s])
            colours_closed = colours + [colours[0]]
            ax.scatter(h_closed, r_closed, c=colours_closed, s=42,
                       zorder=5, edgecolor="black", linewidth=0.5)

        ax.set_rmax(r_max)
        ax.set_title(title, fontsize=10, pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.10),
                  fontsize=8)
        ax.grid(True, alpha=0.4)

    fig.suptitle(
        f"CSOV decision-matrix polar -- {slot.label} "
        f"(V_w={slot.Vw:.1f} m/s, Hs={slot.Hs:.2f} m, Tp={slot.Tp:.2f} s, "
        f"theta_env={np.degrees(slot.theta_env_compass):.0f} deg)\n"
        f"radial axis: vessel base displacement [m]; "
        f"polar angle: vessel heading (compass, N up)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = (Path(__file__).resolve().parent
           / f"csov_wcfdi_decision_polar_h{hour:02d}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"    wrote {out}")


if __name__ == "__main__":
    main()
