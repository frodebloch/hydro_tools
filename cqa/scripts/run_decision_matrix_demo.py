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
    out = Path(__file__).resolve().parent.parent / "csov_wcfdi_decision_matrix.png"
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


if __name__ == "__main__":
    main()
