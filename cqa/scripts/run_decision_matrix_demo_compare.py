"""Side-by-side decision matrix: legacy 15-state vs NEW 27-state observer.

Runs the same synthetic 24h CSOV storm forecast as
``run_decision_matrix_demo.py``, evaluates the decision matrix twice
(once with use_obs_transient=False, once with True), and renders a
2-row x 3-col heatmap so the operability swing of the new pipeline
is immediately visible:

    row 0: legacy 15-state pipeline   (intact | WCFDI | overall)
    row 1: NEW   27-state pipeline    (intact | WCFDI | overall)

The intact column should be identical row-to-row (intact axis is
unaffected by use_obs_transient). The WCFDI column shows the swing
from "4x low" (legacy) to "1.6x" (new) at the brucon-pwq30 calibration
point; in the demo storm where Hs grows to ~6 m the legacy pipeline
underpredicts and the new one shifts more cells to amber/red.

Run:
    PYTHONPATH=. .venv/bin/python scripts/run_decision_matrix_demo_compare.py
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
    t_h = np.arange(n_slots)
    Vw = 7.0 + 9.0 * (1.0 - np.abs(t_h - 12.0) / 12.0)
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
            Hs=Hs, Tp=Tp,
            Vc=Vc,
            theta_env_compass=float(theta_rad[k]),
        ))
    return slots


def traffic_to_int(grid):
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
    headings_deg = np.arange(0.0, 360.0, 30.0)
    headings = np.radians(headings_deg)
    n_total = len(slots) * len(headings)

    print(f"Building 15-state matrix: {len(slots)}x{len(headings)} = {n_total} cells")
    mx_15 = wcfdi_decision_matrix(
        cfg, joint, slots, headings,
        use_obs_transient=False,
        progress_cb=lambda k, n, _: print(f"  cell {k}/{n}", end="\r"),
    )
    print()

    print(f"Building 27-state matrix: {n_total} cells")
    mx_27 = wcfdi_decision_matrix(
        cfg, joint, slots, headings,
        use_obs_transient=True,
        progress_cb=lambda k, n, _: print(f"  cell {k}/{n}", end="\r"),
    )
    print()

    cmap = ListedColormap(["#2a8a2a", "#dba32a", "#cc3333"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    grids = {
        ("15-state legacy", "intact"):  traffic_to_int(mx_15.intact_grid()),
        ("15-state legacy", "WCFDI"):   traffic_to_int(mx_15.wcfdi_grid()),
        ("15-state legacy", "overall"): traffic_to_int(mx_15.overall_grid()),
        ("27-state NEW",    "intact"):  traffic_to_int(mx_27.intact_grid()),
        ("27-state NEW",    "WCFDI"):   traffic_to_int(mx_27.wcfdi_grid()),
        ("27-state NEW",    "overall"): traffic_to_int(mx_27.overall_grid()),
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)
    n_s = len(slots)
    n_h = len(headings)
    slot_labels = [s.label for s in slots]

    pipeline_order = ["15-state legacy", "27-state NEW"]
    panel_order = ["intact", "WCFDI", "overall"]

    for r, pipe in enumerate(pipeline_order):
        for c, panel in enumerate(panel_order):
            ax = axes[r, c]
            grid = grids[(pipe, panel)]
            ax.imshow(
                grid.T, aspect="auto", interpolation="nearest",
                cmap=cmap, norm=norm,
                extent=(-0.5, n_s - 0.5, -0.5, n_h - 0.5),
                origin="lower",
            )
            n_g = int((grid == 0).sum())
            n_a = int((grid == 1).sum())
            n_r_ = int((grid == 2).sum())
            ax.set_title(f"{pipe}: {panel}\n"
                         f"{n_g}G / {n_a}A / {n_r_}R", fontsize=10)
            if c == 0:
                ax.set_yticks(np.arange(n_h))
                ax.set_yticklabels([f"{int(d):03d}" for d in headings_deg])
                ax.set_ylabel("vessel heading [deg]")
            if r == 1:
                ax.set_xticks(np.arange(n_s))
                ax.set_xticklabels(slot_labels, rotation=45, ha="right",
                                   fontsize=7)
                ax.set_xlabel("forecast slot (hour of day)")
            ax.grid(False)

    fig.suptitle(
        "CSOV WCFDI decision matrix: legacy 15-state vs NEW 27-state\n"
        "synthetic 24h storm; alpha=2/3, gamma_imm=0.5, T_realloc=10s; "
        "intact identical between rows by construction",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(__file__).resolve().parent / "csov_wcfdi_decision_matrix_compare.png"
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")

    # Numeric comparison: WCFDI pos_peak per cell
    print()
    print("=== Per-cell WCFDI pos_peak comparison (15-state vs 27-state) ===")
    for s_i, slot in enumerate(slots):
        for h_i, head in enumerate(headings):
            c15 = mx_15.cell(s_i, h_i)
            c27 = mx_27.cell(s_i, h_i)
            if c15.wcfdi_traffic != c27.wcfdi_traffic:
                print(
                    f"  {slot.label} h={int(np.degrees(head)):03d}: "
                    f"15st={c15.wcfdi_pos_peak_m:.2f}m({c15.wcfdi_traffic}) "
                    f"-> 27st={c27.wcfdi_pos_peak_m:.2f}m({c27.wcfdi_traffic})"
                )

    g15 = traffic_to_int(mx_15.overall_grid())
    g27 = traffic_to_int(mx_27.overall_grid())
    print()
    print(f"Overall traffic distribution:")
    print(f"  15-state: {(g15==0).sum()}G / {(g15==1).sum()}A / {(g15==2).sum()}R")
    print(f"  27-state: {(g27==0).sum()}G / {(g27==1).sum()}A / {(g27==2).sum()}R")


if __name__ == "__main__":
    main()
