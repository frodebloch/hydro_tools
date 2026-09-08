"""Time trace of the synthetic 24-hour storm driving the decision matrix.

The decision matrix (run_decision_matrix_utilisation.py) is evaluated
against a synthetic passing storm, but the matrix itself shows only
the resulting utilisation. This plot shows the forecast that drives
it, so the reader can connect the storm peak and the veering
direction to the pattern of cells in the matrix.

Wind, waves and current are collinear in this forecast; the sea state
follows the DNV-ST-0111 wind-wave correlation, so Hs and Tp track the
wind speed.

Run:
    python scripts/run_storm_profile_demo.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run_decision_matrix_utilisation import synthetic_storm_forecast  # noqa: E402


def main() -> None:
    slots = synthetic_storm_forecast()
    t = np.arange(len(slots))
    Vw = np.array([s.Vw for s in slots])
    Hs = np.array([s.Hs for s in slots])
    Tp = np.array([s.Tp for s in slots])
    Vc = np.array([s.Vc for s in slots])
    # Unwrap: the profile veers 315 -> 360 deg, and the modulo in the
    # forecast generator wraps the final slot to 0, which would plot as
    # a spurious 358 deg drop.
    theta = np.degrees(np.unwrap([s.theta_env_compass for s in slots]))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.0, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    # --- speeds / sea state ---
    ax1.plot(t, Vw, color="#1f77b4", lw=2.2, marker="o", ms=3.5,
             label="wind speed $V_w$ [m/s]")
    ax1.plot(t, Vc, color="#2ca02c", lw=1.8, ls="-.",
             label="current $V_c$ [m/s]")
    ax1.set_ylabel("speed [m/s]")
    ax1.grid(True, alpha=0.3)

    # Tp is deliberately not plotted. The Pierson-Moskowitz relation
    # gives Tp = 2*pi*U/(0.877*g) ~= 0.71*U, so Tp is exactly
    # proportional to Vw and its curve simply overlays the wind trace
    # on any similarly scaled axis. Hs is quadratic in Vw and does
    # carry visible shape, so it is kept.
    ax1b = ax1.twinx()
    ax1b.plot(t, Hs, color="#d62728", lw=2.0, ls="--",
              label="$H_s$ [m]")
    ax1b.set_ylabel("$H_s$ [m]")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=3)
    ax1.set_title("Synthetic 24-hour storm driving the decision matrix",
                  fontsize=11)

    # --- direction ---
    ax2.plot(t, theta, color="#444444", lw=2.0, marker="o", ms=3.5)
    ax2.set_ylabel("direction\n[deg, true]")
    ax2.set_xlabel("forecast slot (hour of day)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(t[::2])
    ax2.set_xticklabels([s.label for s in slots][::2], rotation=45,
                        ha="right", fontsize=8)
    ax2.text(0.02, 0.12,
             "wind, waves and current collinear",
             transform=ax2.transAxes, fontsize=8, color="#444444")

    fig.tight_layout()
    out_dir = Path.home() / "Documents" / "MTS_26" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "csov_storm_profile.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  Vw {Vw.min():.1f}-{Vw.max():.1f} m/s, peak at "
          f"{slots[int(np.argmax(Vw))].label}")
    print(f"  Hs {Hs.min():.2f}-{Hs.max():.2f} m, "
          f"Tp {Tp.min():.1f}-{Tp.max():.1f} s")
    print(f"  direction {theta.min():.0f}-{theta.max():.0f} deg, "
          f"Vc constant {Vc[0]:.2f} m/s")


if __name__ == "__main__":
    main()
