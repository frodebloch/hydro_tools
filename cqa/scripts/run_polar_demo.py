"""Demo: build the excursion polar for the CSOV at a representative weather.

Plots:
  1. 95% radial position-excursion polar (intact state).
  2. Standard deviations sigma_n / sigma_e / sigma_psi vs. relative direction.

Run:
    python scripts/run_polar_demo.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Allow running without `pip install -e .`
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cqa import csov_default_config, excursion_polar


def main() -> None:
    cfg = csov_default_config()

    # Representative W2W operating envelope (mid-range weather):
    Vw_mean = 12.0  # m/s, ~Beaufort 6
    Hs = 2.5  # m
    Tp = 8.5  # s
    Vc = 0.5  # m/s

    res = excursion_polar(
        cfg,
        Vw_mean=Vw_mean,
        Hs=Hs,
        Tp=Tp,
        Vc=Vc,
        n_directions=72,
    )

    fig = plt.figure(figsize=(11, 5))

    # Polar plot of 95% radial excursion (semi-major axis of position ellipse).
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    # In a meteorological convention, 0 = head-on, positive = clockwise from bow.
    # We plot theta_rel directly (radians) as is.
    ax_polar.plot(res.theta_rel_rad, res.ellipse_semi_major, "b-", lw=2, label="95% semi-major")
    ax_polar.plot(res.theta_rel_rad, res.ellipse_semi_minor, "g--", lw=1, label="95% semi-minor")
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_title(
        f"CSOV 95% position-excursion polar\nVw={Vw_mean:.0f} m/s, Hs={Hs:.1f} m, Tp={Tp:.1f} s, Vc={Vc:.1f} m/s",
        fontsize=10,
    )
    ax_polar.legend(loc="lower right", fontsize=8)
    ax_polar.grid(True, alpha=0.4)

    # Sigma-vs-direction Cartesian plot.
    ax = fig.add_subplot(1, 2, 2)
    deg = np.degrees(res.theta_rel_rad)
    ax.plot(deg, res.sigma_n, label=r"$\sigma_{\mathrm{surge}}$ [m]")
    ax.plot(deg, res.sigma_e, label=r"$\sigma_{\mathrm{sway}}$ [m]")
    ax.plot(deg, np.degrees(res.sigma_psi), label=r"$\sigma_{\mathrm{heading}}$ [deg]")
    ax.set_xlabel("Relative weather direction [deg]")
    ax.set_ylabel("Std deviation")
    ax.set_title("Position/heading std vs. direction (intact)")
    ax.grid(True, alpha=0.4)
    ax.legend()

    fig.tight_layout()
    out = os.path.join(HERE, "csov_excursion_polar.png")
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")
    print()
    print("Summary at this operating point:")
    i_worst = int(np.argmax(res.ellipse_semi_major))
    print(
        f"  Worst direction: {np.degrees(res.theta_rel_rad[i_worst]):6.1f} deg, "
        f"95% semi-major = {res.ellipse_semi_major[i_worst]:.2f} m, "
        f"semi-minor = {res.ellipse_semi_minor[i_worst]:.2f} m, "
        f"sigma_psi = {np.degrees(res.sigma_psi[i_worst]):.2f} deg"
    )
    i_best = int(np.argmin(res.ellipse_semi_major))
    print(
        f"  Best  direction: {np.degrees(res.theta_rel_rad[i_best]):6.1f} deg, "
        f"95% semi-major = {res.ellipse_semi_major[i_best]:.2f} m, "
        f"semi-minor = {res.ellipse_semi_minor[i_best]:.2f} m"
    )


if __name__ == "__main__":
    main()
