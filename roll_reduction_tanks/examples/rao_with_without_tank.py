"""Roll RAO of CSOV with and without an open-top U-tube tank.

For each wave frequency we run a full coupled time-domain simulation to
steady state and pick off the peak roll amplitude. Two RAO curves are
compared: bare vessel and vessel + tank.

Output:
  examples/output/csov_rao_with_without_tank.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from roll_reduction_tanks.analysis import compute_rao
from roll_reduction_tanks.coupling import CoupledSystem
from roll_reduction_tanks.pdstrip_io import load_csov
from roll_reduction_tanks.tanks.utube_open import OpenUtubeConfig, OpenUtubeTank
from roll_reduction_tanks.vessel import RollVessel, RollVesselConfig

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "csov"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


def make_vessel_builder(pd, with_tank: bool, GM=3.0):
    def build(M_wave):
        v = RollVessel(RollVesselConfig(
            I44=pd.I44, a44=pd.a44_assumed, b44_lin=pd.b44_assumed,
            GM=GM, displacement=pd.displacement, rho=pd.rho, g=pd.g,
        ))
        tanks = []
        if with_tank:
            cfg = OpenUtubeConfig(
                duct_below_waterline=6.5,
                undisturbed_fluid_height=2.5,
                utube_duct_height=0.6,
                resevoir_duct_width=2.0,
                utube_duct_width=16.0,
                tank_thickness=5.0,
                tank_to_xcog=0.0,
                tank_wall_friction_coef=0.05,
                tank_height=5.0,
            )
            tanks = [OpenUtubeTank(cfg)]
        return CoupledSystem(v, tanks=tanks, M_wave_func=M_wave)
    return build


def main():
    pd = load_csov(DATA / "csov_pdstrip.dat", DATA / "csov_pdstrip.inp")

    # 25 frequencies bracketing the roll natural frequency
    # (~0.55 rad/s for GM = 3.0 m).
    omegas = np.linspace(0.30, 1.10, 25)

    print("Sweeping bare vessel...")
    pts_bare = compute_rao(make_vessel_builder(pd, with_tank=False),
                           pd, omegas)
    print("Sweeping vessel + tank...")
    pts_tnk = compute_rao(make_vessel_builder(pd, with_tank=True),
                          pd, omegas)

    rao_bare = np.array([p.rao_deg_per_m for p in pts_bare])
    rao_tnk = np.array([p.rao_deg_per_m for p in pts_tnk])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(omegas, rao_bare, "o-", label="bare vessel")
    ax.plot(omegas, rao_tnk,  "s-", label="vessel + open U-tube")
    ax.set_xlabel("wave frequency omega [rad/s]")
    ax.set_ylabel("roll RAO [deg / m]")
    ax.set_title("CSOV roll RAO, beam seas, GM = 3.0 m - open U-tube tank")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUT / "csov_rao_with_without_tank.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
