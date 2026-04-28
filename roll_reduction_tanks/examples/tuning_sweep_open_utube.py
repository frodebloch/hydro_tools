"""Tuning sweep for an open U-tube tank.

Sweeps the duct height (which controls the tank natural frequency) and
records steady-state roll amplitude in a fixed regular beam-sea wave.
The U-shaped curve identifies the optimum geometric tuning.

Output:
  examples/output/tuning_sweep_open_utube.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from roll_reduction_tanks.coupling import CoupledSystem
from roll_reduction_tanks.pdstrip_io import load_csov
from roll_reduction_tanks.simulation import run_simulation
from roll_reduction_tanks.tanks.utube_open import OpenUtubeConfig, OpenUtubeTank
from roll_reduction_tanks.vessel import RollVessel, RollVesselConfig
from roll_reduction_tanks.waves import RegularWave, roll_moment_from_pdstrip

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "csov"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


def build_vessel(pd, GM=3.0) -> RollVessel:
    return RollVessel(RollVesselConfig(
        I44=pd.I44, a44=pd.a44_assumed, b44_lin=pd.b44_assumed,
        GM=GM, displacement=pd.displacement, rho=pd.rho, g=pd.g,
    ))


def main():
    pd = load_csov(DATA / "csov_pdstrip.dat", DATA / "csov_pdstrip.inp")
    # Resonant beam seas at vessel T_n.
    wave = RegularWave(omega=2 * np.pi / 11.4, amplitude=1.0,
                       heading_deg=90.0, speed=0.0)
    M_wave = roll_moment_from_pdstrip(wave, pd)

    duct_heights = np.linspace(0.30, 1.50, 18)
    dt = 0.025
    t_end = 300.0
    n_tail = int(0.3 * t_end / dt)

    amplitudes = []
    tank_periods = []
    for h_d in duct_heights:
        v = build_vessel(pd)
        tank = OpenUtubeTank(OpenUtubeConfig(
            duct_below_waterline=6.5, undisturbed_fluid_height=2.5,
            utube_duct_height=float(h_d), resevoir_duct_width=2.0,
            utube_duct_width=16.0, tank_thickness=5.0,
            tank_to_xcog=0.0, tank_wall_friction_coef=0.05,
            tank_height=5.0,
        ))
        sys = CoupledSystem(v, tanks=[tank], M_wave_func=M_wave)
        res = run_simulation(sys, dt=dt, t_end=t_end)
        amp = float(np.max(np.abs(res.phi_deg[-n_tail:])))
        amplitudes.append(amp)
        tank_periods.append(tank.natural_period)

    amplitudes = np.array(amplitudes)
    tank_periods = np.array(tank_periods)
    vessel_T = build_vessel(pd).config.natural_period

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(duct_heights, amplitudes, "o-")
    ax1.set_xlabel("duct height h_d [m]")
    ax1.set_ylabel("steady-state roll amplitude [deg]")
    ax1.set_title("Tuning sweep: roll vs. duct height")
    ax1.grid(True, alpha=0.3)

    ax2.plot(tank_periods, amplitudes, "o-")
    ax2.axvline(vessel_T, ls="--", color="k",
                label=f"vessel T_n = {vessel_T:.2f} s")
    ax2.set_xlabel("tank natural period T_tau [s]")
    ax2.set_ylabel("steady-state roll amplitude [deg]")
    ax2.set_title("Tuning sweep: roll vs. tank period")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("CSOV open-U-tube tuning, beam seas T = 11.4 s, zeta_a = 1 m, GM = 3.0 m")
    fig.tight_layout()
    out = OUT / "tuning_sweep_open_utube.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")

    i_min = int(np.argmin(amplitudes))
    print(
        f"Optimum: h_d = {duct_heights[i_min]:.3f} m, "
        f"T_tau = {tank_periods[i_min]:.2f} s, amp = {amplitudes[i_min]:.2f} deg"
    )


if __name__ == "__main__":
    main()
