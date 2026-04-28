"""CSOV in regular beam seas, with and without a passive open-top U-tube tank.

Demonstrates the loose-coupling pattern:

  - The vessel is a 1-DOF roll model with hydrostatics, inertia and damping
    derived from CSOV pdstrip data.
  - The wave-exciting roll moment is back-calculated from the pdstrip RAO
    via the inverse linear roll EOM (see :mod:`waves`).
  - The tank is an :class:`OpenUtubeTank` whose geometry is tuned so its
    fluid natural period roughly matches the CSOV roll natural period.
  - The vessel and tank are coupled via the explicit-Jacobi
    :class:`CoupledSystem` and stepped together; the vessel only ever sees
    the tank as an "external moment" — no compile-time coupling between
    the two state vectors.

Run::

    python examples/csov_passive_utube.py

A single PNG is written to ``examples/output/csov_passive_utube.png``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from roll_reduction_tanks.coupling import CoupledSystem
from roll_reduction_tanks.pdstrip_io import load_csov
from roll_reduction_tanks.plotting import (
    overlay_roll_histories,
    plot_moments,
    plot_tank_state,
)
from roll_reduction_tanks.simulation import run_simulation
from roll_reduction_tanks.tanks.utube_open import OpenUtubeConfig, OpenUtubeTank
from roll_reduction_tanks.vessel import RollVessel, RollVesselConfig
from roll_reduction_tanks.waves import RegularWave, roll_moment_from_pdstrip

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "csov"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


def build_vessel(pdstrip_data, GM=3.0) -> RollVessel:
    """Build the CSOV roll vessel from the pdstrip dataset.

    Default ``GM = 3.0 m`` represents a typical operational CSOV loading
    condition (the bundled pdstrip run was at ``GM = 1.787 m``; the wave
    moment back-calculation in :mod:`waves` correctly uses the *pdstrip*
    GM internally regardless of what the simulator's vessel uses).
    """
    cfg = RollVesselConfig(
        I44=pdstrip_data.I44,
        a44=pdstrip_data.a44_assumed,
        b44_lin=pdstrip_data.b44_assumed,
        GM=GM,
        displacement=pdstrip_data.displacement,
        rho=pdstrip_data.rho,
        g=pdstrip_data.g,
    )
    return RollVessel(cfg)


def build_tank() -> OpenUtubeTank:
    """Open U-tube tuned to the CSOV roll natural period at GM = 3.0 m.

    Geometry is sized to fit the CSOV beam (B = 22.4 m). With

      utube_duct_width   = 16 m   (horizontal connecting duct)
      resevoir_duct_width = 2 m   (each vertical leg)

    the total physical tank width is 16 + 2*2 = 20 m. brucon's
    ``tank_width`` (= duct + one leg = 18 m) is what enters the
    coefficient algebra. Lateral thickness is 5 m, fluid depth 2.5 m.
    Duct height 0.6 m is chosen so

        omega_tau = sqrt(g / [b_r * (tank_width/(2 h_d) + h_0/b_r)])

    lands near 0.55 rad/s (the CSOV roll natural frequency at GM = 3.0 m).

    Fluid mass: 2 reservoirs of 25 m^3 + duct (16*5*0.6 = 48 m^3) = ~98 t,
    about 0.9 % of vessel displacement (typical for marine U-tubes).
    """
    cfg = OpenUtubeConfig(
        duct_below_waterline=6.5,        # duct at keel level (T = 6.5 m below WL)
        undisturbed_fluid_height=2.5,
        utube_duct_height=0.6,           # tuned: gives T_tau ~= 11.4 s
        resevoir_duct_width=2.0,
        utube_duct_width=16.0,
        tank_thickness=5.0,
        tank_to_xcog=0.0,
        tank_wall_friction_coef=0.05,    # zeta ~ 0.07: practical compromise
                                          # between resonance peak suppression
                                          # and split-resonance amplification
        tank_height=5.0,
    )
    return OpenUtubeTank(cfg)


def main():
    # --- Load pdstrip data ----------------------------------------------------
    pd = load_csov(DATA / "csov_pdstrip.dat", DATA / "csov_pdstrip.inp")

    # --- Wave: beam seas at the vessel roll natural period -----------------
    # The headline use case for a passive tuned U-tube is at hull resonance,
    # where the tank fluid acts as a tuned dynamic absorber. At T_n = 11.4 s
    # we expect ~80% reduction; at substantially off-resonance frequencies
    # (e.g. T_w = 10 s in 8-12 s seas) the static free-surface effect
    # *amplifies* roll, so a single-period tuned tank is not a panacea --
    # see csov_air_valve_compare.py and rao_with_without_tank.py for the
    # broader picture.
    wave = RegularWave(
        omega=2 * np.pi / 11.4,
        amplitude=1.0,             # 1 m wave amplitude
        heading_deg=90.0,
        speed=0.0,
    )
    print(f"Wave omega = {wave.omega:.3f} rad/s, period = {2*np.pi/wave.omega:.2f} s")

    M_wave = roll_moment_from_pdstrip(wave, pd)

    # --- Build two systems: bare vessel, vessel + tank -----------------------
    vessel_bare = build_vessel(pd)
    vessel_tnk = build_vessel(pd)
    tank = build_tank()
    print(
        f"Vessel: T_n = {vessel_bare.config.natural_period:.2f} s, "
        f"omega_n = {vessel_bare.config.natural_frequency:.3f} rad/s"
    )
    print(
        f"Tank:   T_n = {tank.natural_period:.2f} s, "
        f"omega_n = {tank.natural_frequency:.3f} rad/s, "
        f"zeta = {tank.damping_ratio:.4f}"
    )

    sys_bare = CoupledSystem(vessel_bare, tanks=[], M_wave_func=M_wave)
    sys_tnk = CoupledSystem(vessel_tnk, tanks=[tank], M_wave_func=M_wave)

    # --- Integrate ------------------------------------------------------------
    dt = 0.025
    t_end = 300.0
    res_bare = run_simulation(sys_bare, dt=dt, t_end=t_end)
    res_tnk = run_simulation(sys_tnk, dt=dt, t_end=t_end)

    # Steady-state roll amplitudes (last 30 % of the record)
    n_tail = int(0.3 * len(res_bare.t))
    amp_bare = np.max(np.abs(res_bare.phi_deg[-n_tail:]))
    amp_tnk = np.max(np.abs(res_tnk.phi_deg[-n_tail:]))
    print(f"Steady-state roll amplitude: bare = {amp_bare:.2f} deg, "
          f"with tank = {amp_tnk:.2f} deg "
          f"(reduction = {100*(1 - amp_tnk/amp_bare):.1f} %)")

    # --- Plot -----------------------------------------------------------------
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(3, 1, hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(res_bare.t, res_bare.phi_deg, label="bare vessel", color="C0")
    ax1.plot(res_tnk.t, res_tnk.phi_deg, label="vessel + open U-tube", color="C1")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("roll [deg]")
    ax1.set_title("CSOV roll response at resonance, beam seas, T = 11.4 s, zeta_a = 1 m, GM = 3.0 m")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(gs[1])
    plot_moments(res_tnk, ax=ax2)
    ax2.set_title("Roll moments (with tank)")

    ax3 = fig.add_subplot(gs[2])
    plot_tank_state(
        res_tnk, tank_index=0, component=0,
        ax=ax3, label="tau [deg]", scale=np.rad2deg(1.0),
    )
    ax3.axhline(np.rad2deg(tank.tau_max),  ls=":", color="k", alpha=0.5)
    ax3.axhline(-np.rad2deg(tank.tau_max), ls=":", color="k", alpha=0.5,
                label="tau_max")
    ax3.set_ylabel("tank tilt [deg]")
    ax3.set_title("U-tube fluid angle")
    ax3.legend()

    out_path = OUT / "csov_passive_utube.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
