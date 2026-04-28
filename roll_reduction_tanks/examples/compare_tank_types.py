"""Compare all tank types side-by-side at the CSOV roll resonance.

For each tank we tune the most obvious "natural-frequency" knob to land
near the vessel roll natural frequency and then run the same regular
beam-seas sim:

  * Open-top U-tube
  * Air-valve U-tube (passive open valve; same as open tube but uses the
    air-valve tank class to verify the limit)
  * Free-surface (rectangular) tank
  * Tuned-mass damper (canonical SDOF baseline; tuned per Den Hartog)

NOTE: The reductions you see depend strongly on each device's effective
"participation mass" (U-tube fluid carried, sloshing equivalent mass, or
TMD point mass). The geometries below are illustrative. The TMD is
included as the canonical first-order representation of any passive
resonant absorber and is sized using Den Hartog optimal tuning for a
modest mass ratio (mu = 5 %), giving a theoretical upper-bound benchmark.

Output:
  examples/output/compare_tank_types.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from roll_reduction_tanks.controllers.constant import FullyOpenValve
from roll_reduction_tanks.coupling import CoupledSystem
from roll_reduction_tanks.pdstrip_io import load_csov
from roll_reduction_tanks.simulation import run_simulation
from roll_reduction_tanks.tanks.free_surface import (
    FreeSurfaceConfig, FreeSurfaceTank, tune_self_consistent,
)
from roll_reduction_tanks.tanks.tuned_mass_damper import (
    TunedMassDamperConfig, TunedMassDamperTank, den_hartog_optimal,
)
from roll_reduction_tanks.tanks.utube_air import (
    AirValveUtubeConfig, AirValveUtubeTank,
)
from roll_reduction_tanks.tanks.utube_open import (
    OpenUtubeConfig, OpenUtubeTank,
)
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


def make_open_utube():
    return OpenUtubeTank(OpenUtubeConfig(
        duct_below_waterline=6.5, undisturbed_fluid_height=2.5,
        utube_duct_height=0.6, resevoir_duct_width=2.0,
        utube_duct_width=16.0, tank_thickness=5.0,
        tank_to_xcog=0.0, tank_wall_friction_coef=0.05,
        tank_height=5.0,
    ))


def make_air_utube():
    cfg = AirValveUtubeConfig(
        duct_below_waterline=6.5, undisturbed_fluid_height=2.5,
        utube_duct_height=0.6, resevoir_duct_width=2.0,
        utube_duct_width=16.0, tank_thickness=5.0,
        tank_to_xcog=0.0, tank_wall_friction_coef=0.05,
        tank_height=5.0,
        # Note: a *very* large valve area causes orifice-formula stiffness;
        # we use a moderate area which still gives sub-1 % deviation from
        # the open-tube limit at the time scales of interest.
        chamber_volume_each=50.0, valve_area_max=0.5,
    )
    return AirValveUtubeTank(cfg, controller=FullyOpenValve())


def make_free_surface(c44: float, I_tot: float):
    # The free-surface tank's static surface-tilt term ``dc44_extra``
    # destabilises the hull, lowering the effective vessel restoring
    # stiffness from ``c44`` to ``c44 - dc44_extra``. Naively tuning the
    # tank to the *bare* vessel period (~11.4 s) would leave it mistuned
    # in operation against the resulting longer effective period.
    #
    # We use ``tune_self_consistent`` to fix-point iterate: with the tank
    # length pinned at the beam (L = 22.4 m, the geometric maximum), the
    # depth ``h`` is the tuning knob (Faltinsen 1990 eq. 3.76 in the
    # shallow limit ``omega^2 ~ pi^2 g h / L^2``). For W = 8 m this gives
    # h ~ 1.25 m, T_eff ~ 12.85 s, GM_loss/GM ~ 0.22 (mid-Faltinsen
    # range). Compare ``examples/free_surface_self_consistent.py`` for
    # a width sweep.
    cfg, _info = tune_self_consistent(
        length=22.4, width=8.0,
        z_tank=8.5, z_cog=2.5,        # h_arm = 6 m (mounted high)
        damping_ratio=0.10,
        vessel_c44=c44, vessel_inertia_total=I_tot,
    )
    return FreeSurfaceTank(cfg)


def make_tmd(I44_total: float, omega_p: float):
    """Tuned-mass-damper baseline with Den Hartog optimal tuning.

    Mass ratio mu = m * h_arm^2 / I44_total = 5 % (a representative
    value: comparable to a 100 t TMD on a 6 m arm with the CSOV's
    ~7e7 kg*m^2 effective roll inertia).
    """
    h_arm = 6.0
    mu = 0.05
    mass = mu * I44_total / h_arm ** 2
    omega_t, zeta_t = den_hartog_optimal(mass, h_arm, I44_total, omega_p)
    return TunedMassDamperTank(TunedMassDamperConfig(
        mass=mass,
        natural_frequency=omega_t,
        z_mount=2.5 + h_arm,
        z_cog=2.5,
        damping_ratio=zeta_t,
    ))


def main():
    pd = load_csov(DATA / "csov_pdstrip.dat", DATA / "csov_pdstrip.inp")
    # Resonant beam seas at vessel T_n.
    omega_p = 2 * np.pi / 11.4
    wave = RegularWave(omega=omega_p, amplitude=1.0,
                       heading_deg=90.0, speed=0.0)
    M_wave = roll_moment_from_pdstrip(wave, pd)
    I44_total = pd.I44 + pd.a44_assumed

    cases = [
        ("bare vessel",       lambda: None),
        ("open U-tube",       make_open_utube),
        ("air-valve (open)",  make_air_utube),
        ("free-surface",      lambda: make_free_surface(
                                  pd.rho * pd.g * pd.displacement * 3.0,
                                  I44_total)),
        ("TMD (mu=5%, opt)",  lambda: make_tmd(I44_total, omega_p)),
    ]

    dt = 0.025
    t_end = 300.0
    n_tail = int(0.3 * t_end / dt)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    print(f"{'tank':<22s} {'T_n_tank [s]':>14s} {'amp [deg]':>12s}")
    amp_bare = None
    for label, factory in cases:
        v = build_vessel(pd)
        tank = factory()
        tanks = [tank] if tank is not None else []
        sys = CoupledSystem(v, tanks=tanks, M_wave_func=M_wave)
        res = run_simulation(sys, dt=dt, t_end=t_end)
        amp = float(np.max(np.abs(res.phi_deg[-n_tail:])))
        if amp_bare is None:
            amp_bare = amp
        T_str = f"{tank.natural_period:.2f}" if (tank is not None
                and hasattr(tank, "natural_period")) else "-"
        # Air-valve tank: report open-valve period.
        if isinstance(tank, AirValveUtubeTank):
            T_str = f"{2*np.pi/tank.open_valve_natural_frequency:.2f}"
        print(f"{label:<22s} {T_str:>14s} {amp:>12.2f}")
        ax.plot(res.t, res.phi_deg, label=f"{label}  ({amp:.2f} deg)")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("roll [deg]")
    ax.set_title("CSOV beam seas, T = 11.4 s (resonance), zeta_a = 1 m, GM = 3.0 m"
                 " - tank-type comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = OUT / "compare_tank_types.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
