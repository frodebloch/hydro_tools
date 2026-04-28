"""Roll RAO sweep: bare CSOV vs each tank type.

Drives the same coupled system as ``compare_tank_types.py`` but sweeps
the wave period ``T_w in [6, 18] s`` instead of pinning to vessel
resonance. The result is the steady-state roll amplitude per metre of
wave amplitude vs wave period -- one curve per tank type, plus the
bare-vessel reference.

Why this is revealing:
  * Bare vessel: classic 1-DOF resonance peak at T_n ~ 11.4 s.
  * Open / air-valve U-tube: anti-resonance notch near the tank's
    natural period; the original single peak splits into two side
    peaks (the classical TMD/U-tube two-peak signature).
  * Free-surface (self-consistent): vessel peak is *shifted* (longer
    T_eff ~ 12.85 s) due to dc44_extra, AND the tank itself opens an
    anti-resonance notch at that shifted period.
  * TMD (Den Hartog optimal): the canonical two-peak shape with both
    peaks of equal height, tightest reduction at resonance.

Wave amplitude is held at 1 m (operating point); nonlinearities such
as the U-tube quadratic damping engage near resonance and bend the
curves down compared with a pure linear RAO.

Output:
  examples/output/csov_rao_all_tanks.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from roll_reduction_tanks.analysis import compute_rao
from roll_reduction_tanks.controllers.constant import FullyOpenValve
from roll_reduction_tanks.coupling import CoupledSystem
from roll_reduction_tanks.pdstrip_io import load_csov
from roll_reduction_tanks.tanks.free_surface import (
    FreeSurfaceTank, tune_self_consistent,
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

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "csov"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

GM = 3.0


# -------------------------------------------------------- tank factories
# Same designs as compare_tank_types.py (kept in sync intentionally).


def _vessel(pd):
    return RollVessel(RollVesselConfig(
        I44=pd.I44, a44=pd.a44_assumed, b44_lin=pd.b44_assumed,
        GM=GM, displacement=pd.displacement, rho=pd.rho, g=pd.g,
    ))


def _open_utube():
    return OpenUtubeTank(OpenUtubeConfig(
        duct_below_waterline=6.5, undisturbed_fluid_height=2.5,
        utube_duct_height=0.6, resevoir_duct_width=2.0,
        utube_duct_width=16.0, tank_thickness=5.0,
        tank_to_xcog=0.0, tank_wall_friction_coef=0.05,
        tank_height=5.0,
    ))


def _open_utube_faltinsen():
    """Open U-tube tuned ~8% above vessel roll frequency (T_n_tank ~ 10.53 s).

    Faltinsen 1990 §5 notes that some designers prefer the passive tank
    natural frequency to sit 6-10% above the vessel's, on the grounds
    that real wave spectra (Pierson-Moskowitz / JONSWAP shape) carry
    more energy on the short-period side of vessel resonance than the
    long-period side; the upward shift moves the absorber notch into
    the high-energy half of the spectrum. Here we use the mid-range
    (+8 %), achieved by raising ``utube_duct_height`` from 0.60 -> 0.719 m
    (lowering tank inertia ``a_tau``).
    """
    return OpenUtubeTank(OpenUtubeConfig(
        duct_below_waterline=6.5, undisturbed_fluid_height=2.5,
        utube_duct_height=0.719,    # tuned for +8% shift
        resevoir_duct_width=2.0,
        utube_duct_width=16.0, tank_thickness=5.0,
        tank_to_xcog=0.0, tank_wall_friction_coef=0.05,
        tank_height=5.0,
    ))


def _air_utube():
    cfg = AirValveUtubeConfig(
        duct_below_waterline=6.5, undisturbed_fluid_height=2.5,
        utube_duct_height=0.6, resevoir_duct_width=2.0,
        utube_duct_width=16.0, tank_thickness=5.0,
        tank_to_xcog=0.0, tank_wall_friction_coef=0.05,
        tank_height=5.0,
        chamber_volume_each=50.0, valve_area_max=0.5,
    )
    return AirValveUtubeTank(cfg, controller=FullyOpenValve())


def _air_utube_faltinsen():
    """Air-valve U-tube with the same +8% Faltinsen shift as the open
    variant. Open-valve limit; quasi-passive."""
    cfg = AirValveUtubeConfig(
        duct_below_waterline=6.5, undisturbed_fluid_height=2.5,
        utube_duct_height=0.719,    # +8% shift
        resevoir_duct_width=2.0,
        utube_duct_width=16.0, tank_thickness=5.0,
        tank_to_xcog=0.0, tank_wall_friction_coef=0.05,
        tank_height=5.0,
        chamber_volume_each=50.0, valve_area_max=0.5,
    )
    return AirValveUtubeTank(cfg, controller=FullyOpenValve())


def _free_surface(c44, I_tot):
    cfg, _ = tune_self_consistent(
        length=22.4, width=8.0,
        z_tank=8.5, z_cog=2.5, damping_ratio=0.10,
        vessel_c44=c44, vessel_inertia_total=I_tot,
        warn_fill_ratio=10.0,   # suppress one-shot warning across sweep
    )
    return FreeSurfaceTank(cfg)


def _tmd(I_tot, omega_p):
    h_arm = 6.0
    mu = 0.05
    mass = mu * I_tot / h_arm**2
    omega_t, zeta_t = den_hartog_optimal(mass, h_arm, I_tot, omega_p)
    return TunedMassDamperTank(TunedMassDamperConfig(
        mass=mass, natural_frequency=omega_t,
        z_mount=2.5 + h_arm, z_cog=2.5, damping_ratio=zeta_t,
    ))


def _builder(pd, tank_factory):
    """Return a system_builder closure compatible with compute_rao."""
    def build(M_wave):
        v = _vessel(pd)
        tank = tank_factory()
        tanks = [tank] if tank is not None else []
        return CoupledSystem(v, tanks=tanks, M_wave_func=M_wave)
    return build


# -------------------------------------------------------- main


def main():
    pd = load_csov(DATA / "csov_pdstrip.dat", DATA / "csov_pdstrip.inp")
    c44 = pd.rho * pd.g * pd.displacement * GM
    I_tot = pd.I44 + pd.a44_assumed
    omega_n = float(np.sqrt(c44 / I_tot))
    T_n = 2 * np.pi / omega_n

    # T in [6, 18] s, denser near vessel T_n.
    T_grid = np.concatenate([
        np.linspace(6.0, 9.0, 7, endpoint=False),
        np.linspace(9.0, 14.0, 21, endpoint=False),   # dense near T_n
        np.linspace(14.0, 18.0, 9),
    ])
    omegas = 2 * np.pi / T_grid

    cases = [
        ("bare vessel",                 lambda: None,             "k",  "-"),
        ("open U-tube (T_n match)",     _open_utube,              "C0", "-"),
        ("open U-tube (Faltinsen +8%)", _open_utube_faltinsen,    "C0", "--"),
        ("air-valve (open)",            _air_utube,               "C1", "-"),
        ("air-valve (Faltinsen +8%)",   _air_utube_faltinsen,     "C1", "--"),
        ("free-surface",                lambda: _free_surface(c44, I_tot), "C2", "-"),
        ("TMD (mu=5%, opt)",            lambda: _tmd(I_tot, omega_n),       "C3", "-"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for label, factory, color, ls in cases:
        print(f"Sweeping {label} ({len(omegas)} freqs)...")
        pts = compute_rao(_builder(pd, factory), pd, omegas,
                          wave_amplitude=1.0)
        rao = np.array([p.rao_deg_per_m for p in pts])
        ax.plot(T_grid, rao, ls, color=color, marker="o",
                ms=3.5, lw=1.5, label=label)

    ax.axvline(T_n, color="k", linestyle=":", alpha=0.4,
               label=f"bare T_n = {T_n:.2f} s")
    ax.set_xlabel("wave period T [s]")
    ax.set_ylabel("roll RAO [deg / m]")
    ax.set_title(
        "CSOV roll RAO, beam seas, GM = 3.0 m, zeta_a = 1 m"
        " - bare vs each tank type"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = OUT / "csov_rao_all_tanks.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
