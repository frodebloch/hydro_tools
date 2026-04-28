"""Irregular-seas (JONSWAP) seakeeping comparison of all tank types.

Regular-wave RAO sweeps make a *constant-period* excitation, which is
the easiest possible signal for the frequency-tracking controller of an
active air-valve tank: once the period estimator has converged the valve
is held at one setting forever. The honest test of the active design is
therefore an *irregular* seaway, where the instantaneous wave period
wanders and the controller has to chase it in real time.

This example runs a single JONSWAP realisation (Hs = 2 m, Tp = 11 s,
gamma = 3.3, beam seas) through every tank design in
``csov_rao_all_tanks.py`` plus the active air-valve from
``csov_rao_active_airvalve.py`` and reports the significant roll
amplitude ``phi_1/3 = 4 * std(phi)`` for each. A 60 s warmup is
discarded before the statistic is taken so transients don't leak in.

Output:
  examples/output/csov_irregular_seakeeping.png
  + console table of phi_1/3 per design.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

from roll_reduction_tanks.controllers.constant import FullyOpenValve
from roll_reduction_tanks.controllers.frequency_tracking import (
    FrequencyTrackingController,
)
from roll_reduction_tanks.coupling import CoupledSystem
from roll_reduction_tanks.pdstrip_io import load_csov
from roll_reduction_tanks.simulation import run_simulation
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
from roll_reduction_tanks.waves import (
    IrregularWave, roll_moment_from_irregular,
)

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "csov"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

GM = 3.0

# -------------------------------------------------------- seaway

HS = 3.0
TP = 8.5
GAMMA = 3.3
HEADING = 90.0
SEED = 20260428

T_WARMUP = 60.0
T_SIM = 600.0
DT = 0.05  # 20 Hz; well above 1/T_p

# -------------------------------------------------------- tank factories
# Mirror csov_rao_all_tanks.py and csov_rao_active_airvalve.py exactly so
# the regular-RAO and irregular-seas results stay directly comparable.


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
    return OpenUtubeTank(OpenUtubeConfig(
        duct_below_waterline=6.5, undisturbed_fluid_height=2.5,
        utube_duct_height=0.719, resevoir_duct_width=2.0,
        utube_duct_width=16.0, tank_thickness=5.0,
        tank_to_xcog=0.0, tank_wall_friction_coef=0.05,
        tank_height=5.0,
    ))


def _free_surface(c44, I_tot):
    cfg, _ = tune_self_consistent(
        length=22.4, width=8.0,
        z_tank=8.5, z_cog=2.5, damping_ratio=0.10,
        vessel_c44=c44, vessel_inertia_total=I_tot,
        warn_fill_ratio=10.0,
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


# Active air-valve designs.
#
# WIDE span [8, 14] s mirrors csov_rao_active_airvalve.py. Useful as a
# sanity check vs the regular-RAO sweep, but its passive endpoints sit
# far from vessel resonance so the controller can never park the notch
# right on the wave; expected to underperform a matched passive tank.
#
# TIGHT span [10.5, 12.5] s straddles vessel T_n = 11.4 s. Both passive
# endpoints are now near-effective, and the controller can shift the
# notch within the high-energy half of the JONSWAP spectrum as the
# instantaneous wave period wanders.

_AIR_BASE = dict(
    duct_below_waterline=6.5,
    undisturbed_fluid_height=2.5,
    resevoir_duct_width=2.0,
    utube_duct_width=16.0,
    tank_thickness=5.0,
    tank_to_xcog=0.0,
    tank_wall_friction_coef=0.05,
    tank_height=5.0,
    valve_area_max=0.5,
    valve_discharge_coef=0.6,
)


def _air_active_wide():
    cfg = AirValveUtubeConfig(
        utube_duct_height=0.3896,    # T_open  = 14 s
        chamber_volume_each=48.86,   # T_closed = 8 s
        **_AIR_BASE,
    )
    ctrl = FrequencyTrackingController(
        T_closed=8.0, T_open=14.0, smoothing_tau=5.0,
    )
    return AirValveUtubeTank(cfg, controller=ctrl)


def _air_active_tight():
    cfg = AirValveUtubeConfig(
        utube_duct_height=0.496,     # T_open  = 12.5 s
        chamber_volume_each=243.0,   # T_closed = 10.5 s
        **_AIR_BASE,
    )
    ctrl = FrequencyTrackingController(
        T_closed=10.5, T_open=12.5, smoothing_tau=5.0,
    )
    return AirValveUtubeTank(cfg, controller=ctrl)


def _air_active_lowend():
    """Span [7, 11] s — designed for short-Tp North Sea seas.

    Brackets typical CSOV operational Tp (~7-10 s) and reaches up to
    vessel resonance. With Tp = 8.5 s the controller can park within
    the band rather than against an endpoint.
    """
    cfg = AirValveUtubeConfig(
        utube_duct_height=0.648,     # T_open  = 11 s
        chamber_volume_each=67.85,   # T_closed = 7 s
        **_AIR_BASE,
    )
    ctrl = FrequencyTrackingController(
        T_closed=7.0, T_open=11.0, smoothing_tau=5.0,
    )
    return AirValveUtubeTank(cfg, controller=ctrl)


# -------------------------------------------------------- run


def _run_one(pd, M_wave, tank_factory):
    v = _vessel(pd)
    tanks = [] if tank_factory is None else [tank_factory()]
    sys = CoupledSystem(v, tanks=tanks, M_wave_func=M_wave)
    return run_simulation(sys, dt=DT, t_end=T_WARMUP + T_SIM)


def _phi_significant(results) -> float:
    """phi_1/3 = 4 * sigma_phi over the post-warmup window."""
    n_skip = int(T_WARMUP / DT)
    phi_tail = results.phi[n_skip:]
    return float(4.0 * np.std(phi_tail))


def main():
    pd = load_csov(DATA / "csov_pdstrip.dat", DATA / "csov_pdstrip.inp")
    c44 = pd.rho * pd.g * pd.displacement * GM
    I_tot = pd.I44 + pd.a44_assumed
    omega_n = float(np.sqrt(c44 / I_tot))
    T_n = 2 * np.pi / omega_n

    print(f"Vessel T_n = {T_n:.2f} s, sea Tp = {TP:.2f} s, "
          f"Hs = {HS:.1f} m, gamma = {GAMMA:.1f}")

    wave = IrregularWave(
        Hs=HS, Tp=TP, gamma=GAMMA, heading_deg=HEADING,
        omega_min=0.15, omega_max=1.5, n_components=512, seed=SEED,
    )
    M_wave = roll_moment_from_irregular(wave, pd)

    cases = [
        ("bare vessel",                 None,                              "k",  "-"),
        ("open U-tube (T_n match)",     _open_utube,                       "C0", "-"),
        ("open U-tube (Faltinsen +8%)", _open_utube_faltinsen,             "C0", "--"),
        ("free-surface",                lambda: _free_surface(c44, I_tot), "C2", "-"),
        ("TMD (mu=5%, opt)",            lambda: _tmd(I_tot, omega_n),      "C3", "-"),
        ("active air-valve [8,14] s",   _air_active_wide,                  "C4", "--"),
        ("active air-valve [10.5,12.5] s", _air_active_tight,              "C4", "-"),
        ("active air-valve [7,11] s",   _air_active_lowend,                "C4", ":"),
    ]

    fig, (ax_phi, ax_bar) = plt.subplots(
        2, 1, figsize=(11, 7),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    bare_sig = None
    labels, sigs = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for label, factory, color, ls in cases:
            print(f"  running {label}...")
            res = _run_one(pd, M_wave, factory)
            sig = _phi_significant(res)
            sig_deg = np.rad2deg(sig)
            if bare_sig is None:
                bare_sig = sig
            reduction = 100.0 * (1.0 - sig / bare_sig)
            print(f"    phi_1/3 = {sig_deg:6.3f} deg "
                  f"({reduction:+5.1f} % vs bare)")
            labels.append(label)
            sigs.append(sig_deg)
            ax_phi.plot(res.t, np.rad2deg(res.phi), ls, color=color,
                        lw=0.9, alpha=0.85, label=label)

    ax_phi.axvline(T_WARMUP, color="grey", linestyle=":",
                   alpha=0.5, label=f"warmup ends ({T_WARMUP:.0f} s)")
    ax_phi.set_xlabel("time [s]")
    ax_phi.set_ylabel("roll [deg]")
    ax_phi.set_title(
        f"CSOV in JONSWAP seaway, Hs = {HS:.1f} m, Tp = {TP:.1f} s, "
        f"gamma = {GAMMA:.1f}, beam seas (seed = {SEED})"
    )
    ax_phi.grid(True, alpha=0.3)
    ax_phi.legend(loc="upper right", fontsize=8, ncol=2)

    colours = ["k", "C0", "C0", "C2", "C3", "C4", "C4", "C4"]
    bars = ax_bar.bar(range(len(labels)), sigs, color=colours)
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax_bar.set_ylabel("phi_1/3 [deg]")
    ax_bar.grid(True, axis="y", alpha=0.3)
    for b, val, base in zip(bars, sigs, [sigs[0]] * len(sigs)):
        red = 100.0 * (1.0 - val / base)
        ax_bar.text(b.get_x() + b.get_width() / 2, val,
                    f"{val:.2f}\n({red:+.0f}%)" if base != val
                    else f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = OUT / "csov_irregular_seakeeping.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
