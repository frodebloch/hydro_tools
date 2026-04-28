"""CSOV in regular beam seas, comparing air-valve U-tube modes
on the **active-design** tank (T_open = 14 s, T_closed = T_n = 11.4 s)
against the **passive** open U-tube tuned exactly to T_n:

  1. Bare vessel (no tank)
  2. Passive open U-tube tuned to T_n = 11.4 s (fluid spring only)
  3. Air-valve U-tube, valve fully open  (u=1, tank tuned to T_open=14 s)
  4. Air-valve U-tube, valve fully closed (u=0, tank tuned to T_n via gas spring)
  5. Air-valve U-tube, frequency-tracking controller

Demonstrates the loose-coupling pattern with an active controller that
uses *only* vessel kinematics — no access to tank state.

The passive baseline is included to make an important honest finding
visible: at the passive's design point, **no air-valve setting beats
the passive**. Active retuning via the gas spring restores the tank's
*natural period* but at the cost of a smaller liquid-side momentum
coupling (smaller ``a_tau``), which weakens the cancellation. The
air-valve concept is genuinely useful only when the wave period is far
enough from the passive design point that the passive's response has
fallen off enough for the gas-spring-detuned configuration to win.

Per Hoppe's product literature, this is also why their commercial
air-damped U-tank is marketed as "the most cost effective solution
applicable for vessels with a relatively narrow band of variation in
loading condition" — for serious adaptive tuning Hoppe sells variable
water-duct cross-section instead, which directly modulates the
liquid-side parameters (not modelled here).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from roll_reduction_tanks.controllers.constant import (
    FullyClosedValve,
    FullyOpenValve,
)
from roll_reduction_tanks.controllers.frequency_tracking import (
    FrequencyTrackingController,
)
from roll_reduction_tanks.coupling import CoupledSystem
from roll_reduction_tanks.pdstrip_io import load_csov
from roll_reduction_tanks.simulation import run_simulation
from roll_reduction_tanks.tanks.utube_air import (
    AirValveUtubeConfig,
    AirValveUtubeTank,
)
from roll_reduction_tanks.tanks.utube_open import (
    OpenUtubeConfig,
    OpenUtubeTank,
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


def build_tank(controller) -> AirValveUtubeTank:
    """Active-design air-valve tank.

    The U-tube geometry is detuned (small ``utube_duct_height``) so that
    with the valve fully open the tank period is T_open = 14 s. The
    chamber volume is then sized so that with the valve fully closed
    the trapped-gas spring lifts the period to T_closed = T_n = 11.4 s.
    Controllable band thus covers the wave-period range of interest.
    """
    cfg = AirValveUtubeConfig(
        duct_below_waterline=6.5,
        undisturbed_fluid_height=2.5,
        utube_duct_height=0.3896,        # detuned: T_open = 14 s
        resevoir_duct_width=2.0,
        utube_duct_width=16.0,
        tank_thickness=5.0,
        tank_to_xcog=0.0,
        tank_wall_friction_coef=0.05,
        tank_height=5.0,
        chamber_volume_each=198.3,       # sized so T_closed = 11.4 s
        valve_area_max=0.5,
        valve_discharge_coef=0.6,
    )
    return AirValveUtubeTank(cfg, controller=controller)


def build_passive_utube() -> OpenUtubeTank:
    """Passive open U-tube tuned to T_n = 11.4 s (h_d = 0.6 m)."""
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
    return OpenUtubeTank(cfg)


def main():
    pd = load_csov(DATA / "csov_pdstrip.dat", DATA / "csov_pdstrip.inp")
    # Resonant beam seas (T_w = 11.4 s ~ vessel T_n at GM = 3.0 m).
    wave = RegularWave(omega=2 * np.pi / 11.4, amplitude=1.0,
                       heading_deg=90.0, speed=0.0)
    M_wave = roll_moment_from_pdstrip(wave, pd)

    runs = []
    # 1. Bare
    v = build_vessel(pd)
    sys = CoupledSystem(v, tanks=[], M_wave_func=M_wave)
    runs.append(("bare vessel", sys))
    # 2. Passive baseline tuned to T_n
    v = build_vessel(pd)
    runs.append(("passive open U-tube (T_n = 11.4 s)",
                 CoupledSystem(v, tanks=[build_passive_utube()],
                               M_wave_func=M_wave)))
    # 3. Air-valve, fully open  (tank period = T_open = 14 s)
    v = build_vessel(pd)
    runs.append(("air valve fully open  (u=1, T_tank=14.0 s)",
                 CoupledSystem(v, tanks=[build_tank(FullyOpenValve())],
                               M_wave_func=M_wave)))
    # 4. Air-valve, fully closed  (tank period = T_closed = T_n = 11.4 s)
    v = build_vessel(pd)
    runs.append(("air valve fully closed (u=0, T_tank=11.4 s)",
                 CoupledSystem(v, tanks=[build_tank(FullyClosedValve())],
                               M_wave_func=M_wave)))
    # 5. Frequency-tracking adaptive controller
    v = build_vessel(pd)
    runs.append(("air valve, frequency-tracking ctrl",
                 CoupledSystem(
                     v,
                     tanks=[build_tank(FrequencyTrackingController(
                         T_closed=11.4, T_open=14.0, smoothing_tau=5.0,
                     ))],
                     M_wave_func=M_wave)))

    dt = 0.025
    t_end = 300.0

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_phi, ax_tau = axes

    n_tail = int(0.3 * (t_end / dt))

    print(f"{'mode':<45s} {'amp[deg]':>10s} {'reduction':>11s}")
    amp_bare = None
    for label, sys in runs:
        res = run_simulation(sys, dt=dt, t_end=t_end)
        amp = float(np.max(np.abs(res.phi_deg[-n_tail:])))
        if amp_bare is None:
            amp_bare = amp
            reduc = ""
        else:
            reduc = f"{100*(1-amp/amp_bare):+6.1f} %"
        print(f"{label:<45s} {amp:>10.2f} {reduc:>11s}")

        ax_phi.plot(res.t, res.phi_deg, label=label)
        if res.tank_states:
            tau = res.tank_states[0][:, 0]
            ax_tau.plot(res.t, np.rad2deg(tau), label=label)

    ax_phi.set_ylabel("roll [deg]")
    ax_phi.grid(True, alpha=0.3)
    ax_phi.legend(loc="upper right", fontsize=9)
    ax_phi.set_title("CSOV beam seas, T = 11.4 s (resonance), zeta_a = 1 m, GM = 3.0 m"
                     "  -- passive U-tube vs active-design air valve")

    ax_tau.set_xlabel("time [s]")
    ax_tau.set_ylabel("tank tilt tau [deg]")
    ax_tau.grid(True, alpha=0.3)
    ax_tau.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out = OUT / "csov_air_valve_compare.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
