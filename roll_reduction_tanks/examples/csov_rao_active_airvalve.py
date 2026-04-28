"""RAO comparison for the active (frequency-tracking) air-valve U-tube tank.

Demonstrates the operating principle of a controlled-crossover air-valve
tank: by varying the valve opening, the tank's effective natural
frequency can be modulated continuously between two design extremes
``omega_closed > omega_open``. A controller that tracks the dominant
roll period can therefore *follow* the wave as its period changes,
keeping the absorber notch lined up with the wave excitation across a
much wider band than any single passive tank.

Design used here (CSOV beam):

  * ``utube_duct_height = 0.39 m`` -> ``T_open = 14 s``
    (geometry sets the lowest natural period of the tank).
  * ``chamber_volume_each = 48.9 m^3`` -> ``T_closed = 8 s``
    (gas spring sized to give the desired upper-frequency limit).

The valve controller (:class:`FrequencyTrackingController`) is set to
``T_closed = 8 s, T_open = 14 s`` so the tank notch sweeps across the
full operational wave-period band as the wave changes.

Curves on the resulting plot:

  * bare vessel (reference);
  * valve fully open  -> tank stuck at T_open = 14 s;
  * valve fully closed -> tank stuck at T_closed = 8 s;
  * frequency-tracking controller (the active design).

Expected: the frequency-tracking curve is the *lower envelope* of the
two constant-opening curves (and equals or beats both at every wave
period in the band), giving a wide-band absorber.

Output:
  examples/output/csov_rao_active_airvalve.png
"""
from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

from roll_reduction_tanks.analysis import compute_rao
from roll_reduction_tanks.controllers.constant import (
    FullyClosedValve, FullyOpenValve,
)
from roll_reduction_tanks.controllers.frequency_tracking import (
    FrequencyTrackingController,
)
from roll_reduction_tanks.coupling import CoupledSystem
from roll_reduction_tanks.pdstrip_io import load_csov
from roll_reduction_tanks.tanks.utube_air import (
    AirValveUtubeConfig, AirValveUtubeTank,
)
from roll_reduction_tanks.vessel import RollVessel, RollVesselConfig

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "csov"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

GM = 3.0

# --- Active air-valve design ------------------------------------------------
#
# Geometry tuned so:
#   T_open  = 14 s   (utube_duct_height = 0.3896 m)
#   T_closed = 8 s   (chamber_volume_each = 48.86 m^3)
# Gives a tuning span of 14/8 = 1.75x in period (3.06x in stiffness).

UTUBE_DUCT_HEIGHT = 0.3896
CHAMBER_VOLUME    = 48.86
T_OPEN  = 14.0
T_CLOSED = 8.0


def _vessel(pd):
    return RollVessel(RollVesselConfig(
        I44=pd.I44, a44=pd.a44_assumed, b44_lin=pd.b44_assumed,
        GM=GM, displacement=pd.displacement, rho=pd.rho, g=pd.g,
    ))


def _make_tank(controller):
    cfg = AirValveUtubeConfig(
        duct_below_waterline=6.5,
        undisturbed_fluid_height=2.5,
        utube_duct_height=UTUBE_DUCT_HEIGHT,
        resevoir_duct_width=2.0,
        utube_duct_width=16.0,
        tank_thickness=5.0,
        tank_to_xcog=0.0,
        tank_wall_friction_coef=0.05,
        tank_height=5.0,
        chamber_volume_each=CHAMBER_VOLUME,
        valve_area_max=0.5,
        valve_discharge_coef=0.6,
    )
    return AirValveUtubeTank(cfg, controller=controller)


def _builder(pd, controller_factory):
    """system_builder closure for compute_rao."""
    def build(M):
        v = _vessel(pd)
        if controller_factory is None:
            tanks = []
        else:
            tanks = [_make_tank(controller_factory())]
        return CoupledSystem(v, tanks=tanks, M_wave_func=M)
    return build


def main():
    pd = load_csov(DATA / "csov_pdstrip.dat", DATA / "csov_pdstrip.inp")
    c44 = pd.rho * pd.g * pd.displacement * GM
    I_tot = pd.I44 + pd.a44_assumed
    omega_n = float(np.sqrt(c44 / I_tot))
    T_n = 2 * np.pi / omega_n

    # Sanity-print actual tank natural periods.
    t_open = _make_tank(FullyOpenValve())
    print(f"Active tank design: T_open = "
          f"{2*np.pi/t_open.open_valve_natural_frequency:.2f} s, "
          f"T_closed = {2*np.pi/t_open.closed_valve_natural_frequency:.2f} s")
    print(f"Vessel: T_n = {T_n:.2f} s")

    T_grid = np.concatenate([
        np.linspace(6.0, 9.0, 7, endpoint=False),
        np.linspace(9.0, 14.0, 21, endpoint=False),
        np.linspace(14.0, 18.0, 9),
    ])
    omegas = 2 * np.pi / T_grid

    cases = [
        ("bare vessel",                   None,                                  "k",  "-"),
        ("air-valve open  (T = 14 s)",    FullyOpenValve,                        "C0", "--"),
        ("air-valve closed (T = 8 s)",    FullyClosedValve,                      "C1", "--"),
        ("freq-tracking [8 s, 14 s]",
            lambda: FrequencyTrackingController(
                T_closed=T_CLOSED, T_open=T_OPEN, smoothing_tau=5.0,
            ),
            "C3", "-"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Suppress one-shot fill-ratio warnings (irrelevant here; this is U-tube).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for label, ctrl, color, ls in cases:
            print(f"Sweeping {label} ({len(omegas)} freqs)...")
            pts = compute_rao(_builder(pd, ctrl), pd, omegas,
                              wave_amplitude=1.0)
            rao = np.array([p.rao_deg_per_m for p in pts])
            ax.plot(T_grid, rao, ls, color=color, marker="o",
                    ms=3.5, lw=1.6, label=label)

    # Annotate band of operation.
    ax.axvspan(T_CLOSED, T_OPEN, color="C3", alpha=0.06,
               label=f"controller span [{T_CLOSED:.0f}, {T_OPEN:.0f}] s")
    ax.axvline(T_n, color="k", linestyle=":", alpha=0.4,
               label=f"bare T_n = {T_n:.2f} s")
    ax.set_xlabel("wave period T [s]")
    ax.set_ylabel("roll RAO [deg / m]")
    ax.set_title(
        "CSOV roll RAO, beam seas, GM = 3.0 m, zeta_a = 1 m"
        " - active air-valve U-tube"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = OUT / "csov_rao_active_airvalve.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
