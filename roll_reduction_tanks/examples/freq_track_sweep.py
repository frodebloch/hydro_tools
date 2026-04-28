"""Validate FrequencyTrackingController against constant-opening baselines.

Sweeps wave period across [11.0, 14.0] s on the active-design air-valve
tank (T_open = 14 s, T_closed = T_n = 11.4 s) and compares:

  - bare vessel
  - valve fully open (u=1, tank tuned to T_open=14 s)
  - valve fully closed (u=0, tank tuned to T_closed=11.4 s)
  - frequency-tracking controller (should pick best of the two ends)

Expected: at each T_w the controller should match-or-beat the better of
the two constant baselines, modulo settling/smoothing transients.
"""
from __future__ import annotations

from pathlib import Path

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
from roll_reduction_tanks.vessel import RollVessel, RollVesselConfig
from roll_reduction_tanks.waves import RegularWave, roll_moment_from_pdstrip

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "csov"


def build_vessel(pd, GM=3.0) -> RollVessel:
    return RollVessel(RollVesselConfig(
        I44=pd.I44, a44=pd.a44_assumed, b44_lin=pd.b44_assumed,
        GM=GM, displacement=pd.displacement, rho=pd.rho, g=pd.g,
    ))


def build_tank(controller) -> AirValveUtubeTank:
    """Active-design air-valve tank: T_open=14 s, T_closed=T_n=11.4 s."""
    cfg = AirValveUtubeConfig(
        duct_below_waterline=6.5,
        undisturbed_fluid_height=2.5,
        utube_duct_height=0.3896,        # detuned so T_open = 14 s
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


def amplitude(pd, T_w, controller_factory):
    wave = RegularWave(omega=2 * np.pi / T_w, amplitude=1.0,
                       heading_deg=90.0, speed=0.0)
    M_wave = roll_moment_from_pdstrip(wave, pd)
    v = build_vessel(pd)
    if controller_factory is None:
        sys = CoupledSystem(v, tanks=[], M_wave_func=M_wave)
    else:
        ctrl = controller_factory()
        tank = build_tank(ctrl)
        sys = CoupledSystem(v, tanks=[tank], M_wave_func=M_wave)
    res = run_simulation(sys, dt=0.025, t_end=400.0)
    n_tail = int(0.3 * len(res.t))
    amp = float(np.max(np.abs(res.phi_deg[-n_tail:])))
    # If freq-track, also report the latest period estimate and opening.
    extra = {}
    if controller_factory is not None:
        ctrl_used = sys.tanks[0].controller
        if isinstance(ctrl_used, FrequencyTrackingController):
            extra["T_meas"] = ctrl_used.estimated_period
            # Sample the smoothed opening at the final time.
            phi_dot = res.phi_dot[-1]
            extra["u_final"] = ctrl_used.opening({"phi_dot": phi_dot}, res.t[-1])
    return amp, extra


def main():
    pd = load_csov(DATA / "csov_pdstrip.dat", DATA / "csov_pdstrip.inp")

    T_ws = [11.0, 11.4, 12.0, 12.7, 13.5, 14.0]

    rows = []
    for T_w in T_ws:
        amp_bare, _ = amplitude(pd, T_w, None)
        amp_open, _ = amplitude(pd, T_w, FullyOpenValve)
        amp_closed, _ = amplitude(pd, T_w, FullyClosedValve)
        amp_track, extra = amplitude(
            pd, T_w,
            lambda: FrequencyTrackingController(
                T_closed=11.4, T_open=14.0, smoothing_tau=5.0,
            ),
        )
        rows.append((T_w, amp_bare, amp_open, amp_closed, amp_track, extra))

    print(f"{'T_w':>5s} {'bare':>7s} {'open':>7s} {'closed':>7s} "
          f"{'track':>7s} {'best_const':>10s} {'T_meas':>8s} {'u_fin':>6s}")
    for T_w, ab, ao, ac, at, ex in rows:
        best_const = min(ao, ac)
        T_meas = ex.get("T_meas")
        u_fin = ex.get("u_final")
        T_meas_s = f"{T_meas:.2f}" if T_meas is not None else "  -  "
        u_fin_s = f"{u_fin:.2f}" if u_fin is not None else "  -  "
        marker = "*" if at <= best_const + 0.05 else " "
        print(f"{T_w:5.1f} {ab:7.2f} {ao:7.2f} {ac:7.2f} {at:7.2f} "
              f"{best_const:10.2f} {T_meas_s:>8s} {u_fin_s:>6s} {marker}")
    print("\n* = controller within 0.05 deg of best constant opening")


if __name__ == "__main__":
    main()
