#!/usr/bin/env python3
"""Diagnostic: compare factory combinator and optimiser at D=4.66 vs D=4.80.

Prints the factory schedule, optimiser operating points, and fuel rates
at both diameters for a range of typical thrust demands, to understand
why the pitch/RPM saving doubled when the diameter changed.
"""

import math
import sys
import numpy as np

from propeller_model import CSeriesPropeller, load_c_series_data
from optimiser import find_min_fuel_operating_point
from models.constants import (
    DATA_PATH_C440, DATA_PATH_C455, DATA_PATH_C470,
    PROP_BAR, PROP_DESIGN_PITCH,
    GEAR_RATIO, SHAFT_EFF, KN_TO_MS,
    HULL_SPEEDS_KN, HULL_WAKE,
    RHO_WATER,
)
from models.combinator import FactoryCombinator
from optimiser import make_man_l27_38


def make_prop(diameter):
    """Create a propeller model at a given diameter."""
    data_40 = load_c_series_data(DATA_PATH_C440)
    data_55 = load_c_series_data(DATA_PATH_C455)
    data_70 = load_c_series_data(DATA_PATH_C470)
    bar_data = {0.40: data_40, 0.55: data_55, 0.70: data_70}
    return CSeriesPropeller(bar_data, design_pitch=PROP_DESIGN_PITCH,
                            diameter=diameter, area_ratio=PROP_BAR,
                            rho=RHO_WATER)


def main():
    engine = make_man_l27_38()
    speed_kn = 10.0
    w = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_WAKE))
    Va = speed_kn * KN_TO_MS * (1.0 - w)

    print(f"Speed: {speed_kn} kn, wake fraction: {w:.3f}, Va: {Va:.3f} m/s")
    print()

    for D in [4.66, 4.80]:
        prop = make_prop(D)
        factory = FactoryCombinator(engine, prop)

        print("=" * 90)
        print(f"  DIAMETER = {D} m")
        print("=" * 90)

        # --- Factory combinator schedule ---
        print(f"\n  Factory combinator schedule ({len(factory._combo_thrust_kn)} points):")
        print(f"  {'Lever%':>7s}  {'T [kN]':>8s}  {'Pitch':>7s}  {'RPM':>7s}  {'EngRPM':>8s}")
        for i in range(len(factory._combo_lever)):
            lv = factory._combo_lever[i]
            t = factory._combo_thrust_kn[i]
            p = factory._combo_pitch[i]
            r = factory._combo_rpm[i]
            er = r * GEAR_RATIO
            print(f"  {lv:7.1f}  {t:8.1f}  {p:7.3f}  {r:7.1f}  {er:8.1f}")

        # --- Compare at typical thrust demands ---
        print(f"\n  Operating point comparison at Va={Va:.3f} m/s:")
        print(f"  {'T[kN]':>7s}  {'Fac P/D':>8s}  {'Fac RPM':>8s}  {'Fac fuel':>9s}  "
              f"{'Fac eta0':>8s}  {'Opt P/D':>8s}  {'Opt RPM':>8s}  {'Opt fuel':>9s}  "
              f"{'Opt eta0':>8s}  {'Save%':>6s}")

        for T_kN in np.arange(80, 200, 10):
            T_N = T_kN * 1000.0

            fac = factory.evaluate(T_N, Va)
            opt = find_min_fuel_operating_point(
                prop, Va, T_N, engine,
                gear_ratio=GEAR_RATIO,
                shaft_efficiency=SHAFT_EFF,
                pitch_step=0.01,
            )

            if fac is None or not opt.found:
                fac_str = "INFEAS" if fac is None else ""
                opt_str = "INFEAS" if not opt.found else ""
                print(f"  {T_kN:7.0f}  {fac_str:>8s}  {' ':8s}  {' ':9s}  "
                      f"{' ':8s}  {opt_str:>8s}")
                continue

            save_pct = 100.0 * (fac["fuel_rate"] - opt.fuel_rate) / fac["fuel_rate"]

            print(f"  {T_kN:7.0f}  {fac['pitch']:8.3f}  {fac['rpm']:8.1f}  "
                  f"{fac['fuel_rate']:9.0f}  {fac['eta0']:8.4f}  "
                  f"{opt.pitch:8.3f}  {opt.rpm:8.1f}  {opt.fuel_rate:9.0f}  "
                  f"{prop.eta0(opt.pitch, opt.rpm / 60.0, Va):8.4f}  "
                  f"{save_pct:6.1f}")

        # --- Propeller efficiency landscape at a single thrust point ---
        T_target = 130.0  # kN, typical calm-water at 10 kn
        T_N = T_target * 1000.0
        print(f"\n  Propeller efficiency at T={T_target} kN (finding RPM for each pitch to match thrust):")
        print(f"  {'P/D':>6s}  {'n[rps]':>8s}  {'RPM':>7s}  {'EngRPM':>8s}  {'eta0':>8s}  {'Q[Nm]':>10s}  {'Pshaft[kW]':>10s}")

        for p_int in range(50, 96, 5):
            pitch = p_int / 100.0
            # Bisect RPM to find n that delivers T_target
            n_lo, n_hi = 0.5, 3.5
            found = False
            for _ in range(60):
                n_mid = (n_lo + n_hi) / 2.0
                T = prop.thrust(pitch, n_mid, Va)
                if abs(T - T_N) < 50:
                    found = True
                    break
                if T < T_N:
                    n_lo = n_mid
                else:
                    n_hi = n_mid
            if not found:
                n_mid = (n_lo + n_hi) / 2.0
                T = prop.thrust(pitch, n_mid, Va)
                if abs(T - T_N) > 2000:
                    continue

            eta = prop.eta0(pitch, n_mid, Va)
            Q = prop.torque(pitch, n_mid, Va)
            P_shaft = 2.0 * math.pi * n_mid * Q / 1000.0
            eng_rpm = n_mid * 60.0 * GEAR_RATIO
            print(f"  {pitch:6.2f}  {n_mid:8.3f}  {n_mid*60:7.1f}  {eng_rpm:8.1f}  "
                  f"{eta:8.4f}  {Q:10.0f}  {P_shaft:10.1f}")

        print()


if __name__ == "__main__":
    main()
