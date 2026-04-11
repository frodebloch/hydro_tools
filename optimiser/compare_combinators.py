#!/usr/bin/env python3
"""Compare our generated factory combinator with the hydrodynamicist's combinator.

Parses the real combinator from ~/Downloads/pdn.html (Plotly interactive P/D-N
diagram) and compares operating points side-by-side with our algorithmically
generated FactoryCombinator, at a range of thrust demands.

The real combinator's propeller curve gives (shaft_RPM, P_engine_kW) for 171
points.  To find P/D at each point, we use the iso-speed traces from the same
diagram to identify the ship speed, then invert our propeller model.
"""

import json
import math
import re
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import LinearNDInterpolator, griddata

from propeller_model import CSeriesPropeller, load_c_series_data
from optimiser import find_min_fuel_operating_point, make_man_l27_38
from models.combinator import FactoryCombinator
from models.constants import (
    DATA_PATH_C440, DATA_PATH_C455, DATA_PATH_C470,
    PROP_DIAMETER, PROP_BAR, PROP_DESIGN_PITCH,
    GEAR_RATIO, SHAFT_EFF, KN_TO_MS,
    HULL_SPEEDS_KN, HULL_WAKE, HULL_THRUST_CALM_KN,
    RHO_WATER,
)


# ============================================================
# Parse pdn.html
# ============================================================

def parse_pdn_html(path: str = None) -> dict:
    """Extract all Plotly traces from the P/D-N diagram HTML.

    Returns a dict with keys:
        engine_limit:  (rpm_engine, power_kw) arrays - engine load limit
        propeller_curve: (rpm_shaft, power_kw) arrays - 171-point combinator
        combinator_pts: (rpm_shaft, power_kw) arrays - 5 breakpoints
        levers: list of (rpm_shaft, power_kw) tuples for each lever position
        iso_pitch: dict of {pd_value: (rpm_shaft, power_kw) arrays}
        iso_speed: dict of {speed_kn: (rpm_shaft, power_kw) arrays}
    """
    if path is None:
        import os
        path = os.path.expanduser("~/Downloads/pdn.html")

    with open(path, "r") as f:
        html = f.read()

    match = re.search(
        r'Plotly\.newPlot\(\s*["\'][^"\']+["\'],\s*(\[.*?\])\s*,\s*(\{.*?\})\s*[,)]',
        html, re.DOTALL,
    )
    if not match:
        raise RuntimeError("Could not find Plotly.newPlot data in HTML")

    traces = json.loads(match.group(1))

    result = {
        "engine_limit": None,
        "propeller_curve": None,
        "combinator_pts": None,
        "levers": [],
        "iso_pitch": {},
        "iso_speed": {},
    }

    for trace in traces:
        name = trace.get("name", "")
        x_raw = trace.get("x", [])
        y_raw = trace.get("y", [])

        # Skip traces with binary/encoded data (e.g. Wagner zero pitch power)
        if isinstance(x_raw, dict) or isinstance(y_raw, dict):
            continue

        x = np.array(x_raw, dtype=float)
        y = np.array(y_raw, dtype=float)

        if "Load limit" in name:
            # Engine limit: x = engine RPM, y = power kW
            result["engine_limit"] = (x, y)

        elif name == "Propeller curve":
            result["propeller_curve"] = (x, y)  # shaft RPM, power kW

        elif name == "Combinator points":
            result["combinator_pts"] = (x, y)

        elif name.startswith("Lever "):
            pct = name.replace("Lever ", "").replace("%", "")
            try:
                lever_pct = float(pct)
            except ValueError:
                continue
            # Each lever trace has 2 points: (rpm, power) and (rpm, 0)
            if len(x) >= 1:
                result["levers"].append((lever_pct, float(x[0]), float(y[0])))

        elif name.startswith("P/D "):
            pd_val = float(name.replace("P/D ", ""))
            result["iso_pitch"][pd_val] = (x, y)

        elif name.startswith("Vs "):
            speed_kn = float(name.replace("Vs ", "").replace(" knots", ""))
            result["iso_speed"][speed_kn] = (x, y)

    return result


# ============================================================
# Determine P/D along the real propeller curve
# ============================================================

def interpolate_pitch_on_propeller_curve(pdn_data: dict) -> np.ndarray:
    """Determine P/D at each point of the real propeller curve.

    Strategy: build a 2D interpolation of P/D on the (RPM, Power) plane
    using all the iso-speed / iso-pitch grid points from the diagram.
    Each iso-speed trace has 7 points (one per P/D = 0.4, 0.5, ..., 1.0).
    """
    # Build the grid: each iso-speed trace has 7 points at known P/D values
    pd_values = sorted(pdn_data["iso_pitch"].keys())  # [0.4, 0.5, ..., 1.0]

    # Collect all grid points: (rpm, power) -> pd
    rpm_pts = []
    power_pts = []
    pd_pts = []

    for speed_kn, (rpm_arr, power_arr) in sorted(pdn_data["iso_speed"].items()):
        # Points are ordered P/D 0.4 -> 1.0 (highest RPM first, decreasing)
        assert len(rpm_arr) == len(pd_values), (
            f"Speed {speed_kn}: expected {len(pd_values)} points, got {len(rpm_arr)}")
        for j, pd_val in enumerate(pd_values):
            rpm_pts.append(rpm_arr[j])
            power_pts.append(power_arr[j])
            pd_pts.append(pd_val)

    rpm_pts = np.array(rpm_pts)
    power_pts = np.array(power_pts)
    pd_pts = np.array(pd_pts)

    # Interpolate P/D for each propeller curve point
    prop_rpm, prop_power = pdn_data["propeller_curve"]

    # Use scipy griddata for 2D interpolation
    pd_on_curve = griddata(
        np.column_stack([rpm_pts, power_pts]),
        pd_pts,
        np.column_stack([prop_rpm, prop_power]),
        method="cubic",
    )

    # Fill any NaN (extrapolation) with nearest
    nan_mask = np.isnan(pd_on_curve)
    if nan_mask.any():
        pd_nearest = griddata(
            np.column_stack([rpm_pts, power_pts]),
            pd_pts,
            np.column_stack([prop_rpm[nan_mask], prop_power[nan_mask]]),
            method="nearest",
        )
        pd_on_curve[nan_mask] = pd_nearest

    return pd_on_curve


def interpolate_speed_on_propeller_curve(pdn_data: dict) -> np.ndarray:
    """Determine ship speed (kn) at each point of the real propeller curve.

    Same strategy: 2D interpolation from the iso-speed/iso-pitch grid.
    """
    pd_values = sorted(pdn_data["iso_pitch"].keys())

    rpm_pts = []
    power_pts = []
    speed_pts = []

    for speed_kn, (rpm_arr, power_arr) in sorted(pdn_data["iso_speed"].items()):
        for j in range(len(rpm_arr)):
            rpm_pts.append(rpm_arr[j])
            power_pts.append(power_arr[j])
            speed_pts.append(speed_kn)

    rpm_pts = np.array(rpm_pts)
    power_pts = np.array(power_pts)
    speed_pts = np.array(speed_pts)

    prop_rpm, prop_power = pdn_data["propeller_curve"]

    speed_on_curve = griddata(
        np.column_stack([rpm_pts, power_pts]),
        speed_pts,
        np.column_stack([prop_rpm, prop_power]),
        method="cubic",
    )

    nan_mask = np.isnan(speed_on_curve)
    if nan_mask.any():
        speed_nearest = griddata(
            np.column_stack([rpm_pts, power_pts]),
            speed_pts,
            np.column_stack([prop_rpm[nan_mask], prop_power[nan_mask]]),
            method="nearest",
        )
        speed_on_curve[nan_mask] = speed_nearest

    return speed_on_curve


# ============================================================
# RealCombinator: evaluator using the extracted propeller curve
# ============================================================

@dataclass
class RealCombinatorPoint:
    """A single point on the real combinator's propeller curve."""
    shaft_rpm: float
    power_kw: float  # engine power
    pitch: float     # P/D (interpolated)
    speed_kn: float  # ship speed (interpolated)


class RealCombinator:
    """Real factory combinator extracted from the hydrodynamicist's P/D-N diagram.

    Stores the 171-point propeller curve with interpolated P/D and speed at
    each point.  Evaluates operating points by finding the curve point that
    matches a given thrust demand (using our propeller model to compute thrust).
    """

    def __init__(self, pdn_data: dict, prop: CSeriesPropeller, engine):
        self.prop = prop
        self.engine = engine
        self.gear_ratio = GEAR_RATIO

        prop_rpm, prop_power = pdn_data["propeller_curve"]
        pd_on_curve = interpolate_pitch_on_propeller_curve(pdn_data)
        speed_on_curve = interpolate_speed_on_propeller_curve(pdn_data)

        self.points = []
        for i in range(len(prop_rpm)):
            self.points.append(RealCombinatorPoint(
                shaft_rpm=prop_rpm[i],
                power_kw=prop_power[i],
                pitch=pd_on_curve[i],
                speed_kn=speed_on_curve[i],
            ))

        # Sort by power (proxy for thrust ordering)
        self.points.sort(key=lambda p: p.power_kw)

        # Pre-compute thrust at each curve point using our propeller model,
        # at the interpolated speed. This allows bisection on thrust.
        self._thrust_N = np.zeros(len(self.points))
        self._shaft_rpm = np.array([p.shaft_rpm for p in self.points])
        self._pitch = np.array([p.pitch for p in self.points])
        self._power_kw = np.array([p.power_kw for p in self.points])
        self._speed_kn = np.array([p.speed_kn for p in self.points])

        for i, pt in enumerate(self.points):
            Va = self._va_from_speed(pt.speed_kn)
            n = pt.shaft_rpm / 60.0
            self._thrust_N[i] = prop.thrust(pt.pitch, n, Va)

    def _va_from_speed(self, speed_kn: float) -> float:
        """Compute advance velocity from ship speed using hull wake fraction."""
        w = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_WAKE))
        return speed_kn * KN_TO_MS * (1.0 - w)

    def evaluate(self, T_required_N: float, Va: float) -> Optional[dict]:
        """Find the real combinator operating point for a given thrust and Va.

        Interpolates along the propeller curve to find the (pitch, RPM)
        point that delivers the requested thrust at the given Va.

        Returns dict matching FactoryCombinator.evaluate() format.
        """
        # For the given Va, compute thrust at each curve point
        thrusts = np.array([
            self.prop.thrust(self._pitch[i], self._shaft_rpm[i] / 60.0, Va)
            for i in range(len(self.points))
        ])

        # Check feasibility
        if T_required_N < thrusts.min() or T_required_N > thrusts.max():
            return None

        # Interpolate: find fractional index where thrust = T_required
        # thrusts should be monotonically increasing (sorted by power)
        idx = np.searchsorted(thrusts, T_required_N)
        if idx == 0:
            idx = 1
        if idx >= len(thrusts):
            idx = len(thrusts) - 1

        # Linear interpolation between idx-1 and idx
        T_lo = thrusts[idx - 1]
        T_hi = thrusts[idx]
        if abs(T_hi - T_lo) < 1.0:
            frac = 0.5
        else:
            frac = (T_required_N - T_lo) / (T_hi - T_lo)

        pitch = self._pitch[idx - 1] + frac * (self._pitch[idx] - self._pitch[idx - 1])
        rpm = self._shaft_rpm[idx - 1] + frac * (self._shaft_rpm[idx] - self._shaft_rpm[idx - 1])
        n = rpm / 60.0
        eng_rpm = rpm * self.gear_ratio

        if n < 0.01 or pitch < 0.01:
            return None

        T_check = self.prop.thrust(pitch, n, Va)
        Q = self.prop.torque(pitch, n, Va)
        P_shaft = Q * 2.0 * math.pi * n
        P_shaft_kw = P_shaft / 1000.0
        P_eng_kw = P_shaft_kw / SHAFT_EFF
        eta0 = self.prop.eta0(pitch, n, Va)

        eng_rpm_min = self.engine.min_rpm()
        eng_rpm_max = self.engine.max_rpm()
        if eng_rpm < eng_rpm_min or eng_rpm > eng_rpm_max:
            return None
        if P_eng_kw <= 0 or P_eng_kw > self.engine.max_power(eng_rpm):
            return None

        fuel = self.engine.fuel_rate(P_eng_kw, eng_rpm)
        return {
            "fuel_rate": fuel,
            "pitch": pitch,
            "rpm": rpm,
            "power_kw": P_shaft_kw,
            "engine_power_kw": P_eng_kw,
            "engine_rpm": eng_rpm,
            "eta0": eta0,
        }


# ============================================================
# Comparison
# ============================================================

def make_prop():
    """Create the standard propeller model."""
    data_40 = load_c_series_data(DATA_PATH_C440)
    data_55 = load_c_series_data(DATA_PATH_C455)
    data_70 = load_c_series_data(DATA_PATH_C470)
    bar_data = {0.40: data_40, 0.55: data_55, 0.70: data_70}
    return CSeriesPropeller(bar_data, design_pitch=PROP_DESIGN_PITCH,
                            diameter=PROP_DIAMETER, area_ratio=PROP_BAR,
                            rho=RHO_WATER)


def main():
    print("=" * 100)
    print("  Combinator Comparison: Generated (ours) vs Hydrodynamicist (real)")
    print("=" * 100)
    print()

    # --- Setup ---
    engine = make_man_l27_38()
    prop = make_prop()

    print("Parsing pdn.html...")
    pdn_data = parse_pdn_html()

    print("Building real combinator (interpolating P/D and speed)...")
    real = RealCombinator(pdn_data, prop, engine)

    print("Building our generated factory combinator...")
    generated = FactoryCombinator(engine, prop)

    # --- Print the real combinator schedule summary ---
    print()
    print("-" * 100)
    print("  Real combinator propeller curve (key points)")
    print("-" * 100)
    print(f"  {'#':>3s}  {'RPM':>7s}  {'P_eng[kW]':>10s}  {'P/D':>6s}  "
          f"{'Vs[kn]':>7s}  {'EngRPM':>8s}")
    # Print every ~20th point plus first/last
    indices = list(range(0, len(real.points), 20))
    if indices[-1] != len(real.points) - 1:
        indices.append(len(real.points) - 1)
    for i in indices:
        pt = real.points[i]
        eng_rpm = pt.shaft_rpm * GEAR_RATIO
        print(f"  {i:3d}  {pt.shaft_rpm:7.2f}  {pt.power_kw:10.1f}  "
              f"{pt.pitch:6.3f}  {pt.speed_kn:7.2f}  {eng_rpm:8.1f}")

    # --- Print lever positions ---
    print()
    print("  Real combinator lever positions:")
    print(f"  {'Lever%':>7s}  {'RPM':>7s}  {'P_eng[kW]':>10s}")
    for lev_pct, lev_rpm, lev_power in sorted(pdn_data["levers"]):
        print(f"  {lev_pct:7.0f}  {lev_rpm:7.2f}  {lev_power:10.1f}")

    # --- Print our generated combinator schedule ---
    print()
    print("-" * 100)
    print("  Our generated combinator schedule")
    print("-" * 100)
    print(f"  {'Lever%':>7s}  {'T[kN]':>8s}  {'P/D':>7s}  {'RPM':>7s}  {'EngRPM':>8s}")
    for i in range(len(generated._combo_lever)):
        lv = generated._combo_lever[i]
        t = generated._combo_thrust_kn[i]
        p = generated._combo_pitch[i]
        r = generated._combo_rpm[i]
        er = r * GEAR_RATIO
        print(f"  {lv:7.1f}  {t:8.1f}  {p:7.3f}  {r:7.1f}  {er:8.1f}")

    # --- Operating point comparison across thrust demands ---
    print()
    print("=" * 100)
    print("  Operating point comparison at 10 kn (calm water)")
    print("=" * 100)

    speed_kn = 10.0
    w = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_WAKE))
    Va = speed_kn * KN_TO_MS * (1.0 - w)
    print(f"  Speed: {speed_kn} kn, wake: {w:.3f}, Va: {Va:.3f} m/s")
    print()

    header = (f"  {'T[kN]':>7s}  "
              f"{'Gen P/D':>8s}  {'Gen RPM':>8s}  {'Gen fuel':>9s}  {'Gen eta0':>8s}  "
              f"{'Real P/D':>8s}  {'Real RPM':>9s}  {'Real fuel':>9s}  {'Real eta0':>8s}  "
              f"{'Opt P/D':>8s}  {'Opt RPM':>8s}  {'Opt fuel':>9s}  {'Opt eta0':>8s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for T_kN in np.arange(80, 220, 10):
        T_N = T_kN * 1000.0

        gen = generated.evaluate(T_N, Va)
        rea = real.evaluate(T_N, Va)
        opt = find_min_fuel_operating_point(
            prop, Va, T_N, engine,
            gear_ratio=GEAR_RATIO,
            shaft_efficiency=SHAFT_EFF,
            pitch_step=0.01,
        )

        parts = [f"  {T_kN:7.0f}"]

        if gen is not None:
            parts.append(f"  {gen['pitch']:8.3f}  {gen['rpm']:8.1f}  "
                         f"{gen['fuel_rate']:9.0f}  {gen['eta0']:8.4f}")
        else:
            parts.append(f"  {'---':>8s}  {'---':>8s}  {'---':>9s}  {'---':>8s}")

        if rea is not None:
            parts.append(f"  {rea['pitch']:8.3f}  {rea['rpm']:9.1f}  "
                         f"{rea['fuel_rate']:9.0f}  {rea['eta0']:8.4f}")
        else:
            parts.append(f"  {'---':>8s}  {'---':>9s}  {'---':>9s}  {'---':>8s}")

        if opt.found:
            parts.append(f"  {opt.pitch:8.3f}  {opt.rpm:8.1f}  "
                         f"{opt.fuel_rate:9.0f}  {opt.eta0:8.4f}")
        else:
            parts.append(f"  {'---':>8s}  {'---':>8s}  {'---':>9s}  {'---':>8s}")

        print("".join(parts))

    # --- Fuel saving comparison ---
    print()
    print("=" * 100)
    print("  Fuel saving: Generated vs Real vs Optimiser at 10 kn")
    print("=" * 100)
    print()
    print(f"  {'T[kN]':>7s}  {'Gen fuel':>9s}  {'Real fuel':>9s}  {'Opt fuel':>9s}  "
          f"{'Gen-Real':>9s}  {'GenSave%':>8s}  {'RealSave%':>9s}  {'Gen-Real%':>9s}")
    print("  " + "-" * 85)

    for T_kN in np.arange(80, 220, 10):
        T_N = T_kN * 1000.0

        gen = generated.evaluate(T_N, Va)
        rea = real.evaluate(T_N, Va)
        opt = find_min_fuel_operating_point(
            prop, Va, T_N, engine,
            gear_ratio=GEAR_RATIO,
            shaft_efficiency=SHAFT_EFF,
            pitch_step=0.01,
        )

        gen_fuel = gen["fuel_rate"] if gen else None
        rea_fuel = rea["fuel_rate"] if rea else None
        opt_fuel = opt.fuel_rate if opt.found else None

        parts = [f"  {T_kN:7.0f}"]

        parts.append(f"  {gen_fuel:9.0f}" if gen_fuel else f"  {'---':>9s}")
        parts.append(f"  {rea_fuel:9.0f}" if rea_fuel else f"  {'---':>9s}")
        parts.append(f"  {opt_fuel:9.0f}" if opt_fuel else f"  {'---':>9s}")

        if gen_fuel and rea_fuel:
            diff = gen_fuel - rea_fuel
            parts.append(f"  {diff:+9.0f}")
        else:
            parts.append(f"  {'---':>9s}")

        if gen_fuel and opt_fuel:
            save = 100.0 * (gen_fuel - opt_fuel) / gen_fuel
            parts.append(f"  {save:8.1f}%")
        else:
            parts.append(f"  {'---':>8s}")

        if rea_fuel and opt_fuel:
            save = 100.0 * (rea_fuel - opt_fuel) / rea_fuel
            parts.append(f"  {save:9.1f}%")
        else:
            parts.append(f"  {'---':>9s}")

        if gen_fuel and rea_fuel:
            diff_pct = 100.0 * (gen_fuel - rea_fuel) / rea_fuel
            parts.append(f"  {diff_pct:+9.1f}%")
        else:
            parts.append(f"  {'---':>9s}")

        print("".join(parts))

    # --- RPM comparison across speed range ---
    print()
    print("=" * 100)
    print("  RPM & P/D comparison across ship speeds (calm water, no sea margin)")
    print("=" * 100)
    print()
    print(f"  {'Vs[kn]':>7s}  {'T[kN]':>7s}  "
          f"{'Gen P/D':>8s}  {'Gen RPM':>8s}  "
          f"{'Real P/D':>8s}  {'Real RPM':>9s}  "
          f"{'dP/D':>6s}  {'dRPM':>6s}")
    print("  " + "-" * 80)

    for speed_kn in [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]:
        w = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_WAKE))
        Va = speed_kn * KN_TO_MS * (1.0 - w)
        T_calm_kN = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_THRUST_CALM_KN))
        T_N = T_calm_kN * 1000.0

        gen = generated.evaluate(T_N, Va)
        rea = real.evaluate(T_N, Va)

        gen_pd = gen["pitch"] if gen else None
        gen_rpm = gen["rpm"] if gen else None
        rea_pd = rea["pitch"] if rea else None
        rea_rpm = rea["rpm"] if rea else None

        parts = [f"  {speed_kn:7.1f}  {T_calm_kN:7.1f}"]

        if gen_pd is not None:
            parts.append(f"  {gen_pd:8.3f}  {gen_rpm:8.1f}")
        else:
            parts.append(f"  {'---':>8s}  {'---':>8s}")

        if rea_pd is not None:
            parts.append(f"  {rea_pd:8.3f}  {rea_rpm:9.1f}")
        else:
            parts.append(f"  {'---':>8s}  {'---':>9s}")

        if gen_pd is not None and rea_pd is not None:
            parts.append(f"  {gen_pd - rea_pd:+6.3f}  {gen_rpm - rea_rpm:+6.1f}")
        else:
            parts.append(f"  {'---':>6s}  {'---':>6s}")

        print("".join(parts))

    print()
    print("Notes:")
    print("  - Gen = our algorithmically generated FactoryCombinator")
    print("  - Real = hydrodynamicist's combinator from pdn.html")
    print("  - Opt = fuel-optimal (unconstrained pitch/RPM selection)")
    print("  - P/D on real curve interpolated from iso-speed/iso-pitch grid")
    print("  - Both use the same propeller model (C4-series) and engine model (MAN L27/38)")
    print("  - Our combinator uses 15% sea margin; real combinator's margin is unknown")


if __name__ == "__main__":
    main()
