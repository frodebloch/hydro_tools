"""Factory combinator: vessel 206 pitch/RPM schedule from propeller curve."""

import math
from typing import Optional

import numpy as np
from propeller_model import CSeriesPropeller

from .constants import (
    GEAR_RATIO,
    HULL_RESISTANCE_KN,
    HULL_SPEEDS_KN,
    HULL_T_DEDUCTION,
    HULL_WAKE,
    SHAFT_EFF,
)


class FactoryCombinator:
    """Vessel 206 factory combinator: schedule built from propeller curve.

    Replicates BruCon's MakeNormalCombinator() algorithm:
    for each trial speed, find the (pitch, RPM) pair where the propeller
    delivers the required thrust (with 15% sea margin) AND the engine
    power matches the muzzle diagram's suggested propeller curve power.

    This places the operating line on the propeller curve drawn on the
    muzzle diagram, which sits ~300 kW below the torque limit.  In calm
    water (no sea margin), the engine is below the propeller curve.  As
    seas increase, load climbs towards and eventually past it, with
    headroom up to the torque limit for overload.
    """

    def __init__(self, engine, prop: CSeriesPropeller,
                 sea_margin: float = 0.15,
                 sg_allowance_kw: float = 0.0,
                 sg_load_kw: Optional[float] = None,
                 engine_rpm_min: Optional[float] = None,
                 engine_rpm_max: Optional[float] = None):
        self.engine = engine
        self.prop = prop
        self.gear_ratio = GEAR_RATIO
        self.sg_allowance_kw = sg_allowance_kw   # headroom for schedule construction
        # Actual SG load for fuel evaluation (defaults to allowance for
        # backward compatibility, but typically smaller than the allowance)
        self.sg_load_kw = sg_load_kw if sg_load_kw is not None else sg_allowance_kw
        # RPM limits (default to engine's own limits)
        self._eng_rpm_min = engine_rpm_min if engine_rpm_min is not None else engine.min_rpm()
        self._eng_rpm_max = engine_rpm_max if engine_rpm_max is not None else engine.max_rpm()

        self._build_schedule(engine, prop, sea_margin)

    def _build_schedule(self, engine, prop, sea_margin):
        """Build the combinator schedule from the propeller curve.

        For each trial speed:
          1. Compute T_required from hull calm-water resistance + sea margin
          2. Sweep pitch, bisect RPM to deliver T_required at Va
          3. Select the (pitch, RPM) pair where engine power ~ PropellerCurvePower(engineRPM)
          4. Map the resulting schedule points to lever 0-100%
        """
        min_eng_rpm = self._eng_rpm_min
        max_eng_rpm = self._eng_rpm_max
        n_min = (min_eng_rpm / self.gear_ratio) / 60.0  # rev/s
        n_max = (max_eng_rpm / self.gear_ratio) / 60.0  # rev/s

        schedule = []  # list of (T_kN, pitch, shaft_rpm)

        for i, speed_kn in enumerate(HULL_SPEEDS_KN):
            Va_ms = speed_kn * 0.5144 * (1.0 - float(np.interp(
                speed_kn, HULL_SPEEDS_KN, HULL_WAKE)))
            # Calm-water resistance + sea margin -> propeller thrust
            R_calm_kN = float(np.interp(speed_kn, HULL_SPEEDS_KN,
                                        HULL_RESISTANCE_KN))
            t_ded = float(np.interp(speed_kn, HULL_SPEEDS_KN,
                                    HULL_T_DEDUCTION))
            T_req_N = R_calm_kN * (1.0 + sea_margin) / (1.0 - t_ded) * 1000.0

            best_pitch = 0.0
            best_rpm = 0.0
            best_err = 1e30

            def _try_pitch(p):
                """Try a single pitch value. Returns (err, P_engine - P_target, pitch, shaft_rpm) or None."""
                # Bisect RPM to find n that delivers T_req at this pitch
                T_at_max = prop.thrust(p, n_max, Va_ms)
                if T_at_max < T_req_N:
                    return None

                lo_n, hi_n = n_min, n_max
                for _ in range(40):
                    mid = (lo_n + hi_n) * 0.5
                    T = prop.thrust(p, mid, Va_ms)
                    if T < T_req_N:
                        lo_n = mid
                    else:
                        hi_n = mid
                n = (lo_n + hi_n) * 0.5
                shaft_rpm = n * 60.0
                eng_rpm = shaft_rpm * self.gear_ratio

                if eng_rpm < min_eng_rpm - 1.0 or eng_rpm > max_eng_rpm + 1.0:
                    return None

                Q = prop.torque(p, n, Va_ms)
                P_shaft = 2.0 * math.pi * n * Q  # Watts
                P_engine = (P_shaft / 1000.0) / SHAFT_EFF  # kW
                # Target: propeller curve power (~300 kW below torque limit).
                # Total engine load = P_propulsion + P_sg must sit on this
                # curve, keeping the same ~300 kW headroom as without SG.
                P_target = engine.propeller_curve_power(eng_rpm)  # kW
                if P_target <= 0.0:
                    return None
                # Total engine load includes SG allowance
                P_engine_total = P_engine + self.sg_allowance_kw
                if P_engine_total > engine.max_power(eng_rpm):
                    return None

                signed_err = P_engine_total - P_target
                return (abs(signed_err), signed_err, p, shaft_rpm)

            # Very coarse sweep (0.05 steps) to bracket the zero-crossing,
            # then bisect on pitch to find P_engine ~ P_target.
            coarse_results = []
            for p_int in range(6, 31):  # pitch 0.30 to 1.50 in 0.05 steps
                p = p_int * 0.05
                r = _try_pitch(p)
                if r is not None:
                    coarse_results.append(r)
                    if r[0] < best_err:
                        best_err, _, best_pitch, best_rpm = r

            # Find the bracket where signed_err changes sign (P_engine crosses P_target)
            # Then bisect on pitch within that bracket.
            if len(coarse_results) >= 2:
                for k in range(len(coarse_results) - 1):
                    err_a = coarse_results[k][1]   # signed error
                    err_b = coarse_results[k + 1][1]
                    if err_a * err_b <= 0:  # sign change
                        p_lo = coarse_results[k][2]
                        p_hi = coarse_results[k + 1][2]
                        for _ in range(20):  # bisection on pitch
                            p_mid = (p_lo + p_hi) * 0.5
                            r = _try_pitch(p_mid)
                            if r is None:
                                p_lo = p_mid
                                continue
                            if r[0] < best_err:
                                best_err, _, best_pitch, best_rpm = r
                            if r[1] < 0:  # P_engine < P_target -> need more pitch
                                p_lo = p_mid
                            else:
                                p_hi = p_mid
                        break

            if best_err < 1e29:
                schedule.append((T_req_N / 1000.0, best_pitch, best_rpm))

        # Sort by thrust
        schedule.sort(key=lambda x: x[0])

        if len(schedule) == 0:
            raise RuntimeError("FactoryCombinator: no feasible schedule points")

        # Insert lever-0 point: zero thrust at zero speed (Va=0).
        # At Va=0 with P/D=0, the propeller produces no thrust.
        min_shaft_rpm = min_eng_rpm / self.gear_ratio
        schedule.insert(0, (0.0, 0.0, min_shaft_rpm))

        N = len(schedule)
        # Map to lever: 0% = lowest thrust (feathered), 100% = highest thrust
        self._combo_lever = np.array(
            [100.0 * i / (N - 1) if N > 1 else 0.0 for i in range(N)])
        self._combo_thrust_kn = np.array([s[0] for s in schedule])
        self._combo_pitch = np.array([s[1] for s in schedule])
        self._combo_rpm = np.array([s[2] for s in schedule])

    def _rpm(self, lever: float) -> float:
        return float(np.interp(lever, self._combo_lever, self._combo_rpm))

    def _pitch(self, lever: float) -> float:
        return float(np.interp(lever, self._combo_lever, self._combo_pitch))

    def evaluate(self, T_required_N: float, Va: float) -> Optional[dict]:
        """Find the factory combinator operating point for given thrust.

        Returns dict with fuel_rate [g/h], pitch, rpm, power_kw, engine_rpm,
        eta0, or None if infeasible.
        """
        # Bisection on lever to find thrust equilibrium
        lv = self._find_lever(T_required_N, Va)
        if lv is None:
            return None

        pitch = self._pitch(lv)
        rpm = self._rpm(lv)
        n = rpm / 60.0
        eng_rpm = rpm * self.gear_ratio

        if n < 0.01 or pitch < 0.01:
            return None

        T_check = self.prop.thrust(pitch, n, Va)
        if abs(T_check - T_required_N) > 500:
            return None

        Q = self.prop.torque(pitch, n, Va)
        P_shaft = Q * 2.0 * math.pi * n
        P_shaft_kw = P_shaft / 1000.0
        P_eng_kw = P_shaft_kw / SHAFT_EFF + self.sg_load_kw
        eta0 = self.prop.eta0(pitch, n, Va)

        if (eng_rpm < self._eng_rpm_min
                or eng_rpm > self._eng_rpm_max
                or P_eng_kw <= 0
                or P_eng_kw > self.engine.max_power(eng_rpm)):
            return None

        fuel = self.engine.fuel_rate(P_eng_kw, eng_rpm)
        return {
            "fuel_rate": fuel,  # g/h
            "pitch": pitch,
            "rpm": rpm,
            "power_kw": P_shaft_kw,
            "engine_rpm": eng_rpm,
            "eta0": eta0,
        }

    def _find_lever(self, T_req_N: float, Va: float,
                    tol_N: float = 100) -> Optional[float]:
        """Bisection to find lever position delivering required thrust."""
        lo, hi = 0.0, 100.0

        p_hi, r_hi = self._pitch(hi), self._rpm(hi)
        n_hi = r_hi / 60.0
        T_max = self.prop.thrust(p_hi, n_hi, Va) if n_hi > 0.01 and p_hi > 0.01 else 0
        if T_req_N > T_max + tol_N:
            return None

        p_lo, r_lo = self._pitch(lo), self._rpm(lo)
        n_lo = r_lo / 60.0
        T_min = self.prop.thrust(p_lo, n_lo, Va) if n_lo > 0.01 and p_lo > 0.01 else 0
        if T_req_N < T_min - tol_N:
            return None

        for _ in range(80):
            mid = (lo + hi) / 2.0
            p = self._pitch(mid)
            r = self._rpm(mid)
            n = r / 60.0
            if n < 0.01 or p < 0.01:
                lo = mid
                continue
            T = self.prop.thrust(p, n, Va)
            if abs(T - T_req_N) < tol_N:
                return mid
            if T < T_req_N:
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2.0
