"""Core voyage evaluation engine.

Contains the per-voyage simulation loop (evaluate_voyage), the round-trip
combiner, and the parallel worker function for annual runs.
"""

from datetime import datetime, timedelta

import numpy as np
from optimiser import find_min_fuel_operating_point

from models.constants import (
    GENSET_SFOC,
    GEAR_RATIO,
    HULL_SPEEDS_KN,
    HULL_RESISTANCE_KN,
    HULL_T_DEDUCTION,
    HULL_THRUST_CALM_KN,
    HULL_WAKE,
    KN_TO_MS,
    SHAFT_EFF,
)
from models.roughness import (
    hull_roughness_delta_R_kN,
    propeller_roughness_fuel_factor,
)
from models.weather import WeatherAlongRoute
from models.wind_resistance import wind_resistance_kN
from .results import HourlyResult, VoyageResult


def _combine_round_trip(outbound: VoyageResult, inbound: VoyageResult) -> VoyageResult:
    """Combine outbound + return voyage into a single round-trip VoyageResult.

    Fuel totals are summed.  Weather / force means are averaged (weighted
    equally -- both legs have the same number of hours at the same speed).
    Feasibility counters are summed.  Hourly lists are concatenated
    (outbound first, then return).  ``departure`` is the outbound departure.
    """
    n_out = outbound.n_hours_total or 1
    n_ret = inbound.n_hours_total or 1
    n_total = n_out + n_ret

    total_fac = outbound.total_fuel_factory_kg + inbound.total_fuel_factory_kg
    total_fac_nf = (outbound.total_fuel_factory_no_flettner_kg
                    + inbound.total_fuel_factory_no_flettner_kg)
    total_opt_nf = (outbound.total_fuel_opt_no_flettner_kg
                    + inbound.total_fuel_opt_no_flettner_kg)
    total_opt = outbound.total_fuel_optimised_kg + inbound.total_fuel_optimised_kg

    saving_kg = total_fac - total_opt
    saving_pct = 100.0 * saving_kg / total_fac if total_fac > 0 else 0.0
    saving_pr = total_fac_nf - total_opt_nf
    saving_fl = total_opt_nf - total_opt

    return VoyageResult(
        departure=outbound.departure,
        total_hours=outbound.total_hours + inbound.total_hours,
        total_fuel_factory_kg=total_fac,
        total_fuel_factory_no_flettner_kg=total_fac_nf,
        total_fuel_opt_no_flettner_kg=total_opt_nf,
        total_fuel_optimised_kg=total_opt,
        saving_pct=saving_pct,
        saving_kg=saving_kg,
        saving_pitch_rpm_kg=saving_pr,
        saving_flettner_kg=saving_fl,
        mean_hs=(outbound.mean_hs * n_out + inbound.mean_hs * n_ret) / n_total,
        mean_wind=(outbound.mean_wind * n_out
                   + inbound.mean_wind * n_ret) / n_total,
        mean_R_aw_kN=(outbound.mean_R_aw_kN * n_out
                      + inbound.mean_R_aw_kN * n_ret) / n_total,
        mean_R_wind_kN=(outbound.mean_R_wind_kN * n_out
                        + inbound.mean_R_wind_kN * n_ret) / n_total,
        mean_F_flettner_kN=(outbound.mean_F_flettner_kN * n_out
                           + inbound.mean_F_flettner_kN * n_ret) / n_total,
        mean_rotor_power_kW=(outbound.mean_rotor_power_kW * n_out
                             + inbound.mean_rotor_power_kW * n_ret) / n_total,
        total_rotor_fuel_kg=(outbound.total_rotor_fuel_kg
                              + inbound.total_rotor_fuel_kg),
        hull_ks_um=outbound.hull_ks_um,
        blade_ks_um=outbound.blade_ks_um,
        R_roughness_kN=outbound.R_roughness_kN,
        blade_fuel_factor=outbound.blade_fuel_factor,
        n_hours_both_feasible=(outbound.n_hours_both_feasible
                               + inbound.n_hours_both_feasible),
        n_hours_factory_infeasible=(outbound.n_hours_factory_infeasible
                                    + inbound.n_hours_factory_infeasible),
        n_hours_total=outbound.n_hours_total + inbound.n_hours_total,
        hourly=outbound.hourly + inbound.hourly,
    )


def evaluate_voyage(
    departure: datetime,
    route,
    weather,
    drift_tf,
    flettner,
    factory,
    prop,
    engine,
    speed_kn: float = 10.0,
    verbose: bool = False,
    _opt_cache: dict | None = None,
    _factory_cache: dict | None = None,
    hull_ks_m: float = 0.0,
    blade_ks_m: float = 0.0,
) -> VoyageResult:
    """Evaluate a single voyage, comparing factory vs optimised fuel consumption.

    Steps along the route at hourly intervals, computing thrust demand from:
    - Calm-water resistance (interpolated from hull data)
    - Added resistance in waves (from PdStrip drift TF + NORA3 Hs/Tp)
    - Flettner rotor thrust reduction (from NORA3 wind + heading)

    Then evaluates fuel rate under factory combinator and optimised pitch/RPM.

    Parameters
    ----------
    _opt_cache : dict, optional
        Pre-computed lookup table mapping quantised thrust [kN] to optimiser
        result dict.  Built by build_optimiser_cache() for speed.
    _factory_cache : dict, optional
        Pre-computed factory combinator lookup.
    """
    route_points = route.interpolate(dt_hours=1.0)

    # Interpolate hull data for this speed
    w = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_WAKE))
    t_ded = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_T_DEDUCTION))
    R_calm_kN = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_RESISTANCE_KN))
    Vs = speed_kn * KN_TO_MS
    Va = Vs * (1.0 - w)

    # Hull roughness: extra resistance above clean baseline [kN]
    # Constant for a given speed -- computed once, not per-hour.
    R_roughness_kN = 0.0
    if hull_ks_m > 30e-6:
        R_roughness_kN = hull_roughness_delta_R_kN(speed_kn, hull_ks_m)

    # Propeller blade roughness: fuel rate multiplier (>= 1.0)
    # Depends on RPM which varies per operating point, but the blade
    # section Reynolds number at 0.75R is dominated by tangential velocity
    # (omega*r >> Va), so the fuel factor is insensitive to exact RPM.
    # Use nominal shaft RPM scaled from max (117.6 RPM at ~15 kn).
    blade_fuel_factor = 1.0
    if blade_ks_m > 10e-6:
        nominal_shaft_rpm = 117.6 * (speed_kn / 15.0)
        blade_fuel_factor = propeller_roughness_fuel_factor(
            speed_kn, nominal_shaft_rpm, blade_ks_m)

    hourly_results = []
    total_factory_feasible = 0.0     # factory fuel (all-four-feasible hours) [kg]
    total_factory_nf_feasible = 0.0  # factory at no-Flettner thrust [kg]
    total_optimised_feasible = 0.0   # optimiser fuel (all-four-feasible hours) [kg]
    total_opt_noflettner_feasible = 0.0  # opt-no-flettner [kg]
    n_both_feasible = 0
    n_factory_infeasible = 0
    sum_hs = 0.0
    sum_wind = 0.0
    sum_R_aw = 0.0
    sum_R_wind = 0.0
    sum_F_flettner = 0.0
    sum_rotor_power = 0.0
    sum_rotor_fuel = 0.0

    for i, rp in enumerate(route_points):
        dt = departure + timedelta(hours=rp.time_hours)

        # Get weather at this position and time
        wx = weather.get_weather(rp.lat, rp.lon, dt)

        # Added resistance from waves
        R_aw_kN = drift_tf.added_resistance(
            Hs=wx["hs"], Tp=wx["tp"],
            wave_dir_deg=wx["wave_dir"],
            heading_deg=rp.heading_deg,
        )

        # Flettner thrust
        flettner.compute_from_true_wind(
            Vs=Vs,
            heading_deg=rp.heading_deg,
            wind_speed=wx["wind_speed"],
            wind_dir_deg=wx["wind_dir"],
        )
        F_flettner_kN = max(0.0, flettner.surge_force_kN)
        rotor_power_kW = flettner.rotor_power_kW if F_flettner_kN > 0 else 0.0
        # Fuel cost of spinning the rotor [g/h] via auxiliary genset
        rotor_fuel_gh = rotor_power_kW * GENSET_SFOC  # g/h

        # Wind resistance on hull (Blendermann model)
        R_wind_kN = wind_resistance_kN(
            Vs=Vs,
            heading_deg=rp.heading_deg,
            wind_speed=wx["wind_speed"],
            wind_dir_deg=wx["wind_dir"],
        )

        # Net thrust demand on propeller
        # T * (1-t) = R_calm + R_aw + R_wind + R_roughness - F_flettner
        # => T = (R_calm + R_aw + R_wind + R_roughness - F_flettner) / (1-t)
        R_total_kN = R_calm_kN + R_aw_kN + R_wind_kN + R_roughness_kN - F_flettner_kN
        T_required_kN = R_total_kN / (1.0 - t_ded)
        T_required_kN = max(0.0, T_required_kN)  # can't have negative thrust
        T_required_N = T_required_kN * 1000.0

        # Thrust demand without Flettner (for splitting savings)
        R_total_nfl_kN = R_calm_kN + R_aw_kN + R_wind_kN + R_roughness_kN
        T_no_flettner_kN = R_total_nfl_kN / (1.0 - t_ded)
        T_no_flettner_kN = max(0.0, T_no_flettner_kN)

        # Evaluate factory combinator
        if _factory_cache is not None:
            T_key = round(T_required_kN * 2) / 2.0
            fr = _factory_cache.get(T_key)
        else:
            fr = factory.evaluate(T_required_N, Va)

        # Evaluate optimised pitch/RPM
        o_fuel = None
        o_pitch = None
        o_rpm = None
        if T_required_N > 100:  # need meaningful thrust
            if _opt_cache is not None:
                # Use pre-computed lookup (quantised to 0.5 kN)
                T_key = round(T_required_kN * 2) / 2.0
                cached = _opt_cache.get(T_key)
                if cached is not None:
                    o_fuel = cached["fuel_rate"]
                    o_pitch = cached["pitch"]
                    o_rpm = cached["rpm"]
            else:
                op = find_min_fuel_operating_point(
                    prop, Va, T_required_N, engine,
                    gear_ratio=GEAR_RATIO,
                    shaft_efficiency=SHAFT_EFF,
                    pitch_step=0.02,
                )
                if op.found and op.fuel_rate is not None:
                    o_fuel = op.fuel_rate
                    o_pitch = op.pitch
                    o_rpm = op.rpm

        # Evaluate optimised pitch/RPM WITHOUT Flettner (same cache, different thrust)
        o_nf_fuel = None
        T_nf_N = T_no_flettner_kN * 1000.0
        if T_nf_N > 100:
            if _opt_cache is not None:
                T_nf_key = round(T_no_flettner_kN * 2) / 2.0
                cached_nf = _opt_cache.get(T_nf_key)
                if cached_nf is not None:
                    o_nf_fuel = cached_nf["fuel_rate"]
            else:
                op_nf = find_min_fuel_operating_point(
                    prop, Va, T_nf_N, engine,
                    gear_ratio=GEAR_RATIO,
                    shaft_efficiency=SHAFT_EFF,
                    pitch_step=0.02,
                )
                if op_nf.found and op_nf.fuel_rate is not None:
                    o_nf_fuel = op_nf.fuel_rate

        # Evaluate factory combinator at no-Flettner thrust (for split baseline)
        fr_nf = None
        if T_nf_N > 100:
            if _factory_cache is not None:
                T_nf_key = round(T_no_flettner_kN * 2) / 2.0
                fr_nf = _factory_cache.get(T_nf_key)
            else:
                fr_nf = factory.evaluate(T_nf_N, Va)

        hr = HourlyResult(
            time_hours=rp.time_hours,
            lat=rp.lat, lon=rp.lon,
            heading_deg=rp.heading_deg,
            hs=wx["hs"], tp=wx["tp"],
            wind_speed=wx["wind_speed"],
            R_calm_kN=R_calm_kN,
            R_aw_kN=R_aw_kN,
            R_wind_kN=R_wind_kN,
            F_flettner_kN=F_flettner_kN,
            T_required_kN=T_required_kN,
            T_required_no_flettner_kN=T_no_flettner_kN,
            factory_fuel_rate=fr["fuel_rate"] if fr else None,
            factory_no_flettner_fuel_rate=fr_nf["fuel_rate"] if fr_nf else None,
            optimised_fuel_rate=o_fuel,
            optimised_no_flettner_fuel_rate=o_nf_fuel,
            rotor_power_kW=rotor_power_kW,
            rotor_fuel_rate=rotor_fuel_gh,
            factory_pitch=fr["pitch"] if fr else None,
            factory_rpm=fr["rpm"] if fr else None,
            optimised_pitch=o_pitch,
            optimised_rpm=o_rpm,
        )
        hourly_results.append(hr)

        # Accumulate fuel (g/h -> kg for 1 hour interval)
        # "all-four feasible" = hours where factory+Fl, factory_NF, opt+Fl,
        # and opt_NF are all feasible.  This ensures a consistent basis for
        # computing split savings.
        all_four = (fr is not None and fr_nf is not None
                    and o_fuel is not None and o_nf_fuel is not None)
        if all_four:
            rotor_fuel_kg = rotor_fuel_gh / 1000.0  # g/h -> kg for 1h
            total_factory_feasible += fr["fuel_rate"] / 1000.0 * blade_fuel_factor + rotor_fuel_kg
            total_factory_nf_feasible += fr_nf["fuel_rate"] / 1000.0 * blade_fuel_factor
            total_optimised_feasible += o_fuel / 1000.0 * blade_fuel_factor + rotor_fuel_kg
            total_opt_noflettner_feasible += o_nf_fuel / 1000.0 * blade_fuel_factor
            n_both_feasible += 1
        elif fr is None and o_fuel is not None:
            # Factory infeasible but optimiser can deliver
            n_factory_infeasible += 1

        sum_hs += wx["hs"]
        sum_wind += wx["wind_speed"]
        sum_R_aw += R_aw_kN
        sum_R_wind += R_wind_kN
        sum_F_flettner += F_flettner_kN
        sum_rotor_power += rotor_power_kW
        sum_rotor_fuel += rotor_fuel_gh / 1000.0  # g/h -> kg for 1h

    n_pts = len(route_points)
    # Overall saving: factory (with Flettner) vs optimiser (with Flettner)
    # Both evaluated at the same Flettner-reduced thrust.
    saving_kg = total_factory_feasible - total_optimised_feasible
    saving_pct = (100.0 * saving_kg / total_factory_feasible) if total_factory_feasible > 0 else 0.0

    # Split savings (feasible-hours basis):
    # Baseline = Factory at no-Flettner thrust.
    # Total saving = Factory_NF - Opt_withFlettner.
    #   Pitch/RPM saving = Factory_NF - Opt_NF (both at same thrust, no Flettner)
    #   Flettner saving = Opt_NF - Opt_withFlettner (Flettner thrust reduction)
    saving_pitch_rpm_kg = total_factory_nf_feasible - total_opt_noflettner_feasible
    saving_flettner_kg = total_opt_noflettner_feasible - total_optimised_feasible

    if verbose:
        infeasible_str = ""
        if n_factory_infeasible > 0:
            infeasible_str = (f", factory_infeasible={n_factory_infeasible}/"
                              f"{n_pts}h")
        baseline = total_factory_nf_feasible if total_factory_nf_feasible > 0 else total_factory_feasible
        pct_pr = (100.0 * saving_pitch_rpm_kg / baseline) if baseline > 0 else 0.0
        pct_fl = (100.0 * saving_flettner_kg / baseline) if baseline > 0 else 0.0
        print(f"  {departure.strftime('%Y-%m-%d')}: "
              f"save={saving_pct:+.1f}% "
              f"(pitch/RPM {pct_pr:+.1f}%, Flettner {pct_fl:+.1f}%), "
              f"Hs={sum_hs / n_pts:.1f} m, "
              f"wind={sum_wind / n_pts:.1f} m/s"
              f"{infeasible_str}")

    return VoyageResult(
        departure=departure,
        total_hours=route.total_time_hours,
        total_fuel_factory_kg=total_factory_feasible,
        total_fuel_factory_no_flettner_kg=total_factory_nf_feasible,
        total_fuel_opt_no_flettner_kg=total_opt_noflettner_feasible,
        total_fuel_optimised_kg=total_optimised_feasible,
        saving_pct=saving_pct,
        saving_kg=saving_kg,
        saving_pitch_rpm_kg=saving_pitch_rpm_kg,
        saving_flettner_kg=saving_flettner_kg,
        mean_hs=sum_hs / n_pts,
        mean_wind=sum_wind / n_pts,
        mean_R_aw_kN=sum_R_aw / n_pts,
        mean_R_wind_kN=sum_R_wind / n_pts,
        mean_F_flettner_kN=sum_F_flettner / n_pts,
        mean_rotor_power_kW=sum_rotor_power / n_pts,
        total_rotor_fuel_kg=sum_rotor_fuel,
        hull_ks_um=hull_ks_m * 1e6,
        blade_ks_um=blade_ks_m * 1e6,
        R_roughness_kN=R_roughness_kN,
        blade_fuel_factor=blade_fuel_factor,
        n_hours_both_feasible=n_both_feasible,
        n_hours_factory_infeasible=n_factory_infeasible,
        n_hours_total=n_pts,
        hourly=hourly_results,
    )


def _voyage_worker(args):
    """Worker function for parallel voyage evaluation.

    Each worker opens its own WeatherAlongRoute (xarray datasets can't be
    pickled), evaluates a batch of departures, and returns the results.

    If a return route is provided (round-trip mode), each departure yields
    an outbound voyage followed by a return voyage whose departure time is
    the outbound arrival time.  The two legs are combined into a single
    round-trip ``VoyageResult`` via ``_combine_round_trip()``.
    """
    (departures, route, data_dir, drift_tf, flettner, factory,
     prop, engine, speed_kn, opt_cache, factory_cache,
     return_route, hull_ks_m, blade_ks_m) = args
    weather = WeatherAlongRoute(data_dir)
    results = []
    for departure in departures:
        try:
            vr_out = evaluate_voyage(
                departure=departure,
                route=route,
                weather=weather,
                drift_tf=drift_tf,
                flettner=flettner,
                factory=factory,
                prop=prop,
                engine=engine,
                speed_kn=speed_kn,
                verbose=False,
                _opt_cache=opt_cache,
                _factory_cache=factory_cache,
                hull_ks_m=hull_ks_m,
                blade_ks_m=blade_ks_m,
            )
            if return_route is not None:
                # Return leg departs when outbound arrives
                return_departure = departure + timedelta(hours=vr_out.total_hours)
                vr_ret = evaluate_voyage(
                    departure=return_departure,
                    route=return_route,
                    weather=weather,
                    drift_tf=drift_tf,
                    flettner=flettner,
                    factory=factory,
                    prop=prop,
                    engine=engine,
                    speed_kn=speed_kn,
                    verbose=False,
                    _opt_cache=opt_cache,
                    _factory_cache=factory_cache,
                    hull_ks_m=hull_ks_m,
                    blade_ks_m=blade_ks_m,
                )
                vr = _combine_round_trip(vr_out, vr_ret)
            else:
                vr = vr_out
            results.append(vr)
        except Exception as e:
            results.append((departure, str(e)))
    weather.close()
    return results
