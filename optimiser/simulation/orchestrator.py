"""Orchestration: annual comparison, speed sweep, scheduling analysis.

These are the top-level driver functions that build models, pre-compute
caches, and dispatch parallel voyage evaluations.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from optimiser import make_man_l27_38
from propeller_model import CSeriesPropeller, load_c_series_data

from models.combinator import FactoryCombinator
from models.constants import (
    DATA_PATH_C440,
    DATA_PATH_C455,
    DATA_PATH_C470,
    HULL_SPEEDS_KN,
    HULL_WAKE,
    KN_TO_MS,
    NORA3_DATA_DIR,
    PDSTRIP_DAT,
    PROP_BAR,
    PROP_DESIGN_PITCH,
    PROP_DIAMETER,
    RHO_WATER,
)
from models.route import ROUTE_ROTTERDAM_GOTHENBURG
from models.drift_force import DriftTransferFunction
from models.flettner import FlettnerRotor
from models.route import Route, Waypoint
from .cache import build_factory_cache, build_optimiser_cache
from .engine import _voyage_worker
from .results import SpeedSweepResult, VoyageResult


def _days_in_year(year: int) -> int:
    """Number of days in a given year."""
    return 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365


def run_annual_comparison(
    year: int,
    speed_kn: float = 10.0,
    waypoints: list[Waypoint] | None = None,
    data_dir: Path | None = None,
    pdstrip_path: str = PDSTRIP_DAT,
    flettner_enabled: bool = True,
    verbose: bool = True,
    sg_load_kw: float = 0.0,
    sg_factory_allowance_kw: float = 0.0,
    sg_freq_min: float = 0.0,
    sg_freq_max: float = 0.0,
    round_trip: bool = True,
    hull_ks_m: float = 0.0,
    blade_ks_m: float = 0.0,
) -> list[VoyageResult]:
    """Run the full annual comparison: one voyage per day.

    Parameters
    ----------
    sg_load_kw : float
        Actual shaft generator electrical load [kW] for the optimiser.
    sg_factory_allowance_kw : float
        SG power allowance baked into the factory combinator schedule [kW].
    sg_freq_min, sg_freq_max : float
        Shaft generator frequency band [Hz].  If both > 0, constrains
        engine RPM range.  The PTO gear ratio is sized so sg_freq_max
        corresponds to the engine's maximum RPM.
    round_trip : bool
        If True (default), each departure is a round trip: outbound on the
        given waypoints followed by an immediate return on the reversed
        route.  This eliminates directional wind bias from prevailing
        winds.  If False, only the outbound leg is evaluated.

    Returns a list of VoyageResult for each departure day.
    """
    if waypoints is None:
        waypoints = ROUTE_ROTTERDAM_GOTHENBURG
    if data_dir is None:
        data_dir = NORA3_DATA_DIR

    # --- Shaft generator RPM constraints ---
    engine_rpm_min_sg: Optional[float] = None
    engine_rpm_max_sg: Optional[float] = None
    if sg_freq_min > 0 and sg_freq_max > 0:
        # PTO gear ratio sized so sg_freq_max = engine max RPM
        # Engine RPM band = [max_rpm * freq_min/freq_max, max_rpm]
        _engine_max = make_man_l27_38().max_rpm()
        engine_rpm_min_sg = _engine_max * sg_freq_min / sg_freq_max
        engine_rpm_max_sg = float(_engine_max)

    print("=" * 78)
    print(f"ANNUAL VOYAGE COMPARISON: Factory Combinator vs Optimiser")
    print(f"  Year: {year}, Speed: {speed_kn} kn")
    trip_mode = "round-trip" if round_trip else "one-way"
    print(f"  Route: {waypoints[0].name} -> {waypoints[-1].name} ({trip_mode})")
    if sg_load_kw > 0 or sg_factory_allowance_kw > 0:
        print(f"  Shaft generator: factory allowance {sg_factory_allowance_kw:.0f} kW, "
              f"actual load {sg_load_kw:.0f} kW")
        if engine_rpm_min_sg is not None:
            print(f"  SG frequency band: {sg_freq_min:.0f}-{sg_freq_max:.0f} Hz "
                  f"-> engine RPM {engine_rpm_min_sg:.0f}-{engine_rpm_max_sg:.0f}")
    if hull_ks_m > 0 or blade_ks_m > 0:
        print(f"  Hull roughness: {hull_ks_m * 1e6:.0f} \u00b5m, "
              f"blade roughness: {blade_ks_m * 1e6:.0f} \u00b5m")
    print("=" * 78)

    # --- Build models ---
    print("\nLoading propeller model ...")
    data_40 = load_c_series_data(DATA_PATH_C440)
    data_55 = load_c_series_data(DATA_PATH_C455)
    data_70 = load_c_series_data(DATA_PATH_C470)
    bar_data = {0.40: data_40, 0.55: data_55, 0.70: data_70}
    prop = CSeriesPropeller(bar_data, design_pitch=PROP_DESIGN_PITCH,
                            diameter=PROP_DIAMETER, area_ratio=PROP_BAR,
                            rho=RHO_WATER)
    print(f"  Propeller: D={PROP_DIAMETER}m, P/D={PROP_DESIGN_PITCH}, BAR={PROP_BAR}")

    engine = make_man_l27_38()
    print(f"  Engine: {engine.name}")

    print("\nLoading PdStrip drift transfer function ...")
    drift_tf = DriftTransferFunction(pdstrip_path, speed_ms=speed_kn * KN_TO_MS)

    print("\nSetting up Flettner rotor ...")
    flettner = FlettnerRotor(height=28.0, diameter=4.0, max_rpm=220.0,
                             endplate_ratio=2.0, roughness_factor=3.0)
    if flettner_enabled:
        flettner.set_target_spin_ratio(3.0)
        print(f"  Rotor: H=28m, D=4m, max 220 RPM, target SR=3, k_rough=3.0")
    else:
        flettner.set_target_spin_ratio(0.0)  # effectively off
        print(f"  Flettner DISABLED (--no-flettner)")

    print("\nSetting up route ...")
    route = Route(waypoints, speed_kn)
    print(f"  Outbound: {route.total_distance_nm:.0f} nm, "
          f"{route.total_time_hours:.1f} hours")
    for i, leg in enumerate(route.legs):
        print(f"    Leg {i + 1}: {leg['from'].name} -> {leg['to'].name}: "
              f"{leg['dist_nm']:.0f} nm, bearing {leg['bearing_deg']:.0f} deg")

    # Build return route (reversed waypoints) for round-trip mode
    return_route = None
    if round_trip:
        return_waypoints = list(reversed(waypoints))
        return_route = Route(return_waypoints, speed_kn)
        print(f"  Return:   {return_route.total_distance_nm:.0f} nm, "
              f"{return_route.total_time_hours:.1f} hours")
        for i, leg in enumerate(return_route.legs):
            print(f"    Leg {i + 1}: {leg['from'].name} -> {leg['to'].name}: "
                  f"{leg['dist_nm']:.0f} nm, bearing {leg['bearing_deg']:.0f} deg")

    factory = FactoryCombinator(engine, prop,
                                sg_allowance_kw=sg_factory_allowance_kw,
                                engine_rpm_min=engine_rpm_min_sg,
                                engine_rpm_max=engine_rpm_max_sg)
    print(f"  Factory combinator: {len(factory._combo_lever)} schedule points")
    if sg_factory_allowance_kw > 0:
        print(f"    SG allowance: {sg_factory_allowance_kw:.0f} kW")
    print(f"    RPM range: {factory._combo_rpm[0]:.1f} - {factory._combo_rpm[-1]:.1f} shaft")
    print(f"    Pitch range: {factory._combo_pitch[0]:.3f} - {factory._combo_pitch[-1]:.3f}")

    print("\nLoading NORA3 weather data ...")
    # Weather is loaded per-worker for parallel execution (xarray can't be pickled)
    # We just verify the data directory exists here.
    if not data_dir.exists():
        raise FileNotFoundError(f"NORA3 data directory not found: {data_dir}")

    # --- Pre-compute operating point caches ---
    w = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_WAKE))
    Vs = speed_kn * KN_TO_MS
    Va = Vs * (1.0 - w)

    print("\nPre-computing operating point caches ...")
    opt_cache = build_optimiser_cache(prop, engine, Va,
                                       auxiliary_power_kw=sg_load_kw,
                                       engine_rpm_min=engine_rpm_min_sg,
                                       engine_rpm_max=engine_rpm_max_sg)
    factory_cache = build_factory_cache(factory, Va)

    # --- Run voyages ---
    n_days = _days_in_year(year)
    n_workers = min(os.cpu_count() or 1, 16)
    voy_label = "round-trip voyages" if round_trip else "one-way voyages"
    print(f"\nRunning {n_days} {voy_label} using {n_workers} workers ...\n")

    departures = [
        datetime(year, 1, 1, 6, 0, 0, tzinfo=timezone.utc) + timedelta(days=day)
        for day in range(n_days)
    ]

    # Split departures into batches (one per worker)
    batch_size = max(1, (n_days + n_workers - 1) // n_workers)
    batches = []
    for i in range(0, n_days, batch_size):
        batch_deps = departures[i:i + batch_size]
        batches.append((
            batch_deps, route, data_dir, drift_tf, flettner,
            factory, prop, engine, speed_kn, opt_cache, factory_cache,
            return_route, hull_ks_m, blade_ks_m,
        ))

    results = []
    errors = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_voyage_worker, b) for b in batches]
        for f in as_completed(futures):
            for item in f.result():
                if isinstance(item, VoyageResult):
                    results.append(item)
                else:
                    dep, err = item
                    errors.append((dep, err))
                    print(f"  {dep.strftime('%Y-%m-%d')}: ERROR: {err}")

    # Sort by departure date
    results.sort(key=lambda r: r.departure)

    if verbose:
        for vr in results:
            inf_str = ""
            if vr.n_hours_factory_infeasible > 0:
                inf_str = f", factory_infeasible={vr.n_hours_factory_infeasible}/{vr.n_hours_total}h"
            baseline = vr.total_fuel_factory_no_flettner_kg if vr.total_fuel_factory_no_flettner_kg > 0 else vr.total_fuel_factory_kg
            pct_pr = (100.0 * vr.saving_pitch_rpm_kg / baseline) if baseline > 0 else 0.0
            pct_fl = (100.0 * vr.saving_flettner_kg / baseline) if baseline > 0 else 0.0
            print(f"  {vr.departure.strftime('%Y-%m-%d')}: "
                  f"save={vr.saving_pct:+.1f}% "
                  f"(pitch/RPM {pct_pr:+.1f}%, Flettner {pct_fl:+.1f}%)"
                  f"{inf_str}")

    if errors:
        print(f"\n  {len(errors)} voyages failed")

    return results


def _summarise_for_speed(
    results: list[VoyageResult],
    speed_kn: float,
    idle_pct: float,
) -> SpeedSweepResult:
    """Compute aggregated statistics for one speed."""
    transit_h = results[0].total_hours
    sailing_hours_year = 365.25 * 24.0 * (1.0 - idle_pct / 100.0)
    vpyr = sailing_hours_year / transit_h

    fac_nf = np.array([r.total_fuel_factory_no_flettner_kg for r in results])
    fac_fl = np.array([r.total_fuel_factory_kg for r in results])
    opt_nf = np.array([r.total_fuel_opt_no_flettner_kg for r in results])
    opt_fl = np.array([r.total_fuel_optimised_kg for r in results])
    sav_pr = np.array([r.saving_pitch_rpm_kg for r in results])
    sav_fl = np.array([r.saving_flettner_kg for r in results])
    sav_tot = sav_pr + sav_fl
    sav_pct = np.array([r.saving_pct for r in results])

    m_fac_nf = float(np.mean(fac_nf))
    m_fac_fl = float(np.mean(fac_fl))
    m_opt_nf = float(np.mean(opt_nf))
    m_opt_fl = float(np.mean(opt_fl))
    m_sav_pr = float(np.mean(sav_pr))
    m_sav_fl = float(np.mean(sav_fl))
    m_sav_tot = float(np.mean(sav_tot))
    m_sav_pct = float(np.mean(sav_pct))

    pct_pr = 100.0 * m_sav_pr / m_fac_nf if m_fac_nf > 0 else 0.0
    pct_fl = 100.0 * m_sav_fl / m_fac_nf if m_fac_nf > 0 else 0.0

    total_hrs = sum(r.n_hours_total for r in results)
    total_inf = sum(r.n_hours_factory_infeasible for r in results)

    return SpeedSweepResult(
        speed_kn=speed_kn,
        transit_hours=transit_h,
        voyages_per_year=vpyr,
        n_voyages=len(results),
        mean_fuel_factory_nf_kg=m_fac_nf,
        mean_fuel_factory_fl_kg=m_fac_fl,
        mean_fuel_opt_nf_kg=m_opt_nf,
        mean_fuel_opt_fl_kg=m_opt_fl,
        mean_saving_pitch_rpm_kg=m_sav_pr,
        mean_saving_flettner_kg=m_sav_fl,
        mean_saving_total_kg=m_sav_tot,
        mean_saving_pct=m_sav_pct,
        pct_pitch_rpm=pct_pr,
        pct_flettner=pct_fl,
        ann_fuel_factory_nf_t=m_fac_nf * vpyr / 1000.0,
        ann_fuel_opt_fl_t=m_opt_fl * vpyr / 1000.0,
        ann_saving_pitch_rpm_t=m_sav_pr * vpyr / 1000.0,
        ann_saving_flettner_t=m_sav_fl * vpyr / 1000.0,
        ann_saving_total_t=m_sav_tot * vpyr / 1000.0,
        mean_hs=float(np.mean([r.mean_hs for r in results])),
        mean_wind=float(np.mean([r.mean_wind for r in results])),
        mean_R_aw_kN=float(np.mean([r.mean_R_aw_kN for r in results])),
        mean_R_wind_kN=float(np.mean([r.mean_R_wind_kN for r in results])),
        mean_F_flettner_kN=float(np.mean([r.mean_F_flettner_kN for r in results])),
        mean_rotor_power_kW=float(np.mean([r.mean_rotor_power_kW for r in results])),
        hull_ks_um=results[0].hull_ks_um,
        blade_ks_um=results[0].blade_ks_um,
        R_roughness_kN=results[0].R_roughness_kN,
        blade_fuel_factor=results[0].blade_fuel_factor,
        pct_factory_infeasible=100.0 * total_inf / total_hrs if total_hrs else 0.0,
    )


def run_speed_sweep(
    speeds: list[float],
    year: int = 2024,
    idle_pct: float = 15.0,
    data_dir: Path | None = None,
    pdstrip_path: str = None,
    flettner_enabled: bool = True,
    verbose: bool = False,
    sg_load_kw: float = 0.0,
    sg_factory_allowance_kw: float = 0.0,
    sg_freq_min: float = 0.0,
    sg_freq_max: float = 0.0,
    round_trip: bool = True,
    hull_ks_m: float = 0.0,
    blade_ks_m: float = 0.0,
) -> list[SpeedSweepResult]:
    """Run annual comparisons at multiple speeds and return summary per speed.

    Parameters
    ----------
    speeds : list of float
        Transit speeds in knots to evaluate.
    idle_pct : float
        Percentage of year idle, for annualization.
    round_trip : bool
        If True, each departure is a round trip (outbound + return).

    Returns
    -------
    list of SpeedSweepResult, one per speed (same order as input).
    """
    if pdstrip_path is None:
        pdstrip_path = PDSTRIP_DAT

    sweep_results = []
    for i, spd in enumerate(speeds):
        print(f"\n{'#' * 78}")
        print(f"# SPEED SWEEP: {spd:.1f} kn  ({i + 1}/{len(speeds)})")
        print(f"{'#' * 78}\n")

        results = run_annual_comparison(
            year=year,
            speed_kn=spd,
            data_dir=data_dir,
            pdstrip_path=pdstrip_path,
            flettner_enabled=flettner_enabled,
            verbose=verbose,
            sg_load_kw=sg_load_kw,
            sg_factory_allowance_kw=sg_factory_allowance_kw,
            sg_freq_min=sg_freq_min,
            sg_freq_max=sg_freq_max,
            round_trip=round_trip,
            hull_ks_m=hull_ks_m,
            blade_ks_m=blade_ks_m,
        )

        if not results:
            print(f"  WARNING: No valid voyages at {spd:.1f} kn -- skipping")
            continue

        sr = _summarise_for_speed(results, spd, idle_pct)
        sweep_results.append(sr)

        # Print brief inline summary
        print(f"\n  {spd:.0f} kn summary: {sr.n_voyages} voyages, "
              f"fac_NF={sr.mean_fuel_factory_nf_kg:.0f} kg/voy, "
              f"opt+Fl={sr.mean_fuel_opt_fl_kg:.0f} kg/voy, "
              f"saving={sr.mean_saving_pct:.1f}% "
              f"(P/R {sr.pct_pitch_rpm:.1f}% + Fl {sr.pct_flettner:.1f}%)")

    return sweep_results


def run_scheduling_analysis(
    speeds: list[float],
    year: int = 2024,
    data_dir: Path | None = None,
    pdstrip_path: str | None = None,
    flettner_enabled: bool = True,
    round_trip: bool = True,
    hull_ks_m: float = 0.0,
    blade_ks_m: float = 0.0,
    sg_load_kw: float = 0.0,
    sg_factory_allowance_kw: float = 0.0,
    sg_freq_min: float = 0.0,
    sg_freq_max: float = 0.0,
) -> dict[float, list[VoyageResult]]:
    """Run annual comparisons at multiple speeds, return per-speed results.

    Returns
    -------
    dict mapping speed_kn -> list[VoyageResult] (sorted by departure date).
    """
    if pdstrip_path is None:
        pdstrip_path = PDSTRIP_DAT

    all_results: dict[float, list[VoyageResult]] = {}
    for i, spd in enumerate(speeds):
        print(f"\n{'#' * 78}")
        print(f"# SCHEDULING ANALYSIS: {spd:.1f} kn  ({i + 1}/{len(speeds)})")
        print(f"{'#' * 78}\n")
        results = run_annual_comparison(
            year=year, speed_kn=spd, data_dir=data_dir,
            pdstrip_path=pdstrip_path, flettner_enabled=flettner_enabled,
            verbose=False, round_trip=round_trip,
            hull_ks_m=hull_ks_m, blade_ks_m=blade_ks_m,
            sg_load_kw=sg_load_kw,
            sg_factory_allowance_kw=sg_factory_allowance_kw,
            sg_freq_min=sg_freq_min, sg_freq_max=sg_freq_max,
        )
        if results:
            all_results[spd] = results
    return all_results
