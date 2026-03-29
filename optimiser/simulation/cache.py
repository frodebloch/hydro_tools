"""Pre-computed operating point caches for optimiser and factory combinator."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
from optimiser import find_min_fuel_operating_point

from models.constants import GEAR_RATIO, SHAFT_EFF


def _opt_cache_worker(args):
    """Worker function for parallel optimiser cache build."""
    T_kN_batch, prop, Va, engine, aux_kw, rpm_min, rpm_max = args
    results = {}
    for T_kN in T_kN_batch:
        T_N = T_kN * 1000.0
        op = find_min_fuel_operating_point(
            prop, Va, T_N, engine,
            gear_ratio=GEAR_RATIO,
            shaft_efficiency=SHAFT_EFF,
            auxiliary_power_kw=aux_kw,
            pitch_step=0.01,
            engine_rpm_min=rpm_min,
            engine_rpm_max=rpm_max,
        )
        if op.found and op.fuel_rate is not None:
            results[T_kN] = {
                "fuel_rate": op.fuel_rate,
                "pitch": op.pitch,
                "rpm": op.rpm,
            }
    return results


def build_optimiser_cache(
    prop,
    engine,
    Va: float,
    T_min_kN: float = 10.0,
    T_max_kN: float = 350.0,
    T_step_kN: float = 0.5,
    n_workers: int = 0,
    auxiliary_power_kw: float = 0.0,
    engine_rpm_min: Optional[float] = None,
    engine_rpm_max: Optional[float] = None,
) -> dict:
    """Pre-compute optimiser results for a range of thrust demands.

    Returns a dict mapping thrust [kN] (quantised to T_step_kN) to result
    dicts with fuel_rate, pitch, rpm.  This avoids running the expensive
    pitch sweep for every hourly evaluation.

    Uses multiprocessing when n_workers > 1.
    """
    T_values = np.arange(T_min_kN, T_max_kN + T_step_kN / 2, T_step_kN)
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 16)

    aux_label = f", aux={auxiliary_power_kw:.0f} kW" if auxiliary_power_kw > 0 else ""
    rpm_label = ""
    if engine_rpm_min is not None or engine_rpm_max is not None:
        lo = engine_rpm_min if engine_rpm_min is not None else engine.min_rpm()
        hi = engine_rpm_max if engine_rpm_max is not None else engine.max_rpm()
        rpm_label = f", RPM {lo:.0f}-{hi:.0f}"
    print(f"  Pre-computing optimiser for {len(T_values)} thrust points "
          f"({T_min_kN:.0f}-{T_max_kN:.0f} kN, step {T_step_kN} kN) "
          f"using {n_workers} workers{aux_label}{rpm_label} ...")

    if n_workers <= 1:
        # Sequential fallback
        cache = {}
        for T_kN in T_values:
            T_N = T_kN * 1000.0
            op = find_min_fuel_operating_point(
                prop, Va, T_N, engine,
                gear_ratio=GEAR_RATIO,
                shaft_efficiency=SHAFT_EFF,
                auxiliary_power_kw=auxiliary_power_kw,
                pitch_step=0.01,
                engine_rpm_min=engine_rpm_min,
                engine_rpm_max=engine_rpm_max,
            )
            if op.found and op.fuel_rate is not None:
                cache[T_kN] = {
                    "fuel_rate": op.fuel_rate,
                    "pitch": op.pitch,
                    "rpm": op.rpm,
                }
        print(f"  Cache built: {len(cache)} feasible / {len(T_values)} total")
        return cache

    # Split into batches for parallel execution
    batch_size = max(1, len(T_values) // n_workers)
    batches = []
    for i in range(0, len(T_values), batch_size):
        batch = T_values[i:i + batch_size].tolist()
        batches.append((batch, prop, Va, engine,
                        auxiliary_power_kw, engine_rpm_min, engine_rpm_max))

    cache = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_opt_cache_worker, b) for b in batches]
        for f in as_completed(futures):
            cache.update(f.result())

    print(f"  Cache built: {len(cache)} feasible / {len(T_values)} total")
    return cache


def build_factory_cache(
    factory,
    Va: float,
    T_min_kN: float = 10.0,
    T_max_kN: float = 350.0,
    T_step_kN: float = 0.5,
) -> dict:
    """Pre-compute factory combinator results for a range of thrust demands."""
    cache = {}
    T_values = np.arange(T_min_kN, T_max_kN + T_step_kN / 2, T_step_kN)

    for T_kN in T_values:
        T_N = T_kN * 1000.0
        fr = factory.evaluate(T_N, Va)
        if fr is not None:
            cache[T_kN] = fr

    print(f"  Factory cache: {len(cache)} feasible / {len(T_values)} total")
    return cache
