"""Speed strategy study with ocean current along the route.

The annual voyage engine (simulation/engine.py) assumes a constant speed
through water.  With current the speed through water (STW) varies along
the route even when the speed over ground (SOG) is held, so this module
evaluates voyages segment by segment with a 2-D fuel table
fuel(STW, thrust) from the optimiser, and compares speed strategies that
all arrive at the same time:

  sog      constant speed over ground (the onboard speed controller)
  stw      constant speed through water (~ constant power / fuel cap in calm water)
  opt_cur  speed profile optimised with knowledge of the current only
           (calm-water fuel model), evaluated in the actual weather
  opt_all  speed profile optimised with knowledge of current and weather
  nocur    reference: same voyage without current (constant STW = SOG)

Current: CMEMS NWS reanalysis (environment/fetch_cmems_nws_currents.py),
hourly surface u/v.  Only the along-track component is used; the
cross-track component (drift angle) is ignored.

Approximations:
- Weather and current at each segment are taken at the nominal schedule
  time (constant SOG).  Arrival-time differences between strategies at a
  segment are typically tens of minutes, small relative to the 12.4 h
  tidal period.
- Wave added resistance from the PdStrip transfer function at nominal speed.
- Hull resistance below the lowest hull-data speed is scaled with V^2.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from models.constants import (
    GEAR_RATIO,
    HULL_ETA_R,
    HULL_RESISTANCE_KN,
    HULL_SPEEDS_KN,
    HULL_T_DEDUCTION,
    HULL_WAKE,
    KN_TO_MS,
    SHAFT_EFF,
)
from models.route import Route, Waypoint
from models.wind_resistance import wind_resistance_kN

STRATEGIES = ("nocur", "sog", "stw", "opt_cur", "opt_all")

# Fuel cap levels, relative to the calm-water fuel rate at the nominal speed.
CAP_FACTORS = (1.1, 1.2, 1.3, 1.5)
CAP_STRATEGIES = tuple(f"cap{int(round(100 * k))}" for k in CAP_FACTORS)

# Speed-through-water grid for the fuel table and the profile optimisation.
STW_TABLE_KN = np.arange(5.0, 15.01, 0.5)
STW_FINE_KN = np.arange(5.0, 15.001, 0.05)
THRUST_TABLE_KN = np.arange(5.0, 600.01, 5.0)


# ============================================================
# Hull and fuel models
# ============================================================

def calm_resistance_kN(stw_kn: np.ndarray) -> np.ndarray:
    """Calm-water resistance [kN]; V^2 scaling below the lowest data speed."""
    v = np.asarray(stw_kn, dtype=float)
    r = np.interp(v, HULL_SPEEDS_KN, HULL_RESISTANCE_KN)
    lo = v < HULL_SPEEDS_KN[0]
    r[lo] = HULL_RESISTANCE_KN[0] * (v[lo] / HULL_SPEEDS_KN[0]) ** 2
    return r


def _fuel_row(args):
    """Optimiser fuel rate [g/h] over the thrust grid at one STW."""
    from optimiser import find_min_fuel_operating_point
    stw, prop, engine, aux_kw = args
    w = float(np.interp(stw, HULL_SPEEDS_KN, HULL_WAKE))
    eta_r = float(np.interp(stw, HULL_SPEEDS_KN, HULL_ETA_R))
    va = stw * KN_TO_MS * (1.0 - w)
    row = np.full(len(THRUST_TABLE_KN), np.nan)
    for j, t in enumerate(THRUST_TABLE_KN):
        op = find_min_fuel_operating_point(prop, va, t * 1000.0, engine, gear_ratio=GEAR_RATIO,
                                           shaft_efficiency=SHAFT_EFF, auxiliary_power_kw=aux_kw,
                                           pitch_step=0.02, eta_R=eta_r)
        if op.found and op.fuel_rate is not None:
            row[j] = op.fuel_rate
    return row


def build_fuel_table(prop, engine, aux_kw: float = 0.0) -> np.ndarray:
    """fuel[g/h] on STW_TABLE_KN x THRUST_TABLE_KN (NaN where infeasible)."""
    args = [(v, prop, engine, aux_kw) for v in STW_TABLE_KN]
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, 16)) as ex:
        rows = list(ex.map(_fuel_row, args))
    return np.vstack(rows)


class FuelModel:
    """Fuel rate [g/h] as a function of STW and environmental resistance."""

    def __init__(self, table: np.ndarray):
        self.table = table
        # Pre-interpolate onto the fine STW grid (linear in STW).
        self.fine = np.vstack([np.interp(STW_FINE_KN, STW_TABLE_KN, table[:, j])
                               for j in range(table.shape[1])]).T
        self.t_ded = np.interp(STW_FINE_KN, HULL_SPEEDS_KN, HULL_T_DEDUCTION)
        self.r_calm = calm_resistance_kN(STW_FINE_KN)

    def fuel_on_grid(self, r_env_kN: np.ndarray) -> np.ndarray:
        """Fuel [g/h] on STW_FINE_KN for extra resistance r_env_kN(STW) [kN]."""
        thrust = (self.r_calm + r_env_kN) / (1.0 - self.t_ded)
        x = (np.clip(thrust, THRUST_TABLE_KN[0], THRUST_TABLE_KN[-1]) - THRUST_TABLE_KN[0]) / (
            THRUST_TABLE_KN[1] - THRUST_TABLE_KN[0])
        j = np.minimum(x.astype(int), len(THRUST_TABLE_KN) - 2)
        f = x - j
        i = np.arange(len(STW_FINE_KN))
        out = (1 - f) * self.fine[i, j] + f * self.fine[i, j + 1]
        out[thrust > THRUST_TABLE_KN[-1]] = np.nan
        return out


# ============================================================
# Environment sampled along the route
# ============================================================

@dataclass
class Segment:
    lat: float
    lon: float
    heading_deg: float
    dist_nm: float
    cum_mid_nm: float


def route_segments(waypoints: list[Waypoint], seg_nm: float = 5.0) -> list[Segment]:
    route = Route(waypoints, 10.0)
    n = max(1, int(round(route.total_distance_nm / seg_nm)))
    d = route.total_distance_nm / n
    segs = []
    for k in range(n):
        mid = (k + 0.5) * d
        lat, lon, hdg = route._position_at_distance(mid)
        segs.append(Segment(lat, lon, hdg, d, mid))
    return segs


def _nearest_valid(lat2d, lon2d, valid, lat, lon):
    d2 = (lat2d - lat) ** 2 + ((lon2d - lon) * np.cos(np.radians(lat))) ** 2
    d2 = np.where(valid, d2, np.inf)
    return np.unravel_index(np.argmin(d2), d2.shape)


def sample_currents(segs: list[Segment], cmems_dir: Path, year: int) -> tuple[np.ndarray, np.ndarray]:
    """Hourly along-track current [kn] per segment, shape (n_seg, n_hours), and hour axis."""
    import xarray as xr
    ds = xr.open_mfdataset(sorted(cmems_dir.glob(f"cmems_nws_uv_{year}*.nc")))
    lat2d, lon2d = np.meshgrid(ds.latitude.values, ds.longitude.values, indexing="ij")
    valid = np.isfinite(ds.uo.isel(time=0).values)
    iy, ix = zip(*[_nearest_valid(lat2d, lon2d, valid, s.lat, s.lon) for s in segs])
    u = ds.uo.values[:, list(iy), list(ix)]  # (time, seg)
    v = ds.vo.values[:, list(iy), list(ix)]
    hdg = np.radians([s.heading_deg for s in segs])
    along = (u * np.sin(hdg) + v * np.cos(hdg)) / KN_TO_MS
    times = ds.time.values.astype("datetime64[h]")
    ds.close()
    return np.nan_to_num(along.T), times


def sample_weather(segs: list[Segment], nora3_dir: Path, year: int) -> tuple[dict, np.ndarray]:
    """Hourly hs, tp, wave_dir, wind_speed, wind_dir per segment, shape (n_seg, n_hours)."""
    import xarray as xr
    out = {k: [] for k in ("hs", "tp", "thq", "ff", "dd")}
    times = []
    idx = None
    for m in range(1, 13):
        ds = xr.open_dataset(nora3_dir / f"nora3_wave_{year}{m:02d}.nc")
        name = "latitude" if "latitude" in ds else "lat"
        lat2d = ds[name].values
        lon2d = ds["longitude" if name == "latitude" else "lon"].values
        dims = ds[name].dims
        if idx is None:
            valid = np.isfinite(ds["hs"].isel(time=0).values)
            idx = [_nearest_valid(lat2d, lon2d, valid, s.lat, s.lon) for s in segs]
        sel = {dims[0]: xr.DataArray([i[0] for i in idx], dims="seg"),
               dims[1]: xr.DataArray([i[1] for i in idx], dims="seg")}
        sub = ds[list(out)].isel(**sel).load()
        for k in out:
            out[k].append(sub[k].transpose("seg", "time").values)
        times.append(ds.time.values.astype("datetime64[h]"))
        ds.close()
    return {k: np.nan_to_num(np.concatenate(v, axis=1)) for k, v in out.items()}, np.concatenate(times)


# ============================================================
# Voyage evaluation
# ============================================================

@dataclass
class VoyageStrategies:
    departure: datetime
    fuel_kg: dict = field(default_factory=dict)        # strategy -> fuel [kg]
    hours: dict = field(default_factory=dict)          # strategy -> time [h]
    stw_range: dict = field(default_factory=dict)      # strategy -> (min, max) STW [kn]
    mean_abs_current_kn: float = 0.0
    cap_bound_frac: dict = field(default_factory=dict)  # cap strategy -> fraction of distance where cap binds
    feasible: bool = True


def _flettner_terms(flettner, v_kn, heading, wind_speed, wind_dir):
    """Rotor surge force [kN] and rotor power [kW] at STW v_kn.  The rotor is
    spun only if that improves surge over the stopped (drag) rotor, as in
    simulation/engine.py."""
    vs = v_kn * KN_TO_MS
    flettner.compute_from_true_wind(Vs=vs, heading_deg=heading, wind_speed=wind_speed, wind_dir_deg=wind_dir)
    f_spin, p_spin = flettner.surge_force_kN, flettner.rotor_power_kW
    f_stop = flettner.stopped_surge_from_true_wind(Vs=vs, heading_deg=heading, wind_speed=wind_speed,
                                                   wind_dir_deg=wind_dir)
    return (f_spin, p_spin) if f_spin > f_stop else (f_stop, 0.0)


def _segment_fuel_grids(segs, k_hours, current, wx, drift_tf, fuel_model, speed_kn, flettner=None):
    """Fuel [g/h] on STW_FINE_KN per segment, with and without waves/wind.

    With a Flettner rotor, its surge force reduces the resistance and its
    motor power is added as genset fuel (GENSET_SFOC).  The calm-water grid
    (used by the current-only optimisation) has neither wind nor rotor."""
    from models.constants import GENSET_SFOC
    n = len(segs)
    g_all = np.empty((n, len(STW_FINE_KN)))
    g_calm = np.empty_like(g_all)
    vgrid = np.array([STW_FINE_KN[0], 7.5, speed_kn, 12.5, STW_FINE_KN[-1]])
    zeros = np.zeros_like(STW_FINE_KN)
    for i, s in enumerate(segs):
        k = k_hours[i]
        ws, wd = wx["ff"][i, k], wx["dd"][i, k]
        r_aw = drift_tf.added_resistance(Hs=wx["hs"][i, k], Tp=wx["tp"][i, k],
                                         wave_dir_deg=wx["thq"][i, k], heading_deg=s.heading_deg)
        r_w = [wind_resistance_kN(Vs=v * KN_TO_MS, heading_deg=s.heading_deg, wind_speed=ws, wind_dir_deg=wd)
               for v in vgrid]
        r_env = r_aw + np.interp(STW_FINE_KN, vgrid, r_w)
        rotor_fuel = zeros
        if flettner is not None:
            fl = [_flettner_terms(flettner, v, s.heading_deg, ws, wd) for v in vgrid]
            r_env = r_env - np.interp(STW_FINE_KN, vgrid, [f for f, _ in fl])
            rotor_fuel = np.interp(STW_FINE_KN, vgrid, [p for _, p in fl]) * GENSET_SFOC
        g_all[i] = fuel_model.fuel_on_grid(r_env) + rotor_fuel
        g_calm[i] = fuel_model.fuel_on_grid(zeros)
    return g_all, g_calm


def _optimal_profile(grid, c, d, t_total):
    """Per-segment STW index minimising fuel for fixed total time (Lagrange multiplier bisection)."""
    sog = STW_FINE_KN[None, :] + c[:, None]
    ok = (sog > 0.5) & np.isfinite(grid)
    time = np.where(ok, d[:, None] / np.where(ok, sog, 1.0), np.inf)
    fuel = np.where(ok, grid * time, np.inf)

    def pick(lam):
        j = np.argmin(np.where(ok, fuel + lam * np.where(ok, time, 0.0), np.inf), axis=1)
        return j, time[np.arange(len(c)), j].sum()

    lo, hi = 0.0, 1e7
    j, t = pick(lo)
    if t <= t_total:
        return j
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        j, t = pick(mid)
        if t > t_total:
            lo = mid
        else:
            hi = mid
    return pick(hi)[0]


def evaluate_voyage_strategies(departure, segs, cur, cur_t0, wx, wx_t0, drift_tf, fuel_model,
                               speed_kn, flettner=None) -> VoyageStrategies:
    d = np.array([s.dist_nm for s in segs])
    t_total = d.sum() / speed_kn
    t_mid = np.array([s.cum_mid_nm for s in segs]) / speed_kn
    dep64 = np.datetime64(departure.replace(tzinfo=None), "h")
    k_cur = ((dep64 - cur_t0).astype(int) + np.round(t_mid).astype(int))
    k_wx = ((dep64 - wx_t0).astype(int) + np.round(t_mid).astype(int))
    c = cur[np.arange(len(segs)), k_cur]
    g_all, g_calm = _segment_fuel_grids(segs, k_wx, cur, wx, drift_tf, fuel_model, speed_kn, flettner)

    def evaluate(v, cc):
        """Fuel [kg] and time [h] for STW profile v [kn].  STW is interpolated on
        the fine grid, and fuel is corrected to the exact scheduled time with
        fuel ~ 1/t^2 (fuel rate ~ V^3), removing grid-quantisation bias."""
        x = np.clip((v - STW_FINE_KN[0]) / 0.05, 0, len(STW_FINE_KN) - 1.000001)
        i0 = x.astype(int)
        fr = x - i0
        rows = np.arange(len(segs))
        f = (1 - fr) * g_all[rows, i0] + fr * g_all[rows, i0 + 1]
        tseg = d / (v + cc)
        t = float(tseg.sum())
        fuel = float(np.sum(f * tseg) / 1000.0) * (t / t_total) ** 2
        return fuel, t, (float(v.min()), float(v.max()))

    res = VoyageStrategies(departure=departure, mean_abs_current_kn=float(np.mean(np.abs(c))))
    v_nocur = np.full(len(segs), speed_kn)
    v_sog = speed_kn - c
    lo, hi = 1.0, 25.0  # constant STW giving the scheduled time with current
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if np.sum(d / np.maximum(m + c, 0.1)) > t_total:
            lo = m
        else:
            hi = m
    v_stw = np.full(len(segs), 0.5 * (lo + hi))
    v_cur = STW_FINE_KN[_optimal_profile(g_calm, c, d, t_total)]
    v_all = STW_FINE_KN[_optimal_profile(g_all, c, d, t_total)]

    # Fuel cap: speed control (constant SOG setpoint S) with STW limited where the
    # fuel rate would exceed the cap; S raised uniformly to keep the arrival time.
    f_nom = float(np.interp(speed_kn, STW_FINE_KN, g_calm[0]))
    caps = {}
    for name, k in zip(CAP_STRATEGIES, CAP_FACTORS):
        cap = k * f_nom
        over = ~(g_all <= cap)  # NaN (infeasible) counts as over
        idx = np.where(over.all(axis=1), 0, np.argmax(over, axis=1) - 1)
        idx = np.where(~over.any(axis=1), len(STW_FINE_KN) - 1, idx)
        v_cap = STW_FINE_KN[np.clip(idx, 0, None)]
        lo, hi = speed_kn, speed_kn + 10.0
        for _ in range(60):
            m = 0.5 * (lo + hi)
            v = np.minimum(m - c, v_cap)
            if np.sum(d / np.maximum(v + c, 0.1)) > t_total:
                lo = m
            else:
                hi = m
        v = np.minimum(hi - c, v_cap)
        if np.sum(d / np.maximum(v + c, 0.1)) > t_total * 1.001:
            v = np.full(len(segs), np.nan)  # cap too tight to make the schedule
        caps[name] = v
        res.cap_bound_frac[name] = float(np.sum(d[v >= v_cap - 1e-9]) / d.sum()) if np.isfinite(v).all() else 1.0

    for name, v, cc in [(n, caps[n], c) for n in CAP_STRATEGIES] + [("nocur", v_nocur, np.zeros_like(c)), ("sog", v_sog, c), ("stw", v_stw, c),
                        ("opt_cur", v_cur, c), ("opt_all", v_all, c)]:
        if name in caps and not np.isfinite(v).all():
            res.fuel_kg[name], res.hours[name], res.stw_range[name] = np.nan, np.nan, (np.nan, np.nan)
            continue
        f, t, rng = evaluate(v, cc)
        res.fuel_kg[name], res.hours[name], res.stw_range[name] = f, t, rng
        if not np.isfinite(f) and name not in caps:
            res.feasible = False
    return res


def _worker(args):
    deps, segs, cur, cur_t0, wx, wx_t0, drift_tf, fuel_model, speed_kn, flettner = args
    return [evaluate_voyage_strategies(dep, segs, cur, cur_t0, wx, wx_t0, drift_tf, fuel_model, speed_kn, flettner)
            for dep in deps]


def run_current_study(waypoints, year, speed_kn, cmems_dir, nora3_dir, drift_tf, fuel_model,
                      seg_nm=5.0, verbose=True, flettner=None):
    """Daily departures over the year in the direction given by waypoints."""
    segs = route_segments(waypoints, seg_nm)
    if verbose:
        print(f"  {len(segs)} segments of {segs[0].dist_nm:.2f} nm")
    cur, cur_t = sample_currents(segs, cmems_dir, year)
    wx, wx_t = sample_weather(segs, nora3_dir, year)
    t_total = sum(s.dist_nm for s in segs) / speed_kn
    n_days = int((np.datetime64(f"{year + 1}-01-01") - np.datetime64(f"{year}-01-01")).astype(int))
    last_ok = min(cur_t[-1], wx_t[-1]) - np.timedelta64(int(np.ceil(t_total)) + 1, "h")
    deps = [datetime(year, 1, 1, 6, tzinfo=timezone.utc) + timedelta(days=k) for k in range(n_days)]
    deps = [d for d in deps if np.datetime64(d.replace(tzinfo=None), "h") <= last_ok]
    n_w = min(os.cpu_count() or 1, 16)
    batches = [deps[i::n_w] for i in range(n_w)]
    args = [(b, segs, cur, cur_t[0], wx, wx_t[0], drift_tf, fuel_model, speed_kn, flettner) for b in batches if b]
    with ProcessPoolExecutor(max_workers=n_w) as ex:
        results = [r for rs in ex.map(_worker, args) for r in rs]
    results.sort(key=lambda r: r.departure)
    return results
