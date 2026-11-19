"""Speed strategies with ocean current, Rotterdam <-> Gothenburg, 2024.

Compares constant SOG (speed control), constant STW, and current/weather
optimised speed profiles with the same arrival time.  See
simulation/current_study.py.

Usage:
    python current_study.py [--speed 10] [--year 2024] [--plot]

Requires:
    data/nora3_route/nora3_wave_YYYYMM.nc   (voyage_comparison.py --download)
    data/cmems_nws/cmems_nws_uv_YYYYMM.nc   (environment/fetch_cmems_nws_currents.py)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from models.constants import (DATA_PATH_C440, DATA_PATH_C455, DATA_PATH_C470, KN_TO_MS, NORA3_DATA_DIR,
                              PDSTRIP_DAT, PROP_BAR, PROP_DESIGN_PITCH, PROP_DIAMETER, RHO_WATER)
from models.drift_force import DriftTransferFunction
from models.route import ROUTE_GOTHENBURG_ROTTERDAM, ROUTE_ROTTERDAM_GOTHENBURG
from optimiser import make_man_l27_38
from propeller_model import CSeriesPropeller, load_c_series_data
from simulation.current_study import CAP_FACTORS, CAP_STRATEGIES, STRATEGIES, FuelModel, build_fuel_table, run_current_study

CMEMS_DIR = Path(__file__).parent / "data" / "cmems_nws"
LABELS = {
    "nocur": "No current (reference)",
    "sog": "Constant SOG (speed control)",
    "stw": "Constant STW",
    "opt_cur": "Optimised, current known",
    "opt_all": "Optimised, current + weather known",
}


def summarise(name, res):
    ok = [r for r in res if r.feasible]
    tot = {s: sum(r.fuel_kg[s] for r in ok) for s in STRATEGIES}
    base = tot["sog"]
    print(f"\n{name}: {len(ok)} of {len(res)} voyages feasible, "
          f"mean |along-track current| {np.mean([r.mean_abs_current_kn for r in ok]):.2f} kn")
    print(f"  {'Strategy':38s} {'Fuel [t]':>9s} {'vs SOG':>8s} {'per-voyage P10/P50/P90 vs SOG':>32s}")
    for s in STRATEGIES:
        rel = np.array([100 * (r.fuel_kg[s] / r.fuel_kg["sog"] - 1) for r in ok])
        print(f"  {LABELS[s]:38s} {tot[s] / 1000:9.1f} {100 * (tot[s] / base - 1):+7.2f}%"
              f"   {np.percentile(rel, 10):+6.2f} {np.percentile(rel, 50):+6.2f} {np.percentile(rel, 90):+6.2f} %")
    print(f"  {'Fuel cap (x calm fuel rate at nominal speed)':44s} {'vs SOG':>8s} {'P10/P50/P90':>20s} {'cap binds':>10s} {'cap too tight':>14s}")
    for s, k in zip(CAP_STRATEGIES, CAP_FACTORS):
        good = [r for r in ok if np.isfinite(r.fuel_kg[s])]
        rel = np.array([100 * (r.fuel_kg[s] / r.fuel_kg["sog"] - 1) for r in good])
        tot_s = sum(r.fuel_kg[s] for r in good) / sum(r.fuel_kg["sog"] for r in good)
        bind = np.mean([r.cap_bound_frac[s] for r in good])
        print(f"  {f'cap {k:.1f}x':44s} {100 * (tot_s - 1):+7.2f}%  {np.percentile(rel, 10):+5.2f} {np.percentile(rel, 50):+5.2f} {np.percentile(rel, 90):+5.2f} %"
              f" {100 * bind:9.1f}% {len(ok) - len(good):9d} voy")
    dt = max(abs(r.hours[s] - r.hours["nocur"]) for r in ok for s in STRATEGIES)
    print(f"  max arrival-time deviation before correction to schedule: {dt * 60:.1f} min")
    return ok


def plot(results, speed_kn, out="current_study.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (name, ok) in zip(axs, results.items()):
        days = [r.departure for r in ok]
        for s, c in (("stw", "C1"), ("opt_cur", "C2"), ("opt_all", "C3"), ("nocur", "grey")):
            rel = [100 * (r.fuel_kg[s] / r.fuel_kg["sog"] - 1) for r in ok]
            ax.plot(days, rel, ".", ms=3, color=c, label=LABELS[s])
        ax.axhline(0, color="C0", lw=1.5, label=LABELS["sog"])
        ax.set_title(f"{name}, {speed_kn} kn", fontsize=10)
        ax.set_ylabel("Voyage fuel relative to constant SOG [%]")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, loc="lower left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nSaved {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--speed", type=float, default=10.0)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--seg-nm", type=float, default=5.0)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--flettner", action="store_true",
                   help="Include the Flettner rotor (H=28 m, D=4 m, spin ratio 3) as in the annual runs.")
    a = p.parse_args()

    bd = {0.40: load_c_series_data(DATA_PATH_C440), 0.55: load_c_series_data(DATA_PATH_C455),
          0.70: load_c_series_data(DATA_PATH_C470)}
    prop = CSeriesPropeller(bd, design_pitch=PROP_DESIGN_PITCH, diameter=PROP_DIAMETER, area_ratio=PROP_BAR,
                            rho=RHO_WATER)
    engine = make_man_l27_38()
    print("Building fuel table fuel(STW, thrust) ...")
    cache = Path(__file__).parent / ".cache_current_fuel_table.npy"
    if cache.exists():
        table = np.load(cache)
    else:
        table = build_fuel_table(prop, engine)
        np.save(cache, table)
    fuel_model = FuelModel(table)
    drift_tf = DriftTransferFunction(PDSTRIP_DAT, speed_ms=a.speed * KN_TO_MS)
    flettner = None
    if a.flettner:
        from models.flettner import FlettnerRotor
        flettner = FlettnerRotor(height=28.0, diameter=4.0, max_rpm=220.0, endplate_ratio=2.0, roughness_factor=3.0)
        flettner.set_target_spin_ratio(3.0)
        print("Flettner rotor included")

    results = {}
    for name, wps in (("Rotterdam -> Gothenburg", ROUTE_ROTTERDAM_GOTHENBURG),
                      ("Gothenburg -> Rotterdam", ROUTE_GOTHENBURG_ROTTERDAM)):
        print(f"\n{name} ...")
        res = run_current_study(wps, a.year, a.speed, CMEMS_DIR, NORA3_DATA_DIR, drift_tf, fuel_model, a.seg_nm,
                                flettner=flettner)
        results[name] = summarise(name, res)
    if a.plot:
        plot(results, a.speed, "current_study_flettner.png" if a.flettner else "current_study.png")


if __name__ == "__main__":
    main()
