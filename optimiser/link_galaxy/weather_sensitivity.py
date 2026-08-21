"""Weather sensitivity analysis for Link Galaxy propulsion-optimiser savings.

Reruns the annual 2024 comparison, then correlates per-hour and per-voyage
savings against weather features (Hs, wind speed, added resistance,
thrust demand, relative wind angle) to characterise WHICH conditions
favour the optimiser.

Physical hypothesis (to test):
  The optimiser saves fuel by pulling the operating point off rated RPM
  down into the BSFC island. That is only possible when the propeller is
  UNDER-LOADED at rated RPM. Under heavy load (rough seas, big added
  resistance) the operating point is pushed toward MCR and the optimiser
  has less room to move -> smaller %-saving in bad weather.

Output:
  weather_sensitivity.png   — scatter + binned means
  Printed Pearson correlations and per-bin saving statistics.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONNOUSERSITE", "1")


def _install_link_galaxy_constants() -> None:
    import models
    import models.vessel_link_galaxy as lg
    sys.modules["models.constants"] = lg
    models.constants = lg
    stale = [name for name in list(sys.modules)
             if name.startswith("simulation.") or name in {
                 "models.combinator", "models.roughness",
                 "models.wind_resistance", "models.flettner"}]
    for name in stale:
        del sys.modules[name]


_install_link_galaxy_constants()

import numpy as np                              # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402
from models.route import ROUTE_LINK_GALAXY_ROUNDTRIP    # noqa: E402
from models import constants as _c                       # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--speed", type=float, default=12.5)
    p.add_argument("--as-commissioned",
                   default="/home/blofro/src/brucon/modules/config_link_galaxy/"
                           "gear_control_tables_4.prototxt.in")
    p.add_argument("--as-commissioned-block", default="harbor")
    p.add_argument("--out", default=str(Path(__file__).parent / "weather_sensitivity.png"))
    args = p.parse_args()

    from simulation.orchestrator import run_annual_comparison  # noqa: E402

    print(f"Running annual comparison ({args.year}, {args.speed} kn) ...")
    results = run_annual_comparison(
        year=args.year,
        speed_kn=args.speed,
        waypoints=ROUTE_LINK_GALAXY_ROUNDTRIP,
        data_dir=_c.NORA3_DATA_DIR,
        pdstrip_path=_c.PDSTRIP_DAT,
        flettner_enabled=False,
        verbose=False,
        round_trip=False,
        as_commissioned_prototxt=args.as_commissioned,
        as_commissioned_block=args.as_commissioned_block,
    )

    # --- Aggregate hourly data ---
    hs, ws, R_aw, R_wind, T_req = [], [], [], [], []
    fuel_fac, fuel_opt = [], []
    rpm_fac, rpm_opt = [], []
    pitch_fac, pitch_opt = [], []
    for vr in results:
        for h in vr.hourly:
            if h.factory_no_flettner_fuel_rate is None or h.optimised_no_flettner_fuel_rate is None:
                continue
            hs.append(h.hs)
            ws.append(h.wind_speed)
            R_aw.append(h.R_aw_kN)
            R_wind.append(h.R_wind_kN)
            T_req.append(h.T_required_no_flettner_kN)
            fuel_fac.append(h.factory_no_flettner_fuel_rate)
            fuel_opt.append(h.optimised_no_flettner_fuel_rate)
            rpm_fac.append(h.factory_rpm if h.factory_rpm is not None else np.nan)
            rpm_opt.append(h.optimised_rpm if h.optimised_rpm is not None else np.nan)
            pitch_fac.append(h.factory_pitch if h.factory_pitch is not None else np.nan)
            pitch_opt.append(h.optimised_pitch if h.optimised_pitch is not None else np.nan)

    hs = np.array(hs); ws = np.array(ws); R_aw = np.array(R_aw)
    R_wind = np.array(R_wind); T_req = np.array(T_req)
    fuel_fac = np.array(fuel_fac); fuel_opt = np.array(fuel_opt)
    rpm_fac = np.array(rpm_fac); rpm_opt = np.array(rpm_opt)
    pitch_fac = np.array(pitch_fac); pitch_opt = np.array(pitch_opt)
    saving_pct = 100.0 * (fuel_fac - fuel_opt) / fuel_fac

    print(f"\n{len(hs)} hourly samples across {len(results)} voyages")
    print(f"Overall mean hourly saving: {saving_pct.mean():.2f} %  "
          f"(median {np.median(saving_pct):.2f}, "
          f"P10 {np.percentile(saving_pct, 10):.2f}, "
          f"P90 {np.percentile(saving_pct, 90):.2f})")

    # --- Pearson correlations ---
    print("\nPearson correlation of hourly saving [%] with weather / load:")
    for name, x in [("Hs [m]", hs), ("Wind speed [m/s]", ws),
                    ("Added resistance R_aw [kN]", R_aw),
                    ("Wind resistance R_wind [kN]", R_wind),
                    ("Thrust demand T_req [kN]", T_req)]:
        r = np.corrcoef(x, saving_pct)[0, 1]
        print(f"  {name:<32s}  r = {r:+.3f}")

    # --- Binned means (Hs and T_req are the two big drivers) ---
    print("\nSaving [%] vs Hs bins:")
    hs_bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    for lo, hi in zip(hs_bins[:-1], hs_bins[1:]):
        mask = (hs >= lo) & (hs < hi)
        if mask.sum() > 20:
            print(f"  Hs {lo:.1f}-{hi:.1f} m  ({mask.sum():5d} h):  "
                  f"saving {saving_pct[mask].mean():.2f} %  "
                  f"(T_req mean {T_req[mask].mean():.0f} kN)")

    print("\nSaving [%] vs T_req bins:")
    t_bins = [50, 100, 130, 160, 190, 220, 260, 320]
    for lo, hi in zip(t_bins[:-1], t_bins[1:]):
        mask = (T_req >= lo) & (T_req < hi)
        if mask.sum() > 20:
            print(f"  T_req {lo:3d}-{hi:3d} kN ({mask.sum():5d} h):  "
                  f"saving {saving_pct[mask].mean():.2f} %  "
                  f"(RPM fac {rpm_fac[mask].mean():.0f} -> opt {rpm_opt[mask].mean():.0f}; "
                  f"pitch fac {pitch_fac[mask].mean():.2f} -> opt {pitch_opt[mask].mean():.2f})")

    # --- Plot ---
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    ax[0, 0].scatter(hs, saving_pct, s=3, alpha=0.15, color="steelblue")
    _binned_line(ax[0, 0], hs, saving_pct, np.linspace(0, hs.max(), 12), "tab:red")
    ax[0, 0].set_xlabel("Significant wave height Hs [m]")
    ax[0, 0].set_ylabel("Hourly saving [%]")
    ax[0, 0].set_title("vs Hs")
    ax[0, 0].grid(True, alpha=0.3)

    ax[0, 1].scatter(ws, saving_pct, s=3, alpha=0.15, color="steelblue")
    _binned_line(ax[0, 1], ws, saving_pct, np.linspace(0, ws.max(), 12), "tab:red")
    ax[0, 1].set_xlabel("Wind speed [m/s]")
    ax[0, 1].set_ylabel("Hourly saving [%]")
    ax[0, 1].set_title("vs wind speed")
    ax[0, 1].grid(True, alpha=0.3)

    ax[1, 0].scatter(T_req, saving_pct, s=3, alpha=0.15, color="steelblue")
    _binned_line(ax[1, 0], T_req, saving_pct, np.linspace(T_req.min(), T_req.max(), 14), "tab:red")
    ax[1, 0].set_xlabel("Thrust demand T_req [kN]")
    ax[1, 0].set_ylabel("Hourly saving [%]")
    ax[1, 0].set_title("vs propeller load (THE dominant driver)")
    ax[1, 0].grid(True, alpha=0.3)

    ax[1, 1].scatter(R_wind, saving_pct, s=3, alpha=0.15, color="steelblue")
    _binned_line(ax[1, 1], R_wind, saving_pct, np.linspace(R_wind.min(), R_wind.max(), 12), "tab:red")
    ax[1, 1].set_xlabel("Wind resistance R_wind [kN]  (negative = tailwind)")
    ax[1, 1].set_ylabel("Hourly saving [%]")
    ax[1, 1].set_title("vs wind resistance (sign = relative direction)")
    ax[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Link Galaxy — Weather sensitivity of optimiser saving "
                 f"({args.speed} kn, {args.year} NORA3)\n"
                 f"Optimiser strategy: reduce RPM (up to −15 %) toward SFOC "
                 f"island; pitch settles near design P/D 1.05 "
                 f"(hydrodynamic optimum, NOT a cap).\n"
                 f"Savings collapse in heavy seas because higher required "
                 f"thrust forces higher RPM to stay on the propeller polar.",
                 fontsize=10.5)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print(f"\nWrote {args.out}")


def _binned_line(ax, x, y, edges, color):
    mids, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() > 20:
            mids.append(0.5 * (lo + hi))
            means.append(y[mask].mean())
    ax.plot(mids, means, "o-", color=color, lw=2, ms=6,
            label="Bin mean", zorder=5)
    ax.legend(loc="best", fontsize=9)


if __name__ == "__main__":
    main()
