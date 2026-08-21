"""Fuel-vs-pitch sweeps at fixed thrust — educational diagnostic.

At three representative loading conditions (light / medium / heavy),
sweep pitch ratio P/D across the CPP's physical range and plot the
fuel consumption. Shows why the optimiser converges near design pitch
(P/D ~ 1.05) rather than pushing pitch to its physical maximum:

  - Wageningen C-series propeller optimised at design pitch: eta_0 peaks
    at P/D = 1.05, drops off in both directions.
  - Beyond ~P/D 1.10, torque coefficient Kq rises faster than thrust
    coefficient Kt with pitch -> more torque per unit thrust required.
  - The engine SFOC-island gain from lower RPM is more than offset by
    the propeller-efficiency loss.

The propeller CAN mechanically go to P/D 1.4+ but you wouldn't want to.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import models
import models.vessel_link_galaxy as lg
sys.modules["models.constants"] = lg
models.constants = lg

import numpy as np                              # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402
from propeller_model import CSeriesPropeller, load_c_series_data  # noqa: E402
from optimiser import make_wartsila_vasa32_16v                    # noqa: E402
from models.constants import (                                    # noqa: E402
    DATA_PATH_C440, DATA_PATH_C455, DATA_PATH_C470,
    RHO_WATER, PROP_DIAMETER, PROP_DESIGN_PITCH, PROP_BAR,
    HULL_WAKE, GEAR_RATIO,
)

WAKE_FRACTION = float(HULL_WAKE[0])

OUT_FILE = Path(__file__).parent / "pitch_sweep.png"

DESIGN_SPEED_KN = 12.5
KN_TO_MS = 0.5144
ETA_R = 1.025
ENG_MIN_RPM = 475.0
ENG_MAX_RPM = 720.0


def find_shaft_n_for_thrust(prop, pd, T_kN, Va_ms):
    lo, hi = 0.4, 3.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        T = prop.thrust(pd, mid, Va_ms) * 1e-3
        if T < T_kN:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def evaluate(prop, engine, pd, T_kN, Va_ms):
    n_s = find_shaft_n_for_thrust(prop, pd, T_kN, Va_ms)
    Q = prop.torque(pd, n_s, Va_ms)
    P_prop_kW = 2 * np.pi * n_s * Q / 1000.0
    P_eng_kW = P_prop_kW / ETA_R
    n_eng = n_s * 60.0 * GEAR_RATIO   # GEAR_RATIO = engine_rpm / shaft_rpm
    if not (ENG_MIN_RPM <= n_eng <= ENG_MAX_RPM):
        return None
    sfoc = engine.sfoc(P_eng_kW, n_eng)
    kg_h = sfoc * P_eng_kW / 1000.0
    return dict(n_s_rpm=n_s * 60.0, n_eng=n_eng, P_eng_kW=P_eng_kW,
                sfoc=sfoc, kg_h=kg_h)


def main():
    bar_data = {
        0.40: load_c_series_data(DATA_PATH_C440),
        0.55: load_c_series_data(DATA_PATH_C455),
        0.70: load_c_series_data(DATA_PATH_C470),
    }
    prop = CSeriesPropeller(bar_data, design_pitch=PROP_DESIGN_PITCH,
                            diameter=PROP_DIAMETER, area_ratio=PROP_BAR,
                            rho=RHO_WATER)
    engine = make_wartsila_vasa32_16v()

    Va = DESIGN_SPEED_KN * KN_TO_MS * (1 - WAKE_FRACTION)

    load_cases = [
        (180.0, "Light load  (T = 180 kN)  — calm, ~50 % of hours", "tab:green"),
        (220.0, "Medium load (T = 220 kN)  — moderate seas, ~30 % of hours",
         "tab:orange"),
        (290.0, "Heavy load  (T = 290 kN)  — Hs > 2 m, ~8 % of hours",
         "tab:red"),
    ]

    pd_grid = np.linspace(0.70, 1.55, 44)

    fig, (ax_fuel, ax_sfoc) = plt.subplots(1, 2, figsize=(13.5, 6.0))

    print(f"Va = {Va:.2f} m/s (STW {DESIGN_SPEED_KN} kn, w = {WAKE_FRACTION:.3f})")
    for T_kN, label, colour in load_cases:
        pds, kg_h, sfoc, n_eng = [], [], [], []
        opt_pd = opt_kg = None
        for pd in pd_grid:
            res = evaluate(prop, engine, pd, T_kN, Va)
            if res is None:
                continue
            pds.append(pd); kg_h.append(res["kg_h"])
            sfoc.append(res["sfoc"]); n_eng.append(res["n_eng"])
            if opt_kg is None or res["kg_h"] < opt_kg:
                opt_kg, opt_pd = res["kg_h"], pd
        pds = np.array(pds); kg_h = np.array(kg_h)
        sfoc = np.array(sfoc); n_eng = np.array(n_eng)

        ax_fuel.plot(pds, kg_h, "-", color=colour, lw=2, label=label)
        ax_fuel.plot([opt_pd], [opt_kg], "o", color=colour, mec="black", ms=9)
        ax_fuel.annotate(f"opt P/D = {opt_pd:.2f}",
                         xy=(opt_pd, opt_kg), xytext=(6, -14),
                         textcoords="offset points", fontsize=9,
                         color=colour, fontweight="bold")

        ax_sfoc.plot(pds, sfoc, "-", color=colour, lw=2, label=label)
        # Mark design pitch
        print(f"\n{label}")
        print(f"  optimum: P/D = {opt_pd:.2f}, {opt_kg:.1f} kg/h")
        j105 = np.argmin(np.abs(pds - 1.05))
        j140 = np.argmin(np.abs(pds - 1.40))
        print(f"  P/D 1.05: {kg_h[j105]:.1f} kg/h (n_eng {n_eng[j105]:.0f})")
        if 1.30 < pds[j140] < 1.50:
            print(f"  P/D 1.40: {kg_h[j140]:.1f} kg/h "
                  f"(+{100*(kg_h[j140]/opt_kg-1):.1f} % over optimum)")

    for ax in (ax_fuel, ax_sfoc):
        ax.axvline(PROP_DESIGN_PITCH, color="k", lw=0.8, ls=":", alpha=0.6)
        ax.annotate("design P/D = 1.05", xy=(PROP_DESIGN_PITCH, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(5, 8), textcoords="offset points",
                    fontsize=8, color="k")
        ax.set_xlabel("Blade pitch ratio P/D")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
        ax.set_xlim(pd_grid[0], pd_grid[-1])

    ax_fuel.set_ylabel("Fuel consumption [kg/h]")
    ax_fuel.set_title("Fuel vs pitch at fixed thrust demand\n"
                      "Minimum near design pitch, NOT at physical maximum")
    ax_sfoc.set_ylabel("Engine SFOC [g/kWh]")
    ax_sfoc.set_title("Engine SFOC vs pitch\n"
                      "Higher pitch → lower RPM → into the BSFC island …\n"
                      "… but propeller efficiency drops faster")

    fig.suptitle("Link Galaxy — Why the optimiser stops near design pitch\n"
                 "Wärtsilä Vasa 32 16V (hybrid SFOC map) + Wageningen C4/40 propeller",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=140, bbox_inches="tight")
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
