"""Sanity-check the Holtrop-Mennen calm-water curve for Link Galaxy.

Compares:
  - Holtrop-Mennen port (link_galaxy/holtrop_mennen.py) at Tm = 5.0 m.
  - V2597 towing-tank endpoint: 608.5 kN @ 19.5 kn (Tm ~4.5 m).
  - A quadratic R = 608.5 * (V/19.5)^2 to the tank endpoint.
  - Viscous / wave-making split from the port itself.

Also cross-checks:
  - Volume displacement vs the pdstrip section-integration value (8972.6 m^3).
  - Denny-Mumford wetted surface vs a wetted-surface estimate
    S ~ 2.5 * sqrt(V * L) (Ayre / simple rule of thumb).

Prints a table and writes a PNG (link_galaxy/resistance_check.png).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from link_galaxy.holtrop_mennen import (          # noqa: E402
    _hull_params, calm_water_kN, viscous_kN, wave_making_kN,
)


LPP, B, T = 137.2, 19.0, 5.0
V_TANK_KN = 19.5
R_TANK_KN = 608.5
PDSTRIP_VOL = 8972.63


def main() -> None:
    hp = _hull_params(LPP, B, T)
    print("HullParams from sections.prototxt (39-station resample):")
    for k, v in hp.items():
        print(f"  {k:12s} = {v:.4f}")

    S_ayre = 2.5 * np.sqrt(hp["volume"] * LPP)
    print(f"\nVolume vs pdstrip section-integration:")
    print(f"  H-M port : {hp['volume']:.1f} m^3")
    print(f"  pdstrip  : {PDSTRIP_VOL:.1f} m^3   "
          f"(delta = {(hp['volume'] - PDSTRIP_VOL) / PDSTRIP_VOL * 100:+.2f}%)")
    print(f"\nWetted surface:")
    print(f"  Denny-M  : {hp['S_wet']:.1f} m^2")
    print(f"  Ayre ~2.5*sqrt(V*L) : {S_ayre:.1f} m^2   "
          f"(delta = {(hp['S_wet'] - S_ayre) / S_ayre * 100:+.1f}%)")

    speeds_kn = np.linspace(0.0, 20.0, 41)
    speeds_ms = speeds_kn * 0.5144444
    R_visc = np.array([viscous_kN(v, hp) for v in speeds_ms])
    R_wave = np.array([wave_making_kN(v, hp) for v in speeds_ms])
    R_calm = R_visc + R_wave
    R_quad = R_TANK_KN * (speeds_kn / V_TANK_KN) ** 2

    print(f"\n{'V [kn]':>7s}  {'R_visc':>8s}  {'R_wave':>8s}  "
          f"{'R_calm':>8s}  {'R_quad':>8s}  {'diff %':>8s}")
    for v, rv, rw, rc, rq in zip(speeds_kn, R_visc, R_wave, R_calm, R_quad):
        if v < 5.0 or v % 2 > 1e-9:
            continue
        d = (rc - rq) / rq * 100 if rq > 0 else 0.0
        print(f"{v:>7.1f}  {rv:>8.1f}  {rw:>8.1f}  "
              f"{rc:>8.1f}  {rq:>8.1f}  {d:>+8.1f}")

    R_at_tank = np.interp(V_TANK_KN, speeds_kn, R_calm)
    print(f"\nAt V = {V_TANK_KN} kn:")
    print(f"  Holtrop-Mennen : {R_at_tank:.1f} kN")
    print(f"  V2597 tank     : {R_TANK_KN:.1f} kN")
    print(f"  H-M / tank     : {R_at_tank / R_TANK_KN * 100:.1f} %")

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(speeds_kn, R_calm, "-", lw=2.2, color="#1f77b4",
            label="Holtrop-Mennen (port from brucon)")
    ax.plot(speeds_kn, R_visc, "--", lw=1.2, color="#2ca02c",
            label="ITTC 1957 viscous (form factor 1.3)")
    ax.plot(speeds_kn, R_wave, "--", lw=1.2, color="#d62728",
            label="Wave-making")
    ax.plot(speeds_kn, R_quad, ":", lw=1.4, color="#7f7f7f",
            label=f"Quadratic to V2597 endpoint "
                  f"({R_TANK_KN:.0f} kN @ {V_TANK_KN} kn)")
    ax.plot(V_TANK_KN, R_TANK_KN, "s", ms=9, color="k",
            label="V2597 towing tank")
    ax.set_xlabel("Ship speed [kn]")
    ax.set_ylabel("Calm-water resistance [kN]")
    ax.set_title(f"MV Link Galaxy calm-water resistance  "
                 f"(Lpp={LPP} m, B={B} m, T={T} m)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    out = Path(__file__).parent / "resistance_check.png"
    fig.savefig(out, dpi=140)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
