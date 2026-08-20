"""Holtrop-Mennen calm-water resistance for MV Link Galaxy.

Python port of brucon's Holtrop-Mennen implementation
(``libs/dp/vessel_model/wave_making_resistance.cpp`` and the ITTC 1957
viscous component in
``libs/simulator/propulsion_optimiser_simulator/vessel_resistance.cpp``).

Reads section offsets/breadths/drafts/areas from
``brucon/modules/config_link_galaxy/hull_geometry/sections.prototxt``
and derives HullParams the same way ``VesselResistance`` does
(volume displacement, wetted surface via Denny-Mumford, Cp from V/(L*Amax),
bow half-angle of entrance from the 2nd/3rd sections from bow, transom
area from the stern section).

Values match brucon's ``HoltropCalmWaterResistance(Vs)`` bit-for-bit
modulo interpolation of the section grid.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np


SECTIONS_PROTOTXT = Path(
    "/home/blofro/src/brucon/modules/config_link_galaxy/hull_geometry/"
    "sections.prototxt"
)

RHO = 1025.0            # kg/m^3
G = 9.80665             # m/s^2
NU = 1.19e-6            # m^2/s  (seawater at 15 deg C)
ROUGHNESS_M = 150e-6    # ITTC default AHR
FORM_FACTOR = 1.3       # (1+k), same default as brucon


# ------------------------------------------------------------
# Section parsing and hull-form derivation
# ------------------------------------------------------------

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _parse_sections(path: Path):
    """Return (offsets, breadths, drafts, areas), stern -> bow."""
    text = path.read_text()
    offs, brs, drs, ars = [], [], [], []
    for block in text.split("section {")[1:]:
        vals = {}
        for line in block.splitlines():
            m = re.match(r"\s*(offset|breadth|draft|area):\s*(\S+)", line)
            if m:
                vals[m.group(1)] = float(m.group(2))
        if {"offset", "breadth", "draft", "area"} <= vals.keys():
            offs.append(vals["offset"])
            brs.append(vals["breadth"])
            drs.append(vals["draft"])
            ars.append(vals["area"])
    idx = np.argsort(offs)
    return (np.array(offs)[idx], np.array(brs)[idx],
            np.array(drs)[idx], np.array(ars)[idx])


def _hull_params(lpp: float, breadth: float, draft: float):
    """Reproduce brucon's HullParams derivation (39-station resample)."""
    offs, brs, _, ars = _parse_sections(SECTIONS_PROTOTXT)

    # 39-station resample (same as kResampledStations in brucon).
    n = 39
    xs = np.linspace(offs.min(), offs.max(), n)
    areas = np.interp(xs, offs, ars)
    breadths = np.interp(xs, offs, brs)
    step = (offs.max() - offs.min()) / (n - 1)

    # Trapezoidal integration of section areas -> volume.
    volume = 0.5 * (areas[0] + areas[-1]) + areas[1:-1].sum()
    volume *= step

    # Denny-Mumford wetted surface.
    S_wet = 1.7 * lpp * draft + volume / draft

    # Prismatic coefficient.
    a_mid = areas.max()
    Cp = volume / (lpp * a_mid) if a_mid > 0 else 0.0

    # Bow half-angle of entrance from 2nd/3rd sections from bow.
    b2, b3 = breadths[-2], breadths[-3]
    x2, x3 = xs[-2], xs[-3]
    iE_deg = math.degrees(math.atan2((b3 - b2) / 2.0, x2 - x3))

    # Transom area = stern section.
    A_transom = areas[0]

    return dict(
        lpp=lpp, breadth=breadth, draft=draft,
        volume=volume, S_wet=S_wet, Cp=Cp,
        iE_deg=iE_deg, A_transom=A_transom,
        form_factor=FORM_FACTOR,
    )


# ------------------------------------------------------------
# Holtrop-Mennen wave-making (brucon port)
# ------------------------------------------------------------


def _c7(L, B):
    r = B / L
    if r < 0.11:
        return 0.229577 * r ** 0.33333
    if r < 0.25:
        return r
    return 0.5 - 0.0625 * r


def _c1(c7, T, B, iE):
    if B <= 0 or iE >= 90.0:
        return 0.0
    return 2223105.0 * c7 ** 3.78613 * (T / B) ** 1.07961 * (90.0 - iE) ** -1.37565


def _c5(A_t, T, B, Cm):
    if B * T * Cm <= 0:
        return 1.0
    return 1.0 - 0.8 * A_t / (B * T * Cm)


def _c16(Cp):
    if Cp < 0.8:
        return 8.07981 * Cp - 13.8673 * Cp ** 2 + 6.984388 * Cp ** 3
    return 1.73014 - 0.7067 * Cp


def _m1(L, T, B, V, c16):
    return (0.0140407 * L / T
            - 1.75254 * V ** (1.0 / 3.0) / L
            - 4.79323 * B / L
            - c16)


def _c15(L, V):
    r = L ** 3 / V
    if r < 512.0:
        return -1.69385
    if L ** 3 < 1726.91:
        return -1.69385 + (L / V ** (1.0 / 3.0) - 8.0) / 2.36
    return 0.0


def _m4(c15, Fn):
    if Fn <= 0.1:
        return 0.0
    return c15 * 0.4 * math.exp(-0.034 * Fn ** -3.29)


def _lam(L, B, Cp):
    if L / B < 12.0:
        return 1.446 * Cp - 0.03 * L / B
    return 1.446 * Cp - 0.36


def wave_making_kN(V_ms: float, hp: dict) -> float:
    L, B, T = hp["lpp"], hp["breadth"], hp["draft"]
    Vol, Cp, iE, At = hp["volume"], hp["Cp"], hp["iE_deg"], hp["A_transom"]
    if V_ms <= 0 or L * B * T * Cp <= 0:
        return 0.0
    Cm = (Vol / (L * B * T)) / Cp
    Fn = min(V_ms / math.sqrt(G * L), 0.5)
    if Fn < 0.01:
        return 0.0
    c7 = _c7(L, B)
    c1 = _c1(c7, T, B, iE)
    c5 = _c5(At, T, B, Cm)
    c15 = _c15(L, Vol)
    c16 = _c16(Cp)
    m1 = _m1(L, T, B, Vol, c16)
    m4 = _m4(c15, Fn)
    lam = _lam(L, B, Cp)
    # brucon returns negative; caller negates. We return positive kN.
    R_N = (c1 * c5 * Vol * RHO * G
           * math.exp(m1 * Fn ** -0.9 + m4 * math.cos(lam * Fn ** -2)))
    return R_N / 1000.0


def viscous_kN(V_ms: float, hp: dict) -> float:
    if V_ms <= 1e-6:
        return 0.0
    Re = V_ms * hp["lpp"] / NU
    log10Re = math.log10(Re)
    if log10Re <= 2.0:
        return 0.0
    Cf = 0.075 / (log10Re - 2.0) ** 2
    ks_over_L = ROUGHNESS_M / hp["lpp"]
    dCf = (105.0 * ks_over_L ** (1.0 / 3.0) - 0.64) * 1e-3
    Cf_total = Cf + max(0.0, dCf)
    R_N = hp["form_factor"] * 0.5 * RHO * V_ms ** 2 * Cf_total * hp["S_wet"]
    return R_N / 1000.0


def calm_water_kN(V_ms: float, hp: dict) -> float:
    return viscous_kN(V_ms, hp) + wave_making_kN(V_ms, hp)


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------


def resistance_table(speeds_kn: np.ndarray,
                     lpp: float, breadth: float, draft: float) -> np.ndarray:
    """Return calm-water resistance [kN] at each speed [kn]."""
    hp = _hull_params(lpp, breadth, draft)
    speeds_ms = speeds_kn * 0.5144444
    return np.array([calm_water_kN(v, hp) for v in speeds_ms])


if __name__ == "__main__":
    lpp, B, T = 137.2, 19.0, 5.0
    hp = _hull_params(lpp, B, T)
    print("HullParams (derived from sections.prototxt):")
    for k, v in hp.items():
        print(f"  {k:12s} = {v:.4f}")
    print()
    speeds = np.array([0, 5, 8, 10, 11, 12, 12.5, 13, 14, 15, 16, 17, 18, 19.5])
    R = resistance_table(speeds, lpp, B, T)
    print(f"{'Vs [kn]':>8s}  {'R_calm [kN]':>12s}")
    for v, r in zip(speeds, R):
        print(f"{v:>8.1f}  {r:>12.1f}")
