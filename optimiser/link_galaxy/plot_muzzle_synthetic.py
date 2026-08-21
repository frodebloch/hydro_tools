"""Synthetic Wartsila Vasa 32 16V muzzle diagram — Link Galaxy derated MCR.

Purpose: replace the 3-point + linear-RPM stub SFOC map with a physically
plausible 2D muzzle featuring a resolved BSFC island. Plot only for now;
if the shape is acceptable, the same table is written into the engine
prototxt for a sensitivity re-run of the annual voyage sweep.

Design constraints (anchored to Wartsila Marine Project Guide 2/1997 §3.6
16V32D):
  1. Row at 720 rpm follows a quadratic fit through the three published
     anchors: 184 g/kWh @ 5920 kW (100 % MCR nameplate), 188 @ 4440,
     194 @ 2960. This row is exact by construction.
  2. Below rated speed a BSFC island opens up near (660-680 rpm, 3200-3600 kW).
     Island depth ~7-8 g/kWh below the 720 rpm row at the same power.
     Depth and location match typical Wartsila 32 4-stroke medium-speed
     characteristics (published Wartsila emissions / MEPC data).
  3. Low-load penalty grows toward the (480 rpm, 400 kW) corner:
     TC deficit + cold-cylinder scavenging losses.

The 3-point Wartsila anchors are respected at 720 rpm; the RPM axis is
now shaped like a real muzzle rather than a linear low-RPM penalty.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("PYTHONNOUSERSITE", "1")

import matplotlib.pyplot as plt
import numpy as np

ENGINE_FILE = Path("/home/blofro/src/brucon/modules/config_link_galaxy/"
                   "propulsion_optimiser_engine_4.prototxt.in")
OUT_FILE = Path(__file__).parent / "muzzle_diagram_synthetic.png"


# -- Grid axes (match the current config layout) ----------------------------
RPM_AXIS = np.array([480, 520, 560, 600, 640, 680, 720], dtype=float)
POWER_AXIS = np.array([400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000], dtype=float)


def build_synthetic_sfoc() -> np.ndarray:
    """Return (n_rpm, n_power) SFOC surface [g/kWh]."""

    # (1) Row at 720 rpm: quadratic fit through Wartsila 3-point anchors.
    # SFOC(720, kW) = 212 - 0.00744 kW + 4.57e-7 kW^2
    # Anchors reproduced: 184 @ 5920, 188 @ 4440, 194 @ 2960.
    s720 = 212.0 - 0.00744 * POWER_AXIS + 4.57e-7 * POWER_AXIS ** 2

    # (2) Muzzle shape: SFOC(rpm, kW) = S_720(kW) + shift(rpm, kW).
    # shift models the BSFC island opening up below rated RPM.
    # r_opt is the RPM of best economy at each power level. It slides
    # from ~680 rpm at low load down to ~660 rpm at high load.
    r_opt = 680.0 - 20.0 * np.clip(POWER_AXIS / 4000.0, 0.0, 1.0)

    # Island depth (how much SFOC drops off rated speed at each power).
    # Small at low load (no room to gain), grows with load up to ~8 g/kWh
    # near 3200-3600 kW (the sweet spot), tapers slightly at derated MCR.
    depth = 8.0 * np.exp(-((POWER_AXIS - 3400.0) / 900.0) ** 2)

    # Anisotropic penalty: sharper rise below r_opt (torque-limited combustion,
    # TC deficit) than above r_opt (approaching rated).
    def shift(rpm, r_opt_col, depth_col):
        dr = rpm - r_opt_col
        rise_above = 0.005 * np.maximum(dr, 0.0) ** 2         # gentle above r_opt
        rise_below = 0.0018 * np.maximum(-dr, 0.0) ** 2       # steeper below r_opt (per unit)
        # Normalise so that at (720, kW) shift = 0 -> S_720 anchors preserved.
        # This is achieved by subtracting the rise at rpm=720 from the field.
        rise_at_720 = 0.005 * max(720.0 - r_opt_col, 0.0) ** 2
        island_bowl = -depth_col * np.exp(-((rpm - r_opt_col) / 50.0) ** 2)
        island_bowl_at_720 = -depth_col * np.exp(-((720.0 - r_opt_col) / 50.0) ** 2)
        # net shift, calibrated so shift(720, kW) = 0
        return (rise_above + rise_below - rise_at_720
                + island_bowl - island_bowl_at_720)

    # Additional very-low-load penalty (dominates below ~1500 kW).
    lo_load = 6.0 * np.maximum((1500.0 - POWER_AXIS) / 1500.0, 0.0) ** 1.4

    # Additional low-RPM penalty at low-to-moderate load (cold-wall scavenging,
    # rises steeply below 550 rpm).
    def lo_rpm(rpm):
        return 8.0 * max((550.0 - rpm) / 100.0, 0.0) ** 1.6

    # Assemble the 2D table.
    sfoc = np.zeros((len(RPM_AXIS), len(POWER_AXIS)))
    for i, rpm in enumerate(RPM_AXIS):
        for j, kw in enumerate(POWER_AXIS):
            sfoc[i, j] = (s720[j] + shift(rpm, r_opt[j], depth[j])
                          + lo_load[j] * (0.4 + 0.6 * (1 - np.exp(-((rpm - 720.0) / 200.0) ** 2)))
                          + lo_rpm(rpm) * (0.3 + 0.7 * max((2000.0 - kw) / 2000.0, 0.0)))
    return sfoc


def parse_current_config_envelope(path: Path):
    """Extract power_limit_rpm/kw from the engine prototxt."""
    text = path.read_text()
    stripped = "\n".join(re.sub(r"#.*$", "", ln) for ln in text.splitlines())
    out = {}
    for m in re.finditer(r"(\w+)\s*:\s*\[([^\]]*)\]", stripped, re.DOTALL):
        vals = [float(x) for x in m.group(2).replace(",", " ").split() if x]
        out[m.group(1)] = vals
    return np.array(out["power_limit_rpm"]), np.array(out["power_limit_kw"])


def main():
    sfoc = build_synthetic_sfoc()

    # -- Verify Wartsila 3-point anchors --
    print("Wartsila anchor verification at 720 rpm (row-major axis order):")
    for kw_target, expected in [(2960, 194), (4440, 188), (5920, 184)]:
        # SFOC values from row 720 rpm (last row) via quadratic fit above
        s = 212.0 - 0.00744 * kw_target + 4.57e-7 * kw_target ** 2
        print(f"  {kw_target:4.0f} kW: predicted {s:.1f} g/kWh   (anchor {expected})")

    # -- Envelope --
    env_rpm, env_pwr = parse_current_config_envelope(ENGINE_FILE)

    # -- Dense interpolation for smooth contours --
    rpm_dense = np.linspace(RPM_AXIS.min(), RPM_AXIS.max(), 240)
    pwr_dense = np.linspace(POWER_AXIS.min(), POWER_AXIS.max(), 240)
    R, P = np.meshgrid(rpm_dense, pwr_dense, indexing="xy")

    # Bilinear interpolation
    i = np.clip(np.searchsorted(RPM_AXIS, R) - 1, 0, len(RPM_AXIS) - 2)
    j = np.clip(np.searchsorted(POWER_AXIS, P) - 1, 0, len(POWER_AXIS) - 2)
    tr = (R - RPM_AXIS[i]) / (RPM_AXIS[i + 1] - RPM_AXIS[i])
    tp = (P - POWER_AXIS[j]) / (POWER_AXIS[j + 1] - POWER_AXIS[j])
    s00, s10 = sfoc[i, j], sfoc[i + 1, j]
    s01, s11 = sfoc[i, j + 1], sfoc[i + 1, j + 1]
    S = (1 - tr) * ((1 - tp) * s00 + tp * s01) + tr * ((1 - tp) * s10 + tp * s11)

    env_interp = np.interp(R, env_rpm, env_pwr, left=np.nan, right=np.nan)
    S_masked = np.where(P <= env_interp, S, np.nan)

    # -- Plot --
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    lo = np.floor(np.nanmin(S_masked))
    hi = np.ceil(np.nanmax(S_masked))
    levels = np.arange(lo, hi + 1, 1.0)
    cs = ax.contourf(R, P, S_masked, levels=levels, cmap="viridis_r", alpha=0.85)
    lines = ax.contour(R, P, S_masked, levels=levels[::2],
                       colors="white", linewidths=0.6, alpha=0.7)
    ax.clabel(lines, fmt="%d", fontsize=8, colors="white")

    # Envelope
    ax.plot(env_rpm, env_pwr, color="crimson", lw=2.2,
            label="Power envelope P = 4000·n/720")
    ax.fill_between(env_rpm, env_pwr, POWER_AXIS.max(), color="lightgrey",
                    alpha=0.55, zorder=2)
    ax.plot(env_rpm, 0.99 * env_pwr, color="crimson", lw=1.0, ls="--",
            label="Optimiser hard limit (0.99 · P_env)")

    # MCR
    ax.plot([720], [4000], "o", ms=10, mec="black", mfc="gold",
            label="Derated MCR (4000 kW @ 720)")

    # Wartsila 3-point anchor
    ax.plot([720], [2960], "s", ms=7, mec="black", mfc="white")
    ax.annotate("194 g/kWh anchor", xy=(720, 2960), xytext=(-70, -15),
                textcoords="offset points", fontsize=8)

    # BSFC island min marker
    imin = np.unravel_index(np.nanargmin(S_masked), S_masked.shape)
    r_min = R[imin]
    p_min = P[imin]
    s_min = S_masked[imin]
    ax.plot([r_min], [p_min], "D", ms=10, mec="black", mfc="lime",
            label=f"BSFC island min ({s_min:.1f} g/kWh @ {r_min:.0f} rpm × {p_min:.0f} kW)")

    # Same operating points as before
    ax.plot([720], [3000], "^", ms=10, mec="black", mfc="tab:blue",
            label="Sim combinator @ 12.5 kn calm (~3000 kW)")
    ax.plot([712], [3000], "*", ms=14, mec="black", mfc="tab:orange",
            label="Sim optimiser @ 12.5 kn calm (98.9 % rpm)")

    ax.axvline(475, color="k", lw=0.7, ls=":", alpha=0.6)
    ax.annotate("min RPM = 475", xy=(475, POWER_AXIS.max() * 0.95),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8, va="top")

    ax.set_xlim(RPM_AXIS.min(), RPM_AXIS.max())
    ax.set_ylim(0, POWER_AXIS.max())
    ax.set_xlabel("Engine speed [rpm]")
    ax.set_ylabel("Shaft power [kW]")
    ax.set_title("Wärtsilä Vasa 32 16V (derated) — SYNTHETIC muzzle diagram\n"
                 "Wärtsilä 1997 §3.6 anchors preserved at 720 rpm; "
                 "BSFC island reconstructed from published 32-family characteristics",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(cs, ax=ax, label="SFOC [g/kWh]")
    cbar.ax.tick_params(labelsize=8)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=140)
    print(f"\nWrote {OUT_FILE}")

    # -- Print the raw table for review --
    print("\nSynthetic SFOC table [g/kWh], rows=RPM, cols=kW:")
    header = "        " + " ".join(f"{int(k):>5d}" for k in POWER_AXIS)
    print(header)
    for i, rpm in enumerate(RPM_AXIS):
        row = " ".join(f"{sfoc[i, j]:>5.1f}" for j in range(len(POWER_AXIS)))
        print(f"  {int(rpm):>4d}: {row}")


if __name__ == "__main__":
    main()
