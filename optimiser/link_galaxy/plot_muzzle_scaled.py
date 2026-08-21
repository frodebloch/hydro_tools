"""Scale-to-fit MAN L27/38 muzzle onto the Wartsila Vasa 32 16V envelope.

Approach:
  1. Take the real MAN L27/38 SFOC table from config_optimiser/
     propulsion_optimiser_engine_1.prototxt.in (Vessel 206, Aas Mekvik).
     This is a published-source muzzle with realistic off-design behaviour.
  2. Normalise its axes to [0,1]: rpm' = (rpm-480)/(800-480), kW' = kW/2920.
  3. Map to the Vasa 32 domain by re-dimensionalising:
        rpm_vasa = 480 + rpm' * (720 - 480)
        kW_vasa  = kW' * 5920      (nameplate, so 3-point anchors align)
     Sample the MAN SFOC on the LG grid via bilinear interp in normalised space.
  4. Affine fit sfoc_vasa = a * sfoc_man + b via least squares over the three
     Wartsila 3-point anchors (720 rpm × 2960/4440/5920 kW → 194/188/184).
     Two unknowns, three constraints -> residuals quantify shape mismatch.
  5. Print residuals + table, plot muzzle, envelope, sim operating points.

Shape assumption: a Wartsila Vasa 32 4-stroke medium-speed diesel has the
same *shape* as an MAN L27/38 of the same era/class. Only the anchors are
Wartsila-specific.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("PYTHONNOUSERSITE", "1")

import matplotlib.pyplot as plt
import numpy as np

MAN_FILE = Path("/home/blofro/src/brucon/modules/config_optimiser/"
                "propulsion_optimiser_engine_1.prototxt.in")
LG_FILE = Path("/home/blofro/src/brucon/modules/config_link_galaxy/"
               "propulsion_optimiser_engine_4.prototxt.in")
OUT_FILE = Path(__file__).parent / "muzzle_diagram_scaled.png"

# LG grid axes (target)
RPM_AXIS = np.array([480, 520, 560, 600, 640, 680, 720], dtype=float)
POWER_AXIS = np.array([400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000], dtype=float)

# MAN L27/38 envelope reference
MAN_RPM_MIN, MAN_RPM_MAX = 480.0, 800.0
MAN_POWER_MAX = 2920.0

# Vasa 32 nameplate & derated
VASA_RPM_MIN, VASA_RPM_MAX = 480.0, 720.0
VASA_POWER_NAMEPLATE = 5920.0   # for anchor alignment
VASA_POWER_DERATED = 4000.0     # derated MCR (config domain)

# Wartsila 3-point anchors at 720 rpm (Marine Project Guide 2/1997 §3.6)
ANCHORS = [(720.0, 2960.0, 194.0),
           (720.0, 4440.0, 188.0),
           (720.0, 5920.0, 184.0)]


def parse_prototxt_arrays(path: Path) -> dict:
    text = "\n".join(re.sub(r"#.*$", "", ln) for ln in path.read_text().splitlines())
    out = {}
    for m in re.finditer(r"(\w+)\s*:\s*\[([^\]]*)\]", text, re.DOTALL):
        vals = [float(x) for x in m.group(2).replace(",", " ").split() if x]
        out[m.group(1)] = np.array(vals)
    return out


def bilinear(rpm_axis, power_axis, table, R, P):
    i = np.clip(np.searchsorted(rpm_axis, R) - 1, 0, len(rpm_axis) - 2)
    j = np.clip(np.searchsorted(power_axis, P) - 1, 0, len(power_axis) - 2)
    tr = (R - rpm_axis[i]) / (rpm_axis[i + 1] - rpm_axis[i])
    tp = (P - power_axis[j]) / (power_axis[j + 1] - power_axis[j])
    s00, s10 = table[i, j], table[i + 1, j]
    s01, s11 = table[i, j + 1], table[i + 1, j + 1]
    return (1 - tr) * ((1 - tp) * s00 + tp * s01) + tr * ((1 - tp) * s10 + tp * s11)


def main():
    # -- Load MAN L27/38 muzzle --
    man = parse_prototxt_arrays(MAN_FILE)
    man_rpm = man["sfoc_rpm"]
    man_pwr = man["sfoc_power_kw"]
    man_sfoc = man["sfoc_table"].reshape(len(man_rpm), len(man_pwr))
    print(f"MAN L27/38: rpm {man_rpm.min():.0f}-{man_rpm.max():.0f}, "
          f"kW {man_pwr.min():.0f}-{man_pwr.max():.0f}, "
          f"SFOC {man_sfoc.min():.1f}-{man_sfoc.max():.1f} g/kWh "
          f"(spread {man_sfoc.max() - man_sfoc.min():.1f})")

    # -- Sample MAN on the LG grid via normalised-axis mapping --
    # For each (rpm_vasa, kW_vasa) point, compute normalised MAN coords:
    #   rpm' = (rpm_vasa - VASA_RPM_MIN) / (VASA_RPM_MAX - VASA_RPM_MIN)
    #   kW'  = kW_vasa / VASA_POWER_NAMEPLATE
    # Then interpret those in MAN units.
    Rv, Pv = np.meshgrid(RPM_AXIS, POWER_AXIS, indexing="ij")
    rpm_norm = (Rv - VASA_RPM_MIN) / (VASA_RPM_MAX - VASA_RPM_MIN)
    pwr_norm = Pv / VASA_POWER_NAMEPLATE
    R_man = MAN_RPM_MIN + rpm_norm * (MAN_RPM_MAX - MAN_RPM_MIN)
    P_man = pwr_norm * MAN_POWER_MAX
    # Clip to MAN table domain (extrapolation avoided; corners saturate).
    R_man = np.clip(R_man, man_rpm.min(), man_rpm.max())
    P_man = np.clip(P_man, man_pwr.min(), man_pwr.max())
    S_man_on_lg = bilinear(man_rpm, man_pwr, man_sfoc, R_man, P_man)

    # -- Hybrid fit ---
    # Wartsila 720-rpm row: exact quadratic through 3-point anchors.
    #   S720(kW) = 212 - 0.00744*kW + 4.57e-7*kW^2  (nameplate kW domain)
    # For (rpm < 720, kW), take the MAN delta off its own rated row:
    #   delta(rpm,kW) = sfoc_man(rpm_eq, kW_eq) - sfoc_man(800, kW_eq)
    # Then: sfoc_vasa(rpm,kW) = S720(kW) + delta(rpm,kW).
    # At rpm=720, rpm_eq=800, delta=0 -> anchors exact by construction.

    def wartsila_720_row(kw):
        return 212.0 - 0.00744 * kw + 4.57e-7 * kw ** 2

    sfoc_vasa = np.zeros_like(Rv)
    for i, rpm in enumerate(RPM_AXIS):
        rn = (rpm - VASA_RPM_MIN) / (VASA_RPM_MAX - VASA_RPM_MIN)
        rm = MAN_RPM_MIN + rn * (MAN_RPM_MAX - MAN_RPM_MIN)
        for j, kw in enumerate(POWER_AXIS):
            pn = kw / VASA_POWER_NAMEPLATE
            pm = np.clip(pn * MAN_POWER_MAX, man_pwr.min(), man_pwr.max())
            s_off = bilinear(man_rpm, man_pwr, man_sfoc,
                             np.array([[rm]]), np.array([[pm]]))[0, 0]
            s_rated = bilinear(man_rpm, man_pwr, man_sfoc,
                               np.array([[MAN_RPM_MAX]]),
                               np.array([[pm]]))[0, 0]
            delta = s_off - s_rated
            sfoc_vasa[i, j] = wartsila_720_row(kw) + delta

    print("\nHybrid fit: Wartsila quadratic at 720 rpm + MAN off-rated delta")
    print("Anchor evaluation at Wartsila 3-point (720 rpm):")
    for r, p, target in ANCHORS:
        pred = wartsila_720_row(p)  # anchors are all at rpm=720 -> delta=0
        print(f"  {p:>4.0f} kW: target {target:.1f}, predicted {pred:.2f}, "
              f"residual {pred - target:+.2f} g/kWh")

    sfoc_vasa_flat = sfoc_vasa  # already in shape (n_rpm, n_power)
    # (S_man_on_lg no longer needed for the hybrid method)

    # -- Print table --
    print(f"\nScaled SFOC table [g/kWh], rows=RPM, cols=kW  "
          f"(spread {sfoc_vasa.max() - sfoc_vasa.min():.1f}):")
    print("        " + " ".join(f"{int(k):>5d}" for k in POWER_AXIS))
    for i, rpm in enumerate(RPM_AXIS):
        row = " ".join(f"{sfoc_vasa[i, j]:>5.1f}" for j in range(len(POWER_AXIS)))
        print(f"  {int(rpm):>4d}: {row}")

    # -- Envelope for plotting --
    lg = parse_prototxt_arrays(LG_FILE)
    env_rpm = lg["power_limit_rpm"]
    env_pwr = lg["power_limit_kw"]

    # Dense interpolation for smooth contours
    rpm_dense = np.linspace(RPM_AXIS.min(), RPM_AXIS.max(), 240)
    pwr_dense = np.linspace(POWER_AXIS.min(), POWER_AXIS.max(), 240)
    R, P = np.meshgrid(rpm_dense, pwr_dense, indexing="xy")
    S = bilinear(RPM_AXIS, POWER_AXIS, sfoc_vasa, R, P)
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

    ax.plot(env_rpm, env_pwr, color="crimson", lw=2.2,
            label="Power envelope P = 4000·n/720")
    ax.fill_between(env_rpm, env_pwr, POWER_AXIS.max(), color="lightgrey",
                    alpha=0.55, zorder=2)
    ax.plot(env_rpm, 0.99 * env_pwr, color="crimson", lw=1.0, ls="--",
            label="Optimiser hard limit (0.99 · P_env)")

    ax.plot([720], [4000], "o", ms=10, mec="black", mfc="gold",
            label="Derated MCR (4000 kW @ 720)")

    # Wartsila 3-point anchor (only the 2960 kW one is inside the derated domain)
    ax.plot([720], [2960], "s", ms=7, mec="black", mfc="white")
    ax.annotate("194 g/kWh anchor", xy=(720, 2960), xytext=(-70, -15),
                textcoords="offset points", fontsize=8)

    # BSFC island min
    imin = np.unravel_index(np.nanargmin(S_masked), S_masked.shape)
    r_min, p_min, s_min = R[imin], P[imin], S_masked[imin]
    ax.plot([r_min], [p_min], "D", ms=10, mec="black", mfc="lime",
            label=f"BSFC island min ({s_min:.1f} g/kWh @ {r_min:.0f} rpm × {p_min:.0f} kW)")

    ax.plot([720], [3000], "^", ms=10, mec="black", mfc="tab:blue",
            label="Sim combinator @ 12.5 kn calm (~3000 kW)")
    ax.plot([712], [3000], "*", ms=14, mec="black", mfc="tab:orange",
            label="Sim optimiser @ 12.5 kn calm (98.9 % rpm)")

    ax.axvline(475, color="k", lw=0.7, ls=":", alpha=0.6)
    ax.annotate("min RPM = 475", xy=(475, POWER_AXIS.max() * 0.95),
                xytext=(6, 0), textcoords="offset points", fontsize=8, va="top")

    ax.set_xlim(RPM_AXIS.min(), RPM_AXIS.max())
    ax.set_ylim(0, POWER_AXIS.max())
    ax.set_xlabel("Engine speed [rpm]")
    ax.set_ylabel("Shaft power [kW]")
    ax.set_title("Wärtsilä Vasa 32 16V (derated) — hybrid muzzle\n"
                 "Wärtsilä 3-point exact at 720 rpm; MAN L27/38 shape for off-rated behaviour",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(cs, ax=ax, label="SFOC [g/kWh]")
    cbar.ax.tick_params(labelsize=8)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=140)
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
