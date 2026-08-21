"""Muzzle diagram for the Wartsila Vasa 32 16V (Link Galaxy derated).

Reads modules/config_link_galaxy/propulsion_optimiser_engine_4.prototxt.in,
plots iso-SFOC contours on (engine RPM, shaft power) axes with the
constant-torque envelope overlaid.
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
OUT_FILE = Path(__file__).parent / "muzzle_diagram.png"

_SCALAR = re.compile(r"^\s*(\w+)\s*:\s*([-\d.]+)")
_ARRAY = re.compile(r"^\s*(\w+)\s*:\s*\[\s*([^\]]*)\s*\]")


def parse_engine_proto(path: Path) -> dict:
    """Extract flat scalar/array fields from the engine prototxt.

    Handles multi-line arrays (`name: [ ... ]` spanning several lines,
    optionally with `# ...` comments inside).
    """
    text = path.read_text()
    # Strip full-line comments but preserve line breaks so scalars still parse.
    stripped = "\n".join(re.sub(r"#.*$", "", ln) for ln in text.splitlines())

    out: dict = {}
    # Multi-line arrays first (greedy on whitespace, non-greedy on content).
    for m in re.finditer(r"(\w+)\s*:\s*\[([^\]]*)\]", stripped, re.DOTALL):
        name = m.group(1)
        vals = [float(x) for x in m.group(2).replace(",", " ").split() if x]
        out[name] = vals
    # Scalars (single line, one field per line).
    for ln in stripped.splitlines():
        m = re.match(r"^\s*(\w+)\s*:\s*([-\d.]+)\s*$", ln)
        if m and m.group(1) not in out:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def build_sfoc_grid(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rpm_axis, power_axis, sfoc_matrix[nrpm, npower])."""
    rpm = np.array(cfg["sfoc_rpm"])
    pwr = np.array(cfg["sfoc_power_kw"])
    tbl = np.array(cfg["sfoc_table"]).reshape(len(rpm), len(pwr))
    return rpm, pwr, tbl


def main() -> None:
    cfg = parse_engine_proto(ENGINE_FILE)

    rpm_axis, pwr_axis, sfoc = build_sfoc_grid(cfg)

    # Envelope
    env_rpm = np.array(cfg["power_limit_rpm"])
    env_pwr = np.array(cfg["power_limit_kw"])
    max_pwr = cfg["max_power_kw"]
    max_rpm = cfg["max_rpm"]
    min_rpm = cfg["min_rpm"]

    # -- Dense grid for smooth contours --
    rpm_dense = np.linspace(rpm_axis.min(), rpm_axis.max(), 240)
    pwr_dense = np.linspace(pwr_axis.min(), pwr_axis.max(), 240)
    R, P = np.meshgrid(rpm_dense, pwr_dense, indexing="xy")

    # Bilinear interpolation on the 7x10 table (matches
    # MuzzleDiagramEngine::InterpolateSfoc semantics).
    i = np.clip(np.searchsorted(rpm_axis, R) - 1, 0, len(rpm_axis) - 2)
    j = np.clip(np.searchsorted(pwr_axis, P) - 1, 0, len(pwr_axis) - 2)
    tr = (R - rpm_axis[i]) / (rpm_axis[i + 1] - rpm_axis[i])
    tp = (P - pwr_axis[j]) / (pwr_axis[j + 1] - pwr_axis[j])
    s00 = sfoc[i, j]
    s10 = sfoc[i + 1, j]
    s01 = sfoc[i, j + 1]
    s11 = sfoc[i + 1, j + 1]
    S = (1 - tr) * ((1 - tp) * s00 + tp * s01) + tr * ((1 - tp) * s10 + tp * s11)

    # Mask above envelope for display
    env_interp = np.interp(R, env_rpm, env_pwr, left=np.nan, right=np.nan)
    S_masked = np.where(P <= env_interp, S, np.nan)

    # -- Plot --
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    levels = np.arange(np.nanmin(S) // 1, np.nanmax(S) // 1 + 2, 1.0)
    cs = ax.contourf(R, P, S_masked, levels=levels, cmap="viridis_r", alpha=0.85)
    lines = ax.contour(R, P, S_masked, levels=levels[::2], colors="white",
                       linewidths=0.6, alpha=0.7)
    ax.clabel(lines, fmt="%d", fontsize=8, colors="white")

    # Envelope (constant torque)
    ax.plot(env_rpm, env_pwr, color="crimson", lw=2.2, label="Power envelope P = 4000·n/720")
    ax.fill_between(env_rpm, env_pwr, pwr_axis.max(), color="lightgrey",
                    alpha=0.55, zorder=2)

    # Optimiser hard cushion (99%) + typical sea-state margin envelope for context
    ax.plot(env_rpm, 0.99 * env_pwr, color="crimson", lw=1.0, ls="--",
            label="Optimiser hard limit (0.99 · P_env)")

    # MCR and anchor points
    ax.plot([max_rpm], [max_pwr], "o", ms=10, mec="black", mfc="gold",
            label=f"Derated MCR (4000 kW @ 720)")

    # Wartsila 3-point SFOC anchors at 720 rpm (100/75/50 % of 5920 nameplate)
    anchors = [(720, 5920, 184), (720, 4440, 188), (720, 2960, 194)]
    for n, kw, sf in anchors:
        if kw <= pwr_axis.max():
            ax.plot([n], [kw], "s", ms=7, mec="black", mfc="white")
            ax.annotate(f"{sf} g/kWh", xy=(n, kw), xytext=(6, 6),
                        textcoords="offset points", fontsize=8)

    # Combinator observed operating point from the sim (12.5 kn calm):
    #   pitch 78.6%, rpm 100% -> 720 engine rpm, ~3000 kW shaft
    ax.plot([720], [3000], "^", ms=10, mec="black", mfc="tab:blue",
            label="Combinator @ 12.5 kn calm (~3000 kW)")
    ax.plot([712], [3000], "*", ms=14, mec="black", mfc="tab:orange",
            label="Optimiser @ 12.5 kn calm (98.9% rpm)")

    ax.axvline(min_rpm, color="k", lw=0.7, ls=":", alpha=0.6)
    ax.annotate(f"min RPM = {min_rpm:.0f}", xy=(min_rpm, pwr_axis.max() * 0.95),
                xytext=(6, 0), textcoords="offset points", fontsize=8, va="top")

    ax.set_xlim(rpm_axis.min(), rpm_axis.max())
    ax.set_ylim(0, pwr_axis.max())
    ax.set_xlabel("Engine speed [rpm]")
    ax.set_ylabel("Shaft power [kW]")
    ax.set_title("Wärtsilä Vasa 32 16V – Link Galaxy muzzle diagram\n"
                 "iso-SFOC [g/kWh], constant-torque envelope, sim operating point",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(cs, ax=ax, label="SFOC [g/kWh]")
    cbar.ax.tick_params(labelsize=8)

    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=140)
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
