#!/usr/bin/env python3
"""Polar capability plots (SVG, no dependencies) for the cable lay propulsion study.

Reads the .out files written by CablelayCapabilityStudy and writes SVG plots with the azimuth and shaft line
options overlaid. Radius = max mean wind speed [m/s] that can be held (Beaufort limits as rings), angle = weather direction relative to the bow
(0 = head weather, clockwise to starboard).

  capability_<mode>_<case>.svg   One plot per mode (DP, LAY) and case (Intact, Bus0, Bus1, Bus2, WCF)
  capability_overview.svg        2 x 2 overview: DP / LAY, Intact / WCF
  power_<mode>_Bf<n>.svg, power_overview.svg   Power consumption vs weather direction (intact, Bf 3 / 6)

WCF (worst case failure) = per heading the lowest capability of all bus failure cases (Bus0, Bus1, Bus2).

Usage: plot_capability.py <study dir> [prefix_a label_a prefix_b label_b]
"""
import csv
import glob
import math
import os
import sys

MODES = {"DP": "DP station keeping", "LAY": "Plough / lay: 0.8 kn + 80 t pull"}
CASES = {"Intact": "Intact", "Bus0": "Loss of port bus", "Bus1": "Loss of centre bus", "Bus2": "Loss of stbd bus",
         "WCF": "Worst case failure (bus loss)"}
COLORS = ["#1f77b4", "#d62728", "#2ca02c"]
R_MAX = 34.0  # m/s
PANEL_W, PANEL_H = 580, 620


def read(path):
    with open(path) as f:
        return [(float(r["dir"]), float(r["windSpeed"]) if int(r["Feasible"]) else 0.0, int(r["Feasible"]),
                 float(r["Bf[-]"])) for r in csv.DictReader(f, delimiter="\t")]


BF_WIND = [0.0, 1.5, 3.4, 5.4, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6]


def read_case(folder, prefix, mode, case):
    """Rows (dir, Bf, feasible[, governing case]) for a case. WCF = per heading minimum over the bus failures."""
    if case != "WCF":
        f = os.path.join(folder, f"{prefix}_{mode}_{case}.out")
        return [(d, ws, fe, case, bf) for d, ws, fe, bf in read(f)] if os.path.exists(f) else None
    files = sorted(glob.glob(os.path.join(folder, f"{prefix}_{mode}_Bus[0-9].out")))
    if not files:
        return None
    cases = [(os.path.basename(f).split("_")[-1][:-4], read(f)) for f in files]
    rows = []
    for i in range(len(cases[0][1])):
        name, row = min(((n, r[i]) for n, r in cases), key=lambda x: (x[1][2], x[1][1]))  # infeasible first
        rows.append((row[0], row[1], row[2], name, row[3]))
    return rows


def point(cx, cy, r, ang_deg, scale):
    a = math.radians(ang_deg)
    return cx + scale * r * math.sin(a), cy - scale * r * math.cos(a)


def panel(series, title, ox=0, oy=0):
    """Polar plot of max wind speed [m/s] held vs weather direction, Beaufort limits (DNV-ST-0111) as rings."""
    cx, cy, scale = ox + PANEL_W / 2, oy + 290, 240.0 / R_MAX
    out = [f'<text x="{ox + PANEL_W/2}" y="{oy + 24}" text-anchor="middle" font-size="16">{title}</text>']
    for bf in range(3, 12):
        r = BF_WIND[bf] * scale
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="none" stroke="#ccc"/>')
        out.append(f'<text x="{cx+3}" y="{cy-r+11:.1f}" font-size="9" fill="#888">Bf{bf} {BF_WIND[bf]}</text>')
    for ang in range(0, 360, 30):
        x, y = point(cx, cy, R_MAX, ang, scale)
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#ddd"/>')
        lx, ly = point(cx, cy, R_MAX + 2.2, ang, scale)
        out.append(f'<text x="{lx:.1f}" y="{ly+4:.1f}" text-anchor="middle" font-size="11">{ang}</text>')
    for i, (label, rows) in enumerate(series):
        pts = [point(cx, cy, min(ws, R_MAX), d, scale) for d, ws, *_ in rows]
        pts.append(pts[0])
        out.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="{COLORS[i]}" '
                   f'fill-opacity="0.12" stroke="{COLORS[i]}" stroke-width="2"/>')
        worst = min(rows, key=lambda r: (r[2], r[1]))
        n_inf = sum(1 for r in rows if not r[2])
        note = f"min {worst[1]:.1f} m/s (Bf {worst[4]:.1f}) at {worst[0]:.0f} deg"
        if n_inf:
            note = f"NOT FEASIBLE at {n_inf}/{len(rows)} headings"
        out.append(f'<rect x="{ox + 20}" y="{oy + PANEL_H - 70 + i*20}" width="14" height="10" fill="{COLORS[i]}"/>')
        out.append(f'<text x="{ox + 40}" y="{oy + PANEL_H - 61 + i*20}" font-size="12">{label}: {note}</text>')
    return out


def write_svg(path, elements, w, h, title=None):
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif">',
            f'<rect width="{w}" height="{h}" fill="white"/>']
    if title:
        for k, line in enumerate(title.split(" | ")):
            size, weight = (18, "bold") if k == 0 else (13, "normal")
            head.append(f'<text x="{w/2}" y="{22 + 18*k}" text-anchor="middle" font-size="{size}" '
                        f'font-weight="{weight}">{line}</text>')
    with open(path, "w") as f:
        f.write("\n".join(head + elements + ["</svg>"]))
    print("wrote", path)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    opts = [("cablelay_azi", "Azimuth option"), ("cablelay_shaft", "Shaft line option")]
    if len(sys.argv) >= 6:
        opts = [(sys.argv[2], sys.argv[3]), (sys.argv[4], sys.argv[5])]

    def series_for(mode, case):
        s = []
        for prefix, label in opts:
            rows = read_case(folder, prefix, mode, case)
            if rows:
                s.append((label, rows))
        return s

    for mode, mode_title in MODES.items():
        for case, case_title in CASES.items():
            s = series_for(mode, case)
            if s:
                write_svg(os.path.join(folder, f"capability_{mode}_{case}.svg"),
                          panel(s, f"{mode_title} - {case_title}"), PANEL_W, PANEL_H)

    overview = []
    for row, mode in enumerate(MODES):
        for col, case in enumerate(["Intact", "WCF"]):
            s = series_for(mode, case)
            if s:
                overview += panel(s, f"{MODES[mode]} - {CASES[case]}", col * PANEL_W, 50 + row * PANEL_H)
    write_svg(os.path.join(folder, "capability_overview.svg"), overview, 2 * PANEL_W, 50 + 2 * PANEL_H,
              "DP capability, max mean wind speed [m/s] - azimuth vs shaft line | Wind-wave-current relation per DNV-ST-0111 (Beaufort limits as rings), dynamic allowance 1.25")
    power_opts = list(opts)
    if os.path.exists(os.path.join(folder, "cablelay_azi_biaslow_DP_Intact_Bf3_power.out")):
        power_opts.append(("cablelay_azi_biaslow", "Azimuth option, bias on"))
        power_opts[0] = (power_opts[0][0], power_opts[0][1] + ", bias off")
    power_plots(folder, power_opts)


# ----- Power consumption polar plots -----
def read_power(folder, prefix, mode, bf):
    f = os.path.join(folder, f"{prefix}_{mode}_Intact_Bf{bf}_power.out")
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        return [(float(r["dir"]), float(r["P_total(kW)"]), int(r["Feasible"])) for r in csv.DictReader(fh, delimiter="\t")]


def power_panel(series, title, p_max, ox=0, oy=0):
    cx, cy, r_px = ox + PANEL_W / 2, oy + 290, 240.0
    step = 1000 if p_max > 3000 else 500
    ring_max = step * math.ceil(p_max / step)
    scale = r_px / ring_max
    out = [f'<text x="{ox + PANEL_W/2}" y="{oy + 24}" text-anchor="middle" font-size="16">{title}</text>']
    for p in range(step, ring_max + 1, step):
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{p*scale:.1f}" fill="none" stroke="#ccc"/>')
        out.append(f'<text x="{cx+3}" y="{cy-p*scale+12:.1f}" font-size="10" fill="#888">{p} kW</text>')
    for ang in range(0, 360, 30):
        x, y = cx + r_px * math.sin(math.radians(ang)), cy - r_px * math.cos(math.radians(ang))
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#ddd"/>')
        lx, ly = cx + (r_px + 16) * math.sin(math.radians(ang)), cy - (r_px + 16) * math.cos(math.radians(ang))
        out.append(f'<text x="{lx:.1f}" y="{ly+4:.1f}" text-anchor="middle" font-size="11">{ang}</text>')
    for i, (label, rows) in enumerate(series):
        pts = [(cx + p * scale * math.sin(math.radians(d)), cy - p * scale * math.cos(math.radians(d))) for d, p, _ in rows]
        pts.append(pts[0])
        out.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="{COLORS[i]}" '
                   f'fill-opacity="0.12" stroke="{COLORS[i]}" stroke-width="2"/>')
        mean = sum(p for _, p, _ in rows) / len(rows)
        mx = max(rows, key=lambda r: r[1])
        n_inf = sum(1 for r in rows if not r[2])
        note = f"mean {mean:.0f} kW, max {mx[1]:.0f} kW at {mx[0]:.0f} deg" + (f", {n_inf} infeasible" if n_inf else "")
        out.append(f'<rect x="{ox + 20}" y="{oy + PANEL_H - 70 + i*20}" width="14" height="10" fill="{COLORS[i]}"/>')
        out.append(f'<text x="{ox + 40}" y="{oy + PANEL_H - 61 + i*20}" font-size="12">{label}: {note}</text>')
    return out


def power_plots(folder, opts):
    overview = []
    for row, mode in enumerate(MODES):
        for col, bf in enumerate([3, 6]):
            series = [(label, r) for prefix, label in opts if (r := read_power(folder, prefix, mode, bf))]
            if not series:
                continue
            p_max = max(p for _, rows in series for _, p, _ in rows)
            title = f"{MODES[mode]} - Bf {bf}, intact"
            write_svg(os.path.join(folder, f"power_{mode}_Bf{bf}.svg"), power_panel(series, title, p_max), PANEL_W,
                      PANEL_H)
            overview += power_panel(series, title, p_max, col * PANEL_W, 50 + row * PANEL_H)
    if overview:
        write_svg(os.path.join(folder, "power_overview.svg"), overview, 2 * PANEL_W, 50 + 2 * PANEL_H,
                  "Propulsion power vs weather direction - azimuth vs shaft line | Wind-wave-current relation per DNV-ST-0111")


if __name__ == "__main__":
    main()
