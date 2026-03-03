#!/usr/bin/env python3
"""
setup_nemoh.py — Unified Nemoh 3.0 case generator

Creates a complete Nemoh case directory from either:
  - A PDStrip geomet.out file (for real ship hulls: KVLCC2, FPSO, etc.)
  - Analytical parameters (Wigley hull, rectangular barge)

Generates: mesh .dat file, Nemoh.cal, Mesh.cal, input_solver.txt,
           Mechanics/ (Kh.dat, Inertia.dat, Km.dat, Badd.dat + _correct backups)

Usage:
  # From geomet.out (most common):
  setup_nemoh.py geomet --geomet-file path/to/geomet.out -o case_dir \\
      --lpp 328.186 --beam 58.0 --draft 20.8 --mass 320437550 \\
      --gmt 5.71 --kxx 17.3 --kyy 76.5 --kzz 76.5 --zcg -2.2

  # Wigley hull monohull:
  setup_nemoh.py wigley -o wigley_mono --length 100 --beam 10 --draft 6.25

  # Wigley catamaran:
  setup_nemoh.py wigley -o wigley_cat --length 100 --beam 10 --draft 6.25 \\
      --hulld 10 --catamaran

  # Rectangular barge:
  setup_nemoh.py barge -o barge_case --length 328.186 --beam 58 --draft 20.8

  # Common options for all hull types:
  --omega-min 0.3 --omega-max 2.5 --n-omega 40
  --beta-min 0 --beta-max 350 --n-beta 36
  --qtf-omega-min 0.3 --qtf-omega-max 2.5 --n-qtf-omega 40
  --depth 0  (0=infinite)
  --rho 1025 --g 9.81
  --nx 20 --nz 10  (mesh resolution for analytical hulls)
"""

import argparse
import numpy as np
import os
import sys
import shutil


# ============================================================
# Mesh generation: geomet.out parser
# ============================================================

def parse_geomet(filepath):
    """Parse PDStrip geomet.out file. Returns list of section dicts."""
    with open(filepath) as f:
        lines = f.readlines()

    parts = lines[0].split()
    nsections = int(parts[0])
    # parts[1] may be sym flag, parts[2] is draft
    draft_file = float(parts[2]) if len(parts) > 2 else None

    sections = []
    idx = 1
    for _ in range(nsections):
        parts = lines[idx].split()
        x_pos = float(parts[0])
        npts = int(parts[1])
        flag = int(parts[2]) if len(parts) > 2 else 0
        idx += 1
        y_vals = list(map(float, lines[idx].split()))
        idx += 1
        z_vals = list(map(float, lines[idx].split()))
        idx += 1
        sections.append({
            'x': x_pos,
            'npts': npts,
            'flag': flag,
            'y': np.array(y_vals),
            'z': np.array(z_vals),
        })

    return sections, draft_file


def mesh_from_geomet(sections, min_pts=20):
    """
    Build Nemoh half-hull mesh from geomet.out sections.

    Uses y>=0 half (port side in PDStrip convention) with Isym=1.
    Adds keel centerline closure point (y=0) at each section.

    Returns: nodes list [(x,y,z),...], panels list [(n1,n2,n3,n4),...] (1-indexed)
    """
    # Filter to sections with enough points
    use_sections = [s for s in sections if s['npts'] >= min_pts]
    if len(use_sections) < 2:
        raise ValueError(f"Need at least 2 sections with >={min_pts} points, "
                         f"got {len(use_sections)}")

    # Extract y>=0 half for each section
    half_hull = []
    for sec in use_sections:
        npts = sec['npts']
        # For symmetric sections: y>=0 half is indices npts//2 to npts-1
        half_start = npts // 2
        y_half = sec['y'][half_start:npts]
        z_half = sec['z'][half_start:npts]

        # Add keel centerline closure: y=0 at deepest z
        z_keel = z_half[0]
        y_new = np.concatenate([[0.0], y_half])
        z_new = np.concatenate([[z_keel], z_half])

        half_hull.append({
            'x': sec['x'],
            'y': y_new,
            'z': z_new,
        })

    nsec = len(half_hull)
    npts_per_sec = len(half_hull[0]['y'])

    # Verify all sections have same point count
    for i, sec in enumerate(half_hull):
        if len(sec['y']) != npts_per_sec:
            raise ValueError(f"Section {i} has {len(sec['y'])} points, "
                             f"expected {npts_per_sec}")

    # Build node table (1-indexed)
    nodes = []
    for i, sec in enumerate(half_hull):
        for j in range(npts_per_sec):
            nodes.append((sec['x'], sec['y'][j], sec['z'][j]))

    def node_id(isec, jpt):
        return isec * npts_per_sec + jpt + 1

    # Build quad panels
    panels = []
    for i in range(nsec - 1):
        for j in range(npts_per_sec - 1):
            n1 = node_id(i, j)      # aft, lower
            n2 = node_id(i+1, j)    # fwd, lower
            n3 = node_id(i+1, j+1)  # fwd, upper
            n4 = node_id(i, j+1)    # aft, upper
            panels.append((n1, n2, n3, n4))

    # Verify outward normals: side panel at midship should have +y normal
    mid_sec = nsec // 2
    mid_pt = npts_per_sec - 2  # near waterline
    pidx = mid_sec * (npts_per_sec - 1) + mid_pt
    p = panels[pidx]
    p1, p2, p4 = [np.array(nodes[n-1]) for n in (p[0], p[1], p[3])]
    normal = np.cross(p2 - p1, p4 - p1)

    side_reversed = False
    if normal[1] < 0:
        # Reverse all panels: swap n2<->n4
        panels = [(n1, n4, n3, n2) for n1, n2, n3, n4 in panels]
        side_reversed = True
        print("  Normal check: reversed panel winding (now outward)")
    else:
        print("  Normal check: winding correct (outward)")

    # Verify bottom panel normal (-z)
    pidx_bot = mid_sec * (npts_per_sec - 1) + 0
    p = panels[pidx_bot]
    p1, p2, p4 = [np.array(nodes[n-1]) for n in (p[0], p[1], p[3])]
    normal_bot = np.cross(p2 - p1, p4 - p1)
    if normal_bot[2] > 0:
        print(f"  WARNING: bottom panel normal points upward ({normal_bot[2]:.3f})")

    # --- Stern transom closure panels ---
    # Close the stern (section 0) with a "curtain" of quad panels that
    # fill the flat transom rectangle from the hull profile up to z=0.
    # This ensures waterline segments exist at the transom, closing the
    # waterline contour for Nemoh's QTF waterline integral (DUOK Term 3).
    # The bow is NOT capped — bulbous/fine bows taper naturally.

    n_side_panels = len(panels)
    x_stern = half_hull[0]['x']

    # For each stern profile node below z=0, create a corresponding
    # node at (x_stern, same_y, 0). Build a mapping: j -> z0_node_id.
    z0_node_id = {}  # profile point index -> 1-indexed node ID at z=0
    for j in range(npts_per_sec):
        pnode = nodes[node_id(0, j) - 1]
        if abs(pnode[2]) < 1e-6:
            # Already at z=0 — use the existing profile node
            z0_node_id[j] = node_id(0, j)
        else:
            # Below waterline — add new node at (x_stern, y_j, 0)
            nodes.append((x_stern, pnode[1], 0.0))
            z0_node_id[j] = len(nodes)  # 1-indexed

    # Create quad panels between consecutive profile segments and z=0.
    # Each quad: profile[j] -> profile[j+1] -> z0[j+1] -> z0[j]
    # All lie in the x=x_stern plane.
    stern_cap_panels = []
    for j in range(npts_per_sec - 1):
        bot_lo = node_id(0, j)       # profile node j (lower)
        bot_hi = node_id(0, j + 1)   # profile node j+1 (higher)
        top_hi = z0_node_id[j + 1]   # z=0 node above j+1
        top_lo = z0_node_id[j]       # z=0 node above j

        # Skip degenerate panels where nodes coincide
        # (e.g. both profile nodes already at z=0)
        unique_ids = set([bot_lo, bot_hi, top_hi, top_lo])
        if len(unique_ids) < 4:
            continue

        stern_cap_panels.append((bot_lo, bot_hi, top_hi, top_lo))

    # Verify stern cap normal direction (-x = outward/aft)
    if len(stern_cap_panels) > 0:
        sp = stern_cap_panels[len(stern_cap_panels) // 2]
        sp1, sp2, sp4 = [np.array(nodes[n-1]) for n in (sp[0], sp[1], sp[3])]
        stern_normal = np.cross(sp2 - sp1, sp4 - sp1)
        if stern_normal[0] > 0:
            # Wrong direction, reverse winding: swap n2<->n4
            stern_cap_panels = [(a, d, c, b) for a, b, c, d in stern_cap_panels]
            print("  Stern cap: reversed winding (now -x normal)")
        else:
            print(f"  Stern cap: winding correct (-x normal)")

    panels.extend(stern_cap_panels)
    n_extra_nodes = len(nodes) - nsec * npts_per_sec
    print(f"  Stern transom: {len(stern_cap_panels)} closure panels, "
          f"{n_extra_nodes} extra node(s) at z=0")

    return nodes, panels


def waterplane_from_geomet(sections, min_pts=20):
    """
    Compute waterplane properties from geomet.out sections.

    Uses the waterline point (last point, z≈0) of each section's y>=0 half.
    Computes Awp, xF (center of flotation), Ix (transverse 2nd moment),
    Iy (longitudinal 2nd moment about origin), and volume-related quantities.

    Returns dict with:
        Awp:  waterplane area [m²] (full, both sides)
        xF:   longitudinal center of flotation [m] (from origin)
        Ix:   transverse 2nd moment of waterplane [m⁴]
        Iy:   longitudinal 2nd moment about origin [m⁴]
        S_x:  first moment of waterplane about y-axis [m³] (= Awp * xF)
        V:    submerged volume [m³] (from section area integration)
        zB:   z of center of buoyancy from waterline [m] (negative = below)
    """
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

    use_sections = [s for s in sections if s['npts'] >= min_pts]
    if len(use_sections) < 2:
        return None

    # Extract waterline half-breadth for each section
    wl_x = []
    wl_y = []
    half_hulls = []  # for volume/zB integration
    for sec in use_sections:
        npts = sec['npts']
        half_start = npts // 2
        y_half = sec['y'][half_start:npts]
        z_half = sec['z'][half_start:npts]
        wl_x.append(sec['x'])
        wl_y.append(y_half[-1])  # waterline point (largest y, z≈0)
        half_hulls.append({'x': sec['x'], 'y': y_half, 'z': z_half})

    wl_x = np.array(wl_x)
    wl_y = np.array(wl_y)

    # Waterplane area (full hull = 2× half)
    Awp = 2.0 * trapz(wl_y, wl_x)

    # First moment about y-axis (S_x = integral x dA over waterplane)
    S_x = 2.0 * trapz(wl_y * wl_x, wl_x)

    # Center of flotation
    xF = S_x / Awp if abs(Awp) > 1e-10 else 0.0

    # Second moment about x-axis (transverse, for BM_T)
    # Ix = 2 * (2/3) * integral y³ dx  (strip second moment about centerline)
    Ix = 2.0 * (2.0 / 3.0) * trapz(wl_y**3, wl_x)

    # Second moment about y-axis through origin (longitudinal, for BM_L)
    # Then about xF: Iy_xF = Iy_origin - Awp * xF²
    Iy = 2.0 * trapz(wl_y * wl_x**2, wl_x)

    # Submerged volume and center of buoyancy from section areas
    V_half = 0.0
    zB_num = 0.0
    for i in range(len(half_hulls) - 1):
        s0, s1 = half_hulls[i], half_hulls[i + 1]
        # Section area (half hull) = integral y dz (trapezoidal)
        A0 = abs(trapz(s0['y'], s0['z']))
        A1 = abs(trapz(s1['y'], s1['z']))
        dx = s1['x'] - s0['x']
        V_half += 0.5 * (A0 + A1) * dx
        # z-centroid of each section
        zc0 = trapz(s0['y'] * s0['z'], s0['z']) / trapz(s0['y'], s0['z']) \
            if abs(trapz(s0['y'], s0['z'])) > 1e-10 else 0.0
        zc1 = trapz(s1['y'] * s1['z'], s1['z']) / trapz(s1['y'], s1['z']) \
            if abs(trapz(s1['y'], s1['z'])) > 1e-10 else 0.0
        zB_num += 0.5 * (A0 * zc0 + A1 * zc1) * dx

    V = 2.0 * abs(V_half)
    zB = zB_num / abs(V_half) if abs(V_half) > 1e-10 else 0.0

    return {
        'Awp': Awp,
        'xF': xF,
        'Ix': Ix,
        'Iy': Iy,
        'S_x': S_x,
        'V': V,
        'zB': zB,
    }


# ============================================================
# Mesh generation: Wigley hull (analytical)
# ============================================================

def mesh_wigley(L, B, T, nx, nz, hulld=None):
    """
    Generate Wigley hull mesh: y(x,z) = (B/2)*(1-(2x/L)^2)*(1-(z/T)^2)

    If hulld is given, generates catamaran starboard hull at y_global = hulld + y_local.
    Uses Isym=1 (Nemoh mirrors to get port hull / second demihull).

    Returns: nodes list [(x,y,z),...], panels list [(n1,n2,n3,n4),...] (1-indexed)
    """
    x_nodes = np.linspace(L/2.0, -L/2.0, nx + 1)
    z_nodes = np.linspace(0.0, -T, nz + 1)

    def wigley_y(x, z):
        return (B/2.0) * (1.0 - (2.0*x/L)**2) * (1.0 - (z/T)**2)

    nodes = []
    node_idx = {}
    idx = 1
    for ix, x in enumerate(x_nodes):
        for iz, z in enumerate(z_nodes):
            y_local = wigley_y(x, z)
            y = y_local if hulld is None else hulld + y_local
            nodes.append((x, y, z))
            node_idx[(ix, iz)] = idx
            idx += 1

    panels = []
    for ix in range(nx):
        for iz in range(nz):
            n1 = node_idx[(ix, iz)]
            n2 = node_idx[(ix, iz+1)]
            n3 = node_idx[(ix+1, iz+1)]
            n4 = node_idx[(ix+1, iz)]

            # Check outward normal (+y)
            p1 = np.array(nodes[n1-1])
            p2 = np.array(nodes[n2-1])
            p4 = np.array(nodes[n4-1])
            cross = np.cross(p2 - p1, p4 - p1)
            if cross[1] < 0:
                panels.append((n1, n4, n3, n2))
            else:
                panels.append((n1, n2, n3, n4))

    return nodes, panels


# ============================================================
# Mesh generation: rectangular barge (analytical)
# ============================================================

def mesh_barge(L, B, T, nx, ny, nz):
    """
    Generate rectangular barge mesh with perfectly vertical sides.
    Half-hull (y>=0) with Isym=1. Four faces: bottom, port side, bow, stern.

    Returns: nodes list [(x,y,z),...], panels list [(n1,n2,n3,n4),...] (1-indexed)
    """
    HALF_L = L / 2.0
    HALF_B = B / 2.0

    x_vals = np.linspace(-HALF_L, HALF_L, nx + 1)
    y_vals = np.linspace(0, HALF_B, ny + 1)
    z_vals = np.linspace(-T, 0, nz + 1)

    all_nodes = []
    node_map = {}

    def add_node(x, y, z):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in node_map:
            node_map[key] = len(all_nodes) + 1
            all_nodes.append(key)
        return node_map[key]

    panels = []

    # Bottom: z=-T, normal=(0,0,-1)
    for i in range(nx):
        for j in range(ny):
            n1 = add_node(x_vals[i],   y_vals[j],   -T)
            n2 = add_node(x_vals[i+1], y_vals[j],   -T)
            n3 = add_node(x_vals[i+1], y_vals[j+1], -T)
            n4 = add_node(x_vals[i],   y_vals[j+1], -T)
            panels.append((n1, n4, n3, n2))

    # Port side: y=B/2, normal=(0,+1,0)
    for i in range(nx):
        for k in range(nz):
            n1 = add_node(x_vals[i],   HALF_B, z_vals[k])
            n2 = add_node(x_vals[i+1], HALF_B, z_vals[k])
            n3 = add_node(x_vals[i+1], HALF_B, z_vals[k+1])
            n4 = add_node(x_vals[i],   HALF_B, z_vals[k+1])
            panels.append((n1, n4, n3, n2))

    # Bow transom: x=+L/2, normal=(+1,0,0)
    for j in range(ny):
        for k in range(nz):
            n1 = add_node(HALF_L, y_vals[j],   z_vals[k])
            n2 = add_node(HALF_L, y_vals[j+1], z_vals[k])
            n3 = add_node(HALF_L, y_vals[j+1], z_vals[k+1])
            n4 = add_node(HALF_L, y_vals[j],   z_vals[k+1])
            panels.append((n1, n2, n3, n4))

    # Stern transom: x=-L/2, normal=(-1,0,0)
    for j in range(ny):
        for k in range(nz):
            n1 = add_node(-HALF_L, y_vals[j],   z_vals[k])
            n2 = add_node(-HALF_L, y_vals[j+1], z_vals[k])
            n3 = add_node(-HALF_L, y_vals[j+1], z_vals[k+1])
            n4 = add_node(-HALF_L, y_vals[j],   z_vals[k+1])
            panels.append((n1, n4, n3, n2))

    nodes = [all_nodes[i] for i in range(len(all_nodes))]
    return nodes, panels


# ============================================================
# File writers
# ============================================================

def write_mesh_dat(filepath, nodes, panels, isym=1):
    """Write Nemoh mesh file (.dat)."""
    with open(filepath, 'w') as f:
        f.write(f"    2    {isym}\n")
        for i, (x, y, z) in enumerate(nodes):
            f.write(f"  {i+1:d}      {x:.6f}       {y:.6f}      {z:.6f}\n")
        f.write("    0   0.000000   0.000000   0.000000\n")
        for n1, n2, n3, n4 in panels:
            f.write(f"  {n1:d}  {n2:d}  {n3:d}  {n4:d}\n")
        f.write("       0     0     0     0\n")
    print(f"  Mesh: {filepath} ({len(nodes)} nodes, {len(panels)} panels)")


def write_nemoh_cal(filepath, meshfile, n_nodes, n_panels,
                    rho, g, depth,
                    n_omega, omega_min, omega_max,
                    n_beta, beta_min, beta_max,
                    n_qtf_omega, qtf_omega_min, qtf_omega_max,
                    qtf_contrib=2, bidirectional=0):
    """Write Nemoh.cal configuration file."""
    dashes = '-' * 114
    with open(filepath, 'w') as f:
        f.write(f"--- Environment {dashes}\n")
        f.write(f"{rho:.1f}\t\t\t\t! RHO \t\t\t! KG/M**3 \t! Fluid specific volume\n")
        f.write(f"{g:.2f}\t\t\t\t! G\t\t\t\t! M/S**2\t! Gravity\n")
        f.write(f"{depth:.1f}\t\t\t\t\t! DEPTH\t\t\t! M\t\t! Water depth (0=infinite)\n")
        f.write(f"0.\t0.\t\t\t\t! XEFF YEFF\t\t! M\t\t! Wave measurement point\n")
        f.write(f"--- Description of floating bodies {dashes}\n")
        f.write(f"1\t\t\t\t\t! Number of bodies\n")
        f.write(f"--- Body 1 {dashes}\n")
        f.write(f"{meshfile}\t\t\t\t! Name of meshfile\n")
        f.write(f"{n_nodes} {n_panels}\t\t\t\t\t! Number of points and number of panels\n")
        f.write(f"6\t\t\t\t\t\t! Number of degrees of freedom\n")
        f.write(f"1 1. 0.\t0. 0. 0. 0.\t\t\t\t! Surge\n")
        f.write(f"1 0. 1.\t0. 0. 0. 0.\t\t\t\t! Sway\n")
        f.write(f"1 0. 0. 1. 0. 0. 0.\t\t\t\t! Heave\n")
        f.write(f"2 1. 0. 0. 0. 0. 0.\t\t\t\t! Roll about origin\n")
        f.write(f"2 0. 1. 0. 0. 0. 0.\t\t\t\t! Pitch about origin\n")
        f.write(f"2 0. 0. 1. 0. 0. 0.\t\t\t\t! Yaw about origin\n")
        f.write(f"6\t\t\t\t\t\t! Number of resulting generalised forces\n")
        f.write(f"1 1. 0.\t0. 0. 0. 0.\t\t\t\t! Force in x direction\n")
        f.write(f"1 0. 1.\t0. 0. 0. 0.\t\t\t\t! Force in y direction\n")
        f.write(f"1 0. 0. 1. 0. 0. 0.\t\t\t\t! Force in z direction\n")
        f.write(f"2 1. 0. 0. 0. 0. 0.\t\t\t\t! Moment about x\n")
        f.write(f"2 0. 1. 0. 0. 0. 0.\t\t\t\t! Moment about y\n")
        f.write(f"2 0. 0. 1. 0. 0. 0.\t\t\t\t! Moment about z\n")
        f.write(f"0\t\t\t\t\t\t! Number of lines of additional information\n")
        f.write(f"--- Load cases to be solved {dashes}\n")
        f.write(f"1 {n_omega}\t{omega_min:.4f}\t{omega_max:.4f}\t\t\t! Freq type 1=rad/s, N, Min, Max\n")
        f.write(f"{n_beta}\t{beta_min:.6f}\t{beta_max:.6f}\t\t\t! N_beta, beta_min, beta_max (degrees)\n")
        f.write(f"--- Post processing {dashes}\n")
        f.write(f"0\t0.1\t10.\t\t\t! IRF (0=no), dt, duration\n")
        f.write(f"0\t\t\t\t\t! Show pressure\n")
        f.write(f"0\t0.\t180.\t\t\t! Kochin function: N_theta, min, max\n")
        f.write(f"0\t50\t400.\t400.\t\t! Free surface: Nx, Ny, Lx, Ly\n")
        f.write(f"1\t\t\t\t\t! RAO (1=calculate)\n")
        f.write(f"1\t\t\t\t\t! output freq type 1=rad/s\n")
        f.write(f"--- QTF{dashes}\n")
        f.write(f"1\t\t\t\t\t! QTF flag (1=enable)\n")
        f.write(f"{n_qtf_omega}\t{qtf_omega_min:.4f}\t{qtf_omega_max:.4f}\t\t! N_omega_QTF, min, max\n")
        f.write(f"{bidirectional}\t\t\t\t\t! 0=unidirectional, 1=bidirectional\n")
        f.write(f"{qtf_contrib}\t\t\t\t\t! Contributing terms: 2=DUOK+HASBO\n")
        f.write(f"NA\t\t\t\t\t! FS mesh file (NA if not full QTF)\n")
        f.write(f"0\t0\t0\t\t\t! FS QTF params (not used for terms<=2)\n")
        f.write(f"0\t\t\t\t\t! Hydrostatic quadratic terms\n")
        f.write(f"1\t\t\t\t\t! Output freq type 1=rad/s\n")
        f.write(f"1\t\t\t\t\t! Include DUOK in total QTFs\n")
        f.write(f"1\t\t\t\t\t! Include HASBO in total QTFs\n")
        f.write(f"0\t\t\t\t\t! Include HASFS+ASYMP in total QTFs\n")
        f.write(f"{dashes}\n")
    print(f"  Nemoh.cal: {filepath}")


def write_mesh_cal(filepath, meshname, isym, xG, yG, zG, npanels,
                   rho, g):
    """Write Mesh.cal (used by hydrosCal)."""
    with open(filepath, 'w') as f:
        f.write(f"{meshname}\n")
        f.write(f"{isym}\n")
        f.write(f"0.0  0.0\n")
        f.write(f"{xG:.6f}  {yG:.6f}  {zG:.6f}\n")
        f.write(f"{npanels}\n")
        f.write(f"2\n")
        f.write(f"1.0\n")
        f.write(f"{rho:.1f}\n")
        f.write(f"{g:.2f}\n")
    print(f"  Mesh.cal: {filepath}")


def write_input_solver(filepath):
    """Write input_solver.txt with standard settings."""
    with open(filepath, 'w') as f:
        f.write("2\t\t\t\t! Gauss quadrature N^2, specify N=[1,4]\n")
        f.write("0.001\t\t\t! eps_zmin\n")
        f.write("1\t\t\t\t! 0=GAUSS ELIM, 1=LU DECOMP, 2=GMRES\n")
        f.write("10 1e-5 1000\t! GMRES params (restart, tol, maxiter)\n")
    print(f"  input_solver.txt: {filepath}")


def write_mechanics(mech_dir, mass, xG, yG, zG,
                    C33, C44, C55, C35=0.0,
                    kxx=None, kyy=None, kzz=None, rho=1025.0, g=9.81):
    """
    Write Mechanics/ files: Inertia.dat, Kh.dat, Km.dat, Badd.dat.
    Also writes _correct.dat backups for Kh and Inertia.

    kxx, kyy, kzz: radii of gyration about COG. If None, estimated
    from mass and stiffness.
    """
    os.makedirs(mech_dir, exist_ok=True)

    # ---- Kh.dat ----
    Kh = np.zeros((6, 6))
    Kh[2, 2] = C33
    Kh[3, 3] = C44
    Kh[4, 4] = C55
    Kh[2, 4] = C35
    Kh[4, 2] = C35

    def write_6x6(filepath, mat):
        with open(filepath, 'w') as f:
            for i in range(6):
                row = "  ".join(f"{mat[i,j] + 0.0:15.6E}" for j in range(6))
                f.write(f" {row}\n")

    kh_path = os.path.join(mech_dir, "Kh.dat")
    write_6x6(kh_path, Kh)
    shutil.copy2(kh_path, os.path.join(mech_dir, "Kh_correct.dat"))
    print(f"  Kh.dat + Kh_correct.dat (C33={C33:.4e}, C44={C44:.4e}, C55={C55:.4e})")

    # ---- Inertia.dat ----
    # 6x6 mass matrix about origin, with parallel-axis coupling from COG offset
    M = np.zeros((6, 6))
    M[0, 0] = mass
    M[1, 1] = mass
    M[2, 2] = mass

    # Translation-rotation coupling (skew-symmetric of rG = [xG, yG, zG])
    M[0, 4] = mass * zG    # surge-pitch
    M[4, 0] = mass * zG
    M[0, 5] = -mass * yG   # surge-yaw
    M[5, 0] = -mass * yG
    M[1, 3] = -mass * zG   # sway-roll
    M[3, 1] = -mass * zG
    M[1, 5] = mass * xG    # sway-yaw
    M[5, 1] = mass * xG
    M[2, 3] = mass * yG    # heave-roll
    M[3, 2] = mass * yG
    M[2, 4] = -mass * xG   # heave-pitch
    M[4, 2] = -mass * xG

    # Rotational inertia about origin (parallel axis theorem)
    # I_origin = I_cog + M*(|rG|^2*I3 - rG x rG)
    if kxx is None:
        # Estimate: use C44/(mass*g) as GM proxy
        kxx = np.sqrt(C44 / (mass * g)) if C44 > 0 else 0.1 * np.sqrt(mass)
    if kyy is None:
        kyy = np.sqrt(C55 / (mass * g)) if C55 > 0 else 0.25 * np.sqrt(mass)
    if kzz is None:
        kzz = kyy  # common assumption

    Ixx_cog = mass * kxx**2
    Iyy_cog = mass * kyy**2
    Izz_cog = mass * kzz**2

    r2 = xG**2 + yG**2 + zG**2
    M[3, 3] = Ixx_cog + mass * (r2 - xG**2)  # = Ixx_cog + mass*(yG^2 + zG^2)
    M[4, 4] = Iyy_cog + mass * (r2 - yG**2)  # = Iyy_cog + mass*(xG^2 + zG^2)
    M[5, 5] = Izz_cog + mass * (r2 - zG**2)  # = Izz_cog + mass*(xG^2 + yG^2)

    # Off-diagonal rotational coupling
    M[3, 4] = -mass * xG * yG
    M[4, 3] = -mass * xG * yG
    M[3, 5] = -mass * xG * zG
    M[5, 3] = -mass * xG * zG
    M[4, 5] = -mass * yG * zG
    M[5, 4] = -mass * yG * zG

    inertia_path = os.path.join(mech_dir, "Inertia.dat")
    write_6x6(inertia_path, M)
    shutil.copy2(inertia_path, os.path.join(mech_dir, "Inertia_correct.dat"))
    print(f"  Inertia.dat + Inertia_correct.dat (mass={mass:.1f}, kxx={kxx:.2f}, kyy={kyy:.2f})")

    # ---- Km.dat (mooring stiffness) ----
    km_path = os.path.join(mech_dir, "Km.dat")
    write_6x6(km_path, np.zeros((6, 6)))

    # ---- Badd.dat (additional damping) ----
    badd_path = os.path.join(mech_dir, "Badd.dat")
    write_6x6(badd_path, np.zeros((6, 6)))

    print(f"  Km.dat, Badd.dat (zeros)")


# ============================================================
# Hydrostatics computation
# ============================================================

def wigley_hydrostatics(L, B, T, rho, g, hulld=None):
    """
    Compute Wigley hull hydrostatic parameters analytically.
    y(x,z) = (B/2)*(1-(2x/L)^2)*(1-(z/T)^2)

    Returns dict with: mass, V, Awp, Ix, Iy, zB, C33, C44, C55, zG, kxx, kyy, kzz
    """
    # Volume (single demihull, both sides)
    # V = 2 * integral_{-L/2}^{L/2} integral_{-T}^{0} y(x,z) dz dx
    # V = 2 * (B/2) * integral (1-(2x/L)^2) dx * integral (1-(z/T)^2) dz
    # integral_{-L/2}^{L/2} (1-(2x/L)^2) dx = L * (1 - 1/3) = 2L/3
    # integral_{-T}^{0} (1-(z/T)^2) dz = T * (1 - 1/3) = 2T/3
    V_demi = 2 * (B/2) * (2*L/3) * (2*T/3)  # = 4BLT/9

    # Waterplane area (single demihull, both sides)
    # Awp = 2 * integral_{-L/2}^{L/2} y(x,0) dx = 2*(B/2)*integral (1-(2x/L)^2) dx
    # = B * 2L/3 = 2BL/3
    Awp_demi = 2 * B * L / 3

    # Center of buoyancy (z-coordinate)
    # zB = (1/V) * 2 * integral y(x,z) * z dz dx
    # integral_{-T}^{0} z*(1-(z/T)^2) dz = [-T^2/2 - (-T^2/4)] = ... computed:
    # = [z^2/2 - z^4/(4T^2)]_{-T}^{0} = 0 - (T^2/2 - T^2/4) = -T^2/4
    # Actually: integral_{-T}^{0} z*(1-(z/T)^2) dz
    # = [z^2/2 - z^4/(4T^2)]_{-T}^{0}
    # = 0 - (T^2/2 - T^4/(4T^2)) = -(T^2/2 - T^2/4) = -T^2/4
    # V = 2*(B/2)*(2L/3)*(-1) * (-T^2/4) ... no, let me redo
    # Numerator = 2 * integral_{-L/2}^{L/2} integral_{-T}^{0} y(x,z)*z dz dx
    # = 2*(B/2) * (2L/3) * integral_{-T}^{0} z*(1-(z/T)^2) dz
    # integral_{-T}^{0} z*(1-(z/T)^2) dz = integral z dz - integral z^3/T^2 dz
    # = [z^2/2]_{-T}^{0} - [z^4/(4T^2)]_{-T}^{0}
    # = (0 - T^2/2) - (0 - T^2/4) = -T^2/2 + T^2/4 = -T^2/4
    # So numerator = 2*(B/2)*(2L/3)*(-T^2/4) = -(2BLT^2)/(12)
    # zB = numerator / V = [-(2BLT^2)/12] / [2BLT/9] = -(9T)/(12*1) * T/T  hmm
    # = -(2BLT^2/12) / (2BLT/9) = -(T/12) * 9 = -9T/12 = -3T/4
    # Wait that can't be right... for a Wigley hull zB should be around -0.6T to -0.4T
    # Let me recompute more carefully
    # zB = (1/V) * integral over hull z dV
    # = (2/V) * int_{-L/2}^{L/2} int_{-T}^{0} z * y(x,z) dz dx
    # = (2/V) * (B/2) * int_{-L/2}^{L/2} (1-(2x/L)^2) dx * int_{-T}^{0} z*(1-(z/T)^2) dz
    # = (2/V) * (B/2) * (2L/3) * (-T^2/4)
    # = (2 * B/2 * 2L/3 * (-T^2/4)) / (2BLT/9)
    # = (B * 2L/3 * (-T^2/4)) / (2BLT/9)
    # = (2L/3 * (-T^2/4)) / (2LT/9)
    # = (-T/4) / (2/9 * T * 3/(2L) * L) ... let me just compute numerically
    # = (-2BLT^2/12) / (2BLT/9) = (-T/12) * (9/1) = -3T/4
    # Hmm, that gives zB = -3T/4 = -4.6875 for T=6.25
    # Actually for a Wigley hull this might be correct; the shape tapers strongly
    # toward the waterline so most volume is deep.
    # Let me verify numerically
    from scipy import integrate
    def integrand_z(z):
        return z * (1 - (z/T)**2)
    Iz, _ = integrate.quad(integrand_z, -T, 0)
    def integrand_1(z):
        return (1 - (z/T)**2)
    I1, _ = integrate.quad(integrand_1, -T, 0)
    zB = Iz / I1
    # This gives zB relative to waterline (z=0 at waterline, negative down)

    # Waterplane second moments
    # Ix = 2 * integral_{-L/2}^{L/2} y(x,0)^3 / 3 dx  (about y=0)
    # For half-hull with symmetry: Ix = 2 * (1/3) * integral (B/2)^3 * (1-(2x/L)^2)^3 dx
    from scipy.integrate import quad
    def y_wl(x):
        return (B/2) * (1 - (2*x/L)**2)
    Ix_demi, _ = quad(lambda x: 2 * y_wl(x)**3 / 3, -L/2, L/2)

    # Iy = 2 * integral_{-L/2}^{L/2} x^2 * y(x,0) dx
    Iy_demi, _ = quad(lambda x: 2 * x**2 * y_wl(x), -L/2, L/2)

    # Metacentric heights
    BM_T = Ix_demi / V_demi
    BM_L = Iy_demi / V_demi
    KB = -zB  # KB from keel (positive up from keel = T + zB ... no)
    # zB is measured from waterline (negative). KB from keel = T + zB (since keel is at z=-T)
    KB_from_keel = T + zB

    # For Wigley hull test cases, KG = KB (COG at center of buoyancy)
    KG_from_keel = KB_from_keel
    zG = zB  # zG from waterline (negative since below waterline)

    GM_T = KB_from_keel + BM_T - KG_from_keel
    GM_L = KB_from_keel + BM_L - KG_from_keel

    mass_demi = rho * V_demi

    C33_demi = rho * g * Awp_demi
    C44_demi = rho * g * V_demi * GM_T
    C55_demi = rho * g * V_demi * GM_L

    # Radii of gyration (typical for ships)
    kxx = 0.35 * B
    kyy = 0.25 * L
    kzz = 0.25 * L

    result = {
        'mass': mass_demi, 'V': V_demi, 'Awp': Awp_demi,
        'Ix': Ix_demi, 'Iy': Iy_demi,
        'zB': zB, 'zG': zG,
        'KB': KB_from_keel, 'BM_T': BM_T, 'BM_L': BM_L,
        'GM_T': GM_T, 'GM_L': GM_L,
        'C33': C33_demi, 'C44': C44_demi, 'C55': C55_demi,
        'kxx': kxx, 'kyy': kyy, 'kzz': kzz,
    }

    if hulld is not None:
        # Catamaran: double everything, adjust roll stiffness for hull spacing
        mass_cat = 2 * mass_demi
        V_cat = 2 * V_demi
        C33_cat = 2 * C33_demi
        Ix_cat = 2 * (Ix_demi + Awp_demi * hulld**2)
        BM_T_cat = Ix_cat / V_cat
        GM_T_cat = KB_from_keel + BM_T_cat - KG_from_keel
        C44_cat = rho * g * V_cat * GM_T_cat
        Iy_cat = 2 * Iy_demi
        BM_L_cat = Iy_cat / V_cat
        GM_L_cat = KB_from_keel + BM_L_cat - KG_from_keel
        C55_cat = rho * g * V_cat * GM_L_cat

        kxx_cat = np.sqrt(kxx**2 + hulld**2)
        kyy_cat = kyy
        kzz_cat = np.sqrt(kzz**2 + hulld**2)

        result.update({
            'mass': mass_cat, 'V': V_cat,
            'C33': C33_cat, 'C44': C44_cat, 'C55': C55_cat,
            'kxx': kxx_cat, 'kyy': kyy_cat, 'kzz': kzz_cat,
            'GM_T': GM_T_cat, 'GM_L': GM_L_cat,
        })

    return result


def barge_hydrostatics(L, B, T, rho, g):
    """Compute rectangular barge hydrostatic parameters."""
    V = L * B * T
    mass = rho * V
    Awp = L * B
    zB = -T / 2.0
    zG = -T / 2.0  # COG at center of buoyancy for stability

    Ix = L * B**3 / 12.0
    Iy = B * L**3 / 12.0
    BM_T = Ix / V
    BM_L = Iy / V
    KB_from_keel = T / 2.0
    KG_from_keel = T / 2.0
    GM_T = KB_from_keel + BM_T - KG_from_keel
    GM_L = KB_from_keel + BM_L - KG_from_keel

    C33 = rho * g * Awp
    C44 = rho * g * V * GM_T
    C55 = rho * g * V * GM_L

    kxx = B / np.sqrt(12)
    kyy = L / np.sqrt(12)
    kzz = np.sqrt(L**2 + B**2) / np.sqrt(12)

    return {
        'mass': mass, 'V': V, 'Awp': Awp,
        'Ix': Ix, 'Iy': Iy,
        'zB': zB, 'zG': zG,
        'C33': C33, 'C44': C44, 'C55': C55,
        'kxx': kxx, 'kyy': kyy, 'kzz': kzz,
    }


# ============================================================
# Subcommand: geomet
# ============================================================

def cmd_geomet(args):
    """Build Nemoh case from PDStrip geomet.out."""
    outdir = os.path.abspath(args.output)
    os.makedirs(outdir, exist_ok=True)

    print(f"=== setup_nemoh: geomet -> {outdir} ===")

    # Parse geometry
    sections, draft_file = parse_geomet(args.geomet_file)
    print(f"  Parsed {len(sections)} sections from {args.geomet_file}")
    print(f"  x range: [{sections[0]['x']:.1f}, {sections[-1]['x']:.1f}]")

    # Build mesh
    nodes, panels = mesh_from_geomet(sections, min_pts=args.min_section_pts)
    meshname = args.mesh_name or os.path.basename(outdir)
    meshfile = f"{meshname}.dat"
    write_mesh_dat(os.path.join(outdir, meshfile), nodes, panels)

    # Required hull parameters
    Lpp = args.lpp
    beam = args.beam
    draft = args.draft
    mass = args.mass
    rho = args.rho
    g = args.g

    # Hydrostatics from waterplane integration
    V = mass / rho
    zG = args.zcg  # z of COG from waterline (negative = below WL)
    wp = waterplane_from_geomet(sections, min_pts=args.min_section_pts)

    if wp is not None:
        Awp = wp['Awp']
        xF = wp['xF']
        KB_est = draft + wp['zB']  # KB from keel (zB is negative from WL)
        KG = draft + zG            # KG from keel
        Cwp = Awp / (Lpp * beam) if Lpp * beam > 0 else 0.0
        print(f"  Waterplane: Awp={Awp:.1f} m² (Cwp={Cwp:.3f}), "
              f"xF={xF:.2f} m, KB={KB_est:.2f} m")

    # C33: heave stiffness
    if args.c33:
        C33 = args.c33
    elif wp is not None:
        C33 = rho * g * Awp
    else:
        C33 = rho * g * Lpp * beam * 0.85  # fallback

    # C44: roll stiffness (from GMT, or user-provided, or estimate)
    if args.gmt is not None:
        C44 = rho * g * V * args.gmt
    elif args.c44:
        C44 = args.c44
    elif wp is not None:
        BM_T = wp['Ix'] / V
        GM_T = KB_est + BM_T - KG
        C44 = rho * g * V * GM_T
        print(f"  WARNING: C44 estimated from waterplane (BM_T={BM_T:.2f}, "
              f"GM_T={GM_T:.2f}). Provide --gmt for accuracy.")
    else:
        C44 = 0.0

    # C55: pitch stiffness
    if args.c55:
        C55 = args.c55
    elif wp is not None:
        # C55 = rho*g*Iy + rho*g*V*(zB - zG)  (Iy about origin)
        C55 = rho * g * wp['Iy'] + rho * g * V * (wp['zB'] - zG)
    else:
        # Crude fallback
        Awp_est = Lpp * beam * 0.85
        Iy_est = Awp_est * Lpp**2 / 12.0
        BM_L = Iy_est / V
        KB_est2 = draft / 2.0
        GM_L = KB_est2 + BM_L - (draft + zG)
        C55 = rho * g * V * GM_L

    # C35: heave-pitch coupling
    # Nemoh convention: C35 = -rho*g * integral(x dA) = -rho*g * S_x
    # See Nemoh/Mesh/hydre.f90 VOLELMT subroutine, line ~585:
    #   KH(3,5) = -RHO*G*SF*P0G(1)
    if args.c35:
        C35 = args.c35
    elif wp is not None:
        C35 = -rho * g * wp['S_x']
    else:
        C35 = 0.0

    # COG
    xG = args.xcg if args.xcg else 0.0
    yG = 0.0

    # Write all config files
    write_nemoh_cal(
        os.path.join(outdir, "Nemoh.cal"),
        meshfile, len(nodes), len(panels),
        rho, g, args.depth,
        args.n_omega, args.omega_min, args.omega_max,
        args.n_beta, args.beta_min, args.beta_max,
        args.n_qtf_omega, args.qtf_omega_min, args.qtf_omega_max,
        qtf_contrib=args.qtf_contrib,
        bidirectional=1 if args.bidirectional else 0,
    )

    write_mesh_cal(
        os.path.join(outdir, "Mesh.cal"),
        meshfile, 1, xG, yG, zG, len(panels), rho, g,
    )

    write_input_solver(os.path.join(outdir, "input_solver.txt"))

    write_mechanics(
        os.path.join(outdir, "Mechanics"),
        mass, xG, yG, zG,
        C33, C44, C55, C35,
        kxx=args.kxx, kyy=args.kyy, kzz=args.kzz,
        rho=rho, g=g,
    )

    print_summary(outdir, meshfile, len(nodes), len(panels), args)


# ============================================================
# Subcommand: wigley
# ============================================================

def cmd_wigley(args):
    """Build Nemoh case for Wigley hull."""
    outdir = os.path.abspath(args.output)
    os.makedirs(outdir, exist_ok=True)

    is_cat = args.catamaran
    hulld = args.hulld if is_cat else None

    label = "catamaran" if is_cat else "monohull"
    print(f"=== setup_nemoh: Wigley {label} -> {outdir} ===")

    L, B, T = args.length, args.beam, args.draft
    rho, g = args.rho, args.g

    # Generate mesh
    nodes, panels = mesh_wigley(L, B, T, args.nx, args.nz, hulld=hulld)
    meshname = args.mesh_name or ("wigley_cat" if is_cat else "wigley")
    meshfile = f"{meshname}.dat"
    write_mesh_dat(os.path.join(outdir, meshfile), nodes, panels)

    # Compute hydrostatics
    hydro = wigley_hydrostatics(L, B, T, rho, g, hulld=hulld)
    mass = hydro['mass']
    zG = hydro['zG']
    xG = 0.0  # symmetric hull
    yG = 0.0

    print(f"  Hydrostatics: mass={mass:.1f}, V={hydro['V']:.3f}, zG={zG:.4f}")
    print(f"  GM_T={hydro['GM_T']:.4f}, GM_L={hydro['GM_L']:.4f}")
    print(f"  C33={hydro['C33']:.1f}, C44={hydro['C44']:.1f}, C55={hydro['C55']:.1f}")

    # Write config files
    write_nemoh_cal(
        os.path.join(outdir, "Nemoh.cal"),
        meshfile, len(nodes), len(panels),
        rho, g, args.depth,
        args.n_omega, args.omega_min, args.omega_max,
        args.n_beta, args.beta_min, args.beta_max,
        args.n_qtf_omega, args.qtf_omega_min, args.qtf_omega_max,
        qtf_contrib=args.qtf_contrib,
        bidirectional=1 if args.bidirectional else 0,
    )

    write_mesh_cal(
        os.path.join(outdir, "Mesh.cal"),
        meshfile, 1, xG, yG, zG, len(panels), rho, g,
    )

    write_input_solver(os.path.join(outdir, "input_solver.txt"))

    write_mechanics(
        os.path.join(outdir, "Mechanics"),
        mass, xG, yG, zG,
        hydro['C33'], hydro['C44'], hydro['C55'],
        kxx=hydro['kxx'], kyy=hydro['kyy'], kzz=hydro['kzz'],
        rho=rho, g=g,
    )

    print_summary(outdir, meshfile, len(nodes), len(panels), args)


# ============================================================
# Subcommand: barge
# ============================================================

def cmd_barge(args):
    """Build Nemoh case for rectangular barge."""
    outdir = os.path.abspath(args.output)
    os.makedirs(outdir, exist_ok=True)

    print(f"=== setup_nemoh: barge -> {outdir} ===")

    L, B, T = args.length, args.beam, args.draft
    rho, g = args.rho, args.g

    nodes, panels = mesh_barge(L, B, T, args.nx, args.ny, args.nz)
    meshname = args.mesh_name or "barge"
    meshfile = f"{meshname}.dat"
    write_mesh_dat(os.path.join(outdir, meshfile), nodes, panels)

    hydro = barge_hydrostatics(L, B, T, rho, g)
    mass = hydro['mass']
    zG = hydro['zG']

    print(f"  Hydrostatics: mass={mass:.1f}, V={hydro['V']:.3f}, zG={zG:.4f}")
    print(f"  C33={hydro['C33']:.1f}, C44={hydro['C44']:.1f}, C55={hydro['C55']:.1f}")

    write_nemoh_cal(
        os.path.join(outdir, "Nemoh.cal"),
        meshfile, len(nodes), len(panels),
        rho, g, args.depth,
        args.n_omega, args.omega_min, args.omega_max,
        args.n_beta, args.beta_min, args.beta_max,
        args.n_qtf_omega, args.qtf_omega_min, args.qtf_omega_max,
        qtf_contrib=args.qtf_contrib,
        bidirectional=1 if args.bidirectional else 0,
    )

    write_mesh_cal(
        os.path.join(outdir, "Mesh.cal"),
        meshfile, 1, 0.0, 0.0, zG, len(panels), rho, g,
    )

    write_input_solver(os.path.join(outdir, "input_solver.txt"))

    write_mechanics(
        os.path.join(outdir, "Mechanics"),
        mass, 0.0, 0.0, zG,
        hydro['C33'], hydro['C44'], hydro['C55'],
        kxx=hydro['kxx'], kyy=hydro['kyy'], kzz=hydro['kzz'],
        rho=rho, g=g,
    )

    print_summary(outdir, meshfile, len(nodes), len(panels), args)


# ============================================================
# Helpers
# ============================================================

def print_summary(outdir, meshfile, nnodes, npanels, args):
    """Print run instructions."""
    print(f"\n{'='*60}")
    print(f"Nemoh case ready: {outdir}")
    print(f"{'='*60}")
    print(f"  Mesh: {meshfile} ({nnodes} nodes, {npanels} panels, Isym=1)")
    print(f"  First-order: {args.n_omega} freq, w=[{args.omega_min:.4f}, {args.omega_max:.4f}]")
    print(f"  QTF: {args.n_qtf_omega} freq, w=[{args.qtf_omega_min:.4f}, {args.qtf_omega_max:.4f}]")
    print(f"  Headings: {args.n_beta}, beta=[{args.beta_min:.1f}, {args.beta_max:.1f}] deg")
    print(f"\nTo run:")
    print(f"  run_nemoh.sh {outdir}")
    print(f"  # or: run_nemoh.sh -v {outdir}")


# ============================================================
# Argument parser
# ============================================================

def add_common_args(parser):
    """Add arguments common to all subcommands."""
    parser.add_argument('-o', '--output', required=True,
                        help='Output case directory')
    parser.add_argument('--mesh-name', default=None,
                        help='Mesh filename base (default: auto from hull type)')
    parser.add_argument('--rho', type=float, default=1025.0,
                        help='Water density [kg/m^3] (default: 1025)')
    parser.add_argument('--g', type=float, default=9.81,
                        help='Gravity [m/s^2] (default: 9.81)')
    parser.add_argument('--depth', type=float, default=0.0,
                        help='Water depth [m] (0=infinite, default: 0)')

    # First-order frequency range
    parser.add_argument('--omega-min', type=float, default=0.3,
                        help='Min frequency [rad/s] (default: 0.3)')
    parser.add_argument('--omega-max', type=float, default=2.5,
                        help='Max frequency [rad/s] (default: 2.5)')
    parser.add_argument('--n-omega', type=int, default=40,
                        help='Number of first-order frequencies (default: 40)')

    # Wave headings
    parser.add_argument('--beta-min', type=float, default=0.0,
                        help='Min wave heading [deg] (default: 0)')
    parser.add_argument('--beta-max', type=float, default=350.0,
                        help='Max wave heading [deg] (default: 350)')
    parser.add_argument('--n-beta', type=int, default=36,
                        help='Number of headings (default: 36)')

    # QTF frequency range
    parser.add_argument('--qtf-omega-min', type=float, default=None,
                        help='QTF min frequency (default: same as --omega-min)')
    parser.add_argument('--qtf-omega-max', type=float, default=None,
                        help='QTF max frequency (default: same as --omega-max)')
    parser.add_argument('--n-qtf-omega', type=int, default=None,
                        help='Number of QTF frequencies (default: same as --n-omega)')
    parser.add_argument('--qtf-contrib', type=int, default=2,
                        help='QTF contributing terms: 2=DUOK+HASBO (default: 2)')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Enable bidirectional QTF')


def main():
    parser = argparse.ArgumentParser(
        description='setup_nemoh.py - Unified Nemoh case generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Hull type')

    # ---- geomet subcommand ----
    p_geom = subparsers.add_parser('geomet', help='Build from PDStrip geomet.out')
    add_common_args(p_geom)
    p_geom.add_argument('--geomet-file', required=True,
                        help='Path to geomet.out file')
    p_geom.add_argument('--lpp', type=float, required=True,
                        help='Length between perpendiculars [m]')
    p_geom.add_argument('--beam', type=float, required=True,
                        help='Beam (breadth) [m]')
    p_geom.add_argument('--draft', type=float, required=True,
                        help='Draft [m]')
    p_geom.add_argument('--mass', type=float, required=True,
                        help='Mass [kg]')
    p_geom.add_argument('--zcg', type=float, default=-2.0,
                        help='z of COG from waterline (negative=below WL, default: -2.0)')
    p_geom.add_argument('--xcg', type=float, default=0.0,
                        help='x of COG (default: 0.0)')
    p_geom.add_argument('--gmt', type=float, default=None,
                        help='Transverse metacentric height GM_T [m]')
    p_geom.add_argument('--c33', type=float, default=None,
                        help='Heave stiffness C33 [N/m]')
    p_geom.add_argument('--c44', type=float, default=None,
                        help='Roll stiffness C44 [N-m/rad]')
    p_geom.add_argument('--c55', type=float, default=None,
                        help='Pitch stiffness C55 [N-m/rad]')
    p_geom.add_argument('--c35', type=float, default=0.0,
                        help='Heave-pitch coupling C35 [N/rad] (default: 0)')
    p_geom.add_argument('--kxx', type=float, default=None,
                        help='Roll radius of gyration about COG [m]')
    p_geom.add_argument('--kyy', type=float, default=None,
                        help='Pitch radius of gyration about COG [m]')
    p_geom.add_argument('--kzz', type=float, default=None,
                        help='Yaw radius of gyration about COG [m]')
    p_geom.add_argument('--min-section-pts', type=int, default=20,
                        help='Min points per section to include (default: 20)')

    # ---- wigley subcommand ----
    p_wig = subparsers.add_parser('wigley', help='Wigley hull (analytical)')
    add_common_args(p_wig)
    p_wig.add_argument('--length', type=float, required=True, help='Hull length L [m]')
    p_wig.add_argument('--beam', type=float, required=True, help='Hull beam B [m]')
    p_wig.add_argument('--draft', type=float, required=True, help='Hull draft T [m]')
    p_wig.add_argument('--nx', type=int, default=20, help='Panels in x (default: 20)')
    p_wig.add_argument('--nz', type=int, default=10, help='Panels in z (default: 10)')
    p_wig.add_argument('--catamaran', action='store_true', help='Catamaran mode')
    p_wig.add_argument('--hulld', type=float, default=10.0,
                        help='Hull center-to-center distance [m] (default: 10)')

    # ---- barge subcommand ----
    p_bar = subparsers.add_parser('barge', help='Rectangular barge (analytical)')
    add_common_args(p_bar)
    p_bar.add_argument('--length', type=float, required=True, help='Barge length L [m]')
    p_bar.add_argument('--beam', type=float, required=True, help='Barge beam B [m]')
    p_bar.add_argument('--draft', type=float, required=True, help='Barge draft T [m]')
    p_bar.add_argument('--nx', type=int, default=60, help='Panels in x (default: 60)')
    p_bar.add_argument('--ny', type=int, default=10, help='Panels in y (default: 10)')
    p_bar.add_argument('--nz', type=int, default=4, help='Panels in z (default: 4)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Default QTF freq to match first-order
    if args.qtf_omega_min is None:
        args.qtf_omega_min = args.omega_min
    if args.qtf_omega_max is None:
        args.qtf_omega_max = args.omega_max
    if args.n_qtf_omega is None:
        args.n_qtf_omega = args.n_omega

    if args.command == 'geomet':
        cmd_geomet(args)
    elif args.command == 'wigley':
        cmd_wigley(args)
    elif args.command == 'barge':
        cmd_barge(args)


if __name__ == '__main__':
    main()
