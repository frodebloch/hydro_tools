#!/usr/bin/env python3
"""
Visualize a Nemoh mesh from a case directory.

Reads mesh/Mesh.tec (post-hydrosCal) or the raw .dat mesh file
(pre-hydrosCal, referenced in Nemoh.cal).

Usage:
    plot_mesh.py CASE_DIR [options]

Examples:
    plot_mesh.py /path/to/csov_nemoh       # 3D wireframe + normals
    plot_mesh.py /path/to/csov_nemoh --full # mirror to show full hull
    plot_mesh.py /path/to/csov_nemoh -o mesh.png  # save to file
"""

import argparse
import os
import sys
import re
import numpy as np


def read_mesh_tec(filepath):
    """Read Nemoh mesh/Mesh.tec (Tecplot FE format)."""
    nodes = []
    panels = []
    normals_per_node = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse header: ZONE N=nnn, E=eee
    n_nodes = 0
    n_panels = 0
    header_end = 0
    for i, line in enumerate(lines):
        m = re.search(r'N=\s*(\d+)\s*,\s*E=\s*(\d+)', line)
        if m:
            n_nodes = int(m.group(1))
            n_panels = int(m.group(2))
            header_end = i + 1
            break

    if n_nodes == 0:
        raise ValueError(f"Could not parse ZONE header in {filepath}")

    # Read nodes: X Y Z NX NY NZ A
    for i in range(header_end, header_end + n_nodes):
        vals = lines[i].split()
        x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
        nx, ny, nz = float(vals[3]), float(vals[4]), float(vals[5])
        nodes.append((x, y, z))
        normals_per_node.append((nx, ny, nz))

    # Read panels: 4 node indices (1-based)
    panel_start = header_end + n_nodes
    for i in range(panel_start, panel_start + n_panels):
        vals = lines[i].split()
        panels.append(tuple(int(v) for v in vals[:4]))

    return np.array(nodes), panels, np.array(normals_per_node)


def read_mesh_dat(filepath):
    """Read Nemoh raw mesh .dat file."""
    nodes = []
    panels = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # First line: header (2 isym) or similar
    idx = 1  # skip header

    # Read nodes until we hit 0 0 0 0 line
    while idx < len(lines):
        vals = lines[idx].split()
        if len(vals) >= 4 and float(vals[0]) == 0 and float(vals[1]) == 0:
            idx += 1
            break
        if len(vals) >= 4:
            x, y, z = float(vals[1]), float(vals[2]), float(vals[3])
            nodes.append((x, y, z))
        idx += 1

    # Read panels until we hit 0 0 0 0 line
    while idx < len(lines):
        vals = lines[idx].split()
        if len(vals) >= 4:
            n1, n2, n3, n4 = int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])
            if n1 == 0 and n2 == 0:
                break
            panels.append((n1, n2, n3, n4))
        idx += 1

    return np.array(nodes), panels, None


def compute_panel_normals(nodes, panels):
    """Compute panel centroids and outward normals."""
    centroids = []
    normals = []

    for p in panels:
        # 0-indexed
        p1 = nodes[p[0] - 1]
        p2 = nodes[p[1] - 1]
        p3 = nodes[p[2] - 1]
        p4 = nodes[p[3] - 1]

        centroid = 0.25 * (p1 + p2 + p3 + p4)

        # Normal from cross product of diagonals
        d1 = p3 - p1
        d2 = p4 - p2
        n = np.cross(d1, d2)
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            n = n / norm

        centroids.append(centroid)
        normals.append(n)

    return np.array(centroids), np.array(normals)


def classify_panels(centroids, normals, nodes, panels):
    """Classify panels by type based on normal direction and position."""
    labels = []
    x_min = nodes[:, 0].min()
    x_max = nodes[:, 0].max()

    for i, (c, n) in enumerate(zip(centroids, normals)):
        # Check if this is a transom/end-cap panel (mostly x-facing normal)
        if abs(n[0]) > 0.7:
            if c[0] < x_min + 0.1 * (x_max - x_min):
                labels.append('stern_cap')
            elif c[0] > x_max - 0.1 * (x_max - x_min):
                labels.append('bow_cap')
            else:
                labels.append('hull')
        elif abs(n[2]) > 0.7:
            labels.append('bottom')
        else:
            labels.append('hull')

    return labels


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Nemoh mesh from a case directory')
    parser.add_argument('case_dir', help='Nemoh case directory')
    parser.add_argument('-o', '--output', help='Save plot to file (e.g., mesh.png)')
    parser.add_argument('--full', action='store_true',
                        help='Mirror half-hull to show full hull (Isym=1)')
    parser.add_argument('--normals', action='store_true',
                        help='Show panel normal vectors')
    parser.add_argument('--normal-scale', type=float, default=2.0,
                        help='Normal arrow length scale (default: 2.0)')
    parser.add_argument('--elev', type=float, default=25,
                        help='View elevation angle (default: 25)')
    parser.add_argument('--azim', type=float, default=-135,
                        help='View azimuth angle (default: -135)')
    parser.add_argument('--no-waterline', action='store_true',
                        help='Do not draw waterline at z=0')
    args = parser.parse_args()

    case_dir = args.case_dir

    # Try to read mesh/Mesh.tec first (post-hydrosCal, has normals)
    mesh_tec = os.path.join(case_dir, 'mesh', 'Mesh.tec')
    mesh_dat = None

    # Find raw .dat mesh file from Nemoh.cal
    nemoh_cal = os.path.join(case_dir, 'Nemoh.cal')
    if os.path.exists(nemoh_cal):
        with open(nemoh_cal) as f:
            for line in f:
                line_s = line.strip()
                if line_s.startswith('---'):
                    continue
                candidate = line_s.split('!')[0].strip()
                if candidate.endswith('.dat'):
                    mesh_dat = os.path.join(case_dir, candidate)
                    break

    # Read mesh
    if os.path.exists(mesh_tec):
        print(f"Reading {mesh_tec}")
        nodes, panels, node_normals = read_mesh_tec(mesh_tec)
        source = 'Mesh.tec'
    elif mesh_dat and os.path.exists(mesh_dat):
        print(f"Reading {mesh_dat}")
        nodes, panels, node_normals = read_mesh_dat(mesh_dat)
        source = os.path.basename(mesh_dat)
    else:
        print(f"Error: No mesh file found in {case_dir}")
        print(f"  Tried: {mesh_tec}")
        if mesh_dat:
            print(f"  Tried: {mesh_dat}")
        sys.exit(1)

    print(f"  {len(nodes)} nodes, {len(panels)} panels")

    # Compute panel properties
    centroids, normals = compute_panel_normals(nodes, panels)
    labels = classify_panels(centroids, normals, nodes, panels)

    # Count by type
    from collections import Counter
    counts = Counter(labels)
    for lbl in ['hull', 'bottom', 'stern_cap', 'bow_cap']:
        if lbl in counts:
            print(f"  {lbl}: {counts[lbl]} panels")

    # Import matplotlib
    import matplotlib
    if args.output:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Color map for panel types
    colors = {
        'hull': (0.6, 0.8, 1.0, 0.5),
        'bottom': (0.5, 0.7, 0.9, 0.5),
        'stern_cap': (1.0, 0.4, 0.4, 0.7),
        'bow_cap': (0.4, 1.0, 0.4, 0.7),
    }

    def draw_panels(node_arr, panel_list, label_list, mirror=False):
        """Draw panels as a Poly3DCollection."""
        polys = []
        face_colors = []
        for i, p in enumerate(panel_list):
            verts = []
            for nid in p:
                pt = node_arr[nid - 1].copy()
                if mirror:
                    pt[1] = -pt[1]
                verts.append(pt)
            polys.append(verts)
            face_colors.append(colors.get(label_list[i], (0.7, 0.7, 0.7, 0.4)))

        pc = Poly3DCollection(polys, linewidths=0.3, edgecolors='k', alpha=0.5)
        pc.set_facecolor(face_colors)
        ax.add_collection3d(pc)

    # Draw half hull (port side)
    draw_panels(nodes, panels, labels, mirror=False)

    # Mirror for full hull
    if args.full:
        draw_panels(nodes, panels, labels, mirror=True)

    # Draw normals
    if args.normals:
        scale = args.normal_scale
        for i, (c, n, lbl) in enumerate(zip(centroids, normals, labels)):
            color = 'red' if lbl == 'stern_cap' else ('green' if lbl == 'bow_cap' else 'blue')
            ax.quiver(c[0], c[1], c[2], n[0]*scale, n[1]*scale, n[2]*scale,
                      color=color, linewidth=0.5, arrow_length_ratio=0.2)
            if args.full:
                ax.quiver(c[0], -c[1], c[2], n[0]*scale, -n[1]*scale, n[2]*scale,
                          color=color, linewidth=0.5, arrow_length_ratio=0.2)

    # Waterline at z=0
    if not args.no_waterline:
        x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
        y_max = nodes[:, 1].max()
        margin = 5
        wl_x = [x_min - margin, x_max + margin, x_max + margin, x_min - margin]
        wl_y_lo = -y_max - margin if args.full else -margin
        wl_y_hi = y_max + margin
        wl_y = [wl_y_lo, wl_y_lo, wl_y_hi, wl_y_hi]
        wl_z = [0, 0, 0, 0]
        wl_poly = Poly3DCollection([[list(zip(wl_x, wl_y, wl_z))][0]],
                                    alpha=0.08, facecolor='cyan', edgecolor='cyan',
                                    linewidth=0.5)
        ax.add_collection3d(wl_poly)

    # Set axis limits
    x_range = nodes[:, 0].max() - nodes[:, 0].min()
    y_range = nodes[:, 1].max()  # half-beam
    z_range = nodes[:, 2].max() - nodes[:, 2].min()

    x_mid = 0.5 * (nodes[:, 0].max() + nodes[:, 0].min())
    max_range = max(x_range, 2 * y_range if args.full else y_range, z_range) * 0.6

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    if args.full:
        ax.set_ylim(-max_range, max_range)
    else:
        ax.set_ylim(-max_range * 0.2, max_range)
    ax.set_zlim(-max_range * 0.5, max_range * 0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.view_init(elev=args.elev, azim=args.azim)

    # Title
    case_name = os.path.basename(os.path.normpath(case_dir))
    title = f'{case_name} — {len(nodes)} nodes, {len(panels)} panels'
    if counts.get('stern_cap', 0) > 0:
        title += f' (stern: {counts["stern_cap"]})'
    if counts.get('bow_cap', 0) > 0:
        title += f' (bow: {counts["bow_cap"]})'
    ax.set_title(title, fontsize=12)

    # Legend
    import matplotlib.patches as mpatches
    legend_items = []
    for lbl, clr in colors.items():
        if lbl in counts:
            legend_items.append(mpatches.Patch(color=clr[:3], alpha=0.7,
                                               label=f'{lbl} ({counts[lbl]})'))
    if legend_items:
        ax.legend(handles=legend_items, loc='upper left', fontsize=9)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved: {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
