#!/usr/bin/env python3
"""
Length convergence study for FIXED-BODY drift force.

Compare drift force per unit length (Fy/L) for a semi-circular cylinder
of radius R=1m at two different lengths (L=20m and L=100m) in beam seas.

If Fy/L is the same for both lengths, then 3D end effects are negligible
and the L=20m result already represents the 2D limit.

Uses near-field (direct pressure integration) method.
Fixed body only — no motions, so no radiation or rotation terms.
"""

import numpy as np
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_potential, airy_waves_velocity
import logging
import time

cpt.set_logging(logging.WARNING)

# ============================================================
# Parameters
# ============================================================
R = 1.0
rho = 1025.0
g = 9.81
beta = np.pi / 2  # beam seas

wavelengths = np.array([22.0, 55.0, 90.0])
lengths = [20.0, 50.0, 100.0, 200.0]

# Mesh resolution: keep nr and ntheta constant; scale nx with length
nr = 10
ntheta = 40


# ============================================================
# Waterline edge extraction (from existing scripts)
# ============================================================
def extract_waterline_edges(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    z_tol = 0.05
    wl_mask = np.abs(vertices[:, 2]) < z_tol
    wl_verts = set(np.where(wl_mask)[0])

    edges = set()
    face_for_edge = {}
    for fi, face in enumerate(faces):
        if face[0] == face[3]:
            verts_list = [face[0], face[1], face[2]]
        else:
            verts_list = [face[0], face[1], face[2], face[3]]
        n_verts = len(verts_list)
        for kk in range(n_verts):
            v1 = verts_list[kk]
            v2 = verts_list[(kk + 1) % n_verts]
            if v1 in wl_verts and v2 in wl_verts:
                edge = (min(v1, v2), max(v1, v2))
                edges.add(edge)
                face_for_edge[edge] = fi
    edges = list(edges)
    n_edges = len(edges)
    if n_edges == 0:
        return [], np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))

    edge_centers = np.zeros((n_edges, 3))
    edge_lengths_arr = np.zeros(n_edges)
    edge_normals = np.zeros((n_edges, 3))
    face_normals = mesh.faces_normals
    face_centers = mesh.faces_centers

    for i, (v1, v2) in enumerate(edges):
        p1, p2 = vertices[v1], vertices[v2]
        edge_centers[i] = 0.5 * (p1 + p2)
        edge_vec = p2 - p1
        edge_lengths_arr[i] = np.linalg.norm(edge_vec)
        t = edge_vec / (edge_lengths_arr[i] + 1e-30)
        n1 = np.array([t[1], -t[0], 0.0])
        n2 = np.array([-t[1], t[0], 0.0])
        fi = face_for_edge[(min(v1, v2), max(v1, v2))]
        fc = face_centers[fi]
        to_face = fc[:2] - edge_centers[i, :2]
        if np.dot(n1[:2], to_face) < 0:
            edge_normals[i] = n1
        else:
            edge_normals[i] = n2
    return edges, edge_centers, edge_lengths_arr, edge_normals


# ============================================================
# Compute fixed-body drift force for one body length
# ============================================================
def compute_fixed_drift(L, wavelengths_arr):
    """Compute fixed-body drift Fy at each wavelength for a body of length L."""

    # Scale nx with length to keep panel aspect ratio roughly constant
    # For L=20, nx=50 → panel size ~0.4m along x
    # For L=100, nx=250 → same panel size
    # For L=200, cap at nx=400 to keep computation tractable (~0.5m panels)
    nx = min(int(round(50 * L / 20.0)), 400)
    mesh_res = (nr, ntheta, nx)

    print(f"\n{'='*70}")
    print(f"Building body: L={L}m, mesh resolution=({nr},{ntheta},{nx})")
    t0 = time.time()

    mesh_full = cpt.mesh_horizontal_cylinder(
        length=L, radius=R, center=(0, 0, 0),
        resolution=mesh_res, name=f"hull_L{int(L)}"
    )
    hull_mesh = mesh_full.immersed_part()
    lid = hull_mesh.generate_lid(z=-0.01)
    body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name=f"hull_L{int(L)}")
    body.center_of_mass = np.array([0.0, 0.0, -4 * R / (3 * np.pi)])
    body.rotation_center = body.center_of_mass
    body.add_all_rigid_body_dofs()

    print(f"  Hull faces: {hull_mesh.nb_faces}, Lid faces: {lid.nb_faces}")
    print(f"  Mesh build time: {time.time()-t0:.1f}s")

    # Extract waterline
    edges, edge_centers, edge_lengths_arr, edge_normals = extract_waterline_edges(hull_mesh)
    print(f"  Waterline: {len(edges)} edges, total length={np.sum(edge_lengths_arr):.1f}m")

    # Waterline eval points
    wl_eval_pts = edge_centers.copy()
    wl_eval_pts[:, 2] = -0.001

    # Hull info
    hull_normals = hull_mesh.faces_normals
    hull_areas = hull_mesh.faces_areas
    hull_centers = hull_mesh.faces_centers

    solver = cpt.BEMSolver()

    results = {}

    for lam in wavelengths_arr:
        k = 2 * np.pi / lam
        omega = np.sqrt(k * g)

        print(f"\n  lambda={lam:.0f}m (k={k:.4f}, omega={omega:.4f})")
        t1 = time.time()

        # Diffraction solve (fixed body — only diffraction, no radiation)
        diff_prob = cpt.DiffractionProblem(
            body=body, wave_direction=beta, omega=omega,
            water_depth=np.inf, rho=rho, g=g
        )
        diff_result = solver.solve(diff_prob)

        # Total potential = incident + diffraction on hull
        inc_pot_hull = airy_waves_potential(hull_centers, diff_prob)
        diff_pot_hull = diff_result.potential[:hull_mesh.nb_faces]
        total_pot_hull = inc_pot_hull + diff_pot_hull

        # Total velocity on hull
        inc_vel_hull = airy_waves_velocity(hull_centers, diff_prob)
        diff_vel_hull = solver.compute_velocity(hull_centers, diff_result)
        total_vel_hull = inc_vel_hull + diff_vel_hull

        # Total potential at waterline
        inc_pot_wl = airy_waves_potential(wl_eval_pts, diff_prob)
        diff_pot_wl = solver.compute_potential(wl_eval_pts, diff_result)
        total_pot_wl = inc_pot_wl + diff_pot_wl

        # --- Waterline term ---
        eta_wl = (1j * omega / g) * total_pot_wl
        eta_sq = np.abs(eta_wl)**2
        F_wl_y = 0.25 * rho * g * np.sum(eta_sq * edge_normals[:, 1] * edge_lengths_arr)

        # --- Velocity (Bernoulli) term ---
        vel_sq = np.sum(np.abs(total_vel_hull)**2, axis=1)
        F_vel_y = -0.25 * rho * np.sum(vel_sq * hull_normals[:, 1] * hull_areas)

        # Total drift Fy
        F_total_y = F_wl_y + F_vel_y

        dt = time.time() - t1
        print(f"    F_wl_y  = {F_wl_y:12.3f} N")
        print(f"    F_vel_y = {F_vel_y:12.3f} N")
        print(f"    F_total = {F_total_y:12.3f} N")
        print(f"    F/L     = {F_total_y/L:12.4f} N/m")
        print(f"    Solve time: {dt:.1f}s")

        results[lam] = {
            'F_wl_y': F_wl_y,
            'F_vel_y': F_vel_y,
            'F_total_y': F_total_y,
            'F_per_L': F_total_y / L,
        }

    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("FIXED-BODY DRIFT FORCE CONVERGENCE STUDY")
    print(f"Comparing Fy/L for L={lengths} in beam seas (beta=90deg)")
    print("Semi-circular cylinder R=1m, deep water")
    print("=" * 70)

    all_results = {}
    for L in lengths:
        all_results[L] = compute_fixed_drift(L, wavelengths)

    # ============================================================
    # Summary table
    # ============================================================
    print("\n\n" + "=" * 90)
    print("SUMMARY: Fixed-body drift force Fy per unit length [N/m]")
    print("=" * 90)

    # Header
    header = f"{'lambda':>8s}"
    for L in lengths:
        header += f"  {'L='+str(int(L))+' Fy/L':>14s}"
    print(header)
    print("-" * 90)

    for lam in wavelengths:
        line = f"{lam:8.0f}"
        for L in lengths:
            r = all_results[L][lam]
            line += f"  {r['F_per_L']:14.4f}"
        print(line)

    # Ratio table (relative to longest body)
    L_ref = max(lengths)
    print(f"\nRatio relative to L={int(L_ref)}m:")
    header2 = f"{'lambda':>8s}"
    for L in lengths:
        header2 += f"  {'L='+str(int(L)):>14s}"
    print(header2)
    print("-" * 90)

    for lam in wavelengths:
        line = f"{lam:8.0f}"
        ref_val = all_results[L_ref][lam]['F_per_L']
        for L in lengths:
            r = all_results[L][lam]
            ratio = r['F_per_L'] / ref_val if abs(ref_val) > 1e-10 else float('nan')
            line += f"  {ratio:14.4f}"
        print(line)

    print()
    print(f"If ratio → 1.0, end effects are negligible at that L.")
    print(f"lambda/L ratio for reference:")
    for L in lengths:
        ratios_str = ", ".join([f"{lam/L:.2f}" for lam in wavelengths])
        print(f"  L={int(L):>3d}m: lambda/L = {ratios_str}")

    # Also print breakdown
    print("\n\nBreakdown by term (per unit length):")
    print(f"{'lambda':>8s}  {'L':>5s}  {'F_wl/L':>12s}  {'F_vel/L':>12s}  {'F_tot/L':>12s}")
    print("-" * 55)
    for lam in wavelengths:
        for L in lengths:
            r = all_results[L][lam]
            print(f"{lam:8.0f}  {L:5.0f}  {r['F_wl_y']/L:12.4f}  "
                  f"{r['F_vel_y']/L:12.4f}  {r['F_per_L']:12.4f}")
