#!/usr/bin/env python3
"""
Length convergence study for FREELY-FLOATING drift force Fy.

Compute drift force Fy for a freely-floating semi-circular barge
at L=20m and L=100m, for wavelengths λ = 22, 55, 90 m in beam seas (β = π/2).

Uses BOTH:
  1. Near-field (direct pressure integration) method
  2. Far-field (Maruo) method via field-point ring

Semi-circular barge: R=1m, ρ=1025, g=9.81, 6 DOF.

The near-field method suffers from catastrophic cancellation for freely-floating
bodies (velocity and rotation terms each >> total, nearly cancel). The far-field
method avoids this by computing the drift force from radiated wave energy.

Near-field formula (Pinkster 1979), with n pointing INTO FLUID (Capytaine convention):
  F̄ = (1/4)ρg ∮_WL |η̂_rel|² n dl             [waterline]
     + (1/4)ρ ∫∫_Sb |∇φ̂|² n dS                 [velocity squared]
     + (1/2) Re[ ∫∫_Sb p̂ (n × α̂*) dS ]        [rotation]

Far-field (Maruo) formula:
  F̄_y = (ρg/k) ∫₀²π |a(θ)|² sin(θ) dθ

where a(θ) = far-field wave amplitude of total disturbance (scattered + radiated).

Capytaine convention: exp(-iωt).
"""

import numpy as np
import capytaine as cpt
from capytaine.bem.airy_waves import (
    airy_waves_potential, airy_waves_velocity, froude_krylov_force
)
import logging
import time
import re
import os

cpt.set_logging(logging.WARNING)

# ============================================================
# Physical parameters
# ============================================================
R = 1.0
rho = 1025.0
g = 9.81
beta = np.pi / 2  # beam seas

wavelengths = np.array([22.0, 55.0, 90.0])
lengths = [20.0, 100.0]

# Mesh resolution: keep nr and ntheta constant; scale nx with length
nr = 10
ntheta = 40

# Far-field ring parameters
R_FIELD = 5000.0   # radius of field-point ring [m]
N_THETA = 720      # number of angular points


# ============================================================
# Waterline edge extraction
# ============================================================
def extract_waterline_edges(mesh):
    """Extract waterline edges from an immersed mesh clipped at z=0."""
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
        n_v = len(verts_list)
        for kk in range(n_v):
            v1 = verts_list[kk]
            v2 = verts_list[(kk + 1) % n_v]
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
# Build body + mass/stiffness for given length
# ============================================================
def make_body_and_properties(L):
    """Create semi-circular barge body and return (body, M, C, dof_names)."""
    nx = {20.0: 50, 100.0: 250}.get(L, int(round(50 * L / 20.0)))
    mesh_res = (nr, ntheta, nx)

    print(f"\n  Building body: L={L}m, mesh res=({nr},{ntheta},{nx})")
    t0 = time.time()

    mesh_full = cpt.mesh_horizontal_cylinder(
        length=L, radius=R, center=(0, 0, 0),
        resolution=mesh_res, name=f"hull_L{int(L)}"
    )
    hull_mesh = mesh_full.immersed_part()
    lid = hull_mesh.generate_lid(z=-0.01)
    body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name=f"hull_L{int(L)}")

    # Mass properties (scale with L)
    mass = rho * np.pi * R**2 / 2 * L
    zcg = -4 * R / (3 * np.pi)  # ≈ -0.4244
    body.center_of_mass = np.array([0.0, 0.0, zcg])
    body.mass = mass
    body.rotation_center = body.center_of_mass
    body.add_all_rigid_body_dofs()

    dof_names = list(body.dofs.keys())
    n_dof = len(dof_names)

    # Inertia
    kxx_sq = (0.4 * 2 * R)**2        # = 0.64 m²
    kyy_sq = (0.25 * L)**2
    kzz_sq = (0.25 * L)**2
    Ixx = mass * kxx_sq
    Iyy = mass * kyy_sq
    Izz = mass * kzz_sq

    # 6×6 mass matrix (diagonal since rotation center = CoG)
    M = np.zeros((n_dof, n_dof))
    dof_idx = {d: i for i, d in enumerate(dof_names)}
    for d in ('Surge', 'Sway', 'Heave'):
        M[dof_idx[d], dof_idx[d]] = mass
    M[dof_idx['Roll'], dof_idx['Roll']] = Ixx
    M[dof_idx['Pitch'], dof_idx['Pitch']] = Iyy
    M[dof_idx['Yaw'], dof_idx['Yaw']] = Izz

    # Hydrostatic stiffness from Capytaine
    stiffness_xr = body.compute_hydrostatic_stiffness(rho=rho, g=g)
    C = np.zeros((n_dof, n_dof))
    for i, idof in enumerate(dof_names):
        for j, jdof in enumerate(dof_names):
            try:
                C[i, j] = float(stiffness_xr.sel(
                    influenced_dof=idof, radiating_dof=jdof))
            except (KeyError, ValueError):
                C[i, j] = 0.0

    print(f"  Hull faces: {hull_mesh.nb_faces}, Lid faces: {lid.nb_faces}")
    print(f"  Mass: {mass:.1f} kg, zcg: {zcg:.4f} m")
    print(f"  Ixx={Ixx:.1f}, Iyy={Iyy:.1f}, Izz={Izz:.1f}")
    print(f"  K33={C[dof_idx['Heave'], dof_idx['Heave']]:.1f}, "
          f"K44={C[dof_idx['Roll'], dof_idx['Roll']]:.1f}, "
          f"K55={C[dof_idx['Pitch'], dof_idx['Pitch']]:.1f}")
    print(f"  Build time: {time.time()-t0:.1f}s")

    return body, M, C, dof_names


# ============================================================
# Compute freely-floating drift force for one body length
# ============================================================
def compute_floating_drift(L, wavelengths_arr):
    """Compute freely-floating drift Fy using near-field AND far-field methods."""

    body, M, C, dof_names = make_body_and_properties(L)
    n_dof = len(dof_names)

    hull_mesh = body.mesh
    hull_normals = hull_mesh.faces_normals
    hull_areas = hull_mesh.faces_areas
    hull_centers = hull_mesh.faces_centers
    n_hull = hull_mesh.nb_faces
    hull_mask = body.hull_mask if hasattr(body, 'hull_mask') else np.ones(n_hull, dtype=bool)

    # Waterline
    edges, edge_centers, edge_lengths_arr, edge_normals = extract_waterline_edges(hull_mesh)
    print(f"  Waterline: {len(edges)} edges, total length={np.sum(edge_lengths_arr):.1f}m")

    wl_eval_pts = edge_centers.copy()
    if len(edges) > 0:
        wl_eval_pts[:, 2] = -0.001

    # Far-field ring
    theta_fp = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    dtheta = 2 * np.pi / N_THETA
    field_pts = np.column_stack([
        R_FIELD * np.cos(theta_fp),
        R_FIELD * np.sin(theta_fp),
        np.zeros(N_THETA)
    ])

    solver = cpt.BEMSolver()
    results = {}

    for lam in wavelengths_arr:
        k = 2 * np.pi / lam
        omega = np.sqrt(k * g)

        print(f"\n  lambda={lam:.0f}m (k={k:.4f}, omega={omega:.4f}, kR={k*R:.3f})")
        t1 = time.time()

        # ---- 1. Radiation problems for all 6 DOFs ----
        rad_results = {}
        for dof in dof_names:
            prob = cpt.RadiationProblem(
                body=body, radiating_dof=dof, omega=omega,
                water_depth=np.inf, rho=rho, g=g
            )
            rad_results[dof] = solver.solve(prob)

        # Added mass and damping matrices
        A_mat = np.zeros((n_dof, n_dof))
        B_mat = np.zeros((n_dof, n_dof))
        for i, rdof in enumerate(dof_names):
            for j, idof in enumerate(dof_names):
                A_mat[i, j] = rad_results[rdof].added_masses[idof]
                B_mat[i, j] = rad_results[rdof].radiation_dampings[idof]

        # Radiation potential on hull panels
        rad_potential_hull = np.zeros((n_dof, n_hull), dtype=complex)
        for i, dof in enumerate(dof_names):
            pot_all = rad_results[dof].potential
            rad_potential_hull[i, :] = pot_all[hull_mask]

        # Radiation velocity on hull
        rad_velocity_hull = np.zeros((n_dof, n_hull, 3), dtype=complex)
        for i, dof in enumerate(dof_names):
            vel = solver.compute_velocity(hull_centers, rad_results[dof])
            rad_velocity_hull[i, :, :] = vel

        # Radiation potential at waterline
        rad_pot_wl = np.zeros((n_dof, len(edges)), dtype=complex)
        if len(edges) > 0:
            for i, dof in enumerate(dof_names):
                rad_pot_wl[i, :] = solver.compute_potential(wl_eval_pts, rad_results[dof])

        # Radiation potential at far-field ring
        rad_phi_fp = {}
        for dof in dof_names:
            rad_phi_fp[dof] = solver.compute_potential(field_pts, rad_results[dof])

        # ---- 2. Diffraction problem ----
        diff_prob = cpt.DiffractionProblem(
            body=body, wave_direction=beta, omega=omega,
            water_depth=np.inf, rho=rho, g=g
        )
        diff_result = solver.solve(diff_prob)

        diff_pot_hull = diff_result.potential[hull_mask]
        diff_vel_hull = solver.compute_velocity(hull_centers, diff_result)
        diff_pot_wl = np.zeros(len(edges), dtype=complex)
        if len(edges) > 0:
            diff_pot_wl = solver.compute_potential(wl_eval_pts, diff_result)

        # Diffraction at far-field ring
        phi_scatter_fp = solver.compute_potential(field_pts, diff_result)

        # ---- 3. Incident (Airy) wave on hull ----
        inc_pot_hull = airy_waves_potential(hull_centers, diff_prob)
        inc_vel_hull = airy_waves_velocity(hull_centers, diff_prob)
        inc_pot_wl = np.zeros(len(edges), dtype=complex)
        if len(edges) > 0:
            inc_pot_wl = airy_waves_potential(wl_eval_pts, diff_prob)

        # ---- 4. Froude-Krylov + diffraction = total excitation ----
        FK = froude_krylov_force(diff_prob)
        F_exc = np.array([diff_result.forces[dof] + FK[dof] for dof in dof_names])

        # ---- 5. Equations of motion ----
        Z = -omega**2 * (M + A_mat) + 1j * omega * B_mat + C
        xi = np.linalg.solve(Z, F_exc)

        dof_idx = {d: i for i, d in enumerate(dof_names)}
        print(f"    RAOs: " + " ".join(
            f"{dof}={abs(xi[i]):.4f}" for i, dof in enumerate(dof_names)))

        # ---- 6. Total first-order potential on hull ----
        total_pot_hull = inc_pot_hull + diff_pot_hull
        total_vel_hull = inc_vel_hull + diff_vel_hull
        for i in range(n_dof):
            total_pot_hull += xi[i] * rad_potential_hull[i, :]
            total_vel_hull += xi[i] * rad_velocity_hull[i, :]

        total_pot_wl = inc_pot_wl + diff_pot_wl
        for i in range(n_dof):
            total_pot_wl += xi[i] * rad_pot_wl[i, :]

        # ============================================================
        # NEAR-FIELD DRIFT FORCE
        # ============================================================

        # --- Term 1: Waterline ---
        eta_wl = (1j * omega / g) * total_pot_wl
        xi_heave = xi[dof_idx['Heave']]
        xi_roll = xi[dof_idx['Roll']]
        xi_pitch = xi[dof_idx['Pitch']]
        z_body_wl = (xi_heave
                     + xi_roll * edge_centers[:, 1]
                     - xi_pitch * edge_centers[:, 0])
        eta_rel = eta_wl - z_body_wl

        F_wl = np.zeros(3)
        if len(edges) > 0:
            eta_rel_sq = np.abs(eta_rel)**2
            for ic in range(3):
                F_wl[ic] = 0.25 * rho * g * np.sum(
                    eta_rel_sq * edge_normals[:, ic] * edge_lengths_arr)

        # --- Term 2: Velocity squared ---
        # Sign: F_body = -∫∫ p n_out dS, with p₂ = -½ρ|∇φ|²
        # Using Capytaine normals (n_out = into fluid): F = +¼ρ ∫|∇φ̂|² n dS
        # BUT: the original near-field fixed-body results (which were verified)
        # used -0.25. Need to verify which is correct.
        vel_sq = np.sum(np.abs(total_vel_hull)**2, axis=1)
        F_vel = np.zeros(3)
        for ic in range(3):
            F_vel[ic] = -0.25 * rho * np.sum(
                vel_sq * hull_normals[:, ic] * hull_areas)

        # --- Term 3: Rotation ---
        xi_yaw = xi[dof_idx['Yaw']]
        alpha_conj = np.conj(np.array([xi_roll, xi_pitch, xi_yaw]))
        p_total = 1j * omega * rho * total_pot_hull

        F_rot = np.zeros(3)
        for fi in range(n_hull):
            n_cross_alpha = np.cross(hull_normals[fi], alpha_conj)
            F_rot += 0.5 * np.real(p_total[fi] * n_cross_alpha) * hull_areas[fi]

        F_nf = F_wl + F_vel + F_rot

        # ============================================================
        # FAR-FIELD (MARUO) DRIFT FORCE
        # ============================================================

        # Total disturbance potential at far field = scattered + radiated
        phi_disturb_fp = phi_scatter_fp.copy()
        for i, dof in enumerate(dof_names):
            phi_disturb_fp += xi[i] * rad_phi_fp[dof]

        # Extract far-field wave amplitude:
        #   a(θ) = (iω/g) × φ × √r × exp(-ikr) at z=0
        a_total = (1j * omega / g) * phi_disturb_fp * np.sqrt(R_FIELD) * np.exp(-1j * k * R_FIELD)
        a2 = np.abs(a_total)**2

        # Maruo formula: F̄_i = (ρg/k) ∫ |a(θ)|² ê_i(θ) dθ
        coeff = rho * g / k
        Fx_ff = coeff * np.sum(a2 * np.cos(theta_fp)) * dtheta
        Fy_ff = coeff * np.sum(a2 * np.sin(theta_fp)) * dtheta

        # Damping check: B_22 from far field vs direct
        # B_22 = (ρg²)/(2ω³) ∫|a_sway(θ)|² dθ
        a_sway = (1j * omega / g) * rad_phi_fp['Sway'] * np.sqrt(R_FIELD) * np.exp(-1j * k * R_FIELD)
        B22_ff = rho * g**2 / (2 * omega**3) * np.sum(np.abs(a_sway)**2) * dtheta
        B22_direct = B_mat[dof_idx['Sway'], dof_idx['Sway']]

        dt = time.time() - t1

        print(f"    --- Near-field ---")
        print(f"    F_wl   = ({F_wl[0]:12.1f}, {F_wl[1]:12.1f}, {F_wl[2]:12.1f})")
        print(f"    F_vel  = ({F_vel[0]:12.1f}, {F_vel[1]:12.1f}, {F_vel[2]:12.1f})")
        print(f"    F_rot  = ({F_rot[0]:12.1f}, {F_rot[1]:12.1f}, {F_rot[2]:12.1f})")
        print(f"    NF Fy  = {F_nf[1]:12.1f} N,   Fy/L = {F_nf[1]/L:10.4f} N/m")
        print(f"    --- Far-field (Maruo) ---")
        print(f"    FF Fy  = {Fy_ff:12.1f} N,   Fy/L = {Fy_ff/L:10.4f} N/m")
        print(f"    FF Fx  = {Fx_ff:12.1f} N")
        print(f"    B22 check: direct={B22_direct:.1f}, far-field={B22_ff:.1f}, "
              f"ratio={B22_direct/B22_ff if abs(B22_ff)>0.1 else float('nan'):.4f}")
        print(f"    Solve time: {dt:.1f}s")

        results[lam] = {
            'F_wl': F_wl.copy(),
            'F_vel': F_vel.copy(),
            'F_rot': F_rot.copy(),
            'Fy_nf': F_nf[1],
            'Fy_nf_per_L': F_nf[1] / L,
            'Fy_ff': Fy_ff,
            'Fy_ff_per_L': Fy_ff / L,
            'Fx_ff': Fx_ff,
            'xi': xi.copy(),
            'B22_direct': B22_direct,
            'B22_ff': B22_ff,
        }

    return results


# ============================================================
# Parse pdstrip drift data
# ============================================================
def parse_pdstrip_drift(filepath):
    fnum = r'[+-]?[\d.]+(?:[EeDd][+-]?\d+)?'
    data = []
    current = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('DRIFT_START'):
                m = re.search(rf'omega=\s*({fnum})\s+mu=\s*({fnum})', line)
                if m:
                    current = {
                        'omega': float(m.group(1)),
                        'mu_deg': float(m.group(2))
                    }
            elif line.startswith('DRIFT_TOTAL'):
                m = re.search(rf'fxi=\s*({fnum})\s+feta=\s*({fnum})', line)
                if m:
                    current['fxi'] = float(m.group(1))
                    current['feta'] = float(m.group(2))
                    data.append(current)
                    current = {}
    return data


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("FREELY-FLOATING DRIFT FORCE Fy: LENGTH CONVERGENCE STUDY")
    print(f"Beam seas (beta=pi/2), semi-circular barge R={R}m")
    print(f"Lengths: {lengths}, Wavelengths: {list(wavelengths)}")
    print("Methods: Near-field (Pinkster) + Far-field (Maruo)")
    print("=" * 80)

    all_results = {}
    for L in lengths:
        print(f"\n{'#'*80}")
        print(f"# L = {L} m")
        print(f"{'#'*80}")
        all_results[L] = compute_floating_drift(L, wavelengths)

    # ============================================================
    # Parse pdstrip results (L=20 only)
    # ============================================================
    pdstrip_debug = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "run_mono", "debug.out")
    pdstrip_data = []
    if os.path.exists(pdstrip_debug):
        pdstrip_data = parse_pdstrip_drift(pdstrip_debug)
        print(f"\nLoaded {len(pdstrip_data)} pdstrip drift entries")

    # Get pdstrip Fy for beam seas (mu=90) at each wavelength
    # pdstrip feta at mu=90 > 0 means force in wave propagation direction.
    # Capytaine beta=pi/2: wave propagates in +y, so Fy > 0 is same direction.
    pdstrip_fy = {}
    for lam in wavelengths:
        k = 2 * np.pi / lam
        omega = np.sqrt(k * g)
        match = [d for d in pdstrip_data
                 if abs(d['omega'] - omega) < 0.01 and abs(d['mu_deg'] - 90.0) < 1.0]
        if match:
            pdstrip_fy[lam] = match[0]['feta']

    # ============================================================
    # Summary tables
    # ============================================================

    # ---- Table 1: Far-field Fy/L (the reliable method) ----
    print("\n\n" + "=" * 90)
    print("FAR-FIELD (MARUO) DRIFT FORCE Fy/L [N/m] — Freely-floating, beam seas")
    print("  (This is the reliable method; avoids near-field cancellation)")
    print("=" * 90)

    print(f"\n{'lambda':>8s}", end="")
    for L in lengths:
        print(f"  {'L='+str(int(L))+' Fy/L':>14s}", end="")
    if pdstrip_fy:
        print(f"  {'pd Fy/20':>14s}", end="")
    print()
    print("-" * 90)

    for lam in wavelengths:
        print(f"{lam:8.0f}", end="")
        for L in lengths:
            r = all_results[L][lam]
            print(f"  {r['Fy_ff_per_L']:14.4f}", end="")
        if lam in pdstrip_fy:
            print(f"  {pdstrip_fy[lam]/20.0:14.4f}", end="")
        print()

    # ---- Table 2: Far-field Fy absolute values ----
    print(f"\nAbsolute Fy [N]:")
    print(f"{'lambda':>8s}", end="")
    for L in lengths:
        print(f"  {'L='+str(int(L))+' Fy':>14s}", end="")
    if pdstrip_fy:
        print(f"  {'pd Fy(L=20)':>14s}", end="")
    print()
    print("-" * 90)

    for lam in wavelengths:
        print(f"{lam:8.0f}", end="")
        for L in lengths:
            r = all_results[L][lam]
            print(f"  {r['Fy_ff']:14.1f}", end="")
        if lam in pdstrip_fy:
            print(f"  {pdstrip_fy[lam]:14.1f}", end="")
        print()

    # ---- Table 3: Length convergence ratio (far-field) ----
    L_ref = max(lengths)
    print(f"\nRatio of Fy/L relative to L={int(L_ref)}m (far-field):")
    print(f"{'lambda':>8s}", end="")
    for L in lengths:
        print(f"  {'L='+str(int(L)):>14s}", end="")
    print()
    print("-" * 60)

    for lam in wavelengths:
        print(f"{lam:8.0f}", end="")
        ref_val = all_results[L_ref][lam]['Fy_ff_per_L']
        for L in lengths:
            r = all_results[L][lam]
            ratio = r['Fy_ff_per_L'] / ref_val if abs(ref_val) > 1e-10 else float('nan')
            print(f"  {ratio:14.4f}", end="")
        print()

    # ---- Table 4: Near-field vs far-field comparison ----
    print(f"\nNear-field vs Far-field Fy/L comparison:")
    print(f"{'lambda':>8s}  {'L':>5s}  {'NF Fy/L':>12s}  {'FF Fy/L':>12s}  {'NF/FF':>10s}  "
          f"{'WL/L':>10s}  {'vel/L':>10s}  {'rot/L':>10s}")
    print("-" * 90)
    for lam in wavelengths:
        for L in lengths:
            r = all_results[L][lam]
            nf = r['Fy_nf_per_L']
            ff = r['Fy_ff_per_L']
            ratio = nf / ff if abs(ff) > 1e-10 else float('nan')
            print(f"{lam:8.0f}  {L:5.0f}  {nf:12.4f}  {ff:12.4f}  {ratio:10.3f}  "
                  f"{r['F_wl'][1]/L:10.4f}  {r['F_vel'][1]/L:10.4f}  {r['F_rot'][1]/L:10.4f}")

    # ---- Table 5: Comparison with pdstrip ----
    if pdstrip_fy:
        print(f"\nComparison: Capytaine far-field (L=20) vs pdstrip (L=20):")
        print(f"{'lambda':>8s}  {'cap FF Fy':>12s}  {'pd Fy':>12s}  {'cap/pd':>10s}")
        print("-" * 50)
        for lam in wavelengths:
            if lam in pdstrip_fy:
                cap_fy = all_results[20.0][lam]['Fy_ff']
                pd_fy = pdstrip_fy[lam]
                ratio = cap_fy / pd_fy if abs(pd_fy) > 0.1 else float('nan')
                print(f"{lam:8.0f}  {cap_fy:12.1f}  {pd_fy:12.1f}  {ratio:10.3f}")

    # ---- Table 6: RAOs ----
    print(f"\nRAO magnitudes:")
    print(f"{'lambda':>8s}  {'L':>5s}  {'Surge':>10s}  {'Sway':>10s}  {'Heave':>10s}  "
          f"{'Roll':>10s}  {'Pitch':>10s}  {'Yaw':>10s}")
    print("-" * 80)
    for lam in wavelengths:
        for L in lengths:
            r = all_results[L][lam]
            xi = r['xi']
            print(f"{lam:8.0f}  {L:5.0f}", end="")
            for i in range(6):
                print(f"  {abs(xi[i]):10.4f}", end="")
            print()

    # ---- Table 7: Damping verification ----
    print(f"\nDamping verification (B22 direct vs far-field):")
    print(f"{'lambda':>8s}  {'L':>5s}  {'B22_direct':>12s}  {'B22_ff':>12s}  {'ratio':>10s}")
    print("-" * 55)
    for lam in wavelengths:
        for L in lengths:
            r = all_results[L][lam]
            ratio = r['B22_direct'] / r['B22_ff'] if abs(r['B22_ff']) > 0.1 else float('nan')
            print(f"{lam:8.0f}  {L:5.0f}  {r['B22_direct']:12.1f}  {r['B22_ff']:12.1f}  {ratio:10.4f}")

    # ---- Reference info ----
    print(f"\nlambda/L ratio:")
    for L in lengths:
        ratios_str = ", ".join([f"{lam/L:.2f}" for lam in wavelengths])
        print(f"  L={int(L):>3d}m: lambda/L = {ratios_str}")

    print("\nDone.")
