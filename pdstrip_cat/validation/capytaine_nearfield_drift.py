#!/usr/bin/env python3
"""
Compute mean drift forces using the NEAR-FIELD (direct pressure integration)
method in Capytaine, for comparison with pdstrip.

Near-field formula for mean second-order drift force (Pinkster 1979, eq 3.34):

  F̄_i = (1/2)ρg ∮_WL |η_rel|² n_i dl                    [waterline term]
       - (1/2)ρ ∫∫_Sb |∇φ|² n_i dS                        [velocity squared term]
       + (1/2) Re[ ∫∫_Sb p (n × α*) ] · e_i               [rotation term]

where:
  η_rel = relative wave elevation at waterline = (iω/g)φ_total at z=0
  ∇φ = velocity gradient of the total first-order potential on the body surface
  p = first-order pressure = -iωρφ (for exp(-iωt) convention) = iωρφ (Capytaine)
  α = complex rotation vector (roll, pitch, yaw) RAO
  n = outward normal to body into fluid

Capytaine convention: exp(-iωt), so:
  p = iωρφ    (pressure from potential)
  v = ∇φ      (velocity)
  η = (iω/g)φ  at z=0

Time averaging: <Re(A e^{-iωt}) × Re(B e^{-iωt})> = (1/2) Re(A conj(B))

NOTE: This is a FIXED-BODY computation first, to avoid rotation term complexity.
For the semi-circular barge, the body barely moves at high frequencies (short waves),
where the drift force is largest.
"""

import numpy as np
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_potential, airy_waves_velocity, froude_krylov_force
import logging
import sys
import os
import re

cpt.set_logging(logging.WARNING)

# ============================================================
# Parameters (must match pdstrip input exactly)
# ============================================================
R = 1.0         # cylinder radius [m]
L = 20.0        # barge length [m]
rho = 1025.0    # water density [kg/m^3]
g = 9.81        # gravity [m/s^2]

# Mesh resolution
mesh_res = (10, 40, 50)  # (nr_endcap, ntheta, nx)

# pdstrip frequencies - use a subset for initial testing
wavelengths_all = np.array([3, 4, 5, 6, 8, 10, 13, 17, 22, 28, 35, 45, 55, 70, 90])

# Use subset for quick testing, or full set
if '--full' in sys.argv:
    wavelengths = wavelengths_all
else:
    wavelengths = np.array([3, 6, 10, 22, 55])  # quick subset
    print(f"[Quick mode: {len(wavelengths)} wavelengths. Use --full for all 15]")

k_values = 2 * np.pi / wavelengths
omega_values = np.sqrt(k_values * g)

# Wave directions
wave_directions = np.array([0, np.pi/2, np.pi])  # following, beam, head seas


# ============================================================
# Mesh creation
# ============================================================
def make_hull_body(R, L, y_offset=0.0, name="hull"):
    mesh_full = cpt.mesh_horizontal_cylinder(
        length=L, radius=R, center=(0, y_offset, 0),
        resolution=mesh_res, name=name
    )
    hull_mesh = mesh_full.immersed_part()
    lid = hull_mesh.generate_lid(z=-0.01)
    body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name=name)
    # Mass properties MUST match pdstrip.inp exactly:
    #   32201.3 0.0 0.0 -0.4244 0.6400 25.0000 25.0000 0.0 0.0 0.0
    #   mass    xcg ycg  zcg    kxx²   kyy²    kzz²    kxy² kxz² kyz²
    # pdstrip radii of gyration values are k² in m²
    mass_val = 32201.3
    zcg = -0.4244
    kxx_sq = 0.64    # (0.4 * beam)^2 = (0.4*2)^2 = 0.64 m²
    kyy_sq = 25.0    # (0.25 * L)^2 = (0.25*20)^2 = 25.0 m²
    kzz_sq = 25.0
    Ixx = mass_val * kxx_sq   # 20608.8 kg·m²
    Iyy = mass_val * kyy_sq   # 805032.5 kg·m²
    Izz = mass_val * kzz_sq   # 805032.5 kg·m²
    body.center_of_mass = np.array([0.0, y_offset, zcg])
    body.mass = mass_val
    body.rotation_center = body.center_of_mass
    body.add_all_rigid_body_dofs()
    return body, mass_val, zcg, np.diag([Ixx, Iyy, Izz])


# ============================================================
# Waterline edge extraction
# ============================================================
def extract_waterline_edges(mesh):
    """Extract waterline edges from an immersed mesh clipped at z=0.
    
    Returns:
        edges: list of (v1_idx, v2_idx) vertex index pairs on the waterline
        edge_centers: (n_edges, 3) array of edge midpoints
        edge_lengths: (n_edges,) array of edge lengths
        edge_outward_normals: (n_edges, 3) array of outward horizontal normals
            (pointing into the fluid, perpendicular to edge, in the z=0 plane)
    """
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Find vertices on the waterline (z ≈ 0)
    z_tol = 0.05  # tolerance for waterline detection
    wl_mask = np.abs(vertices[:, 2]) < z_tol
    wl_verts = set(np.where(wl_mask)[0])
    
    # Find edges where BOTH vertices are on the waterline
    edges = set()
    face_for_edge = {}  # maps edge -> face index (for normal computation)
    
    for fi, face in enumerate(faces):
        # Get unique vertices (triangles have face[0]==face[3])
        if face[0] == face[3]:
            verts = [face[0], face[1], face[2]]
        else:
            verts = [face[0], face[1], face[2], face[3]]
        
        n_verts = len(verts)
        for k in range(n_verts):
            v1 = verts[k]
            v2 = verts[(k + 1) % n_verts]
            if v1 in wl_verts and v2 in wl_verts:
                edge = (min(v1, v2), max(v1, v2))
                edges.add(edge)
                face_for_edge[edge] = fi
    
    edges = list(edges)
    n_edges = len(edges)
    
    if n_edges == 0:
        return [], np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))
    
    edge_centers = np.zeros((n_edges, 3))
    edge_lengths = np.zeros(n_edges)
    edge_outward_normals = np.zeros((n_edges, 3))
    
    face_normals = mesh.faces_normals
    face_centers = mesh.faces_centers
    
    for i, (v1, v2) in enumerate(edges):
        p1 = vertices[v1]
        p2 = vertices[v2]
        edge_centers[i] = 0.5 * (p1 + p2)
        
        edge_vec = p2 - p1
        edge_lengths[i] = np.linalg.norm(edge_vec)
        
        # The outward normal in the waterplane is perpendicular to the edge
        # and points AWAY from the body (into the fluid).
        # We use the face normal to determine the correct direction.
        
        # Edge tangent (in z=0 plane approximately)
        t = edge_vec / (edge_lengths[i] + 1e-30)
        
        # Candidate outward normal: rotate tangent by -90° in the xy plane
        # n_candidate = (t_y, -t_x, 0)  -- or (+t_y, -t_x)
        n1 = np.array([t[1], -t[0], 0.0])
        n2 = np.array([-t[1], t[0], 0.0])
        
        # The correct outward normal should point AWAY from the face center
        fi = face_for_edge[(min(v1, v2), max(v1, v2))]
        fc = face_centers[fi]
        
        # Vector from edge center to face center
        to_face = fc[:2] - edge_centers[i, :2]
        
        # The outward normal should point OPPOSITE to the face center direction
        if np.dot(n1[:2], to_face) < 0:
            edge_outward_normals[i] = n1
        else:
            edge_outward_normals[i] = n2
    
    return edges, edge_centers, edge_lengths, edge_outward_normals


# ============================================================
# Near-field drift force computation
# ============================================================
def compute_nearfield_drift(body, mass_val, zcg, inertia_matrix, omegas, betas):
    """
    Compute mean drift forces using near-field (pressure integration) method.
    
    Returns dict keyed by (omega, beta) with drift force components and breakdown.
    """
    solver = cpt.BEMSolver()
    dof_names = list(body.dofs.keys())
    n_dof = len(dof_names)
    
    # Build 6x6 mass matrix with off-diagonal coupling from zcg
    # Standard 6x6 rigid body mass matrix about rotation center = CoG:
    #   M[surge,surge] = m,  M[sway,sway] = m,  M[heave,heave] = m
    #   M[roll,roll] = Ixx, M[pitch,pitch] = Iyy, M[yaw,yaw] = Izz
    #   Off-diag from CoG offset (xcg=0, ycg=0, zcg != 0):
    #     M[surge,pitch] = M[pitch,surge] = m*zcg  (positive)
    #     M[sway,roll] = M[roll,sway] = -m*zcg  (negative)
    # NOTE: Capytaine rotation center = CoG, so coupling depends on convention.
    # Actually, with rotation_center = center_of_mass, if dofs are defined about CoG,
    # the mass matrix is diagonal (no coupling). The coupling only appears when
    # the rotation center differs from CoG, or when we express things about the origin.
    #
    # Capytaine convention: dofs are defined about the body's rotation_center.
    # Since rotation_center = center_of_mass, the mass matrix should be diagonal
    # for the translational and rotational parts. BUT the hydrostatic stiffness
    # matrix from Capytaine already accounts for the CoG position.
    #
    # For the equation of motion: [-ω²(M+A) + iωB + C] ξ = F
    # Capytaine's radiation/diffraction are done about the rotation center,
    # so M should be the mass matrix about the rotation center = CoG = diagonal.
    M = np.zeros((n_dof, n_dof))
    dof_idx = {dof: i for i, dof in enumerate(dof_names)}
    for i, dof in enumerate(dof_names):
        if dof in ('Surge', 'Sway', 'Heave'):
            M[i, i] = mass_val
        elif dof == 'Roll':
            M[i, i] = inertia_matrix[0, 0]
        elif dof == 'Pitch':
            M[i, i] = inertia_matrix[1, 1]
        elif dof == 'Yaw':
            M[i, i] = inertia_matrix[2, 2]
    
    # Hydrostatic stiffness
    stiffness_xr = body.compute_hydrostatic_stiffness()
    C = np.zeros((n_dof, n_dof))
    for i, idof in enumerate(dof_names):
        for j, jdof in enumerate(dof_names):
            try:
                C[i, j] = float(stiffness_xr.sel(
                    influenced_dof=idof, radiating_dof=jdof))
            except (KeyError, ValueError):
                C[i, j] = 0.0
    
    # Extract waterline info
    hull_mesh = body.mesh
    edges, edge_centers, edge_lengths, edge_normals = extract_waterline_edges(hull_mesh)
    print(f"Waterline: {len(edges)} edges, total length = {np.sum(edge_lengths):.2f} m")
    
    # Get hull panel info
    hull_normals = hull_mesh.faces_normals      # (n_hull, 3) outward into fluid
    hull_areas = hull_mesh.faces_areas          # (n_hull,)
    hull_centers = hull_mesh.faces_centers      # (n_hull, 3)
    n_hull = hull_mesh.nb_faces
    hull_mask = body.hull_mask if hasattr(body, 'hull_mask') else np.ones(n_hull, dtype=bool)
    
    results = {}
    
    for omega in omegas:
        k = omega**2 / g
        lam = 2 * np.pi / k
        print(f"\n--- omega={omega:.3f}, lambda={lam:.1f}m ---")
        
        # 1. Solve radiation problems
        rad_results = {}
        for dof in dof_names:
            prob = cpt.RadiationProblem(
                body=body, radiating_dof=dof, omega=omega,
                water_depth=np.inf, rho=rho, g=g
            )
            rad_results[dof] = solver.solve(prob)
        
        # Build added mass and damping
        A = np.zeros((n_dof, n_dof))
        B_mat = np.zeros((n_dof, n_dof))
        for i, rdof in enumerate(dof_names):
            for j, idof in enumerate(dof_names):
                A[i, j] = rad_results[rdof].added_masses[idof]
                B_mat[i, j] = rad_results[rdof].radiation_dampings[idof]
        
        # Radiation potentials on hull (for combining later)
        # result.potential includes hull+lid; we need hull only
        rad_potential_hull = np.zeros((n_dof, n_hull), dtype=complex)
        for i, dof in enumerate(dof_names):
            pot_all = rad_results[dof].potential
            rad_potential_hull[i, :] = pot_all[hull_mask]
        
        # Radiation velocity on hull surface
        rad_velocity_hull = np.zeros((n_dof, n_hull, 3), dtype=complex)
        for i, dof in enumerate(dof_names):
            vel = solver.compute_velocity(hull_centers, rad_results[dof])
            rad_velocity_hull[i, :, :] = vel
        
        # Radiation potential at waterline edge centers
        # NOTE: evaluate slightly below z=0 to avoid Green function singularity
        rad_pot_wl = np.zeros((n_dof, len(edges)), dtype=complex)
        wl_eval_pts = edge_centers.copy()
        if len(edges) > 0:
            wl_eval_pts[:, 2] = -0.001  # slightly submerged
            for i, dof in enumerate(dof_names):
                pot_wl = solver.compute_potential(wl_eval_pts, rad_results[dof])
                rad_pot_wl[i, :] = pot_wl
        
        for beta in betas:
            print(f"  beta={np.degrees(beta):.0f}°")
            
            # 2. Solve diffraction
            diff_prob = cpt.DiffractionProblem(
                body=body, wave_direction=beta, omega=omega,
                water_depth=np.inf, rho=rho, g=g
            )
            diff_result = solver.solve(diff_prob)
            
            # Diffraction potential on hull
            diff_pot_hull = diff_result.potential[hull_mask]
            
            # Diffraction velocity on hull
            diff_vel_hull = solver.compute_velocity(hull_centers, diff_result)
            
            # Diffraction potential at waterline
            diff_pot_wl = np.zeros(len(edges), dtype=complex)
            if len(edges) > 0:
                diff_pot_wl = solver.compute_potential(wl_eval_pts, diff_result)
            
            # Incident wave potential on hull
            inc_pot_hull = airy_waves_potential(hull_centers, diff_prob)
            
            # Incident wave velocity on hull
            inc_vel_hull = airy_waves_velocity(hull_centers, diff_prob)
            
            # Incident wave potential at waterline
            inc_pot_wl = np.zeros(len(edges), dtype=complex)
            if len(edges) > 0:
                inc_pot_wl = airy_waves_potential(wl_eval_pts, diff_prob)
            
            # Excitation forces and RAOs
            # diff_result.forces contains ONLY the diffraction (scattered) force.
            # Must add Froude-Krylov to get total excitation.
            FK = froude_krylov_force(diff_prob)
            F_exc = np.array([diff_result.forces[dof] + FK[dof] for dof in dof_names])
            Z = -omega**2 * (M + A) + 1j * omega * B_mat + C
            xi = np.linalg.solve(Z, F_exc)
            
            # 3. TOTAL first-order potential = incident + diffraction + sum(xi_j * radiation_j)
            # On hull panels:
            total_pot_hull = inc_pot_hull + diff_pot_hull
            total_vel_hull = inc_vel_hull + diff_vel_hull
            for i in range(n_dof):
                total_pot_hull += xi[i] * rad_potential_hull[i, :]
                total_vel_hull += xi[i] * rad_velocity_hull[i, :, :]
            
            # At waterline:
            total_pot_wl = inc_pot_wl + diff_pot_wl
            for i in range(n_dof):
                total_pot_wl += xi[i] * rad_pot_wl[i, :]
            
            # 4. DRIFT FORCE COMPUTATION
            
            # === Term 1: Waterline integral ===
            # F̄_i^WL = (1/2) ρg ∮_WL |η_rel|² n_i dl
            #
            # For a FIXED body: η_rel = η = (iω/g) φ_total at z=0
            # For a MOVING body: η_rel = η - body_displacement_z at waterline
            #
            # The wave elevation: η = (iω/g) * φ_total  (Capytaine convention)
            # Body vertical displacement at waterline point x:
            #   z_body(x) = ξ_3 + ξ_4 * y - ξ_5 * x  (heave + roll*y - pitch*x)
            #
            # η_rel = η - z_body
            
            # Wave elevation at waterline
            eta_wl = (1j * omega / g) * total_pot_wl
            
            # Body vertical displacement at each waterline edge center
            # xi = [Surge, Sway, Heave, Roll, Pitch, Yaw]
            xi_surge, xi_sway, xi_heave = xi[0], xi[1], xi[2]
            xi_roll, xi_pitch, xi_yaw = xi[3], xi[4], xi[5]
            
            # Vertical body displacement at waterline points
            z_body_wl = (xi_heave 
                        + xi_roll * edge_centers[:, 1] 
                        - xi_pitch * edge_centers[:, 0])
            
            eta_rel = eta_wl - z_body_wl
            
            # Waterline drift force
            # Time average: <η²(t)> = (1/2)|η̂|², so the full formula is:
            # F̄_i^WL = (1/2)ρg <η²_rel> n_i dl = (1/4)ρg |η̂_rel|² n_i dl
            F_wl = np.zeros(3)
            if len(edges) > 0:
                # |η_rel|² at each edge
                eta_rel_sq = np.abs(eta_rel)**2
                
                # F_wl_i = (1/4) ρg Σ |η̂_rel|² n_i dl
                for i_comp in range(3):
                    F_wl[i_comp] = 0.25 * rho * g * np.sum(
                        eta_rel_sq * edge_normals[:, i_comp] * edge_lengths)
            
            # === Term 2: Velocity squared (Bernoulli) on hull surface ===
            # F̄_i^vel = -(1/4) ρ ∫∫_Sb |∇φ̂_total|² n_i dS
            #   (1/4 = 1/2 from formula × 1/2 from time averaging)
            #
            # But with body motion, the relative velocity matters:
            # ∇φ_rel = ∇φ_total - v_body
            # where v_body = iω * [ξ_1:3 + ξ_4:6 × (x - x_rot)]
            #
            # Actually, the standard Pinkster formula uses the TOTAL potential gradient
            # (not relative), because the body surface integral already accounts for
            # the body motion through the moving boundary.
            #
            # The correct formula (Pinkster 1979 eq 3.34) is:
            # F̄ = -(1/2)ρ ∫∫ |∇φ|² n dS  (using total ∇φ in fluid)
            #
            # However, at the body surface, the NORMAL component of ∇φ equals
            # the normal body velocity (kinematic BC), while the TANGENTIAL 
            # components come from the potential gradient along the surface.
            #
            # Capytaine's compute_velocity gives ∇φ at the face centers, which
            # is the total velocity in the fluid at those points.
            
            vel_sq = np.sum(np.abs(total_vel_hull)**2, axis=1)  # |∇φ|² per panel
            
            F_vel = np.zeros(3)
            for i_comp in range(3):
                F_vel[i_comp] = -0.25 * rho * np.sum(
                    vel_sq * hull_normals[:, i_comp] * hull_areas)
            
            # === Term 3: Rotation correction ===
            # F̄_i^rot = (1/2) Re[ ∫∫_Sb p_total * (n × α*) dS ] · e_i
            #
            # where α = rotation vector = (ξ_4, ξ_5, ξ_6)
            # p_total = iωρ φ_total (Capytaine convention)
            #
            # The cross product n × α* gives a 3-vector for each panel.
            # Then dotted with the total pressure and integrated.
            
            alpha_conj = np.conj(np.array([xi_roll, xi_pitch, xi_yaw]))
            p_total = 1j * omega * rho * total_pot_hull  # pressure per panel
            
            F_rot = np.zeros(3)
            for fi in range(n_hull):
                n_cross_alpha = np.cross(hull_normals[fi], alpha_conj)
                F_rot += 0.5 * np.real(p_total[fi] * n_cross_alpha) * hull_areas[fi]
            
            # === Total drift force ===
            F_total = F_wl + F_vel + F_rot
            
            # Store results
            results[(omega, beta)] = {
                'xi': xi,
                'F_wl': F_wl.copy(),
                'F_vel': F_vel.copy(),
                'F_rot': F_rot.copy(),
                'F_total': F_total.copy(),
                'eta_rel_rms': np.sqrt(np.mean(np.abs(eta_rel)**2)) if len(edges) > 0 else 0,
            }
            
            mu_deg = np.degrees(beta)
            print(f"    F_wl   = ({F_wl[0]:12.1f}, {F_wl[1]:12.1f}, {F_wl[2]:12.1f})")
            print(f"    F_vel  = ({F_vel[0]:12.1f}, {F_vel[1]:12.1f}, {F_vel[2]:12.1f})")
            print(f"    F_rot  = ({F_rot[0]:12.1f}, {F_rot[1]:12.1f}, {F_rot[2]:12.1f})")
            print(f"    TOTAL  = ({F_total[0]:12.1f}, {F_total[1]:12.1f}, {F_total[2]:12.1f})")
            print(f"    RAOs: surge={np.abs(xi[0]):.4f} sway={np.abs(xi[1]):.4f} "
                  f"heave={np.abs(xi[2]):.4f} roll={np.abs(xi[3]):.4f} "
                  f"pitch={np.abs(xi[4]):.4f} yaw={np.abs(xi[5]):.4f}")
    
    return results


# ============================================================
# Parse pdstrip
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
    output_dir = "/home/blofro/src/pdstrip_test/validation"
    
    print("=" * 80)
    print("NEAR-FIELD DRIFT FORCE VALIDATION: Capytaine vs pdstrip")
    print("Direct pressure/velocity integration on body surface")
    print("=" * 80)
    
    body, mass_val, zcg, inertia = make_hull_body(R, L, name="monohull")
    print(f"Mesh: {body.mesh.nb_faces} hull + {body.lid_mesh.nb_faces} lid faces")
    print(f"Mass: {mass_val:.1f} kg, zcg: {zcg:.4f} m")
    print(f"Ixx: {inertia[0,0]:.1f}, Iyy: {inertia[1,1]:.1f}, Izz: {inertia[2,2]:.1f} kg·m²")
    print()
    
    # Compute drift forces
    drift_results = compute_nearfield_drift(
        body, mass_val, zcg, inertia, omega_values, wave_directions)
    
    # Parse pdstrip results
    pdstrip_debug = os.path.join(output_dir, "run_mono", "debug.out")
    pdstrip_data = parse_pdstrip_drift(pdstrip_debug)
    
    # ============================================================
    # Comparison tables
    # ============================================================
    
    def ratio(a, b):
        return a / b if abs(b) > 1.0 else float('nan')
    
    # BEAM SEAS - Fy
    # Coordinate mapping: pdstrip y=starboard, Capytaine y=port
    # So Fy_capytaine_convention = -feta_pdstrip
    # beta=pi/2 (Capytaine) = mu=90 (pdstrip): wave FROM starboard, propagates toward port
    print()
    print("=" * 100)
    print("BEAM SEAS (mu=90°, beta=π/2): Fy comparison (Capytaine y=port convention)")
    print("  pd_Fy = -feta (converted from pdstrip starboard to Capytaine port)")
    print("=" * 100)
    print(f"{'lam':>5s} {'pd_Fy':>10s} {'cap_Fy':>10s} {'cap_WL':>10s} {'cap_vel':>10s} "
          f"{'cap_rot':>10s} {'pd/cap':>8s}")
    print("-" * 70)
    
    for omega, lam in zip(omega_values, wavelengths):
        pd_match = [d for d in pdstrip_data
                    if abs(d['omega'] - omega) < 0.01 and abs(d['mu_deg'] - 90.0) < 1.0]
        # Negate feta to convert from starboard-positive to port-positive
        pd_fy = -pd_match[0]['feta'] if pd_match else float('nan')
        
        key = None
        for (om, beta) in drift_results:
            if abs(om - omega) < 0.01 and abs(beta - np.pi/2) < 0.01:
                key = (om, beta)
                break
        
        if key:
            r = drift_results[key]
            cap_fy = r['F_total'][1]
            print(f"{lam:5.0f} {pd_fy:10.1f} {cap_fy:10.1f} {r['F_wl'][1]:10.1f} "
                  f"{r['F_vel'][1]:10.1f} {r['F_rot'][1]:10.1f} "
                  f"{ratio(pd_fy, cap_fy):8.2f}")
    
    # HEAD SEAS - Fx
    print()
    print("=" * 100)
    print("HEAD SEAS (mu=180°, beta=π): Fx comparison")
    print("=" * 100)
    print(f"{'lam':>5s} {'pd_Fx':>10s} {'cap_Fx':>10s} {'cap_WL':>10s} {'cap_vel':>10s} "
          f"{'cap_rot':>10s} {'pd/cap':>8s}")
    print("-" * 70)
    
    for omega, lam in zip(omega_values, wavelengths):
        pd_match = [d for d in pdstrip_data
                    if abs(d['omega'] - omega) < 0.01 and abs(d['mu_deg'] - 180.0) < 1.0]
        pd_fx = pd_match[0]['fxi'] if pd_match else float('nan')
        
        key = None
        for (om, beta) in drift_results:
            if abs(om - omega) < 0.01 and abs(beta - np.pi) < 0.01:
                key = (om, beta)
                break
        
        if key:
            r = drift_results[key]
            cap_fx = r['F_total'][0]
            print(f"{lam:5.0f} {pd_fx:10.1f} {cap_fx:10.1f} {r['F_wl'][0]:10.1f} "
                  f"{r['F_vel'][0]:10.1f} {r['F_rot'][0]:10.1f} "
                  f"{ratio(pd_fx, cap_fx):8.2f}")
    
    # FOLLOWING SEAS - Fx
    print()
    print("=" * 100)
    print("FOLLOWING SEAS (mu=0°, beta=0): Fx comparison")
    print("=" * 100)
    print(f"{'lam':>5s} {'pd_Fx':>10s} {'cap_Fx':>10s} {'cap_WL':>10s} {'cap_vel':>10s} "
          f"{'cap_rot':>10s} {'pd/cap':>8s}")
    print("-" * 70)
    
    for omega, lam in zip(omega_values, wavelengths):
        pd_match = [d for d in pdstrip_data
                    if abs(d['omega'] - omega) < 0.01 and abs(d['mu_deg'] - 0.0) < 1.0]
        pd_fx = pd_match[0]['fxi'] if pd_match else float('nan')
        
        key = None
        for (om, beta) in drift_results:
            if abs(om - omega) < 0.01 and abs(beta - 0.0) < 0.01:
                key = (om, beta)
                break
        
        if key:
            r = drift_results[key]
            cap_fx = r['F_total'][0]
            print(f"{lam:5.0f} {pd_fx:10.1f} {cap_fx:10.1f} {r['F_wl'][0]:10.1f} "
                  f"{r['F_vel'][0]:10.1f} {r['F_rot'][0]:10.1f} "
                  f"{ratio(pd_fx, cap_fx):8.2f}")
    
    # Save
    np.savez(os.path.join(output_dir, "nearfield_drift_comparison.npz"),
             wavelengths=wavelengths,
             omega_values=omega_values,
             **{f"beam_{comp}_{term}": np.array([
                 drift_results.get((om, np.pi/2), {}).get(f'F_{term}', np.zeros(3))[ci]
                 for om in omega_values])
                for ci, comp in enumerate(['Fx', 'Fy', 'Fz'])
                for term in ['wl', 'vel', 'rot', 'total']},
             **{f"head_{comp}_{term}": np.array([
                 drift_results.get((om, np.pi), {}).get(f'F_{term}', np.zeros(3))[ci]
                 for om in omega_values])
                for ci, comp in enumerate(['Fx', 'Fy', 'Fz'])
                for term in ['wl', 'vel', 'rot', 'total']},
    )
    print(f"\nResults saved to {os.path.join(output_dir, 'nearfield_drift_comparison.npz')}")
