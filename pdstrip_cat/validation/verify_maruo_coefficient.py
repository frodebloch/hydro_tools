#!/usr/bin/env python3
"""
Verify the Maruo far-field drift force coefficient by comparing
against the near-field result for a FIXED body.

For a fixed body, the near-field drift force has only:
  - Waterline term: F_wl = +¼ ρg ∮ |η̂|² n dl
  - Velocity term:  F_vel = +¼ ρ ∫∫ |∇φ̂|² n dS  [with n pointing INTO fluid]

(The +¼ρ sign for the velocity term follows from:
  Force on body = -∫∫ p n_out dS, where n_out = Capytaine normal
  p₂ = -½ρ|∇φ₁|², time avg → +¼ρ|∇φ̂|²
  F_vel = -∫∫ p₂ n_out = +¼ρ ∫∫ |∇φ̂|² n_out)

Wait — for the ORIGINAL near-field script (which was verified to work),
the velocity term used -0.25. Let me check both signs.

The far-field formula candidate is:
  F_y = C × ∫|a(θ)|² sin(θ) dθ

where a(θ) = (iω/g) φ_d √r exp(-ikr) is the far-field elevation amplitude,
verified by B_jj = (ρg²)/(2ω³) ∫|a_j|² dθ.

We test: what value of C makes F_y^{far} = F_y^{near} ?
"""

import numpy as np
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_potential, airy_waves_velocity
import logging

cpt.set_logging(logging.WARNING)

R = 1.0; L = 20.0; rho = 1025.0; g = 9.81
mesh_res = (10, 40, 50)

# Build body
mesh_full = cpt.mesh_horizontal_cylinder(
    length=L, radius=R, center=(0, 0, 0), resolution=mesh_res, name="hull")
hull_mesh = mesh_full.immersed_part()
lid = hull_mesh.generate_lid(z=-0.01)
body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name="hull")
body.center_of_mass = np.array([0.0, 0.0, -4*R/(3*np.pi)])
body.rotation_center = body.center_of_mass
body.add_all_rigid_body_dofs()

solver = cpt.BEMSolver()

hull_normals = hull_mesh.faces_normals
hull_areas = hull_mesh.faces_areas
hull_centers = hull_mesh.faces_centers
n_hull = hull_mesh.nb_faces

# Waterline edges
def extract_waterline_edges(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    z_tol = 0.05
    wl_mask = np.abs(vertices[:, 2]) < z_tol
    wl_verts = set(np.where(wl_mask)[0])
    edges = set()
    face_for_edge = {}
    for fi, face in enumerate(faces):
        verts = [face[0], face[1], face[2]] if face[0] == face[3] else list(face)
        n_v = len(verts)
        for kk in range(n_v):
            v1, v2 = verts[kk], verts[(kk+1) % n_v]
            if v1 in wl_verts and v2 in wl_verts:
                edge = (min(v1, v2), max(v1, v2))
                edges.add(edge)
                face_for_edge[edge] = fi
    edges = list(edges)
    n_edges = len(edges)
    if n_edges == 0:
        return [], np.zeros((0,3)), np.zeros(0), np.zeros((0,3))
    edge_centers = np.zeros((n_edges, 3))
    edge_lengths = np.zeros(n_edges)
    edge_normals = np.zeros((n_edges, 3))
    face_centers = mesh.faces_centers
    for i, (v1, v2) in enumerate(edges):
        p1, p2 = vertices[v1], vertices[v2]
        edge_centers[i] = 0.5*(p1+p2)
        edge_vec = p2 - p1
        edge_lengths[i] = np.linalg.norm(edge_vec)
        t = edge_vec / (edge_lengths[i]+1e-30)
        n1 = np.array([t[1], -t[0], 0.0])
        n2 = np.array([-t[1], t[0], 0.0])
        fi = face_for_edge[(min(v1,v2), max(v1,v2))]
        fc = face_centers[fi]
        to_face = fc[:2] - edge_centers[i,:2]
        edge_normals[i] = n1 if np.dot(n1[:2], to_face) < 0 else n2
    return edges, edge_centers, edge_lengths, edge_normals

edges, edge_centers, edge_lengths, edge_normals = extract_waterline_edges(hull_mesh)
wl_eval_pts = edge_centers.copy()
wl_eval_pts[:, 2] = -0.001

# Field-point ring
N_THETA = 720
R_FIELD = 5000.0
theta_fp = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
dtheta = 2*np.pi / N_THETA
field_pts = np.column_stack([
    R_FIELD * np.cos(theta_fp),
    R_FIELD * np.sin(theta_fp),
    np.zeros(N_THETA)
])

beta = np.pi / 2  # beam seas
wavelengths = [3, 5, 10, 22, 55, 90]

print("FIXED BODY: Verify Maruo coefficient")
print(f"Body: semicircular barge R={R}m, L={L}m")
print(f"Mesh: {hull_mesh.nb_faces} hull + {lid.nb_faces} lid panels")
print(f"Field ring: r={R_FIELD}m, N={N_THETA}")
print()
print(f"{'lam':>5} {'k':>8} {'NF(+vel)':>12} {'NF(-vel)':>12} "
      f"{'∫|a|²sinθ':>14} {'C_eff(+)':>12} {'C_eff(-)':>12} "
      f"{'ρg/k':>12} {'ratio(+)':>10} {'ratio(-)':>10} {'B_ratio':>8}")
print("-"*140)

for lam in wavelengths:
    k = 2*np.pi / lam
    omega = np.sqrt(k*g)
    
    # Diffraction solve
    diff_prob = cpt.DiffractionProblem(body=body, wave_direction=beta,
                                        omega=omega, water_depth=np.inf)
    diff_result = solver.solve(diff_prob, keep_details=True)
    
    # Total potential on hull = incident + scattered
    inc_pot_hull = airy_waves_potential(hull_centers, diff_prob)
    diff_pot_hull = solver.compute_potential(hull_centers, diff_result)
    total_pot_hull = inc_pot_hull + diff_pot_hull
    
    inc_vel_hull = airy_waves_velocity(hull_centers, diff_prob)
    diff_vel_hull = solver.compute_velocity(hull_centers, diff_result)
    total_vel_hull = inc_vel_hull + diff_vel_hull
    
    inc_pot_wl = airy_waves_potential(wl_eval_pts, diff_prob)
    diff_pot_wl = solver.compute_potential(wl_eval_pts, diff_result)
    total_pot_wl = inc_pot_wl + diff_pot_wl
    
    # === Near-field: waterline ===
    eta_wl = (1j * omega / g) * total_pot_wl
    F_wl_y = 0.25 * rho * g * np.sum(np.abs(eta_wl)**2 * edge_normals[:, 1] * edge_lengths)
    
    # === Near-field: velocity (try BOTH signs) ===
    vel_sq = np.sum(np.abs(total_vel_hull)**2, axis=1)
    F_vel_y_pos = +0.25 * rho * np.sum(vel_sq * hull_normals[:, 1] * hull_areas)
    F_vel_y_neg = -0.25 * rho * np.sum(vel_sq * hull_normals[:, 1] * hull_areas)
    
    F_nf_pos = F_wl_y + F_vel_y_pos  # with +0.25 (our derivation)
    F_nf_neg = F_wl_y + F_vel_y_neg  # with -0.25 (original code)
    
    # === Far-field: scattered wave amplitude ===
    phi_scat_fp = solver.compute_potential(field_pts, diff_result)
    a_scat = (1j * omega / g) * phi_scat_fp * np.sqrt(R_FIELD) * np.exp(-1j * k * R_FIELD)
    
    int_a2_sin = np.sum(np.abs(a_scat)**2 * np.sin(theta_fp)) * dtheta
    
    # Empirical coefficient: NF = C_eff × ∫|a|²sinθ
    C_eff_pos = F_nf_pos / int_a2_sin if abs(int_a2_sin) > 1e-15 else float('nan')
    C_eff_neg = F_nf_neg / int_a2_sin if abs(int_a2_sin) > 1e-15 else float('nan')
    
    C_theory = rho * g / k
    ratio_pos = C_eff_pos / C_theory if abs(C_theory) > 0 else float('nan')
    ratio_neg = C_eff_neg / C_theory if abs(C_theory) > 0 else float('nan')
    
    # Damping verification
    rad_prob = cpt.RadiationProblem(body=body, radiating_dof='Sway', omega=omega, water_depth=np.inf)
    rad_result = solver.solve(rad_prob, keep_details=True)
    B_direct = rad_result.radiation_dampings['Sway']
    phi_rad_fp = solver.compute_potential(field_pts, rad_result)
    a_rad = (1j * omega / g) * phi_rad_fp * np.sqrt(R_FIELD) * np.exp(-1j * k * R_FIELD)
    B_from_a = rho * g**2 / (2 * omega**3) * np.sum(np.abs(a_rad)**2) * dtheta
    B_ratio = B_direct / B_from_a
    
    print(f"{lam:5.0f} {k:8.4f} {F_nf_pos:12.1f} {F_nf_neg:12.1f} "
          f"{int_a2_sin:14.6e} {C_eff_pos:12.1f} {C_eff_neg:12.1f} "
          f"{C_theory:12.1f} {ratio_pos:10.4f} {ratio_neg:10.4f} {B_ratio:8.4f}")

print()
print("ratio(+) = C_eff(+vel) / (ρg/k): if ~1.0, the +vel sign is correct and Maruo uses ρg/k")
print("ratio(-) = C_eff(-vel) / (ρg/k): if ~1.0, the -vel sign is correct and Maruo uses ρg/k")
print()
print("If neither ratio is close to 1.0, the far-field formula F = (ρg/k)∫|a_s|²sinθ dθ")
print("is INCOMPLETE for a fixed body — there must be a cross-term with the incident wave.")
print()
print("The COMPLETE Maruo formula should be:")
print("  F_y = C₁ ∫|a_s|² sinθ dθ + C₂ × (cross-term)")
print()
print("Let's also check if (ρg/k) × total_a (inc+scat) works instead of just scattered:")
print()

# Now try with TOTAL a(θ) including incident wave
print(f"{'lam':>5} {'F_nf(-vel)':>12} {'FF_scat':>12} {'FF_total_a':>12} {'NF/FF_s':>10} {'NF/FF_t':>10}")
print("-"*65)

for lam in wavelengths:
    k = 2*np.pi / lam
    omega = np.sqrt(k*g)
    
    diff_prob = cpt.DiffractionProblem(body=body, wave_direction=beta,
                                        omega=omega, water_depth=np.inf)
    diff_result = solver.solve(diff_prob, keep_details=True)
    
    # Near-field (with -0.25 velocity term)
    inc_pot_hull = airy_waves_potential(hull_centers, diff_prob)
    diff_pot_hull = solver.compute_potential(hull_centers, diff_result)
    inc_vel_hull = airy_waves_velocity(hull_centers, diff_prob)
    diff_vel_hull = solver.compute_velocity(hull_centers, diff_result)
    total_vel_hull = inc_vel_hull + diff_vel_hull
    total_pot_wl = airy_waves_potential(wl_eval_pts, diff_prob) + solver.compute_potential(wl_eval_pts, diff_result)
    
    eta_wl = (1j * omega / g) * total_pot_wl
    F_wl_y = 0.25 * rho * g * np.sum(np.abs(eta_wl)**2 * edge_normals[:, 1] * edge_lengths)
    vel_sq = np.sum(np.abs(total_vel_hull)**2, axis=1)
    F_vel_y = -0.25 * rho * np.sum(vel_sq * hull_normals[:, 1] * hull_areas)
    F_nf = F_wl_y + F_vel_y
    
    # Far-field: scattered only
    phi_scat_fp = solver.compute_potential(field_pts, diff_result)
    a_scat = (1j * omega / g) * phi_scat_fp * np.sqrt(R_FIELD) * np.exp(-1j * k * R_FIELD)
    FF_scat = (rho * g / k) * np.sum(np.abs(a_scat)**2 * np.sin(theta_fp)) * dtheta
    
    # Far-field: total (incident + scattered)
    # Incident wave: η_inc = exp(ik(x cosβ + y sinβ))
    # At field points: η_inc = exp(ikR cos(θ-β))
    # In the far field: using stationary phase for cylindrical coordinates:
    #   η_inc ≈ √(2π/(kR)) × [exp(ikR - iπ/4) × δ(θ-β) + ...]
    # This doesn't work for the |η_total|² approach because the incident wave
    # is a plane wave, not cylindrical.
    
    # Instead, compute the total potential at field points
    phi_inc_fp = airy_waves_potential(field_pts, diff_prob)
    phi_total_fp = phi_inc_fp + phi_scat_fp
    
    # Total elevation amplitude at field points
    eta_total_fp = (1j * omega / g) * phi_total_fp
    eta_inc_fp = (1j * omega / g) * phi_inc_fp
    
    # |η_total|² - |η_inc|² = |η_s|² + 2Re(η_s × η_inc*)
    cross_plus_quad = np.abs(eta_total_fp)**2 - np.abs(eta_inc_fp)**2
    
    # Try the momentum-flux based formula:
    # F_y = -(ρg/(4k)) × R × ∫₀²π [|η_total|² - |η_inc|²] sin(θ) dθ
    # The R and 1/√r factors should cancel if we use the raw potentials
    
    # Actually, the deep-water momentum flux formula (from Faltinsen) is:
    # F_y = (ρg/(4k)) ∫₀²π [|η_total|² - |η_inc|²] ê_y dS_contour
    # where the contour integral is ∫₀²π ... R dθ, with R the radius.
    # But |η| values depend on position, so this only works in the far field
    # where the scattered wave has decayed as 1/√r.
    
    # Let's try: use |η|² at the actual field points
    int_diff_sin = np.sum(cross_plus_quad * np.sin(theta_fp)) * dtheta
    
    # Various candidate formulas:
    FF_try1 = -(rho * g / (4*k)) * R_FIELD * int_diff_sin
    FF_try2 = +(rho * g / (4*k)) * R_FIELD * int_diff_sin
    
    ratio_s = F_nf / FF_scat if abs(FF_scat) > 0.1 else float('nan')
    ratio_t1 = F_nf / FF_try1 if abs(FF_try1) > 0.1 else float('nan')
    
    print(f"{lam:5.0f} {F_nf:12.1f} {FF_scat:12.1f} {FF_try1:12.1f} {ratio_s:10.4f} {ratio_t1:10.4f}")

print()
print("Done.")
