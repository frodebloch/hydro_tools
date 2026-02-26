#!/usr/bin/env python3
"""
Empirically determine the correct Maruo formula coefficients by fitting
against the known near-field drift force for a fixed body.

We have a(θ) at each wavelength and the NF drift force.
The Maruo formula should be:
  F_y = α × ∫|a_d|² sinθ dθ  +  γ × Re[e^{iψ} a_d(β)] sinβ

where α, γ, ψ are to be determined (they may depend on k, ω, etc.)

We'll also try: F_y = α(k) × ∫|a_d|² sinθ dθ  +  γ(k) × cross

to see if the coefficients have a systematic k-dependence.
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

# Field-point ring
N_THETA = 720
theta_fp = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
dtheta = 2*np.pi / N_THETA
R_FIELD = 5000.0

field_pts = np.column_stack([
    R_FIELD * np.cos(theta_fp),
    R_FIELD * np.sin(theta_fp),
    np.zeros(N_THETA)
])

# Waterline extraction
def extract_waterline_edges(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    z_tol = 0.05
    wl_mask = np.abs(vertices[:, 2]) < z_tol
    wl_verts = set(np.where(wl_mask)[0])
    edges_list = set()
    face_for_edge = {}
    for fi, face in enumerate(faces):
        verts = [face[0], face[1], face[2]] if face[0] == face[3] else [face[0], face[1], face[2], face[3]]
        n_verts = len(verts)
        for kk in range(n_verts):
            v1 = verts[kk]; v2 = verts[(kk+1)%n_verts]
            if v1 in wl_verts and v2 in wl_verts:
                edge = (min(v1,v2), max(v1,v2))
                edges_list.add(edge)
                face_for_edge[edge] = fi
    edges_list = list(edges_list)
    n_edges = len(edges_list)
    if n_edges == 0:
        return [], np.zeros((0,3)), np.zeros(0), np.zeros((0,3))
    edge_centers = np.zeros((n_edges, 3))
    edge_lengths = np.zeros(n_edges)
    edge_normals = np.zeros((n_edges, 3))
    face_normals = mesh.faces_normals
    face_centers = mesh.faces_centers
    for i, (v1, v2) in enumerate(edges_list):
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
    return edges_list, edge_centers, edge_lengths, edge_normals

edges, edge_centers, edge_lengths, edge_normals = extract_waterline_edges(hull_mesh)
hull_normals = hull_mesh.faces_normals
hull_areas = hull_mesh.faces_areas
hull_centers = hull_mesh.faces_centers
wl_eval_pts = edge_centers.copy()
wl_eval_pts[:, 2] = -0.001

beta = np.pi / 2  # beam seas
wavelengths = [3, 4, 5, 6, 8, 10, 13, 17, 22, 28, 35, 45, 55, 70, 90]

print("="*120)
print("Empirical Maruo coefficient analysis: FIXED BODY, beam seas")
print("="*120)
print(f"\n{'lam':>5} {'k':>8} {'NF_y':>12} "
      f"{'∫|a|²sinθ':>14} {'Re_ad_beta':>12} {'Im_ad_beta':>12} "
      f"{'α_eff':>12} {'α_eff/k':>10}")
print("-"*100)

results = []
for lam in wavelengths:
    k = 2*np.pi / lam
    omega = np.sqrt(k*g)
    
    # Diffraction solve
    diff_prob = cpt.DiffractionProblem(body=body, wave_direction=beta,
                                        omega=omega, water_depth=np.inf)
    diff_result = solver.solve(diff_prob, keep_details=True)
    
    # === Near-field drift ===
    inc_pot_hull = airy_waves_potential(hull_centers, diff_prob)
    diff_pot_hull = solver.compute_potential(hull_centers, diff_result)
    total_pot_hull = inc_pot_hull + diff_pot_hull
    inc_vel_hull = airy_waves_velocity(hull_centers, diff_prob)
    diff_vel_hull = solver.compute_velocity(hull_centers, diff_result)
    total_vel_hull = inc_vel_hull + diff_vel_hull
    inc_pot_wl = airy_waves_potential(wl_eval_pts, diff_prob)
    diff_pot_wl = solver.compute_potential(wl_eval_pts, diff_result)
    total_pot_wl = inc_pot_wl + diff_pot_wl
    
    eta_wl = (1j * omega / g) * total_pot_wl
    F_wl_y = 0.25 * rho * g * np.sum(np.abs(eta_wl)**2 * edge_normals[:, 1] * edge_lengths)
    vel_sq = np.sum(np.abs(total_vel_hull)**2, axis=1)
    F_vel_y = -0.25 * rho * np.sum(vel_sq * hull_normals[:, 1] * hull_areas)
    F_nf_y = F_wl_y + F_vel_y
    
    # === Far-field amplitude ===
    phi_scat_fp = solver.compute_potential(field_pts, diff_result)
    a_scat = (1j * omega / g) * phi_scat_fp * np.sqrt(R_FIELD) * np.exp(-1j * k * R_FIELD)
    
    a2 = np.abs(a_scat)**2
    int_a2_sin = np.sum(a2 * np.sin(theta_fp)) * dtheta
    
    idx_beta = np.argmin(np.abs(theta_fp - beta))
    a_at_beta = a_scat[idx_beta]
    
    # Also get damping to verify a(θ) is correct
    rad_prob = cpt.RadiationProblem(body=body, radiating_dof='Sway', omega=omega, water_depth=np.inf)
    rad_result = solver.solve(rad_prob, keep_details=True)
    B_direct = rad_result.radiation_dampings['Sway']
    phi_rad_fp = solver.compute_potential(field_pts, rad_result)
    a_rad = (1j * omega / g) * phi_rad_fp * np.sqrt(R_FIELD) * np.exp(-1j * k * R_FIELD)
    B_from_a = rho * g**2 / (2 * omega**3) * np.sum(np.abs(a_rad)**2) * dtheta
    B_ratio = B_direct / B_from_a
    
    # Effective coefficient: if F_nf = α × ∫|a|²sinθ, what is α?
    alpha_eff = F_nf_y / int_a2_sin if abs(int_a2_sin) > 1e-15 else float('nan')
    alpha_over_k = alpha_eff / k if abs(k) > 0 else float('nan')
    
    print(f"{lam:5.0f} {k:8.4f} {F_nf_y:12.1f} "
          f"{int_a2_sin:14.6e} {a_at_beta.real:12.4f} {a_at_beta.imag:12.4f} "
          f"{alpha_eff:12.1f} {alpha_over_k:10.1f}")
    
    results.append({
        'lam': lam, 'k': k, 'omega': omega,
        'NF_y': F_nf_y,
        'int_a2_sin': int_a2_sin,
        'a_at_beta': a_at_beta,
        'B_ratio': B_ratio,
        'alpha_eff': alpha_eff,
    })

# Now try to fit: F_y = alpha * ∫|a|²sinθ + gamma * Re[exp(i*psi) * a_d(beta)] * sin(beta)
# with various functional forms
print("\n\nFitting analysis:")
print("="*100)

# Option 1: Is alpha_eff ∝ k? (i.e., α = C×k)
# Option 2: Is alpha_eff ∝ 1/k? 
# Option 3: Is alpha_eff constant?
# Option 4: Does it include a(β) correction?

# Check if there's a simple 2-parameter fit 
from scipy.optimize import least_squares

NF_arr = np.array([r['NF_y'] for r in results])
a2sin_arr = np.array([r['int_a2_sin'] for r in results])
k_arr = np.array([r['k'] for r in results])
omega_arr = np.array([r['omega'] for r in results])
a_beta_arr = np.array([r['a_at_beta'] for r in results])

# Try: F = (ρg × C_q / k) × ∫|a|² sinθ  +  (ρg × C_c) × Re[exp(-iπ/4) a(β)] sinβ
# where C_q and C_c are dimensionless constants
def model1(params):
    C_q, C_c = params
    F_pred = (rho * g * C_q / k_arr) * a2sin_arr + \
             (rho * g * C_c) * np.real(np.exp(-1j*np.pi/4) * a_beta_arr) * np.sin(beta)
    return (F_pred - NF_arr) / np.abs(NF_arr).clip(min=100)

res1 = least_squares(model1, [1.0, 1.0])
print(f"\nModel 1: F = (ρg C_q/k) ∫|a|²sinθ + (ρg C_c) Re[e^(-iπ/4) a(β)] sinβ")
print(f"  C_q = {res1.x[0]:.6f}, C_c = {res1.x[1]:.6f}")
print(f"  Residual norm: {np.sqrt(np.sum(res1.fun**2)):.6f}")

# Try model with Re(a(β)) and Im(a(β)) separately
def model2(params):
    C_q, C_re, C_im = params
    F_pred = (rho * g * C_q / k_arr) * a2sin_arr + \
             (rho * g * C_re) * a_beta_arr.real * np.sin(beta) + \
             (rho * g * C_im) * a_beta_arr.imag * np.sin(beta)
    return (F_pred - NF_arr) / np.abs(NF_arr).clip(min=100)

res2 = least_squares(model2, [1.0, 0.5, 0.5])
print(f"\nModel 2: F = (ρg C_q/k) ∫|a|²sinθ + ρg sinβ [C_re Re(a(β)) + C_im Im(a(β))]")
print(f"  C_q = {res2.x[0]:.6f}, C_re = {res2.x[1]:.6f}, C_im = {res2.x[2]:.6f}")
print(f"  Residual norm: {np.sqrt(np.sum(res2.fun**2)):.6f}")

# Try: F = (ρg C_q) × ∫|a|² sinθ  +  ...
def model3(params):
    C_q, C_c = params
    F_pred = (rho * g * C_q) * a2sin_arr + \
             (rho * g * C_c * np.sqrt(2*np.pi/k_arr)) * np.real(np.exp(-1j*np.pi/4) * a_beta_arr) * np.sin(beta)
    return (F_pred - NF_arr) / np.abs(NF_arr).clip(min=100)

res3 = least_squares(model3, [1.0, 1.0])
print(f"\nModel 3: F = ρg C_q ∫|a|²sinθ + ρg C_c √(2π/k) Re[e^(-iπ/4) a(β)] sinβ")
print(f"  C_q = {res3.x[0]:.6f}, C_c = {res3.x[1]:.6f}")
print(f"  Residual norm: {np.sqrt(np.sum(res3.fun**2)):.6f}")

# Print detailed comparison for best model
print("\n\nBest model (Model 2) predictions vs NF:")
print(f"{'lam':>5} {'NF_y':>12} {'FF_y':>12} {'ratio':>8} {'B_ratio':>8}")
print("-"*50)
C_q, C_re, C_im = res2.x
for r in results:
    F_pred = (rho * g * C_q / r['k']) * r['int_a2_sin'] + \
             (rho * g * C_re) * r['a_at_beta'].real * np.sin(beta) + \
             (rho * g * C_im) * r['a_at_beta'].imag * np.sin(beta)
    ratio = r['NF_y'] / F_pred if abs(F_pred) > 0.1 else float('inf')
    print(f"{r['lam']:5.0f} {r['NF_y']:12.1f} {F_pred:12.1f} {ratio:8.4f} {r['B_ratio']:8.4f}")
