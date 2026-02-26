#!/usr/bin/env python3
"""
Determine the correct Maruo formula coefficients for FIXED BODY, beam seas.

We know:
  F_y = C_quad × ∫|a|² sinθ dθ + C_cross × Re[exp(iψ) a(β)] × sinβ

where a(θ) = (iω/g) φ_d √r exp(-ikr) is the far-field scattered amplitude.

We have the near-field Fy as ground truth. Find C_quad, C_cross, and ψ.

Since we have 15 wavelengths and 3 unknowns, this is overdetermined.
We'll first try fixing ψ to various values (0, π/4, π/2, -π/4) and fitting C_quad, C_cross.
"""

import numpy as np
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_potential, airy_waves_velocity
import logging
from scipy.optimize import least_squares

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

# Waterline
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

edges, edge_centers, edge_lengths_arr, edge_normals = extract_waterline_edges(hull_mesh)
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

beta = np.pi / 2
idx_beta = N_THETA // 4  # θ=π/2

wavelengths = [3, 4, 5, 6, 8, 10, 13, 17, 22, 28, 35, 45, 55, 70, 90]

# Collect data
data = []
for lam in wavelengths:
    k = 2*np.pi / lam
    omega = np.sqrt(k*g)
    
    diff_prob = cpt.DiffractionProblem(body=body, wave_direction=beta,
                                        omega=omega, water_depth=np.inf)
    diff_result = solver.solve(diff_prob, keep_details=True)
    
    # Near-field
    total_vel = airy_waves_velocity(hull_centers, diff_prob) + solver.compute_velocity(hull_centers, diff_result)
    total_pot_wl = airy_waves_potential(wl_eval_pts, diff_prob) + solver.compute_potential(wl_eval_pts, diff_result)
    
    eta_wl = (1j * omega / g) * total_pot_wl
    F_wl_y = 0.25 * rho * g * np.sum(np.abs(eta_wl)**2 * edge_normals[:, 1] * edge_lengths_arr)
    vel_sq = np.sum(np.abs(total_vel)**2, axis=1)
    F_vel_y = -0.25 * rho * np.sum(vel_sq * hull_normals[:, 1] * hull_areas)
    F_nf = F_wl_y + F_vel_y
    
    # Far-field
    phi_scat = solver.compute_potential(field_pts, diff_result)
    a_scat = (1j * omega / g) * phi_scat * np.sqrt(R_FIELD) * np.exp(-1j * k * R_FIELD)
    
    I_quad = np.sum(np.abs(a_scat)**2 * np.sin(theta_fp)) * dtheta
    a_at_beta = a_scat[idx_beta]
    
    data.append({
        'lam': lam, 'k': k, 'omega': omega,
        'F_nf': F_nf, 'F_wl': F_wl_y, 'F_vel': F_vel_y,
        'I_quad': I_quad,
        'a_beta_re': a_at_beta.real,
        'a_beta_im': a_at_beta.imag,
        'a_beta': a_at_beta,
    })
    
    print(f"lambda={lam:3d}: F_nf={F_nf:12.1f}  I_quad={I_quad:12.4e}  "
          f"a(β)=({a_at_beta.real:10.4f}, {a_at_beta.imag:10.4f})")

# Now fit models
F_nf_arr = np.array([d['F_nf'] for d in data])
I_quad_arr = np.array([d['I_quad'] for d in data])
k_arr = np.array([d['k'] for d in data])
a_beta_arr = np.array([d['a_beta'] for d in data])

print("\n" + "="*100)
print("FITTING: F_y = C_q(k) × I_quad + C_c(k) × cross_term")
print("="*100)

# Model A: F_y = (ρg/k) × I_quad  (pure Maruo, no cross-term)
# Already checked — doesn't work.

# Model B: F_y = C₁ × I_quad + C₂ × Re[a(β)] + C₃ × Im[a(β)]
# where C₁, C₂, C₃ may depend on k
print("\n--- Model B: constant coefficients ---")
print("F_y = C₁ I_quad + C₂ Re[a(β)] + C₃ Im[a(β)]")

A = np.column_stack([I_quad_arr, 
                     np.array([d['a_beta_re'] for d in data]),
                     np.array([d['a_beta_im'] for d in data])])
# Weighted least squares
W = np.diag(1.0 / np.abs(F_nf_arr).clip(min=100))
C_fit = np.linalg.lstsq(W @ A, W @ F_nf_arr, rcond=None)[0]
F_pred = A @ C_fit
print(f"C₁={C_fit[0]:.2f}, C₂={C_fit[1]:.2f}, C₃={C_fit[2]:.2f}")

# Model C: F_y = (ρg/k) × [I_quad + c₁ √(2π/k) Re[exp(iψ) a(β)]]
# where c₁ and ψ are universal constants
print("\n--- Model C: Maruo + stationary-phase cross-term ---")
print("F_y = -(ρg/(4k)) × [I_quad + 2√(2π/k) × {c_r Re[a(β)] + c_i Im[a(β)]}]")

# Rearrange: F_y × (-4k/(ρg)) = I_quad + 2√(2π/k) × {c_r Re[a(β)] + c_i Im[a(β)]}
# So: I_quad - F_y × 4k/(ρg) = -2√(2π/k) × {c_r Re[a(β)] + c_i Im[a(β)]}
# Let LHS = I_quad + F_nf × 4k/(ρg)
# RHS = 2√(2π/k) × [c_r Re[a(β)] + c_i Im[a(β)]]

LHS = I_quad_arr + F_nf_arr * 4 * k_arr / (rho * g)
sqrt_factor = 2 * np.sqrt(2*np.pi/k_arr)
A_cross = np.column_stack([sqrt_factor * np.array([d['a_beta_re'] for d in data]),
                           sqrt_factor * np.array([d['a_beta_im'] for d in data])])
c_fit = np.linalg.lstsq(A_cross, LHS, rcond=None)[0]
print(f"c_r = {c_fit[0]:.6f}, c_i = {c_fit[1]:.6f}")
print(f"This corresponds to exp(iψ) where ψ = atan2(c_i, c_r) = {np.degrees(np.arctan2(c_fit[1], c_fit[0])):.1f}°")
print(f"|c| = {np.sqrt(c_fit[0]**2 + c_fit[1]**2):.6f}")

F_pred_C = -(rho*g/(4*k_arr)) * (I_quad_arr + A_cross @ c_fit)
print(f"\n{'lam':>5} {'F_nf':>12} {'F_pred_C':>12} {'ratio':>10}")
print("-"*45)
for i, d in enumerate(data):
    ratio = d['F_nf'] / F_pred_C[i] if abs(F_pred_C[i]) > 0.1 else float('nan')
    print(f"{d['lam']:5d} {d['F_nf']:12.1f} {F_pred_C[i]:12.1f} {ratio:10.4f}")

# Model D: Maybe the correct formula is simply:
# F_y = -(ρg/(4k)) × [I_quad + 2√(2π/k) sin(β) Re[exp(iπ/4) a(β)]]
# Let's test this (note: sin(β) = 1 for β=π/2)
print("\n--- Model D: Standard textbook Maruo + cross-term ---")
# ψ=π/4 (from stationary phase of exp(ikR(1-sinθ)) → exp(iπ/4) factor)
cross_D = 2 * np.sqrt(2*np.pi/k_arr) * np.real(np.exp(1j*np.pi/4) * a_beta_arr) * np.sin(beta)
F_pred_D = -(rho*g/(4*k_arr)) * (I_quad_arr + cross_D)
print(f"\n{'lam':>5} {'F_nf':>12} {'F_pred_D':>12} {'ratio':>10}")
print("-"*45)
for i, d in enumerate(data):
    ratio = d['F_nf'] / F_pred_D[i] if abs(F_pred_D[i]) > 0.1 else float('nan')
    print(f"{d['lam']:5d} {d['F_nf']:12.1f} {F_pred_D[i]:12.1f} {ratio:10.4f}")

# Model E: Use different overall coefficient
# Maybe F_y = (ρg/(2k)) × [I_quad + √(2π/k) Re[exp(iπ/4) a(β)] × 2sinβ]
# (coefficient ρg/(2k) instead of -ρg/(4k))
print("\n--- Model E: Factor of 2 test ---")
F_pred_E = (rho*g/(2*k_arr)) * (I_quad_arr + 2*np.sqrt(2*np.pi/k_arr) * np.real(np.exp(1j*np.pi/4) * a_beta_arr) * np.sin(beta))
print(f"\n{'lam':>5} {'F_nf':>12} {'F_pred_E':>12} {'ratio':>10}")
print("-"*45)
for i, d in enumerate(data):
    ratio = d['F_nf'] / F_pred_E[i] if abs(F_pred_E[i]) > 0.1 else float('nan')
    print(f"{d['lam']:5d} {d['F_nf']:12.1f} {F_pred_E[i]:12.1f} {ratio:10.4f}")

# Model F: Most general 2-parameter fit with k-dependent coefficients
# F_y = alpha(k) × I_quad + gamma(k) × Re[exp(iπ/4) a(β)]
# Check if alpha(k) = ρg × f(k) and gamma(k) = ρg × h(k)
print("\n--- Model F: Free fit alpha(k) and gamma(k) ---")
print("F_y = α × I_quad + γ × Re[exp(iπ/4) a(β)]")
print(f"{'lam':>5} {'k':>8} {'α':>14} {'α/(ρg/k)':>10} {'γ':>14} {'γ/√(ρg×2π/k³)':>16}")
print("-"*75)

for i, d in enumerate(data):
    # With only 1 equation and 2 unknowns per wavelength, can't solve uniquely.
    # But let's try assuming α = -(ρg/(4k)) and solving for γ:
    alpha_fixed = -(rho*g/(4*d['k']))
    cross_re = np.real(np.exp(1j*np.pi/4) * d['a_beta'])
    if abs(cross_re) > 1e-15:
        gamma = (d['F_nf'] - alpha_fixed * d['I_quad']) / cross_re
        gamma_norm = gamma / np.sqrt(rho*g*2*np.pi/d['k']**3) if d['k'] > 0 else float('nan')
    else:
        gamma = float('nan')
        gamma_norm = float('nan')
    
    print(f"{d['lam']:5d} {d['k']:8.4f} {alpha_fixed:14.1f} {'(fixed)':>10} {gamma:14.1f} {gamma_norm:16.4f}")

print("\nIf γ_norm is constant across wavelengths, the formula is:")
print("F_y = -(ρg/(4k)) × I_quad + γ_norm × √(ρg×2π/k³) × Re[exp(iπ/4) a(β)]")
print()

# Final model: completely free 2-param fit (per wavelength)
# F_y = α × I_quad + γ_re × Re[a(β)] + γ_im × Im[a(β)]
# With 3 unknowns and many wavelengths, overdetermined.
print("\n--- Model G: Fully free global fit ---")
print("F_y = α × I_quad + γ_re × Re[a(β)] + γ_im × Im[a(β)]")
A_full = np.column_stack([I_quad_arr, 
                          np.array([d['a_beta_re'] for d in data]),
                          np.array([d['a_beta_im'] for d in data])])
# Weight by inverse of |F_nf|
weights = 1.0 / np.abs(F_nf_arr).clip(min=100)
Aw = A_full * weights[:, None]
bw = F_nf_arr * weights
coeffs = np.linalg.lstsq(Aw, bw, rcond=None)[0]
F_pred_G = A_full @ coeffs
print(f"α = {coeffs[0]:.2f}, γ_re = {coeffs[1]:.2f}, γ_im = {coeffs[2]:.2f}")
print(f"{'lam':>5} {'F_nf':>12} {'F_pred':>12} {'ratio':>10}")
print("-"*45)
for i, d in enumerate(data):
    ratio = d['F_nf'] / F_pred_G[i] if abs(F_pred_G[i]) > 0.1 else float('nan')
    print(f"{d['lam']:5d} {d['F_nf']:12.1f} {F_pred_G[i]:12.1f} {ratio:10.4f}")

residual = np.sqrt(np.sum((F_pred_G - F_nf_arr)**2 * weights**2) / len(data))
print(f"\nWeighted RMS residual: {residual:.4f}")

print("\nDone.")
