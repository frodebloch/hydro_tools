#!/usr/bin/env python3
"""
Compute the mean drift force by DIRECTLY evaluating the momentum flux
through a control cylinder at large radius R.

This avoids all issues with the asymptotic Maruo formula coefficients.

The mean y-force on the body = -(mean y-momentum flux outward through cylinder)

For a control cylinder at r=R, the mean y-momentum flux is:
  F_y = -∫₀²π dθ ∫_{-∞}^{0} dz R [ <p> sinθ + ρ<v_y v_r> ]

where:
  <p> = -ρ<∂φ/∂t> - ρg<z> - (ρ/2)<|∇φ|²>
  
For the second-order mean:
  <p>^(2) = -(ρ/2)<|∇φ|²>  (the first-order terms average to zero)

So:
  F_y = -∫₀²π dθ ∫_{-∞}^{0} dz R [ -(ρ/2)<|∇φ|²> sinθ + ρ<v_y v_r> ]

Actually, the complete expression for mean drift from the control surface is:

  F_y = -∮ [ <p + ρgz> n_y + ρ<(v·n) v_y> ] dS - (ρg/2) ∮_{WL} <η²> n_y dl

where the first integral is over the underwater control surface S at mean position,
and the second is the waterline (free-surface) correction.

For a vertical cylinder at r=R, n = r̂, n_y = sinθ, v·n = v_r = ∂φ/∂r.

The second-order mean pressure at a point on S is:
  <p + ρgz>^(2) = -(ρ/2)<|∇φ|²> + ... 

Hmm, let me use the proper Bernoulli equation.

For irrotational flow, the Bernoulli equation (unsteady) is:
  p/ρ + ∂φ/∂t + (1/2)|∇φ|² + gz = C(t)

For the mean (second-order):
  <p>^(2)/ρ = -(1/2)<|∇φ|²>  

(since <∂φ/∂t> = 0 for first-order φ, and gz averages out)

Wait: the first-order potential φ^(1) has <∂φ^(1)/∂t> = 0 (harmonic in time).
And <|∇φ^(1)|²> is the time-averaged squared velocity.

So: <p>^(2) = -(ρ/2)<|∇φ^(1)|²>

The momentum flux integral:
  F_y = -∫₀²π dθ R ∫_{-∞}^{0} dz [ <p>^(2) sinθ + ρ<v_y^(1) v_r^(1)> ]
        -(ρg/2) ∫₀²π <η²> sinθ R dθ

  = -∫₀²π dθ R ∫_{-∞}^{0} dz [ -(ρ/2)<|∇φ|²> sinθ + ρ<v_y v_r> ]
    -(ρg/2) ∫₀²π <η²> sinθ R dθ

  = ∫₀²π dθ R ∫_{-∞}^{0} dz [ (ρ/2)<|∇φ|²> sinθ - ρ<v_y v_r> ]
    -(ρg/2) ∫₀²π <η²> sinθ R dθ

For harmonid φ with complex amplitude φ̂:
  <|∇φ|²> = (1/2)|∇φ̂|²
  <v_y v_r> = (1/2) Re(v̂_y v̂_r*)

So:
  F_y = ∫₀²π dθ R ∫_{-∞}^{0} dz [ (ρ/4)|∇φ̂|² sinθ - (ρ/2)Re(v̂_y v̂_r*) ]
        -(ρg/4) ∫₀²π |η̂|² sinθ R dθ

where η̂ = (iω/g)φ̂|_{z=0} is the complex elevation amplitude (for exp(-iωt) convention),
or η̂ = -(iω/g)φ̂ for exp(+iωt) convention, but |η̂|² doesn't depend on convention.

This is the exact expression. The depth integral for deep water can be done analytically 
for plane waves, but for the total field (plane + cylindrical) we need to be careful.

SIMPLIFIED APPROACH: Since the depth dependence of both the incident and scattered 
potentials is exp(kz) in deep water, all velocity products have depth dependence 
exp(2kz), and the depth integral gives a factor 1/(2k).

So for deep-water:
  F_y = (R/(2k)) ∫₀²π [ (ρ/4)|∇φ̂|²_{z=0} sinθ - (ρ/2)Re(v̂_y v̂_r*)_{z=0} ] dθ
        -(ρg/4) R ∫₀²π |η̂|²_{z=0} sinθ dθ

where all quantities are evaluated at z=0 and r=R.

Let me compute this numerically at the field points.
"""

import numpy as np
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_potential, airy_waves_velocity, froude_krylov_force
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

beta = np.pi / 2  # beam seas
wavelengths = [3, 5, 10, 22, 55, 90]

# Near-field: extract waterline edges
def extract_waterline_edges(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    z_tol = 0.05
    wl_mask = np.abs(vertices[:, 2]) < z_tol
    wl_verts = set(np.where(wl_mask)[0])
    
    edges_list = set()
    face_for_edge = {}
    for fi, face in enumerate(faces):
        if face[0] == face[3]:
            verts = [face[0], face[1], face[2]]
        else:
            verts = [face[0], face[1], face[2], face[3]]
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
        if np.dot(n1[:2], to_face) < 0:
            edge_normals[i] = n1
        else:
            edge_normals[i] = n2
    return edges_list, edge_centers, edge_lengths, edge_normals

edges, edge_centers, edge_lengths, edge_normals = extract_waterline_edges(hull_mesh)
hull_normals = hull_mesh.faces_normals
hull_areas = hull_mesh.faces_areas
hull_centers = hull_mesh.faces_centers
n_hull = hull_mesh.nb_faces
wl_eval_pts = edge_centers.copy()
wl_eval_pts[:, 2] = -0.001

print(f"FIXED BODY: Direct momentum flux vs Near-field drift Fy (beam seas)")
print(f"r_field = {R_FIELD}m")
print(f"\n{'lam':>5} {'NF_tot':>10} {'MF_sub':>12} {'MF_wl':>12} {'MF_tot':>12} {'NF/MF':>8}")
print("-" * 65)

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
    eta_sq = np.abs(eta_wl)**2
    F_wl_y = 0.25 * rho * g * np.sum(eta_sq * edge_normals[:, 1] * edge_lengths)
    
    vel_sq = np.sum(np.abs(total_vel_hull)**2, axis=1)
    F_vel_y = -0.25 * rho * np.sum(vel_sq * hull_normals[:, 1] * hull_areas)
    
    F_nf_y = F_wl_y + F_vel_y
    
    # === Momentum flux through control cylinder ===
    # Total potential and velocity at field points
    # Note: compute_potential returns ONLY scattered potential
    phi_scat = solver.compute_potential(field_pts, diff_result)
    phi_inc = airy_waves_potential(field_pts, diff_prob)
    phi_total = phi_inc + phi_scat
    
    # Velocities (need ∂φ/∂r, ∂φ/∂y, and |∇φ|²)
    # For total velocity, we need both inc and scattered
    vel_inc = airy_waves_velocity(field_pts, diff_prob)   # shape (N, 3) complex
    vel_scat = solver.compute_velocity(field_pts, diff_result)  # shape (N, 3) complex
    vel_total = vel_inc + vel_scat  # (vx, vy, vz) in Cartesian
    
    # Convert to v_r and v_y at each field point
    # v_r = vx cosθ + vy sinθ
    cos_th = np.cos(theta_fp)
    sin_th = np.sin(theta_fp)
    
    vr_hat = vel_total[:, 0] * cos_th + vel_total[:, 1] * sin_th
    vy_hat = vel_total[:, 1]  # v_y in Cartesian
    
    # |∇φ|² at z=0 
    grad_sq = np.sum(np.abs(vel_total)**2, axis=1)
    
    # η at z=0
    eta_hat = (1j * omega / g) * phi_total  # complex elevation amplitude
    eta_sq_fp = np.abs(eta_hat)**2
    
    # Time averages (for exp(-iωt) convention, or any convention):
    # <|∇φ|²> = (1/2) |∇φ̂|²
    # <v_y v_r> = (1/2) Re(v̂_y v̂_r*)
    # <|η|²> = (1/2) |η̂|²
    
    # F_y = (R/(2k)) ∫ [ (ρ/4)|∇φ̂|² sinθ - (ρ/2)Re(v̂_y v̂_r*) ] dθ
    #        -(ρg/4) R ∫ |η̂|² sinθ dθ
    
    # Subsurface term
    integrand_sub = (rho/4) * grad_sq * sin_th - (rho/2) * np.real(vy_hat * np.conj(vr_hat))
    F_sub_y = R_FIELD / (2*k) * np.sum(integrand_sub) * dtheta
    
    # Waterline term
    F_wl_ctrl_y = -(rho * g / 4) * R_FIELD * np.sum(eta_sq_fp * sin_th) * dtheta
    
    F_mf_y = F_sub_y + F_wl_ctrl_y
    
    ratio = F_nf_y / F_mf_y if abs(F_mf_y) > 0.1 else float('inf')
    
    print(f"{lam:5.0f} {F_nf_y:10.1f} {F_sub_y:12.1f} {F_wl_ctrl_y:12.1f} {F_mf_y:12.1f} {ratio:8.4f}")
