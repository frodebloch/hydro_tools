#!/usr/bin/env python3
"""
Derive the correct Maruo formula by computing the cross-term analytically.

The mean drift force from momentum flux through a control cylinder at r=R is:

  F̄_y = -(ρg/(4k)) × ∫₀²π [|η_total|² - |η_inc|²] × [-sin(θ)] R dθ

Wait, let me be more careful. The momentum flux through a vertical cylinder
in deep water gives (Faltinsen 1990, eq 6.38):

  F̄_y = (ρg/(4k)) × ∫₀²π |η̂_total|² sin(θ) R dθ  
         - (ρg/(4k)) × ∫₀²π |η̂_inc|² sin(θ) R dθ
         + (free-surface waterline correction)

Actually, the correct formula from energy/momentum conservation in deep water is
(see Faltinsen 1990, section 6.3):

  F̄_i = -(ρg/(4k)) × ∮_C (|η̂|² - |η̂_inc|²) nᵢ dl

where C is a closed contour at the free surface at r→∞.

For a circle at r=R: dl = R dθ, n = (cosθ, sinθ), so:

  F̄_y = -(ρg/(4k)) × R × ∫₀²π (|η̂_total|² - |η̂_inc|²) sinθ dθ

Now: η_total = η_inc + η_d, where η_d = a(θ) exp(ikr)/√r for the disturbance.

|η_total|² - |η_inc|² = |η_d|² + 2Re(η_d × η_inc*)

At r=R:
  η_d = a(θ) × exp(ikR)/√R
  |η_d|² = |a(θ)|²/R
  
  η_inc = exp(ikR sinθ)  (for beam seas, β=π/2)

So:
  [|η_total|² - |η_inc|²] = |a(θ)|²/R + 2Re[a(θ) exp(ikR)/√R × exp(-ikR sinθ)]

Integral I₁ (quadratic disturbance):
  R × ∫ |a(θ)|²/R × sinθ dθ = ∫ |a(θ)|² sinθ dθ

Integral I₂ (cross-term):
  R × ∫ 2Re[a(θ) exp(ikR)/√R × exp(-ikR sinθ)] sinθ dθ
  = 2√R × Re[∫ a(θ) exp(ikR(1-sinθ)) sinθ dθ]

For large kR, the integrand oscillates rapidly except where 1-sinθ ≈ 0,
i.e., near θ=π/2. Using stationary phase at θ=π/2:

  sinθ ≈ 1 - (θ-π/2)²/2  near θ=π/2
  1-sinθ ≈ (θ-π/2)²/2

  exp(ikR(θ-π/2)²/2) is a Gaussian with width ~ 1/√(kR)

  ∫ a(θ) sinθ exp(ikR(1-sinθ)) dθ ≈ a(π/2) × 1 × ∫ exp(ikRu²/2) du
  = a(π/2) × √(2π/(kR)) × exp(iπ/4)

So:
  I₂ = 2√R × Re[a(π/2) × √(2π/(kR)) × exp(iπ/4)]
     = 2 × Re[a(π/2) × √(2π/k) × exp(iπ/4)]

Therefore:
  F̄_y = -(ρg/(4k)) × {∫|a|² sinθ dθ + 2√(2π/k) Re[a(β) exp(iπ/4)] sinβ}

Wait, I need to be more careful with the sinθ in the cross-term integral.
Let me redo the stationary phase.

At θ=π/2: sinθ=1, so the sinθ factor is just 1.
The integrand is: a(θ) × sinθ × exp(ikR(1-sinθ))

Let u = θ - π/2. Then sinθ = cos(u) ≈ 1 - u²/2, 1-sinθ ≈ u²/2.
∫ a(π/2+u) cos(u) exp(ikRu²/2) du ≈ a(π/2) × ∫ exp(ikRu²/2) du
= a(π/2) × √(2π/(kR)) × exp(iπ/4)

So I₂ = 2√R × Re[a(π/2) √(2π/(kR)) exp(iπ/4)]
       = 2 × Re[a(π/2) √(2π/k) exp(iπ/4)]

And: F̄_y = -(ρg/(4k)) × {∫|a|² sinθ dθ + 2√(2π/k) Re[exp(iπ/4) a(β)]}

For general β (not just π/2), there are TWO stationary phase points: θ=β and θ=π-β.
For β=π/2, these coincide. For β≠π/2, the second gives sin(π-β)=sinβ contribution.

Let me verify this numerically and also check: there may be ANOTHER stationary 
phase point at θ=3π/2 (where sinθ=-1 and the incident wave goes in the -y direction).
Actually for 1-sinθ, at θ=3π/2: 1-(-1)=2, which is a maximum, not stationary.
At θ=-π/2=3π/2: d(sinθ)/dθ = cosθ = 0, so it IS a stationary phase point!
But 1-sinθ = 2 at this point, so exp(2ikR) oscillates. Wait — stationary phase
requires d/dθ[ikR(1-sinθ)] = -ikR cosθ = 0, which happens at θ=π/2 AND θ=3π/2.

At θ=3π/2: sinθ=-1, 1-sinθ=2.
Second derivative: d²(sinθ)/dθ² = -sinθ = +1 (positive).
exp(ikR(1-sinθ)) = exp(2ikR), which has a definite phase.
The integral contribution from θ=3π/2:
  a(3π/2) × sin(3π/2) × exp(2ikR) × √(2π/(kR)) × exp(-iπ/4)
  = -a(3π/2) × exp(2ikR) × √(2π/(kR)) × exp(-iπ/4)

Note the -iπ/4 (vs +iπ/4 at θ=π/2) because the curvature of sinθ is
opposite (concave up at 3π/2 vs concave down at π/2).

So the full cross-term is:
I₂ = 2√R × Re{[a(π/2) √(2π/(kR)) exp(iπ/4)] + [-a(3π/2) √(2π/(kR)) exp(2ikR-iπ/4)]}
   = 2 Re{√(2π/k) [a(π/2) exp(iπ/4) - a(3π/2) exp(2ikR-iπ/4)]}

The second term has exp(2ikR) which for large kR oscillates rapidly between
different numerical evaluations (R is fixed at 5000m, but k varies). This means
the cross-term from θ=3π/2 is NOT ZERO but oscillates with k.

For the cross-term to be useful analytically, we need to handle this carefully.
In the strict asymptotic limit R→∞, both terms are finite (they don't depend on R
after the stationary phase). So the Maruo formula is:

F̄_y = -(ρg/(4k)) × {∫|a|² sinθ dθ + 2√(2π/k) Re[exp(iπ/4) a(β)]}
       + oscillating term from θ=3π/2 that should cancel with something...

Hmm, actually for a FIXED body, the oscillating term should not be there 
physically. Let me reconsider.

Actually wait - I think the issue is simpler. Let me just compute numerically.
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

beta = np.pi / 2  # beam seas

# Use MULTIPLE field-point radii to understand R-dependence
R_FIELDS = [500.0, 1000.0, 2000.0, 5000.0, 10000.0]
N_THETA = 720

wavelengths = [5, 10, 22, 55]

print("CROSS-TERM ANALYSIS: Fixed body, beam seas")
print(f"Body: semicircular barge R={R}m, L={L}m")
print()

for lam in wavelengths:
    k = 2*np.pi / lam
    omega = np.sqrt(k*g)
    
    diff_prob = cpt.DiffractionProblem(body=body, wave_direction=beta,
                                        omega=omega, water_depth=np.inf)
    diff_result = solver.solve(diff_prob, keep_details=True)
    
    # Near-field drift (with -0.25 velocity term — the original)
    inc_pot_hull = airy_waves_potential(hull_centers, diff_prob)
    diff_pot_hull = solver.compute_potential(hull_centers, diff_result)
    total_vel_hull = airy_waves_velocity(hull_centers, diff_prob) + solver.compute_velocity(hull_centers, diff_result)
    total_pot_wl = airy_waves_potential(wl_eval_pts, diff_prob) + solver.compute_potential(wl_eval_pts, diff_result)
    
    eta_wl = (1j * omega / g) * total_pot_wl
    F_wl_y = 0.25 * rho * g * np.sum(np.abs(eta_wl)**2 * edge_normals[:, 1] * edge_lengths_arr)
    vel_sq = np.sum(np.abs(total_vel_hull)**2, axis=1)
    F_vel_y = -0.25 * rho * np.sum(vel_sq * hull_normals[:, 1] * hull_areas)
    F_nf = F_wl_y + F_vel_y
    
    print(f"=== lambda={lam}m, k={k:.4f}, kR_field values: " + 
          ", ".join(f"{k*r:.0f}" for r in R_FIELDS))
    print(f"    NF drift Fy = {F_nf:.1f}  (WL={F_wl_y:.1f}, vel={F_vel_y:.1f})")
    
    for R_FIELD in R_FIELDS:
        theta_fp = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
        dtheta = 2*np.pi / N_THETA
        field_pts = np.column_stack([
            R_FIELD * np.cos(theta_fp),
            R_FIELD * np.sin(theta_fp),
            np.zeros(N_THETA)
        ])
        
        # Scattered potential at field points
        phi_scat = solver.compute_potential(field_pts, diff_result)
        
        # Far-field amplitude: a(θ) = (iω/g) × φ_s × √r × exp(-ikr)
        a_scat = (1j * omega / g) * phi_scat * np.sqrt(R_FIELD) * np.exp(-1j * k * R_FIELD)
        
        # Quadratic term
        I_quad = np.sum(np.abs(a_scat)**2 * np.sin(theta_fp)) * dtheta
        
        # Cross term computed numerically:
        # η_inc at field points: η_inc = exp(ik(x cosβ + y sinβ)) = exp(ikR sinθ) for β=π/2
        eta_inc_fp = np.exp(1j * k * R_FIELD * np.sin(theta_fp))
        eta_scat_fp = (1j * omega / g) * phi_scat  # scattered elevation at field points
        
        # Cross: 2Re(η_s × η_inc*) × sinθ
        cross = 2 * np.real(eta_scat_fp * np.conj(eta_inc_fp))
        I_cross = np.sum(cross * np.sin(theta_fp)) * dtheta
        
        # Stationary phase prediction for cross term:
        # At θ=π/2: a(π/2) is the scattered amplitude in the wave direction
        a_beta = a_scat[N_THETA // 4]  # θ=π/2
        a_minus_beta = a_scat[3 * N_THETA // 4]  # θ=3π/2
        
        SP_cross = 2 * np.sqrt(2*np.pi/k) * np.real(
            np.exp(1j*np.pi/4) * a_beta - 
            np.exp(2j*k*R_FIELD - 1j*np.pi/4) * a_minus_beta
        ) / R_FIELD
        
        # The formula predicts:
        # F_y = -(ρg/(4k)) × R × [I_quad_at_R + I_cross_at_R]
        # where I_quad_at_R = ∫|η_s|² sinθ dθ, I_cross_at_R = ∫ 2Re(η_s η_inc*) sinθ dθ
        # Note: |η_s|² at field point = |a|²/R, so R × I_quad_at_R = I_quad (R-independent)
        
        # But I_cross_at_R involves |η_s|~1/√R and |η_inc|~1, so the product ~1/√R,
        # and R × (1/√R) = √R. So the cross contribution grows as √R? That can't be right.
        
        # Wait: R × I_cross_at_R = R × ∫ 2Re(a(θ)exp(ikR)/√R × exp(-ikR sinθ)) sinθ dθ
        # = √R × 2Re[∫ a(θ) sinθ exp(ikR(1-sinθ)) dθ]
        # Stationary phase: ~ √R × |a(β)| × √(2π/(kR)) = |a(β)| × √(2π/k) = O(1)
        
        # So the cross term is O(1), same as the quadratic term. Good.
        
        # Let me compute both terms properly using the actual field-point data.
        # η_s at field pt = (iω/g) × φ_s = a(θ) × exp(ikR) / √R
        # |η_s|² = |a|²/R
        # R × ∫|η_s|² sinθ dθ = ∫|a|² sinθ dθ   (matches I_quad)
        
        # Cross: R × ∫ 2Re(η_s η_inc*) sinθ dθ = R × I_cross 
        R_I_cross = R_FIELD * I_cross
        
        FF_formula = -(rho * g / (4*k)) * (I_quad + R_I_cross)
        
        print(f"    R={R_FIELD:7.0f}: I_quad={I_quad:12.4e}, R*I_cross={R_I_cross:12.4e}, "
              f"FF_y={FF_formula:12.1f}, NF/FF={F_nf/FF_formula if abs(FF_formula)>0.1 else float('nan'):8.4f}")
    
    print()

print("Done.")
