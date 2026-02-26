#!/usr/bin/env python3
"""
Test the Maruo formula on a FIXED body (diffraction only, no motion).
For a fixed body, there's no radiation — just the scattered wave.

The drift force on a fixed body is well-known:
  F̄_y = (ρg/k) ∫|a_s|² sinθ dθ

Wait, is this right? Let me check against the near-field formula.
For a fixed body, the near-field drift force has only the waterline
and velocity terms (no rotation term), and both are computable
from the diffraction potential alone.

This test should help distinguish whether the Maruo formula needs
extra terms for the freely-floating case vs fixed case.
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

# Waterline extraction (simplified from nearfield script)
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
            verts = [face[0], face[1], face[2]]
        else:
            verts = [face[0], face[1], face[2], face[3]]
        n_verts = len(verts)
        for kk in range(n_verts):
            v1 = verts[kk]; v2 = verts[(kk+1)%n_verts]
            if v1 in wl_verts and v2 in wl_verts:
                edge = (min(v1,v2), max(v1,v2))
                edges.add(edge)
                face_for_edge[edge] = fi
    edges = list(edges)
    n_edges = len(edges)
    if n_edges == 0:
        return [], np.zeros((0,3)), np.zeros(0), np.zeros((0,3))
    
    edge_centers = np.zeros((n_edges, 3))
    edge_lengths = np.zeros(n_edges)
    edge_normals = np.zeros((n_edges, 3))
    face_normals = mesh.faces_normals
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
        if np.dot(n1[:2], to_face) < 0:
            edge_normals[i] = n1
        else:
            edge_normals[i] = n2
    return edges, edge_centers, edge_lengths, edge_normals

edges, edge_centers, edge_lengths, edge_normals = extract_waterline_edges(hull_mesh)
hull_normals = hull_mesh.faces_normals
hull_areas = hull_mesh.faces_areas
hull_centers = hull_mesh.faces_centers
n_hull = hull_mesh.nb_faces

# Waterline eval points (slightly below surface)
wl_eval_pts = edge_centers.copy()
wl_eval_pts[:, 2] = -0.001

beta = np.pi / 2  # beam seas
wavelengths = [3, 5, 10, 22, 55, 90]

print(f"FIXED BODY: Maruo vs Near-field drift Fy (beam seas)")
print(f"r_field = {R_FIELD}m")
print(f"\n{'lam':>5} {'NF_wl':>10} {'NF_vel':>10} {'NF_tot':>10} "
      f"{'FF_quad':>10} {'FF_cross':>10} "
      f"{'FF_total':>10} {'NF/FF':>8}")
print("-" * 85)

for lam in wavelengths:
    k = 2*np.pi / lam
    omega = np.sqrt(k*g)
    
    # Diffraction solve
    diff_prob = cpt.DiffractionProblem(body=body, wave_direction=beta,
                                        omega=omega, water_depth=np.inf)
    diff_result = solver.solve(diff_prob, keep_details=True)
    
    # === Near-field: fixed body ===
    # Total potential = incident + diffraction
    inc_pot_hull = airy_waves_potential(hull_centers, diff_prob)
    diff_pot_hull = solver.compute_potential(hull_centers, diff_result)
    total_pot_hull = inc_pot_hull + diff_pot_hull
    
    inc_vel_hull = airy_waves_velocity(hull_centers, diff_prob)
    diff_vel_hull = solver.compute_velocity(hull_centers, diff_result)
    total_vel_hull = inc_vel_hull + diff_vel_hull
    
    inc_pot_wl = airy_waves_potential(wl_eval_pts, diff_prob)
    diff_pot_wl = solver.compute_potential(wl_eval_pts, diff_result)
    total_pot_wl = inc_pot_wl + diff_pot_wl
    
    # Waterline term: (1/4) ρg |η|² n dl   (η = iω/g × φ at z=0)
    eta_wl = (1j * omega / g) * total_pot_wl
    eta_sq = np.abs(eta_wl)**2
    F_wl_y = 0.25 * rho * g * np.sum(eta_sq * edge_normals[:, 1] * edge_lengths)
    
    # Velocity term: -(1/4) ρ |∇φ|² n dS
    vel_sq = np.sum(np.abs(total_vel_hull)**2, axis=1)
    F_vel_y = -0.25 * rho * np.sum(vel_sq * hull_normals[:, 1] * hull_areas)
    
    F_nf_y = F_wl_y + F_vel_y
    
    # === Far-field: Maruo ===
    # Scattered potential at field points
    phi_scat_fp = solver.compute_potential(field_pts, diff_result)
    
    # a(θ) = (iω/g) × φ × √r × exp(-ikr)
    a_scat = (1j * omega / g) * phi_scat_fp * np.sqrt(R_FIELD) * np.exp(-1j * k * R_FIELD)
    
    # === Method 1: Pure disturbance term only (KNOWN WRONG) ===
    a2 = np.abs(a_scat)**2
    F_ff_y_distonly = rho * g / k * np.sum(a2 * np.sin(theta_fp)) * dtheta
    
    # === Method 2: Direct momentum flux through control cylinder ===
    # Compute the total field at field points and evaluate the momentum flux
    # integral directly, without trying to use the asymptotic Maruo formula.
    #
    # In deep water, the mean drift force equals the net momentum flux through
    # a control surface surrounding the body. For a vertical cylinder at r=R:
    #
    # F_y = ∫₀²π S_ry(R,θ) R dθ
    #
    # where S_ry is the mean y-momentum flux in the r-direction, integrated over depth.
    #
    # The radiation stress tensor for deep water waves gives:
    # ∫_{-∞}^{0} <S_ry> dz = -(ρg/(4k)) |η|² sin(2θ)/2 ... no
    #
    # Let me compute this from first principles using the velocity potential.
    # The mean y-force from momentum flux through a cylinder at r=R is:
    #
    # F_y = ∫₀²π dθ ∫_{-∞}^{0} [ -<p> sinθ - ρ<v_y v_r> ] R dz  (approximately)
    #
    # This is getting complicated. Let me try a different approach entirely.
    # 
    # APPROACH: Use the Haskind-Newman relation for drift force.
    #
    # For a FIXED body, the drift force can be related to the far-field
    # diffraction amplitude via the optical theorem approach.
    #
    # Actually, let me try the SIMPLEST thing: compute at multiple radii
    # and see if we can identify the correct formula empirically.
    
    # === Method 2a: Empirical coefficient search ===
    # We know the quadratic integral and the cross term value.
    # Try different coefficient combinations to match NF.
    
    int_a2_sin = np.sum(a2 * np.sin(theta_fp)) * dtheta
    a_at_beta = a_scat[idx_beta]
    cross_val = np.real(np.exp(-1j * np.pi/4) * a_at_beta) * np.sin(beta)
    
    # Try to find coefficients alpha, gamma such that:
    # F_nf_y = alpha * int_a2_sin + gamma * cross_val
    # We have two unknowns but only one equation per wavelength.
    # Let's just print the raw values so we can see the pattern.
    
    # Also try a completely different approach: compute with Newman's original formula
    # Newman (1967) eq 10 uses his C(θ) with:
    # F_y = (ρg/2) Re[C(β)] sinβ + (ρg/(2k)) ∫|C|² sinθ dθ
    # where C(θ) is the Kochin function in his normalization.
    # 
    # His relation to our a_d: need to figure this out from the damping formula.
    # Newman's damping: B = (ρk/2) ∫|C|² dθ  (his eq 7 or thereabouts)
    # Our damping: B = ρg/(2ωk) ∫|a|² dθ
    # So: (ρk/2)|C|² = (ρg/(2ωk))|a|² → |C|² = g/(ω k²) |a|²
    # → C = a √(g/(ωk²)) × (phase factor)
    # Since g/ω = ω/k (from ω²=gk): C = a × √(1/k³) × √ω × (phase)
    # Hmm, let me compute this differently.
    # |C|²/|a|² = g/(ωk²)
    # In Newman's formula:
    # quad = (ρg/(2k)) |C|² = (ρg/(2k)) × g/(ωk²) × |a|² = ρg²/(2ωk³) |a|²
    # cross = (ρg/2) Re[C(β)] sinβ
    
    # From ω²=gk → g = ω²/k, so:
    # quad_coeff = ρ(ω²/k)²/(2ωk³) = ρω⁴/(2ωk⁵) = ρω³/(2k⁵)  ... this is getting messy
    # Let me just compute numerically.
    
    ratio_C2_a2 = g / (omega * k**2)
    C2_sin_int = ratio_C2_a2 * int_a2_sin
    F_newman_quad = rho * g / (2*k) * C2_sin_int
    
    C_at_beta = a_at_beta * np.sqrt(ratio_C2_a2)  # magnitude, but need phase
    # Phase: C = a_d × (-i) × exp(+iπ/4) × √(πk/2)  ... from earlier attempt
    # Actually let me derive it fresh.
    # Newman: φ ~ (gA/ω) C(θ) exp(kz) exp(ikr-iπ/4) / √(πkr/2)
    # Us: η = a(θ) exp(ikr)/√r, φ = -ig/ω × η = -ig/ω × a(θ) exp(ikr)/√r
    # Newman: φ = (gA/ω) C(θ) × √(2/(πkr)) × exp(ikr-iπ/4) × exp(kz)
    # At z=0, A=1: φ = (g/ω) C(θ) × √(2/(πk)) × exp(ikr)/√r × exp(-iπ/4)
    # Equating: -ig/ω × a(θ) = (g/ω) × C(θ) × √(2/(πk)) × exp(-iπ/4)
    # → -i × a(θ) = C(θ) × √(2/(πk)) × exp(-iπ/4)
    # → C(θ) = -i a(θ) / (√(2/(πk)) × exp(-iπ/4))
    #         = -i a(θ) × exp(+iπ/4) × √(πk/2)
    #         = a(θ) × (-i) × exp(iπ/4) × √(πk/2)
    #         = a(θ) × exp(-iπ/2+iπ/4) × √(πk/2)
    #         = a(θ) × exp(-iπ/4) × √(πk/2)
    
    C_newman = a_scat * np.exp(-1j * np.pi/4) * np.sqrt(np.pi * k / 2)
    C_newman_at_beta = C_newman[idx_beta]
    
    # Verify: |C|² = |a|² × πk/2
    # Newman's B: (ρk/2) ∫|C|² dθ = (ρk/2)(πk/2) ∫|a|²dθ = ρπk²/4 ∫|a|²dθ
    # Our B: ρg/(2ωk) ∫|a|²dθ
    # Ratio: (ρπk²/4) / (ρg/(2ωk)) = πk²/(4) × 2ωk/g = πωk³/(2g) = π k²/(2) (using ω²=gk→ω/g=1/(ω/k)=k/ω... 
    # ω²=gk→ω=√(gk), g=ω²/k
    # πωk³/(2g) = πωk³/(2ω²/k) = πk⁴/(2ω)
    # For this to equal 1: πk⁴/(2ω) = 1 → k⁴ = 2ω/π ... not an identity!
    # So Newman's B formula with C(θ) gives a DIFFERENT coefficient than our B.
    # This means my C↔a relation must be WRONG.
    
    # Let me re-derive from damping equality:
    # Newman: B = (ρk/2) ∫|C|²dθ
    # Ours: B = ρg/(2ωk) ∫|a|²dθ
    # So: |C|² = g/(ωk²) × |a|² = ω/(k³) × |a|² (using g=ω²/k)
    # → |C| = |a| × √(ω/k³) = |a| × √(ω)/k^{3/2}
    
    C_mag_ratio = np.sqrt(omega / k**3)
    
    # Newman cross: (ρg/2) Re[C(β)] sinβ
    # Need to get the phase right. Using C = a × exp(-iπ/4) × √(πk/2):
    # |C|/|a| = √(πk/2) ... but we need |C|/|a| = √(ω/k³)
    # √(πk/2) vs √(ω/k³): these are equal when πk/2 = ω/k³ → πk⁴/2 = ω = √(gk)
    # Not an identity. So the relation C = a × exp(-iπ/4) × √(πk/2) is WRONG.
    
    # The issue: maybe Newman's B formula is not (ρk/2)∫|C|²dθ for deep water.
    # Or maybe his C(θ) convention is different from what I assumed.
    # 
    # Let me forget about Newman's convention and just work with a(θ) directly.
    # I'll derive the drift formula numerically.
    
    # === Method 2b: BRUTE FORCE numerical momentum flux ===
    # Compute the total velocity potential at z-levels below the surface
    # at radius R_FIELD, and numerically integrate the momentum flux.
    #
    # This is expensive but unambiguous.
    # For now, let me just print diagnostic info.
    
    F_maruo_y = F_quad_y + F_cross_y  # placeholder with ρg/4 formula
    
    ratio_nf_maruo = F_nf_y / F_maruo_y if abs(F_maruo_y) > 0.1 else float('inf')
    
    print(f"{lam:5.0f} {F_wl_y:10.1f} {F_vel_y:10.1f} {F_nf_y:10.1f} "
          f"{F_quad_y:10.1f} {F_cross_y:10.1f} "
          f"{F_maruo_y:10.1f} {ratio_nf_maruo:8.3f}")
