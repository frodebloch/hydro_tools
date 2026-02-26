#!/usr/bin/env python3
"""
Debug the near-field waterline integral by comparing Capytaine's waterline
pressures/elevations with pdstrip's, for beam seas at lambda=3m.

pdstrip debug.out shows for beam seas lambda=3m (omega=4.533, mu=90):
  |p_stb| = 20557 Pa/m   (starboard = wave-facing side for mu=90)
  |p_port| = 1745 Pa/m   (lee side)
  dfeta = 0.25 * dx2 * (|p_stb|^2 - |p_port|^2) / (rho*g)

pdstrip's waterline contribution: feta_cum = 127139 N (positive = toward starboard)

We need to understand:
1. What |pressure| values does Capytaine give at equivalent waterline points?
2. Is the waterline edge extraction correct?
3. Are the normals correct?
"""

import numpy as np
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_potential, airy_waves_velocity
import logging

cpt.set_logging(logging.WARNING)

R = 1.0
L = 20.0
rho = 1025.0
g = 9.81
mesh_res = (10, 40, 50)

def make_hull_body(R, L, name="hull"):
    mesh_full = cpt.mesh_horizontal_cylinder(
        length=L, radius=R, center=(0, 0, 0),
        resolution=mesh_res, name=name
    )
    hull_mesh = mesh_full.immersed_part()
    lid = hull_mesh.generate_lid(z=-0.01)
    body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name=name)
    volume = np.pi * R**2 / 2 * L
    mass_val = rho * volume
    zcg = -4 * R / (3 * np.pi)
    body.center_of_mass = np.array([0.0, 0.0, zcg])
    body.mass = mass_val
    Ixx = mass_val * R**2 / 4
    Iyy = mass_val * (L**2 / 12 + R**2 / 4)
    Izz = mass_val * (L**2 / 12 + R**2 / 4)
    body.rotation_center = body.center_of_mass
    body.add_all_rigid_body_dofs()
    return body, mass_val, np.diag([Ixx, Iyy, Izz])


body, mass_val, inertia = make_hull_body(R, L)
hull_mesh = body.mesh

print("=== MESH INFO ===")
print(f"Hull faces: {hull_mesh.nb_faces}")
print(f"Hull vertices: {hull_mesh.nb_vertices}")

# Check waterline vertices
vertices = hull_mesh.vertices
z_tol = 0.05
wl_verts_idx = np.where(np.abs(vertices[:, 2]) < z_tol)[0]
print(f"\nWaterline vertices (z~0): {len(wl_verts_idx)}")
if len(wl_verts_idx) > 0:
    print(f"  y range: [{vertices[wl_verts_idx, 1].min():.3f}, {vertices[wl_verts_idx, 1].max():.3f}]")
    print(f"  x range: [{vertices[wl_verts_idx, 0].min():.3f}, {vertices[wl_verts_idx, 0].max():.3f}]")
    # Show some waterline vertices sorted by y
    wl_sorted = wl_verts_idx[np.argsort(vertices[wl_verts_idx, 1])]
    print(f"  First 5 (by y): {vertices[wl_sorted[:5]]}")
    print(f"  Last 5 (by y):  {vertices[wl_sorted[-5:]]}")

# Check normals at waterline faces
faces = hull_mesh.faces
normals = hull_mesh.faces_normals
centers = hull_mesh.faces_centers

# Find faces that touch waterline
wl_verts_set = set(wl_verts_idx)
wl_faces = []
for fi in range(hull_mesh.nb_faces):
    face = faces[fi]
    face_verts = set(face) if face[0] != face[3] else {face[0], face[1], face[2]}
    if face_verts & wl_verts_set:
        wl_faces.append(fi)
print(f"\nFaces touching waterline: {len(wl_faces)}")

# Extract waterline edges
from capytaine_nearfield_drift import extract_waterline_edges
edges, edge_centers, edge_lengths, edge_normals = extract_waterline_edges(hull_mesh)
print(f"\nWaterline edges: {len(edges)}")
print(f"Total waterline length: {np.sum(edge_lengths):.2f} m")

# Analyze waterline geometry
if len(edges) > 0:
    # Sort edges by y coordinate of center
    sort_idx = np.argsort(edge_centers[:, 1])
    
    print("\nSample waterline edges (sorted by y):")
    for idx in [0, len(edges)//4, len(edges)//2, 3*len(edges)//4, len(edges)-1]:
        i = sort_idx[idx]
        v1, v2 = edges[i]
        print(f"  Edge {i}: v1={vertices[v1]}, v2={vertices[v2]}")
        print(f"    center={edge_centers[i]}, length={edge_lengths[i]:.4f}")
        print(f"    outward normal={edge_normals[i]}")
    
    # Check: do normals point outward correctly?
    # For this half-cylinder (y>0 starboard), waterline edges at y>0 should 
    # have normals pointing in +y direction, and edges at y<0 should point -y
    print("\n  Sanity check: edge normals at +y should have n_y > 0, at -y should have n_y < 0")
    pos_y = edge_centers[:, 1] > 0.1
    neg_y = edge_centers[:, 1] < -0.1
    if np.any(pos_y):
        print(f"  Edges at y>0: mean n_y = {edge_normals[pos_y, 1].mean():.3f}")
    if np.any(neg_y):
        print(f"  Edges at y<0: mean n_y = {edge_normals[neg_y, 1].mean():.3f}")

# ============================================================
# Now compute pressures for beam seas lambda=3m
# ============================================================
print("\n\n=== BEAM SEAS LAMBDA=3m (omega=4.533, beta=pi/2) ===")

omega = np.sqrt(2*np.pi/3.0 * g)
k = omega**2 / g
beta = np.pi/2

print(f"omega={omega:.4f}, k={k:.4f}")

solver = cpt.BEMSolver()

# Diffraction
diff_prob = cpt.DiffractionProblem(
    body=body, wave_direction=beta, omega=omega,
    water_depth=np.inf, rho=rho, g=g
)
diff_result = solver.solve(diff_prob)

hull_mask = body.hull_mask

# Incident potential at waterline
wl_eval_pts = edge_centers.copy()
wl_eval_pts[:, 2] = -0.001

inc_pot_wl = airy_waves_potential(wl_eval_pts, diff_prob)
diff_pot_wl = solver.compute_potential(wl_eval_pts, diff_result)
total_pot_wl = inc_pot_wl + diff_pot_wl  # No radiation for fixed body test

# Convert to pressure: p = iωρφ (Capytaine convention)
p_wl = 1j * omega * rho * total_pot_wl
inc_p_wl = 1j * omega * rho * inc_pot_wl

# Wave elevation at waterline
eta_wl = (1j * omega / g) * total_pot_wl

print(f"\n|pressure| at waterline (per unit wave amplitude):")
print(f"  Note: pdstrip reports |p_stb|=20557 and |p_port|=1745 for this case")

# Sort by y to see starboard vs port
sort_y = np.argsort(edge_centers[:, 1])

# Group by y sign
stb_mask = edge_centers[:, 1] > 0.5  # starboard (y > 0 in Capytaine)
port_mask = edge_centers[:, 1] < -0.5  # port (y < 0 in Capytaine)

# Wait - what is Capytaine's convention for beta=pi/2?
# Capytaine: incident wave = exp(ik(x cos beta + y sin beta))
# beta=pi/2: wave propagates in +y direction, so wave comes from -y (port side)
# pdstrip: mu=90 means wave from starboard (y>0 in pdstrip internal coords, 
#          but pdstrip has inverted y convention!)
#
# Actually in pdstrip, y>0 = starboard (internal, sign-flipped from input).
# mu=90 = from starboard = wave propagating in -y direction (in pdstrip internal coords)
# = wave coming from +y side.
#
# In Capytaine, beta=pi/2: wave propagates in +y direction = comes from -y side.
# So Capytaine beta=pi/2 corresponds to pdstrip mu = -90 (from port side).
# 
# For pdstrip mu=90 (starboard), we need Capytaine beta = -pi/2 or 3*pi/2.
# 
# BUT the drift force should be symmetric (|Fy(mu=90)| = |Fy(mu=-90)|) for a
# port-starboard symmetric body, so this doesn't affect magnitudes.

print(f"\nCapytaine beta=pi/2: wave propagates in +y direction")
print(f"  Wave-facing side: y < 0 (port in standard coords)")
print(f"  Lee side: y > 0 (starboard in standard coords)")

if np.any(stb_mask):
    p_stb = np.abs(p_wl[stb_mask])
    print(f"\n  Starboard (y>0, LEE side): |p| range = [{p_stb.min():.1f}, {p_stb.max():.1f}]")
    print(f"    mean |p| = {p_stb.mean():.1f}")

if np.any(port_mask):
    p_port = np.abs(p_wl[port_mask])
    print(f"\n  Port (y<0, WAVE-facing side): |p| range = [{p_port.min():.1f}, {p_port.max():.1f}]")
    print(f"    mean |p| = {p_port.mean():.1f}")

# The incident wave alone
if np.any(stb_mask):
    inc_stb = np.abs(inc_p_wl[stb_mask])
    print(f"\n  Incident-only pressure at starboard: mean |p_inc| = {inc_stb.mean():.1f}")
if np.any(port_mask):
    inc_port = np.abs(inc_p_wl[port_mask])
    print(f"  Incident-only pressure at port: mean |p_inc| = {inc_port.mean():.1f}")

# What's the hydrostatic pressure at waterline? It's ρg×η for the dynamic part
# but at z=0, the hydrostatic pressure is 0. The first-order dynamic pressure is:
# p = iωρφ = ρg × η (at z=0, since η = iω/g × φ)
# So |p| = ρg × |η|
print(f"\n  Check: ρg = {rho*g:.1f}")
print(f"  For |p|=20557 Pa: |η| = {20557/(rho*g):.3f} m  (per unit amp)")
print(f"  For |p|=1745 Pa:  |η| = {1745/(rho*g):.3f} m  (per unit amp)")

# Check |eta| at waterline
if len(edges) > 0:
    eta_stb = np.abs(eta_wl[stb_mask]) if np.any(stb_mask) else []
    eta_port = np.abs(eta_wl[port_mask]) if np.any(port_mask) else []
    if len(eta_stb) > 0:
        print(f"\n  Capytaine |η| at starboard (lee): range [{eta_stb.min():.3f}, {eta_stb.max():.3f}]")
    if len(eta_port) > 0:
        print(f"  Capytaine |η| at port (wave):     range [{eta_port.min():.3f}, {eta_port.max():.3f}]")

# Compute the waterline integral properly and show breakdown
print("\n\n=== WATERLINE INTEGRAL BREAKDOWN ===")

# For FIXED body, η_rel = η (no body motion)
# For the y-component of drift force:
# F̄_y = (1/2) ρg Σ |η|² n_y dl

# Group edges by y position for a section-by-section view
# The semi-circular barge has waterline at y = +R (starboard) and y = -R (port)
# plus end cap edges

# Actually, the waterline of a half-cylinder at z=0 consists of:
# - Two straight lines along the length at y = +R and y = -R (the flat waterplane edges)
# - Two semicircular arcs at the ends (x = +L/2 and x = -L/2)

print(f"\nEdge normal analysis:")
print(f"  n_y > 0 edges: {np.sum(edge_normals[:, 1] > 0.1)}")
print(f"  n_y < 0 edges: {np.sum(edge_normals[:, 1] < -0.1)}")
print(f"  n_x > 0 edges: {np.sum(edge_normals[:, 0] > 0.1)}")
print(f"  n_x < 0 edges: {np.sum(edge_normals[:, 0] < -0.1)}")
print(f"  |n_y| < 0.1 edges: {np.sum(np.abs(edge_normals[:, 1]) < 0.1)}")

# The Fy waterline integral
Fy_wl = 0.5 * rho * g * np.sum(np.abs(eta_wl)**2 * edge_normals[:, 1] * edge_lengths)
Fx_wl = 0.5 * rho * g * np.sum(np.abs(eta_wl)**2 * edge_normals[:, 0] * edge_lengths)

print(f"\nWaterline drift force (fixed body):")
print(f"  Fx_wl = {Fx_wl:.1f}")
print(f"  Fy_wl = {Fy_wl:.1f}")

# Breakdown: starboard (n_y > 0) vs port (n_y < 0) contributions
stb_n = edge_normals[:, 1] > 0.1
port_n = edge_normals[:, 1] < -0.1
end_n = np.abs(edge_normals[:, 1]) < 0.1

Fy_stb = 0.5 * rho * g * np.sum(np.abs(eta_wl[stb_n])**2 * edge_normals[stb_n, 1] * edge_lengths[stb_n])
Fy_port = 0.5 * rho * g * np.sum(np.abs(eta_wl[port_n])**2 * edge_normals[port_n, 1] * edge_lengths[port_n])
Fy_end = 0.5 * rho * g * np.sum(np.abs(eta_wl[end_n])**2 * edge_normals[end_n, 1] * edge_lengths[end_n])

print(f"\n  Fy breakdown:")
print(f"    Starboard edge (n_y>0, lee side): {Fy_stb:.1f}")
print(f"    Port edge (n_y<0, wave side):     {Fy_port:.1f}")
print(f"    End cap edges (|n_y|<0.1):        {Fy_end:.1f}")
print(f"    Total:                            {Fy_stb + Fy_port + Fy_end:.1f}")

# Compare with pdstrip formula
# pdstrip: dfeta = 0.25 * dx2 * (|p_stb|^2 - |p_port|^2) / (ρg)
# For the full length with uniform pressure (5 sections):
# feta_WL = 0.25 * L * (|p_stb|^2 - |p_port|^2) / (ρg)
# Note: "0.25" comes from the time average factor 1/2 × the 1/2 from the section formula
# Actually: the factor is 1/(4*ρg) × dx2... Let me re-derive.
#
# pdstrip's waterline term: dfeta = 0.25 * dx2 * (|p_stb|^2 - |p_port|^2) / (rho*g)
# This is per section. Sum over sections gives the full waterline Fy.
#
# The near-field formula: F̄_y = (1/2)ρg ∮ |η|² n_y dl
# Since p = ρg×η at the waterline (z=0), we have |η| = |p|/(ρg)
# F̄_y = (1/2)ρg ∮ |p/(ρg)|² n_y dl = 1/(2ρg) ∮ |p|² n_y dl
#
# For the straight waterline edges:
#   - Starboard (y=+R): n_y = +1, dl = dx, and p = p_stb
#   - Port (y=-R): n_y = -1, dl = dx, and p = p_port
#
# F̄_y = 1/(2ρg) [∫ |p_stb|² dx - ∫ |p_port|² dx]
#      = 1/(2ρg) × L × (|p_stb|² - |p_port|²)   [for uniform p along length]
#
# pdstrip's sum: Σ 0.25*dx2 = 0.25 × L (since Σdx2 = L)
# So pdstrip's total = 0.25/(ρg) × L × (|p_stb|² - |p_port|²) = L/(4ρg) × (...)
# But our formula gives L/(2ρg) × (...) 
# Factor of 2 difference? Let me check...
# 
# pdstrip: the 0.25 = (1/2) × (1/2)
# The first 1/2 is the time average ⟨Re(Ae^{iwt})²⟩ = (1/2)|A|²
# The second 1/2 comes from... the section spacing formula? 
# Actually: dx2 = half-section spacing, so the total is 2×Σdx2 contributions but 
# each dfeta already uses dx2 (half-width), so Σdfeta = 0.25×(Σdx2)×... = 0.25*L*...
# Hmm, actually Σdx2 = L/2 + L/2 = L for the 5-section case.
# No: dx2 for sec 1 = 2.5, sec 2-4 = 5.0, sec 5 = 2.5, sum = 20 = L ✓
# So pdstrip total WL = 0.25/ρg × L × (|p_stb|² - |p_port|²)  
#                      = L/(4ρg) × Δ|p|²
#
# Our formula = L/(2ρg) × Δ|p|² 
# So our formula should give TWICE pdstrip's waterline contribution.
# 
# Wait, but our formula is (1/2)ρg ∮ |η|² n_y dl
# = (1/2)ρg × |η|² × 2 × L  [two sides, +L from stb and -L from port]
# No, it's: (1/2)ρg × [|η_stb|² × (+1) × L + |η_port|² × (-1) × L]
# = (L/2) ρg × (|η_stb|² - |η_port|²)
# = (L/2) ρg × (|p_stb|²/(ρg)² - |p_port|²/(ρg)²)
# = L/(2ρg) × (|p_stb|² - |p_port|²)  ✓
#
# pdstrip = L/(4ρg) × (|p_stb|² - |p_port|²)
# So pdstrip is HALF of the correct near-field formula? That would be a bug in pdstrip.
# Unless pdstrip's "0.25" factor actually comes from a different derivation...
#
# Actually, wait. pdstrip stores pres() as "pressure per unit wave amplitude" which 
# includes the factor for exp(+iwt). The time average is:
# ⟨p² ⟩ = (1/2) |p_complex|²
# So: pdstrip's 0.25 = (1/4) = (1/2)(time avg) × (1/2)(something else?)
# Or pdstrip's factor is 0.25 = 1/(2×2) where one 2 is time-average and the other is... ρg?
# Let me check units:
# dfeta = 0.25 * dx2 * |p|² / (ρg)  →  [N/m²]² × m / (kg/m³ × m/s²) = Pa² × m × s²/(kg × m)
# = (kg/(m×s²))² × m × s²/(kg×m) = kg²/(m²s⁴) × m × s²/(kg×m) = kg/(m²×s²) = Pa/m???
# Hmm the units don't work out unless |p| is actually pressure/length or something.

# Actually |p| in pdstrip is PRESSURE PER UNIT WAVE AMPLITUDE (Pa/m) based on the 
# debug output showing |p_stb| = 20557. For unit wave amplitude, the incident wave 
# pressure at z=0 would be ρg = 10055 Pa/m. So 20557 is about 2× ρg, consistent 
# with standing wave (reflected + incident).

# Let me just compute what pdstrip would get with these numbers:
p_stb_pd = 20557.5  # from debug.out for mu=90, lambda=3
p_port_pd = 1744.82
L_eff = 20.0
pd_fy_wl = 0.25 * L_eff * (p_stb_pd**2 - p_port_pd**2) / (rho * g)
our_fy_wl = 0.5 * L_eff * (p_stb_pd**2 - p_port_pd**2) / (rho * g) 
print(f"\n\n=== FORMULA COMPARISON ===")
print(f"Using pdstrip's |p_stb|={p_stb_pd:.1f}, |p_port|={p_port_pd:.1f}")
print(f"pdstrip WL Fy = 0.25*L*(|p_s|²-|p_p|²)/(ρg) = {pd_fy_wl:.1f}")
print(f"Our WL Fy     = 0.5*L*(|p_s|²-|p_p|²)/(ρg)  = {our_fy_wl:.1f}")
print(f"pdstrip total Fy (from debug) = 127139 (WL only), 99975 (with triangles)")
print(f"Ratio our/pd = {our_fy_wl/pd_fy_wl:.2f}")

# So the waterline formula factor: pdstrip uses 1/4, standard theory uses 1/2.
# pdstrip's factor 0.25 = (1/2) * (1/2)?
# The standard time average formula is F̄ = (1/2)Re(A × conj(B))
# For squared terms: ⟨(Re Ae^{iwt})²⟩ = (1/2)|A|²
# So the waterline integral is (1/2) × ρg × ∮ (1/2)|η|² n dl = (1/4)ρg ∮ |η|² n dl ?
#
# NO! The factor is (1/2)ρg ∮ <η²> n dl where <η²> = (1/2)|η_complex|²
# So total = (1/2)ρg × (1/2)|η|² × L × ... = (1/4)ρg |η|² L ...
#
# OH! I think I had the wrong formula. The mean second-order pressure at waterline gives:
# F̄ = (1/2) ρg <η²_rel> × n × dl
# where <η²_rel> = (1/2) |η_rel_complex|²  (time average of squared oscillation)
# So: F̄ = (1/4) ρg |η_rel_complex|² × n × dl
# NOT (1/2) ρg |η_rel_complex|² × n × dl !

# The confusion is between <η²> and |η_complex|².
# For η(t) = Re[η̂ e^{-iωt}], we have <η(t)²> = (1/2)|η̂|²
# So (1/2)ρg × <η²> = (1/2)ρg × (1/2)|η̂|² = (1/4)ρg|η̂|²

# Let me verify: pdstrip uses 0.25*|p|²/(ρg) = (1/4)|p|²/(ρg)
# With p = ρg×η, this gives (1/4)|ρgη|²/(ρg) = (1/4)ρg|η|²
# YES! So pdstrip's formula IS the correct near-field formula with the time average.

# My code had (1/2)ρg|η̂|² which is WRONG — should be (1/4)ρg|η̂|²
# Or equivalently (1/2)ρg × <η²> where <η²> = (1/2)|η̂|²

print(f"\n=== CORRECTED FORMULA ===")
print(f"The correct waterline drift force uses (1/2)ρg <η²> = (1/4)ρg|η̂|²")
print(f"My original code used (1/2)ρg|η̂|² which is 2× too large!")
print(f"pdstrip's factor 0.25 = (1/4) is correct: time-avg of Re²")

# Similarly, the velocity-squared term should be:
# F̄_vel = -(1/2)ρ <|∇φ|²> n dS = -(1/2)ρ × (1/2)|∇φ̂|² × n × dS = -(1/4)ρ|∇φ̂|² n dS
# And the rotation term: (1/2)Re(p × n × α*) which already has the (1/2) from time avg
print(f"\nThe velocity-squared term should also use (1/4)ρ|∇φ̂|² not (1/2)ρ|∇φ̂|²")
print(f"The rotation term (1/2)Re(p × n × α*) is CORRECT as written (already time-averaged)")
