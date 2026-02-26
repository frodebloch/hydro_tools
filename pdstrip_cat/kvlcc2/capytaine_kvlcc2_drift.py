#!/usr/bin/env python3
"""
Build a 3D panel mesh for KVLCC2 from pdstrip's geomet.out sections,
then compute Maruo far-field drift forces using Capytaine's Kochin function.

Mesh convention: +x = forward, +y = starboard, +z = up
(Same as pdstrip internal coordinates, so drift forces map directly.)

pdstrip input convention: y=port, z=up -> negate y for our mesh.
"""
import numpy as np
import capytaine as cpt
from capytaine.meshes.meshes import Mesh
from capytaine.meshes.collections import CollectionOfMeshes
from capytaine.bem.airy_waves import froude_krylov_force
import logging
import re
import sys

cpt.set_logging(logging.WARNING)

rho = 1025.0
g = 9.81

# ============================================================
# Parse geomet.out
# ============================================================
def parse_geomet(filepath):
    """Parse pdstrip geometry file. Returns list of sections.
    Each section: {'x': float, 'npts': int, 'y': array, 'z': array}
    y and z are in pdstrip INPUT convention (y=port, z=up).
    """
    with open(filepath) as f:
        lines = f.readlines()
    
    # Line 0: nsections catamaran_flag draft
    parts = lines[0].split()
    nsec = int(parts[0])
    draft = float(parts[2])
    
    sections = []
    i = 1
    for _ in range(nsec):
        # Header: x npts iab
        hdr = lines[i].split()
        x = float(hdr[0])
        npts = int(hdr[1])
        i += 1
        
        # y values (may span multiple lines)
        yvals = []
        while len(yvals) < npts:
            yvals.extend([float(v) for v in lines[i].split()])
            i += 1
        
        # z values
        zvals = []
        while len(zvals) < npts:
            zvals.extend([float(v) for v in lines[i].split()])
            i += 1
        
        sections.append({
            'x': x,
            'npts': npts,
            'y_input': np.array(yvals[:npts]),
            'z_input': np.array(zvals[:npts]),
        })
    
    return sections, draft

sections, draft = parse_geomet('geomet.out')
print(f"Parsed {len(sections)} sections, draft={draft}m")
print(f"X range: {sections[0]['x']:.1f} to {sections[-1]['x']:.1f}")

# Section x-positions
x_positions = np.array([s['x'] for s in sections])
Lpp = x_positions[-1] - x_positions[0]
print(f"Lpp (from sections) = {Lpp:.1f}m")

# ============================================================
# Build 3D mesh by lofting between sections
# ============================================================
# Strategy: 
# 1. For sections with different numbers of points, we interpolate to a common resolution
# 2. Connect adjacent sections with quadrilateral panels (split into triangles)
#
# All sections except the last have 20 points.
# The last section (stern) has only 6 points — we'll handle it specially.

# First, let's resample all sections to a common number of points
# Use arc-length parameterization for each section

def resample_section(y_input, z_input, n_target):
    """Resample a section to n_target points using arc-length parameterization.
    Input: y (port=+), z (up=+) in pdstrip input convention.
    Output: y, z arrays with n_target points.
    """
    # Compute cumulative arc length
    dy = np.diff(y_input)
    dz = np.diff(z_input)
    ds = np.sqrt(dy**2 + dz**2)
    s = np.zeros(len(y_input))
    s[1:] = np.cumsum(ds)
    s_total = s[-1]
    
    # Uniform arc-length parameterization
    s_new = np.linspace(0, s_total, n_target)
    y_new = np.interp(s_new, s, y_input)
    z_new = np.interp(s_new, s, z_input)
    
    return y_new, z_new

# Target resolution: use the same as the majority of sections (20 points)
N_CIRC = 20  # circumferential points per section

# Resample all sections
print("\nResampling sections...")
resampled = []
for sec in sections:
    if sec['npts'] == N_CIRC:
        y_r, z_r = sec['y_input'], sec['z_input']
    else:
        y_r, z_r = resample_section(sec['y_input'], sec['z_input'], N_CIRC)
    resampled.append({'x': sec['x'], 'y': y_r, 'z': z_r})

# Convert to Capytaine convention: +y = starboard (negate input y)
# z stays the same (both up)
for sec in resampled:
    sec['y'] = -sec['y']  # port -> starboard

# Build vertices and faces
# Vertices: nsec * N_CIRC points
# Faces: (nsec-1) * (N_CIRC-1) quads -> 2 triangles each, or just quads
n_sec = len(resampled)
vertices = []
for i_sec in range(n_sec):
    s = resampled[i_sec]
    for j in range(N_CIRC):
        vertices.append([s['x'], s['y'][j], s['z'][j]])

vertices = np.array(vertices)
print(f"Total vertices: {len(vertices)}")

# Build quadrilateral faces
# Each quad connects (sec_i, pt_j) -> (sec_i, pt_j+1) -> (sec_i+1, pt_j+1) -> (sec_i+1, pt_j)
faces = []
for i_sec in range(n_sec - 1):
    for j in range(N_CIRC - 1):
        v00 = i_sec * N_CIRC + j
        v01 = i_sec * N_CIRC + j + 1
        v10 = (i_sec + 1) * N_CIRC + j
        v11 = (i_sec + 1) * N_CIRC + j + 1
        faces.append([v00, v10, v11, v01])

faces = np.array(faces)
print(f"Total faces: {len(faces)}")

# Create Capytaine mesh
hull_mesh = Mesh(vertices=vertices, faces=faces, name="kvlcc2_hull")

# Check mesh quality
print(f"Mesh bounds: x=[{vertices[:,0].min():.1f}, {vertices[:,0].max():.1f}], "
      f"y=[{vertices[:,1].min():.1f}, {vertices[:,1].max():.1f}], "
      f"z=[{vertices[:,2].min():.1f}, {vertices[:,2].max():.1f}]")

# Verify normals point outward
midship_mask = np.abs(hull_mesh.faces_centers[:, 0]) < 10.0
midship_normals = hull_mesh.faces_normals[midship_mask]
midship_centers = hull_mesh.faces_centers[midship_mask]
stb_mask = midship_centers[:, 1] > 5.0
if stb_mask.any():
    avg_ny = np.mean(midship_normals[stb_mask, 1])
    print(f"Normal check (stb midship): avg ny = {avg_ny:.4f} (should be >0)")
    assert avg_ny > 0, "Normals point inward — check face winding!"

print(f"Mesh volume: {hull_mesh.volume:.0f} m³")

# ============================================================
# Build Capytaine body
# ============================================================
# Add a lid mesh at z slightly below waterline for irregular frequency removal
lid = hull_mesh.generate_lid(z=-0.01)
body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name="kvlcc2")

# KVLCC2 mass properties
# From pdstrip.inp line 8: 320437550 11.1 0 -2.2 300 5849 5849 0.0 0.0 0.0
# mass=320437550 kg, xcg=11.1m, ycg=0, zcg=-2.2m
# kxx²=300 m², kyy²=5849 m², kzz²=5849 m²
mass_val = 320437550.0
xcg = 11.1
zcg = -2.2
kxx_sq = 300.0   # roll radius of gyration squared
kyy_sq = 5849.0  # pitch
kzz_sq = 5849.0  # yaw

body.center_of_mass = np.array([xcg, 0.0, zcg])
body.mass = mass_val
body.rotation_center = body.center_of_mass
body.add_all_rigid_body_dofs()

solver = cpt.BEMSolver()
dof_names = list(body.dofs.keys())
n_dof = len(dof_names)
print(f"\nDOFs: {dof_names}")

# Inertia matrix
M = np.zeros((n_dof, n_dof))
for i, dof in enumerate(dof_names):
    if dof in ('Surge', 'Sway', 'Heave'): M[i,i] = mass_val
    elif dof == 'Roll': M[i,i] = mass_val * kxx_sq
    elif dof == 'Pitch': M[i,i] = mass_val * kyy_sq
    elif dof == 'Yaw': M[i,i] = mass_val * kzz_sq

# Hydrostatic stiffness
stiffness_xr = body.compute_hydrostatic_stiffness()
C = np.zeros((n_dof, n_dof))
for i, idof in enumerate(dof_names):
    for j, jdof in enumerate(dof_names):
        try: C[i,j] = float(stiffness_xr.sel(influenced_dof=idof, radiating_dof=jdof))
        except: C[i,j] = 0.0

print(f"\nHydrostatic stiffness diagonal:")
for i, dof in enumerate(dof_names):
    if abs(C[i,i]) > 0:
        print(f"  C[{dof},{dof}] = {C[i,i]:.0f}")

# ============================================================
# Custom Kochin function that handles lid mesh correctly  
# ============================================================
def kochin_full(result, theta_arr):
    """Compute Kochin function H(θ) using full mesh (hull + lid)."""
    if body.lid_mesh is not None:
        full_mesh = CollectionOfMeshes([body.mesh, body.lid_mesh])
    else:
        full_mesh = body.mesh
    centers = full_mesh.faces_centers
    areas = full_mesh.faces_areas
    kk = result.wavenumber
    omega_bar = centers[:, 0:2] @ np.array([np.cos(theta_arr), np.sin(theta_arr)])
    cih = np.exp(kk * centers[:, 2])
    zs = (cih[:, None] * np.exp(-1j * kk * omega_bar) * areas[:, None])
    return (zs.T @ result.sources) / (4 * np.pi)

# Kochin parameters
N_THETA = 720
theta_arr = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
dtheta = 2*np.pi / N_THETA

# ============================================================
# Phase 1: Verify Kochin normalization via damping
# ============================================================
print(f"\n{'='*90}")
print("PHASE 1: Kochin normalization check (B_direct vs B_kochin)")
print("="*90)

# Use a medium wavelength for verification
test_lams = [100, 200, 400]
print(f"\n{'lam':>6} {'DOF':>8} {'B_direct':>14} {'B_kochin':>14} {'ratio':>8}")
print("-"*55)

for lam in test_lams:
    k = 2*np.pi/lam; omega = np.sqrt(k*g)
    for dof in ['Heave', 'Pitch']:
        rad_prob = cpt.RadiationProblem(body=body, radiating_dof=dof,
                                         omega=omega, water_depth=np.inf, rho=rho, g=g)
        rad_result = solver.solve(rad_prob, keep_details=True)
        B_direct = rad_result.radiation_dampings[dof]
        
        H_rad = kochin_full(rad_result, theta_arr)
        H2_int = np.sum(np.abs(H_rad)**2) * dtheta
        B_kochin = 4 * rho * k * np.pi / omega * H2_int
        
        r = B_direct / B_kochin if abs(B_kochin) > 1e-6 else float('nan')
        print(f"{lam:6.0f} {dof:>8} {B_direct:14.1f} {B_kochin:14.1f} {r:8.4f}")

# ============================================================
# Phase 2: Compute drift forces — HEAD SEAS (β=π), zero speed
# ============================================================
print(f"\n{'='*90}")
print("PHASE 2: KVLCC2 Maruo far-field drift — HEAD SEAS (β=π)")
print("="*90)

# Use pdstrip wavelengths — skip shortest ones where mesh is too coarse
# Full set from pdstrip.inp:
wavelengths_all = np.array([27.395, 30.440, 33.823, 37.583, 41.760, 46.402, 51.559, 57.290,
                             63.658, 70.734, 78.596, 87.333, 97.040, 107.826, 119.811, 133.128,
                             147.926, 164.368, 182.638, 202.939, 225.496, 250.560, 278.411,
                             309.356, 343.742, 381.950, 424.405, 471.578, 523.995, 582.238,
                             646.956, 718.866, 798.770, 887.555, 986.209])

# Use every wavelength — the mesh should be adequate for λ > ~50m (λ/panel_size > 5)
# For shorter wavelengths results will be approximate
wavelengths = wavelengths_all

beta_head = np.pi  # head seas

# Compute radiation for all wavelengths first
rad_cache = {}
AM_cache = {}

def get_radiation(lam):
    if lam in AM_cache:
        return AM_cache[lam], {dof: rad_cache[(lam, dof)] for dof in dof_names}
    k = 2*np.pi/lam; omega = np.sqrt(k*g)
    rad_results = {}
    for dof in dof_names:
        rad_prob = cpt.RadiationProblem(body=body, radiating_dof=dof,
                                         omega=omega, water_depth=np.inf, rho=rho, g=g)
        rad_results[dof] = solver.solve(rad_prob, keep_details=True)
        rad_cache[(lam, dof)] = rad_results[dof]
    A_mat = np.zeros((n_dof, n_dof)); B_mat = np.zeros((n_dof, n_dof))
    for i, rdof in enumerate(dof_names):
        for j, idof in enumerate(dof_names):
            A_mat[i,j] = rad_results[rdof].added_masses[idof]
            B_mat[i,j] = rad_results[rdof].radiation_dampings[idof]
    AM_cache[lam] = (A_mat, B_mat)
    return (A_mat, B_mat), rad_results

def compute_drift(lam, beta, component='x'):
    k = 2*np.pi/lam; omega = np.sqrt(k*g)
    (A_mat, B_mat), rad_results = get_radiation(lam)
    
    diff_prob = cpt.DiffractionProblem(body=body, wave_direction=beta,
                                        omega=omega, water_depth=np.inf, rho=rho, g=g)
    diff_result = solver.solve(diff_prob, keep_details=True)
    
    FK = froude_krylov_force(diff_prob)
    F_exc = np.array([diff_result.forces[dof] + FK[dof] for dof in dof_names])
    Z = -omega**2 * (M + A_mat) + 1j * omega * B_mat + C
    xi = np.linalg.solve(Z, F_exc)
    
    H_total = kochin_full(diff_result, theta_arr)
    for i, dof in enumerate(dof_names):
        H_total = H_total + xi[i] * kochin_full(rad_results[dof], theta_arr)
    
    if component == 'x':
        integral = np.sum(np.abs(H_total)**2 * np.cos(theta_arr)) * dtheta
    elif component == 'y':
        integral = np.sum(np.abs(H_total)**2 * np.sin(theta_arr)) * dtheta
    
    F = 2 * rho * np.pi * k * integral
    return F, xi

# Also compute fixed-body drift for reference
def compute_fixed_drift(lam, beta, component='x'):
    k = 2*np.pi/lam; omega = np.sqrt(k*g)
    diff_prob = cpt.DiffractionProblem(body=body, wave_direction=beta,
                                        omega=omega, water_depth=np.inf, rho=rho, g=g)
    diff_result = solver.solve(diff_prob, keep_details=True)
    H_scat = kochin_full(diff_result, theta_arr)
    if component == 'x':
        integral = np.sum(np.abs(H_scat)**2 * np.cos(theta_arr)) * dtheta
    elif component == 'y':
        integral = np.sum(np.abs(H_scat)**2 * np.sin(theta_arr)) * dtheta
    F = 2 * rho * np.pi * k * integral
    return F

# Normalization
B_ship = 58.0
norm_raw = rho * g * B_ship**2 / Lpp

print(f"\nLpp = {Lpp:.1f}m, B = {B_ship}m")
print(f"Normalization: rho*g*B^2/Lpp = {norm_raw:.0f} N/m")
print(f"\nComputing head seas drift for {len(wavelengths)} wavelengths...")
print(f"{'lam':>8} {'lam/L':>7} {'Fx_float':>14} {'Fx_fixed':>14} {'sigma_aw':>10} {'|heave|':>8} {'|pitch|/k':>10}")
print("-"*80)

results_head = []
for il, lam in enumerate(wavelengths):
    k = 2*np.pi/lam; omega = np.sqrt(k*g)
    lam_L = lam / Lpp
    
    try:
        F_x_float, xi = compute_drift(lam, beta_head, 'x')
        F_x_fixed = compute_fixed_drift(lam, beta_head, 'x')
        
        sigma_aw = -F_x_float / norm_raw  # positive = resistance
        
        heave_idx = dof_names.index('Heave')
        pitch_idx = dof_names.index('Pitch')
        heave_amp = np.abs(xi[heave_idx])
        pitch_amp_over_k = np.abs(xi[pitch_idx]) / k
        
        results_head.append({
            'wavelength': lam, 'lam_L': lam_L,
            'Fx_float': F_x_float, 'Fx_fixed': F_x_fixed,
            'sigma_aw': sigma_aw,
            'heave': heave_amp, 'pitch_k': pitch_amp_over_k,
            'xi': xi
        })
        
        print(f"{lam:8.1f} {lam_L:7.3f} {F_x_float:14.1f} {F_x_fixed:14.1f} {sigma_aw:10.4f} "
              f"{heave_amp:8.4f} {pitch_amp_over_k:10.4f}")
    except Exception as e:
        print(f"{lam:8.1f} {lam_L:7.3f}  ERROR: {e}")
        results_head.append({
            'wavelength': lam, 'lam_L': lam_L,
            'Fx_float': np.nan, 'Fx_fixed': np.nan,
            'sigma_aw': np.nan, 'heave': np.nan, 'pitch_k': np.nan,
            'xi': np.full(n_dof, np.nan)
        })
    
    sys.stdout.flush()

# ============================================================
# Phase 3: BEAM SEAS (β=π/2) — transverse drift
# ============================================================
print(f"\n{'='*90}")
print("PHASE 3: KVLCC2 Maruo far-field drift — BEAM SEAS (β=π/2)")
print("="*90)

beta_beam = np.pi / 2
norm_beam = rho * g * B_ship

print(f"\nNormalization: rho*g*B = {norm_beam:.0f} N/m")
print(f"\n{'lam':>8} {'lam/B':>7} {'Fy_float':>14} {'sigma_y':>10} {'|sway|':>8} {'|heave|':>8} {'|roll|/k':>10}")
print("-"*80)

results_beam = []
for il, lam in enumerate(wavelengths):
    k = 2*np.pi/lam; omega = np.sqrt(k*g)
    lam_B = lam / B_ship
    
    try:
        F_y_float, xi = compute_drift(lam, beta_beam, 'y')
        
        sigma_y = F_y_float / norm_beam
        
        sway_idx = dof_names.index('Sway')
        heave_idx = dof_names.index('Heave')
        roll_idx = dof_names.index('Roll')
        
        results_beam.append({
            'wavelength': lam, 'lam_B': lam_B,
            'Fy_float': F_y_float, 'sigma_y': sigma_y,
            'sway': np.abs(xi[sway_idx]),
            'heave': np.abs(xi[heave_idx]),
            'roll_k': np.abs(xi[roll_idx]) / k,
        })
        
        print(f"{lam:8.1f} {lam_B:7.2f} {F_y_float:14.1f} {sigma_y:10.4f} "
              f"{np.abs(xi[sway_idx]):8.4f} {np.abs(xi[heave_idx]):8.4f} "
              f"{np.abs(xi[roll_idx])/k:10.4f}")
    except Exception as e:
        print(f"{lam:8.1f} {lam_B:7.2f}  ERROR: {e}")
        results_beam.append({
            'wavelength': lam, 'lam_B': lam_B,
            'Fy_float': np.nan, 'sigma_y': np.nan,
            'sway': np.nan, 'heave': np.nan, 'roll_k': np.nan,
        })
    
    sys.stdout.flush()

# ============================================================
# Save results
# ============================================================
np.savez('capytaine_kvlcc2_drift.npz',
         wavelengths=wavelengths,
         Lpp=Lpp, B=B_ship, draft=draft,
         rho=rho, g=g,
         # Head seas
         Fx_float_head=np.array([r['Fx_float'] for r in results_head]),
         Fx_fixed_head=np.array([r['Fx_fixed'] for r in results_head]),
         sigma_aw=np.array([r['sigma_aw'] for r in results_head]),
         heave_head=np.array([r['heave'] for r in results_head]),
         pitch_k_head=np.array([r['pitch_k'] for r in results_head]),
         # Beam seas
         Fy_float_beam=np.array([r['Fy_float'] for r in results_beam]),
         sigma_y=np.array([r['sigma_y'] for r in results_beam]),
         )
print("\nSaved: capytaine_kvlcc2_drift.npz")
