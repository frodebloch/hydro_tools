#!/usr/bin/env python3
"""
KVLCC2 Capytaine drift with refined mesh.
- Use 40 circumferential points (doubled from 20)
- Use xOz symmetry to halve the BEM problem size
- Focus on wavelengths λ > 50m where mesh quality is adequate
"""
import numpy as np
import capytaine as cpt
from capytaine.meshes.meshes import Mesh
from capytaine.meshes.collections import CollectionOfMeshes
from capytaine.meshes.symmetric import ReflectionSymmetricMesh
from capytaine.bem.airy_waves import froude_krylov_force
import logging
import sys

cpt.set_logging(logging.WARNING)

rho = 1025.0; g = 9.81

# ============================================================
# Parse geomet.out
# ============================================================
def parse_geomet(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    parts = lines[0].split()
    nsec = int(parts[0]); draft = float(parts[2])
    sections = []; i = 1
    for _ in range(nsec):
        hdr = lines[i].split()
        x = float(hdr[0]); npts = int(hdr[1]); i += 1
        yvals = []
        while len(yvals) < npts:
            yvals.extend([float(v) for v in lines[i].split()]); i += 1
        zvals = []
        while len(zvals) < npts:
            zvals.extend([float(v) for v in lines[i].split()]); i += 1
        sections.append({'x': x, 'npts': npts,
                         'y_input': np.array(yvals[:npts]),
                         'z_input': np.array(zvals[:npts])})
    return sections, draft

sections, draft = parse_geomet('geomet.out')
print(f"Parsed {len(sections)} sections, draft={draft}m")

x_positions = np.array([s['x'] for s in sections])
Lpp = x_positions[-1] - x_positions[0]
print(f"Lpp = {Lpp:.1f}m")

# ============================================================
# Build refined mesh
# ============================================================
def resample_section(y_input, z_input, n_target):
    dy = np.diff(y_input); dz = np.diff(z_input)
    ds = np.sqrt(dy**2 + dz**2)
    s = np.zeros(len(y_input)); s[1:] = np.cumsum(ds)
    s_new = np.linspace(0, s[-1], n_target)
    return np.interp(s_new, s, y_input), np.interp(s_new, s, z_input)

# Refine: double circumferential resolution
N_CIRC = 40

resampled = []
for sec in sections:
    y_r, z_r = resample_section(sec['y_input'], sec['z_input'], N_CIRC)
    # Convert to +y=starboard (negate pdstrip input y)
    resampled.append({'x': sec['x'], 'y': -y_r, 'z': z_r})

# Also add interpolated sections between existing ones for longitudinal refinement
# This doubles the longitudinal resolution
sections_fine = []
for i in range(len(resampled)):
    sections_fine.append(resampled[i])
    if i < len(resampled) - 1:
        # Interpolate between section i and i+1
        x_mid = 0.5 * (resampled[i]['x'] + resampled[i+1]['x'])
        y_mid = 0.5 * (resampled[i]['y'] + resampled[i+1]['y'])
        z_mid = 0.5 * (resampled[i]['z'] + resampled[i+1]['z'])
        sections_fine.append({'x': x_mid, 'y': y_mid, 'z': z_mid})

resampled = sections_fine
n_sec = len(resampled)
print(f"Refined: {n_sec} sections x {N_CIRC} points = {n_sec * N_CIRC} vertices")

# Build vertices
vertices = []
for i_sec in range(n_sec):
    s = resampled[i_sec]
    for j in range(N_CIRC):
        vertices.append([s['x'], s['y'][j], s['z'][j]])
vertices = np.array(vertices)

# Build quad faces (outward normals verified)
faces = []
for i_sec in range(n_sec - 1):
    for j in range(N_CIRC - 1):
        v00 = i_sec * N_CIRC + j
        v01 = i_sec * N_CIRC + j + 1
        v10 = (i_sec + 1) * N_CIRC + j
        v11 = (i_sec + 1) * N_CIRC + j + 1
        faces.append([v00, v10, v11, v01])
faces = np.array(faces)

print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")
print(f"Bounds: x=[{vertices[:,0].min():.1f}, {vertices[:,0].max():.1f}], "
      f"y=[{vertices[:,1].min():.1f}, {vertices[:,1].max():.1f}], "
      f"z=[{vertices[:,2].min():.1f}, {vertices[:,2].max():.1f}]")

hull_mesh = Mesh(vertices=vertices, faces=faces, name="kvlcc2_hull")

# Verify normals
midship_mask = np.abs(hull_mesh.faces_centers[:, 0]) < 10.0
midship_normals = hull_mesh.faces_normals[midship_mask]
midship_centers = hull_mesh.faces_centers[midship_mask]
stb_mask = midship_centers[:, 1] > 5.0
if stb_mask.any():
    avg_ny = np.mean(midship_normals[stb_mask, 1])
    print(f"Normal check: avg ny(stb) = {avg_ny:.4f}")
    assert avg_ny > 0, "Normals inverted!"

print(f"Mesh volume: {hull_mesh.volume:.0f} m³ (expected: {320437550/1025:.0f} m³)")

# ============================================================
# Build body with lid
# ============================================================
lid = hull_mesh.generate_lid(z=-0.01)
body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name="kvlcc2")
print(f"Hull panels: {hull_mesh.nb_faces}, Lid panels: {lid.nb_faces}")

mass_val = 320437550.0
body.center_of_mass = np.array([11.1, 0.0, -2.2])
body.mass = mass_val
body.rotation_center = body.center_of_mass
body.add_all_rigid_body_dofs()

solver = cpt.BEMSolver()
dof_names = list(body.dofs.keys())
n_dof = len(dof_names)

# Mass matrix
M = np.zeros((n_dof, n_dof))
for i, dof in enumerate(dof_names):
    if dof in ('Surge', 'Sway', 'Heave'): M[i,i] = mass_val
    elif dof == 'Roll': M[i,i] = mass_val * 300.0
    elif dof == 'Pitch': M[i,i] = mass_val * 5849.0
    elif dof == 'Yaw': M[i,i] = mass_val * 5849.0

# Hydrostatic stiffness
stiffness_xr = body.compute_hydrostatic_stiffness()
C = np.zeros((n_dof, n_dof))
for i, idof in enumerate(dof_names):
    for j, jdof in enumerate(dof_names):
        try: C[i,j] = float(stiffness_xr.sel(influenced_dof=idof, radiating_dof=jdof))
        except: C[i,j] = 0.0

print(f"C_heave = {C[dof_names.index('Heave'), dof_names.index('Heave')]:.0f}")
print(f"C_pitch = {C[dof_names.index('Pitch'), dof_names.index('Pitch')]:.0f}")

# ============================================================
# Kochin function
# ============================================================
def kochin_full(result, theta_arr):
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

N_THETA = 720
theta_arr = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
dtheta = 2*np.pi / N_THETA

# ============================================================
# Kochin normalization check
# ============================================================
print(f"\n{'='*70}")
print("Kochin normalization check")
print("="*70)
for test_lam in [200, 400]:
    k = 2*np.pi/test_lam; omega = np.sqrt(k*g)
    for dof in ['Heave']:
        rad_prob = cpt.RadiationProblem(body=body, radiating_dof=dof,
                                         omega=omega, water_depth=np.inf, rho=rho, g=g)
        rad_result = solver.solve(rad_prob, keep_details=True)
        B_direct = rad_result.radiation_dampings[dof]
        H_rad = kochin_full(rad_result, theta_arr)
        H2_int = np.sum(np.abs(H_rad)**2) * dtheta
        B_kochin = 4 * rho * k * np.pi / omega * H2_int
        r = B_direct / B_kochin
        print(f"  λ={test_lam}m {dof}: B_direct/B_kochin = {r:.4f}")

# ============================================================
# Compute drift — HEAD SEAS  
# ============================================================
print(f"\n{'='*70}")
print("HEAD SEAS (β=π) — Maruo far-field drift")
print("="*70)

# Focus on wavelengths where mesh should be adequate
wavelengths_all = np.array([27.395, 30.440, 33.823, 37.583, 41.760, 46.402, 51.559, 57.290,
                             63.658, 70.734, 78.596, 87.333, 97.040, 107.826, 119.811, 133.128,
                             147.926, 164.368, 182.638, 202.939, 225.496, 250.560, 278.411,
                             309.356, 343.742, 381.950, 424.405, 471.578, 523.995, 582.238,
                             646.956, 718.866, 798.770, 887.555, 986.209])
wavelengths = wavelengths_all  # try all

beta_head = np.pi
B_ship = 58.0
norm_raw = rho * g * B_ship**2 / Lpp

rad_cache = {}; AM_cache = {}

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

print(f"\nLpp={Lpp:.1f}m, norm_raw={norm_raw:.0f} N/m")
print(f"{'lam':>8} {'lam/L':>7} {'sigma_aw':>10} {'heave':>8} {'pitch/k':>10}")
print("-"*50)

results = {'wavelengths': [], 'sigma_aw': [], 'heave': [], 'pitch_k': [],
           'Fx_float': [], 'Fy_float': [], 'sigma_y': []}

for lam in wavelengths:
    k = 2*np.pi/lam; lam_L = lam / Lpp
    try:
        F_x, xi = compute_drift(lam, beta_head, 'x')
        sigma = -F_x / norm_raw
        h_i = dof_names.index('Heave'); p_i = dof_names.index('Pitch')
        h = np.abs(xi[h_i]); pk = np.abs(xi[p_i]) / k
        results['wavelengths'].append(lam)
        results['sigma_aw'].append(sigma)
        results['heave'].append(h); results['pitch_k'].append(pk)
        results['Fx_float'].append(F_x)
        print(f"{lam:8.1f} {lam_L:7.3f} {sigma:10.4f} {h:8.4f} {pk:10.4f}")
    except Exception as e:
        print(f"{lam:8.1f} {lam_L:7.3f}  ERROR: {e}")
    sys.stdout.flush()

# ============================================================
# BEAM SEAS
# ============================================================
print(f"\n{'='*70}")
print("BEAM SEAS (β=π/2)")
print("="*70)
beta_beam = np.pi / 2
norm_beam = rho * g * B_ship

print(f"{'lam':>8} {'lam/B':>7} {'sigma_y':>10} {'sway':>8} {'heave':>8} {'roll/k':>10}")
print("-"*60)

for lam in wavelengths:
    k = 2*np.pi/lam; lam_B = lam / B_ship
    try:
        F_y, xi = compute_drift(lam, beta_beam, 'y')
        sigma_y = F_y / norm_beam
        sw_i = dof_names.index('Sway'); h_i = dof_names.index('Heave'); r_i = dof_names.index('Roll')
        results['Fy_float'].append(F_y)
        results['sigma_y'].append(sigma_y)
        print(f"{lam:8.1f} {lam_B:7.2f} {sigma_y:10.4f} {np.abs(xi[sw_i]):8.4f} "
              f"{np.abs(xi[h_i]):8.4f} {np.abs(xi[r_i])/k:10.4f}")
    except Exception as e:
        print(f"{lam:8.1f} {lam_B:7.2f}  ERROR: {e}")
    sys.stdout.flush()

# Save
np.savez('capytaine_kvlcc2_drift_refined.npz',
         wavelengths=np.array(results['wavelengths']),
         Lpp=Lpp, B=B_ship, draft=draft, rho=rho, g=g,
         sigma_aw=np.array(results['sigma_aw']),
         Fx_float=np.array(results['Fx_float']),
         heave=np.array(results['heave']),
         pitch_k=np.array(results['pitch_k']),
         Fy_float=np.array(results.get('Fy_float', [])),
         sigma_y=np.array(results.get('sigma_y', [])))
print("\nSaved: capytaine_kvlcc2_drift_refined.npz")
