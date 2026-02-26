#!/usr/bin/env python3
"""Quick test: build KVLCC2 mesh and verify it looks sensible."""
import numpy as np
import capytaine as cpt
from capytaine.meshes.meshes import Mesh
import logging

cpt.set_logging(logging.WARNING)

# Parse geomet.out
def parse_geomet(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    parts = lines[0].split()
    nsec = int(parts[0])
    draft = float(parts[2])
    sections = []
    i = 1
    for _ in range(nsec):
        hdr = lines[i].split()
        x = float(hdr[0])
        npts = int(hdr[1])
        i += 1
        yvals = []
        while len(yvals) < npts:
            yvals.extend([float(v) for v in lines[i].split()])
            i += 1
        zvals = []
        while len(zvals) < npts:
            zvals.extend([float(v) for v in lines[i].split()])
            i += 1
        sections.append({'x': x, 'npts': npts,
                         'y_input': np.array(yvals[:npts]),
                         'z_input': np.array(zvals[:npts])})
    return sections, draft

sections, draft = parse_geomet('geomet.out')
print(f"Parsed {len(sections)} sections, draft={draft}m")

# Check point counts
for i, s in enumerate(sections):
    if s['npts'] != 20:
        print(f"  Section {i}: x={s['x']:.1f}, npts={s['npts']}")

# Check section geometry
print(f"\nSection 0 (bow, x={sections[0]['x']:.1f}):")
print(f"  y range: [{sections[0]['y_input'].min():.1f}, {sections[0]['y_input'].max():.1f}]")
print(f"  z range: [{sections[0]['z_input'].min():.1f}, {sections[0]['z_input'].max():.1f}]")

mid = len(sections) // 2
print(f"\nSection {mid} (midship, x={sections[mid]['x']:.1f}):")
print(f"  y range: [{sections[mid]['y_input'].min():.1f}, {sections[mid]['y_input'].max():.1f}]")
print(f"  z range: [{sections[mid]['z_input'].min():.1f}, {sections[mid]['z_input'].max():.1f}]")

print(f"\nSection {len(sections)-2} (near stern, x={sections[-2]['x']:.1f}):")
print(f"  y range: [{sections[-2]['y_input'].min():.1f}, {sections[-2]['y_input'].max():.1f}]")
print(f"  z range: [{sections[-2]['z_input'].min():.1f}, {sections[-2]['z_input'].max():.1f}]")

print(f"\nSection {len(sections)-1} (stern, x={sections[-1]['x']:.1f}):")
print(f"  y range: [{sections[-1]['y_input'].min():.1f}, {sections[-1]['y_input'].max():.1f}]")
print(f"  z range: [{sections[-1]['z_input'].min():.1f}, {sections[-1]['z_input'].max():.1f}]")
print(f"  y_input: {sections[-1]['y_input']}")
print(f"  z_input: {sections[-1]['z_input']}")

# Resample all sections to common resolution
def resample_section(y_input, z_input, n_target):
    dy = np.diff(y_input)
    dz = np.diff(z_input)
    ds = np.sqrt(dy**2 + dz**2)
    s = np.zeros(len(y_input))
    s[1:] = np.cumsum(ds)
    s_total = s[-1]
    s_new = np.linspace(0, s_total, n_target)
    y_new = np.interp(s_new, s, y_input)
    z_new = np.interp(s_new, s, z_input)
    return y_new, z_new

N_CIRC = 20
resampled = []
for sec in sections:
    if sec['npts'] == N_CIRC:
        y_r, z_r = sec['y_input'].copy(), sec['z_input'].copy()
    else:
        y_r, z_r = resample_section(sec['y_input'], sec['z_input'], N_CIRC)
    resampled.append({'x': sec['x'], 'y': -y_r, 'z': z_r})  # negate y for +y=stb

# Build mesh
n_sec = len(resampled)
vertices = []
for i_sec in range(n_sec):
    s = resampled[i_sec]
    for j in range(N_CIRC):
        vertices.append([s['x'], s['y'][j], s['z'][j]])
vertices = np.array(vertices)

faces = []
for i_sec in range(n_sec - 1):
    for j in range(N_CIRC - 1):
        v00 = i_sec * N_CIRC + j
        v01 = i_sec * N_CIRC + j + 1
        v10 = (i_sec + 1) * N_CIRC + j
        v11 = (i_sec + 1) * N_CIRC + j + 1
        faces.append([v00, v10, v11, v01])
faces = np.array(faces)

print(f"\nMesh: {len(vertices)} vertices, {len(faces)} faces")
print(f"Bounds: x=[{vertices[:,0].min():.1f}, {vertices[:,0].max():.1f}]")
print(f"        y=[{vertices[:,1].min():.1f}, {vertices[:,1].max():.1f}]")
print(f"        z=[{vertices[:,2].min():.1f}, {vertices[:,2].max():.1f}]")

hull_mesh = Mesh(vertices=vertices, faces=faces, name="kvlcc2_hull")

# Check normals direction
midship_mask = np.abs(hull_mesh.faces_centers[:, 0]) < 10.0
midship_centers = hull_mesh.faces_centers[midship_mask]
midship_normals = hull_mesh.faces_normals[midship_mask]
stb_mask = midship_centers[:, 1] > 5.0
if stb_mask.any():
    avg_ny = np.mean(midship_normals[stb_mask, 1])
    print(f"\nNormal check (stb midship): avg ny = {avg_ny:.4f} (should be >0 for outward)")

# Check waterplane area
# The z=0 edge should give the waterplane
wp_z = vertices[:, 2]
at_wl = np.abs(wp_z) < 0.5
print(f"\nVertices near waterline (|z|<0.5): {np.sum(at_wl)}")

# Compute displaced volume (approximate from mesh)
hull_vol = hull_mesh.volume
print(f"Mesh volume: {hull_vol:.0f} m³")
expected_mass = 320437550.0
expected_vol = expected_mass / 1025.0
print(f"Expected displacement volume: {expected_vol:.0f} m³")
print(f"Volume ratio: {hull_vol/expected_vol:.3f}")

# Try building body and checking hydrostatics
lid = hull_mesh.generate_lid(z=-0.01)
body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name="kvlcc2")
body.center_of_mass = np.array([11.1, 0.0, -2.2])
body.mass = 320437550.0
body.rotation_center = body.center_of_mass
body.add_all_rigid_body_dofs()

hs = body.compute_hydrostatic_stiffness()
print(f"\nHydrostatic stiffness:")
for dof in ['Heave', 'Roll', 'Pitch']:
    try:
        val = float(hs.sel(influenced_dof=dof, radiating_dof=dof))
        print(f"  C[{dof},{dof}] = {val:.0f}")
    except:
        pass

# Quick single-frequency test
solver = cpt.BEMSolver()
lam = 200.0
k = 2*np.pi/lam
omega = np.sqrt(k*9.81)
print(f"\nSingle-frequency test: lambda={lam}m, omega={omega:.4f}")

rad_prob = cpt.RadiationProblem(body=body, radiating_dof='Heave',
                                 omega=omega, water_depth=np.inf, rho=1025.0, g=9.81)
try:
    rad_result = solver.solve(rad_prob)
    print(f"  A_heave = {rad_result.added_masses['Heave']:.0f}")
    print(f"  B_heave = {rad_result.radiation_dampings['Heave']:.0f}")
    print("  BEM solve SUCCESS")
except Exception as e:
    print(f"  BEM solve FAILED: {e}")
