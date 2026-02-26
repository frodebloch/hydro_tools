#!/usr/bin/env python3
"""
Clean test: compute near-field drift on FIXED semi-circular barge at one frequency.
Compare with far-field momentum argument.

For a FIXED body (no motion):
- Drift has only WL and velocity terms (no rotation)
- Far-field: F_y = (rho*g/k) * integral |a_s(theta)|^2 sin(theta) dtheta
  where a_s is the far-field scattered wave amplitude

Physical expectation:
- Wave propagates in +y direction (beta=pi/2)
- Body pushed in +y direction (toward lee) → F_y > 0
"""
import numpy as np
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_potential, airy_waves_velocity
import logging

cpt.set_logging(logging.WARNING)

R = 1.0; L = 20.0; rho = 1025.0; g = 9.81

# Build body
mesh_full = cpt.mesh_horizontal_cylinder(
    length=L, radius=R, center=(0, 0, 0),
    resolution=(10, 40, 50), name="hull")
hull_mesh = mesh_full.immersed_part()
lid = hull_mesh.generate_lid(z=-0.01)
body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name="hull")
body.center_of_mass = np.array([0, 0, -4*R/(3*np.pi)])
body.rotation_center = body.center_of_mass
body.add_all_rigid_body_dofs()

solver = cpt.BEMSolver()

# Hull info
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
        vlist = [face[0], face[1], face[2]] if face[0] == face[3] else list(face)
        for kk in range(len(vlist)):
            v1, v2 = vlist[kk], vlist[(kk+1)%len(vlist)]
            if v1 in wl_verts and v2 in wl_verts:
                e = (min(v1,v2), max(v1,v2))
                edges.add(e)
                face_for_edge[e] = fi
    edges = list(edges)
    n_e = len(edges)
    ec = np.zeros((n_e, 3)); el = np.zeros(n_e); en = np.zeros((n_e, 3))
    for i, (v1,v2) in enumerate(edges):
        p1, p2 = vertices[v1], vertices[v2]
        ec[i] = 0.5*(p1+p2)
        ev = p2 - p1; el[i] = np.linalg.norm(ev)
        t = ev/(el[i]+1e-30)
        n1 = np.array([t[1], -t[0], 0.0]); n2 = -n1
        fc = mesh.faces_centers[face_for_edge[(min(v1,v2), max(v1,v2))]]
        to_face = fc[:2] - ec[i,:2]
        en[i] = n1 if np.dot(n1[:2], to_face) < 0 else n2
    return edges, ec, el, en

edges, edge_centers, edge_lengths, edge_normals = extract_waterline_edges(hull_mesh)
wl_pts = edge_centers.copy(); wl_pts[:, 2] = -0.001

print(f"Hull: {n_hull} panels, Waterline: {len(edges)} edges")
print(f"Total WL length: {np.sum(edge_lengths):.2f}m (expected: ~2*L=40m)")

# === Single frequency, beam seas ===
beta = np.pi / 2
wavelengths = [3, 5, 10, 22, 55]

print(f"\n{'lam':>5} {'kR':>6} {'NF_wl':>10} {'NF_vel':>10} {'NF_tot':>10} "
      f"{'FF_Fy':>10} {'NF/FF':>8}")
print("-"*70)

for lam in wavelengths:
    k = 2*np.pi/lam
    omega = np.sqrt(k*g)
    kR = k*R
    
    # Solve diffraction
    diff_prob = cpt.DiffractionProblem(
        body=body, wave_direction=beta, omega=omega,
        water_depth=np.inf, rho=rho, g=g)
    diff_result = solver.solve(diff_prob, keep_details=True)
    
    # Total potential = incident + scattered (FIXED body, no radiation)
    inc_pot_hull = airy_waves_potential(hull_centers, diff_prob)
    scat_pot_hull = diff_result.potential[:n_hull]  # hull only (not lid)
    total_pot_hull = inc_pot_hull + scat_pot_hull
    
    inc_vel_hull = airy_waves_velocity(hull_centers, diff_prob)
    scat_vel_hull = solver.compute_velocity(hull_centers, diff_result)
    total_vel_hull = inc_vel_hull + scat_vel_hull
    
    inc_pot_wl = airy_waves_potential(wl_pts, diff_prob)
    scat_pot_wl = solver.compute_potential(wl_pts, diff_result)
    total_pot_wl = inc_pot_wl + scat_pot_wl
    
    # WL term
    eta_wl = (1j * omega / g) * total_pot_wl
    eta_sq = np.abs(eta_wl)**2
    F_wl_y = 0.25 * rho * g * np.sum(eta_sq * edge_normals[:, 1] * edge_lengths)
    
    # Velocity term
    vel_sq = np.sum(np.abs(total_vel_hull)**2, axis=1)
    F_vel_y = -0.25 * rho * np.sum(vel_sq * hull_normals[:, 1] * hull_areas)
    
    F_nf_y = F_wl_y + F_vel_y
    
    # Far-field: scattered potential at large radius
    N_THETA = 720
    R_FIELD = max(5000.0, 100*lam)
    theta_fp = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
    dtheta = 2*np.pi / N_THETA
    field_pts = np.column_stack([
        R_FIELD * np.cos(theta_fp),
        R_FIELD * np.sin(theta_fp),
        np.zeros(N_THETA)])
    
    phi_scat_fp = solver.compute_potential(field_pts, diff_result)
    a_scat = (1j * omega / g) * phi_scat_fp * np.sqrt(R_FIELD) * np.exp(-1j * k * R_FIELD)
    a2 = np.abs(a_scat)**2
    
    # Maruo: F_y = (rho*g/k) * int |a_s|^2 sin(theta) dtheta
    F_ff_y = (rho * g / k) * np.sum(a2 * np.sin(theta_fp)) * dtheta
    
    r = F_nf_y / F_ff_y if abs(F_ff_y) > 0.1 else float('nan')
    
    print(f"{lam:5.0f} {kR:6.3f} {F_wl_y:10.1f} {F_vel_y:10.1f} {F_nf_y:10.1f} "
          f"{F_ff_y:10.1f} {r:8.3f}")

# === Also check: is the scattered potential correctly separated? ===
print("\n\nSanity check: scattered wave at weather vs lee side")
print(f"{'lam':>5} {'|eta_inc|@W':>12} {'|eta_inc|@L':>12} "
      f"{'|eta_tot|@W':>12} {'|eta_tot|@L':>12} {'|eta_sca|@W':>12} {'|eta_sca|@L':>12}")
print("-"*90)

for lam in wavelengths:
    k = 2*np.pi/lam; omega = np.sqrt(k*g)
    
    diff_prob = cpt.DiffractionProblem(
        body=body, wave_direction=beta, omega=omega,
        water_depth=np.inf, rho=rho, g=g)
    diff_result = solver.solve(diff_prob, keep_details=True)
    
    # Sample points at waterline: weather (-y) and lee (+y) sides
    pts_w = np.array([[0.0, -R-0.01, -0.001]])  # just outside weather side
    pts_l = np.array([[0.0, R+0.01, -0.001]])    # just outside lee side
    
    inc_w = airy_waves_potential(pts_w, diff_prob)[0]
    inc_l = airy_waves_potential(pts_l, diff_prob)[0]
    scat_w = solver.compute_potential(pts_w, diff_result)[0]
    scat_l = solver.compute_potential(pts_l, diff_result)[0]
    
    eta_inc_w = abs((1j*omega/g) * inc_w)
    eta_inc_l = abs((1j*omega/g) * inc_l)
    eta_tot_w = abs((1j*omega/g) * (inc_w + scat_w))
    eta_tot_l = abs((1j*omega/g) * (inc_l + scat_l))
    eta_sca_w = abs((1j*omega/g) * scat_w)
    eta_sca_l = abs((1j*omega/g) * scat_l)
    
    print(f"{lam:5.0f} {eta_inc_w:12.4f} {eta_inc_l:12.4f} "
          f"{eta_tot_w:12.4f} {eta_tot_l:12.4f} {eta_sca_w:12.4f} {eta_sca_l:12.4f}")

# === Check the damping coefficient as validation of a(theta) extraction ===
print("\n\nFar-field amplitude validation: B_22 from damping vs from |a|^2 integral")
print(f"{'lam':>5} {'B_direct':>12} {'B_from_a':>12} {'ratio':>8}")
print("-"*40)

for lam in [3, 5, 10, 22]:
    k = 2*np.pi/lam; omega = np.sqrt(k*g)
    
    rad_prob = cpt.RadiationProblem(
        body=body, radiating_dof='Sway', omega=omega,
        water_depth=np.inf, rho=rho, g=g)
    rad_result = solver.solve(rad_prob, keep_details=True)
    B_direct = rad_result.radiation_dampings['Sway']
    
    N_THETA = 720; R_FIELD = max(5000.0, 100*lam)
    theta_fp = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
    dtheta = 2*np.pi / N_THETA
    fp = np.column_stack([R_FIELD*np.cos(theta_fp), R_FIELD*np.sin(theta_fp), np.zeros(N_THETA)])
    
    phi_fp = solver.compute_potential(fp, rad_result)
    a_rad = (1j * omega / g) * phi_fp * np.sqrt(R_FIELD) * np.exp(-1j * k * R_FIELD)
    a2_int = np.sum(np.abs(a_rad)**2) * dtheta
    
    # Standard relation: B = rho*g^2/(2*omega^3) * int |a|^2 dtheta
    B_from_a = rho * g**2 / (2*omega**3) * a2_int
    
    r = B_direct / B_from_a if abs(B_from_a) > 1e-6 else float('nan')
    print(f"{lam:5.0f} {B_direct:12.1f} {B_from_a:12.1f} {r:8.4f}")
