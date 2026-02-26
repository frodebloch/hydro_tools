#!/usr/bin/env python3
"""
Debug the Kochin function computation. Compare:
1. Our custom compute_kochin_hull_only (uses first n_hull sources)
2. Capytaine's built-in compute_kochin (if it works)
3. A version without lid mesh at all

Also try computing on a body WITHOUT a lid to remove that complication.
"""

import numpy as np
import capytaine as cpt
import logging

cpt.set_logging(logging.WARNING)

R = 1.0
L = 20.0
rho = 1025.0
g = 9.81

mesh_res = (10, 40, 50)

def compute_kochin_hull_only(result, theta):
    """Our custom function: uses first n_hull sources."""
    k = result.wavenumber
    n_hull = result.body.mesh.nb_faces
    sources_hull = result.sources[:n_hull]
    centers = result.body.mesh.faces_centers
    areas = result.body.mesh.faces_areas
    omega_bar = centers[:, 0:2] @ np.array([np.cos(theta), np.sin(theta)])
    cih = np.exp(k * centers[:, 2])
    zs = cih[:, None] * np.exp(-1j * k * omega_bar) * areas[:, None]
    return (zs.T @ sources_hull) / (4 * np.pi)


def compute_kochin_all_sources(result, theta):
    """Use ALL sources (hull + lid) with their respective panel geometry."""
    k = result.wavenumber
    # Get the full mesh including lid
    # result.sources has n_hull + n_lid entries
    # The first n_hull correspond to hull faces, rest to lid faces
    n_hull = result.body.mesh.nb_faces
    n_total = len(result.sources)
    n_lid = n_total - n_hull

    # Hull contribution
    hull_centers = result.body.mesh.faces_centers
    hull_areas = result.body.mesh.faces_areas
    sources_hull = result.sources[:n_hull]

    omega_bar_h = hull_centers[:, 0:2] @ np.array([np.cos(theta), np.sin(theta)])
    cih_h = np.exp(k * hull_centers[:, 2])
    zs_h = cih_h[:, None] * np.exp(-1j * k * omega_bar_h) * hull_areas[:, None]
    H_hull = (zs_h.T @ sources_hull) / (4 * np.pi)

    # Lid contribution
    if n_lid > 0 and hasattr(result.body, 'lid_mesh') and result.body.lid_mesh is not None:
        lid_centers = result.body.lid_mesh.faces_centers
        lid_areas = result.body.lid_mesh.faces_areas
        sources_lid = result.sources[n_hull:]

        omega_bar_l = lid_centers[:, 0:2] @ np.array([np.cos(theta), np.sin(theta)])
        cih_l = np.exp(k * lid_centers[:, 2])
        zs_l = cih_l[:, None] * np.exp(-1j * k * omega_bar_l) * lid_areas[:, None]
        H_lid = (zs_l.T @ sources_lid) / (4 * np.pi)
    else:
        H_lid = np.zeros_like(H_hull)

    return H_hull, H_lid, H_hull + H_lid


# ============================================================
# Test 1: Body WITHOUT lid mesh (no irregular frequency issues, but clean Kochin)
# ============================================================
print("=" * 80)
print("TEST 1: Body WITHOUT lid mesh")
print("=" * 80)

mesh_full = cpt.mesh_horizontal_cylinder(
    length=L, radius=R, center=(0, 0, 0),
    resolution=mesh_res, name="nolid"
)
hull_mesh = mesh_full.immersed_part()
body_nolid = cpt.FloatingBody(mesh=hull_mesh, name="nolid")
body_nolid.add_all_rigid_body_dofs()
body_nolid.center_of_mass = np.array([0.0, 0.0, -4*R/(3*np.pi)])

print(f"Hull faces: {hull_mesh.nb_faces}")

solver = cpt.BEMSolver()
n_kochin = 720
theta = np.linspace(0, 2*np.pi, n_kochin, endpoint=False)
dtheta = 2 * np.pi / n_kochin

wavelengths = np.array([5, 10, 20, 40, 90])
k_values = 2 * np.pi / wavelengths
omega_values = np.sqrt(k_values * g)

for dof_name in ['Heave', 'Sway']:
    print(f"\n--- DOF: {dof_name} (no lid) ---")
    print(f"{'lambda':>8s} {'omega':>8s} {'B_direct':>12s} {'int|H|^2':>14s} "
          f"{'alpha':>14s} {'alpha/(k*w)':>14s}")
    print("-" * 80)

    for lam, k, omega in zip(wavelengths, k_values, omega_values):
        prob = cpt.RadiationProblem(
            body=body_nolid, radiating_dof=dof_name, omega=omega,
            water_depth=np.inf, rho=rho, g=g
        )
        result = solver.solve(prob)
        B_direct = result.radiation_dampings[dof_name]

        # No lid, so all sources are hull
        H = compute_kochin_hull_only(result, theta)
        H2_int = np.sum(np.abs(H)**2) * dtheta

        alpha = B_direct / (rho * H2_int) if abs(H2_int) > 1e-20 else float('nan')
        alpha_kw = alpha / (k * omega) if abs(k * omega) > 1e-20 else float('nan')

        print(f"{lam:8.1f} {omega:8.3f} {B_direct:12.1f} {H2_int:14.6e} "
              f"{alpha:14.6e} {alpha_kw:14.6f}")

# Also try Capytaine's built-in kochin on the no-lid body
print(f"\n--- Built-in Kochin (Heave, no lid) ---")
print(f"{'lambda':>8s} {'|H_custom|_max':>14s} {'|H_builtin|_max':>16s} {'ratio':>8s}")
print("-" * 60)

for lam, k, omega in zip(wavelengths, k_values, omega_values):
    prob = cpt.RadiationProblem(
        body=body_nolid, radiating_dof='Heave', omega=omega,
        water_depth=np.inf, rho=rho, g=g
    )
    result = solver.solve(prob)

    H_custom = compute_kochin_hull_only(result, theta)
    try:
        H_builtin = result.solver_result if hasattr(result, 'solver_result') else None
        # Try the proper way
        kochin_data = solver.compute_kochin(result, theta)
        H_builtin = kochin_data
        max_custom = np.max(np.abs(H_custom))
        max_builtin = np.max(np.abs(H_builtin))
        ratio = max_builtin / max_custom if max_custom > 1e-20 else float('nan')
        print(f"{lam:8.1f} {max_custom:14.6e} {max_builtin:16.6e} {ratio:8.4f}")
    except Exception as e:
        max_custom = np.max(np.abs(H_custom))
        print(f"{lam:8.1f} {max_custom:14.6e}  builtin failed: {e}")


# ============================================================
# Test 2: Body WITH lid mesh - compare source counts
# ============================================================
print()
print("=" * 80)
print("TEST 2: Body WITH lid mesh - source analysis")
print("=" * 80)

mesh_full2 = cpt.mesh_horizontal_cylinder(
    length=L, radius=R, center=(0, 0, 0),
    resolution=mesh_res, name="withlid"
)
hull_mesh2 = mesh_full2.immersed_part()
lid2 = hull_mesh2.generate_lid(z=-0.01)
body_lid = cpt.FloatingBody(mesh=hull_mesh2, lid_mesh=lid2, name="withlid")
body_lid.add_all_rigid_body_dofs()
body_lid.center_of_mass = np.array([0.0, 0.0, -4*R/(3*np.pi)])

print(f"Hull faces: {hull_mesh2.nb_faces}, Lid faces: {lid2.nb_faces}")

for dof_name in ['Heave', 'Sway']:
    print(f"\n--- DOF: {dof_name} (with lid) ---")
    print(f"{'lambda':>8s} {'n_src':>6s} {'n_hull':>6s} {'B_direct':>12s} "
          f"{'H2_hull':>14s} {'H2_all':>14s} {'alpha_hull':>14s} {'alpha_all':>14s}")
    print("-" * 100)

    for lam, k, omega in zip(wavelengths, k_values, omega_values):
        prob = cpt.RadiationProblem(
            body=body_lid, radiating_dof=dof_name, omega=omega,
            water_depth=np.inf, rho=rho, g=g
        )
        result = solver.solve(prob)
        B_direct = result.radiation_dampings[dof_name]

        n_src = len(result.sources)
        n_hull = result.body.mesh.nb_faces

        H_hull_only = compute_kochin_hull_only(result, theta)
        H_hull_part, H_lid_part, H_all = compute_kochin_all_sources(result, theta)

        H2_hull = np.sum(np.abs(H_hull_only)**2) * dtheta
        H2_all = np.sum(np.abs(H_all)**2) * dtheta

        alpha_hull = B_direct / (rho * H2_hull) if abs(H2_hull) > 1e-20 else float('nan')
        alpha_all = B_direct / (rho * H2_all) if abs(H2_all) > 1e-20 else float('nan')

        print(f"{lam:8.1f} {n_src:6d} {n_hull:6d} {B_direct:12.1f} "
              f"{H2_hull:14.6e} {H2_all:14.6e} {alpha_hull:14.6e} {alpha_all:14.6e}")

# ============================================================
# Test 3: Compare no-lid vs with-lid damping (are they close?)
# ============================================================
print()
print("=" * 80)
print("TEST 3: Damping comparison no-lid vs with-lid")
print("=" * 80)

for dof_name in ['Heave', 'Sway']:
    print(f"\n--- DOF: {dof_name} ---")
    print(f"{'lambda':>8s} {'B_nolid':>12s} {'B_lid':>12s} {'ratio':>8s}")
    print("-" * 44)

    for lam, k, omega in zip(wavelengths, k_values, omega_values):
        prob_nolid = cpt.RadiationProblem(
            body=body_nolid, radiating_dof=dof_name, omega=omega,
            water_depth=np.inf, rho=rho, g=g
        )
        prob_lid = cpt.RadiationProblem(
            body=body_lid, radiating_dof=dof_name, omega=omega,
            water_depth=np.inf, rho=rho, g=g
        )
        r_nolid = solver.solve(prob_nolid)
        r_lid = solver.solve(prob_lid)

        B_nolid = r_nolid.radiation_dampings[dof_name]
        B_lid = r_lid.radiation_dampings[dof_name]
        ratio = B_lid / B_nolid if abs(B_nolid) > 1e-12 else float('nan')
        print(f"{lam:8.1f} {B_nolid:12.1f} {B_lid:12.1f} {ratio:8.4f}")
