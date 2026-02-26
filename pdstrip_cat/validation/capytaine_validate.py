#!/usr/bin/env python3
"""
Capytaine validation script for pdstrip catamaran implementation.

Creates a half-immersed semi-circular cylinder barge (R=1m, L=20m) and computes:
- Added mass and damping coefficients
- Wave excitation forces
For both single hull (monohull) and twin hulls (catamaran) at various spacings.

Uses Capytaine's built-in mesh_horizontal_cylinder, clipped at waterline,
with a lid mesh for irregular frequency removal.

Results saved to netCDF files for comparison with pdstrip.
"""

import numpy as np
import capytaine as cpt
import logging
import os
import sys
import json

cpt.set_logging(logging.WARNING)

# ============================================================
# Parameters
# ============================================================
R = 1.0         # cylinder radius [m]
L = 20.0        # barge length [m]
rho = 1025.0    # water density [kg/m^3]
g = 9.81        # gravity [m/s^2]

# Catamaran spacings: hulld/R = 2, 3, 5
hulld_ratios = [2, 3, 5]

# Mesh resolution for built-in cylinder
# resolution = (nr_endcap, ntheta, nx)
mesh_res = (10, 40, 50)

# Frequencies: match pdstrip's internal nu = omega^2 * R / g
# Select values that span the interesting range and align with pdstrip's 52 standard frequencies
nu_values = np.array([
    0.01, 0.02, 0.04, 0.06, 0.08,
    0.10, 0.15, 0.20, 0.25, 0.31,
    0.40, 0.50, 0.63, 0.80, 1.00,
    1.25, 1.55, 1.90, 2.40, 3.00,
    3.60, 4.50, 5.00,
])
omega_values = np.sqrt(nu_values * g / R)

# Wave directions (Capytaine convention: beta, radians)
# beta=0: waves propagate in +x; beta=pi/2: waves propagate in +y; beta=pi: waves in -x
#
# pdstrip convention mapping:
#   pdstrip mu=0° (following seas, waves in +x) → Capytaine beta=0
#   pdstrip mu=90° (from stb) → need to determine
#   pdstrip mu=180° (head seas, waves in -x) → Capytaine beta=pi
#
# For a symmetric body about y=0 (monohull), only |beta| matters for sway/heave/roll
# magnitudes. For catamaran symmetric about y=0, same applies.
# Use beta = pi/2 for beam seas comparison.
wave_directions = np.array([0, np.pi/2, np.pi])


# ============================================================
# Mesh creation
# ============================================================
def make_hull_body(R, L, y_offset=0.0, name="hull"):
    """
    Create a FloatingBody for a half-immersed horizontal cylinder.
    
    Uses Capytaine's built-in mesh_horizontal_cylinder, clipped at waterline,
    with a lid mesh for irregular frequency removal.
    
    Parameters
    ----------
    R : float - cylinder radius
    L : float - length along x
    y_offset : float - lateral offset of hull center
    name : str - body name
    
    Returns
    -------
    cpt.FloatingBody with all 6 rigid body DOFs
    """
    # Create full cylinder centered at (0, y_offset, 0)
    mesh_full = cpt.mesh_horizontal_cylinder(
        length=L, radius=R, center=(0, y_offset, 0),
        resolution=mesh_res,
        name=name
    )
    
    # Clip at waterline z=0
    hull_mesh = mesh_full.immersed_part()
    
    # Generate lid for irregular frequency removal
    lid = hull_mesh.generate_lid(z=-0.01)
    
    body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name=name)
    body.add_all_rigid_body_dofs()
    
    return body


def make_catamaran_body(R, L, hulld, name="catamaran"):
    """
    Create a catamaran FloatingBody with two hulls at y=±hulld.
    
    In Capytaine: y>0 is port, y<0 is starboard (matching pdstrip's internal convention).
    In pdstrip: hulld = distance from CL to hull center, stb hull at -hulld, port hull at +hulld.
    """
    body_stb = make_hull_body(R, L, y_offset=-hulld, name="stb_hull")
    body_port = make_hull_body(R, L, y_offset=+hulld, name="port_hull")
    
    body_cat = body_stb + body_port
    return body_cat


# ============================================================
# Solve
# ============================================================
def solve_hydrodynamics(body, omegas, betas, label=""):
    """Solve radiation and diffraction problems."""
    solver = cpt.BEMSolver()
    
    problems = []
    for omega in omegas:
        for dof in body.dofs:
            problems.append(cpt.RadiationProblem(
                body=body, radiating_dof=dof, omega=omega,
                water_depth=np.inf, rho=rho, g=g
            ))
        for beta in betas:
            problems.append(cpt.DiffractionProblem(
                body=body, wave_direction=beta, omega=omega,
                water_depth=np.inf, rho=rho, g=g
            ))
    
    print(f"  Solving {len(problems)} problems for {label}...")
    results = solver.solve_all(problems, progress_bar=True)
    dataset = cpt.assemble_dataset(results)
    
    return dataset


def save_dataset(ds, filepath):
    """Save dataset to numpy npz file, avoiding netCDF compatibility issues."""
    np_path = filepath.replace('.nc', '.npz')
    save_dict = {}
    for var in ds.data_vars:
        data = ds[var].values
        if np.iscomplexobj(data):
            save_dict[var + '_real'] = data.real
            save_dict[var + '_imag'] = data.imag
        else:
            save_dict[var] = data
    for coord in ds.coords:
        vals = ds.coords[coord].values
        if vals.ndim == 0:
            continue  # skip scalar coords
        try:
            save_dict['coord_' + coord] = np.array(vals, dtype=float)
        except (ValueError, TypeError):
            save_dict['coord_' + coord] = np.array([str(v) for v in vals])
    np.savez(np_path, **save_dict)
    print(f"  Saved to {np_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    output_dir = "/home/blofro/src/pdstrip_test/validation"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Frequency range: omega = {omega_values[0]:.3f} to {omega_values[-1]:.3f} rad/s")
    print(f"nu = omega^2*R/g range: {nu_values[0]} to {nu_values[-1]}")
    print(f"Wave directions (deg): {np.degrees(wave_directions)}")
    print()
    
    # ----------------------------------------------------------
    # 1. Monohull
    # ----------------------------------------------------------
    print("=" * 60)
    print("MONOHULL: Single semi-circular barge")
    print("=" * 60)
    
    body_mono = make_hull_body(R, L, y_offset=0.0, name="monohull")
    print(f"  Hull mesh: {body_mono.mesh.nb_faces} faces")
    print(f"  Volume: {body_mono.mesh.volume:.2f} m^3 (expected {np.pi/2*L:.2f})")
    
    ds_mono = solve_hydrodynamics(body_mono, omega_values, wave_directions, label="monohull")
    nc_path = os.path.join(output_dir, "capytaine_monohull.nc")
    save_dataset(ds_mono, nc_path)
    
    # Print summary
    print(f"\n  {'nu':>6s} {'omega':>6s} {'A22/L':>10s} {'B22/L':>10s} {'A33/L':>10s} {'B33/L':>10s}")
    for omega in omega_values[::3]:
        nu = omega**2 * R / g
        a22 = float(ds_mono['added_mass'].sel(radiating_dof='Sway', influenced_dof='Sway').sel(omega=omega, method='nearest'))
        b22 = float(ds_mono['radiation_damping'].sel(radiating_dof='Sway', influenced_dof='Sway').sel(omega=omega, method='nearest'))
        a33 = float(ds_mono['added_mass'].sel(radiating_dof='Heave', influenced_dof='Heave').sel(omega=omega, method='nearest'))
        b33 = float(ds_mono['radiation_damping'].sel(radiating_dof='Heave', influenced_dof='Heave').sel(omega=omega, method='nearest'))
        print(f"  {nu:6.3f} {omega:6.2f} {a22/L:10.1f} {b22/L:10.1f} {a33/L:10.1f} {b33/L:10.1f}")
    
    # ----------------------------------------------------------
    # 2. Catamaran at each spacing
    # ----------------------------------------------------------
    for hulld_ratio in hulld_ratios:
        hulld = hulld_ratio * R
        gap = 2 * hulld - 2 * R  # gap between inner waterlines
        
        print(f"\n{'=' * 60}")
        print(f"CATAMARAN: hulld/R={hulld_ratio}, hulld={hulld:.1f}m, gap={gap:.1f}m")
        print(f"{'=' * 60}")
        
        body_cat = make_catamaran_body(R, L, hulld)
        print(f"  Total mesh: {body_cat.mesh.nb_faces} faces")
        print(f"  DOFs: {list(body_cat.dofs.keys())}")
        
        ds_cat = solve_hydrodynamics(body_cat, omega_values, wave_directions,
                                      label=f"catamaran hulld/R={hulld_ratio}")
        
        fname = f"capytaine_cat_hulld{hulld_ratio}.nc"
        nc_path = os.path.join(output_dir, fname)
        save_dataset(ds_cat, nc_path)
        
        # Print combined sway/heave added mass (sum of all 4 hull-hull terms)
        print(f"\n  Combined catamaran coefficients / L:")
        print(f"  {'nu':>6s} {'omega':>6s} {'A22/L':>10s} {'B22/L':>10s} {'A33/L':>10s} {'B33/L':>10s}")
        for omega in omega_values[::3]:
            nu = omega**2 * R / g
            
            a22_total = 0.0
            b22_total = 0.0
            a33_total = 0.0
            b33_total = 0.0
            for rdof_prefix in ['stb_hull', 'port_hull']:
                for idof_prefix in ['stb_hull', 'port_hull']:
                    a22_total += float(ds_cat['added_mass'].sel(
                        radiating_dof=f'{rdof_prefix}__Sway',
                        influenced_dof=f'{idof_prefix}__Sway').sel(
                        omega=omega, method='nearest'))
                    b22_total += float(ds_cat['radiation_damping'].sel(
                        radiating_dof=f'{rdof_prefix}__Sway',
                        influenced_dof=f'{idof_prefix}__Sway').sel(
                        omega=omega, method='nearest'))
                    a33_total += float(ds_cat['added_mass'].sel(
                        radiating_dof=f'{rdof_prefix}__Heave',
                        influenced_dof=f'{idof_prefix}__Heave').sel(
                        omega=omega, method='nearest'))
                    b33_total += float(ds_cat['radiation_damping'].sel(
                        radiating_dof=f'{rdof_prefix}__Heave',
                        influenced_dof=f'{idof_prefix}__Heave').sel(
                        omega=omega, method='nearest'))
            
            print(f"  {nu:6.3f} {omega:6.2f} {a22_total/L:10.1f} {b22_total/L:10.1f} {a33_total/L:10.1f} {b33_total/L:10.1f}")
    
    print(f"\n{'=' * 60}")
    print("All Capytaine computations completed!")
    print(f"{'=' * 60}")
