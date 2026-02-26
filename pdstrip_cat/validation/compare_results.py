#!/usr/bin/env python3
"""
Compare pdstrip vs Capytaine results for validation.
Reads pdstrip sectionresults and Capytaine .npz files.

pdstrip outputs section-level 2D hydrodynamic coefficients.
Capytaine outputs 3D total ship coefficients.
For a prismatic hull: Capytaine_total / L ≈ pdstrip_section (per unit length).

Convention mapping:
  pdstrip DOF: (1)=sway, (2)=heave, (3)=roll
  Capytaine DOF: Surge, Sway, Heave, Roll, Pitch, Yaw
  
  pdstrip stores omega^2 * addedm where addedm = a + b/(i*omega)
  => Re(omega^2 * addedm) = omega^2 * a
  => Im(omega^2 * addedm) = -omega * b
  => a = Re(.) / omega^2, b = -Im(.) / omega

  Capytaine stores added_mass (a) and radiation_damping (b) directly.
  
  For the prismatic barge, all 5 pdstrip sections are identical.
  Use section_idx=2 (middle section at x=0) for comparison.
"""

import numpy as np
import sys
import os

sys.path.insert(0, '/home/blofro/src/pdstrip_test/validation')
from parse_sectionresults import parse_sectionresults_v2

R = 1.0
L = 20.0
rho = 1025.0
g = 9.81

base = '/home/blofro/src/pdstrip_test/validation'


def extract_pdstrip(run_dir, section_idx=2):
    """Extract added mass and damping from pdstrip sectionresults."""
    sr_file = os.path.join(run_dir, 'sectionresults')
    data = parse_sectionresults_v2(sr_file, section_idx=section_idx)
    if data is None:
        raise RuntimeError(f"Failed to parse {sr_file}")
    
    nfre = data['nfre']
    omega = data['omega']
    nu = omega**2 * R / g
    
    # Extract diagonal added mass and damping
    a22 = np.zeros(nfre)
    b22 = np.zeros(nfre)
    a33 = np.zeros(nfre)
    b33 = np.zeros(nfre)
    a44 = np.zeros(nfre)
    b44 = np.zeros(nfre)
    
    for i in range(nfre):
        am = data['addedm'][i]
        w = omega[i]
        a22[i] = am[0, 0].real / w**2
        b22[i] = -am[0, 0].imag / w
        a33[i] = am[1, 1].real / w**2
        b33[i] = -am[1, 1].imag / w
        a44[i] = am[2, 2].real / w**2
        b44[i] = -am[2, 2].imag / w
    
    # Excitation forces at beam seas (imu=2, mu=90 deg)
    exc_sway = np.zeros(nfre, dtype=complex)
    exc_heave = np.zeros(nfre, dtype=complex)
    exc_roll = np.zeros(nfre, dtype=complex)
    
    imu = 2  # +90 degrees
    for i in range(nfre):
        exc = data['diff'][i][:, imu] + data['frkr'][i][:, imu]
        exc_sway[i] = exc[0]
        exc_heave[i] = exc[1]
        exc_roll[i] = exc[2]
    
    return {
        'omega': omega, 'nu': nu,
        'a22': a22, 'b22': b22,
        'a33': a33, 'b33': b33,
        'a44': a44, 'b44': b44,
        'exc_sway': exc_sway, 'exc_heave': exc_heave, 'exc_roll': exc_roll,
    }


def extract_capytaine_mono(npz_file):
    """Extract monohull Capytaine results from .npz file."""
    import xarray as xr
    import capytaine as cpt
    
    # Re-solve from npz isn't practical. Instead, we use the xarray dataset
    # that was kept in memory. Let's re-run the capytaine solver for just extraction.
    # Actually, better to just parse the npz with knowledge of the array layout.
    
    # Hmm, the npz doesn't preserve the xarray structure well enough.
    # Let me instead run capytaine inline here for the monohull extraction.
    # OR: load dataset from a pickle.
    
    # Simplest: re-run the Capytaine solve. It only takes ~1 minute for monohull.
    pass


def load_capytaine_xarray(label, omega_values, wave_directions):
    """Solve Capytaine and return xarray dataset directly."""
    import capytaine as cpt
    import logging
    cpt.set_logging(logging.WARNING)
    
    mesh_res = (10, 40, 50)
    solver = cpt.BEMSolver()
    
    if label == 'mono':
        mesh_full = cpt.mesh_horizontal_cylinder(
            length=L, radius=R, center=(0, 0, 0),
            resolution=mesh_res, name='barge')
        hull_mesh = mesh_full.immersed_part()
        lid = hull_mesh.generate_lid(z=-0.01)
        body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name='barge')
        body.add_all_rigid_body_dofs()
    else:
        # Catamaran: label = 'cat2', 'cat3', 'cat5'
        hulld = int(label[3:]) * R
        
        mesh_stb = cpt.mesh_horizontal_cylinder(
            length=L, radius=R, center=(0, -hulld, 0),
            resolution=mesh_res, name='stb_hull')
        hull_stb = mesh_stb.immersed_part()
        lid_stb = hull_stb.generate_lid(z=-0.01)
        body_stb = cpt.FloatingBody(mesh=hull_stb, lid_mesh=lid_stb, name='stb_hull')
        body_stb.add_all_rigid_body_dofs()
        
        mesh_port = cpt.mesh_horizontal_cylinder(
            length=L, radius=R, center=(0, hulld, 0),
            resolution=mesh_res, name='port_hull')
        hull_port = mesh_port.immersed_part()
        lid_port = hull_port.generate_lid(z=-0.01)
        body_port = cpt.FloatingBody(mesh=hull_port, lid_mesh=lid_port, name='port_hull')
        body_port.add_all_rigid_body_dofs()
        
        body = body_stb + body_port
    
    problems = []
    for omega in omega_values:
        for dof in body.dofs:
            problems.append(cpt.RadiationProblem(
                body=body, radiating_dof=dof, omega=omega,
                water_depth=np.inf, rho=rho, g=g))
        for beta in wave_directions:
            problems.append(cpt.DiffractionProblem(
                body=body, wave_direction=beta, omega=omega,
                water_depth=np.inf, rho=rho, g=g))
    
    print(f"  Solving {len(problems)} Capytaine problems for {label}...")
    results = solver.solve_all(problems, progress_bar=True)
    dataset = cpt.assemble_dataset(results)
    return dataset


def extract_capytaine_from_ds(ds, label, omega_values):
    """Extract added mass/damping from a Capytaine xarray dataset."""
    nfre = len(omega_values)
    nu = omega_values**2 * R / g
    
    a22 = np.zeros(nfre)
    b22 = np.zeros(nfre)
    a33 = np.zeros(nfre)
    b33 = np.zeros(nfre)
    
    if label == 'mono':
        for i, w in enumerate(omega_values):
            a22[i] = float(ds['added_mass'].sel(
                radiating_dof='Sway', influenced_dof='Sway').sel(omega=w, method='nearest'))
            b22[i] = float(ds['radiation_damping'].sel(
                radiating_dof='Sway', influenced_dof='Sway').sel(omega=w, method='nearest'))
            a33[i] = float(ds['added_mass'].sel(
                radiating_dof='Heave', influenced_dof='Heave').sel(omega=w, method='nearest'))
            b33[i] = float(ds['radiation_damping'].sel(
                radiating_dof='Heave', influenced_dof='Heave').sel(omega=w, method='nearest'))
    else:
        # Catamaran: sum all 4 hull-hull cross terms
        for i, w in enumerate(omega_values):
            for rdof_pfx in ['stb_hull', 'port_hull']:
                for idof_pfx in ['stb_hull', 'port_hull']:
                    a22[i] += float(ds['added_mass'].sel(
                        radiating_dof=f'{rdof_pfx}__Sway',
                        influenced_dof=f'{idof_pfx}__Sway').sel(omega=w, method='nearest'))
                    b22[i] += float(ds['radiation_damping'].sel(
                        radiating_dof=f'{rdof_pfx}__Sway',
                        influenced_dof=f'{idof_pfx}__Sway').sel(omega=w, method='nearest'))
                    a33[i] += float(ds['added_mass'].sel(
                        radiating_dof=f'{rdof_pfx}__Heave',
                        influenced_dof=f'{idof_pfx}__Heave').sel(omega=w, method='nearest'))
                    b33[i] += float(ds['radiation_damping'].sel(
                        radiating_dof=f'{rdof_pfx}__Heave',
                        influenced_dof=f'{idof_pfx}__Heave').sel(omega=w, method='nearest'))
    
    return {
        'omega': omega_values, 'nu': nu,
        'a22': a22, 'b22': b22,
        'a33': a33, 'b33': b33,
    }


# ============================================================
# Main comparison
# ============================================================
if __name__ == "__main__":
    # Frequency range for Capytaine (subset that avoids mesh resolution issues)
    nu_capy = np.array([
        0.01, 0.02, 0.04, 0.06, 0.08,
        0.10, 0.15, 0.20, 0.25, 0.31,
        0.40, 0.50, 0.63, 0.80, 1.00,
        1.25, 1.55, 1.90, 2.40, 3.00,
        3.60, 4.50, 5.00,
    ])
    omega_capy = np.sqrt(nu_capy * g / R)
    wave_dirs_capy = np.array([0, np.pi/2, np.pi])
    
    configs = [
        ('mono', os.path.join(base, 'run_mono')),
        ('cat2', os.path.join(base, 'run_cat2')),
        ('cat3', os.path.join(base, 'run_cat3')),
        ('cat5', os.path.join(base, 'run_cat5')),
    ]
    
    all_results = {}
    
    for label, run_dir in configs:
        print(f"\n{'='*70}")
        print(f"Configuration: {label}")
        print(f"{'='*70}")
        
        # Extract pdstrip
        pd = extract_pdstrip(run_dir)
        
        # Run/load Capytaine
        ds = load_capytaine_xarray(label, omega_capy, wave_dirs_capy)
        cy = extract_capytaine_from_ds(ds, label, omega_capy)
        
        all_results[label] = {'pdstrip': pd, 'capytaine': cy}
        
        # Normalize Capytaine to per-unit-length for comparison
        cy_a22_L = cy['a22'] / L
        cy_b22_L = cy['b22'] / L
        cy_a33_L = cy['a33'] / L
        cy_b33_L = cy['b33'] / L
        
        # Print comparison table for sway added mass
        print(f"\n  Sway added mass (a22) per unit length:")
        print(f"  {'nu':>8s} {'pdstrip':>12s} {'capy/L':>12s} {'ratio':>8s}")
        for i in range(0, len(omega_capy), 3):
            nu = nu_capy[i]
            # Find closest pdstrip frequency
            j = np.argmin(np.abs(pd['nu'] - nu))
            pd_val = pd['a22'][j]
            cy_val = cy_a22_L[i]
            ratio = cy_val / pd_val if abs(pd_val) > 1 else float('nan')
            print(f"  {nu:8.3f} {pd_val:12.1f} {cy_val:12.1f} {ratio:8.3f}")
        
        # Print comparison table for heave added mass
        print(f"\n  Heave added mass (a33) per unit length:")
        print(f"  {'nu':>8s} {'pdstrip':>12s} {'capy/L':>12s} {'ratio':>8s}")
        for i in range(0, len(omega_capy), 3):
            nu = nu_capy[i]
            j = np.argmin(np.abs(pd['nu'] - nu))
            pd_val = pd['a33'][j]
            cy_val = cy_a33_L[i]
            ratio = cy_val / pd_val if abs(pd_val) > 1 else float('nan')
            print(f"  {nu:8.3f} {pd_val:12.1f} {cy_val:12.1f} {ratio:8.3f}")
        
        # Print comparison table for sway damping
        print(f"\n  Sway damping (b22) per unit length:")
        print(f"  {'nu':>8s} {'pdstrip':>12s} {'capy/L':>12s} {'ratio':>8s}")
        for i in range(0, len(omega_capy), 3):
            nu = nu_capy[i]
            j = np.argmin(np.abs(pd['nu'] - nu))
            pd_val = pd['b22'][j]
            cy_val = cy_b22_L[i]
            ratio = cy_val / pd_val if abs(pd_val) > 10 else float('nan')
            print(f"  {nu:8.3f} {pd_val:12.1f} {cy_val:12.1f} {ratio:8.3f}")
        
        # Print comparison table for heave damping
        print(f"\n  Heave damping (b33) per unit length:")
        print(f"  {'nu':>8s} {'pdstrip':>12s} {'capy/L':>12s} {'ratio':>8s}")
        for i in range(0, len(omega_capy), 3):
            nu = nu_capy[i]
            j = np.argmin(np.abs(pd['nu'] - nu))
            pd_val = pd['b33'][j]
            cy_val = cy_b33_L[i]
            ratio = cy_val / pd_val if abs(pd_val) > 10 else float('nan')
            print(f"  {nu:8.3f} {pd_val:12.1f} {cy_val:12.1f} {ratio:8.3f}")
    
    # ================================================================
    # Summary: catamaran/monohull ratios compared between pdstrip and Capytaine
    # ================================================================
    print(f"\n\n{'='*70}")
    print("CATAMARAN / MONOHULL RATIOS")
    print(f"{'='*70}")
    
    pd_mono = all_results['mono']['pdstrip']
    cy_mono = all_results['mono']['capytaine']
    
    for cat_label in ['cat2', 'cat3', 'cat5']:
        hulld = int(cat_label[3:])
        print(f"\n--- hulld/R = {hulld} ---")
        
        pd_cat = all_results[cat_label]['pdstrip']
        cy_cat = all_results[cat_label]['capytaine']
        
        print(f"\n  Sway added mass ratio (catamaran / monohull):")
        print(f"  {'nu':>8s} {'pd_ratio':>10s} {'cy_ratio':>10s} {'diff%':>8s}")
        for i in range(0, len(omega_capy), 3):
            nu = nu_capy[i]
            # pdstrip
            j_mono = np.argmin(np.abs(pd_mono['nu'] - nu))
            j_cat = np.argmin(np.abs(pd_cat['nu'] - nu))
            if abs(pd_mono['a22'][j_mono]) > 1:
                pd_ratio = pd_cat['a22'][j_cat] / pd_mono['a22'][j_mono]
            else:
                pd_ratio = float('nan')
            # capytaine
            if abs(cy_mono['a22'][i]) > 1:
                cy_ratio = cy_cat['a22'][i] / cy_mono['a22'][i]
            else:
                cy_ratio = float('nan')
            diff = (pd_ratio - cy_ratio) / cy_ratio * 100 if not np.isnan(cy_ratio) and abs(cy_ratio) > 0.01 else float('nan')
            print(f"  {nu:8.3f} {pd_ratio:10.3f} {cy_ratio:10.3f} {diff:8.1f}")
        
        print(f"\n  Heave added mass ratio (catamaran / monohull):")
        print(f"  {'nu':>8s} {'pd_ratio':>10s} {'cy_ratio':>10s} {'diff%':>8s}")
        for i in range(0, len(omega_capy), 3):
            nu = nu_capy[i]
            j_mono = np.argmin(np.abs(pd_mono['nu'] - nu))
            j_cat = np.argmin(np.abs(pd_cat['nu'] - nu))
            if abs(pd_mono['a33'][j_mono]) > 1:
                pd_ratio = pd_cat['a33'][j_cat] / pd_mono['a33'][j_mono]
            else:
                pd_ratio = float('nan')
            if abs(cy_mono['a33'][i]) > 1:
                cy_ratio = cy_cat['a33'][i] / cy_mono['a33'][i]
            else:
                cy_ratio = float('nan')
            diff = (pd_ratio - cy_ratio) / cy_ratio * 100 if not np.isnan(cy_ratio) and abs(cy_ratio) > 0.01 else float('nan')
            print(f"  {nu:8.3f} {pd_ratio:10.3f} {cy_ratio:10.3f} {diff:8.1f}")
    
    print(f"\nComparison complete.")
