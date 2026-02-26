#!/usr/bin/env python3
"""
Debug: check convergence of |phi|*sqrt(r) vs r for different wavelengths.
If the asymptotic holds, |phi|*sqrt(r) should be constant for large r.
"""
import numpy as np
import capytaine as cpt
import logging

cpt.set_logging(logging.WARNING)

R = 1.0; L = 20.0; rho = 1025.0; g = 9.81
mesh_res = (10, 40, 50)

mesh_full = cpt.mesh_horizontal_cylinder(
    length=L, radius=R, center=(0, 0, 0), resolution=mesh_res, name="hull")
hull_mesh = mesh_full.immersed_part()
lid = hull_mesh.generate_lid(z=-0.01)
body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name="hull")
body.center_of_mass = np.array([0.0, 0.0, -4*R/(3*np.pi)])
body.rotation_center = body.center_of_mass
body.add_all_rigid_body_dofs()

solver = cpt.BEMSolver()

# Fixed angle: broadside (theta = pi/2)
theta_test = np.pi / 2

wavelengths = [3, 5, 10, 22, 90]
radii = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

for lam in wavelengths:
    k = 2*np.pi / lam
    omega = np.sqrt(k * g)
    
    rad_prob = cpt.RadiationProblem(body=body, radiating_dof='Sway',
                                     omega=omega, water_depth=np.inf)
    rad_result = solver.solve(rad_prob, keep_details=True)
    B_direct = rad_result.radiation_dampings['Sway']
    
    print(f"\nlambda={lam}m, omega={omega:.3f}, k={k:.4f}, B_direct={B_direct:.1f}")
    print(f"{'r':>7} {'kr':>8} {'|phi|':>14} {'|phi|*sqrt(r)':>16} {'|A|':>12} {'phase_A':>10}")
    print("-" * 75)
    
    for r in radii:
        pt = np.array([[r * np.cos(theta_test), r * np.sin(theta_test), 0.0]])
        phi = solver.compute_potential(pt, rad_result)[0]
        
        phi_scaled = np.abs(phi) * np.sqrt(r)
        
        # Extract A
        phase_factor = np.exp(-1j * k * r + 1j * np.pi/4)
        amplitude_factor = np.sqrt(np.pi * k * r / 2)
        A = phi * (1j * omega / g) * amplitude_factor * phase_factor
        
        print(f"{r:7.0f} {k*r:8.1f} {np.abs(phi):14.6e} {phi_scaled:16.6f} "
              f"{np.abs(A):12.6f} {np.angle(A):10.4f}")
