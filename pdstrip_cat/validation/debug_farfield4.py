#!/usr/bin/env python3
"""
Full damping verification using field-point ring at sufficiently large radius.
Use r=5000 for all wavelengths to ensure convergence.
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

N_THETA = 720
theta = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
dtheta = 2*np.pi / N_THETA

# Use large radius for all
r = 5000.0

field_pts = np.column_stack([
    r * np.cos(theta),
    r * np.sin(theta),
    np.zeros(N_THETA)
])

wavelengths = [3, 5, 8, 10, 13, 22, 45, 90]

print(f"Field-point ring at r={r}m, N_theta={N_THETA}")
print(f"\n{'lam':>5} {'omega':>7} {'k':>7} {'kr':>8} "
      f"{'B_dir':>10} {'B_A':>10} {'ratio':>8}")
print("-" * 65)

for lam in wavelengths:
    k = 2*np.pi / lam
    omega = np.sqrt(k * g)
    
    rad_prob = cpt.RadiationProblem(body=body, radiating_dof='Sway',
                                     omega=omega, water_depth=np.inf)
    rad_result = solver.solve(rad_prob, keep_details=True)
    B_direct = rad_result.radiation_dampings['Sway']
    
    phi_fp = solver.compute_potential(field_pts, rad_result)
    
    # Extract A(θ)
    phase_factor = np.exp(-1j * k * r + 1j * np.pi/4)
    amplitude_factor = np.sqrt(np.pi * k * r / 2)
    A_rad = phi_fp * (1j * omega / g) * amplitude_factor * phase_factor
    
    A2_int = np.sum(np.abs(A_rad)**2) * dtheta
    B_from_A = rho * g**2 / (2 * omega**3) * A2_int
    ratio = B_direct / B_from_A
    
    print(f"{lam:5.0f} {omega:7.3f} {k:7.4f} {k*r:8.0f} "
          f"{B_direct:10.1f} {B_from_A:10.1f} {ratio:8.4f}")
