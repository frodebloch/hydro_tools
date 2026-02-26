#!/usr/bin/env python3
"""
Debug: verify damping at multiple wavelengths systematically.
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

# Use adaptive radius: r = max(200, 5*lambda) to ensure we're in far field
# but not so far that numerical precision suffers
wavelengths = np.array([3, 5, 8, 10, 13, 22, 45, 90])

print(f"{'lam':>5} {'omega':>7} {'k':>7} {'r':>6} {'kr':>7} "
      f"{'B_dir':>10} {'B_A':>10} {'ratio':>8}")
print("-" * 72)

for lam in wavelengths:
    k = 2*np.pi / lam
    omega = np.sqrt(k * g)
    
    # Choose radius: far enough for asymptotic, but not crazy large
    r = max(100, 3*lam)
    kr = k * r
    
    field_pts = np.column_stack([
        r * np.cos(theta),
        r * np.sin(theta),
        np.zeros(N_THETA)
    ])
    
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
    
    print(f"{lam:5.0f} {omega:7.3f} {k:7.4f} {r:6.0f} {kr:7.1f} "
          f"{B_direct:10.1f} {B_from_A:10.1f} {ratio:8.4f}")

# Also check: are the Phase 1 results in the main script wrong because
# the main script uses a FIXED r=200 for all wavelengths?
print(f"\n\nSame but with FIXED r=200 for all wavelengths:")
r = 200
field_pts = np.column_stack([
    r * np.cos(theta),
    r * np.sin(theta),
    np.zeros(N_THETA)
])

print(f"{'lam':>5} {'omega':>7} {'k':>7} {'kr':>7} "
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
    
    phase_factor = np.exp(-1j * k * r + 1j * np.pi/4)
    amplitude_factor = np.sqrt(np.pi * k * r / 2)
    A_rad = phi_fp * (1j * omega / g) * amplitude_factor * phase_factor
    
    A2_int = np.sum(np.abs(A_rad)**2) * dtheta
    B_from_A = rho * g**2 / (2 * omega**3) * A2_int
    ratio = B_direct / B_from_A
    
    print(f"{lam:5.0f} {omega:7.3f} {k:7.4f} {k*r:7.1f} "
          f"{B_direct:10.1f} {B_from_A:10.1f} {ratio:8.4f}")
