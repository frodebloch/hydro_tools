#!/usr/bin/env python3
"""
Debug: check what compute_potential returns at far-field points,
and verify the asymptotic amplitude extraction at different radii.
"""
import numpy as np
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_potential
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

# Test at a single wavelength
lam = 10.0
k = 2*np.pi / lam
omega = np.sqrt(k * g)
print(f"lambda={lam}m, omega={omega:.4f}, k={k:.4f}")

# Solve radiation for Sway
rad_prob = cpt.RadiationProblem(body=body, radiating_dof='Sway', 
                                 omega=omega, water_depth=np.inf)
rad_result = solver.solve(rad_prob, keep_details=True)
B_direct = rad_result.radiation_dampings['Sway']
print(f"B_direct(Sway) = {B_direct:.1f}")

# Also solve diffraction for beam seas
beta = np.pi/2
diff_prob = cpt.DiffractionProblem(body=body, wave_direction=beta,
                                    omega=omega, water_depth=np.inf)
diff_result = solver.solve(diff_prob, keep_details=True)

N_THETA = 720
theta = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
dtheta = 2*np.pi / N_THETA

# Try different radii
radii = [20, 50, 100, 200, 500]

print(f"\n{'r':>6}  {'B_from_A':>12}  {'ratio':>8}  {'|phi|_max':>12}  {'|A|_max':>12}")
print("-" * 65)

for r in radii:
    field_pts = np.column_stack([
        r * np.cos(theta),
        r * np.sin(theta),
        np.zeros(N_THETA)
    ])
    
    phi_fp = solver.compute_potential(field_pts, rad_result)
    
    # Extract A(θ)
    # φ_d = -(ig/ω) A(θ) √(2/(πkr)) exp(ikr - iπ/4)   at z=0
    # A(θ) = φ_d × (iω/g) × √(πkr/2) × exp(-ikr + iπ/4)
    phase_factor = np.exp(-1j * k * r + 1j * np.pi/4)
    amplitude_factor = np.sqrt(np.pi * k * r / 2)
    A_rad = phi_fp * (1j * omega / g) * amplitude_factor * phase_factor
    
    A2_int = np.sum(np.abs(A_rad)**2) * dtheta
    B_from_A = rho * g**2 / (2 * omega**3) * A2_int
    ratio = B_direct / B_from_A
    
    print(f"{r:6.0f}  {B_from_A:12.1f}  {ratio:8.4f}  {np.max(np.abs(phi_fp)):12.6f}  {np.max(np.abs(A_rad)):12.6f}")

# Now check individual theta points to see if the asymptotic form holds
print(f"\n\nDetailed check at theta=pi/2 (broadside) for radiation Sway:")
print(f"{'r':>6}  {'|phi|':>12}  {'|phi|*sqrt(r)':>14}  {'phase_of_phi':>14}  {'|A|':>12}")
theta_single = np.pi/2

for r in [20, 50, 100, 200, 500, 1000]:
    pt = np.array([[r * np.cos(theta_single), r * np.sin(theta_single), 0.0]])
    phi = solver.compute_potential(pt, rad_result)[0]
    
    # If asymptotic: |phi| ∝ 1/√r, so |phi|×√r should be constant
    phi_scaled = np.abs(phi) * np.sqrt(r)
    
    # Extract A
    phase_factor = np.exp(-1j * k * r + 1j * np.pi/4)
    amplitude_factor = np.sqrt(np.pi * k * r / 2)
    A = phi * (1j * omega / g) * amplitude_factor * phase_factor
    
    # Phase of phi after removing exp(ikr)/sqrt(r) behavior
    phi_phase = np.angle(phi * np.sqrt(r) * np.exp(-1j * k * r))
    
    print(f"{r:6.0f}  {np.abs(phi):12.6e}  {phi_scaled:14.6f}  {phi_phase:14.6f}  {np.abs(A):12.6f}")

# Check: does compute_potential include incident wave or just scattered?
print(f"\n\nCheck: diffraction potential at far field")
print("Does compute_potential include incident wave?")
for r in [50, 100, 200]:
    pt = np.array([[r * np.cos(np.pi/2), r * np.sin(np.pi/2), 0.0]])
    
    # Diffraction result potential
    phi_diff = solver.compute_potential(pt, diff_result)[0]
    
    # Incident potential at same point
    phi_inc = airy_waves_potential(pt, diff_prob)[0]
    
    # If compute_potential returns ONLY scattered part, phi_diff should NOT
    # include phi_inc. If it returns total, phi_diff ≈ phi_inc at far field.
    
    print(f"  r={r}: |phi_diff|={np.abs(phi_diff):.6e}, |phi_inc|={np.abs(phi_inc):.6e}, "
          f"|phi_diff/phi_inc|={np.abs(phi_diff/phi_inc):.4f}, "
          f"phase_diff={np.angle(phi_diff):.4f}, phase_inc={np.angle(phi_inc):.4f}")
