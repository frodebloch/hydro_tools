#!/usr/bin/env python3
"""
Determine the H(θ) ↔ a(θ) relation using the Haskind identity.

Haskind: F_exc,j = -(ρgω/2) H_j(β)   (Newman convention, for unit amplitude incident wave)
   or equivalently: H_j(β) = -2 F_exc,j / (ρgω)

Newman drift force (1967):
   F_y = (ρg/2) sinβ Im[H(β)]  +  (ρgk)/(4π) ∫|H(θ)|² sinθ dθ

We know a(θ) from field points and F_exc from Capytaine.
From Haskind, we get H(β). From the field points, we get a(β).
The ratio H(β)/a(β) gives us the phase+magnitude relation.
Then we can convert Newman's formula to terms of a(θ).

ALSO: Newman's damping: B_jj = (ρk)/(4π) ∫|H_j|² dθ
Our damping: B_jj = ρg/(2ωk) ∫|a_j|² dθ
Setting equal: |H|² × k/(4π) = |a|² × g/(2ωk)
→ |H|² = 2πg/(ωk²) × |a|² = 2πω/(k³) × |a|²

So: H(θ) = a(θ) × e^{iψ} × √(2πω/k³)  where ψ is the unknown phase.

We find ψ from: H(β) = a(β) × e^{iψ} × √(2πω/k³)
and: H(β) = -2 F_exc,sway / (ρgω)
"""

import numpy as np
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_potential, froude_krylov_force
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

# Field-point ring
N_THETA = 720
theta_fp = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
dtheta = 2*np.pi / N_THETA
R_FIELD = 5000.0
field_pts = np.column_stack([
    R_FIELD * np.cos(theta_fp),
    R_FIELD * np.sin(theta_fp),
    np.zeros(N_THETA)
])

beta = np.pi / 2
wavelengths = [3, 5, 10, 22, 55, 90]

print("Haskind relation: determining H(β)/a(β) phase")
print("="*100)
print(f"{'lam':>5} {'|a(β)|':>10} {'arg_a':>8} {'|H_hask|':>10} {'arg_H':>8} "
      f"{'|H|/|a|':>10} {'√(2πω/k³)':>12} {'|ratio|':>8} {'ψ=arg(H/a)':>12}")
print("-"*100)

for lam in wavelengths:
    k = 2*np.pi / lam
    omega = np.sqrt(k*g)
    
    # Radiation solve for Sway
    rad_prob = cpt.RadiationProblem(body=body, radiating_dof='Sway', omega=omega, water_depth=np.inf)
    rad_result = solver.solve(rad_prob, keep_details=True)
    
    # Diffraction solve
    diff_prob = cpt.DiffractionProblem(body=body, wave_direction=beta, omega=omega, water_depth=np.inf)
    diff_result = solver.solve(diff_prob, keep_details=True)
    
    # F_exc from Capytaine (diffraction + FK)
    FK = froude_krylov_force(diff_prob)
    F_exc_sway = diff_result.forces['Sway'] + FK['Sway']
    
    # H(β) from Haskind
    H_haskind = -2 * F_exc_sway / (rho * g * omega)
    
    # a(β) from field points (radiation solution for sway)
    phi_rad_fp = solver.compute_potential(field_pts, rad_result)
    a_rad = (1j * omega / g) * phi_rad_fp * np.sqrt(R_FIELD) * np.exp(-1j * k * R_FIELD)
    idx_beta = np.argmin(np.abs(theta_fp - beta))
    a_at_beta = a_rad[idx_beta]
    
    # Expected magnitude ratio
    mag_ratio_expected = np.sqrt(2 * np.pi * omega / k**3)
    
    # Actual ratio
    ratio = H_haskind / a_at_beta
    
    print(f"{lam:5.0f} {abs(a_at_beta):10.4f} {np.degrees(np.angle(a_at_beta)):8.1f}° "
          f"{abs(H_haskind):10.4f} {np.degrees(np.angle(H_haskind)):8.1f}° "
          f"{abs(ratio):10.4f} {mag_ratio_expected:12.4f} "
          f"{abs(ratio)/mag_ratio_expected:8.4f} "
          f"{np.degrees(np.angle(ratio)):12.1f}°")

# Now use DIFFRACTION a(β) (not radiation) for the drift force
print("\n\nDiffraction a(β) from field points:")
print("="*100)
print(f"{'lam':>5} {'|a_d(β)|':>10} {'arg_a_d':>8} {'|H_d|':>10} {'arg_H_d':>8} "
      f"{'H_d_from_a':>12} {'H_d_hask':>12}")
print("-"*100)

for lam in wavelengths:
    k = 2*np.pi / lam
    omega = np.sqrt(k*g)
    
    # Diffraction solve
    diff_prob = cpt.DiffractionProblem(body=body, wave_direction=beta, omega=omega, water_depth=np.inf)
    diff_result = solver.solve(diff_prob, keep_details=True)
    
    # a_d(β) from field points
    phi_scat_fp = solver.compute_potential(field_pts, diff_result)
    a_scat = (1j * omega / g) * phi_scat_fp * np.sqrt(R_FIELD) * np.exp(-1j * k * R_FIELD)
    idx_beta = np.argmin(np.abs(theta_fp - beta))
    a_d_beta = a_scat[idx_beta]
    
    # F_exc for ALL DOFs
    FK = froude_krylov_force(diff_prob)
    print(f"\n  λ={lam}m:")
    for dof in ['Sway', 'Heave', 'Roll']:
        F_exc = diff_result.forces[dof] + FK[dof]
        H_hask = -2 * F_exc / (rho * g * omega)
        
        # Get radiation a_j(β)
        rad_prob = cpt.RadiationProblem(body=body, radiating_dof=dof, omega=omega, water_depth=np.inf)
        rad_result = solver.solve(rad_prob, keep_details=True)
        phi_rad = solver.compute_potential(field_pts, rad_result)
        a_j_rad = (1j * omega / g) * phi_rad * np.sqrt(R_FIELD) * np.exp(-1j * k * R_FIELD)
        a_j_beta = a_j_rad[idx_beta]
        
        ratio = H_hask / a_j_beta if abs(a_j_beta) > 1e-12 else float('nan')
        
        # Check B also
        B_direct = rad_result.radiation_dampings[dof]
        a2_int = np.sum(np.abs(a_j_rad)**2) * dtheta
        B_from_a = rho * g**2 / (2 * omega**3) * a2_int
        
        print(f"    {dof:>8s}: |a_j(β)|={abs(a_j_beta):.4f}  arg={np.degrees(np.angle(a_j_beta)):7.1f}°  "
              f"|H_j|={abs(H_hask):.4f}  arg={np.degrees(np.angle(H_hask)):7.1f}°  "
              f"|H/a|={abs(ratio):.4f}  arg(H/a)={np.degrees(np.angle(ratio)):7.1f}°  "
              f"√(2πω/k³)={np.sqrt(2*np.pi*omega/k**3):.4f}  "
              f"B_dir/B_a={B_direct/B_from_a:.4f}")
