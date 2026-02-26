#!/usr/bin/env python3
"""
Compare |H(theta)|^2 from Kochin vs |A(theta)|^2 from field points,
to find the frequency-dependent relation.
"""
import numpy as np
import capytaine as cpt
from capytaine.meshes.collections import CollectionOfMeshes
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

def kochin_full(result, theta_arr):
    """Kochin with hull + lid sources, with 1/(4pi) factor."""
    if body.lid_mesh is not None:
        full_mesh = CollectionOfMeshes([body.mesh, body.lid_mesh])
    else:
        full_mesh = body.mesh
    centers = full_mesh.faces_centers
    areas = full_mesh.faces_areas
    kk = result.wavenumber
    omega_bar = centers[:, 0:2] @ np.array([np.cos(theta_arr), np.sin(theta_arr)])
    cih = np.exp(kk * centers[:, 2])
    zs = (cih[:, None] * np.exp(-1j * kk * omega_bar) * areas[:, None])
    return (zs.T @ result.sources) / (4 * np.pi)

r = 5000.0  # far enough for convergence
field_pts = np.column_stack([
    r * np.cos(theta), r * np.sin(theta), np.zeros(N_THETA)])

wavelengths = [3, 5, 10, 22, 90]

print("Comparing |A(theta)|^2 from field points vs |H(theta)|^2 from Kochin")
print("Theory predicts: |A|^2 = (omega^2/g^2) * 2*pi*k * C^2 * |H|^2")
print("With C=2 and our H having 1/(4pi): actual |A|^2 / |H|^2 = (omega^2/g^2)*2*pi*k*4*(4pi)^2?")
print("Actually, let's just compute the ratio empirically.\n")

print(f"{'lam':>5} {'omega':>7} {'k':>7} "
      f"{'∫|H|²':>12} {'∫|A|²':>12} "
      f"{'|A|²/|H|²':>12} "
      f"{'B_dir':>10} {'B_kochin':>10} {'B_field':>10}")
print("-" * 100)

for lam in wavelengths:
    k = 2*np.pi / lam
    omega = np.sqrt(k * g)
    
    rad_prob = cpt.RadiationProblem(body=body, radiating_dof='Sway',
                                     omega=omega, water_depth=np.inf)
    rad_result = solver.solve(rad_prob, keep_details=True)
    B_direct = rad_result.radiation_dampings['Sway']
    
    # Kochin function
    H = kochin_full(rad_result, theta)
    H2_int = np.sum(np.abs(H)**2) * dtheta
    
    # Field-point amplitude
    phi_fp = solver.compute_potential(field_pts, rad_result)
    phase_factor = np.exp(-1j * k * r + 1j * np.pi/4)
    amplitude_factor = np.sqrt(np.pi * k * r / 2)
    A = phi_fp * (1j * omega / g) * amplitude_factor * phase_factor
    A2_int = np.sum(np.abs(A)**2) * dtheta
    
    # Ratio (should be constant if theory is correct)
    ratio_A2_H2 = A2_int / H2_int
    
    # B from Kochin (using C=2 formula that was verified)
    B_kochin = 4 * rho * np.pi * k / omega * H2_int
    
    # B from field points
    B_field = rho * g**2 / (2 * omega**3) * A2_int
    
    print(f"{lam:5.0f} {omega:7.3f} {k:7.4f} "
          f"{H2_int:12.6f} {A2_int:12.6f} "
          f"{ratio_A2_H2:12.4f} "
          f"{B_direct:10.1f} {B_kochin:10.1f} {B_field:10.1f}")

# The theory says:
# phi ~ -(ig/omega) * A(theta) * sqrt(2/(pi*k*r)) * exp(ikr - i*pi/4)
# And from Kochin with C=2:
# phi ~ -2 * sqrt(2*pi*k/r) * exp(kz) * H(theta) * exp(ikr + i*pi/4)  
#   (from Capytaine theory manual eq 82-86, with our H having 1/(4pi) absorbed)
#
# Wait, let me be more careful. Capytaine's far-field potential is:
# phi_j ~ -(2ig/omega) * sqrt(2*pi/k) * exp(kz) * H_j(theta) * 1/sqrt(r) * exp(ikr + i*pi/4)
# where H_j is the Kochin function WITHOUT the 1/(4pi) factor.
#
# Our kochin_full includes 1/(4pi), so H_ours = H_cap / (4pi)
# phi_j ~ -(2ig/omega) * sqrt(2*pi/k) * exp(kz) * (4pi)*H_ours(theta) * 1/sqrt(r) * exp(ikr + i*pi/4)
#
# Compare with our A(theta) formula:
# phi = -(ig/omega) * A * sqrt(2/(pi*k*r)) * exp(ikr - i*pi/4)
#
# So:
# A * sqrt(2/(pi*k)) * exp(-i*pi/4) = 2*sqrt(2*pi/k) * (4pi)*H_ours * exp(+i*pi/4)
# A = 2*(4pi) * sqrt(2*pi/k) / sqrt(2/(pi*k)) * exp(i*pi/2) * H_ours
# A = 8pi * sqrt(2*pi/k * pi*k/2) * exp(i*pi/2) * H_ours
# A = 8pi * pi * exp(i*pi/2) * H_ours
# A = 8*pi^2 * i * H_ours
#
# |A|^2 = 64*pi^4 * |H_ours|^2
#
# That's a constant! 64*pi^4 = 6234.2
print(f"\n\nTheory prediction: |A|²/|H|² = 64π⁴ = {64*np.pi**4:.1f}")
print("But the actual ratios above are NOT constant, so the theory is inconsistent.")
print("\nLet me check what the actual Capytaine asymptotic formula is...")

# Actually let me just check the ratio of |phi|*sqrt(r) / |H| at theta=pi/2
# This gives the proportionality constant directly.
print(f"\n\nDirect ratio: |phi|*sqrt(r) vs |H| at theta=pi/2")
print(f"{'lam':>5} {'|phi|*sqrt(r)':>16} {'|H|':>12} {'ratio':>12} {'ratio²':>12}")
print("-" * 65)

theta_idx = N_THETA // 4  # theta = pi/2

for lam in wavelengths:
    k = 2*np.pi / lam
    omega = np.sqrt(k * g)
    
    rad_prob = cpt.RadiationProblem(body=body, radiating_dof='Sway',
                                     omega=omega, water_depth=np.inf)
    rad_result = solver.solve(rad_prob, keep_details=True)
    
    # Kochin at theta=pi/2
    H_single = kochin_full(rad_result, np.array([np.pi/2]))[0]
    
    # Field point at r=5000, theta=pi/2
    pt = np.array([[0.0, r, 0.0]])
    phi = solver.compute_potential(pt, rad_result)[0]
    phi_scaled = np.abs(phi) * np.sqrt(r)
    
    ratio = phi_scaled / np.abs(H_single)
    
    # Theory: phi_scaled = C_factor * H, where C_factor may depend on k
    # From Green function: C_factor = (g/omega) * 2 * sqrt(2*pi*k) * (4*pi)
    C_theory = (g/omega) * 2 * np.sqrt(2*np.pi*k) * (4*np.pi)
    # Wait, that's with the 4pi factor for our H definition
    # Let me just compute it:
    # phi ~ -(ig/omega) * A * sqrt(2/(pi*k*r)) * exp(ikr-ipi/4) at z=0
    # |phi|*sqrt(r) = (g/omega) * |A| * sqrt(2/(pi*k))
    # 
    # If A = 8*pi^2 * i * H_ours, then:
    # |phi|*sqrt(r) = (g/omega) * 8*pi^2 * |H| * sqrt(2/(pi*k))
    C_pred = (g/omega) * 8 * np.pi**2 * np.sqrt(2/(np.pi*k))
    
    print(f"{lam:5.0f} {phi_scaled:16.6f} {np.abs(H_single):12.6f} "
          f"{ratio:12.4f} {ratio**2:12.4f} C_pred={C_pred:.4f}")
