#!/usr/bin/env python3
"""
Verify the Kochin function normalization in Capytaine by comparing:
  B_direct = damping from BEM (direct result)
  B_kochin = (rho * alpha * k / omega) * integral |H(theta)|^2 dtheta

where alpha is the unknown normalization factor we want to determine.

The standard textbook relation (e.g., Newman, Faltinsen) states:
  B_jj = (rho * k / (2*pi)) * integral_0^2pi |H_j(theta)|^2 dtheta   (deep water, 2D)

But Capytaine's H(theta) may differ by a constant factor from the textbook H.
If Capytaine's H = H_textbook / C, then:
  B_jj = (rho * k * C^2) / (2*pi) * integral |H_cap|^2 dtheta

We want to find C^2.

Also verify using far-field potential computation if available.
"""

import numpy as np
import capytaine as cpt
import logging

cpt.set_logging(logging.WARNING)

R = 1.0
L = 20.0
rho = 1025.0
g = 9.81

# Create the same body
mesh_res = (10, 40, 50)

def make_hull_body(R, L, name="hull"):
    mesh_full = cpt.mesh_horizontal_cylinder(
        length=L, radius=R, center=(0, 0, 0),
        resolution=mesh_res, name=name
    )
    hull_mesh = mesh_full.immersed_part()
    lid = hull_mesh.generate_lid(z=-0.01)
    body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name=name)
    volume = np.pi * R**2 / 2 * L
    mass_val = rho * volume
    zcg = -4 * R / (3 * np.pi)
    body.center_of_mass = np.array([0.0, 0.0, zcg])
    body.mass = mass_val
    body.rotation_center = body.center_of_mass
    body.add_all_rigid_body_dofs()
    return body

def compute_kochin_hull_only(result, theta):
    """Compute Kochin function using only hull panels (not lid panels)."""
    k = result.wavenumber
    n_hull = result.body.mesh.nb_faces
    sources_hull = result.sources[:n_hull]
    centers = result.body.mesh.faces_centers
    areas = result.body.mesh.faces_areas
    omega_bar = centers[:, 0:2] @ np.array([np.cos(theta), np.sin(theta)])
    cih = np.exp(k * centers[:, 2])  # deep water
    zs = cih[:, None] * np.exp(-1j * k * omega_bar) * areas[:, None]
    return (zs.T @ sources_hull) / (4 * np.pi)


body = make_hull_body(R, L)
solver = cpt.BEMSolver()

n_kochin = 720
theta = np.linspace(0, 2*np.pi, n_kochin, endpoint=False)
dtheta = 2 * np.pi / n_kochin

# Test wavelengths
wavelengths = np.array([5, 10, 20, 40, 90])
k_values = 2 * np.pi / wavelengths
omega_values = np.sqrt(k_values * g)

print("=" * 80)
print("KOCHIN FUNCTION NORMALIZATION VERIFICATION")
print("=" * 80)
print()

# Test with heave (index 2) since it has the strongest signal
test_dofs = ['Heave', 'Sway', 'Surge']

for dof_name in test_dofs:
    print(f"\n--- DOF: {dof_name} ---")
    print(f"{'lambda':>8s} {'omega':>8s} {'B_direct':>12s} {'B_kochin_1':>14s} {'ratio':>8s} "
          f"{'B_kochin_4':>14s} {'ratio4':>8s}")
    print("-" * 80)

    for lam, k, omega in zip(wavelengths, k_values, omega_values):
        prob = cpt.RadiationProblem(
            body=body, radiating_dof=dof_name, omega=omega,
            water_depth=np.inf, rho=rho, g=g
        )
        result = solver.solve(prob)

        # Direct damping
        B_direct = result.radiation_dampings[dof_name]

        # Kochin function
        H = compute_kochin_hull_only(result, theta)
        H2_int = np.sum(np.abs(H)**2) * dtheta

        # Formula 1: B = (rho * k / (2*pi)) * int |H|^2 dtheta  (textbook, alpha=1/(2*pi))
        # If Capytaine H = H_textbook, this should work
        # But the 3D deep-water formula from Newman (1977) eq 6.170 is:
        #   B_jj = (rho * omega * k) / (2*pi) * int |H_j|^2 dtheta  ... no, let me check
        #
        # Actually from energy conservation:
        #   P_radiated = (1/2) * omega^2 * xi_j^2 * B_jj
        #   P_radiated = (rho*g * C_g) / (2*k) * int_0^2pi |A(theta)|^2 dtheta
        # where A(theta) is the far-field wave amplitude in direction theta
        #   A(theta) = some function of H(theta)
        #
        # Capytaine theory manual: the potential is
        #   phi ~ -sqrt(2*pi*k/rho) * exp(kz) * H(theta) * exp(ikr) * exp(i*pi/4)  (eq 82)
        # The wave elevation is eta = (1/g) * dphi/dt = (i*omega/g) * phi (for exp(-iwt))
        # Wait - Capytaine uses exp(-iwt)?  Need to check.
        #
        # Let's just compute the ratio empirically.

        # Try alpha = 1/(2*pi): B = rho*k/(2*pi) * int|H|^2 dtheta ... nah, dimensions wrong
        # The standard 3D formula (e.g. Lee 1995, Newman) for deep water is:
        #   B_jj = (rho * omega^2 * k / pi) * int_0^2pi |H_j(theta)|^2 dtheta
        # No wait, let me think about dimensions:
        #   [B] = kg/s for translational DOFs
        #   [rho] = kg/m^3
        #   [k] = 1/m
        #   [H] = ? from Capytaine's formula H = (1/4pi) int sigma*exp(kz)*exp(-ik..) dS
        #         [sigma] = 1/(m*s) (source strength, phi = int G*sigma dS)
        #         [H] = [1/4pi * sigma * 1 * 1 * m^2] = m/s
        #   So [rho * k * |H|^2 * dtheta] = kg/m^3 * 1/m * m^2/s^2 = kg/(m^2*s^2)
        #   That has dimensions of pressure / length, not damping
        #
        # The correct relation must include omega or similar.
        # From Capytaine theory manual, the drift force formula (eq 88) is:
        #   F_x = rho*g * (A^2 / 2) * { stuff with Kochin functions }
        # Actually, let me just try various formulas and see which gives ratio = 1

        # Try: B = rho * omega * k / (4*pi) * int |H|^2 dtheta  (from energy, assuming C=1)
        B_kochin_1 = rho * omega * k / (4*np.pi) * H2_int
        ratio1 = B_direct / B_kochin_1 if abs(B_kochin_1) > 1e-12 else float('nan')

        # Try: B = rho * omega * k / pi * int |H|^2 dtheta  (assuming C=2, extra factor 4)
        B_kochin_4 = rho * omega * k / np.pi * H2_int
        ratio4 = B_direct / B_kochin_4 if abs(B_kochin_4) > 1e-12 else float('nan')

        print(f"{lam:8.1f} {omega:8.3f} {B_direct:12.1f} {B_kochin_1:14.1f} {ratio1:8.4f} "
              f"{B_kochin_4:14.1f} {ratio4:8.4f}")

print()
print("=" * 80)
print("SEARCHING FOR THE CORRECT FORMULA")
print("=" * 80)
print()
print("Trying B = rho * alpha * int|H|^2 dtheta, solving for alpha at each frequency")
print()

# Use Heave as the test DOF
dof_name = 'Heave'
print(f"DOF: {dof_name}")
print(f"{'lambda':>8s} {'omega':>8s} {'k':>10s} {'B_direct':>12s} {'int|H|^2':>14s} {'alpha':>14s} {'alpha*pi/k/omega':>18s}")
print("-" * 90)

for lam, k, omega in zip(wavelengths, k_values, omega_values):
    prob = cpt.RadiationProblem(
        body=body, radiating_dof=dof_name, omega=omega,
        water_depth=np.inf, rho=rho, g=g
    )
    result = solver.solve(prob)
    B_direct = result.radiation_dampings[dof_name]
    H = compute_kochin_hull_only(result, theta)
    H2_int = np.sum(np.abs(H)**2) * dtheta

    alpha = B_direct / (rho * H2_int) if abs(H2_int) > 1e-20 else float('nan')
    alpha_norm = alpha * np.pi / (k * omega) if abs(k * omega) > 1e-20 else float('nan')

    print(f"{lam:8.1f} {omega:8.3f} {k:10.4f} {B_direct:12.1f} {H2_int:14.6e} {alpha:14.6e} {alpha_norm:18.6f}")

print()
print("If alpha_norm is constant across frequencies, then B = rho*(k*omega)/(alpha_norm*pi) * int|H|^2 dtheta")
print("alpha_norm = 1 means B = rho*k*omega/pi * int|H|^2, i.e. factor = 1/pi")
print("alpha_norm = 0.25 means B = 4*rho*k*omega/pi * int|H|^2, i.e. factor = 4/pi")

# Also check with diffraction: the relationship between cross-sections and Kochin
print()
print("=" * 80)
print("DIFFRACTION CHECK: Excitation force from Kochin (Haskind)")
print("=" * 80)
print()
print("Haskind relation: F_exc_j = -4*rho*omega * sqrt(2*pi*k) * exp(i*pi/4) * H_D(-beta+pi)")
print("   or some variant depending on normalization")
print()

beta = np.pi/2  # beam seas
for lam, k, omega in zip(wavelengths, k_values, omega_values):
    diff_prob = cpt.DiffractionProblem(
        body=body, wave_direction=beta, omega=omega,
        water_depth=np.inf, rho=rho, g=g
    )
    diff_result = solver.solve(diff_prob)

    # Direct excitation force (Sway for beam seas)
    F_sway_direct = diff_result.forces['Sway']

    # Kochin function of diffraction
    H_diff = compute_kochin_hull_only(diff_result, theta)

    # Evaluate H at theta = beta + pi (back-scatter direction for Haskind)
    # beta = pi/2, so theta = 3*pi/2
    idx_back = np.argmin(np.abs(theta - (beta + np.pi)))
    H_back = H_diff[idx_back]

    # Also try theta = beta (forward direction)
    idx_fwd = np.argmin(np.abs(theta - beta))
    H_fwd = H_diff[idx_fwd]

    # Standard Haskind: F_j = -4*rho*omega * sqrt(2*pi/k) * exp(i*pi/4) * H_j(beta+pi)
    # But this uses the radiation Kochin of DOF j, not diffraction Kochin
    # The diffraction Kochin relates to the scattering amplitude
    # Let me just print values for analysis
    print(f"lambda={lam:5.1f}  |F_sway|={abs(F_sway_direct):10.1f}  |H_back|={abs(H_back):10.6f}  "
          f"|H_fwd|={abs(H_fwd):10.6f}  ratio_back={abs(F_sway_direct)/(rho*omega*np.sqrt(2*np.pi/k)*abs(H_back)):10.4f}")
