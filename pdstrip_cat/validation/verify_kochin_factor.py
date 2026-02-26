#!/usr/bin/env python3
"""
Verify the factor-of-2 discrepancy in Capytaine theory manual eq (86).

From the corrected far-field:
  Φ ~ -C * √(2πk/ρ) * exp(kz) * H(θ) * exp(ikρ) * exp(iπ/4)

where C = 1 (as written in theory manual eq 82) or C = 2 (from eq 85→86 derivation).

The damping from energy conservation (deep water):
  B_jj = (ρ * C² * 2πk) / (2g) * ω * ∫|H_j(θ)|² dθ    ... let me derive this properly

Actually, let me derive the damping-Kochin relation from first principles.

In deep water, the far-field potential for a unit-amplitude oscillation of DOF j is:
  Φ_j(r,θ,z,t) = Re[ φ_j(r,θ,z) * exp(-iωt) ]

where:
  φ_j ~ -C √(2πk/ρ) * exp(kz) * H_j(θ) * exp(ikρ+iπ/4)

The free surface elevation is:
  η_j = (iω/g) * φ_j |_{z=0}   (from eq 80)
     ~ -(iω/g) * C √(2πk/ρ) * H_j(θ) * exp(ikρ+iπ/4)

The far-field wave amplitude A_j(θ) is defined by:
  η_j ~ A_j(θ) * exp(ikρ) / √ρ   (cylindrical spreading)

So:
  A_j(θ) = -(iω/g) * C √(2πk) * H_j(θ) * exp(iπ/4)

|A_j(θ)|² = (ω²/g²) * C² * 2πk * |H_j(θ)|²

Power radiated per unit angle:
  dP/dθ = (ρg²/(4ω)) * |A_j(θ)|²    (deep water, Cg = g/(2ω))
        = (ρg²/(4ω)) * (ω²/g²) * C² * 2πk * |H_j(θ)|²
        = (ρω C² πk / 2) * |H_j(θ)|²

Total radiated power:
  P = ∫₀²π dP/dθ dθ = (ρω C² πk / 2) * ∫|H_j|² dθ

Damping:
  P = ½ ω² |ξ_j|² B_jj   (for unit amplitude ξ_j=1)
  B_jj = 2P/ω² = (ρ C² πk / ω) * ∫|H_j|² dθ

So the formula is:
  B_jj = (ρ * π * k * C²) / ω * ∫₀²π |H_j(θ)|² dθ

If C=1:  B = ρπk/ω * ∫|H|²dθ
If C=2:  B = 4ρπk/ω * ∫|H|²dθ

Let me test both.

Similarly, the drift force formula (Maruo far-field):
For a fixed body (diffraction only), the drift force is:
  F_x = -(ρg/2k) cos(β) + (ρg C² 2πk)/(4g) * ∫|A_total|² cosθ dθ / (something)

Actually, let me be more careful. The standard Maruo formula in terms of far-field 
wave amplitude A(θ) (where η ~ A(θ) exp(ikρ)/√ρ) is:

  F_x = -(ρg/2k) cos(β) + (ρg/(4πk)) ∫₀²π |kA(θ)|² cos(θ) dθ

Wait, let me use the Maruo formula as given in e.g., Faltinsen (1990).

Standard Maruo far-field formula (deep water):
  F_x = (ρg)/(4π) ∫₀²π |A_total(θ)/A_I|² (cos(θ) - cos(β)) dθ   ... no this isn't right either.

Let me use Newman's formulation. For a body in waves with incident wave amplitude a,
wave direction β, the mean drift force in the x-direction is:

  F̄_x / a² = -(ρg)/(2k) cos(β) + (ρg Cg)/(2) ∫₀²π |A_S(θ)/a|² cos(θ) dθ / (2π/(kCg))

OK this is getting confusing with normalizations. Let me just verify C empirically.
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

def compute_kochin_all(result, theta):
    """Compute Kochin function using ALL sources (hull + lid)."""
    k = result.wavenumber
    n_hull = result.body.mesh.nb_faces
    n_total = len(result.sources)
    
    # Hull part
    hull_centers = result.body.mesh.faces_centers
    hull_areas = result.body.mesh.faces_areas
    sources_hull = result.sources[:n_hull]
    
    omega_bar_h = hull_centers[:, 0:2] @ np.array([np.cos(theta), np.sin(theta)])
    cih_h = np.exp(k * hull_centers[:, 2])
    zs_h = cih_h[:, None] * np.exp(-1j * k * omega_bar_h) * hull_areas[:, None]
    H_hull = (zs_h.T @ sources_hull) / (4 * np.pi)
    
    # Lid part
    n_lid = n_total - n_hull
    if n_lid > 0 and hasattr(result.body, 'lid_mesh') and result.body.lid_mesh is not None:
        lid_centers = result.body.lid_mesh.faces_centers
        lid_areas = result.body.lid_mesh.faces_areas
        sources_lid = result.sources[n_hull:]
        
        omega_bar_l = lid_centers[:, 0:2] @ np.array([np.cos(theta), np.sin(theta)])
        cih_l = np.exp(k * lid_centers[:, 2])
        zs_l = cih_l[:, None] * np.exp(-1j * k * omega_bar_l) * lid_areas[:, None]
        H_lid = (zs_l.T @ sources_lid) / (4 * np.pi)
        return H_hull + H_lid
    else:
        return H_hull


# Create body WITHOUT lid (cleaner for verification)
mesh_full = cpt.mesh_horizontal_cylinder(
    length=L, radius=R, center=(0, 0, 0),
    resolution=mesh_res, name="nolid"
)
hull_mesh = mesh_full.immersed_part()
body_nolid = cpt.FloatingBody(mesh=hull_mesh, name="nolid")
body_nolid.add_all_rigid_body_dofs()
body_nolid.center_of_mass = np.array([0.0, 0.0, -4*R/(3*np.pi)])

# Create body WITH lid
mesh_full2 = cpt.mesh_horizontal_cylinder(
    length=L, radius=R, center=(0, 0, 0),
    resolution=mesh_res, name="withlid"
)
hull_mesh2 = mesh_full2.immersed_part()
lid2 = hull_mesh2.generate_lid(z=-0.01)
body_lid = cpt.FloatingBody(mesh=hull_mesh2, lid_mesh=lid2, name="withlid")
body_lid.add_all_rigid_body_dofs()
body_lid.center_of_mass = np.array([0.0, 0.0, -4*R/(3*np.pi)])

solver = cpt.BEMSolver()

n_kochin = 720
theta = np.linspace(0, 2*np.pi, n_kochin, endpoint=False)
dtheta = 2 * np.pi / n_kochin

wavelengths = np.array([3, 5, 8, 10, 13, 20, 40, 90])
k_values = 2 * np.pi / wavelengths
omega_values = np.sqrt(k_values * g)

print("=" * 90)
print("VERIFICATION: B_direct vs B_kochin = C² * ρπk/ω * ∫|H|² dθ")
print("If C=1: ratio = B_direct / (ρπk/ω ∫|H|²dθ) should be 1")
print("If C=2: ratio should be 4")
print("=" * 90)

for body, body_name in [(body_nolid, "NO LID"), (body_lid, "WITH LID")]:
    for dof_name in ['Sway', 'Heave', 'Surge']:
        print(f"\n--- {body_name}, DOF: {dof_name} ---")
        print(f"{'lambda':>8s} {'omega':>8s} {'B_direct':>12s} {'B_C1':>12s} {'ratio_C1':>10s} "
              f"{'B_C4':>12s} {'ratio_C4':>10s}")
        print("-" * 75)

        for lam, k, omega in zip(wavelengths, k_values, omega_values):
            prob = cpt.RadiationProblem(
                body=body, radiating_dof=dof_name, omega=omega,
                water_depth=np.inf, rho=rho, g=g
            )
            result = solver.solve(prob)
            B_direct = result.radiation_dampings[dof_name]

            H = compute_kochin_all(result, theta)
            H2_int = np.sum(np.abs(H)**2) * dtheta

            # C=1: B = ρπk/ω * ∫|H|²dθ
            B_C1 = rho * np.pi * k / omega * H2_int
            ratio_C1 = B_direct / B_C1 if abs(B_C1) > 1e-12 else float('nan')

            # C=2: B = 4ρπk/ω * ∫|H|²dθ
            B_C4 = 4 * rho * np.pi * k / omega * H2_int
            ratio_C4 = B_direct / B_C4 if abs(B_C4) > 1e-12 else float('nan')

            print(f"{lam:8.1f} {omega:8.3f} {B_direct:12.1f} {B_C1:12.1f} {ratio_C1:10.4f} "
                  f"{B_C4:12.1f} {ratio_C4:10.4f}")

# Also verify with Haskind relation for the excitation force
# Haskind: F_exc,j = -C * 4π * ρ * (iω) * √(2πk) * exp(iπ/4) * H_j(β+π) ... 
# Actually the standard Haskind relation depends on the normalization.
# 
# The incident potential is (eq 19):
#   Φ₀ = -ig/ω * exp(kz) * exp(ik(x cosβ + y sinβ))
#
# For unit incident wave amplitude a=1 (not unit potential), 
# the elevation is η₀ = (iω/g)Φ₀ = exp(ik(x cosβ + y sinβ)) at z=0.
# So Capytaine's Φ₀ is for unit wave amplitude.
#
# Haskind (using Capytaine's Kochin H):
#   F_j = ρ * (something) * H_rad_j evaluated at θ = β + π
#
# Actually the Haskind relation says (for unit incident wave amplitude):
#   F_j = -iρg * C_group * 4π * H_j(β) * (-ig/ω) * ... 
#
# This is getting complicated. Let me just test the Haskind relation using
# the energy-based approach:
#   |F_j|² = 8πρg²Cg * B_jj * ... no, that's the excitation-damping relation.
#
# The excitation-damping relation (from Haskind) is:
#   |F_j(β)|² / B_jj = 8π ρ g Cg     (deep water, Cg = g/(2ω))
#                     = 4π ρ g²/ω
#
# Let me verify this:
print()
print("=" * 90)
print("HASKIND RELATION CHECK: |F_j|² / B_jj = 4πρg²/ω")
print("(This is independent of Kochin normalization)")
print("=" * 90)

beta = np.pi / 2  # beam seas
for dof_name in ['Sway', 'Heave']:
    print(f"\n--- DOF: {dof_name}, beta=pi/2 ---")
    print(f"{'lambda':>8s} {'|F|²':>14s} {'B':>12s} {'|F|²/B':>14s} {'4πρg²/ω':>14s} {'ratio':>8s}")
    print("-" * 80)

    for lam, k, omega in zip(wavelengths, k_values, omega_values):
        # Radiation
        rad_prob = cpt.RadiationProblem(
            body=body_nolid, radiating_dof=dof_name, omega=omega,
            water_depth=np.inf, rho=rho, g=g
        )
        rad_result = solver.solve(rad_prob)
        B = rad_result.radiation_dampings[dof_name]

        # Diffraction
        diff_prob = cpt.DiffractionProblem(
            body=body_nolid, wave_direction=beta, omega=omega,
            water_depth=np.inf, rho=rho, g=g
        )
        diff_result = solver.solve(diff_prob)
        F = diff_result.forces[dof_name]

        lhs = abs(F)**2 / B
        rhs = 4 * np.pi * rho * g**2 / omega

        print(f"{lam:8.1f} {abs(F)**2:14.1f} {B:12.1f} {lhs:14.1f} {rhs:14.1f} {lhs/rhs:8.4f}")
