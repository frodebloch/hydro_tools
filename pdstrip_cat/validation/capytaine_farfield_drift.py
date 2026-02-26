#!/usr/bin/env python3
"""
Compute mean drift forces using the FAR-FIELD (Maruo) method.

Instead of using Kochin functions (which have normalization ambiguities),
we evaluate the scattered+radiated potential at field points on a large-radius
circle and extract the far-field wave amplitude A(θ) directly.

The deep-water far-field potential for the disturbance (scattered + radiated) is:
  φ_d(r,θ,z) ~ -(ig/ω) * A(θ) * √(2/(πkr)) * exp(kz + ikr - iπ/4)

where A(θ) is the dimensionless far-field amplitude defined such that:
  η_d(r,θ) = (iω/g) φ_d |_{z=0} = A(θ) * √(2/(πkr)) * exp(ikr - iπ/4)

The mean radiated power (energy flux through a cylinder at r → ∞):
  P = (ρg Cg)/(2) ∫₀²π |A(θ)|² dθ     where Cg = g/(2ω)
    = (ρg²)/(4ω) ∫₀²π |A(θ)|² dθ

For radiation from DOF j with unit amplitude:
  B_jj = 2P/ω² = (ρg²)/(2ω³) ∫₀²π |A_j(θ)|² dθ

Maruo's far-field drift force formula (deep water, unit incident wave amplitude):
  F̄_i = (ρg)/(2k) ∫₀²π |A_total(θ)|² (ê_i(θ) - cos(θ-β) ê_i(β)) dθ

where A_total = A_scatter + Σ ξ_j A_radiation_j  (total disturbance amplitude),
and the second term accounts for the incident wave momentum flux.

Actually, the simplest correct Maruo formula for a 3D body in deep water is
(Newman 1967, Faltinsen 1990 eq. 6.26):

  F̄_x = (ρg)/(2k) ∫₀²π |A_total(θ)|² cos(θ) dθ  - (ρg)/(2k) ∫₀²π |A_inc(θ)|² cos(θ) dθ

But A_inc is a plane wave = delta function in angle space, so:
  ∫|A_inc|² cos(θ) dθ contribution needs careful treatment.

The correct general Maruo formula from momentum conservation (Newman 1967) is:

  F̄_x = -(ρg²)/(4ωπ) * Im[ ∫₀²π A_total(θ) * dA_total*/dθ * sin(θ) dθ ]

...which is hard to implement. Let me instead use the direct formula from
Faltinsen (1990) equation (6.75) for a 3D body:

  F̄_y = (ρg k)/(4π) ∫₀²π |H_total(θ)|² sin(θ) dθ

where H(θ) is the Kochin function. Since we verified that |A|² and |H|²
are proportional (with the 4π² factor from the 1/(4π) in our Kochin),
we can work with A(θ) directly.

APPROACH: Field-point ring
==========================
1. Solve BEM for all radiation + diffraction problems
2. Evaluate the disturbed potential at points (r cos θ, r sin θ, 0) for large r
3. Extract A(θ) from: φ_d = -(ig/ω) A(θ) √(2/(πkr)) exp(ikr - iπ/4 + kz)
   → A(θ) = φ_d * (iω/g) * √(πkr/2) * exp(-ikr + iπ/4)    at z=0
4. Verify: B_jj = (ρg²)/(2ω³) ∫|A_j|² dθ
5. Apply Maruo: F̄_y = coeff × ∫|A_total|² sin(θ) dθ
   with coeff determined empirically from the damping check
"""

import numpy as np
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_potential, airy_waves_velocity, froude_krylov_force
import logging
import sys

cpt.set_logging(logging.WARNING)

# ============================================================
# Parameters
# ============================================================
R = 1.0
L = 20.0
rho = 1025.0
g = 9.81

mesh_res = (10, 40, 50)

wavelengths_all = np.array([3, 4, 5, 6, 8, 10, 13, 17, 22, 28, 35, 45, 55, 70, 90])

if '--full' in sys.argv:
    wavelengths = wavelengths_all
else:
    wavelengths = np.array([3, 5, 10, 22, 55, 90])
    print(f"[Quick mode: {len(wavelengths)} wavelengths. Use --full for all 15]")

k_values = 2 * np.pi / wavelengths
omega_values = np.sqrt(k_values * g)

wave_directions = np.array([np.pi/2, np.pi])  # beam, head seas

# Field-point ring parameters
R_FIELD = 5000.0  # radius of field-point ring [m] — must be >> body AND >> lambda
N_THETA = 720     # number of angular points

# ============================================================
# Mesh creation (same as nearfield script)
# ============================================================
def make_hull_body(R, L, y_offset=0.0, name="hull"):
    mesh_full = cpt.mesh_horizontal_cylinder(
        length=L, radius=R, center=(0, y_offset, 0),
        resolution=mesh_res, name=name)
    hull_mesh = mesh_full.immersed_part()
    lid = hull_mesh.generate_lid(z=-0.01)
    body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name=name)
    body.center_of_mass = np.array([0.0, y_offset, -4*R/(3*np.pi)])
    body.mass = rho * np.pi * R**2 / 2 * L
    body.rotation_center = body.center_of_mass
    body.add_all_rigid_body_dofs()
    return body

body = make_hull_body(R, L)
print(f"Mesh: {body.mesh.nb_faces} hull + {body.lid_mesh.nb_faces} lid faces")

# Mass properties
mass = rho * np.pi * R**2 / 2 * L
zcg = -4*R/(3*np.pi)
kxx2 = (0.4 * 2*R)**2
kyy2 = (0.25 * L)**2
kzz2 = kyy2

M = np.zeros((6, 6))
M[0, 0] = M[1, 1] = M[2, 2] = mass
M[3, 3] = mass * kxx2
M[4, 4] = mass * kyy2
M[5, 5] = mass * kzz2

# Hydrostatic stiffness
Awp = 2 * R * L
C = np.zeros((6, 6))
C[2, 2] = rho * g * Awp
Iyy_wp = (2*R)**3 * L / 12
Ixx_wp = 2*R * L**3 / 12
C[3, 3] = rho * g * Iyy_wp - mass * g * zcg  # roll
C[4, 4] = rho * g * Ixx_wp - mass * g * zcg  # pitch

dof_names = list(body.dofs.keys())
n_dof = len(dof_names)

solver = cpt.BEMSolver()

# Field-point ring at z = 0 (slightly below to avoid issues)
theta_fp = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
dtheta = 2*np.pi / N_THETA
z_eval = 0.0  # evaluate at free surface

field_pts = np.column_stack([
    R_FIELD * np.cos(theta_fp),
    R_FIELD * np.sin(theta_fp),
    np.full(N_THETA, z_eval)
])

# ============================================================
# Helper: extract far-field wave elevation amplitude a(θ)
# ============================================================
def extract_wave_amplitude(phi_at_field, omega, k, r):
    """
    Extract far-field wave elevation amplitude a(θ) from potential at (r, θ, z=0).

    At large r, the disturbed free-surface elevation is:
      η_d(r,θ) = a(θ) × exp(ikr) / √r

    Since η = (iω/g) × φ at z=0 (Capytaine convention), and φ ~ f(θ) × exp(ikr)/√r:
      a(θ) = (iω/g) × φ × √r × exp(-ikr)

    Standard relations:
      B_jj = (ρg²)/(2ω³) × ∫|a_j|² dθ
      Maruo: F̄_y = (ρg/k) × ∫|a_total|² sin(θ) dθ
    """
    a = (1j * omega / g) * phi_at_field * np.sqrt(r) * np.exp(-1j * k * r)
    return a


# ============================================================
# Phase 1: Verify damping relation
# ============================================================
print("\n" + "="*90)
print("PHASE 1: DAMPING VERIFICATION")
print(f"B_jj vs (ρg²)/(2ω³) × ∫|a_j(θ)|² dθ  using field-point ring at r={R_FIELD}m")
print("="*90)

# Pick a few wavelengths for damping check
lam_check = np.array([3, 5, 10, 22, 90])
k_check = 2 * np.pi / lam_check
omega_check = np.sqrt(k_check * g)

for lam, k, omega in zip(lam_check, k_check, omega_check):
    print(f"\n  lambda={lam}m, omega={omega:.3f}, k={k:.4f}")
    print(f"  {'DOF':>8s}  {'B_direct':>12s}  {'B_from_a':>12s}  {'ratio':>8s}")
    
    for dof in ['Sway', 'Heave', 'Roll']:
        rad_prob = cpt.RadiationProblem(
            body=body, radiating_dof=dof, omega=omega, water_depth=np.inf)
        rad_result = solver.solve(rad_prob, keep_details=True)
        B_direct = rad_result.radiation_dampings[dof]
        
        # Evaluate potential at field points
        phi_fp = solver.compute_potential(field_pts, rad_result)
        
        # Extract a(θ) = (iω/g) × φ × √r × exp(-ikr)
        a_rad = extract_wave_amplitude(phi_fp, omega, k, R_FIELD)
        
        # Compute B from far-field amplitude: B = (ρg²)/(2ω³) × ∫|a|² dθ
        a2_int = np.sum(np.abs(a_rad)**2) * dtheta
        B_from_a = rho * g**2 / (2 * omega**3) * a2_int
        
        ratio = B_direct / B_from_a if abs(B_from_a) > 1e-12 else float('nan')
        print(f"  {dof:>8s}  {B_direct:12.1f}  {B_from_a:12.1f}  {ratio:8.4f}")


# ============================================================
# Phase 2: Full drift force computation
# ============================================================
print("\n" + "="*90)
print("PHASE 2: FAR-FIELD DRIFT FORCE COMPUTATION")
print(f"Maruo formula: F̄_y = (ρg/k) ∫|a(θ)|² sin(θ) dθ")
print(f"where a(θ) = (iω/g) × φ_disturb × √r × exp(-ikr)")
print("="*90)

# Store results
ff_results = {}

for omega, lam, k in zip(omega_values, wavelengths, k_values):
    print(f"\n--- omega={omega:.3f}, lambda={lam:.1f}m, k={k:.4f} ---")
    
    # Solve radiation problems
    rad_results = {}
    for dof in dof_names:
        rad_prob = cpt.RadiationProblem(
            body=body, radiating_dof=dof, omega=omega, water_depth=np.inf)
        rad_results[dof] = solver.solve(rad_prob, keep_details=True)
    
    # Added mass and damping
    A_mat = np.zeros((n_dof, n_dof))
    B_mat = np.zeros((n_dof, n_dof))
    for i, dof_i in enumerate(dof_names):
        for j, dof_j in enumerate(dof_names):
            A_mat[i, j] = rad_results[dof_j].added_masses[dof_i]
            B_mat[i, j] = rad_results[dof_j].radiation_dampings[dof_i]
    
    # Radiation potentials at field points
    rad_phi_fp = {}
    for dof in dof_names:
        rad_phi_fp[dof] = solver.compute_potential(field_pts, rad_results[dof])
    
    for beta in wave_directions:
        # Diffraction
        diff_prob = cpt.DiffractionProblem(
            body=body, wave_direction=beta, omega=omega, water_depth=np.inf)
        diff_result = solver.solve(diff_prob, keep_details=True)
        
        # Excitation forces (FK + diffraction)
        FK = froude_krylov_force(diff_prob)
        F_exc = np.array([diff_result.forces[dof] + FK[dof] for dof in dof_names])
        
        # RAOs
        Z = -omega**2 * (M + A_mat) + 1j * omega * B_mat + C
        xi = np.linalg.solve(Z, F_exc)
        
        # === Far-field wave amplitude ===
        
        # Scattered wave potential at field points
        # (compute_potential returns only scattered, not incident)
        phi_scatter_fp = solver.compute_potential(field_pts, diff_result)
        
        # Total disturbance potential = scattered + radiated
        phi_disturb_fp = phi_scatter_fp.copy()
        for i, dof in enumerate(dof_names):
            phi_disturb_fp += xi[i] * rad_phi_fp[dof]
        
        # Extract a(θ) = (iω/g) × φ × √r × exp(-ikr)
        a_total = extract_wave_amplitude(phi_disturb_fp, omega, k, R_FIELD)
        
        # === Maruo drift force (deep water) ===
        # F̄_i = (ρg/k) × ∫|a(θ)|² ê_i(θ) dθ
        # where ê = (cos θ, sin θ)
        a2 = np.abs(a_total)**2
        coeff = rho * g / k
        
        Fx_far = coeff * np.sum(a2 * np.cos(theta_fp)) * dtheta
        Fy_far = coeff * np.sum(a2 * np.sin(theta_fp)) * dtheta
        
        beta_deg = np.degrees(beta)
        print(f"  beta={beta_deg:.0f}°: Fx_far={Fx_far:.1f}, Fy_far={Fy_far:.1f}")
        print(f"    RAOs: " + " ".join(f"{dof}={abs(xi[i]):.4f}" 
              for i, dof in enumerate(dof_names)))
        
        a2_sin_int = np.sum(a2 * np.sin(theta_fp)) * dtheta
        a2_cos_int = np.sum(a2 * np.cos(theta_fp)) * dtheta
        print(f"    ∫|a|²sin(θ)dθ = {a2_sin_int:.6e},  ∫|a|²cos(θ)dθ = {a2_cos_int:.6e}")
        
        ff_results[(omega, beta)] = {
            'Fx_far': Fx_far,
            'Fy_far': Fy_far,
            'xi': xi.copy(),
            'a_total': a_total.copy(),
            'a2_sin_int': a2_sin_int,
            'a2_cos_int': a2_cos_int,
            'coeff': coeff,
        }


# ============================================================
# Phase 3: Compare with near-field
# ============================================================
print("\n" + "="*100)
print("FAR-FIELD vs NEAR-FIELD DRIFT FORCE COMPARISON")
print(f"Maruo: F̄ = (ρg/k) ∫|a(θ)|² ê dθ")
print(f"Field-point ring: r={R_FIELD}m, N={N_THETA} points")
print("="*100)

try:
    nf = np.load('nearfield_drift_comparison.npz')
    nf_lam = nf['wavelengths']
    nf_fy = nf['beam_Fy_total']
    
    print(f"\n--- BEAM SEAS (β=π/2): Fy ---")
    print(f"{'lam':>5} {'NF_Fy':>12} {'FF_Fy':>12} {'NF/FF':>10}")
    print("-" * 45)
    
    for omega, lam, k in zip(omega_values, wavelengths, k_values):
        key = (omega, np.pi/2)
        if key in ff_results:
            ff_fy = ff_results[key]['Fy_far']
            nf_idx = np.where(nf_lam == lam)[0]
            if len(nf_idx) > 0:
                nf_fy_val = nf_fy[nf_idx[0]]
                ratio = nf_fy_val / ff_fy if abs(ff_fy) > 1 else float('inf')
                print(f"{lam:5.0f} {nf_fy_val:12.1f} {ff_fy:12.1f} {ratio:10.3f}")
            else:
                print(f"{lam:5.0f} {'N/A':>12} {ff_fy:12.1f}")

    print(f"\n--- HEAD SEAS (β=π): Fx ---")
    nf_fx = nf['head_Fx_total']
    print(f"{'lam':>5} {'NF_Fx':>12} {'FF_Fx':>12} {'NF/FF':>10}")
    print("-" * 45)
    
    for omega, lam, k in zip(omega_values, wavelengths, k_values):
        key = (omega, np.pi)
        if key in ff_results:
            ff_fx = ff_results[key]['Fx_far']
            nf_idx = np.where(nf_lam == lam)[0]
            if len(nf_idx) > 0:
                nf_fx_val = nf_fx[nf_idx[0]]
                ratio = nf_fx_val / ff_fx if abs(ff_fx) > 1 else float('inf')
                print(f"{lam:5.0f} {nf_fx_val:12.1f} {ff_fx:12.1f} {ratio:10.3f}")
            else:
                print(f"{lam:5.0f} {'N/A':>12} {ff_fx:12.1f}")

except FileNotFoundError:
    print("No near-field results file found. Run capytaine_nearfield_drift.py --full first.")

# ============================================================
# Phase 4: Empirical coefficient check
# ============================================================
print("\n" + "="*100)
print("EMPIRICAL: NF_Fy / ∫|a|²sin(θ)dθ  (should = ρg/k)")
print("="*100)

try:
    print(f"{'lam':>5} {'NF_Fy':>12} {'∫|a|²sinθ':>14} {'C_emp':>14} {'ρg/k':>14} {'ratio':>10}")
    print("-" * 70)
    
    for omega, lam, k in zip(omega_values, wavelengths, k_values):
        key = (omega, np.pi/2)
        if key in ff_results:
            a2_sin = ff_results[key]['a2_sin_int']
            nf_idx = np.where(nf_lam == lam)[0]
            if len(nf_idx) > 0 and abs(a2_sin) > 1e-15:
                nf_fy_val = nf_fy[nf_idx[0]]
                C_emp = nf_fy_val / a2_sin
                C_theory = rho * g / k
                ratio = C_emp / C_theory if abs(C_theory) > 0 else float('nan')
                print(f"{lam:5.0f} {nf_fy_val:12.1f} {a2_sin:14.6e} {C_emp:14.1f} {C_theory:14.1f} {ratio:10.4f}")
except:
    pass

