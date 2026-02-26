#!/usr/bin/env python3
"""
Far-field drift force using Capytaine's Kochin function + Maruo formula.

CORRECTED normalization (Feb 2026):

Capytaine's compute_kochin() returns H_cap(θ) defined as:
  H_cap(θ) = (1/4π) ∫_Γ σ(ξ) exp(kξ₃) exp(-ik(ξ₁cosθ + ξ₂sinθ)) dξ

The far-field potential has a factor of 2 missing from the Capytaine manual eq(87).
The CORRECT asymptotic is:
  Φ ~ -2·√(2πk/ρ)·exp(kz)·H_cap(θ)·exp(ikρ+iπ/4)

This leads to the CORRECTED formulas (verified via damping cross-check):

  Damping:  B_jj = (4ρkπ/ω) ∫₀²π |H_cap_j(θ)|² dθ

  Drift:    F_x = 2ρπk ∫₀²π |H_cap_total(θ)|² cos(θ) dθ
            F_y = 2ρπk ∫₀²π |H_cap_total(θ)|² sin(θ) dθ

  where H_cap_total = H_scatter + Σ_j ξ_j H_rad_j  (floating body)
        H_cap_total = H_scatter                      (fixed body)

IMPORTANT: When using a lid mesh for irregular frequency removal,
Capytaine's compute_kochin() fails because result.sources includes lid panels
but the function uses body.mesh (hull only). We use a custom kochin_full()
function that correctly uses hull + lid mesh for face centers and areas.
"""
import numpy as np
import capytaine as cpt
from capytaine.meshes.collections import CollectionOfMeshes
from capytaine.bem.airy_waves import froude_krylov_force
import logging
import re

cpt.set_logging(logging.WARNING)

R = 1.0; L = 20.0; rho = 1025.0; g = 9.81

# ============================================================
# Build body with lid mesh for irregular frequency removal
# ============================================================
mesh_full = cpt.mesh_horizontal_cylinder(
    length=L, radius=R, center=(0, 0, 0),
    resolution=(10, 40, 50), name="hull")
hull_mesh = mesh_full.immersed_part()
lid = hull_mesh.generate_lid(z=-0.01)
body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name="hull")

mass_val = 32201.3
zcg = -0.4244
kxx_sq = 0.64; kyy_sq = 25.0; kzz_sq = 25.0
body.center_of_mass = np.array([0, 0, zcg])
body.mass = mass_val
body.rotation_center = body.center_of_mass
body.add_all_rigid_body_dofs()

solver = cpt.BEMSolver()
dof_names = list(body.dofs.keys())
n_dof = len(dof_names)

# Mass matrix (diagonal since rotation_center = CoM)
M = np.zeros((n_dof, n_dof))
for i, dof in enumerate(dof_names):
    if dof in ('Surge', 'Sway', 'Heave'): M[i,i] = mass_val
    elif dof == 'Roll': M[i,i] = mass_val * kxx_sq
    elif dof == 'Pitch': M[i,i] = mass_val * kyy_sq
    elif dof == 'Yaw': M[i,i] = mass_val * kzz_sq

# Hydrostatic stiffness
stiffness_xr = body.compute_hydrostatic_stiffness()
C = np.zeros((n_dof, n_dof))
for i, idof in enumerate(dof_names):
    for j, jdof in enumerate(dof_names):
        try: C[i,j] = float(stiffness_xr.sel(influenced_dof=idof, radiating_dof=jdof))
        except: C[i,j] = 0.0

# ============================================================
# Custom Kochin function that handles lid mesh correctly
# ============================================================
def kochin_full(result, theta_arr):
    """
    Compute Kochin function H(θ) = (1/4π) ∫ σ exp(kz) exp(-ik(x cosθ + y sinθ)) dS
    using the FULL mesh (hull + lid) to match result.sources.
    """
    if body.lid_mesh is not None:
        full_mesh = CollectionOfMeshes([body.mesh, body.lid_mesh])
    else:
        full_mesh = body.mesh
    centers = full_mesh.faces_centers
    areas = full_mesh.faces_areas
    kk = result.wavenumber
    # omega_bar[i,j] = x_i*cos(theta_j) + y_i*sin(theta_j)
    omega_bar = centers[:, 0:2] @ np.array([np.cos(theta_arr), np.sin(theta_arr)])
    cih = np.exp(kk * centers[:, 2])
    zs = (cih[:, None] * np.exp(-1j * kk * omega_bar) * areas[:, None])
    return (zs.T @ result.sources) / (4 * np.pi)

# Kochin function parameters
N_THETA = 720
theta_arr = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
dtheta = 2*np.pi / N_THETA

# ============================================================
# Phase 1: Verify Kochin normalization via damping coefficient
# ============================================================
print("="*90)
print("PHASE 1: Kochin normalization verification (CORRECTED)")
print("B_jj = (4ρkπ/ω) ∫|H_cap|² dθ")
print("="*90)

test_lam = [3, 5, 10, 22, 55]
print(f"\n{'lam':>5} {'DOF':>8} {'B_direct':>12} {'B_kochin':>12} {'ratio':>8}")
print("-"*50)

for lam in test_lam:
    k = 2*np.pi/lam; omega = np.sqrt(k*g)
    
    for dof in ['Sway', 'Heave']:
        rad_prob = cpt.RadiationProblem(body=body, radiating_dof=dof,
                                         omega=omega, water_depth=np.inf, rho=rho, g=g)
        rad_result = solver.solve(rad_prob, keep_details=True)
        B_direct = rad_result.radiation_dampings[dof]
        
        H_rad = kochin_full(rad_result, theta_arr)
        H2_int = np.sum(np.abs(H_rad)**2) * dtheta
        # CORRECTED formula:
        B_kochin = 4 * rho * k * np.pi / omega * H2_int
        
        r = B_direct / B_kochin if abs(B_kochin) > 1e-6 else float('nan')
        print(f"{lam:5.0f} {dof:>8} {B_direct:12.1f} {B_kochin:12.1f} {r:8.4f}")

# ============================================================
# Phase 2: Drift forces — FIXED body
# ============================================================
print(f"\n\n{'='*90}")
print("PHASE 2: FIXED BODY drift force (scattering only) — CORRECTED")
print("F_y = 2ρπk ∫|H_s(θ)|² sinθ dθ")
print("="*90)

beta_beam = np.pi/2
beta_head = np.pi

wavelengths = np.array([3, 4, 5, 6, 8, 10, 13, 17, 22, 28, 35, 45, 55, 70, 90])

print(f"\nBEAM SEAS (β=π/2): Fy drift")
print(f"{'lam':>5} {'kR':>6} {'Fy_Maruo':>12} {'∫|H|²sinθ':>14}")
print("-"*45)

ff_beam_fy = []
for lam in wavelengths:
    k = 2*np.pi/lam; omega = np.sqrt(k*g); kR = k*R
    
    diff_prob = cpt.DiffractionProblem(body=body, wave_direction=beta_beam,
                                        omega=omega, water_depth=np.inf, rho=rho, g=g)
    diff_result = solver.solve(diff_prob, keep_details=True)
    
    H_scat = kochin_full(diff_result, theta_arr)
    H2_sin_int = np.sum(np.abs(H_scat)**2 * np.sin(theta_arr)) * dtheta
    
    # CORRECTED formula:
    F_y = 2 * rho * np.pi * k * H2_sin_int
    ff_beam_fy.append(F_y)
    
    print(f"{lam:5.0f} {kR:6.3f} {F_y:12.1f} {H2_sin_int:14.6f}")

print(f"\nHEAD SEAS (β=π): Fx drift")
print(f"{'lam':>5} {'kR':>6} {'Fx_Maruo':>12} {'∫|H|²cosθ':>14}")
print("-"*45)

ff_head_fx = []
for lam in wavelengths:
    k = 2*np.pi/lam; omega = np.sqrt(k*g); kR = k*R
    
    diff_prob = cpt.DiffractionProblem(body=body, wave_direction=beta_head,
                                        omega=omega, water_depth=np.inf, rho=rho, g=g)
    diff_result = solver.solve(diff_prob, keep_details=True)
    
    H_scat = kochin_full(diff_result, theta_arr)
    H2_cos_int = np.sum(np.abs(H_scat)**2 * np.cos(theta_arr)) * dtheta
    
    # CORRECTED formula:
    F_x = 2 * rho * np.pi * k * H2_cos_int
    ff_head_fx.append(F_x)
    
    print(f"{lam:5.0f} {kR:6.3f} {F_x:12.1f} {H2_cos_int:14.6f}")

# ============================================================
# Phase 3: Drift forces — FREELY FLOATING body
# ============================================================
print(f"\n\n{'='*90}")
print("PHASE 3: FREELY FLOATING body drift force — CORRECTED")
print("H_total = H_scatter + Σ ξ_j H_rad_j")
print("F_y = 2ρπk ∫|H_total|² sinθ dθ")
print("="*90)

# Compute radiation for all wavelengths first, reuse for beam and head seas
rad_cache = {}  # {(lam, dof): rad_result}
AM_cache = {}   # {lam: (A_matrix, B_matrix)}

def get_radiation(lam):
    """Compute and cache radiation results + A/B matrices for a wavelength."""
    if lam in AM_cache:
        return AM_cache[lam], {dof: rad_cache[(lam, dof)] for dof in dof_names}
    
    k = 2*np.pi/lam; omega = np.sqrt(k*g)
    rad_results = {}
    for dof in dof_names:
        rad_prob = cpt.RadiationProblem(body=body, radiating_dof=dof,
                                         omega=omega, water_depth=np.inf, rho=rho, g=g)
        rad_results[dof] = solver.solve(rad_prob, keep_details=True)
        rad_cache[(lam, dof)] = rad_results[dof]
    
    A_mat = np.zeros((n_dof, n_dof)); B_mat = np.zeros((n_dof, n_dof))
    for i, rdof in enumerate(dof_names):
        for j, idof in enumerate(dof_names):
            A_mat[i,j] = rad_results[rdof].added_masses[idof]
            B_mat[i,j] = rad_results[rdof].radiation_dampings[idof]
    
    AM_cache[lam] = (A_mat, B_mat)
    return (A_mat, B_mat), rad_results

def compute_floating_drift(lam, beta, component='x'):
    """Compute Maruo far-field drift force for freely floating body."""
    k = 2*np.pi/lam; omega = np.sqrt(k*g)
    
    (A_mat, B_mat), rad_results = get_radiation(lam)
    
    # Diffraction
    diff_prob = cpt.DiffractionProblem(body=body, wave_direction=beta,
                                        omega=omega, water_depth=np.inf, rho=rho, g=g)
    diff_result = solver.solve(diff_prob, keep_details=True)
    
    # RAOs
    FK = froude_krylov_force(diff_prob)
    F_exc = np.array([diff_result.forces[dof] + FK[dof] for dof in dof_names])
    Z = -omega**2 * (M + A_mat) + 1j * omega * B_mat + C
    xi = np.linalg.solve(Z, F_exc)
    
    # Total Kochin = scatter + sum(xi_j * radiation_j)
    H_total = kochin_full(diff_result, theta_arr)
    for i, dof in enumerate(dof_names):
        H_total = H_total + xi[i] * kochin_full(rad_results[dof], theta_arr)
    
    if component == 'x':
        integral = np.sum(np.abs(H_total)**2 * np.cos(theta_arr)) * dtheta
    elif component == 'y':
        integral = np.sum(np.abs(H_total)**2 * np.sin(theta_arr)) * dtheta
    else:
        raise ValueError(f"Unknown component: {component}")
    
    # CORRECTED Maruo formula:
    F = 2 * rho * np.pi * k * integral
    return F, xi

# --- Beam seas ---
print(f"\nBEAM SEAS (β=π/2): Fy drift")
print(f"{'lam':>5} {'kR':>6} {'Fy_float':>12} {'Fy_fixed':>12} {'ratio':>8}  {'|ξ_sway|':>9} {'|ξ_heave|':>10} {'|ξ_roll|':>9}")
print("-"*90)

ff_beam_fy_float = []
beam_raos = []
for lam in wavelengths:
    k = 2*np.pi/lam; kR = k*R
    F_y, xi = compute_floating_drift(lam, beta_beam, 'y')
    ff_beam_fy_float.append(F_y)
    beam_raos.append(xi)
    
    idx = list(wavelengths).index(lam)
    fy_fixed = ff_beam_fy[idx]
    r = F_y / fy_fixed if abs(fy_fixed) > 0.01 else float('nan')
    
    sway_idx = dof_names.index('Sway')
    heave_idx = dof_names.index('Heave')
    roll_idx = dof_names.index('Roll')
    
    print(f"{lam:5.0f} {kR:6.3f} {F_y:12.1f} {fy_fixed:12.1f} {r:8.3f}  "
          f"{np.abs(xi[sway_idx]):9.4f} {np.abs(xi[heave_idx]):10.4f} {np.abs(xi[roll_idx]):9.4f}")

# --- Head seas ---
print(f"\nHEAD SEAS (β=π): Fx drift")
print(f"{'lam':>5} {'kR':>6} {'Fx_float':>12} {'Fx_fixed':>12} {'ratio':>8}  {'|ξ_surge|':>10} {'|ξ_heave|':>10} {'|ξ_pitch|':>10}")
print("-"*90)

ff_head_fx_float = []
head_raos = []
for lam in wavelengths:
    k = 2*np.pi/lam; kR = k*R
    F_x, xi = compute_floating_drift(lam, beta_head, 'x')
    ff_head_fx_float.append(F_x)
    head_raos.append(xi)
    
    idx = list(wavelengths).index(lam)
    fx_fixed = ff_head_fx[idx]
    r = F_x / fx_fixed if abs(fx_fixed) > 0.01 else float('nan')
    
    surge_idx = dof_names.index('Surge')
    heave_idx = dof_names.index('Heave')
    pitch_idx = dof_names.index('Pitch')
    
    print(f"{lam:5.0f} {kR:6.3f} {F_x:12.1f} {fx_fixed:12.1f} {r:8.3f}  "
          f"{np.abs(xi[surge_idx]):10.4f} {np.abs(xi[heave_idx]):10.4f} {np.abs(xi[pitch_idx]):10.4f}")

# ============================================================
# Phase 4: Compare with pdstrip
# ============================================================
print(f"\n\n{'='*100}")
print("PHASE 4: pdstrip vs Capytaine far-field (Maruo) comparison")
print("Coordinate mapping: feta_pdstrip = Fy_capytaine (both +y = starboard)")
print("                    fxi_pdstrip  = Fx_capytaine (both +x = forward)")
print("="*100)

# Parse pdstrip debug.out
fnum = r'[+-]?[\d.]+(?:[EeDd][+-]?\d+)?'
blocks = []
current = None
with open("/home/blofro/src/pdstrip_test/validation/run_mono/debug.out") as f:
    for line in f:
        line = line.strip()
        if line.startswith('DRIFT_START'):
            m = re.search(rf'omega=\s*({fnum})\s+mu=\s*({fnum})', line)
            if m:
                current = {'omega': float(m.group(1)), 'mu': float(m.group(2))}
        elif line.startswith('DRIFT_TOTAL') and current is not None:
            m = re.search(rf'fxi=\s*({fnum})\s+feta=\s*({fnum})\s+fxi_WL=\s*({fnum})\s+feta_WL=\s*({fnum})\s+fxi_vel=\s*({fnum})\s+fxi_rot=\s*({fnum})', line)
            if m:
                current['fxi'] = float(m.group(1))
                current['feta'] = float(m.group(2))
                current['fxi_WL'] = float(m.group(3))
                current['feta_WL'] = float(m.group(4))
                current['fxi_vel'] = float(m.group(5))
                current['fxi_rot'] = float(m.group(6))
                blocks.append(current)
                current = None
            else:
                # Fallback: try simpler pattern
                m2 = re.search(rf'fxi=\s*({fnum})\s+feta=\s*({fnum})', line)
                if m2:
                    current['fxi'] = float(m2.group(1))
                    current['feta'] = float(m2.group(2))
                    blocks.append(current)
                    current = None

print(f"\nParsed {len(blocks)} DRIFT_TOTAL entries from debug.out")

# The barge pdstrip.inp has 3 headings: -90, 0, 90
# After mirroring: -90, 0, 90, 180  (4 headings)
# Block ordering: for each wavelength → for each heading
# Headings: [-90, 0, 90, 180] → indices [0, 1, 2, 3]
# mu=90 is index 2, mu=180 is index 3

n_headings = 4  # -90, 0, 90, 180
n_speeds = 1

# Verify by checking a few blocks
if len(blocks) > 0:
    print(f"First block: omega={blocks[0]['omega']:.4f}, mu={blocks[0]['mu']:.1f}")
    if len(blocks) > 3:
        print(f"Block 3:     omega={blocks[3]['omega']:.4f}, mu={blocks[3]['mu']:.1f}")

print(f"\n--- BEAM SEAS: pdstrip feta (mu=90) vs Capytaine Maruo Fy (β=π/2) ---")
print(f"{'lam':>5} {'kR':>6} {'pd_feta':>12} {'FF_Fy':>12} {'ratio':>8}  {'pd_WL':>10} {'pd_vel':>10} {'pd_rot':>10}")
print("-"*85)

for i, lam in enumerate(wavelengths):
    k = 2*np.pi/lam; kR = k*R
    
    # Find the block for this wavelength at mu=90
    block_idx = i * n_headings + 2  # mu=90 is index 2
    if block_idx >= len(blocks):
        print(f"{lam:5.0f} {kR:6.3f}  ** no pdstrip data **")
        continue
    
    b = blocks[block_idx]
    pd_feta = b['feta']
    ff_fy = ff_beam_fy_float[i]
    r = pd_feta / ff_fy if abs(ff_fy) > 0.01 else float('nan')
    
    wl = b.get('feta_WL', float('nan'))
    vel = b.get('fxi_vel', float('nan'))  # Note: fxi_vel tracks x-component
    rot = b.get('fxi_rot', float('nan'))
    
    print(f"{lam:5.0f} {kR:6.3f} {pd_feta:12.1f} {ff_fy:12.1f} {r:8.3f}  "
          f"{wl:10.1f} {vel:10.1f} {rot:10.1f}")

print(f"\n--- HEAD SEAS: pdstrip fxi (mu=180) vs Capytaine Maruo Fx (β=π) ---")
print(f"{'lam':>5} {'kR':>6} {'pd_fxi':>12} {'FF_Fx':>12} {'ratio':>8}  {'pd_WL':>10} {'pd_vel':>10} {'pd_rot':>10}")
print("-"*85)

for i, lam in enumerate(wavelengths):
    k = 2*np.pi/lam; kR = k*R
    
    block_idx = i * n_headings + 3  # mu=180 is index 3
    if block_idx >= len(blocks):
        print(f"{lam:5.0f} {kR:6.3f}  ** no pdstrip data **")
        continue
    
    b = blocks[block_idx]
    pd_fxi = b['fxi']
    ff_fx = ff_head_fx_float[i]
    r = pd_fxi / ff_fx if abs(ff_fx) > 0.01 else float('nan')
    
    wl = b.get('fxi_WL', float('nan'))
    vel = b.get('fxi_vel', float('nan'))
    rot = b.get('fxi_rot', float('nan'))
    
    print(f"{lam:5.0f} {kR:6.3f} {pd_fxi:12.1f} {ff_fx:12.1f} {r:8.3f}  "
          f"{wl:10.1f} {vel:10.1f} {rot:10.1f}")

# Save results
np.savez("/home/blofro/src/pdstrip_test/validation/maruo_drift_comparison.npz",
         wavelengths=wavelengths,
         ff_beam_fy_fixed=np.array(ff_beam_fy),
         ff_beam_fy_float=np.array(ff_beam_fy_float),
         ff_head_fx_fixed=np.array(ff_head_fx),
         ff_head_fx_float=np.array(ff_head_fx_float))
print("\nSaved: maruo_drift_comparison.npz")
