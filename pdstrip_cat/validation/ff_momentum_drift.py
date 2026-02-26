#!/usr/bin/env python3
"""
Correct far-field drift force computation using Maruo's formula
with proper treatment of the incident-scattered cross term.

Reference: Faltinsen (1990) "Sea Loads on Ships and Offshore Structures"
Chapter 6, Section 6.3

The mean drift force from momentum flux conservation is:

F_y = ρg Cg ∫₀²π |A_d(θ)|² sinθ dθ + cross_term

where Cg = g/(2ω) is the group velocity (deep water),
A_d(θ) is the disturbance (scattered+radiated) far-field wave amplitude,
defined by: η_d(r,θ) = A_d(θ) exp(ikr)/√r as r→∞

The cross term comes from the interaction of the incident and scattered waves.
Using the approach of extracting the total field momentum flux and subtracting
the incident wave momentum flux.

Alternative approach: use the formula derived by Newman (1967):
For a FIXED body in deep water (only scattering, no radiation):

F̄ = -ρg Cg ∫₀²π |A_total(θ)|² ê(θ) dθ + incident_wave_momentum_rate

where the incident wave carries momentum flux ρg A² Cg / 2 per unit width
in the direction of propagation.

Actually, the simplest CORRECT approach for drift force on a fixed body:

Method: Evaluate the total field (incident + scattered) at the field-point ring,
compute the far-field wave amplitude of the total field, and use:

F_y = ρg/(4k) × ∫₀²π [ |A_total(θ)|² - |A_inc(θ)|² ] sinθ dθ

But A_inc is a plane wave and doesn't have a standard 1/√r far-field form...
So we need to be more careful.

CORRECT APPROACH: Use the momentum flux directly.
The time-averaged y-momentum flux through a vertical cylinder of radius R is:

F_y = -∫₀²π ∫_{-∞}^{0} [<p> sinθ + ρ<v_y v_r>] R dz dθ

For potential flow:
- <p> = -ρ<|∇φ|²>/2 (mean dynamic pressure from Bernoulli)
- <v_y v_r> = <(∂φ/∂y)(∂φ/∂r)>

This gets complicated. Let me use the ESTABLISHED result instead.

The correct Maruo formula (deep water, zero speed) is:
F = ρg/(2k) ∫₀²π |H(θ)|² ê(θ) dθ

where H(θ) is the Kochin function of the disturbance field, properly normalized
so that B_jj = ρ·ω·k/(4π) ∫|H_j|² dθ (this is Newman's normalization).

BUT we already verified the a(θ) extraction via damping:
B = ρg²/(2ω³) ∫|a|² dθ

From Newman's B formula: B = ρωk/(4π) ∫|H|² dθ
Equating: |H|² = (2πg²)/(ω⁴k) |a|²

Newman's drift: F = ρg/(2k) ∫|H|² sinθ dθ + cross term
            = ρg/(2k) × (2πg²)/(ω⁴k) ∫|a|² sinθ dθ + cross
            = ρπg³/(ω⁴k²) ∫|a|² sinθ dθ + cross
            = ... messy

Let me just use the DIRECT momentum flux calculation numerically, evaluating 
at a large-radius control surface.
"""

import numpy as np
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_potential, airy_waves_velocity
import logging

cpt.set_logging(logging.WARNING)

R = 1.0; L = 20.0; rho = 1025.0; g = 9.81

# Build body
mesh_full = cpt.mesh_horizontal_cylinder(
    length=L, radius=R, center=(0, 0, 0),
    resolution=(10, 40, 50), name="hull")
hull_mesh = mesh_full.immersed_part()
lid = hull_mesh.generate_lid(z=-0.01)
body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name="hull")
body.center_of_mass = np.array([0, 0, -4*R/(3*np.pi)])
body.rotation_center = body.center_of_mass
body.add_all_rigid_body_dofs()

solver = cpt.BEMSolver()

beta = np.pi / 2  # beam seas

# ============================================================
# METHOD: Numerical momentum flux through a control cylinder
# ============================================================
# The time-averaged y-force from momentum flux through a cylinder at r=R_c:
#
# F_y = ∫₀²π dθ ∫_{-∞}^{0} dz × R_c × [
#    -<p_dynamic> sinθ - ρ <u_y u_r> 
#  ] - 0   (mean free surface term vanishes at z=0 for linear waves)
#
# Actually, for the complete momentum balance:
# F_y = -∮_S [ <p> n_y + ρ<v_y v_n> ] dS  - ρg/(4) ∮_WL |η|² n_y dl  
#
# where S is the control surface (vertical cylinder) and the WL term comes from
# the free-surface contribution.
#
# For deep water, using complex potential:
# φ = φ_total = φ_inc + φ_scatter
# <p_dyn> = -ρ/4 Σ |∇φ|²  (time average of -ρ|∇φ|²/2 for cos²)
# <v_y v_r> = (1/2) Re(v̂_y v̂_r*)
#
# All quantities evaluated at (R_c cosθ, R_c sinθ, z) and integrated.
#
# For deep water, φ ~ exp(kz), so the z-integral gives 1/(2k).
#
# The z-integral of |∇φ|² × exp(0) ... wait, that's not right because
# |∇φ|² involves products of exp(kz) terms.
#
# Let me think about this differently.
# 
# For deep water waves, all field quantities decay as exp(kz).
# The velocity potential: φ(x,y,z,t) = Re{ φ̂(x,y) exp(kz) exp(-iωt) }
# The velocity: ∇φ = (∂φ̂/∂x, ∂φ̂/∂y, kφ̂) × exp(kz) × exp(-iωt)
# 
# Time averages of products of exp(-iωt) terms:
# <Re(A e^{-iωt}) Re(B e^{-iωt})> = (1/2) Re(A B*)
#
# Products of z-dependent terms:
# exp(kz) × exp(kz) = exp(2kz) → integral from -∞ to 0 = 1/(2k)
#
# So the z-integrated momentum flux becomes a 1D integral in θ at the cylinder.

# The practical formula (from Faltinsen 1990, simplified for deep water):
# 
# At large r, the TOTAL potential in polar coords (r,θ) at z=0 is:
#
# φ̂(r,θ,0) = φ̂_inc + φ̂_d 
#            = -(ig/ω) exp(ikr cosα) + -(ig/ω) a_d(θ) exp(ikr)/√r + ...
#
# where α = θ - β (angle relative to wave direction).
# (Here I use the convention φ = -(ig/ω)η with η = wave elevation.)
#
# The incident wave: φ̂_inc = -(ig/ω) exp(ik(x cosβ + y sinβ))
#                            = -(ig/ω) exp(ikr cos(θ-β))
#
# The scattered wave: φ̂_d ~ -(ig/ω) a_d(θ) exp(ikr)/√r
#
# The y-drift from the far-field amplitude a_d(θ):
# Using stationary phase and momentum flux (see Faltinsen eq 6.26 adapted):
#
# F_y = (ρg)/(4k) ∫₀²π |a_d|² sinθ dθ 
#     + (ρg)/k × Im{ a_d(β) × sin(β) }  × √(πk/(2)) × exp(iπ/4)
#     ... no, this is getting circular.
#
# Let me just compute it numerically using velocity at the field-point ring.

def compute_ff_drift_numerical(omega, beta, solver, body, n_theta=360, r_field=1000.0, nz=100):
    """
    Compute drift force by numerical momentum flux integration.
    
    Evaluate velocity potential at points on a control cylinder at radius r_field,
    and integrate the time-averaged momentum flux.
    """
    k = omega**2 / g
    
    # Solve diffraction (fixed body)
    diff_prob = cpt.DiffractionProblem(
        body=body, wave_direction=beta, omega=omega,
        water_depth=np.inf, rho=rho, g=g)
    diff_result = solver.solve(diff_prob, keep_details=True)
    
    # Evaluation points: (r_field*cosθ, r_field*sinθ, z_j)
    theta_arr = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    dtheta = 2*np.pi / n_theta
    
    # z-levels: from 0 to deep enough that exp(2kz) is negligible
    # Need z such that exp(2kz) < eps → z < ln(eps)/(2k)
    z_min = np.log(1e-6) / (2*k)  # exp(2kz_min) = 1e-6
    z_arr = np.linspace(z_min, 0, nz)
    dz = z_arr[1] - z_arr[0]  # positive (going upward)
    
    # We need the TOTAL velocity at each point
    # For the control surface integral, we need:
    # - v_r = radial velocity
    # - v_θ = tangential velocity  
    # - v_z = vertical velocity
    # - pressure p
    #
    # At each (θ, z): evaluate φ_total = φ_inc + φ_scat
    # Then compute velocity components.
    #
    # But evaluating at many (θ,z) points is expensive.
    # Since for deep water φ ~ f(x,y) exp(kz), we can evaluate at z=0
    # and scale by exp(kz), then integrate analytically in z.
    
    # Evaluate scattered potential at z=0 on the ring
    pts_z0 = np.column_stack([
        r_field * np.cos(theta_arr),
        r_field * np.sin(theta_arr),
        np.zeros(n_theta)])
    
    phi_scat_z0 = solver.compute_potential(pts_z0, diff_result)
    
    # Incident potential at z=0
    phi_inc_z0 = airy_waves_potential(pts_z0, diff_prob)
    
    phi_total_z0 = phi_inc_z0 + phi_scat_z0
    
    # Incident potential: φ_inc = -(ig/ω) exp(ik(x cosβ + y sinβ)) exp(kz)
    # At z=0: φ_inc(z=0) = -(ig/ω) exp(ik r cos(θ-β))
    
    # Total velocity at z=0 in Cartesian:
    vel_scat_z0 = solver.compute_velocity(pts_z0, diff_result)
    vel_inc_z0 = airy_waves_velocity(pts_z0, diff_prob)
    vel_total_z0 = vel_inc_z0 + vel_scat_z0  # (n_theta, 3) complex
    
    # Convert to polar: v_r = v_x cosθ + v_y sinθ
    #                    v_θ = -v_x sinθ + v_y cosθ
    ct = np.cos(theta_arr); st = np.sin(theta_arr)
    vr_z0 = vel_total_z0[:,0]*ct + vel_total_z0[:,1]*st
    vy_z0 = vel_total_z0[:,1]  # keep Cartesian y-velocity
    vz_z0 = vel_total_z0[:,2]
    
    # For deep water: φ(x,y,z) = φ(x,y,0) exp(kz)
    #   v_x(z) = v_x(0) exp(kz)
    #   v_y(z) = v_y(0) exp(kz)  
    #   v_z(z) = k φ(0) exp(kz) ... wait, v_z = ∂φ/∂z = k φ exp(kz) for the exp(kz) part
    
    # The z-integral of time-averaged products:
    # ∫_{-∞}^0 <Re(A exp(kz) e^{-iωt}) Re(B exp(kz) e^{-iωt})> dz
    # = ∫_{-∞}^0 (1/2) Re(AB*) exp(2kz) dz
    # = (1/2) Re(AB*) × 1/(2k)
    # = Re(AB*) / (4k)
    
    # The momentum flux through the control cylinder (outward pointing n_r = ê_r):
    # F_y = -∫_S [ p n_y + ρ u_y u_r ] dS - ρg/4 ∮_WL |η|² n_y dl
    #
    # where S is the vertical part (cylinder wall) and WL is the intersection with z=0.
    
    # On the cylinder wall at r=R_c:
    #   dS = R_c dθ dz
    #   n = ê_r (outward)
    #   n_y = sinθ
    
    # Term 1: -∫∫ p n_y dS
    # <p> = -ρ/2 <|∇φ|²> = -ρ/4 |∇φ̂|²  (for exp(-iωt) convention)
    # But actually, for the MEAN pressure (Bernoulli):
    # p_mean = -ρ/2 <(∂φ/∂t)²/... > ... no.
    # The time-averaged dynamic pressure for linear potential flow:
    # <p_dyn> = -ρ/2 <|∇φ|²> = -ρ/4 × (|v̂_x|² + |v̂_y|² + |v̂_z|²) exp(2kz)
    #
    # Wait, that's the second-order term. The full time-averaged Bernoulli equation:
    # <p> = -ρ<∂φ/∂t> - ρg z - ρ/2<|∇φ|²>
    # The first term: <∂φ/∂t> = 0 for harmonic waves (time average of sin is 0)
    # The second term: -ρgz is the hydrostatic term, doesn't contribute to momentum flux
    # (actually it does through the free surface, but that's separate)
    # So: <p_dyn> = -ρ/4 |v̂|² exp(2kz)
    
    # ∫_{-∞}^0 <p_dyn> dz = -ρ/4 × 1/(2k) × |v̂(z=0)|²
    
    # But this isn't right either. Let me be more careful about the Bernoulli equation.
    # 
    # The EXACT mean pressure from Bernoulli: 
    # p = -ρ ∂φ/∂t - ρgz - (1/2)ρ|∇φ|²
    # Time average: <p> = -ρgz - (1/2)ρ<|∇φ|²>
    # The -ρgz integrates to give a hydrostatic term that cancels in the overall balance.
    # The dynamic mean pressure: <p_dyn> = -(1/2)ρ<|∇φ|²>
    #   = -(ρ/4)|∇φ̂|² exp(2kz)  for complex amplitudes
    
    # Hmm wait - I think for the control volume momentum balance, the relevant quantity is:
    # F_y = -∫_S (p n_y + ρ v_y v_n) dS  evaluated as time average
    #      = -∫_S (<p> n_y + ρ <v_y v_r>) R_c dθ dz
    #
    # <p> at a point includes both the fluctuating and mean parts.
    # For waves: p = -ρ ∂φ/∂t - ρgz - (1/2)ρ|∇φ|²
    # <p> = -ρgz - (1/4)ρ|∇φ̂|² exp(2kz)
    #
    # <v_y v_r> = (1/2) Re(v̂_y v̂_r*) exp(2kz)
    
    # So the z-integral of the dynamic part:
    # ∫_{-∞}^0 [-(1/4)ρ|∇φ̂|² exp(2kz)] × sinθ × R_c dz
    # = -(1/4)ρ × 1/(2k) × |v̂(z=0)|² × sinθ × R_c
    # = -(ρ/(8k)) × (|v̂_x|² + |v̂_y|² + |v̂_z|²) × sinθ × R_c
    
    # ∫_{-∞}^0 [(1/2) Re(v̂_y v̂_r*) exp(2kz)] × R_c dz
    # = (1/2) Re(v̂_y(0) v̂_r(0)*) × 1/(2k) × R_c
    # = Re(v̂_y v̂_r*) / (4k) × R_c
    
    # Total y-force from vertical cylinder wall:
    # F_y_wall = -R_c × dθ × 1/(4k) × Σ_θ [
    #    -ρ/2 × |v̂|² × sinθ + ρ × Re(v̂_y v̂_r*) ]
    
    vel_sq = np.abs(vel_total_z0[:,0])**2 + np.abs(vel_total_z0[:,1])**2 + np.abs(vel_total_z0[:,2])**2
    vy_vr_star = vel_total_z0[:,1] * np.conj(vr_z0)
    
    F_y_wall = -r_field * dtheta / (4*k) * np.sum(
        -rho/2 * vel_sq * st + rho * np.real(vy_vr_star))
    
    # Free surface contribution (the WL term at the control surface):
    # The mean free surface elevation has a second-order component:
    # <η²> contributes to the momentum balance.
    # F_y_fs = -ρg/(4) ∮ |η̂|² sinθ R_c dθ
    #
    # η̂ = (iω/g) φ̂(z=0)
    
    eta_hat = (1j * omega / g) * phi_total_z0
    eta_sq = np.abs(eta_hat)**2
    
    F_y_fs = -rho * g / 4 * r_field * dtheta * np.sum(eta_sq * st)
    
    F_y_total = F_y_wall + F_y_fs
    
    # Also compute: subtract the incident wave contribution
    # For incident wave only:
    vel_sq_inc = np.sum(np.abs(vel_inc_z0)**2, axis=1)
    vr_inc = vel_inc_z0[:,0]*ct + vel_inc_z0[:,1]*st
    vy_vr_inc = vel_inc_z0[:,1] * np.conj(vr_inc)
    eta_inc = (1j * omega / g) * phi_inc_z0
    
    F_y_wall_inc = -r_field * dtheta / (4*k) * np.sum(
        -rho/2 * vel_sq_inc * st + rho * np.real(vy_vr_inc))
    F_y_fs_inc = -rho * g / 4 * r_field * dtheta * np.sum(np.abs(eta_inc)**2 * st)
    F_y_inc = F_y_wall_inc + F_y_fs_inc
    
    return {
        'F_y_total': F_y_total,
        'F_y_wall': F_y_wall,
        'F_y_fs': F_y_fs,
        'F_y_inc': F_y_inc,
        'F_y_drift': F_y_total - F_y_inc,
    }


# ============================================================
# Run for multiple wavelengths
# ============================================================
wavelengths = [3, 5, 10, 22, 55]

print(f"\n{'='*100}")
print(f"FAR-FIELD DRIFT: Numerical momentum flux through control cylinder")
print(f"Fixed body (no motion), beam seas (β=π/2)")
print(f"{'='*100}")
print(f"\n{'lam':>5} {'kR':>6} {'F_total':>12} {'F_inc':>12} {'F_drift':>12} "
      f"{'F_wall':>12} {'F_fs':>12}")
print("-"*80)

for lam in wavelengths:
    k = 2*np.pi/lam
    omega = np.sqrt(k*g)
    kR = k*R
    
    r_field = max(5000.0, 200*lam)
    result = compute_ff_drift_numerical(omega, beta, solver, body,
                                         n_theta=720, r_field=r_field)
    
    print(f"{lam:5.0f} {kR:6.3f} {result['F_y_total']:12.1f} {result['F_y_inc']:12.1f} "
          f"{result['F_y_drift']:12.1f} {result['F_y_wall']:12.1f} {result['F_y_fs']:12.1f}")

# ============================================================
# Compare with pdstrip
# ============================================================
print(f"\n\n{'='*100}")
print(f"COMPARISON: pdstrip feta vs far-field drift")
print(f"pdstrip Fy_geo = -feta (internal y → starboard)")
print(f"{'='*100}")

import re
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
            m = re.search(rf'fxi=\s*({fnum})\s+feta=\s*({fnum})', line)
            if m:
                current['fxi'] = float(m.group(1)); current['feta'] = float(m.group(2))
                blocks.append(current); current = None

pd_wavelengths = np.array([3, 4, 5, 6, 8, 10, 13, 17, 22, 28, 35, 45, 55, 70, 90])

print(f"\n{'lam':>5} {'kR':>6} {'pd_Fy':>12} {'FF_drift':>12} {'ratio':>8}")
print("-"*50)

for lam in wavelengths:
    k = 2*np.pi/lam; omega = np.sqrt(k*g); kR = k*R
    
    pd_idx = np.where(pd_wavelengths == lam)[0][0]
    b = blocks[pd_idx * 4 + 2]  # mu=90
    pd_Fy = -b['feta']
    
    r_field = max(5000.0, 200*lam)
    result = compute_ff_drift_numerical(omega, beta, solver, body,
                                         n_theta=720, r_field=r_field)
    ff_drift = result['F_y_drift']
    
    r = pd_Fy / ff_drift if abs(ff_drift) > 0.1 else float('nan')
    print(f"{lam:5.0f} {kR:6.3f} {pd_Fy:12.1f} {ff_drift:12.1f} {r:8.3f}")
