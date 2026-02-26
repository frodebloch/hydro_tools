#!/usr/bin/env python3
"""
Diagnose the scaling disagreement between pdstrip and Capytaine drift forces.

Strategy:
1. Pick beam seas (mu=90 / beta=pi/2) at a SHORT wavelength (lambda=3m, kR=2.09)
   where both codes should be most reliable.
2. Compare raw numbers and check units.
3. Also check the waterline pressure on the Capytaine mesh vs what pdstrip computes.

Key facts:
- pdstrip: fxi, feta are per unit wave amplitude squared [N/m²]  
- Capytaine: using unit amplitude incident wave, so drift is also per A² [N/m²]
- Both use rho=1025, g=9.81
- Barge: R=1, L=20, semi-circular cross-section
"""
import numpy as np
import sys, os

# ============================================================
# 1. Parse pdstrip debug.out for beam seas
# ============================================================
debug_path = "/home/blofro/src/pdstrip_test/validation/run_mono/debug.out"

print("="*80)
print("PDSTRIP DRIFT FORCES (from debug.out)")
print("="*80)

import re
fnum = r'[+-]?[\d.]+(?:[EeDd][+-]?\d+)?'

blocks = []
current = None
with open(debug_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith('DRIFT_START'):
            m = re.search(rf'omega=\s*({fnum})\s+mu=\s*({fnum})', line)
            if m:
                current = {'omega': float(m.group(1)), 'mu': float(m.group(2)),
                           'wl_lines': [], 'tri_lines': []}
        elif line.startswith('WL ') and current is not None:
            current['wl_lines'].append(line)
        elif line.startswith('TRI_SUM') and current is not None:
            current['tri_lines'].append(line)
        elif line.startswith('DRIFT_TOTAL') and current is not None:
            m = re.search(
                rf'fxi=\s*({fnum})\s+feta=\s*({fnum})\s+'
                rf'fxi_WL=\s*({fnum})\s+feta_WL=\s*({fnum})\s+'
                rf'fxi_vel=\s*({fnum})\s+fxi_rot=\s*({fnum})',
                line)
            if m:
                current['fxi'] = float(m.group(1))
                current['feta'] = float(m.group(2))
                current['fxi_WL'] = float(m.group(3))
                current['feta_WL'] = float(m.group(4))
                current['fxi_vel'] = float(m.group(5))
                current['fxi_rot'] = float(m.group(6))
            blocks.append(current)
            current = None

# Barge wavelengths and headings
wavelengths = np.array([3, 4, 5, 6, 8, 10, 13, 17, 22, 28, 35, 45, 55, 70, 90])
# Headings order: -90, 0, 90, 180

print(f"\nTotal blocks parsed: {len(blocks)}")
print(f"Expected: {len(wavelengths)} × 4 headings = {len(wavelengths)*4}")

# Print beam seas (mu=90) data
print(f"\n{'lam':>5} {'omega':>7} {'feta':>12} {'feta_WL':>12} {'fxi':>12} {'fxi_WL':>12} {'fxi_vel':>12} {'fxi_rot':>12}")
print("-"*95)

for i, lam in enumerate(wavelengths):
    base = i * 4
    # mu ordering: -90=0, 0=1, 90=2, 180=3
    b = blocks[base + 2]  # mu=90
    assert abs(b['mu'] - 90.0) < 1.0, f"Expected mu=90, got {b['mu']}"
    print(f"{lam:5.0f} {b['omega']:7.3f} {b['feta']:12.2f} {b['feta_WL']:12.2f} "
          f"{b['fxi']:12.2f} {b['fxi_WL']:12.2f} {b['fxi_vel']:12.2f} {b['fxi_rot']:12.2f}")

# Also print waterline pressures for a specific wavelength
print("\n" + "="*80)
print("PDSTRIP WATERLINE PRESSURES at lambda=3m, mu=90 (beam seas)")
print("="*80)

b3_beam = blocks[0*4 + 2]  # wavelength index 0 (lam=3), heading index 2 (mu=90)
print(f"omega={b3_beam['omega']:.3f}, mu={b3_beam['mu']:.0f}")
for wl_line in b3_beam['wl_lines']:
    print(f"  {wl_line}")

# ============================================================
# 2. Load Capytaine cached results
# ============================================================
print("\n" + "="*80)
print("CAPYTAINE DRIFT FORCES (from nearfield_drift_comparison.npz)")
print("="*80)

npz_path = "/home/blofro/src/pdstrip_test/validation/nearfield_drift_comparison.npz"
data = np.load(npz_path)

cap_lam = data['wavelengths']
print(f"Capytaine wavelengths: {cap_lam}")

# Beam seas Fy
print(f"\n{'lam':>5} {'Fy_total':>12} {'Fy_wl':>12} {'Fy_vel':>12} {'Fy_rot':>12}")
print("-"*60)
for i, lam in enumerate(cap_lam):
    print(f"{lam:5.0f} {data['beam_Fy_total'][i]:12.2f} {data['beam_Fy_wl'][i]:12.2f} "
          f"{data['beam_Fy_vel'][i]:12.2f} {data['beam_Fy_rot'][i]:12.2f}")

# Head seas Fx
print(f"\n{'lam':>5} {'Fx_total':>12} {'Fx_wl':>12} {'Fx_vel':>12} {'Fx_rot':>12}")
print("-"*60)
for i, lam in enumerate(cap_lam):
    print(f"{lam:5.0f} {data['head_Fx_total'][i]:12.2f} {data['head_Fx_wl'][i]:12.2f} "
          f"{data['head_Fx_vel'][i]:12.2f} {data['head_Fx_rot'][i]:12.2f}")

# ============================================================
# 3. Direct comparison at matching wavelengths
# ============================================================
print("\n" + "="*80)
print("DIRECT COMPARISON: pdstrip Fy_geo = -feta vs Capytaine Fy")
print("(Both should be per unit wave amplitude squared, in N/m²)")
print("="*80)

rho = 1025.0
g = 9.81

print(f"\n{'lam':>5} {'kR':>6} {'pd_Fy':>10} {'cap_Fy':>12} {'ratio':>8} "
      f"{'pd_WL':>10} {'cap_WL':>12} {'r_WL':>8}")
print("-"*80)

for i, lam in enumerate(cap_lam):
    k = 2*np.pi/lam
    kR = k * 1.0  # R=1
    
    # Find pdstrip block
    pd_idx = np.where(wavelengths == lam)[0]
    if len(pd_idx) == 0:
        continue
    pd_i = pd_idx[0]
    b = blocks[pd_i * 4 + 2]  # beam seas
    
    pd_Fy = -b['feta']  # F_y_geo = -feta
    pd_WL = -b['feta_WL']
    
    cap_i = np.where(cap_lam == lam)[0][0]
    cap_Fy = data['beam_Fy_total'][cap_i]
    cap_WL = data['beam_Fy_wl'][cap_i]
    
    r = pd_Fy / cap_Fy if abs(cap_Fy) > 0.001 else float('nan')
    r_wl = pd_WL / cap_WL if abs(cap_WL) > 0.001 else float('nan')
    
    print(f"{lam:5.0f} {kR:6.3f} {pd_Fy:10.2f} {cap_Fy:12.2f} {r:8.4f} "
          f"{pd_WL:10.2f} {cap_WL:12.2f} {r_wl:8.4f}")

# ============================================================
# 4. Analytical check: waterline pressure for infinite cylinder
# ============================================================
print("\n" + "="*80)
print("ANALYTICAL CHECK: Waterline pressure for infinite 2D cylinder")
print("(MacCamy & Fuchs solution for beam seas)")
print("="*80)

# For a circular cylinder of radius R in beam seas (wave in y direction),
# the total surface pressure at the waterline (z=0, r=R) is:
# p(theta) = -rho*g*A * sum_n epsilon_n * J'_n(kR)/(H'_n(kR)) * cos(n*theta) * (-i)^n
# where the diffraction potential adds to the incident wave.
#
# For pdstrip (2D), each section solves the 2D problem independently.
# The |p|^2 at the waterline should match the 2D analytical solution.

from scipy.special import jvp, hankel1

def mccamy_fuchs_pressure_at_waterline(kR, n_terms=30):
    """
    Total complex pressure at r=R, z=0 for unit amplitude wave in y-direction.
    
    p(theta, z=0) = -rho*g * sum_n eps_n * [J_n(kR) - J'n(kR)/H'n(kR) * H_n(kR)] 
                     * cos(n*theta) * (-i)^n * exp(kz)
    
    At z=0, exp(kz)=1. For the complex amplitude (Capytaine conv exp(-iwt)):
    
    Actually, the standard 2D diffraction result for a vertical cylinder gives:
    |p/(-rho*g*A)|^2 = |sum of series|^2
    
    Let me compute |p| / (rho*g*A) at theta=0 and theta=pi (weather and lee sides).
    
    For beam seas, the incident wave propagates in +y direction.
    In polar coords, theta=pi/2 is the weather side, theta=-pi/2 is the lee side.
    """
    from scipy.special import jv as besselj, jvp as besseljp
    from scipy.special import hankel1 as H1
    
    def H1p(n, x):
        """Derivative of H1_n(x)"""
        return 0.5 * (H1(n-1, x) - H1(n+1, x)) if n > 0 else -H1(1, x)
    
    # theta = angle from x-axis; wave comes from +y so:
    # incident potential: phi_inc = -(ig/omega) * exp(iky sin(theta') r)
    # In polar: exp(iky) = sum eps_n i^n J_n(kr) cos(n*theta)
    # where theta is measured from y-axis... convention issues.
    
    # Let me just compute for a specific theta range
    theta = np.linspace(0, 2*np.pi, 360)
    
    # For a vertical cylinder in beam seas (wave in +y), the total pressure is:
    # p / (-rho g A) = sum_n epsilon_n * [J_n(kR) - Jn'(kR)/Hn'(kR) * H_n(kR)]
    #                  * cos(n * (theta - pi/2)) * (-i)^n  ... need to be careful
    
    # Actually for simplicity let's just note that pdstrip solves a 2D section
    # problem, so the comparison should be section-by-section.
    # The key question is just the ORDER OF MAGNITUDE.
    
    return None

# Instead of the full 2D analytical solution, let me just check the 
# order of magnitude of the waterline integral.
#
# For beam seas on a cylinder of radius R, the waterline is at z=0.
# The waterline contour is: y from -R to +R (going around the cylinder top).
# The outward normal in the y-direction: n_y = y/R.
#
# For long waves (kR << 1), the pressure is approximately uniform: 
#   |p| ≈ rho*g*A
# and the waterline integral:
#   F_WL_y = (1/4)*rho*g * integral of |p/(rho*g)|^2 * n_y * dl
#          = (1/4)*rho*g * integral of 1 * (y/R) * dl
#
# For the top of a unit semicircle (the waterline):
#   ∫ (y/R) dl = ∫ sin(theta) * R dtheta from 0 to pi = 2R
#   So F_WL_y = (1/4)*rho*g * (rho*g)^2/(rho*g)^2 * 2R = (1/4)*rho*g*2R = rho*g*R/2
#
# Wait, that's per unit LENGTH along x. For the full 3D barge of length L:
#   F_WL_y_total = F_WL_y_per_section * ... 
#
# Actually the waterline of the 3D barge is TWO straight lines:
#   starboard: (x, +R, 0) for x from -L/2 to +L/2, outward normal = (0, +1, 0)
#   port: (x, -R, 0) for x from +L/2 to -L/2, outward normal = (0, -1, 0)
#
# For the 3D case:
#   F_WL_y = (1/4) rho g * [ integral_starboard |eta_rel|^2 * (+1) dx 
#                            + integral_port |eta_rel|^2 * (-1) dx ]
#
# For beam seas, |eta| is the same at both sides if the wave is perpendicular,
# but the AMPLITUDE differs: weather side has higher |eta| than lee side due to
# diffraction/radiation effects.
#
# For kR >> 1 (short waves), the weather side has |eta| ≈ 2*A (reflection),
# and the lee side has |eta| ≈ 0 (shadow).
# So: F_WL_y ≈ (1/4)*rho*g * [4*L - 0] * 1 = rho*g*L  (for unit A)
#
# For our barge: rho*g*L = 1025*9.81*20 = 201,105 N/m²
#
# pdstrip beam Fy at lam=3: let me check the actual number.

print(f"\nCharacteristic force scales:")
print(f"  rho*g*L = {rho*g*20:.0f} N/m²")
print(f"  rho*g*R = {rho*g*1:.0f} N/m²")
print(f"  rho*g*R*L = {rho*g*1*20:.0f} N/m²")
print(f"  rho*g*R² = {rho*g*1:.0f} N/m²")

# ============================================================
# 5. Check: is Capytaine's incident wave amplitude actually 1?
# ============================================================
print("\n" + "="*80)
print("CHECK: Capytaine incident wave amplitude convention")
print("="*80)
print("""
Capytaine DiffractionProblem uses unit incident wave amplitude by default.
The potential is phi_inc = -(ig/omega) * exp(ky*sin(beta) + kx*cos(beta)) * exp(kz)
with amplitude = g/omega (in potential) which gives eta = (iw/g)*phi = 1 at the origin.

So both pdstrip and Capytaine use unit amplitude. The forces should be directly comparable.
""")

# ============================================================
# 6. Check if there's a factor-of-2 issue in pdstrip
# ============================================================
print("="*80)
print("POSSIBLE FACTOR-OF-2 ISSUES")
print("="*80)
print("""
pdstrip formula at line 2703:
  dfxistb = 0.25 * |pres(1)|^2 * dystb / rho / g

The 0.25 = 1/4 comes from:
  Drift = (1/2) * <|eta_rel|^2> * n * dl
  <|eta|^2> = (1/2)|eta_hat|^2  (time average of cos^2)
  So: Drift = (1/4) |eta_hat|^2 * n * dl  ← correct

  But wait: eta = p / (rho*g) at z=0, so |eta|^2 = |p|^2 / (rho*g)^2
  Drift = (1/4) |p|^2 / (rho*g)^2 * (rho*g) * n * dl
        = (1/4) |p|^2 / (rho*g) * n * dl  ← matches the code (0.25 * |p|^2 * dl / rho / g)

Capytaine formula at line 396:
  F_wl = 0.25 * rho * g * |eta_rel|^2 * n * dl
  
  Same: (1/4) * rho*g * |eta|^2 * n * dl = (1/4) * |p|^2/(rho*g) * n * dl ← same formula

So no factor-of-2 difference in the WL term.
""")

# ============================================================
# 7. Let me check if there's a LENGTH issue  
# ============================================================
print("="*80)
print("KEY QUESTION: Does pdstrip sum over all sections correctly?")
print("="*80)
print("""
pdstrip drift is accumulated in a loop over sections (ise1).
Each section contributes:
  feta += dfeta  (waterline)
  fxi += dfxistb + dfxiprt  (waterline)
  feta += from triangles  (velocity + rotation terms)
  fxi += from triangles  (velocity + rotation terms)

The waterline term for feta is:
  dfeta = 0.25 * dx2 * (-|pres(1)|^2 + |pres(npres)|^2) / rho / g

where dx2 = length of section in x (distance between section midpoints).
So the waterline term integrates along x, accumulating the total force.

The triangle (hull) terms use the actual triangle areas (flvec has area built in).

So pdstrip gives the TOTAL force on the entire hull, per unit amplitude squared.

Capytaine also computes the TOTAL force on the entire 3D mesh.

Both should give the same dimensional result if the physics agrees.
""")

# Check total waterline length
print("pdstrip section lengths (from WL lines):")
for wl_line in b3_beam['wl_lines']:
    m = re.search(rf'sec=\s*(\d+)\s+dx2=\s*({fnum})', wl_line)
    if m:
        sec = int(m.group(1))
        dx2 = float(m.group(2))
        print(f"  sec={sec}: dx2={dx2:.3f}m")

print(f"\nTotal length from dx2 sum: should be ≈ {20.0}m for L=20 barge")

# Sum dx2 values
dx2_sum = 0
for wl_line in b3_beam['wl_lines']:
    m = re.search(rf'dx2=\s*({fnum})', wl_line)
    if m:
        dx2_sum += float(m.group(1))
print(f"Actual sum(dx2) = {dx2_sum:.3f}m")
