"""
Deeper analysis of the Fy discrepancy.

The EOM is: (-omega^2*(M+A) + i*omega*B + C)*eta = F_exc
=> F_hydro_total = -omega^2*M*eta  (Newton's 2nd law)
=> F_hydro_total = F_exc + (-omega^2*A + i*omega*B)*eta + C*eta

The panel-integrated pressure force should give F_hydro_total.
If it doesn't match -omega^2*M*eta, either:
1. The panel integration is inconsistent with the EOM coefficients (A, B, C)
2. The EOM coefficients don't match what the panels would give

Key question: is the Fy discrepancy proportional to anything we know?
Let's check if it correlates with the RESTORING force C*eta.
"""
import re
import numpy as np
from collections import defaultdict

debug_file = 'debug.out'

# Parse DRIFT_START + FORCE_CHECK + DRIFT_SWAY together
results = []
current = {}

with open(debug_file) as f:
    for line in f:
        if 'DRIFT_START' in line:
            m = re.search(r'omega=\s*([\d.]+)\s+mu=\s*([-\d.]+)', line)
            if m:
                current = {'omega': float(m.group(1)), 'mu': float(m.group(2))}
        elif 'FORCE_CHECK' in line and current:
            parts = line.split()
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except ValueError:
                    continue
            if len(vals) == 8:
                current['fy_pres'] = complex(vals[0], vals[1])
                current['fy_newton'] = complex(vals[2], vals[3])
                current['fz_pres'] = complex(vals[4], vals[5])
                current['fz_newton'] = complex(vals[6], vals[7])
                results.append(current)
                current = {}

# Filter beam seas V=0 
beam_90 = [r for r in results if abs(r['mu'] - 90.0) < 0.1]
by_omega = defaultdict(list)
for r in beam_90:
    by_omega[r['omega']].append(r)

# The Fy discrepancy: dFy = Fy_pres - Fy_newton
# If this is a restoring force issue, dFy should be proportional to C_22*eta_2
# At beam seas, the restoring force in sway is C_22*eta_2 = 0 (no sway restoring in standard strip theory!)
# 
# But wait - the issue might be that the PANEL integration of pst gives a DIFFERENT
# restoring force than the restorematr used in the EOM.
# restorematr is computed from waterplane area integrals (separate code)
# while the panel pst integration uses triangle panels
#
# Key insight: the Newton prediction uses -omega^2*M*eta
# which equals F_exc - (-omega^2*A + i*omega*B)*eta - C*eta (from EOM)
#
# But the panel integration gives: F_exc(panels) + (-omega^2*A(panels) + i*omega*B(panels))*eta + C(panels)*eta
#
# If the PANEL-based C(panels) differs from EOM-based C, that's the discrepancy.
#
# Actually, let's think about this differently.
# F_pres = integral of p*n*dS over ALL triangles
# p = p_wave + p_rad*eta + p_st*eta
# F_pres = F_exc(FK+diff) + (-omega^2*A_panel + i*omega*B_panel)*eta + C_panel*eta
#
# Newton says: F_pres = -omega^2*M*eta
# So: F_exc + (-omega^2*A + i*omega*B)*eta + C*eta = -omega^2*M*eta
# => F_exc = -omega^2*(M+A)*eta + i*omega*B*eta + C*eta  (the EOM)
#
# The DISCREPANCY is:
# dF = F_pres(panels) - (-omega^2*M*eta)
#    = [F_exc_panel - F_exc_EOM] + [(-omega^2*A_panel - (-omega^2*A_EOM))]*eta + ...
#    = differences in excitation, added mass, damping, restoring between panels and EOM
#
# Since restoring in strip theory is computed from waterplane integrals, and the panel
# triangles also cover the hull surface, the pst*n integration should give C(panels).
# If C(panels) != C(EOM), that creates a systematic discrepancy.

# Let's also extract motion data to check the cross-term
# Parse motion from pdstrip.out (RAOs)
print("ANALYSIS: What does the Fy discrepancy correlate with?")
print("="*80)

g = 9.81
rho = 1025.0
Lpp = 328.2

# Extract the discrepancy and check if it grows with frequency in a way
# consistent with added mass discrepancy vs restoring vs excitation
print(f"\n{'omega':>6} {'lam/L':>6} {'|dFy|':>12} {'|Fy_n|':>12} {'dFy/omega^2':>14} {'|dFy|/omega^0':>14}")
print("-"*80)

discrepancies = []
for omega in sorted(by_omega.keys(), reverse=True):
    entries = by_omega[omega]
    r = entries[0]
    
    lam = 2*np.pi*g / omega**2
    lam_over_L = lam / Lpp
    
    dfy = r['fy_pres'] - r['fy_newton']
    
    discrepancies.append((omega, lam_over_L, dfy, r['fy_newton'], r['fy_pres']))

# Check scaling: if dF ~ omega^2 * something, it's added mass
# if dF ~ omega^0 (constant), it's restoring
# if dF ~ omega^1, it's damping

for omega, lam_L, dfy, fy_n, fy_p in discrepancies:
    print(f"{omega:6.3f} {lam_L:6.2f} {abs(dfy):12.0f} {abs(fy_n):12.0f} {abs(dfy)/omega**2:14.0f} {abs(dfy):14.0f}")

# Now let's also look at Fy_pres more carefully
# The HUGE growth of Fy_pres at roll resonance (omega~0.38) suggests it tracks the roll motion
# eta_4 (roll) has a resonance there
# The hydrostatic restoring force from pst includes rho*g*y*eta_4 integrated over all panels
# This gives a LATERAL force from roll!
# 
# Specifically: integral of pst_roll * n_y * dS = integral of rho*g*y*eta_4 * n_y * dS
# This is NOT zero for a hull with asymmetric y-z geometry per section
# 
# Wait, actually for a port-starboard symmetric hull, integral of y*n_y over the full hull IS non-zero
# because it's related to the waterplane second moment of area (I_yy)
# But this contributes to the roll RESTORING moment, not the sway force...
# 
# Actually, the integral of pst(:,4,:)*n_y over the hull gives the (2,4) coupling: C_24
# In strip theory, C_24 = rho*g * integral of y*n_y dS
# For a symmetric hull, this is related to the waterplane area ... hmm.
# 
# Let's check: what is the pst contribution to sway from roll?
# F_y_pst_roll = integral of rho*g*y * eta_4 * n_y dS
# By Gauss's theorem: integral of y*n_y dS = integral of div(y*e_y) dV = Volume
# So F_y_pst_roll = rho*g*V*eta_4 where V is displaced volume
# This is a BUOYANCY FORCE: when the ship rolls, the displaced volume shifts laterally
# and creates a restoring force in sway!
# 
# But in the EOM, the restoring matrix C_24 should capture this.
# If it DOESN'T (because restorematr is computed differently), that's the discrepancy!

print("\n\n=== KEY HYPOTHESIS: The panel pst integration gives different restoring forces than restorematr ===")
print("Specifically, the cross-coupling terms C_24 (sway-roll) and C_26 (sway-yaw) may differ.")
print("The pst contribution to sway grows with roll amplitude, which peaks at roll resonance.")

# Let's check: at roll resonance, eta_4 is large. The extra sway force from pst would be:
# dF_y ~ rho*g * V * eta_4  (from C_24 discrepancy)
# V = mass/rho = 320e6/1025 = 312195 m^3
mass = 320e6  # kg
V_disp = mass / rho
print(f"\nDisplaced volume: {V_disp:.0f} m^3")
print(f"rho*g*V = {rho*g*V_disp:.3e} N")

# Now let's parse motion data to get eta_4 at beam seas
# We need to read the pdstrip.out or debug.out for motion data
# Let me check what motion info is in debug.out
print("\n\nChecking for motion data in debug output...")
