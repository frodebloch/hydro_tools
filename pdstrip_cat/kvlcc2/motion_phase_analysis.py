#!/usr/bin/env python3
"""
Analyze motion RAO phases at beam seas (mu=90°, V=0) near roll resonance.
Check if there are phase inconsistencies that could cause the rotation term blow-up.
"""
import re
import numpy as np

# Parse pdstrip.out for motion data
# Format: after "wave angle  90.0" and "speed   0.00"
# Translation line: Re(1) Im(1) Abs(1) Re(2) Im(2) Abs(2) Re(3) Im(3) Abs(3)
# Rotation/k line: same format for DOFs 4,5,6

# KVLCC2 parameters
Lpp = 328.2; g = 9.81; rho = 1025.0

blocks = []
with open('pdstrip.out') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i]
    # Look for the frequency/heading header line
    m = re.match(r'\s*Wave circ\. frequency\s+([\d.]+)\s+encounter frequ\.\s+([\d.]+)\s+wave length\s+([\d.]+)\s+wave number\s+([\d.]+)\s+wave angle\s+([\d.-]+)', line)
    if m:
        om = float(m.group(1))
        ome = float(m.group(2))
        wl = float(m.group(3))
        wn = float(m.group(4))
        mu = float(m.group(5))
        
        # Next line: speed
        if i+1 < len(lines):
            m2 = re.match(r'\s*speed\s+([\d.]+)', lines[i+1])
            if m2:
                vs = float(m2.group(1))
                
                # Skip header line, then Translation, then Rotation/k
                if i+3 < len(lines) and i+4 < len(lines):
                    tline = lines[i+3].strip()
                    rline = lines[i+4].strip()
                    
                    # Parse Translation line
                    tm = re.findall(r'[-\d.]+', tline)
                    # Parse Rotation/k line  
                    rm = re.findall(r'[-\d.]+', rline)
                    
                    if len(tm) >= 9 and len(rm) >= 9:
                        # Skip the label
                        tvals = [float(x) for x in tm[-9:]]
                        rvals = [float(x) for x in rm[-9:]]
                        
                        block = {
                            'om': om, 'ome': ome, 'wl': wl, 'wn': wn, 'mu': mu, 'vs': vs,
                            'eta1': complex(tvals[0], tvals[1]),  # surge
                            'eta2': complex(tvals[3], tvals[4]),  # sway
                            'eta3': complex(tvals[6], tvals[7]),  # heave
                            'eta4_k': complex(rvals[0], rvals[1]),  # roll/k
                            'eta5_k': complex(rvals[3], rvals[4]),  # pitch/k
                            'eta6_k': complex(rvals[6], rvals[7]),  # yaw/k
                        }
                        blocks.append(block)
    i += 1

# Filter for beam seas V=0
beam_v0 = [b for b in blocks if abs(b['mu'] - 90.0) < 0.1 and abs(b['vs']) < 0.1]
beam_v0.sort(key=lambda b: b['om'])

print("="*120)
print("MOTION RAOs AT BEAM SEAS (mu=90°, V=0) — KVLCC2")
print("Internal coords: x=forward, y=starboard, z=down")
print("Positive roll (eta4) = starboard down")
print("Positive heave (eta3) = downward")
print("="*120)
print(f"{'omega':>6} {'lam/L':>6} | {'|eta2|':>8} {'ph(eta2)':>9} | {'|eta3|':>8} {'ph(eta3)':>9} | "
      f"{'|eta4|':>8} {'ph(eta4)':>9} | {'ph(3-4)':>8} {'Re[e3*c(e4)]':>13} {'|e3||e4|':>10}")
print("-"*120)

for b in beam_v0:
    om = b['om']
    k = om**2 / g
    lam_L = 2*np.pi / k / Lpp
    
    eta2 = b['eta2']
    eta3 = b['eta3']
    eta4 = b['eta4_k'] * k  # Convert from /k to actual radians
    
    ph2 = np.degrees(np.angle(eta2))
    ph3 = np.degrees(np.angle(eta3))
    ph4 = np.degrees(np.angle(eta4))
    
    # Phase difference between heave and roll
    ph_diff = np.degrees(np.angle(eta3 * np.conj(eta4)))
    
    # The cross-term that drives the rotation blow-up
    cross_term = np.real(eta3 * np.conj(eta4))
    prod_amp = abs(eta3) * abs(eta4)
    
    if lam_L > 0.2 and lam_L < 3.0:
        print(f"{om:6.3f} {lam_L:6.2f} | {abs(eta2):8.4f} {ph2:9.1f}° | {abs(eta3):8.4f} {ph3:9.1f}° | "
              f"{abs(eta4):8.5f} {ph4:9.1f}° | {ph_diff:8.1f}° {cross_term:13.6f} {prod_amp:10.6f}")

# Now check the key cross-term in the rotation drift
# The rotation term includes: Re[pst_3 * eta3 * conj(eta4)] where pst_3 = rho*g
# This creates a term proportional to Re[eta3 * conj(eta4)]
# If eta3 and eta4 are 90° out of phase, this term should be zero
# If they are in phase, it's maximized

print("\n\n" + "="*120)
print("HEAVE-ROLL CROSS-TERM ANALYSIS")
print("The rotation term blow-up comes from Re[rho*g*eta3 * conj(eta4)] * integral(y * flvec_z)")
print("="*120)
print(f"{'omega':>6} {'lam/L':>6} {'|eta3|':>8} {'|eta4|':>8} {'phase_diff':>11} {'Re[e3*c(e4)]':>13} "
      f"{'|e3||e4|cos':>12} {'cos_factor':>11}")
print("-"*100)

for b in beam_v0:
    om = b['om']
    k = om**2 / g
    lam_L = 2*np.pi / k / Lpp
    eta3 = b['eta3']
    eta4 = b['eta4_k'] * k
    
    ph_diff = np.angle(eta3 * np.conj(eta4))
    cross = np.real(eta3 * np.conj(eta4))
    prod = abs(eta3) * abs(eta4)
    cos_f = np.cos(ph_diff) if prod > 1e-10 else 0
    
    if lam_L > 0.5 and lam_L < 2.5:
        print(f"{om:6.3f} {lam_L:6.2f} {abs(eta3):8.4f} {abs(eta4):8.5f} {np.degrees(ph_diff):11.1f}° "
              f"{cross:13.6f} {prod*cos_f:12.6f} {cos_f:11.3f}")

# Also show the sway-roll coupling
print("\n\n" + "="*120)
print("SWAY-ROLL PHASE RELATIONSHIP")
print("="*120)
print(f"{'omega':>6} {'lam/L':>6} {'|eta2|':>8} {'ph(eta2)':>9} {'|eta4|':>8} {'ph(eta4)':>9} {'ph(2-4)':>8}")
print("-"*80)

for b in beam_v0:
    om = b['om']
    k = om**2 / g
    lam_L = 2*np.pi / k / Lpp
    eta2 = b['eta2']
    eta4 = b['eta4_k'] * k
    
    ph2 = np.degrees(np.angle(eta2))
    ph4 = np.degrees(np.angle(eta4))
    ph_diff = np.degrees(np.angle(eta2 * np.conj(eta4)))
    
    if lam_L > 0.5 and lam_L < 2.5:
        print(f"{om:6.3f} {lam_L:6.2f} {abs(eta2):8.4f} {ph2:9.1f}° {abs(eta4):8.5f} {ph4:9.1f}° {ph_diff:8.1f}°")

# Finally, check whether the equation of motion is self-consistent
# At roll resonance, the restoring force = -c44*eta4
# The hydrostatic pressure gives a roll moment = integral(y * rho*g*y*eta4 * n_z * dS)
# which should equal c44*eta4 (from waterplane area)
print("\n\n" + "="*120)
print("CONSISTENCY CHECK: Does pst give the same restoring force as restorematr?")
print("If not, the drift force pst contribution is inconsistent with the EoM.")
print("="*120)
print()
print("The restoring matrix C44 = rho*g*(I_wp - V*KB) + g*M*KG")
print("The pst-based C44 = integral(y * rho*g*y * n_z * dS) over all panels")
print("These SHOULD be equal but may differ because:")
print("  - restorematr uses exact waterplane area integrals")
print("  - pst is evaluated at discrete pressure points on the hull surface")
print("  - The pst integral is over the submerged body, not just the waterplane")
print()
print("For the drift force rotation term, what matters is:")
print("  F_rot_y = -(1/2) Re[ integral(P * conj(alpha) x f) dS ]")
print("  The pst part gives: -(1/2) Re[ integral(rho*g*(eta3 + y*eta4 - x*eta5) * (conj(alpha) x f)) dS ]")
print("  The heave-pst cross-term: -(1/2) Re[ rho*g*eta3 * integral(conj(alpha) x f) dS ]")
print()
print("The integral(conj(alpha) x f) should be related to the restoring force,")
print("but only the WATERPLANE contribution, not the full submerged surface.")
print()
print("KEY INSIGHT: pst = rho*g*eta3 is applied to ALL pressure points (submerged hull),")
print("but the physical waterplane restoring only acts at the waterline.")
print("This creates an ARTIFICIAL pressure over the entire hull that doesn't exist physically!")
