#!/usr/bin/env python3
"""
Compute theoretical sway rotation term from Newton's 2nd law and compare
with the actual rotation term from pdstrip.

Theory: feta_rot ≈ (1/2) Re[conj(eta4) * Fz_total]
where Fz_total = -omega^2 * M * eta3 (from Newton's 2nd law)

This gives: feta_rot_newton = -(omega^2 * M / 2) * Re[eta3 * conj(eta4)]

Compare with:
- feta_rot from code (using actual Fz_pres from triangle integration)
- feta_rot_theory (using Fz from Newton's law)
"""
import re
import numpy as np

# KVLCC2 parameters
Lpp = 328.2; g = 9.81; rho = 1025.0
mass = 320e6  # kg (total ship mass)
norm_sway = 2 * rho * g * Lpp  # sway normalization

# Parse motions from pdstrip.out
blocks = []
with open('pdstrip.out') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i]
    m = re.match(r'\s*Wave circ\. frequency\s+([\d.]+)\s+encounter frequ\.\s+([\d.]+)\s+wave length\s+([\d.]+)\s+wave number\s+([\d.]+)\s+wave angle\s+([\d.-]+)', line)
    if m:
        om = float(m.group(1))
        mu = float(m.group(5))
        if i+1 < len(lines):
            m2 = re.match(r'\s*speed\s+([\d.]+)', lines[i+1])
            if m2:
                vs = float(m2.group(1))
                if i+4 < len(lines):
                    tline = lines[i+3].strip()
                    rline = lines[i+4].strip()
                    tm = re.findall(r'[-\d.]+', tline)
                    rm = re.findall(r'[-\d.]+', rline)
                    if len(tm) >= 9 and len(rm) >= 9:
                        tvals = [float(x) for x in tm[-9:]]
                        rvals = [float(x) for x in rm[-9:]]
                        k = om**2 / g
                        blocks.append({
                            'om': om, 'mu': mu, 'vs': vs,
                            'eta3': complex(tvals[6], tvals[7]),
                            'eta4': complex(rvals[0], rvals[1]) * k,  # convert from /k to radians
                        })
    i += 1

# Parse FORCE_CHECK data  
force_data = {}
current_omega = None; current_mu = None
with open('debug.out') as f:
    for line in f:
        m = re.search(r'DRIFT_START omega=\s*([\d.]+)\s+mu=\s*([\d.-]+)', line)
        if m:
            current_omega = float(m.group(1))
            current_mu = float(m.group(2))
            continue
        m = re.search(r'FORCE_CHECK Fy_pres=\s*([\d.E+-]+)\s+([\d.E+-]+)\s+Fy_newton=\s*([\d.E+-]+)\s+([\d.E+-]+)\s+Fz_pres=\s*([\d.E+-]+)\s+([\d.E+-]+)\s+Fz_newton=\s*([\d.E+-]+)\s+([\d.E+-]+)', line)
        if m and current_omega and abs(current_mu - 90.0) < 1.0:
            key = current_omega
            if key not in force_data:
                force_data[key] = {
                    'fz_pres': complex(float(m.group(5)), float(m.group(6))),
                    'fz_newton': complex(float(m.group(7)), float(m.group(8))),
                }

# Parse DRIFT_SWAY data
drift_data = {}
current_omega = None; current_mu = None
with open('debug.out') as f:
    for line in f:
        m = re.search(r'DRIFT_START omega=\s*([\d.]+)\s+mu=\s*([\d.-]+)', line)
        if m:
            current_omega = float(m.group(1))
            current_mu = float(m.group(2))
            continue
        m = re.search(r'DRIFT_SWAY feta_vel=\s*([\d.E+-]+)\s+feta_rot=\s*([\d.E+-]+)', line)
        if m and current_omega and abs(current_mu - 90.0) < 1.0:
            key = current_omega
            if key not in drift_data:
                drift_data[key] = float(m.group(2))

# Filter for beam seas V=0
beam_v0 = []
seen = set()
for b in blocks:
    if abs(b['mu'] - 90.0) < 0.1 and abs(b['vs']) < 0.1 and b['om'] not in seen:
        seen.add(b['om'])
        beam_v0.append(b)
beam_v0.sort(key=lambda b: b['om'])

print("="*140)
print("ROTATION TERM COMPARISON: Actual (from triangles) vs Theoretical (from Newton's 2nd law)")
print("All values normalized by 2*rho*g*Lpp (sway normalization)")
print("="*140)
print(f"{'omega':>6} {'lam/L':>6} | {'rot_actual':>11} {'rot_theory':>11} {'ratio':>8} {'diff':>11} | "
      f"{'Fz_ratio':>9} {'|eta3|':>7} {'|eta4|':>9} {'ph(3-4)':>8} {'Re[e3c4]':>10}")
print("-"*140)

for b in beam_v0:
    om = b['om']
    lam = 2*np.pi*g/om**2
    lam_L = lam / Lpp
    
    if lam_L < 0.05 or lam_L > 3.5:
        continue
    
    eta3 = b['eta3']
    eta4 = b['eta4']
    
    # Phase difference
    ph_diff = np.degrees(np.angle(eta3 * np.conj(eta4)))
    cross = np.real(eta3 * np.conj(eta4))
    
    # Theoretical rotation term from Newton: feta_rot = -(omega^2*M/2) * Re[eta3*conj(eta4)]
    # But wait — this is the TOTAL force from pressure, not just the rotation Pinkster component
    # The rotation Pinkster component is: (1/2) Re[conj(alpha) × (P·f)]_y
    # which at beam seas ≈ (1/2) Re[conj(eta4) * Fz_total]
    # And Fz_total = cpres_fz_sum from the triangle integration
    
    # METHOD 1: Use actual Fz from triangles
    rot_from_Fz_pres = None
    if om in force_data:
        fz_p = force_data[om]['fz_pres']
        fz_n = force_data[om]['fz_newton']
        rot_from_Fz_pres = 0.5 * np.real(np.conj(eta4) * fz_p)
        rot_from_Fz_newton = 0.5 * np.real(np.conj(eta4) * fz_n)
        fz_ratio = abs(fz_p) / max(abs(fz_n), 1)
    else:
        rot_from_Fz_newton = -om**2 * mass / 2 * cross
        fz_ratio = float('nan')
    
    # Actual rotation term from code
    rot_actual = drift_data.get(om, float('nan'))
    
    # Normalize
    rot_actual_n = rot_actual / norm_sway if not np.isnan(rot_actual) else float('nan')
    rot_theory_n = rot_from_Fz_newton / norm_sway
    
    if not np.isnan(rot_actual_n) and abs(rot_theory_n) > 1e-6:
        ratio = rot_actual_n / rot_theory_n
    else:
        ratio = float('nan')
    
    diff_n = rot_actual_n - rot_theory_n if not np.isnan(rot_actual_n) else float('nan')
    
    print(f"{om:6.3f} {lam_L:6.2f} | {rot_actual_n:11.5f} {rot_theory_n:11.5f} {ratio:8.3f} {diff_n:11.5f} | "
          f"{fz_ratio:9.3f} {abs(eta3):7.4f} {abs(eta4):9.6f} {ph_diff:8.1f}° {cross:10.6f}")

print()
print("INTERPRETATION:")
print("- ratio ≈ 1.0: rotation term is well-explained by the simple cross-product formula")
print("- ratio >> 1 or << 1: other terms contribute significantly (e.g., Fy × pitch, Fx × yaw)")
print("- If 'rot_theory' using Newton matches 'rot_actual', the blow-up is inherent")
print("  (the formula -(omega²M/2)*Re[eta3*conj(eta4)] IS the correct physics)")
print("- If they differ, the 3D triangle integration introduces additional errors")

# Also show what the rotation term would be if we used Newton's Fz exactly
print("\n\n" + "="*140)
print("DECOMPOSITION: rot_actual vs rot_from_Fz_pres vs rot_from_Fz_newton")
print("="*140)
print(f"{'omega':>6} {'lam/L':>6} | {'rot_actual':>11} {'from_Fz_p':>11} {'from_Fz_n':>11} | "
      f"{'act-Fz_p':>11} {'act-Fz_n':>11} | {'Fz_p-Fz_n':>11}")
print("-"*120)

for b in beam_v0:
    om = b['om']
    lam = 2*np.pi*g/om**2
    lam_L = lam / Lpp
    eta3 = b['eta3']
    eta4 = b['eta4']
    
    if lam_L < 0.5 or lam_L > 2.5 or om not in force_data or om not in drift_data:
        continue
    
    fz_p = force_data[om]['fz_pres']
    fz_n = force_data[om]['fz_newton']
    
    rot_actual = drift_data[om] / norm_sway
    rot_fz_p = 0.5 * np.real(np.conj(eta4) * fz_p) / norm_sway
    rot_fz_n = 0.5 * np.real(np.conj(eta4) * fz_n) / norm_sway
    
    print(f"{om:6.3f} {lam_L:6.2f} | {rot_actual:11.5f} {rot_fz_p:11.5f} {rot_fz_n:11.5f} | "
          f"{rot_actual-rot_fz_p:11.5f} {rot_actual-rot_fz_n:11.5f} | {rot_fz_p-rot_fz_n:11.5f}")
