#!/usr/bin/env python3
"""Compare mono drift forces to catamaran stb-only and stb+port at overlapping wavelengths.

At head seas: catamaran interaction modifies forces vs monohull, so exact match not expected.
But provides useful reference for interaction effects.
"""
import re
import numpy as np

def parse_output(filename):
    headers = []
    seen = set()
    drift_forces = []
    with open(filename) as f:
        for line in f:
            m = re.search(r'Wave circ\. frequency\s+([\d.]+)\s+encounter frequ\.\s+[\d.]+\s+wave length\s+([\d.]+)\s+wave number\s+[\d.]+\s+wave angle\s+([-\d.]+)', line)
            if m:
                omega, wl, angle = float(m.group(1)), float(m.group(2)), float(m.group(3))
                key = (omega, angle)
                if key not in seen:
                    seen.add(key)
                    headers.append({'omega': omega, 'wavelength': wl, 'angle': angle})
            m = re.search(r'Longitudinal and transverse drift force per wave amplitude squared\s+([-\d.E+]+)\s+([-\d.E+]+)', line)
            if m:
                drift_forces.append((float(m.group(1)), float(m.group(2))))
    results = []
    for i, (fxi, feta) in enumerate(drift_forces):
        if i < len(headers):
            h = headers[i]
            results.append({'omega': h['omega'], 'wavelength': h['wavelength'], 'angle': h['angle'], 'fxi': fxi, 'feta': feta})
    return results

print("Parsing mono output...")
mono = parse_output('/home/blofro/src/pdstrip_test/test_convergence/mono/pdstrip.out')
print(f"  {len(mono)} entries")

print("Parsing catamaran stb-only output...")
stb = parse_output('/home/blofro/src/pdstrip_test/test_convergence/cat_20/pdstrip_out_stb_only.out')
print(f"  {len(stb)} entries")

print("Parsing catamaran stb+port output...")
both = parse_output('/home/blofro/src/pdstrip_test/test_convergence/cat_20/pdstrip_out_stb_port.out')
print(f"  {len(both)} entries")

# Build lookups
mono_lk = {(round(r['omega'],3), round(r['angle'],1)): r for r in mono}
stb_lk = {(round(r['omega'],3), round(r['angle'],1)): r for r in stb}
both_lk = {(round(r['omega'],3), round(r['angle'],1)): r for r in both}

# HEAD SEAS comparison
print("\n" + "="*130)
print("HEAD SEAS (mu=0): Mono vs Cat_stb vs Cat_both")
print("Cat_stb/Mono = catamaran interaction factor on one hull")
print("Cat_both/Mono = total catamaran / monohull ratio (expect ~2 for wide separation)")
print("="*130)
print(f"{'wl':>8} {'omega':>8} | {'mono_fxi':>12} {'cat_stb_fxi':>12} {'stb/mono':>10} | {'cat_both_fxi':>14} {'both/mono':>10} {'both/(2*mono)':>14}")
print("-"*120)

for r in sorted(mono, key=lambda x: -x['omega']):
    if abs(r['angle']) > 0.01:
        continue
    key = (round(r['omega'],3), 0.0)
    s = stb_lk.get(key)
    b = both_lk.get(key)
    if s is None or b is None:
        continue
    
    ratio_stb = s['fxi'] / r['fxi'] if abs(r['fxi']) > 1.0 else float('nan')
    ratio_both = b['fxi'] / r['fxi'] if abs(r['fxi']) > 1.0 else float('nan')
    ratio_half = b['fxi'] / (2*r['fxi']) if abs(r['fxi']) > 1.0 else float('nan')
    
    print(f"{r['wavelength']:8.2f} {r['omega']:8.3f} | {r['fxi']:12.0f} {s['fxi']:12.0f} {ratio_stb:10.4f} | {b['fxi']:14.0f} {ratio_both:10.4f} {ratio_half:14.4f}")

# ALL ANGLES for one omega
print("\n" + "="*130)
print("ALL ANGLES at omega where overlap exists: Mono vs Cat_stb vs Cat_both")
print("="*130)

# Find overlap omegas
mono_omegas = set(round(r['omega'],3) for r in mono)
stb_omegas = set(round(r['omega'],3) for r in stb)
common_omegas = sorted(mono_omegas & stb_omegas, reverse=True)
print(f"Common omegas: {common_omegas}")

# Pick first common omega to show full angle comparison
if common_omegas:
    omega0 = common_omegas[0]
    print(f"\nAngle comparison at omega={omega0}:")
    print(f"{'angle':>8} | {'mono_fxi':>12} {'cat_stb_fxi':>12} {'stb/mono':>10} | {'mono_feta':>12} {'cat_stb_feta':>12} {'stb_eta/mono':>12}")
    print("-"*100)
    for mu in range(-90, 91, 10):
        key = (omega0, float(mu))
        m = mono_lk.get(key)
        s = stb_lk.get(key)
        if m is None or s is None:
            continue
        ratio_xi = s['fxi'] / m['fxi'] if abs(m['fxi']) > 1.0 else float('nan')
        ratio_eta = s['feta'] / m['feta'] if abs(m['feta']) > 1.0 else float('nan')
        print(f"{mu:8d} | {m['fxi']:12.0f} {s['fxi']:12.0f} {ratio_xi:10.4f} | {m['feta']:12.0f} {s['feta']:12.0f} {ratio_eta:12.4f}")
