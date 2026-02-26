#!/usr/bin/env python3
"""Check symmetry in stb-only output to see if BEM itself breaks symmetry at certain omegas."""

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

print("=== STB-ONLY symmetry check ===")
stb = parse_output('/home/blofro/src/pdstrip_test/test_convergence/cat_20/pdstrip_out_stb_only.out')
stb_lk = {(round(r['omega'],3), round(r['angle'],1)): r for r in stb}

print(f"{'omega':>8} {'mu':>6} | {'fxi(+)':>12} {'fxi(-)':>12} {'rel_diff%':>10} | {'feta(+)':>12} {'feta(-)':>12} {'anti_err%':>10}")
print("-"*100)

omegas = sorted(set(round(r['omega'],3) for r in stb), reverse=True)
for omega in omegas:
    for mu in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        kp = (omega, float(mu))
        kn = (omega, float(-mu))
        if kp not in stb_lk or kn not in stb_lk:
            continue
        rp = stb_lk[kp]
        rn = stb_lk[kn]
        
        # For a single stb hull, fxi(+mu) and fxi(-mu) come from different BEM solutions
        # but same physical geometry seen from different sides. 
        # In catamaran BEM, the solution depends on the wave angle, so +mu and -mu give different results.
        denom_xi = max(abs(rp['fxi']), abs(rn['fxi']), 1.0)
        xi_err = abs(rp['fxi'] - rn['fxi']) / denom_xi * 100
        
        denom_eta = max(abs(rp['feta']), abs(rn['feta']), 1.0)
        eta_err = abs(rp['feta'] + rn['feta']) / denom_eta * 100
        
        flag = ""
        if xi_err > 5: flag += " xi!"
        if eta_err > 5: flag += " eta!"
        
        if xi_err > 5 or eta_err > 5 or omega == 1.458:
            print(f"{omega:8.3f} {mu:6d} | {rp['fxi']:12.0f} {rn['fxi']:12.0f} {xi_err:10.4f} | {rp['feta']:12.0f} {rn['feta']:12.0f} {eta_err:10.4f}{flag}")

print("\n=== STB+PORT symmetry check (same omegas) ===")
both = parse_output('/home/blofro/src/pdstrip_test/test_convergence/cat_20/pdstrip_out_stb_port.out')
both_lk = {(round(r['omega'],3), round(r['angle'],1)): r for r in both}

print(f"{'omega':>8} {'mu':>6} | {'fxi(+)':>12} {'fxi(-)':>12} {'rel_diff%':>10} | {'feta(+)':>12} {'feta(-)':>12} {'anti_err%':>10}")
print("-"*100)
for omega in omegas:
    for mu in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        kp = (omega, float(mu))
        kn = (omega, float(-mu))
        if kp not in both_lk or kn not in both_lk:
            continue
        rp = both_lk[kp]
        rn = both_lk[kn]
        
        denom_xi = max(abs(rp['fxi']), abs(rn['fxi']), 1.0)
        xi_err = abs(rp['fxi'] - rn['fxi']) / denom_xi * 100
        
        denom_eta = max(abs(rp['feta']), abs(rn['feta']), 1.0)
        eta_err = abs(rp['feta'] + rn['feta']) / denom_eta * 100
        
        flag = ""
        if xi_err > 5: flag += " xi!"
        if eta_err > 5: flag += " eta!"
        
        if xi_err > 5 or eta_err > 5 or omega == 1.458:
            print(f"{omega:8.3f} {mu:6d} | {rp['fxi']:12.0f} {rn['fxi']:12.0f} {xi_err:10.4f} | {rp['feta']:12.0f} {rn['feta']:12.0f} {eta_err:10.4f}{flag}")
