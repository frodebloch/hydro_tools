#!/usr/bin/env python3
"""Compare stb-only vs stb+port catamaran drift forces (same code version).

Both outputs are from the current pdstrip.f90, same input (cat_20: hulld=20, V=0, 35 wl, 19+17=36 headings).
The ONLY difference is that port hull integration was disabled for the stb-only run.

Key validation:
- Head seas (mu=0): imirr = imu, so port = stb. Expect stb+port = 2 * stb-only.
- Following seas (mu=180): same. Expect ratio = 2.0.
- General: feta(mu=0) should be ~0 in stb+port (symmetry).
"""

import re
import numpy as np

def parse_output(filename):
    """Parse pdstrip output with new batch-format drift forces."""
    # Extract (omega, wavelength, angle) from wave header lines
    headers = []
    seen = set()
    drift_forces = []
    
    with open(filename) as f:
        for line in f:
            m = re.search(r'Wave circ\. frequency\s+([\d.]+)\s+encounter frequ\.\s+[\d.]+\s+wave length\s+([\d.]+)\s+wave number\s+[\d.]+\s+wave angle\s+([-\d.]+)', line)
            if m:
                omega = float(m.group(1))
                wl = float(m.group(2))
                angle = float(m.group(3))
                key = (omega, angle)
                if key not in seen:
                    seen.add(key)
                    headers.append({'omega': omega, 'wavelength': wl, 'angle': angle})
            
            m = re.search(r'Longitudinal and transverse drift force per wave amplitude squared\s+([-\d.E+]+)\s+([-\d.E+]+)', line)
            if m:
                drift_forces.append((float(m.group(1)), float(m.group(2))))
    
    # Also parse yaw and roll drift moments
    # For now just use fxi and feta
    results = []
    for i, (fxi, feta) in enumerate(drift_forces):
        if i < len(headers):
            h = headers[i]
            results.append({
                'omega': h['omega'], 'wavelength': h['wavelength'],
                'angle': h['angle'], 'fxi': fxi, 'feta': feta
            })
    
    return results

print("Parsing stb-only output...")
stb = parse_output('/home/blofro/src/pdstrip_test/test_convergence/cat_20/pdstrip_out_stb_only.out')
print(f"  {len(stb)} entries")

print("Parsing stb+port output...")
both = parse_output('/home/blofro/src/pdstrip_test/test_convergence/cat_20/pdstrip_out_stb_port.out')
print(f"  {len(both)} entries")

# Build lookups
stb_lk = {(round(r['omega'],3), round(r['angle'],1)): r for r in stb}
both_lk = {(round(r['omega'],3), round(r['angle'],1)): r for r in both}

# ===== HEAD SEAS (mu=0) =====
print("\n" + "="*120)
print("HEAD SEAS (mu=0): stb+port should = 2.0 * stb-only")
print("="*120)
print(f"{'wl':>8} {'omega':>8} | {'stb_fxi':>12} {'both_fxi':>12} {'ratio':>10} | {'stb_feta':>12} {'both_feta':>12} {'ratio_eta':>10}")
print("-"*120)

head_ratios = []
for r in sorted(both, key=lambda x: -x['omega']):
    if abs(r['angle']) > 0.01:
        continue
    key = (round(r['omega'],3), 0.0)
    s = stb_lk.get(key)
    if s is None:
        continue
    
    ratio_xi = r['fxi'] / s['fxi'] if abs(s['fxi']) > 1.0 else float('nan')
    ratio_eta = r['feta'] / s['feta'] if abs(s['feta']) > 1.0 else float('nan')
    
    flag = ""
    if not np.isnan(ratio_xi) and abs(ratio_xi - 2.0) > 0.01:
        flag += " ***"
    
    head_ratios.append(ratio_xi)
    print(f"{r['wavelength']:8.2f} {r['omega']:8.3f} | {s['fxi']:12.0f} {r['fxi']:12.0f} {ratio_xi:10.4f} | {s['feta']:12.0f} {r['feta']:12.0f} {ratio_eta:10.4f}{flag}")

head_ratios = [x for x in head_ratios if not np.isnan(x)]
if head_ratios:
    print(f"\nHead seas surge ratio: mean={np.mean(head_ratios):.6f}, std={np.std(head_ratios):.6f}, min={np.min(head_ratios):.6f}, max={np.max(head_ratios):.6f}")

# ===== FOLLOWING SEAS (mu=180) =====
print("\n" + "="*120)
print("FOLLOWING SEAS (mu=180): stb+port should = 2.0 * stb-only")
print("="*120)
print(f"{'wl':>8} {'omega':>8} | {'stb_fxi':>12} {'both_fxi':>12} {'ratio':>10} | {'stb_feta':>12} {'both_feta':>12}")
print("-"*100)

foll_ratios = []
for r in sorted(both, key=lambda x: -x['omega']):
    if abs(r['angle'] - 180.0) > 0.01:
        continue
    key = (round(r['omega'],3), 180.0)
    s = stb_lk.get(key)
    if s is None:
        continue
    
    ratio_xi = r['fxi'] / s['fxi'] if abs(s['fxi']) > 1.0 else float('nan')
    foll_ratios.append(ratio_xi)
    flag = ""
    if not np.isnan(ratio_xi) and abs(ratio_xi - 2.0) > 0.01:
        flag += " ***"
    
    print(f"{r['wavelength']:8.2f} {r['omega']:8.3f} | {s['fxi']:12.0f} {r['fxi']:12.0f} {ratio_xi:10.4f} | {s['feta']:12.0f} {r['feta']:12.0f}{flag}")

foll_ratios = [x for x in foll_ratios if not np.isnan(x)]
if foll_ratios:
    print(f"\nFollowing seas surge ratio: mean={np.mean(foll_ratios):.6f}, std={np.std(foll_ratios):.6f}")

# ===== SWAY CANCELLATION AT HEAD SEAS =====
print("\n" + "="*120)
print("HEAD SEAS feta (should be ~0 in stb+port by symmetry)")
print("="*120)
print(f"{'wl':>8} {'omega':>8} | {'both_fxi':>12} {'both_feta':>12} {'|feta/fxi|':>12}")
print("-"*70)

for r in sorted(both, key=lambda x: -x['omega']):
    if abs(r['angle']) > 0.01:
        continue
    ratio = abs(r['feta'] / r['fxi']) if abs(r['fxi']) > 1.0 else float('nan')
    flag = " ***" if not np.isnan(ratio) and ratio > 0.01 else ""
    print(f"{r['wavelength']:8.2f} {r['omega']:8.3f} | {r['fxi']:12.0f} {r['feta']:12.0f} {ratio:12.8f}{flag}")

# ===== ALL-ANGLE RATIO SUMMARY =====
print("\n" + "="*120)
print("ALL ANGLES: stb+port / stb-only ratio summary")
print("="*120)
print(f"{'angle':>8} | {'avg_ratio_xi':>14} {'std':>10} {'min':>10} {'max':>10} | {'avg_ratio_eta':>14} {'n':>4}")
print("-"*90)

all_angles = sorted(set(round(r['angle'],1) for r in both))
for mu in all_angles:
    ratios_xi = []
    ratios_eta = []
    for r in both:
        if abs(round(r['angle'],1) - mu) > 0.01:
            continue
        key = (round(r['omega'],3), mu)
        s = stb_lk.get(key)
        if s is None:
            continue
        if abs(s['fxi']) > 100:
            ratios_xi.append(r['fxi'] / s['fxi'])
        if abs(s['feta']) > 100:
            ratios_eta.append(r['feta'] / s['feta'])
    
    if ratios_xi:
        print(f"{mu:8.1f} | {np.mean(ratios_xi):14.6f} {np.std(ratios_xi):10.6f} {np.min(ratios_xi):10.4f} {np.max(ratios_xi):10.4f} | {np.mean(ratios_eta):14.6f} {len(ratios_xi):4d}" if ratios_eta else
              f"{mu:8.1f} | {np.mean(ratios_xi):14.6f} {np.std(ratios_xi):10.6f} {np.min(ratios_xi):10.4f} {np.max(ratios_xi):10.4f} | {'N/A':>14} {len(ratios_xi):4d}")

# ===== SYMMETRY CHECK ON stb+port =====
print("\n" + "="*120)
print("SYMMETRY CHECK: fxi(+mu) = fxi(-mu), feta(+mu) = -feta(-mu) in stb+port")
print("="*120)
print(f"{'omega':>8} {'mu':>6} | {'fxi(+)':>12} {'fxi(-)':>12} {'rel_diff%':>10} | {'feta(+)':>12} {'feta(-)':>12} {'anti_err%':>10}")
print("-"*100)

sym_err_xi = []
sym_err_eta = []
test_omegas = sorted(set(round(r['omega'],3) for r in both), reverse=True)[:5]
for omega in test_omegas:
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
        
        sym_err_xi.append(xi_err)
        sym_err_eta.append(eta_err)
        
        flag = ""
        if xi_err > 1: flag += " xi!"
        if eta_err > 1: flag += " eta!"
        
        print(f"{omega:8.3f} {mu:6d} | {rp['fxi']:12.0f} {rn['fxi']:12.0f} {xi_err:10.4f} | {rp['feta']:12.0f} {rn['feta']:12.0f} {eta_err:10.4f}{flag}")

if sym_err_xi:
    print(f"\nMax symmetry error: surge={max(sym_err_xi):.4f}%, sway antisymmetry={max(sym_err_eta):.4f}%")
