#!/usr/bin/env python3
"""Compare new catamaran (stb+port) drift forces at hulld=20 against old stb-only reference.

Old reference: pdstrip_out_cat20 — stb hull only, 5 speeds, 35 wavelengths, 29 headings
New run: test_convergence/cat_20/pdstrip.out — stb+port, V=0 only, 35 wavelengths, 36 headings

Key validation:
- Head seas (mu=0): new fxi should = 2 * old fxi (port = stb), new feta should ~ 0
- Symmetry: fxi(+mu) = fxi(-mu), feta(+mu) = -feta(-mu)
"""

import re
import numpy as np

def parse_old_format(filename):
    """Parse old pdstrip output where drift forces are inline with motion blocks."""
    results = []
    current_omega = None
    current_speed = None
    current_angle = None
    current_wavelength = None
    
    with open(filename) as f:
        for line in f:
            m = re.search(r'Wave circ\. frequency\s+([\d.]+)\s+encounter frequ\.\s+[\d.]+\s+wave length\s+([\d.]+)\s+wave number\s+[\d.]+\s+wave angle\s+([-\d.]+)', line)
            if m:
                current_omega = float(m.group(1))
                current_wavelength = float(m.group(2))
                current_angle = float(m.group(3))
                continue
            
            m = re.search(r'speed\s+([-\d.]+)\s+wetted', line)
            if m:
                current_speed = float(m.group(1))
                continue
            
            m = re.search(r'Longitudinal and transverse drift force per wave amplitude squared\s+([-\d.E+]+)\s+([-\d.E+]+)', line)
            if m and current_omega is not None:
                fxi = float(m.group(1))
                feta = float(m.group(2))
                results.append({
                    'omega': current_omega,
                    'wavelength': current_wavelength,
                    'speed': current_speed,
                    'angle': current_angle,
                    'fxi': fxi,
                    'feta': feta
                })
    return results

def parse_new_format(filename):
    """Parse new pdstrip output where drift forces are in a batch after all motion blocks.
    
    The new format lists all motion blocks first, then all drift forces in order:
    heading loop (36 headings) nested inside omega loop (35 wavelengths).
    """
    # First, extract all heading labels and omega values from motion blocks
    omegas_and_headings = []
    seen_omega_angle = set()
    
    with open(filename) as f:
        for line in f:
            m = re.search(r'Wave circ\. frequency\s+([\d.]+)\s+encounter frequ\.\s+[\d.]+\s+wave length\s+([\d.]+)\s+wave number\s+[\d.]+\s+wave angle\s+([-\d.]+)', line)
            if m:
                omega = float(m.group(1))
                wavelength = float(m.group(2))
                angle = float(m.group(3))
                key = (omega, angle)
                if key not in seen_omega_angle:
                    seen_omega_angle.add(key)
                    omegas_and_headings.append({
                        'omega': omega,
                        'wavelength': wavelength,
                        'angle': angle,
                    })
    
    # Now extract drift forces (just the raw list)
    drift_forces = []
    with open(filename) as f:
        for line in f:
            m = re.search(r'Longitudinal and transverse drift force per wave amplitude squared\s+([-\d.E+]+)\s+([-\d.E+]+)', line)
            if m:
                fxi = float(m.group(1))
                feta = float(m.group(2))
                drift_forces.append((fxi, feta))
    
    print(f"  Found {len(omegas_and_headings)} (omega, angle) combinations and {len(drift_forces)} drift force entries")
    
    if len(drift_forces) != len(omegas_and_headings):
        print(f"  WARNING: mismatch! Trying sequential assignment anyway.")
    
    results = []
    for i, (fxi, feta) in enumerate(drift_forces):
        if i < len(omegas_and_headings):
            oa = omegas_and_headings[i]
            results.append({
                'omega': oa['omega'],
                'wavelength': oa['wavelength'],
                'angle': oa['angle'],
                'speed': 0.0,
                'fxi': fxi,
                'feta': feta
            })
        else:
            results.append({
                'omega': 0,
                'wavelength': 0,
                'angle': 0,
                'speed': 0.0,
                'fxi': fxi,
                'feta': feta
            })
    
    return results

print("Parsing old (stb-only) reference...")
old_all = parse_old_format('/home/blofro/src/pdstrip_test/pdstrip_out_cat20')
print(f"  Total entries: {len(old_all)}")

print("\nParsing new (stb+port) output...")
new_all = parse_new_format('/home/blofro/src/pdstrip_test/test_convergence/cat_20/pdstrip.out')
print(f"  Total entries: {len(new_all)}")

# Filter old to V=0 only
old_v0 = [r for r in old_all if abs(r['speed']) < 0.01]
print(f"\nOld V=0 entries: {len(old_v0)}")

# Show first few entries from each to verify alignment
print("\nFirst 5 new entries:")
for i, r in enumerate(new_all[:5]):
    print(f"  {i}: omega={r['omega']:.3f} wl={r['wavelength']:.2f} angle={r['angle']:.1f} fxi={r['fxi']:.0f} feta={r['feta']:.0f}")

print("\nFirst 5 old V=0 entries:")
for i, r in enumerate(old_v0[:5]):
    print(f"  {i}: omega={r['omega']:.3f} wl={r['wavelength']:.2f} angle={r['angle']:.1f} fxi={r['fxi']:.0f} feta={r['feta']:.0f}")

# Build lookup for old V=0 data: key = (omega, angle)
old_lookup = {}
for r in old_v0:
    key = (round(r['omega'], 3), round(r['angle'], 1))
    old_lookup[key] = r

# Build lookup for new data
new_lookup = {}
for r in new_all:
    key = (round(r['omega'], 3), round(r['angle'], 1))
    new_lookup[key] = r

# ===== HEAD SEAS (mu=0) COMPARISON =====
print("\n" + "="*110)
print("HEAD SEAS (mu=0): New (stb+port) should be 2x Old (stb-only)")
print("At head seas, mirror angle = current angle, so port hull = stb hull (reversed panel order).")
print("="*110)
print(f"{'wavelength':>10} {'omega':>8} | {'old_fxi':>12} {'new_fxi':>12} {'ratio_xi':>10} | {'old_feta':>12} {'new_feta':>12} {'ratio_eta':>10}")
print("-"*110)

head_errors_xi = []
for r in sorted(new_all, key=lambda x: -x['omega']):
    if abs(r['angle']) > 0.01:
        continue
    key = (round(r['omega'], 3), 0.0)
    if key not in old_lookup:
        print(f"{r['wavelength']:10.2f} {r['omega']:8.3f} | {'N/A':>12} {r['fxi']:12.0f} {'N/A':>10} | {'N/A':>12} {r['feta']:12.0f} {'N/A':>10}")
        continue
    old = old_lookup[key]
    
    ratio_xi = r['fxi'] / old['fxi'] if abs(old['fxi']) > 1.0 else float('nan')
    ratio_eta = r['feta'] / old['feta'] if abs(old['feta']) > 1.0 else float('nan')
    
    if not np.isnan(ratio_xi):
        head_errors_xi.append(abs(ratio_xi - 2.0) / 2.0)
    
    flag = ""
    if not np.isnan(ratio_xi) and abs(ratio_xi - 2.0) > 0.05:
        flag += " ***"
    
    print(f"{r['wavelength']:10.2f} {r['omega']:8.3f} | {old['fxi']:12.0f} {r['fxi']:12.0f} {ratio_xi:10.4f} | {old['feta']:12.0f} {r['feta']:12.0f} {ratio_eta:10.4f}{flag}")

if head_errors_xi:
    print(f"\nSurge ratio at head seas: mean = {np.mean([r['fxi'] for r in new_all if abs(r['angle'])<0.01]) / np.mean([old_lookup[(round(r['omega'],3), 0.0)]['fxi'] for r in new_all if abs(r['angle'])<0.01 and (round(r['omega'],3), 0.0) in old_lookup]):.4f} (expect 2.0)")
    print(f"Max relative error from 2.0: {max(head_errors_xi)*100:.2f}%")

# ===== SWAY SYMMETRY CHECK =====
print("\n" + "="*110)
print("SWAY SYMMETRY: fxi(+mu) = fxi(-mu), feta(+mu) = -feta(-mu)")
print("="*110)
print(f"{'omega':>8} {'angle':>6} | {'fxi(+mu)':>12} {'fxi(-mu)':>12} {'rel_diff':>10} | {'feta(+mu)':>12} {'feta(-mu)':>12} {'+feta_sum':>12}")
print("-"*110)

sym_xi_max = 0
sym_eta_max = 0
# Show for first 3 omegas, all positive angles
test_omegas = sorted(set(round(r['omega'], 3) for r in new_all), reverse=True)[:3]
for omega in test_omegas:
    for mu in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        key_pos = (omega, float(mu))
        key_neg = (omega, float(-mu))
        if key_pos not in new_lookup or key_neg not in new_lookup:
            continue
        rp = new_lookup[key_pos]
        rn = new_lookup[key_neg]
        
        xi_diff = abs(rp['fxi'] - rn['fxi']) / max(abs(rp['fxi']), 1.0)
        eta_sum = abs(rp['feta'] + rn['feta']) / max(abs(rp['feta']), 1.0)
        
        sym_xi_max = max(sym_xi_max, xi_diff)
        sym_eta_max = max(sym_eta_max, eta_sum)
        
        flag = ""
        if xi_diff > 0.01: flag += " xi!"
        if eta_sum > 0.01: flag += " eta!"
        
        print(f"{omega:8.3f} {mu:6.0f} | {rp['fxi']:12.0f} {rn['fxi']:12.0f} {xi_diff:10.6f} | {rp['feta']:12.0f} {rn['feta']:12.0f} {rp['feta']+rn['feta']:12.0f}{flag}")

print(f"\nMax symmetry error: surge={sym_xi_max*100:.4f}%, sway antisymmetry={sym_eta_max*100:.4f}%")

# ===== ALL ANGLES RATIO (new/old) =====
print("\n" + "="*110)
print("ALL ANGLES: Average ratio new/old (expect 2.0 at head/following, other values elsewhere)")
print("="*110)
print(f"{'angle':>8} | {'avg_xi':>10} {'std_xi':>10} {'min_xi':>10} {'max_xi':>10} | {'avg_eta':>10} {'n':>4}")
print("-"*80)

common_angles = sorted(set(round(r['angle'], 1) for r in new_all) & set(round(r['angle'], 1) for r in old_v0))

for mu in common_angles:
    ratios_xi = []
    ratios_eta = []
    for r in new_all:
        if abs(round(r['angle'], 1) - mu) > 0.01:
            continue
        key = (round(r['omega'], 3), mu)
        if key not in old_lookup:
            continue
        old = old_lookup[key]
        if abs(old['fxi']) > 100:
            ratios_xi.append(r['fxi'] / old['fxi'])
        if abs(old['feta']) > 100:
            ratios_eta.append(r['feta'] / old['feta'])
    
    if ratios_xi:
        avg_xi = np.mean(ratios_xi)
        std_xi = np.std(ratios_xi)
        min_xi = np.min(ratios_xi)
        max_xi = np.max(ratios_xi)
    else:
        avg_xi = std_xi = min_xi = max_xi = float('nan')
    if ratios_eta:
        avg_eta = np.mean(ratios_eta)
    else:
        avg_eta = float('nan')
    
    flag = ""
    if abs(mu) < 1 and not np.isnan(avg_xi) and abs(avg_xi - 2.0) > 0.01:
        flag = " *** HEAD SEAS"
    if abs(mu - 180) < 1 and not np.isnan(avg_xi) and abs(avg_xi - 2.0) > 0.01:
        flag = " *** FOLLOWING SEAS"
    
    print(f"{mu:8.1f} | {avg_xi:10.4f} {std_xi:10.4f} {min_xi:10.4f} {max_xi:10.4f} | {avg_eta:10.4f} {len(ratios_xi):4d}{flag}")

# ===== HEAD SEAS feta check =====
print("\n" + "="*110)
print("HEAD SEAS feta check (should be ~0 by symmetry)")
print("="*110)
print(f"{'wavelength':>10} {'omega':>8} | {'new_fxi':>12} {'new_feta':>12} {'|feta/fxi|':>12}")
print("-"*70)
for r in sorted(new_all, key=lambda x: -x['omega']):
    if abs(r['angle']) > 0.01:
        continue
    ratio = abs(r['feta'] / r['fxi']) if abs(r['fxi']) > 1.0 else float('nan')
    flag = " ***" if not np.isnan(ratio) and ratio > 0.01 else ""
    print(f"{r['wavelength']:10.2f} {r['omega']:8.3f} | {r['fxi']:12.0f} {r['feta']:12.0f} {ratio:12.6f}{flag}")

# ===== FOLLOWING SEAS (mu=180) =====
print("\n" + "="*110)
print("FOLLOWING SEAS (mu=180): should also be 2x old (mirror = self)")
print("="*110)
print(f"{'wavelength':>10} {'omega':>8} | {'old_fxi':>12} {'new_fxi':>12} {'ratio_xi':>10} | {'new_feta':>12}")
print("-"*90)
for r in sorted(new_all, key=lambda x: -x['omega']):
    if abs(r['angle'] - 180.0) > 0.01:
        continue
    key = (round(r['omega'], 3), 180.0)
    if key not in old_lookup:
        print(f"{r['wavelength']:10.2f} {r['omega']:8.3f} | {'N/A':>12} {r['fxi']:12.0f} {'N/A':>10} | {r['feta']:12.0f}")
        continue
    old = old_lookup[key]
    ratio_xi = r['fxi'] / old['fxi'] if abs(old['fxi']) > 1.0 else float('nan')
    flag = " ***" if not np.isnan(ratio_xi) and abs(ratio_xi - 2.0) > 0.05 else ""
    print(f"{r['wavelength']:10.2f} {r['omega']:8.3f} | {old['fxi']:12.0f} {r['fxi']:12.0f} {ratio_xi:10.4f} | {r['feta']:12.0f}{flag}")
