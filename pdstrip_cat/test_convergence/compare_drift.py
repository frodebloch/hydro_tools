#!/usr/bin/env python3
"""Compare catamaran (wide spacing) drift forces against 2x monohull drift forces.
At hulld >> beam, the two hulls don't interact, so catamaran drift should equal 2x monohull."""

import re
import sys
import numpy as np

def extract_drift_forces(filename):
    """Extract (fxi, feta) pairs from pdstrip.out in order of appearance."""
    forces = []
    with open(filename) as f:
        for line in f:
            m = re.search(r'Longitudinal and transverse drift force per wave amplitude squared\s+([-\d.E+]+)\s+([-\d.E+]+)', line)
            if m:
                fxi = float(m.group(1))
                feta = float(m.group(2))
                forces.append((fxi, feta))
    return forces

def extract_wavelengths(filename):
    """Extract wavelengths and angles from pdstrip.out."""
    wavelengths = []
    angles = []
    with open(filename) as f:
        for line in f:
            m = re.search(r'Wave length\s+([\d.]+)', line)
            if m:
                wavelengths.append(float(m.group(1)))
            m = re.search(r'Wave encounter angle\s+([-\d.]+)', line)
            if m:
                angles.append(float(m.group(1)))
    return wavelengths, angles

mono_forces = extract_drift_forces('mono/pdstrip.out')
cat_forces = extract_drift_forces('cat_wide/pdstrip.out')

mono_wl, mono_ang = extract_wavelengths('mono/pdstrip.out')
cat_wl, cat_ang = extract_wavelengths('cat_wide/pdstrip.out')

print(f"Monohull: {len(mono_forces)} drift force pairs")
print(f"Catamaran: {len(cat_forces)} drift force pairs")
print(f"Monohull wavelengths: {len(set(mono_wl))} unique")
print(f"Monohull angles: {len(set(mono_ang))} unique")

if len(mono_forces) == 0 or len(cat_forces) == 0:
    print("ERROR: No drift forces found!")
    sys.exit(1)

# The forces should be in the same order: wavelength x speed x angle
# For V=0, there's 1 speed, so n_forces = n_wavelengths * n_angles
n_mono = len(mono_forces)
n_cat = len(cat_forces)

if n_mono != n_cat:
    print(f"WARNING: different number of drift force entries: mono={n_mono}, cat={n_cat}")
    n = min(n_mono, n_cat)
else:
    n = n_mono

print(f"\n{'idx':>4} {'mono_fxi':>12} {'2*mono_fxi':>12} {'cat_fxi':>12} {'ratio_xi':>10} | {'mono_feta':>12} {'2*mono_feta':>12} {'cat_feta':>12} {'ratio_eta':>10}")
print("-" * 120)

max_xi_err = 0
max_eta_err = 0
for i in range(n):
    mx, my = mono_forces[i]
    cx, cy = cat_forces[i]
    # Ratio: cat / (2 * mono). Should be ~1.0 for wide spacing
    ratio_xi = cx / (2*mx) if abs(mx) > 1e-6 else float('nan')
    ratio_eta = cy / (2*my) if abs(my) > 1e-6 else float('nan')
    
    rel_err_xi = abs(ratio_xi - 1.0) if not np.isnan(ratio_xi) else 0
    rel_err_eta = abs(ratio_eta - 1.0) if not np.isnan(ratio_eta) else 0
    max_xi_err = max(max_xi_err, rel_err_xi)
    max_eta_err = max(max_eta_err, rel_err_eta)
    
    flag = ""
    if rel_err_xi > 0.05 or rel_err_eta > 0.05:
        flag = " *** >5% error"
    
    print(f"{i:4d} {mx:12.1f} {2*mx:12.1f} {cx:12.1f} {ratio_xi:10.4f} | {my:12.1f} {2*my:12.1f} {cy:12.1f} {ratio_eta:10.4f}{flag}")

print(f"\nMax relative error: surge={max_xi_err:.4f} ({max_xi_err*100:.1f}%), sway={max_eta_err:.4f} ({max_eta_err*100:.1f}%)")
if max_xi_err < 0.05 and max_eta_err < 0.05:
    print("PASS: All drift forces within 5% of 2x monohull")
else:
    print("FAIL: Some drift forces differ by >5% from 2x monohull")
