#!/usr/bin/env python3
"""Detailed analysis of pdstrip drift forces vs Seo et al. benchmark.
Focus on: (1) roll resonance spikes, (2) short-wave behavior, (3) head seas shape."""

import re
import numpy as np

results = []
with open('pdstrip.out', 'r') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i]
    m = re.match(r'\s*Wave circ\. frequency\s+([\d.]+)\s+encounter frequ\.\s+([\d.]+)\s+'
                 r'wave length\s+([\d.]+)\s+wave number\s+([\d.]+)\s+wave angle\s+([\d.]+)', line)
    if m:
        omega = float(m.group(1))
        omega_e = float(m.group(2))
        wavelength = float(m.group(3))
        wavenumber = float(m.group(4))
        wave_angle = float(m.group(5))
        i += 1
        m2 = re.match(r'\s*speed\s+([\d.]+)', lines[i])
        speed = float(m2.group(1)) if m2 else None
        i += 1  # header
        i += 1  # Translation
        m3 = re.match(r'\s*Translation\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                       r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                       r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', lines[i])
        if m3:
            surge_abs = float(m3.group(3))
            sway_abs = float(m3.group(6))
            heave_abs = float(m3.group(9))
        i += 1  # Rotation
        m4 = re.match(r'\s*Rotation/k\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                       r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                       r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', lines[i])
        if m4:
            roll_abs = float(m4.group(3))
            pitch_abs = float(m4.group(6))
            yaw_abs = float(m4.group(9))
        i += 1  # Drift
        m5 = re.match(r'\s*Longitudinal and transverse drift force.*?\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)', lines[i])
        fxi = float(m5.group(1)) if m5 else 0
        feta = float(m5.group(2)) if m5 else 0
        results.append({
            'omega': omega, 'omega_e': omega_e, 'wavelength': wavelength,
            'wavenumber': wavenumber, 'wave_angle': wave_angle, 'speed': speed,
            'surge_abs': surge_abs, 'sway_abs': sway_abs, 'heave_abs': heave_abs,
            'roll_abs': roll_abs, 'pitch_abs': pitch_abs, 'yaw_abs': yaw_abs,
            'fxi': fxi, 'feta': feta,
        })
    i += 1

rho = 1025.0
g = 9.81
Lpp = 328.2
B = 58.0
norm = rho * g * B**2 / Lpp

# ============================================================
# 1. HEAD SEAS ANALYSIS - Where is the pdstrip peak vs Seo?
# ============================================================
print("=" * 90)
print("HEAD SEAS (180°) AT V=3 m/s - DETAILED")
print("=" * 90)
print(f"Seo SWAN1: peak σ_aw ≈ 2.0-2.5 at λ/L ≈ 0.35-0.50")
print(f"Seo Exp:   peak σ_aw ≈ 2.5-3.5 at λ/L ≈ 0.40-0.50")
print()

head_3 = [r for r in results if abs(r['wave_angle'] - 180.0) < 0.5 and abs(r['speed'] - 3.0) < 0.1]
head_3.sort(key=lambda r: r['wavelength'])

print(f"{'λ/L':>8} {'σ_aw':>8} {'Heave':>8} {'Pitch/k':>8} {'ω(rad/s)':>10} {'Note':}")
for r in head_3:
    lam_L = r['wavelength'] / Lpp
    sigma = -r['fxi'] / norm
    note = ""
    if sigma > 5.0:
        note = " *** PEAK (too high!)"
    elif sigma > 2.0:
        note = " * elevated"
    print(f"{lam_L:8.3f} {sigma:8.3f} {r['heave_abs']:8.3f} {r['pitch_abs']:8.3f} {r['omega']:10.4f}{note}")

# Find peak
peak_r = max(head_3, key=lambda r: -r['fxi']/norm)
peak_lam_L = peak_r['wavelength'] / Lpp
peak_sigma = -peak_r['fxi'] / norm
print(f"\npdstrip peak: σ_aw = {peak_sigma:.2f} at λ/L = {peak_lam_L:.3f}")
print(f"This is {peak_sigma/2.5:.1f}× the Seo SWAN1 peak, and shifted to {peak_lam_L:.2f} vs ~0.45")

# ============================================================
# 2. ROLL RESONANCE ANALYSIS at oblique headings
# ============================================================
print("\n" + "=" * 90)
print("ROLL RESONANCE CHECK AT VARIOUS HEADINGS (V=3 m/s)")
print("=" * 90)

for beta in [150, 120, 90, 60, 30]:
    recs = [r for r in results if abs(r['wave_angle'] - beta) < 0.5 and abs(r['speed'] - 3.0) < 0.1]
    recs.sort(key=lambda r: r['wavelength'])
    
    # Find max roll/k
    max_roll_r = max(recs, key=lambda r: r['roll_abs'])
    max_roll_lam_L = max_roll_r['wavelength'] / Lpp
    
    # Find max |sigma_aw|
    max_sigma_r = max(recs, key=lambda r: abs(-r['fxi']/norm))
    max_sigma = -max_sigma_r['fxi'] / norm
    max_sigma_lam_L = max_sigma_r['wavelength'] / Lpp
    
    print(f"\nβ = {beta}°:")
    print(f"  Max roll/k = {max_roll_r['roll_abs']:.2f} at λ/L = {max_roll_lam_L:.3f}")
    print(f"  Max |σ_aw| = {abs(max_sigma):.2f} (σ_aw = {max_sigma:.2f}) at λ/L = {max_sigma_lam_L:.3f}")
    
    # Print around roll resonance
    print(f"  {'λ/L':>8} {'σ_aw':>8} {'Roll/k':>8} {'Heave':>8} {'Sway':>8}")
    for r in recs:
        lam_L = r['wavelength'] / Lpp
        if 0.8 < lam_L < 2.0:
            sigma = -r['fxi'] / norm
            print(f"  {lam_L:8.3f} {sigma:8.3f} {r['roll_abs']:8.3f} {r['heave_abs']:8.3f} {r['sway_abs']:8.3f}")

# ============================================================
# 3. SHORT-WAVE ASYMPTOTE CHECK
# ============================================================
print("\n" + "=" * 90)
print("SHORT-WAVE BEHAVIOR (λ/L < 0.2) AT V=3 m/s")
print("=" * 90)
print("In short waves, drift ≈ pure diffraction (no body motion).")
print("For head seas, SWAN1 shows σ_aw rising from 0 at λ/L=0.15 to ~1.5 at λ/L=0.30")
print()

for beta in [180, 150, 120, 90]:
    recs = [r for r in results if abs(r['wave_angle'] - beta) < 0.5 and abs(r['speed'] - 3.0) < 0.1]
    recs.sort(key=lambda r: r['wavelength'])
    print(f"β = {beta}°:")
    print(f"  {'λ/L':>8} {'σ_aw':>8}")
    for r in recs:
        lam_L = r['wavelength'] / Lpp
        if lam_L < 0.25:
            sigma = -r['fxi'] / norm
            print(f"  {lam_L:8.3f} {sigma:8.3f}")
    print()

# ============================================================
# 4. COMPARE V=0 vs V=3 HEAD SEAS
# ============================================================
print("=" * 90)
print("HEAD SEAS: V=0 vs V=3 m/s")
print("=" * 90)

head_0 = [r for r in results if abs(r['wave_angle'] - 180.0) < 0.5 and abs(r['speed']) < 0.01]
head_0.sort(key=lambda r: r['wavelength'])

print(f"{'λ/L':>8} {'σ_aw(V=0)':>12} {'σ_aw(V=3)':>12} {'ratio':>8}")
for r0 in head_0:
    lam_L = r0['wavelength'] / Lpp
    sigma0 = -r0['fxi'] / norm
    # Find matching V=3 record
    matches = [r for r in head_3 if abs(r['wavelength'] - r0['wavelength']) < 1.0]
    if matches:
        r3 = matches[0]
        sigma3 = -r3['fxi'] / norm
        ratio = sigma3 / sigma0 if abs(sigma0) > 0.01 else float('nan')
        print(f"{lam_L:8.3f} {sigma0:12.3f} {sigma3:12.3f} {ratio:8.2f}")
