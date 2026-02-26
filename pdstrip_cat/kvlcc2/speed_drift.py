#!/usr/bin/env python3
"""Extract forward-speed added resistance for KVLCC2."""
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
        wavelength = float(m.group(3))
        wavenumber = float(m.group(4))
        wave_angle = float(m.group(5))
        i += 1
        m2 = re.match(r'\s*speed\s+([\d.]+)', lines[i])
        speed = float(m2.group(1)) if m2 else 0
        i += 1  # header
        i += 1  # Translation
        m3 = re.match(r'\s*Translation\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                       r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                       r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', lines[i])
        if m3:
            heave_abs = float(m3.group(9))
        i += 1  # Rotation
        m4 = re.match(r'\s*Rotation/k\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                       r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', lines[i])
        if m4:
            pitch_abs = float(m4.group(6))
        i += 1
        m5 = re.match(r'\s*Longitudinal and transverse drift force.*?\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)', lines[i])
        if m5:
            fxi = float(m5.group(1))
            feta = float(m5.group(2))
            results.append({
                'omega': omega, 'wavelength': wavelength, 'wavenumber': wavenumber,
                'wave_angle': wave_angle, 'speed': speed,
                'heave_abs': heave_abs, 'pitch_abs': pitch_abs,
                'fxi': fxi, 'feta': feta,
            })
    i += 1

# KVLCC2 parameters
rho = 1025.0
g = 9.81
Lpp = 328.2
B = 58.0
norm_raw = rho * g * B**2 / Lpp

# Typical KVLCC2 design speed: 15.5 kn = 7.97 m/s, Fn = 0.142
# Available speed closest to this is 7.96 m/s
target_speed = 7.96

head_v796 = [r for r in results if abs(r['wave_angle'] - 180.0) < 0.1 and abs(r['speed'] - target_speed) < 0.1]
head_v796.sort(key=lambda r: r['wavelength'])

print(f"KVLCC2 Head Seas at speed = {target_speed} m/s (Fn = {target_speed/np.sqrt(g*Lpp):.3f})")
print(f"Normalization: rho*g*B^2/Lpp = {norm_raw:.0f} N/m")
print()
print(f"{'λ(m)':>8} {'λ/Lpp':>8} {'ω√(L/g)':>8} {'Fxi':>12} {'σ_aw=-Fxi/N':>12} {'Heave':>8} {'Pitch/k':>8}")

for r in head_v796:
    lam_L = r['wavelength'] / Lpp
    omega_nd = r['omega'] * np.sqrt(Lpp / g)
    sigma = -r['fxi'] / norm_raw
    print(f"{r['wavelength']:8.1f} {lam_L:8.3f} {omega_nd:8.3f} {r['fxi']:12.1f} {sigma:12.5f} "
          f"{r['heave_abs']:8.3f} {r['pitch_abs']:8.3f}")

# Also compare zero speed vs design speed for the peak region
print(f"\n{'=== Comparison: Zero Speed vs {target_speed} m/s at peak region ===':^80}")
head_v0 = [r for r in results if abs(r['wave_angle'] - 180.0) < 0.1 and abs(r['speed']) < 0.01]
head_v0.sort(key=lambda r: r['wavelength'])

print(f"{'λ(m)':>8} {'λ/L':>8} {'Fxi(V=0)':>12} {'σ(V=0)':>10} {'Fxi(V=8)':>12} {'σ(V=8)':>10}")
for r0 in head_v0:
    matches = [r for r in head_v796 if abs(r['wavelength'] - r0['wavelength']) < 1.0]
    if matches:
        rv = matches[0]
        s0 = -r0['fxi'] / norm_raw
        sv = -rv['fxi'] / norm_raw
        print(f"{r0['wavelength']:8.1f} {r0['wavelength']/Lpp:8.3f} {r0['fxi']:12.1f} {s0:10.5f} "
              f"{rv['fxi']:12.1f} {sv:10.5f}")

# Reference: ITTC benchmark data for KVLCC2 at Fn=0.142, head seas
# From Park et al. 2016, Liu & Papanikolaou 2016, etc.
# Typical peak added resistance sigma_aw ~ 5-8 at lambda/L ~ 0.6-0.8
# At lambda/L ~ 1.0: sigma_aw ~ 2-4
# At lambda/L ~ 1.5: sigma_aw ~ 0.5-1.0
print(f"\nReference: Published KVLCC2 added resistance at Fn≈0.142:")
print(f"  lambda/L ~ 0.6-0.8: sigma_aw ≈ 5-8 (peak)")
print(f"  lambda/L ~ 1.0:     sigma_aw ≈ 2-4")
print(f"  lambda/L ~ 1.5:     sigma_aw ≈ 0.5-1.0")
print(f"  (Note: these include speed effects and Salvesen correction)")
