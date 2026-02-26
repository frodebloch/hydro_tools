#!/usr/bin/env python3
"""Parse KVLCC2 pdstrip.out and extract drift forces and RAOs."""
import re
import numpy as np

results = []

with open('pdstrip.out', 'r') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i]
    
    # Match: Wave circ. frequency ... wave length ... wave angle ...
    m = re.match(r'\s*Wave circ\. frequency\s+([\d.]+)\s+encounter frequ\.\s+([\d.]+)\s+'
                 r'wave length\s+([\d.]+)\s+wave number\s+([\d.]+)\s+wave angle\s+([\d.]+)', line)
    if m:
        omega = float(m.group(1))
        omega_e = float(m.group(2))
        wavelength = float(m.group(3))
        wavenumber = float(m.group(4))
        wave_angle = float(m.group(5))
        
        # Next line: speed
        i += 1
        m2 = re.match(r'\s*speed\s+([\d.]+)', lines[i])
        speed = float(m2.group(1)) if m2 else None
        
        # Skip header line
        i += 1
        
        # Translation line
        i += 1
        m3 = re.match(r'\s*Translation\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                       r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                       r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', lines[i])
        if m3:
            surge_re, surge_im, surge_abs = float(m3.group(1)), float(m3.group(2)), float(m3.group(3))
            sway_re, sway_im, sway_abs = float(m3.group(4)), float(m3.group(5)), float(m3.group(6))
            heave_re, heave_im, heave_abs = float(m3.group(7)), float(m3.group(8)), float(m3.group(9))
        
        # Rotation/k line
        i += 1
        m4 = re.match(r'\s*Rotation/k\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                       r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                       r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', lines[i])
        if m4:
            roll_re, roll_im, roll_abs = float(m4.group(1)), float(m4.group(2)), float(m4.group(3))
            pitch_re, pitch_im, pitch_abs = float(m4.group(4)), float(m4.group(5)), float(m4.group(6))
            yaw_re, yaw_im, yaw_abs = float(m4.group(7)), float(m4.group(8)), float(m4.group(9))
        
        # Drift force line
        i += 1
        m5 = re.match(r'\s*Longitudinal and transverse drift force.*?\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)', lines[i])
        if m5:
            fxi = float(m5.group(1))
            feta = float(m5.group(2))
        
        # Yaw drift moment
        i += 1
        m6 = re.match(r'\s*Yaw drift moment.*?\s+([-\d.Ee+]+)', lines[i])
        yaw_drift = float(m6.group(1)) if m6 else 0.0
        
        # Roll drift moment
        i += 1
        m7 = re.match(r'\s*Roll drift moment.*?\s+([-\d.Ee+]+)', lines[i])
        roll_drift = float(m7.group(1)) if m7 else 0.0
        
        results.append({
            'omega': omega, 'omega_e': omega_e, 'wavelength': wavelength, 
            'wavenumber': wavenumber, 'wave_angle': wave_angle, 'speed': speed,
            'surge_abs': surge_abs, 'sway_abs': sway_abs, 'heave_abs': heave_abs,
            'roll_abs': roll_abs, 'pitch_abs': pitch_abs, 'yaw_abs': yaw_abs,
            'fxi': fxi, 'feta': feta, 'yaw_drift': yaw_drift, 'roll_drift': roll_drift,
        })
    i += 1

print(f"Total records parsed: {len(results)}")

# Focus on zero speed
zero_speed = [r for r in results if abs(r['speed']) < 0.01]
print(f"Zero speed records: {len(zero_speed)}")

# Get unique wavelengths and angles
wavelengths = sorted(set(r['wavelength'] for r in zero_speed))
angles = sorted(set(r['wave_angle'] for r in zero_speed))
print(f"Wavelengths: {len(wavelengths)}")
print(f"Wave angles: {angles}")

# Beam seas (angle 90 or 270 deg)
# pdstrip adds angles >90 automatically: 90 stays as 90, and 
# the -90 input becomes 270 (=360-90) in the output
# Let's check what angles we actually have
print(f"\nAll unique angles: {sorted(set(r['wave_angle'] for r in results))}")

# Extract beam seas (90 deg) at zero speed
beam_90 = [r for r in zero_speed if abs(r['wave_angle'] - 90.0) < 0.1]
beam_270 = [r for r in zero_speed if abs(r['wave_angle'] - 270.0) < 0.1]
head_180 = [r for r in zero_speed if abs(r['wave_angle'] - 180.0) < 0.1]

print(f"\nBeam seas (90°) zero-speed records: {len(beam_90)}")
print(f"Beam seas (270°) zero-speed records: {len(beam_270)}")
print(f"Head seas (180°) zero-speed records: {len(head_180)}")

# Print drift forces for beam seas 90° at zero speed
if beam_90:
    print(f"\n{'=== BEAM SEAS 90° - Zero Speed ===':^80}")
    print(f"{'λ(m)':>8} {'kL':>8} {'Fxi':>14} {'Feta':>14} {'Sway':>8} {'Heave':>8} {'Roll/k':>8} {'Pitch/k':>8}")
    beam_90_sorted = sorted(beam_90, key=lambda r: r['wavelength'])
    Lpp = 328.0  # approximate
    for r in beam_90_sorted:
        kL = r['wavenumber'] * Lpp
        print(f"{r['wavelength']:8.1f} {kL:8.2f} {r['fxi']:14.1f} {r['feta']:14.1f} "
              f"{r['sway_abs']:8.3f} {r['heave_abs']:8.3f} {r['roll_abs']:8.3f} {r['pitch_abs']:8.3f}")

# Print head seas 180°
if head_180:
    print(f"\n{'=== HEAD SEAS 180° - Zero Speed ===':^80}")
    print(f"{'λ(m)':>8} {'kL':>8} {'Fxi':>14} {'Feta':>14} {'Surge':>8} {'Heave':>8} {'Pitch/k':>8}")
    head_180_sorted = sorted(head_180, key=lambda r: r['wavelength'])
    for r in head_180_sorted:
        kL = r['wavenumber'] * Lpp
        print(f"{r['wavelength']:8.1f} {kL:8.2f} {r['fxi']:14.1f} {r['feta']:14.1f} "
              f"{r['surge_abs']:8.3f} {r['heave_abs']:8.3f} {r['pitch_abs']:8.3f}")

# Also print normalized drift: Fxi/(rho*g*B^2/2) and Feta/(rho*g*B^2/2)
rho = 1025.0
g = 9.81
B = 58.0  
Lpp = 328.0
norm = rho * g * B**2 / 2  # ~ 16.9 MN/m

print(f"\nNormalization: rho*g*B^2/2 = {norm:.0f} N/m")
print(f"  (drift forces are per wave amplitude squared, in N/m^2)")

# Nondimensional drift for head seas
if head_180:
    print(f"\n{'=== HEAD SEAS 180° - Nondimensional Fxi/(rho*g*B²/2) ===':^80}")
    print(f"{'λ(m)':>8} {'λ/L':>8} {'Fxi_nd':>12} {'Feta_nd':>12}")
    for r in sorted(head_180, key=lambda r: r['wavelength']):
        lam_L = r['wavelength'] / Lpp
        # The drift force output is per wave amplitude squared
        fxi_nd = r['fxi'] / norm
        feta_nd = r['feta'] / norm
        print(f"{r['wavelength']:8.1f} {lam_L:8.3f} {fxi_nd:12.6f} {feta_nd:12.6f}")
