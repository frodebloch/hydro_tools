#!/usr/bin/env python3
"""Nondimensionalize KVLCC2 drift forces for ITTC comparison."""
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
        speed = float(m2.group(1)) if m2 else None
        i += 1  # header
        i += 1  # Translation
        m3 = re.match(r'\s*Translation\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                       r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                       r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', lines[i])
        heave_abs = float(m3.group(9)) if m3 else 0
        i += 1  # Rotation
        m4 = re.match(r'\s*Rotation/k\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                       r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', lines[i])
        pitch_abs = float(m4.group(6)) if m4 else 0
        i += 1  # Drift
        m5 = re.match(r'\s*Longitudinal and transverse drift force.*?\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)', lines[i])
        fxi = float(m5.group(1)) if m5 else 0
        feta = float(m5.group(2)) if m5 else 0
        
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
Lpp = 325.5  # KVLCC2 Lpp between perpendiculars (from SIMMAN/SHOPERA data)
# Actually let me use the section range from the geometry
# x ranges from -165.386 to 162.800, so Lpp ≈ 328.2
Lpp = 328.2
B = 58.0
T = 20.8

# Normalization for added resistance: sigma_aw = Raw / (rho * g * zeta_a^2 * B^2 / Lpp)
# Since Fxi = drift_per_amp_sq, Raw = -Fxi (in head seas convention)
# sigma_aw = -Fxi / (rho * g * B^2 / Lpp)
norm_raw = rho * g * B**2 / Lpp

# Also Newman normalization: C_aw = Raw / (rho * g * L) 
# where Raw already divided by zeta_a^2
# Actually many papers use: C = R_aw*L/(rho*g*B^2*zeta_a^2)
# which is sigma = R_aw/(rho*g*zeta_a^2*B^2/L)

print(f"KVLCC2 Parameters: Lpp={Lpp}m, B={B}m, T={T}m")
print(f"Normalization factor rho*g*B^2/Lpp = {norm_raw:.0f} N/m")
print()

# HEAD SEAS - zero speed
head_v0 = [r for r in results if abs(r['wave_angle'] - 180.0) < 0.1 and abs(r['speed']) < 0.01]
head_v0.sort(key=lambda r: r['wavelength'])

print(f"{'=== HEAD SEAS (180°) ADDED RESISTANCE - Zero Speed ===':^80}")
print(f"{'λ(m)':>8} {'λ/Lpp':>8} {'ω√(L/g)':>8} {'Fxi':>12} {'σ_aw':>10} {'Heave':>8} {'Pitch/k':>8}")
for r in head_v0:
    lam_L = r['wavelength'] / Lpp
    omega_nd = r['omega'] * np.sqrt(Lpp / g)
    sigma = -r['fxi'] / norm_raw  # positive = resistance
    print(f"{r['wavelength']:8.1f} {lam_L:8.3f} {omega_nd:8.3f} {r['fxi']:12.1f} {sigma:10.6f} "
          f"{r['heave_abs']:8.3f} {r['pitch_abs']:8.3f}")

# BEAM SEAS - zero speed  
beam_v0 = [r for r in results if abs(r['wave_angle'] - 90.0) < 0.1 and abs(r['speed']) < 0.01]
beam_v0.sort(key=lambda r: r['wavelength'])

# Normalization for transverse drift: sigma_y = Feta / (rho * g * B)
# Per unit wave amplitude squared
norm_beam = rho * g * B

print(f"\n{'=== BEAM SEAS (90°) TRANSVERSE DRIFT - Zero Speed ===':^80}")
print(f"Normalization: rho*g*B = {norm_beam:.0f} N/m")
print(f"{'λ(m)':>8} {'λ/B':>8} {'Feta':>14} {'σ_y':>10} {'Fxi':>12}")
for r in beam_v0:
    lam_B = r['wavelength'] / B
    sigma_y = r['feta'] / norm_beam
    print(f"{r['wavelength']:8.1f} {lam_B:8.2f} {r['feta']:14.1f} {sigma_y:10.4f} {r['fxi']:12.1f}")

# Check: do higher speeds show reasonable trends?
print(f"\n{'=== HEAD SEAS (180°) - Multiple Speeds ===':^80}")
speeds = sorted(set(r['speed'] for r in results))
print(f"Available speeds: {speeds}")

# Pick a representative wavelength near lambda/L = 1
target_lam = 310.0  # close to Lpp
print(f"\nAt λ ≈ {target_lam}m (λ/L ≈ {target_lam/Lpp:.2f}):")
print(f"{'Speed':>8} {'Fxi':>12} {'σ_aw':>10} {'Heave':>8} {'Pitch/k':>8}")
for spd in speeds:
    rec = [r for r in results if abs(r['wave_angle'] - 180.0) < 0.1 
           and abs(r['speed'] - spd) < 0.01
           and abs(r['wavelength'] - target_lam) < 20]
    if rec:
        r = rec[0]
        sigma = -r['fxi'] / norm_raw
        print(f"{r['speed']:8.2f} {r['fxi']:12.1f} {sigma:10.6f} {r['heave_abs']:8.3f} {r['pitch_abs']:8.3f}")
