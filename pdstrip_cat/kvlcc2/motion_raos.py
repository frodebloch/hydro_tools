#!/usr/bin/env python3
"""
Parse heave and pitch RAOs from pdstrip.out at head seas (beta=180) and compare
with any known data. Also check the pressures: are they dominated by FK+diffraction
or radiation?
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Lpp = 328.2
B = 58.0
g = 9.81

def parse_motions(fname, target_speed=3.0, target_angle=180.0):
    """Parse pdstrip.out to extract motion RAOs at given speed and heading."""
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    results = []
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
            i += 1  # skip header
            i += 1  # Translation/k line
            m3 = re.match(r'\s*Translation/k\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', lines[i])
            surge_k = float(m3.group(1)) if m3 else 0
            sway_k = float(m3.group(2)) if m3 else 0
            heave_k = float(m3.group(3)) if m3 else 0
            i += 1  # Rotation/k
            m4 = re.match(r'\s*Rotation/k\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', lines[i])
            roll_k = float(m4.group(1)) if m4 else 0
            pitch_k = float(m4.group(2)) if m4 else 0
            yaw_k = float(m4.group(3)) if m4 else 0
            
            if abs(speed - target_speed) < 0.1 and abs(wave_angle - target_angle) < 1.0:
                lam_L = wavelength / Lpp
                # Convert Translation/k to actual RAO:
                # Translation/k means translation amplitude / (wave amplitude * k), 
                # but more commonly it's |xi|/(kA) which = |xi|/wave_slope
                # Actually in pdstrip, it's xi_amplitude / (k * wave_amplitude) for translations
                # and theta_amplitude / (k * wave_amplitude) for rotations
                # So heave RAO = heave_k * k = heave_k * (2*pi/wavelength)
                # Actually: Translation/k = |amplitude| / (wave_amplitude)
                # Since the output says "Translation/k", it's amplitude/(k*A) for translations
                # and "Rotation/k" means angle/(k*A)
                # heave/(kA) = heave_k, so heave/A = heave_k * k
                # pitch/(kA) = pitch_k, so pitch_amplitude = pitch_k * k * A
                # For RAO: heave/A = heave_k * k, pitch/(kA) = pitch_k
                k = wavenumber
                results.append({
                    'omega': omega,
                    'omega_e': omega_e,
                    'lam_L': lam_L,
                    'k': k,
                    'heave_k': heave_k,
                    'pitch_k': pitch_k,
                    'surge_k': surge_k,
                    'roll_k': roll_k,
                    'heave_A': abs(heave_k),       # heave/(kA) as given
                    'pitch_kA': abs(pitch_k),      # pitch/(kA) as given
                })
        i += 1
    
    results.sort(key=lambda x: x['lam_L'])
    return results


# Parse motions
motions_15 = parse_motions('/home/blofro/src/pdstrip_test/kvlcc2/pdstrip_15pct.out')
motions_orig = parse_motions('/home/blofro/src/pdstrip_test/kvlcc2_original/pdstrip.out')

print("HEAD SEAS (beta=180°) MOTION RAOs")
print("="*100)
print(f"{'lam/L':>7} | {'heave/kA mod':>12} | {'pitch/kA mod':>12} | {'heave/kA orig':>13} | {'pitch/kA orig':>13} | {'surge/kA mod':>12}")
print("-"*100)

for r in motions_15:
    # Find matching original
    orig_match = None
    for ro in motions_orig:
        if abs(ro['lam_L'] - r['lam_L']) < 0.01:
            orig_match = ro
            break
    
    if 0.3 <= r['lam_L'] <= 2.0:
        print(f"{r['lam_L']:7.3f} | {r['heave_A']:12.4f} | {r['pitch_kA']:12.4f} | "
              f"{orig_match['heave_A']:13.4f} | {orig_match['pitch_kA']:13.4f} | {abs(r['surge_k']):12.4f}" 
              if orig_match else 
              f"{r['lam_L']:7.3f} | {r['heave_A']:12.4f} | {r['pitch_kA']:12.4f} | {'N/A':>13} | {'N/A':>13} | {abs(r['surge_k']):12.4f}")


# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Heave RAO
ax = axes[0]
x = [r['lam_L'] for r in motions_15 if r['lam_L'] <= 3]
y = [r['heave_A'] for r in motions_15 if r['lam_L'] <= 3]
ax.plot(x, y, 'g-s', markersize=3, linewidth=1.5, label='modified (15%)')
x_o = [r['lam_L'] for r in motions_orig if r['lam_L'] <= 3]
y_o = [r['heave_A'] for r in motions_orig if r['lam_L'] <= 3]
ax.plot(x_o, y_o, 'r-^', markersize=3, linewidth=1.5, alpha=0.7, label='original')
ax.set_xlabel(r'$\lambda/L$')
ax.set_ylabel(r'$|heave|/(kA)$')
ax.set_title('Heave RAO')
ax.set_xlim(0, 2)
ax.grid(True, alpha=0.3)
ax.legend()

# Pitch RAO
ax = axes[1]
x = [r['lam_L'] for r in motions_15 if r['lam_L'] <= 3]
y = [r['pitch_kA'] for r in motions_15 if r['lam_L'] <= 3]
ax.plot(x, y, 'g-s', markersize=3, linewidth=1.5, label='modified (15%)')
x_o = [r['lam_L'] for r in motions_orig if r['lam_L'] <= 3]
y_o = [r['pitch_kA'] for r in motions_orig if r['lam_L'] <= 3]
ax.plot(x_o, y_o, 'r-^', markersize=3, linewidth=1.5, alpha=0.7, label='original')
ax.set_xlabel(r'$\lambda/L$')
ax.set_ylabel(r'$|pitch|/(kA)$')
ax.set_title('Pitch RAO')
ax.set_xlim(0, 2)
ax.grid(True, alpha=0.3)
ax.legend()

# Surge RAO
ax = axes[2]
x = [r['lam_L'] for r in motions_15 if r['lam_L'] <= 3]
y = [abs(r['surge_k']) for r in motions_15 if r['lam_L'] <= 3]
ax.plot(x, y, 'g-s', markersize=3, linewidth=1.5, label='modified (15%)')
x_o = [r['lam_L'] for r in motions_orig if r['lam_L'] <= 3]
y_o = [abs(r['surge_k']) for r in motions_orig if r['lam_L'] <= 3]
ax.plot(x_o, y_o, 'r-^', markersize=3, linewidth=1.5, alpha=0.7, label='original')
ax.set_xlabel(r'$\lambda/L$')
ax.set_ylabel(r'$|surge|/(kA)$')
ax.set_title('Surge RAO')
ax.set_xlim(0, 2)
ax.grid(True, alpha=0.3)
ax.legend()

plt.suptitle('Motion RAOs at head seas ($\\beta=180°$, V=3 m/s)', fontsize=13)
plt.tight_layout()
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/motion_raos_headseas.png', dpi=150, bbox_inches='tight')
print("\nSaved: motion_raos_headseas.png")
