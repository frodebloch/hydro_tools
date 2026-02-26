#!/usr/bin/env python3
"""
Compare pdstrip KVLCC2 surge drift force against Seo et al. Figure 11.
Seo, Ha, Nam, Kim - "Surge drift force on KVLCC2: V=6 knots, H/L=1/50"
Normalization: -F_x / (rho*g*A^2*B^2/L)  where A = wave amplitude
This is the SAME as our sigma_aw = -fxi / (rho*g*B^2/Lpp) since pdstrip drift is per unit amp^2.

Wave heading convention: beta=180 = head seas, beta=0 = following seas (same as pdstrip mu).

V = 6 knots = 3.086 m/s. pdstrip has speed = 3.0 m/s (close enough).
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 1. DIGITIZED DATA FROM SEO ET AL. FIGURE 11
# ============================================================
# Format: {beta: {'swan1': [(lam_L, sigma_aw), ...], 'exp_forces': [...], 'exp_linet': [...]}}
# sigma_aw = -Fx/(rho*g*A^2*B^2/L)

seo_data = {}

# ============================================================
# Re-digitized from actual Figure 11 image (Seo et al. 2021)
# V = 6 knots, H/L = 1/50
# Y-axis: -F_x/(rho*g*A^2*B^2/L)
# ============================================================

# (a) beta = 180° (head seas) — y-axis: -1 to 5
# SWAN1: rises from ~0 at short waves, peak ~1.5 at λ/L≈0.6,
# dip to ~0.3 at λ/L≈0.8, second peak ~2.5 at λ/L≈1.1, drops to ~0
seo_data[180] = {
    'swan1': [
        (0.30, 0.0), (0.35, 0.1), (0.40, 0.3), (0.45, 0.6),
        (0.50, 1.0), (0.55, 1.3), (0.60, 1.5), (0.65, 1.3),
        (0.70, 0.9), (0.75, 0.5), (0.80, 0.3), (0.85, 0.5),
        (0.90, 1.0), (0.95, 1.6), (1.00, 2.1), (1.05, 2.4),
        (1.10, 2.5), (1.15, 2.3), (1.20, 1.8), (1.25, 1.3),
        (1.30, 0.8), (1.35, 0.5), (1.40, 0.3), (1.50, 0.1),
        (1.60, 0.0), (1.80, 0.0), (2.00, 0.0), (2.50, 0.0),
    ],
    'exp_forces': [
        (0.50, 2.0), (0.60, 2.2), (0.70, 1.2), (0.80, 1.0),
        (0.90, 2.1), (1.00, 2.5), (1.05, 3.0), (1.10, 2.8),
        (1.20, 1.5), (1.30, 0.5),
    ],
    'exp_linet': [
        (0.50, 1.7), (0.60, 1.9), (0.70, 1.0), (0.80, 0.8),
        (0.90, 1.7), (1.00, 2.2), (1.05, 2.5), (1.10, 2.3),
        (1.20, 1.2),
    ],
}

# (b) beta = 150° — y-axis: -1 to 5
# SWAN1: peak ~2.0 at λ/L≈0.5, dip ~0.3 at λ/L≈0.7, second peak ~1.5 at λ/L≈1.0,
# small bump ~0.5 at λ/L≈1.5
seo_data[150] = {
    'swan1': [
        (0.30, 0.0), (0.35, 0.3), (0.40, 0.8), (0.45, 1.5),
        (0.50, 2.0), (0.55, 1.7), (0.60, 1.0), (0.65, 0.5),
        (0.70, 0.3), (0.75, 0.3), (0.80, 0.5), (0.85, 0.8),
        (0.90, 1.1), (0.95, 1.3), (1.00, 1.5), (1.05, 1.4),
        (1.10, 1.2), (1.15, 0.9), (1.20, 0.6), (1.25, 0.4),
        (1.30, 0.3), (1.40, 0.3), (1.50, 0.5), (1.55, 0.5),
        (1.60, 0.4), (1.70, 0.2), (1.80, 0.1), (2.00, 0.0),
        (2.50, 0.0),
    ],
    'exp_forces': [
        (0.50, 3.0), (0.55, 3.5), (0.60, 3.8), (0.70, 1.8),
        (0.80, 1.0), (0.90, 1.5), (1.00, 2.0), (1.10, 1.5),
        (1.20, 0.8), (1.30, 0.5),
    ],
    'exp_linet': [
        (0.50, 2.3), (0.55, 2.8), (0.60, 3.0), (0.70, 1.5),
        (0.80, 0.8), (0.90, 1.2), (1.00, 1.6), (1.10, 1.2),
        (1.20, 0.6),
    ],
}

# (c) beta = 120° — y-axis: -1 to 5
# SWAN1: peak ~1.5 at λ/L≈0.5, dip ~-0.2 at λ/L≈0.7, rise ~0.5 at λ/L≈1.0,
# bump ~0.5 near λ/L≈1.5
seo_data[120] = {
    'swan1': [
        (0.30, 0.0), (0.35, 0.2), (0.40, 0.6), (0.45, 1.1),
        (0.50, 1.5), (0.55, 1.2), (0.60, 0.5), (0.65, 0.1),
        (0.70, -0.2), (0.75, -0.1), (0.80, 0.1), (0.85, 0.3),
        (0.90, 0.4), (0.95, 0.5), (1.00, 0.5), (1.05, 0.4),
        (1.10, 0.2), (1.15, 0.1), (1.20, 0.0), (1.30, 0.1),
        (1.40, 0.3), (1.50, 0.5), (1.55, 0.4), (1.60, 0.3),
        (1.70, 0.1), (1.80, 0.0), (2.00, 0.0), (2.50, 0.0),
    ],
    'exp_forces': [
        (0.50, 3.5), (0.55, 3.8), (0.60, 4.0), (0.70, 1.5),
        (0.80, 0.5), (0.90, 0.8), (1.00, 1.0), (1.10, 0.6),
        (1.20, 0.3),
    ],
    'exp_linet': [
        (0.50, 2.5), (0.55, 3.0), (0.60, 3.3), (0.70, 1.2),
        (0.80, 0.3), (0.90, 0.6), (1.00, 0.8), (1.10, 0.4),
    ],
}

# (d) beta = 90° (beam seas) — y-axis: -2 to 3
# SWAN1: peak ~2.0 at λ/L≈0.5, drops to ~0 at λ/L≈0.7,
# slightly negative ~-0.3, small bump ~0.3 at λ/L≈1.5
seo_data[90] = {
    'swan1': [
        (0.30, 0.0), (0.35, 0.3), (0.40, 0.8), (0.45, 1.5),
        (0.50, 2.0), (0.55, 1.6), (0.60, 0.8), (0.65, 0.2),
        (0.70, -0.1), (0.75, -0.3), (0.80, -0.3), (0.85, -0.2),
        (0.90, -0.1), (0.95, 0.0), (1.00, 0.0), (1.10, 0.0),
        (1.20, 0.0), (1.30, 0.1), (1.40, 0.2), (1.50, 0.3),
        (1.55, 0.2), (1.60, 0.1), (1.70, 0.0), (1.80, 0.0),
        (2.00, 0.0), (2.50, 0.0),
    ],
    'exp_forces': [
        (0.50, 1.5), (0.60, 0.8), (0.70, 0.0), (0.80, -0.2),
        (0.90, -0.1), (1.00, 0.1), (1.10, 0.2),
    ],
    'exp_linet': [
        (0.50, 1.2), (0.60, 0.5), (0.70, -0.1), (0.80, -0.3),
        (0.90, -0.2), (1.00, 0.0),
    ],
}

# (e) beta = 60° — y-axis: -3 to 1
# SWAN1: small positive ~0.1 at short waves, drops negative,
# trough ~-0.5 at λ/L≈0.7, rises to ~0, then deeper trough ~-2.5 at λ/L≈1.5-1.7
seo_data[60] = {
    'swan1': [
        (0.30, 0.0), (0.35, 0.05), (0.40, 0.1), (0.45, 0.1),
        (0.50, 0.0), (0.55, -0.2), (0.60, -0.3), (0.65, -0.5),
        (0.70, -0.5), (0.75, -0.4), (0.80, -0.2), (0.85, -0.1),
        (0.90, 0.0), (0.95, 0.0), (1.00, 0.0), (1.10, -0.1),
        (1.20, -0.3), (1.30, -0.8), (1.40, -1.5), (1.50, -2.2),
        (1.60, -2.5), (1.70, -2.3), (1.80, -1.8), (1.90, -1.2),
        (2.00, -0.7), (2.20, -0.2), (2.50, 0.0),
    ],
    'exp_forces': [
        (0.50, 0.0), (0.60, -0.2), (0.70, -0.3), (0.80, -0.1),
        (0.90, 0.0), (1.00, 0.0),
    ],
    'exp_linet': [
        (0.50, -0.1), (0.60, -0.3), (0.70, -0.4), (0.80, -0.2),
        (0.90, -0.1),
    ],
}

# (f) beta = 30° — y-axis: -3 to 1
# SWAN1: slightly negative, then deeper trough ~-2.0 at λ/L≈1.5-1.8
seo_data[30] = {
    'swan1': [
        (0.30, 0.0), (0.35, 0.0), (0.40, 0.0), (0.50, -0.1),
        (0.55, -0.1), (0.60, -0.2), (0.65, -0.2), (0.70, -0.2),
        (0.75, -0.1), (0.80, -0.1), (0.90, 0.0), (1.00, 0.0),
        (1.10, -0.1), (1.20, -0.3), (1.30, -0.7), (1.40, -1.2),
        (1.50, -1.7), (1.60, -2.0), (1.70, -2.0), (1.80, -1.7),
        (1.90, -1.3), (2.00, -0.8), (2.20, -0.3), (2.50, 0.0),
    ],
    'exp_forces': [
        (0.50, -0.1), (0.60, -0.1), (0.70, -0.1), (0.80, -0.1),
        (0.90, 0.0), (1.00, 0.0),
    ],
    'exp_linet': [
        (0.50, -0.1), (0.60, -0.1), (0.70, -0.2), (0.80, -0.1),
        (0.90, 0.0),
    ],
}

# (g) beta = 0° (following seas) — y-axis: -3 to 0
# SWAN1: small negative values throughout, max ~-0.3 at λ/L≈0.7
seo_data[0] = {
    'swan1': [
        (0.30, 0.0), (0.40, 0.0), (0.50, -0.1), (0.60, -0.2),
        (0.70, -0.3), (0.80, -0.2), (0.90, -0.1), (1.00, -0.1),
        (1.10, -0.1), (1.20, -0.1), (1.30, -0.1), (1.50, -0.1),
        (1.80, 0.0), (2.00, 0.0), (2.50, 0.0),
    ],
    'exp_forces': [
        (0.50, -0.1), (0.60, -0.2), (0.70, -0.2), (0.80, -0.1),
        (0.90, -0.1),
    ],
    'exp_linet': [
        (0.50, -0.1), (0.60, -0.2), (0.70, -0.2), (0.80, -0.1),
    ],
}


# ============================================================
# 2. PARSE PDSTRIP RESULTS
# ============================================================
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

print(f"Total records parsed: {len(results)}")

# KVLCC2 parameters
rho = 1025.0
g = 9.81
Lpp = 328.2
B = 58.0
norm = rho * g * B**2 / Lpp  # normalization factor

print(f"Normalization: rho*g*B^2/Lpp = {norm:.1f} N/m")

# Get available speeds
speeds_all = sorted(set(r['speed'] for r in results))
print(f"Available speeds: {speeds_all}")

# ============================================================
# 3. SELECT SPEED = 3.0 m/s (closest to 6 knots = 3.086 m/s)
# ============================================================
target_speed = 3.0

# pdstrip output angles: input -90..90 maps to output 0..90 + 270..360
# mu_out=0 → following (beta=0)
# mu_out=10 → beta=10 ... mu_out=90 → beta=90
# mu_out=100 → beta=100 ... mu_out=180 → beta=180
# mu_out=190 → beta=190 ... mu_out=260 → beta=260
# mu_out=270 → beta=270 (= -90)

# Seo's angles: 0, 30, 60, 90, 120, 150, 180
# pdstrip output angles: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, ...

# Map seo beta to pdstrip output angle (they should match directly for 0-180)
seo_betas = [180, 150, 120, 90, 60, 30, 0]

# ============================================================
# 4. PRINT TABULAR COMPARISON  
# ============================================================
for beta in seo_betas:
    pdstrip_angle = float(beta)
    
    recs = [r for r in results 
            if abs(r['speed'] - target_speed) < 0.1 
            and abs(r['wave_angle'] - pdstrip_angle) < 0.5]
    recs.sort(key=lambda r: r['wavelength'])
    
    if not recs:
        print(f"\nNo pdstrip data for beta={beta}° at speed={target_speed}")
        continue
    
    print(f"\n{'='*80}")
    print(f"beta = {beta}° (head seas=180°, following=0°), speed = {target_speed} m/s")
    print(f"{'='*80}")
    print(f"{'lam(m)':>8} {'lam/L':>8} {'fxi(N/m)':>12} {'sigma_aw':>10} {'Surge':>8} {'Heave':>8} {'Pitch/k':>8} {'Roll/k':>8}")
    
    for r in recs:
        lam_L = r['wavelength'] / Lpp
        sigma = -r['fxi'] / norm  # sigma_aw = -Fx/(rho*g*B^2/L)
        print(f"{r['wavelength']:8.1f} {lam_L:8.3f} {r['fxi']:12.1f} {sigma:10.3f} "
              f"{r['surge_abs']:8.3f} {r['heave_abs']:8.3f} {r['pitch_abs']:8.3f} {r['roll_abs']:8.3f}")


# ============================================================
# 5. CREATE COMPARISON PLOTS
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes_flat = axes.flatten()

# Subplot labels matching Seo's figure
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']

for idx, beta in enumerate(seo_betas):
    ax = axes_flat[idx]
    
    # Plot Seo data
    sd = seo_data[beta]
    
    if sd['swan1']:
        x, y = zip(*sd['swan1'])
        ax.plot(x, y, 'b-o', markersize=3, label='SWAN1 (Seo)', linewidth=1.5)
    
    if sd['exp_forces']:
        x, y = zip(*sd['exp_forces'])
        ax.plot(x, y, 's', color='gray', markersize=6, label='Exp ForceS')
    
    if sd['exp_linet']:
        x, y = zip(*sd['exp_linet'])
        ax.plot(x, y, 'D', color='orange', markersize=5, label='Exp LineT')
    
    # Plot pdstrip
    pdstrip_angle = float(beta)
    recs = [r for r in results 
            if abs(r['speed'] - target_speed) < 0.1 
            and abs(r['wave_angle'] - pdstrip_angle) < 0.5]
    recs.sort(key=lambda r: r['wavelength'])
    
    if recs:
        lam_L = [r['wavelength'] / Lpp for r in recs]
        sigma = [-r['fxi'] / norm for r in recs]
        ax.plot(lam_L, sigma, 'r-^', markersize=4, label='pdstrip', linewidth=1.5)
    
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$-F_x/(\rho g A^2 B^2/L)$')
    ax.set_title(f'{subplot_labels[idx]} $\\beta = {beta}°$')
    ax.set_xlim(0, 2.5)
    # Y-axis limits matching Seo paper Figure 11
    ylims = {180: (-1, 5), 150: (-1, 5), 120: (-1, 5), 90: (-2, 3), 60: (-3, 1), 30: (-3, 1), 0: (-3, 0.5)}
    if beta in ylims:
        ax.set_ylim(ylims[beta])
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='-')
    ax.grid(True, alpha=0.3)
    
    if idx == 0:
        ax.legend(fontsize=8, loc='upper right')

# Remove unused subplots
for idx in range(len(seo_betas), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle('Surge drift force on KVLCC2: V ≈ 6 knots (3 m/s)\nComparison: pdstrip vs Seo et al. (SWAN1 + Exp)', 
             fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('seo_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to seo_comparison.png")

# ============================================================
# 6. Also do zero speed for reference
# ============================================================
fig2, axes2 = plt.subplots(3, 3, figsize=(16, 14))
axes2_flat = axes2.flatten()

for idx, beta in enumerate(seo_betas):
    ax = axes2_flat[idx]
    
    # Plot Seo data (these are at V=6kts, but useful for shape comparison)
    sd = seo_data[beta]
    if sd['swan1']:
        x, y = zip(*sd['swan1'])
        ax.plot(x, y, 'b-o', markersize=3, label='SWAN1 (6kts)', linewidth=1.0, alpha=0.5)
    if sd['exp_forces']:
        x, y = zip(*sd['exp_forces'])
        ax.plot(x, y, 's', color='gray', markersize=5, alpha=0.4, label='Exp (6kts)')
    
    # pdstrip at V=3 m/s
    pdstrip_angle = float(beta)
    recs_3 = [r for r in results 
              if abs(r['speed'] - 3.0) < 0.1 
              and abs(r['wave_angle'] - pdstrip_angle) < 0.5]
    recs_3.sort(key=lambda r: r['wavelength'])
    if recs_3:
        lam_L = [r['wavelength'] / Lpp for r in recs_3]
        sigma = [-r['fxi'] / norm for r in recs_3]
        ax.plot(lam_L, sigma, 'r-^', markersize=4, label='pdstrip V=3', linewidth=1.5)
    
    # pdstrip at V=0
    recs_0 = [r for r in results 
              if abs(r['speed']) < 0.01 
              and abs(r['wave_angle'] - pdstrip_angle) < 0.5]
    recs_0.sort(key=lambda r: r['wavelength'])
    if recs_0:
        lam_L = [r['wavelength'] / Lpp for r in recs_0]
        sigma = [-r['fxi'] / norm for r in recs_0]
        ax.plot(lam_L, sigma, 'g--s', markersize=3, label='pdstrip V=0', linewidth=1.0)
    
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$-F_x/(\rho g A^2 B^2/L)$')
    ax.set_title(f'{subplot_labels[idx]} $\\beta = {beta}°$')
    ax.set_xlim(0, 2.5)
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='-')
    ax.grid(True, alpha=0.3)
    
    if idx == 0:
        ax.legend(fontsize=7, loc='upper right')

for idx in range(len(seo_betas), len(axes2_flat)):
    axes2_flat[idx].set_visible(False)

plt.suptitle('Surge drift force on KVLCC2: pdstrip (V=0 & V=3) vs Seo et al. (V=6kts)\nShape/magnitude comparison', 
             fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('seo_comparison_speeds.png', dpi=150, bbox_inches='tight')
print(f"Speed comparison plot saved to seo_comparison_speeds.png")
