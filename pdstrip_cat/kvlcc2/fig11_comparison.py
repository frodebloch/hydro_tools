#!/usr/bin/env python3
"""
Reproduce Liu & Papanikolaou (2021) Figure 11: Surge drift force on KVLCC2
V = 6 knots, H/L = 1/50
Normalization: -F_x / (rho*g*A^2*B^2/L)

Compare pdstrip (15% roll damping, V=3 m/s) against:
  - SWAN1 (digitized from Fig 11)
  - Exp. ForceS (gray squares)
  - Exp. LineT (orange diamonds)
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 1. Digitized Seo/Liu data from Figure 11
# ============================================================
seo_data = {}

# (a) beta = 180° (head seas) — y-axis: -1 to 5
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

# (b) beta = 150°
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

# (c) beta = 120°
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

# (d) beta = 90° (beam seas)
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

# (e) beta = 60°
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

# (f) beta = 30°
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

# (g) beta = 0° (following seas)
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
# 2. Parse pdstrip output (15% damping run)
# ============================================================
results = []
with open('pdstrip_15pct.out', 'r') as f:
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
            heave_abs = float(m3.group(9))
        i += 1  # Rotation
        i += 1  # Drift
        m5 = re.match(r'\s*Longitudinal and transverse drift force.*?\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)', lines[i])
        fxi = float(m5.group(1)) if m5 else 0
        feta = float(m5.group(2)) if m5 else 0

        results.append({
            'omega': omega, 'omega_e': omega_e, 'wavelength': wavelength,
            'wavenumber': wavenumber, 'wave_angle': wave_angle, 'speed': speed,
            'fxi': fxi, 'feta': feta,
        })
    i += 1

print(f"Parsed {len(results)} records from pdstrip_15pct.out")

# KVLCC2 parameters
rho = 1025.0
g = 9.81
Lpp = 328.2
B = 58.0
norm = rho * g * B**2 / Lpp  # normalization factor
target_speed = 3.0

speeds_all = sorted(set(r['speed'] for r in results))
print(f"Available speeds: {speeds_all}")
print(f"Normalization: rho*g*B^2/Lpp = {norm:.1f} N/m^2")

# ============================================================
# 3. Create Figure 11 comparison (7 panels)
# ============================================================
seo_betas = [180, 150, 120, 90, 60, 30, 0]
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']

# Y-axis limits matching Figure 11 in the paper
ylims = {180: (-1, 5), 150: (-1, 5), 120: (-1, 5), 90: (-2, 3), 60: (-3, 1), 30: (-3, 1), 0: (-3, 1)}

fig, axes = plt.subplots(3, 3, figsize=(15, 13))
axes_flat = axes.flatten()

for idx, beta in enumerate(seo_betas):
    ax = axes_flat[idx]
    sd = seo_data[beta]

    # --- SWAN1 ---
    if sd['swan1']:
        x, y = zip(*sd['swan1'])
        ax.plot(x, y, 'b-o', markersize=4, linewidth=1.5, label='Cal. (SWAN1)', zorder=5)

    # --- Exp ForceS ---
    if sd['exp_forces']:
        x, y = zip(*sd['exp_forces'])
        ax.plot(x, y, 's', color='gray', markersize=7, markerfacecolor='gray',
                markeredgecolor='black', markeredgewidth=0.5, label='Exp. (ForceS)', zorder=4)

    # --- Exp LineT ---
    if sd['exp_linet']:
        x, y = zip(*sd['exp_linet'])
        ax.plot(x, y, 'D', color='orange', markersize=6, markerfacecolor='orange',
                markeredgecolor='black', markeredgewidth=0.5, label='Exp. (LineT)', zorder=4)

    # --- pdstrip ---
    pdstrip_angle = float(beta)
    recs = [r for r in results
            if abs(r['speed'] - target_speed) < 0.1
            and abs(r['wave_angle'] - pdstrip_angle) < 0.5]
    recs.sort(key=lambda r: r['wavelength'])

    if recs:
        lam_L = [r['wavelength'] / Lpp for r in recs]
        sigma = [-r['fxi'] / norm for r in recs]
        ax.plot(lam_L, sigma, 'r-^', markersize=4, linewidth=1.5, label='pdstrip', zorder=6)

    # --- Formatting ---
    ax.set_xlabel(r'$\lambda/L$', fontsize=11)
    ax.set_ylabel(r'$-F_x/(\rho g A^2 B^2/L)$', fontsize=10)
    ax.set_title(f'{subplot_labels[idx]} $\\beta = {beta}°$', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 2.5)
    if beta in ylims:
        ax.set_ylim(ylims[beta])
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='-')
    ax.grid(True, alpha=0.3)

    if idx == 0:
        ax.legend(fontsize=8, loc='upper right')

# Hide unused subplots
for idx in range(len(seo_betas), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle('Surge drift force on KVLCC2: V = 6 knots, H/L = 1/50\n'
             'pdstrip (15% roll damping, V=3 m/s) vs Liu & Papanikolaou (2021) Fig. 11',
             fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('fig11_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: fig11_comparison.png")

# ============================================================
# 4. Print tabular comparison at head seas for quick reference
# ============================================================
from scipy.interpolate import interp1d
seo_180 = seo_data[180]['swan1']
seo_interp = interp1d([s[0] for s in seo_180], [s[1] for s in seo_180],
                       bounds_error=False, fill_value=np.nan)

recs_180 = [r for r in results
            if abs(r['speed'] - target_speed) < 0.1
            and abs(r['wave_angle'] - 180.0) < 0.5]
recs_180.sort(key=lambda r: r['wavelength'])

print(f"\n{'='*75}")
print(f"HEAD SEAS (beta=180°) — pdstrip vs SWAN1")
print(f"{'='*75}")
print(f"{'lam/L':>7} | {'pdstrip':>9} | {'SWAN1':>9} | {'ratio':>7}")
print(f"{'-'*42}")
for r in recs_180:
    ll = r['wavelength'] / Lpp
    sigma_pd = -r['fxi'] / norm
    sigma_seo = seo_interp(ll)
    if not np.isnan(sigma_seo) and abs(sigma_seo) > 0.05:
        ratio = sigma_pd / sigma_seo
        print(f"{ll:7.3f} | {sigma_pd:9.3f} | {sigma_seo:9.3f} | {ratio:7.2f}")
    else:
        print(f"{ll:7.3f} | {sigma_pd:9.3f} | {sigma_seo:9.3f} |    ---")
