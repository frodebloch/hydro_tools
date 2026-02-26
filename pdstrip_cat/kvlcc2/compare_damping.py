#!/usr/bin/env python3
"""
Compare drift forces: undamped vs 5%/15%/25% roll damping vs Seo et al. SWAN1.
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# KVLCC2 parameters
Lpp = 328.2
B = 58.0
rho = 1025.0
g = 9.81
norm = rho * g * B**2 / Lpp

# Seo data (SWAN1 only, for brevity)
seo_data = {}
seo_data[180] = [(0.30, 0.0), (0.35, 0.1), (0.40, 0.3), (0.45, 0.6), (0.50, 1.0), (0.55, 1.3), (0.60, 1.5), (0.65, 1.3), (0.70, 0.9), (0.75, 0.5), (0.80, 0.3), (0.85, 0.5), (0.90, 1.0), (0.95, 1.6), (1.00, 2.1), (1.05, 2.4), (1.10, 2.5), (1.15, 2.3), (1.20, 1.8), (1.25, 1.3), (1.30, 0.8), (1.35, 0.5), (1.40, 0.3), (1.50, 0.1), (1.60, 0.0)]
seo_data[150] = [(0.30, 0.0), (0.35, 0.3), (0.40, 0.8), (0.45, 1.5), (0.50, 2.0), (0.55, 1.7), (0.60, 1.0), (0.65, 0.5), (0.70, 0.3), (0.75, 0.3), (0.80, 0.5), (0.85, 0.8), (0.90, 1.1), (0.95, 1.3), (1.00, 1.5), (1.05, 1.4), (1.10, 1.2), (1.15, 0.9), (1.20, 0.6), (1.25, 0.4), (1.30, 0.3), (1.40, 0.3), (1.50, 0.5), (1.55, 0.5), (1.60, 0.4), (1.70, 0.2), (1.80, 0.1)]
seo_data[120] = [(0.30, 0.0), (0.35, 0.2), (0.40, 0.6), (0.45, 1.1), (0.50, 1.5), (0.55, 1.2), (0.60, 0.5), (0.65, 0.1), (0.70, -0.2), (0.75, -0.1), (0.80, 0.1), (0.85, 0.3), (0.90, 0.4), (0.95, 0.5), (1.00, 0.5), (1.05, 0.4), (1.10, 0.2), (1.15, 0.1), (1.20, 0.0), (1.30, 0.1), (1.40, 0.3), (1.50, 0.5), (1.55, 0.4), (1.60, 0.3), (1.70, 0.1), (1.80, 0.0)]
seo_data[90] = [(0.30, 0.0), (0.35, 0.3), (0.40, 0.8), (0.45, 1.5), (0.50, 2.0), (0.55, 1.6), (0.60, 0.8), (0.65, 0.2), (0.70, -0.1), (0.75, -0.3), (0.80, -0.3), (0.85, -0.2), (0.90, -0.1), (0.95, 0.0), (1.00, 0.0), (1.10, 0.0), (1.20, 0.0), (1.30, 0.1), (1.40, 0.2), (1.50, 0.3), (1.55, 0.2)]
seo_data[60] = [(0.30, 0.0), (0.35, 0.05), (0.40, 0.1), (0.45, 0.1), (0.50, 0.0), (0.55, -0.2), (0.60, -0.3), (0.65, -0.5), (0.70, -0.5), (0.75, -0.4), (0.80, -0.2), (0.85, -0.1), (0.90, 0.0), (0.95, 0.0), (1.00, 0.0), (1.10, -0.1), (1.20, -0.3), (1.30, -0.8), (1.40, -1.5), (1.50, -2.2), (1.60, -2.5), (1.70, -2.3), (1.80, -1.8), (1.90, -1.2), (2.00, -0.7)]
seo_data[30] = [(0.30, 0.0), (0.35, 0.0), (0.40, 0.0), (0.50, -0.1), (0.55, -0.1), (0.60, -0.2), (0.65, -0.2), (0.70, -0.2), (0.75, -0.1), (0.80, -0.1), (0.90, 0.0), (1.00, 0.0), (1.10, -0.1), (1.20, -0.3), (1.30, -0.7), (1.40, -1.2), (1.50, -1.7), (1.60, -2.0), (1.70, -2.0), (1.80, -1.7), (1.90, -1.3), (2.00, -0.8)]
seo_data[0] = [(0.30, 0.0), (0.40, 0.0), (0.50, -0.1), (0.60, -0.2), (0.70, -0.3), (0.80, -0.2), (0.90, -0.1), (1.00, -0.1)]


def parse_pdstrip(fname, target_speed=3.0):
    """Parse pdstrip.out and return dict {beta: [(lam_L, sigma_aw, roll_k), ...]}"""
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    results = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'\s*Wave circ\. frequency\s+([\d.]+)\s+encounter frequ\.\s+([\d.]+)\s+'
                     r'wave length\s+([\d.]+)\s+wave number\s+([\d.]+)\s+wave angle\s+([\d.]+)', line)
        if m:
            wavelength = float(m.group(3))
            wave_angle = float(m.group(5))
            i += 1
            m2 = re.match(r'\s*speed\s+([\d.]+)', lines[i])
            speed = float(m2.group(1)) if m2 else None
            i += 1; i += 1  # skip header
            i += 1  # Translation
            m3 = re.match(r'\s*Rotation/k\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', lines[i])
            roll_abs = float(m3.group(1)) if m3 else 0
            i += 1  # Drift
            m5 = re.match(r'\s*Longitudinal and transverse drift force.*?\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)', lines[i])
            fxi = float(m5.group(1)) if m5 else 0
            
            if abs(speed - target_speed) < 0.1:
                beta = int(round(wave_angle))
                lam_L = wavelength / Lpp
                sigma = -fxi / norm
                if beta not in results:
                    results[beta] = []
                results[beta].append((lam_L, sigma, roll_abs))
        i += 1
    
    for beta in results:
        results[beta].sort()
    return results


# Parse all damping levels
runs = {
    '0% (undamped)': ('pdstrip_no_rolldamp.out', 'r', '^', 0.4),
    '5%': ('pdstrip_5pct.out', 'orange', 'v', 0.5),
    '15%': ('pdstrip_15pct.out', 'g', 's', 0.8),
    '25%': ('pdstrip_25pct.out', 'm', 'D', 1.0),
}

parsed = {}
for label, (fname, _, _, _) in runs.items():
    print(f"Parsing {label}...")
    parsed[label] = parse_pdstrip(fname)

# Plot comparison for all headings
seo_betas = [180, 150, 120, 90, 60, 30, 0]
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes_flat = axes.flatten()

for idx, beta in enumerate(seo_betas):
    ax = axes_flat[idx]
    
    # Seo SWAN1
    if beta in seo_data:
        x, y = zip(*seo_data[beta])
        ax.plot(x, y, 'b-o', markersize=4, linewidth=2.5, label='SWAN1 (Seo)', zorder=10)
    
    # pdstrip at various damping levels
    for label, (fname, color, marker, alpha) in runs.items():
        res = parsed[label]
        if beta in res:
            x = [d[0] for d in res[beta]]
            y = [d[1] for d in res[beta]]
            ax.plot(x, y, color=color, marker=marker, markersize=2, linewidth=1.0, 
                    alpha=alpha, label=f'pdstrip ({label})')
    
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$\sigma_{aw}$')
    ax.set_title(f'{subplot_labels[idx]} $\\beta = {beta}°$')
    ax.set_xlim(0, 2.5)
    ylims = {180: (-1, 8), 150: (-2, 8), 120: (-3, 6), 90: (-3, 5), 60: (-4, 2), 30: (-3, 1), 0: (-1, 1)}
    ax.set_ylim(ylims.get(beta, (-5, 10)))
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    if idx == 0:
        ax.legend(fontsize=6, loc='upper right')

for idx in range(len(seo_betas), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle('Surge drift force: effect of linear roll damping\npdstrip vs Seo et al. SWAN1, V = 3 m/s (6 knots)',
             fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/damping_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: damping_comparison.png")


# Roll RAO comparison at oblique headings
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
for plot_idx, beta in enumerate([90, 120, 150, 60]):
    ax = axes2.flatten()[plot_idx]
    for label, (fname, color, marker, alpha) in runs.items():
        res = parsed[label]
        if beta in res:
            x = [d[0] for d in res[beta]]
            r = [abs(d[2]) for d in res[beta]]
            ax.plot(x, r, color=color, linewidth=1.5, alpha=max(0.5, alpha), label=label)
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel('|Roll/k|')
    ax.set_title(f'$\\beta = {beta}°$')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3)

plt.suptitle('Roll RAO: effect of linear roll damping', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/roll_rao_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: roll_rao_comparison.png")
