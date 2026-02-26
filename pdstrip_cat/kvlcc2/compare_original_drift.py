#!/usr/bin/env python3
"""
Compare original (unmodified) pdstrip drift force against our modified version.
Checks whether the Pinkster overprediction existed in the original code.
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

Lpp = 328.2
B = 58.0
rho = 1025.0
g = 9.81
norm = rho * g * B**2 / Lpp

# SWAN1 head seas data
seo_180 = [
    (0.30, 0.0), (0.35, 0.1), (0.40, 0.3), (0.45, 0.6),
    (0.50, 1.0), (0.55, 1.3), (0.60, 1.5), (0.65, 1.3),
    (0.70, 0.9), (0.75, 0.5), (0.80, 0.3), (0.85, 0.5),
    (0.90, 1.0), (0.95, 1.6), (1.00, 2.1), (1.05, 2.4),
    (1.10, 2.5), (1.15, 2.3), (1.20, 1.8), (1.25, 1.3),
    (1.30, 0.8), (1.35, 0.5), (1.40, 0.3), (1.50, 0.1),
    (1.60, 0.0), (1.80, 0.0), (2.00, 0.0), (2.50, 0.0),
]


def parse_drift_from_pdstrip_out(fname):
    """Parse 'Longitudinal and transverse drift force' lines from pdstrip.out"""
    results = []
    with open(fname) as f:
        for line in f:
            if 'Longitudinal and transverse drift force' in line:
                parts = line.split()
                fxi = float(parts[-2])
                feta = float(parts[-1])
                results.append((fxi, feta))
    return results


def parse_omega_mu_from_out(fname):
    """Parse omega and heading from pdstrip output to match drift lines"""
    current_omega = None
    current_mu = None
    drift_records = []
    
    with open(fname) as f:
        for line in f:
            if 'Wave circ. frequency' in line:
                parts = line.split()
                # format: "Wave circ. frequency 1.500 encounter ... wave angle 90.0"
                current_omega = float(parts[3])
                current_mu = float(parts[-1])
            elif 'Longitudinal and transverse drift force' in line:
                parts = line.split()
                fxi = float(parts[-2])
                feta = float(parts[-1])
                drift_records.append({
                    'omega': current_omega,
                    'mu': current_mu,
                    'fxi': fxi,
                    'feta': feta,
                })
    return drift_records


# Parse both files
print("Parsing original pdstrip.out...")
orig = parse_omega_mu_from_out('/home/blofro/src/pdstrip_test/kvlcc2_original/pdstrip.out')
print(f"  Found {len(orig)} drift records")

print("Parsing modified pdstrip_15pct.out...")
mod = parse_omega_mu_from_out('/home/blofro/src/pdstrip_test/kvlcc2/pdstrip_15pct.out')
print(f"  Found {len(mod)} drift records")

# Extract unique omegas/headings
all_omegas = sorted(set(r['omega'] for r in orig if r['omega'] is not None))
all_mus = sorted(set(r['mu'] for r in orig if r['mu'] is not None))
print(f"  {len(all_omegas)} omegas, {len(all_mus)} headings")

# For head seas (mu=180), speed_idx=2, extract comparison
# Data order: for each omega: for each speed: for each heading
n_headings = len(all_mus)
n_omegas = len(all_omegas)
n_speeds = len(orig) // (n_omegas * n_headings)
print(f"  {n_speeds} speeds")

# Find heading index for 180 degrees
mu_target = 180.0
mu_idx = None
for i, m in enumerate(all_mus):
    if abs(m - mu_target) < 1.0:
        mu_idx = i
        break
print(f"  mu=180 index: {mu_idx}")

speed_idx = 2  # V=3 m/s

# Extract head seas data for both
def extract_heading_data(records, n_omegas, n_speeds, n_headings, mu_idx, speed_idx):
    results = []
    for iom in range(n_omegas):
        idx = iom * (n_speeds * n_headings) + speed_idx * n_headings + mu_idx
        if idx >= len(records):
            continue
        r = records[idx]
        if r['omega'] is None:
            continue
        wavelength = 2 * np.pi * g / r['omega']**2
        lam_L = wavelength / Lpp
        sigma = -r['fxi'] / norm
        results.append({'lam_L': lam_L, 'sigma': sigma, 'omega': r['omega']})
    results.sort(key=lambda x: x['lam_L'])
    return results

orig_180 = extract_heading_data(orig, n_omegas, n_speeds, n_headings, mu_idx, speed_idx)
mod_180 = extract_heading_data(mod, n_omegas, n_speeds, n_headings, mu_idx, speed_idx)

# Print comparison table
print("\n" + "="*90)
print("HEAD SEAS (beta=180): Original vs Modified pdstrip vs SWAN1")
print("="*90)
print(f"{'lam/L':>7} | {'Original':>10} | {'Modified':>10} | {'Ratio':>8} | {'SWAN1':>7}")
print("-"*90)

seo_interp = interp1d([s[0] for s in seo_180], [s[1] for s in seo_180],
                       bounds_error=False, fill_value=np.nan)

for o, m in zip(orig_180, mod_180):
    seo_val = seo_interp(o['lam_L'])
    ratio = m['sigma'] / o['sigma'] if abs(o['sigma']) > 0.01 else float('nan')
    sv = f"{seo_val:7.2f}" if not np.isnan(seo_val) else "    ---"
    print(f"{o['lam_L']:7.3f} | {o['sigma']:10.3f} | {m['sigma']:10.3f} | {ratio:8.3f} | {sv}")

# Also extract a few other headings for quick check
print("\n\nPeak values at key headings:")
for mu_target in [180, 150, 120, 90, 60, 30, 0]:
    mi = None
    for i, m in enumerate(all_mus):
        if abs(m - mu_target) < 1.0:
            mi = i
            break
    if mi is None:
        continue
    o_data = extract_heading_data(orig, n_omegas, n_speeds, n_headings, mi, speed_idx)
    m_data = extract_heading_data(mod, n_omegas, n_speeds, n_headings, mi, speed_idx)
    if mu_target >= 90:
        peak_o = max(r['sigma'] for r in o_data) if o_data else 0
        peak_m = max(r['sigma'] for r in m_data) if m_data else 0
    else:
        peak_o = min(r['sigma'] for r in o_data) if o_data else 0
        peak_m = min(r['sigma'] for r in m_data) if m_data else 0
    print(f"  beta={mu_target:3d}: Original={peak_o:7.2f}  Modified={peak_m:7.2f}  Ratio={peak_m/peak_o if abs(peak_o)>0.01 else float('nan'):6.2f}")


# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot([r['lam_L'] for r in orig_180], [r['sigma'] for r in orig_180], 'k-o', 
        markersize=3, linewidth=1.5, label='Original pdstrip')
ax.plot([r['lam_L'] for r in mod_180], [r['sigma'] for r in mod_180], 'r-^',
        markersize=3, linewidth=1.5, label='Modified pdstrip (15% damp)')
ax.plot([s[0] for s in seo_180], [s[1] for s in seo_180], 'b-s',
        markersize=4, linewidth=2, label='SWAN1', zorder=10)
ax.set_xlabel(r'$\lambda/L$')
ax.set_ylabel(r'$\sigma_{aw}$')
ax.set_title(r'Head seas ($\beta=180°$): Original vs Modified vs SWAN1')
ax.set_xlim(0.2, 2.0)
ax.set_ylim(-1, 8)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
ratios = []
x_ratio = []
for o, m in zip(orig_180, mod_180):
    if abs(o['sigma']) > 0.05:
        x_ratio.append(o['lam_L'])
        ratios.append(m['sigma'] / o['sigma'])
ax.plot(x_ratio, ratios, 'g-o', markersize=4, linewidth=2)
ax.axhline(y=1, color='k', linewidth=1, linestyle='--')
ax.set_xlabel(r'$\lambda/L$')
ax.set_ylabel('Modified / Original')
ax.set_title('Ratio of modified to original drift force')
ax.set_xlim(0.2, 2.0)
ax.set_ylim(0.5, 2.0)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/original_vs_modified_drift.png', dpi=150, bbox_inches='tight')
print("\nSaved: original_vs_modified_drift.png")
