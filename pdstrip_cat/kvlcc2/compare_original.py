#!/usr/bin/env python3
"""
Compare drift forces: original pdstrip vs modified pdstrip (15% damping) vs Seo SWAN1.
Focus on head seas (beta=180) where roll=0 and the ~2-10x overprediction persists.
Also compare at all headings.
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

# Seo SWAN1 data
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


# Parse both outputs
print("Parsing original pdstrip output...")
original = parse_pdstrip('/home/blofro/src/pdstrip_test/kvlcc2_original/pdstrip.out')
print(f"  Found headings: {sorted(original.keys())}")

print("Parsing modified pdstrip (15% damping) output...")
modified = parse_pdstrip('/home/blofro/src/pdstrip_test/kvlcc2/pdstrip_15pct.out')
print(f"  Found headings: {sorted(modified.keys())}")

# Also parse undamped modified for reference
print("Parsing modified pdstrip (undamped) output...")
undamped = parse_pdstrip('/home/blofro/src/pdstrip_test/kvlcc2/pdstrip_no_rolldamp.out')
print(f"  Found headings: {sorted(undamped.keys())}")


# ============================================================
# Plot 1: Head seas comparison (beta=180)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

beta = 180
ax = axes[0]

# Seo SWAN1
x_seo, y_seo = zip(*seo_data[beta])
ax.plot(x_seo, y_seo, 'b-o', markersize=5, linewidth=2.5, label='SWAN1 (Seo)', zorder=10)

# Original pdstrip
if beta in original:
    x = [d[0] for d in original[beta]]
    y = [d[1] for d in original[beta]]
    ax.plot(x, y, 'r-^', markersize=3, linewidth=1.5, alpha=0.8, label='pdstrip original')

# Modified undamped
if beta in undamped:
    x = [d[0] for d in undamped[beta]]
    y = [d[1] for d in undamped[beta]]
    ax.plot(x, y, 'orange', marker='v', markersize=3, linewidth=1.5, alpha=0.6, label='pdstrip modified (0% damp)')

# Modified 15% damping
if beta in modified:
    x = [d[0] for d in modified[beta]]
    y = [d[1] for d in modified[beta]]
    ax.plot(x, y, 'g-s', markersize=3, linewidth=1.5, alpha=0.8, label='pdstrip modified (15% damp)')

ax.set_xlabel(r'$\lambda/L$')
ax.set_ylabel(r'$\sigma_{aw}$')
ax.set_title(f'Head seas ($\\beta = 180°$)')
ax.set_xlim(0, 2.0)
ax.set_ylim(-2, 12)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# Ratio plot
ax2 = axes[1]
if beta in original and beta in modified:
    # Interpolate Seo onto pdstrip wavelengths for ratio
    from scipy.interpolate import interp1d
    seo_interp = interp1d([s[0] for s in seo_data[beta]], [s[1] for s in seo_data[beta]], 
                           bounds_error=False, fill_value=np.nan)
    
    x_orig = np.array([d[0] for d in original[beta]])
    y_orig = np.array([d[1] for d in original[beta]])
    y_seo_at_orig = seo_interp(x_orig)
    
    x_mod = np.array([d[0] for d in modified[beta]])
    y_mod = np.array([d[1] for d in modified[beta]])
    y_seo_at_mod = seo_interp(x_mod)
    
    # Only compute ratio where Seo > 0.1 to avoid division by near-zero
    mask_orig = np.abs(y_seo_at_orig) > 0.1
    mask_mod = np.abs(y_seo_at_mod) > 0.1
    
    ax2.plot(x_orig[mask_orig], y_orig[mask_orig] / y_seo_at_orig[mask_orig], 
             'r-^', markersize=3, linewidth=1.5, label='original / SWAN1')
    ax2.plot(x_mod[mask_mod], y_mod[mask_mod] / y_seo_at_mod[mask_mod], 
             'g-s', markersize=3, linewidth=1.5, label='modified (15%) / SWAN1')
    ax2.axhline(y=1, color='b', linewidth=2, linestyle='--', label='perfect agreement')
    ax2.set_xlabel(r'$\lambda/L$')
    ax2.set_ylabel('pdstrip / SWAN1 ratio')
    ax2.set_title('Ratio to SWAN1 at head seas')
    ax2.set_xlim(0.3, 1.6)
    ax2.set_ylim(0, 15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

plt.suptitle('Original vs modified pdstrip at head seas ($\\beta=180°$, V=3 m/s)',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/original_vs_modified_headseas.png', dpi=150, bbox_inches='tight')
print("Saved: original_vs_modified_headseas.png")


# ============================================================
# Plot 2: All headings comparison — original vs modified vs Seo
# ============================================================
seo_betas = [180, 150, 120, 90, 60, 30, 0]
fig2, axes2 = plt.subplots(3, 3, figsize=(18, 15))
axes_flat = axes2.flatten()

for idx, beta in enumerate(seo_betas):
    ax = axes_flat[idx]
    
    # Seo SWAN1
    if beta in seo_data:
        x, y = zip(*seo_data[beta])
        ax.plot(x, y, 'b-o', markersize=4, linewidth=2.5, label='SWAN1 (Seo)', zorder=10)
    
    # Original pdstrip
    if beta in original:
        x = [d[0] for d in original[beta]]
        y = [d[1] for d in original[beta]]
        ax.plot(x, y, 'r-^', markersize=2, linewidth=1.0, alpha=0.7, label='pdstrip original')
    
    # Modified 15% damping
    if beta in modified:
        x = [d[0] for d in modified[beta]]
        y = [d[1] for d in modified[beta]]
        ax.plot(x, y, 'g-s', markersize=2, linewidth=1.0, alpha=0.8, label='pdstrip modified (15%)')
    
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$\sigma_{aw}$')
    ax.set_title(f'$\\beta = {beta}°$')
    ax.set_xlim(0, 2.5)
    ylims = {180: (-2, 12), 150: (-3, 10), 120: (-8, 6), 90: (-8, 6), 60: (-4, 2), 30: (-3, 1), 0: (-1, 1)}
    ax.set_ylim(ylims.get(beta, (-5, 10)))
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    if idx == 0:
        ax.legend(fontsize=7, loc='upper right')

for idx in range(len(seo_betas), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle('Original vs modified pdstrip vs Seo SWAN1 (V=3 m/s)',
             fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/original_vs_modified_all.png', dpi=150, bbox_inches='tight')
print("Saved: original_vs_modified_all.png")


# ============================================================
# Print head seas table for analysis
# ============================================================
print("\n" + "="*90)
print("HEAD SEAS (beta=180) DETAILED COMPARISON")
print("="*90)
print(f"{'lam/L':>7} | {'Seo SWAN1':>10} | {'pdstrip orig':>12} | {'pdstrip mod':>12} | {'orig/Seo':>9} | {'mod/Seo':>9}")
print("-"*90)

if beta in original and beta in modified:
    from scipy.interpolate import interp1d
    seo_interp = interp1d([s[0] for s in seo_data[180]], [s[1] for s in seo_data[180]],
                           bounds_error=False, fill_value=np.nan)
    
    # Use modified wavelengths as reference
    for d in modified[180]:
        lam_L = d[0]
        sig_mod = d[1]
        sig_seo = seo_interp(lam_L)
        
        # Find closest original wavelength
        orig_arr = np.array([dd[0] for dd in original[180]])
        idx_near = np.argmin(np.abs(orig_arr - lam_L))
        if abs(orig_arr[idx_near] - lam_L) < 0.02:
            sig_orig = original[180][idx_near][1]
        else:
            sig_orig = np.nan
        
        ratio_orig = sig_orig / sig_seo if not np.isnan(sig_seo) and abs(sig_seo) > 0.05 else np.nan
        ratio_mod = sig_mod / sig_seo if not np.isnan(sig_seo) and abs(sig_seo) > 0.05 else np.nan
        
        print(f"{lam_L:7.3f} | {sig_seo:10.3f} | {sig_orig:12.3f} | {sig_mod:12.3f} | {ratio_orig:9.2f} | {ratio_mod:9.2f}")
