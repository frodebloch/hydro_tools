#!/usr/bin/env python3
"""
Parse debug output files to extract drift force component decomposition.
Compare WL/vel/rot at head seas (beta=180) between modified (15% damping) and look at 
component-level behavior to understand the ~4-10x overprediction.
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

# Seo SWAN1 data at head seas
seo_180 = [(0.30, 0.0), (0.35, 0.1), (0.40, 0.3), (0.45, 0.6), (0.50, 1.0), (0.55, 1.3), (0.60, 1.5), (0.65, 1.3), (0.70, 0.9), (0.75, 0.5), (0.80, 0.3), (0.85, 0.5), (0.90, 1.0), (0.95, 1.6), (1.00, 2.1), (1.05, 2.4), (1.10, 2.5), (1.15, 2.3), (1.20, 1.8), (1.25, 1.3), (1.30, 0.8), (1.35, 0.5), (1.40, 0.3), (1.50, 0.1), (1.60, 0.0)]


def parse_debug_components(fname, target_speed_idx=2, target_mu=180.0):
    """
    Parse debug.out to extract drift force components at specific speed and heading.
    
    Data layout: for each omega (35 total) -> for each speed (8) -> for each heading (36)
    So record index = iom * (8*36) + iv * 36 + imu
    
    We need speed index 2 (V=3 m/s) and the heading corresponding to mu=180.
    
    Headings in order: 
    Input: -90 -80 -70 -60 -50 -40 -30 -20 -10 0 10 20 30 40 50 60 70 80 90 (19 angles)
    Extended by symmetry to 36 angles: -90...90 then 100 110 ... 260
    mu=180 should be in the extended range.
    
    Input angles (degrees): -90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90
    Extended: index 0=-90, 1=-80, ..., 9=0, ..., 18=90, 19=100, ..., 27=180, ..., 35=260
    So mu=180 is at heading index 27.
    """
    target_mu_idx = None
    
    # Read DRIFT_START and DRIFT_TOTAL lines
    starts = []
    totals = []
    
    print(f"Parsing {fname}...")
    with open(fname, 'r') as f:
        for line in f:
            if 'DRIFT_START' in line:
                m = re.search(r'omega=\s*([\d.]+)\s+mu=\s*([-\d.]+)', line)
                if m:
                    starts.append((float(m.group(1)), float(m.group(2))))
            elif 'DRIFT_TOTAL' in line:
                m = re.search(r'fxi=\s*([-\d.Ee+]+)\s+feta=\s*([-\d.Ee+]+)\s+'
                              r'fxi_WL=\s*([-\d.Ee+]+)\s+feta_WL=\s*([-\d.Ee+]+)\s+'
                              r'fxi_vel=\s*([-\d.Ee+]+)\s+fxi_rot=\s*([-\d.Ee+]+)', line)
                if m:
                    totals.append({
                        'fxi': float(m.group(1)),
                        'feta': float(m.group(2)),
                        'fxi_WL': float(m.group(3)),
                        'feta_WL': float(m.group(4)),
                        'fxi_vel': float(m.group(5)),
                        'fxi_rot': float(m.group(6)),
                    })
    
    print(f"  Found {len(starts)} DRIFT_START, {len(totals)} DRIFT_TOTAL records")
    assert len(starts) == len(totals), "Mismatch between start and total records"
    
    # Find unique omegas
    omegas = sorted(set(s[0] for s in starts))
    # Find unique mus
    mus = sorted(set(s[1] for s in starts))
    print(f"  {len(omegas)} omega values, {len(mus)} mu values")
    print(f"  Headings: {mus}")
    
    # Find records matching target speed and heading
    # Layout: for each omega -> for each speed (8) -> for each heading (36)
    n_speeds = 8
    n_headings = len(mus)
    n_omegas = len(omegas)
    
    # Find the mu index for 180.0
    mu_idx = mus.index(target_mu) if target_mu in mus else None
    if mu_idx is None:
        # Try close match
        for i, m in enumerate(mus):
            if abs(m - target_mu) < 1.0:
                mu_idx = i
                break
    
    if mu_idx is None:
        print(f"  ERROR: Could not find mu={target_mu} in headings!")
        return None
    
    print(f"  mu={target_mu} at heading index {mu_idx}")
    
    # Extract data for target speed and heading
    results = []
    for iom in range(n_omegas):
        record_idx = iom * (n_speeds * n_headings) + target_speed_idx * n_headings + mu_idx
        if record_idx < len(starts):
            omega = starts[record_idx][0]
            mu = starts[record_idx][1]
            # Sanity check
            if abs(mu - target_mu) > 1.0:
                print(f"  WARNING: Expected mu={target_mu} but got mu={mu} at record {record_idx}")
                continue
            
            # Convert omega to wavelength
            wavelength = 2 * np.pi * g / omega**2
            lam_L = wavelength / Lpp
            
            d = totals[record_idx]
            # Normalize: sigma = -fxi / norm
            results.append({
                'omega': omega,
                'lam_L': lam_L,
                'sigma_total': -d['fxi'] / norm,
                'sigma_WL': -d['fxi_WL'] / norm,
                'sigma_vel': -d['fxi_vel'] / norm,
                'sigma_rot': -d['fxi_rot'] / norm,
            })
    
    results.sort(key=lambda x: x['lam_L'])
    return results


# Parse 15% damped debug file
results_15 = parse_debug_components('/home/blofro/src/pdstrip_test/kvlcc2/debug_15pct.out')

# Also parse undamped for comparison
results_0 = parse_debug_components('/home/blofro/src/pdstrip_test/kvlcc2/debug_no_rolldamp.out')

# Print table
print("\n" + "="*110)
print("HEAD SEAS (beta=180°) COMPONENT DECOMPOSITION — 15% roll damping")
print("="*110)
print(f"{'lam/L':>7} | {'sigma_total':>11} | {'sigma_WL':>10} | {'sigma_vel':>10} | {'sigma_rot':>10} | {'Seo SWAN1':>10} | {'ratio':>7}")
print("-"*110)

from scipy.interpolate import interp1d
seo_interp = interp1d([s[0] for s in seo_180], [s[1] for s in seo_180], bounds_error=False, fill_value=np.nan)

for r in results_15:
    seo_val = seo_interp(r['lam_L'])
    ratio = r['sigma_total'] / seo_val if not np.isnan(seo_val) and abs(seo_val) > 0.05 else np.nan
    print(f"{r['lam_L']:7.3f} | {r['sigma_total']:11.3f} | {r['sigma_WL']:10.3f} | {r['sigma_vel']:10.3f} | {r['sigma_rot']:10.3f} | {seo_val:10.3f} | {ratio:7.2f}")


# ============================================================
# Plot: Component decomposition at head seas
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Plot 1: Components for 15% damping ---
ax = axes[0, 0]
if results_15:
    x = [r['lam_L'] for r in results_15]
    ax.plot(x, [r['sigma_total'] for r in results_15], 'k-o', markersize=3, linewidth=2, label='Total')
    ax.plot(x, [r['sigma_WL'] for r in results_15], 'b-s', markersize=2, linewidth=1.5, label='WL (waterline)')
    ax.plot(x, [r['sigma_vel'] for r in results_15], 'r-^', markersize=2, linewidth=1.5, label='Vel (velocity²)')
    ax.plot(x, [r['sigma_rot'] for r in results_15], 'g-v', markersize=2, linewidth=1.5, label='Rot (rotation)')
    
    # Seo reference
    ax.plot([s[0] for s in seo_180], [s[1] for s in seo_180], 'c-o', markersize=4, linewidth=2.5, label='SWAN1 (Seo)', zorder=10)

ax.set_xlabel(r'$\lambda/L$')
ax.set_ylabel(r'$\sigma_{aw}$')
ax.set_title('Components — modified pdstrip (15% damp)')
ax.set_xlim(0, 2.0)
ax.set_ylim(-5, 12)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)

# --- Plot 2: Components for undamped ---
ax = axes[0, 1]
if results_0:
    x = [r['lam_L'] for r in results_0]
    ax.plot(x, [r['sigma_total'] for r in results_0], 'k-o', markersize=3, linewidth=2, label='Total')
    ax.plot(x, [r['sigma_WL'] for r in results_0], 'b-s', markersize=2, linewidth=1.5, label='WL (waterline)')
    ax.plot(x, [r['sigma_vel'] for r in results_0], 'r-^', markersize=2, linewidth=1.5, label='Vel (velocity²)')
    ax.plot(x, [r['sigma_rot'] for r in results_0], 'g-v', markersize=2, linewidth=1.5, label='Rot (rotation)')
    
    ax.plot([s[0] for s in seo_180], [s[1] for s in seo_180], 'c-o', markersize=4, linewidth=2.5, label='SWAN1 (Seo)', zorder=10)

ax.set_xlabel(r'$\lambda/L$')
ax.set_ylabel(r'$\sigma_{aw}$')
ax.set_title('Components — modified pdstrip (undamped)')
ax.set_xlim(0, 2.0)
ax.set_ylim(-5, 12)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)

# --- Plot 3: WL comparison between damped/undamped (should be identical at head seas) ---
ax = axes[1, 0]
if results_15 and results_0:
    x15 = [r['lam_L'] for r in results_15]
    x0 = [r['lam_L'] for r in results_0]
    ax.plot(x15, [r['sigma_WL'] for r in results_15], 'b-s', markersize=3, linewidth=1.5, label='WL (15% damp)')
    ax.plot(x0, [r['sigma_WL'] for r in results_0], 'r--^', markersize=3, linewidth=1.5, label='WL (undamped)')
    ax.plot(x15, [r['sigma_vel'] for r in results_15], 'b-o', markersize=2, linewidth=1, alpha=0.6, label='Vel (15%)')
    ax.plot(x0, [r['sigma_vel'] for r in results_0], 'r--o', markersize=2, linewidth=1, alpha=0.6, label='Vel (undamped)')

ax.set_xlabel(r'$\lambda/L$')
ax.set_ylabel(r'$\sigma_{aw}$')
ax.set_title('WL & Vel: damped vs undamped at head seas')
ax.set_xlim(0, 2.0)
ax.set_ylim(-5, 12)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)

# --- Plot 4: Focus on the hump region ---
ax = axes[1, 1]
if results_15:
    x = [r['lam_L'] for r in results_15]
    total = [r['sigma_total'] for r in results_15]
    wl = [r['sigma_WL'] for r in results_15]
    vel = [r['sigma_vel'] for r in results_15]
    rot = [r['sigma_rot'] for r in results_15]
    
    ax.bar(x, wl, width=0.03, color='blue', alpha=0.6, label='WL')
    # Stack vel on top (note: vel is usually negative -> stacks downward)
    ax.bar(x, vel, width=0.03, color='red', alpha=0.6, label='Vel', bottom=wl)
    ax.plot(x, total, 'ko-', markersize=5, linewidth=2, zorder=10, label='Total')
    ax.plot([s[0] for s in seo_180], [s[1] for s in seo_180], 'c-o', markersize=4, linewidth=2.5, label='SWAN1', zorder=10)

ax.set_xlabel(r'$\lambda/L$')
ax.set_ylabel(r'$\sigma_{aw}$')
ax.set_title('Head seas: WL dominance in hump region')
ax.set_xlim(0.3, 1.6)
ax.set_ylim(-5, 12)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)

plt.suptitle('Head seas ($\\beta=180°$) drift force component decomposition\nV = 3 m/s, KVLCC2',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/headseas_components.png', dpi=150, bbox_inches='tight')
print("\nSaved: headseas_components.png")


# ============================================================
# Detailed comparison table: undamped vs damped at head seas
# ============================================================
print("\n" + "="*130)
print("HEAD SEAS: UNDAMPED vs 15% DAMPED component comparison")
print("="*130)
print(f"{'lam/L':>7} | {'tot_0%':>8} | {'WL_0%':>8} | {'vel_0%':>8} | {'rot_0%':>8} || {'tot_15%':>8} | {'WL_15%':>8} | {'vel_15%':>8} | {'rot_15%':>8} || {'Seo':>5}")
print("-"*130)

for i, r15 in enumerate(results_15):
    lam_L = r15['lam_L']
    seo_val = seo_interp(lam_L)
    
    # Find matching undamped record
    r0 = None
    for rr in results_0:
        if abs(rr['lam_L'] - lam_L) < 0.01:
            r0 = rr
            break
    
    if r0:
        print(f"{lam_L:7.3f} | {r0['sigma_total']:8.3f} | {r0['sigma_WL']:8.3f} | {r0['sigma_vel']:8.3f} | {r0['sigma_rot']:8.3f} || "
              f"{r15['sigma_total']:8.3f} | {r15['sigma_WL']:8.3f} | {r15['sigma_vel']:8.3f} | {r15['sigma_rot']:8.3f} || {seo_val:5.2f}")
