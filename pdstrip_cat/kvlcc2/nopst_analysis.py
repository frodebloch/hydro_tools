#!/usr/bin/env python3
"""
Analyze the pst-excluded drift force results from the new debug.out.
Compare WL_nopst vs WL_original vs SWAN1 at all 7 benchmark headings.
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# KVLCC2 parameters
Lpp = 328.2
B = 58.0
rho = 1025.0
g = 9.81
norm = rho * g * B**2 / Lpp
target_speed_idx = 2  # V=3 m/s

# ============================================================
# SWAN1 digitized data (from fig11_comparison.py)
# ============================================================
seo_data = {}
seo_data[180] = [
    (0.30, 0.0), (0.35, 0.1), (0.40, 0.3), (0.45, 0.6),
    (0.50, 1.0), (0.55, 1.3), (0.60, 1.5), (0.65, 1.3),
    (0.70, 0.9), (0.75, 0.5), (0.80, 0.3), (0.85, 0.5),
    (0.90, 1.0), (0.95, 1.6), (1.00, 2.1), (1.05, 2.4),
    (1.10, 2.5), (1.15, 2.3), (1.20, 1.8), (1.25, 1.3),
    (1.30, 0.8), (1.35, 0.5), (1.40, 0.3), (1.50, 0.1),
    (1.60, 0.0), (1.80, 0.0), (2.00, 0.0), (2.50, 0.0),
]
seo_data[150] = [
    (0.30, 0.0), (0.35, 0.3), (0.40, 0.8), (0.45, 1.5),
    (0.50, 2.0), (0.55, 1.7), (0.60, 1.0), (0.65, 0.5),
    (0.70, 0.3), (0.75, 0.3), (0.80, 0.5), (0.85, 0.8),
    (0.90, 1.1), (0.95, 1.3), (1.00, 1.5), (1.05, 1.4),
    (1.10, 1.2), (1.15, 0.9), (1.20, 0.6), (1.25, 0.4),
    (1.30, 0.3), (1.40, 0.3), (1.50, 0.5), (1.55, 0.5),
    (1.60, 0.4), (1.70, 0.2), (1.80, 0.1), (2.00, 0.0),
    (2.50, 0.0),
]
seo_data[120] = [
    (0.30, 0.0), (0.35, 0.2), (0.40, 0.6), (0.45, 1.1),
    (0.50, 1.5), (0.55, 1.2), (0.60, 0.5), (0.65, 0.1),
    (0.70, -0.2), (0.75, -0.1), (0.80, 0.1), (0.85, 0.3),
    (0.90, 0.4), (0.95, 0.5), (1.00, 0.5), (1.05, 0.4),
    (1.10, 0.2), (1.15, 0.1), (1.20, 0.0), (1.30, 0.1),
    (1.40, 0.3), (1.50, 0.5), (1.55, 0.4), (1.60, 0.3),
    (1.70, 0.1), (1.80, 0.0), (2.00, 0.0), (2.50, 0.0),
]
seo_data[90] = [
    (0.30, 0.0), (0.35, 0.3), (0.40, 0.8), (0.45, 1.5),
    (0.50, 2.0), (0.55, 1.6), (0.60, 0.8), (0.65, 0.2),
    (0.70, -0.1), (0.75, -0.3), (0.80, -0.3), (0.85, -0.2),
    (0.90, -0.1), (0.95, 0.0), (1.00, 0.0), (1.10, 0.0),
    (1.20, 0.0), (1.30, 0.1), (1.40, 0.2), (1.50, 0.3),
    (1.55, 0.2), (1.60, 0.1), (1.70, 0.0), (1.80, 0.0),
    (2.00, 0.0), (2.50, 0.0),
]
seo_data[60] = [
    (0.30, 0.0), (0.35, 0.05), (0.40, 0.1), (0.45, 0.1),
    (0.50, 0.0), (0.55, -0.2), (0.60, -0.3), (0.65, -0.5),
    (0.70, -0.5), (0.75, -0.4), (0.80, -0.2), (0.85, -0.1),
    (0.90, 0.0), (0.95, 0.0), (1.00, 0.0), (1.10, -0.1),
    (1.20, -0.3), (1.30, -0.8), (1.40, -1.5), (1.50, -2.2),
    (1.60, -2.5), (1.70, -2.3), (1.80, -1.8), (1.90, -1.2),
    (2.00, -0.7), (2.20, -0.2), (2.50, 0.0),
]
seo_data[30] = [
    (0.30, 0.0), (0.35, 0.0), (0.40, 0.0), (0.50, -0.1),
    (0.55, -0.1), (0.60, -0.2), (0.65, -0.2), (0.70, -0.2),
    (0.75, -0.1), (0.80, -0.1), (0.90, 0.0), (1.00, 0.0),
    (1.10, -0.1), (1.20, -0.3), (1.30, -0.7), (1.40, -1.2),
    (1.50, -1.7), (1.60, -2.0), (1.70, -2.0), (1.80, -1.7),
    (1.90, -1.3), (2.00, -0.8), (2.20, -0.3), (2.50, 0.0),
]
seo_data[0] = [
    (0.30, 0.0), (0.40, 0.0), (0.50, -0.1), (0.60, -0.2),
    (0.70, -0.3), (0.80, -0.2), (0.90, -0.1), (1.00, -0.1),
    (1.10, -0.1), (1.20, -0.1), (1.30, -0.1), (1.50, -0.1),
    (1.80, 0.0), (2.00, 0.0), (2.50, 0.0),
]


# ============================================================
# Parse debug.out for DRIFT_START, DRIFT_TOTAL, and DRIFT_NOPST
# ============================================================
def parse_debug_with_nopst(fname):
    """Parse debug file to extract TOTAL and NOPST data for all conditions."""
    starts = []
    totals = []
    nopsts = []
    
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
            elif 'DRIFT_NOPST' in line:
                m = re.search(r'fxi_WL_nopst=\s*([-\d.Ee+]+)\s+feta_WL_nopst=\s*([-\d.Ee+]+)', line)
                if m:
                    nopsts.append({
                        'fxi_WL_nopst': float(m.group(1)),
                        'feta_WL_nopst': float(m.group(2)),
                    })
    
    print(f"  Found {len(starts)} DRIFT_START, {len(totals)} DRIFT_TOTAL, {len(nopsts)} DRIFT_NOPST")
    assert len(starts) == len(totals) == len(nopsts), \
        f"Mismatch: starts={len(starts)}, totals={len(totals)}, nopsts={len(nopsts)}"
    
    # Get unique values
    omegas = sorted(set(s[0] for s in starts))
    mus = sorted(set(s[1] for s in starts))
    n_headings = len(mus)
    n_omegas = len(omegas)
    n_speeds = len(starts) // (n_omegas * n_headings)
    print(f"  {n_omegas} omegas, {n_headings} headings, {n_speeds} speeds")
    print(f"  Headings: {mus}")
    
    return starts, totals, nopsts, omegas, mus, n_speeds, n_headings, n_omegas


def extract_heading(starts, totals, nopsts, omegas, mus, n_speeds, n_headings, n_omegas,
                    target_mu, speed_idx=2):
    """Extract results for a specific heading and speed."""
    mu_idx = None
    for i, m in enumerate(mus):
        if abs(m - target_mu) < 1.0:
            mu_idx = i
            break
    if mu_idx is None:
        print(f"  WARNING: mu={target_mu} not found!")
        return None
    
    results = []
    for iom in range(n_omegas):
        record_idx = iom * (n_speeds * n_headings) + speed_idx * n_headings + mu_idx
        if record_idx >= len(starts):
            continue
        
        omega = starts[record_idx][0]
        mu = starts[record_idx][1]
        if abs(mu - target_mu) > 1.0:
            continue
        
        wavelength = 2 * np.pi * g / omega**2
        lam_L = wavelength / Lpp
        
        d = totals[record_idx]
        n = nopsts[record_idx]
        
        # Total with pst: fxi = fxi_WL + fxi_vel + fxi_rot
        # Total without pst: fxi_nopst = fxi_WL_nopst + fxi_vel + fxi_rot
        fxi_nopst = n['fxi_WL_nopst'] + d['fxi_vel'] + d['fxi_rot']
        
        results.append({
            'omega': omega,
            'lam_L': lam_L,
            'sigma_total': -d['fxi'] / norm,
            'sigma_WL': -d['fxi_WL'] / norm,
            'sigma_vel': -d['fxi_vel'] / norm,
            'sigma_rot': -d['fxi_rot'] / norm,
            'sigma_WL_nopst': -n['fxi_WL_nopst'] / norm,
            'sigma_total_nopst': -fxi_nopst / norm,
        })
    
    results.sort(key=lambda x: x['lam_L'])
    return results


# ============================================================
# Parse the new debug file
# ============================================================
data = parse_debug_with_nopst('/home/blofro/src/pdstrip_test/kvlcc2/debug.out')
starts, totals, nopsts, omegas, mus, n_speeds, n_headings, n_omegas = data

# ============================================================
# Head seas detailed table
# ============================================================
print("\n" + "="*130)
print("HEAD SEAS (beta=180) — pst vs no-pst comparison")
print("="*130)
print(f"{'lam/L':>7} | {'sigma_total':>11} | {'sigma_WL':>10} | {'sigma_vel':>10} | "
      f"{'sigma_WL_np':>11} | {'sigma_tot_np':>12} | {'SWAN1':>7} | {'ratio_orig':>10} | {'ratio_nopst':>11}")
print("-"*130)

res_180 = extract_heading(*data, target_mu=180.0)
seo_interp_180 = interp1d([s[0] for s in seo_data[180]], [s[1] for s in seo_data[180]],
                           bounds_error=False, fill_value=np.nan)

for r in res_180:
    seo_val = seo_interp_180(r['lam_L'])
    ratio_orig = r['sigma_total'] / seo_val if not np.isnan(seo_val) and abs(seo_val) > 0.05 else np.nan
    ratio_nopst = r['sigma_total_nopst'] / seo_val if not np.isnan(seo_val) and abs(seo_val) > 0.05 else np.nan
    print(f"{r['lam_L']:7.3f} | {r['sigma_total']:11.3f} | {r['sigma_WL']:10.3f} | {r['sigma_vel']:10.3f} | "
          f"{r['sigma_WL_nopst']:11.3f} | {r['sigma_total_nopst']:12.3f} | {seo_val:7.2f} | "
          f"{'---' if np.isnan(ratio_orig) else f'{ratio_orig:10.2f}'} | "
          f"{'---' if np.isnan(ratio_nopst) else f'{ratio_nopst:11.2f}'}")


# ============================================================
# Figure 1: 7-panel comparison (like fig11) with original + nopst + SWAN1
# ============================================================
seo_betas = [180, 150, 120, 90, 60, 30, 0]
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']
ylims = {180: (-2, 8), 150: (-2, 8), 120: (-2, 8), 90: (-3, 5), 60: (-4, 3), 30: (-4, 3), 0: (-3, 2)}

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes_flat = axes.flatten()

for idx, beta in enumerate(seo_betas):
    ax = axes_flat[idx]
    
    # SWAN1 reference
    sd = seo_data[beta]
    x_sw, y_sw = zip(*sd)
    ax.plot(x_sw, y_sw, 'b-o', markersize=4, linewidth=2, label='SWAN1', zorder=10)
    
    # pdstrip original
    res = extract_heading(*data, target_mu=float(beta))
    if res:
        x_pd = [r['lam_L'] for r in res]
        y_orig = [r['sigma_total'] for r in res]
        y_nopst = [r['sigma_total_nopst'] for r in res]
        y_wl = [r['sigma_WL'] for r in res]
        y_wl_np = [r['sigma_WL_nopst'] for r in res]
        
        ax.plot(x_pd, y_orig, 'r-^', markersize=3, linewidth=1.5, label='pdstrip (with pst)', zorder=5)
        ax.plot(x_pd, y_nopst, 'g-s', markersize=3, linewidth=1.5, label='pdstrip (no pst)', zorder=6)
    
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$\sigma_{aw}$')
    ax.set_title(f'{subplot_labels[idx]} $\\beta = {beta}°$', fontweight='bold')
    ax.set_xlim(0, 2.5)
    ax.set_ylim(ylims.get(beta, (-3, 8)))
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7, loc='upper right')

for idx in range(len(seo_betas), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle('Effect of removing pst (hydrostatic restoring) from WL pressure\n'
             'KVLCC2, V=3 m/s, pdstrip near-field Pinkster drift force',
             fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/nopst_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: nopst_comparison.png")


# ============================================================
# Figure 2: Head seas detailed decomposition (with vs without pst)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if res_180:
    x = [r['lam_L'] for r in res_180]
    
    # Panel 1: Total: original vs nopst vs SWAN1
    ax = axes[0, 0]
    ax.plot(x, [r['sigma_total'] for r in res_180], 'r-^', markersize=3, linewidth=1.5, label='Total (with pst)')
    ax.plot(x, [r['sigma_total_nopst'] for r in res_180], 'g-s', markersize=3, linewidth=1.5, label='Total (no pst)')
    ax.plot([s[0] for s in seo_data[180]], [s[1] for s in seo_data[180]], 'b-o', markersize=4, linewidth=2, label='SWAN1', zorder=10)
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$\sigma_{aw}$')
    ax.set_title('Head seas: Total drift force')
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-2, 10)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Panel 2: WL component only: with vs without pst
    ax = axes[0, 1]
    ax.plot(x, [r['sigma_WL'] for r in res_180], 'r-^', markersize=3, linewidth=1.5, label='WL (with pst)')
    ax.plot(x, [r['sigma_WL_nopst'] for r in res_180], 'g-s', markersize=3, linewidth=1.5, label='WL (no pst)')
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$\sigma_{aw}$')
    ax.set_title('Head seas: WL component with/without pst')
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-5, 15)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Panel 3: All components (no-pst version)
    ax = axes[1, 0]
    ax.plot(x, [r['sigma_total_nopst'] for r in res_180], 'k-o', markersize=3, linewidth=2, label='Total (no pst)')
    ax.plot(x, [r['sigma_WL_nopst'] for r in res_180], 'g-s', markersize=2, linewidth=1.5, label='WL (no pst)')
    ax.plot(x, [r['sigma_vel'] for r in res_180], 'r-^', markersize=2, linewidth=1.5, label='Vel')
    ax.plot(x, [r['sigma_rot'] for r in res_180], 'm-v', markersize=2, linewidth=1.5, label='Rot')
    ax.plot([s[0] for s in seo_data[180]], [s[1] for s in seo_data[180]], 'b-o', markersize=4, linewidth=2, label='SWAN1', zorder=10)
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$\sigma_{aw}$')
    ax.set_title('Head seas: No-pst component decomposition')
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-5, 10)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    
    # Panel 4: Difference (pst contribution to WL)
    ax = axes[1, 1]
    pst_contrib = [r['sigma_WL'] - r['sigma_WL_nopst'] for r in res_180]
    ax.plot(x, pst_contrib, 'purple', linewidth=2, label='pst contribution to WL')
    ax.fill_between(x, 0, pst_contrib, alpha=0.3, color='purple')
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$\Delta\sigma_{aw}$')
    ax.set_title('pst contribution = WL(with pst) - WL(no pst)')
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-2, 10)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle('Head seas ($\\beta=180°$): Effect of pst removal on drift force\n'
             'KVLCC2, V=3 m/s',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/nopst_headseas_detail.png', dpi=150, bbox_inches='tight')
print("Saved: nopst_headseas_detail.png")


# ============================================================
# Summary statistics
# ============================================================
print("\n" + "="*80)
print("SUMMARY: Effect of removing pst at head seas (beta=180)")
print("="*80)

# Focus on the resonant hump region (0.7 < lam/L < 1.3)
hump = [r for r in res_180 if 0.7 <= r['lam_L'] <= 1.3]
if hump:
    peak_orig = max(r['sigma_total'] for r in hump)
    peak_nopst = max(r['sigma_total_nopst'] for r in hump)
    peak_swan = 2.5  # from digitized data
    
    print(f"Peak sigma_total (with pst):    {peak_orig:.2f}")
    print(f"Peak sigma_total (no pst):      {peak_nopst:.2f}")
    print(f"Peak SWAN1:                     {peak_swan:.2f}")
    print(f"Reduction from removing pst:    {peak_orig - peak_nopst:.2f} ({(peak_orig - peak_nopst)/peak_orig*100:.0f}%)")
    print(f"Remaining overprediction:       {peak_nopst/peak_swan:.2f}x SWAN1")
    
    # RMS comparison across hump region
    seo_vals = [seo_interp_180(r['lam_L']) for r in hump]
    orig_vals = [r['sigma_total'] for r in hump]
    nopst_vals = [r['sigma_total_nopst'] for r in hump]
    
    rms_orig = np.sqrt(np.nanmean([(o - s)**2 for o, s in zip(orig_vals, seo_vals) if not np.isnan(s)]))
    rms_nopst = np.sqrt(np.nanmean([(n - s)**2 for n, s in zip(nopst_vals, seo_vals) if not np.isnan(s)]))
    print(f"\nRMS error vs SWAN1 (0.7<lam/L<1.3):")
    print(f"  With pst:    {rms_orig:.2f}")
    print(f"  Without pst: {rms_nopst:.2f}")
    print(f"  Improvement: {(1 - rms_nopst/rms_orig)*100:.0f}%")
