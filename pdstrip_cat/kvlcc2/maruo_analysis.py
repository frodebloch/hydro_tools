#!/usr/bin/env python3
"""
Analyze Maruo/Gerritsma-Beukelman far-field drift force results.
Compare: Pinkster near-field (original) vs Maruo far-field vs SWAN1 benchmark.
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
# SWAN1 digitized data
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
# Parse debug.out for DRIFT_START, DRIFT_TOTAL, DRIFT_NOPST, DRIFT_MARUO
# ============================================================
def parse_debug_all(fname):
    """Parse debug file to extract all drift data."""
    starts = []
    totals = []
    nopsts = []
    maruos = []
    
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
            elif 'DRIFT_MARUO' in line:
                m = re.search(r'fxi_maruo=\s*([-\d.Ee+]+)\s+fxi_maruo_heave=\s*([-\d.Ee+]+)\s+'
                              r'fxi_maruo_sway=\s*([-\d.Ee+]+)', line)
                if m:
                    maruos.append({
                        'fxi_maruo': float(m.group(1)),
                        'fxi_maruo_heave': float(m.group(2)),
                        'fxi_maruo_sway': float(m.group(3)),
                    })
    
    print(f"  Found {len(starts)} DRIFT_START, {len(totals)} DRIFT_TOTAL, "
          f"{len(nopsts)} DRIFT_NOPST, {len(maruos)} DRIFT_MARUO")
    
    n = len(starts)
    assert len(totals) == n, f"totals mismatch: {len(totals)} vs {n}"
    assert len(maruos) == n, f"maruos mismatch: {len(maruos)} vs {n}"
    
    omegas = sorted(set(s[0] for s in starts))
    mus = sorted(set(s[1] for s in starts))
    n_headings = len(mus)
    n_omegas = len(omegas)
    n_speeds = n // (n_omegas * n_headings)
    print(f"  {n_omegas} omegas, {n_headings} headings, {n_speeds} speeds")
    
    return starts, totals, nopsts, maruos, omegas, mus, n_speeds, n_headings, n_omegas


def extract_heading(starts, totals, nopsts, maruos, omegas, mus, 
                    n_speeds, n_headings, n_omegas, target_mu, speed_idx=2):
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
        mar = maruos[record_idx]
        
        results.append({
            'omega': omega,
            'lam_L': lam_L,
            'sigma_pinkster': -d['fxi'] / norm,
            'sigma_WL': -d['fxi_WL'] / norm,
            'sigma_vel': -d['fxi_vel'] / norm,
            'sigma_rot': -d['fxi_rot'] / norm,
            'sigma_maruo': -mar['fxi_maruo'] / norm,
            'sigma_maruo_heave': mar['fxi_maruo_heave'] / norm,  # note: positive = resistance
            'sigma_maruo_sway': mar['fxi_maruo_sway'] / norm,
        })
    
    results.sort(key=lambda x: x['lam_L'])
    return results


# ============================================================
# Parse the debug file
# ============================================================
data = parse_debug_all('/home/blofro/src/pdstrip_test/kvlcc2/debug.out')
starts, totals, nopsts, maruos, omegas, mus, n_speeds, n_headings, n_omegas = data

# ============================================================
# Head seas detailed table
# ============================================================
print("\n" + "="*120)
print("HEAD SEAS (beta=180) — Pinkster vs Maruo/G-B comparison")
print("="*120)
print(f"{'lam/L':>7} | {'Pinkster':>10} | {'Maruo':>10} | {'M_heave':>10} | {'M_sway':>10} | "
      f"{'SWAN1':>7} | {'P/SWAN':>8} | {'M/SWAN':>8}")
print("-"*120)

res_180 = extract_heading(*data, target_mu=180.0)
seo_interp_180 = interp1d([s[0] for s in seo_data[180]], [s[1] for s in seo_data[180]],
                           bounds_error=False, fill_value=np.nan)

for r in res_180:
    seo_val = seo_interp_180(r['lam_L'])
    ratio_p = r['sigma_pinkster'] / seo_val if not np.isnan(seo_val) and abs(seo_val) > 0.05 else np.nan
    ratio_m = r['sigma_maruo'] / seo_val if not np.isnan(seo_val) and abs(seo_val) > 0.05 else np.nan
    rp_str = f"{ratio_p:8.2f}" if not np.isnan(ratio_p) else "     ---"
    rm_str = f"{ratio_m:8.2f}" if not np.isnan(ratio_m) else "     ---"
    print(f"{r['lam_L']:7.3f} | {r['sigma_pinkster']:10.3f} | {r['sigma_maruo']:10.3f} | "
          f"{r['sigma_maruo_heave']:10.3f} | {r['sigma_maruo_sway']:10.3f} | "
          f"{seo_val:7.2f} | {rp_str} | {rm_str}")


# ============================================================
# Figure 1: 7-panel comparison with Pinkster + Maruo + SWAN1
# ============================================================
seo_betas = [180, 150, 120, 90, 60, 30, 0]
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']
ylims = {180: (-1, 7), 150: (-1, 5), 120: (-1, 5), 90: (-2, 3), 60: (-4, 2), 30: (-4, 2), 0: (-2, 2)}

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes_flat = axes.flatten()

for idx, beta in enumerate(seo_betas):
    ax = axes_flat[idx]
    
    # SWAN1 reference
    sd = seo_data[beta]
    x_sw, y_sw = zip(*sd)
    ax.plot(x_sw, y_sw, 'b-o', markersize=4, linewidth=2, label='SWAN1', zorder=10)
    
    # pdstrip results
    res = extract_heading(*data, target_mu=float(beta))
    if res:
        x_pd = [r['lam_L'] for r in res]
        y_pinkster = [r['sigma_pinkster'] for r in res]
        y_maruo = [r['sigma_maruo'] for r in res]
        
        ax.plot(x_pd, y_pinkster, 'r-^', markersize=3, linewidth=1.5, 
                label='Pinkster (near-field)', alpha=0.7, zorder=5)
        ax.plot(x_pd, y_maruo, 'g-s', markersize=4, linewidth=2, 
                label='Maruo/G-B (far-field)', zorder=8)
    
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$\sigma_{aw}$')
    ax.set_title(f'{subplot_labels[idx]} $\\beta = {beta}°$', fontweight='bold')
    ax.set_xlim(0, 2.5)
    ax.set_ylim(ylims.get(beta, (-3, 7)))
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7, loc='upper right')

for idx in range(len(seo_betas), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle('Surge drift force on KVLCC2: Pinkster near-field vs Maruo/G-B far-field vs SWAN1\n'
             'V=3 m/s (6 kn), 15% roll damping',
             fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/maruo_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: maruo_comparison.png")


# ============================================================
# Figure 2: Head seas detailed — Maruo components
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if res_180:
    x = [r['lam_L'] for r in res_180]
    
    # Panel 1: All three: Pinkster vs Maruo vs SWAN1
    ax = axes[0, 0]
    ax.plot(x, [r['sigma_pinkster'] for r in res_180], 'r-^', markersize=3, linewidth=1.5, 
            label='Pinkster (near-field)')
    ax.plot(x, [r['sigma_maruo'] for r in res_180], 'g-s', markersize=4, linewidth=2, 
            label='Maruo/G-B (far-field)')
    ax.plot([s[0] for s in seo_data[180]], [s[1] for s in seo_data[180]], 
            'b-o', markersize=4, linewidth=2, label='SWAN1', zorder=10)
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$\sigma_{aw}$')
    ax.set_title('Head seas: Total drift force comparison')
    ax.set_xlim(0.2, 2.0)
    ax.set_ylim(-1, 7)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Panel 2: Maruo heave vs sway components
    ax = axes[0, 1]
    ax.plot(x, [r['sigma_maruo'] for r in res_180], 'g-s', markersize=4, linewidth=2, label='Maruo total')
    ax.plot(x, [r['sigma_maruo_heave'] for r in res_180], 'm-^', markersize=3, linewidth=1.5, 
            label='Heave contrib')
    ax.plot(x, [r['sigma_maruo_sway'] for r in res_180], 'c-v', markersize=3, linewidth=1.5, 
            label='Sway contrib')
    ax.plot([s[0] for s in seo_data[180]], [s[1] for s in seo_data[180]], 
            'b-o', markersize=4, linewidth=2, label='SWAN1', zorder=10)
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$\sigma_{aw}$')
    ax.set_title('Head seas: Maruo heave/sway decomposition')
    ax.set_xlim(0.2, 2.0)
    ax.set_ylim(-1, 7)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Panel 3: Pinkster WL + vel vs Maruo
    ax = axes[1, 0]
    ax.plot(x, [r['sigma_WL'] for r in res_180], 'r--', linewidth=1, alpha=0.6, label='Pinkster WL')
    ax.plot(x, [r['sigma_vel'] for r in res_180], 'm--', linewidth=1, alpha=0.6, label='Pinkster vel')
    ax.plot(x, [r['sigma_pinkster'] for r in res_180], 'r-^', markersize=3, linewidth=1.5, 
            label='Pinkster total')
    ax.plot(x, [r['sigma_maruo'] for r in res_180], 'g-s', markersize=4, linewidth=2, 
            label='Maruo total')
    ax.plot([s[0] for s in seo_data[180]], [s[1] for s in seo_data[180]], 
            'b-o', markersize=4, linewidth=2, label='SWAN1', zorder=10)
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$\sigma_{aw}$')
    ax.set_title('Component comparison')
    ax.set_xlim(0.2, 2.0)
    ax.set_ylim(-3, 12)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    
    # Panel 4: Ratio to SWAN1
    ax = axes[1, 1]
    seo_interp = interp1d([s[0] for s in seo_data[180]], [s[1] for s in seo_data[180]],
                           bounds_error=False, fill_value=np.nan)
    ratios_p = []
    ratios_m = []
    x_valid = []
    for r in res_180:
        sv = seo_interp(r['lam_L'])
        if not np.isnan(sv) and abs(sv) > 0.1:
            x_valid.append(r['lam_L'])
            ratios_p.append(r['sigma_pinkster'] / sv)
            ratios_m.append(r['sigma_maruo'] / sv)
    ax.plot(x_valid, ratios_p, 'r-^', markersize=3, linewidth=1.5, label='Pinkster/SWAN1')
    ax.plot(x_valid, ratios_m, 'g-s', markersize=4, linewidth=2, label='Maruo/SWAN1')
    ax.axhline(y=1, color='k', linewidth=1, linestyle='--')
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel('Ratio to SWAN1')
    ax.set_title('Head seas: Accuracy ratio')
    ax.set_xlim(0.3, 1.5)
    ax.set_ylim(0, 12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle('Head seas ($\\beta=180°$): Pinkster vs Maruo drift force analysis\n'
             'KVLCC2, V=3 m/s, 15% roll damping',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/maruo_headseas_detail.png', dpi=150, bbox_inches='tight')
print("Saved: maruo_headseas_detail.png")


# ============================================================
# Summary statistics
# ============================================================
print("\n" + "="*80)
print("SUMMARY: Maruo vs Pinkster at head seas (beta=180)")
print("="*80)

hump = [r for r in res_180 if 0.5 <= r['lam_L'] <= 1.3]
if hump:
    peak_pink = max(r['sigma_pinkster'] for r in hump)
    peak_maruo = max(r['sigma_maruo'] for r in hump)
    peak_swan = 2.5
    
    print(f"Peak sigma (Pinkster):     {peak_pink:.2f}  ({peak_pink/peak_swan:.1f}x SWAN1)")
    print(f"Peak sigma (Maruo):        {peak_maruo:.2f}  ({peak_maruo/peak_swan:.1f}x SWAN1)")
    print(f"Peak SWAN1:                {peak_swan:.2f}")
    
    seo_vals = [seo_interp_180(r['lam_L']) for r in hump]
    pink_vals = [r['sigma_pinkster'] for r in hump]
    maruo_vals = [r['sigma_maruo'] for r in hump]
    
    rms_pink = np.sqrt(np.nanmean([(p - s)**2 for p, s in zip(pink_vals, seo_vals) if not np.isnan(s)]))
    rms_maruo = np.sqrt(np.nanmean([(m - s)**2 for m, s in zip(maruo_vals, seo_vals) if not np.isnan(s)]))
    print(f"\nRMS error vs SWAN1 (0.5<lam/L<1.3):")
    print(f"  Pinkster:  {rms_pink:.2f}")
    print(f"  Maruo:     {rms_maruo:.2f}")
    print(f"  Improvement: {(1 - rms_maruo/rms_pink)*100:.0f}%")

# All headings summary
print("\n" + "="*80)
print("PEAK VALUES across all headings")
print("="*80)
for beta in seo_betas:
    res = extract_heading(*data, target_mu=float(beta))
    if not res:
        continue
    seo = seo_data[beta]
    seo_interp = interp1d([s[0] for s in seo], [s[1] for s in seo],
                           bounds_error=False, fill_value=np.nan)
    
    hump = [r for r in res if 0.3 <= r['lam_L'] <= 2.0]
    if beta >= 90:
        peak_p = max(r['sigma_pinkster'] for r in hump) if hump else 0
        peak_m = max(r['sigma_maruo'] for r in hump) if hump else 0
        peak_s = max(s[1] for s in seo if 0.3 <= s[0] <= 2.0) if seo else 0
    else:
        peak_p = min(r['sigma_pinkster'] for r in hump) if hump else 0
        peak_m = min(r['sigma_maruo'] for r in hump) if hump else 0
        peak_s = min(s[1] for s in seo if 0.3 <= s[0] <= 2.0) if seo else 0
    
    print(f"  beta={beta:3d}: Pinkster={peak_p:7.2f}  Maruo={peak_m:7.2f}  SWAN1={peak_s:7.2f}")
