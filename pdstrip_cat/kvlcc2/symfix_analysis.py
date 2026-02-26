#!/usr/bin/env python3
"""
Compare symmetry-fixed pdstrip drift forces against old results and SWAN1 benchmark.
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
norm = rho * g * B**2 / Lpp  # 103064.8 N/m²

# ============================================================
# SWAN1 digitized data from Liu & Papanikolaou Figure 11
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


def parse_debug(fname):
    """Parse debug file to extract drift data."""
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
    
    n = len(starts)
    print(f"  Found {n} DRIFT_START, {len(totals)} DRIFT_TOTAL, "
          f"{len(nopsts)} DRIFT_NOPST, {len(maruos)} DRIFT_MARUO")
    
    assert len(totals) == n, f"totals mismatch: {len(totals)} vs {n}"
    
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
        return None
    
    has_maruo = len(maruos) == len(starts)
    
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
        r = {
            'omega': omega,
            'lam_L': lam_L,
            'sigma_pinkster': -d['fxi'] / norm,
            'sigma_WL': -d['fxi_WL'] / norm,
            'sigma_vel': -d['fxi_vel'] / norm,
            'sigma_rot': -d['fxi_rot'] / norm,
        }
        if has_maruo:
            mar = maruos[record_idx]
            r['sigma_maruo'] = -mar['fxi_maruo'] / norm
        
        results.append(r)
    
    results.sort(key=lambda x: x['lam_L'])
    return results


# ============================================================
# Parse both debug files
# ============================================================
print("="*80)
print("SYMMETRY-FIXED pdstrip vs OLD pdstrip vs SWAN1")
print("="*80)

data_new = parse_debug('/home/blofro/src/pdstrip_test/kvlcc2/debug_symfix.out')
data_old = parse_debug('/home/blofro/src/pdstrip_test/kvlcc2/debug_15pct.out')

# ============================================================
# Head seas comparison table
# ============================================================
print("\n" + "="*120)
print("HEAD SEAS (beta=180) — Symmetry-fixed vs Old vs SWAN1")
print("="*120)

res_new_180 = extract_heading(*data_new, target_mu=180.0)
res_old_180 = extract_heading(*data_old, target_mu=180.0)

seo_interp_180 = interp1d([s[0] for s in seo_data[180]], [s[1] for s in seo_data[180]],
                           bounds_error=False, fill_value=np.nan)

print(f"{'lam/L':>7} | {'New_total':>10} | {'New_WL':>10} | {'New_vel':>10} | "
      f"{'Old_total':>10} | {'Old_WL':>10} | {'SWAN1':>7} | {'New/SWAN':>8} | {'Old/SWAN':>8}")
print("-"*120)

for rn, ro in zip(res_new_180, res_old_180):
    seo_val = seo_interp_180(rn['lam_L'])
    ratio_n = rn['sigma_pinkster'] / seo_val if not np.isnan(seo_val) and abs(seo_val) > 0.05 else np.nan
    ratio_o = ro['sigma_pinkster'] / seo_val if not np.isnan(seo_val) and abs(seo_val) > 0.05 else np.nan
    rn_str = f"{ratio_n:8.2f}" if not np.isnan(ratio_n) else "     ---"
    ro_str = f"{ratio_o:8.2f}" if not np.isnan(ratio_o) else "     ---"
    print(f"{rn['lam_L']:7.3f} | {rn['sigma_pinkster']:10.3f} | {rn['sigma_WL']:10.3f} | "
          f"{rn['sigma_vel']:10.3f} | {ro['sigma_pinkster']:10.3f} | {ro['sigma_WL']:10.3f} | "
          f"{seo_val:7.2f} | {rn_str} | {ro_str}")

# Compute RMS errors
common_lam = np.linspace(0.35, 2.0, 100)
seo_vals = seo_interp_180(common_lam)

new_interp = interp1d([r['lam_L'] for r in res_new_180], [r['sigma_pinkster'] for r in res_new_180],
                       bounds_error=False, fill_value=np.nan)
old_interp = interp1d([r['lam_L'] for r in res_old_180], [r['sigma_pinkster'] for r in res_old_180],
                       bounds_error=False, fill_value=np.nan)

new_vals = new_interp(common_lam)
old_vals = old_interp(common_lam)

mask = ~np.isnan(seo_vals) & ~np.isnan(new_vals) & ~np.isnan(old_vals)
rms_new = np.sqrt(np.mean((new_vals[mask] - seo_vals[mask])**2))
rms_old = np.sqrt(np.mean((old_vals[mask] - seo_vals[mask])**2))
print(f"\nHead seas RMS error: New = {rms_new:.3f}, Old = {rms_old:.3f}, Improvement = {(1-rms_new/rms_old)*100:.1f}%")

# Peak values
peak_new = max(r['sigma_pinkster'] for r in res_new_180)
peak_old = max(r['sigma_pinkster'] for r in res_old_180)
print(f"Peak sigma: New = {peak_new:.3f}, Old = {peak_old:.3f}, SWAN1 = 2.5")


# ============================================================
# 7-panel comparison plot
# ============================================================
seo_betas = [180, 150, 120, 90, 60, 30, 0]
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']
ylims = {180: (-1, 7), 150: (-1, 5), 120: (-2, 5), 90: (-2, 3), 60: (-4, 2), 30: (-4, 2), 0: (-2, 2)}

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes_flat = axes.flatten()

for idx, beta in enumerate(seo_betas):
    ax = axes_flat[idx]
    
    # SWAN1 reference
    sd = seo_data[beta]
    x_sw, y_sw = zip(*sd)
    ax.plot(x_sw, y_sw, 'b-o', markersize=4, linewidth=2, label='SWAN1 (Liu)', zorder=10)
    
    # New (symmetry-fixed) results
    res_n = extract_heading(*data_new, target_mu=float(beta))
    if res_n:
        x_n = [r['lam_L'] for r in res_n]
        y_n = [r['sigma_pinkster'] for r in res_n]
        ax.plot(x_n, y_n, 'g-s', markersize=4, linewidth=2,
                label='Pinkster (sym-fixed)', zorder=8)
    
    # Old (unfixed) results
    res_o = extract_heading(*data_old, target_mu=float(beta))
    if res_o:
        x_o = [r['lam_L'] for r in res_o]
        y_o = [r['sigma_pinkster'] for r in res_o]
        ax.plot(x_o, y_o, 'r--^', markersize=3, linewidth=1, alpha=0.5,
                label='Pinkster (old)', zorder=5)
    
    # Maruo if available
    has_maruo_new = res_n and 'sigma_maruo' in res_n[0]
    if has_maruo_new:
        y_m = [r['sigma_maruo'] for r in res_n]
        ax.plot(x_n, y_m, 'm-d', markersize=3, linewidth=1.5, alpha=0.7,
                label='Maruo/G-B', zorder=7)
    
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

plt.suptitle('KVLCC2 surge drift force: BEM symmetry fix\n'
             'Green = fixed Pinkster, Red dashed = old Pinkster, Blue = SWAN1\n'
             'V=3 m/s, 15% roll damping',
             fontsize=12, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/symfix_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: symfix_comparison.png")


# ============================================================
# Head seas detail: components breakdown
# ============================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Total drift comparison
ax = axes2[0]
sd = seo_data[180]
x_sw, y_sw = zip(*sd)
ax.plot(x_sw, y_sw, 'b-o', markersize=5, linewidth=2, label='SWAN1')

res_n = extract_heading(*data_new, target_mu=180.0)
res_o = extract_heading(*data_old, target_mu=180.0)
x_n = [r['lam_L'] for r in res_n]
x_o = [r['lam_L'] for r in res_o]

ax.plot(x_n, [r['sigma_pinkster'] for r in res_n], 'g-s', markersize=5, linewidth=2, label='Pinkster (sym-fixed)')
ax.plot(x_o, [r['sigma_pinkster'] for r in res_o], 'r--^', markersize=3, linewidth=1, alpha=0.5, label='Pinkster (old)')

has_maruo_new = 'sigma_maruo' in res_n[0]
if has_maruo_new:
    ax.plot(x_n, [r['sigma_maruo'] for r in res_n], 'm-d', markersize=4, linewidth=1.5, alpha=0.7, label='Maruo/G-B')

ax.set_xlabel(r'$\lambda/L$')
ax.set_ylabel(r'$\sigma_{aw}$')
ax.set_title('Head seas (β=180°): Total drift force')
ax.set_xlim(0, 2.5)
ax.set_ylim(-2, 7)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# Panel 2: Component breakdown (new)
ax = axes2[1]
ax.plot(x_n, [r['sigma_WL'] for r in res_n], 'c-o', markersize=4, linewidth=1.5, label='WL (sym-fixed)')
ax.plot(x_n, [r['sigma_vel'] for r in res_n], 'm-s', markersize=4, linewidth=1.5, label='Velocity²')
ax.plot(x_n, [r['sigma_rot'] for r in res_n], 'y-^', markersize=3, linewidth=1, label='Rotation')
ax.plot(x_n, [r['sigma_pinkster'] for r in res_n], 'g-D', markersize=4, linewidth=2, label='Total (sym-fixed)')

# Old WL for comparison
ax.plot(x_o, [r['sigma_WL'] for r in res_o], 'r--', linewidth=1, alpha=0.4, label='WL (old)')

ax.set_xlabel(r'$\lambda/L$')
ax.set_ylabel(r'$\sigma_{aw}$')
ax.set_title('Head seas (β=180°): Component breakdown')
ax.set_xlim(0, 2.5)
ax.set_ylim(-4, 10)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/symfix_headseas_detail.png', dpi=150, bbox_inches='tight')
print("Saved: symfix_headseas_detail.png")

# ============================================================
# Per-heading RMS errors
# ============================================================
print("\n" + "="*80)
print("RMS errors by heading (lam/L range 0.35-2.0)")
print("="*80)
print(f"{'Beta':>6} | {'RMS_new':>8} | {'RMS_old':>8} | {'Improvement':>12} | {'Peak_new':>9} | {'Peak_old':>9} | {'Peak_SWAN':>10}")
print("-"*80)

for beta in seo_betas:
    sd = seo_data[beta]
    seo_interp = interp1d([s[0] for s in sd], [s[1] for s in sd],
                           bounds_error=False, fill_value=np.nan)
    
    res_n = extract_heading(*data_new, target_mu=float(beta))
    res_o = extract_heading(*data_old, target_mu=float(beta))
    if not res_n or not res_o:
        continue
    
    new_interp = interp1d([r['lam_L'] for r in res_n], [r['sigma_pinkster'] for r in res_n],
                           bounds_error=False, fill_value=np.nan)
    old_interp = interp1d([r['lam_L'] for r in res_o], [r['sigma_pinkster'] for r in res_o],
                           bounds_error=False, fill_value=np.nan)
    
    common = np.linspace(0.35, 2.0, 100)
    sv = seo_interp(common)
    nv = new_interp(common)
    ov = old_interp(common)
    
    mask = ~np.isnan(sv) & ~np.isnan(nv) & ~np.isnan(ov)
    if mask.sum() == 0:
        continue
    
    rms_n = np.sqrt(np.mean((nv[mask] - sv[mask])**2))
    rms_o = np.sqrt(np.mean((ov[mask] - sv[mask])**2))
    impr = (1 - rms_n/rms_o)*100 if rms_o > 0 else 0
    
    peak_n = max(r['sigma_pinkster'] for r in res_n)
    peak_o = max(r['sigma_pinkster'] for r in res_o)
    peak_sw = max(s[1] for s in sd)
    
    print(f"{beta:6d} | {rms_n:8.3f} | {rms_o:8.3f} | {impr:11.1f}% | {peak_n:9.3f} | {peak_o:9.3f} | {peak_sw:10.1f}")
