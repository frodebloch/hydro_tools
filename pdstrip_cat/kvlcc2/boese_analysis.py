#!/usr/bin/env python3
"""
Boese (1970) drift force analysis for KVLCC2.
Parse DRIFT_BOESE from debug.out and compare against SWAN1 benchmark at all 7 headings.
Also includes Pinkster and Maruo for comparison.
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
norm = rho * g * B**2 / Lpp  # normalization: sigma = -fxi / norm

# ============================================================
# SWAN1 reference data (digitized from Liu & Papanikolaou 2021, Fig 11)
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
# Parse debug.out
# ============================================================
def parse_debug(fname):
    """Parse all drift force data from debug.out."""
    starts = []
    totals = []
    maruos = []
    boeses = []
    
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
                    totals.append({k: float(v) for k, v in zip(
                        ['fxi','feta','fxi_WL','feta_WL','fxi_vel','fxi_rot'],
                        [m.group(i) for i in range(1,7)])})
            elif 'DRIFT_MARUO' in line:
                m = re.search(r'fxi_maruo=\s*([-\d.Ee+]+)\s+fxi_maruo_heave=\s*([-\d.Ee+]+)\s+'
                              r'fxi_maruo_sway=\s*([-\d.Ee+]+)', line)
                if m:
                    maruos.append({'fxi_maruo': float(m.group(1)),
                                   'fxi_maruo_heave': float(m.group(2)),
                                   'fxi_maruo_sway': float(m.group(3))})
            elif 'DRIFT_BOESE' in line:
                m = re.search(r'fxi_boese=\s*([-\d.Ee+]+)\s+fxi_boese_wp=\s*([-\d.Ee+]+)\s+'
                              r'fxi_maruo=\s*([-\d.Ee+]+)', line)
                if m:
                    boeses.append({'fxi_boese': float(m.group(1)),
                                   'fxi_boese_wp': float(m.group(2)),
                                   'fxi_maruo_boese': float(m.group(3))})
    
    n = len(starts)
    omegas = sorted(set(s[0] for s in starts), reverse=True)  # high to low (as in file)
    mus = sorted(set(s[1] for s in starts))
    n_h = len(mus)
    n_o = len(omegas)
    n_s = n // (n_o * n_h)
    print(f"  {n} records: {n_o} omega x {n_h} headings x {n_s} speeds")
    print(f"  TOTAL: {len(totals)}, MARUO: {len(maruos)}, BOESE: {len(boeses)}")
    
    return starts, totals, maruos, boeses, omegas, mus, n_s, n_h, n_o

data = parse_debug('/home/blofro/src/pdstrip_test/kvlcc2/debug.out')
starts, totals, maruos, boeses, omegas, mus, n_s, n_h, n_o = data

# ============================================================
# Extract data for speed index 2 (V=3 m/s) at all headings
# ============================================================
# File ordering: for each omega (high to low), for each heading (mu sorted), for each speed (0-7)
speed_idx = 2  # 0-indexed; speed=3.0 m/s is index 2

def get_data_at_heading(target_mu, speed_idx):
    """Extract drift force data at a specific heading and speed."""
    mu_idx = None
    for i, m in enumerate(mus):
        if abs(m - target_mu) < 1.0:
            mu_idx = i
            break
    if mu_idx is None:
        return []
    
    results = []
    for iom in range(n_o):
        idx = iom * (n_s * n_h) + speed_idx * n_h + mu_idx
        if idx >= len(starts):
            continue
        omega = starts[idx][0]
        mu = starts[idx][1]
        if abs(mu - target_mu) > 1.0:
            continue
        
        lam_L = 2*np.pi*g/omega**2 / Lpp
        
        r = {'omega': omega, 'lam_L': lam_L}
        
        if idx < len(totals):
            r['fxi'] = totals[idx]['fxi']
            r['sigma_pinkster'] = -totals[idx]['fxi'] / norm
        if idx < len(maruos):
            r['fxi_maruo'] = maruos[idx]['fxi_maruo']
            r['sigma_maruo'] = -maruos[idx]['fxi_maruo'] / norm
        if idx < len(boeses):
            r['fxi_boese'] = boeses[idx]['fxi_boese']
            r['fxi_boese_wp'] = boeses[idx]['fxi_boese_wp']
            r['sigma_boese'] = -boeses[idx]['fxi_boese'] / norm
            r['sigma_boese_wp'] = -boeses[idx]['fxi_boese_wp'] / norm
        
        results.append(r)
    
    results.sort(key=lambda x: x['lam_L'])
    return results


# ============================================================
# Print table for head seas
# ============================================================
target_betas = [180, 150, 120, 90, 60, 30, 0]

print("\n" + "="*120)
print("HEAD SEAS (beta=180°) — Boese vs Pinkster vs Maruo vs SWAN1")
print("="*120)

res_180 = get_data_at_heading(180.0, speed_idx)
seo_i_180 = interp1d([s[0] for s in seo_data[180]], [s[1] for s in seo_data[180]],
                       bounds_error=False, fill_value=np.nan)

print(f"{'lam/L':>7} | {'Pinkster':>9} | {'Maruo':>9} | {'Boese':>9} | {'Boese_wp':>9} | {'SWAN1':>9}")
print("-"*70)
for r in res_180:
    s = seo_i_180(r['lam_L'])
    sp = r.get('sigma_pinkster', np.nan)
    sm = r.get('sigma_maruo', np.nan)
    sb = r.get('sigma_boese', np.nan)
    swp = r.get('sigma_boese_wp', np.nan)
    s_str = f"{s:9.3f}" if not np.isnan(s) else "      ---"
    print(f"{r['lam_L']:7.3f} | {sp:9.3f} | {sm:9.3f} | {sb:9.3f} | {swp:9.3f} | {s_str}")


# ============================================================
# RMS errors at all headings
# ============================================================
print("\n" + "="*80)
print("RMS ERRORS: sigma_aw vs SWAN1 at each heading")
print("="*80)
print(f"{'beta':>5} | {'Pinkster':>10} | {'Maruo':>10} | {'Boese':>10} | {'Best':>10}")
print("-"*55)

for beta in target_betas:
    # Map beta to mu in pdstrip: pdstrip mu matches figure beta
    results_h = get_data_at_heading(float(beta), speed_idx)
    if not results_h:
        print(f"{beta:5d} | {'no data':>10} |")
        continue
    
    sd = seo_data[beta]
    seo_i = interp1d([s[0] for s in sd], [s[1] for s in sd],
                      bounds_error=False, fill_value=np.nan)
    
    common = np.linspace(0.35, 2.0, 100)
    seo_v = seo_i(common)
    
    rms = {}
    for label, key in [('Pinkster', 'sigma_pinkster'), ('Maruo', 'sigma_maruo'), ('Boese', 'sigma_boese')]:
        vals = [r.get(key, np.nan) for r in results_h]
        lams = [r['lam_L'] for r in results_h]
        valid = [(l, v) for l, v in zip(lams, vals) if not np.isnan(v)]
        if len(valid) < 3:
            rms[label] = np.nan
            continue
        interp_f = interp1d([v[0] for v in valid], [v[1] for v in valid],
                            bounds_error=False, fill_value=np.nan)
        pred = interp_f(common)
        mask = ~np.isnan(seo_v) & ~np.isnan(pred)
        if mask.sum() > 0:
            rms[label] = np.sqrt(np.mean((pred[mask] - seo_v[mask])**2))
        else:
            rms[label] = np.nan
    
    best = min(rms, key=lambda k: rms[k] if not np.isnan(rms[k]) else 999)
    print(f"{beta:5d} | {rms['Pinkster']:10.3f} | {rms['Maruo']:10.3f} | {rms['Boese']:10.3f} | {best:>10}")


# ============================================================
# 7-panel comparison plot
# ============================================================
seo_betas = [180, 150, 120, 90, 60, 30, 0]
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']
ylims = {180: (-1, 7), 150: (-1, 5), 120: (-2, 5), 90: (-2, 3), 60: (-3, 2), 30: (-3, 2), 0: (-3, 2)}

fig, axes = plt.subplots(3, 3, figsize=(15, 13))
axes_flat = axes.flatten()

for idx, beta in enumerate(seo_betas):
    ax = axes_flat[idx]
    sd = seo_data[beta]
    
    # SWAN1
    x_s, y_s = zip(*sd)
    ax.plot(x_s, y_s, 'b-o', ms=4, lw=1.5, label='SWAN1', zorder=10)
    
    # pdstrip data
    results_h = get_data_at_heading(float(beta), speed_idx)
    
    if results_h:
        lam = [r['lam_L'] for r in results_h]
        
        # Pinkster
        sp = [r.get('sigma_pinkster', np.nan) for r in results_h]
        ax.plot(lam, sp, 'r-^', ms=3, lw=1, alpha=0.5, label='Pinkster', zorder=5)
        
        # Maruo
        sm = [r.get('sigma_maruo', np.nan) for r in results_h]
        ax.plot(lam, sm, 'm-d', ms=3, lw=1, alpha=0.5, label='Maruo/G-B', zorder=6)
        
        # Boese (thick line, main result)
        sb = [r.get('sigma_boese', np.nan) for r in results_h]
        ax.plot(lam, sb, 'g-s', ms=4, lw=2, label='Boese', zorder=8)
    
    ax.set_xlabel(r'$\lambda/L$', fontsize=10)
    ax.set_ylabel(r'$\sigma_{aw}$', fontsize=10)
    ax.set_title(f'{subplot_labels[idx]} $\\beta = {beta}°$', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 2.5)
    if beta in ylims:
        ax.set_ylim(ylims[beta])
    ax.axhline(0, color='k', lw=0.5)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7, loc='upper right')

# Hide unused subplots
for idx in range(len(seo_betas), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle('KVLCC2 Surge Drift Force: Boese (1970) vs SWAN1\n'
             'V=3 m/s, speed index 2', fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/boese_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: boese_comparison.png")


# ============================================================
# Focused head seas plot: Boese decomposition
# ============================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

res = get_data_at_heading(180.0, speed_idx)
lam = [r['lam_L'] for r in res]

# Panel 1: Total comparison
sd = seo_data[180]
ax1.plot([s[0] for s in sd], [s[1] for s in sd], 'b-o', ms=4, lw=2, label='SWAN1', zorder=10)
ax1.plot(lam, [r.get('sigma_pinkster', np.nan) for r in res], 'r-^', ms=3, lw=1, alpha=0.6, label='Pinkster')
ax1.plot(lam, [r.get('sigma_maruo', np.nan) for r in res], 'm-d', ms=3, lw=1, alpha=0.6, label='Maruo/G-B')
ax1.plot(lam, [r.get('sigma_boese', np.nan) for r in res], 'g-s', ms=4, lw=2, label='Boese total')
ax1.set_xlabel(r'$\lambda/L$')
ax1.set_ylabel(r'$\sigma_{aw}$')
ax1.set_title('Head seas: Total surge drift force')
ax1.set_xlim(0, 2.5); ax1.set_ylim(-2, 7)
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3); ax1.axhline(0, color='k', lw=0.5)

# Panel 2: Boese decomposition
ax2.plot(lam, [r.get('sigma_boese', np.nan) for r in res], 'g-s', ms=4, lw=2, label='Boese total')
ax2.plot(lam, [r.get('sigma_boese_wp', np.nan) for r in res], 'c-v', ms=3, lw=1.5, label='Boese WP integral')
ax2.plot(lam, [r.get('sigma_maruo', np.nan) for r in res], 'm-d', ms=3, lw=1.5, label='G-B (damping)')
sd = seo_data[180]
ax2.plot([s[0] for s in sd], [s[1] for s in sd], 'b-o', ms=4, lw=2, label='SWAN1', zorder=10)
ax2.set_xlabel(r'$\lambda/L$')
ax2.set_ylabel(r'$\sigma_{aw}$')
ax2.set_title('Head seas: Boese decomposition')
ax2.set_xlim(0, 2.5); ax2.set_ylim(-2, 4)
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3); ax2.axhline(0, color='k', lw=0.5)

plt.suptitle('KVLCC2 Head Seas: Boese (1970) Drift Force Decomposition\nV=3 m/s', fontsize=12, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/boese_headseas.png', dpi=150, bbox_inches='tight')
print("Saved: boese_headseas.png")
