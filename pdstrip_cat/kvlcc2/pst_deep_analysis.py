#!/usr/bin/env python3
"""
Deep analysis of WL integral with and without pst.
Focus: understand what drives overprediction and test remedies.
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

# SWAN1 data
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

def parse_debug(fname):
    starts, totals, nopsts, maruos = [], [], [], []
    print(f"Parsing {fname}...")
    with open(fname, 'r') as f:
        for line in f:
            if 'DRIFT_START' in line:
                m = re.search(r'omega=\s*([\d.]+)\s+mu=\s*([-\d.]+)', line)
                if m: starts.append((float(m.group(1)), float(m.group(2))))
            elif 'DRIFT_TOTAL' in line:
                m = re.search(r'fxi=\s*([-\d.Ee+]+)\s+feta=\s*([-\d.Ee+]+)\s+'
                              r'fxi_WL=\s*([-\d.Ee+]+)\s+feta_WL=\s*([-\d.Ee+]+)\s+'
                              r'fxi_vel=\s*([-\d.Ee+]+)\s+fxi_rot=\s*([-\d.Ee+]+)', line)
                if m:
                    totals.append({k: float(v) for k, v in zip(
                        ['fxi','feta','fxi_WL','feta_WL','fxi_vel','fxi_rot'],
                        [m.group(i) for i in range(1,7)])})
            elif 'DRIFT_NOPST' in line:
                m = re.search(r'fxi_WL_nopst=\s*([-\d.Ee+]+)\s+feta_WL_nopst=\s*([-\d.Ee+]+)\s+fxi_rot_nopst=\s*([-\d.Ee+]+)', line)
                if not m:
                    m = re.search(r'fxi_WL_nopst=\s*([-\d.Ee+]+)\s+feta_WL_nopst=\s*([-\d.Ee+]+)', line)
                if m:
                    d = {'fxi_WL_nopst': float(m.group(1)), 'feta_WL_nopst': float(m.group(2))}
                    if m.lastindex >= 3:
                        d['fxi_rot_nopst'] = float(m.group(3))
                    nopsts.append(d)
            elif 'DRIFT_MARUO' in line:
                m = re.search(r'fxi_maruo=\s*([-\d.Ee+]+)\s+fxi_maruo_heave=\s*([-\d.Ee+]+)\s+'
                              r'fxi_maruo_sway=\s*([-\d.Ee+]+)', line)
                if m:
                    maruos.append({'fxi_maruo': float(m.group(1)),
                                   'fxi_maruo_heave': float(m.group(2)),
                                   'fxi_maruo_sway': float(m.group(3))})
    
    n = len(starts)
    omegas = sorted(set(s[0] for s in starts))
    mus = sorted(set(s[1] for s in starts))
    n_h = len(mus); n_o = len(omegas); n_s = n // (n_o * n_h)
    print(f"  {n} records: {n_o} ω × {n_h} headings × {n_s} speeds")
    print(f"  NOPST: {len(nopsts)}, MARUO: {len(maruos)}")
    return starts, totals, nopsts, maruos, omegas, mus, n_s, n_h, n_o

data = parse_debug('/home/blofro/src/pdstrip_test/kvlcc2/debug.out')
starts, totals, nopsts, maruos, omegas, mus, n_s, n_h, n_o = data

# Extract head seas at speed index 2 (V=3 m/s)
target_mu = 180.0
speed_idx = 2
mu_idx = None
for i, m in enumerate(mus):
    if abs(m - target_mu) < 1.0:
        mu_idx = i; break

results = []
for iom in range(n_o):
    idx = iom * (n_s * n_h) + speed_idx * n_h + mu_idx
    if idx >= len(starts): continue
    omega = starts[idx][0]
    mu = starts[idx][1]
    if abs(mu - target_mu) > 1.0: continue
    
    lam_L = 2*np.pi*g/omega**2 / Lpp
    d = totals[idx]
    np_d = nopsts[idx] if idx < len(nopsts) else None
    m_d = maruos[idx] if idx < len(maruos) else None
    
    r = {
        'omega': omega, 'lam_L': lam_L,
        'fxi': d['fxi'], 'fxi_WL': d['fxi_WL'], 'fxi_vel': d['fxi_vel'], 'fxi_rot': d['fxi_rot'],
    }
    if np_d:
        r['fxi_WL_nopst'] = np_d['fxi_WL_nopst']
        r['fxi_nopst_total'] = np_d['fxi_WL_nopst'] + d['fxi_vel'] + d['fxi_rot']
        if 'fxi_rot_nopst' in np_d:
            r['fxi_rot_nopst'] = np_d['fxi_rot_nopst']
            r['fxi_fullnopst_total'] = np_d['fxi_WL_nopst'] + d['fxi_vel'] + np_d['fxi_rot_nopst']
    if m_d:
        r['fxi_maruo'] = m_d['fxi_maruo']
    results.append(r)

results.sort(key=lambda x: x['lam_L'])

# Compute sigma values
for r in results:
    r['sigma'] = -r['fxi'] / norm
    r['sigma_WL'] = -r['fxi_WL'] / norm
    r['sigma_vel'] = -r['fxi_vel'] / norm
    r['sigma_rot'] = -r['fxi_rot'] / norm
    if 'fxi_WL_nopst' in r:
        r['sigma_WL_nopst'] = -r['fxi_WL_nopst'] / norm
        r['sigma_nopst_total'] = -r['fxi_nopst_total'] / norm
    if 'fxi_rot_nopst' in r:
        r['sigma_rot_nopst'] = -r['fxi_rot_nopst'] / norm
        r['sigma_fullnopst_total'] = -r['fxi_fullnopst_total'] / norm
    if 'fxi_maruo' in r:
        r['sigma_maruo'] = -r['fxi_maruo'] / norm

# Table
print("\n" + "="*170)
print("HEAD SEAS (β=180°) — Full decomposition")
print("="*170)
print(f"{'λ/L':>6} | {'Total':>8} | {'WL':>8} | {'WL_nopst':>8} | {'vel²':>8} | {'rot':>8} | {'rot_np':>8} | "
      f"{'nopst+v+r':>9} | {'full_np':>8} | {'Maruo':>8} | {'SWAN1':>7} | {'pst_WL':>8} | {'pst_rot':>8}")
print("-"*170)

seo_i = interp1d([s[0] for s in seo_data[180]], [s[1] for s in seo_data[180]],
                  bounds_error=False, fill_value=np.nan)

for r in results:
    s = seo_i(r['lam_L'])
    nopst_tot = r.get('sigma_nopst_total', np.nan)
    fullnp = r.get('sigma_fullnopst_total', np.nan)
    maruo = r.get('sigma_maruo', np.nan)
    rot_np = r.get('sigma_rot_nopst', np.nan)
    pst_WL = r['sigma_WL'] - r.get('sigma_WL_nopst', np.nan)
    pst_rot = r['sigma_rot'] - r.get('sigma_rot_nopst', np.nan) if 'sigma_rot_nopst' in r else np.nan
    s_str = f"{s:7.2f}" if not np.isnan(s) else "    ---"
    m_str = f"{maruo:8.3f}" if not np.isnan(maruo) else "     ---"
    print(f"{r['lam_L']:6.3f} | {r['sigma']:8.3f} | {r['sigma_WL']:8.3f} | "
          f"{r.get('sigma_WL_nopst', np.nan):8.3f} | {r['sigma_vel']:8.3f} | {r['sigma_rot']:8.3f} | "
          f"{rot_np:8.3f} | "
          f"{nopst_tot:9.3f} | {fullnp:8.3f} | {m_str} | {s_str} | {pst_WL:8.3f} | {pst_rot:8.3f}")


# ============================================================
# Plot: 4-panel comparison
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Total drift comparison
ax = axes[0,0]
sd = seo_data[180]
ax.plot([s[0] for s in sd], [s[1] for s in sd], 'b-o', ms=4, lw=2, label='SWAN1', zorder=10)

lam = [r['lam_L'] for r in results]
ax.plot(lam, [r['sigma'] for r in results], 'r-^', ms=3, lw=1.5, label='Pinkster total', zorder=5)
ax.plot(lam, [r.get('sigma_nopst_total', np.nan) for r in results], 'g-s', ms=3, lw=1.5, 
        label='WL_nopst + vel² + rot', zorder=7)
ax.plot(lam, [r.get('sigma_fullnopst_total', np.nan) for r in results], 'c-v', ms=4, lw=2, 
        label='WL_nopst + vel² + rot_nopst', zorder=8)
ax.plot(lam, [r.get('sigma_maruo', np.nan) for r in results], 'm-d', ms=3, lw=1.5, alpha=0.7,
        label='Maruo/G-B', zorder=6)
ax.set_xlim(0, 2.5); ax.set_ylim(-4, 7)
ax.set_xlabel('λ/L'); ax.set_ylabel('σ_aw')
ax.set_title('(a) Total drift: various methods')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.axhline(0, color='k', lw=0.5)

# Panel 2: WL components
ax = axes[0,1]
ax.plot(lam, [r['sigma_WL'] for r in results], 'r-^', ms=3, lw=1.5, label='WL (with pst)')
ax.plot(lam, [r.get('sigma_WL_nopst', np.nan) for r in results], 'g-s', ms=4, lw=2, label='WL (no pst)')
pst_c = [r['sigma_WL'] - r.get('sigma_WL_nopst', np.nan) for r in results]
ax.plot(lam, pst_c, 'c--o', ms=3, lw=1, label='pst contribution')
ax.set_xlim(0, 2.5); ax.set_ylim(-5, 12)
ax.set_xlabel('λ/L'); ax.set_ylabel('σ_aw')
ax.set_title('(b) WL integral decomposition')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.axhline(0, color='k', lw=0.5)

# Panel 3: no-pst total vs SWAN1
ax = axes[1,0]
ax.plot([s[0] for s in sd], [s[1] for s in sd], 'b-o', ms=4, lw=2, label='SWAN1', zorder=10)
ax.plot(lam, [r.get('sigma_nopst_total', np.nan) for r in results], 'g-s', ms=3, lw=1.5,
        label='WL_nopst + vel² + rot', zorder=7)
ax.plot(lam, [r.get('sigma_fullnopst_total', np.nan) for r in results], 'c-v', ms=4, lw=2,
        label='WL_nopst + vel² + rot_nopst', zorder=8)
ax.plot(lam, [r.get('sigma_maruo', np.nan) for r in results], 'm-d', ms=3, lw=1.5, alpha=0.7,
        label='Maruo/G-B', zorder=6)

ax.set_xlim(0, 2.5); ax.set_ylim(-4, 4)
ax.set_xlabel('λ/L'); ax.set_ylabel('σ_aw')
ax.set_title('(c) Alternative methods vs SWAN1')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.axhline(0, color='k', lw=0.5)

# Panel 4: Component stacking
ax = axes[1,1]
ax.fill_between(lam, 0, [r.get('sigma_WL_nopst', 0) for r in results], alpha=0.3, color='green', label='WL (no pst)')
ax.fill_between(lam, [r.get('sigma_WL_nopst', 0) for r in results], 
                [r.get('sigma_WL_nopst', 0) + r['sigma_vel'] for r in results], alpha=0.3, color='purple', label='vel²')
ax.plot(lam, [r.get('sigma_nopst_total', np.nan) for r in results], 'g-s', ms=3, lw=2, label='Total (no-pst)')
ax.plot([s[0] for s in sd], [s[1] for s in sd], 'b-o', ms=4, lw=2, label='SWAN1', zorder=10)
ax.set_xlim(0, 2.5); ax.set_ylim(-4, 4)
ax.set_xlabel('λ/L'); ax.set_ylabel('σ_aw')
ax.set_title('(d) Component stacking (no-pst)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.axhline(0, color='k', lw=0.5)

plt.suptitle('KVLCC2 Head Seas Drift Force: pst analysis\n'
             'V=3 m/s, 15% roll damping, sym-fixed BEM', fontsize=12, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/pst_deep_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved: pst_deep_analysis.png")

# Key metrics
print("\n" + "="*80)
print("KEY METRICS at head seas")
print("="*80)
# Find resonance peak (around lam/L ~ 1.0-1.1)
mask = [0.8 <= r['lam_L'] <= 1.3 for r in results]
res_peak = [r for r, m in zip(results, mask) if m]
if res_peak:
    i_peak = np.argmax([r['sigma'] for r in res_peak])
    rp = res_peak[i_peak]
    print(f"Pinkster peak: σ={rp['sigma']:.3f} at λ/L={rp['lam_L']:.3f}")
    print(f"  WL component: σ_WL={rp['sigma_WL']:.3f}")
    print(f"  WL no-pst:    σ_WL_nopst={rp.get('sigma_WL_nopst', np.nan):.3f}")
    print(f"  pst contrib:  {rp['sigma_WL'] - rp.get('sigma_WL_nopst', np.nan):.3f}")
    print(f"  vel² component: σ_vel={rp['sigma_vel']:.3f}")
    print(f"  rot (with pst): {rp.get('sigma_rot', np.nan):.3f}")
    print(f"  rot (no pst):   {rp.get('sigma_rot_nopst', np.nan):.3f}")
    print(f"  no-pst WL+vel+rot:     {rp.get('sigma_nopst_total', np.nan):.3f}")
    print(f"  full no-pst (WL+vel+rot_np): {rp.get('sigma_fullnopst_total', np.nan):.3f}")
    print(f"  Maruo:        {rp.get('sigma_maruo', np.nan):.3f}")
    print(f"  SWAN1:        ~2.5")

# RMS errors
common = np.linspace(0.35, 2.0, 100)
seo_v = seo_i(common)

for label, key in [('Pinkster (full)', 'sigma'), 
                    ('no-pst WL+vel+rot', 'sigma_nopst_total'),
                    ('full no-pst', 'sigma_fullnopst_total'),
                    ('Maruo', 'sigma_maruo')]:
    vals = [r.get(key, np.nan) for r in results]
    lams = [r['lam_L'] for r in results]
    valid = [(l, v) for l, v in zip(lams, vals) if not np.isnan(v)]
    if len(valid) < 3: continue
    interp = interp1d([v[0] for v in valid], [v[1] for v in valid], bounds_error=False, fill_value=np.nan)
    pred = interp(common)
    m = ~np.isnan(seo_v) & ~np.isnan(pred)
    if m.sum() > 0:
        rms = np.sqrt(np.mean((pred[m] - seo_v[m])**2))
        print(f"RMS error ({label:20s}): {rms:.3f}")
