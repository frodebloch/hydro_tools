#!/usr/bin/env python3
"""
Detailed head seas (beta=180°) analysis for KVLCC2 at V=3 m/s.
Plot pdstrip drift force components (WL, vel, rot) against SWAN1,
plus heave and pitch RAOs.
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# KVLCC2 parameters
rho = 1025.0
g = 9.81
Lpp = 328.2
B = 58.0
norm = rho * g * B**2 / Lpp
target_speed = 3.0

# Seo SWAN1 head seas data
seo_180 = [
    (0.30, 0.0), (0.35, 0.1), (0.40, 0.3), (0.45, 0.6),
    (0.50, 1.0), (0.55, 1.3), (0.60, 1.5), (0.65, 1.3),
    (0.70, 0.9), (0.75, 0.5), (0.80, 0.3), (0.85, 0.5),
    (0.90, 1.0), (0.95, 1.6), (1.00, 2.1), (1.05, 2.4),
    (1.10, 2.5), (1.15, 2.3), (1.20, 1.8), (1.25, 1.3),
    (1.30, 0.8), (1.35, 0.5), (1.40, 0.3), (1.50, 0.1),
    (1.60, 0.0), (1.80, 0.0), (2.00, 0.0), (2.50, 0.0),
]
seo_exp_f = [
    (0.50, 2.0), (0.60, 2.2), (0.70, 1.2), (0.80, 1.0),
    (0.90, 2.1), (1.00, 2.5), (1.05, 3.0), (1.10, 2.8),
    (1.20, 1.5), (1.30, 0.5),
]
seo_exp_l = [
    (0.50, 1.7), (0.60, 1.9), (0.70, 1.0), (0.80, 0.8),
    (0.90, 1.7), (1.00, 2.2), (1.05, 2.5), (1.10, 2.3),
    (1.20, 1.2),
]

# ============================================================
# 1. Parse pdstrip output (15% damping) — motions + drift
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
            surge_re, surge_im, surge_abs = float(m3.group(1)), float(m3.group(2)), float(m3.group(3))
            sway_re, sway_im, sway_abs = float(m3.group(4)), float(m3.group(5)), float(m3.group(6))
            heave_re, heave_im, heave_abs = float(m3.group(7)), float(m3.group(8)), float(m3.group(9))
        i += 1  # Rotation
        m4 = re.match(r'\s*Rotation/k\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                       r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                       r'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', lines[i])
        if m4:
            roll_abs = float(m4.group(3))
            pitch_re, pitch_im, pitch_abs = float(m4.group(4)), float(m4.group(5)), float(m4.group(6))
            yaw_abs = float(m4.group(9))
        i += 1  # Drift
        m5 = re.match(r'\s*Longitudinal and transverse drift force.*?\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)', lines[i])
        fxi = float(m5.group(1)) if m5 else 0
        feta = float(m5.group(2)) if m5 else 0

        results.append({
            'omega': omega, 'omega_e': omega_e, 'wavelength': wavelength,
            'wavenumber': wavenumber, 'wave_angle': wave_angle, 'speed': speed,
            'surge_abs': surge_abs, 'heave_abs': heave_abs, 'pitch_abs': pitch_abs,
            'heave_re': heave_re, 'heave_im': heave_im,
            'pitch_re': pitch_re, 'pitch_im': pitch_im,
            'roll_abs': roll_abs, 'yaw_abs': yaw_abs,
            'fxi': fxi, 'feta': feta,
        })
    i += 1

print(f"Parsed {len(results)} records")

# ============================================================
# 2. Parse debug output — drift force components
# ============================================================
def parse_debug_components(fname, target_speed_idx=2, target_mu=180.0):
    starts = []
    totals = []
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
                        'fxi': float(m.group(1)), 'feta': float(m.group(2)),
                        'fxi_WL': float(m.group(3)), 'feta_WL': float(m.group(4)),
                        'fxi_vel': float(m.group(5)), 'fxi_rot': float(m.group(6)),
                    })

    mus = sorted(set(s[1] for s in starts))
    n_headings = len(mus)
    n_speeds = 8
    mu_idx = None
    for i, m in enumerate(mus):
        if abs(m - target_mu) < 1.0:
            mu_idx = i
            break
    if mu_idx is None:
        print(f"  ERROR: mu={target_mu} not found in {mus}")
        return None

    omegas = sorted(set(s[0] for s in starts))
    n_omegas = len(omegas)

    comp_results = []
    for iom in range(n_omegas):
        record_idx = iom * (n_speeds * n_headings) + target_speed_idx * n_headings + mu_idx
        if record_idx < len(starts):
            omega = starts[record_idx][0]
            mu = starts[record_idx][1]
            if abs(mu - target_mu) > 1.0:
                continue
            wavelength = 2 * np.pi * g / omega**2
            lam_L = wavelength / Lpp
            d = totals[record_idx]
            comp_results.append({
                'omega': omega, 'lam_L': lam_L,
                'sigma_total': -d['fxi'] / norm,
                'sigma_WL': -d['fxi_WL'] / norm,
                'sigma_vel': -d['fxi_vel'] / norm,
                'sigma_rot': -d['fxi_rot'] / norm,
            })

    comp_results.sort(key=lambda x: x['lam_L'])
    return comp_results

print("Parsing debug components...")
comp = parse_debug_components('debug_15pct.out')

# ============================================================
# 3. Collect head seas pdstrip data at V=3 m/s
# ============================================================
head = [r for r in results
        if abs(r['speed'] - target_speed) < 0.1
        and abs(r['wave_angle'] - 180.0) < 0.5]
head.sort(key=lambda r: r['wavelength'])

lam_L_pd = np.array([r['wavelength'] / Lpp for r in head])
sigma_pd = np.array([-r['fxi'] / norm for r in head])
heave_rao = np.array([r['heave_abs'] for r in head])     # heave/A
pitch_rao = np.array([r['pitch_abs'] for r in head])     # pitch/(kA)
wavenum = np.array([r['wavenumber'] for r in head])

# ============================================================
# 4. Create 4-panel figure
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- (a) Total drift force ---
ax = axes[0, 0]
sx, sy = zip(*seo_180)
ax.plot(sx, sy, 'b-o', markersize=4, linewidth=1.8, label='SWAN1 (Seo)', zorder=5)
ex, ey = zip(*seo_exp_f)
ax.plot(ex, ey, 's', color='gray', markersize=7, markeredgecolor='black',
        markeredgewidth=0.5, label='Exp. (ForceS)', zorder=4)
lx, ly = zip(*seo_exp_l)
ax.plot(lx, ly, 'D', color='orange', markersize=6, markeredgecolor='black',
        markeredgewidth=0.5, label='Exp. (LineT)', zorder=4)
ax.plot(lam_L_pd, sigma_pd, 'r-^', markersize=4, linewidth=1.5, label='pdstrip', zorder=6)
ax.set_xlim(0, 2.0)
ax.set_ylim(-1, 7)
ax.set_xlabel(r'$\lambda/L$', fontsize=11)
ax.set_ylabel(r'$\sigma_{aw} = -F_x/(\rho g A^2 B^2/L)$', fontsize=10)
ax.set_title(r'(a) Total surge drift force, $\beta=180°$', fontsize=12, fontweight='bold')
ax.axhline(0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='upper right')

# --- (b) Component decomposition ---
ax = axes[0, 1]
if comp:
    cx = [c['lam_L'] for c in comp]
    ax.plot(cx, [c['sigma_total'] for c in comp], 'k-o', ms=3, lw=2, label='Total')
    ax.plot(cx, [c['sigma_WL'] for c in comp], 'b-s', ms=3, lw=1.5, label='WL (waterline)')
    ax.plot(cx, [c['sigma_vel'] for c in comp], 'r-^', ms=3, lw=1.5, label='Vel (velocity²)')
    ax.plot(cx, [c['sigma_rot'] for c in comp], 'g-v', ms=3, lw=1.5, label='Rot (rotation)')
    ax.plot(sx, sy, 'c--o', ms=3, lw=2, alpha=0.7, label='SWAN1', zorder=10)
ax.set_xlim(0, 2.0)
ax.set_ylim(-5, 12)
ax.set_xlabel(r'$\lambda/L$', fontsize=11)
ax.set_ylabel(r'$\sigma_{aw}$', fontsize=10)
ax.set_title(r'(b) Drift components, $\beta=180°$', fontsize=12, fontweight='bold')
ax.axhline(0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, loc='upper right')

# --- (c) Heave RAO ---
ax = axes[1, 0]
ax.plot(lam_L_pd, heave_rao, 'r-^', markersize=4, linewidth=1.5, label='pdstrip heave/A')
# Liu Fig 9(b) digitized heave RAO at 6 kts head seas (approximate)
liu_heave = [
    (0.3, 0.0), (0.4, 0.02), (0.5, 0.05), (0.6, 0.1), (0.7, 0.2),
    (0.8, 0.35), (0.9, 0.55), (1.0, 0.75), (1.05, 0.85), (1.1, 0.92),
    (1.15, 0.97), (1.2, 1.0), (1.3, 1.02), (1.5, 1.02), (1.8, 1.0),
    (2.0, 1.0), (2.5, 1.0),
]
lhx, lhy = zip(*liu_heave)
ax.plot(lhx, lhy, 'b--o', markersize=3, linewidth=1.2, alpha=0.7, label='Liu SWAN1 (approx)')
ax.set_xlim(0, 2.0)
ax.set_ylim(0, 1.4)
ax.set_xlabel(r'$\lambda/L$', fontsize=11)
ax.set_ylabel(r'$\xi_3/A$', fontsize=10)
ax.set_title(r'(c) Heave RAO, $\beta=180°$', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# --- (d) Pitch RAO ---
ax = axes[1, 1]
# pitch_abs is pitch/(kA), convert to pitch/kA
ax.plot(lam_L_pd, pitch_rao, 'r-^', markersize=4, linewidth=1.5, label='pdstrip pitch/(kA)')
# Liu Fig 9(c) digitized pitch RAO at 6 kts head seas (approximate)
liu_pitch = [
    (0.3, 0.0), (0.4, 0.02), (0.5, 0.05), (0.6, 0.15), (0.7, 0.3),
    (0.8, 0.5), (0.9, 0.7), (0.95, 0.8), (1.0, 0.88), (1.05, 0.92),
    (1.1, 0.9), (1.15, 0.85), (1.2, 0.78), (1.3, 0.65), (1.5, 0.48),
    (1.8, 0.3), (2.0, 0.2), (2.5, 0.1),
]
lpx, lpy = zip(*liu_pitch)
ax.plot(lpx, lpy, 'b--o', markersize=3, linewidth=1.2, alpha=0.7, label='Liu SWAN1 (approx)')
ax.set_xlim(0, 2.0)
ax.set_ylim(0, 1.2)
ax.set_xlabel(r'$\lambda/L$', fontsize=11)
ax.set_ylabel(r'$\xi_5/(kA)$', fontsize=10)
ax.set_title(r'(d) Pitch RAO, $\beta=180°$', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

plt.suptitle('KVLCC2 Head Seas Analysis: V = 3 m/s (≈6 kts), 15% roll damping',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('headseas_detail.png', dpi=150, bbox_inches='tight')
print("Saved: headseas_detail.png")

# ============================================================
# 5. Print component table
# ============================================================
from scipy.interpolate import interp1d
seo_interp = interp1d([s[0] for s in seo_180], [s[1] for s in seo_180],
                       bounds_error=False, fill_value=np.nan)

print(f"\n{'='*110}")
print(f"HEAD SEAS (beta=180°) COMPONENT DECOMPOSITION — 15% roll damping, V=3 m/s")
print(f"{'='*110}")
print(f"{'lam/L':>7} | {'total':>8} | {'WL':>8} | {'vel':>8} | {'rot':>8} | {'SWAN1':>7} | {'ratio':>7} | {'WL/tot':>7}")
print(f"{'-'*90}")

if comp:
    for c in comp:
        sval = seo_interp(c['lam_L'])
        ratio = c['sigma_total'] / sval if not np.isnan(sval) and abs(sval) > 0.05 else np.nan
        wl_frac = c['sigma_WL'] / c['sigma_total'] if abs(c['sigma_total']) > 0.01 else np.nan
        r_str = f"{ratio:7.2f}" if not np.isnan(ratio) else "    ---"
        w_str = f"{wl_frac:7.1%}" if not np.isnan(wl_frac) else "    ---"
        print(f"{c['lam_L']:7.3f} | {c['sigma_total']:8.3f} | {c['sigma_WL']:8.3f} | "
              f"{c['sigma_vel']:8.3f} | {c['sigma_rot']:8.3f} | {sval:7.3f} | {r_str} | {w_str}")
