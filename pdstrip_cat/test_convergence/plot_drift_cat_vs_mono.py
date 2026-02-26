#!/usr/bin/env python3
"""Plot catamaran vs monohull drift force transfer functions.

Nondimensionalization:
  Surge: sigma_xi = F_xi / (rho*g*A^2 * B^2/L)
  Sway:  sigma_eta = F_eta / (rho*g*A^2 * L)

Catamaran: hulld=20m (gap ≈ 1 beam width), both hulls integrated.
Monohull: same geometry, single hull.

x-axis: lambda/L (wavelength / ship length)
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ===== Geometry parameters =====
Lpp = 91.9      # m
B = 20.09       # m (full beam, single hull)
T = 6.0         # m (draft)
rho = 1025.0    # kg/m^3
g = 9.81        # m/s^2
hulld = 20.0    # m (center-to-CL)

norm_xi = rho * g * B**2 / Lpp   # surge normalization
norm_eta = rho * g * Lpp          # sway normalization

# ===== Parse output =====
def parse_output(filename):
    headers = []
    seen = set()
    drift_forces = []
    with open(filename) as f:
        for line in f:
            m = re.search(r'Wave circ\. frequency\s+([\d.]+)\s+encounter frequ\.\s+[\d.]+\s+wave length\s+([\d.]+)\s+wave number\s+[\d.]+\s+wave angle\s+([-\d.]+)', line)
            if m:
                omega, wl, angle = float(m.group(1)), float(m.group(2)), float(m.group(3))
                key = (omega, angle)
                if key not in seen:
                    seen.add(key)
                    headers.append({'omega': omega, 'wavelength': wl, 'angle': angle})
            m = re.search(r'Longitudinal and transverse drift force per wave amplitude squared\s+([-\d.E+]+)\s+([-\d.E+]+)', line)
            if m:
                drift_forces.append((float(m.group(1)), float(m.group(2))))
    results = []
    for i, (fxi, feta) in enumerate(drift_forces):
        if i < len(headers):
            h = headers[i]
            results.append({
                'omega': h['omega'], 'wavelength': h['wavelength'],
                'angle': h['angle'], 'fxi': fxi, 'feta': feta,
                'lam_L': h['wavelength'] / Lpp
            })
    return results

print("Parsing catamaran (stb+port) output...")
cat = parse_output('/home/blofro/src/pdstrip_test/test_convergence/cat_20/pdstrip_out_stb_port.out')
print(f"  {len(cat)} entries")

print("Parsing monohull output...")
mono = parse_output('/home/blofro/src/pdstrip_test/test_convergence/mono_35/pdstrip.out')
print(f"  {len(mono)} entries")

# Build lookups
cat_lk = {(round(r['omega'], 3), round(r['angle'], 1)): r for r in cat}
mono_lk = {(round(r['omega'], 3), round(r['angle'], 1)): r for r in mono}

# ===== Headings to plot =====
plot_angles = [180, 150, 120, 90, 60, 30, 0]
angle_labels = {
    0: r'$\mu=0°$ (following)',
    30: r'$\mu=30°$',
    60: r'$\mu=60°$',
    90: r'$\mu=90°$ (beam)',
    120: r'$\mu=120°$',
    150: r'$\mu=150°$',
    180: r'$\mu=180°$ (head)',
}

all_omegas = sorted(set(round(r['omega'], 3) for r in cat))

# ===== Build data arrays =====
def get_curves(lookup, angles):
    curves = {}
    for mu in angles:
        lam_L_list, sigma_xi_list, sigma_eta_list = [], [], []
        for omega in sorted(all_omegas):
            key = (omega, float(mu))
            r = lookup.get(key)
            if r is None:
                continue
            lam_L_list.append(r['lam_L'])
            sigma_xi_list.append(r['fxi'] / norm_xi)
            sigma_eta_list.append(r['feta'] / norm_eta)
        curves[mu] = {
            'lam_L': np.array(lam_L_list),
            'sigma_xi': np.array(sigma_xi_list),
            'sigma_eta': np.array(sigma_eta_list),
        }
    return curves

cat_curves = get_curves(cat_lk, plot_angles)
mono_curves = get_curves(mono_lk, plot_angles)

# Minimum lambda/L cutoff — short wavelengths are noisy in catamaran BEM
LAM_MIN = 0.35

# ===== Color scheme — one color per angle =====
cmap_vals = [0.95, 0.82, 0.65, 0.50, 0.35, 0.18, 0.05]
colors = {mu: plt.cm.plasma(v) for mu, v in zip(plot_angles, cmap_vals)}

# ====================================================================
# PLOT 1: Surge TF — all angles overlaid
# ====================================================================
fig, ax = plt.subplots(figsize=(13, 7))

for mu in plot_angles:
    cc = cat_curves[mu]
    mc = mono_curves[mu]
    mask = cc['lam_L'] >= LAM_MIN
    mask_m = mc['lam_L'] >= LAM_MIN

    ax.plot(cc['lam_L'][mask], cc['sigma_xi'][mask], '-', color=colors[mu],
            linewidth=2.0, label=angle_labels[mu])
    ax.plot(mc['lam_L'][mask_m], mc['sigma_xi'][mask_m], '--', color=colors[mu],
            linewidth=1.2, alpha=0.55)

ax.set_xlabel(r'$\lambda\;/\;L$', fontsize=14)
ax.set_ylabel(r'$\sigma_{\xi} = F_{\xi}\;/\;(\rho\,g\,A^2\,B^2/L)$', fontsize=14)
ax.set_title('Surge Drift Force — Catamaran (solid) vs Monohull (dashed)\n'
             f'hulld = {hulld:.0f} m, gap ≈ 1 B, V = 0', fontsize=14)
ax.axhline(0, color='k', linewidth=0.4)
ax.set_xlim(LAM_MIN, 7.5)
ax.set_ylim(-15, 15)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=10, ncol=2, loc='lower right',
          title='Solid = catamaran, dashed = monohull')

plt.tight_layout()
plt.savefig('/home/blofro/src/pdstrip_test/test_convergence/drift_surge_overlay.png', dpi=150)
print("Saved drift_surge_overlay.png")
plt.close()

# ====================================================================
# PLOT 2: Sway TF — all angles overlaid
# ====================================================================
fig, ax = plt.subplots(figsize=(13, 7))

for mu in plot_angles:
    cc = cat_curves[mu]
    mc = mono_curves[mu]
    mask = cc['lam_L'] >= LAM_MIN
    mask_m = mc['lam_L'] >= LAM_MIN

    ax.plot(cc['lam_L'][mask], cc['sigma_eta'][mask], '-', color=colors[mu],
            linewidth=2.0, label=angle_labels[mu])
    ax.plot(mc['lam_L'][mask_m], mc['sigma_eta'][mask_m], '--', color=colors[mu],
            linewidth=1.2, alpha=0.55)

ax.set_xlabel(r'$\lambda\;/\;L$', fontsize=14)
ax.set_ylabel(r'$\sigma_{\eta} = F_{\eta}\;/\;(\rho\,g\,A^2\,L)$', fontsize=14)
ax.set_title('Sway Drift Force — Catamaran (solid) vs Monohull (dashed)\n'
             f'hulld = {hulld:.0f} m, gap ≈ 1 B, V = 0', fontsize=14)
ax.axhline(0, color='k', linewidth=0.4)
ax.set_xlim(LAM_MIN, 7.5)
ax.set_ylim(-3, 3)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=10, ncol=2, loc='lower right',
          title='Solid = catamaran, dashed = monohull')

plt.tight_layout()
plt.savefig('/home/blofro/src/pdstrip_test/test_convergence/drift_sway_overlay.png', dpi=150)
print("Saved drift_sway_overlay.png")
plt.close()

# ====================================================================
# PLOT 3: Per-angle panels — surge (left y-axis) + sway (right y-axis)
# ====================================================================
focus_angles = [180, 150, 120, 90, 60, 30]
nrows, ncols = 3, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 16), sharex=True)

for idx, mu in enumerate(focus_angles):
    row, col = idx // ncols, idx % ncols
    ax = axes[row, col]

    cc = cat_curves[mu]
    mc = mono_curves[mu]
    mask = cc['lam_L'] >= LAM_MIN
    mask_m = mc['lam_L'] >= LAM_MIN

    # Surge on left axis
    ax.plot(cc['lam_L'][mask], cc['sigma_xi'][mask], 'b-', linewidth=2.2,
            label=r'Cat $F_{\xi}$')
    ax.plot(mc['lam_L'][mask_m], mc['sigma_xi'][mask_m], 'b--', linewidth=1.4,
            alpha=0.55, label=r'Mono $F_{\xi}$')
    ax.plot(mc['lam_L'][mask_m], 2*mc['sigma_xi'][mask_m], 'b:', linewidth=1.0,
            alpha=0.40, label=r'2 × Mono $F_{\xi}$')

    # Auto y-limits for surge based on 5th-95th percentile
    xi_vals = np.concatenate([
        cc['sigma_xi'][mask], mc['sigma_xi'][mask_m],
        2*mc['sigma_xi'][mask_m],
    ])
    if len(xi_vals) > 0:
        lo = np.percentile(xi_vals, 3)
        hi = np.percentile(xi_vals, 97)
        margin = max(0.15 * (hi - lo), 0.3)
        ax.set_ylim(lo - margin, hi + margin)

    ax.set_ylabel(r'$\sigma_{\xi}$ (surge)', fontsize=10, color='blue')
    ax.tick_params(axis='y', labelcolor='blue')

    # Sway on right axis
    ax2 = ax.twinx()
    ax2.plot(cc['lam_L'][mask], cc['sigma_eta'][mask], 'r-', linewidth=2.2,
             label=r'Cat $F_{\eta}$')
    ax2.plot(mc['lam_L'][mask_m], mc['sigma_eta'][mask_m], 'r--', linewidth=1.4,
             alpha=0.55, label=r'Mono $F_{\eta}$')

    # Auto y-limits for sway
    eta_vals = np.concatenate([
        cc['sigma_eta'][mask], mc['sigma_eta'][mask_m],
    ])
    if len(eta_vals) > 0:
        lo_e = np.percentile(eta_vals, 3)
        hi_e = np.percentile(eta_vals, 97)
        margin_e = max(0.15 * (hi_e - lo_e), 0.05)
        ax2.set_ylim(lo_e - margin_e, hi_e + margin_e)

    ax2.set_ylabel(r'$\sigma_{\eta}$ (sway)', fontsize=10, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    label_str = angle_labels[mu]
    if mu == 180:
        label_str += ' **(head)**'
    elif mu == 90:
        label_str += ' **(beam)**'
    ax.set_title(angle_labels[mu], fontsize=13, fontweight='bold')
    ax.axhline(0, color='k', linewidth=0.4)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(LAM_MIN, 7.5)

    if idx == 0:
        # Combined legend from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=8, ncol=2, loc='best')
    if row == nrows - 1:
        ax.set_xlabel(r'$\lambda\;/\;L$', fontsize=12)

fig.suptitle('Drift Force Transfer Functions — Catamaran vs Monohull\n'
             f'hulld = {hulld:.0f} m, gap ≈ 1 B,  V = 0,  '
             f'B = {B:.1f} m,  L = {Lpp:.1f} m,  T = {T:.1f} m',
             fontsize=14, y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/home/blofro/src/pdstrip_test/test_convergence/drift_panels.png', dpi=150)
print("Saved drift_panels.png")
plt.close()

# ====================================================================
# PLOT 4: Interaction ratio cat / (2*mono) — per angle
# ====================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, (component, comp_key, comp_label) in zip(axes, [
    ('Surge', 'sigma_xi', r'$F_{\xi,cat}\;/\;(2\,F_{\xi,mono})$'),
    ('Sway',  'sigma_eta', r'$F_{\eta,cat}\;/\;(2\,F_{\eta,mono})$'),
]):
    for mu in plot_angles:
        cc = cat_curves[mu]
        mc = mono_curves[mu]
        mask = (cc['lam_L'] >= LAM_MIN)
        # threshold: mono drift must be significant to avoid 0/0 noise
        thresh = 0.15 if component == 'Surge' else 0.03
        sig = np.abs(mc[comp_key]) > thresh
        ok = mask & sig
        if np.sum(ok) < 2:
            continue
        ratio = cc[comp_key][ok] / (2 * mc[comp_key][ok])
        ax.plot(cc['lam_L'][ok], ratio, '-', color=colors[mu], linewidth=1.6,
                label=angle_labels[mu])

    ax.axhline(1.0, color='red', linewidth=1, linestyle=':', alpha=0.6,
               label='1.0 (no interaction)')
    ax.set_xlabel(r'$\lambda\;/\;L$', fontsize=13)
    ax.set_ylabel(comp_label, fontsize=13)
    ax.set_title(f'{component} interaction ratio', fontsize=13)
    ax.set_xlim(LAM_MIN, 7.5)
    ax.set_ylim(-3, 5)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc='best')

plt.tight_layout()
plt.savefig('/home/blofro/src/pdstrip_test/test_convergence/drift_interaction_ratio.png', dpi=150)
print("Saved drift_interaction_ratio.png")
plt.close()

print("\nDone — 4 plots saved.")
