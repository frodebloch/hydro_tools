#!/usr/bin/env python3
"""Plot catamaran drift forces across multiple hull separations.

Shows how interaction effects vary with hull separation.
Marks gap resonance wavelengths (lambda = 2*gap/n) which are a known
limitation of the 2D catamaran BEM in strip theory.

Hull separations: hulld = 20, 30, 50, 100 m
Reference: monohull (same geometry, single hull)
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== Geometry parameters =====
Lpp = 91.9
B = 20.09
T = 6.0
rho = 1025.0
g = 9.81

norm_xi = rho * g * B**2 / Lpp
norm_eta = rho * g * Lpp

LAM_MIN = 0.35

# ===== Parser =====
def parse_output(filename):
    headers, seen, drift_forces = [], set(), []
    with open(filename) as f:
        for line in f:
            m = re.search(
                r'Wave circ\. frequency\s+([\d.]+)\s+encounter frequ\.\s+[\d.]+\s+'
                r'wave length\s+([\d.]+)\s+wave number\s+[\d.]+\s+wave angle\s+([-\d.]+)', line)
            if m:
                omega, wl, angle = float(m.group(1)), float(m.group(2)), float(m.group(3))
                key = (omega, angle)
                if key not in seen:
                    seen.add(key)
                    headers.append({'omega': omega, 'wavelength': wl, 'angle': angle})
            m = re.search(
                r'Longitudinal and transverse drift force per wave amplitude squared'
                r'\s+([-\d.E+]+)\s+([-\d.E+]+)', line)
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

# ===== Load all cases =====
base = '/home/blofro/src/pdstrip_test/test_convergence'
cases = [
    (20,  f'{base}/cat_20/pdstrip_out_stb_port.out'),
    (30,  f'{base}/cat_30/pdstrip.out'),
    (50,  f'{base}/cat_50/pdstrip.out'),
    (100, f'{base}/cat_100/pdstrip.out'),
]

print("Loading monohull...")
mono = parse_output(f'{base}/mono_35/pdstrip.out')
mono_lk = {(round(r['omega'], 3), round(r['angle'], 1)): r for r in mono}
print(f"  {len(mono)} entries")

cat_data = {}
for hulld, path in cases:
    print(f"Loading cat hulld={hulld}...")
    data = parse_output(path)
    cat_data[hulld] = {(round(r['omega'], 3), round(r['angle'], 1)): r for r in data}
    print(f"  {len(data)} entries")

all_omegas = sorted(set(round(r['omega'], 3) for r in mono))

def get_curves(lookup, angles):
    curves = {}
    for mu in angles:
        lam_L, sigma_xi, sigma_eta = [], [], []
        for omega in sorted(all_omegas):
            r = lookup.get((omega, float(mu)))
            if r is None:
                continue
            lam_L.append(r['lam_L'])
            sigma_xi.append(r['fxi'] / norm_xi)
            sigma_eta.append(r['feta'] / norm_eta)
        curves[mu] = {
            'lam_L': np.array(lam_L),
            'sigma_xi': np.array(sigma_xi),
            'sigma_eta': np.array(sigma_eta),
        }
    return curves

# ===== Colors and styles for hull separations =====
hulld_colors = {20: '#d62728', 30: '#ff7f0e', 50: '#2ca02c', 100: '#1f77b4'}
hulld_styles = {20: '-', 30: '--', 50: '-.', 100: ':'}
hulld_lw = {20: 2.2, 30: 1.8, 50: 1.8, 100: 1.8}

# Gap in beam widths
gap_m = {h: 2*h - B for h in [20, 30, 50, 100]}
gap_B = {h: gap_m[h] / B for h in [20, 30, 50, 100]}

# Gap resonance wavelengths: lambda = 2*gap/n
def gap_resonances(hulld, n_max=8, lam_min=0.3, lam_max=8.0):
    gap = gap_m[hulld]
    resonances = []
    for n in range(1, n_max+1):
        lam_L = 2 * gap / (n * Lpp)
        if lam_L < lam_min:
            break
        if lam_L <= lam_max:
            resonances.append(lam_L)
    return resonances

angle_titles = {
    0: r'$\mu=0°$ (following)', 30: r'$\mu=30°$', 60: r'$\mu=60°$',
    90: r'$\mu=90°$ (beam)', 120: r'$\mu=120°$', 150: r'$\mu=150°$',
    180: r'$\mu=180°$ (head)',
}

def add_resonance_ticks(ax, hulld, color, ypos_frac=0.95):
    """Add small tick marks at gap resonance wavelengths along top of panel."""
    res = gap_resonances(hulld)
    ylims = ax.get_ylim()
    ytop = ylims[1]
    ybot = ylims[0]
    tick_len = 0.03 * (ytop - ybot)
    for lam_L in res:
        ax.plot([lam_L, lam_L], [ytop - tick_len, ytop], '-',
                color=color, linewidth=1.0, alpha=0.5, clip_on=True)


# ====================================================================
# PLOT 1: Surge drift — head seas only, focused comparison
# ====================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

focus_angles_surge = [180, 150, 90, 0]

for idx, mu in enumerate(focus_angles_surge):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]

    mc = get_curves(mono_lk, [mu])[mu]
    mask_m = mc['lam_L'] >= LAM_MIN
    ax.plot(mc['lam_L'][mask_m], 2 * mc['sigma_xi'][mask_m], 'k-',
            linewidth=2.5, alpha=0.35, label=r'2 × Mono', zorder=1)

    for hulld in [20, 30, 50, 100]:
        cc = get_curves(cat_data[hulld], [mu])[mu]
        mask = cc['lam_L'] >= LAM_MIN
        # Clip extreme BEM resonance values for cleaner plots
        xi_clipped = np.clip(cc['sigma_xi'][mask], -15, 15)
        lbl = f'hulld={hulld} (gap={gap_B[hulld]:.1f}B)'
        ax.plot(cc['lam_L'][mask], xi_clipped,
                linestyle=hulld_styles[hulld], color=hulld_colors[hulld],
                linewidth=hulld_lw[hulld], label=lbl, zorder=2)

    ax.set_title(angle_titles[mu], fontsize=13, fontweight='bold')
    ax.axhline(0, color='k', linewidth=0.4)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(LAM_MIN, 7.5)
    ax.set_ylim(-15, 8)

    # Add resonance markers
    for hulld in [20, 30, 50, 100]:
        add_resonance_ticks(ax, hulld, hulld_colors[hulld])

    if idx == 0:
        ax.legend(fontsize=8, loc='best')
    ax.set_xlabel(r'$\lambda\;/\;L$', fontsize=11)
    if col == 0:
        ax.set_ylabel(r'$\sigma_{\xi} = F_{\xi}\;/\;(\rho g A^2 B^2/L)$', fontsize=11)

fig.suptitle('Surge Drift Force — Effect of Hull Separation\n'
             f'L = {Lpp:.1f} m,  B = {B:.1f} m,  T = {T:.1f} m,  V = 0'
             '  (tick marks = gap resonance $\\lambda = 2 \\cdot gap / n$)',
             fontsize=13, y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{base}/drift_surge_separation.png', dpi=150)
print("Saved drift_surge_separation.png")
plt.close()

# ====================================================================
# PLOT 2: Sway drift — select headings, all separations
# ====================================================================
focus_angles_sway = [150, 120, 90, 60]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for idx, mu in enumerate(focus_angles_sway):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]

    mc = get_curves(mono_lk, [mu])[mu]
    mask_m = mc['lam_L'] >= LAM_MIN
    ax.plot(mc['lam_L'][mask_m], 2 * mc['sigma_eta'][mask_m], 'k-',
            linewidth=2.5, alpha=0.35, label=r'2 × Mono', zorder=1)

    for hulld in [20, 30, 50, 100]:
        cc = get_curves(cat_data[hulld], [mu])[mu]
        mask = cc['lam_L'] >= LAM_MIN
        eta_clipped = np.clip(cc['sigma_eta'][mask], -3, 3)
        lbl = f'hulld={hulld} (gap={gap_B[hulld]:.1f}B)'
        ax.plot(cc['lam_L'][mask], eta_clipped,
                linestyle=hulld_styles[hulld], color=hulld_colors[hulld],
                linewidth=hulld_lw[hulld], label=lbl, zorder=2)

    ax.set_title(angle_titles[mu], fontsize=13, fontweight='bold')
    ax.axhline(0, color='k', linewidth=0.4)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(LAM_MIN, 7.5)
    ax.set_ylim(-3, 1.5)

    for hulld in [20, 30, 50, 100]:
        add_resonance_ticks(ax, hulld, hulld_colors[hulld])

    if idx == 0:
        ax.legend(fontsize=8, loc='best')
    ax.set_xlabel(r'$\lambda\;/\;L$', fontsize=11)
    if col == 0:
        ax.set_ylabel(r'$\sigma_{\eta} = F_{\eta}\;/\;(\rho g A^2 L)$', fontsize=11)

fig.suptitle('Sway Drift Force — Effect of Hull Separation\n'
             f'L = {Lpp:.1f} m,  B = {B:.1f} m,  T = {T:.1f} m,  V = 0'
             '  (tick marks = gap resonance $\\lambda = 2 \\cdot gap / n$)',
             fontsize=13, y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{base}/drift_sway_separation.png', dpi=150)
print("Saved drift_sway_separation.png")
plt.close()

# ====================================================================
# PLOT 3: Interaction ratio cat/(2*mono) — with resonance markers
# ====================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

ratio_angles = [(180, 'Surge', 'sigma_xi'), (0, 'Surge', 'sigma_xi'),
                (90, 'Sway', 'sigma_eta'), (60, 'Sway', 'sigma_eta')]
panel_titles = [r'Head seas ($\mu=180°$) — Surge',
                r'Following seas ($\mu=0°$) — Surge',
                r'Beam seas ($\mu=90°$) — Sway',
                r'$\mu=60°$ — Sway']

for idx, (mu, comp, key) in enumerate(ratio_angles):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]

    mc = get_curves(mono_lk, [mu])[mu]

    for hulld in [20, 30, 50, 100]:
        cc = get_curves(cat_data[hulld], [mu])[mu]
        mask = cc['lam_L'] >= LAM_MIN
        thresh = 0.10
        sig = np.abs(mc[key]) > thresh
        ok = mask & sig
        if np.sum(ok) < 2:
            continue
        ratio = cc[key][ok] / (2 * mc[key][ok])
        # Clip ratio to visible range
        ratio_clipped = np.clip(ratio, -4, 6)
        lbl = f'hulld={hulld} (gap={gap_B[hulld]:.1f}B)'
        ax.plot(cc['lam_L'][ok], ratio_clipped,
                linestyle=hulld_styles[hulld], color=hulld_colors[hulld],
                linewidth=hulld_lw[hulld], label=lbl)

    ax.axhline(1.0, color='grey', linewidth=1.5, linestyle=':', alpha=0.7,
               label='1.0 (no interaction)')
    ax.set_title(panel_titles[idx], fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$\lambda\;/\;L$', fontsize=11)
    ax.set_ylabel(r'$F_{cat}\;/\;(2 \times F_{mono})$', fontsize=11)
    ax.set_xlim(LAM_MIN, 7.5)
    ax.set_ylim(-4, 6)
    ax.grid(True, alpha=0.25)

    # Add resonance markers
    for hulld in [20, 30, 50, 100]:
        add_resonance_ticks(ax, hulld, hulld_colors[hulld])

    ax.legend(fontsize=8, loc='best')

fig.suptitle('Interaction Ratio $F_{cat} / (2 \\times F_{mono})$\n'
             'Tick marks at top = gap resonance wavelengths ($\\lambda = 2 \\cdot gap / n$)',
             fontsize=13, y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{base}/drift_interaction_separation.png', dpi=150)
print("Saved drift_interaction_separation.png")
plt.close()

# ====================================================================
# PLOT 4: Gap resonance summary — show resonance wavelengths vs hulld
# ====================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for hulld in [20, 30, 50, 100]:
    gap = gap_m[hulld]
    ns = np.arange(1, 20)
    lam_L = 2 * gap / (ns * Lpp)
    valid = (lam_L >= 0.2) & (lam_L <= 8.0)
    ax.scatter(lam_L[valid], [hulld]*np.sum(valid), s=80, color=hulld_colors[hulld],
               marker='|', linewidths=2.5,
               label=f'hulld={hulld} (gap={gap_B[hulld]:.1f}B)')

# Shade the useful wavelength range
ax.axvspan(0.4, 3.0, alpha=0.08, color='green', label=r'Useful range ($0.4 < \lambda/L < 3$)')
ax.axvline(LAM_MIN, color='red', linewidth=1, linestyle=':', alpha=0.5, label=f'Plot cutoff ({LAM_MIN})')

ax.set_xlabel(r'$\lambda\;/\;L$', fontsize=13)
ax.set_ylabel('hulld (m)', fontsize=13)
ax.set_yticks([20, 30, 50, 100])
ax.set_xlim(0.2, 8)
ax.set_title('Gap Resonance Wavelengths $\\lambda = 2 \\cdot gap / n$\n'
             '2D catamaran BEM produces standing-wave artifacts at these wavelengths',
             fontsize=13)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(f'{base}/drift_gap_resonances.png', dpi=150)
print("Saved drift_gap_resonances.png")
plt.close()

print("\nDone — 4 plots saved.")
