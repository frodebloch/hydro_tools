#!/usr/bin/env python3
"""
Plot blended drift force results: Pinkster (unblended), REFL asymptote, blended,
and true original baseline. Surge and sway, all headings, V=0.
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Lpp = 328.2
B = 58.0
rho = 1025.0
g = 9.81
surge_norm = rho * g * B**2 / Lpp    # 103,065 N
sway_norm = 2 * rho * g * Lpp        # 6,600,266 N (not actually used for sway normalization yet)

# SWAN1 reference data (digitized from Liu & Papanikolaou 2021, Fig 11)
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


def parse_pdstrip_out(fname, has_unblended=False):
    """Parse drift force data from pdstrip.out.
    Returns dict: (speed, heading) -> list of (wavelength, fxi, feta, fxi_pink, feta_pink, fxi_refl, feta_refl)
    """
    with open(fname, 'r') as f:
        lines = f.readlines()

    data = {}
    wl = spd = hdg = None
    fxi_refl = feta_refl = None

    for i, line in enumerate(lines):
        m = re.search(r'wave length\s+([\d.]+).*wave number\s+([\d.]+).*wave angle\s+([-\d.]+)', line)
        if m:
            wl = float(m.group(1))
            wn = float(m.group(2))
            hdg = float(m.group(3))

        m = re.search(r'speed\s+([\d.]+)\s+wetted', line)
        if m:
            spd = float(m.group(1))

        m = re.search(r'Short-wave reflection asymptotic: fxi=\s*([-\d.Ee+]+)\s+feta=\s*([-\d.Ee+]+)', line)
        if m:
            fxi_refl = float(m.group(1))
            feta_refl = float(m.group(2))

        if has_unblended and 'Pinkster (unblended)' in line:
            vals = re.findall(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?', line)
            if len(vals) >= 2:
                fxi_pink = float(vals[-2])
                feta_pink = float(vals[-1])
                # Next line: blended
                for j in range(i+1, min(i+5, len(lines))):
                    if 'Longitudinal and transverse drift' in lines[j]:
                        vals2 = re.findall(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?', lines[j])
                        if len(vals2) >= 2:
                            fxi_b = float(vals2[-2])
                            feta_b = float(vals2[-1])
                            key = (spd, hdg)
                            if key not in data:
                                data[key] = []
                            data[key].append({
                                'wl': wl, 'fxi': fxi_b, 'feta': feta_b,
                                'fxi_pink': fxi_pink, 'feta_pink': feta_pink,
                                'fxi_refl': fxi_refl, 'feta_refl': feta_refl,
                            })
                        break

        elif not has_unblended and 'Longitudinal and transverse drift' in line:
            vals = re.findall(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?', line)
            if len(vals) >= 2:
                fxi_b = float(vals[-2])
                feta_b = float(vals[-1])
                key = (spd, hdg)
                if key not in data:
                    data[key] = []
                data[key].append({
                    'wl': wl, 'fxi': fxi_b, 'feta': feta_b,
                    'fxi_pink': fxi_b, 'feta_pink': feta_b,  # same (no unblended)
                    'fxi_refl': None, 'feta_refl': None,
                })

    return data


print("Parsing current (blended) output...")
current = parse_pdstrip_out('/home/blofro/src/pdstrip_test/kvlcc2/pdstrip.out', has_unblended=True)
print(f"  {len(current)} (speed, heading) combos")

print("Parsing true original output...")
original = parse_pdstrip_out('/home/blofro/src/pdstrip/kvlcc2_section/pdstrip.out', has_unblended=False)
print(f"  {len(original)} (speed, heading) combos")


# ============================================================
# SURGE DRIFT PLOTS — 7 headings, V=0
# ============================================================
headings = [180, 150, 120, 90]
subplot_labels = ['(a)', '(b)', '(c)', '(d)']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes_flat = axes.flatten()

for idx, hdg in enumerate(headings):
    ax = axes_flat[idx]

    key_cur = (0.0, float(hdg))
    key_orig = (0.0, float(hdg))

    # Original baseline
    if key_orig in original:
        d = sorted(original[key_orig], key=lambda r: r['wl'])
        lol = [r['wl']/Lpp for r in d]
        sig = [-r['fxi']/surge_norm for r in d]
        ax.plot(lol, sig, 'k-o', ms=3, lw=1.5, label='Original (Pinkster)', zorder=5)

    # Current: Pinkster unblended, blended, REFL
    if key_cur in current:
        d = sorted(current[key_cur], key=lambda r: r['wl'])
        lol = [r['wl']/Lpp for r in d]

        # Pinkster unblended
        sig_pink = [-r['fxi_pink']/surge_norm for r in d]
        ax.plot(lol, sig_pink, 'r--^', ms=3, lw=1, alpha=0.6, label='Pinkster (modified, unblended)', zorder=4)

        # REFL asymptote
        sig_refl = [-r['fxi_refl']/surge_norm for r in d if r['fxi_refl'] is not None]
        lol_refl = [r['wl']/Lpp for r in d if r['fxi_refl'] is not None]
        if sig_refl:
            ax.plot(lol_refl, sig_refl, 'g:', ms=2, lw=1.5, alpha=0.7, label='REFL asymptote', zorder=3)

        # Blended result
        sig_blend = [-r['fxi']/surge_norm for r in d]
        ax.plot(lol, sig_blend, 'b-s', ms=4, lw=2, label='Blended result', zorder=8)

        # Mark where REFL was used
        for r in d:
            if abs(r['fxi'] - r['fxi_pink']) > 1e-6:
                ax.plot(r['wl']/Lpp, -r['fxi']/surge_norm, 'gD', ms=7, zorder=9, alpha=0.5)

    # SWAN1 reference
    if hdg in seo_data:
        sd = seo_data[hdg]
        ax.plot([s[0] for s in sd], [s[1] for s in sd], 'c-', ms=3, lw=2, alpha=0.6, label='SWAN1 (ref)', zorder=7)

    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$\sigma_{aw} = -F_x / (\rho g B^2/L)$')
    ax.set_title(f'{subplot_labels[idx]} Head={hdg}°, V=0')
    ax.set_xlim(0, 3.2)
    ax.axhline(0, color='k', lw=0.5)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7, loc='upper right')

plt.suptitle('KVLCC2 Surge Drift Force: Blended Pinkster+REFL vs Original\n'
             'V=0, Green diamonds = REFL used', fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/blending_surge_v0.png', dpi=150, bbox_inches='tight')
print("Saved: blending_surge_v0.png")


# ============================================================
# SWAY DRIFT PLOTS — 4 headings, V=0
# ============================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
axes2_flat = axes2.flatten()

for idx, hdg in enumerate(headings):
    ax = axes2_flat[idx]

    key_cur = (0.0, float(hdg))
    key_orig = (0.0, float(hdg))

    # Original baseline
    if key_orig in original:
        d = sorted(original[key_orig], key=lambda r: r['wl'])
        lol = [r['wl']/Lpp for r in d]
        sig = [r['feta']/surge_norm for r in d]  # use same normalization for comparison
        ax.plot(lol, sig, 'k-o', ms=3, lw=1.5, label='Original', zorder=5)

    # Current
    if key_cur in current:
        d = sorted(current[key_cur], key=lambda r: r['wl'])
        lol = [r['wl']/Lpp for r in d]

        sig_pink = [r['feta_pink']/surge_norm for r in d]
        ax.plot(lol, sig_pink, 'r--^', ms=3, lw=1, alpha=0.6, label='Pinkster (unblended)', zorder=4)

        sig_refl = [r['feta_refl']/surge_norm for r in d if r['feta_refl'] is not None]
        lol_refl = [r['wl']/Lpp for r in d if r['feta_refl'] is not None]
        if sig_refl:
            ax.plot(lol_refl, sig_refl, 'g:', ms=2, lw=1.5, alpha=0.7, label='REFL asymptote', zorder=3)

        sig_blend = [r['feta']/surge_norm for r in d]
        ax.plot(lol, sig_blend, 'b-s', ms=4, lw=2, label='Blended', zorder=8)

        for r in d:
            if abs(r['feta'] - r['feta_pink']) > 1e-6:
                ax.plot(r['wl']/Lpp, r['feta']/surge_norm, 'gD', ms=7, zorder=9, alpha=0.5)

    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$F_y / (\rho g B^2/L)$')
    ax.set_title(f'{subplot_labels[idx]} Head={hdg}°, V=0')
    ax.set_xlim(0, 3.2)
    ax.axhline(0, color='k', lw=0.5)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7, loc='upper right')

plt.suptitle('KVLCC2 Sway Drift Force: Blended Pinkster+REFL vs Original\n'
             'V=0, Green diamonds = REFL used', fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/blending_sway_v0.png', dpi=150, bbox_inches='tight')
print("Saved: blending_sway_v0.png")


# ============================================================
# DETAILED HEAD SEAS PLOT — close-up showing REFL transition
# ============================================================
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

key = (0.0, 180.0)

# Left: full range
if key in current:
    d = sorted(current[key], key=lambda r: r['wl'])
    lol = [r['wl']/Lpp for r in d]
    ax1.plot(lol, [-r['fxi_pink']/surge_norm for r in d], 'r--^', ms=4, lw=1.5, label='Pinkster (unblended)')
    ax1.plot(lol, [-r['fxi_refl']/surge_norm for r in d if r['fxi_refl'] is not None][:len(lol)],
             'g--', ms=3, lw=1.5, label='REFL asymptote')
    ax1.plot(lol, [-r['fxi']/surge_norm for r in d], 'b-s', ms=5, lw=2, label='Blended')
    for r in d:
        if abs(r['fxi'] - r['fxi_pink']) > 1e-6:
            ax1.plot(r['wl']/Lpp, -r['fxi']/surge_norm, 'gD', ms=8, alpha=0.5)
if key in original:
    d = sorted(original[key], key=lambda r: r['wl'])
    ax1.plot([r['wl']/Lpp for r in d], [-r['fxi']/surge_norm for r in d], 'k-o', ms=3, lw=1.5, label='Original')
if 180 in seo_data:
    sd = seo_data[180]
    ax1.plot([s[0] for s in sd], [s[1] for s in sd], 'c-', lw=2, alpha=0.6, label='SWAN1')

ax1.set_xlim(0, 3.2); ax1.set_ylim(-1, 6)
ax1.set_xlabel(r'$\lambda/L$'); ax1.set_ylabel(r'$\sigma_{aw}$')
ax1.set_title('Head seas (180°) — Full range')
ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3); ax1.axhline(0, color='k', lw=0.5)

# Right: zoom into short waves
if key in current:
    d = sorted(current[key], key=lambda r: r['wl'])
    lol = [r['wl']/Lpp for r in d]
    lol_refl = [r['wl']/Lpp for r in d if r['fxi_refl'] is not None]
    ax2.plot(lol, [-r['fxi_pink']/surge_norm for r in d], 'r--^', ms=5, lw=1.5, label='Pinkster (unblended)')
    ax2.plot(lol_refl, [-r['fxi_refl']/surge_norm for r in d if r['fxi_refl'] is not None],
             'g--s', ms=4, lw=1.5, label='REFL asymptote')
    ax2.plot(lol, [-r['fxi']/surge_norm for r in d], 'b-s', ms=5, lw=2, label='Blended')
    for r in d:
        if abs(r['fxi'] - r['fxi_pink']) > 1e-6:
            ax2.plot(r['wl']/Lpp, -r['fxi']/surge_norm, 'gD', ms=9, alpha=0.5)
if key in original:
    d = sorted(original[key], key=lambda r: r['wl'])
    ax2.plot([r['wl']/Lpp for r in d], [-r['fxi']/surge_norm for r in d], 'k-o', ms=4, lw=1.5, label='Original')
if 180 in seo_data:
    sd = seo_data[180]
    ax2.plot([s[0] for s in sd], [s[1] for s in sd], 'c-', lw=2, alpha=0.6, label='SWAN1')

ax2.set_xlim(0, 0.6); ax2.set_ylim(-1, 3)
ax2.set_xlabel(r'$\lambda/L$'); ax2.set_ylabel(r'$\sigma_{aw}$')
ax2.set_title('Head seas (180°) — Short wave zoom')
ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3); ax2.axhline(0, color='k', lw=0.5)

plt.suptitle('KVLCC2 Head Seas: Pinkster+REFL Blending Detail\nV=0', fontsize=12, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/blending_headseas_detail.png', dpi=150, bbox_inches='tight')
print("Saved: blending_headseas_detail.png")


# ============================================================
# NEAR-HEAD SEAS DETAIL: 170, 160, 150 — where blending is active
# ============================================================
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))

for idx, hdg in enumerate([170, 160, 150]):
    ax = axes4[idx]
    key_cur = (0.0, float(hdg))
    key_orig = (0.0, float(hdg))

    if key_orig in original:
        d = sorted(original[key_orig], key=lambda r: r['wl'])
        ax.plot([r['wl']/Lpp for r in d], [-r['fxi']/surge_norm for r in d], 'k-o', ms=3, lw=1.5, label='Original')

    if key_cur in current:
        d = sorted(current[key_cur], key=lambda r: r['wl'])
        lol = [r['wl']/Lpp for r in d]
        ax.plot(lol, [-r['fxi_pink']/surge_norm for r in d], 'r--^', ms=3, lw=1, alpha=0.6, label='Pinkster (unbl.)')
        lol_refl = [r['wl']/Lpp for r in d if r['fxi_refl'] is not None]
        sig_refl = [-r['fxi_refl']/surge_norm for r in d if r['fxi_refl'] is not None]
        if sig_refl:
            ax.plot(lol_refl, sig_refl, 'g:', lw=1.5, alpha=0.7, label='REFL')
        ax.plot(lol, [-r['fxi']/surge_norm for r in d], 'b-s', ms=4, lw=2, label='Blended')
        for r in d:
            if abs(r['fxi'] - r['fxi_pink']) > 1e-6:
                ax.plot(r['wl']/Lpp, -r['fxi']/surge_norm, 'gD', ms=7, zorder=9, alpha=0.5)

    if hdg in seo_data:
        sd = seo_data[hdg]
        ax.plot([s[0] for s in sd], [s[1] for s in sd], 'c-', lw=2, alpha=0.6, label='SWAN1')

    ax.set_xlim(0, 2.5); ax.set_ylim(-2, 5)
    ax.set_xlabel(r'$\lambda/L$'); ax.set_ylabel(r'$\sigma_{aw}$')
    ax.set_title(f'Heading={hdg}°, V=0')
    ax.axhline(0, color='k', lw=0.5); ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7, loc='upper right')

plt.suptitle('KVLCC2 Near-Head Seas: Blending Detail at 170°, 160°, 150°\n'
             'V=0, Green diamonds = REFL used', fontsize=12, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/blending_nearhead_detail.png', dpi=150, bbox_inches='tight')
print("Saved: blending_nearhead_detail.png")

print("\nAll plots saved.")
