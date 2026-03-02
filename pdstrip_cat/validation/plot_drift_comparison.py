#!/usr/bin/env python3
"""
Comprehensive drift force comparison: PDStrip mono/cat vs Nemoh mono/cat.
4 data sources x 4 headings x 2 DOFs (surge + sway).

Datasets:
  1. PDStrip monohull — Pinkster (near-field)
  2. PDStrip catamaran — Pinkster (cleaned)
  3. Nemoh monohull DUOK (3D panel method QTF)
  4. Nemoh catamaran DUOK (corrected mesh, 3D panel method QTF)

Output figures:
  1. Surge drift at 4 headings (Pinkster vs DUOK, mono + cat)
  2. Sway drift at 4 headings (Pinkster vs DUOK, mono + cat)
  3. Catamaran/monohull ratio (PDStrip Pinkster vs Nemoh DUOK)
  4. Nemoh DUOK per-term breakdown (cat, 3 headings)
  5. Head seas detail (Pinkster vs DUOK, mono + cat)
  6. Monohull comparison (Pinkster vs DUOK, 4 headings)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Coordinate conventions:
#   PDStrip: x forward, y starboard, z down.  mu = wave propagation angle from +x.
#   Nemoh:   x forward, y port,      z up.    beta = wave propagation angle from +x.
# Surge (fxi): same sign in both codes (x forward in both).
# Sway (feta): OPPOSITE sign — PDStrip y-starboard vs Nemoh y-port.
#   -> Negate PDStrip feta when comparing with Nemoh.

# Wigley hull parameters
L = 100.0       # length [m]
B = 10.0        # beam (single hull) [m]
T = 6.25        # draft [m]
HULLD = 10.0    # half CL-to-CL distance [m]
G = 9.81
RHO = 1025.0
NORM = RHO * G * B**2 / L  # = 10055.25 N/m^2

def rk(x):
    return round(x, 3)

# =====================================================================
# Parsers
# =====================================================================

def read_nemoh_diagonal(qtf_file, dof, beta_target):
    """Read Nemoh DUOK diagonal (w1=w2) for specific DOF and heading (in rad)."""
    omega_list, re_list = [], []
    with open(qtf_file) as f:
        next(f)  # skip header
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            w1, w2 = float(parts[0]), float(parts[1])
            beta1 = float(parts[2])
            d = int(parts[4])
            re_val = float(parts[5])
            if abs(w1 - w2) < 1e-4 and d == dof and abs(beta1 - beta_target) < 0.02:
                omega_list.append(w1)
                re_list.append(re_val)
    return np.array(omega_list), np.array(re_list)


def parse_mono_debug(debug_file):
    """Parse monohull debug.out: DRIFT_START -> DRIFT_TOTAL."""
    data = {}
    current_omega = None
    current_mu = None

    with open(debug_file) as f:
        for line in f:
            line = line.strip()

            if line.startswith('DRIFT_START'):
                parts = line.split()
                for i, t in enumerate(parts):
                    if t.startswith('omega=') and len(t.split('=')[1]) > 0:
                        current_omega = rk(float(t.split('=')[1]))
                    elif t == 'omega=' and i+1 < len(parts):
                        current_omega = rk(float(parts[i+1]))
                    if t.startswith('mu=') and len(t.split('=')[1]) > 0:
                        current_mu = round(float(t.split('=')[1]), 1)
                    elif t == 'mu=' and i+1 < len(parts):
                        current_mu = round(float(parts[i+1]), 1)

            elif line.startswith('DRIFT_TOTAL') and current_omega is not None:
                key = (current_omega, current_mu)
                data.setdefault(key, {})
                parts = line.split()
                for i, t in enumerate(parts):
                    if '=' in t and not t.startswith('DRIFT'):
                        k, _, v = t.partition('=')
                        if v:
                            try: data[key][k] = float(v)
                            except ValueError: pass
                        elif i+1 < len(parts):
                            try: data[key][k] = float(parts[i+1])
                            except (ValueError, IndexError): pass

    return data


def parse_cat_cleaned(debug_file):
    """Parse CAT_CLEANED lines: post-processed drift forces (after cap + antisymmetry)."""
    sway, surge = {}, {}
    with open(debug_file) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('CAT_CLEANED omega='):
                continue
            parts = line.split()
            kv = {}
            i = 1
            while i < len(parts):
                tok = parts[i]
                if '=' in tok:
                    key, _, val = tok.partition('=')
                    val = val.strip()
                    if val:
                        kv[key] = val
                    elif i+1 < len(parts) and '=' not in parts[i+1]:
                        kv[key] = parts[i+1]
                        i += 1
                i += 1
            try:
                omega = rk(float(kv['omega']))
                mu = round(float(kv['mu']), 1)
                feta = float(kv['feta'])
                fxi = float(kv['fxi'])
                k = (omega, mu)
                sway[k] = feta
                surge[k] = fxi
            except (KeyError, ValueError):
                pass
    return sway, surge


def extract_heading(data, mu_target, key):
    """Extract omega, value arrays for a specific heading from raw debug dict."""
    omegas, vals = [], []
    for (omega, mu), d in sorted(data.items()):
        if abs(mu - mu_target) < 0.5 and key in d:
            omegas.append(omega)
            vals.append(d[key])
    return np.array(omegas), np.array(vals)


def extract_cleaned_heading(cleaned_dict, mu_target):
    """Extract omega, value arrays from cleaned dict."""
    omegas, vals = [], []
    for (omega, mu), v in sorted(cleaned_dict.items()):
        if abs(mu - mu_target) < 0.5:
            omegas.append(omega)
            vals.append(v)
    return np.array(omegas), np.array(vals)


# =====================================================================
# Load all data
# =====================================================================
print("Loading data...")

# PDStrip monohull
mono_debug = parse_mono_debug('/home/blofro/src/pdstrip_test/wigley_mono/debug.out')
print(f"  Mono debug: {len(mono_debug)} (omega, mu) entries")

# PDStrip catamaran
cat_sway_cleaned, cat_surge_cleaned = parse_cat_cleaned('/home/blofro/src/pdstrip_test/wigley_cat/debug.out')
print(f"  Cat cleaned: {len(cat_sway_cleaned)} entries")

# Nemoh monohull
nemoh_mono_dir = '/home/blofro/src/pdstrip_test/wigley_nemoh_mono/results/QTF'

# Nemoh catamaran (corrected mesh)
nemoh_cat_dir = '/home/blofro/src/pdstrip_test/wigley_nemoh_cat_fixed/results/QTF'

outdir = '/home/blofro/src/pdstrip_test/wigley_nemoh_cat_fixed'

# Heading mapping: (label, PDStrip_internal_mu, Nemoh_beta_rad)
# PDStrip mu=180 is head seas, mu=150 is 30deg off bow, etc.
# Nemoh beta in radians: pi=head seas, 5pi/6=150deg, etc.
headings = [
    ("Head seas (180\u00b0)",     180.0,   np.pi),
    ("Oblique (150\u00b0)",       150.0,   5*np.pi/6),
    ("Oblique (120\u00b0)",       120.0,   2*np.pi/3),
    ("Beam seas (90\u00b0)",       90.0,   np.pi/2),
]

# Trapping frequency
def trap_omega(mu_deg, hulld=HULLD):
    sin_mu = abs(np.sin(np.radians(mu_deg)))
    if sin_mu < 0.05:
        return None
    k = np.pi / (2.0 * hulld * sin_mu)
    return np.sqrt(G * k)


# =====================================================================
# Figure 1: SURGE drift — catamaran: Pinkster, Maruo, Nemoh DUOK
# =====================================================================
print("\nGenerating Figure 1: Surge comparison...")
fig1, axes1 = plt.subplots(1, 4, figsize=(22, 5))
fig1.suptitle('Catamaran Surge Drift Force — Wigley Hull\n'
              r'$\sigma_{aw} = -F_x / (\rho g B^2/L)$, positive = added resistance',
              fontsize=14, y=1.02)

OMEGA_MAX = 1.8  # max practical frequency

for col, (label, mu_pd, beta_nem) in enumerate(headings):
    ax = axes1[col]
    trap_om = trap_omega(mu_pd)

    # PDStrip cat Pinkster (cleaned)
    om_cp, fxi_cp = extract_cleaned_heading(cat_surge_cleaned, mu_pd)
    if len(om_cp) > 0:
        mask = om_cp <= OMEGA_MAX
        sigma_cp = -fxi_cp / NORM
        ax.plot(om_cp[mask], sigma_cp[mask], 'r-o', ms=3, lw=1.5, label='PDStrip Pinkster')

    # Nemoh cat DUOK
    om_nc, re_nc = read_nemoh_diagonal(f'{nemoh_cat_dir}/QTFM_DUOK.dat', dof=1, beta_target=beta_nem)
    if len(om_nc) > 0:
        mask = om_nc <= OMEGA_MAX
        sigma_nc = -re_nc / NORM
        ax.plot(om_nc[mask], sigma_nc[mask], 'k--^', ms=3, lw=1.2, label='Nemoh DUOK')

    if trap_om is not None and trap_om <= OMEGA_MAX:
        ax.axvline(trap_om, color='purple', ls=':', lw=1, alpha=0.5, label=f'Trap ({trap_om:.2f})')

    ax.set_title(f'{label}', fontsize=10)
    ax.set_xlabel('\u03c9 [rad/s]')
    ax.set_xlim(0, OMEGA_MAX)
    if col == 0:
        ax.set_ylabel(r'$\sigma_{aw}$')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5)

plt.tight_layout()
fig1.savefig(f'{outdir}/drift_surge_4hdg.png', dpi=150)
print(f"  Saved {outdir}/drift_surge_4hdg.png")


# =====================================================================
# Figure 2: SWAY drift — catamaran: Pinkster, Nemoh DUOK
# =====================================================================
print("Generating Figure 2: Sway comparison...")
fig2, axes2 = plt.subplots(1, 4, figsize=(22, 5))
fig2.suptitle('Catamaran Sway Drift Force — Wigley Hull\n'
              r'$\sigma_{sway} = F_y / (\rho g B^2/L)$, positive = toward port (Nemoh convention)',
              fontsize=14, y=1.02)

for col, (label, mu_pd, beta_nem) in enumerate(headings):
    ax = axes2[col]
    trap_om = trap_omega(mu_pd)

    # PDStrip cat cleaned sway — negate feta (y-starboard -> y-port)
    om_cs, feta_cs = extract_cleaned_heading(cat_sway_cleaned, mu_pd)
    if len(om_cs) > 0:
        mask = om_cs <= OMEGA_MAX
        sigma_cs = -feta_cs / NORM
        ax.plot(om_cs[mask], sigma_cs[mask], 'r-o', ms=3, lw=1.5, label='PDStrip Pinkster')

    # Nemoh cat DUOK sway
    om_ncs, re_ncs = read_nemoh_diagonal(f'{nemoh_cat_dir}/QTFM_DUOK.dat', dof=2, beta_target=beta_nem)
    if len(om_ncs) > 0:
        mask = om_ncs <= OMEGA_MAX
        sigma_ncs = re_ncs / NORM
        ax.plot(om_ncs[mask], sigma_ncs[mask], 'k--^', ms=3, lw=1.2, label='Nemoh DUOK')

    if trap_om is not None and trap_om <= OMEGA_MAX:
        ax.axvline(trap_om, color='purple', ls=':', lw=1, alpha=0.5, label=f'Trap ({trap_om:.2f})')

    ax.set_title(f'{label}', fontsize=10)
    ax.set_xlabel('\u03c9 [rad/s]')
    ax.set_xlim(0, OMEGA_MAX)
    if col == 0:
        ax.set_ylabel(r'$\sigma_{sway}$')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5)

plt.tight_layout()
fig2.savefig(f'{outdir}/drift_sway_4hdg.png', dpi=150)
print(f"  Saved {outdir}/drift_sway_4hdg.png")


# =====================================================================
# Figure 3: Cat/mono ratio — PDStrip Pinkster vs Nemoh DUOK
# =====================================================================
print("Generating Figure 3: Cat/mono ratio...")
fig3, axes3 = plt.subplots(2, 4, figsize=(22, 10))
fig3.suptitle('Catamaran / Monohull Ratio — Surge & Sway\n'
              'Dashed line = 2.0 (non-interacting hulls)', fontsize=14, y=0.98)

for col, (label, mu_pd, beta_nem) in enumerate(headings):
    trap_om = trap_omega(mu_pd)

    # --- Top: Surge ratio ---
    ax = axes3[0, col]

    # PDStrip Pinkster ratio
    om_mp, fxi_mp = extract_heading(mono_debug, mu_pd, 'fxi')
    om_cp, fxi_cp = extract_cleaned_heading(cat_surge_cleaned, mu_pd)
    if len(om_mp) > 0 and len(om_cp) > 0:
        mono_dict_p = {rk(o): v for o, v in zip(om_mp, fxi_mp)}
        cat_dict_p = {rk(o): v for o, v in zip(om_cp, fxi_cp)}
        common_om_p = sorted(set(cat_dict_p.keys()) & set(mono_dict_p.keys()))
        ratio_om_p, ratio_v_p = [], []
        for o in common_om_p:
            vm, vc = mono_dict_p.get(o), cat_dict_p.get(o)
            if vm is not None and vc is not None and abs(vm) > 100:
                ratio_om_p.append(o)
                ratio_v_p.append(vc / vm)
        if ratio_om_p:
            ax.plot(ratio_om_p, ratio_v_p, 'b-o', ms=3, lw=1.2, label='PDStrip Pinkster')

    # Nemoh DUOK ratio
    om_nm, re_nm = read_nemoh_diagonal(f'{nemoh_mono_dir}/QTFM_DUOK.dat', dof=1, beta_target=beta_nem)
    om_nc, re_nc = read_nemoh_diagonal(f'{nemoh_cat_dir}/QTFM_DUOK.dat', dof=1, beta_target=beta_nem)
    if len(om_nm) > 0 and len(om_nc) > 0:
        nm_dict = {rk(o): v for o, v in zip(om_nm, re_nm)}
        nc_dict = {rk(o): v for o, v in zip(om_nc, re_nc)}
        common_nem = sorted(set(nm_dict.keys()) & set(nc_dict.keys()))
        ratio_nem_om, ratio_nem_v = [], []
        for o in common_nem:
            vm, vc = nm_dict.get(o), nc_dict.get(o)
            if vm is not None and vc is not None and abs(vm) > 1:
                ratio_nem_om.append(o)
                ratio_nem_v.append(vc / vm)
        if ratio_nem_om:
            ax.plot(ratio_nem_om, ratio_nem_v, 'r-^', ms=3, lw=1.0, alpha=0.7, label='Nemoh DUOK')

    ax.axhline(2.0, color='k', ls='--', lw=1, alpha=0.5)
    if trap_om is not None:
        ax.axvline(trap_om, color='purple', ls=':', lw=1, alpha=0.5)
    ax.set_title(f'Surge — {label}', fontsize=10)
    ax.set_xlabel('\u03c9 [rad/s]')
    if col == 0:
        ax.set_ylabel('Cat / Mono ratio')
    ax.set_ylim(0, 6)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # --- Bottom: Sway ratio ---
    ax2 = axes3[1, col]

    # PDStrip Pinkster sway ratio
    om_ms, feta_ms = extract_heading(mono_debug, mu_pd, 'feta')
    om_cs, feta_cs = extract_cleaned_heading(cat_sway_cleaned, mu_pd)
    if len(om_ms) > 0 and len(om_cs) > 0:
        mono_sway_dict = {rk(o): v for o, v in zip(om_ms, feta_ms)}
        cat_sway_dict = {rk(o): v for o, v in zip(om_cs, feta_cs)}
        common_sway = sorted(set(cat_sway_dict.keys()) & set(mono_sway_dict.keys()))
        ratio_sway_om, ratio_sway_v = [], []
        for o in common_sway:
            vm, vc = mono_sway_dict.get(o), cat_sway_dict.get(o)
            if vm is not None and vc is not None and abs(vm) > 100:
                ratio_sway_om.append(o)
                ratio_sway_v.append(vc / vm)
        if ratio_sway_om:
            ax2.plot(ratio_sway_om, ratio_sway_v, 'b-o', ms=3, lw=1.2, label='PDStrip Pinkster')

    # Nemoh DUOK sway ratio
    om_ns, re_ns = read_nemoh_diagonal(f'{nemoh_mono_dir}/QTFM_DUOK.dat', dof=2, beta_target=beta_nem)
    om_ncs, re_ncs = read_nemoh_diagonal(f'{nemoh_cat_dir}/QTFM_DUOK.dat', dof=2, beta_target=beta_nem)
    if len(om_ns) > 0 and len(om_ncs) > 0:
        ns_dict = {rk(o): v for o, v in zip(om_ns, re_ns)}
        ncs_dict = {rk(o): v for o, v in zip(om_ncs, re_ncs)}
        common_ns = sorted(set(ns_dict.keys()) & set(ncs_dict.keys()))
        ratio_ns_om, ratio_ns_v = [], []
        for o in common_ns:
            vm, vc = ns_dict.get(o), ncs_dict.get(o)
            if vm is not None and vc is not None and abs(vm) > 1:
                ratio_ns_om.append(o)
                ratio_ns_v.append(vc / vm)
        if ratio_ns_om:
            ax2.plot(ratio_ns_om, ratio_ns_v, 'r-^', ms=3, lw=1.0, alpha=0.7, label='Nemoh DUOK')

    ax2.axhline(2.0, color='k', ls='--', lw=1, alpha=0.5)
    if trap_om is not None:
        ax2.axvline(trap_om, color='purple', ls=':', lw=1, alpha=0.5)
    ax2.set_title(f'Sway — {label}', fontsize=10)
    ax2.set_xlabel('\u03c9 [rad/s]')
    if col == 0:
        ax2.set_ylabel('Cat / Mono ratio')
    ax2.set_ylim(0, 6)
    ax2.legend(fontsize=7, loc='best')
    ax2.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig3.savefig(f'{outdir}/drift_catmono_ratio.png', dpi=150)
print(f"  Saved {outdir}/drift_catmono_ratio.png")


# =====================================================================
# Figure 4: Nemoh DUOK per-term breakdown (catamaran, 3 headings)
# =====================================================================
print("Generating Figure 4: Nemoh per-term breakdown...")
fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10))
fig4.suptitle('Nemoh DUOK Per-Term Breakdown — Wigley Catamaran (corrected mesh)\n'
              'Top: Surge (DOF 1), Bottom: Sway (DOF 2)', fontsize=14, y=0.98)

term_names = ['T1: Velocity\u00b2', 'T2: Disp\u00d7Press', 'T3: Waterline',
              'T4: Rot\u00d7Inertia', 'T5: Trans moment', 'T6: Quad stiff']
term_colors = ['c', 'm', 'b', 'orange', 'gray', 'brown']

for col, (label, mu_pd, beta_nem) in enumerate(headings[:3]):
    for row, dof in enumerate([1, 2]):
        ax = axes4[row, col]

        for term_idx in range(1, 7):
            fname = f'{nemoh_cat_dir}/QTFM_DUOK_term_{term_idx}.dat'
            try:
                om_t, re_t = read_nemoh_diagonal(fname, dof=dof, beta_target=beta_nem)
                if np.any(np.abs(re_t) > 1e-10):
                    if dof == 1:
                        ax.plot(om_t, -re_t/NORM, color=term_colors[term_idx-1],
                                label=term_names[term_idx-1], lw=1)
                    else:
                        ax.plot(om_t, re_t/NORM, color=term_colors[term_idx-1],
                                label=term_names[term_idx-1], lw=1)
            except FileNotFoundError:
                pass

        om_total, re_total = read_nemoh_diagonal(f'{nemoh_cat_dir}/QTFM_DUOK.dat',
                                                  dof=dof, beta_target=beta_nem)
        if dof == 1:
            ax.plot(om_total, -re_total/NORM, 'r-', lw=2, label='DUOK Total')
        else:
            ax.plot(om_total, re_total/NORM, 'r-', lw=2, label='DUOK Total')

        dof_name = 'Surge' if dof == 1 else 'Sway'
        ax.set_title(f'{dof_name} — {label}', fontsize=10)
        ax.set_xlabel('\u03c9 [rad/s]')
        if col == 0:
            ax.set_ylabel(r'$\sigma$')
        ax.legend(fontsize=6, ncol=2, loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', lw=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig4.savefig(f'{outdir}/drift_nemoh_terms.png', dpi=150)
print(f"  Saved {outdir}/drift_nemoh_terms.png")


# =====================================================================
# Figure 5: Head seas detail — Pinkster vs DUOK, mono + cat
# =====================================================================
print("Generating Figure 5: Head seas detail...")
fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))
fig5.suptitle('Head Seas Surge — Pinkster vs Nemoh DUOK\n'
              r'$\sigma_{aw} = -F_x / (\rho g B^2/L)$', fontsize=13, y=1.02)

# Left: Monohull
ax = axes5[0]
om_mp, fxi_mp = extract_heading(mono_debug, 180.0, 'fxi')
if len(om_mp) > 0:
    ax.plot(om_mp, -fxi_mp/NORM, 'b-o', ms=4, lw=1.5, label='PDStrip Pinkster')
om_nm, re_nm = read_nemoh_diagonal(f'{nemoh_mono_dir}/QTFM_DUOK.dat', dof=1, beta_target=np.pi)
if len(om_nm) > 0:
    ax.plot(om_nm, -re_nm/NORM, 'r-^', ms=4, lw=1.5, label='Nemoh DUOK')
ax.set_title('Monohull — Head Seas', fontsize=11)
ax.set_xlabel('\u03c9 [rad/s]')
ax.set_ylabel(r'$\sigma_{aw}$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', lw=0.5)

# Right: Catamaran
ax2 = axes5[1]
om_cp, fxi_cp = extract_cleaned_heading(cat_surge_cleaned, 180.0)
if len(om_cp) > 0:
    ax2.plot(om_cp, -fxi_cp/NORM, 'r-o', ms=4, lw=1.5, label='PDStrip Cat Pinkster')
om_nc, re_nc = read_nemoh_diagonal(f'{nemoh_cat_dir}/QTFM_DUOK.dat', dof=1, beta_target=np.pi)
if len(om_nc) > 0:
    ax2.plot(om_nc, -re_nc/NORM, 'm-^', ms=4, lw=1.5, label='Nemoh Cat DUOK')
# Also overlay mono Pinkster for reference
if len(om_mp) > 0:
    ax2.plot(om_mp, -fxi_mp/NORM, 'b--', ms=2, lw=0.8, alpha=0.5, label='PDStrip Mono Pinkster')
ax2.set_title('Catamaran — Head Seas', fontsize=11)
ax2.set_xlabel('\u03c9 [rad/s]')
ax2.set_ylabel(r'$\sigma_{aw}$')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='k', lw=0.5)

plt.tight_layout()
fig5.savefig(f'{outdir}/drift_headseas_detail.png', dpi=150)
print(f"  Saved {outdir}/drift_headseas_detail.png")


# =====================================================================
# Figure 6: Monohull comparison — PDStrip Pinkster vs Nemoh DUOK
# =====================================================================
print("Generating Figure 6: Monohull Pinkster vs Nemoh...")
fig6, axes6 = plt.subplots(2, 4, figsize=(22, 10))
fig6.suptitle('Monohull — PDStrip Pinkster vs Nemoh DUOK\n'
              r'Surge: $\sigma_{aw}=-F_x/(\rho g B^2/L)$, Sway: $\sigma=F_y/(\rho g B^2/L)$',
              fontsize=14, y=0.98)

for col, (label, mu_pd, beta_nem) in enumerate(headings):
    # Surge
    ax = axes6[0, col]
    om_mp, fxi_mp = extract_heading(mono_debug, mu_pd, 'fxi')
    if len(om_mp) > 0:
        ax.plot(om_mp, -fxi_mp/NORM, 'b-o', ms=3, lw=1.5, label='PDStrip Pinkster')
    om_nm, re_nm = read_nemoh_diagonal(f'{nemoh_mono_dir}/QTFM_DUOK.dat', dof=1, beta_target=beta_nem)
    if len(om_nm) > 0:
        ax.plot(om_nm, -re_nm/NORM, 'r-^', ms=3, lw=1.5, label='Nemoh DUOK')

    ax.set_title(f'Surge — {label}', fontsize=10)
    ax.set_xlabel('\u03c9 [rad/s]')
    if col == 0:
        ax.set_ylabel(r'$\sigma_{aw}$')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5)

    # Sway — negate PDStrip feta (y-starboard -> y-port)
    ax2 = axes6[1, col]
    om_ms, feta_ms = extract_heading(mono_debug, mu_pd, 'feta')
    if len(om_ms) > 0:
        ax2.plot(om_ms, -feta_ms/NORM, 'b-o', ms=3, lw=1.5, label='PDStrip Pinkster')
    om_ns, re_ns = read_nemoh_diagonal(f'{nemoh_mono_dir}/QTFM_DUOK.dat', dof=2, beta_target=beta_nem)
    if len(om_ns) > 0:
        ax2.plot(om_ns, re_ns/NORM, 'r-^', ms=3, lw=1.5, label='Nemoh DUOK')

    ax2.set_title(f'Sway — {label}', fontsize=10)
    ax2.set_xlabel('\u03c9 [rad/s]')
    if col == 0:
        ax2.set_ylabel(r'$\sigma_{sway}$')
    ax2.legend(fontsize=7, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', lw=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig6.savefig(f'{outdir}/drift_mono_comparison.png', dpi=150)
print(f"  Saved {outdir}/drift_mono_comparison.png")


# =====================================================================
# Print summary table for head seas
# =====================================================================
print("\n" + "="*80)
print("HEAD SEAS SURGE COMPARISON TABLE (Pinkster vs DUOK)")
print("="*80)
print(f"{'omega':>8}  {'PD mono':>10}  {'PD cat':>10}  "
      f"{'Nem mono':>10}  {'Nem cat':>10}  {'cat/mono PD':>12}  {'cat/mono Nem':>12}")
print("-"*80)

om_mp, fxi_mp = extract_heading(mono_debug, 180.0, 'fxi')
om_cp, fxi_cp = extract_cleaned_heading(cat_surge_cleaned, 180.0)
om_nm, re_nm = read_nemoh_diagonal(f'{nemoh_mono_dir}/QTFM_DUOK.dat', dof=1, beta_target=np.pi)
om_nc, re_nc = read_nemoh_diagonal(f'{nemoh_cat_dir}/QTFM_DUOK.dat', dof=1, beta_target=np.pi)

mp_dict = {rk(o): -v/NORM for o, v in zip(om_mp, fxi_mp)} if len(om_mp)>0 else {}
cp_dict = {rk(o): -v/NORM for o, v in zip(om_cp, fxi_cp)} if len(om_cp)>0 else {}
nm_dict = {rk(o): -v/NORM for o, v in zip(om_nm, re_nm)} if len(om_nm)>0 else {}
nc_dict = {rk(o): -v/NORM for o, v in zip(om_nc, re_nc)} if len(om_nc)>0 else {}

all_omegas = sorted(set(list(mp_dict.keys()) + list(nm_dict.keys())))
for w in all_omegas:
    pdm = mp_dict.get(w)
    pdc = cp_dict.get(w)
    nem_m = nm_dict.get(w)
    nem_c = nc_dict.get(w)

    pdm_s = f"{pdm:10.3f}" if pdm is not None else f"{'---':>10}"
    pdc_s = f"{pdc:10.3f}" if pdc is not None else f"{'---':>10}"
    nem_m_s = f"{nem_m:10.3f}" if nem_m is not None else f"{'---':>10}"
    nem_c_s = f"{nem_c:10.3f}" if nem_c is not None else f"{'---':>10}"

    ratio_pd = ""
    if pdm is not None and pdc is not None and abs(pdm) > 0.01:
        ratio_pd = f"{pdc/pdm:12.2f}"
    else:
        ratio_pd = f"{'---':>12}"

    ratio_nem = ""
    if nem_m is not None and nem_c is not None and abs(nem_m) > 0.01:
        ratio_nem = f"{nem_c/nem_m:12.2f}"
    else:
        ratio_nem = f"{'---':>12}"

    print(f"{w:8.3f}  {pdm_s}  {pdc_s}  {nem_m_s}  {nem_c_s}  {ratio_pd}  {ratio_nem}")

print(f"\nNormalization: NORM = rho*g*B^2/L = {NORM:.2f} N/m^2")
print(f"Positive sigma_aw = added resistance (physically correct for head seas)")


# =====================================================================
# Helper functions for seaway integration
# =====================================================================

def bretschneider_spectrum(omega, hs, tp):
    """Bretschneider (Pierson-Moskowitz family) wave spectrum.
    S(w) = (5/16) * Hs^2 * wp^4 / w^5 * exp(-5/4 * (wp/w)^4)
    where wp = 2*pi/Tp.
    Returns S(w) in m^2 s/rad.
    """
    wp = 2.0 * np.pi / tp
    S = np.zeros_like(omega)
    mask = omega > 0
    S[mask] = (5.0/16.0) * hs**2 * wp**4 / omega[mask]**5 * np.exp(-1.25 * (wp / omega[mask])**4)
    return S


def compute_seaway_drift(omega_arr, drift_grid, heading_rad, theta0_rad, hs, tp, n_cos=2):
    """
    Compute mean drift force in a seaway for a given main direction theta0.

    F_drift(theta0) = integral over omega and theta of:
        S(omega) * D(theta; theta0) * f(omega, theta) * dtheta * domega

    where D(theta; theta0) = cos^n(theta-theta0) / integral(cos^n) is the
    normalized directional spreading function, and f(omega, theta) is the
    drift force per unit wave amplitude squared (in physical units, N/m^2).

    Parameters:
        omega_arr: array of frequencies [rad/s]
        drift_grid: dict {heading_rad: drift_array} — drift force / NORM at each heading
        heading_rad: sorted array of headings in radians
        theta0_rad: main wave direction in radians
        hs: significant wave height [m]
        tp: peak period [s]
        n_cos: exponent for cos^n spreading (default 2)

    Returns:
        F_drift: mean drift force [N] (= sigma * NORM * integral)
    """
    n_om = len(omega_arr)
    dtheta = np.radians(10)  # 10-degree spacing

    # Compute spreading weights (normalized)
    weights = {}
    weight_sum = 0.0
    for theta in heading_rad:
        diff = theta - theta0_rad
        diff = (diff + np.pi) % (2*np.pi) - np.pi
        w = max(np.cos(diff), 0.0) ** n_cos
        weights[theta] = w
        if w > 1e-12:
            weight_sum += w * dtheta
    # Normalize
    if weight_sum > 1e-12:
        for theta in weights:
            weights[theta] /= weight_sum

    # Compute spectrum
    S = bretschneider_spectrum(omega_arr, hs, tp)

    # Integrate: sum over omega of S(w) * [sum over theta of D(theta)*f(w,theta)*dtheta] * domega
    F_total = 0.0
    for i in range(n_om):
        # Frequency spacing (trapezoidal-ish)
        if i == 0:
            domega = (omega_arr[1] - omega_arr[0]) if n_om > 1 else 1.0
        elif i == n_om - 1:
            domega = (omega_arr[-1] - omega_arr[-2]) if n_om > 1 else 1.0
        else:
            domega = (omega_arr[i+1] - omega_arr[i-1]) / 2.0

        # Directional integral
        dir_sum = 0.0
        for theta in heading_rad:
            if weights[theta] > 1e-15 and theta in drift_grid and i < len(drift_grid[theta]):
                dir_sum += weights[theta] * drift_grid[theta][i] * dtheta
        F_total += S[i] * dir_sum * domega

    return F_total * NORM  # convert from sigma to force [N]


# =====================================================================
# Build full heading grids for PDStrip and Nemoh catamaran
# All keys normalized to [0, 2*pi).  Both grids mirrored to full circle.
# Symmetry: surge(360-beta) = surge(beta), sway(360-beta) = -sway(beta)
# =====================================================================

def normalize_rad(r):
    """Normalize angle to [0, 2*pi)."""
    return round(r % (2.0 * np.pi), 6)


# --- PDStrip: read raw data, normalize keys to [0, 2pi) ---
pd_headings_deg = np.arange(-90, 270, 10)
pd_surge_grid = {}
pd_sway_grid = {}

om_ref, _ = extract_cleaned_heading(cat_surge_cleaned, 180.0)
om_ref = om_ref[om_ref <= OMEGA_MAX]

for mu_deg in pd_headings_deg:
    key = normalize_rad(np.radians(mu_deg))
    om_s, fxi_s = extract_cleaned_heading(cat_surge_cleaned, mu_deg)
    om_sw, feta_sw = extract_cleaned_heading(cat_sway_cleaned, mu_deg)
    if len(om_s) > 0:
        mask = om_s <= OMEGA_MAX
        pd_surge_grid[key] = -fxi_s[mask] / NORM
    if len(om_sw) > 0:
        mask = om_sw <= OMEGA_MAX
        pd_sway_grid[key] = -feta_sw[mask] / NORM

# PDStrip already covers full circle [-90°,260°] = [0°,360°) after normalization.
# Port-starboard mirror produces no new headings.

print(f"  PDStrip grid: {len(pd_surge_grid)} headings")


# --- Nemoh: read [90°, 270°], normalize keys, then mirror to full circle ---
nemoh_surge_grid = {}
nemoh_sway_grid = {}

for beta_deg in np.arange(90, 280, 10):
    beta_rad = np.radians(beta_deg)
    key = normalize_rad(beta_rad)
    om_n, re_n = read_nemoh_diagonal(f'{nemoh_cat_dir}/QTFM_DUOK.dat', dof=1, beta_target=beta_rad)
    om_ns, re_ns = read_nemoh_diagonal(f'{nemoh_cat_dir}/QTFM_DUOK.dat', dof=2, beta_target=beta_rad)
    if len(om_n) > 0:
        mask = om_n <= OMEGA_MAX
        nemoh_surge_grid[key] = -re_n[mask] / NORM
    if len(om_ns) > 0:
        mask = om_ns <= OMEGA_MAX
        nemoh_sway_grid[key] = re_ns[mask] / NORM

# Mirror Nemoh using FORE-AFT symmetry (Wigley hull specific!).
# For a fore-aft symmetric hull at zero speed:
#   sigma_surge(pi - beta) = -sigma_surge(beta)   (x-force reverses)
#   sigma_sway(pi - beta)  =  sigma_sway(beta)    (y-force same)
# This maps [90°,270°] -> [0°,360°) by creating new entries at (180°-beta).
# Port-starboard mirror (360°-beta) maps within [90°,270°], producing nothing new.
for key, vals in list(nemoh_surge_grid.items()):
    mirror = normalize_rad(np.pi - key)       # 180° - beta
    if mirror not in nemoh_surge_grid:
        nemoh_surge_grid[mirror] = -vals      # surge flips sign
for key, vals in list(nemoh_sway_grid.items()):
    mirror = normalize_rad(np.pi - key)
    if mirror not in nemoh_sway_grid:
        nemoh_sway_grid[mirror] = vals        # sway keeps sign

om_nemoh_ref, _ = read_nemoh_diagonal(f'{nemoh_cat_dir}/QTFM_DUOK.dat', dof=1, beta_target=np.pi)
om_nemoh_ref = om_nemoh_ref[om_nemoh_ref <= OMEGA_MAX]

print(f"  Nemoh grid: {len(nemoh_surge_grid)} headings after mirroring")

pd_heading_rad = np.array(sorted(pd_surge_grid.keys()))
nemoh_heading_rad = np.array(sorted(nemoh_surge_grid.keys()))


# =====================================================================
# Figure 7: Standard x-y plot — mean drift force in seaway vs main direction
# =====================================================================
print("\nGenerating Figure 7: Seaway drift force vs wave direction...")

sea_states = [
    ("Hs=2.5m, Tp=7s", 2.5, 7.0),
    ("Hs=4.0m, Tp=9.5s", 4.0, 9.5),
]

# Sweep main directions every 10 degrees (0–360)
theta0_deg_arr = np.arange(0, 370, 10)  # include 360 to close the loop
theta0_rad_arr = np.radians(theta0_deg_arr)

fig7, axes7 = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
fig7.suptitle('Mean Drift Force in Short-Crested Seaway\n'
              'Bretschneider spectrum, cos² spreading — Wigley Catamaran\n'
              '0° = following seas, 90° = stbd beam, 180° = head seas',
              fontsize=13, y=1.02)

for row, (ss_label, hs, tp) in enumerate(sea_states):
    for col, (dof_label, pd_grid, nem_grid) in enumerate([
        ("Surge (added resistance)", pd_surge_grid, nemoh_surge_grid),
        ("Sway (lateral drift, positive = port)", pd_sway_grid, nemoh_sway_grid),
    ]):
        ax = axes7[row, col]

        pd_vals = []
        nem_vals = []
        for theta0 in theta0_rad_arr:
            f_pd = compute_seaway_drift(om_ref, pd_grid, pd_heading_rad, theta0, hs, tp, n_cos=2)
            f_nem = compute_seaway_drift(om_nemoh_ref, nem_grid, nemoh_heading_rad, theta0, hs, tp, n_cos=2)
            pd_vals.append(f_pd)
            nem_vals.append(f_nem)

        pd_vals = np.array(pd_vals) / 1000  # kN
        nem_vals = np.array(nem_vals) / 1000  # kN

        ax.plot(theta0_deg_arr, pd_vals, 'r-o', ms=3, lw=1.5, label='PDStrip Pinkster')
        ax.plot(theta0_deg_arr, nem_vals, 'k--^', ms=3, lw=1.2, label='Nemoh DUOK')
        ax.axhline(0, color='gray', lw=0.5)

        # Shade following-seas region where PDStrip goes negative (if surge)
        if col == 0:
            ax.fill_between(theta0_deg_arr, 0, pd_vals, where=(pd_vals < 0),
                            color='red', alpha=0.1, label='PDStrip < 0 (thrust)')

        ax.set_title(f'{dof_label} — {ss_label}', fontsize=11)
        if row == 1:
            ax.set_xlabel('Main wave direction θ₀ [deg]')
        ax.set_ylabel('Force [kN]')
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 30))
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        # Print peak values
        pd_max_idx = np.argmax(np.abs(pd_vals))
        nem_max_idx = np.argmax(np.abs(nem_vals))
        print(f"  {dof_label[:5]} {ss_label}: PDStrip max={pd_vals[pd_max_idx]:.1f} kN at {theta0_deg_arr[pd_max_idx]}°, "
              f"Nemoh max={nem_vals[nem_max_idx]:.1f} kN at {theta0_deg_arr[nem_max_idx]}°")

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig7.savefig(f'{outdir}/drift_seaway_vs_direction.png', dpi=150, bbox_inches='tight')
print(f"  Saved {outdir}/drift_seaway_vs_direction.png")


# =====================================================================
# Print seaway drift summary table
# =====================================================================
print("\n" + "="*90)
print("MEAN DRIFT FORCE IN SEAWAY [kN] — Bretschneider spectrum, cos² spreading")
print("="*90)

# Use 0–350 range for table (skip 360 which duplicates 0)
table_deg = np.arange(0, 360, 30)
table_rad = np.radians(table_deg)
for ss_label, hs, tp in sea_states:
    print(f"\n  {ss_label}")
    print(f"  {'θ₀ [deg]':>10}  {'PD surge':>10}  {'Nem surge':>10}  {'ratio':>8}  "
          f"{'PD sway':>10}  {'Nem sway':>10}  {'ratio':>8}")
    print("  " + "-"*78)
    for i, theta0 in enumerate(table_rad):
        f_pd_x = compute_seaway_drift(om_ref, pd_surge_grid, pd_heading_rad, theta0, hs, tp)
        f_nem_x = compute_seaway_drift(om_nemoh_ref, nemoh_surge_grid, nemoh_heading_rad, theta0, hs, tp)
        f_pd_y = compute_seaway_drift(om_ref, pd_sway_grid, pd_heading_rad, theta0, hs, tp)
        f_nem_y = compute_seaway_drift(om_nemoh_ref, nemoh_sway_grid, nemoh_heading_rad, theta0, hs, tp)

        r_x = f"{f_pd_x/f_nem_x:.2f}" if abs(f_nem_x) > 10 else "---"
        r_y = f"{f_pd_y/f_nem_y:.2f}" if abs(f_nem_y) > 10 else "---"
        print(f"  {table_deg[i]:10.0f}  {f_pd_x/1000:10.2f}  {f_nem_x/1000:10.2f}  {r_x:>8}  "
              f"{f_pd_y/1000:10.2f}  {f_nem_y/1000:10.2f}  {r_y:>8}")

print("\nAll plots saved to:", outdir)
