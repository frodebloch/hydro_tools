#!/usr/bin/env python3
"""
Generate validation plots: pdstrip vs Capytaine for semi-circular barge.

Produces a multi-panel figure comparing added mass and damping coefficients
for monohull and catamaran configurations at hulld/R = 2, 3, 5.

Output: validation/validation_plots.png (and .pdf)
"""

import numpy as np
import sys
import os

sys.path.insert(0, '/home/blofro/src/pdstrip_test/validation')
from parse_sectionresults import parse_sectionresults_v2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# Parameters
# ============================================================
R = 1.0
L = 20.0
rho = 1025.0
g = 9.81
base = '/home/blofro/src/pdstrip_test/validation'

# Capytaine frequencies
nu_capy = np.array([
    0.01, 0.02, 0.04, 0.06, 0.08,
    0.10, 0.15, 0.20, 0.25, 0.31,
    0.40, 0.50, 0.63, 0.80, 1.00,
    1.25, 1.55, 1.90, 2.40, 3.00,
    3.60, 4.50, 5.00,
])
omega_capy = np.sqrt(nu_capy * g / R)
wave_dirs_capy = np.array([0, np.pi/2, np.pi])


# ============================================================
# Data extraction
# ============================================================
def extract_pdstrip(run_dir, section_idx=2):
    """Extract added mass and damping from pdstrip sectionresults."""
    sr_file = os.path.join(run_dir, 'sectionresults')
    data = parse_sectionresults_v2(sr_file, section_idx=section_idx)
    if data is None:
        raise RuntimeError(f"Failed to parse {sr_file}")

    nfre = data['nfre']
    omega = data['omega']
    nu = omega**2 * R / g

    a22 = np.zeros(nfre)
    b22 = np.zeros(nfre)
    a33 = np.zeros(nfre)
    b33 = np.zeros(nfre)

    for i in range(nfre):
        am = data['addedm'][i]
        w = omega[i]
        a22[i] = am[0, 0].real / w**2
        b22[i] = -am[0, 0].imag / w
        a33[i] = am[1, 1].real / w**2
        b33[i] = -am[1, 1].imag / w

    return {'omega': omega, 'nu': nu, 'a22': a22, 'b22': b22, 'a33': a33, 'b33': b33}


def run_capytaine(label, omega_values, wave_directions):
    """Run Capytaine solver and extract results."""
    import capytaine as cpt
    import logging
    cpt.set_logging(logging.WARNING)

    mesh_res = (10, 40, 50)
    solver = cpt.BEMSolver()

    if label == 'mono':
        mesh_full = cpt.mesh_horizontal_cylinder(
            length=L, radius=R, center=(0, 0, 0),
            resolution=mesh_res, name='barge')
        hull_mesh = mesh_full.immersed_part()
        lid = hull_mesh.generate_lid(z=-0.01)
        body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name='barge')
        body.add_all_rigid_body_dofs()
    else:
        hulld = int(label[3:]) * R
        bodies = []
        for y_off, nm in [(-hulld, 'stb_hull'), (hulld, 'port_hull')]:
            mf = cpt.mesh_horizontal_cylinder(
                length=L, radius=R, center=(0, y_off, 0),
                resolution=mesh_res, name=nm)
            hm = mf.immersed_part()
            ld = hm.generate_lid(z=-0.01)
            b = cpt.FloatingBody(mesh=hm, lid_mesh=ld, name=nm)
            b.add_all_rigid_body_dofs()
            bodies.append(b)
        body = bodies[0] + bodies[1]

    problems = []
    for omega in omega_values:
        for dof in body.dofs:
            problems.append(cpt.RadiationProblem(
                body=body, radiating_dof=dof, omega=omega,
                water_depth=np.inf, rho=rho, g=g))
        for beta in wave_directions:
            problems.append(cpt.DiffractionProblem(
                body=body, wave_direction=beta, omega=omega,
                water_depth=np.inf, rho=rho, g=g))

    print(f"  Solving {len(problems)} problems for {label}...")
    results = solver.solve_all(problems, progress_bar=True)
    ds = cpt.assemble_dataset(results)

    # Extract results
    nfre = len(omega_values)
    nu = omega_values**2 * R / g
    a22 = np.zeros(nfre)
    b22 = np.zeros(nfre)
    a33 = np.zeros(nfre)
    b33 = np.zeros(nfre)

    if label == 'mono':
        for i, w in enumerate(omega_values):
            a22[i] = float(ds['added_mass'].sel(radiating_dof='Sway', influenced_dof='Sway').sel(omega=w, method='nearest'))
            b22[i] = float(ds['radiation_damping'].sel(radiating_dof='Sway', influenced_dof='Sway').sel(omega=w, method='nearest'))
            a33[i] = float(ds['added_mass'].sel(radiating_dof='Heave', influenced_dof='Heave').sel(omega=w, method='nearest'))
            b33[i] = float(ds['radiation_damping'].sel(radiating_dof='Heave', influenced_dof='Heave').sel(omega=w, method='nearest'))
    else:
        for i, w in enumerate(omega_values):
            for rp in ['stb_hull', 'port_hull']:
                for ip in ['stb_hull', 'port_hull']:
                    a22[i] += float(ds['added_mass'].sel(
                        radiating_dof=f'{rp}__Sway', influenced_dof=f'{ip}__Sway').sel(omega=w, method='nearest'))
                    b22[i] += float(ds['radiation_damping'].sel(
                        radiating_dof=f'{rp}__Sway', influenced_dof=f'{ip}__Sway').sel(omega=w, method='nearest'))
                    a33[i] += float(ds['added_mass'].sel(
                        radiating_dof=f'{rp}__Heave', influenced_dof=f'{ip}__Heave').sel(omega=w, method='nearest'))
                    b33[i] += float(ds['radiation_damping'].sel(
                        radiating_dof=f'{rp}__Heave', influenced_dof=f'{ip}__Heave').sel(omega=w, method='nearest'))

    return {'omega': omega_values, 'nu': nu, 'a22': a22, 'b22': b22, 'a33': a33, 'b33': b33}


# ============================================================
# Gather all data
# ============================================================
print("Extracting pdstrip results...")
pd_data = {}
for label, run_dir in [('mono', 'run_mono'), ('cat2', 'run_cat2'), ('cat3', 'run_cat3'), ('cat5', 'run_cat5')]:
    pd_data[label] = extract_pdstrip(os.path.join(base, run_dir))

print("Running Capytaine solves...")
cy_data = {}
for label in ['mono', 'cat2', 'cat3', 'cat5']:
    cy_data[label] = run_capytaine(label, omega_capy, wave_dirs_capy)

# Normalise: pdstrip values are per-section (2D), Capytaine are total 3D -> divide by L
for label in cy_data:
    for key in ['a22', 'b22', 'a33', 'b33']:
        cy_data[label][key] = cy_data[label][key] / L

# Reference value for non-dimensionalisation
rho_pi_R2 = rho * np.pi * R**2   # = 3220 kg/m  (infinite-fluid added mass of a circle)


# ============================================================
# Figure 1: Monohull + 3 catamaran panels (4 rows x 2 cols)
# ============================================================
print("Generating plots...")

configs = [
    ('mono', 'Monohull'),
    ('cat2', r'Catamaran $d/R = 2$  (gap $= 2R$)'),
    ('cat3', r'Catamaran $d/R = 3$  (gap $= 4R$)'),
    ('cat5', r'Catamaran $d/R = 5$  (gap $= 8R$)'),
]

fig, axes = plt.subplots(4, 2, figsize=(14, 20), constrained_layout=True)

nu_lim = (0, 5.2)

for irow, (label, title) in enumerate(configs):
    ax_a = axes[irow, 0]   # added mass
    ax_b = axes[irow, 1]   # damping

    pd = pd_data[label]
    cy = cy_data[label]

    # Filter pdstrip to nu < 5.2
    mask_pd = pd['nu'] < 5.2

    # --- Added mass ---
    ax_a.plot(pd['nu'][mask_pd], pd['a22'][mask_pd] / rho_pi_R2,
              '-', color='#2166ac', linewidth=1.5, label=r'$a_{22}$ pdstrip')
    ax_a.plot(cy['nu'], cy['a22'] / rho_pi_R2,
              'o', color='#2166ac', markersize=4, markerfacecolor='none', linewidth=1.2,
              label=r'$a_{22}$ Capytaine')

    ax_a.plot(pd['nu'][mask_pd], pd['a33'][mask_pd] / rho_pi_R2,
              '-', color='#b2182b', linewidth=1.5, label=r'$a_{33}$ pdstrip')
    ax_a.plot(cy['nu'], cy['a33'] / rho_pi_R2,
              's', color='#b2182b', markersize=4, markerfacecolor='none', linewidth=1.2,
              label=r'$a_{33}$ Capytaine')

    ax_a.set_xlim(nu_lim)
    ax_a.set_ylabel(r'$a_{jj}\,/\,\rho\pi R^2$', fontsize=11)
    ax_a.set_title(f'{title} — Added mass', fontsize=11, fontweight='bold')
    ax_a.axhline(0, color='grey', linewidth=0.5, linestyle='-')
    ax_a.grid(True, alpha=0.3)
    if irow == 0:
        ax_a.legend(fontsize=8, ncol=2, loc='upper right')
    # For catamaran panels, set y-limits to show detail without resonance domination
    if irow > 0:
        ax_a.set_ylim(-8, 8)

    # --- Damping ---
    ax_b.plot(pd['nu'][mask_pd], pd['b22'][mask_pd] / (rho_pi_R2 * np.sqrt(g * R)),
              '-', color='#2166ac', linewidth=1.5, label=r'$b_{22}$ pdstrip')
    ax_b.plot(cy['nu'], cy['b22'] / (rho_pi_R2 * np.sqrt(g * R)),
              'o', color='#2166ac', markersize=4, markerfacecolor='none', linewidth=1.2,
              label=r'$b_{22}$ Capytaine')

    ax_b.plot(pd['nu'][mask_pd], pd['b33'][mask_pd] / (rho_pi_R2 * np.sqrt(g * R)),
              '-', color='#b2182b', linewidth=1.5, label=r'$b_{33}$ pdstrip')
    ax_b.plot(cy['nu'], cy['b33'] / (rho_pi_R2 * np.sqrt(g * R)),
              's', color='#b2182b', markersize=4, markerfacecolor='none', linewidth=1.2,
              label=r'$b_{33}$ Capytaine')

    ax_b.set_xlim(nu_lim)
    ax_b.set_ylabel(r'$b_{jj}\,/\,(\rho\pi R^2\sqrt{gR})$', fontsize=11)
    ax_b.set_title(f'{title} — Damping', fontsize=11, fontweight='bold')
    ax_b.axhline(0, color='grey', linewidth=0.5, linestyle='-')
    ax_b.grid(True, alpha=0.3)
    if irow == 0:
        ax_b.legend(fontsize=8, ncol=2, loc='upper right')
    # For catamaran panels, set y-limits to show detail without resonance domination
    if irow > 0:
        ax_b.set_ylim(-4, 4)

# Common x-labels
for ax in axes[3, :]:
    ax.set_xlabel(r'$\nu = \omega^2 R / g$', fontsize=11)

fig.suptitle(
    'pdstrip vs Capytaine — Semi-circular barge ($R=1$ m, $L=20$ m)\n'
    'Lines: pdstrip 2D strip theory  |  Symbols: Capytaine 3D BEM / $L$',
    fontsize=13, fontweight='bold')

out1 = os.path.join(base, 'validation_plot_coefficients.png')
fig.savefig(out1, dpi=180, bbox_inches='tight')
print(f"  Saved {out1}")

out1pdf = os.path.join(base, 'validation_plot_coefficients.pdf')
fig.savefig(out1pdf, bbox_inches='tight')
print(f"  Saved {out1pdf}")
plt.close(fig)


# ============================================================
# Figure 2: Catamaran / Monohull ratios (3 catamaran spacings)
# ============================================================
fig2, axes2 = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)

cat_configs = [
    ('cat2', r'$d/R = 2$  (gap $= 2R$)', '#e66101'),
    ('cat3', r'$d/R = 3$  (gap $= 4R$)', '#5e3c99'),
    ('cat5', r'$d/R = 5$  (gap $= 8R$)', '#1b7837'),
]

for irow, (cat_label, cat_title, color) in enumerate(cat_configs):
    ax_a = axes2[irow, 0]
    ax_b = axes2[irow, 1]

    pd_m = pd_data['mono']
    pd_c = pd_data[cat_label]
    cy_m = cy_data['mono']
    cy_c = cy_data[cat_label]

    # pdstrip ratio (dense, as line): match freqs between mono and cat
    # Both have same frequency grid (52 standard frequencies)
    mask = pd_m['nu'] < 5.2
    pd_ratio_a22 = pd_c['a22'][mask] / pd_m['a22'][mask]
    pd_ratio_a33 = pd_c['a33'][mask] / pd_m['a33'][mask]
    pd_ratio_b22 = pd_c['b22'][mask] / pd_m['b22'][mask]
    pd_ratio_b33 = pd_c['b33'][mask] / pd_m['b33'][mask]

    # Capytaine ratio
    cy_ratio_a22 = cy_c['a22'] / cy_m['a22']
    cy_ratio_a33 = cy_c['a33'] / cy_m['a33']
    cy_ratio_b22 = cy_c['b22'] / cy_m['b22']
    cy_ratio_b33 = cy_c['b33'] / cy_m['b33']

    # Clip extreme ratios for readability
    clip = 8
    pd_ratio_a22 = np.clip(pd_ratio_a22, -clip, clip)
    pd_ratio_a33 = np.clip(pd_ratio_a33, -clip, clip)
    cy_ratio_a22 = np.clip(cy_ratio_a22, -clip, clip)
    cy_ratio_a33 = np.clip(cy_ratio_a33, -clip, clip)
    pd_ratio_b22 = np.clip(pd_ratio_b22, -clip, clip)
    pd_ratio_b33 = np.clip(pd_ratio_b33, -clip, clip)
    cy_ratio_b22 = np.clip(cy_ratio_b22, -clip, clip)
    cy_ratio_b33 = np.clip(cy_ratio_b33, -clip, clip)

    # --- Added mass ratios ---
    ax_a.plot(pd_m['nu'][mask], pd_ratio_a22,
              '-', color='#2166ac', linewidth=1.5, label=r'$a_{22}$ pdstrip')
    ax_a.plot(cy_m['nu'], cy_ratio_a22,
              'o', color='#2166ac', markersize=5, markerfacecolor='none', linewidth=1.2,
              label=r'$a_{22}$ Capytaine')
    ax_a.plot(pd_m['nu'][mask], pd_ratio_a33,
              '-', color='#b2182b', linewidth=1.5, label=r'$a_{33}$ pdstrip')
    ax_a.plot(cy_m['nu'], cy_ratio_a33,
              's', color='#b2182b', markersize=5, markerfacecolor='none', linewidth=1.2,
              label=r'$a_{33}$ Capytaine')

    ax_a.axhline(2.0, color='grey', linewidth=0.8, linestyle='--', label='ratio = 2')
    ax_a.set_xlim(nu_lim)
    ax_a.set_ylim(-clip - 0.5, clip + 0.5)
    ax_a.set_ylabel('catamaran / monohull', fontsize=11)
    ax_a.set_title(f'{cat_title} — Added mass ratio', fontsize=11, fontweight='bold')
    ax_a.grid(True, alpha=0.3)
    if irow == 0:
        ax_a.legend(fontsize=8, ncol=3, loc='upper right')

    # --- Damping ratios ---
    ax_b.plot(pd_m['nu'][mask], pd_ratio_b22,
              '-', color='#2166ac', linewidth=1.5, label=r'$b_{22}$ pdstrip')
    ax_b.plot(cy_m['nu'], cy_ratio_b22,
              'o', color='#2166ac', markersize=5, markerfacecolor='none', linewidth=1.2,
              label=r'$b_{22}$ Capytaine')
    ax_b.plot(pd_m['nu'][mask], pd_ratio_b33,
              '-', color='#b2182b', linewidth=1.5, label=r'$b_{33}$ pdstrip')
    ax_b.plot(cy_m['nu'], cy_ratio_b33,
              's', color='#b2182b', markersize=5, markerfacecolor='none', linewidth=1.2,
              label=r'$b_{33}$ Capytaine')

    ax_b.axhline(2.0, color='grey', linewidth=0.8, linestyle='--', label='ratio = 2')
    ax_b.set_xlim(nu_lim)
    ax_b.set_ylim(-clip - 0.5, clip + 0.5)
    ax_b.set_ylabel('catamaran / monohull', fontsize=11)
    ax_b.set_title(f'{cat_title} — Damping ratio', fontsize=11, fontweight='bold')
    ax_b.grid(True, alpha=0.3)
    if irow == 0:
        ax_b.legend(fontsize=8, ncol=3, loc='upper right')

for ax in axes2[2, :]:
    ax.set_xlabel(r'$\nu = \omega^2 R / g$', fontsize=11)

fig2.suptitle(
    'Catamaran / Monohull ratio — pdstrip vs Capytaine\n'
    'Dashed grey: ratio = 2 (non-interacting hulls)',
    fontsize=13, fontweight='bold')

out2 = os.path.join(base, 'validation_plot_ratios.png')
fig2.savefig(out2, dpi=180, bbox_inches='tight')
print(f"  Saved {out2}")

out2pdf = os.path.join(base, 'validation_plot_ratios.pdf')
fig2.savefig(out2pdf, bbox_inches='tight')
print(f"  Saved {out2pdf}")
plt.close(fig2)


# ============================================================
# Figure 3: Zoomed-in monohull comparison at mid-frequencies
# ============================================================
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

pd_m = pd_data['mono']
cy_m = cy_data['mono']

mask_zoom = (pd_m['nu'] >= 0.1) & (pd_m['nu'] <= 5.0)
mask_cy_zoom = (cy_m['nu'] >= 0.1) & (cy_m['nu'] <= 5.0)

# Added mass
ax = axes3[0]
ax.plot(pd_m['nu'][mask_zoom], pd_m['a22'][mask_zoom] / rho_pi_R2,
        '-', color='#2166ac', linewidth=2, label=r'$a_{22}$ pdstrip (2D strip)')
ax.plot(cy_m['nu'][mask_cy_zoom], cy_m['a22'][mask_cy_zoom] / rho_pi_R2,
        'o', color='#2166ac', markersize=6, markerfacecolor='none', linewidth=1.5,
        label=r'$a_{22}$ Capytaine (3D)$/L$')
ax.plot(pd_m['nu'][mask_zoom], pd_m['a33'][mask_zoom] / rho_pi_R2,
        '-', color='#b2182b', linewidth=2, label=r'$a_{33}$ pdstrip (2D strip)')
ax.plot(cy_m['nu'][mask_cy_zoom], cy_m['a33'][mask_cy_zoom] / rho_pi_R2,
        's', color='#b2182b', markersize=6, markerfacecolor='none', linewidth=1.5,
        label=r'$a_{33}$ Capytaine (3D)$/L$')
ax.set_xlim(0.1, 5.0)
ax.set_xlabel(r'$\nu = \omega^2 R / g$', fontsize=12)
ax.set_ylabel(r'$a_{jj}\,/\,\rho\pi R^2$', fontsize=12)
ax.set_title('Monohull — Added mass per unit length', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='grey', linewidth=0.5)

# Damping
ax = axes3[1]
ax.plot(pd_m['nu'][mask_zoom], pd_m['b22'][mask_zoom] / (rho_pi_R2 * np.sqrt(g * R)),
        '-', color='#2166ac', linewidth=2, label=r'$b_{22}$ pdstrip (2D strip)')
ax.plot(cy_m['nu'][mask_cy_zoom], cy_m['b22'][mask_cy_zoom] / (rho_pi_R2 * np.sqrt(g * R)),
        'o', color='#2166ac', markersize=6, markerfacecolor='none', linewidth=1.5,
        label=r'$b_{22}$ Capytaine (3D)$/L$')
ax.plot(pd_m['nu'][mask_zoom], pd_m['b33'][mask_zoom] / (rho_pi_R2 * np.sqrt(g * R)),
        '-', color='#b2182b', linewidth=2, label=r'$b_{33}$ pdstrip (2D strip)')
ax.plot(cy_m['nu'][mask_cy_zoom], cy_m['b33'][mask_cy_zoom] / (rho_pi_R2 * np.sqrt(g * R)),
        's', color='#b2182b', markersize=6, markerfacecolor='none', linewidth=1.5,
        label=r'$b_{33}$ Capytaine (3D)$/L$')
ax.set_xlim(0.1, 5.0)
ax.set_xlabel(r'$\nu = \omega^2 R / g$', fontsize=12)
ax.set_ylabel(r'$b_{jj}\,/\,(\rho\pi R^2\sqrt{gR})$', fontsize=12)
ax.set_title('Monohull — Damping per unit length', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='grey', linewidth=0.5)

fig3.suptitle(
    'Monohull validation: pdstrip 2D strip theory vs Capytaine 3D BEM\n'
    r'Semi-circular barge, $R = 1$ m, $L = 20$ m, $\nu > 0.1$ (mid-to-high frequency)',
    fontsize=13, fontweight='bold')

out3 = os.path.join(base, 'validation_plot_monohull.png')
fig3.savefig(out3, dpi=180, bbox_inches='tight')
print(f"  Saved {out3}")

out3pdf = os.path.join(base, 'validation_plot_monohull.pdf')
fig3.savefig(out3pdf, bbox_inches='tight')
print(f"  Saved {out3pdf}")
plt.close(fig3)

print("\nAll plots generated.")
