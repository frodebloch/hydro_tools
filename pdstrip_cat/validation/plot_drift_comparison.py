#!/usr/bin/env python3
"""
Plot mean drift force comparison: pdstrip vs Capytaine (near-field method).
Reads pdstrip debug.out and Capytaine nearfield_drift_comparison.npz.
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import os

# ============================================================
# Parse pdstrip drift forces from debug.out
# ============================================================
wavelengths = np.array([3, 4, 5, 6, 8, 10, 13, 17, 22, 28, 35, 45, 55, 70, 90])
# Directions in order: mu = -90, 0, +90, 180

pd_lines = []
with open('run_mono/debug.out') as f:
    for line in f:
        if 'DRIFT_TOTAL' in line:
            pd_lines.append(line.strip())

def parse_drift_line(line):
    vals = re.findall(r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?', line)
    return {
        'fxi': float(vals[0]),
        'feta': float(vals[1]),
        'fxi_WL': float(vals[2]),
        'feta_WL': float(vals[3]),
    }

# Extract beam seas (mu=+90, index 2 of 4 per wavelength)
pd_beam = []
pd_head = []
pd_follow = []
for i in range(len(wavelengths)):
    base = i * 4
    pd_beam.append(parse_drift_line(pd_lines[base + 2]))    # mu=+90
    pd_head.append(parse_drift_line(pd_lines[base + 3]))     # mu=180
    pd_follow.append(parse_drift_line(pd_lines[base + 1]))   # mu=0

# pdstrip Fy = -feta (internal y is opposite to geometric y)
pd_beam_Fy = np.array([-d['feta'] for d in pd_beam])
pd_beam_Fy_WL = np.array([-d['feta_WL'] for d in pd_beam])
pd_beam_Fy_hull = pd_beam_Fy - pd_beam_Fy_WL

pd_head_Fx = np.array([d['fxi'] for d in pd_head])
pd_follow_Fx = np.array([d['fxi'] for d in pd_follow])

# ============================================================
# Load Capytaine results
# ============================================================
data = np.load('nearfield_drift_comparison.npz', allow_pickle=True)
cap_lam = data['wavelengths']

cap_beam_Fy_wl = data['beam_Fy_wl']
cap_beam_Fy_vel = data['beam_Fy_vel']
cap_beam_Fy_rot = data['beam_Fy_rot']
cap_beam_Fy_total = data['beam_Fy_total']
cap_beam_Fy_hull = cap_beam_Fy_vel + cap_beam_Fy_rot

cap_head_Fx_total = data['head_Fx_total']
cap_head_Fx_wl = data['head_Fx_wl']

# ============================================================
# Nondimensionalize: F / (rho * g * A^2 / 2) per unit wave amplitude^2
# For mean drift, normalize by rho*g*R where R=radius for a 
# characteristic force. Or just plot in N/m^2 (per amplitude squared).
# Actually, let's just plot in N (drift force per unit wave amplitude squared).
# ============================================================

rho = 1025.0
g = 9.81

# Nondimensionalize by rho*g*R*L = 1025*9.81*1*20 = 201,105 N
F_norm = rho * g * 1.0 * 20.0  # = 201105

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Plot 1: Beam seas Fy total ---
ax = axes[0, 0]
ax.plot(wavelengths, pd_beam_Fy / F_norm, 'bo-', label='pdstrip', markersize=5)
ax.plot(cap_lam, cap_beam_Fy_total / F_norm, 'rs-', label='Capytaine 3D', markersize=5)
ax.set_xlabel('Wavelength [m]')
ax.set_ylabel(r'$\bar{F}_y / (\rho g R L)$')
ax.set_title('Beam seas: Total lateral drift force $F_y$')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
# Mark resonance region
ax.axvspan(8, 15, alpha=0.1, color='orange', label='Roll resonance region')

# --- Plot 2: Beam seas Fy decomposed ---
ax = axes[0, 1]
ax.plot(wavelengths, pd_beam_Fy_WL / F_norm, 'b^--', label='pdstrip WL', markersize=5)
ax.plot(cap_lam, cap_beam_Fy_wl / F_norm, 'r^--', label='Capytaine WL', markersize=5)
ax.plot(wavelengths, pd_beam_Fy_hull / F_norm, 'bv:', label='pdstrip hull', markersize=5)
ax.plot(cap_lam, cap_beam_Fy_hull / F_norm, 'rv:', label='Capytaine hull (vel+rot)', markersize=5)
ax.set_xlabel('Wavelength [m]')
ax.set_ylabel(r'$\bar{F}_y / (\rho g R L)$')
ax.set_title('Beam seas: Drift force decomposition')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvspan(8, 15, alpha=0.1, color='orange')

# --- Plot 3: Short-wave zoom (λ=3-6) ---
ax = axes[1, 0]
mask_short = wavelengths <= 6
mask_cap_short = cap_lam <= 6
ax.plot(wavelengths[mask_short], pd_beam_Fy[mask_short] / F_norm, 'bo-', label='pdstrip total', markersize=7)
ax.plot(cap_lam[mask_cap_short], cap_beam_Fy_total[mask_cap_short] / F_norm, 'rs-', label='Capytaine total', markersize=7)
ax.plot(wavelengths[mask_short], pd_beam_Fy_WL[mask_short] / F_norm, 'b^--', label='pdstrip WL', markersize=6, alpha=0.7)
ax.plot(cap_lam[mask_cap_short], cap_beam_Fy_wl[mask_cap_short] / F_norm, 'r^--', label='Capytaine WL', markersize=6, alpha=0.7)
ax.plot(wavelengths[mask_short], pd_beam_Fy_hull[mask_short] / F_norm, 'bv:', label='pdstrip hull', markersize=6, alpha=0.7)
ax.plot(cap_lam[mask_cap_short], cap_beam_Fy_hull[mask_cap_short] / F_norm, 'rv:', label='Capytaine hull', markersize=6, alpha=0.7)
ax.set_xlabel('Wavelength [m]')
ax.set_ylabel(r'$\bar{F}_y / (\rho g R L)$')
ax.set_title(r'Short waves ($\lambda \leq 6$m): Best comparison region')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Add ratio annotations
for i, lam in enumerate(wavelengths[mask_short]):
    j = np.where(cap_lam == lam)[0][0]
    r = pd_beam_Fy[i] / cap_beam_Fy_total[j]
    ax.annotate(f'{r:.2f}', (lam, pd_beam_Fy[i] / F_norm),
                textcoords="offset points", xytext=(5, 5), fontsize=7, color='blue')

# --- Plot 4: Head seas Fx ---
ax = axes[1, 1]
ax.plot(wavelengths, pd_head_Fx / F_norm, 'bo-', label='pdstrip', markersize=5)
ax.plot(cap_lam, cap_head_Fx_total / F_norm, 'rs-', label='Capytaine 3D', markersize=5)
ax.set_xlabel('Wavelength [m]')
ax.set_ylabel(r'$\bar{F}_x / (\rho g R L)$')
ax.set_title('Head seas: Longitudinal drift force $F_x$')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.annotate('Strip theory cannot capture\nend-cap diffraction effects\non this blunt barge',
            xy=(30, pd_head_Fx[8] / F_norm), fontsize=8, style='italic',
            xytext=(40, -0.05),
            arrowprops=dict(arrowstyle='->', color='blue'),
            color='blue')

plt.suptitle('Mean Drift Force Validation: pdstrip (2D strip) vs Capytaine (3D BEM)\n'
             'Semi-circular barge R=1m, L=20m, beam seas, deep water',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('drift_force_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: drift_force_comparison.png")

# ============================================================
# Also plot beam seas Fz (heave drift)
# ============================================================
pd_beam_Fz_total = np.array([d['fxi'] for d in pd_beam])  # fxi in beam seas ≈ Fz? 
# Actually for beam seas, fxi is the x-direction drift, which should be ~0
# Let's check what pdstrip gives for heave drift... pdstrip doesn't compute Fz drift
# (it only computes fxi=surge drift and feta=sway drift, plus mdrift=yaw moment)

fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

# Capytaine beam seas Fz
cap_beam_Fz_wl = data['beam_Fz_wl']
cap_beam_Fz_vel = data['beam_Fz_vel']
cap_beam_Fz_rot = data['beam_Fz_rot']
cap_beam_Fz_total = data['beam_Fz_total']

ax2.plot(cap_lam, cap_beam_Fz_total / F_norm, 'rs-', label='Capytaine total', markersize=5)
ax2.plot(cap_lam, cap_beam_Fz_wl / F_norm, 'r^--', label='Capytaine WL', markersize=5, alpha=0.7)
ax2.plot(cap_lam, cap_beam_Fz_vel / F_norm, 'rv:', label='Capytaine vel', markersize=5, alpha=0.7)
ax2.plot(cap_lam, cap_beam_Fz_rot / F_norm, 'rd-.', label='Capytaine rot', markersize=5, alpha=0.7)
ax2.set_xlabel('Wavelength [m]')
ax2.set_ylabel(r'$\bar{F}_z / (\rho g R L)$')
ax2.set_title('Beam seas: Vertical drift force (Capytaine only)\n'
              'pdstrip does not compute vertical drift force')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('drift_force_Fz_capytaine.png', dpi=150, bbox_inches='tight')
print("Saved: drift_force_Fz_capytaine.png")

plt.close('all')
