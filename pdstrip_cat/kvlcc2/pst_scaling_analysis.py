#!/usr/bin/env python3
"""
Investigate: what pst scaling factor would make Pinkster match SWAN1?
Uses the DRIFT_NOPST data (WL without pst) and DRIFT_TOTAL (WL with full pst)
to interpolate what fraction of pst gives the best match.
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

Lpp = 328.2
B = 58.0
rho = 1025.0
g = 9.81
norm = rho * g * B**2 / Lpp

# SWAN1 data
seo_180 = [
    (0.30, 0.0), (0.35, 0.1), (0.40, 0.3), (0.45, 0.6),
    (0.50, 1.0), (0.55, 1.3), (0.60, 1.5), (0.65, 1.3),
    (0.70, 0.9), (0.75, 0.5), (0.80, 0.3), (0.85, 0.5),
    (0.90, 1.0), (0.95, 1.6), (1.00, 2.1), (1.05, 2.4),
    (1.10, 2.5), (1.15, 2.3), (1.20, 1.8), (1.25, 1.3),
    (1.30, 0.8), (1.35, 0.5), (1.40, 0.3), (1.50, 0.1),
    (1.60, 0.0), (1.80, 0.0), (2.00, 0.0), (2.50, 0.0),
]
seo_interp = interp1d([s[0] for s in seo_180], [s[1] for s in seo_180],
                       bounds_error=False, fill_value=np.nan)

# Parse debug file
starts = []
totals = []
nopsts = []

with open('/home/blofro/src/pdstrip_test/kvlcc2/debug.out') as f:
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
                    'fxi_WL': float(m.group(3)),
                    'fxi_vel': float(m.group(5)),
                    'fxi_rot': float(m.group(6)),
                })
        elif 'DRIFT_NOPST' in line:
            m = re.search(r'fxi_WL_nopst=\s*([-\d.Ee+]+)', line)
            if m:
                nopsts.append(float(m.group(1)))

omegas = sorted(set(s[0] for s in starts))
mus = sorted(set(s[1] for s in starts))
n_headings = len(mus)
n_omegas = len(omegas)
n_speeds = len(starts) // (n_omegas * n_headings)

mu_idx = None
for i, m in enumerate(mus):
    if abs(m - 180.0) < 1.0:
        mu_idx = i
        break

speed_idx = 2

# Extract head seas data
records = []
for iom in range(n_omegas):
    idx = iom * (n_speeds * n_headings) + speed_idx * n_headings + mu_idx
    if idx >= len(starts):
        continue
    omega = starts[idx][0]
    mu = starts[idx][1]
    if abs(mu - 180.0) > 1.0:
        continue
    wavelength = 2 * np.pi * g / omega**2
    lam_L = wavelength / Lpp
    
    # fxi_WL with full pst, fxi_WL_nopst without pst
    # Assume: fxi_WL(alpha) = fxi_WL_nopst + alpha * (fxi_WL_full - fxi_WL_nopst)
    # i.e., linear interpolation between no-pst and full-pst in the WL integral
    # This is NOT exact (pst enters quadratically through |p|^2) but let's see
    fxi_WL_full = totals[idx]['fxi_WL']
    fxi_WL_nopst = nopsts[idx] if idx < len(nopsts) else fxi_WL_full
    fxi_vel = totals[idx]['fxi_vel']
    fxi_rot = totals[idx]['fxi_rot']
    
    records.append({
        'lam_L': lam_L,
        'fxi_WL_full': fxi_WL_full,
        'fxi_WL_nopst': fxi_WL_nopst,
        'fxi_vel': fxi_vel,
        'fxi_rot': fxi_rot,
    })

records.sort(key=lambda x: x['lam_L'])

# For each alpha (pst scaling), compute sigma and RMS error vs SWAN1
def compute_sigma_for_alpha(alpha, records):
    """Compute drift force for a given pst scaling factor.
    
    Since pst enters the pressure as p = p_nopst + alpha*p_pst, and the WL integral
    is proportional to |p|^2, we can't simply linearly interpolate.
    But we can write: |p|^2 = |p_nopst + alpha*p_pst|^2
    = |p_nopst|^2 + 2*alpha*Re(p_nopst*conj(p_pst)) + alpha^2*|p_pst|^2
    
    We know:
    - WL(alpha=0) = integral of |p_nopst|^2 * dy/dx  => fxi_WL_nopst
    - WL(alpha=1) = integral of |p_nopst + p_pst|^2 * dy/dx => fxi_WL_full
    
    So: WL(1) = WL(0) + 2*cross + pst_only
    where cross = integral of Re(p_nopst*conj(p_pst)) * dy/dx
    and pst_only = integral of |p_pst|^2 * dy/dx
    
    WL(alpha) = WL(0) + 2*alpha*cross + alpha^2*pst_only
    
    We have 2 equations (alpha=0 and alpha=1) but 3 unknowns.
    We need to estimate pst_only separately, or just use linear interpolation
    as an approximation.
    
    Actually, let's just do WL(alpha) = (1-alpha)*WL(0) + alpha*WL(1)
    This is a linear approximation. Not exact but gives the trend.
    """
    sigmas = []
    lam_Ls = []
    for r in records:
        fxi_WL = (1 - alpha) * r['fxi_WL_nopst'] + alpha * r['fxi_WL_full']
        fxi = fxi_WL + r['fxi_vel'] + r['fxi_rot']
        sigma = -fxi / norm
        sigmas.append(sigma)
        lam_Ls.append(r['lam_L'])
    return lam_Ls, sigmas


def rms_error(alpha):
    lam_Ls, sigmas = compute_sigma_for_alpha(alpha, records)
    errors = []
    for l, s in zip(lam_Ls, sigmas):
        sv = seo_interp(l)
        if not np.isnan(sv) and 0.4 <= l <= 1.5:
            errors.append((s - sv)**2)
    return np.sqrt(np.mean(errors)) if errors else 1e6


# Scan alpha from 0 to 1.5
alphas = np.linspace(0, 1.5, 151)
rms_values = [rms_error(a) for a in alphas]
best_alpha = alphas[np.argmin(rms_values)]
best_rms = min(rms_values)

print(f"Best pst scaling alpha = {best_alpha:.3f} (RMS = {best_rms:.3f})")
print(f"RMS at alpha=0 (no pst):   {rms_error(0):.3f}")
print(f"RMS at alpha=1 (full pst): {rms_error(1):.3f}")
print(f"RMS at alpha=0.5:          {rms_error(0.5):.3f}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Different alpha values
ax = axes[0, 0]
for alpha, color, ls in [(0, 'orange', '--'), (0.3, 'purple', '-'), 
                          (best_alpha, 'green', '-'), (0.5, 'cyan', '-.'), (1.0, 'red', '-')]:
    lam_Ls, sigmas = compute_sigma_for_alpha(alpha, records)
    ax.plot(lam_Ls, sigmas, color=color, linestyle=ls, linewidth=1.5, 
            label=f'α={alpha:.2f}' + (' (best)' if alpha == best_alpha else ''))
ax.plot([s[0] for s in seo_180], [s[1] for s in seo_180], 'b-o', 
        markersize=4, linewidth=2, label='SWAN1', zorder=10)
ax.set_xlabel(r'$\lambda/L$')
ax.set_ylabel(r'$\sigma_{aw}$')
ax.set_title('Head seas: Effect of pst scaling on drift force')
ax.set_xlim(0.3, 2.0)
ax.set_ylim(-4, 8)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)

# Panel 2: RMS error vs alpha
ax = axes[0, 1]
ax.plot(alphas, rms_values, 'k-', linewidth=2)
ax.axvline(x=best_alpha, color='g', linestyle='--', label=f'Best α={best_alpha:.3f}')
ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='Full pst (α=1)')
ax.axvline(x=0.0, color='orange', linestyle='--', alpha=0.5, label='No pst (α=0)')
ax.set_xlabel('pst scaling factor α')
ax.set_ylabel('RMS error vs SWAN1')
ax.set_title(f'Optimal pst scaling: α={best_alpha:.3f}')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# Panel 3: Best alpha vs SWAN1 at head seas
ax = axes[1, 0]
lam_Ls, sigmas_best = compute_sigma_for_alpha(best_alpha, records)
_, sigmas_full = compute_sigma_for_alpha(1.0, records)
ax.plot(lam_Ls, sigmas_full, 'r-^', markersize=3, linewidth=1.5, label='Pinkster (full pst)')
ax.plot(lam_Ls, sigmas_best, 'g-s', markersize=4, linewidth=2, 
        label=f'Pinkster (α={best_alpha:.2f})')
ax.plot([s[0] for s in seo_180], [s[1] for s in seo_180], 'b-o',
        markersize=4, linewidth=2, label='SWAN1', zorder=10)
ax.set_xlabel(r'$\lambda/L$')
ax.set_ylabel(r'$\sigma_{aw}$')
ax.set_title(f'Head seas: Optimally-scaled pst (α={best_alpha:.2f}) vs SWAN1')
ax.set_xlim(0.3, 2.0)
ax.set_ylim(-2, 7)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# Panel 4: Component magnitudes
ax = axes[1, 1]
lam = [r['lam_L'] for r in records]
ax.plot(lam, [-r['fxi_WL_full']/norm for r in records], 'r-', linewidth=1.5, label='σ_WL (full pst)')
ax.plot(lam, [-r['fxi_WL_nopst']/norm for r in records], 'orange', linewidth=1.5, 
        linestyle='--', label='σ_WL (no pst)')
ax.plot(lam, [-r['fxi_vel']/norm for r in records], 'm-', linewidth=1.5, label='σ_vel')
ax.plot(lam, [-r['fxi_rot']/norm for r in records], 'c-', linewidth=1.5, label='σ_rot')
ax.plot([s[0] for s in seo_180], [s[1] for s in seo_180], 'b-o',
        markersize=4, linewidth=2, label='SWAN1', zorder=10)
ax.set_xlabel(r'$\lambda/L$')
ax.set_ylabel(r'$\sigma_{aw}$')
ax.set_title('Component comparison: WL (with/without pst), vel, rot')
ax.set_xlim(0.3, 2.0)
ax.set_ylim(-4, 12)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)

plt.suptitle('Pinkster drift force: pst scaling analysis (head seas, KVLCC2)\n'
             'V=3 m/s, 15% roll damping', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/pst_scaling_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved: pst_scaling_analysis.png")

# Also print detailed table for best alpha
print(f"\n{'='*100}")
print(f"Detailed comparison at alpha={best_alpha:.3f}")
print(f"{'='*100}")
print(f"{'lam/L':>7} | {'full_pst':>10} | {'best_alpha':>10} | {'SWAN1':>7} | {'ratio_full':>10} | {'ratio_best':>10}")
print(f"{'-'*100}")
for l, sf, sb in zip(lam_Ls, sigmas_full, sigmas_best):
    sv = seo_interp(l)
    if 0.3 <= l <= 2.0:
        rf = sf/sv if not np.isnan(sv) and abs(sv) > 0.05 else float('nan')
        rb = sb/sv if not np.isnan(sv) and abs(sv) > 0.05 else float('nan')
        svs = f"{sv:7.2f}" if not np.isnan(sv) else "    ---"
        rfs = f"{rf:10.2f}" if not np.isnan(rf) else "       ---"
        rbs = f"{rb:10.2f}" if not np.isnan(rb) else "       ---"
        print(f"{l:7.3f} | {sf:10.3f} | {sb:10.3f} | {svs} | {rfs} | {rbs}")
