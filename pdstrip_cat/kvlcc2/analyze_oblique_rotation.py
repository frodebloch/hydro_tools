#!/usr/bin/env python3
"""
Extended rotation sign analysis at oblique headings.
Parse debug.out drift decomposition for V=3 m/s at beta=120°, 150° (and others).
Compare current conjg(+motion) vs flipped conjg(-motion) to see if flipping
provides the cancellation at roll resonance observed in Seo et al. Figure 8.

Also includes Seo et al. digitized data for direct overlay.
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# KVLCC2 parameters
Lpp = 328.2  # m
B = 58.0     # m
T = 20.8     # m
rho = 1025.0
g = 9.81

norm_raw = rho * g * B**2 / Lpp  # normalization for sigma_aw

# ============================================================
# Seo et al. Figure 11 digitized data (V=6 knots, H/L=1/50)
# ============================================================
seo_data = {}
seo_data[180] = {
    'swan1': [
        (0.30, 0.0), (0.35, 0.1), (0.40, 0.3), (0.45, 0.6),
        (0.50, 1.0), (0.55, 1.3), (0.60, 1.5), (0.65, 1.3),
        (0.70, 0.9), (0.75, 0.5), (0.80, 0.3), (0.85, 0.5),
        (0.90, 1.0), (0.95, 1.6), (1.00, 2.1), (1.05, 2.4),
        (1.10, 2.5), (1.15, 2.3), (1.20, 1.8), (1.25, 1.3),
        (1.30, 0.8), (1.35, 0.5), (1.40, 0.3), (1.50, 0.1),
        (1.60, 0.0), (1.80, 0.0), (2.00, 0.0), (2.50, 0.0),
    ],
}
seo_data[150] = {
    'swan1': [
        (0.30, 0.0), (0.35, 0.3), (0.40, 0.8), (0.45, 1.5),
        (0.50, 2.0), (0.55, 1.7), (0.60, 1.0), (0.65, 0.5),
        (0.70, 0.3), (0.75, 0.3), (0.80, 0.5), (0.85, 0.8),
        (0.90, 1.1), (0.95, 1.3), (1.00, 1.5), (1.05, 1.4),
        (1.10, 1.2), (1.15, 0.9), (1.20, 0.6), (1.25, 0.4),
        (1.30, 0.3), (1.40, 0.3), (1.50, 0.5), (1.55, 0.5),
        (1.60, 0.4), (1.70, 0.2), (1.80, 0.1), (2.00, 0.0),
        (2.50, 0.0),
    ],
}
seo_data[120] = {
    'swan1': [
        (0.30, 0.0), (0.35, 0.2), (0.40, 0.6), (0.45, 1.1),
        (0.50, 1.5), (0.55, 1.2), (0.60, 0.5), (0.65, 0.1),
        (0.70, -0.2), (0.75, -0.1), (0.80, 0.1), (0.85, 0.3),
        (0.90, 0.4), (0.95, 0.5), (1.00, 0.5), (1.05, 0.4),
        (1.10, 0.2), (1.15, 0.1), (1.20, 0.0), (1.30, 0.1),
        (1.40, 0.3), (1.50, 0.5), (1.55, 0.4), (1.60, 0.3),
        (1.70, 0.1), (1.80, 0.0), (2.00, 0.0), (2.50, 0.0),
    ],
}
seo_data[90] = {
    'swan1': [
        (0.30, 0.0), (0.35, 0.3), (0.40, 0.8), (0.45, 1.5),
        (0.50, 2.0), (0.55, 1.6), (0.60, 0.8), (0.65, 0.2),
        (0.70, -0.1), (0.75, -0.3), (0.80, -0.3), (0.85, -0.2),
        (0.90, -0.1), (0.95, 0.0), (1.00, 0.0), (1.10, 0.0),
        (1.20, 0.0), (1.30, 0.1), (1.40, 0.2), (1.50, 0.3),
        (1.55, 0.2), (1.60, 0.1), (1.70, 0.0), (1.80, 0.0),
        (2.00, 0.0), (2.50, 0.0),
    ],
}
seo_data[60] = {
    'swan1': [
        (0.30, 0.0), (0.35, 0.05), (0.40, 0.1), (0.45, 0.1),
        (0.50, 0.0), (0.55, -0.2), (0.60, -0.3), (0.65, -0.5),
        (0.70, -0.5), (0.75, -0.4), (0.80, -0.2), (0.85, -0.1),
        (0.90, 0.0), (0.95, 0.0), (1.00, 0.0), (1.10, -0.1),
        (1.20, -0.3), (1.30, -0.8), (1.40, -1.5), (1.50, -2.2),
        (1.60, -2.5), (1.70, -2.3), (1.80, -1.8), (1.90, -1.2),
        (2.00, -0.7), (2.20, -0.2), (2.50, 0.0),
    ],
}
seo_data[30] = {
    'swan1': [
        (0.30, 0.0), (0.35, 0.0), (0.40, 0.0), (0.50, -0.1),
        (0.55, -0.1), (0.60, -0.2), (0.65, -0.2), (0.70, -0.2),
        (0.75, -0.1), (0.80, -0.1), (0.90, 0.0), (1.00, 0.0),
        (1.10, -0.1), (1.20, -0.3), (1.30, -0.7), (1.40, -1.2),
        (1.50, -1.7), (1.60, -2.0), (1.70, -2.0), (1.80, -1.7),
        (1.90, -1.3), (2.00, -0.8), (2.20, -0.3), (2.50, 0.0),
    ],
}
seo_data[0] = {
    'swan1': [
        (0.30, 0.0), (0.40, 0.0), (0.50, -0.1), (0.60, -0.2),
        (0.70, -0.3), (0.80, -0.2), (0.90, -0.1), (1.00, -0.1),
        (1.10, -0.1), (1.20, -0.1), (1.30, -0.1), (1.50, -0.1),
        (1.80, 0.0), (2.00, 0.0), (2.50, 0.0),
    ],
}

# ============================================================
# Parse debug.out DRIFT_TOTAL lines
# ============================================================
data = []

print("Parsing debug.out (this may take a while for 149MB)...")
with open('/home/blofro/src/pdstrip_test/kvlcc2/debug.out', 'r') as f:
    current_omega = None
    current_mu = None
    for line in f:
        if line.startswith('DRIFT_START'):
            m = re.search(r'omega=\s*([\d.+-eE]+)\s+mu=\s*([\d.+-eE]+)', line)
            if m:
                current_omega = float(m.group(1))
                current_mu = float(m.group(2))
        elif 'DRIFT_TOTAL' in line:
            m = re.search(
                r'fxi=\s*([\d.+-eE]+)\s+feta=\s*([\d.+-eE]+)\s+'
                r'fxi_WL=\s*([\d.+-eE]+)\s+feta_WL=\s*([\d.+-eE]+)\s+'
                r'fxi_vel=\s*([\d.+-eE]+)\s+fxi_rot=\s*([\d.+-eE]+)',
                line
            )
            if m and current_omega is not None:
                data.append({
                    'omega': current_omega,
                    'mu': current_mu,
                    'fxi': float(m.group(1)),
                    'feta': float(m.group(2)),
                    'fxi_WL': float(m.group(3)),
                    'feta_WL': float(m.group(4)),
                    'fxi_vel': float(m.group(5)),
                    'fxi_rot': float(m.group(6)),
                })

print(f"Found {len(data)} DRIFT_TOTAL entries")

n_speeds = 8
n_headings = 36
n_wavelengths = 35

expected = n_wavelengths * n_speeds * n_headings
assert len(data) == expected, f"Expected {expected}, got {len(data)}"

# Build heading map from first wavelength, first speed block
first_block_mus = [data[i]['mu'] for i in range(n_headings)]
print(f"Headings (first block): {first_block_mus}")

# Build heading index lookup
heading_map = {}
for i, mu in enumerate(first_block_mus):
    heading_map[mu] = i

# Speed ordering: 0.0, 2.0, 3.0, 4.0, 6.0, 7.96, 9.095, 10.09
# V=3.0 m/s → iv=2
speed_names = {0: '0.0', 1: '2.0', 2: '3.0', 3: '4.0', 4: '6.0', 5: '7.96', 6: '9.095', 7: '10.09'}

def get_idx(iom, iv, imu):
    """Get data index for wavelength iom, speed iv, heading imu."""
    return iom * (n_speeds * n_headings) + iv * n_headings + imu

def extract_heading_data(iv, target_mu):
    """Extract drift decomposition for given speed index and heading (in degrees)."""
    # Find closest heading index
    imu = None
    for mu_val, idx in heading_map.items():
        if abs(mu_val - target_mu) < 0.5:
            imu = idx
            break
    if imu is None:
        print(f"WARNING: heading {target_mu} not found. Available: {sorted(heading_map.keys())}")
        return None
    
    result = []
    for iom in range(n_wavelengths):
        d = data[get_idx(iom, iv, imu)]
        omega = d['omega']
        wavelength = 2 * np.pi / (omega**2 / g)  # deep water
        lamL = wavelength / Lpp
        
        s_aw = -d['fxi'] / norm_raw
        s_WL = -d['fxi_WL'] / norm_raw
        s_vel = -d['fxi_vel'] / norm_raw
        s_rot = -d['fxi_rot'] / norm_raw
        # With conjg(-motion): rotation flips sign → total = total_current - 2*rot_current
        s_aw_flipped = s_aw - 2 * s_rot
        
        result.append({
            'omega': omega, 'lamL': lamL,
            's_aw': s_aw, 's_WL': s_WL, 's_vel': s_vel, 's_rot': s_rot,
            's_aw_flipped': s_aw_flipped,
        })
    
    result.sort(key=lambda x: x['lamL'])
    return result


# ============================================================
# ANALYSIS AT V=3 m/s FOR ALL SEO HEADINGS
# ============================================================
iv_seo = 2  # V=3.0 m/s
seo_betas = [180, 150, 120, 90, 60, 30, 0]

print(f"\n{'='*100}")
print(f"ROTATION SIGN ANALYSIS: V=3.0 m/s")
print(f"{'='*100}")

heading_results = {}
for beta in seo_betas:
    hdata = extract_heading_data(iv_seo, float(beta))
    if hdata is None:
        continue
    heading_results[beta] = hdata
    
    print(f"\n--- beta = {beta}° ---")
    print(f"{'lam/L':>8} {'s_aw(+)':>9} {'s_WL':>8} {'s_vel':>8} {'s_rot':>8} {'s_aw(-)':>9} {'diff':>8}")
    
    for d in hdata:
        diff = d['s_aw_flipped'] - d['s_aw']
        print(f"{d['lamL']:8.3f} {d['s_aw']:9.3f} {d['s_WL']:8.3f} {d['s_vel']:8.3f} "
              f"{d['s_rot']:8.3f} {d['s_aw_flipped']:9.3f} {diff:8.3f}")


# ============================================================
# PLOT 1: Comparison at all headings — current vs flipped vs Seo SWAN1
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes_flat = axes.flatten()
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']

for idx, beta in enumerate(seo_betas):
    ax = axes_flat[idx]
    
    # Seo SWAN1
    if beta in seo_data and seo_data[beta]['swan1']:
        x, y = zip(*seo_data[beta]['swan1'])
        ax.plot(x, y, 'b-o', markersize=3, linewidth=1.5, label='SWAN1 (Seo)')
    
    # pdstrip current (conjg(+motion))
    if beta in heading_results:
        hd = heading_results[beta]
        lamL = [d['lamL'] for d in hd]
        s_current = [d['s_aw'] for d in hd]
        s_flipped = [d['s_aw_flipped'] for d in hd]
        
        ax.plot(lamL, s_current, 'r-^', markersize=4, linewidth=1.5, label='pdstrip conjg(+)')
        ax.plot(lamL, s_flipped, 'g--s', markersize=3, linewidth=1.5, label='pdstrip conjg(-)')
    
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$\sigma_{aw}$')
    ax.set_title(f'{subplot_labels[idx]} $\\beta = {beta}°$')
    ax.set_xlim(0, 2.5)
    ylims = {180: (-2, 8), 150: (-4, 8), 120: (-8, 8), 90: (-4, 4), 60: (-4, 2), 30: (-4, 2), 0: (-3, 1)}
    ax.set_ylim(ylims.get(beta, (-5, 10)))
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    if idx == 0:
        ax.legend(fontsize=7, loc='upper right')

for idx in range(len(seo_betas), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle('Surge drift force: effect of rotation term sign\n'
             'conjg(+motion) [current] vs conjg(-motion) [original] at V=3 m/s',
             fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/rotation_sign_all_headings.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: rotation_sign_all_headings.png")


# ============================================================
# PLOT 2: Component decomposition at beta=120° (cf. Seo Figure 8)
# ============================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

for plot_idx, beta in enumerate([120, 150]):
    if beta not in heading_results:
        continue
    hd = heading_results[beta]
    lamL = np.array([d['lamL'] for d in hd])
    s_WL = np.array([d['s_WL'] for d in hd])
    s_vel = np.array([d['s_vel'] for d in hd])
    s_rot = np.array([d['s_rot'] for d in hd])
    s_aw = np.array([d['s_aw'] for d in hd])
    s_aw_flip = np.array([d['s_aw_flipped'] for d in hd])
    
    # Left: component decomposition (current sign)
    ax = axes2[plot_idx, 0]
    ax.plot(lamL, s_WL, 'g-^', markersize=4, label='WL (waterline)')
    ax.plot(lamL, s_vel, 'm-v', markersize=4, label='Vel (velocity²)')
    ax.plot(lamL, s_rot, 'c-d', markersize=4, label='Rot conjg(+)')
    ax.plot(lamL, s_aw, 'r-o', markersize=5, linewidth=2, label='Total conjg(+)')
    # Also overlay what flipped rot would be
    ax.plot(lamL, -s_rot, 'c--d', markersize=3, alpha=0.5, label='Rot conjg(-)')
    ax.plot(lamL, s_aw_flip, 'b--o', markersize=4, linewidth=2, label='Total conjg(-)')
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$\sigma$')
    ax.set_title(f'$\\beta = {beta}°$: Component decomposition')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.5)
    ax.axhline(0, color='k', linewidth=0.5)
    
    # Right: total comparison with Seo
    ax = axes2[plot_idx, 1]
    if beta in seo_data and seo_data[beta]['swan1']:
        x, y = zip(*seo_data[beta]['swan1'])
        ax.plot(x, y, 'b-o', markersize=4, linewidth=2, label='SWAN1 (Seo)')
    ax.plot(lamL, s_aw, 'r-^', markersize=4, linewidth=1.5, label='pdstrip conjg(+)')
    ax.plot(lamL, s_aw_flip, 'g--s', markersize=4, linewidth=1.5, label='pdstrip conjg(-)')
    ax.set_xlabel(r'$\lambda/L$')
    ax.set_ylabel(r'$\sigma_{aw}$')
    ax.set_title(f'$\\beta = {beta}°$: Total vs Seo SWAN1')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.5)
    ax.axhline(0, color='k', linewidth=0.5)

plt.suptitle('KVLCC2 Drift Force Decomposition at V=3 m/s\n'
             'Effect of rotation term sign on roll resonance spike',
             fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/component_decomp_oblique.png', dpi=150, bbox_inches='tight')
print(f"Saved: component_decomp_oblique.png")


# ============================================================
# PLOT 3: Focused view on roll resonance region (lambda/L = 1.0 - 2.0)
# ============================================================
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

for plot_idx, beta in enumerate([120, 150]):
    if beta not in heading_results:
        continue
    hd = heading_results[beta]
    lamL = np.array([d['lamL'] for d in hd])
    s_aw = np.array([d['s_aw'] for d in hd])
    s_aw_flip = np.array([d['s_aw_flipped'] for d in hd])
    s_rot = np.array([d['s_rot'] for d in hd])
    
    ax = axes3[plot_idx]
    if beta in seo_data and seo_data[beta]['swan1']:
        x, y = zip(*seo_data[beta]['swan1'])
        ax.plot(x, y, 'b-o', markersize=5, linewidth=2, label='SWAN1 (Seo)')
    ax.plot(lamL, s_aw, 'r-^', markersize=5, linewidth=2, label='pdstrip conjg(+) [current]')
    ax.plot(lamL, s_aw_flip, 'g-s', markersize=5, linewidth=2, label='pdstrip conjg(-) [original]')
    
    # Mark the roll resonance peak
    peak_idx = np.argmax(np.abs(s_aw))
    ax.axvline(lamL[peak_idx], color='red', alpha=0.3, linestyle=':', label=f'peak at λ/L={lamL[peak_idx]:.2f}')
    
    ax.set_xlabel(r'$\lambda/L$', fontsize=12)
    ax.set_ylabel(r'$\sigma_{aw}$', fontsize=12)
    ax.set_title(f'$\\beta = {beta}°$', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.3, 2.5)
    ax.axhline(0, color='k', linewidth=0.5)

plt.suptitle('Roll resonance region: does flipping rotation sign help?', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/rotation_sign_rollres.png', dpi=150, bbox_inches='tight')
print(f"Saved: rotation_sign_rollres.png")


# ============================================================
# Print key diagnostics around roll resonance
# ============================================================
print(f"\n{'='*100}")
print("KEY DIAGNOSTICS AROUND ROLL RESONANCE")
print(f"{'='*100}")

for beta in [120, 150, 90]:
    if beta not in heading_results:
        continue
    hd = heading_results[beta]
    
    # Find peak |s_aw|
    peak_idx = max(range(len(hd)), key=lambda i: abs(hd[i]['s_aw']))
    d = hd[peak_idx]
    
    print(f"\nbeta={beta}°:")
    print(f"  Peak |sigma_aw| at lambda/L = {d['lamL']:.3f}")
    print(f"  sigma_aw (conjg(+), current)  = {d['s_aw']:.3f}")
    print(f"  sigma_aw (conjg(-), original) = {d['s_aw_flipped']:.3f}")
    print(f"  Components: WL={d['s_WL']:.3f}, vel={d['s_vel']:.3f}, rot={d['s_rot']:.3f}")
    print(f"  Rotation magnitude: |rot| = {abs(d['s_rot']):.3f}")
    print(f"  Rotation as fraction of total: {abs(d['s_rot'])/max(abs(d['s_aw']),1e-10)*100:.1f}%")
    
    # Also check nearby points
    print(f"\n  Lambda/L range 1.0-2.0 detail:")
    for d in hd:
        if 1.0 <= d['lamL'] <= 2.0:
            marker = " <-- PEAK" if abs(d['s_aw']) == max(abs(dd['s_aw']) for dd in hd) else ""
            print(f"    lamL={d['lamL']:.3f}: aw(+)={d['s_aw']:7.3f} aw(-)={d['s_aw_flipped']:7.3f} "
                  f"WL={d['s_WL']:7.3f} vel={d['s_vel']:7.3f} rot={d['s_rot']:7.3f}{marker}")

print("\nDone.")
