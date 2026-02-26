#!/usr/bin/env python3
"""
Parse KVLCC2 debug.out drift decomposition and compare rotation term sign variants.
Also parse pdstrip.out for RAOs (heave and pitch) to help validate.
Produces sigma_aw vs lambda/L plot for zero-speed head seas.
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# KVLCC2 parameters
Lpp = 328.2  # m (from section geometry range)
B = 58.0     # m
T = 20.8     # m
rho = 1025.0
g = 9.81

norm_raw = rho * g * B**2 / Lpp  # N/m for sigma_aw normalization

# Parse debug.out DRIFT_TOTAL lines
data = []  # list of dicts with omega, mu, fxi, feta, fxi_WL, feta_WL, fxi_vel, fxi_rot

print("Parsing debug.out...")
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

# We need to identify which entries correspond to zero speed.
# debug.out doesn't directly encode speed, but we know the ordering:
# For each (iom, iv, imu): 35 wavelengths × 8 speeds × 36 headings
# The order is: for each wavelength, for each speed, for each heading.
# So entries 0..287 are wavelength[0], speed[0..7] × heading[0..35]
# For speed index 0 (zero speed), heading indices for each wavelength block.

n_speeds = 8
n_headings = 36
n_wavelengths = 35

# Verify total
assert len(data) == n_wavelengths * n_speeds * n_headings, \
    f"Expected {n_wavelengths * n_speeds * n_headings}, got {len(data)}"

# Extract zero-speed (iv=0) head seas (mu=180)
# Index: for wavelength iom, speed iv, heading imu:
#   idx = iom * (n_speeds * n_headings) + iv * n_headings + imu

# First, figure out which heading index is mu=180
# From first wavelength block, extract all omegas and mus for speed 0
first_block_mus = [data[i]['mu'] for i in range(n_headings)]
print(f"Headings for first wavelength, first speed: {first_block_mus}")

# Find heading index for mu=180 (head seas)
try:
    imu_head = first_block_mus.index(180.0)
except ValueError:
    # Try close match
    for i, m in enumerate(first_block_mus):
        if abs(m - 180.0) < 0.5:
            imu_head = i
            break
    else:
        print("ERROR: mu=180 not found in headings!")
        print(f"Available headings: {first_block_mus}")
        imu_head = None

print(f"Head seas index: {imu_head}")

# Also find beam seas (mu=90 and mu=-90)
imu_beam90 = None
imu_beam270 = None
for i, m in enumerate(first_block_mus):
    if abs(m - 90.0) < 0.5:
        imu_beam90 = i
    if abs(m - 270.0) < 0.5 or abs(m + 90.0) < 0.5:
        imu_beam270 = i

print(f"Beam seas 90 index: {imu_beam90}, 270 index: {imu_beam270}")

# Extract zero-speed head seas data
iv_zero = 0  # zero speed is first
head_seas_data = []
for iom in range(n_wavelengths):
    idx = iom * (n_speeds * n_headings) + iv_zero * n_headings + imu_head
    d = data[idx]
    omega = d['omega']
    wavelength = 2 * np.pi / (omega**2 / g)  # deep water: omega^2 = g*k, k=omega^2/g, lambda=2pi/k
    lamL = wavelength / Lpp
    head_seas_data.append({
        'omega': omega,
        'wavelength': wavelength,
        'lamL': lamL,
        'fxi': d['fxi'],
        'fxi_WL': d['fxi_WL'],
        'fxi_vel': d['fxi_vel'],
        'fxi_rot': d['fxi_rot'],
    })

head_seas_data.sort(key=lambda x: x['lamL'])

print("\n=== Zero-speed Head Seas (mu=180) ===")
print(f"{'lam/L':>8} {'sig_aw':>8} {'sig_WL':>8} {'sig_vel':>8} {'sig_rot':>8} {'sig_aw_alt':>10}")
print(f"{'':>8} {'conjg(+)':>8} {'':>8} {'':>8} {'conjg(+)':>8} {'conjg(-)':>10}")

lamL_arr = []
sig_aw = []
sig_WL = []
sig_vel = []
sig_rot = []
sig_aw_alt = []  # with conjg(-motion), i.e., flip rotation sign

for d in head_seas_data:
    s_aw = -d['fxi'] / norm_raw
    s_WL = -d['fxi_WL'] / norm_raw
    s_vel = -d['fxi_vel'] / norm_raw
    s_rot = -d['fxi_rot'] / norm_raw
    # With conjg(-motion): rotation term flips sign
    s_aw_alt = s_aw - 2 * s_rot  # subtract current rot, add flipped rot = s_aw - 2*s_rot
    
    lamL_arr.append(d['lamL'])
    sig_aw.append(s_aw)
    sig_WL.append(s_WL)
    sig_vel.append(s_vel)
    sig_rot.append(s_rot)
    sig_aw_alt.append(s_aw_alt)
    
    print(f"{d['lamL']:8.3f} {s_aw:8.3f} {s_WL:8.3f} {s_vel:8.3f} {s_rot:8.3f} {s_aw_alt:10.3f}")

lamL_arr = np.array(lamL_arr)
sig_aw = np.array(sig_aw)
sig_WL = np.array(sig_WL)
sig_vel = np.array(sig_vel)
sig_rot = np.array(sig_rot)
sig_aw_alt = np.array(sig_aw_alt)

# ==========================================
# Plot 1: sigma_aw comparison for two rotation signs
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

ax = axes[0]
ax.plot(lamL_arr, sig_aw, 'b-o', markersize=4, label=r'$\sigma_{aw}$ conjg(+motion) [brucon/current]')
ax.plot(lamL_arr, sig_aw_alt, 'r-s', markersize=4, label=r'$\sigma_{aw}$ conjg(-motion) [original]')
ax.set_xlabel(r'$\lambda / L_{pp}$')
ax.set_ylabel(r'$\sigma_{aw} = -F_x / (\rho g B^2 / L_{pp})$')
ax.set_title('KVLCC2 Head Seas Added Resistance - Zero Speed\nComparison of rotation term sign variants')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 3.5)
ax.axhline(0, color='k', linewidth=0.5)

# Add reference: typical published values peak around sigma_aw ~ 2-4 at lambda/L ~ 0.5-1.0
ax.axhspan(1.5, 4.0, alpha=0.1, color='green', label='Typical published range (approx)')

ax = axes[1]
ax.plot(lamL_arr, sig_WL, 'g-^', markersize=4, label=r'$\sigma_{WL}$ (waterline)')
ax.plot(lamL_arr, sig_vel, 'm-v', markersize=4, label=r'$\sigma_{vel}$ (velocity)')
ax.plot(lamL_arr, sig_rot, 'c-d', markersize=4, label=r'$\sigma_{rot}$ (rotation, conjg(+motion))')
ax.plot(lamL_arr, sig_aw, 'b-o', markersize=4, label=r'$\sigma_{aw}$ (total)')
ax.set_xlabel(r'$\lambda / L_{pp}$')
ax.set_ylabel(r'$\sigma$')
ax.set_title('KVLCC2 Head Seas - Drift Force Decomposition (Zero Speed)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 3.5)
ax.axhline(0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/sigma_aw_rotation_comparison.png', dpi=150)
print(f"\nSaved plot to kvlcc2/sigma_aw_rotation_comparison.png")

# ==========================================
# Also extract forward speed data (Fn=0.142, V=7.96 m/s) for head seas
# Speed index 5 (0.0, 2.0, 3.0, 4.0, 6.0, 7.96, 9.095, 10.09) -> iv=5
# ==========================================
iv_design = 5
V_design = 7.96
Fn_design = V_design / np.sqrt(g * Lpp)

print(f"\n=== Design Speed V={V_design} m/s, Fn={Fn_design:.4f} ===")
print(f"{'lam/L':>8} {'sig_aw':>8} {'sig_WL':>8} {'sig_vel':>8} {'sig_rot':>8} {'sig_aw_alt':>10}")

speed_data = []
for iom in range(n_wavelengths):
    idx = iom * (n_speeds * n_headings) + iv_design * n_headings + imu_head
    d = data[idx]
    omega = d['omega']
    wavelength = 2 * np.pi / (omega**2 / g)
    lamL = wavelength / Lpp
    s_aw = -d['fxi'] / norm_raw
    s_WL = -d['fxi_WL'] / norm_raw
    s_vel = -d['fxi_vel'] / norm_raw
    s_rot = -d['fxi_rot'] / norm_raw
    s_aw_alt = s_aw - 2 * s_rot
    speed_data.append((lamL, s_aw, s_WL, s_vel, s_rot, s_aw_alt))
    print(f"{lamL:8.3f} {s_aw:8.3f} {s_WL:8.3f} {s_vel:8.3f} {s_rot:8.3f} {s_aw_alt:10.3f}")

# ==========================================
# Plot 2: Zero speed vs design speed
# ==========================================
speed_data = np.array(speed_data)
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
ax2.plot(lamL_arr, sig_aw, 'b-o', markersize=4, label=f'V=0 conjg(+)')
ax2.plot(lamL_arr, sig_aw_alt, 'b--s', markersize=3, alpha=0.5, label=f'V=0 conjg(-)')
ax2.plot(speed_data[:, 0], speed_data[:, 1], 'r-o', markersize=4, label=f'V={V_design} m/s conjg(+)')
ax2.plot(speed_data[:, 0], speed_data[:, 5], 'r--s', markersize=3, alpha=0.5, label=f'V={V_design} m/s conjg(-)')
ax2.set_xlabel(r'$\lambda / L_{pp}$')
ax2.set_ylabel(r'$\sigma_{aw}$')
ax2.set_title(f'KVLCC2 Head Seas Added Resistance\nZero Speed vs Fn={Fn_design:.3f}')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 3.5)
ax2.axhline(0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/sigma_aw_speed_comparison.png', dpi=150)
print(f"\nSaved speed comparison plot")

# Print summary statistics
print(f"\n=== Summary ===")
print(f"Zero speed head seas:")
print(f"  Peak sigma_aw (conjg(+)): {np.max(sig_aw):.3f} at lambda/L = {lamL_arr[np.argmax(sig_aw)]:.3f}")
print(f"  Peak sigma_aw (conjg(-)): {np.max(sig_aw_alt):.3f} at lambda/L = {lamL_arr[np.argmax(sig_aw_alt)]:.3f}")
print(f"Design speed head seas:")
print(f"  Peak sigma_aw (conjg(+)): {np.max(speed_data[:,1]):.3f} at lambda/L = {speed_data[np.argmax(speed_data[:,1]),0]:.3f}")
print(f"  Peak sigma_aw (conjg(-)): {np.max(speed_data[:,5]):.3f} at lambda/L = {speed_data[np.argmax(speed_data[:,5]),0]:.3f}")
