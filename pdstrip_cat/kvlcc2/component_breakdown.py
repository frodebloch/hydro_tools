#!/usr/bin/env python3
"""
Extract head seas drift force components at speed_idx=2 from debug_15pct.out.
Show WL, vel, rot contributions to understand the overprediction.
"""
import re
import numpy as np

Lpp = 328.2
B = 58.0
rho = 1025.0
g = 9.81
norm = rho * g * B**2 / Lpp

# Parse debug file
starts = []
totals = []

with open('/home/blofro/src/pdstrip_test/kvlcc2/debug_15pct.out') as f:
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
                    'feta': float(m.group(2)),
                    'fxi_WL': float(m.group(3)),
                    'feta_WL': float(m.group(4)),
                    'fxi_vel': float(m.group(5)),
                    'fxi_rot': float(m.group(6)),
                })

print(f"Parsed {len(starts)} starts, {len(totals)} totals")

omegas = sorted(set(s[0] for s in starts))
mus = sorted(set(s[1] for s in starts))
n_headings = len(mus)
n_omegas = len(omegas)
n_speeds = len(starts) // (n_omegas * n_headings)
print(f"{n_omegas} omegas, {n_headings} headings, {n_speeds} speeds")

# Find mu=180 index
mu_idx = None
for i, m in enumerate(mus):
    if abs(m - 180.0) < 1.0:
        mu_idx = i
        break
print(f"mu=180 index: {mu_idx}")

speed_idx = 2

print(f"\n{'='*120}")
print(f"HEAD SEAS (beta=180), Speed idx={speed_idx}: Component breakdown")
print(f"norm = rho*g*B^2/Lpp = {norm:.1f}")
print(f"{'='*120}")
print(f"{'lam/L':>7} | {'omega':>6} | {'fxi':>12} | {'fxi_WL':>12} | {'fxi_vel':>12} | {'fxi_rot':>10} | "
      f"{'sigma':>8} | {'sig_WL':>8} | {'sig_vel':>8} | {'sig_rot':>8} | {'WL/total':>9}")
print(f"{'-'*120}")

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
    
    d = totals[idx]
    sigma = -d['fxi'] / norm
    sig_WL = -d['fxi_WL'] / norm
    sig_vel = -d['fxi_vel'] / norm
    sig_rot = -d['fxi_rot'] / norm
    
    wl_ratio = d['fxi_WL'] / d['fxi'] if abs(d['fxi']) > 1.0 else float('nan')
    
    if 0.3 <= lam_L <= 2.5:
        print(f"{lam_L:7.3f} | {omega:6.3f} | {d['fxi']:12.1f} | {d['fxi_WL']:12.1f} | "
              f"{d['fxi_vel']:12.1f} | {d['fxi_rot']:10.1f} | "
              f"{sigma:8.3f} | {sig_WL:8.3f} | {sig_vel:8.3f} | {sig_rot:8.3f} | {wl_ratio:9.2f}")

print(f"\nKey: sigma = -fxi/norm (positive = added resistance)")
print(f"fxi_WL = waterline integral (contains pst)")
print(f"fxi_vel = velocity-squared integral (always negative fxi -> positive sigma)")  
print(f"fxi_rot = rotation/pressure cross term (small)")
print(f"WL/total = ratio of WL to total fxi")
