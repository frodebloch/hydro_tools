#!/usr/bin/env python3
"""
Sway drift force analysis with velocity/rotation decomposition.
Compares pdstrip Pinkster near-field sway drift with NEWDRIFT reference data.
Includes interpolation through roll resonance region.
"""
import re
import numpy as np

# KVLCC2 parameters
Lpp = 328.2; B = 58.0; T = 20.8; rho = 1025.0; g = 9.81
nabla = 320e6 / rho  # displacement volume

# Reference data (approximate from figure)
# NEWDRIFT for KVLCC2 at beam seas, V=0
# Using 2*rho*g*Lpp normalization (best-fit)
ref_newdrift = [
    (0.2, 0.35), (0.3, 0.42), (0.4, 0.45), (0.5, 0.42),
    (0.6, 0.30), (0.7, 0.18), (0.8, 0.10), (0.9, 0.05),
    (1.0, 0.02), (1.2, 0.00),
]

# Extended reference for long-wave tail (drift force should -> 0)
ref_newdrift_ext = ref_newdrift + [(1.5, 0.00), (2.0, 0.00), (3.0, 0.00)]

# Parse debug.out
starts = []; totals = []; sway_decomp = []
with open('debug.out') as f:
    for line in f:
        if 'DRIFT_START' in line:
            m = re.search(r'omega=\s*([\d.]+)\s+mu=\s*([-\d.]+)', line)
            if m: starts.append((float(m.group(1)), float(m.group(2))))
        elif 'DRIFT_TOTAL' in line:
            m = re.search(r'fxi=\s*([-\d.Ee+]+)\s+feta=\s*([-\d.Ee+]+).*feta_WL=\s*([-\d.Ee+]+)', line)
            if m: totals.append({'feta': float(m.group(1)), 'feta_total': float(m.group(2)), 'feta_WL': float(m.group(3))})
        elif 'DRIFT_SWAY' in line:
            m = re.search(r'feta_vel=\s*([-\d.Ee+]+)\s+feta_rot=\s*([-\d.Ee+]+)\s+feta_rot_wpst=\s*([-\d.Ee+]+)', line)
            if m: sway_decomp.append({'feta_vel': float(m.group(1)), 'feta_rot': float(m.group(2)), 'feta_rot_wpst': float(m.group(3))})

omegas_desc = sorted(set(s[0] for s in starts), reverse=True)
mus = sorted(set(s[1] for s in starts))
n_o, n_h = len(omegas_desc), len(mus)
n_s = len(starts) // (n_o * n_h)
mu_90_idx = mus.index(90.0)

def get_idx(iom, ispeed, imu):
    return iom * (n_s * n_h) + ispeed * n_h + imu

# Normalization: 2*rho*g*Lpp (best fit from previous analysis)
norm = 2*rho*g*Lpp

ref_lam = np.array([r[0] for r in ref_newdrift])
ref_val = np.array([r[1] for r in ref_newdrift])
ref_lam_ext = np.array([r[0] for r in ref_newdrift_ext])
ref_val_ext = np.array([r[1] for r in ref_newdrift_ext])

# Collect data for beam seas, V=0
data_rows = []
for iom in range(n_o):
    idx = get_idx(iom, 0, mu_90_idx)
    omega = starts[idx][0]
    k = omega**2 / g
    lam_L = 2*np.pi/k / Lpp
    
    feta = totals[idx]['feta_total']
    feta_WL = totals[idx]['feta_WL']
    feta_vel = sway_decomp[idx]['feta_vel']
    feta_rot = sway_decomp[idx]['feta_rot']
    feta_rot_wpst = sway_decomp[idx]['feta_rot_wpst']
    feta_tri = feta - feta_WL
    
    data_rows.append((lam_L, omega, feta, feta_WL, feta_vel, feta_rot, feta_tri, feta_rot_wpst))

data_rows.sort()  # sort by lambda/L ascending

# Extract arrays for interpolation
lam_arr = np.array([r[0] for r in data_rows])
omega_arr = np.array([r[1] for r in data_rows])
feta_arr = np.array([r[2] for r in data_rows])
feta_WL_arr = np.array([r[3] for r in data_rows])
feta_vel_arr = np.array([r[4] for r in data_rows])
feta_rot_arr = np.array([r[5] for r in data_rows])

# ---- Interpolation through roll resonance ----
omega_roll = 0.382  # from pdstrip output
print("="*110)
print("INTERPOLATION THROUGH ROLL RESONANCE")
print("="*110)

def interpolate_through_resonance(omega_vals, feta_vals, omega_roll, bw_frac):
    """Linearly interpolate feta through the resonance window.
    omega_vals: ascending order by lambda/L => descending by omega.
    We work in omega space (ascending for interpolation convenience).
    """
    # Convert to omega-ascending order for interpolation
    # Note: data_rows is sorted by lam_L ascending = omega descending
    n = len(omega_vals)
    om = omega_vals[::-1]  # now ascending in omega
    fe = feta_vals[::-1]   # same reorder
    
    om_lo = omega_roll * (1 - bw_frac)
    om_hi = omega_roll * (1 + bw_frac)
    
    # Find indices inside window
    inside = (om >= om_lo) & (om <= om_hi)
    if not np.any(inside):
        return feta_vals.copy(), 0
    
    idx_inside = np.where(inside)[0]
    idx_lo = idx_inside[0]
    idx_hi = idx_inside[-1]
    
    # Find good points outside window
    idx_left = idx_lo - 1   # lower omega side
    idx_right = idx_hi + 1  # higher omega side
    
    if idx_left < 0 or idx_right >= n:
        return feta_vals.copy(), 0
    
    # Linear interpolation in omega
    fe_interp = fe.copy()
    for i in range(idx_lo, idx_hi + 1):
        wt = (om[i] - om[idx_left]) / (om[idx_right] - om[idx_left])
        fe_interp[i] = fe[idx_left] + wt * (fe[idx_right] - fe[idx_left])
    
    # Reverse back to original order (lam_L ascending = omega descending)
    return fe_interp[::-1], idx_hi - idx_lo + 1

# Test multiple bandwidths
best_rms = 999.0
best_bw = 0.0
bw_results = []

for bw_frac in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
    feta_interp, n_replaced = interpolate_through_resonance(omega_arr, feta_arr, omega_roll, bw_frac)
    
    # Compute RMS vs NEWDRIFT (extended reference)
    our_norm_interp = -feta_interp / norm
    our_interp_at_ref = np.interp(ref_lam_ext, lam_arr, our_norm_interp)
    rms_ext = np.sqrt(np.mean((our_interp_at_ref - ref_val_ext)**2))
    
    # Also vs original reference
    our_interp_at_ref_orig = np.interp(ref_lam, lam_arr, our_norm_interp)
    rms_orig = np.sqrt(np.mean((our_interp_at_ref_orig - ref_val)**2))
    
    bw_results.append((bw_frac, n_replaced, rms_orig, rms_ext))
    if rms_ext < best_rms:
        best_rms = rms_ext
        best_bw = bw_frac

# Also compute baseline (no interpolation) RMS
our_norm_base = -feta_arr / norm
our_interp_base = np.interp(ref_lam_ext, lam_arr, our_norm_base)
rms_base_ext = np.sqrt(np.mean((our_interp_base - ref_val_ext)**2))
our_interp_base_orig = np.interp(ref_lam, lam_arr, our_norm_base)
rms_base_orig = np.sqrt(np.mean((our_interp_base_orig - ref_val)**2))

print(f"\nomega_roll = {omega_roll:.4f} rad/s")
print(f"\n{'BW%':>5} {'N_repl':>7} {'RMS(orig)':>10} {'RMS(ext)':>10}")
print("-"*40)
print(f"{'none':>5} {'--':>7} {rms_base_orig:10.4f} {rms_base_ext:10.4f}  (no interpolation)")
for bw_frac, n_replaced, rms_orig, rms_ext in bw_results:
    marker = " <-- best" if bw_frac == best_bw else ""
    print(f"{bw_frac*100:4.0f}% {n_replaced:7d} {rms_orig:10.4f} {rms_ext:10.4f}{marker}")

print(f"\nBest bandwidth: ±{best_bw*100:.0f}% of omega_roll")

# Apply best interpolation
feta_best, n_best = interpolate_through_resonance(omega_arr, feta_arr, omega_roll, best_bw)
feta_rot_best, _ = interpolate_through_resonance(omega_arr, feta_rot_arr, omega_roll, best_bw)

print("\n" + "="*110)
print("SWAY DRIFT FORCE DECOMPOSITION: Original vs Interpolated")
print(f"Normalization: 2*rho*g*Lpp = {norm:.0f} N")
print(f"mu = 90° (beam seas), V = 0 m/s, 0% roll damping + interpolation (±{best_bw*100:.0f}%)")
print("="*110)
print(f"{'λ/L':>6} {'omega':>6} {'Orig':>10} {'Interp':>10} {'WL':>10} {'Vel':>10} {'Rot(o)':>10} {'Rot(i)':>10} {'Ref':>10} {'ΔOrig':>10} {'ΔInterp':>10}")
print("-"*110)

for i, (lam_L, omega, feta, feta_WL, feta_vel, feta_rot, feta_tri, feta_rot_wpst) in enumerate(data_rows):
    if lam_L < 0.15 or lam_L > 3.5:
        continue
    
    ref_i = np.interp(lam_L, ref_lam_ext, ref_val_ext)
    t_orig = -feta / norm
    t_interp = -feta_best[i] / norm
    wl = -feta_WL / norm
    vel = -feta_vel / norm
    rot_orig = -feta_rot / norm
    rot_interp = -feta_rot_best[i] / norm
    
    d_orig = t_orig - ref_i
    d_interp = t_interp - ref_i
    
    marker = " *" if abs(omega - omega_roll) < omega_roll * best_bw else ""
    print(f"{lam_L:6.3f} {omega:6.3f} {t_orig:10.4f} {t_interp:10.4f} {wl:10.4f} {vel:10.4f} {rot_orig:10.4f} {rot_interp:10.4f} {ref_i:10.4f} {d_orig:+10.4f} {d_interp:+10.4f}{marker}")

# Final RMS comparison
print("\n" + "="*110)
print("RMS ERROR COMPARISON")
print("="*110)

# 15% damping reference from previous sessions
print(f"  0% damping, no interpolation:  RMS(orig ref) = {rms_base_orig:.4f}, RMS(ext ref) = {rms_base_ext:.4f}")
our_norm_best = -feta_best / norm
rms_best_orig = np.sqrt(np.mean((np.interp(ref_lam, lam_arr, our_norm_best) - ref_val)**2))
rms_best_ext = np.sqrt(np.mean((np.interp(ref_lam_ext, lam_arr, our_norm_best) - ref_val_ext)**2))
print(f"  0% damping + interp (±{best_bw*100:.0f}%):  RMS(orig ref) = {rms_best_orig:.4f}, RMS(ext ref) = {rms_best_ext:.4f}")
print(f"  15% damping (previous):         RMS(orig ref) = 0.0631, RMS(ext ref) = 0.0569")
print(f"  0% + cap=0.06 (previous):       RMS(orig ref) = ~0.05,  RMS(ext ref) = 0.0499")

# --- Validate Fortran interpolation ---
print("\n" + "="*110)
print("FORTRAN INTERPOLATION VALIDATION (from DRIFT_INTERP in debug.out)")
print("="*110)

# Parse DRIFT_INTERP lines
interp_data = {}
with open('debug.out') as f:
    for line in f:
        if 'DRIFT_INTERP' in line:
            m = re.search(r'omega=\s*([\d.]+)\s+mu=\s*([-\d.]+)\s+iv=\s*(\d+)\s+feta_interp=\s*([-\d.Ee+]+)', line)
            if m:
                om_val = float(m.group(1))
                mu_val = float(m.group(2))
                iv_val = int(m.group(3))
                feta_val = float(m.group(4))
                if iv_val == 1 and abs(mu_val - 90.0) < 0.1:
                    interp_data[om_val] = feta_val

if interp_data:
    # Build combined feta array using Fortran-interpolated values where available
    feta_fortran = feta_arr.copy()
    for i, (lam_L, omega, feta, *_) in enumerate(data_rows):
        # Check if this omega has an interpolated value
        for om_key, feta_val in interp_data.items():
            if abs(omega - om_key) < 0.001:
                feta_fortran[i] = feta_val
                break
    
    our_norm_fortran = -feta_fortran / norm
    rms_fortran_orig = np.sqrt(np.mean((np.interp(ref_lam, lam_arr, our_norm_fortran) - ref_val)**2))
    rms_fortran_ext = np.sqrt(np.mean((np.interp(ref_lam_ext, lam_arr, our_norm_fortran) - ref_val_ext)**2))
    
    print(f"  Fortran interp (±12%):  RMS(orig ref) = {rms_fortran_orig:.4f}, RMS(ext ref) = {rms_fortran_ext:.4f}")
    print(f"  Python  interp (±{best_bw*100:.0f}%):  RMS(orig ref) = {rms_best_orig:.4f}, RMS(ext ref) = {rms_best_ext:.4f}")
    
    print(f"\n{'omega':>8} {'Fortran':>12} {'Python':>12} {'Original':>12} {'Diff(F-P)':>12}")
    print("-"*60)
    for i, (lam_L, omega, feta, *_) in enumerate(data_rows):
        f_val = -feta_fortran[i] / norm
        p_val = -feta_best[i] / norm
        o_val = -feta / norm
        if abs(f_val - o_val) > 1e-6 or abs(p_val - o_val) > 1e-6:
            print(f"{omega:8.4f} {f_val:12.4f} {p_val:12.4f} {o_val:12.4f} {f_val-p_val:+12.6f}")
else:
    print("  No DRIFT_INTERP lines found in debug.out")

