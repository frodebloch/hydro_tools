#!/usr/bin/env python3
"""
Cancellation analysis: Do Pinkster components cancel at roll resonance?

From Figure 8 (3D panel code), Components I-IV individually become large at 
roll resonance but cancel to give small total. In pdstrip, this cancellation 
fails. This script quantifies the component magnitudes and their cancellation.

Pinkster near-field components:
  I   = WL integral (waterline pressure squared)
  II  = Velocity squared integral (∮ |∇φ|² n dS)
  III = Rotation term (-½ Re[p × conj(α)] × f_area)
  Total = I + II + III

In pdstrip: WL = feta_WL, Vel = feta_vel, Rot = feta_rot, Total = feta
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# KVLCC2 parameters
Lpp = 328.2; B = 58.0; T = 20.8; rho = 1025.0; g = 9.81
norm = 2*rho*g*Lpp  # sway normalization

# Reference data (approximate from NEWDRIFT figure)
ref_newdrift = [
    (0.2, 0.35), (0.3, 0.42), (0.4, 0.45), (0.5, 0.42),
    (0.6, 0.30), (0.7, 0.18), (0.8, 0.10), (0.9, 0.05),
    (1.0, 0.02), (1.2, 0.00),
]
ref_lam = np.array([r[0] for r in ref_newdrift])
ref_val = np.array([r[1] for r in ref_newdrift])

# Parse debug output
def parse_debug(filename):
    starts = []; totals = []; sway_decomp = []
    with open(filename) as f:
        for line in f:
            if 'DRIFT_START' in line:
                m = re.search(r'omega=\s*([\d.]+)\s+mu=\s*([-\d.]+)', line)
                if m: starts.append((float(m.group(1)), float(m.group(2))))
            elif 'DRIFT_TOTAL' in line:
                m = re.search(r'fxi=\s*([-\d.Ee+]+)\s+feta=\s*([-\d.Ee+]+).*feta_WL=\s*([-\d.Ee+]+)', line)
                if m: totals.append({'feta_total': float(m.group(2)), 'feta_WL': float(m.group(3))})
            elif 'DRIFT_SWAY' in line:
                m = re.search(r'feta_vel=\s*([-\d.Ee+]+)\s+feta_rot=\s*([-\d.Ee+]+)\s+feta_rot_wpst=\s*([-\d.Ee+]+)', line)
                if m: sway_decomp.append({'feta_vel': float(m.group(1)), 'feta_rot': float(m.group(2)), 'feta_rot_wpst': float(m.group(3))})
    if len(sway_decomp) == 0 and len(totals) > 0:
        sway_decomp = [{'feta_vel': 0.0, 'feta_rot': 0.0, 'feta_rot_wpst': 0.0}] * len(totals)
    return starts, totals, sway_decomp

# Parse motions from pdstrip.out
def parse_motions(filename):
    motions = {}
    current_omega = None; current_mu = None; current_speed = None
    expect_translation = False; expect_rotation = False
    with open(filename) as f:
        for line in f:
            m = re.search(r'Wave circ\. frequency\s+([\d.]+).*wave angle\s+([-\d.]+)', line)
            if m:
                current_omega = float(m.group(1)); current_mu = float(m.group(2))
                continue
            if current_omega is not None and 'speed' in line and 'wetted' in line:
                m2 = re.search(r'speed\s+([\d.]+)', line)
                if m2: current_speed = float(m2.group(1))
                continue
            if 'Real part(1)' in line and 'Imagin' in line:
                expect_translation = True; continue
            if expect_translation and 'Translation' in line:
                expect_translation = False; expect_rotation = True
                parts = line.split()
                if (len(parts) >= 10 and current_mu is not None and 
                    abs(current_mu - 90.0) < 0.1 and 
                    current_speed is not None and abs(current_speed) < 0.1):
                    motions.setdefault(current_omega, {})
                    motions[current_omega]['sway_re'] = float(parts[3])
                    motions[current_omega]['sway_im'] = float(parts[6])
                    motions[current_omega]['sway'] = float(parts[6])  # amplitude from imaginary part
                    motions[current_omega]['heave_re'] = float(parts[4])
                    motions[current_omega]['heave_im'] = float(parts[7])
                    motions[current_omega]['heave'] = float(parts[9])
                continue
            if expect_rotation and 'Rotation' in line:
                expect_rotation = False
                parts = line.split()
                if (len(parts) >= 10 and current_mu is not None and 
                    abs(current_mu - 90.0) < 0.1 and 
                    current_speed is not None and abs(current_speed) < 0.1):
                    motions.setdefault(current_omega, {})
                    motions[current_omega]['roll_re'] = float(parts[3])
                    motions[current_omega]['roll_im'] = float(parts[6])
                    motions[current_omega]['roll_k'] = float(parts[3])  # roll/k
                continue
    return motions


# Damping variants
variants = [
    ('0% damp', 'debug_0pct_new.out', 'pdstrip_0pct_new.out'),
    ('15% damp', 'debug.out', 'pdstrip.out'),
    ('25% damp', 'debug_25pct_new.out', 'pdstrip_25pct_new.out'),
    ('50% damp', 'debug_50pct_new.out', 'pdstrip_50pct_new.out'),
]

def get_sway_data(starts, totals, sway_decomp):
    omegas_desc = sorted(set(s[0] for s in starts), reverse=True)
    mus = sorted(set(s[1] for s in starts))
    n_o, n_h = len(omegas_desc), len(mus)
    n_s = len(starts) // (n_o * n_h)
    mu_90_idx = mus.index(90.0)
    data = []
    for iom in range(n_o):
        idx = iom * (n_s * n_h) + 0 * n_h + mu_90_idx
        omega = starts[idx][0]
        k = omega**2 / g
        lam_L = 2*np.pi/k / Lpp
        feta = totals[idx]['feta_total']
        feta_WL = totals[idx]['feta_WL']
        feta_vel = sway_decomp[idx]['feta_vel']
        feta_rot = sway_decomp[idx]['feta_rot']
        data.append({'omega': omega, 'lam_L': lam_L,
                     'total': -feta/norm, 'WL': -feta_WL/norm,
                     'vel': -feta_vel/norm, 'rot': -feta_rot/norm,
                     'total_raw': feta, 'WL_raw': feta_WL,
                     'vel_raw': feta_vel, 'rot_raw': feta_rot})
    data.sort(key=lambda x: x['lam_L'])
    return data

# Parse all
all_data = {}; all_motions = {}
for label, dbg, pds in variants:
    try:
        s, t, d = parse_debug(dbg)
        all_data[label] = get_sway_data(s, t, d)
    except Exception as e:
        print(f"WARN: {dbg}: {e}")
    try:
        all_motions[label] = parse_motions(pds)
    except Exception as e:
        print(f"WARN: {pds}: {e}")


# ==========================================
# MAIN ANALYSIS: Cancellation at roll resonance
# ==========================================
print("=" * 130)
print("CANCELLATION ANALYSIS: Pinkster components at roll resonance (sway drift, mu=90°, V=0)")
print("In 3D codes, WL + Vel + Rot cancel at resonance. In pdstrip, they don't.")
print(f"Normalization: 2*rho*g*Lpp = {norm:.0f} N")
print("=" * 130)

for label in ['0% damp', '15% damp', '50% damp']:
    if label not in all_data:
        continue
    print(f"\n--- {label} ---")
    print(f"{'λ/L':>6} {'omega':>6} {'WL':>10} {'Vel':>10} {'Rot':>10} {'Total':>10} {'Ref':>10} {'|WL+Vel|':>10} {'WL+Vel+Rot':>10} {'Cancel%':>10}")
    print("-" * 120)
    data = all_data[label]
    for row in data:
        lam_L = row['lam_L']
        if lam_L < 0.15 or lam_L > 2.0: continue
        ref_i = np.interp(lam_L, ref_lam, ref_val)
        wl = row['WL']
        vel = row['vel']
        rot = row['rot']
        total = row['total']
        wl_vel = wl + vel
        wl_vel_rot = wl + vel + rot
        # Cancellation: how much of |WL| + |Rot| cancels
        sum_abs = abs(wl) + abs(vel) + abs(rot)
        if sum_abs > 1e-10:
            cancel_pct = (1 - abs(total)/sum_abs) * 100
        else:
            cancel_pct = 0
        print(f"{lam_L:6.3f} {row['omega']:6.3f} {wl:10.4f} {vel:10.4f} {rot:10.4f} {total:10.4f} {ref_i:10.4f} {wl_vel:10.4f} {wl_vel_rot:10.4f} {cancel_pct:9.1f}%")

# ==========================================
# KEY COMPARISON: Roll amplitude vs component magnitudes
# ==========================================
print("\n" + "=" * 130)
print("ROLL AMPLITUDE vs COMPONENT MAGNITUDES")
print("If rotation term ∝ |roll|², we'd expect rot/roll² = const. Let's check.")
print("=" * 130)

for label in ['0% damp', '15% damp', '50% damp']:
    if label not in all_data or label not in all_motions:
        continue
    print(f"\n--- {label} ---")
    print(f"{'λ/L':>6} {'roll(°)':>10} {'roll²':>12} {'WL':>10} {'Rot':>10} {'Total':>10} {'Rot/roll²':>12} {'WL/roll²':>12}")
    print("-" * 110)
    data = all_data[label]
    motions = all_motions[label]
    for row in data:
        lam_L = row['lam_L']
        if lam_L < 0.4 or lam_L > 2.0: continue
        omega = row['omega']
        k = omega**2 / g
        roll_deg = 0
        if omega in motions:
            roll_k = motions[omega].get('roll_k', 0)
            roll_deg = roll_k * k * 180/np.pi
        roll_sq = roll_deg**2
        wl = row['WL']
        rot = row['rot']
        total = row['total']
        r_rr = rot/roll_sq if roll_sq > 1e-10 else 0
        wl_rr = wl/roll_sq if roll_sq > 1e-10 else 0
        print(f"{lam_L:6.3f} {roll_deg:10.4f} {roll_sq:12.6f} {wl:10.4f} {rot:10.4f} {total:10.4f} {r_rr:12.4f} {wl_rr:12.4f}")


# ==========================================
# WHAT SHOULD HAPPEN: Expected cancellation
# ==========================================
print("\n" + "=" * 130)
print("EXPECTED vs ACTUAL CANCELLATION")
print("At roll resonance, total drift should be small (∝ damping).")
print("In 3D code (Fig 8): total peaks at ~5 (norm units), components individually reach ~30-40")
print("The MISSING cancellation = rotation term not being large/opposite enough to cancel WL")
print("=" * 130)

label = '0% damp'
if label in all_data:
    data = all_data[label]
    print(f"\n{label}:")
    print(f"{'λ/L':>6} {'WL':>10} {'Rot':>10} {'Needed_Rot':>12} {'Actual/Needed':>14} {'Total':>10} {'Ref':>10}")
    print("-" * 90)
    for row in data:
        lam_L = row['lam_L']
        if lam_L < 0.7 or lam_L > 2.0: continue
        ref_i = np.interp(lam_L, ref_lam, ref_val)
        wl = row['WL']
        vel = row['vel']
        rot = row['rot']
        total = row['total']
        # Needed rotation to make total = ref
        needed_rot = ref_i - wl - vel
        ratio = rot / needed_rot if abs(needed_rot) > 1e-10 else float('inf')
        print(f"{lam_L:6.3f} {wl:10.4f} {rot:10.4f} {needed_rot:12.4f} {ratio:14.4f} {total:10.4f} {ref_i:10.4f}")


# ==========================================
# PLOT: Component magnitudes and cancellation
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Pinkster Component Cancellation Analysis — KVLCC2, beam seas (μ=90°), V=0', 
             fontsize=13, fontweight='bold')

colors = {'0% damp': '#e41a1c', '15% damp': '#377eb8', '25% damp': '#4daf4a', '50% damp': '#984ea3'}

# Panel (a): All components for 0% damping (worst case)
ax = axes[0, 0]
label = '0% damp'
if label in all_data:
    data = all_data[label]
    lams = [r['lam_L'] for r in data if 0.15 < r['lam_L'] < 2.0]
    wls = [r['WL'] for r in data if 0.15 < r['lam_L'] < 2.0]
    vels = [r['vel'] for r in data if 0.15 < r['lam_L'] < 2.0]
    rots = [r['rot'] for r in data if 0.15 < r['lam_L'] < 2.0]
    tots = [r['total'] for r in data if 0.15 < r['lam_L'] < 2.0]
    ax.plot(lams, wls, 'b-', linewidth=1.5, label='Comp I (WL)')
    ax.plot(lams, vels, 'r--', linewidth=1.5, label='Comp II (Vel)')
    ax.plot(lams, rots, 'g-.', linewidth=1.5, label='Comp III (Rot)')
    ax.plot(lams, tots, 'k-', linewidth=2.5, label='Total')
    ax.plot(ref_lam, ref_val, 'ko', markersize=5, label='NEWDRIFT (3D)')
ax.set_xlabel('λ/L')
ax.set_ylabel('F_y / (2ρgζₐ²Lpp)')
ax.set_title(f'(a) {label} — All Components')
ax.set_xlim(0.15, 2.0)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axvline(x=1.3, color='gray', linestyle=':', alpha=0.5)
ax.text(1.35, ax.get_ylim()[1]*0.85, 'roll\nresonance', fontsize=7, color='gray')

# Panel (b): All components for 15% damping
ax = axes[0, 1]
label = '15% damp'
if label in all_data:
    data = all_data[label]
    lams = [r['lam_L'] for r in data if 0.15 < r['lam_L'] < 2.0]
    wls = [r['WL'] for r in data if 0.15 < r['lam_L'] < 2.0]
    vels = [r['vel'] for r in data if 0.15 < r['lam_L'] < 2.0]
    rots = [r['rot'] for r in data if 0.15 < r['lam_L'] < 2.0]
    tots = [r['total'] for r in data if 0.15 < r['lam_L'] < 2.0]
    ax.plot(lams, wls, 'b-', linewidth=1.5, label='Comp I (WL)')
    ax.plot(lams, vels, 'r--', linewidth=1.5, label='Comp II (Vel)')
    ax.plot(lams, rots, 'g-.', linewidth=1.5, label='Comp III (Rot)')
    ax.plot(lams, tots, 'k-', linewidth=2.5, label='Total')
    ax.plot(ref_lam, ref_val, 'ko', markersize=5, label='NEWDRIFT (3D)')
ax.set_xlabel('λ/L')
ax.set_ylabel('F_y / (2ρgζₐ²Lpp)')
ax.set_title(f'(b) {label} — All Components')
ax.set_xlim(0.15, 2.0)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axvline(x=1.3, color='gray', linestyle=':', alpha=0.5)

# Panel (c): Cancellation metric across damping levels
ax = axes[1, 0]
for label, _, _ in variants:
    if label not in all_data: continue
    data = all_data[label]
    lams = []
    cancel_pcts = []
    for row in data:
        if 0.15 < row['lam_L'] < 2.0:
            lams.append(row['lam_L'])
            sum_abs = abs(row['WL']) + abs(row['vel']) + abs(row['rot'])
            if sum_abs > 1e-10:
                cancel_pcts.append((1 - abs(row['total'])/sum_abs) * 100)
            else:
                cancel_pcts.append(0)
    ax.plot(lams, cancel_pcts, color=colors[label], linewidth=1.5, label=label)
ax.set_xlabel('λ/L')
ax.set_ylabel('Cancellation %')
ax.set_title('(c) Component Cancellation Level')
ax.set_xlim(0.15, 2.0)
ax.set_ylim(-10, 100)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=1.3, color='gray', linestyle=':', alpha=0.5)

# Panel (d): Component ratio (Rot/WL) — should be ~-1 at resonance for cancellation
ax = axes[1, 1]
for label, _, _ in variants:
    if label not in all_data: continue
    data = all_data[label]
    lams = []
    ratios = []
    for row in data:
        if 0.6 < row['lam_L'] < 2.0 and abs(row['WL']) > 1e-6:
            lams.append(row['lam_L'])
            ratios.append(-row['rot']/row['WL'])  # negative because they should be opposite sign
    ax.plot(lams, ratios, color=colors[label], linewidth=1.5, label=label)
ax.set_xlabel('λ/L')
ax.set_ylabel('-Rot/WL')
ax.set_title('(d) Rotation-to-WL Ratio (should → 1 at resonance)')
ax.set_xlim(0.6, 2.0)
ax.axhline(y=1.0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axvline(x=1.3, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('cancellation_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: cancellation_analysis.png")


# ==========================================
# ENERGY ARGUMENT
# ==========================================
print("\n" + "=" * 130)
print("ENERGY BALANCE ARGUMENT")
print("=" * 130)
print("""
At roll resonance, the energy balance requires:
  Power_in (from waves) = Power_dissipated (by damping) + Power_drift (drift force × group velocity)

In a linear system with low damping:
  - Power_in ∝ |roll|² × damping  (because excitation ∝ |roll| × damping at resonance)
  - Power_dissipated ∝ |roll|² × damping
  - Power_drift should be SMALL (bounded)

The Pinkster near-field method computes the drift force from:
  F_drift = WL + Vel + Rot
  
Each component individually ∝ |roll|² at resonance (they all involve |motion|²).
But the TOTAL should remain bounded because drift force ∝ energy flux.

In a 3D panel code: WL, Vel, Rot are all computed from the SAME 3D potential.
The self-consistency of the 3D solution ensures cancellation.

In strip theory: WL uses 2D section pressures (pres_nopst). Rot uses presaverage 
(including pst). The velocity field comes from 2D section solutions.
These are NOT guaranteed to be self-consistent — they come from different 
approximations of the 3D problem.

SPECIFIC ISSUE: The pst (hydrostatic restoring) term in the rotation component 
creates a contribution:
  F_rot_pst = -½ Re[ρg(η₃ + y·η₄) × conj(η₄)] × ∫(y·f_area_y)dS
  
This term ∝ |η₄|² at roll resonance and has no corresponding cancelling term 
in WL or Vel because pst is EXCLUDED from WL (correctly, for surge).

For SWAY: removing pst from rotation breaks everything (sign flip).
But INCLUDING pst creates a term that blows up at roll resonance without 
a matching cancellation from WL.

This is the fundamental inconsistency in applying Pinkster to strip theory.
""")
