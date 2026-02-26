#!/usr/bin/env python3
"""
Analyze the effect of roll damping on sway drift force components.
Uses NEW code (with bug fixes) runs at 0%, 15% (current default), 25%, 50% damping.
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

# Damping variants (all NEW code)
variants = [
    ('0% damp', 'debug_0pct_new.out', 'pdstrip_0pct_new.out'),
    ('15% damp', 'debug.out', 'pdstrip.out'),
    ('25% damp', 'debug_25pct_new.out', 'pdstrip_25pct_new.out'),
    ('50% damp', 'debug_50pct_new.out', 'pdstrip_50pct_new.out'),
]


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
                    motions[current_omega]['sway'] = float(parts[6])
                    motions[current_omega]['heave'] = float(parts[9])
                continue
            if expect_rotation and 'Rotation' in line:
                expect_rotation = False
                parts = line.split()
                if (len(parts) >= 10 and current_mu is not None and 
                    abs(current_mu - 90.0) < 0.1 and 
                    current_speed is not None and abs(current_speed) < 0.1):
                    motions.setdefault(current_omega, {})
                    motions[current_omega]['roll_k'] = float(parts[3])
                continue
    return motions


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
                     'vel': -feta_vel/norm, 'rot': -feta_rot/norm})
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

ref_label = '15% damp'

# ==========================================
# ROLL RAO
# ==========================================
print("=" * 110)
print("ROLL RAO at mu=90° (beam seas), V=0 — per unit wave amplitude")
print("=" * 110)
header = f"{'λ/L':>6} {'ω':>6}"
for label, _, _ in variants:
    header += f" {label:>12}"
print(header)
print("-" * 110)

ref_omegas = sorted(all_motions[ref_label].keys())
for omega in ref_omegas:
    k = omega**2 / g
    lam_L = 2*np.pi/k / Lpp
    if lam_L < 0.15 or lam_L > 2.0: continue
    line = f"{lam_L:6.3f} {omega:6.3f}"
    for label, _, _ in variants:
        if label in all_motions and omega in all_motions[label]:
            roll_k = all_motions[label][omega].get('roll_k', 0)
            roll_deg = roll_k * k * 180/np.pi
            line += f" {roll_deg:11.3f}°"
        else:
            line += f" {'N/A':>12}"
    print(line)

# ==========================================
# SWAY DRIFT TOTAL
# ==========================================
print("\n" + "=" * 110)
print("SWAY DRIFT FORCE (TOTAL) — effect of roll damping")
print(f"Normalized by 2ρgLpp = {norm:.0f} N")
print("=" * 110)
header = f"{'λ/L':>6}"
for label, _, _ in variants:
    header += f" {label:>12}"
header += f" {'NEWDRIFT':>10}"
print(header)
print("-" * 110)

ref_data = all_data[ref_label]
for row in ref_data:
    lam_L = row['lam_L']
    if lam_L < 0.15 or lam_L > 2.0: continue
    ref_i = np.interp(lam_L, ref_lam, ref_val)
    line = f"{lam_L:6.3f}"
    for label, _, _ in variants:
        if label in all_data:
            match = [r for r in all_data[label] if abs(r['lam_L'] - lam_L) < 0.001]
            line += f" {match[0]['total']:12.4f}" if match else f" {'N/A':>12}"
    line += f" {ref_i:10.4f}"
    print(line)

# ==========================================
# SWAY DRIFT WL
# ==========================================
print("\n" + "=" * 110)
print("SWAY DRIFT FORCE (WL INTEGRAL) — effect of roll damping")
print(f"Normalized by 2ρgLpp = {norm:.0f} N")
print("=" * 110)
header = f"{'λ/L':>6}"
for label, _, _ in variants:
    header += f" {label:>12}"
header += f" {'NEWDRIFT':>10}"
print(header)
print("-" * 110)

for row in ref_data:
    lam_L = row['lam_L']
    if lam_L < 0.15 or lam_L > 2.0: continue
    ref_i = np.interp(lam_L, ref_lam, ref_val)
    line = f"{lam_L:6.3f}"
    for label, _, _ in variants:
        if label in all_data:
            match = [r for r in all_data[label] if abs(r['lam_L'] - lam_L) < 0.001]
            line += f" {match[0]['WL']:12.4f}" if match else f" {'N/A':>12}"
    line += f" {ref_i:10.4f}"
    print(line)

# ==========================================
# SWAY DRIFT ROTATION TERM
# ==========================================
print("\n" + "=" * 110)
print("SWAY DRIFT FORCE (ROTATION TERM) — effect of roll damping")
print(f"Normalized by 2ρgLpp = {norm:.0f} N")
print("=" * 110)
header = f"{'λ/L':>6}"
for label, _, _ in variants:
    header += f" {label:>12}"
print(header)
print("-" * 110)

for row in ref_data:
    lam_L = row['lam_L']
    if lam_L < 0.15 or lam_L > 2.0: continue
    line = f"{lam_L:6.3f}"
    for label, _, _ in variants:
        if label in all_data:
            match = [r for r in all_data[label] if abs(r['lam_L'] - lam_L) < 0.001]
            line += f" {match[0]['rot']:12.4f}" if match else f" {'N/A':>12}"
    print(line)

# ==========================================
# RMS errors
# ==========================================
print("\n" + "=" * 80)
print("RMS ERRORS vs NEWDRIFT")
print("=" * 80)
for label, _, _ in variants:
    if label in all_data:
        data = all_data[label]
        our_lam = np.array([r['lam_L'] for r in data])
        our_total = np.array([r['total'] for r in data])
        our_interp = np.interp(ref_lam, our_lam, our_total)
        rms = np.sqrt(np.mean((our_interp - ref_val)**2))
        print(f"  {label:20s}: RMS = {rms:.4f}")

# ==========================================
# FOCUS: Critical range λ/L = 0.3-0.6
# ==========================================
print("\n" + "=" * 110)
print("FOCUS: λ/L = 0.25-0.65 — WL, Rot, Total for each damping level vs NEWDRIFT")
print("=" * 110)
labels = [l for l,_,_ in variants if l in all_data]
header = f"{'λ/L':>6}"
for label in labels:
    header += f" {'WL':>7} {'Rot':>7} {'Tot':>7}  |"
header += f" {'Ref':>7}"
print(header)
print("-" * 110)

for row in ref_data:
    lam_L = row['lam_L']
    if lam_L < 0.25 or lam_L > 0.65: continue
    ref_i = np.interp(lam_L, ref_lam, ref_val)
    line = f"{lam_L:6.3f}"
    for label in labels:
        match = [r for r in all_data[label] if abs(r['lam_L'] - lam_L) < 0.001]
        if match:
            line += f" {match[0]['WL']:7.4f} {match[0]['rot']:7.4f} {match[0]['total']:7.4f}  |"
        else:
            line += f" {'':>7} {'':>7} {'':>7}  |"
    line += f" {ref_i:7.4f}"
    print(line)

# ==========================================
# Quantify: Roll sensitivity at λ/L ≈ 0.45
# ==========================================
print("\n" + "=" * 80)
print("ROLL SENSITIVITY SUMMARY at λ/L ≈ 0.45")
print("=" * 80)
target_lam = 0.45
for label, _, _ in variants:
    if label in all_data and label in all_motions:
        match = [r for r in all_data[label] if abs(r['lam_L'] - target_lam) < 0.015]
        if match:
            r = match[0]
            omega = r['omega']
            k = omega**2 / g
            roll_deg = 0
            if omega in all_motions[label]:
                roll_k = all_motions[label][omega].get('roll_k', 0)
                roll_deg = roll_k * k * 180/np.pi
            print(f"  {label:14s}: roll={roll_deg:6.3f}°  WL={r['WL']:.4f}  Rot={r['rot']:.4f}  Total={r['total']:.4f}  (Ref={np.interp(r['lam_L'], ref_lam, ref_val):.4f})")

# ==========================================
# Natural frequency and resonance check
# ==========================================
print("\n" + "=" * 80)
print("ROLL RESONANCE PEAK")
print("=" * 80)
for label, _, _ in variants:
    if label in all_motions:
        motions = all_motions[label]
        max_roll = 0; max_omega = 0
        for omega in sorted(motions.keys()):
            k = omega**2 / g
            roll_k = motions[omega].get('roll_k', 0)
            roll_deg = roll_k * k * 180/np.pi
            if roll_deg > max_roll:
                max_roll = roll_deg; max_omega = omega
        lam_L_peak = 2*np.pi/(max_omega**2/g)/Lpp if max_omega > 0 else 0
        print(f"  {label:14s}: Peak roll = {max_roll:.3f}° at ω = {max_omega:.3f} (λ/L = {lam_L_peak:.3f})")
    
    # Also print roll at λ/L ≈ 0.45
    if label in all_motions:
        for omega in sorted(all_motions[label].keys()):
            k = omega**2 / g
            lam_L = 2*np.pi/k / Lpp
            if abs(lam_L - 0.45) < 0.015:
                roll_k = all_motions[label][omega].get('roll_k', 0)
                roll_deg = roll_k * k * 180/np.pi
                print(f"    -> Roll at λ/L≈0.45: {roll_deg:.3f}°")
                break

# ==========================================
# PLOTS
# ==========================================
colors = {'0% damp': '#e41a1c', '15% damp': '#377eb8', '25% damp': '#4daf4a', '50% damp': '#984ea3'}
styles = {'0% damp': '-', '15% damp': '-', '25% damp': '--', '50% damp': '-.'}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Effect of Roll Damping on Sway Drift Force — KVLCC2, beam seas (μ=90°), V=0', fontsize=13, fontweight='bold')

# --- Panel (a): Roll RAO ---
ax = axes[0, 0]
for label, _, _ in variants:
    if label not in all_motions: continue
    motions = all_motions[label]
    omegas = sorted(motions.keys())
    lams = []; rolls = []
    for omega in omegas:
        k = omega**2 / g
        lam_L = 2*np.pi/k / Lpp
        if lam_L < 0.15 or lam_L > 2.0: continue
        roll_k = motions[omega].get('roll_k', 0)
        roll_deg = roll_k * k * 180/np.pi
        lams.append(lam_L); rolls.append(roll_deg)
    ax.plot(lams, rolls, styles[label], color=colors[label], linewidth=1.5, label=label)
ax.set_xlabel('λ/L')
ax.set_ylabel('Roll amplitude (°/ζₐ)')
ax.set_title('(a) Roll RAO')
ax.set_xlim(0.15, 2.0)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
# Shade the problem region
ax.axvspan(0.35, 0.55, alpha=0.08, color='red')
ax.text(0.45, ax.get_ylim()[1]*0.9, 'WL deficit\nregion', ha='center', fontsize=7, color='red', alpha=0.7)

# --- Panel (b): Sway drift total ---
ax = axes[0, 1]
for label, _, _ in variants:
    if label not in all_data: continue
    data = all_data[label]
    lams = [r['lam_L'] for r in data if 0.15 < r['lam_L'] < 2.0]
    tots = [r['total'] for r in data if 0.15 < r['lam_L'] < 2.0]
    ax.plot(lams, tots, styles[label], color=colors[label], linewidth=1.5, label=label)
ax.plot(ref_lam, ref_val, 'ko-', markersize=5, linewidth=2, label='NEWDRIFT (3D)', zorder=5)
ax.set_xlabel('λ/L')
ax.set_ylabel('F_y / (2ρgζₐ²Lpp)')
ax.set_title('(b) Sway Drift Force — Total')
ax.set_xlim(0.15, 1.5)
ax.set_ylim(-0.05, 0.55)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axvspan(0.35, 0.55, alpha=0.08, color='red')

# --- Panel (c): WL integral ---
ax = axes[1, 0]
for label, _, _ in variants:
    if label not in all_data: continue
    data = all_data[label]
    lams = [r['lam_L'] for r in data if 0.15 < r['lam_L'] < 2.0]
    wls = [r['WL'] for r in data if 0.15 < r['lam_L'] < 2.0]
    ax.plot(lams, wls, styles[label], color=colors[label], linewidth=1.5, label=label)
ax.plot(ref_lam, ref_val, 'ko-', markersize=5, linewidth=2, label='NEWDRIFT (3D)', zorder=5)
ax.set_xlabel('λ/L')
ax.set_ylabel('F_y / (2ρgζₐ²Lpp)')
ax.set_title('(c) Sway Drift Force — WL Integral Only')
ax.set_xlim(0.15, 1.5)
ax.set_ylim(-0.1, 0.55)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axvspan(0.35, 0.55, alpha=0.08, color='red')
ax.annotate('WL curves nearly\nidentical across\nall damping levels', xy=(0.45, 0.33), xytext=(0.8, 0.45),
            fontsize=8, ha='center', arrowprops=dict(arrowstyle='->', color='gray'), color='gray')

# --- Panel (d): Rotation term ---
ax = axes[1, 1]
for label, _, _ in variants:
    if label not in all_data: continue
    data = all_data[label]
    lams = [r['lam_L'] for r in data if 0.15 < r['lam_L'] < 2.0]
    rots = [r['rot'] for r in data if 0.15 < r['lam_L'] < 2.0]
    ax.plot(lams, rots, styles[label], color=colors[label], linewidth=1.5, label=label)
ax.set_xlabel('λ/L')
ax.set_ylabel('F_y / (2ρgζₐ²Lpp)')
ax.set_title('(d) Sway Drift Force — Rotation Term Only')
ax.set_xlim(0.15, 2.0)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.annotate('0% damping:\nroll resonance\nblow-up', xy=(1.29, 0.378), xytext=(1.6, 0.3),
            fontsize=8, ha='center', arrowprops=dict(arrowstyle='->', color='red'), color='red')

plt.tight_layout()
plt.savefig('roll_damping_sway.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: roll_damping_sway.png")
