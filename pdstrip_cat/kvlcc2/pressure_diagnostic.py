#!/usr/bin/env python3
"""
Diagnostic plot: Waterline pressure distribution along KVLCC2 hull
at head seas (beta=180°), lambda/L=0.847, V=3 m/s.

Shows how the pst (hydrostatic restoring) contribution creates
massive fore-aft pressure asymmetry that drives the WL drift overprediction.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rho = 1025.0
g = 9.81
Lpp = 328.2
B = 58.0
norm = rho * g * B**2 / Lpp

# --- Data from debug_15pct.out: speed index 2 (V=3 m/s), omega=0.471, mu=180 ---
# Sections 1 (bow) to 62 (stern)
nsec = 62
p_stb = np.array([10232.9, 8892.96, 8404.23, 7782.12, 7047.98, 6233.83, 5367.63, 4477.30,
         3600.42, 2839.17, 2390.59, 2487.49, 3097.44, 3989.02, 4985.23, 5991.10,
         6960.19, 7842.99, 8633.51, 9323.19, 9912.86, 10389.7, 10763.7, 11024.8,
         11159.8, 11209.5, 11179.2, 11065.3, 10870.9, 10606.6, 10278.5, 9898.73,
         9475.84, 9032.88, 8599.34, 8210.40, 7889.08, 7669.18, 7605.94, 7719.01,
         8022.15, 8514.43, 9177.19, 9965.80, 10849.1, 11795.2, 12805.4, 13826.9,
         14797.9, 15735.7, 16684.9, 17652.6, 18635.3, 19617.5, 20523.3, 21351.1,
         22249.8, 23716.3, 27072.3, 29973.3, 26404.3, 18234.5])

dystb = np.array([1.3585, 2.6050, 2.4200, 2.2140, 1.9580, 1.7910, 1.6345, 1.4375,
         1.2525, 1.0555, 0.8465, 0.6180, 0.4030, 0.2285, 0.1260, 0.0570,
         0.0055, -0.0070, -0.0105, -0.0105, -0.0085, -0.0070, -0.0055, -0.0015,
         0.0020, 0.0040, 0.0050, 0.0055, 0.0060, 0.0055, 0.0060, 0.0065,
         0.0060, 0.0060, 0.0060, 0.0060, 0.0060, 0.0060, 0.0065, 0.0065,
         0.0065, 0.0065, 0.0050, 0.0000, -0.0130, -0.0325, -0.0700, -0.1065,
         -0.1195, -0.1725, -0.3235, -0.5660, -0.9045, -1.3665, -1.8840, -2.3805,
         -3.0365, -4.1445, -5.8185, -5.7510, -2.3525, 0.0000])

# x-coordinates (approximate): section 1 at bow, 62 at stern
# From geometry: dx2 ≈ 5.38m between most sections, first/last dx2=2.69m
x_sec = np.zeros(nsec)
x_sec[0] = Lpp / 2  # bow at +Lpp/2
for i in range(1, nsec):
    x_sec[i] = x_sec[i-1] - 5.38
# Adjust last point
x_sec[-1] = -Lpp / 2

p_FK = rho * g  # Froude-Krylov pressure for unit amplitude

# Approximate pst contribution: rho*g*(eta3 - x*eta5)
# Motion amplitudes at this frequency
heave_abs = 0.139  # m per unit wave amplitude
pitch_kA = 0.295
k = 2 * np.pi / (0.847 * Lpp)
pitch_rad = pitch_kA * k  # rad per unit amplitude

pst_approx = rho * g * np.abs(heave_abs - x_sec * pitch_rad)

# WL contribution per section (stb + port = 2x since symmetric)
dfxi_wl = 2 * 0.25 * p_stb**2 * dystb / (rho * g)

# Cumulative sum
dfxi_cum = np.cumsum(dfxi_wl)

# ============================================================
# Create 4-panel figure
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- (a) Waterline pressure along hull ---
ax = axes[0, 0]
ax.plot(x_sec / Lpp, p_stb / p_FK, 'r-o', markersize=3, linewidth=1.5, label='|p_total| / ρgA')
ax.axhline(1.0, color='b', linestyle='--', linewidth=1, label='Froude-Krylov (ρgA)')
ax.plot(x_sec / Lpp, pst_approx / p_FK, 'g--s', markersize=2, linewidth=1, alpha=0.7,
        label='|pst·motion| / ρgA (approx)')
ax.set_xlabel('x/L (+ = bow, - = stern)', fontsize=11)
ax.set_ylabel('|p| / ρgA', fontsize=10)
ax.set_title('(a) Waterline pressure distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.55, 0.55)
ax.annotate(f'Stern: {p_stb[-3]/p_FK:.1f}×ρgA', xy=(x_sec[-3]/Lpp, p_stb[-3]/p_FK),
            fontsize=8, ha='right', color='red')
ax.annotate(f'Bow: {p_stb[0]/p_FK:.1f}×ρgA', xy=(x_sec[0]/Lpp, p_stb[0]/p_FK),
            fontsize=8, ha='left', color='red')

# --- (b) Waterline slope (dy/dx) ---
ax = axes[0, 1]
ax.bar(x_sec / Lpp, dystb, width=5.38/Lpp*0.8, color='steelblue', alpha=0.7)
ax.set_xlabel('x/L (+ = bow, - = stern)', fontsize=11)
ax.set_ylabel('dystb (m)', fontsize=10)
ax.set_title('(b) Waterline slope dy/dx', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.55, 0.55)
ax.axhline(0, color='k', linewidth=0.5)
ax.annotate('Entrance\n(positive)', xy=(0.4, 2), fontsize=9, ha='center', color='blue')
ax.annotate('Run/Stern\n(negative)', xy=(-0.35, -3), fontsize=9, ha='center', color='blue')

# --- (c) WL drift contribution per section ---
ax = axes[1, 0]
colors = ['green' if d > 0 else 'red' for d in dfxi_wl]
ax.bar(x_sec / Lpp, dfxi_wl / norm, width=5.38/Lpp*0.8, color=colors, alpha=0.7)
ax.set_xlabel('x/L (+ = bow, - = stern)', fontsize=11)
ax.set_ylabel(r'$\Delta\sigma_{WL}$ per section', fontsize=10)
ax.set_title('(c) WL drift force contribution per section', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.55, 0.55)
ax.axhline(0, color='k', linewidth=0.5)
ax.annotate('Positive:\nbow entrance\n(small |p|, large dy/dx)',
            xy=(0.35, 0.05), fontsize=8, ha='center', color='green')
ax.annotate('Negative:\nstern run\n(HUGE |p|, large -dy/dx)',
            xy=(-0.35, -1.5), fontsize=8, ha='center', color='red')

# --- (d) Cumulative WL drift ---
ax = axes[1, 1]
ax.plot(x_sec / Lpp, dfxi_cum / norm, 'k-o', markersize=3, linewidth=2)
ax.axhline(-dfxi_cum[-1] / norm, color='r', linestyle='--', linewidth=1,
           label=f'Total: σ_WL = {-dfxi_cum[-1]/norm:.1f}')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel('x/L (+ = bow, - = stern)', fontsize=11)
ax.set_ylabel(r'Cumulative $-f_{xi,WL}$ / norm', fontsize=10)
ax.set_title('(d) Cumulative WL drift (bow to stern)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.55, 0.55)
ax.legend(fontsize=9)

# Summary text
total_pos = dfxi_wl[dfxi_wl > 0].sum()
total_neg = dfxi_wl[dfxi_wl < 0].sum()
fig.text(0.5, -0.02,
         f'λ/L = 0.847 | Head seas (β=180°) | V=3 m/s | '
         f'WL positive: σ={-total_pos/norm:.1f}, WL negative: σ={-total_neg/norm:.1f}, '
         f'Net σ_WL = {-(total_pos+total_neg)/norm:.1f} (SWAN1 total ≈ 0.5)',
         ha='center', fontsize=10, style='italic')

plt.suptitle('KVLCC2 Waterline Pressure & Drift Force Diagnostic\n'
             'The pst (hydrostatic restoring) term drives massive fore-aft pressure asymmetry',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('pressure_diagnostic.png', dpi=150, bbox_inches='tight')
print("Saved: pressure_diagnostic.png")

# Print summary stats
print(f"\nSummary:")
print(f"  Bow pressure (sec 1): {p_stb[0]:.0f} Pa = {p_stb[0]/p_FK:.2f} × ρgA")
print(f"  Minimum pressure (sec {np.argmin(p_stb)+1}): {p_stb.min():.0f} Pa = {p_stb.min()/p_FK:.2f} × ρgA")
print(f"  Stern pressure (sec 60): {p_stb[59]:.0f} Pa = {p_stb[59]/p_FK:.2f} × ρgA")
print(f"  pst at bow: ~{pst_approx[0]:.0f} Pa = {pst_approx[0]/p_FK:.2f} × ρgA")
print(f"  pst at stern: ~{pst_approx[-3]:.0f} Pa = {pst_approx[-3]/p_FK:.2f} × ρgA")
print(f"  Sum dystb = {dystb.sum():.4f} (not zero: hull not closed at waterline)")
print(f"  σ_WL (total) = {-(total_pos+total_neg)/norm:.2f}")
print(f"  σ_WL (entrance positive) = {-total_pos/norm:.2f}")
print(f"  σ_WL (run/stern negative) = {-total_neg/norm:.2f}")
