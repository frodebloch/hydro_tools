#!/usr/bin/env python3
"""
Extract per-section WL contributions for short-wave analysis.
Focus on shortest wavelength (27.4m, lambda/L=0.083) at zero-speed head seas.
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Lpp = 328.2
B = 58.0
rho = 1025.0
g = 9.81

# We need to find the correct block in debug.out.
# The shortest wavelength = 27.395m → omega = sqrt(g*k) = sqrt(g*2*pi/27.395)
# k = 2*pi/27.395 = 0.2294, omega = sqrt(9.81*0.2294) = 1.500 rad/s
# At zero speed, ome = omega = 1.500
# Head seas: mu = 180.0

target_omega = 1.500
target_mu = 180.0

# Also collect a medium wavelength for comparison
# lambda = 278.4m → lambda/L = 0.848, k=0.02258, omega=sqrt(9.81*0.02258)=0.4705
target_omega_mid = 0.470
target_mu_mid = 180.0

print(f"Looking for omega≈{target_omega}, mu={target_mu}")
print(f"Also looking for omega≈{target_omega_mid}, mu={target_mu_mid}")

# Parse debug.out for per-section WL data
# Format: WL sec=  3 dx2=  5.260 dystb=  0.040 |p_stb|=  34020.6     |p_port|=  34020.6     dfeta=  0.00000     feta_cum= -95.0832

def parse_block(filepath, omega_target, mu_target, tol_omega=0.01, tol_mu=0.5):
    """Extract WL section data and TRI_SUM section data for a specific omega/mu block."""
    wl_sections = []
    tri_sections = []
    in_block = False
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('DRIFT_START'):
                m = re.search(r'omega=\s*([\d.+-eE]+)\s+mu=\s*([\d.+-eE]+)', line)
                if m:
                    omega = float(m.group(1))
                    mu = float(m.group(2))
                    if abs(omega - omega_target) < tol_omega and abs(mu - mu_target) < tol_mu:
                        in_block = True
                        wl_sections = []
                        tri_sections = []
                    else:
                        in_block = False
            elif in_block and '  WL sec=' in line:
                m = re.search(
                    r'WL sec=\s*(\d+)\s+dx2=\s*([\d.+-eE]+)\s+dystb=\s*([\d.+-eE]+)\s+'
                    r'\|p_stb\|=\s*([\d.+-eE]+)\s+\|p_port\|=\s*([\d.+-eE]+)\s+'
                    r'dfeta=\s*([\d.+-eE]+)\s+feta_cum=\s*([\d.+-eE]+)',
                    line
                )
                if m:
                    wl_sections.append({
                        'sec': int(m.group(1)),
                        'dx2': float(m.group(2)),
                        'dystb': float(m.group(3)),
                        'p_stb': float(m.group(4)),
                        'p_port': float(m.group(5)),
                        'dfeta': float(m.group(6)),
                        'feta_cum': float(m.group(7)),
                    })
            elif in_block and '  TRI_SUM' in line:
                m = re.search(r'TRI_SUM sec=\s*(\d+)\s+feta_tri=\s*([\d.+-eE]+)', line)
                if m:
                    tri_sections.append({
                        'sec': int(m.group(1)),
                        'feta_tri': float(m.group(2)),
                    })
            elif in_block and 'DRIFT_TOTAL' in line:
                m = re.search(
                    r'fxi=\s*([\d.+-eE]+)\s+feta=\s*([\d.+-eE]+)\s+'
                    r'fxi_WL=\s*([\d.+-eE]+)\s+feta_WL=\s*([\d.+-eE]+)\s+'
                    r'fxi_vel=\s*([\d.+-eE]+)\s+fxi_rot=\s*([\d.+-eE]+)',
                    line
                )
                if m:
                    totals = {
                        'fxi': float(m.group(1)),
                        'feta': float(m.group(2)),
                        'fxi_WL': float(m.group(3)),
                        'feta_WL': float(m.group(4)),
                        'fxi_vel': float(m.group(5)),
                        'fxi_rot': float(m.group(6)),
                    }
                    return wl_sections, tri_sections, totals
    return wl_sections, tri_sections, None

filepath = '/home/blofro/src/pdstrip_test/kvlcc2/debug.out'

# Short wave
print("\n=== SHORT WAVE (lambda/L ≈ 0.083) ===")
wl_short, tri_short, totals_short = parse_block(filepath, target_omega, target_mu)
if totals_short:
    print(f"Total fxi={totals_short['fxi']:.1f}, fxi_WL={totals_short['fxi_WL']:.1f}, fxi_vel={totals_short['fxi_vel']:.1f}")
    print(f"\nPer-section WL contributions:")
    print(f"{'Sec':>4} {'dx2':>8} {'dystb':>8} {'|p_stb|':>12} {'|p_port|':>12} {'dfxi_stb':>12} {'dfxi_prt':>12} {'dfxi_WL':>12}")
    
    total_dfxi_wl = 0
    sec_data = []
    for s in wl_short:
        # Reconstruct dfxi contributions
        dfxi_stb = 0.25 * s['p_stb']**2 * s['dystb'] / rho / g
        dyprt_approx = s['dystb']  # For symmetric hull, approximate
        dfxi_prt = -0.25 * s['p_port']**2 * s['dystb'] / rho / g  # Approximate — dystb ≈ dyprt for symmetric hull
        dfxi_wl = dfxi_stb + dfxi_prt  # For symmetric hull and head seas, this should be small
        total_dfxi_wl += dfxi_wl
        sec_data.append((s['sec'], s['dx2'], s['dystb'], s['p_stb'], s['p_port'], dfxi_stb, dfxi_prt, dfxi_wl))
        
    for sd in sec_data:
        print(f"{sd[0]:4d} {sd[1]:8.3f} {sd[2]:8.4f} {sd[3]:12.1f} {sd[4]:12.1f} {sd[5]:12.2f} {sd[6]:12.2f} {sd[7]:12.2f}")
    
    print(f"\nSum of dfxi_WL ≈ {total_dfxi_wl:.2f}")
    print(f"Actual fxi_WL from totals: {totals_short['fxi_WL']:.2f}")
    print(f"Note: My reconstruction is approximate because I assumed dyprt=dystb")
else:
    print("Block not found!")

# Also check: for head seas with symmetric hull, pressures should be equal on both sides
# So dfeta_WL should be near zero (which it is if p_stb ≈ p_port)
if wl_short:
    print(f"\nWL feta_cum (cumulative lateral) = {wl_short[-1]['feta_cum']:.2f}")
    print(f"(Should be ~0 for head seas due to symmetry)")

# Now let me also extract the actual fxi_WL per section.
# The total fxi_WL = sum of (dfxistb + dfxiprt) over all sections.
# From the code: dfxistb = 0.25*|p(1,1,ise1)|^2*dystb/rho/g
#                dfxiprt = -0.25*|p(npres,1,ise1)|^2*dyprt/rho/g
# For head seas with symmetric hull: |p_stb| = |p_port|
# So dfxi_WL per section = 0.25*|p|^2*(dystb - dyprt)/(rho*g)
# The WL pushes forward (negative fxi) when dystb > dyprt (bow narrowing from stern perspective)
# Wait, dystb = d(yint(1))/dx. For a ship widening from bow to stern:
#   At bow sections: yint(1) is DECREASING toward bow → dystb < 0
#   At midship: yint(1) roughly constant → dystb ≈ 0
#   At stern: depends on hull shape

print("\n\n=== MEDIUM WAVE (lambda/L ≈ 0.85) ===")
wl_mid, tri_mid, totals_mid = parse_block(filepath, target_omega_mid, target_mu, tol_omega=0.02)
if totals_mid:
    print(f"Total fxi={totals_mid['fxi']:.1f}, fxi_WL={totals_mid['fxi_WL']:.1f}, fxi_vel={totals_mid['fxi_vel']:.1f}")
    print(f"\nPer-section WL contributions:")
    print(f"{'Sec':>4} {'dx2':>8} {'dystb':>8} {'|p_stb|':>12} {'|p_port|':>12}")
    for s in wl_mid[:10]:
        print(f"{s['sec']:4d} {s['dx2']:8.3f} {s['dystb']:8.4f} {s['p_stb']:12.1f} {s['p_port']:12.1f}")
    print("  ... (truncated)")
else:
    print("Block not found! Trying nearby omegas...")
    # Let's check what omegas are available
    with open(filepath, 'r') as f:
        omegas = []
        for line in f:
            if line.startswith('DRIFT_START') and 'mu=   180.0' in line:
                m = re.search(r'omega=\s*([\d.+-eE]+)', line)
                if m:
                    omegas.append(float(m.group(1)))
        print(f"Available omegas for mu=180: {sorted(set(omegas))[:10]} ... {sorted(set(omegas))[-5:]}")

# Plot per-section WL contributions for short wave
if wl_short:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    secs = [s['sec'] for s in wl_short]
    p_stb = [s['p_stb'] for s in wl_short]
    p_port = [s['p_port'] for s in wl_short]
    dystbs = [s['dystb'] for s in wl_short]
    dx2s = [s['dx2'] for s in wl_short]
    
    # Compute per-section dfxi_WL (need actual dyprt, approximate from symmetry)
    dfxi_per_sec = [0.25 * ps**2 * dy / rho / g + (-0.25 * pp**2 * dy / rho / g) 
                    for ps, pp, dy in zip(p_stb, p_port, dystbs)]
    
    ax = axes[0, 0]
    ax.plot(secs, p_stb, 'b-o', markersize=3, label='|p_stb| (code: port geom)')
    ax.plot(secs, p_port, 'r-s', markersize=3, label='|p_port| (code: stbd geom)')
    ax.set_xlabel('Section number')
    ax.set_ylabel('|p| (Pa)')
    ax.set_title(f'Pressure magnitude at waterline (λ/L={27.4/Lpp:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(secs, dystbs, 'g-o', markersize=3)
    ax.set_xlabel('Section number')
    ax.set_ylabel('dystb (m)')
    ax.set_title('Waterline slope dy/dx (code stb = geom port)')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    
    ax = axes[1, 0]
    ax.plot(secs, dfxi_per_sec, 'b-o', markersize=3)
    ax.set_xlabel('Section number')
    ax.set_ylabel('dfxi_WL (N/m²)')
    ax.set_title('Per-section WL x-drift contribution')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    
    ax = axes[1, 1]
    cum_fxi = np.cumsum(dfxi_per_sec)
    ax.plot(secs, cum_fxi, 'b-o', markersize=3)
    ax.set_xlabel('Section number')
    ax.set_ylabel('Cumulative fxi_WL (N/m²)')
    ax.set_title('Cumulative WL x-drift (forward = negative)')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    
    plt.suptitle(f'KVLCC2 Short Wave WL Analysis: λ={27.4:.1f}m, λ/L={27.4/Lpp:.3f}, head seas, zero speed')
    plt.tight_layout()
    plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/short_wave_wl_analysis.png', dpi=150)
    print(f"\nSaved short_wave_wl_analysis.png")
