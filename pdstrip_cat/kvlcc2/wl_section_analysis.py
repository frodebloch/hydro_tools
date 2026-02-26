#!/usr/bin/env python3
"""
Parse section-by-section waterline contributions from debug file for head seas.
Focus on 3 wavelengths:
  - lambda/L ~ 0.62 (where pdstrip matches Seo)
  - lambda/L ~ 0.85 (worst overprediction, 11x)
  - lambda/L ~ 1.05 (near Seo peak, 2.4x)
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
norm = rho * g * B**2 / Lpp

# Target wavelengths (lambda/L)
target_lam_L = [0.62, 0.85, 1.05]
# Corresponding approximate omega values: omega = sqrt(2*pi*g / (lam_L * Lpp))
target_omegas = [np.sqrt(2 * np.pi * g / (ll * Lpp)) for ll in target_lam_L]
print("Target wavelengths and omegas:")
for ll, om in zip(target_lam_L, target_omegas):
    print(f"  lambda/L = {ll:.2f}  -> omega ~ {om:.4f}")

# Parse the debug file for head seas (mu=180), speed index 2
# We'll read block by block
def parse_wl_sections(fname, target_omegas, target_mu=180.0, speed_idx=2):
    """
    Parse debug file and extract per-section WL contributions for specific omegas at head seas.
    """
    n_speeds = 8
    n_headings = 36
    mu_idx = 27  # mu=180 is at index 27
    
    results = {}  # omega -> [(sec, dfxi_stb, dfxi_prt, dfxi_total), ...]
    
    # We need to track which omega/heading block we're in
    current_omega = None
    current_mu = None
    record_count = 0
    collecting = False
    sections = []
    
    with open(fname, 'r') as f:
        for line in f:
            if 'DRIFT_START' in line:
                m = re.search(r'omega=\s*([\d.]+)\s+mu=\s*([-\d.]+)', line)
                if m:
                    current_omega = float(m.group(1))
                    current_mu = float(m.group(2))
                    
                    # Check if this is a target block
                    omega_match = any(abs(current_omega - to) < 0.002 for to in target_omegas)
                    mu_match = abs(current_mu - target_mu) < 1.0
                    
                    # Check speed: we need to figure out which speed this is
                    # record_count within this omega tells us speed and heading
                    # But we track globally: record = iom * (n_speeds * n_headings) + iv * n_headings + imu
                    # So within the current omega, the record offset is iv * n_headings + imu
                    # This is harder to track with global counting, so let's use a different approach:
                    # Just check if mu is 180 and omega matches, and track per-omega counter
                    
                    collecting = omega_match and mu_match
                    if collecting:
                        sections = []
                
            elif collecting and 'WL sec=' in line and 'DRIFT' not in line:
                m = re.search(r'WL sec=\s*(\d+)\s+dx2=\s*([\d.]+)\s+dystb=\s*([-\d.]+)\s+dyprt=\s*([-\d.]+)\s+'
                              r'\|p_stb\|=\s*([\d.Ee+]+)\s+\|p_port\|=\s*([\d.Ee+]+)\s+'
                              r'dfxistb=\s*([-\d.Ee+]+)\s+dfxiprt=\s*([-\d.Ee+]+)', line)
                if m:
                    sec = int(m.group(1))
                    dx2 = float(m.group(2))
                    dystb = float(m.group(3))
                    dyprt = float(m.group(4))
                    p_stb = float(m.group(5))
                    p_prt = float(m.group(6))
                    dfxi_stb = float(m.group(7))
                    dfxi_prt = float(m.group(8))
                    sections.append({
                        'sec': sec, 'dx2': dx2, 'dystb': dystb, 'dyprt': dyprt,
                        'p_stb': p_stb, 'p_prt': p_prt,
                        'dfxi_stb': dfxi_stb, 'dfxi_prt': dfxi_prt,
                        'dfxi_total': dfxi_stb + dfxi_prt,
                    })
            
            elif collecting and 'DRIFT_TOTAL' in line:
                # Save the collected sections
                if sections:
                    # Find best matching target omega
                    best_omega = min(target_omegas, key=lambda to: abs(current_omega - to))
                    results[current_omega] = sections
                    print(f"  Collected {len(sections)} sections for omega={current_omega:.4f} (target {best_omega:.4f}), mu={current_mu}")
                collecting = False
    
    return results

print("\nParsing debug file for section-level WL data...")
wl_data = parse_wl_sections('/home/blofro/src/pdstrip_test/kvlcc2/debug_15pct.out', target_omegas)

if not wl_data:
    print("No matching records found! Let me check what omegas are available...")
    # Read the first few DRIFT_START lines to get omega values
    omegas_found = set()
    with open('/home/blofro/src/pdstrip_test/kvlcc2/debug_15pct.out', 'r') as f:
        for line in f:
            if 'DRIFT_START' in line:
                m = re.search(r'omega=\s*([\d.]+)', line)
                if m:
                    omegas_found.add(float(m.group(1)))
    
    omegas_sorted = sorted(omegas_found)
    print(f"Available omegas: {omegas_sorted}")
    
    # Find closest matches
    for to in target_omegas:
        closest = min(omegas_sorted, key=lambda o: abs(o - to))
        print(f"  Target omega {to:.4f} -> closest available {closest:.4f} (lambda/L = {2*np.pi*g/(closest**2)/Lpp:.3f})")
    
    # Retry with closest omegas
    closest_omegas = [min(omegas_sorted, key=lambda o: abs(o - to)) for to in target_omegas]
    print(f"\nRetrying with closest omegas: {closest_omegas}")
    wl_data = parse_wl_sections('/home/blofro/src/pdstrip_test/kvlcc2/debug_15pct.out', closest_omegas)


# Now the issue: the data in wl_data has EVERY mu=180 match for the given omega,
# but there are 8 speed blocks per omega. We need speed index 2.
# Actually, the approach above catches ALL mu=180 blocks at that omega (8 of them).
# We need a more careful parser.

# Let me rewrite to properly track speed index
def parse_wl_sections_v2(fname, target_omegas_exact, target_mu=180.0, speed_idx=2):
    """
    More careful parser that tracks speed index within each omega block.
    """
    n_speeds = 8
    n_headings = 36
    
    results = {}
    
    block_count = 0  # global block counter
    current_omega = None
    current_mu = None
    sections = []
    collecting = False
    
    with open(fname, 'r') as f:
        for line in f:
            if 'DRIFT_START' in line:
                m = re.search(r'omega=\s*([\d.]+)\s+mu=\s*([-\d.]+)', line)
                if m:
                    current_omega = float(m.group(1))
                    current_mu = float(m.group(2))
                    
                    # Which omega index are we at?
                    # block_count = iom * (n_speeds * n_headings) + iv * n_headings + imu
                    within_omega = block_count % (n_speeds * n_headings)
                    iv = within_omega // n_headings
                    imu = within_omega % n_headings
                    
                    omega_match = current_omega in target_omegas_exact
                    mu_match = abs(current_mu - target_mu) < 1.0
                    speed_match = (iv == speed_idx)
                    
                    collecting = omega_match and mu_match and speed_match
                    if collecting:
                        sections = []
                    
                    block_count += 1
                
            elif collecting and 'WL sec=' in line and 'DRIFT' not in line:
                m = re.search(r'WL sec=\s*(\d+)\s+dx2=\s*([\d.]+)\s+dystb=\s*([-\d.]+)\s+dyprt=\s*([-\d.]+)\s+'
                              r'\|p_stb\|=\s*([\d.Ee+]+)\s+\|p_port\|=\s*([\d.Ee+]+)\s+'
                              r'dfxistb=\s*([-\d.Ee+]+)\s+dfxiprt=\s*([-\d.Ee+]+)', line)
                if m:
                    sections.append({
                        'sec': int(m.group(1)),
                        'dx2': float(m.group(2)),
                        'dystb': float(m.group(3)),
                        'dyprt': float(m.group(4)),
                        'p_stb': float(m.group(5)),
                        'p_prt': float(m.group(6)),
                        'dfxi_stb': float(m.group(7)),
                        'dfxi_prt': float(m.group(8)),
                    })
            
            elif collecting and 'DRIFT_TOTAL' in line:
                if sections:
                    lam_L = 2 * np.pi * g / (current_omega**2) / Lpp
                    results[round(lam_L, 3)] = {
                        'omega': current_omega,
                        'sections': sections,
                    }
                    print(f"  Collected {len(sections)} WL sections for omega={current_omega:.4f}, lam/L={lam_L:.3f}, mu={current_mu}")
                collecting = False
    
    return results


# First find exact omega values
omegas_found = set()
with open('/home/blofro/src/pdstrip_test/kvlcc2/debug_15pct.out', 'r') as f:
    for line in f:
        if 'DRIFT_START' in line:
            m = re.search(r'omega=\s*([\d.]+)', line)
            if m:
                omegas_found.add(float(m.group(1)))

omegas_sorted = sorted(omegas_found)

# Find the exact omegas closest to our targets
exact_omegas = []
for to in target_omegas:
    closest = min(omegas_sorted, key=lambda o: abs(o - to))
    exact_omegas.append(closest)
    lam_L_actual = 2 * np.pi * g / (closest**2) / Lpp
    print(f"  Target lam/L={2*np.pi*g/(to**2)/Lpp:.3f} -> omega={closest:.4f}, actual lam/L={lam_L_actual:.3f}")

print("\nParsing with exact omegas and speed tracking...")
wl_data = parse_wl_sections_v2('/home/blofro/src/pdstrip_test/kvlcc2/debug_15pct.out', exact_omegas)

if wl_data:
    # Plot section-by-section contributions
    fig, axes = plt.subplots(len(wl_data), 1, figsize=(14, 4*len(wl_data)))
    if len(wl_data) == 1:
        axes = [axes]
    
    for idx, (lam_L_key, data) in enumerate(sorted(wl_data.items())):
        ax = axes[idx]
        secs = data['sections']
        
        sec_nums = [s['sec'] for s in secs]
        # Section x-position along ship (approx, from dx2 accumulation)
        x_pos = np.cumsum([s['dx2'] for s in secs]) - np.array([s['dx2'] for s in secs]) / 2
        x_pos_norm = x_pos / Lpp  # normalize to L
        
        dfxi_stb = np.array([s['dfxi_stb'] for s in secs])
        dfxi_prt = np.array([s['dfxi_prt'] for s in secs])
        dfxi_total = dfxi_stb + dfxi_prt
        p_stb = np.array([s['p_stb'] for s in secs])
        p_prt = np.array([s['p_prt'] for s in secs])
        
        # Normalize
        dfxi_total_norm = -dfxi_total / norm
        dfxi_stb_norm = -dfxi_stb / norm
        dfxi_prt_norm = -dfxi_prt / norm
        
        ax.bar(sec_nums, dfxi_total_norm, color='blue', alpha=0.6, label=f'WL total (stb+prt)')
        ax.bar(sec_nums, dfxi_stb_norm, color='red', alpha=0.3, label='WL starboard')
        ax.bar(sec_nums, dfxi_prt_norm, color='green', alpha=0.3, label='WL port')
        
        total_sum = np.sum(dfxi_total_norm)
        ax.set_title(f'$\\lambda/L = {lam_L_key:.3f}$, $\\omega = {data["omega"]:.4f}$, '
                     f'$\\Sigma\\sigma_{{WL}} = {total_sum:.3f}$')
        ax.set_xlabel('Section number')
        ax.set_ylabel(r'Section $\sigma_{aw,WL}$')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        ax.axhline(y=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('/home/blofro/src/pdstrip_test/kvlcc2/wl_section_contributions.png', dpi=150, bbox_inches='tight')
    print("\nSaved: wl_section_contributions.png")
    
    # Also print the pressure magnitudes
    for lam_L_key, data in sorted(wl_data.items()):
        secs = data['sections']
        print(f"\n--- lambda/L = {lam_L_key:.3f} ---")
        print(f"{'sec':>4} | {'|p_stb|':>10} | {'|p_prt|':>10} | {'dfxi_stb':>10} | {'dfxi_prt':>10} | {'dfxi_sum':>10} | {'dy_stb':>8} | {'dy_prt':>8}")
        total_stb = 0
        total_prt = 0
        for s in secs:
            total_stb += s['dfxi_stb']
            total_prt += s['dfxi_prt']
            print(f"{s['sec']:4d} | {s['p_stb']:10.1f} | {s['p_prt']:10.1f} | {s['dfxi_stb']:10.1f} | {s['dfxi_prt']:10.1f} | {s['dfxi_stb']+s['dfxi_prt']:10.1f} | {s['dystb']:8.4f} | {s['dyprt']:8.4f}")
        print(f"TOTALS: stb={total_stb:.1f}, prt={total_prt:.1f}, sum={total_stb+total_prt:.1f}")
        print(f"  sigma_WL = {-(total_stb+total_prt)/norm:.4f}")

else:
    print("ERROR: No WL section data collected!")
