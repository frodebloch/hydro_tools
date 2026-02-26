#!/usr/bin/env python3
"""Extract sway drift force transfer function from pdstrip.out files for comparison."""

import re
import sys

def extract_drift_data(filename):
    """Extract omega, wavelength, and drift forces for speed=0, mu=90deg from pdstrip.out."""
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find all "Wave circ. frequency" lines and "Longitudinal and transverse drift" lines
    freq_lines = []  # (line_index, omega, wavelength)
    drift_lines = []  # (line_index, fxi, feta)
    
    freq_pattern = re.compile(
        r'Wave circ\. frequency\s+([\d.]+)\s+encounter frequ\.\s+[\d.]+\s+'
        r'wave length\s+([\d.]+)\s+wave number\s+[\d.]+\s+wave angle\s+([-\d.]+)'
    )
    drift_pattern = re.compile(
        r'Longitudinal and transverse drift force per wave amplitude squared\s+'
        r'([-\d.E+]+)\s+([-\d.E+]+)'
    )
    
    for i, line in enumerate(lines):
        m = freq_pattern.search(line)
        if m:
            omega = float(m.group(1))
            wavelength = float(m.group(2))
            angle = float(m.group(3))
            freq_lines.append((i, omega, wavelength, angle))
        
        m = drift_pattern.search(line)
        if m:
            fxi_str = m.group(1)
            feta_str = m.group(2)
            # Handle Fortran-style numbers like 0.262172E+07
            fxi = float(fxi_str)
            feta = float(feta_str)
            drift_lines.append((i, fxi, feta))
    
    # Now we need to find the drift force entries for speed=0, heading=90deg
    # Structure: for each frequency, loop over 8 speeds, for each speed loop over 36 headings
    # Total per frequency: 8*36 = 288 entries
    # For speed=0, heading index 18 (mu=90): offset = 0*36 + 18 = 18
    
    n_speeds = 8
    n_headings = 36
    entries_per_freq = n_speeds * n_headings
    target_heading_idx = 18  # mu=90
    target_speed_idx = 0
    target_offset = target_speed_idx * n_headings + target_heading_idx
    
    n_freq = len(drift_lines) // entries_per_freq
    
    results = []
    for ifreq in range(n_freq):
        idx = ifreq * entries_per_freq + target_offset
        _, fxi, feta = drift_lines[idx]
        
        # Get omega and wavelength from the corresponding freq_line
        freq_idx = ifreq * entries_per_freq + target_offset
        _, omega, wavelength, angle = freq_lines[freq_idx]
        
        # Ship length for KVLCC2 is 320m
        ship_length = 320.0
        lambda_over_L = wavelength / ship_length
        
        results.append((omega, lambda_over_L, fxi, feta, angle))
    
    return results

def main():
    file_current = '/home/blofro/src/pdstrip_test/kvlcc2/pdstrip.out'
    file_original = '/home/blofro/src/pdstrip_test/kvlcc2_original/pdstrip.out'
    
    print("Extracting from current (modified) file...")
    data_current = extract_drift_data(file_current)
    print(f"  Found {len(data_current)} frequency points")
    
    print("Extracting from original file...")
    data_original = extract_drift_data(file_original)
    print(f"  Found {len(data_original)} frequency points")
    
    # Verify angles match
    for i, (dc, do) in enumerate(zip(data_current, data_original)):
        if abs(dc[0] - do[0]) > 1e-6:
            print(f"WARNING: omega mismatch at index {i}: {dc[0]} vs {do[0]}")
        if abs(dc[4] - do[4]) > 1e-6:
            print(f"WARNING: angle mismatch at index {i}: {dc[4]} vs {do[4]}")
    
    print(f"\nAll entries at mu = {data_current[0][4]}° (heading index 18), speed index 0 (V=0)")
    print()
    
    # Print header
    print(f"{'omega':>8s}  {'λ/L':>8s}  {'fxi_orig':>14s}  {'fxi_curr':>14s}  {'fxi_ratio':>10s}  {'feta_orig':>14s}  {'feta_curr':>14s}  {'feta_ratio':>10s}")
    print("-" * 110)
    
    for i in range(len(data_current)):
        omega_c, lol_c, fxi_c, feta_c, _ = data_current[i]
        omega_o, lol_o, fxi_o, feta_o, _ = data_original[i]
        
        if abs(feta_o) > 1e-3:
            feta_ratio = feta_c / feta_o
        else:
            feta_ratio = float('nan')
        
        if abs(fxi_o) > 1e-3:
            fxi_ratio = fxi_c / fxi_o
        else:
            fxi_ratio = float('nan')
        
        print(f"{omega_c:8.3f}  {lol_c:8.4f}  {fxi_o:14.1f}  {fxi_c:14.1f}  {fxi_ratio:10.4f}  {feta_o:14.1f}  {feta_c:14.1f}  {feta_ratio:10.4f}")
    
    # Also print a focused sway-only table
    print("\n\n=== SWAY DRIFT FORCE (feta) COMPARISON ===")
    print(f"Speed index 0 (V=0), heading mu=90°")
    print()
    print(f"{'omega':>8s}  {'λ/L':>8s}  {'feta_orig':>14s}  {'feta_curr':>14s}  {'ratio':>10s}  {'diff':>14s}")
    print("-" * 80)
    
    for i in range(len(data_current)):
        omega_c, lol_c, fxi_c, feta_c, _ = data_current[i]
        omega_o, lol_o, fxi_o, feta_o, _ = data_original[i]
        
        if abs(feta_o) > 1e-3:
            feta_ratio = feta_c / feta_o
        else:
            feta_ratio = float('nan')
        
        diff = feta_c - feta_o
        
        print(f"{omega_c:8.3f}  {lol_c:8.4f}  {feta_o:14.1f}  {feta_c:14.1f}  {feta_ratio:10.6f}  {diff:14.1f}")

if __name__ == '__main__':
    main()
