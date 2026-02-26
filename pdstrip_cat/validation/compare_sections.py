#!/usr/bin/env python3
"""Compare pdstrip drift forces for 5-section and 21-section barge with Capytaine Maruo."""
import numpy as np
import re

rho = 1025.0; g = 9.81; R = 1.0; L = 20.0

fnum = r'[+-]?[\d.]+(?:[EeDd][+-]?\d+)?'

def parse_debug(path):
    blocks = []
    current = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('DRIFT_START'):
                m = re.search(rf'omega=\s*({fnum})\s+mu=\s*({fnum})', line)
                if m:
                    current = {'omega': float(m.group(1)), 'mu': float(m.group(2))}
            elif line.startswith('DRIFT_TOTAL') and current is not None:
                m = re.search(rf'fxi=\s*({fnum})\s+feta=\s*({fnum})', line)
                if m:
                    current['fxi'] = float(m.group(1))
                    current['feta'] = float(m.group(2))
                    blocks.append(current)
                    current = None
    return blocks

blocks_5  = parse_debug("/home/blofro/src/pdstrip_test/validation/run_mono/debug.out")
blocks_21 = parse_debug("/home/blofro/src/pdstrip_test/validation/run_21sec/debug.out")

# Load Capytaine Maruo results
data = np.load("/home/blofro/src/pdstrip_test/validation/maruo_drift_comparison.npz")
wavelengths = data['wavelengths']
ff_beam_fy = data['ff_beam_fy_float']
ff_head_fx = data['ff_head_fx_float']

n_headings = 4  # -90, 0, 90, 180

print("="*100)
print("BEAM SEAS (mu=90 / beta=pi/2): Fy comparison")
print(f"{'lam':>5} {'kR':>6}  {'5sec':>12} {'21sec':>12} {'Capytaine':>12}  {'5s/Cap':>8} {'21s/Cap':>8}")
print("-"*80)

for i, lam in enumerate(wavelengths):
    k = 2*np.pi/lam; kR = k*R
    
    b5  = blocks_5[i * n_headings + 2]   # mu=90
    b21 = blocks_21[i * n_headings + 2]  # mu=90
    cap = ff_beam_fy[i]
    
    r5  = b5['feta'] / cap if abs(cap) > 0.01 else float('nan')
    r21 = b21['feta'] / cap if abs(cap) > 0.01 else float('nan')
    
    print(f"{lam:5.0f} {kR:6.3f}  {b5['feta']:12.1f} {b21['feta']:12.1f} {cap:12.1f}  {r5:8.3f} {r21:8.3f}")

print(f"\n{'='*100}")
print("HEAD SEAS (mu=180 / beta=pi): Fx comparison")
print(f"{'lam':>5} {'kR':>6}  {'5sec':>12} {'21sec':>12} {'Capytaine':>12}  {'5s/Cap':>8} {'21s/Cap':>8}")
print("-"*80)

for i, lam in enumerate(wavelengths):
    k = 2*np.pi/lam; kR = k*R
    
    b5  = blocks_5[i * n_headings + 3]   # mu=180
    b21 = blocks_21[i * n_headings + 3]  # mu=180
    cap = ff_head_fx[i]
    
    r5  = b5['fxi'] / cap if abs(cap) > 0.01 else float('nan')
    r21 = b21['fxi'] / cap if abs(cap) > 0.01 else float('nan')
    
    print(f"{lam:5.0f} {kR:6.3f}  {b5['fxi']:12.1f} {b21['fxi']:12.1f} {cap:12.1f}  {r5:8.3f} {r21:8.3f}")
