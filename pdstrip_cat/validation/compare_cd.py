#!/usr/bin/env python3
"""Compare pdstrip drift: with Cd vs without Cd vs Capytaine Maruo."""
import numpy as np
import re

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

blocks_cd   = parse_debug("/home/blofro/src/pdstrip_test/validation/run_mono/debug.out")
blocks_nocd = parse_debug("/home/blofro/src/pdstrip_test/validation/run_mono_nocd/debug.out")

data = np.load("/home/blofro/src/pdstrip_test/validation/maruo_drift_comparison.npz")
wavelengths = data['wavelengths']
ff_beam_fy = data['ff_beam_fy_float']
ff_head_fx = data['ff_head_fx_float']

n_headings = 4

print("="*110)
print("BEAM SEAS (mu=90): pdstrip with Cd vs without Cd vs Capytaine")
print(f"{'lam':>5} {'kR':>6}  {'pd_Cd':>12} {'pd_noCd':>12} {'Capytaine':>12}  {'Cd/Cap':>8} {'noCd/Cap':>9}")
print("-"*80)

for i, lam in enumerate(wavelengths):
    k = 2*np.pi/lam; kR = k*1.0
    
    bcd   = blocks_cd[i * n_headings + 2]
    bnocd = blocks_nocd[i * n_headings + 2]
    cap   = ff_beam_fy[i]
    
    rcd   = bcd['feta'] / cap if abs(cap) > 0.01 else float('nan')
    rnocd = bnocd['feta'] / cap if abs(cap) > 0.01 else float('nan')
    
    print(f"{lam:5.0f} {kR:6.3f}  {bcd['feta']:12.1f} {bnocd['feta']:12.1f} {cap:12.1f}  {rcd:8.3f} {rnocd:9.3f}")

print(f"\n{'='*110}")
print("HEAD SEAS (mu=180): pdstrip with Cd vs without Cd vs Capytaine")
print(f"{'lam':>5} {'kR':>6}  {'pd_Cd':>12} {'pd_noCd':>12} {'Capytaine':>12}  {'Cd/Cap':>8} {'noCd/Cap':>9}")
print("-"*80)

for i, lam in enumerate(wavelengths):
    k = 2*np.pi/lam; kR = k*1.0
    
    bcd   = blocks_cd[i * n_headings + 3]
    bnocd = blocks_nocd[i * n_headings + 3]
    cap   = ff_head_fx[i]
    
    rcd   = bcd['fxi'] / cap if abs(cap) > 0.01 else float('nan')
    rnocd = bnocd['fxi'] / cap if abs(cap) > 0.01 else float('nan')
    
    print(f"{lam:5.0f} {kR:6.3f}  {bcd['fxi']:12.1f} {bnocd['fxi']:12.1f} {cap:12.1f}  {rcd:8.3f} {rnocd:9.3f}")
