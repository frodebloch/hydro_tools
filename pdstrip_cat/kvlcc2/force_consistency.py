"""
Analyze pressure-force consistency: does sum(p*n*dS) = -ome^2*M*eta (Newton's 2nd law)?
Parses FORCE_CHECK lines from debug.out.
"""
import re
import numpy as np

debug_file = 'debug.out'

# Parse DRIFT_START + FORCE_CHECK pairs
results = []
current_omega = None
current_mu = None

with open(debug_file) as f:
    for line in f:
        if 'DRIFT_START' in line:
            m = re.search(r'omega=\s*([\d.]+)\s+mu=\s*([-\d.]+)', line)
            if m:
                current_omega = float(m.group(1))
                current_mu = float(m.group(2))
        elif 'FORCE_CHECK' in line and current_omega is not None:
            # Parse: FORCE_CHECK Fy_pres= Re Im Fy_newton= Re Im Fz_pres= Re Im Fz_newton= Re Im
            nums = re.findall(r'[-+]?\d+\.?\d*E?[+-]?\d*', line.replace('E+0', 'E+0').replace('E-0', 'E-0'))
            # More robust: split by known labels
            parts = line.split()
            # Find indices of numeric values
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except ValueError:
                    continue
            if len(vals) == 8:
                fy_pres = complex(vals[0], vals[1])
                fy_newton = complex(vals[2], vals[3])
                fz_pres = complex(vals[4], vals[5])
                fz_newton = complex(vals[6], vals[7])
                results.append({
                    'omega': current_omega,
                    'mu': current_mu,
                    'fy_pres': fy_pres,
                    'fy_newton': fy_newton,
                    'fz_pres': fz_pres,
                    'fz_newton': fz_newton,
                })

print(f"Parsed {len(results)} FORCE_CHECK entries")

# Filter for beam seas (mu=90) and V=0 (speed index 0)
# Ordering: omega(outer, high-to-low) x speed(middle, 8 speeds) x heading(inner, 36 headings)
# So for each omega, mu=90 appears at heading index where mu=90
# Each omega block has 8*36=288 entries
# Speed 0 is first speed in each omega block

# Actually, let's just filter by mu=90 and look at all speeds
beam_90 = [r for r in results if abs(r['mu'] - 90.0) < 0.1]
print(f"\nBeam seas (mu=90): {len(beam_90)} entries ({len(beam_90)//8 if len(beam_90)>0 else 0} frequencies x 8 speeds)")

# For V=0, every 8th entry starting from the first one at each omega
# Actually the entries for a given mu come in groups of 8 speeds per omega
# Let's group by omega
from collections import defaultdict
by_omega = defaultdict(list)
for r in beam_90:
    by_omega[r['omega']].append(r)

print(f"\nFrequencies: {sorted(by_omega.keys(), reverse=True)[:5]}... ({len(by_omega)} total)")

# V=0 is speed index 0 -> first entry in each omega group
print("\n" + "="*120)
print(f"{'omega':>6} {'lam/L':>6} | {'|Fy_pres|':>12} {'|Fy_newt|':>12} {'Fy_ratio':>10} {'Fy_err%':>10} | {'|Fz_pres|':>12} {'|Fz_newt|':>12} {'Fz_ratio':>10} {'Fz_err%':>10}")
print("="*120)

g = 9.81
Lpp = 328.2

for omega in sorted(by_omega.keys(), reverse=True):
    entries = by_omega[omega]
    # V=0 is the first entry (speed index 0)
    r = entries[0]  
    
    lam = 2*np.pi*g / omega**2
    lam_over_L = lam / Lpp
    
    fy_p = r['fy_pres']
    fy_n = r['fy_newton']
    fz_p = r['fz_pres']
    fz_n = r['fz_newton']
    
    fy_ratio = abs(fy_p)/abs(fy_n) if abs(fy_n) > 1 else float('nan')
    fz_ratio = abs(fz_p)/abs(fz_n) if abs(fz_n) > 1 else float('nan')
    
    fy_err = abs(fy_p - fy_n)/abs(fy_n)*100 if abs(fy_n) > 1 else float('nan')
    fz_err = abs(fz_p - fz_n)/abs(fz_n)*100 if abs(fz_n) > 1 else float('nan')
    
    print(f"{omega:6.3f} {lam_over_L:6.2f} | {abs(fy_p):12.1f} {abs(fy_n):12.1f} {fy_ratio:10.3f} {fy_err:9.1f}% | {abs(fz_p):12.1f} {abs(fz_n):12.1f} {fz_ratio:10.3f} {fz_err:9.1f}%")

# Now focus on the key question: what's the phase relationship?
print("\n\nDETAILED COMPLEX VALUES at beam seas (V=0)")
print("="*140)
print(f"{'omega':>6} {'lam/L':>6} | {'Re(Fy_p)':>14} {'Im(Fy_p)':>14} {'Re(Fy_n)':>14} {'Im(Fy_n)':>14} | {'Re(Fz_p)':>14} {'Im(Fz_p)':>14} {'Re(Fz_n)':>14} {'Im(Fz_n)':>14}")
print("="*140)

for omega in sorted(by_omega.keys(), reverse=True):
    entries = by_omega[omega]
    r = entries[0]
    
    lam = 2*np.pi*g / omega**2
    lam_over_L = lam / Lpp
    
    if lam_over_L < 0.5 or lam_over_L > 2.5:
        continue
    
    print(f"{omega:6.3f} {lam_over_L:6.2f} | {r['fy_pres'].real:14.1f} {r['fy_pres'].imag:14.1f} {r['fy_newton'].real:14.1f} {r['fy_newton'].imag:14.1f} | {r['fz_pres'].real:14.1f} {r['fz_pres'].imag:14.1f} {r['fz_newton'].real:14.1f} {r['fz_newton'].imag:14.1f}")

# Key analysis: compute the discrepancy vector
print("\n\nDISCREPANCY ANALYSIS: Delta = F_pres - F_newton")
print("If Delta ~= 0, the pressure field is consistent with Newton's 2nd law.")
print("If Delta != 0, something is missing from the pressure sum.")
print("="*100)
print(f"{'omega':>6} {'lam/L':>6} | {'|dFy|':>12} {'|Fy_n|':>12} {'dFy/Fy%':>10} | {'|dFz|':>12} {'|Fz_n|':>12} {'dFz/Fz%':>10} | {'note':>20}")
print("="*100)

for omega in sorted(by_omega.keys(), reverse=True):
    entries = by_omega[omega]
    r = entries[0]
    
    lam = 2*np.pi*g / omega**2
    lam_over_L = lam / Lpp

    dfy = r['fy_pres'] - r['fy_newton']
    dfz = r['fz_pres'] - r['fz_newton']
    
    fy_err_pct = abs(dfy)/abs(r['fy_newton'])*100 if abs(r['fy_newton']) > 1 else float('nan')
    fz_err_pct = abs(dfz)/abs(r['fz_newton'])*100 if abs(r['fz_newton']) > 1 else float('nan')
    
    note = ''
    if abs(lam_over_L - 1.29) < 0.05:
        note = '<-- ROLL RESONANCE'
    elif fy_err_pct > 50:
        note = '<-- LARGE Fy ERROR'
    elif fz_err_pct > 50:
        note = '<-- LARGE Fz ERROR'
    
    print(f"{omega:6.3f} {lam_over_L:6.2f} | {abs(dfy):12.1f} {abs(r['fy_newton']):12.1f} {fy_err_pct:9.1f}% | {abs(dfz):12.1f} {abs(r['fz_newton']):12.1f} {fz_err_pct:9.1f}% | {note}")
