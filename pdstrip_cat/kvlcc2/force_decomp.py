"""
Analyze decomposed pressure-force consistency.
Compare panel-integrated wave/rad/pst forces against EOM-based values.
This identifies which component causes the Fy discrepancy.
"""
import re
import numpy as np
from collections import defaultdict

debug_file = 'debug.out'

# Parse all relevant lines
results = []
current = {}

with open(debug_file) as f:
    for line in f:
        if 'DRIFT_START' in line:
            m = re.search(r'omega=\s*([\d.]+)\s+mu=\s*([-\d.]+)', line)
            if m:
                current = {'omega': float(m.group(1)), 'mu': float(m.group(2))}
        elif 'FORCE_CHECK' in line and 'FCHECK' not in line and current:
            vals = []
            for p in line.split():
                try: vals.append(float(p))
                except: pass
            if len(vals) == 8:
                current['fy_pres'] = complex(vals[0], vals[1])
                current['fy_newton'] = complex(vals[2], vals[3])
                current['fz_pres'] = complex(vals[4], vals[5])
                current['fz_newton'] = complex(vals[6], vals[7])
        elif 'FCHECK_WG ' in line and current:
            vals = []
            for p in line.split():
                try: vals.append(float(p))
                except: pass
            if len(vals) == 6:
                current['fy_wg'] = complex(vals[0], vals[1])
                current['fy_rad'] = complex(vals[2], vals[3])
                current['fy_pst'] = complex(vals[4], vals[5])
        elif 'FCHECK_WGZ' in line and current:
            vals = []
            for p in line.split():
                try: vals.append(float(p))
                except: pass
            if len(vals) == 6:
                current['fz_wg'] = complex(vals[0], vals[1])
                current['fz_rad'] = complex(vals[2], vals[3])
                current['fz_pst'] = complex(vals[4], vals[5])
        elif 'FCHECK_EOM ' in line and current:
            vals = []
            for p in line.split():
                try: vals.append(float(p))
                except: pass
            if len(vals) == 6:
                current['fy_exc_eom'] = complex(vals[0], vals[1])
                current['fy_rad_eom'] = complex(vals[2], vals[3])
                current['fy_rst_eom'] = complex(vals[4], vals[5])
        elif 'FCHECK_EOMZ' in line and current:
            vals = []
            for p in line.split():
                try: vals.append(float(p))
                except: pass
            if len(vals) == 6:
                current['fz_exc_eom'] = complex(vals[0], vals[1])
                current['fz_rad_eom'] = complex(vals[2], vals[3])
                current['fz_rst_eom'] = complex(vals[4], vals[5])
                results.append(current)
                current = {}

print(f"Parsed {len(results)} complete entries")

# Filter beam seas V=0
beam_90 = [r for r in results if abs(r['mu'] - 90.0) < 0.1]
by_omega = defaultdict(list)
for r in beam_90:
    by_omega[r['omega']].append(r)

g = 9.81
Lpp = 328.2

print("\n" + "="*130)
print("SWAY (Fy) DECOMPOSITION: Panel vs EOM — Beam seas, V=0")
print("Panel: Fy = Fy_wg + Fy_rad + Fy_pst  (from integrating p*n*dS over triangles)")
print("EOM:   Fy = Fy_exc + Fy_rad_eom + Fy_rst_eom  (from equation of motion coefficients)")
print("="*130)
print(f"{'omega':>6} {'lam/L':>6} | {'|Fy_wg|':>12} {'|Fy_exc|':>12} {'wg_err%':>8} | {'|Fy_rad|':>12} {'|Fy_rad_e|':>12} {'rad_err%':>8} | {'|Fy_pst|':>12} {'|Fy_rst|':>12} {'pst_err%':>8}")
print("-"*130)

for omega in sorted(by_omega.keys(), reverse=True):
    r = by_omega[omega][0]  # V=0
    
    lam = 2*np.pi*g / omega**2
    lam_L = lam / Lpp
    
    if 'fy_wg' not in r:
        continue
    
    wg_err = abs(r['fy_wg'] - r['fy_exc_eom'])/max(abs(r['fy_exc_eom']),1)*100
    rad_err = abs(r['fy_rad'] - r['fy_rad_eom'])/max(abs(r['fy_rad_eom']),1)*100
    pst_err = abs(r['fy_pst'] - r['fy_rst_eom'])/max(abs(r['fy_rst_eom']),1)*100 if abs(r['fy_rst_eom'])>1 else float('nan')
    
    print(f"{omega:6.3f} {lam_L:6.2f} | {abs(r['fy_wg']):12.0f} {abs(r['fy_exc_eom']):12.0f} {wg_err:7.1f}% | {abs(r['fy_rad']):12.0f} {abs(r['fy_rad_eom']):12.0f} {rad_err:7.1f}% | {abs(r['fy_pst']):12.0f} {abs(r['fy_rst_eom']):12.0f} {pst_err:7.1f}%")


print("\n\n" + "="*130)
print("HEAVE (Fz) DECOMPOSITION: Panel vs EOM — Beam seas, V=0")
print("="*130)
print(f"{'omega':>6} {'lam/L':>6} | {'|Fz_wg|':>12} {'|Fz_exc|':>12} {'wg_err%':>8} | {'|Fz_rad|':>12} {'|Fz_rad_e|':>12} {'rad_err%':>8} | {'|Fz_pst|':>12} {'|Fz_rst|':>12} {'pst_err%':>8}")
print("-"*130)

for omega in sorted(by_omega.keys(), reverse=True):
    r = by_omega[omega][0]
    
    lam = 2*np.pi*g / omega**2
    lam_L = lam / Lpp
    
    if 'fz_wg' not in r:
        continue
    
    wg_err = abs(r['fz_wg'] - r['fz_exc_eom'])/max(abs(r['fz_exc_eom']),1)*100
    rad_err = abs(r['fz_rad'] - r['fz_rad_eom'])/max(abs(r['fz_rad_eom']),1)*100
    pst_err = abs(r['fz_pst'] - r['fz_rst_eom'])/max(abs(r['fz_rst_eom']),1)*100 if abs(r['fz_rst_eom'])>1 else float('nan')
    
    print(f"{omega:6.3f} {lam_L:6.2f} | {abs(r['fz_wg']):12.0f} {abs(r['fz_exc_eom']):12.0f} {wg_err:7.1f}% | {abs(r['fz_rad']):12.0f} {abs(r['fz_rad_eom']):12.0f} {rad_err:7.1f}% | {abs(r['fz_pst']):12.0f} {abs(r['fz_rst_eom']):12.0f} {pst_err:7.1f}%")


# Now let's look at the absolute discrepancy per component to see which dominates
print("\n\n" + "="*110)
print("SWAY (Fy) ABSOLUTE DISCREPANCY per component: dF = F_panel - F_eom")
print("="*110)
print(f"{'omega':>6} {'lam/L':>6} | {'|dFy_wg|':>12} {'|dFy_rad|':>12} {'|dFy_pst|':>12} | {'|dFy_tot|':>12} {'|sum_comp|':>12} {'dominant':>10}")
print("-"*110)

for omega in sorted(by_omega.keys(), reverse=True):
    r = by_omega[omega][0]
    
    lam = 2*np.pi*g / omega**2
    lam_L = lam / Lpp
    
    if 'fy_wg' not in r:
        continue
    
    d_wg = r['fy_wg'] - r['fy_exc_eom']
    d_rad = r['fy_rad'] - r['fy_rad_eom']
    d_pst = r['fy_pst'] - r['fy_rst_eom']
    d_tot = r['fy_pres'] - r['fy_newton']
    d_sum = d_wg + d_rad + d_pst
    
    parts = {'wg': abs(d_wg), 'rad': abs(d_rad), 'pst': abs(d_pst)}
    dominant = max(parts, key=parts.get)
    
    print(f"{omega:6.3f} {lam_L:6.2f} | {abs(d_wg):12.0f} {abs(d_rad):12.0f} {abs(d_pst):12.0f} | {abs(d_tot):12.0f} {abs(d_sum):12.0f} {dominant:>10}")
