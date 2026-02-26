#!/usr/bin/env python3
"""Compare catamaran sectionresults (wide spacing) against 2x monohull values.

For widely separated hulls, the catamaran section hydrodynamic coefficients
should approach 2x the monohull values (two non-interacting identical hulls).

Specifically:
- Added mass (heave-heave): cat ~ 2 * mono
- Added mass (sway-sway): cat ~ 2 * mono 
- Added mass (roll-roll): cat ~ 2*mono_roll + 2*hulld^2*mono_sway  (parallel axis)
- Diffraction forces: cat ~ 2 * mono (in magnitude, with phase shift)
- FK forces: cat ~ 2 * mono (in magnitude, with phase shift)

For sway and heave added mass and head-seas (mu=180, symmetric case), the 2x
relationship should hold closely.
"""

import sys
import re

def parse_sectionresults(filename):
    """Parse sectionresults file, return structured data."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip title line
    idx = 1  # line 0 is title
    sections = []
    
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        
        # Try to read nfre, x
        parts = line.split()
        if len(parts) < 2:
            idx += 1
            continue
            
        try:
            nfre = int(parts[0])
            xpos = float(parts[1])
        except (ValueError, IndexError):
            idx += 1
            continue
        
        idx += 1
        section = {'x': xpos, 'nfre': nfre, 'frequencies': []}
        
        for ifre in range(nfre):
            if idx >= len(lines):
                break
            
            # Line: omega, nmu, wave angles
            freq_line = lines[idx].strip()
            idx += 1
            freq_parts = freq_line.split()
            omega = float(freq_parts[0])
            nmu = int(freq_parts[1])
            
            # Line: radiation forces (9 complex = 18 real values)
            rad_line = lines[idx].strip()
            idx += 1
            rad_vals = parse_complex_line(rad_line)
            
            # Line: diffraction forces (3*nmu complex values)
            diff_line = lines[idx].strip()
            idx += 1
            diff_vals = parse_complex_line(diff_line)
            
            # Line: FK forces (3*nmu complex values)
            fk_line = lines[idx].strip()
            idx += 1
            fk_vals = parse_complex_line(fk_line)
            
            # Line: pressures (if npres > 0)
            # Check if next line starts with an integer (pressure point index)
            pres_vals = None
            if idx < len(lines):
                next_line = lines[idx].strip()
                next_parts = next_line.split()
                if len(next_parts) > 2:
                    try:
                        # If first token is "1" it's a pressure line
                        test = int(next_parts[0])
                        if test == 1:
                            pres_vals = next_line
                            idx += 1
                    except ValueError:
                        pass
            
            section['frequencies'].append({
                'omega': omega,
                'nmu': nmu,
                'radiation': rad_vals,  # 9 complex values
                'diffraction': diff_vals,
                'fk': fk_vals,
            })
        
        sections.append(section)
    
    return sections


def parse_complex_line(line):
    """Parse a line of Fortran complex numbers in (re,im) format."""
    # Find all (re,im) pairs
    pairs = re.findall(r'\(([^,]+),([^)]+)\)', line)
    result = []
    for re_str, im_str in pairs:
        result.append(complex(float(re_str), float(im_str)))
    return result


def compare_sections(mono_file, cat_file):
    """Compare catamaran vs 2x monohull section results."""
    mono = parse_sectionresults(mono_file)
    cat = parse_sectionresults(cat_file)
    
    print(f"Monohull: {len(mono)} sections")
    print(f"Catamaran: {len(cat)} sections")
    print()
    
    nsections = min(len(mono), len(cat))
    
    # Compare first few sections, first few frequencies
    for isec in range(min(nsections, 3)):  # first 3 sections
        print(f"=== Section {isec+1}, x = {mono[isec]['x']:.1f} ===")
        
        nfre = min(len(mono[isec]['frequencies']), len(cat[isec]['frequencies']))
        
        for ifre in range(min(nfre, 5)):  # first 5 frequencies
            mf = mono[isec]['frequencies'][ifre]
            cf = cat[isec]['frequencies'][ifre]
            omega = mf['omega']
            
            print(f"\n  omega = {omega:.4f}")
            
            # Radiation forces (omega^2 * addedm): indices 0-8 for (j,l) pairs
            # (1,1)=sway-sway idx0, (1,2)=sway-heave idx1, (1,3)=sway-roll idx2
            # (2,1)=heave-sway idx3, (2,2)=heave-heave idx4, (2,3)=heave-roll idx5
            # (3,1)=roll-sway idx6, (3,2)=roll-heave idx7, (3,3)=roll-roll idx8
            
            labels = ['sway-sway', 'sway-heave', 'sway-roll',
                      'heave-sway', 'heave-heave', 'heave-roll',
                      'roll-sway', 'roll-heave', 'roll-roll']
            
            if len(mf['radiation']) >= 9 and len(cf['radiation']) >= 9:
                print(f"  Radiation forces (omega^2 * addedm):")
                print(f"  {'Component':>14s}  {'|mono|':>12s}  {'|cat|':>12s}  {'|cat|/2|mono|':>14s}")
                for i, label in enumerate(labels):
                    m_val = mf['radiation'][i]
                    c_val = cf['radiation'][i]
                    m_abs = abs(m_val)
                    c_abs = abs(c_val)
                    if m_abs > 1e-6:
                        ratio = c_abs / (2 * m_abs)
                    else:
                        ratio = float('nan')
                    print(f"  {label:>14s}  {m_abs:12.4f}  {c_abs:12.4f}  {ratio:14.6f}")
            
            # Head seas diffraction (mu=180 is the middle angle, index nmu//2 for symmetric range)
            nmu = mf['nmu']
            # mu angles go from -90 to +90. Head seas = index for mu=0 (beam=0 in internal coords?)
            # Actually mu=0 in pdstrip = following seas, mu=180 = head seas
            # But wave angles in sectionresults are the actual angles used
            # For our 19-angle case: -90,-80,...,0,...,80,90 -> index 9 is mu=0
            # Actually these are in the hydrodynamic section computation frame
            # Let's just compare the head-seas case (middle angle)
            mid = nmu // 2  # index of mu=0
            
            if len(mf['diffraction']) >= 3*nmu and len(cf['diffraction']) >= 3*nmu:
                print(f"\n  Diffraction at mu=0 (index {mid}):")
                print(f"  {'Component':>14s}  {'|mono|':>12s}  {'|cat|':>12s}  {'|cat|/2|mono|':>14s}")
                for j, name in enumerate(['sway', 'heave', 'roll']):
                    m_val = mf['diffraction'][j*nmu + mid]
                    c_val = cf['diffraction'][j*nmu + mid]
                    m_abs = abs(m_val)
                    c_abs = abs(c_val)
                    if m_abs > 1e-6:
                        ratio = c_abs / (2 * m_abs)
                    else:
                        ratio = float('nan')
                    print(f"  {name:>14s}  {m_abs:12.4f}  {c_abs:12.4f}  {ratio:14.6f}")
            
            if len(mf['fk']) >= 3*nmu and len(cf['fk']) >= 3*nmu:
                print(f"\n  FK at mu=0 (index {mid}):")
                print(f"  {'Component':>14s}  {'|mono|':>12s}  {'|cat|':>12s}  {'|cat|/2|mono|':>14s}")
                for j, name in enumerate(['sway', 'heave', 'roll']):
                    m_val = mf['fk'][j*nmu + mid]
                    c_val = cf['fk'][j*nmu + mid]
                    m_abs = abs(m_val)
                    c_abs = abs(c_val)
                    if m_abs > 1e-6:
                        ratio = c_abs / (2 * m_abs)
                    else:
                        ratio = float('nan')
                    print(f"  {name:>14s}  {m_abs:12.4f}  {c_abs:12.4f}  {ratio:14.6f}")


if __name__ == '__main__':
    mono_file = sys.argv[1] if len(sys.argv) > 1 else 'sectionresults_monohull'
    cat_file = sys.argv[2] if len(sys.argv) > 2 else 'sectionresults_cat500'
    compare_sections(mono_file, cat_file)
