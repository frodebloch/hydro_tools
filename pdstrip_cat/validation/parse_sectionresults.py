#!/usr/bin/env python3
"""
Comprehensive validation: pdstrip vs Capytaine for semi-circular cylinder barge.
Compares section-level hydrodynamic coefficients (added mass, damping, excitation)
for both monohull and catamaran configurations.

Approach:
- pdstrip outputs section hydrodynamics in 'sectionresults' file.
  We parse these for the middle section (section 3, x=0).
  pdstrip's complex added mass: addedm = a + b/(i*omega)
  The file stores omega^2 * addedm, so:
    Re(omega^2 * addedm) = omega^2 * a
    Im(omega^2 * addedm) = -omega * b
  => a = Re / omega^2, b = -Im / omega

- Capytaine solves the full 3D problem. For a long prismatic hull,
  the results per unit length should match the 2D section results.
  total_added_mass = section_added_mass * L (approximately)
  So: a_section = A_total / L

Convention mapping (pdstrip internal → Capytaine):
  pdstrip internal: y positive to starboard, z positive downward
  Capytaine: y positive to port, z positive upward
  pdstrip DOF ordering: (1)=sway, (2)=heave, (3)=roll
  Capytaine DOFs: Surge(x), Sway(y), Heave(z), Roll, Pitch, Yaw

  Sign mapping for forces/added mass:
    pdstrip_sway ↔ -Capytaine_Sway  (opposite y convention)
    pdstrip_heave ↔ -Capytaine_Heave  (opposite z convention)
    pdstrip_roll ↔ Capytaine_Roll  (both flip → cancels)
  But magnitudes of diagonal terms should match regardless of sign convention.
"""

import numpy as np
import os
import sys

# ============================================================
# Parameters
# ============================================================
R = 1.0
L = 20.0
rho = 1025.0
g = 9.81

# ============================================================
# Parse pdstrip sectionresults
# ============================================================
def parse_sectionresults(filepath, section_idx=2):
    """
    Parse pdstrip sectionresults file.
    
    Returns dict with keys: omega, addedm_real[3,3,nfre], addedm_imag[3,3,nfre],
    diff[3,nmu,nfre], frkr[3,nmu,nfre], wangles[nmu]
    
    section_idx: 0-based index of which section to extract (default=2 = 3rd section at x=0)
    """
    with open(filepath, 'r') as f:
        text_line = f.readline()  # header text
        
        results = {}
        ise = 0
        
        while True:
            line = f.readline()
            if not line:
                break
            
            # Check if this is a section header: "nfre  x_position"
            parts = line.split()
            if len(parts) == 2:
                try:
                    nfre = int(parts[0])
                    xpos = float(parts[1])
                except ValueError:
                    continue
                
                omegas = []
                addedm_list = []  # will be [nfre][3][3] complex
                diff_list = []    # will be [nfre][3][nmu] complex
                frkr_list = []    # will be [nfre][3][nmu] complex
                wangles = None
                npres = 0
                
                for ifre in range(nfre):
                    # Frequency line: omega, nmu, angles
                    freq_line = f.readline()
                    freq_parts = freq_line.split()
                    omega = float(freq_parts[0])
                    nmu = int(freq_parts[1])
                    angles = [float(x) for x in freq_parts[2:2+nmu]]
                    
                    omegas.append(omega)
                    if wangles is None:
                        wangles = np.array(angles)
                    
                    # Read radiation data: 3 rows of 3 complex values = omega^2 * addedm
                    # But it's stored as one big line with complex numbers in (re,im) format
                    # Let's read enough tokens
                    rad_tokens = []
                    while len(rad_tokens) < 9 * 2:  # 9 complex = 18 real
                        rad_line = f.readline()
                        # Parse complex numbers in (re,im) format
                        rad_line = rad_line.replace('(', '').replace(')', '')
                        rad_tokens.extend(rad_line.split(','))
                        # Actually this isn't right - let me re-examine the format
                    
                    # Hmm, the format is tricky. Let me use a different approach.
                    # Read the raw complex data more carefully
                    pass
                
                if ise == section_idx:
                    results['x'] = xpos
                    results['nfre'] = nfre
                    # We'll fill these below
                    break
                
                ise += 1
    
    # The format is too complex for simple line-by-line parsing.
    # Let me use a token-based approach.
    return None


def parse_sectionresults_v2(filepath, section_idx=2):
    """
    Token-based parser for sectionresults.
    Complex numbers are stored as (real,imag) pairs.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace parentheses and commas to make it easier to parse
    # Complex numbers appear as (real,imag) — replace commas inside parens with spaces
    # and remove parens
    import re
    # Match complex number pattern: (number,number)
    def complex_repl(m):
        return m.group(1) + ' ' + m.group(2)
    
    content = re.sub(r'\(([^,]+),([^)]+)\)', complex_repl, content)
    
    tokens = content.split()
    pos = 0
    
    def next_token():
        nonlocal pos
        t = tokens[pos]
        pos += 1
        return t
    
    def next_float():
        return float(next_token())
    
    def next_int():
        return int(next_token())
    
    def next_complex():
        re_val = next_float()
        im_val = next_float()
        return complex(re_val, im_val)
    
    # Skip header text — it's the first line, which got tokenized
    # We need to find where the actual data starts
    # The header text was one line, followed by "nfre x_position"
    # Let's skip tokens until we find a plausible nfre (integer ~52)
    # Actually, let's be smarter: read line-by-line first to get the header
    
    lines = content.split('\n')
    # First line is text header
    # Then for each section: one line with "nfre x_position"
    # Then nfre frequency blocks
    
    # Restart with lines
    header = lines[0]
    line_idx = 1
    
    ise = 0
    while line_idx < len(lines):
        # Section header
        parts = lines[line_idx].split()
        line_idx += 1
        if len(parts) < 2:
            continue
        try:
            nfre = int(parts[0])
            xpos = float(parts[1])
        except (ValueError, IndexError):
            continue
        
        if ise != section_idx:
            # Skip this section entirely — we need to skip nfre frequency blocks
            # Each block has: 1 freq line + radiation lines + diffraction lines + FK lines + pressure lines
            # This is hard to count without parsing. Let's just parse everything.
            pass
        
        section_data = {
            'x': xpos, 'nfre': nfre,
            'omega': [], 'wangles': None, 'nmu': 0,
            'addedm': [],  # [nfre] × 3×3 complex (omega^2 * addedm)
            'diff': [],    # [nfre] × 3×nmu complex
            'frkr': [],    # [nfre] × 3×nmu complex
        }
        
        for ifre in range(nfre):
            # Collect all tokens for this frequency block
            # First line: omega nmu angle1 angle2 ...
            freq_parts = lines[line_idx].split()
            line_idx += 1
            omega = float(freq_parts[0])
            nmu = int(freq_parts[1])
            angles = [float(x) for x in freq_parts[2:2+nmu]]
            
            section_data['omega'].append(omega)
            if section_data['wangles'] is None:
                section_data['wangles'] = np.array(angles)
                section_data['nmu'] = nmu
            
            # Next: radiation data = 3 rows × 3 cols × complex
            # Plus off-diagonal cross-coupling to sway motion at different angles (3 × nmu complex)
            # Format from pdstrip: write(20,*)(om(i)**2*addedm(j,1:3,i),j=1,3)
            # This writes 9 complex values: addedm(1,1), addedm(1,2), addedm(1,3),
            #                                addedm(2,1), addedm(2,2), addedm(2,3),
            #                                addedm(3,1), addedm(3,2), addedm(3,3)
            # Complex format: (re,im)(re,im)... 
            # After our regex replacement: re im re im ...
            
            # Radiation: 9 complex = 18 float tokens
            rad_tokens = []
            while len(rad_tokens) < 18:
                rad_tokens.extend(lines[line_idx].split())
                line_idx += 1
            rad_tokens = [float(x) for x in rad_tokens[:18]]
            addedm = np.zeros((3, 3), dtype=complex)
            for j in range(3):
                for k in range(3):
                    idx = (j * 3 + k) * 2
                    addedm[j, k] = complex(rad_tokens[idx], rad_tokens[idx + 1])
            section_data['addedm'].append(addedm)
            
            # Diffraction: 3 × nmu complex = 6*nmu float tokens
            diff_tokens = []
            needed = 6 * nmu
            while len(diff_tokens) < needed:
                diff_tokens.extend(lines[line_idx].split())
                line_idx += 1
            diff_tokens = [float(x) for x in diff_tokens[:needed]]
            diff = np.zeros((3, nmu), dtype=complex)
            for j in range(3):
                for imu in range(nmu):
                    idx = (j * nmu + imu) * 2
                    diff[j, imu] = complex(diff_tokens[idx], diff_tokens[idx + 1])
            section_data['diff'].append(diff)
            
            # Froude-Krylov: 3 × nmu complex
            fk_tokens = []
            while len(fk_tokens) < needed:
                fk_tokens.extend(lines[line_idx].split())
                line_idx += 1
            fk_tokens = [float(x) for x in fk_tokens[:needed]]
            frkr = np.zeros((3, nmu), dtype=complex)
            for j in range(3):
                for imu in range(nmu):
                    idx = (j * nmu + imu) * 2
                    frkr[j, imu] = complex(fk_tokens[idx], fk_tokens[idx + 1])
            section_data['frkr'].append(frkr)
            
            # Pressure data: npres lines, each with index + (3+nmu) complex values
            # We need to detect if pressure data exists
            # Look at the next line — if it starts with an integer index (1-20), it's pressure
            # Actually from the write statement: if(npres>0) write(20,*)(ii,pr(ii,1:3+nmu,i),ii=1,npres)
            # This writes all npres pressure records in ONE write statement
            # Each record: integer ii, then (3+nmu) complex values
            # Total tokens per frequency: npres * (1 + 2*(3+nmu))
            # For npres=20, nmu=3: 20 * (1 + 2*6) = 20 * 13 = 260 tokens
            
            # Determine npres from context — we know it's 20 from the input
            npres = 20  # hardcoded for our test case
            if npres > 0:
                pres_needed = npres * (1 + 2 * (3 + nmu))
                pres_tokens = []
                while len(pres_tokens) < pres_needed:
                    pres_tokens.extend(lines[line_idx].split())
                    line_idx += 1
                # We don't need to parse pressure data for this comparison
        
        if ise == section_idx:
            section_data['omega'] = np.array(section_data['omega'])
            return section_data
        
        ise += 1
    
    return None


# ============================================================
# Quick test of the parser
# ============================================================
if __name__ == "__main__":
    sr_file = "/home/blofro/src/pdstrip_test/validation/run_mono/sectionresults"
    
    # Parse middle section (index 2, x=0)
    data = parse_sectionresults_v2(sr_file, section_idx=2)
    
    if data is None:
        print("Failed to parse sectionresults!")
        sys.exit(1)
    
    print(f"Section at x = {data['x']}")
    print(f"Number of frequencies: {data['nfre']}")
    print(f"Wave angles: {np.degrees(data['wangles'])} deg")
    print(f"Omega range: {data['omega'][0]:.4f} to {data['omega'][-1]:.4f}")
    
    # Extract added mass and damping
    # omega^2 * addedm = omega^2 * a - i * omega * b
    # a = Re(.) / omega^2
    # b = -Im(.) / omega
    print(f"\nSection added mass and damping (2D, per unit length):")
    print(f"{'nu':>8s} {'omega':>8s} {'a22':>12s} {'b22':>12s} {'a33':>12s} {'b33':>12s} {'a44':>12s} {'b44':>12s}")
    
    for i in range(0, data['nfre'], 5):
        omega = data['omega'][i]
        nu = omega**2 * R / g
        am = data['addedm'][i]
        
        # Diagonal terms
        a22 = am[0, 0].real / omega**2
        b22 = -am[0, 0].imag / omega
        a33 = am[1, 1].real / omega**2
        b33 = -am[1, 1].imag / omega
        a44 = am[2, 2].real / omega**2
        b44 = -am[2, 2].imag / omega
        
        print(f"{nu:8.3f} {omega:8.3f} {a22:12.4f} {b22:12.4f} {a33:12.4f} {b33:12.4f} {a44:12.4f} {b44:12.4f}")
    
    # Excitation forces at beam seas (mu = pi/2, index 2)
    print(f"\nExcitation force (diff + FK) at beam seas (mu=90°):")
    print(f"{'nu':>8s} {'omega':>8s} {'|F2|':>12s} {'|F3|':>12s} {'|F4|':>12s}")
    imu = 2  # 90 degrees
    for i in range(0, data['nfre'], 5):
        omega = data['omega'][i]
        nu = omega**2 * R / g
        exc = data['diff'][i][:, imu] + data['frkr'][i][:, imu]
        print(f"{nu:8.3f} {omega:8.3f} {abs(exc[0]):12.4f} {abs(exc[1]):12.4f} {abs(exc[2]):12.4f}")
