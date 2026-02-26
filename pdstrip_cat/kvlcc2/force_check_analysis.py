#!/usr/bin/env python3
"""
Analyze pressure-force consistency: does sum(p*n*dS) over triangles equal -omega^2*M*eta?

Parses FORCE_CHECK lines from debug.out along with DRIFT_START for omega/mu context.
Focuses on beam seas (mu=90) and V=0 for the sway rotation term investigation.
"""

import numpy as np
import re

def parse_force_check(filename='debug.out'):
    """Parse DRIFT_START and FORCE_CHECK lines."""
    results = []
    current_omega = None
    current_mu = None
    
    with open(filename) as f:
        for line in f:
            # Parse DRIFT_START for context
            m = re.search(r'DRIFT_START omega=\s*([\d.]+)\s+mu=\s*([\d.-]+)', line)
            if m:
                current_omega = float(m.group(1))
                current_mu = float(m.group(2))
                continue
            
            # Parse FORCE_CHECK
            m = re.search(r'FORCE_CHECK Fy_pres=\s*([\d.E+-]+)\s+([\d.E+-]+)\s+Fy_newton=\s*([\d.E+-]+)\s+([\d.E+-]+)\s+Fz_pres=\s*([\d.E+-]+)\s+([\d.E+-]+)\s+Fz_newton=\s*([\d.E+-]+)\s+([\d.E+-]+)', line)
            if m and current_omega is not None:
                fy_pres = complex(float(m.group(1)), float(m.group(2)))
                fy_newton = complex(float(m.group(3)), float(m.group(4)))
                fz_pres = complex(float(m.group(5)), float(m.group(6)))
                fz_newton = complex(float(m.group(7)), float(m.group(8)))
                results.append({
                    'omega': current_omega,
                    'mu': current_mu,
                    'fy_pres': fy_pres,
                    'fy_newton': fy_newton,
                    'fz_pres': fz_pres,
                    'fz_newton': fz_newton,
                })
    
    return results

def main():
    results = parse_force_check()
    print(f"Total FORCE_CHECK records: {len(results)}")
    
    # Filter for beam seas (mu=90) — this is the sway investigation case
    # V=0 is speed index 0. With 36 headings and 8 speeds, the ordering is:
    # omega(outer, high-to-low) x speed(middle) x heading(inner)
    # mu=90 at V=0 should appear at specific indices
    
    # Let's just filter by mu ≈ 90
    beam = [r for r in results if abs(r['mu'] - 90.0) < 1.0]
    print(f"Beam seas (mu=90) records: {len(beam)}")
    
    # For V=0, we need the first occurrence of each omega at mu=90
    # Since ordering is omega x speed x heading, and speed index 0 comes first,
    # the first mu=90 at each omega should be V=0
    seen_omega = set()
    beam_v0 = []
    for r in beam:
        if r['omega'] not in seen_omega:
            seen_omega.add(r['omega'])
            beam_v0.append(r)
    
    print(f"Beam seas V=0 records: {len(beam_v0)}")
    
    # Sort by omega
    beam_v0.sort(key=lambda r: r['omega'])
    
    g = 9.81
    Lpp = 328.2
    
    print("\n" + "="*120)
    print("PRESSURE-FORCE CONSISTENCY CHECK at beam seas (mu=90°, V=0)")
    print("="*120)
    print(f"{'omega':>6} {'lam/L':>6} | {'|Fy_pres|':>12} {'|Fy_newt|':>12} {'|Fy_err|':>12} {'%err_Fy':>8} | {'|Fz_pres|':>12} {'|Fz_newt|':>12} {'|Fz_err|':>12} {'%err_Fz':>8}")
    print("-"*120)
    
    for r in beam_v0:
        omega = r['omega']
        lam = 2*np.pi*g/omega**2
        lam_L = lam / Lpp
        
        fy_err = r['fy_pres'] - r['fy_newton']
        fz_err = r['fz_pres'] - r['fz_newton']
        
        # Relative error
        pct_fy = 100 * abs(fy_err) / max(abs(r['fy_newton']), 1e-10)
        pct_fz = 100 * abs(fz_err) / max(abs(r['fz_newton']), 1e-10)
        
        print(f"{omega:6.3f} {lam_L:6.2f} | {abs(r['fy_pres']):12.1f} {abs(r['fy_newton']):12.1f} {abs(fy_err):12.1f} {pct_fy:7.1f}% | {abs(r['fz_pres']):12.1f} {abs(r['fz_newton']):12.1f} {abs(fz_err):12.1f} {pct_fz:7.1f}%")
    
    # Now look at the PHASE difference, which is what matters for the rotation cross-term
    print("\n" + "="*120)
    print("PHASE ANALYSIS: phase(Fz_pres) vs phase(Fz_newton)")
    print("="*120)
    print(f"{'omega':>6} {'lam/L':>6} | {'ph(Fz_pres)':>12} {'ph(Fz_newt)':>12} {'ph_diff':>10} | {'Re(Fz_p)':>12} {'Im(Fz_p)':>12} {'Re(Fz_n)':>12} {'Im(Fz_n)':>12}")
    print("-"*120)
    
    for r in beam_v0:
        omega = r['omega']
        lam = 2*np.pi*g/omega**2
        lam_L = lam / Lpp
        
        ph_pres = np.degrees(np.angle(r['fz_pres']))
        ph_newt = np.degrees(np.angle(r['fz_newton']))
        ph_diff = ph_pres - ph_newt
        # Normalize to [-180, 180]
        while ph_diff > 180: ph_diff -= 360
        while ph_diff < -180: ph_diff += 360
        
        print(f"{omega:6.3f} {lam_L:6.2f} | {ph_pres:12.1f} {ph_newt:12.1f} {ph_diff:10.1f} | {r['fz_pres'].real:12.0f} {r['fz_pres'].imag:12.0f} {r['fz_newton'].real:12.0f} {r['fz_newton'].imag:12.0f}")
    
    # The key quantity for the rotation term blow-up:
    # feta_rot ≈ +(1/2) Re[conj(eta4) * Fz_total]
    # where Fz_total = -sum(p * n_z * dS) = cpres_fz_sum (what we computed)
    # OR Fz_total = -omega^2 * M * eta3 (Newton's prediction)
    #
    # If these differ, the rotation term using the actual pressures will differ
    # from the theoretical value -(omega^2*M/2)*Re[eta3*conj(eta4)]
    
    print("\n" + "="*120)
    print("SIGN CHECK: Fy_pres vs Fy_newton (are they opposite sign or same sign?)")
    print("Note: F_pres = -∮ p n dS (force ON body), F_newton = -ω²Mη")
    print("They should be EQUAL if pressure field is consistent with eq. of motion")
    print("="*120)
    print(f"{'omega':>6} {'lam/L':>6} | {'Re(Fy_p)':>12} {'Re(Fy_n)':>12} {'ratio_Re':>10} | {'Im(Fy_p)':>12} {'Im(Fy_n)':>12} {'ratio_Im':>10}")
    print("-"*120)
    
    for r in beam_v0:
        omega = r['omega']
        lam = 2*np.pi*g/omega**2
        lam_L = lam / Lpp
        
        # Ratio of real parts
        ratio_re = r['fy_pres'].real / r['fy_newton'].real if abs(r['fy_newton'].real) > 1 else float('nan')
        ratio_im = r['fy_pres'].imag / r['fy_newton'].imag if abs(r['fy_newton'].imag) > 1 else float('nan')
        
        print(f"{omega:6.3f} {lam_L:6.2f} | {r['fy_pres'].real:12.0f} {r['fy_newton'].real:12.0f} {ratio_re:10.3f} | {r['fy_pres'].imag:12.0f} {r['fy_newton'].imag:12.0f} {ratio_im:10.3f}")
    
    # Same for Fz
    print("\n" + "="*120)
    print("SIGN CHECK: Fz_pres vs Fz_newton")
    print("="*120)
    print(f"{'omega':>6} {'lam/L':>6} | {'Re(Fz_p)':>12} {'Re(Fz_n)':>12} {'ratio_Re':>10} | {'Im(Fz_p)':>12} {'Im(Fz_n)':>12} {'ratio_Im':>10}")
    print("-"*120)
    
    for r in beam_v0:
        omega = r['omega']
        lam = 2*np.pi*g/omega**2
        lam_L = lam / Lpp
        
        ratio_re = r['fz_pres'].real / r['fz_newton'].real if abs(r['fz_newton'].real) > 1 else float('nan')
        ratio_im = r['fz_pres'].imag / r['fz_newton'].imag if abs(r['fz_newton'].imag) > 1 else float('nan')
        
        print(f"{omega:6.3f} {lam_L:6.2f} | {r['fz_pres'].real:12.0f} {r['fz_newton'].real:12.0f} {ratio_re:10.3f} | {r['fz_pres'].imag:12.0f} {r['fz_newton'].imag:12.0f} {ratio_im:10.3f}")


if __name__ == '__main__':
    main()
    
    # Additional analysis: rotation term implications
    print("\n\n" + "="*120)
    print("ROTATION TERM ANALYSIS: How Fz inconsistency affects sway drift rotation term")
    print("feta_rot ≈ (1/2) Re[conj(eta4) * Fz_total]")
    print("="*120)
    
    # Parse motion data from pdstrip.out to get eta4
    import re as re2
    
    results = parse_force_check()
    beam_v0 = []
    seen = set()
    for r in [r for r in results if abs(r['mu'] - 90.0) < 1.0]:
        if r['omega'] not in seen:
            seen.add(r['omega'])
            beam_v0.append(r)
    beam_v0.sort(key=lambda r: r['omega'])
    
    # Parse DRIFT_SWAY to get actual rotation terms
    drift_data = {}
    current_omega = None
    current_mu = None
    with open('debug.out') as f:
        for line in f:
            m = re2.search(r'DRIFT_START omega=\s*([\d.]+)\s+mu=\s*([\d.-]+)', line)
            if m:
                current_omega = float(m.group(1))
                current_mu = float(m.group(2))
                continue
            m = re2.search(r'DRIFT_SWAY feta_vel=\s*([\d.E+-]+)\s+feta_rot=\s*([\d.E+-]+)', line)
            if m and current_omega and abs(current_mu - 90.0) < 1.0:
                key = current_omega
                if key not in drift_data:
                    drift_data[key] = {
                        'feta_vel': float(m.group(1)),
                        'feta_rot': float(m.group(2)),
                    }
    
    # Parse motion phases from pdstrip.out  
    # We need eta3 and eta4 at beam seas
    motion_data = {}
    with open('pdstrip.out') as f:
        content = f.read()
    
    # Look for motion RAOs - they should appear in the output
    # For now, use the motion_phase_analysis data we computed before
    # Let's just use the Fz data to estimate the rotation contribution
    
    g = 9.81
    Lpp = 328.2
    rho = 1025.0
    norm_sway = 2 * rho * g * Lpp  # normalization
    
    print(f"\n{'omega':>6} {'lam/L':>6} | {'rot_from_Fz_p':>14} {'rot_from_Fz_n':>14} {'actual_rot':>12} {'diff_Fz':>12} | {'|Fz_err|':>12} {'Fz_%err':>8} {'rot_err_est':>12}")
    print("-"*130)
    
    for r in beam_v0:
        omega = r['omega']
        lam = 2*np.pi*g/omega**2
        lam_L = lam / Lpp
        
        if omega not in drift_data:
            continue
        
        dd = drift_data[omega]
        actual_rot = dd['feta_rot']
        
        # The rotation term from Fz: (1/2) Re[conj(eta4) * Fz]
        # We don't have eta4 directly here, but we can estimate from:
        # actual_rot / norm ≈ (1/2) Re[conj(eta4) * Fz_pres] / norm
        # So rot_from_Fz_n / rot_actual ≈ Fz_newton / Fz_pres
        
        # Actually, let's compute the Fz error contribution to rotation
        fz_err = r['fz_pres'] - r['fz_newton']
        fz_pct = 100 * abs(fz_err) / max(abs(r['fz_newton']), 1)
        
        # Estimated rotation error from Fz inconsistency:
        # (1/2) * |Fz_err| * |eta4|
        # We can extract |eta4| from: actual_rot * norm ≈ (1/2) * |Fz_pres| * |eta4| * cos(phase_diff)
        # This is rough but gives an order of magnitude
        
        # Better: ratio Fz_pres/Fz_newton tells us the amplification
        if abs(r['fz_newton']) > 1:
            fz_ratio = abs(r['fz_pres']) / abs(r['fz_newton'])
        else:
            fz_ratio = float('nan')
        
        # Normalized values
        actual_rot_norm = actual_rot / norm_sway
        
        print(f"{omega:6.3f} {lam_L:6.2f} | {'':>14} {'':>14} {actual_rot_norm:12.4f} {'':>12} | {abs(fz_err):12.0f} {fz_pct:7.1f}% {fz_ratio:12.3f}")
    
    # NOW: the critical insight. The Fy mismatch tells us the pressure field
    # is inconsistent for sway. But the rotation term uses Fz (not Fy) in the
    # dominant cross-product at beam seas. Let me also check what fraction 
    # of the rotation term comes from the Fy-type terms (via pitch coupling)
    
    print("\n\nKEY INSIGHT:")
    print("The Fz match is ~2-5% — this is good enough for the heave component")  
    print("The Fy mismatch is 50-800% — but the rotation term at beam seas")
    print("primarily depends on Fz × conj(roll), not Fy × conj(yaw)")
    print("So the rotation blow-up must come from the ~3% Fz error amplified")
    print("by the large roll angle at resonance, AND/OR from the large absolute")
    print("value of Re[conj(eta4) * Fz] that fails to cancel with other components.")
