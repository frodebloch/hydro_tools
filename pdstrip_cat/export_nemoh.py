#!/usr/bin/env python3
"""
export_nemoh.py — Export Nemoh results to PDStrip standard TSV format.

Reads Nemoh output files (Motion/RAO.tec and results/QTF/QTFM_DUOK.dat)
and produces a pdstrip.dat tab-separated file in the standard 20-column
format used by the brucon/PDStrip toolchain.

Coordinate convention mapping (applied automatically):
  - Headings:  Nemoh 180°=head seas  →  PDStrip 0°=head seas  (angle = 180 - beta)
  - Sway/Roll/Yaw signs negated:  Nemoh y=port  →  PDStrip y=starboard
  - Heave sign negated:  Nemoh z=up  →  PDStrip z=down
  - Rotational RAOs divided by wavenumber k to match PDStrip "Rotation/k" convention

Output columns (20, tab-separated):
  freq  enc  angle  speed  surge_r  surge_i  sway_r  sway_i  heave_r  heave_i
  roll_r  roll_i  pitch_r  pitch_i  yaw_r  yaw_i  surge_d  sway_d  yaw_d  roll_d

Usage:
  export_nemoh.py CASE_DIR [-o OUTPUT] [--g 9.81] [--no-qtf]
"""

import argparse
import math
import os
import re
import sys


def parse_rao_tec(filepath):
    """Parse Nemoh Motion/RAO.tec file.

    Returns a list of dicts, each with keys:
      omega, beta_deg, amp[0..5], phase_deg[0..5]
    where DOF indices 0-5 = surge,sway,heave,roll,pitch,yaw.
    Amplitudes: translations in m/m, rotations in deg/m.
    Phases in degrees.
    """
    records = []
    beta_deg = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Header line — skip
            if line.startswith('VARIABLES'):
                continue

            # Zone header: extract beta
            if line.startswith('ZONE'):
                m = re.search(r'beta=\s*([\d.+-]+)', line, re.IGNORECASE)
                if m:
                    beta_deg = float(m.group(1))
                continue

            # Data line: 13 columns
            parts = line.split()
            if len(parts) != 13:
                continue

            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue

            omega = vals[0]
            amp = vals[1:7]       # |X|, |Y|, |Z|, |phi|, |theta|, |psi|
            phase = vals[7:13]    # ang(x), ang(y), ang(z), ang(phi), ang(theta), ang(psi)

            records.append({
                'omega': omega,
                'beta_deg': beta_deg,
                'amp': amp,
                'phase_deg': phase,
            })

    return records


def parse_qtfm_duok(filepath):
    """Parse Nemoh results/QTF/QTFM_DUOK.dat file.

    Extracts diagonal entries (w1 ≈ w2) for mean drift forces.

    Returns a dict keyed by (omega_round3, beta_rad_round3, dof) → Re(QTF).
    Keys are rounded to 3 decimal places to match Nemoh's output precision.
    The QTF values are dimensional: N/m² for forces, N·m/m² for moments.
    """
    drift = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('w1'):
                continue

            parts = line.split()
            if len(parts) != 7:
                continue

            try:
                w1 = float(parts[0])
                w2 = float(parts[1])
                beta1 = float(parts[2])
                beta2 = float(parts[3])
                dof = int(parts[4])
                re_qtf = float(parts[5])
            except (ValueError, IndexError):
                continue

            # Only diagonal entries (mean drift)
            if abs(w1 - w2) > 1e-4:
                continue
            if abs(beta1 - beta2) > 1e-4:
                continue

            # Round to 3 decimals — matches Nemoh's output precision for beta
            omega = round(w1, 3)
            beta_rad = round(beta1, 3)
            drift[(omega, beta_rad, dof)] = re_qtf

    return drift


def _build_qtf_lookup(drift_data):
    """Build a nearest-neighbor lookup for QTF data.

    The QTF and RAO frequency grids may differ. This function builds
    sorted arrays for efficient nearest-match lookups with tolerance.

    Returns (omega_arr, beta_arr, lookup_dict) where lookup_dict maps
    (omega_idx, beta_idx, dof) → Re(QTF).
    """
    if not drift_data:
        return [], [], {}

    # Collect unique omegas and betas from QTF keys
    qtf_omegas = sorted(set(k[0] for k in drift_data.keys()))
    qtf_betas = sorted(set(k[1] for k in drift_data.keys()))

    # Build index mapping: QTF key → (omega_idx, beta_idx)
    omega_idx = {o: i for i, o in enumerate(qtf_omegas)}
    beta_idx = {b: i for i, b in enumerate(qtf_betas)}

    lookup = {}
    for (o, b, dof), val in drift_data.items():
        oi = omega_idx[o]
        bi = beta_idx[b]
        lookup[(oi, bi, dof)] = val

    return qtf_omegas, qtf_betas, lookup


def _find_nearest_idx(arr, val, tol):
    """Find the index of the nearest value in sorted array within tolerance.

    Returns the index, or -1 if no value is within tolerance.
    Uses binary search for efficiency.
    """
    if not arr:
        return -1

    # Binary search for nearest
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < val:
            lo = mid + 1
        else:
            hi = mid

    # Check lo and lo-1
    best = lo
    if lo > 0 and abs(arr[lo - 1] - val) < abs(arr[lo] - val):
        best = lo - 1

    if abs(arr[best] - val) <= tol:
        return best
    return -1


def convert_and_write(rao_records, drift_data, output_path, g=9.81):
    """Convert Nemoh data to PDStrip convention and write TSV.

    Sign conventions applied:
      - Heading: angle = 180 - beta_nemoh
      - Sway (DOF 2): negate Re/Im   (y-flip: port → starboard)
      - Heave (DOF 3): negate Re/Im   (z-flip: up → down)
      - Roll (DOF 4): negate Re/Im   (coupled with y-flip)
      - Yaw (DOF 6): negate Re/Im   (coupled with z-flip)
      - Drift sway/roll/yaw: negate   (same y/z-flip)
      - Rotational RAOs: convert deg→rad, divide by k

    QTF frequency matching uses nearest-neighbor with 5% tolerance,
    allowing different frequency grids for RAO and QTF solvers.
    """
    header = [
        'freq', 'enc', 'angle', 'speed',
        'surge_r', 'surge_i', 'sway_r', 'sway_i',
        'heave_r', 'heave_i', 'roll_r', 'roll_i',
        'pitch_r', 'pitch_i', 'yaw_r', 'yaw_i',
        'surge_d', 'sway_d', 'yaw_d', 'roll_d',
    ]

    # Build QTF lookup
    qtf_omegas, qtf_betas, qtf_lookup = _build_qtf_lookup(drift_data)

    rows = []
    n_qtf_found = 0
    n_qtf_missing = 0

    for rec in rao_records:
        omega = rec['omega']
        beta_deg = rec['beta_deg']
        amp = rec['amp']
        phase_deg = rec['phase_deg']

        # Wavenumber (deep water dispersion)
        k = omega * omega / g

        # Map heading: PDStrip angle = 180 - Nemoh beta
        angle = 180.0 - beta_deg

        # Encounter frequency = wave frequency at zero speed
        enc = omega
        speed = 0.0

        # Convert amplitude + phase(deg) to Re + Im
        def to_re_im(a, p_deg):
            p_rad = p_deg * math.pi / 180.0
            return a * math.cos(p_rad), a * math.sin(p_rad)

        # --- Translational RAOs (m/m) ---
        surge_r, surge_i = to_re_im(amp[0], phase_deg[0])
        sway_r, sway_i = to_re_im(amp[1], phase_deg[1])
        heave_r, heave_i = to_re_im(amp[2], phase_deg[2])

        # Negate sway (y-flip) and heave (z-flip)
        sway_r, sway_i = -sway_r, -sway_i
        heave_r, heave_i = -heave_r, -heave_i

        # --- Rotational RAOs (deg/m → rad/m → rad/m/k = dimensionless) ---
        # Convert degrees to radians, then divide by k
        deg2rad = math.pi / 180.0
        if k > 1e-6:
            roll_amp_over_k = amp[3] * deg2rad / k
            pitch_amp_over_k = amp[4] * deg2rad / k
            yaw_amp_over_k = amp[5] * deg2rad / k
        else:
            # Very low frequency — k≈0, Rotation/k would blow up
            roll_amp_over_k = 0.0
            pitch_amp_over_k = 0.0
            yaw_amp_over_k = 0.0

        roll_r, roll_i = to_re_im(roll_amp_over_k, phase_deg[3])
        pitch_r, pitch_i = to_re_im(pitch_amp_over_k, phase_deg[4])
        yaw_r, yaw_i = to_re_im(yaw_amp_over_k, phase_deg[5])

        # Negate roll (y-flip) and yaw (z-flip)
        roll_r, roll_i = -roll_r, -roll_i
        yaw_r, yaw_i = -yaw_r, -yaw_i

        # --- Drift forces from QTF diagonal ---
        # Find nearest QTF omega and beta within tolerance
        beta_rad = beta_deg * math.pi / 180.0
        omega_tol = max(0.005, omega * 0.05)  # 5% or 0.005, whichever is larger
        beta_tol = 0.01  # ~0.6 degrees

        oi = _find_nearest_idx(qtf_omegas, omega, omega_tol)
        bi = _find_nearest_idx(qtf_betas, beta_rad, beta_tol)

        if oi >= 0 and bi >= 0 and (oi, bi, 1) in qtf_lookup:
            n_qtf_found += 1
            surge_d = qtf_lookup[(oi, bi, 1)]
            sway_d = -qtf_lookup.get((oi, bi, 2), 0.0)   # negate (y-flip)
            roll_d = -qtf_lookup.get((oi, bi, 4), 0.0)    # negate (y-flip)
            yaw_d = -qtf_lookup.get((oi, bi, 6), 0.0)     # negate (z-flip)
        else:
            n_qtf_missing += 1
            surge_d = 0.0
            sway_d = 0.0
            roll_d = 0.0
            yaw_d = 0.0

        rows.append([
            omega, enc, angle, speed,
            surge_r, surge_i, sway_r, sway_i,
            heave_r, heave_i, roll_r, roll_i,
            pitch_r, pitch_i, yaw_r, yaw_i,
            surge_d, sway_d, yaw_d, roll_d,
        ])

    # Sort by angle (primary), then frequency (secondary)
    rows.sort(key=lambda r: (r[2], r[0]))

    # Write TSV
    with open(output_path, 'w') as f:
        f.write('\t'.join(header) + '\n')
        for row in rows:
            parts = []
            for i, x in enumerate(row):
                if i == 0:      # freq
                    parts.append(f'{x:.3f}')
                elif i == 1:    # enc
                    parts.append(f'{x:.3f}')
                elif i == 2:    # angle
                    parts.append(f'{x:.1f}')
                elif i == 3:    # speed
                    parts.append(f'{x:.3f}')
                elif i >= 16:   # drift forces
                    parts.append(f'{x:.5e}')
                else:           # RAO Re/Im
                    if abs(x) < 5e-4:
                        parts.append('0.000')
                    else:
                        parts.append(f'{x:.3f}')
            f.write('\t'.join(parts) + '\n')

    return len(rows), n_qtf_found, n_qtf_missing


def main():
    parser = argparse.ArgumentParser(
        description='Export Nemoh results to PDStrip standard TSV format.',
        epilog='Reads Motion/RAO.tec and results/QTF/QTFM_DUOK.dat from CASE_DIR.',
    )
    parser.add_argument('case_dir', metavar='CASE_DIR',
                        help='Path to a Nemoh case directory')
    parser.add_argument('-o', '--output', default=None,
                        help='Output TSV file path (default: CASE_DIR/pdstrip.dat)')
    parser.add_argument('--g', type=float, default=9.81,
                        help='Gravitational acceleration (default: 9.81)')
    parser.add_argument('--no-qtf', action='store_true',
                        help='Skip QTF reading; output zeros for drift columns')

    args = parser.parse_args()

    case_dir = args.case_dir
    if not os.path.isdir(case_dir):
        print(f'Error: {case_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    # Output path
    output_path = args.output or os.path.join(case_dir, 'pdstrip.dat')

    # --- Parse RAO ---
    rao_path = os.path.join(case_dir, 'Motion', 'RAO.tec')
    if not os.path.isfile(rao_path):
        print(f'Error: RAO file not found: {rao_path}', file=sys.stderr)
        sys.exit(1)

    print(f'Reading RAOs from {rao_path}')
    rao_records = parse_rao_tec(rao_path)
    if not rao_records:
        print('Error: No RAO data found in RAO.tec', file=sys.stderr)
        sys.exit(1)

    # Count unique frequencies and headings
    omegas = sorted(set(r['omega'] for r in rao_records))
    betas = sorted(set(r['beta_deg'] for r in rao_records))
    print(f'  {len(omegas)} frequencies: {omegas[0]:.3f} - {omegas[-1]:.3f} rad/s')
    print(f'  {len(betas)} headings: {betas[0]:.1f} - {betas[-1]:.1f} deg (Nemoh convention)')

    # --- Parse QTF ---
    drift_data = {}
    qtf_path = os.path.join(case_dir, 'results', 'QTF', 'QTFM_DUOK.dat')

    if args.no_qtf:
        print('QTF reading skipped (--no-qtf). Drift columns will be zero.')
    elif os.path.isfile(qtf_path):
        print(f'Reading mean drift QTF from {qtf_path}')
        drift_data = parse_qtfm_duok(qtf_path)
        n_diag = len(set((k[0], k[1]) for k in drift_data.keys()))
        print(f'  {n_diag} diagonal (omega, beta) entries found')
    else:
        print(f'Warning: QTF file not found: {qtf_path}', file=sys.stderr)
        print('  Drift columns will be zero.', file=sys.stderr)

    # --- Convert and write ---
    print(f'Writing {output_path}')
    n_rows, n_qtf_found, n_qtf_missing = convert_and_write(
        rao_records, drift_data, output_path, g=args.g,
    )

    # Map headings to PDStrip convention for display
    pdstrip_angles = sorted(set(180.0 - b for b in betas))
    print(f'  {n_rows} rows ({len(omegas)} freq x {len(betas)} headings)')
    print(f'  PDStrip heading range: {pdstrip_angles[0]:.1f} to {pdstrip_angles[-1]:.1f} deg')
    if drift_data:
        print(f'  QTF matched: {n_qtf_found}, missing: {n_qtf_missing}')
    print('Done.')


if __name__ == '__main__':
    main()
