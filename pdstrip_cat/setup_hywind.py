#!/usr/bin/env python3
"""
setup_hywind.py - Set up Nemoh case for OC3 Hywind spar floater

Copies the 978-panel mesh from the Nemoh test cases and creates a complete
Nemoh case directory with:
  - Proper Nemoh.cal for 36 headings (0-350 deg) and 40 uniform frequencies
  - Mesh.cal with correct COG position
  - Mechanics/ with OC3 Hywind properties from Jonkman (2010) / HAMS reference:
    - Inertia.dat: mass/inertia matrix about waterline origin
    - Kh.dat: hydrostatic restoring (backed up as _correct for run_nemoh.sh)
    - Km.dat: linearized mooring stiffness
    - Badd.dat: external linear damping

Physical properties from:
  Jonkman, J. (2010). Definition of the Floating System for Phase IV of OC3.
  NREL/TP-500-47535.

Cross-referenced with HAMS CertTest/HywindSpar reference data.

Usage:
    python setup_hywind.py [-o OUTPUT_DIR] [--no-qtf] [--n-omega N] [--n-beta N]
"""

import argparse
import numpy as np
import os
import shutil

# ============================================================
# OC3 Hywind spar physical properties
# ============================================================

# Environment
RHO = 1025.0         # kg/m^3
G = 9.81             # m/s^2
DEPTH = 320.0        # m (Nemoh test case value)

# Platform mass properties (total system: platform + tower + RNA)
MASS = 1.40179e7     # kg (from HAMS reference)

# COG position relative to SWL (Nemoh origin = waterline center)
# OC3 report: COG at 78.0 m below SWL for total system
XG = 0.0
YG = 0.0
ZG = -78.0           # m below waterline

# Moments of inertia ABOUT COG (from HAMS reference)
IXX_COG = 8.53882e9  # kg-m^2 (roll)
IYY_COG = 8.53882e9  # kg-m^2 (pitch)
IZZ_COG = 1.07485e10 # kg-m^2 (yaw)

# Mooring stiffness (linearized, from HAMS External Restoring Matrix)
# Jonkman (2010) / HAMS CertTest/HywindSpar/Input/Hydrostatic.in
KM = np.array([
    [ 4.11800e4,  0.0,         0.0,        0.0,        -2.82100e6,  0.0       ],
    [ 0.0,        4.11800e4,   0.0,        2.82100e6,   0.0,         0.0       ],
    [ 0.0,        0.0,         1.19400e4,  0.0,         0.0,         0.0       ],
    [ 0.0,        2.81600e6,   0.0,        3.11100e8,   0.0,         0.0       ],
    [-2.81600e6,  0.0,         0.0,        0.0,         3.11100e8,   0.0       ],
    [ 0.0,        0.0,         0.0,        0.0,         0.0,         1.15600e7 ],
])

# External linear damping (from HAMS reference)
BADD = np.array([
    [1.0e5, 0.0,   0.0,   0.0, 0.0,   0.0   ],
    [0.0,   1.0e5, 0.0,   0.0, 0.0,   0.0   ],
    [0.0,   0.0,   1.3e5, 0.0, 0.0,   0.0   ],
    [0.0,   0.0,   0.0,   0.0, 0.0,   0.0   ],
    [0.0,   0.0,   0.0,   0.0, 0.0,   0.0   ],
    [0.0,   0.0,   0.0,   0.0, 0.0,   1.3e7 ],
])


# ============================================================
# Nemoh mesh source
# ============================================================

NEMOH_TESTCASE = "/home/blofro/src/Nemoh/TestCases/11_QTF_OC3_Hywind"
MESH_978_DIR = os.path.join(NEMOH_TESTCASE, "mesh978_Floating")
MESH_4842_DIR = os.path.join(NEMOH_TESTCASE, "mesh4842_Floating")

MESH_978 = {
    'dir': MESH_978_DIR,
    'file': 'OC3_Hywind_978_Nem.dat',
    'npoints': 978,
    'npanels': 972,
    'isym': 0,  # full mesh, no symmetry
}

MESH_4842 = {
    'dir': MESH_4842_DIR,
    'file': 'OC3_hywind-4842_Nem.dat',
    'npoints': 4848,
    'npanels': 4842,
    'isym': 0,
}


# ============================================================
# OC3 Hywind geometry for hydrostatic computation
# ============================================================

def compute_hywind_hydrostatics():
    """
    Compute hydrostatic restoring coefficients for the OC3 Hywind spar.

    Geometry:
      - Upper column: r=3.25 m, z=0 to z=-4 m
      - Tapered transition: z=-4 to z=-12 m (r=3.25 to r=4.7 m)
      - Main cylinder: r=4.7 m, z=-12 to z=-120 m

    Returns dict with V_disp, zB, C33, C44, C55.
    """
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

    # Waterplane properties (at z=0, r=3.25m)
    r_wp = 3.25
    Awp = np.pi * r_wp**2
    Ix_wp = np.pi / 4 * r_wp**4  # second moment of waterplane area

    # Displaced volume and center of buoyancy by integration
    # Upper column: z=0 to z=-4m, r=3.25m
    z1 = np.linspace(0, -4, 50)
    r1 = np.full_like(z1, 3.25)

    # Taper: z=-4 to z=-12m, r=3.25 to r=4.7m (linear)
    z2 = np.linspace(-4, -12, 100)
    r2 = 3.25 + (4.7 - 3.25) * (z2 - (-4)) / (-12 - (-4))

    # Main cylinder: z=-12 to z=-120m, r=4.7m
    z3 = np.linspace(-12, -120, 200)
    r3 = np.full_like(z3, 4.7)

    # Combine
    z_all = np.concatenate([z1, z2[1:], z3[1:]])
    r_all = np.concatenate([r1, r2[1:], r3[1:]])

    # Volume: V = integral pi*r^2 dz (note: z goes negative, so use |dz|)
    A_sections = np.pi * r_all**2
    V_disp = -trapz(A_sections, z_all)  # negative because z decreasing

    # Center of buoyancy: zB = (1/V) * integral z * pi*r^2 dz
    zB = -trapz(z_all * A_sections, z_all) / V_disp

    # Hydrostatic stiffness
    # C33 = rho * g * Awp
    C33 = RHO * G * Awp

    # C44 = C55 = rho*g*Ix_wp + rho*g*V*(zB - zG)
    # For a spar with deep COG (zG=-78m) and zB~-62m:
    # (zB - zG) = (-62) - (-78) = +16m, so positive contribution
    C44 = RHO * G * Ix_wp + RHO * G * V_disp * (zB - ZG)
    C55 = C44  # axisymmetric

    return {
        'Awp': Awp,
        'Ix_wp': Ix_wp,
        'V_disp': V_disp,
        'zB': zB,
        'C33': C33,
        'C44': C44,
        'C55': C55,
    }


# ============================================================
# File writers
# ============================================================

def write_6x6(filepath, mat):
    """Write a 6x6 matrix file in Nemoh Mechanics format."""
    with open(filepath, 'w') as f:
        for i in range(6):
            row = "  ".join(f"{mat[i,j] + 0.0:15.6E}" for j in range(6))
            f.write(f" {row}\n")


def build_inertia_matrix():
    """
    Build 6x6 mass/inertia matrix about the waterline origin (0,0,0).

    The OC3 Hywind inertia is given about the COG. We use the parallel
    axis theorem to transfer to the origin:
        I_origin = I_cog + m * (|r|^2 * I3 - r (x) r)
    where r = (xG, yG, zG) is the COG position.
    """
    M = np.zeros((6, 6))

    # Translational mass
    M[0, 0] = MASS
    M[1, 1] = MASS
    M[2, 2] = MASS

    # Translation-rotation coupling (off-diagonal)
    M[0, 4] = MASS * ZG      # surge-pitch
    M[4, 0] = MASS * ZG
    M[0, 5] = -MASS * YG     # surge-yaw
    M[5, 0] = -MASS * YG
    M[1, 3] = -MASS * ZG     # sway-roll
    M[3, 1] = -MASS * ZG
    M[1, 5] = MASS * XG      # sway-yaw
    M[5, 1] = MASS * XG
    M[2, 3] = MASS * YG      # heave-roll
    M[3, 2] = MASS * YG
    M[2, 4] = -MASS * XG     # heave-pitch
    M[4, 2] = -MASS * XG

    # Rotational inertia about origin (parallel axis theorem)
    r2 = XG**2 + YG**2 + ZG**2
    M[3, 3] = IXX_COG + MASS * (r2 - XG**2)   # Ixx + m*(yG^2 + zG^2)
    M[4, 4] = IYY_COG + MASS * (r2 - YG**2)   # Iyy + m*(xG^2 + zG^2)
    M[5, 5] = IZZ_COG + MASS * (r2 - ZG**2)   # Izz + m*(xG^2 + yG^2)

    # Off-diagonal rotational coupling
    M[3, 4] = -MASS * XG * YG
    M[4, 3] = -MASS * XG * YG
    M[3, 5] = -MASS * XG * ZG
    M[5, 3] = -MASS * XG * ZG
    M[4, 5] = -MASS * YG * ZG
    M[5, 4] = -MASS * YG * ZG

    return M


def write_nemoh_cal(filepath, meshfile, npoints, npanels,
                    n_omega, omega_min, omega_max,
                    n_beta, beta_min, beta_max,
                    n_qtf_omega, qtf_omega_min, qtf_omega_max,
                    qtf_contrib=2, enable_qtf=True):
    """Write Nemoh.cal configuration file."""
    dashes = '-' * 114
    qtf_flag = 1 if enable_qtf else 0
    with open(filepath, 'w') as f:
        f.write(f"--- Environment {dashes}\n")
        f.write(f"{RHO:.1f}\t\t\t\t! RHO \t\t\t! KG/M**3 \t! Fluid specific volume\n")
        f.write(f"{G:.2f}\t\t\t\t! G\t\t\t\t! M/S**2\t! Gravity\n")
        f.write(f"{DEPTH:.1f}\t\t\t\t! DEPTH\t\t\t! M\t\t! Water depth\n")
        f.write(f"0.\t0.\t\t\t\t! XEFF YEFF\t\t! M\t\t! Wave measurement point\n")
        f.write(f"--- Description of floating bodies {dashes}\n")
        f.write(f"1\t\t\t\t\t! Number of bodies\n")
        f.write(f"--- Body 1 {dashes}\n")
        f.write(f"{meshfile}\t\t\t\t! Name of meshfile\n")
        f.write(f"{npoints} {npanels}\t\t\t\t\t! Number of points and number of panels\n")
        f.write(f"6\t\t\t\t\t\t! Number of degrees of freedom\n")
        f.write(f"1 1. 0.\t0. 0. 0. 0.\t\t\t\t! Surge\n")
        f.write(f"1 0. 1.\t0. 0. 0. 0.\t\t\t\t! Sway\n")
        f.write(f"1 0. 0. 1. 0. 0. 0.\t\t\t\t! Heave\n")
        f.write(f"2 1. 0. 0. 0. 0. 0.000000\t\t\t! Roll about a point\n")
        f.write(f"2 0. 1. 0. 0. 0. 0.000000\t\t\t! Pitch about a point\n")
        f.write(f"2 0. 0. 1. 0. 0. 0.000000\t\t\t! Yaw about a point\n")
        f.write(f"6\t\t\t\t\t\t! Number of resulting generalised forces\n")
        f.write(f"1 1. 0.\t0. 0. 0. 0.\t\t\t\t! Force in x direction\n")
        f.write(f"1 0. 1.\t0. 0. 0. 0.\t\t\t\t! Force in y direction\n")
        f.write(f"1 0. 0. 1. 0. 0. 0.\t\t\t\t! Force in z direction\n")
        f.write(f"2 1. 0. 0. 0. 0. 0.000000\t\t\t! Moment about x\n")
        f.write(f"2 0. 1. 0. 0. 0. 0.000000\t\t\t! Moment about y\n")
        f.write(f"2 0. 0. 1. 0. 0. 0.000000\t\t\t! Moment about z\n")
        f.write(f"0\t\t\t\t\t\t! Number of lines of additional information\n")
        f.write(f"--- Load cases to be solved {dashes}\n")
        f.write(f"1 {n_omega}\t{omega_min:.4f}\t{omega_max:.4f}\t\t\t! Freq type 1=rad/s, N, Min, Max\n")
        f.write(f"{n_beta}\t{beta_min:.6f}\t{beta_max:.6f}\t\t\t! N_beta, beta_min, beta_max (degrees)\n")
        f.write(f"--- Post processing {dashes}\n")
        f.write(f"0\t0.1\t10.\t\t\t! IRF (0=no), dt, duration\n")
        f.write(f"0\t\t\t\t\t! Show pressure\n")
        f.write(f"0\t0.\t180.\t\t\t! Kochin function: N_theta, min, max\n")
        f.write(f"0\t50\t400.\t400.\t\t! Free surface: Nx, Ny, Lx, Ly\n")
        f.write(f"1\t\t\t\t\t! RAO (1=calculate)\n")
        f.write(f"1\t\t\t\t\t! output freq type 1=rad/s\n")
        f.write(f"--- QTF{dashes}\n")
        f.write(f"{qtf_flag}\t\t\t\t\t! QTF flag (1=enable)\n")
        if enable_qtf:
            f.write(f"{n_qtf_omega}\t{qtf_omega_min:.4f}\t{qtf_omega_max:.4f}\t\t! N_omega_QTF, min, max\n")
            f.write(f"0\t\t\t\t\t! 0=unidirectional, 1=bidirectional\n")
            f.write(f"{qtf_contrib}\t\t\t\t\t! Contributing terms: 2=DUOK+HASBO\n")
            f.write(f"NA\t\t\t\t\t! FS mesh file (NA if not full QTF)\n")
            f.write(f"0\t0\t0\t\t\t! FS QTF params (not used for terms<=2)\n")
            f.write(f"0\t\t\t\t\t! Hydrostatic quadratic terms\n")
            f.write(f"1\t\t\t\t\t! Output freq type 1=rad/s\n")
            f.write(f"1\t\t\t\t\t! Include DUOK in total QTFs\n")
            f.write(f"1\t\t\t\t\t! Include HASBO in total QTFs\n")
            f.write(f"0\t\t\t\t\t! Include HASFS+ASYMP in total QTFs\n")
        else:
            f.write(f"40\t0.0500\t2.0000\t\t! (QTF disabled - placeholder values)\n")
            f.write(f"0\n")
            f.write(f"2\n")
            f.write(f"NA\n")
            f.write(f"0\t0\t0\n")
            f.write(f"0\n")
            f.write(f"1\n")
            f.write(f"1\n")
            f.write(f"1\n")
            f.write(f"0\n")
        f.write(f"{dashes}\n")


def write_mesh_cal(filepath, meshfile, isym, npanels):
    """Write Mesh.cal for hydrosCal."""
    with open(filepath, 'w') as f:
        meshbase = meshfile.replace('.dat', '')
        f.write(f"{meshbase}\n")
        f.write(f"{isym}\n")
        f.write(f" 0. 0. \n")
        f.write(f" {XG:.6f} {YG:.6f} {ZG:.6f} \n")
        f.write(f"{npanels}\n")
        f.write(f" 2 \n")
        f.write(f" 0. \n")
        f.write(f" 1.\n")
        f.write(f"{RHO:.1f}\n")
        f.write(f"{G:.2f}\n")


def write_input_solver(filepath):
    """Write input_solver.txt - use GMRES solver."""
    with open(filepath, 'w') as f:
        f.write("2\t\t\t\t! Gauss quadrature N^2, specify N=[1,4]\n")
        f.write("0.001\t\t\t! eps_zmin\n")
        f.write("2\t\t\t\t! 0=GAUSS ELIM, 1=LU DECOMP, 2=GMRES\n")
        f.write("10 1e-5 1000\t! GMRES params (restart, tol, maxiter)\n")


# ============================================================
# Main setup
# ============================================================

def setup_case(args):
    """Create the complete Nemoh case directory."""
    outdir = os.path.abspath(args.output)
    os.makedirs(outdir, exist_ok=True)

    # Select mesh
    if args.fine_mesh:
        mesh = MESH_4842
        print(f"Using fine mesh: {mesh['file']} ({mesh['npanels']} panels)")
    else:
        mesh = MESH_978
        print(f"Using coarse mesh: {mesh['file']} ({mesh['npanels']} panels)")

    print(f"Output directory: {outdir}")
    print()

    # --- Copy mesh file ---
    src_mesh = os.path.join(mesh['dir'], mesh['file'])
    dst_mesh = os.path.join(outdir, mesh['file'])
    if not os.path.exists(src_mesh):
        print(f"ERROR: mesh file not found: {src_mesh}")
        return 1
    shutil.copy2(src_mesh, dst_mesh)
    print(f"  Mesh: {mesh['file']} -> {outdir}")

    # Also copy info file if it exists
    info_file = mesh['file'].replace('.dat', '_info.dat')
    src_info = os.path.join(mesh['dir'], info_file)
    if os.path.exists(src_info):
        shutil.copy2(src_info, os.path.join(outdir, info_file))

    # --- Write Nemoh.cal ---
    write_nemoh_cal(
        os.path.join(outdir, "Nemoh.cal"),
        mesh['file'], mesh['npoints'], mesh['npanels'],
        n_omega=args.n_omega, omega_min=args.omega_min, omega_max=args.omega_max,
        n_beta=args.n_beta, beta_min=args.beta_min, beta_max=args.beta_max,
        n_qtf_omega=args.n_qtf_omega or args.n_omega,
        qtf_omega_min=args.qtf_omega_min or args.omega_min,
        qtf_omega_max=args.qtf_omega_max or args.omega_max,
        qtf_contrib=args.qtf_contrib,
        enable_qtf=not args.no_qtf,
    )
    print(f"  Nemoh.cal written")

    # --- Write Mesh.cal ---
    write_mesh_cal(
        os.path.join(outdir, "Mesh.cal"),
        mesh['file'], mesh['isym'], mesh['npanels'],
    )
    print(f"  Mesh.cal written (COG at [{XG}, {YG}, {ZG}] m)")

    # --- Write input_solver.txt ---
    write_input_solver(os.path.join(outdir, "input_solver.txt"))
    print(f"  input_solver.txt written")

    # --- Compute hydrostatics ---
    hydro = compute_hywind_hydrostatics()
    print(f"\n  Hydrostatic properties (computed from OC3 geometry):")
    print(f"    V_displaced = {hydro['V_disp']:.1f} m^3")
    print(f"    zB = {hydro['zB']:.2f} m (center of buoyancy)")
    print(f"    Awp = {hydro['Awp']:.2f} m^2 (waterplane area)")
    print(f"    Ix_wp = {hydro['Ix_wp']:.2f} m^4 (waterplane 2nd moment)")
    print(f"    C33 = {hydro['C33']:.3e} N/m (heave)")
    print(f"    C44 = C55 = {hydro['C44']:.3e} N-m/rad (roll/pitch)")
    print(f"    zB - zG = {hydro['zB'] - ZG:.2f} m (positive = stable)")

    # --- Write Mechanics files ---
    mech_dir = os.path.join(outdir, "Mechanics")
    os.makedirs(mech_dir, exist_ok=True)

    # Inertia matrix (about waterline origin)
    M = build_inertia_matrix()
    inertia_path = os.path.join(mech_dir, "Inertia.dat")
    write_6x6(inertia_path, M)
    shutil.copy2(inertia_path, os.path.join(mech_dir, "Inertia_correct.dat"))
    print(f"\n  Inertia.dat + _correct (mass={MASS:.3e} kg, zG={ZG} m)")
    print(f"    I44_origin = {M[3,3]:.3e} kg-m^2")
    print(f"    I55_origin = {M[4,4]:.3e} kg-m^2")
    print(f"    I66_origin = {M[5,5]:.3e} kg-m^2")
    print(f"    M[0,4] = {M[0,4]:.3e} (surge-pitch coupling from COG offset)")
    print(f"    M[1,3] = {M[1,3]:.3e} (sway-roll coupling from COG offset)")

    # Hydrostatic stiffness
    # We write our computed values and back them up as _correct.
    # hydrosCal will overwrite Kh.dat, but run_nemoh.sh restores _correct.
    Kh = np.zeros((6, 6))
    Kh[2, 2] = hydro['C33']
    Kh[3, 3] = hydro['C44']
    Kh[4, 4] = hydro['C55']

    kh_path = os.path.join(mech_dir, "Kh.dat")
    write_6x6(kh_path, Kh)
    shutil.copy2(kh_path, os.path.join(mech_dir, "Kh_correct.dat"))
    print(f"\n  Kh.dat + _correct written (hydrostatic restoring)")

    # Mooring stiffness
    km_path = os.path.join(mech_dir, "Km.dat")
    write_6x6(km_path, KM)
    print(f"  Km.dat written (mooring stiffness)")
    print(f"    K_surge = {KM[0,0]:.1f} N/m")
    print(f"    K_sway  = {KM[1,1]:.1f} N/m")
    print(f"    K_heave = {KM[2,2]:.1f} N/m")
    print(f"    K_roll  = {KM[3,3]:.3e} N-m/rad")
    print(f"    K_pitch = {KM[4,4]:.3e} N-m/rad")
    print(f"    K_yaw   = {KM[5,5]:.3e} N-m/rad")

    # External damping
    badd_path = os.path.join(mech_dir, "Badd.dat")
    write_6x6(badd_path, BADD)
    print(f"  Badd.dat written (external damping)")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"OC3 Hywind Nemoh case ready: {outdir}")
    print(f"{'='*60}")
    print(f"  Mesh: {mesh['file']} ({mesh['npanels']} panels, Isym={mesh['isym']})")
    print(f"  Frequencies: {args.n_omega}, w=[{args.omega_min:.4f}, {args.omega_max:.4f}] rad/s")
    print(f"  Headings: {args.n_beta}, beta=[{args.beta_min:.1f}, {args.beta_max:.1f}] deg")
    print(f"  QTF: {'enabled' if not args.no_qtf else 'disabled'}")
    print(f"  Depth: {DEPTH} m")
    print(f"\nTo run:")
    print(f"  ./run_nemoh.sh -v {outdir}")
    print(f"\nTo export results:")
    print(f"  python export_nemoh.py {outdir} -o hywind_results.dat")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Set up Nemoh case for OC3 Hywind spar',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('-o', '--output', default='hywind_nemoh',
                        help='Output case directory (default: hywind_nemoh)')
    parser.add_argument('--fine-mesh', action='store_true',
                        help='Use 4842-panel mesh instead of 978-panel')

    # Frequency settings
    parser.add_argument('--n-omega', type=int, default=40,
                        help='Number of frequencies (default: 40)')
    parser.add_argument('--omega-min', type=float, default=0.05,
                        help='Min frequency [rad/s] (default: 0.05)')
    parser.add_argument('--omega-max', type=float, default=2.0,
                        help='Max frequency [rad/s] (default: 2.0)')

    # Heading settings
    parser.add_argument('--n-beta', type=int, default=36,
                        help='Number of headings (default: 36)')
    parser.add_argument('--beta-min', type=float, default=0.0,
                        help='Min heading [deg] (default: 0)')
    parser.add_argument('--beta-max', type=float, default=350.0,
                        help='Max heading [deg] (default: 350)')

    # QTF settings
    parser.add_argument('--no-qtf', action='store_true',
                        help='Disable QTF computation')
    parser.add_argument('--n-qtf-omega', type=int, default=None,
                        help='Number of QTF frequencies (default: same as --n-omega)')
    parser.add_argument('--qtf-omega-min', type=float, default=None,
                        help='QTF min frequency (default: same as --omega-min)')
    parser.add_argument('--qtf-omega-max', type=float, default=None,
                        help='QTF max frequency (default: same as --omega-max)')
    parser.add_argument('--qtf-contrib', type=int, default=2,
                        help='QTF contributing terms: 2=DUOK+HASBO (default: 2)')

    args = parser.parse_args()
    return setup_case(args)


if __name__ == '__main__':
    exit(main())
