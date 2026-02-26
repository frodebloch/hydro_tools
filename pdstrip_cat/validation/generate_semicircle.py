#!/usr/bin/env python3
"""
Generate pdstrip geometry and input files for a semicircular barge
with a variable number of sections.

Usage: python generate_semicircle.py [nsections] [output_dir]
  nsections: number of equally-spaced sections (default: 5)
  output_dir: directory to write files to (default: current directory)
"""

import sys
import os
import numpy as np

# Barge parameters (must match validation setup)
R = 1.0          # radius [m]
L = 20.0         # length [m]
rho = 1025.0     # water density [kg/m^3]
g = 9.81         # gravity [m/s^2]
npoints = 21     # offset points per section

# Parse arguments
nsections = int(sys.argv[1]) if len(sys.argv) > 1 else 5
output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

os.makedirs(output_dir, exist_ok=True)

# Section x-positions: equally spaced from -L/2 to +L/2
x_positions = np.linspace(-L/2, L/2, nsections)

# Offset points: semicircle from y=-R (port waterline) to y=+R (starboard waterline)
# going through the keel at y=0, z=-R
# Parameterize by angle theta from -pi/2 to +pi/2
theta = np.linspace(np.pi, 0, npoints)  # from port (y=-R) to starboard (y=+R)
y_offsets = R * np.cos(theta)    # y: -R to +R
z_offsets = -R * np.sin(theta)   # z: 0 to -R to 0 (keel at bottom)

# Clean up floating point noise at endpoints
y_offsets = np.round(y_offsets, 10)
z_offsets = np.round(z_offsets, 10)
# Ensure exact zeros at waterline endpoints
z_offsets[0] = 0.0
z_offsets[-1] = 0.0

# Write geometry file
geom_file = os.path.join(output_dir, "geomet_semicircle.out")
with open(geom_file, 'w') as f:
    f.write(f"  {nsections} f    {R:.3f}\n")
    for x in x_positions:
        f.write(f"  {x:10.3f} {npoints} 0\n")
        # y-offsets line
        y_str = "  " + "  ".join(f"{y:8.4f}" for y in y_offsets)
        f.write(y_str + "\n")
        # z-offsets line
        z_str = "  " + "  ".join(f"{z:8.4f}" for z in z_offsets)
        f.write(z_str + "\n")
    f.write("\n")

# Compute mass properties
volume = np.pi * R**2 / 2 * L
mass = rho * volume   # 32201.3 kg
zcg = -4 * R / (3 * np.pi)  # -0.4244 m

# pdstrip input uses radii of gyration SQUARED (k² in m²), not moments of inertia.
# Use standard estimates: kxx = 0.4 * beam, kyy = kzz = 0.25 * L
# For beam = 2*R = 2m, L = 20m:
#   kxx² = (0.4 * 2)² = 0.64 m²
#   kyy² = kzz² = (0.25 * 20)² = 25.0 m²
kxx_sq = (0.4 * 2 * R)**2   # 0.64 m²
kyy_sq = (0.25 * L)**2      # 25.0 m²
kzz_sq = (0.25 * L)**2      # 25.0 m²

# Write input file
inp_file = os.path.join(output_dir, "pdstrip.inp")
with open(inp_file, 'w') as f:
    f.write(f"-{int(L)} t t f\n")
    f.write(f"Semi-circular cylinder barge R=1m L=20m {nsections} sections\n")
    f.write(f"{g} {rho} 0 -1e6 {zcg:.4f}\n")
    f.write(f"3 -90 0 90\n")
    f.write(f"geomet_semicircle.out\n")
    f.write(f"f 0.0          catamaran? hull center-to-CL distance\n")
    f.write(f"\n")
    f.write(f"f\n")
    f.write(f"{mass:.1f} 0.0 0.0 {zcg:.4f} {kxx_sq:.4f} {kyy_sq:.4f} {kzz_sq:.4f} 0.0 0.0 0.0\n")
    f.write(f"\n")
    # Flow separation flags: one per section
    iab_line = " ".join(["0"] * nsections) + "  flow separation"
    f.write(f"{iab_line}\n")
    f.write(f"\n")
    f.write(f"0.1 6.0       wave steepness; max wave height\n")
    # CD values: nse pairs of (cdy, cdz), written in lines of ~5 pairs
    cd_pairs = ["0.8 0.6"] * nsections
    # Write in groups of 5 pairs per line
    for i in range(0, nsections, 5):
        chunk = cd_pairs[i:i+5]
        f.write("  ".join(chunk) + "\n")
    f.write(f"\n")
    f.write(f"0       fin\n")
    f.write(f"\n")
    f.write(f"0       sails\n")
    f.write(f"\n")
    f.write(f"0 forces depending on motions\n")
    f.write(f"\n")
    f.write(f"0.0 0.0 0.0 0.0 0.0 suspended weight\n")
    f.write(f"\n")
    f.write(f"1  motion points\n")
    f.write(f"0.0 0.0 0.0\n")
    f.write(f"\n")
    f.write(f"15\n")
    f.write(f"3.0 4.0 5.0 6.0 8.0 10.0 13.0 17.0 22.0 28.0 35.0 45.0 55.0 70.0 90.0\n")
    f.write(f"\n")
    f.write(f"1\n")
    f.write(f"0.0 t\n")

print(f"Generated {nsections}-section semicircle barge:")
print(f"  Geometry: {geom_file}")
print(f"  Input:    {inp_file}")
print(f"  Section x-positions: {x_positions}")
