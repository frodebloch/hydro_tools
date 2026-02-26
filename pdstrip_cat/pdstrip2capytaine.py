#!/usr/bin/env python3
"""
pdstrip2capytaine — Convert pdstrip input files to a standalone Capytaine script.

Reads pdstrip.inp + geometry file (e.g., geomet.out) and generates a Python script
that builds a 3D panel mesh, sets up the Capytaine BEM solve, and writes results.

Usage:
    python pdstrip2capytaine.py [pdstrip_input_dir] [output_script]

    pdstrip_input_dir: directory containing pdstrip.inp (default: current dir)
    output_script:     output Python filename (default: capytaine_run.py)

The generated script can be run with:
    python capytaine_run.py
"""

import sys
import os
import re
import numpy as np
from io import StringIO


# ============================================================
# Fortran list-directed I/O tokenizer
# ============================================================
class FortranReader:
    """Reads tokens from a Fortran list-directed input file.

    Handles: free-format values separated by spaces/commas,
    multi-line continuation, and Fortran complex literals (re, im).

    Each call to read_record() starts fresh, mimicking Fortran's
    'read(unit,*)' which starts reading from a new line.
    Use read_int/read_float/etc. within a record for multi-value reads.
    Call flush() between Fortran READ statements to discard any leftover tokens.
    """

    def __init__(self, filepath):
        with open(filepath) as f:
            self.lines = f.readlines()
        self.pos = 0       # current line
        self.tokens = []    # remaining tokens on current line(s)
        self._in_record = False

    def _advance(self):
        """Move to next line that has tokens."""
        while self.pos < len(self.lines):
            line = self.lines[self.pos].strip()
            self.pos += 1
            if not line:
                continue
            # Tokenize: split on whitespace and commas
            raw = re.split(r'[,\s]+', line)
            self.tokens = [t for t in raw if t]
            if self.tokens:
                return
        self.tokens = []

    def flush(self):
        """Discard remaining tokens (end of Fortran READ statement)."""
        self.tokens = []

    def read_line_raw(self):
        """Read one full line as a string (for title/text reads with format '(a)')."""
        self.flush()
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            self.pos += 1
            return line.rstrip('\n')
        return ''

    def _ensure_tokens(self):
        if not self.tokens:
            self._advance()

    def read_int(self):
        self._ensure_tokens()
        val = int(self.tokens.pop(0))
        return val

    def read_float(self):
        self._ensure_tokens()
        s = self.tokens.pop(0)
        # Handle Fortran-style exponents like 1.0d3 or 1.0D-4
        s = s.replace('d', 'e').replace('D', 'E')
        return float(s)

    def read_logical(self):
        self._ensure_tokens()
        s = self.tokens.pop(0).lower()
        if s in ('t', '.true.', 'true'):
            return True
        elif s in ('f', '.false.', 'false'):
            return False
        raise ValueError(f"Cannot parse logical: {s}")

    def read_string(self):
        self._ensure_tokens()
        return self.tokens.pop(0)

    def read_ints(self, n):
        return [self.read_int() for _ in range(n)]

    def read_floats(self, n):
        return [self.read_float() for _ in range(n)]

    def read_complex(self):
        """Read a Fortran complex literal like (1.0, 2.0)."""
        self._ensure_tokens()
        # Complex might be tokenized as "(1.0" "2.0)" or "(1.0,2.0)"
        s = self.tokens.pop(0)
        if s.startswith('('):
            full = s
            while ')' not in full:
                self._ensure_tokens()
                full += ',' + self.tokens.pop(0)
            inner = full.strip('()')
            parts = inner.split(',')
            return complex(float(parts[0]), float(parts[1]))
        # If not in parens, try as two separate floats
        re_part = float(s.replace('d', 'e').replace('D', 'E'))
        im_part = self.read_float()
        return complex(re_part, im_part)


# ============================================================
# Parser for pdstrip.inp
# ============================================================
def parse_pdstrip_inp(inp_path):
    """Parse a pdstrip input file and the referenced geometry file.

    inp_path: path to the pdstrip .inp file (e.g., 'pdstrip.inp' or 'dir/pdstrip.inp')

    Returns a dict with all parsed parameters.
    """
    inp_dir = os.path.dirname(os.path.abspath(inp_path))
    reader = FortranReader(inp_path)

    cfg = {}

    # Line 1: npres, lsect, ltrans, lsign
    cfg['npres'] = reader.read_int()
    cfg['lsect'] = reader.read_logical()
    cfg['ltrans'] = reader.read_logical()
    cfg['lsign'] = reader.read_logical()
    reader.flush()

    lpres = True
    if cfg['npres'] < 0:
        lpres = False
        cfg['npres'] = abs(cfg['npres'])
    cfg['lpres'] = lpres

    # Line 2: title
    cfg['title'] = reader.read_line_raw()

    # Line 3: g, rho, zwl, zbot, zdrift
    cfg['g'] = reader.read_float()
    cfg['rho'] = reader.read_float()
    cfg['zwl'] = reader.read_float()
    cfg['zbot'] = reader.read_float()
    cfg['zdrift'] = reader.read_float()
    reader.flush()

    # Line 4: nmu, wave angles in degrees
    cfg['nmu'] = reader.read_int()
    cfg['wangl_deg'] = reader.read_floats(cfg['nmu'])
    reader.flush()

    # Line 5: geometry filename
    cfg['offsetfile'] = reader.read_string()
    reader.flush()

    # Line 6: catamaran, hulld
    cfg['catamaran'] = reader.read_logical()
    cfg['hulld'] = reader.read_float()
    reader.flush()
    if not cfg['catamaran']:
        cfg['hulld'] = 0.0

    # Parse geometry file — try path relative to input dir, then as-is,
    # then just the basename in the input dir
    geom_candidates = [
        os.path.join(inp_dir, cfg['offsetfile']),
        cfg['offsetfile'],
        os.path.join(inp_dir, os.path.basename(cfg['offsetfile'])),
    ]
    geom_path = None
    for candidate in geom_candidates:
        if os.path.isfile(candidate):
            geom_path = candidate
            break
    if geom_path is None:
        raise FileNotFoundError(
            f"Cannot find geometry file '{cfg['offsetfile']}'. "
            f"Tried: {geom_candidates}")
    cfg['geometry'] = parse_geometry(geom_path)

    # Line 7: intersection forces flag
    cfg['ls'] = reader.read_logical()
    reader.flush()
    ns = cfg['geometry']['nse'] if cfg['ls'] else 1

    # Line 8: mass/COG/inertia (ns sets)
    cfg['bodies'] = []
    for _ in range(ns):
        body = {}
        body['mass'] = reader.read_float()
        body['xg'] = reader.read_float()
        body['yg'] = reader.read_float()  # port positive in input
        body['zg'] = reader.read_float()  # up positive in input
        body['thxx'] = reader.read_float()
        body['thyy'] = reader.read_float()
        body['thzz'] = reader.read_float()
        body['thxy'] = reader.read_float()
        body['thyz'] = reader.read_float()
        body['thxz'] = reader.read_float()
        cfg['bodies'].append(body)
    reader.flush()

    nse = cfg['geometry']['nse']

    # Line 9: flow separation flags
    cfg['iab'] = reader.read_ints(nse)
    reader.flush()

    # Line 10: wave steepness, max height
    cfg['steepness'] = reader.read_float()
    cfg['maxheight'] = reader.read_float()
    reader.flush()

    # Line 11: cross-flow drag coefficients (if steepness > 0)
    if cfg['steepness'] > 0:
        cdy = []
        cdz = []
        for _ in range(nse):
            cdy.append(reader.read_float())
            cdz.append(reader.read_float())
        cfg['cdy'] = cdy
        cfg['cdz'] = cdz
        reader.flush()

    # Line 12: fins
    cfg['nfin'] = reader.read_int()
    reader.flush()
    cfg['fins'] = []
    for _ in range(cfg['nfin']):
        fin = {}
        fin['xfin'] = reader.read_floats(3)
        fin['afin'] = reader.read_floats(3)
        fin['cdelfin'] = [reader.read_complex() for _ in range(6)]
        fin['lfin'] = reader.read_float()
        fin['cfin'] = reader.read_float()
        fin['cmfin'] = reader.read_float()
        fin['clgrfin'] = reader.read_float()
        fin['rfin'] = reader.read_float()
        fin['sfin'] = reader.read_float()
        fin['cdfin'] = reader.read_float()
        cfg['fins'].append(fin)
    reader.flush()

    # Line 14: sails
    cfg['nsail'] = reader.read_int()
    reader.flush()
    if cfg['nsail'] > 0:
        cfg['uw'] = reader.read_float()
        cfg['muw'] = reader.read_float()
        reader.flush()
        cfg['sails'] = []
        for _ in range(cfg['nsail']):
            sail = {}
            sail['xsail'] = reader.read_floats(3)
            sail['csail'] = reader.read_floats(3)
            sail['msail'] = reader.read_floats(3)
            sail['dcndasail'] = reader.read_float()
            sail['cmsail'] = reader.read_float()
            cfg['sails'].append(sail)
        reader.flush()

    # Line 17: motion-dependent forces
    cfg['nforce'] = reader.read_int()
    reader.flush()
    if cfg['nforce'] > 0:
        cfg['forces'] = []
        for _ in range(cfg['nforce']):
            force = {}
            force['xforce'] = reader.read_floats(3)
            force['aforce'] = reader.read_floats(3)
            force['calforce'] = [reader.read_complex() for _ in range(7)]
            cfg['forces'].append(force)
        reader.flush()

    # Line 19: suspended weight
    cfg['rml'] = reader.read_float()
    cfg['xl'] = reader.read_float()
    cfg['yl'] = reader.read_float()
    cfg['zl'] = reader.read_float()
    cfg['cablelength'] = reader.read_float()
    reader.flush()

    # Line 20: motion points
    cfg['nb'] = reader.read_int()
    reader.flush()
    cfg['motion_points'] = []
    for _ in range(cfg['nb']):
        xb = reader.read_float()
        yb = reader.read_float()
        zb = reader.read_float()
        cfg['motion_points'].append((xb, yb, zb))
    reader.flush()

    # Line 22: wavelengths
    cfg['nom'] = reader.read_int()
    cfg['wavelengths'] = reader.read_floats(cfg['nom'])
    reader.flush()

    # Line 23: speeds
    cfg['nv'] = reader.read_int()
    reader.flush()
    cfg['speeds'] = []
    for _ in range(cfg['nv']):
        spd = reader.read_float()
        ltrwet = reader.read_logical()
        cfg['speeds'].append((spd, ltrwet))
        reader.flush()

    return cfg


# ============================================================
# Parser for geometry file
# ============================================================
def parse_geometry(geom_path):
    """Parse a pdstrip geometry file (geomet.out format)."""
    reader = FortranReader(geom_path)

    geom = {}
    geom['nse'] = reader.read_int()
    geom['sym'] = reader.read_logical()
    geom['tg'] = reader.read_float()
    reader.flush()

    sections = []
    for _ in range(geom['nse']):
        sec = {}
        sec['x'] = reader.read_float()
        sec['nof'] = reader.read_int()
        sec['ngap'] = reader.read_int()
        if sec['ngap'] > 0:
            sec['gaps'] = reader.read_ints(sec['ngap'])
        else:
            sec['gaps'] = []
        reader.flush()
        sec['yof'] = reader.read_floats(sec['nof'])
        sec['zof'] = reader.read_floats(sec['nof'])
        reader.flush()
        sections.append(sec)

    geom['sections'] = sections
    return geom


# ============================================================
# Mesh lofting
# ============================================================
def loft_mesh(sections, sym, zwl=0.0):
    """Loft section offsets into a 3D quad-panel mesh.

    Input coordinates are in pdstrip input convention:
        x forward, y port-positive, z up-positive.

    Capytaine uses the same convention (x fwd, y port, z up),
    so no coordinate transformation is needed.

    Returns: vertices (N,3), faces (M,4) — quad panels.
    """
    # Build full section contours (if sym, mirror to get full)
    full_sections = []
    for sec in sections:
        y = np.array(sec['yof'])
        z = np.array(sec['zof'])

        if sym:
            # sym=True: points go from centerline (y=0) to port waterline
            # Mirror to get full section: stb points (y negated, reversed) + port points
            # Skip the first point (y=0) in the mirror to avoid duplication
            y_full = np.concatenate([-y[::-1][:-1], y])
            z_full = np.concatenate([z[::-1][:-1], z])
        else:
            y_full = y
            z_full = z

        full_sections.append({
            'x': sec['x'],
            'y': y_full,
            'z': z_full,
            'gaps': sec.get('gaps', []),
        })

    # Interpolate sections to common point count for clean lofting
    # Strategy: use the section with the most points as the target count,
    # or resample all sections to a common count
    npts_max = max(len(s['y']) for s in full_sections)

    # For sections with different point counts, resample parametrically
    resampled = []
    for sec in full_sections:
        y, z = sec['y'], sec['z']
        n = len(y)

        if n == npts_max:
            resampled.append(sec)
        else:
            # Parametric resampling by arc length
            ds = np.sqrt(np.diff(y)**2 + np.diff(z)**2)
            s = np.concatenate([[0], np.cumsum(ds)])
            s_norm = s / s[-1] if s[-1] > 0 else np.linspace(0, 1, n)
            s_target = np.linspace(0, 1, npts_max)
            y_new = np.interp(s_target, s_norm, y)
            z_new = np.interp(s_target, s_norm, z)
            resampled.append({
                'x': sec['x'],
                'y': y_new,
                'z': z_new,
                'gaps': [],  # gaps lost in resampling
            })

    # Build vertices and quad faces
    nsec = len(resampled)
    npt = npts_max
    vertices = []
    vert_idx = {}  # (isec, ipt) -> vertex index

    for isec, sec in enumerate(resampled):
        for ipt in range(npt):
            idx = len(vertices)
            vert_idx[(isec, ipt)] = idx
            vertices.append([sec['x'], sec['y'][ipt], sec['z'][ipt]])

    faces = []
    for isec in range(nsec - 1):
        for ipt in range(npt - 1):
            v00 = vert_idx[(isec, ipt)]
            v01 = vert_idx[(isec, ipt + 1)]
            v10 = vert_idx[(isec + 1, ipt)]
            v11 = vert_idx[(isec + 1, ipt + 1)]
            faces.append([v00, v01, v11, v10])

    vertices = np.array(vertices)
    faces = np.array(faces)

    return vertices, faces


# ============================================================
# Generate Capytaine script
# ============================================================
def generate_script(cfg, output_path):
    """Generate a standalone Capytaine Python script."""

    geom = cfg['geometry']
    g = cfg['g']
    rho = cfg['rho']
    zwl = cfg['zwl']
    zbot = cfg['zbot']

    # Deep water if zbot is very large negative
    water_depth = abs(zbot - zwl) if abs(zbot) < 1e5 else np.inf

    # Convert wavelengths to omega via deep-water dispersion relation
    wavelengths = np.array(cfg['wavelengths'])
    k_values = 2 * np.pi / wavelengths
    if np.isinf(water_depth):
        omega_values = np.sqrt(g * k_values)
    else:
        # General dispersion: omega^2 = g*k*tanh(k*h)
        omega_values = np.sqrt(g * k_values * np.tanh(k_values * water_depth))

    # Unique, sorted frequencies
    omega_values = np.sort(np.unique(np.round(omega_values, 8)))

    # Wave directions: pdstrip degrees -> Capytaine radians
    # pdstrip: 0 = following seas (wave propagation in +x)
    # Capytaine: beta is wave propagation direction; beta=0 means propagation in +x
    # So they are the same convention: beta_capy = mu_pdstrip in radians
    wave_dirs_deg = np.array(cfg['wangl_deg'])
    wave_dirs_rad = np.deg2rad(wave_dirs_deg)

    # Loft the mesh
    vertices, faces = loft_mesh(geom['sections'], geom['sym'], zwl)

    # Clip to below waterline: keep only panels where at least one vertex is below zwl
    # (Capytaine's immersed_part will handle this more precisely, but pre-filter helps)
    # Actually, let Capytaine handle this via immersed_part()

    catamaran = cfg['catamaran']
    hulld = cfg['hulld']
    body0 = cfg['bodies'][0]

    # Format arrays for embedding in the script
    def fmt_array(arr, per_line=8, precision=6):
        lines = []
        for i in range(0, len(arr), per_line):
            chunk = arr[i:i+per_line]
            lines.append(', '.join(f'{v:.{precision}f}' for v in chunk))
        return ',\n        '.join(lines)

    def fmt_vertices(verts):
        lines = []
        for v in verts:
            lines.append(f'    [{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}],')
        return '\n'.join(lines)

    def fmt_faces(fs):
        lines = []
        for f in fs:
            lines.append(f'    [{f[0]}, {f[1]}, {f[2]}, {f[3]}],')
        return '\n'.join(lines)

    water_depth_str = 'np.inf' if np.isinf(water_depth) else f'{water_depth:.4f}'

    script = f'''#!/usr/bin/env python3
"""
Capytaine BEM script generated from pdstrip input files.

Source: {cfg['title']}
Generated by pdstrip2capytaine.py

Geometry: {geom['nse']} sections, sym={geom['sym']}, draft={geom['tg']:.3f} m
Catamaran: {catamaran}, hulld={hulld:.3f} m
Frequencies: {len(omega_values)} values ({omega_values[0]:.4f} to {omega_values[-1]:.4f} rad/s)
Wave directions: {len(wave_dirs_deg)} values ({wave_dirs_deg[0]:.1f} to {wave_dirs_deg[-1]:.1f} deg)

Usage:
    python {os.path.basename(output_path)}
"""

import numpy as np
import capytaine as cpt
import logging
import xarray as xr

cpt.set_logging(logging.WARNING)

# ============================================================
# Physical constants
# ============================================================
g = {g}
rho = {rho}
water_depth = {water_depth_str}

# ============================================================
# Frequencies and wave directions
# ============================================================
omega_values = np.array([
        {fmt_array(omega_values)}
])

wave_directions = np.array([
        {fmt_array(wave_dirs_rad, precision=8)}
])
wave_directions_deg = np.array([
        {fmt_array(wave_dirs_deg, precision=1)}
])

# ============================================================
# Hull mesh: {geom['nse']} sections lofted into quad panels
# ============================================================
vertices = np.array([
{fmt_vertices(vertices)}
])

faces = np.array([
{fmt_faces(faces)}
])

print(f"Hull mesh: {{vertices.shape[0]}} vertices, {{faces.shape[0]}} quad faces")

'''

    if not catamaran:
        script += f'''
# ============================================================
# Monohull body setup
# ============================================================
hull_mesh = cpt.Mesh(vertices=vertices, faces=faces, name='hull')

# Clip to below waterline
hull_mesh = hull_mesh.immersed_part(water_depth={water_depth_str})
print(f"Immersed mesh: {{hull_mesh.nb_faces}} faces")

# Add lid to suppress irregular frequencies
lid = hull_mesh.generate_lid(z={zwl - 0.01:.4f})

body = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid, name='vessel')
body.add_all_rigid_body_dofs()
print(f"Body DOFs: {{list(body.dofs.keys())}}")
'''
    else:
        script += f'''
# ============================================================
# Catamaran body setup (two hulls at y = +/-hulld)
# ============================================================
hulld = {hulld:.6f}

# Starboard hull: offset by -hulld in y (pdstrip convention: stb = negative y internally,
# but input coords have y port-positive, so stb hull is at y = -hulld)
vertices_stb = vertices.copy()
vertices_stb[:, 1] -= hulld   # shift to y = y_demi - hulld

vertices_port = vertices.copy()
vertices_port[:, 1] += hulld   # shift to y = y_demi + hulld

mesh_stb = cpt.Mesh(vertices=vertices_stb, faces=faces.copy(), name='stb_hull')
mesh_stb = mesh_stb.immersed_part(water_depth={water_depth_str})
lid_stb = mesh_stb.generate_lid(z={zwl - 0.01:.4f})
body_stb = cpt.FloatingBody(mesh=mesh_stb, lid_mesh=lid_stb, name='stb_hull')
body_stb.add_all_rigid_body_dofs()

mesh_port = cpt.Mesh(vertices=vertices_port, faces=faces.copy(), name='port_hull')
mesh_port = mesh_port.immersed_part(water_depth={water_depth_str})
lid_port = mesh_port.generate_lid(z={zwl - 0.01:.4f})
body_port = cpt.FloatingBody(mesh=mesh_port, lid_mesh=lid_port, name='port_hull')
body_port.add_all_rigid_body_dofs()

body = body_stb + body_port
print(f"Catamaran: stb={{mesh_stb.nb_faces}} faces, port={{mesh_port.nb_faces}} faces")
print(f"Body DOFs: {{list(body.dofs.keys())}}")
'''

    script += f'''
# ============================================================
# Set up and solve BEM problems
# ============================================================
problems = []
for omega in omega_values:
    for dof in body.dofs:
        problems.append(cpt.RadiationProblem(
            body=body, radiating_dof=dof, omega=omega,
            water_depth=water_depth, rho=rho, g=g))
    for beta in wave_directions:
        problems.append(cpt.DiffractionProblem(
            body=body, wave_direction=beta, omega=omega,
            water_depth=water_depth, rho=rho, g=g))

print(f"Solving {{len(problems)}} BEM problems...")
solver = cpt.BEMSolver()
results = solver.solve_all(problems, progress_bar=True)

# Assemble into xarray dataset
dataset = cpt.assemble_dataset(results)

# ============================================================
# Save results
# ============================================================
# Save as numpy archive (avoids xarray/netCDF dtype issues with complex data)
output_file = '{os.path.splitext(os.path.basename(output_path))[0]}_results.npz'
np.savez(output_file,
    omega=omega_values,
    wave_directions=wave_directions,
    wave_directions_deg=wave_directions_deg,
    added_mass=dataset['added_mass'].values,
    radiation_damping=dataset['radiation_damping'].values,
    diffraction_force=dataset['diffraction_force'].values if 'diffraction_force' in dataset else np.array([]),
    Froude_Krylov_force=dataset['Froude_Krylov_force'].values if 'Froude_Krylov_force' in dataset else np.array([]),
    radiating_dofs=np.array(list(dataset.coords['radiating_dof'].values)),
    influenced_dofs=np.array(list(dataset.coords['influenced_dof'].values)),
    g=g, rho=rho, water_depth={water_depth_str if not np.isinf(water_depth) else 'np.inf'},
)
print(f"Results saved to {{output_file}}")

# ============================================================
# Print summary table
# ============================================================
print("\\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
dofs = list(body.dofs.keys())
for dof in ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw'] + [d for d in dofs if '__' in d]:
    if dof not in dofs and not any(dof in d for d in dofs):
        continue
    # Find matching DOF name
    matches = [d for d in dofs if d == dof or d.endswith(dof)]
    for dof_name in matches[:1]:
        try:
            am = dataset['added_mass'].sel(radiating_dof=dof_name, influenced_dof=dof_name)
            rd = dataset['radiation_damping'].sel(radiating_dof=dof_name, influenced_dof=dof_name)
            print(f"\\n{{dof_name}}:")
            print(f"  omega     added_mass      damping")
            for iw, w in enumerate(omega_values[:10]):
                a_val = float(am.sel(omega=w, method='nearest'))
                b_val = float(rd.sel(omega=w, method='nearest'))
                print(f"  {{w:8.4f}}  {{a_val:14.4f}}  {{b_val:14.4f}}")
            if len(omega_values) > 10:
                print(f"  ... ({{len(omega_values) - 10}} more frequencies)")
        except (KeyError, ValueError):
            pass

print("\\nDone.")
'''

    with open(output_path, 'w') as f:
        f.write(script)

    print(f"Generated Capytaine script: {output_path}")
    print(f"  Geometry: {geom['nse']} sections, {len(vertices)} vertices, {len(faces)} quad faces")
    print(f"  Frequencies: {len(omega_values)} ({omega_values[0]:.4f} to {omega_values[-1]:.4f} rad/s)")
    print(f"  Wave directions: {len(wave_dirs_deg)}")
    print(f"  Catamaran: {catamaran}" + (f", hulld={hulld:.3f}" if catamaran else ""))
    print(f"\nRun with: python {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python pdstrip2capytaine.py <pdstrip.inp> [output_script.py]")
        print("  <pdstrip.inp>    : path to pdstrip input file")
        print("  [output_script]  : output Python filename (default: capytaine_run.py)")
        sys.exit(1)

    inp_path = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else 'capytaine_run.py'

    print(f"Reading pdstrip input from: {inp_path}")
    cfg = parse_pdstrip_inp(inp_path)

    print(f"\nParsed configuration:")
    print(f"  Title: {cfg['title']}")
    print(f"  g={cfg['g']}, rho={cfg['rho']}, zwl={cfg['zwl']}, zbot={cfg['zbot']}")
    print(f"  Sections: {cfg['geometry']['nse']}, sym={cfg['geometry']['sym']}")
    print(f"  Wavelengths: {cfg['nom']} ({cfg['wavelengths'][0]:.1f} to {cfg['wavelengths'][-1]:.1f} m)")
    print(f"  Wave angles: {cfg['nmu']} ({cfg['wangl_deg'][0]:.1f} to {cfg['wangl_deg'][-1]:.1f} deg)")
    print(f"  Speeds: {cfg['nv']}")
    print(f"  Catamaran: {cfg['catamaran']}, hulld={cfg['hulld']}")
    print(f"  Mass: {cfg['bodies'][0]['mass']:.1f} kg")

    generate_script(cfg, output)


if __name__ == '__main__':
    main()
