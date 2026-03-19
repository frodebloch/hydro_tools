"""Floating turbine geometry — simple spar + tower + rotor representation.

Based on the OC3 Hywind spar dimensions from the dp_simulator floating
platform model.
"""

import numpy as np
import pyvista as pv
import vtk

# ── OC3 Hywind spar dimensions (from PlatformModel defaults) ────────────
SPAR_DRAFT = 120.0  # m below SWL
SPAR_UPPER_DIAMETER = 6.5  # m (above taper)
SPAR_LOWER_DIAMETER = 9.4  # m (below taper)
SPAR_TAPER_TOP = -4.0  # m (below SWL)
SPAR_TAPER_BOTTOM = -12.0  # m (below SWL)
SPAR_FREEBOARD = 10.0  # m above SWL

# Turbine dimensions
HUB_HEIGHT = 90.0  # m above SWL
ROTOR_DIAMETER = 126.0  # m
TOWER_BASE_DIAMETER = 6.5  # m
TOWER_TOP_DIAMETER = 3.87  # m
NACELLE_LENGTH = 14.0  # m
NACELLE_WIDTH = 4.0  # m
NACELLE_HEIGHT = 4.0  # m


def _build_spar() -> pv.PolyData:
    """Build the spar buoy as a series of cylinder segments.

    Simplified as three segments: lower cylinder, taper, upper cylinder + freeboard.
    """
    # Lower cylinder: from -120m to taper bottom (-12m)
    lower = pv.Cylinder(
        center=(0, 0, (-120.0 + -12.0) / 2.0),
        direction=(0, 0, 1),
        radius=SPAR_LOWER_DIAMETER / 2.0,
        height=120.0 - 12.0,
        resolution=12,
        capping=True,
    )

    # Upper cylinder: from taper top (-4m) to freeboard (+10m)
    upper = pv.Cylinder(
        center=(0, 0, (-4.0 + SPAR_FREEBOARD) / 2.0),
        direction=(0, 0, 1),
        radius=SPAR_UPPER_DIAMETER / 2.0,
        height=SPAR_FREEBOARD - (-4.0),
        resolution=12,
        capping=True,
    )

    # Taper section: frustum from -12m (D=9.4) to -4m (D=6.5)
    # Build as a truncated cone using parametric points
    n_circ = 12
    theta = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
    # Bottom ring
    r_bot = SPAR_LOWER_DIAMETER / 2.0
    z_bot = SPAR_TAPER_BOTTOM
    x_bot = r_bot * np.cos(theta)
    y_bot = r_bot * np.sin(theta)
    z_bot_arr = np.full(n_circ, z_bot)
    # Top ring
    r_top = SPAR_UPPER_DIAMETER / 2.0
    z_top = SPAR_TAPER_TOP
    x_top = r_top * np.cos(theta)
    y_top = r_top * np.sin(theta)
    z_top_arr = np.full(n_circ, z_top)

    vertices = np.vstack([
        np.column_stack([x_bot, y_bot, z_bot_arr]),
        np.column_stack([x_top, y_top, z_top_arr]),
    ])
    faces = []
    for i in range(n_circ):
        i_next = (i + 1) % n_circ
        faces.extend([4, i, i_next, n_circ + i_next, n_circ + i])
    taper = pv.PolyData(vertices, np.array(faces))

    spar = lower.merge(taper).merge(upper)
    return spar


def _build_tower() -> pv.PolyData:
    """Tapered tower from spar top to hub height."""
    # Simple cylinder (slight taper is hard to see at engineering vis scale)
    tower = pv.Cylinder(
        center=(0, 0, (SPAR_FREEBOARD + HUB_HEIGHT) / 2.0),
        direction=(0, 0, 1),
        radius=(TOWER_BASE_DIAMETER + TOWER_TOP_DIAMETER) / 4.0,
        height=HUB_HEIGHT - SPAR_FREEBOARD,
        resolution=8,
        capping=True,
    )
    return tower


def _build_nacelle() -> pv.PolyData:
    """Nacelle box at hub height."""
    nacelle = pv.Box(bounds=(
        -NACELLE_WIDTH / 2, NACELLE_WIDTH / 2,
        -NACELLE_LENGTH / 2, NACELLE_LENGTH / 2,
        HUB_HEIGHT - NACELLE_HEIGHT / 2, HUB_HEIGHT + NACELLE_HEIGHT / 2,
    ))
    return nacelle


def _build_rotor() -> pv.PolyData:
    """Three-bladed rotor in the X-Z plane at hub height.

    Each blade extends radially from the hub centre.  The rotor rotation axis
    is aligned with the Y-axis (the forward/wind direction of the turbine).
    Blades are built pointing along +Z from origin, then rotated about the
    Y-axis to their 120-degree spacing, and finally translated up to hub height.
    """
    blade_length = ROTOR_DIAMETER / 2.0
    blade_width = 4.0   # m chord (simplified)
    blade_thickness = 0.5  # m

    blades = []
    for angle_deg in [0, 120, 240]:
        # Build blade as a thin box from 0 to blade_length along +Z
        blade = pv.Box(bounds=(
            -blade_width / 2.0, blade_width / 2.0,
            -blade_thickness / 2.0, blade_thickness / 2.0,
            0.0, blade_length,
        ))
        # Rotate about the Y-axis so blades fan out in the X-Z plane
        blade.rotate_y(angle_deg, point=(0, 0, 0), inplace=True)
        # Translate up to hub height
        blade.translate((0, 0, HUB_HEIGHT), inplace=True)
        blades.append(blade)

    rotor = blades[0]
    for b in blades[1:]:
        rotor = rotor.merge(b)

    # Hub sphere
    hub = pv.Sphere(radius=2.0, center=(0, 0, HUB_HEIGHT))
    rotor = rotor.merge(hub)
    return rotor


class TurbineGeometry:
    """Complete floating wind turbine geometry with 6DOF transform.

    Body-frame coordinate system (before transform):
        X = East (lateral)
        Y = North (forward)
        Z = Up
        Origin at waterline centre of spar.
    """

    def __init__(self):
        spar = _build_spar()
        tower = _build_tower()
        nacelle = _build_nacelle()
        rotor = _build_rotor()

        self.mesh = spar.merge(tower).merge(nacelle).merge(rotor)

        # VTK transform for fast rigid-body updates
        self._vtk_transform = vtk.vtkTransform()
        self._vtk_transform.PostMultiply()

    def update_transform(
        self,
        north: float,
        east: float,
        heading_deg: float,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
        heave: float = 0.0,
    ):
        """Apply 6DOF transform — same convention as VesselGeometry."""
        t = self._vtk_transform
        t.Identity()
        t.RotateX(roll_deg)
        t.RotateY(-pitch_deg)
        t.RotateZ(heading_deg)
        t.Translate(east, north, heave)
