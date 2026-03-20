"""Floating turbine geometry — spar + tower + nacelle + spinning rotor.

Based on the OC3 Hywind spar dimensions and NREL 5MW turbine parameters
from the dp_simulator floating platform model.

The rotor is a separate mesh with its own VTK transform so it can spin
independently of the platform 6DOF motion.
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

# NREL 5MW turbine dimensions
HUB_HEIGHT = 90.0  # m above SWL
ROTOR_DIAMETER = 126.0  # m
ROTOR_RADIUS = ROTOR_DIAMETER / 2.0
TOWER_BASE_DIAMETER = 6.5  # m
TOWER_TOP_DIAMETER = 3.87  # m
NACELLE_LENGTH = 14.0  # m
NACELLE_WIDTH = 4.0  # m
NACELLE_HEIGHT = 4.0  # m
SHAFT_OVERHANG = 5.0  # m upwind of tower centre (NREL 5MW definition report)

# NREL 5MW rotor speed parameters (matching C++ TurbineThrustModel)
RATED_ROTOR_SPEED = 1.2671  # rad/s (12.1 rpm)
MIN_ROTOR_SPEED = 0.7225  # rad/s (6.9 rpm)
RATED_WIND_SPEED = 11.4  # m/s
CUT_IN_WIND_SPEED = 3.0  # m/s
OPTIMAL_TIP_SPEED_RATIO = 7.0

# Visualization spin-up/down dynamics
ROTOR_TIME_CONSTANT = 15.0  # s — first-order lag for visual rotor response


def rotor_speed(wind_speed: float, turbine_state: int = 0) -> float:
    """Compute rotor speed [rad/s] from hub-height wind speed.

    Replicates the C++ TurbineThrustModel::RotorSpeed() logic:
      - Below cut-in (3 m/s): 0 rad/s
      - Region 2 (3-11.4 m/s): variable speed, optimal TSR tracking
      - Region 3 (>11.4 m/s): rated speed (12.1 rpm)

    Args:
        wind_speed: Hub-height wind speed [m/s]
        turbine_state: 0=operating, 1=shutdown, 2=idling

    Returns:
        Rotor speed [rad/s]
    """
    if turbine_state == 1:  # shutdown
        return 0.0
    if turbine_state == 2:  # idling
        # Small residual rotation (5% of what it would be)
        factor = 0.05
    else:
        factor = 1.0

    if wind_speed < CUT_IN_WIND_SPEED:
        return 0.0

    if wind_speed <= RATED_WIND_SPEED:
        # Region 2: variable speed, optimal TSR tracking
        omega = OPTIMAL_TIP_SPEED_RATIO * wind_speed / ROTOR_RADIUS
        omega = max(MIN_ROTOR_SPEED, min(omega, RATED_ROTOR_SPEED))
    else:
        # Region 3: constant rotor speed
        omega = RATED_ROTOR_SPEED

    return omega * factor


def rotor_speed_rpm(wind_speed: float, turbine_state: int = 0) -> float:
    """Compute rotor speed [rpm] from hub-height wind speed."""
    return rotor_speed(wind_speed, turbine_state) * 60.0 / (2.0 * np.pi)


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
    n_circ = 12
    theta = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
    r_bot = SPAR_LOWER_DIAMETER / 2.0
    z_bot = SPAR_TAPER_BOTTOM
    x_bot = r_bot * np.cos(theta)
    y_bot = r_bot * np.sin(theta)
    z_bot_arr = np.full(n_circ, z_bot)
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
    """Three-bladed rotor centred at origin, rotation axis along Y.

    The rotor is built with the hub at the origin and blades in the X-Z plane.
    It will be translated to hub height by the VTK transform chain.

    The rotation axis is Y (north/forward in body frame) — matching the
    nacelle orientation where the shaft points into the wind.
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
        blades.append(blade)

    rotor = blades[0]
    for b in blades[1:]:
        rotor = rotor.merge(b)

    # Hub sphere at origin
    hub = pv.Sphere(radius=2.0, center=(0, 0, 0))
    rotor = rotor.merge(hub)
    return rotor


class TurbineGeometry:
    """Complete floating wind turbine geometry with 6DOF transform.

    Body-frame coordinate system (before transform):
        X = East (lateral)
        Y = North (forward)
        Z = Up
        Origin at waterline centre of spar.

    The rotor is a separate mesh with its own VTK transform so it can
    spin independently. The transform chain for the rotor is:
        1. RotateY(rotor_angle) — spin about the shaft axis
        2. Translate to hub (0, +SHAFT_OVERHANG, HUB_HEIGHT) — upwind of tower
        3. Platform 6DOF (pitch, roll, heading, translate)
    """

    def __init__(self):
        spar = _build_spar()
        tower = _build_tower()
        nacelle = _build_nacelle()

        # Static mesh: spar + tower + nacelle (no rotor)
        self.mesh = spar.merge(tower).merge(nacelle)

        # Rotor mesh: centred at origin, will be positioned by transform
        self.rotor_mesh = _build_rotor()

        # VTK transform for the static structure (6DOF)
        self._vtk_transform = vtk.vtkTransform()
        self._vtk_transform.PostMultiply()

        # VTK transform for the rotor (6DOF + spin)
        self._rotor_vtk_transform = vtk.vtkTransform()
        self._rotor_vtk_transform.PostMultiply()

        # Current rotor angle [degrees] — accumulated over time
        self._rotor_angle_deg = 0.0

        # Actual rotor angular velocity [rad/s] — smoothed toward target
        self._omega = 0.0

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
        # Static structure transform
        t = self._vtk_transform
        t.Identity()
        t.RotateX(pitch_deg)     # pitch about X (starboard), +bow up
        t.RotateY(roll_deg)      # roll about Y (forward), +stbd down
        t.RotateZ(-heading_deg)
        t.Translate(east, north, -heave)  # heave: sim +down (NED), viz +up -> negate

        # Rotor transform: spin first (in body frame), then hub offset, then 6DOF
        r = self._rotor_vtk_transform
        r.Identity()
        r.RotateY(self._rotor_angle_deg)  # spin about shaft (Y-axis)
        r.Translate(0, SHAFT_OVERHANG, HUB_HEIGHT)  # hub: upwind of tower, at hub height
        r.RotateX(pitch_deg)
        r.RotateY(roll_deg)
        r.RotateZ(-heading_deg)
        r.Translate(east, north, -heave)

    def update_rotor_angle(self, dt: float, wind_speed: float, turbine_state: int = 0):
        """Advance rotor angle based on wind speed and time step.

        Uses a first-order lag to smooth transitions — gives a visible
        coast-down on shutdown and gradual spin-up on start.

        Args:
            dt: Time step since last call [s]
            wind_speed: Hub-height wind speed [m/s]
            turbine_state: 0=operating, 1=shutdown, 2=idling
        """
        omega_target = rotor_speed(wind_speed, turbine_state)  # rad/s

        # First-order lag: omega -> omega_target with time constant tau
        if dt > 0 and ROTOR_TIME_CONSTANT > 0:
            alpha = 1.0 - np.exp(-dt / ROTOR_TIME_CONSTANT)
            self._omega += alpha * (omega_target - self._omega)
        else:
            self._omega = omega_target

        self._rotor_angle_deg += np.degrees(self._omega * dt)
        # Keep angle in [0, 360) to avoid float precision issues over long runs
        self._rotor_angle_deg %= 360.0
