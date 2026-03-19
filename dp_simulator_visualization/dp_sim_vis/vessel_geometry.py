"""Vessel geometry — simple 3D hull built from CSOV cross-section data.

Uses the 39-station section data from the Norwind SOV (config_csov) to build
a realistic hull shape, plus a simplified superstructure.
"""

import numpy as np
import pyvista as pv
import vtk

# ── CSOV section data: (offset_m, breadth_m, draft_m, area_m2) ──────────
# Offset: longitudinal position, positive = forward, zero near midship.
# From /brucon/modules/config_csov/vessel_data.prototxt.in
CSOV_SECTIONS = [
    (-52.65, 17.16, 1.83, 16.50),
    (-49.85, 19.67, 2.32, 27.02),
    (-47.05, 20.98, 2.85, 39.12),
    (-44.25, 21.71, 4.84, 52.46),
    (-41.45, 22.12, 6.11, 67.73),
    (-38.65, 22.33, 6.31, 81.16),
    (-35.85, 22.40, 6.35, 93.74),
    (-33.05, 22.40, 6.35, 106.04),
    (-30.25, 22.40, 6.37, 116.97),
    (-27.45, 22.40, 6.43, 125.99),
    (-24.65, 22.40, 6.49, 132.39),
    (-21.85, 22.40, 6.50, 136.75),
    (-19.05, 22.40, 6.50, 139.93),
    (-16.25, 22.40, 6.50, 141.91),
    (-13.45, 22.40, 6.50, 142.68),
    (-10.65, 22.40, 6.50, 142.74),
    (-7.85, 22.40, 6.50, 142.74),
    (-5.05, 22.40, 6.50, 142.74),
    (-2.25, 22.40, 6.50, 142.74),
    (0.55, 22.40, 6.50, 142.74),
    (3.35, 22.40, 6.50, 142.74),
    (6.15, 22.40, 6.50, 142.72),
    (8.95, 22.40, 6.50, 142.37),
    (11.75, 22.40, 6.50, 141.57),
    (14.55, 22.40, 6.50, 140.10),
    (17.35, 22.40, 6.50, 137.64),
    (20.15, 22.39, 6.50, 133.74),
    (22.95, 22.26, 6.50, 127.93),
    (25.75, 21.82, 6.50, 119.88),
    (28.55, 20.87, 6.50, 109.53),
    (31.35, 19.23, 6.50, 97.20),
    (34.15, 17.19, 6.50, 83.88),
    (36.95, 14.49, 6.50, 69.26),
    (39.75, 11.27, 6.50, 55.13),
    (42.55, 7.88, 6.50, 42.25),
    (45.35, 5.30, 6.50, 29.97),
    (48.15, 3.89, 6.24, 19.54),
    (50.95, 2.69, 5.44, 11.07),
    (53.75, 1.43, 4.05, 4.15),
]

VESSEL_LOA = 111.5
VESSEL_LPP = 101.1
VESSEL_BREADTH = 22.4
VESSEL_DESIGN_DRAFT = 6.5

# Estimated freeboard heights above waterline (Z=0) at each station.
# A CSOV like the Norwind SOV typically has:
#   - ~7m freeboard at midship (main deck height above waterline)
#   - ~8-9m at the bow (sheer)
#   - ~6-7m at the stern
# The hull sides above the waterline are approximately vertical (same breadth
# as at waterline) with modest sheer.
MIDSHIP_FREEBOARD = 7.0  # m above waterline at midship
BOW_SHEER = 2.5  # m additional at bow
STERN_SHEER = 0.5  # m additional at stern


def _freeboard_at_offset(offset: float) -> float:
    """Estimate freeboard height at a longitudinal station.

    Linear interpolation of sheer: higher at bow, slightly raised at stern.
    """
    # Offset range: stern ~ -53m, bow ~ +54m
    stern_x = CSOV_SECTIONS[0][0]
    bow_x = CSOV_SECTIONS[-1][0]
    t = (offset - stern_x) / (bow_x - stern_x)  # 0 at stern, 1 at bow
    # Sheer curve: parabolic, lowest at midship
    sheer = STERN_SHEER * (1.0 - t) ** 2 + BOW_SHEER * t**2
    return MIDSHIP_FREEBOARD + sheer


def _build_hull_mesh() -> pv.PolyData:
    """Build a 3D hull mesh by lofting cross-sections from keel to main deck.

    Each section has:
      - A curved underwater profile (parabolic bottom)
      - Vertical hull sides from waterline up to the main deck (freeboard)
      - A deck surface closing the top

    Coordinate system: X = East (starboard), Y = North (forward), Z = Up.
    Waterline at Z=0, keel below, deck above.
    """
    n_sections = len(CSOV_SECTIONS)
    n_bottom = 8  # points across the bottom curve (port to starboard)

    # Each section profile: bottom curve (n_bottom+1 pts) + starboard side up +
    # deck edge starboard + deck edge port + port side down
    # Total per section: (n_bottom+1) + 1 + 1 + 1 = n_bottom + 4
    # Profile order: 0..n_bottom = bottom curve (port to starboard at keel)
    #                n_bottom+1   = starboard at deck
    #                n_bottom+2   = port at deck

    section_profiles = []
    for offset, breadth, draft, _area in CSOV_SECTIONS:
        half_b = breadth / 2.0
        fb = _freeboard_at_offset(offset)

        profile = []
        # Bottom curve: port -> starboard through keel
        for i in range(n_bottom + 1):
            t = i / n_bottom  # 0=port, 1=starboard
            x = -half_b + t * breadth
            frac = (2.0 * t - 1.0) ** 2  # 1 at edges, 0 at keel
            z = -draft + draft * 0.3 * frac
            profile.append((x, offset, z))

        # Starboard deck edge (top of hull side)
        profile.append((half_b, offset, fb))
        # Port deck edge
        profile.append((-half_b, offset, fb))

        section_profiles.append(profile)

    n_pts = len(section_profiles[0])  # n_bottom + 3

    vertices = []
    for profile in section_profiles:
        vertices.extend(profile)

    faces = []
    for s in range(n_sections - 1):
        b0 = s * n_pts
        b1 = (s + 1) * n_pts

        # Bottom hull surface (quads across the bottom curve)
        for i in range(n_bottom):
            faces.extend([4, b0 + i, b0 + i + 1, b1 + i + 1, b1 + i])

        # Starboard side: from bottom curve starboard edge up to deck
        stbd_bottom = n_bottom      # index of starboard-most bottom point
        stbd_deck = n_bottom + 1    # index of starboard deck edge
        faces.extend([4, b0 + stbd_bottom, b0 + stbd_deck,
                      b1 + stbd_deck, b1 + stbd_bottom])

        # Port side: from deck down to bottom curve port edge
        port_deck = n_bottom + 2
        port_bottom = 0
        faces.extend([4, b0 + port_deck, b0 + port_bottom,
                      b1 + port_bottom, b1 + port_deck])

        # Deck surface (quad from port deck edge to starboard deck edge)
        faces.extend([4, b0 + stbd_deck, b0 + port_deck,
                      b1 + port_deck, b1 + stbd_deck])

    # Stern transom (close the aft end)
    b = 0
    stbd_d = n_bottom + 1
    port_d = n_bottom + 2
    # Transom: deck edges to bottom curve
    for i in range(n_bottom):
        faces.extend([3, b + i + 1, b + i, b + n_bottom + 1])  # triangles
    faces.extend([3, b + port_deck, b + 0, b + stbd_deck])
    # Fill the deck-to-hull triangles for the transom
    faces.extend([3, b + n_bottom, b + stbd_deck, b + 0])

    vertices = np.array(vertices)
    faces = np.array(faces)
    mesh = pv.PolyData(vertices, faces)
    return mesh


def _build_superstructure() -> pv.PolyData:
    """Simple box superstructure on the aft part of the vessel.

    Positioned roughly over the aft third, representing the bridge/accommodation.
    Sits on top of the main deck (at freeboard height).
    """
    deck_z = MIDSHIP_FREEBOARD + STERN_SHEER  # approx deck height in aft region
    # Bridge block: centered aft, slightly narrower than beam
    bridge = pv.Box(bounds=(
        -8.0, 8.0,       # x: port-starboard
        -35.0, -10.0,    # y: aft section
        deck_z, deck_z + 10.0,  # z: from deck to ~10m above deck
    ))
    # Wheelhouse on top
    wheelhouse = pv.Box(bounds=(
        -5.0, 5.0,
        -30.0, -18.0,
        deck_z + 10.0, deck_z + 14.0,
    ))
    # Funnel
    funnel = pv.Box(bounds=(
        -2.5, 2.5,
        -40.0, -35.0,
        deck_z, deck_z + 12.0,
    ))
    superstructure = bridge.merge(wheelhouse).merge(funnel)
    return superstructure


def _build_helideck() -> pv.PolyData:
    """Helideck disc on the foredeck area."""
    deck_z = MIDSHIP_FREEBOARD + BOW_SHEER * 0.5  # approximate deck height forward
    deck = pv.Disc(
        center=(0.0, 35.0, deck_z + 0.3),
        inner=0.0, outer=10.0, normal=(0, 0, 1),
    )
    return deck


class VesselGeometry:
    """Complete vessel geometry with hull, superstructure, and 6DOF transform.

    Coordinate convention for the mesh (before transform):
        X = East (starboard positive)
        Y = North (forward positive)
        Z = Up (above waterline positive)
        Origin at midship waterline.

    The transform maps to the NED world frame used by the simulator:
        World X = East
        World Y = North
        World Z = Up (note: we use Z-up for visualization, not NED Z-down)
    """

    def __init__(self):
        hull = _build_hull_mesh()
        superstructure = _build_superstructure()
        helideck = _build_helideck()

        # Merge into a single mesh
        self.mesh = hull.merge(superstructure).merge(helideck)

        # VTK transform for fast rigid-body updates (no mesh point modification)
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
        """Apply 6DOF transform to the vessel mesh via VTK transform (GPU-side).

        Parameters
        ----------
        north, east : float
            Position in NED frame [m].
        heading_deg : float
            Heading [deg], 0 = North, 90 = East (clockwise from North).
        roll_deg : float
            Roll angle [deg], positive = starboard down.
        pitch_deg : float
            Pitch angle [deg], positive = bow down.
        heave : float
            Vertical displacement [m], positive up.
        """
        t = self._vtk_transform
        t.Identity()
        # Order: rotate in body frame (roll, pitch, yaw), then translate
        # VTK RotateY = about Y axis, etc., applied in reverse reading order
        # with PostMultiply, transforms are applied left-to-right.
        t.RotateX(roll_deg)   # roll about X (starboard axis)
        t.RotateY(-pitch_deg)  # pitch about Y (forward axis), sign convention
        t.RotateZ(heading_deg)  # yaw/heading about Z
        t.Translate(east, north, heave)
