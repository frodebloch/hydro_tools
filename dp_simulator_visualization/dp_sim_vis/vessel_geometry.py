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
    """Build superstructure matching the Vard 985 CSOV side-profile.

    Profile reference: wind_profile.csv — a crude side-profile where x=0 is
    the stern and the bow is around x=103.  Mapped to body-frame coordinates
    (Y=0 at midship, forward positive).

    Reference image shows (stern to bow):
        - Low aft working deck (~2.4 m above WL, hull only)
        - Aft accommodation block with stepped upper decks
        - Mid-level working deck area with crane platform
        - Forward accommodation / bridge block (tallest accommodation)
        - Wheelhouse with bridge wings
        - Cantilevered helideck over the bow

    Key features (body Y / height Z):
        Aft accommodation block   Y ≈ -4 to  +6,  Z up to 29 m
        Working deck (mid-level)  Y ≈ +6 to +19,  Z up to 19 m
        Forward accomm / bridge   Y ≈ +19 to +50, Z up to 29 m
        Helideck (cantilevered)   Y ≈ +46 to +58, Z ≈ 29 m
    """
    deck_z = MIDSHIP_FREEBOARD  # ~ 7.0 m above waterline

    blocks = []

    # ── Aft accommodation block ─────────────────────────────────
    # Lower accommodation: Y=-4 to +6, 4 decks (~16m above deck)
    blocks.append(pv.Box(bounds=(
        -9.5, 9.5,         # x: port-starboard (nearly full beam)
        -6.0, 6.0,         # y: body frame
        deck_z, 23.0,      # z: from deck to top of lower block
    )))
    # Upper accommodation step-back (narrower, taller)
    blocks.append(pv.Box(bounds=(
        -8.0, 8.0,
        -4.0, 5.0,
        23.0, 29.0,
    )))
    # Wheelhouse / bridge deck on top of aft block
    blocks.append(pv.Box(bounds=(
        -6.0, 6.0,
        -2.0, 4.0,
        29.0, 32.0,
    )))
    # Bridge wings (thin extensions port & starboard)
    blocks.append(pv.Box(bounds=(
        -10.5, -6.0,       # port bridge wing
        -1.0, 2.0,
        29.0, 31.0,
    )))
    blocks.append(pv.Box(bounds=(
        6.0, 10.5,         # starboard bridge wing
        -1.0, 2.0,
        29.0, 31.0,
    )))

    # ── Offshore crane on aft deck ─────────────────────────────
    # Crane pedestal (on centreline, aft of accommodation)
    blocks.append(pv.Box(bounds=(
        -2.0, 2.0,
        -16.0, -12.0,
        deck_z, 16.0,       # pedestal ~9m above deck
    )))
    # Crane A-frame / boom base (wider at top)
    blocks.append(pv.Box(bounds=(
        -3.0, 3.0,
        -17.0, -11.0,
        16.0, 20.0,
    )))
    # Crane boom (extending to port and slightly aft)
    blocks.append(pv.Box(bounds=(
        -20.0, -2.0,
        -15.5, -13.5,
        18.0, 19.5,
    )))

    # ── Exhaust stacks (on roof of aft accommodation, well aft) ─
    blocks.append(pv.Box(bounds=(
        -2.0, 0.0,
        -5.0, -3.5,
        32.0, 36.0,         # short stacks on aft wheelhouse roof
    )))
    blocks.append(pv.Box(bounds=(
        1.0, 3.0,
        -5.0, -3.5,
        32.0, 36.0,
    )))

    # ── Working deck (mid-level between blocks) ─────────────────
    # Starboard side: working deck with crane pedestal area
    blocks.append(pv.Box(bounds=(
        -1.0, 9.5,         # starboard half + a bit past centre
        6.0, 19.0,
        deck_z, 14.0,      # lower level
    )))
    # Crane pedestal (starboard side, between blocks)
    blocks.append(pv.Box(bounds=(
        3.0, 7.0,
        10.0, 14.0,
        14.0, 19.0,
    )))

    # ── Forward accommodation / bridge block ────────────────────
    # Main forward block: Y=+19 to +49, ~29m tall
    blocks.append(pv.Box(bounds=(
        -9.5, 9.5,
        19.0, 49.0,
        deck_z, 23.0,
    )))
    # Upper forward block (step-back)
    blocks.append(pv.Box(bounds=(
        -8.5, 8.5,
        22.0, 47.0,
        23.0, 29.0,
    )))
    # Forward wheelhouse on top (with windows — just a box for now)
    blocks.append(pv.Box(bounds=(
        -7.0, 7.0,
        36.0, 47.0,
        29.0, 33.0,
    )))
    # Forward bridge wings
    blocks.append(pv.Box(bounds=(
        -11.0, -7.0,       # port
        38.0, 44.0,
        29.0, 31.5,
    )))
    blocks.append(pv.Box(bounds=(
        7.0, 11.0,         # starboard
        38.0, 44.0,
        29.0, 31.5,
    )))

    # ── Helideck cantilevered over the bow ──────────────────────
    # Large platform extending forward past the hull bow
    blocks.append(pv.Box(bounds=(
        -10.0, 10.0,
        46.0, 60.0,        # extends well past hull bow (~54m)
        29.0, 29.8,        # thin deck plate at top of forward block
    )))
    # Helideck support columns (port & starboard)
    blocks.append(pv.Box(bounds=(
        -10.0, -8.5,
        49.0, 58.0,
        23.0, 29.0,
    )))
    blocks.append(pv.Box(bounds=(
        8.5, 10.0,
        49.0, 58.0,
        23.0, 29.0,
    )))

    # ── Mast / antenna platform (on top of aft wheelhouse) ──────
    # Centred on the aft wheelhouse roof (Y=-2 to +4), well clear of gangway
    # Thin mast pole
    blocks.append(pv.Box(bounds=(
        -0.5, 0.5,
        0.0, 1.0,
        32.0, 40.0,
    )))
    # Antenna platform / radar scanner at top
    blocks.append(pv.Box(bounds=(
        -2.5, 2.5,
        -0.5, 1.5,
        38.0, 39.0,
    )))

    mesh = blocks[0]
    for b in blocks[1:]:
        mesh = mesh.merge(b)
    return mesh


def _build_gangway_tower() -> pv.PolyData:
    """Walk-to-work gangway tower on the port side.

    Positioned at roughly Y ≈ +6 to +14 (about 5–10 m forward of midship).
    The tower sits on the port side of the working deck and rises to ~37 m
    (the highest point on the vessel profile).

    Based on the reference image, the gangway system consists of:
        - A wide base/platform (housing the motion compensation system)
        - A tall cylindrical tower (approximated as octagonal prism)
        - The gangway boom arm extending to port
        - A radome (dome) near the tower top
    """
    deck_z = MIDSHIP_FREEBOARD  # ~ 7.0 m
    parts = []

    # ── Tower base / platform on port side ──────────────────────
    parts.append(pv.Box(bounds=(
        -11.5, -1.0,       # port side (negative X)
        6.0, 14.0,         # y range
        deck_z, 20.0,      # base housing
    )))

    # ── Main tower column (approximated as box — could use cylinder) ─
    parts.append(pv.Box(bounds=(
        -8.5, -3.5,
        8.5, 12.5,
        20.0, 37.0,        # rises to 37 m (peak)
    )))

    # ── Tower crown / equipment platform ────────────────────────
    parts.append(pv.Box(bounds=(
        -10.0, -2.0,
        8.0, 13.0,
        35.0, 37.0,        # wider platform at top
    )))

    # ── Gangway boom arm (extending to port from tower) ─────────
    # The boom connects at the tower column (~X=-8.5) and extends to port.
    # Simplified as a box; the boom is ~25m long in reality.
    parts.append(pv.Box(bounds=(
        -35.0, -8.5,       # from tower face outward to port
        9.5, 11.5,         # narrow fore-aft, centred on tower
        27.0, 28.5,        # at roughly 28m height on the tower
    )))
    # Boom tip (slightly wider — the gangway platform)
    parts.append(pv.Box(bounds=(
        -38.0, -34.0,
        8.5, 12.5,
        25.0, 29.0,
    )))

    # ── Radome (dome near gangway tower) ────────────────────────
    # Approximated as a sphere
    radome = pv.Sphere(
        radius=2.5,
        center=(-6.0, 14.5, 24.0),
        theta_resolution=12,
        phi_resolution=12,
    )
    parts.append(radome)

    mesh = parts[0]
    for p in parts[1:]:
        mesh = mesh.merge(p)
    return mesh


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
        gangway = _build_gangway_tower()

        # Merge into a single mesh
        self.mesh = hull.merge(gangway)

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
