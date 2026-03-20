"""Vessel geometry — simple 3D hull built from CSOV cross-section data.

Uses the 39-station section data from the Norwind SOV (config_csov) to build
a realistic hull shape, plus an articulated gangway system.
"""

import math
import numpy as np
import pyvista as pv
import vtk

from .udp_receiver import GangwayConfigData, GangwayStateData

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
    """DEPRECATED: Static gangway tower — replaced by articulated GangwayGeometry.

    Kept for reference but not called.
    """
    raise NotImplementedError("Use GangwayGeometry instead")


# ── Gangway coordinate mapping ─────────────────────────────────────────
# Config body frame:  X forward, Y starboard, Z down (origin: midship/keel)
# Viz body frame:     X east (starboard), Y north (forward), Z up (origin: midship/waterline)
# Mapping: viz_X = config_Y, viz_Y = config_X, viz_Z = -config_Z - DESIGN_DRAFT
# where DESIGN_DRAFT is the distance from keel to waterline.

def _config_to_viz_body(cfg_x: float, cfg_y: float, cfg_z: float) -> tuple[float, float, float]:
    """Convert config body-frame position to viz body-frame position.

    Config: X=forward, Y=starboard, Z=down from keel.
    Viz: X=starboard(east), Y=forward(north), Z=up from waterline.
    """
    return (cfg_y, cfg_x, -cfg_z - VESSEL_DESIGN_DRAFT)


class GangwayGeometry:
    """Articulated gangway with tower base, tower column, boom, and tip platform.

    Each part has its own VTK mesh and transform. The transforms compose
    the vessel 6DOF with the gangway joint kinematics.

    Gangway kinematic chain:
        Vessel frame → Tower base (fixed on deck)
                     → Tower column (extends vertically by 'height')
                     → Boom (slews horizontally, luffs vertically, length = total_length)
                     → Tip platform (at boom end)
    """

    # Gangway slewing convention (from gangway.cpp):
    #   0 deg = forward (body +X = viz +Y)
    #   90 deg = starboard (body +Y = viz +X)
    #   180 deg = aft
    #   270 deg = port
    # Increases clockwise viewed from above.
    #
    # Boom angle convention (from gangway.cpp):
    #   Body frame Z points DOWN, so:
    #   Positive boom angle → boom tip goes DOWN
    #   Negative boom angle → boom tip goes UP
    #   Range: -30 to +30 degrees
    #
    # Height convention:
    #   Height is measured upward from the antenna_position (= gangway base).
    #   Rotation center Z (body) = antenna_position_z - height (more negative = higher)
    #   Rotation center Z (viz)  = base_viz_z + height

    # Default config matching CSOV posrefs
    DEFAULT_CONFIG = GangwayConfigData(
        base_x=5.0, base_y=-9.0, base_z=-8.0,
        max_height=25.0, min_length=18.0, max_length=32.0,
    )

    def __init__(self, config: GangwayConfigData | None = None):
        cfg = config or self.DEFAULT_CONFIG

        # Store config in viz body frame
        self._base_viz = _config_to_viz_body(cfg.base_x, cfg.base_y, cfg.base_z)
        self._max_height = cfg.max_height
        self._min_length = cfg.min_length
        self._max_length = cfg.max_length

        # The config z_position is the base of the height measurement.
        # In viz frame: base_z = -cfg_z - DRAFT.
        # The rotation center is at base_z + height.
        # Everything below the main deck (MIDSHIP_FREEBOARD) is hidden inside the hull,
        # so we only need to draw the tower column from deck level upward.

        # Geometry dimensions
        self._tower_width = 5.0      # width of tower column [m]
        self._boom_width = 2.0       # width of boom arm [m]
        self._boom_height = 1.5      # height of boom arm [m]
        self._tip_size = 4.0         # side length of tip platform [m]

        # Build meshes in their local coordinate systems (centred at origin)
        self._build_meshes()

        # VTK transforms — one per part
        self._tower_base_transform = vtk.vtkTransform()
        self._tower_base_transform.PostMultiply()
        self._tower_col_transform = vtk.vtkTransform()
        self._tower_col_transform.PostMultiply()
        self._boom_transform = vtk.vtkTransform()
        self._boom_transform.PostMultiply()
        self._tip_transform = vtk.vtkTransform()
        self._tip_transform.PostMultiply()

    def _build_meshes(self):
        """Build the gangway part meshes in local coordinates.

        The gangway base (z_position) is typically below the main deck — it is
        the origin for the height measurement, inside the hull structure.  The
        telescoping column extends upward by 'height' from that point, emerging
        through the deck when height is large enough.

        We draw:
          - tower_base_mesh: a small housing at deck level representing the
            visible pedestal / deck penetration where the column emerges.
          - tower_col_mesh: the telescoping column from the base upward by
            'height' (unit-height mesh, Z-scaled in the transform).
          - boom_mesh / tip_mesh: the boom arm and tip platform.
        """
        bx, by, bz = self._base_viz
        tw = self._tower_width

        # Deck penetration housing: sits at deck level, small visible structure.
        deck_z = MIDSHIP_FREEBOARD
        housing_height = 2.0  # 2m tall housing at deck level
        self.tower_base_mesh = pv.Box(bounds=(
            -tw * 1.2, tw * 1.2,   # local X
            -tw * 0.8, tw * 0.8,   # local Y
            0.0, housing_height,    # local Z: 0 to housing_height
        ))
        # The housing is positioned at deck level in viz frame
        self._housing_z = deck_z

        # Tower column: vertical box that grows with 'height'.
        # Built with unit height — will be Z-scaled by actual height in the transform.
        # The column starts at the gangway base (bz) and extends upward.
        # Only the portion above deck is visible, but VTK will clip/hide the
        # below-deck portion behind the hull mesh.
        self.tower_col_mesh = pv.Box(bounds=(
            -tw / 2, tw / 2,
            -tw / 2, tw / 2,
            0.0, 1.0,    # unit height — will be scaled by actual height
        ))

        # Boom arm: long narrow box. Built along +Y (forward in viz body frame)
        # with origin at the rotation center end.
        # Length 1.0 — will be scaled to total_length.
        bw = self._boom_width
        bhh = self._boom_height
        self.boom_mesh = pv.Box(bounds=(
            -bw / 2, bw / 2,    # X
            0.0, 1.0,           # Y: from rotation center outward (unit length)
            -bhh / 2, bhh / 2,  # Z
        ))

        # Tip platform: small box at the end of the boom.
        ts = self._tip_size
        self.tip_mesh = pv.Box(bounds=(
            -ts / 2, ts / 2,
            -ts / 2, ts / 2,
            -ts / 2, ts / 2,
        ))

    def update_transforms(
        self,
        vessel_north: float,
        vessel_east: float,
        vessel_heading_deg: float,
        vessel_roll_deg: float,
        vessel_pitch_deg: float,
        vessel_heave: float,
        gangway_state: GangwayStateData,
    ):
        """Update all gangway part transforms for the current frame.

        Composes vessel 6DOF with gangway joint kinematics.

        The gangway kinematic chain (from gangway.cpp):
          - z_position is the gangway base (height measurement origin)
          - rotation center is at z_position + height (upward)
          - boom extends from rotation center at slewing/boom angles
        """
        bx, by, bz = self._base_viz
        height = gangway_state.height
        slew_deg = gangway_state.slewing_angle
        boom_deg = gangway_state.boom_angle
        total_length = gangway_state.total_length

        # ── Helper: build vessel rotation transform ──────────────────
        # This is the same rotation order as VesselGeometry.update_transform
        def _set_vessel_pose(t: vtk.vtkTransform):
            """Apply vessel 6DOF to a transform (PostMultiply mode)."""
            t.RotateX(vessel_pitch_deg)     # pitch about X, +bow up
            t.RotateY(vessel_roll_deg)      # roll about Y, +stbd down
            t.RotateZ(-vessel_heading_deg)
            t.Translate(vessel_east, vessel_north, -vessel_heave)  # heave: sim +down (NED), viz +up → negate

        # ── Tower base housing: fixed on vessel at deck level ─────────
        t = self._tower_base_transform
        t.Identity()
        t.Translate(bx, by, self._housing_z)  # housing sits at deck level
        _set_vessel_pose(t)

        # ── Tower column: from base reference upward by 'height' ─────
        t = self._tower_col_transform
        t.Identity()
        t.Scale(1.0, 1.0, max(height, 0.1))  # scale Z to actual height
        t.Translate(bx, by, bz)               # column starts at base reference
        _set_vessel_pose(t)

        # ── Boom: at rotation center (base + height) ─────────────────
        # Rotation center position in viz body frame:
        rc_x = bx
        rc_y = by
        rc_z = bz + height

        # The boom in its local frame points along +Y (forward).
        # We need to:
        # 1. Scale Y to total_length
        # 2. Rotate to apply slewing (about local Z) and luffing (about local X)
        #
        # Slewing convention: 0=forward (+Y in viz), 90=starboard (+X in viz), clockwise.
        # In VTK: RotateZ rotates CCW, so slewing_deg clockwise = RotateZ(-slewing_deg).
        #
        # Boom angle convention (from gangway.cpp):
        #   In the body frame (Z-down): dz = sin(boom_angle) * length
        #   Positive boom_angle → positive dz → downward in body frame
        #   Negative boom_angle → upward (boom tip above rotation center)
        #
        # In viz frame (Z-up), we need to negate: the VTK RotateX(+angle)
        # rotates +Y towards +Z (up), but a negative boom_angle means "up",
        # so we apply RotateX(-boom_deg) to get the correct visual tilt.

        t = self._boom_transform
        t.Identity()
        t.Scale(1.0, total_length, 1.0)  # scale Y to actual boom length
        t.RotateX(-boom_deg)             # luffing: negate because negative boom = up
        t.RotateZ(-slew_deg)             # slewing: clockwise from forward
        t.Translate(rc_x, rc_y, rc_z)    # move to rotation center
        _set_vessel_pose(t)

        # ── Tip platform: at the end of the boom ─────────────────────
        # Compute tip position in viz body frame using the gangway kinematics.
        # From gangway.cpp, body-frame vector from rotation center to bumper:
        #   dx_body = cos(boom) * length * cos(slew)   [forward]
        #   dy_body = cos(boom) * length * sin(slew)   [starboard]
        #   dz_body = sin(boom) * length               [DOWN in body frame]
        #
        # Convert to viz body frame (X=starboard, Y=forward, Z=up):
        #   viz_dx = dy_body (starboard)
        #   viz_dy = dx_body (forward)
        #   viz_dz = -dz_body (negate: body Z-down → viz Z-up)
        slew_rad = math.radians(slew_deg)
        boom_rad = math.radians(boom_deg)
        cos_boom = math.cos(boom_rad)
        sin_boom = math.sin(boom_rad)
        cos_slew = math.cos(slew_rad)
        sin_slew = math.sin(slew_rad)

        # Config body-frame offset from rotation center to tip
        dx_cfg = cos_boom * total_length * cos_slew  # forward
        dy_cfg = cos_boom * total_length * sin_slew  # starboard
        dz_cfg = sin_boom * total_length              # DOWN (body frame)

        # Convert to viz body frame
        tip_x = rc_x + dy_cfg    # starboard
        tip_y = rc_y + dx_cfg    # forward
        tip_z = rc_z - dz_cfg    # up (negate body Z-down)

        t = self._tip_transform
        t.Identity()
        # Tip orientation follows boom (slewing + luffing)
        t.RotateX(-boom_deg)
        t.RotateZ(-slew_deg)
        t.Translate(tip_x, tip_y, tip_z)
        _set_vessel_pose(t)

    @property
    def meshes_and_transforms(self) -> list[tuple[pv.PolyData, vtk.vtkTransform]]:
        """Return list of (mesh, transform) for all gangway parts."""
        return [
            (self.tower_base_mesh, self._tower_base_transform),
            (self.tower_col_mesh, self._tower_col_transform),
            (self.boom_mesh, self._boom_transform),
            (self.tip_mesh, self._tip_transform),
        ]


class VesselGeometry:
    """Complete vessel geometry with hull and articulated gangway.

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
        self.mesh = _build_hull_mesh()

        # Articulated gangway (initially with default CSOV config)
        self.gangway = GangwayGeometry()

        # VTK transform for fast rigid-body updates (no mesh point modification)
        self._vtk_transform = vtk.vtkTransform()
        self._vtk_transform.PostMultiply()

    def set_gangway_config(self, config: GangwayConfigData):
        """Re-create the gangway geometry with the given configuration."""
        self.gangway = GangwayGeometry(config)

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
            Pitch angle [deg], positive = bow up.
        heave : float
            Vertical displacement [m], positive down.
        """
        t = self._vtk_transform
        t.Identity()
        # Order: rotate in body frame (pitch, roll, yaw), then translate.
        # With PostMultiply, transforms are applied left-to-right.
        #
        # Viz frame: X=East(starboard), Y=North(forward), Z=Up
        # Simulator body frame: X=fwd, Y=stbd, Z=down
        # Pitch = rotation about starboard axis = X in viz. +pitch = bow UP.
        #         RotateX(+pitch_deg): +Y(bow) rotates towards +Z(up). ✓
        # Roll  = rotation about forward axis = Y in viz. +roll = stbd down.
        #         RotateY(+roll_deg): +X(stbd) rotates towards -Z(down). ✓
        t.RotateX(pitch_deg)     # pitch about X (starboard), +bow up
        t.RotateY(roll_deg)      # roll about Y (forward), +stbd down
        t.RotateZ(-heading_deg)  # heading: NED clockwise from North, VTK rotates CCW
        t.Translate(east, north, -heave)  # heave: sim +down (NED), viz +up → negate

