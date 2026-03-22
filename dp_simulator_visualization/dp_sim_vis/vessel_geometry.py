"""Vessel geometry — 3D hull from Nemoh panel mesh plus superstructure.

The underwater hull is loaded from a Nemoh mesh file (csov_nemoh3.dat) which
provides the actual panel geometry for the CSOV (Vard 985 / Norwind SOV).
The mesh is mirrored to get the port side, extended vertically to freeboard
height, and capped with a deck surface.  An articulated gangway system is
built on top.
"""

import os
import numpy as np
import pyvista as pv
import vtk

from .udp_receiver import GangwayConfigData, GangwayStateData

# ── Vessel principal dimensions ──────────────────────────────────────────
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

# Path to the Nemoh mesh file (bundled alongside this module)
_NEMOH_MESH_PATH = os.path.join(os.path.dirname(__file__), "csov_nemoh3.dat")

# Stern and bow X positions in Nemoh coordinates (longitudinal, forward positive)
_STERN_X = -52.65
_BOW_X = 53.75


def _freeboard_at_offset(offset: float) -> float:
    """Estimate freeboard height at a longitudinal station.

    Linear interpolation of sheer: higher at bow, slightly raised at stern.
    offset is in Nemoh X coordinates (forward positive).
    """
    t = (offset - _STERN_X) / (_BOW_X - _STERN_X)  # 0 at stern, 1 at bow
    # Sheer curve: parabolic, lowest at midship
    sheer = STERN_SHEER * (1.0 - t) ** 2 + BOW_SHEER * t**2
    return MIDSHIP_FREEBOARD + sheer


def _parse_nemoh_mesh(path: str) -> tuple[np.ndarray, list[list[int]]]:
    """Parse a Nemoh-format mesh file.

    Returns
    -------
    vertices : ndarray, shape (N, 3)
        Vertex coordinates (X=fwd, Y=stbd, Z=up, waterline=0).
    panels : list of list of int
        Quad panels as 0-indexed vertex indices.
    """
    vertices = []
    panels = []
    with open(path) as f:
        _header = f.readline()  # "2  1" — format version, symmetry flag
        # Read vertices until terminator (index 0)
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "0":
                break
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        # Read panels until terminator (all zeros)
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and all(p == "0" for p in parts):
                break
            # Convert from 1-indexed to 0-indexed
            panels.append([int(p) - 1 for p in parts])
    return np.array(vertices, dtype=np.float64), panels


def _build_hull_mesh() -> pv.PolyData:
    """Build a 3D hull mesh from the Nemoh panel mesh.

    The Nemoh mesh provides the wetted (underwater) hull surface for the
    starboard half only (symmetry=1).  This function:
      1. Loads and mirrors the mesh to create the full hull below waterline
      2. Identifies the waterline edge (top row of each section)
      3. Extends the hull sides vertically from waterline to freeboard
      4. Closes the top with a deck surface

    Coordinate system (output): X = East (starboard), Y = North (forward), Z = Up.
    Waterline at Z=0, keel below, deck above.

    Nemoh coordinate system: X = forward, Y = starboard (port=0 on centreline), Z = up.
    Mapping: viz_X = nemoh_Y, viz_Y = nemoh_X, viz_Z = nemoh_Z.
    """
    nemoh_verts, nemoh_panels = _parse_nemoh_mesh(_NEMOH_MESH_PATH)

    # ── Step 1: Separate hull body panels from stern transom panels ──────
    # The Nemoh mesh has 39 sections.  Section 0 (stern) has special extra
    # vertices at the end of the vertex list that form the stern transom.
    # Regular hull sections: 11 vertices each starting from the keel (Y=0)
    # to the waterline edge (Z≈0).  Section 0 also has 11 "regular" verts
    # then 10 extra verts at indices 429-438 (0-indexed) for the transom.
    #
    # We keep the Nemoh panels as-is for the underwater surface.

    n_nemoh = len(nemoh_verts)

    # ── Step 2: Identify sections and their waterline-edge vertices ──────
    # Group vertices by X coordinate (longitudinal position) to find sections.
    # The waterline edge vertex in each section is the one with the largest Y
    # (outermost) that has Z ≈ 0.
    from collections import OrderedDict
    sections = OrderedDict()  # x_rounded -> list of (orig_idx, x, y, z)
    for i, (x, y, z) in enumerate(nemoh_verts):
        key = round(x, 2)
        if key not in sections:
            sections[key] = []
        sections[key].append((i, x, y, z))

    # For each section, find the waterline edge vertex (max Y with Z ≈ 0)
    # These are the vertices where we'll extend upward to freeboard.
    wl_edge_indices = []  # (nemoh_vert_idx, nemoh_x, nemoh_y) for each section
    section_x_values = sorted(sections.keys())
    for x_key in section_x_values:
        pts = sections[x_key]
        # Find vertices near the waterline (Z > -0.2)
        wl_pts = [(i, x, y, z) for i, x, y, z in pts if z > -0.2]
        if wl_pts:
            # Take the one with largest Y (outermost at waterline)
            best = max(wl_pts, key=lambda p: p[2])
            wl_edge_indices.append((best[0], best[1], best[2]))
        else:
            # Bow section might not reach Z=0 exactly; take highest Z vertex
            best = max(pts, key=lambda p: p[3])
            wl_edge_indices.append((best[0], best[1], best[2]))

    # ── Step 3: Build the full mesh ──────────────────────────────────────
    # We need:
    #   A) Starboard underwater hull (Nemoh panels as-is)
    #   B) Port underwater hull (mirrored: negate Y)
    #   C) Starboard freeboard side (waterline edge → deck edge, per section)
    #   D) Port freeboard side (mirrored)
    #   E) Deck surface (port deck edge → starboard deck edge, per section)
    #   F) Stern transom extension (waterline to deck)
    #   G) Bow closing

    # Start building combined vertex and face arrays.
    # First: all starboard Nemoh verts mapped to viz coords.
    # Viz: X=nemoh_Y (stbd), Y=nemoh_X (fwd), Z=nemoh_Z (up)
    viz_verts = []
    for x, y, z in nemoh_verts:
        viz_verts.append((y, x, z))  # (stbd, fwd, up)

    # A) Starboard underwater faces — directly from Nemoh panels
    faces = []
    for panel in nemoh_panels:
        faces.extend([4] + panel)

    # B) Port underwater hull — mirror all Nemoh verts (negate viz_X = negate nemoh_Y)
    port_offset = len(viz_verts)
    for x, y, z in nemoh_verts:
        viz_verts.append((-y, x, z))  # mirrored: -stbd, fwd, up

    # Port faces: same connectivity but with offset indices and reversed winding
    for panel in nemoh_panels:
        p = [idx + port_offset for idx in panel]
        faces.extend([4, p[0], p[3], p[2], p[1]])  # reversed winding for outward normals

    # C + D + E) Freeboard extension and deck ─────────────────────────────
    # For each pair of adjacent sections, add:
    #   - Starboard freeboard quad (wl_edge → deck_edge)
    #   - Port freeboard quad (mirrored)
    #   - Deck quad (stbd_deck → port_deck)

    # Create deck-edge vertices for each section (stbd and port)
    stbd_deck_indices = []  # index into viz_verts for starboard deck edge
    port_deck_indices = []  # index into viz_verts for port deck edge
    for nemoh_idx, nemoh_x, nemoh_y in wl_edge_indices:
        fb = _freeboard_at_offset(nemoh_x)
        # Starboard deck edge in viz coords
        stbd_idx = len(viz_verts)
        viz_verts.append((nemoh_y, nemoh_x, fb))  # (stbd, fwd, fb)
        stbd_deck_indices.append(stbd_idx)
        # Port deck edge in viz coords
        port_idx = len(viz_verts)
        viz_verts.append((-nemoh_y, nemoh_x, fb))  # (-stbd, fwd, fb)
        port_deck_indices.append(port_idx)

    # Now create quads between adjacent sections for freeboard and deck
    for s in range(len(wl_edge_indices) - 1):
        # Waterline edge vertex indices (starboard)
        wl_s0 = wl_edge_indices[s][0]      # stbd waterline, section s
        wl_s1 = wl_edge_indices[s + 1][0]  # stbd waterline, section s+1

        # Port waterline edge (mirrored vertices)
        wl_p0 = wl_s0 + port_offset
        wl_p1 = wl_s1 + port_offset

        # Deck edge indices
        dk_s0 = stbd_deck_indices[s]
        dk_s1 = stbd_deck_indices[s + 1]
        dk_p0 = port_deck_indices[s]
        dk_p1 = port_deck_indices[s + 1]

        # Starboard freeboard side: waterline → deck
        faces.extend([4, wl_s0, wl_s1, dk_s1, dk_s0])

        # Port freeboard side: waterline → deck (reversed winding)
        faces.extend([4, wl_p0, dk_p0, dk_p1, wl_p1])

        # Deck surface: stbd deck → port deck
        faces.extend([4, dk_s0, dk_s1, dk_p1, dk_p0])

    # F) Stern transom extension ──────────────────────────────────────────
    # Close the stern (section 0) from waterline up to deck.
    # The stern transom in Nemoh is already closed at waterline.
    # We add a flat quad from stern waterline to stern deck.
    # Section 0 waterline edge:
    stern_wl_stbd = wl_edge_indices[0][0]
    stern_wl_port = stern_wl_stbd + port_offset
    stern_dk_stbd = stbd_deck_indices[0]
    stern_dk_port = port_deck_indices[0]
    # Stern centreline vertices at waterline and deck
    # The keel vertex at section 0 is at Y=0 (centreline) — index 0 in Nemoh
    stern_keel = 0  # centreline at keel
    stern_cl_wl_idx = len(viz_verts)
    nemoh_x_stern = nemoh_verts[0][0]  # X position of stern
    viz_verts.append((0.0, nemoh_x_stern, 0.0))  # centreline at waterline
    stern_cl_dk_idx = len(viz_verts)
    fb_stern = _freeboard_at_offset(nemoh_x_stern)
    viz_verts.append((0.0, nemoh_x_stern, fb_stern))  # centreline at deck

    # Stern transom face: two quads (stbd half + port half) from waterline to deck
    faces.extend([4, stern_cl_wl_idx, stern_wl_stbd, stern_dk_stbd, stern_cl_dk_idx])
    faces.extend([4, stern_wl_port, stern_cl_wl_idx, stern_cl_dk_idx, stern_dk_port])

    # Also close the below-waterline stern transom centreline gap if needed
    # (Nemoh already has stern transom panels, but the mirrored port side
    # may leave a gap at the centreline — the Nemoh transom verts 429-438
    # are at Y≥0 so the port mirror will handle them)

    # G) Bow closing ──────────────────────────────────────────────────────
    # The bow (last) section forms a bulbous profile in the Y-Z plane.
    # Both the keel vertex and the near-waterline vertex are on the
    # centreline (nemoh Y=0), so with the port mirror the section forms a
    # closed loop.  We close the forward face with a triangle fan from a
    # centroid point to each adjacent pair of perimeter vertices.

    bow_sec_idx = len(wl_edge_indices) - 1
    bow_wl_stbd = wl_edge_indices[bow_sec_idx][0]
    bow_wl_port = bow_wl_stbd + port_offset
    bow_dk_stbd = stbd_deck_indices[bow_sec_idx]
    bow_dk_port = port_deck_indices[bow_sec_idx]

    # Find all Nemoh vertices in the bow section (sorted by index)
    nemoh_x_bow = nemoh_verts[bow_wl_stbd][0]
    bow_nemoh_ids = sorted(
        i for i, (x, y, z) in enumerate(nemoh_verts)
        if abs(x - nemoh_x_bow) < 0.1
    )

    # Build the full perimeter loop in viz vertex indices.
    # Starboard half: from near-waterline [last] down to keel [first]
    # Port half: from keel [first+port_offset] up to near-waterline [last+port_offset]
    # The keel and near-waterline vertices are on the centreline (Y=0),
    # so we include the keel vertex once (from stbd) and the near-waterline
    # vertex once (from stbd) to avoid duplication.
    bow_perimeter = []  # viz vertex indices going around the full loop

    # Starboard: from near-waterline down to keel (inclusive)
    for nemoh_id in reversed(bow_nemoh_ids):
        bow_perimeter.append(nemoh_id)

    # Port: from one-above-keel up to one-below-waterline
    # (skip first = keel at Y=0, skip last = near-wl at Y=0 — already included)
    for nemoh_id in bow_nemoh_ids[1:-1]:
        bow_perimeter.append(nemoh_id + port_offset)

    # Add a centroid vertex for the triangle fan
    bow_centroid_yz = np.mean(
        nemoh_verts[bow_nemoh_ids][:, 1:3], axis=0
    )
    bow_fan_center = len(viz_verts)
    viz_verts.append((0.0, nemoh_x_bow, bow_centroid_yz[1]))  # centroid in viz coords (X=0 centreline approx)

    # Triangle fan: fan center to each consecutive pair on the perimeter
    n_perim = len(bow_perimeter)
    for i in range(n_perim):
        v0 = bow_perimeter[i]
        v1 = bow_perimeter[(i + 1) % n_perim]
        faces.extend([3, bow_fan_center, v0, v1])

    # Close the above-waterline bow face (freeboard extension).
    # The freeboard side narrows to a point at the bow, so close with
    # triangles from a centreline deck vertex to the deck and waterline edges.
    bow_cl_dk_idx = len(viz_verts)
    fb_bow = _freeboard_at_offset(nemoh_x_bow)
    viz_verts.append((0.0, nemoh_x_bow, fb_bow))  # centreline at deck

    # Stbd freeboard triangle: wl_edge → deck_edge → centreline_deck
    faces.extend([3, bow_wl_stbd, bow_dk_stbd, bow_cl_dk_idx])
    # Port freeboard triangle
    faces.extend([3, bow_dk_port, bow_wl_port, bow_cl_dk_idx])
    # Close from near-waterline centreline (bow_perimeter[0] = stbd near-wl = port near-wl)
    # to the deck edges via centreline
    bow_near_wl = bow_perimeter[0]  # near-waterline centreline vertex
    faces.extend([3, bow_near_wl, bow_wl_stbd, bow_cl_dk_idx])  # stbd: wl_centre → wl_stbd → deck_cl
    faces.extend([3, bow_wl_port, bow_near_wl, bow_cl_dk_idx])  # port: wl_port → wl_centre → deck_cl

    # ── Build PyVista mesh ───────────────────────────────────────────────
    vertices_arr = np.array(viz_verts, dtype=np.float64)
    faces_arr = np.array(faces, dtype=np.int64)
    mesh = pv.PolyData(vertices_arr, faces_arr)
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
        self._inner_boom_width = 1.5   # width of inner (telescoping) boom [m]
        self._inner_boom_height = 2.5  # height of inner boom [m]
        self._boom_clearance = 0.2     # clearance on each side for outer boom [m]
        self._outer_boom_width = self._inner_boom_width + 2 * self._boom_clearance
        self._outer_boom_height = self._inner_boom_height + 2 * self._boom_clearance

        # Build meshes in their local coordinate systems (centred at origin)
        self._build_meshes()

        # VTK transforms — one per part
        self._tower_base_transform = vtk.vtkTransform()
        self._tower_base_transform.PostMultiply()
        self._tower_col_transform = vtk.vtkTransform()
        self._tower_col_transform.PostMultiply()
        self._outer_boom_transform = vtk.vtkTransform()
        self._outer_boom_transform.PostMultiply()
        self._inner_boom_transform = vtk.vtkTransform()
        self._inner_boom_transform.PostMultiply()

        # Indicator ring transforms
        self._max_ring_transform = vtk.vtkTransform()
        self._max_ring_transform.PostMultiply()
        self._warn_ring_transform = vtk.vtkTransform()
        self._warn_ring_transform.PostMultiply()
        self._min_warn_ring_transform = vtk.vtkTransform()
        self._min_warn_ring_transform.PostMultiply()

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
          - outer_boom_mesh: the sleeve (fixed at min_length).
          - inner_boom_mesh: the telescoping section (scales to total_length).
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

        # Outer boom (sleeve): fixed length = min_length.
        # Larger cross-section that the inner boom slides through.
        # Built along +Y with unit length — Y-scaled to min_length in transform.
        obw = self._outer_boom_width
        obh = self._outer_boom_height
        self.outer_boom_mesh = pv.Box(bounds=(
            -obw / 2, obw / 2,    # X
            0.0, 1.0,             # Y: unit length
            -obh / 2, obh / 2,   # Z
        ))

        # Inner boom (telescoping): length varies from min_length to max_length.
        # Smaller cross-section that slides inside the outer boom.
        # Built along +Y with unit length — Y-scaled to total_length in transform.
        ibw = self._inner_boom_width
        ibh = self._inner_boom_height
        self.inner_boom_mesh = pv.Box(bounds=(
            -ibw / 2, ibw / 2,    # X
            0.0, 1.0,             # Y: unit length
            -ibh / 2, ibh / 2,   # Z
        ))

        # ── Indicator rings on the inner boom ──────────────────────────
        # Thin rectangular bands around the boom cross-section, perpendicular
        # to the boom axis (+Y).  Each ring sits at Y=0 in its local frame;
        # the transform positions it along the boom axis each frame.
        ring_proud = 0.03   # how far the ring protrudes above boom surface [m]
        ring_thick = 0.08   # thickness along boom axis (Y) [m]
        rw = ibw / 2 + ring_proud  # outer half-width in X
        rh = ibh / 2 + ring_proud  # outer half-height in Z
        ht = ring_thick / 2        # half-thickness in Y

        def _make_ring_band(rw, rh, ht):
            """Build a thin rectangular band in the XZ plane as a PolyData.

            The band is a hollow rectangle (4 flat quads forming the sides)
            at Y ∈ [-ht, +ht], outer extent ±rw in X and ±rh in Z.
            """
            # 8 vertices: outer rect at Y=-ht and Y=+ht
            verts = np.array([
                # Front face (Y = +ht), going CW viewed from +Y
                [-rw, +ht, -rh],  # 0: bottom-left
                [+rw, +ht, -rh],  # 1: bottom-right
                [+rw, +ht, +rh],  # 2: top-right
                [-rw, +ht, +rh],  # 3: top-left
                # Back face (Y = -ht)
                [-rw, -ht, -rh],  # 4: bottom-left
                [+rw, -ht, -rh],  # 5: bottom-right
                [+rw, -ht, +rh],  # 6: top-right
                [-rw, -ht, +rh],  # 7: top-left
            ], dtype=np.float64)
            # 4 side quads (the visible ring faces)
            faces = np.array([
                4, 0, 1, 5, 4,  # bottom side (-Z)
                4, 1, 2, 6, 5,  # right side (+X)
                4, 2, 3, 7, 6,  # top side (+Z)
                4, 3, 0, 4, 7,  # left side (-X)
                # Front and back faces (thin edges, barely visible but close the mesh)
                4, 0, 1, 2, 3,  # front (+Y)
                4, 5, 4, 7, 6,  # back (-Y)
            ], dtype=np.int64)
            return pv.PolyData(verts, faces)

        self.max_ring_mesh = _make_ring_band(rw, rh, ht)
        self.warn_ring_mesh = _make_ring_band(rw, rh, ht)
        self.min_warn_ring_mesh = _make_ring_band(rw, rh, ht)

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

        # ── Outer boom (sleeve): fixed at min_length ──────────────
        # Same kinematic chain as the old single boom, but Y-scaled to min_length.
        t = self._outer_boom_transform
        t.Identity()
        t.Scale(1.0, self._min_length, 1.0)  # fixed length = min_length
        t.RotateX(-boom_deg)                  # luffing: negate because negative boom = up
        t.RotateZ(-slew_deg)                  # slewing: clockwise from forward
        t.Translate(rc_x, rc_y, rc_z)         # move to rotation center
        _set_vessel_pose(t)

        # ── Inner boom (telescoping): scales to total_length ───
        t = self._inner_boom_transform
        t.Identity()
        t.Scale(1.0, total_length, 1.0)      # dynamic length
        t.RotateX(-boom_deg)
        t.RotateZ(-slew_deg)
        t.Translate(rc_x, rc_y, rc_z)
        _set_vessel_pose(t)

        # ── Indicator rings along the inner boom ───────────────────
        # Each ring is positioned at a specific distance from the rotation
        # center along the boom axis (+Y in pre-rotation frame).
        # The transform chain: translate along boom axis → luff → slew
        # → rotation center → vessel pose.
        #
        # Two marks painted on the inner boom at fixed distances from the tip:
        #
        #   max_ring (red): Fixed at (max_length - min_length) from the tip.
        #     Distance from RC = total_length - (max_length - min_length).
        #     Reaches the sleeve exit when fully extended (total == max_length).
        #     Hidden inside sleeve when retracted.
        #
        #   warn_ring (amber): 1m closer to the tip than max_ring.
        #     Distance from RC = total_length - (max_length - min_length) + 1.
        #     Emerges from the sleeve 1m before the max_ring does.
        max_protrusion = self._max_length - self._min_length
        max_ring_dist = total_length - max_protrusion
        warn_ring_dist = max_ring_dist + 1.0

        # min_warn_ring (amber): Fixed 1m from the inner boom tip.
        #   Distance from RC = total_length - 1.0 (moves with telescoping).
        #   As the gangway retracts, this ring approaches the sleeve exit.
        #   It enters the sleeve when total_length == min_length + 1,
        #   giving 1m warning before reaching minimum length.
        min_warn_ring_dist = total_length - 1.0

        for ring_t, dist in [
            (self._max_ring_transform, max_ring_dist),
            (self._warn_ring_transform, warn_ring_dist),
            (self._min_warn_ring_transform, min_warn_ring_dist),
        ]:
            ring_t.Identity()
            ring_t.Translate(0.0, dist, 0.0)  # position along boom axis
            ring_t.RotateX(-boom_deg)
            ring_t.RotateZ(-slew_deg)
            ring_t.Translate(rc_x, rc_y, rc_z)
            _set_vessel_pose(ring_t)

    @property
    def meshes_and_transforms(self) -> list[tuple[pv.PolyData, vtk.vtkTransform]]:
        """Return list of (mesh, transform) for all gangway parts."""
        return [
            (self.tower_base_mesh, self._tower_base_transform),
            (self.tower_col_mesh, self._tower_col_transform),
            (self.outer_boom_mesh, self._outer_boom_transform),
            (self.inner_boom_mesh, self._inner_boom_transform),
        ]

    @property
    def indicator_meshes_and_transforms(self) -> list[tuple[pv.PolyData, vtk.vtkTransform, str]]:
        """Return list of (mesh, transform, color_key) for boom limit indicators.

        color_key is 'max' (red) or 'warn' (amber).
        """
        return [
            (self.max_ring_mesh, self._max_ring_transform, "max"),
            (self.warn_ring_mesh, self._warn_ring_transform, "warn"),
            (self.min_warn_ring_mesh, self._min_warn_ring_transform, "warn"),
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

