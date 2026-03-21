"""Parse offshore_structures.prototxt and convert lat/lon to local NED.

The prototxt file uses a simple text-based protobuf format (not binary).
We parse it with basic string matching — no protobuf library needed.

WGS84 lat/lon to NED conversion replicates the C++ functions from
brucon/libs/common/math/navigation.cpp:
    MetersOfOneDegreeLatitude(lat)
    MetersOfOneDegreeLongitude(lat)
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OffshoreStructure:
    """Parsed offshore structure with NED position."""
    name: str
    north: float           # metres, relative to NED origin
    east: float            # metres, relative to NED origin
    azimuth_deg: float     # as-built azimuth [deg, 0=N, CW]
    floating: bool


# ── WGS84 series expansions (from navigation.cpp) ──────────────────────

def meters_of_one_degree_latitude(lat_deg: float) -> float:
    """Metres per degree of latitude at given latitude [deg]."""
    lat_rad = math.radians(lat_deg)
    return (111132.92
            - 559.82 * math.cos(2.0 * lat_rad)
            + 1.175 * math.cos(4.0 * lat_rad)
            - 0.0023 * math.cos(6.0 * lat_rad))


def meters_of_one_degree_longitude(lat_deg: float) -> float:
    """Metres per degree of longitude at given latitude [deg]."""
    lat_rad = math.radians(lat_deg)
    return (111412.84 * math.cos(lat_rad)
            - 93.5 * math.cos(3.0 * lat_rad)
            + 0.118 * math.cos(6.0 * lat_rad))


def latlon_to_ned(
    lat: float, lon: float,
    origin_lat: float, origin_lon: float,
) -> tuple[float, float]:
    """Convert WGS84 lat/lon to local NED (north, east) relative to origin.

    Uses the same series expansion as the C++ simulator so positions match.

    Parameters
    ----------
    lat, lon : float
        Target position [degrees].
    origin_lat, origin_lon : float
        NED origin [degrees].

    Returns
    -------
    north, east : float
        NED coordinates [metres].
    """
    avg_lat = (lat + origin_lat) / 2.0
    delta_lat = lat - origin_lat
    delta_lon = lon - origin_lon
    # Wrap delta_lon to [-180, 180] (matches C++ AngleDiff)
    while delta_lon > 180.0:
        delta_lon -= 360.0
    while delta_lon < -180.0:
        delta_lon += 360.0

    north = meters_of_one_degree_latitude(avg_lat) * delta_lat
    east = meters_of_one_degree_longitude(avg_lat) * delta_lon
    return north, east


# ── Prototxt parser ─────────────────────────────────────────────────────

def _parse_blocks(text: str, block_name: str) -> list[str]:
    """Extract blocks of the form `block_name { ... }`.

    Handles nested braces correctly.  Allows leading whitespace so this
    works for both top-level and indented (nested) blocks.
    """
    blocks = []
    pattern = re.compile(rf'^\s*{re.escape(block_name)}\s*\{{', re.MULTILINE)
    for m in pattern.finditer(text):
        start = m.end()
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == '{':
                depth += 1
            elif text[pos] == '}':
                depth -= 1
            pos += 1
        blocks.append(text[start:pos - 1])
    return blocks


def _get_field(block: str, field_name: str) -> str | None:
    """Extract a simple scalar field value: `field_name: value`."""
    m = re.search(
        rf'^\s*{re.escape(field_name)}\s*:\s*(.+?)\s*$',
        block, re.MULTILINE,
    )
    if m:
        val = m.group(1)
        # Strip quotes from string values
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        return val
    return None


def _get_nested_field(block: str, outer: str, inner: str) -> str | None:
    """Extract a field from a nested block: `outer { inner: value }`."""
    nested = _parse_blocks(block, outer)
    if nested:
        return _get_field(nested[0], inner)
    return None


def parse_offshore_structures(
    config_path: str | Path,
    origin_lat: float,
    origin_lon: float,
) -> list[OffshoreStructure]:
    """Parse offshore_structures.prototxt and return structures with NED positions.

    Parameters
    ----------
    config_path : str or Path
        Path to the prototxt configuration file.
    origin_lat, origin_lon : float
        NED origin in WGS84 [degrees]. This should match the vessel start
        position used by the simulator.

    Returns
    -------
    List of OffshoreStructure with NED coordinates.
    """
    text = Path(config_path).read_text()

    structures = []
    for block in _parse_blocks(text, 'offshore_structures'):
        name = _get_field(block, 'name') or 'unknown'

        lat_str = _get_nested_field(block, 'position', 'latitude')
        lon_str = _get_nested_field(block, 'position', 'longitude')
        if lat_str is None or lon_str is None:
            continue
        lat = float(lat_str)
        lon = float(lon_str)

        azimuth_str = _get_field(block, 'as_built_azimuth')
        azimuth = float(azimuth_str) if azimuth_str else 0.0

        floating_str = _get_field(block, 'floating')
        floating = floating_str is not None and floating_str.lower() == 'true'

        north, east = latlon_to_ned(lat, lon, origin_lat, origin_lon)

        structures.append(OffshoreStructure(
            name=name,
            north=north,
            east=east,
            azimuth_deg=azimuth,
            floating=floating,
        ))

    return structures
