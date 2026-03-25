"""
Coordinate conversion between lat/lon and north/east meters.

Based on the WGS84 series approximation from brucon::navigation.
"""

import math


def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def meters_of_one_degree_latitude(at_latitude: float) -> float:
    """Meters per degree of latitude at the given latitude (WGS84 series)."""
    lat_rad = deg_to_rad(at_latitude)
    return (
        111132.92
        - 559.82 * math.cos(2.0 * lat_rad)
        + 1.175 * math.cos(4.0 * lat_rad)
        - 0.0023 * math.cos(6.0 * lat_rad)
    )


def meters_of_one_degree_longitude(at_latitude: float) -> float:
    """Meters per degree of longitude at the given latitude (WGS84 series)."""
    lat_rad = deg_to_rad(at_latitude)
    return (
        111412.84 * math.cos(lat_rad)
        - 93.5 * math.cos(3.0 * lat_rad)
        + 0.118 * math.cos(6.0 * lat_rad)
    )


def latlon_to_north_east(
    lat_from: float, lon_from: float, lat_to: float, lon_to: float
) -> tuple[float, float]:
    """
    Convert a lat/lon displacement to north/east meters.

    Returns (north, east) in meters.
    """
    mid_lat = (lat_from + lat_to) / 2.0
    delta_lat = lat_to - lat_from
    delta_lon = lon_to - lon_from
    # Map delta_lon to [-180, 180]
    while delta_lon > 180.0:
        delta_lon -= 360.0
    while delta_lon < -180.0:
        delta_lon += 360.0

    north = meters_of_one_degree_latitude(mid_lat) * delta_lat
    east = meters_of_one_degree_longitude(mid_lat) * delta_lon
    return north, east


def north_east_to_latlon(
    delta_north: float, delta_east: float, start_lat: float, start_lon: float
) -> tuple[float, float]:
    """
    Convert a north/east displacement (meters) from a start lat/lon to a new lat/lon.

    Returns (latitude, longitude).
    """
    if abs(abs(start_lat) - 90.0) < 1e-10:
        return start_lat, start_lon

    delta_latitude = delta_north / meters_of_one_degree_latitude(start_lat)
    mid_lat = start_lat + delta_latitude / 2.0
    delta_longitude = delta_east / meters_of_one_degree_longitude(mid_lat)

    end_lat = start_lat + delta_latitude
    end_lon = start_lon + delta_longitude
    # Map to [-180, 180]
    while end_lon > 180.0:
        end_lon -= 360.0
    while end_lon < -180.0:
        end_lon += 360.0

    return end_lat, end_lon
