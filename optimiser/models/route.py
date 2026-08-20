"""Route definition, great-circle navigation, and waypoints."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Waypoint:
    lat: float  # degrees N
    lon: float  # degrees E
    name: str = ""


@dataclass
class RoutePoint:
    """Position and heading at a point along the route."""
    lat: float
    lon: float
    heading_deg: float  # compass heading [deg, 0=N]
    distance_nm: float  # cumulative distance from departure [nm]
    time_hours: float   # elapsed time from departure [h]


class Route:
    """Great-circle route through waypoints at constant speed.

    Provides interpolated positions at regular time intervals (default 1 hour).
    """

    def __init__(self, waypoints: list[Waypoint], speed_kn: float):
        self.waypoints = waypoints
        self.speed_kn = speed_kn
        self._build_legs()

    def _build_legs(self):
        """Compute leg distances and cumulative distance."""
        self.legs = []
        total_nm = 0.0
        for i in range(len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i + 1]
            dist_nm = haversine_nm(wp1.lat, wp1.lon, wp2.lat, wp2.lon)
            bearing = initial_bearing(wp1.lat, wp1.lon, wp2.lat, wp2.lon)
            self.legs.append({
                "from": wp1, "to": wp2,
                "dist_nm": dist_nm,
                "bearing_deg": bearing,
                "cum_start_nm": total_nm,
            })
            total_nm += dist_nm

        self.total_distance_nm = total_nm
        self.total_time_hours = total_nm / self.speed_kn

    def interpolate(self, dt_hours: float = 1.0) -> list[RoutePoint]:
        """Interpolate route at regular time intervals.

        Returns a list of RoutePoint with position, heading, distance,
        and elapsed time at each step.
        """
        points = []
        n_steps = int(math.ceil(self.total_time_hours / dt_hours))

        for i in range(n_steps + 1):
            t_h = i * dt_hours
            if t_h > self.total_time_hours:
                t_h = self.total_time_hours
            d_nm = t_h * self.speed_kn

            # Find which leg we're on
            lat, lon, hdg = self._position_at_distance(d_nm)
            points.append(RoutePoint(
                lat=lat, lon=lon, heading_deg=hdg,
                distance_nm=d_nm, time_hours=t_h,
            ))

        return points

    def _position_at_distance(self, d_nm: float):
        """Get lat/lon/heading at cumulative distance d_nm along route."""
        for leg in self.legs:
            if d_nm <= leg["cum_start_nm"] + leg["dist_nm"] + 1e-6:
                frac = (d_nm - leg["cum_start_nm"]) / leg["dist_nm"] \
                    if leg["dist_nm"] > 0 else 0.0
                frac = max(0.0, min(1.0, frac))
                lat, lon = intermediate_point(
                    leg["from"].lat, leg["from"].lon,
                    leg["to"].lat, leg["to"].lon,
                    frac)
                return lat, lon, leg["bearing_deg"]

        # Past end: return final waypoint
        last = self.waypoints[-1]
        return last.lat, last.lon, self.legs[-1]["bearing_deg"]


# ============================================================
# Great-circle geometry helpers
# ============================================================

def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    R_nm = 3440.065  # Earth radius in nm
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R_nm * math.asin(math.sqrt(a))


def initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing (compass, 0=N) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlam))
    theta = math.atan2(x, y)
    return (math.degrees(theta) + 360) % 360


def intermediate_point(lat1: float, lon1: float,
                       lat2: float, lon2: float,
                       frac: float) -> tuple[float, float]:
    """Intermediate point on a great circle at fraction frac from p1 to p2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    lam1, lam2 = math.radians(lon1), math.radians(lon2)
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    delta = 2 * math.asin(math.sqrt(a))

    if delta < 1e-10:
        return lat1, lon1

    A = math.sin((1 - frac) * delta) / math.sin(delta)
    B = math.sin(frac * delta) / math.sin(delta)

    x = A * math.cos(phi1) * math.cos(lam1) + B * math.cos(phi2) * math.cos(lam2)
    y = A * math.cos(phi1) * math.sin(lam1) + B * math.cos(phi2) * math.sin(lam2)
    z = A * math.sin(phi1) + B * math.sin(phi2)

    lat = math.degrees(math.atan2(z, math.sqrt(x ** 2 + y ** 2)))
    lon = math.degrees(math.atan2(y, x))
    return lat, lon


# ============================================================
# Default routes
# ============================================================

# Default route: Rotterdam Europoort -> Gothenburg
ROUTE_ROTTERDAM_GOTHENBURG = [
    Waypoint(51.98, 4.05, "Rotterdam Europoort"),
    Waypoint(52.50, 3.50, "North Sea (clearing coast)"),
    Waypoint(54.00, 4.50, "Central North Sea"),
    Waypoint(56.00, 6.00, "Northern North Sea"),
    Waypoint(57.20, 7.50, "Off Hanstholm"),       # clear Jutland west coast
    Waypoint(58.00, 9.50, "Skagerrak entrance"),   # north of Skagen
    Waypoint(57.70, 11.00, "Skagerrak"),
    Waypoint(57.70, 11.80, "Gothenburg"),
]

# Return route: Gothenburg -> Rotterdam (reversed waypoints)
ROUTE_GOTHENBURG_ROTTERDAM = list(reversed(ROUTE_ROTTERDAM_GOTHENBURG))


# ============================================================
# MV Link Galaxy nominal 2-week roundtrip
# ============================================================
#
# Source: modules/config_link_galaxy/operational_profile_4.txt.in
#   (Marine Traffic verified roundtrip 01-14 Feb 2025).
# Port coordinates confirmed with operator (2026-08-20):
#   Husoy   -> Husoy, Karmoy    (59.377 N, 5.264 E)
#   Lardal  -> Laerdalsoyri     (61.101 N, 7.484 E)
# Intermediate open-sea waypoints keep the great-circle segments
# clear of the Norwegian skerries, the Jutland/Skagen peninsula,
# and the Danish straits. Distances match the schedule's 188 h at
# 10.0 - 12.5 kn (approx. 2200 nm sailing).
#
# NOTE: The Baltic leg (Kobenhavn - Swinoujscie - Lysekil, ~13-14 E)
# sits at the eastern edge of NORA3 coverage. Verify NORA3 availability
# for that box before running the annual comparison, or switch to
# BALTIC-HINDCAST for those legs.

ROUTE_LINK_GALAXY_ROUNDTRIP = [
    Waypoint(62.673,  8.549, "Sunndalsora"),
    Waypoint(62.860,  8.560, "Tingvollfjord N"),
    Waypoint(63.020,  8.100, "Kristiansund E"),
    Waypoint(63.050,  7.850, "Freifjorden (mouth)"),
    Waypoint(63.100,  7.500, "Griphavet"),
    Waypoint(62.600,  6.100, "Off Molde"),
    Waypoint(62.400,  5.100, "Off Stad (Stadhavet)"),
    Waypoint(61.900,  4.500, "Off Maloy"),
    Waypoint(60.900,  4.300, "Off Fedje"),
    Waypoint(60.200,  4.500, "Off Bergen (Marsteinen)"),
    Waypoint(59.377,  5.264, "Husoy, Karmoy"),

    Waypoint(58.500,  5.200, "Off Egersund"),
    Waypoint(57.900,  7.000, "Lindesnes SE"),
    Waypoint(58.100,  9.500, "Skagerrak central"),
    Waypoint(58.600, 10.500, "Off Faerder"),
    Waypoint(59.208, 10.945, "Fredrikstad"),

    Waypoint(58.700, 10.800, "Oslofjord exit"),
    Waypoint(57.800, 11.500, "Off Vinga"),
    Waypoint(56.700, 12.500, "Kattegat central"),
    Waypoint(56.100, 12.700, "Oresund north"),
    Waypoint(55.700, 12.596, "Kobenhavn"),

    Waypoint(54.800, 13.000, "Off Rugen"),
    Waypoint(53.911, 14.242, "Swinoujscie"),

    Waypoint(54.800, 13.000, "Off Rugen"),
    Waypoint(55.700, 12.596, "Kobenhavn (transit)"),
    Waypoint(56.100, 12.700, "Oresund north"),
    Waypoint(57.500, 11.500, "Kattegat / Skagen approach"),
    Waypoint(58.276, 11.435, "Lysekil"),

    Waypoint(58.400, 10.800, "Off Skagerrak central"),
    Waypoint(58.000,  8.000, "Lindesnes SW"),
    Waypoint(58.500,  5.500, "Off Egersund"),
    Waypoint(59.377,  5.264, "Husoy, Karmoy"),

    Waypoint(60.200,  4.500, "Off Bergen"),
    Waypoint(60.900,  4.300, "Off Fedje"),
    Waypoint(61.050,  4.700, "Sognesjoen (fjord mouth)"),
    Waypoint(61.080,  5.400, "Sognefjord W (Lavik)"),
    Waypoint(61.070,  6.100, "Sognefjord mid (Balestrand)"),
    Waypoint(61.070,  6.900, "Sognefjord E (Kaupanger)"),
    Waypoint(61.101,  7.484, "Laerdal"),
    Waypoint(61.200,  7.500, "Aardalsfjord mouth"),
    Waypoint(61.238,  7.700, "Ardalstangen"),

    Waypoint(61.200,  7.500, "Aardalsfjord mouth (return)"),
    Waypoint(61.070,  6.900, "Sognefjord E"),
    Waypoint(61.070,  6.100, "Sognefjord mid"),
    Waypoint(61.080,  5.400, "Sognefjord W"),
    Waypoint(61.050,  4.700, "Sognesjoen (out)"),
    Waypoint(61.500,  4.500, "Off Sognesjoen N"),
    Waypoint(62.400,  5.100, "Off Stad"),
    Waypoint(62.600,  6.100, "Off Molde"),
    Waypoint(63.100,  7.500, "Griphavet"),
    Waypoint(63.050,  7.850, "Freifjorden (mouth)"),
    Waypoint(63.020,  8.100, "Kristiansund E"),
    Waypoint(62.860,  8.560, "Tingvollfjord N"),
    Waypoint(62.673,  8.549, "Sunndalsora"),
]
