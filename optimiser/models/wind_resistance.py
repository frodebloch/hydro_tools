"""Blendermann (1994) hull wind resistance model."""

import math

import numpy as np

from .constants import (
    BLEND_ANGLES,
    BLEND_CX,
    RHO_AIR,
    WIND_AREA_FRONTAL_M2,
    WIND_AREA_LATERAL_M2,
)
from .geometry import angle_diff


def wind_resistance_kN(
    Vs: float,
    heading_deg: float,
    wind_speed: float,
    wind_dir_deg: float,
    A_frontal: float = WIND_AREA_FRONTAL_M2,
    A_lateral: float = WIND_AREA_LATERAL_M2,
) -> float:
    """Compute wind resistance on the hull using Blendermann (1994).

    Uses apparent wind (true wind + vessel speed) and angle-dependent
    force coefficients from Blendermann's tabulated data for a general
    cargo vessel.

    Returns only the INCREMENT over still-air drag, since the hull's
    calm-water resistance data already includes still-air drag from the
    vessel's own forward motion (the hull was tested in real air, not
    vacuum).

    Parameters
    ----------
    Vs : float
        Vessel speed [m/s].
    heading_deg : float
        Vessel heading [deg, compass: 0=N, 90=E].
    wind_speed : float
        True wind speed at 10m [m/s].
    wind_dir_deg : float
        True wind direction [deg, compass: direction wind comes FROM].
    A_frontal : float
        Frontal (transverse) projected wind area [m^2].
    A_lateral : float
        Lateral projected wind area [m^2].

    Returns
    -------
    float
        Wind resistance INCREMENT in the surge direction [kN].
        Positive = resistance (opposes forward motion / adds to thrust demand).
        This is the force that must be OVERCOME by the propeller, above and
        beyond the still-air drag already included in calm-water data.
    """
    if wind_speed < 0.5:
        return 0.0

    # --- Apparent wind ---
    # Same convention as Flettner: decompose true wind into body frame
    rel_angle_deg = angle_diff(heading_deg, wind_dir_deg)
    rel_angle_rad = math.radians(rel_angle_deg)

    # True wind components in body frame (surge positive = from ahead)
    wind_surge = wind_speed * math.cos(rel_angle_rad)
    wind_sway = wind_speed * math.sin(rel_angle_rad)

    # Apparent wind = true wind + vessel motion (vessel moves forward)
    app_surge = Vs + wind_surge
    app_sway = wind_sway

    app_speed = math.sqrt(app_surge ** 2 + app_sway ** 2)
    if app_speed < 0.1:
        return 0.0

    # Apparent wind angle from bow [deg], 0 = head, 90 = beam, 180 = following
    # atan2(sway, surge) gives angle in vessel body frame
    app_angle_deg = math.degrees(math.atan2(abs(app_sway), app_surge))
    # Clamp to [0, 180] (symmetric)
    app_angle_deg = max(0.0, min(180.0, app_angle_deg))

    # --- Blendermann CX (surge coefficient, referenced to A_frontal) ---
    CX = float(np.interp(app_angle_deg, BLEND_ANGLES, BLEND_CX))

    # --- Wind force (total) ---
    q = 0.5 * RHO_AIR * app_speed ** 2  # dynamic pressure [Pa]
    F_surge_total_N = CX * q * A_frontal  # [N], negative = resistance

    # --- Still-air drag (from vessel motion only, no wind) ---
    # Apparent wind = Vs from dead ahead, angle = 0 deg
    CX_head = float(BLEND_CX[0])  # CX at 0 deg (headwind)
    q_still = 0.5 * RHO_AIR * Vs ** 2
    F_surge_still_N = CX_head * q_still * A_frontal  # negative

    # --- Increment over still-air ---
    # Both are negative for headwind; increment can be positive or negative
    delta_F_N = F_surge_total_N - F_surge_still_N

    # Return as POSITIVE resistance [kN]:
    # CX is negative for headwinds (resistance) and positive for following
    # (drive).  We want R_wind = force opposing motion, so negate.
    R_wind_kN = -delta_F_N / 1000.0

    return R_wind_kN
