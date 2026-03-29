"""Geometry helper functions used across multiple models."""

import math


def angle_diff(heading: float, direction: float) -> float:
    """Signed angle difference, result in [-180, 180].

    direction - heading, wrapped to [-180, 180].
    """
    d = direction - heading
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d
