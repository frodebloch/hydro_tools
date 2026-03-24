"""
Environmental forces model.

Simplified wind and current models for the ploughing simulation.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """Environmental conditions."""
    # Current
    current_speed: float = 0.5      # [m/s]
    current_direction: float = 0.0  # [deg] direction FROM which current flows (met. convention)

    # Wind
    wind_speed: float = 10.0        # [m/s] at 10m height
    wind_direction: float = 0.0     # [deg] direction FROM which wind blows

    # Water depth
    water_depth: float = 100.0      # [m]


class EnvironmentModel:
    """Compute environmental forces on the vessel."""

    def __init__(self, config: EnvironmentConfig, vessel_config):
        self.cfg = config
        self.vessel = vessel_config

    def current_velocity_ned(self) -> np.ndarray:
        """Current velocity in NED [m/s]. Direction is where current flows TO."""
        # Met convention: direction FROM. Flow direction = FROM + 180
        flow_dir = np.radians(self.cfg.current_direction + 180.0)
        return np.array([
            self.cfg.current_speed * np.cos(flow_dir),
            self.cfg.current_speed * np.sin(flow_dir)
        ])

    def current_force_body(self, u: float, v: float, psi: float,
                           vessel_model) -> np.ndarray:
        """
        Current force on vessel in body frame.

        Current effectively changes the relative water velocity.
        The vessel model already uses speed through water, so current
        is accounted for by adjusting the vessel velocity inputs.

        Returns the velocity correction to apply.
        """
        # Current in NED
        Vc = self.current_velocity_ned()

        # Vessel velocity in NED
        c, s = np.cos(psi), np.sin(psi)

        # Current in body frame
        Vc_body_x = c * Vc[0] + s * Vc[1]
        Vc_body_y = -s * Vc[0] + c * Vc[1]

        return np.array([Vc_body_x, Vc_body_y])

    def wind_force_body(self, psi: float) -> np.ndarray:
        """
        Wind force on vessel in body frame [Fx, Fy, Mz].

        Uses a simplified wind force model based on projected areas.
        """
        Vw = self.cfg.wind_speed
        if Vw < 0.1:
            return np.zeros(3)

        rho_air = 1.225
        # Wind direction (FROM) in radians
        wind_from = np.radians(self.cfg.wind_direction)
        # Wind velocity NED (direction wind blows TO)
        wind_to = wind_from + np.pi

        # Relative wind angle in body frame
        gamma_w = wind_to - psi  # angle of wind relative to bow
        gamma_w = np.arctan2(np.sin(gamma_w), np.cos(gamma_w))

        # Wind force coefficients (simplified Blendermann)
        Cx = -0.5 * np.cos(gamma_w)  # surge: headwind negative
        Cy = 0.7 * np.sin(gamma_w)   # sway: beam wind
        Cn = 0.15 * np.sin(2 * gamma_w)  # yaw: quarter wind max

        A_frontal = self.vessel.frontal_wind_area
        A_lateral = self.vessel.lateral_wind_area
        L = self.vessel.lpp

        q = 0.5 * rho_air * Vw**2

        Fx = Cx * q * A_frontal
        Fy = Cy * q * A_lateral
        Mz = Cn * q * A_lateral * L

        return np.array([Fx, Fy, Mz])

    def inline_current_speed(self, track_direction: float) -> float:
        """
        Current speed component along the track direction [m/s].
        Positive = current flowing in tow direction.
        """
        Vc = self.current_velocity_ned()
        return Vc[0] * np.cos(track_direction) + Vc[1] * np.sin(track_direction)
