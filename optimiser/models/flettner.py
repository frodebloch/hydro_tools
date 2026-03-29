"""Flettner rotor model (ported from BruCon C++).

CL/CD tables from Bordogna et al. (2019) with aspect-ratio correction.
Includes Theodorsen & Regier (1944) rotor drive power model.
"""

import math

import numpy as np
from propeller_model import PchipInterpolator

from .constants import NU_AIR, RHO_AIR
from .geometry import angle_diff


class FlettnerRotor:
    """Flettner rotor model.

    Ported from brucon::propulsion_optimiser::FlettnerRotor.
    CL/CD tables from Bordogna et al. (2019) with AR correction.

    Convention:
        apparent_wind_angle: 0 = bow, 90 = starboard, 180 = stern.
    """

    # CL and CD vs spin ratio (SR = omega * r / V_wind)
    _SR_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    _CL_TABLE = np.array([0.0, 1.1, 3.0, 4.5, 6.1, 7.0, 7.2, 7.8, 8.4, 8.8, 9.0])
    _CD_TABLE = np.array([0.6, 0.8, 1.3, 2.0, 3.0, 4.2, 5.5, 6.8, 8.5, 10.0, 12.0])

    _MOTOR_EFF = 0.85       # motor + drive efficiency
    _MIN_WIND = 0.5         # m/s, below this no force

    def __init__(self, height: float, diameter: float, max_rpm: float,
                 endplate_ratio: float = 2.0, roughness_factor: float = 3.0):
        self.height = height
        self.diameter = diameter
        self.radius = diameter / 2.0
        self.max_rpm = max_rpm
        self.ref_area = diameter * height
        self.endplate_ratio = endplate_ratio
        self.roughness_factor = roughness_factor
        self._cl_interp = PchipInterpolator(self._SR_TABLE, self._CL_TABLE)
        self._cd_interp = PchipInterpolator(self._SR_TABLE, self._CD_TABLE)

        # Operating mode: target spin ratio (default) or fixed RPM
        self._use_target_sr = True
        self._target_sr = 3.0
        self._rotor_rpm = 0.0

        # Outputs
        self.surge_force_kN = 0.0
        self.sway_force_kN = 0.0
        self.cl = 0.0
        self.cd = 0.0
        self.spin_ratio = 0.0
        self.rotor_rpm = 0.0
        self.rotor_power_kW = 0.0

    def set_target_spin_ratio(self, sr: float):
        self._use_target_sr = True
        self._target_sr = max(0.0, sr)

    def set_rpm(self, rpm: float):
        self._use_target_sr = False
        self._rotor_rpm = max(0.0, min(rpm, self.max_rpm))

    def compute(self, apparent_wind_speed: float, apparent_wind_angle_deg: float):
        """Compute rotor forces from apparent wind.

        Parameters
        ----------
        apparent_wind_speed : float
            Apparent wind speed [m/s].
        apparent_wind_angle_deg : float
            Apparent wind angle [deg]. 0 = bow, 90 = starboard, 180 = stern.
        """
        self.surge_force_kN = 0.0
        self.sway_force_kN = 0.0
        self.cl = 0.0
        self.cd = 0.0
        self.spin_ratio = 0.0
        self.rotor_power_kW = 0.0

        if apparent_wind_speed < self._MIN_WIND:
            if self._use_target_sr:
                self._rotor_rpm = 0.0
            self.rotor_rpm = self._rotor_rpm
            return

        # Determine rotor RPM
        if self._use_target_sr:
            desired_rpm = (self._target_sr * apparent_wind_speed * 60.0
                           / (2.0 * math.pi * self.radius))
            self._rotor_rpm = max(0.0, min(desired_rpm, self.max_rpm))
        self.rotor_rpm = self._rotor_rpm

        # Actual spin ratio
        tip_speed = self._rotor_rpm * 2.0 * math.pi / 60.0 * self.radius
        self.spin_ratio = tip_speed / apparent_wind_speed

        # CL, CD from PCHIP interpolation
        self.cl = max(0.0, self._cl_interp(self.spin_ratio))
        self.cd = max(0.0, self._cd_interp(self.spin_ratio))

        # Dynamic pressure * reference area
        q_A = 0.5 * RHO_AIR * apparent_wind_speed ** 2 * self.ref_area
        F_lift_N = self.cl * q_A
        F_drag_N = self.cd * q_A

        beta_rad = math.radians(apparent_wind_angle_deg)
        sin_beta = math.sin(beta_rad)
        cos_beta = math.cos(beta_rad)

        # Surge: lift always contributes |sin(beta)|, drag opposes cos(beta)
        self.surge_force_kN = (F_lift_N * abs(sin_beta)
                               - F_drag_N * cos_beta) / 1000.0

        # Sway
        sign_beta = 1.0 if sin_beta >= 0.0 else -1.0
        self.sway_force_kN = (-(F_lift_N * abs(cos_beta)
                                + F_drag_N * abs(sin_beta))
                              * sign_beta / 1000.0)

        # Rotor drive power estimate
        # Theodorsen & Regier (1944) turbulent skin friction on a rotating
        # cylinder: Cf = 0.035 / Re^0.2, where Re = V_tip * r / nu_air.
        # The roughness_factor (default 3.0) accounts for endplates, surface
        # roughness, bearing/seal losses, and Magnus-reaction torque that are
        # not captured by the smooth-cylinder correlation.  k=3 gives ~35 kW
        # at 10 m/s apparent wind and ~107 kW at 15 m/s for a 28x4 m rotor,
        # consistent with published Norsepower operating data (~50-90 kW
        # in typical North Sea conditions).
        if tip_speed > 0.0:
            Re = tip_speed * self.radius / NU_AIR
            Cf = self.roughness_factor * 0.035 / Re ** 0.2
            P_aero = (Cf * 0.5 * RHO_AIR * tip_speed ** 3
                      * math.pi * self.diameter * self.height)
            self.rotor_power_kW = P_aero / (1000.0 * self._MOTOR_EFF)

    def compute_from_true_wind(self, Vs: float, heading_deg: float,
                               wind_speed: float, wind_dir_deg: float):
        """Compute rotor forces from true wind + vessel speed.

        Parameters
        ----------
        Vs : float
            Vessel speed [m/s].
        heading_deg : float
            Vessel heading [deg, compass: 0=N, 90=E].
        wind_speed : float
            True wind speed [m/s].
        wind_dir_deg : float
            True wind direction [deg, compass: direction wind comes FROM].
        """
        if wind_speed < self._MIN_WIND:
            self.compute(0.0, 0.0)
            return

        # Relative wind angle: angle from which wind comes, relative to heading
        rel_angle_deg = angle_diff(heading_deg, wind_dir_deg)
        rel_angle_rad = math.radians(rel_angle_deg)

        # Decompose into body-fixed (surge = along heading, sway = perpendicular)
        wind_surge = wind_speed * math.cos(rel_angle_rad)
        wind_sway = wind_speed * math.sin(rel_angle_rad)

        # Apparent wind = true wind + vessel velocity in body frame
        app_surge = Vs + wind_surge
        app_sway = wind_sway

        app_speed = math.sqrt(app_surge ** 2 + app_sway ** 2)
        app_angle_deg = math.degrees(math.atan2(app_sway, app_surge))

        self.compute(app_speed, app_angle_deg)
