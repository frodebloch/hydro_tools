"""
Long-term hindcast-based voyage comparison: factory combinator vs optimiser.

Simulates repeated voyages (e.g. one departure per day over a year) on a
fixed route, using NORA3 wave hindcast data and PdStrip drift transfer
functions to compute realistic thrust demands at each hourly position.
Compares fuel consumption between the factory combinator schedule and the
fuel-optimal pitch/RPM selection.

Includes a Flettner rotor model (ported from BruCon C++) to quantify
wind-assist thrust reduction on the propeller.

Usage:
    # 1. Pre-download NORA3 data for the route (one-time):
    python voyage_comparison.py --download --year 2024

    # 2. Run the annual comparison:
    python voyage_comparison.py --year 2024

    # 3. Run with plots:
    python voyage_comparison.py --year 2024 --plot
"""

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Propeller model and optimiser imports
# ---------------------------------------------------------------------------
from propeller_model import (
    CSeriesPropeller,
    PchipInterpolator,
    load_c_series_data,
)
from optimiser import (
    _find_n_for_thrust,
    find_min_fuel_operating_point,
    make_man_l27_38,
)


# ============================================================
# Constants
# ============================================================

DATA_PATH_C440 = "/home/blofro/src/prop_model/c4_40.dat"
DATA_PATH_C455 = "/home/blofro/src/prop_model/C4_55.dat"
DATA_PATH_C470 = "/home/blofro/src/prop_model/c4_70.dat"

PDSTRIP_DAT = ("/home/blofro/src/brucon/build/bin/"
               "vessel_simulator_config/propulsion_optimiser_pdstrip.dat")

NORA3_DATA_DIR = Path(__file__).parent / "data" / "nora3_route"

# Vessel 206 parameters
PROP_DIAMETER = 4.66        # m
PROP_BAR = 0.432            # blade area ratio
PROP_DESIGN_PITCH = 0.771   # P/D

GEAR_RATIO = 800.0 / 117.6  # engine rpm / shaft rpm = 6.803
SHAFT_EFF = 0.97             # shaft line efficiency

KN_TO_MS = 0.5144           # knots to m/s
G = 9.81                    # m/s^2
RHO_WATER = 1025.0          # kg/m^3
RHO_AIR = 1.225             # kg/m^3

# Hull data from service prediction (vessel 206, with 15% sea margin)
HULL_SPEEDS_KN = np.array([8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5,
                           12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5])
HULL_WAKE = np.array([.239, .240, .242, .243, .243, .242, .242, .242,
                      .242, .242, .242, .243, .242, .242, .240, .239])
HULL_THRUST_KN = np.array([67.1, 89.8, 102.9, 116.5, 130.3, 144.2, 158.6,
                           173.5, 188.8, 204.5, 220.6, 237.3, 254.7, 273.1,
                           292.7, 315.2])
# Thrust deduction factor (from service prediction)
HULL_T_DEDUCTION = np.array([.190, .183, .177, .170, .165, .160, .155, .152,
                             .148, .145, .143, .140, .138, .136, .134, .132])

# Note: HULL_THRUST_KN already includes 15% sea margin from the service
# prediction. For our comparison we use calm-water resistance (no sea margin)
# as the baseline, then add wave and wind effects explicitly from hindcast.
# Calm-water thrust = HULL_THRUST_KN / 1.15
HULL_THRUST_CALM_KN = HULL_THRUST_KN / 1.15


# ============================================================
# Flettner rotor model (ported from BruCon C++)
# ============================================================

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

    _CF_ROTOR = 0.01        # skin friction coefficient
    _MOTOR_EFF = 0.85       # motor + drive efficiency
    _MIN_WIND = 0.5         # m/s, below this no force

    def __init__(self, height: float, diameter: float, max_rpm: float,
                 endplate_ratio: float = 2.0):
        self.height = height
        self.diameter = diameter
        self.radius = diameter / 2.0
        self.max_rpm = max_rpm
        self.ref_area = diameter * height
        self.endplate_ratio = endplate_ratio
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
        if tip_speed > 0.0:
            P_aero = (self._CF_ROTOR * 0.5 * RHO_AIR * tip_speed ** 3
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
        rel_angle_deg = _angle_diff(heading_deg, wind_dir_deg)
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


def _angle_diff(heading: float, direction: float) -> float:
    """Signed angle difference, result in [-180, 180].

    direction - heading, wrapped to [-180, 180].
    """
    d = direction - heading
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d


# ============================================================
# Blendermann wind resistance model
# ============================================================

# Vessel wind area parameters.
# These are for a general cargo / multi-purpose vessel of ~5000 DWT class,
# approximately 100m Loa.  Values based on typical general arrangement:
#   Frontal (transverse) projected wind area: superstructure + hull freeboard
#   Lateral projected wind area: full profile view above waterline
WIND_AREA_FRONTAL_M2 = 280.0   # A_F [m^2]
WIND_AREA_LATERAL_M2 = 1100.0  # A_L [m^2]
VESSEL_LOA_M = 100.0           # Overall length [m] (for yaw moment, not used here)

# Blendermann (1994) wind force coefficients for a general cargo vessel.
# Tabulated CX(alpha) and CY(alpha) where alpha = apparent wind angle
# measured from the bow (0 = head wind, 90 = beam, 180 = following).
#
# CX is the surge force coefficient (positive = driving force aft->fwd)
#   - Referenced to FRONTAL area A_F
#   - F_x = CX * 0.5 * rho_air * V_app^2 * A_F
#
# CY is the sway force coefficient (positive = force to starboard)
#   - Referenced to LATERAL area A_L
#   - F_y = CY * 0.5 * rho_air * V_app^2 * A_L
#
# We only need CX (surge) for propulsion resistance.  CY included for
# completeness but not used in thrust demand.
#
# Source: Blendermann, W. (1994). "Parameter identification of wind loads
# on ships." J. Wind Eng. Ind. Aerodyn., 51(3), 339-351.
# Vessel type 7: "General cargo / multi-purpose" (Table 3)
#
# The coefficients are symmetric about 0/180 (port/starboard symmetry).
# Tabulated at 10-degree intervals, 0-180.

_BLEND_ANGLES = np.array([
    0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
    100, 110, 120, 130, 140, 150, 160, 170, 180
])

# CX: negative = resistance (opposing forward motion), positive = drive
# Head wind CX ~ -0.70, beam CX ~ 0, following CX ~ +0.40
_BLEND_CX = np.array([
    -0.70, -0.72, -0.72, -0.68, -0.55, -0.38, -0.18, -0.02, 0.10, 0.15,
     0.18,  0.22,  0.25,  0.30,  0.32,  0.34,  0.38,  0.40, 0.38
])

# CY: zero at 0/180, peak ~+0.70 at beam (referenced to A_L)
_BLEND_CY = np.array([
    0.00, 0.12, 0.25, 0.40, 0.54, 0.65, 0.72, 0.74, 0.72, 0.66,
    0.58, 0.48, 0.37, 0.26, 0.16, 0.08, 0.03, 0.00, 0.00
])


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
    rel_angle_deg = _angle_diff(heading_deg, wind_dir_deg)
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
    CX = float(np.interp(app_angle_deg, _BLEND_ANGLES, _BLEND_CX))

    # --- Wind force (total) ---
    q = 0.5 * RHO_AIR * app_speed ** 2  # dynamic pressure [Pa]
    F_surge_total_N = CX * q * A_frontal  # [N], negative = resistance

    # --- Still-air drag (from vessel motion only, no wind) ---
    # Apparent wind = Vs from dead ahead, angle = 0 deg
    CX_head = float(_BLEND_CX[0])  # CX at 0 deg (headwind)
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


# ============================================================
# Drift transfer function and added resistance
# ============================================================

class DriftTransferFunction:
    """Load PdStrip surge drift force and integrate over wave spectrum.

    The surge_d column in the PdStrip .dat file gives the mean longitudinal
    drift force per unit wave amplitude squared [N/m^2], as a function of
    wave frequency and relative heading angle.

    For head seas (PdStrip angle 180 deg), surge_d is negative (force
    opposing forward motion), so added resistance R_AW > 0.
    """

    def __init__(self, dat_path: str, speed_ms: float = 0.0):
        """Load drift TF from PdStrip .dat file.

        Parameters
        ----------
        dat_path : str
            Path to the PdStrip tab-separated .dat file.
        speed_ms : float
            Vessel speed [m/s]. Selects closest speed in the dataset.
        """
        import pandas as pd

        df = pd.read_csv(dat_path, sep=r"\s+", header=0)

        # Select closest speed
        speeds = sorted(df["speed"].unique())
        speed_map = {s: (0.0 if s < 0 else s) for s in speeds}
        best_speed = min(speeds, key=lambda s: abs(speed_map[s] - speed_ms)
                         + (0.01 if s < 0 else 0))
        print(f"DriftTF: selected speed={best_speed} "
              f"(mapped {speed_map[best_speed]:.1f} m/s) for Vs={speed_ms:.1f} m/s")

        sub = df[df["speed"] == best_speed].copy()

        self.freqs = np.array(sorted(sub["freq"].unique()))
        self.angles = np.array(sorted(sub["angle"].unique()))
        n_freq = len(self.freqs)
        n_angle = len(self.angles)

        # Build 2D array: surge_d[freq_idx, angle_idx]
        self.surge_d = np.zeros((n_freq, n_angle))
        freq_idx = {f: i for i, f in enumerate(self.freqs)}
        angle_idx = {a: i for i, a in enumerate(self.angles)}

        for _, row in sub.iterrows():
            fi = freq_idx[row["freq"]]
            ai = angle_idx[row["angle"]]
            self.surge_d[fi, ai] = row["surge_d"]

        print(f"DriftTF: {n_freq} frequencies ({self.freqs[0]:.3f}-"
              f"{self.freqs[-1]:.3f} rad/s) x {n_angle} angles "
              f"({self.angles[0]:.0f}-{self.angles[-1]:.0f} deg)")

    def _interp_surge_d(self, omega: float, mu_pdstrip: float) -> float:
        """Bilinear interpolation of surge_d at (omega, mu_pdstrip).

        Clamps to the table boundaries.
        """
        omega = np.clip(omega, self.freqs[0], self.freqs[-1])
        mu = np.clip(mu_pdstrip, self.angles[0], self.angles[-1])

        fi = np.searchsorted(self.freqs, omega) - 1
        fi = max(0, min(fi, len(self.freqs) - 2))
        ai = np.searchsorted(self.angles, mu) - 1
        ai = max(0, min(ai, len(self.angles) - 2))

        # Bilinear weights
        f0, f1 = self.freqs[fi], self.freqs[fi + 1]
        a0, a1 = self.angles[ai], self.angles[ai + 1]
        tf = (omega - f0) / (f1 - f0) if f1 != f0 else 0.0
        ta = (mu - a0) / (a1 - a0) if a1 != a0 else 0.0

        v00 = self.surge_d[fi, ai]
        v10 = self.surge_d[fi + 1, ai]
        v01 = self.surge_d[fi, ai + 1]
        v11 = self.surge_d[fi + 1, ai + 1]

        return (v00 * (1 - tf) * (1 - ta) + v10 * tf * (1 - ta)
                + v01 * (1 - tf) * ta + v11 * tf * ta)

    def added_resistance(self, Hs: float, Tp: float,
                         wave_dir_deg: float, heading_deg: float,
                         s: float = 2.0) -> float:
        """Compute mean added resistance R_AW [kN] for given sea state.

        Integrates surge drift TF over a directional JONSWAP spectrum
        with cos^2s spreading.  Fully vectorized with numpy.

        Parameters
        ----------
        Hs : float
            Significant wave height [m].
        Tp : float
            Peak period [s].
        wave_dir_deg : float
            Mean wave direction [deg, compass: dir waves come FROM].
        heading_deg : float
            Vessel heading [deg, compass: 0=N].
        s : float
            Spreading parameter for cos^2s distribution (default 2).

        Returns
        -------
        float
            Added resistance [kN], positive when opposing forward motion.
        """
        if Hs < 0.05 or Tp < 1.0:
            return 0.0

        # Relative mean wave direction in PdStrip convention
        # PdStrip: 180 = head seas (waves from bow), 0 = following seas
        # Compass: wave_dir is direction waves come FROM
        # Relative angle: wave_dir - heading (compass), then convert to PdStrip
        rel_compass = _angle_diff(heading_deg, wave_dir_deg)  # [-180, 180]
        # PdStrip convention: 0 = following, 180 = head
        # If rel_compass = 0, waves from same direction as heading = head seas = 180 PdStrip
        mu_mean_pdstrip = 180.0 - rel_compass

        # JONSWAP spectrum parameters
        omega_p = 2.0 * math.pi / Tp
        gamma_j = 3.3  # JONSWAP peak enhancement
        alpha_pm = (5.0 / 16.0) * Hs ** 2 * omega_p ** 4

        # Frequency grid (1D array)
        omega_min = max(0.15, self.freqs[0])
        omega_max = min(3.0, self.freqs[-1])
        n_omega = 60
        omegas = np.linspace(omega_min, omega_max, n_omega)
        d_omega = omegas[1] - omegas[0]

        # Directional grid (1D array)
        n_dir = 36
        d_mu_deg = 360.0 / n_dir
        mu_offsets = np.linspace(-180.0, 180.0 - d_mu_deg, n_dir)
        d_mu_rad = np.radians(d_mu_deg)

        # --- Vectorized JONSWAP spectrum S(omega) ---
        sigma = np.where(omegas <= omega_p, 0.07, 0.09)
        r_exp = np.exp(-((omegas - omega_p) ** 2)
                       / (2.0 * sigma ** 2 * omega_p ** 2))
        S = (alpha_pm / omegas ** 5
             * np.exp(-1.25 * (omega_p / omegas) ** 4)
             * gamma_j ** r_exp)  # shape (n_omega,)

        # --- Vectorized cos^2s spreading D(mu) ---
        from math import gamma as math_gamma
        norm_D = (2.0 ** (2 * s) * math_gamma(s + 1) ** 2
                  / math_gamma(2 * s + 1))
        half_off_rad = np.radians(mu_offsets) / 2.0
        cos_vals = np.cos(half_off_rad)
        D = np.abs(cos_vals) ** (2 * s) / (norm_D * math.pi)  # shape (n_dir,)

        # --- Vectorized TF lookup on 2D grid ---
        # PdStrip angles for each directional bin
        mu_pdstrip = mu_mean_pdstrip + mu_offsets  # shape (n_dir,)
        # Wrap to table range [angles[0], angles[-1]]
        angle_span = self.angles[-1] - self.angles[0]
        mu_wrapped = self.angles[0] + np.mod(mu_pdstrip - self.angles[0], angle_span)

        # Build 2D mesh: (n_omega, n_dir)
        omega_grid, mu_grid = np.meshgrid(omegas, mu_wrapped, indexing='ij')

        # Vectorized bilinear interpolation of surge_d
        omega_clipped = np.clip(omega_grid, self.freqs[0], self.freqs[-1])
        mu_clipped = np.clip(mu_grid, self.angles[0], self.angles[-1])

        fi = np.searchsorted(self.freqs, omega_clipped.ravel()) - 1
        fi = np.clip(fi, 0, len(self.freqs) - 2).reshape(omega_grid.shape)
        ai = np.searchsorted(self.angles, mu_clipped.ravel()) - 1
        ai = np.clip(ai, 0, len(self.angles) - 2).reshape(mu_grid.shape)

        f0 = self.freqs[fi]
        f1 = self.freqs[fi + 1]
        a0 = self.angles[ai]
        a1 = self.angles[ai + 1]

        tf = np.where(f1 != f0, (omega_clipped - f0) / (f1 - f0), 0.0)
        ta = np.where(a1 != a0, (mu_clipped - a0) / (a1 - a0), 0.0)

        H = (self.surge_d[fi, ai] * (1 - tf) * (1 - ta)
             + self.surge_d[fi + 1, ai] * tf * (1 - ta)
             + self.surge_d[fi, ai + 1] * (1 - tf) * ta
             + self.surge_d[fi + 1, ai + 1] * tf * ta)  # (n_omega, n_dir)

        # --- Integrate: R_AW = 2 * sum(H * S * D) * d_omega * d_mu ---
        # S has shape (n_omega,), D has shape (n_dir,)
        integrand = H * S[:, np.newaxis] * D[np.newaxis, :]  # (n_omega, n_dir)
        R_AW = 2.0 * np.sum(integrand) * d_omega * d_mu_rad

        # surge_d is negative for head seas (opposing motion), so R_AW is
        # negative. We return positive added resistance (opposing force).
        return -R_AW / 1000.0  # N -> kN


# ============================================================
# Route definition
# ============================================================

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
            dist_nm = _haversine_nm(wp1.lat, wp1.lon, wp2.lat, wp2.lon)
            bearing = _initial_bearing(wp1.lat, wp1.lon, wp2.lat, wp2.lon)
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
                lat, lon = _intermediate_point(
                    leg["from"].lat, leg["from"].lon,
                    leg["to"].lat, leg["to"].lon,
                    frac)
                return lat, lon, leg["bearing_deg"]

        # Past end: return final waypoint
        last = self.waypoints[-1]
        return last.lat, last.lon, self.legs[-1]["bearing_deg"]


def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    R_nm = 3440.065  # Earth radius in nm
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R_nm * math.asin(math.sqrt(a))


def _initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing (compass, 0=N) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlam))
    theta = math.atan2(x, y)
    return (math.degrees(theta) + 360) % 360


def _intermediate_point(lat1: float, lon1: float,
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
# NORA3 weather data along route
# ============================================================

class WeatherAlongRoute:
    """Load pre-downloaded NORA3 hindcast and extract weather at route points.

    Expects monthly NetCDF files in data_dir with naming convention:
        nora3_wave_YYYYMM.nc

    Variables used: hs, tp, thq (wave direction), ff (wind speed), dd (wind dir).
    """

    def __init__(self, data_dir: Path):
        import xarray as xr

        self.data_dir = data_dir
        self._datasets = {}   # yyyymm -> xr.Dataset
        self._lat2d = None
        self._lon2d = None
        self._lat_dims = None

    def _ensure_month(self, yyyymm: str):
        """Lazy-load a monthly dataset."""
        if yyyymm in self._datasets:
            return

        import xarray as xr

        path = self.data_dir / f"nora3_wave_{yyyymm}.nc"
        if not path.exists():
            raise FileNotFoundError(
                f"NORA3 data not found: {path}\n"
                f"Run: python voyage_comparison.py --download --year {yyyymm[:4]}")

        ds = xr.open_dataset(path)
        self._datasets[yyyymm] = ds

        # Cache lat/lon grid (same for all months)
        if self._lat2d is None:
            for name in ["latitude", "lat"]:
                if name in ds.coords or name in ds.data_vars:
                    self._lat2d = ds[name].values
                    self._lon2d = ds[{
                        "latitude": "longitude",
                        "lat": "lon",
                    }[name]].values
                    self._lat_dims = ds[name].dims
                    break
            if self._lat2d is None:
                raise ValueError("Cannot find latitude/longitude in NORA3 data")

    def _nearest_idx(self, lat: float, lon: float) -> tuple:
        """Find nearest grid point indices."""
        dist2 = (self._lat2d - lat) ** 2 + (self._lon2d - lon) ** 2
        return np.unravel_index(np.argmin(dist2), dist2.shape)

    def get_weather(self, lat: float, lon: float,
                    dt: datetime) -> dict:
        """Extract weather at (lat, lon, time).

        Returns dict with keys: hs, tp, wave_dir, wind_speed, wind_dir.
        Missing values default to 0.
        """
        yyyymm = dt.strftime("%Y%m")
        self._ensure_month(yyyymm)
        ds = self._datasets[yyyymm]

        idx = self._nearest_idx(lat, lon)
        sel = {self._lat_dims[0]: idx[0], self._lat_dims[1]: idx[1]}

        # Select nearest time
        time_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            point = ds.sel(time=time_str, method="nearest").isel(**sel)
        except Exception:
            # Fallback: find nearest time manually
            times = ds["time"].values
            target = np.datetime64(dt.replace(tzinfo=None))
            ti = int(np.argmin(np.abs(times - target)))
            point = ds.isel(time=ti, **sel)

        def _val(name):
            if name in point:
                v = float(point[name].values)
                return v if np.isfinite(v) else 0.0
            return 0.0

        return {
            "hs": _val("hs"),
            "tp": _val("tp"),
            "wave_dir": _val("thq"),    # mean wave direction [deg from]
            "wind_speed": _val("ff"),    # 10m wind speed [m/s]
            "wind_dir": _val("dd"),      # 10m wind direction [deg from]
        }

    def close(self):
        for ds in self._datasets.values():
            ds.close()
        self._datasets.clear()


# ============================================================
# NORA3 data download helper
# ============================================================

def download_nora3_for_route(year: int, route_waypoints: list[Waypoint],
                             output_dir: Path, margin_deg: float = 0.5):
    """Pre-download NORA3 wave data for a bounding box around the route.

    Downloads 12 monthly files using the existing fetch infrastructure.
    """
    # Add environment directory to path for imports
    env_dir = Path(__file__).parent.parent / "environment"
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))

    from fetch_nora3_wave import fetch_month

    # Compute bounding box
    lats = [wp.lat for wp in route_waypoints]
    lons = [wp.lon for wp in route_waypoints]
    lat_range = (min(lats) - margin_deg, max(lats) + margin_deg)
    lon_range = (min(lons) - margin_deg, max(lons) + margin_deg)

    output_dir.mkdir(parents=True, exist_ok=True)

    variables = ["hs", "tp", "thq", "ff", "dd", "latitude", "longitude"]

    print(f"\nDownloading NORA3 wave data for {year}")
    print(f"  Bounding box: lat [{lat_range[0]:.1f}, {lat_range[1]:.1f}], "
          f"lon [{lon_range[0]:.1f}, {lon_range[1]:.1f}]")
    print(f"  Variables: {variables}")
    print(f"  Output: {output_dir}")

    for month in range(1, 13):
        yyyymm = f"{year}{month:02d}"
        out_file = output_dir / f"nora3_wave_{yyyymm}.nc"
        if out_file.exists():
            print(f"\n  {yyyymm}: already exists, skipping")
            continue
        print(f"\n  Fetching {yyyymm} ...")
        fetch_month(yyyymm, output_dir, lat_range=lat_range,
                    lon_range=lon_range, variables=variables)

    print("\nDownload complete.")


# ============================================================
# Factory combinator (vessel 206)
# ============================================================

class FactoryCombinator:
    """Vessel 206 factory combinator: schedule built from propeller curve.

    Replicates BruCon's MakeNormalCombinator() algorithm:
    for each trial speed, find the (pitch, RPM) pair where the propeller
    delivers the required thrust (with 15% sea margin) AND the engine
    power matches the muzzle diagram's suggested propeller curve power.

    This places the operating line on the propeller curve drawn on the
    muzzle diagram, which sits ~300 kW below the torque limit.  In calm
    water (no sea margin), the engine is below the propeller curve.  As
    seas increase, load climbs towards and eventually past it, with
    headroom up to the torque limit for overload.
    """

    def __init__(self, engine, prop: CSeriesPropeller,
                 sea_margin: float = 0.15,
                 sg_allowance_kw: float = 0.0,
                 engine_rpm_min: Optional[float] = None,
                 engine_rpm_max: Optional[float] = None):
        self.engine = engine
        self.prop = prop
        self.gear_ratio = GEAR_RATIO
        self.sg_allowance_kw = sg_allowance_kw
        # RPM limits (default to engine's own limits)
        self._eng_rpm_min = engine_rpm_min if engine_rpm_min is not None else engine.min_rpm()
        self._eng_rpm_max = engine_rpm_max if engine_rpm_max is not None else engine.max_rpm()

        self._build_schedule(engine, prop, sea_margin)

    def _build_schedule(self, engine, prop, sea_margin):
        """Build the combinator schedule from the propeller curve.

        For each trial speed:
          1. Compute T_required from hull calm-water resistance + sea margin
          2. Sweep pitch, bisect RPM to deliver T_required at Va
          3. Select the (pitch, RPM) pair where engine power ≈ PropellerCurvePower(engineRPM)
          4. Map the resulting schedule points to lever 0-100%
        """
        min_eng_rpm = self._eng_rpm_min
        max_eng_rpm = self._eng_rpm_max
        n_min = (min_eng_rpm / self.gear_ratio) / 60.0  # rev/s
        n_max = (max_eng_rpm / self.gear_ratio) / 60.0  # rev/s

        schedule = []  # list of (T_kN, pitch, shaft_rpm)

        for i, speed_kn in enumerate(HULL_SPEEDS_KN):
            Va_ms = speed_kn * KN_TO_MS * (1.0 - float(np.interp(
                speed_kn, HULL_SPEEDS_KN, HULL_WAKE)))
            # Calm-water thrust + sea margin
            T_calm_kN = float(np.interp(speed_kn, HULL_SPEEDS_KN,
                                        HULL_THRUST_CALM_KN))
            T_req_N = T_calm_kN * (1.0 + sea_margin) * 1000.0

            best_pitch = 0.0
            best_rpm = 0.0
            best_err = 1e30

            def _try_pitch(p):
                """Try a single pitch value. Returns (err, P_engine - P_target, pitch, shaft_rpm) or None."""
                # Bisect RPM to find n that delivers T_req at this pitch
                T_at_max = prop.thrust(p, n_max, Va_ms)
                if T_at_max < T_req_N:
                    return None

                lo_n, hi_n = n_min, n_max
                for _ in range(40):
                    mid = (lo_n + hi_n) * 0.5
                    T = prop.thrust(p, mid, Va_ms)
                    if T < T_req_N:
                        lo_n = mid
                    else:
                        hi_n = mid
                n = (lo_n + hi_n) * 0.5
                shaft_rpm = n * 60.0
                eng_rpm = shaft_rpm * self.gear_ratio

                if eng_rpm < min_eng_rpm - 1.0 or eng_rpm > max_eng_rpm + 1.0:
                    return None

                Q = prop.torque(p, n, Va_ms)
                P_shaft = 2.0 * math.pi * n * Q  # Watts
                P_engine = (P_shaft / 1000.0) / SHAFT_EFF  # kW
                # Target: propeller curve power (~300 kW below torque limit).
                # Total engine load = P_propulsion + P_sg must sit on this
                # curve, keeping the same ~300 kW headroom as without SG.
                P_target = engine.propeller_curve_power(eng_rpm)  # kW
                if P_target <= 0.0:
                    return None
                # Total engine load includes SG allowance
                P_engine_total = P_engine + self.sg_allowance_kw
                if P_engine_total > engine.max_power(eng_rpm):
                    return None

                signed_err = P_engine_total - P_target
                return (abs(signed_err), signed_err, p, shaft_rpm)

            # Very coarse sweep (0.05 steps) to bracket the zero-crossing,
            # then bisect on pitch to find P_engine ≈ P_target.
            coarse_results = []
            for p_int in range(6, 31):  # pitch 0.30 to 1.50 in 0.05 steps
                p = p_int * 0.05
                r = _try_pitch(p)
                if r is not None:
                    coarse_results.append(r)
                    if r[0] < best_err:
                        best_err, _, best_pitch, best_rpm = r

            # Find the bracket where signed_err changes sign (P_engine crosses P_target)
            # Then bisect on pitch within that bracket.
            if len(coarse_results) >= 2:
                for k in range(len(coarse_results) - 1):
                    err_a = coarse_results[k][1]   # signed error
                    err_b = coarse_results[k + 1][1]
                    if err_a * err_b <= 0:  # sign change
                        p_lo = coarse_results[k][2]
                        p_hi = coarse_results[k + 1][2]
                        for _ in range(20):  # bisection on pitch
                            p_mid = (p_lo + p_hi) * 0.5
                            r = _try_pitch(p_mid)
                            if r is None:
                                p_lo = p_mid
                                continue
                            if r[0] < best_err:
                                best_err, _, best_pitch, best_rpm = r
                            if r[1] < 0:  # P_engine < P_target → need more pitch
                                p_lo = p_mid
                            else:
                                p_hi = p_mid
                        break

            if best_err < 1e29:
                schedule.append((T_req_N / 1000.0, best_pitch, best_rpm))

        # Sort by thrust
        schedule.sort(key=lambda x: x[0])

        if len(schedule) == 0:
            raise RuntimeError("FactoryCombinator: no feasible schedule points")

        # Insert lever-0 point: zero thrust at zero speed (Va=0).
        # At Va=0 with P/D=0, the propeller produces no thrust.
        min_shaft_rpm = min_eng_rpm / self.gear_ratio
        schedule.insert(0, (0.0, 0.0, min_shaft_rpm))

        N = len(schedule)
        # Map to lever: 0% = lowest thrust (feathered), 100% = highest thrust
        self._combo_lever = np.array(
            [100.0 * i / (N - 1) if N > 1 else 0.0 for i in range(N)])
        self._combo_thrust_kn = np.array([s[0] for s in schedule])
        self._combo_pitch = np.array([s[1] for s in schedule])
        self._combo_rpm = np.array([s[2] for s in schedule])

    def _rpm(self, lever: float) -> float:
        return float(np.interp(lever, self._combo_lever, self._combo_rpm))

    def _pitch(self, lever: float) -> float:
        return float(np.interp(lever, self._combo_lever, self._combo_pitch))

    def evaluate(self, T_required_N: float, Va: float) -> Optional[dict]:
        """Find the factory combinator operating point for given thrust.

        Returns dict with fuel_rate [g/h], pitch, rpm, power_kw, engine_rpm,
        eta0, or None if infeasible.
        """
        # Bisection on lever to find thrust equilibrium
        lv = self._find_lever(T_required_N, Va)
        if lv is None:
            return None

        pitch = self._pitch(lv)
        rpm = self._rpm(lv)
        n = rpm / 60.0
        eng_rpm = rpm * self.gear_ratio

        if n < 0.01 or pitch < 0.01:
            return None

        T_check = self.prop.thrust(pitch, n, Va)
        if abs(T_check - T_required_N) > 500:
            return None

        Q = self.prop.torque(pitch, n, Va)
        P_shaft = Q * 2.0 * math.pi * n
        P_shaft_kw = P_shaft / 1000.0
        P_eng_kw = P_shaft_kw / SHAFT_EFF + self.sg_allowance_kw
        eta0 = self.prop.eta0(pitch, n, Va)

        if (eng_rpm < self._eng_rpm_min
                or eng_rpm > self._eng_rpm_max
                or P_eng_kw <= 0
                or P_eng_kw > self.engine.max_power(eng_rpm)):
            return None

        fuel = self.engine.fuel_rate(P_eng_kw, eng_rpm)
        return {
            "fuel_rate": fuel,  # g/h
            "pitch": pitch,
            "rpm": rpm,
            "power_kw": P_shaft_kw,
            "engine_rpm": eng_rpm,
            "eta0": eta0,
        }

    def _find_lever(self, T_req_N: float, Va: float,
                    tol_N: float = 100) -> Optional[float]:
        """Bisection to find lever position delivering required thrust."""
        lo, hi = 0.0, 100.0

        p_hi, r_hi = self._pitch(hi), self._rpm(hi)
        n_hi = r_hi / 60.0
        T_max = self.prop.thrust(p_hi, n_hi, Va) if n_hi > 0.01 and p_hi > 0.01 else 0
        if T_req_N > T_max + tol_N:
            return None

        p_lo, r_lo = self._pitch(lo), self._rpm(lo)
        n_lo = r_lo / 60.0
        T_min = self.prop.thrust(p_lo, n_lo, Va) if n_lo > 0.01 and p_lo > 0.01 else 0
        if T_req_N < T_min - tol_N:
            return None

        for _ in range(80):
            mid = (lo + hi) / 2.0
            p = self._pitch(mid)
            r = self._rpm(mid)
            n = r / 60.0
            if n < 0.01 or p < 0.01:
                lo = mid
                continue
            T = self.prop.thrust(p, n, Va)
            if abs(T - T_req_N) < tol_N:
                return mid
            if T < T_req_N:
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2.0


# ============================================================
# Core voyage evaluation
# ============================================================

@dataclass
class HourlyResult:
    """Result for a single hourly evaluation."""
    time_hours: float
    lat: float
    lon: float
    heading_deg: float
    hs: float
    tp: float
    wind_speed: float
    R_calm_kN: float
    R_aw_kN: float
    R_wind_kN: float                                     # Blendermann wind resistance
    F_flettner_kN: float
    T_required_kN: float
    T_required_no_flettner_kN: float     # thrust without Flettner reduction
    factory_fuel_rate: Optional[float]   # g/h (with Flettner thrust reduction)
    factory_no_flettner_fuel_rate: Optional[float] = None  # g/h (without Flettner)
    optimised_fuel_rate: Optional[float] = None  # g/h (with Flettner)
    optimised_no_flettner_fuel_rate: Optional[float] = None  # g/h (without Flettner)
    factory_pitch: Optional[float] = None
    factory_rpm: Optional[float] = None
    optimised_pitch: Optional[float] = None
    optimised_rpm: Optional[float] = None


def _opt_cache_worker(args):
    """Worker function for parallel optimiser cache build."""
    T_kN_batch, prop, Va, engine, aux_kw, rpm_min, rpm_max = args
    results = {}
    for T_kN in T_kN_batch:
        T_N = T_kN * 1000.0
        op = find_min_fuel_operating_point(
            prop, Va, T_N, engine,
            gear_ratio=GEAR_RATIO,
            shaft_efficiency=SHAFT_EFF,
            auxiliary_power_kw=aux_kw,
            pitch_step=0.01,
            engine_rpm_min=rpm_min,
            engine_rpm_max=rpm_max,
        )
        if op.found and op.fuel_rate is not None:
            results[T_kN] = {
                "fuel_rate": op.fuel_rate,
                "pitch": op.pitch,
                "rpm": op.rpm,
            }
    return results


def build_optimiser_cache(
    prop: CSeriesPropeller,
    engine,
    Va: float,
    T_min_kN: float = 10.0,
    T_max_kN: float = 350.0,
    T_step_kN: float = 0.5,
    n_workers: int = 0,
    auxiliary_power_kw: float = 0.0,
    engine_rpm_min: Optional[float] = None,
    engine_rpm_max: Optional[float] = None,
) -> dict:
    """Pre-compute optimiser results for a range of thrust demands.

    Returns a dict mapping thrust [kN] (quantised to T_step_kN) to result
    dicts with fuel_rate, pitch, rpm.  This avoids running the expensive
    pitch sweep for every hourly evaluation.

    Uses multiprocessing when n_workers > 1.
    """
    T_values = np.arange(T_min_kN, T_max_kN + T_step_kN / 2, T_step_kN)
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 16)

    aux_label = f", aux={auxiliary_power_kw:.0f} kW" if auxiliary_power_kw > 0 else ""
    rpm_label = ""
    if engine_rpm_min is not None or engine_rpm_max is not None:
        lo = engine_rpm_min if engine_rpm_min is not None else engine.min_rpm()
        hi = engine_rpm_max if engine_rpm_max is not None else engine.max_rpm()
        rpm_label = f", RPM {lo:.0f}-{hi:.0f}"
    print(f"  Pre-computing optimiser for {len(T_values)} thrust points "
          f"({T_min_kN:.0f}-{T_max_kN:.0f} kN, step {T_step_kN} kN) "
          f"using {n_workers} workers{aux_label}{rpm_label} ...")

    if n_workers <= 1:
        # Sequential fallback
        cache = {}
        for T_kN in T_values:
            T_N = T_kN * 1000.0
            op = find_min_fuel_operating_point(
                prop, Va, T_N, engine,
                gear_ratio=GEAR_RATIO,
                shaft_efficiency=SHAFT_EFF,
                auxiliary_power_kw=auxiliary_power_kw,
                pitch_step=0.01,
                engine_rpm_min=engine_rpm_min,
                engine_rpm_max=engine_rpm_max,
            )
            if op.found and op.fuel_rate is not None:
                cache[T_kN] = {
                    "fuel_rate": op.fuel_rate,
                    "pitch": op.pitch,
                    "rpm": op.rpm,
                }
        print(f"  Cache built: {len(cache)} feasible / {len(T_values)} total")
        return cache

    # Split into batches for parallel execution
    batch_size = max(1, len(T_values) // n_workers)
    batches = []
    for i in range(0, len(T_values), batch_size):
        batch = T_values[i:i + batch_size].tolist()
        batches.append((batch, prop, Va, engine,
                        auxiliary_power_kw, engine_rpm_min, engine_rpm_max))

    cache = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_opt_cache_worker, b) for b in batches]
        for f in as_completed(futures):
            cache.update(f.result())

    print(f"  Cache built: {len(cache)} feasible / {len(T_values)} total")
    return cache


def build_factory_cache(
    factory: 'FactoryCombinator',
    Va: float,
    T_min_kN: float = 10.0,
    T_max_kN: float = 350.0,
    T_step_kN: float = 0.5,
) -> dict:
    """Pre-compute factory combinator results for a range of thrust demands."""
    cache = {}
    T_values = np.arange(T_min_kN, T_max_kN + T_step_kN / 2, T_step_kN)

    for T_kN in T_values:
        T_N = T_kN * 1000.0
        fr = factory.evaluate(T_N, Va)
        if fr is not None:
            cache[T_kN] = fr

    print(f"  Factory cache: {len(cache)} feasible / {len(T_values)} total")
    return cache


@dataclass
class VoyageResult:
    """Result for a complete voyage."""
    departure: datetime
    total_hours: float
    # All four on the SAME feasible-hours basis (hours where factory+Fl feasible
    # AND optimiser feasible AND factory_NF feasible AND opt_NF feasible):
    total_fuel_factory_kg: float        # factory at Flettner-reduced thrust
    total_fuel_factory_no_flettner_kg: float  # factory at no-Flettner thrust
    total_fuel_opt_no_flettner_kg: float  # optimiser at no-Flettner thrust
    total_fuel_optimised_kg: float      # optimiser at Flettner-reduced thrust
    saving_pct: float           # (factory_fl - opt_fl) / factory_fl
    saving_kg: float            # factory_fl - opt_fl
    # Split savings (same feasible-hours basis):
    # Baseline = factory no-Flettner
    saving_pitch_rpm_kg: float  # factory_nf - opt_nf
    saving_flettner_kg: float   # opt_nf - opt_fl
    mean_hs: float
    mean_wind: float
    mean_R_aw_kN: float
    mean_R_wind_kN: float
    mean_F_flettner_kN: float
    n_hours_both_feasible: int = 0
    n_hours_factory_infeasible: int = 0  # factory can't deliver, optimiser can
    n_hours_total: int = 0
    hourly: list[HourlyResult] = field(default_factory=list)


def _combine_round_trip(outbound: VoyageResult, inbound: VoyageResult) -> VoyageResult:
    """Combine outbound + return voyage into a single round-trip VoyageResult.

    Fuel totals are summed.  Weather / force means are averaged (weighted
    equally — both legs have the same number of hours at the same speed).
    Feasibility counters are summed.  Hourly lists are concatenated
    (outbound first, then return).  ``departure`` is the outbound departure.
    """
    n_out = outbound.n_hours_total or 1
    n_ret = inbound.n_hours_total or 1
    n_total = n_out + n_ret

    total_fac = outbound.total_fuel_factory_kg + inbound.total_fuel_factory_kg
    total_fac_nf = (outbound.total_fuel_factory_no_flettner_kg
                    + inbound.total_fuel_factory_no_flettner_kg)
    total_opt_nf = (outbound.total_fuel_opt_no_flettner_kg
                    + inbound.total_fuel_opt_no_flettner_kg)
    total_opt = outbound.total_fuel_optimised_kg + inbound.total_fuel_optimised_kg

    saving_kg = total_fac - total_opt
    saving_pct = 100.0 * saving_kg / total_fac if total_fac > 0 else 0.0
    saving_pr = total_fac_nf - total_opt_nf
    saving_fl = total_opt_nf - total_opt

    return VoyageResult(
        departure=outbound.departure,
        total_hours=outbound.total_hours + inbound.total_hours,
        total_fuel_factory_kg=total_fac,
        total_fuel_factory_no_flettner_kg=total_fac_nf,
        total_fuel_opt_no_flettner_kg=total_opt_nf,
        total_fuel_optimised_kg=total_opt,
        saving_pct=saving_pct,
        saving_kg=saving_kg,
        saving_pitch_rpm_kg=saving_pr,
        saving_flettner_kg=saving_fl,
        mean_hs=(outbound.mean_hs * n_out + inbound.mean_hs * n_ret) / n_total,
        mean_wind=(outbound.mean_wind * n_out
                   + inbound.mean_wind * n_ret) / n_total,
        mean_R_aw_kN=(outbound.mean_R_aw_kN * n_out
                      + inbound.mean_R_aw_kN * n_ret) / n_total,
        mean_R_wind_kN=(outbound.mean_R_wind_kN * n_out
                        + inbound.mean_R_wind_kN * n_ret) / n_total,
        mean_F_flettner_kN=(outbound.mean_F_flettner_kN * n_out
                            + inbound.mean_F_flettner_kN * n_ret) / n_total,
        n_hours_both_feasible=(outbound.n_hours_both_feasible
                               + inbound.n_hours_both_feasible),
        n_hours_factory_infeasible=(outbound.n_hours_factory_infeasible
                                    + inbound.n_hours_factory_infeasible),
        n_hours_total=outbound.n_hours_total + inbound.n_hours_total,
        hourly=outbound.hourly + inbound.hourly,
    )


@dataclass
class SpeedSweepResult:
    """Aggregated results for one speed in a speed-sensitivity sweep."""
    speed_kn: float
    transit_hours: float
    voyages_per_year: float
    n_voyages: int
    # Per-voyage means
    mean_fuel_factory_nf_kg: float
    mean_fuel_factory_fl_kg: float
    mean_fuel_opt_nf_kg: float
    mean_fuel_opt_fl_kg: float
    mean_saving_pitch_rpm_kg: float
    mean_saving_flettner_kg: float
    mean_saving_total_kg: float
    mean_saving_pct: float
    pct_pitch_rpm: float            # % of factory_nf baseline
    pct_flettner: float             # % of factory_nf baseline
    # Annualized (tonnes/year)
    ann_fuel_factory_nf_t: float
    ann_fuel_opt_fl_t: float
    ann_saving_pitch_rpm_t: float
    ann_saving_flettner_t: float
    ann_saving_total_t: float
    # Weather
    mean_hs: float
    mean_wind: float
    mean_R_aw_kN: float
    mean_R_wind_kN: float
    mean_F_flettner_kN: float
    # Feasibility
    pct_factory_infeasible: float


def evaluate_voyage(
    departure: datetime,
    route: Route,
    weather: WeatherAlongRoute,
    drift_tf: DriftTransferFunction,
    flettner: FlettnerRotor,
    factory: FactoryCombinator,
    prop: CSeriesPropeller,
    engine,
    speed_kn: float = 10.0,
    verbose: bool = False,
    _opt_cache: dict | None = None,
    _factory_cache: dict | None = None,
) -> VoyageResult:
    """Evaluate a single voyage, comparing factory vs optimised fuel consumption.

    Steps along the route at hourly intervals, computing thrust demand from:
    - Calm-water resistance (interpolated from hull data)
    - Added resistance in waves (from PdStrip drift TF + NORA3 Hs/Tp)
    - Flettner rotor thrust reduction (from NORA3 wind + heading)

    Then evaluates fuel rate under factory combinator and optimised pitch/RPM.

    Parameters
    ----------
    _opt_cache : dict, optional
        Pre-computed lookup table mapping quantised thrust [kN] to optimiser
        result dict.  Built by build_optimiser_cache() for speed.
    _factory_cache : dict, optional
        Pre-computed factory combinator lookup.
    """
    route_points = route.interpolate(dt_hours=1.0)

    # Interpolate hull data for this speed
    w = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_WAKE))
    t_ded = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_T_DEDUCTION))
    R_calm_kN = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_THRUST_CALM_KN))
    # R_calm_kN is the calm-water thrust [kN] including thrust deduction
    # i.e. T_calm = R_calm / (1 - t), but HULL_THRUST_CALM_KN is already T,
    # so calm-water resistance R = T * (1 - t)
    T_calm_kN = R_calm_kN  # This IS the calm-water thrust demand
    Vs = speed_kn * KN_TO_MS
    Va = Vs * (1.0 - w)

    hourly_results = []
    total_factory_feasible = 0.0     # factory fuel (all-four-feasible hours) [kg]
    total_factory_nf_feasible = 0.0  # factory at no-Flettner thrust [kg]
    total_optimised_feasible = 0.0   # optimiser fuel (all-four-feasible hours) [kg]
    total_opt_noflettner_feasible = 0.0  # opt-no-flettner [kg]
    n_both_feasible = 0
    n_factory_infeasible = 0
    sum_hs = 0.0
    sum_wind = 0.0
    sum_R_aw = 0.0
    sum_R_wind = 0.0
    sum_F_flettner = 0.0

    for i, rp in enumerate(route_points):
        dt = departure + timedelta(hours=rp.time_hours)

        # Get weather at this position and time
        wx = weather.get_weather(rp.lat, rp.lon, dt)

        # Added resistance from waves
        R_aw_kN = drift_tf.added_resistance(
            Hs=wx["hs"], Tp=wx["tp"],
            wave_dir_deg=wx["wave_dir"],
            heading_deg=rp.heading_deg,
        )

        # Flettner thrust
        flettner.compute_from_true_wind(
            Vs=Vs,
            heading_deg=rp.heading_deg,
            wind_speed=wx["wind_speed"],
            wind_dir_deg=wx["wind_dir"],
        )
        F_flettner_kN = max(0.0, flettner.surge_force_kN)

        # Wind resistance on hull (Blendermann model)
        R_wind_kN = wind_resistance_kN(
            Vs=Vs,
            heading_deg=rp.heading_deg,
            wind_speed=wx["wind_speed"],
            wind_dir_deg=wx["wind_dir"],
        )

        # Net thrust demand on propeller
        # T_total = T_calm + (R_aw + R_wind - F_flettner) / (1-t)
        # The added resistance, wind resistance, and Flettner force act on the
        # hull, so they modify the resistance which is then divided by (1-t)
        # to get thrust.
        T_required_kN = T_calm_kN + (R_aw_kN + R_wind_kN - F_flettner_kN) / (1.0 - t_ded)
        T_required_kN = max(0.0, T_required_kN)  # can't have negative thrust
        T_required_N = T_required_kN * 1000.0

        # Thrust demand without Flettner (for splitting savings)
        T_no_flettner_kN = T_calm_kN + (R_aw_kN + R_wind_kN) / (1.0 - t_ded)
        T_no_flettner_kN = max(0.0, T_no_flettner_kN)

        # Evaluate factory combinator
        if _factory_cache is not None:
            T_key = round(T_required_kN * 2) / 2.0
            fr = _factory_cache.get(T_key)
        else:
            fr = factory.evaluate(T_required_N, Va)

        # Evaluate optimised pitch/RPM
        o_fuel = None
        o_pitch = None
        o_rpm = None
        if T_required_N > 100:  # need meaningful thrust
            if _opt_cache is not None:
                # Use pre-computed lookup (quantised to 0.5 kN)
                T_key = round(T_required_kN * 2) / 2.0
                cached = _opt_cache.get(T_key)
                if cached is not None:
                    o_fuel = cached["fuel_rate"]
                    o_pitch = cached["pitch"]
                    o_rpm = cached["rpm"]
            else:
                op = find_min_fuel_operating_point(
                    prop, Va, T_required_N, engine,
                    gear_ratio=GEAR_RATIO,
                    shaft_efficiency=SHAFT_EFF,
                    pitch_step=0.02,
                )
                if op.found and op.fuel_rate is not None:
                    o_fuel = op.fuel_rate
                    o_pitch = op.pitch
                    o_rpm = op.rpm

        # Evaluate optimised pitch/RPM WITHOUT Flettner (same cache, different thrust)
        o_nf_fuel = None
        T_nf_N = T_no_flettner_kN * 1000.0
        if T_nf_N > 100:
            if _opt_cache is not None:
                T_nf_key = round(T_no_flettner_kN * 2) / 2.0
                cached_nf = _opt_cache.get(T_nf_key)
                if cached_nf is not None:
                    o_nf_fuel = cached_nf["fuel_rate"]
            else:
                op_nf = find_min_fuel_operating_point(
                    prop, Va, T_nf_N, engine,
                    gear_ratio=GEAR_RATIO,
                    shaft_efficiency=SHAFT_EFF,
                    pitch_step=0.02,
                )
                if op_nf.found and op_nf.fuel_rate is not None:
                    o_nf_fuel = op_nf.fuel_rate

        # Evaluate factory combinator at no-Flettner thrust (for split baseline)
        fr_nf = None
        if T_nf_N > 100:
            if _factory_cache is not None:
                T_nf_key = round(T_no_flettner_kN * 2) / 2.0
                fr_nf = _factory_cache.get(T_nf_key)
            else:
                fr_nf = factory.evaluate(T_nf_N, Va)

        hr = HourlyResult(
            time_hours=rp.time_hours,
            lat=rp.lat, lon=rp.lon,
            heading_deg=rp.heading_deg,
            hs=wx["hs"], tp=wx["tp"],
            wind_speed=wx["wind_speed"],
            R_calm_kN=T_calm_kN * (1.0 - t_ded),
            R_aw_kN=R_aw_kN,
            R_wind_kN=R_wind_kN,
            F_flettner_kN=F_flettner_kN,
            T_required_kN=T_required_kN,
            T_required_no_flettner_kN=T_no_flettner_kN,
            factory_fuel_rate=fr["fuel_rate"] if fr else None,
            factory_no_flettner_fuel_rate=fr_nf["fuel_rate"] if fr_nf else None,
            optimised_fuel_rate=o_fuel,
            optimised_no_flettner_fuel_rate=o_nf_fuel,
            factory_pitch=fr["pitch"] if fr else None,
            factory_rpm=fr["rpm"] if fr else None,
            optimised_pitch=o_pitch,
            optimised_rpm=o_rpm,
        )
        hourly_results.append(hr)

        # Accumulate fuel (g/h -> kg for 1 hour interval)
        # "all-four feasible" = hours where factory+Fl, factory_NF, opt+Fl,
        # and opt_NF are all feasible.  This ensures a consistent basis for
        # computing split savings.
        all_four = (fr is not None and fr_nf is not None
                    and o_fuel is not None and o_nf_fuel is not None)
        if all_four:
            total_factory_feasible += fr["fuel_rate"] / 1000.0
            total_factory_nf_feasible += fr_nf["fuel_rate"] / 1000.0
            total_optimised_feasible += o_fuel / 1000.0
            total_opt_noflettner_feasible += o_nf_fuel / 1000.0
            n_both_feasible += 1
        elif fr is None and o_fuel is not None:
            # Factory infeasible but optimiser can deliver
            n_factory_infeasible += 1

        sum_hs += wx["hs"]
        sum_wind += wx["wind_speed"]
        sum_R_aw += R_aw_kN
        sum_R_wind += R_wind_kN
        sum_F_flettner += F_flettner_kN

    n_pts = len(route_points)
    # Overall saving: factory (with Flettner) vs optimiser (with Flettner)
    # Both evaluated at the same Flettner-reduced thrust.
    saving_kg = total_factory_feasible - total_optimised_feasible
    saving_pct = (100.0 * saving_kg / total_factory_feasible) if total_factory_feasible > 0 else 0.0

    # Split savings (feasible-hours basis):
    # Baseline = Factory at no-Flettner thrust.
    # Total saving = Factory_NF - Opt_withFlettner.
    #   Pitch/RPM saving = Factory_NF - Opt_NF (both at same thrust, no Flettner)
    #   Flettner saving = Opt_NF - Opt_withFlettner (Flettner thrust reduction)
    saving_pitch_rpm_kg = total_factory_nf_feasible - total_opt_noflettner_feasible
    saving_flettner_kg = total_opt_noflettner_feasible - total_optimised_feasible

    if verbose:
        infeasible_str = ""
        if n_factory_infeasible > 0:
            infeasible_str = (f", factory_infeasible={n_factory_infeasible}/"
                              f"{n_pts}h")
        baseline = total_factory_nf_feasible if total_factory_nf_feasible > 0 else total_factory_feasible
        pct_pr = (100.0 * saving_pitch_rpm_kg / baseline) if baseline > 0 else 0.0
        pct_fl = (100.0 * saving_flettner_kg / baseline) if baseline > 0 else 0.0
        print(f"  {departure.strftime('%Y-%m-%d')}: "
              f"save={saving_pct:+.1f}% "
              f"(pitch/RPM {pct_pr:+.1f}%, Flettner {pct_fl:+.1f}%), "
              f"Hs={sum_hs / n_pts:.1f} m, "
              f"wind={sum_wind / n_pts:.1f} m/s"
              f"{infeasible_str}")

    return VoyageResult(
        departure=departure,
        total_hours=route.total_time_hours,
        total_fuel_factory_kg=total_factory_feasible,
        total_fuel_factory_no_flettner_kg=total_factory_nf_feasible,
        total_fuel_opt_no_flettner_kg=total_opt_noflettner_feasible,
        total_fuel_optimised_kg=total_optimised_feasible,
        saving_pct=saving_pct,
        saving_kg=saving_kg,
        saving_pitch_rpm_kg=saving_pitch_rpm_kg,
        saving_flettner_kg=saving_flettner_kg,
        mean_hs=sum_hs / n_pts,
        mean_wind=sum_wind / n_pts,
        mean_R_aw_kN=sum_R_aw / n_pts,
        mean_R_wind_kN=sum_R_wind / n_pts,
        mean_F_flettner_kN=sum_F_flettner / n_pts,
        n_hours_both_feasible=n_both_feasible,
        n_hours_factory_infeasible=n_factory_infeasible,
        n_hours_total=n_pts,
        hourly=hourly_results,
    )


# ============================================================
# Annual comparison
# ============================================================

def _voyage_worker(args):
    """Worker function for parallel voyage evaluation.

    Each worker opens its own WeatherAlongRoute (xarray datasets can't be
    pickled), evaluates a batch of departures, and returns the results.

    If a return route is provided (round-trip mode), each departure yields
    an outbound voyage followed by a return voyage whose departure time is
    the outbound arrival time.  The two legs are combined into a single
    round-trip ``VoyageResult`` via ``_combine_round_trip()``.
    """
    (departures, route, data_dir, drift_tf, flettner, factory,
     prop, engine, speed_kn, opt_cache, factory_cache,
     return_route) = args
    weather = WeatherAlongRoute(data_dir)
    results = []
    for departure in departures:
        try:
            vr_out = evaluate_voyage(
                departure=departure,
                route=route,
                weather=weather,
                drift_tf=drift_tf,
                flettner=flettner,
                factory=factory,
                prop=prop,
                engine=engine,
                speed_kn=speed_kn,
                verbose=False,
                _opt_cache=opt_cache,
                _factory_cache=factory_cache,
            )
            if return_route is not None:
                # Return leg departs when outbound arrives
                return_departure = departure + timedelta(hours=vr_out.total_hours)
                vr_ret = evaluate_voyage(
                    departure=return_departure,
                    route=return_route,
                    weather=weather,
                    drift_tf=drift_tf,
                    flettner=flettner,
                    factory=factory,
                    prop=prop,
                    engine=engine,
                    speed_kn=speed_kn,
                    verbose=False,
                    _opt_cache=opt_cache,
                    _factory_cache=factory_cache,
                )
                vr = _combine_round_trip(vr_out, vr_ret)
            else:
                vr = vr_out
            results.append(vr)
        except Exception as e:
            results.append((departure, str(e)))
    weather.close()
    return results


def run_annual_comparison(
    year: int,
    speed_kn: float = 10.0,
    waypoints: list[Waypoint] | None = None,
    data_dir: Path | None = None,
    pdstrip_path: str = PDSTRIP_DAT,
    flettner_enabled: bool = True,
    verbose: bool = True,
    sg_load_kw: float = 0.0,
    sg_factory_allowance_kw: float = 0.0,
    sg_freq_min: float = 0.0,
    sg_freq_max: float = 0.0,
    round_trip: bool = True,
) -> list[VoyageResult]:
    """Run the full annual comparison: one voyage per day.

    Parameters
    ----------
    sg_load_kw : float
        Actual shaft generator electrical load [kW] for the optimiser.
    sg_factory_allowance_kw : float
        SG power allowance baked into the factory combinator schedule [kW].
    sg_freq_min, sg_freq_max : float
        Shaft generator frequency band [Hz].  If both > 0, constrains
        engine RPM range.  The PTO gear ratio is sized so sg_freq_max
        corresponds to the engine's maximum RPM.
    round_trip : bool
        If True (default), each departure is a round trip: outbound on the
        given waypoints followed by an immediate return on the reversed
        route.  This eliminates directional wind bias from prevailing
        winds.  If False, only the outbound leg is evaluated.

    Returns a list of VoyageResult for each departure day.
    """
    if waypoints is None:
        waypoints = ROUTE_ROTTERDAM_GOTHENBURG
    if data_dir is None:
        data_dir = NORA3_DATA_DIR

    # --- Shaft generator RPM constraints ---
    engine_rpm_min_sg: Optional[float] = None
    engine_rpm_max_sg: Optional[float] = None
    if sg_freq_min > 0 and sg_freq_max > 0:
        # PTO gear ratio sized so sg_freq_max = engine max RPM
        # Engine RPM band = [max_rpm * freq_min/freq_max, max_rpm]
        _engine_max = make_man_l27_38().max_rpm()
        engine_rpm_min_sg = _engine_max * sg_freq_min / sg_freq_max
        engine_rpm_max_sg = float(_engine_max)

    print("=" * 78)
    print(f"ANNUAL VOYAGE COMPARISON: Factory Combinator vs Optimiser")
    print(f"  Year: {year}, Speed: {speed_kn} kn")
    trip_mode = "round-trip" if round_trip else "one-way"
    print(f"  Route: {waypoints[0].name} -> {waypoints[-1].name} ({trip_mode})")
    if sg_load_kw > 0 or sg_factory_allowance_kw > 0:
        print(f"  Shaft generator: factory allowance {sg_factory_allowance_kw:.0f} kW, "
              f"actual load {sg_load_kw:.0f} kW")
        if engine_rpm_min_sg is not None:
            print(f"  SG frequency band: {sg_freq_min:.0f}-{sg_freq_max:.0f} Hz "
                  f"-> engine RPM {engine_rpm_min_sg:.0f}-{engine_rpm_max_sg:.0f}")
    print("=" * 78)

    # --- Build models ---
    print("\nLoading propeller model ...")
    data_40 = load_c_series_data(DATA_PATH_C440)
    data_55 = load_c_series_data(DATA_PATH_C455)
    data_70 = load_c_series_data(DATA_PATH_C470)
    bar_data = {0.40: data_40, 0.55: data_55, 0.70: data_70}
    prop = CSeriesPropeller(bar_data, design_pitch=PROP_DESIGN_PITCH,
                            diameter=PROP_DIAMETER, area_ratio=PROP_BAR,
                            rho=RHO_WATER)
    print(f"  Propeller: D={PROP_DIAMETER}m, P/D={PROP_DESIGN_PITCH}, BAR={PROP_BAR}")

    engine = make_man_l27_38()
    print(f"  Engine: {engine.name}")

    print("\nLoading PdStrip drift transfer function ...")
    drift_tf = DriftTransferFunction(pdstrip_path, speed_ms=speed_kn * KN_TO_MS)

    print("\nSetting up Flettner rotor ...")
    flettner = FlettnerRotor(height=28.0, diameter=4.0, max_rpm=220.0,
                             endplate_ratio=2.0)
    if flettner_enabled:
        flettner.set_target_spin_ratio(3.0)
        print(f"  Rotor: H=28m, D=4m, max 220 RPM, target SR=3")
    else:
        flettner.set_target_spin_ratio(0.0)  # effectively off
        print(f"  Flettner DISABLED (--no-flettner)")

    print("\nSetting up route ...")
    route = Route(waypoints, speed_kn)
    print(f"  Outbound: {route.total_distance_nm:.0f} nm, "
          f"{route.total_time_hours:.1f} hours")
    for i, leg in enumerate(route.legs):
        print(f"    Leg {i + 1}: {leg['from'].name} -> {leg['to'].name}: "
              f"{leg['dist_nm']:.0f} nm, bearing {leg['bearing_deg']:.0f} deg")

    # Build return route (reversed waypoints) for round-trip mode
    return_route = None
    if round_trip:
        return_waypoints = list(reversed(waypoints))
        return_route = Route(return_waypoints, speed_kn)
        print(f"  Return:   {return_route.total_distance_nm:.0f} nm, "
              f"{return_route.total_time_hours:.1f} hours")
        for i, leg in enumerate(return_route.legs):
            print(f"    Leg {i + 1}: {leg['from'].name} -> {leg['to'].name}: "
                  f"{leg['dist_nm']:.0f} nm, bearing {leg['bearing_deg']:.0f} deg")

    factory = FactoryCombinator(engine, prop,
                                sg_allowance_kw=sg_factory_allowance_kw,
                                engine_rpm_min=engine_rpm_min_sg,
                                engine_rpm_max=engine_rpm_max_sg)
    print(f"  Factory combinator: {len(factory._combo_lever)} schedule points")
    if sg_factory_allowance_kw > 0:
        print(f"    SG allowance: {sg_factory_allowance_kw:.0f} kW")
    print(f"    RPM range: {factory._combo_rpm[0]:.1f} - {factory._combo_rpm[-1]:.1f} shaft")
    print(f"    Pitch range: {factory._combo_pitch[0]:.3f} - {factory._combo_pitch[-1]:.3f}")

    print("\nLoading NORA3 weather data ...")
    # Weather is loaded per-worker for parallel execution (xarray can't be pickled)
    # We just verify the data directory exists here.
    if not data_dir.exists():
        raise FileNotFoundError(f"NORA3 data directory not found: {data_dir}")

    # --- Pre-compute operating point caches ---
    w = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_WAKE))
    Vs = speed_kn * KN_TO_MS
    Va = Vs * (1.0 - w)

    print("\nPre-computing operating point caches ...")
    opt_cache = build_optimiser_cache(prop, engine, Va,
                                       auxiliary_power_kw=sg_load_kw,
                                       engine_rpm_min=engine_rpm_min_sg,
                                       engine_rpm_max=engine_rpm_max_sg)
    factory_cache = build_factory_cache(factory, Va)

    # --- Run voyages ---
    n_days = _days_in_year(year)
    n_workers = min(os.cpu_count() or 1, 16)
    voy_label = "round-trip voyages" if round_trip else "one-way voyages"
    print(f"\nRunning {n_days} {voy_label} using {n_workers} workers ...\n")

    departures = [
        datetime(year, 1, 1, 6, 0, 0, tzinfo=timezone.utc) + timedelta(days=day)
        for day in range(n_days)
    ]

    # Split departures into batches (one per worker)
    batch_size = max(1, (n_days + n_workers - 1) // n_workers)
    batches = []
    for i in range(0, n_days, batch_size):
        batch_deps = departures[i:i + batch_size]
        batches.append((
            batch_deps, route, data_dir, drift_tf, flettner,
            factory, prop, engine, speed_kn, opt_cache, factory_cache,
            return_route,
        ))

    results = []
    errors = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_voyage_worker, b) for b in batches]
        for f in as_completed(futures):
            for item in f.result():
                if isinstance(item, VoyageResult):
                    results.append(item)
                else:
                    dep, err = item
                    errors.append((dep, err))
                    print(f"  {dep.strftime('%Y-%m-%d')}: ERROR: {err}")

    # Sort by departure date
    results.sort(key=lambda r: r.departure)

    if verbose:
        for vr in results:
            inf_str = ""
            if vr.n_hours_factory_infeasible > 0:
                inf_str = f", factory_infeasible={vr.n_hours_factory_infeasible}/{vr.n_hours_total}h"
            baseline = vr.total_fuel_factory_no_flettner_kg if vr.total_fuel_factory_no_flettner_kg > 0 else vr.total_fuel_factory_kg
            pct_pr = (100.0 * vr.saving_pitch_rpm_kg / baseline) if baseline > 0 else 0.0
            pct_fl = (100.0 * vr.saving_flettner_kg / baseline) if baseline > 0 else 0.0
            print(f"  {vr.departure.strftime('%Y-%m-%d')}: "
                  f"save={vr.saving_pct:+.1f}% "
                  f"(pitch/RPM {pct_pr:+.1f}%, Flettner {pct_fl:+.1f}%)"
                  f"{inf_str}")

    if errors:
        print(f"\n  {len(errors)} voyages failed")

    return results


def _days_in_year(year: int) -> int:
    """Number of days in a given year."""
    return 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365


# ============================================================
# Summary and reporting
# ============================================================

def print_summary(results: list[VoyageResult], speed_kn: float,
                  idle_pct: float = 15.0, fuel_price_eur_per_t: float = 0.0,
                  round_trip: bool = True):
    """Print summary statistics for the annual comparison.

    Parameters
    ----------
    idle_pct : float
        Percentage of year spent idle / in port.  Used to estimate
        the number of voyages per year and annualize fuel.
    fuel_price_eur_per_t : float
        Fuel price in EUR per tonne.  If > 0, cost estimates are printed.
    round_trip : bool
        If True, results represent round trips and labels reflect that.
    """

    if not results:
        print("No results to summarise.")
        return

    transit_h = results[0].total_hours

    # Annualization: how many voyages fit in a year?
    # Available sailing hours = 365.25 * 24 * (1 - idle_pct/100)
    # Each voyage takes transit_h hours (round-trip or one-way).
    sailing_hours_year = 365.25 * 24.0 * (1.0 - idle_pct / 100.0)
    voyages_per_year = sailing_hours_year / transit_h
    voy_label = "round-trip" if round_trip else "one-way"

    savings_pct = np.array([r.saving_pct for r in results])
    savings_kg = np.array([r.saving_kg for r in results])
    savings_pitch_rpm_kg = np.array([r.saving_pitch_rpm_kg for r in results])
    savings_flettner_kg = np.array([r.saving_flettner_kg for r in results])
    fuel_factory = np.array([r.total_fuel_factory_kg for r in results])
    fuel_factory_nf = np.array([r.total_fuel_factory_no_flettner_kg for r in results])
    fuel_optimised = np.array([r.total_fuel_optimised_kg for r in results])
    fuel_opt_noflettner = np.array([r.total_fuel_opt_no_flettner_kg for r in results])
    mean_hs = np.array([r.mean_hs for r in results])
    mean_wind = np.array([r.mean_wind for r in results])
    mean_R_aw = np.array([r.mean_R_aw_kN for r in results])
    mean_R_wind = np.array([r.mean_R_wind_kN for r in results])
    mean_F_flett = np.array([r.mean_F_flettner_kN for r in results])

    print("\n" + "=" * 78)
    print("ANNUAL SUMMARY")
    print("=" * 78)

    print(f"\n  Voyages simulated: {len(results)} ({voy_label})")
    print(f"  Transit speed: {speed_kn:.0f} kn")
    print(f"  Transit time per voyage: {transit_h:.1f} h ({transit_h / 24:.1f} days)")
    print(f"  Idle / port time: {idle_pct:.0f}%")
    print(f"  Estimated {voy_label} voyages per year: {voyages_per_year:.0f}")

    # Factory feasibility statistics
    total_hours_all = sum(r.n_hours_total for r in results)
    total_both_feasible = sum(r.n_hours_both_feasible for r in results)
    total_factory_infeasible = sum(r.n_hours_factory_infeasible for r in results)
    pct_infeasible = 100.0 * total_factory_infeasible / total_hours_all if total_hours_all else 0

    print(f"\n  Factory combinator feasibility:")
    print(f"    Total evaluation hours: {total_hours_all}")
    print(f"    Both feasible:          {total_both_feasible} "
          f"({100 * total_both_feasible / total_hours_all:.0f}%)")
    print(f"    Factory infeasible:     {total_factory_infeasible} "
          f"({pct_infeasible:.0f}%)")
    print(f"    (Factory infeasible = engine over-power at factory combinator's")
    print(f"     fixed RPM schedule; optimiser adapts pitch/RPM to stay feasible)")

    n_voyages_with_infeasible = sum(1 for r in results if r.n_hours_factory_infeasible > 0)
    print(f"    Voyages with >= 1 infeasible hour: {n_voyages_with_infeasible} "
          f"({100 * n_voyages_with_infeasible / len(results):.0f}%)")

    # --- Per-voyage statistics ---
    print(f"\n  PER-VOYAGE AVERAGES (across {len(results)} simulated voyages):")
    print(f"\n  {'Metric':<35s}  {'Mean':>8s}  {'Median':>8s}  "
          f"{'P10':>8s}  {'P90':>8s}  {'Min':>8s}  {'Max':>8s}")
    print(f"  {'-' * 35}  {'-' * 8}  {'-' * 8}  "
          f"{'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")

    def _row(label, arr, fmt=".1f"):
        p10, p50, p90 = np.percentile(arr, [10, 50, 90])
        print(f"  {label:<35s}  {np.mean(arr):>8{fmt}}  {p50:>8{fmt}}  "
              f"{p10:>8{fmt}}  {p90:>8{fmt}}  {np.min(arr):>8{fmt}}  "
              f"{np.max(arr):>8{fmt}}")

    _row("Saving [%] (fac vs opt+Fl)", savings_pct)
    _row("Saving [kg/voy] (fac vs opt+Fl)", savings_kg, ".0f")
    _row("  of which pitch/RPM [kg]", savings_pitch_rpm_kg, ".0f")
    _row("  of which Flettner [kg]", savings_flettner_kg, ".0f")
    _row("Factory no-Fl fuel [kg/voy]", fuel_factory_nf, ".0f")
    _row("Factory+Fl fuel [kg/voy]", fuel_factory, ".0f")
    _row("Opt no-Flettner [kg/voy]", fuel_opt_noflettner, ".0f")
    _row("Opt + Flettner [kg/voy]", fuel_optimised, ".0f")
    _row("Mean Hs [m]", mean_hs)
    _row("Mean wind speed [m/s]", mean_wind)
    _row("Mean added resistance [kN]", mean_R_aw)
    _row("Mean wind resistance [kN]", mean_R_wind)
    _row("Mean Flettner thrust [kN]", mean_F_flett)

    # Compute split percentages relative to factory-no-Flettner baseline
    pct_pitch_rpm = np.where(fuel_factory_nf > 0,
                             savings_pitch_rpm_kg / fuel_factory_nf * 100.0, 0.0)
    pct_flettner = np.where(fuel_factory_nf > 0,
                            savings_flettner_kg / fuel_factory_nf * 100.0, 0.0)
    _row("  Pitch/RPM saving [%]", pct_pitch_rpm)
    _row("  Flettner saving [%]", pct_flettner)

    # --- Annualized figures ---
    # Per-voyage means (on feasible-hours basis)
    mean_fac_nf_kg = np.mean(fuel_factory_nf)
    mean_fac_fl_kg = np.mean(fuel_factory)
    mean_opt_nf_kg = np.mean(fuel_opt_noflettner)
    mean_opt_fl_kg = np.mean(fuel_optimised)
    mean_sav_pr_kg = np.mean(savings_pitch_rpm_kg)
    mean_sav_fl_kg = np.mean(savings_flettner_kg)

    ann_fac_nf = mean_fac_nf_kg * voyages_per_year / 1000.0   # tonnes
    ann_fac_fl = mean_fac_fl_kg * voyages_per_year / 1000.0
    ann_opt_nf = mean_opt_nf_kg * voyages_per_year / 1000.0
    ann_opt_fl = mean_opt_fl_kg * voyages_per_year / 1000.0
    ann_sav_pr = mean_sav_pr_kg * voyages_per_year / 1000.0
    ann_sav_fl = mean_sav_fl_kg * voyages_per_year / 1000.0
    ann_sav_total = ann_sav_pr + ann_sav_fl

    print(f"\n  ANNUALIZED FUEL ({idle_pct:.0f}% idle -> {voyages_per_year:.0f} {voy_label} voyages/year):")
    print(f"    Factory no-Flettner:     {ann_fac_nf:7.1f} tonnes/year  <- baseline")
    print(f"    Factory + Flettner:      {ann_fac_fl:7.1f} tonnes/year")
    print(f"    Opt no-Flettner:         {ann_opt_nf:7.1f} tonnes/year")
    print(f"    Opt + Flettner:          {ann_opt_fl:7.1f} tonnes/year")
    if ann_fac_nf > 0:
        print(f"\n    Saving (fac_NF - opt_FL):  {ann_sav_total:7.1f} tonnes/year "
              f"({100 * ann_sav_total / ann_fac_nf:.1f}%)")
        print(f"      Pitch/RPM (fac_NF-opt_NF): {ann_sav_pr:7.1f} tonnes/year "
              f"({100 * ann_sav_pr / ann_fac_nf:.1f}%)")
        print(f"      Flettner (opt_NF-opt_FL):  {ann_sav_fl:7.1f} tonnes/year "
              f"({100 * ann_sav_fl / ann_fac_nf:.1f}%)")
        print(f"      Check: P/R + Fl = {ann_sav_total:.1f} tonnes")

    if fuel_price_eur_per_t > 0:
        fp = fuel_price_eur_per_t
        print(f"\n  FUEL COST ESTIMATES (MGO @ \u20ac{fp:.0f}/tonne):")
        print(f"    Factory no-Flettner:     \u20ac{ann_fac_nf * fp:>10,.0f}/year")
        print(f"    Factory + Flettner:      \u20ac{ann_fac_fl * fp:>10,.0f}/year")
        print(f"    Opt no-Flettner:         \u20ac{ann_opt_nf * fp:>10,.0f}/year")
        print(f"    Opt + Flettner:          \u20ac{ann_opt_fl * fp:>10,.0f}/year")
        if ann_fac_nf > 0:
            cost_saving = ann_sav_total * fp
            cost_pr = ann_sav_pr * fp
            cost_fl = ann_sav_fl * fp
            print(f"\n    Annual saving:       \u20ac{cost_saving:>10,.0f}/year "
                  f"({100 * ann_sav_total / ann_fac_nf:.1f}%)")
            print(f"      Pitch/RPM:        \u20ac{cost_pr:>10,.0f}/year")
            print(f"      Flettner:         \u20ac{cost_fl:>10,.0f}/year")

    # --- Seasonal breakdown ---
    print(f"\n  SEASONAL BREAKDOWN (per-voyage averages):")
    print(f"\n  {'Season':<12s}  {'Voyages':>7s}  {'Fac_NF':>8s}  {'Opt+Fl':>8s}  "
          f"{'Total %':>8s}  {'P/R %':>6s}  {'Flet %':>7s}  "
          f"{'Mean Hs':>8s}  {'Mean wind':>9s}  {'R_aw':>6s}  {'R_wind':>6s}  "
          f"{'%h infeas':>9s}")
    print(f"  {'-' * 12}  {'-' * 7}  {'-' * 8}  {'-' * 8}  "
          f"{'-' * 8}  {'-' * 6}  {'-' * 7}  "
          f"{'-' * 8}  {'-' * 9}  {'-' * 6}  {'-' * 6}  {'-' * 9}")

    quarters = {"Q1 (Jan-Mar)": (1, 3), "Q2 (Apr-Jun)": (4, 6),
                "Q3 (Jul-Sep)": (7, 9), "Q4 (Oct-Dec)": (10, 12)}
    for name, (m1, m2) in quarters.items():
        q_results = [r for r in results if m1 <= r.departure.month <= m2]
        if q_results:
            q_fac_nf = np.mean([r.total_fuel_factory_no_flettner_kg for r in q_results])
            q_opt_fl = np.mean([r.total_fuel_optimised_kg for r in q_results])
            q_save = np.mean([r.saving_pct for r in q_results])
            q_fuel_fac_nf = np.array([r.total_fuel_factory_no_flettner_kg for r in q_results])
            q_sav_pr = np.array([r.saving_pitch_rpm_kg for r in q_results])
            q_sav_fl = np.array([r.saving_flettner_kg for r in q_results])
            q_pct_pr = np.mean(q_sav_pr / q_fuel_fac_nf * 100.0) if np.all(q_fuel_fac_nf > 0) else 0.0
            q_pct_fl = np.mean(q_sav_fl / q_fuel_fac_nf * 100.0) if np.all(q_fuel_fac_nf > 0) else 0.0
            q_hs = np.mean([r.mean_hs for r in q_results])
            q_wind = np.mean([r.mean_wind for r in q_results])
            q_raw = np.mean([r.mean_R_aw_kN for r in q_results])
            q_rwind = np.mean([r.mean_R_wind_kN for r in q_results])
            q_total_h = sum(r.n_hours_total for r in q_results)
            q_infeas = sum(r.n_hours_factory_infeasible for r in q_results)
            q_pct_inf = 100 * q_infeas / q_total_h if q_total_h else 0
            print(f"  {name:<12s}  {len(q_results):>7d}  "
                  f"{q_fac_nf:>7.0f}kg  {q_opt_fl:>7.0f}kg  "
                  f"{q_save:>+7.1f}%  "
                  f"{q_pct_pr:>+5.1f}%  {q_pct_fl:>+6.1f}%  "
                  f"{q_hs:>8.1f}  {q_wind:>9.1f}  {q_raw:>5.1f}  {q_rwind:>5.1f}  "
                  f"{q_pct_inf:>8.0f}%")


def plot_results(results: list[VoyageResult]):
    """Generate summary plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    dates = [r.departure for r in results]
    savings_pct = np.array([r.saving_pct for r in results])
    mean_hs = np.array([r.mean_hs for r in results])
    mean_wind = np.array([r.mean_wind for r in results])
    mean_R_aw = np.array([r.mean_R_aw_kN for r in results])
    mean_F_flett = np.array([r.mean_F_flettner_kN for r in results])

    # Split saving percentages (relative to factory-no-Flettner baseline)
    fuel_factory_nf = np.array([r.total_fuel_factory_no_flettner_kg for r in results])
    fuel_factory = np.array([r.total_fuel_factory_kg for r in results])
    sav_pr_pct = np.where(fuel_factory_nf > 0,
                          np.array([r.saving_pitch_rpm_kg for r in results])
                          / fuel_factory_nf * 100.0, 0.0)
    sav_fl_pct = np.where(fuel_factory_nf > 0,
                          np.array([r.saving_flettner_kg for r in results])
                          / fuel_factory_nf * 100.0, 0.0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # 1. Split saving: stacked area (pitch/RPM + Flettner)
    # All percentages relative to factory-no-Flettner baseline for
    # consistent additive decomposition.
    sav_total_pct = sav_pr_pct + sav_fl_pct
    ax = axes[0]
    ax.fill_between(dates, 0, sav_pr_pct, alpha=0.6, color="steelblue",
                    label=f"Pitch/RPM ({np.mean(sav_pr_pct):.1f}% mean)")
    ax.fill_between(dates, sav_pr_pct, sav_total_pct,
                    alpha=0.6, color="coral",
                    label=f"Flettner ({np.mean(sav_fl_pct):.1f}% mean)")
    ax.plot(dates, sav_total_pct, "k-", linewidth=0.5, alpha=0.6,
            label=f"Total ({np.mean(sav_total_pct):.1f}% mean)")
    ax.axhline(np.mean(sav_total_pct), color="k", linestyle="--",
               linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Fuel saving [% of factory NF]")
    ax.set_title("Optimiser vs Factory Combinator: Daily Fuel Saving (Split)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Significant wave height
    ax = axes[1]
    ax.plot(dates, mean_hs, "g-", linewidth=0.7, alpha=0.8)
    ax.set_ylabel("Mean Hs [m]")
    ax.set_title("Voyage-Mean Significant Wave Height")
    ax.grid(True, alpha=0.3)

    # 3. Added resistance and Flettner thrust
    ax = axes[2]
    ax.plot(dates, mean_R_aw, "r-", linewidth=0.7, alpha=0.8, label="Added resistance")
    ax.plot(dates, mean_F_flett, "b-", linewidth=0.7, alpha=0.8, label="Flettner thrust")
    ax.set_ylabel("[kN]")
    ax.set_title("Voyage-Mean Added Resistance and Flettner Thrust")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    out_path = Path(__file__).parent / "voyage_comparison_results.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")

    # Scatter: Saving vs Hs (separate figure)
    fig_sc, ax_sc = plt.subplots(figsize=(10, 6))
    sc = ax_sc.scatter(mean_hs, sav_total_pct, s=15, alpha=0.5,
                       c=mean_wind, cmap="viridis")
    ax_sc.set_xlabel("Mean Hs [m]")
    ax_sc.set_ylabel("Fuel saving [% of factory NF]")
    ax_sc.set_title("Fuel Saving vs Sea State (baseline = factory no-Flettner, colour = wind speed)")
    ax_sc.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax_sc, label="Wind speed [m/s]")
    plt.tight_layout()
    out_path_sc = Path(__file__).parent / "voyage_comparison_scatter.png"
    plt.savefig(out_path_sc, dpi=150)
    print(f"Scatter saved: {out_path_sc}")

    # Histogram: separate subplots for each saving component
    fig2, (ax_h1, ax_h2, ax_h3) = plt.subplots(1, 3, figsize=(16, 5))

    # Total saving
    ax_h1.hist(savings_pct, bins=30, edgecolor="black", alpha=0.7,
               color="grey")
    ax_h1.axvline(np.mean(savings_pct), color="k", linestyle="--",
                  label=f"Mean: {np.mean(savings_pct):.1f}%")
    ax_h1.set_xlabel("Fuel saving [%]")
    ax_h1.set_ylabel("Number of voyages")
    ax_h1.set_title("Total (Factory+Fl vs Opt+Fl)")
    ax_h1.legend(fontsize=9)
    ax_h1.grid(True, alpha=0.3)

    # Pitch/RPM saving
    ax_h2.hist(sav_pr_pct, bins=30, edgecolor="black", alpha=0.7,
               color="steelblue")
    ax_h2.axvline(np.mean(sav_pr_pct), color="k", linestyle="--",
                  label=f"Mean: {np.mean(sav_pr_pct):.1f}%")
    ax_h2.set_xlabel("Fuel saving [%]")
    ax_h2.set_title("Pitch/RPM (Fac_NF vs Opt_NF)")
    ax_h2.legend(fontsize=9)
    ax_h2.grid(True, alpha=0.3)

    # Flettner saving
    ax_h3.hist(sav_fl_pct, bins=30, edgecolor="black", alpha=0.7,
               color="coral")
    ax_h3.axvline(np.mean(sav_fl_pct), color="k", linestyle="--",
                  label=f"Mean: {np.mean(sav_fl_pct):.1f}%")
    ax_h3.set_xlabel("Fuel saving [%]")
    ax_h3.set_title("Flettner (Opt_NF vs Opt_FL)")
    ax_h3.legend(fontsize=9)
    ax_h3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path2 = Path(__file__).parent / "voyage_comparison_histogram.png"
    plt.savefig(out_path2, dpi=150)
    print(f"Histogram saved: {out_path2}")

    # Route map
    plot_route(ROUTE_ROTTERDAM_GOTHENBURG)


def plot_comparison(results_std: list[VoyageResult],
                    results_sg: list[VoyageResult],
                    speed_kn: float,
                    idle_pct: float = 15.0):
    """Generate comparison plots: standard vs shaft-generator mode."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib not available; skipping comparison plots.")
        return

    out_dir = Path(__file__).parent

    # --- Helper: extract arrays from result list ---
    def _arrays(results):
        d = {}
        d["dates"] = [r.departure for r in results]
        d["fuel_fac_nf"] = np.array([r.total_fuel_factory_no_flettner_kg
                                      for r in results])
        d["fuel_fac_fl"] = np.array([r.total_fuel_factory_kg for r in results])
        d["fuel_opt_nf"] = np.array([r.total_fuel_opt_no_flettner_kg
                                      for r in results])
        d["fuel_opt_fl"] = np.array([r.total_fuel_optimised_kg
                                      for r in results])
        d["sav_pr_kg"] = np.array([r.saving_pitch_rpm_kg for r in results])
        d["sav_fl_kg"] = np.array([r.saving_flettner_kg for r in results])
        d["sav_pct"] = np.array([r.saving_pct for r in results])
        d["mean_hs"] = np.array([r.mean_hs for r in results])
        d["mean_wind"] = np.array([r.mean_wind for r in results])
        d["mean_R_aw"] = np.array([r.mean_R_aw_kN for r in results])
        d["mean_F_fl"] = np.array([r.mean_F_flettner_kN for r in results])
        # Percentage savings (relative to factory no-Fl baseline)
        d["pct_pr"] = np.where(d["fuel_fac_nf"] > 0,
                               d["sav_pr_kg"] / d["fuel_fac_nf"] * 100.0, 0.0)
        d["pct_fl"] = np.where(d["fuel_fac_nf"] > 0,
                               d["sav_fl_kg"] / d["fuel_fac_nf"] * 100.0, 0.0)
        return d

    std = _arrays(results_std)
    sg = _arrays(results_sg)
    transit_h = results_std[0].total_hours
    voyages_yr = 365.25 * 24.0 * (1.0 - idle_pct / 100.0) / transit_h

    # Consistent colours
    C_PR_STD = "#3574a3"     # steel blue
    C_FL_STD = "#e8774a"     # coral
    C_PR_SG = "#1a4a70"      # dark blue
    C_FL_SG = "#b84520"      # dark coral

    # ==================================================================
    # FIGURE 1: Daily savings time series — side by side
    # ==================================================================
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # Panel 1: Standard mode stacked area
    ax1.fill_between(std["dates"], 0, std["pct_pr"], alpha=0.55,
                     color=C_PR_STD,
                     label=f"Pitch/RPM ({np.mean(std['pct_pr']):.1f}%)")
    ax1.fill_between(std["dates"], std["pct_pr"],
                     std["pct_pr"] + std["pct_fl"],
                     alpha=0.55, color=C_FL_STD,
                     label=f"Flettner ({np.mean(std['pct_fl']):.1f}%)")
    ax1.axhline(np.mean(std["sav_pct"]), color="k", ls="--", lw=0.8,
                alpha=0.5)
    ax1.set_ylabel("Fuel saving [%]")
    ax1.set_title("Standard Mode — Daily Saving (Pitch/RPM + Flettner)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Panel 2: SG mode stacked area
    ax2.fill_between(sg["dates"], 0, sg["pct_pr"], alpha=0.55,
                     color=C_PR_SG,
                     label=f"Pitch/RPM ({np.mean(sg['pct_pr']):.1f}%)")
    ax2.fill_between(sg["dates"], sg["pct_pr"],
                     sg["pct_pr"] + sg["pct_fl"],
                     alpha=0.55, color=C_FL_SG,
                     label=f"Flettner ({np.mean(sg['pct_fl']):.1f}%)")
    ax2.axhline(np.mean(sg["sav_pct"]), color="k", ls="--", lw=0.8,
                alpha=0.5)
    ax2.set_ylabel("Fuel saving [%]")
    ax2.set_title("Shaft Generator Mode — Daily Saving (Pitch/RPM + Flettner)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # Match y-axis range across both panels
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(0, ymax)
    ax2.set_ylim(0, ymax)

    # Panel 3: Sea state (shared context)
    ax3.plot(std["dates"], std["mean_hs"], color="#2e8b57", lw=0.8,
             alpha=0.8, label="Hs")
    ax3_tw = ax3.twinx()
    ax3_tw.plot(std["dates"], std["mean_wind"], color="#666", lw=0.6,
                alpha=0.6, label="Wind")
    ax3.set_ylabel("Mean Hs [m]", color="#2e8b57")
    ax3_tw.set_ylabel("Mean wind [m/s]", color="#666")
    ax3.set_title("Weather Conditions")
    ax3.grid(True, alpha=0.3)

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    p = out_dir / "comparison_timeseries.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"\nPlot saved: {p}")

    # ==================================================================
    # FIGURE 2: Annualized fuel & savings bar chart
    # ==================================================================
    fig, (ax_fuel, ax_sav) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: annual fuel consumption (4 bars x 2 modes)
    labels = ["Fac NF", "Fac+Fl", "Opt NF", "Opt+Fl"]
    std_vals = np.array([np.mean(std["fuel_fac_nf"]),
                         np.mean(std["fuel_fac_fl"]),
                         np.mean(std["fuel_opt_nf"]),
                         np.mean(std["fuel_opt_fl"])]) * voyages_yr / 1000.0
    sg_vals = np.array([np.mean(sg["fuel_fac_nf"]),
                        np.mean(sg["fuel_fac_fl"]),
                        np.mean(sg["fuel_opt_nf"]),
                        np.mean(sg["fuel_opt_fl"])]) * voyages_yr / 1000.0

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax_fuel.bar(x - w / 2, std_vals, w, label="Standard",
                        color=C_PR_STD, alpha=0.8)
    bars2 = ax_fuel.bar(x + w / 2, sg_vals, w, label="SG mode",
                        color=C_PR_SG, alpha=0.8)
    for bar, val in zip(bars1, std_vals):
        ax_fuel.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, sg_vals):
        ax_fuel.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    ax_fuel.set_xticks(x)
    ax_fuel.set_xticklabels(labels)
    ax_fuel.set_ylabel("Fuel [tonnes/year]")
    ax_fuel.set_title(f"Annualized Fuel Consumption\n"
                      f"({idle_pct:.0f}% idle, {voyages_yr:.0f} voyages/yr)")
    ax_fuel.legend()
    ax_fuel.grid(True, alpha=0.3, axis="y")

    # Right: savings breakdown (stacked bars — P/R + Flettner)
    std_pr = np.mean(std["sav_pr_kg"]) * voyages_yr / 1000.0
    std_fl = np.mean(std["sav_fl_kg"]) * voyages_yr / 1000.0
    sg_pr = np.mean(sg["sav_pr_kg"]) * voyages_yr / 1000.0
    sg_fl = np.mean(sg["sav_fl_kg"]) * voyages_yr / 1000.0

    std_fac_nf_ann = np.mean(std["fuel_fac_nf"]) * voyages_yr / 1000.0
    sg_fac_nf_ann = np.mean(sg["fuel_fac_nf"]) * voyages_yr / 1000.0

    x2 = np.arange(2)
    pr_vals = [std_pr, sg_pr]
    fl_vals = [std_fl, sg_fl]
    bars_pr = ax_sav.bar(x2, pr_vals, 0.5, label="Pitch/RPM",
                         color=[C_PR_STD, C_PR_SG], alpha=0.85)
    bars_fl = ax_sav.bar(x2, fl_vals, 0.5, bottom=pr_vals,
                         label="Flettner",
                         color=[C_FL_STD, C_FL_SG], alpha=0.85)
    # Annotate totals and percentages
    for i, (pr, fl, baseline) in enumerate(
            zip(pr_vals, fl_vals, [std_fac_nf_ann, sg_fac_nf_ann])):
        total = pr + fl
        pct = 100 * total / baseline if baseline > 0 else 0
        ax_sav.text(i, total + 2, f"{total:.0f} t ({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        # Sub-labels
        ax_sav.text(i, pr / 2, f"P/R\n{pr:.0f} t",
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold")
        ax_sav.text(i, pr + fl / 2, f"Fl\n{fl:.0f} t",
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold")

    ax_sav.set_xticks(x2)
    ax_sav.set_xticklabels(["Standard", "SG mode"])
    ax_sav.set_ylabel("Fuel saving [tonnes/year]")
    ax_sav.set_title("Annual Savings Breakdown\n(vs Factory no-Flettner baseline)")
    ax_sav.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p = out_dir / "comparison_annual.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Plot saved: {p}")

    # ==================================================================
    # FIGURE 3: Seasonal per-voyage breakdown (grouped bars)
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    quarters = {"Q1\nJan-Mar": (1, 3), "Q2\nApr-Jun": (4, 6),
                "Q3\nJul-Sep": (7, 9), "Q4\nOct-Dec": (10, 12)}
    q_names = list(quarters.keys())

    for ax_idx, (mode_label, res, c_pr, c_fl) in enumerate([
        ("Standard", results_std, C_PR_STD, C_FL_STD),
        ("SG Mode", results_sg, C_PR_SG, C_FL_SG),
    ]):
        ax = axes[ax_idx]
        q_pr_pct = []
        q_fl_pct = []
        q_fac_kg = []
        q_opt_kg = []
        for _, (m1, m2) in quarters.items():
            qr = [r for r in res if m1 <= r.departure.month <= m2]
            fac_nf = np.array([r.total_fuel_factory_no_flettner_kg for r in qr])
            spr = np.array([r.saving_pitch_rpm_kg for r in qr])
            sfl = np.array([r.saving_flettner_kg for r in qr])
            q_pr_pct.append(np.mean(spr / fac_nf * 100) if np.all(fac_nf > 0) else 0)
            q_fl_pct.append(np.mean(sfl / fac_nf * 100) if np.all(fac_nf > 0) else 0)
            q_fac_kg.append(np.mean(fac_nf))
            q_opt_kg.append(np.mean([r.total_fuel_optimised_kg for r in qr]))

        xq = np.arange(len(q_names))
        ax.bar(xq, q_pr_pct, 0.6, label="Pitch/RPM", color=c_pr, alpha=0.85)
        ax.bar(xq, q_fl_pct, 0.6, bottom=q_pr_pct, label="Flettner",
               color=c_fl, alpha=0.85)
        for i in range(len(q_names)):
            total = q_pr_pct[i] + q_fl_pct[i]
            ax.text(i, total + 0.15, f"{total:.1f}%", ha="center",
                    va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(xq)
        ax.set_xticklabels(q_names)
        ax.set_ylabel("Fuel saving [%]")
        ax.set_title(f"{mode_label} — Seasonal Savings")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    # Match y-axis
    ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(0, ymax)
    axes[1].set_ylim(0, ymax)

    plt.tight_layout()
    p = out_dir / "comparison_seasonal.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Plot saved: {p}")

    # ==================================================================
    # FIGURE 4: Scatter — saving vs Hs, both modes
    # ==================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sc1 = ax1.scatter(std["mean_hs"], std["sav_pct"], s=12, alpha=0.5,
                      c=std["mean_wind"], cmap="viridis", vmin=2, vmax=17)
    ax1.set_xlabel("Mean Hs [m]")
    ax1.set_ylabel("Total saving [%]")
    ax1.set_title("Standard Mode")
    ax1.grid(True, alpha=0.3)

    sc2 = ax2.scatter(sg["mean_hs"], sg["sav_pct"], s=12, alpha=0.5,
                      c=sg["mean_wind"], cmap="viridis", vmin=2, vmax=17)
    ax2.set_xlabel("Mean Hs [m]")
    ax2.set_title("SG Mode")
    ax2.grid(True, alpha=0.3)

    plt.colorbar(sc2, ax=[ax1, ax2], label="Wind speed [m/s]",
                 shrink=0.8, pad=0.02)
    fig.subplots_adjust(left=0.07, right=0.88, wspace=0.08)
    p = out_dir / "comparison_scatter.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Plot saved: {p}")

    # ==================================================================
    # FIGURE 5: Overlaid histograms — saving distributions
    # ==================================================================
    fig, (ax_h1, ax_h2, ax_h3) = plt.subplots(1, 3, figsize=(16, 5))

    bins_total = np.linspace(0, max(np.max(std["sav_pct"]),
                                     np.max(sg["sav_pct"])) * 1.05, 35)
    ax_h1.hist(std["sav_pct"], bins=bins_total, alpha=0.6,
               color=C_PR_STD, edgecolor="white", lw=0.5,
               label=f"Std ({np.mean(std['sav_pct']):.1f}%)")
    ax_h1.hist(sg["sav_pct"], bins=bins_total, alpha=0.6,
               color=C_PR_SG, edgecolor="white", lw=0.5,
               label=f"SG ({np.mean(sg['sav_pct']):.1f}%)")
    ax_h1.axvline(np.mean(std["sav_pct"]), color=C_PR_STD, ls="--", lw=1.2)
    ax_h1.axvline(np.mean(sg["sav_pct"]), color=C_PR_SG, ls="--", lw=1.2)
    ax_h1.set_xlabel("Fuel saving [%]")
    ax_h1.set_ylabel("Number of voyages")
    ax_h1.set_title("Total Saving (Fac+Fl vs Opt+Fl)")
    ax_h1.legend(fontsize=9)
    ax_h1.grid(True, alpha=0.3)

    bins_pr = np.linspace(0, max(np.max(std["pct_pr"]),
                                  np.max(sg["pct_pr"])) * 1.05, 35)
    ax_h2.hist(std["pct_pr"], bins=bins_pr, alpha=0.6,
               color=C_PR_STD, edgecolor="white", lw=0.5,
               label=f"Std ({np.mean(std['pct_pr']):.1f}%)")
    ax_h2.hist(sg["pct_pr"], bins=bins_pr, alpha=0.6,
               color=C_PR_SG, edgecolor="white", lw=0.5,
               label=f"SG ({np.mean(sg['pct_pr']):.1f}%)")
    ax_h2.axvline(np.mean(std["pct_pr"]), color=C_PR_STD, ls="--", lw=1.2)
    ax_h2.axvline(np.mean(sg["pct_pr"]), color=C_PR_SG, ls="--", lw=1.2)
    ax_h2.set_xlabel("Pitch/RPM saving [%]")
    ax_h2.set_title("Pitch/RPM Saving (Fac_NF vs Opt_NF)")
    ax_h2.legend(fontsize=9)
    ax_h2.grid(True, alpha=0.3)

    bins_fl = np.linspace(0, max(np.max(std["pct_fl"]),
                                  np.max(sg["pct_fl"])) * 1.05, 35)
    ax_h3.hist(std["pct_fl"], bins=bins_fl, alpha=0.6,
               color=C_FL_STD, edgecolor="white", lw=0.5,
               label=f"Std ({np.mean(std['pct_fl']):.1f}%)")
    ax_h3.hist(sg["pct_fl"], bins=bins_fl, alpha=0.6,
               color=C_FL_SG, edgecolor="white", lw=0.5,
               label=f"SG ({np.mean(sg['pct_fl']):.1f}%)")
    ax_h3.axvline(np.mean(std["pct_fl"]), color=C_FL_STD, ls="--", lw=1.2)
    ax_h3.axvline(np.mean(sg["pct_fl"]), color=C_FL_SG, ls="--", lw=1.2)
    ax_h3.set_xlabel("Flettner saving [%]")
    ax_h3.set_title("Flettner Saving (Opt_NF vs Opt_FL)")
    ax_h3.legend(fontsize=9)
    ax_h3.grid(True, alpha=0.3)

    plt.tight_layout()
    p = out_dir / "comparison_histograms.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Plot saved: {p}")

    # Route map (shared — same route for both modes)
    plot_route(ROUTE_ROTTERDAM_GOTHENBURG)


def _load_land_polygons(bbox: tuple[float, float, float, float],
                        ) -> list[list[tuple[float, float]]]:
    """Load land polygons that intersect a bounding box.

    Searches for data sources in order of preference:
    1. Natural Earth 10m land (best detail for regional maps)
    2. Natural Earth 50m land (good compromise)
    3. GSHHG crude L1 (cartopy bundled fallback)

    Parameters
    ----------
    bbox : (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    list of polygons, each a list of (lon, lat) tuples.
    """
    import shapefile

    data_dir = Path(__file__).parent / "data"
    candidates = [
        data_dir / "ne_10m" / "ne_10m_land.shp",
        data_dir / "ne_50m" / "ne_50m_land.shp",
        Path("/usr/share/cartopy/data/shapefiles/gshhs/c/GSHHS_c_L1.shp"),
        data_dir / "gshhg" / "GSHHS_c_L1.shp",
    ]

    shp_path = None
    for c in candidates:
        if c.exists():
            shp_path = c
            break

    if shp_path is None:
        print("Warning: no coastline data found.")
        print(f"  Searched: {[str(c) for c in candidates]}")
        return []

    print(f"  Coastline data: {shp_path.name}")

    sf = shapefile.Reader(str(shp_path))
    lon_min, lat_min, lon_max, lat_max = bbox
    margin = 2.0
    polygons = []

    for shape in sf.shapes():
        sb = shape.bbox  # [min_lon, min_lat, max_lon, max_lat]
        if (sb[2] < lon_min - margin or sb[0] > lon_max + margin or
                sb[3] < lat_min - margin or sb[1] > lat_max + margin):
            continue

        parts = list(shape.parts) + [len(shape.points)]
        for k in range(len(parts) - 1):
            ring = shape.points[parts[k]:parts[k + 1]]
            # Quick filter: skip rings entirely outside the bbox
            ring_lons = [p[0] for p in ring]
            ring_lats = [p[1] for p in ring]
            if (max(ring_lons) < lon_min - margin or
                    min(ring_lons) > lon_max + margin or
                    max(ring_lats) < lat_min - margin or
                    min(ring_lats) > lat_max + margin):
                continue
            polygons.append([(p[0], p[1]) for p in ring])

    return polygons


def plot_route(waypoints: list[Waypoint]):
    """Plot the route on a map with GSHHG coastlines.

    Uses the Global Self-consistent Hierarchical High-resolution Shoreline
    Database (GSHHG) for accurate coastlines.  Falls back to a plain frame
    if the shapefiles are not available.

    The map extent is derived automatically from the waypoints, so this
    function works for any route.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection
    except ImportError:
        return

    # --- Compute map extent from waypoints (with margin) ---
    wlats = [wp.lat for wp in waypoints]
    wlons = [wp.lon for wp in waypoints]
    lat_span = max(wlats) - min(wlats)
    lon_span = max(wlons) - min(wlons)
    margin_lat = max(1.5, lat_span * 0.30)
    margin_lon = max(2.0, lon_span * 0.30)
    lon_min = min(wlons) - margin_lon
    lon_max = max(wlons) + margin_lon
    lat_min = min(wlats) - margin_lat
    lat_max = max(wlats) + margin_lat

    bbox = (lon_min, lat_min, lon_max, lat_max)

    # --- Load coastlines ---
    polygons = _load_land_polygons(bbox)

    # --- Set up figure ---
    mid_lat = np.mean(wlats)
    aspect = 1.0 / np.cos(np.radians(mid_lat))
    fig_width = 10
    fig_height = fig_width * (lat_max - lat_min) / (lon_max - lon_min) * aspect
    fig_height = max(6, min(14, fig_height))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Sea background
    ax.set_facecolor("#c8dce8")

    # --- Draw land polygons ---
    if polygons:
        patches = []
        for poly_pts in polygons:
            if len(poly_pts) >= 3:
                patches.append(MplPolygon(poly_pts, closed=True))
        if patches:
            pc = PatchCollection(patches,
                                 facecolor="#e8e4d8",
                                 edgecolor="#8a8578",
                                 linewidth=0.6,
                                 zorder=2)
            ax.add_collection(pc)
    else:
        # Fallback: just frame with no land
        ax.text(0.5, 0.5, "(GSHHG coastline data not available)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="0.5")

    # --- Draw route ---
    ax.plot(wlons, wlats, "o-", color="#c0392b", linewidth=2.5,
            markersize=6, markeredgecolor="white", markeredgewidth=0.8,
            zorder=5, label="Route")

    # Label departure and arrival
    ax.annotate(waypoints[0].name, (wlons[0], wlats[0]),
                textcoords="offset points", xytext=(-10, -15),
                fontsize=9, fontweight="bold", color="#c0392b",
                ha="right", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.8, edgecolor="none"))
    ax.annotate(waypoints[-1].name, (wlons[-1], wlats[-1]),
                textcoords="offset points", xytext=(-10, 10),
                fontsize=9, fontweight="bold", color="#c0392b",
                ha="right", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.8, edgecolor="none"))

    # --- Map formatting ---
    ax.set_aspect(aspect)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude [deg E]")
    ax.set_ylabel("Latitude [deg N]")
    ax.set_title(f"{waypoints[0].name} \u2014 {waypoints[-1].name}")
    ax.grid(True, alpha=0.25, linestyle="--", color="0.5")

    # Distance annotation
    total_nm = sum(
        _haversine_nm(waypoints[i].lat, waypoints[i].lon,
                      waypoints[i + 1].lat, waypoints[i + 1].lon)
        for i in range(len(waypoints) - 1)
    )
    ax.text(0.02, 0.02, f"Total distance: {total_nm:.0f} nm",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    out_path = Path(__file__).parent / "voyage_comparison_route.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Route map saved: {out_path}")


# ============================================================
# Speed sensitivity sweep
# ============================================================

def _summarise_for_speed(
    results: list[VoyageResult],
    speed_kn: float,
    idle_pct: float,
) -> SpeedSweepResult:
    """Compute aggregated statistics for one speed."""
    transit_h = results[0].total_hours
    sailing_hours_year = 365.25 * 24.0 * (1.0 - idle_pct / 100.0)
    vpyr = sailing_hours_year / transit_h

    fac_nf = np.array([r.total_fuel_factory_no_flettner_kg for r in results])
    fac_fl = np.array([r.total_fuel_factory_kg for r in results])
    opt_nf = np.array([r.total_fuel_opt_no_flettner_kg for r in results])
    opt_fl = np.array([r.total_fuel_optimised_kg for r in results])
    sav_pr = np.array([r.saving_pitch_rpm_kg for r in results])
    sav_fl = np.array([r.saving_flettner_kg for r in results])
    sav_tot = sav_pr + sav_fl
    sav_pct = np.array([r.saving_pct for r in results])

    m_fac_nf = float(np.mean(fac_nf))
    m_fac_fl = float(np.mean(fac_fl))
    m_opt_nf = float(np.mean(opt_nf))
    m_opt_fl = float(np.mean(opt_fl))
    m_sav_pr = float(np.mean(sav_pr))
    m_sav_fl = float(np.mean(sav_fl))
    m_sav_tot = float(np.mean(sav_tot))
    m_sav_pct = float(np.mean(sav_pct))

    pct_pr = 100.0 * m_sav_pr / m_fac_nf if m_fac_nf > 0 else 0.0
    pct_fl = 100.0 * m_sav_fl / m_fac_nf if m_fac_nf > 0 else 0.0

    total_hrs = sum(r.n_hours_total for r in results)
    total_inf = sum(r.n_hours_factory_infeasible for r in results)

    return SpeedSweepResult(
        speed_kn=speed_kn,
        transit_hours=transit_h,
        voyages_per_year=vpyr,
        n_voyages=len(results),
        mean_fuel_factory_nf_kg=m_fac_nf,
        mean_fuel_factory_fl_kg=m_fac_fl,
        mean_fuel_opt_nf_kg=m_opt_nf,
        mean_fuel_opt_fl_kg=m_opt_fl,
        mean_saving_pitch_rpm_kg=m_sav_pr,
        mean_saving_flettner_kg=m_sav_fl,
        mean_saving_total_kg=m_sav_tot,
        mean_saving_pct=m_sav_pct,
        pct_pitch_rpm=pct_pr,
        pct_flettner=pct_fl,
        ann_fuel_factory_nf_t=m_fac_nf * vpyr / 1000.0,
        ann_fuel_opt_fl_t=m_opt_fl * vpyr / 1000.0,
        ann_saving_pitch_rpm_t=m_sav_pr * vpyr / 1000.0,
        ann_saving_flettner_t=m_sav_fl * vpyr / 1000.0,
        ann_saving_total_t=m_sav_tot * vpyr / 1000.0,
        mean_hs=float(np.mean([r.mean_hs for r in results])),
        mean_wind=float(np.mean([r.mean_wind for r in results])),
        mean_R_aw_kN=float(np.mean([r.mean_R_aw_kN for r in results])),
        mean_R_wind_kN=float(np.mean([r.mean_R_wind_kN for r in results])),
        mean_F_flettner_kN=float(np.mean([r.mean_F_flettner_kN for r in results])),
        pct_factory_infeasible=100.0 * total_inf / total_hrs if total_hrs else 0.0,
    )


def run_speed_sweep(
    speeds: list[float],
    year: int = 2024,
    idle_pct: float = 15.0,
    data_dir: Path | None = None,
    pdstrip_path: str = None,
    flettner_enabled: bool = True,
    verbose: bool = False,
    sg_load_kw: float = 0.0,
    sg_factory_allowance_kw: float = 0.0,
    sg_freq_min: float = 0.0,
    sg_freq_max: float = 0.0,
    round_trip: bool = True,
) -> list[SpeedSweepResult]:
    """Run annual comparisons at multiple speeds and return summary per speed.

    Parameters
    ----------
    speeds : list of float
        Transit speeds in knots to evaluate.
    idle_pct : float
        Percentage of year idle, for annualization.
    round_trip : bool
        If True, each departure is a round trip (outbound + return).

    Returns
    -------
    list of SpeedSweepResult, one per speed (same order as input).
    """
    if pdstrip_path is None:
        pdstrip_path = PDSTRIP_DAT

    sweep_results = []
    for i, spd in enumerate(speeds):
        print(f"\n{'#' * 78}")
        print(f"# SPEED SWEEP: {spd:.1f} kn  ({i + 1}/{len(speeds)})")
        print(f"{'#' * 78}\n")

        results = run_annual_comparison(
            year=year,
            speed_kn=spd,
            data_dir=data_dir,
            pdstrip_path=pdstrip_path,
            flettner_enabled=flettner_enabled,
            verbose=verbose,
            sg_load_kw=sg_load_kw,
            sg_factory_allowance_kw=sg_factory_allowance_kw,
            sg_freq_min=sg_freq_min,
            sg_freq_max=sg_freq_max,
            round_trip=round_trip,
        )

        if not results:
            print(f"  WARNING: No valid voyages at {spd:.1f} kn — skipping")
            continue

        sr = _summarise_for_speed(results, spd, idle_pct)
        sweep_results.append(sr)

        # Print brief inline summary
        print(f"\n  {spd:.0f} kn summary: {sr.n_voyages} voyages, "
              f"fac_NF={sr.mean_fuel_factory_nf_kg:.0f} kg/voy, "
              f"opt+Fl={sr.mean_fuel_opt_fl_kg:.0f} kg/voy, "
              f"saving={sr.mean_saving_pct:.1f}% "
              f"(P/R {sr.pct_pitch_rpm:.1f}% + Fl {sr.pct_flettner:.1f}%)")

    return sweep_results


def print_speed_sweep_summary(sweep: list[SpeedSweepResult],
                              idle_pct: float = 15.0,
                              fuel_price_eur_per_t: float = 0.0):
    """Print a compact comparison table across speeds."""
    if not sweep:
        print("No speed sweep results to display.")
        return

    print("\n" + "=" * 110)
    print("SPEED SENSITIVITY SUMMARY")
    print("=" * 110)
    print(f"  Idle: {idle_pct:.0f}%    Route distance: same for all speeds")

    # --- Per-voyage table ---
    print(f"\n  PER-VOYAGE AVERAGES:")
    hdr = (f"  {'Speed':>6s}  {'Transit':>8s}  {'Voy/yr':>6s}  "
           f"{'Fac_NF':>8s}  {'Opt+Fl':>8s}  "
           f"{'Total%':>7s}  {'P/R%':>6s}  {'Flet%':>6s}  "
           f"{'Sav kg':>7s}  {'P/R kg':>7s}  {'Flet kg':>8s}  "
           f"{'%infeas':>7s}")
    print(hdr)
    print(f"  {'-' * 6}  {'-' * 8}  {'-' * 6}  "
          f"{'-' * 8}  {'-' * 8}  "
          f"{'-' * 7}  {'-' * 6}  {'-' * 6}  "
          f"{'-' * 7}  {'-' * 7}  {'-' * 8}  "
          f"{'-' * 7}")
    for s in sweep:
        print(f"  {s.speed_kn:5.1f}kn  "
              f"{s.transit_hours:6.1f} h  "
              f"{s.voyages_per_year:6.0f}  "
              f"{s.mean_fuel_factory_nf_kg:7.0f}kg  "
              f"{s.mean_fuel_opt_fl_kg:7.0f}kg  "
              f"{s.mean_saving_pct:>+6.1f}%  "
              f"{s.pct_pitch_rpm:>+5.1f}%  {s.pct_flettner:>+5.1f}%  "
              f"{s.mean_saving_total_kg:>7.0f}  "
              f"{s.mean_saving_pitch_rpm_kg:>7.0f}  "
              f"{s.mean_saving_flettner_kg:>8.0f}  "
              f"{s.pct_factory_infeasible:>6.0f}%")

    # --- Annualized table ---
    print(f"\n  ANNUALIZED FUEL (tonnes/year, {idle_pct:.0f}% idle):")
    hdr2 = (f"  {'Speed':>6s}  {'Fac_NF':>9s}  {'Opt+Fl':>9s}  "
            f"{'Saving':>9s}  {'P/R':>8s}  {'Flettner':>9s}  "
            f"{'Total%':>7s}")
    print(hdr2)
    print(f"  {'-' * 6}  {'-' * 9}  {'-' * 9}  "
          f"{'-' * 9}  {'-' * 8}  {'-' * 9}  "
          f"{'-' * 7}")
    for s in sweep:
        pct_total = 100.0 * s.ann_saving_total_t / s.ann_fuel_factory_nf_t \
            if s.ann_fuel_factory_nf_t > 0 else 0.0
        print(f"  {s.speed_kn:5.1f}kn  "
              f"{s.ann_fuel_factory_nf_t:8.1f} t  "
              f"{s.ann_fuel_opt_fl_t:8.1f} t  "
              f"{s.ann_saving_total_t:8.1f} t  "
              f"{s.ann_saving_pitch_rpm_t:7.1f} t  "
              f"{s.ann_saving_flettner_t:8.1f} t  "
              f"{pct_total:>+6.1f}%")

    # --- Weather conditions ---
    print(f"\n  MEAN WEATHER CONDITIONS (voyage average across year):")
    hdr3 = (f"  {'Speed':>6s}  {'Hs [m]':>7s}  {'Wind [m/s]':>10s}  "
            f"{'R_aw [kN]':>9s}  {'R_wind [kN]':>11s}  {'F_flett [kN]':>12s}")
    print(hdr3)
    print(f"  {'-' * 6}  {'-' * 7}  {'-' * 10}  {'-' * 9}  {'-' * 11}  {'-' * 12}")
    for s in sweep:
        print(f"  {s.speed_kn:5.1f}kn  "
              f"{s.mean_hs:>7.2f}  "
              f"{s.mean_wind:>10.1f}  "
              f"{s.mean_R_aw_kN:>9.1f}  "
              f"{s.mean_R_wind_kN:>11.1f}  "
              f"{s.mean_F_flettner_kN:>12.1f}")

    if fuel_price_eur_per_t > 0:
        fp = fuel_price_eur_per_t
        print(f"\n  FUEL COST ESTIMATES (@ \u20ac{fp:.0f}/tonne):")
        hdr4 = (f"  {'Speed':>6s}  {'Fac_NF':>12s}  {'Opt+Fl':>12s}  "
                f"{'Saving':>12s}  {'P/R':>10s}  {'Flettner':>10s}")
        print(hdr4)
        print(f"  {'-' * 6}  {'-' * 12}  {'-' * 12}  "
              f"{'-' * 12}  {'-' * 10}  {'-' * 10}")
        for s in sweep:
            cost_fac = s.ann_fuel_factory_nf_t * fp
            cost_opt = s.ann_fuel_opt_fl_t * fp
            cost_sav = s.ann_saving_total_t * fp
            cost_pr = s.ann_saving_pitch_rpm_t * fp
            cost_fl = s.ann_saving_flettner_t * fp
            print(f"  {s.speed_kn:5.1f}kn  "
                  f"\u20ac{cost_fac:>10,.0f}  "
                  f"\u20ac{cost_opt:>10,.0f}  "
                  f"\u20ac{cost_sav:>10,.0f}  "
                  f"\u20ac{cost_pr:>8,.0f}  "
                  f"\u20ac{cost_fl:>8,.0f}")

    print()


def plot_speed_sweep(sweep: list[SpeedSweepResult]):
    """Generate speed sensitivity plots.

    Produces two figures:
    1. speed_sweep_fuel.png -- Fuel consumption & savings vs speed
    2. speed_sweep_savings.png -- Savings breakdown vs speed
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping speed sweep plots")
        return

    if len(sweep) < 2:
        print("Need at least 2 speeds for sweep plots — skipping")
        return

    out_dir = Path(__file__).parent
    speeds = [s.speed_kn for s in sweep]

    # ---- Figure 1: Fuel & savings overview (3-panel) ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Per-voyage fuel consumption
    ax = axes[0]
    ax.plot(speeds, [s.mean_fuel_factory_nf_kg for s in sweep],
            "s-", color="#c0392b", linewidth=2, markersize=8,
            label="Factory (no Flettner)")
    ax.plot(speeds, [s.mean_fuel_opt_nf_kg for s in sweep],
            "^-", color="#e67e22", linewidth=2, markersize=8,
            label="Optimiser (no Flettner)")
    ax.plot(speeds, [s.mean_fuel_opt_fl_kg for s in sweep],
            "o-", color="#27ae60", linewidth=2, markersize=8,
            label="Optimiser + Flettner")
    ax.set_xlabel("Speed [kn]")
    ax.set_ylabel("Fuel per voyage [kg]")
    ax.set_title("Per-Voyage Fuel Consumption")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Add percentage annotations
    for s in sweep:
        ax.annotate(f"{s.mean_saving_pct:+.1f}%",
                     (s.speed_kn, s.mean_fuel_opt_fl_kg),
                     textcoords="offset points", xytext=(0, -18),
                     fontsize=8, ha="center", color="#27ae60",
                     fontweight="bold")

    # Panel 2: Annualized fuel
    ax = axes[1]
    ax.plot(speeds, [s.ann_fuel_factory_nf_t for s in sweep],
            "s-", color="#c0392b", linewidth=2, markersize=8,
            label="Factory (no Flettner)")
    ax.plot(speeds, [s.ann_fuel_opt_fl_t for s in sweep],
            "o-", color="#27ae60", linewidth=2, markersize=8,
            label="Optimiser + Flettner")
    # Shade the saving region
    ax.fill_between(speeds,
                     [s.ann_fuel_opt_fl_t for s in sweep],
                     [s.ann_fuel_factory_nf_t for s in sweep],
                     alpha=0.15, color="#27ae60")
    ax.set_xlabel("Speed [kn]")
    ax.set_ylabel("Fuel [tonnes/year]")
    ax.set_title("Annualized Fuel Consumption")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Annotate annual saving
    for s in sweep:
        mid_y = (s.ann_fuel_factory_nf_t + s.ann_fuel_opt_fl_t) / 2
        ax.annotate(f"{s.ann_saving_total_t:.0f} t/yr",
                     (s.speed_kn, mid_y),
                     fontsize=8, ha="center", color="#27ae60",
                     fontweight="bold")

    # Panel 3: Saving percentages (stacked)
    ax = axes[2]
    bar_w = 0.6 * min(np.diff(speeds)) if len(speeds) > 1 else 0.6
    bar_w = min(bar_w, 0.8)
    pr_pcts = [s.pct_pitch_rpm for s in sweep]
    fl_pcts = [s.pct_flettner for s in sweep]
    bars1 = ax.bar(speeds, pr_pcts, width=bar_w,
                    color="#3574a3", label="Pitch/RPM optimisation")
    bars2 = ax.bar(speeds, fl_pcts, width=bar_w, bottom=pr_pcts,
                    color="#e8774a", label="Flettner wind assist")
    # Labels on bars
    for i, s in enumerate(sweep):
        total = s.pct_pitch_rpm + s.pct_flettner
        ax.text(s.speed_kn, total + 0.3, f"{total:.1f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Speed [kn]")
    ax.set_ylabel("Saving [% of factory baseline]")
    ax.set_title("Savings Breakdown by Speed")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(speeds)

    plt.tight_layout()
    p = out_dir / "speed_sweep_fuel.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Plot saved: {p}")

    # ---- Figure 2: Savings detail (2-panel) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Per-voyage savings in kg (stacked bar)
    ax = axes[0]
    pr_kg = [s.mean_saving_pitch_rpm_kg for s in sweep]
    fl_kg = [s.mean_saving_flettner_kg for s in sweep]
    ax.bar(speeds, pr_kg, width=bar_w,
           color="#3574a3", label="Pitch/RPM")
    ax.bar(speeds, fl_kg, width=bar_w, bottom=pr_kg,
           color="#e8774a", label="Flettner")
    for i, s in enumerate(sweep):
        total = s.mean_saving_total_kg
        ax.text(s.speed_kn, total + 5, f"{total:.0f} kg",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Speed [kn]")
    ax.set_ylabel("Saving per voyage [kg]")
    ax.set_title("Per-Voyage Fuel Saving")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(speeds)

    # Panel 2: Annualized savings in tonnes (stacked bar)
    ax = axes[1]
    pr_t = [s.ann_saving_pitch_rpm_t for s in sweep]
    fl_t = [s.ann_saving_flettner_t for s in sweep]
    ax.bar(speeds, pr_t, width=bar_w,
           color="#3574a3", label="Pitch/RPM")
    ax.bar(speeds, fl_t, width=bar_w, bottom=pr_t,
           color="#e8774a", label="Flettner")
    for i, s in enumerate(sweep):
        total = s.ann_saving_total_t
        ax.text(s.speed_kn, total + 1, f"{total:.0f} t/yr",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Speed [kn]")
    ax.set_ylabel("Saving [tonnes/year]")
    ax.set_title("Annualized Fuel Saving")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(speeds)

    plt.tight_layout()
    p = out_dir / "speed_sweep_savings.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"Plot saved: {p}")


# ============================================================
# CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Long-term hindcast voyage comparison: "
                    "factory combinator vs optimiser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--year", type=int, default=2024,
                        help="Hindcast year (default: 2024)")
    parser.add_argument("--speed", type=float, default=10.0,
                        help="Transit speed [kn] (default: 10)")
    parser.add_argument("--download", action="store_true",
                        help="Download NORA3 data for the route (run first)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="NORA3 data directory (default: ./data/nora3_route)")
    parser.add_argument("--pdstrip", type=str, default=PDSTRIP_DAT,
                        help="Path to PdStrip .dat file")
    parser.add_argument("--no-flettner", action="store_true",
                        help="Disable Flettner rotor (waves only)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate summary plots")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-voyage output")
    parser.add_argument("--sg-load", type=float, default=0.0,
                        help="Actual shaft generator load [kW] for the "
                             "optimiser (default: 0)")
    parser.add_argument("--sg-factory-allowance", type=float, default=0.0,
                        help="SG power allowance in factory combinator "
                             "schedule [kW] (default: 0)")
    parser.add_argument("--sg-freq-min", type=float, default=0.0,
                        help="SG minimum frequency [Hz] (default: 0, "
                             "no constraint)")
    parser.add_argument("--sg-freq-max", type=float, default=0.0,
                        help="SG maximum frequency [Hz] (default: 0, "
                             "no constraint)")
    parser.add_argument("--idle-pct", type=float, default=15.0,
                        help="Percentage of year spent idle/in port "
                             "(default: 15). Used to annualize fuel figures.")
    parser.add_argument("--compare-sg", action="store_true",
                        help="Run both standard and SG modes, then produce "
                             "side-by-side comparison plots. Requires "
                             "--sg-load and --sg-factory-allowance.")
    parser.add_argument("--speed-sweep", nargs="+", type=float, default=None,
                        metavar="KN",
                        help="Run a speed sensitivity sweep at multiple "
                             "speeds [kn]. Example: --speed-sweep 8 10 12 14. "
                             "Produces comparison tables and plots.")
    parser.add_argument("--fuel-price", type=float, default=650.0,
                        help="Fuel price [EUR/tonne] for cost estimates "
                             "(default: 650, typical MGO in ECA zone)")
    parser.add_argument("--one-way", action="store_true",
                        help="Evaluate only the outbound leg (default is "
                             "round-trip to eliminate directional wind bias)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else NORA3_DATA_DIR
    do_round_trip = not args.one_way

    if args.download:
        download_nora3_for_route(args.year, ROUTE_ROTTERDAM_GOTHENBURG,
                                 data_dir)
        return

    if args.speed_sweep:
        speeds = sorted(set(args.speed_sweep))
        print(f"\n{'#' * 78}")
        print(f"# SPEED SENSITIVITY SWEEP: {', '.join(f'{s:.0f}' for s in speeds)} kn")
        print(f"{'#' * 78}\n")
        sweep = run_speed_sweep(
            speeds=speeds,
            year=args.year,
            idle_pct=args.idle_pct,
            data_dir=data_dir,
            pdstrip_path=args.pdstrip,
            flettner_enabled=not args.no_flettner,
            verbose=not args.quiet,
            sg_load_kw=args.sg_load,
            sg_factory_allowance_kw=args.sg_factory_allowance,
            sg_freq_min=args.sg_freq_min,
            sg_freq_max=args.sg_freq_max,
            round_trip=do_round_trip,
        )
        print_speed_sweep_summary(sweep, idle_pct=args.idle_pct,
                                  fuel_price_eur_per_t=args.fuel_price)
        plot_speed_sweep(sweep)
        return

    if args.compare_sg:
        # Run both standard and SG modes
        print("\n" + "#" * 78)
        print("# STANDARD MODE (no shaft generator)")
        print("#" * 78 + "\n")
        results_std = run_annual_comparison(
            year=args.year,
            speed_kn=args.speed,
            data_dir=data_dir,
            pdstrip_path=args.pdstrip,
            flettner_enabled=not args.no_flettner,
            verbose=not args.quiet,
            round_trip=do_round_trip,
        )
        print_summary(results_std, args.speed, idle_pct=args.idle_pct,
                      fuel_price_eur_per_t=args.fuel_price,
                      round_trip=do_round_trip)

        print("\n\n" + "#" * 78)
        print("# SHAFT GENERATOR MODE")
        print("#" * 78 + "\n")
        results_sg = run_annual_comparison(
            year=args.year,
            speed_kn=args.speed,
            data_dir=data_dir,
            pdstrip_path=args.pdstrip,
            flettner_enabled=not args.no_flettner,
            verbose=not args.quiet,
            sg_load_kw=args.sg_load,
            sg_factory_allowance_kw=args.sg_factory_allowance,
            sg_freq_min=args.sg_freq_min,
            sg_freq_max=args.sg_freq_max,
            round_trip=do_round_trip,
        )
        print_summary(results_sg, args.speed, idle_pct=args.idle_pct,
                      fuel_price_eur_per_t=args.fuel_price,
                      round_trip=do_round_trip)

        plot_comparison(results_std, results_sg, args.speed,
                        idle_pct=args.idle_pct)
        return

    results = run_annual_comparison(
        year=args.year,
        speed_kn=args.speed,
        data_dir=data_dir,
        pdstrip_path=args.pdstrip,
        flettner_enabled=not args.no_flettner,
        verbose=not args.quiet,
        sg_load_kw=args.sg_load,
        sg_factory_allowance_kw=args.sg_factory_allowance,
        sg_freq_min=args.sg_freq_min,
        sg_freq_max=args.sg_freq_max,
        round_trip=do_round_trip,
    )

    print_summary(results, args.speed, idle_pct=args.idle_pct,
                  fuel_price_eur_per_t=args.fuel_price,
                  round_trip=do_round_trip)

    if args.plot:
        plot_results(results)


if __name__ == "__main__":
    main()
