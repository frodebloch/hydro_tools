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
NU_AIR = 1.5e-5             # kinematic viscosity of air [m^2/s]
GENSET_SFOC = 215.0         # auxiliary genset SFOC [g/kWh] for rotor motor power

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
        # at 10 m/s apparent wind and ~107 kW at 15 m/s for a 28×4 m rotor,
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
# Hull and propeller roughness models
# ============================================================

# Hull geometry for roughness penalty calculation
HULL_S_WET = 3153.0         # wetted surface area with appendages [m^2]
HULL_LWL = 96.0             # waterline length [m] (approx from Loa 100m)
NU_WATER = 1.19e-6          # kinematic viscosity of seawater at 10°C [m^2/s]
PROP_N_BLADES = 4           # number of propeller blades


# --- Biofouling growth model (from rdt_analysis/fouling_analysis.py) -------
# Three scenarios based on Schultz (2007), Townsin (2003), Uzun (2019).
# Each gives ks [µm] vs time [years] as piecewise-linear interpolation.

@dataclass
class FoulingScenario:
    """Piecewise-linear fouling roughness growth trajectory.

    Defined by discrete (year, ks) data points from literature,
    with linear interpolation between them.
    """
    name: str
    years: list      # [year, ...] must start at 0
    ks_values: list  # [ks in µm, ...]

    def ks_um_at(self, t: float) -> float:
        """Return ks [µm] at time t [years] by linear interpolation."""
        return float(np.interp(t, self.years, self.ks_values))

    def ks_m_at(self, t: float) -> float:
        """Return ks [m] at time t [years]."""
        return self.ks_um_at(t) * 1e-6


# Hull fouling scenarios
FOULING_LOW = FoulingScenario(
    name="Low (Nordic, good AF)",
    years=     [0,  1,   2,   3,   4,   5],
    ks_values= [30, 50,  80,  150, 250, 500],   # µm
)
FOULING_CENTRAL = FoulingScenario(
    name="Central (temperate, std SPC)",
    years=     [0,  1,   2,   3,    4,    5],
    ks_values= [50, 100, 300, 700,  1500, 3000],
)
FOULING_HIGH = FoulingScenario(
    name="High (tropical, poor AF)",
    years=     [0,   1,   2,    3,    4,    5],
    ks_values= [100, 500, 2000, 5000, 8000, 10000],
)

# Propeller blade fouling scenario.
# Blades are partially self-cleaning due to hydrodynamic shear at RPM,
# so roughness grows much slower than the hull.  Typical values for a
# coated CPP blade in temperate waters (Nordic/North Sea).
BLADE_FOULING = FoulingScenario(
    name="Blade (coated CPP, Nordic)",
    years=     [0,  1,   2,   3,   4,    5],
    ks_values= [10, 20,  40,  70,  100,  150],  # µm
)


def hull_roughness_delta_R_kN(
    speed_kn: float,
    ks_hull_m: float,
    ks_clean_m: float = 30e-6,
    S_wet: float = HULL_S_WET,
    L_wl: float = HULL_LWL,
    nu: float = NU_WATER,
) -> float:
    """Compute hull frictional resistance increase due to roughness [kN].

    Uses the Townsin-Dey roughness penalty formula (Townsin 2003):

        10³ × ΔCF = 44 × [(ks/L)^(1/3) − 10·Re_L^(-1/3)] + 0.125

    i.e. ΔCF = {44 × [(ks/L)^(1/3) − 10·Re^(-1/3)] + 0.125} × 10⁻³

    applied as the difference between the current roughness state and the
    clean-hull baseline (ks_clean, default 30 µm for new AF coating).
    The clean-hull resistance is already in our calm-water data, so we
    only add the INCREMENT above the clean baseline.

    Parameters
    ----------
    speed_kn : float
        Vessel speed [knots].
    ks_hull_m : float
        Current hull equivalent sand-grain roughness [m].
    ks_clean_m : float
        Clean-hull roughness [m]. Default 30e-6 (new AF coating).
    S_wet : float
        Wetted surface area [m²].
    L_wl : float
        Waterline length [m].
    nu : float
        Kinematic viscosity [m²/s].

    Returns
    -------
    delta_R_kN : float
        Extra resistance due to roughness above clean baseline [kN].
        Always >= 0.
    """
    if ks_hull_m <= ks_clean_m:
        return 0.0

    V = speed_kn * KN_TO_MS
    Re_L = V * L_wl / nu

    def _townsin_delta_cf(ks_m):
        """Townsin-Dey ΔCF for a given ks."""
        if ks_m <= 0 or Re_L < 1:
            return 0.0
        return max(0.0, (44.0 * ((ks_m / L_wl) ** (1.0 / 3.0)
                                  - 10.0 * Re_L ** (-1.0 / 3.0))
                         + 0.125) * 1e-3)

    dcf_current = _townsin_delta_cf(ks_hull_m)
    dcf_clean = _townsin_delta_cf(ks_clean_m)
    delta_cf = max(0.0, dcf_current - dcf_clean)

    # ΔR = ΔCF × 0.5 × ρ × V² × S_wet
    delta_R_N = delta_cf * 0.5 * RHO_WATER * V ** 2 * S_wet
    return delta_R_N / 1000.0


def propeller_roughness_fuel_factor(
    speed_kn: float,
    shaft_rpm: float,
    ks_blade_m: float,
    ks_clean_m: float = 10e-6,
    D: float = PROP_DIAMETER,
    BAR: float = PROP_BAR,
    n_blades: int = PROP_N_BLADES,
    nu: float = NU_WATER,
) -> float:
    """Estimate propeller fuel rate multiplier due to blade surface roughness.

    Uses a simplified strip-theory approach at the 0.75R reference station
    (standard ITTC practice).  The extra blade drag is converted to a power
    increase fraction, which is then applied as a fuel rate multiplier.

    Method:
        1. Compute resultant velocity W at 0.75R from RPM and advance speed
        2. Compute chord at 0.75R from Wageningen B-series chord distribution
        3. Compute ΔCF = CF_rough − CF_smooth using ITTC 1957 + Townsin
        4. ΔCD = 2 × ΔCF (both blade surfaces)
        5. Estimate ΔP/P from the drag-to-power ratio at 0.75R

    Parameters
    ----------
    speed_kn : float
        Vessel speed [knots].
    shaft_rpm : float
        Shaft (propeller) RPM.
    ks_blade_m : float
        Blade surface equivalent sand-grain roughness [m].
    ks_clean_m : float
        Clean blade roughness [m]. Default 10e-6 (polished).
    D : float
        Propeller diameter [m].
    BAR : float
        Blade area ratio.
    n_blades : int
        Number of blades.
    nu : float
        Kinematic viscosity [m²/s].

    Returns
    -------
    fuel_factor : float
        Multiplier on fuel rate (>= 1.0).  E.g. 1.03 = 3% increase.
    """
    if ks_blade_m <= ks_clean_m or shaft_rpm < 1.0:
        return 1.0

    R = D / 2.0
    r_075 = 0.75 * R
    omega = shaft_rpm * 2.0 * math.pi / 60.0

    # Advance speed (use wake-corrected Va for consistency, but we
    # approximate with vessel speed here — the wake correction is small
    # and the same for smooth and rough)
    Va = speed_kn * KN_TO_MS * (1.0 - 0.24)  # approximate w ≈ 0.24

    # Resultant velocity at 0.75R
    V_tan = omega * r_075
    W = math.sqrt(Va ** 2 + V_tan ** 2)

    # Chord at 0.75R — Wageningen C4 chord distribution:
    # c/D = BAR × k_c(r/R) / n_blades, where k_c is the expanded chord factor.
    # For Wageningen B/C series at r/R=0.75, k_c ≈ 1.54 (from tabulated data).
    K_C_075 = 1.54
    c_075 = D * BAR * K_C_075 / n_blades

    # Chord Reynolds number
    Re_c = W * c_075 / nu

    # ITTC 1957 friction line for smooth blade:
    CF_smooth = 0.075 / (math.log10(Re_c) - 2.0) ** 2

    # Townsin roughness penalty at chord scale:
    # 10³ × ΔCF = 44 × [(ks/c)^(1/3) - 10*Re_c^(-1/3)] + 0.125
    def _townsin_cf(ks_m, Re):
        if ks_m <= 0 or Re < 1:
            return 0.0
        L_c = c_075  # use chord as length scale
        return max(0.0, (44.0 * ((ks_m / L_c) ** (1.0 / 3.0)
                                  - 10.0 * Re ** (-1.0 / 3.0))
                         + 0.125) * 1e-3)

    dcf_rough = _townsin_cf(ks_blade_m, Re_c)
    dcf_clean = _townsin_cf(ks_clean_m, Re_c)
    delta_CF = max(0.0, dcf_rough - dcf_clean)

    # ΔCD for both blade surfaces, with form factor for airfoil sections
    FORM_FACTOR = 1.3  # typical for marine propeller sections
    delta_CD = FORM_FACTOR * 2.0 * delta_CF

    # Power increase fraction:
    # At 0.75R the power-absorbing (tangential) fraction of drag is cos(φ)
    # where φ = atan2(Va, V_tan) is the inflow angle.
    # ΔP/P ≈ (ΔCD / CD_total) × (viscous drag fraction of total power)
    #
    # More directly: the smooth-blade CD ≈ 2 × CF_smooth × FORM_FACTOR
    # The viscous drag fraction of propeller power is approximately
    # CD / (CD + CL × tan(φ_i)) where φ_i is the induced inflow angle.
    # For a lightly loaded prop this is roughly 10-20% of total power.
    #
    # Simplification: ΔP/P ≈ ΔCD / (2 × CF_smooth × FORM_FACTOR) × f_drag
    # where f_drag ≈ 0.15 is the viscous drag power fraction.
    #
    # Actually the cleanest: ΔP/P = (n_blades × ΔD × V_tan × r_075 × ω) / P_total
    # but we don't have P_total here.  Use the ratio approach:
    CD_smooth = FORM_FACTOR * 2.0 * CF_smooth
    if CD_smooth < 1e-10:
        return 1.0

    # The relative drag increase applies to the viscous drag component only.
    # Viscous losses are typically 12-18% of delivered power for a CPP at
    # moderate loading.  Use 15% as representative.
    DRAG_POWER_FRACTION = 0.15
    delta_P_frac = (delta_CD / CD_smooth) * DRAG_POWER_FRACTION

    return 1.0 + max(0.0, delta_P_frac)


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
    rotor_power_kW: float = 0.0          # Flettner rotor electrical power [kW]
    rotor_fuel_rate: float = 0.0         # rotor fuel cost [g/h] via genset SFOC
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
    mean_rotor_power_kW: float = 0.0     # mean Flettner rotor electrical power
    total_rotor_fuel_kg: float = 0.0     # total fuel for rotor drive (all hours)
    hull_ks_um: float = 0.0              # hull roughness [µm]
    blade_ks_um: float = 0.0             # blade roughness [µm]
    R_roughness_kN: float = 0.0          # hull roughness resistance increment [kN]
    blade_fuel_factor: float = 1.0       # propeller roughness fuel multiplier
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
        mean_rotor_power_kW=(outbound.mean_rotor_power_kW * n_out
                             + inbound.mean_rotor_power_kW * n_ret) / n_total,
        total_rotor_fuel_kg=(outbound.total_rotor_fuel_kg
                              + inbound.total_rotor_fuel_kg),
        hull_ks_um=outbound.hull_ks_um,
        blade_ks_um=outbound.blade_ks_um,
        R_roughness_kN=outbound.R_roughness_kN,
        blade_fuel_factor=outbound.blade_fuel_factor,
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
    mean_rotor_power_kW: float
    # Feasibility
    pct_factory_infeasible: float
    # Roughness
    hull_ks_um: float = 0.0
    blade_ks_um: float = 0.0
    R_roughness_kN: float = 0.0
    blade_fuel_factor: float = 1.0


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
    hull_ks_m: float = 0.0,
    blade_ks_m: float = 0.0,
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

    # Hull roughness: extra resistance above clean baseline [kN]
    # Constant for a given speed — computed once, not per-hour.
    R_roughness_kN = 0.0
    if hull_ks_m > 30e-6:
        R_roughness_kN = hull_roughness_delta_R_kN(speed_kn, hull_ks_m)

    # Propeller blade roughness: fuel rate multiplier (>= 1.0)
    # Depends on RPM which varies per operating point, but the blade
    # section Reynolds number at 0.75R is dominated by tangential velocity
    # (ω·r >> Va), so the fuel factor is insensitive to exact RPM.
    # Use nominal shaft RPM scaled from max (117.6 RPM at ~15 kn).
    blade_fuel_factor = 1.0
    if blade_ks_m > 10e-6:
        nominal_shaft_rpm = 117.6 * (speed_kn / 15.0)
        blade_fuel_factor = propeller_roughness_fuel_factor(
            speed_kn, nominal_shaft_rpm, blade_ks_m)

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
    sum_rotor_power = 0.0
    sum_rotor_fuel = 0.0

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
        rotor_power_kW = flettner.rotor_power_kW if F_flettner_kN > 0 else 0.0
        # Fuel cost of spinning the rotor [g/h] via auxiliary genset
        rotor_fuel_gh = rotor_power_kW * GENSET_SFOC  # g/h

        # Wind resistance on hull (Blendermann model)
        R_wind_kN = wind_resistance_kN(
            Vs=Vs,
            heading_deg=rp.heading_deg,
            wind_speed=wx["wind_speed"],
            wind_dir_deg=wx["wind_dir"],
        )

        # Net thrust demand on propeller
        # T_total = T_calm + (R_aw + R_wind + R_roughness - F_flettner) / (1-t)
        # The added resistance, wind resistance, hull roughness, and Flettner
        # force act on the hull, so they modify the resistance which is then
        # divided by (1-t) to get thrust.
        T_required_kN = T_calm_kN + (R_aw_kN + R_wind_kN + R_roughness_kN - F_flettner_kN) / (1.0 - t_ded)
        T_required_kN = max(0.0, T_required_kN)  # can't have negative thrust
        T_required_N = T_required_kN * 1000.0

        # Thrust demand without Flettner (for splitting savings)
        T_no_flettner_kN = T_calm_kN + (R_aw_kN + R_wind_kN + R_roughness_kN) / (1.0 - t_ded)
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
            rotor_power_kW=rotor_power_kW,
            rotor_fuel_rate=rotor_fuel_gh,
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
            rotor_fuel_kg = rotor_fuel_gh / 1000.0  # g/h -> kg for 1h
            total_factory_feasible += fr["fuel_rate"] / 1000.0 * blade_fuel_factor + rotor_fuel_kg
            total_factory_nf_feasible += fr_nf["fuel_rate"] / 1000.0 * blade_fuel_factor
            total_optimised_feasible += o_fuel / 1000.0 * blade_fuel_factor + rotor_fuel_kg
            total_opt_noflettner_feasible += o_nf_fuel / 1000.0 * blade_fuel_factor
            n_both_feasible += 1
        elif fr is None and o_fuel is not None:
            # Factory infeasible but optimiser can deliver
            n_factory_infeasible += 1

        sum_hs += wx["hs"]
        sum_wind += wx["wind_speed"]
        sum_R_aw += R_aw_kN
        sum_R_wind += R_wind_kN
        sum_F_flettner += F_flettner_kN
        sum_rotor_power += rotor_power_kW
        sum_rotor_fuel += rotor_fuel_gh / 1000.0  # g/h -> kg for 1h

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
        mean_rotor_power_kW=sum_rotor_power / n_pts,
        total_rotor_fuel_kg=sum_rotor_fuel,
        hull_ks_um=hull_ks_m * 1e6,
        blade_ks_um=blade_ks_m * 1e6,
        R_roughness_kN=R_roughness_kN,
        blade_fuel_factor=blade_fuel_factor,
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
     return_route, hull_ks_m, blade_ks_m) = args
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
                hull_ks_m=hull_ks_m,
                blade_ks_m=blade_ks_m,
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
                    hull_ks_m=hull_ks_m,
                    blade_ks_m=blade_ks_m,
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
    hull_ks_m: float = 0.0,
    blade_ks_m: float = 0.0,
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
    if hull_ks_m > 0 or blade_ks_m > 0:
        print(f"  Hull roughness: {hull_ks_m * 1e6:.0f} µm, "
              f"blade roughness: {blade_ks_m * 1e6:.0f} µm")
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
                             endplate_ratio=2.0, roughness_factor=3.0)
    if flettner_enabled:
        flettner.set_target_spin_ratio(3.0)
        print(f"  Rotor: H=28m, D=4m, max 220 RPM, target SR=3, k_rough=3.0")
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
            return_route, hull_ks_m, blade_ks_m,
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
    mean_rotor_pwr = np.array([r.mean_rotor_power_kW for r in results])
    rotor_fuel = np.array([r.total_rotor_fuel_kg for r in results])

    print("\n" + "=" * 78)
    print("ANNUAL SUMMARY")
    print("=" * 78)

    print(f"\n  Voyages simulated: {len(results)} ({voy_label})")
    print(f"  Transit speed: {speed_kn:.0f} kn")
    print(f"  Transit time per voyage: {transit_h:.1f} h ({transit_h / 24:.1f} days)")
    print(f"  Idle / port time: {idle_pct:.0f}%")
    print(f"  Estimated {voy_label} voyages per year: {voyages_per_year:.0f}")

    # Roughness state (from first result)
    r0 = results[0]
    if r0.hull_ks_um > 0 or r0.blade_ks_um > 0:
        print(f"\n  Hull roughness: {r0.hull_ks_um:.0f} µm "
              f"(ΔR = {r0.R_roughness_kN:.1f} kN at {speed_kn:.0f} kn)")
        print(f"  Blade roughness: {r0.blade_ks_um:.0f} µm "
              f"(fuel factor = {r0.blade_fuel_factor:.3f})")

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
    _row("Mean rotor power [kW]", mean_rotor_pwr)
    _row("Rotor fuel [kg/voy]", rotor_fuel, ".0f")

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


def print_departure_analysis(results: list[VoyageResult], speed_kn: float,
                              idle_pct: float = 15.0,
                              fuel_price_eur_per_t: float = 650.0,
                              round_trip: bool = True,
                              windows: tuple[int, ...] = (3, 5, 7, 14)):
    """Analyse the value of departure timing flexibility.

    For each scheduling window size, computes what fuel saving could be
    achieved by choosing the best departure date within consecutive
    non-overlapping windows across the year.

    Also examines correlation between Flettner benefit and weather, and
    quantifies how much "smart scheduling" could enhance the Flettner value.

    Parameters
    ----------
    results : list[VoyageResult]
        One per departure day (365), sorted by departure date.
    speed_kn : float
        Transit speed.
    idle_pct : float
        Percentage of year idle (for annualization).
    fuel_price_eur_per_t : float
        Fuel price EUR/tonne.
    round_trip : bool
        Whether results are round-trips.
    windows : tuple[int, ...]
        Scheduling flexibility windows in days.
    """
    if not results or len(results) < 14:
        print("Insufficient results for departure analysis.")
        return

    n = len(results)
    transit_h = results[0].total_hours
    sailing_hours_year = 365.25 * 24.0 * (1.0 - idle_pct / 100.0)
    voyages_per_year = sailing_hours_year / transit_h
    voy_label = "round-trip" if round_trip else "one-way"

    # Extract arrays
    dates = [r.departure for r in results]
    fuel_fac_nf = np.array([r.total_fuel_factory_no_flettner_kg for r in results])
    fuel_opt_fl = np.array([r.total_fuel_optimised_kg for r in results])
    fuel_opt_nf = np.array([r.total_fuel_opt_no_flettner_kg for r in results])
    fuel_fac_fl = np.array([r.total_fuel_factory_kg for r in results])
    sav_total_kg = fuel_fac_nf - fuel_opt_fl       # total saving vs baseline
    sav_flettner_kg = fuel_opt_nf - fuel_opt_fl     # Flettner contribution
    sav_pitch_rpm_kg = fuel_fac_nf - fuel_opt_nf    # pitch/RPM contribution
    mean_wind = np.array([r.mean_wind for r in results])
    mean_hs = np.array([r.mean_hs for r in results])
    mean_F_fl = np.array([r.mean_F_flettner_kN for r in results])

    print("\n" + "=" * 78)
    print("DEPARTURE TIMING ANALYSIS")
    print("=" * 78)

    # ---- 1. Overall fuel variability ----
    print(f"\n  1. FUEL VARIABILITY ACROSS {n} DEPARTURE DATES")
    print(f"  {'':>3s}  {'':>33s} {'Mean':>8s} {'Std':>8s} "
          f"{'Min':>8s} {'Max':>8s} {'Range':>8s}")

    for label, arr in [("Factory NF [kg/voy]", fuel_fac_nf),
                       ("Opt+Flettner [kg/voy]", fuel_opt_fl),
                       ("Total saving [kg/voy]", sav_total_kg),
                       ("Flettner saving [kg]", sav_flettner_kg)]:
        print(f"     {label:<33s} {np.mean(arr):>8.0f} {np.std(arr):>8.0f} "
              f"{np.min(arr):>8.0f} {np.max(arr):>8.0f} "
              f"{np.max(arr) - np.min(arr):>8.0f}")

    cv_fac = 100 * np.std(fuel_fac_nf) / np.mean(fuel_fac_nf)
    cv_opt = 100 * np.std(fuel_opt_fl) / np.mean(fuel_opt_fl)
    print(f"\n     Coefficient of variation:  Factory NF = {cv_fac:.1f}%,  "
          f"Opt+Fl = {cv_opt:.1f}%")
    print(f"     => Weather causes ~{cv_fac:.0f}% voyage-to-voyage fuel variation")

    # ---- 2. Scheduling window analysis ----
    print(f"\n  2. VALUE OF DEPARTURE FLEXIBILITY")
    print(f"     (choosing the lowest-fuel departure within a sliding window)\n")
    print(f"     {'Window':>8s}  {'Mean best':>10s}  {'Mean worst':>11s}  "
          f"{'Mean saving':>12s}  {'Save %':>7s}  "
          f"{'Annual':>8s}  {'EUR/yr':>10s}")
    print(f"     {'[days]':>8s}  {'[kg/voy]':>10s}  {'[kg/voy]':>11s}  "
          f"{'[kg/voy]':>12s}  {'':>7s}  "
          f"{'[t/yr]':>8s}  {'':>10s}")
    print(f"     {'-' * 8}  {'-' * 10}  {'-' * 11}  "
          f"{'-' * 12}  {'-' * 7}  {'-' * 8}  {'-' * 10}")

    # Use the factory NF fuel for scheduling analysis (baseline, no optimiser)
    # This shows the value of timing alone.
    # Also compute for opt+Fl to show combined value.
    for w in windows:
        if w > n:
            continue
        # Sliding window: for each possible window start, find min and max
        best_fac_nf = []
        worst_fac_nf = []
        best_opt_fl = []
        worst_opt_fl = []
        for i in range(n - w + 1):
            chunk_fac = fuel_fac_nf[i:i + w]
            chunk_opt = fuel_opt_fl[i:i + w]
            best_fac_nf.append(np.min(chunk_fac))
            worst_fac_nf.append(np.max(chunk_fac))
            best_opt_fl.append(np.min(chunk_opt))
            worst_opt_fl.append(np.max(chunk_opt))

        mean_best = np.mean(best_fac_nf)
        mean_worst = np.mean(worst_fac_nf)
        mean_sav = mean_worst - mean_best
        pct = 100 * mean_sav / mean_worst if mean_worst > 0 else 0
        annual_t = mean_sav * voyages_per_year / 1000.0
        annual_eur = annual_t * fuel_price_eur_per_t

        print(f"     {w:>8d}  {mean_best:>10.0f}  {mean_worst:>11.0f}  "
              f"{mean_sav:>12.0f}  {pct:>6.1f}%  "
              f"{annual_t:>8.1f}  {annual_eur:>10,.0f}")

    # ---- 3. Scheduling + optimiser + Flettner combined ----
    print(f"\n  3. COMBINED VALUE: TIMING + OPTIMISER + FLETTNER")
    print(f"     (best Opt+Fl day in window vs mean Factory NF)\n")
    print(f"     {'Window':>8s}  {'Mean best':>10s}  {'Baseline':>10s}  "
          f"{'Mean saving':>12s}  {'Save %':>7s}  "
          f"{'Annual':>8s}  {'EUR/yr':>10s}")
    print(f"     {'[days]':>8s}  {'Opt+Fl':>10s}  {'Fac NF':>10s}  "
          f"{'[kg/voy]':>12s}  {'':>7s}  "
          f"{'[t/yr]':>8s}  {'':>10s}")
    print(f"     {'-' * 8}  {'-' * 10}  {'-' * 10}  "
          f"{'-' * 12}  {'-' * 7}  {'-' * 8}  {'-' * 10}")

    mean_fac_nf = np.mean(fuel_fac_nf)
    # Without any flexibility (baseline)
    mean_opt_fl = np.mean(fuel_opt_fl)
    base_sav_kg = mean_fac_nf - mean_opt_fl
    base_pct = 100 * base_sav_kg / mean_fac_nf if mean_fac_nf > 0 else 0
    base_t = base_sav_kg * voyages_per_year / 1000.0
    base_eur = base_t * fuel_price_eur_per_t
    print(f"     {'none':>8s}  {mean_opt_fl:>10.0f}  {mean_fac_nf:>10.0f}  "
          f"{base_sav_kg:>12.0f}  {base_pct:>6.1f}%  "
          f"{base_t:>8.1f}  {base_eur:>10,.0f}")

    for w in windows:
        if w > n:
            continue
        best_combined = []
        for i in range(n - w + 1):
            best_combined.append(np.min(fuel_opt_fl[i:i + w]))
        mean_best_c = np.mean(best_combined)
        sav_c = mean_fac_nf - mean_best_c
        pct_c = 100 * sav_c / mean_fac_nf if mean_fac_nf > 0 else 0
        annual_t = sav_c * voyages_per_year / 1000.0
        annual_eur = annual_t * fuel_price_eur_per_t
        print(f"     {w:>8d}  {mean_best_c:>10.0f}  {mean_fac_nf:>10.0f}  "
              f"{sav_c:>12.0f}  {pct_c:>6.1f}%  "
              f"{annual_t:>8.1f}  {annual_eur:>10,.0f}")

    # ---- 4. Correlation analysis ----
    print(f"\n  4. WEATHER CORRELATIONS")
    # Pearson correlation between Flettner saving and weather
    # Use only voyages where Flettner saving > 0 (i.e. rotor was active)
    corr_fl_wind = np.corrcoef(sav_flettner_kg, mean_wind)[0, 1]
    corr_fl_hs = np.corrcoef(sav_flettner_kg, mean_hs)[0, 1]
    corr_fl_thrust = np.corrcoef(sav_flettner_kg, mean_F_fl)[0, 1]
    corr_fuel_hs = np.corrcoef(fuel_fac_nf, mean_hs)[0, 1]
    corr_fuel_wind = np.corrcoef(fuel_fac_nf, mean_wind)[0, 1]

    print(f"     Pearson r between Flettner saving and mean wind:   {corr_fl_wind:>+.3f}")
    print(f"     Pearson r between Flettner saving and mean Hs:     {corr_fl_hs:>+.3f}")
    print(f"     Pearson r between Flettner saving and mean F_fl:   {corr_fl_thrust:>+.3f}")
    print(f"     Pearson r between Factory NF fuel and mean Hs:     {corr_fuel_hs:>+.3f}")
    print(f"     Pearson r between Factory NF fuel and mean wind:   {corr_fuel_wind:>+.3f}")

    # The key insight: when wind is high, BOTH total fuel goes up (waves, wind
    # drag) AND Flettner saving goes up. Does the Flettner saving outweigh the
    # extra fuel? Look at the net effect.
    corr_opt_fl_wind = np.corrcoef(fuel_opt_fl, mean_wind)[0, 1]
    corr_opt_fl_hs = np.corrcoef(fuel_opt_fl, mean_hs)[0, 1]
    print(f"\n     Pearson r between Opt+Fl fuel and mean wind:      {corr_opt_fl_wind:>+.3f}")
    print(f"     Pearson r between Opt+Fl fuel and mean Hs:        {corr_opt_fl_hs:>+.3f}")

    # Does the Flettner reduce weather sensitivity?
    print(f"\n     Fuel std dev:  Factory NF = {np.std(fuel_fac_nf):.0f} kg,  "
          f"Opt+Fl = {np.std(fuel_opt_fl):.0f} kg")
    if np.std(fuel_opt_fl) < np.std(fuel_fac_nf):
        reduction = 100 * (1 - np.std(fuel_opt_fl) / np.std(fuel_fac_nf))
        print(f"     => Flettner + optimiser reduces fuel variability by "
              f"{reduction:.0f}%")
    else:
        increase = 100 * (np.std(fuel_opt_fl) / np.std(fuel_fac_nf) - 1)
        print(f"     => Opt+Fl has {increase:.0f}% MORE fuel variability than "
              f"Factory NF")
        print(f"        (Flettner adds variance: large saving in wind, "
              f"near-zero in calm)")

    # ---- 5. Monthly breakdown ----
    print(f"\n  5. MONTHLY BREAKDOWN")
    print(f"     {'Month':<10s}  {'N':>4s}  {'FacNF':>7s}  {'Opt+Fl':>7s}  "
          f"{'Save':>6s}  {'Fl save':>8s}  {'Wind':>5s}  {'Hs':>5s}  "
          f"{'F_fl':>5s}")
    print(f"     {'':>10s}  {'':>4s}  {'kg/voy':>7s}  {'kg/voy':>7s}  "
          f"{'%':>6s}  {'kg/voy':>8s}  {'m/s':>5s}  {'m':>5s}  "
          f"{'kN':>5s}")
    print(f"     {'-' * 10}  {'-' * 4}  {'-' * 7}  {'-' * 7}  "
          f"{'-' * 6}  {'-' * 8}  {'-' * 5}  {'-' * 5}  {'-' * 5}")

    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    monthly_sav_pct = []
    for m in range(1, 13):
        idx = [i for i, d in enumerate(dates) if d.month == m]
        if not idx:
            continue
        m_fac_nf = np.mean(fuel_fac_nf[idx])
        m_opt_fl = np.mean(fuel_opt_fl[idx])
        m_sav_pct = 100 * (m_fac_nf - m_opt_fl) / m_fac_nf if m_fac_nf > 0 else 0
        m_fl_sav = np.mean(sav_flettner_kg[idx])
        m_wind = np.mean(mean_wind[idx])
        m_hs = np.mean(mean_hs[idx])
        m_ffl = np.mean(mean_F_fl[idx])
        monthly_sav_pct.append(m_sav_pct)
        print(f"     {months[m - 1]:<10s}  {len(idx):>4d}  {m_fac_nf:>7.0f}  "
              f"{m_opt_fl:>7.0f}  {m_sav_pct:>5.1f}%  {m_fl_sav:>8.0f}  "
              f"{m_wind:>5.1f}  {m_hs:>5.2f}  {m_ffl:>5.1f}")

    # ---- 6. "Smart scheduling" Flettner premium ----
    # If an operator could choose WHICH days to sail (within flexibility),
    # how much more Flettner benefit could they capture?
    print(f"\n  6. SMART SCHEDULING: FLETTNER-AWARE DEPARTURE SELECTION")
    print(f"     (within each window, choose the day with maximum Flettner saving)\n")
    print(f"     {'Window':>8s}  {'Mean Fl':>9s}  {'vs random':>10s}  "
          f"{'Fl boost':>9s}")
    print(f"     {'[days]':>8s}  {'save kg':>9s}  {'Fl save':>10s}  "
          f"{'':>9s}")
    print(f"     {'-' * 8}  {'-' * 9}  {'-' * 10}  {'-' * 9}")

    mean_fl_sav = np.mean(sav_flettner_kg)
    print(f"     {'none':>8s}  {mean_fl_sav:>9.0f}  {'(baseline)':>10s}  "
          f"{'':>9s}")

    for w in windows:
        if w > n:
            continue
        best_fl = []
        for i in range(n - w + 1):
            chunk = sav_flettner_kg[i:i + w]
            best_fl.append(np.max(chunk))
        mean_best_fl = np.mean(best_fl)
        boost = 100 * (mean_best_fl / mean_fl_sav - 1) if mean_fl_sav > 0 else 0
        print(f"     {w:>8d}  {mean_best_fl:>9.0f}  {mean_fl_sav:>10.0f}  "
              f"{boost:>+8.0f}%")

    # ---- 7. Worst-case avoidance ----
    print(f"\n  7. WORST-CASE AVOIDANCE")
    print(f"     (value of avoiding the highest-fuel departure in each window)\n")
    p95_fuel = np.percentile(fuel_fac_nf, 95)
    p99_fuel = np.percentile(fuel_fac_nf, 99)
    worst_fuel = np.max(fuel_fac_nf)
    worst_idx = np.argmax(fuel_fac_nf)
    worst_date = dates[worst_idx]
    print(f"     Worst departure:  {worst_date.strftime('%Y-%m-%d')}  "
          f"fuel = {worst_fuel:.0f} kg  "
          f"(Hs = {mean_hs[worst_idx]:.2f} m, wind = {mean_wind[worst_idx]:.1f} m/s)")
    best_fuel = np.min(fuel_fac_nf)
    best_idx = np.argmin(fuel_fac_nf)
    best_date = dates[best_idx]
    print(f"     Best departure:   {best_date.strftime('%Y-%m-%d')}  "
          f"fuel = {best_fuel:.0f} kg  "
          f"(Hs = {mean_hs[best_idx]:.2f} m, wind = {mean_wind[best_idx]:.1f} m/s)")
    print(f"     Range: {worst_fuel - best_fuel:.0f} kg  "
          f"({100 * (worst_fuel - best_fuel) / best_fuel:.0f}% of best)")
    print(f"     P95 fuel: {p95_fuel:.0f} kg,  P99: {p99_fuel:.0f} kg")

    # Best Opt+Fl departure (may differ from best Factory NF!)
    best_opt_idx = np.argmin(fuel_opt_fl)
    best_opt_date = dates[best_opt_idx]
    print(f"\n     Best Opt+Fl departure: {best_opt_date.strftime('%Y-%m-%d')}  "
          f"fuel = {fuel_opt_fl[best_opt_idx]:.0f} kg  "
          f"(Hs = {mean_hs[best_opt_idx]:.2f} m, "
          f"wind = {mean_wind[best_opt_idx]:.1f} m/s, "
          f"Fl thrust = {mean_F_fl[best_opt_idx]:.1f} kN)")
    if best_opt_idx != best_idx:
        print(f"     NOTE: best Opt+Fl departure differs from best Factory NF "
              f"departure!")
        print(f"           Factory NF prefers calm; Opt+Fl can prefer windy "
              f"(Flettner benefit).")

    n_above_p95 = np.sum(fuel_fac_nf > p95_fuel)
    print(f"\n     Departures above P95 ({p95_fuel:.0f} kg): {n_above_p95}")
    print(f"     With 7-day flexibility, {n_above_p95}->{{}}-of-these avoidable:".format(
        "most" if n_above_p95 > 5 else "some"))

    # Actually compute: within each 7-day window, could the P95 departures
    # have been avoided?
    w = 7
    n_avoidable = 0
    n_p95_checked = 0
    for i in range(n):
        if fuel_fac_nf[i] > p95_fuel:
            n_p95_checked += 1
            # Check if any day in [i-w+1 .. i+w-1] (clamped) is below P95
            lo = max(0, i - w + 1)
            hi = min(n, i + w)
            if np.min(fuel_fac_nf[lo:hi]) <= p95_fuel:
                n_avoidable += 1
    if n_p95_checked > 0:
        print(f"     With +/-{w} day flexibility: {n_avoidable}/{n_p95_checked} "
              f"P95 departures avoidable "
              f"({100 * n_avoidable / n_p95_checked:.0f}%)")


def plot_departure_analysis(results: list[VoyageResult], speed_kn: float,
                            round_trip: bool = True):
    """Generate departure timing analysis plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib not available; skipping departure analysis plots.")
        return

    n = len(results)
    dates = [r.departure for r in results]
    fuel_fac_nf = np.array([r.total_fuel_factory_no_flettner_kg for r in results])
    fuel_opt_fl = np.array([r.total_fuel_optimised_kg for r in results])
    fuel_opt_nf = np.array([r.total_fuel_opt_no_flettner_kg for r in results])
    sav_flettner_kg = fuel_opt_nf - fuel_opt_fl
    sav_total_kg = fuel_fac_nf - fuel_opt_fl
    mean_wind = np.array([r.mean_wind for r in results])
    mean_hs = np.array([r.mean_hs for r in results])
    mean_F_fl = np.array([r.mean_F_flettner_kN for r in results])

    voy_label = "round-trip" if round_trip else "one-way"

    # ---- Figure 1: Fuel time series with scheduling windows ----
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Departure Timing Analysis — {speed_kn:.0f} kn {voy_label}",
                 fontsize=14, fontweight="bold")

    # Panel 1: Fuel per voyage for both cases
    ax = axes[0]
    ax.plot(dates, fuel_fac_nf, "C3-", alpha=0.7, linewidth=0.8,
            label="Factory NF")
    ax.plot(dates, fuel_opt_fl, "C0-", alpha=0.7, linewidth=0.8,
            label="Opt+Flettner")
    # 7-day rolling min for opt+fl (achievable with 7-day flexibility)
    rolling_min = np.array([np.min(fuel_opt_fl[max(0, i - 3):min(n, i + 4)])
                            for i in range(n)])
    ax.plot(dates, rolling_min, "C2--", linewidth=1.2,
            label="Best Opt+Fl in 7-day window")
    ax.set_ylabel("Fuel [kg/voyage]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Flettner saving
    ax = axes[1]
    ax.bar(dates, sav_flettner_kg, width=1.0, color="C0", alpha=0.6,
           label="Flettner saving")
    ax.set_ylabel("Flettner saving [kg]")
    ax.axhline(np.mean(sav_flettner_kg), color="C0", linestyle="--",
               linewidth=1, alpha=0.8, label=f"Mean = {np.mean(sav_flettner_kg):.0f} kg")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Mean wind speed
    ax = axes[2]
    ax.bar(dates, mean_wind, width=1.0, color="C7", alpha=0.6)
    ax.set_ylabel("Mean wind [m/s]")
    ax.axhline(np.mean(mean_wind), color="k", linestyle="--",
               linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Panel 4: Mean Hs
    ax = axes[3]
    ax.bar(dates, mean_hs, width=1.0, color="C4", alpha=0.6)
    ax.set_ylabel("Mean Hs [m]")
    ax.axhline(np.mean(mean_hs), color="k", linestyle="--",
               linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].set_xlabel("Departure date (2024)")

    fig.tight_layout()
    fig.savefig("departure_timing_timeseries.png", dpi=150, bbox_inches="tight")
    print(f"\n  Saved: departure_timing_timeseries.png")

    # ---- Figure 2: Scatter matrix — correlations ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Weather vs Fuel & Savings Correlations — "
                 f"{speed_kn:.0f} kn {voy_label}",
                 fontsize=14, fontweight="bold")

    # Scatter 1: Wind vs Flettner saving
    ax = axes[0, 0]
    sc = ax.scatter(mean_wind, sav_flettner_kg, c=mean_hs, cmap="viridis",
                    s=12, alpha=0.6)
    ax.set_xlabel("Mean wind speed [m/s]")
    ax.set_ylabel("Flettner saving [kg/voyage]")
    ax.set_title(f"r = {np.corrcoef(mean_wind, sav_flettner_kg)[0, 1]:+.3f}")
    plt.colorbar(sc, ax=ax, label="Hs [m]", shrink=0.8)
    ax.grid(True, alpha=0.3)

    # Scatter 2: Wind vs total fuel (Factory NF)
    ax = axes[0, 1]
    sc = ax.scatter(mean_wind, fuel_fac_nf, c=mean_hs, cmap="viridis",
                    s=12, alpha=0.6)
    ax.set_xlabel("Mean wind speed [m/s]")
    ax.set_ylabel("Factory NF fuel [kg/voyage]")
    ax.set_title(f"r = {np.corrcoef(mean_wind, fuel_fac_nf)[0, 1]:+.3f}")
    plt.colorbar(sc, ax=ax, label="Hs [m]", shrink=0.8)
    ax.grid(True, alpha=0.3)

    # Scatter 3: Hs vs Opt+Fl fuel (does Flettner reduce weather sensitivity?)
    ax = axes[1, 0]
    ax.scatter(mean_hs, fuel_fac_nf, s=12, alpha=0.5, color="C3",
               label="Factory NF")
    ax.scatter(mean_hs, fuel_opt_fl, s=12, alpha=0.5, color="C0",
               label="Opt+Flettner")
    ax.set_xlabel("Mean Hs [m]")
    ax.set_ylabel("Fuel [kg/voyage]")
    ax.set_title("Weather sensitivity: Factory vs Opt+Fl")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Scatter 4: Mean Flettner thrust vs Flettner fuel saving
    ax = axes[1, 1]
    sc = ax.scatter(mean_F_fl, sav_flettner_kg, c=mean_wind, cmap="plasma",
                    s=12, alpha=0.6)
    ax.set_xlabel("Mean Flettner thrust [kN]")
    ax.set_ylabel("Flettner fuel saving [kg/voyage]")
    ax.set_title(f"r = {np.corrcoef(mean_F_fl, sav_flettner_kg)[0, 1]:+.3f}")
    plt.colorbar(sc, ax=ax, label="Wind [m/s]", shrink=0.8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("departure_timing_correlations.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: departure_timing_correlations.png")

    # ---- Figure 3: Scheduling window value ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Value of Scheduling Flexibility — {speed_kn:.0f} kn {voy_label}",
                 fontsize=14, fontweight="bold")

    windows = [1, 2, 3, 5, 7, 10, 14, 21, 28]
    # Left: mean best fuel within window (Factory NF and Opt+Fl)
    mean_best_fac = []
    mean_best_opt = []
    for w in windows:
        b_fac = [np.min(fuel_fac_nf[max(0, i - w // 2):min(n, i + (w + 1) // 2)])
                 for i in range(n)]
        b_opt = [np.min(fuel_opt_fl[max(0, i - w // 2):min(n, i + (w + 1) // 2)])
                 for i in range(n)]
        mean_best_fac.append(np.mean(b_fac))
        mean_best_opt.append(np.mean(b_opt))

    ax = axes[0]
    ax.plot(windows, mean_best_fac, "C3o-", label="Factory NF (best in window)")
    ax.plot(windows, mean_best_opt, "C0o-", label="Opt+Fl (best in window)")
    ax.axhline(np.mean(fuel_fac_nf), color="C3", linestyle="--", alpha=0.5,
               label=f"Factory NF mean = {np.mean(fuel_fac_nf):.0f}")
    ax.axhline(np.mean(fuel_opt_fl), color="C0", linestyle="--", alpha=0.5,
               label=f"Opt+Fl mean = {np.mean(fuel_opt_fl):.0f}")
    ax.set_xlabel("Scheduling window [days]")
    ax.set_ylabel("Mean achievable fuel [kg/voyage]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: saving % vs baseline (Factory NF mean) for different windows
    ax = axes[1]
    sav_pct_fac = [100 * (1 - v / np.mean(fuel_fac_nf)) for v in mean_best_fac]
    sav_pct_opt = [100 * (1 - v / np.mean(fuel_fac_nf)) for v in mean_best_opt]
    sav_pct_opt_base = 100 * (1 - np.mean(fuel_opt_fl) / np.mean(fuel_fac_nf))
    ax.plot(windows, sav_pct_fac, "C3o-",
            label="Timing only (Factory NF)")
    ax.plot(windows, sav_pct_opt, "C0o-",
            label="Timing + Optimiser + Flettner")
    ax.axhline(sav_pct_opt_base, color="C0", linestyle="--", alpha=0.5,
               label=f"Opt+Fl no timing = {sav_pct_opt_base:.1f}%")
    ax.axhline(0, color="C3", linestyle="--", alpha=0.5,
               label="Factory NF no timing = 0%")
    ax.set_xlabel("Scheduling window [days]")
    ax.set_ylabel("Saving vs Factory NF mean [%]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("departure_timing_windows.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: departure_timing_windows.png")

    plt.close("all")


# ---------------------------------------------------------------------------
# Scheduling analysis: 2D (departure time, speed) optimization
# ---------------------------------------------------------------------------

def run_scheduling_analysis(
    speeds: list[float],
    year: int = 2024,
    data_dir: Path | None = None,
    pdstrip_path: str | None = None,
    flettner_enabled: bool = True,
    round_trip: bool = True,
    hull_ks_m: float = 0.0,
    blade_ks_m: float = 0.0,
    sg_load_kw: float = 0.0,
    sg_factory_allowance_kw: float = 0.0,
    sg_freq_min: float = 0.0,
    sg_freq_max: float = 0.0,
) -> dict[float, list[VoyageResult]]:
    """Run annual comparisons at multiple speeds, return per-speed results.

    Returns
    -------
    dict mapping speed_kn -> list[VoyageResult] (sorted by departure date).
    """
    if pdstrip_path is None:
        pdstrip_path = PDSTRIP_DAT

    all_results: dict[float, list[VoyageResult]] = {}
    for i, spd in enumerate(speeds):
        print(f"\n{'#' * 78}")
        print(f"# SCHEDULING ANALYSIS: {spd:.1f} kn  ({i + 1}/{len(speeds)})")
        print(f"{'#' * 78}\n")
        results = run_annual_comparison(
            year=year, speed_kn=spd, data_dir=data_dir,
            pdstrip_path=pdstrip_path, flettner_enabled=flettner_enabled,
            verbose=False, round_trip=round_trip,
            hull_ks_m=hull_ks_m, blade_ks_m=blade_ks_m,
            sg_load_kw=sg_load_kw,
            sg_factory_allowance_kw=sg_factory_allowance_kw,
            sg_freq_min=sg_freq_min, sg_freq_max=sg_freq_max,
        )
        if results:
            all_results[spd] = results
    return all_results


def print_scheduling_analysis(
    all_results: dict[float, list[VoyageResult]],
    idle_pct: float = 15.0,
    fuel_price_eur_per_t: float = 650.0,
    round_trip: bool = True,
    total_windows_days: tuple[float, ...] = (4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 14.0),
):
    """Analyse 2D (departure time, speed) scheduling optimization.

    For a given total scheduling window W (from earliest possible departure
    to latest acceptable arrival), the operator chooses both WHEN to depart
    and at WHAT SPEED for each individual voyage.

    The key insight is that all candidate voyages cover the same distance
    (509 nm one-way, 1018 nm round-trip), so fuel per voyage is a fair
    comparison metric.  The annualization uses a FIXED commercial schedule
    (same number of voyages/year regardless of speed — the speed flexibility
    is used to optimize individual voyages, not to change throughput).

    Parameters
    ----------
    all_results : dict[float, list[VoyageResult]]
        Per-speed annual results from run_scheduling_analysis().
    total_windows_days : tuple[float, ...]
        Total scheduling windows [days] from earliest departure to latest
        arrival.
    """
    if not all_results:
        print("No results for scheduling analysis.")
        return

    speeds = sorted(all_results.keys())
    voy_label = "round-trip" if round_trip else "one-way"

    # Reference speed (design speed) is the one closest to 10 kn
    ref_speed = min(speeds, key=lambda s: abs(s - 10.0))
    ref_results = all_results[ref_speed]

    # Build day index from ref speed results
    day_indices = {}
    for r in ref_results:
        day_indices[r.departure.date()] = len(day_indices)
    n_common = len(day_indices)

    # Build 2D fuel matrices: fuel[speed_idx, day_idx]
    fuel_fac_nf = np.full((len(speeds), n_common), np.nan)
    fuel_opt_fl = np.full((len(speeds), n_common), np.nan)
    fuel_opt_nf = np.full((len(speeds), n_common), np.nan)
    transit_hours = {}

    for si, spd in enumerate(speeds):
        results = all_results[spd]
        if results:
            transit_hours[spd] = results[0].total_hours
        for r in results:
            di = day_indices.get(r.departure.date())
            if di is not None:
                fuel_fac_nf[si, di] = r.total_fuel_factory_no_flettner_kg
                fuel_opt_fl[si, di] = r.total_fuel_optimised_kg
                fuel_opt_nf[si, di] = r.total_fuel_opt_no_flettner_kg

    # Annualization: use ref speed's voyage rate for ALL comparisons
    # (commercial schedule is fixed, speed flex is per-voyage operational)
    sailing_h_yr = 365.25 * 24.0 * (1.0 - idle_pct / 100.0)
    ref_t_h = transit_hours[ref_speed]
    ref_vpy = sailing_h_yr / ref_t_h
    ref_si = speeds.index(ref_speed)

    print("\n" + "=" * 78)
    print("SCHEDULING OPTIMIZATION: 2D (DEPARTURE TIME x SPEED)")
    print("=" * 78)

    # ---- 1. Overview: per-voyage fuel at each speed ----
    print(f"\n  1. PER-VOYAGE FUEL AT EACH SPEED (annual mean, no scheduling)")
    print(f"     Transit time and voyages/year shown for reference;\n"
          f"     annualization uses ref schedule ({ref_vpy:.0f} voy/yr @ "
          f"{ref_speed:.0f} kn).\n")

    print(f"     {'Speed':>6s}  {'Transit':>8s}  "
          f"{'FacNF':>8s}  {'Opt+Fl':>8s}  {'Save':>6s}  "
          f"{'Fl sav':>7s}  {'P/R sav':>8s}")
    print(f"     {'[kn]':>6s}  {'[hours]':>8s}  "
          f"{'kg/voy':>8s}  {'kg/voy':>8s}  {'%':>6s}  "
          f"{'kg/voy':>7s}  {'kg/voy':>8s}")
    print(f"     {'-' * 6}  {'-' * 8}  "
          f"{'-' * 8}  {'-' * 8}  {'-' * 6}  "
          f"{'-' * 7}  {'-' * 8}")

    for si, spd in enumerate(speeds):
        t_h = transit_hours.get(spd, 0)
        m_fac = np.nanmean(fuel_fac_nf[si])
        m_opt = np.nanmean(fuel_opt_fl[si])
        m_opt_nf = np.nanmean(fuel_opt_nf[si])
        sav = 100 * (m_fac - m_opt) / m_fac if m_fac > 0 else 0
        fl_sav = m_opt_nf - m_opt
        pr_sav = m_fac - m_opt_nf
        marker = " <-- ref" if spd == ref_speed else ""
        print(f"     {spd:>6.1f}  {t_h:>8.1f}  "
              f"{m_fac:>8.0f}  {m_opt:>8.0f}  {sav:>5.1f}%  "
              f"{fl_sav:>7.0f}  {pr_sav:>8.0f}{marker}")

    # ---- 2. Per-voyage scheduling optimization ----
    # For each window W, for each day in the year, find the best
    # (departure_within_flex, speed) combination that minimizes fuel.
    # This is the key table: per-voyage fuel with 2D optimization.
    print(f"\n  2. PER-VOYAGE 2D SCHEDULING OPTIMIZATION")
    print(f"     For each departure opportunity, the operator chooses the")
    print(f"     best (departure day within flex, speed) to minimize fuel.")
    print(f"     All speeds must fit within the total window.\n")

    ref_mean_fac = np.nanmean(fuel_fac_nf[ref_si])
    ref_mean_opt = np.nanmean(fuel_opt_fl[ref_si])

    print(f"     {'Window':>8s}  {'Feasible speeds':>26s}  "
          f"{'Mean':>8s}  {'Mean':>8s}  {'vs ref':>6s}  "
          f"{'Ann sav':>8s}  {'EUR/yr':>10s}")
    print(f"     {'[days]':>8s}  {'':>26s}  "
          f"{'FacNF':>8s}  {'Opt+Fl':>8s}  {'FacNF':>6s}  "
          f"{'[t/yr]':>8s}  {'':>10s}")
    print(f"     {'-' * 8}  {'-' * 26}  "
          f"{'-' * 8}  {'-' * 8}  {'-' * 6}  "
          f"{'-' * 8}  {'-' * 10}")

    # Baseline (no scheduling, ref speed)
    base_sav_pct = 100 * (ref_mean_fac - ref_mean_opt) / ref_mean_fac
    print(f"     {'(none)':>8s}  {ref_speed:>25.0f}*  "
          f"{ref_mean_fac:>8.0f}  {ref_mean_opt:>8.0f}  "
          f"{'---':>6s}  {'---':>8s}  {'---':>10s}")

    scheduling_results = []

    for W in total_windows_days:
        # Determine which speeds are feasible for this window
        feasible = []
        for si, spd in enumerate(speeds):
            t_h = transit_hours.get(spd, 0)
            if t_h > 0 and t_h / 24.0 <= W:
                dep_flex = int(W - t_h / 24.0)
                feasible.append((si, spd, dep_flex))

        if not feasible:
            continue

        # For each day, find the best fuel across all feasible
        # (speed, departure_within_flex) combinations
        best_fac_per_day = []
        best_opt_per_day = []
        for d in range(n_common):
            day_best_fac = float("inf")
            day_best_opt = float("inf")
            for si, spd, dep_flex in feasible:
                lo = d
                hi = min(n_common, d + dep_flex + 1)
                # Factory NF: best in window at this speed
                c_fac = fuel_fac_nf[si, lo:hi]
                v_fac = c_fac[~np.isnan(c_fac)]
                if len(v_fac) > 0:
                    day_best_fac = min(day_best_fac, np.min(v_fac))
                # Opt+Fl: best in window at this speed
                c_opt = fuel_opt_fl[si, lo:hi]
                v_opt = c_opt[~np.isnan(c_opt)]
                if len(v_opt) > 0:
                    day_best_opt = min(day_best_opt, np.min(v_opt))
            if day_best_fac < float("inf"):
                best_fac_per_day.append(day_best_fac)
            if day_best_opt < float("inf"):
                best_opt_per_day.append(day_best_opt)

        if not best_opt_per_day:
            continue

        mean_best_fac = np.mean(best_fac_per_day)
        mean_best_opt = np.mean(best_opt_per_day)

        # Saving vs ref baseline (Factory NF at ref speed, no scheduling)
        sav_fac_pct = 100 * (ref_mean_fac - mean_best_fac) / ref_mean_fac
        sav_opt_pct = 100 * (ref_mean_fac - mean_best_opt) / ref_mean_fac
        ann_sav_t = (ref_mean_fac - mean_best_opt) * ref_vpy / 1000
        ann_eur = ann_sav_t * fuel_price_eur_per_t

        spd_str = ", ".join(
            f"{spd:.0f}({dep_flex}d)" for _, spd, dep_flex in feasible)

        print(f"     {W:>8.1f}  {spd_str:>26s}  "
              f"{mean_best_fac:>8.0f}  {mean_best_opt:>8.0f}  "
              f"{sav_opt_pct:>5.1f}%  "
              f"{ann_sav_t:>8.1f}  {ann_eur:>10,.0f}")

        scheduling_results.append({
            "window": W, "feasible": feasible,
            "mean_fac": mean_best_fac, "mean_opt": mean_best_opt,
            "sav_pct": sav_opt_pct, "ann_sav_t": ann_sav_t,
        })

    print(f"\n     Speeds shown as: speed(departure_flexibility_days)")
    print(f"     Saving % vs Factory NF @ {ref_speed:.0f} kn, no scheduling "
          f"({ref_mean_fac:.0f} kg/voy)")

    # ---- 3. Decomposition: what comes from where ----
    print(f"\n  3. SAVING DECOMPOSITION (per voyage, vs Factory NF @ "
          f"{ref_speed:.0f} kn baseline)")
    print(f"     Showing mean kg/voyage saved by each lever.\n")

    print(f"     {'Window':>8s}  {'Speed':>8s}  {'Timing':>8s}  "
          f"{'Pitch/RPM':>10s}  {'Flettner':>9s}  {'Total':>8s}  "
          f"{'Total':>6s}")
    print(f"     {'[days]':>8s}  {'choice':>8s}  {'choice':>8s}  "
          f"{'optim':>10s}  {'':>9s}  {'kg/voy':>8s}  "
          f"{'%':>6s}")
    print(f"     {'-' * 8}  {'-' * 8}  {'-' * 8}  "
          f"{'-' * 10}  {'-' * 9}  {'-' * 8}  {'-' * 6}")

    for sr in scheduling_results:
        W = sr["window"]
        feasible = sr["feasible"]

        # (a) Speed-only saving (best constant speed, no timing, Factory NF)
        best_speed_only_fac = float("inf")
        for si, spd, _ in feasible:
            m = np.nanmean(fuel_fac_nf[si])
            if m < best_speed_only_fac:
                best_speed_only_fac = m
        sav_speed = ref_mean_fac - best_speed_only_fac

        # (b) Timing-only saving (ref speed, timing within flex, Factory NF)
        ref_dep_flex = max(0, int(W - ref_t_h / 24.0))
        timing_fac = []
        for d in range(n_common):
            lo, hi = d, min(n_common, d + ref_dep_flex + 1)
            c = fuel_fac_nf[ref_si, lo:hi]
            v = c[~np.isnan(c)]
            if len(v) > 0:
                timing_fac.append(np.min(v))
        sav_timing = ref_mean_fac - np.mean(timing_fac) if timing_fac else 0

        # Total saving from 2D scheduling (FacNF basis)
        sav_2d_fac = ref_mean_fac - sr["mean_fac"]
        # Additional from pitch/RPM + Flettner
        sav_opt = sr["mean_fac"] - sr["mean_opt"]
        # Total
        sav_total = ref_mean_fac - sr["mean_opt"]
        pct_total = 100 * sav_total / ref_mean_fac if ref_mean_fac > 0 else 0

        # Decompose: speed + timing + interaction = sav_2d_fac
        # Then opt+fl on top
        # For the decomposition we separate timing from speed more carefully:
        # sav_2d_fac = sav_speed + sav_timing + interaction
        sav_interaction = sav_2d_fac - sav_speed - sav_timing

        print(f"     {W:>8.1f}  {sav_speed:>8.0f}  "
              f"{sav_timing + sav_interaction:>8.0f}  "
              f"{sr['mean_fac'] - np.nanmean(fuel_opt_nf[ref_si]):>10.0f}  "  # approximate
              f"{sav_opt - (sr['mean_fac'] - np.nanmean(fuel_opt_nf[ref_si])):>9.0f}  "  # approximate
              f"{sav_total:>8.0f}  {pct_total:>5.1f}%")

    # ---- 4. Simpler table: what the operator actually sees ----
    print(f"\n  4. OPERATOR DECISION TABLE")
    print(f"     For a given scheduling window, what is the annual benefit?\n")
    print(f"     {'Window':>8s}  {'No opt':>10s}  {'Opt+Fl':>10s}  "
          f"{'2D sched':>10s}  {'Full 2D':>10s}")
    print(f"     {'[days]':>8s}  {'FacNF':>10s}  {'@{:.0f}kn'.format(ref_speed):>10s}  "
          f"{'FacNF':>10s}  {'Opt+Fl':>10s}")
    print(f"     {'':>8s}  {'t/yr':>10s}  {'t/yr':>10s}  "
          f"{'t/yr':>10s}  {'t/yr':>10s}")
    print(f"     {'-' * 8}  {'-' * 10}  {'-' * 10}  "
          f"{'-' * 10}  {'-' * 10}")

    # Column 1: No optimization at all (Factory NF @ ref speed)
    col1 = ref_mean_fac * ref_vpy / 1000

    # Column 2: Opt+Fl at ref speed, no scheduling
    col2 = ref_mean_opt * ref_vpy / 1000

    print(f"     {'(none)':>8s}  {col1:>10.1f}  {col2:>10.1f}  "
          f"{'---':>10s}  {'---':>10s}")

    for sr in scheduling_results:
        W = sr["window"]
        # Column 3: 2D scheduling, Factory NF
        col3 = sr["mean_fac"] * ref_vpy / 1000
        # Column 4: 2D scheduling, Opt+Fl
        col4 = sr["mean_opt"] * ref_vpy / 1000
        print(f"     {W:>8.1f}  {col1:>10.1f}  {col2:>10.1f}  "
              f"{col3:>10.1f}  {col4:>10.1f}")

    print(f"\n     All in tonnes/year (annualized at {ref_vpy:.0f} voy/yr, "
          f"{idle_pct:.0f}% idle).")
    print(f"     'No opt' = Factory NF @ {ref_speed:.0f} kn, no scheduling "
          f"= {col1:.1f} t/yr")
    print(f"     'Opt+Fl' = Optimiser + Flettner @ {ref_speed:.0f} kn, "
          f"no scheduling = {col2:.1f} t/yr")
    print(f"     '2D sched FacNF' = best (day, speed) per voyage, Factory NF")
    print(f"     'Full 2D Opt+Fl' = best (day, speed) per voyage, Opt+Fl")

    return (scheduling_results, all_results, speeds, fuel_fac_nf,
            fuel_opt_fl, fuel_opt_nf, transit_hours, n_common,
            ref_speed, ref_si, ref_vpy, ref_mean_fac, ref_mean_opt)


def plot_scheduling_analysis(
    all_results: dict[float, list[VoyageResult]],
    scheduling_data: tuple,
    idle_pct: float = 15.0,
    fuel_price_eur_per_t: float = 650.0,
    round_trip: bool = True,
):
    """Generate scheduling analysis plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping scheduling plots.")
        return

    (sched_results, _, speeds, fuel_fac_nf, fuel_opt_fl,
     fuel_opt_nf, transit_hours, n_common,
     ref_speed, ref_si, ref_vpy, ref_mean_fac, ref_mean_opt) = scheduling_data

    sailing_h_yr = 365.25 * 24.0 * (1.0 - idle_pct / 100.0)
    voy_label = "round-trip" if round_trip else "one-way"

    # ---- Figure 1: Per-voyage fuel at each speed ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"Speed & Scheduling — Per-Voyage Fuel — {voy_label}",
                 fontsize=14, fontweight="bold")

    mean_fac = [np.nanmean(fuel_fac_nf[si]) for si in range(len(speeds))]
    mean_opt = [np.nanmean(fuel_opt_fl[si]) for si in range(len(speeds))]

    ax = axes[0]
    ax.plot(speeds, mean_fac, "C3o-", label="Factory NF", markersize=8)
    ax.plot(speeds, mean_opt, "C0o-", label="Opt+Flettner", markersize=8)
    ax.set_xlabel("Transit speed [kn]")
    ax.set_ylabel("Fuel per voyage [kg]")
    ax.set_title("Mean fuel per voyage vs speed")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Figure 1b: Best per-voyage fuel with 2D scheduling ----
    windows = [3, 4, 5, 6, 7, 8, 10, 14]
    ax = axes[1]

    # For each window, compute the mean best per-voyage Opt+Fl fuel
    # (best across all feasible speeds and departure days within flex)
    mean_best_opt_by_window = []
    mean_best_fac_by_window = []
    valid_windows = []
    for W in windows:
        feasible = [(si, spd, max(0, int(W - transit_hours.get(spd, 999) / 24.0)))
                     for si, spd in enumerate(speeds)
                     if transit_hours.get(spd, 999) / 24.0 <= W]
        if not feasible:
            continue
        best_opt = []
        best_fac = []
        for d in range(n_common):
            d_best_opt = float("inf")
            d_best_fac = float("inf")
            for si, spd, dep_flex in feasible:
                lo, hi = d, min(n_common, d + dep_flex + 1)
                c_opt = fuel_opt_fl[si, lo:hi]
                v_opt = c_opt[~np.isnan(c_opt)]
                if len(v_opt) > 0:
                    d_best_opt = min(d_best_opt, np.min(v_opt))
                c_fac = fuel_fac_nf[si, lo:hi]
                v_fac = c_fac[~np.isnan(c_fac)]
                if len(v_fac) > 0:
                    d_best_fac = min(d_best_fac, np.min(v_fac))
            if d_best_opt < float("inf"):
                best_opt.append(d_best_opt)
            if d_best_fac < float("inf"):
                best_fac.append(d_best_fac)
        mean_best_opt_by_window.append(np.mean(best_opt))
        mean_best_fac_by_window.append(np.mean(best_fac))
        valid_windows.append(W)

    ax.plot(valid_windows, mean_best_fac_by_window, "C3o-",
            label="Best 2D (Factory NF)", markersize=7)
    ax.plot(valid_windows, mean_best_opt_by_window, "C0o-",
            label="Best 2D (Opt+Fl)", markersize=7)
    ax.axhline(ref_mean_fac, color="C3", linestyle="--", alpha=0.5,
               label=f"Fac NF @ {ref_speed:.0f} kn = {ref_mean_fac:.0f}")
    ax.axhline(ref_mean_opt, color="C0", linestyle="--", alpha=0.5,
               label=f"Opt+Fl @ {ref_speed:.0f} kn = {ref_mean_opt:.0f}")
    ax.set_xlabel("Total scheduling window [days]")
    ax.set_ylabel("Mean fuel per voyage [kg]")
    ax.set_title("Per-voyage fuel with 2D scheduling")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("scheduling_speed_window.png", dpi=150, bbox_inches="tight")
    print(f"\n  Saved: scheduling_speed_window.png")

    # ---- Figure 2: Heatmap — per-voyage fuel (speed x window) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"Per-Voyage Opt+Fl Fuel — Speed x Window — {voy_label}",
                 fontsize=14, fontweight="bold")

    windows_fine = [3, 4, 5, 6, 7, 8, 10, 12, 14]

    # Left: per-speed, best-in-window per-voyage fuel
    heat_voy = np.full((len(speeds), len(windows_fine)), np.nan)
    for si, spd in enumerate(speeds):
        t_h = transit_hours.get(spd, 0)
        if t_h <= 0:
            continue
        for wi, W in enumerate(windows_fine):
            if W < t_h / 24.0:
                continue
            dep_flex = max(0, int(W - t_h / 24.0))
            bl = []
            for d in range(n_common):
                lo, hi = d, min(n_common, d + dep_flex + 1)
                c = fuel_opt_fl[si, lo:hi]
                v = c[~np.isnan(c)]
                if len(v) > 0:
                    bl.append(np.min(v))
            if bl:
                heat_voy[si, wi] = np.mean(bl)

    ax = axes[0]
    im = ax.imshow(heat_voy / 1000, aspect="auto", origin="lower",
                   cmap="RdYlGn_r",
                   extent=[windows_fine[0] - 0.5, windows_fine[-1] + 0.5,
                           speeds[0] - 0.25, speeds[-1] + 0.25])
    ax.set_xlabel("Total scheduling window [days]")
    ax.set_ylabel("Transit speed [kn]")
    ax.set_title("Mean Opt+Fl fuel per voyage [tonnes]")
    plt.colorbar(im, ax=ax, label="Fuel [t/voy]", shrink=0.8)

    for si, spd in enumerate(speeds):
        t_h = transit_hours.get(spd, 0)
        for wi, W in enumerate(windows_fine):
            val = heat_voy[si, wi]
            if not np.isnan(val):
                ax.text(W, spd, f"{val / 1000:.1f}", ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="white" if val > np.nanmedian(heat_voy[~np.isnan(heat_voy)]) else "black")
            elif t_h > 0 and W < t_h / 24.0:
                ax.text(W, spd, "X", ha="center", va="center",
                        fontsize=10, color="gray", alpha=0.5)
    ax.set_xticks(windows_fine)
    ax.set_yticks(speeds)

    # Right: saving % vs ref baseline
    heat_sav = np.full_like(heat_voy, np.nan)
    for si in range(len(speeds)):
        for wi in range(len(windows_fine)):
            if not np.isnan(heat_voy[si, wi]):
                heat_sav[si, wi] = 100 * (ref_mean_fac - heat_voy[si, wi]) / ref_mean_fac

    ax = axes[1]
    im2 = ax.imshow(heat_sav, aspect="auto", origin="lower",
                    cmap="RdYlGn",
                    extent=[windows_fine[0] - 0.5, windows_fine[-1] + 0.5,
                            speeds[0] - 0.25, speeds[-1] + 0.25])
    ax.set_xlabel("Total scheduling window [days]")
    ax.set_ylabel("Transit speed [kn]")
    ax.set_title(f"Saving vs Factory NF @ {ref_speed:.0f} kn [%]")
    plt.colorbar(im2, ax=ax, label="Saving [%]", shrink=0.8)

    for si, spd in enumerate(speeds):
        t_h = transit_hours.get(spd, 0)
        for wi, W in enumerate(windows_fine):
            val = heat_sav[si, wi]
            if not np.isnan(val):
                ax.text(W, spd, f"{val:.0f}%", ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="white" if val < np.nanmedian(heat_sav[~np.isnan(heat_sav)]) else "black")
            elif t_h > 0 and W < t_h / 24.0:
                ax.text(W, spd, "X", ha="center", va="center",
                        fontsize=10, color="gray", alpha=0.5)
    ax.set_xticks(windows_fine)
    ax.set_yticks(speeds)

    fig.tight_layout()
    fig.savefig("scheduling_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: scheduling_heatmap.png")

    plt.close("all")


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
        mean_rotor_power_kW=float(np.mean([r.mean_rotor_power_kW for r in results])),
        hull_ks_um=results[0].hull_ks_um,
        blade_ks_um=results[0].blade_ks_um,
        R_roughness_kN=results[0].R_roughness_kN,
        blade_fuel_factor=results[0].blade_fuel_factor,
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
    hull_ks_m: float = 0.0,
    blade_ks_m: float = 0.0,
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
            hull_ks_m=hull_ks_m,
            blade_ks_m=blade_ks_m,
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
    if sweep[0].hull_ks_um > 0 or sweep[0].blade_ks_um > 0:
        print(f"  Hull roughness: {sweep[0].hull_ks_um:.0f} µm, "
              f"blade roughness: {sweep[0].blade_ks_um:.0f} µm")

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
            f"{'R_aw [kN]':>9s}  {'R_wind [kN]':>11s}  {'F_flett [kN]':>12s}  "
            f"{'P_rotor [kW]':>12s}")
    print(hdr3)
    print(f"  {'-' * 6}  {'-' * 7}  {'-' * 10}  {'-' * 9}  {'-' * 11}  {'-' * 12}  {'-' * 12}")
    for s in sweep:
        print(f"  {s.speed_kn:5.1f}kn  "
              f"{s.mean_hs:>7.2f}  "
              f"{s.mean_wind:>10.1f}  "
              f"{s.mean_R_aw_kN:>9.1f}  "
              f"{s.mean_R_wind_kN:>11.1f}  "
              f"{s.mean_F_flettner_kN:>12.1f}  "
              f"{s.mean_rotor_power_kW:>12.1f}")

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
# Roughness sweep summary
# ============================================================

def _print_roughness_sweep_summary(
    sweep_data: list[tuple],
    speed_kn: float,
    idle_pct: float = 15.0,
    fuel_price_eur_per_t: float = 650.0,
    round_trip: bool = True,
):
    """Print a summary table comparing results across fouling ages.

    Parameters
    ----------
    sweep_data : list of (age_years, hull_ks_m, blade_ks_m, results)
    """
    if not sweep_data:
        print("No roughness sweep results to summarise.")
        return

    voy_label = "RT" if round_trip else "OW"
    r0 = sweep_data[0][3][0]  # first age, first voyage
    transit_h = r0.total_hours
    sailing_hours_year = 365.25 * 24.0 * (1.0 - idle_pct / 100.0)
    vpyr = sailing_hours_year / transit_h

    print(f"\n{'=' * 100}")
    print(f"ROUGHNESS SWEEP SUMMARY  ({speed_kn:.0f} kn, {voy_label}, "
          f"{idle_pct:.0f}% idle -> {vpyr:.0f} voy/yr)")
    print(f"{'=' * 100}")

    fp = fuel_price_eur_per_t

    print(f"\n  {'Age':>4s}  {'Hull ks':>8s}  {'Blade ks':>9s}  "
          f"{'ΔR_hull':>8s}  {'BladeFac':>8s}  "
          f"{'Fac_NF':>9s}  {'Opt+Fl':>9s}  {'Save':>6s}  "
          f"{'P/R %':>6s}  {'Fl %':>6s}  "
          f"{'€ saving':>10s}")
    print(f"  {'[yr]':>4s}  {'[µm]':>8s}  {'[µm]':>9s}  "
          f"{'[kN]':>8s}  {'[-]':>8s}  "
          f"{'[t/yr]':>9s}  {'[t/yr]':>9s}  {'[%]':>6s}  "
          f"{'':>6s}  {'':>6s}  "
          f"{'[€/yr]':>10s}")
    print(f"  {'-' * 4}  {'-' * 8}  {'-' * 9}  "
          f"{'-' * 8}  {'-' * 8}  "
          f"{'-' * 9}  {'-' * 9}  {'-' * 6}  "
          f"{'-' * 6}  {'-' * 6}  "
          f"{'-' * 10}")

    for age, h_ks, b_ks, results in sweep_data:
        fac_nf = np.mean([r.total_fuel_factory_no_flettner_kg for r in results])
        opt_fl = np.mean([r.total_fuel_optimised_kg for r in results])
        sav_pr = np.mean([r.saving_pitch_rpm_kg for r in results])
        sav_fl = np.mean([r.saving_flettner_kg for r in results])
        sav_tot = sav_pr + sav_fl

        ann_fac_nf = fac_nf * vpyr / 1000.0
        ann_opt_fl = opt_fl * vpyr / 1000.0
        ann_sav_tot = sav_tot * vpyr / 1000.0

        pct_tot = 100.0 * sav_tot / fac_nf if fac_nf > 0 else 0.0
        pct_pr = 100.0 * sav_pr / fac_nf if fac_nf > 0 else 0.0
        pct_fl = 100.0 * sav_fl / fac_nf if fac_nf > 0 else 0.0

        r_rough = results[0].R_roughness_kN
        b_fac = results[0].blade_fuel_factor
        cost_sav = ann_sav_tot * fp

        print(f"  {age:4.1f}  {h_ks * 1e6:8.0f}  {b_ks * 1e6:9.0f}  "
              f"{r_rough:8.1f}  {b_fac:8.3f}  "
              f"{ann_fac_nf:9.1f}  {ann_opt_fl:9.1f}  {pct_tot:6.1f}  "
              f"{pct_pr:6.1f}  {pct_fl:6.1f}  "
              f"{cost_sav:10,.0f}")

    # Compare clean vs. dirtiest
    if len(sweep_data) >= 2:
        _, _, _, r_clean = sweep_data[0]
        _, _, _, r_dirty = sweep_data[-1]
        clean_fac_nf = np.mean([r.total_fuel_factory_no_flettner_kg for r in r_clean])
        dirty_fac_nf = np.mean([r.total_fuel_factory_no_flettner_kg for r in r_dirty])
        fuel_increase_pct = 100.0 * (dirty_fac_nf - clean_fac_nf) / clean_fac_nf if clean_fac_nf > 0 else 0.0
        print(f"\n  Fuel increase from {sweep_data[0][0]:.0f} yr to "
              f"{sweep_data[-1][0]:.0f} yr fouling: "
              f"{fuel_increase_pct:+.1f}% (factory no-Flettner baseline)")

    print()


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
    parser.add_argument("--hull-roughness", type=float, default=0.0,
                        metavar="KS_UM",
                        help="Hull roughness ks [µm]. 0=clean (default). "
                             "Typical: 30=new AF, 150=2yr Nordic, 500=5yr.")
    parser.add_argument("--blade-roughness", type=float, default=0.0,
                        metavar="KS_UM",
                        help="Propeller blade roughness ks [µm]. 0=clean "
                             "(default). Typical: 10=polished, 70=3yr Nordic.")
    parser.add_argument("--fouling-years", type=float, default=None,
                        metavar="T",
                        help="Time since last cleaning [years]. Uses "
                             "FOULING_LOW for hull and BLADE_FOULING for "
                             "propeller. Overrides --hull/blade-roughness.")
    parser.add_argument("--roughness-sweep", nargs="+", type=float,
                        default=None, metavar="T",
                        help="Run annual comparison at multiple fouling ages "
                             "[years], e.g. --roughness-sweep 0 1 2 3 4 5")
    parser.add_argument("--departure-analysis", action="store_true",
                        help="Analyse the value of departure timing "
                             "flexibility (implies --plot)")
    parser.add_argument("--scheduling-analysis", nargs="+", type=float,
                        default=None, metavar="KN",
                        help="Run 2D (departure time x speed) scheduling "
                             "optimization. Specify candidate speeds, e.g. "
                             "--scheduling-analysis 8 9 10 11 12")

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else NORA3_DATA_DIR
    do_round_trip = not args.one_way

    # Resolve roughness parameters
    hull_ks_m = args.hull_roughness * 1e-6  # µm -> m
    blade_ks_m = args.blade_roughness * 1e-6
    if args.fouling_years is not None:
        hull_ks_m = FOULING_LOW.ks_m_at(args.fouling_years)
        blade_ks_m = BLADE_FOULING.ks_m_at(args.fouling_years)

    if args.download:
        download_nora3_for_route(args.year, ROUTE_ROTTERDAM_GOTHENBURG,
                                 data_dir)
        return

    if args.roughness_sweep:
        ages = sorted(set(args.roughness_sweep))
        print(f"\n{'#' * 78}")
        print(f"# ROUGHNESS SWEEP: fouling ages = "
              f"{', '.join(f'{a:.1f}' for a in ages)} years")
        print(f"{'#' * 78}\n")
        sweep_summaries = []
        for i, age in enumerate(ages):
            h_ks = FOULING_LOW.ks_m_at(age)
            b_ks = BLADE_FOULING.ks_m_at(age)
            print(f"\n{'=' * 78}")
            print(f"  Fouling age: {age:.1f} yr  "
                  f"hull ks={h_ks * 1e6:.0f} µm, blade ks={b_ks * 1e6:.0f} µm"
                  f"  ({i + 1}/{len(ages)})")
            print(f"{'=' * 78}\n")
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
                hull_ks_m=h_ks,
                blade_ks_m=b_ks,
            )
            if results:
                sweep_summaries.append((age, h_ks, b_ks, results))
        _print_roughness_sweep_summary(sweep_summaries, args.speed,
                                       idle_pct=args.idle_pct,
                                       fuel_price_eur_per_t=args.fuel_price,
                                       round_trip=do_round_trip)
        return

    if args.scheduling_analysis:
        sched_speeds = sorted(set(args.scheduling_analysis))
        print(f"\n{'#' * 78}")
        print(f"# SCHEDULING ANALYSIS: speeds = "
              f"{', '.join(f'{s:.0f}' for s in sched_speeds)} kn")
        print(f"{'#' * 78}\n")
        all_results = run_scheduling_analysis(
            speeds=sched_speeds,
            year=args.year,
            data_dir=data_dir,
            pdstrip_path=args.pdstrip,
            flettner_enabled=not args.no_flettner,
            round_trip=do_round_trip,
            hull_ks_m=hull_ks_m,
            blade_ks_m=blade_ks_m,
            sg_load_kw=args.sg_load,
            sg_factory_allowance_kw=args.sg_factory_allowance,
            sg_freq_min=args.sg_freq_min,
            sg_freq_max=args.sg_freq_max,
        )
        sched_data = print_scheduling_analysis(
            all_results, idle_pct=args.idle_pct,
            fuel_price_eur_per_t=args.fuel_price,
            round_trip=do_round_trip,
        )
        if sched_data:
            plot_scheduling_analysis(
                all_results, sched_data,
                idle_pct=args.idle_pct,
                fuel_price_eur_per_t=args.fuel_price,
                round_trip=do_round_trip,
            )
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
            hull_ks_m=hull_ks_m,
            blade_ks_m=blade_ks_m,
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
            hull_ks_m=hull_ks_m,
            blade_ks_m=blade_ks_m,
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
            hull_ks_m=hull_ks_m,
            blade_ks_m=blade_ks_m,
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
        hull_ks_m=hull_ks_m,
        blade_ks_m=blade_ks_m,
    )

    print_summary(results, args.speed, idle_pct=args.idle_pct,
                  fuel_price_eur_per_t=args.fuel_price,
                  round_trip=do_round_trip)

    if args.departure_analysis:
        print_departure_analysis(results, args.speed, idle_pct=args.idle_pct,
                                 fuel_price_eur_per_t=args.fuel_price,
                                 round_trip=do_round_trip)
        plot_departure_analysis(results, args.speed,
                                round_trip=do_round_trip)

    if args.plot:
        plot_results(results)


if __name__ == "__main__":
    main()
