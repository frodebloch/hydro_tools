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
    Waypoint(56.00, 6.50, "Northern North Sea"),
    Waypoint(57.30, 9.50, "Skagerrak entrance"),
    Waypoint(57.60, 11.00, "Skagerrak"),
    Waypoint(57.70, 11.80, "Gothenburg"),
]


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
                 sea_margin: float = 0.15):
        self.engine = engine
        self.prop = prop
        self.gear_ratio = GEAR_RATIO

        self._build_schedule(engine, prop, sea_margin)

    def _build_schedule(self, engine, prop, sea_margin):
        """Build the combinator schedule from the propeller curve.

        For each trial speed:
          1. Compute T_required from hull calm-water resistance + sea margin
          2. Sweep pitch, bisect RPM to deliver T_required at Va
          3. Select the (pitch, RPM) pair where engine power ≈ PropellerCurvePower(engineRPM)
          4. Map the resulting schedule points to lever 0-100%
        """
        min_eng_rpm = engine.min_rpm()
        max_eng_rpm = engine.max_rpm()
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
                P_target = engine.propeller_curve_power(eng_rpm)  # kW
                if P_target <= 0.0:
                    return None
                if P_engine > engine.max_power(eng_rpm):
                    return None

                signed_err = P_engine - P_target
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

        N = len(schedule)
        if N == 0:
            raise RuntimeError("FactoryCombinator: no feasible schedule points")

        # Map to lever: 0% = lowest thrust, 100% = highest thrust
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
        P_eng_kw = P_shaft_kw / SHAFT_EFF
        eta0 = self.prop.eta0(pitch, n, Va)

        if (eng_rpm < self.engine.min_rpm()
                or eng_rpm > self.engine.max_rpm()
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
    T_kN_batch, prop, Va, engine = args
    results = {}
    for T_kN in T_kN_batch:
        T_N = T_kN * 1000.0
        op = find_min_fuel_operating_point(
            prop, Va, T_N, engine,
            gear_ratio=GEAR_RATIO,
            shaft_efficiency=SHAFT_EFF,
            pitch_step=0.01,
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

    print(f"  Pre-computing optimiser for {len(T_values)} thrust points "
          f"({T_min_kN:.0f}-{T_max_kN:.0f} kN, step {T_step_kN} kN) "
          f"using {n_workers} workers ...")

    if n_workers <= 1:
        # Sequential fallback
        cache = {}
        for T_kN in T_values:
            T_N = T_kN * 1000.0
            op = find_min_fuel_operating_point(
                prop, Va, T_N, engine,
                gear_ratio=GEAR_RATIO,
                shaft_efficiency=SHAFT_EFF,
                pitch_step=0.01,
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
        batches.append((batch, prop, Va, engine))

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
    mean_F_flettner_kN: float
    n_hours_both_feasible: int = 0
    n_hours_factory_infeasible: int = 0  # factory can't deliver, optimiser can
    n_hours_total: int = 0
    hourly: list[HourlyResult] = field(default_factory=list)


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

        # Net thrust demand on propeller
        # T_total = T_calm + R_aw/(1-t) - F_flettner/(1-t)
        # The added resistance and Flettner force act on the hull, so they
        # modify the resistance which is then divided by (1-t) to get thrust.
        T_required_kN = T_calm_kN + (R_aw_kN - F_flettner_kN) / (1.0 - t_ded)
        T_required_kN = max(0.0, T_required_kN)  # can't have negative thrust
        T_required_N = T_required_kN * 1000.0

        # Thrust demand without Flettner (for splitting savings)
        T_no_flettner_kN = T_calm_kN + R_aw_kN / (1.0 - t_ded)
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
    """
    (departures, route, data_dir, drift_tf, flettner, factory,
     prop, engine, speed_kn, opt_cache, factory_cache) = args
    weather = WeatherAlongRoute(data_dir)
    results = []
    for departure in departures:
        try:
            vr = evaluate_voyage(
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
) -> list[VoyageResult]:
    """Run the full annual comparison: one voyage per day.

    Returns a list of VoyageResult for each departure day.
    """
    if waypoints is None:
        waypoints = ROUTE_ROTTERDAM_GOTHENBURG
    if data_dir is None:
        data_dir = NORA3_DATA_DIR

    print("=" * 78)
    print(f"ANNUAL VOYAGE COMPARISON: Factory Combinator vs Optimiser")
    print(f"  Year: {year}, Speed: {speed_kn} kn")
    print(f"  Route: {waypoints[0].name} -> {waypoints[-1].name}")
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
    print(f"  Distance: {route.total_distance_nm:.0f} nm")
    print(f"  Transit time: {route.total_time_hours:.1f} hours")
    for i, leg in enumerate(route.legs):
        print(f"    Leg {i + 1}: {leg['from'].name} -> {leg['to'].name}: "
              f"{leg['dist_nm']:.0f} nm, bearing {leg['bearing_deg']:.0f} deg")

    factory = FactoryCombinator(engine, prop)
    print(f"  Factory combinator: {len(factory._combo_lever)} schedule points")
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
    opt_cache = build_optimiser_cache(prop, engine, Va)
    factory_cache = build_factory_cache(factory, Va)

    # --- Run voyages ---
    n_days = _days_in_year(year)
    n_workers = min(os.cpu_count() or 1, 16)
    print(f"\nRunning {n_days} voyages using {n_workers} workers ...\n")

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

def print_summary(results: list[VoyageResult], speed_kn: float):
    """Print summary statistics for the annual comparison."""

    if not results:
        print("No results to summarise.")
        return

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
    mean_F_flett = np.array([r.mean_F_flettner_kN for r in results])

    print("\n" + "=" * 78)
    print("ANNUAL SUMMARY")
    print("=" * 78)

    print(f"\n  Voyages completed: {len(results)}")
    print(f"  Transit speed: {speed_kn:.0f} kn")
    print(f"  Transit time per voyage: {results[0].total_hours:.1f} h")

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
    _row("Factory+Fl fuel [kg/voyage]", fuel_factory, ".0f")
    _row("Factory no-Fl fuel [kg/voyage]", fuel_factory_nf, ".0f")
    _row("Opt no-Flettner [kg/voyage]", fuel_opt_noflettner, ".0f")
    _row("Opt + Flettner [kg/voyage]", fuel_optimised, ".0f")
    _row("Mean Hs [m]", mean_hs)
    _row("Mean wind speed [m/s]", mean_wind)
    _row("Mean added resistance [kN]", mean_R_aw)
    _row("Mean Flettner thrust [kN]", mean_F_flett)

    # Compute split percentages relative to factory-no-Flettner baseline
    pct_pitch_rpm = np.where(fuel_factory_nf > 0,
                             savings_pitch_rpm_kg / fuel_factory_nf * 100.0, 0.0)
    pct_flettner = np.where(fuel_factory_nf > 0,
                            savings_flettner_kg / fuel_factory_nf * 100.0, 0.0)
    _row("  Pitch/RPM saving [%]", pct_pitch_rpm)
    _row("  Flettner saving [%]", pct_flettner)

    # Total annual saving
    total_factory_sum = np.sum(fuel_factory)
    total_factory_nf_sum = np.sum(fuel_factory_nf)
    total_optimised_sum = np.sum(fuel_optimised)
    total_opt_nf_sum = np.sum(fuel_opt_noflettner)
    total_saving_pr = np.sum(savings_pitch_rpm_kg)
    total_saving_fl = np.sum(savings_flettner_kg)
    print(f"\n  Annual total (all voyages, all on same feasible-hours basis):")
    print(f"    Factory no-Flettner:     {total_factory_nf_sum / 1000:.1f} tonnes  <- baseline")
    print(f"    Factory + Flettner:      {total_factory_sum / 1000:.1f} tonnes")
    print(f"    Opt no-Flettner:         {total_opt_nf_sum / 1000:.1f} tonnes")
    print(f"    Opt + Flettner:          {total_optimised_sum / 1000:.1f} tonnes")
    if total_factory_nf_sum > 0:
        total_saving_combined = total_factory_nf_sum - total_optimised_sum
        print(f"\n    Saving (fac_NF - opt_FL):  {total_saving_combined / 1000:.1f} tonnes "
              f"({100 * total_saving_combined / total_factory_nf_sum:.1f}%)")
        print(f"      Pitch/RPM (fac_NF-opt_NF): {total_saving_pr / 1000:.1f} tonnes "
              f"({100 * total_saving_pr / total_factory_nf_sum:.1f}%)")
        print(f"      Flettner (opt_NF-opt_FL):  {total_saving_fl / 1000:.1f} tonnes "
              f"({100 * total_saving_fl / total_factory_nf_sum:.1f}%)")
        # Cross-check
        print(f"      Check: P/R + Fl = {(total_saving_pr + total_saving_fl) / 1000:.1f} tonnes")

    # Seasonal breakdown
    print(f"\n  {'Season':<12s}  {'Voyages':>7s}  {'Total %':>8s}  "
          f"{'P/R %':>6s}  {'Flet %':>7s}  "
          f"{'Mean Hs':>8s}  {'Mean wind':>9s}  {'Mean R_aw':>9s}  "
          f"{'%h infeas':>9s}")
    print(f"  {'-' * 12}  {'-' * 7}  {'-' * 8}  "
          f"{'-' * 6}  {'-' * 7}  "
          f"{'-' * 8}  {'-' * 9}  {'-' * 9}  {'-' * 9}")

    quarters = {"Q1 (Jan-Mar)": (1, 3), "Q2 (Apr-Jun)": (4, 6),
                "Q3 (Jul-Sep)": (7, 9), "Q4 (Oct-Dec)": (10, 12)}
    for name, (m1, m2) in quarters.items():
        q_results = [r for r in results if m1 <= r.departure.month <= m2]
        if q_results:
            q_save = np.mean([r.saving_pct for r in q_results])
            q_fuel_fac_nf = np.array([r.total_fuel_factory_no_flettner_kg for r in q_results])
            q_sav_pr = np.array([r.saving_pitch_rpm_kg for r in q_results])
            q_sav_fl = np.array([r.saving_flettner_kg for r in q_results])
            q_pct_pr = np.mean(q_sav_pr / q_fuel_fac_nf * 100.0) if np.all(q_fuel_fac_nf > 0) else 0.0
            q_pct_fl = np.mean(q_sav_fl / q_fuel_fac_nf * 100.0) if np.all(q_fuel_fac_nf > 0) else 0.0
            q_hs = np.mean([r.mean_hs for r in q_results])
            q_wind = np.mean([r.mean_wind for r in q_results])
            q_raw = np.mean([r.mean_R_aw_kN for r in q_results])
            q_total_h = sum(r.n_hours_total for r in q_results)
            q_infeas = sum(r.n_hours_factory_infeasible for r in q_results)
            q_pct_inf = 100 * q_infeas / q_total_h if q_total_h else 0
            print(f"  {name:<12s}  {len(q_results):>7d}  {q_save:>+7.1f}%  "
                  f"{q_pct_pr:>+5.1f}%  {q_pct_fl:>+6.1f}%  "
                  f"{q_hs:>8.1f}  {q_wind:>9.1f}  {q_raw:>9.1f}  "
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
    ax = axes[0]
    ax.fill_between(dates, 0, sav_pr_pct, alpha=0.6, color="steelblue",
                    label=f"Pitch/RPM ({np.mean(sav_pr_pct):.1f}% mean)")
    ax.fill_between(dates, sav_pr_pct, sav_pr_pct + sav_fl_pct,
                    alpha=0.6, color="coral",
                    label=f"Flettner ({np.mean(sav_fl_pct):.1f}% mean)")
    ax.plot(dates, savings_pct, "k-", linewidth=0.5, alpha=0.6,
            label=f"Total ({np.mean(savings_pct):.1f}% mean)")
    ax.axhline(np.mean(savings_pct), color="k", linestyle="--",
               linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Fuel saving [%]")
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
    sc = ax_sc.scatter(mean_hs, savings_pct, s=15, alpha=0.5,
                       c=mean_wind, cmap="viridis")
    ax_sc.set_xlabel("Mean Hs [m]")
    ax_sc.set_ylabel("Fuel saving [%]")
    ax_sc.set_title("Fuel Saving vs Sea State (colour = wind speed)")
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

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else NORA3_DATA_DIR

    if args.download:
        download_nora3_for_route(args.year, ROUTE_ROTTERDAM_GOTHENBURG,
                                 data_dir)
        return

    results = run_annual_comparison(
        year=args.year,
        speed_kn=args.speed,
        data_dir=data_dir,
        pdstrip_path=args.pdstrip,
        flettner_enabled=not args.no_flettner,
        verbose=not args.quiet,
    )

    print_summary(results, args.speed)

    if args.plot:
        plot_results(results)


if __name__ == "__main__":
    main()
