"""
DP controller for track-following ploughing operations.

Implements a PID controller with feedforward for track-following mode,
similar to the brucon DP controller architecture but simplified for
the ploughing simulation.

The controller is tuned using natural frequency (omega_n) and damping
ratio (zeta), which is the standard approach for DP systems. This
ensures gains are properly scaled to the vessel mass/inertia.

PID gains from 2nd order system:
    Kp = omega_n^2 * M
    Kd = 2 * zeta * omega_n * M
    Ki = (omega_n / 10)^2 * M   (integral ~10x slower than proportional)

Typical DP natural periods:
    - Surge: 60-150 s (omega_n ~ 0.04-0.1 rad/s)
    - Sway:  60-150 s
    - Yaw:   30-60 s  (yaw responds faster)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class DPControllerConfig:
    """DP controller tuning parameters."""
    # Natural periods [s] and damping ratios [-]
    # These define the closed-loop response bandwidth
    T_surge: float = 120.0      # Surge natural period [s]
    zeta_surge: float = 0.8     # Surge damping ratio [-]
    T_sway: float = 100.0       # Sway natural period [s]
    zeta_sway: float = 0.8      # Sway damping ratio [-]
    T_yaw: float = 60.0         # Yaw natural period [s]
    zeta_yaw: float = 0.9       # Yaw damping ratio [-]

    # Integral time constant factor (Ti = factor * T_natural)
    integral_factor: float = 10.0   # Integral 10x slower than proportional

    # Speed controller (force = Kp * mass * speed_error)
    Kp_speed: float = 0.02         # Speed proportional gain (non-dim, * mass)
    Ki_speed: float = 0.001        # Speed integral gain (non-dim, * mass)

    # Feedforward
    tow_force_feedforward: float = 1.0   # 0-1, fraction of estimated tow force to feedforward
    wind_feedforward: float = 0.8

    # Anti-windup (fraction of max thrust)
    integral_limit_fraction: float = 0.5

    # Lowpass filter on position error
    error_filter_tau: float = 3.0    # [s] time constant

    # Gain scheduling: reduce gains at higher speeds
    speed_gain_reduction: float = 0.3


class DPController:
    """
    DP controller for track-following and station-keeping.

    Uses a PID structure tuned via natural frequency/damping with:
    - Feedforward of estimated tow wire force
    - Anti-windup on integral action
    - Gain scheduling with speed
    - Thrust rate limiting
    """

    def __init__(self, config: DPControllerConfig, vessel_mass: dict, dt: float = 0.1):
        """
        Parameters:
            config: Controller tuning parameters
            vessel_mass: dict with keys 'surge', 'sway', 'yaw' [kg, kg, kg*m^2]
            dt: Control time step [s]
        """
        self.cfg = config
        self.dt = dt
        self.mass = vessel_mass

        # Compute PID gains from natural frequency and damping
        # Surge
        omega_surge = 2 * np.pi / config.T_surge
        self.Kp_surge = omega_surge**2 * vessel_mass['surge']
        self.Kd_surge = 2 * config.zeta_surge * omega_surge * vessel_mass['surge']
        self.Ki_surge = (omega_surge / config.integral_factor)**2 * vessel_mass['surge']

        # Sway
        omega_sway = 2 * np.pi / config.T_sway
        self.Kp_sway = omega_sway**2 * vessel_mass['sway']
        self.Kd_sway = 2 * config.zeta_sway * omega_sway * vessel_mass['sway']
        self.Ki_sway = (omega_sway / config.integral_factor)**2 * vessel_mass['sway']

        # Yaw
        omega_yaw = 2 * np.pi / config.T_yaw
        self.Kp_yaw = omega_yaw**2 * vessel_mass['yaw']
        self.Kd_yaw = 2 * config.zeta_yaw * omega_yaw * vessel_mass['yaw']
        self.Ki_yaw = (omega_yaw / config.integral_factor)**2 * vessel_mass['yaw']

        # Speed controller gains (scaled by mass)
        self.Kp_speed_gain = config.Kp_speed * vessel_mass['surge']
        self.Ki_speed_gain = config.Ki_speed * vessel_mass['surge']

        # Integral states
        self.int_surge = 0.0
        self.int_sway = 0.0
        self.int_yaw = 0.0
        self.int_speed = 0.0

        # Filtered errors
        self.err_sway_filt = 0.0
        self.err_yaw_filt = 0.0

        # Previous errors (for derivative)
        self.prev_err_sway = 0.0
        self.prev_err_yaw = 0.0

        # Mode
        self.mode = 'track_following'

        # Track definition
        self.waypoints = []
        self.current_wp_index = 0
        self.desired_speed = 0.5
        self.desired_heading = 0.0

        # Output
        self.tau_command = np.array([0.0, 0.0, 0.0])

        # Thrust limits
        self.max_thrust = np.array([900e3, 500e3, 20e6])
        self.thrust_rate_limit = np.array([100e3, 80e3, 5e6])
        self.prev_tau = np.array([0.0, 0.0, 0.0])

        # Anti-windup limits
        self.integral_limit = self.max_thrust * config.integral_limit_fraction

    def set_track(self, waypoints: list, speed: float):
        """Set the track to follow."""
        self.waypoints = [np.array(wp) for wp in waypoints]
        self.desired_speed = speed
        self.current_wp_index = 0
        if len(waypoints) >= 2:
            dx = waypoints[1][0] - waypoints[0][0]
            dy = waypoints[1][1] - waypoints[0][1]
            self.desired_heading = np.arctan2(dy, dx)

    def set_thrust_limits(self, max_surge: float, max_sway: float, max_yaw: float):
        """Set maximum thrust/moment capacity."""
        self.max_thrust = np.array([max_surge, max_sway, max_yaw])
        self.integral_limit = self.max_thrust * self.cfg.integral_limit_fraction

    def compute_track_errors(self, x: float, y: float, psi: float,
                              u: float, v: float) -> dict:
        """
        Compute along-track and cross-track errors using LOS guidance.
        """
        if len(self.waypoints) < 2:
            return {'along_track': 0.0, 'cross_track': 0.0,
                    'heading_error': 0.0, 'speed_error': 0.0,
                    'los_heading': psi, 'track_angle': psi}

        idx = min(self.current_wp_index, len(self.waypoints) - 2)
        wp0 = self.waypoints[idx]
        wp1 = self.waypoints[idx + 1]

        track_vec = wp1 - wp0
        track_length = np.linalg.norm(track_vec)
        if track_length < 1e-3:
            return {'along_track': 0.0, 'cross_track': 0.0,
                    'heading_error': 0.0, 'speed_error': 0.0,
                    'los_heading': psi, 'track_angle': psi}

        track_dir = track_vec / track_length
        track_angle = np.arctan2(track_dir[1], track_dir[0])

        # Position error from wp0
        pos_err = np.array([x, y]) - wp0

        # Along-track and cross-track (signed)
        along_track = np.dot(pos_err, track_dir)
        cross_track = -pos_err[0] * np.sin(track_angle) + pos_err[1] * np.cos(track_angle)

        # Advance waypoint if passed
        if along_track > track_length and idx < len(self.waypoints) - 2:
            self.current_wp_index += 1

        # LOS guidance heading
        lookahead = max(50.0, 3.0 * abs(cross_track))
        los_heading = track_angle - np.arctan2(cross_track, lookahead)

        # Speed along track
        vel_ned = np.array([u * np.cos(psi) - v * np.sin(psi),
                            u * np.sin(psi) + v * np.cos(psi)])
        speed_along_track = np.dot(vel_ned, track_dir)

        # Heading error
        heading_error = los_heading - psi
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        return {
            'along_track': along_track,
            'cross_track': cross_track,
            'heading_error': heading_error,
            'speed_error': self.desired_speed - speed_along_track,
            'los_heading': los_heading,
            'track_angle': track_angle,
            'distance_to_wp': track_length - along_track,
        }

    def step(self, x: float, y: float, psi: float,
             u: float, v: float, r: float,
             tow_force_body: np.ndarray = None,
             wind_force_body: np.ndarray = None) -> np.ndarray:
        """
        Compute one control step.

        Parameters:
            x, y: Position in NED [m]
            psi: Heading [rad]
            u, v: Body velocities [m/s]
            r: Yaw rate [rad/s]
            tow_force_body: Estimated tow force in body frame [Fx, Fy, Mz]
            wind_force_body: Estimated wind force in body frame [Fx, Fy, Mz]

        Returns:
            tau_command: Thrust command [Fx_surge, Fy_sway, Mz_yaw] in body frame
        """
        dt = self.dt

        if tow_force_body is None:
            tow_force_body = np.zeros(3)
        if wind_force_body is None:
            wind_force_body = np.zeros(3)

        # --- Track errors ---
        errors = self.compute_track_errors(x, y, psi, u, v)
        e_ct = errors['cross_track']
        e_speed = errors['speed_error']
        e_psi = errors['heading_error']

        # --- Filter errors (1st order lowpass) ---
        alpha = dt / (self.cfg.error_filter_tau + dt)
        self.err_sway_filt = (1 - alpha) * self.err_sway_filt + alpha * e_ct
        self.err_yaw_filt = (1 - alpha) * self.err_yaw_filt + alpha * e_psi

        # --- PID: Surge (speed control along track) ---
        self.int_speed += e_speed * dt
        self.int_speed = np.clip(self.int_speed,
                                  -self.integral_limit[0] / max(self.Ki_speed_gain, 1),
                                  self.integral_limit[0] / max(self.Ki_speed_gain, 1))

        tau_surge_pid = (self.Kp_speed_gain * e_speed +
                         self.Ki_speed_gain * self.int_speed)

        # --- PID: Sway (cross-track correction) ---
        # Error derivative
        de_sway = (self.err_sway_filt - self.prev_err_sway) / dt if dt > 0 else 0.0
        self.int_sway += self.err_sway_filt * dt
        self.int_sway = np.clip(self.int_sway,
                                 -self.integral_limit[1] / max(self.Ki_sway, 1),
                                 self.integral_limit[1] / max(self.Ki_sway, 1))

        # Gain scheduling: reduce sway PD gains at higher speeds (integral unchanged)
        speed_factor = 1.0 - self.cfg.speed_gain_reduction * min(abs(u) / 1.0, 1.0)

        # Cross-track correction in the track-perpendicular direction.
        # Convention: positive cross_track = vessel is to starboard of track.
        # We need a force that pushes the vessel back TOWARD the track,
        # i.e., to port if CT > 0 → negative CT correction force.
        track_angle = errors.get('track_angle', psi)
        ct_correction = -speed_factor * (
            self.Kp_sway * self.err_sway_filt +
            self.Ki_sway * self.int_sway +
            self.Kd_sway * de_sway
        )

        # The CT correction is a force perpendicular to the track.
        # Perpendicular direction (to starboard of track) in NED:
        #   perp_x = -sin(track_angle), perp_y = cos(track_angle)
        F_ct_ned_x = ct_correction * (-np.sin(track_angle))
        F_ct_ned_y = ct_correction * (np.cos(track_angle))

        # Rotate NED correction to body frame
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        tau_surge_body_from_ct = c_psi * F_ct_ned_x + s_psi * F_ct_ned_y
        tau_sway_body_from_ct = -s_psi * F_ct_ned_x + c_psi * F_ct_ned_y

        # --- PID: Yaw (heading control) ---
        de_yaw = (self.err_yaw_filt - self.prev_err_yaw) / dt if dt > 0 else 0.0
        self.int_yaw += self.err_yaw_filt * dt
        self.int_yaw = np.clip(self.int_yaw,
                                -self.integral_limit[2] / max(self.Ki_yaw, 1),
                                self.integral_limit[2] / max(self.Ki_yaw, 1))

        tau_yaw_pid = (self.Kp_yaw * self.err_yaw_filt +
                       self.Ki_yaw * self.int_yaw +
                       self.Kd_yaw * de_yaw)

        # --- Feedforward: compensate known external forces ---
        tau_ff = np.zeros(3)
        tau_ff -= self.cfg.tow_force_feedforward * tow_force_body
        tau_ff -= self.cfg.wind_feedforward * wind_force_body

        # --- Total command ---
        tau = np.array([
            tau_surge_pid + tau_surge_body_from_ct + tau_ff[0],
            tau_sway_body_from_ct + tau_ff[1],
            tau_yaw_pid + tau_ff[2],
        ])

        # --- Thrust saturation ---
        tau = np.clip(tau, -self.max_thrust, self.max_thrust)

        # --- Rate limiting ---
        dtau = tau - self.prev_tau
        max_dtau = self.thrust_rate_limit * dt
        dtau = np.clip(dtau, -max_dtau, max_dtau)
        tau = self.prev_tau + dtau

        # Update state
        self.prev_err_sway = self.err_sway_filt
        self.prev_err_yaw = self.err_yaw_filt
        self.prev_tau = tau.copy()
        self.tau_command = tau.copy()

        return tau

    def reset_integrators(self):
        """Reset all integral states."""
        self.int_surge = 0.0
        self.int_sway = 0.0
        self.int_yaw = 0.0
        self.int_speed = 0.0
