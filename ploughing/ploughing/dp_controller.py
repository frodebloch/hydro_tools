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

Surge control modes:
    - 'speed': Direct speed PID (Kp_speed, Ki_speed on speed error).
      Optionally includes tension-speed modulation for ploughing.
    - 'position': Position-tracking mode (moving waypoint along track).
      A reference point advances at the commanded speed. The surge PID
      controls along-track position error using the same omega_n/zeta
      formulation as the sway/yaw axes. Speed variation emerges naturally
      from the catenary tension "fighting" the position controller P-term.
      This is physically realistic for DP systems that implement track mode
      using only position control (no dedicated speed controller).
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class DPControllerConfig:
    """DP controller tuning parameters."""
    # Surge control mode: 'speed' or 'position'
    #   'speed':    Direct speed PID on along-track velocity error.
    #   'position': Moving-waypoint position PID. A reference point moves
    #               along the track at the commanded speed. The surge axis
    #               uses a full Kp/Ki/Kd on the along-track position error.
    #               Speed variation emerges naturally from disturbances
    #               (catenary tension, wave drift) fighting the position
    #               controller, without any explicit tension-speed feedback.
    surge_mode: str = 'speed'

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
    # Only used when surge_mode='speed'
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

    # Tension-based speed modulation (ploughing mode)
    # When enabled, the speed setpoint is reduced when tow tension exceeds
    # the nominal level, and increased when tension is below nominal.
    # This mimics the real DP/operator behavior during ploughing.
    # Only used when surge_mode='speed'.
    tension_speed_modulation: bool = False
    tension_nominal: float = 380e3       # Nominal tow tension [N] (~38t)
    tension_speed_gain: float = 0.5e-6   # Speed reduction per N excess tension [m/s / N]
    tension_speed_filter_tau: float = 20.0  # Filter time constant on tension [s]
    tension_speed_max_reduction: float = 0.07  # Max speed reduction [m/s]
    tension_speed_max_increase: float = 0.08   # Max speed increase [m/s]

    # --- High-tension slowdown (safety override) ---
    # Deterministic override that decelerates the vessel when tow tension
    # exceeds operator-set thresholds.  This is a separate safety system
    # from the proportional tension-speed modulation above.
    #
    # Stage 1: tension > T_stage1 → decelerate at decel_rate [m/s^2]
    # Stage 2: tension > T_stage2 → decelerate at 2 × decel_rate
    # E-stop:  tension > T_estop  → immediate zero speed command
    #
    # IMPORTANT: After a slowdown event, the system shall NOT catch up
    # to the average set speed.  The speed setpoint ramps back to the
    # base setpoint at the acceleration rate, and the speed integrator
    # and position reference are reset to prevent any overshoot.
    #
    # Applies to both 'speed' and 'position' surge modes.
    high_tension_slowdown: bool = False
    tension_stage1: float = 600e3        # Stage 1 threshold [N] (~61t)
    tension_stage2: float = 750e3        # Stage 2 threshold [N] (~76t)
    tension_estop: float = 900e3         # Emergency stop threshold [N] (~92t)
    slowdown_decel_rate: float = 0.005   # Operator-set deceleration rate [m/s^2]
    slowdown_accel_rate: float = 0.003   # Rate to ramp speed back up after release [m/s^2]
    slowdown_filter_tau: float = 5.0     # Filter on tension for slowdown decisions [s]
    slowdown_min_speed: float = 0.0      # Minimum speed during slowdown [m/s] (0 = full stop allowed)


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
        # Surge (position-tracking mode)
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

        # Speed controller gains (scaled by mass) — only for surge_mode='speed'
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
        self.err_surge_filt = 0.0    # For position-tracking mode

        # Previous errors (for derivative)
        self.prev_err_sway = 0.0
        self.prev_err_yaw = 0.0
        self.prev_err_surge = 0.0    # For position-tracking mode

        # Mode
        self.mode = 'track_following'

        # Track definition
        self.waypoints = []
        self.current_wp_index = 0
        self.desired_speed = 0.5
        self.desired_heading = 0.0

        # Position-tracking reference state
        # along_track_ref is the reference position that moves at desired_speed
        self._along_track_ref = None       # Initialized on first step
        self._track_origin = None          # wp0 of current segment
        self._track_dir_vec = None         # Unit vector along track

        # Output
        self.tau_command = np.array([0.0, 0.0, 0.0])

        # Thrust limits
        self.max_thrust = np.array([900e3, 500e3, 20e6])
        self.thrust_rate_limit = np.array([100e3, 80e3, 5e6])
        self.prev_tau = np.array([0.0, 0.0, 0.0])

        # Anti-windup limits
        self.integral_limit = self.max_thrust * config.integral_limit_fraction

        # Tension-based speed modulation state
        self._tension_filtered = config.tension_nominal

        # High-tension slowdown state
        # The slowdown system maintains its own speed setpoint that ramps
        # down during high tension and ramps back up when tension clears.
        # This is the effective ceiling on speed — the actual speed command
        # is min(desired_speed, slowdown_speed_setpoint).
        self._slowdown_state = 'NORMAL'     # NORMAL, STAGE1, STAGE2, ESTOP
        self._slowdown_tension_filt = 0.0   # Filtered tension for slowdown decisions
        self._slowdown_speed = config.tension_nominal  # Will be initialized properly
        self._slowdown_speed_initialized = False

        # Control action breakdown (for diagnostics/plotting)
        # Updated each step BEFORE rate limiting and saturation
        self._last_surge_ff = 0.0       # Feedforward component [N]
        self._last_surge_pid = 0.0      # PID/PI feedback component [N]

    def set_track(self, waypoints: list, speed: float):
        """Set the track to follow."""
        self.waypoints = [np.array(wp) for wp in waypoints]
        self.desired_speed = speed
        self.current_wp_index = 0
        if len(waypoints) >= 2:
            dx = waypoints[1][0] - waypoints[0][0]
            dy = waypoints[1][1] - waypoints[0][1]
            self.desired_heading = np.arctan2(dy, dx)

            # Initialize position-tracking reference state
            self._track_origin = np.array(waypoints[0])
            track_vec = np.array(waypoints[1]) - np.array(waypoints[0])
            track_len = np.linalg.norm(track_vec)
            self._track_dir_vec = track_vec / max(track_len, 1e-3)
            # along_track_ref starts at the vessel's initial position (wp0)
            self._along_track_ref = 0.0  # Will be initialized on first step

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
             wind_force_body: np.ndarray = None,
             tow_tension_horizontal: float = None) -> np.ndarray:
        """
        Compute one control step.

        Parameters:
            x, y: Position in NED [m]
            psi: Heading [rad]
            u, v: Body velocities [m/s]
            r: Yaw rate [rad/s]
            tow_force_body: Estimated tow force in body frame [Fx, Fy, Mz]
            wind_force_body: Estimated wind force in body frame [Fx, Fy, Mz]
            tow_tension_horizontal: Measured horizontal tow tension [N] (for speed modulation)

        Returns:
            tau_command: Thrust command [Fx_surge, Fy_sway, Mz_yaw] in body frame
        """
        dt = self.dt

        if tow_force_body is None:
            tow_force_body = np.zeros(3)
        if wind_force_body is None:
            wind_force_body = np.zeros(3)

        # --- Track errors (for cross-track and heading) ---
        errors = self.compute_track_errors(x, y, psi, u, v)
        e_ct = errors['cross_track']
        e_psi = errors['heading_error']
        track_angle = errors.get('track_angle', psi)

        # --- High-tension slowdown (safety override) ---
        # Computes a speed ceiling that ramps down during high tension events.
        # Applied as a cap on the speed setpoint in both control modes.
        slowdown_speed_ceiling = self._update_high_tension_slowdown(
            tow_tension_horizontal, dt)

        # --- Surge control: position-tracking or speed mode ---
        if self.cfg.surge_mode == 'position':
            tau_surge_pid = self._surge_position_control(
                x, y, psi, u, v, errors, dt, slowdown_speed_ceiling)
        else:
            tau_surge_pid = self._surge_speed_control(
                u, v, psi, errors, dt, tow_tension_horizontal,
                slowdown_speed_ceiling)

        # --- Filter errors (1st order lowpass) ---
        alpha = dt / (self.cfg.error_filter_tau + dt)
        self.err_sway_filt = (1 - alpha) * self.err_sway_filt + alpha * e_ct
        self.err_yaw_filt = (1 - alpha) * self.err_yaw_filt + alpha * e_psi

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

        # --- Store control action breakdown (before saturation/rate limiting) ---
        self._last_surge_ff = tau_ff[0]
        self._last_surge_pid = tau_surge_pid + tau_surge_body_from_ct

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

    def _surge_speed_control(self, u, v, psi, errors, dt, tow_tension_horizontal,
                             slowdown_speed_ceiling):
        """
        Surge control via speed PID with optional tension-speed modulation.

        This is the original surge controller: a PI on speed error,
        with optional tension-based setpoint adjustment.

        The slowdown_speed_ceiling from the high-tension safety system is
        applied as a hard cap on the effective speed setpoint.
        """
        # Tension-based speed modulation
        effective_desired_speed = self.desired_speed
        if self.cfg.tension_speed_modulation and tow_tension_horizontal is not None:
            alpha_t = dt / (self.cfg.tension_speed_filter_tau + dt)
            self._tension_filtered = ((1 - alpha_t) * self._tension_filtered +
                                       alpha_t * tow_tension_horizontal)
            tension_excess = self._tension_filtered - self.cfg.tension_nominal
            speed_delta = -self.cfg.tension_speed_gain * tension_excess
            speed_delta = np.clip(speed_delta,
                                   -self.cfg.tension_speed_max_reduction,
                                   self.cfg.tension_speed_max_increase)
            effective_desired_speed = max(0.01, self.desired_speed + speed_delta)

        # Apply high-tension slowdown ceiling (safety override takes priority)
        effective_desired_speed = min(effective_desired_speed, slowdown_speed_ceiling)

        # Speed error using the effective (possibly modulated) setpoint
        # Compute speed along track
        vel_ned = np.array([u * np.cos(psi) - v * np.sin(psi),
                            u * np.sin(psi) + v * np.cos(psi)])
        track_angle = errors.get('track_angle', psi)
        track_dir = np.array([np.cos(track_angle), np.sin(track_angle)])
        speed_along_track = np.dot(vel_ned, track_dir)
        e_speed = effective_desired_speed - speed_along_track

        self.int_speed += e_speed * dt
        self.int_speed = np.clip(self.int_speed,
                                  -self.integral_limit[0] / max(self.Ki_speed_gain, 1),
                                  self.integral_limit[0] / max(self.Ki_speed_gain, 1))

        return self.Kp_speed_gain * e_speed + self.Ki_speed_gain * self.int_speed

    def _surge_position_control(self, x, y, psi, u, v, errors, dt,
                                slowdown_speed_ceiling):
        """
        Surge control via position-tracking PID.

        A reference point moves along the track at the commanded speed.
        The controller applies PID on the along-track position error
        (reference - actual). This produces natural speed variation when
        external forces (catenary tension, wave drift) push the vessel
        away from the reference — the "fighting" between P-term and
        catenary that occurs in real DP track-mode systems.

        The slowdown_speed_ceiling limits the reference advance rate.
        When the reference is slowed or stopped, the position error
        naturally decreases, reducing thrust.

        Gains are omega_n/zeta-based, identical to sway/yaw:
            Kp = omega_n^2 * M_surge         [N/m]
            Ki = (omega_n/10)^2 * M_surge     [N/(m·s)]
            Kd = 2*zeta*omega_n * M_surge     [N/(m/s)]
        """
        # Track geometry
        track_angle = errors.get('track_angle', psi)
        track_dir = np.array([np.cos(track_angle), np.sin(track_angle)])

        # Along-track position of vessel (relative to track origin)
        if self._track_origin is not None:
            pos_from_origin = np.array([x, y]) - self._track_origin
            along_track_actual = np.dot(pos_from_origin, self._track_dir_vec)
        else:
            along_track_actual = errors.get('along_track', 0.0)

        # Initialize reference on first call (start at vessel's current position)
        if self._along_track_ref is None or self._along_track_ref == 0.0:
            self._along_track_ref = along_track_actual

        # Advance reference at commanded speed, capped by slowdown ceiling.
        # During a high-tension slowdown, the reference advances slower (or
        # stops), so the vessel naturally decelerates.  When tension clears,
        # the reference resumes at desired_speed — it does NOT jump ahead
        # to catch up because the integrator was reset on state transition.
        ref_speed = min(self.desired_speed, slowdown_speed_ceiling)
        self._along_track_ref += ref_speed * dt

        # Position error: positive means vessel is BEHIND the reference
        e_pos = self._along_track_ref - along_track_actual

        # Velocity error: reference velocity is the capped speed, not the
        # base desired_speed.  During slowdown the derivative term should
        # not fight the deceleration.
        vel_ned = np.array([u * np.cos(psi) - v * np.sin(psi),
                            u * np.sin(psi) + v * np.cos(psi)])
        speed_along_track = np.dot(vel_ned, track_dir)
        e_vel = ref_speed - speed_along_track

        # Filter the position error
        alpha = dt / (self.cfg.error_filter_tau + dt)
        self.err_surge_filt = (1 - alpha) * self.err_surge_filt + alpha * e_pos

        # Derivative of filtered error
        de_surge = (self.err_surge_filt - self.prev_err_surge) / dt if dt > 0 else 0.0

        # Integral with anti-windup
        self.int_surge += self.err_surge_filt * dt
        self.int_surge = np.clip(self.int_surge,
                                  -self.integral_limit[0] / max(self.Ki_surge, 1),
                                  self.integral_limit[0] / max(self.Ki_surge, 1))

        # PID force in along-track direction
        F_along_track = (self.Kp_surge * self.err_surge_filt +
                         self.Ki_surge * self.int_surge +
                         self.Kd_surge * de_surge)

        # Project along-track force into body frame (surge component)
        # For a vessel heading close to the track direction, this is mostly surge
        angle_diff = track_angle - psi
        tau_surge = F_along_track * np.cos(angle_diff)

        self.prev_err_surge = self.err_surge_filt

        return tau_surge

    def _update_high_tension_slowdown(self, tow_tension_horizontal: float,
                                       dt: float) -> float:
        """
        High-tension slowdown safety system.

        Monitors filtered tow tension against operator-set thresholds and
        returns a speed ceiling that ramps down during high tension events
        and ramps back up when tension clears.

        State machine:
            NORMAL → STAGE1  when T_filt > T_stage1
            STAGE1 → STAGE2  when T_filt > T_stage2
            STAGE2 → ESTOP   when T_filt > T_estop
            any    → NORMAL  when T_filt < T_stage1 (with hysteresis)
            STAGE2 → STAGE1  when T_filt < T_stage2
            ESTOP  → STAGE2  when T_filt < T_estop

        The speed ceiling ramps DOWN at the prescribed deceleration rate
        while in any slowdown state, and ramps UP at the acceleration rate
        when returning to NORMAL.  The ramp-up is capped at the base
        desired_speed — never above it — so the system does NOT catch up.

        Parameters:
            tow_tension_horizontal: Current measured horizontal tension [N]
            dt: Time step [s]

        Returns:
            Speed ceiling [m/s] to be applied as min(desired_speed, ceiling)
        """
        cfg = self.cfg

        if not cfg.high_tension_slowdown:
            return self.desired_speed

        # Initialize on first call
        if not self._slowdown_speed_initialized:
            self._slowdown_speed = self.desired_speed
            self._slowdown_tension_filt = tow_tension_horizontal if tow_tension_horizontal else 0.0
            self._slowdown_speed_initialized = True

        # Filter tension (fast filter — need to catch real peaks, not just trends)
        if tow_tension_horizontal is not None:
            alpha = dt / (cfg.slowdown_filter_tau + dt)
            self._slowdown_tension_filt = ((1 - alpha) * self._slowdown_tension_filt +
                                            alpha * tow_tension_horizontal)

        T = self._slowdown_tension_filt
        prev_state = self._slowdown_state

        # --- State transitions ---
        if self._slowdown_state == 'NORMAL':
            if T >= cfg.tension_estop:
                self._slowdown_state = 'ESTOP'
            elif T >= cfg.tension_stage2:
                self._slowdown_state = 'STAGE2'
            elif T >= cfg.tension_stage1:
                self._slowdown_state = 'STAGE1'

        elif self._slowdown_state == 'STAGE1':
            if T >= cfg.tension_estop:
                self._slowdown_state = 'ESTOP'
            elif T >= cfg.tension_stage2:
                self._slowdown_state = 'STAGE2'
            elif T < cfg.tension_stage1:
                self._slowdown_state = 'NORMAL'

        elif self._slowdown_state == 'STAGE2':
            if T >= cfg.tension_estop:
                self._slowdown_state = 'ESTOP'
            elif T < cfg.tension_stage1:
                self._slowdown_state = 'NORMAL'
            elif T < cfg.tension_stage2:
                self._slowdown_state = 'STAGE1'

        elif self._slowdown_state == 'ESTOP':
            if T < cfg.tension_stage1:
                self._slowdown_state = 'NORMAL'
            elif T < cfg.tension_stage2:
                self._slowdown_state = 'STAGE1'
            elif T < cfg.tension_estop:
                self._slowdown_state = 'STAGE2'

        # --- Speed ramp ---
        if self._slowdown_state == 'ESTOP':
            # Immediate zero speed
            self._slowdown_speed = cfg.slowdown_min_speed

        elif self._slowdown_state == 'STAGE2':
            # Decelerate at 2× rate
            self._slowdown_speed -= 2.0 * cfg.slowdown_decel_rate * dt
            self._slowdown_speed = max(self._slowdown_speed, cfg.slowdown_min_speed)

        elif self._slowdown_state == 'STAGE1':
            # Decelerate at 1× rate
            self._slowdown_speed -= cfg.slowdown_decel_rate * dt
            self._slowdown_speed = max(self._slowdown_speed, cfg.slowdown_min_speed)

        elif self._slowdown_state == 'NORMAL':
            # Ramp back up toward desired_speed (never above it)
            self._slowdown_speed += cfg.slowdown_accel_rate * dt
            self._slowdown_speed = min(self._slowdown_speed, self.desired_speed)

        # --- On state transitions: reset integrators to prevent catch-up ---
        if self._slowdown_state != prev_state:
            if self._slowdown_state != 'NORMAL':
                # Entering or escalating slowdown: reset speed integrator
                # to prevent accumulated error from causing a speed surge
                # when tension drops.
                self.int_speed = 0.0
            elif prev_state != 'NORMAL':
                # Returning to NORMAL from any slowdown state: reset both
                # speed and position integrators so the controller does not
                # try to catch up the distance lost during the slowdown.
                self.int_speed = 0.0
                self.int_surge = 0.0
                self.err_surge_filt = 0.0
                self.prev_err_surge = 0.0

        return self._slowdown_speed

    def reset_integrators(self):
        """Reset all integral states."""
        self.int_surge = 0.0
        self.int_sway = 0.0
        self.int_yaw = 0.0
        self.int_speed = 0.0
        self.err_surge_filt = 0.0
        self.prev_err_surge = 0.0
