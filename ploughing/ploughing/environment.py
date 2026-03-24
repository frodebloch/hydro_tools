"""
Environmental forces model.

Simplified wind, current, and wave drift force models for the
ploughing simulation.

Wave drift forces are the dominant environmental disturbance for
DP vessels in the surge direction. They consist of:
  - Mean wave drift force (steady, proportional to Hs^2)
  - Slowly-varying drift (difference-frequency, 30-200s periods)
  - 1st order wave-frequency oscillation (at peak wave period)

These forces create the characteristic 'noisy' vessel speed signal
seen in real ploughing operational data.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class WaveDriftConfig:
    """Wave drift force configuration.

    A simplified spectral wave drift model for a cable laying vessel.
    The mean drift force scales as Hs^2. The slowly-varying component
    is modelled as filtered noise with power proportional to the mean
    drift, and the 1st-order wave oscillation is a sinusoidal surge
    at the peak wave period.
    """
    Hs: float = 0.0                # Significant wave height [m] (0 = no waves)
    Tp: float = 8.0               # Peak wave period [s]
    wave_direction: float = 180.0 # Direction waves come FROM [deg, met. convention]

    # Mean drift force coefficient [N/m^2] — vessel-specific
    # For a 130m cable layer, typically 20-60 kN in Hs=1.5m head seas
    # C_drift * Hs^2 * B = mean drift force
    C_drift: float = 5000.0       # [N/m^2 per m^2 Hs^2] × beam

    # Slowly-varying drift: COV relative to mean drift, filter tau
    sv_cov: float = 0.8           # COV of slowly-varying drift [-]
    sv_tau: float = 60.0          # Filter time constant [s] (~1 min)

    # 1st order surge oscillation amplitude factor
    # Surge amplitude ≈ factor * Hs [m/s velocity amplitude]
    first_order_surge_factor: float = 0.015  # [m/s per m Hs]

    # Random seed for wave drift (None = random)
    seed: int = None


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

    # Waves
    waves: WaveDriftConfig = field(default_factory=WaveDriftConfig)


class EnvironmentModel:
    """Compute environmental forces on the vessel."""

    def __init__(self, config: EnvironmentConfig, vessel_config):
        self.cfg = config
        self.vessel = vessel_config

        # Wave drift state
        wc = config.waves
        self._wave_rng = np.random.default_rng(wc.seed)
        self._sv_drift_state = 0.0       # Slowly-varying drift (zero-mean)
        self._wave_phase = self._wave_rng.uniform(0, 2 * np.pi)  # Random initial phase
        self._time = 0.0

        # Precompute mean drift force
        beam = vessel_config.beam
        self._mean_drift = wc.C_drift * wc.Hs**2 * beam  # [N]

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
        Convention: headwind (from ahead) gives negative Fx (resistance).
        """
        Vw = self.cfg.wind_speed
        if Vw < 0.1:
            return np.zeros(3)

        rho_air = 1.225
        # Wind direction (FROM) in radians
        wind_from = np.radians(self.cfg.wind_direction)

        # Relative wind angle: angle wind comes FROM relative to bow
        # alpha_w = 0  → headwind (wind from ahead)
        # alpha_w = pi → following wind (wind from astern)
        alpha_w = wind_from - psi
        alpha_w = np.arctan2(np.sin(alpha_w), np.cos(alpha_w))

        # Wind force coefficients (simplified Blendermann)
        # Headwind (alpha=0): Cx negative (resistance), no sway
        # Beam wind (alpha=pi/2): max sway, reduced surge
        # Following wind (alpha=pi): slight positive Cx
        Cx = -0.5 * np.cos(alpha_w)  # headwind → Cx = -0.5 (resistance)
        Cy = 0.7 * np.sin(alpha_w)   # sway: positive for wind from starboard
        Cn = 0.15 * np.sin(2 * alpha_w)  # yaw: quarter wind max

        A_frontal = self.vessel.frontal_wind_area
        A_lateral = self.vessel.lateral_wind_area
        L = self.vessel.lpp

        q = 0.5 * rho_air * Vw**2

        Fx = Cx * q * A_frontal
        Fy = Cy * q * A_lateral
        Mz = Cn * q * A_lateral * L

        return np.array([Fx, Fy, Mz])

    def wave_drift_force_body(self, psi: float, dt: float = 0.1) -> np.ndarray:
        """
        Wave drift force on vessel in body frame [Fx, Fy, Mz].

        Combines:
          - Mean wave drift force (steady, proportional to Hs^2)
          - Slowly-varying drift (difference-frequency envelope, 30-200s)
          - 1st order wave-frequency surge oscillation (~Tp period)

        Parameters:
            psi: Vessel heading [rad]
            dt: Time step [s] for state update

        Returns:
            Wave force vector [Fx_surge, Fy_sway, Mz_yaw] in body frame [N, N, Nm]
        """
        wc = self.cfg.waves
        self._time += dt

        if wc.Hs < 0.05:
            return np.zeros(3)

        # Wave direction (FROM) in radians
        wave_from = np.radians(wc.wave_direction)

        # Relative wave angle: angle waves come FROM relative to bow
        # beta = 0  → head seas (waves from ahead)
        # beta = pi → following seas (waves from astern)
        beta = wave_from - psi
        beta = np.arctan2(np.sin(beta), np.cos(beta))

        # --- Mean drift force ---
        F_mean = self._mean_drift

        # --- Slowly-varying drift (filtered noise) ---
        alpha_sv = dt / (wc.sv_tau + dt)
        noise = self._wave_rng.normal(0, 1)
        self._sv_drift_state = (1 - alpha_sv) * self._sv_drift_state + alpha_sv * noise
        F_sv = F_mean * wc.sv_cov * self._sv_drift_state

        # --- 1st order wave-frequency oscillation ---
        omega_p = 2 * np.pi / wc.Tp
        # Surge velocity oscillation amplitude
        surge_vel_amp = wc.first_order_surge_factor * wc.Hs
        # Force to produce this velocity oscillation (F = M * a = M * omega * v_amp)
        # Approximate with vessel surge added mass
        # vessel.displacement is already mass [kg], add ~5% surge added mass
        M_surge = self.vessel.displacement * 1.05  # rough added mass
        F_wave1 = M_surge * omega_p * surge_vel_amp * np.sin(omega_p * self._time + self._wave_phase)

        # Total drift force magnitude (along wave propagation direction)
        F_total = F_mean + F_sv + F_wave1

        # Project onto body frame
        # Mean drift force acts in wave propagation direction (FROM → TO)
        # For head seas (beta=0, waves from ahead), drift pushes vessel backward:
        #   Fx = -F_total * cos(beta) = -F_total (negative = aft)
        Fx = -F_total * np.cos(beta)
        # Sway: beam seas create lateral drift
        Fy = -F_total * np.sin(beta)
        # Yaw moment (beam seas create yaw, head seas don't)
        Mz = -F_total * 0.1 * self.vessel.lpp * np.sin(2 * beta)

        return np.array([Fx, Fy, Mz])

    def inline_current_speed(self, track_direction: float) -> float:
        """
        Current speed component along the track direction [m/s].
        Positive = current flowing in tow direction.
        """
        Vc = self.current_velocity_ned()
        return Vc[0] * np.cos(track_direction) + Vc[1] * np.sin(track_direction)
