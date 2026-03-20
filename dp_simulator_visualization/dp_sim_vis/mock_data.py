"""Mock data generator — produces synthetic vessel/platform/wave data
for testing the visualizer without a running dp_simulator.

Generates realistic-looking motions: vessel station-keeping with wave-
frequency oscillations, platform with slow drift and pitch resonance,
and a JONSWAP wave field.
"""

import time
import numpy as np

from .udp_receiver import SimulatorState, WaveSpectrumParams, GangwayStateData
from .wave_model import WaveSpectrum, WaveElevation, default_frequencies, default_directions


class MockDataGenerator:
    """Generates synthetic SimulatorState at each call to step().

    The mock scenario:
    - Vessel at station-keeping near origin, heading ~45 deg
    - Floating platform 200m North, heading ~0 deg
    - Wind sea: Hs=2.0m, Tp=8s, direction=180 deg (from south)
    - Swell: Hs=1.0m, Tp=12s, direction=225 deg (from south-west)
    """

    def __init__(self, random_seed: int = 42):
        self.state = SimulatorState()
        self.start_time = time.time()
        self.seed = random_seed

        # Wave parameters
        self.state.wave = WaveSpectrumParams(
            significant_wave_height=2.0,
            peak_period=8.0,
            direction_deg=180.0,
            spreading_factor=2.0,
        )
        self.state.swell = WaveSpectrumParams(
            significant_wave_height=1.0,
            peak_period=12.0,
            direction_deg=225.0,
            spreading_factor=7.0,
        )
        self.state.random_seed = random_seed

        # Wind
        self.state.wind_speed = 10.0  # m/s
        self.state.wind_direction = 180.0  # from south
        self.state.platform_wind_speed = 10.0  # m/s (same as wind_speed for mock)

        # Platform position (fixed installation location)
        self._platform_install_north = 200.0
        self._platform_install_east = 0.0
        self._platform_install_heading = 0.0

        # Vessel nominal position
        self._vessel_nominal_north = 0.0
        self._vessel_nominal_east = 0.0
        self._vessel_nominal_heading = 45.0  # heading NE

        # Build wave model for computing vessel/platform WF motions
        self._freqs = default_frequencies(n=30, w_min=0.25, w_max=2.0)
        self._dirs = default_directions(n=36)
        self.state.frequencies = self._freqs.tolist()
        self.state.directions = self._dirs.tolist()

        self._wave_elevation = WaveElevation(self._freqs, self._dirs, random_seed)
        wave_spec = WaveSpectrum(
            hs=self.state.wave.significant_wave_height,
            tp=self.state.wave.peak_period,
            direction_deg=self.state.wave.direction_deg,
            spreading_factor=self.state.wave.spreading_factor,
        )
        swell_spec = WaveSpectrum(
            hs=self.state.swell.significant_wave_height,
            tp=self.state.swell.peak_period,
            direction_deg=self.state.swell.direction_deg,
            spreading_factor=self.state.swell.spreading_factor,
        )
        self._wave_elevation.add_spectrum(wave_spec)
        self._wave_elevation.add_spectrum(swell_spec)

        self.state.wave_params_updated = True

    @property
    def wave_elevation(self) -> WaveElevation:
        """The wave elevation model (for the ocean surface to use directly)."""
        return self._wave_elevation

    def step(self) -> SimulatorState:
        """Generate the next state snapshot."""
        t = time.time() - self.start_time
        self.state.sim_time = t

        # ── Vessel motions ────────────────────────────────────────────
        # LF: slow drift (DP station-keeping with some residual drift)
        lf_surge = 1.5 * np.sin(0.02 * t)  # slow surge drift ~5 min period
        lf_sway = 1.0 * np.sin(0.015 * t + 0.5)  # slow sway drift
        lf_yaw = 3.0 * np.sin(0.01 * t + 1.0)  # slow heading variation +/- 3 deg

        # WF: wave-frequency oscillations (approximate, not RAO-based)
        wf_surge = 0.5 * np.sin(0.8 * t + 0.3)
        wf_sway = 0.4 * np.sin(0.75 * t + 1.2)
        wf_heave = 0.6 * np.sin(0.85 * t + 0.7)
        wf_roll = 2.0 * np.sin(0.9 * t + 0.4)  # deg
        wf_pitch = 1.5 * np.sin(0.8 * t + 1.1)  # deg
        wf_yaw = 0.5 * np.sin(0.7 * t + 2.0)  # deg

        heading_rad = np.deg2rad(self._vessel_nominal_heading + lf_yaw)
        # Rotate body-frame LF+WF motions to NED
        surge_total = lf_surge + wf_surge
        sway_total = lf_sway + wf_sway
        self.state.vessel_north = (
            self._vessel_nominal_north
            + surge_total * np.cos(heading_rad)
            - sway_total * np.sin(heading_rad)
        )
        self.state.vessel_east = (
            self._vessel_nominal_east
            + surge_total * np.sin(heading_rad)
            + sway_total * np.cos(heading_rad)
        )
        self.state.vessel_heading = self._vessel_nominal_heading + lf_yaw + wf_yaw
        self.state.vessel_roll = wf_roll
        self.state.vessel_pitch = wf_pitch
        self.state.vessel_heave = wf_heave

        # ── Platform motions ──────────────────────────────────────────
        # LF: moored with slow drift
        plat_lf_surge = 3.0 * np.sin(0.005 * t)  # very slow surge
        plat_lf_sway = 2.0 * np.sin(0.004 * t + 0.8)
        plat_lf_yaw = 5.0 * np.sin(0.003 * t + 1.5)  # slow yaw drift

        # WF heave
        plat_wf_heave = 0.8 * np.sin(0.7 * t + 0.5)

        # Dynamic pitch — spar pitch resonance ~23s period
        plat_pitch = 3.0 * np.sin(2 * np.pi / 23.0 * t + 0.2)
        plat_roll = 2.0 * np.sin(2 * np.pi / 25.0 * t + 1.0)

        plat_heading_rad = np.deg2rad(self._platform_install_heading + plat_lf_yaw)
        self.state.platform_north = (
            self._platform_install_north
            + plat_lf_surge * np.cos(plat_heading_rad)
        )
        self.state.platform_east = (
            self._platform_install_east
            + plat_lf_sway
        )
        self.state.platform_heading = self._platform_install_heading + plat_lf_yaw
        self.state.platform_roll = plat_roll
        self.state.platform_pitch = plat_pitch
        self.state.platform_heave = plat_wf_heave

        # ── Gangway motions ────────────────────────────────────────────
        # Simulate a gangway that starts parked (slew=180, boom=0, height=0)
        # and after 10s moves to a working position (slew~270=port, boom~-5deg = up,
        # height~15m, length~25m), then oscillates slightly.
        # Boom angle convention: negative = up, positive = down (body Z-down frame).
        if t < 10.0:
            # Parked
            gw_slew = 180.0
            gw_boom = 0.0
            gw_height = 0.0
            gw_length = 18.0
            gw_state = 0  # Parked
        else:
            # Smooth transition over ~20 seconds
            ramp = min(1.0, (t - 10.0) / 20.0)
            gw_slew = 180.0 + ramp * 90.0  # 180 → 270 (aft to port)
            gw_boom = ramp * -5.0           # 0 → -5 deg (up)
            gw_height = ramp * 15.0         # 0 → 15 m
            gw_length = 18.0 + ramp * 7.0   # 18 → 25 m
            gw_state = 2 if ramp < 1.0 else 4  # Moving → Connected
            # Add slight oscillation when connected
            if ramp >= 1.0:
                gw_slew += 0.5 * np.sin(0.3 * t)
                gw_boom += 0.3 * np.sin(0.4 * t + 0.5)

        self.state.gangway_state = GangwayStateData(
            total_length=gw_length,
            height=gw_height,
            slewing_angle=gw_slew,
            boom_angle=gw_boom,
            state=gw_state,
        )

        self.state.last_update = time.time()
        return self.state
