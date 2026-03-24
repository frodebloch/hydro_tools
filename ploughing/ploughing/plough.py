"""
Plough force model.

Models the forces on a subsea cable burial plough being towed along the seabed.
The plough experiences:
  - Soil cutting/shearing resistance (primary force, stochastic)
  - Friction between plough skids and seabed
  - Hydrodynamic drag on the plough body
  - Inertia of the plough mass

Soil resistance model (informed by real operational data):
  - Mean resistance: 30-40t horizontal tension for 25t plough
  - Variability: std ~15-20t, range 5-100t
  - Temporal characteristics: rapid fluctuations (10-30s periods) from
    soil heterogeneity, plus occasional spikes from boulders/hard layers
  - The variability is modelled as a filtered random process with
    superimposed Poisson spike events

Events modelled:
  - Normal ploughing in heterogeneous soil (stochastic)
  - Hard soil encounter (sudden increase in resistance)
  - Plough stop (obstruction, resistance exceeds tow capacity)
  - Soft soil / plough lift (tension drops to near zero)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class SoilType(Enum):
    SOFT_CLAY = "soft_clay"
    FIRM_CLAY = "firm_clay"
    LOOSE_SAND = "loose_sand"
    DENSE_SAND = "dense_sand"
    ROCK = "rock"
    MIXED = "mixed"


@dataclass
class SoilProperties:
    """
    Soil mechanical properties.

    Typical seabed conditions for cable ploughing:
      - Soft clay:  Su = 5-15 kPa  (very soft marine sediment)
      - Firm clay:  Su = 15-40 kPa (consolidated seabed)
      - Hard clay:  Su = 40-100 kPa (stiff, may cause plough issues)
      - Sand: friction angle 28-38 deg

    The bearing capacity factor Nc for shallow plough shares is lower
    than for deep foundations (Nc~9). For ploughing at depths of 1-2m,
    Nc ~ 3-5 is more representative (shallow failure mechanism).
    """
    soil_type: SoilType = SoilType.FIRM_CLAY
    undrained_shear_strength: float = 15e3    # Su [Pa] (for clay)
    friction_angle: float = 30.0               # [deg] (for sand)
    submerged_unit_weight: float = 8e3         # gamma' [N/m^3]
    bearing_capacity_factor: float = 4.0       # Nc [-] (shallow plough failure)

    @staticmethod
    def soft_clay():
        """Very soft seabed clay, Su ~ 5 kPa."""
        return SoilProperties(SoilType.SOFT_CLAY, 5e3, 20.0, 5e3, 4.0)

    @staticmethod
    def firm_clay():
        """Typical consolidated seabed clay, Su ~ 15 kPa."""
        return SoilProperties(SoilType.FIRM_CLAY, 15e3, 25.0, 8e3, 4.0)

    @staticmethod
    def hard_clay():
        """Stiff clay, Su ~ 50 kPa. High ploughing resistance."""
        return SoilProperties(SoilType.FIRM_CLAY, 50e3, 30.0, 10e3, 5.0)

    @staticmethod
    def loose_sand():
        """Loose sand, friction angle ~ 30 deg."""
        return SoilProperties(SoilType.LOOSE_SAND, 0.0, 30.0, 9e3, 0.0)

    @staticmethod
    def dense_sand():
        """Dense sand, friction angle ~ 38 deg."""
        return SoilProperties(SoilType.DENSE_SAND, 0.0, 38.0, 10e3, 0.0)

    @staticmethod
    def rock():
        return SoilProperties(SoilType.ROCK, 500e3, 45.0, 15e3, 9.0)


@dataclass
class StochasticSoilConfig:
    """
    Configuration for stochastic soil resistance model.

    The model generates realistic time-varying soil resistance by
    superimposing several processes:

    1. Base resistance: deterministic, from soil mechanics (Nc*Su*w*d)
    2. Slow variation: large-scale soil heterogeneity (period 60-300s)
       Modelled as filtered Gaussian noise with COV_slow
    3. Fast variation: local soil variability (period 10-30s)
       Modelled as filtered Gaussian noise with COV_fast
    4. Spike events: boulders, hard layers, shell beds
       Modelled as Poisson arrivals with random amplitude

    Based on real operational data showing:
      - Mean tension ~30-40t
      - Std ~15-20t (COV ~0.4-0.5)
      - Range 5-100t
      - Rapid fluctuations at 10-30s, larger at 2-5 min
    """
    # Coefficient of variation for slow soil changes (60-300s)
    cov_slow: float = 0.25
    tau_slow: float = 120.0     # Filter time constant [s]

    # Coefficient of variation for fast soil changes (10-30s)
    cov_fast: float = 0.35
    tau_fast: float = 15.0      # Filter time constant [s]

    # Spike events (boulders, hard layers)
    spike_rate: float = 0.005   # Mean spike rate [1/s] (~1 spike per 200s)
    spike_amplitude_mean: float = 2.0   # Mean spike amplitude as multiplier of base
    spike_amplitude_std: float = 0.8    # Std of spike amplitude multiplier
    spike_duration_mean: float = 5.0    # Mean spike duration [s]
    spike_duration_std: float = 3.0     # Std of spike duration [s]

    # Minimum resistance (plough always has some soil contact)
    min_resistance_fraction: float = 0.1  # Minimum as fraction of base

    # Random seed (None for random)
    seed: int = None


@dataclass
class PloughConfig:
    """Plough configuration."""
    mass: float = 25e3              # Plough mass [kg] (25t or 35t)
    width: float = 3.0              # Plough width / share width [m]
    height: float = 2.5             # Plough height [m]
    length: float = 8.0             # Plough body length [m]
    burial_depth: float = 1.5       # Target burial depth [m]
    skid_area: float = 4.0          # Total skid contact area [m^2]
    Cd_plough: float = 2.0          # Drag coefficient for plough body [-]
    frontal_area: float = 6.0       # Frontal area for hydro drag [m^2]
    skid_friction: float = 0.4      # Skid-seabed friction coefficient [-]

    # Tow attachment point on plough (from plough CG)
    tow_attachment_height: float = 1.5  # Height of tow point above seabed [m]

    # Stochastic soil model
    stochastic_soil: StochasticSoilConfig = field(default_factory=StochasticSoilConfig)


class PloughModel:
    """
    Plough force model for cable burial operations.

    Computes the total horizontal resistance force that the plough
    exerts on the tow wire as it is dragged along the seabed.
    Includes stochastic soil variability to match real operational data.
    """

    def __init__(self, config: PloughConfig, soil: SoilProperties):
        self.cfg = config
        self.soil = soil
        self.speed = 0.0
        self.position = np.array([0.0, 0.0])  # NED position on seabed
        self.is_stopped = False
        self._hard_soil_active = False
        self._hard_soil_factor = 1.0

        # Stochastic soil state
        sc = config.stochastic_soil
        self._rng = np.random.default_rng(sc.seed)
        self._soil_slow = 0.0       # Slow variation state (zero-mean)
        self._soil_fast = 0.0       # Fast variation state (zero-mean)
        self._spike_active = False
        self._spike_factor = 1.0
        self._spike_remaining = 0.0  # Time remaining in current spike [s]

        # Enable/disable stochastic model
        self.stochastic_enabled = True

    def base_soil_cutting_force(self) -> float:
        """
        Deterministic base soil cutting/shearing resistance [N].

        For clay: F = Nc * Su * w * d
        For sand: F = 0.5 * Kp * gamma' * d^2 * w
        """
        d = self.cfg.burial_depth
        w = self.cfg.width

        if self.soil.soil_type in (SoilType.SOFT_CLAY, SoilType.FIRM_CLAY):
            F_cut = self.soil.bearing_capacity_factor * self.soil.undrained_shear_strength * w * d
        elif self.soil.soil_type in (SoilType.LOOSE_SAND, SoilType.DENSE_SAND):
            phi_rad = np.radians(self.soil.friction_angle)
            Kp = np.tan(np.pi / 4 + phi_rad / 2)**2
            F_cut = 0.5 * Kp * self.soil.submerged_unit_weight * d**2 * w
        elif self.soil.soil_type == SoilType.ROCK:
            F_cut = 1e6
        else:
            F_clay = self.soil.bearing_capacity_factor * self.soil.undrained_shear_strength * w * d
            phi_rad = np.radians(self.soil.friction_angle)
            Kp = np.tan(np.pi / 4 + phi_rad / 2)**2
            F_sand = 0.5 * Kp * self.soil.submerged_unit_weight * d**2 * w
            F_cut = 0.5 * (F_clay + F_sand)

        return F_cut * self._hard_soil_factor

    def soil_cutting_force(self, speed: float = None, dt: float = 0.1) -> float:
        """
        Soil cutting force with stochastic variability [N].

        The stochastic model produces time-varying resistance that captures
        the real-world soil heterogeneity observed in ploughing operations.
        """
        F_base = self.base_soil_cutting_force()

        if not self.stochastic_enabled:
            return F_base

        sc = self.cfg.stochastic_soil

        # --- Update slow variation (filtered white noise) ---
        alpha_slow = dt / (sc.tau_slow + dt)
        noise_slow = self._rng.normal(0, 1)
        self._soil_slow = (1 - alpha_slow) * self._soil_slow + alpha_slow * noise_slow
        slow_factor = 1.0 + sc.cov_slow * self._soil_slow

        # --- Update fast variation ---
        alpha_fast = dt / (sc.tau_fast + dt)
        noise_fast = self._rng.normal(0, 1)
        self._soil_fast = (1 - alpha_fast) * self._soil_fast + alpha_fast * noise_fast
        fast_factor = 1.0 + sc.cov_fast * self._soil_fast

        # --- Update spike events ---
        if self._spike_active:
            self._spike_remaining -= dt
            if self._spike_remaining <= 0:
                self._spike_active = False
                self._spike_factor = 1.0
        else:
            # Check for new spike (Poisson process)
            if self._rng.random() < sc.spike_rate * dt:
                self._spike_active = True
                self._spike_factor = max(1.0,
                    self._rng.normal(sc.spike_amplitude_mean, sc.spike_amplitude_std))
                self._spike_remaining = max(1.0,
                    self._rng.normal(sc.spike_duration_mean, sc.spike_duration_std))

        # Combine all factors
        total_factor = slow_factor * fast_factor * self._spike_factor

        # Enforce minimum (plough always has some resistance)
        total_factor = max(total_factor, sc.min_resistance_fraction)

        return F_base * total_factor

    def skid_friction_force(self) -> float:
        """
        Friction force between plough skids and seabed [N].
        """
        rho_water = 1025.0
        plough_volume = self.cfg.mass / 7850.0
        W_sub = (self.cfg.mass * 9.81) - (rho_water * 9.81 * plough_volume)
        return self.cfg.skid_friction * W_sub

    def hydrodynamic_drag(self, speed: float, current_speed: float = 0.0) -> float:
        """
        Hydrodynamic drag on the plough body [N].
        """
        rho = 1025.0
        V_rel = speed - current_speed
        return 0.5 * rho * self.cfg.Cd_plough * self.cfg.frontal_area * V_rel * abs(V_rel)

    def total_resistance(self, speed: float, current_speed: float = 0.0,
                         dt: float = 0.1) -> float:
        """
        Total horizontal resistance force on the plough [N].

        Sum of soil cutting (stochastic), skid friction, and hydrodynamic drag.
        """
        if self.is_stopped:
            return 0.0

        F_soil = self.soil_cutting_force(speed, dt)
        F_friction = self.skid_friction_force()
        F_drag = self.hydrodynamic_drag(speed, current_speed)

        # Speed dependence: at zero speed, only static resistance
        speed_factor = 1.0 + 0.1 * abs(speed)

        return (F_soil * speed_factor + F_friction + F_drag)

    def set_hard_soil(self, factor: float = 3.0):
        """Activate hard soil encounter."""
        self._hard_soil_active = True
        self._hard_soil_factor = factor

    def clear_hard_soil(self):
        """Return to normal soil conditions."""
        self._hard_soil_active = False
        self._hard_soil_factor = 1.0

    def set_stopped(self, stopped: bool = True):
        """Set plough as stopped (hit obstruction)."""
        self.is_stopped = stopped

    def step(self, speed: float, direction: float, dt: float):
        """
        Update plough position.

        Parameters:
            speed: Plough speed over ground [m/s]
            direction: Direction of travel [rad] in NED
            dt: Time step [s]
        """
        if not self.is_stopped:
            self.speed = speed
            self.position[0] += speed * np.cos(direction) * dt
            self.position[1] += speed * np.sin(direction) * dt
        else:
            self.speed = 0.0

    def force_summary(self, speed: float, current_speed: float = 0.0,
                      dt: float = 0.1) -> dict:
        """Return breakdown of forces for analysis."""
        return {
            'soil_cutting': self.soil_cutting_force(speed, dt),
            'skid_friction': self.skid_friction_force(),
            'hydro_drag': self.hydrodynamic_drag(speed, current_speed),
            'total': self.total_resistance(speed, current_speed, dt),
            'is_stopped': self.is_stopped,
            'hard_soil_active': self._hard_soil_active,
            'hard_soil_factor': self._hard_soil_factor,
            'spike_active': self._spike_active,
            'spike_factor': self._spike_factor,
        }
