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
    2. Fractional Gaussian noise (fGn): models the broad-band soil
       heterogeneity that a plough encounters as it advances.  Soil
       properties (Su, grain size, layering) vary at every spatial
       scale from centimetres to hundreds of metres, giving a 1/f-type
       power spectrum.  The Hurst exponent H controls the spectral
       slope: PSD ~ f^(-(2H-1)).  H~0.85-0.95 gives the "colored"
       appearance seen in real operational data.
    3. Spike events: boulders, hard layers, shell beds
       Modelled as Poisson arrivals with random amplitude
    4. Soil zone transitions: extended periods of soft or hard soil
       Modelled as a Markov chain with transition rates between zones

    Based on real operational data showing:
      - Mean tension ~30-40t
      - Std ~15-20t (COV ~0.4-0.5)
      - Range 5-100t
      - Power at all temporal scales from seconds to minutes
      - Spectral slope approximately -1 (pink noise)
      - Extended soft zones (5-10 min) where tension drops to 5-20t
      - Occasional very high spikes near 100t
    """
    # Fractional Gaussian noise — replaces the old slow/fast filter pair
    # with a single broadband process that has power at all frequencies
    hurst: float = 0.90          # Hurst exponent (0.5=white, 1.0=pink)
                                  # 0.90 gives PSD ~ f^-0.8, close to pink
    fgn_cov: float = 0.55        # COV of the fGn soil factor [-]
                                  # Controls overall amplitude of soil variability
    fgn_lp_tau: float = 4.0      # Low-pass filter tau [s] applied to fGn
                                  # Attenuates HF content above ~2*pi*tau period
                                  # Physically: plough body length (~8m) and mass (~25t)
                                  # smooth out the fastest soil heterogeneities; the
                                  # share itself is only ~0.2-0.3m wide (the cable
                                  # passes through it and is buried by the furrow
                                  # closing behind), but the plough inertia provides
                                  # the dominant mechanical low-pass effect

    # Simulation parameters (set automatically by PloughModel.__init__)
    fgn_dt: float = 0.2          # Time step for fGn generation [s]
    fgn_duration: float = 3600.0 # Total duration for pre-generation [s]

    # Spike events (boulders, hard layers)
    spike_rate: float = 0.005   # Mean spike rate [1/s] (~1 spike per 200s)
    spike_amplitude_mean: float = 2.0   # Mean spike amplitude as multiplier of base
    spike_amplitude_std: float = 0.8    # Std of spike amplitude multiplier
    spike_duration_mean: float = 5.0    # Mean spike duration [s]
    spike_duration_std: float = 3.0     # Std of spike duration [s]

    # Soil zone transitions (Markov model)
    # Zones: 'normal', 'soft', 'hard'
    # Transition rates [1/s] — mean time in zone = 1/rate_out
    zone_normal_to_soft_rate: float = 0.0005    # ~1 transition per 2000s
    zone_normal_to_hard_rate: float = 0.0003    # ~1 transition per 3300s
    zone_soft_to_normal_rate: float = 0.003     # mean soft zone duration ~330s (~5 min)
    zone_hard_to_normal_rate: float = 0.005     # mean hard zone duration ~200s (~3 min)
    zone_soft_factor: float = 0.35              # Resistance multiplier in soft zone
    zone_hard_factor: float = 2.0               # Resistance multiplier in hard zone
    zone_transition_tau: float = 30.0           # Smooth transition time constant [s]

    # Minimum resistance (plough always has some soil contact)
    min_resistance_fraction: float = 0.1  # Minimum as fraction of base

    # Random seed (None for random)
    seed: int = None


def generate_fgn(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate fractional Gaussian noise using the Davies-Harte (circulant
    embedding) method.

    Produces a sequence of length n with Hurst exponent H.
    The output has zero mean and unit variance.

    PSD of fGn ~ f^(-(2H-1)) for 0.5 < H < 1:
      H = 0.5  → white noise (flat PSD)
      H = 0.75 → PSD ~ f^-0.5
      H = 0.90 → PSD ~ f^-0.8  (close to pink)
      H → 1.0  → PSD ~ f^-1    (pink noise)

    Parameters:
        n: Number of samples to generate
        H: Hurst exponent (0 < H < 1)
        rng: numpy random Generator instance

    Returns:
        Array of length n, zero-mean, unit-variance fGn
    """
    # Autocovariance of fGn at lag k:
    #   gamma(k) = 0.5 * (|k-1|^(2H) - 2|k|^(2H) + |k+1|^(2H))
    # For the circulant embedding we need gamma(0..m-1) where m = 2*n_pad
    n_pad = 1
    while n_pad < n:
        n_pad *= 2
    m = 2 * n_pad

    k = np.arange(n_pad + 1, dtype=np.float64)
    gamma = 0.5 * (np.abs(k - 1)**(2*H) - 2.0 * np.abs(k)**(2*H) + np.abs(k + 1)**(2*H))

    # Build first row of circulant matrix: [gamma(0), gamma(1), ..., gamma(n_pad), gamma(n_pad-1), ..., gamma(1)]
    row = np.zeros(m)
    row[:n_pad + 1] = gamma
    row[n_pad + 1:] = gamma[n_pad - 1:0:-1]

    # Eigenvalues of circulant matrix (real, via FFT)
    eigenvalues = np.fft.rfft(row).real

    # All eigenvalues should be non-negative for valid covariance
    # If any are negative (rare for H close to 1), clamp to zero
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Generate complex Gaussian in frequency domain
    # For rfft of length m, we need m//2 + 1 complex values
    n_freq = m // 2 + 1
    z = rng.normal(0, 1, n_freq) + 1j * rng.normal(0, 1, n_freq)
    # DC and Nyquist components are real
    z[0] = z[0].real * np.sqrt(2)
    z[-1] = z[-1].real * np.sqrt(2)

    # Multiply by sqrt of eigenvalues and transform back
    w = np.fft.irfft(np.sqrt(eigenvalues) * z, n=m)

    # Take first n samples and normalize to unit variance
    fgn = w[:n].real
    std = np.std(fgn)
    if std > 1e-12:
        fgn = (fgn - np.mean(fgn)) / std

    return fgn


@dataclass
class PloughConfig:
    """Plough configuration."""
    mass: float = 25e3              # Plough mass [kg] (25t or 35t)
    width: float = 3.0              # Effective interaction width [m]
                                    # Note: the actual share (cutting blade) is only
                                    # ~0.2-0.3m wide — the cable passes through it and
                                    # is buried by the furrow closing behind the plough.
                                    # This "width" is an effective parameter for the soil
                                    # failure mechanism including passive earth pressure
                                    # on the plough body, furrow wall shearing, and
                                    # displaced soil volume.  Roughly = plough body width.
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
        self._spike_active = False
        self._spike_factor = 1.0
        self._spike_remaining = 0.0  # Time remaining in current spike [s]

        # Pre-generate fractional Gaussian noise for soil variability
        n_fgn = int(sc.fgn_duration / sc.fgn_dt) + 1
        fgn_raw = generate_fgn(n_fgn, sc.hurst, self._rng)

        # Apply low-pass filter to attenuate high-frequency content.
        # The plough share integrates soil over its width, and plough
        # inertia smooths out the very fastest force changes.
        if sc.fgn_lp_tau > 0:
            alpha = sc.fgn_dt / (sc.fgn_lp_tau + sc.fgn_dt)
            fgn_filtered = np.empty_like(fgn_raw)
            fgn_filtered[0] = fgn_raw[0]
            for k in range(1, len(fgn_raw)):
                fgn_filtered[k] = (1 - alpha) * fgn_filtered[k - 1] + alpha * fgn_raw[k]
            # Re-normalize to unit variance (filter reduces variance)
            std = np.std(fgn_filtered)
            if std > 1e-12:
                fgn_filtered = (fgn_filtered - np.mean(fgn_filtered)) / std
            self._fgn_sequence = fgn_filtered
        else:
            self._fgn_sequence = fgn_raw

        self._fgn_index = 0  # Current index into fGn sequence
        self._fgn_dt = sc.fgn_dt

        # Soil zone state (Markov chain)
        self._zone = 'normal'           # Current zone: 'normal', 'soft', 'hard'
        self._zone_factor_target = 1.0  # Target zone factor
        self._zone_factor = 1.0         # Smoothed zone factor (1st-order filtered)

        # Enable/disable stochastic model
        self.stochastic_enabled = True

        # Cache for stochastic state to prevent double-advancing when
        # soil_cutting_force() is called multiple times per timestep
        # (e.g. via total_resistance() and then force_summary())
        self._stochastic_call_id = -1     # call_id of last stochastic update
        self._cached_soil_factor = 1.0    # cached total stochastic factor
        self._call_id = 0                 # incremented by advance_step()

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

    def advance_step(self):
        """Mark the start of a new simulation timestep.

        Must be called once per timestep BEFORE any force calculations.
        This ensures the stochastic state advances exactly once per step.
        """
        self._call_id += 1

    def _update_stochastic_state(self, dt: float) -> float:
        """
        Advance stochastic soil state by one timestep and return the
        total multiplicative factor.

        Uses call_id (set by advance_step()) to ensure the state only
        advances once per simulation timestep, even if called multiple
        times (e.g. from both total_resistance() and force_summary()).
        """
        # Return cached result if already computed this timestep
        if self._call_id == self._stochastic_call_id:
            return self._cached_soil_factor

        sc = self.cfg.stochastic_soil

        # --- Update soil zone (Markov chain transitions) ---
        if self._zone == 'normal':
            if self._rng.random() < sc.zone_normal_to_soft_rate * dt:
                self._zone = 'soft'
                self._zone_factor_target = sc.zone_soft_factor
            elif self._rng.random() < sc.zone_normal_to_hard_rate * dt:
                self._zone = 'hard'
                self._zone_factor_target = sc.zone_hard_factor
        elif self._zone == 'soft':
            if self._rng.random() < sc.zone_soft_to_normal_rate * dt:
                self._zone = 'normal'
                self._zone_factor_target = 1.0
        elif self._zone == 'hard':
            if self._rng.random() < sc.zone_hard_to_normal_rate * dt:
                self._zone = 'normal'
                self._zone_factor_target = 1.0

        # Smooth zone transition (1st-order filter)
        alpha_zone = dt / (sc.zone_transition_tau + dt)
        self._zone_factor = (1 - alpha_zone) * self._zone_factor + alpha_zone * self._zone_factor_target

        # --- Fractional Gaussian noise soil variability ---
        # Read current fGn value (pre-generated broadband process)
        idx = min(self._fgn_index, len(self._fgn_sequence) - 1)
        fgn_value = self._fgn_sequence[idx]
        # Advance index (one step per dt; if dt != fgn_dt, nearest sample)
        self._fgn_index += max(1, round(dt / self._fgn_dt))

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

        # Combine factors:
        # Zone affects the mean level, but noise is only partially suppressed.
        # In soft soil the plough still encounters heterogeneous material —
        # the variability reduces but doesn't vanish.
        #
        # Approach: the fGn noise amplitude scales with max(sqrt(zone_factor), 0.4)
        # rather than zone_factor itself.  This means in soft zone (factor=0.15),
        # noise_scale = max(sqrt(0.15), 0.4) = 0.4, so the noise is 40% of
        # normal amplitude even though the mean is only 15%.
        noise_scale = max(np.sqrt(self._zone_factor), 0.4)
        # mean_part: zone_factor (sets the mean level)
        # noise_part: noise_scale * cov * fgn (additive around the zone mean)
        total_factor = self._zone_factor + noise_scale * sc.fgn_cov * fgn_value
        total_factor *= self._spike_factor

        # Enforce minimum (plough always has some resistance)
        total_factor = max(total_factor, sc.min_resistance_fraction)

        # Cache result
        self._stochastic_call_id = self._call_id
        self._cached_soil_factor = total_factor

        return total_factor

    def soil_cutting_force(self, speed: float = None, dt: float = 0.1) -> float:
        """
        Soil cutting force with stochastic variability [N].

        The stochastic model produces time-varying resistance using
        fractional Gaussian noise (fGn) to capture the broad-band soil
        heterogeneity observed in real ploughing operations.
        Includes soil zone transitions (soft/hard patches) and spike events.
        """
        F_base = self.base_soil_cutting_force()

        if not self.stochastic_enabled:
            return F_base

        total_factor = self._update_stochastic_state(dt)

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
        In soft soil zones, skid friction also reduces (plough is less embedded).
        """
        if self.is_stopped:
            return 0.0

        F_soil = self.soil_cutting_force(speed, dt)
        F_friction = self.skid_friction_force()
        F_drag = self.hydrodynamic_drag(speed, current_speed)

        # In soft zones, plough is less embedded — reduce skid friction too
        # Use sqrt of zone factor for friction (less reduction than soil)
        friction_zone_factor = max(np.sqrt(self._zone_factor), 0.3)
        F_friction *= friction_zone_factor

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
            'zone': self._zone,
            'zone_factor': self._zone_factor,
        }
