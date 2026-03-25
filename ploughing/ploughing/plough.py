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

    ** All parameters are SPATIAL (per metre) — soil is a property of **
    ** position along the seabed, not time. **

    The model generates realistic position-varying soil resistance by
    superimposing several processes:

    1. Base resistance: deterministic, from soil mechanics (Nc*Su*w*d)
    2. Fractional Gaussian noise (fGn): models the broad-band soil
       heterogeneity along the seabed.  Soil properties (Su, grain size,
       layering) vary at every spatial scale from centimetres to hundreds
       of metres, giving a 1/f-type power spectrum.  The Hurst exponent H
       controls the spectral slope: PSD ~ k^(-(2H-1)) where k is
       spatial frequency [1/m].  H~0.85-0.95 gives the "colored"
       appearance seen in real operational data.
    3. Spike events: boulders, hard layers, shell beds
       Modelled as Poisson arrivals per metre with random amplitude
    4. Soil zone transitions: extended regions of soft or hard soil
       Modelled as a Markov chain with transition rates per metre

    The conversion from spatial to temporal frequency depends on plough
    speed:  f_temporal = f_spatial * V_plough.  A faster plough sees
    higher temporal frequencies for the same spatial heterogeneity.

    Based on real operational data showing:
      - Mean tension ~30-40t
      - Std ~15-20t (COV ~0.4-0.5)
      - Range 5-100t
      - Power at all temporal scales from seconds to minutes
      - Spectral slope approximately -1 (pink noise)
      - Extended soft zones (50-100m) where tension drops to 5-20t
      - Occasional very high spikes near 100t
    """
    # Fractional Gaussian noise — broadband spatial soil variability
    hurst: float = 0.90          # Hurst exponent (0.5=white, 1.0=pink)
                                  # 0.90 gives PSD ~ k^-0.8, close to pink
    fgn_cov: float = 0.55        # COV of the fGn soil factor [-]
                                  # Controls overall amplitude of soil variability
    fgn_lp_length: float = 0.5   # Spatial low-pass filter length [m]
                                  # Attenuates variability shorter than this scale.
                                  # Physically: plough body length (~8m) and share
                                  # width (~0.3m) smooth out sub-metre heterogeneity;
                                  # the share itself is only ~0.2-0.3m wide (the cable
                                  # passes through it and is buried by the furrow
                                  # closing behind), but the plough body + inertia
                                  # provides the dominant mechanical low-pass effect.

    # Spatial sampling for fGn pre-generation
    fgn_dx: float = 0.025        # Spatial sample interval [m]
    fgn_length: float = 600.0    # Total length of pre-generated seabed [m]

    # Spike events (boulders, hard layers) — all rates and sizes in METRES
    spike_rate: float = 0.04     # Mean spike rate [1/m] (~1 spike per 25m)
    spike_amplitude_mean: float = 2.0   # Mean spike amplitude as multiplier of base
    spike_amplitude_std: float = 0.8    # Std of spike amplitude multiplier
    spike_length_mean: float = 0.6      # Mean spike length [m] along seabed
    spike_length_std: float = 0.4       # Std of spike length [m]

    # Soil zone transitions (Markov model) — rates per METRE
    # Zones: 'normal', 'soft', 'hard'
    # Transition rates [1/m] — mean distance in zone = 1/rate_out
    zone_normal_to_soft_rate: float = 0.002     # ~1 transition per 500m
    zone_normal_to_hard_rate: float = 0.0015    # ~1 transition per 670m
    zone_soft_to_normal_rate: float = 0.015     # mean soft zone length ~67m
    zone_hard_to_normal_rate: float = 0.04      # mean hard zone length ~25m
    zone_soft_factor: float = 0.35              # Resistance multiplier in soft zone
    zone_hard_factor: float = 1.4               # Resistance multiplier in hard zone
    zone_transition_length: float = 2.0         # Smooth transition over this distance [m]

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

    # Speed-dependent resistance
    #
    # The total plough resistance has two components:
    #
    # 1. Quasi-static soil failure: Nc*Su*w*d (clay) or passive earth
    #    pressure (sand).  Speed-independent — this is the force to shear
    #    the soil at any speed.  A tanh(V / V_breakout) ramp models the
    #    transition from static (no cutting) to dynamic (full cutting)
    #    resistance as the plough starts moving.
    #
    # 2. Dynamic soil inertial drag: C_d * rho_soil * A_furrow * V^2
    #    The plough must accelerate the soil it displaces out of the
    #    furrow path.  The momentum transfer scales with rho * A * V^2,
    #    the same dimensional argument as fluid drag (Reece 1964,
    #    Palmer & King 2008).  A_furrow = width * burial_depth.
    #
    #    C_d is an empirical drag coefficient for the soil failure wedge
    #    (typically 2-6 depending on share geometry, soil density, and
    #    furrow shape).  It captures:
    #      - Inertial acceleration of displaced soil mass
    #      - Dynamic pressure on the failure wedge
    #      - Seabed suction effects on the plough body
    #
    # The quasi-static term dominates at low speed; the V^2 term limits
    # maximum plough speed.  Self-regulation is provided naturally: hard
    # soil slows the plough, reducing V^2 drag but keeping the quasi-static
    # term high.  Soft soil lets the plough speed up, and V^2 drag prevents
    # runaway.
    #
    # At operating speed (V ~ 0.12 m/s) with default parameters:
    #   A_furrow = 3.0 * 1.5 = 4.5 m^2
    #   F_inertial = 100 * 1800 * 4.5 * 0.12^2 ≈ 12 kN (~4% of total)
    # At V = 0.25 m/s:
    #   F_inertial ≈ 51 kN (significant speed-limiting contribution)
    # At V = 0.30 m/s:
    #   F_inertial ≈ 73 kN (limits maximum plough speed)
    #
    # Note: C_d for soil-plough interaction is NOT a fluid drag coefficient.
    # It is an empirical dimensionless constant specific to the soil failure
    # mechanism during ploughing.  Values of O(100) are typical because at
    # ploughing speeds (0.1-0.3 m/s) the soil is being sheared and broken,
    # not just displaced — the effective resistance includes cohesion and
    # strain-rate effects beyond pure momentum transfer.  The primary speed
    # sensitivity in operational data comes from the catenary spring dynamics
    # and plough inertia, not the V^2 term, which mainly serves to limit
    # maximum plough speed during runaway conditions (very soft soil).
    speed_breakout: float = 0.02         # Breakout speed for static→dynamic ramp [m/s]
    Cd_soil: float = 100.0               # Soil inertial drag coefficient [-]
    rho_soil: float = 1800.0             # Saturated bulk density of soil [kg/m^3]

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
        self._spike_remaining = 0.0  # Distance remaining in current spike [m]

        # Pre-generate fractional Gaussian noise for spatial soil variability.
        # The fGn sequence represents soil heterogeneity along the seabed,
        # sampled every fgn_dx metres.  The plough reads this sequence as
        # it advances, so faster ploughing gives higher temporal frequency
        # content (correct physical behaviour).
        n_fgn = int(sc.fgn_length / sc.fgn_dx) + 1
        fgn_raw = generate_fgn(n_fgn, sc.hurst, self._rng)

        # Apply spatial low-pass filter to attenuate sub-metre heterogeneity.
        # The plough share integrates soil over its width, and the plough body
        # length (~8m) + inertia smooths out the very finest spatial features.
        if sc.fgn_lp_length > 0:
            # Filter constant: number of dx steps per length scale
            alpha = sc.fgn_dx / (sc.fgn_lp_length + sc.fgn_dx)
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

        self._fgn_index = 0      # Current index into fGn sequence
        self._fgn_dx = sc.fgn_dx # Spatial sample interval [m]
        self._distance_accumulator = 0.0  # Sub-sample distance accumulator [m]

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
        self._pending_distance_step = 0.0 # distance to advance this step [m]

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

    def advance_step(self, distance_step: float = 0.0):
        """Mark the start of a new simulation timestep.

        Must be called once per timestep BEFORE any force calculations.
        This ensures the stochastic state advances exactly once per step.

        Parameters:
            distance_step: Distance the plough advanced this step [m].
                           Typically = plough.speed * dt.  If zero (plough
                           stopped), the stochastic soil state does NOT
                           change — the plough sees the same soil until
                           it moves again.
        """
        self._call_id += 1
        self._pending_distance_step = distance_step

    def _update_stochastic_state(self, distance_step: float) -> float:
        """
        Advance stochastic soil state by a spatial step and return the
        total multiplicative factor.

        All rates and lengths are SPATIAL (per metre along the seabed).
        If distance_step == 0, the plough hasn't moved and soil state
        doesn't change (returns cached value or initial state).

        Uses call_id (set by advance_step()) to ensure the state only
        advances once per simulation timestep, even if called multiple
        times (e.g. from both total_resistance() and force_summary()).
        """
        # Return cached result if already computed this timestep
        if self._call_id == self._stochastic_call_id:
            return self._cached_soil_factor

        sc = self.cfg.stochastic_soil

        # If plough hasn't moved, soil doesn't change
        if distance_step <= 0:
            self._stochastic_call_id = self._call_id
            return self._cached_soil_factor

        # --- Update soil zone (Markov chain transitions) ---
        # Transition probabilities per metre traversed
        if self._zone == 'normal':
            if self._rng.random() < sc.zone_normal_to_soft_rate * distance_step:
                self._zone = 'soft'
                self._zone_factor_target = sc.zone_soft_factor
            elif self._rng.random() < sc.zone_normal_to_hard_rate * distance_step:
                self._zone = 'hard'
                self._zone_factor_target = sc.zone_hard_factor
        elif self._zone == 'soft':
            if self._rng.random() < sc.zone_soft_to_normal_rate * distance_step:
                self._zone = 'normal'
                self._zone_factor_target = 1.0
        elif self._zone == 'hard':
            if self._rng.random() < sc.zone_hard_to_normal_rate * distance_step:
                self._zone = 'normal'
                self._zone_factor_target = 1.0

        # Smooth zone transition (spatial 1st-order filter)
        alpha_zone = distance_step / (sc.zone_transition_length + distance_step)
        self._zone_factor = (1 - alpha_zone) * self._zone_factor + alpha_zone * self._zone_factor_target

        # --- Fractional Gaussian noise soil variability ---
        # Accumulate distance and advance fGn index when we've moved fgn_dx
        self._distance_accumulator += distance_step
        steps = int(self._distance_accumulator / self._fgn_dx)
        if steps > 0:
            self._fgn_index += steps
            self._distance_accumulator -= steps * self._fgn_dx

        # Read current fGn value (pre-generated broadband spatial process)
        idx = min(self._fgn_index, len(self._fgn_sequence) - 1)
        fgn_value = self._fgn_sequence[idx]

        # --- Update spike events ---
        if self._spike_active:
            self._spike_remaining -= distance_step  # remaining distance [m]
            if self._spike_remaining <= 0:
                self._spike_active = False
                self._spike_factor = 1.0
        else:
            # Check for new spike (Poisson process per metre)
            if self._rng.random() < sc.spike_rate * distance_step:
                self._spike_active = True
                self._spike_factor = max(1.0,
                    self._rng.normal(sc.spike_amplitude_mean, sc.spike_amplitude_std))
                self._spike_remaining = max(0.1,
                    self._rng.normal(sc.spike_length_mean, sc.spike_length_std))

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

        The stochastic model produces spatially-varying resistance using
        fractional Gaussian noise (fGn) to capture the broad-band soil
        heterogeneity observed in real ploughing operations.
        Includes soil zone transitions (soft/hard patches) and spike events.

        The distance_step used for stochastic state advancement is set by
        advance_step() at the start of each timestep.
        """
        F_base = self.base_soil_cutting_force()

        if not self.stochastic_enabled:
            return F_base

        distance_step = self._pending_distance_step
        total_factor = self._update_stochastic_state(distance_step)

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

    def soil_inertial_drag(self, speed: float) -> float:
        """
        Dynamic soil inertial drag [N].

        The plough must accelerate the soil it displaces out of the furrow.
        This momentum transfer scales as:

            F = C_d * rho_soil * A_furrow * V^2

        where A_furrow = width * burial_depth is the furrow cross-section.

        This is the same dimensional form as fluid drag (Reece 1964,
        Palmer & King 2008).  C_d is an empirical coefficient capturing
        the inertial soil acceleration, dynamic pressure on the failure
        wedge, and seabed suction on the plough body.
        """
        V = abs(speed)
        A_furrow = self.cfg.width * self.cfg.burial_depth
        return self.cfg.Cd_soil * self.cfg.rho_soil * A_furrow * V * V

    def total_resistance(self, speed: float, current_speed: float = 0.0,
                         dt: float = 0.1) -> float:
        """
        Total horizontal resistance force on the plough [N].

        Four components:

        1. Quasi-static soil cutting (stochastic, speed-independent):
               F_soil = F_base * stochastic_factor * tanh(V / V_breakout)
           The tanh ramp provides a smooth transition from zero resistance
           (stationary plough) to full cutting resistance.  A stationary
           plough has no cutting force — only passive earth pressure
           (captured by the residual skid friction).

        2. Dynamic soil inertial drag:
               F_inertial = C_d * rho_soil * A_furrow * V^2
           Momentum transfer to displaced soil.  Dominates at higher speeds,
           providing the primary speed-limiting mechanism.

        3. Skid friction:
               F_friction = mu * W_submerged
           Coulomb friction, modulated by soil zone factor.

        4. Hydrodynamic drag:
               F_hydro = 0.5 * rho_w * Cd * A * V_rel^2
           Usually small compared to soil forces.

        Self-regulation: in hard soil the vessel slows, reducing the V^2
        inertial drag while the quasi-static term stays high.  In soft soil
        the vessel speeds up and V^2 drag limits the speed increase.
        """
        if self.is_stopped:
            return 0.0

        F_soil = self.soil_cutting_force(speed, dt)
        F_friction = self.skid_friction_force()
        F_hydro = self.hydrodynamic_drag(speed, current_speed)
        F_inertial = self.soil_inertial_drag(speed)

        # In soft zones, plough is less embedded — reduce skid friction too
        friction_zone_factor = max(np.sqrt(self._zone_factor), 0.3)
        F_friction *= friction_zone_factor

        # Static→dynamic cutting ramp
        V = abs(speed)
        cutting_ramp = np.tanh(V / self.cfg.speed_breakout)

        return (F_soil * cutting_ramp + F_inertial + F_friction + F_hydro)

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
            'soil_inertial': self.soil_inertial_drag(speed),
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
