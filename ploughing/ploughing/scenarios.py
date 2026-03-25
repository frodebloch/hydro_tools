"""
Predefined simulation scenarios for cable laying ploughing operations.

Scenarios:
1. Steady-state ploughing at different depths (15m, 100m, 500m, 1500m)
2. Plough hits hard soil (sudden increase in resistance)
3. Plough stops completely (obstruction)
4. Speed variation during ploughing
5. Comparison: 25t vs 35t plough
"""

import numpy as np
import matplotlib.pyplot as plt
from .vessel import VesselConfig
from .catenary import TowWireConfig
from .plough import PloughConfig, SoilProperties, StochasticSoilConfig
from .dp_controller import DPControllerConfig
from .environment import EnvironmentConfig, WaveDriftConfig
from .simulation import SimulationConfig, run_simulation, SimulationResult
from .plotting import (plot_overview, plot_force_breakdown, plot_layback_analysis,
                       plot_catenary_snapshot, plot_stochastic_soil,
                       plot_dp_mode_comparison,
                       print_summary)
from .catenary import CatenaryModel


def make_vessel_config() -> VesselConfig:
    """Standard 130m cable laying vessel."""
    return VesselConfig(
        name="Cable Layer 130m",
        lpp=130.0,
        beam=23.0,
        draft=7.8,
        Cb=0.72,
        max_thrust_surge=900e3,   # ~90t
        max_thrust_sway=500e3,    # ~50t
        max_thrust_yaw=20e6,
        tow_point_x=-60.0,       # stern tow
    )


def compute_wire_length(water_depth: float, vessel_draft: float = 7.8,
                        T_h_expected: float = 301e3, buffer: float = 75.0,
                        wire_submerged_weight: float = None) -> float:
    """
    Compute deployed wire length from inextensible catenary arc length + buffer.

    In real ploughing operations, wire deployment is managed so that only a
    small buffer (50-100m) sits on the seabed at operating tension.  This
    avoids excessive grounded wire (which complicates the spring model) while
    ensuring enough reserve to handle tension fluctuations without lifting
    the plough off the seabed.

    Parameters:
        water_depth: Water depth [m]
        vessel_draft: Vessel draft [m] (tow point depth)
        T_h_expected: Expected horizontal tension at plough [N]
                      (~301 kN for Su=12kPa firm clay at 0.12 m/s)
        buffer: Extra wire beyond catenary arc length [m]
                Sits on seabed as reserve. 50-100m typical.
        wire_submerged_weight: Wire submerged weight [N/m].
                               If None, computed from default 76mm/25kg/m wire.

    Returns:
        Total wire length [m]
    """
    if wire_submerged_weight is None:
        # Default 76mm, 25 kg/m wire rope
        rho_water = 1025.0
        diameter = 0.076
        area = np.pi * diameter**2 / 4.0
        wire_submerged_weight = 25.0 * 9.81 - rho_water * 9.81 * area

    h = water_depth - vessel_draft
    if h < 1.0:
        h = 1.0
    w = wire_submerged_weight
    a = T_h_expected / w
    D = a * np.arccosh(1.0 + h / a)
    s = a * np.sinh(D / a)

    wire_length = s + buffer
    return max(wire_length, 150.0)  # minimum 150m for very shallow water


def make_wire_config(water_depth: float) -> TowWireConfig:
    """Tow wire sized for the depth: catenary arc length + buffer."""
    wire_length = compute_wire_length(water_depth)

    return TowWireConfig(
        diameter=0.076,
        linear_mass=25.0,
        total_length=wire_length,
    )


def make_plough_config(mass_tonnes: float) -> PloughConfig:
    """Plough configuration for given mass."""
    return PloughConfig(
        mass=mass_tonnes * 1e3,
        width=3.0 if mass_tonnes <= 25 else 3.5,
        burial_depth=1.5,
    )


# =============================================================================
# Scenario 1: Steady-state ploughing at different depths
# =============================================================================

def scenario_steady_state(water_depth: float = 100.0, plough_mass_t: float = 25.0,
                          speed: float = 0.5, duration: float = 600.0,
                          soil: SoilProperties = None) -> SimulationResult:
    """
    Steady-state ploughing at a given depth.

    Parameters:
        water_depth: Water depth [m]
        plough_mass_t: Plough mass [tonnes]
        speed: Desired ploughing speed [m/s]
        duration: Simulation duration [s]
        soil: Soil properties (default: firm clay)
    """
    if soil is None:
        soil = SoilProperties.firm_clay()

    config = SimulationConfig(
        dt=0.1,
        duration=duration,
        vessel=make_vessel_config(),
        wire=make_wire_config(water_depth),
        plough=make_plough_config(plough_mass_t),
        soil=soil,
        env=EnvironmentConfig(
            water_depth=water_depth,
            current_speed=0.3,
            current_direction=0.0,    # head current (from ahead)
            wind_speed=8.0,
            wind_direction=0.0,       # headwind (from ahead)
        ),
        track_start=(0.0, 0.0),
        track_end=(duration * speed * 2, 0.0),
        track_speed=speed,
    )

    print(f"\n{'='*60}")
    print(f"SCENARIO: Steady-state ploughing")
    print(f"  Depth: {water_depth} m | Plough: {plough_mass_t}t | Speed: {speed} m/s")
    print(f"  Soil: {soil.soil_type.value}")
    print(f"{'='*60}")

    result = run_simulation(config)
    print_summary(result, config)
    return result, config


# =============================================================================
# Scenario 2: Plough hits hard soil
# =============================================================================

def scenario_hard_soil(water_depth: float = 100.0, plough_mass_t: float = 25.0,
                       speed: float = 0.5, hard_soil_time: float = 200.0,
                       hard_soil_factor: float = 3.0,
                       hard_soil_duration: float = 60.0,
                       duration: float = 600.0) -> SimulationResult:
    """
    Ploughing with a hard soil encounter.

    The plough hits a patch of hard soil at `hard_soil_time`, which increases
    the soil cutting force by `hard_soil_factor` for `hard_soil_duration` seconds.
    """
    def event_hard_soil_on(plough, **kwargs):
        print(f"  EVENT: Hard soil encounter (factor x{hard_soil_factor})")
        plough.set_hard_soil(hard_soil_factor)

    def event_hard_soil_off(plough, **kwargs):
        print(f"  EVENT: Hard soil ended, back to normal")
        plough.clear_hard_soil()

    events = [
        (hard_soil_time, event_hard_soil_on),
        (hard_soil_time + hard_soil_duration, event_hard_soil_off),
    ]

    config = SimulationConfig(
        dt=0.1,
        duration=duration,
        vessel=make_vessel_config(),
        wire=make_wire_config(water_depth),
        plough=make_plough_config(plough_mass_t),
        soil=SoilProperties.firm_clay(),
        env=EnvironmentConfig(
            water_depth=water_depth,
            current_speed=0.3,
            current_direction=0.0,    # head current (from ahead)
            wind_speed=8.0,
            wind_direction=0.0,       # headwind (from ahead)
        ),
        track_start=(0.0, 0.0),
        track_end=(duration * speed * 2, 0.0),
        track_speed=speed,
        events=events,
    )

    print(f"\n{'='*60}")
    print(f"SCENARIO: Hard soil encounter")
    print(f"  Depth: {water_depth} m | Plough: {plough_mass_t}t | Speed: {speed} m/s")
    print(f"  Hard soil at t={hard_soil_time}s, factor x{hard_soil_factor}, "
          f"duration {hard_soil_duration}s")
    print(f"{'='*60}")

    result = run_simulation(config)
    print_summary(result, config)
    return result, config


# =============================================================================
# Scenario 3: Plough stops (obstruction)
# =============================================================================

def scenario_plough_stop(water_depth: float = 100.0, plough_mass_t: float = 25.0,
                         speed: float = 0.5, stop_time: float = 200.0,
                         stop_duration: float = 120.0,
                         duration: float = 600.0) -> SimulationResult:
    """
    Plough hits an obstruction and stops completely.

    The vessel continues forward, increasing wire tension until the DP
    system detects the overload and adjusts (or wire breaks / operator intervenes).
    """
    def event_plough_stop(plough, **kwargs):
        print(f"  EVENT: Plough STOPPED (obstruction)")
        plough.set_stopped(True)

    def event_plough_release(plough, **kwargs):
        print(f"  EVENT: Plough released, resuming")
        plough.set_stopped(False)

    events = [
        (stop_time, event_plough_stop),
        (stop_time + stop_duration, event_plough_release),
    ]

    config = SimulationConfig(
        dt=0.1,
        duration=duration,
        vessel=make_vessel_config(),
        wire=make_wire_config(water_depth),
        plough=make_plough_config(plough_mass_t),
        soil=SoilProperties.firm_clay(),
        env=EnvironmentConfig(
            water_depth=water_depth,
            current_speed=0.3,
            current_direction=0.0,    # head current (from ahead)
            wind_speed=8.0,
            wind_direction=0.0,       # headwind (from ahead)
        ),
        track_start=(0.0, 0.0),
        track_end=(duration * speed * 2, 0.0),
        track_speed=speed,
        events=events,
    )

    print(f"\n{'='*60}")
    print(f"SCENARIO: Plough stop (obstruction)")
    print(f"  Depth: {water_depth} m | Plough: {plough_mass_t}t | Speed: {speed} m/s")
    print(f"  Stop at t={stop_time}s, duration {stop_duration}s")
    print(f"{'='*60}")

    result = run_simulation(config)
    print_summary(result, config)
    return result, config


# =============================================================================
# Scenario 4: Depth comparison
# =============================================================================

def scenario_depth_comparison(depths=[15, 100, 500, 1500],
                               plough_mass_t: float = 25.0,
                               speed: float = 0.5,
                               duration: float = 600.0) -> dict:
    """
    Compare ploughing operations at different water depths.
    Returns dict of {depth: (result, config)}.
    """
    results = {}
    for depth in depths:
        print(f"\n--- Running depth = {depth} m ---")
        res, cfg = scenario_steady_state(depth, plough_mass_t, speed, duration)
        results[depth] = (res, cfg)
    return results


# =============================================================================
# Scenario 5: Plough mass comparison
# =============================================================================

def scenario_plough_comparison(water_depth: float = 100.0,
                                plough_masses=[25, 35],
                                speed: float = 0.5,
                                duration: float = 600.0) -> dict:
    """Compare 25t and 35t plough."""
    results = {}
    for mass in plough_masses:
        print(f"\n--- Running plough mass = {mass}t ---")
        res, cfg = scenario_steady_state(water_depth, mass, speed, duration)
        results[mass] = (res, cfg)
    return results


# =============================================================================
# Scenario 6: Stochastic soil — 1 hour in challenging seabed
# =============================================================================

def scenario_stochastic_soil(water_depth: float = 30.0, plough_mass_t: float = 25.0,
                              speed: float = 0.12, duration: float = 3600.0,
                              seed: int = 42) -> SimulationResult:
    """
    1-hour ploughing simulation in challenging, heterogeneous seabed.

    Calibrated against real operational data showing:
      - Tow tension: mean ~30-40t, range 5-100t
      - Vessel speed: ~0.05-0.15 m/s, inversely correlated with tension
      - Extended soft soil zones (5-10 min duration)
      - High-frequency tension variability with occasional large spikes

    Uses shallow water (default 20m) which is typical for nearshore cable
    burial — the short catenary gives stiff coupling so plough force
    variations transmit directly to the vessel, matching operational data.

    Parameters:
        water_depth: Water depth [m]
        plough_mass_t: Plough mass [tonnes]
        speed: Desired base ploughing speed [m/s] (DP setpoint; tension-speed
               modulation adjusts effective speed ±0.10 m/s around this value
               in response to soil resistance variations)
        duration: Simulation duration [s] (3600 = 1 hour)
        seed: Random seed for reproducibility
    """
    # Tuned stochastic soil parameters to match operational data
    # All rates and lengths are SPATIAL (per metre along seabed).
    # Converted from temporal parameters using reference speed ~0.12 m/s.
    stochastic_cfg = StochasticSoilConfig(
        # Fractional Gaussian noise for broadband soil variability
        hurst=0.90,                 # PSD ~ f^-0.8 (close to pink noise)
        fgn_cov=0.50,              # 50% COV — creates the characteristic spread
        fgn_dx=0.025,              # 2.5 cm spatial sample interval
        fgn_length=600.0,          # 600m of pre-generated seabed

        # Spike events (boulders, hard layers) — per metre
        spike_rate=0.04,            # ~1 spike per 25m
        spike_amplitude_mean=1.4,   # Mean spike is 1.4x base (moderate)
        spike_amplitude_std=0.3,    # Tight variability (less extreme spikes)
        spike_length_mean=1.4,      # 1.4m average spike extent along seabed
        spike_length_std=1.0,       # Std of spike length

        # Soil zone transitions — rates per metre
        zone_normal_to_soft_rate=0.002,    # ~1 soft zone per 500m
        zone_normal_to_hard_rate=0.0015,   # ~1 hard zone per 670m
        zone_soft_to_normal_rate=0.015,    # Soft zone ~67m long
        zone_hard_to_normal_rate=0.06,     # Hard zone ~17m long
        zone_soft_factor=0.15,             # Tension drops to 15% in soft zone
        zone_hard_factor=1.4,              # Tension increases to 140% in hard zone
        zone_transition_length=2.0,        # 2m smooth transition
        min_resistance_fraction=0.02,      # Min 2% of base resistance
        seed=seed,
    )

    plough_cfg = make_plough_config(plough_mass_t)
    plough_cfg.stochastic_soil = stochastic_cfg

    # Wire length: catenary arc length at operating tension + buffer.
    wire_cfg = TowWireConfig(
        diameter=0.076,
        linear_mass=25.0,
        total_length=compute_wire_length(water_depth),
    )

    config = SimulationConfig(
        dt=0.2,                     # 0.2s step (adequate for 1hr sim)
        duration=duration,
        vessel=make_vessel_config(),
        wire=wire_cfg,
        plough=plough_cfg,
        soil=SoilProperties(
            undrained_shear_strength=15e3,  # Su = 15 kPa (firm clay)
            submerged_unit_weight=8.0e3,
        ),
        dp=DPControllerConfig(
            surge_mode='speed',
            T_surge=114.0,          # Surge natural period [s] → omega_n=0.055 rad/s
            zeta_surge=0.9,         # Slightly overdamped (station-keeping typical)
            Kp_speed=0.099,         # 2*zeta*omega_n = 0.099 (non-dim, ×M_surge)
            Ki_speed=0.003,         # omega_n^2 = 0.003 (non-dim, ×M_surge)
            tow_force_feedforward=0.85,  # 85% feedforward for force compensation
            wind_feedforward=0.8,
            # Tension-based speed modulation (ploughing mode)
            # Mimics operator/auto-pilot adjusting speed in response to tow tension.
            # tension_nominal set at ~35t — the self-consistent mean tension when
            # the nonlinear F(V) model is active and V_mean ≈ 0.12 m/s.  The
            # nonlinear force-speed relationship (F ∝ V^0.4) provides the primary
            # self-regulation: hard soil → vessel slows → resistance drops.
            # The tension-speed modulation adds a secondary feedback that mimics
            # operator intervention (reducing setpoint in sustained hard soil).
            tension_speed_modulation=True,
            tension_nominal=350e3,         # ~35t (self-consistent with nonlinear F(V))
            tension_speed_gain=0.5e-6,     # Moderate: ~0.025 m/s per 50kN excess
            tension_speed_filter_tau=20.0,  # 20s filter — recovers from hard patches
            tension_speed_max_reduction=0.07,  # Max 0.07 m/s reduction → V_min ≈ 0.05
            tension_speed_max_increase=0.08,   # Max 0.08 m/s increase → V_max ≈ 0.20
        ),
        env=EnvironmentConfig(
            water_depth=water_depth,
            current_speed=0.2,
            current_direction=0.0,    # Head current (from ahead)
            wind_speed=6.0,
            wind_direction=20.0,      # Quartering headwind (from ahead, slight offset)
            waves=WaveDriftConfig(
                Hs=1.2,                        # Moderate swell
                Tp=7.5,                        # Typical North Sea swell period
                wave_direction=0.0,            # Head seas (waves from ahead)
                C_drift=5000.0,                # Mean drift coefficient
                sv_cov=0.8,                    # Slowly-varying drift COV
                sv_tau=60.0,                   # ~1 min slowly-varying filter
                first_order_surge_factor=0.3,  # Surge RAO [m/m] at Tp=7.5s
                seed=seed + 1000,              # Different seed from soil (uncorrelated)
            ),
        ),
        track_start=(0.0, 0.0),
        track_end=(duration * speed * 2, 0.0),
        track_speed=speed,
    )

    print(f"\n{'='*60}")
    print(f"SCENARIO: Stochastic soil — 1 hour challenging seabed")
    print(f"  Depth: {water_depth} m | Plough: {plough_mass_t}t | Speed: {speed} m/s")
    print(f"  Duration: {duration/60:.0f} min | dt: {config.dt} s")
    print(f"  Soil zones: normal/soft(x{stochastic_cfg.zone_soft_factor})"
          f"/hard(x{stochastic_cfg.zone_hard_factor})")
    print(f"{'='*60}")

    result = run_simulation(config)
    print_summary(result, config)
    return result, config


# =============================================================================
# Scenario 7: DP mode comparison — speed control vs position control
# =============================================================================

def scenario_dp_mode_comparison(water_depth: float = 30.0, plough_mass_t: float = 25.0,
                                 speed: float = 0.12, duration: float = 3600.0,
                                 seed: int = 42):
    """
    Compare speed-control and position-control DP modes on the same soil.

    Runs two 1-hour simulations with identical stochastic soil, environment,
    vessel, and plough — only the DP surge controller differs:

    1. Speed control: PI on speed error + tension-based speed modulation.
       The speed setpoint is adjusted based on filtered tow tension —
       high tension slows the vessel down, low tension speeds it up.
       Speed variation is engineered through the explicit feedback loop.

    2. Position control: PID on along-track position error.
       A reference point moves along the track at constant speed.
       Speed variation emerges naturally from catenary tension "fighting"
       the position controller P-term. No explicit tension feedback.

    Returns dict with both results for comparison.
    """
    # Common stochastic soil configuration — SPATIAL (per metre)
    # Tuned to match real operational data: tension mean ~35t, range 5-100t,
    # rich broadband variability at all temporal scales, occasional spikes,
    # and extended soft zones where tension drops to 5-20t.
    stochastic_cfg = StochasticSoilConfig(
        hurst=0.90,                 # PSD ~ f^-0.8 (close to pink noise)
        fgn_cov=0.75,              # 75% COV — wider variability for [5,100]t range
        fgn_dx=0.025,              # 2.5 cm spatial sample interval
        fgn_lp_length=5.0,         # 5m LP filter — the plough body (~8m long) and
                                    # its active failure wedge integrate soil properties
                                    # over several metres.  This smooths sub-metre noise
                                    # while preserving variability at scales > 10m.
        fgn_length=600.0,          # 600m of pre-generated seabed

        # Spike events — boulders, hard layers, cobble lenses
        spike_rate=0.05,            # ~1 spike per 20m
        spike_amplitude_mean=2.5,   # Mean spike 2.5x base (bigger spikes for 100t peaks)
        spike_amplitude_std=0.8,    # Wide variability in spike size
        spike_length_mean=1.5,      # ~1.5m spike extent
        spike_length_std=1.0,       # Variable spike length

        # Soil zone transitions
        zone_normal_to_soft_rate=0.003,    # ~1 soft zone per 330m
        zone_normal_to_hard_rate=0.002,    # ~1 hard zone per 500m
        zone_soft_to_normal_rate=0.015,    # Soft zone ~67m long (~9 min at 0.12 m/s)
        zone_hard_to_normal_rate=0.06,     # Hard zone ~17m long
        zone_soft_factor=0.10,             # Tension drops to 10% in soft zone (deeper drops)
        zone_hard_factor=1.60,             # Tension rises to 160% in hard zone
        zone_transition_length=2.0,        # 2m smooth transition
        min_resistance_fraction=0.02,      # Min 2% of base
        seed=seed,
    )

    # Wire length: catenary arc length at operating tension + buffer.
    wire_cfg = TowWireConfig(
        diameter=0.076, linear_mass=25.0,
        total_length=compute_wire_length(water_depth),
    )

    soil = SoilProperties(
        undrained_shear_strength=12e3,  # Su = 12 kPa (reduced from 15)
        submerged_unit_weight=8.0e3,
    )

    env_cfg = EnvironmentConfig(
        water_depth=water_depth,
        current_speed=0.2,
        current_direction=0.0,
        wind_speed=6.0,
        wind_direction=20.0,
        waves=WaveDriftConfig(
            Hs=1.2, Tp=7.5, wave_direction=0.0,
            C_drift=5000.0, sv_cov=0.8, sv_tau=60.0,
            first_order_surge_factor=0.3,  # Surge RAO [m/m] at Tp=7.5s
            seed=seed + 1000,
        ),
    )

    def make_plough():
        """Fresh plough config with stochastic soil (same seed each time)."""
        pcfg = make_plough_config(plough_mass_t)
        pcfg.stochastic_soil = StochasticSoilConfig(
            hurst=stochastic_cfg.hurst,
            fgn_cov=stochastic_cfg.fgn_cov,
            fgn_dx=stochastic_cfg.fgn_dx,
            fgn_length=stochastic_cfg.fgn_length,
            spike_rate=stochastic_cfg.spike_rate,
            spike_amplitude_mean=stochastic_cfg.spike_amplitude_mean,
            spike_amplitude_std=stochastic_cfg.spike_amplitude_std,
            spike_length_mean=stochastic_cfg.spike_length_mean,
            spike_length_std=stochastic_cfg.spike_length_std,
            zone_normal_to_soft_rate=stochastic_cfg.zone_normal_to_soft_rate,
            zone_normal_to_hard_rate=stochastic_cfg.zone_normal_to_hard_rate,
            zone_soft_to_normal_rate=stochastic_cfg.zone_soft_to_normal_rate,
            zone_hard_to_normal_rate=stochastic_cfg.zone_hard_to_normal_rate,
            zone_soft_factor=stochastic_cfg.zone_soft_factor,
            zone_hard_factor=stochastic_cfg.zone_hard_factor,
            zone_transition_length=stochastic_cfg.zone_transition_length,
            min_resistance_fraction=stochastic_cfg.min_resistance_fraction,
            seed=seed,
        )
        return pcfg

    # -----------------------------------------------------------------------
    # Run 1: Speed control with tension-speed modulation
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"SCENARIO: DP Mode Comparison — Speed Control")
    print(f"  Setpoint: {speed} m/s with tension-speed modulation")
    print(f"{'='*60}")

    # Speed control: vessel tracks the desired speed, with tension-speed
    # modulation creating speed variation.  High soil resistance → slow down,
    # low resistance → speed up.  tension_nominal is set near the mean
    # plough resistance (~36t) so modulation is centered around zero.
    # With spatial soil, the modulation must be gentle to avoid positive
    # feedback (slow → stuck in hard patch → slower).
    config_speed = SimulationConfig(
        dt=0.2, duration=duration,
        vessel=make_vessel_config(), wire=wire_cfg,
        plough=make_plough(), soil=soil,
        dp=DPControllerConfig(
            surge_mode='speed',
            T_surge=114.0, zeta_surge=0.9,
            Kp_speed=0.099, Ki_speed=0.003,
            tow_force_feedforward=0.85,
            wind_feedforward=0.8,
            # Tension-speed modulation: mimics operator/auto-pilot adjusting
            # speed based on tow tension.  With the elastic catenary spring,
            # the natural tension variability from soil drives speed changes.
            tension_speed_modulation=True,
            tension_nominal=410e3,          # ~42t — tuned to center the modulation so
                                            # that the mean speed stays near the setpoint.
            tension_speed_gain=0.7e-6,      # ~0.035 m/s per 50kN excess tension
            tension_speed_filter_tau=15.0,   # 15s filter — faster response to tension changes
            tension_speed_max_reduction=0.08,  # Max 0.08 m/s reduction → V_min ≈ 0.04
            tension_speed_max_increase=0.08,   # Max 0.08 m/s increase → V_max ≈ 0.20
            # High-tension slowdown (safety override)
            high_tension_slowdown=True,
            tension_stage1=690e3,            # Stage 1: ~70t — begin deceleration
            tension_stage2=834e3,            # Stage 2: ~85t — double decel rate
            tension_estop=980e3,             # E-stop: ~100t — immediate stop
            slowdown_decel_rate=0.005,       # 0.005 m/s^2 — operator-set decel rate
            slowdown_accel_rate=0.005,       # 0.005 m/s^2 — match decel for symmetric recovery
            slowdown_filter_tau=5.0,         # 5s filter on tension for slowdown decisions
            slowdown_min_speed=0.0,          # Allow full stop
        ),
        env=env_cfg,
        track_start=(0.0, 0.0),
        track_end=(duration * speed * 2, 0.0),
        track_speed=speed,
    )
    res_speed = run_simulation(config_speed)
    print_summary(res_speed, config_speed)

    # -----------------------------------------------------------------------
    # Run 2: Position control (moving waypoint)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"SCENARIO: DP Mode Comparison — Position Control")
    print(f"  Reference speed: {speed} m/s (constant, no tension feedback)")
    print(f"{'='*60}")

    # Position control: the reference moves at the target mean speed.
    # Speed variation emerges from catenary tension fighting the PID.
    config_pos = SimulationConfig(
        dt=0.2, duration=duration,
        vessel=make_vessel_config(), wire=wire_cfg,
        plough=make_plough(), soil=soil,
        dp=DPControllerConfig(
            surge_mode='position',
            T_surge=114.0, zeta_surge=0.9,
            tow_force_feedforward=0.7,   # Lower FF → more for PID to fight
            wind_feedforward=0.8,
            # High-tension slowdown (same safety limits as speed control)
            high_tension_slowdown=True,
            tension_stage1=690e3,
            tension_stage2=834e3,
            tension_estop=980e3,
            slowdown_decel_rate=0.005,
            slowdown_accel_rate=0.005,
            slowdown_filter_tau=5.0,
            slowdown_min_speed=0.0,
        ),
        env=env_cfg,
        track_start=(0.0, 0.0),
        track_end=(duration * speed * 2, 0.0),
        track_speed=speed,
    )
    res_pos = run_simulation(config_pos)
    print_summary(res_pos, config_pos)

    return {
        'speed_control': (res_speed, config_speed),
        'position_control': (res_pos, config_pos),
    }


# =============================================================================
# Catenary parameter study
# =============================================================================

def catenary_parameter_study(depths=[15, 50, 100, 200, 500, 1000, 1500],
                              horizontal_forces=[50e3, 100e3, 200e3, 500e3],
                              save_path=None):
    """
    Study catenary behavior across depth and horizontal force range.

    Plots layback, wire angle, and tension at vessel as functions of depth
    and horizontal force.
    """
    wire = TowWireConfig(diameter=0.076, linear_mass=25.0, total_length=4000.0)
    cat = CatenaryModel(wire)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Catenary Parameter Study', fontsize=14, fontweight='bold')

    for F_h in horizontal_forces:
        laybacks = []
        angles = []
        tensions = []
        wire_lengths = []

        for d in depths:
            r = cat.solve_catenary(d, F_h, tow_point_depth=5.0)
            laybacks.append(r['layback'])
            angles.append(np.degrees(r['angle_vessel']))
            tensions.append(r['tension_vessel'] / 1e3)
            wire_lengths.append(r['wire_length'])

        label = f'F_h = {F_h/1e3:.0f} kN'
        axes[0, 0].plot(depths, laybacks, '-o', markersize=4, label=label)
        axes[0, 1].plot(depths, angles, '-o', markersize=4, label=label)
        axes[1, 0].plot(depths, tensions, '-o', markersize=4, label=label)
        axes[1, 1].plot(depths, wire_lengths, '-o', markersize=4, label=label)

    axes[0, 0].set_xlabel('Water depth [m]')
    axes[0, 0].set_ylabel('Layback [m]')
    axes[0, 0].set_title('Layback vs Depth')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Water depth [m]')
    axes[0, 1].set_ylabel('Wire angle [deg]')
    axes[0, 1].set_title('Wire Angle at Vessel vs Depth')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Water depth [m]')
    axes[1, 0].set_ylabel('Tension [kN]')
    axes[1, 0].set_title('Tension at Vessel vs Depth')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Water depth [m]')
    axes[1, 1].set_ylabel('Suspended wire length [m]')
    axes[1, 1].set_title('Wire Length vs Depth')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def run_all_scenarios():
    """Run all predefined scenarios and generate plots."""

    print("\n" + "=" * 70)
    print("   CABLE LAYING PLOUGHING SIMULATION - FULL STUDY")
    print("=" * 70)

    # 1. Catenary parameter study
    print("\n\n--- CATENARY PARAMETER STUDY ---")
    catenary_parameter_study()

    # 2. Steady state at 100m, 25t plough, 0.5 m/s
    print("\n\n--- BASELINE: 100m depth, 25t plough, 0.5 m/s ---")
    res_base, cfg_base = scenario_steady_state(100, 25, 0.5, 600)
    plot_overview(res_base, "Baseline: 100m depth, 25t plough, 0.5 m/s")
    plot_force_breakdown(res_base, "Baseline Force Breakdown")

    # 3. Hard soil encounter
    print("\n\n--- HARD SOIL ENCOUNTER ---")
    res_hard, cfg_hard = scenario_hard_soil(100, 25, 0.5, 200, 3.0, 60, 600)
    plot_overview(res_hard, "Hard Soil: x3 resistance at t=200s")
    plot_force_breakdown(res_hard, "Hard Soil Force Breakdown")

    # 4. Plough stop
    print("\n\n--- PLOUGH STOP ---")
    res_stop, cfg_stop = scenario_plough_stop(100, 25, 0.5, 200, 120, 600)
    plot_overview(res_stop, "Plough Stop at t=200s")
    plot_force_breakdown(res_stop, "Plough Stop Force Breakdown")

    # 5. Depth comparison
    print("\n\n--- DEPTH COMPARISON ---")
    depth_results = scenario_depth_comparison([15, 100, 500, 1500], 25, 0.5, 600)

    # 6. Plough comparison
    print("\n\n--- PLOUGH MASS COMPARISON ---")
    plough_results = scenario_plough_comparison(100, [25, 35], 0.5, 600)

    # 7. Stochastic soil - 1 hour challenging seabed
    print("\n\n--- STOCHASTIC SOIL: 1 HOUR CHALLENGING SEABED ---")
    res_stoch, cfg_stoch = scenario_stochastic_soil()
    plot_stochastic_soil(res_stoch,
        title="Tow tension-based vessel speed variations\n"
              "during ploughing operation in challenging seabed / 1 hour period",
        save_path="results/stochastic_soil_1hr.png")

    return {
        'baseline': (res_base, cfg_base),
        'hard_soil': (res_hard, cfg_hard),
        'plough_stop': (res_stop, cfg_stop),
        'depth_comparison': depth_results,
        'plough_comparison': plough_results,
        'stochastic_soil': (res_stoch, cfg_stoch),
    }
