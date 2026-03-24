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
from .plough import PloughConfig, SoilProperties
from .dp_controller import DPControllerConfig
from .environment import EnvironmentConfig
from .simulation import SimulationConfig, run_simulation, SimulationResult
from .plotting import (plot_overview, plot_force_breakdown, plot_layback_analysis,
                       plot_catenary_snapshot, print_summary)
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


def make_wire_config(water_depth: float) -> TowWireConfig:
    """Tow wire sized for the depth."""
    # Wire length needs to be at least ~1.5x water depth for catenary
    wire_length = max(water_depth * 2.5, 500.0)
    wire_length = min(wire_length, 4000.0)

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
            current_direction=180.0,  # head current
            wind_speed=8.0,
            wind_direction=180.0,     # headwind
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
            current_direction=180.0,
            wind_speed=8.0,
            wind_direction=180.0,
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
            current_direction=180.0,
            wind_speed=8.0,
            wind_direction=180.0,
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

    return {
        'baseline': (res_base, cfg_base),
        'hard_soil': (res_hard, cfg_hard),
        'plough_stop': (res_stop, cfg_stop),
        'depth_comparison': depth_results,
        'plough_comparison': plough_results,
    }
