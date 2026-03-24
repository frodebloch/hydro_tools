"""
Main simulation runner for cable laying ploughing operations.

Ties together the vessel model, DP controller, catenary model, plough model,
and environment to run time-domain simulations of ploughing operations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional

from .vessel import VesselConfig, VesselModel, VesselSimulator
from .catenary import TowWireConfig, CatenaryModel
from .plough import PloughConfig, PloughModel, SoilProperties
from .dp_controller import DPControllerConfig, DPController
from .environment import EnvironmentConfig, EnvironmentModel


@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""
    # Time
    dt: float = 0.1              # Simulation time step [s]
    duration: float = 600.0       # Total simulation time [s]

    # Components
    vessel: VesselConfig = field(default_factory=VesselConfig)
    wire: TowWireConfig = field(default_factory=TowWireConfig)
    plough: PloughConfig = field(default_factory=PloughConfig)
    soil: SoilProperties = field(default_factory=SoilProperties.firm_clay)
    dp: DPControllerConfig = field(default_factory=DPControllerConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    # Track definition
    track_start: tuple = (0.0, 0.0)      # NED start position [m]
    track_end: tuple = (2000.0, 0.0)     # NED end position [m]
    track_speed: float = 0.5              # Desired ploughing speed [m/s]

    # Initial conditions
    initial_heading: float = 0.0          # [deg] vessel heading at start

    # Events: list of (time, event_function) tuples
    events: list = field(default_factory=list)


@dataclass
class SimulationResult:
    """Container for simulation results."""
    time: np.ndarray = None

    # Vessel state
    x: np.ndarray = None           # North position [m]
    y: np.ndarray = None           # East position [m]
    psi: np.ndarray = None         # Heading [rad]
    u: np.ndarray = None           # Surge velocity [m/s]
    v: np.ndarray = None           # Sway velocity [m/s]
    r: np.ndarray = None           # Yaw rate [rad/s]

    # Plough state
    plough_x: np.ndarray = None
    plough_y: np.ndarray = None
    plough_speed: np.ndarray = None

    # Catenary
    wire_tension_vessel: np.ndarray = None
    wire_tension_plough: np.ndarray = None
    wire_angle_vessel: np.ndarray = None
    layback: np.ndarray = None
    wire_length_suspended: np.ndarray = None
    horizontal_force: np.ndarray = None
    vertical_force: np.ndarray = None

    # DP controller
    tau_surge: np.ndarray = None   # Commanded surge thrust [N]
    tau_sway: np.ndarray = None
    tau_yaw: np.ndarray = None
    cross_track_error: np.ndarray = None
    speed_error: np.ndarray = None
    heading_error: np.ndarray = None

    # Forces
    tow_force_surge: np.ndarray = None  # Tow force in body frame
    tow_force_sway: np.ndarray = None
    tow_moment_yaw: np.ndarray = None

    # Plough forces
    plough_soil_force: np.ndarray = None
    plough_friction_force: np.ndarray = None
    plough_total_resistance: np.ndarray = None

    # Environment
    wind_force_surge: np.ndarray = None
    wind_force_sway: np.ndarray = None

    # Safety
    wire_safety_factor: np.ndarray = None

    # Utilization
    thrust_utilization_surge: np.ndarray = None
    thrust_utilization_sway: np.ndarray = None


def run_simulation(config: SimulationConfig,
                   event_callbacks: Optional[List] = None) -> SimulationResult:
    """
    Run a time-domain ploughing simulation.

    Parameters:
        config: Full simulation configuration
        event_callbacks: List of (time, callable) pairs. Callable receives
                        (sim_state_dict) and can modify plough/environment.

    Returns:
        SimulationResult with all time histories
    """
    dt = config.dt
    n_steps = int(config.duration / dt)
    t = np.arange(n_steps) * dt

    # --- Initialize components ---
    vessel_model = VesselModel(config.vessel)
    vessel_sim = VesselSimulator(vessel_model, dt)

    catenary = CatenaryModel(config.wire)
    plough = PloughModel(config.plough, config.soil)
    env = EnvironmentModel(config.env, config.vessel)

    # Set initial vessel position (ahead of track start by correct layback)
    track_dir = np.arctan2(config.track_end[1] - config.track_start[1],
                            config.track_end[0] - config.track_start[0])
    heading = np.radians(config.initial_heading) if config.initial_heading != 0.0 else track_dir

    # Compute initial layback from catenary equilibrium:
    # The plough resistance at the desired speed determines the required
    # horizontal tension T_h, which sets the catenary geometry.
    plough_resistance_initial = plough.total_resistance(config.track_speed)
    T_h_initial = max(plough_resistance_initial, 10e3)  # at least 10 kN
    cat_init = catenary.solve_catenary(
        config.env.water_depth, T_h_initial,
        tow_point_depth=config.vessel.draft
    )
    initial_layback = cat_init['layback']
    # Add tow point offset (tow point is behind vessel CG)
    initial_layback += abs(config.vessel.tow_point_x)

    vessel_x0 = config.track_start[0] + initial_layback * np.cos(track_dir)
    vessel_y0 = config.track_start[1] + initial_layback * np.sin(track_dir)

    vessel_sim.set_position(vessel_x0, vessel_y0, heading)
    vessel_sim.set_velocity(config.track_speed, 0.0, 0.0)

    # Initialize plough at track start, already moving at target speed
    plough.position = np.array([config.track_start[0], config.track_start[1]])
    plough.speed = config.track_speed

    print(f"  Initial layback: {initial_layback:.0f} m "
          f"(catenary: {cat_init['layback']:.0f} m + tow offset: {abs(config.vessel.tow_point_x):.0f} m)")
    print(f"  Initial wire tension at vessel: {cat_init['tension_vessel']/1e3:.1f} kN")
    print(f"  Plough resistance at {config.track_speed} m/s: {plough_resistance_initial/1e3:.1f} kN")

    # DP controller
    vessel_mass = {
        'surge': vessel_model.M_surge,
        'sway': vessel_model.M_sway,
        'yaw': vessel_model.M_yaw,
    }
    dp = DPController(config.dp, vessel_mass, dt)

    # Track: vessel follows a track that is ahead of the plough by the layback
    # The vessel track is offset forward from the plough track
    wp_start = np.array([vessel_x0, vessel_y0])
    track_length = np.sqrt((config.track_end[0] - config.track_start[0])**2 +
                            (config.track_end[1] - config.track_start[1])**2)
    wp_end = np.array([
        config.track_start[0] + (track_length + initial_layback) * np.cos(track_dir),
        config.track_start[1] + (track_length + initial_layback) * np.sin(track_dir),
    ])
    dp.set_track([wp_start.tolist(), wp_end.tolist()], config.track_speed)
    dp.set_thrust_limits(config.vessel.max_thrust_surge,
                         config.vessel.max_thrust_sway,
                         config.vessel.max_thrust_yaw)

    # --- Allocate result arrays ---
    res = SimulationResult()
    res.time = t
    for attr in ['x', 'y', 'psi', 'u', 'v', 'r',
                 'plough_x', 'plough_y', 'plough_speed',
                 'wire_tension_vessel', 'wire_tension_plough',
                 'wire_angle_vessel', 'layback', 'wire_length_suspended',
                 'horizontal_force', 'vertical_force',
                 'tau_surge', 'tau_sway', 'tau_yaw',
                 'cross_track_error', 'speed_error', 'heading_error',
                 'tow_force_surge', 'tow_force_sway', 'tow_moment_yaw',
                 'plough_soil_force', 'plough_friction_force', 'plough_total_resistance',
                 'wind_force_surge', 'wind_force_sway',
                 'wire_safety_factor',
                 'thrust_utilization_surge', 'thrust_utilization_sway']:
        setattr(res, attr, np.zeros(n_steps))

    # Sort events by time
    events = sorted(config.events, key=lambda e: e[0]) if config.events else []
    event_idx = 0

    # --- Simulation loop ---
    for i in range(n_steps):
        ti = t[i]

        # --- Process events ---
        while event_idx < len(events) and events[event_idx][0] <= ti:
            event_func = events[event_idx][1]
            event_func(plough=plough, env=env, catenary=catenary,
                       vessel_sim=vessel_sim, dp=dp, config=config)
            event_idx += 1

        if event_callbacks:
            for evt_time, evt_func in event_callbacks:
                if abs(ti - evt_time) < dt * 0.5:
                    evt_func(plough=plough, env=env, catenary=catenary,
                             vessel_sim=vessel_sim, dp=dp, config=config)

        # --- Environment ---
        wind_force = env.wind_force_body(vessel_sim.psi)
        current_body = env.current_force_body(vessel_sim.u, vessel_sim.v,
                                               vessel_sim.psi, vessel_model)
        inline_current = env.inline_current_speed(track_dir)

        # --- Catenary: solve for wire tension given positions ---
        tow_point = vessel_sim.tow_point_ned()
        tow_depth = config.vessel.draft  # approximate tow point depth

        cat_result = catenary.solve_for_plough_position(
            water_depth=config.env.water_depth,
            vessel_tow_point=tow_point,
            plough_position=plough.position,
            current_speed=inline_current,
            tow_point_depth=tow_depth,
        )

        # Wire force at vessel in body frame
        wire_force_body = catenary.vessel_force_body_frame(cat_result, vessel_sim.psi)

        # Add yaw moment from tow point offset
        tow_point_body = vessel_model.tow_point_body()
        yaw_moment = catenary.compute_yaw_moment(wire_force_body, tow_point_body)
        wire_force_body[2] = yaw_moment

        # The tow force on the vessel is pulling it toward the plough (aft)
        # This is an external force on the vessel
        tow_force_on_vessel = wire_force_body.copy()

        # --- Plough: compute resistance and update position ---
        plough_resistance = plough.total_resistance(plough.speed, inline_current)

        # Plough speed is governed by the horizontal wire tension at the plough
        # minus the plough resistance. If wire tension < resistance, plough decelerates.
        wire_horizontal_at_plough = cat_result['tension_plough']

        if not plough.is_stopped:
            # Simple plough dynamics: F = ma
            # Net force = wire tension (horizontal) - resistance
            F_net = wire_horizontal_at_plough - plough_resistance
            plough_mass = config.plough.mass
            plough_accel = F_net / plough_mass
            # Limit acceleration
            plough_accel = np.clip(plough_accel, -2.0, 2.0)
            new_speed = plough.speed + plough_accel * dt
            new_speed = max(new_speed, 0.0)  # plough can't go backwards
            plough.step(new_speed, track_dir, dt)
        else:
            plough.step(0.0, track_dir, dt)

        # --- DP controller ---
        tau_dp = dp.step(
            vessel_sim.x, vessel_sim.y, vessel_sim.psi,
            vessel_sim.u, vessel_sim.v, vessel_sim.r,
            tow_force_body=tow_force_on_vessel,
            wind_force_body=wind_force,
        )

        # --- Vessel dynamics ---
        # Total external forces = tow + wind (current modifies relative velocity)
        tau_external = tow_force_on_vessel + wind_force

        # Adjust vessel velocities for current (speed through water)
        # The hydrodynamic forces use speed through water
        # u_water = u - current_body_x, v_water = v - current_body_y
        # For simplicity, we add current as an external force approximation
        # A more rigorous approach modifies the velocity inputs to the hydro model

        vessel_sim.step(tau_external, tau_dp)

        # --- Store results ---
        res.x[i] = vessel_sim.x
        res.y[i] = vessel_sim.y
        res.psi[i] = vessel_sim.psi
        res.u[i] = vessel_sim.u
        res.v[i] = vessel_sim.v
        res.r[i] = vessel_sim.r

        res.plough_x[i] = plough.position[0]
        res.plough_y[i] = plough.position[1]
        res.plough_speed[i] = plough.speed

        res.wire_tension_vessel[i] = cat_result['tension_vessel']
        res.wire_tension_plough[i] = cat_result['tension_plough']
        res.wire_angle_vessel[i] = np.degrees(cat_result['angle_vessel'])
        res.layback[i] = cat_result['layback']
        res.wire_length_suspended[i] = cat_result['wire_length']
        res.horizontal_force[i] = cat_result['horizontal_force']
        res.vertical_force[i] = cat_result['vertical_force']

        res.tau_surge[i] = tau_dp[0]
        res.tau_sway[i] = tau_dp[1]
        res.tau_yaw[i] = tau_dp[2]

        errors = dp.compute_track_errors(vessel_sim.x, vessel_sim.y, vessel_sim.psi,
                                          vessel_sim.u, vessel_sim.v)
        res.cross_track_error[i] = errors['cross_track']
        res.speed_error[i] = errors['speed_error']
        res.heading_error[i] = errors.get('heading_error', 0.0)

        res.tow_force_surge[i] = tow_force_on_vessel[0]
        res.tow_force_sway[i] = tow_force_on_vessel[1]
        res.tow_moment_yaw[i] = tow_force_on_vessel[2]

        forces = plough.force_summary(plough.speed, inline_current)
        res.plough_soil_force[i] = forces['soil_cutting']
        res.plough_friction_force[i] = forces['skid_friction']
        res.plough_total_resistance[i] = forces['total']

        res.wind_force_surge[i] = wind_force[0]
        res.wind_force_sway[i] = wind_force[1]

        res.wire_safety_factor[i] = cat_result['safety_factor']

        res.thrust_utilization_surge[i] = abs(tau_dp[0]) / config.vessel.max_thrust_surge
        res.thrust_utilization_sway[i] = abs(tau_dp[1]) / config.vessel.max_thrust_sway

    return res
