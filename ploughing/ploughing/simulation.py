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
    wave_force_surge: np.ndarray = None
    wave_force_sway: np.ndarray = None

    # Safety
    wire_safety_factor: np.ndarray = None

    # Utilization
    thrust_utilization_surge: np.ndarray = None
    thrust_utilization_sway: np.ndarray = None

    # DP control action breakdown (surge axis only)
    surge_feedforward: np.ndarray = None    # Feedforward component [N]
    surge_pid: np.ndarray = None            # PID/PI feedback component [N]

    # High-tension slowdown state
    slowdown_speed_ceiling: np.ndarray = None   # Speed ceiling from slowdown system [m/s]
    slowdown_state: np.ndarray = None           # State: 0=NORMAL, 1=STAGE1, 2=STAGE2, 3=ESTOP


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

    # Quadratic soil-inertial damping coefficient (defined here for use in
    # both the equilibrium calculation and the simulation loop).
    # See detailed comments below where _C_soil_drag is also referenced.
    _C_soil_drag = 3000e3  # [N/(m/s)^2] — see comments below
    _V_ref_drag = config.track_speed  # reference speed for drag baseline

    # Compute initial layback from catenary equilibrium:
    # The plough resistance at the desired speed determines the required
    # horizontal tension T_h, which sets the catenary geometry.
    # The soil-inertial drag is zero at V_ref by construction (see below),
    # so no drag correction is needed in the equilibrium calculation.
    plough_resistance_initial = plough.total_resistance(config.track_speed)
    T_h_initial = max(plough_resistance_initial, 10e3)
    cat_init = catenary.solve_catenary(
        config.env.water_depth, T_h_initial,
        tow_point_depth=config.vessel.draft
    )
    print(f"  Plough resistance at {config.track_speed} m/s: {plough_resistance_initial/1e3:.1f} kN")
    print(f"  Inextensible catenary: layback={cat_init['layback']:.0f}m, "
          f"T_vessel={cat_init['tension_vessel']/1e3:.1f} kN")

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
                 'wave_force_surge', 'wave_force_sway',
                 'wire_safety_factor',
                 'thrust_utilization_surge', 'thrust_utilization_sway',
                 'surge_feedforward', 'surge_pid',
                 'slowdown_speed_ceiling', 'slowdown_state']:
        setattr(res, attr, np.zeros(n_steps))

    # Sort events by time
    events = sorted(config.events, key=lambda e: e[0]) if config.events else []
    event_idx = 0

    # Plough dynamics: effective mass for Newton's 2nd law (along-track).
    # A plough cutting a furrow through saturated soil entrains a large
    # volume of soil + water that must be accelerated with the plough body.
    # The entrained mass includes:
    #   - Soil in the active failure wedge ahead of the share (~2-4x plough mass)
    #   - Water in the furrow behind and around the plough body
    #   - Hydrodynamic added mass of a bluff body near the seabed
    # Studies on subsea ploughs (Cathie & Wintgens 2001, Palmer & King 2008)
    # suggest effective mass factors of 3-8x dry mass in soft-firm clays.
    # We use 5x as a representative value for firm clay conditions.
    # This high effective mass provides the natural mechanical low-pass
    # filtering that prevents the plough from responding to every spatial
    # fluctuation in soil resistance — matching the observed smooth plough
    # speed response in operational data.
    plough_mass = config.plough.mass * 5.0

    # Soil-inertial damping: at higher speeds, the plough must displace
    # more soil per unit time.  The volume flow rate of displaced soil
    # scales with V, and the dynamic pressure on the plough body scales
    # with V^2.  This gives a quadratic damping term:
    #   F_soil_drag = C_soil * max(0, V^2 - V_ref^2)
    # where C_soil [N/(m/s)^2] is calibrated so the damping is zero at
    # the operating speed (V_ref) and grows rapidly above it.
    #
    # The drag at V_ref is already implicitly included in the calibrated
    # base resistance F_base — there's no need to add it again.  The
    # excess drag term captures the ADDITIONAL dynamic soil loading that
    # occurs when the plough moves faster than the calibration speed:
    # inertial effects from accelerating more soil per unit time, increased
    # dynamic pressure on the failure wedge, and seabed suction.
    #
    # At V=0.12 (V_ref): F = 0        (absorbed into F_base)
    # At V=0.18:          F = C * 0.018 ≈ 108 kN  (moderate)
    # At V=0.24:          F = C * 0.043 ≈ 259 kN  (significant)
    # At V=0.30:          F = C * 0.076 ≈ 454 kN  (dominant — limits speed)
    #
    # Physical basis: a 3m wide × 1.5m deep plough displaces ~4.5 m^2
    # cross-section of saturated soil (rho ~ 2000 kg/m^3).  Dynamic
    # pressure loading: F ~ 0.5 * rho * A * V^2 * C_d with C_d ~ 3-5
    # for bluff body in granular flow, plus seabed suction effects.
    # _C_soil_drag is defined above (before equilibrium calculation).

    # --- Pre-compute catenary spring table: T_h(layback) ---
    # The catenary acts as a nonlinear spring between vessel and plough.
    # For a given water depth and tow geometry, there is a one-to-one
    # mapping between horizontal tension T_h and horizontal distance D.
    # We tabulate this at initialization and interpolate during the sim.
    #
    # The wire is modelled as a taut elastic catenary with fixed deployed
    # (unstretched) length L0.  For a given horizontal distance D between
    # tow point and plough, the catenary shape requires a specific arc
    # length s(D), which must equal the elastically stretched wire:
    #   s(D) = L0 * (1 + T/EA)
    #
    # We use the parabolic cable approximation for the sag correction
    # (valid for nearly-taut cables with sag/span << 1):
    #   chord = sqrt(D^2 + h^2)
    #   w_n = w * cos(alpha)          (weight component normal to chord)
    #   s = chord + w_n^2 * chord^3 / (24 * T^2)   (arc length with sag)
    #   s = L0 * (1 + T / EA)        (elastic compatibility)
    #
    # Solve for T, then T_h = T * cos(alpha).
    #
    # The resulting spring is asymmetric:
    #   - D < D_natural (slack): wire sags more, tension drops (sag-dominated)
    #   - D > D_natural (taut): sag reduces, then EA/L stiffness takes over
    # This matches real operational data: broad tension dips in soft soil
    # and sharp spikes when the wire goes taut in hard soil.
    _h = config.env.water_depth - config.vessel.draft
    _w = catenary.wire.submerged_weight
    _EA = catenary.wire.axial_stiffness  # EA of wire [N]

    # --- Determine deployed wire length L0 ---
    # The deployed length is constrained by the physical wire on the drum.
    # In shallow water, all wire may be deployed; in deep water, only
    # the catenary-required portion is suspended.
    _wire_total = catenary.wire.total_length
    _a_init = T_h_initial / _w
    _s_catenary = _a_init * np.sinh(cat_init['layback'] / _a_init)

    if _wire_total < _s_catenary:
        # Wire is shorter than the inextensible catenary needs.
        # All wire is deployed.  The "natural" inextensible equilibrium
        # T_h for this wire length is lower than the plough resistance,
        # so the wire operates in its taut/elastic regime.
        _s_deployed = _wire_total
        _a_natural = (_s_deployed**2 - _h**2) / (2.0 * _h)
        _T_h_natural = _a_natural * _w
        _D_natural = _a_natural * np.arccosh(1.0 + _h / _a_natural)
        print(f"  Wire fully deployed: {_s_deployed:.0f}m "
              f"(drum={_wire_total:.0f}m, catenary needs {_s_catenary:.0f}m)")
        print(f"  Natural catenary: T_h={_T_h_natural/1e3:.1f} kN ({_T_h_natural/9810:.1f}t), "
              f"D={_D_natural:.1f}m")
        print(f"  Plough resistance {T_h_initial/1e3:.1f} kN > natural T_h → "
              f"elastic stretch provides extra tension")
    else:
        # Wire is longer than needed — only the catenary portion is suspended.
        _s_deployed = _s_catenary
        _D_natural = cat_init['layback']
        _T_h_natural = T_h_initial

    # Unstretched wire length (remove elastic stretch at initial tension)
    _T_mean_deployed = T_h_initial * np.cosh(
        _w * _s_deployed / (2.0 * T_h_initial))
    _L0 = _s_deployed / (1.0 + _T_mean_deployed / _EA)
    _grounded_wire = _wire_total - _s_deployed
    print(f"  Wire: deployed={_s_deployed:.1f}m, L0={_L0:.1f}m, "
          f"stretch={_s_deployed - _L0:.3f}m, grounded={_grounded_wire:.0f}m")

    from scipy.optimize import brentq as _brentq

    def _solve_Th_at_D(_D):
        """Solve elastic catenary for T_h at horizontal distance D."""
        _chord = np.sqrt(_D**2 + _h**2)
        _cos_alpha = _D / _chord
        _w_n = _w * _cos_alpha

        def _sag_residual(_T):
            if _T < 10.0:
                return 1e10
            _s_geom = _chord + _w_n**2 * _chord**3 / (24.0 * _T**2)
            _s_elastic = _L0 * (1.0 + _T / _EA)
            return _s_geom - _s_elastic

        try:
            _T_total = _brentq(_sag_residual, 10.0, 50e6, rtol=1e-10)
        except ValueError:
            _T_total = _w * _D**2 / (2.0 * _h)
        return max(_T_total * _cos_alpha, 1e3)

    # Find the equilibrium D where T_h matches plough resistance.
    # Search around D_natural (the inextensible equilibrium span).
    def _eq_residual(_D):
        return _solve_Th_at_D(_D) - T_h_initial

    # Bracket the root: at D_natural, T_h < T_h_initial (if wire is taut)
    # or T_h ≈ T_h_initial (if wire is long enough).  Search taut side.
    _D_search_lo = max(_h + 1.0, _D_natural - 20.0)
    _D_search_hi = _D_natural + 10.0
    # Ensure we bracket the root
    while _solve_Th_at_D(_D_search_hi) < T_h_initial:
        _D_search_hi += 5.0
        if _D_search_hi > _L0 * 2:
            break
    while _solve_Th_at_D(_D_search_lo) > T_h_initial:
        _D_search_lo -= 10.0
        if _D_search_lo < _h + 1:
            _D_search_lo = _h + 1.0
            break

    try:
        _D_eq = _brentq(_eq_residual, _D_search_lo, _D_search_hi, rtol=1e-8)
    except ValueError:
        _D_eq = _D_natural  # fallback

    # --- Build spring table covering full operational range ---
    # With the wire length set to catenary arc + small buffer (~75m),
    # there is minimal grounded wire at operating tension.  The elastic
    # cable model handles the full operational range without needing a
    # grounding/pay-off model.
    #
    # Slack side: need to cover D down to where T_h is very low.
    #   Use ~2% of equilibrium T_h as the floor.
    # Taut side: need to cover the full elastic stretch range.
    #   At maximum soil force (~5x base), dD ≈ (T_max - T_eq) / k_eq.
    #   With k~500 kN/m, dD ~= 1500/500 = 3m.  Use 10m for safety.
    _T_h_min_table = max(T_h_initial * 0.02, _w * _h * 1.01)  # floor: just above wire weight
    _a_min = _T_h_min_table / _w
    _D_min_table = _a_min * np.arccosh(1.0 + _h / _a_min) if _h / _a_min < 700 else _h + 1.0
    _D_min_table = max(_D_min_table, _h + 1.0)

    # Taut side: 10m beyond equilibrium (EA/L stiffness makes it very steep)
    _D_max_table = _D_eq + 10.0

    _n_table = 2000
    _D_table = np.linspace(_D_min_table, _D_max_table, _n_table)
    _T_h_table = np.zeros(_n_table)

    for _k in range(_n_table):
        _T_h_table[_k] = _solve_Th_at_D(_D_table[_k])

    # Verify monotonicity
    _dT = np.diff(_T_h_table)
    if not np.all(_dT >= 0):
        for _k in range(1, _n_table):
            _T_h_table[_k] = max(_T_h_table[_k], _T_h_table[_k - 1])

    # Create interpolator
    def cat_spring(horizontal_distance):
        """Catenary nonlinear spring: horizontal tension from horizontal distance."""
        return np.interp(horizontal_distance, _D_table, _T_h_table)

    # Verify: elastic spring at D_eq should give ~initial T_h
    _T_h_check = cat_spring(_D_eq)
    # Stiffness at the equilibrium point
    _D_eq_idx = np.searchsorted(_D_table, _D_eq)
    _D_eq_idx = min(max(_D_eq_idx, 1), _n_table - 2)
    _stiffness_at_init = (_T_h_table[_D_eq_idx + 1] - _T_h_table[_D_eq_idx - 1]) / \
                         (_D_table[_D_eq_idx + 1] - _D_table[_D_eq_idx - 1])
    print(f"  Elastic catenary equilibrium: D_eq={_D_eq:.1f}m → "
          f"T_h={_T_h_check/1e3:.1f} kN (target {T_h_initial/1e3:.1f} kN)")
    print(f"  Spring stiffness at equilibrium: {_stiffness_at_init/1e3:.1f} kN/m")
    print(f"  Spring table: D=[{_D_table[0]:.0f}, {_D_table[-1]:.0f}]m, "
          f"T_h=[{_T_h_table[0]/9810:.1f}, {_T_h_table[-1]/9810:.1f}]t")

    # --- Set initial vessel and plough positions ---
    # Use D_eq from the elastic spring (may differ from inextensible catenary layback)
    initial_layback = _D_eq + abs(config.vessel.tow_point_x)

    vessel_x0 = config.track_start[0] + initial_layback * np.cos(track_dir)
    vessel_y0 = config.track_start[1] + initial_layback * np.sin(track_dir)

    vessel_sim.set_position(vessel_x0, vessel_y0, heading)
    vessel_sim.set_velocity(config.track_speed, 0.0, 0.0)

    plough.position = np.array([config.track_start[0], config.track_start[1]])
    plough.speed = config.track_speed

    print(f"  Initial layback: {initial_layback:.0f}m "
          f"(wire span: {_D_eq:.0f}m + tow offset: {abs(config.vessel.tow_point_x):.0f}m)")

    # --- DP controller ---
    vessel_mass = {
        'surge': vessel_model.M_surge,
        'sway': vessel_model.M_sway,
        'yaw': vessel_model.M_yaw,
    }
    dp = DPController(config.dp, vessel_mass, dt)

    # Track: vessel follows a track that is ahead of the plough by the layback
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

    # --- First-order wave surge motion ---
    # Real DP systems use a wave filter (notch filter at wave frequencies)
    # to exclude first-order vessel motion from the control feedback loop.
    # The DP controller operates on the low-frequency (slowly-varying)
    # vessel position, while the vessel physically oscillates at wave
    # frequency without the DP trying to counteract it.
    #
    # We model this by adding first-order surge displacement directly
    # to the tow point position used for catenary lookup, bypassing the
    # DP controller.  This produces the correct physics:
    #   - Vessel oscillates at wave period (Tp ~ 7-10s)
    #   - Layback modulated at wave frequency → tension oscillation
    #   - DP does not fight the wave motion (wave filter effect)
    #   - Plough barely responds (125t inertia, too slow for wave period)
    #   - Result: "hairy" tension texture matching real operational data
    #
    # Surge amplitude from linear wave theory:
    #   x_amp = RAO_surge * Hs/2
    # For a 130m cable layer at Tp=7.5s (L/Lpp~0.68), RAO_surge ~ 0.3-0.5.
    # With Hs=1.2m: x_amp ~ 0.18-0.30 m.  The wave_surge_amplitude config
    # parameter sets x_amp directly [m], derived from RAO * Hs/2.
    _wave_cfg = config.env.waves
    _Hs = _wave_cfg.Hs
    _Tp = _wave_cfg.Tp
    _omega_wave = 2.0 * np.pi / _Tp if _Tp > 0.1 else 0.0
    # Surge displacement amplitude [m]:
    # first_order_surge_factor is RAO_surge [m/m], multiplied by Hs/2
    _surge_amp = _wave_cfg.first_order_surge_factor * _Hs / 2.0
    _wave_phase = env._wave_phase  # random initial phase from environment
    _has_wave_surge = _surge_amp > 0.001

    if _has_wave_surge:
        print(f"  First-order wave surge: amplitude={_surge_amp:.3f} m at Tp={_Tp:.1f} s")
        print(f"    Tension modulation: ±{_stiffness_at_init * _surge_amp / 1e3:.1f} kN "
              f"= ±{_stiffness_at_init * _surge_amp / 9810:.1f} t at wave frequency")

    # --- Simulation loop ---
    for i in range(n_steps):
        ti = t[i]

        # Mark new timestep for plough stochastic state caching.
        # Compute distance the plough moved this step (spatial soil indexing).
        distance_step = max(plough.speed, 0.0) * dt
        plough.advance_step(distance_step)

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
        wave_force = env.wave_drift_force_body(vessel_sim.psi, dt)
        current_body = env.current_force_body(vessel_sim.u, vessel_sim.v,
                                               vessel_sim.psi, vessel_model)
        inline_current = env.inline_current_speed(track_dir)

        # --- Plough: compute resistance at current plough speed ---
        plough_resistance = plough.total_resistance(plough.speed, inline_current)

        # --- Catenary: tabulated nonlinear spring ---
        # T_h is looked up from the pre-computed table based on the actual
        # horizontal separation between vessel tow point and plough.
        tow_point = vessel_sim.tow_point_ned()
        tow_depth = config.vessel.draft

        # Add first-order wave surge displacement to the tow point.
        # This is the vessel's physical oscillation at wave frequency;
        # the DP controller does not see it (wave filter effect).
        # Applied along the track direction (head seas assumed for surge).
        if _has_wave_surge:
            wave_surge_disp = _surge_amp * np.sin(_omega_wave * ti + _wave_phase)
            tow_point[0] += wave_surge_disp * np.cos(track_dir)
            tow_point[1] += wave_surge_disp * np.sin(track_dir)

        dx = tow_point[0] - plough.position[0]
        dy = tow_point[1] - plough.position[1]
        horizontal_distance = np.sqrt(dx**2 + dy**2)

        T_h = cat_spring(horizontal_distance)

        # Compute vessel-end tension and angle from elastic cable geometry.
        # The inextensible catenary solve (solve_catenary) gives wrong
        # angles/tensions when T_h is low because it assumes the wire
        # freely sags to match T_h, giving a short span.  Our elastic
        # cable has fixed length and spans the actual horizontal_distance.
        #
        # For a taut cable of length ~L0 spanning height h:
        #   - Vertical force at vessel = total wire weight = w * L0
        #   - Horizontal force = T_h (constant along wire)
        #   - Tension at vessel = sqrt(T_h^2 + (w*L0)^2)
        #   - Angle at vessel = atan2(w*L0, T_h)  (from horizontal)
        #   Note: this slightly overestimates the vertical force (should
        #   be w*s_above_plough, not w*L0) but the error is small for
        #   our nearly-flat cable.
        _V_wire = _w * _L0  # vertical force = total wire weight
        T_vessel = np.sqrt(T_h**2 + _V_wire**2)
        angle_vessel = np.arctan2(_V_wire, T_h)

        # Build a cat_result-like dict for downstream code
        cat_result = {
            'tension_vessel': T_vessel,
            'tension_plough': T_h,
            'angle_vessel': angle_vessel,
            'angle_plough': np.arctan2(_h, horizontal_distance),
            'horizontal_force': T_h,
            'vertical_force': _V_wire,
            'layback': horizontal_distance,  # actual horizontal distance
            'wire_length': _L0,
            'wire_x': np.array([0.0, horizontal_distance]),
            'wire_z': np.array([0.0, _h]),
            'grounded_length': 0.0,
            'safety_factor': catenary.wire.breaking_load / T_vessel if T_vessel > 0 else float('inf'),
        }

        # Wire force at vessel in body frame.
        # The catenary pulls the vessel aft along the track direction.
        # Vessel-end tension from catenary (accounts for wire weight/geometry).
        T_vessel = cat_result['tension_vessel']
        angle_vessel = cat_result['angle_vessel']  # angle from horizontal at vessel

        # Force on vessel in NED: along track direction, pulling aft
        wire_force_ned = np.array([
            -T_vessel * np.cos(angle_vessel) * np.cos(track_dir),
            -T_vessel * np.cos(angle_vessel) * np.sin(track_dir),
            0.0
        ])
        # Transform to body frame
        cos_psi = np.cos(vessel_sim.psi)
        sin_psi = np.sin(vessel_sim.psi)
        wire_force_body = np.array([
            wire_force_ned[0] * cos_psi + wire_force_ned[1] * sin_psi,
            -wire_force_ned[0] * sin_psi + wire_force_ned[1] * cos_psi,
            0.0
        ])
        # Add yaw moment from tow point offset
        tow_point_body = vessel_model.tow_point_body()
        yaw_moment = catenary.compute_yaw_moment(wire_force_body, tow_point_body)
        wire_force_body[2] = yaw_moment

        tow_force_on_vessel = wire_force_body.copy()

        # --- Plough dynamics: Newton's 2nd law along track ---
        #     m * dV/dt = T_h(D) - F_resist(V) - F_soil_drag(V)
        #
        # Implicit Euler with coupled catenary spring:
        #   The catenary tension T_h depends on the horizontal distance D
        #   between vessel tow point and plough.  If the plough speed
        #   differs from the vessel speed, D changes within the timestep:
        #     D_new = D_old - (V_new - V_vessel) * dt
        #   We include this coupling in the implicit equation so that
        #   the spring provides negative feedback within a single step:
        #   when the plough speeds up, D shrinks, T_h drops, limiting
        #   the speed excursion.
        #
        #   g(V) = m/dt * (V - V_old)
        #        + F_resist(V) + C_soil * V^2
        #        - T_h(D_old - (V - V_vessel) * dt) = 0
        #
        # g is monotonically increasing in V (inertia + resistance increase
        # with V, and T_h decreases with V via the spring coupling), so
        # Newton's method converges reliably.
        _V_max_plough = config.track_speed * 3.0
        _V_vessel = vessel_sim.u  # vessel surge speed (along track)
        if not plough.is_stopped:
            V_old = plough.speed
            m_over_dt = plough_mass / dt

            def _g(V):
                # Coupled spring: predict where the plough would be
                D_predicted = horizontal_distance - (V - _V_vessel) * dt
                T_h_predicted = cat_spring(D_predicted)
                # Soil-inertial drag: excess above V_ref^2
                F_soil_drag = _C_soil_drag * max(0.0, V**2 - _V_ref_drag**2)
                return (m_over_dt * (V - V_old)
                        + plough.total_resistance(max(V, 0.0), inline_current)
                        + F_soil_drag
                        - T_h_predicted)

            # Newton iterations (typically converges in 2-4 steps)
            V_new = V_old  # initial guess
            for _iter in range(12):
                g_val = _g(V_new)
                # Numerical derivative dg/dV
                _dV = max(abs(V_new) * 1e-4, 1e-6)
                dg = (_g(V_new + _dV) - g_val) / _dV
                if abs(dg) < 1e-10:
                    break
                V_new = V_new - g_val / dg
                V_new = max(V_new, 0.0)  # speed >= 0
                if abs(g_val) < 1.0:     # converged to < 1 N
                    break

            V_new = min(V_new, _V_max_plough)
            plough.step(V_new, track_dir, dt)
        else:
            plough.step(0.0, track_dir, dt)

        # --- DP controller ---
        tau_dp = dp.step(
            vessel_sim.x, vessel_sim.y, vessel_sim.psi,
            vessel_sim.u, vessel_sim.v, vessel_sim.r,
            tow_force_body=tow_force_on_vessel,
            wind_force_body=wind_force,
            tow_tension_horizontal=cat_result['horizontal_force'],
        )

        # --- Vessel dynamics ---
        # Total external forces = tow + wind + waves (current modifies relative velocity)
        tau_external = tow_force_on_vessel + wind_force + wave_force

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

        res.wave_force_surge[i] = wave_force[0]
        res.wave_force_sway[i] = wave_force[1]

        res.wire_safety_factor[i] = cat_result['safety_factor']

        res.thrust_utilization_surge[i] = abs(tau_dp[0]) / config.vessel.max_thrust_surge
        res.thrust_utilization_sway[i] = abs(tau_dp[1]) / config.vessel.max_thrust_sway

        # Control action breakdown (pre-saturation/rate-limiting components)
        res.surge_feedforward[i] = dp._last_surge_ff
        res.surge_pid[i] = dp._last_surge_pid

        # High-tension slowdown state
        res.slowdown_speed_ceiling[i] = dp._slowdown_speed
        _state_map = {'NORMAL': 0, 'STAGE1': 1, 'STAGE2': 2, 'ESTOP': 3}
        res.slowdown_state[i] = _state_map.get(dp._slowdown_state, 0)

    return res
