"""Main entry point — argument parsing, component wiring, and animation loop."""

import argparse
import sys
import time

import numpy as np

from .wave_model import WaveSpectrum, WaveElevation, default_frequencies, default_directions
from .ocean_surface import OceanSurface
from .vessel_geometry import VesselGeometry
from .turbine_geometry import TurbineGeometry
from .scene import Scene
from .udp_receiver import UdpReceiver, SimulatorState, GangwayStateData
from .mock_data import MockDataGenerator


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="DP Simulator 3D Visualization — real-time engineering "
        "visualization of vessel, floating turbine, and ocean surface.",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Run with synthetic demo data (no dp_simulator needed).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=9000,
        help="UDP listen port for JSON data from VisualisationInterface (default: 9000).",
    )
    p.add_argument(
        "--ocean-size",
        type=float,
        default=300.0,
        help="Side length of the ocean patch in meters (default: 300).",
    )
    p.add_argument(
        "--grid-res",
        type=int,
        default=80,
        help="Ocean grid resolution — vertices per side (default: 80).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Target update rate in Hz (default: 15).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for wave phases in mock mode (default: 42).",
    )
    return p.parse_args(argv)


def run(args):
    """Set up all components and start the visualization."""

    print("DP Simulator Visualization")
    print("==========================")
    print(f"Mode:       {'Mock data' if args.mock else f'UDP port {args.port}'}")
    print(f"Ocean:      {args.ocean_size:.0f}m x {args.ocean_size:.0f}m, "
          f"{args.grid_res}x{args.grid_res} grid")
    print(f"Target FPS: {args.fps:.0f}")
    print()
    print("Controls:")
    print("  Mouse drag  — orbit camera")
    print("  Scroll      — zoom")
    print("  Shift+drag  — pan")
    print("  f           — toggle follow vessel")
    print("  r           — reset camera")
    print("  q           — quit")
    print()

    # ── Data source ────────────────────────────────────────────────
    udp_receiver = None
    mock_gen = None

    if args.mock:
        mock_gen = MockDataGenerator(random_seed=args.seed)
        wave_elevation = mock_gen.wave_elevation
        state = mock_gen.step()
    else:
        udp_receiver = UdpReceiver(port=args.port)
        # Start with default wave field until we receive parameters
        freqs = default_frequencies(n=30, w_min=0.25, w_max=2.0)
        dirs = default_directions(n=36)
        wave_elevation = WaveElevation(freqs, dirs, random_seed=args.seed)
        wave_spec = WaveSpectrum(hs=1.5, tp=8.0, direction_deg=180.0, spreading_factor=2.0)
        wave_elevation.add_spectrum(wave_spec)
        state = udp_receiver.state

    # ── Build geometry ─────────────────────────────────────────────
    ocean = OceanSurface(
        wave_elevation=wave_elevation,
        size=args.ocean_size,
        resolution=args.grid_res,
    )
    vessel = VesselGeometry()
    turbine = TurbineGeometry()

    # Initial positions
    ocean.update(0.0)
    vessel.update_transform(
        north=state.vessel_north,
        east=state.vessel_east,
        heading_deg=state.vessel_heading,
        roll_deg=state.vessel_roll,
        pitch_deg=state.vessel_pitch,
        heave=state.vessel_heave,
    )
    vessel.gangway.update_transforms(
        vessel_north=state.vessel_north,
        vessel_east=state.vessel_east,
        vessel_heading_deg=state.vessel_heading,
        vessel_roll_deg=state.vessel_roll,
        vessel_pitch_deg=state.vessel_pitch,
        vessel_heave=state.vessel_heave,
        gangway_state=state.gangway_state,
    )
    turbine.update_transform(
        north=state.platform_north,
        east=state.platform_east,
        heading_deg=state.platform_heading,
        roll_deg=state.platform_roll,
        pitch_deg=state.platform_pitch,
        heave=state.platform_heave,
    )

    # ── Build scene ────────────────────────────────────────────────
    scene = Scene(
        ocean=ocean,
        vessel=vessel,
        turbine=turbine,
        ocean_size=args.ocean_size,
        update_rate_hz=args.fps,
    )

    # ── Animation callback ─────────────────────────────────────────
    frame_count = [0]
    last_fps_time = [time.time()]
    last_frame_time = [time.time()]
    fps_display = [0.0]

    # Track previous wave params to avoid unnecessary rebuilds.
    # The C++ side resends periodically (for late-joining clients),
    # but we only rebuild when something actually changed.
    prev_wave_key = [None]

    def on_update():
        """Called each frame by the scene timer."""
        # Get latest state
        if mock_gen:
            st = mock_gen.step()
        else:
            udp_receiver.poll()
            st = udp_receiver.state
            # If wave params message arrived, check if anything actually changed
            if st.wave_params_updated and st.frequencies:
                wave_key = (
                    st.random_seed,
                    tuple(st.frequencies),
                    tuple(st.directions),
                    st.swell.significant_wave_height,
                    st.swell.peak_period,
                    st.swell.direction_deg,
                    st.swell.spreading_factor,
                    st.wave.significant_wave_height,
                    st.wave.peak_period,
                    st.wave.direction_deg,
                    st.wave.spreading_factor,
                )
                st.wave_params_updated = False
                if wave_key != prev_wave_key[0]:
                    prev_wave_key[0] = wave_key
                    wave_elevation.clear_spectra()
                    freqs_arr = np.array(st.frequencies)
                    dirs_arr = np.array(st.directions)
                    # Rebuild elevation model with new params.
                    # Phase generation uses RandomState(seed) + libstdc++
                    # generate_canonical formula to produce phases identical
                    # to the C++ simulator.
                    wave_elevation.__init__(freqs_arr, dirs_arr, st.random_seed)
                    if st.swell.significant_wave_height > 0:
                        wave_elevation.add_spectrum(WaveSpectrum(
                            hs=st.swell.significant_wave_height,
                            tp=st.swell.peak_period,
                            direction_deg=st.swell.direction_deg,
                            spreading_factor=st.swell.spreading_factor,
                        ))
                    if st.wave.significant_wave_height > 0:
                        wave_elevation.add_spectrum(WaveSpectrum(
                            hs=st.wave.significant_wave_height,
                            tp=st.wave.peak_period,
                            direction_deg=st.wave.direction_deg,
                            spreading_factor=st.wave.spreading_factor,
                        ))
                    n_f = len(st.frequencies)
                    n_d = len(st.directions)
                    print(f"[wave] Rebuilt wave model: {n_f} freqs x {n_d} dirs, "
                          f"seed={st.random_seed}")


        # Update ocean surface
        ocean.update(st.sim_time)

        # Compute Python wave elevation at vessel's LF position for comparison.
        # Note: C++ evaluates at LF-only NED position (no wave-frequency offset),
        # but we receive total position (LF+WF) over UDP. For DP the vessel stays
        # near origin so we use the total position — the spatial difference is
        # small relative to wavelength.
        # Time offset: OceanSimulationTime is ahead of the internal wave time
        # used by the C++ simulator. Empirically, t+0.05 gives the best match
        # (the C++ wave evaluation time falls between two OceanSimulationTime steps).
        wave_compare_time = st.sim_time - 0.05
        py_wave_elev = float(wave_elevation.elevation(
            wave_compare_time, st.vessel_north, st.vessel_east))

        # Update vessel transform
        vessel.update_transform(
            north=st.vessel_north,
            east=st.vessel_east,
            heading_deg=st.vessel_heading,
            roll_deg=st.vessel_roll,
            pitch_deg=st.vessel_pitch,
            heave=st.vessel_heave,
        )

        # Update gangway articulation (follows vessel + gangway joints)
        # If gangway config was received from simulator, apply it once
        if not mock_gen and st.gangway_config_received:
            if not hasattr(on_update, '_gangway_config_applied'):
                vessel.set_gangway_config(st.gangway_config)
                # Re-add gangway actors to scene with new geometry
                # (only needed if config changed — happens once at start)
                on_update._gangway_config_applied = True

        vessel.gangway.update_transforms(
            vessel_north=st.vessel_north,
            vessel_east=st.vessel_east,
            vessel_heading_deg=st.vessel_heading,
            vessel_roll_deg=st.vessel_roll,
            vessel_pitch_deg=st.vessel_pitch,
            vessel_heave=st.vessel_heave,
            gangway_state=st.gangway_state,
        )

        # Update turbine transform
        turbine.update_transform(
            north=st.platform_north,
            east=st.platform_east,
            heading_deg=st.platform_heading,
            roll_deg=st.platform_roll,
            pitch_deg=st.platform_pitch,
            heave=st.platform_heave,
        )

        # Animate rotor — compute dt from wall clock and advance angle
        now_frame = time.time()
        dt_frame = min(now_frame - last_frame_time[0], 0.2)  # cap at 200ms
        last_frame_time[0] = now_frame

        # Wind speed at hub: in live mode from platform data, in mock from state
        wind_speed_hub = st.platform_wind_speed if not mock_gen else st.wind_speed
        turbine.update_rotor_angle(dt_frame, wind_speed_hub, st.turbine_state)

        # Camera follow
        scene.follow_vessel_camera(
            st.vessel_north, st.vessel_east, st.vessel_heading,
        )

        # FPS counter
        frame_count[0] += 1
        now = time.time()
        dt = now - last_fps_time[0]
        if dt >= 1.0:
            fps_display[0] = frame_count[0] / dt
            frame_count[0] = 0
            last_fps_time[0] = now

        # Info text
        gw = st.gangway_state
        gw_states = ["Parked", "Parking", "Moving", "Connecting", "Connected"]
        gw_state_str = gw_states[gw.state] if 0 <= gw.state < len(gw_states) else "?"
        turbine_states = ["Operating", "Shutdown", "Idling"]
        turb_state_str = turbine_states[st.turbine_state] if 0 <= st.turbine_state < 3 else "?"
        rpm = turbine._omega * 60.0 / (2.0 * np.pi)  # actual smoothed RPM
        mode = "MOCK" if mock_gen else "LIVE"
        wave_diff = py_wave_elev - st.sim_wave_elevation
        scene.update_info_text(
            f"[{mode}]  t = {st.sim_time:.1f} s  |  "
            f"FPS: {fps_display[0]:.0f}\n"
            f"Vessel:   N={st.vessel_north:+.1f} E={st.vessel_east:+.1f}  "
            f"HDG={st.vessel_heading:.1f} deg  "
            f"Roll={st.vessel_roll:+.1f} Pitch={st.vessel_pitch:+.1f} "
            f"Heave={st.vessel_heave:+.2f}\n"
            f"Platform: N={st.platform_north:+.1f} E={st.platform_east:+.1f}  "
            f"HDG={st.platform_heading:.1f} deg  "
            f"Roll={st.platform_roll:+.1f} Pitch={st.platform_pitch:+.1f} "
            f"Heave={st.platform_heave:+.2f}\n"
            f"Turbine:  {turb_state_str}  "
            f"Wind={wind_speed_hub:.1f}m/s  "
            f"Rotor={rpm:.1f}rpm\n"
            f"Gangway:  {gw_state_str}  "
            f"Slew={gw.slewing_angle:.1f} Boom={gw.boom_angle:+.1f} "
            f"H={gw.height:.1f}m L={gw.total_length:.1f}m\n"
            f"WaveElev: C++={st.sim_wave_elevation:+.3f}m  "
            f"Py={py_wave_elev:+.3f}m  "
            f"diff={wave_diff:+.4f}m"
        )

    scene.on_update = on_update

    # ── Go! ────────────────────────────────────────────────────────
    try:
        scene.start()
    except KeyboardInterrupt:
        pass
    finally:
        if udp_receiver:
            udp_receiver.close()
        print("\nVisualization closed.")


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
