"""
Plotting and analysis tools for ploughing simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from .simulation import SimulationResult


def plot_overview(res: SimulationResult, title: str = "Ploughing Simulation",
                  save_path: str = None):
    """
    Generate an overview plot with key simulation results.

    6-panel figure showing:
    1. Track plot (vessel and plough positions)
    2. Wire tension at vessel
    3. Vessel speed and plough speed
    4. DP thrust commands
    5. Cross-track error
    6. Thrust utilization
    """
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    t = res.time

    # --- Panel 1: Track plot ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(res.y, res.x, 'b-', linewidth=1, label='Vessel')
    ax1.plot(res.plough_y, res.plough_x, 'r-', linewidth=1, label='Plough')
    ax1.plot(res.y[0], res.x[0], 'bo', markersize=8, label='Vessel start')
    ax1.plot(res.plough_y[0], res.plough_x[0], 'ro', markersize=8, label='Plough start')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.set_title('Track Plot')
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Wire tension ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, res.wire_tension_vessel / 1e3, 'b-', linewidth=1, label='Tension at vessel')
    ax2.plot(t, res.horizontal_force / 1e3, 'g--', linewidth=1, label='Horizontal component')
    ax2.plot(t, res.vertical_force / 1e3, 'r--', linewidth=1, label='Vertical component')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Force [kN]')
    ax2.set_title('Wire Tension')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Speeds ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, res.u, 'b-', linewidth=1, label='Vessel surge speed')
    ax3.plot(t, res.plough_speed, 'r-', linewidth=1, label='Plough speed')
    ax3.axhline(y=0, color='k', linewidth=0.5)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Speed [m/s]')
    ax3.set_title('Speeds')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: DP thrust ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, res.tau_surge / 1e3, 'b-', linewidth=1, label='Surge thrust')
    ax4.plot(t, res.tau_sway / 1e3, 'r-', linewidth=1, label='Sway thrust')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Thrust [kN]')
    ax4.set_title('DP Thrust Commands')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # --- Panel 5: Errors ---
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(t, res.cross_track_error, 'r-', linewidth=1, label='Cross-track error')
    ax5_r = ax5.twinx()
    ax5_r.plot(t, np.degrees(res.psi), 'b-', linewidth=1, label='Heading')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Cross-track [m]', color='r')
    ax5_r.set_ylabel('Heading [deg]', color='b')
    ax5.set_title('Track Errors & Heading')
    ax5.grid(True, alpha=0.3)

    # --- Panel 6: Utilization & Safety ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(t, res.thrust_utilization_surge * 100, 'b-', linewidth=1, label='Surge utilization')
    ax6.plot(t, res.thrust_utilization_sway * 100, 'r-', linewidth=1, label='Sway utilization')
    ax6_r = ax6.twinx()
    ax6_r.plot(t, res.wire_safety_factor, 'g-', linewidth=1, label='Wire SF')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Thrust utilization [%]')
    ax6_r.set_ylabel('Wire safety factor [-]', color='g')
    ax6.set_title('Utilization & Safety')
    ax6.legend(loc='upper left', fontsize=8)
    ax6_r.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_catenary_snapshot(catenary_model, water_depth, horizontal_force,
                           current_speed=0.0, tow_point_depth=5.0,
                           title="Catenary Profile", save_path=None):
    """Plot the catenary wire shape for a given condition."""
    result = catenary_model.solve_catenary(water_depth, horizontal_force,
                                            current_speed, tow_point_depth,
                                            n_points=200)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Wire profile (x from plough touchdown, z from seabed)
    ax.plot(result['wire_x'], result['wire_z'], 'b-', linewidth=2)
    ax.axhline(y=water_depth - tow_point_depth, color='c', linewidth=0.5,
               linestyle='--', label=f'Tow point depth')
    ax.axhline(y=0, color='brown', linewidth=2, label='Seabed')

    # Annotations
    ax.annotate(f'Layback: {result["layback"]:.0f} m',
                xy=(result["layback"] / 2, water_depth / 2),
                fontsize=10, ha='center')
    ax.annotate(f'T_vessel: {result["tension_vessel"]/1e3:.0f} kN\n'
                f'Angle: {np.degrees(result["angle_vessel"]):.1f} deg',
                xy=(result['wire_x'][-1], result['wire_z'][-1]),
                xytext=(result['wire_x'][-1] - 50, result['wire_z'][-1] - 20),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlabel('Horizontal distance from touchdown [m]')
    ax.set_ylabel('Height above seabed [m]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return result


def plot_force_breakdown(res: SimulationResult, title: str = "Force Breakdown",
                         save_path: str = None):
    """Detailed force breakdown plot."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    t = res.time

    # Panel 1: Plough forces
    ax = axes[0]
    ax.plot(t, res.plough_soil_force / 1e3, 'r-', label='Soil cutting')
    ax.plot(t, res.plough_friction_force / 1e3, 'g-', label='Skid friction')
    ax.plot(t, res.plough_total_resistance / 1e3, 'k-', linewidth=2, label='Total resistance')
    ax.set_ylabel('Force [kN]')
    ax.set_title('Plough Resistance')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Wire forces on vessel
    ax = axes[1]
    ax.plot(t, res.tow_force_surge / 1e3, 'b-', label='Tow surge (body)')
    ax.plot(t, res.tow_force_sway / 1e3, 'r-', label='Tow sway (body)')
    ax.plot(t, res.tow_moment_yaw / 1e3, 'g-', label='Tow yaw moment / 1000')
    ax.set_ylabel('Force [kN] / Moment [kNm/1000]')
    ax.set_title('Wire Forces on Vessel (Body Frame)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: DP response
    ax = axes[2]
    ax.plot(t, res.tau_surge / 1e3, 'b-', label='DP surge')
    ax.plot(t, res.tau_sway / 1e3, 'r-', label='DP sway')
    ax.plot(t, -res.tow_force_surge / 1e3, 'b--', alpha=0.5, label='-Tow surge (ideal FF)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [kN]')
    ax.set_title('DP Thrust vs Tow Force')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_layback_analysis(res: SimulationResult, title: str = "Layback Analysis",
                          save_path: str = None):
    """Plot layback and wire geometry over time."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    t = res.time

    ax = axes[0]
    ax.plot(t, res.layback, 'b-', linewidth=1.5)
    ax.set_ylabel('Layback [m]')
    ax.set_title('Horizontal Layback')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, res.wire_angle_vessel, 'b-', linewidth=1.5)
    ax.set_ylabel('Wire angle [deg]')
    ax.set_title('Wire Angle at Vessel (from horizontal)')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, res.wire_length_suspended, 'b-', linewidth=1.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Length [m]')
    ax.set_title('Suspended Wire Length')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_stochastic_soil(res: SimulationResult,
                         title: str = "Tow tension-based vessel speed variations\n"
                                      "during ploughing operation in challenging seabed / 1 hour period",
                         save_path: str = None,
                         trim_start: float = 120.0):
    """
    Dual-axis plot matching real operational data format:
    tow tension (orange, left axis) and vessel speed (blue, right axis)
    with clock-time x-axis (HH:MM:SS).

    Designed to visually compare with real ploughing operational logs.

    Parameters:
        res: SimulationResult
        title: Plot title
        save_path: Path to save figure (optional)
        trim_start: Seconds to trim from start to remove initial transient
    """
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta

    fig, ax1 = plt.subplots(1, 1, figsize=(16, 7))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Trim initial transient
    dt = res.time[1] - res.time[0]
    i_start = int(trim_start / dt)
    t = res.time[i_start:]

    # Tow tension in tonnes.
    # A real load pin at the vessel fairlead measures the wire tension,
    # which is the catenary horizontal force T_h — the force transmitted
    # through the nonlinear catenary spring from plough to vessel.
    # This is NOT the same as the raw plough soil resistance: the catenary
    # spring filters the plough force through the layback-tension mapping,
    # and the plough inertia provides additional temporal smoothing.
    # Using horizontal_force gives the correct operational tension character.
    tension_t = res.horizontal_force[i_start:] / (1e3 * 9.81)
    # Vessel speed through water (surge)
    speed = res.u[i_start:]
    # Plough speed along track
    plough_speed = res.plough_speed[i_start:]

    # Create datetime x-axis starting at 10:30:00 (matching real data style)
    t0 = datetime(2026, 3, 24, 10, 30, 0)
    times = [t0 + timedelta(seconds=float(ti - t[0])) for ti in t]

    # --- Tow tension (orange, left axis) ---
    ax1.plot(times, tension_t, color='tab:orange', linewidth=0.6, alpha=0.9)
    ax1.set_ylabel('Tow Tension (t)', fontsize=11)
    ax1.set_ylim(0, 150)
    ax1.set_xlabel('')
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3)

    # --- Vessel speed (blue) and plough speed (red) on right axis ---
    ax2 = ax1.twinx()
    ax2.plot(times, speed, color='tab:blue', linewidth=0.7, alpha=0.9,
             label='Vessel')
    ax2.plot(times, plough_speed, color='tab:red', linewidth=0.5, alpha=0.6,
             label='Plough')
    ax2.set_ylabel('Speed (m/s)', fontsize=11)
    ax2.set_ylim(-0.05, 0.5)
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.7)

    # Format x-axis as HH:MM:SS
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    fig.autofmt_xdate(rotation=45, ha='right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def print_summary(res: SimulationResult, config=None):
    """Print summary statistics of the simulation."""
    print("=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)

    # Trim initial transient (first 10%)
    n_trim = len(res.time) // 10
    sl = slice(n_trim, None)

    print(f"\nDuration: {res.time[-1]:.0f} s ({res.time[-1]/60:.1f} min)")

    print(f"\n--- Vessel ---")
    print(f"  Speed:   mean={np.mean(res.u[sl]):.3f} m/s, "
          f"std={np.std(res.u[sl]):.3f} m/s")
    print(f"  Heading: mean={np.degrees(np.mean(res.psi[sl])):.1f} deg, "
          f"std={np.degrees(np.std(res.psi[sl])):.2f} deg")
    print(f"  Distance covered: {res.x[-1] - res.x[0]:.0f} m")

    print(f"\n--- Plough ---")
    print(f"  Speed:   mean={np.mean(res.plough_speed[sl]):.3f} m/s")
    print(f"  Total resistance: mean={np.mean(res.plough_total_resistance[sl])/1e3:.1f} kN, "
          f"max={np.max(res.plough_total_resistance[sl])/1e3:.1f} kN")

    print(f"\n--- Wire ---")
    print(f"  Tension at vessel: mean={np.mean(res.wire_tension_vessel[sl])/1e3:.1f} kN, "
          f"max={np.max(res.wire_tension_vessel[sl])/1e3:.1f} kN")
    print(f"  Layback: mean={np.mean(res.layback[sl]):.0f} m, "
          f"range=[{np.min(res.layback[sl]):.0f}, {np.max(res.layback[sl]):.0f}] m")
    print(f"  Wire angle: mean={np.mean(res.wire_angle_vessel[sl]):.1f} deg")
    print(f"  Safety factor: min={np.min(res.wire_safety_factor[sl]):.1f}")

    print(f"\n--- DP ---")
    print(f"  Cross-track error: mean={np.mean(res.cross_track_error[sl]):.2f} m, "
          f"max={np.max(np.abs(res.cross_track_error[sl])):.2f} m")
    print(f"  Surge thrust: mean={np.mean(res.tau_surge[sl])/1e3:.1f} kN, "
          f"max={np.max(np.abs(res.tau_surge[sl]))/1e3:.1f} kN")
    print(f"  Thrust utilization surge: mean={np.mean(res.thrust_utilization_surge[sl])*100:.1f}%, "
          f"max={np.max(res.thrust_utilization_surge[sl])*100:.1f}%")
    print(f"  Thrust utilization sway:  mean={np.mean(res.thrust_utilization_sway[sl])*100:.1f}%, "
          f"max={np.max(res.thrust_utilization_sway[sl])*100:.1f}%")

    print("=" * 60)


def plot_dp_mode_comparison(res_speed: SimulationResult,
                             res_position: SimulationResult,
                             title: str = "DP Surge Control Mode Comparison:\n"
                                          "Speed Control vs Position Control",
                             save_path: str = None,
                             trim_start: float = 120.0):
    """
    Side-by-side comparison of speed-control and position-control DP modes.

    4-row layout (left = speed control, right = position control):
        Row 1: Tow tension (single axis, orange)
        Row 2: Vessel + plough speed (blue + red)
        Row 3: DP surge thrust (total command)
        Row 4: Control action breakdown (feedforward + PID/PI components)

    Both simulations use the same stochastic soil realization (same seed),
    so differences are purely due to the control architecture.

    Parameters:
        res_speed: SimulationResult from speed-control mode
        res_position: SimulationResult from position-control mode
        title: Figure title
        save_path: Path to save figure (optional)
        trim_start: Seconds to trim from start to remove initial transient
    """
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta

    fig = plt.figure(figsize=(18, 18))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(4, 2, hspace=0.40, wspace=0.30,
                           height_ratios=[1.0, 0.8, 0.8, 0.8])

    # Common trim
    dt = res_speed.time[1] - res_speed.time[0]
    i_start = int(trim_start / dt)

    # Create clock-time x-axis
    t0 = datetime(2026, 3, 24, 10, 30, 0)
    t_speed = res_speed.time[i_start:]
    t_pos = res_position.time[i_start:]
    times_speed = [t0 + timedelta(seconds=float(ti - t_speed[0])) for ti in t_speed]
    times_pos = [t0 + timedelta(seconds=float(ti - t_pos[0])) for ti in t_pos]

    # Time in minutes (for rows 2-4)
    t_min_speed = (res_speed.time[i_start:] - res_speed.time[i_start]) / 60
    t_min_pos = (res_position.time[i_start:] - res_position.time[i_start]) / 60

    # Tension in tonnes (catenary horizontal force = what a tow wire load pin measures)
    T_speed = res_speed.horizontal_force[i_start:] / (1e3 * 9.81)
    T_pos = res_position.horizontal_force[i_start:] / (1e3 * 9.81)
    V_speed = res_speed.u[i_start:]
    V_pos = res_position.u[i_start:]

    # Trim for stats (skip transient)
    n_trim = len(res_speed.time) // 10
    sl = slice(n_trim, None)

    # --- Stats ---
    corr_speed = np.corrcoef(
        res_speed.horizontal_force[sl], res_speed.u[sl])[0, 1]
    corr_pos = np.corrcoef(
        res_position.horizontal_force[sl], res_position.u[sl])[0, 1]

    # =========================================================================
    # Row 1: Tow tension only (orange)
    # =========================================================================
    ax1a = fig.add_subplot(gs[0, 0])
    ax1a.plot(times_speed, T_speed, color='tab:orange', linewidth=0.6, alpha=0.9)
    ax1a.set_ylabel('Tow Tension (t)', fontsize=10)
    ax1a.set_ylim(0, 150)
    ax1a.grid(True, alpha=0.3)
    ax1a.set_title(
        f'Speed Control (tension-speed modulation)\n'
        f'T_mean={np.mean(T_speed):.1f}t, T_max={np.max(T_speed):.1f}t, '
        f'corr(T,V)={corr_speed:.3f}',
        fontsize=10)
    ax1a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1a.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))

    ax1b = fig.add_subplot(gs[0, 1])
    ax1b.plot(times_pos, T_pos, color='tab:orange', linewidth=0.6, alpha=0.9)
    ax1b.set_ylabel('Tow Tension (t)', fontsize=10)
    ax1b.set_ylim(0, 150)
    ax1b.grid(True, alpha=0.3)
    ax1b.set_title(
        f'Position Control (moving waypoint)\n'
        f'T_mean={np.mean(T_pos):.1f}t, T_max={np.max(T_pos):.1f}t, '
        f'corr(T,V)={corr_pos:.3f}',
        fontsize=10)
    ax1b.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1b.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))

    # =========================================================================
    # Row 2: Vessel + plough speed
    # =========================================================================
    ax2a = fig.add_subplot(gs[1, 0])
    ax2a.plot(t_min_speed, V_speed, 'b-', linewidth=0.6, alpha=0.8,
              label=f'Vessel (mean={np.mean(V_speed):.3f}, std={np.std(V_speed):.3f})')
    ax2a.plot(t_min_speed, res_speed.plough_speed[i_start:], 'r-',
              linewidth=0.3, alpha=0.4, label='Plough')
    ax2a.set_ylabel('Speed (m/s)', fontsize=10)
    ax2a.set_title('Vessel & Plough Speed — Speed Control', fontsize=10)
    ax2a.legend(fontsize=8, loc='upper right')
    ax2a.grid(True, alpha=0.3)
    ax2a.set_xlabel('Time (min)')

    ax2b = fig.add_subplot(gs[1, 1])
    ax2b.plot(t_min_pos, V_pos, 'b-', linewidth=0.6, alpha=0.8,
              label=f'Vessel (mean={np.mean(V_pos):.3f}, std={np.std(V_pos):.3f})')
    ax2b.plot(t_min_pos, res_position.plough_speed[i_start:], 'r-',
              linewidth=0.3, alpha=0.4, label='Plough')
    ax2b.set_ylabel('Speed (m/s)', fontsize=10)
    ax2b.set_title('Vessel & Plough Speed — Position Control', fontsize=10)
    ax2b.legend(fontsize=8, loc='upper right')
    ax2b.grid(True, alpha=0.3)
    ax2b.set_xlabel('Time (min)')

    # Match speed y-axes
    ylim_speed = (min(ax2a.get_ylim()[0], ax2b.get_ylim()[0]),
                  max(ax2a.get_ylim()[1], ax2b.get_ylim()[1]))
    ax2a.set_ylim(ylim_speed)
    ax2b.set_ylim(ylim_speed)

    # =========================================================================
    # Row 3: DP surge thrust (total command)
    # =========================================================================
    ax3a = fig.add_subplot(gs[2, 0])
    ax3a.plot(t_min_speed, res_speed.tau_surge[i_start:] / 1e3,
              'b-', linewidth=0.5, alpha=0.8)
    ax3a.set_ylabel('Surge Thrust (kN)', fontsize=10)
    ax3a.set_title('DP Surge Thrust — Speed Control', fontsize=10)
    ax3a.grid(True, alpha=0.3)
    ax3a.set_xlabel('Time (min)')

    ax3b = fig.add_subplot(gs[2, 1])
    ax3b.plot(t_min_pos, res_position.tau_surge[i_start:] / 1e3,
              'b-', linewidth=0.5, alpha=0.8)
    ax3b.set_ylabel('Surge Thrust (kN)', fontsize=10)
    ax3b.set_title('DP Surge Thrust — Position Control', fontsize=10)
    ax3b.grid(True, alpha=0.3)
    ax3b.set_xlabel('Time (min)')

    # Match thrust y-axes
    ylim_thrust = (min(ax3a.get_ylim()[0], ax3b.get_ylim()[0]),
                   max(ax3a.get_ylim()[1], ax3b.get_ylim()[1]))
    ax3a.set_ylim(ylim_thrust)
    ax3b.set_ylim(ylim_thrust)

    # =========================================================================
    # Row 4: Control action breakdown (feedforward + PID/PI)
    # =========================================================================
    has_breakdown = (res_speed.surge_feedforward is not None and
                     res_speed.surge_pid is not None and
                     np.any(res_speed.surge_feedforward != 0))

    if has_breakdown:
        ax4a = fig.add_subplot(gs[3, 0])
        ff_s = res_speed.surge_feedforward[i_start:] / 1e3
        pid_s = res_speed.surge_pid[i_start:] / 1e3
        total_s = (res_speed.surge_feedforward[i_start:] +
                   res_speed.surge_pid[i_start:]) / 1e3
        ax4a.plot(t_min_speed, ff_s, 'g-', linewidth=0.6, alpha=0.8,
                  label='Feedforward')
        ax4a.plot(t_min_speed, pid_s, 'r-', linewidth=0.6, alpha=0.8,
                  label='PI feedback')
        ax4a.plot(t_min_speed, total_s, 'b-', linewidth=0.5, alpha=0.5,
                  label='Total (pre-sat)')
        ax4a.set_ylabel('Force (kN)', fontsize=10)
        ax4a.set_title('Control Action Breakdown — Speed Control', fontsize=10)
        ax4a.legend(fontsize=8, loc='upper right')
        ax4a.grid(True, alpha=0.3)
        ax4a.set_xlabel('Time (min)')

        ax4b = fig.add_subplot(gs[3, 1])
        ff_p = res_position.surge_feedforward[i_start:] / 1e3
        pid_p = res_position.surge_pid[i_start:] / 1e3
        total_p = (res_position.surge_feedforward[i_start:] +
                   res_position.surge_pid[i_start:]) / 1e3
        ax4b.plot(t_min_pos, ff_p, 'g-', linewidth=0.6, alpha=0.8,
                  label='Feedforward')
        ax4b.plot(t_min_pos, pid_p, 'r-', linewidth=0.6, alpha=0.8,
                  label='PID feedback')
        ax4b.plot(t_min_pos, total_p, 'b-', linewidth=0.5, alpha=0.5,
                  label='Total (pre-sat)')
        ax4b.set_ylabel('Force (kN)', fontsize=10)
        ax4b.set_title('Control Action Breakdown — Position Control', fontsize=10)
        ax4b.legend(fontsize=8, loc='upper right')
        ax4b.grid(True, alpha=0.3)
        ax4b.set_xlabel('Time (min)')

        # Match breakdown y-axes
        ylim_bd = (min(ax4a.get_ylim()[0], ax4b.get_ylim()[0]),
                   max(ax4a.get_ylim()[1], ax4b.get_ylim()[1]))
        ax4a.set_ylim(ylim_bd)
        ax4b.set_ylim(ylim_bd)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()
