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
