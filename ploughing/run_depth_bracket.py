#!/usr/bin/env python3
"""
Depth bracket study: run speed-control simulations at multiple water depths
with identical soil to compare tension and speed signatures.

The catenary stiffness varies dramatically with water depth, which affects:
  - Tension range and variability (stiffer spring → sharper spikes)
  - Speed response to soil disturbances
  - Natural period of the catenary-vessel system
  - Layback range and wire geometry

By comparing the character of these outputs across depths, we can
bracket the water depth visible in the reference operational plot.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ploughing'))

from ploughing.vessel import VesselConfig, VesselModel
from ploughing.catenary import TowWireConfig
from ploughing.plough import PloughConfig, SoilProperties, StochasticSoilConfig
from ploughing.dp_controller import DPControllerConfig
from ploughing.environment import EnvironmentConfig, WaveDriftConfig
from ploughing.simulation import SimulationConfig, run_simulation
from ploughing.scenarios import make_vessel_config, make_plough_config, compute_wire_length


def run_depth_bracket(depths=None, speed=0.12, duration=3600.0, seed=42):
    """
    Run speed-control simulations at multiple water depths.

    Uses identical stochastic soil realization (same seed) at each depth
    so differences are purely due to catenary geometry.
    """
    if depths is None:
        depths = [30, 75, 150, 300]

    results = {}

    for depth in depths:
        print(f"\n{'='*70}")
        print(f"  DEPTH BRACKET: {depth} m")
        print(f"{'='*70}")

        # Common stochastic soil — SPATIAL (per metre), same seed
        stochastic_cfg = StochasticSoilConfig(
            hurst=0.90,
            fgn_cov=0.75,
            fgn_dx=0.025,
            fgn_lp_length=5.0,
            fgn_length=600.0,
            spike_rate=0.05,
            spike_amplitude_mean=2.5,
            spike_amplitude_std=0.8,
            spike_length_mean=1.5,
            spike_length_std=1.0,
            zone_normal_to_soft_rate=0.003,
            zone_normal_to_hard_rate=0.002,
            zone_soft_to_normal_rate=0.015,
            zone_hard_to_normal_rate=0.06,
            zone_soft_factor=0.10,
            zone_hard_factor=1.60,
            zone_transition_length=2.0,
            min_resistance_fraction=0.02,
            seed=seed,
        )

        plough_cfg = make_plough_config(25.0)
        plough_cfg.stochastic_soil = stochastic_cfg

        # Wire length: catenary arc length at operating tension + buffer.
        wire_length = compute_wire_length(float(depth))
        wire_cfg = TowWireConfig(
            diameter=0.076,
            linear_mass=25.0,
            total_length=wire_length,
        )

        soil = SoilProperties(
            undrained_shear_strength=12e3,
            submerged_unit_weight=8.0e3,
        )

        env_cfg = EnvironmentConfig(
            water_depth=float(depth),
            current_speed=0.2,
            current_direction=0.0,
            wind_speed=6.0,
            wind_direction=20.0,
            waves=WaveDriftConfig(
                Hs=1.2, Tp=7.5, wave_direction=0.0,
                C_drift=5000.0, sv_cov=0.8, sv_tau=60.0,
                first_order_surge_factor=0.3,  # Surge RAO [m/m] — wave tension texture
                seed=seed + 1000,
            ),
        )

        config = SimulationConfig(
            dt=0.2, duration=duration,
            vessel=make_vessel_config(),
            wire=wire_cfg,
            plough=plough_cfg,
            soil=soil,
            dp=DPControllerConfig(
                surge_mode='speed',
                T_surge=114.0, zeta_surge=0.9,
                Kp_speed=0.099, Ki_speed=0.003,
                tow_force_feedforward=0.85,
                wind_feedforward=0.8,
                tension_speed_modulation=True,
                tension_nominal=410e3,
                tension_speed_gain=0.7e-6,
                tension_speed_filter_tau=15.0,
                tension_speed_max_reduction=0.08,
                tension_speed_max_increase=0.08,
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

        res = run_simulation(config)
        results[depth] = (res, config)

    return results


def plot_depth_bracket(results, save_path=None, trim_start=120.0):
    """
    Multi-panel comparison plot: one row per depth.
    Each row: dual-axis tension (orange) + speed (blue), matching operational data format.
    Right side: stats summary.
    """
    depths = sorted(results.keys())
    n_depths = len(depths)

    fig = plt.figure(figsize=(18, 4.0 * n_depths + 1.5))
    fig.suptitle('Depth Bracket Study: Tension & Speed Signatures vs Water Depth\n'
                 '(Same stochastic soil realization, speed-control mode)',
                 fontsize=13, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(n_depths, 1, hspace=0.45,
                           top=0.93, bottom=0.04, left=0.06, right=0.94)

    # Clock time axis
    t0 = datetime(2026, 3, 24, 10, 30, 0)

    for row, depth in enumerate(depths):
        res, config = results[depth]

        dt = res.time[1] - res.time[0]
        i_start = int(trim_start / dt)
        t = res.time[i_start:]
        times = [t0 + timedelta(seconds=float(ti - t[0])) for ti in t]

        # Tension in tonnes (catenary horizontal force at vessel)
        tension_t = res.horizontal_force[i_start:] / (1e3 * 9.81)
        speed = res.u[i_start:]
        plough_speed = res.plough_speed[i_start:]

        # Stats (skip first 10%)
        n_trim = len(res.time) // 10
        sl = slice(n_trim, None)
        T_all = res.horizontal_force[sl] / (1e3 * 9.81)
        V_all = res.u[sl]
        corr = np.corrcoef(T_all, V_all)[0, 1]

        # Layback stats
        D_all = res.layback[sl]

        # Catenary spring stiffness estimate at equilibrium
        # Use gradient of horizontal_force vs layback near the mean
        T_h_all = res.horizontal_force[sl]
        # Sort by layback and compute local slope
        sort_idx = np.argsort(D_all)
        D_sorted = D_all[sort_idx]
        T_sorted = T_h_all[sort_idx]
        # Use central 50% to estimate stiffness
        n4 = len(D_sorted) // 4
        if n4 > 10:
            dD = D_sorted[3*n4] - D_sorted[n4]
            dT = T_sorted[3*n4] - T_sorted[n4]
            k_est = dT / max(dD, 0.01)
        else:
            k_est = 0.0

        # Natural period estimate
        M_surge = 17996307.0  # kg (from vessel model)
        if k_est > 0:
            T_nat = 2 * np.pi / np.sqrt(k_est / M_surge)
        else:
            T_nat = float('inf')

        # --- Plot ---
        ax1 = fig.add_subplot(gs[row, 0])

        # Tension (orange, left axis)
        ax1.plot(times, tension_t, color='tab:orange', linewidth=0.5, alpha=0.9)
        ax1.set_ylabel('Tension (t)', fontsize=9, color='tab:orange')
        # Auto-scale y-axis with headroom for peaks
        t_max_plot = max(np.percentile(tension_t, 99.5) * 1.2, 20)
        ax1.set_ylim(0, t_max_plot)
        ax1.tick_params(axis='y', labelcolor='tab:orange', labelsize=8)
        ax1.grid(True, alpha=0.2)

        # Speed (blue=vessel, red=plough, right axis)
        ax2 = ax1.twinx()
        ax2.plot(times, speed, color='tab:blue', linewidth=0.5, alpha=0.85,
                 label='Vessel')
        ax2.plot(times, plough_speed, color='tab:red', linewidth=0.4, alpha=0.55,
                 label='Plough')
        ax2.set_ylabel('Speed (m/s)', fontsize=9, color='tab:blue')
        ax2.set_ylim(-0.05, 0.30)
        ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=8)
        if row == 0:
            ax2.legend(loc='upper right', fontsize=7, framealpha=0.7)

        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        ax1.tick_params(axis='x', labelsize=8)

        # Title with stats
        ax1.set_title(
            f'Depth {depth}m  |  Wire {config.wire.total_length:.0f}m  |  '
            f'Layback {np.mean(D_all):.0f}m [{np.min(D_all):.0f}-{np.max(D_all):.0f}]  |  '
            f'k_cat~{k_est/1000:.0f} kN/m  T_cat~{T_nat:.0f}s  |  '
            f'T: {np.mean(T_all):.1f}t [{np.min(T_all):.1f}-{np.max(T_all):.1f}]  '
            f'V: {np.mean(V_all):.3f} std={np.std(V_all):.3f}  '
            f'corr={corr:.2f}',
            fontsize=9, fontweight='bold', loc='left')

    # X-axis label on bottom panel only
    fig.text(0.5, 0.01, 'Time', ha='center', fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
    plt.show()


def print_depth_summary(results):
    """Print a comparison table of key metrics across depths."""
    depths = sorted(results.keys())

    print(f"\n{'='*110}")
    print(f"DEPTH BRACKET SUMMARY")
    print(f"{'='*110}")
    print(f"{'Depth':>6} {'Wire':>6} {'Layback':>12} {'D range':>10} "
          f"{'k_cat':>10} {'T_cat':>8} {'T_mean':>8} {'T_range':>14} "
          f"{'V_mean':>8} {'V_std':>7} {'corr':>7} {'Slowdowns':>10}")
    print(f"{'[m]':>6} {'[m]':>6} {'[m]':>12} {'[m]':>10} "
          f"{'[kN/m]':>10} {'[s]':>8} {'[t]':>8} {'[t]':>14} "
          f"{'[m/s]':>8} {'[m/s]':>7} {'':>7} {'':>10}")
    print('-' * 110)

    M_surge = 17996307.0

    for depth in depths:
        res, config = results[depth]
        n_trim = len(res.time) // 10
        sl = slice(n_trim, None)

        T_t = res.horizontal_force[sl] / (1e3 * 9.81)
        V = res.u[sl]
        D = res.layback[sl]
        T_h = res.horizontal_force[sl]
        corr = np.corrcoef(T_t, V)[0, 1]

        # Stiffness estimate
        sort_idx = np.argsort(D)
        D_sorted = D[sort_idx]
        T_sorted = T_h[sort_idx]
        n4 = len(D_sorted) // 4
        if n4 > 10:
            dD = D_sorted[3*n4] - D_sorted[n4]
            dT = T_sorted[3*n4] - T_sorted[n4]
            k_est = dT / max(dD, 0.01)
        else:
            k_est = 0.0
        T_nat = 2 * np.pi / np.sqrt(k_est / M_surge) if k_est > 0 else float('inf')

        # Slowdown events
        slowdown_steps = np.sum(res.slowdown_state[sl] > 0)
        slowdown_time = slowdown_steps * (res.time[1] - res.time[0])

        print(f"{depth:>6} {config.wire.total_length:>6.0f} "
              f"{np.mean(D):>7.0f}±{np.std(D):>3.0f}m "
              f"[{np.min(D):>3.0f}-{np.max(D):>3.0f}] "
              f"{k_est/1000:>10.0f} {T_nat:>8.0f} "
              f"{np.mean(T_t):>8.1f} [{np.min(T_t):>5.1f}-{np.max(T_t):>5.1f}] "
              f"{np.mean(V):>8.3f} {np.std(V):>7.3f} {corr:>7.2f} "
              f"{slowdown_time:>8.0f}s")

    print(f"{'='*110}")


if __name__ == '__main__':
    depths = [30, 75, 150, 300]
    print("Running depth bracket study...")
    print(f"Depths: {depths} m")
    print(f"This will run {len(depths)} x 1-hour simulations.\n")

    results = run_depth_bracket(depths=depths)
    print_depth_summary(results)
    plot_depth_bracket(results, save_path='results/depth_bracket.png')
