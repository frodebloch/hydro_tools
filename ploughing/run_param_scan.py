"""
Parameter scan to find settings that produce fgn_lp_seed58-like speed character.

Target from seed58 plot:
  - V_std ~ 0.05 m/s
  - Smooth slow undulations (2-5 min timescale)
  - Strong inverse correlation with tension
  - Speed range ~0.05-0.25 m/s

We scan key parameters for both speed-control and position-control modes.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import sys

from ploughing.vessel import VesselConfig
from ploughing.catenary import TowWireConfig
from ploughing.plough import PloughConfig, SoilProperties, StochasticSoilConfig
from ploughing.dp_controller import DPControllerConfig
from ploughing.environment import EnvironmentConfig, WaveDriftConfig
from ploughing.simulation import SimulationConfig, run_simulation
from ploughing.scenarios import make_vessel_config, make_plough_config


def make_common_config(duration=3600.0, seed=42):
    """Return shared soil, wire, env configs."""
    water_depth = 20.0
    stochastic_cfg = StochasticSoilConfig(
        hurst=0.90, fgn_cov=0.50, fgn_dx=0.025,
        fgn_length=600.0,
        spike_rate=0.04, spike_amplitude_mean=1.4, spike_amplitude_std=0.3,
        spike_length_mean=1.4, spike_length_std=1.0,
        zone_normal_to_soft_rate=0.002, zone_normal_to_hard_rate=0.0015,
        zone_soft_to_normal_rate=0.015, zone_hard_to_normal_rate=0.06,
        zone_soft_factor=0.15, zone_hard_factor=1.4,
        zone_transition_length=2.0, min_resistance_fraction=0.02,
        seed=seed,
    )
    plough_cfg = make_plough_config(25.0)
    plough_cfg.stochastic_soil = stochastic_cfg

    wire_cfg = TowWireConfig(diameter=0.076, linear_mass=25.0,
                              total_length=max(water_depth * 3.0, 200.0))
    soil = SoilProperties(undrained_shear_strength=15e3, submerged_unit_weight=8.0e3)
    env_cfg = EnvironmentConfig(
        water_depth=water_depth, current_speed=0.2, current_direction=0.0,
        wind_speed=6.0, wind_direction=20.0,
        waves=WaveDriftConfig(
            Hs=1.2, Tp=7.5, wave_direction=0.0, C_drift=5000.0,
            sv_cov=0.8, sv_tau=60.0, first_order_surge_factor=0.0,
            seed=seed + 1000,
        ),
    )
    return water_depth, plough_cfg, wire_cfg, soil, env_cfg, stochastic_cfg


def run_speed_control(label, track_speed, Kp_speed, Ki_speed, ff,
                       tension_gain, tension_tau, tension_nom,
                       max_red, max_inc,
                       duration=3600.0, seed=42):
    """Run speed-control mode with given params, return stats."""
    wd, plough_cfg, wire_cfg, soil, env_cfg, stoch = make_common_config(duration, seed)
    # Need fresh plough (stochastic state is mutable)
    pcfg = make_plough_config(25.0)
    pcfg.stochastic_soil = StochasticSoilConfig(
        hurst=stoch.hurst, fgn_cov=stoch.fgn_cov, fgn_dx=stoch.fgn_dx,
        fgn_length=stoch.fgn_length,
        spike_rate=stoch.spike_rate, spike_amplitude_mean=stoch.spike_amplitude_mean,
        spike_amplitude_std=stoch.spike_amplitude_std,
        spike_length_mean=stoch.spike_length_mean,
        spike_length_std=stoch.spike_length_std,
        zone_normal_to_soft_rate=stoch.zone_normal_to_soft_rate,
        zone_normal_to_hard_rate=stoch.zone_normal_to_hard_rate,
        zone_soft_to_normal_rate=stoch.zone_soft_to_normal_rate,
        zone_hard_to_normal_rate=stoch.zone_hard_to_normal_rate,
        zone_soft_factor=stoch.zone_soft_factor, zone_hard_factor=stoch.zone_hard_factor,
        zone_transition_length=stoch.zone_transition_length,
        min_resistance_fraction=stoch.min_resistance_fraction,
        seed=seed,
    )

    config = SimulationConfig(
        dt=0.2, duration=duration,
        vessel=make_vessel_config(), wire=wire_cfg, plough=pcfg, soil=soil,
        dp=DPControllerConfig(
            surge_mode='speed',
            T_surge=114.0, zeta_surge=0.9,
            Kp_speed=Kp_speed, Ki_speed=Ki_speed,
            tow_force_feedforward=ff, wind_feedforward=0.8,
            tension_speed_modulation=True,
            tension_nominal=tension_nom,
            tension_speed_gain=tension_gain,
            tension_speed_filter_tau=tension_tau,
            tension_speed_max_reduction=max_red,
            tension_speed_max_increase=max_inc,
        ),
        env=env_cfg,
        track_start=(0.0, 0.0),
        track_end=(duration * 0.5 * 2, 0.0),
        track_speed=track_speed,
    )
    res = run_simulation(config)
    n = len(res.time) // 10
    sl = slice(n, None)
    V_mean = np.mean(res.u[sl])
    V_std = np.std(res.u[sl])
    corr = np.corrcoef(res.plough_total_resistance[sl], res.u[sl])[0, 1]
    T_mean = np.mean(res.plough_total_resistance[sl]) / (1e3 * 9.81)
    T_max = np.max(res.plough_total_resistance[sl]) / (1e3 * 9.81)
    print(f"  {label:30s} | V_mean={V_mean:.3f} V_std={V_std:.3f} corr={corr:.3f} "
          f"| T_mean={T_mean:.1f}t T_max={T_max:.1f}t")
    return res, config


def run_position_control(label, track_speed, ff, T_surge, zeta,
                          duration=3600.0, seed=42):
    """Run position-control mode with given params, return stats."""
    wd, plough_cfg, wire_cfg, soil, env_cfg, stoch = make_common_config(duration, seed)
    pcfg = make_plough_config(25.0)
    pcfg.stochastic_soil = StochasticSoilConfig(
        hurst=stoch.hurst, fgn_cov=stoch.fgn_cov, fgn_dx=stoch.fgn_dx,
        fgn_length=stoch.fgn_length,
        spike_rate=stoch.spike_rate, spike_amplitude_mean=stoch.spike_amplitude_mean,
        spike_amplitude_std=stoch.spike_amplitude_std,
        spike_length_mean=stoch.spike_length_mean,
        spike_length_std=stoch.spike_length_std,
        zone_normal_to_soft_rate=stoch.zone_normal_to_soft_rate,
        zone_normal_to_hard_rate=stoch.zone_normal_to_hard_rate,
        zone_soft_to_normal_rate=stoch.zone_soft_to_normal_rate,
        zone_hard_to_normal_rate=stoch.zone_hard_to_normal_rate,
        zone_soft_factor=stoch.zone_soft_factor, zone_hard_factor=stoch.zone_hard_factor,
        zone_transition_length=stoch.zone_transition_length,
        min_resistance_fraction=stoch.min_resistance_fraction,
        seed=seed,
    )

    config = SimulationConfig(
        dt=0.2, duration=duration,
        vessel=make_vessel_config(), wire=wire_cfg, plough=pcfg, soil=soil,
        dp=DPControllerConfig(
            surge_mode='position',
            T_surge=T_surge, zeta_surge=zeta,
            tow_force_feedforward=ff, wind_feedforward=0.8,
        ),
        env=env_cfg,
        track_start=(0.0, 0.0),
        track_end=(duration * 0.5 * 2, 0.0),
        track_speed=track_speed,
    )
    res = run_simulation(config)
    n = len(res.time) // 10
    sl = slice(n, None)
    V_mean = np.mean(res.u[sl])
    V_std = np.std(res.u[sl])
    corr = np.corrcoef(res.plough_total_resistance[sl], res.u[sl])[0, 1]
    T_mean = np.mean(res.plough_total_resistance[sl]) / (1e3 * 9.81)
    T_max = np.max(res.plough_total_resistance[sl]) / (1e3 * 9.81)
    print(f"  {label:30s} | V_mean={V_mean:.3f} V_std={V_std:.3f} corr={corr:.3f} "
          f"| T_mean={T_mean:.1f}t T_max={T_max:.1f}t")
    return res, config


print("=" * 90)
print("SPEED CONTROL PARAMETER SCAN")
print("=" * 90)
print()

# Baseline (current comparison params)
print("--- Baseline ---")
run_speed_control("S0: current",
    track_speed=0.03, Kp_speed=0.099, Ki_speed=0.003, ff=0.85,
    tension_gain=2e-6, tension_tau=8.0, tension_nom=300e3,
    max_red=0.15, max_inc=0.12)

# Increase tension filter tau (slower setpoint changes → smoother speed)
print("\n--- Vary tension_speed_filter_tau ---")
for tau in [30, 60, 90, 120]:
    run_speed_control(f"S1: tau={tau}s",
        track_speed=0.03, Kp_speed=0.099, Ki_speed=0.003, ff=0.85,
        tension_gain=2e-6, tension_tau=tau, tension_nom=300e3,
        max_red=0.15, max_inc=0.12)

# Increase tension gain (stronger speed response to tension)
print("\n--- Vary tension_speed_gain with tau=60 ---")
for gain in [3e-6, 4e-6, 5e-6]:
    run_speed_control(f"S2: gain={gain:.0e}, tau=60",
        track_speed=0.03, Kp_speed=0.099, Ki_speed=0.003, ff=0.85,
        tension_gain=gain, tension_tau=60.0, tension_nom=300e3,
        max_red=0.15, max_inc=0.12)

# Weaker DP gains (slower tracking → more natural speed variation)
print("\n--- Weaker DP with tau=60 ---")
for kp, ki in [(0.05, 0.001), (0.03, 0.0005)]:
    run_speed_control(f"S3: Kp={kp}, Ki={ki}, tau=60",
        track_speed=0.03, Kp_speed=kp, Ki_speed=ki, ff=0.85,
        tension_gain=2e-6, tension_tau=60.0, tension_nom=300e3,
        max_red=0.15, max_inc=0.12)

# Wider limits
print("\n--- Wider limits with tau=60 ---")
run_speed_control("S4: wider limits, tau=60",
    track_speed=0.03, Kp_speed=0.099, Ki_speed=0.003, ff=0.85,
    tension_gain=2e-6, tension_tau=60.0, tension_nom=300e3,
    max_red=0.20, max_inc=0.15)

# Lower feedforward (more disturbance reaches PID)
print("\n--- Lower feedforward with tau=60 ---")
for ff_val in [0.7, 0.5]:
    run_speed_control(f"S5: ff={ff_val}, tau=60",
        track_speed=0.03, Kp_speed=0.099, Ki_speed=0.003, ff=ff_val,
        tension_gain=2e-6, tension_tau=60.0, tension_nom=300e3,
        max_red=0.15, max_inc=0.12)

print()
print("=" * 90)
print("POSITION CONTROL PARAMETER SCAN")
print("=" * 90)
print()

# Baseline
print("--- Baseline ---")
run_position_control("P0: current",
    track_speed=0.12, ff=0.7, T_surge=114.0, zeta=0.9)

# Lower feedforward (more fighting)
print("\n--- Lower feedforward ---")
for ff_val in [0.5, 0.3, 0.0]:
    run_position_control(f"P1: ff={ff_val}",
        track_speed=0.12, ff=ff_val, T_surge=114.0, zeta=0.9)

# Softer position stiffness (longer natural period → more compliant)
print("\n--- Softer position gains (longer T_surge) ---")
for T in [200, 300, 400]:
    run_position_control(f"P2: T_surge={T}s, ff=0.3",
        track_speed=0.12, ff=0.3, T_surge=T, zeta=0.9)

# Lower damping (less velocity damping → more speed excursion)
print("\n--- Lower damping with T=300, ff=0.3 ---")
for z in [0.7, 0.5, 0.3]:
    run_position_control(f"P3: zeta={z}, T=300, ff=0.3",
        track_speed=0.12, ff=0.3, T_surge=300.0, zeta=z)

# Zero feedforward, soft gains, low damping
print("\n--- Aggressive: ff=0.0, T=300, varying zeta ---")
for z in [0.9, 0.5, 0.3]:
    run_position_control(f"P4: ff=0.0, T=300, z={z}",
        track_speed=0.12, ff=0.0, T_surge=300.0, zeta=z)

print("\nDone!")
