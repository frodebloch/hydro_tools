#!/usr/bin/env python3
"""
combined_motion.py - Combined surge motion analysis for OC3 Hywind
                     during DP alongside operations (turbine stopped)

Brings together ALL motion contributors computed in prior scripts:
  1. First-order surge at SWL (from coupled RAOs, bimodal spectrum)
  2. First-order apparent surge at height z (surge-pitch coupling)
  3. Slow-drift from QTF with bimodal spectrum (swell+windsea+cross)
  4. Mean wave drift from diagonal QTF
  5. Wind turbulence surge (Frøya spectrum, stopped turbine)
  6. Mean offsets (current drag + wind drag + mean wave drift)
  7. Shutdown transient contribution (parameterized)
  8. Extreme value combination with proper timescale separation

The bimodal spectrum S_total(ω) = JONSWAP_windsea + JONSWAP_swell is used
for BOTH first-order and second-order computations. For the QTF slow-drift,
this automatically captures swell-swell, windsea-windsea, and cross-spectral
difference-frequency contributions.

This analysis forms the frequency-domain foundation for a time-domain
DP simulator for testing DP alongside floating wind turbines.

Import strategy:
  - slow_drift_swell.py: QTF parsing, slow-drift force spectrum, catenary,
    JONSWAP spectrum, mean drift, surge transfer function
  - surge_budget.py: RAO parsing, apparent surge, wind drag, Frøya spectrum,
    current drag

References:
  - Jonkman (2010). OC3 definition. NREL/TP-500-47535.
  - DNV-RP-C205 (2021). Environmental conditions and loads.
  - Andersen & Løvseth (1992). Frøya wind spectrum.
"""

import argparse
import numpy as np
import sys
import os

# Compatibility
trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- Imports from slow_drift_swell.py (QTF/catenary/spectrum) ----
from slow_drift_swell import (
    parse_qtf_total,
    jonswap,
    compute_slow_drift_force_spectrum,
    surge_transfer_function,
    compute_statistics,
    compute_mean_drift,
    find_catenary_equilibrium,
    mooring_restoring_force,
    compute_current_drag as compute_current_drag_simple,
    MASS, A11_LOW, B_EXT, K_SURGE, OMEGA_N, T_N, Q_FACTOR,
    RHO, G, RHOG, DURATION_3H,
)

# ---- Imports from surge_budget.py (RAO/wind/current) ----
from surge_budget import (
    parse_rao_surge_pitch,
    compute_apparent_surge_at_z,
    compute_first_order_surge,
    compute_wind_surge,
    compute_wind_drag_elements,
    compute_current_drag,
    froya_spectrum,
    wind_profile,
    M_EFF,
    RHO_AIR,
    HUB_HEIGHT,
)

# ---- Imports from shutdown_transient.py ----
from shutdown_transient import (
    make_oc3_params,
    make_tampen_params,
)

# ============================================================
# Default file paths
# ============================================================

RAO_FILE_FINE = "/home/blofro/src/pdstrip_test/hywind_nemoh_fine/Motion/RAO.tec"
QTF_FILE_FINE = "/home/blofro/src/pdstrip_test/hywind_nemoh_fine/results/QTF/OUT_QTFM_N.dat"
RAO_FILE_COARSE = "/home/blofro/src/pdstrip_test/hywind_nemoh/Motion/RAO.tec"
RAO_FILE_SWELL = "/home/blofro/src/pdstrip_test/hywind_nemoh_swell/Motion/RAO.tec"


# ============================================================
# Bimodal spectrum
# ============================================================

def bimodal_spectrum(omega, hs_ws, tp_ws, gamma_ws,
                     hs_sw, tp_sw, gamma_sw):
    """
    Bimodal wave spectrum: superposition of wind-sea and swell JONSWAP.

    S_total(ω) = JONSWAP(hs_ws, tp_ws, gamma_ws) + JONSWAP(hs_sw, tp_sw, gamma_sw)

    The total Hs of the bimodal sea is:
        Hs_total = sqrt(Hs_ws^2 + Hs_sw^2)

    Returns S_total [m²·s/rad].
    """
    S_ws = jonswap(omega, hs_ws, tp_ws, gamma=gamma_ws)
    S_sw = jonswap(omega, hs_sw, tp_sw, gamma=gamma_sw)
    return S_ws + S_sw


# ============================================================
# First-order surge with bimodal spectrum
# ============================================================

def compute_first_order_bimodal(rao_file, omega_rao, hs_ws, tp_ws, gamma_ws,
                                hs_sw, tp_sw, gamma_sw,
                                z_above_swl=0.0):
    """
    First-order surge response using bimodal spectrum and coupled RAOs.

    If z_above_swl > 0, computes apparent surge including pitch coupling.

    Parameters
    ----------
    rao_file : str, path to RAO.tec
    omega_rao : None (use RAO frequencies from file)
    hs_ws, tp_ws, gamma_ws : wind-sea parameters
    hs_sw, tp_sw, gamma_sw : swell parameters
    z_above_swl : float, height above SWL for apparent surge [m]

    Returns
    -------
    dict with sigma, sig_amp, x_max, Tz, breakdown, omega/rao arrays
    """
    omega, surge_amp, surge_ph, pitch_amp_deg, pitch_ph = \
        parse_rao_surge_pitch(rao_file)

    # Convert pitch from deg/m to rad/m
    pitch_amp_rad = np.radians(pitch_amp_deg)

    # Complex RAOs
    X_surge = surge_amp * np.exp(1j * surge_ph)
    Theta = pitch_amp_rad * np.exp(1j * pitch_ph)

    # Apparent surge RAO
    X_app = X_surge + z_above_swl * Theta

    # Bimodal spectrum at RAO frequencies
    S = bimodal_spectrum(omega, hs_ws, tp_ws, gamma_ws,
                         hs_sw, tp_sw, gamma_sw)

    # Response spectrum
    S_resp = np.abs(X_app)**2 * S
    var = trapz(S_resp, omega)
    sigma = np.sqrt(max(0, var))

    # Zero-crossing period
    m2 = trapz(S_resp * omega**2, omega)
    if m2 > 0 and var > 0:
        Tz = 2 * np.pi * np.sqrt(var / m2)
        N = DURATION_3H / Tz
    else:
        Tz = tp_sw  # fallback
        N = 1
    x_max = sigma * np.sqrt(2 * np.log(max(N, 1))) if sigma > 0 else 0.0

    # Breakdown: wind-sea only, swell only
    S_ws = jonswap(omega, hs_ws, tp_ws, gamma=gamma_ws)
    S_sw = jonswap(omega, hs_sw, tp_sw, gamma=gamma_sw)
    var_ws = trapz(np.abs(X_app)**2 * S_ws, omega)
    var_sw = trapz(np.abs(X_app)**2 * S_sw, omega)
    sigma_ws = np.sqrt(max(0, var_ws))
    sigma_sw = np.sqrt(max(0, var_sw))

    # Surge-only and pitch-only contributions
    var_surge = trapz(surge_amp**2 * S, omega)
    var_pitch = trapz((z_above_swl * pitch_amp_rad)**2 * S, omega)
    sigma_surge = np.sqrt(max(0, var_surge))
    sigma_pitch = np.sqrt(max(0, var_pitch))

    return {
        'sigma': sigma,
        'sig_amp': 2.0 * sigma,
        'x_max': x_max,
        'Tz': Tz,
        'N_cycles': N,
        'sigma_ws': sigma_ws,
        'sigma_sw': sigma_sw,
        'sigma_surge_only': sigma_surge,
        'sigma_pitch_contrib': sigma_pitch,
        'omega': omega,
        'rao_app': np.abs(X_app),
        'S_resp': S_resp,
    }


# ============================================================
# Slow-drift with bimodal spectrum (QTF)
# ============================================================

def compute_slow_drift_bimodal(qtf_omegas, qtf_T, hs_ws, tp_ws, gamma_ws,
                               hs_sw, tp_sw, gamma_sw,
                               k_surge=K_SURGE):
    """
    Slow-drift surge response from QTF with bimodal spectrum.

    The TOTAL bimodal spectrum is fed into the QTF computation, which
    automatically captures:
      - swell × swell difference frequencies
      - windsea × windsea difference frequencies
      - swell × windsea cross-spectral terms (mu ~ |ωp_sw - ωp_ws|)

    Returns dict with sigma, x_max, force spectrum, response spectrum,
    mean drift, and breakdown.
    """
    # Bimodal spectrum at QTF frequencies
    S_total = bimodal_spectrum(qtf_omegas, hs_ws, tp_ws, gamma_ws,
                               hs_sw, tp_sw, gamma_sw)

    # Also compute individual spectra for breakdown
    S_ws = jonswap(qtf_omegas, hs_ws, tp_ws, gamma=gamma_ws)
    S_sw = jonswap(qtf_omegas, hs_sw, tp_sw, gamma=gamma_sw)

    # Slow-drift force spectrum from TOTAL bimodal spectrum
    mu, S_F = compute_slow_drift_force_spectrum(qtf_omegas, qtf_T, S_total)

    # Surge transfer function
    H = surge_transfer_function(mu, MASS, A11_LOW, B_EXT, k_surge)

    # Response spectrum
    S_x = np.abs(H)**2 * S_F
    stats = compute_statistics(mu, S_x)

    # Mean drift from diagonal QTF with bimodal spectrum
    F_mean_total, x_mean_total = compute_mean_drift(qtf_omegas, qtf_T, S_total, k_surge)

    # Breakdown: slow-drift from swell-only and windsea-only
    mu_sw, SF_sw = compute_slow_drift_force_spectrum(qtf_omegas, qtf_T, S_sw)
    H_sw = surge_transfer_function(mu_sw, MASS, A11_LOW, B_EXT, k_surge)
    Sx_sw = np.abs(H_sw)**2 * SF_sw
    stats_sw = compute_statistics(mu_sw, Sx_sw)

    mu_ws, SF_ws = compute_slow_drift_force_spectrum(qtf_omegas, qtf_T, S_ws)
    H_ws = surge_transfer_function(mu_ws, MASS, A11_LOW, B_EXT, k_surge)
    Sx_ws = np.abs(H_ws)**2 * SF_ws
    stats_ws = compute_statistics(mu_ws, Sx_ws)

    # Mean drift breakdown
    F_mean_ws, x_mean_ws = compute_mean_drift(qtf_omegas, qtf_T, S_ws, k_surge)
    F_mean_sw, x_mean_sw = compute_mean_drift(qtf_omegas, qtf_T, S_sw, k_surge)

    # Cross-spectral contribution (difference between total and sum of individuals)
    # Variance is NOT additive for QTF (nonlinear), so cross-term matters
    var_cross = max(0, stats['var_dynamic'] - stats_sw['var_dynamic'] - stats_ws['var_dynamic'])

    return {
        'sigma': stats['sigma'],
        'sig_amp': stats['sig_amp'],
        'x_max': stats['x_max'],
        'N_cycles': stats['N_cycles'],
        'F_mean': F_mean_total,
        'x_mean_drift': x_mean_total,
        # Breakdown
        'sigma_sw': stats_sw['sigma'],
        'sigma_ws': stats_ws['sigma'],
        'sigma_cross': np.sqrt(var_cross) if var_cross > 0 else 0.0,
        'F_mean_sw': F_mean_sw,
        'F_mean_ws': F_mean_ws,
        'x_mean_sw': x_mean_sw,
        'x_mean_ws': x_mean_ws,
        # Spectra for plotting
        'mu': mu,
        'S_F': S_F,
        'S_x': S_x,
        'k_surge': k_surge,
    }


# ============================================================
# Shutdown transient (parameterized)
# ============================================================

def shutdown_transient_params(params, ramp_time=0.0):
    """
    Parameterized shutdown transient peak and decay.

    Rather than running the full time-domain simulation, returns
    estimated peak overshoot and characteristic decay times based
    on the analytical damped oscillator model.

    For instant shutdown (ramp_time=0):
        x_peak = x0 (initial thrust offset)
        Overshoot past zero: x_overshoot ≈ x0 * exp(-pi*zeta/sqrt(1-zeta^2))

    For ramp shutdown (ramp_time = T_ramp):
        Reduction factor ≈ sin(pi * T_ramp / T_n) / (pi * T_ramp / T_n)
        for T_ramp < T_n.  For T_ramp >= T_n, quasi-static (minimal overshoot).

    Returns dict with x0, x_peak, t_50pct, t_10pct, T_osc.
    """
    mass = params['mass']
    a_inf = params.get('a_inf', params.get('a_low', 8e6))
    k = params['k_surge']
    b = params['b_ext']
    thrust = params['thrust_rated']

    M_eff = mass + a_inf
    omega_n = np.sqrt(k / M_eff)
    T_n = 2 * np.pi / omega_n
    zeta = b / (2 * M_eff * omega_n)  # damping ratio

    # Initial offset under rated thrust (linearized)
    x0 = thrust / k

    # Peak overshoot for instant shutdown (underdamped decay from x0)
    # First overshoot past zero: x_peak = x0 * exp(-pi*zeta/sqrt(1-zeta^2))
    if zeta < 1:
        overshoot_factor = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
    else:
        overshoot_factor = 0.0  # overdamped, no overshoot

    if ramp_time <= 0:
        x_peak = x0 * overshoot_factor
        reduction = 1.0
    else:
        # Ramp reduction: for a linear force ramp from F to 0 over T_ramp,
        # the transient amplitude is reduced.  For T_ramp/T_n < 0.5, use
        # the impulse response; for larger ratios, quasi-static.
        ratio = ramp_time / T_n
        if ratio < 2.0:
            # Sinc-like reduction (from pulse loading response)
            reduction = np.abs(np.sin(np.pi * ratio) / (np.pi * ratio)) if ratio > 0 else 1.0
            # Clamp: reduction can't exceed 1
            reduction = min(reduction, 1.0)
        else:
            reduction = 0.05  # quasi-static, <5% transient

        x_peak = x0 * reduction * overshoot_factor

    # Decay envelope: amplitude decays as exp(-zeta*omega_n*t)
    # Time to 50%: t_50 = ln(2) / (zeta*omega_n)
    # Time to 10%: t_10 = ln(10) / (zeta*omega_n)
    if zeta > 0:
        t_50 = np.log(2) / (zeta * omega_n)
        t_10 = np.log(10) / (zeta * omega_n)
    else:
        t_50 = np.inf
        t_10 = np.inf

    return {
        'x0': x0,
        'x_peak_overshoot': x_peak,
        'reduction_factor': reduction if ramp_time > 0 else 1.0,
        'overshoot_factor': overshoot_factor,
        'T_osc': T_n,
        'zeta': zeta,
        'omega_n': omega_n,
        't_50': t_50,
        't_10': t_10,
        'ramp_time': ramp_time,
    }


# ============================================================
# Extreme value combination
# ============================================================

def combine_extremes(results, duration=DURATION_3H):
    """
    Combine motion contributors with proper timescale separation.

    Motion contributors fall into three timescale bands:
      1. Slow-drift / wind turbulence: T ~ 100-300 s (slowly varying)
      2. First-order wave: T ~ 7-25 s (rides on top of slow-drift)
      3. Shutdown transient: T ~ 150 s (one-time event, decays)

    For extreme value estimation:
      - Mean offsets are additive: x_mean = x_wind + x_current + x_drift
      - Slow-drift and wind turbulence overlap in frequency → RSS their sigmas
      - First-order rides on top → add x_max_1st to slow envelope
      - Total extreme: x_extreme = x_mean + x_max_slow + x_max_1st

    This is the "additive" approach, conservative for 3-hour exposure.
    Also compute RSS total for comparison.

    Parameters
    ----------
    results : dict with keys:
        'x_mean_wind', 'x_mean_current', 'x_mean_drift',
        'sigma_slow' (RSS of slow-drift + wind turb),
        'sigma_1st' (first-order wave at SWL or at z),
        'Tz_1st' (zero-crossing period of first-order),
        'sigma_wind_turb', 'sigma_slow_drift',
        'x_shutdown_peak' (optional)
    duration : float, exposure duration [s]

    Returns
    -------
    dict with combined extremes
    """
    x_mean = (results.get('x_mean_wind', 0) +
              results.get('x_mean_current', 0) +
              results.get('x_mean_drift', 0))

    # Slow-drift + wind turbulence (overlapping timescales, RSS)
    sigma_slow = np.sqrt(
        results.get('sigma_slow_drift', 0)**2 +
        results.get('sigma_wind_turb', 0)**2
    )

    # Expected max of slow envelope
    # Number of slow cycles (use slow-drift natural period ~147s)
    T_slow = T_N  # surge natural period
    N_slow = duration / T_slow
    x_max_slow = sigma_slow * np.sqrt(2 * np.log(max(N_slow, 1))) if sigma_slow > 0 else 0

    # First-order wave
    sigma_1st = results.get('sigma_1st', 0)
    Tz_1st = results.get('Tz_1st', 15.0)
    N_1st = duration / Tz_1st
    x_max_1st = sigma_1st * np.sqrt(2 * np.log(max(N_1st, 1))) if sigma_1st > 0 else 0

    # Method 1: Additive extremes (conservative)
    # The worst-case first-order peak occurs ON TOP of the worst slow-drift excursion
    x_extreme_additive = x_mean + x_max_slow + x_max_1st

    # Method 2: RSS of all dynamic sigmas (less conservative)
    sigma_total_rss = np.sqrt(sigma_slow**2 + sigma_1st**2)
    # Use the shorter Tz for cycle count (conservative)
    Tz_combined = min(Tz_1st, T_slow)
    N_combined = duration / Tz_combined
    x_max_rss = sigma_total_rss * np.sqrt(2 * np.log(max(N_combined, 1)))
    x_extreme_rss = x_mean + x_max_rss

    # Method 3: DNV-style SRSS with MPM (most probable maximum)
    # x_mpm = x_mean + sqrt(x_max_slow^2 + x_max_1st^2)
    x_extreme_srss = x_mean + np.sqrt(x_max_slow**2 + x_max_1st**2)

    # Shutdown transient (if present, adds to one-time peak)
    x_shutdown = results.get('x_shutdown_peak', 0)

    # Peak-to-peak: double amplitude of the combined dynamic motion
    # For the additive case: peak-to-peak = 2 * (x_max_slow + x_max_1st)
    pp_additive = 2 * (x_max_slow + x_max_1st)
    pp_rss = 2 * x_max_rss

    return {
        'x_mean': x_mean,
        'sigma_slow': sigma_slow,
        'sigma_1st': sigma_1st,
        'sigma_total_rss': sigma_total_rss,
        'x_max_slow': x_max_slow,
        'x_max_1st': x_max_1st,
        'x_extreme_additive': x_extreme_additive,
        'x_extreme_rss': x_extreme_rss,
        'x_extreme_srss': x_extreme_srss,
        'x_shutdown_peak': x_shutdown,
        'pp_additive': pp_additive,
        'pp_rss': pp_rss,
        'N_slow': N_slow,
        'N_1st': N_1st,
    }


# ============================================================
# Single sea state analysis
# ============================================================

def analyze_sea_state(hs_ws, tp_ws, gamma_ws,
                      hs_sw, tp_sw, gamma_sw,
                      U10, Uc, z_above_swl,
                      qtf_omegas, qtf_T,
                      rao_file=None,
                      blade_state='feathered',
                      ramp_time=0.0):
    """
    Complete combined motion analysis for one sea state.

    Parameters
    ----------
    hs_ws, tp_ws, gamma_ws : wind-sea parameters
    hs_sw, tp_sw, gamma_sw : swell parameters
    U10 : float, 10-min mean wind speed at 10m [m/s]
    Uc : float, current speed [m/s]
    z_above_swl : float, evaluation height above SWL [m]
    qtf_omegas, qtf_T : QTF data (from parse_qtf_total)
    rao_file : str, path to RAO.tec (default: fine run)
    blade_state : str, 'feathered' or 'parked'
    ramp_time : float, shutdown ramp time [s] (0 = instant)

    Returns
    -------
    dict with all motion components and combined extremes
    """
    if rao_file is None:
        rao_file = RAO_FILE_FINE

    # --- 1. First-order surge (bimodal) ---
    fo = compute_first_order_bimodal(
        rao_file, None,
        hs_ws, tp_ws, gamma_ws,
        hs_sw, tp_sw, gamma_sw,
        z_above_swl=z_above_swl)

    # Also at SWL for reference
    fo_swl = compute_first_order_bimodal(
        rao_file, None,
        hs_ws, tp_ws, gamma_ws,
        hs_sw, tp_sw, gamma_sw,
        z_above_swl=0.0)

    # --- 2. Slow-drift (bimodal QTF) ---
    sd = compute_slow_drift_bimodal(
        qtf_omegas, qtf_T,
        hs_ws, tp_ws, gamma_ws,
        hs_sw, tp_sw, gamma_sw,
        k_surge=K_SURGE)

    # --- 3. Wind ---
    wind = compute_wind_surge(U10, blade_state=blade_state)

    # --- 4. Current ---
    current = compute_current_drag(Uc)

    # --- 5. Shutdown transient ---
    oc3 = make_oc3_params()
    shutdown = shutdown_transient_params(oc3, ramp_time=ramp_time)

    # --- 6. Combined ---
    combo_input = {
        'x_mean_wind': wind['x_mean'],
        'x_mean_current': current['x_mean'],
        'x_mean_drift': sd['x_mean_drift'],
        'sigma_slow_drift': sd['sigma'],
        'sigma_wind_turb': wind['sigma'],
        'sigma_1st': fo['sigma'],
        'Tz_1st': fo['Tz'],
        'x_shutdown_peak': shutdown['x_peak_overshoot'],
    }
    combined = combine_extremes(combo_input)

    # --- Also combine at SWL ---
    combo_swl = {
        'x_mean_wind': wind['x_mean'],
        'x_mean_current': current['x_mean'],
        'x_mean_drift': sd['x_mean_drift'],
        'sigma_slow_drift': sd['sigma'],
        'sigma_wind_turb': wind['sigma'],
        'sigma_1st': fo_swl['sigma'],
        'Tz_1st': fo_swl['Tz'],
        'x_shutdown_peak': shutdown['x_peak_overshoot'],
    }
    combined_swl = combine_extremes(combo_swl)

    return {
        'first_order': fo,
        'first_order_swl': fo_swl,
        'slow_drift': sd,
        'wind': wind,
        'current': current,
        'shutdown': shutdown,
        'combined': combined,
        'combined_swl': combined_swl,
        'z': z_above_swl,
        'hs_ws': hs_ws, 'tp_ws': tp_ws, 'gamma_ws': gamma_ws,
        'hs_sw': hs_sw, 'tp_sw': tp_sw, 'gamma_sw': gamma_sw,
        'U10': U10, 'Uc': Uc,
    }


# ============================================================
# Output formatting
# ============================================================

def print_header():
    """Print analysis header."""
    print("=" * 90)
    print("COMBINED SURGE MOTION ANALYSIS — OC3 Hywind spar")
    print("DP alongside operations, turbine stopped")
    print("=" * 90)
    print()
    print(f"  Platform parameters (OC3 Hywind):")
    print(f"    Mass (total):      {MASS:.3e} kg")
    print(f"    A11 (low-freq):    {A11_LOW:.3e} kg")
    print(f"    M_eff:             {M_EFF:.3e} kg")
    print(f"    B_ext:             {B_EXT:.1e} N·s/m")
    print(f"    K_surge (linear):  {K_SURGE:.0f} N/m ({K_SURGE/1e3:.2f} kN/m)")
    print(f"    omega_n:           {OMEGA_N:.4f} rad/s")
    print(f"    T_n:               {T_N:.1f} s ({T_N/60:.1f} min)")
    print(f"    Q-factor:          {Q_FACTOR:.1f}")
    print()


def print_single_result(result, label=""):
    """Print detailed results for a single sea state analysis."""
    fo = result['first_order']
    fo_swl = result['first_order_swl']
    sd = result['slow_drift']
    wind = result['wind']
    current = result['current']
    shutdown = result['shutdown']
    combined = result['combined']
    combined_swl = result['combined_swl']
    z = result['z']

    hs_total = np.sqrt(result['hs_ws']**2 + result['hs_sw']**2)

    if label:
        print(f"\n  {'─' * 80}")
        print(f"  {label}")
        print(f"  {'─' * 80}")

    print(f"\n  Sea state:")
    print(f"    Wind-sea: Hs={result['hs_ws']:.1f} m, Tp={result['tp_ws']:.0f} s, γ={result['gamma_ws']:.1f}")
    print(f"    Swell:    Hs={result['hs_sw']:.1f} m, Tp={result['tp_sw']:.0f} s, γ={result['gamma_sw']:.1f}")
    print(f"    Combined: Hs={hs_total:.2f} m")
    print(f"    Wind:     U10={result['U10']:.0f} m/s (hub: {wind['U_hub']:.1f} m/s)")
    print(f"    Current:  Uc={result['Uc']:.2f} m/s ({result['Uc']*1.9438:.1f} kt)")
    print(f"    Eval height: z={z:.0f} m above SWL")
    print()

    # Component breakdown table
    w = 45  # label width
    print(f"  {'Component':<{w}s} {'Mean':>8s} {'σ':>8s} {'x_sig':>8s} {'x_max':>8s}")
    print(f"  {'':>{w}s} {'[m]':>8s} {'[m]':>8s} {'[m]':>8s} {'[m]':>8s}")
    print(f"  {'─'*w} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    # Mean offsets
    print(f"  {'Wind drag, mean':.<{w}s} {wind['x_mean']:8.3f} {'—':>8s} {'—':>8s} {'—':>8s}")
    print(f"  {'Current drag, mean':.<{w}s} {current['x_mean']:8.3f} {'—':>8s} {'—':>8s} {'—':>8s}")
    print(f"  {'Mean wave drift (QTF diagonal)':.<{w}s} {sd['x_mean_drift']:8.3f} {'—':>8s} {'—':>8s} {'—':>8s}")
    x_mean_total = wind['x_mean'] + current['x_mean'] + sd['x_mean_drift']
    print(f"  {'  TOTAL MEAN OFFSET':.<{w}s} {x_mean_total:8.3f} {'—':>8s} {'—':>8s} {'—':>8s}")
    print()

    # Dynamic components
    print(f"  {'Wind turbulence (Frøya)':.<{w}s} {'—':>8s} {wind['sigma']:8.3f} "
          f"{wind['sig_amp']:8.3f} {wind['x_max']:8.3f}")

    print(f"  {'Slow-drift, total (bimodal QTF)':.<{w}s} {'—':>8s} {sd['sigma']:8.3f} "
          f"{sd['sig_amp']:8.3f} {sd['x_max']:8.3f}")
    print(f"  {'  └ swell only':.<{w}s} {'':>8s} {sd['sigma_sw']:8.3f} {'':>8s} {'':>8s}")
    print(f"  {'  └ windsea only':.<{w}s} {'':>8s} {sd['sigma_ws']:8.3f} {'':>8s} {'':>8s}")
    print(f"  {'  └ cross-spectral':.<{w}s} {'':>8s} {sd['sigma_cross']:8.3f} {'':>8s} {'':>8s}")

    print(f"  {'Mean wave drift breakdown':.<{w}s} {'':>8s} {'':>8s} {'':>8s} {'':>8s}")
    print(f"  {'  └ swell':.<{w}s} {sd['x_mean_sw']:8.3f} {'':>8s} {'':>8s} {'':>8s}")
    print(f"  {'  └ windsea':.<{w}s} {sd['x_mean_ws']:8.3f} {'':>8s} {'':>8s} {'':>8s}")
    print()

    print(f"  {'1st-order surge at SWL (z=0)':.<{w}s} {'—':>8s} {fo_swl['sigma']:8.3f} "
          f"{fo_swl['sig_amp']:8.3f} {fo_swl['x_max']:8.3f}")
    if z > 0:
        print(f"  {'1st-order apparent surge (z={0}m)'.format(int(z)):.<{w}s} {'—':>8s} "
              f"{fo['sigma']:8.3f} {fo['sig_amp']:8.3f} {fo['x_max']:8.3f}")
        print(f"  {'  └ windsea contribution':.<{w}s} {'':>8s} {fo['sigma_ws']:8.3f} {'':>8s} {'':>8s}")
        print(f"  {'  └ swell contribution':.<{w}s} {'':>8s} {fo['sigma_sw']:8.3f} {'':>8s} {'':>8s}")
        print(f"  {'  └ surge-only at z':.<{w}s} {'':>8s} {fo['sigma_surge_only']:8.3f} {'':>8s} {'':>8s}")
        print(f"  {'  └ pitch contribution (z*θ)':.<{w}s} {'':>8s} {fo['sigma_pitch_contrib']:8.3f} {'':>8s} {'':>8s}")
        print(f"  {'  └ Tz (zero-crossing period)':.<{w}s} {'':>8s} {fo['Tz']:8.1f} {'s':>8s} {'':>8s}")
    print()

    # Shutdown transient
    sh = shutdown
    print(f"  {'Shutdown transient (OC3):':.<{w}s}")
    print(f"  {'  Initial thrust offset':.<{w}s} {sh['x0']:8.1f} {'':>8s} {'':>8s} {'':>8s}")
    if sh['ramp_time'] > 0:
        print(f"  {'  Ramp time':.<{w}s} {sh['ramp_time']:8.0f} {'s':>8s} {'':>8s} {'':>8s}")
        print(f"  {'  Reduction factor':.<{w}s} {sh['reduction_factor']:8.2f} {'':>8s} {'':>8s} {'':>8s}")
    print(f"  {'  Peak overshoot past zero':.<{w}s} {sh['x_peak_overshoot']:8.2f} {'':>8s} {'':>8s} {'':>8s}")
    print(f"  {'  Time to 50%':.<{w}s} {sh['t_50']/60:8.1f} {'min':>8s} {'':>8s} {'':>8s}")
    print(f"  {'  Time to 10%':.<{w}s} {sh['t_10']/60:8.1f} {'min':>8s} {'':>8s} {'':>8s}")

    # Combined extremes
    print()
    print(f"  {'═' * 78}")
    z_label = f"z={int(z)}m" if z > 0 else "SWL"
    c = combined if z > 0 else combined_swl
    print(f"  COMBINED EXTREMES ({z_label}, 3-hour exposure):")
    print(f"  {'═' * 78}")
    print()
    print(f"    Mean offset:                     {c['x_mean']:8.3f} m")
    print(f"    σ (slow-drift + wind turb, RSS): {c['sigma_slow']:8.3f} m")
    print(f"    σ (1st-order wave):               {c['sigma_1st']:8.3f} m")
    print(f"    σ (all dynamic, RSS):             {c['sigma_total_rss']:8.3f} m")
    print()
    print(f"    x_max slow (N={c['N_slow']:.0f} cycles):    {c['x_max_slow']:8.3f} m")
    print(f"    x_max 1st  (N={c['N_1st']:.0f} cycles):   {c['x_max_1st']:8.3f} m")
    print()
    print(f"    --- Extreme excursion from mean position ---")
    print(f"    Additive (conservative):          {c['x_extreme_additive']:8.2f} m")
    print(f"    SRSS (DNV-style):                 {c['x_extreme_srss']:8.2f} m")
    print(f"    RSS (unconservative):             {c['x_extreme_rss']:8.2f} m")
    print()
    print(f"    --- Peak-to-peak (double amplitude) ---")
    print(f"    Additive:                         {c['pp_additive']:8.2f} m")
    print(f"    RSS:                              {c['pp_rss']:8.2f} m")

    if z > 0:
        # Also show SWL for comparison
        cs = combined_swl
        print()
        print(f"    --- For comparison at SWL (z=0) ---")
        print(f"    Additive extreme:                 {cs['x_extreme_additive']:8.2f} m")
        print(f"    SRSS extreme:                     {cs['x_extreme_srss']:8.2f} m")
        print(f"    Peak-to-peak (additive):          {cs['pp_additive']:8.2f} m")

    if c['x_shutdown_peak'] > 0.1:
        print()
        print(f"    --- With shutdown transient ---")
        print(f"    Shutdown adds (one-time):         {c['x_shutdown_peak']:8.2f} m")
        print(f"    Peak during shutdown:              {c['x_extreme_additive'] + c['x_shutdown_peak']:8.2f} m")

    print()


# ============================================================
# Parametric sweep
# ============================================================

def run_parametric_sweep(qtf_omegas, qtf_T, rao_file=None):
    """
    Sweep across realistic Tampen bimodal sea states.

    Varied parameters:
      - Wind-sea: Hs=2.0-3.0m, Tp=7-9s, γ=3.3
      - Swell:    Hs=1.0-3.0m, Tp=15-21s, γ=5.0
      - Current:  0.0 and 0.5 m/s
      - Wind:     U10=8, 10, 12 m/s
      - Heights:  z=0 (SWL), z=15m (deck)
    """
    if rao_file is None:
        rao_file = RAO_FILE_FINE

    print()
    print("=" * 90)
    print("PARAMETRIC SWEEP — Realistic Tampen bimodal sea states")
    print("=" * 90)
    print()
    print("  All values use OC3 Hywind parameters (mooring, mass, damping).")
    print("  Tampen scaling noted where applicable.")
    print()

    # Define sweep axes
    windsea_cases = [
        (2.0, 7, 3.3),
        (2.5, 8, 3.3),
        (3.0, 9, 3.3),
    ]
    swell_cases = [
        (1.0, 15, 5.0),
        (1.0, 17, 5.0),
        (1.0, 19, 5.0),
        (1.5, 17, 5.0),
        (1.5, 19, 5.0),
        (2.0, 17, 5.0),
        (2.0, 19, 5.0),
        (2.0, 21, 5.0),
        (3.0, 19, 5.0),
    ]
    currents = [0.0, 0.5]
    winds = [8, 10, 12]
    heights = [0, 15]

    # Compact table: show key metrics for each combination
    for z in heights:
        z_label = "SWL (z=0)" if z == 0 else f"Deck (z={z}m)"
        print(f"\n  {'━' * 85}")
        print(f"  {z_label} — U10=10 m/s, Uc=0.5 m/s")
        print(f"  {'━' * 85}")
        print()
        print(f"  {'Wind-sea':^12s} {'Swell':^12s} | {'Hs_tot':>6s} "
              f"{'σ_1st':>7s} {'σ_sd':>6s} {'σ_wnd':>6s} "
              f"{'x_mean':>7s} {'x_add':>7s} {'x_srss':>7s} "
              f"{'pp_add':>7s}")
        print(f"  {'Hs/Tp':^12s} {'Hs/Tp':^12s} | {'[m]':>6s} "
              f"{'[m]':>7s} {'[m]':>6s} {'[m]':>6s} "
              f"{'[m]':>7s} {'[m]':>7s} {'[m]':>7s} "
              f"{'[m]':>7s}")
        print(f"  {'─'*12} {'─'*12} + {'─'*6} "
              f"{'─'*7} {'─'*6} {'─'*6} "
              f"{'─'*7} {'─'*7} {'─'*7} "
              f"{'─'*7}")

        for hs_ws, tp_ws, g_ws in windsea_cases:
            for hs_sw, tp_sw, g_sw in swell_cases:
                r = analyze_sea_state(
                    hs_ws, tp_ws, g_ws,
                    hs_sw, tp_sw, g_sw,
                    U10=10.0, Uc=0.5,
                    z_above_swl=z,
                    qtf_omegas=qtf_omegas, qtf_T=qtf_T,
                    rao_file=rao_file)

                c = r['combined'] if z > 0 else r['combined_swl']
                hs_tot = np.sqrt(hs_ws**2 + hs_sw**2)

                ws_tag = f"{hs_ws}/{tp_ws}"
                sw_tag = f"{hs_sw}/{tp_sw}"

                print(f"  {ws_tag:^12s} {sw_tag:^12s} | "
                      f"{hs_tot:6.2f} "
                      f"{c['sigma_1st']:7.3f} "
                      f"{c['sigma_slow']:6.3f} "
                      f"{r['wind']['sigma']:6.3f} "
                      f"{c['x_mean']:7.3f} "
                      f"{c['x_extreme_additive']:7.2f} "
                      f"{c['x_extreme_srss']:7.2f} "
                      f"{c['pp_additive']:7.2f}")

    # Wind speed sensitivity
    print(f"\n\n  {'━' * 85}")
    print(f"  Wind speed sensitivity — z=15m, Swell Hs=1.5m Tp=19s, WS Hs=2.5m Tp=8s, Uc=0.5 m/s")
    print(f"  {'━' * 85}")
    print()
    print(f"  {'U10':>5s} | {'σ_1st':>7s} {'σ_sd':>6s} {'σ_wnd':>6s} "
          f"{'x_mean':>7s} {'x_add':>7s} {'x_srss':>7s} {'pp_add':>7s}")
    print(f"  {'[m/s]':>5s} | {'[m]':>7s} {'[m]':>6s} {'[m]':>6s} "
          f"{'[m]':>7s} {'[m]':>7s} {'[m]':>7s} {'[m]':>7s}")
    print(f"  {'─'*5} + {'─'*7} {'─'*6} {'─'*6} "
          f"{'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    for U10 in winds:
        r = analyze_sea_state(
            2.5, 8, 3.3, 1.5, 19, 5.0,
            U10=U10, Uc=0.5,
            z_above_swl=15.0,
            qtf_omegas=qtf_omegas, qtf_T=qtf_T,
            rao_file=rao_file)
        c = r['combined']
        print(f"  {U10:5.0f} | {c['sigma_1st']:7.3f} {c['sigma_slow']:6.3f} "
              f"{r['wind']['sigma']:6.3f} {c['x_mean']:7.3f} "
              f"{c['x_extreme_additive']:7.2f} {c['x_extreme_srss']:7.2f} "
              f"{c['pp_additive']:7.2f}")

    # Current sensitivity
    print(f"\n\n  {'━' * 85}")
    print(f"  Current sensitivity — z=15m, Swell Hs=1.5m Tp=19s, WS Hs=2.5m Tp=8s, U10=10 m/s")
    print(f"  {'━' * 85}")
    print()
    print(f"  {'Uc':>5s} | {'σ_1st':>7s} {'σ_sd':>6s} {'σ_wnd':>6s} "
          f"{'x_mean':>7s} {'x_add':>7s} {'x_srss':>7s} {'pp_add':>7s}")
    print(f"  {'[m/s]':>5s} | {'[m]':>7s} {'[m]':>6s} {'[m]':>6s} "
          f"{'[m]':>7s} {'[m]':>7s} {'[m]':>7s} {'[m]':>7s}")
    print(f"  {'─'*5} + {'─'*7} {'─'*6} {'─'*6} "
          f"{'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    for Uc in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = analyze_sea_state(
            2.5, 8, 3.3, 1.5, 19, 5.0,
            U10=10.0, Uc=Uc,
            z_above_swl=15.0,
            qtf_omegas=qtf_omegas, qtf_T=qtf_T,
            rao_file=rao_file)
        c = r['combined']
        print(f"  {Uc:5.2f} | {c['sigma_1st']:7.3f} {c['sigma_slow']:6.3f} "
              f"{r['wind']['sigma']:6.3f} {c['x_mean']:7.3f} "
              f"{c['x_extreme_additive']:7.2f} {c['x_extreme_srss']:7.2f} "
              f"{c['pp_additive']:7.2f}")

    print()


# ============================================================
# Tampen scaling summary
# ============================================================

def print_tampen_scaling():
    """Print Tampen parameter scaling and shutdown transient comparison."""
    print()
    print("=" * 90)
    print("TAMPEN SCALING — Shutdown transient comparison")
    print("=" * 90)
    print()

    oc3 = make_oc3_params()
    tampen = make_tampen_params()

    print(f"  {'Parameter':<30s} {'OC3':>15s} {'Tampen':>15s} {'Ratio':>8s}")
    print(f"  {'─'*30} {'─'*15} {'─'*15} {'─'*8}")

    params_to_show = [
        ('Mass [Mt]', oc3['mass']/1e6, tampen['mass']/1e6),
        ('A_inf [Mt]', oc3['a_inf']/1e6, tampen['a_inf']/1e6),
        ('M_eff [Mt]', (oc3['mass']+oc3['a_inf'])/1e6, (tampen['mass']+tampen['a_inf'])/1e6),
        ('B_ext [kN·s/m]', oc3['b_ext']/1e3, tampen['b_ext']/1e3),
        ('K_surge [kN/m]', oc3['k_surge']/1e3, tampen['k_surge']/1e3),
        ('Rated thrust [kN]', oc3['thrust_rated']/1e3, tampen['thrust_rated']/1e3),
        ('D_L_eff [m²]', oc3['D_L_eff'], tampen['D_L_eff']),
    ]
    for label, v_oc3, v_tamp in params_to_show:
        ratio = v_tamp / v_oc3 if v_oc3 > 0 else 0
        print(f"  {label:<30s} {v_oc3:15.2f} {v_tamp:15.2f} {ratio:8.2f}")

    # Derived quantities
    M_eff_oc3 = oc3['mass'] + oc3['a_inf']
    M_eff_tamp = tampen['mass'] + tampen['a_inf']
    omega_n_oc3 = np.sqrt(oc3['k_surge'] / M_eff_oc3)
    omega_n_tamp = np.sqrt(tampen['k_surge'] / M_eff_tamp)
    T_n_oc3 = 2 * np.pi / omega_n_oc3
    T_n_tamp = 2 * np.pi / omega_n_tamp
    zeta_oc3 = oc3['b_ext'] / (2 * M_eff_oc3 * omega_n_oc3)
    zeta_tamp = tampen['b_ext'] / (2 * M_eff_tamp * omega_n_tamp)
    Q_oc3 = 1 / (2 * zeta_oc3)
    Q_tamp = 1 / (2 * zeta_tamp)

    print()
    print(f"  {'Derived':<30s} {'OC3':>15s} {'Tampen':>15s}")
    print(f"  {'─'*30} {'─'*15} {'─'*15}")
    print(f"  {'T_n [s]':<30s} {T_n_oc3:15.1f} {T_n_tamp:15.1f}")
    print(f"  {'ω_n [rad/s]':<30s} {omega_n_oc3:15.4f} {omega_n_tamp:15.4f}")
    print(f"  {'ζ (damping ratio)':<30s} {zeta_oc3:15.4f} {zeta_tamp:15.4f}")
    print(f"  {'Q-factor':<30s} {Q_oc3:15.1f} {Q_tamp:15.1f}")
    print(f"  {'Thrust offset T/K [m]':<30s} "
          f"{oc3['thrust_rated']/oc3['k_surge']:15.1f} "
          f"{tampen['thrust_rated']/tampen['k_surge']:15.1f}")

    # Shutdown transient comparison
    print()
    print(f"  Shutdown transient (instant shutdown):")
    sh_oc3 = shutdown_transient_params(oc3, ramp_time=0.0)
    sh_tamp = shutdown_transient_params(tampen, ramp_time=0.0)
    print(f"  {'':30s} {'OC3':>15s} {'Tampen':>15s}")
    print(f"  {'─'*30} {'─'*15} {'─'*15}")
    print(f"  {'x0 (thrust offset) [m]':<30s} {sh_oc3['x0']:15.1f} {sh_tamp['x0']:15.1f}")
    print(f"  {'Overshoot past zero [m]':<30s} {sh_oc3['x_peak_overshoot']:15.2f} {sh_tamp['x_peak_overshoot']:15.2f}")
    print(f"  {'Time to 50% [min]':<30s} {sh_oc3['t_50']/60:15.1f} {sh_tamp['t_50']/60:15.1f}")
    print(f"  {'Time to 10% [min]':<30s} {sh_oc3['t_10']/60:15.1f} {sh_tamp['t_10']/60:15.1f}")

    # Controlled shutdown
    print()
    print(f"  Controlled shutdown (ramp = 0.8 × T_n):")
    ramp_oc3 = 0.8 * T_n_oc3
    ramp_tamp = 0.8 * T_n_tamp
    sh_oc3_r = shutdown_transient_params(oc3, ramp_time=ramp_oc3)
    sh_tamp_r = shutdown_transient_params(tampen, ramp_time=ramp_tamp)
    print(f"  {'Ramp time [s]':<30s} {ramp_oc3:15.0f} {ramp_tamp:15.0f}")
    print(f"  {'Reduction factor':<30s} {sh_oc3_r['reduction_factor']:15.2f} {sh_tamp_r['reduction_factor']:15.2f}")
    print(f"  {'Peak overshoot [m]':<30s} {sh_oc3_r['x_peak_overshoot']:15.2f} {sh_tamp_r['x_peak_overshoot']:15.2f}")
    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Combined surge motion analysis for OC3 Hywind alongside ops')
    parser.add_argument('--rao-file', default=RAO_FILE_FINE,
                        help='RAO file (default: fine Nemoh run)')
    parser.add_argument('--qtf-file', default=QTF_FILE_FINE,
                        help='QTF file (default: fine Nemoh run)')
    parser.add_argument('--no-sweep', action='store_true',
                        help='Skip parametric sweep (faster)')
    args = parser.parse_args()

    print_header()

    # ---- Load QTF ----
    print(f"  Loading QTF from: {args.qtf_file}")
    qtf_omegas, qtf_T = parse_qtf_total(args.qtf_file, dof=1, beta=0.0)
    n_freq = len(qtf_omegas)
    dw = qtf_omegas[1] - qtf_omegas[0] if n_freq > 1 else 0.01
    print(f"    {n_freq} frequencies: [{qtf_omegas[0]:.4f}, {qtf_omegas[-1]:.4f}] rad/s, "
          f"Δω={dw:.4f}")
    print(f"    Smallest diff freq: {dw:.4f} rad/s (T={2*np.pi/dw:.0f} s)")
    print(f"    Surge resonance:    {OMEGA_N:.4f} rad/s (T={T_N:.0f} s)")
    print()

    # ---- Load RAO (check file exists) ----
    rao_file = args.rao_file
    if not os.path.exists(rao_file):
        print(f"  WARNING: RAO file not found: {rao_file}")
        print(f"  Falling back to swell run RAO")
        rao_file = RAO_FILE_SWELL
    print(f"  Using RAO file: {rao_file}")
    print()

    # ============================================================
    # PART 1: Reference case — detailed analysis
    # ============================================================
    print("=" * 90)
    print("PART 1: REFERENCE CASE — Detailed analysis")
    print("=" * 90)

    ref = analyze_sea_state(
        hs_ws=2.5, tp_ws=8, gamma_ws=3.3,
        hs_sw=1.5, tp_sw=19, gamma_sw=5.0,
        U10=10.0, Uc=0.5,
        z_above_swl=15.0,
        qtf_omegas=qtf_omegas, qtf_T=qtf_T,
        rao_file=rao_file,
        blade_state='feathered')

    print_single_result(ref, label="Reference: WS 2.5m/8s + Sw 1.5m/19s, U10=10, Uc=0.5, z=15m")

    # ============================================================
    # PART 2: Height comparison
    # ============================================================
    print()
    print("=" * 90)
    print("PART 2: HEIGHT COMPARISON — Same sea state at z=0 vs z=15m")
    print("=" * 90)
    print()

    ref_swl = analyze_sea_state(
        hs_ws=2.5, tp_ws=8, gamma_ws=3.3,
        hs_sw=1.5, tp_sw=19, gamma_sw=5.0,
        U10=10.0, Uc=0.5,
        z_above_swl=0.0,
        qtf_omegas=qtf_omegas, qtf_T=qtf_T,
        rao_file=rao_file)

    print(f"  {'Metric':<40s} {'z=0 (SWL)':>12s} {'z=15m (deck)':>12s} {'Ratio':>8s}")
    print(f"  {'─'*40} {'─'*12} {'─'*12} {'─'*8}")

    c0 = ref_swl['combined_swl']
    c15 = ref['combined']

    rows = [
        ('σ 1st-order [m]', c0['sigma_1st'], c15['sigma_1st']),
        ('σ slow-drift+wind [m]', c0['sigma_slow'], c15['sigma_slow']),
        ('Mean offset [m]', c0['x_mean'], c15['x_mean']),
        ('x_extreme additive [m]', c0['x_extreme_additive'], c15['x_extreme_additive']),
        ('x_extreme SRSS [m]', c0['x_extreme_srss'], c15['x_extreme_srss']),
        ('Peak-to-peak additive [m]', c0['pp_additive'], c15['pp_additive']),
    ]
    for label, v0, v15 in rows:
        ratio = v15 / v0 if v0 > 0 else 0
        print(f"  {label:<40s} {v0:12.3f} {v15:12.3f} {ratio:8.2f}")

    print()
    print(f"  The surge-pitch coupling amplifies the apparent motion at deck level")
    print(f"  by a factor of {c15['sigma_1st']/c0['sigma_1st']:.1f}x for first-order")
    print(f"  (platform COG at -78m, pitch resonance near swell peak).")

    # ============================================================
    # PART 3: Dominant contributor analysis
    # ============================================================
    print()
    print("=" * 90)
    print("PART 3: CONTRIBUTOR RANKING")
    print("=" * 90)
    print()

    contributors = [
        ('Current drag (mean)', ref['current']['x_mean'], 'mean'),
        ('Wind drag (mean)', ref['wind']['x_mean'], 'mean'),
        ('Mean wave drift', ref['slow_drift']['x_mean_drift'], 'mean'),
        ('1st-order wave (z=15m)', ref['first_order']['sigma'], 'sigma'),
        ('Wind turbulence', ref['wind']['sigma'], 'sigma'),
        ('Slow-drift (bimodal QTF)', ref['slow_drift']['sigma'], 'sigma'),
    ]

    # Sort dynamic by sigma, mean by value
    means = [(l, v) for l, v, t in contributors if t == 'mean']
    dynamics = [(l, v) for l, v, t in contributors if t == 'sigma']
    means.sort(key=lambda x: abs(x[1]), reverse=True)
    dynamics.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"  Mean offsets (ranked by magnitude):")
    for rank, (label, val) in enumerate(means, 1):
        print(f"    {rank}. {label:<35s} {val:8.3f} m")

    print()
    print(f"  Dynamic components (ranked by σ):")
    for rank, (label, val) in enumerate(dynamics, 1):
        ratio = val / dynamics[0][1] if dynamics[0][1] > 0 else 0
        print(f"    {rank}. {label:<35s} σ = {val:8.3f} m  ({ratio*100:5.1f}% of largest)")

    print()
    print(f"  The first-order wave response (with surge-pitch coupling at deck level)")
    print(f"  is the DOMINANT dynamic contributor, ~{dynamics[0][1]/dynamics[1][1]:.0f}x larger than the next.")
    print(f"  Slow-drift and wind turbulence are secondary contributors.")

    # ============================================================
    # PART 4: Parametric sweep
    # ============================================================
    if not args.no_sweep:
        run_parametric_sweep(qtf_omegas, qtf_T, rao_file=rao_file)

    # ============================================================
    # PART 5: Tampen scaling
    # ============================================================
    print_tampen_scaling()

    # ============================================================
    # CONCLUSIONS
    # ============================================================
    print()
    print("=" * 90)
    print("CONCLUSIONS — Combined motion analysis for DP alongside")
    print("=" * 90)
    print("""
  1. FIRST-ORDER WAVE RESPONSE DOMINATES at deck level (z=15m):
     - Surge-pitch coupling amplifies apparent motion ~2x vs SWL
     - σ = 1.5-3 m in typical Tampen swell (Hs=1-2m, Tp=15-21s)
     - Expected max in 3 hours: 5-12 m peak-to-peak at deck level
     - Period: 15-25 s (wave frequency, NOT slow-drift)

  2. SLOW-DRIFT IS SECONDARY but non-negligible:
     - σ = 0.05-0.15 m from bimodal QTF (with cross-spectral terms)
     - The swell-swell and windsea-windsea self-interaction terms dominate
     - Cross-spectral (swell×windsea) terms produce difference frequencies
       ~0.4-0.6 rad/s, far from surge resonance — negligible contribution

  3. MEAN OFFSETS are dominated by current:
     - Current 0.5 m/s → 3.6 m mean offset (largest single contributor)
     - Wind (U10=10, feathered) → ~1 m mean offset
     - Mean wave drift → ~0.1-0.3 m (small)

  4. WIND TURBULENCE is moderate:
     - σ = 0.2-0.4 m (Frøya spectrum, stopped turbine)
     - Frequency content overlaps slow-drift → RSS combination correct

  5. SHUTDOWN TRANSIENT adds one-time peak:
     - OC3: 19.4 m initial offset, overshoot ~12 m
     - Tampen: 18.6 m initial offset, overshoot ~12 m
     - Controlled ramp of ~0.8×T_n reduces peak by ~80%
     - Residual oscillation decays over 10-20 minutes

  6. WORST CASE COMBINED MOTION (additive extremes at z=15m):
     - Mean + slow max + 1st-order max
     - Reference case: ~10-15 m extreme excursion from rest
     - Peak-to-peak: ~10-20 m depending on sea state
     - Consistent with reported 10-15 m motions at Hywind Tampen

  7. IMPLICATIONS FOR DP ALONGSIDE SIMULATOR:
     - Must model first-order wave response at deck level (dominant)
     - Must include surge-pitch coupling (critical for apparent motion)
     - Slow-drift and wind turbulence set the slowly-varying baseline
     - Current sets the mean offset (DP feedforward target)
     - Shutdown transient is the most critical single event
     - DP vessel tracking of ~20 s wave-frequency platform motion
       is the key operational challenge
""")


if __name__ == '__main__':
    main()
