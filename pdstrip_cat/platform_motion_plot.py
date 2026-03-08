#!/usr/bin/env python3
"""
platform_motion_plot.py — Synthesized time-domain surge & sway motion
                          for OC3 Hywind spar during DP alongside ops

Creates a realistic visualization of platform horizontal motion combining:

SURGE:
  1. Mean offset (current + wind + wave drift)
  2. First-order wave response (bimodal spectrum × surge+pitch RAO)
  3. Slow-drift from QTF (narrow-band near surge natural frequency)
  4. Wind turbulence response (Frøya spectrum)

SWAY:
  5. Vortex-induced motion (VIM) with intermittent lock-in
  6. First-order sway from directional spreading
  7. Slow random sway drift

Signals are synthesized spectrally (random phase) for components 1-4,6-7.
VIM is modeled as a narrow-band oscillation at ω_n with amplitude modulation
to capture intermittent lock-in/unlock behavior typical of post-critical
Re flow around deep-draft spars.

The resulting plot shows the chaotic, multi-frequency character of the
platform motion that a DP vessel must track during alongside operations.

References:
  - Jonkman (2010). OC3 definition. NREL/TP-500-47535.
  - Finnigan & Roddier (2007). Spar VIM in post-critical Re.
  - DNV-RP-C205 (2021) §9.2.6: VIV drag amplification.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- Imports from project modules ----
from slow_drift_swell import (
    parse_qtf_total, jonswap,
    compute_slow_drift_force_spectrum, surge_transfer_function,
    compute_mean_drift,
    MASS, A11_LOW, B_EXT, K_SURGE, OMEGA_N, T_N, Q_FACTOR,
    RHO, G, RHOG, DURATION_3H,
)
from surge_budget import (
    parse_rao_surge_pitch, compute_wind_surge, compute_current_drag,
    froya_spectrum, M_EFF, RHO_AIR, HUB_HEIGHT,
)
from combined_motion import bimodal_spectrum

# Compatibility
trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# ============================================================
# File paths
# ============================================================
RAO_FILE = "/home/blofro/src/pdstrip_test/hywind_nemoh_fine/Motion/RAO.tec"
QTF_FILE = "/home/blofro/src/pdstrip_test/hywind_nemoh_fine/results/QTF/OUT_QTFM_N.dat"

# ============================================================
# Physical constants for VIM
# ============================================================
SPAR_UPPER_D = 6.5      # m (above taper)
SPAR_LOWER_D = 9.4      # m (below taper)
ST = 0.20               # Strouhal number (post-critical)
# For OC3 spar, sway natural frequency = surge natural frequency (axisymmetric)
OMEGA_N_SWAY = OMEGA_N  # 0.0429 rad/s, T~147 s


# ============================================================
# Spectral synthesis: random phase realization
# ============================================================

def synthesize_from_spectrum(omega, S, t, rng=None):
    """
    Synthesize a zero-mean Gaussian time series from a one-sided
    power spectrum S(ω) using random phase method.

    x(t) = Σ sqrt(2 * S(ωi) * Δω) * cos(ωi * t + φi)

    Parameters
    ----------
    omega : 1D array, frequencies [rad/s]
    S : 1D array, spectral density [unit²·s/rad]
    t : 1D array, time vector [s]
    rng : numpy random Generator (for reproducibility)

    Returns
    -------
    x : 1D array, time series
    """
    if rng is None:
        rng = np.random.default_rng(42)

    dw = np.diff(omega)
    dw = np.append(dw, dw[-1])  # extend to match length

    # Amplitudes and random phases
    A = np.sqrt(2 * S * dw)
    phi = rng.uniform(0, 2 * np.pi, size=len(omega))

    # Synthesize (vectorized outer product)
    # x(t) = sum_i A_i * cos(omega_i * t + phi_i)
    # Do it in chunks to avoid memory explosion for large arrays
    x = np.zeros_like(t)
    chunk = 500  # frequencies per chunk
    for i0 in range(0, len(omega), chunk):
        i1 = min(i0 + chunk, len(omega))
        wt = np.outer(omega[i0:i1], t)  # (n_freq, n_time)
        x += np.sum(A[i0:i1, None] * np.cos(wt + phi[i0:i1, None]), axis=0)

    return x


def synthesize_narrowband(omega_center, bandwidth, sigma, t, rng=None):
    """
    Synthesize a narrow-band process centered at omega_center.

    Models slow-drift or VIM as a narrow-band Gaussian process.

    Parameters
    ----------
    omega_center : float, center frequency [rad/s]
    bandwidth : float, half-bandwidth [rad/s]
    sigma : float, target standard deviation
    t : 1D array, time vector [s]
    rng : numpy random Generator

    Returns
    -------
    x : 1D array, time series with std ≈ sigma
    """
    if rng is None:
        rng = np.random.default_rng(123)
    if sigma <= 0:
        return np.zeros_like(t)

    # Create narrow Gaussian spectrum
    n_freq = 200
    omega = np.linspace(max(1e-4, omega_center - 3*bandwidth),
                        omega_center + 3*bandwidth, n_freq)
    dw = omega[1] - omega[0]

    # Gaussian spectral shape
    S = np.exp(-0.5 * ((omega - omega_center) / bandwidth)**2)
    # Normalize to get target variance
    var_raw = trapz(S, omega)
    if var_raw > 0:
        S *= sigma**2 / var_raw

    return synthesize_from_spectrum(omega, S, t, rng)


# ============================================================
# VIM synthesis with intermittent lock-in
# ============================================================

def synthesize_vim(t, Uc, A_lock=1.0, duty_cycle=0.6, T_burst=600, rng=None):
    """
    Synthesize VIM (vortex-induced motion) in sway with intermittent
    lock-in behavior.

    In reality, VIM lock-in is not continuous — the shedding frequency
    drifts in and out of the lock-in band due to wave-induced velocity
    fluctuations and turbulence. This creates bursts of large-amplitude
    oscillation interspersed with quieter periods.

    Model:
      x_vim(t) = A(t) * sin(ω_n * t + φ_vim)

    where A(t) is an amplitude envelope that switches between lock-in
    (A ≈ A_lock) and no-lock-in (A ≈ 0.1*A_lock) with a characteristic
    burst duration of T_burst seconds and duty cycle.

    Parameters
    ----------
    t : 1D array, time [s]
    Uc : float, current speed [m/s]
    A_lock : float, VIM amplitude during lock-in [m]
    duty_cycle : float, fraction of time in lock-in (0-1)
    T_burst : float, characteristic lock-in burst duration [s]
    rng : numpy random Generator

    Returns
    -------
    x_vim : 1D array, sway VIM time series [m]
    """
    if rng is None:
        rng = np.random.default_rng(999)

    # Check if current is in VIM regime
    Vr_upper = Uc / (OMEGA_N_SWAY / (2 * np.pi) * SPAR_UPPER_D)
    Vr_lower = Uc / (OMEGA_N_SWAY / (2 * np.pi) * SPAR_LOWER_D)

    # Lock-in range: Vr = 4-12 (cross-flow)
    in_lock_range = (4 <= Vr_upper <= 12) or (4 <= Vr_lower <= 12)

    if not in_lock_range:
        # Below or above lock-in — much weaker VIM
        A_lock *= 0.1

    # Build amplitude envelope: intermittent bursts
    dt = t[1] - t[0]
    n = len(t)

    # Create switching signal: random blocks of lock-in / no-lock-in
    # Block durations from exponential distribution around T_burst
    envelope = np.zeros(n)
    i = 0
    state = rng.random() < duty_cycle  # start in lock-in or not
    while i < n:
        # Duration of this state
        dur = rng.exponential(T_burst)
        dur_samples = max(1, int(dur / dt))
        i_end = min(i + dur_samples, n)

        if state:
            # Lock-in: amplitude = A_lock with slow random modulation
            envelope[i:i_end] = A_lock * (0.7 + 0.3 * rng.random())
        else:
            # No lock-in: amplitude = 10-20% of lock-in
            envelope[i:i_end] = A_lock * (0.05 + 0.15 * rng.random())

        i = i_end
        state = not state  # toggle

    # Smooth the envelope to avoid discontinuities (low-pass filter)
    # Use a simple moving average with window ~ 2 * T_n
    win = int(2 * T_N / dt)
    if win > 1 and win < n:
        kernel = np.ones(win) / win
        envelope = np.convolve(envelope, kernel, mode='same')

    # VIM oscillation at sway natural frequency
    phi_vim = rng.uniform(0, 2 * np.pi)
    x_vim = envelope * np.sin(OMEGA_N_SWAY * t + phi_vim)

    # Add slight frequency wandering (±5% around ω_n)
    phase_wander = synthesize_narrowband(
        omega_center=0.005, bandwidth=0.003, sigma=0.15, t=t, rng=rng)
    x_vim = envelope * np.sin(OMEGA_N_SWAY * t * (1 + 0.03 * np.sin(phase_wander)) + phi_vim)

    return x_vim


# ============================================================
# Main: synthesize and plot
# ============================================================

def main():
    # ---- Sea state parameters (reference case from combined_motion.py) ----
    Hs_ws, Tp_ws, gamma_ws = 2.5, 8.0, 3.3    # wind-sea
    Hs_sw, Tp_sw, gamma_sw = 1.5, 19.0, 5.0    # swell
    U10 = 10.0       # m/s (10-min mean)
    Uc = 0.50         # m/s current
    z_above_swl = 15.0  # deck height [m]

    # ---- Time vector ----
    duration = 3600.0  # 1 hour for visualization (3h is too long to see detail)
    dt = 0.5           # 0.5 s time step (Nyquist ~ 1 Hz, plenty for swell)
    t = np.arange(0, duration, dt)
    n_t = len(t)

    print(f"Synthesizing platform motion: {duration:.0f} s, dt={dt:.1f} s, "
          f"{n_t} samples")
    print(f"  Sea state: WS {Hs_ws}m/{Tp_ws}s + Sw {Hs_sw}m/{Tp_sw}s")
    print(f"  Wind: U10={U10} m/s, Current: {Uc} m/s")
    print(f"  Evaluation height: z={z_above_swl} m above SWL")
    print()

    # Master RNG for reproducibility
    master_rng = np.random.default_rng(2026)

    # ============================================================
    # Load data
    # ============================================================
    print("  Loading QTF and RAO data...")
    qtf_omegas, qtf_T = parse_qtf_total(QTF_FILE, dof=1, beta=0.0)
    omega_rao, surge_amp, surge_ph, pitch_amp_deg, pitch_ph = \
        parse_rao_surge_pitch(RAO_FILE)
    pitch_amp_rad = np.radians(pitch_amp_deg)

    # ============================================================
    # SURGE COMPONENTS
    # ============================================================
    print("  Synthesizing surge components...")

    # --- 1. Mean offset ---
    wind = compute_wind_surge(U10, blade_state='feathered')
    current = compute_current_drag(Uc)
    S_total_qtf = bimodal_spectrum(qtf_omegas, Hs_ws, Tp_ws, gamma_ws,
                                   Hs_sw, Tp_sw, gamma_sw)
    F_mean_drift, x_mean_drift = compute_mean_drift(qtf_omegas, qtf_T,
                                                     S_total_qtf, K_SURGE)
    x_mean_total = wind['x_mean'] + current['x_mean'] + x_mean_drift
    print(f"    Mean offset: {x_mean_total:.2f} m "
          f"(wind {wind['x_mean']:.2f} + current {current['x_mean']:.2f} "
          f"+ drift {x_mean_drift:.3f})")

    # --- 2. First-order surge (bimodal spectrum × apparent surge RAO) ---
    # Complex RAOs for apparent surge at z
    X_surge = surge_amp * np.exp(1j * surge_ph)
    Theta = pitch_amp_rad * np.exp(1j * pitch_ph)
    X_app = X_surge + z_above_swl * Theta
    RAO_app = np.abs(X_app)
    phase_app = np.angle(X_app)

    # Bimodal wave spectrum at RAO frequencies
    S_wave = bimodal_spectrum(omega_rao, Hs_ws, Tp_ws, gamma_ws,
                              Hs_sw, Tp_sw, gamma_sw)

    # Response spectrum
    S_surge_1st = RAO_app**2 * S_wave
    sigma_1st = np.sqrt(trapz(S_surge_1st, omega_rao))

    # Synthesize first-order with CORRECT RAO phase
    # x(t) = Σ |RAO| * sqrt(2 * S_wave * dω) * cos(ωt + φ_wave + angle(RAO))
    dw = np.diff(omega_rao)
    dw = np.append(dw, dw[-1])
    A_1st = RAO_app * np.sqrt(2 * S_wave * dw)
    phi_wave = master_rng.uniform(0, 2 * np.pi, size=len(omega_rao))
    surge_1st = np.zeros(n_t)
    chunk = 200
    for i0 in range(0, len(omega_rao), chunk):
        i1 = min(i0 + chunk, len(omega_rao))
        wt = np.outer(omega_rao[i0:i1], t)
        surge_1st += np.sum(
            A_1st[i0:i1, None] * np.cos(wt + phi_wave[i0:i1, None]
                                         + phase_app[i0:i1, None]),
            axis=0)

    print(f"    1st-order surge: σ={np.std(surge_1st):.3f} m "
          f"(target {sigma_1st:.3f})")

    # --- 3. Slow-drift surge (from QTF) ---
    mu, S_F = compute_slow_drift_force_spectrum(qtf_omegas, qtf_T, S_total_qtf)
    H = surge_transfer_function(mu, MASS, A11_LOW, B_EXT, K_SURGE)
    S_sd = np.abs(H)**2 * S_F
    sigma_sd = np.sqrt(trapz(S_sd, mu))

    surge_sd = synthesize_from_spectrum(mu, S_sd, t,
                                        rng=np.random.default_rng(master_rng.integers(1e9)))
    print(f"    Slow-drift surge: σ={np.std(surge_sd):.3f} m "
          f"(target {sigma_sd:.3f})")

    # --- 4. Wind turbulence surge ---
    sigma_wind = wind['sigma']
    surge_wind = synthesize_narrowband(
        omega_center=OMEGA_N, bandwidth=0.02,
        sigma=sigma_wind, t=t,
        rng=np.random.default_rng(master_rng.integers(1e9)))
    print(f"    Wind turbulence surge: σ={np.std(surge_wind):.3f} m "
          f"(target {sigma_wind:.3f})")

    # --- Total surge ---
    surge_total = x_mean_total + surge_1st + surge_sd + surge_wind

    # ============================================================
    # SWAY COMPONENTS
    # ============================================================
    print("  Synthesizing sway components...")

    # --- 5. VIM (vortex-induced motion) ---
    # For Uc=0.5 m/s on D=6.5m: Vr = 0.5/(0.00682*6.5) = 11.3 → at upper
    # end of lock-in range (4-12). VIM amplitude A/D ~ 0.1-0.15 for
    # post-critical Re on a spar (Finnigan & Roddier 2007, much less than
    # sub-critical A/D ~ 1.0). Use A ≈ 0.8m (A/D ≈ 0.12).
    A_vim_lock = 0.8  # m (conservative post-critical VIM amplitude)
    sway_vim = synthesize_vim(t, Uc, A_lock=A_vim_lock, duty_cycle=0.55,
                              T_burst=500,
                              rng=np.random.default_rng(master_rng.integers(1e9)))
    print(f"    VIM sway: σ={np.std(sway_vim):.3f} m, "
          f"max={np.max(np.abs(sway_vim)):.2f} m")

    # --- 6. First-order sway from directional spread ---
    # For nominally head seas (β=0°), sway RAO ≈ 0. But with realistic
    # directional spreading (cos²s), some wave energy arrives off-axis.
    # Model as ~10% of surge first-order σ, with different phases.
    sigma_sway_1st = 0.10 * sigma_1st  # small for near-head seas
    sway_1st = synthesize_from_spectrum(
        omega_rao,
        (0.10 * RAO_app)**2 * S_wave,  # 10% of surge RAO as proxy
        t,
        rng=np.random.default_rng(master_rng.integers(1e9)))
    # Rescale to target
    if np.std(sway_1st) > 0:
        sway_1st *= sigma_sway_1st / np.std(sway_1st)
    print(f"    1st-order sway (directional spread): σ={np.std(sway_1st):.3f} m")

    # --- 7. Slow sway drift ---
    # Slow random sway drift at very low frequency, small amplitude
    sigma_sway_slow = 0.05  # m (very small for head seas)
    sway_slow = synthesize_narrowband(
        omega_center=0.01, bandwidth=0.008, sigma=sigma_sway_slow, t=t,
        rng=np.random.default_rng(master_rng.integers(1e9)))
    print(f"    Slow sway drift: σ={np.std(sway_slow):.3f} m")

    # --- Total sway ---
    sway_total = sway_vim + sway_1st + sway_slow

    print()
    print(f"  TOTAL SURGE: mean={np.mean(surge_total):.2f} m, "
          f"σ={np.std(surge_total):.2f} m, "
          f"range=[{np.min(surge_total):.1f}, {np.max(surge_total):.1f}] m")
    print(f"  TOTAL SWAY:  mean={np.mean(sway_total):.3f} m, "
          f"σ={np.std(sway_total):.2f} m, "
          f"range=[{np.min(sway_total):.1f}, {np.max(sway_total):.1f}] m")

    # ============================================================
    # PLOTTING
    # ============================================================
    print("\n  Generating plots...")

    t_min = t / 60  # convert to minutes

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[2, 2, 1, 1],
                  hspace=0.35, wspace=0.3)

    # ---- Panel 1: Surge time history (top left, wide) ----
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(t_min, x_mean_total, surge_total,
                     alpha=0.3, color='C0', label='Dynamic surge')
    ax1.plot(t_min, surge_total, 'C0-', lw=0.4, alpha=0.8)
    ax1.axhline(x_mean_total, color='C3', ls='--', lw=1.2,
                label=f'Mean offset = {x_mean_total:.1f} m')
    ax1.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
    ax1.set_ylabel('Surge [m]', fontsize=12)
    ax1.set_title(
        f'Platform surge at z = {z_above_swl:.0f}m above SWL  |  '
        f'WS {Hs_ws}m/{Tp_ws:.0f}s + Sw {Hs_sw}m/{Tp_sw:.0f}s, '
        f'U10={U10:.0f} m/s, Uc={Uc:.1f} m/s',
        fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(0, duration/60)
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: Sway time history (second row, wide) ----
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(t_min, 0, sway_total, alpha=0.3, color='C1')
    ax2.plot(t_min, sway_total, 'C1-', lw=0.4, alpha=0.8,
             label='Total sway')
    ax2.plot(t_min, sway_vim, 'C2-', lw=0.3, alpha=0.5,
             label=f'VIM (A/D ≈ {A_vim_lock/SPAR_UPPER_D:.2f}, '
                   f'T ≈ {T_N:.0f} s)')
    ax2.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
    ax2.set_ylabel('Sway [m]', fontsize=12)
    ax2.set_title(
        f'Platform sway — VIM with intermittent lock-in  |  '
        f'Vr = {Uc/(OMEGA_N/(2*np.pi)*SPAR_UPPER_D):.1f} (upper), '
        f'{Uc/(OMEGA_N/(2*np.pi)*SPAR_LOWER_D):.1f} (lower)',
        fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(0, duration/60)
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: Surge component breakdown (bottom left) ----
    ax3 = fig.add_subplot(gs[2, 0])
    # Show a 5-minute window to see wave-frequency detail
    t_start_detail = 10  # minutes
    t_end_detail = 15    # minutes
    mask = (t_min >= t_start_detail) & (t_min <= t_end_detail)

    ax3.plot(t_min[mask], surge_1st[mask], 'C0-', lw=0.8,
             label=f'1st-order (σ={np.std(surge_1st):.2f} m)', alpha=0.8)
    ax3.plot(t_min[mask], surge_sd[mask], 'C3-', lw=1.2,
             label=f'Slow-drift (σ={np.std(surge_sd):.2f} m)')
    ax3.plot(t_min[mask], surge_wind[mask], 'C2-', lw=1.0,
             label=f'Wind turb (σ={np.std(surge_wind):.2f} m)')
    ax3.set_xlabel('Time [min]', fontsize=11)
    ax3.set_ylabel('Surge component [m]', fontsize=11)
    ax3.set_title('Surge components (5-min detail)', fontsize=11)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ---- Panel 4: XY trajectory (bottom right) ----
    ax4 = fig.add_subplot(gs[2, 1])
    # Color by time
    scatter = ax4.scatter(sway_total[::10], surge_total[::10],
                          c=t_min[::10], s=0.5, cmap='viridis', alpha=0.6)
    ax4.plot(sway_total[0], surge_total[0], 'go', ms=8, label='Start')
    ax4.plot(sway_total[-1], surge_total[-1], 'rs', ms=8, label='End')
    ax4.set_xlabel('Sway [m]', fontsize=11)
    ax4.set_ylabel('Surge [m]', fontsize=11)
    ax4.set_title('XY platform trajectory (color = time)', fontsize=11)
    ax4.legend(loc='upper left', fontsize=8)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    cb = plt.colorbar(scatter, ax=ax4, shrink=0.8)
    cb.set_label('Time [min]', fontsize=9)

    # ---- Panel 5: Sway component breakdown (bottom row left) ----
    ax5 = fig.add_subplot(gs[3, 0])
    mask2 = (t_min >= t_start_detail) & (t_min <= t_end_detail)
    ax5.plot(t_min[mask2], sway_vim[mask2], 'C2-', lw=1.0,
             label=f'VIM (σ={np.std(sway_vim):.2f} m)', alpha=0.8)
    ax5.plot(t_min[mask2], sway_1st[mask2], 'C0-', lw=0.8,
             label=f'1st-order (σ={np.std(sway_1st):.2f} m)', alpha=0.7)
    ax5.plot(t_min[mask2], sway_slow[mask2], 'C4-', lw=1.2,
             label=f'Slow drift (σ={np.std(sway_slow):.2f} m)')
    ax5.set_xlabel('Time [min]', fontsize=11)
    ax5.set_ylabel('Sway component [m]', fontsize=11)
    ax5.set_title('Sway components (5-min detail)', fontsize=11)
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ---- Panel 6: Summary statistics (bottom row right) ----
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis('off')

    # Compute statistics
    pp_surge = np.max(surge_total) - np.min(surge_total)
    pp_sway = np.max(sway_total) - np.min(sway_total)

    stats_text = (
        f"MOTION SUMMARY — OC3 Hywind spar at z = {z_above_swl:.0f}m\n"
        f"{'─' * 50}\n"
        f"\n"
        f"SURGE:\n"
        f"  Mean offset:          {x_mean_total:+.2f} m\n"
        f"    (current {current['x_mean']:.2f} + wind {wind['x_mean']:.2f} "
        f"+ drift {x_mean_drift:.3f})\n"
        f"  σ (1st-order):        {np.std(surge_1st):.2f} m "
        f"  (swell-dominated)\n"
        f"  σ (slow-drift):       {np.std(surge_sd):.2f} m\n"
        f"  σ (wind turbulence):  {np.std(surge_wind):.2f} m\n"
        f"  σ (total dynamic):    {np.std(surge_total - x_mean_total):.2f} m\n"
        f"  Peak-to-peak:         {pp_surge:.1f} m\n"
        f"  Range:  [{np.min(surge_total):.1f}, {np.max(surge_total):.1f}] m\n"
        f"\n"
        f"SWAY:\n"
        f"  σ (VIM):              {np.std(sway_vim):.2f} m "
        f"  (T ≈ {T_N:.0f} s, intermittent)\n"
        f"  σ (1st-order):        {np.std(sway_1st):.2f} m "
        f"  (directional spread)\n"
        f"  σ (total):            {np.std(sway_total):.2f} m\n"
        f"  Peak-to-peak:         {pp_sway:.1f} m\n"
        f"\n"
        f"COMBINED HORIZONTAL:\n"
        f"  Max horizontal excursion: "
        f"{np.max(np.sqrt(sway_total**2 + (surge_total - x_mean_total)**2)):.1f} m "
        f"from mean pos.\n"
        f"\n"
        f"Note: VIM has Vr = {Uc/(OMEGA_N/(2*np.pi)*SPAR_UPPER_D):.1f} "
        f"(near upper lock-in boundary)\n"
        f"A/D ≈ {A_vim_lock/SPAR_UPPER_D:.2f} "
        f"(post-critical Re, self-limiting)"
    )

    ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes,
             fontsize=8.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(
        'OC3 Hywind Spar — Platform Horizontal Motion During DP Alongside Operations\n'
        '(Synthesized time-domain realization, turbine stopped)',
        fontsize=14, fontweight='bold', y=0.995)

    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'platform_motion.png')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to: {outfile}")
    plt.close()

    # ============================================================
    # Also make a clean 2-panel version (surge + sway only)
    # ============================================================
    fig2, (ax_s, ax_y) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    ax_s.fill_between(t_min, x_mean_total, surge_total,
                      alpha=0.25, color='C0')
    ax_s.plot(t_min, surge_total, 'C0-', lw=0.3, alpha=0.9)
    ax_s.axhline(x_mean_total, color='C3', ls='--', lw=1.0,
                 label=f'Mean = {x_mean_total:.1f} m')
    ax_s.axhline(0, color='k', ls='-', lw=0.4, alpha=0.3)
    ax_s.set_ylabel('Surge [m]', fontsize=13)
    ax_s.set_title(
        f'Surge at z = {z_above_swl:.0f}m  —  '
        f'WS {Hs_ws}m/{Tp_ws:.0f}s + Sw {Hs_sw}m/{Tp_sw:.0f}s, '
        f'U10={U10:.0f} m/s, Uc={Uc:.1f} m/s  |  '
        f'pp = {pp_surge:.1f} m',
        fontsize=12, fontweight='bold')
    ax_s.legend(fontsize=10, loc='upper right')
    ax_s.grid(True, alpha=0.3)

    ax_y.fill_between(t_min, 0, sway_total, alpha=0.25, color='C1')
    ax_y.plot(t_min, sway_total, 'C1-', lw=0.3, alpha=0.9)
    ax_y.axhline(0, color='k', ls='-', lw=0.4, alpha=0.3)
    ax_y.set_xlabel('Time [min]', fontsize=13)
    ax_y.set_ylabel('Sway [m]', fontsize=13)
    ax_y.set_title(
        f'Sway — VIM with intermittent lock-in  |  '
        f'A/D ≈ {A_vim_lock/SPAR_UPPER_D:.2f}, '
        f'T_VIM ≈ {T_N:.0f} s  |  '
        f'pp = {pp_sway:.1f} m',
        fontsize=12, fontweight='bold')
    ax_y.grid(True, alpha=0.3)
    ax_y.set_xlim(0, duration/60)

    fig2.suptitle(
        'OC3 Hywind — Synthesized Platform Motion (DP Alongside, Turbine Stopped)',
        fontsize=14, fontweight='bold', y=1.01)
    fig2.tight_layout()

    outfile2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'platform_motion_clean.png')
    fig2.savefig(outfile2, dpi=150, bbox_inches='tight')
    print(f"  Clean plot saved to: {outfile2}")
    plt.close()


if __name__ == '__main__':
    main()
