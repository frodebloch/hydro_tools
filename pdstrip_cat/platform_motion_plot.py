#!/usr/bin/env python3
"""
platform_motion_plot.py — Synthesized time-domain surge, sway & vertical
                          motion and velocity for OC3 Hywind spar during
                          DP alongside ops

Creates a realistic visualization of platform motion and velocity combining:

SURGE:
  1. Mean offset (current + wind + wave drift)
  2. First-order wave response (bimodal spectrum × surge+pitch RAO)
  3. Slow-drift from QTF (narrow-band near surge natural frequency)
  4. Wind turbulence response (Frøya spectrum)

SWAY:
  5. Vortex-induced motion (VIM) with intermittent lock-in
  6. First-order sway from directional spreading
  7. Slow random sway drift

VERTICAL (at deck level z above SWL):
  8. First-order heave (bimodal spectrum × heave RAO)
  9. First-order pitch (bimodal spectrum × pitch RAO with viscous damping
      correction — self-consistent Morison drag linearization on the spar hull)
  10. Combined vertical = heave + z * pitch(t)  [small angle]

Signals are synthesized spectrally (random phase) for components 1-4,6-10.
VIM is modeled as a narrow-band oscillation at ω_n with amplitude modulation
to capture intermittent lock-in/unlock behavior typical of post-critical
Re flow around deep-draft spars.

IMPORTANT: All spectral synthesis uses GEOMETRICALLY spaced frequencies
to avoid artificial repetition. With linear spacing Δω, the signal repeats
after T_repeat = 2π/Δω (e.g., 628 s for Δω=0.01). Geometric spacing
breaks this common-divisor relationship, giving effectively infinite
repeat time. The RAO and QTF data (on Nemoh's linear grid) are
interpolated onto the geometric grid before synthesis.

Heave, pitch and surge share the SAME wave phase realization (phi_wave)
to preserve the correct inter-DOF phase relationships from the RAOs.

The resulting plot shows the chaotic, multi-frequency character of the
platform motion that a DP vessel must track during alongside operations.
Platform velocity (time derivative of displacement) is also computed and
displayed, as it directly determines the required DP vessel tracking speed
and thruster demand.

References:
  - Jonkman (2010). OC3 definition. NREL/TP-500-47535.
  - Finnigan & Roddier (2007). Spar VIM in post-critical Re.
  - DNV-RP-C205 (2021) §9.2.6: VIV drag amplification.
  - Skaare et al. (2007). Hywind pitch damping from model tests.
  - Robertson et al. (2014). OC4 phase II results comparison.
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
    compute_current_variability_surge,
    MASS, A11_LOW, B_EXT, K_SURGE, OMEGA_N, T_N, Q_FACTOR,
    RHO, G, RHOG, DURATION_3H,
)
from surge_budget import (
    parse_rao_surge_pitch, parse_rao_heave_pitch,
    parse_radiation_coefficients,
    compute_wind_surge, compute_current_drag,
    froya_spectrum, M_EFF, RHO_AIR, HUB_HEIGHT,
)
from slow_drift_swell import mooring_heave_stiffness
from combined_motion import bimodal_spectrum

# Compatibility
trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def make_geometric_omega(omega_min, omega_max, n):
    """
    Create a geometrically spaced frequency vector.

    With geometric spacing, consecutive frequency ratios are constant:
        ω_{i+1} / ω_i = r = (omega_max/omega_min)^(1/(n-1))

    This breaks the common-divisor relationship between frequencies that
    causes synthesized signals to repeat after T = 2π/Δω with linear
    spacing. The result is an effectively infinite repeat time.

    Parameters
    ----------
    omega_min : float, lowest frequency [rad/s]
    omega_max : float, highest frequency [rad/s]
    n : int, number of frequencies

    Returns
    -------
    omega : 1D array, geometrically spaced frequencies [rad/s]
    """
    return np.geomspace(omega_min, omega_max, n)


# ============================================================
# File paths
# ============================================================
RAO_FILE = "/home/blofro/src/pdstrip_test/hywind_nemoh_fine/Motion/RAO.tec"
QTF_FILE = "/home/blofro/src/pdstrip_test/hywind_nemoh_fine/results/QTF/OUT_QTFM_N.dat"
RAD_FILE = "/home/blofro/src/pdstrip_test/hywind_nemoh_fine/results/RadiationCoefficients.tec"

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
    Uses geometric frequency spacing to avoid artificial repetition.

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

    # Create narrow Gaussian spectrum on GEOMETRIC frequency grid
    n_freq = 200
    omega_lo = max(1e-4, omega_center - 3*bandwidth)
    omega_hi = omega_center + 3*bandwidth
    omega = make_geometric_omega(omega_lo, omega_hi, n_freq)

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
    sigma_Uc = 0.10   # m/s rms current fluctuation (conservative estimate)
    T_peak_current = 1800.0  # s (30 min peak period for current variability)
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
    omega_rao_lin, surge_amp_lin, surge_ph_lin, pitch_amp_deg_lin, pitch_ph_lin = \
        parse_rao_surge_pitch(RAO_FILE)
    pitch_amp_rad_lin = np.radians(pitch_amp_deg_lin)

    # ============================================================
    # Create GEOMETRIC frequency grid for first-order synthesis
    # ============================================================
    # With Nemoh's linear Δω=0.01 rad/s, signals repeat after 2π/Δω=628 s.
    # Geometric spacing breaks this: ω_{i+1}/ω_i = const → no common divisor.
    N_GEO = 200  # more points than original 80 for good spectral resolution
    omega_geo = make_geometric_omega(omega_rao_lin[0], omega_rao_lin[-1], N_GEO)
    print(f"  Geometric frequency grid: {N_GEO} points, "
          f"ω = [{omega_geo[0]:.4f}, {omega_geo[-1]:.3f}] rad/s, "
          f"ratio = {omega_geo[1]/omega_geo[0]:.6f}")

    # Interpolate RAO data onto geometric grid
    # (amplitude and phase interpolated separately to avoid complex wrapping issues)
    surge_amp = np.interp(omega_geo, omega_rao_lin, surge_amp_lin)
    surge_ph = np.interp(omega_geo, omega_rao_lin, surge_ph_lin)
    pitch_amp_rad = np.interp(omega_geo, omega_rao_lin, pitch_amp_rad_lin)
    pitch_ph = np.interp(omega_geo, omega_rao_lin, pitch_ph_lin)
    omega_rao = omega_geo  # use geometric grid for all first-order work

    # ---- Load heave & pitch RAOs (for vertical motion synthesis) ----
    print("  Loading heave & pitch RAOs and radiation data...")
    _, heave_amp_lin, heave_ph_lin, pitch_amp_heave_deg_lin, pitch_ph_heave_lin = \
        parse_rao_heave_pitch(RAO_FILE)
    heave_amp = np.interp(omega_geo, omega_rao_lin, heave_amp_lin)
    heave_ph = np.interp(omega_geo, omega_rao_lin, heave_ph_lin)

    # Pitch RAO for vertical motion (same as already loaded, but
    # we keep the heave parser's pitch output for consistency check)
    # pitch_amp_rad and pitch_ph already interpolated above — reuse them.

    # ---- Load radiation coefficients for pitch damping correction ----
    omega_rad, A55_arr, B55_arr = parse_radiation_coefficients(RAD_FILE, 5, 5)
    _, A33_arr, B33_arr = parse_radiation_coefficients(RAD_FILE, 3, 3)
    _, A35_arr, B35_arr = parse_radiation_coefficients(RAD_FILE, 3, 5)

    # ---- Complex RAOs on geometric grid (needed before damping correction) ----
    X_surge = surge_amp * np.exp(1j * surge_ph)
    Theta_uncorr = pitch_amp_rad * np.exp(1j * pitch_ph)

    # ---- Bimodal wave spectrum on geometric grid ----
    # (needed early for the self-consistent pitch damping iteration)
    S_wave = bimodal_spectrum(omega_rao, Hs_ws, Tp_ws, gamma_ws,
                              Hs_sw, Tp_sw, gamma_sw)

    # ---- Self-consistent Morison viscous pitch damping ----
    # Nemoh gives only potential-flow (radiation) damping. For a deep-draft
    # spar, viscous drag on the hull provides significant additional pitch
    # damping, particularly near the pitch natural period.
    #
    # We compute B55_visc from first-principles Morison drag on the hull,
    # using stochastic linearization (Borgman 1967):
    #   B55_visc = (8/(3π)) × 0.5 × ρ × Cd × σ_θ̇ × I_drag
    # where I_drag = ∫ D(z) × |r(z)|³ dz over the draft, r(z) is the
    # moment arm from the pitch axis (CG), and σ_θ̇ is the rms pitch
    # velocity. Since σ_θ̇ depends on the damping, we iterate to
    # self-consistency (~5 iterations).
    #
    # OC3 spar geometry:
    #   SWL to -4m:    D = 6.5 m (upper cylinder)
    #   -4m to -12m:   taper 6.5→9.4 m
    #   -12m to -120m: D = 9.4 m (lower cylinder)
    #   CG at z = -78m below SWL (Jonkman 2010)

    # Pitch natural frequency (from RAO peak — look for max pitch response)
    idx_pitch_peak = np.argmax(pitch_amp_rad_lin)
    omega_pitch_n = omega_rao_lin[idx_pitch_peak]
    T_pitch_n = 2 * np.pi / omega_pitch_n
    print(f"    Pitch natural period (from RAO peak): {T_pitch_n:.1f} s "
          f"(ω = {omega_pitch_n:.3f} rad/s)")

    # Pitch inertia at natural frequency
    A55_at_peak = np.interp(omega_pitch_n, omega_rad, A55_arr)
    I55 = 2.02e10  # kg·m² — OC3 pitch moment of inertia (Jonkman 2010)
    I55_eff = I55 + A55_at_peak
    B55_rad_peak = np.interp(omega_pitch_n, omega_rad, B55_arr)

    # Critical damping for pitch
    # For OC3: K55 ≈ ω_n² * I55_eff (from the resonance condition)
    K55 = omega_pitch_n**2 * I55_eff
    B55_crit = 2 * np.sqrt(K55 * I55_eff)  # = 2 * I55_eff * omega_n

    zeta_rad = B55_rad_peak / B55_crit
    print(f"    B55_crit = {B55_crit:.3e} N·m·s/rad, "
          f"radiation ζ = {zeta_rad:.4f}")

    # --- Compute I_drag = ∫ D(z) × |r(z)|³ dz ---
    # Numerical integration over the spar draft with fine spacing
    z_cg = -78.0   # m below SWL (pitch axis at CG)
    n_int = 2000    # integration points
    z_int = np.linspace(0.0, -120.0, n_int)   # SWL to keel
    dz_int = np.abs(z_int[1] - z_int[0])

    # Diameter profile D(z)
    D_profile = np.empty(n_int)
    for iz in range(n_int):
        z = z_int[iz]
        if z >= -4.0:
            D_profile[iz] = 6.5      # upper cylinder
        elif z >= -12.0:
            # Linear taper from 6.5 at -4m to 9.4 at -12m
            frac = (z - (-4.0)) / (-12.0 - (-4.0))  # 0 at -4m, 1 at -12m
            D_profile[iz] = 6.5 + frac * (9.4 - 6.5)
        else:
            D_profile[iz] = 9.4      # lower cylinder

    # Moment arms from CG
    r_arm = z_int - z_cg   # positive above CG, negative below CG

    # Morison drag integral: I_drag = ∫ D(z) × |r(z)|³ dz
    I_drag = np.sum(D_profile * np.abs(r_arm)**3 * dz_int)

    # Cd for pitch drag (post-critical Re, smooth/marine growth)
    Cd_pitch = 1.0
    print(f"    I_drag = {I_drag:.3e} m⁵  (Morison drag integral)")
    print(f"    Cd = {Cd_pitch:.1f} (post-critical Re)")

    # --- Self-consistent iteration ---
    # We iterate on B55_visc. At each step:
    #   1. Compute correction factor = |H_total(ω)| / |H_rad(ω)|
    #      where H_rad has Nemoh's radiation damping only and
    #      H_total adds B55_visc.
    #   2. Apply correction to Nemoh pitch RAO: RAO_corr = RAO_nemoh × factor
    #   3. Compute σ_θ̇ = sqrt(∫ ω² × |RAO_corr|² × S_wave dω)
    #   4. Update B55_visc from Morison linearization
    A55_geo = np.interp(omega_geo, omega_rad, A55_arr)
    B55_geo = np.interp(omega_geo, omega_rad, B55_arr)
    I_eff_geo = I55 + A55_geo

    # Pre-compute H_rad (radiation-only transfer function, what Nemoh used)
    H_rad_iter = 1.0 / (-I_eff_geo * omega_geo**2
                         + 1j * omega_geo * B55_geo + K55)

    sigma_thetadot = 0.01  # rad/s initial guess
    MAX_ITER = 20
    TOL = 1e-4
    print(f"    Self-consistent iteration (tol={TOL}):")

    for it in range(MAX_ITER):
        # Linearized viscous damping coefficient
        B55_visc_iter = (8.0 / (3.0 * np.pi)) * 0.5 * RHO * Cd_pitch \
                        * sigma_thetadot * I_drag

        # Transfer function with total damping
        H_total_iter = 1.0 / (-I_eff_geo * omega_geo**2
                               + 1j * omega_geo * (B55_geo + B55_visc_iter)
                               + K55)

        # Correction factor applied to Nemoh RAO
        corr_factor = np.abs(H_total_iter) / np.abs(H_rad_iter)
        pitch_rao_corr = pitch_amp_rad * corr_factor  # [rad/m]

        # Pitch velocity response spectrum:
        #   S_θ̇(ω) = ω² × |RAO_corr(ω)|² × S_wave(ω)
        S_pitch_vel = omega_geo**2 * pitch_rao_corr**2 * S_wave

        # New sigma_theta_dot
        sigma_thetadot_new = np.sqrt(trapz(S_pitch_vel, omega_geo))

        # Convergence check
        rel_change = abs(sigma_thetadot_new - sigma_thetadot) / max(sigma_thetadot, 1e-10)
        zeta_iter = (B55_rad_peak + B55_visc_iter) / B55_crit

        if it < 5 or rel_change < TOL:
            print(f"      iter {it:2d}: σ_θ̇ = {sigma_thetadot_new:.5f} rad/s, "
                  f"B55_visc = {B55_visc_iter:.3e}, ζ_total = {zeta_iter:.4f}")

        if rel_change < TOL and it > 0:
            print(f"      Converged after {it+1} iterations")
            break
        sigma_thetadot = sigma_thetadot_new

    B55_visc = B55_visc_iter
    zeta_visc = B55_visc / B55_crit
    zeta_total = zeta_rad + zeta_visc
    zeta_target = zeta_total  # use converged value (not hardcoded!)

    # Report hull velocities for physical sanity check
    # At SWL: r = 78m, u_rms = sigma_thetadot * 78
    r_swl = abs(0.0 - z_cg)  # 78m
    u_rms_swl = sigma_thetadot * r_swl
    print(f"    Converged: B55_visc = {B55_visc:.3e} N·m·s/rad, "
          f"ζ_visc = {zeta_visc:.4f}, ζ_total = {zeta_total:.4f}")
    print(f"    Hull velocity at SWL (r={r_swl:.0f}m): "
          f"u_rms = {u_rms_swl:.2f} m/s, u_3σ = {3*u_rms_swl:.2f} m/s")

    # Apply pitch damping correction to the pitch RAO
    # The Nemoh RAO was computed with radiation damping only.
    # With additional viscous damping, the transfer function changes:
    #   H(ω) = 1 / [-I_eff*ω² + i*ω*(B_rad + B_visc) + K55]
    # The correction factor at each frequency:
    #   RAO_corrected = RAO_nemoh * |H_rad(ω)| / |H_total(ω)|
    # where H_rad uses B55_rad(ω) and H_total uses B55_rad(ω) + B55_visc
    # (A55_geo, B55_geo, I_eff_geo already computed in the iteration above)

    # Transfer function with radiation damping only (what Nemoh computed)
    H_rad = 1.0 / (-I_eff_geo * omega_geo**2 + 1j * omega_geo * B55_geo + K55)
    # Transfer function with total damping
    H_total = 1.0 / (-I_eff_geo * omega_geo**2
                      + 1j * omega_geo * (B55_geo + B55_visc) + K55)
    # Correction factor: reduce RAO where viscous damping matters
    pitch_damping_factor = np.abs(H_total) / np.abs(H_rad)

    # Apply to pitch RAO (affects both surge coupling and vertical motion)
    pitch_amp_rad_corrected = pitch_amp_rad * pitch_damping_factor
    # Update the phase: the added damping shifts the phase slightly
    pitch_ph_corrected = np.angle(
        np.abs(H_total) / np.abs(H_rad) * np.exp(1j * pitch_ph)
        * np.exp(1j * (np.angle(H_total) - np.angle(H_rad))))

    # Find the index of omega_pitch_n on the geometric grid for display
    idx_pitch_geo = np.argmin(np.abs(omega_geo - omega_pitch_n))
    print(f"    Pitch RAO correction at peak: factor = "
          f"{pitch_damping_factor[idx_pitch_geo]:.4f} "
          f"(ω = {omega_geo[idx_pitch_geo]:.3f} rad/s)")
    # Recompute apparent surge RAO with corrected pitch
    Theta_corrected = pitch_amp_rad_corrected * np.exp(1j * pitch_ph_corrected)
    X_app_corrected = X_surge + z_above_swl * Theta_corrected
    RAO_app = np.abs(X_app_corrected)
    phase_app = np.angle(X_app_corrected)

    # Also store corrected pitch for vertical motion synthesis
    pitch_amp_for_vert = pitch_amp_rad_corrected
    pitch_ph_for_vert = pitch_ph_corrected

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
    # RAO_app and phase_app already computed above with pitch damping correction
    # S_wave already computed above (before damping iteration)

    # Response spectrum (for σ target on geometric grid)
    S_surge_1st = RAO_app**2 * S_wave
    sigma_1st = np.sqrt(trapz(S_surge_1st, omega_rao))

    # Synthesize first-order with CORRECT RAO phase on GEOMETRIC grid
    # x(t) = Σ |RAO| * sqrt(2 * S_wave * Δω_i) * cos(ω_i*t + φ_wave + angle(RAO))
    # For geometric spacing, Δω varies — use per-frequency spacing
    dw = np.diff(omega_rao)
    dw = np.append(dw, dw[-1])  # extend to match length
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
    mu_lin, S_F = compute_slow_drift_force_spectrum(qtf_omegas, qtf_T, S_total_qtf)
    H_lin = surge_transfer_function(mu_lin, MASS, A11_LOW, B_EXT, K_SURGE)
    S_sd_lin = np.abs(H_lin)**2 * S_F
    sigma_sd = np.sqrt(trapz(S_sd_lin, mu_lin))

    # Interpolate slow-drift response spectrum onto geometric mu grid
    # (mu is difference-frequency, typically 0 to ~0.2 rad/s)
    mu_pos = mu_lin[mu_lin > 0]
    S_sd_pos = S_sd_lin[mu_lin > 0]
    if len(mu_pos) > 2:
        N_MU_GEO = 150
        mu_geo = make_geometric_omega(mu_pos[0], mu_pos[-1], N_MU_GEO)
        S_sd_geo = np.interp(mu_geo, mu_pos, S_sd_pos)
    else:
        mu_geo = mu_pos
        S_sd_geo = S_sd_pos

    surge_sd = synthesize_from_spectrum(mu_geo, S_sd_geo, t,
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

    # --- 5. Current variability surge ---
    # Slowly varying surge from ocean current fluctuations (tidal, inertial,
    # internal waves). The response spectrum S_x(f) is computed by
    # compute_current_variability_surge, then synthesized into a time series.
    cv_result = compute_current_variability_surge(
        Uc, sigma_Uc, T_peak_current,
        spectrum_type='generic', verbose=False)
    if cv_result is not None:
        # cv_result['f'] is in Hz, cv_result['S_x'] in m²/Hz
        # Convert to omega [rad/s] and S_x [m²/(rad/s)] for synthesis
        f_cv = cv_result['f']
        omega_cv = 2 * np.pi * f_cv
        S_x_cv_omega = cv_result['S_x'] / (2 * np.pi)  # m²/Hz → m²/(rad/s)
        sigma_cv = cv_result['sigma']

        # Interpolate onto a geometric omega grid for synthesis
        # Filter to the relevant frequency range (avoid extremely low freqs
        # that would produce periods >> simulation duration)
        f_min_cv = max(1.0 / duration, f_cv[f_cv > 0][0])  # at least 1 cycle
        f_max_cv = 0.05  # 20s period — well above surge resonance
        omega_min_cv = 2 * np.pi * f_min_cv
        omega_max_cv = 2 * np.pi * f_max_cv
        N_CV_GEO = 200
        omega_cv_geo = make_geometric_omega(omega_min_cv, omega_max_cv, N_CV_GEO)
        S_cv_geo = np.interp(omega_cv_geo, omega_cv, S_x_cv_omega,
                             left=0.0, right=0.0)

        surge_cv = synthesize_from_spectrum(
            omega_cv_geo, S_cv_geo, t,
            rng=np.random.default_rng(master_rng.integers(1e9)))
        # Also add the rectification mean shift
        x_cv_mean_shift = cv_result['x_mean_shift']
        print(f"    Current variability surge: σ={np.std(surge_cv):.3f} m "
              f"(target {sigma_cv:.3f}), "
              f"rect shift={x_cv_mean_shift:.3f} m")
    else:
        surge_cv = np.zeros(n_t)
        x_cv_mean_shift = 0.0
        sigma_cv = 0.0
        print(f"    Current variability surge: SKIPPED (mooring capacity exceeded)")

    # --- Total surge ---
    # Note: current variability mean shift is added on top of the mean offset.
    # The dynamic component surge_cv oscillates about zero.
    surge_total = (x_mean_total + x_cv_mean_shift
                   + surge_1st + surge_sd + surge_wind + surge_cv)

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

    # ============================================================
    # HEAVE & PITCH — first-order vertical motion
    # ============================================================
    # The vertical motion at deck level (z above SWL) combines heave and
    # pitch: z_vert(t) = heave(t) + z_above_swl * pitch(t)  [small angles]
    #
    # Both heave and pitch see the SAME incident waves as surge, so they
    # must use the SAME wave phase realizations (phi_wave) to preserve
    # the correct phase relationships between DOFs.
    print("  Synthesizing heave & pitch components...")

    # --- 8. First-order heave ---
    # heave(t) = Σ |H_heave(ωi)| * sqrt(2 * S(ωi) * Δωi) * cos(ωi*t + φ_wave + phase_heave)
    A_heave = heave_amp * np.sqrt(2 * S_wave * dw)
    heave_1st = np.zeros(n_t)
    for i0 in range(0, len(omega_rao), chunk):
        i1 = min(i0 + chunk, len(omega_rao))
        wt = np.outer(omega_rao[i0:i1], t)
        heave_1st += np.sum(
            A_heave[i0:i1, None] * np.cos(wt + phi_wave[i0:i1, None]
                                           + heave_ph[i0:i1, None]),
            axis=0)

    S_heave_1st = heave_amp**2 * S_wave
    sigma_heave = np.sqrt(trapz(S_heave_1st, omega_rao))
    print(f"    1st-order heave: σ={np.std(heave_1st):.3f} m "
          f"(target {sigma_heave:.3f})")

    # --- 9. First-order pitch (with viscous damping correction) ---
    # pitch(t) = Σ |Θ(ωi)| * sqrt(2 * S(ωi) * Δωi) * cos(ωi*t + φ_wave + phase_pitch)
    # Using corrected pitch RAO (pitch_amp_for_vert in rad/m)
    A_pitch = pitch_amp_for_vert * np.sqrt(2 * S_wave * dw)
    pitch_1st = np.zeros(n_t)  # [rad]
    for i0 in range(0, len(omega_rao), chunk):
        i1 = min(i0 + chunk, len(omega_rao))
        wt = np.outer(omega_rao[i0:i1], t)
        pitch_1st += np.sum(
            A_pitch[i0:i1, None] * np.cos(wt + phi_wave[i0:i1, None]
                                           + pitch_ph_for_vert[i0:i1, None]),
            axis=0)

    S_pitch_1st = pitch_amp_for_vert**2 * S_wave
    sigma_pitch = np.sqrt(trapz(S_pitch_1st, omega_rao))
    print(f"    1st-order pitch: σ={np.degrees(np.std(pitch_1st)):.3f} deg "
          f"(target {np.degrees(sigma_pitch):.3f} deg)")

    # --- 10. Vertical motion at deck level ---
    # z_vertical(t) = heave(t) + z_above_swl * sin(pitch(t))
    # For small angles: sin(θ) ≈ θ [θ in rad]
    z_vertical = heave_1st + z_above_swl * pitch_1st
    sigma_vert = np.std(z_vertical)
    pp_vert = np.max(z_vertical) - np.min(z_vertical)

    # Also compute the heave-only and pitch-contribution separately for breakdown
    pitch_contribution = z_above_swl * pitch_1st

    print(f"    Vertical at z={z_above_swl:.0f}m: σ={sigma_vert:.3f} m, "
          f"pp={pp_vert:.2f} m")
    print(f"      Heave alone:          σ={np.std(heave_1st):.3f} m")
    print(f"      Pitch at {z_above_swl:.0f}m:        "
          f"σ={np.std(pitch_contribution):.3f} m")

    print()
    print(f"  TOTAL SURGE: mean={np.mean(surge_total):.2f} m, "
          f"σ={np.std(surge_total):.2f} m, "
          f"range=[{np.min(surge_total):.1f}, {np.max(surge_total):.1f}] m")
    print(f"  TOTAL SWAY:  mean={np.mean(sway_total):.3f} m, "
          f"σ={np.std(sway_total):.2f} m, "
          f"range=[{np.min(sway_total):.1f}, {np.max(sway_total):.1f}] m")
    print(f"  VERTICAL:    σ={sigma_vert:.3f} m, pp={pp_vert:.2f} m")

    # ============================================================
    # PLATFORM VELOCITIES (time derivative of displacement)
    # ============================================================
    # np.gradient gives central differences (2nd-order accurate), which
    # preserves the frequency content up to the Nyquist frequency.
    # For DP operations, the platform velocity directly determines the
    # required DP vessel tracking speed and thruster demand.
    print("  Computing platform velocities...")

    surge_vel = np.gradient(surge_total, dt)  # m/s
    sway_vel = np.gradient(sway_total, dt)    # m/s

    # Component velocities (for breakdown)
    surge_1st_vel = np.gradient(surge_1st, dt)
    surge_sd_vel = np.gradient(surge_sd, dt)
    surge_wind_vel = np.gradient(surge_wind, dt)
    surge_cv_vel = np.gradient(surge_cv, dt)
    sway_vim_vel = np.gradient(sway_vim, dt)
    sway_1st_vel = np.gradient(sway_1st, dt)

    # Vertical velocity at deck level
    vert_vel = np.gradient(z_vertical, dt)     # m/s
    heave_vel = np.gradient(heave_1st, dt)     # m/s
    pitch_vel = np.gradient(pitch_1st, dt)     # rad/s

    # Horizontal speed (magnitude)
    horiz_speed = np.sqrt(surge_vel**2 + sway_vel**2)

    print(f"  SURGE VEL: σ={np.std(surge_vel):.3f} m/s, "
          f"max={np.max(np.abs(surge_vel)):.3f} m/s")
    print(f"  SWAY VEL:  σ={np.std(sway_vel):.3f} m/s, "
          f"max={np.max(np.abs(sway_vel)):.3f} m/s")
    print(f"  VERT VEL:  σ={np.std(vert_vel):.3f} m/s, "
          f"max={np.max(np.abs(vert_vel)):.3f} m/s")
    print(f"  HORIZ SPEED: max={np.max(horiz_speed):.3f} m/s "
          f"({np.max(horiz_speed)*1.944:.2f} kn)")

    # ============================================================
    # PLOTTING
    # ============================================================
    print("\n  Generating plots...")

    t_min = t / 60  # convert to minutes

    fig = plt.figure(figsize=(18, 28))
    gs = GridSpec(8, 2, figure=fig,
                  height_ratios=[2, 1, 2, 1, 2, 1, 1, 1],
                  hspace=0.40, wspace=0.3)

    # ---- Panel 1: Surge displacement (row 0, wide) ----
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

    # ---- Panel 2: Surge velocity (row 1, wide) ----
    ax1v = fig.add_subplot(gs[1, :], sharex=ax1)
    ax1v.fill_between(t_min, 0, surge_vel, where=surge_vel >= 0,
                      alpha=0.25, color='C0', interpolate=True)
    ax1v.fill_between(t_min, 0, surge_vel, where=surge_vel < 0,
                      alpha=0.25, color='C3', interpolate=True)
    ax1v.plot(t_min, surge_vel, 'k-', lw=0.3, alpha=0.7)
    ax1v.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
    ax1v.set_ylabel('Surge vel [m/s]', fontsize=12)
    max_surge_vel = np.max(np.abs(surge_vel))
    ax1v.set_title(
        f'Surge velocity  |  '
        f'σ = {np.std(surge_vel):.3f} m/s, '
        f'max = {max_surge_vel:.3f} m/s ({max_surge_vel*1.944:.2f} kn)',
        fontsize=11, fontweight='bold')
    ax1v.set_xlim(0, duration/60)
    ax1v.grid(True, alpha=0.3)

    # ---- Panel 3: Sway displacement (row 2, wide) ----
    ax2 = fig.add_subplot(gs[2, :], sharex=ax1)
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

    # ---- Panel 4: Sway velocity (row 3, wide) ----
    ax2v = fig.add_subplot(gs[3, :], sharex=ax1)
    ax2v.fill_between(t_min, 0, sway_vel, where=sway_vel >= 0,
                      alpha=0.25, color='C1', interpolate=True)
    ax2v.fill_between(t_min, 0, sway_vel, where=sway_vel < 0,
                      alpha=0.25, color='C5', interpolate=True)
    ax2v.plot(t_min, sway_vel, 'k-', lw=0.3, alpha=0.7)
    ax2v.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
    ax2v.set_ylabel('Sway vel [m/s]', fontsize=12)
    max_sway_vel = np.max(np.abs(sway_vel))
    ax2v.set_title(
        f'Sway velocity  |  '
        f'σ = {np.std(sway_vel):.3f} m/s, '
        f'max = {max_sway_vel:.3f} m/s ({max_sway_vel*1.944:.2f} kn)',
        fontsize=11, fontweight='bold')
    ax2v.set_xlabel('Time [min]', fontsize=11)
    ax2v.set_xlim(0, duration/60)
    ax2v.grid(True, alpha=0.3)

    # ---- Panel 5: Vertical displacement at deck level (row 4, wide) ----
    ax_vd = fig.add_subplot(gs[4, :], sharex=ax1)
    ax_vd.fill_between(t_min, 0, z_vertical, alpha=0.25, color='C4')
    ax_vd.plot(t_min, z_vertical, 'C4-', lw=0.4, alpha=0.8,
               label=f'Total vertical (σ={sigma_vert:.2f} m)')
    ax_vd.plot(t_min, heave_1st, 'C9-', lw=0.3, alpha=0.4,
               label=f'Heave only (σ={np.std(heave_1st):.2f} m)')
    ax_vd.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
    ax_vd.set_ylabel('Vertical [m]', fontsize=12)
    max_vert = np.max(np.abs(z_vertical))
    ax_vd.set_title(
        f'Vertical motion at z = {z_above_swl:.0f}m (heave + pitch coupling)  |  '
        f'σ = {sigma_vert:.2f} m, pp = {pp_vert:.1f} m, '
        f'pitch damping ζ = {zeta_target:.1%} (Morison self-consistent)',
        fontsize=12, fontweight='bold')
    ax_vd.legend(loc='upper right', fontsize=10)
    ax_vd.set_xlim(0, duration/60)
    ax_vd.grid(True, alpha=0.3)

    # ---- Panel 6: Vertical velocity (row 5, wide) ----
    max_vert_vel = np.max(np.abs(vert_vel))
    ax_vv = fig.add_subplot(gs[5, :], sharex=ax1)
    ax_vv.fill_between(t_min, 0, vert_vel, where=vert_vel >= 0,
                       alpha=0.25, color='C4', interpolate=True)
    ax_vv.fill_between(t_min, 0, vert_vel, where=vert_vel < 0,
                       alpha=0.25, color='C6', interpolate=True)
    ax_vv.plot(t_min, vert_vel, 'k-', lw=0.3, alpha=0.7)
    ax_vv.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
    ax_vv.set_ylabel('Vert vel [m/s]', fontsize=12)
    ax_vv.set_title(
        f'Vertical velocity at z = {z_above_swl:.0f}m  |  '
        f'σ = {np.std(vert_vel):.3f} m/s, '
        f'max = {max_vert_vel:.3f} m/s',
        fontsize=11, fontweight='bold')
    ax_vv.set_xlim(0, duration/60)
    ax_vv.grid(True, alpha=0.3)

    # ---- Panel 7: Surge component breakdown (row 6 left) ----
    ax3 = fig.add_subplot(gs[6, 0])
    # Find the 5-minute window with the largest surge peak-to-peak motion
    # (sliding window search, 0.5 min steps)
    window_dur = 5.0  # minutes
    best_pp = 0.0
    best_t_start = 0.0
    for t_try in np.arange(0, duration/60 - window_dur, 0.5):
        m = (t_min >= t_try) & (t_min <= t_try + window_dur)
        pp = np.max(surge_1st[m]) - np.min(surge_1st[m])
        if pp > best_pp:
            best_pp = pp
            best_t_start = t_try
    t_start_detail = best_t_start
    t_end_detail = best_t_start + window_dur
    mask = (t_min >= t_start_detail) & (t_min <= t_end_detail)
    print(f"  Detail window: {t_start_detail:.1f}–{t_end_detail:.1f} min "
          f"(largest 1st-order pp = {best_pp:.2f} m)")

    ax3.plot(t_min[mask], surge_1st[mask], 'C0-', lw=0.8,
             label=f'1st-order (σ={np.std(surge_1st):.2f} m)', alpha=0.8)
    ax3.plot(t_min[mask], surge_sd[mask], 'C3-', lw=1.2,
             label=f'Slow-drift (σ={np.std(surge_sd):.2f} m)')
    ax3.plot(t_min[mask], surge_wind[mask], 'C2-', lw=1.0,
             label=f'Wind turb (σ={np.std(surge_wind):.2f} m)')
    ax3.plot(t_min[mask], surge_cv[mask], 'C5-', lw=1.5,
             label=f'Current var (σ={np.std(surge_cv):.2f} m)')
    ax3.set_xlabel('Time [min]', fontsize=11)
    ax3.set_ylabel('Surge component [m]', fontsize=11)
    ax3.set_title('Surge components (5-min detail)', fontsize=11)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ---- Panel 8: XY trajectory (row 6 right) ----
    ax4 = fig.add_subplot(gs[6, 1])
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

    # ---- Panel 9: Sway component breakdown (row 7 left) ----
    ax5 = fig.add_subplot(gs[7, 0])
    # Use same detail window as surge
    ax5.plot(t_min[mask], sway_vim[mask], 'C2-', lw=1.0,
             label=f'VIM (σ={np.std(sway_vim):.2f} m)', alpha=0.8)
    ax5.plot(t_min[mask], sway_1st[mask], 'C0-', lw=0.8,
             label=f'1st-order (σ={np.std(sway_1st):.2f} m)', alpha=0.7)
    ax5.plot(t_min[mask], sway_slow[mask], 'C4-', lw=1.2,
             label=f'Slow drift (σ={np.std(sway_slow):.2f} m)')
    ax5.set_xlabel('Time [min]', fontsize=11)
    ax5.set_ylabel('Sway component [m]', fontsize=11)
    ax5.set_title('Sway components (5-min detail)', fontsize=11)
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ---- Panel 10: Summary statistics (row 7 right) ----
    ax6 = fig.add_subplot(gs[7, 1])
    ax6.axis('off')

    # Compute statistics
    pp_surge = np.max(surge_total) - np.min(surge_total)
    pp_sway = np.max(sway_total) - np.min(sway_total)

    x_mean_with_rect = x_mean_total + x_cv_mean_shift

    stats_text = (
        f"MOTION SUMMARY — OC3 Hywind spar at z = {z_above_swl:.0f}m\n"
        f"{'─' * 50}\n"
        f"\n"
        f"SURGE:\n"
        f"  Mean offset:          {x_mean_with_rect:+.2f} m\n"
        f"    (current {current['x_mean']:.2f} + wind {wind['x_mean']:.2f} "
        f"+ drift {x_mean_drift:.3f}"
        f" + rect {x_cv_mean_shift:.3f})\n"
        f"  σ (1st-order):        {np.std(surge_1st):.2f} m "
        f"  (swell-dominated)\n"
        f"  σ (slow-drift):       {np.std(surge_sd):.2f} m\n"
        f"  σ (wind turbulence):  {np.std(surge_wind):.2f} m\n"
        f"  σ (current var):      {np.std(surge_cv):.2f} m"
        f"  (σ_Uc={sigma_Uc:.2f}, T={T_peak_current/60:.0f}min)\n"
        f"  σ (total dynamic):    {np.std(surge_total - x_mean_with_rect):.2f} m\n"
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
        f"VERTICAL at z = {z_above_swl:.0f}m:\n"
        f"  σ (heave):            {np.std(heave_1st):.2f} m\n"
        f"  σ (pitch at {z_above_swl:.0f}m):     {np.std(pitch_contribution):.2f} m\n"
        f"  σ (total vertical):   {sigma_vert:.2f} m\n"
        f"  Peak-to-peak:         {pp_vert:.1f} m\n"
        f"  Pitch ζ (total):      {zeta_target:.1%}  "
        f"(rad {zeta_rad:.2%} + visc {zeta_visc:.2%})\n"
        f"  Pitch T_n:            {T_pitch_n:.1f} s\n"
        f"\n"
        f"VELOCITY (DP tracking demand):\n"
        f"  Surge vel:  σ = {np.std(surge_vel):.3f} m/s, "
        f"max = {max_surge_vel:.3f} m/s\n"
        f"  Sway vel:   σ = {np.std(sway_vel):.3f} m/s, "
        f"max = {max_sway_vel:.3f} m/s\n"
        f"  Vert vel:   σ = {np.std(vert_vel):.3f} m/s, "
        f"max = {max_vert_vel:.3f} m/s\n"
        f"  Max horiz speed:      {np.max(horiz_speed):.3f} m/s "
        f"({np.max(horiz_speed)*1.944:.2f} kn)\n"
        f"\n"
        f"Note: VIM Vr = {Uc/(OMEGA_N/(2*np.pi)*SPAR_UPPER_D):.1f} "
        f"(near upper lock-in boundary)\n"
        f"A/D ≈ {A_vim_lock/SPAR_UPPER_D:.2f} "
        f"(post-critical Re, self-limiting)"
    )

    ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes,
             fontsize=8.0, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(
        'OC3 Hywind Spar — Platform Motion & Velocity During DP Alongside Operations\n'
        '(Synthesized time-domain realization, turbine stopped)',
        fontsize=14, fontweight='bold', y=0.995)

    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'platform_motion.png')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to: {outfile}")
    plt.close()

    # ============================================================
    # Also make a clean 6-panel version (surge + vel + sway + vel + vert + vel)
    # ============================================================
    fig2, (ax_s, ax_sv, ax_y, ax_yv, ax_z, ax_zv) = plt.subplots(
        6, 1, figsize=(16, 20), sharex=True,
        gridspec_kw={'height_ratios': [2, 1, 2, 1, 2, 1]})

    # Surge displacement
    ax_s.fill_between(t_min, x_mean_with_rect, surge_total,
                      alpha=0.25, color='C0')
    ax_s.plot(t_min, surge_total, 'C0-', lw=0.3, alpha=0.9)
    ax_s.axhline(x_mean_with_rect, color='C3', ls='--', lw=1.0,
                 label=f'Mean = {x_mean_with_rect:.1f} m')
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

    # Surge velocity
    ax_sv.fill_between(t_min, 0, surge_vel, where=surge_vel >= 0,
                       alpha=0.25, color='C0', interpolate=True)
    ax_sv.fill_between(t_min, 0, surge_vel, where=surge_vel < 0,
                       alpha=0.25, color='C3', interpolate=True)
    ax_sv.plot(t_min, surge_vel, 'k-', lw=0.25, alpha=0.7)
    ax_sv.axhline(0, color='k', ls='-', lw=0.4, alpha=0.3)
    ax_sv.set_ylabel('Surge vel [m/s]', fontsize=13)
    ax_sv.set_title(
        f'Surge velocity  |  '
        f'σ = {np.std(surge_vel):.3f} m/s, '
        f'max = {max_surge_vel:.3f} m/s ({max_surge_vel*1.944:.2f} kn)',
        fontsize=11, fontweight='bold')
    ax_sv.grid(True, alpha=0.3)

    # Sway displacement
    ax_y.fill_between(t_min, 0, sway_total, alpha=0.25, color='C1')
    ax_y.plot(t_min, sway_total, 'C1-', lw=0.3, alpha=0.9)
    ax_y.axhline(0, color='k', ls='-', lw=0.4, alpha=0.3)
    ax_y.set_ylabel('Sway [m]', fontsize=13)
    ax_y.set_title(
        f'Sway — VIM with intermittent lock-in  |  '
        f'A/D ≈ {A_vim_lock/SPAR_UPPER_D:.2f}, '
        f'T_VIM ≈ {T_N:.0f} s  |  '
        f'pp = {pp_sway:.1f} m',
        fontsize=12, fontweight='bold')
    ax_y.grid(True, alpha=0.3)

    # Sway velocity
    ax_yv.fill_between(t_min, 0, sway_vel, where=sway_vel >= 0,
                       alpha=0.25, color='C1', interpolate=True)
    ax_yv.fill_between(t_min, 0, sway_vel, where=sway_vel < 0,
                       alpha=0.25, color='C5', interpolate=True)
    ax_yv.plot(t_min, sway_vel, 'k-', lw=0.25, alpha=0.7)
    ax_yv.axhline(0, color='k', ls='-', lw=0.4, alpha=0.3)
    ax_yv.set_ylabel('Sway vel [m/s]', fontsize=13)
    ax_yv.set_title(
        f'Sway velocity  |  '
        f'σ = {np.std(sway_vel):.3f} m/s, '
        f'max = {max_sway_vel:.3f} m/s ({max_sway_vel*1.944:.2f} kn)',
        fontsize=11, fontweight='bold')
    ax_yv.grid(True, alpha=0.3)

    # Vertical displacement at deck level
    ax_z.fill_between(t_min, 0, z_vertical, alpha=0.25, color='C4')
    ax_z.plot(t_min, z_vertical, 'C4-', lw=0.3, alpha=0.9,
              label=f'Heave + pitch (ζ_pitch = {zeta_target:.1%})')
    ax_z.plot(t_min, heave_1st, 'C9-', lw=0.25, alpha=0.4,
              label=f'Heave only (σ={np.std(heave_1st):.2f} m)')
    ax_z.axhline(0, color='k', ls='-', lw=0.4, alpha=0.3)
    ax_z.set_ylabel('Vertical [m]', fontsize=13)
    ax_z.set_title(
        f'Vertical at z = {z_above_swl:.0f}m (heave + pitch coupling)  |  '
        f'σ = {sigma_vert:.2f} m, pp = {pp_vert:.1f} m',
        fontsize=12, fontweight='bold')
    ax_z.legend(fontsize=10, loc='upper right')
    ax_z.grid(True, alpha=0.3)

    # Vertical velocity
    ax_zv.fill_between(t_min, 0, vert_vel, where=vert_vel >= 0,
                       alpha=0.25, color='C4', interpolate=True)
    ax_zv.fill_between(t_min, 0, vert_vel, where=vert_vel < 0,
                       alpha=0.25, color='C6', interpolate=True)
    ax_zv.plot(t_min, vert_vel, 'k-', lw=0.25, alpha=0.7)
    ax_zv.axhline(0, color='k', ls='-', lw=0.4, alpha=0.3)
    ax_zv.set_xlabel('Time [min]', fontsize=13)
    ax_zv.set_ylabel('Vert vel [m/s]', fontsize=13)
    ax_zv.set_title(
        f'Vertical velocity at z = {z_above_swl:.0f}m  |  '
        f'σ = {np.std(vert_vel):.3f} m/s, '
        f'max = {max_vert_vel:.3f} m/s',
        fontsize=11, fontweight='bold')
    ax_zv.grid(True, alpha=0.3)
    ax_zv.set_xlim(0, duration/60)

    fig2.suptitle(
        'OC3 Hywind — Platform Motion & Velocity (DP Alongside, Turbine Stopped)',
        fontsize=14, fontweight='bold', y=1.01)
    fig2.tight_layout()

    outfile2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'platform_motion_clean.png')
    fig2.savefig(outfile2, dpi=150, bbox_inches='tight')
    print(f"  Clean plot saved to: {outfile2}")
    plt.close()


if __name__ == '__main__':
    main()
