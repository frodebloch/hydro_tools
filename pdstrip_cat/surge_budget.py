#!/usr/bin/env python3
"""
surge_budget.py - Compare contributions to OC3 Hywind surge motion
                  during alongside operations (turbine stopped)

Computes and compares:
  1. First-order (wave-frequency) surge from Nemoh RAOs
  2. Second-order (slow-drift) surge from QTFs (hardcoded from prior analysis)
  3. Wind-driven surge on stopped turbine (Frøya spectrum + structural drag)
  4. Mean wind drag offset (tower + nacelle + parked blades + spar above SWL)
  5. Mean current drag offset on submerged spar

Physics:
  - Turbine is STOPPED (no aerodynamic thrust, blades feathered/parked)
  - Wind loading = drag on exposed structure (tower, nacelle, blades, spar freeboard)
  - Wind turbulence spectrum: Frøya (Andersen & Løvseth 1992 / ISO 19901-1)
  - Wind profile: power law with alpha=0.12 (open sea)
  - Current: steady drag on submerged spar, no dynamic component at surge resonance
  - Vortex shedding: assessed in Part 8. At low current speeds (0.2-0.5 m/s),
      shedding frequency approaches surge/sway resonance (lock-in possible)

References:
  - Andersen, O.J. & Løvseth, J. (1992). "The Frøya database and maritime
    boundary layer wind description." Marine Structures, 5(5), 421-449.
  - ISO 19901-1:2015. Petroleum and natural gas industries — Specific
    requirements for offshore structures — Part 1: Metocean design and
    operating considerations.
  - DNV-RP-C205 (2021). Environmental conditions and environmental loads.
  - Jonkman, J. (2010). Definition of the Floating System for Phase IV of OC3.
    NREL/TP-500-47535.
"""

import numpy as np

# Compatibility
trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

from scipy.optimize import brentq

# Import slow-drift QTF computation from companion script
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slow_drift_swell import (
    parse_qtf_total, compute_slow_drift_force_spectrum,
    surge_transfer_function, compute_statistics, compute_mean_drift,
    RHOG,
)

# ============================================================
# Physical parameters — OC3 Hywind
# ============================================================

RHO_WATER = 1025.0
RHO_AIR = 1.225       # kg/m^3
G = 9.81
RHOG = RHO_WATER * G

MASS = 1.40179e7       # kg (total system)
A11_LOW = 8.381e6      # kg (added mass in surge at low freq, from Nemoh)
B_EXT = 1.0e5          # N·s/m (external linear damping in surge)
K_SURGE = 4.118e4      # N/m (linearized mooring stiffness)

M_EFF = MASS + A11_LOW
OMEGA_N = np.sqrt(K_SURGE / M_EFF)
F_N = OMEGA_N / (2 * np.pi)
T_N = 1.0 / F_N
Q_FACTOR = M_EFF * OMEGA_N / B_EXT

DURATION_3H = 3 * 3600

# ============================================================
# OC3 Hywind geometry above and below waterline
# ============================================================

# Spar above waterline (freeboard)
SPAR_FREEBOARD = 10.0   # m above SWL
SPAR_DIAM_UPPER = 6.5   # m (upper column diameter)

# Tower (from platform top to hub)
# OC3: tower base at 10m above SWL, hub at 90m above SWL
# Tower is a tapered steel tube
TOWER_BASE_Z = 10.0     # m above SWL
TOWER_TOP_Z = 87.6      # m above SWL (yaw bearing)
HUB_HEIGHT = 90.0       # m above SWL
TOWER_DIAM_BASE = 6.0   # m (at base, roughly matches spar top)
TOWER_DIAM_TOP = 3.87   # m (at top)

# Nacelle (approximately a rectangular box)
NACELLE_LENGTH = 18.0   # m (along wind direction)
NACELLE_WIDTH = 6.0     # m
NACELLE_HEIGHT = 6.0    # m
NACELLE_Z = HUB_HEIGHT  # m above SWL (center height)
NACELLE_CD = 1.0        # bluff body

# Parked/feathered blades
# NREL 5MW: 3 blades, each 61.5m length
# When feathered (pitched to vane), they present minimal frontal area
# When parked/stopped in arbitrary position, projected area is significant
# Assume blades feathered (pitched ~90 deg): projected width ~ mean chord * cos(pitch)
# Mean chord ~ 3.5m, feathered presents edge-on ~ 0.3m effective width per blade
# Conservative: assume NOT fully feathered but in a stopped/idling position
BLADE_LENGTH = 61.5     # m
BLADE_MEAN_CHORD = 3.5  # m
N_BLADES = 3
# Feathered: blades present ~edge thickness ~0.3m projected width
# Parked (not feathered): project ~chord * sin(pitch_angle)
# We consider two cases below

# Spar below waterline
# Upper column: 0 to -12m, D=6.5m
# Taper: -12 to -4m (?), complex taper
# Actually OC3 geometry: upper D=6.5m from SWL to -4m, then taper to D=9.4m
# at -12m, then D=9.4m to -120m below SWL
SPAR_UPPER_D = 6.5      # m (0 to -4m)
SPAR_TAPER_TOP = -4.0   # m
SPAR_TAPER_BOT = -12.0  # m
SPAR_LOWER_D = 9.4      # m (-12m to -120m)
SPAR_DRAFT = 120.0      # m

# Drag coefficient for circular cylinder in current
CD_CYL_CURRENT = 1.05   # Re > 1e6, rough cylinder


# ============================================================
# Wind profile and Frøya spectrum
# ============================================================

ALPHA_SEA = 0.12  # power law exponent for open sea


def wind_profile(U10, z):
    """Mean wind speed at height z [m] given U10 (10-min mean at 10m)."""
    return U10 * (z / 10.0) ** ALPHA_SEA


def froya_spectrum(f, z, U10):
    """
    Frøya wind spectrum S_u(f) [m^2/s^2 / Hz].

    From Andersen & Løvseth (1992), as presented in ISO 19901-1 and
    DNV-RP-C205 Section 2.3:

        S(f) = 320 * (U10/10)^2 * (z/10)^0.45 / (1 + f_tilde^n)^(5/(3n))

    where:
        f_tilde = 172 * f * (z/10)^(2/3) * (U10/10)^(-0.75)
        n = 0.468

    Parameters
    ----------
    f : array-like, frequencies [Hz] (must be > 0)
    z : float, height above sea level [m]
    U10 : float, 10-min mean wind speed at 10m [m/s]

    Returns
    -------
    S : array, spectral density [m^2/s^2 / Hz]

    Notes
    -----
    The spectrum is single-sided (defined for f > 0).
    The total variance (integral over f) gives sigma_u^2.
    """
    f = np.asarray(f, dtype=float)
    S = np.zeros_like(f)
    mask = f > 0

    # Non-dimensional frequency
    n = 0.468
    f_tilde = 172.0 * f[mask] * (z / 10.0)**(2.0/3) * (U10 / 10.0)**(-0.75)

    # Spectral density
    S[mask] = 320.0 * (U10 / 10.0)**2 * (z / 10.0)**0.45 / \
              (1.0 + f_tilde**n)**(5.0 / (3.0 * n))

    return S


def froya_sigma_u(z, U10):
    """
    Turbulence standard deviation from Frøya model.

    Integrate the spectrum numerically to get sigma_u.
    Also available as closed-form approximation from ISO 19901-1:
        I_u(z) ≈ 0.06 * (1 + 0.043 * U10) * (z/10)^(-0.22)
    """
    I_u = 0.06 * (1.0 + 0.043 * U10) * (z / 10.0)**(-0.22)
    U_z = wind_profile(U10, z)
    return I_u * U_z


# ============================================================
# Wave spectrum
# ============================================================

def jonswap(omega, hs, tp, gamma=3.3):
    wp = 2 * np.pi / tp
    sigma = np.where(omega <= wp, 0.07, 0.09)
    b = np.exp(-(omega - wp)**2 / (2 * sigma**2 * wp**2))
    gamma_factor = gamma**b
    S_pm = (5.0 / 16.0) * hs**2 * wp**4 / omega**5 * np.exp(-1.25 * (wp / omega)**4)
    C_gamma = 1.0 - 0.287 * np.log(gamma)
    S = S_pm / C_gamma * gamma_factor
    S[omega < 0.01] = 0.0
    return S


# ============================================================
# 1. First-order surge from RAOs
# ============================================================

def parse_rao_surge(filepath, beta_target=0.0):
    """Parse Nemoh RAO.tec and extract surge RAO |X|(w) at given heading."""
    omega_list, amp_list = [], []
    found_zone = False
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('ZONE') or line.startswith('zone'):
                beta_str = line.split('beta=')[1].split('(deg)')[0].strip()
                beta = float(beta_str)
                if abs(beta - beta_target) < 0.1:
                    found_zone = True
                    omega_list, amp_list = [], []
                else:
                    if found_zone:
                        break
                continue
            if line.startswith('VARIABLES'):
                continue
            if found_zone:
                parts = line.split()
                omega_list.append(float(parts[0]))
                amp_list.append(float(parts[1]))
    return np.array(omega_list), np.array(amp_list)


def parse_rao_surge_pitch(filepath, beta_target=0.0):
    """
    Parse Nemoh RAO.tec and extract surge + pitch RAOs at given heading.

    Returns:
      omega: array of frequencies [rad/s]
      surge_amp: |X| [m/m]
      surge_phase: phase of X [rad]
      pitch_amp: |theta| [deg/m]
      pitch_phase: phase of theta [rad]
    """
    omega_list = []
    surge_amp_list, surge_phase_list = [], []
    pitch_amp_list, pitch_phase_list = [], []
    found_zone = False
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('ZONE') or line.startswith('zone'):
                beta_str = line.split('beta=')[1].split('(deg)')[0].strip()
                beta = float(beta_str)
                if abs(beta - beta_target) < 0.1:
                    found_zone = True
                    omega_list = []
                    surge_amp_list, surge_phase_list = [], []
                    pitch_amp_list, pitch_phase_list = [], []
                else:
                    if found_zone:
                        break
                continue
            if line.startswith('VARIABLES'):
                continue
            if found_zone:
                parts = line.split()
                omega_list.append(float(parts[0]))
                surge_amp_list.append(float(parts[1]))     # |X| [m/m]
                pitch_amp_list.append(float(parts[5]))      # |theta| [deg/m]
                surge_phase_list.append(float(parts[7]))    # ang(x) [deg]
                pitch_phase_list.append(float(parts[11]))   # ang(theta) [deg]

    return (np.array(omega_list),
            np.array(surge_amp_list),
            np.radians(np.array(surge_phase_list)),
            np.array(pitch_amp_list),
            np.radians(np.array(pitch_phase_list)))


def compute_first_order_surge(rao_file, hs, tp, gamma=3.3):
    """First-order wave-frequency surge statistics from RAOs."""
    omega, rao = parse_rao_surge(rao_file)
    S = jonswap(omega, hs, tp, gamma=gamma)
    S_surge = rao**2 * S
    var = trapz(S_surge, omega)
    sigma = np.sqrt(var) if var > 0 else 0.0

    m0 = var
    m2 = trapz(S_surge * omega**2, omega)
    if m2 > 0:
        Tz = 2 * np.pi * np.sqrt(m0 / m2)
        N = DURATION_3H / Tz
    else:
        Tz = tp
        N = 1
    x_max = sigma * np.sqrt(2 * np.log(max(N, 1)))

    return {
        'sigma': sigma,
        'sig_amp': 2 * sigma,
        'x_max': x_max,
        'Tz': Tz,
        'N_cycles': N,
    }


def compute_apparent_surge_at_z(rao_file, hs, tp, z_above_swl, gamma=3.3):
    """
    Compute apparent surge at height z above SWL.

    At elevation z, the apparent horizontal displacement is:
        x_app(t) = x_surge(t) + z * theta(t)
    where theta is the pitch angle [rad].

    The spectral approach uses the complex RAOs:
        X_app(w) = X_surge(w) + z * Theta(w)
    where Theta is converted from deg/m to rad/m.

    S_app(w) = |X_app(w)|^2 * S_wave(w)

    Parameters
    ----------
    rao_file : str, path to Nemoh RAO.tec
    hs, tp : float, sea state parameters
    z_above_swl : float, height above SWL [m] (positive upward)
    gamma : float, JONSWAP gamma

    Returns
    -------
    dict with sigma, sig_amp, x_max, Tz, and RAO breakdown
    """
    omega, surge_amp, surge_ph, pitch_amp_deg, pitch_ph = \
        parse_rao_surge_pitch(rao_file)

    # Convert pitch from deg/m to rad/m
    pitch_amp_rad = np.radians(pitch_amp_deg)

    # Complex RAOs
    X_surge = surge_amp * np.exp(1j * surge_ph)      # [m/m]
    Theta = pitch_amp_rad * np.exp(1j * pitch_ph)     # [rad/m]

    # Apparent surge RAO at height z
    X_app = X_surge + z_above_swl * Theta              # [m/m]

    S = jonswap(omega, hs, tp, gamma=gamma)
    S_app = np.abs(X_app)**2 * S
    var = trapz(S_app, omega)
    sigma = np.sqrt(var) if var > 0 else 0.0

    m0 = var
    m2 = trapz(S_app * omega**2, omega)
    if m2 > 0:
        Tz = 2 * np.pi * np.sqrt(m0 / m2)
        N = DURATION_3H / Tz
    else:
        Tz = tp
        N = 1
    x_max = sigma * np.sqrt(2 * np.log(max(N, 1)))

    # Also compute surge-only and pitch-only contributions for breakdown
    S_surge_only = surge_amp**2 * S
    S_pitch_only = (z_above_swl * pitch_amp_rad)**2 * S
    sigma_surge = np.sqrt(max(0, trapz(S_surge_only, omega)))
    sigma_pitch = np.sqrt(max(0, trapz(S_pitch_only, omega)))

    # Peak apparent RAO
    peak_idx = np.argmax(np.abs(X_app))
    peak_omega = omega[peak_idx]

    return {
        'sigma': sigma,
        'sig_amp': 2 * sigma,
        'x_max': x_max,
        'Tz': Tz,
        'N_cycles': N,
        'sigma_surge': sigma_surge,
        'sigma_pitch': sigma_pitch,
        'peak_rao': np.abs(X_app[peak_idx]),
        'peak_omega': peak_omega,
        'peak_T': 2 * np.pi / peak_omega if peak_omega > 0 else np.inf,
        'omega': omega,
        'rao_app': np.abs(X_app),
        'rao_surge': surge_amp,
        'rao_pitch_z': z_above_swl * pitch_amp_rad,
    }


# ============================================================
# 2. Wind drag on stopped turbine
# ============================================================

def compute_wind_drag_elements(U10):
    """
    Compute mean wind drag force on stopped Hywind structure.

    Integrates drag over:
      - Spar above waterline (0 to 10m): cylinder D=6.5m
      - Tower (10 to 87.6m): tapered cylinder D=6.0 to 3.87m
      - Nacelle (~90m): bluff body
      - Parked blades: depends on orientation

    Returns dict with total force and breakdown.
    """
    # Discretize height for numerical integration
    dz = 0.5  # m resolution

    # --- Spar above waterline ---
    z_spar = np.arange(dz/2, SPAR_FREEBOARD, dz)
    D_spar = np.full_like(z_spar, SPAR_DIAM_UPPER)
    Cd_spar = 1.2  # cylinder, somewhat rough
    U_spar = wind_profile(U10, z_spar)
    F_spar = trapz(0.5 * RHO_AIR * Cd_spar * D_spar * U_spar**2, z_spar)

    # --- Tower ---
    z_tower = np.arange(TOWER_BASE_Z + dz/2, TOWER_TOP_Z, dz)
    # Linear taper
    frac = (z_tower - TOWER_BASE_Z) / (TOWER_TOP_Z - TOWER_BASE_Z)
    D_tower = TOWER_DIAM_BASE + frac * (TOWER_DIAM_TOP - TOWER_DIAM_BASE)
    Cd_tower = 1.0  # smooth cylinder at high Re, some supercritical reduction
    U_tower = wind_profile(U10, z_tower)
    F_tower = trapz(0.5 * RHO_AIR * Cd_tower * D_tower * U_tower**2, z_tower)

    # --- Nacelle ---
    U_nacelle = wind_profile(U10, NACELLE_Z)
    A_nacelle = NACELLE_WIDTH * NACELLE_HEIGHT  # frontal area
    Cd_nacelle = NACELLE_CD
    F_nacelle = 0.5 * RHO_AIR * Cd_nacelle * A_nacelle * U_nacelle**2

    # --- Parked blades ---
    # Two scenarios:
    # (a) Feathered (pitched ~90): edge-on, effective width ~0.3m per blade
    # (b) Stopped/parked (arbitrary pitch): effective width ~chord*sin(angle)
    #     Assume average pitch ~45 deg for "worst case stopped"
    #     Projected width per blade = chord * sin(45) ≈ chord * 0.7
    #
    # Blade drag: treat each blade as a tapered flat plate/airfoil section
    # Integrate over span from hub to tip
    z_blade_root = HUB_HEIGHT  # hub center
    # Blades could be at any azimuth. For head-on wind, the projected
    # frontal area is what matters. Average over azimuth positions.
    #
    # Simplification: compute total projected area * mean dynamic pressure
    # For feathered blades:
    blade_eff_width_feathered = 0.3  # m (edge thickness)
    blade_eff_width_parked = BLADE_MEAN_CHORD * 0.5  # ~1.75m average projection
    Cd_blade = 1.5  # flat plate / airfoil at high angle

    # Blade spans from hub to tip; blade root at hub height, tip at hub +/- 61.5m
    # For a 3-blade rotor in arbitrary azimuth, average height distribution
    # Simplify: all blades are at approximately hub height (conservative for drag)
    U_blade = wind_profile(U10, HUB_HEIGHT)

    # Feathered case
    A_blade_feathered = N_BLADES * BLADE_LENGTH * blade_eff_width_feathered
    F_blade_feathered = 0.5 * RHO_AIR * Cd_blade * A_blade_feathered * U_blade**2

    # Parked/stopped case
    A_blade_parked = N_BLADES * BLADE_LENGTH * blade_eff_width_parked
    F_blade_parked = 0.5 * RHO_AIR * Cd_blade * A_blade_parked * U_blade**2

    return {
        'F_spar': F_spar,
        'F_tower': F_tower,
        'F_nacelle': F_nacelle,
        'F_blade_feathered': F_blade_feathered,
        'F_blade_parked': F_blade_parked,
        'F_total_feathered': F_spar + F_tower + F_nacelle + F_blade_feathered,
        'F_total_parked': F_spar + F_tower + F_nacelle + F_blade_parked,
        'U_hub': wind_profile(U10, HUB_HEIGHT),
    }


def compute_linearized_drag_sensitivity(U10):
    """
    Compute dF/du — the linearized drag sensitivity for dynamic response.

    For drag F = 0.5 * rho * Cd * A * U^2, the linearized fluctuation is:
        dF = rho * Cd * A * U * du

    Integrate this sensitivity over all structural elements.
    Returns dF/du [N/(m/s)] at a reference height (we use the integrated
    weighted sensitivity).
    """
    dz = 0.5

    # Spar above waterline
    z_spar = np.arange(dz/2, SPAR_FREEBOARD, dz)
    U_spar = wind_profile(U10, z_spar)
    dFdu_spar = trapz(RHO_AIR * 1.2 * SPAR_DIAM_UPPER * U_spar, z_spar)

    # Tower
    z_tower = np.arange(TOWER_BASE_Z + dz/2, TOWER_TOP_Z, dz)
    frac = (z_tower - TOWER_BASE_Z) / (TOWER_TOP_Z - TOWER_BASE_Z)
    D_tower = TOWER_DIAM_BASE + frac * (TOWER_DIAM_TOP - TOWER_DIAM_BASE)
    U_tower = wind_profile(U10, z_tower)
    dFdu_tower = trapz(RHO_AIR * 1.0 * D_tower * U_tower, z_tower)

    # Nacelle
    U_nac = wind_profile(U10, NACELLE_Z)
    dFdu_nacelle = RHO_AIR * NACELLE_CD * NACELLE_WIDTH * NACELLE_HEIGHT * U_nac

    # Blades (feathered)
    A_blade = N_BLADES * BLADE_LENGTH * 0.3  # feathered
    U_blade = wind_profile(U10, HUB_HEIGHT)
    dFdu_blade_feathered = RHO_AIR * 1.5 * A_blade * U_blade

    # Blades (parked)
    A_blade_p = N_BLADES * BLADE_LENGTH * BLADE_MEAN_CHORD * 0.5
    dFdu_blade_parked = RHO_AIR * 1.5 * A_blade_p * U_blade

    return {
        'dFdu_feathered': dFdu_spar + dFdu_tower + dFdu_nacelle + dFdu_blade_feathered,
        'dFdu_parked': dFdu_spar + dFdu_tower + dFdu_nacelle + dFdu_blade_parked,
    }


def compute_wind_surge(U10, blade_state='feathered'):
    """
    Compute wind-driven surge response for stopped turbine.

    Uses Frøya spectrum at an effective height, with the integrated
    drag sensitivity as the force transfer function.

    The wind force spectrum is:
        S_F(f) = (dF/du)^2 * S_u(f, z_eff) * chi^2(f)

    where chi^2(f) is a spatial coherence reduction factor (structure
    is ~90m tall, so the wind is not perfectly correlated over the height).

    Simplified coherence: chi^2(f) ≈ 1/(1 + (c*f*H/U_mean)^2)
    where H is structure height, c ≈ 10 (Davenport decay coefficient).
    """
    drag = compute_wind_drag_elements(U10)
    sens = compute_linearized_drag_sensitivity(U10)

    if blade_state == 'feathered':
        F_mean = drag['F_total_feathered']
        dFdu = sens['dFdu_feathered']
    else:
        F_mean = drag['F_total_parked']
        dFdu = sens['dFdu_parked']

    x_mean = F_mean / K_SURGE

    # Effective height for spectrum (weighted by drag distribution)
    # Approximate: use ~50m (midpoint of tower, where most drag acts)
    z_eff = 50.0
    U_eff = wind_profile(U10, z_eff)

    # Frequency array [Hz]
    f = np.logspace(-4, 0, 4000)  # 0.0001 to 1 Hz
    omega = 2 * np.pi * f

    # Frøya wind spectrum at effective height
    S_u = froya_spectrum(f, z_eff, U10)

    # Spatial coherence reduction (Davenport-like)
    # Structure height ~90m, decay coefficient c~10
    H_struct = TOWER_TOP_Z  # ~87.6m
    c_decay = 10.0
    chi2 = 1.0 / (1.0 + (c_decay * f * H_struct / U_eff)**2)

    # Force spectrum [N^2/Hz]
    S_F = dFdu**2 * S_u * chi2

    # Mechanical surge transfer function
    H_surge = np.zeros(len(omega), dtype=complex)
    for k in range(len(omega)):
        denom = -M_EFF * omega[k]**2 + K_SURGE - 1j * omega[k] * B_EXT
        if abs(denom) > 0:
            H_surge[k] = 1.0 / denom

    # Response spectrum [m^2/Hz]
    S_x = np.abs(H_surge)**2 * S_F

    # Statistics
    var = trapz(S_x, f)
    sigma = np.sqrt(var) if var > 0 else 0.0

    # Zero-crossing period
    m0 = var
    m2 = trapz(S_x * (2 * np.pi * f)**2, f)
    if m2 > 0:
        Tz = 2 * np.pi * np.sqrt(m0 / m2)
        N = DURATION_3H / Tz
    else:
        Tz = T_N
        N = 1
    x_max = sigma * np.sqrt(2 * np.log(max(N, 1)))

    # Variance breakdown
    mask_res = (f > 0.003) & (f < 0.015)  # around surge resonance
    var_res = trapz(S_x[mask_res], f[mask_res]) if mask_res.sum() > 1 else 0.0

    return {
        'U10': U10,
        'U_hub': drag['U_hub'],
        'F_mean': F_mean,
        'x_mean': x_mean,
        'dFdu': dFdu,
        'sigma': sigma,
        'sig_amp': 2 * sigma,
        'x_max': x_max,
        'Tz': Tz,
        'N_cycles': N,
        'var_total': var,
        'var_resonance': var_res,
        'pct_resonance': var_res / var * 100 if var > 0 else 0,
        'f': f,
        'S_F': S_F,
        'S_x': S_x,
        'S_u': S_u,
    }


# ============================================================
# 3. Current drag on submerged spar
# ============================================================

def compute_current_drag(U_current):
    """
    Mean current drag force on submerged spar.

    OC3 Hywind spar geometry:
      - Upper column: SWL to -4m, D=6.5m
      - Taper: -4m to -12m, D transitions 6.5m to 9.4m
      - Lower column: -12m to -120m, D=9.4m

    F = integral 0.5 * rho * Cd * D(z) * U(z)^2 dz

    Current profile: assume uniform for simplicity (conservative).
    In reality, current decreases with depth.
    """
    dz = 0.5
    F_total = 0.0

    # Upper column: 0 to -4m
    z_up = np.arange(dz/2, 4.0, dz)
    F_up = 0.5 * RHO_WATER * CD_CYL_CURRENT * SPAR_UPPER_D * U_current**2 * len(z_up) * dz

    # Taper: -4m to -12m
    z_tap = np.arange(4.0 + dz/2, 12.0, dz)
    frac = (z_tap - 4.0) / (12.0 - 4.0)
    D_tap = SPAR_UPPER_D + frac * (SPAR_LOWER_D - SPAR_UPPER_D)
    F_tap = trapz(0.5 * RHO_WATER * CD_CYL_CURRENT * D_tap * U_current**2, z_tap)

    # Lower column: -12m to -120m
    length_lower = SPAR_DRAFT - 12.0  # 108m
    F_low = 0.5 * RHO_WATER * CD_CYL_CURRENT * SPAR_LOWER_D * U_current**2 * length_lower

    F_total = F_up + F_tap + F_low
    x_mean = F_total / K_SURGE

    return {
        'U_current': U_current,
        'U_current_kt': U_current * 1.9438,
        'F_total': F_total,
        'F_upper': F_up,
        'F_taper': F_tap,
        'F_lower': F_low,
        'x_mean': x_mean,
    }


# ============================================================
# 4. Nonlinear catenary mooring model
# ============================================================

# OC3 mooring configuration (Jonkman 2010, Table 3-1):
#   3 catenary lines, 120-deg apart (azimuths 180, 60, 300 deg)
#   Line length: 902.2 m (unstretched)
#   Anchor radius: 853.87 m (from centerline), on seabed at -320m
#   Fairlead radius: 5.2 m (from centerline), at -70m below SWL
#   Line mass in water: 77.7066 kg/m
#   Line EA (extensional stiffness): 384.243e6 N (not used — inextensible)
#
# Geometry: vertical span h = 320 - 70 = 250m
#           horizontal span L_H = 853.87 - 5.2 = 848.67m (nominal)
#           straight-line distance = sqrt(848.67^2 + 250^2) = 884.7m
#           line length = 902.2m > 884.7m → catenary sag, ~2% slack
#
# Key finding: ALL chain is suspended (no seabed contact at nominal offset).
# The line goes taut at ~18m surge offset.

MOORING_N_LINES = 3
MOORING_LINE_LENGTH = 902.2       # m (unstretched)
MOORING_ANCHOR_RADIUS = 853.87    # m from center
MOORING_FAIRLEAD_RADIUS = 5.2     # m from center
MOORING_FAIRLEAD_DEPTH = 70.0     # m below SWL
MOORING_WEIGHT_PER_M = 77.7066 * G  # N/m (weight in water)
MOORING_WATER_DEPTH = 320.0       # m
MOORING_H_SPAN = MOORING_WATER_DEPTH - MOORING_FAIRLEAD_DEPTH  # 250m
MOORING_LINE_AZIMUTHS = [180.0, 60.0, 300.0]  # degrees


def solve_catenary_T_H(L_H_val):
    """
    Solve for horizontal tension T_H given horizontal span L_H.

    Uses the catenary identity for a uniform chain between two points:
        (2*a*sinh(L_H/(2*a)))^2 = L^2 - h^2
    where a = T_H / w.

    Returns T_H [N], or None if the line goes taut (L_H >= L_H_max).
    """
    w = MOORING_WEIGHT_PER_M
    L = MOORING_LINE_LENGTH
    h = MOORING_H_SPAN

    L_H_max = np.sqrt(L**2 - h**2)  # taut limit
    if L_H_val >= L_H_max - 0.01:
        return None  # line goes taut

    target = L_H_max  # = sqrt(L^2 - h^2)

    def func(a):
        x = L_H_val / (2 * a)
        if x > 50:
            return np.inf
        return 2 * a * np.sinh(x) - target

    try:
        a_sol = brentq(func, 1.0, 1e8, xtol=0.01)
        return a_sol * w
    except Exception:
        return None


def mooring_restoring_force(dx):
    """
    Total mooring restoring force in surge for platform offset dx.

    Returns force [N] (negative = restoring, opposing positive dx).
    Returns None if any line goes taut.
    """
    F_surge = 0.0

    for phi_deg in MOORING_LINE_AZIMUTHS:
        phi = np.radians(phi_deg)
        # Anchor position (fixed on seabed)
        xa = MOORING_ANCHOR_RADIUS * np.cos(phi)
        ya = MOORING_ANCHOR_RADIUS * np.sin(phi)
        # Fairlead position (moves with platform)
        xf = MOORING_FAIRLEAD_RADIUS * np.cos(phi) + dx
        yf = MOORING_FAIRLEAD_RADIUS * np.sin(phi)

        L_H = np.sqrt((xa - xf)**2 + (ya - yf)**2)
        T_H = solve_catenary_T_H(L_H)

        if T_H is None:
            return None  # line taut

        # Surge component: horizontal tension * direction cosine
        dir_x = (xa - xf) / L_H
        F_surge += T_H * dir_x

    return F_surge


def catenary_force_offset():
    """
    Compute mooring restoring force and tangent stiffness vs surge offset.

    Returns (offsets, K_tang, F_restore) arrays.
    K_tang is positive (restoring stiffness).
    F_restore is positive (magnitude of restoring force opposing offset).
    """
    offsets = np.arange(0, 18.1, 1.0)
    F_restore = np.zeros_like(offsets)
    K_tang = np.zeros_like(offsets)

    for i, dx in enumerate(offsets):
        F = mooring_restoring_force(dx)
        if F is None:
            # Truncate at taut limit
            offsets = offsets[:i]
            F_restore = F_restore[:i]
            K_tang = K_tang[:i]
            break

        F_restore[i] = -F  # positive = restoring magnitude

        # Tangent stiffness (finite difference)
        dd = 0.05
        Fp = mooring_restoring_force(dx + dd / 2)
        Fm = mooring_restoring_force(dx - dd / 2)
        if Fp is not None and Fm is not None:
            K_tang[i] = -(Fp - Fm) / dd  # positive = restoring
        else:
            K_tang[i] = K_tang[i - 1] * 2 if i > 0 else 41180  # extrapolate

    return offsets, K_tang, F_restore


def find_catenary_equilibrium(F_external):
    """
    Find equilibrium offset where mooring restoring equals external force.

    F_external [N]: positive force pushing in +x direction.
    Returns (x_eq, K_tangent_at_eq).
    """
    def residual(x):
        F_moor = mooring_restoring_force(x)
        if F_moor is None:
            return -1e12  # taut, overshoot
        return F_moor + F_external  # equilibrium when = 0

    try:
        x_eq = brentq(residual, 0, 17.5, xtol=0.01)
    except Exception:
        return None, None

    # Tangent stiffness at equilibrium
    dd = 0.05
    Fp = mooring_restoring_force(x_eq + dd / 2)
    Fm = mooring_restoring_force(x_eq - dd / 2)
    if Fp is not None and Fm is not None:
        K_tang = -(Fp - Fm) / dd
    else:
        K_tang = None

    return x_eq, K_tang


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("SURGE MOTION BUDGET — OC3 Hywind spar (alongside, turbine stopped)")
    print("=" * 80)
    print()
    print(f"  Surge natural period:  T_n = {T_N:.1f} s  (f_n = {F_N:.4f} Hz, "
          f"omega_n = {OMEGA_N:.4f} rad/s)")
    print(f"  Q-factor:              Q = {Q_FACTOR:.1f}")
    print(f"  Effective mass:        M+A = {M_EFF:.3e} kg")
    print(f"  Mooring stiffness:     K = {K_SURGE:.0f} N/m")
    print(f"  External damping:      B = {B_EXT:.0e} N·s/m")
    print(f"  Wind profile:          power law, alpha = {ALPHA_SEA}")
    print(f"  Wind spectrum:         Frøya (Andersen & Løvseth 1992)")
    print()

    # ---- Check Frøya spectrum sanity ----
    print("--- Frøya spectrum check ---")
    for z_check in [10, 50, 90]:
        sig_approx = froya_sigma_u(z_check, 10.0)
        U_z = wind_profile(10.0, z_check)
        I_u = sig_approx / U_z
        print(f"  z={z_check:3d}m: U={U_z:.1f} m/s, sigma_u={sig_approx:.2f} m/s, "
              f"I_u={I_u:.3f}")
    print()

    # ---- Operational wind speeds (U10) ----
    wind_speeds_U10 = [5.0, 8.0, 10.0, 12.0]

    # ============================================================
    # Part 1: Mean wind drag on stopped turbine
    # ============================================================
    print("-" * 80)
    print("1. MEAN WIND DRAG ON STOPPED TURBINE")
    print("-" * 80)
    print()
    print(f"   {'U10':>5s} {'U_hub':>6s} | {'F_spar':>7s} {'F_tower':>7s} "
          f"{'F_nacl':>7s} {'F_blade':>7s} | {'F_total':>8s} {'x_mean':>7s}")
    print(f"   {'[m/s]':>5s} {'[m/s]':>6s} | {'[kN]':>7s} {'[kN]':>7s} "
          f"{'[kN]':>7s} {'[kN]':>7s} | {'[kN]':>8s} {'[m]':>7s}")
    print(f"   {'-----':>5s} {'------':>6s} + {'-'*7} {'-'*7} "
          f"{'-'*7} {'-'*7} + {'-'*8} {'-'*7}")

    for U10 in wind_speeds_U10:
        d = compute_wind_drag_elements(U10)
        for state_label, F_bl, F_tot in [
            ('feath', d['F_blade_feathered'], d['F_total_feathered']),
            ('parkd', d['F_blade_parked'], d['F_total_parked']),
        ]:
            x_mean = F_tot / K_SURGE
            print(f"   {U10:5.1f} {d['U_hub']:6.1f} | {d['F_spar']/1e3:7.2f} "
                  f"{d['F_tower']/1e3:7.2f} {d['F_nacelle']/1e3:7.2f} "
                  f"{F_bl/1e3:7.2f} | {F_tot/1e3:8.2f} {x_mean:7.2f}  ({state_label})")

    print()
    print("   'feath' = blades feathered (edge-on to wind)")
    print("   'parkd' = blades parked at ~45 deg pitch (more exposed)")
    print()

    # ============================================================
    # Part 2: Dynamic wind-driven surge (Frøya spectrum)
    # ============================================================
    print("-" * 80)
    print("2. DYNAMIC WIND-DRIVEN SURGE (Frøya spectrum, stopped turbine)")
    print("-" * 80)
    print()

    for blade_state in ['feathered', 'parked']:
        print(f"   Blade state: {blade_state}")
        print(f"   {'U10':>5s} | {'F_mean':>8s} {'x_mean':>7s} | "
              f"{'sigma':>7s} {'x_sig':>7s} {'x_max':>7s} | "
              f"{'%_res':>6s} {'Tz':>7s}")
        print(f"   {'[m/s]':>5s} | {'[kN]':>8s} {'[m]':>7s} | "
              f"{'[m]':>7s} {'[m]':>7s} {'[m]':>7s} | "
              f"{'':>6s} {'[s]':>7s}")
        print(f"   {'-----':>5s} + {'-'*8} {'-'*7} + "
              f"{'-'*7} {'-'*7} {'-'*7} + "
              f"{'-'*6} {'-'*7}")

        for U10 in wind_speeds_U10:
            w = compute_wind_surge(U10, blade_state=blade_state)
            print(f"   {U10:5.1f} | {w['F_mean']/1e3:8.2f} {w['x_mean']:7.3f} | "
                  f"{w['sigma']:7.3f} {w['sig_amp']:7.3f} {w['x_max']:7.3f} | "
                  f"{w['pct_resonance']:5.1f}% {w['Tz']:7.1f}")

        print()

    # ============================================================
    # Part 3: Current drag
    # ============================================================
    print("-" * 80)
    print("3. CURRENT DRAG ON SUBMERGED SPAR")
    print("-" * 80)
    print()
    print(f"   Assumes uniform current over full draft (conservative)")
    print(f"   Cd = {CD_CYL_CURRENT} for rough cylinder at high Re")
    print()

    current_speeds = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    print(f"   {'U_curr':>7s} {'[kt]':>5s} | {'F_total':>8s} {'x_mean':>7s}")
    print(f"   {'[m/s]':>7s} {'':>5s} | {'[kN]':>8s} {'[m]':>7s}")
    print(f"   {'-'*7} {'-'*5} + {'-'*8} {'-'*7}")

    for Uc in current_speeds:
        c = compute_current_drag(Uc)
        print(f"   {Uc:7.2f} {c['U_current_kt']:5.1f} | "
              f"{c['F_total']/1e3:8.1f} {c['x_mean']:7.2f}")

    print()
    print(f"   Note: Vortex shedding f_vs = St*U/D, St~0.2")
    print(f"   At Uc=0.25 m/s, D={SPAR_UPPER_D}m: f_vs = "
          f"{0.2 * 0.25 / SPAR_UPPER_D:.4f} Hz ~ f_n = {F_N:.4f} Hz")
    print(f"   Lock-in risk at low currents — see Part 8 for detailed assessment")
    print()

    # ============================================================
    # Part 3b: Nonlinear mooring stiffness — resonance shift
    # ============================================================
    print("-" * 80)
    print("3b. NONLINEAR CATENARY MOORING — RESONANCE SHIFT WITH MEAN OFFSET")
    print("-" * 80)
    print()
    print("   Catenary moorings are HARDENING: tangent stiffness increases")
    print("   with offset. For OC3 in 320m depth, ALL chain is suspended")
    print("   (no seabed contact) — line goes taut at ~18m offset.")
    print("   A mean current/wind offset shifts surge resonance to HIGHER")
    print("   frequency, where wave difference-frequency spectra have more energy.")
    print()

    # Compute catenary force-offset curve for OC3 3-line system
    offsets, K_tang, F_moor = catenary_force_offset()

    print(f"   {'Offset':>8s} {'F_moor':>9s} {'K_tang':>9s} {'f_n':>8s} "
          f"{'T_n':>8s} {'omega_n':>8s}")
    print(f"   {'[m]':>8s} {'[kN]':>9s} {'[kN/m]':>9s} {'[Hz]':>8s} "
          f"{'[s]':>8s} {'[rad/s]':>8s}")
    print(f"   {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*8}")

    for i, x_off in enumerate(offsets):
        K = K_tang[i]
        omega_n_shifted = np.sqrt(K / M_EFF) if K > 0 else 0
        f_n_shifted = omega_n_shifted / (2 * np.pi)
        T_n_shifted = 1.0 / f_n_shifted if f_n_shifted > 0 else np.inf
        print(f"   {x_off:8.1f} {F_moor[i]/1e3:9.2f} {K/1e3:9.2f} "
              f"{f_n_shifted:8.4f} {T_n_shifted:8.1f} {omega_n_shifted:8.4f}")

    print()

    # Show what current offsets correspond to
    print("   Current → offset → resonance shift:")
    for Uc in [0.5, 0.75, 1.0, 1.5]:
        c = compute_current_drag(Uc)
        x_eq, K_eq = find_catenary_equilibrium(c['F_total'])
        if x_eq is not None and K_eq is not None and K_eq > 0:
            omega_n_eq = np.sqrt(K_eq / M_EFF)
            f_n_eq = omega_n_eq / (2 * np.pi)
            T_n_eq = 1.0 / f_n_eq if f_n_eq > 0 else np.inf
            print(f"   Uc={Uc:.2f} m/s ({Uc*1.9438:.1f} kt): "
                  f"F_drag={c['F_total']/1e3:.0f} kN → x={x_eq:.1f} m → "
                  f"K_tang={K_eq/1e3:.1f} kN/m → T_n={T_n_eq:.0f}s "
                  f"(omega_n={omega_n_eq:.3f} rad/s)")
        else:
            print(f"   Uc={Uc:.2f} m/s ({Uc*1.9438:.1f} kt): "
                  f"F_drag={c['F_total']/1e3:.0f} kN → EXCEEDS MOORING CAPACITY")

    print()
    print("   IMPLICATION: At 1.5 m/s current, the mooring stiffens and")
    print("   surge resonance moves to a higher frequency where wave")
    print("   difference-frequency spectra have significantly more energy.")
    print("   This could INCREASE slow-drift response despite the offset.")
    print()

    # ============================================================
    # Part 4: First-order wave-frequency surge
    # ============================================================
    print("-" * 80)
    print("4. FIRST-ORDER (wave-frequency) SURGE from Nemoh RAOs")
    print("-" * 80)
    print()

    rao_file_coarse = "/home/blofro/src/pdstrip_test/hywind_nemoh/Motion/RAO.tec"
    rao_file_swell = "/home/blofro/src/pdstrip_test/hywind_nemoh_swell/Motion/RAO.tec"

    sea_states = [
        (1.5, 6.0,  3.3, "Mild wind-sea"),
        (2.5, 8.0,  3.3, "Moderate wind-sea"),
        (3.0, 10.0, 3.3, "DP limit wind-sea"),
        (1.0, 15.0, 5.0, "Swell Tp=15s"),
        (1.0, 19.0, 5.0, "Swell Tp=19s"),
        (1.0, 21.0, 5.0, "Swell Tp=21s"),
    ]

    print(f"   {'Sea state':<25s} {'Hs':>4s} {'Tp':>4s} | "
          f"{'sigma':>7s} {'x_sig':>7s} {'x_max':>7s} {'Tz':>7s}")
    print(f"   {'':25s} {'[m]':>4s} {'[s]':>4s} | "
          f"{'[m]':>7s} {'[m]':>7s} {'[m]':>7s} {'[s]':>7s}")
    print(f"   {'-'*25} {'----':>4s} {'----':>4s} + "
          f"{'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for hs, tp, gamma, label in sea_states:
        if tp >= 15:
            rao_file = rao_file_swell
        else:
            rao_file = rao_file_coarse
        try:
            r = compute_first_order_surge(rao_file, hs, tp, gamma=gamma)
            print(f"   {label:<25s} {hs:4.1f} {tp:4.0f} | "
                  f"{r['sigma']:7.3f} {r['sig_amp']:7.3f} "
                  f"{r['x_max']:7.3f} {r['Tz']:7.1f}")
        except Exception as e:
            print(f"   {label:<25s} {hs:4.1f} {tp:4.0f} | "
                  f"{'ERROR':>7s} — {e}")

    print()

    # ============================================================
    # Part 5: Summary budget (with live QTF slow-drift computation)
    # ============================================================
    print("=" * 80)
    print("5. SURGE MOTION BUDGET — SUMMARY")
    print("   Alongside operations, turbine stopped")
    print("=" * 80)
    print()

    # Reference condition: U10=10 m/s, Hs=2.5m Tp=8s wind-sea,
    # swell Hs=1.0m Tp=19s, current 0.5 m/s
    U10_ref = 10.0
    Uc_ref = 0.5

    w_feath = compute_wind_surge(U10_ref, blade_state='feathered')
    w_parked = compute_wind_surge(U10_ref, blade_state='parked')
    c_ref = compute_current_drag(Uc_ref)

    try:
        r_wave_ws = compute_first_order_surge(rao_file_coarse, 2.5, 8.0, gamma=3.3)
        sigma_wave_ws = r_wave_ws['sigma']
    except Exception:
        sigma_wave_ws = 0.3  # fallback estimate

    try:
        r_wave_sw = compute_first_order_surge(rao_file_swell, 1.0, 19.0, gamma=5.0)
        sigma_wave_sw = r_wave_sw['sigma']
    except Exception:
        sigma_wave_sw = 0.5  # fallback

    # ---- Live QTF slow-drift computation ----
    qtf_file = "/home/blofro/src/pdstrip_test/hywind_nemoh_swell/results/QTF/OUT_QTFM_N.dat"
    try:
        qtf_omegas, qtf_T = parse_qtf_total(qtf_file, dof=1, beta=0.0)
        have_qtf = True
        print("   [QTF loaded for live slow-drift computation]")
    except Exception as e:
        have_qtf = False
        print(f"   [QTF not available: {e}. Using fallback values.]")

    # Compute slow-drift at LINEARIZED stiffness (baseline)
    if have_qtf:
        # Wind-sea: Hs=2.5m Tp=8s gamma=3.3
        S_ws = jonswap(qtf_omegas, 2.5, 8.0, gamma=3.3)
        mu_ws, SF_ws = compute_slow_drift_force_spectrum(qtf_omegas, qtf_T, S_ws)
        H_ws = surge_transfer_function(mu_ws, MASS, A11_LOW, B_EXT, K_SURGE)
        Sx_ws = np.abs(H_ws)**2 * SF_ws
        stats_ws = compute_statistics(mu_ws, Sx_ws)
        sigma_drift_windsea = stats_ws['sigma']

        # Swell: Hs=1m Tp=19s gamma=5
        S_sw = jonswap(qtf_omegas, 1.0, 19.0, gamma=5.0)
        mu_sw, SF_sw = compute_slow_drift_force_spectrum(qtf_omegas, qtf_T, S_sw)
        H_sw = surge_transfer_function(mu_sw, MASS, A11_LOW, B_EXT, K_SURGE)
        Sx_sw = np.abs(H_sw)**2 * SF_sw
        stats_sw = compute_statistics(mu_sw, Sx_sw)
        sigma_drift_swell = stats_sw['sigma']

        # Also compute at SHIFTED stiffness (with current offset)
        x_eq, K_eq = find_catenary_equilibrium(c_ref['F_total'])
        if x_eq is not None and K_eq is not None:
            H_ws_shifted = surge_transfer_function(mu_ws, MASS, A11_LOW, B_EXT, K_eq)
            Sx_ws_shifted = np.abs(H_ws_shifted)**2 * SF_ws
            stats_ws_shifted = compute_statistics(mu_ws, Sx_ws_shifted)
            sigma_drift_ws_shifted = stats_ws_shifted['sigma']

            H_sw_shifted = surge_transfer_function(mu_sw, MASS, A11_LOW, B_EXT, K_eq)
            Sx_sw_shifted = np.abs(H_sw_shifted)**2 * SF_sw
            stats_sw_shifted = compute_statistics(mu_sw, Sx_sw_shifted)
            sigma_drift_sw_shifted = stats_sw_shifted['sigma']
        else:
            sigma_drift_ws_shifted = sigma_drift_windsea
            sigma_drift_sw_shifted = sigma_drift_swell
    else:
        sigma_drift_windsea = 0.22   # fallback from prior analysis
        sigma_drift_swell = 0.18
        sigma_drift_ws_shifted = sigma_drift_windsea
        sigma_drift_sw_shifted = sigma_drift_swell
        K_eq = K_SURGE
        x_eq = c_ref['x_mean']

    print(f"   Reference conditions: U10={U10_ref} m/s, current={Uc_ref} m/s ({c_ref['U_current_kt']:.1f} kt)")
    print(f"                         Wind-sea Hs=2.5m Tp=8s, Swell Hs=1.0m Tp=19s")
    if x_eq is not None and K_eq is not None:
        omega_n_eq = np.sqrt(K_eq / M_EFF)
        print(f"   Current offset: x_eq = {x_eq:.1f} m, K_tang = {K_eq/1e3:.1f} kN/m, "
              f"omega_n = {omega_n_eq:.4f} rad/s (T = {2*np.pi/omega_n_eq:.0f} s)")
    print()

    print(f"   {'Component':<42s} {'Mean':>8s} {'sigma':>8s} {'x_sig':>8s}")
    print(f"   {'':42s} {'[m]':>8s} {'[m]':>8s} {'[m]':>8s}")
    print(f"   {'-'*42} {'-'*8} {'-'*8} {'-'*8}")
    print(f"   {'Wind drag, mean (feathered)':<42s} "
          f"{w_feath['x_mean']:8.3f} {'—':>8s} {'—':>8s}")
    print(f"   {'Wind drag, dynamic (feathered)':<42s} "
          f"{'—':>8s} {w_feath['sigma']:8.3f} {w_feath['sig_amp']:8.3f}")
    print(f"   {'Current drag, mean ({0} m/s)'.format(Uc_ref):<42s} "
          f"{c_ref['x_mean']:8.3f} {'—':>8s} {'—':>8s}")
    print(f"   {'1st-order wave, wind-sea (Hs=2.5 Tp=8)':<42s} "
          f"{'—':>8s} {sigma_wave_ws:8.3f} {2*sigma_wave_ws:8.3f}")
    print(f"   {'1st-order wave, swell (Hs=1 Tp=19)':<42s} "
          f"{'—':>8s} {sigma_wave_sw:8.3f} {2*sigma_wave_sw:8.3f}")
    src = 'QTF' if have_qtf else 'fallback'
    print(f"   {'2nd-order drift, wind-sea [{0}]'.format(src):<42s} "
          f"{'—':>8s} {sigma_drift_windsea:8.3f} {2*sigma_drift_windsea:8.3f}")
    print(f"   {'2nd-order drift, swell [{0}]'.format(src):<42s} "
          f"{'—':>8s} {sigma_drift_swell:8.3f} {2*sigma_drift_swell:8.3f}")
    if have_qtf and K_eq != K_SURGE:
        print()
        print(f"   --- With nonlinear mooring (K={K_eq/1e3:.1f} kN/m at Uc={Uc_ref} m/s) ---")
        print(f"   {'2nd-order drift, wind-sea (shifted K)':<42s} "
              f"{'—':>8s} {sigma_drift_ws_shifted:8.3f} {2*sigma_drift_ws_shifted:8.3f}")
        print(f"   {'2nd-order drift, swell (shifted K)':<42s} "
              f"{'—':>8s} {sigma_drift_sw_shifted:8.3f} {2*sigma_drift_sw_shifted:8.3f}")
        ratio_ws = sigma_drift_ws_shifted / sigma_drift_windsea if sigma_drift_windsea > 0 else 0
        ratio_sw = sigma_drift_sw_shifted / sigma_drift_swell if sigma_drift_swell > 0 else 0
        print(f"   {'  (ratio vs linearized K)':<42s} "
              f"{'':>8s} {'x{:.2f}'.format(ratio_ws):>8s} {'x{:.2f}'.format(ratio_sw):>8s}")

    print()

    # Combined (assume independent, RSS for dynamic components)
    # Use linearized slow-drift (conservative: nonlinear mooring reduces it)
    for blade_label, w in [('feathered', w_feath), ('parked', w_parked)]:
        x_mean_total = w['x_mean'] + c_ref['x_mean']
        sigma_total = np.sqrt(
            w['sigma']**2 +
            sigma_wave_ws**2 +
            sigma_wave_sw**2 +
            sigma_drift_windsea**2 +
            sigma_drift_swell**2
        )
        x_95 = x_mean_total + 2 * sigma_total

        print(f"   COMBINED ({blade_label} blades):")
        print(f"     Mean offset:         {x_mean_total:8.3f} m "
              f"(wind {w['x_mean']:.3f} + current {c_ref['x_mean']:.3f})")
        print(f"     Dynamic sigma (RSS): {sigma_total:8.3f} m")
        print(f"     x_sig (2*sigma):     {2*sigma_total:8.3f} m")
        print(f"     ~95th pctl excursion: {x_95:8.2f} m (mean + 2*sigma)")
        print()

    # ============================================================
    # Part 6: Sensitivity to wind speed
    # ============================================================
    print("-" * 80)
    print("6. WIND SPEED SENSITIVITY (feathered blades)")
    print("-" * 80)
    print()
    print(f"   Sea state: Hs=2.5m Tp=8s, swell Hs=1m Tp=19s, current 0.5 m/s")
    print()
    print(f"   {'U10':>5s} | {'mean_wind':>9s} {'mean_curr':>9s} {'mean_tot':>8s} | "
          f"{'sig_wind':>8s} {'sig_wv':>8s} {'sig_drft':>8s} {'sig_tot':>8s} | "
          f"{'x_95th':>8s}")
    print(f"   {'[m/s]':>5s} | {'[m]':>9s} {'[m]':>9s} {'[m]':>8s} | "
          f"{'[m]':>8s} {'[m]':>8s} {'[m]':>8s} {'[m]':>8s} | "
          f"{'[m]':>8s}")
    print(f"   {'-----':>5s} + {'-'*9} {'-'*9} {'-'*8} + "
          f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} + "
          f"{'-'*8}")

    sigma_drift_comb = np.sqrt(sigma_drift_windsea**2 + sigma_drift_swell**2)
    sigma_wave_comb = np.sqrt(sigma_wave_ws**2 + sigma_wave_sw**2)

    for U10 in wind_speeds_U10:
        w = compute_wind_surge(U10, blade_state='feathered')
        x_mean_wind = w['x_mean']
        x_mean_curr = c_ref['x_mean']
        x_mean_tot = x_mean_wind + x_mean_curr

        sigma_tot = np.sqrt(w['sigma']**2 + sigma_wave_comb**2 + sigma_drift_comb**2)
        x_95 = x_mean_tot + 2 * sigma_tot

        print(f"   {U10:5.1f} | {x_mean_wind:9.3f} {x_mean_curr:9.3f} "
              f"{x_mean_tot:8.3f} | "
              f"{w['sigma']:8.3f} {sigma_wave_comb:8.3f} {sigma_drift_comb:8.3f} "
              f"{sigma_tot:8.3f} | {x_95:8.2f}")

    print()
    print("   x_95th = mean_tot + 2 * sigma_tot (approx 95th percentile peak excursion)")
    print()

    # ============================================================
    # Part 7: Nonlinear mooring effect on slow-drift
    # ============================================================
    if have_qtf:
        print("-" * 80)
        print("7. NONLINEAR MOORING EFFECT ON SLOW-DRIFT (current sweep)")
        print("-" * 80)
        print()
        print("   Current-induced offset stiffens the catenary mooring and")
        print("   shifts surge resonance. Two competing effects:")
        print("   (+) Resonance moves to higher mu where QTF force spectrum has more energy")
        print("   (-) Peak |H(omega_n)|^2 decreases as 1/(B*omega_n)^2")
        print()

        sweep_currents = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        print(f"   {'Uc':>6s} {'x_eq':>6s} {'K_tang':>9s} {'omega_n':>8s} | "
              f"{'sig_ws':>8s} {'sig_sw':>8s} {'ratio_ws':>8s} {'ratio_sw':>8s}")
        print(f"   {'[m/s]':>6s} {'[m]':>6s} {'[kN/m]':>9s} {'[rad/s]':>8s} | "
              f"{'[m]':>8s} {'[m]':>8s} {'':>8s} {'':>8s}")
        print(f"   {'-'*6} {'-'*6} {'-'*9} {'-'*8} + "
              f"{'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for Uc in sweep_currents:
            if Uc == 0:
                x_eq_s, K_eq_s = 0.0, K_SURGE
            else:
                F_drag = compute_current_drag(Uc)['F_total']
                x_eq_s, K_eq_s = find_catenary_equilibrium(F_drag)
                if x_eq_s is None or K_eq_s is None:
                    print(f"   {Uc:6.2f}  EXCEEDS MOORING CAPACITY")
                    continue

            omega_n_s = np.sqrt(K_eq_s / M_EFF)

            # Slow-drift at this K
            H_ws_s = surge_transfer_function(mu_ws, MASS, A11_LOW, B_EXT, K_eq_s)
            sig_ws_s = np.sqrt(max(0, trapz(np.abs(H_ws_s[1:])**2 * SF_ws[1:], mu_ws[1:])))
            H_sw_s = surge_transfer_function(mu_sw, MASS, A11_LOW, B_EXT, K_eq_s)
            sig_sw_s = np.sqrt(max(0, trapz(np.abs(H_sw_s[1:])**2 * SF_sw[1:], mu_sw[1:])))

            r_ws = sig_ws_s / sigma_drift_windsea if sigma_drift_windsea > 0 else 0
            r_sw = sig_sw_s / sigma_drift_swell if sigma_drift_swell > 0 else 0

            print(f"   {Uc:6.2f} {x_eq_s:6.1f} {K_eq_s/1e3:9.2f} {omega_n_s:8.4f} | "
                  f"{sig_ws_s:8.4f} {sig_sw_s:8.4f} {r_ws:8.2f} {r_sw:8.2f}")

        print()
        print("   RESULT: Nonlinear mooring stiffening consistently REDUCES slow-drift.")
        print("   The damping effect (|H| ~ 1/(B*omega_n) drops) dominates the")
        print("   increased QTF force energy at higher difference frequencies.")
        print("   Using linearized K is CONSERVATIVE for the slow-drift budget.")
        print()

    # ============================================================
    # Part 8: Vortex-induced vibration (VIV) assessment
    # ============================================================
    print("-" * 80)
    print("8. VORTEX-INDUCED VIBRATION (VIV) ASSESSMENT")
    print("-" * 80)
    print()

    # Strouhal number for circular cylinder (post-critical Re regime)
    St = 0.20  # conservative (range 0.2-0.27 at post-critical Re)
    NU_WATER = 1.2e-6  # kinematic viscosity [m^2/s]
    CL_POST_CRIT = 0.3  # oscillating lift coefficient (post-critical Re)
    CORR_LEN_NO_LOCK = 3  # correlation lengths [diameters] without lock-in
    CORR_LEN_LOCK = 15    # correlation lengths [diameters] with lock-in

    viv_currents = [0.15, 0.20, 0.25, 0.30, 0.50, 0.75, 1.00, 1.50]

    print(f"   Strouhal number: St = {St}")
    print(f"   Surge/sway natural freq: f_n = {F_N:.4f} Hz (T = {T_N:.0f} s)")
    print(f"   Post-critical C_L: {CL_POST_CRIT}")
    print()

    # VIV frequency table
    print(f"   {'Uc':>6s} | {'Re_up':>9s} {'Re_low':>9s} | "
          f"{'f_vs,up':>8s} {'f_vs,low':>8s} | "
          f"{'Vr,up':>6s} {'Vr,low':>6s} | "
          f"{'f_vs/f_n':>8s} {'regime':>12s}")
    print(f"   {'[m/s]':>6s} | {'D=6.5':>9s} {'D=9.4':>9s} | "
          f"{'[Hz]':>8s} {'[Hz]':>8s} | "
          f"{'':>6s} {'':>6s} | "
          f"{'upper':>8s} {'':>12s}")
    print(f"   {'-'*6} + {'-'*9} {'-'*9} + "
          f"{'-'*8} {'-'*8} + "
          f"{'-'*6} {'-'*6} + "
          f"{'-'*8} {'-'*12}")

    for Uc in viv_currents:
        Re_up = Uc * SPAR_UPPER_D / NU_WATER
        Re_low = Uc * SPAR_LOWER_D / NU_WATER
        f_vs_up = St * Uc / SPAR_UPPER_D
        f_vs_low = St * Uc / SPAR_LOWER_D
        Vr_up = Uc / (F_N * SPAR_UPPER_D) if F_N > 0 else 0
        Vr_low = Uc / (F_N * SPAR_LOWER_D) if F_N > 0 else 0
        ratio_up = f_vs_up / F_N if F_N > 0 else 0

        # Determine regime
        if 4 <= Vr_up <= 12:
            regime = "LOCK-IN!"
        elif 1.5 <= Vr_up < 4:
            regime = "in-line VIV"
        elif Vr_up < 1.5:
            regime = "sub-crit"
        else:
            regime = "above lock"

        print(f"   {Uc:6.2f} | {Re_up:9.2e} {Re_low:9.2e} | "
              f"{f_vs_up:8.4f} {f_vs_low:8.4f} | "
              f"{Vr_up:6.1f} {Vr_low:6.1f} | "
              f"{ratio_up:8.2f} {regime:>12s}")

    print()
    print("   Lock-in range: V_r = 4-12 (cross-flow VIV)")
    print("   In-line VIV:   V_r = 1.5-4 (at 2x f_vs, smaller amplitude)")
    print()

    # Cross-flow VIV force and response estimate at critical current speeds
    print("   --- VIV sway response estimate at critical current speeds ---")
    print()
    print(f"   {'Uc':>6s} | {'f_vs':>8s} {'Vr':>6s} | "
          f"{'F_lift':>10s} {'F_lock':>10s} | "
          f"{'A_sway':>8s} {'A/D':>6s} | "
          f"{'Cd_amp':>6s}")
    print(f"   {'[m/s]':>6s} | {'[Hz]':>8s} {'':>6s} | "
          f"{'[kN]':>10s} {'[kN]':>10s} | "
          f"{'[m]':>8s} {'':>6s} | "
          f"{'':>6s}")
    print(f"   {'-'*6} + {'-'*8} {'-'*6} + "
          f"{'-'*10} {'-'*10} + "
          f"{'-'*8} {'-'*6} + "
          f"{'-'*6}")

    for Uc in [0.15, 0.20, 0.25, 0.30, 0.50, 0.75]:
        # Use upper column D=6.5 for shedding freq, but force acts on full spar
        f_vs = St * Uc / SPAR_UPPER_D
        Vr = Uc / (F_N * SPAR_UPPER_D) if F_N > 0 else 0

        # Lift force per unit length on each section
        f_L_upper = 0.5 * RHO_WATER * Uc**2 * SPAR_UPPER_D * CL_POST_CRIT  # [N/m]
        f_L_lower = 0.5 * RHO_WATER * Uc**2 * SPAR_LOWER_D * CL_POST_CRIT

        # Without lock-in: incoherent over spar, sqrt(N_corr) reduction
        L_upper = 4.0  # m (SWL to -4m)
        L_taper = 8.0  # m (-4 to -12m)
        L_lower = 108.0  # m (-12 to -120m)

        # Incoherent shedding: correlation length ~ 3D
        L_corr_up = CORR_LEN_NO_LOCK * SPAR_UPPER_D  # 19.5m
        L_corr_low = CORR_LEN_NO_LOCK * SPAR_LOWER_D  # 28.2m
        # Number of independent cells
        N_cells_low = max(1, L_lower / L_corr_low)
        # RMS force from incoherent cells: F_rms ~ f_L * L_corr * sqrt(N_cells)
        F_no_lock = (f_L_upper * L_upper
                     + f_L_lower * L_corr_low * np.sqrt(N_cells_low)
                     + f_L_lower * L_taper * 0.5)  # rough

        # With lock-in: coherence over ~15D
        L_corr_lock = CORR_LEN_LOCK * SPAR_UPPER_D  # 97.5m
        # Lock-in extends coherence — most of spar could be correlated
        F_lock = (f_L_upper * L_upper
                  + f_L_lower * min(L_lower, L_corr_lock)
                  + f_L_lower * L_taper * 0.5)

        # Resonant sway response: A = F * |H(omega_n)| = F / (B_ext * omega_n)
        H_res = 1.0 / (B_EXT * OMEGA_N)  # [m/N]
        A_no_lock = F_no_lock * H_res
        A_lock = F_lock * H_res

        # Effective A/D (use upper column diameter as reference)
        A_over_D = A_lock / SPAR_UPPER_D

        # Drag amplification: Cd_amp = Cd * (1 + 2*A/D)  (DNV-RP-C205 Sec 9.2.6)
        Cd_amp = CD_CYL_CURRENT * (1 + 2 * min(A_over_D, 1.0))

        print(f"   {Uc:6.2f} | {f_vs:8.4f} {Vr:6.1f} | "
              f"{F_no_lock/1e3:10.2f} {F_lock/1e3:10.2f} | "
              f"{A_lock:8.3f} {A_over_D:6.3f} | "
              f"{Cd_amp:6.2f}")

    print()
    print("   F_lift: sway VIV force, incoherent shedding (no lock-in)")
    print("   F_lock: sway VIV force with lock-in (coherence ~ 15D)")
    print("   A_sway: resonant sway amplitude with lock-in")
    print("   Cd_amp: amplified drag coefficient (DNV-RP-C205 Sec 9.2.6)")
    print()

    # --- Intermittent shedding in combined wave + current ---
    print("   --- Intermittent shedding in combined wave + current ---")
    print()
    print("   In combined flow, the relative velocity varies cyclically:")
    print("     U_rel(t) = U_current + u_wave(t)")
    print("   The instantaneous shedding frequency sweeps through a range,")
    print("   intermittently passing through the surge/sway natural frequency.")
    print()

    for Uc in [0.25, 0.50, 0.75]:
        # Approximate wave orbital velocity amplitude at mid-draft
        # For swell Tp=19s, Hs=1m: u_a ~ omega*Hs/2 * exp(-k*z)
        # At surface: u_a = pi*Hs/Tp = 0.165 m/s (deep water)
        # At z=-60m (mid-draft): u_a ~ 0.165 * exp(-k*60)
        # Deep water: k = omega^2/g = (2pi/19)^2/9.81 = 0.011 1/m
        # u_a(60) = 0.165 * exp(-0.011*60) = 0.165 * 0.52 = 0.086 m/s
        # But for current drag, the near-surface velocities dominate.
        # Use surface-average over upper 4m: u_a ~ 0.15 m/s
        u_a = 0.15  # m/s (swell, surface estimate)
        u_a_ws = 0.8  # m/s (wind-sea Hs=2.5 Tp=8)

        f_vs_min = St * max(0, Uc - u_a_ws) / SPAR_UPPER_D
        f_vs_max = St * (Uc + u_a_ws) / SPAR_UPPER_D
        f_vs_min_sw = St * max(0, Uc - u_a) / SPAR_UPPER_D
        f_vs_max_sw = St * (Uc + u_a) / SPAR_UPPER_D

        print(f"   Uc={Uc:.2f} m/s:")
        print(f"     Swell (u_a~{u_a:.2f}): f_vs range "
              f"[{f_vs_min_sw:.4f}, {f_vs_max_sw:.4f}] Hz  "
              f"{'INCLUDES f_n' if f_vs_min_sw <= F_N <= f_vs_max_sw else 'misses f_n'}")
        print(f"     Wind-sea (u_a~{u_a_ws:.1f}): f_vs range "
              f"[{f_vs_min:.4f}, {f_vs_max:.4f}] Hz  "
              f"{'INCLUDES f_n' if f_vs_min <= F_N <= f_vs_max else 'misses f_n'}")

    print()
    print("   IMPLICATIONS FOR SURGE BUDGET:")
    print()
    print("   a) SWAY VIV at low currents (0.15-0.50 m/s) can produce sway")
    print("      amplitudes of 0.5-1.5 m from lock-in with cross-flow shedding.")
    print("      This does not directly enter the SURGE budget but affects:")
    print("      - Cd amplification (10-40% increase in mean drag)")
    print("      - Coupled sway-roll dynamics")
    print("      - Fatigue of mooring lines")
    print()
    print("   b) DRAG AMPLIFICATION from VIV oscillation increases Cd by up to")
    print("      40%, increasing mean surge offset by the same factor.")
    print("      At Uc=0.5 m/s: x_mean could increase from 3.6m to ~5.0m.")
    print()
    print("   c) INTERMITTENT SHEDDING in waves+current creates broadband")
    print("      excitation that can include f_n. This adds to slow-drift-like")
    print("      forces but is hard to quantify without CFD or model tests.")
    print("      Post-critical Re and low KC limit the coherence and force")
    print("      magnitude, but the effect is non-negligible at low currents.")
    print()
    print("   d) The original estimate of Cd=1.05 may be UNCONSERVATIVE.")
    print("      Recommend Cd=1.2-1.5 to account for VIV drag amplification,")
    print("      surface roughness (marine growth), and appurtenances.")
    print()

    # ============================================================
    # Part 9: Apparent surge at elevated z (platform deck / hub)
    # ============================================================
    print("-" * 80)
    print("9. APPARENT SURGE AT ELEVATED z — PLATFORM DECK & HUB HEIGHT")
    print("-" * 80)
    print()
    print("   At height z above SWL, the apparent horizontal displacement is:")
    print("     x_app(t) = x_surge(t) + z * theta(t)")
    print("   where theta = pitch angle [rad].")
    print()
    print("   Surge-pitch coupling (COG at -78m, pitch resonance at T~23s)")
    print("   means the pitch contribution can DOMINATE at elevated positions.")
    print("   This explains reported 10-15m oscillatory motions at Hywind Tampen")
    print("   during DP campaigns (observed at platform deck level).")
    print()

    z_levels = [0.0, 10.0, 15.0, 90.0, 110.0]
    z_labels = {0.0: 'SWL', 10.0: 'platform top', 15.0: 'deck (est.)',
                90.0: 'OC3 hub', 110.0: 'SG 8MW hub (est.)'}

    swell_cases_z = [
        (1.0, 17.0, 5.0, "Sw 1.0/17"),
        (1.0, 19.0, 5.0, "Sw 1.0/19"),
        (1.0, 21.0, 5.0, "Sw 1.0/21"),
        (1.5, 17.0, 5.0, "Sw 1.5/17"),
        (1.5, 19.0, 5.0, "Sw 1.5/19"),
        (1.5, 21.0, 5.0, "Sw 1.5/21"),
        (2.0, 19.0, 5.0, "Sw 2.0/19"),
        (2.0, 21.0, 5.0, "Sw 2.0/21"),
    ]

    # 9a: RAO breakdown at reference height z=15m
    print("   9a. Apparent surge RAO at z = 15 m (estimated deck level):")
    print()
    z_ref = 15.0
    try:
        omega_r, surge_amp_r, surge_ph_r, pitch_amp_deg_r, pitch_ph_r = \
            parse_rao_surge_pitch(rao_file_swell)
        pitch_amp_rad_r = np.radians(pitch_amp_deg_r)
        X_surge_r = surge_amp_r * np.exp(1j * surge_ph_r)
        Theta_r = pitch_amp_rad_r * np.exp(1j * pitch_ph_r)
        X_app_r = X_surge_r + z_ref * Theta_r

        print(f"   {'omega':>8s} {'T':>6s} | {'|surge|':>8s} {'|z*pitch|':>9s} "
              f"{'|app|':>8s} | {'note':>20s}")
        print(f"   {'[rad/s]':>8s} {'[s]':>6s} | {'[m/m]':>8s} {'[m/m]':>9s} "
              f"{'[m/m]':>8s} | {'':>20s}")
        print(f"   {'-'*8} {'-'*6} + {'-'*8} {'-'*9} {'-'*8} + {'-'*20}")

        for k, w in enumerate(omega_r):
            T_w = 2 * np.pi / w if w > 0 else np.inf
            s_a = surge_amp_r[k]
            p_a = z_ref * pitch_amp_rad_r[k]
            a_a = np.abs(X_app_r[k])
            note = ''
            if abs(w - 0.280) < 0.005:
                note = '<- pitch resonance'
            elif abs(w - 0.040) < 0.005:
                note = '<- surge resonance'
            elif abs(w - 0.160) < 0.005:
                note = '<- heave resonance'
            if T_w >= 12 and T_w <= 200:  # show swell-relevant range
                print(f"   {w:8.3f} {T_w:6.1f} | {s_a:8.3f} {p_a:9.3f} "
                      f"{a_a:8.3f} | {note:>20s}")
        print()
    except Exception as e:
        print(f"   [Could not parse RAOs: {e}]")
        print()

    # 9b: Response statistics at multiple z levels for various swell conditions
    print("   9b. Apparent surge statistics vs height and sea state:")
    print()
    print(f"   {'Sea state':<12s} |", end='')
    for z in z_levels:
        lbl = z_labels.get(z, f'z={z}m')
        print(f" {'z=' + str(int(z)) + 'm':>9s}", end='')
    print(f" |  {'dominant':>8s}")
    print(f"   {'':12s} |", end='')
    for z in z_levels:
        print(f" {'sig [m]':>9s}", end='')
    print(f" |  {'Tz [s]':>8s}")
    print(f"   {'-'*12} +", end='')
    for _ in z_levels:
        print(f" {'-'*9}", end='')
    print(f" +  {'-'*8}")

    for hs, tp, gamma, label in swell_cases_z:
        row_data = []
        dom_Tz = 0
        for z in z_levels:
            try:
                r = compute_apparent_surge_at_z(rao_file_swell, hs, tp, z, gamma=gamma)
                row_data.append(r['sigma'])
                if z == 15.0:
                    dom_Tz = r['Tz']
            except Exception:
                row_data.append(np.nan)

        print(f"   {label:<12s} |", end='')
        for val in row_data:
            print(f" {val:9.3f}", end='')
        print(f" |  {dom_Tz:8.1f}")

    print()

    # 9c: Extreme values (x_max in 3 hours) at z=15m
    print("   9c. Expected maximum apparent surge at z = 15 m (3-hour exposure):")
    print()
    print(f"   {'Sea state':<12s} | {'sigma':>7s} {'x_sig':>7s} {'x_max':>7s} | "
          f"{'sig_surge':>9s} {'sig_pitch':>9s} {'ratio':>6s} | {'Tz':>6s}")
    print(f"   {'':12s} | {'[m]':>7s} {'[m]':>7s} {'[m]':>7s} | "
          f"{'[m]':>9s} {'[m]':>9s} {'p/s':>6s} | {'[s]':>6s}")
    print(f"   {'-'*12} + {'-'*7} {'-'*7} {'-'*7} + "
          f"{'-'*9} {'-'*9} {'-'*6} + {'-'*6}")

    for hs, tp, gamma, label in swell_cases_z:
        try:
            r = compute_apparent_surge_at_z(rao_file_swell, hs, tp, 15.0, gamma=gamma)
            ratio = r['sigma_pitch'] / r['sigma_surge'] if r['sigma_surge'] > 0 else 0
            print(f"   {label:<12s} | {r['sigma']:7.3f} {r['sig_amp']:7.3f} "
                  f"{r['x_max']:7.2f} | "
                  f"{r['sigma_surge']:9.3f} {r['sigma_pitch']:9.3f} {ratio:6.2f} | "
                  f"{r['Tz']:6.1f}")
        except Exception as e:
            print(f"   {label:<12s} | ERROR: {e}")

    print()

    # 9d: Bimodal seas (wind-sea + swell, independent)
    print("   9d. Bimodal seas at z = 15 m (wind-sea + swell, RSS combination):")
    print()
    bimodal_cases = [
        ((2.0, 8.0, 3.3), (1.0, 19.0, 5.0), "WS 2.0/8 + Sw 1.0/19"),
        ((2.0, 8.0, 3.3), (1.5, 19.0, 5.0), "WS 2.0/8 + Sw 1.5/19"),
        ((2.0, 8.0, 3.3), (2.0, 19.0, 5.0), "WS 2.0/8 + Sw 2.0/19"),
        ((3.0, 9.0, 3.3), (1.0, 19.0, 5.0), "WS 3.0/9 + Sw 1.0/19"),
        ((3.0, 9.0, 3.3), (1.5, 19.0, 5.0), "WS 3.0/9 + Sw 1.5/19"),
        ((2.0, 8.0, 3.3), (1.5, 21.0, 5.0), "WS 2.0/8 + Sw 1.5/21"),
        ((2.0, 8.0, 3.3), (2.0, 21.0, 5.0), "WS 2.0/8 + Sw 2.0/21"),
    ]

    print(f"   {'Sea state':<27s} | {'sig_ws':>7s} {'sig_sw':>7s} {'sig_tot':>7s} "
          f"{'x_sig':>7s} {'x_max':>7s}")
    print(f"   {'':27s} | {'[m]':>7s} {'[m]':>7s} {'[m]':>7s} "
          f"{'[m]':>7s} {'[m]':>7s}")
    print(f"   {'-'*27} + {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for ws_params, sw_params, label in bimodal_cases:
        try:
            r_ws = compute_apparent_surge_at_z(
                rao_file_coarse, ws_params[0], ws_params[1], 15.0, gamma=ws_params[2])
            r_sw = compute_apparent_surge_at_z(
                rao_file_swell, sw_params[0], sw_params[1], 15.0, gamma=sw_params[2])
            sig_tot = np.sqrt(r_ws['sigma']**2 + r_sw['sigma']**2)
            # Max: use shorter Tz for more conservative cycle count
            Tz_eff = min(r_ws['Tz'], r_sw['Tz'])
            N_eff = DURATION_3H / Tz_eff
            x_max = sig_tot * np.sqrt(2 * np.log(max(N_eff, 1)))
            print(f"   {label:<27s} | {r_ws['sigma']:7.3f} {r_sw['sigma']:7.3f} "
                  f"{sig_tot:7.3f} {2*sig_tot:7.3f} {x_max:7.2f}")
        except Exception as e:
            print(f"   {label:<27s} | ERROR: {e}")

    print()

    # 9e: Hub-height motions (relevant for blade clearance / yaw bearing loads)
    print("   9e. Apparent surge at hub height (z = 90 m OC3, z = 110 m SG 8MW):")
    print()
    print(f"   {'Sea state':<12s} | {'z=90m':>9s} {'z=110m':>9s} | "
          f"{'x_max 90':>9s} {'x_max 110':>9s}")
    print(f"   {'':12s} | {'sig [m]':>9s} {'sig [m]':>9s} | "
          f"{'[m]':>9s} {'[m]':>9s}")
    print(f"   {'-'*12} + {'-'*9} {'-'*9} + {'-'*9} {'-'*9}")

    for hs, tp, gamma, label in swell_cases_z:
        try:
            r90 = compute_apparent_surge_at_z(rao_file_swell, hs, tp, 90.0, gamma=gamma)
            r110 = compute_apparent_surge_at_z(rao_file_swell, hs, tp, 110.0, gamma=gamma)
            print(f"   {label:<12s} | {r90['sigma']:9.3f} {r110['sigma']:9.3f} | "
                  f"{r90['x_max']:9.2f} {r110['x_max']:9.2f}")
        except Exception as e:
            print(f"   {label:<12s} | ERROR: {e}")

    print()
    print("   KEY FINDINGS:")
    print("   - At z=15m (deck): sigma = 2-4 m in swell Hs=1.5-2.0m, Tp=17-21s")
    print("     -> x_max(3h) = 8-15 m, matching reported 10-15 m at Hywind Tampen")
    print("   - Pitch contribution exceeds surge contribution at z >= 10m")
    print("   - Dominant period ~20s (wave/pitch frequency, NOT slow-drift)")
    print("   - At hub (z=90-110m): sigma = 5-15 m (!), x_max up to 50+ m")
    print("     -> massive horizontal excursions, relevant for blade clearance")
    print("   - The oscillation period is 15-25s: directly from first-order waves")
    print("     If reported motions have period >100s, different mechanism at play")
    print()

    # ============================================================
    # Conclusions
    # ============================================================
    print("=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print("""
   For a stopped turbine during alongside operations (U10 <= 12 m/s):

   1. MEAN OFFSETS:
      - Wind drag on structure: ~1 m at U10=10 m/s (feathered)
      - Current drag: dominant — 3.6 m at 0.5 m/s, 10 m at 1.0 m/s
      - Current drag overwhelmingly dominates the mean offset budget
      - VIV-amplified Cd (1.2-1.5) could increase current offsets by 15-40%

   2. FIRST-ORDER WAVE SURGE (from coupled Nemoh RAOs):
      - Swell (Hs=1m Tp=19s): sigma ~ 1.3 m — the LARGEST dynamic component
      - Wind-sea (Hs=2.5m Tp=8s): sigma ~ 0.4-0.6 m
      - The large swell response is driven by surge-pitch coupling
        (COG at -78m, pitch resonance at T~23s near swell peak)
      - Nearly insensitive to mooring stiffness changes (<3% for moderate current)

   3. SECOND-ORDER SLOW-DRIFT (from QTF, live computation):
      - Wind-sea Hs=2.5m Tp=8s: sigma ~ 0.04 m (linearized K)
      - Swell Hs=1m Tp=19s: sigma ~ 0.09 m (linearized K)
      - Much smaller than first-order wave surge
      - 97% of variance concentrated at single QTF difference frequency
        (mu=0.04 rad/s, nearest to omega_n=0.043)

   4. DYNAMIC WIND SURGE (Froya spectrum):
      - sigma ~ 0.26 m at U10=10 m/s (feathered)
      - Significant but smaller than first-order wave surge

   5. NONLINEAR MOORING EFFECT:
      - Current-induced offset STIFFENS the catenary mooring
      - This REDUCES slow-drift by de-tuning the surge resonance
      - Damping effect 1/(B*omega_n)^2 dominates the increased QTF energy
      - Using linearized K is conservative for slow-drift
      - First-order response is insensitive to K changes

   6. VORTEX-INDUCED VIBRATION:
      - At low currents (0.15-0.50 m/s), shedding frequency on upper column
        matches the surge/sway natural frequency (V_r = 3-11, lock-in range)
      - Cross-flow (sway) VIV could produce sigma ~ 0.5-1.5 m
      - Drag amplification from VIV: Cd increases 10-40%
      - Intermittent shedding in wave+current broadens excitation spectrum
      - Post-critical Re limits coherence, but the effect is not negligible
      - Recommend Cd = 1.2-1.5 (vs baseline 1.05) for current drag

   7. DOMINANT DYNAMIC COMPONENT:
      - First-order wave surge (coupled surge-pitch) is by far the
        largest dynamic contribution, especially for long-period swell
      - This is a direct wave-frequency response, not a resonance effect
      - The DP vessel must track this ~20s period platform motion

   8. PRACTICAL IMPLICATION FOR DP ALONGSIDE:
       - The platform moves ~1-2 m (sigma) at wave frequencies in swell
       - The DP vessel itself also responds to the same swell
       - Relative motion between DP vessel and platform is what matters
       - A DP vessel with good low-frequency surge response will naturally
         follow some of the platform motion, reducing the challenge

   9. APPARENT SURGE AT ELEVATED POSITIONS (Part 9):
       - At platform deck (z~15m above SWL): surge-pitch coupling adds
         z * theta to the SWL surge, nearly doubling the apparent motion
       - sigma = 2-4 m at z=15m in swell Hs=1.5-2.0m, Tp=17-21s
       - Expected maximum in 3 hours: 8-15 m — explains reported 10-15 m
         oscillatory motions at Hywind Tampen during DP campaigns
       - Period is 15-25s (wave frequency), not slow-drift (>100s)
       - At hub height (z=90-110m): massive excursions, sigma up to 15 m
       - THIS IS THE DOMINANT MECHANISM for observed large motions
 """)


if __name__ == '__main__':
    main()
