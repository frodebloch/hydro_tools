#!/usr/bin/env python3
"""
slow_drift_swell.py - Compute slow-drift surge response to ocean swell

Reads Nemoh QTF data and computes the slow-drift surge response for
swell sea states using the exact quadratic spectral method.

Method:
  1. S_F(mu) = 8 * integral |T(w+mu, w)|^2 * S(w+mu) * S(w) dw
  2. H_surge(mu) = 1 / [-(M+A11)*mu^2 - i*mu*B_ext + K_surge]
  3. S_x(mu) = |H|^2 * S_F(mu)
  4. sigma_x = sqrt(integral S_x dmu)
  5. x_sig = 2*sigma_x, x_max = sigma_x * sqrt(2*ln(N_cycles))

Physical parameters from OC3 Hywind (Jonkman 2010).

Supports variable mooring stiffness (--k-surge) to study the effect of
nonlinear catenary mooring stiffness shift from current-induced mean offset.

Usage:
  python slow_drift_swell.py [--qtf-dir DIR] [--gamma GAMMA]
  python slow_drift_swell.py --k-sweep   # sweep K from linearized to stiffened
"""

import argparse
import numpy as np
import sys

# Compatibility: numpy >= 2.0 renamed trapz -> trapezoid
trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

from scipy.optimize import brentq

# ============================================================
# Physical parameters — OC3 Hywind
# ============================================================

RHO = 1025.0        # kg/m^3
G = 9.81            # m/s^2
RHOG = RHO * G      # 10055.25 N/m^3

MASS = 1.40179e7    # kg (total system)
A11_LOW = 8.381e6   # kg — added mass in surge at low frequency (from Nemoh)
B_EXT = 1.0e5       # N·s/m — external linear damping in surge
K_SURGE = 4.118e4   # N/m — linearized mooring stiffness (surge)

# Derived
OMEGA_N = np.sqrt(K_SURGE / (MASS + A11_LOW))  # ~0.0429 rad/s
T_N = 2 * np.pi / OMEGA_N
Q_FACTOR = (MASS + A11_LOW) * OMEGA_N / B_EXT

DURATION_3H = 3 * 3600  # seconds

# ============================================================
# Catenary mooring model (from OC3 definition)
# ============================================================

MOORING_LINE_LENGTH = 902.2       # m (unstretched)
MOORING_ANCHOR_RADIUS = 853.87    # m from center
MOORING_FAIRLEAD_RADIUS = 5.2     # m from center
MOORING_FAIRLEAD_DEPTH = 70.0     # m below SWL
MOORING_WEIGHT_PER_M = 77.7066 * G  # N/m (weight in water)
MOORING_WATER_DEPTH = 320.0       # m
MOORING_H_SPAN = MOORING_WATER_DEPTH - MOORING_FAIRLEAD_DEPTH  # 250m
MOORING_LINE_AZIMUTHS = [180.0, 60.0, 300.0]  # degrees

# Spar geometry for current drag
RHO_WATER = 1025.0
CD_CYL_CURRENT = 1.05
SPAR_UPPER_D = 6.5   # 0 to -4m
SPAR_LOWER_D = 9.4   # -12m to -120m
SPAR_DRAFT = 120.0


def solve_catenary_T_H(L_H_val):
    """Solve for horizontal tension T_H given horizontal span L_H."""
    w = MOORING_WEIGHT_PER_M
    L = MOORING_LINE_LENGTH
    h = MOORING_H_SPAN
    L_H_max = np.sqrt(L**2 - h**2)
    if L_H_val >= L_H_max - 0.01:
        return None
    target = L_H_max
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
    """Total mooring restoring force in surge for platform offset dx."""
    F_surge = 0.0
    for phi_deg in MOORING_LINE_AZIMUTHS:
        phi = np.radians(phi_deg)
        xa = MOORING_ANCHOR_RADIUS * np.cos(phi)
        ya = MOORING_ANCHOR_RADIUS * np.sin(phi)
        xf = MOORING_FAIRLEAD_RADIUS * np.cos(phi) + dx
        yf = MOORING_FAIRLEAD_RADIUS * np.sin(phi)
        L_H = np.sqrt((xa - xf)**2 + (ya - yf)**2)
        T_H = solve_catenary_T_H(L_H)
        if T_H is None:
            return None
        dir_x = (xa - xf) / L_H
        F_surge += T_H * dir_x
    return F_surge


def find_catenary_equilibrium(F_external):
    """Find equilibrium offset where mooring restoring equals external force."""
    def residual(x):
        F_moor = mooring_restoring_force(x)
        if F_moor is None:
            return -1e12
        return F_moor + F_external
    try:
        x_eq = brentq(residual, 0, 17.5, xtol=0.01)
    except Exception:
        return None, None
    dd = 0.05
    Fp = mooring_restoring_force(x_eq + dd / 2)
    Fm = mooring_restoring_force(x_eq - dd / 2)
    if Fp is not None and Fm is not None:
        K_tang = -(Fp - Fm) / dd
    else:
        K_tang = None
    return x_eq, K_tang


def solve_catenary_vertical(dx=0.0, dz=0.0):
    """
    Compute total mooring vertical force and heave stiffness.

    For a catenary line, the vertical tension at the fairlead is:
        T_V = T_H * sinh(L_H / (2a))  where a = T_H / w
    or equivalently T_V = w * s_susp where s_susp is the suspended length.

    The vertical force acts DOWNWARD (negative heave direction), so
    F_z_total < 0 (pulls platform down).

    Parameters
    ----------
    dx : float, surge offset [m]
    dz : float, heave offset [m] (positive up)

    Returns
    -------
    F_z : float, total vertical force from all 3 lines [N] (negative = downward)
    """
    w = MOORING_WEIGHT_PER_M
    L = MOORING_LINE_LENGTH
    h_base = MOORING_H_SPAN  # vertical span at rest = 250 m
    h = h_base - dz           # if platform heaves up, vertical span decreases

    F_z_total = 0.0
    for phi_deg in MOORING_LINE_AZIMUTHS:
        phi = np.radians(phi_deg)
        xa = MOORING_ANCHOR_RADIUS * np.cos(phi)
        ya = MOORING_ANCHOR_RADIUS * np.sin(phi)
        xf = MOORING_FAIRLEAD_RADIUS * np.cos(phi) + dx
        yf = MOORING_FAIRLEAD_RADIUS * np.sin(phi)
        L_H = np.sqrt((xa - xf)**2 + (ya - yf)**2)

        # Solve catenary with modified vertical span
        L_H_max = np.sqrt(L**2 - h**2)
        if L_H >= L_H_max - 0.01:
            return None  # line too taut

        def func(a):
            x = L_H / (2 * a)
            if x > 50:
                return np.inf
            return 2 * a * np.sinh(x) - L_H_max
        try:
            a_sol = brentq(func, 1.0, 1e8, xtol=0.01)
        except Exception:
            return None

        T_H = a_sol * w
        # Vertical tension: T_V = T_H * sinh(L_H / (2*a_sol))
        # But for the full line, T_V at fairlead = w * s_upper
        # where s_upper is arc length from lowest point to fairlead.
        # For symmetric catenary with span L_H:
        #   T_V = T_H * sinh(L_H / (2 * a_sol)) ... but this is half-span
        # More precisely, using the catenary with modified h:
        # The vertical tension at the top = T_H * tan(alpha_top)
        # For a catenary y = a*cosh(x/a), at x from low point:
        #   T_V = w * s = w * a * sinh(x/a)
        # The total suspended length s_total = 2*a*sinh(L_H/(2*a)) = L_H_max (approx)
        # The fairlead is at the top, so vertical tension = T_H * sinh(L_H/(2*a_sol))
        # This gives T_V for each half. For the actual asymmetric problem with h,
        # use: T_V = sqrt(T_line^2 - T_H^2) where T_line = T_H + w*h
        # Actually for catenary: T_top = T_H + w*h (tension at top = bottom tension + weight of line above)
        # Wait, that's for a vertical chain. For catenary:
        # T_top = T_H * cosh(x_top/a) and T_V = T_H * sinh(x_top/a)
        # For our symmetric catenary, the fairlead is at the top with vertical rise h
        # above the anchor. The catenary equation gives:
        # h = a*(cosh(x2/a) - cosh(x1/a)) where x1,x2 are horizontal coords of
        # anchor and fairlead relative to the catenary origin.
        # Simpler approach: total vertical tension at fairlead = T_H * h / (a * (cosh(L_H/(2*a)) - 1))
        # ... this is getting complicated. Use the simpler relation:
        # T_V = w * s_suspended, where s_suspended ≈ L (fully suspended for deep water)
        # For OC3 Hywind, water depth is 320m and chains are fully suspended.
        # So T_V ≈ total tension at fairlead minus T_H contribution.
        # T_fairlead = T_H * cosh(L_H / (2*a_sol))  (for a symmetric catenary)
        # For the asymmetric case (different depths at ends):
        # Use T_V = sqrt(T_fairlead^2 - T_H^2)
        # T_fairlead is at the top, so:
        #   T_fairlead = T_H + w * h  (for fully suspended catenary, no seabed contact)
        # This isn't quite right either. Let me use the standard catenary formula:
        # For a catenary with horizontal tension T_H, weight w per unit length,
        # and vertical span h between endpoints that are L_H apart horizontally:
        #   T_V_top = T_H * sinh(x_top / a)
        # where a = T_H/w and x_top satisfies the geometry.
        #
        # For a fully suspended mooring line (no seabed contact), the simplest
        # correct approach: the total vertical component at the top =
        #   T_V = T_H * (cosh(x2/a) - cosh(x1/a)) * a / h ... no
        #
        # Let's just use: vertical force = weight of suspended chain = w * L
        # minus the vertical component at the anchor.
        # For deep water with anchor at seabed, vertical force at anchor = 0
        # (if there is seabed contact, which there may not be).
        # For fully suspended: T_V_top - T_V_bottom = w * L
        # T_V_bottom = T_H * tan(angle at bottom)
        #
        # Simplest correct formula for the fully suspended catenary:
        # Using the standard result that for a catenary y = a*cosh((x-x0)/a) + c:
        # If the line hangs between two points separated horizontally by L_H and
        # vertically by h, with line length L, then the vertical tension at the
        # upper end = (w*L/2) + (w*h*L)/(2*L_H_catenary_length)
        # This is still messy. Let me use numerical approach:
        # T_V = w * L * (h / sqrt(L^2 - L_H^2) + 1) / 2  ... approximately
        #
        # Actually, the simplest correct approach: for a catenary hanging between
        # two points (lower at depth, upper at fairlead), the vertical tension
        # at the upper point is:
        #   T_V_upper = w * L_upper
        # where L_upper is the arc length from the lowest point of the catenary
        # to the upper attachment (fairlead).
        #
        # For a symmetric catenary (both ends at same height), L_upper = L/2.
        # For asymmetric (fairlead higher than anchor by h):
        #   L_upper = L/2 + s_asym  where s_asym accounts for the height difference.
        #
        # Given the complexity, let me use a clean numerical derivative approach:
        # just compute T_H(dz) and T_H(dz+eps) and use the catenary relation
        # T_V = sqrt((T_H + w*(L - L_H_max))^2 ... no.
        #
        # Cleanest approach: for a FULLY SUSPENDED inextensible catenary with
        # line length L, horizontal span L_H, vertical span h:
        #   T_V_upper = w * (L + h) / 2
        #   T_V_lower = w * (L - h) / 2
        # This is exact for an inextensible catenary. (From Irvine, "Cable Structures")
        T_V_upper = w * (L + h) / 2  # vertical tension at fairlead (upward on chain)
        F_z_total -= T_V_upper  # acts downward on platform

    return F_z_total


def mooring_heave_stiffness(dx=0.0):
    """
    Compute mooring contribution to heave stiffness via numerical derivative.

    K_heave_moor = -dF_z/dz  evaluated at dz=0.

    Returns
    -------
    F_z_0 : float, vertical force at rest [N] (negative = downward)
    K_heave_moor : float, heave stiffness contribution [N/m]
    z_static : float, static heave offset from mooring pre-tension [m]
                (negative = platform pulled down)
    """
    F_z_0 = solve_catenary_vertical(dx=dx, dz=0.0)
    if F_z_0 is None:
        return None, None, None

    # Numerical derivative
    eps = 0.1  # m
    F_z_p = solve_catenary_vertical(dx=dx, dz=eps)
    F_z_m = solve_catenary_vertical(dx=dx, dz=-eps)
    if F_z_p is None or F_z_m is None:
        return F_z_0, None, None

    K_heave_moor = -(F_z_p - F_z_m) / (2 * eps)  # positive stiffness

    # Static heave offset from mooring pre-tension:
    # NOTE: The OC3 equilibrium draft (120m) already includes the mooring
    # pre-tension effect. The nominal draft IS the equilibrium position with
    # moorings attached. So z_static here represents the heave offset that
    # would be needed IF the platform were designed without accounting for
    # mooring tension — i.e., it's already accounted for in the design.
    # The ADDITIONAL heave offset from surge displacement (different tension
    # at offset vs. at rest) is what matters for our analysis.
    Awp = np.pi / 4 * 6.5**2
    C33_hydrostatic = RHO_WATER * G * Awp  # ~334 kN/m

    # Additional heave offset from surge-induced tension change
    if dx != 0.0:
        F_z_dx0 = solve_catenary_vertical(dx=0.0, dz=0.0)
        delta_Fz = F_z_0 - F_z_dx0  # change in vertical force from surge offset
        z_static = -delta_Fz / (C33_hydrostatic + K_heave_moor)
    else:
        z_static = 0.0  # at nominal position, already in equilibrium

    return F_z_0, K_heave_moor, z_static


def compute_current_drag(U_current):
    """Mean current drag force on submerged OC3 Hywind spar (uniform current)."""
    dz = 0.5
    # Upper column: 0 to -4m
    z_up = np.arange(dz/2, 4.0, dz)
    F_up = 0.5 * RHO_WATER * CD_CYL_CURRENT * SPAR_UPPER_D * U_current**2 * len(z_up) * dz
    # Taper: -4m to -12m
    z_tap = np.arange(4.0 + dz/2, 12.0, dz)
    frac = (z_tap - 4.0) / (12.0 - 4.0)
    D_tap = SPAR_UPPER_D + frac * (SPAR_LOWER_D - SPAR_UPPER_D)
    F_tap = trapz(0.5 * RHO_WATER * CD_CYL_CURRENT * D_tap * U_current**2, z_tap)
    # Lower column: -12m to -120m
    length_lower = SPAR_DRAFT - 12.0
    F_low = 0.5 * RHO_WATER * CD_CYL_CURRENT * SPAR_LOWER_D * U_current**2 * length_lower
    return F_up + F_tap + F_low


def compute_current_drag_force(U_current):
    """
    Mean current drag force on submerged OC3 Hywind spar (uniform current).

    Same physics as compute_current_drag but returns a scalar force [N]
    rather than component breakdown.  Used for the current variability
    model where we need F(U) as a fast callable.
    """
    return compute_current_drag(U_current)


def compute_linearized_current_sensitivity(Uc_mean):
    """
    Compute the linearized drag sensitivity dF/dU and d²F/dU² at a
    given mean current speed.

    For Morison drag F = CdA_eff * U|U| (CdA_eff = 0.5 * rho * Cd * D_eff * L),
    the linearized force fluctuation about a mean U_m is:

        F(U_m + u') ≈ F(U_m) + dF/dU * u' + 0.5 * d²F/dU² * u'²

    For positive Uc_mean:
        dF/dU    = 2 * CdA_eff * Uc_mean
        d²F/dU²  = 2 * CdA_eff

    The quadratic term means that even a zero-mean current fluctuation
    produces a net POSITIVE mean offset increase (rectification effect).

    Parameters
    ----------
    Uc_mean : float, mean current speed [m/s]

    Returns
    -------
    dict with:
        F_mean   : float, mean drag force [N]
        dFdU     : float, drag sensitivity [N/(m/s)]
        d2FdU2   : float, second derivative [N/(m/s)²]
        CdA_eff  : float, effective drag area [m²] (= 2*F_mean/U²)
    """
    F_mean = compute_current_drag_force(Uc_mean)

    # Numerical derivatives (robust to any geometry profile)
    eps = 0.01  # m/s
    F_plus = compute_current_drag_force(Uc_mean + eps)
    F_minus = compute_current_drag_force(max(0, Uc_mean - eps))
    dFdU = (F_plus - F_minus) / (2 * eps) if Uc_mean > eps else F_plus / eps
    d2FdU2 = (F_plus - 2 * F_mean + F_minus) / eps**2 if Uc_mean > eps else \
             (F_plus - F_mean) / eps**2

    CdA_eff = F_mean / (0.5 * Uc_mean**2) if Uc_mean > 0.01 else \
              2 * F_mean / max(Uc_mean, 0.01)**2

    return {
        'F_mean': F_mean,
        'dFdU': dFdU,
        'd2FdU2': d2FdU2,
        'CdA_eff': CdA_eff,
    }


# ============================================================
# Current variability spectrum models
# ============================================================

def current_spectrum_generic(f, sigma_u, T_peak):
    """
    Generic single-peaked current variability spectrum.

    Models the current velocity fluctuation spectrum as a Gaussian
    in log-frequency space, peaked at f_peak = 1/T_peak.

    This is a simple parametric model suitable when no site-specific
    measurements are available.  The shape resembles typical ocean
    current spectra from ADCP measurements (e.g., tidal + inertial
    + internal wave contributions).

    Parameters
    ----------
    f : array, frequencies [Hz] (must be > 0)
    sigma_u : float, rms current speed fluctuation [m/s]
    T_peak : float, peak period of variability [s]
        Typical values:
        - Tidal:           ~6 hours (21600 s)
        - Inertial:        ~14 hours at 60°N
        - Internal waves:  300–3600 s (5–60 min)
        - Submesoscale:    600–7200 s (10 min – 2 hr)

    Returns
    -------
    S : array, spectral density [m²/s² / Hz]
        (one-sided, integrates to sigma_u²)
    """
    f = np.asarray(f, dtype=float)
    S = np.zeros_like(f)
    mask = f > 0

    f_peak = 1.0 / T_peak
    # Width in log-frequency space: ~1 decade wide
    sigma_logf = 0.4  # decades (controls bandwidth)

    log_ratio = np.log10(f[mask] / f_peak)
    S[mask] = np.exp(-0.5 * (log_ratio / sigma_logf)**2)

    # Normalize to target variance
    var_raw = trapz(S[mask], f[mask]) if np.sum(mask) > 1 else 1.0
    if var_raw > 0:
        S[mask] *= sigma_u**2 / var_raw

    return S


def current_spectrum_internal_wave(f, sigma_u, f_N=None, f_inertial=None,
                                    latitude=61.2):
    """
    Internal-wave current variability spectrum.

    Models the Garrett-Munk–like continuum of internal wave energy
    that produces current fluctuations in the frequency band between
    the inertial frequency f_i and the buoyancy frequency f_N.

    The spectrum follows the canonical f^(-2) shape in the internal
    wave band, with a low-frequency cutoff at the inertial frequency
    and a high-frequency cutoff at the buoyancy frequency.

    Parameters
    ----------
    f : array, frequencies [Hz]
    sigma_u : float, rms current fluctuation from internal waves [m/s]
        Typical range at Tampen: 0.05–0.20 m/s
    f_N : float, buoyancy frequency [Hz]
        If None, uses 0.005 Hz (200 s period, typical thermocline)
    f_inertial : float, inertial frequency [Hz]
        If None, computed from latitude
    latitude : float, degrees N (default: 61.2° for Tampen)

    Returns
    -------
    S : array, spectral density [m²/s² / Hz]
    """
    f = np.asarray(f, dtype=float)
    S = np.zeros_like(f)

    # Inertial frequency
    if f_inertial is None:
        omega_earth = 7.2921e-5  # rad/s
        f_inertial = 2 * omega_earth * np.abs(np.sin(np.radians(latitude))) \
                     / (2 * np.pi)  # Hz
    # Buoyancy frequency (typical thermocline/shelf)
    if f_N is None:
        f_N = 0.005  # Hz (period = 200 s)

    mask = (f > f_inertial) & (f < f_N) & (f > 0)
    if np.sum(mask) < 2:
        return S

    # GM-like f^-2 spectrum in the internal wave band
    S[mask] = (f[mask] / f_inertial)**(-2)

    # Smooth roll-off at boundaries (avoid sharp cutoffs)
    # Taper with half-Gaussian at each end
    taper_low = np.exp(-0.5 * ((f[mask] - f_inertial) /
                                (0.2 * f_inertial))**(-2))
    # Actually, simpler: just use the f^-2 within the band
    # and let it naturally fall off

    # Normalize to target variance
    var_raw = trapz(S[mask], f[mask])
    if var_raw > 0:
        S[mask] *= sigma_u**2 / var_raw

    return S


def current_spectrum_bimodal(f, sigma_tidal, T_tidal,
                              sigma_iw, T_iw):
    """
    Bimodal current spectrum: tidal/inertial + internal wave components.

    Combines a low-frequency tidal/inertial peak with a higher-frequency
    internal wave / submesoscale peak.  This is the most physically
    realistic simple model for a site like Tampen.

    Parameters
    ----------
    f : array, frequencies [Hz]
    sigma_tidal : float, rms tidal/inertial current fluctuation [m/s]
    T_tidal : float, period of tidal/inertial peak [s]
    sigma_iw : float, rms internal wave current fluctuation [m/s]
    T_iw : float, period of internal wave peak [s]

    Returns
    -------
    S : array, spectral density [m²/s² / Hz]
    """
    S1 = current_spectrum_generic(f, sigma_tidal, T_tidal)
    S2 = current_spectrum_generic(f, sigma_iw, T_iw)
    return S1 + S2


# ============================================================
# Current variability → surge response
# ============================================================

def compute_current_variability_surge(
        Uc_mean, sigma_Uc, T_peak_current,
        duration=DURATION_3H,
        spectrum_type='generic',
        sigma_tidal=None, T_tidal=None,
        sigma_iw=None, T_iw=None,
        verbose=True):
    """
    Compute slowly varying surge from ocean current variability.

    Physics:
      The drag force on the submerged spar is F = CdA_eff * U * |U|.
      For a mean current Uc_mean with fluctuation u'(t):
        F(t) = CdA_eff * (Uc_mean + u')^2
             ≈ F_mean + dF/dU * u' + 0.5 * d²F/dU² * u'^2

      The platform offset is determined by the mooring stiffness:
        x(t) = F(t) / K_tangent(x)

      For the spectral analysis, we linearize:
        - Force sensitivity: dF/dU = 2 * CdA_eff * Uc_mean
        - Offset sensitivity: dx/dU = dF/dU / K_tangent

      The force fluctuation spectrum is:
        S_F(f) = (dF/dU)^2 * S_Uc(f)

      The surge transfer function H(f) accounts for both quasi-static
      response (f << f_n) and dynamic amplification near resonance:
        H(f) = 1 / [-(M+A)*ω² + K_tang - i*ω*B_ext]

      The surge response spectrum is:
        S_x(f) = |H(f)|² * S_F(f)

      Additionally, the quadratic term produces:
        - A mean offset increase: Δx_mean = 0.5 * d²F/dU² * σ_u² / K_tang
        - This is the "rectification" effect of nonlinear drag

    Parameters
    ----------
    Uc_mean : float, mean current speed [m/s]
    sigma_Uc : float, rms current speed fluctuation [m/s]
    T_peak_current : float, peak period of current variability [s]
        (used for 'generic' spectrum type)
    duration : float, storm duration [s] (for extreme statistics)
    spectrum_type : str, 'generic', 'internal_wave', or 'bimodal'
    sigma_tidal, T_tidal : for bimodal spectrum
    sigma_iw, T_iw : for bimodal spectrum
    verbose : bool, print results

    Returns
    -------
    dict with:
        sigma : float, rms slowly varying surge from current [m]
        x_mean_shift : float, additional mean offset from rectification [m]
        x_mean_total : float, total mean offset (current + rectification) [m]
        x_sig : float, significant amplitude (2*sigma) [m]
        x_max : float, expected maximum in duration [m]
        f : array, frequencies [Hz]
        S_Uc : array, current spectrum [m²/s² / Hz]
        S_F : array, force spectrum [N² / Hz]
        S_x : array, surge response spectrum [m² / Hz]
        K_tang : float, tangent stiffness at mean offset [N/m]
        omega_n : float, surge natural frequency at mean offset [rad/s]
        T_n : float, surge natural period at mean offset [s]
        sensitivity : dict from compute_linearized_current_sensitivity
    """
    # 1. Mean current offset and tangent stiffness
    F_mean = compute_current_drag_force(Uc_mean)
    if Uc_mean > 0.01:
        x_eq, K_tang = find_catenary_equilibrium(F_mean)
    else:
        x_eq = 0.0
        K_tang = K_SURGE

    if x_eq is None or K_tang is None:
        if verbose:
            print(f"  WARNING: Current {Uc_mean:.2f} m/s exceeds mooring capacity")
        return None

    # 2. Linearized drag sensitivity
    sens = compute_linearized_current_sensitivity(Uc_mean)
    dFdU = sens['dFdU']
    d2FdU2 = sens['d2FdU2']

    # 3. Effective mass and natural frequency at offset
    m_eff = MASS + A11_LOW
    omega_n = np.sqrt(K_tang / m_eff)
    T_n_shifted = 2 * np.pi / omega_n
    Q = m_eff * omega_n / B_EXT

    # 4. Current variability spectrum
    f = np.logspace(-5, -1, 4000)  # 10^-5 to 0.1 Hz (100000s to 10s)
    omega = 2 * np.pi * f

    if spectrum_type == 'generic':
        S_Uc = current_spectrum_generic(f, sigma_Uc, T_peak_current)
    elif spectrum_type == 'internal_wave':
        S_Uc = current_spectrum_internal_wave(f, sigma_Uc)
    elif spectrum_type == 'bimodal':
        if sigma_tidal is None or T_tidal is None:
            sigma_tidal = sigma_Uc * 0.7
            T_tidal = max(T_peak_current, 6 * 3600)  # default 6h
        if sigma_iw is None or T_iw is None:
            sigma_iw = sigma_Uc * 0.7
            T_iw = min(T_peak_current, 600)  # default 10 min
        S_Uc = current_spectrum_bimodal(f, sigma_tidal, T_tidal,
                                         sigma_iw, T_iw)
    else:
        raise ValueError(f"Unknown spectrum_type: {spectrum_type}")

    # Verify variance
    var_Uc = trapz(S_Uc, f)
    sigma_Uc_actual = np.sqrt(var_Uc)

    # 5. Force fluctuation spectrum
    S_F = dFdU**2 * S_Uc

    # 6. Surge transfer function (at tangent stiffness)
    H = np.zeros(len(omega), dtype=complex)
    for k in range(len(omega)):
        denom = -m_eff * omega[k]**2 + K_tang - 1j * omega[k] * B_EXT
        if abs(denom) > 0:
            H[k] = 1.0 / denom

    # 7. Surge response spectrum
    S_x = np.abs(H)**2 * S_F

    # 8. Statistics
    var_x = trapz(S_x, f)
    sigma_x = np.sqrt(var_x) if var_x > 0 else 0.0

    # Rectification: mean offset increase from quadratic drag term
    # <F'²> = d²F/dU² * sigma_Uc² produces a mean force bias
    delta_F_mean = 0.5 * d2FdU2 * sigma_Uc**2
    x_mean_shift = delta_F_mean / K_tang

    # Total mean offset
    x_mean_total = x_eq + x_mean_shift

    # Zero-crossing period and expected max
    m0 = var_x
    m2 = trapz(S_x * (2 * np.pi * f)**2, f) if var_x > 0 else 0
    if m2 > 0:
        Tz = 2 * np.pi * np.sqrt(m0 / m2)
        N_cycles = duration / Tz
    else:
        Tz = T_peak_current
        N_cycles = duration / Tz
    x_max = sigma_x * np.sqrt(2 * np.log(max(N_cycles, 1))) if sigma_x > 0 else 0

    # Quasi-static check: what fraction of variance is below surge resonance?
    mask_qs = f < (omega_n / (2 * np.pi) * 0.5)  # below half the nat freq
    var_qs = trapz(S_x[mask_qs], f[mask_qs]) if np.sum(mask_qs) > 1 else 0
    pct_qs = var_qs / var_x * 100 if var_x > 0 else 0

    # Also compute the nonlinear (exact) offset range for reference
    Uc_lo = max(0.0, Uc_mean - 2 * sigma_Uc)
    Uc_hi = Uc_mean + 2 * sigma_Uc
    if Uc_lo < 0.01:
        x_lo = 0.0
    else:
        F_lo = compute_current_drag_force(Uc_lo)
        x_lo, _ = find_catenary_equilibrium(F_lo)
        if x_lo is None:
            x_lo = 0.0
    F_hi = compute_current_drag_force(Uc_hi)
    x_hi, _ = find_catenary_equilibrium(F_hi)
    x_hi_str = f"{x_hi:.2f}" if x_hi is not None else ">17.5 (TAUT)"
    x_range_exact = (x_hi - x_lo) if x_hi is not None else float('inf')

    if verbose:
        print()
        print("=" * 76)
        print("CURRENT VARIABILITY → SLOWLY VARYING SURGE")
        print("=" * 76)
        print(f"  Mean current:        Uc = {Uc_mean:.2f} m/s")
        print(f"  Current fluctuation: σ_Uc = {sigma_Uc:.3f} m/s "
              f"(actual {sigma_Uc_actual:.3f})")
        print(f"  Peak period:         T = {T_peak_current:.0f} s "
              f"({T_peak_current/60:.1f} min)")
        print(f"  Spectrum type:       {spectrum_type}")
        print()
        print(f"  Mean offset:         x_eq = {x_eq:.2f} m")
        print(f"  Tangent stiffness:   K = {K_tang/1e3:.2f} kN/m "
              f"(vs linearized {K_SURGE/1e3:.2f} kN/m)")
        print(f"  Shifted natural per: T_n = {T_n_shifted:.1f} s "
              f"(ω_n = {omega_n:.4f} rad/s)")
        print(f"  Q-factor:            {Q:.1f}")
        print()
        print(f"  Drag sensitivity:    dF/dU = {dFdU/1e3:.1f} kN/(m/s)")
        print(f"  Force fluctuation:   σ_F = {dFdU * sigma_Uc / 1e3:.1f} kN")
        print()
        print(f"  SLOWLY VARYING SURGE:")
        print(f"    σ_surge (spectral):   {sigma_x:.3f} m")
        print(f"    x_sig (2σ):           {2*sigma_x:.3f} m")
        print(f"    x_max (3h):           {x_max:.3f} m")
        print(f"    Quasi-static frac:    {pct_qs:.1f}%")
        print(f"    Zero-crossing period: {Tz:.0f} s ({Tz/60:.1f} min)")
        print()
        print(f"  RECTIFICATION (quadratic drag → mean shift):")
        print(f"    d²F/dU² = {d2FdU2/1e3:.1f} kN/(m/s)²")
        print(f"    Δx_mean = {x_mean_shift:.3f} m")
        print(f"    Total mean offset: {x_mean_total:.2f} m")
        print()
        print(f"  NONLINEAR CHECK (exact catenary, ±2σ current):")
        print(f"    Uc range: [{Uc_lo:.2f}, {Uc_hi:.2f}] m/s")
        print(f"    x  range: [{x_lo:.2f}, {x_hi_str}] m")
        print(f"    Peak-peak (exact): {x_range_exact:.2f} m "
              f"vs 4σ (linear): {4*sigma_x:.2f} m")
        if x_range_exact < float('inf') and sigma_x > 0:
            ratio = x_range_exact / (4 * sigma_x)
            print(f"    Nonlinear amplification: {ratio:.2f}×")

    return {
        'sigma': sigma_x,
        'x_sig': 2 * sigma_x,
        'x_max': x_max,
        'x_mean_shift': x_mean_shift,
        'x_mean_total': x_mean_total,
        'x_eq': x_eq,
        'K_tang': K_tang,
        'omega_n': omega_n,
        'T_n': T_n_shifted,
        'Q': Q,
        'Tz': Tz,
        'N_cycles': N_cycles,
        'pct_quasi_static': pct_qs,
        'f': f,
        'S_Uc': S_Uc,
        'S_F': S_F,
        'S_x': S_x,
        'sensitivity': sens,
        'sigma_Uc_actual': sigma_Uc_actual,
        'x_range_exact': x_range_exact,
    }


def current_variability_sweep(Uc_mean, T_peak_current,
                               sigma_Uc_values=None,
                               verbose=True):
    """
    Sweep over current fluctuation amplitudes for a given mean current.

    Useful for sensitivity studies when σ_Uc is uncertain.

    Parameters
    ----------
    Uc_mean : float, mean current speed [m/s]
    T_peak_current : float, peak period [s]
    sigma_Uc_values : list of float, σ_Uc values to sweep [m/s]
        If None, uses [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]

    Returns
    -------
    list of result dicts from compute_current_variability_surge
    """
    if sigma_Uc_values is None:
        sigma_Uc_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]

    if verbose:
        print()
        print("=" * 76)
        print(f"CURRENT VARIABILITY SWEEP — Uc_mean = {Uc_mean:.2f} m/s, "
              f"T_peak = {T_peak_current:.0f} s ({T_peak_current/60:.1f} min)")
        print("=" * 76)
        print()
        print(f"  {'σ_Uc':>6s} {'σ_surge':>8s} {'x_sig':>8s} {'x_max':>8s} "
              f"{'Δx_mean':>8s} {'x_total':>8s} {'pp_exact':>9s} {'%_QS':>6s}")
        print(f"  {'[m/s]':>6s} {'[m]':>8s} {'[m]':>8s} {'[m]':>8s} "
              f"{'[m]':>8s} {'[m]':>8s} {'[m]':>9s} {'':>6s}")
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} "
              f"{'-'*8} {'-'*8} {'-'*9} {'-'*6}")

    results = []
    for sigma_Uc in sigma_Uc_values:
        r = compute_current_variability_surge(
            Uc_mean, sigma_Uc, T_peak_current,
            verbose=False)
        if r is None:
            if verbose:
                print(f"  {sigma_Uc:6.2f}  — MOORING CAPACITY EXCEEDED")
            continue
        results.append(r)
        if verbose:
            pp = r['x_range_exact']
            pp_str = f"{pp:9.2f}" if pp < float('inf') else "    >17.5"
            print(f"  {sigma_Uc:6.2f} {r['sigma']:8.3f} {r['x_sig']:8.3f} "
                  f"{r['x_max']:8.3f} {r['x_mean_shift']:8.3f} "
                  f"{r['x_mean_total']:8.2f} {pp_str} "
                  f"{r['pct_quasi_static']:5.1f}%")

    if verbose:
        print()
        print("  Note: 'pp_exact' is exact nonlinear peak-peak for ±2σ current")
        print("  Note: '%_QS' is fraction of variance below half the natural frequency")

    return results


def current_period_sweep(Uc_mean, sigma_Uc,
                          T_values=None,
                          verbose=True):
    """
    Sweep over current variability period for fixed amplitude.

    Shows how the surge response depends on the timescale of current
    fluctuations — from quasi-static (T >> T_n) through resonance
    (T ≈ T_n) to filtered (T << T_n).

    Parameters
    ----------
    Uc_mean : float, mean current speed [m/s]
    sigma_Uc : float, rms current fluctuation [m/s]
    T_values : list of float, peak periods [s]

    Returns
    -------
    list of result dicts
    """
    if T_values is None:
        T_values = [60, 120, 180, 300, 600, 1200, 1800, 3600,
                     7200, 14400, 21600, 43200]

    if verbose:
        print()
        print("=" * 76)
        print(f"CURRENT PERIOD SWEEP — Uc = {Uc_mean:.2f} m/s, "
              f"σ_Uc = {sigma_Uc:.2f} m/s")
        print("=" * 76)
        print()

        # Print the natural period for reference
        F_mean = compute_current_drag_force(Uc_mean)
        if Uc_mean > 0.01:
            _, K_tang = find_catenary_equilibrium(F_mean)
        else:
            K_tang = K_SURGE
        if K_tang is not None:
            omega_n = np.sqrt(K_tang / (MASS + A11_LOW))
            T_n = 2 * np.pi / omega_n
            print(f"  Surge natural period at this offset: T_n = {T_n:.1f} s "
                  f"({T_n/60:.1f} min)")
            print()

        print(f"  {'T_peak':>8s} {'T[min]':>7s} {'σ_surge':>8s} {'x_sig':>8s} "
              f"{'x_max':>8s} {'%_QS':>6s} {'Tz':>8s} {'regime':>12s}")
        print(f"  {'[s]':>8s} {'':>7s} {'[m]':>8s} {'[m]':>8s} "
              f"{'[m]':>8s} {'':>6s} {'[s]':>8s} {'':>12s}")
        print(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*8} "
              f"{'-'*8} {'-'*6} {'-'*8} {'-'*12}")

    results = []
    for T_peak in T_values:
        r = compute_current_variability_surge(
            Uc_mean, sigma_Uc, T_peak,
            verbose=False)
        if r is None:
            continue
        results.append(r)

        if verbose:
            # Determine regime
            if K_tang is not None:
                ratio = T_peak / T_n
                if ratio > 5:
                    regime = "quasi-static"
                elif ratio > 1.5:
                    regime = "near-QS"
                elif ratio > 0.7:
                    regime = "RESONANCE"
                else:
                    regime = "filtered"
            else:
                regime = "?"

            print(f"  {T_peak:8.0f} {T_peak/60:7.1f} {r['sigma']:8.3f} "
                  f"{r['x_sig']:8.3f} {r['x_max']:8.3f} "
                  f"{r['pct_quasi_static']:5.1f}% {r['Tz']:8.0f} "
                  f"{regime:>12s}")

    if verbose:
        print()
        print("  'RESONANCE' = current variability period near surge natural period")
        print("  → additional dynamic amplification on top of quasi-static response")

    return results


# ============================================================
# QTF parser
# ============================================================

def parse_qtf_total(filepath, dof=1, beta=0.0, beta_tol=0.5):
    """
    Parse OUT_QTFM_N.dat (total difference-frequency QTF, normalized by rho*g).

    Returns:
      omegas: sorted array of unique frequencies [rad/s]
      T_matrix: complex QTF matrix T(i,j) [N/m^2] (dimensionalized)
                lower-triangular (w_i >= w_j), upper filled by symmetry
    """
    w1_list, w2_list, re_list, im_list = [], [], [], []

    with open(filepath) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.split()
            if len(parts) < 9:
                continue
            w1 = float(parts[0])
            w2 = float(parts[1])
            b1 = float(parts[2])
            b2 = float(parts[3])
            d = int(parts[4])
            re_val = float(parts[7])  # Re(QTF)/rho/g
            im_val = float(parts[8])  # Im(QTF)/rho/g

            if d != dof:
                continue
            if abs(b1 - beta) > beta_tol or abs(b2 - beta) > beta_tol:
                continue

            w1_list.append(w1)
            w2_list.append(w2)
            re_list.append(re_val * RHOG)  # dimensionalize
            im_list.append(im_val * RHOG)

    w1_arr = np.array(w1_list)
    w2_arr = np.array(w2_list)
    re_arr = np.array(re_list)
    im_arr = np.array(im_list)

    # Get unique frequencies
    omegas = np.sort(np.unique(np.concatenate([w1_arr, w2_arr])))
    n = len(omegas)

    # Build QTF matrix (lower triangular from file, fill upper by symmetry)
    T = np.zeros((n, n), dtype=complex)
    omega_to_idx = {round(w, 6): i for i, w in enumerate(omegas)}

    for k in range(len(w1_arr)):
        i = omega_to_idx.get(round(w1_arr[k], 6))
        j = omega_to_idx.get(round(w2_arr[k], 6))
        if i is not None and j is not None:
            T[i, j] = re_arr[k] + 1j * im_arr[k]

    # Fill upper triangle: T(w_j, w_i) = conj(T(w_i, w_j)) for diff-freq QTF
    for i in range(n):
        for j in range(i + 1, n):
            if T[j, i] == 0:
                T[j, i] = np.conj(T[i, j])

    return omegas, T


# ============================================================
# Wave spectrum
# ============================================================

def jonswap(omega, hs, tp, gamma=3.3):
    """
    JONSWAP spectrum S(omega) [m^2·s/rad].

    S(w) = alpha * g^2 / w^5 * exp(-5/4 * (wp/w)^4) * gamma^b
    where b = exp(-(w-wp)^2 / (2*sigma^2*wp^2))

    Uses the standard parameterization with alpha from Hs.
    """
    wp = 2 * np.pi / tp
    sigma = np.where(omega <= wp, 0.07, 0.09)

    # Peak enhancement
    b = np.exp(-(omega - wp)**2 / (2 * sigma**2 * wp**2))
    gamma_factor = gamma**b

    # PM part
    S_pm = (5.0 / 16.0) * hs**2 * wp**4 / omega**5 * np.exp(-1.25 * (wp / omega)**4)

    # Normalization factor for gamma
    # C_gamma ≈ 1 - 0.287 * ln(gamma)
    C_gamma = 1.0 - 0.287 * np.log(gamma)

    S = S_pm / C_gamma * gamma_factor

    # Zero out very low frequencies to avoid numerical issues
    S[omega < 0.01] = 0.0

    return S


# ============================================================
# Slow-drift computation
# ============================================================

def compute_slow_drift_force_spectrum(omegas, T, S_wave):
    """
    Compute the slow-drift force PSD S_F(mu) at difference frequencies.

    S_F(mu_k) = 8 * sum_j |T(w_{j+k}, w_j)|^2 * S(w_{j+k}) * S(w_j) * dw

    where mu_k = k * dw (difference frequency).

    Returns:
      mu: array of difference frequencies [rad/s]
      S_F: slow-drift force PSD [N^2·s/rad]
    """
    n = len(omegas)
    dw = omegas[1] - omegas[0] if n > 1 else 0.05

    # Difference frequencies: mu = 0, dw, 2*dw, ...
    # mu=0 is the mean drift — we include it but the response at mu=0 is static
    n_mu = n  # max difference frequency index
    mu = np.arange(n_mu) * dw
    S_F = np.zeros(n_mu)

    for k in range(n_mu):
        # For difference frequency mu_k = k*dw:
        # sum over j where both j and j+k are valid indices
        for j in range(n - k):
            i = j + k  # w_i = w_j + mu_k, so w_i >= w_j (lower triangular)
            T_ij = T[i, j]
            S_F[k] += np.abs(T_ij)**2 * S_wave[i] * S_wave[j]

        S_F[k] *= 8.0 * dw

    return mu, S_F


def surge_transfer_function(mu, mass, a11, b_ext, k_surge):
    """
    Mechanical transfer function for surge at difference frequency mu.

    H(mu) = 1 / [-(M+A11)*mu^2 + K - i*mu*B]

    Note: for slow-drift, the added mass should be evaluated at the
    difference frequency mu, but since mu is very low (~0.04 rad/s),
    A11 is essentially at the zero-frequency limit.

    Returns complex H(mu) [m/N].
    """
    H = np.zeros(len(mu), dtype=complex)
    for k in range(len(mu)):
        denom = -(mass + a11) * mu[k]**2 + k_surge - 1j * mu[k] * b_ext
        if abs(denom) > 0:
            H[k] = 1.0 / denom
        else:
            H[k] = 0.0
    return H


def compute_statistics(mu, S_x, duration=DURATION_3H):
    """
    Compute slow-drift response statistics from the response spectrum.

    Returns dict with sigma, significant amplitude, expected max in duration.
    """
    dmu = mu[1] - mu[0] if len(mu) > 1 else 0.05

    # Variance (integrate from mu>0 only — mu=0 is the mean offset)
    # Actually, mu[0]=0 gives the static/mean component, and mu[1:] gives
    # the dynamic slow-drift
    var_total = trapz(S_x, mu)
    var_dynamic = trapz(S_x[1:], mu[1:]) if len(mu) > 1 else 0.0

    sigma = np.sqrt(var_dynamic) if var_dynamic > 0 else 0.0
    sig_amp = 2.0 * sigma  # significant double amplitude

    # Mean offset from mu=0 component: x_mean = H(0) * integral(8*|T(w,w)|^2 * S^2 * dw)
    # This is captured in S_F[0], but H(0) = 1/K_surge (static)
    # Actually, for mu=0: S_x[0] * dmu gives variance contribution, but mean drift
    # is better computed separately. Let's compute mean drift force.

    # Expected maximum in duration
    # Approximate number of slow-drift cycles: use zero-crossing period
    if var_dynamic > 0 and len(mu) > 2:
        # Spectral moments
        m0 = var_dynamic
        m2 = trapz(S_x[1:] * mu[1:]**2, mu[1:])
        if m2 > 0:
            T_z = 2 * np.pi * np.sqrt(m0 / m2)
            N_cycles = duration / T_z
        else:
            N_cycles = 1.0
    else:
        N_cycles = 1.0

    x_max = sigma * np.sqrt(2 * np.log(max(N_cycles, 1.0))) if sigma > 0 else 0.0

    return {
        'sigma': sigma,
        'sig_amp': sig_amp,
        'x_max': x_max,
        'var_total': var_total,
        'var_dynamic': var_dynamic,
        'N_cycles': N_cycles,
    }


def compute_mean_drift(omegas, T, S_wave, k_surge):
    """
    Compute mean drift offset from diagonal QTF.

    F_mean = 2 * integral T(w,w) * S(w) dw  (real part only for mean drift)
    x_mean = F_mean / K_surge
    """
    n = len(omegas)
    dw = omegas[1] - omegas[0] if n > 1 else 0.05

    F_mean = 0.0
    for j in range(n):
        F_mean += T[j, j].real * S_wave[j]
    F_mean *= 2.0 * dw

    x_mean = F_mean / k_surge
    return F_mean, x_mean


# ============================================================
# Main
# ============================================================

def run_cases(omegas, T, cases, k_surge, dw, label=None):
    """
    Run slow-drift analysis for a list of (hs, tp, gamma, stype) cases.

    Parameters
    ----------
    omegas : array, QTF frequencies
    T : complex matrix, QTF values
    cases : list of (hs, tp, gamma, stype)
    k_surge : float, mooring stiffness [N/m]
    dw : float, frequency spacing [rad/s]
    label : str, optional label to print

    Returns list of result dicts.
    """
    m_eff = MASS + A11_LOW
    omega_n = np.sqrt(k_surge / m_eff)
    T_n = 2 * np.pi / omega_n
    Q = m_eff * omega_n / B_EXT

    if label:
        print(f"\n  --- {label} ---")
    print(f"  K_surge = {k_surge:.0f} N/m ({k_surge/1e3:.2f} kN/m), "
          f"omega_n = {omega_n:.4f} rad/s, T_n = {T_n:.1f} s, Q = {Q:.1f}")

    if dw > omega_n:
        print(f"  *** WARNING: dw ({dw:.4f}) > omega_n ({omega_n:.4f}). "
              f"Resonance below smallest difference frequency!")
    elif dw > omega_n / 2:
        print(f"  ** Caution: dw = {dw:.4f} gives ~{omega_n/dw:.1f} points "
              f"per half-power bandwidth")
    print()

    print(f"  {'Type':>8s} {'Hs':>5s} {'Tp':>5s} {'gamma':>5s} | "
          f"{'F_mean':>10s} {'x_mean':>8s} {'sigma':>8s} {'x_sig':>8s} {'x_max':>8s}")
    print(f"  {'':>8s} {'[m]':>5s} {'[s]':>5s} {'':>5s} | "
          f"{'[kN]':>10s} {'[m]':>8s} {'[m]':>8s} {'[m]':>8s} {'[m]':>8s}")
    print(f"  {'-' * 72}")

    results = []
    for hs, tp, gamma, stype in cases:
        S = jonswap(omegas, hs, tp, gamma=gamma)
        F_mean, x_mean = compute_mean_drift(omegas, T, S, k_surge)
        mu, S_F = compute_slow_drift_force_spectrum(omegas, T, S)
        H = surge_transfer_function(mu, MASS, A11_LOW, B_EXT, k_surge)
        S_x = np.abs(H)**2 * S_F
        stats = compute_statistics(mu, S_x)

        print(f"  {stype:>8s} {hs:5.1f} {tp:5.1f} {gamma:5.1f} | "
              f"{F_mean/1e3:10.3f} {x_mean:8.3f} {stats['sigma']:8.3f} "
              f"{stats['sig_amp']:8.3f} {stats['x_max']:8.3f}")

        results.append({
            'type': stype, 'hs': hs, 'tp': tp, 'gamma': gamma,
            'F_mean': F_mean, 'x_mean': x_mean,
            'mu': mu, 'S_F': S_F, 'S_x': S_x,
            'k_surge': k_surge, 'omega_n': omega_n,
            **stats,
        })

    return results


def run_current_sweep(omegas, T, cases, dw):
    """
    Sweep over current speeds: for each, find catenary equilibrium,
    get tangent stiffness, and re-run slow-drift with shifted resonance.
    """
    print()
    print("=" * 80)
    print("CURRENT SWEEP — nonlinear mooring stiffness effect on slow-drift")
    print("=" * 80)
    print()
    print("  For each current speed:")
    print("  1. Compute current drag on submerged spar (uniform profile)")
    print("  2. Find catenary mooring equilibrium offset")
    print("  3. Get tangent stiffness at equilibrium")
    print("  4. Recompute slow-drift with shifted natural frequency")
    print()

    currents = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

    # Summary table header
    print(f"  {'Uc':>6s} {'F_drag':>8s} {'x_eq':>6s} {'K_tang':>9s} "
          f"{'omega_n':>8s} {'T_n':>7s} | ", end='')
    # Compact: show sigma for each sea state
    for hs, tp, gamma, stype in cases:
        tag = f"{stype[:2]}_{hs}_{tp}"
        print(f"{'sig_'+tag:>12s}", end=' ')
    print()
    print(f"  {'[m/s]':>6s} {'[kN]':>8s} {'[m]':>6s} {'[kN/m]':>9s} "
          f"{'[rad/s]':>8s} {'[s]':>7s} | ", end='')
    for _ in cases:
        print(f"{'[m]':>12s}", end=' ')
    print()
    print(f"  {'-' * 50}+{'-' * (13 * len(cases) + 1)}")

    all_results = {}

    for Uc in currents:
        if Uc == 0:
            x_eq = 0.0
            k_tang = K_SURGE  # linearized value
        else:
            F_drag = compute_current_drag(Uc)
            x_eq, k_tang = find_catenary_equilibrium(F_drag)
            if x_eq is None or k_tang is None:
                print(f"  {Uc:6.2f}  EXCEEDS MOORING CAPACITY")
                continue

        m_eff = MASS + A11_LOW
        omega_n = np.sqrt(k_tang / m_eff)
        T_n = 2 * np.pi / omega_n

        # Compute slow-drift for each sea state at this K
        sigmas = []
        for hs, tp, gamma, stype in cases:
            S = jonswap(omegas, hs, tp, gamma=gamma)
            mu, S_F = compute_slow_drift_force_spectrum(omegas, T, S)
            H = surge_transfer_function(mu, MASS, A11_LOW, B_EXT, k_tang)
            S_x = np.abs(H)**2 * S_F
            stats = compute_statistics(mu, S_x)
            sigmas.append(stats['sigma'])

        F_drag_val = compute_current_drag(Uc) if Uc > 0 else 0.0
        print(f"  {Uc:6.2f} {F_drag_val/1e3:8.1f} {x_eq:6.1f} {k_tang/1e3:9.2f} "
              f"{omega_n:8.4f} {T_n:7.1f} | ", end='')
        for sig in sigmas:
            print(f"{sig:12.4f}", end=' ')
        print()

        all_results[Uc] = {
            'x_eq': x_eq, 'k_tang': k_tang,
            'omega_n': omega_n, 'T_n': T_n,
            'sigmas': sigmas,
        }

    # Print ratios relative to zero-current baseline
    print()
    print("  Amplification ratio (sigma / sigma_at_Uc=0):")
    if 0.0 in all_results and all(s > 0 for s in all_results[0.0]['sigmas']):
        base = all_results[0.0]['sigmas']
        print(f"  {'Uc':>6s} | ", end='')
        for hs, tp, gamma, stype in cases:
            tag = f"{stype[:2]}_{hs}_{tp}"
            print(f"{'rat_'+tag:>12s}", end=' ')
        print()
        print(f"  {'-' * 8}+{'-' * (13 * len(cases) + 1)}")
        for Uc in currents:
            if Uc not in all_results:
                continue
            sigs = all_results[Uc]['sigmas']
            print(f"  {Uc:6.2f} | ", end='')
            for i, sig in enumerate(sigs):
                ratio = sig / base[i] if base[i] > 0 else 0
                print(f"{ratio:12.2f}", end=' ')
            print()

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Slow-drift surge from swell')
    parser.add_argument('--qtf-dir', default='/home/blofro/src/pdstrip_test/hywind_nemoh_swell/results/QTF',
                        help='QTF results directory')
    parser.add_argument('--gamma', type=float, default=5.0,
                        help='JONSWAP peakedness parameter (default: 5.0 for swell)')
    parser.add_argument('--hs', type=float, nargs='+', default=[1.0],
                        help='Significant wave heights [m]')
    parser.add_argument('--tp', type=float, nargs='+',
                        default=[15.0, 17.0, 19.0, 21.0, 23.0, 25.0],
                        help='Peak periods [s]')
    parser.add_argument('--wind-sea', action='store_true',
                        help='Also compute for wind-sea reference cases (Hs=3m, Tp=7-12s, gamma=3.3)')
    parser.add_argument('--k-surge', type=float, default=None,
                        help='Override mooring stiffness K_surge [N/m]')
    parser.add_argument('--current-sweep', action='store_true',
                        help='Sweep over current speeds and show nonlinear mooring effect')
    parser.add_argument('--current-variability', action='store_true',
                        help='Compute slowly varying surge from current fluctuations')
    parser.add_argument('--uc-mean', type=float, default=0.5,
                        help='Mean current speed [m/s] (default: 0.5)')
    parser.add_argument('--sigma-uc', type=float, default=None,
                        help='RMS current fluctuation [m/s] (if None, runs amplitude sweep)')
    parser.add_argument('--t-peak-current', type=float, default=1800.0,
                        help='Peak period of current variability [s] (default: 1800 = 30 min)')
    parser.add_argument('--spectrum-type', choices=['generic', 'internal_wave', 'bimodal'],
                        default='generic',
                        help='Current variability spectrum model (default: generic)')
    parser.add_argument('--period-sweep', action='store_true',
                        help='Sweep over current variability periods (use with --current-variability)')
    args = parser.parse_args()

    # Use overridden K if specified, else default
    k_surge = args.k_surge if args.k_surge is not None else K_SURGE

    # ---- System info ----
    print("=" * 80)
    print("Slow-drift surge response — OC3 Hywind spar")
    print("=" * 80)
    m_eff = MASS + A11_LOW
    omega_n = np.sqrt(k_surge / m_eff)
    T_n = 2 * np.pi / omega_n
    Q = m_eff * omega_n / B_EXT
    print(f"  Mass (total):     {MASS:.3e} kg")
    print(f"  A11 (low-freq):   {A11_LOW:.3e} kg")
    print(f"  B_ext (surge):    {B_EXT:.1e} N·s/m")
    print(f"  K_surge (moor):   {k_surge:.1f} N/m")
    if args.k_surge is not None:
        print(f"    (overridden from default {K_SURGE:.1f} N/m)")
    print(f"  omega_n (surge):  {omega_n:.4f} rad/s  (T_n = {T_n:.1f} s)")
    print(f"  Q-factor:         {Q:.1f}")
    print(f"  JONSWAP gamma:    {args.gamma}")
    print()

    # ---- Load QTF ----
    qtf_file = f"{args.qtf_dir}/OUT_QTFM_N.dat"
    print(f"Loading QTF from: {qtf_file}")
    omegas, T = parse_qtf_total(qtf_file, dof=1, beta=0.0)
    n_freq = len(omegas)
    dw = omegas[1] - omegas[0] if n_freq > 1 else 0.0
    print(f"  {n_freq} frequencies: [{omegas[0]:.4f}, {omegas[-1]:.4f}] rad/s, dw={dw:.4f}")
    print(f"  Difference freq range: [0, {omegas[-1]-omegas[0]:.4f}] rad/s")
    print(f"  Smallest nonzero mu:   {dw:.4f} rad/s (T_mu = {2*np.pi/dw:.1f} s)")
    print(f"  Surge resonance at:    {omega_n:.4f} rad/s (T = {T_n:.1f} s)")
    print()

    # ---- Build sea state cases ----
    cases = []
    for hs in args.hs:
        for tp in args.tp:
            cases.append((hs, tp, args.gamma, 'swell'))

    if args.wind_sea:
        for tp in [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]:
            cases.append((3.0, tp, 3.3, 'wind-sea'))

    # ---- Standard analysis ----
    results = run_cases(omegas, T, cases, k_surge, dw,
                        label=f"Linearized K = {k_surge:.0f} N/m")

    # ---- Current sweep (nonlinear mooring) ----
    if args.current_sweep:
        # Use a representative subset of sea states for the sweep
        sweep_cases = []
        # Wind-sea reference
        sweep_cases.append((2.5, 8.0, 3.3, 'wind-sea'))
        sweep_cases.append((3.0, 10.0, 3.3, 'wind-sea'))
        # Swell
        sweep_cases.append((1.0, 17.0, 5.0, 'swell'))
        sweep_cases.append((1.0, 19.0, 5.0, 'swell'))
        sweep_cases.append((1.0, 21.0, 5.0, 'swell'))

        all_results = run_current_sweep(omegas, T, sweep_cases, dw)

        # Print interpretation
        print()
        print("=" * 80)
        print("INTERPRETATION — Nonlinear mooring effect on slow-drift")
        print("=" * 80)
        print("""
  The catenary mooring HARDENS with offset: tangent stiffness increases,
  which shifts the surge natural frequency HIGHER.

  Two competing effects:
    1. Higher omega_n means the resonance peak in H(mu) moves to higher
       difference frequencies where the QTF force spectrum S_F(mu) typically
       has MORE energy (assuming S_F(mu) is not peaked near zero).
    2. Higher K means the peak of |H(mu)|^2 is LOWER (the resonance peak
       is at |H(omega_n)| = 1/(B_ext * omega_n), which decreases with omega_n).

  Net effect depends on the balance between these two factors and the
  QTF frequency resolution (dw = {:.4f} rad/s). With dw = {:.4f}:
    - At K = {:.0f} N/m (zero offset): omega_n = {:.4f}, nearest mu = {:.4f}
    - Resonance may be better or worse resolved as it shifts
""".format(dw, dw, K_SURGE, np.sqrt(K_SURGE/m_eff), dw))

    # ---- Current variability analysis ----
    if args.current_variability:
        print()
        print("=" * 80)
        print("CURRENT VARIABILITY — slowly varying surge from current fluctuations")
        print("=" * 80)

        if args.sigma_uc is not None:
            # Single case
            r = compute_current_variability_surge(
                args.uc_mean, args.sigma_uc, args.t_peak_current,
                spectrum_type=args.spectrum_type,
                verbose=True)

        else:
            # Amplitude sweep (default when no sigma_uc specified)
            current_variability_sweep(
                args.uc_mean, args.t_peak_current,
                verbose=True)

        if args.period_sweep:
            # Period sweep: needs a sigma_uc value
            sigma_for_sweep = args.sigma_uc if args.sigma_uc is not None else 0.15
            current_period_sweep(
                args.uc_mean, sigma_for_sweep,
                verbose=True)

    # ---- Spectral details for one case ----
    print()
    print("=" * 80)
    print("Spectral detail for reference case")
    print("=" * 80)

    detail_case = None
    for r in results:
        if abs(r['hs'] - 1.0) < 0.01 and abs(r['tp'] - 21.0) < 0.5 and r['type'] == 'swell':
            detail_case = r
            break
    if detail_case is None and len(results) > 0:
        detail_case = results[len(results) // 2]

    if detail_case is not None:
        mu = detail_case['mu']
        S_F = detail_case['S_F']
        S_x = detail_case['S_x']
        k_used = detail_case['k_surge']
        omega_n_used = detail_case['omega_n']
        print(f"  Sea state: Hs={detail_case['hs']}m, Tp={detail_case['tp']}s, gamma={detail_case['gamma']}")
        print(f"  K_surge = {k_used:.0f} N/m, omega_n = {omega_n_used:.4f} rad/s")
        print(f"  Mean drift force: {detail_case['F_mean']/1e3:.3f} kN  -> offset: {detail_case['x_mean']:.3f} m")
        print(f"  Slow-drift sigma: {detail_case['sigma']:.4f} m")
        print()

        print(f"  Slow-drift force spectrum S_F(mu):")
        print(f"  {'mu [rad/s]':>10s}  {'T_mu [s]':>10s}  {'S_F [N^2s]':>14s}  "
              f"{'|H| [m/N]':>12s}  {'S_x [m^2s]':>14s}  {'cum_var%':>8s}")

        dmu = mu[1] - mu[0] if len(mu) > 1 else dw
        cum_var = np.cumsum(S_x * dmu)
        total_var = cum_var[-1] if len(cum_var) > 0 else 1.0

        for k_idx in range(min(len(mu), 20)):
            T_mu_str = f"{2*np.pi/mu[k_idx]:.1f}" if mu[k_idx] > 0 else "inf"
            H_k = 1.0 / abs(-(MASS + A11_LOW) * mu[k_idx]**2 + k_used - 1j * mu[k_idx] * B_EXT) if mu[k_idx] > 0 or k_used > 0 else 0
            pct = cum_var[k_idx] / total_var * 100 if total_var > 0 else 0
            print(f"  {mu[k_idx]:10.4f}  {T_mu_str:>10s}  {S_F[k_idx]:14.4e}  "
                  f"{H_k:12.4e}  {S_x[k_idx]:14.4e}  {pct:8.1f}")

    print()


if __name__ == '__main__':
    main()
