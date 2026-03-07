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
