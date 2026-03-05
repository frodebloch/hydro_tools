#!/usr/bin/env python3
"""
bisym_foil.py — Bisymmetric airfoil section generator with tunable LE radius.

Generates symmetric airfoil sections optimised for tunnel thruster blades
that see flow from both directions.  All profiles are:

    * symmetric about the chord line  (zero camber)
    * symmetric about x/c = 0.5       (identical LE and TE shapes)
    * G2-continuous everywhere         (position, slope, curvature)

The only permitted fore-aft asymmetry is an optional trailing-edge gap for
XFOIL panel analysis (default y_TE = 0; use XFOIL GDES->TGAP if needed).

Three parameterisation methods are provided, each allowing independent
control of:

    * t/c   — maximum thickness-to-chord ratio
    * r_LE  — leading-edge radius (normalised by chord)
    * y_te  — trailing-edge gap (default 0; purely for XFOIL numerics)
    * n     — number of points per side

Output is a Selig-format .dat file directly loadable by XFOIL ("LOAD").

Methods
-------
1. modified_naca   NACA-style sqrt(x) nose polynomial defined on [0, 0.5],
                   mirrored to [0.5, 1].  The polynomial coefficients are
                   chosen to satisfy LE radius, max thickness at x/c=0.5,
                   and G2 continuity at the join (dy/dx=0, d²y/dx²
                   continuous).  The a0 coefficient controls LE radius:
                       r_LE = 0.5 * (t/0.20)² * a0²

2. elliptic_blend  Generalised superellipse (Lamé curve) defined by
                   |2(x-0.5)|^p + |2y/tc|^2 = 1.  The exponent p is set
                   from r_LE: p = 4*r_LE/tc².  For p=2 the profile is a
                   standard ellipse.  For p>2 the shape becomes squarer
                   (fuller body, beneficial for cavitation delay).

3. cst             Class-Shape Transformation (Kulfan) with symmetric class
                   function exponents N1=N2=0.5 giving C(x)=sqrt(x(1-x)),
                   and palindromic Bernstein coefficients (A_i = A_{n-i}).
                   Analytically symmetric and C-infinity smooth everywhere.

Usage
-----
    python bisym_foil.py --method modified_naca --tc 0.06 --rle 0.010 -o thruster06.dat
    python bisym_foil.py --method elliptic_blend --tc 0.10 --rle 0.020
    python bisym_foil.py --method cst --tc 0.08 --rle 0.015 --cst-order 6
    python bisym_foil.py --compare --tc 0.06 --rle 0.010
    python bisym_foil.py --method modified_naca --tc 0.06 --rle-sweep 0.005,0.010,0.015,0.020

Author: xfoil_cav project
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import comb as binom_coeff

# Default trailing-edge gap / chord.  0 = sharp TE (symmetric).
# Use XFOIL GDES->TGAP to add a gap for panel analysis if desired.
DEFAULT_YTE = 0.0


# ---------------------------------------------------------------------------
#  Cosine point distribution — bunches points at LE and TE
# ---------------------------------------------------------------------------

def cosine_spacing(n: int) -> NDArray:
    """Return *n* x-coordinates in [0, 1] with half-cosine bunching.

    Dense near LE (x=0) and TE (x=1), which is what XFOIL expects for good
    panel resolution of the suction peak and the trailing edge.
    """
    beta = np.linspace(0.0, np.pi, n)
    return 0.5 * (1.0 - np.cos(beta))


# ===================================================================
#  Method 1 — Modified NACA 00xx, mirrored for fore-aft symmetry
# ===================================================================

def _naca_rle_from_a0(a0: float, tc: float) -> float:
    """LE radius of a NACA-style thickness polynomial given a0 and t/c.

    For the polynomial  y = (t/0.20)[a0*sqrt(x) + …],
    the curvature at x->0 gives  r_LE = 0.5 * (t/0.20)² * a0².
    """
    return 0.5 * (tc / 0.20) ** 2 * a0 ** 2


def _naca_a0_from_rle(rle: float, tc: float) -> float:
    """Inverse of _naca_rle_from_a0: solve for a0 given desired LE radius."""
    return np.sqrt(2.0 * rle) / (tc / 0.20)


def modified_naca(tc: float, rle: float, n: int = 161,
                  y_te: float = DEFAULT_YTE) -> tuple[NDArray, NDArray, dict]:
    """Generate a modified NACA 00xx section, symmetric about x/c = 0.5.

    The half-thickness is defined on [0, 0.5] using a NACA-style polynomial:

        y(xi) = scale * [a0*sqrt(xi) + c1*xi + c2*xi² + c3*xi³ + c4*xi⁴]

    where xi is in [0, 0.5] and scale = t/0.20.  The coefficients satisfy:

        1. y(0.5) = tc/2          — max thickness at midchord
        2. y'(0.5) = 0            — zero slope at midchord (true maximum)
        3. y''(0.5) = prescribed  — midchord curvature (ellipse-like default)
        4. y'''(0.5) = 0          — G2 continuity when mirrored

    The polynomial is then mirrored: y(x) = y(1-x) for x in [0.5, 1].
    This guarantees exact fore-aft symmetry with G2 continuity at the join.

    The midchord curvature (constraint 3) is set to match that of an
    ellipse with the same t/c: d²y/dx² = -2*tc at x=0.5.  This ensures
    a natural, smooth peak and monotonically increasing thickness on
    [0, 0.5].

    Parameters
    ----------
    tc : float
        Maximum thickness-to-chord ratio (e.g. 0.06 for 6%).
    rle : float
        Desired leading-edge radius / chord.
    n : int
        Number of points per side (total airfoil will have 2n-1 points).
    y_te : float
        Trailing-edge total gap / chord (default 0).

    Returns
    -------
    x, y : ndarray
        Full airfoil coordinates in Selig order (upper TE -> LE -> lower TE).
    info : dict
        Diagnostic information (coefficients, actual r_LE, etc.).
    """
    a0 = _naca_a0_from_rle(rle, tc)
    yt_max = tc / 2.0          # half-thickness target at x=0.5
    scale = tc / 0.20           # overall multiplier

    # Polynomial on [0, 0.5]:
    #   y(xi)/scale = a0*xi^0.5 + c1*xi + c2*xi^2 + c3*xi^3 + c4*xi^4
    #
    # Four constraints -> 4x4 linear system for (c1, c2, c3, c4):
    #   1) y(0.5) = yt_max              — max thickness value
    #   2) y'(0.5) = 0                  — zero slope (true maximum)
    #   3) y''(0.5) = d2y_mid           — midchord curvature
    #   4) y'''(0.5) = 0                — G2 continuity when mirrored
    #
    # For the mirrored profile y_R(x) = y(1-x), at x=0.5 (xi=0.5):
    #   y_R'  = -y'   -> both zero: OK
    #   y_R'' = +y''   -> automatically matched
    #   y_R'''= -y''' -> continuous only if y'''(0.5) = 0

    h = 0.5  # half-chord

    # Midchord curvature of an ellipse with semi-axes a=0.5, b=tc/2:
    #   y = b*sqrt(1 - ((x-a)/a)^2),  y'' at x=a is -b/a^2 = -2*tc
    d2y_mid = -2.0 * tc

    # sqrt term derivatives at xi:
    #   d/dx[a0*xi^0.5]     = a0 * 0.5 * xi^(-0.5)
    #   d²/dx²[a0*xi^0.5]   = a0 * (-0.25) * xi^(-1.5)
    #   d³/dx³[a0*xi^0.5]   = a0 * (3/8) * xi^(-2.5)

    # Constraint 1: value at xi=0.5
    A_val = np.array([h, h**2, h**3, h**4])
    rhs_val = yt_max / scale - a0 * np.sqrt(h)

    # Constraint 2: dy/dx = 0 at xi=0.5
    A_der = np.array([1.0, 2*h, 3*h**2, 4*h**3])
    rhs_der = -a0 * 0.5 * h**(-0.5)

    # Constraint 3: d²y/dx² = d2y_mid at xi=0.5
    A_d2 = np.array([0.0, 2.0, 6*h, 12*h**2])
    rhs_d2 = d2y_mid / scale - a0 * (-0.25) * h**(-1.5)

    # Constraint 4: d³y/dx³ = 0 at xi=0.5
    A_d3 = np.array([0.0, 0.0, 6.0, 24*h])
    rhs_d3 = -a0 * (3.0 / 8.0) * h**(-2.5)

    A = np.vstack([A_val, A_der, A_d2, A_d3])
    rhs = np.array([rhs_val, rhs_der, rhs_d2, rhs_d3])

    c = np.linalg.solve(A, rhs)
    c1, c2, c3, c4 = c

    # Generate coordinates on [0, 1]
    xx = cosine_spacing(n)
    xx[0] = 0.0
    xx[-1] = 1.0

    yt = np.zeros(n)
    for i in range(n):
        xi = xx[i]
        # Map to [0, 0.5]: use xi directly if <= 0.5, else use (1-xi)
        xi_half = xi if xi <= 0.5 else 1.0 - xi
        if xi_half > 0:
            yt[i] = scale * (a0 * np.sqrt(xi_half)
                             + c1 * xi_half
                             + c2 * xi_half**2
                             + c3 * xi_half**3
                             + c4 * xi_half**4)
        else:
            yt[i] = 0.0

    # Ensure exact zero at LE and TE
    yt[0] = 0.0
    yt[-1] = 0.0

    # Add TE gap ramp (linear, breaks symmetry only at TE)
    yte_half = y_te / 2.0
    if yte_half > 0:
        yt += xx * yte_half

    actual_tc = 2.0 * np.max(yt)
    actual_rle = _naca_rle_from_a0(a0, tc)
    geom_rle = _compute_le_radius(xx, yt)

    x_full, y_full = _assemble_selig(xx, yt)

    info = {
        'method': 'modified_naca',
        'tc_target': tc,
        'tc_actual': actual_tc,
        'rle_target': rle,
        'rle_formula': actual_rle,
        'rle_geometric': geom_rle,
        'yte': y_te,
        'yte_actual': 2.0 * yt[-1],
        'a0': a0,
        'coeffs': (c1, c2, c3, c4),
        'n_points': len(x_full),
        'a0_standard_naca': 0.2969,
        'rle_standard_naca': _naca_rle_from_a0(0.2969, tc),
    }

    return x_full, y_full, info


# ===================================================================
#  Method 2 — Generalised superellipse with tunable LE radius
# ===================================================================

def elliptic_blend(tc: float, rle: float, n: int = 161,
                   y_te: float = DEFAULT_YTE,
                   doc: float = 0.5) -> tuple[NDArray, NDArray, dict]:
    """Generate a section using a generalised superellipse (Lamé curve).

    The half-thickness is defined by:

        |2(x - 0.5)|^p  +  |2y / tc|^2  =  1

    giving:

        y(x) = (tc/2) * sqrt(1 - |1 - 2x|^p)

    The profile is inherently symmetric about x/c = 0.5 with:

        * Max thickness tc at midchord (exact).
        * LE radius  r_LE = tc² p / 4  (exact, leading order).
        * G2 continuity everywhere for p > 1 (C-infinity smooth).
        * Monotonically increasing thickness on [0, 0.5].

    The exponent *p* is determined from the desired LE radius:

        p = 4 r_LE / tc²

    For p = 2 the profile is a standard ellipse.  For p > 2 (r_LE larger
    than the ellipse value tc²/2) the shape becomes 'squarer' — fuller
    over the middle portion of the chord, which can be beneficial for
    delaying cavitation inception in tunnel thrusters.

    The ``doc`` parameter is accepted for CLI compatibility but unused;
    the profile shape is fully determined by tc and rle.

    Parameters
    ----------
    tc : float
        Maximum thickness-to-chord ratio.
    rle : float
        Desired leading-edge radius / chord.
        Minimum sensible value: tc²/4 (p = 1 limit; below this the
        midchord develops a slope discontinuity).
    n : int
        Number of points per side.
    y_te : float
        Trailing-edge total gap / chord (default 0).
    doc : float
        Unused (retained for CLI compatibility).

    Returns
    -------
    x, y : ndarray
        Full airfoil coordinates in Selig order.
    info : dict
        Diagnostic information.
    """
    b = tc / 2.0   # half-thickness at midchord

    # Superellipse exponent from desired LE radius
    p = 4.0 * rle / tc**2

    # Validity check: p must be > 1 for a smooth midchord
    if p <= 1.0:
        import warnings
        warnings.warn(
            f"elliptic_blend: r_LE = {rle:.6f} gives p = {p:.3f} <= 1.  "
            f"Minimum r_LE for smooth profile is tc²/4 = {tc**2/4:.6f}.  "
            f"Clamping p to 1.01.",
            stacklevel=2)
        p = 1.01

    # Ellipse LE radius for reference: r_ell = b²/a = tc²/2
    r_ellipse = tc**2 / 2.0

    # Generate coordinates
    xx = cosine_spacing(n)
    xx[0] = 0.0
    xx[-1] = 1.0

    yt = np.zeros(n)
    for i in range(n):
        u = abs(1.0 - 2.0 * xx[i])          # distance from midchord [0,1]
        arg = 1.0 - u**p
        if arg > 0:
            yt[i] = b * np.sqrt(arg)
        else:
            yt[i] = 0.0

    # Exact zeros at endpoints (numerical insurance)
    yt[0] = 0.0
    yt[-1] = 0.0

    # Add TE gap ramp if requested
    yte_half = y_te / 2.0
    if yte_half > 0:
        yt += xx * yte_half

    actual_tc = 2.0 * np.max(yt)
    geom_rle = _compute_le_radius(xx, yt)

    x_full, y_full = _assemble_selig(xx, yt)

    info = {
        'method': 'elliptic_blend',
        'tc_target': tc,
        'tc_actual': actual_tc,
        'rle_target': rle,
        'rle_formula': tc**2 * p / 4.0,
        'rle_ellipse': r_ellipse,
        'rle_geometric': geom_rle,
        'yte': y_te,
        'yte_actual': 2.0 * yt[-1],
        'superellipse_p': p,
        'n_points': len(x_full),
    }

    return x_full, y_full, info


# ===================================================================
#  Method 3 — CST (Kulfan) with symmetric class function and coeffs
# ===================================================================

def cst(tc: float, rle: float, n: int = 161,
        order: int = 6,
        y_te: float = DEFAULT_YTE) -> tuple[NDArray, NDArray, dict]:
    """Generate a symmetric section using CST (Kulfan) parameterisation.

    Uses symmetric class function exponents N1 = N2 = 0.5, giving:

        C(x) = sqrt(x * (1 - x))

    which is symmetric about x/c = 0.5.  The Bernstein coefficients are
    constrained to be palindromic (A_i = A_{n-i}), ensuring the shape
    function S(x) is also symmetric about midchord.

    The LE radius appears through A0: since near x->0,
        y ~ t_scale * A0 * sqrt(x)
    we get  r_LE = 0.5 * (t_scale * A0)².

    If the optimizer fails to achieve t/c within 1% at the requested order,
    the order is automatically increased (up to 16) until convergence.

    Parameters
    ----------
    tc : float
        Maximum thickness-to-chord ratio.
    rle : float
        Desired leading-edge radius / chord.
    n : int
        Number of points per side.
    order : int
        Initial Bernstein polynomial order (must be even; auto-increased
        if needed for convergence).
    y_te : float
        Trailing-edge total gap / chord (default 0).

    Returns
    -------
    x, y : ndarray
        Full airfoil coordinates in Selig order.
    info : dict
        Diagnostic information (Bernstein coefficients, etc.).
    """
    # Force even order for palindromic symmetry
    if order % 2 != 0:
        order += 1

    yte_half = y_te / 2.0

    N1, N2 = 0.5, 0.5   # symmetric class function exponents

    def class_fn(x):
        return np.where((x > 0) & (x < 1),
                        x**N1 * (1.0 - x)**N2, 0.0)

    def bernstein(x, i, n_ord):
        return binom_coeff(n_ord, i, exact=True) * x**i * (1.0 - x)**(n_ord - i)

    def shape_fn(x, A_full):
        S = np.zeros_like(x)
        n_ord = len(A_full) - 1
        for i, ai in enumerate(A_full):
            S += ai * bernstein(x, i, n_ord)
        return S

    def expand_palindromic(A_half):
        """Expand half-coefficients to full palindromic vector."""
        n_half = len(A_half)
        n_ord = (n_half - 1) * 2
        A_full = np.zeros(n_ord + 1)
        for i in range(n_half):
            A_full[i] = A_half[i]
            A_full[n_ord - i] = A_half[i]
        return A_full

    def yt_from_A(x, A_full, t_scale):
        return class_fn(x) * shape_fn(x, A_full) * t_scale

    # Optimisation grid (denser than output for accurate penalty evaluation)
    xx_opt = cosine_spacing(201)
    xx_opt[0] = 1e-10  # avoid 0^0 issues

    def _solve_cst(order_):
        """Run CST optimiser at a given Bernstein order.  Returns result dict."""
        n_indep_ = order_ // 2 + 1
        n_free_ = n_indep_ - 1

        def objective(params):
            t_s = params[-1]
            A_half = np.empty(n_indep_)
            A_half[0] = np.sqrt(2.0 * rle) / t_s
            A_half[1:] = params[:n_free_]
            A_full = expand_palindromic(A_half)

            yt = yt_from_A(xx_opt, A_full, t_s)

            i_mid = len(xx_opt) // 2
            midchord_err = (yt[i_mid] - tc / 2.0)**2 * 1e6

            i_half = len(xx_opt) // 2 + 1
            dyt_left = np.diff(yt[:i_half])
            mono_penalty = np.sum(np.minimum(dyt_left, 0.0)**2) * 1e12

            S = shape_fn(xx_opt, A_full)
            dS = np.diff(S)
            d2S = np.diff(dS)
            smooth_penalty = np.sum(d2S**2) * 1e2

            neg_penalty = np.sum(np.minimum(yt, 0.0)**2) * 1e12

            return midchord_err + mono_penalty + smooth_penalty + neg_penalty

        # Initial guess
        t_scale_init = tc / 2.0
        A0_init = np.sqrt(2.0 * rle) / t_scale_init
        p0 = np.ones(n_free_ + 1)
        p0[:n_free_] = max(A0_init * 0.5, 0.5)
        p0[-1] = t_scale_init

        A_ub = max(A0_init * 3.0, 5.0)
        bounds = [(0.01, A_ub)] * n_free_ + [(tc / 8.0, tc * 2.0)]

        best_result = None
        best_cost = float('inf')

        for trial in range(3):
            if trial == 0:
                p_start = p0.copy()
            else:
                rng = np.random.RandomState(42 + trial)
                p_start = p0.copy()
                p_start[:n_free_] *= rng.uniform(0.5, 1.5, n_free_)
                p_start[-1] *= rng.uniform(0.8, 1.2)
                for k in range(len(p_start)):
                    p_start[k] = np.clip(p_start[k], bounds[k][0], bounds[k][1])

            res = minimize(objective, p_start, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-15})
            if res.fun < best_cost:
                best_cost = res.fun
                best_result = res

        t_scale = best_result.x[-1]
        A_half = np.empty(n_indep_)
        A_half[0] = np.sqrt(2.0 * rle) / t_scale
        A_half[1:] = best_result.x[:n_free_]
        A_full = expand_palindromic(A_half)

        # Evaluate midchord thickness for convergence check
        i_mid = len(xx_opt) // 2
        yt_check = yt_from_A(xx_opt, A_full, t_scale)
        tc_actual_check = 2.0 * yt_check[i_mid]

        return {
            'result': best_result,
            'A_half': A_half,
            'A_full': A_full,
            't_scale': t_scale,
            'order': order_,
            'tc_err': abs(tc_actual_check - tc) / tc,
        }

    # Try requested order; auto-increase if tc error > 1%
    max_order = 16
    sol = _solve_cst(order)
    while sol['tc_err'] > 0.01 and sol['order'] < max_order:
        order_new = sol['order'] + 2
        sol = _solve_cst(order_new)
    order = sol['order']
    result = sol['result']
    t_scale = sol['t_scale']
    A_half = sol['A_half']
    A_full = sol['A_full']

    # Generate final coordinates
    xx = cosine_spacing(n)
    xx[0] = 0.0
    xx[-1] = 1.0

    yt = np.zeros(n)
    for i in range(n):
        xi = xx[i]
        if xi > 0 and xi < 1:
            yt[i] = (xi**N1 * (1.0 - xi)**N2 *
                     sum(A_full[j] * bernstein(xi, j, order)
                         for j in range(order + 1)) *
                     t_scale)
        else:
            yt[i] = 0.0

    # Add TE gap ramp if requested
    if yte_half > 0:
        yt += xx * yte_half

    actual_tc = 2.0 * np.max(yt)
    geom_rle = _compute_le_radius(xx, yt)

    x_full, y_full = _assemble_selig(xx, yt)

    info = {
        'method': 'cst',
        'tc_target': tc,
        'tc_actual': actual_tc,
        'rle_target': rle,
        'rle_geometric': geom_rle,
        'yte': y_te,
        'yte_actual': 2.0 * yt[-1],
        'A_coefficients': A_full.tolist(),
        'A_half_independent': A_half.tolist(),
        't_scale': t_scale,
        'bernstein_order': order,
        'n_points': len(x_full),
        'optimiser_success': result.success,
        'optimiser_message': result.message,
    }

    return x_full, y_full, info


# ===================================================================
#  Utilities
# ===================================================================

def _assemble_selig(xx: NDArray, yt: NDArray) -> tuple[NDArray, NDArray]:
    """Assemble half-thickness arrays into full Selig-format coordinates.

    Selig order: upper surface from TE (x=1) to LE (x=0),
                 then lower surface from LE (x=0) to TE (x=1).
    For a symmetric section, y_upper = +yt, y_lower = -yt.
    """
    # xx goes 0->1, yt is half-thickness (>= 0)
    # Upper surface: reverse order (TE->LE), y positive
    x_upper = xx[::-1].copy()
    y_upper = yt[::-1].copy()

    # Lower surface: skip LE (already included), LE->TE, y negative
    x_lower = xx[1:].copy()
    y_lower = -yt[1:].copy()

    x_full = np.concatenate([x_upper, x_lower])
    y_full = np.concatenate([y_upper, y_lower])

    return x_full, y_full


def _compute_le_radius(xx: NDArray, yt: NDArray) -> float:
    """Compute LE radius from generated geometry by extrapolating y²/(2x) to x→0.

    For any smooth-nosed profile, y ~ sqrt(2*r_LE*x) as x→0, so
    y²/(2x) → r_LE.  Higher-order terms (linear in sqrt(x) for NACA-type,
    linear in x for CST) cause y²/(2x) to drift away from r_LE at finite x.

    We fit y²/(2x) = r_LE + b*sqrt(x) using the first few points and
    extrapolate to x=0.  This gives much better accuracy than the simpler
    forced-origin regression of y² on x.

    For profiles with a finite TE gap, the linear ramp x*y_te/2 is
    subtracted before fitting so that only the leading-edge shape
    contributes to the radius estimate.
    """
    # Estimate and subtract the TE-gap ramp:  yt ~ yt_shape + x * yte_half
    yte_half = yt[-1] if xx[-1] >= 0.999 else 0.0
    yt_shape = yt - xx * yte_half

    # Use points where 0 < x < 0.01
    mask = (xx > 0) & (xx < 0.01)
    if np.sum(mask) < 3:
        mask = np.zeros_like(xx, dtype=bool)
        idx = np.where(xx > 0)[0][:5]
        mask[idx] = True

    x_fit = xx[mask]
    y_fit = yt_shape[mask]

    if len(x_fit) < 2 or np.all(y_fit == 0):
        return 0.0

    # Compute y²/(2x) at each point
    r_local = y_fit**2 / (2.0 * x_fit)

    # Fit r_local = a + b*sqrt(x) and extrapolate to x=0 (intercept a = r_LE)
    sqx = np.sqrt(x_fit)
    # Linear regression: r_local = a + b * sqx
    n = len(sqx)
    sx = np.sum(sqx)
    sy = np.sum(r_local)
    sxx = np.sum(sqx**2)
    sxy = np.sum(sqx * r_local)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-30:
        return np.mean(r_local)

    rle = (sxx * sy - sx * sxy) / denom  # intercept

    return max(rle, 0.0)


def _compute_le_radius_curvature(xx: NDArray, yt: NDArray) -> float:
    """Compute LE radius via parametric curvature of the upper surface near LE.

    Independent verification method.
    """
    n_pts = min(10, len(xx))
    x = xx[:n_pts]
    y = yt[:n_pts]

    if len(x) < 3:
        return 0.0

    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.concatenate([[0.0], np.cumsum(ds)])

    if s[-1] == 0:
        return 0.0

    dx_ds = np.gradient(x, s)
    dy_ds = np.gradient(y, s)
    d2x_ds2 = np.gradient(dx_ds, s)
    d2y_ds2 = np.gradient(dy_ds, s)

    kappa = np.abs(dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / (
        dx_ds**2 + dy_ds**2)**1.5

    kappa_le = kappa[1] if len(kappa) > 1 else kappa[0]

    if kappa_le > 0:
        return 1.0 / kappa_le
    return 0.0


def _check_symmetry(xx: NDArray, yt: NDArray, y_te: float = 0.0,
                    tol: float = 1e-10) -> dict:
    """Verify fore-aft symmetry and G2 continuity at x/c = 0.5.

    Returns a dict with symmetry errors and curvature continuity info.
    """
    # Remove TE gap ramp for symmetry check
    yte_half = y_te / 2.0
    yt_shape = yt - xx * yte_half

    # Check y(x) == y(1-x) at each point
    n = len(xx)
    max_sym_err = 0.0
    for i in range(n):
        x_mirror = 1.0 - xx[i]
        # Find nearest point to x_mirror
        j = np.argmin(np.abs(xx - x_mirror))
        if abs(xx[j] - x_mirror) < 1e-12:
            err = abs(yt_shape[i] - yt_shape[j])
            max_sym_err = max(max_sym_err, err)

    # Curvature near x/c = 0.5
    mask = np.abs(xx - 0.5) < 0.05
    if np.sum(mask) > 4:
        x_mid = xx[mask]
        y_mid = yt_shape[mask]
        # Numerical second derivative
        dx = np.diff(x_mid)
        dy = np.diff(y_mid)
        d2y = np.diff(dy) / np.diff(0.5 * (x_mid[:-1] + x_mid[1:]))
        curvature_jump = abs(d2y[len(d2y)//2] - d2y[len(d2y)//2 - 1]) if len(d2y) > 1 else 0.0
    else:
        curvature_jump = float('nan')

    return {
        'max_symmetry_error': max_sym_err,
        'symmetric': max_sym_err < tol,
        'curvature_jump_at_midchord': curvature_jump,
        'g2_continuous': curvature_jump < 1e-4 if not np.isnan(curvature_jump) else False,
    }


# ===================================================================
#  XFOIL .dat file writer
# ===================================================================

def write_dat(filepath: str | Path, x: NDArray, y: NDArray,
              name: str = "Bisym foil") -> None:
    """Write airfoil coordinates in Selig .dat format for XFOIL."""
    filepath = Path(filepath)
    with open(filepath, 'w') as f:
        f.write(f"{name}\n")
        for xi, yi in zip(x, y):
            f.write(f"  {xi:11.7f}  {yi:11.7f}\n")


# ===================================================================
#  XFOIL batch runner
# ===================================================================

def _find_xfoil_binary(xfoil_bin: str | None = None) -> Path:
    """Locate the XFOIL executable.

    Search order:
        1. Explicit *xfoil_bin* path (if given).
        2. ``bin/xfoil`` relative to this script.
        3. ``xfoil`` on the system PATH.

    Raises FileNotFoundError if nothing is found.
    """
    if xfoil_bin is not None:
        p = Path(xfoil_bin)
        if p.is_file():
            return p
        raise FileNotFoundError(f"XFOIL binary not found at {xfoil_bin}")

    # Relative to this script
    script_dir = Path(__file__).resolve().parent
    p = script_dir / "bin" / "xfoil"
    if p.is_file():
        return p

    # System PATH
    found = shutil.which("xfoil")
    if found:
        return Path(found)

    raise FileNotFoundError(
        "XFOIL binary not found.  Provide --xfoil-bin or place xfoil in "
        f"{script_dir / 'bin'} or on PATH.")


def _run_xfoil(commands: str, xfoil_bin: str | Path | None = None,
               timeout: float = 60) -> str:
    """Run XFOIL with piped commands and return stdout.

    Parameters
    ----------
    commands : str
        Newline-separated XFOIL commands (must end with QUIT).
    xfoil_bin : str or Path, optional
        Path to the XFOIL executable.
    timeout : float
        Maximum wall-clock seconds before the process is killed.

    Returns
    -------
    str
        XFOIL stdout.
    """
    xfoil = _find_xfoil_binary(xfoil_bin)
    proc = subprocess.run(
        [str(xfoil)],
        input=commands,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.stdout


def xfoil_cp(x: NDArray, y: NDArray, alpha: float = 0.0,
             xfoil_bin: str | None = None,
             timeout: float = 30) -> tuple[NDArray, NDArray]:
    """Run XFOIL inviscid analysis and return Cp(x) at a given alpha.

    Parameters
    ----------
    x, y : ndarray
        Airfoil coordinates in Selig format.
    alpha : float
        Angle of attack in degrees.
    xfoil_bin : str, optional
        Path to XFOIL binary.

    Returns
    -------
    x_cp, cp : ndarray
        Cp distribution (Selig order: upper TE→LE, lower LE→TE).
    """
    with tempfile.TemporaryDirectory(prefix="bisym_xfoil_") as tmpdir:
        dat_path = Path(tmpdir) / "foil.dat"
        cp_path = Path(tmpdir) / "cp.dat"

        write_dat(dat_path, x, y, name="bisym_foil")

        commands = f"""\
LOAD {dat_path}
PANE
OPER
ALFA {alpha:.4f}
CPWR {cp_path}

QUIT
"""
        _run_xfoil(commands, xfoil_bin=xfoil_bin, timeout=timeout)

        if not cp_path.exists():
            raise RuntimeError(
                f"XFOIL did not write Cp file at alpha={alpha:.2f}")

        data = np.loadtxt(cp_path, skiprows=1)
        return data[:, 0], data[:, 1]


def xfoil_cavitation_bucket(
        x: NDArray, y: NDArray,
        alpha_start: float = 0.0,
        alpha_end: float = 6.0,
        alpha_step: float = 0.25,
        xfoil_bin: str | None = None,
        timeout: float = 120) -> tuple[NDArray, NDArray, NDArray]:
    """Compute inviscid cavitation bucket: sigma_i(alpha) = -Cp_min(alpha).

    Uses a single XFOIL session, sweeping from *alpha_start* upward in
    small steps (then mirroring for negative alpha by symmetry).

    Parameters
    ----------
    x, y : ndarray
        Airfoil coordinates in Selig format.
    alpha_start, alpha_end, alpha_step : float
        Sweep range in degrees.  Negative alphas are obtained by symmetry
        (Cp_min at -alpha equals Cp_min at +alpha for a symmetric section
        at the mirrored surface).
    xfoil_bin : str, optional
        Path to XFOIL binary.

    Returns
    -------
    alphas : ndarray
        Angles of attack (includes negative mirror).
    sigma_i : ndarray
        Cavitation inception number sigma_i = -Cp_min at each alpha.
    cp_min : ndarray
        Minimum Cp at each alpha.
    """
    alphas_pos = np.arange(alpha_start, alpha_end + alpha_step * 0.5,
                           alpha_step)

    with tempfile.TemporaryDirectory(prefix="bisym_xfoil_") as tmpdir:
        dat_path = Path(tmpdir) / "foil.dat"
        write_dat(dat_path, x, y, name="bisym_foil")

        # Build command string: one ALFA + CPWR per alpha
        cmd_lines = [
            f"LOAD {dat_path}",
            "PANE",
            "OPER",
        ]
        cp_paths = {}
        for i, a in enumerate(alphas_pos):
            cp_file = Path(tmpdir) / f"cp_{i:03d}.dat"
            cp_paths[i] = cp_file
            cmd_lines.append(f"ALFA {a:.4f}")
            cmd_lines.append(f"CPWR {cp_file}")

        cmd_lines.append("")
        cmd_lines.append("QUIT")
        commands = "\n".join(cmd_lines) + "\n"

        _run_xfoil(commands, xfoil_bin=xfoil_bin, timeout=timeout)

        # Parse Cp files, extract min Cp
        alphas_out = []
        cpmin_out = []
        for i, a in enumerate(alphas_pos):
            if cp_paths[i].exists():
                data = np.loadtxt(cp_paths[i], skiprows=1)
                alphas_out.append(a)
                cpmin_out.append(np.min(data[:, 1]))

    alphas_out = np.array(alphas_out)
    cpmin_out = np.array(cpmin_out)

    # Mirror for negative alpha (symmetric section):
    # At -alpha, the suction/pressure sides swap, so Cp_min is the same
    # as at +alpha but on the opposite surface.  For the full bucket we
    # need both sides.  For a perfectly symmetric section the bucket is
    # symmetric, but at nonzero alpha the LE suction peak is stronger on
    # the suction side.  We already compute both surfaces in one CPWR, so
    # Cp_min captures the worst-case surface automatically.
    if alpha_start == 0.0 and len(alphas_out) > 0:
        neg_alpha = -alphas_out[1:][::-1]  # skip 0
        neg_cpmin = cpmin_out[1:][::-1]
        alphas_out = np.concatenate([neg_alpha, alphas_out])
        cpmin_out = np.concatenate([neg_cpmin, cpmin_out])

    sigma_i = -cpmin_out

    return alphas_out, sigma_i, cpmin_out


def xfoil_cavity(
        x: NDArray, y: NDArray,
        alpha: float, sigma: float,
        xfoil_bin: str | None = None,
        timeout: float = 30) -> dict:
    """Run XFOIL inviscid cavitation analysis at given alpha and sigma.

    Uses the CAVE command to enable cavitation, then solves at the given
    alpha and dumps cavity data via CDMP.

    Parameters
    ----------
    x, y : ndarray
        Airfoil coordinates in Selig format.
    alpha : float
        Angle of attack in degrees.
    sigma : float
        Cavitation number.
    xfoil_bin : str, optional
        Path to XFOIL binary.

    Returns
    -------
    dict with keys:
        'sigma', 'alpha' : operating point
        'has_cavity' : bool
        'x_cav', 'y_cav', 'h_cav', 'cp_cav' : ndarray (cavity stations)
        'cavity_length', 'max_thickness', 'cd_cav' : float
        'stdout' : raw XFOIL output for further parsing
    """
    with tempfile.TemporaryDirectory(prefix="bisym_xfoil_") as tmpdir:
        dat_path = Path(tmpdir) / "foil.dat"
        cav_path = Path(tmpdir) / "cav.dat"
        cp_path = Path(tmpdir) / "cp.dat"

        write_dat(dat_path, x, y, name="bisym_foil")

        commands = f"""\
LOAD {dat_path}
PANE
OPER
CAVE {sigma:.4f}
ALFA {alpha:.4f}
CDMP {cav_path}
CPWR {cp_path}

QUIT
"""
        stdout = _run_xfoil(commands, xfoil_bin=xfoil_bin, timeout=timeout)

        result = {
            'sigma': sigma,
            'alpha': alpha,
            'has_cavity': False,
            'x_cav': np.array([]),
            'y_cav': np.array([]),
            'h_cav': np.array([]),
            'cp_cav': np.array([]),
            'cavity_length': 0.0,
            'max_thickness': 0.0,
            'cd_cav': 0.0,
            'x_cp': np.array([]),
            'cp': np.array([]),
            'stdout': stdout,
        }

        # Parse Cp
        if cp_path.exists():
            data = np.loadtxt(cp_path, skiprows=1)
            result['x_cp'] = data[:, 0]
            result['cp'] = data[:, 1]

        # Parse cavity dump
        if cav_path.exists():
            try:
                data = np.loadtxt(cav_path, comments='#')
                if data.size > 0:
                    if data.ndim == 1:
                        data = data.reshape(1, -1)
                    result['has_cavity'] = True
                    result['x_cav'] = data[:, 0]
                    result['y_cav'] = data[:, 1]
                    result['h_cav'] = data[:, 2]
                    result['cp_cav'] = data[:, 3]
                    result['cavity_length'] = (data[-1, 0] - data[0, 0])
                    result['max_thickness'] = np.max(data[:, 2])
            except (ValueError, IndexError):
                pass

        # Parse summary from stdout
        for line in stdout.splitlines():
            if 'Total cavity CDcav' in line:
                try:
                    result['cd_cav'] = float(line.split('=')[1].strip())
                except (ValueError, IndexError):
                    pass

        return result


def xfoil_viscous_polar(
        x: NDArray, y: NDArray,
        re: float = 1.0e6,
        alpha_start: float = 0.0,
        alpha_end: float = 4.0,
        alpha_step: float = 0.25,
        iter_limit: int = 200,
        xfoil_bin: str | None = None,
        timeout: float = 120) -> dict:
    """Run XFOIL viscous polar sweep with small alpha steps.

    Starts from *alpha_start* and steps upward in small increments so the
    BL solver can use the previous converged solution as initial guess.
    Bisymmetric sections have blunt TEs that challenge the BL solver, so
    small steps and high iteration limits are essential.

    Parameters
    ----------
    x, y : ndarray
        Airfoil coordinates in Selig format (should have finite TE gap
        for viscous analysis).
    re : float
        Reynolds number.
    alpha_start, alpha_end, alpha_step : float
        Sweep range in degrees.
    iter_limit : int
        XFOIL VISCAL iteration limit per alpha point.
    xfoil_bin : str, optional
        Path to XFOIL binary.

    Returns
    -------
    dict with keys:
        'alpha', 'cl', 'cd', 'cdp', 'cm' : ndarray
        'top_xtr', 'bot_xtr' : ndarray (transition locations)
        'converged_count' : int
        'stdout' : raw XFOIL output
    """
    with tempfile.TemporaryDirectory(prefix="bisym_xfoil_") as tmpdir:
        dat_path = Path(tmpdir) / "foil.dat"
        pol_path = Path(tmpdir) / "polar.dat"

        write_dat(dat_path, x, y, name="bisym_foil")

        commands = f"""\
LOAD {dat_path}
PANE
OPER
VISC {re:.4E}
ITER {iter_limit}
PACC
{pol_path}

ASEQ {alpha_start:.4f} {alpha_end:.4f} {alpha_step:.4f}
PACC

QUIT
"""
        stdout = _run_xfoil(commands, xfoil_bin=xfoil_bin, timeout=timeout)

        result = {
            'alpha': np.array([]),
            'cl': np.array([]),
            'cd': np.array([]),
            'cdp': np.array([]),
            'cm': np.array([]),
            'top_xtr': np.array([]),
            'bot_xtr': np.array([]),
            'converged_count': 0,
            're': re,
            'stdout': stdout,
        }

        if pol_path.exists():
            try:
                data = np.loadtxt(pol_path, skiprows=12)
                if data.size > 0:
                    if data.ndim == 1:
                        data = data.reshape(1, -1)
                    result['alpha'] = data[:, 0]
                    result['cl'] = data[:, 1]
                    result['cd'] = data[:, 2]
                    result['cdp'] = data[:, 3]
                    result['cm'] = data[:, 4]
                    result['top_xtr'] = data[:, 5]
                    result['bot_xtr'] = data[:, 6]
                    result['converged_count'] = len(data)
            except (ValueError, IndexError):
                pass

        return result


# ===================================================================
#  Plotting / comparison
# ===================================================================

def plot_comparison(results: list[tuple[NDArray, NDArray, dict]],
                    savefile: str | None = None) -> None:
    """Plot airfoil shapes and thickness distributions for comparison."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # --- Airfoil shape ---
    ax = axes[0, 0]
    for x, y, info in results:
        label = info['method']
        ax.plot(x, y, label=label, linewidth=1.2)
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.set_title('Airfoil shape')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- LE zoom ---
    ax = axes[0, 1]
    for x, y, info in results:
        label = info['method']
        ax.plot(x, y, label=label, linewidth=1.2)
    # Plot target LE radius circle
    rle = results[0][2]['rle_target']
    theta = np.linspace(-np.pi, np.pi, 200)
    xc = rle + rle * np.cos(theta)
    yc = rle * np.sin(theta)
    ax.plot(xc, yc, 'k--', linewidth=0.8, alpha=0.5,
            label=f'target r_LE={rle:.4f}')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.set_title('LE detail')
    ax.set_aspect('equal')
    lim = max(4.0 * rle, 0.03)
    ax.set_xlim(-lim * 0.3, lim)
    ax.set_ylim(-lim * 0.7, lim * 0.7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Thickness distribution ---
    ax = axes[1, 0]
    for x, y, info in results:
        n_half = (len(x) + 1) // 2
        x_upper = x[:n_half][::-1]   # LE->TE
        y_upper = y[:n_half][::-1]
        ax.plot(x_upper, 2.0 * y_upper, label=info['method'], linewidth=1.2)
    ax.set_xlabel('x/c')
    ax.set_ylabel('t/c')
    ax.set_title('Thickness distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Summary table ---
    ax = axes[1, 1]
    ax.axis('off')
    col_labels = ['Method', 't/c target', 't/c actual', 'r_LE target',
                  'r_LE geom.', 'y_TE', 'N pts', 'Symmetric']
    table_data = []
    for _, _, info in results:
        sym = info.get('symmetry_check', {})
        table_data.append([
            info['method'],
            f"{info['tc_target']:.4f}",
            f"{info['tc_actual']:.4f}",
            f"{info['rle_target']:.5f}",
            f"{info.get('rle_geometric', 0):.5f}",
            f"{info.get('yte_actual', 0):.5f}",
            str(info['n_points']),
            'Yes' if sym.get('symmetric', False) else
            f"err={sym.get('max_symmetry_error', '?')}",
        ])
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    ax.set_title('Summary', fontsize=10)

    fig.suptitle(
        f"Bisymmetric foil comparison -- t/c = {results[0][2]['tc_target']:.3f}, "
        f"r_LE/c = {results[0][2]['rle_target']:.4f}, "
        f"y_TE/c = {results[0][2].get('yte', DEFAULT_YTE):.4f}",
        fontsize=12, fontweight='bold')

    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to {savefile}")
    else:
        plt.show()


def plot_rle_sweep(method_fn, tc: float, rle_values: list[float],
                   n: int = 161, savefile: str | None = None,
                   **kwargs) -> None:
    """Generate and plot a sweep of LE radii for a given method."""
    import matplotlib.pyplot as plt

    results = []
    for rle in rle_values:
        x, y, info = method_fn(tc=tc, rle=rle, n=n, **kwargs)
        results.append((x, y, info))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Shape overlay
    ax = axes[0]
    for x, y, info in results:
        rle = info['rle_target']
        ax.plot(x, y, label=f'r_LE={rle:.4f}', linewidth=1.0)
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.set_title(f"{results[0][2]['method']} -- shape (t/c={tc:.3f})")
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # LE zoom
    ax = axes[1]
    for x, y, info in results:
        rle = info['rle_target']
        ax.plot(x, y, label=f'r_LE={rle:.4f}', linewidth=1.0)
    rle_max = max(rle_values)
    lim = max(4.0 * rle_max, 0.05)
    ax.set_xlim(-lim * 0.3, lim)
    ax.set_ylim(-lim * 0.7, lim * 0.7)
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.set_title('LE detail')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Thickness
    ax = axes[2]
    for x, y, info in results:
        rle = info['rle_target']
        n_half = (len(x) + 1) // 2
        x_upper = x[:n_half][::-1]
        y_upper = y[:n_half][::-1]
        ax.plot(x_upper, 2.0 * y_upper, label=f'r_LE={rle:.4f}', linewidth=1.0)
    ax.set_xlabel('x/c')
    ax.set_ylabel('t/c')
    ax.set_title('Thickness distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"LE radius sweep -- {results[0][2]['method']}, t/c = {tc:.3f}",
        fontsize=12, fontweight='bold')

    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to {savefile}")
    else:
        plt.show()


def plot_cp_comparison(results: list[tuple[NDArray, NDArray, dict]],
                       alpha: float = 0.0,
                       xfoil_bin: str | None = None,
                       savefile: str | None = None) -> None:
    """Plot Cp distributions for multiple methods at a given alpha.

    Runs XFOIL inviscid on each result and overlays the Cp(x) curves.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Cp distribution ---
    ax = axes[0]
    cp_data = []
    for x, y, info in results:
        try:
            x_cp, cp = xfoil_cp(x, y, alpha=alpha, xfoil_bin=xfoil_bin)
            ax.plot(x_cp, cp, label=info['method'], linewidth=1.0)
            cp_data.append((x_cp, cp, info))
        except (RuntimeError, FileNotFoundError) as e:
            print(f"  Warning: XFOIL Cp failed for {info['method']}: {e}")
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cp')
    ax.invert_yaxis()
    ax.set_title(f'Cp distribution (alpha = {alpha:.1f} deg, inviscid)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Cp near LE ---
    ax = axes[1]
    for x_cp, cp, info in cp_data:
        ax.plot(x_cp, cp, label=info['method'], linewidth=1.0)
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cp')
    ax.invert_yaxis()
    ax.set_xlim(-0.01, 0.15)
    ax.set_title('Cp -- LE detail')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    tc = results[0][2]['tc_target']
    rle = results[0][2]['rle_target']
    fig.suptitle(
        f"Cp comparison -- t/c = {tc:.3f}, r_LE/c = {rle:.4f}, "
        f"alpha = {alpha:.1f} deg",
        fontsize=12, fontweight='bold')

    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"  Cp plot saved to {savefile}")
    else:
        plt.show()


def plot_cavitation_bucket(bucket_data: list[tuple[NDArray, NDArray, str]],
                           sigma_op: float | None = None,
                           savefile: str | None = None) -> None:
    """Plot cavitation bucket: sigma_i vs alpha.

    Parameters
    ----------
    bucket_data : list of (alphas, sigma_i, label)
        Each entry is one curve to plot.
    sigma_op : float, optional
        Operating sigma — draw a horizontal line.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))

    for alphas, sigma_i, label in bucket_data:
        ax.plot(alphas, sigma_i, 'o-', label=label, linewidth=1.2,
                markersize=3)

    if sigma_op is not None:
        ax.axhline(sigma_op, color='r', linestyle='--', linewidth=1.0,
                   label=f'sigma_op = {sigma_op:.2f}')

    ax.set_xlabel('Angle of attack (deg)')
    ax.set_ylabel('sigma_i = -Cp_min')
    ax.set_title('Cavitation bucket (inviscid)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"  Bucket plot saved to {savefile}")
    else:
        plt.show()


def plot_cavity_overlay(x: NDArray, y: NDArray, cav_result: dict,
                        savefile: str | None = None) -> None:
    """Plot airfoil profile with cavity thickness overlaid.

    Parameters
    ----------
    x, y : ndarray
        Airfoil coordinates (Selig format).
    cav_result : dict
        Output from xfoil_cavity().
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                             gridspec_kw={'height_ratios': [2, 1]})

    # --- Airfoil + cavity ---
    ax = axes[0]
    ax.plot(x, y, 'k-', linewidth=1.0, label='Airfoil')

    if cav_result['has_cavity']:
        xc = cav_result['x_cav']
        yc = cav_result['y_cav']
        hc = cav_result['h_cav']
        # Cavity sits on the suction surface; plot it as filled region
        ax.fill_between(xc, yc, yc + hc, alpha=0.3, color='C0',
                        label='Cavity')
        ax.plot(xc, yc + hc, 'C0-', linewidth=1.0)

    sigma = cav_result['sigma']
    alpha = cav_result['alpha']
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.set_title(f'Cavity overlay (alpha = {alpha:.1f} deg, '
                 f'sigma = {sigma:.2f})')
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Cp distribution with cavity ---
    ax = axes[1]
    if len(cav_result['x_cp']) > 0:
        ax.plot(cav_result['x_cp'], cav_result['cp'], 'k-',
                linewidth=0.8, label='Cp')
    ax.axhline(-sigma, color='r', linestyle='--', linewidth=0.8,
               label=f'-sigma = {-sigma:.2f}')
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cp')
    ax.invert_yaxis()
    ax.set_title('Cp distribution with cavitation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if cav_result['has_cavity']:
        length = cav_result['cavity_length']
        hmax = cav_result['max_thickness']
        cd = cav_result['cd_cav']
        fig.text(0.02, 0.01,
                 f"Cavity length/c = {length:.4f},  "
                 f"max h/c = {hmax:.5f},  "
                 f"CDcav = {cd:.5f}",
                 fontsize=9, family='monospace')
    else:
        fig.text(0.02, 0.01, "No cavity at this operating point.",
                 fontsize=9)

    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"  Cavity plot saved to {savefile}")
    else:
        plt.show()


# ===================================================================
#  CLI
# ===================================================================

def print_info(info: dict) -> None:
    """Pretty-print the diagnostics dictionary."""
    print(f"\n  Method:          {info['method']}")
    print(f"  t/c target:      {info['tc_target']:.5f}")
    print(f"  t/c actual:      {info['tc_actual']:.5f}")
    print(f"  r_LE target:     {info['rle_target']:.6f}")
    print(f"  r_LE geometric:  {info.get('rle_geometric', 0):.6f}")
    if 'rle_formula' in info:
        print(f"  r_LE formula:    {info['rle_formula']:.6f}")
    if 'rle_ellipse' in info:
        print(f"  r_LE ellipse:    {info['rle_ellipse']:.6f}")
    if 'rle_standard_naca' in info:
        print(f"  r_LE std NACA:   {info['rle_standard_naca']:.6f}")
    print(f"  y_TE target:     {info.get('yte', DEFAULT_YTE):.5f}")
    print(f"  y_TE actual:     {info.get('yte_actual', 0):.5f}")
    if 'a0' in info:
        print(f"  a0 coefficient:  {info['a0']:.6f}  (standard NACA = 0.2969)")
    if 'A_coefficients' in info:
        A = info['A_coefficients']
        print(f"  CST coeffs:     {['%.4f' % a for a in A]}")
    if 'doc' in info:
        print(f"  LERA blend d/c:  {info['doc']:.3f}")
    if 'superellipse_p' in info:
        print(f"  Superellipse p:  {info['superellipse_p']:.4f}"
              f"  (p=2 is standard ellipse)")
    print(f"  Total points:    {info['n_points']}")

    # Symmetry check
    sym = info.get('symmetry_check', {})
    if sym:
        status = "YES" if sym['symmetric'] else f"NO (err={sym['max_symmetry_error']:.2e})"
        print(f"  Fore-aft sym:    {status}")
        g2 = "YES" if sym.get('g2_continuous', False) else \
            f"NO (jump={sym.get('curvature_jump_at_midchord', '?')})"
        print(f"  G2 at midchord:  {g2}")


def main():
    parser = argparse.ArgumentParser(
        description='Bisymmetric airfoil generator for tunnel thruster blades. '
                    'All profiles are symmetric about x/c=0.5 (identical LE/TE) '
                    'with tunable LE radius for cavitation suppression.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --method modified_naca --tc 0.06 --rle 0.010 -o thruster06.dat
  %(prog)s --method elliptic_blend --tc 0.10 --rle 0.020
  %(prog)s --method cst --tc 0.08 --rle 0.015 --cst-order 6
  %(prog)s --compare --tc 0.06 --rle 0.010
  %(prog)s --method modified_naca --tc 0.06 --rle-sweep 0.005,0.010,0.015,0.020
  %(prog)s --compare --tc 0.06 --rle 0.010 --plot-cp
  %(prog)s --cav-bucket --tc 0.06 --rle 0.010 --alpha-range 0,6,0.25
  %(prog)s --sigma 0.5 --alpha 4.0 --tc 0.06 --rle 0.010

Notes:
  All methods produce profiles symmetric about x/c = 0.5 with G2 continuity
  at the midchord join.  Default TE gap is 0 (sharp, symmetric TE).
  Use --yte to add a finite TE gap for XFOIL panel analysis, or apply it
  in XFOIL via GDES -> TGAP after loading the foil.

  XFOIL analysis (--plot-cp, --cav-bucket, --sigma) uses inviscid mode.
  The XFOIL binary is auto-detected at bin/xfoil relative to this script,
  or provide --xfoil-bin explicitly.
        """)

    parser.add_argument('--method',
                        choices=['modified_naca', 'elliptic_blend', 'cst'],
                        default='modified_naca',
                        help='Parameterisation method (default: modified_naca)')
    parser.add_argument('--tc', type=float, default=0.06,
                        help='Max thickness-to-chord ratio (default: 0.06)')
    parser.add_argument('--rle', type=float, default=None,
                        help='Desired LE radius / chord.  If omitted, uses '
                             'the standard NACA value for the given t/c.')
    parser.add_argument('--rle-multiplier', type=float, default=None,
                        help='LE radius as a multiplier of the standard NACA '
                             'value (e.g. 1.5 = 50%% larger than standard).')
    parser.add_argument('--rle-sweep', type=str, default=None,
                        help='Comma-separated list of r_LE values for a sweep '
                             '(e.g. 0.005,0.010,0.015)')
    parser.add_argument('-n', '--npoints', type=int, default=161,
                        help='Points per side (default: 161, total = 2n-1)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output .dat file path')
    parser.add_argument('--name', type=str, default=None,
                        help='Airfoil name in .dat file header')
    parser.add_argument('--compare', action='store_true',
                        help='Generate all three methods and plot comparison')
    parser.add_argument('--plot', action='store_true',
                        help='Show plot of generated airfoil')
    parser.add_argument('--plot-save', type=str, default=None,
                        help='Save plot to file instead of showing')
    parser.add_argument('--doc', type=float, default=0.5,
                        help='(Unused, retained for compatibility.) '
                             'Formerly controlled blending distance for '
                             'elliptic_blend; the method now uses a '
                             'superellipse with no blending parameter.')
    parser.add_argument('--cst-order', type=int, default=6,
                        help='Initial Bernstein polynomial order for CST '
                             '(default: 6, must be even; auto-increased for '
                             'large r_LE/tc ratios)')
    parser.add_argument('--yte', type=float, default=DEFAULT_YTE,
                        help=f'Trailing-edge total gap / chord '
                             f'(default: {DEFAULT_YTE})')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress info output')
    parser.add_argument('--check-symmetry', action='store_true',
                        help='Run and report fore-aft symmetry and G2 checks')

    # XFOIL analysis options
    xfoil_group = parser.add_argument_group('XFOIL analysis')
    xfoil_group.add_argument('--xfoil-bin', type=str, default=None,
                             help='Path to XFOIL binary (auto-detected if omitted)')
    xfoil_group.add_argument('--alpha', type=float, default=0.0,
                             help='Angle of attack for Cp extraction (default: 0)')
    xfoil_group.add_argument('--plot-cp', action='store_true',
                             help='Run XFOIL and plot Cp distribution')
    xfoil_group.add_argument('--cav-bucket', action='store_true',
                             help='Compute and plot cavitation bucket '
                                  '(sigma_i vs alpha)')
    xfoil_group.add_argument('--alpha-range', type=str, default='0,6,0.25',
                             help='Alpha sweep: start,end,step in degrees '
                                  '(default: 0,6,0.25)')
    xfoil_group.add_argument('--sigma', type=float, default=None,
                             help='Cavitation number for cavity analysis')
    xfoil_group.add_argument('--re', type=float, default=None,
                             help='Reynolds number for viscous polar '
                                  '(omit for inviscid-only analysis)')
    xfoil_group.add_argument('--viscous-polar', action='store_true',
                             help='Run viscous polar sweep (requires --re)')

    args = parser.parse_args()

    # --- Resolve LE radius ---
    rle_std_naca = _naca_rle_from_a0(0.2969, args.tc)

    if args.rle is not None:
        rle = args.rle
    elif args.rle_multiplier is not None:
        rle = rle_std_naca * args.rle_multiplier
    else:
        rle = rle_std_naca  # default to standard NACA value

    if not args.quiet:
        print(f"\n  Bisymmetric Foil Generator (fore-aft symmetric)")
        print(f"  ------------------------------------------------")
        print(f"  t/c = {args.tc:.4f},  r_LE/c = {rle:.6f},  "
              f"y_TE/c = {args.yte:.5f}")
        print(f"  (Standard NACA r_LE/c = {rle_std_naca:.6f} for this t/c)")
        if rle != rle_std_naca:
            print(f"  LE radius multiplier vs standard: {rle/rle_std_naca:.2f}x")

    def _add_symmetry_check(x, y, info):
        """Add symmetry check results to info dict."""
        n_half = (len(x) + 1) // 2
        xx_half = x[:n_half][::-1]
        yt_half = y[:n_half][::-1]
        info['symmetry_check'] = _check_symmetry(xx_half, yt_half, info['yte'])

    # --- LE radius sweep mode ---
    if args.rle_sweep:
        rle_values = [float(v.strip()) for v in args.rle_sweep.split(',')]
        method_map = {
            'modified_naca': modified_naca,
            'elliptic_blend': elliptic_blend,
            'cst': cst,
        }
        method_fn = method_map[args.method]
        extra_kwargs = {'y_te': args.yte}
        if args.method == 'elliptic_blend':
            extra_kwargs['doc'] = args.doc
        elif args.method == 'cst':
            extra_kwargs['order'] = args.cst_order

        for rv in rle_values:
            x, y, info = method_fn(tc=args.tc, rle=rv, n=args.npoints,
                                   **extra_kwargs)
            if args.check_symmetry:
                _add_symmetry_check(x, y, info)
            if not args.quiet:
                print_info(info)
            if args.output:
                stem = Path(args.output).stem
                suffix = Path(args.output).suffix or '.dat'
                parent = Path(args.output).parent
                outpath = parent / f"{stem}_rle{rv:.4f}{suffix}"
                name = (args.name or
                        f"Bisym {args.method} tc={args.tc:.3f} rle={rv:.4f}")
                write_dat(outpath, x, y, name=name)
                if not args.quiet:
                    print(f"  Written to {outpath}")

        plot_rle_sweep(method_fn, args.tc, rle_values, n=args.npoints,
                       savefile=args.plot_save, **extra_kwargs)
        return

    # --- Comparison mode ---
    if args.compare:
        results = []
        for method_name, method_fn, mkw in [
            ('modified_naca', modified_naca,
             {'y_te': args.yte}),
            ('elliptic_blend', elliptic_blend,
             {'y_te': args.yte, 'doc': args.doc}),
            ('cst', cst,
             {'order': args.cst_order, 'y_te': args.yte}),
        ]:
            x, y, info = method_fn(tc=args.tc, rle=rle, n=args.npoints, **mkw)
            if args.check_symmetry:
                _add_symmetry_check(x, y, info)
            results.append((x, y, info))
            if not args.quiet:
                print_info(info)

        plot_comparison(results, savefile=args.plot_save)

        # Cp comparison overlay
        if args.plot_cp:
            cp_save = None
            if args.plot_save:
                stem = Path(args.plot_save).stem
                cp_save = str(Path(args.plot_save).parent /
                              f"{stem}_cp{Path(args.plot_save).suffix}")
            plot_cp_comparison(results, alpha=args.alpha,
                               xfoil_bin=args.xfoil_bin, savefile=cp_save)

        if args.output:
            for x, y, info in results:
                stem = Path(args.output).stem
                suffix = Path(args.output).suffix or '.dat'
                parent = Path(args.output).parent
                outpath = parent / f"{stem}_{info['method']}{suffix}"
                name = (args.name or
                        f"Bisym {info['method']} tc={args.tc:.3f} rle={rle:.4f}")
                write_dat(outpath, x, y, name=name)
                if not args.quiet:
                    print(f"  Written to {outpath}")
        return

    # --- Single method mode ---
    if args.method == 'modified_naca':
        x, y, info = modified_naca(tc=args.tc, rle=rle, n=args.npoints,
                                   y_te=args.yte)
    elif args.method == 'elliptic_blend':
        x, y, info = elliptic_blend(tc=args.tc, rle=rle, n=args.npoints,
                                    y_te=args.yte, doc=args.doc)
    elif args.method == 'cst':
        x, y, info = cst(tc=args.tc, rle=rle, n=args.npoints,
                         order=args.cst_order, y_te=args.yte)
    else:
        print(f"Unknown method: {args.method}", file=sys.stderr)
        sys.exit(1)

    if args.check_symmetry:
        _add_symmetry_check(x, y, info)

    if not args.quiet:
        print_info(info)

    # Determine output path
    if args.output:
        outpath = args.output
    else:
        outpath = f"bisym_{args.method}_tc{args.tc:.3f}_rle{rle:.4f}.dat"

    name = args.name or f"Bisym {args.method} tc={args.tc:.3f} rle={rle:.4f}"
    write_dat(outpath, x, y, name=name)
    if not args.quiet:
        print(f"\n  Written to {outpath}")

    if args.plot or args.plot_save:
        plot_comparison([(x, y, info)], savefile=args.plot_save)

    # --- XFOIL Cp plot ---
    if args.plot_cp:
        cp_save = None
        if args.plot_save:
            stem = Path(args.plot_save).stem
            cp_save = str(Path(args.plot_save).parent /
                          f"{stem}_cp{Path(args.plot_save).suffix}")
        plot_cp_comparison([(x, y, info)], alpha=args.alpha,
                           xfoil_bin=args.xfoil_bin, savefile=cp_save)

    # --- Cavitation bucket ---
    if args.cav_bucket:
        a_parts = [float(v.strip()) for v in args.alpha_range.split(',')]
        a_start, a_end, a_step = a_parts[0], a_parts[1], a_parts[2]
        if not args.quiet:
            print(f"\n  Computing cavitation bucket "
                  f"(alpha {a_start:.1f} to {a_end:.1f} step {a_step:.2f}) ...")
        alphas, sigma_i, cp_min = xfoil_cavitation_bucket(
            x, y, alpha_start=a_start, alpha_end=a_end, alpha_step=a_step,
            xfoil_bin=args.xfoil_bin)
        if not args.quiet:
            print(f"  {'alpha':>7s}  {'sigma_i':>8s}  {'Cp_min':>8s}")
            for a, s, c in zip(alphas, sigma_i, cp_min):
                print(f"  {a:7.2f}  {s:8.4f}  {c:8.4f}")
        label = f"{info['method']} tc={args.tc:.3f} rle={rle:.4f}"
        bkt_save = None
        if args.plot_save:
            stem = Path(args.plot_save).stem
            bkt_save = str(Path(args.plot_save).parent /
                           f"{stem}_bucket{Path(args.plot_save).suffix}")
        plot_cavitation_bucket(
            [(alphas, sigma_i, label)],
            sigma_op=args.sigma,
            savefile=bkt_save)

    # --- Cavity analysis at given sigma ---
    if args.sigma is not None and not args.cav_bucket:
        if not args.quiet:
            print(f"\n  Running cavity analysis: "
                  f"alpha={args.alpha:.1f}, sigma={args.sigma:.3f} ...")
        cav = xfoil_cavity(x, y, alpha=args.alpha, sigma=args.sigma,
                           xfoil_bin=args.xfoil_bin)
        if not args.quiet:
            if cav['has_cavity']:
                print(f"  Cavity detected:")
                print(f"    Length/c     = {cav['cavity_length']:.5f}")
                print(f"    Max h/c      = {cav['max_thickness']:.6f}")
                print(f"    CDcav        = {cav['cd_cav']:.6f}")
            else:
                print(f"  No cavity at sigma={args.sigma:.3f}, "
                      f"alpha={args.alpha:.1f}")
        cav_save = None
        if args.plot_save:
            stem = Path(args.plot_save).stem
            cav_save = str(Path(args.plot_save).parent /
                           f"{stem}_cavity{Path(args.plot_save).suffix}")
        plot_cavity_overlay(x, y, cav, savefile=cav_save)

    # --- Viscous polar ---
    if args.viscous_polar:
        if args.re is None:
            print("  Error: --viscous-polar requires --re", file=sys.stderr)
            sys.exit(1)
        a_parts = [float(v.strip()) for v in args.alpha_range.split(',')]
        a_start, a_end, a_step = a_parts[0], a_parts[1], a_parts[2]
        if not args.quiet:
            print(f"\n  Running viscous polar at Re={args.re:.2E} "
                  f"(alpha {a_start:.1f} to {a_end:.1f} step {a_step:.2f}) ...")
        polar = xfoil_viscous_polar(
            x, y, re=args.re,
            alpha_start=a_start, alpha_end=a_end, alpha_step=a_step,
            xfoil_bin=args.xfoil_bin)
        if not args.quiet:
            n_conv = polar['converged_count']
            print(f"  Converged points: {n_conv}")
            if n_conv > 0:
                print(f"  {'alpha':>7s}  {'CL':>8s}  {'CD':>9s}  "
                      f"{'CDp':>9s}  {'CM':>8s}  {'Xtr_top':>7s}")
                for i in range(n_conv):
                    print(f"  {polar['alpha'][i]:7.2f}  "
                          f"{polar['cl'][i]:8.4f}  "
                          f"{polar['cd'][i]:9.5f}  "
                          f"{polar['cdp'][i]:9.5f}  "
                          f"{polar['cm'][i]:8.4f}  "
                          f"{polar['top_xtr'][i]:7.4f}")


if __name__ == '__main__':
    main()
