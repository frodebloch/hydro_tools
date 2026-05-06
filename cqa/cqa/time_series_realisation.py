"""Time-domain realisation of stationary disturbances and motions.

Used by the prior-vs-posterior demo to feed the online Bayesian sigma
estimator with a *physically realised* time series (rather than a toy
AR(1)). Mirrors the brucon vessel_simulator two-timescale architecture:

  * Low-frequency channel: closed-loop ODE driven by Shinozuka
    realisations of the wind-gust, slow-drift, and current-variability
    force PSDs. Output = position deviation eta(t) and velocity nu(t).

  * Wave-frequency channel: 1st-order RAO Shinozuka realisation of the
    6-DOF body motion at the body origin, summed over a directional
    spreading quadrature.

The two are uncorrelated by construction (different frequency bands,
different physical mechanisms) and superimpose linearly to give the
observed deviation at any sensor (gangway base position, telescope
length, etc).

Shinozuka harmonic superposition
--------------------------------
For a one-sided real PSD S(omega) [signal^2 / (rad/s)] on a uniform
grid {omega_k} with step d_omega, a realisation of the stationary
Gaussian process is

    x(t) = sum_k sqrt(2 S(omega_k) d_omega) cos(omega_k t + phi_k),
    phi_k ~ U(0, 2*pi) iid.

For a *vector* PSD S(omega) [3x3 hermitian, real for our PSDs] we use
the matrix-square-root extension (Shinozuka 1972, Shinozuka & Deodatis
1991): factor 2 S(omega_k) d_omega = L_k L_k^T (Cholesky) and write

    F(t) = sum_k L_k * [cos(omega_k t + phi_k_1), ...]^T

with independent phases per channel per frequency. The result has
exact one-sided PSD == S in the limit of dense omega grids and
duration T much larger than the largest correlation time.

For the wave channel, the elevation realisation is

    eta(t, x; phi) = sum_{k,j} sqrt(2 S_eta(om_k) d_om * D(phi_j) d_phi)
                     * cos(om_k t - k_k * (x cos(beta+phi_j) + y sin(.))
                          + psi_{k,j})

with k_k = om_k^2 / g (deep water). The body motion at the origin
(x=y=0) is

    xi_dof(t) = sum_{k,j} A_{k,j} * |H_dof(om_k, beta+phi_j)|
                * cos(om_k t + psi_{k,j} + arg H)

where A_{k,j} = sqrt(2 S_eta(om_k) d_om * D(phi_j) d_phi) is the wave
amplitude in that (frequency, direction) bin. The 6 DOFs share the
same {psi_{k,j}} -- they are perfectly coherent at the wave-frequency
band, only the complex RAO transforms phase and amplitude per DOF.

References
----------
* Shinozuka, M. (1972), "Monte Carlo solution of structural dynamics",
  Computers & Structures 2, 855-874.
* Shinozuka, M. & Deodatis, G. (1991), "Simulation of stochastic
  processes by spectral representation", ASME Appl. Mech. Rev. 44,
  191-204.
* DNV-RP-C205 (2021) sec. 3.3.4 (irregular wave realisation).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .closed_loop import ClosedLoop
from .config import CqaConfig
from .gangway import (
    GangwayJointState,
    telescope_sensitivity,
    telescope_sensitivity_6dof,
)
from .psd import jonswap_psd, wave_elevation_psd, WaveSpectrumKind
from .rao import RaoTable, evaluate_rao
from .sea_spreading import SeaSpreading, spreading_quadrature
from .wave_response import cqa_theta_rel_to_pdstrip_beta_deg


G_STD: float = 9.81


# ---------------------------------------------------------------------------
# Low-frequency realisation: vector force time series + closed-loop integration
# ---------------------------------------------------------------------------


def realise_vector_force_time_series(
    S_F_funcs: list[Callable[[float], np.ndarray]],
    omega_grid: np.ndarray,
    t: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shinozuka realisation of a 3-channel disturbance force time series.

    Parameters
    ----------
    S_F_funcs : list of callables omega -> 3x3 PSD [N^2 / (rad/s)].
        Each callable is summed (independent additive sources, e.g.
        wind gust + slow drift + current variability).
    omega_grid : (N_omega,) strictly-increasing angular frequencies
        [rad/s] bracketing the closed-loop band. May be non-uniform
        (e.g. ``np.geomspace``); per-bin widths are computed from
        midpoint differences so the closed-loop response is captured
        across decades with far fewer points than a linear grid.
    t : (N_t,) time samples [s]. Should satisfy
        T_total = t[-1] - t[0] >> 2*pi/omega_grid[0]
        so the realisation is long enough to look stationary.
    rng : numpy random Generator. Drives the random phases.

    Returns
    -------
    F : (3, N_t) realised force time series [N or N*m for yaw moment].

    Notes
    -----
    * Cholesky factorisation per frequency requires PSD to be positive
      semi-definite. Tiny eigenvalues are clipped to zero (jitter
      protection). The output is real because the PSD callables are
      real-symmetric (no Coriolis-style cross-spectra in our use case).
    * Per-bin width Delta_omega_k = 0.5 (omega_{k+1} - omega_{k-1})
      with one-sided differences at the endpoints. This is the trapezoid
      cell width and recovers exactly d_omega for a uniform grid.
    * Memory: builds a (3, N_omega, N_t) intermediate; for the demo
      defaults (N_omega ~ 256, N_t ~ 30000) that is ~90 MB of floats.
    """
    omega_grid = np.asarray(omega_grid, dtype=float)
    t = np.asarray(t, dtype=float)
    if omega_grid.ndim != 1 or omega_grid.size < 2:
        raise ValueError("omega_grid must be a 1-D array with size >= 2")
    if np.any(np.diff(omega_grid) <= 0.0):
        raise ValueError("omega_grid must be strictly increasing")
    # Per-bin width from midpoint differences (trapezoid cells).
    d_omega_k = np.empty_like(omega_grid)
    d_omega_k[1:-1] = 0.5 * (omega_grid[2:] - omega_grid[:-2])
    d_omega_k[0] = omega_grid[1] - omega_grid[0]
    d_omega_k[-1] = omega_grid[-1] - omega_grid[-2]
    N_om = omega_grid.size
    N_t = t.size

    # Build the cumulative 3x3 PSD on the grid.
    S_total = np.zeros((N_om, 3, 3))
    for S_F in S_F_funcs:
        for i, w in enumerate(omega_grid):
            S = np.asarray(S_F(float(w)))
            if S.shape != (3, 3):
                raise ValueError(
                    f"S_F callable must return 3x3, got {S.shape} at "
                    f"omega = {w}"
                )
            S_total[i] += S

    # Symmetrise (numerical safety) and factor.
    S_total = 0.5 * (S_total + S_total.transpose(0, 2, 1))

    # Random phases per (frequency, channel). Independent phases mean
    # the output is white in the *channel* dimension before the L_k
    # mixing -- which is exactly what we want.
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(N_om, 3))

    # Per-frequency factor: L_k L_k^T = 2 S(omega_k) d_omega_k.
    # Use eigendecomposition (more robust to near-singular PSD than
    # Cholesky since wind PSD has zero off-diagonal entries).
    F_t = np.zeros((3, N_t))
    cos_args = np.outer(omega_grid, t) + phases[:, 0:1]  # placeholder shape
    # Loop over frequencies -- N_om is small (~256), keeps memory low.
    for k in range(N_om):
        S_k = 2.0 * S_total[k] * d_omega_k[k]
        eigvals, eigvecs = np.linalg.eigh(S_k)
        eigvals = np.clip(eigvals, 0.0, None)
        # Amplitude per "factor" channel: sqrt(eigval) along eigvec.
        amps = np.sqrt(eigvals)  # (3,)
        # Three independent cosines (one per eigen channel) sharing the
        # same omega_k but with independent phases.
        c0 = amps[0] * np.cos(omega_grid[k] * t + phases[k, 0])
        c1 = amps[1] * np.cos(omega_grid[k] * t + phases[k, 1])
        c2 = amps[2] * np.cos(omega_grid[k] * t + phases[k, 2])
        # Mix back to the original (surge, sway, yaw-moment) basis.
        F_t += eigvecs[:, 0:1] * c0 + eigvecs[:, 1:2] * c1 + eigvecs[:, 2:3] * c2

    return F_t


def integrate_closed_loop_response(
    cl: ClosedLoop,
    F_lf: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Integrate the linear closed loop driven by a force time series.

    State: x = [eta_n, eta_e, psi, u, v, r] (6-vector).
    ODE:   x_dot = A_cl x + B_w F(t).

    Parameters
    ----------
    cl : ClosedLoop with A_cl (6x6) and B_w (6x3).
    F_lf : (3, N_t) low-frequency force time series.
    t : (N_t,) time samples.

    Returns
    -------
    x_lf : (6, N_t) closed-loop state trajectory.

    Notes
    -----
    Uses the matrix-exponential ZOH discretisation per uniform time
    step dt: x_{k+1} = exp(A_cl dt) x_k + (A_cl^{-1}(exp(A_cl dt) - I)) B_w F_k.
    Exact for ZOH inputs and stable for any dt (no numerical CFL limit).
    Cost: one 6x6 matrix exponential per call, then N_t cheap multiplies.
    """
    from scipy.linalg import expm

    t = np.asarray(t, dtype=float)
    if F_lf.shape[1] != t.size:
        raise ValueError(
            f"F_lf has {F_lf.shape[1]} time samples, t has {t.size}"
        )
    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt):
        raise ValueError("t must be uniformly spaced")

    A = cl.A_cl
    Bw = cl.B_w
    N = t.size

    Phi = expm(A * dt)
    # Gamma = A^{-1} (Phi - I) Bw    (ZOH input matrix).
    # Use solve for numerical stability.
    Gamma = np.linalg.solve(A, (Phi - np.eye(A.shape[0])) @ Bw)

    x = np.zeros((6, N))
    x_k = np.zeros(6)
    for k in range(N - 1):
        x_k = Phi @ x_k + Gamma @ F_lf[:, k]
        x[:, k + 1] = x_k
    return x


# ---------------------------------------------------------------------------
# Wave-frequency realisation: 6-DOF rigid-body motion at body origin
# ---------------------------------------------------------------------------


def realise_wave_motion_6dof(
    rao_table: RaoTable,
    Hs: float,
    Tp: float,
    theta_wave_rel: float,
    t: np.ndarray,
    rng: np.random.Generator,
    *,
    omega_grid: Optional[np.ndarray] = None,
    spreading: Optional[SeaSpreading] = None,
    gamma: float = 3.3,
    spectrum: WaveSpectrumKind = "bretschneider",
) -> np.ndarray:
    """6-DOF wave-frequency body motion realisation at the body origin.

    For each (omega_k, phi_j) bin, the wave amplitude is

        A_{k,j} = sqrt(2 * S_eta(om_k) * d_om * D(phi_j) * d_phi)

    and the body motion contribution per DOF is

        xi_dof(t) += A_{k,j} * |H_dof(om_k, beta+phi_j)|
                   * cos(om_k t + psi_{k,j} + arg H_dof)

    where beta corresponds to ``theta_wave_rel`` via the standard
    cqa->pdstrip mapping.

    Parameters
    ----------
    rao_table : pdstrip-derived RAO table (see load_pdstrip_rao).
    Hs, Tp : significant wave height [m] and peak period [s].
    theta_wave_rel : mean relative wave direction [rad], cqa convention
        (0 = head, +pi/2 = from port).
    t : (N_t,) uniformly-spaced time samples [s].
    rng : numpy random Generator.
    omega_grid : optional (N_om,) strictly-increasing [rad/s]. May be
        non-uniform; per-bin widths are computed from midpoint
        differences. Default: 128 linear points across the RAO range.
    spreading : SeaSpreading. Default cos-2s s=15 (DNV-RP-C205).
    gamma : JONSWAP peakedness. Default 3.3. Ignored when
        ``spectrum == 'bretschneider'``.
    spectrum : wave-elevation PSD shape. Default ``'bretschneider'``
        (IMCA / DNV-ST-0111 / brucon vessel_simulator default).

    Returns
    -------
    xi : (6, N_t) array, DOF order [surge, sway, heave, roll, pitch, yaw].
        Translations in m, rotations in rad.
    """
    t = np.asarray(t, dtype=float)
    N_t = t.size
    if omega_grid is None:
        om_lo = float(rao_table.omega[0])
        om_hi = float(rao_table.omega[-1])
        omega_grid = np.linspace(om_lo, om_hi, 128)
    omega_grid = np.asarray(omega_grid, dtype=float)
    if np.any(np.diff(omega_grid) <= 0.0):
        raise ValueError("omega_grid must be strictly increasing")
    d_omega_k = np.empty_like(omega_grid)
    d_omega_k[1:-1] = 0.5 * (omega_grid[2:] - omega_grid[:-2])
    d_omega_k[0] = omega_grid[1] - omega_grid[0]
    d_omega_k[-1] = omega_grid[-1] - omega_grid[-2]

    if spreading is None:
        spreading = SeaSpreading()
    angles_rel, w_dir = spreading_quadrature(spreading, theta_wave_rel)

    # Wave-elevation PSD on the grid (Bretschneider by default; JONSWAP
    # opt-in via ``spectrum='jonswap'``).
    S_eta = wave_elevation_psd(omega_grid, Hs=Hs, Tp=Tp, kind=spectrum, gamma=gamma)  # (N_om,)

    xi = np.zeros((6, N_t))
    for j, theta_j in enumerate(angles_rel):
        # RAO at this absolute incidence direction.
        beta_deg_j = cqa_theta_rel_to_pdstrip_beta_deg(theta_j)
        H = evaluate_rao(rao_table, omega_grid, beta_deg_j)  # (N_om, 6)
        amp_H = np.abs(H)             # (N_om, 6)
        phase_H = np.angle(H)         # (N_om, 6)
        # Spectral amplitude per (omega) bin within this directional bin.
        # Note: spreading_quadrature returns weights that already
        # absorb d_phi (sum_j w_j == 1 implies w_j = D(phi_j) d_phi).
        A_om = np.sqrt(2.0 * S_eta * d_omega_k * w_dir[j])  # (N_om,)
        # Independent random phases per (omega, this directional bin)
        # shared across the 6 DOFs (rigid body excited coherently).
        psi = rng.uniform(0.0, 2.0 * np.pi, size=omega_grid.size)  # (N_om,)
        # cos(om t + psi + arg H) summed over omega for each DOF.
        # Build (N_om, N_t) matrix once per direction.
        phi_t = omega_grid[:, None] * t[None, :]  # (N_om, N_t)
        for d in range(6):
            xi[d] += np.sum(
                (A_om * amp_H[:, d])[:, None]
                * np.cos(phi_t + psi[:, None] + phase_H[:, d, None]),
                axis=0,
            )
    return xi


# ---------------------------------------------------------------------------
# Channel projections
# ---------------------------------------------------------------------------


def base_position_xy_time_series(
    x_lf: np.ndarray,
    xi_wf: np.ndarray,
    cfg: CqaConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis gangway-base position deviation (dx(t), dy(t)) [m].

    Same construction as ``radial_position_time_series`` but returns
    the two scalar axes separately *before* the ``sqrt(dx^2+dy^2)``
    collapse. These are the natural channels for the online Bayesian
    estimator: x and y are zero-mean by DP construction (the DP
    integral term and the observer bias estimator regulate them to
    setpoint), so the A2 (zero-mean) assumption that
    ``BayesianSigmaEstimator.posterior()`` relies on holds. The
    aggregated ``r = sqrt(dx^2+dy^2)`` is Rayleigh-distributed and has
    a structural non-zero mean ``E[r] = sqrt(pi/2)*sigma`` even when
    (dx, dy) are perfectly zero-mean -- which contaminates the
    ``sample_mean_over_sigma`` health primitive.

    Combine the per-axis posteriors for the Rice formula via

        sigma_radial_post = sqrt(sigma_x_post^2 + sigma_y_post^2),

    valid when (x, y) are uncorrelated (the canonical case for
    decoupled surge/sway controllers; oblique forcing introduces a
    cross-term that is small for the operating points considered here).

    Returns
    -------
    (dx, dy) : tuple of (N_t,) arrays, in metres, base position
        deviation in the body-frame x and y directions.
    """
    base_x_b, base_y_b, _ = cfg.gangway.base_position_body
    c_x = np.array([1.0, 0.0, -base_y_b])
    c_y = np.array([0.0, 1.0,  base_x_b])
    eta_lf = x_lf[0:3]
    eta_wf = xi_wf[[0, 1, 5]]
    eta_total = eta_lf + eta_wf
    dx = c_x @ eta_total
    dy = c_y @ eta_total
    return dx, dy


def radial_position_time_series(
    x_lf: np.ndarray,
    xi_wf: np.ndarray,
    cfg: CqaConfig,
) -> np.ndarray:
    """Radial gangway-base position deviation [m].

    Combines:
      * Low-frequency: c_base . eta_lf with c_base_x = (1, 0, -base_y),
        c_base_y = (0, 1, base_x). Gives (delta_x, delta_y) at gangway
        base.
      * Wave-frequency: same projection on the 6-DOF rigid-body
        motion at the body origin -- only surge, sway, yaw matter for
        horizontal base displacement (heave, roll, pitch contribute
        height, not radial range).

    For the online Bayesian estimator, prefer
    ``base_position_xy_time_series`` -- the per-axis (dx, dy) channels
    are zero-mean by DP construction (A2 holds), whereas the radial
    aggregate r is Rayleigh-distributed with a structural non-zero
    mean.

    Returns
    -------
    r(t) : (N_t,) radial distance from setpoint at the gangway base.
    """
    dx, dy = base_position_xy_time_series(x_lf, xi_wf, cfg)
    return np.sqrt(dx * dx + dy * dy)


def telescope_length_deviation_time_series(
    x_lf: np.ndarray,
    xi_wf: np.ndarray,
    joint: GangwayJointState,
    cfg: CqaConfig,
) -> np.ndarray:
    """Telescope-length deviation Delta_L(t) [m].

    Sums the 3-DOF slow projection (c . eta_lf) and the 6-DOF wave
    projection (c6 . xi_wf). Sign convention: matches
    ``telescope_sensitivity`` (positive = MORE telescope needed to keep
    tip on landing point).
    """
    c3 = telescope_sensitivity(joint, cfg.gangway)  # (3,)
    c6 = telescope_sensitivity_6dof(joint, cfg.gangway)  # (6,)
    eta_lf = x_lf[0:3]
    return c3 @ eta_lf + c6 @ xi_wf


__all__ = [
    "realise_vector_force_time_series",
    "integrate_closed_loop_response",
    "realise_wave_motion_6dof",
    "radial_position_time_series",
    "base_position_xy_time_series",
    "telescope_length_deviation_time_series",
]
