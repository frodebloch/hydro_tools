"""Closed-loop linearisation and frequency-domain covariance.

State:  x = [eta(3), nu(3)]   (6-vector)
Plant:  x_dot = A x + B tau + B_w w
Control: tau = -K x   with K = [Kp, Kd]
Closed loop: x_dot = A_cl x + B_w w,   A_cl = A - B K

Two equivalent ways to compute the steady-state state covariance P from a
disturbance with one-sided force PSD S_F(omega) [N^2/(rad/s)]:

(a) Frequency-domain integration:
        P = (1/pi) * integral_0^inf  H(omega) S_F(omega) H(omega)^H d omega
    with H(omega) = (j*omega*I - A_cl)^{-1} B_w.

(b) Lyapunov equation, approximating S_F as locally white in the closed-loop band:
        A_cl P + P A_cl^T + B_w (pi * S_F_avg) B_w^T = 0.

We provide (a) as the primary path because it is numerically robust against
spectra with strong frequency dependence (e.g. NPD wind gust). (b) is kept
for diagnostics and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from .vessel import LinearVesselModel
from .controller import LinearDpController


@dataclass
class ClosedLoop:
    A_cl: np.ndarray  # 6x6
    B_w: np.ndarray  # 6x3 (force enters via M^{-1})

    @classmethod
    def build(cls, vessel: LinearVesselModel, controller: LinearDpController) -> "ClosedLoop":
        A, B = vessel.state_space()  # 6x6, 6x3
        Kp, Kd = controller.feedback()
        K = np.hstack([Kp, Kd])  # 3x6
        A_cl = A - B @ K
        return cls(A_cl=A_cl, B_w=B)


def state_covariance_freqdomain_general(
    A: np.ndarray,
    B_w: np.ndarray,
    S_F_funcs: list[Callable[[np.ndarray], np.ndarray]],
    omega_lo: float = 1e-4,
    omega_hi: float = 1.0,
    n_points: int = 1024,
) -> np.ndarray:
    """Steady-state state covariance for a general linear system.

    P = integral_0^inf H(omega) S_F(omega) H(omega)^H d omega
    with H(omega) = (j*omega*I - A)^{-1} B_w.

    Generic version (arbitrary A, B_w): used both for the 6-state intact
    closed loop and for the 12-state augmented system (eta, nu, b_hat,
    tau_thr) needed by the WCFDI starting-state Monte-Carlo to produce
    a non-trivial steady-state covariance on b_hat and tau_thr.

    The disturbance PSD `S_F` is expressed in *force* units (3x3) and
    enters via `B_w`, exactly as in the 6-state case. For the augmented
    system the caller passes `B_w_aug` with the wind/drift/current
    forces injected only into the nu channel; b_hat and tau_thr have no
    direct stochastic input but acquire variance through the closed-loop
    coupling with eta.
    """
    omega = np.logspace(np.log10(omega_lo), np.log10(omega_hi), n_points)
    n_state = A.shape[0]
    I_n = np.eye(n_state)
    integrand = np.zeros((n_points, n_state, n_state), dtype=complex)
    for i, w in enumerate(omega):
        H = np.linalg.solve(1j * w * I_n - A, B_w)
        S_total = np.zeros((B_w.shape[1], B_w.shape[1]))
        for S_F in S_F_funcs:
            S_total = S_total + S_F(w)
        integrand[i] = H @ S_total @ H.conj().T
    P_complex = np.trapezoid(integrand, omega, axis=0)
    P = P_complex.real
    P = 0.5 * (P + P.T)
    return P


def state_covariance_freqdomain(
    cl: ClosedLoop,
    S_F_funcs: list[Callable[[np.ndarray], np.ndarray]],
    omega_lo: float = 1e-4,
    omega_hi: float = 1.0,
    n_points: int = 1024,
) -> np.ndarray:
    """Steady-state state covariance for the 6-state intact closed loop.

    Thin wrapper around `state_covariance_freqdomain_general`. Kept as the
    public 6-state entry point used by P1 (excursion polar).
    """
    return state_covariance_freqdomain_general(
        cl.A_cl, cl.B_w, S_F_funcs, omega_lo, omega_hi, n_points
    )


def lyapunov_position_covariance(cl: ClosedLoop, W: np.ndarray) -> np.ndarray:
    """Solve A_cl P + P A_cl^T + B_w W B_w^T = 0 (Lyapunov, white-noise input).

    Use only when the disturbance can credibly be approximated as white in
    the closed-loop band. For coloured disturbances (NPD wind), prefer
    `state_covariance_freqdomain`.
    """
    Q = cl.B_w @ W @ cl.B_w.T
    P = solve_continuous_lyapunov(cl.A_cl, -Q)
    P = 0.5 * (P + P.T)
    return P


def position_std_dev(P: np.ndarray) -> tuple[float, float, float]:
    """Return (sigma_eta_n, sigma_eta_e, sigma_psi) from the 6x6 covariance."""
    return float(np.sqrt(P[0, 0])), float(np.sqrt(P[1, 1])), float(np.sqrt(P[2, 2]))


def position_covariance_2d(P: np.ndarray) -> np.ndarray:
    """Return the 2x2 horizontal-plane position covariance."""
    return P[0:2, 0:2]


def state_psd_freqdomain(
    A: np.ndarray,
    B_w: np.ndarray,
    S_F_funcs: list[Callable[[np.ndarray], np.ndarray]],
    omega: np.ndarray,
) -> np.ndarray:
    """Per-frequency state PSD matrix S_x(omega) = H S_F H^H, on a given grid.

    Parameters
    ----------
    A : (n_state, n_state) closed-loop state matrix.
    B_w : (n_state, n_force) disturbance input matrix.
    S_F_funcs : list of callables mu -> (n_force, n_force) one-sided
        force PSD matrices [N^2 / (rad/s)]. Summed across the list.
    omega : (n,) angular frequency grid [rad/s], strictly positive
        (PSD is one-sided).

    Returns
    -------
    S_x : (n, n_state, n_state) complex array of one-sided state PSD
        matrices. Diagonal entries S_x[i, k, k] are real and >=0;
        the matrix is Hermitian at each frequency.

    Notes
    -----
    This is the one-sided PSD: variance recovery is
    sigma_k^2 = integral_0^inf S_x[:, k, k].real d omega,
    which equals the kk diagonal of state_covariance_freqdomain_general.

    For the 1-D position spectra needed by extreme-value analysis,
    the caller picks an axis (e.g. surge index 0) and uses
    ``S_x[:, 0, 0].real`` as the scalar one-sided PSD.
    """
    omega = np.asarray(omega, dtype=float)
    n_state = A.shape[0]
    n_force = B_w.shape[1]
    I_n = np.eye(n_state)
    S_x = np.zeros((omega.size, n_state, n_state), dtype=complex)
    for i, w in enumerate(omega):
        H = np.linalg.solve(1j * w * I_n - A, B_w)             # (n_state, n_force)
        S_total = np.zeros((n_force, n_force))
        for S_F in S_F_funcs:
            S_total = S_total + S_F(w)
        S_x[i] = H @ S_total @ H.conj().T
    return S_x


def position_psd(
    cl: ClosedLoop,
    S_F_funcs: list[Callable[[np.ndarray], np.ndarray]],
    omega: np.ndarray,
) -> np.ndarray:
    """One-sided eta = (surge, sway, yaw) PSD matrix on a given omega grid.

    Convenience wrapper around :func:`state_psd_freqdomain`. Returns a
    complex (n, 3, 3) array; the real diagonals are the per-axis
    one-sided position spectra.

    Verified by integration: ``trapezoid(S_xx[:, k, k].real, omega) ==
    P_eta[k, k]`` from :func:`state_covariance_freqdomain` to within
    quadrature noise.
    """
    S_x = state_psd_freqdomain(cl.A_cl, cl.B_w, S_F_funcs, omega)
    return S_x[:, 0:3, 0:3]


def axis_psd(
    cl: ClosedLoop,
    S_F_funcs: list[Callable[[np.ndarray], np.ndarray]],
    axis_vector: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    """One-sided scalar PSD of a linear combination y = c^T eta.

    Useful for the gangway-base radial axis (footprint) and for the
    telescope-length axis (c = telescope_sensitivity).

    Parameters
    ----------
    cl : ClosedLoop.
    S_F_funcs : disturbance force PSDs (list of callables omega -> 3x3).
    axis_vector : (3,) real coefficients c such that y = c^T eta.
    omega : (n,) angular frequency grid [rad/s].

    Returns
    -------
    (n,) real array S_y(omega) [unit_y^2 / (rad/s)].
        S_y(omega) = c^T S_eta(omega) c, where S_eta is the 3x3
        one-sided position-state PSD matrix.
    """
    c = np.asarray(axis_vector, dtype=float).reshape(3)
    S_eta = position_psd(cl, S_F_funcs, omega)            # (n, 3, 3) complex
    # c^T S_eta(omega) c is real since S_eta is Hermitian.
    Sy = np.einsum("i,kij,j->k", c, S_eta, c).real
    return np.maximum(Sy, 0.0)
