"""Tests for the closed-loop covariance solver.

The cleanest sanity check is the 1-DOF mass-spring-damper analytic result:

    m x_ddot + d x_dot + k x = w(t),  E[w(t) w(s)] = W delta(t-s)
    => sigma_x^2 = W / (2 d k),  sigma_xdot^2 = W / (2 d m).

We embed this 1-DOF system in the 6-DOF state-space form and verify the
Lyapunov solution agrees within tight tolerance.
"""

from __future__ import annotations

import numpy as np

from cqa.closed_loop import lyapunov_position_covariance, ClosedLoop
from cqa.controller import LinearDpController
from cqa.vessel import LinearVesselModel


def test_lyapunov_1dof_analytic():
    # Single-DOF in surge: m=1e6, d=2e4, omega_n=0.1, zeta=0.7
    m = 1.0e6
    d = 2.0e4
    M = np.diag([m, 1e9, 1e12])  # huge sway/yaw to decouple
    D = np.diag([d, 1e7, 1e10])
    vessel = LinearVesselModel(M=M, D=D, name="test")
    omega_n = 0.1
    zeta = 0.7
    k = m * omega_n ** 2  # so that closed-loop k = m omega^2
    # Build controller via from_bandwidth:
    controller = LinearDpController.from_bandwidth(
        M, D, omega_n=(omega_n, 1e-3, 1e-3), zeta=(zeta, 0.7, 0.7)
    )
    cl = ClosedLoop.build(vessel, controller)
    # White-noise force intensity in surge only:
    W_force = 1.0e6  # N^2 s
    W = np.diag([W_force, 0.0, 0.0])
    P = lyapunov_position_covariance(cl, W)
    sigma_x_lyap = np.sqrt(P[0, 0])

    # Analytic: closed-loop is m x_ddot + (d + Kd) x_dot + Kp x = w
    # Effective damping d_eff = 2 zeta m omega_n  (by construction of Kd)
    # Effective stiffness k_eff = m omega_n^2
    d_eff = 2 * zeta * m * omega_n
    k_eff = m * omega_n ** 2
    sigma_x_ana = np.sqrt(W_force / (2.0 * d_eff * k_eff))

    rel_err = abs(sigma_x_lyap - sigma_x_ana) / sigma_x_ana
    assert rel_err < 1e-3, f"sigma_x_lyap={sigma_x_lyap}, sigma_x_ana={sigma_x_ana}, rel_err={rel_err}"


def test_freqdomain_matches_lyapunov_for_white_noise():
    """Freq-domain integration must agree with Lyapunov for a white-noise input.

    Convention check: for a one-sided PSD S_F(omega) [N^2/(rad/s)] in rad/s,
    var(x) = integral H(w) S_F(w) H^H dw  (no 1/pi factor).
    The equivalent Lyapunov white-noise intensity is W = pi * S0.
    """
    from cqa.closed_loop import state_covariance_freqdomain

    m = 1.0e7
    d = 1.0e5
    M = np.diag([m, 1e10, 1e13])
    D = np.diag([d, 1e8, 1e11])
    vessel = LinearVesselModel(M=M, D=D, name="white-noise-1dof")
    controller = LinearDpController.from_bandwidth(
        M, D, omega_n=(0.1, 1e-3, 1e-3), zeta=(0.7, 0.7, 0.7)
    )
    cl = ClosedLoop.build(vessel, controller)

    S0 = 1.0e8  # N^2/(rad/s)

    def S_F(w):
        out = np.zeros((3, 3))
        out[0, 0] = S0
        return out

    P_freq = state_covariance_freqdomain(
        cl, [S_F], omega_lo=1e-4, omega_hi=10.0, n_points=4096
    )
    W = np.diag([np.pi * S0, 0.0, 0.0])
    P_lyap = lyapunov_position_covariance(cl, W)

    rel_err = abs(P_freq[0, 0] - P_lyap[0, 0]) / P_lyap[0, 0]
    assert rel_err < 1e-2, f"freq={P_freq[0,0]}, lyap={P_lyap[0,0]}, rel_err={rel_err}"



def test_excursion_grows_with_wind():
    """Higher wind => larger 95% excursion."""
    from cqa import csov_default_config, excursion_polar

    cfg = csov_default_config()
    common = dict(Hs=2.0, Tp=8.0, Vc=0.5, n_directions=18)
    res_low = excursion_polar(cfg, Vw_mean=8.0, **common)
    res_high = excursion_polar(cfg, Vw_mean=18.0, **common)
    # The maximum semi-major over all directions should grow with wind.
    assert res_high.ellipse_semi_major.max() > res_low.ellipse_semi_major.max()


def test_excursion_grows_with_hs():
    from cqa import csov_default_config, excursion_polar

    cfg = csov_default_config()
    common = dict(Vw_mean=10.0, Tp=8.0, Vc=0.3, n_directions=18)
    res_low = excursion_polar(cfg, Hs=1.0, **common)
    res_high = excursion_polar(cfg, Hs=4.0, **common)
    assert res_high.ellipse_semi_major.max() > res_low.ellipse_semi_major.max()


# ---------------------------------------------------------------------------
# Position PSD / axis PSD consistency with integrated covariance
# ---------------------------------------------------------------------------


def _build_simple_closed_loop():
    """Build a simple 1-DOF-in-surge closed loop, return cl + a S_F."""
    m = 1.0e7
    d = 1.0e5
    M = np.diag([m, 1e10, 1e13])
    D = np.diag([d, 1e8, 1e11])
    vessel = LinearVesselModel(M=M, D=D, name="psd-test")
    controller = LinearDpController.from_bandwidth(
        M, D, omega_n=(0.1, 1e-3, 1e-3), zeta=(0.7, 0.7, 0.7)
    )
    cl = ClosedLoop.build(vessel, controller)
    S0 = 1.0e8
    def S_F(w):
        out = np.zeros((3, 3))
        out[0, 0] = S0
        return out
    return cl, S_F


def test_position_psd_integrates_to_state_covariance_diag():
    """The trapezoidal integral of S_xx[k, k].real over the omega grid
    must match the kk diagonal of state_covariance_freqdomain to within
    quadrature noise."""
    from cqa.closed_loop import position_psd, state_covariance_freqdomain

    cl, S_F = _build_simple_closed_loop()
    omega = np.logspace(-4, 1, 4096)

    S_eta = position_psd(cl, [S_F], omega)               # (n, 3, 3) complex
    var_surge_psd = float(np.trapezoid(S_eta[:, 0, 0].real, omega))

    P = state_covariance_freqdomain(cl, [S_F],
                                    omega_lo=1e-4, omega_hi=10.0,
                                    n_points=4096)
    assert np.isclose(var_surge_psd, P[0, 0], rtol=1e-3)


def test_axis_psd_matches_diagonal_for_unit_axis():
    """axis_psd with c = e_0 must equal the surge diagonal of position_psd."""
    from cqa.closed_loop import axis_psd, position_psd

    cl, S_F = _build_simple_closed_loop()
    omega = np.logspace(-4, 1, 1024)

    S_eta = position_psd(cl, [S_F], omega)
    S_axis = axis_psd(cl, [S_F], np.array([1.0, 0.0, 0.0]), omega)

    np.testing.assert_allclose(S_axis, S_eta[:, 0, 0].real, rtol=1e-12)


def test_axis_psd_linear_combination():
    """axis_psd of a sum of components is the bilinear form c^T S_eta c."""
    from cqa.closed_loop import axis_psd, position_psd

    cl, S_F = _build_simple_closed_loop()
    omega = np.logspace(-3, 0, 256)
    c = np.array([0.6, 0.8, 0.0])  # arbitrary unit vector in surge-sway plane

    S_axis = axis_psd(cl, [S_F], c, omega)
    S_eta = position_psd(cl, [S_F], omega)
    expected = np.einsum("i,kij,j->k", c, S_eta, c).real
    np.testing.assert_allclose(S_axis, np.maximum(expected, 0.0), rtol=1e-12)
    # Must be non-negative (one-sided PSD of a real-valued process).
    assert np.all(S_axis >= 0.0)


def test_axis_psd_integrates_to_axis_variance():
    """Integral of axis_psd equals c^T P_eta c."""
    from cqa.closed_loop import axis_psd, state_covariance_freqdomain

    cl, S_F = _build_simple_closed_loop()
    omega = np.logspace(-4, 1, 4096)
    c = np.array([1.0, 0.0, 0.0])

    S_axis = axis_psd(cl, [S_F], c, omega)
    var_psd = float(np.trapezoid(S_axis, omega))

    P = state_covariance_freqdomain(cl, [S_F],
                                    omega_lo=1e-4, omega_hi=10.0,
                                    n_points=4096)
    var_cov = float(c @ P[0:3, 0:3] @ c)
    assert np.isclose(var_psd, var_cov, rtol=1e-3)
