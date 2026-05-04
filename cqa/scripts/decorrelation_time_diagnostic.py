"""Diagnostic: which decorrelation time is right for Bartlett ESS?

Question
--------
`BayesianSigmaEstimator` uses ``T_decorr = 1/(zeta*omega_n)`` to compute
the effective sample size

    n_eff = n_raw * dt / max(dt, T_decorr).

For the canonical CSOV operating point with omega_n=0.060, zeta=1 this
gives T_decorr = 16.7 s, hence n_eff = 16 over a 5-min, 1-Hz sliding
window. The resulting posterior 90% CI on sigma is ~30% half-width,
which the user finds suspiciously narrow.

The suspect mechanism: 1/(zeta*omega_n) is the closed-loop *impulse-
response* decorrelation time under broadband white-noise input. But the
actual driving force has very narrowband-LF content (current
variability with tau=600 s + slow-drift below ~0.15 rad/s), so the
output autocovariance R(tau) decays much more slowly than the loop
impulse response.

The right ESS decorrelation time for a stationary Gaussian process X(t)
sampled at dt is the *integrated autocorrelation time*

    T_int = (1/R(0)) * integral_{-infty}^{infty} R(tau) dtau
          = (2*pi/m0) * S_X(omega=0)

(where R(0) = m0 = sigma^2 and S_X is the one-sided PSD; the factor
2*pi appears because S(omega=0) [unit^2/(rad/s)] integrates to 2*m0
over [0, infty] as part of m0). For variance estimation specifically
(quadratic functional) the right scale is

    T_int_var = (2/sigma^4) * integral_0^infty R(tau)^2 dtau
              = (2*pi / (2*m0^2)) * integral_0^infty S_X(omega)^2 domega / (2*pi)
                               (Wiener-Khinchin / Plancherel)
              = (1/m0^2) * integral_0^infty S_X(omega)^2 domega.

We compute several candidates and compare on the radial-position and
telescope-slow channels for the demo operating point.

Run:
    python scripts/decorrelation_time_diagnostic.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cqa import (
    csov_default_config,
    GangwayJointState,
    load_pdstrip_rao,
    npd_wind_gust_force_psd,
    current_variability_force_psd,
    slow_drift_force_psd_newman_pdstrip,
    closed_loop_decorrelation_time,
)
from cqa.closed_loop import ClosedLoop, axis_psd
from cqa.controller import LinearDpController
from cqa.vessel import LinearVesselModel, CurrentForceModel
from cqa.psd import WindForceModel
from cqa.extreme_value import spectral_moments, zero_upcrossing_rate
from cqa.operator_view import telescope_sensitivity

PDSTRIP_PATH = Path.home() / "src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"


def autocorr_integrated_time(S_omega: np.ndarray, omega: np.ndarray) -> float:
    """T_int = (1/m0) * R(0)-normalised time integral of R(tau).

    For a one-sided real PSD S(omega) with m0 = integral_0^infty S domega,
    R(tau) = integral_0^infty S(omega) cos(omega*tau) domega, so
    R(0) = m0 and integral_{-infty}^{infty} R(tau) dtau = 2*pi*S(0).
    Hence T_int = 2*pi * S(0) / m0.
    """
    m0 = float(np.trapezoid(S_omega, omega))
    # Linear extrapolation of S to omega=0 from the two lowest points.
    if omega[0] <= 1e-9:
        S0 = float(S_omega[0])
    else:
        # Linear extrapolation in linear (omega, S) space.
        slope = (S_omega[1] - S_omega[0]) / (omega[1] - omega[0])
        S0 = float(S_omega[0] - slope * omega[0])
        S0 = max(S0, 0.0)
    return 2.0 * np.pi * S0 / m0 if m0 > 0 else float("inf")


def autocorr_squared_integrated_time(
    S_omega: np.ndarray, omega: np.ndarray
) -> float:
    """Variance-estimator decorrelation time.

    For variance estimation the relevant ESS uses

        T_int_var = (2/sigma^4) * integral_0^infty R(tau)^2 dtau
                  = (1/m0^2) * integral_0^infty S(omega)^2 domega

    by Plancherel (one-sided convention, since R is even). This is the
    Bartlett scale that should appear in n_eff for a posterior on
    sigma^2.
    """
    m0 = float(np.trapezoid(S_omega, omega))
    if m0 <= 0.0:
        return float("inf")
    int_S2 = float(np.trapezoid(S_omega * S_omega, omega))
    return int_S2 / (m0 * m0)


def report_channel(name: str, S: np.ndarray, omega: np.ndarray,
                   T_decorr_controller: float) -> None:
    m = spectral_moments(S, omega, [0, 1, 2])
    m0, m1, m2 = m[0], m[1], m[2]
    sigma = float(np.sqrt(m0))
    nu0 = float(zero_upcrossing_rate(S, omega))   # [Hz]
    omega_mean = m1 / m0 if m0 > 0 else 0.0
    omega_peak = float(omega[int(np.argmax(S))])
    T_period_mean = 2.0 * np.pi / omega_mean if omega_mean > 0 else float("inf")
    T_inter_up = 1.0 / nu0 if nu0 > 0 else float("inf")
    T_int = autocorr_integrated_time(S, omega)
    T_int_var = autocorr_squared_integrated_time(S, omega)

    print(f"\n--- {name} ---")
    print(f"  sigma                 = {sigma:.4f} m")
    print(f"  m0, m1, m2            = {m0:.4e}, {m1:.4e}, {m2:.4e}")
    print(f"  omega_peak            = {omega_peak:.4f} rad/s "
          f"(period {2*np.pi/omega_peak if omega_peak>0 else float('inf'):.1f} s)")
    print(f"  omega_mean = m1/m0    = {omega_mean:.4f} rad/s "
          f"(mean period {T_period_mean:.1f} s)")
    print(f"  nu_0+      = sqrt(m2/m0)/(2pi) = {nu0:.4f} Hz "
          f"(1/nu_0+ = {T_inter_up:.1f} s)")
    print(f"  T_decorr (controller, 1/(zeta*omega_n)) = "
          f"{T_decorr_controller:.1f} s   <-- currently used")
    print(f"  T_int   = 2*pi*S(0)/m0          = {T_int:.1f} s "
          f"(integrated autocorr time, mean estimator ESS)")
    print(f"  T_int_var = int S^2 / m0^2      = {T_int_var:.1f} s "
          f"(VARIANCE-estimator ESS scale; this is what the posterior "
          f"on sigma^2 should use)")


def main() -> None:
    cfg = csov_default_config()
    Vw_mean, Hs, Tp, Vc = 14.0, 2.8, 9.0, 0.5
    theta_rel = np.radians(30.0)

    gw = cfg.gangway
    L0 = 0.5 * (gw.telescope_min + gw.telescope_max)
    joint = GangwayJointState(
        h=gw.rotation_centre_height_above_base,
        alpha_g=-np.pi / 2.0, beta_g=0.0, L=L0,
    )

    if not PDSTRIP_PATH.exists():
        raise SystemExit(f"pdstrip data not found: {PDSTRIP_PATH}")
    rao_table = load_pdstrip_rao(PDSTRIP_PATH)

    vp, wp, cp = cfg.vessel, cfg.wind, cfg.current
    vessel = LinearVesselModel.from_config(vp)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D,
        omega_n=cfg.controller.omega_n,
        zeta=cfg.controller.zeta,
    )
    omega_n_surge = cfg.controller.omega_n_surge
    omega_n_sway = cfg.controller.omega_n_sway
    zeta_surge = cfg.controller.zeta_surge
    zeta_sway = cfg.controller.zeta_sway
    cl = ClosedLoop.build(vessel, controller)

    wind_model = WindForceModel(wp=wp, loa=vp.loa)
    S_wind = npd_wind_gust_force_psd(wind_model, Vw_mean, theta_rel)
    S_drift = slow_drift_force_psd_newman_pdstrip(
        rao_table=rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel,
    )
    current_model = CurrentForceModel(
        cp=cp, lateral_area_underwater=vp.lpp * vp.draft,
        frontal_area_underwater=vp.beam * vp.draft, loa=vp.loa,
    )
    F0 = current_model.force(Vc, theta_rel)
    dFdVc = 2.0 * F0 / Vc
    S_curr = current_variability_force_psd(dFdVc, sigma_Vc=0.1, tau=600.0)
    S_F_funcs = [S_wind, S_drift, S_curr]

    # Wide low-frequency grid: 0 ... 0.6 rad/s, dense near 0 to capture
    # the current-variability tail (corner at 1/600 = 1.67e-3 rad/s).
    omega = np.unique(np.concatenate([
        np.linspace(1e-5, 1e-2, 200),
        np.linspace(1e-2, 6e-1, 800),
    ]))

    base_x_b, base_y_b, _ = cfg.gangway.base_position_body
    c_base_x = np.array([1.0, 0.0, -base_y_b])
    c_base_y = np.array([0.0, 1.0,  base_x_b])
    c_L = telescope_sensitivity(joint, cfg.gangway)

    S_x = axis_psd(cl, S_F_funcs, c_base_x, omega)
    S_y = axis_psd(cl, S_F_funcs, c_base_y, omega)
    S_L = axis_psd(cl, S_F_funcs, c_L, omega)

    T_pos = closed_loop_decorrelation_time(cfg.controller, "position")
    T_sway = closed_loop_decorrelation_time(cfg.controller, "sway")
    T_surge = closed_loop_decorrelation_time(cfg.controller, "surge")

    print("=" * 72)
    print("Decorrelation-time diagnostic for the CSOV demo operating point")
    print("=" * 72)
    print(f"controller: omega_n_surge={omega_n_surge:.4f} rad/s, "
          f"zeta_surge={zeta_surge:.2f};  "
          f"omega_n_sway={omega_n_sway:.4f}, zeta_sway={zeta_sway:.2f}")
    print(f"  closed-loop response time 1/(zeta*omega_n) "
          f"= {1.0/(zeta_surge*omega_n_surge):.1f} s (surge), "
          f"{1.0/(zeta_sway*omega_n_sway):.1f} s (sway)")

    report_channel("S_base_x  (surge-base-position channel)", S_x, omega, T_surge)
    report_channel("S_base_y  (sway-base-position channel)",  S_y, omega, T_sway)
    report_channel("S_L       (telescope-length slow channel)", S_L, omega, T_sway)

    print("\n" + "=" * 72)
    print("Interpretation")
    print("=" * 72)
    print("""
For a posterior on sigma^2 driven by S = sum_i x_i^2, the variance of
S is set by integral_0^infty R(tau)^2 dtau, NOT by integral_0^infty
R(tau) dtau. The right Bartlett ESS for variance estimation is
therefore

    n_eff_var = T_window / T_int_var,    T_int_var = int S^2 / m0^2.

For broadband forcing T_int_var ~ 1/(zeta*omega_n), recovering the
controller-bandwidth scale. For narrowband forcing T_int_var grows as
the bandwidth narrows: the ESS should drop accordingly. The numbers
above tell us by how much.""")


if __name__ == "__main__":
    main()
