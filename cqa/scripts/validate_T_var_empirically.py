"""Empirically validate T_var on the ACTUAL closed-loop output.

The synthetic-spectrum test in tests/test_extreme_value.py checked the
spectral formula against an empirical autocovariance for a Gaussian-bump
PSD. This script does the same but on the production pipeline:

  1. Build the closed-loop output PSD `S_dL_slow(omega)` for the
     canonical CSOV operating point (the same PSD that
     summarise_intact_prior integrates).
  2. Realise dL_slow(t) for a long duration (60 min) via the production
     time-domain stack.
  3. Compute empirical R(tau) by FFT, integrate
     T_var_emp = (2/sigma^4) * int_0^infty R(tau)^2 dtau.
  4. Compare with the spectral
     T_var_spec = pi * int S^2 domega / m0^2.

Why dL_slow and not r(t)?
-------------------------
dL_slow(t) is itself a stationary zero-mean Gaussian process so the
autocovariance R(tau) is directly the quantity that enters the
Isserlis identity for Var(sum dL^2). r(t) = sqrt(x^2+y^2) is Rayleigh-
distributed; the sufficient statistic for sigma^2 is r^2 = x^2+y^2,
whose autocovariance is 2(R_x^2 + R_y^2). The cleaner univariate
check is on dL_slow.

This is the gold-standard sanity check: end-to-end production code +
empirical R(tau) measured on the actual realisation that drives the
estimator.

Run:
    python scripts/validate_T_var_empirically.py
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
    realise_vector_force_time_series,
    integrate_closed_loop_response,
    realise_wave_motion_6dof,
    telescope_length_deviation_time_series,
    summarise_intact_prior,
    sigma_L_wave,
    variance_decorrelation_time_from_psd,
)
from cqa.vessel import LinearVesselModel, CurrentForceModel
from cqa.controller import LinearDpController
from cqa.closed_loop import ClosedLoop, axis_psd
from cqa.psd import WindForceModel
from cqa.operator_view import telescope_sensitivity

PDSTRIP_PATH = Path.home() / "src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"


def empirical_T_var(x: np.ndarray, dt: float, n_lag_max: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute empirical T_var = (2/sigma^4) int R^2 dtau on a single
    realisation, using unbiased autocovariance (FFT-based)."""
    N = x.size
    xc = x - x.mean()
    F = np.fft.rfft(xc, n=2 * N)
    R_full = np.fft.irfft(F * np.conj(F))[:N]
    counts = N - np.arange(N)
    R = R_full / counts            # unbiased autocovariance
    sigma2 = float(R[0])
    n_lag = min(n_lag_max, N // 4)  # unbiased estimator OK for lag < N/4
    tau = np.arange(n_lag) * dt
    R_trunc = R[:n_lag]
    int_R2 = float(np.trapezoid(R_trunc * R_trunc, tau))
    T_var_emp = 2.0 * int_R2 / (sigma2 ** 2)
    return T_var_emp, tau, R_trunc


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
        omega_n=cfg.controller.omega_n, zeta=cfg.controller.zeta,
    )
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

    # --- Spectral T_var via the production-grade integration grid ---
    wave = sigma_L_wave(joint, cfg, rao_table, Hs=Hs, Tp=Tp,
                       theta_wave_rel=theta_rel)
    sigma_L_wave_m = float(wave.sigma_L_wave)

    prior = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint,
        T_op_s=300.0, sigma_L_wave=sigma_L_wave_m, Tp_wave_s=Tp,
    )
    T_var_spec_dL = float(prior.gw_T_decorr_var_s)
    sigma_slow_spec = float(prior.gw_sigma_slow_m)
    print("=" * 72)
    print("Empirical T_var validation on the production closed-loop output")
    print("=" * 72)
    print(f"Operating point: Hs={Hs}, Tp={Tp}, theta={np.degrees(theta_rel):.0f} deg, "
          f"Vw={Vw_mean}, Vc={Vc}")
    print(f"\nSpectral (from prior PSD on logspace(-4,0,1024)):")
    print(f"  sigma_dL_slow      = {sigma_slow_spec:.4f} m")
    print(f"  T_var_dL_slow_spec = {T_var_spec_dL:.2f} s   "
          f"(this is what BayesianSigmaEstimator now uses)")

    # --- Realise the slow telescope channel for a long duration -----------
    # Use only LF disturbances (drop the 1st-order WF channel for this
    # check; including it would mix two bands and make the empirical R(tau)
    # bimodal).
    rng = np.random.default_rng(2026)
    T_total_s = 60.0 * 60.0       # 60 min: ~24 closed-loop periods
    dt = 0.5
    t = np.arange(0.0, T_total_s, dt)
    omega_grid_lf = np.linspace(1.0e-4, 0.6, 1024)
    print(f"\nRealising LF response: T_total={T_total_s/60:.0f} min, "
          f"dt={dt}s, N={t.size}, n_freq={omega_grid_lf.size}")
    F_lf = realise_vector_force_time_series(S_F_funcs, omega_grid_lf, t, rng)
    x_lf = integrate_closed_loop_response(cl, F_lf, t)

    # Project to the telescope-length channel (LF only, no wave-band
    # 6-DOF motion).
    xi_zero = np.zeros((6, t.size))
    dL_slow = telescope_length_deviation_time_series(x_lf, xi_zero, joint, cfg)

    sigma_emp = float(np.sqrt(np.mean(dL_slow * dL_slow)))
    print(f"\nEmpirical (single 60-min realisation):")
    print(f"  sigma_dL_emp        = {sigma_emp:.4f} m  "
          f"(model prior: {sigma_slow_spec:.4f} m)")

    # --- Empirical T_var --------------------------------------------------
    n_lag_max = int(800.0 / dt)   # 800 s of lag is plenty (T_var ~ 100 s)
    T_var_emp_dL, tau, R_trunc = empirical_T_var(dL_slow, dt, n_lag_max)
    print(f"  T_var_dL_emp        = {T_var_emp_dL:.2f} s   "
          f"(spec: {T_var_spec_dL:.2f} s, ratio "
          f"{T_var_emp_dL/T_var_spec_dL:.3f})")

    # Sanity: the autocovariance should be ~exponential-ish with the right
    # scale. Print a few points.
    print(f"\n  R(0)        = {R_trunc[0]:.4f}    (sigma^2 = {sigma_emp**2:.4f})")
    print(f"  R(50 s)/R(0)  = {R_trunc[int(50/dt)]/R_trunc[0]:+.3f}")
    print(f"  R(100 s)/R(0) = {R_trunc[int(100/dt)]/R_trunc[0]:+.3f}")
    print(f"  R(200 s)/R(0) = {R_trunc[int(200/dt)]/R_trunc[0]:+.3f}")
    print(f"  R(400 s)/R(0) = {R_trunc[int(400/dt)]/R_trunc[0]:+.3f}")

    # Check against the time-axis (lag) integral of S^2.
    # Direct computation as a sanity replicate of variance_decorrelation_time_from_psd
    # on the SAME omega grid as used by summarise_intact_prior:
    omega_check = np.logspace(-4, 0, 1024)
    c_L = telescope_sensitivity(joint, cfg.gangway)
    S_dL_check = axis_psd(cl, S_F_funcs, c_L, omega_check)
    T_var_check = variance_decorrelation_time_from_psd(S_dL_check, omega_check)
    print(f"\n  T_var spectral re-check (replicate logspace grid) = "
          f"{T_var_check:.2f} s")

    print("\n" + "=" * 72)
    if abs(T_var_emp_dL / T_var_spec_dL - 1.0) < 0.20:
        print(f"PASS: empirical {T_var_emp_dL:.1f} s vs spectral "
              f"{T_var_spec_dL:.1f} s within 20%")
    else:
        print(f"FAIL or surprising: empirical {T_var_emp_dL:.1f} s vs spectral "
              f"{T_var_spec_dL:.1f} s, ratio "
              f"{T_var_emp_dL/T_var_spec_dL:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
