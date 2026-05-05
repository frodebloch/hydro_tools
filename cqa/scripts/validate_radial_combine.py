"""Tier B coverage validation for `combine_radial_posterior`.

What this checks
----------------
``combine_radial_posterior`` claims to produce an equal-tail credible
interval on the radial scale ``sigma_R = sqrt(sigma_x^2 + sigma_y^2)``
(and on ``E[|R|]``) given two independent per-axis InvGamma posteriors.
For the CI to be operationally meaningful we need:

    P(true sigma_R in posterior 90% CI)  ~  0.90

across many independent realisations. This script measures that
probability empirically and decides PASS / FAIL on a binomial test.

Pipeline
--------
For each of M seeds:

    1. Realise 5 min of LF disturbance + WF wave motion at the
       canonical CSOV operating point.
    2. Project to (dx, dy) via ``base_position_xy_time_series``.
    3. Feed (dx, dy) into two ``BayesianSigmaEstimator`` instances at
       1 Hz (the operator-facing rate).
    4. Combine -> ``RadialPosterior`` with 90% CI on sigma_R and E[|R|].
    5. Record:
         a. Whether the SPECTRAL sigma_R (computed once, ground truth
            for the population) lies in the 90% CI on sigma_R.
         b. Whether the SPECTRAL E[|R|] (Hoyt mean from the spectral
            sigma_x, sigma_y) lies in the 90% CI on E[|R|].

Pass criterion
--------------
For M=200 seeds at a 90% credible level the binomial 95% acceptance
window for "true coverage = 0.90" is approximately [85%, 94%]. We use
``scipy.stats.binomtest`` with the two-sided p-value and pass if
p > 0.05 (cannot reject calibration).

Failure modes this catches
--------------------------
* MC bias in the combine (e.g. wrong InvGamma parameterisation -- would
  systematically inflate or deflate the CI).
* Per-axis posterior mis-calibration (would propagate to the radial CI).
* Hidden cov(dx, dy) ~ 0 violation that the independence assumption
  ignores (would narrow the empirical spread vs the model spread,
  showing as over-coverage on the variance and under-coverage on E[|R|]
  -- or vice versa).
* Wrong spectral integration (would shift the truth value relative to
  the data-conditioned posterior centre).

Run:
    python scripts/validate_radial_combine.py [--m-seeds 200]

Runtime ~ 2-3 minutes for M=200 (dominated by closed-loop ODE).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cqa import (
    csov_default_config,
    GangwayJointState,
    summarise_intact_prior,
    load_pdstrip_rao,
    sigma_L_wave,
    npd_wind_gust_force_psd,
    current_variability_force_psd,
    slow_drift_force_psd_newman_pdstrip,
    BayesianSigmaEstimator,
    combine_radial_posterior,
    realise_vector_force_time_series,
    integrate_closed_loop_response,
    realise_wave_motion_6dof,
    base_position_xy_time_series,
)
from cqa.vessel import LinearVesselModel, CurrentForceModel
from cqa.controller import LinearDpController
from cqa.closed_loop import ClosedLoop
from cqa.psd import WindForceModel

PDSTRIP_PATH = Path.home() / "src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"


def build_operating_point():
    """Set up the canonical CSOV operating point. Returns everything
    the per-seed loop needs: (cl, S_F_funcs, rao_table, joint, cfg,
    Hs, Tp, theta_rel, sigma_x_true, sigma_y_true, sigma_L_wave_m,
    T_decorr_pos_x, T_decorr_pos_y).
    """
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

    wave = sigma_L_wave(joint, cfg, rao_table, Hs=Hs, Tp=Tp,
                       theta_wave_rel=theta_rel)
    sigma_L_wave_m = float(wave.sigma_L_wave)

    # Spectral truth: per-axis sigmas from summarise_intact_prior.
    prior = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint,
        T_op_s=300.0, sigma_L_wave=sigma_L_wave_m, Tp_wave_s=Tp,
    )
    return dict(
        cfg=cfg, joint=joint, rao_table=rao_table,
        cl=cl, S_F_funcs=S_F_funcs,
        Hs=Hs, Tp=Tp, theta_rel=theta_rel,
        sigma_L_wave_m=sigma_L_wave_m,
        sigma_x_true=float(prior.pos_sigma_x_m),
        sigma_y_true=float(prior.pos_sigma_y_m),
        T_decorr_pos_x=float(prior.pos_T_decorr_var_x_s),
        T_decorr_pos_y=float(prior.pos_T_decorr_var_y_s),
    )


def hoyt_mean_R(sigma_x: float, sigma_y: float, n_mc: int = 200_000,
                rng: np.random.Generator = None) -> float:
    """E[|R|] for R = sqrt(X^2+Y^2), X~N(0,sx), Y~N(0,sy), via MC.
    Equal-variance limit: sigma * sqrt(pi/2). 200k draws gives ~0.2%
    SE which is negligible vs the posterior CI half-width.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    X = rng.standard_normal(n_mc) * sigma_x
    Y = rng.standard_normal(n_mc) * sigma_y
    return float(np.mean(np.sqrt(X * X + Y * Y)))


def run_one_seed(seed: int, op, t, dt_s, omega_grid_lf, obs_idx, dt_obs,
                 T_op_s, credible: float):
    """Single realisation: realise -> project -> estimate -> combine.
    Returns (sigma_R_lo, sigma_R_hi, ER_lo, ER_hi, sigma_R_med, ER_med).
    """
    rng = np.random.default_rng(seed)
    F_lf = realise_vector_force_time_series(
        op["S_F_funcs"], omega_grid_lf, t, rng,
    )
    x_lf = integrate_closed_loop_response(op["cl"], F_lf, t)
    xi_wf = realise_wave_motion_6dof(
        op["rao_table"], Hs=op["Hs"], Tp=op["Tp"],
        theta_wave_rel=op["theta_rel"],
        t=t, rng=rng,
    )
    dx, dy = base_position_xy_time_series(x_lf, xi_wf, op["cfg"])

    est_x = BayesianSigmaEstimator(
        prior_sigma2=op["sigma_x_true"] ** 2,
        T_decorr_s=op["T_decorr_pos_x"],
        dt_s=dt_obs, window_s=T_op_s,
    )
    est_y = BayesianSigmaEstimator(
        prior_sigma2=op["sigma_y_true"] ** 2,
        T_decorr_s=op["T_decorr_pos_y"],
        dt_s=dt_obs, window_s=T_op_s,
    )
    for k in obs_idx:
        est_x.update(float(dx[k]))
        est_y.update(float(dy[k]))
    px = est_x.posterior(credible=credible)
    py = est_y.posterior(credible=credible)

    # Use a fresh seeded rng for the MC combine so coverage doesn't
    # depend on the realisation rng's leftover state. Seed offset by
    # 10**6 to decouple from the realisation seed deterministically.
    rad = combine_radial_posterior(
        px, py, credible=credible, n_mc=2000,
        rng=np.random.default_rng(seed + 1_000_000),
    )
    return (
        rad.sigma_R_lo, rad.sigma_R_hi,
        rad.expected_R_lo, rad.expected_R_hi,
        rad.sigma_R_median, rad.expected_R_median,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--m-seeds", type=int, default=200,
                        help="Number of independent realisations (default 200).")
    parser.add_argument("--credible", type=float, default=0.90,
                        help="Credible level the CI claims (default 0.90).")
    args = parser.parse_args()

    M = int(args.m_seeds)
    credible = float(args.credible)

    print("=" * 72)
    print(f"Tier B coverage validation: combine_radial_posterior")
    print(f"  M={M} seeds, credible level = {credible*100:.0f}%")
    print("=" * 72)

    op = build_operating_point()
    sigma_R_true = float(np.sqrt(op["sigma_x_true"] ** 2
                                 + op["sigma_y_true"] ** 2))
    ER_true = hoyt_mean_R(op["sigma_x_true"], op["sigma_y_true"],
                          n_mc=400_000,
                          rng=np.random.default_rng(99))

    print(f"\nSpectral truth (canonical CSOV operating point):")
    print(f"  sigma_x_true = {op['sigma_x_true']:.4f} m")
    print(f"  sigma_y_true = {op['sigma_y_true']:.4f} m")
    print(f"  sigma_R_true = {sigma_R_true:.4f} m  "
          f"(= sqrt(sigma_x^2+sigma_y^2))")
    print(f"  E[|R|]_true  = {ER_true:.4f} m  "
          f"(Hoyt mean, ratio sx/sy = {op['sigma_x_true']/op['sigma_y_true']:.3f})")

    # Realisation grid (same as the demo).
    T_op_s = 5.0 * 60.0
    dt_s = 0.5
    t = np.arange(0.0, T_op_s, dt_s)
    omega_grid_lf = np.geomspace(1.0e-4, 0.6, 256)
    obs_stride = 2  # 1 Hz observer
    dt_obs = dt_s * obs_stride
    obs_idx = np.arange(0, t.size, obs_stride)

    print(f"\nPer-seed setup: T_op={T_op_s/60:.0f} min, dt={dt_s} s, "
          f"observer at 1 Hz")
    print(f"\nRunning {M} seeds...")

    in_ci_sigma = np.zeros(M, dtype=bool)
    in_ci_ER = np.zeros(M, dtype=bool)
    sigma_R_meds = np.zeros(M)
    ER_meds = np.zeros(M)
    sigma_R_widths = np.zeros(M)
    ER_widths = np.zeros(M)

    t0 = time.time()
    for i in range(M):
        seed = 10_000 + i
        s_lo, s_hi, e_lo, e_hi, s_med, e_med = run_one_seed(
            seed, op, t, dt_s, omega_grid_lf, obs_idx, dt_obs,
            T_op_s, credible,
        )
        in_ci_sigma[i] = (s_lo <= sigma_R_true <= s_hi)
        in_ci_ER[i] = (e_lo <= ER_true <= e_hi)
        sigma_R_meds[i] = s_med
        ER_meds[i] = e_med
        sigma_R_widths[i] = s_hi - s_lo
        ER_widths[i] = e_hi - e_lo
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (M - (i + 1)) / rate
            print(f"  [{i+1:4d}/{M}]  elapsed {elapsed:5.1f}s  "
                  f"rate {rate:4.1f}/s  ETA {eta:5.1f}s")

    elapsed_total = time.time() - t0
    print(f"\nDone in {elapsed_total:.1f} s "
          f"({elapsed_total/M*1000:.0f} ms/seed)")

    # --- Coverage analysis -------------------------------------------------
    n_in_sigma = int(in_ci_sigma.sum())
    n_in_ER = int(in_ci_ER.sum())
    cov_sigma = n_in_sigma / M
    cov_ER = n_in_ER / M

    bt_sigma = binomtest(n_in_sigma, M, p=credible)
    bt_ER = binomtest(n_in_ER, M, p=credible)

    print("\n" + "=" * 72)
    print(f"Coverage results (target = {credible*100:.0f}%)")
    print("=" * 72)

    def _verdict(bt, cov):
        # Two-sided binomial test against H0: true coverage == credible.
        # Pass if we cannot reject H0 at alpha = 0.05.
        if bt.pvalue > 0.05:
            return f"PASS  (cannot reject calibration, p={bt.pvalue:.3f})"
        elif cov < credible:
            return (f"FAIL  (under-cover, "
                    f"p={bt.pvalue:.4f} -- CI too narrow / posterior over-confident)")
        else:
            return (f"FAIL  (over-cover, "
                    f"p={bt.pvalue:.4f} -- CI too wide / posterior under-confident)")

    print(f"\nsigma_R coverage:")
    print(f"  {n_in_sigma}/{M} seeds had sigma_R_true in 90% CI  "
          f"-> empirical coverage = {cov_sigma*100:.1f}%")
    print(f"  Wilson 95% CI on coverage: "
          f"[{bt_sigma.proportion_ci(0.95).low*100:.1f}%, "
          f"{bt_sigma.proportion_ci(0.95).high*100:.1f}%]")
    print(f"  median CI width = {np.median(sigma_R_widths):.3f} m  "
          f"(median posterior = {np.median(sigma_R_meds):.3f} m vs "
          f"truth {sigma_R_true:.3f} m)")
    print(f"  median bias     = "
          f"{np.median(sigma_R_meds) - sigma_R_true:+.4f} m")
    print(f"  -> {_verdict(bt_sigma, cov_sigma)}")

    print(f"\nE[|R|] coverage:")
    print(f"  {n_in_ER}/{M} seeds had E[|R|]_true in 90% CI  "
          f"-> empirical coverage = {cov_ER*100:.1f}%")
    print(f"  Wilson 95% CI on coverage: "
          f"[{bt_ER.proportion_ci(0.95).low*100:.1f}%, "
          f"{bt_ER.proportion_ci(0.95).high*100:.1f}%]")
    print(f"  median CI width = {np.median(ER_widths):.3f} m  "
          f"(median posterior = {np.median(ER_meds):.3f} m vs "
          f"truth {ER_true:.3f} m)")
    print(f"  median bias     = "
          f"{np.median(ER_meds) - ER_true:+.4f} m")
    print(f"  -> {_verdict(bt_ER, cov_ER)}")

    print("\n" + "=" * 72)
    overall_pass = (bt_sigma.pvalue > 0.05) and (bt_ER.pvalue > 0.05)
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
