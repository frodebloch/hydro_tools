"""Sanity-check: how does the realised sigma converge with realisation length?

Runs the canonical CSOV operating point for several T_op values, and
for each prints:
  * empirical sigma of r(t) and dL(t),
  * Bayesian posterior sigma_median + 90% credible interval,
  * model prior sigma for comparison.

Repeats over n_seeds independent realisations to expose run-to-run
variability vs T_op.

Run:
    python scripts/sigma_convergence_sweep.py
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
    summarise_intact_prior,
    load_pdstrip_rao,
    sigma_L_wave,
    npd_wind_gust_force_psd,
    current_variability_force_psd,
    slow_drift_force_psd_newman_pdstrip,
    BayesianSigmaEstimator,
    closed_loop_decorrelation_time,
    realise_vector_force_time_series,
    integrate_closed_loop_response,
    realise_wave_motion_6dof,
    radial_position_time_series,
    telescope_length_deviation_time_series,
)
from cqa.vessel import LinearVesselModel, CurrentForceModel
from cqa.controller import LinearDpController
from cqa.closed_loop import ClosedLoop
from cqa.psd import WindForceModel

PDSTRIP_PATH = Path.home() / "src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"


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

    rao_table = load_pdstrip_rao(PDSTRIP_PATH)
    sigma_L_wave_m = float(sigma_L_wave(joint, cfg, rao_table,
                                        Hs=Hs, Tp=Tp,
                                        theta_wave_rel=theta_rel).sigma_L_wave)

    vp, wp, cp = cfg.vessel, cfg.wind, cfg.current
    vessel = LinearVesselModel.from_config(vp)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D,
        omega_n=cfg.controller.omega_n, zeta=cfg.controller.zeta,
    )
    cl = ClosedLoop.build(vessel, controller)

    S_wind = npd_wind_gust_force_psd(WindForceModel(wp=wp, loa=vp.loa),
                                     Vw_mean, theta_rel)
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

    # Model-prior sigmas (independent of T_op via stationary spectrum).
    prior_ref = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint,
        T_op_s=20*60.0, sigma_L_wave=sigma_L_wave_m, Tp_wave_s=Tp,
    )
    sig_pos_prior = prior_ref.pos_sigma_m
    sig_gw_prior  = prior_ref.gw_sigma_slow_m

    print(f"Model-prior sigma_radial = {sig_pos_prior:.3f} m")
    print(f"Model-prior sigma_slow   = {sig_gw_prior:.3f} m")
    print(f"DP omega_n = {cfg.controller.omega_n_surge:.3f} rad/s "
          f"(period {2*np.pi/cfg.controller.omega_n_surge:.0f} s)")
    print()

    T_decorr_pos = closed_loop_decorrelation_time(cfg.controller, "position")
    T_decorr_gw  = closed_loop_decorrelation_time(cfg.controller, "sway")

    dt_s = 0.5
    omega_grid_lf = np.geomspace(1.0e-4, 0.6, 256)

    T_ops_min = [5, 15, 30, 60]
    n_seeds = 8

    header = (f"{'T_op':>7} {'n_periods':>10} {'n_eff_pos':>10}  "
              f"{'sig_r emp (mean+/-std)':>26}  "
              f"{'sig_r post (mean+/-std)':>27}  "
              f"{'sig_dL emp':>20}  {'sig_dL post':>20}")
    print(header)
    print("-" * len(header))

    for T_op_min in T_ops_min:
        T_op_s = float(T_op_min) * 60.0
        t = np.arange(0.0, T_op_s, dt_s)
        n_periods = T_op_s / (2*np.pi/cfg.controller.omega_n_surge)

        sig_r_emp = np.zeros(n_seeds)
        sig_dL_emp = np.zeros(n_seeds)
        sig_r_post = np.zeros(n_seeds)
        sig_dL_post = np.zeros(n_seeds)

        for s in range(n_seeds):
            rng = np.random.default_rng(20260504 + 100*s + T_op_min)
            F_lf = realise_vector_force_time_series(S_F_funcs, omega_grid_lf, t, rng)
            x_lf = integrate_closed_loop_response(cl, F_lf, t)
            xi_wf = realise_wave_motion_6dof(
                rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel,
                t=t, rng=rng,
            )
            r = radial_position_time_series(x_lf, xi_wf, cfg)
            dL = telescope_length_deviation_time_series(x_lf, xi_wf, joint, cfg)

            sig_r_emp[s]  = np.sqrt(np.mean(r * r))   # zero-mean assumption
            sig_dL_emp[s] = np.sqrt(np.mean(dL * dL))

            est_pos = BayesianSigmaEstimator(
                prior_sigma2=sig_pos_prior**2, T_decorr_s=T_decorr_pos,
                dt_s=1.0, window_s=T_op_s,
            )
            est_gw = BayesianSigmaEstimator(
                prior_sigma2=sig_gw_prior**2, T_decorr_s=T_decorr_gw,
                dt_s=1.0, window_s=T_op_s,
            )
            stride = int(round(1.0 / dt_s))
            for k in range(0, t.size, stride):
                est_pos.update(float(r[k]))
                est_gw.update(float(dL[k]))
            sig_r_post[s]  = est_pos.posterior().sigma_median
            sig_dL_post[s] = est_gw.posterior().sigma_median

        n_eff_pos = est_pos.n_eff  # same for all seeds at this T_op
        print(
            f"{T_op_min:>5} min "
            f"{n_periods:>10.1f} "
            f"{n_eff_pos:>10.1f}  "
            f"{sig_r_emp.mean():>10.3f} +/- {sig_r_emp.std():>5.3f} m       "
            f"{sig_r_post.mean():>10.3f} +/- {sig_r_post.std():>5.3f} m      "
            f"{sig_dL_emp.mean():>8.3f}+/-{sig_dL_emp.std():>5.3f} m  "
            f"{sig_dL_post.mean():>8.3f}+/-{sig_dL_post.std():>5.3f} m"
        )

    print()
    print(f"Reference (model prior): sigma_r = {sig_pos_prior:.3f} m, "
          f"sigma_dL_slow = {sig_gw_prior:.3f} m")


if __name__ == "__main__":
    main()
