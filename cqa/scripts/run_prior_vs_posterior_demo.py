"""Prior vs data-conditioned posterior intact-prior panel.

Question 4 of the operator framework: "after observing the actual
position / telescope channels for a few minutes, how does the predicted
operability shift?"

Pipeline
--------
1. Fix the canonical CSOV operating point (bow quartering 30 deg, Vw=14,
   Hs=2.8, Tp=9, Vc=0.5).

2. Build the closed loop and the wind / drift / current force PSDs in
   the usual cqa way.

3. Realise the disturbances in the time domain (Shinozuka harmonic
   superposition) and integrate the closed-loop ODE to get the
   low-frequency state x_lf(t). In parallel, realise the 6-DOF wave-
   frequency body motion xi_wf(t) from the pdstrip RAOs.

4. Project both onto the two operator channels:
       r(t)    -- radial gangway-base position deviation [m]
       dL(t)   -- telescope-length deviation [m]
   These are the SAME channels the bridge sensors would expose.

5. Feed r(t) and dL(t) into two BayesianSigmaEstimator instances at the
   operator-facing rate (1 Hz) for the operator-facing window (5 min).
   The estimators carry their cqa-derived priors (sigma from the
   closed-loop spectrum) and decorrelation times from the controller.

6. Build TWO IntactPriorSummary panels:
       - "prior"     = pure model (no posterior override).
       - "posterior" = same model SHAPE (nu_0+, q) but LEVEL replaced
                       by the data-conditioned sigma_median from each
                       estimator.

7. Render side-by-side as a 2x2 figure (top row = prior, bottom row =
   posterior) and save to scripts/csov_prior_vs_posterior.png.

Run:
    python scripts/run_prior_vs_posterior_demo.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cqa import (
    csov_default_config,
    GangwayJointState,
    summarise_intact_prior,
    plot_intact_prior,
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
    predictive_running_max_quantile,
    bandsplit_lowpass,
    variance_decorrelation_time_from_psd,
)
from cqa.vessel import LinearVesselModel, CurrentForceModel
from cqa.controller import LinearDpController
from cqa.closed_loop import ClosedLoop
from cqa.psd import WindForceModel

PDSTRIP_PATH = Path.home() / "src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"


def main() -> None:
    cfg = csov_default_config()

    # --- Canonical CSOV operating point --------------------------------
    Vw_mean = 14.0
    Hs = 2.8
    Tp = 9.0
    Vc = 0.5
    theta_rel = np.radians(30.0)  # bow quartering, port

    gw = cfg.gangway
    L0 = 0.5 * (gw.telescope_min + gw.telescope_max)
    joint = GangwayJointState(
        h=gw.rotation_centre_height_above_base,
        alpha_g=-np.pi / 2.0,
        beta_g=0.0,
        L=L0,
    )

    # --- pdstrip RAO + sigma_L_wave ------------------------------------
    if not PDSTRIP_PATH.exists():
        raise SystemExit(
            f"pdstrip data not found at {PDSTRIP_PATH}; this demo requires it."
        )
    rao_table = load_pdstrip_rao(PDSTRIP_PATH)
    wave = sigma_L_wave(joint, cfg, rao_table, Hs=Hs, Tp=Tp,
                        theta_wave_rel=theta_rel)
    sigma_L_wave_m = float(wave.sigma_L_wave)
    print(f"sigma_L_wave (1st-order) = {sigma_L_wave_m*100:.1f} cm")

    # --- Build closed loop + LF force PSDs -----------------------------
    vp = cfg.vessel
    wp = cfg.wind
    cp = cfg.current

    vessel = LinearVesselModel.from_config(vp)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D,
        omega_n=cfg.controller.omega_n,
        zeta=cfg.controller.zeta,
    )
    cl = ClosedLoop.build(vessel, controller)

    wind_model = WindForceModel(wp=wp, loa=vp.loa)
    S_wind = npd_wind_gust_force_psd(wind_model, Vw_mean, theta_rel)
    S_drift = slow_drift_force_psd_newman_pdstrip(
        rao_table=rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel,
    )
    current_model = CurrentForceModel(
        cp=cp,
        lateral_area_underwater=vp.lpp * vp.draft,
        frontal_area_underwater=vp.beam * vp.draft,
        loa=vp.loa,
    )
    if Vc > 1e-9:
        F0 = current_model.force(Vc, theta_rel)
        dFdVc = 2.0 * F0 / Vc
    else:
        dFdVc = np.zeros(3)
    S_curr = current_variability_force_psd(dFdVc, sigma_Vc=0.1, tau=600.0)

    S_F_funcs = [S_wind, S_drift, S_curr]

    # --- T_op + uniform integration grid -------------------------------
    T_op_s = 5.0 * 60.0   # operator window: 5 min
    dt_s = 0.5            # 2 Hz LF integration; supersamples WF too
    t = np.arange(0.0, T_op_s, dt_s)
    N_t = t.size

    # LF realisation grid: 256 freqs across the closed-loop band.
    omega_grid_lf = np.linspace(1.0e-3, 0.6, 256)

    rng = np.random.default_rng(3)  # seed chosen so empirical sigma is
    # within ~5% of the model prior at T_op=5 min (typical draw, not a
    # lucky calm window). See scripts/sigma_convergence_sweep.py for the
    # run-to-run sigma distribution at this operating point.

    print(f"\nRealising {T_op_s/60:.0f} min of disturbances + motion at dt={dt_s}s ...")
    F_lf = realise_vector_force_time_series(S_F_funcs, omega_grid_lf, t, rng)
    x_lf = integrate_closed_loop_response(cl, F_lf, t)

    # WF realisation on the same time grid.
    xi_wf = realise_wave_motion_6dof(
        rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel,
        t=t, rng=rng,
    )

    # --- Channel projections -------------------------------------------
    r = radial_position_time_series(x_lf, xi_wf, cfg)             # (N_t,)
    dL = telescope_length_deviation_time_series(x_lf, xi_wf, joint, cfg)

    # Band-split the telescope channel into low-frequency (closed-loop
    # response to wind/drift/current) and wave-frequency (1st-order RAO
    # response to waves). The two physical bands are separated by more
    # than 2 octaves; a 4th-order zero-phase Butterworth at 0.3 rad/s
    # gives clean separation (validated by scripts/validate_bandsplit.py).
    fs_dt_hz = 1.0 / dt_s   # 2 Hz from dt=0.5 s integration
    omega_split_rad_s = 0.3
    dL_lf, dL_wf = bandsplit_lowpass(
        dL, fs_hz=fs_dt_hz, omega_split_rad_s=omega_split_rad_s,
    )

    # WF variance-estimator decorrelation time from the wave PSD itself.
    # The wave PSD `wave.integrand = |c6 . H_xi(omega)|^2 * S_eta` is
    # the proper one-sided PSD of the WF telescope channel, so T_var
    # follows directly. For Tp=9, q~0.3 narrowband, T_var_wf ~ 10 s.
    T_decorr_wave_var_s = float(
        variance_decorrelation_time_from_psd(wave.integrand, wave.omega)
    )

    print(f"  empirical sigma(r)     [sqrt E[r^2]]    = "
          f"{np.sqrt(np.mean(r*r)):.3f} m")
    print(f"  empirical sigma(dL)    [sqrt E[dL^2]]   = "
          f"{np.sqrt(np.mean(dL*dL)):.3f} m   (full-band)")
    print(f"  empirical sigma(dL_lf) [sqrt E[dL_lf^2]] = "
          f"{np.sqrt(np.mean(dL_lf*dL_lf)):.3f} m  (slow band)")
    print(f"  empirical sigma(dL_wf) [sqrt E[dL_wf^2]] = "
          f"{np.sqrt(np.mean(dL_wf*dL_wf)):.3f} m  (wave band)")
    print(f"  T_var_wf (from wave PSD) = {T_decorr_wave_var_s:.1f} s")
    print(f"  (NB: r(t) is Rayleigh-distributed so np.std(r) != sigma_radial)")

    # --- Build the prior summary first to read off model sigmas --------
    prior = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint,
        T_op_s=T_op_s, sigma_L_wave=sigma_L_wave_m, Tp_wave_s=Tp,
        quantiles=(0.50, 0.90),
    )
    print(f"\nModel-prior sigma_radial = {prior.pos_sigma_m:.3f} m")
    print(f"Model-prior sigma_slow   = {prior.gw_sigma_slow_m:.3f} m")
    print(f"Model-prior sigma_wave   = {prior.gw_sigma_wave_m:.3f} m  (RAO, fixed)")

    # --- Online estimators ---------------------------------------------
    # Operator sensors at 1 Hz; estimator window = T_op so it sees the
    # whole observation interval.
    fs_obs_hz = 1.0
    obs_stride = int(round(1.0 / (fs_obs_hz * dt_s)))  # = 2 with dt=0.5s
    dt_obs = dt_s * obs_stride
    obs_idx = np.arange(0, N_t, obs_stride)

    T_decorr_pos = float(prior.pos_T_decorr_var_s)
    T_decorr_gw  = float(prior.gw_T_decorr_var_s)
    T_decorr_pos_legacy = closed_loop_decorrelation_time(cfg.controller, "position")
    T_decorr_gw_legacy  = closed_loop_decorrelation_time(cfg.controller, "sway")
    print(f"\nT_decorr (position, variance-estimator from PSD) = "
          f"{T_decorr_pos:.1f} s   "
          f"[legacy 1/(zeta*omega_n) = {T_decorr_pos_legacy:.1f} s, "
          f"ratio {T_decorr_pos/T_decorr_pos_legacy:.2f}x]")
    print(f"T_decorr (gangway slow, variance-estimator from PSD) = "
          f"{T_decorr_gw:.1f} s   "
          f"[legacy 1/(zeta*omega_n) = {T_decorr_gw_legacy:.1f} s, "
          f"ratio {T_decorr_gw/T_decorr_gw_legacy:.2f}x]")

    est_pos = BayesianSigmaEstimator(
        prior_sigma2=prior.pos_sigma_m ** 2,
        T_decorr_s=T_decorr_pos,
        dt_s=dt_obs,
        window_s=T_op_s,
    )
    est_gw = BayesianSigmaEstimator(
        prior_sigma2=prior.gw_sigma_slow_m ** 2,
        T_decorr_s=T_decorr_gw,
        dt_s=dt_obs,
        window_s=T_op_s,
    )
    # WF telescope estimator: data-conditions sigma_L_wave from dL_wf(t).
    # Acts as a sea-state consistency check on the operator-supplied
    # (Hs, Tp, theta_w). Uses T_var_wf from the wave PSD itself.
    est_gw_wf = BayesianSigmaEstimator(
        prior_sigma2=sigma_L_wave_m ** 2,
        T_decorr_s=T_decorr_wave_var_s,
        dt_s=dt_obs,
        window_s=T_op_s,
    )
    for k in obs_idx:
        est_pos.update(float(r[k]))
        est_gw.update(float(dL_lf[k]))
        est_gw_wf.update(float(dL_wf[k]))

    sig_pos_post_p   = est_pos.posterior(credible=0.90)
    sig_gw_post_p    = est_gw.posterior(credible=0.90)
    sig_gw_wf_post_p = est_gw_wf.posterior(credible=0.90)
    sig_pos_post   = sig_pos_post_p.sigma_median
    sig_gw_post    = sig_gw_post_p.sigma_median
    sig_gw_wf_post = sig_gw_wf_post_p.sigma_median
    n_eff_pos   = est_pos.n_eff
    n_eff_gw    = est_gw.n_eff
    n_eff_gw_wf = est_gw_wf.n_eff
    print(f"\nPosterior sigma_radial = {sig_pos_post:.3f} m  "
          f"(90% CI [{sig_pos_post_p.sigma_lo:.3f}, {sig_pos_post_p.sigma_hi:.3f}], "
          f"n_eff={n_eff_pos:.1f}, warm={est_pos.is_warm()})")
    print(f"Posterior sigma_slow   = {sig_gw_post:.3f} m  "
          f"(90% CI [{sig_gw_post_p.sigma_lo:.3f}, {sig_gw_post_p.sigma_hi:.3f}], "
          f"n_eff={n_eff_gw:.1f}, warm={est_gw.is_warm()})")
    print(f"Posterior sigma_wave   = {sig_gw_wf_post:.3f} m  "
          f"(90% CI [{sig_gw_wf_post_p.sigma_lo:.3f}, {sig_gw_wf_post_p.sigma_hi:.3f}], "
          f"n_eff={n_eff_gw_wf:.1f}, warm={est_gw_wf.is_warm()})")
    print(f"Ratios: radial post/prior = {sig_pos_post/prior.pos_sigma_m:.3f}, "
          f"slow post/prior = {sig_gw_post/prior.gw_sigma_slow_m:.3f}, "
          f"wave post/prior = {sig_gw_wf_post/sigma_L_wave_m:.3f}")

    # --- Posterior summary (same shapes, data-conditioned median sigma) ---
    posterior = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint,
        T_op_s=T_op_s, sigma_L_wave=sigma_L_wave_m, Tp_wave_s=Tp,
        quantiles=(0.50, 0.90),
        posterior_sigma_radial_m=sig_pos_post,
        posterior_sigma_telescope_slow_m=sig_gw_post,
        posterior_sigma_telescope_wave_m=sig_gw_wf_post,
        T_decorr_var_telescope_wave_s=T_decorr_wave_var_s,
    )

    # --- Marginalised predictive P90: integrates over the posterior
    # uncertainty in sigma. ONE number per channel that the operator
    # can act on directly: "with 90% confidence the running max stays
    # below this value over the next T_op."
    p_target = 1.0 - 0.90  # P(M > a) <= 0.10  <=>  P90 of the running max
    a_pred_pos = predictive_running_max_quantile(
        p=p_target,
        bands=[((sig_pos_post_p.alpha, sig_pos_post_p.beta),
                posterior.pos_nu0_max, posterior.pos_q)],
        T=T_op_s, bilateral=True, clustering="vanmarcke", n_quad=64,
    )
    gw_bands = [((sig_gw_post_p.alpha, sig_gw_post_p.beta),
                 posterior.gw_nu0_slow, posterior.gw_q_slow)]
    if sigma_L_wave_m > 0.0 and posterior.gw_nu0_wave > 0.0:
        gw_bands.append(((sig_gw_wf_post_p.alpha, sig_gw_wf_post_p.beta),
                         posterior.gw_nu0_wave, posterior.gw_q_wave))
    a_pred_gw = predictive_running_max_quantile(
        p=p_target, bands=gw_bands, T=T_op_s,
        bilateral=True, clustering="vanmarcke", n_quad=64,
    )
    print(f"\nPredictive P90 (sigma marginalised over posterior):")
    print(f"  vessel footprint = {a_pred_pos:.2f} m  "
          f"(plug-in median P90 = {posterior.pos_a_p90:.2f} m, "
          f"inflation +{(a_pred_pos - posterior.pos_a_p90)*100:.0f} cm)")
    print(f"  telescope dL     = {a_pred_gw:.2f} m  "
          f"(plug-in median P90 = {posterior.gw_a_p90:.2f} m, "
          f"inflation +{(a_pred_gw - posterior.gw_a_p90)*100:.0f} cm)")

    # --- 2x2 figure: prior (top) + posterior (bottom) ------------------
    fig = plt.figure(figsize=(13.5, 9.0))
    subfigs = fig.subfigures(2, 1, hspace=0.08)
    plot_intact_prior(prior, fig=subfigs[0])
    plot_intact_prior(posterior, fig=subfigs[1])

    # Overlay the marginalised predictive P90 on the posterior bars
    # as a heavy black diamond marker. This is the operator-actionable
    # number: "with 90% confidence the running max stays below this."
    post_axes = subfigs[1].axes
    pred_pairs = [
        (post_axes[0], a_pred_pos, "vessel footprint"),
        (post_axes[1], a_pred_gw,  "telescope dL"),
    ]
    for ax, a_pred, label in pred_pairs:
        ax.plot([a_pred], [0.0], marker="D", markersize=15,
                markeredgecolor="black", markerfacecolor="none",
                markeredgewidth=2.2, zorder=5,
                label=f"Predictive P90 ({label}) = {a_pred:.2f} m")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.95, ncol=1)

    # plot_intact_prior unconditionally sets a suptitle on the figure
    # passed to it; override here so the row label is clean.
    subfigs[0].suptitle(
        f"PRIOR  (model only, T_op = {T_op_s/60:.0f} min)",
        fontsize=13, fontweight="bold", y=0.92, color="#444444",
    )
    subfigs[1].suptitle(
        f"POSTERIOR  (after {T_op_s/60:.0f} min observation, "
        f"n_eff_pos={n_eff_pos:.1f}, n_eff_gw_lf={n_eff_gw:.1f}, "
        f"n_eff_gw_wf={n_eff_gw_wf:.1f}; "
        f"hollow black diamond = predictive P90 marginalised over \u03c3 posterior)",
        fontsize=13, fontweight="bold", y=0.92, color="#1f6f1f",
    )
    fig.suptitle(
        "CSOV intact-prior length-scale panel: prior vs data-conditioned posterior",
        fontsize=14, y=0.995,
    )

    out = os.path.join(HERE, "csov_prior_vs_posterior.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out}")

    # --- Compact comparison table --------------------------------------
    print("\n=== Quantile comparison (P50 / P90 of running max over T_op) ===")
    print(f"  Vessel footprint  prior:     P50={prior.pos_a_p50:.2f} m, "
          f"P90={prior.pos_a_p90:.2f} m  [{prior.pos_traffic_prior.upper()}]")
    print(f"  Vessel footprint  posterior: P50={posterior.pos_a_p50:.2f} m, "
          f"P90={posterior.pos_a_p90:.2f} m  [{posterior.pos_traffic_prior.upper()}]")
    print(f"  Telescope dL      prior:     P50={prior.gw_a_p50:.2f} m, "
          f"P90={prior.gw_a_p90:.2f} m  [{prior.gw_traffic_prior.upper()}]")
    print(f"  Telescope dL      posterior: P50={posterior.gw_a_p50:.2f} m, "
          f"P90={posterior.gw_a_p90:.2f} m  [{posterior.gw_traffic_prior.upper()}]")


if __name__ == "__main__":
    main()
