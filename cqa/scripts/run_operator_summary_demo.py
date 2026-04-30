"""Operator-facing summary of the WCFDI consequence at one operating point.

Runs `wcfdi_mc(sample_mode='full12')` and renders the two-bar summary
intended for the bridge: vessel position vs warning/alarm radii, and
gangway telescope vs end-stops, with conditional P_exceed numbers per
limit.

Now also includes the 1st-order wave-frequency telescope contribution
sigma_L_wave from pdstrip RAOs (see cqa.wave_response). The combined
operability statement uses margin shrink: each per-sample margin is
reduced by k_wave * sigma_L_wave (k_wave=1.96 default).

Run:
    python scripts/run_operator_summary_demo.py
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
    wcfdi_mc,
    WcfdiScenario,
    GangwayJointState,
    summarise_for_operator,
    plot_operator_summary,
    summarise_intact_prior,
    plot_intact_prior,
    load_pdstrip_rao,
    sigma_L_wave,
    npd_wind_gust_force_psd,
    current_variability_force_psd,
    slow_drift_force_psd_newman_pdstrip,
)
from cqa.vessel import LinearVesselModel, CurrentForceModel
from cqa.controller import LinearDpController
from cqa.closed_loop import ClosedLoop
from cqa.psd import WindForceModel

PDSTRIP_PATH = Path.home() / "src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"


def _bandwidth_label(q: float) -> str:
    if q < 0.3:
        return "narrowband"
    if q < 0.6:
        return "transition"
    return "wideband"


def main() -> None:
    cfg = csov_default_config()

    Vw_mean = 14.0
    Hs = 2.8
    Tp = 9.0
    Vc = 0.5
    # Bow quartering: 30 deg off the bow on the port side. More realistic
    # operating heading for W2W than pure beam (operators routinely
    # weather-vane the bow into the prevailing seas, accepting some
    # quartering rather than pure head sea so the gangway side can pick
    # the "lee" of the vessel against drift).
    theta_rel = np.radians(30.0)

    scenario = WcfdiScenario(alpha=(0.8, 0.8, 0.8), T_realloc=10.0)

    gw = cfg.gangway
    L0 = 0.5 * (gw.telescope_min + gw.telescope_max)
    joint = GangwayJointState(
        h=gw.rotation_centre_height_above_base,
        alpha_g=-np.pi / 2.0,  # port
        beta_g=0.0,
        L=L0,
    )

    # 1st-order wave telescope contribution from pdstrip RAOs.
    # Also opt in to pdstrip-driven Pinkster mean drift + Newman slow-drift
    # PSD (replaces the parametric cfg.wave_drift constants in wcfdi_mc).
    if PDSTRIP_PATH.exists():
        rao_table = load_pdstrip_rao(PDSTRIP_PATH)
        wave = sigma_L_wave(joint, cfg, rao_table, Hs=Hs, Tp=Tp,
                            theta_wave_rel=theta_rel)
        sigma_L_wave_m = wave.sigma_L_wave
        print(f"sigma_L_wave (1st-order, beam Hs={Hs} Tp={Tp}) = {sigma_L_wave_m*100:.1f} cm")
        print(f"  per-DOF [surge,sway,heave,roll,pitch,yaw] cm:",
              [f'{x*100:.1f}' for x in wave.sigma_L_wave_per_dof])
    else:
        print(f"WARNING: pdstrip data not found at {PDSTRIP_PATH}; "
              f"running with sigma_L_wave = 0 (DP slow content only).")
        sigma_L_wave_m = 0.0
        rao_table = None

    print("\nRunning WCFDI MC (full12, n_samples=1000, pdstrip drift)...")
    res = wcfdi_mc(
        cfg,
        Vw_mean=Vw_mean,
        Hs=Hs, Tp=Tp,
        Vc=Vc,
        theta_rel=theta_rel,
        scenario=scenario,
        joint=joint,
        n_samples=1000,
        t_end=200.0,
        n_t=201,
        rng_seed=0,
        sample_mode="full12",
        rao_table=rao_table,
    )

    weather_summary = (
        f"Vw={Vw_mean:.0f} m/s, Hs={Hs:.1f} m, Tp={Tp:.1f} s, Vc={Vc:.1f} m/s, "
        f"theta_rel={np.degrees(theta_rel):.0f} deg (port bow quartering), "
        f"alpha={scenario.alpha[0]:.2f}, T_realloc={scenario.T_realloc:.0f} s"
    )

    summary = summarise_for_operator(
        res, cfg, weather_summary=weather_summary,
        sigma_L_wave=sigma_L_wave_m,
    )

    # Console summary
    print("\n=== Operator summary (IMCA M254 Rev.1 Fig.8, conditional on WCF now) ===")
    print(f"Operating point: {weather_summary}")
    print()
    print(f"VESSEL CAPABILITY  [{summary.pos_traffic.upper()}]")
    print(f"  footprint (P95) = {summary.pos_p95:.2f} m   "
          f"(IMCA amber > {summary.pos_warning_radius_m:.0f} m, "
          f"red > {summary.pos_alarm_radius_m:.0f} m)")
    print(f"  if WCF now: P(footprint > {summary.pos_warning_radius_m:.0f} m) "
          f"= {summary.pos_p_warning*100:.2f} %, "
          f"P(footprint > {summary.pos_alarm_radius_m:.0f} m) "
          f"= {summary.pos_p_alarm*100:.2f} %")
    print(f"  P50 peak = {summary.pos_p50:.2f} m, P95 peak = {summary.pos_p95:.2f} m")
    print()
    print(f"GANGWAY CAPABILITY  [{summary.gw_traffic.upper()}]")
    print(f"  utilisation = {summary.gw_util_imca*100:.1f} %   "
          f"(IMCA amber > {summary.gw_imca_warning_frac*100:.0f} %, "
          f"red > {summary.gw_imca_alarm_frac*100:.0f} %)")
    print(f"  L_min = {summary.gw_telescope_min_m:.1f} m, "
          f"L_max = {summary.gw_telescope_max_m:.1f} m, "
          f"L_setpoint = {summary.gw_L0_m:.2f} m")
    print(f"  sigma_L_wave = {summary.gw_sigma_L_wave_m:.3f} m  "
          f"(k_wave={summary.gw_k_wave:.2f} -> reach {summary.gw_k_sigma_L_wave_m:.2f} m)")
    print(f"  if WCF now: P(end-stop hit, slow+wave) = {summary.gw_p_alarm*100:.2f} %")
    print(f"  Peak |dL_slow|: P50 = {summary.gw_p50:.2f} m, P95 = {summary.gw_p95:.2f} m")
    print(f"  Lower margin (combined): P50 = {summary.gw_lower_margin_p50:.2f} m, "
          f"worst (P95) = {summary.gw_lower_margin_p95:.2f} m")
    print(f"  Upper margin (combined): P50 = {summary.gw_upper_margin_p50:.2f} m, "
          f"worst (P95) = {summary.gw_upper_margin_p95:.2f} m")

    fig = plot_operator_summary(summary)
    out = os.path.join(HERE, "csov_operator_summary.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out}")

    # ------------------------------------------------------------------
    # Intact-prior panel (Q1: "before operation, what's the chance the
    # vessel breaches its footprint or the gangway hits an end-stop in
    # the next T_op minutes if NO failure occurs?")
    # ------------------------------------------------------------------
    print("\n=== Intact-prior summary (Rice / Cartwright-Longuet-Higgins) ===")

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

    if rao_table is not None:
        S_drift = slow_drift_force_psd_newman_pdstrip(
            rao_table=rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel,
        )
    else:
        S_drift = lambda w: np.zeros((3, 3))

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

    T_op_s = 20 * 60.0
    prior = summarise_intact_prior(
        cl, [S_wind, S_drift, S_curr], cfg, joint,
        T_op_s=T_op_s, sigma_L_wave=sigma_L_wave_m, Tp_wave_s=Tp,
        quantiles=(0.50, 0.90),
    )

    p_lo_pct, p_hi_pct = (q * 100.0 for q in prior.quantiles)
    print(f"Operation duration T_op = {T_op_s/60.0:.0f} min, "
          f"quantiles = (P{p_lo_pct:.0f}, P{p_hi_pct:.0f})")
    print(f"Vessel footprint:  sigma_radial = {prior.pos_sigma_m:.2f} m, "
          f"nu0_max = {prior.pos_nu0_max*60:.2f} /min, "
          f"q = {prior.pos_q:.2f} ({_bandwidth_label(prior.pos_q)})")
    print(f"  IMCA radii: warn = {prior.pos_warning_radius_m:.1f} m, "
          f"alarm = {prior.pos_alarm_radius_m:.1f} m")
    print(f"  P{p_lo_pct:.0f} peak |dp_base| = {prior.pos_a_p50:.2f} m, "
          f"P{p_hi_pct:.0f} = {prior.pos_a_p90:.2f} m")
    print(f"  diagnostic: P(footprint > {prior.pos_warning_radius_m:.0f} m) "
          f"= {prior.pos_p_breach*100:.2f} %, "
          f"P(footprint > {prior.pos_alarm_radius_m:.0f} m) "
          f"= {prior.pos_p_breach_alarm*100:.4f} %")
    print(f"  -> traffic [{prior.pos_traffic_prior.upper()}] "
          f"(P{p_hi_pct:.0f} vs warn/alarm radii)")
    print(f"Gangway telescope: stroke = {prior.gw_threshold_used_m:.2f} m, "
          f"warn = {prior.gw_warn_m:.2f} m, alarm = {prior.gw_alarm_m:.2f} m")
    print(f"  sigma_slow = {prior.gw_sigma_slow_m*100:.1f} cm "
          f"(q_slow = {prior.gw_q_slow:.2f}), "
          f"sigma_wave = {prior.gw_sigma_wave_m*100:.1f} cm "
          f"(q_wave = {prior.gw_q_wave:.2f})")
    print(f"  P{p_lo_pct:.0f} peak |dL| = {prior.gw_a_p50:.2f} m, "
          f"P{p_hi_pct:.0f} = {prior.gw_a_p90:.2f} m")
    print(f"  diagnostic: P(|dL| > warn) = {prior.gw_p_breach_warn*100:.4f} %, "
          f"P(|dL| > alarm) = {prior.gw_p_breach_alarm*100:.4f} %")
    print(f"  Per-band attribution at warn: slow {prior.gw_p_breach_per_band[0]*100:.3f} %, "
          f"wave {prior.gw_p_breach_per_band[1]*100:.3f} %")
    print(f"  -> traffic [{prior.gw_traffic_prior.upper()}] "
          f"(P{p_hi_pct:.0f} vs warn/alarm radii)")

    fig_prior = plot_intact_prior(prior)
    out_prior = os.path.join(HERE, "csov_intact_prior.png")
    fig_prior.savefig(out_prior, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out_prior}")


if __name__ == "__main__":
    main()
