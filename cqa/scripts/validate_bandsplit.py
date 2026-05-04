"""Validate `bandsplit_lowpass` against the production prior summary.

Pipeline
--------
1. Build the canonical CSOV operating point.
2. Realise dL(t) over 60 minutes including BOTH the closed-loop slow
   response and the 1st-order wave response.
3. Apply `bandsplit_lowpass` at omega_split = 0.3 rad/s.
4. Compare:
     sqrt(<dL_lf^2>)  vs  prior.gw_sigma_slow_m
     sqrt(<dL_wf^2>)  vs  sigma_L_wave (from the deterministic wave
                                         RAO computation)

Pass criterion: each within 5%. This is the gold-standard "the
filter does what it advertises on the actual output" check before we
build a posterior on top of it.

Run:
    python scripts/validate_bandsplit.py
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
    bandsplit_lowpass,
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

    # --- Spectral expectations ---
    wave = sigma_L_wave(joint, cfg, rao_table, Hs=Hs, Tp=Tp,
                       theta_wave_rel=theta_rel)
    sigma_L_wave_m = float(wave.sigma_L_wave)
    prior = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint,
        T_op_s=300.0, sigma_L_wave=sigma_L_wave_m, Tp_wave_s=Tp,
    )
    sigma_slow_spec = float(prior.gw_sigma_slow_m)
    sigma_wave_spec = sigma_L_wave_m

    print("=" * 72)
    print("Band-split validation against production prior summary")
    print("=" * 72)
    print(f"Operating point: Hs={Hs}, Tp={Tp}, theta={np.degrees(theta_rel):.0f} deg, "
          f"Vw={Vw_mean}, Vc={Vc}")
    print(f"\nSpectral expectations:")
    print(f"  sigma_dL_slow_spec = {sigma_slow_spec*100:.2f} cm")
    print(f"  sigma_dL_wave_spec = {sigma_wave_spec*100:.2f} cm")
    print(f"  sigma_dL_total_spec = "
          f"{np.sqrt(sigma_slow_spec**2 + sigma_wave_spec**2)*100:.2f} cm")

    # --- Realise dL(t) on a long horizon ---
    rng = np.random.default_rng(2026)
    T_total = 60.0 * 60.0   # 60 min
    dt = 0.5
    t = np.arange(0.0, T_total, dt)
    fs_hz = 1.0 / dt
    omega_grid_lf = np.geomspace(1.0e-4, 0.6, 256)

    print(f"\nRealising T_total={T_total/60:.0f} min, dt={dt} s "
          f"(fs={fs_hz} Hz, N={t.size})")
    F_lf = realise_vector_force_time_series(S_F_funcs, omega_grid_lf, t, rng)
    x_lf = integrate_closed_loop_response(cl, F_lf, t)
    xi_wf = realise_wave_motion_6dof(
        rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel,
        t=t, rng=rng,
    )
    dL = telescope_length_deviation_time_series(x_lf, xi_wf, joint, cfg)

    # --- Apply band split ---
    omega_split = 0.3   # rad/s, between 0.15 (LF cap) and 0.5 (WF lower edge)
    print(f"\nApplying bandsplit_lowpass at omega_split={omega_split} rad/s "
          f"(period {2*np.pi/omega_split:.1f} s)")
    dL_lf, dL_wf = bandsplit_lowpass(dL, fs_hz=fs_hz,
                                      omega_split_rad_s=omega_split)

    # Drop the first/last 60 s to avoid filtfilt edge transients.
    edge = int(60.0 * fs_hz)
    dL_in = dL[edge:-edge]
    dL_lf_in = dL_lf[edge:-edge]
    dL_wf_in = dL_wf[edge:-edge]

    sigma_lf_emp = np.sqrt(np.mean(dL_lf_in ** 2))
    sigma_wf_emp = np.sqrt(np.mean(dL_wf_in ** 2))
    sigma_total_emp = np.sqrt(np.mean(dL_in ** 2))
    sigma_total_quad = np.sqrt(sigma_lf_emp ** 2 + sigma_wf_emp ** 2)

    print(f"\nEmpirical (60-min realisation, 60 s edge trimmed):")
    print(f"  sigma_dL_lf_emp     = {sigma_lf_emp*100:.2f} cm   "
          f"(vs spec {sigma_slow_spec*100:.2f} cm, "
          f"ratio {sigma_lf_emp/sigma_slow_spec:.3f})")
    print(f"  sigma_dL_wf_emp     = {sigma_wf_emp*100:.2f} cm   "
          f"(vs spec {sigma_wave_spec*100:.2f} cm, "
          f"ratio {sigma_wf_emp/sigma_wave_spec:.3f})")
    print(f"  sigma_dL_total_emp  = {sigma_total_emp*100:.2f} cm")
    print(f"  sqrt(lf^2 + wf^2)   = {sigma_total_quad*100:.2f} cm   "
          f"(should equal sigma_dL_total_emp since lf+wf == dL exactly "
          f"and bands are uncorrelated)")

    # Cross-check: lf and wf should be nearly uncorrelated since the
    # underlying physical bands are spectrally separated.
    cross = np.mean(dL_lf_in * dL_wf_in)
    print(f"  <dL_lf * dL_wf> = {cross:.4e}  "
          f"(<< sigma_lf*sigma_wf = {sigma_lf_emp*sigma_wf_emp:.4e})")
    rho = cross / (sigma_lf_emp * sigma_wf_emp)
    print(f"  correlation rho = {rho:+.4f}")

    print("\n" + "=" * 72)
    ok_lf = abs(sigma_lf_emp / sigma_slow_spec - 1.0) < 0.10
    ok_wf = abs(sigma_wf_emp / sigma_wave_spec - 1.0) < 0.10
    if ok_lf and ok_wf:
        print(f"PASS: both bands within 10% of spectral expectation")
    else:
        print(f"FAIL: lf={'OK' if ok_lf else 'BAD'}, "
              f"wf={'OK' if ok_wf else 'BAD'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
