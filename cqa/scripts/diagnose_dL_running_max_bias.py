"""Localise the source of the running-max calibration error for dL(t).

The validate_running_max_cdf.py script revealed that the predicted CDF
of the telescope-deviation running max M_T = max|dL| over T_op
underpredicts by ~25% on P50/P90, with empirical coverage at predicted
P90 of only 75% (should be 90%).

dL is the projection of (LF + WF) body motion onto the telescope
direction. The LF band is the closed-loop response to wind/drift/current
forces; the WF band is the linear RAO response to first-order waves.
Both should be (approximately) Gaussian.

Three independent diagnostics:

  1. Marginal Gaussianity check for dL, dL_lf, dL_wf.
     Q-Q plot + skew/kurtosis. If marginals aren't Gaussian, no
     Rice-style formula works.

  2. LF-WF independence check. Cross-correlation between dL_lf and
     dL_wf, and between their squares (envelope coupling). The
     multiband Rice formula assumes statistical independence.

  3. Spectral parameter validation. Empirical zero-upcrossing rate
     and Vanmarcke q from the *realised* time series, compared with
     the model values from sigma_L_wave + closed-loop spectrum.

Plus a sigma-level check (sigma_emp vs sigma_model per band).

Run:
    python scripts/diagnose_dL_running_max_bias.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal

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
    realise_vector_force_time_series,
    integrate_closed_loop_response,
    realise_wave_motion_6dof,
    telescope_length_deviation_time_series,
    bandsplit_lowpass,
    vanmarcke_bandwidth_q,
    zero_upcrossing_rate,
)
from cqa.vessel import LinearVesselModel, CurrentForceModel
from cqa.controller import LinearDpController
from cqa.closed_loop import ClosedLoop
from cqa.psd import WindForceModel

PDSTRIP_PATH = Path.home() / "src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def empirical_zero_upcrossing_rate(x: np.ndarray, dt: float) -> float:
    """Count sign changes from negative to positive, divided by total time."""
    x = np.asarray(x) - np.mean(x)
    s = np.sign(x)
    # Treat zeros as no-cross (rare)
    crossings = np.sum((s[:-1] < 0) & (s[1:] > 0))
    return float(crossings / (dt * (x.size - 1)))


def empirical_vanmarcke_q_from_psd(
    x: np.ndarray, dt: float, omega_max: float | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Estimate q = sqrt(1 - lambda_1^2 / (lambda_0 lambda_2)) from
    Welch-averaged PSD of x. Also returns nu_0+ from spectral moments.

    Returns (q, nu_0+, omega, S_one_sided)
    """
    fs = 1.0 / dt
    nperseg = min(8192, x.size // 8)
    f, S_two = signal.welch(x, fs=fs, nperseg=nperseg, return_onesided=True,
                             scaling="density", detrend="constant")
    # signal.welch returns one-sided in cycles/s; convert to rad/s and
    # divide PSD by 2pi to get density per rad/s.
    omega = 2.0 * np.pi * f
    S_one = S_two / (2.0 * np.pi)
    if omega_max is not None:
        m = omega <= omega_max
        omega = omega[m]
        S_one = S_one[m]
    # Spectral moments. cqa convention: S is one-sided in omega, so
    # lambda_n = integral_0^inf omega^n S(omega) d omega.
    lam0 = np.trapezoid(S_one, omega)
    lam1 = np.trapezoid(omega * S_one, omega)
    lam2 = np.trapezoid(omega ** 2 * S_one, omega)
    if lam0 <= 0 or lam2 <= 0:
        return 1.0, 0.0, omega, S_one
    q2 = 1.0 - (lam1 ** 2) / (lam0 * lam2)
    q = float(np.sqrt(max(q2, 0.0)))
    nu0 = float(np.sqrt(lam2 / lam0) / (2.0 * np.pi))
    return q, nu0, omega, S_one


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    cfg = csov_default_config()
    Vw_mean = 14.0; Hs = 2.8; Tp = 9.0; Vc = 0.5
    theta_rel = np.radians(30.0)
    gw = cfg.gangway
    L0 = 0.5 * (gw.telescope_min + gw.telescope_max)
    joint = GangwayJointState(h=gw.rotation_centre_height_above_base,
                              alpha_g=-np.pi/2, beta_g=0.0, L=L0)

    rao_table = load_pdstrip_rao(PDSTRIP_PATH)
    wave = sigma_L_wave(joint, cfg, rao_table, Hs=Hs, Tp=Tp,
                        theta_wave_rel=theta_rel)
    sigma_L_wave_m = float(wave.sigma_L_wave)

    vp, wp, cp = cfg.vessel, cfg.wind, cfg.current
    vessel = LinearVesselModel.from_config(vp)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D,
        omega_n=cfg.controller.omega_n, zeta=cfg.controller.zeta,
    )
    cl = ClosedLoop.build(vessel, controller)
    wm = WindForceModel(wp=wp, loa=vp.loa)
    S_wind = npd_wind_gust_force_psd(wm, Vw_mean, theta_rel)
    S_drift = slow_drift_force_psd_newman_pdstrip(
        rao_table=rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel,
    )
    cm = CurrentForceModel(cp=cp,
        lateral_area_underwater=vp.lpp*vp.draft,
        frontal_area_underwater=vp.beam*vp.draft, loa=vp.loa)
    F0 = cm.force(Vc, theta_rel); dFdVc = 2.0*F0/Vc
    S_curr = current_variability_force_psd(dFdVc, sigma_Vc=0.1, tau=600.0)
    S_F_funcs = [S_wind, S_drift, S_curr]

    # Long realisation (12 h; same as validate_running_max_cdf for
    # consistency).
    T_realisation_s = 12.0 * 3600.0
    dt_s = 0.5
    t = np.arange(0.0, T_realisation_s, dt_s)
    print(f"Realising {T_realisation_s/3600:.1f} h at dt={dt_s}s ...")

    omega_grid_lf = np.linspace(1.0e-3, 0.6, 256)
    rng = np.random.default_rng(7)
    F_lf = realise_vector_force_time_series(S_F_funcs, omega_grid_lf, t, rng)
    x_lf = integrate_closed_loop_response(cl, F_lf, t)
    xi_wf = realise_wave_motion_6dof(rao_table, Hs=Hs, Tp=Tp,
                                      theta_wave_rel=theta_rel, t=t, rng=rng)
    dL = telescope_length_deviation_time_series(x_lf, xi_wf, joint, cfg)

    # Band-split.
    fs = 1.0 / dt_s
    omega_split = 0.3
    dL_lf, dL_wf = bandsplit_lowpass(dL, fs_hz=fs, omega_split_rad_s=omega_split)

    # ---- Build the model prior to read off model spectral params ----
    T_op_s = 5.0 * 60.0
    prior = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint,
        T_op_s=T_op_s, sigma_L_wave=sigma_L_wave_m, Tp_wave_s=Tp,
        quantiles=(0.50, 0.90),
    )

    # =================================================================
    # Diagnostic 1: sigma-level audit
    # =================================================================
    print("\n=== Diagnostic 1: sigma-level audit ===")
    sig_dL_emp = float(np.sqrt(np.mean(dL ** 2)))
    sig_lf_emp = float(np.sqrt(np.mean(dL_lf ** 2)))
    sig_wf_emp = float(np.sqrt(np.mean(dL_wf ** 2)))
    sig_dL_mod = float(np.sqrt(prior.gw_sigma_slow_m**2 + prior.gw_sigma_wave_m**2))
    print(f"  sigma(dL)      emp={sig_dL_emp:.4f} m  model={sig_dL_mod:.4f} m  "
          f"ratio={sig_dL_emp/sig_dL_mod:.3f}")
    print(f"  sigma(dL_lf)   emp={sig_lf_emp:.4f} m  model={prior.gw_sigma_slow_m:.4f} m  "
          f"ratio={sig_lf_emp/prior.gw_sigma_slow_m:.3f}")
    print(f"  sigma(dL_wf)   emp={sig_wf_emp:.4f} m  model={prior.gw_sigma_wave_m:.4f} m  "
          f"ratio={sig_wf_emp/prior.gw_sigma_wave_m:.3f}")

    # =================================================================
    # Diagnostic 2: marginal Gaussianity
    # =================================================================
    print("\n=== Diagnostic 2: marginal Gaussianity (skew, kurtosis, KS) ===")
    for name, x in [("dL", dL), ("dL_lf", dL_lf), ("dL_wf", dL_wf)]:
        x = x - x.mean()
        sk = float(stats.skew(x))
        ek = float(stats.kurtosis(x, fisher=True))  # excess kurtosis
        # Subsample for KS (full 86400 makes p-values misleading)
        sub = x[::5]
        ks_D, ks_p = stats.kstest(sub, "norm",
                                  args=(0.0, sub.std()))
        print(f"  {name:6s}  skew={sk:+.3f}  excess_kurt={ek:+.3f}  "
              f"KS(N): D={ks_D:.4f} p={ks_p:.3g}  (n_sub={sub.size})")

    # =================================================================
    # Diagnostic 3: LF-WF independence
    # =================================================================
    print("\n=== Diagnostic 3: LF-WF independence ===")
    rho_direct = float(np.corrcoef(dL_lf, dL_wf)[0, 1])
    rho_squared = float(np.corrcoef(dL_lf**2, dL_wf**2)[0, 1])
    rho_abs = float(np.corrcoef(np.abs(dL_lf), np.abs(dL_wf))[0, 1])
    print(f"  corr(dL_lf, dL_wf)         = {rho_direct:+.4f}  "
          f"(should be ~0 by spectral disjoint-ness)")
    print(f"  corr(|dL_lf|, |dL_wf|)     = {rho_abs:+.4f}  "
          f"(envelope coupling -- nonzero if amplitude-modulated)")
    print(f"  corr(dL_lf^2, dL_wf^2)     = {rho_squared:+.4f}  "
          f"(variance coupling)")

    # =================================================================
    # Diagnostic 4: spectral parameters (nu_0+, q) per band
    # =================================================================
    print("\n=== Diagnostic 4: spectral parameters per band ===")
    # Empirical from time series:
    nu0_emp_lf = empirical_zero_upcrossing_rate(dL_lf, dt_s)
    nu0_emp_wf = empirical_zero_upcrossing_rate(dL_wf, dt_s)
    q_lf_emp, nu0_lf_psd, om_lf, S_lf = empirical_vanmarcke_q_from_psd(
        dL_lf, dt_s, omega_max=omega_split * 1.5,
    )
    q_wf_emp, nu0_wf_psd, om_wf, S_wf = empirical_vanmarcke_q_from_psd(
        dL_wf, dt_s, omega_max=2.0,  # well above WF content
    )
    print(f"  LF band:")
    print(f"    nu_0+ empirical (sign-change count) = {nu0_emp_lf*1000:.3f} mHz")
    print(f"    nu_0+ empirical (PSD spectral mom.) = {nu0_lf_psd*1000:.3f} mHz")
    print(f"    nu_0+ model                         = {prior.gw_nu0_slow*1000:.3f} mHz")
    print(f"    q empirical                         = {q_lf_emp:.3f}")
    print(f"    q model                             = {prior.gw_q_slow:.3f}")
    print(f"  WF band:")
    print(f"    nu_0+ empirical (sign-change count) = {nu0_emp_wf*1000:.3f} mHz")
    print(f"    nu_0+ empirical (PSD spectral mom.) = {nu0_wf_psd*1000:.3f} mHz")
    print(f"    nu_0+ model                         = {prior.gw_nu0_wave*1000:.3f} mHz")
    print(f"    q empirical                         = {q_wf_emp:.3f}")
    print(f"    q model                             = {prior.gw_q_wave:.3f}")

    # =================================================================
    # Figure: marginal Q-Q + spectral comparison
    # =================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Row 1: marginal Q-Q vs Gaussian
    for ax, name, x in zip(axes[0], ("dL", "dL_lf", "dL_wf"),
                            (dL, dL_lf, dL_wf)):
        x_centred = x - x.mean()
        sub = x_centred[::5]
        stats.probplot(sub, dist="norm", plot=ax)
        ax.set_title(f"{name}: Q-Q vs Gaussian "
                     f"(skew={stats.skew(x_centred):+.2f}, "
                     f"kurt={stats.kurtosis(x_centred, fisher=True):+.2f})")
        ax.grid(alpha=0.3)

    # Row 2 col 0: empirical vs model PSD, LF band
    omega_model_lf = np.linspace(1e-3, 0.6, 400)
    # Reconstruct model dL_lf PSD: |c.H_x|^2 * S_F summed over forces.
    # That's already inside summarise_intact_prior; rather than rebuild,
    # show the empirical PSD with the model nu_0+/q numerically annotated.
    ax = axes[1, 0]
    ax.semilogy(om_lf, S_lf, "-", color="C0", label="empirical (Welch)")
    ax.axvline(omega_split, color="k", linestyle=":", linewidth=0.8,
               label=f"split omega={omega_split} rad/s")
    ax.set_xlabel("omega [rad/s]")
    ax.set_ylabel("S_dL_lf(omega) [m^2 s/rad]")
    ax.set_title(f"LF band: nu_0+ emp={nu0_emp_lf*1000:.2f}mHz vs "
                 f"model={prior.gw_nu0_slow*1000:.2f}mHz, "
                 f"q emp={q_lf_emp:.2f} vs {prior.gw_q_slow:.2f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(0, 0.6)

    # Row 2 col 1: WF band PSD
    ax = axes[1, 1]
    ax.semilogy(om_wf, S_wf, "-", color="C2", label="empirical (Welch)")
    ax.axvline(omega_split, color="k", linestyle=":", linewidth=0.8,
               label=f"split omega={omega_split} rad/s")
    ax.set_xlabel("omega [rad/s]")
    ax.set_ylabel("S_dL_wf(omega) [m^2 s/rad]")
    ax.set_title(f"WF band: nu_0+ emp={nu0_emp_wf*1000:.1f}mHz vs "
                 f"model={prior.gw_nu0_wave*1000:.1f}mHz, "
                 f"q emp={q_wf_emp:.2f} vs {prior.gw_q_wave:.2f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(0, 2.0)

    # Row 2 col 2: scatter dL_lf vs dL_wf for visual independence check
    ax = axes[1, 2]
    sub_idx = np.random.default_rng(0).choice(dL_lf.size, size=4000, replace=False)
    ax.plot(dL_lf[sub_idx], dL_wf[sub_idx], ".", markersize=1.5, alpha=0.4)
    ax.set_xlabel("dL_lf [m]")
    ax.set_ylabel("dL_wf [m]")
    ax.set_title(f"LF vs WF scatter (rho_direct={rho_direct:+.3f}, "
                 f"rho_|.|={rho_abs:+.3f})")
    ax.grid(alpha=0.3); ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(
        f"dL running-max bias diagnostics (CSOV, {T_realisation_s/3600:.0f}h "
        f"realisation, Hs={Hs} Tp={Tp})",
        fontsize=12,
    )
    fig.tight_layout()
    out = os.path.join(HERE, "diagnose_dL_running_max_bias.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
