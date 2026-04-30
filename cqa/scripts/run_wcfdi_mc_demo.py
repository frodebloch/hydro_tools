"""Demo: Monte-Carlo over WCF starting state, single CSOV operating point.

For one CQA-guarded operating point, samples (eta, nu) at t=0- from the
intact 6x6 closed-loop covariance, propagates the deterministic post-WCF
augmented ODE for each draw, and produces three outputs to help us decide
which presentation actually informs an operator:

  (1) Empirical CDF of peak |Delta_L| over the recovery window, with the
      hard end-stop margins (L0 - L_min, L_max - L0) marked, and the
      linearised P2 prediction for the same percentile drawn as a
      reference line.

  (2) 50/95/99 percentile bands on L(t), overlaid on the linearised
      mean +/- 1.96 sigma envelope. Lets you see when (in time) the
      worst realisations bite, and whether the linearised picture is a
      good summary of the MC envelope.

  (3) Starting-state sensitivity table: linear regression of signed peak
      Delta_L on the 6 whitened components of x(0-), reported as
      'peak shift in metres per 1-sigma offset of this component'. Tells
      us whether position offset, velocity, or heading rate dominates.

Run:
    python scripts/run_wcfdi_mc_demo.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cqa import (
    csov_default_config,
    WcfdiScenario,
    GangwayJointState,
    wcfdi_mc,
    starting_state_sensitivity,
)


def main() -> None:
    cfg = csov_default_config()
    gw = cfg.gangway

    # CQA-guarded operating point (matches the P2/P3 demo conditions)
    Vw_mean = 14.0
    Hs = 2.8
    Tp = 9.0
    Vc = 0.5
    theta_rel = np.pi / 2.0  # beam

    # Mid-stroke gangway, pointing port (landing on lee side)
    joint = GangwayJointState(
        h=gw.rotation_centre_height_above_base,
        alpha_g=-np.pi / 2.0,
        beta_g=0.0,
        L=0.5 * (gw.telescope_min + gw.telescope_max),
    )
    L0 = joint.L

    scenario = WcfdiScenario(alpha=(0.8, 0.8, 0.8), T_realloc=10.0)

    print(f"Operating point: Vw={Vw_mean} m/s, Hs={Hs} m, Tp={Tp} s, "
          f"Vc={Vc} m/s, beam")
    print(f"Scenario: alpha={scenario.alpha}, T_realloc={scenario.T_realloc} s, "
          f"gamma_immediate={scenario.gamma_immediate}")
    print(f"Gangway: L0={L0} m, alpha_g={np.degrees(joint.alpha_g)} deg")
    print()

    n_samples = 1000
    print(f"Running {n_samples} MC samples...")
    res = wcfdi_mc(
        cfg,
        Vw_mean=Vw_mean, Hs=Hs, Tp=Tp, Vc=Vc, theta_rel=theta_rel,
        scenario=scenario, joint=joint,
        n_samples=n_samples, t_end=180.0, n_t=181,
        rng_seed=0,
    )
    print(f"  done: {res.info['n_failed']} failed integrations")
    print()

    # --- Console summary ---
    cqa_v = res.info["cqa_precondition_violated"]
    if np.any(cqa_v):
        print(f"*** CQA precondition violated in DOFs: {np.where(cqa_v)[0]}")
    else:
        print("CQA precondition OK in all DOFs.")
    print(f"Operable fraction (gangway in [L_min, L_max] for full recovery): "
          f"{res.operable_fraction():.3f}")
    print()
    print("Peak |Delta_L| percentiles [m]:")
    for q in (50, 75, 90, 95, 99):
        print(f"  P{q:2d}: {np.percentile(res.dL_peak_abs, q):.3f}")
    print()

    # --- (3) Starting-state sensitivity table ---
    sens = starting_state_sensitivity(res)
    print(f"Starting-state sensitivity (linear regression of signed peak on "
          f"x(0-); R^2 = {sens['r2']:.3f}):")
    print(f"  {'component':<14s} {'sigma':>10s}   "
          f"{'beta':>14s}   {'peak per 1-sigma [m]':>22s}")
    for label, sig, b, bps in zip(
        sens["labels"], sens["sigmas"], sens["beta"], sens["beta_per_sigma"]
    ):
        print(f"  {label:<14s} {sig:>10.4f}   {b:>14.4f}   {bps:>22.4f}")
    print()

    # --- Plots ---
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.30)

    # (1) CDF of peak |Delta_L|
    ax1 = fig.add_subplot(gs[0, 0])
    s, F = res.cdf_peak_abs()
    ax1.plot(s, F, "b-", lw=2, label="MC empirical CDF")
    # Linearised P2 reference: peak |dL_lin_mean| + 1.96*max(L_std_linear)
    # Crude: a single point at the analytic 1.96-sigma envelope max excursion
    dL_lin_mean = res.L_mean_linear - L0
    lin_peak_95 = float(np.max(np.abs(dL_lin_mean) + 1.96 * res.L_std_linear))
    ax1.axvline(lin_peak_95, color="g", ls="--", lw=1,
                label=f"Linearised P2 95% peak ~ {lin_peak_95:.2f} m")
    # Hard end-stop margins
    margin_low = L0 - gw.telescope_min
    margin_high = gw.telescope_max - L0
    margin_min = min(margin_low, margin_high)
    ax1.axvline(margin_min, color="r", ls="-", lw=1.5,
                label=f"Closest end-stop margin = {margin_min:.2f} m")
    ax1.set_xlabel(r"Peak $|\Delta L|$ [m]")
    ax1.set_ylabel("Empirical CDF")
    ax1.set_title("(1) Distribution of WCF peak telescope excursion\n"
                  f"over {n_samples} starting-state samples")
    ax1.grid(True, alpha=0.4)
    ax1.legend(loc="lower right", fontsize=9)

    # (2) Percentile bands on L(t) overlaid on linearised envelope
    ax2 = fig.add_subplot(gs[0, 1])
    bands = res.percentile_bands_L((1, 5, 25, 50, 75, 95, 99))
    t = res.t
    ax2.fill_between(t, bands[1], bands[99], color="C0", alpha=0.15,
                     label="MC 1-99 % band")
    ax2.fill_between(t, bands[5], bands[95], color="C0", alpha=0.30,
                     label="MC 5-95 % band")
    ax2.fill_between(t, bands[25], bands[75], color="C0", alpha=0.50,
                     label="MC 25-75 % band")
    ax2.plot(t, bands[50], "C0-", lw=2, label="MC median")
    # Linearised baseline
    ax2.plot(t, res.L_mean_linear, "g-", lw=2, label="Linearised mean")
    ax2.fill_between(
        t, res.L_mean_linear - 1.96 * res.L_std_linear,
        res.L_mean_linear + 1.96 * res.L_std_linear,
        color="g", alpha=0.10,
        label=r"Linearised mean $\pm 1.96\sigma$",
    )
    ax2.axhline(gw.telescope_min, color="r", ls="--", lw=1, label=f"L_min = {gw.telescope_min:.0f} m")
    ax2.axhline(gw.telescope_max, color="r", ls="--", lw=1, label=f"L_max = {gw.telescope_max:.0f} m")
    ax2.set_xlabel("t [s] after WCF")
    ax2.set_ylabel("Telescope length L(t) [m]")
    ax2.set_title("(2) MC percentile bands on L(t), with linearised baseline")
    ax2.grid(True, alpha=0.4)
    ax2.legend(loc="best", fontsize=8, ncol=2)

    # (3) Sensitivity bar chart (per-sigma)
    ax3 = fig.add_subplot(gs[1, 0])
    yp = np.arange(len(sens["labels"]))
    bars = sens["beta_per_sigma"]
    colors = ["C2" if b < 0 else "C3" for b in bars]
    ax3.barh(yp, bars, color=colors, alpha=0.7)
    ax3.set_yticks(yp)
    ax3.set_yticklabels(sens["labels"])
    ax3.axvline(0.0, color="k", lw=0.5)
    ax3.set_xlabel(r"$\partial \mathrm{peak}\,\Delta L / \partial \sigma_i$ [m]")
    ax3.set_title("(3) Per-1$\\sigma$ contribution of each starting-state component to the WCF peak"
                  f"\n(linear regression $R^2$ = {sens['r2']:.2f})")
    ax3.grid(True, alpha=0.4, axis="x")

    # Bottom-right: scatter of (eta_e at t=0-) vs peak Delta_L (the dominant pair)
    # to make the sensitivity table concrete.
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(res.x0_samples[:, 1], res.dL_peak, s=8, alpha=0.4, c="C0")
    ax4.set_xlabel("Sway offset at WCF instant [m]")
    ax4.set_ylabel(r"signed peak $\Delta L$  [m]")
    ax4.axhline(0.0, color="k", lw=0.5)
    ax4.axvline(0.0, color="k", lw=0.5)
    ax4.set_title("Scatter: starting sway offset vs signed peak\n"
                  "(visualises the dominant sensitivity)")
    ax4.grid(True, alpha=0.4)

    fig.suptitle(
        f"WCFDI starting-state Monte-Carlo  |  CSOV  |  "
        f"Vw={Vw_mean} m/s, Hs={Hs} m, beam, $\\alpha$={scenario.alpha[0]}, "
        f"$T_{{realloc}}$={scenario.T_realloc} s",
        y=0.995, fontsize=11,
    )
    out = os.path.join(HERE, "csov_wcfdi_mc.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
