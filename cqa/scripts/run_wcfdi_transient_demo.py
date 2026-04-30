"""Demo: WCFDI transient at the worst weather direction for the CSOV.

Operating point: representative W2W weather, vessel held with weather on the beam
(typical worst direction from the P1 polar).

Two failure scenarios shown side-by-side:
  - alpha = 0.5 (lose half capability per DOF, baseline DP-class-2 WCFDI)
  - alpha = 0.7 (smaller failure, e.g. one of three thruster groups)

Plots:
  Top row: eta_n, eta_e, psi vs time, mean +/- 1.96 sigma envelope.
  Bottom row: predicted telescope length deviation Delta L = e_L^T eta_rc(t)
              vs time, with end-stop margins, for a representative
              gangway-pointing direction.

Run:
    python scripts/run_wcfdi_transient_demo.py
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
    wcfdi_transient,
    WcfdiScenario,
    GangwayJointState,
    telescope_sensitivity,
    evaluate_operability,
)


def main() -> None:
    cfg = csov_default_config()

    Vw_mean = 14.0
    Hs = 2.8
    Tp = 9.0
    Vc = 0.5
    theta_rel = np.pi / 2.0  # beam (worst from the P1 polar)

    # Two scenarios at the same CQA-guarded operating point. Both lose 20%
    # of intact capability post-failure (alpha=0.8, well within the static
    # CQA envelope at this weather). The two scenarios differ in how
    # quickly the surviving thrusters reallocate:
    scenarios = [
        ("WCFDI: alpha=0.8, T_realloc=5 s",
         WcfdiScenario(alpha=(0.8, 0.8, 0.8), T_realloc=5.0)),
        ("WCFDI: alpha=0.8, T_realloc=20 s",
         WcfdiScenario(alpha=(0.8, 0.8, 0.8), T_realloc=20.0)),
    ]

    # Gangway operating point: pointing toward port (alpha_g = -pi/2) so
    # the landing point is on the lee side when the weather is on the
    # starboard beam. Mid-stroke length, mid-height rotation centre,
    # horizontal boom.
    gw = cfg.gangway
    L0 = 0.5 * (gw.telescope_min + gw.telescope_max)
    joint = GangwayJointState(
        h=gw.rotation_centre_height_above_base,
        alpha_g=-np.pi / 2.0,
        beta_g=0.0,
        L=L0,
    )
    c_L = telescope_sensitivity(joint, gw)  # 3-vector

    # Layout: 4 rows x N_scenarios cols. Rows: eta_n, eta_e, psi, telescope L.
    # Each pose component gets its own y-axis to avoid the dual-axis trap
    # (different units overlap visually if forced onto the same axes).
    n_col = len(scenarios)
    fig, axes = plt.subplots(4, n_col, figsize=(7 * n_col, 11), sharex=True)
    if n_col == 1:
        axes = axes.reshape(4, 1)

    for col, (label, scen) in enumerate(scenarios):
        res = wcfdi_transient(
            cfg,
            Vw_mean=Vw_mean,
            Hs=Hs,
            Tp=Tp,
            Vc=Vc,
            theta_rel=theta_rel,
            scenario=scen,
        )
        lo, hi = res.env_lower_upper(k=1.96)

        # Row 0: surge offset (eta_n in setpoint-aligned frame)
        ax = axes[0, col]
        ax.plot(res.t, res.eta_mean[:, 0], "b-", lw=1.5, label=r"mean")
        ax.fill_between(res.t, lo[:, 0], hi[:, 0], alpha=0.25, color="b",
                        label=r"$\pm 1.96\sigma$")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("Surge offset [m]")
        ax.set_title(f"{label}\n(Vw={Vw_mean:.0f} m/s, Hs={Hs:.1f} m, beam)")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="best", fontsize=8)

        # Row 1: sway offset (eta_e in setpoint-aligned frame)
        ax = axes[1, col]
        ax.plot(res.t, res.eta_mean[:, 1], "r-", lw=1.5, label=r"mean")
        ax.fill_between(res.t, lo[:, 1], hi[:, 1], alpha=0.25, color="r",
                        label=r"$\pm 1.96\sigma$")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("Sway offset [m]")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="best", fontsize=8)

        # Row 2: heading offset (psi, degrees)
        ax = axes[2, col]
        ax.plot(res.t, np.degrees(res.eta_mean[:, 2]), "g-", lw=1.5, label=r"mean")
        ax.fill_between(
            res.t,
            np.degrees(lo[:, 2]),
            np.degrees(hi[:, 2]),
            alpha=0.25, color="g",
            label=r"$\pm 1.96\sigma$",
        )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("Heading offset [deg]")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="best", fontsize=8)

        # Telescope projection via the gangway module: Delta_L(t) = c^T eta(t)
        dL_mean = res.eta_mean @ c_L
        sigma_dL = np.sqrt(
            np.maximum(np.einsum("i,nij,j->n", c_L, res.P[:, 0:3, 0:3], c_L), 0.0)
        )
        L_t = L0 + dL_mean
        L_lo = L_t - 1.96 * sigma_dL
        L_hi = L_t + 1.96 * sigma_dL

        # Row 3: telescope L
        ax = axes[3, col]
        ax.plot(res.t, L_t, "k-", lw=2, label="Telescope L mean")
        ax.fill_between(res.t, L_lo, L_hi, alpha=0.25, color="k",
                        label=r"$\pm 1.96\sigma$")
        ax.axhline(gw.telescope_min, color="r", ls="--", lw=1,
                   label=f"L_min = {gw.telescope_min:.0f} m")
        ax.axhline(gw.telescope_max, color="r", ls="--", lw=1,
                   label=f"L_max = {gw.telescope_max:.0f} m")
        ax.set_xlabel("t [s] after WCF")
        ax.set_ylabel("Telescope length [m]")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="best", fontsize=8)

        # Console summary
        peak_idx_lo = int(np.argmin(L_lo))
        peak_idx_hi = int(np.argmax(L_hi))
        # Operability at the peak time (use eta covariance and a zero
        # mean shift relative to L_t; we feed P_eta at the worst time so
        # the operability check sees the worst-case sigma_L).
        worst_t = peak_idx_hi if (L_hi[peak_idx_hi] - L0) > (L0 - L_lo[peak_idx_lo]) else peak_idx_lo
        P_eta_worst = res.P[worst_t, 0:3, 0:3]
        # Adjust nominal length for the operability check to L_t at worst time.
        op = evaluate_operability(
            joint, gw, P_eta_worst, k_sigma=1.96, L_nominal=float(L_t[worst_t])
        )
        print(f"\n=== {label} ===")
        print(f"  Pre-failure tau_env (surge, sway, yaw):  {res.info['tau_env']}")
        print(f"  Pre-failure tau_thr (surge, sway, yaw):  {res.info['x0_intact'][9:12]}")
        print(f"  Post-failure cap (per DOF):              {res.info['tau_cap_post']}")
        cqa_v = res.info['cqa_precondition_violated']
        if np.any(cqa_v):
            print(f"  *** CQA PRECONDITION VIOLATED in DOFs: {np.where(cqa_v)[0]}")
        else:
            print(f"  CQA precondition OK: |tau_env| <= cap_post in all DOFs.")
        print(f"  Step force imbalance Delta_tau:          {res.info['delta_tau_step']}")
        print(f"  Pre-failure pos sigma (n,e,psi):         "
              f"{res.info['P0_eta_diag'][0]:.3f} m, {res.info['P0_eta_diag'][1]:.3f} m, "
              f"{np.degrees(res.info['P0_eta_diag'][2]):.3f} deg")
        print(
            f"  Telescope:  min envelope = {L_lo[peak_idx_lo]:.2f} m at t = {res.t[peak_idx_lo]:.1f} s,"
            f"  max envelope = {L_hi[peak_idx_hi]:.2f} m at t = {res.t[peak_idx_hi]:.1f} s"
        )
        print(
            f"  Operability @ worst t={res.t[worst_t]:.1f}s: "
            f"L=[{op.L_lower:.2f}, {op.L_upper:.2f}] m, "
            f"margin_low={op.margin_low:+.2f} m, margin_high={op.margin_high:+.2f} m, "
            f"PASS={op.pass_endstops}"
        )

    fig.tight_layout()
    out = os.path.join(HERE, "csov_wcfdi_transient.png")
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
