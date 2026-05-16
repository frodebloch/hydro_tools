"""Diagnostic: tau_cmd_mean/sigma_tau_cmd/cap(t) vs brucon Order_y peaks
(sec.12.21.21.17).

Tests the user's hypothesis (sec.12.21.21.17): regime-B saturation in
bf8_q10_w45 is driven by **peaks in the slowly varying environmental
force** riding on top of the post-WCF mean -- not by a slow drift of the
mean itself toward cap. The brief deterministic transient (deficit
pulse + IC re-init) only amplifies the env-load peaks for a short
window.

What the script does, for two cells (one severe, one safe):

  1. Run ``wcfdi_transient`` with brucon polytope + calibrated lost-bus.
  2. Pull out ``tau_cmd_mean(t)``, ``sigma_tau_cmd(t)``, ``cap(t)`` --
     the three series that go into the bistability severity score.
  3. Overlay the actual brucon ensemble of OrderTauSway(t) traces
     centred on the WCF event (post-WCF window 0..200 s).
  4. Mark the residual polytope cap and count empirical exceedances.

If the user's hypothesis is right we should see:
  * cqa's tau_cmd_mean_y(t) stays well inside the cap for both cells.
  * brucon's Order_y(t) ensemble has wide stochastic spread, and in
    bf8_q10_w45 a non-trivial fraction of seeds spend regime-B time
    above cap, while bf8_h0 (safe) stays inside.
  * cqa's sigma_tau_cmd_y(t) is significantly *smaller* than the
    empirical std of the brucon Order_y ensemble post-WCF -- because
    the linearised covariance uses the intact A_cl (per transient.py:909).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))
sys.path.insert(0, str(THIS))

from cqa.transient import wcfdi_transient                    # noqa: E402
from cqa.rao import load_pdstrip_rao                         # noqa: E402

from run_comparison import setup_cqa                         # noqa: E402
from wcfdi_transient_regime_b_check import (                 # noqa: E402
    _scenario_for_cell, _theta_rel_for_cell, CELLS, PDSTRIP_PATH,
    _RESIDUAL_CAP_N_NM, POST_FAILURE_S, N_T,
)
from saturation_regime_scan import load_seed                 # noqa: E402

T_WCF_S = 1560.0
T_POST_WIN = 200.0
SEEDS = range(1000, 1030)

CELLS_TO_PLOT = ["bf8_q10_w45", "bf8_h0", "bf6_h0"]


def cqa_severity_traces(cfg, rao, tag: str):
    """Return (t_post, tau_cmd_mean[N,3], sigma_tau_cmd[N,3], cap[N,3]).

    Mirrors the calculation in transient.py:955-967 but exposes the
    pre-max series so we can plot them.
    """
    from cqa.transient import (
        LinearVesselModel, LinearDpController, build_augmented_system,
    )
    scenario = _scenario_for_cell(tag)
    _, env, _, _, _ = CELLS[tag]
    theta_rel = _theta_rel_for_cell(tag)
    Vw_in = 0.0 if tag == "pwq30" else float(env["Vw"])
    Vc_in = 0.0 if tag == "pwq30" else float(env["Vc"])
    tr = wcfdi_transient(
        cfg=cfg, Vw_mean=Vw_in,
        Hs=float(env["Hs"]), Tp=float(env["Tp"]),
        Vc=Vc_in, theta_rel=theta_rel, scenario=scenario,
        t_end=POST_FAILURE_S, n_t=N_T, rao_table=rao,
    )

    # Reconstruct K_tau the same way transient.py:955 does, and recompute
    # tau_cmd_mean / sigma_tau_cmd from x_mean / P_t which aren't returned
    # by wcfdi_transient. So we have to re-build the augmented A to get
    # Kp/Kd/Ki dimensions consistent.
    # ... actually simpler: rerun the inner pieces. Re-create vessel +
    # controller from the cfg to recover Kp/Kd, then evaluate K_tau against
    # the returned x_mean (in TransientResult.x_mean).
    vp = cfg.vessel
    cp_ctrl = cfg.controller
    vessel = LinearVesselModel.from_config(vp)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D, omega_n=cp_ctrl.omega_n, zeta=cp_ctrl.zeta,
    )
    aug = build_augmented_system(
        vessel, controller, T_b=cp_ctrl.bias_time_constant_s,
        T_thr=cp_ctrl.thruster_time_constant_s,
    )
    n_aug = aug.n_state

    K_tau = np.zeros((3, n_aug))
    K_tau[:, 0:3] = -aug.Kp
    K_tau[:, 3:6] = -aug.Kd
    K_tau[:, 6:9] = -np.eye(3)
    if aug.include_integrator:
        K_tau[:, 12:15] = -aug.Ki

    x_mean = tr.x_mean
    P = tr.P
    tau_cmd_mean = (K_tau @ x_mean.T).T
    tau_cmd_var = np.einsum("ij,tjk,lk->til", K_tau, P, K_tau)
    sigma_tau_cmd = np.sqrt(np.maximum(
        np.diagonal(tau_cmd_var, axis1=1, axis2=2), 0.0
    ))
    cap = np.array([scenario.cap_at_time(float(tt), cfg) for tt in tr.t])
    return tr.t, tau_cmd_mean, sigma_tau_cmd, cap


def brucon_ensemble_order_y(tag: str):
    """Pull OrderTauSway(t) ensembles, return (t_post, Y[seeds, N])."""
    rows = []
    for s in SEEDS:
        d = load_seed(tag, s)
        if d is None:
            continue
        t = d["t"] - T_WCF_S
        m = (t >= 0) & (t <= T_POST_WIN)
        if not m.any():
            continue
        rows.append((t[m], d["OrderSway"][m]))
    if not rows:
        return None, None
    # Resample to common grid
    t_common = np.linspace(0, T_POST_WIN, 401)
    Y = np.array([np.interp(t_common, t, y) for t, y in rows])
    return t_common, Y


def main() -> None:
    cfg, _ = setup_cqa()
    print(f"[rao] loading {PDSTRIP_PATH}")
    rao = load_pdstrip_rao(PDSTRIP_PATH)

    cap_y = _RESIDUAL_CAP_N_NM[1]   # N

    fig, axes = plt.subplots(len(CELLS_TO_PLOT), 1,
                              figsize=(11, 3.2 * len(CELLS_TO_PLOT)),
                              sharex=True)

    for ax, tag in zip(axes, CELLS_TO_PLOT):
        t_cqa, mu_tau, sig_tau, cap_t = cqa_severity_traces(cfg, rao, tag)
        t_br, Y = brucon_ensemble_order_y(tag)

        # cqa prediction (sway DOF index 1)
        mu_y = mu_tau[:, 1] / 1e3
        sig_y = sig_tau[:, 1] / 1e3
        cap_pos = cap_t[:, 1] / 1e3

        # brucon ensemble. NOTE: brucon's .out file ALREADY stores forces
        # in kN (verified by force-balance check, sec.12.21.21.18:
        # pre-WCF Ty mean = +573 kN balances env_Y mean = -569 kN with
        # +4 kN residual). cqa internally uses N, so divide cqa by 1e3
        # but leave brucon as-is.
        if Y is not None:
            Y_kN = Y
            br_mean = Y_kN.mean(axis=0)
            br_std = Y_kN.std(axis=0)
            for k in range(Y_kN.shape[0]):
                ax.plot(t_br, Y_kN[k], color="0.7", lw=0.4, alpha=0.5)
            ax.plot(t_br, br_mean, "k-", lw=1.6,
                    label="brucon Order_y ensemble mean")
            ax.fill_between(t_br, br_mean - br_std, br_mean + br_std,
                            color="0.4", alpha=0.25,
                            label=r"brucon $\pm 1\sigma$ envelope")

        # cqa prediction
        ax.plot(t_cqa, mu_y, "C0-", lw=2.0,
                label=r"cqa $\mu_{\tau_y,\mathrm{cmd}}(t)$")
        ax.fill_between(t_cqa, mu_y - sig_y, mu_y + sig_y,
                        color="C0", alpha=0.25,
                        label=r"cqa $\pm 1 \sigma_{\tau_y,\mathrm{cmd}}(t)$")

        ax.axhline(+cap_y / 1e3, color="r", ls="--", lw=1.2,
                   label="residual polytope $|\\tau_y| = "
                         f"{cap_y/1e3:.0f}$ kN")
        ax.axhline(-cap_y / 1e3, color="r", ls="--", lw=1.2)

        # Empirical exceedance count
        if Y is not None:
            exceed_frac = (np.abs(Y_kN) > cap_y / 1e3).mean(axis=0)
            ax.text(0.99, 0.02,
                    f"empirical max exceedance frac over time = "
                    f"{exceed_frac.max():.2f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, color="r",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="r"))

        ax.set_title(f"{tag}: cqa tau_cmd vs brucon OrderTauSway")
        ax.set_ylabel("tau_y [kN]")
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1].set_xlabel("t - t_WCF [s]")
    out = THIS / "wcfdi_transient_tau_y_overlay.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
