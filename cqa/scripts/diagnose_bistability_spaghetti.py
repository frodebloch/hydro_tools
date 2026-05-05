"""Bistability spaghetti plot for analysis.md §12.14.

Visualises the saturation-bistability finding: at a beam operating
point inside the bistability band (V_w = 14 m/s, theta = 90 deg, CSOV
defaults), the deterministic mean-ODE prediction lies on the
recovering branch, but a non-trivial fraction of stochastic
realisations diverges onto a coexisting runaway branch.

Two-panel figure (sway only, since the bistability is sway-dominated
at beam):

  (a) Per-realisation eta_y(t) traces, colour-coded by recovery
      (green: |eta_y(t_end)| < threshold; red: diverged), with the
      deterministic linearised-predictor mean overlaid in bold.
  (b) Histogram of per-realisation peak |eta_y|, with the
      deterministic predicted peak and the recovery threshold marked.

Run:
    python scripts/diagnose_bistability_spaghetti.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt

from cqa.config import csov_default_config
from cqa.transient import WcfdiScenario, wcfdi_transient
from cqa.wcfdi_self_mc import wcfdi_self_mc


def main() -> None:
    cfg = csov_default_config()

    # Operating point inside the bistability band (per session §12.14).
    Vw = 14.0
    theta = np.pi / 2
    Hs = 0.21 * Vw
    Tp = max(4.0, 4.0 * np.sqrt(Hs))
    Vc = 0.5

    scenario = WcfdiScenario(
        alpha=(2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
        gamma_immediate=0.5,
        T_realloc=10.0,
    )

    M = 128
    t_end = 400.0
    n_t = 801  # 0.5 s grid
    recovery_threshold = 5.0  # m on |eta_y(t_end)|

    print(
        f"Running self-MC: V_w={Vw} m/s, theta={np.degrees(theta):.0f} deg, "
        f"Hs={Hs:.2f} m, Tp={Tp:.2f} s, Vc={Vc} m/s, M={M}, t_end={t_end} s"
    )

    res = wcfdi_self_mc(
        cfg,
        Vw_mean=Vw, Hs=Hs, Tp=Tp, Vc=Vc, theta_rel=theta,
        scenario=scenario,
        n_realisations=M,
        t_end=t_end, n_t=n_t,
        seed=0,
        return_realisations=True,
        progress_cb=lambda k, n: print(f"  realisation {k}/{n}", end="\r"),
    )
    print()

    eta_all = res.eta_realisations  # (M, n_t, 3)
    assert eta_all is not None
    eta_y = eta_all[:, :, 1]  # sway

    eta_y_end = eta_y[:, -1]
    diverged = np.abs(eta_y_end) > recovery_threshold
    n_diverged = int(diverged.sum())
    print(
        f"  diverged: {n_diverged}/{M} = {100.0 * n_diverged / M:.1f}%  "
        f"(threshold |eta_y(t_end)| > {recovery_threshold} m)"
    )

    # Linear-predictor reference (already inside res, but call once for
    # extra info: bistability score).
    lin = wcfdi_transient(
        cfg, Vw_mean=Vw, Hs=Hs, Tp=Tp, Vc=Vc, theta_rel=theta,
        scenario=scenario, t_end=t_end, n_t=n_t,
    )
    score = lin.info.get("bistability_risk_score", float("nan"))
    print(f"  bistability_risk_score = {score:.2f}")

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]},
    )
    t = res.t

    # Panel (a): spaghetti
    for k in range(M):
        c = "#cc3333" if diverged[k] else "#2a8a2a"
        ax1.plot(t, eta_y[k], color=c, alpha=0.25, lw=0.6)
    # Deterministic predictor
    ax1.plot(
        t, res.eta_mean_lin[:, 1], color="black", lw=2.2,
        label="deterministic mean (linear predictor)",
    )
    # Empirical mean for reference (drifts with the diverging fraction)
    ax1.plot(
        t, res.eta_mean_emp[:, 1], color="black", lw=1.2, ls="--",
        label="empirical ensemble mean",
    )
    ax1.axhline(recovery_threshold, color="grey", ls=":", lw=0.8)
    ax1.axhline(-recovery_threshold, color="grey", ls=":", lw=0.8)
    ax1.set_xlabel("time after WCFDI [s]")
    ax1.set_ylabel(r"sway $\eta_y$ [m]")
    ax1.set_title(
        f"Per-realisation sway trajectories (M={M})\n"
        f"red: diverged ({n_diverged}/{M}={100.0 * n_diverged / M:.0f}%), "
        f"green: recovered"
    )
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel (b): histogram of per-realisation peak |eta_y|
    peak_y = np.max(np.abs(eta_y), axis=1)
    bins = np.linspace(0, max(peak_y.max() * 1.05, 10.0), 40)
    ax2.hist(
        peak_y[~diverged], bins=bins, color="#2a8a2a", alpha=0.7,
        label=f"recovered ({(~diverged).sum()})",
    )
    ax2.hist(
        peak_y[diverged], bins=bins, color="#cc3333", alpha=0.7,
        label=f"diverged ({diverged.sum()})",
    )
    det_peak = float(np.max(np.abs(res.eta_mean_lin[:, 1])))
    ax2.axvline(
        det_peak, color="black", lw=2.0,
        label=f"deterministic peak = {det_peak:.2f} m",
    )
    ax2.axvline(
        recovery_threshold, color="grey", ls=":", lw=1.0,
        label=f"recovery threshold = {recovery_threshold:.0f} m",
    )
    ax2.set_xlabel(r"peak $|\eta_y|$ [m]")
    ax2.set_ylabel("count")
    ax2.set_title("Per-realisation peak distribution")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Saturated-WCFDI bistability spaghetti  "
        f"(CSOV, V_w={Vw} m/s beam, alpha=2/3, score={score:.2f})",
        fontsize=12,
    )
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "csov_wcfdi_bistability_spaghetti.png"
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
