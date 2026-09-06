"""Schematic of the post-WCF thrust recovery profile
(MTS DP Houston 2026 paper, Sec. 4.1).

Purely illustrative --- no numeric axes. Single pre-WCF operating
state at tau_pre; one exponential recovery envelope toward tau_pre
after the WCF; two different residual-capacity caps illustrate the
two operational regimes:

  (1) Healthy: tau_residual_cap sits above tau_pre. The exponential
      recovery curve reaches tau_pre without clipping. Classical CA
      reports green.
  (2) Saturated: tau_residual_cap sits below tau_pre. The same
      recovery envelope is clipped at the cap, asymptoting there
      instead of at tau_pre. Classical CA raises Force Off; the
      motion layer's saturation-probability (p_sat) channel of
      Sec. 4.4 raises the Large Transient alert.

Run:
    python scripts/run_tau_recovery_schematic.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    # Purely schematic units.
    tau_pre = 1.00                      # pre-WCF demand
    cap_healthy = 1.15                  # residual cap in the healthy regime
    cap_sat = 0.75                      # residual cap in the saturated regime
    gamma_imm = 0.45                    # immediately recovered fraction
    T_realloc = 1.0                     # exponential time constant
    t_wcf = 0.6

    # Time grids.
    t_pre = np.linspace(0.0, t_wcf, 40)
    t_post = np.linspace(t_wcf, t_wcf + 5.0 * T_realloc, 400)
    tau = t_post - t_wcf

    # Uncapped recovery envelope: beta(t) * tau_pre.
    beta = 1.0 + (gamma_imm - 1.0) * np.exp(-tau / T_realloc)
    curve_uncapped = beta * tau_pre
    curve_clipped_sat = np.minimum(curve_uncapped, cap_sat)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)

    # Single pre-WCF flat.
    ax.plot(t_pre, np.full_like(t_pre, tau_pre), color='0.25', lw=2.2)

    # Instantaneous step at WCF (shared).
    ax.plot([t_wcf, t_wcf], [tau_pre, gamma_imm * tau_pre],
            color='0.25', lw=2.2)

    # Healthy recovery: uncapped exponential to tau_pre.
    ax.plot(t_post, curve_uncapped, color='C0', lw=2.2,
            label='healthy: $\\tau_{\\mathrm{residual,cap}} \\geq \\tau_{\\mathrm{pre}}$ (classical CA green)')

    # Saturated recovery: same envelope clipped at cap_sat.
    ax.plot(t_post, curve_clipped_sat, color='C3', lw=2.2, ls=(0, (6, 3)),
            label='saturated: $\\tau_{\\mathrm{residual,cap}} < \\tau_{\\mathrm{pre}}$ (Force Off + $p_{\\mathrm{sat}}$)')

    # Cap reference lines.
    ax.axhline(cap_healthy, color='C0', ls=':', lw=1.0, alpha=0.7)
    ax.axhline(cap_sat, color='C3', ls=':', lw=1.0, alpha=0.7)
    ax.axhline(tau_pre, color='0.35', ls=':', lw=0.9, alpha=0.6)
    ax.axhline(0.0, color='k', lw=0.6)

    # WCF marker.
    ax.axvline(t_wcf, color='k', ls='--', lw=0.9, alpha=0.7)
    ax.text(t_wcf, -0.09, 'WCF', ha='center', va='top', fontsize=10)

    # T_realloc bracket, aligned with the recovery settling to
    # within ~5% of its healthy asymptote (i.e. ~3 T_realloc).
    t_settle = 3.0 * T_realloc
    y_bracket = -0.03
    ax.annotate(
        '', xy=(t_wcf, y_bracket),
        xytext=(t_wcf + t_settle, y_bracket),
        arrowprops=dict(arrowstyle='<->', color='0.25', lw=1.0),
    )
    ax.text(t_wcf + 0.5 * t_settle, y_bracket - 0.06,
            r'$\sim 3\,T_{\mathrm{realloc}}$ (characteristic time)',
            ha='center', va='top', fontsize=10, color='0.2')

    # tau_lost bracket on the step.
    x_arrow = t_wcf - 0.03
    ax.annotate(
        '', xy=(x_arrow, tau_pre),
        xytext=(x_arrow, gamma_imm * tau_pre),
        arrowprops=dict(arrowstyle='<->', color='0.25', lw=1.0),
    )
    ax.text(x_arrow - 0.02,
            0.5 * (tau_pre + gamma_imm * tau_pre),
            r'$\tau_{\mathrm{lost}}$', ha='right', va='center',
            fontsize=11)

    # Level labels on the right.
    x_label = t_post[-1] + 0.05
    ax.text(x_label, tau_pre,
            r'$\tau_{\mathrm{pre}}$',
            va='center', fontsize=10, color='0.25')
    ax.text(x_label, cap_healthy,
            r'$\tau_{\mathrm{residual,cap}}$ (healthy)',
            va='center', fontsize=10, color='C0')
    ax.text(x_label, cap_sat,
            r'$\tau_{\mathrm{residual,cap}}$ (saturated)',
            va='center', fontsize=10, color='C3')

    ax.set_xlabel('time')
    ax.set_ylabel('available thrust capability, per DOF')
    ax.set_title('Post-WCF thrust recovery profile (schematic)')
    ax.set_xlim(0.0, t_post[-1] + 2.4)
    ax.set_ylim(-0.20, cap_healthy * 1.20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    out_dir = Path.home() / 'Documents' / 'MTS_26' / 'images'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'tau_recovery_schematic.png'
    fig.savefig(out_path, dpi=200)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
