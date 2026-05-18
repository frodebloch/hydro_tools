"""sec.12.21.21.29 follow-up: synthetic amber-regime demonstration.

Brucon's test matrix gives 3.9 sigma sway headroom (worst cell
bf8_q10_w45) on the yaw-priority conditional cap, so Option 2
correctly returns ~0 m P95 everywhere. This script demonstrates that
Option 2 returns non-trivial, monotonically growing P95 once the
demand approaches the conditional cap, closing the validation loop
that brucon structurally cannot.

Setup (mirrors the diagnostic on bf8_q10_w45 seed 1012):
    sigma_sway = 80 kN, cap_sway = 801 kN.
We sweep mu_sway across headroom ratios r = (cap - mu) / sigma in
{4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0} (the 4.0 row is the
real brucon operating point) and call:

    estimate_regime_b_severity     (synthetic tau_buffer matching
                                    target mu, sigma)
    estimate_post_wcf_excursion_distribution

at the same conditional cap. Saves a table and a 2-panel figure
(p_sat & P95 vs headroom).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("scripts/p7_brucon_validation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, ".")

from cqa.config import csov_default_config
from cqa.live_operator_view import _build_aug_for_live
from cqa.live_regime_b import (
    estimate_post_wcf_excursion_distribution,
    estimate_regime_b_severity,
)
import roll_up_live_operator_panel as roll


def make_synthetic_buffer(
    mu: np.ndarray, sigma: np.ndarray, fs_hz: float, window_s: float,
    seed: int = 0,
) -> np.ndarray:
    """Build a synthetic (N, 3) tau buffer with target moments.

    Uses Gauss-Markov forcing per DOF with tau_corr_s = 15 (LF), then
    rescales each column to the exact target sigma and shifts to the
    target mu. This makes the LF-filtered moments inside
    estimate_regime_b_severity recover (mu, sigma) faithfully.
    """
    rng = np.random.default_rng(seed)
    n = int(round(window_s * fs_hz))
    dt = 1.0 / fs_hz
    tau_corr = 15.0
    alpha = np.exp(-dt / tau_corr)
    out = np.zeros((n, 3))
    for j in range(3):
        x = 0.0
        col = np.empty(n)
        # OU innovation std s.t. steady-state var = 1
        sigma_eps = np.sqrt(1.0 - alpha * alpha)
        for k in range(n):
            x = alpha * x + sigma_eps * rng.standard_normal()
            col[k] = x
        # Rescale so empirical std after LF filtering matches target;
        # for a 15-s OU sampled at fs >= 1 Hz with omega_lp ~ 1 rad/s
        # the LF pass keeps essentially all the variance.
        s = col.std()
        if s > 0:
            col *= sigma[j] / s
        col += mu[j] - col.mean()
        out[:, k_dof := j] = col  # noqa: F841 (k_dof for clarity)
    return out


def main() -> None:
    fs_hz = 10.0
    window_s = 200.0

    # Anchor on the diagnostic measurements for bf8_q10_w45 seed 1012.
    sigma_sway = 80.0e3      # 80 kN
    cap_sway = 801.0e3       # 801 kN (yaw-priority conditional)
    # surge & yaw kept small / fixed so the conditional sway cap stays put.
    mu_surge = 0.0
    sigma_surge = 50.0e3
    # Mean yaw demand chosen so the yaw-priority conditional cap on sway
    # = ~ 800 kN (matches bf8_q10_w45 seed 1012's measured conditional cap).
    mu_yaw = 15.0e6          # 15 MN*m
    sigma_yaw = 5.0e6        # 5 MN*m
    # Decoupled surge cap (passes through unchanged).
    surge_cap_N = roll._REGB_SURGE_CAP_N
    geometry = roll._REGB_GEOMETRY

    cfg = csov_default_config()
    aug = _build_aug_for_live(cfg, Tp_obs_s=10.0)

    headroom_sigmas = np.array([4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0])
    rows = []
    for r in headroom_sigmas:
        mu_sway = cap_sway - r * sigma_sway
        mu = np.array([mu_surge, mu_sway, mu_yaw])
        sigma = np.array([sigma_surge, sigma_sway, sigma_yaw])

        tau_buf = make_synthetic_buffer(mu, sigma, fs_hz, window_s, seed=int(1000 + 10 * r))
        rb = estimate_regime_b_severity(
            tau_buffer=tau_buf, fs_hz=fs_hz,
            geometry=geometry, surge_cap_N=surge_cap_N,
        )
        excur = estimate_post_wcf_excursion_distribution(
            aug, mu=rb.mu, sigma=rb.sigma, cap_residual=rb.cap_residual,
            t_horizon_s=200.0,
        )
        rows.append({
            "r_sigma": r,
            "mu_sway_kN": rb.mu[1] / 1e3,
            "sigma_sway_kN": rb.sigma[1] / 1e3,
            "cap_sway_kN": rb.cap_residual[1] / 1e3,
            "p_sat_sway": float(rb.p_sat[1]),
            "traffic": rb.traffic,
            "mu_dtau_sway_kN": excur.mu_dtau[1] / 1e3,
            "sigma_dtau_sway_kN": excur.sigma_dtau[1] / 1e3,
            "sigma_eta_sway_m": excur.sigma_eta[1],
            "eta_sway_p95_m": excur.eta_p95[1],
            "eta_sway_p95_with_offset_m": excur.eta_p95_with_offset[1],
            "xy_p95_with_offset_m": excur.eta_xy_p95_with_offset,
        })

    # ---- table ----
    print()
    print("Synthetic amber-regime sweep (CSOV, post-WCF, yaw-priority cap):")
    print("  sigma_sway = 80 kN, cap_sway = 801 kN (= bf8_q10_w45 seed 1012 anchor)")
    print()
    hdr = (f"{'r_sig':>6} | {'mu_sw':>7} {'cap_sw':>7} | "
           f"{'p_sat':>9} {'traffic':>7} | "
           f"{'mu_dt':>7} {'sig_dt':>7} | "
           f"{'sig_eta':>9} {'eta_p95':>9} {'p95+off':>9} {'xy_p95':>9}")
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        print(
            f"{row['r_sigma']:>+6.1f} | "
            f"{row['mu_sway_kN']:>7.1f} {row['cap_sway_kN']:>7.1f} | "
            f"{row['p_sat_sway']:>9.2e} {row['traffic']:>7} | "
            f"{row['mu_dtau_sway_kN']:>7.2f} {row['sigma_dtau_sway_kN']:>7.2f} | "
            f"{row['sigma_eta_sway_m']:>9.3e} "
            f"{row['eta_sway_p95_m']:>9.3e} "
            f"{row['eta_sway_p95_with_offset_m']:>9.3e} "
            f"{row['xy_p95_with_offset_m']:>9.3e}"
        )

    # ---- figure ----
    r_arr = np.array([row['r_sigma'] for row in rows])
    p_sat = np.array([row['p_sat_sway'] for row in rows])
    p95_off = np.array([row['eta_sway_p95_with_offset_m'] for row in rows])
    xy_p95 = np.array([row['xy_p95_with_offset_m'] for row in rows])

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    ax = axs[0]
    ax.semilogy(r_arr, np.maximum(p_sat, 1e-12), 'o-', color='C0')
    ax.axhline(0.01, color='orange', ls='--', label='IMCA amber (p_sat=1e-2)')
    ax.axhline(0.10, color='red', ls='--', label='IMCA red (p_sat=1e-1)')
    ax.invert_xaxis()
    ax.set_xlabel('sway headroom r = (cap - mu) / sigma [-]')
    ax.set_ylabel('Regime-B p_sat (sway)')
    ax.set_title('Regime-B severity vs headroom')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')

    ax = axs[1]
    ax.semilogy(r_arr, np.maximum(p95_off, 1e-6), 'o-', color='C2',
                label='|eta_y| P95 (with mean offset)')
    ax.semilogy(r_arr, np.maximum(xy_p95, 1e-6), 's--', color='C3',
                label='|eta_xy| P95 (with mean offset)')
    ax.axvline(3.9, color='k', ls=':', alpha=0.5,
               label='bf8_q10_w45 real operating point (3.9 sigma)')
    ax.invert_xaxis()
    ax.set_xlabel('sway headroom r = (cap - mu) / sigma [-]')
    ax.set_ylabel('Option-2 P95 excursion [m]')
    ax.set_title('Option-2 excursion vs headroom (T_h = 200 s)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')

    fig.suptitle(
        "Synthetic amber-regime demo: Option 2 in its named regime\n"
        "(Brucon test matrix sits at r = 3.9 sigma -- structurally green)",
        fontsize=10,
    )
    fig.tight_layout()
    out_png = ROOT / "synthetic_amber_demo_option2.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nWrote {out_png}")


if __name__ == "__main__":
    main()
