"""Validate the predicted CDF of the running maximum M_T against data.

Question: when the operator-facing panel says "with probability 0.90 the
running max stays below a_P90 over T_op", is that prediction *actually*
calibrated for our model + projection chain?

Two things can break the prediction even if sigma is right:
  1. Process not Gaussian (radial r is Rayleigh; drift forces have
     skewness; gangway projection is exactly linear so should be OK).
  2. Spectral shape parameters (nu_0+, q) misestimated -- wrong RAO
     heading, missing swell band, controller saturation, etc.

This script tests *shape* with sigma held at its model value: any
mismatch is then attributable to (1) or (2), not to a level error
(which the BayesianSigmaEstimator already handles).

Phase 1 (this script): bootstrap on a single 12 h realisation. Draw B
contiguous blocks of length T_op with replacement, take the max in each,
build the empirical CDF. Compare against the Rice/Vanmarcke predicted
CDF via Q-Q plot, KS distance, and coverage at standard quantiles.

Channels validated:
  * r(t)     -- vessel-footprint radial position. Predicted CDF uses the
                bilateral Gaussian Rice approximation (sigma_radial =
                sqrt(sigma_x^2 + sigma_y^2)). Since r is actually
                Rayleigh-distributed, this is the most interesting test.
  * dL(t)    -- telescope deviation, full band (LF + WF combined via
                independent multiband Rice).

Run:
    python scripts/validate_running_max_cdf.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
    radial_position_time_series,
    telescope_length_deviation_time_series,
    p_exceed_rice,
    p_exceed_rice_multiband,
)
from cqa.vessel import LinearVesselModel, CurrentForceModel
from cqa.controller import LinearDpController
from cqa.closed_loop import ClosedLoop
from cqa.psd import WindForceModel

PDSTRIP_PATH = Path.home() / "src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"


# ---------------------------------------------------------------------
# Bootstrap windowed-max sampler
# ---------------------------------------------------------------------

def bootstrap_max_blocks(
    x: np.ndarray,
    block_len: int,
    n_blocks: int,
    rng: np.random.Generator,
    abs_value: bool = True,
) -> np.ndarray:
    """Draw `n_blocks` contiguous blocks of length `block_len` from `x`
    with replacement and return the (possibly absolute) max in each.

    abs_value=True: uses max|x| inside each block (matches the bilateral
    Rice prediction). For the radial channel r(t) >= 0 this is a no-op.
    """
    N = x.size
    if block_len > N:
        raise ValueError(f"block_len={block_len} exceeds series length {N}")
    starts = rng.integers(low=0, high=N - block_len + 1, size=n_blocks)
    out = np.empty(n_blocks, dtype=float)
    if abs_value:
        for i, s in enumerate(starts):
            out[i] = float(np.max(np.abs(x[s:s + block_len])))
    else:
        for i, s in enumerate(starts):
            out[i] = float(np.max(x[s:s + block_len]))
    return out


# ---------------------------------------------------------------------
# Predicted CDF wrappers (Rice / Vanmarcke)
# ---------------------------------------------------------------------

def predicted_cdf_single_band(
    a_grid: np.ndarray, sigma: float, nu0: float, q: float | None,
    T: float, bilateral: bool,
) -> np.ndarray:
    """F_M(a) = 1 - P(|X|_max > a) over duration T, single band."""
    F = np.empty_like(a_grid)
    for i, a in enumerate(a_grid):
        r = p_exceed_rice(
            sigma=sigma, nu_0_plus=nu0, threshold=float(a),
            T=T, bilateral=bilateral, clustering="vanmarcke", q=q,
        )
        F[i] = 1.0 - r.p_breach
    return F


def predicted_cdf_multiband(
    a_grid: np.ndarray, bands: list[tuple], T: float, bilateral: bool,
) -> np.ndarray:
    """F_M(a) for the multiband-Rice combination (e.g. dL = LF + WF)."""
    F = np.empty_like(a_grid)
    for i, a in enumerate(a_grid):
        d = p_exceed_rice_multiband(
            bands=bands, threshold=float(a), T=T,
            bilateral=bilateral, clustering="vanmarcke",
        )
        F[i] = 1.0 - d["p_breach"]
    return F


def quantile_from_cdf(F: np.ndarray, a_grid: np.ndarray, p: float) -> float:
    """Invert a monotone-increasing CDF on a grid by linear interp."""
    # F(a_grid) must be non-decreasing; clamp tiny non-monotone numerical
    # noise.
    F_mono = np.maximum.accumulate(F)
    return float(np.interp(p, F_mono, a_grid))


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def ks_against_predicted(
    samples: np.ndarray, F_predicted_callable,
) -> tuple[float, float]:
    """One-sample KS test of `samples` against the predicted CDF.

    Returns (D, p_value). D is the KS statistic (sup |F_emp - F_pred|),
    p-value from the asymptotic Kolmogorov distribution (valid because
    the predicted CDF has no fitted parameters from the bootstrap
    sample).
    """
    res = stats.ks_1samp(samples, F_predicted_callable, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def coverage_table(
    samples: np.ndarray, a_pred_at_p: dict[float, float],
) -> dict[float, float]:
    """For each predicted P_p quantile, return the empirical coverage
    P(M <= a_pred_at_p[p]). Should equal p if calibrated.
    """
    return {
        p: float(np.mean(samples <= a))
        for p, a in a_pred_at_p.items()
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    cfg = csov_default_config()

    # --- Canonical CSOV operating point ----------------------------------
    Vw_mean = 14.0
    Hs = 2.8
    Tp = 9.0
    Vc = 0.5
    theta_rel = np.radians(30.0)

    gw = cfg.gangway
    L0 = 0.5 * (gw.telescope_min + gw.telescope_max)
    joint = GangwayJointState(
        h=gw.rotation_centre_height_above_base,
        alpha_g=-np.pi / 2.0, beta_g=0.0, L=L0,
    )

    if not PDSTRIP_PATH.exists():
        raise SystemExit(f"pdstrip data not found at {PDSTRIP_PATH}")
    rao_table = load_pdstrip_rao(PDSTRIP_PATH)
    wave = sigma_L_wave(joint, cfg, rao_table, Hs=Hs, Tp=Tp,
                        theta_wave_rel=theta_rel)
    sigma_L_wave_m = float(wave.sigma_L_wave)

    # --- Build closed loop + LF force PSDs ------------------------------
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

    # --- Realisation: 12 h at dt=0.5 s -----------------------------------
    T_realisation_s = 12.0 * 3600.0
    dt_s = 0.5
    t = np.arange(0.0, T_realisation_s, dt_s)
    N_t = t.size
    print(f"Realising {T_realisation_s/3600:.1f} h at dt={dt_s}s "
          f"({N_t} samples) ...")

    omega_grid_lf = np.geomspace(1.0e-4, 0.6, 256)
    rng = np.random.default_rng(7)

    F_lf = realise_vector_force_time_series(S_F_funcs, omega_grid_lf, t, rng)
    x_lf = integrate_closed_loop_response(cl, F_lf, t)
    xi_wf = realise_wave_motion_6dof(
        rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel,
        t=t, rng=rng,
    )

    r  = radial_position_time_series(x_lf, xi_wf, cfg)
    dL = telescope_length_deviation_time_series(x_lf, xi_wf, joint, cfg)

    print(f"  empirical sigma(r)  [sqrt E[r^2]] = "
          f"{np.sqrt(np.mean(r*r)):.3f} m")
    print(f"  empirical sigma(dL) [sqrt E[dL^2]]= "
          f"{np.sqrt(np.mean(dL*dL)):.3f} m")

    # --- Build the prior (model sigmas, nu_0+, q for each channel) -------
    T_op_s = 5.0 * 60.0
    prior = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint,
        T_op_s=T_op_s, sigma_L_wave=sigma_L_wave_m, Tp_wave_s=Tp,
        quantiles=(0.50, 0.90),
    )
    print(f"\nModel: sigma_radial={prior.pos_sigma_m:.3f} m, "
          f"nu_0+_pos={prior.pos_nu0_max*1000:.2f} mHz, "
          f"q_pos={prior.pos_q:.3f}")
    print(f"Model: sigma_slow={prior.gw_sigma_slow_m:.3f} m, "
          f"sigma_wave={prior.gw_sigma_wave_m:.3f} m, "
          f"nu_0+_slow={prior.gw_nu0_slow*1000:.2f} mHz, "
          f"q_slow={prior.gw_q_slow:.3f}, "
          f"nu_0+_wave={prior.gw_nu0_wave*1000:.1f} mHz, "
          f"q_wave={prior.gw_q_wave:.3f}")

    # --- Bootstrap windowed maxima ---------------------------------------
    block_len = int(round(T_op_s / dt_s))   # 600 samples for T_op=300s
    n_blocks = 5000
    rng_boot = np.random.default_rng(13)
    print(f"\nBootstrapping {n_blocks} blocks of {block_len} samples "
          f"({T_op_s} s) each ...")

    Mr_emp  = bootstrap_max_blocks(r,  block_len, n_blocks, rng_boot,
                                   abs_value=True)  # r >= 0 anyway
    MdL_emp = bootstrap_max_blocks(dL, block_len, n_blocks, rng_boot,
                                   abs_value=True)

    print(f"  Bootstrap max(r):  P50={np.median(Mr_emp):.3f} m, "
          f"P90={np.quantile(Mr_emp, 0.90):.3f} m, "
          f"P95={np.quantile(Mr_emp, 0.95):.3f} m, "
          f"max={Mr_emp.max():.3f} m")
    print(f"  Bootstrap max(dL): P50={np.median(MdL_emp):.3f} m, "
          f"P90={np.quantile(MdL_emp, 0.90):.3f} m, "
          f"P95={np.quantile(MdL_emp, 0.95):.3f} m, "
          f"max={MdL_emp.max():.3f} m")

    # --- Predicted CDFs ---------------------------------------------------
    a_grid_r  = np.linspace(0.05, max(Mr_emp.max(), 1.05*prior.pos_a_p90)*1.05, 400)
    a_grid_dL = np.linspace(0.05, max(MdL_emp.max(), 1.05*prior.gw_a_p90)*1.05, 400)

    # Radial channel: single-band bilateral Gaussian Rice
    # (this is the approximation the operator panel currently uses)
    F_pred_r = predicted_cdf_single_band(
        a_grid_r, sigma=prior.pos_sigma_m, nu0=prior.pos_nu0_max,
        q=prior.pos_q, T=T_op_s, bilateral=True,
    )

    # Telescope channel: two-band multiband Rice (LF + WF)
    dL_bands = [(prior.gw_sigma_slow_m, prior.gw_nu0_slow, prior.gw_q_slow)]
    if prior.gw_sigma_wave_m > 0.0 and prior.gw_nu0_wave > 0.0:
        dL_bands.append((prior.gw_sigma_wave_m, prior.gw_nu0_wave,
                         prior.gw_q_wave))
    F_pred_dL = predicted_cdf_multiband(
        a_grid_dL, dL_bands, T=T_op_s, bilateral=True,
    )

    # --- KS test ----------------------------------------------------------
    def F_pred_r_callable(a_array):
        return np.interp(a_array, a_grid_r,
                         np.maximum.accumulate(F_pred_r))

    def F_pred_dL_callable(a_array):
        return np.interp(a_array, a_grid_dL,
                         np.maximum.accumulate(F_pred_dL))

    D_r,  p_r  = ks_against_predicted(Mr_emp,  F_pred_r_callable)
    D_dL, p_dL = ks_against_predicted(MdL_emp, F_pred_dL_callable)
    print(f"\nKS test (empirical vs predicted CDF of M_T):")
    print(f"  radial r:    D={D_r:.4f}, p={p_r:.3g}")
    print(f"  telescope dL: D={D_dL:.4f}, p={p_dL:.3g}")
    print(f"  (D > 1.36/sqrt(N) = {1.36/np.sqrt(n_blocks):.4f} "
          f"=> reject calibration at 5%)")

    # --- Coverage table ---------------------------------------------------
    quantiles_check = [0.50, 0.75, 0.90, 0.95, 0.99]
    a_pred_r = {p: quantile_from_cdf(F_pred_r,  a_grid_r,  p) for p in quantiles_check}
    a_pred_dL = {p: quantile_from_cdf(F_pred_dL, a_grid_dL, p) for p in quantiles_check}
    cov_r  = coverage_table(Mr_emp,  a_pred_r)
    cov_dL = coverage_table(MdL_emp, a_pred_dL)

    print("\nCoverage check (empirical fraction of M_T <= predicted P_p):")
    print(f"  {'P':>6s} | {'a_pred_r':>9s} {'cov_r':>7s}  | "
          f"{'a_pred_dL':>10s} {'cov_dL':>7s}")
    for p in quantiles_check:
        print(f"  {p:>6.2f} | {a_pred_r[p]:>9.3f} {cov_r[p]:>7.3f}  | "
              f"{a_pred_dL[p]:>10.3f} {cov_dL[p]:>7.3f}")
    print("  (well-calibrated: empirical coverage equals nominal P)")

    # --- Q-Q plot ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, samples, F_pred, a_grid, label, color, a_pred_dict in [
        (axes[0], Mr_emp,  F_pred_r,  a_grid_r,  "radial r [m]",      "C0", a_pred_r),
        (axes[1], MdL_emp, F_pred_dL, a_grid_dL, "telescope dL [m]",  "C2", a_pred_dL),
    ]:
        # Theoretical quantiles at the empirical plotting positions.
        n = samples.size
        plot_pos = (np.arange(1, n + 1) - 0.5) / n  # Hazen
        emp_sorted = np.sort(samples)
        F_mono = np.maximum.accumulate(F_pred)
        theo = np.interp(plot_pos, F_mono, a_grid)

        ax.plot(theo, emp_sorted, ".", markersize=2, color=color, alpha=0.5,
                label="Q-Q points")
        lo = min(theo.min(), emp_sorted.min())
        hi = max(theo.max(), emp_sorted.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2,
                label="perfect calibration")

        # Mark predicted P50/P90/P95 with vertical/horizontal guides + empirical coverage.
        for p, color_q in [(0.50, "tab:gray"), (0.90, "tab:orange"), (0.95, "tab:red")]:
            a_p = a_pred_dict[p]
            cov = float(np.mean(samples <= a_p))
            ax.axvline(a_p, color=color_q, linewidth=0.8, alpha=0.6)
            ax.text(a_p, lo + 0.04*(hi - lo),
                    f"P{int(p*100)}\n cov={cov:.2f}",
                    color=color_q, fontsize=8, ha="left", va="bottom")

        ax.set_xlabel(f"predicted quantile of M_T  [{label.split('[')[1]}")
        ax.set_ylabel(f"empirical quantile of M_T  [{label.split('[')[1]}")
        ax.set_title(label.split(" [")[0])
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Running-max CDF validation (CSOV, Hs={Hs} Tp={Tp} theta=30deg, "
        f"T_op={T_op_s:.0f}s, bootstrap N={n_blocks} from {T_realisation_s/3600:.0f} h realisation)",
        fontsize=12,
    )
    fig.tight_layout()
    out = os.path.join(HERE, "validate_running_max_cdf.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
