"""Per-seed time-trace comparison: brucon truth vs cqa live prediction.

For a given 12-cell tag (default bf8_q10_w45), plot on a single
radial-distance R(t) panel:

  * Brucon truth R(t) per seed (thin grey) -- vector-demeaned LF
    deviation, hypot(SurgeDev-pre, SwayDev-pre), aligned to t=0 at
    the WCF event time T_WCF=560 s.
  * Cqa per-seed predicted R(t) (thin C0) -- pulse_response (with
    lift coupling) on the live tau_lost = (1-beta)*b_hat applied at
    each seed's observed b_hat / eta_hat / sigma posterior, with the
    b_hat_bias_correction_factor from VesselParticulars applied
    (production live_decision pipeline).
  * Ensemble means: brucon (bold black) and cqa (bold C0).
  * Cqa per-instant sigma spread band: P50 and P95 of |eta_hat +
    delta_eta(t) + nu_total|, where nu_total is the 2D vector sum
    of three independent Gaussian halos -- LF, WF and b_hat -- each
    sized by the ensemble-mean Bayesian posterior at this cell. The
    per-instant band is for visual feel; the operator panel uses the
    Gumbel window-max formulation in _radial_window_max_quantiles.

Usage:
    .venv/bin/python scripts/p7_brucon_validation/time_traces_vs_brucon.py \\
        --tag bf8_q10_w45
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))  # cqa pkg
sys.path.insert(0, str(THIS))                # live_cell_per_seed_pwq30

import live_cell_per_seed_pwq30 as live_cell                  # noqa: E402
from live_cell_per_seed_pwq30 import (                         # noqa: E402
    _load_tsv, load_seed, build_live_sigma_posterior,
)

from cqa.config import csov_default_config                    # noqa: E402
from cqa.live_decision import _build_aug_for_live              # noqa: E402
from cqa.live_operator_view import _sigmas_intact_axis         # noqa: E402
from cqa.transient_obs import (                                # noqa: E402
    pulse_response, pulse_response_with_lift_coupling, N_STATE,
)
from cqa.transient import WcfdiScenario                        # noqa: E402


from _constants import T_WCF_S as T_WCF  # noqa: E402  # active script: refresh sec.12.21.16
T_PRE_S = 60.0       # default pre-WCF window length, override with --t-pre
T_HORIZON_S = 60.0   # post-WCF window from t=0 to t=T_HORIZON_S
N_T_POST = 241       # post-WCF time grid for cqa pulse response
TP_OBS_S = 10.0
N_MC = 2000
SEED_RANGE = (1000, 1030)


def load_brucon_truth_R(tag: str, t_grid_full: np.ndarray,
                        demean_win_s: float = 60.0) -> tuple[np.ndarray, list[int]]:
    """Return (R_seeds [n_seeds, n_t], seed_ids) on t_grid_full (which
    spans pre+post WCF, t=0 at T_WCF).

    R_seeds[i, k] = hypot(SurgeDev - sd_pre, SwayDev - wd_pre) for
    seed i resampled to absolute time T_WCF + t_grid_full[k]. Pre-WCF
    baseline computed over [T_WCF - demean_win_s - 1, T_WCF - 1].

    If demean_win_s <= 0, the raw SurgeDev/SwayDev are used (no
    demean). This is the apples-to-apples comparison against the
    cqa prediction, which is `eta_hat_xy + delta_eta(t)` in the
    same setpoint-relative body-frame coordinate system.
    """
    work = live_cell.WORK_ROOT
    R_list: list[np.ndarray] = []
    seed_ids: list[int] = []
    for seed in range(SEED_RANGE[0], SEED_RANGE[1]):
        seed_dir = work / f"{tag}_seed{seed:04d}"
        main_p = next((p for p in seed_dir.glob("*.out") if "estimator" not in p.name), None)
        if main_p is None:
            continue
        M = _load_tsv(main_p)
        if demean_win_s > 0:
            win_start = T_WCF - demean_win_s - 1.0
            win_end = T_WCF - 1.0
            win_m = (M["t"] >= win_start) & (M["t"] <= win_end)
            sd_pre = float(M["SurgeDev"][win_m].mean())
            wd_pre = float(M["SwayDev"][win_m].mean())
        else:
            sd_pre = 0.0
            wd_pre = 0.0
        t_abs = T_WCF + t_grid_full
        s_x = np.interp(t_abs, M["t"], M["SurgeDev"]) - sd_pre
        s_y = np.interp(t_abs, M["t"], M["SwayDev"]) - wd_pre
        R_list.append(np.hypot(s_x, s_y))
        seed_ids.append(seed)
    return np.asarray(R_list), seed_ids


def cqa_per_seed_traces(tag: str, t_grid_post: np.ndarray, t_grid_pre: np.ndarray,
                        calib_npz: Path):
    """Returns dict with per-seed cqa traces (pre+post WCF concatenated)
    and posterior summary.

    Pre-WCF (t<0): the live operator panel's intact-axis prediction is
    a constant snapshot R = |eta_hat_xy|, with a per-instant 2D
    Gaussian halo sized by (sigma_lf_x, sigma_lf_y) only. There is no
    WF / b_hat contribution on the intact bar because the LF
    posterior already absorbs the wave-frequency residual under
    intact closed-loop conditions. We mirror that here by producing
    a CONSTANT R_cqa = |eta_hat_xy| trace over t_grid_pre.

    Post-WCF (t>=0): the live operator panel's WCF-axis pulse response
    plus the three independent LF/WF/b_hat halos.
    """
    calib = np.load(calib_npz, allow_pickle=True)
    sigma_R_b_hat_m = float(calib["sigma_R_b_hat_m"])

    cfg = csov_default_config()
    K_lift = float(getattr(cfg.vessel, "lift_coupling_K_per_rad", 0.0))
    b_corr = float(cfg.vessel.b_hat_bias_correction_factor)

    scenario = WcfdiScenario(alpha=(2.0/3.0,)*3, gamma_immediate=0.5, T_realloc=10.0)
    aug = _build_aug_for_live(cfg, Tp_obs_s=TP_OBS_S)

    gamma_imm = float(scenario.gamma_immediate)
    T_realloc = float(scenario.T_realloc) if scenario.T_realloc > 0 else 1e-9
    beta_t = 1.0 + (gamma_imm - 1.0) * np.exp(-t_grid_post / T_realloc)

    n_pre = len(t_grid_pre)
    n_post = len(t_grid_post)
    R_list: list[np.ndarray] = []
    eta_xy_list: list[np.ndarray] = []  # signed body-frame (eta_hat + delta_eta) per seed, full window
    seed_ids: list[int] = []
    sig_lf_x_list, sig_lf_y_list = [], []
    sig_wf_x_list, sig_wf_y_list = [], []
    sig_bh_list = []
    for seed in range(SEED_RANGE[0], SEED_RANGE[1]):
        d = load_seed(seed)
        if d is None:
            continue
        sigma_post = build_live_sigma_posterior(d, sigma_R_b_hat_m=sigma_R_b_hat_m)
        eta_hat_lf = np.asarray(d["eta_hat"], dtype=float)
        b_hat = b_corr * np.asarray(d["b_hat"], dtype=float)
        tau_lost = (beta_t[:, None] - 1.0) * (-b_hat[None, :])

        if K_lift > 0.0:
            X = pulse_response_with_lift_coupling(
                aug, t_grid_post, tau_lost, b_hat0=b_hat, K_lift=K_lift,
                x0=np.zeros(N_STATE),
            )
        else:
            X = pulse_response(aug, t_grid_post, tau_lost, x0=np.zeros(N_STATE))
        delta_eta_post = X[:, 0:3]
        eta_xy_post = eta_hat_lf[None, 0:2] + delta_eta_post[:, 0:2]  # (n_post, 2)

        # Pre-WCF: cqa intact prediction is the constant snapshot.
        eta_xy_pre = np.tile(eta_hat_lf[None, 0:2], (n_pre, 1))  # (n_pre, 2)

        eta_xy_full = np.vstack([eta_xy_pre, eta_xy_post])       # (n_pre+n_post, 2)
        R_full = np.hypot(eta_xy_full[:, 0], eta_xy_full[:, 1])
        R_list.append(R_full)
        eta_xy_list.append(eta_xy_full)
        seed_ids.append(seed)

        sig_lf_x, sig_lf_y = _sigmas_intact_axis(sigma_post)
        sig_lf_x_list.append(sig_lf_x)
        sig_lf_y_list.append(sig_lf_y)
        sig_wf_x_list.append(float(sigma_post.posterior_wf_x.sigma_median))
        sig_wf_y_list.append(float(sigma_post.posterior_wf_y.sigma_median))
        sig_bh_list.append(float(sigma_post.sigma_R_b_hat_m) / float(np.sqrt(2.0)))

    return dict(
        R_seeds=np.asarray(R_list),
        eta_xy_seeds=np.asarray(eta_xy_list),  # (n_seeds, n_pre+n_post, 2)
        seed_ids=seed_ids,
        sigma_R_b_hat_m=sigma_R_b_hat_m,
        sig_lf_x=float(np.mean(sig_lf_x_list)),
        sig_lf_y=float(np.mean(sig_lf_y_list)),
        sig_wf_x=float(np.mean(sig_wf_x_list)),
        sig_wf_y=float(np.mean(sig_wf_y_list)),
        sig_bh_axis=float(np.mean(sig_bh_list)),
        n_pre=n_pre,
    )


def per_instant_band_segmented(eta_xy_mean_t: np.ndarray,
                               n_pre: int,
                               sig_lf_x: float, sig_lf_y: float,
                               sig_wf_x: float, sig_wf_y: float,
                               sig_bh_axis: float,
                               n_mc: int = N_MC,
                               rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-instant P50 and P95 of |eta_xy_mean_t + nu_total|.

    Pre-WCF (k < n_pre): nu_total uses LF halo ONLY -- this matches
    the live operator panel's intact-axis treatment, where the LF
    posterior is already the steady-state envelope of station-keeping
    error under intact closed-loop conditions.

    Post-WCF (k >= n_pre): nu_total adds LF + WF + b_hat 2D Gaussian
    halos, matching the live operator panel's WCF-axis halo (but
    per-instant, not Gumbel window-max).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_t = eta_xy_mean_t.shape[0]
    p50 = np.zeros(n_t)
    p95 = np.zeros(n_t)
    for k in range(n_t):
        nu_x = rng.standard_normal(n_mc) * sig_lf_x
        nu_y = rng.standard_normal(n_mc) * sig_lf_y
        if k >= n_pre:
            nu_x = nu_x + rng.standard_normal(n_mc) * sig_wf_x
            nu_y = nu_y + rng.standard_normal(n_mc) * sig_wf_y
            nu_x = nu_x + rng.standard_normal(n_mc) * sig_bh_axis
            nu_y = nu_y + rng.standard_normal(n_mc) * sig_bh_axis
        R = np.hypot(eta_xy_mean_t[k, 0] + nu_x, eta_xy_mean_t[k, 1] + nu_y)
        p50[k] = float(np.quantile(R, 0.50))
        p95[k] = float(np.quantile(R, 0.95))
    return p50, p95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bf8_q10_w45")
    ap.add_argument("--t-pre", type=float, default=T_PRE_S,
                    help=f"Pre-WCF window length in s (default {T_PRE_S}); "
                         f"max ~{T_WCF-10:.0f} given sim starts at t=0 and "
                         f"T_WCF={T_WCF:.0f} s")
    ap.add_argument("--demean-window", type=float, default=60.0,
                    help="Pre-WCF demean baseline window length in s, "
                         "anchored to t in [T_WCF - demean_window - 1, T_WCF - 1] "
                         "(default 60). Set <=0 to disable demean and use "
                         "raw SurgeDev/SwayDev directly (apples-to-apples "
                         "vs cqa's setpoint-relative coordinate).")
    args = ap.parse_args()
    tag = args.tag
    t_pre_s = float(args.t_pre)
    demean_win_s = float(args.demean_window)

    live_cell.TAG = tag
    live_cell.T_WCF = T_WCF
    live_cell.T_EVAL = T_WCF - 5.0
    live_cell.WIN_END = T_WCF - 1.0
    live_cell.WIN_START = live_cell.WIN_END - live_cell.WIN_S
    live_cell.SEEDS = list(range(SEED_RANGE[0], SEED_RANGE[1]))
    live_cell.CALIB_NPZ = THIS / f"scenario_{tag}_calibration.npz"

    # Time grids
    dt = T_HORIZON_S / (N_T_POST - 1)
    t_grid_pre = np.arange(-t_pre_s, 0.0, dt)
    t_grid_post = np.linspace(0.0, T_HORIZON_S, N_T_POST)
    t_grid_full = np.concatenate([t_grid_pre, t_grid_post])
    n_pre = len(t_grid_pre)

    # ---- brucon truth ----
    R_truth, ids_truth = load_brucon_truth_R(tag, t_grid_full, demean_win_s)
    R_truth_mean = R_truth.mean(axis=0)
    demean_label = (f"demean baseline = last {demean_win_s:.0f} s"
                    if demean_win_s > 0 else "RAW (no demean, setpoint=0)")
    print(f"Brucon truth: {len(ids_truth)} seeds, "
          f"pre-WCF mean R over [-{t_pre_s:.0f},0] = {R_truth[:, :n_pre].mean():.2f} m, "
          f"post-WCF peak mean = {R_truth_mean[n_pre:].max():.2f} m  "
          f"({demean_label})")

    # ---- cqa ----
    cqa = cqa_per_seed_traces(tag, t_grid_post, t_grid_pre, live_cell.CALIB_NPZ)
    R_cqa = cqa["R_seeds"]
    eta_xy_cqa_mean = cqa["eta_xy_seeds"].mean(axis=0)  # (n_full, 2)
    R_cqa_mean = np.hypot(eta_xy_cqa_mean[:, 0], eta_xy_cqa_mean[:, 1])
    print(f"Cqa:          {len(cqa['seed_ids'])} seeds, "
          f"pre-WCF mean R = {R_cqa[:, :n_pre].mean():.2f} m, "
          f"post-WCF peak mean = {R_cqa_mean[n_pre:].max():.2f} m")
    print(f"Posterior means: sigma_lf=({cqa['sig_lf_x']:.2f}, {cqa['sig_lf_y']:.2f}) m, "
          f"sigma_wf=({cqa['sig_wf_x']:.2f}, {cqa['sig_wf_y']:.2f}) m, "
          f"sigma_bh_axis={cqa['sig_bh_axis']:.2f} m")

    # ---- per-instant band ----
    rng = np.random.default_rng(0)
    band_p50, band_p95 = per_instant_band_segmented(
        eta_xy_cqa_mean, n_pre,
        cqa["sig_lf_x"], cqa["sig_lf_y"],
        cqa["sig_wf_x"], cqa["sig_wf_y"],
        cqa["sig_bh_axis"],
        n_mc=N_MC, rng=rng,
    )
    print(f"Cqa per-instant band: pre P95 = {band_p95[:n_pre].max():.2f} m, "
          f"post P95 peak = {band_p95[n_pre:].max():.2f} m")

    # ---- plot ----
    fig, ax = plt.subplots(1, 1, figsize=(11, 6.5))

    # background per-seed traces
    for i in range(R_truth.shape[0]):
        ax.plot(t_grid_full, R_truth[i], color="0.55", lw=0.6, alpha=0.55,
                label="brucon per-seed" if i == 0 else None)
    for i in range(R_cqa.shape[0]):
        ax.plot(t_grid_full, R_cqa[i], color="C0", lw=0.6, alpha=0.45,
                label="cqa per-seed" if i == 0 else None)

    # ensemble means
    ax.plot(t_grid_full, R_truth_mean, color="k", lw=2.2, label="brucon mean")
    ax.plot(t_grid_full, R_cqa_mean, color="C0", lw=2.2,
            label=f"cqa R_det (ensemble mean, post peak {R_cqa_mean[n_pre:].max():.2f} m)")

    # per-instant band
    ax.fill_between(t_grid_full, band_p50, band_p95, color="C0", alpha=0.15,
                    label="cqa P50-P95 spread (per-instant)")
    ax.plot(t_grid_full, band_p95, color="C0", lw=1.0, ls="--", alpha=0.8,
            label=f"cqa P95 (post peak {band_p95[n_pre:].max():.2f} m)")

    # WCF event marker
    ax.axvline(0.0, color="0.2", lw=1.2, ls="-", alpha=0.7)
    ax.text(0.2, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.95,
            "WCFDI", color="0.2", fontsize=9, va="top")

    # IMCA reference lines
    ax.axhline(2.0, color="orange", lw=1.0, ls=":", alpha=0.7, label="IMCA amber 2 m")
    ax.axhline(4.0, color="red", lw=1.0, ls=":", alpha=0.7, label="IMCA red 4 m")

    ax.set_xlabel("t since WCFDI [s]")
    ax.set_ylabel(r"$R(t) = |\delta\eta_{LF}|$  [m]")
    ax.set_xlim(-t_pre_s, T_HORIZON_S)
    ax.set_ylim(0.0, None)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title(
        f"Radial deviation: brucon truth vs cqa live prediction  |  "
        f"{tag}  |  n={R_truth.shape[0]} seeds  |  "
        f"{demean_label}",
        fontsize=11,
    )
    plt.tight_layout()
    demean_tag = (f"demean{int(demean_win_s)}s" if demean_win_s > 0 else "raw")
    out = THIS / f"time_traces_vs_brucon_{tag}_{demean_tag}.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
