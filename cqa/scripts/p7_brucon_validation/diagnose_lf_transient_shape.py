"""Diagnose what the missing LF transient actually looks like.

Background
----------
After §12.21.8.1, the calibrated cqa MC under-predicts brucon LF P95
by ~28 % AND has the wrong ensemble-mean shape (cqa peaks ~18 s after
WCF and recovers by 60 s; brucon peaks 35-150 s after WCF with a
sustained offset).

This script avoids guessing at the mechanism. Instead it characterises
the gap directly so we can decide whether we are hunting an IC effect
or an assumption/forcing/model error. Two discriminators:

  (A) Per-seed truth Δη_LF(t) vs per-seed cqa-pred Δη_mean(t),
      ensemble-averaged. IC contributions are zero-mean Gaussian, so
      they vanish in the ensemble mean. Any residual mean bias is a
      MODEL/FORCING problem, not an IC problem.

  (B) Pre-WCF "no-event" trajectory: same baseline-subtract pipeline
      applied to the [T_WCF-150, T_WCF-30] window of the truth (no WCF
      event there). If this window shows similar slow wandering with
      magnitude comparable to the post-WCF "transient", the brucon LF
      channel has natural slow drift (LF residual variance over ~60 s)
      that the cqa pred (which decays cleanly to zero from x0=0) cannot
      represent. That would make some of the apparent gap structural,
      not transient.

For each seed, we compute:

  1. Truth body-frame LF Δ(SurgeDev, SwayDev, HeadingDev)(t),
     baseline-subtracted using the [WIN_START, WIN_END] = pre-WCF
     window mean.
  2. cqa pred Δη_mean(t) = pulse_response_with_lift_coupling(
         aug(cfg, Tp_obs_s=Tp from estimator),
         t_grid = [0, 120 s],
         tau_lost(t) from scenario(α=2/3, γ_imm=0.5, T_realloc=10 s),
         tau_env = +b̂(T_WCF - 5 s) from EstBias{Surge,Sway,Yaw} kN,
         x0 = zeros(N_STATE),
     )
     Note: x0=zeros means cqa starts from a zero state at t_eval, NOT
     from η̂_LF(t_eval) / ν̂(t_eval) / η_wave(t_eval) / etc. This is
     deliberate in the live cell (the prediction is a Δ overlaid on
     η̂_LF(t_eval)) but it means cqa CANNOT carry any IC-driven
     ballistic motion into the post-WCF window.
  3. Pre-WCF "no-event" baseline-subtracted Δ(SurgeDev, SwayDev, HeadingDev)
     over [T_WCF-150, T_WCF-30] -- a 120 s window with no WCF.

Outputs
-------
  diagnose_lf_transient_shape_<TAG>.png:
    - 3 rows × 3 cols.
    - Cols: surge, sway, yaw.
    - Row 1: per-seed truth Δη_LF(t) thin lines + ensemble mean thick.
    - Row 2: ensemble-mean overlay (truth vs cqa-pred), with the
      "no-event" pre-WCF ensemble-mean as a third reference (shifted
      to start at t=0_WCF on the same axis).
    - Row 3: per-seed peak |Δη| scatter, truth vs cqa-pred.

  Console table summarising:
    - ensemble-mean peak time and amplitude per DOF (truth vs pred)
    - no-event RMS amplitude (proxy for natural LF drift in 120 s)
    - per-seed peak |Δη| P50 / P95 truth vs pred
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))

from cqa.config import csov_default_config
from cqa.live_operator_view import _build_aug_for_live, N_STATE
from cqa.transient import WcfdiScenario
from cqa.transient_obs import (
    pulse_response,
    pulse_response_with_lift_coupling,
)


# ------------------------- knobs ------------------------------------------
T_WCF = 560.0
T_EVAL = T_WCF - 5.0           # cqa snapshot time
WIN_END = T_WCF - 1.0          # baseline window end
WIN_START = WIN_END - 60.0     # baseline window start (60 s)
T_NO_EVENT_END = T_WCF - 30.0
T_NO_EVENT_START = T_WCF - 150.0  # 120 s "no-event" baseline-window
T_PRED_END = 120.0
N_T = 241
T_PLOT_PRE = 10.0
T_PLOT_POST = 120.0

# Default cell
TAG_DEFAULT = "bf8_h0"
SEED_RANGE_DEFAULT = (1000, 1030)


# ------------------------- helpers ----------------------------------------
def _load_tsv(path: Path) -> dict[str, np.ndarray]:
    with open(path) as f:
        hdr = f.readline().strip().split("\t")
    data = np.loadtxt(path, skiprows=1, delimiter="\t")
    return {h: data[:, i] for i, h in enumerate(hdr)}


def load_seed(work_root: Path, tag: str, seed: int):
    """Return per-seed payload or None if log truncated / missing."""
    seed_dir = work_root / f"{tag}_seed{seed:04d}"
    if not seed_dir.exists():
        return None
    main_p = next(
        (p for p in seed_dir.glob("*.out") if "estimator" not in p.name),
        None,
    )
    est_p = seed_dir / f"{tag}_seed{seed:04d}_estimator.out"
    if main_p is None or not est_p.exists():
        return None

    M = _load_tsv(main_p)
    E = _load_tsv(est_p)
    t_main = M["t"]
    t_est = E["Time"]
    if t_main[-1] < T_WCF + T_PLOT_POST:
        return None

    win_m = (t_main >= WIN_START) & (t_main <= WIN_END)
    win_e = (t_est >= WIN_START) & (t_est <= WIN_END)

    # b̂ at t_eval (kN, kN, kNm) -> N, N, Nm
    i_eval_e = int(np.argmin(np.abs(t_est - T_EVAL)))
    b_hat = 1e3 * np.array([
        E["EstBiasSurge"][i_eval_e],
        E["EstBiasSway"][i_eval_e],
        E["EstBiasYaw"][i_eval_e],
    ])

    # Tp from estimator (mean over the window of EstWavePeriod{Surge,Sway})
    Tp_obs = float(np.mean(np.concatenate([
        E["EstWavePeriodSurge"][win_e],
        E["EstWavePeriodSway"][win_e],
    ])))
    if not np.isfinite(Tp_obs) or Tp_obs <= 1.0 or Tp_obs > 25.0:
        Tp_obs = 10.0

    # Truth LF Δη(t), baseline-subtracted to pre-WCF window mean.
    # SurgeDev/SwayDev: brucon DP estimator's LF body-frame deviation in m.
    # HeadingDev: DEGREES (per the unit-fix commit 6ecd251). Convert to rad.
    surge = M["SurgeDev"].copy()
    sway = M["SwayDev"].copy()
    yaw = np.deg2rad(M["HeadingDev"])
    surge_base = surge[win_m].mean()
    sway_base = sway[win_m].mean()
    yaw_base = yaw[win_m].mean()
    surge -= surge_base
    sway -= sway_base
    yaw -= yaw_base

    # Post-WCF trajectory on a common grid relative to t_eval.
    t_post = np.arange(-T_PLOT_PRE, T_PLOT_POST + 0.05, 0.1)
    truth_dx = np.interp(t_post, t_main - T_EVAL, surge)
    truth_dy = np.interp(t_post, t_main - T_EVAL, sway)
    truth_dyaw = np.interp(t_post, t_main - T_EVAL, yaw)

    # Pre-WCF "no-event" 120 s window, also baseline-subtracted to its
    # OWN window-mean (separate baseline so we measure the natural LF
    # drift inside that window, not the difference relative to the
    # later WCF baseline).
    win_ne = (t_main >= T_NO_EVENT_START) & (t_main <= T_NO_EVENT_END)
    surge_ne = M["SurgeDev"][win_ne].copy()
    sway_ne = M["SwayDev"][win_ne].copy()
    yaw_ne = np.deg2rad(M["HeadingDev"][win_ne]).copy()
    surge_ne -= surge_ne.mean()
    sway_ne -= sway_ne.mean()
    yaw_ne -= yaw_ne.mean()
    t_ne = t_main[win_ne] - T_NO_EVENT_START  # 0 .. 120 s
    # Resample to a common 0 .. 120 s grid at 0.1 s spacing for averaging.
    t_ne_grid = np.arange(0.0, 120.0 + 0.05, 0.1)
    ne_dx = np.interp(t_ne_grid, t_ne, surge_ne)
    ne_dy = np.interp(t_ne_grid, t_ne, sway_ne)
    ne_dyaw = np.interp(t_ne_grid, t_ne, yaw_ne)

    return dict(
        seed=seed,
        b_hat=b_hat,
        Tp_obs=Tp_obs,
        t_post=t_post,
        truth_dx=truth_dx,
        truth_dy=truth_dy,
        truth_dyaw=truth_dyaw,
        t_ne=t_ne_grid,
        ne_dx=ne_dx,
        ne_dy=ne_dy,
        ne_dyaw=ne_dyaw,
    )


def cqa_predict(cfg, b_hat: np.ndarray, Tp_obs: float, scenario: WcfdiScenario):
    """Return (t_grid, delta_eta_mean(t)) on a 0..T_PRED_END grid."""
    aug = _build_aug_for_live(cfg, Tp_obs_s=Tp_obs)
    t_grid = np.linspace(0.0, T_PRED_END, N_T)
    tau_env = b_hat
    gamma_imm = float(scenario.gamma_immediate)
    T_realloc = float(scenario.T_realloc) if scenario.T_realloc > 0 else 1e-9
    beta_t = 1.0 + (gamma_imm - 1.0) * np.exp(-t_grid / T_realloc)
    tau_lost = (beta_t[:, None] - 1.0) * (-tau_env[None, :])

    K_lift = float(getattr(cfg.vessel, "lift_coupling_K_per_rad", 0.0))
    if K_lift > 0.0:
        X = pulse_response_with_lift_coupling(
            aug, t_grid, tau_lost,
            b_hat0=tau_env, K_lift=K_lift,
            x0=np.zeros(N_STATE),
        )
    else:
        X = pulse_response(aug, t_grid, tau_lost, x0=np.zeros(N_STATE))
    return t_grid, X[:, 0:3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=TAG_DEFAULT)
    p.add_argument("--seeds", default=f"{SEED_RANGE_DEFAULT[0]}-{SEED_RANGE_DEFAULT[1]}")
    p.add_argument("--work-root", default=str(THIS / "work"))
    p.add_argument("--out", default=None,
                   help="Output PNG (default: diagnose_lf_transient_shape_<tag>.png)")
    args = p.parse_args()
    lo, hi = (int(x) for x in args.seeds.split("-"))
    work_root = Path(args.work_root)
    out_path = (
        Path(args.out) if args.out
        else THIS / f"diagnose_lf_transient_shape_{args.tag}.png"
    )

    cfg = csov_default_config()
    scenario = WcfdiScenario(
        alpha=(2.0 / 3.0,) * 3,
        gamma_immediate=0.5,
        T_realloc=10.0,
    )

    payloads = []
    skipped = []
    for seed in range(lo, hi):
        d = load_seed(work_root, args.tag, seed)
        if d is None:
            skipped.append(seed)
            continue
        t_grid, dEta = cqa_predict(cfg, d["b_hat"], d["Tp_obs"], scenario)
        d["t_pred"] = t_grid
        d["pred_dx"] = dEta[:, 0]
        d["pred_dy"] = dEta[:, 1]
        d["pred_dyaw"] = dEta[:, 2]
        payloads.append(d)

    if not payloads:
        print(f"No usable seeds for {args.tag} in {work_root}")
        return
    n_seed = len(payloads)
    print(f"{args.tag}: {n_seed} usable seeds, skipped={skipped}")

    # ----- aggregate ensemble-mean trajectories -----
    t_post = payloads[0]["t_post"]
    t_pred = payloads[0]["t_pred"]
    t_ne = payloads[0]["t_ne"]

    truth_dx = np.stack([d["truth_dx"] for d in payloads])  # (S, T_post)
    truth_dy = np.stack([d["truth_dy"] for d in payloads])
    truth_dyaw = np.stack([d["truth_dyaw"] for d in payloads])
    pred_dx = np.stack([d["pred_dx"] for d in payloads])    # (S, T_pred)
    pred_dy = np.stack([d["pred_dy"] for d in payloads])
    pred_dyaw = np.stack([d["pred_dyaw"] for d in payloads])
    ne_dx = np.stack([d["ne_dx"] for d in payloads])         # (S, T_ne)
    ne_dy = np.stack([d["ne_dy"] for d in payloads])
    ne_dyaw = np.stack([d["ne_dyaw"] for d in payloads])

    # Per-seed peak |Δη| (over the post-WCF window only, t >= 0).
    post_mask = t_post >= 0.0
    peak_truth = np.array([
        np.max(np.abs(truth_dx[:, post_mask]), axis=1),
        np.max(np.abs(truth_dy[:, post_mask]), axis=1),
        np.max(np.abs(truth_dyaw[:, post_mask]), axis=1),
    ])  # (3, S)
    peak_pred = np.array([
        np.max(np.abs(pred_dx), axis=1),
        np.max(np.abs(pred_dy), axis=1),
        np.max(np.abs(pred_dyaw), axis=1),
    ])  # (3, S)

    # No-event RMS per DOF per seed (a proxy for natural LF drift in 120 s)
    ne_rms = np.array([
        np.sqrt(np.mean(ne_dx**2, axis=1)),
        np.sqrt(np.mean(ne_dy**2, axis=1)),
        np.sqrt(np.mean(ne_dyaw**2, axis=1)),
    ])

    # ----- text summary -----
    dofs = ["surge [m]", "sway [m]", "yaw [rad]"]
    print()
    print(f"Cell {args.tag}, n_seed={n_seed}")
    print("Per-seed peak |Δη| over [t_WCF, t_WCF+120s]:")
    print(f"  {'DOF':<12}  {'truth P50':>10}  {'truth P95':>10}  "
          f"{'pred P50':>10}  {'pred P95':>10}  {'NE RMS':>10}")
    for i, lab in enumerate(dofs):
        tp50 = np.quantile(peak_truth[i], 0.50)
        tp95 = np.quantile(peak_truth[i], 0.95)
        pp50 = np.quantile(peak_pred[i], 0.50)
        pp95 = np.quantile(peak_pred[i], 0.95)
        ne = np.mean(ne_rms[i])
        print(f"  {lab:<12}  {tp50:>10.3f}  {tp95:>10.3f}  "
              f"{pp50:>10.3f}  {pp95:>10.3f}  {ne:>10.3f}")

    # Ensemble-mean peak time + amplitude
    print()
    print("Ensemble-mean peak (truth vs pred):")
    print(f"  {'DOF':<12}  {'truth t*':>10}  {'truth Δ*':>10}  "
          f"{'pred t*':>10}  {'pred Δ*':>10}")
    truth_means = [truth_dx.mean(0), truth_dy.mean(0), truth_dyaw.mean(0)]
    pred_means = [pred_dx.mean(0), pred_dy.mean(0), pred_dyaw.mean(0)]
    ne_means = [ne_dx.mean(0), ne_dy.mean(0), ne_dyaw.mean(0)]
    for i, lab in enumerate(dofs):
        tm = truth_means[i]
        pm = pred_means[i]
        # Argmax over the post-WCF window only
        i_t = int(np.argmax(np.abs(tm[post_mask])))
        t_t = t_post[post_mask][i_t]
        v_t = tm[post_mask][i_t]
        i_p = int(np.argmax(np.abs(pm)))
        t_p = t_pred[i_p]
        v_p = pm[i_p]
        print(f"  {lab:<12}  {t_t:>10.1f}  {v_t:>+10.4f}  "
              f"{t_p:>10.1f}  {v_p:>+10.4f}")

    # ----- plot -----
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    titles = ["surge Δ [m]", "sway Δ [m]", "yaw Δ [rad]"]

    # Row 1: per-seed truth + ensemble mean
    truth_arrs = [truth_dx, truth_dy, truth_dyaw]
    for j in range(3):
        ax = axes[0, j]
        for s in range(n_seed):
            ax.plot(t_post, truth_arrs[j][s], color="C0", alpha=0.18, lw=0.6)
        ax.plot(t_post, truth_arrs[j].mean(0), color="C0", lw=2.0,
                label=f"truth ens-mean (n={n_seed})")
        ax.axvline(0.0, color="0.5", lw=0.7, ls="--")
        ax.axhline(0.0, color="0.7", lw=0.5)
        ax.set_title(f"{titles[j]} -- truth per-seed")
        ax.set_xlim(-T_PLOT_PRE, T_PLOT_POST)
        ax.legend(loc="best", fontsize=8)
        if j == 0:
            ax.set_ylabel("amplitude")

    # Row 2: ensemble-mean truth vs pred + no-event mean
    pred_arrs = [pred_dx, pred_dy, pred_dyaw]
    ne_arrs = [ne_dx, ne_dy, ne_dyaw]
    for j in range(3):
        ax = axes[1, j]
        ax.plot(t_post, truth_arrs[j].mean(0), color="C0", lw=2.0,
                label="truth ens-mean")
        ax.plot(t_pred, pred_arrs[j].mean(0), color="C3", lw=2.0,
                label="cqa pred ens-mean")
        # no-event mean shifted to start at t=0 on this axis
        ax.plot(t_ne, ne_arrs[j].mean(0), color="0.5", lw=1.0, ls=":",
                label="pre-WCF no-event ens-mean")
        ax.axvline(0.0, color="0.5", lw=0.7, ls="--")
        ax.axhline(0.0, color="0.7", lw=0.5)
        ax.set_title(f"{titles[j]} -- ensemble means")
        ax.set_xlim(-T_PLOT_PRE, T_PLOT_POST)
        ax.legend(loc="best", fontsize=8)
        if j == 0:
            ax.set_ylabel("amplitude")

    # Row 3: per-seed peak truth vs pred scatter
    for j in range(3):
        ax = axes[2, j]
        ax.scatter(peak_truth[j], peak_pred[j], s=24, alpha=0.7)
        lim = max(peak_truth[j].max(), peak_pred[j].max()) * 1.1
        ax.plot([0, lim], [0, lim], "k--", lw=0.6, alpha=0.5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("truth peak |Δ|")
        ax.set_ylabel("pred peak |Δ|")
        ax.set_title(f"{titles[j]} -- per-seed peak scatter")

    fig.suptitle(
        f"LF transient diagnostic, cell={args.tag}, n_seed={n_seed}\n"
        f"(t=0 is t_WCF; baseline = pre-WCF [{int(WIN_START)},{int(WIN_END)}]s mean)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=110)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
