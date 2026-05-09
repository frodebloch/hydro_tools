"""Test the 'slack-spring surge coupling' hypothesis at pwo (beam-on).

Hypothesis
----------
At pwo (waves from +90 deg relative bow), the steady wave-drift force is
predominantly in *sway*. Surge drift is small, so the controller's surge
state sits near the slack part of its integrator (no setpoint bias).

Post-WCF, the bus_port WCFDI removes asymmetric thrust, generating yaw.
Even a few degrees of yaw rotates the wave-drift vector in body frame,
producing a *new* surge component. This surge axis then "discovers"
substantial mean force it did not have intact — but is poorly damped
because the linearised model around intact equilibrium does not see this
coupling.

The cqa live-cell pipeline holds b_hat (= -tau_env) constant in body
frame during the post-WCF transient. It cannot capture the mid-transient
growth of body-frame surge force as heading rotates.

Test
----
For each pwo seed:
  1. Extract pre-WCF mean (over last 60 s of intact window):
       SurgeDev_pre, SwayDev_pre, Heading_pre.
  2. Extract post-WCF time series over [t_WCF, t_WCF + 90 s]:
       SurgeDev(t), SwayDev(t), Heading(t).
  3. Compute body-frame deviation R_x(t) = SurgeDev(t) - SurgeDev_pre,
     R_y(t) = SwayDev(t) - SwayDev_pre.
  4. Find peak |R| in the window, and decompose into peak |R_x| and
     |R_y| separately.
  5. Heading drift: max |Heading(t) - Heading_pre| in the window.

Then compare to the live cell's per-axis prediction. The live cell
output gives a single |R| envelope; we extract the per-axis prediction
from pulse_response(b_hat) directly.

Pass criterion for the hypothesis:
  - Truth |R_x| (surge) growth is comparable to or larger than what
    pulse_response(b_hat) predicts on the surge axis, especially at
    late times when heading has rotated.
  - Heading drift in pwo is non-negligible (>few deg).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THIS = Path(__file__).resolve().parent
CQA_ROOT = THIS.parent.parent
sys.path.insert(0, str(CQA_ROOT))
sys.path.insert(0, str(THIS))

TAG = "pwo"
WORK_DIR = THIS / "work"
N_SEEDS = 30
BASE_SEED = 1000
T_WCF = 560.0  # matches pwq30 / pwo lua
PRE_WINDOW_S = 60.0
POST_WINDOW_S = 90.0


def _load_seed(seed: int):
    out_path = WORK_DIR / f"{TAG}_seed{seed}" / f"{TAG}_seed{seed}.out"
    # Header columns; load all and slice.
    with open(out_path) as f:
        header = f.readline().strip().split("\t")
    data = np.loadtxt(out_path, skiprows=1, delimiter="\t")
    cols = {name: i for i, name in enumerate(header)}
    return data, cols


def _extract_seed(seed: int):
    data, cols = _load_seed(seed)
    t = data[:, cols["t"]]
    surge = data[:, cols["SurgeDev"]]
    sway = data[:, cols["SwayDev"]]
    heading = data[:, cols["HeadingDev"]]  # deviation from setpoint, deg, no wrap
    # Pre-WCF window: [T_WCF - PRE_WINDOW_S, T_WCF].
    pre_mask = (t >= T_WCF - PRE_WINDOW_S) & (t < T_WCF)
    surge_pre = float(np.mean(surge[pre_mask]))
    sway_pre = float(np.mean(sway[pre_mask]))
    heading_pre = float(np.mean(heading[pre_mask]))
    # Post-WCF window: [T_WCF, T_WCF + POST_WINDOW_S].
    post_mask = (t >= T_WCF) & (t <= T_WCF + POST_WINDOW_S)
    t_post = t[post_mask] - T_WCF
    Rx = surge[post_mask] - surge_pre
    Ry = sway[post_mask] - sway_pre
    dpsi = heading[post_mask] - heading_pre
    return dict(t_post=t_post, Rx=Rx, Ry=Ry, dpsi=dpsi,
                surge_pre=surge_pre, sway_pre=sway_pre,
                heading_pre=heading_pre)


def main():
    print("=" * 72)
    print(f"pwo surge-coupling diagnostic ({N_SEEDS} seeds)")
    print("=" * 72)

    seeds = list(range(BASE_SEED, BASE_SEED + N_SEEDS))
    per_seed = [_extract_seed(s) for s in seeds]

    # Heading: brucon writes 'HeadingDev' in degrees (deviation from setpoint,
    # no wrap-around).
    heading_pre_arr = np.array([s["heading_pre"] for s in per_seed])
    print(f"\nPre-WCF HeadingDev: mean = {heading_pre_arr.mean():.4f} deg, "
          f"std = {heading_pre_arr.std():.4f} deg")
    print(f"  (range [{heading_pre_arr.min():.4f}, {heading_pre_arr.max():.4f}])")
    rad2deg = 1.0  # already deg

    # Per-seed peak |R_x|, |R_y|, |R| total, and peak |dpsi|.
    peak_Rx = np.array([np.max(np.abs(s["Rx"])) for s in per_seed])
    peak_Ry = np.array([np.max(np.abs(s["Ry"])) for s in per_seed])
    peak_R = np.array([np.max(np.sqrt(s["Rx"] ** 2 + s["Ry"] ** 2))
                        for s in per_seed])
    peak_dpsi_deg = np.array([np.max(np.abs(s["dpsi"])) * rad2deg
                              for s in per_seed])

    print(f"\n=== Per-seed peak deviations in [t_WCF, t_WCF + {POST_WINDOW_S:.0f}s] ===")
    print(f"  |R_x| (surge):  mean={peak_Rx.mean():.3f}  P5={np.percentile(peak_Rx, 5):.2f}"
          f"  P50={np.percentile(peak_Rx, 50):.2f}"
          f"  P95={np.percentile(peak_Rx, 95):.2f}")
    print(f"  |R_y| (sway):   mean={peak_Ry.mean():.3f}  P5={np.percentile(peak_Ry, 5):.2f}"
          f"  P50={np.percentile(peak_Ry, 50):.2f}"
          f"  P95={np.percentile(peak_Ry, 95):.2f}")
    print(f"  |R|  (total):   mean={peak_R.mean():.3f}   P5={np.percentile(peak_R, 5):.2f}"
          f"  P50={np.percentile(peak_R, 50):.2f}"
          f"  P95={np.percentile(peak_R, 95):.2f}")
    print(f"  |dpsi| (deg):   mean={peak_dpsi_deg.mean():.3f}  "
          f"P50={np.percentile(peak_dpsi_deg, 50):.2f}  "
          f"P95={np.percentile(peak_dpsi_deg, 95):.2f}")

    # Surge fraction of total radial deviation: peak |R_x| / peak |R|.
    surge_frac = peak_Rx / peak_R
    sway_frac = peak_Ry / peak_R
    print(f"\n  peak|R_x|/peak|R| (surge share):   "
          f"mean={surge_frac.mean():.3f}  P50={np.median(surge_frac):.3f}")
    print(f"  peak|R_y|/peak|R| (sway share):    "
          f"mean={sway_frac.mean():.3f}  P50={np.median(sway_frac):.3f}")

    # Compare to the live-cell axis prediction. The live cell predicts |R|
    # by integrating pulse_response on b_hat (which we measured pre-WCF).
    # b_hat for beam-on should be ~all in body sway (waves push from port
    # = +90 deg compass relative to a 180-heading vessel = body +y).
    # Load the calibration npz which holds the precomputed delta_eta_mean
    # (ensemble mean), already in body frame.
    npz_path = THIS / f"scenario_{TAG}_calibration.npz"
    cal = np.load(npz_path)
    delta_eta_mean = cal["delta_eta_mean"]  # (T, 6) typically
    t_pred = cal["t_post_s"] if "t_post_s" in cal.files else None
    if t_pred is None:
        # Try common alternate names.
        for key in ("t_post", "t", "time_s"):
            if key in cal.files:
                t_pred = cal[key]
                break
    print(f"\n  npz keys: {list(cal.files)}")
    print(f"  delta_eta_mean shape: {delta_eta_mean.shape}")
    if t_pred is not None:
        print(f"  t_pred range: [{t_pred[0]:.1f}, {t_pred[-1]:.1f}] s")

    # delta_eta_mean rows are 6-DOF body deviations; col 0 = surge, col 1 = sway.
    if delta_eta_mean.ndim == 2 and delta_eta_mean.shape[1] >= 2:
        pred_Rx = delta_eta_mean[:, 0]
        pred_Ry = delta_eta_mean[:, 1]
        pred_R = np.sqrt(pred_Rx ** 2 + pred_Ry ** 2)
        print(f"\n=== Live-cell pulse_response(b_hat) prediction (axis decomposition) ===")
        print(f"  peak |R_x| (surge):  {np.max(np.abs(pred_Rx)):.3f} m")
        print(f"  peak |R_y| (sway):   {np.max(np.abs(pred_Ry)):.3f} m")
        print(f"  peak |R|  (total):   {np.max(pred_R):.3f} m")

        # Surge share of prediction.
        if np.max(pred_R) > 1e-6:
            i_peak = int(np.argmax(pred_R))
            pred_surge_share = abs(pred_Rx[i_peak]) / np.max(pred_R)
            pred_sway_share = abs(pred_Ry[i_peak]) / np.max(pred_R)
            print(f"  surge share at peak: {pred_surge_share:.3f}")
            print(f"  sway share at peak:  {pred_sway_share:.3f}")

    # ------------------------------------------------------------------
    # Plot: heading drift overlay + ensemble-mean R_x, R_y vs time.
    # ------------------------------------------------------------------
    t_grid = per_seed[0]["t_post"]
    Rx_stack = np.array([np.interp(t_grid, s["t_post"], s["Rx"]) for s in per_seed])
    Ry_stack = np.array([np.interp(t_grid, s["t_post"], s["Ry"]) for s in per_seed])
    dpsi_stack_deg = np.array([np.interp(t_grid, s["t_post"], s["dpsi"]) * rad2deg
                                for s in per_seed])

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    for i in range(N_SEEDS):
        ax.plot(t_grid, dpsi_stack_deg[i], color="0.7", lw=0.6, alpha=0.6)
    ax.plot(t_grid, dpsi_stack_deg.mean(axis=0), "k-", lw=2,
            label="ensemble mean")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Heading deviation (deg)")
    ax.set_title(f"pwo post-WCF coupling diagnostic ({N_SEEDS} seeds)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i in range(N_SEEDS):
        ax.plot(t_grid, Rx_stack[i], color="C0", lw=0.6, alpha=0.4)
    ax.plot(t_grid, Rx_stack.mean(axis=0), "C0-", lw=2,
            label="brucon ensemble mean R_x (surge)")
    if delta_eta_mean.ndim == 2 and t_pred is not None:
        ax.plot(t_pred, delta_eta_mean[:, 0], "C3--", lw=2,
                label="live-cell pulse_response (surge)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("R_x = SurgeDev - pre (m)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for i in range(N_SEEDS):
        ax.plot(t_grid, Ry_stack[i], color="C1", lw=0.6, alpha=0.4)
    ax.plot(t_grid, Ry_stack.mean(axis=0), "C1-", lw=2,
            label="brucon ensemble mean R_y (sway)")
    if delta_eta_mean.ndim == 2 and t_pred is not None:
        ax.plot(t_pred, delta_eta_mean[:, 1], "C3--", lw=2,
                label="live-cell pulse_response (sway)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("R_y = SwayDev - pre (m)")
    ax.set_xlabel("t since WCF (s)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png = THIS / f"diagnose_{TAG}_surge_coupling.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
