"""Long-settle (settle_s=3000) waves-only ensemble: re-validate sigma_y_LF.

Motivation
----------
The standard ensemble uses settle_s=500 with analysis window [300, 555] s.
The brucon estimators have time constants:
  - bias estimator: T_b = 1000 s  (so 3*T_b = 3000 s for asymptotic settling)
  - wave-period estimator: convergence trace shows it stabilises by t~400 s
  - sandbox closed-loop: dominant slow pole ~ 1/0.008 = 125 s, settled by ~500 s

With settle_s=500 the bias estimator has only 0.5*T_b of free DP after
station-keeping activation; possible that the residual ~0.50 m of unexplained
sigma_y_LF variance vs the sandbox is initialisation tail.

Plan
----
  - settle_s=3000  (intact-DP window [60, 3060] s)
  - analyse last 1500 s : window [1500, 3000] s
  - 30 seeds, ~50 s wall clock per seed -> ~25 min total on 12 workers
  - Compare brucon sigma_y_LF in the late window vs:
      (a) brucon sigma_y_LF in the original [300, 555] window (same seeds)
      (b) sandbox closed-loop predictions
  - If brucon late-window sigma drops -> residual is initialisation tail
  - If late-window sigma stays at ~0.65 m -> the gap is genuine steady-state physics

Outputs
-------
  - work-dir tree: scripts/p7_brucon_validation/work_long/pwo_long_seed*/
  - PNG: long_run_sigma_comparison.png  (gitignored)
  - Console summary table
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from harness import ScenarioSpec, run_ensemble  # noqa: E402

# ---- scenario (matches run_comparison_waves_only.py) ----
HS = 4.19571865443425
TP = 10.22443464601827
WAVE_DIR_COMPASS = 270.0
VESSEL_HEADING_COMPASS = 180.0
CSOV_WCF_BUS_PORT = (1, 4)  # (matches CSOV_WCF_GROUPS["bus_port"]; deactivated even though
                            #  we never reach failure_t in this script)

# ---- run length ----
SETTLE_S = 3000.0          # 6x default
POST_FAILURE_S = 60.0      # we don't analyse this; keep small
ACTIVATE_SK_S = 60.0
N_SEEDS = 30
TAG = "pwo_long"
WORK_DIR = THIS / "work_long"

# ---- analysis windows ----
WINDOW_LATE = (1500.0, 3000.0)   # post-settling steady-state
WINDOW_EARLY = (300.0, 555.0)    # original window for comparison


def get_brucon_sway_lf(result, t_start: float, t_end: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (sigma_y_LF, t_w, sway_LF_w) within [t_start, t_end].

    sway_LF is body-frame: project NED (x, y) through heading, subtract body-frame WF (xHf, yHf).
    Then mean-subtract within the window.
    """
    t = result.columns["t"]
    mask = (t >= t_start) & (t <= t_end)
    if not mask.any():
        return float("nan"), np.array([]), np.array([])
    x_ned = result.columns["x"][mask]
    y_ned = result.columns["y"][mask]
    psi_deg = result.columns["heading"][mask]  # heading in deg (brucon uses lowercase)
    x_hf = result.columns["xHf"][mask]
    y_hf = result.columns["yHf"][mask]
    psi = np.deg2rad(psi_deg)
    # Project NED -> body
    cos_p = np.cos(psi)
    sin_p = np.sin(psi)
    x_body = cos_p * x_ned + sin_p * y_ned
    y_body = -sin_p * x_ned + cos_p * y_ned
    # WF channels are already in body frame
    sway_LF = y_body - y_hf
    sway_LF -= sway_LF.mean()
    return float(sway_LF.std()), t[mask], sway_LF


def get_brucon_bias_stats(result_dir: Path, t_start: float, t_end: float) -> tuple[float, float]:
    """Return (mean, std) of EstBiasSway in window, in kN.

    Note: brucon's *_estimator.out reports forces in kN (same convention as the
    main .out file), not SI N.
    """
    est_file = next(result_dir.glob("*_estimator.out"))
    data = np.loadtxt(est_file, skiprows=1)
    t = data[:, 0]
    bias = data[:, 20]  # EstBiasSway [kN]
    mask = (t >= t_start) & (t <= t_end)
    return float(bias[mask].mean()), float(bias[mask].std())


def main() -> None:
    print(f"=" * 78)
    print(f"Long-run waves-only validation: settle_s={SETTLE_S:.0f}s, n_seeds={N_SEEDS}")
    print(f"=" * 78)
    print(f"  Hs={HS:.2f}m, Tp={TP:.2f}s, wave_from={WAVE_DIR_COMPASS:.0f}deg, heading={VESSEL_HEADING_COMPASS:.0f}deg")
    print(f"  Total sim time per seed: {ACTIVATE_SK_S + SETTLE_S + POST_FAILURE_S:.0f} s")
    print(f"  Analysis windows: late={WINDOW_LATE} s, early={WINDOW_EARLY} s")
    print()

    spec = ScenarioSpec(
        Hs=HS, Tp=TP, wave_dir_compass=WAVE_DIR_COMPASS,
        wind_speed=0.0, wind_dir_compass=WAVE_DIR_COMPASS,
        current_speed=0.0, current_dir_compass=WAVE_DIR_COMPASS,
        vessel_heading_compass=VESSEL_HEADING_COMPASS,
        failed_thruster_indices=CSOV_WCF_BUS_PORT,
        activate_sk_s=ACTIVATE_SK_S,
        settle_s=SETTLE_S,
        post_failure_s=POST_FAILURE_S,
        print_every_steps=1,
    )

    # --- run ensemble ---
    print(f"[sim] running {N_SEEDS}-seed ensemble (this will take ~25 min) ...")
    t0 = time.time()
    results = run_ensemble(spec, n_seeds=N_SEEDS, work_dir=WORK_DIR, tag=TAG)
    dt = time.time() - t0
    print(f"  ensemble done in {dt:.0f} s wall ({N_SEEDS * spec.total_seconds / dt:.0f}x realtime aggregate)")
    print()

    # --- compute sigma per seed in both windows ---
    sig_late = []
    sig_early = []
    for r in results:
        s_l, _, _ = get_brucon_sway_lf(r, *WINDOW_LATE)
        s_e, _, _ = get_brucon_sway_lf(r, *WINDOW_EARLY)
        sig_late.append(s_l)
        sig_early.append(s_e)
    sig_late = np.array(sig_late)
    sig_early = np.array(sig_early)

    # --- bias stats per seed in both windows ---
    bias_late = []
    bias_early = []
    for k in range(N_SEEDS):
        seed = 1000 + k
        rd = WORK_DIR / f"{TAG}_seed{seed}"
        m_l, s_l = get_brucon_bias_stats(rd, *WINDOW_LATE)
        m_e, s_e = get_brucon_bias_stats(rd, *WINDOW_EARLY)
        bias_late.append((m_l, s_l))
        bias_early.append((m_e, s_e))
    bias_late = np.array(bias_late)
    bias_early = np.array(bias_early)

    # --- summary ---
    print(f"\nResults across {N_SEEDS} seeds:")
    print(f"  brucon sigma_y_LF in EARLY window {WINDOW_EARLY} s:")
    print(f"     median={np.median(sig_early):.3f} m, mean={sig_early.mean():.3f} m, "
          f"range [{sig_early.min():.3f}, {sig_early.max():.3f}] m")
    print(f"  brucon sigma_y_LF in LATE  window {WINDOW_LATE} s:")
    print(f"     median={np.median(sig_late):.3f} m, mean={sig_late.mean():.3f} m, "
          f"range [{sig_late.min():.3f}, {sig_late.max():.3f}] m")
    print()
    print(f"  EstBiasSway in EARLY window:")
    print(f"     mean of seed-means = {bias_early[:, 0].mean():.1f} kN, "
          f"std-of-means across seeds = {bias_early[:, 0].std():.1f} kN")
    print(f"     mean of seed-stds  = {bias_early[:, 1].mean():.2f} kN")
    print(f"  EstBiasSway in LATE window:")
    print(f"     mean of seed-means = {bias_late[:, 0].mean():.1f} kN, "
          f"std-of-means across seeds = {bias_late[:, 0].std():.1f} kN")
    print(f"     mean of seed-stds  = {bias_late[:, 1].mean():.2f} kN")
    print()
    delta = np.median(sig_late) - np.median(sig_early)
    print(f"  Delta median sigma_y_LF (late - early) = {delta*100:+.1f} cm")
    if abs(delta) < 0.03:
        print(f"  --> sigma essentially unchanged: residual gap is genuine steady-state physics")
    elif delta < -0.05:
        print(f"  --> sigma drops significantly in late window: residual was initialisation tail")
    else:
        print(f"  --> sigma changed but not decisively; review per-seed")

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].scatter(np.arange(N_SEEDS), sig_early, label=f"early {WINDOW_EARLY[0]:.0f}-{WINDOW_EARLY[1]:.0f} s", alpha=0.7)
        axes[0].scatter(np.arange(N_SEEDS), sig_late, label=f"late  {WINDOW_LATE[0]:.0f}-{WINDOW_LATE[1]:.0f} s", alpha=0.7)
        axes[0].axhline(np.median(sig_early), color="C0", ls="--", alpha=0.5)
        axes[0].axhline(np.median(sig_late), color="C1", ls="--", alpha=0.5)
        axes[0].set_xlabel("seed index")
        axes[0].set_ylabel("sigma_y_LF [m]")
        axes[0].set_title(f"brucon sigma_y_LF, early vs late window")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(np.arange(N_SEEDS), bias_early[:, 0], label="early mean", alpha=0.7)
        axes[1].scatter(np.arange(N_SEEDS), bias_late[:, 0], label="late mean", alpha=0.7)
        axes[1].set_xlabel("seed index")
        axes[1].set_ylabel("EstBiasSway window-mean [kN]")
        axes[1].set_title("Bias estimator: settling check")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = THIS / "long_run_sigma_comparison.png"
        fig.savefig(out_png, dpi=110)
        print(f"\nSaved: {out_png}")
    except Exception as e:
        print(f"plot failed: {e}")


if __name__ == "__main__":
    main()
