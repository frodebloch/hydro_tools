"""Re-load the existing P7 ensemble outputs and produce an *intact-window*
time-series figure analogous to the post-failure transient figure.

Motivation
----------
The transient panel shows post-failure traces that are baseline-subtracted
per-seed at t = failure, so they look "calm". But the intact-window CDF
shows P50 running-max |pos| ~ 12 m. That can only be true if the intact
window itself is not actually steady -- either there is a slow drift, a
large mean offset, or low-frequency oscillation. This script visualises
what the intact window actually looks like, per seed, in body-frame
surge/sway, using the simulator's own SurgeDev/SwayDev columns (no manual
NED->body rotation -- the simulator already exports body-frame deviation
from the held setpoint).

No new simulator runs are performed. Reads `work/p7v1_seed*/p7v1_seedNNNN.out`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import parse_output


# --- Mirror canonical scenario windows (must match run_comparison.py) ---
ACTIVATE_SK_S = 60.0
SETTLE_S = 500.0
POST_FAILURE_S = 180.0
FAILURE_TIME_S = ACTIVATE_SK_S + SETTLE_S
TOTAL_S = FAILURE_TIME_S + POST_FAILURE_S
SIM_DT = 0.1
INTACT_SAMPLE_S = 200.0   # last X seconds of intact phase used for stats

VW = 14.0
HS = 4.196
TP = 10.224
VESSEL_HEADING_COMPASS = 180.0


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    work_dir = out_dir / "work"

    seed_dirs = sorted(p for p in work_dir.iterdir() if p.is_dir() and p.name.startswith("p7v1_seed"))
    if not seed_dirs:
        raise SystemExit(f"No p7v1_seed* dirs under {work_dir}")

    print(f"Loading {len(seed_dirs)} seeds ...")
    results = []
    for d in seed_dirs:
        out_path = d / f"{d.name}.out"
        results.append(parse_output(out_path))

    n_min = min(r.n_rows for r in results)
    t = results[0]["t"][:n_min]
    surge_dev = np.array([r["SurgeDev"][:n_min] for r in results])  # (N, T) body-frame
    sway_dev = np.array([r["SwayDev"][:n_min] for r in results])
    pos_dev = np.array([r["PosDev"][:n_min] for r in results])      # |body-frame deviation|

    # Sanity check: NED magnitude should equal PosDev in beam-on hold (within tiny num error)
    x_ned = np.array([r["x"][:n_min] for r in results])
    y_ned = np.array([r["y"][:n_min] for r in results])
    pos_ned = np.hypot(x_ned, y_ned)
    diff_max = np.max(np.abs(pos_dev - pos_ned))
    print(f"  max |PosDev - hypot(x,y)| over ensemble = {diff_max:.4f} m  "
          f"(expect ~0 since setpoint is at NED origin)")

    # Window masks (matching run_comparison.py: only the late portion of
    # the intact phase, after settling has decayed).
    intact_mask = (
        (t >= FAILURE_TIME_S - INTACT_SAMPLE_S)
        & (t < FAILURE_TIME_S - 1.0)
    )
    precond_mask = t < ACTIVATE_SK_S
    post_mask = t >= FAILURE_TIME_S

    n_seeds = len(results)
    print(f"  precond window     : {precond_mask.sum() * SIM_DT:5.0f} s")
    print(f"  intact window      : {intact_mask.sum() * SIM_DT:5.0f} s")
    print(f"  post-failure window: {post_mask.sum() * SIM_DT:5.0f} s")

    # --- Per-seed running-max |pos| over intact window ---
    intact_pos = pos_dev[:, intact_mask]
    intact_running_max = np.maximum.accumulate(intact_pos, axis=1)
    intact_max_per_seed = intact_running_max[:, -1]

    print(f"\nIntact running-max |pos|:")
    print(f"  median  = {np.median(intact_max_per_seed):.2f} m")
    print(f"  P90     = {np.quantile(intact_max_per_seed, 0.90):.2f} m")
    print(f"  min/max = {intact_max_per_seed.min():.2f} / {intact_max_per_seed.max():.2f} m")

    # Mean of |pos| over intact window per seed (sanity-check vs cqa sigma scale)
    intact_mean_per_seed = intact_pos.mean(axis=1)
    intact_std_per_seed_surge = surge_dev[:, intact_mask].std(axis=1)
    intact_std_per_seed_sway = sway_dev[:, intact_mask].std(axis=1)
    print(f"\nIntact mean |pos| per seed: median = {np.median(intact_mean_per_seed):.2f} m")
    print(f"Intact std(surge) per seed : median = {np.median(intact_std_per_seed_surge):.2f} m")
    print(f"Intact std(sway)  per seed : median = {np.median(intact_std_per_seed_sway):.2f} m")

    # Ensemble means for clean overlay
    surge_mean = surge_dev.mean(axis=0)
    sway_mean = sway_dev.mean(axis=0)
    surge_q_lo = np.quantile(surge_dev, 0.25, axis=0)
    surge_q_hi = np.quantile(surge_dev, 0.75, axis=0)
    sway_q_lo = np.quantile(sway_dev, 0.25, axis=0)
    sway_q_hi = np.quantile(sway_dev, 0.75, axis=0)

    # ----------------------------------------------------------------
    # Figure 1: full timeline, body-frame surge + sway, all seeds
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, comp_name, traces, ens_mean, q_lo, q_hi in [
        (axes[0], "surge", surge_dev, surge_mean, surge_q_lo, surge_q_hi),
        (axes[1], "sway",  sway_dev,  sway_mean,  sway_q_lo,  sway_q_hi),
    ]:
        for k in range(n_seeds):
            ax.plot(t, traces[k], color="gray", alpha=0.20, lw=0.6)
        ax.plot(t, ens_mean, color="tab:blue", lw=1.6, label="ensemble mean")
        ax.fill_between(t, q_lo, q_hi, color="tab:blue", alpha=0.18, label="IQR (25-75%)")
        ax.axvspan(0.0, ACTIVATE_SK_S, color="orange", alpha=0.10, label="precond (SetFixedCourseAndSpeed)")
        ax.axvspan(ACTIVATE_SK_S, FAILURE_TIME_S, color="green", alpha=0.06, label="intact DP")
        ax.axvspan(FAILURE_TIME_S, TOTAL_S, color="red", alpha=0.06, label="post-WCFDI")
        ax.axvline(FAILURE_TIME_S, color="black", ls="--", alpha=0.7, lw=1.0)
        ax.set_ylabel(f"{comp_name} deviation [m]\n(simulator SurgeDev/SwayDev)")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper left", fontsize=8, ncol=2)

    axes[1].set_xlabel("simulator time [s]")
    fig.suptitle(
        f"P7 simulator -- full-timeline body-frame deviation (N={n_seeds} seeds)\n"
        f"Vw={VW:.1f} m/s, Hs={HS:.2f} m, Tp={TP:.2f} s, beam-on (theta_rel=+90 deg), Bus port lost at t={FAILURE_TIME_S:.0f} s",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out1 = out_dir / "p7_validation_full_timeline.png"
    fig.savefig(out1, dpi=130)
    plt.close(fig)
    print(f"\n  wrote {out1}")

    # ----------------------------------------------------------------
    # Figure 2: zoom on intact window only -- body-frame surge + sway + |pos|
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    t_intact = t[intact_mask]
    surge_int = surge_dev[:, intact_mask]
    sway_int = sway_dev[:, intact_mask]
    pos_int = pos_dev[:, intact_mask]

    for ax, comp_name, traces, unit in [
        (axes[0], "surge dev",  surge_int, "m"),
        (axes[1], "sway dev",   sway_int,  "m"),
        (axes[2], "|pos dev|",  pos_int,   "m"),
    ]:
        for k in range(n_seeds):
            ax.plot(t_intact, traces[k], color="gray", alpha=0.25, lw=0.6)
        ax.plot(t_intact, traces.mean(axis=0), color="tab:blue", lw=1.8, label="ensemble mean")
        ax.fill_between(t_intact,
                        np.quantile(traces, 0.25, axis=0),
                        np.quantile(traces, 0.75, axis=0),
                        color="tab:blue", alpha=0.18, label="IQR (25-75%)")
        ax.set_ylabel(f"{comp_name} [{unit}]")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper right", fontsize=9)

    axes[2].set_xlabel("simulator time [s]  (intact-DP window)")
    fig.suptitle(
        f"P7 simulator -- intact-DP window zoom (N={n_seeds} seeds, {intact_mask.sum()*SIM_DT:.0f} s)\n"
        f"Vw={VW:.1f} m/s, Hs={HS:.2f} m, Tp={TP:.2f} s, beam-on, "
        f"sim P50 running-max |pos| = {np.median(intact_max_per_seed):.2f} m, "
        f"P90 = {np.quantile(intact_max_per_seed, 0.90):.2f} m",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out2 = out_dir / "p7_validation_intact_timeseries.png"
    fig.savefig(out2, dpi=130)
    plt.close(fig)
    print(f"  wrote {out2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
