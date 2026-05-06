"""Long-settle waves-only ensemble with brucon's wave-period estimator LOCKED.

Hypothesis (user, this round)
-----------------------------
The brucon-vs-sandbox sigma_y_LF gap is dominated by the f < 0.002 Hz band
(see analysis.md sect. 12.20.10 once written; per-seed std 0.67 m vs Welch-
integrated 0.40 m). Could the brucon wave-period estimator drifting Tp_est
over the simulation window induce a slow time-varying observer transfer that
shows up as low-frequency variance?

Experiment
----------
Same as long_run_validation.py (settle_s=3000, 30 seeds, intact sway-only)
but with brucon's wave-period estimator turned off. We append the block

  wave_filter_response_frequency {
    wave_filter_surge_peak_frequency: LOCKED
    wave_filter_sway_peak_frequency: LOCKED
    wave_filter_heading_peak_frequency: LOCKED
    wave_filter_rate_of_turn_peak_frequency: LOCKED
    locked_peak_period: 10.22
  }

to a per-seed shadow of observer.prototxt (real config_csov is untouched).
This pins all four observer wave-filter peak periods to 10.22 s -- the same
value the sandbox uses analytically.

Decision
--------
  - if median(sigma_y_LF) drops toward sandbox 0.40 m  -> Tp drift is the
    dominant mechanism behind the sub-mHz variance
  - if median stays at ~0.67 m                          -> Tp drift exonerated;
    the gap lives elsewhere (look at I-term saturation, allocator coupling,
    or unmodelled vessel nonlinearity)
  - intermediate                                        -> partial contribution

Outputs
-------
  - work tree: work_long_lockedTp/pwo_lockedTp_seedNNNN/
  - PNG: long_run_lockedTp_sigma_comparison.png  (gitignored)
  - Console: per-seed sigma_y_LF and observer Tp readback (verifies LOCKED)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from harness import (  # noqa: E402
    BRUCON_BIN,
    CSOV_CONFIG,
    ScenarioSpec,
    run_ensemble,
)

# ---- scenario (matches long_run_validation.py exactly) ----
HS = 4.19571865443425
TP = 10.22443464601827
WAVE_DIR_COMPASS = 270.0
VESSEL_HEADING_COMPASS = 180.0
CSOV_WCF_BUS_PORT = (1, 4)

# ---- run length ----
SETTLE_S = 3000.0
POST_FAILURE_S = 60.0
ACTIVATE_SK_S = 60.0
N_SEEDS = 30
TAG = "pwo_lockedTp"
WORK_DIR = THIS / "work_long_lockedTp"

# ---- analysis windows (match the unlocked baseline) ----
WINDOW_LATE = (1500.0, 3000.0)
WINDOW_EARLY = (300.0, 555.0)

LOCKED_TP_S = 10.22       # match sandbox-fixed Tp (== TP rounded to file precision)


def build_observer_override() -> str:
    """Read CSOV observer.prototxt and append the LOCKED-Tp block."""
    src = (CSOV_CONFIG / "observer.prototxt").read_text()
    appended = src.rstrip() + "\n\n" + (
        "wave_filter_response_frequency {\n"
        "  wave_filter_surge_peak_frequency: LOCKED\n"
        "  wave_filter_sway_peak_frequency: LOCKED\n"
        "  wave_filter_heading_peak_frequency: LOCKED\n"
        "  wave_filter_rate_of_turn_peak_frequency: LOCKED\n"
        f"  locked_peak_period: {LOCKED_TP_S}\n"
        "}\n"
    )
    return appended


def get_brucon_sway_lf(result, t_start: float, t_end: float):
    t = result.columns["t"]
    mask = (t >= t_start) & (t <= t_end)
    if not mask.any():
        return float("nan")
    x_ned = result.columns["x"][mask]
    y_ned = result.columns["y"][mask]
    psi = np.deg2rad(result.columns["heading"][mask])
    cos_p, sin_p = np.cos(psi), np.sin(psi)
    y_body = -sin_p * x_ned + cos_p * y_ned
    y_hf = result.columns["yHf"][mask]
    sway_lf = y_body - y_hf
    sway_lf -= sway_lf.mean()
    return float(sway_lf.std())


def read_estimator_tp(result_dir: Path, t_start: float, t_end: float):
    """Return (mean, std) of EstWavePeriodPitch in window. Sanity-check that
    the locked-Tp setting actually freezes the wave-filter period."""
    est_file = next(result_dir.glob("*_estimator.out"))
    data = np.loadtxt(est_file, skiprows=1)
    t = data[:, 0]
    tp_pitch = data[:, 30] if data.shape[1] > 30 else None  # EstWavePeriodPitch index per dp_runfast_simulator.cpp:1126
    if tp_pitch is None:
        return float("nan"), float("nan")
    mask = (t >= t_start) & (t <= t_end)
    return float(tp_pitch[mask].mean()), float(tp_pitch[mask].std())


def main() -> None:
    print("=" * 78)
    print(f"Long-run waves-only validation with LOCKED Tp = {LOCKED_TP_S} s")
    print(f"  settle_s={SETTLE_S:.0f}s, n_seeds={N_SEEDS}")
    print("=" * 78)

    override_text = build_observer_override()
    # Sanity: print the appended block so the user can eyeball it
    print("\n--- appended observer.prototxt block (per-seed shadow) ---")
    print(override_text.split("\n\n", 1)[-1])
    print("--- (rest of file is unchanged from CSOV_CONFIG/observer.prototxt) ---\n")

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
        config_overrides={"observer.prototxt": override_text},
    )

    print(f"[sim] running {N_SEEDS}-seed ensemble (~25 min) ...")
    t0 = time.time()
    results = run_ensemble(spec, n_seeds=N_SEEDS, work_dir=WORK_DIR, tag=TAG)
    dt = time.time() - t0
    print(f"  done in {dt:.0f} s wall ({N_SEEDS * spec.total_seconds / dt:.0f}x realtime aggregate)")
    print()

    # --- sigma_y_LF per seed in both windows ---
    sig_late = np.array([get_brucon_sway_lf(r, *WINDOW_LATE) for r in results])
    sig_early = np.array([get_brucon_sway_lf(r, *WINDOW_EARLY) for r in results])

    # --- Tp readback per seed in late window ---
    tp_means = []
    tp_stds = []
    for k in range(N_SEEDS):
        seed = 1000 + k
        # run_simulation uses the literal seed (e.g. 1000), and the dir is
        # f"{tag}_seed{seed:04d}"; override matches.
        rd = WORK_DIR / f"{TAG}_seed{seed:04d}"
        m, s = read_estimator_tp(rd, *WINDOW_LATE)
        tp_means.append(m)
        tp_stds.append(s)
    tp_means = np.array(tp_means)
    tp_stds = np.array(tp_stds)

    print(f"--- locked-Tp readback (EstWavePeriodPitch in window {WINDOW_LATE}) ---")
    print(f"  mean across seeds: {tp_means.mean():.3f} s  (std-of-means {tp_means.std():.4f})")
    print(f"  mean per-seed std: {tp_stds.mean():.4f} s")
    print(f"  if locking worked, the wave-filter period in the observer is pinned to "
          f"{LOCKED_TP_S} regardless of the pitch estimator value above.")
    print()

    print(f"--- sigma_y_LF results across {N_SEEDS} seeds (LOCKED Tp = {LOCKED_TP_S} s) ---")
    print(f"  EARLY {WINDOW_EARLY}: median={np.median(sig_early):.3f} m, "
          f"mean={sig_early.mean():.3f} m, range [{sig_early.min():.3f}, {sig_early.max():.3f}] m")
    print(f"  LATE  {WINDOW_LATE}: median={np.median(sig_late):.3f} m, "
          f"mean={sig_late.mean():.3f} m, range [{sig_late.min():.3f}, {sig_late.max():.3f}] m")
    print()

    print("--- comparison vs unlocked baseline ---")
    print(f"  baseline (unlocked Tp) late-window median sigma_y_LF: ~0.67 m")
    print(f"  sandbox prediction (analytic, fixed Tp=10.22): ~0.29 m")
    delta = np.median(sig_late) - 0.67
    print(f"  delta vs unlocked baseline = {delta*100:+.1f} cm")
    if abs(delta) < 0.04:
        verdict = "Tp drift exonerated -- residual gap is elsewhere"
    elif delta < -0.10:
        verdict = "Tp drift IS a major contributor -- locking it closes the gap meaningfully"
    elif delta < -0.04:
        verdict = "Tp drift contributes partially; not the sole mechanism"
    else:
        verdict = "Locking Tp made it worse?? -- check shadow config and Tp readback"
    print(f"  verdict: {verdict}")

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax = axes[0]
        idx = np.arange(N_SEEDS)
        ax.scatter(idx, sig_early, label=f"locked-Tp early {WINDOW_EARLY[0]:.0f}-{WINDOW_EARLY[1]:.0f} s", alpha=0.7)
        ax.scatter(idx, sig_late, label=f"locked-Tp late  {WINDOW_LATE[0]:.0f}-{WINDOW_LATE[1]:.0f} s", alpha=0.7)
        ax.axhline(np.median(sig_late), color="C1", ls="--", alpha=0.5,
                   label=f"locked-Tp late median = {np.median(sig_late):.3f} m")
        ax.axhline(0.67, color="k", ls=":", alpha=0.5,
                   label="unlocked baseline (median, 0.67 m)")
        ax.axhline(0.40, color="g", ls=":", alpha=0.5,
                   label="sandbox prediction (Welch-integrated, 0.40 m)")
        ax.set_xlabel("seed index"); ax.set_ylabel("sigma_y_LF [m]")
        ax.set_title(f"brucon sigma_y_LF with LOCKED Tp = {LOCKED_TP_S} s")
        ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)

        ax = axes[1]
        ax.scatter(idx, tp_means, label=f"EstWavePeriodPitch mean (per seed)")
        ax.axhline(LOCKED_TP_S, color="r", ls="--",
                   label=f"locked Tp used by wave filter = {LOCKED_TP_S} s")
        ax.set_xlabel("seed index"); ax.set_ylabel("Tp [s]")
        ax.set_title("Pitch-derived Tp estimator (locked => not used by observer)")
        ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        out_png = THIS / "long_run_lockedTp_sigma_comparison.png"
        fig.savefig(out_png, dpi=110)
        print(f"\nSaved: {out_png}")
    except Exception as e:
        print(f"plot failed: {e}")


if __name__ == "__main__":
    main()
