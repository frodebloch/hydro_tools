"""Brucon pwq30 ensemble re-run with use_tau_feedback = TRUE in the observer.

DIAGNOSTIC ONLY — not an operationally-recommended setting.

Hypothesis (this round)
-----------------------
The brucon truth ensemble-mean Δsway peaks at -0.50 m at t=35 s post-WCF.
The cqa analytic linearised pulse-response (using the exact same
tau_lost(t) shape as the cqa MC, and that same shape matches the brucon
ensemble (Order - Delivered) deficit to within ~10 % per seed) peaks at
-0.31 m at t=15 s. So the gap (60 % larger, 20 s later) cannot be in the
disturbance shape: it must be in the loop dynamics.

The most plausible mechanism: the CSOV observer is configured with
``use_tau_feedback = false``, meaning the observer's plant model is driven
by the controller's *commanded* (allocated) thrust, not the actually-
delivered thrust. During the post-WCF spool-up window (~10 s), the
controller commands +144 kN sway that the surviving thrusters cannot
deliver yet. The observer's internal plant model integrates this
non-existent force, so its η_hat and ν_hat slew toward "the vessel is
recovering" while the truth is still drifting. The controller closes its
feedback on η_hat / ν_hat and therefore *under-corrects*, leaving the
real vessel to drift further before the observer realises and the
controller catches up.

(The CSOV default is use_tau_feedback=false for failure-handling robustness:
the feedback signal could fail silently in a thruster-sensor fault and
corrupt the observer in exactly the wrong condition. Using the order is
trivially trustworthy. So this experiment is to *quantify* the cost of
that design choice in the post-WCF transient regime, not to recommend
flipping it.)

Experiment
----------
Re-run the pwq30 30-seed ensemble (waves-only, bow-quartering 30°,
bus_port WCF) with a per-seed shadow of observer.prototxt that flips
``use_tau_feedback: false`` -> ``use_tau_feedback: true``. Compare the
ensemble-mean Δsurge / Δsway and the per-seed |Δradial|_peak in [0, 60] s
post-WCF against (a) the existing pwq30 ensemble (use_tau_feedback=false)
and (b) the cqa analytic linearised pulse response.

Decision
--------
  - if peak shrinks toward cqa analytic (-0.31 m) and shifts earlier
    (toward t=15 s) -> observer-input mismatch is the dominant gap
    mechanism. cqa needs to model an "observer driven by tau_cmd" form
    for the linearised dynamics to match brucon at the operationally-
    used setting.
  - if peak essentially unchanged -> mechanism is elsewhere (posref
    filter, tau_cmd-driven bias estimator, allocator dynamics).
  - intermediate -> partial contribution.

Outputs
-------
  - work tree: work_pwq30_taufb/pwq30_taufb_seedNNNN/
  - PNG: pwq30_taufb_overlay.png
  - npz: /tmp/pwq30_taufb_status.npz with truth_dsurge_mean,
    truth_dsway_mean, t_truth, peaks_60 etc., for the next analysis pass.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from harness import (  # noqa: E402
    CSOV_CONFIG,
    CSOV_WCF_GROUPS,
    ScenarioSpec,
    SIM_DT,
    parse_output,
    run_ensemble,
)


# ---- scenario (matches run_comparison_waves_only_quartering30.py / pwq30) ----
HS = 4.19571865443425
TP = 10.22443464601827
WAVE_DIR_COMPASS = 210.0
VESSEL_HEADING_COMPASS = 180.0
N_SEEDS = 30
T_WCF_S = 560.0
T_POST_S = 180.0
SETTLE_S = 500.0
ACTIVATE_SK_S = 60.0

TAG = "pwq30_taufb"
WORK_DIR = THIS / "work_pwq30_taufb"


def build_observer_override() -> str:
    """Read CSOV observer.prototxt and flip use_tau_feedback to true."""
    src = (CSOV_CONFIG / "observer.prototxt").read_text()
    if "use_tau_feedback: false" not in src:
        raise RuntimeError(
            "expected 'use_tau_feedback: false' in CSOV observer.prototxt; "
            "found:\n" + src
        )
    return src.replace("use_tau_feedback: false", "use_tau_feedback: true")


def extract_seed_post_eta(seed_dir: Path):
    out_path = seed_dir / f"{TAG}_seed{seed_dir.name.split('seed')[-1]}.out"
    if not out_path.exists():
        return None
    res = parse_output(out_path)
    t = res.columns["t"]
    surge = res.columns["SurgeDev"]
    sway = res.columns["SwayDev"]
    mask = (t >= T_WCF_S) & (t <= T_WCF_S + T_POST_S)
    t_post = t[mask] - T_WCF_S
    surge0 = float(surge[mask][0])
    sway0 = float(sway[mask][0])
    dsurge = surge[mask] - surge0
    dsway = sway[mask] - sway0
    return t_post, dsurge, dsway


def main():
    print("=" * 78)
    print("Brucon pwq30 re-run with use_tau_feedback = TRUE")
    print(f"  N_SEEDS = {N_SEEDS}, work_dir = {WORK_DIR}")
    print("=" * 78)

    override_text = build_observer_override()
    flag_line = next(l for l in override_text.splitlines() if "use_tau_feedback" in l)
    print(f"\n[verify] override observer.prototxt: '{flag_line.strip()}'")

    spec = ScenarioSpec(
        Hs=HS, Tp=TP, wave_dir_compass=WAVE_DIR_COMPASS,
        wind_speed=0.0, wind_dir_compass=WAVE_DIR_COMPASS,
        current_speed=0.0, current_dir_compass=WAVE_DIR_COMPASS,
        vessel_heading_compass=VESSEL_HEADING_COMPASS,
        failed_thruster_indices=CSOV_WCF_GROUPS["bus_port"],
        activate_sk_s=ACTIVATE_SK_S,
        settle_s=SETTLE_S,
        post_failure_s=T_POST_S,
        print_every_steps=1,
        config_overrides={"observer.prototxt": override_text},
    )

    print(f"\n[sim] running {N_SEEDS}-seed ensemble (~25 min) ...")
    t0 = time.time()
    results = run_ensemble(spec, n_seeds=N_SEEDS, work_dir=WORK_DIR, tag=TAG)
    dt = time.time() - t0
    print(f"  done in {dt:.0f} s wall ({N_SEEDS * spec.total_seconds / dt:.0f}x realtime aggregate)\n")

    # --- collect per-seed Δη trajectories and peaks ---
    per_seed = []
    for k in range(N_SEEDS):
        seed = 1000 + k
        rd = WORK_DIR / f"{TAG}_seed{seed:04d}"
        ext = extract_seed_post_eta(rd)
        if ext is None:
            print(f"  WARN: seed {seed} had no .out; skipping")
            continue
        per_seed.append((seed, *ext))

    if not per_seed:
        raise RuntimeError("no seeds produced output")

    t_post = per_seed[0][1]
    n_t = len(t_post)
    dsurge_all = np.array([s[2][:n_t] for s in per_seed])
    dsway_all = np.array([s[3][:n_t] for s in per_seed])

    truth_dsurge_mean = dsurge_all.mean(axis=0)
    truth_dsway_mean = dsway_all.mean(axis=0)

    # Per-seed |Δradial|_peak in [0, 60] s
    mask60 = t_post <= 60.0
    radial = np.sqrt(dsurge_all[:, mask60] ** 2 + dsway_all[:, mask60] ** 2)
    peaks_60 = radial.max(axis=1)

    # --- load existing pwq30 (use_tau_feedback=false) for overlay ---
    npz_path = "/tmp/status_ki_on.npz"
    have_baseline = False
    try:
        base = np.load(npz_path)
        t_base = base["t_truth"]
        base_dsway = base["truth_dsway_mean"]
        base_dsurge = base["truth_dsurge_mean"]
        base_peaks_60 = base["truth_peaks_60"]
        have_baseline = True
    except FileNotFoundError:
        print(f"  WARN: baseline {npz_path} not found; plotting taufb only")

    # --- plot ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    if have_baseline:
        ax.plot(t_base, base_dsurge, "C0-", lw=1.5,
                label=f"baseline (use_tau_feedback=false), n={N_SEEDS}")
    ax.plot(t_post, truth_dsurge_mean, "C2-", lw=2,
            label=f"taufb=TRUE, n={len(per_seed)}")
    ax.set_title("Brucon ensemble-mean Δsurge")
    ax.set_xlabel("t since WCF [s]"); ax.set_ylabel("Δsurge [m]")
    ax.axhline(0, color="k", lw=0.4); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[0, 1]
    if have_baseline:
        ax.plot(t_base, base_dsway, "C0-", lw=1.5,
                label=f"baseline (use_tau_feedback=false), n={N_SEEDS}")
    ax.plot(t_post, truth_dsway_mean, "C2-", lw=2,
            label=f"taufb=TRUE, n={len(per_seed)}")
    ax.axhline(-0.31, color="k", ls=":", alpha=0.5,
               label="cqa analytic linearised pulse peak (-0.31 m)")
    ax.set_title("Brucon ensemble-mean Δsway")
    ax.set_xlabel("t since WCF [s]"); ax.set_ylabel("Δsway [m]")
    ax.axhline(0, color="k", lw=0.4); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 0]
    for s in dsway_all:
        ax.plot(t_post, s, color="C2", alpha=0.25, lw=0.6)
    ax.plot(t_post, truth_dsway_mean, "C2-", lw=2, label="taufb=TRUE mean")
    if have_baseline:
        ax.plot(t_base, base_dsway, "C0-", lw=1.5, label="baseline mean")
    ax.set_title("Per-seed Δsway (taufb=TRUE) + means")
    ax.set_xlabel("t since WCF [s]"); ax.set_ylabel("Δsway [m]")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 1]
    bins = np.linspace(0, 3.5, 30)
    ax.hist(peaks_60, bins=bins, alpha=0.7, color="C2",
            label=f"taufb=TRUE  P50={np.median(peaks_60):.2f}  P95={np.quantile(peaks_60, 0.95):.2f} m")
    if have_baseline:
        ax.hist(base_peaks_60, bins=bins, alpha=0.5, color="C0",
                label=f"baseline   P50={np.median(base_peaks_60):.2f}  P95={np.quantile(base_peaks_60, 0.95):.2f} m")
    ax.set_title("Per-seed |Δradial|_peak in [0, 60] s")
    ax.set_xlabel("|Δradial|_peak [m]"); ax.set_ylabel("count")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("pwq30 brucon ensemble: use_tau_feedback flag toggle", y=1.00)
    plt.tight_layout()
    out_png = THIS / "pwq30_taufb_overlay.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"saved {out_png}")

    # --- summary ---
    print()
    print("--- summary ---")
    peak_mean_y = float(np.max(np.abs(truth_dsway_mean)))
    t_peak_mean_y = float(t_post[np.argmax(np.abs(truth_dsway_mean))])
    print(f"taufb=TRUE: ensemble-mean Δsway peak = {peak_mean_y:.3f} m at t={t_peak_mean_y:.1f} s")
    print(f"taufb=TRUE: per-seed peak60: P50={np.median(peaks_60):.3f} m, "
          f"P95={np.quantile(peaks_60, 0.95):.3f} m, max={peaks_60.max():.3f} m")
    if have_baseline:
        base_peak_mean_y = float(np.max(np.abs(base_dsway)))
        t_base_peak = float(t_base[np.argmax(np.abs(base_dsway))])
        print(f"baseline:   ensemble-mean Δsway peak = {base_peak_mean_y:.3f} m at t={t_base_peak:.1f} s")
        print(f"baseline:   per-seed peak60: P50={np.median(base_peaks_60):.3f} m, "
              f"P95={np.quantile(base_peaks_60, 0.95):.3f} m, max={base_peaks_60.max():.3f} m")
        print()
        d_peak = peak_mean_y - base_peak_mean_y
        d_t = t_peak_mean_y - t_base_peak
        print(f"  Δ(peak amplitude) = {d_peak*100:+.1f} cm")
        print(f"  Δ(peak time)      = {d_t:+.1f} s")
        print(f"  vs cqa analytic linearised peak = -0.31 m at t=15.5 s")
        if peak_mean_y < 0.40:
            verdict = ("Observer-input mismatch IS the dominant gap mechanism: "
                       "the peak shrinks toward the cqa analytic prediction.")
        elif peak_mean_y < base_peak_mean_y - 0.05:
            verdict = "Partial contribution; mechanism real but not the only factor."
        else:
            verdict = ("Mechanism rejected: observer-input mismatch is NOT the "
                       "dominant cause of the gap.")
        print(f"  verdict: {verdict}")

    np.savez(
        "/tmp/pwq30_taufb_status.npz",
        t_truth=t_post,
        truth_dsway_mean=truth_dsway_mean,
        truth_dsurge_mean=truth_dsurge_mean,
        peaks_60=peaks_60,
        dsway_all=dsway_all,
        dsurge_all=dsurge_all,
    )
    print(f"\nsaved /tmp/pwq30_taufb_status.npz")


if __name__ == "__main__":
    main()
