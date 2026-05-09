"""Visualise per-seed delta_eta_LF(t) traces and compare candidates for the
'mean trajectory' the live cell will use.

Inputs (from peak_R_tau_lost_sigma_pwq30.py artefact):
  - delta_eta_seeds [n_seeds, N, 3]: per-seed body-frame delta_eta_LF traces
    obtained by pulse_response on the seed's realised tau_lost(t).
  - delta_eta_mean  [N, 3]: ensemble vector mean across seeds.
  - peak_R_seeds    [n_seeds]: per-seed peak |delta_eta_LF| (surge/sway).

Candidates compared:
  C1 vector-mean (current default in the calibration npz)
  C3a P50-by-peak: the seed whose peak |delta_eta_LF| sits at the median
       of the per-seed peak distribution
  C3b P75-by-peak: same, at the 75th percentile

Each candidate is shown as (delta_eta_x, delta_eta_y, |delta_eta_LF|) and
its peak |R| reported. Per-seed traces are plotted in the background as
thin grey lines.

Run:
    PYTHONPATH=. .venv/bin/python \
        scripts/p7_brucon_validation/peak_R_per_seed_traces_pwq30.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))

CALIB_NPZ = THIS / "scenario_pwq30_calibration.npz"


def main():
    if not CALIB_NPZ.exists():
        sys.exit(
            f"missing {CALIB_NPZ}\n"
            f"Run scripts/p7_brucon_validation/peak_R_b_hat_sigma_pwq30.py first."
        )
    d = np.load(CALIB_NPZ, allow_pickle=True)
    t = d["t_grid"]                                   # [N]
    deta_seeds = d["delta_eta_seeds"]                 # [n_seeds, N, 3]
    deta_mean = d["delta_eta_mean"]                   # [N, 3]
    peak_R_seeds = d["peak_R_seeds"]                  # [n_seeds]
    seed_ids = d["seed_ids"]                          # [n_seeds]
    sigma_R_b_hat = float(d["sigma_R_b_hat_m"])
    tag = str(d["tag"])
    n_seeds = int(d["n_seeds"])

    # Per-seed |delta_eta_LF|(t).
    R_seeds = np.hypot(deta_seeds[..., 0], deta_seeds[..., 1])     # [n_seeds, N]
    R_mean = np.hypot(deta_mean[:, 0], deta_mean[:, 1])             # [N]

    # ---- candidate selection ----
    # C1: vector mean (already in artefact).
    C1 = deta_mean
    C1_R = R_mean

    # C3a: seed at median peak.
    order = np.argsort(peak_R_seeds)
    i_p50 = int(order[n_seeds // 2])
    C3a = deta_seeds[i_p50]
    C3a_R = R_seeds[i_p50]

    # C3b: seed at 75th percentile peak.
    i_p75 = int(order[int(round(0.75 * (n_seeds - 1)))])
    C3b = deta_seeds[i_p75]
    C3b_R = R_seeds[i_p75]

    # Per-seed peak distribution stats.
    p50_peak = float(np.median(peak_R_seeds))
    p75_peak = float(np.quantile(peak_R_seeds, 0.75))

    print(f"Per-seed peak |delta_eta_LF| distribution (n={n_seeds}):")
    print(f"  mean = {peak_R_seeds.mean():.3f}  std = {peak_R_seeds.std():.3f}  m")
    print(f"  P50  = {p50_peak:.3f}  P75  = {p75_peak:.3f}  m")
    print(f"  range = [{peak_R_seeds.min():.3f}, {peak_R_seeds.max():.3f}] m")
    print()
    print(f"Candidates (peak |R|):")
    print(f"  C1  vector mean       = {C1_R.max():.3f}  m  "
          f"(seed used: ALL averaged)")
    print(f"  C3a P50-by-peak seed  = {C3a_R.max():.3f}  m  "
          f"(seed id {int(seed_ids[i_p50])})")
    print(f"  C3b P75-by-peak seed  = {C3b_R.max():.3f}  m  "
          f"(seed id {int(seed_ids[i_p75])})")

    # ---- plot ----
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    titles = [r"$\delta\eta_{LF,x}(t)$  [m, body frame]",
              r"$\delta\eta_{LF,y}(t)$  [m, body frame]",
              r"$|\delta\eta_{LF}|(t) = \sqrt{\delta\eta_x^2 + \delta\eta_y^2}$  [m]"]

    for k_ax, ax in enumerate(axes):
        # per-seed background
        if k_ax < 2:
            for i in range(n_seeds):
                ax.plot(t, deta_seeds[i, :, k_ax], color="0.7",
                        lw=0.6, alpha=0.6)
        else:
            for i in range(n_seeds):
                ax.plot(t, R_seeds[i], color="0.7", lw=0.6, alpha=0.6)

        # candidates
        if k_ax < 2:
            ax.plot(t, C1[:, k_ax], color="C0", lw=2.0,
                    label="C1 vector mean")
            ax.plot(t, C3a[:, k_ax], color="C2", lw=2.0,
                    label=f"C3a P50-by-peak seed ({int(seed_ids[i_p50])})")
            ax.plot(t, C3b[:, k_ax], color="C3", lw=2.0,
                    label=f"C3b P75-by-peak seed ({int(seed_ids[i_p75])})")
        else:
            ax.plot(t, C1_R, color="C0", lw=2.0,
                    label=f"C1 vector mean   (peak {C1_R.max():.2f})")
            ax.plot(t, C3a_R, color="C2", lw=2.0,
                    label=f"C3a P50-by-peak ({int(seed_ids[i_p50])})  "
                          f"(peak {C3a_R.max():.2f})")
            ax.plot(t, C3b_R, color="C3", lw=2.0,
                    label=f"C3b P75-by-peak ({int(seed_ids[i_p75])})  "
                          f"(peak {C3b_R.max():.2f})")
            ax.axhline(p50_peak, color="C2", ls="--", lw=1, alpha=0.7,
                       label=f"P50 of per-seed peak distribution = {p50_peak:.2f}")
            ax.axhline(p75_peak, color="C3", ls="--", lw=1, alpha=0.7,
                       label=f"P75 of per-seed peak distribution = {p75_peak:.2f}")

        ax.set_ylabel(titles[k_ax])
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("t since WCFDI [s]")
    plt.suptitle(
        f"Per-seed delta_eta_LF(t) at {tag} (n={n_seeds})  |  "
        f"sigma_R_b_hat = {sigma_R_b_hat:.2f} m",
        fontsize=12,
    )
    plt.tight_layout()
    out = THIS / "peak_R_per_seed_traces_pwq30.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
