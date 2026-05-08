"""Diagnose dual-injection hypothesis by comparing brucon's eta_hat (the
controller's view of position) against eta_truth post-WCF.

Hypothesis (user, prior session): with use_tau_feedback=false, the observer
is fed tau_cmd which does NOT drop during WCFDI reallocation, while truth
nu DOES drop. The resulting eta_hat / nu_hat diverge from truth, the
controller acts on the (wrong) eta_hat, and the closed-loop excursion is
amplified. cqa's 15-state model has no explicit observer states and so
cannot represent this.

Diagnostic (cheap, no modelling):
  - Truth body surge/sway = project(x, y; heading) - project at t=t_WCF^-
  - eta_hat body surge/sway = SurgeDev, SwayDev (already body-frame LF
    estimator output that the regulator consumes).
  - Plot ensemble means of both, plus error = truth - eta_hat, over
    [-30, +120] s relative to t_WCF.
  - If error is O(0.2 m) for O(20-30 s) post-WCF, the dual-injection
    mechanism is real and warrants extending cqa to explicit observer
    states. Otherwise reject.

Convention: SwayDev sign convention = body-frame sway position deviation
from setpoint. Truth sway-body around a fixed setpoint (the lua holds a
reference point) is also a deviation if we subtract the t<t_WCF mean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent.parent))

import os
WORK_ROOT = THIS / os.environ.get("CQA_WORK_DIR", "work")
TAG = os.environ.get("CQA_TAG", "pwq30")
SEEDS = list(range(1000, 1030))
T_WCF = 560.0
T_PRE = 30.0      # window before WCF
T_POST = 120.0    # window after WCF
DT = 0.1


def project_ned_to_body(north, east, heading_deg):
    h = np.deg2rad(heading_deg)
    surge = np.cos(h) * north + np.sin(h) * east
    sway = -np.sin(h) * north + np.cos(h) * east
    return surge, sway


def load_main(seed_dir: Path) -> dict[str, np.ndarray]:
    main = next(p for p in seed_dir.glob("*.out") if "estimator" not in p.name)
    with open(main) as f:
        header = f.readline().strip().split("\t")
    data = np.loadtxt(main, skiprows=1, delimiter="\t")
    return {h: data[:, i] for i, h in enumerate(header)}


def main():
    t_grid = np.arange(-T_PRE, T_POST + DT / 2, DT)
    n_t = len(t_grid)

    truth_dsurge = []
    truth_dsway = []
    etahat_dsurge = []
    etahat_dsway = []
    n_loaded = 0

    for seed in SEEDS:
        seed_dir = WORK_ROOT / f"{TAG}_seed{seed:04d}"
        if not seed_dir.exists():
            continue
        try:
            cols = load_main(seed_dir)
        except (StopIteration, OSError):
            continue
        t = cols["t"]
        # Truth body coords from NED x,y + heading
        s_b, w_b = project_ned_to_body(cols["x"], cols["y"], cols["heading"])
        # Set zero at t_WCF^-
        idx0 = int(np.searchsorted(t, T_WCF) - 1)
        s_b -= s_b[idx0]
        w_b -= w_b[idx0]
        # eta_hat body LF
        sd = cols["SurgeDev"] - cols["SurgeDev"][idx0]
        wd = cols["SwayDev"] - cols["SwayDev"][idx0]
        # Resample on common grid t - T_WCF
        tau = t - T_WCF
        truth_dsurge.append(np.interp(t_grid, tau, s_b))
        truth_dsway.append(np.interp(t_grid, tau, w_b))
        etahat_dsurge.append(np.interp(t_grid, tau, sd))
        etahat_dsway.append(np.interp(t_grid, tau, wd))
        n_loaded += 1

    print(f"Loaded {n_loaded}/{len(SEEDS)} seeds")
    if n_loaded == 0:
        return
    truth_dsurge = np.array(truth_dsurge)
    truth_dsway = np.array(truth_dsway)
    etahat_dsurge = np.array(etahat_dsurge)
    etahat_dsway = np.array(etahat_dsway)

    err_surge = truth_dsurge - etahat_dsurge
    err_sway = truth_dsway - etahat_dsway

    # Stats
    truth_dsway_m = truth_dsway.mean(0)
    etahat_dsway_m = etahat_dsway.mean(0)
    err_sway_m = err_sway.mean(0)
    truth_dsurge_m = truth_dsurge.mean(0)
    etahat_dsurge_m = etahat_dsurge.mean(0)
    err_surge_m = err_surge.mean(0)

    # Late window
    post_mask = t_grid > 0
    abs_err_sway_peak = np.max(np.abs(err_sway_m[post_mask]))
    abs_err_sway_peak_t = t_grid[post_mask][np.argmax(np.abs(err_sway_m[post_mask]))]
    err_sway_late = err_sway_m[(t_grid > 100) & (t_grid <= 120)].mean()

    truth_sway_peak = np.min(truth_dsway_m[post_mask])
    truth_sway_peak_t = t_grid[post_mask][np.argmin(truth_dsway_m[post_mask])]
    etahat_sway_peak = np.min(etahat_dsway_m[post_mask])
    etahat_sway_peak_t = t_grid[post_mask][np.argmin(etahat_dsway_m[post_mask])]

    print("\n--- sway ---")
    print(f"  truth   peak    : {truth_sway_peak:+.3f} m at t={truth_sway_peak_t:+.1f} s")
    print(f"  eta_hat peak    : {etahat_sway_peak:+.3f} m at t={etahat_sway_peak_t:+.1f} s")
    print(f"  |truth - eta_hat| peak (mean): {abs_err_sway_peak:.3f} m at t={abs_err_sway_peak_t:+.1f} s")
    print(f"  truth - eta_hat late (t in [100,120] s): {err_sway_late:+.3f} m")

    print("\n--- surge ---")
    print(f"  |err|_peak (mean): {np.max(np.abs(err_surge_m[post_mask])):.3f} m")

    print("\n  decision: if |err_sway peak| >> 0.05 m and lasts O(20-30 s) post-WCF,")
    print("            OR err_sway late-time offset matches the +0.2 m sustained truth offset,")
    print("            => dual-injection mechanism CONFIRMED, cqa needs explicit observer states.")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    ax.fill_between(t_grid, np.percentile(truth_dsway, 5, 0), np.percentile(truth_dsway, 95, 0),
                    alpha=0.15, color="C0", label="truth body Δsway, 5-95%")
    ax.plot(t_grid, truth_dsway_m, color="C0", lw=2, label="truth body Δsway, mean")
    ax.fill_between(t_grid, np.percentile(etahat_dsway, 5, 0), np.percentile(etahat_dsway, 95, 0),
                    alpha=0.15, color="C3", label="η̂ Δsway (SwayDev), 5-95%")
    ax.plot(t_grid, etahat_dsway_m, color="C3", lw=2, ls="--", label="η̂ Δsway, mean")
    ax.axvline(0, color="k", lw=0.5)
    ax.axhline(0, color="k", lw=0.3)
    ax.set_ylabel("Δsway [m]")
    ax.set_title(f"Brucon truth η vs η̂ (controller's view) — {n_loaded} seeds, post-WCF")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.fill_between(t_grid, np.percentile(err_sway, 5, 0), np.percentile(err_sway, 95, 0),
                    alpha=0.15, color="C2", label="error, 5-95%")
    ax.plot(t_grid, err_sway_m, color="C2", lw=2, label="error mean (truth − η̂)")
    ax.axvline(0, color="k", lw=0.5)
    ax.axhline(0, color="k", lw=0.3)
    ax.set_ylabel("error Δsway [m]")
    ax.set_title("Sway estimator error (truth − η̂)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t_grid, truth_dsurge_m, color="C0", lw=2, label="truth body Δsurge, mean")
    ax.plot(t_grid, etahat_dsurge_m, color="C3", lw=2, ls="--", label="η̂ Δsurge, mean")
    ax.plot(t_grid, err_surge_m, color="C2", lw=1.5, label="error mean")
    ax.axvline(0, color="k", lw=0.5)
    ax.axhline(0, color="k", lw=0.3)
    ax.set_ylabel("Δsurge [m]")
    ax.set_xlabel("t − t_WCF [s]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = THIS / f"brucon_etahat_vs_etatruth_{TAG}.png"
    plt.savefig(out_png, dpi=120)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
