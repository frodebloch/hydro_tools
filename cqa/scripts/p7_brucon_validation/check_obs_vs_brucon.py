"""Compare cqa 21-state observer-augmented pulse response against the
brucon truth ensemble (pwq30, default config: use_tau_feedback=false).

For each seed in pwq30 we have:
  - truth body Δsway, Δsurge (via project_ned_to_body of x, y, heading)
  - eta_hat body LF (SwayDev, SurgeDev cols of the .out file)

The cqa 21-state model is driven by the SAME tau_lost(t) pulse used in
pulse_response_diagnostic.py:
  - sway peak deficit = -200 kN, T_pulse = 9 s, linear decay
  (a per-DOF, per-seed tau_lost trace would be more accurate; this uses
  the ensemble-mean pulse as a first comparison.)

Plot
----
2x2 panel:
  top-left  : Δsway truth (brucon ensemble) vs cqa 21-state truth Δsway
  top-right : η̂ Δsway   (brucon ensemble) vs cqa 21-state η̂ Δsway
  bot-left  : Δsurge truth (brucon)         vs cqa 21-state truth Δsurge
  bot-right : truth-η̂ residual (brucon)    vs cqa 21-state residual

Run with:
    PYTHONPATH=. .venv/bin/python scripts/p7_brucon_validation/check_obs_vs_brucon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent.parent))

from cqa.config import csov_default_config
from cqa.vessel import LinearVesselModel
from cqa.controller import LinearDpController
from cqa.transient_obs import (
    build_observer_augmented_system_full,
    pulse_response,
    IDX_ETA, IDX_ETA_HAT,
    N_STATE,
)

WORK_ROOT = THIS / "work"
TAG = "pwq30"
SEEDS = list(range(1000, 1030))
T_WCF = 560.0
T_PRE = 30.0
T_POST = 120.0
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


def load_brucon_ensemble():
    t_grid = np.arange(-T_PRE, T_POST + DT / 2, DT)
    truth_dsurge, truth_dsway = [], []
    etahat_dsurge, etahat_dsway = [], []
    n = 0
    for seed in SEEDS:
        seed_dir = WORK_ROOT / f"{TAG}_seed{seed:04d}"
        if not seed_dir.exists():
            continue
        try:
            cols = load_main(seed_dir)
        except (StopIteration, OSError):
            continue
        t = cols["t"]
        s_b, w_b = project_ned_to_body(cols["x"], cols["y"], cols["heading"])
        idx0 = int(np.searchsorted(t, T_WCF) - 1)
        s_b -= s_b[idx0]
        w_b -= w_b[idx0]
        sd = cols["SurgeDev"] - cols["SurgeDev"][idx0]
        wd = cols["SwayDev"] - cols["SwayDev"][idx0]
        tau = t - T_WCF
        truth_dsurge.append(np.interp(t_grid, tau, s_b))
        truth_dsway.append(np.interp(t_grid, tau, w_b))
        etahat_dsurge.append(np.interp(t_grid, tau, sd))
        etahat_dsway.append(np.interp(t_grid, tau, wd))
        n += 1
    return (
        t_grid, n,
        np.array(truth_dsurge), np.array(truth_dsway),
        np.array(etahat_dsurge), np.array(etahat_dsway),
    )


def build_cqa():
    cfg = csov_default_config()
    vessel = LinearVesselModel.from_config(cfg.vessel)
    omega_n = np.array([0.060, 0.080, 0.120])
    zeta = np.array([0.95, 0.95, 0.95])
    ctrl = LinearDpController.from_bandwidth(vessel.M, vessel.D, omega_n=omega_n, zeta=zeta)
    return build_observer_augmented_system_full(vessel, ctrl, T_thr=5.0)


def cqa_pulse(t_grid):
    """Pulse response of 21-state model to brucon-typical sway deficit."""
    aug = build_cqa()
    PEAK_KN = -200.0
    T_PULSE = 9.0
    # Map t_grid (which spans [-T_PRE, +T_POST]) to a "time since WCF" axis
    # that the pulse_response driver expects starting at 0. We seed the
    # state at t=0+ (t_grid index where t_grid==0) with all-zero
    # perturbation and run forward.
    idx_wcf = int(np.argmin(np.abs(t_grid)))
    t_post = t_grid[idx_wcf:] - t_grid[idx_wcf]   # 0 .. T_POST
    tau_lost = np.zeros((len(t_post), 3))
    tau_lost[:, 1] = np.where(t_post <= T_PULSE, PEAK_KN * 1e3 * (1 - t_post / T_PULSE), 0.0)
    X_post = pulse_response(aug, t_post, tau_lost, x0=np.zeros(N_STATE))
    # Stitch onto t_grid: pre-WCF window holds zero perturbation.
    X = np.zeros((len(t_grid), N_STATE))
    X[idx_wcf:] = X_post
    return X


def main():
    t_grid, n_loaded, dsurge, dsway, ehd_surge, ehd_sway = load_brucon_ensemble()
    print(f"Loaded {n_loaded}/{len(SEEDS)} brucon seeds")

    X = cqa_pulse(t_grid)
    cqa_truth_surge = X[:, 0]
    cqa_truth_sway = X[:, 1]
    cqa_etahat_surge = X[:, 6]
    cqa_etahat_sway = X[:, 7]

    # Brucon ensemble means
    truth_dsway_m = dsway.mean(0)
    etahat_dsway_m = ehd_sway.mean(0)
    truth_dsurge_m = dsurge.mean(0)
    err_brucon_sway = (dsway - ehd_sway).mean(0)
    err_cqa_sway = cqa_truth_sway - cqa_etahat_sway

    # Headlines
    post = t_grid > 0
    print("\n--- sway peak comparison ---")
    print(f"  brucon truth  ensemble peak : {truth_dsway_m[post].min():+.3f} m at t={t_grid[post][np.argmin(truth_dsway_m[post])]:+.1f} s")
    print(f"  brucon η̂    ensemble peak : {etahat_dsway_m[post].min():+.3f} m")
    print(f"  cqa-21 truth        peak  : {cqa_truth_sway[post].min():+.3f} m at t={t_grid[post][np.argmin(cqa_truth_sway[post])]:+.1f} s")
    print(f"  cqa-21 η̂           peak  : {cqa_etahat_sway[post].min():+.3f} m")
    print(f"  brucon |truth-η̂|   peak : {np.max(np.abs(err_brucon_sway[post])):.3f} m at t={t_grid[post][np.argmax(np.abs(err_brucon_sway[post]))]:+.1f} s")
    print(f"  cqa-21 |truth-η̂|   peak : {np.max(np.abs(err_cqa_sway[post])):.3f} m at t={t_grid[post][np.argmax(np.abs(err_cqa_sway[post]))]:+.1f} s")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

    # Δsway truth comparison
    ax = axes[0, 0]
    ax.fill_between(t_grid, np.percentile(dsway, 5, 0), np.percentile(dsway, 95, 0),
                    alpha=0.15, color="C0")
    ax.plot(t_grid, truth_dsway_m, color="C0", lw=2, label=f"brucon truth, mean ({n_loaded} seeds)")
    ax.plot(t_grid, cqa_truth_sway, color="C3", lw=2, ls="--", label="cqa-21 truth (analytic pulse)")
    ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.3)
    ax.set_ylabel("Δsway truth [m]")
    ax.set_title("Truth body Δsway: brucon ensemble vs cqa 21-state")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # η̂ sway
    ax = axes[0, 1]
    ax.fill_between(t_grid, np.percentile(ehd_sway, 5, 0), np.percentile(ehd_sway, 95, 0),
                    alpha=0.15, color="C0")
    ax.plot(t_grid, etahat_dsway_m, color="C0", lw=2, label="brucon η̂, mean")
    ax.plot(t_grid, cqa_etahat_sway, color="C3", lw=2, ls="--", label="cqa-21 η̂")
    ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.3)
    ax.set_ylabel("Δsway η̂ [m]")
    ax.set_title("Observer η̂ Δsway: brucon vs cqa")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # surge truth
    ax = axes[1, 0]
    ax.fill_between(t_grid, np.percentile(dsurge, 5, 0), np.percentile(dsurge, 95, 0),
                    alpha=0.15, color="C0")
    ax.plot(t_grid, truth_dsurge_m, color="C0", lw=2, label="brucon truth, mean")
    ax.plot(t_grid, cqa_truth_surge, color="C3", lw=2, ls="--", label="cqa-21 truth")
    ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.3)
    ax.set_xlabel("t − t_WCF [s]")
    ax.set_ylabel("Δsurge truth [m]")
    ax.set_title("Truth body Δsurge: brucon vs cqa")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # truth - η̂ residual
    ax = axes[1, 1]
    ax.plot(t_grid, err_brucon_sway, color="C0", lw=2, label="brucon: truth − η̂, mean")
    ax.plot(t_grid, err_cqa_sway, color="C3", lw=2, ls="--", label="cqa-21: truth − η̂")
    ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.3)
    ax.set_xlabel("t − t_WCF [s]")
    ax.set_ylabel("residual [m]")
    ax.set_title("Sway observer error (truth − η̂)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = THIS / "check_obs_vs_brucon.png"
    plt.savefig(out_png, dpi=120)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
