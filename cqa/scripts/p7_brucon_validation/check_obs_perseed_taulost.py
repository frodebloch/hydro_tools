"""Drive cqa-27 with PER-SEED tau_lost(t) extracted from brucon logs.

Background
----------
The earlier ``check_obs_vs_brucon.py`` drove cqa-27 with an
ensemble-typical sway pulse (-200 kN peak, 9 s linear decay). That
recovered the post-WCF amplitude (1% match on truth Δsway peak) but
left an 11-s peak-timing offset (cqa peak at +23 s vs brucon mean +34 s).

The expected cause is brucon's thruster RPM spool-down dynamics:
when a thruster trips, RPM ramps down at a rate-limited rate (~10%/s)
and thrust ~ RPM·|RPM|, so the deficit envelope is delayed and
asymmetric, NOT a step or 9-s exponential. Allocator re-tasking of
healthy thrusters and saturation also shape the actual deficit.

This script extracts the actual per-seed deficit
::
    tau_lost(t) = OrderTau(t) - T(t)     for t >= t_WCF
    tau_lost(t) = 0                      for t <  t_WCF

where ``OrderTau{Surge,Sway,Yaw}`` is the controller demand at the
allocator output and ``T{x,y,z}`` is the simulated total delivered
thruster force on the hull (per the user's domain steer). This
captures spool-down + reallocation + saturation in one signal,
seed by seed.

The cqa-27 model is then driven with each seed's tau_lost(t), and
the resulting truth Δsway and η̂ Δsway are overlaid on the brucon
ensemble to test whether the per-seed deficit closes the timing gap.

Run with::
    PYTHONPATH=. .venv/bin/python scripts/p7_brucon_validation/check_obs_perseed_taulost.py
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
    N_STATE,
)

WORK_ROOT = THIS / "work"
TAG = "pwq30"
SEEDS = list(range(1000, 1030))
T_WCF = 560.0  # match check_obs_vs_brucon.py: aligns to onset of Order-T divergence
               # (Alert.log fires at 562.1 s but the actual deficit starts ~2 s earlier)
BASELINE_START = 5.0   # baseline window: [T_WCF - BASELINE_END, T_WCF - BASELINE_START]
BASELINE_END = 30.0    # 25 s of pre-WCF data, ~2.5 wave periods (Tp ~ 10 s)
                       # -> wave-frequency mean residual ~ 1/sqrt(2.5) of std ~ 0.3 m -> 0.06 m SE per seed
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


def load_seed(seed: int):
    """Return per-seed arrays interpolated onto t_grid (rel. to t_WCF).

    Returns dict with keys:
      t_grid, truth_dsurge, truth_dsway, etahat_dsurge, etahat_dsway,
      tau_lost (N, 3) -- (surge, sway, yaw) deficit in [N, N, Nm], SI.
    All quantities are deviations from t_WCF^- baseline.
    """
    seed_dir = WORK_ROOT / f"{TAG}_seed{seed:04d}"
    if not seed_dir.exists():
        return None
    try:
        cols = load_main(seed_dir)
    except (StopIteration, OSError):
        return None
    t = cols["t"]
    if t[-1] < T_WCF + 30.0:
        return None  # truncated, skip

    # Body-frame truth deviations.
    # Baseline: per-seed time-AVERAGED mean over a pre-WCF window
    # [T_WCF - BASELINE_END, T_WCF - BASELINE_START], NOT the instantaneous
    # value at t_WCF. The latter would inject a per-seed wave-frequency
    # offset of O(0.5 m std) into the trace, which after ensemble-averaging
    # creates a spurious coherent ~10-15s sway swing around t=0 (the
    # "upward bump" before the WCF dip in earlier diagnostics). Time-mean
    # baseline is unbiased: each seed's wave-frequency component averages
    # to ~0 over an integer number of wave periods.
    s_b, w_b = project_ned_to_body(cols["x"], cols["y"], cols["heading"])
    base_mask = (t >= T_WCF - BASELINE_END) & (t <= T_WCF - BASELINE_START)
    s_b -= s_b[base_mask].mean()
    w_b -= w_b[base_mask].mean()
    sd = cols["SurgeDev"] - cols["SurgeDev"][base_mask].mean()
    wd = cols["SwayDev"] - cols["SwayDev"][base_mask].mean()

    # Per-seed tau_lost: the hull receives T but cqa's tau_thr at SS would
    # equal -Order (i.e. cqa expects the controller's commanded force to be
    # delivered). The "lost" contribution that breaks the SS is therefore
    #     tau_lost = T - Order      (signed delta added to truth nu_dot via B_lost = +Minv)
    # Sanity: in seed 1000 around t_WCF, Order_sway = +108 kN, T_y = -29 kN,
    # so tau_lost_sway = -29 - 108 = -137 kN (NEGATIVE: less +sway force than
    # the model expects -> truth drifts in -sway, matching brucon truth).
    deficit_surge = (cols["Tx"] - cols["OrderTauSurge"]) * 1e3
    deficit_sway = (cols["Ty"] - cols["OrderTauSway"]) * 1e3
    deficit_yaw = (cols["Tz"] - cols["OrderTauYaw"]) * 1e3
    # Zero before WCF (per user: "raw, zero before WCF")
    pre = t < T_WCF
    deficit_surge[pre] = 0.0
    deficit_sway[pre] = 0.0
    deficit_yaw[pre] = 0.0

    # Resample to common t_grid (relative to t_WCF)
    t_rel = t - T_WCF
    t_grid = np.arange(-T_PRE, T_POST + DT / 2, DT)
    return dict(
        t_grid=t_grid,
        truth_dsurge=np.interp(t_grid, t_rel, s_b),
        truth_dsway=np.interp(t_grid, t_rel, w_b),
        etahat_dsurge=np.interp(t_grid, t_rel, sd),
        etahat_dsway=np.interp(t_grid, t_rel, wd),
        tau_lost=np.column_stack([
            np.interp(t_grid, t_rel, deficit_surge),
            np.interp(t_grid, t_rel, deficit_sway),
            np.interp(t_grid, t_rel, deficit_yaw),
        ]),
    )


def build_cqa():
    cfg = csov_default_config()
    vessel = LinearVesselModel.from_config(cfg.vessel)
    omega_n = np.array([0.060, 0.080, 0.120])
    zeta = np.array([0.95, 0.95, 0.95])
    ctrl = LinearDpController.from_bandwidth(vessel.M, vessel.D, omega_n=omega_n, zeta=zeta)
    return build_observer_augmented_system_full(vessel, ctrl, T_thr=5.0)


def cqa_response_to_seed_taulost(aug, t_grid: np.ndarray, tau_lost_full: np.ndarray):
    """Drive cqa-27 with the per-seed tau_lost trace.

    Pre-WCF section is held at zero perturbation; post-WCF section is
    integrated forward via pulse_response.
    """
    idx_wcf = int(np.argmin(np.abs(t_grid)))
    t_post = t_grid[idx_wcf:] - t_grid[idx_wcf]
    tau_post = tau_lost_full[idx_wcf:, :]
    X_post = pulse_response(aug, t_post, tau_post, x0=np.zeros(N_STATE))
    X = np.zeros((len(t_grid), N_STATE))
    X[idx_wcf:] = X_post
    return X


def main():
    aug = build_cqa()

    seed_data = []
    for seed in SEEDS:
        d = load_seed(seed)
        if d is None:
            print(f"  seed {seed}: skipped (missing or truncated)")
            continue
        seed_data.append((seed, d))
    print(f"Loaded {len(seed_data)}/{len(SEEDS)} seeds")
    if not seed_data:
        sys.exit("No seeds loaded.")

    t_grid = seed_data[0][1]["t_grid"]

    # Stack brucon arrays
    truth_dsway = np.array([d["truth_dsway"] for _, d in seed_data])
    truth_dsurge = np.array([d["truth_dsurge"] for _, d in seed_data])
    etahat_dsway = np.array([d["etahat_dsway"] for _, d in seed_data])
    tau_lost = np.array([d["tau_lost"] for _, d in seed_data])  # (Nseed, Nt, 3)

    # Drive cqa-27 per seed
    cqa_truth_sway = np.zeros_like(truth_dsway)
    cqa_truth_surge = np.zeros_like(truth_dsurge)
    cqa_etahat_sway = np.zeros_like(etahat_dsway)
    for k, (seed, d) in enumerate(seed_data):
        X = cqa_response_to_seed_taulost(aug, t_grid, d["tau_lost"])
        cqa_truth_surge[k] = X[:, 0]
        cqa_truth_sway[k] = X[:, 1]
        cqa_etahat_sway[k] = X[:, 7]

    # Headlines
    post = t_grid > 0
    truth_dsway_m = truth_dsway.mean(0)
    cqa_truth_sway_m = cqa_truth_sway.mean(0)
    etahat_dsway_m = etahat_dsway.mean(0)
    cqa_etahat_sway_m = cqa_etahat_sway.mean(0)
    print("\n--- sway peak comparison (ensemble means) ---")
    i_b = np.argmin(truth_dsway_m[post])
    i_c = np.argmin(cqa_truth_sway_m[post])
    print(f"  brucon truth   peak : {truth_dsway_m[post][i_b]:+.3f} m at t={t_grid[post][i_b]:+.1f} s")
    print(f"  cqa-27 truth   peak : {cqa_truth_sway_m[post][i_c]:+.3f} m at t={t_grid[post][i_c]:+.1f} s")
    i_b = np.argmin(etahat_dsway_m[post])
    i_c = np.argmin(cqa_etahat_sway_m[post])
    print(f"  brucon η̂      peak : {etahat_dsway_m[post][i_b]:+.3f} m at t={t_grid[post][i_b]:+.1f} s")
    print(f"  cqa-27 η̂      peak : {cqa_etahat_sway_m[post][i_c]:+.3f} m at t={t_grid[post][i_c]:+.1f} s")

    # Inspect tau_lost ensemble character
    tl_sway_m = tau_lost[:, :, 1].mean(0) * 1e-3
    tl_sway_p5 = np.percentile(tau_lost[:, :, 1], 5, 0) * 1e-3
    tl_sway_p95 = np.percentile(tau_lost[:, :, 1], 95, 0) * 1e-3
    print(f"\n--- per-seed tau_lost (sway) statistics ---")
    pst = (t_grid >= 0) & (t_grid <= 30)
    print(f"  ensemble-mean sway deficit peak : {tl_sway_m[pst].max():+.1f} kN at t={t_grid[pst][np.argmax(tl_sway_m[pst])]:+.1f} s")
    print(f"  ensemble-mean sway deficit min  : {tl_sway_m[pst].min():+.1f} kN at t={t_grid[pst][np.argmin(tl_sway_m[pst])]:+.1f} s")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

    # Δsway truth comparison (per-seed brucon vs per-seed cqa, plus means)
    ax = axes[0, 0]
    ax.fill_between(t_grid, np.percentile(truth_dsway, 5, 0), np.percentile(truth_dsway, 95, 0),
                    alpha=0.15, color="C0", label="brucon 5-95%")
    ax.fill_between(t_grid, np.percentile(cqa_truth_sway, 5, 0), np.percentile(cqa_truth_sway, 95, 0),
                    alpha=0.12, color="C3")
    ax.plot(t_grid, truth_dsway_m, color="C0", lw=2, label=f"brucon mean ({len(seed_data)})")
    ax.plot(t_grid, cqa_truth_sway_m, color="C3", lw=2, ls="--", label="cqa-27 mean (per-seed τ_lost)")
    ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.3)
    ax.set_ylabel("Δsway truth [m]")
    ax.set_title("Truth Δsway: per-seed driven cqa-27 vs brucon")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    # η̂ sway comparison
    ax = axes[0, 1]
    ax.fill_between(t_grid, np.percentile(etahat_dsway, 5, 0), np.percentile(etahat_dsway, 95, 0),
                    alpha=0.15, color="C0")
    ax.fill_between(t_grid, np.percentile(cqa_etahat_sway, 5, 0), np.percentile(cqa_etahat_sway, 95, 0),
                    alpha=0.12, color="C3")
    ax.plot(t_grid, etahat_dsway_m, color="C0", lw=2, label="brucon η̂ mean")
    ax.plot(t_grid, cqa_etahat_sway_m, color="C3", lw=2, ls="--", label="cqa-27 η̂ mean")
    ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.3)
    ax.set_ylabel("Δsway η̂ [m]")
    ax.set_title("Observer η̂ Δsway")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    # tau_lost sway envelope
    ax = axes[1, 0]
    ax.fill_between(t_grid, tl_sway_p5, tl_sway_p95, alpha=0.2, color="C2")
    ax.plot(t_grid, tl_sway_m, color="C2", lw=2, label="ensemble mean")
    # Overlay the old -200 kN / 9 s heuristic for comparison
    t_post_pulse = np.where(t_grid > 0, t_grid, 0.0)
    pulse_old = np.where((t_grid > 0) & (t_grid <= 9.0),
                         -200.0 * (1 - t_post_pulse / 9.0), 0.0)
    ax.plot(t_grid, pulse_old, color="C1", lw=1.5, ls=":", label="old -200 kN / 9 s heuristic")
    ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.3)
    ax.set_xlabel("t − t_WCF [s]")
    ax.set_ylabel("τ_lost sway [kN]")
    ax.set_title("Per-seed sway thrust deficit (Order − T)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    # Δsurge truth comparison
    ax = axes[1, 1]
    ax.fill_between(t_grid, np.percentile(truth_dsurge, 5, 0), np.percentile(truth_dsurge, 95, 0),
                    alpha=0.15, color="C0")
    ax.fill_between(t_grid, np.percentile(cqa_truth_surge, 5, 0), np.percentile(cqa_truth_surge, 95, 0),
                    alpha=0.12, color="C3")
    ax.plot(t_grid, truth_dsurge.mean(0), color="C0", lw=2, label="brucon truth mean")
    ax.plot(t_grid, cqa_truth_surge.mean(0), color="C3", lw=2, ls="--", label="cqa-27 truth mean")
    ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.3)
    ax.set_xlabel("t − t_WCF [s]")
    ax.set_ylabel("Δsurge truth [m]")
    ax.set_title("Truth Δsurge")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = THIS / "check_obs_perseed_taulost.png"
    plt.savefig(out_png, dpi=120)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
