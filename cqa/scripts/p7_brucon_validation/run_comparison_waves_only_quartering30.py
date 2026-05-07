"""P7 cross-validation, WAVES-ONLY variant.

Same as run_comparison.py but with wind_speed = 0 and current_speed = 0
on BOTH the brucon side (SetWindCondition / SetCurrentCondition take 0)
and the cqa side (Vw = 0 zeros S_wind in the Lyapunov sum, Vc = 0 zeros
S_curr). This isolates the wave-drift -> closed-loop transfer so we can
attribute any residual gap in the full-env P7 comparison to the wind /
current channels rather than the slow-drift physics that was just fixed
in §12.18.

Lua sequence is otherwise identical to the full-env P7: same precondition
+ settle + WCFDI failure (Bus port = Bow1 + PortMP) + post-failure window.
This means the WCFDI transient is included; with no env to push the
vessel between failure and recovery it should be small, and that's part
of the diagnostic value (large WCFDI excursion in waves-only would
indicate the failure-event itself, not env loading, drives the
transient).

Output artefacts use a `pwo_` prefix and `p7_waves_only_` filename
prefix to keep the artefact tree disjoint from the full-env P7 outputs:

    work/pwo_seedNNNN/                 brucon ensemble work dir
    p7_waves_only_validation_intact_cdf.png
    p7_waves_only_validation_transient.png

Run from cqa root with:
    .venv/bin/python scripts/p7_brucon_validation/run_comparison_waves_only.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from harness import (
    ScenarioSpec, run_ensemble, CSOV_WCF_GROUPS, SIM_DT,
)

from cqa.transient import wcfdi_transient, WcfdiScenario
from cqa.decision_matrix import _build_intact_prior_at_forecast
from cqa.sea_state_relations import pm_hs_from_vw, pm_tp_from_vw
from cqa.rao import load_pdstrip_rao

# Reuse setup_cqa (cqa controller tuning + gangway state) from the
# full-env script -- waves-only does not change the vessel/controller
# configuration, only the environment.
from run_comparison import setup_cqa  # noqa: E402

PDSTRIP_PATH = (
    "/home/blofro/src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"
)


# ----------------------------------------------------------------------
# Scenario constants -- mirror run_comparison.py except VW = 0, VC = 0
# ----------------------------------------------------------------------
# We keep VW around as a *nominal* PM seed-state for Hs/Tp (so the wave
# spectrum is identical to full-env P7), but we pass 0.0 to brucon /
# cqa for the actual wind disturbance.
VW_FOR_PM = 14.0                       # m/s, only used to derive Hs / Tp
HS = pm_hs_from_vw(VW_FOR_PM)          # 4.20 m
TP = pm_tp_from_vw(VW_FOR_PM)          # 10.22 s

VW_DISTURBANCE = 0.0                   # m/s -- waves-only
VC_DISTURBANCE = 0.0                   # m/s -- waves-only

WAVE_DIR_COMPASS = 210.0   # 30 deg off port bow (β = +30°, bow-quartering port)
VESSEL_HEADING_COMPASS = 180.0
N_SEEDS = 30
T_OP_S = 30.0 * 60.0
ENSEMBLE_TAG = "pwq30"


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    work_dir = out_dir / "work"

    print("==========================================================")
    print("P7 cross-validation -- WAVES-ONLY (Vw=0, Vc=0)")
    print("==========================================================")
    print(f"  Hs (PM @ Vw=14)= {HS:.3f} m")
    print(f"  Tp (PM @ Vw=14)= {TP:.3f} s")
    print(f"  Vw (disturb.)  = {VW_DISTURBANCE:.2f} m/s")
    print(f"  Vc (disturb.)  = {VC_DISTURBANCE:.2f} m/s")
    print(f"  wave dir       = {WAVE_DIR_COMPASS:.1f} deg compass (from)")
    print(f"  vessel heading = {VESSEL_HEADING_COMPASS:.1f} deg compass")
    rel_deg = (WAVE_DIR_COMPASS - VESSEL_HEADING_COMPASS + 540) % 360 - 180
    print(f"  -> relative weather direction (into vessel) = {rel_deg:.1f} deg "
          f"({'beam' if 60 <= abs(rel_deg) <= 120 else 'oblique'})")
    print(f"  WCFDI         = Bus port (Bow1 + PortMP)")
    print(f"  N seeds       = {N_SEEDS}")

    # ----------------------------------------------------------------
    # 1. cqa side
    # ----------------------------------------------------------------
    print("\n[cqa] building intact prior + WCFDI transient (waves only) ...")
    cfg, joint = setup_cqa()

    print(f"  loading RAO+QTF from {PDSTRIP_PATH}")
    rao = load_pdstrip_rao(PDSTRIP_PATH)

    theta_rel = np.radians(rel_deg)

    t_cqa_start = time.time()
    prior = _build_intact_prior_at_forecast(
        cfg, joint,
        Vw=VW_DISTURBANCE,                   # 0 -> S_wind = 0
        Hs=HS, Tp=TP,
        Vc=VC_DISTURBANCE,                   # 0 -> S_curr = 0 (and dFdVc not used)
        theta_rel=theta_rel,
        rao_table=rao,                       # spectral pdstrip slow-drift
        sigma_Vc=0.1, tau_Vc=600.0,          # ignored when Vc = 0
        T_op_s=T_OP_S, quantile_p=0.90,
        omega_grid=None, use_pm_for_drift=False,
    )

    scenario = WcfdiScenario(
        alpha=(0.5, 0.7, 0.5),
        gamma_immediate=0.8,
        T_realloc=5.0,
    )

    spec = ScenarioSpec(
        Hs=HS, Tp=TP, wave_dir_compass=WAVE_DIR_COMPASS,
        wind_speed=VW_DISTURBANCE,           # 0
        wind_dir_compass=WAVE_DIR_COMPASS,
        current_speed=VC_DISTURBANCE,        # 0
        current_dir_compass=WAVE_DIR_COMPASS,
        vessel_heading_compass=VESSEL_HEADING_COMPASS,
        failed_thruster_indices=CSOV_WCF_GROUPS["bus_port"],
        activate_sk_s=60.0,
        settle_s=500.0,
        post_failure_s=180.0,
        print_every_steps=1,
    )

    transient = wcfdi_transient(
        cfg=cfg, Vw_mean=VW_DISTURBANCE, Hs=HS, Tp=TP, Vc=VC_DISTURBANCE,
        theta_rel=theta_rel, scenario=scenario,
        sigma_Vc=0.1, tau_Vc=600.0,
        t_end=spec.post_failure_s,
        n_t=int(spec.post_failure_s / SIM_DT) + 1,
        rao_table=rao,
    )
    cqa_dt = time.time() - t_cqa_start
    print(f"  cqa pipeline ran in {cqa_dt:.2f} s")
    print(f"  intact P50 |pos| = {prior.pos_a_p50:.2f} m, "
          f"P90 |pos| = {prior.pos_a_p90:.2f} m")
    print(f"  transient peak |eta_mean| (surge,sway,yaw) = "
          f"({np.max(np.abs(transient.eta_mean[:, 0])):.2f} m, "
          f"{np.max(np.abs(transient.eta_mean[:, 1])):.2f} m, "
          f"{np.degrees(np.max(np.abs(transient.eta_mean[:, 2]))):.2f} deg)")
    print(f"  bistability_risk_score = "
          f"{transient.info.get('bistability_risk_score', 0.0):.2f}")

    # ----------------------------------------------------------------
    # 2. Simulator ensemble (separate tag => separate work-dir tree)
    # ----------------------------------------------------------------
    print(f"\n[sim] running {N_SEEDS}-seed ensemble (waves-only) ...")
    t_sim_start = time.time()
    results = run_ensemble(spec, n_seeds=N_SEEDS, work_dir=work_dir,
                           tag=ENSEMBLE_TAG)
    sim_dt = time.time() - t_sim_start
    print(f"  ensemble ran in {sim_dt:.1f} s wall "
          f"({N_SEEDS} x {spec.total_seconds:.0f} sim-s -> "
          f"{N_SEEDS * spec.total_seconds / sim_dt:.0f}x realtime aggregate)")

    # ----------------------------------------------------------------
    # 3. Reduce simulator data (same logic as run_comparison.py)
    # ----------------------------------------------------------------
    n_min = min(r.n_rows for r in results)
    t_sim = results[0]["t"][:n_min]
    surge_arr = np.array([r["SurgeDev"][:n_min] for r in results])
    sway_arr = np.array([r["SwayDev"][:n_min] for r in results])
    pos_arr = np.array([r["PosDev"][:n_min] for r in results])

    pos_check = np.hypot(surge_arr, sway_arr)
    pos_dev_err = np.max(np.abs(pos_arr - pos_check))
    if pos_dev_err > 0.05:
        print(f"  [warn] PosDev != hypot(SurgeDev,SwayDev): max diff = {pos_dev_err:.3f} m")

    INTACT_SAMPLE_S = 200.0
    intact_mask = (
        (t_sim >= spec.failure_time_s - INTACT_SAMPLE_S)
        & (t_sim < spec.failure_time_s - 1.0)
    )
    post_mask = t_sim >= spec.failure_time_s

    intact_running_max = np.array([np.maximum.accumulate(pos_arr[k, intact_mask])
                                    for k in range(N_SEEDS)])
    intact_max_per_seed = intact_running_max[:, -1]
    intact_p50_emp = np.median(intact_max_per_seed)
    intact_p90_emp = np.quantile(intact_max_per_seed, 0.90)

    print(f"\n[compare] intact-stats over {intact_mask.sum() * SIM_DT:.0f} s "
          f"window x {N_SEEDS} seeds (waves only):")
    print(f"  cqa  P50 |pos| = {prior.pos_a_p50:.2f} m   "
          f"sim P50 = {intact_p50_emp:.2f} m   "
          f"diff = {intact_p50_emp - prior.pos_a_p50:+.2f} m")
    print(f"  cqa  P90 |pos| = {prior.pos_a_p90:.2f} m   "
          f"sim P90 = {intact_p90_emp:.2f} m   "
          f"diff = {intact_p90_emp - prior.pos_a_p90:+.2f} m")
    intact_std_surge = surge_arr[:, intact_mask].std(axis=1)
    intact_std_sway = sway_arr[:, intact_mask].std(axis=1)
    intact_mean_surge = surge_arr[:, intact_mask].mean(axis=1)
    intact_mean_sway = sway_arr[:, intact_mask].mean(axis=1)
    print(f"  sim mean(surge) per seed: median = {np.median(intact_mean_surge):+.2f} m, "
          f"range [{intact_mean_surge.min():+.2f}, {intact_mean_surge.max():+.2f}]")
    print(f"  sim mean(sway)  per seed: median = {np.median(intact_mean_sway):+.2f} m, "
          f"range [{intact_mean_sway.min():+.2f}, {intact_mean_sway.max():+.2f}]")
    print(f"  sim std(surge)  per seed: median = {np.median(intact_std_surge):.2f} m")
    print(f"  sim std(sway)   per seed: median = {np.median(intact_std_sway):.2f} m")

    t_post = t_sim[post_mask] - spec.failure_time_s
    surge_post = surge_arr[:, post_mask]
    sway_post = sway_arr[:, post_mask]

    # Subtract per-seed offset at t=failure (matches run_comparison.py).
    surge_post = surge_post - surge_post[:, :1]
    sway_post = sway_post - sway_post[:, :1]

    surge_mean_emp = surge_post.mean(axis=0)
    sway_mean_emp = sway_post.mean(axis=0)
    surge_q_lo = np.quantile(surge_post, 0.25, axis=0)
    surge_q_hi = np.quantile(surge_post, 0.75, axis=0)
    sway_q_lo = np.quantile(sway_post, 0.25, axis=0)
    sway_q_hi = np.quantile(sway_post, 0.75, axis=0)

    t_cqa = transient.t
    eta_surge = transient.eta_mean[:, 0]
    eta_sway = transient.eta_mean[:, 1]
    sigma_surge = transient.eta_std[:, 0]
    sigma_sway = transient.eta_std[:, 1]
    K = 0.674

    # ----------------------------------------------------------------
    # 4. Plots
    # ----------------------------------------------------------------
    print("\n[plot] rendering comparison figures ...")

    fig, ax = plt.subplots(figsize=(7, 5))
    sorted_max = np.sort(intact_max_per_seed)
    cdf = np.arange(1, N_SEEDS + 1) / (N_SEEDS + 1)
    ax.plot(sorted_max, cdf, "o-", color="tab:blue",
            label=f"sim empirical CDF (N={N_SEEDS})", lw=1.5, markersize=5)
    ax.axvline(prior.pos_a_p50, color="tab:orange", ls="--",
               label=f"cqa P50 = {prior.pos_a_p50:.2f} m")
    ax.axvline(prior.pos_a_p90, color="tab:red", ls="--",
               label=f"cqa P90 = {prior.pos_a_p90:.2f} m")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.axhline(0.9, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("running-max |pos| over intact window [m]")
    ax.set_ylabel("empirical CDF")
    ax.set_title(f"WAVES-ONLY intact-stats: cqa quantiles vs simulator running-max\n"
                 f"Hs={HS:.2f} m, Tp={TP:.2f} s, beam-on (Vw=0, Vc=0), "
                 f"window = {intact_mask.sum() * SIM_DT:.0f} s")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    out1 = out_dir / "p7_waves_only_validation_intact_cdf.png"
    fig.savefig(out1, dpi=130)
    plt.close(fig)
    print(f"  wrote {out1}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, comp_name, eta, sigma, sim_mean, sim_q_lo, sim_q_hi, sim_traces in [
        (axes[0], "surge", eta_surge, sigma_surge,
         surge_mean_emp, surge_q_lo, surge_q_hi, surge_post),
        (axes[1], "sway",  eta_sway,  sigma_sway,
         sway_mean_emp,  sway_q_lo,  sway_q_hi,  sway_post),
    ]:
        for k in range(N_SEEDS):
            ax.plot(t_post, sim_traces[k], color="gray", alpha=0.15, lw=0.6)
        ax.plot(t_post, sim_mean, color="tab:blue", lw=2.0,
                label="sim ensemble mean")
        ax.fill_between(t_post, sim_q_lo, sim_q_hi, color="tab:blue", alpha=0.18,
                        label="sim IQR (25-75%)")
        ax.plot(t_cqa, eta, color="tab:red", lw=2.0, ls="--",
                label="cqa eta_mean")
        ax.fill_between(t_cqa, eta - K * sigma, eta + K * sigma,
                        color="tab:red", alpha=0.18,
                        label=f"cqa mean +/- {K:.3f}*sigma")
        ax.set_ylabel(f"{comp_name} deviation [m]")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.4)
        ax.axvline(0.0, color="black", ls=":", alpha=0.4)

    axes[1].set_xlabel("time since failure [s]")
    fig.suptitle(
        f"WAVES-ONLY post-WCFDI transient -- cqa vs simulator (N={N_SEEDS} seeds)\n"
        f"Hs={HS:.2f} m, Tp={TP:.2f} s, theta_rel={rel_deg:+.0f} deg, "
        f"Vw=0, Vc=0, Bus port lost\n"
        f"bistability_risk_score = {transient.info.get('bistability_risk_score', 0.0):.2f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out2 = out_dir / "p7_waves_only_validation_transient.png"
    fig.savefig(out2, dpi=130)
    plt.close(fig)
    print(f"  wrote {out2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
