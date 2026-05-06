"""P7 cross-validation: compare cqa predictions against the brucon
RunFastStandalone nonlinear simulator at one (sea state, heading,
WCFDI) scenario.

Pipeline
--------
1. Define the canonical scenario (PM-consistent at Vw=14 m/s, beam-on,
   Bus port WCFDI = Bow1 + PortMP).

2. cqa side:
     a. _build_intact_prior_at_forecast -> P50/P90 of running-max |pos|
        over T_op.
     b. wcfdi_transient -> deterministic mean trajectory eta_mean(t) and
        std envelope eta_std(t) over the post-failure window.

3. Simulator side:
     run_ensemble(spec, n_seeds=N) -> N realisations with shared (Hs, Tp,
     wind, current) but independent wave seeds.

4. Align time axes:
     simulator t = 0 at script start;
     cqa transient  t = 0 at the failure event.
     -> simulator t' = t - failure_time_s aligns the two.

5. Compute and overlay:
     a. Intact-stats panel: cqa P50/P90 of running-max |pos| over the
        intact window vs the empirical CDF over the same simulator window.
     b. Transient-mean panel: cqa eta_mean(t) (body frame) vs simulator
        ensemble mean of body-frame surge/sway.
     c. Transient-envelope panel: cqa mean +/- k*sigma vs simulator
        ensemble quantile band over time.

All plots saved to scripts/p7_brucon_validation/p7_validation_*.png.

Run from cqa root with:
    .venv/bin/python scripts/p7_brucon_validation/run_comparison.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
# cqa package root is three levels up: scripts/p7_brucon_validation/ -> scripts/ -> cqa/ -> repo root
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from harness import (
    ScenarioSpec, run_ensemble, CSOV_WCF_GROUPS, SIM_DT,
)

# cqa imports
from cqa.config import csov_default_config
from cqa.gangway import GangwayJointState
from cqa.transient import wcfdi_transient, WcfdiScenario
from cqa.decision_matrix import _build_intact_prior_at_forecast
from cqa.sea_state_relations import pm_hs_from_vw, pm_tp_from_vw
from cqa.rao import load_pdstrip_rao

# Path to the brucon CSOV pdstrip RAO+QTF table; same file used by
# compare_forces.py for the spectral-drift force-level cross-check.
PDSTRIP_PATH = "/home/blofro/src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"


# ----------------------------------------------------------------------
# Canonical scenario
# ----------------------------------------------------------------------
VW = 14.0                                # m/s
HS = pm_hs_from_vw(VW)                   # 4.20 m at Vw=14 (PM fully developed)
TP = pm_tp_from_vw(VW)                   # 10.22 s at Vw=14
VC = 0.5                                 # m/s
WAVE_DIR_COMPASS = 270.0                 # weather coming from west
VESSEL_HEADING_COMPASS = 180.0           # bow south -> beam-on to weather
N_SEEDS = 30
T_OP_S = 30.0 * 60.0                     # cqa intact-prior operation window (30 min)


def setup_cqa() -> tuple:
    """Build cqa CqaConfig and a default GangwayJointState for CSOV."""
    cfg = csov_default_config()

    # Sync cqa's controller bandwidth + damping to the simulator's
    # ~/src/brucon/build/bin/settings/tuning.prototxt values, so the
    # cross-validation compares two identically-tuned closed-loop systems
    # rather than two different ones. Also pin the runtime gain level to
    # kMedium in the lua template (no scaling).
    #   pid_surge.omega = 0.06,  relative_damping = 0.95
    #   pid_sway.omega  = 0.08,  relative_damping = 0.95
    #   pid_yaw.omega   = 0.12,  relative_damping = 0.95
    cfg.controller.omega_n_surge = 0.06
    cfg.controller.omega_n_sway = 0.08
    cfg.controller.omega_n_yaw = 0.12
    cfg.controller.zeta_surge = 0.95
    cfg.controller.zeta_sway = 0.95
    cfg.controller.zeta_yaw = 0.95

    # Use the body-frame gangway base position straight from the config.
    # Mid-stroke L = 25 (telescope_min=18, telescope_max=32 per cfg.gangway).
    gw = cfg.gangway
    L_mid = 0.5 * (gw.telescope_min + gw.telescope_max)
    joint = GangwayJointState(
        h=gw.rotation_centre_height_above_base,
        alpha_g=0.0,
        beta_g=0.0,
        L=L_mid,
    )
    return cfg, joint


def simulator_to_body_frame(x_ned: np.ndarray, y_ned: np.ndarray,
                            heading_deg: np.ndarray,
                            heading0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Project NED simulator position into body frame at the *initial* heading.

    For the intact / WCFDI comparison we want surge/sway *deviation from
    setpoint*. The setpoint is fixed at (0, 0) NED in the simulator
    (no SetPositionSetpoint call is made before failure), so the position
    column already IS the deviation. The body-frame projection uses the
    initial vessel heading (not the time-varying heading), so a positive
    'sway' means deviation to the body-frame port direction at the moment
    of failure -- this is what cqa eta_mean[1] also represents (relative
    to the linearised system about the setpoint heading).
    """
    psi = np.radians(heading0_deg)
    c, s = np.cos(psi), np.sin(psi)
    # Body frame: surge along the bow (heading), sway 90 deg to starboard.
    # NED -> body: rotate by -psi.
    surge = c * x_ned + s * y_ned
    sway = -s * x_ned + c * y_ned
    return surge, sway


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    work_dir = out_dir / "work"

    print("==========================================================")
    print("P7 cross-validation -- brucon vessel_simulator vs cqa")
    print("==========================================================")
    print(f"  Vw            = {VW:.2f} m/s")
    print(f"  Hs (PM)       = {HS:.3f} m")
    print(f"  Tp (PM)       = {TP:.3f} s")
    print(f"  Vc            = {VC:.2f} m/s")
    print(f"  wave dir      = {WAVE_DIR_COMPASS:.1f} deg compass (from)")
    print(f"  vessel heading = {VESSEL_HEADING_COMPASS:.1f} deg compass")
    rel_deg = (WAVE_DIR_COMPASS - VESSEL_HEADING_COMPASS + 540) % 360 - 180
    print(f"  -> relative weather direction (into vessel) = {rel_deg:.1f} deg "
          f"({'beam' if 60 <= abs(rel_deg) <= 120 else 'oblique'})")
    print(f"  WCFDI         = Bus port (Bow1 + PortMP)")
    print(f"  N seeds       = {N_SEEDS}")

    # ----------------------------------------------------------------
    # 1. cqa side
    # ----------------------------------------------------------------
    print("\n[cqa] building intact prior + WCFDI transient ...")
    cfg, joint = setup_cqa()

    # Load brucon's pdstrip RAO+QTF table so cqa uses the *same* mean-drift
    # and slow-drift PSD as the simulator. Without this, cqa falls back to
    # parametric WaveDriftParticulars which for CSOV yield wrong sign and
    # missing yaw-drift (see analysis.md sec.12.16). This was the suspected
    # root cause of the closed-loop intact P50/P90 under-prediction and the
    # post-failure runaway-vs-recovery mismatch.
    print(f"  loading RAO+QTF from {PDSTRIP_PATH}")
    rao = load_pdstrip_rao(PDSTRIP_PATH)

    # Convert relative compass direction to cqa convention:
    # cqa theta_rel = direction the weather comes *into* the vessel,
    # 0 = head-on, +pi/2 = from starboard (beam to starboard).
    # Compass-relative: weather_from - vessel_heading = 90 deg means weather
    # is from starboard side, i.e. theta_rel = +pi/2.
    theta_rel = np.radians(rel_deg)

    t_cqa_start = time.time()
    prior = _build_intact_prior_at_forecast(
        cfg, joint,
        Vw=VW, Hs=HS, Tp=TP, Vc=VC,
        theta_rel=theta_rel,
        rao_table=rao,                  # spectral pdstrip drift (matches brucon)
        sigma_Vc=0.1, tau_Vc=600.0,
        T_op_s=T_OP_S, quantile_p=0.90,
        omega_grid=None, use_pm_for_drift=False,
    )

    scenario = WcfdiScenario(
        # Per-DOF surviving capability after bus-port WCFDI on a CSOV
        # (lost: Bow1 + PortMP; survives: Bow2, BowAz, StbdMP).
        #   alpha_surge = 0.5  : 1 of 2 large main propulsors lost
        #   alpha_sway  = 0.7  : 1 of 2 bow tunnels lost, BowAz still
        #                        contributes sway, mains can also sway via differential
        #   alpha_yaw   = 0.5  : lever-arm capability roughly halved
        alpha=(0.5, 0.7, 0.5),
        # Immediate available cap fraction. brucon's BasicAllocator re-solves
        # in < 1 s on the healthy bus; tunnels/azimuths on the surviving bus
        # are essentially instantaneous, so most of intact capability is
        # available right at t=0+. Use 0.8 (a 20 % sub-second dip) instead
        # of the conservative 0.5.
        gamma_immediate=0.8,
        # Recovery to steady-state cap is dominated by BowAz reallocation,
        # which only actually slews if its commanded direction changes.
        # 5 s is representative; 10 s was originally chosen as a conservative
        # azimuth-slew estimate.
        T_realloc=5.0,
    )

    # Match the simulator's post-failure window exactly.
    spec = ScenarioSpec(
        Hs=HS, Tp=TP, wave_dir_compass=WAVE_DIR_COMPASS,
        wind_speed=VW, wind_dir_compass=WAVE_DIR_COMPASS,
        current_speed=VC, current_dir_compass=WAVE_DIR_COMPASS,
        vessel_heading_compass=VESSEL_HEADING_COMPASS,
        failed_thruster_indices=CSOV_WCF_GROUPS["bus_port"],
        activate_sk_s=60.0,
        settle_s=500.0,
        post_failure_s=180.0,
        print_every_steps=1,            # 10 Hz output for the comparison
    )

    transient = wcfdi_transient(
        cfg=cfg, Vw_mean=VW, Hs=HS, Tp=TP, Vc=VC,
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
    # 2. Simulator ensemble
    # ----------------------------------------------------------------
    print(f"\n[sim] running {N_SEEDS}-seed ensemble (this is parallel) ...")
    t_sim_start = time.time()
    results = run_ensemble(spec, n_seeds=N_SEEDS, work_dir=work_dir,
                           tag="p7v1")
    sim_dt = time.time() - t_sim_start
    print(f"  ensemble ran in {sim_dt:.1f} s wall "
          f"({N_SEEDS} x {spec.total_seconds:.0f} sim-s -> "
          f"{N_SEEDS * spec.total_seconds / sim_dt:.0f}x realtime aggregate)")

    # ----------------------------------------------------------------
    # 3. Align and reduce simulator data
    # ----------------------------------------------------------------
    # Use the simulator's own body-frame deviation columns (SurgeDev,
    # SwayDev, PosDev) instead of NED (x, y) magnitude, because the
    # simulator's setpoint isn't exactly at NED origin (capture happens at
    # the position the vessel held during precondition, which can drift
    # by ~3 m). PosDev is body-frame |deviation from active setpoint| and
    # is what cqa's intact-prior + transient predict.
    n_min = min(r.n_rows for r in results)
    t_sim = results[0]["t"][:n_min]                          # 0 -> total_seconds, 0.1 s spacing
    surge_arr = np.array([r["SurgeDev"][:n_min] for r in results])  # (N, n_min) body-frame
    sway_arr = np.array([r["SwayDev"][:n_min] for r in results])    # (N, n_min) body-frame
    pos_arr = np.array([r["PosDev"][:n_min] for r in results])      # (N, n_min) hypot(SurgeDev, SwayDev)

    # Sanity check: the simulator's PosDev should equal hypot(SurgeDev, SwayDev).
    # If they disagree, something has changed in the column convention.
    pos_check = np.hypot(surge_arr, sway_arr)
    pos_dev_err = np.max(np.abs(pos_arr - pos_check))
    if pos_dev_err > 0.05:
        print(f"  [warn] PosDev != hypot(SurgeDev,SwayDev): max diff = {pos_dev_err:.3f} m")

    # Window masks.
    # Intact-stats window: only the *last 200 s* of the intact phase, so any
    # post-precondition approach transient (decay timescale ~ 1/(zeta*omega)
    # ~ 13 s in surge to 53 s in sway, several time constants until fully
    # settled) is excluded. With settle_s=500 and a 200 s sampling window
    # we leave 300 s of headroom for the slow settling to decay.
    INTACT_SAMPLE_S = 200.0
    intact_mask = (
        (t_sim >= spec.failure_time_s - INTACT_SAMPLE_S)
        & (t_sim < spec.failure_time_s - 1.0)
    )
    post_mask = t_sim >= spec.failure_time_s

    # Empirical running-max statistics over the intact window (per seed).
    intact_running_max = np.array([np.maximum.accumulate(pos_arr[k, intact_mask])
                                    for k in range(N_SEEDS)])
    intact_max_per_seed = intact_running_max[:, -1]   # one max per realisation
    intact_p50_emp = np.median(intact_max_per_seed)
    intact_p90_emp = np.quantile(intact_max_per_seed, 0.90)

    print(f"\n[compare] intact-stats over {intact_mask.sum() * SIM_DT:.0f} s "
          f"window x {N_SEEDS} seeds (using PosDev, body-frame deviation):")
    print(f"  cqa  P50 |pos| = {prior.pos_a_p50:.2f} m   "
          f"sim P50 = {intact_p50_emp:.2f} m   "
          f"diff = {intact_p50_emp - prior.pos_a_p50:+.2f} m")
    print(f"  cqa  P90 |pos| = {prior.pos_a_p90:.2f} m   "
          f"sim P90 = {intact_p90_emp:.2f} m   "
          f"diff = {intact_p90_emp - prior.pos_a_p90:+.2f} m")
    # Per-seed std of surge/sway in the late intact window (steady-state).
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

    # Post-failure traces, aligned so t' = 0 is the failure event.
    t_post = t_sim[post_mask] - spec.failure_time_s
    surge_post = surge_arr[:, post_mask]
    sway_post = sway_arr[:, post_mask]
    pos_post = pos_arr[:, post_mask]

    # Body-frame deviation at the moment of failure (subtract per-seed offset
    # so the cqa transient eta(0)=0 baseline matches the simulator's
    # zero-of-deviation baseline, which is the position right *before* failure).
    # cqa wcfdi_transient launches from x_mean(0) = 0 by construction.
    surge_post = surge_post - surge_post[:, :1]
    sway_post = sway_post - sway_post[:, :1]
    pos_post_dev = np.hypot(surge_post, sway_post)   # |post-failure body-frame deviation|

    # Empirical post-failure stats.
    surge_mean_emp = surge_post.mean(axis=0)
    sway_mean_emp = sway_post.mean(axis=0)
    surge_q_lo = np.quantile(surge_post, 0.25, axis=0)
    surge_q_hi = np.quantile(surge_post, 0.75, axis=0)
    sway_q_lo = np.quantile(sway_post, 0.25, axis=0)
    sway_q_hi = np.quantile(sway_post, 0.75, axis=0)

    # cqa transient (already in body-frame surge/sway).
    t_cqa = transient.t
    eta_surge = transient.eta_mean[:, 0]
    eta_sway = transient.eta_mean[:, 1]
    sigma_surge = transient.eta_std[:, 0]
    sigma_sway = transient.eta_std[:, 1]
    # Match k=0.674 (50% band, IQR-equivalent) to the simulator IQR.
    K = 0.674

    # ----------------------------------------------------------------
    # 4. Plots
    # ----------------------------------------------------------------
    print("\n[plot] rendering comparison figures ...")

    # --- Figure 1: intact-stats CDF ---
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
    ax.set_title(f"Intact-stats: cqa quantiles vs simulator running-max\n"
                 f"Vw={VW:.1f} m/s, Hs={HS:.2f} m, Tp={TP:.2f} s, beam-on, "
                 f"window = {intact_mask.sum() * SIM_DT:.0f} s")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    out1 = out_dir / "p7_validation_intact_cdf.png"
    fig.savefig(out1, dpi=130)
    plt.close(fig)
    print(f"  wrote {out1}")

    # --- Figure 2: transient mean + envelope, surge ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, comp_name, eta, sigma, sim_mean, sim_q_lo, sim_q_hi, sim_traces in [
        (axes[0], "surge", eta_surge, sigma_surge,
         surge_mean_emp, surge_q_lo, surge_q_hi, surge_post),
        (axes[1], "sway",  eta_sway,  sigma_sway,
         sway_mean_emp,  sway_q_lo,  sway_q_hi,  sway_post),
    ]:
        # Faint individual seed traces.
        for k in range(N_SEEDS):
            ax.plot(t_post, sim_traces[k], color="gray", alpha=0.15, lw=0.6)
        # Empirical mean and IQR.
        ax.plot(t_post, sim_mean, color="tab:blue", lw=2.0,
                label="sim ensemble mean")
        ax.fill_between(t_post, sim_q_lo, sim_q_hi, color="tab:blue", alpha=0.18,
                        label="sim IQR (25-75%)")
        # cqa mean and k*sigma envelope.
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
        f"Post-WCFDI transient -- cqa vs simulator (N={N_SEEDS} seeds)\n"
        f"Vw={VW:.1f} m/s, Hs={HS:.2f} m, Tp={TP:.2f} s, "
        f"theta_rel={rel_deg:+.0f} deg, Bus port lost\n"
        f"bistability_risk_score = {transient.info.get('bistability_risk_score', 0.0):.2f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out2 = out_dir / "p7_validation_transient.png"
    fig.savefig(out2, dpi=130)
    plt.close(fig)
    print(f"  wrote {out2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
