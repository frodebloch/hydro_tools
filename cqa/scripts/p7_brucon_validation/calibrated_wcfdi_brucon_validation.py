"""G2: brucon-ensemble validation of measurement-calibrated wcfdi_mc.

Goal
----
Show that injecting per-seed measured (σ_intact_lf_body, tau_env) into
``cqa.calibrated_wcfdi.wcfdi_mc_calibrated`` recovers the brucon WCF
transient that the operator-nominal raw ``wcfdi_mc`` predicts as
essentially flat (analysis.md §12.21.6.2).

Truth metric (locked, §12.21.6.2):

    Δsurge(t) = SurgeDev(t) − SurgeDev(t_WCF)        (body frame, m)
    Δsway(t)  = SwayDev(t)  − SwayDev(t_WCF)         (body frame, m)
    Δradial_peak_per_seed = max_t |(Δsurge, Δsway)|  (body frame, m)

Compared across:
  * brucon truth (ensemble of 30 seeds, ``pwo`` waves-only ensemble at
    Hs=4.20, Tp=10.22, beam-on, Bus port WCF at t_WCF=560 s).
  * cqa raw ``wcfdi_mc`` with operator nominal (Hs=4.20, Tp=10.22,
    Vw=0, Vc=0, β=π/2). This is built ONCE, the same for every seed.
  * cqa ``wcfdi_mc_calibrated`` with per-seed measured σ_lf_body
    (extracted from SurgeDev/SwayDev/HeadingDev over [360, 560) s)
    and per-seed measured tau_env (from estimator EstBias{Surge,Sway,Yaw}
    at t = t_WCF − 0.1 s, converted from kN/kNm → N/Nm).

Channels and units
------------------
* ``pwo_seedNNNN.out`` : t, heading [deg], SurgeDev [m, body frame],
  SwayDev [m, body frame], HeadingDev [deg, body frame].
* ``pwo_seedNNNN_estimator.out`` : col 0 t, col 19 EstBiasSurge [kN],
  col 20 EstBiasSway [kN], col 21 EstBiasYaw [kNm]. Estimator output
  is sampled at 10 Hz on the same clock as ``.out``.

Pre-WCF window for σ extraction: [360, 560) s. This is 200 s of settled
intact data after the 60 s SK activation + 300 s settling tail. With
T_var ~ 50-100 s for body-frame total position at this sea state
(§12.20), n_eff ≈ 2-4: the σ posterior is data-dominated but noisy
per seed -- exactly the operating regime the calibration is intended
to handle (σ̂ rather than σ_truth feeds the prediction).

For G2 we use the EMPIRICAL std of the [360, 560) s window directly
(not the InvGamma posterior) as the σ_intact_lf_body input. The
posterior plumbing was validated end-to-end in G1; using the empirical
std here removes one variability source from the G2 comparison so we
can attribute residuals to the wcfdi propagation pipeline rather than
the σ estimator.

Output
------
* console table per seed: tau_env_meas, σ_meas, brucon truth pos_peak,
  raw cqa pos_peak, calibrated cqa pos_peak.
* ensemble summary: median + P95 of pos_peak (brucon vs raw vs
  calibrated), median amplitude error and median time-of-peak error
  for ensemble-mean Δsurge / Δsway.
* PNG: 2x2 grid -- top row Δsurge(t) and Δsway(t) ensemble means
  with raw + calibrated overlays; bottom row pos_peak CDF (brucon vs
  raw vs calibrated) and per-seed scatter (calibrated cqa pos_peak vs
  brucon truth).

Companion test in ``tests/test_calibrated_wcfdi_brucon.py`` runs seed
1000 only and asserts the calibrated prediction beats the raw
prediction on the seed's |Δradial| peak.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))                       # for `harness`
sys.path.insert(0, str(THIS.parent.parent))         # for `cqa` package

from harness import parse_output, SIM_DT, CSOV_WCF_GROUPS  # noqa: E402

# Reuse setup_cqa from run_comparison so cqa controller bandwidth /
# damping / gangway joint exactly match the brucon-side run that
# produced the pwo ensemble.
from run_comparison import setup_cqa  # noqa: E402

# ---- ensemble + sea state (matches run_comparison_waves_only.py) ----
ENSEMBLE_DIR = THIS / "work"
TAG = "pwo"
N_SEEDS = 30
SEED_FIRST = 1000
HS = 4.19571865443425
TP = 10.22443464601827
THETA_REL = np.pi / 2          # β = 90° (beam-on)
VW_NOMINAL = 0.0               # waves-only ensemble
VC_NOMINAL = 0.0
T_WCF_S = 560.0                # activate_sk_s 60 + settle_s 500
T_POST_S = 180.0               # post_failure_s
SIGMA_WINDOW = (360.0, 560.0)  # 200 s pre-WCF window for σ extraction
DT_S = SIM_DT                  # 0.1 s

# ---- pdstrip RAO file ----
PDSTRIP_PATH = (
    "/home/blofro/src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"
)

# ---- MC sample size for cqa-side wcfdi_mc ----
N_MC_SAMPLES = 500

# ---- tau_lost pulse shape (G2 step 4 / sec.12.21.7) ----
# Per-seed measurement extracts:
#   tau_lost_pre_wcf  = peak deficit (Order - delivered) over [0, 30] s post-WCF.
#                       This is per-DOF, signed.
#   tau_lost_T_eff    = per-DOF equivalent square-pulse duration =
#                       (impulse over [0,30]s) / peak_deficit. Brucon ensemble
#                       gives ~8-9 s for sway, ~12 s for yaw (sec.12.21.7).
# We use a SQUARE pulse with the per-seed peak amplitude and a single
# scalar duration set to the across-seed, across-DOF median of T_eff.
# This trades per-DOF fidelity for the existing scalar-duration API in
# CalibratedContext; per-DOF duration would be a Phase 2 enhancement
# (would let yaw decay over 12 s while sway decays over 9 s).
TAU_LOST_PULSE_SHAPE = "square"


def extract_seed_inputs(seed: int):
    """Read brucon outputs and return per-seed measurement-derived inputs.

    Returns
    -------
    dict with keys:
        t_full, surge_full, sway_full, heading_full   -- entire .out time series
        sigma_lf_body                                  -- (3,) σ_x, σ_y, σ_ψ [m, m, rad]
        tau_env_meas                                   -- (3,) Fx, Fy, Mz [N, N, Nm]
        delta_surge_post                               -- t in [0, T_POST_S], SurgeDev - SurgeDev(t_WCF)
        delta_sway_post                                -- same for sway
        t_post                                         -- post-WCF time grid (t - t_WCF)
        delta_radial_peak                              -- max_t sqrt(Δsurge^2 + Δsway^2)
    """
    seed_dir = ENSEMBLE_DIR / f"{TAG}_seed{seed:04d}"
    out_path = seed_dir / f"{TAG}_seed{seed:04d}.out"
    est_path = seed_dir / f"{TAG}_seed{seed:04d}_estimator.out"
    if not out_path.exists() or not est_path.exists():
        return None
    res = parse_output(out_path)
    t = res.columns["t"]
    surge = res.columns["SurgeDev"]
    sway = res.columns["SwayDev"]
    heading_dev = res.columns["HeadingDev"]   # deg

    # Actual delivered body-frame thrust (post thruster dynamics)
    Tx = res.columns["Tx"]
    Ty = res.columns["Ty"]
    Tz = res.columns["Tz"]

    # tau_lost(t=0+) = (Tx,Ty,Tz)[t_WCF^-] - (Tx,Ty,Tz)[t_WCF^+]
    # i.e. the immediate drop in delivered force at the WCF instant.
    # Sign convention: tau_lost is what the surviving thrusters cannot
    # immediately deliver; it subtracts from the effective force on the
    # vessel hull (cqa.transient._augmented_rhs_post does this).
    i_minus = int(np.searchsorted(t, T_WCF_S - 0.05) - 1)
    i_plus = int(np.searchsorted(t, T_WCF_S + 0.15) - 1)
    tau_delivered_pre = np.array([Tx[i_minus], Ty[i_minus], Tz[i_minus]])
    tau_delivered_post = np.array([Tx[i_plus], Ty[i_plus], Tz[i_plus]])
    # Per-seed effective deficit measured from the actual Order vs delivered:
    # peak deficit (within first 30 s) and impulse over [0, 30 s]; effective
    # duration T_eff = impulse / peak (square-pulse-equivalent).
    OrderSway = res.columns["OrderTauSway"]
    OrderYaw = res.columns["OrderTauYaw"]
    OrderSurge = res.columns["OrderTauSurge"]
    mask_def = (t >= T_WCF_S) & (t <= T_WCF_S + 30.0)
    dt_def = float(np.median(np.diff(t[mask_def])))
    def_surge = OrderSurge[mask_def] - Tx[mask_def]
    def_sway  = OrderSway[mask_def]  - Ty[mask_def]
    def_yaw   = OrderYaw[mask_def]   - Tz[mask_def]
    peak_def = np.array([
        def_surge[np.argmax(np.abs(def_surge))],
        def_sway[np.argmax(np.abs(def_sway))],
        def_yaw[np.argmax(np.abs(def_yaw))],
    ])
    impulse_def = np.array([
        def_surge.sum() * dt_def,
        def_sway.sum() * dt_def,
        def_yaw.sum() * dt_def,
    ])
    # SI conversion (kN, kNm) -> (N, Nm)
    peak_def_SI = 1e3 * peak_def
    impulse_def_SI = 1e3 * impulse_def
    # T_eff per DOF; guard against zero peak.
    T_eff = np.where(np.abs(peak_def_SI) > 1.0, impulse_def_SI / peak_def_SI, 0.0)

    # σ extraction window
    mask_sigma = (t >= SIGMA_WINDOW[0]) & (t < SIGMA_WINDOW[1])
    if not mask_sigma.any():
        raise RuntimeError(f"seed {seed}: σ window {SIGMA_WINDOW} contains no samples")
    sx = float((surge[mask_sigma] - surge[mask_sigma].mean()).std())
    sy = float((sway[mask_sigma] - sway[mask_sigma].mean()).std())
    spsi = float(np.deg2rad(heading_dev[mask_sigma] - heading_dev[mask_sigma].mean()).std())
    sigma_lf_body = np.array([sx, sy, spsi])

    # tau_env from estimator at t = t_WCF - DT (last intact sample)
    est = np.loadtxt(est_path, skiprows=1)
    t_est = est[:, 0]
    bias_surge_kN = est[:, 19]
    bias_sway_kN = est[:, 20]
    bias_yaw_kNm = est[:, 21]
    # last sample with t < T_WCF_S
    idx = int(np.searchsorted(t_est, T_WCF_S - 0.5 * DT_S) - 1)
    if idx < 0:
        raise RuntimeError(f"seed {seed}: no estimator samples before t_WCF")
    tau_env_meas = np.array([
        1e3 * bias_surge_kN[idx],
        1e3 * bias_sway_kN[idx],
        1e3 * bias_yaw_kNm[idx],
    ])

    # Post-WCF Δsurge / Δsway since t_WCF (per-seed offset removed)
    mask_post = t >= T_WCF_S
    t_post = t[mask_post] - T_WCF_S
    delta_surge_post = surge[mask_post] - surge[mask_post][0]
    delta_sway_post = sway[mask_post] - sway[mask_post][0]
    delta_radial_peak = float(np.max(np.sqrt(delta_surge_post**2 + delta_sway_post**2)))

    return {
        "seed": seed,
        "t": t,
        "surge": surge,
        "sway": sway,
        "heading_dev": heading_dev,
        "sigma_lf_body": sigma_lf_body,
        "tau_env_meas": tau_env_meas,
        "tau_lost_pre_wcf": peak_def_SI,         # peak-deficit amplitude (square pulse)
        "tau_lost_T_eff": T_eff,                 # per-DOF equivalent duration
        "tau_lost_init_jump": 1e3 * (tau_delivered_pre - tau_delivered_post),  # diagnostics
        "t_post": t_post,
        "delta_surge_post": delta_surge_post,
        "delta_sway_post": delta_sway_post,
        "delta_radial_peak": delta_radial_peak,
    }


def main() -> None:
    print("=" * 78)
    print("G2: calibrated_wcfdi brucon-ensemble validation")
    print(f"  ensemble: {ENSEMBLE_DIR}/{TAG}_seed{SEED_FIRST:04d}..")
    print(f"  N seeds : {N_SEEDS}")
    print(f"  Hs={HS:.3f} m, Tp={TP:.3f} s, beam-on, Vw=Vc=0, Bus port WCF")
    print(f"  t_WCF   = {T_WCF_S:.0f} s, post window = {T_POST_S:.0f} s")
    print(f"  σ window= {SIGMA_WINDOW} s")
    print("=" * 78)

    if not ENSEMBLE_DIR.exists():
        sys.exit(f"ENSEMBLE NOT FOUND at {ENSEMBLE_DIR}")

    # --- per-seed extraction ---
    print("\n--- extracting per-seed measurements ---")
    seeds = []
    for k in range(N_SEEDS):
        seed = SEED_FIRST + k
        try:
            r = extract_seed_inputs(seed)
        except Exception as e:
            print(f"  seed {seed}: FAILED ({e})")
            continue
        if r is None:
            print(f"  seed {seed}: missing files")
            continue
        seeds.append(r)
        if k < 3 or k == N_SEEDS - 1:
            print(
                f"  seed {seed}: σ_y={r['sigma_lf_body'][1]:.3f} m  "
                f"τ_env=({r['tau_env_meas'][0]/1e3:+.0f}, "
                f"{r['tau_env_meas'][1]/1e3:+.0f}, "
                f"{r['tau_env_meas'][2]/1e3:+.0f}) kN/kNm  "
                f"τ_lost=({r['tau_lost_pre_wcf'][0]/1e3:+.0f}, "
                f"{r['tau_lost_pre_wcf'][1]/1e3:+.0f}, "
                f"{r['tau_lost_pre_wcf'][2]/1e3:+.0f}) kN/kNm  "
                f"|Δr|_peak={r['delta_radial_peak']:.2f} m"
            )
    if not seeds:
        sys.exit("NO SEEDS LOADED")
    print(f"  loaded {len(seeds)} seeds")

    # Compute median T_eff across seeds and across (sway, yaw) DOFs --
    # a single scalar duration for all seeds. Surge T_eff is dominated
    # by noise (peak deficit is small, so T_eff is poorly conditioned).
    T_eff_all = np.array([r["tau_lost_T_eff"][1:3] for r in seeds]).flatten()
    T_eff_all = T_eff_all[np.isfinite(T_eff_all) & (T_eff_all > 0)]
    TAU_LOST_DURATION_S = float(np.median(T_eff_all))
    print(f"  per-seed T_eff (sway+yaw) ensemble median = {TAU_LOST_DURATION_S:.1f} s "
          f"(P25={np.quantile(T_eff_all, 0.25):.1f}, "
          f"P75={np.quantile(T_eff_all, 0.75):.1f})")

    # --- cqa-side imports (defer until after parsing so script fails
    #     fast on missing brucon data) ---
    from cqa.config import csov_default_config  # noqa: F401
    from cqa.transient import WcfdiScenario
    from cqa.wcfdi_mc import wcfdi_mc
    from cqa.calibrated_wcfdi import build_calibrated_context, wcfdi_mc_calibrated
    from cqa.rao import load_pdstrip_rao

    cfg, joint = setup_cqa()

    if not Path(PDSTRIP_PATH).exists():
        sys.exit(f"PDSTRIP RAO not found at {PDSTRIP_PATH}")
    print(f"\n--- loading pdstrip RAO+QTF: {PDSTRIP_PATH} ---")
    rao = load_pdstrip_rao(PDSTRIP_PATH)

    scenario = WcfdiScenario(
        alpha=(0.5, 0.7, 0.5),
        gamma_immediate=0.8,
        T_realloc=5.0,
    )

    n_t = int(T_POST_S / DT_S) + 1   # 1801

    # --- raw wcfdi_mc (operator-nominal, built ONCE) ---
    print(f"\n--- raw wcfdi_mc (operator-nominal, n={N_MC_SAMPLES}, n_t={n_t}) ---")
    t0 = time.time()
    raw = wcfdi_mc(
        cfg=cfg, joint=joint, scenario=scenario,
        Vw_mean=VW_NOMINAL, Hs=HS, Tp=TP, Vc=VC_NOMINAL, theta_rel=THETA_REL,
        sigma_Vc=0.1, tau_Vc=600.0,
        rao_table=rao,
        n_samples=N_MC_SAMPLES, t_end=T_POST_S, n_t=n_t,
        rng_seed=12345, sample_mode="eta_nu",
    )
    print(f"  ran in {time.time()-t0:.1f} s; pos_peak P50={np.median(raw.pos_peak):.2f} m, "
          f"P95={np.quantile(raw.pos_peak, 0.95):.2f} m")

    # The raw cqa transient "ensemble mean Δ" since t=0 is just
    # eta_mean trajectory (the linearised mean, which by construction
    # starts at the intact mean -- so Δ is 0 at t=0 for a stationary
    # intact start). Use the linearised eta_mean.
    raw_dx_mean = raw.info["P6_intact"]   # not used directly; we use lin via re-derivation
    # We need the linearised eta_mean which wcfdi_mc returns inside
    # L_mean_linear / dL_lin_mean only as the gangway projection. For
    # the position-component overlay we should call wcfdi_transient.
    from cqa.transient import wcfdi_transient
    raw_lin = wcfdi_transient(
        cfg=cfg, scenario=scenario,
        Vw_mean=VW_NOMINAL, Hs=HS, Tp=TP, Vc=VC_NOMINAL, theta_rel=THETA_REL,
        sigma_Vc=0.1, tau_Vc=600.0, rao_table=rao,
        t_end=T_POST_S, n_t=n_t,
    )
    # Δsurge(t) for cqa = eta_mean(t) - eta_mean(0)  (start at intact mean)
    raw_dsurge = raw_lin.eta_mean[:, 0] - raw_lin.eta_mean[0, 0]
    raw_dsway = raw_lin.eta_mean[:, 1] - raw_lin.eta_mean[0, 1]
    t_cqa = raw_lin.t

    # --- per-seed calibrated wcfdi_mc ---
    print(f"\n--- calibrated wcfdi_mc per seed (n={N_MC_SAMPLES} each, "
          f"tau_lost shape={TAU_LOST_PULSE_SHAPE}, T={TAU_LOST_DURATION_S} s) ---")
    print(f"  {'seed':>5} {'σy':>5} {'τFy':>6} {'τMz':>7} "
          f"{'lostFy':>7} {'lostMz':>7} | "
          f"{'truth_pk':>8} {'raw_pk':>7} {'cal_pk':>7}")
    cal_dsurge_per_seed = []
    cal_dsway_per_seed = []
    cal_pos_peak_p50 = []
    raw_pos_peak_p50_seed = float(np.median(raw.pos_peak))
    rows = []
    t0 = time.time()
    for r in seeds:
        ctx = build_calibrated_context(
            cfg,
            sigma_measured_lf_body=r["sigma_lf_body"],
            tau_env_measured=r["tau_env_meas"],
            Vw_mean=VW_NOMINAL, Hs=HS, Tp=TP, Vc=VC_NOMINAL, theta_rel=THETA_REL,
            sigma_Vc=0.1, tau_Vc=600.0, rao_table=rao,
            tau_lost_pre_wcf=r["tau_lost_pre_wcf"],
            tau_lost_pulse_shape=TAU_LOST_PULSE_SHAPE,
            tau_lost_duration_s=TAU_LOST_DURATION_S,
        )
        cal = wcfdi_mc_calibrated(
            cfg=cfg, scenario=scenario, joint=joint, ctx=ctx,
            n_samples=N_MC_SAMPLES, t_end=T_POST_S, n_t=n_t,
            rng_seed=10_000 + r["seed"], sample_mode="eta_nu",
        )
        cal_lin = cal  # linearised baseline is in cal.L_mean_linear via gangway, but
        # we want eta_mean. Recompute via wcfdi_transient_calibrated.
        from cqa.calibrated_wcfdi import wcfdi_transient_calibrated
        cal_tr = wcfdi_transient_calibrated(cfg, scenario, ctx, t_end=T_POST_S, n_t=n_t)
        dsurge = cal_tr.eta_mean[:, 0] - cal_tr.eta_mean[0, 0]
        dsway = cal_tr.eta_mean[:, 1] - cal_tr.eta_mean[0, 1]
        cal_dsurge_per_seed.append(dsurge)
        cal_dsway_per_seed.append(dsway)
        cal_p50 = float(np.median(cal.pos_peak))
        cal_pos_peak_p50.append(cal_p50)
        truth_pk = r["delta_radial_peak"]
        print(
            f"  {r['seed']:>5} {r['sigma_lf_body'][1]:>5.2f} "
            f"{r['tau_env_meas'][1]/1e3:>+6.0f} "
            f"{r['tau_env_meas'][2]/1e3:>+7.0f} "
            f"{r['tau_lost_pre_wcf'][1]/1e3:>+7.0f} "
            f"{r['tau_lost_pre_wcf'][2]/1e3:>+7.0f} | "
            f"{truth_pk:>8.2f} {raw_pos_peak_p50_seed:>7.2f} {cal_p50:>7.2f}"
        )
        rows.append({
            "seed": r["seed"],
            "truth_peak": truth_pk,
            "raw_peak_p50": raw_pos_peak_p50_seed,
            "cal_peak_p50": cal_p50,
            "cal_peak_p95": float(np.quantile(cal.pos_peak, 0.95)),
            "sigma_lf_body": r["sigma_lf_body"],
            "tau_env_meas": r["tau_env_meas"],
        })
    print(f"  ({time.time()-t0:.1f} s)")

    # --- aggregate ---
    cal_dsurge_arr = np.stack(cal_dsurge_per_seed)   # (n_seeds, n_t)
    cal_dsway_arr = np.stack(cal_dsway_per_seed)
    cal_dsurge_mean = cal_dsurge_arr.mean(axis=0)
    cal_dsway_mean = cal_dsway_arr.mean(axis=0)

    # Brucon ensemble means (interpolate to cqa grid; should already match if dt=0.1)
    truth_dsurge = np.stack([r["delta_surge_post"] for r in seeds]).mean(axis=0)
    truth_dsway = np.stack([r["delta_sway_post"] for r in seeds]).mean(axis=0)
    t_truth = seeds[0]["t_post"]

    # peak amplitude metrics (signed, on ensemble mean)
    def peak_metrics(t, dx):
        idx = int(np.argmax(np.abs(dx)))
        return float(dx[idx]), float(t[idx])

    truth_dsu_peak, truth_dsu_tpk = peak_metrics(t_truth, truth_dsurge)
    truth_dsw_peak, truth_dsw_tpk = peak_metrics(t_truth, truth_dsway)
    raw_dsu_peak, raw_dsu_tpk = peak_metrics(t_cqa, raw_dsurge)
    raw_dsw_peak, raw_dsw_tpk = peak_metrics(t_cqa, raw_dsway)
    cal_dsu_peak, cal_dsu_tpk = peak_metrics(t_cqa, cal_dsurge_mean)
    cal_dsw_peak, cal_dsw_tpk = peak_metrics(t_cqa, cal_dsway_mean)

    truth_peaks = np.array([r["truth_peak"] for r in rows])
    raw_peak_p50 = raw_pos_peak_p50_seed
    raw_peak_p95 = float(np.quantile(raw.pos_peak, 0.95))
    cal_peaks_p50 = np.array([r["cal_peak_p50"] for r in rows])
    cal_peaks_p95 = np.array([r["cal_peak_p95"] for r in rows])

    print("\n--- ensemble-mean Δ (signed peak amplitude / time) ---")
    print(f"  Δsurge   truth: {truth_dsu_peak:+.2f} m @ t={truth_dsu_tpk:.1f} s")
    print(f"           raw  : {raw_dsu_peak:+.2f} m @ t={raw_dsu_tpk:.1f} s")
    print(f"           cal  : {cal_dsu_peak:+.2f} m @ t={cal_dsu_tpk:.1f} s")
    print(f"  Δsway    truth: {truth_dsw_peak:+.2f} m @ t={truth_dsw_tpk:.1f} s")
    print(f"           raw  : {raw_dsw_peak:+.2f} m @ t={raw_dsw_tpk:.1f} s")
    print(f"           cal  : {cal_dsw_peak:+.2f} m @ t={cal_dsw_tpk:.1f} s")

    print("\n--- per-seed |Δradial|_peak distribution ---")
    print(f"  truth   median = {np.median(truth_peaks):.2f} m, "
          f"P95 = {np.quantile(truth_peaks, 0.95):.2f} m")
    print(f"  raw     median = {raw_peak_p50:.2f} m, P95 = {raw_peak_p95:.2f} m  "
          f"(constant, single ensemble)")
    print(f"  cal P50 median-across-seeds = {np.median(cal_peaks_p50):.2f} m, "
          f"P95-across-seeds = {np.quantile(cal_peaks_p50, 0.95):.2f} m")
    print(f"  cal P95 median-across-seeds = {np.median(cal_peaks_p95):.2f} m, "
          f"P95-across-seeds = {np.quantile(cal_peaks_p95, 0.95):.2f} m")

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # Top-left: ensemble-mean Δsurge(t)
        ax = axes[0, 0]
        for r in seeds:
            ax.plot(r["t_post"], r["delta_surge_post"], color="gray", alpha=0.15, lw=0.6)
        ax.plot(t_truth, truth_dsurge, color="black", lw=2.0,
                label=f"brucon ensemble mean (N={len(seeds)})")
        ax.plot(t_cqa, raw_dsurge, color="tab:red", lw=2.0, ls="--",
                label=f"raw wcfdi_mc (peak {raw_dsu_peak:+.2f} m)")
        ax.plot(t_cqa, cal_dsurge_mean, color="tab:blue", lw=2.0,
                label=f"calibrated wcfdi_mc (peak {cal_dsu_peak:+.2f} m)")
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_xlabel("time since WCF [s]")
        ax.set_ylabel("Δsurge [m]")
        ax.set_title("Ensemble-mean Δsurge since WCF")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

        # Top-right: ensemble-mean Δsway(t)
        ax = axes[0, 1]
        for r in seeds:
            ax.plot(r["t_post"], r["delta_sway_post"], color="gray", alpha=0.15, lw=0.6)
        ax.plot(t_truth, truth_dsway, color="black", lw=2.0,
                label=f"brucon ensemble mean (N={len(seeds)})")
        ax.plot(t_cqa, raw_dsway, color="tab:red", lw=2.0, ls="--",
                label=f"raw wcfdi_mc (peak {raw_dsw_peak:+.2f} m)")
        ax.plot(t_cqa, cal_dsway_mean, color="tab:blue", lw=2.0,
                label=f"calibrated wcfdi_mc (peak {cal_dsw_peak:+.2f} m)")
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_xlabel("time since WCF [s]")
        ax.set_ylabel("Δsway [m]")
        ax.set_title("Ensemble-mean Δsway since WCF")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

        # Bottom-left: pos_peak CDF
        ax = axes[1, 0]
        sorted_truth = np.sort(truth_peaks)
        cdf = np.arange(1, len(sorted_truth) + 1) / (len(sorted_truth) + 1)
        ax.plot(sorted_truth, cdf, "o-", color="black", lw=1.5, ms=5,
                label=f"brucon truth (N={len(seeds)})")
        sorted_raw = np.sort(raw.pos_peak)
        cdf_raw = np.arange(1, len(sorted_raw) + 1) / (len(sorted_raw) + 1)
        ax.plot(sorted_raw, cdf_raw, color="tab:red", ls="--", lw=1.5,
                label=f"raw wcfdi_mc (n={len(sorted_raw)})")
        sorted_cal_p50 = np.sort(cal_peaks_p50)
        cdf_cal = np.arange(1, len(sorted_cal_p50) + 1) / (len(sorted_cal_p50) + 1)
        ax.plot(sorted_cal_p50, cdf_cal, "s-", color="tab:blue", lw=1.5, ms=5,
                label="calibrated P50 (one per seed)")
        ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
        ax.axhline(0.95, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("|Δradial|_peak [m]")
        ax.set_ylabel("empirical CDF")
        ax.set_title("Per-seed peak post-WCF excursion")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

        # Bottom-right: per-seed scatter (calibrated vs truth)
        ax = axes[1, 1]
        ax.scatter(truth_peaks, cal_peaks_p50, s=40, color="tab:blue", alpha=0.7,
                   label="calibrated P50")
        ax.scatter(truth_peaks, cal_peaks_p95, s=40, marker="s",
                   color="tab:cyan", alpha=0.7,
                   label="calibrated P95")
        ax.axhline(raw_peak_p50, color="tab:red", ls="--",
                   label=f"raw P50 = {raw_peak_p50:.2f} m")
        ax.axhline(raw_peak_p95, color="tab:red", ls=":",
                   label=f"raw P95 = {raw_peak_p95:.2f} m")
        lim = [0, max(truth_peaks.max(), cal_peaks_p95.max(), raw_peak_p95) * 1.1]
        ax.plot(lim, lim, "k--", alpha=0.4, lw=1.0, label="y=x")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("brucon truth |Δradial|_peak [m]")
        ax.set_ylabel("cqa prediction [m]")
        ax.set_title("Per-seed prediction vs truth")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

        fig.suptitle(
            f"G2: calibrated wcfdi_mc vs brucon WCF transient (Hs={HS:.2f} m, "
            f"Tp={TP:.2f} s, β=90°, Vw=Vc=0, Bus port WCF, N={len(seeds)} seeds)",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_png = THIS / "calibrated_wcfdi_brucon_validation.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"\nSaved: {out_png}")
    except Exception as e:
        print(f"\nplot failed: {e}")


if __name__ == "__main__":
    main()
