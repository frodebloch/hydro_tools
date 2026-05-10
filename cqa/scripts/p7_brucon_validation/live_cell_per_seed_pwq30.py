"""Per-seed validation of the LIVE operational CQA cell against brucon truth.

Two-pipeline reframe (recap)
----------------------------
The CQA system has two distinct pipelines:

* Forecast pipeline (``cqa.decision_matrix``): inputs are forecast
  (Vw, Hs, Tp, Vc, theta) per cell. Used for operation planning.
* Live operational pipeline (``cqa.live_decision``): inputs are the
  observer state, Bayesian sigma posteriors, and scenario knobs --
  no forecast values at runtime. Used for here-and-now operability
  monitoring.

This script validates the live pipeline at pwq30 (Hs=4.196, Tp=10.224,
theta_env=210 deg, heading=180 deg, theta_rel=+30 deg). For each of
30 brucon seeds:

  1. Read the pre-WCF window of the brucon log.
  2. Snapshot the live observer state at t_eval = T_WCF - 5 s
     (eta_hat, nu_hat, b_hat, eta_wave, heading) -- the same vector
     a real-time controller could publish each second.
  3. Build per-axis BayesianSigmaEstimator posteriors on the same
     pre-WCF window, for both LF (eta_hat) and WF (eta_wave) channels;
     compose into LiveSigmaPosterior.
  4. Call ``evaluate_decision_cell_live`` -> predicted WCFDI envelope
     pos_envelope_t = |eta_hat_LF + delta_eta_mean(t)| + k*sigma_R.
  5. Compute the realised post-WCF radial trajectory from brucon truth
     (body-frame, baseline-corrected over the pre-WCF window).
  6. Overlay predicted envelope vs realised |R(t)| per seed and
     ensemble-mean.

The KEY HYPOTHESIS the live pipeline tests: per-seed agreement should
be much tighter than the forecast pipeline's ensemble-mean agreement
(which is 1.6x brucon at this cell), because each seed conditions on
its actual realised mean force b_hat and its actual realised LF/WF
sigmas, instead of an ensemble-average parametric drift force from a
QTF and a parametric wave PSD.

Run with::

    PYTHONPATH=. .venv/bin/python \\
        scripts/p7_brucon_validation/live_cell_per_seed_pwq30.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))

from cqa.config import csov_default_config
from cqa.gangway import GangwayJointState, telescope_sensitivity_6dof
from cqa.online_estimator import (
    BayesianSigmaEstimator,
    combine_radial_posterior,
    compose_validity_badge,
)
from cqa.live_decision import (
    LiveObserverState,
    LiveSigmaPosterior,
    evaluate_decision_cell_live,
)
from cqa.transient import WcfdiScenario


# --------------------------- scenario knobs ------------------------------
WORK_ROOT = THIS / "work"
# Cell parameters (defaults match pwq30; override on CLI for other cells).
TAG = "pwq30"
SEEDS = list(range(1000, 1030))

T_WCF = 560.0
T_EVAL = T_WCF - 5.0          # snapshot time for the live cell
WIN_S = 60.0                  # Bayesian sigma window length
WIN_END = T_WCF - 1.0         # end of the live-window (just before the WCF)
WIN_START = WIN_END - WIN_S


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default=TAG,
                   help=f"Cell tag, also work-dir prefix (default: {TAG})")
    p.add_argument("--t-wcf", type=float, default=T_WCF,
                   help=f"WCF injection time in seconds (default: {T_WCF})")
    p.add_argument("--seeds", default=f"{SEEDS[0]}-{SEEDS[-1]+1}",
                   help="Seed range as 'lo-hi' (Python-style half-open) "
                        f"(default: {SEEDS[0]}-{SEEDS[-1]+1})")
    return p.parse_args()


def _apply_args(args):
    """Mutate module-level cell parameters from parsed args."""
    global TAG, T_WCF, T_EVAL, WIN_END, WIN_START, SEEDS, CALIB_NPZ
    TAG = args.tag
    T_WCF = float(args.t_wcf)
    T_EVAL = T_WCF - 5.0
    WIN_END = T_WCF - 1.0
    WIN_START = WIN_END - WIN_S
    lo, hi = args.seeds.split("-")
    SEEDS = list(range(int(lo), int(hi)))
    CALIB_NPZ = THIS / f"scenario_{TAG}_calibration.npz"

# Heuristic decorrelation times (s):
#   LF surge/sway: ~ 1 / omega_pid ~ 12-17 s -> use 15 s
#   LF yaw:        ~ 1 / 0.12      ~ 8 s     -> use 10 s
#   WF axes:       ~ Tp / 2         ~ 5 s    -> use 5 s
T_DECORR_LF = 15.0
T_DECORR_LF_YAW = 10.0
T_DECORR_WF = 5.0

# Prior sigma^2 for InvGamma (very weak: posterior dominated by data after
# a few minutes of warm window).
PRIOR_SIGMA_LF = 0.5          # m       (LF position prior std)
PRIOR_SIGMA_LF_YAW = 0.01     # rad
PRIOR_SIGMA_WF = 0.5          # m
PRIOR_SIGMA_WF_YAW = 0.01     # rad
# Roll/pitch/heave priors for the (always-WF) ship-motion channels. Brucon
# DP CSOV runs have effectively zero LF roll/pitch (no static heel under
# DP control on a symmetric vessel; mean trim is set by config and is
# subtracted as the windowed mean here), so the brucon Roll/Pitch/Heave
# channels can be treated as pure WF samples for the purpose of feeding
# the WF sigma posteriors. Roll/pitch in radians (brucon exports them in
# degrees -- converted before being pushed to the estimator).
PRIOR_SIGMA_WF_RP = 0.0175    # rad     (~1 deg roll/pitch prior)
PRIOR_SIGMA_WF_HEAVE = 0.3    # m
PRIOR_N0 = 2.0                # weak

# WCFDI scenario (matches forecast-pipeline default at this cell).
SCENARIO = WcfdiScenario(
    alpha=(2.0 / 3.0,) * 3,
    gamma_immediate=0.5,
    T_realloc=10.0,
)

# Live cell forward horizon.
T_END_WCFDI = 120.0
N_T = 241

# ---- Calibration artefact (offline-precomputed scenario for this cell) ----
# Produced by scripts/p7_brucon_validation/peak_R_b_hat_sigma_pwq30.py.
# Provides:
#   - delta_eta_mean(t): ensemble mean of per-seed delta_eta_LF trajectories
#     (each seed driven by tau_pre := -b_hat snapshot at t = T_WCF - 5)
#   - sigma_R_b_hat_m: std of peak |delta_eta_LF| from MC propagation of
#     the b_hat snapshot noise through cqa-27 (~0.07 m for pwq30)
CALIB_NPZ = THIS / "scenario_pwq30_calibration.npz"

# Plot horizon for the realised trajectory.
T_PLOT_PRE = 30.0             # s before WCF in plots
T_PLOT_POST = 120.0           # s after WCF


# --------------------------- IO helpers ---------------------------------

def _load_tsv(path: Path) -> dict[str, np.ndarray]:
    with open(path) as f:
        hdr = f.readline().strip().split("\t")
    data = np.loadtxt(path, skiprows=1, delimiter="\t")
    return {h: data[:, i] for i, h in enumerate(hdr)}


def load_seed(seed: int):
    """Return the per-seed live-cell payload + realised post-WCF trajectory.

    Returns ``None`` if the seed directory is missing or the log is
    truncated before T_WCF + 30 s.
    """
    seed_dir = WORK_ROOT / f"{TAG}_seed{seed:04d}"
    if not seed_dir.exists():
        return None

    main_p = next((p for p in seed_dir.glob("*.out") if "estimator" not in p.name), None)
    est_p = seed_dir / f"{TAG}_seed{seed:04d}_estimator.out"
    if main_p is None or not est_p.exists():
        return None

    M = _load_tsv(main_p)
    E = _load_tsv(est_p)
    t_main = M["t"]
    t_est = E["Time"]
    if t_main[-1] < T_WCF + 30.0:
        return None

    # Pre-WCF window samples (LF on main, WF on main, bias on estimator).
    win_m = (t_main >= WIN_START) & (t_main <= WIN_END)
    win_e = (t_est >= WIN_START) & (t_est <= WIN_END)

    # ----- snapshot at t_eval -----
    i_eval_m = int(np.argmin(np.abs(t_main - T_EVAL)))
    i_eval_e = int(np.argmin(np.abs(t_est - T_EVAL)))

    eta_hat = np.array([
        M["SurgeDev"][i_eval_m],
        M["SwayDev"][i_eval_m],
        np.deg2rad(M["HeadingDev"][i_eval_m]),  # brucon exports HeadingDev
                                                # in DEGREES (matches all
                                                # other angle channels in
                                                # the .out file: heading,
                                                # RateOfTurn, headingHf).
                                                # The earlier "rad,
                                                # verified" comment was
                                                # incorrect -- it held
                                                # only by coincidence on
                                                # bf4 cells where the
                                                # absolute deviation in
                                                # degrees is small enough
                                                # to look like radians.
                                                # Fixed when adding the
                                                # gangway bar surfaced
                                                # the unit error via the
                                                # c3[2]=-9 m/rad lever
                                                # arm.
    ])
    # RateOfTurn is logged in deg/min (apps/dp/dp_cms_export.cpp).
    nu_hat = np.array([
        M["SurgeSpeed"][i_eval_m],
        M["SwaySpeed"][i_eval_m],
        np.deg2rad(M["RateOfTurn"][i_eval_m]) / 60.0,
    ])
    # Faithful brucon NPO bias output (kN, kN, kNm) -> N, N, Nm.
    b_hat = 1e3 * np.array([
        E["EstBiasSurge"][i_eval_e],
        E["EstBiasSway"][i_eval_e],
        E["EstBiasYaw"][i_eval_e],
    ])
    eta_wave = np.array([
        M["xHf"][i_eval_m],
        M["yHf"][i_eval_m],
        np.deg2rad(M["headingHf"][i_eval_m]),  # DEG -> rad (see HeadingDev note above)
    ])
    heading_compass = np.deg2rad(M["heading"][i_eval_m])

    # Mean Tp from EstWavePeriod (estimator log) over the window.
    Tp_obs = float(np.mean(np.concatenate([
        E["EstWavePeriodSurge"][win_e],
        E["EstWavePeriodSway"][win_e],
    ])))
    if not np.isfinite(Tp_obs) or Tp_obs <= 1.0 or Tp_obs > 25.0:
        Tp_obs = 10.0

    # ----- pre-WCF samples for the Bayesian estimators -----
    # Subtract windowed mean to mimic an offset-free DP setpoint deviation:
    # the BayesianSigmaEstimator in the live pipeline operates on the
    # observer's estimate of the residual *about the setpoint*, so a
    # constant LF mean offset (= setpoint error) belongs to eta_hat, not
    # to the sigma channel. Subtracting the windowed mean here gives the
    # estimator zero-mean fluctuations to digest.
    samples_lf_x = M["SurgeDev"][win_m]
    samples_lf_y = M["SwayDev"][win_m]
    samples_lf_yaw = np.deg2rad(M["HeadingDev"][win_m])  # DEG -> rad
    samples_wf_x = M["xHf"][win_m]
    samples_wf_y = M["yHf"][win_m]
    samples_wf_yaw = np.deg2rad(M["headingHf"][win_m])   # DEG -> rad
    # Roll/Pitch/Heave: brucon main-log channels are TOTAL motion
    # (no LF/WF split exported -- there is no RollHf/PitchHf/HeaveHf
    # in the log header). On a DP CSOV the LF component is
    # effectively zero (no static heel from DP control on a symmetric
    # vessel; mean trim is set by config and is removed as the
    # windowed mean below), so the demeaned series is a clean WF
    # sample. Roll/Pitch are exported in DEGREES (units convention
    # discussed for HeadingDev above); heave in metres.
    samples_wf_heave = M["Heave"][win_m]
    samples_wf_roll = np.deg2rad(M["Roll"][win_m])
    samples_wf_pitch = np.deg2rad(M["Pitch"][win_m])

    samples_lf_x = samples_lf_x - samples_lf_x.mean()
    samples_lf_y = samples_lf_y - samples_lf_y.mean()
    samples_lf_yaw = samples_lf_yaw - samples_lf_yaw.mean()
    samples_wf_x = samples_wf_x - samples_wf_x.mean()
    samples_wf_y = samples_wf_y - samples_wf_y.mean()
    samples_wf_yaw = samples_wf_yaw - samples_wf_yaw.mean()
    samples_wf_heave = samples_wf_heave - samples_wf_heave.mean()
    samples_wf_roll = samples_wf_roll - samples_wf_roll.mean()
    samples_wf_pitch = samples_wf_pitch - samples_wf_pitch.mean()

    # ----- realised post-WCF radial trajectory (body-frame) -----
    # Truth body-frame deviations from the pre-WCF mean.
    h = np.deg2rad(M["heading"])
    surge_b = np.cos(h) * M["x"] + np.sin(h) * M["y"]
    sway_b = -np.sin(h) * M["x"] + np.cos(h) * M["y"]
    surge_b -= surge_b[win_m].mean()
    sway_b -= sway_b[win_m].mean()

    # Resample post-WCF to a common grid relative to t_eval.
    t_pred_grid = np.linspace(0.0, T_END_WCFDI, N_T)        # 0 = t_eval
    t_plot_grid = np.arange(-T_PLOT_PRE, T_PLOT_POST + 0.05, 0.1)
    truth_dx = np.interp(t_plot_grid, t_main - T_EVAL, surge_b)
    truth_dy = np.interp(t_plot_grid, t_main - T_EVAL, sway_b)
    truth_R = np.hypot(truth_dx, truth_dy)

    return dict(
        eta_hat=eta_hat,
        nu_hat=nu_hat,
        b_hat=b_hat,
        eta_wave=eta_wave,
        heading_compass=heading_compass,
        Tp_obs=Tp_obs,
        win_dt=float(t_main[1] - t_main[0]),
        win_dt_est=float(t_est[1] - t_est[0]),
        samples_lf_x=samples_lf_x,
        samples_lf_y=samples_lf_y,
        samples_lf_yaw=samples_lf_yaw,
        samples_wf_x=samples_wf_x,
        samples_wf_y=samples_wf_y,
        samples_wf_yaw=samples_wf_yaw,
        samples_wf_heave=samples_wf_heave,
        samples_wf_roll=samples_wf_roll,
        samples_wf_pitch=samples_wf_pitch,
        t_pred_grid=t_pred_grid,
        t_plot_grid=t_plot_grid,
        truth_dx=truth_dx,
        truth_dy=truth_dy,
        truth_R=truth_R,
    )


# --------------------------- pipeline pieces -----------------------------

def _bayes_post(samples: np.ndarray, dt: float, T_decorr: float,
                prior_sigma: float):
    """Build a fresh BayesianSigmaEstimator, feed the samples, return its
    SigmaPosterior + PosteriorHealth (the latter needed by
    compose_validity_badge)."""
    est = BayesianSigmaEstimator(
        prior_sigma2=prior_sigma * prior_sigma,
        T_decorr_s=T_decorr,
        dt_s=dt,
        prior_strength_n0=PRIOR_N0,
        window_s=WIN_S * 2.0,           # capacity -- we'll only push WIN_S worth
        assume_zero_mean=True,
    )
    for x in samples:
        est.update(float(x))
    return est.posterior(), est.health()


def _zero_up_crossing_period(x: np.ndarray, dt: float) -> float:
    """Mean zero-up-crossing period of a (demeaned) 1D series.

    Counts indices ``i`` where ``x[i] <= 0 < x[i+1]``. Returns
    ``T_zc = (t_{N-1} - t_0) / N_zc`` if ``N_zc >= 2``, else ``nan``.
    Pure helper; no estimator state.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return float("nan")
    crossings = np.where((x[:-1] <= 0.0) & (x[1:] > 0.0))[0]
    n_zc = crossings.size
    if n_zc < 2:
        return float("nan")
    span_s = (crossings[-1] - crossings[0]) * dt
    return float(span_s / max(n_zc - 1, 1))


def build_live_sigma_posterior(
    d: dict,
    sigma_R_b_hat_m: float = 0.0,
    joint: Optional[GangwayJointState] = None,
    cfg=None,
) -> LiveSigmaPosterior:
    """Build the LiveSigmaPosterior for one seed from the per-channel
    pre-WCF samples.

    Includes the optional WF roll/pitch/heave posteriors (always-on
    when the loader provides ``samples_wf_{heave,roll,pitch}``), so
    downstream gangway-bar consumers see ``gangway_wf_coverage =
    "full_6dof"`` instead of the horizontal-3DOF fallback. The
    horizontal channels are unaffected.

    When ``joint`` and ``cfg`` are supplied, also synthesizes a
    pre-WCF dL_WF sample series by projecting the per-DOF WF samples
    through the 6-DOF telescope sensitivity ``c6 = telescope_
    sensitivity_6dof(joint, cfg.gangway)``, demeans, and reports back
    the resulting ``sigma_dL_wf_measured`` (m) and
    ``T_zc_dL_wf_measured`` (s, mean zero-up-crossing period). These
    feed the LF-peak + Gumbel-WF-max gangway WCF prediction in the
    panel.
    """
    dt_m = d["win_dt"]
    plf_x, hlf_x = _bayes_post(d["samples_lf_x"], dt_m, T_DECORR_LF, PRIOR_SIGMA_LF)
    plf_y, hlf_y = _bayes_post(d["samples_lf_y"], dt_m, T_DECORR_LF, PRIOR_SIGMA_LF)
    plf_z, hlf_z = _bayes_post(d["samples_lf_yaw"], dt_m, T_DECORR_LF_YAW, PRIOR_SIGMA_LF_YAW)
    pwf_x, hwf_x = _bayes_post(d["samples_wf_x"], dt_m, T_DECORR_WF, PRIOR_SIGMA_WF)
    pwf_y, hwf_y = _bayes_post(d["samples_wf_y"], dt_m, T_DECORR_WF, PRIOR_SIGMA_WF)
    pwf_z, hwf_z = _bayes_post(d["samples_wf_yaw"], dt_m, T_DECORR_WF, PRIOR_SIGMA_WF_YAW)
    pwf_h = pwf_r = pwf_p = None
    if "samples_wf_heave" in d:
        pwf_h, _ = _bayes_post(d["samples_wf_heave"], dt_m, T_DECORR_WF,
                               PRIOR_SIGMA_WF_HEAVE)
        pwf_r, _ = _bayes_post(d["samples_wf_roll"], dt_m, T_DECORR_WF,
                               PRIOR_SIGMA_WF_RP)
        pwf_p, _ = _bayes_post(d["samples_wf_pitch"], dt_m, T_DECORR_WF,
                               PRIOR_SIGMA_WF_RP)

    rng = np.random.default_rng(0)
    rad_lf = combine_radial_posterior(plf_x, plf_y, n_mc=4000, rng=rng,
                                      sample_mean_x=hlf_x.sample_mean,
                                      sample_mean_y=hlf_y.sample_mean)
    rad_wf = combine_radial_posterior(pwf_x, pwf_y, n_mc=4000, rng=rng,
                                      sample_mean_x=hwf_x.sample_mean,
                                      sample_mean_y=hwf_y.sample_mean)

    # Worst-of-three per band -> single ValidityBadge per band.
    badges_lf = [compose_validity_badge(h) for h in (hlf_x, hlf_y, hlf_z)]
    badges_wf = [compose_validity_badge(h) for h in (hwf_x, hwf_y, hwf_z)]

    def _worst_badge(badges):
        rank = {"OK": 0, "WARMING": 1, "UNSETTLED": 2, "INVALID": 3}
        return max(badges, key=lambda b: rank.get(b.level, 0))

    # ----- Direct gangway-channel WF posterior -----
    # When the joint geometry is supplied, project the per-DOF WF
    # samples through c6 to synthesize a pre-WCF dL_WF series, then
    # measure sigma and T_zc directly. This is the gangway-axis
    # equivalent of what an instrumented gangway would report from
    # its own telescoping channel; in real deployment the vessel
    # would carry the measurement, here we reconstruct it from the
    # already-loaded brucon DOF samples (apples-to-apples with the
    # rest of the loader).
    #
    # WF-only (no LF): the Gumbel/Rice extreme-value formula in the
    # panel assumes a narrow-band Gaussian noise where the peak
    # factor a_q multiplies sigma by sqrt(2 ln N_eff). The LF
    # channel has a much longer correlation time than the WF, so
    # its "effective N_eff" over a 30 s near-peak window is ~1, not
    # tau_LF / T_zc_WF. Including its variance into a single
    # combined sigma would make the Gumbel formula massively
    # overcount the LF tail. The LF deterministic peak is already
    # accounted for additively via |c . delta_eta_mean(t_peak)|, so
    # the residual LF stochastic variance (about that deterministic
    # mean) is small here. Keep WF-only.
    sigma_dL_wf_meas = None
    T_zc_dL_wf_meas = None
    if joint is not None and cfg is not None:
        c6 = telescope_sensitivity_6dof(joint, cfg.gangway)
        wf_stack = np.column_stack([
            d["samples_wf_x"],
            d["samples_wf_y"],
            d["samples_wf_heave"],
            d["samples_wf_roll"],
            d["samples_wf_pitch"],
            d["samples_wf_yaw"],
        ])
        dL_wf_samples = wf_stack @ c6
        dL_wf_samples = dL_wf_samples - dL_wf_samples.mean()
        sigma_dL_wf_meas = float(dL_wf_samples.std(ddof=1))
        T_zc_dL_wf_meas = _zero_up_crossing_period(dL_wf_samples, dt_m)

    return LiveSigmaPosterior(
        posterior_lf_x=plf_x, posterior_lf_y=plf_y, posterior_lf_yaw=plf_z,
        radial_lf=rad_lf, validity_lf=_worst_badge(badges_lf),
        posterior_wf_x=pwf_x, posterior_wf_y=pwf_y, posterior_wf_yaw=pwf_z,
        radial_wf=rad_wf, validity_wf=_worst_badge(badges_wf),
        sigma_R_b_hat_m=float(sigma_R_b_hat_m),
        posterior_wf_heave=pwf_h,
        posterior_wf_roll=pwf_r,
        posterior_wf_pitch=pwf_p,
        sigma_dL_wf_measured=sigma_dL_wf_meas,
        T_zc_dL_wf_measured=T_zc_dL_wf_meas,
    )


def _trivial_joint(cfg) -> GangwayJointState:
    """Gangway joint at mid-stroke, 0 yaw, 0 elevation. The validation
    focuses on vessel position (which is what brucon logs); gangway tip
    motion is exercised inside the live cell but is irrelevant for the
    truth comparison here."""
    L_min = float(cfg.gangway.telescope_min)
    L_max = float(cfg.gangway.telescope_max)
    return GangwayJointState(h=2.0, alpha_g=0.0, beta_g=0.0,
                             L=0.5 * (L_min + L_max))


# ------------------------------ run --------------------------------------

def main():
    cfg = csov_default_config()
    joint = _trivial_joint(cfg)

    # ---- Load offline calibration artefact for this cell ----
    if not CALIB_NPZ.exists():
        sys.exit(
            f"Calibration artefact missing: {CALIB_NPZ}\n"
            f"Run scripts/p7_brucon_validation/peak_R_b_hat_sigma_pwq30.py first."
        )
    calib = np.load(CALIB_NPZ, allow_pickle=True)
    calib_t = calib["t_grid"]
    calib_delta_eta = calib["delta_eta_mean"]                  # [N_calib, 3]
    sigma_R_b_hat_m = float(calib["sigma_R_b_hat_m"])
    print(f"Loaded calibration {CALIB_NPZ.name}: "
          f"sigma_R_b_hat = {sigma_R_b_hat_m:.3f} m, "
          f"N_calib = {len(calib_t)} samples on [0, {calib_t[-1]:.1f}] s")

    seed_data = []
    for seed in SEEDS:
        d = load_seed(seed)
        if d is None:
            print(f"  seed {seed}: skipped (missing/truncated)")
            continue
        seed_data.append((seed, d))
    print(f"Loaded {len(seed_data)}/{len(SEEDS)} seeds")
    if not seed_data:
        sys.exit("No seeds loaded.")

    # Per-seed predictions and realised trajectories.
    t_pred = seed_data[0][1]["t_pred_grid"]
    t_plot = seed_data[0][1]["t_plot_grid"]

    # Resample the precomputed mean delta_eta onto the plot/prediction grid.
    delta_eta_on_grid = np.zeros((len(t_pred), 3))
    for k in range(3):
        delta_eta_on_grid[:, k] = np.interp(t_pred, calib_t, calib_delta_eta[:, k])

    pred_R_envelope = np.zeros((len(seed_data), len(t_pred)))
    pred_R_offset = np.zeros((len(seed_data), len(t_pred)))      # |eta_hat + dEta|, no sigma
    sigma_R_total = np.zeros(len(seed_data))
    sigma_R_lf = np.zeros(len(seed_data))
    sigma_R_wf = np.zeros(len(seed_data))
    truth_R = np.array([d["truth_R"] for _, d in seed_data])
    truth_dx = np.array([d["truth_dx"] for _, d in seed_data])
    truth_dy = np.array([d["truth_dy"] for _, d in seed_data])

    cell_summary = []
    for k, (seed, d) in enumerate(seed_data):
        sigma_post = build_live_sigma_posterior(d, sigma_R_b_hat_m=sigma_R_b_hat_m)
        sigma_R_lf[k] = sigma_post.radial_lf.sigma_R_median
        sigma_R_wf[k] = sigma_post.radial_wf.sigma_R_median
        sigma_R_total[k] = float(np.sqrt(sigma_R_lf[k] ** 2
                                          + sigma_R_wf[k] ** 2
                                          + sigma_R_b_hat_m ** 2))

        obs = LiveObserverState(
            eta_hat=d["eta_hat"],
            nu_hat=d["nu_hat"],
            b_hat=d["b_hat"],
            eta_wave=d["eta_wave"],
            heading_compass=d["heading_compass"],
        )
        cell = evaluate_decision_cell_live(
            cfg, joint, obs, sigma_post,
            scenario=SCENARIO,
            k_sigma=0.674,
            t_end_wcfdi=T_END_WCFDI,
            n_t=N_T,
            Tp_obs_s=d["Tp_obs"],
            precomputed_delta_eta_mean=delta_eta_on_grid,
            precomputed_t_grid=t_pred,
        )
        cell_summary.append((seed, cell))

        # Replicate the deterministic envelope time series for plotting using
        # the same precomputed mean trajectory the live cell consumed.
        eta_xy = d["eta_hat"][None, 0:2] + delta_eta_on_grid[:, 0:2]
        pos_t = np.sqrt(np.sum(eta_xy ** 2, axis=1))
        pred_R_offset[k] = pos_t
        pred_R_envelope[k] = pos_t + 0.674 * sigma_R_total[k]

    # ---- headline statistics ----
    pos_warn = float(cfg.operational_limits.position_warning_radius_m)
    pos_alarm = float(cfg.operational_limits.position_alarm_radius_m)

    print(f"\nWindow: t in [{WIN_START:.0f}, {WIN_END:.0f}] s "
          f"({WIN_S:.0f} s before WCF)  |  T_eval = T_WCF - 5 = {T_EVAL:.0f} s")
    print(f"Operational thresholds: pos_warn={pos_warn:.2f} m, pos_alarm={pos_alarm:.2f} m")

    print(f"\n--- per-seed sigma_R from live Bayesian posteriors ---")
    print(f"  sigma_R_LF   median  = {np.median(sigma_R_lf):.3f}  "
          f"(min {sigma_R_lf.min():.3f}, max {sigma_R_lf.max():.3f})  m")
    print(f"  sigma_R_WF   median  = {np.median(sigma_R_wf):.3f}  "
          f"(min {sigma_R_wf.min():.3f}, max {sigma_R_wf.max():.3f})  m")
    print(f"  sigma_R_tot  median  = {np.median(sigma_R_total):.3f}  m  "
          f"(brucon ensemble truth: 0.840 m)")

    # Predicted vs realised peak |R|.
    pred_peak = pred_R_envelope.max(axis=1)
    truth_peak = truth_R.max(axis=1)
    pred_peak_offset = pred_R_offset.max(axis=1)
    print(f"\n--- per-seed WCFDI peak |R| (m) ---")
    print(f"  predicted (offset only)         : median={np.median(pred_peak_offset):.3f}  "
          f"min={pred_peak_offset.min():.3f}  max={pred_peak_offset.max():.3f}")
    print(f"  predicted envelope (+ k*sigma_R): median={np.median(pred_peak):.3f}  "
          f"min={pred_peak.min():.3f}  max={pred_peak.max():.3f}")
    print(f"  realised (truth)                : median={np.median(truth_peak):.3f}  "
          f"min={truth_peak.min():.3f}  max={truth_peak.max():.3f}")

    n_cover = int(np.sum(pred_peak >= truth_peak))
    print(f"\n  envelope covers truth peak in {n_cover}/{len(seed_data)} seeds "
          f"({100*n_cover/len(seed_data):.0f}%)")

    # Traffic-light tally.
    from collections import Counter
    intact_lights = Counter(c.intact_traffic for _, c in cell_summary)
    wcfdi_lights = Counter(c.wcfdi_traffic for _, c in cell_summary)
    print(f"\n  intact_traffic distribution : {dict(intact_lights)}")
    print(f"  wcfdi_traffic  distribution : {dict(wcfdi_lights)}")

    # ---- plot ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) ensemble |R(t)|: predicted envelope vs realised, t relative to t_eval
    ax = axes[0, 0]
    pred_R_env_med = np.median(pred_R_envelope, axis=0)
    pred_R_env_lo = np.percentile(pred_R_envelope, 5, axis=0)
    pred_R_env_hi = np.percentile(pred_R_envelope, 95, axis=0)
    pred_R_off_med = np.median(pred_R_offset, axis=0)
    truth_R_med = np.median(truth_R, axis=0)
    truth_R_lo = np.percentile(truth_R, 5, axis=0)
    truth_R_hi = np.percentile(truth_R, 95, axis=0)

    # On the plot grid, time origin = t_eval = T_WCF - 5 s, so the
    # WCF event is at t = +5 s. Shift the predicted grid to match.
    t_pred_plot = t_pred + (T_EVAL - T_WCF)   # = -5 + t_pred_after_eval
    # Actually simpler: prediction starts at t_eval; WCF is at +5 s
    # relative to t_eval; we drew truth on a grid aligned to t_eval too,
    # so just plot pred against t_pred+0 vs truth against t_plot.

    ax.fill_between(t_plot, truth_R_lo, truth_R_hi, alpha=0.15, color="C0",
                    label="brucon truth 5-95%")
    ax.plot(t_plot, truth_R_med, color="C0", lw=2,
            label=f"brucon truth median ({len(seed_data)} seeds)")
    ax.fill_between(t_pred, pred_R_env_lo, pred_R_env_hi, alpha=0.15, color="C3")
    ax.plot(t_pred, pred_R_env_med, color="C3", lw=2, ls="--",
            label="live cell envelope (median)")
    ax.plot(t_pred, pred_R_off_med, color="C3", lw=1, ls=":",
            label="live cell offset only (median)")

    ax.axvline(T_WCF - T_EVAL, color="k", lw=0.5, ls=":")
    ax.axhline(pos_warn, color="orange", lw=1, ls="--", label=f"pos_warn={pos_warn:.1f}")
    ax.axhline(pos_alarm, color="red", lw=1, ls="--", label=f"pos_alarm={pos_alarm:.1f}")
    ax.set_xlim(t_plot[0], t_plot[-1])
    ax.set_xlabel(f"t - t_eval [s]   (t_eval = T_WCF - 5; WCF dashed)")
    ax.set_ylabel("|R| body-frame [m]")
    ax.set_title("Live cell predicted envelope vs brucon realised |R(t)|")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # (0,1) per-seed scatter: predicted peak vs realised peak
    ax = axes[0, 1]
    ax.scatter(truth_peak, pred_peak, c="C3", alpha=0.7, s=40, label="envelope")
    ax.scatter(truth_peak, pred_peak_offset, c="C2", alpha=0.7, s=40,
               marker="x", label="offset only")
    lim = max(truth_peak.max(), pred_peak.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="k", lw=0.5, ls="--", label="y = x")
    ax.set_xlabel("realised peak |R| (truth) [m]")
    ax.set_ylabel("predicted peak |R| (live cell) [m]")
    ax.set_title("Per-seed predicted vs realised peak")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (1,0) per-seed sigma_R distribution
    ax = axes[1, 0]
    ax.hist(sigma_R_lf, bins=10, alpha=0.5, color="C0", label="sigma_R LF")
    ax.hist(sigma_R_wf, bins=10, alpha=0.5, color="C2", label="sigma_R WF")
    ax.hist(sigma_R_total, bins=10, alpha=0.5, color="C3", label="sigma_R total")
    ax.axvline(0.840, color="k", lw=1, ls="--",
               label="brucon ensemble truth = 0.840")
    ax.set_xlabel("sigma_R [m]")
    ax.set_ylabel("seeds")
    ax.set_title("Per-seed Bayesian sigma_R posterior medians")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (1,1) per-seed b_hat scatter (sway vs surge), in kN
    ax = axes[1, 1]
    bhat_kN = np.array([d["b_hat"] for _, d in seed_data]) * 1e-3
    ax.scatter(bhat_kN[:, 0], bhat_kN[:, 1], c="C0", s=40, alpha=0.7,
               label=r"$\hat{b}$ per seed")
    ax.axhline(-110, color="C0", lw=0.5, ls=":",
               label=r"$-\overline{\mathrm{Order}}_{\mathrm{sway}} \approx -110$ kN")
    ax.axvline(-52, color="C0", lw=0.5, ls="--",
               label=r"$-\overline{\mathrm{Order}}_{\mathrm{surge}} \approx -52$ kN")
    ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
    ax.set_xlabel(r"$\hat{b}_{surge}$ [kN]")
    ax.set_ylabel(r"$\hat{b}_{sway}$ [kN]")
    ax.set_title(r"Per-seed observer bias estimate $\hat{b}$ at $t_{eval}$")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle(f"Live operational CQA cell vs brucon truth at {TAG}", fontsize=12)
    plt.tight_layout()
    out = THIS / f"live_cell_per_seed_{TAG}.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    _apply_args(_parse_args())
    main()
