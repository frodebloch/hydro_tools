"""G1 brucon-ensemble validation: companion test.

Single-seed end-to-end check that ``BayesianSigmaEstimator`` recovers
brucon's empirical σ from the ``pwo_lockedTp`` time series. Mirrors
the per-seed evaluation in
``scripts/p7_brucon_validation/online_estimator_brucon_validation.py``,
condensed to a single seed and asserted against the documented
acceptance criteria of analysis.md §12.21.4.

Skipped automatically when the brucon ensemble (or the pdstrip RAO)
isn't on disk, in the same way as the other p7-validation tests.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

# pdstrip + brucon ensemble live outside the repo on the dev host.
_BRUCON_ENSEMBLE = Path(__file__).resolve().parents[1] / (
    "scripts/p7_brucon_validation/work_long_lockedTp"
)
_SEED = 1000
_SEED_DIR = _BRUCON_ENSEMBLE / f"pwo_lockedTp_seed{_SEED:04d}"
_OUT = _SEED_DIR / f"pwo_lockedTp_seed{_SEED:04d}.out"
_PDSTRIP = Path("/home/blofro/src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat")

# Sea state (matches scripts/p7_brucon_validation/long_run_locked_tp_validation.py).
_HS = 4.19571865443425
_TP = 10.22443464601827
_THETA_REL = np.pi / 2
_WINDOW = (1500.0, 3000.0)
_DT_S = 0.1


def _import_harness():
    """Add the validation script directory to sys.path and return parse_output."""
    scripts_dir = _BRUCON_ENSEMBLE.parent
    sys.path.insert(0, str(scripts_dir))
    from harness import parse_output
    return parse_output


@pytest.mark.skipif(not _OUT.exists(), reason=f"brucon ensemble not on disk: {_OUT}")
@pytest.mark.skipif(not _PDSTRIP.exists(), reason=f"pdstrip RAO not on disk: {_PDSTRIP}")
def test_bayesian_sigma_estimator_recovers_brucon_truth_seed1000():
    """End-to-end G1 check on seed 1000.

    Runs the §12.20 sea state, body-frame projects (x_ned, y_ned, ψ),
    pushes 1500 s of late-window samples through two
    BayesianSigmaEstimators (one per axis) with the model-derived
    prior σ² and PSD-derived T_var, and asserts:

      1. Posterior σ_median recovers the in-window empirical std to
         within ±5 % per axis (the analysis.md §12.21.4 target is
         ~3 %; we leave a small margin for cross-platform numerics
         and for the asymmetry between √(β/(α-1)) and E[σ]).
      2. The 90 % credible interval covers the empirical std on
         both axes (single-seed coverage check; ensemble coverage
         lives in the script).
      3. n_eff is well above the warmth threshold on both axes
         (≥ 5 by construction at this window size and decorrelation
         time; we assert ≥ 10).
      4. Radial composition produces σ_R consistent with
         √(σ_x² + σ_y²) of the truth, again within ±5 %.
    """
    parse_output = _import_harness()

    from cqa.config import csov_default_config
    from cqa.controller import LinearDpController
    from cqa.vessel import LinearVesselModel
    from cqa.observer import build_observer_augmented_system, total_position_psd
    from cqa.rao import load_pdstrip_rao, evaluate_rao
    from cqa.psd import wave_elevation_psd
    from cqa.drift import slow_drift_force_psd_newman_pdstrip
    from cqa.wave_response import cqa_theta_rel_to_pdstrip_beta_deg
    from cqa.extreme_value import variance_decorrelation_time_from_psd
    from cqa.online_estimator import BayesianSigmaEstimator, combine_radial_posterior
    from scipy.integrate import trapezoid

    # ---- read brucon, project to body frame, slice to window ----
    result = parse_output(_OUT)
    t = result.columns["t"]
    mask = (t >= _WINDOW[0]) & (t <= _WINDOW[1])
    x_ned = result.columns["x"][mask]
    y_ned = result.columns["y"][mask]
    psi = np.deg2rad(result.columns["heading"][mask])
    cp, sp = np.cos(psi), np.sin(psi)
    x_body = cp * x_ned + sp * y_ned
    y_body = -sp * x_ned + cp * y_ned
    x_centred = x_body - x_body.mean()
    y_centred = y_body - y_body.mean()
    truth_x = float(x_centred.std())
    truth_y = float(y_centred.std())
    truth_R = float(np.sqrt(truth_x ** 2 + truth_y ** 2))

    # ---- build model prior σ² and T_var via total_position_psd ----
    cfg = csov_default_config()
    v = LinearVesselModel.from_config(cfg.vessel)
    ctrl = LinearDpController.from_bandwidth(
        v.M, v.D, omega_n=cfg.controller.omega_n, zeta=cfg.controller.zeta,
    )
    rao = load_pdstrip_rao(str(_PDSTRIP))
    S_drift = slow_drift_force_psd_newman_pdstrip(
        rao_table=rao, Hs=_HS, Tp=_TP, theta_wave_rel=_THETA_REL,
    )
    beta_deg = cqa_theta_rel_to_pdstrip_beta_deg(_THETA_REL)

    def S_eta_w(w):
        H = evaluate_rao(rao, np.array([w]), beta_deg)[0]
        H_pos = np.array([H[0], H[1], H[5]])
        S_eta_scalar = wave_elevation_psd(np.array([w]), _HS, _TP)[0]
        return np.diag(np.abs(H_pos) ** 2 * S_eta_scalar)

    aug = build_observer_augmented_system(
        v, ctrl, cfg.observer, Tp=_TP, T_thr=2.0, include_integrator=True,
    )
    omega = np.unique(np.concatenate([
        np.logspace(-4, -1, 256),
        np.linspace(0.1, 2.5, 1024),
    ]))
    S_y = total_position_psd(aug, [S_drift], S_eta_w, omega)
    sigma2_x = float(trapezoid(S_y[:, 0, 0].real, omega))
    sigma2_y = float(trapezoid(S_y[:, 1, 1].real, omega))
    T_var_x = float(variance_decorrelation_time_from_psd(S_y[:, 0, 0].real, omega))
    T_var_y = float(variance_decorrelation_time_from_psd(S_y[:, 1, 1].real, omega))
    assert sigma2_x > 0 and sigma2_y > 0
    assert T_var_x > 0 and T_var_y > 0

    # ---- stream through estimators ----
    window_s = _WINDOW[1] - _WINDOW[0]
    est_x = BayesianSigmaEstimator(
        prior_sigma2=sigma2_x, T_decorr_s=T_var_x, dt_s=_DT_S,
        window_s=window_s, assume_zero_mean=True,
    )
    est_y = BayesianSigmaEstimator(
        prior_sigma2=sigma2_y, T_decorr_s=T_var_y, dt_s=_DT_S,
        window_s=window_s, assume_zero_mean=True,
    )
    for x in x_centred:
        est_x.update(float(x))
    for y in y_centred:
        est_y.update(float(y))
    post_x = est_x.posterior(credible=0.90)
    post_y = est_y.posterior(credible=0.90)

    # ---- (1) recovery within ±5 % per axis ----
    rel_x = (post_x.sigma_median - truth_x) / truth_x
    rel_y = (post_y.sigma_median - truth_y) / truth_y
    assert abs(rel_x) < 0.05, (
        f"σ_x posterior {post_x.sigma_median:.4f} m vs brucon truth "
        f"{truth_x:.4f} m, rel err {rel_x*100:+.2f}% (limit ±5 %)"
    )
    assert abs(rel_y) < 0.05, (
        f"σ_y posterior {post_y.sigma_median:.4f} m vs brucon truth "
        f"{truth_y:.4f} m, rel err {rel_y*100:+.2f}% (limit ±5 %)"
    )

    # ---- (2) 90 % CI covers the empirical truth ----
    assert post_x.sigma_lo <= truth_x <= post_x.sigma_hi, (
        f"σ_x posterior 90 % CI [{post_x.sigma_lo:.4f}, {post_x.sigma_hi:.4f}] "
        f"does not cover brucon truth {truth_x:.4f} m"
    )
    assert post_y.sigma_lo <= truth_y <= post_y.sigma_hi, (
        f"σ_y posterior 90 % CI [{post_y.sigma_lo:.4f}, {post_y.sigma_hi:.4f}] "
        f"does not cover brucon truth {truth_y:.4f} m"
    )

    # ---- (3) ESS warmth ----
    assert est_x.n_eff >= 10.0, f"n_eff_x = {est_x.n_eff:.1f} < 10 (cold)"
    assert est_y.n_eff >= 10.0, f"n_eff_y = {est_y.n_eff:.1f} < 10 (cold)"

    # ---- (4) radial composition ----
    rng = np.random.default_rng(_SEED)
    radial = combine_radial_posterior(
        post_x, post_y, credible=0.90, n_mc=4000, rng=rng,
        sample_mean_x=float(x_body.mean()),
        sample_mean_y=float(y_body.mean()),
    )
    rel_R = (radial.sigma_R_median - truth_R) / truth_R
    assert abs(rel_R) < 0.05, (
        f"σ_R radial posterior {radial.sigma_R_median:.4f} m vs brucon truth "
        f"{truth_R:.4f} m, rel err {rel_R*100:+.2f}% (limit ±5 %)"
    )
