"""G2 step 5: single-seed brucon companion test for calibrated wcfdi_mc.

Light end-to-end check that on brucon seed 1000, the tau_lost-augmented
calibrated wcfdi_mc:
  (a) runs without errors,
  (b) produces a non-trivial sway transient (>= 30 cm peak) -- this
      catches regressions in the tau_lost_fn plumbing,
  (c) beats the raw wcfdi_mc on per-seed pos_peak prediction (i.e. its
      median pos_peak is closer to the seed's true |Δradial|_peak).

Skipped if the brucon ensemble is not on this host. Full 30-seed
validation lives in scripts/p7_brucon_validation/calibrated_wcfdi_brucon_validation.py
(documented in analysis.md sec.12.21.7).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

# Repo root is two parents up from this test file (tests/ -> cqa/).
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts" / "p7_brucon_validation"
ENSEMBLE = SCRIPTS / "work"
SEED = 1000
SEED_DIR = ENSEMBLE / f"pwo_seed{SEED:04d}"
PDSTRIP = Path("/home/blofro/src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat")


_data_missing = (
    not SEED_DIR.exists()
    or not (SEED_DIR / f"pwo_seed{SEED:04d}.out").exists()
    or not (SEED_DIR / f"pwo_seed{SEED:04d}_estimator.out").exists()
    or not PDSTRIP.exists()
)


@pytest.mark.skipif(_data_missing, reason="brucon ensemble or pdstrip RAO not present on this host")
def test_calibrated_wcfdi_brucon_seed1000_predictions():
    """End-to-end on brucon seed 1000."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        from calibrated_wcfdi_brucon_validation import extract_seed_inputs
    finally:
        sys.path.pop(0)

    from cqa.calibrated_wcfdi import build_calibrated_context, wcfdi_mc_calibrated
    from cqa.wcfdi_mc import wcfdi_mc
    from cqa.transient import WcfdiScenario
    from cqa.rao import load_pdstrip_rao
    sys.path.insert(0, str(SCRIPTS))
    try:
        from run_comparison import setup_cqa
    finally:
        sys.path.pop(0)

    r = extract_seed_inputs(SEED)
    assert r is not None, f"seed {SEED} extraction returned None"

    # Sanity: tau_lost should be non-trivial in sway and yaw on a
    # bus_port WCF (failed thrusters were carrying real load).
    assert abs(r["tau_lost_pre_wcf"][1]) > 50_000.0, (
        f"expected |tau_lost_y| > 50 kN, got {r['tau_lost_pre_wcf'][1]:.0f} N"
    )
    assert abs(r["tau_lost_pre_wcf"][2]) > 1_000_000.0, (
        f"expected |tau_lost_yaw| > 1000 kNm, got {r['tau_lost_pre_wcf'][2]:.0f} Nm"
    )

    cfg, joint = setup_cqa()
    rao = load_pdstrip_rao(str(PDSTRIP))
    scenario = WcfdiScenario(alpha=(0.5, 0.7, 0.5), gamma_immediate=0.8, T_realloc=5.0)

    HS = 4.19571865443425
    TP = 10.22443464601827
    THETA = np.pi / 2
    T_END = 180.0
    N_T = 1801

    # Raw (operator-nominal) baseline
    res_raw = wcfdi_mc(
        cfg, Vw_mean=0.0, Hs=HS, Tp=TP, Vc=0.0, theta_rel=THETA,
        scenario=scenario, joint=joint, rao_table=rao,
        n_samples=200, t_end=T_END, n_t=N_T, rng_seed=12345,
    )

    # Calibrated with measured (sigma, tau_env, tau_lost) -- use median
    # T_eff from the validation script as the scalar duration.
    ctx = build_calibrated_context(
        cfg,
        sigma_measured_lf_body=r["sigma_lf_body"],
        tau_env_measured=r["tau_env_meas"],
        Vw_mean=0.0, Hs=HS, Tp=TP, Vc=0.0, theta_rel=THETA,
        rao_table=rao,
        tau_lost_pre_wcf=r["tau_lost_pre_wcf"],
        tau_lost_pulse_shape="square",
        tau_lost_duration_s=10.0,
    )
    res_cal = wcfdi_mc_calibrated(
        cfg, scenario, joint, ctx,
        n_samples=200, t_end=T_END, n_t=N_T, rng_seed=10_000 + SEED,
    )

    truth_peak = r["delta_radial_peak"]
    raw_peak_med = float(np.nanmedian(res_raw.pos_peak))
    cal_peak_med = float(np.nanmedian(res_cal.pos_peak))

    # (a) ran without errors -- implicit at this point
    # (b) calibrated mean transient should be non-trivial (the linearised
    # mean inside res_cal isn't directly exposed via wcfdi_mc_calibrated;
    # check via cal_peak_med which is dominated by the mean response).
    assert cal_peak_med > 0.4, (
        f"calibrated pos_peak P50 should be > 0.4 m on this seed (true peak {truth_peak:.2f} m), "
        f"got {cal_peak_med:.3f} m"
    )
    # (c) calibrated should be closer to truth than raw on |pos_peak|
    err_raw = abs(raw_peak_med - truth_peak)
    err_cal = abs(cal_peak_med - truth_peak)
    assert err_cal < err_raw, (
        f"calibrated prediction should be closer to truth ({truth_peak:.2f} m) "
        f"than raw ({raw_peak_med:.2f} m, err {err_raw:.2f}); "
        f"calibrated = {cal_peak_med:.2f} m, err {err_cal:.2f}"
    )

    # And the calibrated info should report the tau_lost pulse parameters.
    cal_info = res_cal.info["calibration"]
    np.testing.assert_array_equal(
        cal_info["tau_lost_pre_wcf"], r["tau_lost_pre_wcf"],
    )
    assert cal_info["tau_lost_pulse_shape"] == "square"
    assert cal_info["tau_lost_duration_s"] == 10.0
