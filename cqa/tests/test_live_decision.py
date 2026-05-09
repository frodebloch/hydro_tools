"""Focused regression tests for the lift-coupling wiring in live_decision.

These tests pin down the contract between cqa.config.VesselParticulars
.lift_coupling_K_per_rad and cqa.live_decision.evaluate_decision_cell_live:

  - K=0  -> evaluate_decision_cell_live uses plain pulse_response.
  - K>0  -> evaluates with pulse_response_with_lift_coupling and yields
            a DIFFERENT wcfdi_pos_peak_m on a head/quartering b_hat (the
            mechanism is non-trivial in that regime).
  - b_hat colinear with surge but with F_x = 0 (zero-surge bias): the
    coupling forcing -F_x*K*dpsi vanishes, so K=0 vs K=3.4 yield (very
    nearly) identical wcfdi_pos_peak_m.
  - precomputed_delta_eta_mean path bypasses K entirely (delta is taken
    as given, no integration done).

The tests do NOT validate the *quality* of the coupling correction
against brucon -- that lives in scripts/p7_brucon_validation/. They
only ensure the wiring is live and the K knob does what it says.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cqa import (
    csov_default_config,
    GangwayJointState,
    WcfdiScenario,
    LiveObserverState,
    LiveSigmaPosterior,
    evaluate_decision_cell_live,
)
from cqa.online_estimator import (
    SigmaPosterior, RadialPosterior, ValidityBadge,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _trivial_sigma_posterior(sigma: float) -> SigmaPosterior:
    """Build a minimal SigmaPosterior with the only fields the live
    cell consumes (sigma_median). Other fields are filled with
    representative finite values so dataclasses validation passes.
    """
    s2 = sigma * sigma
    return SigmaPosterior(
        sigma2_mean=s2, sigma2_median=s2, sigma2_lo=s2, sigma2_hi=s2,
        sigma_mean=sigma, sigma_median=sigma, sigma_lo=sigma, sigma_hi=sigma,
        n_raw=100, n_eff=50.0, alpha=10.0, beta=s2 * 9.0,
        prior_sigma2=s2, prior_strength_n0=2.0, credible=0.90,
    )


def _trivial_radial_posterior(sigma_R: float) -> RadialPosterior:
    """Minimal RadialPosterior; only sigma_R_median is consumed by
    live_decision via radial_lf.sigma_R_median / radial_wf.sigma_R_median.
    Other fields are filled with representative finite values."""
    return RadialPosterior(
        sigma_R_median=sigma_R, sigma_R_mean=sigma_R,
        sigma_R_lo=sigma_R, sigma_R_hi=sigma_R,
        expected_R_median=sigma_R * float(np.sqrt(np.pi) / 2),
        expected_R_lo=sigma_R, expected_R_hi=sigma_R,
        n_mc=4000, n_eff_min=10.0, n_eff_x=10.0, n_eff_y=10.0,
        is_warm=True, credible=0.90,
        radial_mean_offset_m=0.0, radial_mean_offset_over_sigma=0.0,
    )


def _ok_badge() -> ValidityBadge:
    return ValidityBadge(level="OK", reasons=())


def _make_sigma_post(sigma_lf: float = 0.3, sigma_wf: float = 0.5,
                     sigma_R_b_hat_m: float = 0.1) -> LiveSigmaPosterior:
    """Compose a non-degenerate LiveSigmaPosterior with everything OK."""
    plf = _trivial_sigma_posterior(sigma_lf)
    pwf = _trivial_sigma_posterior(sigma_wf)
    rad_lf = _trivial_radial_posterior(sigma_lf * float(np.sqrt(2)))
    rad_wf = _trivial_radial_posterior(sigma_wf * float(np.sqrt(2)))
    return LiveSigmaPosterior(
        posterior_lf_x=plf, posterior_lf_y=plf, posterior_lf_yaw=plf,
        radial_lf=rad_lf, validity_lf=_ok_badge(),
        posterior_wf_x=pwf, posterior_wf_y=pwf, posterior_wf_yaw=pwf,
        radial_wf=rad_wf, validity_wf=_ok_badge(),
        sigma_R_b_hat_m=sigma_R_b_hat_m,
    )


def _make_obs_state(b_hat_kN: tuple[float, float, float]) -> LiveObserverState:
    """Live observer state with eta_hat ~ 0 and the requested b_hat (in kN/kNm)."""
    return LiveObserverState(
        eta_hat=np.zeros(3),
        nu_hat=np.zeros(3),
        b_hat=np.array([b_hat_kN[0] * 1e3, b_hat_kN[1] * 1e3,
                        b_hat_kN[2] * 1e3], dtype=float),
        eta_wave=np.zeros(3),
        heading_compass=0.0,
    )


def _config_with_K(K: float):
    """Return a CSOV config with lift_coupling_K_per_rad overridden to K."""
    cfg = csov_default_config()
    cfg.vessel = dataclasses.replace(cfg.vessel, lift_coupling_K_per_rad=K)
    return cfg


def _joint():
    return GangwayJointState(h=15.0, alpha_g=0.0, beta_g=0.0, L=25.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_K_zero_matches_uncoupled_baseline():
    """With K = 0 the live cell must reproduce the un-coupled
    pulse_response output. We do not have a separate switch -- we rely
    on the implementation contract that K = 0 hits the plain
    pulse_response branch. Sanity-check that the result is finite and
    self-consistent rather than re-deriving the value here."""
    cfg = _config_with_K(0.0)
    obs = _make_obs_state(b_hat_kN=(-200.0, -100.0, -500.0))
    sigma = _make_sigma_post()
    cell = evaluate_decision_cell_live(cfg, _joint(), obs, sigma)
    assert np.isfinite(cell.wcfdi_pos_peak_m)
    assert cell.wcfdi_pos_peak_m > 0.0
    # No live offset, no coupling -- the WCFDI peak must exceed the
    # sigma halo alone (deterministic transient is non-zero).
    halo = 0.674 * float(np.sqrt(
        sigma.radial_lf.sigma_R_median ** 2
        + sigma.radial_wf.sigma_R_median ** 2
        + sigma.sigma_R_b_hat_m ** 2
    ))
    assert cell.wcfdi_pos_peak_m > halo


def test_K_changes_wcfdi_peak_on_head_quartering_b_hat():
    """With a head/quartering b_hat (non-zero F_x AND F_y), turning on
    the lift coupling must shift wcfdi_pos_peak_m by a measurable
    amount. This pins down that the K knob is actually live."""
    obs = _make_obs_state(b_hat_kN=(-200.0, -100.0, -500.0))
    sigma = _make_sigma_post()

    cell_K0 = evaluate_decision_cell_live(_config_with_K(0.0), _joint(), obs, sigma)
    cell_K = evaluate_decision_cell_live(_config_with_K(3.40), _joint(), obs, sigma)

    # Both must be finite.
    assert np.isfinite(cell_K0.wcfdi_pos_peak_m)
    assert np.isfinite(cell_K.wcfdi_pos_peak_m)
    # The coupled result must differ from the un-coupled one by more
    # than mm. Empirically (from the brucon validation matrix) the
    # impact is decimetre-scale on Bf 8 head cases; we use a tolerant
    # 1 mm threshold so the test is sensitive only to "wired or not".
    assert abs(cell_K.wcfdi_pos_peak_m - cell_K0.wcfdi_pos_peak_m) > 1e-3


def test_K_irrelevant_when_F_x_is_zero():
    """The coupling forcing is -F_x * K * dpsi(t). When b_hat_x = 0
    (pure beam-on environment), the correction must vanish: K = 0 and
    K = 3.4 yield identical wcfdi_pos_peak_m up to numerical noise."""
    obs = _make_obs_state(b_hat_kN=(0.0, -300.0, -500.0))
    sigma = _make_sigma_post()

    cell_K0 = evaluate_decision_cell_live(_config_with_K(0.0), _joint(), obs, sigma)
    cell_K = evaluate_decision_cell_live(_config_with_K(3.40), _joint(), obs, sigma)

    # Allow microscopic numerical drift from the second Picard pass
    # (it integrates with delta_b = 0 the second time, but the order
    # of operations differs slightly from the n_iter=1 path).
    assert abs(cell_K.wcfdi_pos_peak_m - cell_K0.wcfdi_pos_peak_m) < 1e-6


def test_K_ignored_when_precomputed_delta_eta_supplied():
    """precomputed_delta_eta_mean bypasses the integrator entirely. K
    therefore must have NO effect on wcfdi_pos_peak_m in that path."""
    obs = _make_obs_state(b_hat_kN=(-200.0, -100.0, -500.0))
    sigma = _make_sigma_post()

    # Synthetic monotone-ramp deviation trajectory in body-frame.
    t_grid = np.linspace(0.0, 60.0, 121)
    delta = np.zeros((t_grid.size, 3))
    delta[:, 0] = 0.5 * (1.0 - np.exp(-t_grid / 15.0))      # surge ramp to ~0.5 m
    delta[:, 1] = -0.3 * (1.0 - np.exp(-t_grid / 20.0))     # sway ramp to ~-0.3 m
    delta[:, 2] = 0.05 * (1.0 - np.exp(-t_grid / 25.0))     # yaw to ~0.05 rad

    common = dict(joint=_joint(), obs_state=obs, sigma_post=sigma,
                  precomputed_delta_eta_mean=delta, precomputed_t_grid=t_grid)
    cell_K0 = evaluate_decision_cell_live(_config_with_K(0.0), **common)
    cell_K = evaluate_decision_cell_live(_config_with_K(3.40), **common)

    assert cell_K0.wcfdi_pos_peak_m == pytest.approx(cell_K.wcfdi_pos_peak_m, abs=1e-12)


def test_csov_default_has_calibrated_K():
    """csov_default_config wires the brucon-calibrated K (~3.4 per rad)
    so live_decision picks it up out of the box."""
    cfg = csov_default_config()
    assert cfg.vessel.lift_coupling_K_per_rad == pytest.approx(3.40, abs=0.05)
