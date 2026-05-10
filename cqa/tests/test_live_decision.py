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

import numpy as np
import pytest

from cqa import (
    csov_default_config,
    WcfdiScenario,
    evaluate_decision_cell_live,
)

from _live_fixtures import (
    _make_sigma_post,
    _make_obs_state,
    _config_with_K,
    _joint,
)


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
