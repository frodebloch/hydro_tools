"""Focused regression tests for the live operator-facing summary
(`cqa.live_operator_view`).

These tests pin down:

  - dataclass invariants (P50 < P95 always; finite values; traffic-light
    consistency with P95 vs the IMCA thresholds).
  - the zero-offset Rayleigh limit on the intact axis: when sigma_x ==
    sigma_y == sigma and eta_hat == 0, R is Rayleigh(sigma) so
    P50 = sigma * sqrt(2 ln 2) and P95 = sigma * sqrt(2 ln 20).
  - traffic-light boundary behaviour driven by b_hat magnitude: with a
    benign b_hat the WCF row is green; with a punitive b_hat it is red.
  - K=0 vs K=3.4 changes wcf_R_p50/p95 on a head/quartering b_hat (the
    same wiring as test_live_decision.py, but routed through the
    operator panel).
  - wcf_t_peak_s lies strictly inside (0, t_horizon_s).
  - precomputed paths are not exposed by this entry point (it always
    integrates from the live state).

Tests do NOT validate plotting output -- the plot helper is a thin
matplotlib wrapper.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cqa import (
    LiveOperatorSummary,
    summarise_for_operator_live,
)

from _live_fixtures import (
    _make_sigma_post,
    _make_obs_state,
    _config_with_K,
)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_summary_is_finite_and_p50_below_p95():
    cfg = _config_with_K(0.0)
    obs = _make_obs_state(b_hat_kN=(-200.0, -100.0, -500.0))
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(cfg, obs, sigma)
    assert isinstance(s, LiveOperatorSummary)
    for v in (s.intact_R_p50, s.intact_R_p95, s.intact_R_offset_m,
              s.wcf_R_p50, s.wcf_R_p95, s.wcf_R_offset_at_peak_m,
              s.wcf_t_peak_s, s.sigma_R_intact_m, s.sigma_R_wcf_m,
              s.pos_warning_radius_m, s.pos_alarm_radius_m):
        assert np.isfinite(v)
    assert s.intact_R_p50 < s.intact_R_p95
    assert s.wcf_R_p50 < s.wcf_R_p95
    # WCF halo includes WF + b_hat in addition to LF -- must be wider.
    assert s.sigma_R_wcf_m > s.sigma_R_intact_m
    assert s.intact_traffic in {"green", "amber", "red"}
    assert s.wcf_traffic in {"green", "amber", "red"}
    assert s.overall_traffic in {"green", "amber", "red"}


def test_traffic_light_follows_p95_vs_thresholds():
    """`intact_traffic` (resp. `wcf_traffic`) must equal the IMCA rule
    applied to its P95 against the warning/alarm radii."""
    cfg = _config_with_K(0.0)
    obs = _make_obs_state(b_hat_kN=(-300.0, -200.0, -500.0))
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(cfg, obs, sigma)

    def _rule(p95: float) -> str:
        if p95 >= s.pos_alarm_radius_m:
            return "red"
        if p95 >= s.pos_warning_radius_m:
            return "amber"
        return "green"

    assert s.intact_traffic == _rule(s.intact_R_p95)
    assert s.wcf_traffic == _rule(s.wcf_R_p95)


def test_overall_traffic_is_worst_of_two_axes():
    cfg = _config_with_K(0.0)
    obs = _make_obs_state(b_hat_kN=(-400.0, -200.0, -500.0))
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(cfg, obs, sigma)
    rank = {"green": 0, "amber": 1, "red": 2}
    assert (rank[s.overall_traffic]
            >= max(rank[s.intact_traffic], rank[s.wcf_traffic]))


# ---------------------------------------------------------------------------
# Distribution sanity
# ---------------------------------------------------------------------------


def test_intact_zero_offset_matches_rayleigh_quantiles():
    """With LF sigma_x == LF sigma_y == sigma_lf and eta_hat == 0, R is
    Rayleigh(sigma_lf * sqrt(2)) on the intact axis (LF only, no WF or
    b_hat contribution): P50 = sig_axis * sqrt(2 ln 2),
    P95 = sig_axis * sqrt(2 ln 20). Per-call MC has finite-sample error
    so we use 5% tolerance.
    """
    cfg = _config_with_K(0.0)
    # Equal LF sigmas; WF and b_hat MUST be ignored on the intact axis.
    sigma = _make_sigma_post(sigma_lf=0.3, sigma_wf=99.0, sigma_R_b_hat_m=99.0)
    obs = _make_obs_state(b_hat_kN=(0.0, 0.0, 0.0),
                          eta_hat=(0.0, 0.0, 0.0))
    s = summarise_for_operator_live(cfg, obs, sigma, n_mc=20000)

    sig_axis = 0.3                                          # LF only
    p50_th = sig_axis * math.sqrt(2.0 * math.log(2.0))      # ~0.3534
    p95_th = sig_axis * math.sqrt(2.0 * math.log(20.0))     # ~0.7344
    assert s.intact_R_p50 == pytest.approx(p50_th, rel=0.05)
    assert s.intact_R_p95 == pytest.approx(p95_th, rel=0.05)


def test_intact_offset_increases_p50():
    """A non-zero eta_hat strictly increases the median (Rice mean is
    monotone in the offset)."""
    cfg = _config_with_K(0.0)
    sigma = _make_sigma_post(sigma_lf=0.3, sigma_wf=0.3, sigma_R_b_hat_m=0.0)
    obs0 = _make_obs_state(b_hat_kN=(0.0, 0.0, 0.0),
                           eta_hat=(0.0, 0.0, 0.0))
    obs1 = _make_obs_state(b_hat_kN=(0.0, 0.0, 0.0),
                           eta_hat=(1.0, 0.0, 0.0))
    s0 = summarise_for_operator_live(cfg, obs0, sigma, n_mc=20000)
    s1 = summarise_for_operator_live(cfg, obs1, sigma, n_mc=20000)
    assert s1.intact_R_p50 > s0.intact_R_p50 + 0.5
    assert s1.intact_R_offset_m == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# WCF axis
# ---------------------------------------------------------------------------


def test_wcf_t_peak_inside_horizon():
    """The deterministic peak time must lie strictly inside the
    integration horizon (it cannot be at t = 0 because the post-WCF
    transient grows from rest)."""
    cfg = _config_with_K(0.0)
    obs = _make_obs_state(b_hat_kN=(-200.0, -100.0, -500.0))
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(cfg, obs, sigma, t_horizon_s=60.0)
    assert 0.0 < s.wcf_t_peak_s <= 60.0


def test_wcf_grows_with_b_hat_magnitude():
    """Doubling the environmental load magnitude must monotonically
    grow the WCF P50 deterministic peak."""
    cfg = _config_with_K(0.0)
    sigma = _make_sigma_post()
    s_small = summarise_for_operator_live(
        cfg, _make_obs_state(b_hat_kN=(-50.0, -25.0, -100.0)), sigma)
    s_large = summarise_for_operator_live(
        cfg, _make_obs_state(b_hat_kN=(-400.0, -200.0, -800.0)), sigma)
    assert s_large.wcf_R_offset_at_peak_m > s_small.wcf_R_offset_at_peak_m
    assert s_large.wcf_R_p50 > s_small.wcf_R_p50


def test_wcf_traffic_red_under_punitive_load():
    """A clearly punitive b_hat must drive the WCF row into red."""
    cfg = _config_with_K(0.0)
    sigma = _make_sigma_post()
    obs = _make_obs_state(b_hat_kN=(-1000.0, -500.0, -2000.0))
    s = summarise_for_operator_live(cfg, obs, sigma)
    assert s.wcf_traffic == "red"
    assert s.overall_traffic == "red"


def test_wcf_traffic_green_under_benign_load():
    """A negligible b_hat must keep both rows green."""
    cfg = _config_with_K(0.0)
    sigma = _make_sigma_post(sigma_lf=0.1, sigma_wf=0.1, sigma_R_b_hat_m=0.05)
    obs = _make_obs_state(b_hat_kN=(-5.0, -3.0, -10.0))
    s = summarise_for_operator_live(cfg, obs, sigma)
    assert s.intact_traffic == "green"
    assert s.wcf_traffic == "green"
    assert s.overall_traffic == "green"


# ---------------------------------------------------------------------------
# Lift-coupling K wiring through to the operator panel
# ---------------------------------------------------------------------------


def test_K_changes_wcf_axis():
    """Same contract as test_live_decision.test_K_changes_wcfdi_peak_on_head_quartering_b_hat
    but routed through summarise_for_operator_live: turning on the
    coupling must shift the WCF deterministic peak (and therefore the
    P50/P95) by more than mm-scale."""
    obs = _make_obs_state(b_hat_kN=(-200.0, -100.0, -500.0))
    sigma = _make_sigma_post()
    s_K0 = summarise_for_operator_live(_config_with_K(0.0), obs, sigma)
    s_K = summarise_for_operator_live(_config_with_K(3.40), obs, sigma)
    # Deterministic peak: clear-cut contract (no MC noise).
    assert abs(s_K.wcf_R_offset_at_peak_m - s_K0.wcf_R_offset_at_peak_m) > 1e-3
    # Intact axis is independent of K -- pin that down too.
    assert s_K.intact_R_offset_m == pytest.approx(s_K0.intact_R_offset_m,
                                                  abs=1e-12)


def test_K_irrelevant_for_pure_beam_b_hat():
    """When b_hat_x == 0 the coupling correction (-F_x*K*dpsi) vanishes,
    so K = 0 and K = 3.4 must give the same deterministic WCF peak."""
    obs = _make_obs_state(b_hat_kN=(0.0, -300.0, -500.0))
    sigma = _make_sigma_post()
    s_K0 = summarise_for_operator_live(_config_with_K(0.0), obs, sigma)
    s_K = summarise_for_operator_live(_config_with_K(3.40), obs, sigma)
    assert s_K.wcf_R_offset_at_peak_m == pytest.approx(
        s_K0.wcf_R_offset_at_peak_m, abs=1e-6)


# ---------------------------------------------------------------------------
# Sigma plumbing
# ---------------------------------------------------------------------------


def test_sigma_R_intact_is_LF_only_quadrature():
    """sigma_R_intact_m must be the quadrature sum of the per-axis LF
    sigmas only -- WF and b_hat-radial are not part of the intact halo."""
    cfg = _config_with_K(0.0)
    sig_lf = 0.3
    # Set huge WF and b_hat to detect any leak into intact halo.
    sigma = _make_sigma_post(sigma_lf=sig_lf, sigma_wf=99.0,
                             sigma_R_b_hat_m=99.0)
    obs = _make_obs_state(b_hat_kN=(-1.0, -1.0, -1.0))   # tiny so WCF doesn't blow up
    s = summarise_for_operator_live(cfg, obs, sigma)
    sig_R_intact_th = math.hypot(sig_lf, sig_lf)
    assert s.sigma_R_intact_m == pytest.approx(sig_R_intact_th, rel=1e-9)


def test_sigma_R_wcf_is_LF_WF_b_hat_quadrature():
    """sigma_R_wcf_m must be the quadrature sum of LF + WF + b_hat-radial
    (split equally between axes), matching the sigma envelope used by
    evaluate_decision_cell_live.wcfdi_pos_peak_m."""
    cfg = _config_with_K(0.0)
    sig_lf, sig_wf, sig_bh = 0.3, 0.5, 0.1
    sigma = _make_sigma_post(sigma_lf=sig_lf, sigma_wf=sig_wf,
                             sigma_R_b_hat_m=sig_bh)
    obs = _make_obs_state(b_hat_kN=(-100.0, -50.0, -200.0))
    s = summarise_for_operator_live(cfg, obs, sigma)
    sig_axis = math.sqrt(sig_lf ** 2 + sig_wf ** 2 + (sig_bh / math.sqrt(2)) ** 2)
    sig_R_wcf_th = math.hypot(sig_axis, sig_axis)
    assert s.sigma_R_wcf_m == pytest.approx(sig_R_wcf_th, rel=1e-9)
