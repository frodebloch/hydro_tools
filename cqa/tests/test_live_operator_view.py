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


# ---------------------------------------------------------------------------
# Gangway telescope bar
# ---------------------------------------------------------------------------


def _gw_joint_forward(L=25.0, h=15.0):
    """Forward-pointing gangway, horizontal boom: e_L = (1, 0, 0).

    With cfg.gangway base = (5, -9, -8) and h, the rotation centre is
    at body (5, -9, -8 - h). For h=15 -> r_z = -23 m. Sensitivities:
        c3 = -(e_x, e_y, e_x * (-r_y) + e_y * r_x)
           = -(1, 0, 1 * 9 + 0)
           = (-1, 0, -9)
        c6 = -(1, 0, 0, 0, -r_z, -r_y) = (-1, 0, 0, 0, 23, -9)
            (only surge, pitch, yaw entries non-zero).
    """
    from cqa import GangwayJointState
    return GangwayJointState(h=h, alpha_g=0.0, beta_g=0.0, L=L)


def test_gangway_bar_absent_by_default():
    """Backwards compatibility: when joint=None, gangway bar fields keep
    their defaults and the position bars are unchanged."""
    cfg = _config_with_K(0.0)
    obs = _make_obs_state(b_hat_kN=(-200.0, -100.0, -500.0))
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(cfg, obs, sigma)
    assert s.gangway_present is False
    assert s.gangway_dL_p50 == 0.0
    assert s.gangway_dL_p95 == 0.0
    assert s.gangway_traffic == "green"
    # Overall traffic is unchanged from the worst of intact / WCF.
    from cqa.decision_matrix import _worst
    assert s.overall_traffic == _worst(s.intact_traffic, s.wcf_traffic)


def test_gangway_bar_present_when_joint_provided():
    cfg = _config_with_K(0.0)
    obs = _make_obs_state(b_hat_kN=(-200.0, -100.0, -500.0))
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(cfg, obs, sigma, joint=_gw_joint_forward())
    assert s.gangway_present is True
    assert math.isfinite(s.gangway_sigma_dL_intact_m)
    assert math.isfinite(s.gangway_sigma_dL_wcf_m)
    assert s.gangway_sigma_dL_wcf_m >= s.gangway_sigma_dL_intact_m  # WCF includes WF + b_hat
    # Quantile ordering.
    assert s.gangway_dL_p05 < s.gangway_dL_p50 < s.gangway_dL_p95
    # Coverage flag: no roll/pitch/heave WF posteriors -> horizontal_3dof.
    assert s.gangway_wf_coverage == "horizontal_3dof"
    # WCF peak time inside the integration window.
    assert 0.0 <= s.gangway_t_peak_s <= 60.0


def test_gangway_sigma_dL_intact_matches_LF_only_projection():
    """The intact gangway sigma must equal sqrt((c3 .* sig_lf)^2 . 1)
    where c3 is telescope_sensitivity (3-DOF body LF). WF and b_hat
    must NOT leak into the intact sigma_dL."""
    from cqa.gangway import telescope_sensitivity
    cfg = _config_with_K(0.0)
    sig_lf = 0.3
    # Huge WF and b_hat to detect any leak.
    sigma = _make_sigma_post(sigma_lf=sig_lf, sigma_wf=99.0,
                             sigma_R_b_hat_m=99.0)
    obs = _make_obs_state(b_hat_kN=(-1.0, -1.0, -1.0))
    joint = _gw_joint_forward()
    s = summarise_for_operator_live(cfg, obs, sigma, joint=joint)
    c3 = telescope_sensitivity(joint, cfg.gangway)
    sig_lf_vec = np.array([sig_lf, sig_lf, sig_lf])
    sigma_dL_th = float(np.sqrt(np.sum((c3 * sig_lf_vec) ** 2)))
    assert s.gangway_sigma_dL_intact_m == pytest.approx(sigma_dL_th, rel=1e-9)


def test_gangway_sigma_dL_wcf_horizontal_3dof_fallback():
    """Without roll/pitch/heave WF posteriors, the WCF sigma_dL is

        var_LF (3-DOF c3) + var_WF (only x,y,yaw entries of c6, others
        zero) + var_b_hat (b_hat_axis split, projected through c6[0:2]).

    This is the LOWER-BOUND surfaced via the horizontal_3dof coverage
    tag.
    """
    from cqa.gangway import telescope_sensitivity, telescope_sensitivity_6dof
    cfg = _config_with_K(0.0)
    sig_lf, sig_wf, sig_bh = 0.3, 0.5, 0.1
    sigma = _make_sigma_post(sigma_lf=sig_lf, sigma_wf=sig_wf,
                             sigma_R_b_hat_m=sig_bh)
    obs = _make_obs_state(b_hat_kN=(-200.0, -100.0, -500.0))
    joint = _gw_joint_forward()
    s = summarise_for_operator_live(cfg, obs, sigma, joint=joint)

    c3 = telescope_sensitivity(joint, cfg.gangway)
    c6 = telescope_sensitivity_6dof(joint, cfg.gangway)
    var_lf = float(np.sum((c3 * np.array([sig_lf] * 3)) ** 2))
    sig_wf6 = np.array([sig_wf, sig_wf, 0.0, 0.0, 0.0, sig_wf])
    var_wf = float(np.sum((c6 * sig_wf6) ** 2))
    sig_bh_axis = sig_bh / math.sqrt(2.0)
    var_bh = (c6[0] * sig_bh_axis) ** 2 + (c6[1] * sig_bh_axis) ** 2
    sigma_dL_th = math.sqrt(var_lf + var_wf + var_bh)

    assert s.gangway_wf_coverage == "horizontal_3dof"
    assert s.gangway_sigma_dL_wcf_m == pytest.approx(sigma_dL_th, rel=1e-9)


def test_gangway_full_6dof_increases_sigma_when_pitch_provided():
    """Adding a roll/pitch/heave WF posterior must INCREASE sigma_dL
    relative to the horizontal-only fallback (the c6[3,4,5] entries
    pick up additional variance), and must flip the coverage tag to
    full_6dof."""
    from cqa.online_estimator import SigmaPosterior, RadialPosterior, ValidityBadge
    from cqa import LiveSigmaPosterior
    cfg = _config_with_K(0.0)
    sig_lf, sig_wf, sig_bh = 0.05, 0.05, 0.0
    sigma_h = _make_sigma_post(sigma_lf=sig_lf, sigma_wf=sig_wf,
                               sigma_R_b_hat_m=sig_bh)
    # Build an alternative posterior that ALSO carries roll/pitch/
    # heave at ~1 deg / ~1 deg / 0.3 m (small but non-zero).
    sig_rpp = math.radians(1.0)
    sig_heave = 0.3
    from tests._live_fixtures import _trivial_sigma_posterior  # noqa
    p_roll = _trivial_sigma_posterior(sig_rpp)
    p_pitch = _trivial_sigma_posterior(sig_rpp)
    p_heave = _trivial_sigma_posterior(sig_heave)
    import dataclasses
    sigma_full = dataclasses.replace(
        sigma_h,
        posterior_wf_heave=p_heave,
        posterior_wf_roll=p_roll,
        posterior_wf_pitch=p_pitch,
    )

    obs = _make_obs_state(b_hat_kN=(-200.0, -100.0, -500.0))
    joint = _gw_joint_forward()
    s_h = summarise_for_operator_live(cfg, obs, sigma_h, joint=joint)
    s_f = summarise_for_operator_live(cfg, obs, sigma_full, joint=joint)
    assert s_h.gangway_wf_coverage == "horizontal_3dof"
    assert s_f.gangway_wf_coverage == "full_6dof"
    # pitch coefficient is 23 m/rad here -> 1 deg pitch contributes
    # ~0.40 m sigma. Combined with the existing horizontal terms
    # (dominated by the LF yaw-lever-arm c3[2]=-9), the full 6-DOF
    # sigma is ~18 % larger than the horizontal-only fallback.
    assert s_f.gangway_sigma_dL_wcf_m > 1.15 * s_h.gangway_sigma_dL_wcf_m


def test_gangway_signed_dL_extends_for_forward_drift():
    """A vessel drifted forward (eta_n > 0) with a forward-pointing
    gangway means the rotation centre has moved AWAY from the world-
    fixed landing point along +e_L; the telescope must EXTEND, so
    signed dL > 0.

    Specifically: c3 = (-1, 0, -9), eta_hat = (+0.5, 0, 0) gives
    intact dL = c3 . eta_hat = -0.5. Sign is NEGATIVE because cqa's
    convention is "Delta_L > 0 means MORE telescope is required",
    and a positive eta_n is a vessel deviation in +N which keeps the
    same vessel-to-tip vector for a forward-pointing gangway when
    the landing point was set BEHIND the vessel (Delta_L = -e_L .
    Delta_p_rc; Delta_p_rc = +N for a forward drift, e_L_world = +N
    too -> Delta_L = -1 * eta_n). The test pins the SIGN of the
    intact-offset value so a future refactor cannot silently flip
    the convention."""
    from cqa.gangway import telescope_sensitivity
    cfg = _config_with_K(0.0)
    sigma = _make_sigma_post()
    eta_hat = (0.5, 0.0, 0.0)
    obs = _make_obs_state(b_hat_kN=(-1.0, -1.0, -1.0), eta_hat=eta_hat)
    joint = _gw_joint_forward()
    s = summarise_for_operator_live(cfg, obs, sigma, joint=joint)
    c3 = telescope_sensitivity(joint, cfg.gangway)
    expected = float(c3 @ np.array(eta_hat))
    assert s.gangway_dL_intact_offset == pytest.approx(expected, rel=1e-9)
    # And sign matches: c3[0] = -1 < 0 and eta_n = +0.5 > 0 -> dL < 0.
    assert s.gangway_dL_intact_offset < 0.0


def test_gangway_traffic_red_when_dL_p95_exceeds_extend_alarm():
    """Punitive b_hat that drives the WCF transient hard along +e_L
    should push dL_p95 past 0.8 * extend_margin -> RED gangway bar."""
    cfg = _config_with_K(0.0)
    # Big forward bias -> big +e_L excursion under WCFDI for forward-
    # pointing gangway. Small extend margin via L0 close to L_max.
    obs = _make_obs_state(b_hat_kN=(800.0, 0.0, 0.0))
    sigma = _make_sigma_post(sigma_lf=0.5, sigma_wf=0.8,
                             sigma_R_b_hat_m=0.3)
    joint = _gw_joint_forward(L=31.5, h=15.0)  # extend_margin = 0.5 m
    s = summarise_for_operator_live(cfg, obs, sigma, joint=joint)
    assert s.gangway_extend_margin_m == pytest.approx(0.5, abs=1e-9)
    assert s.gangway_traffic == "red"
    assert s.overall_traffic == "red"


def test_gangway_plot_three_rows_when_present(tmp_path):
    """Smoke test the plot helper renders 3 axes when gangway is on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cqa.live_operator_view import plot_live_operator_summary

    cfg = _config_with_K(0.0)
    obs = _make_obs_state(b_hat_kN=(-200.0, -100.0, -500.0))
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(cfg, obs, sigma, joint=_gw_joint_forward())
    fig = plot_live_operator_summary(s)
    assert len(fig.axes) == 3
    plt.close(fig)
    s2 = summarise_for_operator_live(cfg, obs, sigma)  # no joint
    fig2 = plot_live_operator_summary(s2)
    assert len(fig2.axes) == 2
    plt.close(fig2)
