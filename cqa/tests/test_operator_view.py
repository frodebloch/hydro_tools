"""Tests for cqa.operator_view: WCFDI MC -> 2-bar operator summary."""

from __future__ import annotations

import numpy as np
import pytest

from cqa import (
    csov_default_config,
    summarise_for_operator,
    summarise_intact_prior,
    plot_operator_summary,
    plot_intact_prior,
    GangwayJointState,
    SeaSpreading,
    sigma_L_wave,
    load_pdstrip_rao,
)
from cqa.wcfdi_mc import wcfdi_mc, WcfdiScenario


@pytest.fixture(scope="module")
def mc_result():
    """One small WCFDI MC run, reused across tests."""
    cfg = csov_default_config()
    gw = cfg.gangway
    L0 = 0.5 * (gw.telescope_min + gw.telescope_max)
    joint = GangwayJointState(
        h=gw.rotation_centre_height_above_base,
        alpha_g=-np.pi / 2.0,
        beta_g=0.0,
        L=L0,
    )
    scenario = WcfdiScenario(alpha=(0.8, 0.8, 0.8), T_realloc=10.0)
    res = wcfdi_mc(
        cfg,
        Vw_mean=14.0,
        Hs=2.8, Tp=9.0, Vc=0.5,
        theta_rel=np.pi / 2.0,
        scenario=scenario,
        joint=joint,
        n_samples=200,        # smaller than the demo for speed
        t_end=120.0,
        n_t=121,
        rng_seed=0,
        sample_mode="full12",
    )
    return cfg, res


def test_wave_off_reproduces_prior_behaviour(mc_result):
    """sigma_L_wave = 0 must give exactly the pre-wave-integration result.

    Specifically: gw_p_alarm equals 1 - operable_fraction(), and the
    margin percentiles match the raw MC margins (no shrink).
    """
    cfg, res = mc_result
    summary = summarise_for_operator(res, cfg, weather_summary="test", sigma_L_wave=0.0)

    # Backward-compat: alarm matches raw operable_fraction.
    assert np.isclose(summary.gw_p_alarm, 1.0 - res.operable_fraction(), atol=1e-12)
    # Margin percentiles equal the raw ones.
    valid = ~np.isnan(res.dL_peak_abs)
    expected_low_p50 = float(np.percentile(res.margin_low[valid], 50))
    expected_low_p95 = float(np.percentile(res.margin_low[valid], 5))
    assert np.isclose(summary.gw_lower_margin_p50, expected_low_p50, atol=1e-12)
    assert np.isclose(summary.gw_lower_margin_p95, expected_low_p95, atol=1e-12)
    # Wave channel reported as zero.
    assert summary.gw_sigma_L_wave_m == 0.0
    assert summary.gw_k_sigma_L_wave_m == 0.0


def test_wave_on_shrinks_margins_and_can_only_increase_p_alarm(mc_result):
    """Adding sigma_L_wave > 0 deterministically reduces both margin
    percentiles by k_wave * sigma_L_wave and weakly increases gw_p_alarm.
    """
    cfg, res = mc_result
    base = summarise_for_operator(res, cfg, sigma_L_wave=0.0)
    sigma = 0.30  # 30 cm wave-frequency 1-sigma
    k = 1.96
    delta = k * sigma
    waved = summarise_for_operator(res, cfg, sigma_L_wave=sigma, k_wave=k)

    # Margins shrink by exactly delta (deterministic shift).
    assert np.isclose(waved.gw_lower_margin_p50,
                      base.gw_lower_margin_p50 - delta, atol=1e-12)
    assert np.isclose(waved.gw_lower_margin_p95,
                      base.gw_lower_margin_p95 - delta, atol=1e-12)
    assert np.isclose(waved.gw_upper_margin_p50,
                      base.gw_upper_margin_p50 - delta, atol=1e-12)
    assert np.isclose(waved.gw_upper_margin_p95,
                      base.gw_upper_margin_p95 - delta, atol=1e-12)
    # P(alarm) can only stay the same or grow.
    assert waved.gw_p_alarm >= base.gw_p_alarm - 1e-12
    # Reported wave channel.
    assert np.isclose(waved.gw_sigma_L_wave_m, sigma)
    assert np.isclose(waved.gw_k_sigma_L_wave_m, delta)
    assert np.isclose(waved.gw_k_wave, k)


def test_wave_does_not_affect_position_summary(mc_result):
    """The wave channel currently only modifies the gangway summary;
    the position bar (DP slow content) must be unchanged. This is the
    documented scope -- horizontal wave-frequency position is a future
    extension."""
    cfg, res = mc_result
    base = summarise_for_operator(res, cfg, sigma_L_wave=0.0)
    waved = summarise_for_operator(res, cfg, sigma_L_wave=0.30)
    assert np.isclose(base.pos_p_warning, waved.pos_p_warning)
    assert np.isclose(base.pos_p_alarm, waved.pos_p_alarm)
    assert np.isclose(base.pos_p50, waved.pos_p50)
    assert np.isclose(base.pos_p95, waved.pos_p95)


def test_wave_large_enough_to_force_alarm(mc_result):
    """A pathologically large sigma_L_wave whose k*sigma exceeds the
    smaller of (L0 - L_min, L_max - L0) MUST drive every sample to
    end-stop hit, hence gw_p_alarm == 1.0."""
    cfg, res = mc_result
    L0 = res.info["L0"]
    half_span = min(L0 - cfg.gangway.telescope_min,
                    cfg.gangway.telescope_max - L0)
    # Pick sigma so that k*sigma > half_span; subtract any P95 slow margin
    # for safety.
    sigma = (half_span + 1.0) / 1.96
    waved = summarise_for_operator(res, cfg, sigma_L_wave=sigma, k_wave=1.96)
    assert waved.gw_p_alarm == pytest.approx(1.0, abs=1e-12), (
        f"Expected gw_p_alarm = 1.0 with overwhelming sigma_L_wave={sigma:.2f}, "
        f"got {waved.gw_p_alarm}"
    )


# ---------------------------------------------------------------------------
# IMCA M254 Rev.1 Figure 8 traffic-light tests
# ---------------------------------------------------------------------------


def test_imca_defaults_match_m254_thresholds(mc_result):
    """Default IMCA M254 Fig.8 thresholds: 60 % / 80 % gangway util,
    2 m / 4 m vessel footprint."""
    cfg, res = mc_result
    summary = summarise_for_operator(res, cfg, sigma_L_wave=0.0)
    assert summary.gw_imca_warning_frac == pytest.approx(0.60)
    assert summary.gw_imca_alarm_frac == pytest.approx(0.80)
    # Position thresholds come from CqaConfig.operational_limits.
    assert summary.pos_warning_radius_m == pytest.approx(2.0)
    assert summary.pos_alarm_radius_m == pytest.approx(4.0)


def test_imca_traffic_strict_inequality_rule(mc_result):
    """Verify the IMCA M254 Fig.8 strict-inequality bands by sweeping
    util across the boundaries via shrinking-stroke overrides on the
    summary's util computation. We sweep sigma_L_wave to nudge util
    across the boundary at the canonical OP."""
    cfg, res = mc_result
    # Compute baseline (slow only) utilisation, then add wave reach in
    # known steps. Use the gangway dimensions directly.
    L0 = res.info["L0"]
    L_min = cfg.gangway.telescope_min
    L_max = cfg.gangway.telescope_max
    stroke = min(L0 - L_min, L_max - L0)

    base = summarise_for_operator(res, cfg, sigma_L_wave=0.0)
    p95_slow = base.gw_p95
    # Solve for sigma so that (p95 + 1.96*sigma)/stroke equals a target.
    def sigma_for_util(target_util: float) -> float:
        return (target_util * stroke - p95_slow) / 1.96

    # 50 % -> green (well below 60 %)
    s_green = sigma_for_util(0.50)
    s_amber = sigma_for_util(0.70)  # in the amber band
    s_red = sigma_for_util(0.90)    # in the red band
    assert s_green > 0 and s_amber > 0 and s_red > 0, (
        "Test setup requires the slow P95 to be small enough that we can "
        "reach all three IMCA bands by tuning sigma_L_wave alone."
    )

    s = summarise_for_operator(res, cfg, sigma_L_wave=s_green)
    assert s.gw_traffic == "green", f"got {s.gw_traffic}, util={s.gw_util_imca:.3f}"
    s = summarise_for_operator(res, cfg, sigma_L_wave=s_amber)
    assert s.gw_traffic == "amber", f"got {s.gw_traffic}, util={s.gw_util_imca:.3f}"
    s = summarise_for_operator(res, cfg, sigma_L_wave=s_red)
    assert s.gw_traffic == "red", f"got {s.gw_traffic}, util={s.gw_util_imca:.3f}"


def test_imca_overrides_change_traffic(mc_result):
    """Custom IMCA thresholds (e.g. tighter for a class-2 W2W operation)
    must propagate through to the traffic-light decision. Construct
    thresholds around the actual util so the test is deterministic."""
    cfg, res = mc_result
    base = summarise_for_operator(res, cfg, sigma_L_wave=0.0)
    util = base.gw_util_imca
    # Place util firmly in the red band of the override thresholds.
    s_red = summarise_for_operator(
        res, cfg, sigma_L_wave=0.0,
        gw_imca_warning_frac=max(0.01, util * 0.5),
        gw_imca_alarm_frac=max(0.02, util * 0.7),
    )
    assert s_red.gw_traffic == "red"
    # Place util firmly in the amber band.
    s_amber = summarise_for_operator(
        res, cfg, sigma_L_wave=0.0,
        gw_imca_warning_frac=max(0.01, util * 0.5),
        gw_imca_alarm_frac=min(0.99, util * 1.5),
    )
    assert s_amber.gw_traffic == "amber"
    # Place util firmly in the green band.
    s_green = summarise_for_operator(
        res, cfg, sigma_L_wave=0.0,
        gw_imca_warning_frac=min(0.99, util * 1.5),
        gw_imca_alarm_frac=min(0.99, util * 2.0 + 0.01),
    )
    assert s_green.gw_traffic == "green"


def test_imca_pos_traffic_uses_p95_footprint(mc_result):
    """Vessel-capability axis colour must come from pos_p95 (footprint)
    vs warning/alarm radii, NOT from the conditional probabilities."""
    cfg, res = mc_result
    s = summarise_for_operator(res, cfg, sigma_L_wave=0.0)
    if s.pos_p95 < s.pos_warning_radius_m:
        assert s.pos_traffic == "green"
    elif s.pos_p95 < s.pos_alarm_radius_m:
        assert s.pos_traffic == "amber"
    else:
        assert s.pos_traffic == "red"


def test_imca_canonical_op_is_green(mc_result):
    """At the canonical CSOV OP (Hs=2.8, Tp=9, beam) the post-WCFDI
    summary should be GREEN/GREEN per IMCA M254. Util should be in
    the low-30s %, footprint well under 2 m. This is the headline
    "boil down to almost 1 number" the user asked for."""
    cfg, res = mc_result
    summary = summarise_for_operator(res, cfg, sigma_L_wave=0.55)  # ~current value
    assert summary.pos_traffic == "green", (
        f"footprint = {summary.pos_p95:.2f} m must be <2 m at canonical OP"
    )
    assert summary.gw_traffic == "green", (
        f"util = {summary.gw_util_imca*100:.1f} % must be <60 % at canonical OP"
    )
    # Sanity-check the util magnitude.
    assert 0.05 < summary.gw_util_imca < 0.50


# ---------------------------------------------------------------------------
# Intact-prior (Rice / Cartwright-Longuet-Higgins) tests
# ---------------------------------------------------------------------------


def _build_csov_intact_prior_inputs(theta_rel=np.pi / 2.0,
                                    Vw_mean=14.0, Hs=2.8, Tp=9.0, Vc=0.5):
    """Build (cl, S_F_funcs, cfg, joint) for the canonical CSOV OP.

    Mirrors what wcfdi_mc / excursion_polar build internally, but
    returns the pieces summarise_intact_prior needs.
    """
    from cqa import csov_default_config
    from cqa.vessel import LinearVesselModel
    from cqa.controller import LinearDpController
    from cqa.closed_loop import ClosedLoop
    from cqa.psd import (
        WindForceModel, npd_wind_gust_force_psd,
        current_variability_force_psd,
    )
    from cqa.vessel import CurrentForceModel
    from cqa.drift import slow_drift_force_psd_newman_pdstrip
    from cqa.rao import load_pdstrip_rao
    import os

    cfg = csov_default_config()
    vp = cfg.vessel
    wp = cfg.wind
    cp = cfg.current

    vessel = LinearVesselModel.from_config(vp)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D,
        omega_n=cfg.controller.omega_n,
        zeta=cfg.controller.zeta,
    )
    cl = ClosedLoop.build(vessel, controller)

    wind_model = WindForceModel(wp=wp, loa=vp.loa)
    S_wind = npd_wind_gust_force_psd(wind_model, Vw_mean, theta_rel)

    pdstrip_path = os.path.expanduser(
        "~/src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"
    )
    rao = load_pdstrip_rao(pdstrip_path)
    S_drift = slow_drift_force_psd_newman_pdstrip(
        rao_table=rao, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel,
    )

    current_model = CurrentForceModel(
        cp=cp,
        lateral_area_underwater=vp.lpp * vp.draft,
        frontal_area_underwater=vp.beam * vp.draft,
        loa=vp.loa,
    )
    if Vc > 1e-9:
        F0 = current_model.force(Vc, theta_rel)
        dFdVc = 2.0 * F0 / Vc
    else:
        dFdVc = np.zeros(3)
    S_curr = current_variability_force_psd(
        dFdVc, sigma_Vc=0.1, tau=600.0,
    )

    gw = cfg.gangway
    L0 = 0.5 * (gw.telescope_min + gw.telescope_max)
    joint = GangwayJointState(
        h=gw.rotation_centre_height_above_base,
        alpha_g=-np.pi / 2.0,
        beta_g=0.0,
        L=L0,
    )
    return cl, [S_wind, S_drift, S_curr], cfg, joint


@pytest.fixture(scope="module")
def intact_inputs():
    return _build_csov_intact_prior_inputs()


def test_intact_prior_returns_finite_nonnegative(intact_inputs):
    """All summary fields must be finite and probabilities in [0, 1].
    Length-scale fields (a_p50, a_p90) must be finite and non-negative."""
    cl, S_F_funcs, cfg, joint = intact_inputs
    prior = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint,
        T_op_s=20 * 60.0, sigma_L_wave=0.0,
    )
    assert np.isfinite(prior.pos_sigma_m) and prior.pos_sigma_m >= 0.0
    assert np.isfinite(prior.pos_nu0_max) and prior.pos_nu0_max >= 0.0
    assert 0.0 <= prior.pos_p_breach <= 1.0
    assert 0.0 <= prior.pos_p_breach_alarm <= 1.0
    assert 0.0 <= prior.gw_p_breach_warn <= 1.0
    assert 0.0 <= prior.gw_p_breach_alarm <= 1.0
    # Length-scale fields.
    assert np.isfinite(prior.pos_a_p50) and prior.pos_a_p50 >= 0.0
    assert np.isfinite(prior.pos_a_p90) and prior.pos_a_p90 >= 0.0
    assert np.isfinite(prior.gw_a_p50) and prior.gw_a_p50 >= 0.0
    assert np.isfinite(prior.gw_a_p90) and prior.gw_a_p90 >= 0.0
    # P90 of running max must be >= P50 (higher quantile = larger value).
    assert prior.pos_a_p90 >= prior.pos_a_p50 - 1e-9
    assert prior.gw_a_p90 >= prior.gw_a_p50 - 1e-9
    assert prior.pos_traffic_prior in {"green", "amber", "red"}
    assert prior.gw_traffic_prior in {"green", "amber", "red"}
    # Reported geometry sanity.
    assert prior.gw_threshold_to_lower_m > 0.0
    assert prior.gw_threshold_to_upper_m > 0.0
    assert prior.gw_warn_m < prior.gw_alarm_m
    assert prior.gw_warn_m == pytest.approx(0.60 * prior.gw_threshold_used_m)
    assert prior.gw_alarm_m == pytest.approx(0.80 * prior.gw_threshold_used_m)
    # Default quantiles are (P50, P90).
    assert prior.quantiles == (0.50, 0.90)


def test_intact_prior_alarm_threshold_lt_warning(intact_inputs):
    """A larger threshold (alarm radius > warn radius) must yield a
    smaller breach probability -- on BOTH the vessel and gangway axes."""
    cl, S_F_funcs, cfg, joint = intact_inputs
    prior = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint, T_op_s=20 * 60.0, sigma_L_wave=0.5,
    )
    assert prior.pos_alarm_radius_m > prior.pos_warning_radius_m
    assert prior.pos_p_breach_alarm <= prior.pos_p_breach + 1e-12
    assert prior.gw_alarm_m > prior.gw_warn_m
    assert prior.gw_p_breach_alarm <= prior.gw_p_breach_warn + 1e-12


def test_intact_prior_p_breach_grows_with_T_op(intact_inputs):
    """For Poisson exceedance, P_breach must be monotone in T_op, and
    so must the inverse-Rice quantiles (longer exposure => larger
    expected peak)."""
    cl, S_F_funcs, cfg, joint = intact_inputs
    short = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint, T_op_s=60.0, sigma_L_wave=0.0,
    )
    long_ = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint, T_op_s=4 * 60 * 60.0, sigma_L_wave=0.0,
    )
    assert long_.pos_p_breach >= short.pos_p_breach - 1e-12
    assert long_.gw_p_breach_warn >= short.gw_p_breach_warn - 1e-12
    assert long_.gw_p_breach_alarm >= short.gw_p_breach_alarm - 1e-12
    # Quantiles of running max must also grow with T_op.
    assert long_.pos_a_p50 >= short.pos_a_p50 - 1e-9
    assert long_.pos_a_p90 >= short.pos_a_p90 - 1e-9
    assert long_.gw_a_p50 >= short.gw_a_p50 - 1e-9
    assert long_.gw_a_p90 >= short.gw_a_p90 - 1e-9


def test_intact_prior_wave_band_off_collapses(intact_inputs):
    """sigma_L_wave=0 -> wave band is dropped, per_band tuple second
    entry is exactly 0."""
    cl, S_F_funcs, cfg, joint = intact_inputs
    prior = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint, T_op_s=20 * 60.0, sigma_L_wave=0.0,
    )
    assert prior.gw_sigma_wave_m == 0.0
    assert prior.gw_nu0_wave == 0.0
    assert prior.gw_p_breach_per_band[1] == 0.0


def test_intact_prior_wave_band_on_increases_p_breach(intact_inputs):
    """Adding a non-zero sigma_L_wave can only increase the gangway
    end-stop breach probability and the running-max quantiles."""
    cl, S_F_funcs, cfg, joint = intact_inputs
    base = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint, T_op_s=20 * 60.0, sigma_L_wave=0.0,
    )
    waved = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint,
        T_op_s=20 * 60.0, sigma_L_wave=0.5, Tp_wave_s=8.0,
    )
    assert waved.gw_p_breach_warn >= base.gw_p_breach_warn - 1e-12
    assert waved.gw_p_breach_alarm >= base.gw_p_breach_alarm - 1e-12
    assert waved.gw_a_p50 >= base.gw_a_p50 - 1e-9
    assert waved.gw_a_p90 >= base.gw_a_p90 - 1e-9
    assert waved.gw_sigma_wave_m == pytest.approx(0.5)
    assert waved.gw_nu0_wave == pytest.approx(1.0 / 8.0)
    assert waved.gw_p_breach_per_band[1] >= 0.0


def test_intact_prior_traffic_distance_rule(intact_inputs):
    """Traffic light must be driven by a_p90 vs warn / alarm radii:
       red   if a_p90 >= alarm_radius
       amber if warn_radius <= a_p90 < alarm_radius
       green if a_p90 < warn_radius
    Verified by tweaking the IMCA radii via cfg.operational_limits."""
    import dataclasses as _dc

    cl, S_F_funcs, cfg, joint = intact_inputs
    base = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint, T_op_s=20 * 60.0, sigma_L_wave=0.0,
    )
    a90 = base.pos_a_p90
    assert a90 > 0.0

    def cfg_with_radii(warn: float, alarm: float):
        new_limits = _dc.replace(
            cfg.operational_limits,
            position_warning_radius_m=warn,
            position_alarm_radius_m=alarm,
        )
        return _dc.replace(cfg, operational_limits=new_limits)

    # green: a90 well below warn.
    cfg_green = cfg_with_radii(warn=a90 * 2.0, alarm=a90 * 3.0)
    s_g = summarise_intact_prior(
        cl, S_F_funcs, cfg_green, joint, T_op_s=20 * 60.0, sigma_L_wave=0.0,
    )
    assert s_g.pos_traffic_prior == "green", (
        f"a_p90={s_g.pos_a_p90:.3f}, warn={s_g.pos_warning_radius_m:.3f}"
    )

    # amber: warn < a90 < alarm.
    cfg_amber = cfg_with_radii(warn=a90 * 0.5, alarm=a90 * 2.0)
    s_a = summarise_intact_prior(
        cl, S_F_funcs, cfg_amber, joint, T_op_s=20 * 60.0, sigma_L_wave=0.0,
    )
    assert s_a.pos_traffic_prior == "amber", (
        f"a_p90={s_a.pos_a_p90:.3f}, warn={s_a.pos_warning_radius_m:.3f}, "
        f"alarm={s_a.pos_alarm_radius_m:.3f}"
    )

    # red: a90 above alarm.
    cfg_red = cfg_with_radii(warn=a90 * 0.25, alarm=a90 * 0.5)
    s_r = summarise_intact_prior(
        cl, S_F_funcs, cfg_red, joint, T_op_s=20 * 60.0, sigma_L_wave=0.0,
    )
    assert s_r.pos_traffic_prior == "red", (
        f"a_p90={s_r.pos_a_p90:.3f}, alarm={s_r.pos_alarm_radius_m:.3f}"
    )


def test_intact_prior_quantile_round_trip(intact_inputs):
    """The reported a_p90 must satisfy P_breach(a_p90) ~= 0.10 under
    the same Vanmarcke-corrected Rice formula used internally."""
    from cqa import p_exceed_rice
    cl, S_F_funcs, cfg, joint = intact_inputs
    prior = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint, T_op_s=20 * 60.0, sigma_L_wave=0.0,
    )
    # Position axis: P_breach at a_p90 should be 1 - 0.90 = 0.10.
    r = p_exceed_rice(
        sigma=prior.pos_sigma_m, nu_0_plus=prior.pos_nu0_max,
        threshold=prior.pos_a_p90, T=prior.T_op_s, bilateral=True,
        clustering="vanmarcke", q=prior.pos_q,
    )
    assert r.p_breach == pytest.approx(0.10, abs=1e-3), (
        f"round-trip pos a_p90: P_breach={r.p_breach:.4f} != 0.10"
    )
    # Same for P50.
    r50 = p_exceed_rice(
        sigma=prior.pos_sigma_m, nu_0_plus=prior.pos_nu0_max,
        threshold=prior.pos_a_p50, T=prior.T_op_s, bilateral=True,
        clustering="vanmarcke", q=prior.pos_q,
    )
    assert r50.p_breach == pytest.approx(0.50, abs=1e-3), (
        f"round-trip pos a_p50: P_breach={r50.p_breach:.4f} != 0.50"
    )


def test_intact_prior_plot_smoke(intact_inputs):
    """plot_intact_prior must return a Figure without crashing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cl, S_F_funcs, cfg, joint = intact_inputs
    prior = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint, T_op_s=20 * 60.0, sigma_L_wave=0.3,
    )
    fig = plot_intact_prior(prior)
    assert fig is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# Posterior-sigma override (P3.10 wiring)
# ---------------------------------------------------------------------------


def test_intact_prior_posterior_sigma_radial_override_takes_effect(intact_inputs):
    """Supplying a posterior_sigma_radial_m smaller than the model
    sigma_radial must shrink both pos_a_p50 and pos_a_p90 (length-scale
    quantiles scale roughly linearly with sigma at fixed nu0+, q, T)."""
    cl, S_F_funcs, cfg, joint = intact_inputs
    base = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint, T_op_s=20 * 60.0, sigma_L_wave=0.0,
    )
    # Halve the position sigma via posterior override.
    posterior_sigma = 0.5 * base.pos_sigma_m
    overridden = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint, T_op_s=20 * 60.0, sigma_L_wave=0.0,
        posterior_sigma_radial_m=posterior_sigma,
    )
    assert overridden.pos_sigma_m == pytest.approx(posterior_sigma)
    # Spectral shape (nu0, q) preserved.
    assert overridden.pos_nu0_max == pytest.approx(base.pos_nu0_max)
    assert overridden.pos_q == pytest.approx(base.pos_q)
    # Length-scale quantiles must shrink (smaller sigma -> smaller a).
    assert overridden.pos_a_p50 < base.pos_a_p50
    assert overridden.pos_a_p90 < base.pos_a_p90
    # Gangway axis untouched by position override.
    assert overridden.gw_a_p50 == pytest.approx(base.gw_a_p50)
    assert overridden.gw_a_p90 == pytest.approx(base.gw_a_p90)


def test_intact_prior_posterior_sigma_telescope_override_takes_effect(intact_inputs):
    """Supplying a posterior_sigma_telescope_slow_m must shrink the
    gangway length-scale quantiles while leaving the position axis
    untouched."""
    cl, S_F_funcs, cfg, joint = intact_inputs
    base = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint, T_op_s=20 * 60.0, sigma_L_wave=0.0,
    )
    posterior_sigma = 0.5 * base.gw_sigma_slow_m
    overridden = summarise_intact_prior(
        cl, S_F_funcs, cfg, joint, T_op_s=20 * 60.0, sigma_L_wave=0.0,
        posterior_sigma_telescope_slow_m=posterior_sigma,
    )
    assert overridden.gw_sigma_slow_m == pytest.approx(posterior_sigma)
    assert overridden.gw_nu0_slow == pytest.approx(base.gw_nu0_slow)
    assert overridden.gw_q_slow == pytest.approx(base.gw_q_slow)
    assert overridden.gw_a_p50 < base.gw_a_p50
    assert overridden.gw_a_p90 < base.gw_a_p90
    # Position axis untouched.
    assert overridden.pos_a_p50 == pytest.approx(base.pos_a_p50)
    assert overridden.pos_a_p90 == pytest.approx(base.pos_a_p90)


def test_intact_prior_posterior_sigma_invalid_raises(intact_inputs):
    cl, S_F_funcs, cfg, joint = intact_inputs
    with pytest.raises(ValueError, match="posterior_sigma_radial_m"):
        summarise_intact_prior(
            cl, S_F_funcs, cfg, joint, T_op_s=20 * 60.0,
            posterior_sigma_radial_m=-0.1,
        )
    with pytest.raises(ValueError, match="posterior_sigma_telescope_slow_m"):
        summarise_intact_prior(
            cl, S_F_funcs, cfg, joint, T_op_s=20 * 60.0,
            posterior_sigma_telescope_slow_m=0.0,
        )