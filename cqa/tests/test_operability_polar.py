"""Tests for cqa.sea_state_relations and cqa.operability_polar."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cqa import csov_default_config, GangwayJointState
from cqa.sea_state_relations import (
    G_STD,
    PM_HS_COEFF,
    PM_OMEGA_P_COEFF,
    pm_hs_from_vw,
    pm_tp_from_vw,
    pm_sea_state,
)
from cqa.operability_polar import (
    OperabilityPolar,
    operability_polar,
    plot_operability_polar,
)
from cqa.operability_polar import _bisect_boundary


# ---------------------------------------------------------------------------
# sea_state_relations
# ---------------------------------------------------------------------------


def test_pm_hs_at_15ms_matches_closed_form():
    """At U_10 = 15 m/s: Hs = 0.21 * 225 / 9.81 = 4.81 m."""
    Hs = pm_hs_from_vw(15.0)
    expected = PM_HS_COEFF * 15.0 ** 2 / G_STD
    assert Hs == pytest.approx(expected)
    assert Hs == pytest.approx(4.817, abs=1e-3)


def test_pm_tp_at_15ms_matches_closed_form():
    """T_p = 2 pi V / (0.877 g). At V=15: ~10.96 s."""
    Tp = pm_tp_from_vw(15.0)
    expected = 2.0 * math.pi * 15.0 / (PM_OMEGA_P_COEFF * G_STD)
    assert Tp == pytest.approx(expected)
    assert Tp == pytest.approx(10.96, abs=0.05)


def test_pm_zero_wind_edge_cases():
    assert pm_hs_from_vw(0.0) == 0.0
    assert pm_tp_from_vw(0.0) == float("inf")
    s = pm_sea_state(0.0)
    assert s.Hs_m == 0.0
    assert s.Tp_s == float("inf")
    assert s.omega_p == 0.0


def test_pm_negative_wind_raises():
    with pytest.raises(ValueError):
        pm_hs_from_vw(-1.0)
    with pytest.raises(ValueError):
        pm_tp_from_vw(-1.0)
    with pytest.raises(ValueError):
        pm_sea_state(-0.1)


def test_pm_sea_state_bundle_consistency():
    s = pm_sea_state(20.0)
    assert s.Vw_m_s == pytest.approx(20.0)
    assert s.Hs_m == pytest.approx(pm_hs_from_vw(20.0))
    assert s.Tp_s == pytest.approx(pm_tp_from_vw(20.0))
    assert s.omega_p == pytest.approx(2.0 * math.pi / s.Tp_s)


def test_pm_hs_quadratic_in_vw():
    """Doubling V_w must quadruple Hs."""
    assert pm_hs_from_vw(20.0) == pytest.approx(4.0 * pm_hs_from_vw(10.0))


def test_pm_tp_linear_in_vw():
    """Tp is linear in V_w."""
    assert pm_tp_from_vw(20.0) == pytest.approx(2.0 * pm_tp_from_vw(10.0))


# ---------------------------------------------------------------------------
# _bisect_boundary
# ---------------------------------------------------------------------------


def test_bisect_finds_known_root():
    """Find smallest x >= sqrt(2) with x**2 >= 2 in [0, 5]."""
    v, lo, hi = _bisect_boundary(lambda x: x * x, threshold=2.0,
                                 Vw_lo=0.0, Vw_hi=5.0, tol=1e-6)
    assert not lo and not hi
    assert v == pytest.approx(math.sqrt(2.0), abs=1e-3)


def test_bisect_capped_low():
    """If metric(Vw_lo) already exceeds threshold, return Vw_lo + flag."""
    v, lo, hi = _bisect_boundary(lambda x: 100.0, threshold=1.0,
                                 Vw_lo=2.0, Vw_hi=10.0)
    assert lo and not hi
    assert v == 2.0


def test_bisect_capped_high():
    """If metric never crosses threshold within bracket, return Vw_hi + flag."""
    v, lo, hi = _bisect_boundary(lambda x: 0.01, threshold=1.0,
                                 Vw_lo=0.5, Vw_hi=10.0)
    assert hi and not lo
    assert v == 10.0


# ---------------------------------------------------------------------------
# operability_polar driver
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def csov_polar_inputs():
    cfg = csov_default_config()
    gw = cfg.gangway
    L0 = 0.5 * (gw.telescope_min + gw.telescope_max)
    joint = GangwayJointState(
        h=gw.rotation_centre_height_above_base,
        alpha_g=-np.pi / 2.0,
        beta_g=0.0,
        L=L0,
    )
    return cfg, joint


@pytest.fixture(scope="module")
def csov_polar_small(csov_polar_inputs):
    """A small, fast polar (8 directions, coarse Vw tolerance) shared
    across the structural / smoke tests so we pay the bisection cost
    only once."""
    cfg, joint = csov_polar_inputs
    return operability_polar(
        cfg, joint,
        n_directions=8,
        Vw_min=2.0,
        Vw_max=25.0,
        Vc_m_s=0.5,
        T_op_s=20.0 * 60.0,
        bisect_tol_m_s=1.0,  # coarse for speed
    )


def test_polar_validation_errors(csov_polar_inputs):
    cfg, joint = csov_polar_inputs
    with pytest.raises(ValueError, match="Vw_min"):
        operability_polar(cfg, joint, Vw_min=0.0, Vw_max=20.0)
    with pytest.raises(ValueError, match="Vw_max"):
        operability_polar(cfg, joint, Vw_min=10.0, Vw_max=5.0)
    with pytest.raises(ValueError, match="quantile_p"):
        operability_polar(cfg, joint, quantile_p=0.0)
    with pytest.raises(ValueError, match="quantile_p"):
        operability_polar(cfg, joint, quantile_p=1.0)
    with pytest.raises(ValueError, match="n_directions"):
        operability_polar(cfg, joint, n_directions=2)


def test_polar_shapes_and_metadata(csov_polar_small):
    p = csov_polar_small
    assert isinstance(p, OperabilityPolar)
    n = len(p.theta_rel_rad)
    assert n == 8
    for arr in (p.pos_warn_Vw, p.pos_alarm_Vw, p.gw_warn_Vw, p.gw_alarm_Vw):
        assert arr.shape == (n,)
        assert np.all(np.isfinite(arr))
    for arr in (p.pos_warn_capped_low, p.pos_warn_capped_high,
                p.pos_alarm_capped_low, p.pos_alarm_capped_high,
                p.gw_warn_capped_low, p.gw_warn_capped_high,
                p.gw_alarm_capped_low, p.gw_alarm_capped_high):
        assert arr.dtype == bool
        assert arr.shape == (n,)
    # IMCA metadata propagated.
    assert p.pos_warn_radius_m == 2.0
    assert p.pos_alarm_radius_m == 4.0
    # Sweep metadata propagated.
    assert p.Vw_min == 2.0
    assert p.Vw_max == 25.0
    assert p.Vc_m_s == 0.5
    assert p.quantile_p == 0.90


def test_polar_alarm_boundary_geq_warn(csov_polar_small):
    """Alarm threshold (4 m / 80%) is larger than warn (2 m / 60%).
    With a monotone metric, the V_w required to hit alarm must be >=
    the V_w required to hit warn -- per direction. Allow saturation
    cases where both are pinned to Vw_min or Vw_max."""
    p = csov_polar_small
    for i in range(len(p.theta_rel_rad)):
        # Position axis.
        if not (p.pos_warn_capped_low[i] or p.pos_alarm_capped_high[i]):
            assert p.pos_alarm_Vw[i] >= p.pos_warn_Vw[i] - 1e-6, (
                f"dir {i}: alarm Vw {p.pos_alarm_Vw[i]} < warn Vw "
                f"{p.pos_warn_Vw[i]} (pos)"
            )
        # Gangway axis.
        if not (p.gw_warn_capped_low[i] or p.gw_alarm_capped_high[i]):
            assert p.gw_alarm_Vw[i] >= p.gw_warn_Vw[i] - 1e-6, (
                f"dir {i}: alarm Vw {p.gw_alarm_Vw[i]} < warn Vw "
                f"{p.gw_warn_Vw[i]} (gw)"
            )


def test_polar_boundaries_within_sweep_range(csov_polar_small):
    p = csov_polar_small
    for arr in (p.pos_warn_Vw, p.pos_alarm_Vw, p.gw_warn_Vw, p.gw_alarm_Vw):
        assert np.all(arr >= p.Vw_min - 1e-9)
        assert np.all(arr <= p.Vw_max + 1e-9)


def test_polar_thetas_uniform_in_two_pi(csov_polar_small):
    p = csov_polar_small
    th = p.theta_rel_rad
    # Uniformly spaced in [0, 2*pi).
    expected = np.linspace(0.0, 2.0 * np.pi, len(th), endpoint=False)
    np.testing.assert_allclose(th, expected)


def test_polar_plot_smoke(csov_polar_small):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plot_operability_polar(csov_polar_small)
    assert fig is not None
    # Two polar axes.
    assert len(fig.axes) >= 2
    plt.close(fig)
