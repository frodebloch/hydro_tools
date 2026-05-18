"""Regression tests for the regime-B saturation integration in
``summarise_for_operator_live`` (sec.12.21.21.22).

Mirrors the design verified in tests/test_live_regime_b.py but
exercises the wiring through the operator panel:

* When ``obs_state.tau_buffer`` or ``cap_residual_N_Nm`` is missing,
  the regime-B fields are absent / zero and ``overall_traffic`` is
  unchanged.
* When both are provided and the buffer sits comfortably below the
  cap, ``regime_b_traffic == "green"`` and ``overall_traffic`` is at
  most as bad as before.
* When the buffer is large relative to the cap, ``regime_b_traffic``
  flips to ``"amber"`` / ``"red"`` and ``overall_traffic`` reflects
  the worst-of rule.
* Sanity: the (mu, sigma, p_sat) reported by the panel match a direct
  call to ``estimate_regime_b_severity`` on the same buffer.

These tests are pure-Python (no brucon data) so the buffer is built
analytically from a constant-mean Gaussian, then propagated through
the API.
"""

from __future__ import annotations

import numpy as np
import pytest

from cqa import summarise_for_operator_live
from cqa.live_regime_b import estimate_regime_b_severity
from cqa.live_decision import LiveObserverState

from _live_fixtures import _make_sigma_post, _make_obs_state, _config_with_K


# CSOV residual polytope per sec.12.21.21.15 (kN, kN, kNm -> N, N, Nm).
_CAP_RESIDUAL_N_NM = (838.0e3, 1104.0e3, 47929.0e3)


def _obs_with_tau_buffer(
    *,
    mu_Ty_kN: float = 0.0,
    sigma_Ty_kN: float = 50.0,
    n_seconds: float = 400.0,
    fs_hz: float = 1.0,
    seed: int = 0,
) -> LiveObserverState:
    """Build a LiveObserverState with a synthetic Gaussian thrust
    buffer centred on ``(0, mu_Ty_kN, 0)`` with per-DOF sigma scaled
    from ``sigma_Ty_kN`` (sway gets it directly, surge and yaw get
    proportional values to make the dataclass non-degenerate).
    """
    rng = np.random.default_rng(seed)
    n = int(round(n_seconds * fs_hz))
    mu = np.array([0.0, mu_Ty_kN * 1e3, 0.0])
    sig = np.array([sigma_Ty_kN * 1e3 * 0.4,
                    sigma_Ty_kN * 1e3,
                    sigma_Ty_kN * 1e3 * 5e2])
    tau = mu[None, :] + sig[None, :] * rng.standard_normal((n, 3))
    return LiveObserverState(
        eta_hat=np.zeros(3), nu_hat=np.zeros(3),
        b_hat=np.zeros(3), eta_wave=np.zeros(3),
        heading_compass=0.0,
        tau_buffer=tau, tau_buffer_fs_hz=fs_hz,
    )


def test_regime_b_absent_when_buffer_not_supplied():
    cfg = _config_with_K(0.0)
    obs = _make_obs_state(b_hat_kN=(0.0, 0.0, 0.0))
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(cfg, obs, sigma)
    assert s.regime_b_present is False
    assert s.regime_b_severity == 0.0
    assert s.regime_b_traffic == "green"
    assert s.regime_b_p_sat is None


def test_regime_b_absent_when_cap_not_supplied():
    cfg = _config_with_K(0.0)
    obs = _obs_with_tau_buffer(mu_Ty_kN=500.0, sigma_Ty_kN=90.0)
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(cfg, obs, sigma)
    # tau_buffer supplied but cap missing -> regime-B suppressed.
    assert s.regime_b_present is False


def test_regime_b_green_with_large_headroom():
    """bf8_q10_w45-like settings: mu_Ty = 515, sigma_Ty = 90, cap = 1104.
    Expected: z = (1104 - 515) / 90 = 6.5 -> P_sat ~ 4e-11 -> green.
    """
    cfg = _config_with_K(0.0)
    obs = _obs_with_tau_buffer(mu_Ty_kN=515.0, sigma_Ty_kN=90.0, seed=42)
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(
        cfg, obs, sigma, cap_residual_N_Nm=_CAP_RESIDUAL_N_NM,
    )
    assert s.regime_b_present is True
    assert s.regime_b_traffic == "green"
    assert s.regime_b_severity < 0.01  # IMCA green threshold
    # Mu/sigma reported back in Newtons.
    assert s.regime_b_mu_N_Nm.shape == (3,)
    # Sway DOF mean: synthetic Gaussian noise sampled at 1 Hz so the
    # sample mean is mu +/- sigma/sqrt(N); allow a wide tolerance.
    assert abs(s.regime_b_mu_N_Nm[1] - 515.0e3) < 30.0e3


def test_regime_b_red_when_mu_above_cap():
    """Constructed mu well above cap -> P_sat dominates -> red."""
    cfg = _config_with_K(0.0)
    obs = _obs_with_tau_buffer(mu_Ty_kN=1200.0, sigma_Ty_kN=80.0, seed=11)
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(
        cfg, obs, sigma, cap_residual_N_Nm=_CAP_RESIDUAL_N_NM,
    )
    assert s.regime_b_present is True
    assert s.regime_b_traffic == "red"
    assert s.regime_b_severity > 0.5  # most of the time above cap
    # Overall traffic must reflect the worst of intact / wcf / regime-B.
    assert s.overall_traffic == "red"


def test_regime_b_amber_band():
    """Place mu/sigma so P_sat lands in [0.01, 0.10) -> amber.
    For one-sided Gaussian, P(X > cap) = 0.05 at z = 1.645.
    With cap = 1104 kN and sigma = 200 kN, that needs mu = 1104 - 329 = 775 kN.
    """
    cfg = _config_with_K(0.0)
    obs = _obs_with_tau_buffer(mu_Ty_kN=775.0, sigma_Ty_kN=200.0, seed=7)
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(
        cfg, obs, sigma, cap_residual_N_Nm=_CAP_RESIDUAL_N_NM,
    )
    assert s.regime_b_present is True
    assert s.regime_b_traffic == "amber"
    assert 0.01 <= s.regime_b_severity < 0.10


def test_regime_b_consistent_with_direct_call():
    """Wiring sanity: the values the panel reports equal the direct call."""
    cfg = _config_with_K(0.0)
    obs = _obs_with_tau_buffer(mu_Ty_kN=400.0, sigma_Ty_kN=120.0, seed=99)
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(
        cfg, obs, sigma, cap_residual_N_Nm=_CAP_RESIDUAL_N_NM,
    )
    direct = estimate_regime_b_severity(
        tau_buffer=obs.tau_buffer,
        fs_hz=obs.tau_buffer_fs_hz,
        cap_residual=_CAP_RESIDUAL_N_NM,
    )
    np.testing.assert_allclose(s.regime_b_p_sat, direct.p_sat)
    np.testing.assert_allclose(s.regime_b_mu_N_Nm, direct.mu)
    np.testing.assert_allclose(s.regime_b_sigma_N_Nm, direct.sigma)
    assert s.regime_b_severity == pytest.approx(direct.severity)
    assert s.regime_b_traffic == direct.traffic


def test_regime_b_does_not_downgrade_overall_traffic():
    """If intact bar is amber and regime-B is green, overall is at least amber."""
    cfg = _config_with_K(0.0)
    # Big b_hat to push WCF axis amber/red without provoking regime-B.
    obs_state = _make_obs_state(b_hat_kN=(-200.0, -200.0, -1000.0))
    # Add a small benign thrust buffer (sigma small, mu small => p_sat ~ 0).
    rng = np.random.default_rng(0)
    tau = 1e3 * rng.standard_normal((400, 3)) * np.array([10.0, 10.0, 100.0])
    obs = LiveObserverState(
        eta_hat=obs_state.eta_hat, nu_hat=obs_state.nu_hat,
        b_hat=obs_state.b_hat, eta_wave=obs_state.eta_wave,
        heading_compass=obs_state.heading_compass,
        tau_buffer=tau, tau_buffer_fs_hz=1.0,
    )
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(
        cfg, obs, sigma, cap_residual_N_Nm=_CAP_RESIDUAL_N_NM,
    )
    assert s.regime_b_present is True
    assert s.regime_b_traffic == "green"
    # Overall must equal worst(intact, wcf) since regime-B is green.
    expected = s.intact_traffic
    for x in (s.wcf_traffic, s.regime_b_traffic):
        order = {"green": 0, "amber": 1, "red": 2}
        if order[x] > order[expected]:
            expected = x
    assert s.overall_traffic == expected


def test_obs_state_invariant_rejects_mismatched_buffer_args():
    """LiveObserverState validation: must supply both tau_buffer and
    tau_buffer_fs_hz or neither."""
    with pytest.raises(ValueError):
        LiveObserverState(
            eta_hat=np.zeros(3), nu_hat=np.zeros(3),
            b_hat=np.zeros(3), eta_wave=np.zeros(3),
            heading_compass=0.0,
            tau_buffer=np.zeros((10, 3)),
            tau_buffer_fs_hz=None,
        )
    with pytest.raises(ValueError):
        LiveObserverState(
            eta_hat=np.zeros(3), nu_hat=np.zeros(3),
            b_hat=np.zeros(3), eta_wave=np.zeros(3),
            heading_compass=0.0,
            tau_buffer=None, tau_buffer_fs_hz=1.0,
        )


def test_obs_state_invariant_rejects_bad_buffer_shape():
    with pytest.raises(ValueError):
        LiveObserverState(
            eta_hat=np.zeros(3), nu_hat=np.zeros(3),
            b_hat=np.zeros(3), eta_wave=np.zeros(3),
            heading_compass=0.0,
            tau_buffer=np.zeros((10, 2)),  # wrong DOF count
            tau_buffer_fs_hz=1.0,
        )


# ---------------------------------------------------------------------------
# Option-2 post-WCF excursion-distribution wiring (sec.12.21.21.29)
# ---------------------------------------------------------------------------
def test_wcf_excursion_absent_when_regime_b_absent():
    """No tau_buffer / cap -> Option-2 fields are absent / zero."""
    cfg = _config_with_K(0.0)
    obs = _make_obs_state(b_hat_kN=(0.0, 0.0, 0.0))
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(cfg, obs, sigma)
    assert s.regime_b_present is False
    assert s.wcf_excur_present is False
    assert s.wcf_excur_R_xy_p95_m == 0.0
    assert s.wcf_excur_mu_eta_m_rad is None


def test_wcf_excursion_present_when_regime_b_present():
    """tau_buffer + cap -> Option-2 fields populated, finite, sane shape."""
    cfg = _config_with_K(0.0)
    obs = _obs_with_tau_buffer(mu_Ty_kN=775.0, sigma_Ty_kN=200.0, seed=3)
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(
        cfg, obs, sigma, cap_residual_N_Nm=_CAP_RESIDUAL_N_NM,
    )
    assert s.regime_b_present is True
    assert s.wcf_excur_present is True
    for arr in (s.wcf_excur_mu_eta_m_rad,
                s.wcf_excur_sigma_eta_m_rad,
                s.wcf_excur_eta_p95_m_rad,
                s.wcf_excur_eta_p95_with_offset_m_rad):
        assert arr.shape == (3,)
        assert np.all(np.isfinite(arr))
    # Magnitudes make sense for an amber-saturated cell.
    assert s.wcf_excur_sigma_eta_m_rad[1] > 0.0       # sway has variance
    assert s.wcf_excur_R_xy_p95_m > 0.0               # nontrivial headline
    assert np.all(s.wcf_excur_eta_p95_with_offset_m_rad
                  >= s.wcf_excur_eta_p95_m_rad - 1e-12)


def test_wcf_excursion_grows_with_saturation_severity():
    """Worse Regime-B cell -> larger xy-radial P95 excursion headline."""
    cfg = _config_with_K(0.0)
    sigma = _make_sigma_post()
    obs_green = _obs_with_tau_buffer(mu_Ty_kN=400.0, sigma_Ty_kN=80.0, seed=21)
    obs_red = _obs_with_tau_buffer(mu_Ty_kN=1100.0, sigma_Ty_kN=200.0, seed=21)
    s_green = summarise_for_operator_live(
        cfg, obs_green, sigma, cap_residual_N_Nm=_CAP_RESIDUAL_N_NM,
    )
    s_red = summarise_for_operator_live(
        cfg, obs_red, sigma, cap_residual_N_Nm=_CAP_RESIDUAL_N_NM,
    )
    # Severity ordering invariant.
    assert s_red.regime_b_severity > s_green.regime_b_severity
    # Headline xy excursion P95 should follow.
    assert s_red.wcf_excur_R_xy_p95_m > s_green.wcf_excur_R_xy_p95_m


def test_wcf_excursion_does_not_alter_overall_traffic():
    """Option-2 outputs are surfaced for validation but must NOT yet
    drive overall_traffic (gating decision deferred pending 12-cell
    brucon roll-up)."""
    cfg = _config_with_K(0.0)
    obs = _obs_with_tau_buffer(mu_Ty_kN=1100.0, sigma_Ty_kN=200.0, seed=5)
    sigma = _make_sigma_post()
    s = summarise_for_operator_live(
        cfg, obs, sigma, cap_residual_N_Nm=_CAP_RESIDUAL_N_NM,
    )
    # Overall must equal worst of intact / wcf / regime_b alone (the
    # current panel rule pre-Option-2).
    order = {"green": 0, "amber": 1, "red": 2}
    expected = max(
        (s.intact_traffic, s.wcf_traffic, s.regime_b_traffic),
        key=lambda t: order[t],
    )
    assert s.overall_traffic == expected
