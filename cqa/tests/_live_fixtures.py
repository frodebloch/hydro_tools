"""Shared helpers for live-pipeline tests.

Private (underscore-prefixed) module imported by:
  - tests/test_live_decision.py     (lift-coupling wiring regression)
  - tests/test_live_operator_view.py (operator-facing two-bar panel)

These helpers build minimally-valid LiveObserverState / LiveSigmaPosterior
/ CqaConfig fixtures so each test stays focused on the behaviour it
exercises rather than on dataclass plumbing. Not pytest fixtures (plain
callables) so a test can compose / parameterise them inline.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from cqa import (
    csov_default_config,
    GangwayJointState,
    LiveObserverState,
    LiveSigmaPosterior,
)
from cqa.online_estimator import (
    SigmaPosterior, RadialPosterior, ValidityBadge,
)


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


def _make_obs_state(
    b_hat_kN: tuple[float, float, float],
    eta_hat: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> LiveObserverState:
    """Live observer state with the requested eta_hat (in m, m, rad)
    and b_hat (in kN/kN/kNm)."""
    return LiveObserverState(
        eta_hat=np.array(eta_hat, dtype=float),
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


def _joint() -> GangwayJointState:
    return GangwayJointState(h=15.0, alpha_g=0.0, beta_g=0.0, L=25.0)
