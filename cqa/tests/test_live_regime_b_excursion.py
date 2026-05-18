"""Unit tests for Option 2 post-WCF excursion machinery in cqa.live_regime_b.

Coverage so far:
* clipped_gaussian_moments: analytical limits + Monte Carlo cross-check.
* gauss_markov_psd: variance area sanity + analytical sanity.
* eta_frequency_response: DC vs -A^{-1} B_lost and a high-frequency
  asymptote (decay).
* eta_psd_from_dtau + estimate_post_wcf_excursion_distribution:
  Parseval against a closed-loop time-domain Monte Carlo.
"""
from __future__ import annotations

import numpy as np
import pytest

from cqa.config import csov_default_config
from cqa.controller import LinearDpController
from cqa.live_regime_b import (
    clipped_gaussian_moments,
    estimate_post_wcf_excursion_distribution,
    eta_frequency_response,
    eta_psd_from_dtau,
    gauss_markov_psd,
)
from cqa.transient_obs import (
    IDX_ETA,
    build_observer_augmented_system_full,
)
from cqa.vessel import LinearVesselModel


def _mc_moments(mu, sigma, cap, n=2_000_000, rng_seed=0):
    """Monte Carlo reference for the spillover moments."""
    rng = np.random.default_rng(rng_seed)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cap = np.asarray(cap, dtype=float)
    out_mu = np.zeros_like(mu)
    out_sigma = np.zeros_like(mu)
    for i in range(mu.size):
        X = rng.normal(mu[i], max(sigma[i], 1e-12), size=n)
        Yp = np.where(X > cap[i], X - cap[i], 0.0)
        Ym = np.where(X < -cap[i], X + cap[i], 0.0)
        dtau = Yp + Ym
        out_mu[i] = dtau.mean()
        out_sigma[i] = dtau.std()
    return out_mu, out_sigma


def test_clipped_gaussian_moments_no_clipping_returns_zero():
    # cap >> sigma: spillover should be ~0 in both moments.
    mu = np.array([0.0, 0.0, 0.0])
    sigma = np.array([1.0, 1.0, 1.0])
    cap = np.array([10.0, 10.0, 10.0])  # 10 sigma -> P(|X|>cap) < 1e-23
    mu_d, sigma_d = clipped_gaussian_moments(mu, sigma, cap)
    np.testing.assert_allclose(mu_d, [0.0, 0.0, 0.0], atol=1e-18)
    np.testing.assert_allclose(sigma_d, [0.0, 0.0, 0.0], atol=1e-18)


def test_clipped_gaussian_moments_zero_cap_returns_underlying_moments():
    # cap == 0: delta_tau == X identically, so moments collapse to the
    # underlying Gaussian moments.
    mu = np.array([1.0, -2.0, 3.0])
    sigma = np.array([0.5, 1.5, 4.0])
    cap = np.array([0.0, 0.0, 0.0])
    mu_d, sigma_d = clipped_gaussian_moments(mu, sigma, cap)
    np.testing.assert_allclose(mu_d, mu, atol=1e-12)
    np.testing.assert_allclose(sigma_d, sigma, atol=1e-12)


def test_clipped_gaussian_moments_symmetric_zero_mean_has_zero_mean_spillover():
    # mu = 0, symmetric +-cap: spillover mean = 0 by symmetry.
    mu = np.array([0.0, 0.0, 0.0])
    sigma = np.array([1.0, 1.0, 1.0])
    cap = np.array([0.5, 1.0, 2.0])
    mu_d, sigma_d = clipped_gaussian_moments(mu, sigma, cap)
    np.testing.assert_allclose(mu_d, [0.0, 0.0, 0.0], atol=1e-12)
    # sigma_d should decrease as cap increases (less is "spilled").
    assert sigma_d[0] > sigma_d[1] > sigma_d[2] > 0.0


def test_clipped_gaussian_moments_matches_mc_on_typical_dp_case():
    # Active-saturation regime: cap chosen ~ |mu| + 1.5..2 sigma so the
    # spillover tail is ~1-7% (representative of a Regime-B DP cell where
    # the WCF axis actually has signal). Caps farther than ~3 sigma from
    # the mean give analytical spillover < 1e-9 of sigma and MC sample-error
    # dominates, so they make a poor MC validation case.
    # Layout: surge biased near cap on the + side; sway biased on - side;
    # yaw mildly saturated symmetric.
    mu = np.array([800.0e3, -500.0e3, 5.0e6])     # surge, sway, yaw [N, N, N*m]
    sigma = np.array([300.0e3, 250.0e3, 8.0e6])
    cap = np.array([1200.0e3, 900.0e3, 15.0e6])
    mu_d, sigma_d = clipped_gaussian_moments(mu, sigma, cap)
    mu_d_mc, sigma_d_mc = _mc_moments(mu, sigma, cap, n=2_000_000, rng_seed=42)
    # Sanity: MC must actually see spillover for the comparison to be meaningful.
    assert np.all(sigma_d_mc > 1.0), (
        f"test setup broken: MC sigma too small to validate, got {sigma_d_mc}"
    )
    # MC sample-error budget ~ 3 sigma_d / sqrt(N) ~ 0.2% with N=2e6;
    # rel-tol 5% is comfortable.
    rtol = 0.05
    atol_mean = 0.02 * np.abs(sigma_d_mc).max()
    np.testing.assert_allclose(mu_d, mu_d_mc, rtol=rtol, atol=atol_mean)
    np.testing.assert_allclose(sigma_d, sigma_d_mc, rtol=rtol)


def test_clipped_gaussian_moments_biased_mean_above_cap():
    # mu = +2 sigma, cap = +1 sigma: most mass spills out to the positive
    # side, very little on the negative tail; mean spillover should be
    # positive and approach (mu - cap) for very tight clip.
    mu = np.array([2.0])
    sigma = np.array([1.0])
    cap = np.array([1.0])
    mu_d, sigma_d = clipped_gaussian_moments(mu, sigma, cap)
    # Analytic check: cap-=mu => upper-tail prob ~ Phi(-(cap-mu)/sigma)
    # = Phi(1) ~ 0.84. E[X|X>cap]*P ~ mu (1-F(a)) + sigma f(a) with
    # a = -1, so ~ 2*0.84 + 1*0.242 = 1.92. Minus cap*P = 0.84.
    # => E[Y_+] ~ 1.08. Lower-tail negligible (b = -3 -> F(-3) ~ 0.001).
    # Expected E[delta_tau] ~ 1.08.
    assert 0.9 < mu_d[0] < 1.2
    # Cross-validate with MC.
    mu_d_mc, sigma_d_mc = _mc_moments(mu, sigma, cap, n=2_000_000, rng_seed=7)
    np.testing.assert_allclose(mu_d, mu_d_mc, rtol=0.02)
    np.testing.assert_allclose(sigma_d, sigma_d_mc, rtol=0.02)


def test_clipped_gaussian_moments_input_validation():
    with pytest.raises(ValueError, match="share shape"):
        clipped_gaussian_moments(np.zeros(3), np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="sigma must be non-negative"):
        clipped_gaussian_moments(np.zeros(3), -np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match="cap must be non-negative"):
        clipped_gaussian_moments(np.zeros(3), np.ones(3), -np.ones(3))


def test_clipped_gaussian_moments_degenerate_zero_sigma():
    # sigma = 0: spillover is deterministic, variance is exactly zero.
    mu = np.array([0.5, 2.0, -3.0])
    sigma = np.array([0.0, 0.0, 0.0])
    cap = np.array([1.0, 1.0, 1.0])
    mu_d, sigma_d = clipped_gaussian_moments(mu, sigma, cap)
    # mu=0.5 < cap -> no spillover. mu=2 > cap -> spillover = 1.
    # mu=-3 < -cap -> spillover = -2.
    np.testing.assert_allclose(mu_d, [0.0, 1.0, -2.0])
    np.testing.assert_allclose(sigma_d, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Fixtures (shared with test_transient_obs.py style)
# ---------------------------------------------------------------------------
def _build_csov_aug():
    cfg = csov_default_config()
    vessel = LinearVesselModel.from_config(cfg.vessel)
    omega_n = np.array([0.060, 0.080, 0.120])
    zeta = np.array([0.95, 0.95, 0.95])
    ctrl = LinearDpController.from_bandwidth(
        vessel.M, vessel.D, omega_n=omega_n, zeta=zeta
    )
    return build_observer_augmented_system_full(vessel, ctrl, T_thr=5.0)


# ---------------------------------------------------------------------------
# gauss_markov_psd
# ---------------------------------------------------------------------------
def test_gauss_markov_psd_area_equals_variance():
    # int_0^inf S(omega) domega = sigma^2 by construction. The integral
    # truncated at omega_c = K/tau captures sigma^2 * (2/pi) * atan(K),
    # so K=1000 gives a truncation budget of 2*atan(1/1000)/pi ~ 6.4e-4.
    sigma = 3.0
    tau = 12.0
    K = 1000.0
    omega = np.linspace(0.0, K / tau, 16384)
    S = gauss_markov_psd(sigma, tau, omega)
    var = np.trapezoid(S, omega)
    np.testing.assert_allclose(var, sigma ** 2, rtol=1e-3)


def test_gauss_markov_psd_dc_value():
    # S(0) = 2 sigma^2 tau / pi.
    sigma = 2.5
    tau = 8.0
    S0 = gauss_markov_psd(sigma, tau, np.array([0.0]))[0]
    np.testing.assert_allclose(S0, 2.0 * sigma ** 2 * tau / np.pi, rtol=1e-12)


def test_gauss_markov_psd_input_validation():
    with pytest.raises(ValueError, match="sigma must be >= 0"):
        gauss_markov_psd(-1.0, 10.0, np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="tau_corr must be > 0"):
        gauss_markov_psd(1.0, 0.0, np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="omega must be non-negative"):
        gauss_markov_psd(1.0, 10.0, np.array([-0.1, 1.0]))


# ---------------------------------------------------------------------------
# eta_frequency_response
# ---------------------------------------------------------------------------
def test_eta_frequency_response_dc_matches_static_solve():
    # At omega = 0: H(0) = e_eta^T * (-A^{-1}) * B_lost[:, j].
    aug = _build_csov_aug()
    for j in range(3):
        b = aug.B_lost[:, j]
        x_static = -np.linalg.solve(aug.A, b)
        for i in range(3):
            H = eta_frequency_response(
                aug, omega=np.array([0.0]), input_dof=j, output_dof=i,
            )
            np.testing.assert_allclose(
                H[0].real, x_static[IDX_ETA.start + i], rtol=1e-9, atol=1e-15
            )
            np.testing.assert_allclose(H[0].imag, 0.0, atol=1e-9)


def test_eta_frequency_response_high_freq_decays():
    # At omega >> all closed-loop poles, |H| should be ~ |B_lost row| / omega^2
    # for the eta channel (two integrators between tau and eta: nu_dot = ..,
    # eta_dot = nu). Just sanity-check monotone decay over a range.
    aug = _build_csov_aug()
    omega = np.array([1.0, 3.0, 10.0, 30.0, 100.0])
    H = eta_frequency_response(aug, omega, input_dof=1, output_dof=1)
    mag = np.abs(H)
    # Monotone strictly decreasing over the high-freq portion.
    assert np.all(np.diff(mag) < 0), f"|H| should decay; got {mag}"
    # ratios should approach 1/9 (i.e. 1/omega^2 between 1->3 etc.).
    ratio_30_to_100 = mag[3] / mag[4]
    # 100/30 = 3.33; expect (100/30)^2 = 11.1 for an order-2 rolloff.
    assert 9.0 < ratio_30_to_100 < 14.0, f"high-freq rolloff wrong, got {ratio_30_to_100}"


def test_eta_frequency_response_input_validation():
    aug = _build_csov_aug()
    with pytest.raises(ValueError, match="input_dof"):
        eta_frequency_response(aug, np.array([1.0]), input_dof=3, output_dof=0)
    with pytest.raises(ValueError, match="output_dof"):
        eta_frequency_response(aug, np.array([1.0]), input_dof=0, output_dof=-1)


# ---------------------------------------------------------------------------
# eta_psd_from_dtau + estimate_post_wcf_excursion_distribution
# ---------------------------------------------------------------------------
def test_eta_psd_from_dtau_zero_sigma_returns_zero():
    aug = _build_csov_aug()
    omega = np.linspace(0.0, 2.0, 64)
    S = eta_psd_from_dtau(aug, np.zeros(3), tau_corr=15.0, omega=omega)
    np.testing.assert_array_equal(S, np.zeros((3, 64)))


def test_eta_psd_from_dtau_single_dof_variance_against_time_domain_mc():
    # Drive sway only with a known sigma_dtau, tau_corr; check sigma_eta_y
    # via Parseval matches a long time-domain MC realisation of the
    # 21-state augmented system.
    aug = _build_csov_aug()
    sigma_dtau_y = 200.0e3        # 200 kN, mild post-WCF spillover
    tau_corr = 15.0
    sigma_dtau = np.array([0.0, sigma_dtau_y, 0.0])

    # Frequency-domain variance (the article we want to validate).
    omega_max = 5.0
    omega = np.linspace(0.0, omega_max, 4096)
    S_eta = eta_psd_from_dtau(aug, sigma_dtau, tau_corr, omega)
    var_eta_y_fd = float(np.trapezoid(S_eta[1], omega))
    sigma_eta_y_fd = np.sqrt(var_eta_y_fd)

    # Time-domain MC: simulate the LTI system A driven by a Gauss-Markov
    # process on sway lost-thrust. AR(1) for dt=1s, alpha = exp(-dt/tau),
    # innovation std = sigma_dtau * sqrt(1 - alpha^2).
    rng = np.random.default_rng(7)
    dt = 0.5
    n_steps = 6000     # 3000 s, long enough for >>10 tau_corr
    alpha = np.exp(-dt / tau_corr)
    inn_std = sigma_dtau_y * np.sqrt(1.0 - alpha ** 2)
    # Pre-warm the AR(1) at its stationary distribution.
    dtau_y = np.zeros(n_steps)
    dtau_y[0] = rng.normal(0.0, sigma_dtau_y)
    inns = rng.normal(0.0, inn_std, size=n_steps - 1)
    for k in range(1, n_steps):
        dtau_y[k] = alpha * dtau_y[k - 1] + inns[k - 1]

    # Integrate dx/dt = A x + B_lost[:,1] * dtau_y(t), x(0)=0, with RK4.
    n = aug.A.shape[0]
    A = aug.A
    b = aug.B_lost[:, 1]
    x = np.zeros(n)
    eta_y_trace = np.zeros(n_steps)
    for k in range(n_steps - 1):
        u_k = dtau_y[k]
        u_kh = 0.5 * (dtau_y[k] + dtau_y[k + 1])
        u_k1 = dtau_y[k + 1]
        k1 = A @ x + b * u_k
        k2 = A @ (x + 0.5 * dt * k1) + b * u_kh
        k3 = A @ (x + 0.5 * dt * k2) + b * u_kh
        k4 = A @ (x + dt * k3) + b * u_k1
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        eta_y_trace[k + 1] = x[IDX_ETA.start + 1]

    # Discard initial transient (~ 10 tau_corr) before computing std.
    warm = int(10.0 * tau_corr / dt)
    sigma_eta_y_td = float(np.std(eta_y_trace[warm:], ddof=1))

    # MC sample-error budget on std with N_eff ~ duration / (2 tau_eta_corr):
    # eta_corr time is roughly tau_corr (the input correlation dominates
    # because the closed loop is faster), so N_eff ~ (n_steps*dt - 10*tau)
    # / (2*tau) ~ (3000 - 150)/30 ~ 95. Std-of-std ~ sigma / sqrt(2 N) ~ 7%.
    # 15% tolerance comfortably covers MC + first-order GM approximation.
    np.testing.assert_allclose(sigma_eta_y_td, sigma_eta_y_fd, rtol=0.15)


def test_estimate_post_wcf_excursion_distribution_smoke():
    # Smoke test on a representative DP cell: mu ~ 800 kN sway, sigma ~ 250
    # kN, cap ~ 900 kN (mild saturation regime). Verify the result is
    # finite, positive, P95 > P50, and mu_eta has the expected sign.
    aug = _build_csov_aug()
    mu = np.array([0.0, 800.0e3, 0.0])
    sigma = np.array([100.0e3, 250.0e3, 1.0e6])
    cap = np.array([1200.0e3, 900.0e3, 25.0e6])
    res = estimate_post_wcf_excursion_distribution(
        aug, mu=mu, sigma=sigma, cap_residual=cap,
        tau_decorr_lf_s=15.0, t_horizon_s=200.0,
    )
    # Saturation is positive-side dominant on sway, so mu_dtau_y > 0.
    assert res.mu_dtau[1] > 0.0
    # Surge and yaw are far inside the cap; near-zero spillover.
    assert abs(res.mu_dtau[0]) < 1.0
    assert abs(res.mu_dtau[2]) < 1.0
    # eta_y mean offset should be NEGATIVE (lost thrust to starboard
    # means controller can't push, so vessel drifts to port = -y in
    # body frame). Actually the sign depends on the closed-loop static
    # gain sign; just check it is nonzero and the P95 > 0.
    assert abs(res.mu_eta[1]) > 0.0
    assert res.sigma_eta[1] > 0.0
    assert res.eta_p50[1] > 0.0
    assert res.eta_p95[1] > res.eta_p50[1]
    assert res.eta_xy_p95_with_offset > 0.0
    # Finite and sane: no NaN/Inf
    for arr in (res.mu_dtau, res.sigma_dtau, res.mu_eta, res.sigma_eta,
                res.nu_0_plus, res.eta_p50, res.eta_p95):
        assert np.all(np.isfinite(arr))
