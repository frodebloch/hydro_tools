"""Tests for cqa.extreme_value (Rice formula + spectral moments)."""

from __future__ import annotations

import numpy as np
import pytest

from cqa.extreme_value import (
    RiceExceedanceResult,
    spectral_moments,
    zero_upcrossing_rate,
    vanmarcke_bandwidth_q,
    clh_epsilon,
    p_exceed_rice,
    p_exceed_rice_multiband,
    p_exceed_from_psd,
    inverse_rice,
    inverse_rice_multiband,
    predictive_running_max_quantile,
)


# ---------------------------------------------------------------------------
# Spectral moments
# ---------------------------------------------------------------------------


def _narrowband_psd(omega: np.ndarray, omega_n: float, sigma: float,
                    bandwidth: float) -> np.ndarray:
    """A simple narrowband PSD: a Gaussian bump centred on omega_n,
    normalised so that integral S(omega) d omega == sigma^2.
    Used to build closed-form test cases.
    """
    bump = np.exp(-((omega - omega_n) ** 2) / (2.0 * bandwidth ** 2))
    integral = float(np.trapezoid(bump, omega))
    return (sigma ** 2) * bump / integral


def test_spectral_moments_constant_psd_unit_box():
    """For S(omega) = 1 on [0, omega_max], lambda_k = omega_max^(k+1)/(k+1)."""
    omega = np.linspace(0.0, 5.0, 5001)
    S = np.ones_like(omega)
    m = spectral_moments(S, omega, orders=(0, 1, 2))
    assert np.isclose(m[0], 5.0, rtol=1e-6)
    assert np.isclose(m[1], 5.0 ** 2 / 2.0, rtol=1e-6)
    assert np.isclose(m[2], 5.0 ** 3 / 3.0, rtol=1e-6)


def test_spectral_moments_shape_mismatch_raises():
    with pytest.raises(ValueError):
        spectral_moments(np.array([1.0, 2.0]), np.array([0.0, 1.0, 2.0]))


def test_zero_upcrossing_rate_narrowband_equals_omega_n_over_2pi():
    """For a sufficiently narrowband process around omega_n,
    nu_0+ -> omega_n / (2 pi)."""
    omega = np.linspace(0.01, 3.0, 5001)
    omega_n = 1.0
    sigma = 0.5
    S = _narrowband_psd(omega, omega_n=omega_n, sigma=sigma, bandwidth=0.01)
    nu0 = zero_upcrossing_rate(S, omega)
    assert np.isclose(nu0, omega_n / (2.0 * np.pi), rtol=2e-3)


def test_zero_upcrossing_rate_returns_zero_for_zero_spectrum():
    omega = np.linspace(0.01, 3.0, 100)
    S = np.zeros_like(omega)
    assert zero_upcrossing_rate(S, omega) == 0.0


# ---------------------------------------------------------------------------
# Rice exceedance probability
# ---------------------------------------------------------------------------


def test_p_exceed_rice_high_threshold_is_small():
    """At a/sigma = 4, the tail is small. P_breach should be small for
    a 20 min op."""
    res = p_exceed_rice(sigma=1.0, nu_0_plus=0.05, threshold=4.0, T=1200.0)
    assert 0.0 < res.p_breach < 0.05
    assert res.valid


def test_p_exceed_rice_low_threshold_is_high():
    """At a/sigma = 1, exceedance is near-certain over a long window."""
    res = p_exceed_rice(sigma=1.0, nu_0_plus=0.05, threshold=1.0, T=1200.0)
    assert res.p_breach > 0.99
    # Rarity ratio < 2 => not valid.
    assert not res.valid


def test_p_exceed_rice_zero_T_is_zero():
    res = p_exceed_rice(sigma=1.0, nu_0_plus=0.05, threshold=2.0, T=0.0)
    assert res.p_breach == 0.0


def test_p_exceed_rice_zero_sigma_is_zero():
    res = p_exceed_rice(sigma=0.0, nu_0_plus=0.05, threshold=2.0, T=1200.0)
    assert res.p_breach == 0.0


def test_p_exceed_rice_zero_threshold_is_one():
    """Any non-trivial process crosses zero almost surely in any T."""
    res = p_exceed_rice(sigma=1.0, nu_0_plus=0.05, threshold=0.0, T=1200.0)
    assert res.p_breach == 1.0
    assert not res.valid


def test_p_exceed_rice_bilateral_doubles_count():
    """Bilateral counts both sides. Compared to unilateral, the
    expected_count must be exactly 2x."""
    r_uni = p_exceed_rice(sigma=1.0, nu_0_plus=0.05, threshold=3.0,
                          T=1200.0, bilateral=False)
    r_bi = p_exceed_rice(sigma=1.0, nu_0_plus=0.05, threshold=3.0,
                         T=1200.0, bilateral=True)
    assert np.isclose(r_bi.expected_count, 2.0 * r_uni.expected_count)


def test_p_exceed_rice_increases_with_T_and_with_lower_threshold():
    """Monotonicity sanity."""
    base = p_exceed_rice(sigma=1.0, nu_0_plus=0.05, threshold=3.0, T=600.0)
    longer = p_exceed_rice(sigma=1.0, nu_0_plus=0.05, threshold=3.0, T=1200.0)
    lower = p_exceed_rice(sigma=1.0, nu_0_plus=0.05, threshold=2.5, T=600.0)
    assert longer.p_breach > base.p_breach
    assert lower.p_breach > base.p_breach


def test_p_exceed_rice_expected_count_low_p_limit():
    """For very small p_breach, p_breach ~ expected_count (1 - exp(-x) ~ x)."""
    res = p_exceed_rice(sigma=1.0, nu_0_plus=0.05, threshold=4.5, T=600.0)
    assert np.isclose(res.p_breach, res.expected_count, rtol=2e-3)


def test_p_exceed_rice_negative_inputs_raise():
    with pytest.raises(ValueError):
        p_exceed_rice(sigma=-1.0, nu_0_plus=0.05, threshold=2.0, T=600.0)
    with pytest.raises(ValueError):
        p_exceed_rice(sigma=1.0, nu_0_plus=-0.05, threshold=2.0, T=600.0)
    with pytest.raises(ValueError):
        p_exceed_rice(sigma=1.0, nu_0_plus=0.05, threshold=-1.0, T=600.0)
    with pytest.raises(ValueError):
        p_exceed_rice(sigma=1.0, nu_0_plus=0.05, threshold=2.0, T=-1.0)


# ---------------------------------------------------------------------------
# Multiband
# ---------------------------------------------------------------------------


def test_multiband_two_identical_bands_doubles_expected_count():
    """Two identical bands => 2x expected count, same threshold."""
    band = (1.0, 0.05)  # sigma=1, nu0=0.05
    r1 = p_exceed_rice(sigma=1.0, nu_0_plus=0.05, threshold=3.0, T=600.0)
    r2 = p_exceed_rice_multiband([band, band], threshold=3.0, T=600.0)
    assert np.isclose(r2["expected_count_total"], 2.0 * r1.expected_count)


def test_multiband_per_band_attribution_sums():
    bands = [(1.0, 0.05), (0.3, 0.1)]
    r = p_exceed_rice_multiband(bands, threshold=2.5, T=900.0)
    assert len(r["expected_count_per_band"]) == 2
    assert len(r["p_breach_per_band"]) == 2
    # The combined p_breach should be at least as high as the largest
    # per-band p_breach (independent Poisson combination).
    assert r["p_breach"] >= max(r["p_breach_per_band"]) - 1e-12


def test_multiband_zero_band_ignored():
    """A band with zero sigma contributes nothing."""
    one_band = p_exceed_rice_multiband([(1.0, 0.05)], threshold=3.0, T=600.0)
    with_zero = p_exceed_rice_multiband(
        [(1.0, 0.05), (0.0, 0.1)], threshold=3.0, T=600.0
    )
    assert np.isclose(with_zero["p_breach"], one_band["p_breach"])


# ---------------------------------------------------------------------------
# PSD wrapper
# ---------------------------------------------------------------------------


def test_p_exceed_from_psd_round_trip():
    """Build a known narrowband PSD, compute sigma, nu0, q explicitly,
    then check p_exceed_from_psd matches a manual call to p_exceed_rice
    with the same Vanmarcke clustering parameters.
    """
    from cqa.extreme_value import vanmarcke_bandwidth_q

    omega = np.linspace(0.01, 3.0, 5001)
    sigma = 0.4
    omega_n = 0.6
    S = _narrowband_psd(omega, omega_n=omega_n, sigma=sigma, bandwidth=0.05)
    threshold = 1.5
    T = 1200.0

    res_psd = p_exceed_from_psd(S, omega, threshold=threshold, T=T)
    nu0_manual = zero_upcrossing_rate(S, omega)
    sigma_manual = float(np.sqrt(spectral_moments(S, omega, [0])[0]))
    q_manual = vanmarcke_bandwidth_q(S, omega)
    res_manual = p_exceed_rice(
        sigma=sigma_manual, nu_0_plus=nu0_manual,
        threshold=threshold, T=T,
        clustering="vanmarcke", q=q_manual,
    )
    assert np.isclose(res_psd.p_breach, res_manual.p_breach)
    assert np.isclose(res_psd.sigma, sigma_manual, rtol=1e-3)
    assert np.isclose(res_psd.sigma, sigma, rtol=2e-3)
    # Confirm we are actually in the narrowband regime: q << 1.
    assert q_manual < 0.3


def test_p_exceed_from_psd_sigma_override():
    """sigma_override decouples the variance from the spectral shape.
    Useful for the Bayesian-update path: prior PSD shape, posterior var."""
    omega = np.linspace(0.01, 3.0, 2001)
    S = _narrowband_psd(omega, omega_n=0.6, sigma=0.4, bandwidth=0.05)
    res_prior = p_exceed_from_psd(S, omega, threshold=1.5, T=1200.0)
    # Pretend the measured posterior sigma is twice the prior.
    res_post = p_exceed_from_psd(S, omega, threshold=1.5, T=1200.0,
                                 sigma_override=2.0 * res_prior.sigma)
    assert res_post.p_breach > res_prior.p_breach
    # nu_0_plus should be unchanged (depends only on PSD shape).
    assert np.isclose(res_post.nu_0_plus, res_prior.nu_0_plus)


# ---------------------------------------------------------------------------
# Vanmarcke bandwidth + clustering correction
# ---------------------------------------------------------------------------


def _wideband_psd(omega: np.ndarray, omega_lo: float, omega_hi: float,
                  sigma: float) -> np.ndarray:
    """Flat (boxcar) PSD on [omega_lo, omega_hi] scaled to give variance
    sigma^2."""
    width = omega_hi - omega_lo
    S = np.zeros_like(omega)
    mask = (omega >= omega_lo) & (omega <= omega_hi)
    S[mask] = sigma ** 2 / width
    return S


def test_vanmarcke_q_narrowband_is_small():
    """A sharply-peaked spectrum -> q close to 0."""
    omega = np.linspace(0.01, 3.0, 5001)
    S_narrow = _narrowband_psd(omega, omega_n=0.6, sigma=0.4, bandwidth=0.02)
    q = vanmarcke_bandwidth_q(S_narrow, omega)
    assert 0.0 <= q < 0.2, f"narrowband q should be small, got {q}"


def test_vanmarcke_q_wideband_is_large():
    """A wide flat spectrum -> q substantially larger than narrowband.
    For a boxcar [omega_lo, omega_hi] q = sqrt(1 - mean^2/(mean^2+var))
    of the spectral mass distribution; for our [0.1, 2.5] boxcar q
    is ~0.47, which is the transition regime. Compare against an
    explicit narrowband case to make the wideband-vs-narrowband
    inequality unambiguous."""
    omega = np.linspace(0.01, 3.0, 5001)
    S_wide = _wideband_psd(omega, omega_lo=0.1, omega_hi=2.5, sigma=0.4)
    S_narrow = _narrowband_psd(omega, omega_n=1.3, sigma=0.4, bandwidth=0.02)
    q_wide = vanmarcke_bandwidth_q(S_wide, omega)
    q_narrow = vanmarcke_bandwidth_q(S_narrow, omega)
    assert q_wide > q_narrow + 0.2, (
        f"wideband q={q_wide} should clearly exceed narrowband q={q_narrow}"
    )
    assert q_wide <= 1.0


def test_vanmarcke_q_in_unit_interval():
    """q must always lie in [0, 1] regardless of spectrum shape."""
    omega = np.linspace(0.01, 5.0, 4001)
    for fn in [
        lambda w: _narrowband_psd(w, omega_n=0.3, sigma=1.0, bandwidth=0.01),
        lambda w: _narrowband_psd(w, omega_n=2.0, sigma=0.5, bandwidth=0.5),
        lambda w: _wideband_psd(w, omega_lo=0.1, omega_hi=4.0, sigma=1.0),
        lambda w: np.exp(-((w - 1.0) ** 2)) * 0.5,  # JONSWAP-like bump
    ]:
        q = vanmarcke_bandwidth_q(fn(omega), omega)
        assert 0.0 <= q <= 1.0, f"q out of [0,1]: {q}"


def test_vanmarcke_q_zero_spectrum_is_one():
    """Degenerate spectrum -> conservative default q=1 (Poisson)."""
    omega = np.linspace(0.01, 3.0, 1001)
    assert vanmarcke_bandwidth_q(np.zeros_like(omega), omega) == 1.0


def test_clh_epsilon_in_unit_interval():
    """epsilon must lie in [0, 1] for any reasonable spectrum."""
    omega = np.linspace(0.01, 5.0, 4001)
    S = _narrowband_psd(omega, omega_n=0.6, sigma=0.4, bandwidth=0.05)
    eps = clh_epsilon(S, omega)
    assert 0.0 <= eps <= 1.0


def test_vanmarcke_correction_reduces_p_breach_vs_poisson():
    """For a narrowband process, Vanmarcke correction must produce a
    P_breach STRICTLY LOWER than the textbook Poisson (clustering of
    crossings makes them less independent)."""
    sigma = 0.4
    nu0 = 0.1  # 10 s natural period
    threshold = 1.2  # ratio = 3
    T = 1200.0  # 20 min
    res_pois = p_exceed_rice(sigma, nu0, threshold, T,
                             clustering="poisson")
    # Strongly narrowband: q = 0.05.
    res_van = p_exceed_rice(sigma, nu0, threshold, T,
                            clustering="vanmarcke", q=0.05)
    assert res_van.p_breach < res_pois.p_breach, (
        f"narrowband Vanmarcke should be < Poisson, "
        f"got {res_van.p_breach} vs {res_pois.p_breach}"
    )
    # Should be substantially lower (factor 2-5 typical for narrowband).
    assert res_van.p_breach < 0.6 * res_pois.p_breach


def test_vanmarcke_q_one_matches_poisson():
    """q=1 (wideband limit) recovers the Poisson formula exactly."""
    sigma = 0.4
    nu0 = 0.5
    threshold = 1.0
    T = 600.0
    res_pois = p_exceed_rice(sigma, nu0, threshold, T,
                             clustering="poisson")
    res_van = p_exceed_rice(sigma, nu0, threshold, T,
                            clustering="vanmarcke", q=1.0)
    assert np.isclose(res_van.p_breach, res_pois.p_breach, atol=1e-10)


def test_vanmarcke_falls_back_to_poisson_without_q():
    """clustering="vanmarcke" without q -> falls back to Poisson and
    sets clustering="poisson" in the result for traceability."""
    res = p_exceed_rice(0.4, 0.1, 1.0, 1200.0,
                        clustering="vanmarcke", q=None)
    assert res.clustering == "poisson"


def test_p_exceed_from_psd_uses_vanmarcke_by_default():
    """p_exceed_from_psd should auto-compute q and use Vanmarcke."""
    omega = np.linspace(0.01, 3.0, 4001)
    S = _narrowband_psd(omega, omega_n=0.6, sigma=0.4, bandwidth=0.03)
    res = p_exceed_from_psd(S, omega, threshold=1.5, T=1200.0)
    assert res.clustering == "vanmarcke"
    assert 0.0 <= res.q < 0.3, f"narrowband q expected, got {res.q}"

    # And it differs from forcing Poisson.
    res_pois = p_exceed_from_psd(S, omega, threshold=1.5, T=1200.0,
                                 clustering="poisson")
    assert res.p_breach < res_pois.p_breach


def test_multiband_accepts_q_per_band():
    """Multi-band must accept (sigma, nu0, q) triples and apply
    per-band Vanmarcke correction."""
    bands = [
        (0.5, 0.01, 0.4),  # narrow slow band
        (0.3, 0.125, 0.3),  # narrow wave band
    ]
    res = p_exceed_rice_multiband(bands, threshold=2.0, T=1200.0,
                                  bilateral=True, clustering="vanmarcke")
    assert "p_breach" in res
    assert 0.0 <= res["p_breach"] <= 1.0
    assert res["q_per_band"] == [0.4, 0.3]
    # And Vanmarcke gives strictly lower p than Poisson for the same bands.
    res_pois = p_exceed_rice_multiband(
        [(s, n) for (s, n, _) in bands],  # 2-tuples => Poisson per band
        threshold=2.0, T=1200.0, bilateral=True,
    )
    assert res["p_breach"] <= res_pois["p_breach"] + 1e-12


def test_multiband_legacy_2tuple_uses_poisson():
    """Backward-compat: 2-tuples are treated as q=None => Poisson."""
    bands = [(0.5, 0.01), (0.3, 0.125)]
    res = p_exceed_rice_multiband(bands, threshold=2.0, T=1200.0)
    # All bands fell back to Poisson.
    assert all(q == 1.0 for q in res["q_per_band"])


# ---------------------------------------------------------------------------
# Inverse Rice: solve for the level a such that P_breach(T, a) = p
# ---------------------------------------------------------------------------


def test_inverse_rice_round_trip_poisson():
    """Closed-form Poisson inverse: forward(inverse(p)) == p."""
    from cqa.extreme_value import inverse_rice, p_exceed_rice
    sigma = 1.0
    nu0 = 0.05
    T = 1200.0
    for p in (0.01, 0.05, 0.10, 0.25, 0.5, 0.9):
        a = inverse_rice(p, sigma, nu0, T, bilateral=True, clustering="poisson")
        r = p_exceed_rice(sigma, nu0, a, T, bilateral=True, clustering="poisson")
        assert r.p_breach == pytest.approx(p, abs=1e-9), (
            f"poisson round-trip p={p}: a={a}, P_breach={r.p_breach}"
        )


def test_inverse_rice_round_trip_vanmarcke():
    """Bisection Vanmarcke inverse: forward(inverse(p)) == p (~tolerance)."""
    from cqa.extreme_value import inverse_rice, p_exceed_rice
    sigma = 1.0
    nu0 = 0.05
    T = 1200.0
    q = 0.3  # narrowband
    for p in (0.01, 0.05, 0.10, 0.5, 0.9):
        a = inverse_rice(p, sigma, nu0, T, bilateral=True,
                         clustering="vanmarcke", q=q)
        r = p_exceed_rice(sigma, nu0, a, T, bilateral=True,
                          clustering="vanmarcke", q=q)
        assert r.p_breach == pytest.approx(p, abs=1e-6), (
            f"vanmarcke round-trip p={p}, q={q}: a={a}, P_breach={r.p_breach}"
        )


def test_inverse_rice_monotone_in_p():
    """Lower p (rarer event) => larger threshold a."""
    from cqa.extreme_value import inverse_rice
    sigma = 1.0
    nu0 = 0.05
    T = 1200.0
    a_p50 = inverse_rice(0.50, sigma, nu0, T)
    a_p90 = inverse_rice(0.10, sigma, nu0, T)  # P90 of running max
    a_p99 = inverse_rice(0.01, sigma, nu0, T)
    assert a_p50 < a_p90 < a_p99


def test_inverse_rice_monotone_in_T():
    """Longer exposure => larger expected peak."""
    from cqa.extreme_value import inverse_rice
    sigma = 1.0
    nu0 = 0.05
    a_short = inverse_rice(0.10, sigma, nu0, T=60.0)
    a_long = inverse_rice(0.10, sigma, nu0, T=4 * 3600.0)
    assert a_long > a_short


def test_inverse_rice_edge_cases():
    """sigma=0, p=0, p=1, T=0."""
    from cqa.extreme_value import inverse_rice
    assert inverse_rice(0.5, sigma=0.0, nu_0_plus=0.05, T=1200.0) == 0.0
    assert inverse_rice(0.5, sigma=1.0, nu_0_plus=0.0, T=1200.0) == 0.0
    assert inverse_rice(0.5, sigma=1.0, nu_0_plus=0.05, T=0.0) == 0.0
    assert inverse_rice(1.0, sigma=1.0, nu_0_plus=0.05, T=1200.0) == 0.0
    # p=0 => a = a_max_sigma * sigma
    a0 = inverse_rice(0.0, sigma=1.0, nu_0_plus=0.05, T=1200.0,
                      a_max_sigma=8.0)
    assert a0 == pytest.approx(8.0)


def test_inverse_rice_multiband_round_trip():
    """Multi-band inverse: forward(inverse(p)) == p."""
    from cqa.extreme_value import inverse_rice_multiband, p_exceed_rice_multiband
    bands = [(0.5, 0.01, 0.4), (0.3, 0.125, 0.3)]
    T = 1200.0
    for p in (0.01, 0.10, 0.5):
        a = inverse_rice_multiband(p, bands, T, bilateral=True,
                                   clustering="vanmarcke")
        r = p_exceed_rice_multiband(bands, threshold=a, T=T, bilateral=True,
                                    clustering="vanmarcke")
        assert r["p_breach"] == pytest.approx(p, abs=1e-6), (
            f"multiband round-trip p={p}: a={a}, P_breach={r['p_breach']}"
        )


# ---------------------------------------------------------------------------
# Marginal-over-sigma predictive quantile
# ---------------------------------------------------------------------------


def test_predictive_quantile_delta_limit_matches_inverse_rice():
    """Very tight InvGamma posterior on sigma^2 (large alpha) collapses
    to a delta at the prior mean; the predictive quantile must equal
    inverse_rice at that sigma."""
    sigma_true = 0.4
    nu_0_plus = 0.05
    q = 0.4
    T = 1200.0
    p = 0.10

    # Strong posterior: alpha large, sigma2_mean = beta/(alpha-1) = sigma_true^2.
    alpha = 5000.0
    beta = sigma_true ** 2 * (alpha - 1.0)

    a_pred = predictive_running_max_quantile(
        p=p, bands=[((alpha, beta), nu_0_plus, q)], T=T,
        bilateral=True, clustering="vanmarcke", n_quad=64,
    )
    a_ref = inverse_rice(
        p=p, sigma=sigma_true, nu_0_plus=nu_0_plus, T=T,
        bilateral=True, clustering="vanmarcke", q=q,
    )
    # 1% relative tolerance: 64 quadrature nodes plus residual prior
    # variance from the not-quite-delta posterior.
    assert a_pred == pytest.approx(a_ref, rel=0.01), (
        f"a_pred={a_pred:.4f} vs a_ref={a_ref:.4f}"
    )


def test_predictive_quantile_inflates_with_posterior_uncertainty():
    """A weak posterior (small alpha) keeps the same E[sigma^2] but adds
    epistemic variance. The predictive p-quantile must exceed the
    corresponding inverse_rice at the posterior median sigma, because
    the right tail of the sigma posterior contributes disproportionately
    to the breach probability."""
    from scipy.stats import invgamma

    sigma_mean = 0.4
    nu_0_plus = 0.05
    q = 0.4
    T = 1200.0
    p = 0.10

    # Weak posterior: small alpha -> wide spread in sigma^2.
    alpha = 5.0
    beta = sigma_mean ** 2 * (alpha - 1.0)
    sigma_median = float(np.sqrt(invgamma(a=alpha, scale=beta).median()))

    a_pred = predictive_running_max_quantile(
        p=p, bands=[((alpha, beta), nu_0_plus, q)], T=T,
        bilateral=True, clustering="vanmarcke", n_quad=128,
    )
    a_med = inverse_rice(
        p=p, sigma=sigma_median, nu_0_plus=nu_0_plus, T=T,
        bilateral=True, clustering="vanmarcke", q=q,
    )
    # Marginalised quantile should be strictly larger than the
    # plug-in-median quantile (epistemic uncertainty inflates the
    # tail).
    assert a_pred > a_med, (
        f"epistemic inflation expected: a_pred={a_pred:.3f} <= a_med={a_med:.3f}"
    )
    # And by a meaningful margin (>5%) for this width:
    assert (a_pred - a_med) / a_med > 0.05, (
        f"inflation too small: a_pred={a_pred:.3f}, a_med={a_med:.3f}"
    )


def test_predictive_quantile_mixed_bands_fixed_and_posterior():
    """Two bands: one fixed sigma (e.g. wave-RAO), one with InvGamma
    posterior (e.g. slow-drift LF). The result must lie between the
    fixed-only and double-posterior cases and must reduce to
    inverse_rice_multiband when the posterior is delta-tight."""
    sigma_slow = 0.5
    sigma_wave = 0.2
    nu_slow, q_slow = 0.01, 0.4
    nu_wave, q_wave = 0.125, 0.3
    T = 1200.0
    p = 0.10

    # Tight posterior on the slow band centred at sigma_slow.
    alpha = 5000.0
    beta = sigma_slow ** 2 * (alpha - 1.0)

    a_pred = predictive_running_max_quantile(
        p=p,
        bands=[((alpha, beta), nu_slow, q_slow), (sigma_wave, nu_wave, q_wave)],
        T=T, bilateral=True, clustering="vanmarcke", n_quad=64,
    )
    a_ref = inverse_rice_multiband(
        p=p,
        bands=[(sigma_slow, nu_slow, q_slow), (sigma_wave, nu_wave, q_wave)],
        T=T, bilateral=True, clustering="vanmarcke",
    )
    assert a_pred == pytest.approx(a_ref, rel=0.01), (
        f"mixed bands tight-posterior: a_pred={a_pred:.4f} vs a_ref={a_ref:.4f}"
    )
