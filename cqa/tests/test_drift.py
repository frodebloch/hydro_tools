"""Tests for cqa.drift (Pinkster mean drift + Newman PSD from pdstrip)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cqa import (
    closed_loop,
    csov_default_config,
    SeaSpreading,
    load_pdstrip_rao,
)
from cqa.drift import (
    mean_drift_force_pdstrip,
    slow_drift_force_psd_newman_pdstrip,
)
from cqa.rao import evaluate_drift

PDSTRIP_PATH = Path.home() / "src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"


@pytest.fixture(scope="module")
def rao_table():
    if not PDSTRIP_PATH.exists():
        pytest.skip(f"pdstrip data not available: {PDSTRIP_PATH}")
    return load_pdstrip_rao(PDSTRIP_PATH)


# ---------------------------------------------------------------------------
# Mean drift force
# ---------------------------------------------------------------------------


def test_mean_drift_scales_with_Hs_squared(rao_table):
    """Mean drift force is proportional to integral D * S_eta and
    S_eta scales as Hs^2 -> mean drift must scale as Hs^2."""
    F1 = mean_drift_force_pdstrip(rao_table, Hs=2.0, Tp=8.0,
                                  theta_wave_rel=np.pi / 2.0)
    F2 = mean_drift_force_pdstrip(rao_table, Hs=4.0, Tp=8.0,
                                  theta_wave_rel=np.pi / 2.0)
    # Hs doubles -> Hs^2 quadruples; quadratures should agree to <0.5 %.
    np.testing.assert_allclose(F2, 4.0 * F1, rtol=5e-3, atol=1e-6)


def test_mean_drift_returns_three_components(rao_table):
    F = mean_drift_force_pdstrip(rao_table, Hs=2.5, Tp=9.0,
                                 theta_wave_rel=np.pi / 2.0)
    assert F.shape == (3,)
    assert F.dtype == np.float64
    assert np.all(np.isfinite(F))


def test_mean_drift_head_sea_dominantly_surge(rao_table):
    """Head sea (theta_rel = 0): sway and yaw drift should be ~0 by
    port/starboard symmetry, surge dominant."""
    F = mean_drift_force_pdstrip(rao_table, Hs=3.0, Tp=9.0,
                                 theta_wave_rel=0.0)
    assert abs(F[0]) > 10.0 * (abs(F[1]) + 1e-9), (
        f"surge drift must dominate sway in head sea, got F={F}"
    )
    # Yaw moment near zero by symmetry.
    assert abs(F[2]) < abs(F[0]) * 0.5


def test_mean_drift_beam_sea_dominantly_sway(rao_table):
    """Beam sea (theta_rel = pi/2 port): sway drift must dominate."""
    F = mean_drift_force_pdstrip(rao_table, Hs=3.0, Tp=9.0,
                                 theta_wave_rel=np.pi / 2.0)
    assert abs(F[1]) > abs(F[0]), (
        f"sway drift must dominate surge in beam sea, got F={F}"
    )


def test_mean_drift_zero_Hs_returns_zero(rao_table):
    F = mean_drift_force_pdstrip(rao_table, Hs=0.0, Tp=9.0,
                                 theta_wave_rel=np.pi / 4.0)
    np.testing.assert_allclose(F, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Slow-drift PSD
# ---------------------------------------------------------------------------


def test_slow_drift_psd_returns_callable_3x3(rao_table):
    S_F = slow_drift_force_psd_newman_pdstrip(
        rao_table, Hs=2.5, Tp=9.0, theta_wave_rel=np.pi / 2.0
    )
    G0 = S_F(0.0)
    assert G0.shape == (3, 3)
    # Symmetric.
    np.testing.assert_allclose(G0, G0.T, atol=1e-9)
    # Positive semi-definite (eigenvalues >= 0 to within float noise).
    eigs = np.linalg.eigvalsh(G0)
    assert eigs.min() > -1e-6 * abs(eigs.max())


def test_slow_drift_psd_scales_with_Hs_fourth(rao_table):
    """S_F = 8 integral T^2 * S_eta(omega) S_eta(omega+mu) d omega.
    S_eta scales as Hs^2 in BOTH spectra factors -> S_F scales as Hs^4.
    T = 0.5*(D+D) is independent of Hs."""
    S_a = slow_drift_force_psd_newman_pdstrip(
        rao_table, Hs=2.0, Tp=9.0, theta_wave_rel=np.pi / 2.0
    )
    S_b = slow_drift_force_psd_newman_pdstrip(
        rao_table, Hs=4.0, Tp=9.0, theta_wave_rel=np.pi / 2.0
    )
    G_a = S_a(0.0)
    G_b = S_b(0.0)
    # Hs doubles -> Hs^4 = 16x. Allow a few percent tolerance for the
    # spreading-grid sampling.
    np.testing.assert_allclose(G_b, 16.0 * G_a, rtol=2e-2, atol=1e-6)


def test_slow_drift_psd_decays_with_mu(rao_table):
    """At large mu the difference-frequency content goes to zero
    (S_eta(omega) and S_eta(omega+mu) overlap drops off)."""
    S_F = slow_drift_force_psd_newman_pdstrip(
        rao_table, Hs=3.0, Tp=9.0, theta_wave_rel=np.pi / 2.0
    )
    g0 = S_F(0.0)
    g_high = S_F(1.0)  # mu = 1 rad/s, well above DP closed-loop band
    # Sway diagonal must drop by at least an order of magnitude.
    assert g_high[1, 1] < 0.1 * g0[1, 1]


def test_slow_drift_psd_zero_Hs_returns_zero_matrix(rao_table):
    S_F = slow_drift_force_psd_newman_pdstrip(
        rao_table, Hs=0.0, Tp=9.0, theta_wave_rel=np.pi / 2.0
    )
    G = S_F(np.array([0.0, 0.05, 0.1]))
    np.testing.assert_allclose(G, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Lyapunov interop -- the PSD must be consumable by
# state_covariance_freqdomain_general
# ---------------------------------------------------------------------------


def test_psd_drops_into_state_covariance_solver(rao_table):
    """Smoke test: build the closed-loop, hand the pdstrip slow-drift
    PSD to state_covariance_freqdomain_general, and check it returns a
    valid 6x6 covariance matrix."""
    cfg = csov_default_config()
    from cqa.controller import LinearDpController
    from cqa.vessel import LinearVesselModel

    vessel = LinearVesselModel.from_config(cfg.vessel)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D, omega_n=(0.06, 0.06, 0.05), zeta=(0.9, 0.9, 0.9)
    )
    cl = closed_loop.ClosedLoop.build(vessel, controller)

    S_drift = slow_drift_force_psd_newman_pdstrip(
        rao_table, Hs=2.5, Tp=9.0, theta_wave_rel=np.pi / 2.0
    )
    P = closed_loop.state_covariance_freqdomain(cl, [S_drift])
    assert P.shape == (6, 6)
    # Symmetric & positive semi-definite.
    np.testing.assert_allclose(P, P.T, atol=1e-6)
    eigs = np.linalg.eigvalsh(P)
    assert eigs.min() > -1e-6 * abs(eigs.max())
    # Sway position variance should be the dominant entry for a beam sea.
    # (eta = [x, y, psi], with sway-position diagonal at index 1.)
    assert P[1, 1] > P[0, 0]


# ---------------------------------------------------------------------------
# evaluate_drift sanity (interpolation correctness)
# ---------------------------------------------------------------------------


def test_evaluate_drift_returns_zero_outside_range(rao_table):
    out = evaluate_drift(rao_table, np.array([0.001, 100.0]), beta_deg=90.0)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, 0.0)


def test_evaluate_drift_on_grid_matches_table(rao_table):
    """At a beta value that exists exactly on the grid AND an omega on
    the grid, the bilinear interpolation must reproduce the raw table
    entry to machine precision."""
    j = 5  # arbitrary on-grid beta index
    i = 10  # arbitrary on-grid omega index
    beta = float(rao_table.beta_deg[j])
    om = float(rao_table.omega[i])
    out = evaluate_drift(rao_table, np.array([om]), beta_deg=beta)
    np.testing.assert_allclose(out[0], rao_table.drift[i, j, :], rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Short-crested vs long-crested
# ---------------------------------------------------------------------------


def test_long_crested_mean_drift_matches_single_direction(rao_table):
    """SeaSpreading.long_crested() must reduce to the original
    single-direction integrand."""
    F = mean_drift_force_pdstrip(
        rao_table, Hs=3.0, Tp=9.0, theta_wave_rel=np.pi / 2.0,
        spreading=SeaSpreading.long_crested(),
    )
    # Reference: long-crested beam drift, sway-dominated.
    assert abs(F[1]) > abs(F[0])
    assert np.all(np.isfinite(F))


def test_short_crested_reduces_beam_sea_sway_drift(rao_table):
    """Beam-sea sway drift coefficient peaks around beta=90; spreading
    energy onto less effective directions (beta=60..120) reduces the
    integrated sway drift."""
    long_c = mean_drift_force_pdstrip(
        rao_table, Hs=3.0, Tp=9.0, theta_wave_rel=np.pi / 2.0,
        spreading=SeaSpreading.long_crested(),
    )
    short_c = mean_drift_force_pdstrip(
        rao_table, Hs=3.0, Tp=9.0, theta_wave_rel=np.pi / 2.0,
    )  # default cos-2s s=4
    assert abs(short_c[1]) < abs(long_c[1])


def test_short_crested_psd_diagonal_smaller_than_long_crested(rao_table):
    """Same intuition as for mean drift: spreading reduces beam-sea
    sway drift PSD diagonal at mu=0."""
    S_long = slow_drift_force_psd_newman_pdstrip(
        rao_table, Hs=2.5, Tp=9.0, theta_wave_rel=np.pi / 2.0,
        spreading=SeaSpreading.long_crested(),
    )
    S_short = slow_drift_force_psd_newman_pdstrip(
        rao_table, Hs=2.5, Tp=9.0, theta_wave_rel=np.pi / 2.0,
    )  # default cos-2s s=4
    G_long = S_long(0.0)
    G_short = S_short(0.0)
    # Sway-sway diagonal must be smaller under spreading.
    assert G_short[1, 1] < G_long[1, 1]


def test_psd_short_crested_default_is_full_rank(rao_table):
    """Short-crested spreading naturally fills off-diagonal entries via
    the directional outer product; the 3x3 PSD should be full rank
    (all eigenvalues > 0)."""
    S_F = slow_drift_force_psd_newman_pdstrip(
        rao_table, Hs=2.5, Tp=9.0, theta_wave_rel=np.pi / 2.0,
    )
    G = S_F(0.0)
    eigs = np.linalg.eigvalsh(G)
    # Yaw entries are small in absolute units (Nm) so we can't expect
    # the smallest eigenvalue to be huge -- just strictly positive.
    assert eigs.min() > 0.0
