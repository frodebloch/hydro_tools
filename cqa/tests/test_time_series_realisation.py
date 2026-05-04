"""Tests for cqa.time_series_realisation.

Strategy: the realised time series must reproduce the analytical
predictions in the limit of long duration / dense omega grid. Three
cross-checks:

1. Vector force realisation: empirical force variance matches integral
   of the input PSD (one-sided convention: var = integral_0^inf S dom).

2. Closed-loop integration: empirical state covariance matches the
   Lyapunov / frequency-domain steady-state prediction.

3. Wave-frequency 6-DOF realisation: empirical telescope-length std
   matches the analytical sigma_L_wave from cqa.wave_response.

Each test uses a dedicated long realisation (~30 min) to keep noise
floor at a few percent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cqa import (
    csov_default_config,
    LinearVesselModel,
    LinearDpController,
    ClosedLoop,
    GangwayJointState,
    npd_wind_gust_force_psd,
    current_variability_force_psd,
    state_covariance_freqdomain,
    load_pdstrip_rao,
    sigma_L_wave,
)
from cqa.psd import WindForceModel
from cqa.time_series_realisation import (
    realise_vector_force_time_series,
    integrate_closed_loop_response,
    realise_wave_motion_6dof,
    radial_position_time_series,
    telescope_length_deviation_time_series,
)


PDSTRIP_PATH = Path.home() / "src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"


# ---------------------------------------------------------------------------
# Test 1: vector force realisation reproduces input PSD variance
# ---------------------------------------------------------------------------


def test_vector_force_realisation_variance_matches_integral():
    """One-sided PSD convention: empirical variance per channel must
    equal integral_0^inf S_ii(omega) d omega.

    Use the wind gust + current variability PSDs at a single
    operating point. JONSWAP/drift omitted to keep this independent of
    pdstrip data."""
    cfg = csov_default_config()
    Vw = 14.0
    theta_rel = np.radians(30.0)
    Vc = 0.5

    vp = cfg.vessel
    cp = cfg.current
    wp = cfg.wind

    wind_model = WindForceModel(wp=wp, loa=vp.loa)
    S_wind = npd_wind_gust_force_psd(wind_model, Vw, theta_rel)

    # Linearise current force about Vc.
    from cqa.vessel import CurrentForceModel
    current_model = CurrentForceModel(
        cp=cp,
        lateral_area_underwater=vp.lpp * vp.draft,
        frontal_area_underwater=vp.beam * vp.draft,
        loa=vp.loa,
    )
    F0 = current_model.force(Vc, theta_rel)
    dFdVc = 2.0 * F0 / Vc
    S_curr = current_variability_force_psd(dFdVc, sigma_Vc=0.1, tau=600.0)

    omega_grid = np.linspace(2e-3, 0.6, 256)
    dt = 0.5
    T_total = 1800.0  # 30 min per realisation
    t = np.arange(0.0, T_total, dt)

    # Analytical: trapz integral of S_ii(omega) d omega from 0 to omega_max.
    S_diag = np.zeros((omega_grid.size, 3))
    for k, w in enumerate(omega_grid):
        S = np.asarray(S_wind(float(w))) + np.asarray(S_curr(float(w)))
        S_diag[k] = np.diag(S)
    ana_var = np.trapezoid(S_diag, omega_grid, axis=0)

    # Single-realisation variance is noisy for the long-correlation
    # current-moment channel (correlation time ~600 s, only ~3
    # independent draws in 30 min). Average over a handful of seeds to
    # tighten the variance estimator. Target: 10 % per channel after
    # averaging. With 10 seeds the per-channel std/mean (measured
    # offline) is ~3 % surge, ~8 % sway, ~9 % yaw -- so 15 % is a
    # comfortable test tolerance.
    n_seeds = 10
    emp_vars = np.zeros((n_seeds, 3))
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        F = realise_vector_force_time_series(
            [S_wind, S_curr], omega_grid, t, rng,
        )
        assert F.shape == (3, t.size)
        emp_vars[seed] = np.var(F, axis=1)
    emp_var = emp_vars.mean(axis=0)

    np.testing.assert_allclose(emp_var, ana_var, rtol=0.15)


# ---------------------------------------------------------------------------
# Test 2: closed-loop integration reproduces Lyapunov covariance
# ---------------------------------------------------------------------------


def test_closed_loop_realisation_matches_lyapunov():
    """The empirical covariance of the integrated closed-loop state
    must converge to the frequency-domain Lyapunov prediction."""
    cfg = csov_default_config()
    Vw = 14.0
    theta_rel = np.radians(30.0)

    vp = cfg.vessel
    wp = cfg.wind
    vessel = LinearVesselModel.from_config(vp)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D,
        omega_n=cfg.controller.omega_n, zeta=cfg.controller.zeta,
    )
    cl = ClosedLoop.build(vessel, controller)

    wind_model = WindForceModel(wp=wp, loa=vp.loa)
    S_wind = npd_wind_gust_force_psd(wind_model, Vw, theta_rel)

    omega_grid = np.linspace(1e-3, 0.6, 256)
    dt = 0.5
    T_total = 1800.0  # 30 min per seed
    t = np.arange(0.0, T_total, dt)

    # Frequency-domain Lyapunov prediction.
    P_ana = state_covariance_freqdomain(
        cl, [S_wind], omega_lo=omega_grid[0], omega_hi=omega_grid[-1],
        n_points=512,
    )
    ana_var = np.diag(P_ana)

    n_seeds = 6
    emp_vars = np.zeros((n_seeds, 6))
    burn_in = int(300.0 / dt)  # drop first 5 min (closed-loop transient)
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 100)
        F = realise_vector_force_time_series([S_wind], omega_grid, t, rng)
        x = integrate_closed_loop_response(cl, F, t)
        emp_vars[seed] = np.var(x[:, burn_in:], axis=1)
    emp_var = emp_vars.mean(axis=0)

    # Position channels (eta_n, eta_e, psi): within 25% after seed averaging.
    np.testing.assert_allclose(
        emp_var[0:3], ana_var[0:3], rtol=0.25,
        err_msg=f"emp_var={emp_var[0:3]}, ana_var={ana_var[0:3]}",
    )


# ---------------------------------------------------------------------------
# Test 3: wave-frequency telescope std matches sigma_L_wave
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PDSTRIP_PATH.exists(), reason="pdstrip data not available")
def test_wave_realisation_matches_sigma_L_wave():
    """Empirical std of (c6 . xi_wf)(t) must match the analytical
    sigma_L_wave from cqa.wave_response, within sampling noise."""
    cfg = csov_default_config()
    gw = cfg.gangway
    L0 = 0.5 * (gw.telescope_min + gw.telescope_max)
    joint = GangwayJointState(
        h=gw.rotation_centre_height_above_base,
        alpha_g=-np.pi / 2.0, beta_g=0.0, L=L0,
    )
    rao = load_pdstrip_rao(PDSTRIP_PATH)

    Hs = 2.8
    Tp = 9.0
    theta_wave_rel = np.radians(30.0)

    # Analytical reference.
    wave = sigma_L_wave(joint, cfg, rao, Hs=Hs, Tp=Tp,
                       theta_wave_rel=theta_wave_rel)
    sig_ana = wave.sigma_L_wave

    # Realisation.
    dt = 0.25
    T_total = 1800.0  # 30 min
    t = np.arange(0.0, T_total, dt)
    rng = np.random.default_rng(13)

    xi = realise_wave_motion_6dof(
        rao, Hs=Hs, Tp=Tp, theta_wave_rel=theta_wave_rel,
        t=t, rng=rng,
    )
    # Project onto telescope length (wave-only path: x_lf = 0).
    dL = telescope_length_deviation_time_series(
        x_lf=np.zeros((6, t.size)),
        xi_wf=xi,
        joint=joint, cfg=cfg,
    )
    sig_emp = float(np.std(dL))

    # Within 15% (wave-frequency channel is relatively narrowband, so
    # 30 min ~ 200 wave periods gives good sampling).
    rel_err = abs(sig_emp - sig_ana) / sig_ana
    assert rel_err < 0.15, (
        f"empirical sigma_L = {sig_emp*100:.2f} cm, "
        f"analytical = {sig_ana*100:.2f} cm, rel_err = {rel_err*100:.1f}%"
    )


# ---------------------------------------------------------------------------
# Test 4: validation errors
# ---------------------------------------------------------------------------


def test_force_realisation_rejects_non_uniform_grid():
    omega_bad = np.array([0.1, 0.2, 0.4, 0.5])  # non-uniform
    t = np.linspace(0.0, 10.0, 20)
    rng = np.random.default_rng(0)
    S_F = lambda w: np.eye(3)
    with pytest.raises(ValueError, match="uniformly"):
        realise_vector_force_time_series([S_F], omega_bad, t, rng)


def test_force_realisation_rejects_wrong_psd_shape():
    omega = np.linspace(0.01, 0.5, 16)
    t = np.linspace(0.0, 10.0, 20)
    rng = np.random.default_rng(0)
    S_F = lambda w: np.eye(2)  # wrong shape
    with pytest.raises(ValueError, match="3x3"):
        realise_vector_force_time_series([S_F], omega, t, rng)


def test_integrate_closed_loop_rejects_mismatched_sizes():
    cfg = csov_default_config()
    vessel = LinearVesselModel.from_config(cfg.vessel)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D,
        omega_n=cfg.controller.omega_n, zeta=cfg.controller.zeta,
    )
    cl = ClosedLoop.build(vessel, controller)
    F = np.zeros((3, 100))
    t = np.linspace(0.0, 50.0, 50)  # mismatched
    with pytest.raises(ValueError, match="time samples"):
        integrate_closed_loop_response(cl, F, t)
