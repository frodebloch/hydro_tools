"""Tests for the gangway kinematics and operability module."""

from __future__ import annotations

import numpy as np
import pytest

from cqa import (
    csov_default_config,
    GangwayJointState,
    rotation_centre_body,
    telescope_direction_body,
    tip_body,
    tip_world,
    telescope_sensitivity,
    telescope_sensitivity_6dof,
    telescope_std_dev,
    telescope_velocity_std_dev,
    evaluate_operability,
)


def _default_joint(cfg) -> GangwayJointState:
    gw = cfg.gangway
    L0 = 0.5 * (gw.telescope_min + gw.telescope_max)
    return GangwayJointState(
        h=gw.rotation_centre_height_above_base,
        alpha_g=np.pi / 2.0,  # slew to starboard
        beta_g=0.0,           # horizontal
        L=L0,
    )


def test_rotation_centre_body_lifts_above_base():
    """Rotation centre at height h above base: body z decreases by h."""
    cfg = csov_default_config()
    gw = cfg.gangway
    h = 12.5
    joint = GangwayJointState(h=h, alpha_g=0.0, beta_g=0.0, L=25.0)
    p_rc = rotation_centre_body(joint, gw)
    p_base = np.array(gw.base_position_body)
    assert np.allclose(p_rc[:2], p_base[:2])
    assert np.isclose(p_rc[2], p_base[2] - h)


def test_telescope_direction_unit():
    """e_L is a unit vector for any (alpha_g, beta_g)."""
    for alpha in [0.0, 0.5, np.pi / 2, -0.7, np.pi]:
        for beta in [-0.5, 0.0, 0.4, 1.0]:
            joint = GangwayJointState(h=10.0, alpha_g=alpha, beta_g=beta, L=25.0)
            e = telescope_direction_body(joint)
            assert np.isclose(np.linalg.norm(e), 1.0), (
                f"e_L not unit for alpha={alpha}, beta={beta}: |e|={np.linalg.norm(e)}"
            )


def test_telescope_direction_known_orientations():
    """Check key cardinal directions of the telescope unit vector."""
    # alpha=0, beta=0: forward, horizontal -> +x
    j = GangwayJointState(h=10.0, alpha_g=0.0, beta_g=0.0, L=25.0)
    assert np.allclose(telescope_direction_body(j), [1.0, 0.0, 0.0])
    # alpha=pi/2, beta=0: starboard, horizontal -> +y
    j = GangwayJointState(h=10.0, alpha_g=np.pi / 2, beta_g=0.0, L=25.0)
    assert np.allclose(telescope_direction_body(j), [0.0, 1.0, 0.0])
    # alpha=0, beta=pi/2: forward, straight up -> -z (since +z is down)
    j = GangwayJointState(h=10.0, alpha_g=0.0, beta_g=np.pi / 2, L=25.0)
    assert np.allclose(telescope_direction_body(j), [0.0, 0.0, -1.0])


def test_tip_world_zero_eta_equals_body():
    """At zero vessel deviation, tip_world == tip_body (NED-aligned with body)."""
    cfg = csov_default_config()
    joint = _default_joint(cfg)
    p_b = tip_body(joint, cfg.gangway)
    p_w = tip_world(joint, cfg.gangway, np.zeros(3))
    assert np.allclose(p_b, p_w)


def test_telescope_sensitivity_horizontal_starboard():
    """For a starboard-pointing horizontal gangway (alpha=pi/2, beta=0):
    e_L_body = (0, 1, 0), so c = -[0, 1, p_rc_x_body]. Vessel +eta_e
    moves the rotation centre +E in world; that REDUCES the required L
    by 1m per 1m, hence c[1] = -1."""
    cfg = csov_default_config()
    joint = GangwayJointState(
        h=cfg.gangway.rotation_centre_height_above_base,
        alpha_g=np.pi / 2.0,
        beta_g=0.0,
        L=25.0,
    )
    c = telescope_sensitivity(joint, cfg.gangway)
    p_base = np.array(cfg.gangway.base_position_body)
    expected = -np.array([0.0, 1.0, p_base[0]])  # p_rc_x_body == base x
    assert np.allclose(c, expected, atol=1e-9), f"c={c}, expected={expected}"


def test_telescope_std_dev_consistent_with_explicit_projection():
    """sigma_L from telescope_std_dev matches explicit c^T P c."""
    cfg = csov_default_config()
    joint = _default_joint(cfg)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    P_eta = A @ A.T  # SPD
    c = telescope_sensitivity(joint, cfg.gangway)
    sigma = telescope_std_dev(joint, cfg.gangway, P_eta)
    expected = float(np.sqrt(c @ P_eta @ c))
    assert np.isclose(sigma, expected, rtol=1e-12)


def test_operability_passes_at_midstroke_with_small_excursion():
    """At mid-stroke L=25 with small position covariance, both end-stops pass."""
    cfg = csov_default_config()
    joint = _default_joint(cfg)
    P_eta = np.diag([0.1**2, 0.5**2, np.deg2rad(0.5) ** 2])
    res = evaluate_operability(joint, cfg.gangway, P_eta, k_sigma=1.96)
    assert res.pass_endstops, (
        f"Expected end-stops pass: L_lower={res.L_lower:.2f}, L_upper={res.L_upper:.2f}, "
        f"limits=[{cfg.gangway.telescope_min}, {cfg.gangway.telescope_max}]"
    )


def test_operability_fails_when_close_to_endstop():
    """Setting L close to telescope_min and a large sigma makes margin_low negative."""
    cfg = csov_default_config()
    joint = _default_joint(cfg)
    joint.L = cfg.gangway.telescope_min + 0.5  # very close to lower limit
    P_eta = np.diag([0.1**2, 1.0**2, np.deg2rad(0.5) ** 2])  # 1 m sway sigma
    res = evaluate_operability(joint, cfg.gangway, P_eta, k_sigma=1.96)
    assert not res.pass_endstops, (
        f"Expected end-stop fail near L_min: margin_low={res.margin_low:.3f}"
    )


def test_velocity_check_uses_nu_covariance():
    """Velocity sigma scales with nu covariance and triggers fail above threshold."""
    cfg = csov_default_config()
    joint = _default_joint(cfg)
    P_eta = np.diag([0.1**2, 0.3**2, np.deg2rad(0.3) ** 2])
    # Big sway-rate variance: |c_y| = 1, so sigma_Ldot ~ sigma_v
    P_nu_safe = np.diag([0.05**2, 0.05**2, np.deg2rad(0.05) ** 2])
    P_nu_bad = np.diag([0.05**2, 1.0**2, np.deg2rad(0.05) ** 2])
    res_safe = evaluate_operability(joint, cfg.gangway, P_eta, P_nu=P_nu_safe)
    res_bad = evaluate_operability(joint, cfg.gangway, P_eta, P_nu=P_nu_bad)
    assert res_safe.pass_velocity
    assert not res_bad.pass_velocity, (
        f"Expected velocity fail with sigma_v=1 m/s: Ldot_std={res_bad.Ldot_std:.3f}"
    )


# ---------------------------------------------------------------------------
# 6-DOF wave-frequency sensitivity
# ---------------------------------------------------------------------------


def test_sensitivity_6dof_reduces_to_3dof_for_planar_motion():
    """Setting xi_heave = xi_roll = xi_pitch = 0 must give the same Delta_L
    as the existing 3-DOF telescope_sensitivity for surge/sway/yaw inputs.

    This is the consistency check tying the new RAO-driven path to the
    existing P3 result so we don't double-count or sign-flip anything.
    """
    cfg = csov_default_config()
    rng = np.random.default_rng(7)
    for _ in range(8):
        joint = GangwayJointState(
            h=rng.uniform(2.0, 12.0),
            alpha_g=rng.uniform(-np.pi, np.pi),
            beta_g=rng.uniform(-0.4, 0.4),  # boom near horizontal
            L=rng.uniform(20.0, 30.0),
        )
        c3 = telescope_sensitivity(joint, cfg.gangway)        # (eta_n, eta_e, psi)
        c6 = telescope_sensitivity_6dof(joint, cfg.gangway)   # (s, sw, h, r, p, y)
        # Planar map: xi_surge = eta_n, xi_sway = eta_e, xi_yaw = psi.
        # Heave/roll/pitch zero -> Delta_L should be c3 . eta exactly.
        eta = rng.standard_normal(3)
        xi = np.array([eta[0], eta[1], 0.0, 0.0, 0.0, eta[2]])
        dL_3 = float(c3 @ eta)
        dL_6 = float(c6 @ xi)
        assert np.isclose(dL_3, dL_6, atol=1e-12), (
            f"3-DOF and 6-DOF disagree on planar motion: {dL_3} vs {dL_6}"
        )


def test_sensitivity_6dof_heave_for_vertical_telescope():
    """For a straight-up telescope (alpha=0, beta=pi/2), e_L = (0, 0, -1).
    A pure heave xi_heave = +1 m moves the body (and rotation centre)
    DOWN by 1 m in inertial frame (z is +down). The latched tip is fixed
    in world; the rotation centre moves AWAY from the tip along e_L
    (since e_L points up and rc moved down), so MORE telescope is
    required: Delta_L = +1 m. Hence c6[heave] should be +1.
    """
    cfg = csov_default_config()
    joint = GangwayJointState(h=8.0, alpha_g=0.0, beta_g=np.pi / 2, L=25.0)
    c6 = telescope_sensitivity_6dof(joint, cfg.gangway)
    # Order: surge, sway, heave, roll, pitch, yaw
    assert np.isclose(c6[0], 0.0, atol=1e-12)   # no surge coupling
    assert np.isclose(c6[1], 0.0, atol=1e-12)   # no sway coupling
    assert np.isclose(c6[2], 1.0, atol=1e-12), f"c6[heave]={c6[2]}, expected +1"


def test_sensitivity_6dof_pitch_lever_arm():
    """For a forward-pointing horizontal telescope (alpha=0, beta=0):
    e_L = (1, 0, 0). Rotation centre at body-frame x = base_x ~ +5 m,
    y = -9 m (gangway base port-aft of midship), z = base_z - h.
    A small pitch xi_pitch (bow-up positive about +y, RH frame) moves
    the rotation centre down by xi_pitch * r_x (lever arm forward of
    body origin -> goes UP) and forward/back depending on r_z.

    Closed form: c6[pitch] = -(e_x * r_z - e_z * r_x) = -e_x * r_z
    (since e_z = 0). With r_z = base_z - h < 0 (rotation centre above
    body z=0 plane after lifting by h), c6[pitch] should be > 0.
    """
    cfg = csov_default_config()
    h = 8.0
    joint = GangwayJointState(h=h, alpha_g=0.0, beta_g=0.0, L=25.0)
    c6 = telescope_sensitivity_6dof(joint, cfg.gangway)
    p_rc = rotation_centre_body(joint, cfg.gangway)
    expected_pitch = -(1.0 * p_rc[2] - 0.0 * p_rc[0])
    assert np.isclose(c6[4], expected_pitch, atol=1e-12), (
        f"c6[pitch]={c6[4]}, expected {expected_pitch}; r_z={p_rc[2]}"
    )
    # Sanity: with rotation centre above z=0 (r_z negative, since +z is
    # down), pitch sensitivity should be positive.
    assert c6[4] > 0.0


def test_sensitivity_6dof_yaw_matches_3dof_yaw_term():
    """The yaw component of c6 must equal the psi component of c3
    for any joint configuration -- they are derived from the same
    lever-arm cross-product."""
    cfg = csov_default_config()
    rng = np.random.default_rng(11)
    for _ in range(6):
        joint = GangwayJointState(
            h=rng.uniform(2.0, 12.0),
            alpha_g=rng.uniform(-np.pi, np.pi),
            beta_g=rng.uniform(-0.5, 0.5),
            L=rng.uniform(20.0, 30.0),
        )
        c3 = telescope_sensitivity(joint, cfg.gangway)
        c6 = telescope_sensitivity_6dof(joint, cfg.gangway)
        assert np.isclose(c3[2], c6[5], atol=1e-12), (
            f"yaw term mismatch: c3[psi]={c3[2]}, c6[yaw]={c6[5]}"
        )
