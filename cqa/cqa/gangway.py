"""Gangway kinematics and operability for the cqa prototype.

Coordinate convention (matches brucon body frame):
    x : forward
    y : starboard
    z : down

Gangway joint state (operator-set, slowly-varying — treated as constants
for one cqa cycle):
    h        : rotation-centre height above the gangway base [m].
               Constrained 0 <= h <= rotation_centre_height_max.
    alpha_g  : slew angle in body frame [rad]. 0 = forward (along +x).
               Right-handed about +z (down), so positive alpha_g points
               the boom toward starboard.
    beta_g   : boom angle [rad]. 0 = horizontal. Positive = boom up
               (tip rises above rotation centre), so the body-frame z
               component of the telescope direction is -sin(beta_g)
               (i.e. negative, since +z is down).
    L        : telescope length [m]. Constrained
               telescope_min <= L <= telescope_max.

Forward kinematics:
    p_rc_body  = p_base + (0, 0, -h)        # rotation centre, body frame
                                            # (-h because +z is down and
                                            #  h is height *above* base)
    e_L_body   = (cos(beta_g) cos(alpha_g),
                  cos(beta_g) sin(alpha_g),
                  -sin(beta_g))             # telescope unit vector
    p_tip_body = p_rc_body + L * e_L_body   # gangway tip, body frame

In NED, with vessel pose (eta_n, eta_e, psi):
    p_tip_ned  = (eta_n, eta_e, 0) + R_z(psi) p_tip_body
where R_z(psi) is the standard 3D rotation matrix about the (down) z axis.

Operability question: with the landing point fixed in the world, by how
much does the telescope length L need to change to keep the tip on the
landing point as the vessel deviates from its setpoint by eta?

Linearised in eta:
    Delta_L(eta) ~ c^T eta
with the "telescope sensitivity" row vector
    c = [ e_L_x, e_L_y, (e_L_x * (-p_rc_y_body) + e_L_y * p_rc_x_body) ]
(only the horizontal components matter because the landing point is at
the same height as the rotation centre by operator setup, and small
heading deviations psi rotate the lever arm in the horizontal plane).

We also expose a 2x3 Jacobian for the *transverse* tip motion (the two
components orthogonal to e_L), which the operator uses informally to
judge slew/boom margin even though those are display-only in P3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .config import CqaConfig, GangwayConfig


# ---------------------------------------------------------------------------
# Joint state and forward kinematics
# ---------------------------------------------------------------------------


@dataclass
class GangwayJointState:
    """Operator-set gangway joint configuration (one cqa cycle)."""

    h: float            # rotation-centre height above base [m]
    alpha_g: float      # slew angle [rad], 0 = forward
    beta_g: float       # boom angle [rad], 0 = horizontal, +up
    L: float            # telescope length [m]


def rotation_centre_body(joint: GangwayJointState, gw: GangwayConfig) -> np.ndarray:
    """Rotation-centre position in body frame [m]."""
    p_base = np.array(gw.base_position_body, dtype=float)
    # Body z is +down, so an extra height h above base subtracts from z.
    return p_base + np.array([0.0, 0.0, -joint.h])


def telescope_direction_body(joint: GangwayJointState) -> np.ndarray:
    """Unit vector along the telescope (rotation-centre -> tip), body frame."""
    cb = np.cos(joint.beta_g)
    sb = np.sin(joint.beta_g)
    ca = np.cos(joint.alpha_g)
    sa = np.sin(joint.alpha_g)
    return np.array([cb * ca, cb * sa, -sb])


def tip_body(joint: GangwayJointState, gw: GangwayConfig) -> np.ndarray:
    """Gangway tip position in body frame [m]."""
    return rotation_centre_body(joint, gw) + joint.L * telescope_direction_body(joint)


def _rot_z(psi: float) -> np.ndarray:
    """Right-handed rotation about body-z (down) by angle psi.

    For small psi this reduces to I + psi * S where S = [[0,-1,0],[1,0,0],[0,0,0]].
    """
    c, s = np.cos(psi), np.sin(psi)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def tip_world(
    joint: GangwayJointState,
    gw: GangwayConfig,
    eta: np.ndarray,
) -> np.ndarray:
    """Gangway tip position in NED [m].

    `eta = (eta_n, eta_e, psi)` is the vessel deviation from setpoint
    (NED-aligned position deviation + small heading deviation about +z).
    """
    p_b = tip_body(joint, gw)
    R = _rot_z(eta[2])
    return np.array([eta[0], eta[1], 0.0]) + R @ p_b


# ---------------------------------------------------------------------------
# Linearised telescope-length sensitivity
# ---------------------------------------------------------------------------


def telescope_sensitivity(
    joint: GangwayJointState, gw: GangwayConfig
) -> np.ndarray:
    """Row vector c (length 3) such that Delta_L ~ c^T eta.

    Derived by holding the world-frame landing point fixed and asking
    what telescope length change is needed to keep the tip on it as the
    vessel pose deviates by eta = (eta_n, eta_e, psi). For a horizontal
    landing-point geometry (landing at same height as rotation centre),
    only the horizontal components matter:

        Delta_L = e_Lh . (p_LP - p_rc_world)_change
                = -(e_L_x, e_L_y) . [(eta_n, eta_e) + S(psi) (p_rc_x, p_rc_y)]

    Linearising in psi gives the closed form below.
    """
    p_rc_b = rotation_centre_body(joint, gw)
    e_L_b = telescope_direction_body(joint)
    e_x, e_y = e_L_b[0], e_L_b[1]
    p_x, p_y = p_rc_b[0], p_rc_b[1]
    # Note: the sign convention is "Delta_L > 0 means MORE telescope is
    # required to keep the tip on the landing point" -- equivalently, the
    # rotation centre has moved AWAY from the landing point along e_L.
    # For a vessel deviation +eta (CO moves +N, +E), the rotation centre
    # in NED moves by +eta + R(psi) p_rc_body. To keep the tip fixed in
    # world, L must DECREASE by the projection of that motion onto e_L
    # (in world frame).  So Delta_L = - e_L . (eta + S psi p_rc_h).
    #
    # In the linearised analysis e_L_world ~ e_L_body for small psi.
    return -np.array([
        e_x,
        e_y,
        e_x * (-p_y) + e_y * p_x,
    ])


def telescope_sensitivity_6dof(
    joint: GangwayJointState, gw: GangwayConfig
) -> np.ndarray:
    """Row vector c6 (length 6) for wave-frequency 6-DOF body motion.

    Returns the linearised telescope-length sensitivity to a small
    rigid-body 6-DOF motion of the vessel body origin

        xi = (xi_surge, xi_sway, xi_heave, xi_roll, xi_pitch, xi_yaw)

    (translations in m, rotations in rad, defined at the vessel body
    origin used by the pdstrip RAOs), such that

        Delta_L_wave = c6 . xi      (linearised, small motions).

    Derivation. The rotation centre at body-frame position
    r = (r_x, r_y, r_z) = p_rc_body is displaced in inertial frame by

        delta_p_rc = (xi_surge, xi_sway, xi_heave)
                   + (xi_roll, xi_pitch, xi_yaw) x (r_x, r_y, r_z).

    The latched gangway tip is fixed in the world. With the telescope
    direction e_L (world ~ body for small motions), the telescope
    *extension* required to keep the tip on its world-fixed point is

        Delta_L_wave = - e_L . delta_p_rc

    (sign matches telescope_sensitivity: Delta_L > 0 means MORE
    telescope is required because the rotation centre moved away from
    the latched tip along e_L). Expanding the cross product gives the
    closed form returned below.

    Notes
    -----
    * For an SOV with the gangway base on deck (r_z = base_z) and the
      operator-set rotation-centre height h, the body-frame z of the
      rotation centre is r_z = base_z - h (z is +down, so adding height
      makes z more negative). The heave channel and the roll/pitch
      lever-arm terms involving r_z therefore contribute even when the
      rotation centre is directly above the gangway base.
    * Reduces to the existing 3-DOF telescope_sensitivity in the special
      case xi_heave = xi_roll = xi_pitch = 0 (verified by direct
      substitution).
    * The pdstrip RAO body origin is assumed to coincide with the
      vessel reference point used everywhere else in cqa
      (cfg.gangway.base_position_body is given relative to that same
      origin). If a future config exposes a separate "RAO reference
      point", an additional translation would be needed here.
    """
    p_rc_b = rotation_centre_body(joint, gw)
    e_L_b = telescope_direction_body(joint)
    e_x, e_y, e_z = e_L_b
    r_x, r_y, r_z = p_rc_b
    return -np.array([
        e_x,                                # d/d xi_surge
        e_y,                                # d/d xi_sway
        e_z,                                # d/d xi_heave
        -e_y * r_z + e_z * r_y,             # d/d xi_roll
         e_x * r_z - e_z * r_x,             # d/d xi_pitch
        -e_x * r_y + e_y * r_x,             # d/d xi_yaw
    ])


def telescope_std_dev(
    joint: GangwayJointState, gw: GangwayConfig, P_eta: np.ndarray
) -> float:
    """sigma_L = sqrt(c^T P_eta c) [m]. P_eta is the 3x3 (eta_n, eta_e, psi) cov."""
    c = telescope_sensitivity(joint, gw)
    var = float(c @ P_eta @ c)
    return float(np.sqrt(max(var, 0.0)))


def telescope_velocity_std_dev(
    joint: GangwayJointState, gw: GangwayConfig, P_nu: np.ndarray
) -> float:
    """sigma_Ldot = sqrt(c^T P_nu c) [m/s].

    For small psi, the time derivative of c^T eta is c^T eta_dot = c^T (R nu) ~ c^T nu
    in the body frame, since the linearised rotation R drops out at first order.
    P_nu is the 3x3 covariance of (u, v, r).
    """
    c = telescope_sensitivity(joint, gw)
    var = float(c @ P_nu @ c)
    return float(np.sqrt(max(var, 0.0)))


# ---------------------------------------------------------------------------
# Operability gating (telescope only, per P3 scope)
# ---------------------------------------------------------------------------


@dataclass
class OperabilityResult:
    """Telescope-length operability check at one operating point."""

    L_mean: float                  # nominal telescope length at operating point [m]
    L_std: float                   # 1-sigma length deviation [m]
    L_lower: float                 # mean - k * sigma [m]
    L_upper: float                 # mean + k * sigma [m]
    Ldot_std: float                # 1-sigma stroke rate [m/s]
    margin_low: float              # L_lower - L_min  (positive = safe)  [m]
    margin_high: float             # L_max - L_upper  (positive = safe)  [m]
    margin_velocity: float         # threshold - k*Ldot_std (positive = safe) [m/s]
    pass_endstops: bool
    pass_velocity: bool
    pass_overall: bool
    info: dict


def evaluate_operability(
    joint: GangwayJointState,
    gw: GangwayConfig,
    P_eta: np.ndarray,
    P_nu: Optional[np.ndarray] = None,
    k_sigma: float = 1.96,
    Ldot_threshold: Optional[float] = None,
    L_nominal: Optional[float] = None,
) -> OperabilityResult:
    """Evaluate telescope operability at one operating point.

    P3 scope: gating is on telescope only.
      - End-stops: L_mean +/- k_sigma * sigma_L must lie in
        [telescope_min, telescope_max].
      - Stroke velocity: k_sigma * sigma_Ldot must be below the
        configured velocity threshold (display-only otherwise; if
        P_nu is None this check is skipped).

    Slew alpha and boom beta are display-only in P3 (operator picks the
    pointing direction; this function does not gate against alpha/beta
    end-stops).

    `L_nominal` defaults to `joint.L` (the operator's chosen length).
    """
    L_mean = joint.L if L_nominal is None else L_nominal
    sigma_L = telescope_std_dev(joint, gw, P_eta)
    L_lower = L_mean - k_sigma * sigma_L
    L_upper = L_mean + k_sigma * sigma_L
    margin_low = L_lower - gw.telescope_min
    margin_high = gw.telescope_max - L_upper
    pass_endstops = (margin_low > 0.0) and (margin_high > 0.0)

    if P_nu is not None:
        sigma_Ldot = telescope_velocity_std_dev(joint, gw, P_nu)
        thr = (
            gw.telescope_velocity_default_threshold
            if Ldot_threshold is None
            else Ldot_threshold
        )
        margin_velocity = thr - k_sigma * sigma_Ldot
        pass_velocity = margin_velocity > 0.0
    else:
        sigma_Ldot = float("nan")
        margin_velocity = float("nan")
        pass_velocity = True  # not evaluated -> do not fail

    return OperabilityResult(
        L_mean=L_mean,
        L_std=sigma_L,
        L_lower=L_lower,
        L_upper=L_upper,
        Ldot_std=sigma_Ldot,
        margin_low=margin_low,
        margin_high=margin_high,
        margin_velocity=margin_velocity,
        pass_endstops=pass_endstops,
        pass_velocity=pass_velocity,
        pass_overall=pass_endstops and pass_velocity,
        info={
            "k_sigma": k_sigma,
            "telescope_sensitivity_c": telescope_sensitivity(joint, gw),
        },
    )
