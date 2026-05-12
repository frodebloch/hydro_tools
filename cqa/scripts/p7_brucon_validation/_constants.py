"""Shared constants for the p7_brucon_validation matrix.

All downstream validation scripts that depend on common geometry or
timing should import from here rather than hardcoding values, so a
single change here propagates everywhere.

Mirrors the run_validation_matrix.py timings, which all cells of the
validation matrix share:

  ACTIVATE_SK_S    : precondition window before station-keeping
                     activates (vessel held still via
                     SetFixedCourseAndSpeed).
  SETTLE_S         : intact-DP window before WCFDI (allows the slow
                     observer states -- bias estimator tau_b = 1000 s
                     in particular -- to converge before the failure
                     is triggered).
  POST_FAILURE_S   : post-WCFDI window where the transient is
                     evaluated.

  T_WCF_S          : absolute sim time of WCFDI = ACTIVATE_SK_S + SETTLE_S.

Historical note (analysis.md sec.12.21.15): we found that
SETTLE_S=500 left the slow LF / bias states still drifting at WCF
onset, contaminating the demean baseline used by all downstream
validation scripts. SETTLE_S=1500 (1.5*tau_b) is the chosen
compromise between physical settling and simulator-runtime cost.

Gangway joint geometry (analysis.md sec.12.21.16):
  The CSOV forward gangway base is at body (5, -9, -8) m i.e. on the
  port side of the deck, and the boom is intended to point to PORT
  (alpha_g = -pi/2), not forward. Earlier brucon-validation scripts
  hardcoded alpha_g = 0 (forward-pointing); the resulting c3 vector
  (-1, 0, -9) projected the WRONG vessel-deviation channel
  (surge-dominated instead of sway-dominated) into the dL excursion.
  The port-pointing geometry below gives c3 = (0, +1, +5) and
  c6 = (0, +1, 0, +23, 0, +5), which match the physical expectation
  that bow-quartering weather pushes the vessel toward port -> the
  gangway tip (latched off the port beam) is approached -> telescope
  must SHORTEN.
"""
from __future__ import annotations

import math

# -- Simulator timing -------------------------------------------------------

ACTIVATE_SK_S: float = 60.0
SETTLE_S: float = 1500.0
POST_FAILURE_S: float = 180.0

T_WCF_S: float = ACTIVATE_SK_S + SETTLE_S  # = 1560.0 s
TOTAL_SIM_S: float = ACTIVATE_SK_S + SETTLE_S + POST_FAILURE_S  # = 1740.0 s

# Alias used by older scripts.
T_WCF: float = T_WCF_S

# -- Gangway joint geometry -------------------------------------------------

# CSOV forward gangway: port-pointing horizontal boom at mid-stroke.
# Use as ``GangwayJointState(**FORWARD_GANGWAY_JOINT_CSOV)``.
FORWARD_GANGWAY_JOINT_CSOV: dict = dict(
    h=15.0,
    alpha_g=-math.pi / 2.0,   # boom to port (-y body)
    beta_g=0.0,
    L=25.0,
)
