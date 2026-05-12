"""Shared constants for the p7_brucon_validation matrix.

All downstream validation scripts that depend on common geometry or
timing should import from here rather than hardcoding values, so a
single change here propagates everywhere.

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

# CSOV forward gangway: port-pointing horizontal boom at mid-stroke.
# Use as ``GangwayJointState(**FORWARD_GANGWAY_JOINT_CSOV)``.
FORWARD_GANGWAY_JOINT_CSOV: dict = dict(
    h=15.0,
    alpha_g=-math.pi / 2.0,   # boom to port (-y body)
    beta_g=0.0,
    L=25.0,
)
