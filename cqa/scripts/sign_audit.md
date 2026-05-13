# cqa sign-convention audit (sec.12.21.19, in progress)

Purpose: trace the wave/wind/current → body-force → integrated-eta sign chain
end-to-end, one stage at a time. No code changes until every stage is
verified and the cumulative sign is unambiguously identified.

Reference geometry for the running example (run_comparison_waves_only_quartering30.py):

    VESSEL_HEADING_COMPASS = 180 deg  (bow points south)
    WAVE_DIR_COMPASS       = 210 deg  (waves come from SSW;
                                       i.e. the *source* bears 210 deg true)
    -> bearing of source from bow, measured CW (compass-style):
        rel_bearing_cw = (210 - 180) = +30 deg
    -> source is on the STARBOARD-bow.
    -> physical hull force from the waves is toward PORT (-y body).
    -> brucon truth: negative sway hump after WCF. (Anchor.)

============================================================
BODY-FRAME SIGN ANCHOR (Fossen 2011, sec.2.1) -- USE EVERYWHERE
============================================================
Positive directions of (surge, sway, yaw):
    +surge (x)   = AHEAD       (toward the bow)
    +sway  (y)   = TO STARBOARD
    +yaw   (psi) = CLOCKWISE seen from above

Therefore:
  - "force toward port"          <=>  Fy < 0
  - "force toward starboard"     <=>  Fy > 0
  - "force toward bow (ahead)"   <=>  Fx > 0
  - "force toward stern (aft)"   <=>  Fx < 0
  - "vessel displaced to port"   <=>  eta_sway = y < 0
  - "vessel displaced to stbd"   <=>  eta_sway = y > 0
  - "vessel rotated CW (from above)" <=>  eta_yaw = psi > 0
============================================================


## Audit 1 -- Equation-of-motion sign convention (textbook anchor)

The 3-DOF horizontal-plane equation of motion is

    M nu_dot + D nu = tau_total   (body frame, anchor above)

where M, D are positive-definite. With nu = (u, v, r) and
eta = (x, y, psi), eta_dot = R(psi) nu (R = identity in instantaneous
body-aligned analysis at psi=0).

Direct consequence (per anchor):
    tau_total[surge] > 0  =>  vessel accelerates AHEAD          =>  +eta_surge
    tau_total[sway]  > 0  =>  vessel accelerates TO STARBOARD  =>  +eta_sway
    tau_total[yaw]   > 0  =>  vessel accelerates CW (from above) =>  +eta_yaw

>>> Convention: tau on RHS has the *physical* sign of the resultant
>>> force on the hull, in body frame, with the anchor signs above. <<<

Status: VERIFIED (textbook).


## Audit 2 -- WindForceModel.force and CurrentForceModel.force

Source: cqa/cqa/vessel.py:83-141.

Documented convention (cqa internal, vessel.py:88-89):
    "0 = head wind, pi/2 = wind on port beam ... 'wind from' direction
     relative to bow, positive towards port".

i.e. `+theta_rel = source bears toward PORT side of vessel`. Equivalently,
the *direction the wind/current is going* is opposite to theta_rel + pi
(approximately) -- but the convention is anchored by the FROM bearing.

Direct probe (Vw=10 m/s, Vc=1 m/s, CSOV vessel):

    theta_rel  | Convention claim    |    Fx [N]   |    Fy [N]   | Verdict
    +pi/2      | source from PORT    |       0     |   +96744    | OK (vessel pushed STARBOARD)
    -pi/2      | source from STBD    |       0     |   -96744    | OK (vessel pushed PORT)
     0         | source from BOW     |    -30013   |        0    | OK (vessel pushed AFT)
     pi        | source from STERN   |    +30013   |        0    | OK (vessel pushed FWD)

Same pattern for CurrentForceModel.force at Vc=1.

>>> RESULT: WindForceModel.force and CurrentForceModel.force are SELF-
>>> CONSISTENT with the cqa internal convention "+theta_rel = source
>>> from PORT", and that convention is PHYSICALLY CORRECT (matches the
>>> body-frame anchor). <<<

Status: VERIFIED CORRECT (no fix needed here).


## Audit 3 -- pdstrip mean drift force convention

### 3a -- Pdstrip native convention (decoded from brucon source + data)

Source: brucon/libs/dp/vessel_model/wave_response.cpp:60-104, 341-356,
and brucon/libs/dp/vessel_model/response_function.cpp:40-176.

Pdstrip data file columns (csov_pdstrip.dat, 19 cols):
    0:freq  1:enc  2:angle  3:speed
    4..15: surge_r/i .. yaw_r/i  (linear RAOs)
    16:surge_d  17:sway_d  18:yaw_d   (mean drift coefficients)

Brucon ingestion (response_function.cpp:163): drops the first 4 columns,
so internal `response_function[i][j]` indexing is offset by -4:
    raw col 16 (surge_d) -> internal index 12
    raw col 17 (sway_d)  -> internal index 13
    raw col 18 (yaw_d)   -> internal index 14

CalculateMeanDriftForces (wave_response.cpp:341-356) loop:
    for dof = 1..4:  index = 11 + dof  =>  12, 13, 14, (15)
    mean_drift_forces_4dof[dof-1] = weight * sum_i(wave_amp_i^2 * transf_i)
    -> dof=1: surge_d -> mean_drift[0] (surge)
    -> dof=2: sway_d  -> mean_drift[1] (sway)
    -> dof=3: yaw_d   -> mean_drift[2] (yaw)

So brucon takes pdstrip's surge_d / sway_d / yaw_d **DIRECTLY** as the
body-frame components, with no sign flip.

### 3b -- Pdstrip angle <-> body geometry (decoded by reverse-engineering)

Brucon formula (wave_response.cpp:74, 120, 170):
    pdstrip_angle = MapTo360(heading_compass + 180 - wave_direction_from_compass)

Solving for source-bearing-CW-from-bow (= wave_direction_from - heading):
    source_bearing_CW_from_bow = 180 - pdstrip_angle  (mod 360)

Therefore the pdstrip angle <-> source-from-bow geometry is:

    pdstrip angle    source bearing (CW from bow)    intuitive label
        0 deg              180 deg                    source astern (following seas)
       30                  150                        source on stbd-quarter
       90                   90                        source on starboard beam
      150                   30                        source on STARBOARD-BOW
      180                    0                        source on bow (head seas)
      210                  -30 (=330)                 source on port-bow
      270                  -90 (=270)                 source on port beam
      330                  -150                       source on port-quarter

### 3c -- Brucon validation that body frame is stbd=+y (Fossen)

Source: brucon/libs/simulator/vessel_simulator/test/vessel_simulator_model_tests.cpp:493-494
    EXPECT_NEAR(w_resp.MeanDriftForces(heading=90, speed=0)[1], +247268, 18000)
with WaveSpectrum(Hs=4, Tp=9, dominant_dir=0, spread=2)  -> waves from compass 0 (N).
Geometry: bow east, source on port beam.
Physical drift: pushes vessel to starboard => +Fy in body-frame stbd=+y.
Brucon test asserts +247268 -> CONFIRMED stbd=+y body convention.

Same for line 574-575 (heading=45 NE, waves from 0 N, source on port-bow):
    EXPECT_NEAR(MeanDriftForces[0], -60335, ...)   -> -Fx (push astern)  OK
    EXPECT_NEAR(MeanDriftForces[1], +146416, ...)  -> +Fy (push stbd)    OK

### 3d -- Pwq30 specific: brucon-truth Fy

For pwq30: heading=180, wave_from=210
    pdstrip_angle = (180 + 180 - 210) mod 360 = 150 deg
    -> source on STARBOARD-BOW (per 3b table)
    csov_pdstrip.dat sway_d at angle=150, mean over freqs 0.3..1.0 rad/s = -50 kN
    -> brucon-truth Fy = -50 kN (force toward port)
    -> vessel pushed to port, eta_sway < 0
    -> matches independent hand geometry analysis and brucon ensemble PNG.

### 3e -- cqa side: cqa_theta_rel_to_pdstrip_beta_deg mapping audit

Source: cqa/cqa/wave_response.py:103-120.

Current implementation:
    beta_deg = (180 - theta_rel_deg) mod 360
Docstring claim:
    cqa: theta_rel=+pi/2 = "from PORT"
    pdstrip: beta=90 = "beam from PORT (wave going to starboard)"

Verification of pdstrip side from Audit 3a-3d: pdstrip beta=90
corresponds to "source on STARBOARD beam", NOT port. The docstring
label is wrong, and the formula propagates that wrong label into
mean_drift_force_pdstrip.

Probe (Hs=4.2 Tp=10.2 long-crested, csov pdstrip data):

    cqa theta_rel  | pdstrip beta returned  | mean_drift Fy returned  | Brucon truth
    +pi/2 (port)   |     90                 | -311 kN (force to PORT) | +311 kN (force to STBD)
    -pi/2 (stbd)   |    270                 | +311 kN (force to STBD) | -311 kN (force to PORT)
     0    (head)   |    180                 |    0                    |    0  OK
     pi   (follow) |      0                 |    0                    |    0  OK

For the pwq30 cell (cqa theta_rel = -pi/6 after boundary fix):
    pdstrip_beta returned = (180 - (-30)) mod 360 = 210
    -> per pdstrip convention (3b), beta=210 means source on PORT-BOW
    -> mean_drift returns Fy = +135 kN (toward STBD)
    -> brucon-truth for source on STBD-BOW: Fy = -50 kN to -80 kN range
    -> SIGN INVERTED.

The CORRECT mapping is:
    beta_deg = (180 + theta_rel_deg) mod 360
which gives:
    cqa theta_rel  | pdstrip beta corrected  | physical interpretation
    +pi/2 (port)   |    270                  | source on port beam   OK
    -pi/2 (stbd)   |     90                  | source on stbd beam   OK
     0    (head)   |    180                  | head seas             OK
     pi   (follow) |      0 (=360)           | following seas        OK
    -pi/6 (pwq30)  |    150                  | source on stbd-bow    OK matches brucon

>>> RESULT: cqa_theta_rel_to_pdstrip_beta_deg HAS A SIGN BUG.
>>> The fix is to negate the theta_rel_deg term:
>>>    beta_deg = (180 + theta_rel_deg) mod 360
>>> AND update the docstring labels: beta=90 is from STARBOARD, beta=270
>>> is from PORT (not the other way round). <<<

Status: BUG IDENTIFIED. FIX KNOWN. Pending propagation analysis below.


## Audit 4 -- tau_env assembly in wcfdi_transient

Source: cqa/cqa/transient.py:593-616.

    F_wind  = wind_model.force(Vw_mean, theta_rel)         (Audit 2: CORRECT)
    F_curr  = current_model.force(Vc, theta_rel)           (Audit 2: CORRECT)
    if rao_table is not None:
        F_drift = mean_drift_force_pdstrip(rao_table, ..., theta_rel)
                                                           (Audit 3: SIGN-FLIPPED IN sway,yaw)
    else:
        F_drift = (drift_x_amp * Hs^2 * cos(theta_rel),
                   drift_y_amp * Hs^2 * sin(theta_rel),
                   drift_n_amp * Hs^2 * sin(2*theta_rel))  (legacy: CORRECT)
    tau_env = F_wind + F_curr + F_drift

Direct probe (pwq30 cell, Vw=10, Hs=4.2, Tp=10.2, Vc=0.5, theta_rel=-pi/6):

    With CURRENT (buggy) pdstrip path:
      F_wind   = [-25992,  -48372, -518990]
      F_curr   = [ -1131,  -25259, -406512]
      F_drift  = [-60264, +110016, +988951]   <- sway,yaw flipped
      tau_env  = [-87386,  +36385,  +63449]   <- positive sway, vessel pushed STBD

    With pdstrip mapping FIXED (manually negate F_drift sway,yaw):
      F_wind   = [-25992,  -48372, -518990]
      F_curr   = [ -1131,  -25259, -406512]
      F_drift  = [-60264, -110016, -988951]
      tau_env  = [-87386, -183648, -1914453]  <- negative sway, vessel pushed PORT  OK

    Cross-check with LEGACY parametric drift (Hs=2.5, Tp=8, Vc=0.3,
    same theta_rel; this is what regression test
    `test_waves_from_starboard_push_vessel_to_port` actually exercises):
      F_drift_legacy = [+43301, -78125, 0]    <- legacy uses cqa convention,
                                                  sign matches WindForceModel
      tau_env  = [+16903, -135591, -665334]   <- negative sway  OK
    -> The regression test asserts tau_env[1] < 0 and PASSES because the
       legacy path is correct. It would FAIL if rao_table were passed in.

>>> RESULT: Audits 2, 3, 4 jointly confirm: tau_env assembly is CORRECT
>>> if and only if F_drift uses the correct sign convention. The legacy
>>> parametric drift path is correct. The pdstrip path has the
>>> port/starboard angle-mapping bug (Audit 3e). Fixing
>>> cqa_theta_rel_to_pdstrip_beta_deg (negate the theta_rel_deg term and
>>> swap the docstring labels) fixes the entire forecast/drift
>>> assembly. <<<

Status: VERIFIED. Pdstrip-path bug propagates exactly as expected;
legacy path is unaffected.


## Audit 5 -- _augmented_rhs_post tau_env injection sign and tau_lost sign

Source: cqa/cqa/transient.py:400-479.

Current code (line 458):
    nu_dot = Minv_D @ nu + Minv @ tau_thr + Minv @ tau_env - Minv @ tau_lost

Reference (transient_obs.py:48 / live_decision.py:447):
    M nu_dot = -D nu + tau_thr + tau_env + tau_lost
    -> nu_dot = Minv_D @ nu + Minv @ tau_thr + Minv @ tau_env + Minv @ tau_lost

Sign of tau_env: BOTH paths use `+ Minv @ tau_env`.  AGREEMENT.

Sign of tau_lost: _augmented_rhs_post uses `- Minv @ tau_lost`,
                  transient_obs/live_decision use `+ Minv @ tau_lost`.
                  DISAGREEMENT.

Definition of tau_lost (decision_matrix.py:519-527, sec.12.21.17):
    tau_lost(t) := T_post(t) - T_pre = (1 - beta(t)) * tau_env

Physical interpretation:
    T_pre  = thrust BEFORE WCF (delivered, balancing tau_env;
             intact thrusters push opposite to env load: T_pre = -tau_env).
    T_post = thrust AFTER WCF (reduced; deliverable = beta * intact cap;
             for an unsaturated controller, T_post = -beta*tau_env).
    Net change in hull-applied thrust:
        delta_T = T_post - T_pre = -beta*tau_env + tau_env = (1-beta)*tau_env

This delta is **the additional environmental-direction force** that
appears on the hull because the thrusters can no longer fully oppose
tau_env. Its sign matches tau_env (same physical direction as the env
load). So in the equation of motion:

    M nu_dot = -D nu + T_post + tau_env
             = -D nu + (T_pre + delta_T) + tau_env
             = -D nu + T_pre + tau_env + delta_T

If we keep T_pre (the *intact*-controller thrust) as the explicit
tau_thr state and account for the post-WCF deficit via an extra term:
    M nu_dot = -D nu + tau_thr_intact + tau_env + tau_lost
where tau_lost = delta_T = (1-beta)*tau_env.

This matches transient_obs.py:48 and live_decision.py:447: tau_lost
enters with `+`. PHYSICALLY CORRECT.

The `_augmented_rhs_post` `- Minv @ tau_lost` is therefore the BUG.
It implies the deficit pushes the vessel OPPOSITE to the env load,
which is unphysical.

### Independent confirmation via the new regression test

The new regression test exercises legacy-parametric-drift tau_env
(Audit 4 above) with rao_table=None:
    tau_env = [+16903, -135591, -665334]   (sway negative, source from
                                            stbd-bow, force toward port)
    x0_post: tau_thr = -tau_env = [-16903, +135591, +665334]
                                  (intact controller balanced this)
    With current `- Minv @ tau_lost`:
       at t=0+, beta(0) = 0.5 (gamma_imm=0.5)
       hull force = Minv @ (tau_thr + tau_env - (1-beta)*tau_env)
                  = Minv @ (-tau_env + beta*tau_env)
                  = Minv @ ((beta-1)*tau_env)
                  = -0.5 * Minv @ tau_env
                  = -0.5 * (negative) = positive
       -> nu_dot[sway] > 0, vessel goes STARBOARD.
       -> WRONG (env force is toward port, deficit should push hull port-ward).

    With FIXED `+ Minv @ tau_lost`:
       hull force = Minv @ (-tau_env + (1-beta)*tau_env)
                  = Minv @ (-beta*tau_env)
                  = -beta * Minv @ tau_env
                  = -0.5 * (negative) = positive
       -> SAME RESULT?? Let me recheck.

    Hmm. Both signs give the same numerical hull force at t=0+ in this
    setup. Reason: tau_thr - tau_lost vs tau_thr + tau_lost differ by
    2*tau_lost = (1-beta)*tau_env. At beta=0.5, that's +/- 0.5*tau_env;
    different sign of tau_lost gives different sign of hull force.

    Let me redo carefully:
      base = Minv @ (tau_thr + tau_env) = Minv @ (-tau_env + tau_env) = 0
      with  - Minv @ tau_lost: hull = -Minv @ (1-beta)*tau_env
                                    = -0.5 * Minv @ tau_env
                                    = -0.5 * (negative) = +
      with  + Minv @ tau_lost: hull = +Minv @ (1-beta)*tau_env
                                    = +0.5 * Minv @ tau_env
                                    = +0.5 * (negative) = -

    So the two signs of tau_lost give OPPOSITE hull-force directions
    at t=0+, and OPPOSITE eta_sway peaks. Empirical earlier probe:
    eta_sway peak = +0.37 m with current code -> wrong direction.
    Fix to `+` would give -0.37 m -> correct direction.

>>> RESULT: _augmented_rhs_post tau_lost sign IS A BUG. Correct sign:
>>>     nu_dot = ... + Minv @ tau_thr + Minv @ tau_env + Minv @ tau_lost
>>> Matches transient_obs.py:48 and live_decision.py:447. <<<

Status: BUG IDENTIFIED. FIX KNOWN. Independent of Audit 3 (this bug
manifests even on the legacy-drift path, no rao_table required).


### 5b -- Cap-clipping vs explicit tau_lost: is there double-counting?

The `_augmented_rhs_post` has TWO mechanisms that constrain the
post-WCF thrust delivery:

  (i) cap-clipping inside the controller: tau_cmd is clipped to
      cap_at_time(t) before being sent to tau_thr_dot:
        tau_cmd_clipped = clip(tau_cmd, cap_at_time(t))
        tau_thr_dot     = (1/T_thr) * (tau_cmd_clipped - tau_thr)

  (ii) explicit hull-force deficit: + Minv @ tau_lost(t)

Probe (no rao_table, sub-cap pwq30 cell):
  |tau_env|       = [16903, 135591, 665334]
  cap_immediate   = [250000, 350000, 15000000]
  |tau_env| << cap_imm in every DOF -> mechanism (i) NEVER FIRES.

Without tau_lost_fn (mechanism (ii) disabled): sim returns flat
eta = 0 for all t (verified by direct probe). NO excursion.

Brucon truth shows a clear excursion (pwq30 PNG). Therefore the
cap-clipping alone is INSUFFICIENT to model the physics; the
explicit tau_lost injection is required.

Physical interpretation: the WCF instantly removes thrusters from
the hull's force balance. The lost thrusters were producing thrust
in the direction opposite to tau_env (to balance it); their
removal IS a force on the hull in the SAME direction as tau_env,
of magnitude (1-beta(t))*|tau_env|, decaying to (1-alpha)*|tau_env|
or 0 as the survivors re-allocate.

Double-counting concern: only applies when cap-clipping fires
(supra-cap). For CQA-precondition-valid points (|tau_env| <= cap_post),
the clipping does not fire in steady state. Brief immediate-post-WCF
clipping can occur if |tau_env| > cap_immediate, but this is itself
a CQA boundary case worth surfacing rather than papering over.

>>> RECOMMENDATION: apply `+ Minv @ tau_lost` fix in
>>> _augmented_rhs_post. Document the double-counting concern in the
>>> docstring. For supra-cap operating points the result will be
>>> conservatively LARGE (over-estimating deficit), which is a
>>> reasonable failure mode for a pre-CQA guard condition. <<<

Status: Audit 5 COMPLETE. Two-line code fix proposed. Test impact:
test_calibrated_wcfdi.py::test_nonzero_tau_lost_drives_mean_response
needs assertion `< 0` -> `> 0` (the previous flip), AND new pwq30
regression tests will pass.


## Audit 6 -- x0_post (intact_mean_steady_state) vs RHS

Source: cqa/cqa/transient.py:257-284, 619-668.

intact_mean_steady_state returns:
    eta = 0, nu = 0, b_hat = +tau_env, tau_thr = -tau_env, I = 0

Verified analytic: at intact closed-loop steady state with bias FF,
b_hat absorbs the env load and tau_thr cancels it. Vessel is at zero
displacement.

x0_post adjustment (line 667-668):
    tau_thr clipped to cap_immediate per DOF
    -> for sub-cap operations no change; tau_thr stays at -tau_env

This is consistent and physically correct for the pre-WCF state.

>>> RESULT: x0_post is CORRECT. <<<

Status: VERIFIED.


## Audit 7 -- compass -> theta_rel boundary in launchers and decision_matrix

Source: cqa/cqa/decision_matrix.py:680-700 (production formula),
cqa/scripts/p7_brucon_validation/run_comparison*.py (launchers),
cqa/scripts/p7_brucon_validation/diagnose_waves_only_gap.py,
cqa/scripts/p7_brucon_validation/compare_forces.py,
cqa/scripts/p7_brucon_validation/sandbox_passive_observer.py.

Pre-fix (committed code):
    theta_rel = wave_dir_compass - heading_compass

For pwq30: theta_rel = 210 - 180 = +30 deg.

cqa internal convention (Audit 2): +theta_rel = source from PORT.
But +30 deg compass-CW from bow points to STARBOARD-bow.
Boundary mapping is INVERTED (off by negation).

Post-fix (this session, uncommitted, in decision_matrix.py:695 and
all 5 launchers):
    theta_rel = wrap_pi(heading_compass - wave_dir_compass)
For pwq30: theta_rel = 180 - 210 = -30 deg. Source on stbd-bow per
cqa convention (negative theta_rel = source on stbd). MATCHES brucon
geometry.

Cross-check with brucon's mapping:
    pdstrip_angle = (heading + 180 - wave_dir_compass) mod 360
    -> for pwq30: 150 deg -> source on stbd-bow per Audit 3b.
    -> consistent with cqa's theta_rel = -30 deg AFTER the boundary fix.

>>> RESULT: boundary fix is CORRECT (already applied this session). <<<

Status: VERIFIED.


## Audit 8 -- brucon side compass conventions

Source: cqa/scripts/p7_brucon_validation/lua_templates/wcfdi_validation.lua.tpl
and brucon vessel_simulator C++ source (TBD).

The Lua template calls:
    SetWaveCondition(Hs, Tp, wave_dir_compass=210)
    SetVesselSimulatorHeading(180)

Brucon does its own compass->body transform internally and produces the
physically correct sign (the brucon truth that we are anchoring against).

Need to confirm by reading brucon source what the compass-side convention
actually is, to make sure our "anchor" is what we think it is.

Status: PENDING (low priority -- the brucon outcome agrees with hand
geometry analysis, so the brucon convention is internally consistent;
we don't need to touch it).
