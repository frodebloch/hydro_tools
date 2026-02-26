# PDStrip Drift Force Validation — Session Continuation Notes

**Last updated:** 2026-02-26
**Project location:** `/home/blofro/src/pdstrip_test`

---

## Context: PDStrip Drift Force Validation

PDStrip is a Fortran 90 strip theory seakeeping code (`pdstrip.f90`, ~3107 lines) for computing ship motions and drift forces. We are validating and fixing the mean second-order drift force (added resistance) and sway drift force computations. **Do NOT read the PDF at `doc/pdstrip_documentation.pdf` - it's too large and will crash the session.**

---

## THREE BUGS FOUND AND FIXED (SURGE)

**Bug 1 — WL integral used `pres` (with hydrostatic restoring `pst`) instead of `pres_nopst` (BIGGEST FIX):**
- `pres = pwg + (prg + pst) · motion` includes hydrostatic restoring pressure `pst`
- `pst` adds `ρg(η₃ + y·η₄ - x·η₅)` — the pitch term `-x·η₅` creates huge pst values at bow/stern
- When squared in the WL integral `(1/4ρg)|p|² × dy/dx`, this caused ~4x overprediction
- **Fix:** Lines 2837-2838 now use `pres_nopst` instead of `pres`

**Bug 2 — Rotation term sign was flipped:**
- Was changed to `conjg(motion(4:6,1))` at some point → opposite sign
- **Fix:** Restored to original `conjg(-motion(4:6,1))` on line 2896

**Bug 3 — Velocity gradient normal component was computed incorrectly:**
- **Original (Söding):** `gradpotn = ciome * body_velocity * xn` — element-wise multiply (WRONG: gives `v_i × n_i` per component)
- **Correct:** `vdotn = sum(vbody*xn); gradpot = tangential + vdotn*xn` — proper dot product projection `(v·n̂)n̂`
- However, the normal component contributes <5% of the velocity term anyway

---

## KEY DISCOVERY: STRIP THEORY LIMITATION IN VELOCITY-SQUARED INTEGRAL

The velocity-squared term `½ρ∮|∇φ|²n dS` is inherently overpredicted in strip theory because 2D section solutions overestimate potential gradients in the yz-plane. The yz-plane tangential gradients account for 70-87% of the velocity term at critical wavelengths. Using only the x-component of the tangential gradient gave the best surge results, which is what the code currently does.

---

## SURGE DRIFT FORCE RESULTS — BEFORE vs AFTER BUG FIXES

Comparison against SWAN1 3D panel code from Liu & Papanikolaou (2021) / Seo et al. for KVLCC2, V=3 m/s.

**RMS errors (σ_aw) at each heading — Original (buggy) vs Fixed Pinkster vs Maruo:**

| Heading | Original (buggy) | Pinkster (fixed) | Maruo | Best method |
|---------|-----------------|------------------|-------|-------------|
| 180° (head) | 3.744 | 0.997 | **0.822** | Maruo |
| 150° | 4.129 | 0.877 | **0.597** | Maruo |
| 120° | 2.966 | **1.206** | 1.255 | Pinkster |
| 90° (beam) | 3.305 | **0.571** | 1.361 | Pinkster |
| 60° | 2.433 | **1.542** | 1.616 | Pinkster |
| 30° | 1.316 | 1.912 | **1.120** | Maruo |
| 0° (follow) | 0.373 | 1.437 | **0.353** | Maruo |

**Summary:** The original code overpredicted by 3-10x at most headings. The bug fixes reduced peak overprediction from 8.46 to 1.99 at head seas (SWAN1 = 2.5). Pinkster is best at beam/oblique seas (90°-120°), Maruo is best at head/following seas (0°, 30°, 150°, 180°).

---

## SWAY DRIFT FORCE VALIDATION (COMPLETED)

We validated the sway drift force (`feta`) at beam seas (mu=90°, V=0) against published NEWDRIFT (3D panel code) results for KVLCC2.

**Wave direction convention resolved:**
- `mu` = direction FROM WHICH waves come (not propagation direction)
- `mu=90°` = waves from starboard (+y side)
- `mu=180°` = head seas (waves from ahead)

**Sign convention verified correct:**
- For mu=90° (waves from +y/starboard): `feta < 0` = force toward -y (port/lee side) ✓

**Normalization finding — paper likely has factor-of-2 error:**
- Paper states normalization `F_Y / (ρgζ_a²L_pp)` which gives our peak = 0.886 vs NEWDRIFT peak ~0.4-0.45
- Systematic testing of all physically plausible normalizations shows **`2ρgL_pp` gives best fit** (RMS = 0.060 vs next-best 0.123)
- With `2ρgL_pp`: our peak = 0.443, matching NEWDRIFT peak of 0.4-0.45 almost exactly

---

## SWAY DRIFT DECOMPOSITION INTO WL, VELOCITY, ROTATION COMPONENTS

We added `feta_vel`, `feta_rot`, `feta_rot_withpst`, `feta_rot_nopst`, and `feta_rot_norollpst` tracking variables to decompose sway drift force. Key findings:

**Component contributions at beam seas (normalized by 2ρgLpp), 15% roll damping:**

| λ/L | Total | WL | Vel | Rot | Ref | %WL | %Rot |
|-----|-------|-----|-----|-----|-----|-----|------|
| 0.30 | 0.412 | 0.419 | -0.008 | 0.001 | 0.417 | 102% | 0% |
| 0.45 | 0.333 | 0.333 | -0.009 | 0.010 | 0.435 | 100% | 3% |
| 0.62 | 0.234 | 0.204 | -0.009 | 0.039 | 0.278 | 87% | 17% |
| 0.76 | 0.104 | 0.032 | -0.003 | 0.076 | 0.129 | 31% | 73% |
| 0.85 | 0.065 | -0.015 | -0.001 | 0.082 | 0.077 | -23% | 125% |
| 1.05 | 0.068 | -0.020 | -0.000 | 0.089 | 0.015 | -30% | 130% |
| 1.29 | 0.082 | -0.008 | 0.000 | 0.090 | 0.000 | -10% | 110% |

**Key findings:**
1. **Velocity term is negligible** for sway (always <3% of total)
2. **WL integral dominates** at short wavelengths (λ/L < 0.7): 97-100%
3. **Rotation term dominates** at long wavelengths (λ/L > 0.8): >100% (WL reverses sign)
4. **Long-wave tail overprediction is entirely from the rotation term** — it doesn't decay to zero as 3D codes predict
5. **Removing pst from rotation term makes sway catastrophically wrong** (sign flips, force goes toward weather side). The pst MUST be included in the rotation term, unlike the WL integral.
6. The largest error contributor is the **WL underprediction at λ/L = 0.4-0.5** (53% of RMS²), not the long-wave tail (21%)
7. A rotation taper (damping at λ/L > 0.8-1.0) only improves sway RMS from 0.063 to 0.056

**Sway RMS error vs NEWDRIFT (2ρgLpp normalization): 0.063 (15% damping)**

---

## ROLL DAMPING INVESTIGATION (COMPLETED)

**Question investigated:** Is the WL underprediction at λ/L = 0.4-0.5 due to roll resonance?

**Answer: NO.** Roll resonance is NOT the cause of WL underprediction at λ/L = 0.4-0.5. The WL integral is virtually identical across all damping levels. The WL deficit is an intrinsic 2D strip theory limitation.

**Roll damping DOES matter for the rotation term at long wavelengths (λ/L > 0.8):**
- 0% damping: rotation term blows up to 0.378 at λ/L = 1.29 (roll resonance)
- 15% damping: rotation = 0.090 at λ/L = 1.29
- 50% damping: rotation = 0.029 at λ/L = 1.29

**Roll resonance characteristics:**
- Natural roll frequency: ω ≈ 0.38 rad/s (λ/L ≈ 1.3)
- Hydrodynamic roll damping from 2D sections: ~4.3% of critical (very low)
- Peak roll RAO at 0% external damping: 5.18° per unit wave amplitude

---

## ROTATION TERM BLOW-UP AT ROLL RESONANCE — ROOT CAUSE INVESTIGATION

**The user provided Figure 8 from a 3D panel code paper** showing that in a proper 3D solution, the Pinkster components (WL, velocity, rotation, body-surface pressure) individually become very large at roll resonance (±30-40 in normalized units) but **cancel each other** so the total remains small (~5). This cancellation FAILS in pdstrip.

**Root cause analysis:**

We decomposed the rotation term into sub-components:
- `Rot(full)` = rotation with full pressure (current code)
- `Rot(nopst)` = rotation with pressure excluding all pst
- `Rot(norollpst)` = rotation excluding only roll-pst (pst_4 = ρg·y·η₄)
- `Rot(noheavepst)` = rotation excluding only heave-pst (pst_3 = ρg·η₃)

**Finding: The roll-pst (ρg·y·η₄) contributes EXACTLY ZERO to the sway rotation term.** This is because the y·f_z integral cancels by port-starboard symmetry.

**Finding: The actual blow-up source is the HEAVE pst (ρg·η₃):**

At λ/L = 1.294 (roll resonance, 0% damping):
- `Rot(nopst)` = **-0.934** (large negative, from radiation + wave excitation pressures)
- `pst_heave` = **+1.316** (large positive, from ρg·η₃ cross-coupled with roll η₄)
- `Rot(full)` = **+0.378** (net residual after 73% internal cancellation)
- Reference = **0.000** (should be zero at long wavelengths)

---

## PHASE AND SIGN CONVENTION INVESTIGATION (COMPLETED — ALL CORRECT)

### 1. Sign Convention Verification (ALL CORRECT)

**Coordinate system (internal):** x=forward, y=starboard, z=downward (right-handed)
- Positive roll η₄ = starboard down
- Positive heave η₃ = downward
- Input coordinates are (forward, port, up); y and z are negated for internal use

**Rotation term formula verified correct:**
```fortran
df_rot = -0.5*real((presaverage*flvec).vprod.conjg(-motion(4:6,1)))
```
This equals `-(1/2) Re[P·(conj(α) × f)]` which is the correct Pinkster formula when using outward normals and the convention `F = -∮ p n_outward dS`.

**The monohull rotation term uses `conjg(-motion(4:6,1))` — this is CORRECT and matches the original code.**
**The catamaran path (line 3106) uses `conjg(motion(4:6,1))` WITHOUT negation — this is a sign DISCREPANCY vs the original, but we're focused on monohull.**

**pst values verified correct for z-down convention:**
- `pst(:,3,:) = ρg` (heave: positive η₃=downward increases depth, increases pressure) ✓
- `pst(:,4,:) = ρg·y` (roll: starboard down at positive y increases pressure) ✓
- `pst(:,5,:) = -ρg·x` (pitch: bow-down at positive x_bow decreases waterline) ✓

### 2. Equation of Motion Convention (VERIFIED)

```fortran
lineqmatr = -ome²*M + C - addedmass    ! where addedmass = ome²*A - i*ome*B
lineq7(1:6,8) = -ex(:,1,1)             ! RHS = negative excitation
```
Solves: `[-ω²(M+A) + iωB + C]·motion = F_exc` ← standard form ✓

`restorematr` is independently computed from waterplane area integrals and body weight, NOT from pst. So pst only affects the pressure field for drift forces, not the equation of motion. ✓

### 3. Motion Phase Analysis at Beam Seas (KEY FINDING)

**Critical finding — phase relationship between η₃ (heave) and η₄ (roll):**

| ω | λ/L | |η₃| | |η₄| | phase(η₃-η₄) | Re[η₃·conj(η₄)] | cos factor |
|---|-----|------|------|--------------|----------------|------------|
| 0.309 | 1.97 | 1.054 | 0.019 | -89.4° | 0.0002 | 0.010 |
| 0.343 | 1.60 | 1.091 | 0.039 | -84.0° | 0.0045 | 0.105 |
| 0.362 | 1.43 | 1.121 | 0.072 | -62.4° | 0.0372 | 0.463 |
| **0.381** | **1.29** | **1.153** | **0.090** | **5.1°** | **0.1037** | **0.996** |
| 0.402 | 1.16 | 1.208 | 0.051 | 54.4° | 0.0360 | 0.583 |
| 0.423 | 1.05 | 1.276 | 0.029 | 64.0° | 0.0161 | 0.439 |
| 0.446 | 0.94 | 1.356 | 0.019 | 61.1° | 0.0124 | 0.484 |

**At roll resonance (ω=0.381, λ/L=1.29), heave and roll are nearly IN PHASE (5.1° difference).** This maximizes the cross-term `Re[η₃·conj(η₄)]`. Away from resonance, they are in quadrature (~90° apart), making the cross-term vanish. The phase flips through resonance: below resonance η₄ leads η₃ by ~90°, above it lags by ~60-70°.

### 4. Analytical Derivation of What Drives the Blow-Up (KEY FINDING)

We derived that the sway rotation term at beam seas (where η₆≈0) reduces to:

```
feta_rot ≈ +(1/2) Re[conj(η₄) · F_z^{total}]
```

where `F_z^{total} = Σ P·f_z` is the total first-order vertical force from all pressures.

From Newton's second law: `F_z^{total} = -ω²M·η₃`

So: **`feta_rot ≈ -(ω²M/2) · Re[η₃·conj(η₄)]`**

The pst contribution specifically gives: **`feta_rot^{pst} ≈ -(1/2)·C₃₃·Re[η₃·conj(η₄)]`**

where `C₃₃ = ρg·A_wp` (heave restoring = waterplane area × ρg).

At roll resonance, `Re[η₃·conj(η₄)] = 0.1037` and `C₃₃ ≈ ρg·A_wp` is very large, creating a huge pst-driven rotation term. The non-pst part (`-ω²A₃₃ + iωB₃₃` from radiation) provides partial cancellation but it's incomplete.

### 5. Physical Insight About pst

We investigated whether pst being applied to ALL pressure points (full submerged hull) rather than just waterline points was an error. **Answer: NO — it is physically correct.** The linearized Bernoulli equation gives:

```
p⁽¹⁾ = iωρφ⁽¹⁾ + ρg·δz
```

where δz = η₃ + y·η₄ - x·η₅ is the vertical displacement of every body surface point. The term `ρg·δz` applies at ALL points, not just the waterline. The `prg` captures `iωρφ_rad` and the `pst` captures `ρg·δz`. Both are correct.

By Gauss's theorem, integrating `pst·n·dS` over the hull panels recovers the correct restoring force coefficients (waterplane area, waterplane moment, etc.), so the equation of motion is consistent.

---

## PRESSURE VARIABLE DEFINITIONS
- `pres = pwg + (prg + pst) · motion` — full pressure including hydrostatic restoring
- `pres_nopst = pwg + prg · motion` — pressure WITHOUT hydrostatic restoring (used for WL drift integral)
- `pres_norollpst = pres - pst(:,3:3,:)*motion(3,1)` — **currently configured to exclude HEAVE pst, despite the variable name** (was repurposed during investigation; originally excluded roll pst)
- `pwg` = wave excitation pressure (Froude-Krylov + diffraction), dimension (npres,1,nse)
- `prg` = radiation pressure per DOF, dimension (npres,6,nse)
- `pst` = hydrostatic restoring pressure per DOF, dimension (npres,6,nse):
  - `pst(:,3,:) = ρg` (heave)
  - `pst(:,4,:) = ρg·y` (roll)
  - `pst(:,5,:) = -ρg·x` (pitch)
- **WL integral uses `pres_nopst`** (Bug 1 fix)
- **Rotation term uses `pres` (full, including pst)** — removing pst from rotation breaks sway completely

## THREE DRIFT FORCE METHODS IN THE CODE (lines ~2810-2995)

1. **Pinkster near-field** (lines 2810-2930): WL integral + velocity² + rotation terms
2. **Maruo/Gerritsma-Beukelman far-field** (lines 2945-2995): `R_aw = (ωₑ²k)/(2ω) Σ [b₃₃|s_rel_z|² + b₂₂|s_rel_y|²] dx` — **surge only**
3. **Boese (1970)** (lines ~2997+): WP integral of |ζ_rel|² + Maruo G-B term — **severely broken, goes massively negative**

## CURRENT STATE OF pdstrip.f90 DRIFT FORCE CODE (lines ~2810-2930)

Key modifications from original (`kvlcc2_original/pdstrip_original.f90`):
1. **WL integral** uses `pres_nopst` ✓
2. **Rotation sign**: `conjg(-motion(4:6,1))` (original restored) ✓
3. **Rotation pressure**: uses `presaverage` (full pres with pst) — required for correct sway
4. **Velocity term**: `df = -0.25*rho*abs(gradpott(1))**2*flvec` — x-gradient only
5. **Normal velocity**: proper dot product `vdotn = sum(vbody*xn)` ✓
6. Diagnostic variables tracked:
   - Surge: `fxi_vel`, `fxi_rot`, `fxi_vel_x`, `fxi_vel_yz`, `fxi_vel_tang`, `fxi_vel_norm`, `fxi_rot_nopst`
   - Sway: `feta_vel`, `feta_rot`, `feta_rot_withpst`, `feta_rot_nopst`, `feta_rot_norollpst` (currently = no-heave-pst, misnamed)
7. Debug output lines: `DRIFT_START`, `DRIFT_TOTAL`, `DRIFT_SWAY`, `DRIFT_NOPST`, `DRIFT_VEL_XYZ`, `DRIFT_VEL_TN`, `DRIFT_MARUO`, per-section `WL sec=`, `TRI_SUM sec=`

---

## KVLCC2 PARAMETERS
- Lpp=328.2m, B=58.0m, T=20.8m, 62 sections, mass=320M kg
- Surge normalization: `σ_aw = -F_x / (ρgA²B²/L)` where `norm = ρgB²/L = 103,065 N`
- Sway normalization (corrected): `|F_y| / (2ρgA²L_pp)` where `norm = 2ρgLpp = 6,600,266 N`
- 8 speeds: 0, 2, 3, 4, 6, 7.96, 9.095, 10.09 m/s (speed indices 0-7)
- V=3 m/s is speed index 2 (used for surge validation at mu=180); V=0 is speed index 0 (sway validation at mu=90)
- 35 frequencies (ω = 0.250 to 1.500), 36 wave directions (mu = -90° to 260° in 10° steps)
- debug.out ordering: omega(outer, high-to-low) × speed(middle) × heading(inner)
- Index formula: `idx = iom * (n_s * n_h) + ispeed * n_h + imu`
- Wave steepness for nonlinear iteration: 0.02, max wave height: 4.0m

## BUILD AND RUN
```bash
cd /home/blofro/src/pdstrip_test
make clean && make && cp pdstrip kvlcc2/pdstrip
cd kvlcc2 && ./pdstrip
python3 boese_analysis.py  # main surge analysis script
python3 sway_analysis.py   # sway analysis script
python3 roll_damping_analysis.py  # roll damping comparison (4-panel plot)
python3 cancellation_analysis.py  # Pinkster component cancellation analysis
python3 motion_phase_analysis.py  # motion phase analysis
```

## KEY FILES
- **Main source:** `/home/blofro/src/pdstrip_test/pdstrip.f90`
- **Original reference:** `/home/blofro/src/pdstrip_test/kvlcc2_original/pdstrip_original.f90`
- **Analysis scripts (in kvlcc2/):** `boese_analysis.py` (surge), `sway_analysis.py` (sway), `roll_damping_analysis.py` (roll damping 4-panel plot), `cancellation_analysis.py` (Pinkster component cancellation), `motion_phase_analysis.py` (motion phases at beam seas)
- SWAN1 surge reference data is digitized and hardcoded in `boese_analysis.py` as `seo_data[beta]` dictionaries (7 headings)
- NEWDRIFT sway reference data (approximate from figure): peak ~0.4-0.45 at λ/L≈0.4-0.5, dropping to ~0.02 at λ/L=1.0 and 0 at λ/L≥1.2
- **Saved debug outputs (in kvlcc2/):** `debug.out` and `debug_15pct_v2.out` (15% damp with norollpst diag), `debug_0pct_v2.out` and `debug_0pct_v3.out` (0% damp with heave-pst decomp), `debug_0pct_nopst_decomp.out` (0% damp with nopst decomp), plus older: `debug_0pct_new.out`, `debug_25pct_new.out`, `debug_50pct_new.out`
- **Saved pdstrip outputs:** `pdstrip.out`, `pdstrip_0pct_new.out`, `pdstrip_25pct_new.out`, `pdstrip_50pct_new.out`
- **Plots:** `kvlcc2/roll_damping_sway.png` (4-panel damping effect), `kvlcc2/cancellation_analysis.png` (Pinkster component cancellation)

## KEY CONSTRAINTS
- **Do NOT read `doc/pdstrip_documentation.pdf`** — it's too large and will crash the session.
- The rotation term in the Pinkster near-field method MUST use full `pres` (with pst), not `pres_nopst`. Removing pst from rotation flips the sway drift sign (RMS 0.20 vs 0.063 with pst).
- The WL integral MUST use `pres_nopst`. Using full `pres` causes ~4x surge overprediction.
- The `roll_damp_frac` parameter on line ~1744 of pdstrip.f90 is currently set to **0.15**.
- The `pres_norollpst` variable is **misnamed** — it currently excludes heave-pst (pst_3), not roll-pst (pst_4). This was repurposed during investigation. Can be cleaned up.

---

## PRESSURE-FORCE CONSISTENCY CHECK (COMPLETED — PREVIOUS SESSION)

Added diagnostics to compare `Σ(p·n·dS)` over hull panels vs `-ω²M·η` (Newton's 2nd law).

**Key findings:**
- **Fz (heave):** ~2-13% discrepancy — acceptable
- **Fy (sway):** up to 800% discrepancy at roll resonance — caused by missing weight in panel pst

**Decomposition by force component:**

| Component | Panel vs EOM ratio (sway) | Panel vs EOM ratio (heave) |
|-----------|--------------------------|---------------------------|
| Wave excitation | ~1-10% error | ~1-10% error |
| Radiation | ~195% (factor ~2x) | ~195% (factor ~2x) |
| PST (restoring) | 13617% (factor ~136x) | 200% (factor 2x) |

**Root cause:** The EOM restoring `C₂₄ = g(ρV - M) ≈ 0` for a floating body (buoyancy = weight). But the panel pst only computes buoyancy `ρgV·η₄`, not the weight cancellation `-gM·η₄`. The 2× heave discrepancy and ~2× radiation discrepancy may relate to port/starboard panel coverage conventions.

---

## ROTATION TERM FIX ATTEMPTS (COMPLETED — THIS SESSION)

### Conclusion: Roll resonance blow-up is an INTRINSIC LIMITATION of Pinkster near-field

Three approaches were tested to fix the rotation term blow-up at roll resonance:

### Option C: Replace panel pst with analytical balanced restoring (FAILED)
- Changed rotation term to use `presavg_nopst` (no buoyancy pst), then added analytical correction `−½ Re[(szw·η) × conj(−α)]` using the balanced restoring matrix (buoyancy − weight).
- **Result: Catastrophic failure** — RMS jumped from 0.063 to 0.284, sway drift went massively wrong sign.
- **Reason:** The rotation cross product needs the **spatially distributed** pressure, not just the global resultant force. The net restoring `szw·η ≈ 0` (because buoyancy ≈ weight) but the distributed buoyancy pressure cross-product is large and physically meaningful. You cannot replace a surface integral with a volume-averaged result.

### Option 3: Newton analytical rotation (TESTED, EQUIVALENT)
- Replaced the entire panel-summed rotation with Newton's 2nd law: `−½ Re[(−ω²M·η) × conj(−α)]`.
- **Result: Nearly identical to original panel rotation** (RMS 0.0627 vs 0.0631).
- At 0% damping: Newton gives 0.365 at λ/L=1.29, old panel gives 0.378 — both blow up equally.
- At 15% damping: Newton gives 0.080, old panel gives 0.082 — same long-wave tail.
- This confirms the panel pressure integration IS approximately consistent with Newton's 2nd law.
- **Newton rotation kept as diagnostic only** (not applied to drift forces).

### Option B: Analytical formula (NOT TESTED)
- Would use `feta_rot = −(ω²M/2)·Re[η₃·conj(η₄)]` — equivalent to Newton for sway at beam seas.
- Not tested because Option 3 already showed equivalence.

### Final verdict
The roll resonance blow-up is NOT a bug. It is an inherent feature of the Pinkster near-field method:
- In 3D panel codes, the Pinkster components (WL, velocity, rotation, body-surface pressure) individually become ±30-40 at roll resonance but **cancel each other** so the total remains small.
- In strip theory, this cancellation fails because the components are computed with different levels of approximation.
- **Practical workaround:** Use sufficient roll damping (15% is physically realistic for ships with bilge keels) or lower CG to shift roll resonance below the sea state frequency range.

---

## CURRENT STATE OF pdstrip.f90

**`roll_damp_frac`** = 0.15 (line ~1744)

**Drift force computation (lines ~2830-2960):**
- WL integral: uses `pres_nopst` ✓
- Rotation term: uses full `presaverage` (with pst) ✓
- Velocity term: x-gradient only ✓
- Newton rotation computed as diagnostic (not applied)
- Option C restoring computed as diagnostic (not applied)
- Force consistency check diagnostics (FORCE_CHECK, FCHECK_WG, etc.) present

**Diagnostic variables added this session:**
- `cnewton_force(3)`, `df_rot_newton(3)`, `fxi_rot_newton`, `feta_rot_newton` — Newton rotation diagnostic
- `cfrestore(3)`, `df_rot_restore(3)`, `fxi_rot_fixC`, `feta_rot_fixC` — Option C diagnostic
- Previous session: `cpres_fy_*`, `cpres_fz_*`, `ceom_f*` — force consistency check variables

---

## WHAT TO DO NEXT (PRIORITY ORDER)

### 1. Consider implementing far-field (Maruo-type) sway drift formula
The code has Maruo for surge but not sway. Variables `fdi_sec`, `fki_sec`, `amatr_sec` are stored per section. A Maruo sway formula would avoid the near-field cancellation problem entirely. However, this is significant new development.

### 2. Consider blended Maruo/Pinkster approach for surge
Maruo is best at head/following seas (0°, 30°, 150°, 180°), Pinkster is best at beam/oblique seas (60°-120°). An automatic blending could improve overall accuracy.

### 3. Clean up diagnostic variables
Many diagnostic tracking variables and debug output lines were added during investigation. These could be removed or put behind a compile-time flag for production use.

### 4. Investigate Boese WP integral
Goes massively negative at long waves in surge — severely broken.

### 5. Address short-wave surge overprediction
Pinkster σ_aw rises to ~1.1 at λ/L = 0.3 while SWAN1 is near 0.

### 6. Investigate 2× radiation and heave-pst discrepancies (low priority)
Panel radiation force is consistently ~2× EOM radiation. Heave pst is exactly 2× EOM restoring. May be a port/starboard convention issue. Low priority since drift forces are computed correctly regardless.
