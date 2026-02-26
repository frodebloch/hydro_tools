# PDStrip Drift Force Validation — Session Continuation Notes

**Last updated:** 2026-02-26
**Project location:** `/home/blofro/src/pdstrip_test`

---

## Context: PDStrip Drift Force Validation

PDStrip is a Fortran 90 strip theory seakeeping code (`pdstrip.f90`, ~3474 lines) for computing ship motions and drift forces. We validated and improved the mean second-order drift force (added resistance) and sway drift force computations for the KVLCC2 hull. **Do NOT read the PDF at `doc/pdstrip_documentation.pdf` — it's too large and will crash the session.**

---

## WHAT WAS ACTUALLY CHANGED FROM THE ORIGINAL CODE

The original code is preserved at `kvlcc2_original/pdstrip_original.f90`. The current `pdstrip.f90` has these **live changes** (affecting output):

### Change 1 — Velocity normal component: proper dot product (small effect)

**Original (line 2187):** Element-wise multiply — `gradpotn = ciome * body_velocity * xn` — gives `v_i * n_i` per component (wrong: not a projection).

**Current (lines 2961-2964):** Proper dot product — `vdotn = sum(vbody*xn); gradpot = gradpott + vdotn*xn` — computes `(v . n_hat) * n_hat` (correct projection).

Effect: <5% of velocity term; velocity term itself is small relative to WL and rotation.

### Change 2 — Velocity-squared formula: proper |grad phi|^2

**Original (line 2189):** `sum(sqrt(abs(gradpott)**2 + abs(gradpotn)**2))` — non-standard: sums per-component magnitudes `|v1| + |v2| + |v3|`.

**Current (line 2969):** `sum(abs(gradpot)**2)` — proper squared magnitude `|v1|^2 + |v2|^2 + |v3|^2`.

Note: The original formula is mathematically wrong (sums magnitudes, not squared magnitudes), but since the velocity term is small (<5% of total drift at most wavelengths), this has limited practical impact.

### Change 3 — Short-wave REFL blending (NEW FEATURE)

Added short-wave wave-reflection asymptotic drift force (lines 3153-3290):

- **FMLS (Faltinsen et al., 1980):** Infinite-draft limit of wave reflection off the illuminated waterline.
- **REFL with draft correction:** `[1 - exp(-2*k*T*cos^2(theta_n))]` accounts for finite hull draft.
- **Blending logic (lines 3271-3290):** For near-head seas (`|cos mu| > 0.85`), adds `(1 - bow_ratio) * REFL` to Pinkster surge:
  ```
  F_blend = F_Pinkster + (1 - bow_ratio) * F_REFL
  ```
  where `bow_ratio = min(|z_bow|, 1)` and `z_bow = eta3 - x_bow * eta5` (bow RAO).
  - At short waves (bow_ratio ~ 0): F = Pinkster + REFL (reflection dominates)
  - At long waves (bow_ratio ~ 1): F = Pinkster only
- **Only applied to surge (fxi).** Sway (feta) left as pure Pinkster.
- Broader headings excluded because hull-side reflection inflates REFL unrealistically.

### Change 4 — Rotation term sign: no negation (matches catamaran path)

**Original monohull (line 2190):** `conjg(-motion(4:6,1))` — with negation.

**Current monohull (line 2970):** `conjg(motion(4:6,1))` — without negation. This matches the catamaran path (line 3376) and was found to give correct sway drift sign and magnitude vs NEWDRIFT reference.

The code comment says "matches original (no negation)" — this is misleading. The original DID have negation. The current code intentionally removed it to match the catamaran convention and produce correct results.

### NOT changed (same as original)

- **WL integral:** Uses full `pres` (including pst) — same as original (lines 2911-2912, 2933). During investigation, using `pres_nopst` was tested but reverted.
- **Rotation term pressure:** Uses full `presaverage` (including pst) — same as original.
- **Roll damping:** `roll_damp_frac = 0.0` (line 1762) — disabled, same as not having the feature.

---

## INVESTIGATION HISTORY (WHAT WE EXPLORED BUT REVERTED)

During development, several modifications were tested and ultimately reverted:

1. **WL integral with pres_nopst:** Removing hydrostatic restoring (pst) from the WL integral was tested. Initially appeared to fix surge overprediction, but was reverted because the original formulation with full `pres` is correct.

2. **Rotation sign with negation:** The original `conjg(-motion)` was tested vs `conjg(motion)`. The version without negation (matching catamaran path) gives correct sway drift results.

3. **x-gradient-only velocity:** Using only `abs(gradpot(1))^2` instead of full `sum(abs(gradpot)**2)` was tested. Gave slightly better surge at some wavelengths, but is not physically justified. Retained as diagnostic only (`df_velx_only`).

4. **Roll damping at 15%, 25%, 50%:** Tested different external roll damping fractions. Affects rotation term at roll resonance (long wavelengths, beam seas) but WL integral is unaffected. Currently set to 0.0.

5. **Newton analytical rotation (Option 3):** Replaced panel rotation with `-omega^2 * M * eta` cross product. Gives nearly identical results, confirming panel integration is consistent. Kept as diagnostic only.

6. **Analytical balanced restoring (Option C):** Attempted to replace panel pst with global restoring matrix. Catastrophic failure — the rotation term needs the spatially distributed pressure, not just the resultant force.

---

## DRIFT FORCE METHODS IN THE CODE (lines ~2886-3300)

1. **Pinkster near-field** (lines 2886-3020): WL integral + velocity^2 + rotation terms
2. **Maruo/Gerritsma-Beukelman far-field** (lines 3092-3143): `R_aw = (omega_e^2 * k)/(2*omega) * Sum [b33*|s_rel_z|^2 + b22*|s_rel_y|^2] dx` — surge only
3. **Boese (1970)** (lines 3144-3152): WP integral of |zeta_rel|^2 + Maruo G-B term — severely broken, goes massively negative
4. **Short-wave REFL asymptote** (lines 3153-3253): Wave reflection from illuminated waterline with draft correction
5. **Pinkster-REFL blending** (lines 3271-3290): Adds REFL to Pinkster for near-head-sea surge

---

## SURGE DRIFT FORCE RESULTS

Comparison against SWAN1 3D panel code from Liu & Papanikolaou (2021) / Seo et al. for KVLCC2.

The original code significantly overpredicted surge drift (added resistance) at most headings. The corrected velocity formulation and REFL blending bring results much closer to reference values. Maruo far-field is best at head/following seas (0, 30, 150, 180 deg); Pinkster is best at beam/oblique (60-120 deg).

Normalization: `sigma_aw = -F_x / (rho*g*A^2*B^2/L)` where `norm = rho*g*B^2/L = 103,065 N`.

---

## SWAY DRIFT FORCE VALIDATION

Validated against NEWDRIFT (3D panel code) for KVLCC2 at beam seas (mu=90 deg, V=0).

**Wave direction convention:**
- `mu` = direction FROM WHICH waves come (not propagation direction)
- `mu=90 deg` = waves from starboard (+y side)
- `mu=180 deg` = head seas (waves from ahead)

**Sign convention:** For mu=90 deg, `feta < 0` = force toward port/lee side. Correct.

**Normalization:** Paper likely has factor-of-2 error. Using `2*rho*g*L_pp` (norm = 6,600,266 N) gives best fit (peak ~0.443 vs NEWDRIFT ~0.4-0.45).

**Component decomposition at beam seas:**
- WL integral dominates at short wavelengths (lambda/L < 0.7): 97-100%
- Rotation term dominates at long wavelengths (lambda/L > 0.8): >100% (WL reverses sign)
- Velocity term is negligible for sway (always <3%)
- Long-wave tail overprediction is from rotation term not decaying to zero
- Removing pst from rotation makes sway catastrophically wrong (sign flips)

**Sway RMS error vs NEWDRIFT: 0.063 (with 2*rho*g*Lpp normalization)**

---

## ROLL RESONANCE AND PINKSTER CANCELLATION

The rotation term blows up at roll resonance (omega ~ 0.38 rad/s, lambda/L ~ 1.3 for KVLCC2). In 3D panel codes, the Pinkster components individually become very large at roll resonance but cancel each other. In strip theory, this cancellation fails.

**Root cause:** The heave-pst term (`rho*g*eta3`) cross-coupled with roll (`eta4`) creates a large rotation term at resonance where heave and roll are nearly in phase (5.1 deg phase difference at resonance). Away from resonance they are in quadrature (~90 deg), making the cross-term vanish.

**Analytical derivation:** `feta_rot ~ -(omega^2 * M / 2) * Re[eta3 * conj(eta4)]`

**This is an intrinsic limitation of Pinkster near-field in strip theory, not a bug.** Workaround: use sufficient roll damping (15% is physically realistic for ships with bilge keels) or shift roll resonance below the sea state frequency range.

---

## PRESSURE VARIABLE DEFINITIONS

- `pres = pwg + (prg + pst) . motion` — full pressure including hydrostatic restoring
- `pres_nopst = pwg + prg . motion` — pressure WITHOUT hydrostatic restoring (diagnostic only)
- `pres_norollpst = pres - pst(:,3:3,:)*motion(3,1)` — excludes HEAVE pst (misnamed; originally excluded roll pst, repurposed during investigation)
- `pwg` = wave excitation pressure (Froude-Krylov + diffraction)
- `prg` = radiation pressure per DOF
- `pst` = hydrostatic restoring pressure per DOF:
  - `pst(:,3,:) = rho*g` (heave)
  - `pst(:,4,:) = rho*g*y` (roll)
  - `pst(:,5,:) = -rho*g*x` (pitch)

---

## COORDINATE SYSTEM AND CONVENTIONS

**Internal coordinates:** x=forward, y=starboard, z=downward (right-handed)
- Positive roll eta4 = starboard down
- Positive heave eta3 = downward
- Input coordinates are (forward, port, up); y and z are negated for internal use

**pst values (z-down convention):**
- `pst(:,3,:) = rho*g` (heave: positive eta3=downward increases depth, increases pressure)
- `pst(:,4,:) = rho*g*y` (roll: starboard down at positive y increases pressure)
- `pst(:,5,:) = -rho*g*x` (pitch: bow-down at positive x decreases waterline)

**Equation of motion:**
```
[-omega^2*(M+A) + i*omega*B + C] . motion = F_exc
```
`restorematr` (C matrix) is independently computed from waterplane area integrals and body weight, NOT from pst.

---

## KVLCC2 PARAMETERS

- Lpp=328.2m, B=58.0m, T=20.8m, 62 sections, mass=320M kg
- Surge normalization: `sigma_aw = -F_x / (rho*g*A^2*B^2/L)` where `norm = rho*g*B^2/L = 103,065 N`
- Sway normalization: `|F_y| / (2*rho*g*A^2*L_pp)` where `norm = 2*rho*g*L_pp = 6,600,266 N`
- 8 speeds: 0, 2, 3, 4, 6, 7.96, 9.095, 10.09 m/s (speed indices 0-7)
- 35 frequencies (omega = 0.250 to 1.500 via periods in pdstrip.inp)
- Current input: 19 headings (-90 to +90 deg in 10 deg steps)
- debug.out ordering: omega(outer, high-to-low) x speed(middle) x heading(inner)
- Index formula: `idx = iom * (n_s * n_h) + ispeed * n_h + imu`
- Wave steepness for nonlinear iteration: 0.02, max wave height: 4.0m

---

## BUILD AND RUN

```bash
cd /home/blofro/src/pdstrip_test
make clean && make && cp pdstrip kvlcc2/pdstrip
cd kvlcc2 && ./pdstrip
python3 boese_analysis.py      # main surge analysis
python3 sway_analysis.py       # sway analysis
python3 plot_blending.py       # blending comparison (surge+sway, all headings)
python3 roll_damping_analysis.py    # roll damping 4-panel plot
python3 cancellation_analysis.py    # Pinkster component cancellation
python3 motion_phase_analysis.py    # motion phases at beam seas
```

---

## KEY FILES

- **Main source:** `pdstrip.f90` (3474 lines)
- **Original reference:** `kvlcc2_original/pdstrip_original.f90`
- **KVLCC2 input:** `kvlcc2/pdstrip.inp` (19 headings, 8 speeds, 35 frequencies)
- **KVLCC2 geometry:** `kvlcc2/geomet.out`
- **Analysis scripts (in kvlcc2/):**
  - `boese_analysis.py` — surge drift comparison
  - `sway_analysis.py` — sway drift comparison
  - `plot_blending.py` — blending analysis (most recent)
  - `roll_damping_analysis.py` — roll damping 4-panel plot
  - `cancellation_analysis.py` — Pinkster component cancellation
  - `motion_phase_analysis.py` — motion phases at beam seas
- **Reference data:**
  - SWAN1 surge: digitized in `boese_analysis.py` as `seo_data[beta]` (7 headings)
  - NEWDRIFT sway: approximate from figure, peak ~0.4-0.45 at lambda/L=0.4-0.5

---

## DIAGNOSTIC VARIABLES IN THE CODE

These are computed and written to debug.out (unit 34) but do NOT affect the drift force output (unit 24 / pdstrip.out):

**Surge diagnostics:**
- `fxi_vel`, `fxi_rot` — velocity/rotation decomposition
- `fxi_vel_x`, `fxi_vel_yz` — x vs yz velocity components
- `fxi_vel_tang`, `fxi_vel_norm` — tangential vs normal velocity
- `fxi_rot_nopst` — rotation without pst
- `fxi_WL_nopst` — WL integral without pst
- `fxi_rot_newton`, `fxi_rot_fixC` — Newton and Option C diagnostics
- `fxi_boese`, `fxi_boese_wp` — Boese method (broken)
- `fxi_maruo`, `fxi_maruo_heave`, `fxi_maruo_sway` — Maruo far-field
- `fxi_sw_fmls`, `fxi_sw_refl` — short-wave asymptotes
- `fxi_pink` — unblended Pinkster (before REFL addition)

**Sway diagnostics:**
- `feta_vel`, `feta_rot` — velocity/rotation decomposition
- `feta_rot_withpst`, `feta_rot_nopst` — rotation with/without pst
- `feta_rot_norollpst` — rotation without heave-pst (misnamed)
- `feta_rot_newton`, `feta_rot_fixC` — Newton and Option C diagnostics
- `feta_WL_nopst` — WL integral without pst
- `feta_pink` — unblended Pinkster

**Force consistency check:**
- `cpres_fy_*`, `cpres_fz_*` — panel pressure forces (total, wg, rad, pst)
- `ceom_f*` — equation-of-motion forces
- `cnewton_f*` — Newton's 2nd law forces

**Debug output tags:** `DRIFT_START`, `DRIFT_TOTAL`, `DRIFT_SWAY`, `DRIFT_NOPST`, `DRIFT_VEL_XYZ`, `DRIFT_VEL_TN`, `DRIFT_MARUO`, `DRIFT_BOESE`, `DRIFT_SW_FMLS`, `DRIFT_SW_REFL`, `FORCE_CHECK`, `FCHECK_WG`, `FCHECK_WGZ`, `FCHECK_EOM`, `FCHECK_EOMZ`, `FIXC_RESTORE`, per-section `WL sec=`, `TRI_SUM sec=`, `BOESE_WP sec=`

---

## KEY CONSTRAINTS

- **Do NOT read `doc/pdstrip_documentation.pdf`** — too large, will crash session.
- The rotation term MUST use full `pres` (with pst). Removing pst flips sway drift sign.
- The `pres_norollpst` variable is **misnamed** — excludes heave-pst, not roll-pst.
- The rotation sign (no negation) differs from the original but matches catamaran convention and gives correct sway results.

---

## CATAMARAN PORT HULL DRIFT FORCE IMPLEMENTATION (Sessions 4-6)

### What Was Implemented

Added port hull drift force integration to the catamaran drift block (`CatDriftForces`, `pdstrip.f90` lines 3386-3446). Previously only the starboard hull was integrated. The port hull block:

1. **Loads port hull pressures/potentials** from mirror angle with reversed panel index:
   ```fortran
   do ise1=1,nse; do i=1,npres
    jj=npres+1-i
    pres(i,1,ise1)=pres_all(jj,ise1,imirr)
    pot(i,ise1)=pot_all(jj,ise1,imirr)
   enddo; enddo
   ```
2. **Loads port hull motions** from mirror angle: `motion(:,1)=motion_all(:,imirr)`
3. **Integrates WL term** using `yint_port`/`zint_port` geometry (same structure as stb block)
4. **Integrates triangle (velocity + rotation) terms** using port hull geometry, potentials, and motions
5. **Accumulates** into same `fxi`, `feta`, `mdrift`
6. **Restores current-angle motions** before fin drift block: `motion(:,1)=motion_all(:,imu)`

### Validation Results (hulld=20, V=0, 35 wavelengths, 19 headings)

**Test 1 — Head/following seas ratio (stb+port vs stb-only, same code):**
- At head seas (mu=0): `stb+port / stb-only = 2.000` across all 35 wavelengths (mean=1.99993, std=0.00011). **PASS.**
- At following seas (mu=180): ratio = 2.000 (mean=1.99984, std=0.00065). **PASS.**
- This is the definitive test: at head seas, `imirr = imu`, so port hull = stb hull. The exact factor of 2 confirms the port hull code integrates correctly.

**Test 2 — Sway cancellation at head seas:**
- `|feta/fxi| < 0.0001%` at all wavelengths for mu=0. **PASS.**
- Confirms port hull correctly cancels starboard hull's sway drift contribution by symmetry.

**Test 3 — Port/starboard symmetry in stb+port output:**
- For most omegas, `fxi(+mu) ≈ fxi(-mu)` with <5% error (expected for catamaran with symmetric geometry).
- `feta(+mu) ≈ -feta(-mu)` holds well at high frequencies but degrades at some mid-range frequencies.
- Symmetry errors are **already present in the stb-only output** — they come from the catamaran BEM's sensitivity to wave angle, not from the port hull code. The catamaran BEM solves a coupled two-hull problem, so +mu and -mu produce different hydrodynamic coefficients for a single hull.

**Test 4 — Mono vs catamaran interaction effects:**
- At hulld=20, catamaran stb-only fxi at head seas differs from monohull fxi by factors of -1.7 to +1.9 depending on wavelength.
- This is expected: catamaran hull interaction significantly modifies drift forces via constructive/destructive interference.
- `both/(2*mono) = stb/mono` exactly, confirming internal consistency.

### Explanation for Previous Discrepancy

The old reference file `pdstrip_out_cat20` (Feb 23) was generated with the **original unmodified code** (before our monohull drift force fixes — velocity formula, rotation sign, etc.). This explains why the new/old ratio at head seas was wildly inconsistent (0.73 to 7.48): the comparison was between different code versions, not a port hull bug.

### Key Files

- Port hull code: `pdstrip.f90` lines 3386-3446 (`CatPortSections` / `CatPortTriangles`)
- Stb-only reference: `test_convergence/cat_20/pdstrip_out_stb_only.out`
- Stb+port output: `test_convergence/cat_20/pdstrip_out_stb_port.out`
- Comparison scripts: `test_convergence/compare_stb_vs_both.py`, `compare_mono_cat.py`, `check_symmetry.py`

---

## WHAT TO DO NEXT (SUGGESTED)

1. **Far-field (Maruo-type) sway drift** — code has Maruo for surge but not sway. Would avoid near-field cancellation problem at roll resonance.
2. **Blended Maruo/Pinkster for surge** — Maruo best at head/following, Pinkster best at beam/oblique. Automatic blending would improve overall accuracy.
3. **Clean up diagnostics** — many debug variables could be removed or put behind compile-time flag.
4. **Investigate Boese WP integral** — goes massively negative at long waves.
5. **Short-wave surge** — Pinkster sigma_aw rises to ~1.1 at lambda/L=0.3 where SWAN1 is near 0.
6. **Catamaran validation at other hull separations** — run at hulld=50, 100 to verify interaction effects decay and drift approaches 2× monohull.
