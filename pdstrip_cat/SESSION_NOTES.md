# PDStrip Drift Force Validation — Session Continuation Notes

**Last updated:** 2026-03-09
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

## NEMOH 3.0 QTF FOR KVLCC2 — INDEPENDENT DRIFT FORCE VALIDATION (Sessions 8-31)

### Motivation

PDStrip is a 2D strip theory code — we need a 3D panel method reference to validate drift forces independently. SWAN1 reference data exists only for head seas; we need full-heading coverage. Nemoh 3.0 is the only open-source BEM code with a built-in QTF (second-order / mean drift force) module.

### Capytaine — ABANDONED (Sessions 10-11)

Wrote 6 debug scripts (`debug_capytaine_drift.py` through `debug_capytaine_drift6.py`). The Capytaine Kochin/Maruo far-field drift approach is fundamentally broken — `compute_kochin` returns diffraction-only Kochin function but Maruo needs total (incident+diffracted). Results were wildly wrong for all test bodies. **Dead end — do not revisit.**

### Nemoh 3.0 Installation & Build

- **Source:** `/home/blofro/src/Nemoh/`
- **Build dir:** `/home/blofro/src/Nemoh/build/`
- **Build system:** CMake + Ninja (`cd build && cmake .. && ninja -j$(nproc)`)
- **Executables (7-step pipeline):**
  1. `/home/blofro/src/Nemoh/build/preProcessor/preProc`
  2. `/home/blofro/src/Nemoh/build/Mesh/hydrosCal`
  3. `/home/blofro/src/Nemoh/build/Solver/solver`
  4. `/home/blofro/src/Nemoh/build/postProcessor/postProc`
  5. `/home/blofro/src/Nemoh/build/QTF/PreProcessor/QTFpreProc`
  6. `/home/blofro/src/Nemoh/build/QTF/Solver/QTFsolver` — **currently has ID_DEBUG=1 (outputs per-term files)**
  7. `/home/blofro/src/Nemoh/build/QTF/PostProcessor/QTFpostProc`
- **Nemoh is single-core Fortran** but links OpenBLAS-pthread at runtime for parallel LU factorization (24 cores available).
- **ID_DEBUG=1 was set in** `/home/blofro/src/Nemoh/QTF/Solver/Main.f90` line 56.

### Pipeline Run Order
```bash
cd /home/blofro/src/pdstrip_test/<case_dir>
/home/blofro/src/Nemoh/build/preProcessor/preProc .
/home/blofro/src/Nemoh/build/Mesh/hydrosCal .
# CRITICAL: hydrosCal OVERWRITES Mechanics/Kh.dat and Inertia.dat — restore correct files:
cp Mechanics/Inertia_correct.dat Mechanics/Inertia.dat
cp Mechanics/Kh_correct.dat Mechanics/Kh.dat
/home/blofro/src/Nemoh/build/Solver/solver .
/home/blofro/src/Nemoh/build/postProcessor/postProc .
/home/blofro/src/Nemoh/build/QTF/PreProcessor/QTFpreProc .
/home/blofro/src/Nemoh/build/QTF/Solver/QTFsolver .
/home/blofro/src/Nemoh/build/QTF/PostProcessor/QTFpostProc .
```

### Modified Nemoh Source Files (COMPILED AND TESTED)

#### 1. `/home/blofro/src/Nemoh/Common/Face.f90`
- `TWLine` type: added `N2D` field (2D contour normal array)
- `Prepare_Waterline`: computes 2D contour normals from edge tangent direction, with outward-orientation check against panel horizontal normal
- **Why**: Nemoh originally used 3D panel normals projected to 2D for waterline integrals. For non-vertical hull surfaces (like KVLCC2's flare), the 3D panel normal has a significant z-component that, when projected to (nx,ny), gives wrong direction. The edge-derived 2D normal is purely horizontal and geometrically correct.

#### 2. `/home/blofro/src/Nemoh/QTF/Solver/MQSolverPreparation.f90`
- `ComputeNDGamma_WLINE`: changed `N(1:2)=Mesh%N(1:2,ipanel)` → `N(1:2)=Wline%N2D(i,1:2)` in all 4 locations (CASE 1 port+stbd, CASE 2 port+stbd)

#### 3. `/home/blofro/src/Nemoh/QTF/Solver/MQSolver.f90`
- `COMPUTATION_QTF_HASBO` waterline loop (~line 474): replaced `Normal_Vect=Mesh%N(1:3,Wline%IndexPanel(Iwline))` → `Normal_Vect(1:2)=Wline%N2D(Iwline,1:2); Normal_Vect(3)=0.`
- Note: HASBO terms are all zero on the diagonal (w1=w2), so this fix only matters for off-diagonal QTF

#### 4. `/home/blofro/src/Nemoh/QTF/PreProcessor/Main.f90`
- Fixed deallocation: added `WLine%N2D` to DEALLOCATE statement

### Comprehensive Search of Waterline Normal Usage

| Location | File | What | Status |
|----------|------|------|--------|
| `ComputeNDGamma_WLINE` | `MQSolverPreparation.f90` | DUOK T3 generalized normal | **FIXED** (uses `Wline%N2D`) |
| `COMPUTATION_QTF_HASBO` waterline loop | `MQSolver.f90` ~L474 | HASBO waterline integral | **FIXED** (uses `Wline%N2D`) |
| `COMPUTATION_QTF_QUADRATIC` waterline loop | `MQSolver.f90` ~L148 | DUOK T3 (eta-zeta)^2 term | Uses pre-computed `genNormalWLine_dGamma` (from N2D) — **OK** |
| `COMPUTATION_QTF_HASFS` waterline loop | `MQSolver.f90` ~L788 | Free-surface waterline | Uses `MeshFS%BdyLineNormal` (dedicated 2D) — **OK** |
| `Prepare_Waterline` centroid offset | `Face.f90` ~L303 | Small 0.01*dl offset | Uses `VFace%N(I,1:2)` — minor, not critical |

### How Nemoh Detects Waterline Segments

In `Prepare_Waterline` (`Face.f90:276-350`):
1. Filter: panel centroid z < -EPS * BodyDiameter (approx -0.33m for KVLCC2)
2. For qualifying panels, scan edges where z(J) + z(J+1) >= threshold
3. Each qualifying edge becomes a waterline segment with midpoint, length, parent panel, and 2D contour normal
4. **No closure check** — segments are unordered, independent
5. **No connectivity** — just a bag of segments

### How Nemoh Computes Potentials at Waterline Points

Potentials are computed in the QTF preprocessor using the full Green's function BEM:
- Influence matrix is sized `(Npanels + NWlineseg) x Npanels`
- For waterline points (rows > Npanels), the field point is `WLine%XM(i,:)` — the midpoint of the waterline edge, nudged 1% outward
- Full summation: `phi(XM_wline) = Sum_j G(XM_wline, panel_j) * sigma_j`
- This is in `/home/blofro/src/Nemoh/QTF/PreProcessor/Main.f90` via `CONSTRUCT_INFLUENCE_MATRIX`

### KVLCC2 Mesh Generation

Script: `/home/blofro/src/pdstrip_test/kvlcc2_nemoh/setup_kvlcc2_nemoh.py`

Converts from pdstrip's `geomet.out` (62 sections x 20 points) to Nemoh mesh format:
- Uses y-symmetry (flag=1, half-hull with y>=0 only = port side from pdstrip input)
- Added keel centerline closure point (y=0) at each section
- Skipped last section (6 points, tiny stern cap)
- Verified outward-pointing normals (had to swap n2<->n4)
- Result: 671 nodes, 600 panels

### KVLCC2 Nemoh Parameters

- **Lpp = 328.186 m**, **B = 58.0 m**, **T = 20.8 m**
- **Mass = 320,437,550 kg**, rho = 1025 kg/m^3, g = 9.81 m/s^2
- **V = 310,218.9 m^3** (displacement volume from hydrosCal, original hull)
- **KG = 18.6m** above keel (zcg = -2.2m in z-up, z=0 at waterline convention)
- **xB = xF = 5.812m** (center of buoyancy, original hull)
- **zB = -9.879m** (center of buoyancy below waterline)
- **GM_T = 5.71m** (SIMMAN 2008 standard)
- **Radii of gyration:** kxx = sqrt(300) = 17.3m, kyy = kzz = sqrt(5849) = 76.5m
- sigma_aw normalization: `sigma_aw = -Re(QTF_N) * LPP / BEAM^2` (for OUT_QTFM_N.dat, already divided by rho*g)

### Corrected Mechanics for Equilibrium Case (no transoms)

**Mesh.cal line 4:** `5.812000 0.000000 -2.200000` (xG, yG, zG)

**Kh.dat (manually corrected — hydrosCal computes wrong C44):**
C33=1.6739e8, C35=C53=-2.4205e8, C44=1.7811e10, C55=3.6544e11

**Inertia.dat:** 6x6 mass matrix format (required by postProc). Includes off-diagonal coupling from COG offset (parallel axis theorem).

**CRITICAL: hydrosCal OVERWRITES Mechanics/Kh.dat and Mechanics/Inertia.dat when run. Must restore corrected files AFTER running hydrosCal.** The corrected files are saved as `Mechanics/Inertia_correct.dat` and `Mechanics/Kh_correct.dat` in each case directory.

### Nemoh QTF Output Format
```
OUT_QTFM_N.dat (normalized by rho*g, has 1-line header):
w1  w2  beta1  beta2  DOF  MOD  PHASE  Re  Im

QTFM_DUOK.dat (dimensional, has 1-line header):
w1  w2  beta1(rad)  beta2(rad)  DOF  Re  Im
```
- Diagonal (w1=w2) gives mean drift force per unit amplitude squared
- `sigma_aw = -Re(QTF_N) * LPP / BEAM^2` (for OUT_QTFM_N.dat)

### Nemoh Processed Mesh File: `mesh/L10.dat`
Format (text, one line per panel):
```
body_id  centroid_x  centroid_y  centroid_z  normal_x  normal_y  normal_z  area
```
Use this to verify panel normals after preProc.

### Nemoh DUOK Term Definitions (from source code analysis)

| Term | Physical Description | Body Motions? | Domain |
|------|---------------------|---------------|--------|
| 1 | Velocity-squared pressure: `(1/2)rho(nabla phi_1 . nabla phi_2)n dS` | **No** | Body panels |
| 2 | Displacement-pressure coupling: `rho(xi . nabla)(d phi/dt)n dS` | **Yes** | Body panels |
| 3 | Waterline relative wave elevation: `-(1/2)rho*g*(eta-zeta)^2 * n dl` | **Yes** (zeta=0 for fixed) | Waterline |
| 4 | Rotation of inertia force: `R(theta) x F_inertia` | **Yes** | Per body |
| 5 | Translation moment correction | **Yes** | Per body |
| 6 | Quadratic stiffness | **Yes** | Per body |

---

## NEMOH QTF DEBUGGING — DETAILED HISTORY (Sessions 12-31)

### Cylinder Validation (Session 12) — CORRECT

- **Dir:** `/home/blofro/src/pdstrip_test/cylinder_debug/`
- R=6m, draft=14m, 634 panels, full body (Isym=0), 100 first-order freqs, 65 QTF freqs
- Drift force F/(rho*g*R*A^2) approx 0.92 at kR=0.7 — **correct**
- Goes to zero at low frequencies — **correct**
- All DUOK terms individually small and well-behaved

### KVLCC2 — The Low-Frequency Blowup Problem (Sessions 13-24)

At long wavelengths (lambda/L > 1.5), the KVLCC2 drift force blows up catastrophically instead of going to zero. Extensive investigation:

- **No-lid, old-lid (59 panels), new-lid (540 panels), fine mesh (2400 panels):** All blow up
- **Full-hull no-symmetry (Isym=0):** Confirmed symmetry is NOT the problem
- **Fixed-body (zero RAOs):** Flat sigma approx 0.55 everywhere — well-behaved but wrong physically
- **Equilibrium case (xG=xB=5.812):** Best floating result, sigma=2.62 at peak but blows up to 36 at lambda/L=3
- **DUOK sub-term decomposition with ID_DEBUG=1:** Massive cancellation between individually huge terms
- **Cylinder comparison:** Cylinder terms scale correctly (tend to 0 at low freq), KVLCC2 terms blow up

### Root Cause Discovery (Session 25): Open Waterline + Non-Vertical Hull

**Key finding**: The KVLCC2 mesh has an **OPEN waterline contour** with `Sum(N_x * dl) = +6.625` (should be zero for a closed contour). The cylinder (which works) has `Sum(N_x * dl) approx -0.059 approx 0`.

**Second issue**: The hull has non-vertical sides at the waterline (flare). The 3D panel normals used for waterline contour integrals have wrong direction when projected to 2D.

### Waterline Normal Fix (Sessions 27-28): 68% Improvement

Fixed Nemoh source to use edge-derived 2D contour normals instead of 3D panel normal projections.

**Results:**
- **Barge regression:** ratio = 1.000000 (perfect, vertical sides unaffected)
- **KVLCC2 equilibrium at lambda/L=3.0:** sigma went from 36.3 to 11.7 (68% reduction, but still blows up)
- Only T3 (waterline elevation term) was affected: dropped from 70.3 to 45.7
- T1, T2, T4 unchanged (they use body panel normals, not waterline contour normals)
- `Sum(N2D_x * dl)` for half-hull: 6.986 (still far from zero — waterline contour is NOT closed)

### KVLCC2 Waterline Contour Properties (Half-Hull, y>=0)

**Without transoms:**
- Stern endpoint: (x=-165.386, y=8.974, z=0)
- Bow endpoint: (x=152.040, y=4.705, z=0)
- **Sum(N2D_x * dl) = 6.986** (should be 0 for closed contour)

The half-hull waterline runs from stern to bow but never reaches y=0 at the centerline. The geomet.out sections at the bow and stern don't terminate at the centerline at the waterline — they terminate at finite y values.

### Mesh Closure Attempts (Sessions 26-31)

#### Attempt 1: Physical Closure Panels at Bow/Stern (Session 26) — Overcorrection

Created `/home/blofro/src/pdstrip_test/kvlcc2_nemoh_closed/` with 617 panels. T3 reduced from +70 to +18 but T2 worsened, total flipped from +36 to -24 at lambda/L=3. Adding physical panels corrupts the BEM solution.

#### Attempt 2: Tip Section Extrapolation (Session 29) — MUCH WORSE

Created `/home/blofro/src/pdstrip_test/kvlcc2_nemoh_extended/` — added degenerate bow tip section at x=155.764 and stern tip at x=-183.175. Sum(n_x*dl) = 0 but **sigma = 67.6 at lambda/L=3** — much worse. The stern tip extends 18m beyond last real section with very thin elongated panels.

#### Attempt 3: Flat Transom Closure (Sessions 30-31) — Still Broken

Created `/home/blofro/src/pdstrip_test/kvlcc2_nemoh_transom/` — vertical flat panels at existing bow (x=152.040) and stern (x=-165.386) x-positions. 682 nodes, 610 panels.

**Initial problem:** Stern waterline closure was never detected by Nemoh because the stern section is very shallow (z_keel=-3.282m) and the waterline panel centroid was at z=-0.228m, above Nemoh's detection threshold of -0.328m.

**Fixed detection:** Modified z-distribution with `min_wl_depth=-1.0` override. Stern waterline panel centroid now at z=-0.396m < -0.328m threshold.

**Result with fixed detection at lambda/L=3.0: sigma = -39.35** (was +11.70 without transoms). T3 swung from +45.73 to -8.09 — massive overcorrection.

### Per-Term Breakdown at lambda/L=3.0 (omega=0.250) — AFTER N2D Fix

| | Barge | KVLCC2 Fixed-body | KVLCC2 Equilibrium | KVLCC2 Extended | KVLCC2 Transom |
|---|-------|-------------------|-------------------|-----------------|----------------|
| T1 | -0.008 | -0.154 | 3.42 | 6.60 | 4.63 |
| T2 | 0.021 | 0.000 | -26.51 | -35.95 | -26.17 |
| T3 | 0.001 | 0.642 | 45.73 | 109.08 | -8.09 |
| T4 | 0.000 | 0.000 | -10.94 | -12.14 | -9.71 |
| **Total** | **0.014** | **0.488** | **11.70** | **67.60** | **-39.35** |

### Barge Validation Test (Session 27) — DEFINITIVE PROOF

Created rectangular barge at `/home/blofro/src/pdstrip_test/barge_nemoh/`:
- Same dimensions as KVLCC2: L=328.186m, B=58.0m, T=20.8m
- **Perfectly vertical sides** — Sum(nx*dl) = 0.000000, max |nz| = 0.000000
- 920 panels, Isym=1, COG at center of buoyancy

**Results — SMOKING GUN:**
- lambda/L=3.0: sigma = **0.014** (KVLCC2: 36.3)
- lambda/L=2.2: sigma = **0.041** (KVLCC2: 15.9)
- lambda/L=1.05: sigma = **1.422** (KVLCC2: 2.6)

Proves the issue is specifically about non-vertical hull surfaces at the waterline.

---

## KVLCC2 NEMOH CASES DIRECTORY (all at `/home/blofro/src/pdstrip_test/`)

| Directory | Mesh | Panels | Isym | Description | Status |
|-----------|------|--------|------|-------------|--------|
| `kvlcc2_nemoh/` | `kvlcc2.dat` | 600 | 1 | Original half-hull | No-lid results in `results_no_lid/`; old-lid (59p) in `results/` |
| `kvlcc2_nemoh_fine/` | `kvlcc2_fine.dat` | 2400 | 1 | Fine mesh | WORSE — irregular freq amplified |
| `kvlcc2_nemoh_lid/` | `kvlcc2_lid.dat` | 1140 | 1 | 600 hull + 540 lid | Best at peak (sigma=2.37), bad low-freq |
| `kvlcc2_nemoh_debug/` | `kvlcc2.dat` | 600 | 1 | Old wrong-mechanics | DUOK per-term output |
| `kvlcc2_nemoh_nosym/` | `kvlcc2_full.dat` | 1200 | 0 | Full hull | Confirmed sym NOT the problem |
| `kvlcc2_nemoh_fixed/` | `kvlcc2.dat` | 600 | 1 | Corrected mechanics (xG=0) | Blowup persists |
| `kvlcc2_nemoh_fixedbody/` | `kvlcc2.dat` | 600 | 1 | Fixed body (zero RAOs) | sigma approx 0.49 at lambda/L=3 |
| `kvlcc2_nemoh_equilibrium/` | `kvlcc2.dat` | 600 | 1 | **Best floating case** | sigma=11.7 at lambda/L=3 (was 36.3 before N2D fix) |
| `kvlcc2_nemoh_closed/` | 617 panels | 617 | 1 | Bow/stern closure mesh | Overcorrected (sigma=-24) |
| `kvlcc2_nemoh_closed_fixedbody/` | 617 panels | 617 | 1 | Closed mesh + zero RAOs | Testing |
| `kvlcc2_nemoh_extended/` | 610 panels | 610 | 1 | Tip section extrapolation | **sigma=67.6 — MUCH WORSE** |
| `kvlcc2_nemoh_transom/` | 610 panels | 610 | 1 | Flat transom closures | **sigma=-39.35 (overcorrection)** |
| **`barge_nemoh/`** | `barge.dat` | **920** | 1 | **Vertical-sided barge** | **sigma=0.014 — CORRECT** |
| `cylinder_debug/` | 634 panels | 634 | 0 | R=6m cylinder | **CORRECT** — validates Nemoh QTF |

**QTF results backup convention**: Old results (before N2D fix) are saved in `results/QTF_old_normals/` in each case directory.

### Mesh Generation Scripts
- **Original**: `/home/blofro/src/pdstrip_test/kvlcc2_nemoh/setup_kvlcc2_nemoh.py`
- **Extended (failed)**: `/home/blofro/src/pdstrip_test/kvlcc2_nemoh_extended/setup_kvlcc2_nemoh_extended.py`
- **Transom (current)**: `/home/blofro/src/pdstrip_test/kvlcc2_nemoh_transom/setup_kvlcc2_nemoh_transom.py`

### Analysis Scripts
- **`plot_comprehensive_drift.py`** — Main comparison plot
- **`check_waterline.py`** — Reproduces Nemoh's waterline detection

---

## KVLCC2 HULL GEOMETRY DETAILS

### Bow Sections (from geomet.out)
```
x=146.660  wl_y=11.502, z=0.000   <- in mesh
x=152.040  wl_y= 4.705, z=0.000   <- LAST section in mesh (bow endpoint)
x=157.420  wl_y= 0.000, z=-3.998  <- fully submerged (y=0 but z=-4)
x=162.800  npts=6, tiny stern cap  <- skipped (only 6 points)
```

### Stern Sections
```
x=-165.386  wl_y=8.974, z=0.000   <- stern-most 20pt section (stern endpoint)
x=-160.000  wl_y=11.691, z=0.000  <- next section forward
```

### Stern Section Point Details (Port Half, After Keel Closure)
```
Stern (x=-165.386), 11 points (j=0 is keel closure):
  j= 0: y=0.000, z=-3.282   <- keel closure
  j= 1: y=0.509, z=-3.282
  ...
  j= 9: y=8.143, z=-0.583
  j=10: y=8.974, z=-0.000   <- waterline
```

### Bow Section Point Details
```
Bow (x=152.040), 11 points:
  j= 0: y=0.000, z=-19.880  <- keel closure (deep bulb!)
  j= 1: y=1.263, z=-19.880
  ...
  j= 9: y=5.328, z=-2.559
  j=10: y=4.705, z=-0.000   <- waterline
```

### Transom Hull Hydrostatics (from hydrosCal)
- **V = 309,233.2 m^3** (vs 310,219 original)
- **xB = xF = 5.338m** (vs 5.812 original)
- Currently using **stale** Kh and Inertia from the no-transom equilibrium case — NEEDS UPDATE

---

## SWAN1 REFERENCE DATA (Head Seas 180 deg)
```
lambda/L:  0.30  0.35  0.40  0.45  0.50  0.55  0.60  0.65  0.70  0.75  0.80  0.85  0.90  0.95  1.00  1.05  1.10  1.15  1.20  1.25  1.30  1.35  1.40  1.50  1.60  1.80  2.00  2.50
sigma_aw:  0.0   0.1   0.3   0.6   1.0   1.3   1.5   1.3   0.9   0.5   0.3   0.5   1.0   1.6   2.1   2.4   2.5   2.3   1.8   1.3   0.8   0.5   0.3   0.1   0.0   0.0   0.0   0.0
```

---

## WHAT TO DO NEXT

### Priority 1: Fix Transom Hull Hydrostatics

The transom hull uses stale hydrostatics from the no-transom hull. This needs correcting:

1. **Set XG = XF = 5.338** in Mesh.cal and recompute the 6x6 mass matrix in Inertia_correct.dat
2. **Set mass = rho * V = 1025 * 309,233 = 316,964,000 kg** (match displacement)
3. **Recompute Kh_correct.dat** from hydrosCal's computed values (or analytically). Note: hydrosCal's C44 was wrong for the original hull — need to verify for transom hull.
4. Re-run the pipeline from hydrosCal onward and compare results.

### Priority 2: Verify Processed Mesh Quality

Check `mesh/L10.dat` for the transom case:
- Verify stern transom panel normals point aft (-x direction)
- Verify bow transom panel normals point forward (+x direction)
- Check for panels with anomalously small areas or wrong-pointing normals

### Priority 3: Compare First-Order Results

Compare excitation forces, added mass, and damping between equilibrium no-transom case and transom case. If first-order forces are dramatically different at low frequency, the transom geometry is corrupting the BEM solution.

### Priority 4: If Still Broken After Hydrostatics Fix

- Run transom case with fixed body (zero RAOs) to isolate T1 and T3
- Compare T3 between transom-fixed-body and no-transom-fixed-body
- Check if waterline integral closure segments themselves have correct potential values

### Other Open Items (PDStrip)

1. **Far-field (Maruo-type) sway drift** — code has Maruo for surge but not sway
2. **Blended Maruo/Pinkster for surge** — automatic heading-dependent blending
3. **Clean up diagnostics** — debug variables behind compile-time flag
4. **Investigate Boese WP integral** — goes massively negative at long waves
5. **Short-wave surge** — Pinkster sigma rises to ~1.1 at lambda/L=0.3 where SWAN1 is near 0
6. **Catamaran validation at other hull separations** — hulld=50, 100

---

## KEY CONSTRAINTS

- **Do NOT read `doc/pdstrip_documentation.pdf`** — too large, will crash session.
- Running pdstrip: `cd kvlcc2 && ../pdstrip < pdstrip.inp > /dev/null 2>&1`
- Build pdstrip: `cd /home/blofro/src/hydro_tools/pdstrip_cat && make`
- **Capytaine Kochin/Maruo drift is a dead end** — do not revisit
- **hydrosCal OVERWRITES Mechanics/Kh.dat and Inertia.dat** — must restore corrected files after running it
- **pdstrip heading convention**: In pdstrip OUTPUT, wave angle 180 = head seas
- **Nemoh QTF solver currently has ID_DEBUG=1** — outputs per-term files
- Nemoh coordinate system for KVLCC2: forward is +x, stern at x approx -165, bow at x approx +152
- The `input_solver.txt` file is needed by the solver but not always created by preProc. Copy from another case if missing.
- **Nemoh build**: `cd /home/blofro/src/Nemoh/build && cmake .. && ninja -j$(nproc)` (uses Ninja, NOT Make)
- **Nemoh Inertia.dat must be 6x6 mass matrix format** for postProc to work
- **Nemoh does NOT enforce outward normals** — relies entirely on mesh vertex ordering

---

## GIT REPO

- **Repo root:** `/home/blofro/src/hydro_tools` (remote: `git@github.com:frodebloch/hydro_tools.git`, branch `main`)
- **PDStrip subdir:** `pdstrip_cat/`

---

## OC3 HYWIND SPAR — PLATFORM MOTION DURING DP ALONGSIDE OPS (Sessions 32-56)

### Motivation

Analyze platform motion of the OC3 Hywind spar floating wind turbine during **DP (Dynamic Positioning) alongside operations** with the turbine **stopped/feathered**. The goal is to synthesize realistic time-domain motion at deck level that a DP vessel must track, including surge, sway, and vertical (heave + pitch coupling).

### Input Data: Nemoh BEM Results

Located at `/home/blofro/src/pdstrip_test/hywind_nemoh_fine/`:
- **RAO:** `Motion/RAO.tec` — 80 frequencies x 13 columns per heading
- **Radiation:** `results/RadiationCoefficients.tec` — 6 zones x 80 freq x 13 columns
- **QTF:** `results/QTF/OUT_QTFM_N.dat` — mean drift QTF

### Reference Sea State

- Wind-sea: Hs=2.5m, Tp=8s, gamma=3.3 (JONSWAP)
- Swell: Hs=1.5m, Tp=19s, gamma=5.0 (JONSWAP)
- Wind: U10=10 m/s (10-min mean)
- Current: Uc=0.5 m/s
- Evaluation height: z=15m above SWL (deck level)
- Turbine state: stopped/feathered

### Key Physical Parameters (OC3 Hywind)

- **Mass:** 1.40179e7 kg, **Draft:** 120m, **CG at z=-78m** below SWL
- **Spar geometry:**
  - Upper: D=6.5m (SWL to -4m)
  - Taper: 6.5→9.4m (-4m to -12m)
  - Lower: D=9.4m (-12m to -120m)
- **Mooring:** 3 catenary lines at 120 deg apart, 902.2m long, fairleads at -70m, anchors at 853.87m radius
- **Pitch:** I55=2.02e10 kg·m², A55=3.99e10 kg·m², K55=4.38e9 N·m/rad, T_n=23.3s (omega_n=0.270 rad/s)
- **B55_crit** = 3.25e10 N·m·s/rad (critical pitch damping)
- **Surge:** T_n≈147s, Q-factor≈9.6, K_surge=4.118e4 N/m
- **Heave:** C33_hydrostatic=333.7 kN/m, T_n≈27-31s
- **I_drag** = 8.51e7 m^5 (Morison drag integral for pitch: ∫ D(z)|r(z)|^3 dz)

### Core Scripts

All in `/home/blofro/src/hydro_tools/pdstrip_cat/`:

| Script | Lines | Purpose |
|--------|-------|---------|
| `platform_motion_plot.py` | ~1090 | Main script: time-domain synthesis + plotting |
| `slow_drift_swell.py` | ~1500 | QTF parsing, JONSWAP, slow-drift, catenary mooring, current drag, current variability spectra |
| `surge_budget.py` | ~300 | RAO parsing (surge/pitch/heave), radiation coefficients, wind/current |
| `combined_motion.py` | ~200 | Bimodal spectrum, extreme value analysis |
| `shutdown_transient.py` | ~200 | Turbine shutdown transient simulation |

### What platform_motion_plot.py Synthesizes (10 Components)

**SURGE:**
1. Mean offset (current + wind + wave drift) → ~4.8m
2. First-order wave response (bimodal spectrum x apparent surge+pitch RAO) → σ≈2.35m
3. Slow-drift from QTF (near surge natural frequency) → σ≈0.13m
4. Wind turbulence response (Froya spectrum) → σ≈0.28m

**SWAY:**
5. VIM (vortex-induced motion) with intermittent lock-in → σ≈0.44m
6. First-order sway from directional spreading → σ≈0.23m
7. Slow random sway drift → σ≈0.04m

**VERTICAL (at deck level z above SWL):**
8. First-order heave (bimodal spectrum x heave RAO) → σ≈0.08m
9. First-order pitch with viscous damping correction → σ≈1.39 deg
10. Combined vertical = heave + z_above_swl x pitch(t) → σ≈0.39m at z=15m

### Key Technical Details

**Geometric frequency spacing:** All spectral synthesis uses geometrically spaced frequencies (N=200) instead of linear spacing to avoid artificial signal repetition. With linear Δω=0.01, signals repeat after T=2π/Δω=628s. Geometric spacing breaks this common-divisor relationship.

**Shared wave phases:** Heave, pitch, and surge share the SAME wave phase realization (phi_wave) to preserve correct inter-DOF phase correlations from the RAOs.

**Self-consistent Morison pitch damping (implemented Session 56):**
The pitch viscous damping is computed from first-principles Morison drag linearization on the spar hull, NOT from an empirical damping ratio. The stochastic linearization (Borgman 1967) gives:
```
B55_visc = (8/(3π)) × 0.5 × ρ × Cd × σ_θ̇ × I_drag
```
where I_drag = ∫ D(z) × |r(z)|^3 dz over the draft, and r(z) is the moment arm from the pitch axis (CG at z=-78m). Since σ_θ̇ depends on the damping, the code iterates to self-consistency (~4 iterations):

```
Converged: B55_visc = 2.945e8, ζ_visc = 0.91%, ζ_rad = 0.03%, ζ_total = 0.94%
Hull velocity at SWL (r=78m): u_rms = 0.62 m/s, u_3σ = 1.86 m/s
```

**Why ζ=0.94% not 5-8%:** Literature values of 5-8% (Skaare et al. 2007, Robertson et al. 2014) are for the OPERATING turbine where aerodynamic damping from the rotor adds 3-5%. For a stopped/feathered turbine, ~1% total is physically correct. Morison body-only drag gives ζ≈0.9%, and we are not missing significant sources (mooring line damping adds maybe 0.1-0.5%, structural ~0.1-0.2%).

**Pitch response is quasi-static dominated:** The swell peak (ω=0.331) and wind-sea peak (ω=0.785) are both away from pitch resonance (ω_n=0.270), so changing damping from 1% to 6% barely affects total pitch σ (1.39 deg vs 1.14 deg, only 22% difference). The resonance peak gets hammered but contains little energy for this sea state.

### Current Output (Self-Consistent ζ=0.94%)

```
SURGE: mean=4.8m, σ=2.37m, range=[-1.5, 11.6]m
  1st-order σ=2.35m, slow-drift σ=0.13m, wind σ=0.28m
SWAY: σ=0.50m, range=[-1.2, 1.2]m
  VIM σ=0.44m, 1st-order σ=0.23m
VERTICAL at z=15m: σ=0.39m, pp=2.13m
  Heave σ=0.08m, Pitch contribution σ=0.36m
VELOCITY:
  Surge: σ=0.79 m/s, max=2.57 m/s (5.0 kn)
  Sway: σ=0.08 m/s, max=0.30 m/s
  Vert: σ=0.13 m/s, max=0.43 m/s
```

### Functions Added to Supporting Modules

**`surge_budget.py`:**
- `parse_rao_heave_pitch(filepath, beta_target=0.0)` — Parses RAO.tec for heave (|Z| col 3) and pitch (|θ| col 5) with phases. Returns omega, heave_amp, heave_phase [rad], pitch_amp [deg/m], pitch_phase [rad].
- `parse_radiation_coefficients(filepath, dof_i, dof_j)` — Generic parser for RadiationCoefficients.tec. Column indexing: col_A = 1 + 2*(dof_j-1), col_B = col_A + 1. Returns omega, A(ω), B(ω) arrays.

**`slow_drift_swell.py`:**
- `solve_catenary_vertical(dx, dz)` — Computes total vertical mooring force from all 3 lines for given surge offset dx and heave dz.
- `mooring_heave_stiffness(dx)` — Computes mooring heave stiffness via central differences. Result: ~-1143 N/m (slightly destabilizing, negligible vs C33=333.7 kN/m).

### Plot Outputs

- `platform_motion.png` — Detailed 10-panel (8x2 GridSpec): surge+vel, sway+vel, vertical+vel, component breakdowns, XY trajectory, summary stats
- `platform_motion_clean.png` — Clean 6-panel: surge+vel, sway+vel, vertical+vel

### Possible Future Work

1. **Relative-velocity Morison formulation** — include wave orbital velocities in the drag linearization for slightly higher damping (~1.1% vs 0.9%)
2. **Parametric sea state sweep** — run for different Hs/Tp combinations to build operability envelopes
3. **DP capability analysis** — map platform velocity demands against DP vessel thruster capacity
4. **Roll motion** — currently not included; probably small for head seas but relevant for beam swell
5. **Second-order heave** — slow-drift heave from QTF (DOF 3); probably small for spar
6. **Multi-body interaction** — wave shielding/amplification between spar and DP vessel

---

## NEMOH GEOMETRIC FREQUENCY SPACING — FreqType=4 (Session 57)

### Motivation

With linear frequency spacing (Δω constant), time-domain signals synthesized from RAO/QTF data repeat after T_repeat = 2π/Δω. For Δω=0.01 rad/s, T=628s (~10 min). Geometric spacing breaks this common-divisor relationship, giving effectively infinite repeat time.

Previously, `platform_motion_plot.py` worked around this by interpolating Nemoh's linearly-spaced output onto a geometric grid post-hoc (200 points via `make_geometric_omega()`). The new FreqType=4 makes Nemoh produce geometrically-spaced output natively, eliminating the interpolation step and ensuring RAOs and QTFs are computed directly at the desired frequencies.

### Implementation: FreqType=4

**Formula:** `w(j) = wmin * r^(j-1)` where `r = (wmax/wmin)^(1/(N-1))` for j=1..N.

#### Fortran Changes (in `/home/blofro/src/Nemoh/`)

| File | What Changed |
|------|-------------|
| `Common/MNemohCal.f90` | Added `INTEGER, PARAMETER :: IdGeomRadFreq=4`. Added geometric branch to `Discretized_Omega_and_Beta`: when `FreqType==4`, builds geometric grid. |
| `QTF/Solver/MQSolverPreparation.f90` | Added `FreqType` argument to `Discretized_omega_wavenumber_for_QTF`. Added geometric grid branch. Replaced old `InterpPotSwitch` logic (exact `dw` equality test) with robust `grids_match` check using 0.1% relative tolerance on actual frequency values. |
| `QTF/Solver/Main.f90` | Updated call to pass `FreqType` argument. |
| `Common/Results.f90` | `FreqType==4` treated same as `==1` (output in rad/s). |
| `postProcessor/MPP_Compute_RAOs.f90` | `FreqType==4` treated same as `==1` in RAO output. |
| `QTF/PostProcessor/MQpostproc.f90` | `FreqType==4` treated same as `==1` in QTF header labels and data conversion. |
| `preProcessor/Main.f90` | Added `IdGeomRadFreq` to `USE MNemohCal` import. |

#### Python Changes (in `/home/blofro/src/hydro_tools/pdstrip_cat/`)

| File | What Changed |
|------|-------------|
| `setup_nemoh.py` | Added `geometric=False` parameter to `write_nemoh_cal()`. `freq_type = 4 if geometric else 1`. Updated frequency line, output freq type, and QTF output freq type. Added `--geometric` CLI flag. |
| `setup_hywind.py` | Same changes as setup_nemoh.py. |

#### No Changes Needed

| File | Why |
|------|-----|
| `export_nemoh.py` | Already reads frequencies as-is from .tec files. QTF matching uses 5% tolerance nearest-neighbor. Despike interpolation uses actual frequency values. All safe with non-uniform spacing. |
| `platform_motion_plot.py` | Already uses `make_geometric_omega()` to create a geometric grid and interpolates linearly-spaced Nemoh data onto it. Still works with geometric input (interpolation becomes near-identity). When using `--geometric` Nemoh output directly, this interpolation step becomes unnecessary but harmless. |

### Validation

Tested with `setup_hywind.py --geometric --n-omega 20 --omega-min 0.05 --omega-max 2.0`:

1. **Build:** Nemoh compiled cleanly with `ninja` (no errors, only pre-existing warnings).
2. **Full pipeline:** All 8 steps completed successfully (preProc → hydrosCal → solver → postProc → QTFpreProc → QTFsolver → QTFpostProc).
3. **Frequency verification:** Output frequencies match `w(j) = 0.05 * 1.21428^(j-1)` to within 3-decimal-place rounding. Consecutive ratios: mean=1.21428, std=2.4e-3 (from display truncation only).
4. **QTF grid matching:** `InterpPotSwitch=0` (direct copy, no interpolation) correctly detected when first-order and QTF grids match.
5. **QTF solver output** shows non-uniform difference frequencies (`w1-w2` varies from 0.011 to 1.939 rad/s) as expected for geometric spacing.

### Usage

```bash
# Hywind spar with geometric frequencies
python3 setup_hywind.py -o hywind_geom --geometric --n-omega 40

# Generic Nemoh case with geometric frequencies
python3 setup_nemoh.py geomet --geometric --n-omega 40 --omega-min 0.05 --omega-max 2.0

# Linear spacing (default, unchanged behavior)
python3 setup_hywind.py -o hywind_linear --n-omega 40
```

---

## FLOATING WIND TURBINE SIMULATOR — DESIGN PLAN (Session 58)

### Purpose

Simulate a floating wind turbine (starting with OC3 Hywind spar) to produce realistic time-domain motions for DP vessels working alongside with walk-to-work systems. The simulator will be a new library in the brucon project (`libs/simulator/floating_platform/`), following the same two-timescale architecture as the existing vessel simulator.

### Use Case

A DP vessel approaches a floating wind turbine for crew transfer or maintenance. The vessel must track the FWT's motion at the gangway connection point. The simulator provides the FWT's position, velocity, and acceleration time series that the DP system (or a coupled simulation) uses as a moving reference. The turbine may be operating or shut down during the approach — the transition (shutdown transient) is an important scenario.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              FloatingPlatformSimulator                   │
│                                                         │
│  LF dynamics (3-DOF: surge, sway, yaw)                 │
│  ├─ Wind force (tower/nacelle/rotor drag + thrust)      │
│  ├─ Current drag                                        │
│  ├─ Mean wave drift (from QTF diagonal)                 │
│  ├─ Slow-drift forces (Newman's approximation)          │
│  ├─ Mooring restoring (quasi-static catenary or         │
│  │   lumped-mass line model — pluggable)                 │
│  └─ LF radiation damping (linear, from BEM)             │
│                                                         │
│  WF response (6-DOF superposition)                      │
│  ├─ Surge, sway, yaw: from RAOs                         │
│  ├─ Heave: from RAOs                                    │
│  └─ Pitch, roll: dynamic 1-DOF equations with:          │
│       ├─ Wave excitation (from excitation force RAO)     │
│       ├─ Radiation damping (frequency-dependent, BEM)    │
│       ├─ Viscous damping (nonlinear Morison drag on      │
│       │   spar hull, integrated in time domain)          │
│       ├─ Aerodynamic damping (from turbine thrust model) │
│       └─ Hydrostatic restoring                           │
│                                                         │
│  Turbine thrust model                                   │
│  ├─ Ct(U) lookup table (e.g., NREL 5MW)                │
│  ├─ Operating: T = 0.5 ρ_air A Ct(U_hub) U_hub²        │
│  ├─ Feathered: T = 0.5 ρ_air A Cd_tower U²  (drag only)│
│  └─ Shutdown transient: ramp Ct → Cd over 10-30s        │
│                                                         │
│  Cross-flow VIM (vortex-induced motion)                  │
│  ├─ Sway oscillation at/near sway natural frequency      │
│  ├─ Intermittent lock-in envelope (duty cycle model)     │
│  ├─ Amplitude A/D ≈ 0.1-0.15 (post-critical Re)         │
│  └─ Lock-in range: Vr = U/(f_n × D) = 4-12              │
│                                                         │
│  Output: position, velocity, acceleration at any point   │
│          on the platform (deck, crane tip, gangway)      │
└─────────────────────────────────────────────────────────┘
```

### Two-Timescale Separation

Same principle as the existing `VesselSimulatorWithWaves`:

**Low-frequency (LF):** Surge, sway, yaw solved with RK4 integration at dt=0.1s. Forces: wind (mean + turbulence), current drag, mean wave drift, slow-drift (Newman), mooring restoring, VIM excitation. The spar has very long natural periods in these DOFs (surge ~147s, sway similar, yaw ~slow) so they are clearly LF.

**Wave-frequency (WF):** Superposed on LF solution.
- Surge, sway, heave, yaw: linear RAO-based superposition (same as existing vessel sim)
- **Pitch and roll: dynamic 1-DOF equations** (not RAO superposition). This is necessary because:
  1. Pitch resonance (~23s) is in the swell band — response is damping-controlled at resonance
  2. Viscous damping (Morison drag on spar) is nonlinear — quadratic in velocity
  3. Aerodynamic damping from the turbine adds 3-5% critical when operating, ~0% when stopped
  4. Shutdown transients require time-domain dynamics, not frequency-domain transfer functions
  5. The existing vessel sim already does dynamic roll this way

**Total motion:** `x_total(t) = x_LF(t) + R(heading) × x_WF(t)` (WF rotated from body to NED frame)

### Cross-Flow VIM (Vortex-Induced Motion)

VIM is a significant sway excitation for deep-draft spars in current. DP operators report it as particularly challenging because the oscillation is persistent, nearly sinusoidal, and unpredictable in its intermittency — the DP vessel must continuously track a ~0.5-1m amplitude sway oscillation at the spar's sway natural period (~147s for Hywind).

**Physics:** Current flowing past the spar sheds vortices at the Strouhal frequency f_s = St × U / D. When f_s is near the spar's sway natural frequency f_n, lock-in occurs and the spar oscillates at f_n with amplitudes A/D ≈ 0.1-0.15 (post-critical Reynolds number regime, Re > 5×10⁵).

**Key parameters for OC3 Hywind:**
- Upper cylinder: D = 6.5m (SWL to -4m)
- Lower cylinder: D = 9.4m (-12m to -120m)
- Strouhal number: St ≈ 0.20 (post-critical)
- Sway natural period: T_n ≈ 147s (f_n ≈ 0.0068 Hz)
- Lock-in range: Vr = U/(f_n × D) = 4-12
  - For upper cylinder: U_lock = 0.18-0.53 m/s
  - For lower cylinder: U_lock = 0.25-0.77 m/s
- Typical A/D ≈ 0.12 → A_lock ≈ 0.8m (using upper D)

**Intermittent lock-in model:** VIM is not continuous — wave-induced velocity fluctuations and turbulence cause the shedding to drift in and out of the lock-in band. This is modeled as:

```
y_vim(t) = A(t) × sin(ω_n × t × (1 + ε(t)) + φ)
```

where:
- `A(t)`: amplitude envelope switching between lock-in (A ≈ A_lock) and unlocked (A ≈ 0.1×A_lock) states with exponentially distributed burst durations (~600s mean) and a configurable duty cycle (~55%)
- `ε(t)`: small frequency wandering (±3%) from a narrow-band process — the shedding frequency is not perfectly constant
- Envelope is smoothed over ~2 natural periods to avoid discontinuities

**Implementation in FloatingPlatformSimulator:**
- VIM sway force is added to the LF sway dynamics (it's at/near the sway natural frequency, clearly LF)
- The VIM model generates a sway excitation force, not a displacement — the LF dynamics then produce the displacement response including mooring restoring and damping
- Alternative: directly superpose VIM displacement (as in `platform_motion_plot.py`) if LF sway dynamics are simplified
- Lock-in state depends on current speed (updated when environment changes)
- When current drops below lock-in range, VIM amplitude decays to residual level

**Why VIM matters for DP alongside operations:**
- At Uc = 0.5 m/s (typical operational current), σ_VIM ≈ 0.44m with peaks up to ±1.2m
- This is comparable to first-order wave sway (σ ≈ 0.23m) and dominates the sway motion budget
- The ~147s period is slow enough that the DP vessel can track it, but the intermittent on/off behavior creates sudden changes in sway velocity demand
- VIM velocity σ ≈ 0.02 m/s (slow), but the abrupt amplitude changes create transient velocity spikes

### Dynamic Pitch Equation

```
(I55 + A55(ω)) × θ̈ + B55_rad(ω) × θ̇ + B55_visc(θ̇) + B55_aero(θ̇) + C55 × θ = M_wave(t)
```

Where:
- `I55 + A55`: platform + added moment of inertia (from Nemoh)
- `B55_rad`: radiation damping (from Nemoh, frequency-dependent — use value at pitch resonance or convolution)
- `B55_visc(θ̇)`: nonlinear Morison drag `∝ θ̇|θ̇|` integrated over spar hull
- `B55_aero(θ̇)`: aerodynamic damping from turbine thrust derivative
- `C55`: hydrostatic + mooring pitch restoring
- `M_wave(t)`: wave excitation moment, synthesized from excitation force RAO and wave spectrum

At dt=0.1s and pitch period ~23s, we get ~230 steps per cycle — ample resolution for RK4.

Roll equation is analogous but typically less important for head-sea operations.

### Aerodynamic Pitch Damping Model

When the turbine is operating, platform pitch velocity θ̇ changes the apparent wind speed at the rotor:

```
U_apparent = U_wind - z_hub × θ̇  (for small angles)
```

The thrust force derivative gives aerodynamic damping:

```
B55_aero = z_hub² × dT/dU × (evaluated at mean wind speed)
```

where `dT/dU = 0.5 × ρ_air × A × d(Ct×U²)/dU` from the Ct(U) lookup table.

For an operating NREL 5MW at rated wind (~11 m/s): B55_aero ≈ 3-5% of critical pitch damping.
For a stopped/feathered turbine: B55_aero ≈ 0 (negligible tower drag contribution).

This is the primary mechanism that changes platform motions between operating and stopped states.

### Turbine Thrust Model

Simple Ct(U) lookup table approach:

```
T(U) = 0.5 × ρ_air × A_rotor × Ct(U) × U²
```

- **Operating (below rated):** Ct increases with U (optimal tip-speed ratio tracking)
- **Operating (above rated):** Ct decreases as blades pitch to limit power; T ≈ constant ≈ rated thrust
- **Feathered/stopped:** Ct ≈ Cd_tower × A_frontal/A_rotor (small — just tower+nacelle drag)
- **Shutdown transient:** Ramp Ct from operating to feathered over user-specified time (10-30s)

The thrust acts at hub height (z_hub ≈ 90m above SWL for Hywind), creating both:
- LF surge force (added to LF dynamics)
- Pitch moment = T × z_hub (added to dynamic pitch equation)
- Aerodynamic pitch damping (as described above)

### Mooring System — Pluggable Interface

Design for two mooring implementations behind a common interface:

```cpp
class MooringModel {
public:
    virtual ~MooringModel() = default;
    virtual void Step(double dt, const PlatformState& state) = 0;
    virtual Eigen::Vector3d Forces() const = 0;   // surge, sway, yaw
    virtual Eigen::Vector3d Restoring() const = 0; // linearized stiffness for LF damping
};
```

**Phase 1: Quasi-static catenary** (`CatenaryMooring`)
- 3 catenary lines at 120° apart
- Static catenary equation solved at each timestep for given fairlead position
- Already implemented in Python (`slow_drift_swell.py`) — port to C++
- Fast, no inner timestep needed

**Phase 2: Dynamic line model** (`DynamicMooring`)
- Wrap the existing brucon `LineModel` / upcoming improved line model
- 3 instances, one per line
- Captures line dynamics (snap loads, damping from drag on lines)
- Requires small inner timestep (~0.001s)

### Hydrodynamic Data Pipeline

```
Nemoh BEM run  →  export_nemoh.py  →  pdstrip.dat  →  FloatingPlatformSimulator
  (RAO.tec,         (already             (20-col         (ResponseFunctions::
   QTF files)        exists)              TSV)            ReadExportedPdStripDatFile)
```

The existing `export_nemoh.py` already converts Nemoh output to the 20-column PdStrip .dat format. The existing `ResponseFunctions` reader in brucon already parses this format. No new data format needed.

**Additional data not in pdstrip.dat** (needed for dynamic pitch/roll):
- Radiation added mass A55(ω), A44(ω) — from Nemoh `RadiationCoefficients.tec`
- Radiation damping B55(ω), B44(ω) — same file
- Wave excitation moment M5(ω), M4(ω) — from Nemoh `DiffractionForce.tec` or `ExcitationForce.tec`

These will be provided in a supplementary config file (or extend export_nemoh.py to produce them).

### Viscous Damping — Nonlinear Morison in Time Domain

At dt=0.1s we can integrate the nonlinear drag directly:

```
M_visc(t) = -∫₀ᴴ 0.5 × ρ × Cd × D(z) × r(z) × |v(z,t)| × v(z,t) dz
```

where `v(z,t) = r(z) × θ̇(t)` is the hull velocity at depth z due to pitch, and `r(z) = z - z_CG` is the moment arm from the pitch axis.

For the Hywind spar with CG at z=-78m:
- `I_drag = ∫ D(z) × |r(z)|³ dz = 8.51e7 m⁵` (pre-computed from spar geometry)
- The integral is discretized over ~10-20 segments matching the spar geometry (upper cylinder, taper, lower cylinder)

### Wind Model

Reuse the existing brucon `WindModel` (mean + Davenport spectrum turbulence) and `WindForceModel`. The wind force on the FWT combines:

1. **Turbine thrust** (from Ct model above) — dominant when operating
2. **Tower + nacelle drag** — always present, uses frontal area and Cd
3. **Blade drag when feathered** — included in the feathered Ct value

For LF dynamics, only the surge component of wind force matters (head-on wind assumed for simplicity, or resolve by relative direction).

### File Structure in brucon

```
libs/simulator/floating_platform/
├── CMakeLists.txt
├── include/brucon/simulator/floating_platform/
│   ├── floating_platform_simulator.h    # Main simulator class
│   ├── platform_model.h                 # Platform physical properties
│   ├── turbine_thrust_model.h           # Ct(U) lookup + aero damping
│   ├── vim_model.h                      # Cross-flow VIM with lock-in
│   ├── mooring_model.h                  # Interface
│   ├── catenary_mooring.h               # Quasi-static catenary
│   └── morison_damping.h               # Spar viscous pitch/roll damping
├── floating_platform_simulator.cpp
├── platform_model.cpp
├── turbine_thrust_model.cpp
├── vim_model.cpp
├── catenary_mooring.cpp
├── morison_damping.cpp
└── test/
    ├── floating_platform_tests.cpp
    └── config/
        └── hywind_spar.dat              # Nemoh-exported RAOs for OC3 Hywind
```

### Class Design

```cpp
class FloatingPlatformSimulator {
public:
    FloatingPlatformSimulator(
        double timestep,
        const PlatformModel& platform,
        std::unique_ptr<MooringModel> mooring,
        const TurbineThrustModel& turbine
    );

    void SetWaveModel(WaveResponse wave_model);
    void SetWindModel(WindModel wind_model);

    // Main integration step
    void Step(double surge_ext_force, double sway_ext_force,
              double yaw_ext_moment);

    // State output — position/velocity at arbitrary point
    Eigen::Vector3d PositionAt(const Eigen::Vector3d& body_point) const;
    Eigen::Vector3d VelocityAt(const Eigen::Vector3d& body_point) const;

    // Component access
    double SurgePosition() const;    // LF + WF
    double SwayPosition() const;
    double HeavePosition() const;    // WF only
    double RollAngle() const;        // Dynamic
    double PitchAngle() const;       // Dynamic
    double YawAngle() const;         // LF + WF

    // Turbine state
    void SetTurbineOperating(bool operating);
    void InitiateShutdown(double ramp_time_s = 20.0);

private:
    // LF state (surge, sway, yaw in NED)
    double lf_north_, lf_east_, lf_heading_;
    double lf_surge_vel_, lf_sway_vel_, lf_yaw_rate_;

    // Dynamic pitch/roll state
    double pitch_, pitch_rate_;
    double roll_, roll_rate_;

    // WF state (6-DOF from RAOs, pitch/roll overridden by dynamic)
    std::array<double, 6> wf_displacement_;
    std::array<double, 6> wf_velocity_;

    // Sub-models
    PlatformModel platform_;
    std::unique_ptr<MooringModel> mooring_;
    TurbineThrustModel turbine_;
    VimModel vim_;
    WaveResponse wave_model_;
    WindModel wind_model_;
    MorisonDamping morison_;

    // Integration
    void IntegrateLF(double dt, double Fx, double Fy, double Mz);
    void IntegratePitch(double dt, double M_wave, double M_aero);
    void IntegrateRoll(double dt, double M_wave);
    void ComputeWaveFrequencyResponse(double time);
};
```

### VimModel — Cross-Flow Vortex-Induced Motion

```cpp
class VimModel {
public:
    VimModel(double sway_natural_freq,    // ω_n [rad/s]
             double upper_diameter,        // D at SWL [m]
             double lower_diameter,        // D of main column [m]
             double strouhal = 0.20);

    // Call each timestep — returns sway force [N]
    double Step(double dt, double current_speed, double sway_velocity);

    // Lock-in state (for logging/diagnostics)
    bool IsLockedIn() const { return locked_in_; }
    double Amplitude() const { return envelope_; }

private:
    // Parameters
    double omega_n_;          // sway natural frequency
    double D_upper_, D_lower_;
    double St_;

    // State
    double phase_;            // VIM oscillation phase
    double envelope_;         // current amplitude envelope
    bool locked_in_;          // current lock-in state
    double time_in_state_;    // time since last lock-in transition
    double next_transition_;  // time of next state switch

    // Lock-in logic
    bool InLockInRange(double Uc) const;
    void UpdateLockInState(double dt);

    // Random state (for intermittent lock-in)
    std::mt19937 rng_;
};
```

The VIM model generates a sway excitation force at each timestep:
1. Check if current speed puts the reduced velocity Vr = U/(f_n×D) in the lock-in range (4-12)
2. Update the intermittent lock-in state (random exponential burst durations, ~600s mean, ~55% duty cycle)
3. Compute smoothed amplitude envelope (ramp between lock-in amplitude A_lock ≈ 0.12×D and residual ≈ 0.01×D)
4. Apply small frequency wandering (±3%) to prevent perfect periodicity
5. Return force = `(m + A22) × ω_n² × A(t) × sin(ω_n×t + ε(t))` — i.e., the force that would produce the desired displacement through the sway dynamics

### PlatformModel — Key Parameters

```cpp
struct PlatformModel {
    // Mass properties
    double mass;              // kg
    double Ixx, Iyy, Izz;    // kg·m² (moments of inertia about CG)
    Eigen::Vector3d cg;       // CG position [x,y,z] relative to SWL origin

    // Added mass at low frequency (for LF dynamics)
    double A11, A22, A66;     // surge, sway, yaw added mass

    // Added mass/damping at pitch/roll resonance (for dynamic pitch/roll)
    double A44, A55;          // roll, pitch added inertia
    double B44_rad, B55_rad;  // radiation damping at resonance frequency

    // Hydrostatic restoring
    double C33;               // heave (N/m)
    double C44;               // roll (N·m/rad)
    double C55;               // pitch (N·m/rad)

    // Spar geometry for Morison damping
    struct Section {
        double z_top, z_bottom;  // depth range (negative = below SWL)
        double diameter;
        double Cd;               // drag coefficient (~1.0-1.2)
    };
    std::vector<Section> hull_sections;

    // Tower/nacelle for wind force
    double hub_height;         // m above SWL
    double rotor_diameter;     // m
    double tower_Cd;           // ~0.6-0.8
    double tower_frontal_area; // m²

    // Reference point for output (e.g., deck level)
    Eigen::Vector3d deck_point;  // body-frame coordinates
};
```

### Implementation Phases

**Phase 1 — Core simulator (MVP):**
- FloatingPlatformSimulator with 3-DOF LF + 6-DOF WF
- Dynamic pitch with Morison damping (nonlinear, time-domain)
- Ct(U) turbine thrust with aerodynamic pitch damping
- Quasi-static catenary mooring
- Reuse existing WaveResponse, WaveSpectrum, WindModel from vessel_model
- Hardcoded OC3 Hywind parameters
- Unit tests comparing against platform_motion_plot.py output

**Phase 2 — Integration:**
- FloatingPlatformSimulatorWrapper (analogous to VesselSimulatorWrapper)
- Configuration from protobuf (platform geometry, mooring, turbine)
- FMU wrapper for co-simulation with DP vessel
- Connection to DpRunfastSimulator as an external reference target

**Phase 3 — Extensions:**
- Dynamic mooring (wrap brucon LineModel, 3 lines)
- Dynamic roll (same approach as pitch)
- Multiple platform types (semi-sub config)
- Operability envelope computation
- Multi-body hydrodynamic interaction (wave shielding between FWT and DP vessel)

### Validation Strategy

1. **Static equilibrium:** Zero waves/wind/current → platform at rest, mooring forces balance
2. **Free decay:** Release from offset → check natural periods match Hywind data (surge ~147s, pitch ~23s, heave ~28s)
3. **RAO comparison:** White noise wave input → compare response spectrum against Nemoh RAOs
4. **Damping comparison:** Compare pitch damping ratio (operating vs stopped) against literature (Skaare et al. 2007: 5-8% operating, ~1% stopped)
5. **Cross-check with Python:** Compare time series statistics (σ_surge, σ_pitch, etc.) against existing platform_motion_plot.py output for same sea state
6. **Shutdown transient:** Verify pitch amplitude increases realistically when turbine shuts down (expect ~2-3× increase in pitch σ)

### Dependencies

- `vessel_model` library: WaveSpectrum, WindModel, WindForceModel, ResponseFunctions, WaveResponse
- `line_model` library: (Phase 2/3 only) for dynamic mooring
- Eigen: matrix/vector math
- Nemoh + export_nemoh.py: for generating hydrodynamic input data

### Key Differences from Existing Vessel Simulator

| Aspect | VesselSimulator (ship) | FloatingPlatformSimulator (spar) |
|--------|----------------------|--------------------------------|
| Hydrodynamic model | Brix strip theory (maneuvering) | BEM coefficients from Nemoh |
| LF DOFs | surge, sway, yaw + quasi-static roll | surge, sway, yaw |
| WF pitch | From RAO (superposed) | Dynamic 1-DOF equation |
| WF heave | From RAO (superposed) | From RAO (superposed) |
| Viscous damping | Section-wise transverse drag (Brix) | Morison drag on spar hull |
| Mooring | Single anchor chain | 3 catenary lines (quasi-static or dynamic) |
| Wind loads | Blendermann coefficients × A_lateral | Turbine thrust Ct(U) + tower drag |
| Aero damping | None | Turbine thrust derivative × z_hub² |
| Current | Speed-through-water in damping | Direct drag force on spar |
| Resistance | ITTC-57 + Holtrop-Mennen | N/A (moored) |

---

## CURRENT VARIABILITY — Slowly Varying Surge from Ocean Current Fluctuations (Session 59)

### Motivation

Slowly varying motions of the Hywind spar at Tampen in the minute-range timescale were consistently underestimated. The existing slow-drift model (QTF-based) gives only σ ≈ 0.09–0.17 m for typical swell, and wind turbulence contributes σ ≈ 0.26 m. The missing piece: **ocean current variability**.

Ocean currents fluctuate due to tidal, inertial, and internal wave processes. Since drag is quadratic in velocity (F ~ U²), even small current fluctuations produce large force variations that drive slowly varying platform motions. At Uc = 0.5 m/s mean current, the mean offset is ~3.3 m. A current variation of σ_Uc = 0.10 m/s produces σ_surge ≈ 1.85 m — **an order of magnitude larger than the QTF slow-drift**.

### Physics

1. **Quadratic drag amplification**: F = CdA_eff × U|U|. For mean Uc with fluctuation u':
   - F(t) ≈ F_mean + dF/dU × u' + ½ d²F/dU² × u'²
   - dF/dU = 2 × CdA_eff × Uc (linear sensitivity)
   - d²F/dU² = 2 × CdA_eff (rectification → mean offset increase)

2. **Hardening catenary mooring**: Stiffness K increases with offset. At 3.3m offset (Uc=0.5 m/s), K_tang = 48.4 kN/m vs linearized 41.2 kN/m. This shifts the surge natural period from 147s to 135s.

3. **Three response regimes** (from period sweep):
   - **Quasi-static** (T_current >> T_n): platform follows current quasi-statically
   - **Near resonance** (T_current ≈ T_n ≈ 135s): dynamic amplification (Q ≈ 10)
   - **Filtered** (T_current << T_n): inertia prevents response

4. **Rectification effect**: Zero-mean current fluctuations produce a net positive mean offset increase Δx = ½ d²F/dU² × σ_Uc² / K_tang (0.12 m for σ_Uc = 0.10 m/s).

### Implementation

Added ~590 lines to `slow_drift_swell.py` (lines 331–920):

| Function | Description |
|----------|-------------|
| `compute_current_drag_force(U)` | Scalar drag force wrapper |
| `compute_linearized_current_sensitivity(Uc)` | dF/dU, d²F/dU² via central differences |
| `current_spectrum_generic(f, σ, T_peak)` | Gaussian-in-log-f parametric spectrum |
| `current_spectrum_internal_wave(f, σ, ...)` | Garrett-Munk f⁻² spectrum (lat=61.2°N) |
| `current_spectrum_bimodal(f, ...)` | Tidal + internal wave two-peak spectrum |
| `compute_current_variability_surge(...)` | **Main function**: spectral chain from S_Uc → S_F → H(ω) → S_x → statistics |
| `current_variability_sweep(...)` | Amplitude sensitivity sweep |
| `current_period_sweep(...)` | Period sensitivity sweep (identifies regime) |

### CLI Usage

```bash
# Amplitude sweep (default)
python3 slow_drift_swell.py --current-variability --uc-mean 0.5

# Single case with period sweep
python3 slow_drift_swell.py --current-variability --uc-mean 0.5 --sigma-uc 0.10 \
    --t-peak-current 1800 --period-sweep

# Internal wave spectrum
python3 slow_drift_swell.py --current-variability --spectrum-type internal_wave
```

### Integration

- **`platform_motion_plot.py`**: Added as 5th surge component (`surge_cv`). Spectrum from `compute_current_variability_surge()` is interpolated onto geometric omega grid and synthesized via `synthesize_from_spectrum()`. Rectification mean shift added to mean offset. Shown in component breakdown plot and statistics panel.

- **`surge_budget.py`**: Added Part 3c (current variability sweep table) and included σ_cv in the RSS combination in Part 5. Budget now shows σ_total with and without current variability.

### Reference Results (Uc = 0.5 m/s, T_peak = 30 min)

| σ_Uc [m/s] | σ_surge [m] | x_sig [m] | pp_exact [m] |
|-------------|-------------|-----------|-------------|
| 0.05 | 0.93 | 1.85 | 2.43 |
| 0.10 | 1.85 | 3.70 | 4.74 |
| 0.15 | 2.77 | 5.55 | 6.84 |
| 0.20 | 3.70 | 7.40 | 8.63 |

Updated RSS surge budget (feathered blades, reference conditions):
- Without current variability: σ_RSS = 1.38 m
- With current variability (σ_Uc = 0.10 m/s): σ_RSS = 2.31 m (+67%)

### Caveats

1. **σ_Uc is uncertain**: Typical North Sea values likely 0.05–0.20 m/s. Site-specific current measurements at Tampen would be needed to constrain this.
2. **Linearized analysis**: Overestimates for large excursions because the catenary mooring hardens. The nonlinear check shows pp_exact/4σ ≈ 0.64 for σ_Uc = 0.10 m/s.
3. **Generic spectrum model**: The Gaussian-in-log-f shape is a placeholder. Real current spectra at Tampen may differ (tidal harmonics, internal wave bandwidth).
4. **No wave-current interaction**: Current fluctuations and wave drift forces are treated as independent. In reality, current modifies the wave drift force (wave-current interaction QTF).

### Future Work

- Integrate into brucon C++ wind turbine simulator (FloatingPlatformSimulator)
- Add site-specific current spectrum from Tampen ADCP data if available
- Couple with wave-current interaction in the QTF computation

---

## CURRENT MODEL — C++ Design for brucon Floating Platform Simulator (Session 60)

### Motivation

The current variability model developed in Session 59 (Python prototype in `slow_drift_swell.py`) needs to be ported to the brucon C++ floating platform simulator designed in Session 58.  Current variability produces σ_surge ≈ 1.85 m at Tampen for σ_Uc = 0.10 m/s — an order of magnitude larger than QTF slow-drift — making it a critical component for realistic DP alongside simulation.

A key realization: the DP vessel and the FWT platform must see the **same current field**.  If each generates its own independent current realization, their surge motions are uncorrelated, and the DP controller's tracking error will be wrong.  This drove a two-layer architecture: a shared environment provider + per-body force consumers.

### Architecture — Shared Environment + Per-Body Force

```
┌─────────────────────────────────────────────────────────────┐
│              CurrentEnvironment  (shared, one instance)      │
│  - Owns mean current, variability spectrum, depth profile   │
│  - Pre-synthesizes time-domain velocity signal              │
│  - Queried by ALL bodies for U(t, z) at any position/time  │
└─────────┬──────────────────────────────────┬────────────────┘
          │                                  │
          ▼                                  ▼
┌─────────────────────┐       ┌──────────────────────────────┐
│ CurrentForceModel   │       │ VesselSimulatorWrapper        │
│ (FWT spar)          │       │ (DP vessel)                   │
│ - Morison drag on   │       │ - Queries env for U(t)        │
│   spar hull sections│       │ - Feeds into existing speed-  │
│ - Relative velocity │       │   through-water damping &     │
│ - Nonlinear F=CdA×  │       │   resistance model            │
│   U_rel×|U_rel|     │       │ - No new drag code needed     │
└─────────────────────┘       └──────────────────────────────┘
```

This follows the same pattern as `WindModel` — a single environment instance shared across simulators.

Three header files implement this:

```
libs/simulator/
├── include/brucon/simulator/
│   └── current_model.h           # CurrentEnvironment (shared) +
│                                 # CurrentForceModel (per-body)
└── floating_platform/
    ├── include/brucon/simulator/floating_platform/
    │   ├── current_spectrum.h    # Spectrum generation (generic, GM, bimodal)
    │   ├── current_drag.h        # Morison drag + HullSection geometry
    │   └── ... (existing headers)
```

**Dependency chain:** `current_model.h` → `current_drag.h` + `current_spectrum.h`.

### File Summary

#### `current_drag.h` — Morison drag integration

- **`HullSection`** struct: `{z_top, z_bottom, diameter, cd}` — same geometry used by `MorisonDamping` for viscous pitch damping.
- **`OC3HywindHull()`** factory: returns the 3-section spar (upper 6.5m + taper + lower 9.4m, Cd=1.05, draft=120m), with the taper discretized into configurable sub-sections.
- **`CurrentDrag`** class:
  - `Force(U)` — uniform current: F = CdA_eff × U|U|
  - `ForceRelative(U_current, U_platform)` — relative-velocity Morison
  - `ForceWithProfile(U_surface, depth_func, U_platform)` — depth-dependent current via a callable depth factor
  - `Sensitivity(U_mean)` → dF/dU = 2×CdA×|U|
  - `SecondDerivative()` → d²F/dU² = 2×CdA (constant for U>0)
  - `RectificationOffset(sigma_uc, k_mooring)` → Δx = 0.5 × d²F/dU² × σ² / K
  - `NumericalSensitivity(U_mean)` → central-difference {dF/dU, d²F/dU²}

Pre-computes `CdA_eff = 0.5 × ρ × Σ(Cd_i × D_i × L_i)` at construction.

#### `current_spectrum.h` — Variability spectrum models

Three spectrum generators (all return S(f) in [m²/s² / Hz], normalized to σ²):

1. **`GenericCurrentSpectrum(f, σ, T_peak, σ_logf)`** — Gaussian-in-log-f parametric shape.  Width parameter σ_logf controls bandwidth (0.2 = narrow/tonal, 0.6 = broad-band).

2. **`InternalWaveCurrentSpectrum(f, σ, lat, f_N)`** — Garrett-Munk f⁻² in the internal wave band [f_inertial, f_N].  Inertial frequency computed from latitude (default 61.2°N for Tampen).

3. **`BimodalCurrentSpectrum(f, σ_tidal, T_tidal, σ_iw, T_iw)`** — Sum of two generic peaks: low-frequency tidal/inertial + high-frequency internal wave.

Utility functions:
- `MakeLogFrequencies()` / `MakeLinearFrequencies()` — grid generation
- `Trapz()` — trapezoidal integration
- `SynthesizeFromSpectrum(f, S, t, rng)` — random-phase time-domain synthesis

#### `current_model.h` — Shared environment + per-body force

Two main classes in a single header:

**`CurrentEnvironment`** (in `brucon::simulator` namespace — shared across vessel/platform):

- Constructor: `CurrentEnvironment(env_settings)`
- `Initialize(duration)` — pre-computes the full current velocity time series
- Query methods (all const after init, safe for concurrent reads):
  - `Speed(t)`, `SpeedAtDepth(t, z)`, `Direction(t)`
  - `Velocity(t)` → `{Ux, Uy}`, `VelocityAtDepth(t, z)` → `{Ux, Uy}`
  - `MeanSpeed()`, `MeanDirection()`
- Diagnostics: `GetSpectrum()` → `{f_hz, S}`

**`CurrentForceModel`** (in `brucon::simulator::floating_platform` namespace — per-body):

- Constructor: `CurrentForceModel(force_settings, hull_sections, shared_ptr<env>)`
- `ComputeForce(t, surge_vel, sway_vel, heading)` → `{Fx, Fy}` [N]
- `ComputeForceDepthIntegrated(t, ...)` → depth-integrated Morison drag
- Linearized: `MeanDragForce()`, `DragSensitivity()`, `DragSecondDerivative()`, `RectificationOffset(K)`

**Settings hierarchy:**

```
CurrentEnvironmentSettings           CurrentForceSettings
├── enabled: bool                    ├── cd_multiplier: double
├── mean_speed: double               ├── relative_velocity: bool
├── mean_direction_deg: double       ├── nonlinear_drag: bool
├── direction_convention: enum       └── include_rectification: bool
├── variable_direction: bool
├── seed: uint64_t
├── lowpass_cutoff_period: double
│
├── CurrentSpectrumSettings
│   ├── type: enum {None, Generic, InternalWave, Bimodal, TimeSeries}
│   ├── sigma_uc: double
│   ├── t_peak: double
│   ├── sigma_log_f: double
│   ├── latitude: double
│   ├── buoyancy_freq_hz: double
│   ├── sigma_tidal, t_peak_tidal
│   ├── sigma_iw, t_peak_iw
│   ├── time_series_file: string
│   ├── n_components: int
│   ├── f_min_hz, f_max_hz: double
│   └── geometric_spacing: bool
│
└── CurrentProfileSettings
    ├── type: enum {Uniform, PowerLaw, Bilinear, Custom}
    ├── power_law_exponent: double
    ├── power_law_reference_depth
    ├── upper/lower_layer_fraction
    ├── transition_depth/thickness
    └── custom_profile: vector<(z,f)>
```

### Integration into Simulators

**Simulation setup (in the orchestrator / scenario runner):**

```cpp
// Create shared current environment
auto current_env = std::make_shared<CurrentEnvironment>(env_settings);
current_env->Initialize(simulation_duration);

// FWT platform — gets a CurrentForceModel that queries the shared env
auto platform_sim = FloatingPlatformSimulator(dt, platform, mooring, turbine);
platform_sim.SetCurrentEnvironment(current_env);  // creates internal CurrentForceModel

// DP vessel — queries the same environment for speed-through-water
auto vessel_wrapper = VesselSimulatorWrapper(vessel_config);
vessel_wrapper.SetCurrentEnvironment(current_env);  // uses U(t) in STW computation
```

**FloatingPlatformSimulator::Step():**

```cpp
void Step(...) {
    // 1. Wave-frequency response
    ComputeWaveFrequencyResponse(time);

    // 2. LF forces
    double Fx_wind = ...;
    double Fx_wave_drift = ...;

    // 3. Current drag on spar (queries shared environment internally)
    auto [Fx_current, Fy_current] = current_force_.ComputeForce(
        time, lf_surge_vel_, lf_sway_vel_, lf_heading_);

    // 4. VIM sway force — uses current speed from shared environment
    double Uc = current_env_->Speed(time);
    double Fy_vim = vim_.Step(dt_, Uc, lf_sway_vel_);

    // 5. Mooring restoring
    auto [Fx_moor, Fy_moor, Mz_moor] = mooring_->Forces(
        lf_north_, lf_east_, lf_heading_);

    // 6. Integrate LF dynamics
    IntegrateLF(dt_,
        Fx_wind + Fx_wave_drift + Fx_current + Fx_moor,
        Fy_current + Fy_vim + Fy_moor,
        Mz_moor);
}
```

**VesselSimulatorWrapper — current integration:**

The DP vessel already has current handling via speed-through-water. The change is replacing its internal constant current with queries to the shared environment:

```cpp
void VesselSimulatorWrapper::Step(double time, ...) {
    // Query shared current environment (same realization as FWT)
    auto [Uc_x, Uc_y] = current_env_->Velocity(time);

    // Existing vessel model: speed through water = SOG - current
    double stw_x = sog_x - Uc_x;
    double stw_y = sog_y - Uc_y;

    // Feed into existing maneuvering model (hull forces, propeller, etc.)
    vessel_model_.SetSpeedThroughWater(stw_x, stw_y);
    // ...rest of existing Step()
}
```

Key integration points:
1. **Correlated motions**: Both simulators see the same current gust at the same time. When the FWT surges from a current increase, the DP vessel also experiences increased drift, so the DP controller sees a smaller relative error than if they were independent.
2. **VIM consistency**: The VIM model on the FWT gets the same current speed that drives the vessel's drift, so VIM lock-in events coincide with periods of higher DP demand.
3. **Mean offset**: mooring equilibrium finder uses `current_force_.MeanDragForce()` to find the static offset and tangent stiffness.
4. **Depth profile matters differently**: The spar extends to -120m (uses full profile), while the DP vessel has ~8m draft (only surface layer matters). The shared profile handles both correctly via `SpeedAtDepth()`.

### Design Decisions

1. **Shared environment, not pass-through**: We considered having the VesselSimulatorWrapper pass its current speed to the platform simulator, but that couples the two simulators. The shared environment is cleaner: each simulator independently queries the same source of truth. It also scales to Phase 3 multi-body scenarios (multiple FWTs, support vessel, etc.).

2. **Pre-computed time series** (not real-time synthesis): The spectral synthesis is O(N_freq × N_time) which could be expensive for long simulations. Pre-computing at `Initialize()` avoids per-timestep cost. Memory: ~80 KB for 3 hours at 0.1s timestep.

3. **Nonlinear drag option**: The Python prototype uses linearized spectral analysis. The C++ time-domain model can compute the full nonlinear F = CdA × (U_mean + u') × |U_mean + u'| at each timestep, which automatically captures the rectification effect without needing a separate correction.

4. **Relative velocity**: Platform surge/sway velocity is subtracted from current velocity in the drag formula. This provides additional hydrodynamic damping (the "current damping" effect) that stabilizes the LF dynamics.

5. **Depth profile**: Uniform is the default (and what the Python prototype uses). Power-law and bilinear profiles are available for more realistic scenarios. The profile multiplies the surface speed at each hull section's midpoint depth.

6. **Shared HullSection geometry**: `current_drag.h` defines `HullSection` which is also used by `MorisonDamping` for viscous pitch damping. This avoids duplicate geometry definitions.

7. **Thread-safe queries**: `CurrentEnvironment` query methods are const after `Initialize()`, so both simulators can call `Speed(t)` / `Velocity(t)` concurrently without locking. This matters for parallel FMU co-simulation.

8. **Settings split**: Environmental settings (what the current IS) are separated from per-body settings (how this body interacts with it). This avoids duplicating spectrum/profile settings when multiple bodies share the same environment.

### Validation Plan

1. **Static drag**: `CurrentDrag::Force(0.5)` should match Python `compute_current_drag(0.5)` = 30.7 kN
2. **Sensitivity**: `CurrentDrag::Sensitivity(0.5)` ≈ 122.7 kN/(m/s), matches `dFdU` from Python
3. **Spectrum shape**: `GenericCurrentSpectrum` output should match `current_spectrum_generic` from Python
4. **Time series statistics**: synthesized σ should match target σ_Uc to within ~5%
5. **Surge response**: for σ_Uc = 0.10 m/s, T_peak = 1800s, the nonlinear time-domain σ_surge should approximate the Python spectral result of 1.85 m
6. **Rectification**: nonlinear simulation should naturally produce the ~0.12 m mean offset increase without the explicit correction term
7. **Correlation check**: In a two-body simulation, verify that the FWT and DP vessel see identical current speed at the same timestep

### Default Settings for OC3 Hywind at Tampen

```cpp
// Environment (shared)
CurrentEnvironmentSettings env;
env.enabled = true;
env.mean_speed = 0.50;            // m/s (moderate North Sea current)
env.mean_direction_deg = 0.0;     // head-on
env.spectrum.type = CurrentSpectrumType::kGeneric;
env.spectrum.sigma_uc = 0.10;     // m/s (conservative estimate)
env.spectrum.t_peak = 1800.0;     // 30 min (internal wave timescale)
env.spectrum.sigma_log_f = 0.4;   // moderately broad
env.spectrum.n_components = 200;
env.spectrum.geometric_spacing = true;
env.profile.type = CurrentProfileType::kUniform;

// Per-body force (FWT spar)
CurrentForceSettings spar_force;
spar_force.cd_multiplier = 1.0;
spar_force.relative_velocity = true;
spar_force.nonlinear_drag = true;
spar_force.include_rectification = true;  // only used if nonlinear_drag=false
```

---
