# PDStrip Drift Force Validation — Session Continuation Notes

**Last updated:** 2026-02-27
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
