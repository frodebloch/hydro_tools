# XFOIL Sheet Cavitation Model — Validation Report

## 1. Model Description

Sheet cavitation modeling has been added to XFOIL 6.99 as a post-hoc
correction to the standard viscous-inviscid panel method.  The model
detects regions where the pressure coefficient falls below the cavity
pressure (Cp < -σ), prescribes a constant-pressure boundary condition
on the cavity surface, and computes the resulting cavity thickness,
closure shape, and pressure drag.

### 1.1 Physical Assumptions

- **Sheet cavitation only**: the cavity is a thin vapor sheet attached
  to the airfoil surface, starting near the leading edge suction peak
  and closing on the body.
- **Constant cavity pressure**: Cp = -σ everywhere on the cavity,
  corresponding to cavity velocity Qcav = Q∞ √(1 + σ).
- **Thin cavity**: cavity thickness h(s) is computed from a
  mass-deficit integral along the cavity, not from a free-streamline
  deformation of the body.
- **No bubble or cloud cavitation**: nucleation, Rayleigh-Plesset
  bubble dynamics, and collapse are not modeled.
- **No supercavitation**: the cavity must close on the body surface.
  Cavities extending past the trailing edge are not supported
  (requires free-streamline / BEM methods).

### 1.2 Closure Models

Two closure models are available, selected via `CPAR → MODL`:

1. **Franc-Michel (FM)** — Short cavity model.  The last 20% of the
   cavity arc length is tapered smoothly to zero thickness using a
   cosine blend.  This produces a closed cavity with zero (or near-zero)
   trailing thickness, appropriate for partial cavities where the
   closure region is a turbulent mixing zone.

2. **Re-entrant jet (RJ)** — Open cavity model.  The cavity retains
   its computed thickness at the closure point.  An additional drag
   term CDcav_j = 2(1+σ)(h_close/c) accounts for the momentum of the
   re-entrant jet that closes the cavity.

### 1.3 Solution Architecture

The cavitation calculation is layered on top of the standard XFOIL
viscous-inviscid (V-I) coupling:

**Inviscid mode**: Single-pass computation.
  1. Solve the panel problem (SPECAL/SPECCL)
  2. Compute Cp distribution
  3. Detect cavitated region (CAVREGION)
  4. Compute cavity thickness (CAVTHICK)
  5. Apply closure model (CAVCLOSE_FM or CAVCLOSE_RJ)
  6. Compute cavity drag (CAVDRAG)

**Viscous mode**: Two-pass iteration within VISCAL.
  - **Pass 1**: Converge cavity extent.  Outer loop iterates:
    restore BL state → override UEDG=Qcav at cavity stations →
    run inner V-I loop (SETBL/BLSOLV/UPDATE) → recompute Cp →
    re-detect cavity extent → apply adaptive damping.
    Convergence criteria include extent matching, near-convergence
    acceptance, max-extent capping (60% of IBLTE), monotonic growth
    detection, and stuck-loop bailout.
  - **Pass 2**: Frozen extent with ramped CAVMASS feedback.
    The cavity mass source MCAV = h·Qcav augments the BL MASS array,
    perturbing the inviscid solution through DIJ coupling.  Five outer
    iterations ramp the relaxation factor from 0.2 to 1.0.

At cavitated BL stations, the standard BL equations are replaced by
CAVSYS: a free shear layer model with Cf=0, outer-layer-only
dissipation, and shape factor guards (HS ≥ 0.15, Rθ ≥ 40).


## 2. Self-Consistency Checks

Eight self-consistency checks were performed.  All pass.

| # | Check                                     | Result |
|---|-------------------------------------------|--------|
| 1 | Symmetry: NACA 0012 at ±7° gives identical cavity | PASS — Ncav=32, CDcav_p=0.058513 at both ±7° |
| 2 | Inception: l/c → 0 as σ → σ_i             | PASS — cavity vanishes near σ_i |
| 3 | Monotonicity: l/c increases as σ decreases | PASS — verified across full σ range |
| 4 | CDcav_p ≥ 0 for all cases                 | PASS |
| 5 | CDcav_p → 0 at low σ (long cavity)        | PASS — pressure difference diminishes |
| 6 | Higher α → longer cavity at same σ         | PASS |
| 7 | Thinner airfoil has higher σ_i at same α   | PASS — NACA 16-006 σ_i=8.46 vs 0012 σ_i=1.54 at α=4° |
| 8 | Cavity growth rate accelerates with decreasing σ | PASS |


## 3. Inviscid Predictions

### 3.1 NACA 0012 at α = 4° (σ_i = 1.5399)

| σ   | Ncav | l/c    | CDcav_p  |
|-----|------|--------|----------|
| 1.2 | 14   | 0.057  | 0.031452 |
| 1.0 | 19   | 0.103  | 0.037616 |
| 0.8 | 26   | 0.191  | 0.039204 |
| 0.6 | 35   | 0.320  | 0.033189 |
| 0.4 | 45   | 0.484  | 0.020671 |
| 0.2 | 59   | 0.703  | 0.008062 |

### 3.2 NACA 0012 at α = 7° (σ_i = 3.4319)

| σ   | Ncav | l/c    | CDcav_p  |
|-----|------|--------|----------|
| 1.5 | 23   | 0.111  | 0.067784 |
| 1.2 | 28   | 0.169  | 0.064452 |
| 1.0 | 32   | 0.230  | 0.058513 |
| 0.8 | 39   | 0.327  | 0.050133 |
| 0.6 | 46   | 0.441  | 0.036673 |
| 0.4 | 56   | 0.593  | 0.022455 |

### 3.3 NACA 4412 at α = 0° (σ = 0.2)

Both-side cavitation: 56 stations on suction side, 12 on pressure side.
Total CDcav_p = 0.006690.

### 3.4 NACA 4412 at α = -2° (σ = 0.2)

Both-side cavitation: 49 stations on suction side, 25 on pressure side.
Total CDcav_p = 0.004482.

### 3.5 6% Elliptic Airfoil at α = 4° (σ = 0.8)

200-point airfoil without PANE: Ncav = 17, CDcav_p = 0.011707.
Identical result in viscous mode (cavity is short enough that the BL
has minimal influence on the inviscid Cp distribution).


## 4. Viscous Predictions

### 4.1 NACA 0012 at α = 4°, Re = 10⁶

| σ   | Ncav | l/c    | CDcav_p  | Convergence   |
|-----|------|--------|----------|---------------|
| 1.2 | 13   | 0.055  | 0.029186 | Natural       |
| 1.0 | 19   | 0.103  | 0.037616 | Natural       |
| 0.8 | 26   | 0.191  | 0.039204 | Natural       |
| 0.6 | 34   | 0.318  | 0.032042 | Natural       |
| 0.4 | 42   | 0.434  | 0.021482 | Mono-growth   |
| 0.2 | 43   | 0.436  | 0.011120 | Mono-growth   |

**Viscous vs. inviscid comparison**: For σ ≥ 0.6, the viscous and
inviscid cavity lengths agree within 1-2 stations.  At lower σ
(longer cavities), the viscous solver reaches a BL convergence
ceiling at approximately 43 stations (~44% chord).  This is a
fundamental limitation of the decoupled V-I approach: the BL equations
become increasingly difficult to converge when a large fraction of
the suction surface has prescribed UEDG.

### 4.2 NACA 0012 at α = 7°, Re = 10⁶

| σ   | Model | Ncav | CDcav_p  | CDcav_j  |
|-----|-------|------|----------|----------|
| 1.0 | FM    | 34   | 0.060754 | —        |
| 1.0 | RJ    | 34   | 0.062155 | 0.155767 |

The viscous cavity is 2 stations longer than inviscid (34 vs 32),
reflecting the BL displacement effect on the suction peak.


## 5. Comparison with Kinnas & Fine BEM

The NACA 16-006 at α = 4° was compared against cavity lengths from
Kinnas & Fine (1993) nonlinear BEM computations.  The BEM solves the
full free-streamline problem with cavity detachment/closure as part
of the solution, while our model uses the non-cavitating Cp distribution
as a fixed base.

### 5.1 NACA 16-006 at α = 4° (with PANE, 160 panels)

| σ   | Ncav | l/c (XFOIL) | l/c (K&F BEM) | XFOIL/BEM |
|-----|------|-------------|---------------|-----------|
| 1.2 |  4   | 0.018       | 0.025         | 0.72      |
| 1.0 |  5   | 0.031       | 0.045         | 0.69      |
| 0.8 |  7   | 0.057       | 0.085         | 0.67      |
| 0.6 | 11   | 0.109       | 0.175         | 0.62      |
| 0.5 | 15   | 0.161       | 0.260         | 0.62      |
| 0.4 | 24   | 0.279       | 0.375         | 0.74      |
| 0.3 | 42   | 0.514       | 0.520         | 0.99      |
| 0.2 | 57   | 0.709       | 0.680         | 1.04      |

### 5.2 Discussion

For short cavities (σ > 0.6, l/c < 15%), XFOIL underpredicts the
cavity length by 30-40%.  This is expected: the non-cavitating Cp
distribution used as the base solution does not account for the
feedback of the cavity on the pressure field.  A true free-streamline
solution (like Kinnas & Fine) redistributes the pressure to maintain
Cp = -σ exactly on the cavity while adjusting the non-cavity pressure
field self-consistently.  Our model applies Cp = -σ as a post-hoc
condition without this global redistribution.

For long cavities (σ < 0.4, l/c > 25%), the agreement improves
dramatically to within ±5%.  At these conditions, the cavity covers
a large fraction of the chord and the base Cp distribution has less
influence on the predicted extent — the dominant factor is simply
how much of the suction surface has Cp below -σ, which is well
captured by the panel solution.

The crossover point (XFOIL/BEM ≈ 1.0) occurs near σ = 0.3 (l/c ≈ 50%).
Below this, XFOIL slightly overpredicts — likely because the Franc-Michel
closure taper forces the cavity shut slightly beyond where the BEM
would close it.

**Implication for practitioners**: The model is most reliable as a
screening tool for moderate-to-long cavities (l/c > 20%) and for
comparative studies (e.g., ranking design variants).  For short-cavity
quantitative predictions, a dedicated BEM or RANS solver should be used.


## 6. Inception Sigma (σ_i)

The inception sigma σ_i = -Cp_min provides the cavitation number at
which cavitation first appears.  Displayed via `CPAR → SHOW`.

### NACA 0012 inception sigma vs. angle of attack

| α (°) | σ_i    | x/c of Cp_min |
|--------|--------|---------------|
| 0      | 0.4130 | 0.122         |
| 2      | 0.7940 | 0.033         |
| 4      | 1.5399 | 0.011         |
| 6      | 2.6954 | 0.005         |
| 8      | 4.2782 | 0.004         |

σ_i grows roughly as α² near the LE, consistent with thin-airfoil
theory suction peak scaling.  The Cp_min location moves toward the LE
with increasing α, as expected.


## 7. Regression Test Suite

All tests use gfortran with `-g -fbounds-check -finit-real=inf
-ffpe-trap=zero -fallow-argument-mismatch -fdefault-real-8` flags,
which catch NaN, Inf, and array bounds violations at runtime.

### 7.1 Inviscid Tests (9 cases)

| Test                                          | Expected                         | Status |
|-----------------------------------------------|----------------------------------|--------|
| NACA 0012 inv σ=1.0 α=7° FM                  | Ncav=32, CDcav_p=0.058513        | PASS   |
| NACA 0012 inv σ=1.0 α=-7° FM (symmetry)      | Ncav=32, CDcav_p=0.058513        | PASS   |
| NACA 0012 inv σ=1.5 α=7° FM                  | Ncav=23, CDcav_p=0.067784        | PASS   |
| NACA 0012 inv σ=0.8 α=7° FM                  | Ncav=39, CDcav_p=0.050133        | PASS   |
| NACA 0012 inv σ=1.0 α=7° RJ                  | Ncav=32, CDcav_p=0.059778, CDcav_j=0.155116 | PASS |
| Elliptic 6% inv σ=0.8 α=4° (200pts, no PANE) | Ncav=17, CDcav_p=0.011707       | PASS   |
| NACA 4412 inv σ=0.2 α=0°                     | 56+12 stations, CDcav_p=0.006690 | PASS   |
| NACA 4412 inv σ=0.2 α=-2°                    | 49+25 stations, CDcav_p=0.004482 | PASS   |
| ASEQ inv σ=1.0 α=3°,5°,7°,9°                | 10/24/32/39 stations             | PASS   |

### 7.2 Viscous Tests (3 cases)

| Test                                          | Expected                         | Status |
|-----------------------------------------------|----------------------------------|--------|
| NACA 0012 visc σ=1.0 α=7° FM Re=10⁶          | Ncav=34, CDcav_p=0.060754        | PASS   |
| NACA 0012 visc σ=1.0 α=7° RJ Re=10⁶          | Ncav=34, CDcav_p=0.062155, CDcav_j=0.155767 | PASS |
| Elliptic 6% visc σ=0.8 α=4° Re=10⁶           | Ncav=17, CDcav_p=0.011707        | PASS   |

### 7.3 Robustness Tests (1 case)

| Test                                          | Expected                         | Status |
|-----------------------------------------------|----------------------------------|--------|
| NACA 16-006 load + PANE + cave 0.4 + alfa 4  | Ncav=24, CDcav_p=0.010614        | PASS   |

The NACA 16-006 test also validates the PANPLT IDEV guard fix (prevents
SIGFPE crash when loading airfoils with poor panel angle distribution
and graphics disabled).


## 8. Known Limitations

### 8.1 Short Cavity Underprediction

For l/c < 15%, cavity length is underpredicted by 30-40% relative to
full BEM solutions.  Root cause: the non-cavitating Cp distribution is
used as the base, so the cavity's feedback on the pressure field is
not captured self-consistently.

### 8.2 Viscous BL Convergence Ceiling

In viscous mode, cavities longer than ~45% chord encounter BL
convergence difficulties.  The BL solver struggles when most of the
suction surface has prescribed UEDG = Qcav, leaving few stations
with natural boundary layer development.  Symptoms: monotonic growth
exit, RMSBL remaining O(1).  The inviscid predictions remain valid
for these conditions.

### 8.3 No Supercavitation

Cavities must close on the body surface.  The 60% chord extent cap
prevents the cavity from reaching the trailing edge.  Supercavitation
requires free-streamline methods that deform the body boundary.

### 8.4 No Bubble Cavitation

The model detects sheet cavitation only (contiguous region of Cp < -σ).
Bubble cavitation (isolated nucleation events driven by local turbulence
and pressure fluctuations) requires different physics (Rayleigh-Plesset
equation, nuclei population modeling).  However, a cavity flag at
mid-chord can serve as a warning indicator for bubble cavitation risk.

### 8.5 Panel Density Sensitivity

Cavity length is quantized to the panel grid.  With the default 160
PANE panels, the minimum resolvable cavity change is ~0.6% chord.
Very short cavities (Ncav < 5) have significant discretization error.
Using a finer panel distribution near the LE (via PPAR) improves
resolution of short cavities.

### 8.6 Mid-Chord Cavitation

The code architecture supports mid-chord cavitation (CAVREGION scans
all BL stations, no LE assumption).  However, accuracy is limited for
the same reason as §8.1: the base Cp is unmodified.  For airfoils with
flat Cp distributions (NACA 6-series, 16-series), mid-chord cavitation
predictions should be treated as qualitative screening only.


## 9. User Interface Reference

All commands are in the OPER menu unless noted.

| Command      | Description                                        |
|--------------|----------------------------------------------------|
| `CAVE [r]`   | Toggle cavitation on/off; optional σ argument       |
| `SIGM r`     | Set cavitation number σ                             |
| `CAVS`       | Display cavity information                          |
| `CDMP f`     | Write cavity thickness to file (x/c, y/c, h/c, Cp) |
| `CPAR`       | Cavitation parameter submenu                        |
| `CPAR → SHOW`| Display σ, closure model, σ_i, σ/σ_i ratio          |
| `CPAR → MODL`| Select closure model (1=FM, 2=RJ)                   |
| `CPAR → SIGM`| Set σ from within CPAR                               |
| `CPAR → FTAP`| Set FM taper fraction (default 0.20)                |


## 10. Source Files

| File          | Role                                               |
|---------------|----------------------------------------------------|
| `src/XCAV.INC`| COMMON block declarations for all cavitation vars  |
| `src/xcav.f`  | All cavitation subroutines (detection, thickness, closure, drag, display, BL system, mass source, inviscid driver, Cp overlay) |
| `src/xoper.f` | OPER command handlers (CAVE, SIGM, CAVS, CDMP, CPAR); VISCAL Phase 2 cavity iteration; CAVDUMP subroutine |
| `src/xbl.f`   | BL modifications (SETBL cavity branch, MRCHDU bypass, UPDATE DUEDG=0) |
| `src/xplots.f`| CPCAV call in CPX; PANPLT IDEV guard                |
| `src/xpol.f`  | Polar storage of σ, CLcav, CDcav                    |
