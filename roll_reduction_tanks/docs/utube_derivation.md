# U-tube anti-roll tank: derivation and validation

**Status**: substantially resolved (2026-04-27 update).

The definitive reference is **Holden, Perez & Fossen (2011), "A Lagrangian
approach to nonlinear modeling of anti-roll tanks", Ocean Engineering 38,
341–359** (PDF in `references/holden_perez_fossen_2011.pdf`). Their model
is derived from Lagrangian mechanics in the same coordinate system used
by brucon and pdstrip (right-handed body frame: x fore, y starboard,
**z down**), and is **experimentally validated** at model scale
(44 tests; nonlinear models 3 orders of magnitude better fit than linear
models at moderate angles).

**Headline result of this investigation**: brucon's cross-inertia
coefficient `a_phi = Q · (z_d + h_0)` is **structurally correct** and
agrees with Holden's eq. 14 once the change of variable from Holden's
fluid-level coordinate `ξ` (units m) to our angle-like coordinate `τ`
(units rad) is applied. Both my v1 (`Q·(h_0 - z_d)`) and v2
(`Q·(z_d - h_0)`) Lagrangian derivations were wrong. The `(w_d/W)`
factor I derived in §A.4.1 is spurious — it does not appear in
Holden's clean derivation, which integrates the fluid kinetic energy
along the centerline using the cross-coupling parameter
`c = w · w_r · x_t · (h_t + r_d)`.

**Implication for code**: brucon's `a_phi` coefficient is correct; the
prototype should adopt it (with appropriate z-down COG-referenced
convention for `z_d`). Brucon's sign error in `RollMoment` (using
`a_τ` instead of `a_phi`) is real but our Python "fix" introduced a
*different* error (correct coefficient but wrong sign), which mostly
cancelled the original error numerically. Tank EOM and hull moment
should be:

```
Tank:   a_τ τ̈ + b_τ τ̇ + c_τ τ  =  +a_phi · φ̈  +  c_phi · φ
Hull:   M_tank_on_hull            =  +a_phi · τ̈  +  c_phi · τ
```

Both signs **positive** (per Holden eq. 41 mapped to our τ; cross-checked
against the explicit Lloyd-textbook sign-error footnote on p.14 of the
paper).

**The original three "bugs in brucon" claim is partially retracted**:

| # | Original claim                  | Status after Holden review                                    |
|---|---------------------------------|--------------------------------------------------------------|
| 1 | `a_phi = Q·(z_d + h_0)` is wrong | **Retracted** — brucon's formula is correct per Holden eq. 14 |
| 2 | Sign of `c_phi·φ` on tank RHS is wrong (brucon has `-`) | **Open** — needs re-verification per Holden eq. 41             |
| 3 | RollMoment uses `a_τ` instead of `a_phi` | **Confirmed** — brucon's coefficient is wrong; should be `a_phi` |

---

# Part I: Holden, Perez, Fossen (2011) reference model

## I.0 Why this is the definitive source

- **Same coordinate convention as brucon and pdstrip**: x fore, y starboard,
  z down, right-handed (Holden §3.3).
- **Origin at center of floatation Cf** (the roll axis), placed at the
  longitudinal center of the tank — so `r_d` (= our `z_d`) is measured
  *from the roll axis*, not the COG. **For brucon compatibility, this
  means our `z_d` should be interpreted as duct-position-below-roll-axis,
  i.e., need to verify what brucon's `utube_datum_to_cog` actually
  references.** (See §I.7.)
- **Fully Lagrangian derivation** with cross-checked energy properties
  (skew-symmetry of `Ṁ - 2C`, positive-definiteness of mass matrix).
- **44 model-scale experiments** validate the linear and nonlinear
  forms. Nonlinear models track experimental data with mean square error
  three orders of magnitude lower than linear models at moderate roll
  amplitudes.
- Explicitly identifies a sign error in Lloyd (1989, 1998) on the
  tank-induced roll moment (footnote 2, p.14): Lloyd's hull EOM has
  `c_44 x_4 − [a_4τ τ̈ + c_4τ τ]` but should have `+[...]`.
- Explicitly notes that the linearised Lagrangian model `Ll` is
  **functionally identical to Lloyd (1989, 1998) and Faltinsen & Timokha
  (2009) eq. 3.59 p.92**, modulo the `It(0)` term and Lloyd's sign
  error. So Holden gives us a clean line to the textbook canon.

## I.1 Holden's parameters mapped to our notation

Holden's notation (their §2.3, §2.4) and ours:

| Holden       | Description                                            | Our name          |
|--------------|--------------------------------------------------------|-------------------|
| `w`          | Sum of duct width and one reservoir width              | `W = w_d + b_r`   |
| `w_r`        | Reservoir width                                        | `b_r`             |
| `x_t`        | Longitudinal depth of the tank (along ship)            | `t`               |
| `h_d`        | Duct height                                            | `h_d`             |
| `r_d`        | Vertical position of duct center, from roll axis (z-down) | `z_d`           |
| `h_t`        | Datum level (undisturbed fluid height in reservoirs)   | `h_0`             |
| `ρ_t`        | Tank fluid density                                     | `ρ`               |
| `q_1 = φ`    | Generalized coordinate, hull roll angle                | `φ`               |
| `q_2 = ξ`    | Generalized coordinate, fluid level in port reservoir relative to datum (units **meters**) | (different — see below) |
| `υ`          | Average fluid velocity at tank midpoint (positive to port) | (related to τ̇)   |

Our `τ` is dimensionless (radians), defined so that `τ > 0` means the
**starboard** reservoir is *higher* by `(W/2)·τ`. Holden's `ξ` is the
**port** reservoir level above datum (units meters). The conversion is:

```
ξ_holden  =  -(W/2) · τ_us
q̇_2       =  -(W/2) · τ̇_us
q̈_2       =  -(W/2) · τ̈_us
```

## I.2 Holden's key coefficients (linear case)

From Holden eq. 14 (cross-coupling parameter):
```
c  =  w · w_r · x_t · (h_t + r_d)         [units m^4]
```

From Holden eq. 11 (potential-energy linear coefficient):
```
α_3  =  g · ρ_t · x_t · w_r · w           [units kg·m/s²]
α_4  =  g · ρ_t · x_t · w_r               [units kg/s²]
```

From Holden eq. 15 (effective volume):
```
V̄_t  =  w_r · x_t · (2·h_t + w·w_r/h_d)  [units m³]
```

From Holden eq. 41 (linearised tank EOM in lab convention):
```
ρ_t · V̄_t · q̈_2  +  d_2,l · q̇_2  +  2·α_4 · q_2  +  α_3 · φ  +  ρ_t · c · φ̈  =  0
```

## I.3 Mapping to our (τ, φ) form

Substituting `q_2 = -(W/2)·τ` and multiplying through by `-2/W`:
```
ρ_t · V̄_t · τ̈  +  d_2,l · τ̇  +  2·α_4 · τ  =  (2/W)·α_3 · φ  +  (2/W)·ρ_t·c · φ̈
```

Identify terms with `a_τ τ̈ + b_τ τ̇ + c_τ τ = +a_phi·φ̈ + c_phi·φ` (note
the sign on `a_phi·φ̈`):

```
a_τ    =  ρ_t · V̄_t  =  ρ · b_r · t · (2·h_0  +  W·b_r/h_d)
       =  Q · b_r · (W/(2·h_d) + h_0/b_r) · 2/Q · Q  ... let me simplify
       =  2·Q · (h_0/W² · 2/b_r·b_r + 1/(2·h_d) · b_r·W/W²) ... easier numerically
```

Going more carefully: `Q = ρ·b_r·W²·t/2`, so `2·Q/W² = ρ·b_r·t`. Thus:
```
a_τ  =  ρ·b_r·t · (2·h_0 + W·b_r/h_d)
     =  (2·Q/W²) · (2·h_0 + W·b_r/h_d)
     =  (4·Q/W²) · (h_0 + W·b_r/(2·h_d))
```

Compare to our existing code's `a_τ = Q · b_r · (W/(2·h_d) + h_0/b_r)
= Q · (W·b_r/(2·h_d) + h_0)`.

Holden's: `(4·Q/W²) · (h_0 + W·b_r/(2·h_d))`. Ratio of Holden / our code
= `4/W²`. **Not the same!** This `4/W²` factor is the same conversion
factor we just applied — meaning **our `a_τ` is in units consistent
with τ in radians, while Holden's is in units consistent with `ξ` in
meters**. The physics is the same; the mass coefficient must be
re-scaled to match the coordinate.

**Check via natural frequency** (which is invariant under the
`τ ↔ ξ` rescaling):
- Holden: `ω_t² = 2·α_4 / (ρ_t·V̄_t)` (from eq. 41, ignoring damping and coupling).
- Substituting: `ω_t² = 2·g·ρ_t·x_t·w_r / (ρ_t·w_r·x_t·(2·h_t + w·w_r/h_d))
  = 2·g / (2·h_t + w·w_r/h_d) = g / (h_t + w·w_r/(2·h_d))`.
- In our notation: `ω_t² = g / (h_0 + W·b_r/(2·h_d))`.

Our existing code: `ω_t² = c_τ/a_τ = Q·g / [Q·(W·b_r/(2·h_d) + h_0)]
= g / (h_0 + W·b_r/(2·h_d))`. ✓ **Exact match.** Both Holden and our
code give the same natural frequency (Bertram 4.123 also matches).

So our `a_τ = Q·(W·b_r/(2·h_d) + h_0)` and `c_τ = Q·g` are consistent
with Holden's framework after the τ↔ξ change of variable. Our code is
right on these.

**Now the cross-coefficients:**

```
a_phi  =  (2/W) · ρ_t · c
       =  (2/W) · ρ_t · w · w_r · x_t · (h_t + r_d)
       =  (2/W) · ρ · W · b_r · t · (h_0 + z_d)
       =  2 · ρ · b_r · t · (h_0 + z_d)
       =  (4·Q/W²) · (h_0 + z_d)
```

In radian-τ units, the conversion factor `(2/W)·(W/2) = 1` would land
us at `Q·(h_0 + z_d)` — let me redo that step. Cross-term in Lagrangian
is `ρ_t·c·q̇_1·q̇_2 = ρ_t·c·φ̇·ξ̇`. Substituting `ξ̇ = -(W/2)τ̇`:
```
T_cross  =  ρ_t · c · φ̇ · (-(W/2)·τ̇)
         =  -(W/2) · ρ_t · c · φ̇·τ̇
         =  -(W/2) · ρ_t · w·w_r·x_t·(h_t + r_d) · φ̇·τ̇
         =  -(W/2) · ρ · W·b_r·t·(h_0 + z_d) · φ̇·τ̇
         =  -(W²/2) · ρ·b_r·t · (h_0 + z_d) · φ̇·τ̇
         =  -Q · (h_0 + z_d) · φ̇·τ̇
```

The cross-coefficient in `T_cross = α·φ̇·τ̇` is `α = -Q·(h_0 + z_d)`,
giving:
```
∂²T/∂φ̇∂τ̇  =  -Q · (h_0 + z_d)
```

Lagrange's equation contributes `+α·τ̈` to the φ-equation LHS and
`+α·φ̈` to the τ-equation LHS. To match our convention `a_τ τ̈ = ... +a_phi φ̈ + ...`,
the `+α·φ̈` on the LHS of τ-eqn becomes `-α·φ̈` on RHS, which equals
`+Q·(h_0 + z_d)·φ̈` on RHS. So:

> **`a_phi = +Q · (h_0 + z_d) = +Q · (z_d + h_0)`** ✓ **identical to brucon**

Same exercise for the static cross-coupling:
```
α_3 · q_2 · cos(q_1) ≈ α_3 · q_2 · (1 - q_1²/2)  in linearisation just α_3 · q_2.
```
Substituting `q_2 = -(W/2)τ`:
```
α_3 · q_2  =  -(W/2)·α_3·τ  =  -(W/2)·g·ρ_t·x_t·w_r·w·τ
            =  -(W/2)·g·ρ·b_r·t·W·τ
            =  -(W²/2)·g·ρ·b_r·t·τ
            =  -Q·g·τ
            =  -c_phi·τ
```

So Holden's PE term `α_3·q_2·sin(q_1) ≈ α_3·q_2·q_1 = -c_phi·τ·φ` in
our notation.

In Holden eq. 11 the sign of the `α_3·q_2·sin(q_1)` term is **`+`**
(see eq. 11 RHS). So `U ⊃ +α_3·q_2·sin(q_1) = -c_phi·τ·φ` in our terms.

Lagrange: `-∂U/∂q_1 = -α_3·q_2·cos(q_1) → -α_3·q_2` in linear case →
`+(W/2)·α_3·τ = +c_phi·τ` contributes to **τ_L on the hull side via
the φ-equation** (eq. 36/37 of Holden). The hull-side moment from the
fluid offset is `+c_phi·τ` ✓. But for the **tank EOM** (τ-equation),
we need `-∂U/∂q_2 = -α_3·sin(q_1) ≈ -α_3·q_1 = -α_3·φ`. After our
`q_2 = -(W/2)τ` rescaling on the equation for τ, the sign flips. End
result: tank EOM has `+c_phi·φ` on the RHS ✓.

(See §I.5 below for the full tank-EOM derivation in our (τ, φ) form.)

## I.4 Resulting EOM in our (τ, φ) convention

```
Tank:    a_τ · τ̈  +  b_τ · τ̇  +  c_τ · τ   =   +a_phi · φ̈  +  c_phi · φ
Hull:    (I+a_44) · φ̈  +  b_44 · φ̇  +  c_44 · φ   =   M_wave  +  M_tank_on_hull
                                                       
where M_tank_on_hull  =  +a_phi · τ̈  +  c_phi · τ
                         + (higher-order I_t terms, neglected in linear analysis)
```

Coefficients (z-down body frame, COG-referenced for `z_d`, **positive
`z_d` means duct below COG / nearer the keel**):

```
Q       =  ρ · b_r · W² · t / 2

a_τ     =  Q · (W·b_r/(2·h_d) + h_0)            [our existing formula, correct]
b_τ     =  μ · a_τ                                [µ = wall friction; matches our code]
c_τ     =  Q · g                                  [our existing formula, correct]

a_phi   =  Q · (z_d + h_0)                       [BRUCON's formula, validated by Holden]
c_phi   =  Q · g                                  [our existing formula, correct]
```

## I.5 Bug status, revised

| # | Brucon code                                | Our previous "fix"            | Holden-validated correct                  |
|---|--------------------------------------------|-------------------------------|--------------------------------------------|
| 1 | `a_phi = Q·(z_d + h_0)` ✓                  | ✗ Changed to `Q·(z_d - h_0)`  | `Q·(z_d + h_0)` (revert to brucon)         |
| 2 | `c_phi·φ` on tank RHS with `-` sign        | Changed to `+` sign           | **Need to re-verify against Holden eq. 41** |
| 3 | `M_roll = +a_τ·τ̈` (wrong coefficient)     | `M_roll = -a_phi·τ̈` (right coef, wrong sign) | `M_roll = +a_phi·τ̈ + c_phi·τ`     |

So for **bug #1**: brucon was right, our fix was wrong. **Revert.**

For **bug #3**: brucon used the wrong coefficient (`a_τ` instead of
`a_phi`). Our fix used the right coefficient but flipped the sign.
**Both wrong; correct is `+a_phi·τ̈`.**

For **bug #2**: needs explicit comparison against Holden's eq. 41 sign
in our convention. From §I.3 above, working through carefully, the tank
EOM in our (τ, φ) form has `+c_phi·φ` on the RHS — matching our current
code's sign — and *opposite* to brucon's. So **bug #2 stands; brucon is
wrong here, our fix was correct**.

## I.6 Re-evaluation of placement physics

With corrected `a_phi = Q·(z_d + h_0)`:
- `a_phi = 0` when `z_d = -h_0` (duct *above* COG by `h_0` = 2.5 m for
  CSOV; this is the line where the dynamic absorber action vanishes).
- `a_phi > 0` for `z_d > -h_0` (duct anywhere from `h_0` above COG down
  to the keel).
- `a_phi < 0` for `z_d < -h_0` (duct more than `h_0` above the COG, i.e.
  in the upper deckhouse / superstructure).

The dynamic absorber moment `+a_phi·τ̈` on the hull is positive for
`a_phi > 0` and τ̈ > 0. Whether this **damps** φ̇ depends on the phase
of τ̈ relative to φ̇, which at perfect tuning is set by the tank's own
dynamics.

PNA's qualitative claim "fluid mass above the roll center → damping;
below → wrong direction" needs to be re-evaluated in light of the
corrected `a_phi`. Specifically:
- The brucon Winden test uses `z_d = -13.48 m` (duct very high above
  COG), giving `a_phi = -10.98·Q` (negative).
- Our CSOV operating point uses `z_d = -2.5 m` (duct at WL with
  COG 2.5 m above), giving `a_phi = 0` exactly! → No dynamic absorber
  action at all in our headline simulation, just the static `c_phi·τ`
  term. **This may explain low reduction numbers.**
- Placing the duct lower (`z_d = +4 m`, near keel) gives `a_phi = +6.5·Q`.
- Placing the duct higher (`z_d = -10 m`, deckhouse top) gives `a_phi = -7.5·Q`.

Both extremes give large |a_phi|; only placing near `z_d = -h_0` gives
zero. **This is a cleaner version of the same conclusion my §A.9 reached:
the absorber action depends on |a_phi|², so both very high and very low
placements work, with a dead zone in between.** The dead zone is at
`z_d = -h_0` (duct `h_0` meters above COG), not at the COG itself.

For PNA's qualitative claim to hold, the *sign* of the dynamic moment
relative to φ̇ must depend on whether `a_phi > 0` or `< 0`. From §A.9
this is not obvious because the absorber action involves `(c_phi + a_phi·ω²)²`,
which is sign-blind. A more careful frequency-domain analysis with
non-zero hull damping might reveal the sign dependence PNA describes.

## I.7 Outstanding convention question: Cf vs COG for r_d

Holden references `r_d` from the **center of floatation** (the assumed
roll axis). Brucon's variable name is `utube_datum_to_cog`, suggesting
COG-reference. **These differ by `KG`** (the height of COG above the
center of floatation, roughly = above the WL for a typical hull).

- If brucon's `utube_datum_to_cog` is **literally** the duct-to-COG
  distance (z-down), then to use Holden's formula we should set
  `r_d_holden = utube_datum_to_cog_brucon + KG` (since Cf is `KG`
  above COG, and z-down means a higher Cf has *smaller* z; so a duct
  at z-position `z_d_COG` has position `z_d_COG + KG` relative to Cf
  in z-down coordinates).

  Wait, that's not right either. Let me think again. In z-down COG-referenced
  coordinates, COG is at z=0, Cf (≈ WL) is at z = -KG (above COG). A duct
  at z-position `z_brucon` (COG-referenced) is at z-position `z_brucon - (-KG) = z_brucon + KG`
  in Cf-referenced coordinates.

  Wait no. If COG is at COG-coord 0 and Cf-coord +KG (since Cf is above COG by KG, and z-down),
  the offset between the two systems is constant. A point at COG-coord
  `z_COG` is at Cf-coord `z_Cf = z_COG - (-KG) = z_COG + KG`. So
  `z_d_holden = z_d_brucon + KG`.

  For Winden test (`z_d_brucon = -13.48`): `z_d_holden = -13.48 + KG ≈ -13.48 + 2.5 = -10.98`.

  Brucon's formula `Q·(z_d + h_0) = Q·(-13.48 + 2.5) = -10.98·Q`.
  Holden's formula in Cf-ref: `Q·(z_d_holden + h_0) = Q·(-10.98 + 2.5) = -8.48·Q`.

  **They differ.** So brucon's formula and Holden's formula give different
  values for the same physical tank, *if* the conventions differ as
  hypothesized.

But the brucon test value for `a_phi = Q·(z_d + h_0) = 3736125·(-10.98) = -41,022,652.5`
matches the test expectation **exactly**. So brucon's interpretation is
internally consistent: brucon evaluates `Q·(z_d + h_0)` with `z_d` as
their state variable.

The question is whether brucon's `z_d` actually means
duct-to-COG or duct-to-Cf, regardless of the variable name. **If brucon
intended COG-referenced but the formula physically corresponds to
Cf-referenced, brucon has a hidden bug** of magnitude `KG` in the cross-coupling.

→ **This is open.** Need to (a) check whether brucon's downstream code
adjusts for `KG` somewhere, and (b) decide the convention for our
prototype. The prototype README claims brucon uses COG-referenced; if
so, the Cf vs COG question becomes a real physical bug in brucon
(in addition to bug #3) for any `KG ≠ 0`. For our CSOV with KG = 2.5 m,
this is a 0.27 m offset in the "dead zone" location (small but
not negligible).

## I.8 Recommended next actions

1. **Update `tanks/utube_open.py`** to revert `a_phi` to `Q·(z_d + h_0)`
   (matches brucon and Holden), and **flip the sign of the `a_phi·τ̈`
   term in `forces()`** to `+a_phi·τ̈` (matches Holden eq. 36/37).

2. **Switch `utube_datum_to_cog` to z-down convention** (positive = duct
   below COG / near keel). Update all examples accordingly.

3. **Add tests** that lock in agreement with Holden's coefficient
   formulas, and against Bertram 4.122 / 4.123.

4. **Add a `roll_axis_offset` parameter** (defaults to 0 for COG-referenced
   brucon-style usage; can be set to `-KG` for Holden-style Cf-referenced
   usage) to make the `Cf vs COG` question explicit.

5. **Re-run `examples/investigate_reduction.py`** with corrected `a_phi`
   to see the new placement physics.

6. **Update the README** to reflect Holden as the primary reference,
   retract the v1 and v2 placement claims, and update the bug table to
   match §I.5.

---

# Part II: PNA path-integral attempt (alternative validation, partial)

The following sections attempt a validation of `a_phi` via PNA's
ξ_t = S″/S′ framework. They predate the discovery of the Holden paper
and reach a less complete answer. Kept for reasoning trail; superseded
by Part I where they conflict.

## II.1 PNA's framework (p.129)

---

# Part I: PNA path-integral derivation

## I.1 PNA's framework (p.129)

PNA characterises a passive stabilizer by five parameters. Two are
purely static (μ for stiffness reduction, η_s for capacity), one is the
tuning ratio (σ_t = ω_t/ω_n4), one is the damping (ζ_t, typical 0.2–0.4),
and one is the **height parameter ξ_t**, which determines whether the
tank's *dynamic* (acceleration-driven) action damps or anti-damps the
hull's roll motion.

For a U-tube tank, PNA defines:

```
        S″
ξ_t  =  ──                                       (PNA p.129)
        S′

        L
S″  =  ∫ q(v) / R  dv                           (path integral, signed)
        0

        L
S′  =  ∫ A_0 / A(v)  dv                         (path integral, positive)
        0

ω_t  =  √(2g/S′)                                 (matches Bertram 4.123 ✓)
```

where:
- **`v`** is arc length along the **flow path** of the U-tube (start at
  one reservoir top, descend through the duct, ascend the other reservoir).
- **`L`** is the total flow-path length: `L = 2·h_0 + L_duct`
  (two reservoir columns plus the duct centreline length).
- **`A(v)`** is the local cross-section area at flow-path point v.
- **`A_0`** is the reservoir cross-section (used as the reference: at
  reservoir points `A(v) = A_0`, in the duct `A(v) = A_d`).
- **`q(v)`** is the height of the **tangent line to the flow path at
  point v above the roll center**, with the explicit sign rule from
  PNA: positive if the tangent is above the roll center, negative if
  below.
- **`R`** is a reference length (PNA Fig. 95b, not shown). From
  dimensional analysis (`ξ_t` is dimensionless and `S′` has units of
  length), `R` must have units of length. The natural choice is the
  **moment-arm length** = half the centreline distance between
  reservoirs, `R = W/2`. This is confirmed below by matching the
  resulting moment expression to Bertram 4.122.

PNA's qualitative claim (p.130): *"The roll moments that arise due to
the acceleration ... contribution placed above the roll center
contributes to the damping, and below works in the wrong direction."*
This translates directly to: **`ξ_t > 0` adds damping, `ξ_t < 0`
removes it**.

## I.2 The roll center

PNA's `q(v)` is measured relative to the **roll center**, which is the
instantaneous axis of rotation of the rolling hull. For a free-floating
ship in waves the roll center is **typically near the waterline** (a bit
above or below depending on hull form, draft, GM, frequency).

For the prototype and brucon port we adopt:

> **Convention**: roll center at the waterline.

In COG-referenced z-down coordinates with COG at z=0:
```
z_RC  =  -KG       (KG = height of COG above waterline, positive up)
```

For a CSOV with KG = 2.5 m (above WL) and draft T = 6.5 m:
```
z_RC  =  -2.5 m            (waterline is 2.5 m above COG = z = -2.5 in z-down)
keel  =  +(T - KG) = +4 m
```

The roll center being a model parameter is a small extension of brucon's
existing model, which integrates roll about the COG. We track this
explicitly through the derivation so the prototype can use either
reference.

## I.3 Path integral S′ for a standard U-tube

Geometry (matching `OpenUtubeConfig`):
- Reservoir cross-section `A_0 = b_r · t` (port and starboard, same)
- Duct cross-section `A_d = h_d · t`
- Reservoir fluid column height `h_0` (each)
- Duct centreline length `L_duct` (port reservoir centreline to starboard
  reservoir centreline, *along the flow path*)

For a clean U-tube where reservoirs are vertical legs of width `b_r`
and the duct is a horizontal channel of length `w_d`, the flow-path
duct length is approximately `W = w_d + b_r` (centreline-to-centreline).
We use `L_duct = W` for the path integral. (This is the same `W` used
in `Q = ρ b_r W² t / 2`.)

Path parametrisation:
- `v ∈ [0, h_0]`: port reservoir, ascending (from duct top to fluid surface)
- `v ∈ [h_0, h_0 + W]`: duct, port-to-starboard
- `v ∈ [h_0 + W, 2h_0 + W]`: starboard reservoir, descending

`A(v)` along the path:
- Reservoirs: `A(v) = A_0`
- Duct: `A(v) = A_d`

S′ integral:
```
S′  =  ∫₀^h_0  (A_0/A_0) dv  +  ∫_h_0^(h_0+W)  (A_0/A_d) dv  +  ∫_(h_0+W)^(2h_0+W)  (A_0/A_0) dv
    =  h_0  +  W·(A_0/A_d)  +  h_0
    =  2·h_0  +  W·(b_r · t)/(h_d · t)
    =  2·h_0  +  W·b_r / h_d
```

Tank natural frequency:
```
ω_t²  =  2g / S′  =  2g / (2 h_0 + W·b_r/h_d)  =  g / (h_0 + W·b_r/(2·h_d))
```

Compare to our code's `c_τ / a_τ = Q·g / [Q·b_r·(W/(2·h_d) + h_0/b_r)]
= g / (W·b_r/(2·h_d) + h_0)`. ✓ **Exact match.** Bertram and PNA agree
on `ω_t` and so does our code.

This locks in:
```
a_τ  =  Q · (h_0 + W·b_r/(2·h_d))  =  (Q/2) · S′         (via Q·g = c_τ and ω_t² = c_τ/a_τ = 2g/S′)
c_τ  =  Q · g
```

## I.4 Path integral S″ for a standard U-tube

Now the harder one. We need `q(v)` = height of the tangent line at point
v above the roll center.

**Tangent direction along the flow path:**
- Port reservoir, ascending: tangent points in the −z direction (upward,
  since z is down). Tangent is purely vertical.
- Duct, port-to-starboard: tangent points in the +y direction. Tangent
  is purely horizontal.
- Starboard reservoir, descending: tangent points in the +z direction
  (downward).

**Vertical position along the path (in body z, COG-referenced):**
- Port reservoir at v ∈ [0, h_0]:  `z(v) = z_d − v`  (starts at z=z_d at
  the duct top, ascends to z=z_d−h_0 at the surface)
- Duct at v ∈ [h_0, h_0 + W]:  `z(v) = z_d` (constant; duct is horizontal)
- Starboard reservoir at v ∈ [h_0 + W, 2h_0 + W]: `z(v) = z_d − (2h_0+W − v)`
  (starts at duct top z=z_d when v = h_0+W, ascends as v decreases ... wait,
  the path is descending here. Let me re-parametrise: at v = h_0+W we're
  at the top of the starboard reservoir at z = z_d − h_0; at v = 2h_0+W
  we're back at the duct top z = z_d. So `z(v) = z_d − h_0 + (v − h_0 − W)`.)

Let me redo this more carefully:

| segment | v range            | z(v)                           | tangent direction |
|---------|--------------------|--------------------------------|-------------------|
| port leg, going up   | [0, h_0]              | z = z_d − v                       | −ẑ (up)    |
| duct, going to stbd  | [h_0, h_0+W]          | z = z_d                           | +ŷ (stbd)  |
| stbd leg, going down | [h_0+W, 2h_0+W]       | z = z_d − h_0 + (v − h_0 − W)     | +ẑ (down)  |
|                      |                       | = z_d − 2h_0 − W + v              |            |

Check: at v = h_0+W, z = z_d − 2h_0 − W + h_0 + W = z_d − h_0 ✓ (top of starboard column).
At v = 2h_0+W, z = z_d − 2h_0 − W + 2h_0 + W = z_d ✓ (back to duct level).

**Height above roll center**:
```
height_above_RC(v)  =  −(z(v) − z_RC)  =  z_RC − z(v)
```

Substituting `z_RC = −KG`:
```
height_above_RC(v)  =  −KG − z(v)
```

**But PNA's `q(v)` is the height of the *tangent line*, not of the point itself.**

For a horizontal tangent (duct), the tangent line is the horizontal line
at z = z_d. Its height above the roll center is `−KG − z_d` (a single
value).

For a vertical tangent (reservoir), the tangent line is the *vertical
line at the same y position*. A vertical line doesn't have a single
"height" — but PNA's path integral `∫ q(v)/R dv` parametrises q by v,
so `q` should be evaluated at the *current point*, not at the tangent
line as an extended object. I think PNA's wording "tangent line ...
above the roll center" is shorthand for "point on the path is above the
roll center" with the *direction* of the tangent determining the **sign
of the contribution**, not the height itself.

Re-reading PNA: *"q(v) is defined to be negative if the tangent line to
the flow path at the point v is below the roll center axis and to be
positive if it is above."* The natural reading is that `q(v)` is just
the (signed) height of the **point v** above the roll center, with the
sign being that of `(z_RC − z(v))`.

But there's a subtlety: as you traverse a vertical reservoir column, all
points are at the same y (= ±W/2) but z varies. The tangent at each
point is vertical. PNA's "tangent line above/below the roll center" then
refers to whether the tangent **line** (extended in both directions) is
above or below — a vertical line can't be classified as above/below
unless you mean it intersects the horizontal plane through the roll
center somewhere finite, in which case **all vertical lines have the
same intersection** with z = z_RC (just at different y), and the
classification doesn't make sense.

→ I think PNA's `q(v)` for a U-tube path is **the perpendicular distance
from the roll center to the tangent line**, signed by which side. For
horizontal tangents (duct), this is the vertical distance. For vertical
tangents (reservoirs), this is the horizontal distance. Let me try this
interpretation:

**Reinterpretation: `q(v) = signed perpendicular distance from roll center to tangent line at v`.**

| segment              | tangent direction | perpendicular distance                | sign      |
|----------------------|-------------------|---------------------------------------|-----------|
| port leg, going up   | −ẑ                | horizontal distance = |y_port| = W/2  | port = -y |
| duct                 | +ŷ                | vertical distance = z_RC − z_d = −KG−z_d | up = +   |
| stbd leg, going down | +ẑ                | horizontal distance = |y_stbd| = W/2  | stbd = +y |

Hmm, this also has interpretation issues — "above the roll center" for a
vertical tangent line doesn't fit naturally either.

> ⚠️ **PNA's sign rule for `q(v)` is ambiguous without Fig. 95b.** Two
> reasonable readings:
>   (a) `q(v) = z_RC − z(v)` (signed height of the point itself)
>   (b) `q(v)` involves the perpendicular distance and direction of the
>       tangent line, in a way that gives one consistent sign rule for
>       horizontal duct flow.
>
> I will compute `S″` under interpretation **(a)** as it's the most
> straightforward, then check the result against brucon and Bertram.
> If it disagrees, switch to (b).

### S″ computation under interpretation (a): q(v) = z_RC − z(v)

Three integrals:

**Port leg** (v ∈ [0, h_0], z(v) = z_d − v):
```
∫₀^h_0  (z_RC − z_d + v) / R  dv  =  (1/R) · [(z_RC − z_d)·h_0 + h_0²/2]
                                  =  (h_0/R) · [(z_RC − z_d) + h_0/2]
```

**Duct** (v ∈ [h_0, h_0+W], z(v) = z_d):
```
∫_h_0^(h_0+W)  (z_RC − z_d) / R  dv  =  (W/R) · (z_RC − z_d)
```

**Starboard leg** (v ∈ [h_0+W, 2h_0+W], z(v) = z_d − 2h_0 − W + v):
```
∫_(h_0+W)^(2h_0+W)  (z_RC − z_d + 2h_0 + W − v) / R  dv
```
Let u = v − (h_0+W), du = dv, u ∈ [0, h_0], v = u + h_0 + W:
```
∫₀^h_0  (z_RC − z_d + 2h_0 + W − u − h_0 − W) / R  du
  =  ∫₀^h_0  (z_RC − z_d + h_0 − u) / R  du
  =  (1/R) · [(z_RC − z_d + h_0)·h_0 − h_0²/2]
  =  (h_0/R) · [(z_RC − z_d) + h_0 − h_0/2]
  =  (h_0/R) · [(z_RC − z_d) + h_0/2]
```

**Sum:**
```
S″  =  (h_0/R)·[(z_RC − z_d) + h_0/2]  +  (W/R)·(z_RC − z_d)  +  (h_0/R)·[(z_RC − z_d) + h_0/2]
    =  (1/R) · [2·h_0·(z_RC − z_d) + h_0² + W·(z_RC − z_d)]
    =  (1/R) · [(2·h_0 + W)·(z_RC − z_d) + h_0²]
```

Substituting `z_RC = −KG`:
```
S″  =  (1/R) · [(2·h_0 + W)·(−KG − z_d) + h_0²]
    =  −(1/R) · [(2·h_0 + W)·(KG + z_d) − h_0²]
```

### Sanity check on dimensions and limits

- All terms have units of length²/length = length ✓ (S″ should have the
  same units as S′).
- For a tank centred at the waterline (z_d = z_RC = −KG, so the duct is
  at the waterline level): the (2h_0+W)·(z_RC−z_d) term vanishes,
  leaving `S″ = h_0²/R > 0`. So a tank with duct *at* the waterline
  still has positive S″ from the reservoir columns extending upward.
  Reasonable.
- For a tank deep below the waterline (z_d ≫ −KG, so z_RC − z_d very
  negative): `S″ ≈ (2h_0+W)·(z_RC−z_d)/R`, large negative. Tank works
  in the wrong direction. ✓ matches PNA.
- For a tank high in the superstructure (z_d ≪ −KG, z_RC − z_d very
  positive): `S″ ≈ (2h_0+W)·(z_RC−z_d)/R`, large positive. Tank damps. ✓

### Cross-over: where does S″ change sign?

`S″ = 0` when:
```
(2·h_0 + W)·(z_RC − z_d) + h_0²  =  0
z_d − z_RC  =  h_0² / (2·h_0 + W)
```

For CSOV (h_0 = 2.5, W = 18): `z_d − z_RC = 6.25/23 ≈ 0.272 m`. So the
sign-change happens when the duct is **0.272 m below the roll center**
(below the waterline). For a duct *at* the waterline, S″ is just
positive (h_0²/R contribution). For a duct *more than 0.272 m below the
waterline*, S″ goes negative.

For our standard CSOV example with `z_d = +4 m` (duct at keel level,
4 m below COG, ≈ 6.5 m below waterline): well past the sign-change,
deeply in the "wrong direction" regime per PNA.

> **This is the central physics result of this whole investigation.**
> A near-keel-mounted U-tube on a CSOV is, per PNA, working *against*
> the hull's roll damping — not for it. To get the desired anti-roll
> action the tank must be placed **at or above the waterline**, ideally
> well into the superstructure (large negative z_d).

## I.5 Mapping PNA's S″ to our `a_phi`

PNA's ξ_t is the dimensionless height parameter that controls the sign
and magnitude of the dynamic absorber action. In our EOM language, the
dynamic moment from the tank on the hull is `−a_phi · τ̈`, and `a_phi`
plays exactly the role of "what controls absorber sign/magnitude."

Hypothesis: `a_phi ∝ S″`, with proportionality constant set by matching
units and a known limit.

**Dimensional check:**
- `[a_phi] = kg·m²` (cross-inertia in tank EOM `a_τ τ̈ = ... − a_phi φ̈ + ...`,
  with τ in rad and φ in rad gives kg·m²).
- `[S″] = m`.
- So `a_phi = K · S″` requires `[K] = kg·m`.
- Natural candidates: `K = ρ·V·something`, or in our notation `K ∝ Q/(W·something)`.

`Q = ρ b_r W² t / 2`, units kg·m². `Q/W` has units kg·m. So try
`a_phi = (Q/W) · S″ · R` or similar. The factor R · 1/W gives a clean
dimensionless ratio.

Actually, there's a cleaner way: match against the known case.

**Known case 1: a_phi for the reservoir-only contribution.**

From the Lagrangian derivation in Appendix A.4.2, the reservoir cross
term in T is `−Q · h_0 · τ̇ · φ̇`. The coefficient on `τ̇·φ̇` is `−Q·h_0`,
so `a_phi(reservoir only) = +Q·h_0`. (Sign convention: cross-term in
T is `−a_phi · τ̇ · φ̇` per the convention used in App.A.)

Wait, let me re-check the sign. In Appendix A I wrote
`α = −Q·[(w_d/W)·z_d + h_0]` where `T_cross = α · τ̇ · φ̇`. And
`a_phi = −α = +Q·[(w_d/W)·z_d + h_0]`. So reservoir-only (z_d = 0)
gives `a_phi = +Q·h_0`. OK.

For the reservoirs alone (no duct), PNA's path integral has only the
two reservoir legs, no duct contribution. From §I.4:
```
S″_reservoirs_only  =  (1/R) · 2 · (h_0/R)·[(z_RC − z_d) + h_0/2]
                                     ... wait let me redo
```

Actually if there's no duct (or the duct is shrunk to zero length), the
S″ is just the two reservoir-leg integrals:
```
S″_reservoirs_only  =  2 · (h_0/R) · [(z_RC − z_d) + h_0/2]
                    =  (2 h_0/R) · (z_RC − z_d)  +  h_0²/R
```

And S′_reservoirs_only = 2 h_0 (no duct contribution to S′ either).
So ξ_t for reservoirs only is `S″/S′ = [(z_RC−z_d) + h_0/2] / R`.

For the case `z_d = z_RC` (reservoirs centred at the roll center):
`ξ_t = (h_0/2)/R`. With R = W/2, `ξ_t = h_0/W`.

Hmm, this is getting complex. Let me try a different tactic: **derive `a_phi` directly from the PNA framework using PNA's claim that the tank moment is `Δ_tank · ξ_t · ω_t² · φ_amplitude · sin(...)` or similar.**

Actually, the cleanest mapping is:

**PNA does not give an explicit formula for the tank moment as a
function of τ and τ̈. It gives parameters (μ, σ_t, ζ_t, η_s, ξ_t) that
appear in an effective coupled roll/tank EOM (Section 3.8 of PNA, not
shown here). Without that section, we cannot directly map ξ_t to
a_phi.**

→ We need to make do with the **qualitative** PNA result (sign of S″ /
ξ_t determines damping/anti-damping) and the **quantitative** Bertram
result (c_phi = Q·g) plus our own Lagrangian derivation for the inertial
coefficient `a_phi`.

## I.6 Reconciliation with brucon's formula

Brucon uses `a_phi = Q·(z_d + h_0)` in z-down COG-referenced coordinates.
Compare to:
- **Appendix A derivation (this doc)**: `a_phi = Q·[(w_d/W)·z_d + h_0]`.
- **PNA's S″**: `S″ = (1/R)·[(2·h_0 + W)·(z_RC − z_d) + h_0²]`.

The PNA `S″` is referenced to `z_RC = -KG`, but brucon's `a_phi` is
referenced to the COG (z_RC = 0 effectively). If we set `z_RC = 0` in
PNA's S″:
```
S″|_(z_RC=0)  =  (1/R) · [(2·h_0 + W)·(−z_d) + h_0²]
              =  (1/R) · [h_0² − (2·h_0 + W)·z_d]
```

Setting R = W/2:
```
S″|_(z_RC=0, R=W/2)  =  (2/W) · [h_0² − (2·h_0 + W)·z_d]
```

Expanding the brucon expression `Q·(z_d + h_0)` and looking for a match:
- The brucon expression is **linear** in (z_d + h_0). It has no h_0²
  term and no W·z_d term.
- The PNA S″ has both an h_0² piece *and* a (2h_0+W)·z_d piece.
- Brucon's coefficient on z_d is +Q. PNA's coefficient on z_d (in S″,
  with R=W/2) is `−(2/W)(2h_0 + W) = −(4h_0/W + 2)`.

These don't match in either form or sign. **Brucon's `a_phi` formula
does NOT come directly from the PNA path integral in any obvious way.**

Possible explanations:
1. Brucon's formula is a simplified / approximate / wrong form derived
   independently (possibly from Winden's work, with its own conventions
   that differ from PNA's).
2. PNA's `S″` is not the right thing to map to `a_phi`. Maybe it's
   `(S″ − const)` or some other combination.
3. PNA's roll center being at the waterline (not COG) introduces a
   constant offset that absorbs into other terms (b_44, c_44) when
   re-referenced to COG.

This is unresolved. The clean way forward is to **derive `a_phi` from
first principles** (the Lagrangian, Appendix A) and use PNA's *qualitative*
result (sign of S″ at the operating placement) as the validation check.

## I.7 Summary of where we are

**Locked in (confirmed by multiple sources):**
- z-down body frame, COG-referenced for tank geometry (brucon, pdstrip).
- Roll center at the waterline for PNA-style analysis (z_RC = −KG).
- `c_τ = Q·g` (Bertram 4.122 → c_phi same form).
- `a_τ = Q · (h_0 + W·b_r/(2·h_d))` (Bertram 4.123, PNA `2g/S′`).
- `c_phi = Q·g` (Bertram 4.122).

**Open / to investigate next:**
- The exact formula for `a_phi`. Three candidates remain:
  - **brucon**: `Q · (z_d + h_0)` (no path-integral derivation found,
    but used in shipping industry via Winden).
  - **Appendix A (this doc)**: `Q · [(w_d/W)·z_d + h_0]` (Lagrangian
    in COG frame, but the `(w_d/W)` factor is the most uncertain step).
  - **PNA S″ scaled**: some function of `(2·h_0 + W)·(z_RC − z_d) + h_0²`,
    referenced to the waterline rather than COG.
- The **qualitative** PNA prediction "S″ < 0 → tank works wrong direction"
  is unambiguous and gives an empirical test: place the tank below the
  waterline by more than `h_0² / (2h_0 + W) ≈ 0.27 m` and observe
  whether the simulation predicts amplification rather than reduction
  of the hull roll. **If it predicts reduction, our `a_phi` is wrong.**

**Recommended next action**: implement *both* candidate `a_phi` formulas
in the prototype as a switchable option, run `examples/investigate_reduction.py`
with each, and compare the placement-sweep against PNA's qualitative
prediction. The formula whose placement physics matches PNA wins.

---

# Appendix A: superseded direct Lagrangian derivation

The following sections (originally §1–§11) are kept for the reasoning
trail. They contain a direct Lagrangian derivation in COG-referenced
z-down coordinates, but reach a conclusion (`a_phi = Q·[(w_d/W)·z_d + h_0]`,
"place low") that conflicts with PNA's qualitative claim ("place
above the roll center"). The likely error is in §A.4.1 (duct cross-term
with the `(w_d/W)` factor), but it has not yet been fully diagnosed.

The final §A.9 conclusion that "absorber action is sign-independent at
perfect tuning" is **algebraically correct** for a 2-DOF coupled
oscillator at the simultaneous tuning condition `ω_wave = ω_n,hull = ω_n,tank`,
but does **not** rule out off-tuning effects or the kind of damping-vs-anti-damping
distinction PNA makes (which arises at the full coupled-system resonance,
not at the simultaneously-tuned point).

---

## A.1 Coordinate system and sign conventions

## 1. Coordinate system and sign conventions

**Body-fixed right-handed frame**, origin at the ship COG, fixed in the
hull (rotates with it):

| axis | direction                  |
|------|----------------------------|
| +x   | forward (toward the bow)   |
| +y   | starboard                  |
| +z   | down                       |

This is the SNAME / pdstrip / brucon convention. **Note z is down.**

**Roll** φ is rotation about +x by the right-hand rule. With +z down and
+y starboard, φ > 0 means **starboard rolls down, port rolls up**. ("The
ship leans to starboard.")

**Roll rate** φ̇ has the same sign convention. φ̇ > 0 means the ship is
rotating starboard-downward.

For a body-fixed point at position `(x, y, z)`, the inertial velocity
contribution from pure roll φ̇ (other DOFs frozen) is, to first order in φ:

```
Ẏ_inertial  =  +z · φ̇       (a point with z>0, i.e. below COG, moves to +y / starboard)
Ż_inertial  =  -y · φ̇       (a point at +y / starboard moves in -z, i.e. upward)
```

Sanity check: for φ̇ > 0, the keel (large positive z) should move to
starboard (+y) — yes, `Ẏ = +z·φ̇ > 0` ✓. The starboard rail (large +y)
should move down (+z) — but the formula gives `Ż = −y·φ̇ < 0` (upward).
That looks wrong, but it isn't: `Ẏ = +z·φ̇, Ż = −y·φ̇` is the velocity at
a fixed *body-frame* coordinate evaluated in the inertial frame. The
starboard rail does move down in the inertial frame; what `Ż = −y·φ̇`
describes is the velocity of the *body point that is currently at body
position +y*, which after time dt is at body position +y still but at
inertial position rotated by φ̇·dt. Since inertial Z is fixed downward
and body Y-axis is rotating downward, an inertially-fixed observer sees
the body-fixed point moving inertially-down too — let me redo this from
the rotation matrix to be safe.

**Rotation matrix derivation.** A right-handed rotation by φ about +x
takes body coordinates `(x_b, y_b, z_b)` to inertial coordinates
`(X, Y, Z)` (assuming COG at inertial origin and only roll motion):

```
[X]   [1   0       0   ] [x_b]
[Y] = [0   cosφ   -sinφ ] [y_b]
[Z]   [0   sinφ    cosφ ] [z_b]
```

Differentiating with φ small (cosφ→1, sinφ→φ, ḟφ→φ̇):

```
[Ẋ]   [0   0     0  ] [x_b]   [    0      ]
[Ẏ] ≈ [0   0    -φ̇ ] [y_b] = [ -z_b · φ̇ ]
[Ż]   [0   φ̇    0  ] [z_b]   [ +y_b · φ̇ ]
```

So the **correct** linearised body-rotation velocities are:

```
Ẏ_inertial(at body point (y_b, z_b))  =  -z_b · φ̇
Ż_inertial(at body point (y_b, z_b))  =  +y_b · φ̇
```

**Sanity check, redone:** for φ̇ > 0:
- A point on the keel (z_b = +T large positive) gets `Ẏ = -T·φ̇ < 0`, i.e.
  moves to port. But intuitively, "starboard rolls down" means the keel
  swings to *starboard*, not port?

  Let me think again. "Starboard rolls down" means the hull rotates so
  that the starboard rail moves to large +z (down). The keel, which is
  initially below the COG (at large +z), under such a rotation pivots
  about the longitudinal x-axis — it swings to port (-y) as the
  starboard rail comes down. So `Ẏ = -z_b · φ̇ < 0` for keel ✓.

- The starboard rail (y_b = +B/2): `Ż = +(B/2)·φ̇ > 0`, i.e. moves down
  in inertial Z. ✓ matches "starboard rolls down."

Good. So the correct linearised velocity of a body-fixed point
`(0, y_b, z_b)` under pure roll is:

> **Velocity formula** (z-down body frame, +φ = starboard down):
> ```
> Ẏ = -z_b · φ̇
> Ż = +y_b · φ̇
> ```

This is the **opposite sign** of what I had in v1 and v2 of the Python
code (both of which assumed z-up). The correct formula for `a_phi` will
follow from this.

---

## 2. Tank geometry

The U-tube has two vertical reservoirs (port and starboard) connected
at the bottom by a horizontal duct. Geometry parameters:

| symbol | code name                  | description                                   |
|--------|----------------------------|-----------------------------------------------|
| `b_r`  | `resevoir_duct_width`      | reservoir horizontal width (along ±y), m      |
| `w_d`  | `utube_duct_width`         | duct length port-to-starboard (along ±y), m   |
| `h_d`  | `utube_duct_height`        | duct vertical extent (along ±z), m            |
| `t`    | `tank_thickness`           | along-ship extent (along ±x), m               |
| `h_0`  | `undisturbed_fluid_height` | undisturbed fluid level in each reservoir, measured *upward* from duct top, m |
| `H_t`  | `tank_height`              | total reservoir height, m                     |
| `z_d`  | `utube_datum_to_cog`       | duct datum vertical position, body frame, m. **+ve z_d means duct datum is *below* COG** (z is down). |
| `x_T`  | `tank_to_xcog`             | along-ship offset of tank from COG, m         |

**Centreline distance between reservoirs**: `W ≡ w_d + b_r`. (The
reservoirs are at body y = ±W/2.)

**Convention switch from earlier prototype**: the previous Python code
defined `utube_datum_to_cog` as "positive = above COG" (z-up). We are
now matching brucon: **positive = below COG / above keel** (z-down).
For a CSOV with KG = 2.5 m above the waterline and duct at the keel
(draft T = 6.5 m), the duct is at body z = +6.5 − 2.5 = +4 m below COG,
so `z_d = +4 m`. *[verify against actual brucon definition before
locking in]*

**Fluid coordinate τ**: defined so that `τ > 0` means the *starboard*
reservoir sits *higher* (fluid has flowed from port to starboard,
*upward* in the starboard reservoir means *toward smaller z* since z is
down). Equivalently, the starboard free-surface position is
`z_starboard = z_d − h_0 − (W/2)·τ` (smaller z = higher) and port is
`z_port = z_d − h_0 + (W/2)·τ` (larger z = lower).

---

## 3. Fluid kinematics

Treat the fluid as one-dimensional (slug-flow assumption: incompressible,
uniform velocity across each cross-section). With τ̇ > 0 starboard rises,
so:

- **Duct**: fluid moves in +y direction (port to starboard) at speed
  `u_d = (W/2) · τ̇ · (b_r / w_d)·(... no wait, do this from continuity)`.

  Continuity: volume flow rate is the same in duct and reservoir.
  Reservoir cross-section `A_r = b_r · t`, duct cross-section
  `A_d = h_d · t` (the duct is `h_d` tall and `t` along-ship; flow is
  along ±y over length `w_d`).

  Reservoir vertical velocity (starboard): the free surface descends in
  z at rate `Ż_surf,starboard = -(W/2)·τ̇` (rises = moves to smaller z).
  Volume flow into starboard reservoir = `A_r · |Ż_surf| = b_r·t·(W/2)·τ̇`.

  Duct horizontal velocity (toward +y): `u_d = (volume flow) / A_d
  = b_r·t·(W/2)·τ̇ / (h_d·t) = b_r·W·τ̇ / (2·h_d)`.

  So:

  > `u_d = (b_r · W) / (2 · h_d)  ·  τ̇`        [body-frame velocity, +y direction]

- **Starboard reservoir fluid**: vertical velocity (in body frame, in z
  direction) `w_r,stbd = -(W/2) · τ̇` (negative z = upward).

- **Port reservoir fluid**: vertical velocity `w_r,port = +(W/2) · τ̇`.

These are the **body-frame** (relative to the rotating hull) fluid
velocities. The total **inertial** fluid velocity at each point is body
velocity + body-frame-rotation contribution from §1.

---

## 4. Kinetic energy

Compute T = ½ ∫ ρ |v_inertial|² dV for each fluid region.

### 4.1 Duct fluid

Duct centroid at body position `(x = ±along-ship-centred, y = 0, z = z_d)`.
Treat the duct as a thin slab at z = z_d with along-y extent w_d, mass
`m_d = ρ · w_d · h_d · t`.

For an element at body position `(0, y, z_d)` with `y ∈ [-w_d/2, +w_d/2]`:

- Body-frame velocity (fluid sloshing): `v_b = (0, +u_d, 0)` (uniform across duct).
- Inertial velocity from hull roll: `v_rot = (0, -z_d · φ̇, +y · φ̇)`.

Total inertial velocity components:
```
Y-component:  +u_d  -  z_d · φ̇
Z-component:  +y · φ̇
```

Squared magnitude (per unit mass):
```
|v|² = (u_d - z_d·φ̇)²  +  (y·φ̇)²
     = u_d² - 2·u_d·z_d·φ̇ + z_d²·φ̇²  +  y²·φ̇²
```

Integrating over the duct mass (∫ y² dm over y ∈ [-w_d/2, +w_d/2] is
ρ·t·h_d·w_d³/12 = m_d·w_d²/12, a moment-of-inertia term independent of
τ; ∫ y dm = 0 by symmetry):

```
T_duct = ½ m_d (u_d² - 2·u_d·z_d·φ̇ + z_d²·φ̇²)  +  ½ m_d (w_d²/12) φ̇²
```

The **cross term** is `-m_d · u_d · z_d · φ̇`.

Substituting `u_d = (b_r·W)/(2·h_d) · τ̇` and `m_d = ρ·w_d·h_d·t`:

```
cross term in T_duct  =  -ρ·w_d·h_d·t · (b_r·W)/(2·h_d) · z_d · τ̇ · φ̇
                      =  -ρ·w_d·b_r·W·t·z_d · (1/2) · τ̇ · φ̇
                      =  -(W/W) · ρ·w_d·b_r·W·t·z_d/2 · τ̇·φ̇
```

Defining `Q ≡ ρ·b_r·W²·t / 2` (matches brucon and our Python code), and
noting `w_d·W/W² = w_d/W`, this becomes:

```
cross term in T_duct  =  -Q · (w_d/W) · z_d · τ̇·φ̇
```

Hmm — that has a `(w_d/W)` factor, not a clean `Q · z_d · τ̇·φ̇`. Let me
re-examine: the duct fluid mass `m_d = ρ·w_d·h_d·t` is the mass of fluid
*in the duct only*, and `u_d` is the duct-fluid velocity. So
`m_d · u_d = ρ·w_d·h_d·t · (b_r·W)/(2·h_d)·τ̇ = ρ·w_d·b_r·W·t/2 · τ̇
= Q·(w_d/W)·τ̇`.

The factor `(w_d/W) = w_d/(w_d+b_r)` is **not** unity unless `b_r = 0`.
So the duct cross-term is `-Q · (w_d/W) · z_d · τ̇·φ̇`, **not** `-Q·z_d·τ̇·φ̇`.

> ⚠️ This is already different from what brucon, my v1, and my v2 all
> assumed. They all bundled the duct cross term as `±Q·z_d·τ̇·φ̇` with
> coefficient unity. The actual coefficient has a `(w_d/W)` factor.

For the headline CSOV example (`w_d = 16`, `b_r = 2`, `W = 18`):
`w_d/W = 16/18 ≈ 0.889`. Not negligible.

### 4.2 Reservoir fluid

Each reservoir is a vertical column at `y = ±W/2` (centreline; really
spanning `y = ±W/2 ± b_r/2`, but use the centroid for the leading-order
KE), height `h_0` (undisturbed, but the actual height changes with τ;
the change is O(τ) and contributes at O(τ²), so to leading order use h_0).

Reservoir mass per side: `m_r = ρ·b_r·h_0·t`.

**Starboard reservoir** at `y = +W/2`. Body-frame fluid velocity is
purely vertical: `w_r = -(W/2)·τ̇` in body z (upward when τ̇>0).

Centroid position: body coordinates `(0, +W/2, z_d - h_0/2)` (mid-height
of the fluid column, measured from duct top upward = smaller z).

Inertial velocity of an element at body position `(0, +W/2, z)`:
- Body-frame velocity: `(0, 0, w_r) = (0, 0, -(W/2)·τ̇)`.
- Rotation contribution: `(0, -z·φ̇, +(W/2)·φ̇)`.

Total:
```
Y-component:  -z · φ̇
Z-component:  -(W/2)·τ̇  +  (W/2)·φ̇
```

Squared magnitude per unit mass:
```
|v|² = z²·φ̇²  +  ((W/2)·(φ̇ - τ̇))²
     = z²·φ̇²  +  (W²/4)·(φ̇² - 2·φ̇·τ̇ + τ̇²)
```

The cross term is `-m_r · (W²/4) · 2 · ½ · φ̇ · τ̇ = -½·m_r·(W²/2)·φ̇·τ̇`.

Wait, I need to integrate ∫ over the reservoir column. The W and τ̇
contributions are uniform over the column (no z-dependence), so per unit
mass the cross term is `-(W²/2)·τ̇·φ̇·(1/2) = -(W²/4)·τ̇·φ̇` after the
½ from KE. Then multiplied by `m_r`:

```
cross term (starboard)  =  -m_r · (W²/4) · τ̇ · φ̇·... 
```

Let me redo this more carefully. KE per unit mass = ½|v|². The
**linear-in-τ̇·φ̇** term in ½|v|² per unit mass is:
```
½ · 2 · (-(W/2)·τ̇) · (+(W/2)·φ̇)  =  -(W²/4)·τ̇·φ̇
```

Multiplied by `m_r` and integrated (uniform across the column):
```
(T_r,stbd)_cross  =  -m_r · (W²/4) · τ̇ · φ̇
```

**Port reservoir** at `y = -W/2`. Body-frame fluid velocity: `w_r =
+(W/2)·τ̇` (downward = toward larger z when τ̇>0; fluid drains).

Inertial velocity at body position `(0, -W/2, z)`:
```
Y-component:  -z · φ̇
Z-component:  +(W/2)·τ̇  +  (-W/2)·φ̇  =  (W/2)·(τ̇ - φ̇)
```

Linear-in-τ̇·φ̇ term in ½|v|²:
```
½ · 2 · (+(W/2)·τ̇) · (-(W/2)·φ̇)  =  -(W²/4)·τ̇·φ̇
```

Same sign as starboard. Integrated:
```
(T_r,port)_cross  =  -m_r · (W²/4) · τ̇ · φ̇
```

**Total reservoir cross term:**
```
T_reservoirs,cross  =  -2 · m_r · (W²/4) · τ̇·φ̇  =  -(m_r·W²/2) · τ̇·φ̇
```

With `m_r = ρ·b_r·h_0·t`:
```
T_reservoirs,cross  =  -(ρ·b_r·h_0·t · W²/2) · τ̇·φ̇  =  -Q·h_0 · τ̇·φ̇
```
(since `Q = ρ·b_r·W²·t/2`, we have `ρ·b_r·t·W²/2 = Q`, so `ρ·b_r·h_0·t·W²/2 = Q·h_0`).

✓ Clean, no `w_d/W` factor (because reservoirs span the full reservoir width `b_r` and the fluid moves uniformly in z across that width).

### 4.3 Total cross term and `a_phi`

```
T_cross  =  T_duct,cross  +  T_reservoirs,cross
         =  -Q·(w_d/W)·z_d · τ̇·φ̇  -  Q·h_0 · τ̇·φ̇
         =  -Q · [(w_d/W)·z_d + h_0] · τ̇·φ̇
```

The Lagrangian KE contains `T_cross`, and the cross-inertia coefficient
in the EOM is `∂²T / ∂τ̇ ∂φ̇` (the coefficient on `τ̇·φ̇` is *negative* of
this in my equation form; let me be careful).

For a Lagrangian `L = T - V` and Lagrange's equation
`d/dt(∂L/∂q̇) - ∂L/∂q = 0`:

The term `T_cross = α·τ̇·φ̇` contributes:
- to τ-equation: `d/dt(∂T_cross/∂τ̇) = d/dt(α·φ̇) = α·φ̈`
- to φ-equation: `d/dt(∂T_cross/∂φ̇) = d/dt(α·τ̇) = α·τ̈`

So the τ-EOM has `+α·φ̈` on the LHS, equivalently `-α·φ̈` on the RHS.
Likewise the φ-EOM has `+α·τ̈` on the LHS.

With `α = -Q·[(w_d/W)·z_d + h_0]`:

> **Cross-inertia coefficient (z-down convention, brucon-compatible):**
> ```
> a_phi  =  -α  =  Q · [(w_d/W)·z_d + h_0]
> ```

(Defining `a_phi` as the coefficient appearing as `+a_phi·φ̈` on the RHS
of the τ-EOM with a minus sign — i.e., `a_τ·τ̈ + ... = -a_phi·φ̈ + ...`,
and as `+a_phi·τ̈` on the LHS of the φ-EOM, matching the Python code's
sign convention.)

### 4.4 Sanity check against z_d sign

In z-down convention, **larger z_d = duct further below COG = closer to
the keel**. So:

- Duct at COG level (`z_d = 0`): `a_phi = Q·h_0 > 0`. Pure reservoir
  contribution.
- Duct below COG (`z_d > 0`, near keel): `a_phi = Q·[(w_d/W)·z_d + h_0] >
  Q·h_0`. Larger.
- Duct above COG (`z_d < 0`, in superstructure): `a_phi` decreases. At
  `z_d = -h_0·W/w_d` (well above the COG), `a_phi = 0`. Above that, `a_phi`
  goes **negative**.

So in this z-down derivation, **placement near the keel maximises |a_phi|
positive, placement high in the superstructure can drive a_phi to zero
or negative**.

The dynamic absorber moment on the hull is `-a_phi·τ̈`. At resonance with
proper tuning, `τ̈` is in phase with `φ` (and 180° out of phase with the
exciting moment). So the *magnitude* of the absorber action scales with
|a_phi|. **Both** keel-placement (large positive a_phi) and high-mast
placement (large negative a_phi) give large |a_phi| and thus large
absorber action. The intermediate point `z_d ≈ -h_0·W/w_d` (well above COG)
gives **zero** absorber action.

> ⚠️ This **disagrees** with the post-v2 placement sweep result that
> showed monotonic improvement with height. Re-running the placement
> sweep with the correct `a_phi = Q·[(w_d/W)·z_d + h_0]` should reveal a
> minimum at intermediate height and large reduction at *both* extremes.

This is also potentially consistent with both the textbook intuitions:
- Keel placement (Bertram-style flume tank, fluid mass low): large
  positive `a_phi`, deep absorber action.
- High superstructure placement (PNA "place high"): large negative
  `a_phi`, also deep absorber action — and additionally raises the
  metacentric height of the cargo (a separate static benefit not
  captured in this dynamic model).

The intermediate-height "dead zone" at `z_d ≈ -h_0·W/w_d` (e.g. for our
CSOV with h_0=2.5, W=18, w_d=16: `z_d ≈ -2.8 m`, i.e. 2.8 m above COG ≈
on the main deck) is a placement to **avoid**.

---

## 5. Potential energy

Hydrostatic PE of the fluid in the gravitational field. Gravity points
in +z (down). Reservoir fluid centroid (per side) is at body z-coordinate
`z_d - h_0/2 - (W/4)·(±τ) ... ` actually let me set this up cleanly.

For a column of fluid in the starboard reservoir, height changes from
`h_0` (undisturbed) to `h_0 + (W/2)·τ` (when τ>0). Centroid at z =
`z_d − (h_0 + (W/2)·τ)/2`. Mass of column = `ρ·b_r·t·(h_0 + (W/2)·τ)`.

PE per side (gravity potential `g · Z` per unit mass with Z measured downward;
but conventionally PE = -m·g·Z if Z is downward, since PE *increases* with
height = with smaller z; let me use PE = +m·g·(−z) = −m·g·z so that PE
increases as z decreases, i.e. as the fluid rises):

```
V_starboard  =  -ρ·b_r·t·(h_0 + (W/2)·τ) · g · [z_d − (h_0 + (W/2)·τ)/2]
```

Expand and keep terms to O(τ²):

```
Let H_s = h_0 + (W/2)·τ.  Centroid z = z_d − H_s/2.
V_s = -ρ·b_r·t·g·H_s · (z_d − H_s/2)
    = -ρ·b_r·t·g·[H_s·z_d − H_s²/2]
```

Similarly for port with `H_p = h_0 − (W/2)·τ`:
```
V_p = -ρ·b_r·t·g·[H_p·z_d − H_p²/2]
```

Total `V_reservoirs = V_s + V_p`:
```
H_s + H_p = 2·h_0 (independent of τ, drops out as constant)
H_s² + H_p² = 2·h_0² + 2·(W/2)²·τ² = 2·h_0² + (W²/2)·τ²
```

So:
```
V_reservoirs = -ρ·b_r·t·g·[2·h_0·z_d − ½·(2·h_0² + (W²/2)·τ²)]
             = const  +  ρ·b_r·t·g·(W²/4)·τ²
             = const  +  (Q·g/2)·τ²
```

(since `Q = ρ·b_r·W²·t/2`, so `ρ·b_r·t·W²/4 = Q/2`).

So the τ-stiffness contribution is `c_tau = Q·g`. ✓ matches our code and
brucon.

**Cross PE term (τ × φ):** Now include the effect of hull roll φ.
Under roll, the body-frame z-axis tilts. Gravity in the body frame has
components `(0, +g·sinφ, +g·cosφ) ≈ (0, +g·φ, +g)`.

The body-frame Y-component of gravity, `g·φ`, does work on horizontal
fluid displacement. The fluid centre of mass shifts laterally with τ.

When τ > 0, fluid has moved from port to starboard. Net shift: mass `m_τ`
moved from port reservoir centroid `y = -W/2` to starboard reservoir
centroid `y = +W/2`, lateral distance W. Net lateral displacement of total
fluid CoM:

```
ΔY_CoM · M_total  =  m_τ · W
```

with `m_τ = ρ·b_r·t·(W/2)·τ` (mass in the difference column of height
`(W/2)·τ`). So:

```
m_τ · W  =  ρ·b_r·t·(W²/2)·τ  =  Q·τ
```

Body-frame gravity work against this lateral CoM shift, when the hull
also rolls by φ: the gravity vector has body-Y component `g·φ` (in +y
direction, toward starboard, when φ>0 = ship leans starboard).

The PE contribution from a CoM at body-Y position `Y_CoM = ΔY_CoM` in a
body-frame gravity with Y-component `g·φ` is:

```
V_cross = -ΔY_CoM · M_total · (g·φ)  =  -(m_τ · W) · g · φ  =  -Q·g·τ·φ
```

(Negative because gravity in +y does positive work as the CoM moves in
+y, lowering PE.)

So:
```
V_cross  =  -Q·g·τ·φ  =  -c_phi·τ·φ        with c_phi = Q·g
```

Lagrange's equation for τ:
```
∂L/∂τ = -∂V/∂τ = -∂V_τ/∂τ - ∂V_cross/∂τ = -Q·g·τ + Q·g·φ = -c_τ·τ + c_phi·φ
```

So the τ-EOM has `+c_phi · φ` on the RHS (after moving `-c_τ·τ` to LHS):

```
... = +c_phi · φ + ...
```

✓ matches v1 and v2 of our code (and *disagrees* with brucon's `-c_phi·φ`).

Lagrange's equation for φ:
```
∂L/∂φ = -∂V_cross/∂φ = +Q·g·τ = +c_phi·τ
```

So the φ-EOM has `+c_phi·τ` on the RHS, i.e., the **roll moment from
the tank's static fluid offset is `+c_phi·τ`** ✓.

This matches Bertram 4.122 in magnitude:
```
M = c_phi · τ = Q·g·τ = ρ·b_r·t·W²/2 · g · τ
            = ρ·g·(b_r·t)·(W/2·τ)·W
            = ρ·g·A_0·h_col·B_1   ✓
```
with `A_0 = b_r·t` (reservoir cross-section), `h_col = (W/2)·τ` (column
rise), `B_1 = W` (reservoir centre spacing). Confirmed.

---

## 6. Self-inertia and damping (a_tau, b_tau, c_tau)

These are not affected by the sign-convention question. From Bertram
4.123 with our duct parameterisation (duct as h_d × t cross-section,
length W along ±y):

```
ω_n,tank² = 2g / [A_0 · ∫(1/A) ds]
         = 2g / [A_0 · (2·h_0/A_0 + W/A_d)]
         = g / (h_0 + A_0·W/(2·A_d))
         = g / (h_0 + b_r·W/(2·h_d))
```

For an oscillator with `c_tau = Q·g` and `ω_n² = c_tau / a_tau`:
```
a_tau = c_tau / ω_n² = Q·g / [g / (h_0 + b_r·W/(2·h_d))]
      = Q · (h_0 + b_r·W/(2·h_d))
      = Q · b_r · (W/(2·h_d) + h_0/b_r)        ← matches our code line 160
```

✓ confirmed against Bertram.

`b_tau = a_tau · μ` per our code (with `μ = tank_wall_friction_coef`,
dimensionless damping coefficient).

---

## 7. Summary: corrected coefficients

In **z-down body-frame convention** (z_d positive = duct *below* COG /
near keel):

| coefficient | formula                                  | matches?              |
|-------------|------------------------------------------|-----------------------|
| Q           | `ρ · b_r · W² · t / 2`                    | brucon ✓              |
| W           | `w_d + b_r`                               | brucon ✓              |
| a_tau       | `Q · b_r · (W/(2·h_d) + h_0/b_r)`         | brucon ✓, Bertram ✓   |
| b_tau       | `a_tau · μ`                               | brucon ✓              |
| c_tau       | `Q · g`                                   | brucon ✓              |
| **a_phi**   | `Q · [(w_d/W)·z_d + h_0]`                 | **none** — see §4.3   |
| c_phi       | `Q · g`                                   | brucon ✓, Bertram ✓   |

**Equations of motion** (z-down, +φ = stbd down, +τ = stbd reservoir higher):
```
Tank:    a_tau τ̈ + b_tau τ̇ + c_tau τ  =  -a_phi φ̈ + c_phi φ
Hull:    M_tank_on_hull  =  c_phi τ - a_phi τ̈
```

**Differences from earlier versions:**

| version | a_phi formula             | sign convention for z_d |
|---------|---------------------------|--------------------------|
| brucon  | `Q · (z_d + h_0)`         | z-down (+ = below COG)   |
| my v1   | `Q · (h_0 - z_d)`         | z-up (+ = above COG)     |
| my v2   | `Q · (z_d - h_0)`         | z-up (+ = above COG)     |
| **this doc**| `Q · [(w_d/W)·z_d + h_0]` | z-down (+ = below COG)   |

Brucon's formula is closest in form to this doc's (both have `+h_0`,
both have `+z_d`-style term), but lacks the `(w_d/W)` factor. For typical
geometries `w_d/W ≈ 0.85-0.95`, so brucon over-estimates the z_d term by
~5-15%. **This is the smallest of the discrepancies we've discussed.**

The v1 and v2 formulas are wrong because they used z-up convention but
the original Python code's z_d was read as z-down (or vice versa) — the
code/docstring/test inconsistencies in `utube_open.py` are evidence of
this confusion.

---

## 8. Verifications independent of derivation algebra

These are physical invariants that any correct derivation must satisfy.
Each is implemented as a behavioural test in `tests/test_utube_open.py`.

1. **Bertram 4.122 (static c_phi)**: For a tank held at constant τ with
   φ frozen, the static moment on the hull is `ρ·g·A_0·h_col·B_1`.
   → tests `c_phi = Q·g`. **PASS** under all three formulas (a_phi
   doesn't enter).

2. **Bertram 4.123 (tank natural frequency)**: For the uncoupled tank
   sloshing alone, `ω_n² = 2g/(A_0·∫(1/A)ds)`. → tests `c_τ/a_τ`.
   **PASS** under all formulas (a_phi doesn't enter).

3. **Lagrangian reciprocity**: The `τ̈` coefficient in the φ-EOM equals
   the `φ̈` coefficient in the τ-EOM (both are `a_phi`). → enforces our
   bug-fix #3 (brucon used `a_τ` instead of `a_phi` in `RollMoment`).
   **PASS** with correct code regardless of `a_phi` formula.

4. **Static lee-side fluid response**: With the hull held at constant
   φ > 0 and the tank allowed to settle (long-time equilibrium of
   `c_τ τ = c_phi φ` with `φ̈ = τ̈ = 0`), τ_∞ = (c_phi/c_τ)·φ = +φ.
   Fluid sits on the same side as the lean (lee side) — the
   destabilising free-surface effect. → enforces bug-fix #2 (brucon's
   `-c_phi·φ` would give τ_∞ = -φ, fluid on weather side, wrong).
   **PASS** with corrected sign on tank RHS.

5. **Resonant absorber phase relation (Bertram-derived)**: at
   `ω = ω_n,tank`, with negligible tank damping, the steady-state
   tank-on-hull moment is 180° out of phase with the wave moment.
   → constrains the *sign* of `a_phi` — see derivation in §9 below.
   **TBD** as a quantitative test.

6. **Placement-sensitivity invariant**: there exists an intermediate
   placement height where `a_phi → 0` and the dynamic absorber action
   vanishes, with non-trivial reduction at both extremes. This is a
   prediction of the corrected formula `a_phi = Q·[(w_d/W)·z_d + h_0]`
   and contradicts the monotonic-with-height behaviour seen with v2.
   **Should be verified by re-running `examples/investigate_reduction.py`
   after the code is updated.**

---

## 9. The 90°+90° phase invariant (Bertram, p. ~110)

Bertram describes the well-tuned absorber as: wave excites hull with
some phase, hull rolls 90° behind the wave excitation, fluid sloshes
90° behind hull roll, so fluid moment is 180° out of phase with wave
excitation and cancels it.

**Quantitative formulation.** Consider sinusoidal steady state at
ω = ω_n,tank. Let φ(t) = Re[Φ·e^(iωt)], τ(t) = Re[T·e^(iωt)],
M_wave(t) = Re[M_w·e^(iωt)].

Tank EOM in frequency domain:
```
(-a_τ ω² + iω b_τ + c_τ) T  =  -a_phi (-ω²) Φ  +  c_phi Φ
                                = (a_phi ω² + c_phi) Φ
```

At ω = ω_n,tank, `c_τ = a_τ·ω²`, so the LHS reduces to `iω·b_τ·T`:
```
iω·b_τ·T  =  (a_phi ω² + c_phi) Φ
```

Therefore `T / Φ = (a_phi ω² + c_phi) / (iω·b_τ)`. The factor `1/i = -i`
introduces a -90° phase: T lags Φ by 90° (provided `a_phi ω² + c_phi > 0`).

If `a_phi ω² + c_phi < 0` (which can happen with `a_phi` very negative,
e.g. duct extremely high in the superstructure), T leads Φ by 90°
instead. *Either* sign produces a 90° quadrature relationship — what
changes is whether the absorber adds or subtracts at the next stage.

Hull-side tank moment: `M_t = c_phi T - a_phi(-ω²)T = (c_phi + a_phi·ω²)·T`.

Substituting:
```
M_t  =  (c_phi + a_phi·ω²) · (c_phi + a_phi·ω²)·Φ / (iω·b_τ)
     =  (c_phi + a_phi·ω²)² · Φ / (iω·b_τ)
```

The square is **always positive**, regardless of sign of `a_phi`.
Therefore `M_t = K·Φ/i = -i·K·Φ` with `K > 0` — i.e. `M_t` **lags** Φ
by 90°.

Hull EOM at ω = ω_n,hull (resonance): `iω·b_44·Φ = M_w + M_t`.
`iω·b_44·Φ` is +90° ahead of Φ; `M_t` is −90° behind Φ; net hull-side
balance requires `M_w` to be +90° ahead of Φ for the bare ship case
(`M_t = 0`). With the tank, `M_t` adds to the damping term — both are
quadrature with Φ — and the absorber works.

**Conclusion**: at perfect tuning (`ω_n,tank = ω_n,hull = ω_wave`), the
absorber action is **independent of the sign of `a_phi`** — only `(c_phi
+ a_phi·ω²)²` enters. The reduction magnitude scales with this squared
quantity, which is **minimised** when `a_phi·ω² = -c_phi`, i.e. `a_phi
= -c_phi/ω² = -Q·g/ω_n,tank² = -Q·a_τ/Q = -a_τ`. Hmm, that's a clean
condition: **dead-zone placement** is at `a_phi = -a_τ`.

For the corrected formula `a_phi = Q·[(w_d/W)·z_d + h_0]`:
```
a_phi = -a_τ  ⇔  (w_d/W)·z_d + h_0 = -a_τ/Q = -(b_r·W/(2h_d) + h_0)
              ⇔  (w_d/W)·z_d  =  -2·h_0 - b_r·W/(2h_d)
              ⇔  z_d  =  -(W/w_d)·[2·h_0 + b_r·W/(2h_d)]
```

For the CSOV (h_0=2.5, W=18, w_d=16, b_r=2, h_d=0.6):
```
z_d_dead  =  -(18/16) · [5 + 2·18/(2·0.6)]
          =  -1.125 · [5 + 30]
          =  -1.125 · 35
          =  -39.4 m
```

That's 39 m **above** the COG (z-down so negative z_d = above) — way up
in the air, well above any realistic mast. So in practice for sensible
tank placements, `a_phi` is positive and reasonably large, and absorber
action grows monotonically with |a_phi|. The "two extremes both work"
prediction is theoretical but practically only the keel side is
accessible for normal ship geometries.

> **Operational conclusion**: place the tank as **low as possible** in
> the hull (large positive z_d in z-down convention = near the keel).
> This maximises `a_phi = Q·[(w_d/W)·z_d + h_0]` and the absorber
> action `(c_phi + a_phi·ω²)²`. The "place high in superstructure"
> textbook advice must refer either to flume tanks (different physics)
> or to a separate static-stability benefit (raising the cargo deck's
> effective metacentric height by lowering the *combined* CoG of ship
> + tank fluid), neither of which is captured in this dynamic model.

This **also** resolves the v1-vs-v2 conflict: v1 (`Q·(h_0 - z_d)` in z-up
= `Q·(h_0 + z_d)` in z-down, i.e. brucon's formula minus the `(w_d/W)`
factor) gave keel-best placement, which is what the physics wants.
**v2 was wrong, v1 was approximately right, and brucon was off by a
~10% factor `(w_d/W)` and by the missing reciprocity/sign fixes.**

---

## 10. What needs updating in the prototype

After this derivation:

1. **`tanks/utube_open.py`**:
   - Change `utube_datum_to_cog` interpretation to z-down (positive =
     below COG / near keel). Update docstring at line 107.
   - Change `a_phi` formula on line 155 from `Q*(z_d - h_0)` (v2) to
     `Q*((w_d/W)*z_d + h_0)`.
   - Update module docstring at lines 1-93 to reflect this derivation.

2. **All examples and tests**: flip the sign of any `utube_datum_to_cog`
   value (z-up −2.5 m → z-down +4.0 m for keel placement on a CSOV with
   KG = 2.5 m above WL and T = 6.5 m).

3. **README**: update §3 with this derivation (or replace with a pointer
   to this doc), update §3.6 with the corrected placement guidance,
   update bug table to include the `(w_d/W)` factor in addition to the
   sign issues.

4. **`examples/investigate_reduction.py`**: re-run with corrected
   formula; the placement sweep should now show a monotonic decrease
   in reduction as `z_d` decreases (tank rises above keel).

---

## 11. Open questions for textbook cross-check

When PNA Vol III §10 / Bhattacharyya 1978 §8.6 are available:

1. Confirm `a_phi = Q·[(w_d/W)·z_d + h_0]` (or the equivalent in
   whatever convention they use). The `(w_d/W)` factor is the most
   uncertain part — I derived it from m_d·u_d and continuity, but it's
   possible the correct treatment of the duct fluid mass (e.g. including
   the corner regions where the duct meets the reservoir) absorbs the
   factor.

2. Confirm the sign convention: in z-down body frame, does the duct
   contribution to a_phi enter with `+z_d` (this doc) or `-z_d`?

3. Confirm the "place low" operational conclusion. If the textbook says
   "place high", investigate whether they mean for static stability
   reasons (CoG raise) rather than dynamic absorber reasons.
