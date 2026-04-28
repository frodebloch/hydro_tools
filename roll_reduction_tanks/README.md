# roll_reduction_tanks

Time-domain prototype simulator of several roll-reduction tank types coupled to a
1-DOF roll vessel. Wave excitation is derived from pdstrip RAO data.

This is a Python prototype for an eventual C++ implementation that will live next to
`brucon::simulator::VesselSimulator` and be **loosely coupled** to it: the tank module
exposes forces/moments via an abstract interface, and the vessel applies them as
external loads with no compile-time knowledge of the tank class.

## Tank models

| Model                              | Module                  | Notes                                                                                              |
| ---------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------- |
| Open-top passive U-tube            | `tanks/utube_open.py`   | Cleaned re-implementation of the Winden (KTH 2009) formulation currently used in brucon           |
| Air-valve U-tube (passive + active) | `tanks/utube_air.py`    | Sealed chambers with a connecting valve; isothermal gas dynamics; controller modulates valve area |
| Free-surface tank                  | `tanks/free_surface.py` | Lloyd-style equivalent SDOF for a partially-filled rectangular tank                                |
| Tuned-mass damper (TMD)            | `tanks/tuned_mass_damper.py` | Canonical SDOF baseline; Den Hartog optimal-tuning helper                                          |

## Quick start

```bash
cd ~/src/hydro_tools/roll_reduction_tanks
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pytest
python examples/csov_passive_utube.py
```

Examples that produce PNGs in `examples/output/`:

```
csov_passive_utube.py        # bare vessel vs. open U-tube, time history
csov_air_valve_compare.py    # bare / open / closed / freq-tracking modes
freq_track_sweep.py          # sweeps T_w to validate freq-tracking controller
csov_gm_sweep.py             # bare-vessel response at several GM values
rao_with_without_tank.py     # roll RAO sweep with / without tank (slow!)
compare_tank_types.py        # all four tank types side-by-side
tuning_sweep_open_utube.py   # roll amplitude vs. duct height
```

All examples use a realistic CSOV operating point:

* `GM = 3.0 m`     -> roll natural period `T_n ≈ 11.4 s`
* `T_wave ∈ [8, 12] s`  -- typical North Sea operational wind sea
* `zeta_a = 1 m`   -- regular wave amplitude
* Beam seas (heading 90 deg), zero forward speed

The pdstrip RAO file was generated at `GM = 1.787 m`; the simulator runs at
the higher operational GM by exploiting the GM-decoupling argument
(see § 2.1 below).

**Tank geometry**: The headline U-tube is sized to fit inside the CSOV's
22.4 m beam:
```
utube_duct_width      = 16 m       resevoir_duct_width = 2 m  (each leg)
                    -> total physical width = 16 + 2*2 = 20 m
tank_thickness        =  5 m
tank_height           =  5 m       undisturbed_fluid_h = 2.5 m
utube_duct_height     =  0.6 m     (tuned for T_tau ~= 11.4 s)
duct_below_waterline  =  6.5 m     (duct at keel for CSOV draft T = 6.5 m)
tank_wall_friction    =  0.05      (zeta ~ 0.07)
```
Fluid mass ≈ 98 t (~0.9 % of vessel displacement).

**Headline numbers** (`csov_passive_utube.py`, regular beam seas at the
roll natural period `T_n = 11.4 s`, wave amplitude 1 m):

```
bare vessel:           6.39 deg roll amplitude
with passive U-tube:   1.96 deg roll amplitude   (69 % reduction)
```

This sits in the literature range for properly tuned passive anti-roll
tanks at resonance (60–90 %). Off-resonance (e.g. `T_w = 10 s` against
the same `T_n = 11.4 s` tank) the static free-surface effect dominates
and the tank *amplifies* roll by ~40 %; this is also physically
correct and a known limitation of single-period tuned passive tanks.
The air-valve U-tube can be retuned actively to track wave period and
restore reduction across a band — see ``examples/csov_air_valve_compare.py``.

---

## Notation

| Symbol            | Meaning                                                  | Units      |
| ----------------- | -------------------------------------------------------- | ---------- |
| `phi`             | Vessel roll angle (positive = starboard down)            | rad        |
| `phi_dot`         | Roll rate                                                | rad/s      |
| `phi_ddot`        | Roll acceleration                                        | rad/s^2    |
| `I44`             | Vessel rigid-body roll inertia about COG                 | kg·m^2     |
| `a44`             | Added inertia in roll                                    | kg·m^2     |
| `b44`             | Linear roll damping                                      | N·m·s/rad  |
| `c44`             | Hydrostatic restoring stiffness in roll                  | N·m/rad    |
| `GM`              | Metacentric height                                       | m          |
| `Δ` (`disp`)      | Displaced volume                                         | m^3        |
| `ρ`, `g`          | Sea-water density, gravity                               | kg/m^3, m/s^2 |
| `M_wave(t)`       | Wave-exciting roll moment                                | N·m        |
| `M_tank(t)`       | Tank-on-vessel roll moment                               | N·m        |
| `tau`             | U-tube fluid angle                                       | rad        |
| `omega_e`         | Wave encounter frequency                                 | rad/s      |
| `Phi(omega)`      | Complex roll RAO at wave frequency `omega`               | rad/m      |

---

## 1. Vessel EOM

The vessel is a 1-DOF linear roll oscillator with the linearised hydrostatic
restoring written explicitly in terms of `GM`:

```
(I44 + a44) * phi_ddot  +  b44_lin * phi_dot  +  b44_quad * phi_dot * |phi_dot|
        +  rho * g * Δ * GM * phi   =   M_ext(t)
```

with

```
c44  =  rho * g * Δ * GM .
```

`M_ext(t) = M_wave(t) + M_tank(t)`. The vessel module
(`vessel.py:RollVessel`) is unaware of any tank class — it consumes a scalar
external moment supplied by the coupler.

The natural frequency and period are

```
omega_n  =  sqrt(c44 / (I44 + a44))            T_n  =  2π / omega_n .
```

For the CSOV with `GM = 1.787 m` (the value used in the bundled pdstrip run),
the simulator gives `T_n = 14.73 s`. Time integration uses classical
fourth-order Runge–Kutta (`RollVessel.step_rk4`).

---

## 2. Wave-exciting moment from pdstrip RAOs

Pdstrip is a strip-theory linear seakeeping code that outputs complex
response amplitude operators (RAOs) `Phi(omega, beta, U)` for each motion DOF
at each wave frequency `omega`, heading `beta` and ship speed `U`. For roll,
the format is "Rotation/k": the dat file stores `Phi/k`, where
`k = omega^2 / g` is the deep-water wavenumber. This must be multiplied by `k`
to recover the physical RAO `Phi` in rad/m. (Verified against
`brucon::libs/dp/vessel_model/wave_response.cpp` line 271.)

Once the linear RAO is known we can recover the wave-exciting moment via the
**inverse linear roll EOM**. With wave amplitude `zeta_a` and complex roll
amplitude `Phi` (per unit wave amplitude), the steady-state phasor balance is

```
[ -(I44 + a44) * omega_e^2 + i * b44 * omega_e + c44 ] * Phi * zeta_a
        =  M_wave(omega) * zeta_a
```

so

```
M_wave(omega) / zeta_a  =  H(omega_e) * Phi(omega)
H(omega_e)              =  -(I44 + a44) * omega_e^2 + i * b44 * omega_e + c44
```

`waves.py:roll_moment_from_pdstrip` returns the time-domain reconstruction

```
M_wave(t)  =  Re{ M_wave_complex * exp(i * omega_e * t) } .
```

Encounter frequency uses pdstrip's heading convention,

```
omega_e  =  omega  -  k * U * cos(beta)
```

(beam seas: `cos(90°) = 0`, so `omega_e = omega`).

### 2.1 GM-decoupling (the key trick)

`a44`, `b44` and the wave-exciting force are **independent of GM** for a given
hull at a given draft — only the hydrostatic stiffness `c44 = ρ g Δ GM` changes.
Pdstrip computes `Phi(omega)` for one specific value of `GM`
(call it `GM_pdstrip`); the back-calculated `M_wave(t)` is therefore tied to
that GM. But once we have `M_wave(t)`, we can plug it into the time-domain
solver at *any* GM by simply changing `c44` in the vessel config — the wave
forcing remains correct.

This is the GM-sweep argument: one pdstrip run gives wave forcing usable for
a whole family of loading conditions. The simulator stores `GM_pdstrip`
separately from the simulator's own `GM`, and uses the right one in the right
place. Verified by `tests/test_waves.py::test_gm_decoupling` against the
closed-form linear RAO at a different GM, agreement < 0.5%.

### 2.2 What if `a44`, `b44` aren't in pdstrip output?

The CSOV `pdstrip.out` doesn't print per-frequency added mass or damping or
the natural roll frequency. The simulator therefore *assumes* default values
(`a44 = 0.20 * I44`, `b44 = 5%` of critical) — and these must cancel in any
roundtrip *at the same GM*. The roundtrip cancellation is the correctness
test (`test_gm_decoupling`); the assumed values only become physically
relevant when the simulator runs at a *different* GM, where they enter the
new `H(omega_e)` linearly.

---

## 3. Open-top passive U-tube (`utube_open.py`)

**Status:** This module re-implements the U-tube model used in
`brucon::simulator::TankRollModel` from first principles, following
Holden, Perez & Fossen (2011) *A Lagrangian approach to nonlinear
modeling of anti-roll tanks*, Ocean Engineering 38, 341-359 — a
derivation experimentally validated against 44 model-scale tests.
Brucon's coefficient algebra is mostly correct in form but contains:
(i) a hidden vertical-reference ambiguity (`utube_datum_to_cog`
literally implies COG-referenced; the Lagrangian derivation references
the roll axis ≈ waterline), (ii) a sign error on the gravitational
cross-coupling on the tank EOM RHS, and (iii) a Lagrangian
non-reciprocity in the tank-on-hull moment that uses the self-inertia
`a_tau` instead of the cross-inertia `a_phi`. We resolve all three.

The full step-by-step derivation, mapping from Holden's variables to
ours, and the comparison with brucon are written up in
[`docs/utube_derivation.md`](docs/utube_derivation.md) Part I (the
authoritative section). Parts II and Appendix A in that document
record earlier (superseded) derivation attempts and are kept as a
reasoning trail.

References: Holden, Perez & Fossen (2011) (primary); Bertram (2012)
*Practical Ship Hydrodynamics* §4.4 eqs. 4.122-4.123 (corroboration
of `c_phi = Q g` and `omega_tau^2 = 2g/(A_0 ∫ds/A)`); Lloyd (1989,
1998) *Seakeeping: Ship Behaviour in Rough Weather* §13 (linear
treatment, contains a sign typo identified by Holden footnote 2);
Faltinsen & Timokha (2009) *Sloshing* eq. 3.59 (linear model identical
to Holden's modulo `It(0)`); Frahm (1911) (original U-tube concept).

### 3.1 Geometry and sign convention

Two vertical reservoirs of height `H_t`, width `b_r`, joined at the
bottom by a horizontal duct of height `h_d`, length `w_d`. Centreline
distance between reservoirs: `W = w_d + b_r`. Lateral (along-ship)
thickness `t`. Undisturbed free-surface height above the duct datum:
`h_0`.

Vertical reference is the **waterline**. Body frame is z-down (matches
brucon and pdstrip / SNAME), so

* `duct_below_waterline > 0` ⇒ duct is *below* the waterline (e.g. in
  the hull, near the keel).
* `duct_below_waterline < 0` ⇒ duct is *above* the waterline (e.g. on
  the deck or in the superstructure).

We deliberately depart from brucon's `utube_datum_to_cog` because that
name implies a COG datum while the underlying derivation references the
roll axis Cf ≈ waterline; with `KG ≠ 0` the two differ by `Q·KG` in
`a_phi`, which is a real (silent) bias.

Sign convention for the fluid coordinate `tau`:

* `tau > 0` ⇔ starboard (positive y) reservoir sits *higher*
* `phi > 0` ⇔ vessel rolled starboard-down

Quasi-statically, gravity in the body frame pushes fluid toward the lee
side, so a constant `phi > 0` produces `tau > 0` of the same sign.
This is the destabilising free-surface effect.

### 3.2 Coefficients (Holden 2011, in our notation)

```
Q       =  rho * b_r * W^2 * t / 2

# Cross-coupling (reciprocal: both equations use a_phi)
a_phi   =  Q * (z_d + h_0)        # Holden eq. 14, with z_d = duct_below_waterline
c_phi   =  Q * g                  # Bertram eq. 4.122
a_y     = -Q                      # sway coupling
a_psi   = -Q * x_T                # yaw coupling

# Self
a_tau   =  Q * b_r * (W/(2 h_d) + h_0/b_r)   # Bertram 4.123
b_tau   =  Q * mu * b_r * (W/(2 h_d^2) + h_0/b_r)
c_tau   =  Q * g                              # Bertram 4.122
```

`mu` is a non-dimensional wall-friction coefficient. Brucon's Winden
test uses `mu = 0.1`, which makes the tank fluid **strongly overdamped**
(`zeta ≈ 1.87`); for time-domain studies use `mu ∈ [0.01, 0.1]`.

### 3.3 Equations of motion

**Tank** (Holden eq. 41):

```
a_tau * tau_ddot + b_tau * tau_dot + c_tau * tau
       =  +a_phi * phi_ddot  +  c_phi * phi
          - a_y * v_dot      -  a_psi * r_dot
```

`v_dot`, `r_dot` are sway and yaw acceleration (zero in 1-DOF roll).

**Tank moment on hull** (Holden eq. 36/37, mapped via `tau = -2 ξ / W`):

```
M_tank_on_phi  =  +a_phi * tau_ddot  +  c_phi * tau
```

The `+c_phi * tau` term is the static free-surface effect
(destabilising); the `+a_phi * tau_ddot` term provides the resonant
absorber action when `tau_ddot` is in phase opposition to `phi` near
the tank natural frequency.

Natural frequency (Bertram 4.123 / Holden after eq. 41):

```
omega_tau^2  =  c_tau / a_tau  =  g / (h_0 + W b_r / (2 h_d))
```

Tank state is clamped to `tau_max = 2 (H_t - h_0) / W`.

### 3.4 Comparison with `brucon::simulator::TankRollModel`

| # | Item                           | Brucon                         | Here (per Holden)             |
|---|--------------------------------|--------------------------------|-------------------------------|
| 1 | Cross-inertia formula          | `a_phi = Q*(z_d + h_0)`        | `a_phi = Q*(z_d + h_0)` ✓     |
| 2 | Vertical reference for `z_d`   | named "to COG" (ambiguous)     | explicit `duct_below_waterline` (Cf-referenced) |
| 3 | Sign of `c_phi*phi` on tank RHS | `-c_phi*phi`                   | `+c_phi*phi` (Holden eq. 41)  |
| 4 | Tank-on-hull moment            | uses `a_tau` (self-inertia)    | uses `a_phi` (cross-inertia)  |

Brucon item 1 is correct in form; item 2 is a hidden bias of magnitude
`Q·KG` in `a_phi` whenever `KG ≠ 0`. Item 3 inverts the static fluid
response (brucon's static τ has the opposite sign of `phi`, which would
mean fluid running to the *windward* side under gravity in a gentle
heel — unphysical). Item 4 breaks Lagrangian reciprocity: the cross
coefficient `∂²L / (∂phi_dot ∂tau_dot)` must be the same in both
equations. Note brucon's own `TankAngleSemiCoupled` method uses
`a_phi` correctly on the vessel side, so the bug is internal
inconsistency between two methods of the same class.

### 3.5 Practical placement

With `a_phi = Q (z_d + h_0)`, the kinematic cross-coupling vanishes
at `z_d = -h_0` (the duct sits one fluid-depth above the waterline)
and changes sign across that "dead zone" — both extreme-low (keel,
`z_d ≈ T`) and extreme-high (deep in the superstructure, `z_d ≪ -h_0`)
placements give large `|a_phi|`, but the relative sign of
`+a_phi*phi_ddot` and `+c_phi*phi` on the tank RHS is opposite for
the two cases. For typical CSOV operating points, in-hull placement
(`z_d > 0`, near the keel) is the natural choice, both for `|a_phi|`
amplitude and for the practical reasons of weight distribution and
deck space. See `examples/investigate_reduction.py` experiment 0 for a
sweep across placement.

---

## 4. Air-valve U-tube (`utube_air.py`)

Extends the open-top model with two sealed air chambers above each
reservoir, connected by a single controllable orifice. State adds two more
variables: `n1, n2` where `n_i = p_i * V_i = m_i * R * T` (the isothermal
ideal-gas "content"). The tank state vector is `[tau, tau_dot, n1, n2]`.

### 4.1 Chamber kinematics

For tilt `tau`, fluid in leg 1 rises by `(W/2) * tau` (compressing chamber 1)
and falls by the same amount in leg 2:

```
A_res  =  b_r * t                       # one reservoir cross-section
V1(tau)  =  V0  -  A_res * (W/2) * tau
V2(tau)  =  V0  +  A_res * (W/2) * tau
p_i      =  n_i / V_i                   # isothermal ideal gas
```

### 4.2 Pressure-difference forcing

The pressure difference `Δp = p1 - p2` opposes increases in `tau`, acting as
an additional spring on the fluid EOM:

```
a_tau * tau_ddot  +  b_tau * tau_dot  +  c_tau * tau  +  Δp * A_res * (W/2)
       =  +a_phi * phi_ddot  +  c_phi * phi  -  a_y * v_dot  -  a_psi * r_dot
```

### 4.3 Orifice mass flow

Subsonic, simple-form (good enough for the prototype):

```
m_dot       =  sign(Δp) * Cd * A_v * sqrt( 2 * ρ_air_avg * |Δp| )
ρ_air_avg   =  (p1 + p2) / (2 R T)
A_v         =  A_v_max * u(t)         # u in [0, 1] from the controller
n1_dot      = -m_dot * R * T
n2_dot      = +m_dot * R * T
```

### 4.4 Limit cases

* **Fully open valve** (`A_v -> infinity`): pressures equalise, `Δp -> 0`,
  the tank reduces exactly to the open-top model. Verified by
  `tests/test_utube_air_limits.py`. (In practice, *very* large `A_v` makes
  the gas-equilibration time scale so short relative to `dt` that the
  explicit RK4 becomes stiff; choose `A_v_max ≤ ~1 m^2` for typical
  marine installations.)

* **Fully closed valve** (`A_v = 0`): no gas exchange. Linearising
  `p_i = p_atm * V0 / V_i` about `tau = 0` gives an extra restoring stiffness
  ```
  Δk_gas  =  p_atm * A_res^2 * W^2 / (2 V0)
  ```
  so the closed-valve tank natural frequency is
  ```
  omega_tau,closed  =  sqrt( (c_tau + Δk_gas) / a_tau ) .
  ```
  Verified by `test_air_closed_valve_decay_period_matches_analytic`
  at small amplitude (the closed-valve gas spring is strongly nonlinear at
  large `tau`).

### 4.5 Roll moment on the hull

Same as the open-top case:

```
M_tank_on_phi  =  +a_phi * tau_ddot  +  c_phi * tau .
```

The sealed chamber pressures act on rigid chamber walls, so by Newton's
third law their net moment on the hull is zero at first order.

### 4.6 Controllers (`controllers/`)

A controller maps vessel kinematics (`phi, phi_dot, phi_ddot`) and time `t`
to a fractional opening `u in [0, 1]`. **The controller has no access to
tank state** — this matches the user's requirement that the active strategy
must be portable to a real installation where only roll signals are
available.

Provided:

* `ConstantOpening(value)` — fixed valve position.
* `FullyOpenValve`, `FullyClosedValve` — convenience subclasses.
* `FrequencyTrackingController` — estimates the roll period from
  successive sign-changes of `phi_dot` and maps it linearly to an opening
  in `[0, 1]`, low-pass-filtered with a configurable time constant.
  At `u = 0` the chambers are sealed and the trapped-gas spring
  stiffens the fluid, raising the tank natural period to `T_closed`;
  at `u = 1` the chambers vent freely and the period drops to `T_open`.
* `BangBangController` — **deprecated**. The original rule "open when
  `phi_dot * phi_ddot < 0`" reduces to `−sin(2 omega t)` for a sinusoidal
  vessel response and therefore chatters at `2 omega` with no useful
  phase selection. Empirically it is *worse* than a fully-open passive
  valve at resonance (45 % vs 68 % reduction). Use
  `FrequencyTrackingController` instead.

### 4.7 Active design philosophy and its limits

Three real-world ways to give a U-tube adaptive period control are
listed in Hoppe Marine's product literature:

1. **Variation in air damping** — sealed chambers above the reservoirs
   with a controllable orifice; the trapped gas adds spring stiffness.
   This is what `utube_air.py` models.
2. **Variation in water-duct cross-section** (Hoppe concept) — a
   mechanically variable `b_r` (or equivalently `h_d`); modulates the
   liquid-side parameters directly.
3. **Delaying the fluid flow** (INTERING concept) — pneumatic
   retardation of the liquid column.

Only mechanism (1) is currently in this codebase.

For the air-valve mechanism to give controllable retuning, the
controllable band `[T_closed, T_open]` must straddle the wave-period
range of interest. Recommended sizing:

1. Detune the U-tube geometry (small `utube_duct_height`) so that with
   the valve fully open the tank period equals the *upper* end of the
   target wave band (e.g. `T_open = 14 s`).
2. Size `chamber_volume_each` so that the closed-valve gas-spring
   stiffening lifts the period to the vessel resonance
   (`T_closed = T_n`).

For the CSOV at `T_n = 11.4 s`, this gives `utube_duct_height ≈ 0.39 m`
and `chamber_volume_each ≈ 198 m³`. The frequency-tracking controller
then automatically picks `u = 0` near resonance and ramps up to
`u = 1` at the long end of the band.

**Honest comparison at the design point** (`examples/csov_air_valve_compare.py`,
T_w = T_n = 11.4 s):

| configuration                                                      | amp [deg] | reduction |
|--------------------------------------------------------------------|-----------|-----------|
| bare vessel                                                        | 6.39      | —         |
| **passive open U-tube tuned to T_n** (`h_d = 0.6`)                 | **1.96**  | **69 %**  |
| air valve closed (`h_d = 0.39`, gas-spring restores T to 11.4 s)   | 3.22      | 50 %      |
| air valve open  (`h_d = 0.39`, T_open = 14 s, off-tuned)           | 4.59      | 28 %      |
| air valve, frequency-tracking controller                           | 3.22      | 50 %      |

The passive design wins by a wide margin at its design point. Two
findings explain this:

* **Gas-spring tuning weakens the tank.** Restoring the tank period
  to `T_n` via gas-spring stiffness rather than liquid-column geometry
  preserves `omega_tau` but reduces the liquid-side momentum coupling
  `a_tau`. Since the cancellation moment scales with `a_tau`, the
  tank's authority on the hull is smaller even at perfect tuning.
* **Static valve openings between 0 and 1 do not give intermediate
  tank periods.** With `valve_area_max = 0.5 m²` (a typical commercial
  installation), the orifice equilibrates the chamber pressures faster
  than the tank's natural period for any `u >~ 0.05`; the tank is
  effectively bistable in opening. A continuously tunable orifice
  would need `valve_area_max ~ 0.005-0.01 m²`, but then intermediate
  `u` adds gas-throttling damping that hurts more than the partial
  retuning helps.

The conclusion: **air-valve tanks are inferior to a properly tuned
passive U-tube at the passive's design point**, but they offer modest
benefit when the wave period drifts far from any single passive
tuning. This matches Hoppe's own marketing for the air-damped
variant — "the most cost effective solution applicable for vessels
with a relatively narrow band of variation in loading condition."
For genuinely adaptive retuning, the variable-duct-area mechanism (2
above) is a better physical concept and would warrant a separate tank
implementation.

### 4.x Dual-purpose anti-heel duty (not modelled)

Crossover U-tube tanks on offshore vessels often share duty as
*anti-heel* systems for crane operations: by closing the air-crossover
valve and pressurising one chamber relative to the other (via a
compressor), the fluid is forced to sit asymmetrically, generating a
controllable static moment that can offset a crane's heeling moment.

This is **not implemented** in `AirValveUtubeTank`, but the impact on
roll reduction is straightforward to reason about:

  * Closed valve raises the tank natural period from `T_open` to
    `T_closed`, which (for our active design endpoints `[8, 14] s` or
    `[7, 11] s`) moves the absorber notch *off* vessel resonance.
    During anti-heel duty, peak roll RAO at vessel `T_n` is roughly
    doubled compared with valve-open operation.
  * The fluid biased toward one side reduces the available swing
    toward the unloaded side, lowering the maximum roll amplitude the
    tank can absorb before nonlinear saturation by roughly the
    fractional offset (a 70 % static offset roughly halves the
    saturation-limited absorbing capacity).
  * Higher operating pressure (`p_atm + Delta p`) increases the
    closed-valve gas-spring stiffness linearly with mean pressure,
    pushing `T_closed` even shorter.

**Maximum heel-correction capacity for the CSOV designs**: bounded by
*reservoir geometry*, not by air pressure. With our reservoir of
`A_res = 10 m^2` and undisturbed fluid height 2.5 m, a realistic
operational fluid-swing of 70 % gives:

  * transferred fluid mass ~17.9 t;
  * static heel moment ~2.82 MN m;
  * equivalent static heel correction at `GM = 3 m`: ~0.49 deg;
  * equivalent crane payload at 20 m outreach: ~14.3 t.

This is small compared with heavy-lift wind-turbine installation
crane moments (a 600 t lift at 25 m outreach is ~150 MN m, ~50x our
U-tube's capacity), so dual-purpose use is realistic only for *light*
deck-equipment moments (small davits, supply crane, load shifts);
heavy-lift vessels typically install dedicated, larger anti-heel
tanks with pumped transfer in addition to the roll-reduction U-tube.

The choice of `V0` affects only the *speed* and *energy* of fluid
transfer, not the geometric ceiling: smaller `V0` gives a stronger
gas spring that can hold the offset against hydrostatic head without
compressor assistance; larger `V0` requires the compressor to add a
few tenths of a bar.

A natural follow-up implementation would extend `AirValveUtubeTank`
with a `chamber_pressure_differential` input (driven by a heel-angle
controller) and re-linearise the gas-spring stiffness around the
biased equilibrium `tau_eq != 0`.

### 4.y Fully active U-tube via in-duct thruster (not modelled)

A natural extension beyond the air-valve scheme is a U-tube with a
**rim-driven thruster (RDT) installed in the bottom of the duct**,
buffered by a battery or supercapacitor bank. Instead of modulating
the natural dynamics via valve control, the RDT directly drives the
fluid back and forth to produce an *optimally* phased moment that
cancels the wave-exciting moment in real time.

**Performance ceiling.** The fluid is bounded geometrically to the
same ~2.82 MN m of static moment (sec. 4.x), giving a peak tank
moment amplitude of ~2.82 MN m at any frequency. The relevant
comparison is *not* with the broadband wave-moment significant value
(7-8 MN m at our test seastates) but with the wave moment near
vessel resonance, where roll motion is actually generated:

| seastate | wave M significant (full band) | wave M significant (near resonance ±10%) | active tank covers |
| -------- | ------------------------------ | ----------------------------------------- | ------------------ |
| Hs=3 m, Tp=8.5 s (off-resonance) | 7.06 MN m | 1.74 MN m | 162 % |
| Hs=2 m, Tp=11 s (resonance)      | 5.25 MN m | 3.71 MN m |  76 % |

Off-resonance, the tank has 1.6x the moment budget needed to cancel
the resonance-band component; on-resonance, it covers 76 %. Honest
performance estimates against the irregular-seas baseline in
`csov_irregular_seakeeping.py`:

| seastate | best passive (measured) | ideal active U-tube (estimated) |
| -------- | ----------------------- | ------------------------------- |
| Hs=3 m, Tp=8.5 s | 48 % (free-surface) | ~85-90 % |
| Hs=2 m, Tp=11 s  | 54 % (TMD)          | ~70-80 % |

These are linear-theory ceilings; nonlinear free-surface effects at
the required ~1-1.7 m fluid offsets (vs nominal 2.5 m fluid height)
will degrade by 20-30 %, so a *realistic* delivered figure is more
like **70-80 % off-resonance, 60-70 % on-resonance** -- still
roughly a 1.5-2x improvement over the best passive tank, in line
with Holden, Perez & Fossen 2011's reported active-U-tube results
for similarly sized vessels.

**RDT thrust requirement.** To produce the maximum tank moment at
the wave frequency, the in-duct thruster must overcome (i) the
gravity head between the asymmetrically filled reservoirs, (ii) the
inertial reaction of the moving fluid, and (iii) duct friction. For
our CSOV geometry (`A_res = 10 m^2`, `A_duct ~ 3 m^2`, `W_duct =
16 m`):

| metric | Tp=8.5 s (off-resonance) | Tp=11 s (resonance) |
| ------ | ------------------------ | ------------------- |
| required fluid offset tau_a | 1.08 m | 1.75 m |
| peak duct velocity         | 2.66 m/s | 3.34 m/s |
| peak flow rate Q           | 8.0 m^3/s | 10.0 m^3/s |
| **peak thrust required**   | **170 kN** | **210 kN** |
| peak shaft power           | 460 kW    | 700 kW    |
| dissipated power (avg)     | 12 kW     | 24 kW     |

170-210 kN places the actuator in the *bow tunnel thruster* class --
a 1.6-1.8 m diameter rim-driven thruster (e.g. Brunvoll RDT-160 or
Rolls-Royce TT class) is a direct off-the-shelf fit and dramatically
simpler than the equivalent compressor system (which would need
8-10 m^3/s of low-pressure airflow at 20-35 kPa Δp).

**Energy buffering.** The peak shaft power is **reactive**: it
flows back and forth between fluid kinetic energy and gravitational
potential energy each half-cycle. The total swing energy per cycle is
**0.3-0.6 MJ** (~170 Wh = 0.17 kWh). A modest battery or
supercapacitor bank in the kWh class absorbs all peaks, so the ship
electrical bus only sees the **24 kW dissipated** -- *less than a
coffee machine*. For context, the CSOV's installed power is
~5-10 MW, so even the unbuffered peaks would be ~10 % of installed
power; with buffering the active-tank load is essentially invisible
to the power system.

This is the strongest reason to favour an in-duct RDT over an
air-driven scheme: the ship never sees a power spike, the actuator
hardware is already standard marine equipment, and the failure mode
is benign (loss of RDT power reverts the U-tube to a passive
open-top tank with its geometric resonance still intact).

**Caveats specific to in-tank RDTs.** (i) Cavitation: at the duct
elevation in our geometry (~6.5 m below the waterline), ambient
pressure is ~1.65 bar absolute, well above water vapour pressure;
peak suction-side pressure drops are within tolerance for moderate
loading. RDTs are intrinsically free of *tip-vortex* cavitation
because the blade tips are integral with the driven rim (no tip
clearance), which is one of the headline reasons they are favoured
for low-noise / low-cavitation applications -- so the only relevant
cavitation mode here is sheet/bubble cavitation on the blade
suction surface, easily handled by conservative blade loading at
these flow speeds. (ii) Two-phase ingestion: if the free surfaces
in the reservoirs slosh hard and entrain air, that air gets pulled
through the RDT and degrades thrust unpredictably; the deeply
submerged duct geometry (`duct_below_waterline = 6.5 m`) helps.
(iii) Forward-feed control: the controller needs phase-accurate
wave-moment prediction to within ~30 deg; a `phi_dot`-only scheme
will not suffice and a wave-radar or pressure-sensor feed-forward
path is needed for the headline numbers above. Without forward
feed, expect 15-25 % loss of the theoretical reduction.

A natural follow-up implementation would be a new `RDTUtubeTank`
class (or extension of `OpenUtubeTank`) with a
`thrust_command_func: Callable[[VesselKin, TankState, t], float]`
input mapped onto a duct pressure differential `dp = F / A_duct`,
plus an `RDTController` that takes a wave-moment estimate and solves
for the thrust command needed to produce `M_tank = -M_wave_estimate`
at the current state.

---

## 5. Free-surface tank (`free_surface.py`)

Lloyd-style equivalent SDOF for a partially-filled rectangular tank. The
lowest sloshing mode of a tank of length `L` (along beam) and depth `h` has

```
omega_n^2  =  (π g / L) * tanh(π h / L)
```

We model the mode as a tuned-mass-damper: equivalent sloshing mass `m_eq`
moving laterally by `q (m)` at vertical lever `h_arm = z_tank - z_cog`:

```
m_eq * q_ddot  +  b_eq * q_dot  +  k_eq * q  =  -m_eq * h_arm * phi_ddot
M_tank_on_phi                                  =  -m_eq * h_arm * q_ddot
k_eq                                           =  m_eq * omega_n^2
```

Equivalent mass from Faltinsen 1990 eq. 5.46:

```
m_eq  =  (8 / π^3) * tanh(π h / L) * m_fluid .
```

For a deep tank (`π h / L → ∞`), `m_eq → (8/π³) m_fluid ≈ 0.258 m_fluid`
— at most ~26 % of the fluid mass mobilises in the lowest sloshing
mode. For shallow geometries (e.g. when `L` is constrained by the
beam, forcing low `h/L`), `m_eq` drops below 10 % of the fluid mass.

This places a fundamental ceiling on free-surface tank performance:
unlike the U-tube, which mobilises essentially all of its fluid as
effective inertia, a single-bay rectangular free-surface tank can only
ever apply a fraction of its fluid mass to the cancellation moment.
For the CSOV at `T_n = 11.4 s`, fitting a tank inside the 22.4 m beam
forces `L ≈ 21.7 m, h ≈ 1.5 m`, giving `m_eq ≈ 7 % m_fluid` and a
realistic resonance reduction of only ~6 % even at `h_arm = 6 m`.
Multi-bay arrangements (segmented along the beam, each bay tuned)
restore some performance at the cost of structural complexity, but
are not modelled here.

State vector: `[q, q_dot]`.

---

## 6. Tuned-mass damper (`tuned_mass_damper.py`)

The canonical first-order representation of any passive resonant
absorber. A point mass `m_t` slides laterally in a guide rail at lever
`h_arm = z_mount - z_cog`, restrained by a linear spring `k_t` and
dashpot `b_t`. **No gravity coupling** — the mass moves horizontally so
gravity is perpendicular to its motion at small roll angle.

```
omega_n  =  sqrt(k_t / m_t)
b_t      =  2 * zeta * m_t * omega_n

m_t * x_ddot  +  b_t * x_dot  +  k_t * x   =  -m_t * h_arm * phi_ddot
M_tank_on_phi                              =  -m_t * h_arm * x_ddot
```

Why "equivalent": every passive anti-roll device reduces, at first
order, to a TMD with a specific `(m_t, omega_n, zeta, h_arm)`:

* **Free-surface tank** — the lowest sloshing mode is a TMD with
  `m_t = m_eq = (8/π³) tanh(πh/L) m_fluid`. Already implemented as
  `FreeSurfaceTank` (the two classes share their EOM structure).
* **Open U-tube** — the linearised liquid column gives
  `+a_φ τ̈` on the hull and matches the TMD's `-m_t h_arm x_ddot`
  after a sign-and-units mapping. The U-tube has an additional
  gravity-driven cross-coupling `+c_φ τ` that the bare TMD lacks; this
  could be added as an optional parameter but is omitted to keep the
  TMD a clean Den-Hartog SDOF baseline.
* **Air-valve U-tube** — same as the U-tube with an extra gas-spring
  stiffness on the fluid coordinate.

### 6.1 Den Hartog optimal tuning

For a rotational primary system of total inertia `I44_total`
(including added mass) and natural frequency `omega_p`, the equivalent
rotational mass ratio is `mu = m_t * h_arm² / I44_total`. The optimal
absorber tuning that minimises the peak primary response (Den Hartog
1934, "fixed-point" theorem) is

```
omega_t / omega_p   =  1 / (1 + mu)
zeta_opt            =  sqrt( 3 mu / (8 (1 + mu)^3) )
```

Independent of primary damping and excitation. For `mu = 0.05` this
gives `omega_t/omega_p = 0.952` and `zeta_opt = 0.127`. The helper
function `den_hartog_optimal()` in `tuned_mass_damper.py` returns the
pair directly.

### 6.2 Use as upper-bound benchmark

A real tank with effective mass ratio `mu` cannot outperform an
optimally tuned TMD with the same `mu` (this is what "first-order
equivalent" means). In `examples/compare_tank_types.py` the U-tube
gives 1.96° and the optimally-tuned TMD with `mu = 5%` gives 1.85° at
resonance — a 6 % gap, confirming the U-tube is well-tuned and within
the theoretical performance envelope.

A previous iteration of this package contained a `PendulumTank`
modelling a literal swinging gravity pendulum hanging inside the
ship. That class was misleadingly named "equivalent": it was *not*
equivalent to any tank (it had a `+g·φ` gravity-coupling term that no
sliding-mass tank has, and its natural-period requirement
`L_p = g/omega_n²` forces a physically unrealistic rod length for
typical ship roll periods). It has been replaced by this TMD.

---

## 7. Coupling architecture (`coupling.py`)

```
        +-------------+              +-----------------+
        |             |  M_tank →    |                 |
        |  Tank(s)    +───────────── │  Roll vessel    |
        |  (state)    | ← v_kin      |  (state)        |
        +-------------+              +-----------------+
              ↑                              ↑
              │ vessel kinematics            │  M_wave
              └──── controller ──────────────┘
```

**Explicit Jacobi**: at each macro step the vessel and the tanks are
advanced *in parallel* over `[t, t+dt]`, each integrator seeing the other
subsystem's state frozen at the start of the step.

```
def step(t, dt):
    v_kin0       = vessel.kinematics()
    M_tank       = sum(tank.forces(v_kin0)["roll"] for tank in tanks)
    vessel.step_rk4(lambda tt: M_wave(tt) + M_tank, t, dt)   # M_tank held constant
    for tank in tanks:
        tank.step_rk4(lambda tt: v_kin0, t, dt)              # v_kin held constant
```

This is the simplest stable scheme that preserves the loose coupling
(vessel and tank know nothing about each other's internal state layouts)
and is what the C++ implementation will mirror — the vessel will receive
`M_tank` as just another external load via its existing
`apply_external_moment(...)` interface.

For the free-surface and TMD tanks, whose `forces()` depends on the
*current* `phi_ddot`, we recompute that derivative inside `forces()` from
the supplied `vessel_kin` rather than caching the previous step's value
(otherwise the one-step lag destabilises the resonant feedback loop).

### 7.x Multi-tank combinations (not yet exercised)

`CoupledSystem` accepts a `tanks: list[Tank]` argument; nothing in the
implementation restricts it to a single tank. The tanks' contributions
to the vessel roll moment are simply summed each step, and each tank
is advanced independently against the same vessel kinematics.

The `csov_irregular_seakeeping.py` results suggest a non-obvious design
opportunity:

  * the **free-surface tank** is uniquely able to *boost* roll
    stiffness (via the `dc44_extra` static-surface-tilt term, sec. 5),
    which is the only mechanism among the modelled devices that
    improves response *below* vessel resonance;
  * the **U-tube** (or TMD) provides the deep absorber notch *near*
    the (effective) vessel resonance.

A `[free-surface + U-tube]` combination should therefore plausibly
beat any single-device solution: the free-surface re-tunes the
vessel's effective period upward and shrinks sub-resonance response,
while the U-tube — *tuned to the new effective period rather than the
bare-vessel `T_n`* — kills the residual resonant peak. Real offshore
vessels often install several tanks, and while the explicit reasons
are usually redundancy, variable loading, and asymmetric placement,
the dynamics suggest a genuine performance benefit on top of those
practical drivers. A worked combination example is a natural
follow-up; the existing infrastructure supports it directly.

---

## 8. C++-port mapping

| Python                                       | C++ target                                          |
| -------------------------------------------- | --------------------------------------------------- |
| `RollVessel`                                 | `brucon::simulator::VesselSimulator` (existing)     |
| `AbstractTank`                               | New header: `brucon::simulator::IRollReductionTank` |
| `OpenUtubeTank`                              | Replace `brucon::simulator::TankRollModel`          |
| `AirValveUtubeTank`                          | New: `brucon::simulator::AirValveUtubeTank`         |
| `FreeSurfaceTank`, `TunedMassDamperTank`     | Optional, derived from `IRollReductionTank`         |
| `AbstractValveController`                    | New: `brucon::simulator::IValveController`          |
| `BangBangController`, `ConstantOpening`      | Concrete controllers                                |
| `CoupledSystem.step`                         | Vessel-simulator main-loop addition: call           |
|                                              | `tank.forces(...)` and apply via                    |
|                                              | `apply_external_moment(...)`; advance tanks via     |
|                                              | their own `step_rk4(...)`                           |
| `pdstrip_io.load_pdstrip_dat / .inp`         | Already in `brucon::dp::vessel_model` — reuse       |
| `waves.roll_moment_from_pdstrip`             | New helper, or done in pre-processing               |

The C++ classes should mirror the Python state-vector layouts and
coefficient algebra exactly so the algebraic regression tests in
`tests/test_utube_open.py` (which pin Holden 2011 / Bertram 4.122-4.123
identities and reciprocity) can be re-used as regression tests for the
ported C++ code.

---

## 9. Data

`data/csov/` contains a CSOV pdstrip dataset (`csov_pdstrip.dat`,
`csov_pdstrip.inp`) plus a `README.md` with provenance and a parameter
table. Sources:

* `.dat` — `~/src/brucon/libs/dp/vessel_model/test/config/csov_pdstrip.dat`
* `.inp` — `~/src/pdstrip/vard_985/pdstrip.inp`

Key derived numbers:

```
m            =  11_119_698 kg
LCG          =  -1.6765 m       KG = 2.5 m
Ixx/m        =  80.3            -> I44 = 8.929e8 kg·m^2
ρ            =  1025 kg/m^3     g = 9.81 m/s^2
Δ            =  10_848 m^3      GM_pdstrip = 1.787 m
T draft      =  6.5 m           B beam = 22.4 m   L ≈ 108.65 m
c44_pdstrip  =  1.949e8 N·m/rad     T_roll(pdstrip) ≈ 14.7 s
c44(GM=3.0)  =  3.272e8 N·m/rad     T_roll(GM=3.0)  ≈ 11.4 s   (operational)
```

The bundled examples run at the operational `GM = 3.0 m`, with the wave
moment back-calculated from the `GM = 1.787 m` pdstrip RAO via the
GM-decoupling argument (§ 2.1).
