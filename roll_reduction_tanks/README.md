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
| **peak fluid thrust required**   | **170 kN** | **210 kN** |
| peak useful hydraulic power | 460 kW    | 700 kW    |

These are the **fluid-on-fluid** thrust and **useful hydraulic
power** numbers — what the propeller has to deliver to the working
fluid in the duct, not what the customer puts on the bus.

**RDT hardware sizing for that thrust.** A rim-driven thruster in a
tank duct delivers roughly **80 kN of fluid thrust per 1000 kW of
shaft power** for a 1.8 m diameter unit, after lumping all
inefficiencies into a single ratio (see footnote on loss
decomposition below). Achieving the **170-210 kN** thrust ceiling
above therefore demands **~2.1-2.6 MW** of shaft power on a
~2.0-2.2 m diameter RDT — substantially larger than a routine bow-
tunnel thruster, and a real cost driver for the customer.

> *Where the losses go.* A single lumped efficiency hides three
> distinct loss mechanisms: **rim friction** (viscous shear in the
> few-mm magnetic-circuit gap between the rotating impeller rim and
> the stator embedded in the duct wall — typically 30-35 % of shaft
> power, present *whenever the rotor spins* regardless of useful
> thrust), **propeller losses** (induced + profile drag on the
> blades, ~30-40 %), and **duct entry/exit losses** (~5-10 %). The
> rim contribution is the dominant *standby* loss and the main
> reason an oversized RDT is wasteful: a 2x larger unit running at
> 50 % duty cycle still pays the rim tax all the time. Sizing the
> RDT close to the actual peak thrust requirement (rather than with
> wide margin) is therefore preferred.

**Vendor "system thrust" caveat.** Manufacturer data sheets for
tunnel thrusters quote *system* thrust including hull-suction
contributions of order +35-50 % (which arise from the asymmetric
low-pressure zone on the suction side of an open hull tunnel). Our
RDT sits in a *closed* tank duct with no hull-side surface — there
is no equivalent suction reaction. **A vendor-quoted 200 kN tunnel
thruster will deliver only ~120-130 kN of fluid thrust in our
application**, and even that drops to ~80 kN once realistic rim and
duct losses are subtracted at the unsteady duty cycle of active
roll control. Sizing must be done from the **propeller momentum-
theory** thrust at the rated shaft power, not from the catalogue
system-thrust figure.

**Energy balance.** The peak useful hydraulic power (~460-700 kW)
is mostly **reactive**: it cycles between fluid kinetic energy and
gravitational potential energy each half-period. With a 4-quadrant
regenerative drive the *net* electrical demand from the bus is
much smaller — perhaps 150-250 kW continuous in our standard
operating point (see sec. 4.z for measured numbers from the
observer-based controller). However, **rim friction takes its
30-35 % cut every cycle and is not recoverable**: of the 1000 kW
shaft input, ~300-350 kW are continuously dissipated as heat into
the working fluid even with perfect regeneration on the propeller
side. The customer-visible electrical load is therefore on the
order of **300-500 kW continuous** in active operation — small
compared to a CSOV's installed power (~5-10 MW) but *not* "less
than a coffee machine".

This is still the strongest reason to favour an in-duct RDT over
an air-driven scheme: the actuator hardware is standard marine
equipment, the working fluid is incompressible (fast control
response), and the failure mode is recoverable provided the RDT is
designed to *freewheel* on power loss (see caveat (iv) below).

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
(iv) **Stopped-RDT blockage / failure mode.** With realistic blade
solidity ~60 % (typical for a thrust-class RDT), a *locked* rotor
in the duct presents an orifice-style flow blockage with loss
coefficient `K ~ 25-80` referenced to the duct dynamic pressure
(textbook range for ~60 % blockage with sharp- to rounded-edge
blades). For our CSOV duct geometry this is **~30-100x the existing
wall friction**, giving equivalent linear damping ratios of
**80-160 % at typical operating amplitudes** -- i.e. the tank
becomes catastrophically overdamped, the fluid barely moves, and
roll reduction collapses to a few per cent (and may be net negative
in some seastates because the trapped-fluid inertia still adds to
the vessel roll inertia without contributing absorption). The tank
does **not** revert to a useful passive U-tube on a stopped RDT,
contrary to what one might naively assume. The RDT must therefore
be designed to *freewheel* on power loss (drive set to torque-free
mode on fault, or mechanical clutch) so the impeller spins with
the flow and contributes only rim drag (`K ~ 1-3`); this is
standard capability for modern marine RDT drives but must be
specified explicitly. Alternative mitigations are a parallel
bypass duct that opens on fault (doubles hardware cost), feathering
blades (adds mechanical complexity), or simply accepting that the
active U-tube has no passive fallback and providing actuator
redundancy at the system level (two parallel RDTs).
(v) **Air entrainment** is more dangerous than for the equivalent
ducted ship thruster: the closed tank traps any air bubble that
gets sucked through the impeller, so successive cycles
re-encounter it until it migrates out of the duct path. Aerated
working fluid degrades the impeller thrust unpredictably and can
trip cavitation. The deeply submerged duct geometry (`duct_below_
waterline = 6.5 m` ⇒ ~6 m of fluid head above the duct) suppresses
free-surface entrainment in normal operation, but rough operation
or partial fill conditions need explicit consideration.

(vi) **Continuous rim heating of the working fluid.** With ~30-35 %
of shaft power dissipated continuously by rim friction (see "Where
the losses go" footnote above), the active RDT pumps **~300-400 kW
of heat directly into the tank water** during steady active
operation. For our CSOV tank geometry, the working fluid mass is
~100 tonnes (50 m³ × 2 reservoirs + ~50 m³ duct fluid), so the
adiabatic temperature rise is

    dT/dt ≈ P_dissipated / (m * c_p)
          = 350 kW / (100 t × 4.186 kJ/kg/K)
          ≈ 0.83 K / minute  =  3.0 K / hour

For a 24/7 station-keeping operation this gives a **20-30 K
temperature rise after a typical 6-10 h shift** without active
cooling. The tank-to-bilge wall conduction is small (limited
contact area and low thermal conductivity of the surrounding ship
structure), and air convection from an open-top tank's free
surface dissipates only ~5-10 kW at typical CSOV deck temperatures
— neither is a real heat sink at this dissipation rate. **An
active seawater-loop heat exchanger sized for ~300-400 kW is
therefore essentially mandatory** for any active-RDT installation
intended for continuous duty. This is a small piece of standard
marine equipment (similar to a main-engine cooler) but it adds
piping complexity and a sea-chest connection that the customer
must plan for. Closed (air-valve) tank variants are even worse
because they have no free-surface evaporative cooling at all.

(vii) **Hardware sizing margin trade-off.** Because rim friction
is a *standby* loss (proportional to rotational speed, not
thrust), oversizing the RDT for thrust margin is doubly
penalised: not only is the unit more expensive, it also
dissipates more standby power whenever it spins. The optimal
sizing puts the rated peak thrust close to the actual M_wave
cancellation requirement (~210 kN here) without large excess —
opposite to the conventional naval-architecture instinct of "spec
30 % above the nominal load".

Implemented as `tanks/utube_rdt.py` (`RDTUtubeTank`) with two
reference controllers:

  * `InverseDynamicsRDTController` — perfect-`M_wave` inversion (the
    theoretical ceiling; used as a benchmark only).
  * `StateFeedbackRDTController` — PD on `(phi, phi_dot)`, the honest
    signals-only baseline.

Verified at the operating point (Hs=3 m, Tp=8.5 s) in
`examples/csov_irregular_seakeeping_with_rdt.py`. At the legacy
F_max = 200 kN well-sized point (see sec. 4.z), ideal-RDT delivers
~71 % roll reduction in beam seas (matching the predicted
70-80 % ceiling) and PD-RDT delivers ~58 %. At the realistic
F_max = 80 kN single-RDT default (per the sizing analysis above),
the active controllers are authority-starved — see the sec. 4.z
heading sweep for the corrected picture.

### 4.z Wave-moment estimation: Luenberger observer with Sælid resonator

The "ideal" `InverseDynamicsRDTController` above assumes perfect
real-time knowledge of `M_wave(t)`. That is *not* something a real
ship can provide from rate-gyro signals alone — differentiating roll
to recover `phi_ddot` and inverting the EOM gives a causal estimate
that is hopelessly late and contaminated by the very `M_tank` the
controller is choosing.

There are two ways to close that gap on a real vessel:

1. **Forward-looking wave radar** (X-band, e.g. Miros WaveX). Gives
   30-90 s preview of the incoming wave field; the only path to the
   `~70 %` performance ceiling. Expensive and rare outside high-end
   heave-compensated cranes.

2. **Model-based observer driven by inertial measurements.** The
   standard tool for DP wave filtering since Sælid (1983), now in
   Fossen Ch. 8 / Ch. 11. Reconstructs the *dominant-frequency
   component* of the wave-induced motion from `phi(t)` (and
   optionally `phi_dot`) by augmenting the plant with a 2nd-order
   linear oscillator at the encounter frequency `omega_e` and
   estimating its amplitude/phase from the data.

For the prototype we go with option 2 in its simplest form: a
**Luenberger observer with pole placement**, no Kalman filter
machinery. Justification:

- The augmented plant is LTI for fixed `omega_e` (vessel roll EOM is
  linear around upright; the resonator is by construction linear).
- We have no well-characterised process / measurement noise
  covariances. Inventing `Q` and `R` for a Kalman filter buys
  nothing real on a fundamentally linear system.
- Pole placement gives transparent tuning ("this pole governs the
  roll-state error decay; this complex pair governs how aggressively
  the resonator adapts to amplitude changes") instead of opaque
  covariance trade-offs.
- `omega_e` does drift, but slowly (minutes timescale in
  station-keeping). Handle by **gain-scheduling**: design Luenberger
  gains at a few `omega_e` operating points and interpolate. Don't
  augment `omega_e` as a state — that would re-introduce
  nonlinearity (`omega_e^2` enters the resonator) and force EKF/UKF
  for nothing.

#### Augmented state

```
x = [ phi, phi_dot, eta_1, eta_2, M_bias ]^T              (5 states)

phi_dot         = phi_dot
phi_ddot        = (1/I) * ( -b*phi_dot - c*phi
                            + eta_1 + M_bias + M_tank_known )
eta_1_dot       = eta_2
eta_2_dot       = -2*zeta_w*omega_e*eta_2 - omega_e^2 * eta_1
M_bias_dot      = 0                                        (random walk)
```

- `eta_1` is the resonator output: the estimate of the *narrow-band*
  component of `M_wave` at `omega_e`. The model treats it as a free
  oscillator that the observer corrections force to track the
  actual wave-induced roll content.
- `M_bias` absorbs slow / DC components of the wave moment (heel
  from steady wind, mean drift, free-surface trim) and any
  systematic model error in `b, c`. Optional but cheap.
- The `M_tank_known` term is the controller's *commanded and
  saturated* `M_tank` (i.e., what the actuator just produced). The
  observer needs this so it doesn't blame the tank's contribution
  on `M_wave`. **No algebraic loop**: the controller computed
  `F_RDT` last step, the observer reads the resulting `M_tank` this
  step, the controller computes the next `F_RDT` from the updated
  `M_wave_hat`. ZOH on the actuator command makes this exact at the
  macro time-step.

Measurement: `y = phi` (rate-gyro integration / MRU output).
Optionally `y = [phi, phi_dot]^T` if the inertial unit publishes
both — improves observability of the `eta_2` state.

#### Pole placement recipe

| pole | role | guideline |
|---|---|---|
| `p_1, p_2` (real or complex pair) | drive `phi, phi_dot` error to zero | real part `≈ -3 * omega_n_roll`; with `omega_n_roll = 0.55 rad/s` (T_n = 11.4 s) → `≈ -1.6 rad/s` |
| `p_3, p_4` (complex pair) | resonator state correction | place near `±j*omega_e` with damping `zeta_obs ~ 0.4-0.6` (open-loop resonator has `zeta_w ~ 0.05`); i.e. `-zeta_obs*omega_e ± j*omega_e*sqrt(1-zeta_obs^2)` |
| `p_5` (real) | bias drift tracking | slow, e.g. `-0.05 rad/s` (≈ 20 s time constant) |

Sweet spot for the resonator poles: damping `zeta_obs ~ 0.5` (10x
faster decay than the open-loop resonator at `zeta_w = 0.05`), same
imaginary part. Faster than that and the observer amplifies
measurement noise into the M_wave estimate; slower and the
reconstruction undershoots in amplitude (at `zeta_obs = 0.18` the
steady-state gain is only ~0.7; at `zeta_obs = 0.5` it is ~0.9).

#### Implementation

`controllers/luenberger_wave_observer.py`:

```python
class LuenbergerWaveObserver:
    """Augmented [vessel + Sælid resonator + bias] observer.

    Designs the gain L = place(A^T, C^T, poles)^T at construction.
    """
    def __init__(self, I, b, c, omega_e, zeta_w=0.05,
                 observer_poles=None,
                 measure_phi_dot=False):
        ...

    def update(self, y, M_tank_known, dt):
        """One observer time-step (RK4 on the linear augmented plant)."""
        ...

    @property
    def M_wave_hat(self) -> float:
        return self._x[2] + self._x[4]   # eta_1 + M_bias

class ResonatorObserverRDTController(AbstractRDTController):
    """Inverse-dynamics control on the observer's M_wave estimate."""
    def __init__(self, observer: LuenbergerWaveObserver):
        self.obs = observer

    def thrust(self, vessel_kin, t):
        # Use the same algebraic inversion as InverseDynamicsRDTController
        # but with M_wave_hat = obs.M_wave_hat instead of M_wave_func(t).
        ...
```

#### Choice of `omega_e`: vessel resonance, not wave peak

Subtle but consequential: for an active anti-roll application,
`omega_e` should be set to the **vessel roll natural frequency**
`omega_n_roll`, NOT to the wave peak frequency `omega_p = 2*pi/Tp`.

Reasoning:

- The roll signal `phi(t)` is the wave-induced moment filtered through
  the vessel transfer function `H(s) = 1 / (I s^2 + b s + c)`, which
  peaks sharply at `omega_n_roll`. So most of the *observable*
  energy in `phi(t)` lives at `omega_n_roll`, regardless of where
  the wave spectrum peaks.
- The resonator achieves zero phase lag and unit gain at exactly its
  centre frequency. Centring at `omega_n_roll` aligns the
  observer's strongest response with the strongest signal content.
- Centring at `omega_p` instead causes the resonator to filter out
  the resonant roll content (which is what we most need to
  cancel!) and produces a poor `M_wave_hat` reconstruction. We
  measured a 28% reduction with `omega_e = omega_p` versus 57%
  with `omega_e = omega_n_roll` in the standard JONSWAP test case
  -- a 2x performance loss from this single tuning choice.

The lesson: the resonator centre frequency is *not* "wave
frequency", it is "frequency at which I best want to estimate the
wave moment from a roll measurement". For roll stabilisation, that
is unambiguously `omega_n_roll`.

#### Expected performance

Two regimes — well-sized actuator (the original 200 kN sketch) vs
realistic single-RDT (80 kN per the corrected sec. 4.y sizing). All
numbers from `examples/csov_irregular_seakeeping_with_rdt.py` (CSOV,
GM=3 m, T_n=11.4 s, JONSWAP Hs=3 m Tp=8.5 s).

**F_max = 200 kN, beam seas (legacy "well-sized" point):**

| controller | M_wave knowledge | phi_1/3 (% reduction) | mean net power (kW) |
|---|---|---|---|
| Bare vessel | n/a | 4.02° | — |
| PD signals only | none | 1.70° (58%) | 573 |
| **Luenberger + Sælid** | dominant component at `omega_n` | **1.72° (57%)** | **145** |
| Inverse-dynamics ideal | perfect (radar/preview) | 1.18° (71%) | 291 |

Observer-based controller delivers PD-equivalent attenuation at
~1/4 the net energy cost. The gap to the ideal (57 % vs 71 %)
reflects the broadband `M_wave` vs the single-resonator narrow-band
reconstruction; closing it requires parallel resonators or wave
radar preview.

**F_max = 80 kN (realistic single-RDT), heading sweep:**

| heading | bare | passive U | free-surface | TMD | RDT-PD | RDT-observer | RDT-ideal |
|---|---|---|---|---|---|---|---|
| 90° (beam)         | 4.02° | 2.72° (32%) | **2.09° (48%)** | 2.79° (30%) | 3.06° (24%) | 2.72° (32%) | 3.27° (19%) |
| 120° (30° off bow) | 3.93° | 2.88° (27%) | **2.21° (44%)** | 3.09° (22%) | 3.66° (7%)  | 3.82° (3%)  | 4.08° (-4%) |
| 150° (60° off bow) | 2.22° | 1.61° (27%) | **1.27° (43%)** | 1.71° (23%) | 2.33° (-5%) | 2.18° (1%)  | 2.91° (-31%) |

Three findings of substance, none of which were visible from the
single-heading 200 kN cut above:

1. **At F_max = 80 kN the passive free-surface tank dominates at
   every heading.** Free-surface holds ~43-48 % reduction across
   the sweep with no power, no controller, no actuator wear, and
   no thermal load. None of the active controllers come close.

2. **Heavy saturation flips the active-controller ranking.** In
   beam seas at 80 kN the saturation fraction is 78-87 % across
   all three RDT controllers. The "ideal" inverse-dynamics
   controller asks for the largest forces (it actually wants to
   cancel `M_wave`), saturates hardest, and the resulting
   bang-bang action at `omega_n` re-excites the lightly-damped
   resonance — at 150° heading it makes things 31 % *worse* than
   bare. The observer-based controller is gentler (only chases the
   narrow-band resonance component, not the full broadband
   forcing) and degrades to roughly neutral. PD sits in between.
   Conventional ranking only re-emerges once F_max ≥ ~150-200 kN.

3. **Off-axis headings make the active actuator irrelevant.** At
   150° the bare vessel already drops to 2.22°. The remaining
   roll content is broadband residual that no narrow-band
   resonator can grab; even the well-sized 200 kN ideal controller
   can only reach ~33 % reduction at 150° (vs 71 % in beam seas).
   The operational case for spending 1 MW of shaft on active
   stabilisation in favourable headings is essentially zero.

The engineering verdict from this sweep: **a single 1 MW / 1.8 m
RDT (~80 kN fluid thrust) is authority-starved by roughly an order
of magnitude against beam-sea Froude-Krylov roll moments on a
CSOV-class hull.** Direct counter-moment from F_max alone is
~80 kN × 8.5 m ≈ 0.7 MN·m, against a peak `M_wave` of 8-12 MN·m.
The U-tube resonance amplifies this by ~3-5×, but the available
tank moment is still well short of the forcing. The engineering
options that make active U-tube stabilisation worthwhile are
therefore narrower than first appeared:

- **Bigger actuator**: ≥2.5 MW shaft per RDT, putting F_max in the
  200-300 kN regime where the controllers behave conventionally
  and beat the passive baseline.
- **Hybrid passive + active** (sec. 7.y): free-surface handles
  broadband dissipation, active RDT handles only the resonance
  peak. Smaller actuator authority requirement because the active
  loop only needs to bend one peak of an already-attenuated
  spectrum, not cancel the full forcing.
- **A different actuator topology entirely**: a hull-mounted
  azimuth thruster delivers ~1.8× the bollard thrust per kW shaft
  versus a closed-tank RDT, for the reasons given in sec. 4.aa.
  Two pods acting as a pure couple recover the well-sized regime
  from a 2 × 1 MW installation.

#### Caveats

- **Vessel-tank coupling complicates `eta_1`.** Our vessel is near
  resonance with the wave (`T_n = 11.4 s` vs `Tp = 8.5 s`), and the
  U-tube fluid is tuned to `T_n`. So `phi(t)` contains substantial
  resonant content driven by the wave through the *coupled
  (vessel + tank)* transfer function. The observer model above is
  the *vessel-only* roll EOM; the tank moment enters as
  `M_tank_known`. This is correct as long as we feed the actual
  applied `M_tank` (not the commanded one) — the saturation-clipped
  value is what ends up acting on the hull.

- **Single-resonator narrow-band assumption.** A confused sea
  (swell + wind sea, two spectral peaks) would want two resonators
  in parallel at the dominant peaks. JONSWAP gamma=3.3 is narrow
  enough to be well-served by one.

- **`omega_e` shifts with vessel speed and heading.** Negligible
  for station-keeping CSOV; substantial in transit. Handle with
  gain scheduling on a slowly-updated `omega_e` estimate (separate
  frequency tracker, not augmented into the observer state).

### 4.aa Hull-mounted azimuth thruster as roll actuator (proposal)

The heading sweep in sec. 4.z makes a clear case that the closed-
tank RDT is **the wrong actuator topology** for a single-unit
active anti-roll installation on a CSOV-class hull. Two physical
penalties of putting the actuator inside a closed tank duct:

1. **Rim friction** in the few-mm magnetic-circuit gap of an RDT
   eats ~30-35 % of P_shaft as standby heat (sec. 4.y caveat vi).
2. **No hull-suction reaction.** Vendor "system thrust" figures
   for hull-installed tunnel and azimuthing thrusters include a
   Coanda-style suction force on the surrounding hull plating that
   adds ~20-40 % to fluid thrust alone. A closed tank duct has no
   such surface — net force on the vessel comes only via fluid
   acceleration, and the duct walls are internal to the tank.

Both penalties vanish for a **podded azimuthing thruster mounted
under the hull**:

- Conventional shaft / Z-drive losses (~5-10 %), no rim.
- Open-water propeller in a *symmetrical* nozzle (must thrust to
  port and starboard with comparable authority, so asymmetric
  Kort-style optimisation is unavailable). Useful efficiency ≈ 0.50.
- Real hull-suction reaction at the underside of the hull above the
  pod, ~1.15-1.25× depending on standoff distance.

#### Sizing

Bollard thrust for a propeller of disc area `A`, useful shaft
fraction `eta`, and hull-suction multiplier `k_s`:

```
T = (2 * rho * A * (eta * P_shaft)^2)^(1/3) * k_s
```

For `D = 1.8 m`, `P_shaft = 1.0 MW`, `eta = 0.50`, `k_s = 1.20`:
`T ≈ 145 kN`. Sizing comparison with the closed-duct RDT at the
same shaft power and diameter:

| Configuration                        | useful eff. | suction mult. | bollard thrust | direct moment (arm 8.5 m) |
|---|---|---|---|---|
| Closed-duct RDT                       | 0.35 | 1.00 | ~80 kN  | ~0.7 MN·m |
| **Symmetrical-nozzle azimuth pod**    | 0.50 | 1.20 | ~145 kN | ~1.2 MN·m |
| Asymmetric Kort, single-direction     | 0.65 | 1.40 | ~210 kN | ~1.8 MN·m |

So per MW of installed shaft power, a symmetrical-nozzle azimuth
pod buys roughly **1.8× the roll moment** of an in-tank RDT —
short of the 200 kN "well-sized" sweet spot identified in sec. 4.z
for a single unit, but well above the 80 kN regime where active
control loses to the passive free-surface tank.

#### Twin-pod couple architecture

CSOVs typically already carry 2-4 azimuthing pods for DP. Operating
**a forward and an aft pod in opposition** (transverse thrust in
opposite directions) produces:

- **Pure roll couple**: net surge and sway forces cancel.
- **No yaw moment**: the equal-and-opposite transverse forces at
  fore-aft offsets cancel about CG.
- **2× the effective force** for the same per-pod sizing: a 2 ×
  1 MW twin-pod arrangement delivers ~290 kN of effective roll
  authority, comfortably back into the well-sized regime where
  active control beats passive at every heading.
- **Existing actuator reuse**: the pods may already be installed
  for DP, so the marginal cost of adding a roll-control loop is a
  software layer, not new hardware. Power budget contention with
  DP is the real cost.

#### Operational caveats

- **Azimuth slew rate**: a wave half-period at `T_n = 11.4 s` is
  ~5.7 s. Modern azipods slew at 5-8 °/s, so a full 0° → ±90° swing
  in 5.7 s is at the edge of feasibility. Practical operating mode
  is to **hold a fixed transverse orientation and modulate thrust
  magnitude/sign**, identical to a tunnel thruster — the
  azimuthing capability is then used only for slow-timescale
  reconfiguration (e.g. heading change), not wave-cycle control.

- **Cavitation** at high transverse loading and shallow immersion
  limits sustained authority in a seaway. Needs a dedicated
  hydrodynamic check before claiming a given F_max can be held
  through a full Hs = 3 m realisation; we have not done this.

- **Roll-DP coupling**. Single-pod operation generates yaw moment
  and a net sway force that DP must compensate, costing power and
  fighting the roll loop. Twin-pod couple operation eliminates
  both. Single-pod is therefore probably non-viable for continuous
  duty; twin-pod is the only architecture worth implementing.

- **Power-supply contention**. The pods are usually sized for DP
  thrust, which is occasional and slow-timescale. Sustained roll
  modulation at wave frequency draws from the same bus and may
  saturate the generators on a smaller vessel. The energy
  bookkeeping here matters more than for the in-tank RDT case.

#### Mapping into the prototype

The loose-coupling architecture handles this naturally. The
azimuth pod does **not** need a `Tank` subclass — it has no fluid
state, no internal dynamics worth resolving at wave timescales.
The right abstraction is:

```python
class DirectRollActuator:
    """Force actuator that imposes a roll moment directly on the
    vessel. No internal state beyond the saturated command.

    forces() returns {"roll": F_command_clipped * z_arm}.
    step_rk4() updates the cached command (ZOH like the RDT).
    """
    def __init__(self, config: DirectRollActuatorConfig,
                 controller: AbstractRDTController): ...
```

All existing RDT controllers (`StateFeedbackRDTController`,
`ResonatorObserverRDTController`, `InverseDynamicsRDTController`)
**drop in unchanged**: their thrust is a force command in Newtons
and they make no assumption about the actuator topology. The only
adjustment is that the inverse-dynamics controller's algebraic
inversion becomes one-line trivial (no fluid intermediate; just
`F = -M_wave / z_arm`) and saturation is the only nonlinearity.

The hybrid architecture from sec. 7.y becomes natural: a
`FreeSurfaceTank` for broadband dissipation in parallel with a
`DirectRollActuator` for resonance-peak shaving. The active loop
sees the *attenuated* roll spectrum from the free-surface tank, so
its authority requirement is much lower than for standalone active
control — a key sizing economy.

This is a planned follow-on; not yet implemented.

### 4.bb DP thrusters as roll actuators (proposal, deployed concept)

The natural extension of sec. 4.aa is to **stop installing dedicated
roll actuators at all** and instead extract roll authority from the
DP thruster array that the vessel already carries. This is the
architecture used in the field on Voith Schneider–propelled offshore
vessels and on several DP-class platform supply vessels: a
roll-control loop runs on top of the DP allocator and biases the
thruster commands at wave frequency to produce a roll-stabilising
moment, while DP continues to service position and heading
demands at slow timescale.

Why this is attractive:

- A CSOV typically carries 4-6 azimuthing pods, each 1-2.5 MW. The
  *total installed thrust authority* is in the multi-MN range —
  vastly more than any dedicated anti-roll device could justify on
  its own.
- Marginal hardware cost is essentially zero: pods, drives, control
  electronics, and switchboard are already present for DP. Only an
  allocator software change is required.
- No new failure modes outside the DP failure-mode catalogue.

#### Two-loop architecture

DP commands act on the slow drift band (~0.001-0.05 Hz, after the
DP wave filter has stripped wave-frequency content). The roll loop
acts at wave frequency (~0.1-1 Hz). Because the two demand bands
are an order of magnitude apart, they are nominally decoupled:

```
slow loop:  position/heading error  ->  DP controller  ->  (Fx, Fy, Mz)_d,slow
fast loop:  phi observer (sec. 4.z) ->  roll controller ->  (M_roll)_d,fast
                                                                 |
                            allocator finds u (per-thruster command)
                            producing desired (Fx, Fy, Mz, M_roll)
                            subject to per-thruster saturation u_i,max
                            and priority weights w_dp >> w_roll
```

The roll-band thrust appears to the DP loop as small high-frequency
dither that its low-pass filter rejects; the DP demands appear to
the roll loop as a slow bias.

#### Authority budget

Available roll moment depends entirely on the DP reservation
policy. Reference numbers for two 1 MW symmetrical-nozzle pods in
twin-couple geometry, arm 8.5 m (per sec. 4.aa):

| reservation policy             | available roll moment | comment |
|---|---|---|
| Idle DP, full margin           | ~290 kN × 8.5 m ≈ **2.5 MN·m** | comparable to peak `M_wave` in beam Hs=3 m |
| 50% DP / 50% roll              | ~145 kN × 8.5 m ≈ **1.2 MN·m** | meaningful but not authoritative |
| 80% DP / 20% roll              | ~58 kN × 8.5 m ≈ 0.5 MN·m       | trim only |

Even with all pods reserved purely for roll, the available moment
(2.5 MN·m) is well below the peak wave forcing (8-12 MN·m). But it
is comfortably in the regime that flattens the **resonance peak**
of the response, which is the only band that matters in irregular
seas — the broadband content well away from `omega_n` is heavily
filtered by `1 / (Is² + bs + c)` regardless.

#### The two hard problems

**(1) Priority arbitration.** DP must retain saturation priority:
position-keeping is a hard safety constraint (loss of position near
a turbine is a collision hazard), roll comfort is comfort. The
standard formulation is a weighted-priority QP in the allocator:

```
minimize  || P (T u - tau_d) ||² + λ ||u||²    subject to |u_i| ≤ u_i,max
```

with `tau_d = (Fx_d, Fy_d, Mz_d, M_roll_d)`,
`P = diag(w_dp, w_dp, w_dp, w_roll)`, and `w_dp >> w_roll`. As DP
demand grows the residual capacity for roll shrinks; in heavy
weather the pods may be at their limits just holding station and
the roll loop ends up with effectively no authority.

This produces a **state-dependent F_max** for the roll loop, which
is materially harder than the constant F_max in our current
prototype. The observer is unaffected, but the inverse-dynamics
controller's saturation pathology (sec. 4.z, the bang-bang
re-excitation at `omega_n`) gets worse when F_max is itself
drifting on a wave-comparable timescale. A saturation-aware
controller variant is required — either a model-predictive
formulation that anticipates the budget envelope, or a softer
gain-scheduling on F_max(t).

**(2) Cross-coupling and the DP wave filter.** The DP wave filter
(a notch at the encounter frequency, the *other* half-life of the
Sælid resonator) explicitly rejects wave-frequency content from
position feedback because acting on it would burn fuel chasing
wave motion. The roll-augmented architecture deliberately injects
wave-frequency thrust commands. If the allocator is even slightly
imperfect — i.e. the achieved `(Fx, Fy, Mz)` from the roll thrust
is not exactly zero — then a wave-frequency surge/sway/yaw force
leaks into the slow loop. The DP wave filter sees this as
wave-frequency *motion* (not as commanded thrust) and correctly
rejects it, so DP stability is not at stake; but the leakage causes
**uncompensated wave-frequency vessel motion** that costs
station-keeping accuracy.

The mitigation is **geometric pairing** — the same twin-pod couple
from sec. 4.aa: forward pod and aft pod with equal-and-opposite
transverse thrust produces pure roll couple, zero net surge / sway /
yaw. This is the only allocator solution that keeps the two loops
genuinely decoupled. Single-pod or asymmetric pod use will always
leak. For a vessel with 4 pods (two forward, two aft) the natural
choice is two transverse couples, doubling the available roll
moment for the same per-pod thrust.

#### Other practical issues

- **Slew rate**: same as 4.aa — pods cannot azimuth at wave
  frequency, so practical operation is fixed-orientation,
  magnitude-only modulation.
- **Generator response**: stepping 1-2 MW of pod load up and down
  at wave frequency is a serious load swing for the diesel-electric
  bus. Battery hybridisation (BESS) becomes much more attractive
  than for DP-only operation where load is slow. This is the same
  "regenerative drive" argument from sec. 4.z applied to the whole
  bus.
- **Thruster wear**: DP pods are designed for slow-timescale duty;
  wave-rate cycling shortens bearing/seal life and stresses the
  gearbox. This is **the** reason Voith Schneider is the natural
  fit: cycloidal blade pitch reverses transverse thrust without
  shaft direction reversal, so the prime mover sees nearly constant
  load even when the thrust direction is reversing. Conventional
  azipods do not have this property. For a CSOV with conventional
  azipods, a duty-cycle limit (e.g. only enable wave-rate
  modulation in defined sea states / operations) is probably
  necessary.
- **Cavitation duty cycle**: wave-rate cycling between high
  transverse load and cavitation onset stresses the propeller and
  shaft. Continuous-duty cavitation analysis is required before
  claiming a given F_max can be sustained.
- **Classification / safety case**: any software path that lets a
  wave-frequency controller command DP thrusters is a high-integrity
  item under DNV DP-class notation. The approval story is
  non-trivial and is probably the gating issue for a commercial
  product even when the engineering stacks up.

#### Parallel operation with passive roll-reduction tanks

The strongest concept of all: the DP-augmented roll loop runs **in
parallel with a passive free-surface tank**, not as a replacement
for it. This is effectively the sec. 7.y hybrid architecture but
with the active actuator being the existing DP thrusters instead of
a dedicated unit. Specifically:

- **Free-surface tank** handles broadband dissipation across the
  whole encounter spectrum: ~40-50 % reduction baseline (per the
  sec. 4.z heading sweep), zero power, zero allocator burden.
- **DP-augmented roll loop** handles only the **residual resonance
  peak** that the free-surface tank cannot dissipate (its
  attenuation is centred on the sloshing frequency, not the
  vessel's roll natural frequency). After the tank has flattened
  the response peak, the active loop's required wave-frequency
  thrust authority is reduced by a large factor — the loop is
  shaving an already-attenuated peak, not cancelling the full
  forcing.
- **Saturation-aware controller** on the active loop knows the
  current DP reservation budget and gracefully reduces
  aggressiveness when DP needs the thrust, falling back to
  passive-only operation when DP is saturated. Because the passive
  tank carries the bulk of the work, this fallback costs only
  the resonance-peak shaving — not the entire roll-stabilising
  function.

The heading-sweep numbers in sec. 4.z make this very compelling:
free-surface tank already gets 43-48 % reduction, the DP-augmented
roll loop only needs to convert that to maybe 60-70 %, which
requires much less sustained wave-frequency thrust than standalone
roll actuation would need. The combination delivers good
performance across the whole heading sweep without ever pushing
the DP system into saturation conflict.

#### Mapping into the prototype

This requires a different abstraction layer than the
`DirectRollActuator` from sec. 4.aa. Sketched API:

```python
class Thruster:
    """Pod / tunnel / RDT, with kinematic constraints."""
    azimuth_angle: float           # rad, current orientation
    azimuth_rate_max: float        # rad/s, slew rate limit
    thrust_max: float              # N, magnitude cap
    cavitation_envelope: ...       # (azimuth, thrust) -> feasible?

class ThrusterAllocator:
    """Multi-thruster allocator with priority-weighted (DP, roll) demand.

    Solves the constrained QP each macro step to distribute the
    requested generalised force across the available thrusters.
    Reports both the achieved (Fx, Fy, Mz, M_roll) and the
    per-thruster commands.
    """
    thrusters: list[Thruster]
    weights: dict   # {'dp': w_dp, 'roll': w_roll} with w_dp >> w_roll

    def allocate(self, tau_dp: np.ndarray, M_roll_d: float) -> dict:
        """Returns {'u': per-thruster cmds, 'tau_achieved': ndarray,
        'M_roll_achieved': float}."""

class DPRollActuator:
    """Wraps a ThrusterAllocator. Exposes the same Tank-like
    forces() / step_rk4() interface as DirectRollActuator and the
    physical tank classes. The roll controller commands M_roll_d;
    the allocator decides how much actually gets delivered given
    DP reservation and per-thruster saturation, and reports back
    via the existing record_applied_tank_moment() hook so the
    observer-based controller stays consistent."""
```

Three new pieces of work compared to the dedicated-actuator case:

1. **Thruster kinematic model** (azimuth, slew rate, thrust limit,
   cavitation envelope). Conservative starting point: assume
   fixed-orientation magnitude-only, since that is the practical
   wave-rate operating mode anyway.
2. **Priority QP** in the allocator. For the twin-pod-couple
   geometry the QP is small enough to solve in closed form; for
   the general 4-6 pod case use `scipy.optimize.minimize` with
   linear constraints. Cost ~1 ms per macro step.
3. **Saturation-aware controller variant**. Either MPC-style
   anticipation of the budget envelope or a softer gain-scheduling
   on F_max(t). Simplest first cut: the existing observer-based
   controller with the per-step F_max passed in from the allocator
   instead of being a constant — this preserves the controller's
   no-wind-up linear behaviour up to the moving saturation
   boundary.
4. **Synthetic DP demand source** for testing. The honest choice is
   a slowly-varying (Wiener-filtered Gaussian or low-frequency
   sinusoid) `tau_dp(t)` representing wind / current / drift
   loading, then study how the roll loop's effective authority and
   achieved reduction degrade as the DP demand level increases.

This is a substantial piece of work — probably comparable in scope
to all the existing tank classes combined — and is deferred until
the simpler `DirectRollActuator` (sec. 4.aa) has been built and
validated. Together the two will let the prototype answer the
practical question that drives commercial concept selection on
DP-class CSOVs: **at what point does the engineering case for a
dedicated RDT installation beat the case for re-using the existing
DP thrusters with a software upgrade?** The expected answer, given
the heading sweep, is "almost never" — but the prototype is the
right place to verify it before committing to a hull design.

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

### 7.y Recommended architecture: passive free-surface + active U-tube

Combining the threads from sections 4.x (anti-heel duty), 4.y
(active in-duct RDT), 5 (free-surface stiffness boost) and 7.x
(multi-tank summing) leads to a specific architectural recommendation
for offshore vessels where roll reduction matters and active hardware
is acceptable: **install one passive free-surface tank and one active
RDT-driven U-tube, sized to operate together.**

The argument is multi-pronged:

1. **Failure-mode redundancy is solved cleanly.** The active tank's
   worst weakness (caveat 4.y(iv): a stopped or failed RDT presents
   a 60 % blockage that catastrophically overdamps the U-tube) is
   absorbed by the passive free-surface tank, which keeps working
   with zero electrical demand. Loss of RDT power degrades the
   system from "active+passive" to "passive only" -- a graceful
   degradation, not a cliff. Removes the need for exotic freewheel
   / bypass-duct hardware on the RDT for the safety case (though
   freewheel remains good practice).

2. **Frequency-band complementarity.** The free-surface tank is the
   only modelled device that improves *sub-resonance* response (via
   `dc44_extra` adding stiffness, sec. 5). The active U-tube
   handles the resonance band where the free-surface tank's notch
   doesn't reach. They occupy complementary frequency regions.

3. **Vessel re-tune helps both.** The free-surface tank pushes the
   effective vessel period from `T_n` to `T_eff > T_n`, into the
   sparse short-period tail of typical wave spectra (broadly
   beneficial). The active U-tube is then designed for `T_eff`
   rather than the bare-vessel `T_n`.

4. **Naval-architectural orthogonality.** Free-surface tanks must
   sit *high* (above the metacentre is best for `dc44_extra` sign
   and magnitude). U-tubes sit *low* (full beam at double-bottom
   level). No competition for ship volume.

5. **Smaller active hardware.** Once the free-surface tank handles
   the broadband / sub-resonance load, the active U-tube only needs
   enough authority to mop up the residual at vessel resonance. A
   smaller RDT (~1.0-1.2 m diameter, ~70-100 kN thrust) and
   correspondingly smaller energy buffer (~100-200 kJ/cycle vs
   ~600 kJ for a single active tank handling everything) suffice.

6. **Anti-heel duty is naturally orthogonal.** The free-surface
   tank physically cannot do anti-heel correction (the surface is
   always horizontal -- no static-offset capability). The active
   U-tube can, by parking the RDT at a static thrust setpoint that
   holds the desired `tau_eq` offset against gravity. So in the
   dual-tank architecture there is no conflict between roll-
   reduction duty and heel-correction duty: free-surface tank does
   roll continuously, active U-tube switches between roll-mode and
   heel-mode on demand.

Honest qualifier: this combination has not been simulated in this
prototype. The infrastructure supports it directly (sec. 7.x), but
plausible coupled-mode interactions between U-tube fluid swing and
free-surface sloshing -- particularly if their natural frequencies
happen to align -- have not been investigated. A worked
`csov_passive_active_combo.py` example would be the natural next
development if/when the C++ port reaches the multi-tank stage.

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
