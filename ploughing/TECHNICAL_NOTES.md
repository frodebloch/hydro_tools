# Technical Notes: Ploughing Simulator Findings

Notes from the Python prototype simulator (`ploughing/`) for reference when
implementing the C++ ploughing simulation in brucon.

---

## 1. Spatial Soil Variability

### 1.1 Published geotechnical data

The primary reference for spatial Su variability is Phoon & Kulhawy (1999),
a meta-study of 40+ sites. Key numbers for **inherent** variability (after
removing depth trends):

| Property | Range | Typical |
|----------|-------|---------|
| Su COV within one geological unit | 0.10 - 0.55 | 0.25 - 0.40 |
| Horizontal correlation length | 10 - 60 m | 20 - 40 m |
| Vertical correlation length | 1 - 3 m | 1 - 2 m |

Other references:
- Lacasse & Nadim (1996): COV = 0.20 - 0.40 for marine clays
- Baecher & Christian (2003): COV = 0.20 - 0.50
- Vanmarcke (1977): horizontal correlation 10 - 100 m
- El-Ramly et al. (2003): 20 - 80 m typical offshore

### 1.2 What the prototype uses

The prototype uses fGn (fractional Gaussian noise) with COV = 0.75 and a
5 m low-pass filter, giving a correlation length of ~15-20 m (r = 0.5 at
~15 m). This is overlaid with Markov zone transitions (soft/hard patches)
and Poisson spike events (boulders).

The combined stochastic multiplier over a 600 m traverse:
- COV: 0.77
- Range: 0.02 - 5.1 (equivalent Su: 0.2 - 61 kPa from nominal 12 kPa)

### 1.3 Important distinction: within-unit vs between-unit variability

Published COVs describe variability **within one geological unit** — a
single clay layer with consistent depositional history. Real ploughing
routes cross **multiple units**. The variability in operational data comes
from two fundamentally different sources:

1. **Within-unit** (continuous): metre-to-metre Su variation in firm clay.
   COV = 0.25 - 0.40 from literature. This is what fGn should represent.

2. **Between-unit** (discrete): geological boundaries, different
   depositional environments. The 7-minute soft zone visible in our
   reference data is a different soil layer entirely (e.g. recent marine
   sediment overlying glacial till, Su dropping from 12 kPa to 1-2 kPa).
   This is what the Markov zone model represents.

Additionally, **discrete features** (boulders, cobble lenses, cemented
layers, shell beds) are neither within-unit nor between-unit — they are
point/short-extent anomalies modelled as Poisson events.

### 1.4 Recommendations for the C++ implementation

- **fGn COV should be 0.30 - 0.45** for realistic within-unit variability.
  The prototype uses 0.75 which is too high for a single unit — it works
  because the fGn is doing double duty as both inherent variability AND
  unresolved thin interbeds. Better to separate these concerns.

- **Zone transitions should carry the large-scale variability.** A soft
  zone with factor 0.10 - 0.15 correctly represents crossing into a
  different geological unit.

- **Correlation length should be 15 - 40 m** for fGn. This is controlled
  by the Hurst exponent H and the low-pass filter length. H = 0.85 - 0.95
  gives PSD ~ f^(-(2H-1)), close to pink noise.

- **Spike rate of 1 per 20-25 m** is high but reasonable for morainic
  seabeds (North Sea glacial deposits). Reduce to 1 per 50-100 m for
  less challenging environments.

- **All soil parameters must be spatial (per metre), not temporal.** The
  fGn is pre-generated as a spatial field; the plough reads it as it
  advances. Faster ploughing gives higher temporal frequencies for the
  same spatial content — this is physically correct and eliminates the
  speed-dependent spectral distortion that would result from a temporal
  noise model.

- **Consider allowing real CPT/Su profiles as input** instead of purely
  stochastic soil. A measured CPT log along the route would give the
  correct spatial structure naturally. The stochastic model is useful
  for design studies and cases where site data is unavailable.

### 1.5 The plough as a spatial filter

The plough body (~8 m long, ~3 m wide) and its soil failure mechanism
integrate soil properties over a finite volume. The active failure wedge
ahead of the share extends several metres. This means the plough acts as
a spatial low-pass filter on the Su field — sub-metre heterogeneity is
mechanically averaged out.

In the prototype this is modelled with a 1st-order low-pass on the fGn
with a 5 m length scale. A more physical approach for C++ would be to
apply a running average over the share width and failure wedge length,
operating on the raw spatial Su field.

---

## 2. Catenary as a Nonlinear Spring

### 2.1 Key finding: depth-dependent filtering

The catenary between vessel and plough acts as a nonlinear spring whose
stiffness depends strongly on water depth. This was the most important
finding from the depth bracket study. At shallow depths the coupling is
stiff and soil disturbances transmit directly to the vessel; at deep
water the long catenary filters almost everything out.

| Depth | k_cat (kN/m) | Wave ±t | T range | Character |
|-------|-------------|---------|---------|-----------|
| 30 m | 360 | ±6.6 | 9 - 100 t | Stiff, hairy, sharp spikes |
| 75 m | 80 | ±1.5 | 6 - 80 t | Moderate filtering |
| 150 m | 29 | ±0.5 | 13 - 70 t | Strong filtering |
| 300 m | 11 | ±0.2 | 18 - 51 t | Nearly flat, decoupled |

The reference operational data (range 5-100 t, dense wave-frequency
texture) is consistent with shallow water (20-40 m).

### 2.2 Elastic catenary spring model (prototype approach)

The prototype uses a parabolic cable approximation for the sag correction,
pre-tabulated as a T_h(D) lookup table:

```
chord = sqrt(D^2 + h^2)
w_n = w * cos(alpha)
s_geom = chord + w_n^2 * chord^3 / (24 * T^2)    (arc length with sag)
s_elastic = L0 * (1 + T / EA)                      (elastic stretch)
Solve: s_geom = s_elastic for T at each D
Then: T_h = T * cos(alpha)
```

The resulting spring is **asymmetric**:
- **Slack side** (D < D_eq): dominated by sag geometry, tension drops
  slowly — broad, smooth tension dips in soft soil
- **Taut side** (D > D_eq): sag reduces to zero, then EA/L axial
  stiffness takes over — sharp, steep tension spikes in hard soil

This asymmetry matches the visual character of real operational data:
broad tension valleys and sharp tension peaks.

### 2.3 For the lumped mass model in brucon

**NOTE: The lumped mass line model on brucon master / current branch has
not been updated with the latest improvements.** These improvements
(details TBD — likely including seabed contact, friction, internal
damping, and numerical stability at low tension) will be important for
the ploughing sim. The line model must be brought up to date before
starting the ploughing C++ implementation.

The brucon C++ sim will use a lumped mass line model which inherently
captures all of the above — the elastic catenary spring behaviour emerges
naturally from the discretized line dynamics. Key considerations:

- **The lumped mass model replaces the spring table entirely.** No need
  to pre-compute T_h(D). The line dynamics solve for tension at every
  node at every timestep.

- **Wave-frequency tension modulation comes for free.** First-order
  vessel motion drives the top node of the line model. The tension
  response at the vessel fairlead naturally includes the catenary
  spring filtering. No need for the displacement-offset approach used
  in the prototype.

- **The asymmetric spring character should emerge naturally** from the
  line geometry — verify this during commissioning by comparing T_h vs
  layback curves from the lumped mass model against the analytic elastic
  catenary solution.

- **Wire grounding is handled naturally.** As wire contacts the seabed,
  lumped mass nodes rest on the bottom with friction. No special
  grounding model needed (the prototype's grounding model was broken
  and had to be removed — the lumped mass approach avoids this entirely).

- **Element length matters.** The catenary curvature is concentrated
  near the plough (touchdown region). Use shorter elements near the
  seabed and longer elements near the vessel. For a 300 m suspended
  wire, 50-100 elements is typical for good tension accuracy.

- **Seabed friction on grounded wire** should be included. In the
  prototype we used a small buffer (~75 m) of grounded wire. The
  lumped mass model should let grounded nodes slide with Coulomb
  friction — this provides natural damping and prevents the wire
  from "skating" on the seabed.

### 2.4 Wire deployment

Real operations manage wire deployment so that only a small buffer
(50-100 m) sits on the seabed at operating tension. This is enough to
handle tension fluctuations without lifting the plough off bottom, but
not so much that the grounded wire creates problems.

The wire length should be computed from the catenary arc length at
operating tension plus the buffer:

```
a = T_h / w_sub
D = a * arccosh(1 + h / a)
s = a * sinh(D / a)
wire_total = s + buffer    (buffer ~ 50-100 m)
```

This eliminates the need for complex grounding models in most cases.
The scope ratio (wire/depth) is NOT constant — it varies from ~11:1
at 30 m to ~3.5:1 at 300 m.

---

## 3. Plough Dynamics

### 3.1 Effective mass

The plough body (25-35 t dry) entrains a small amount of soil and
water.  For a narrow-furrow plough (share ~0.3 m wide, burial ~1.5 m
deep, furrow cross-section ~0.45 m^2):
  - Active failure wedge ahead of share: ~1-1.5 m^3 × 2 t/m^3 ≈ 2-3 t
  - Hydrodynamic added mass (bluff body near seabed): ~5-10 t
  - Soil adhering to skids and body: ~5-10 t

Total effective mass ≈ 2× dry mass (50 t for a 25 t plough).

### 3.2 Resistance model

The total plough resistance has four components:

```
F_total(V) = F_soil * tanh(V / V_breakout) + C_d * rho_soil * A_furrow * V^2 + F_friction + F_hydro
```

**1. Quasi-static soil cutting** (speed-independent, stochastic):

```
F_soil = Nc * Su * w * d  ×  stochastic_factor
```

The `tanh(V / V_breakout)` ramp provides a smooth transition from zero
resistance (stationary plough) to full cutting resistance.  The quasi-static
soil failure force is independent of speed — it is the force required to
shear the soil regardless of how fast the plough moves.

**2. Dynamic soil inertial drag** (V^2):

```
F_inertial = C_d * rho_soil * A_furrow * V^2
```

The plough must accelerate the soil it displaces out of the furrow path.
This momentum transfer scales with `rho * A * V^2`, the same dimensional
argument as fluid drag (Reece 1964, Palmer & King 2008).  `A_furrow = w * d`
is the furrow cross-section and `rho_soil` is the saturated bulk density.
`C_d` (typically O(100) for soil-plough interaction) is an empirical
drag coefficient — not a fluid drag coefficient.  At ploughing speeds
(0.1-0.3 m/s) the soil is being sheared and broken, so the effective
resistance includes cohesion and strain-rate effects beyond pure momentum
transfer.  The coefficient captures:
  - Inertial acceleration of displaced soil mass
  - Dynamic pressure on the failure wedge
  - Seabed suction effects on the plough body

At operating speed (V ~ 0.12 m/s) with default parameters:
  - `A_furrow = 3.0 × 1.5 = 4.5 m^2`
  - `F_inertial = 100 × 1800 × 4.5 × 0.12^2 ≈ 12 kN` (~4% of total)
At V = 0.25 m/s:
  - `F_inertial ≈ 51 kN` (significant speed-limiting contribution)
At V = 0.30 m/s:
  - `F_inertial ≈ 73 kN` (limits maximum plough speed)

The primary speed sensitivity in operational data comes from the catenary
spring dynamics and plough inertia (the high effective mass smooths
speed oscillations).  The V^2 term mainly serves to limit maximum
plough speed during runaway conditions (very soft soil).

### 3.3 Implicit Euler integration

The prototype uses implicit Euler for the plough dynamics with the
catenary spring coupled into the Newton iteration:

```
g(V) = (m/dt)(V - V_old) + F_resist(V) - T_h(D - (V - V_vessel)*dt) = 0
```

where `F_resist(V) = F_soil * tanh(V/V_b) + C_d * rho * A * V^2 + F_friction + F_hydro`
includes all speed-dependent terms.

The key insight is that when the plough speeds up, D shrinks, T_h
drops, limiting the speed excursion — this coupling must be inside the
implicit solve, not lagged by one timestep. With explicit integration
or lagged coupling, the system can oscillate or go unstable at stiff
spring conditions (shallow water).

For the C++ sim with a lumped mass line, this coupling is handled
differently — the line model provides the tension at the plough node
directly, and the plough integrator uses that tension as its driving
force. The coupling is through the shared boundary condition at the
plough tow point.

---

## 4. Vessel DP and High-Tension Slowdown

### 4.1 Tension-speed modulation

The DP controller adjusts the speed setpoint based on filtered tow
tension. High tension reduces the setpoint; low tension increases it.
This mimics the operator or auto-pilot adjusting speed in response to
soil conditions.

Key parameters (tuned to match reference):
- tension_nominal = 410 kN (~42 t)
- tension_speed_gain = 0.7e-6 (m/s per N)
- filter time constant = 15 s

The gain must be moderate — too high creates positive feedback (slow
down → stuck in hard patch → slower). The nonlinear F(V) model provides
the primary self-regulation; the tension-speed modulation is secondary.

### 4.2 High-tension slowdown state machine

Safety system with four states:

```
NORMAL → STAGE1 (T > 70 t) → STAGE2 (T > 85 t) → ESTOP (T > 100 t)
```

- STAGE1: decelerate at 0.005 m/s^2
- STAGE2: decelerate at 0.010 m/s^2
- ESTOP: immediate speed target = 0

Recovery: accelerate at 0.005 m/s^2 when tension drops below threshold.
No catch-up: the vessel does NOT speed up to recover position lost
during the slowdown. The DP reference point continues at the nominal
speed throughout — the vessel simply falls behind and reconnects.

The slowdown thresholds should be filtered (5 s time constant on the
tension input) to avoid triggering on wave-frequency oscillations.

### 4.3 Speed control vs position control

Both modes were tested:

| Mode | Mechanism | T-V correlation |
|------|-----------|-----------------|
| Speed control | Explicit tension-speed feedback | -0.41 |
| Position control | PID fights catenary load | -0.23 to -0.41 |

Position control can produce similar results to speed control because
the catenary tension acts as a disturbance force on the position PID.
Hard soil increases tension, which decelerates the vessel (PID can't
fully compensate), producing a natural inverse T-V correlation without
any explicit tension feedback.

---

## 5. First-Order Wave Surge

### 5.1 Mechanism

The vessel oscillates at wave frequency (Tp ~ 7-10 s) due to
first-order wave forces. The DP wave filter excludes this motion from
the control feedback, so the DP does not fight it. The physical
oscillation modulates the layback at wave frequency, which modulates
the catenary tension through the spring.

This produces the "hairy" tension texture visible in real operational
data — dense, rapid ±5-10 t fluctuations superimposed on the slower
soil-driven variability.

### 5.2 Prototype implementation

The prototype adds a sinusoidal displacement to the tow point position:

```
x_surge = RAO * (Hs / 2) * sin(omega * t + phase)
```

With RAO = 0.3 m/m, Hs = 1.2 m: amplitude = 0.18 m. Through the
catenary spring at 30 m depth (k = 360 kN/m), this gives ±65 kN =
±6.6 t at the wave period.

### 5.3 For the lumped mass model

The lumped mass model handles this naturally — the vessel top node
moves with the actual vessel motion (including first-order waves),
and the line dynamics propagate this to tension at all nodes. No
special displacement offset needed.

The vessel motion model in brucon should include first-order wave
response (RAOs) with the DP wave filter preventing the controller
from counteracting it. The key parameter is the surge RAO at the
dominant wave period — typically 0.2 - 0.5 for a 130 m cable layer
at Tp = 7-10 s.

---

## 6. Measurement Point: Vessel vs Plough Tension

### 6.1 What the instruments measure

Modern plough spreads instrument both ends:
- **Vessel fairlead**: load pin on the fairlead sheave or winch.
  Measures the wire tension at the vessel end, which equals the
  catenary horizontal force T_h plus the geometric vertical component.
- **Plough tow bridle**: subsea load cell on the tow attachment.
  Measures the wire tension at the plough end — this is NOT the soil
  resistance. It is the catenary tension at the bottom, which is
  driven by the vessel position through the catenary geometry.

### 6.2 Key insight

At the plough end, the tension is what the wire pulls with. This is
determined by the catenary geometry (vessel position, layback, wire
weight), not directly by the soil resistance. The plough-end tension
signal is therefore smooth and slowly-varying — it does NOT contain
wave-frequency texture regardless of water depth, because the plough
has high inertia and the wire angle at the seabed is near-zero.

The "hairy" wave-frequency texture is only visible at the **vessel
end**, where wave surge modulates the layback through the catenary
spring.

### 6.3 Implications

- If reference data shows wave-frequency texture, it is almost
  certainly vessel-end tension.
- At shallow depths (< 40 m), the catenary is stiff enough that
  vessel-end and plough-end tensions are nearly identical in their
  low-frequency content (mean, range, correlation with speed).
  The only difference is the wave-frequency ripple.
- At deep water (> 100 m), vessel-end tension is heavily filtered
  by the soft catenary — a flat signal with narrow range. Plough-end
  tension is similarly flat (driven by vessel position through the
  same soft spring). Both are smooth but for different reasons.

### 6.4 What about soil resistance?

The soil resistance is a separate force acting on the plough body.
It is NOT directly measurable from any tension instrument on the wire.
To measure it you would need strain gauges on the plough share or
skids — rare in operational practice.

The soil resistance can be **inferred** from the plough dynamics:

```
F_soil = m_eff * dV/dt + T_plough
```

But this requires accurate plough acceleration measurements (noisy)
and knowledge of the effective mass (uncertain).

---

## 7. Parameter Summary

Calibrated parameters from the prototype that produced the best match
to the reference operational data (30 m depth, speed control mode):

### Vessel
- LOA: 130 m, beam: 23 m, draft: 7.8 m, Cb: 0.72
- Max thrust surge: 900 kN (~90 t)
- Tow point: stern, x = -60 m from midship

### Wire
- 76 mm wire rope, 25 kg/m, EA = 250 MN (fill factor 0.55)
- Wire length: catenary arc + 75 m buffer
- Submerged weight: 199.6 N/m

### Plough
- Mass: 25 t (effective: 50 t with entrained soil)
- Width: 3 m, burial depth: 1.5 m, A_furrow: 4.5 m^2
- Breakout speed: 0.02 m/s
- Soil inertial drag: C_d = 100, rho_soil = 1800 kg/m^3

### Soil
- Su = 12 kPa, gamma' = 8 kN/m^3, Nc = 4
- Base cutting force: 216 kN (quasi-static, speed-independent)
- Soil inertial drag at V = 0.12 m/s: ~12 kN

### Stochastic soil
- fGn: H = 0.90, COV = 0.75, dx = 2.5 cm, LP = 5 m
- Spikes: rate 1/20 m, amplitude 2.5x mean, length 1.5 m
- Zones: soft (10%, ~67 m long), hard (160%, ~17 m long)
- Transition rates: soft entry 1/330 m, hard entry 1/500 m

### Environment
- Hs = 1.2 m, Tp = 7.5 s, surge RAO = 0.3
- Current: 0.2 m/s head, wind: 6 m/s

### DP controller
- Speed control, T_surge = 114 s, zeta = 0.9
- Tension-speed: nominal 410 kN, gain 0.7e-6, tau 15 s
- Slowdown: Stage1 690 kN, Stage2 834 kN, E-stop 980 kN

### Simulation
- dt = 0.2 s, duration = 3600 s
- Catenary spring stiffness at 30 m: ~360 kN/m

### Results (30 m, speed control)
- T_mean: 36 t, T_range: 9 - 100 t
- V_mean: 0.104 m/s, V_std: 0.028 m/s
- corr(T, V): -0.41
- Wave tension texture: ±6.6 t at Tp = 7.5 s
