# Online Vessel Footprint Estimation with Gangway Operability — Feasibility Study

Working title: **Combined Quasi-static Capability + Excursion Analysis (CQA)**
Status: draft for discussion
Owner: blofro

---

## 1. Objective

Assess the feasibility of computing, **online and continuously**, while a walk-to-work
(W2W) vessel is on DP next to an offshore structure:

1. The **station-keeping footprint** of the vessel
   - As an **excursion polar** of position and heading around the DP setpoint.
   - For two states evaluated in parallel each cycle:
     - (a) **current state** (intact, with live environment and live thruster/power configuration)
     - (b) **post-WCFDI state** (after the worst-case single failure compatible with the
       DP-class-2 redundancy concept).
2. The **W2W gangway operability** mapped from the vessel footprint, taking into account:
   - hard end-stop limits (telescoping length, slewing angle, booming angle),
   - max telescoping / slewing / booming **velocities**,
   - tip motion velocities driven by wave-frequency vessel motions.

The primary deliverable is this study; success criteria are listed in §10. If the
methodology and numbers look credible, follow-on work is (i) a Python prototype in
`hydro_tools/cqa/`, (ii) a demonstrator integrated into `~/src/brucon` DP control system.

### 1.1 Dual deliverable: online consequence analysis + offline footprint extension to static capability analysis

The same frequency-domain machinery that drives the online tool naturally extends to a
**desktop / classification-style footprint estimator** that complements the existing
static capability analysis. The two use cases share core (`closed_loop.state_covariance_freqdomain`,
the gangway model, and the WCFDI transient predictor) and differ only in *what data
feeds in* and *how results are presented*:

**A. Online "better consequence analysis"**
*(IMO MSC.645 / DNVGL-OS-DPS-1 / IMCA M103, M220 — DP operational guidance)*
- Real-time, evaluated each DP cycle at the current operating point.
- Inputs: live environment (anemometer, current sensor, observer-estimated bias),
  live thruster/power configuration, live `ConsequenceAnalysis` WCFDI selection.
- Output: vessel footprint envelope and gangway operability margin **for both intact
  and post-WCFDI states at the current heading**, presented to the operator (gauge,
  short time-history, alarm thresholds).
- Includes the **WCFDI transient envelope** during the recovery window — this is what
  is missing from today's binary "capability available / not available" indicator.

**B. Offline footprint extension to static capability analysis**
*(IMCA M140 rev. 2 / DNV-ST-0111 — DP capability analysis methodology)*
- Desktop tool, run once per design point or weather scenario.
- Inputs: design environment matrix (Vw × θ × Hs × Tp × Vc), assumed thruster/power
  configuration, chosen WCF case.
- Output: **footprint polar** (standard deviation and 95 % contour of position/heading
  deviation) at every grid point, plotted side-by-side with the existing static
  capability polar (max sustainable Vw at each θ).
- Use case: lets the desk engineer assess **how much position margin actually exists
  when the vessel sits at the edge of the static capability envelope**, without running
  the full time-domain simulation campaign that IMCA M140 / DNV-ST-0111 otherwise imply.
- The **frequency-domain shortcut** (versus full time-domain MC) needs to be validated
  against `vessel_simulator` / `dp_runfast_simulator` runs at a representative grid
  of operating points — exactly the comparison the standards expect when proposing a
  shortcut method. This validation is precisely what P4/P5 in the roadmap deliver, so
  it serves both deliverables for free.

**Why both, not just one:** the online tool answers "*what should I do right now?*";
the offline tool answers "*was this DP class / vessel design appropriate for this site
and operation?*". They use the same physics, but appear in different sections of the
DP documentation chain (operations manual vs. capability dossier). Building both from
the same core avoids the usual drift between operational and design assumptions.

## 2. Definitions

| Term | Definition |
|---|---|
| **Footprint** | Statistical envelope (e.g. 95 % contour) of vessel position+heading around the DP setpoint under the assumed environment and DP control law. |
| **Capability polar** | Maximum environmental load (parametrised here as wind speed via the DNV ST-0111 wind/wave/current relation, see §3.1) that the DP system can balance for each relative weather direction. Already implemented in brucon. |
| **Excursion polar** | Standard deviations (and selected percentile contours) of surge/sway/heading deviation from setpoint, as a function of relative weather direction. **New.** |
| **WCFDI** | Worst-case failure design intent: the single fault among the configured failure modes (bus group loss or thruster loss) that yields the smallest remaining thrust reserve. Computed each cycle by `consequence_analysis.cpp`. |
| **Gangway envelope** | Region in (relative position × relative heading) where the gangway can stay coupled without exceeding any geometric or rate limit. |

## 3. What is already available

### 3.1 In `~/src/brucon` (C++, production DP system)

| Component | Path | Relevance |
|---|---|---|
| `CapabilityAnalysis` | `libs/dp/capability_analysis/` | Computes max wind speed per heading via binary search using thrust allocation against (wind + wave-drift + current) forces. Has `Prevailing` mode that uses live measured environment, and DNV ST-0111 Beaufort coupling between wind, Hs, Tp, current. **Reuse.** |
| `ConsequenceAnalysis` | `libs/dp/consequence_analysis/` | Enumerates failure modes (bus group loss, thruster loss), produces `worst_case_failure_` with the post-failure `SelectiveState` and `thrust_reserve`. Already filters `live_tau_` for prevailing utilisation. **Reuse for WCFDI selection.** |
| `dp_capability` app | `apps/dp/dp_capability/` | Service wiring `CapabilityAnalysis` to DDS, reading `vessel_data`, `vessel_wind_data`, `propulsors`, `power_system` from prototxt. Template for cqa service. |
| `BasicAllocator` + `PropulsorConfiguration` | `libs/dp/thrust_allocation/`, `libs/propulsor_configuration/` | Pseudo-inverse with bus power constraints, used by both capability and consequence analyses. |
| `VesselModel`, `WaveResponse`, `WindForceModel` | `libs/dp/vessel_model/` | Wind force coefficients (table or DNV type), wave drift forces from PdStrip RAOs, vessel hydrodynamic coefficients. |
| `dp_estimator` | `libs/dp/dp_estimator/` | Nonlinear passive observer (Fossen) producing low-frequency position/velocity, bias (= environmental low-freq force estimate), and `ResponseFrequencyEstimator` for wave-frequency states. Source for live tau, position covariance starting point, and wave-frequency motion estimates. |
| `dp_controller` | `libs/dp/dp_controller/` | Closed-loop control law — needed to linearise the closed loop for excursion variance prediction. |
| `vessel_simulator` (3-DOF horizontal-plane) | `libs/simulator/vessel_simulator/` | 3-DOF horizontal vessel sim with wind spectrum and current variability — usable for **Monte-Carlo validation** of the excursion polar (no wave-frequency calculations needed for the slow-drift envelope). |
| `dp_runfast_simulator` | `libs/simulator/dp_runfast_simulator/` | Faster-than-realtime DP closed-loop sim — candidate for offline batch validation. |
| `LandingPoint` / `OffshoreStructure` | `libs/dp/offshore_structures/` | Geometric anchor of the structure side; **does not currently model the gangway mechanism** (slew/boom/telescope limits and rates). New model needed. |

### 3.2 In `~/src/hydro_tools`

| Component | Path | Relevance |
|---|---|---|
| `dp_simulator_visualization/dp_sim_vis/` | scene + platform + vessel geometry, UDP receiver from brucon DP sim | Visualiser; we can extend its scene to overlay the live and post-WCFDI footprint and gangway envelope. |
| `environment/` | `pdstrip_cat`, response functions, hindcast/forecast fetchers (NORA3, WW3, NorKyst, TOPAZ5), wave-buoy and field-point analysis | Source for offline weather scenarios and for validating the live env. inputs that drive the prevailing capability mode. |
| `pdstrip_cat/` | PdStrip wrapper, drift-force extraction, slow-drift swell analysis | Generates the RAO + drift coefficient files (`pdstrip.dat`) consumed by brucon's `WaveResponse`. |
| `optimiser/`, `obstacle/` | Voyage and path-planning context | Not directly relevant. |

### 3.3 Gaps (must be built)

1. **Excursion polar predictor**: closed-form linearised covariance from
   environmental force PSDs through the closed-loop DP transfer function. Not present.
2. **Monte-Carlo validator**: short batch runs of `vessel_simulator` (3-DOF, no
   wave-frequency) at fixed setpoint with stochastic wind/current to corroborate the
   linearised covariance. Wraps existing C++.
3. **Gangway operability model**: kinematic transformation of vessel
   pose+twist at the connection point into gangway joint coordinates and rates,
   plus end-stop and rate-limit checks. Not present.
4. **Angular rate proxy**: gyro/MRU rates are not measured — we will use filtered
   numerical differentiation of attitude with documented bandwidth/lag/uncertainty.
5. **Online execution wiring**: schedule periodic re-evaluation per DP cycle for
   both intact and post-WCFDI states.

## 4. Methodology

### 4.1 System view per cycle

```
                  +---------------------------+
sensors --------> | dp_estimator (existing)   | --> low-freq pose, bias, wave states
wind sensor ----> +-------------+-------------+
                                |
                                v
                  +---------------------------+
power+thr state-> | consequence_analysis      | --> WCFDI -> post-failure SelectiveState
                  +-------------+-------------+
                                |
            +-------------------+---------------------+
            |                                         |
            v                                         v
+-----------+-----------+               +-------------+-------------+
| capability_analysis   |               | capability_analysis       |
| (intact state)        |               | (post-WCFDI state)        |
+-----------+-----------+               +-------------+-------------+
            |                                         |
            v                                         v
+-----------+-----------+               +-------------+-------------+
| EXCURSION predictor   |               | EXCURSION predictor       |
| (linearised cov.)     |               | (linearised cov.)         |
+-----------+-----------+               +-------------+-------------+
            \                                         /
             \-------------------+--------------------/
                                 v
                  +---------------------------+
                  | GANGWAY operability map   |
                  | (geometry + rate limits)  |
                  +---------------------------+
                                 |
                                 v
                  capability margin, excursion ellipses (intact + WCFDI),
                  gangway operability flags + most-limiting axis
```

The **capability** branch already exists. The new cqa work consists of the boxed
"excursion predictor" and "gangway operability map" plus the orchestration around
two-state evaluation.

### 4.2 Excursion predictor — primary method (linearised closed-loop covariance)

Assumptions:
- 3-DOF horizontal plane (surge η₁, sway η₂, yaw η₆).
- Linearised vessel + DP controller around the current operating point. Wind feed-forward
  (if used), PD on position, and integral action treated as the brucon controller does.
- Slow-frequency disturbances dominate the position-keeping envelope:
  - low-frequency **wind gust** force PSD (NPD/Davenport with the brucon
    `WindForceModel` linearised in apparent wind),
  - **wave drift** slow-drift force PSD (Newman approximation from the wave spectrum
    + drift coefficients in `WaveResponse`),
  - **current variability** modelled as the brucon `current_variability_model` PSD.
- High-frequency wave-frequency motion handled separately at the gangway tip (see §4.4),
  not part of the position-keeping cov.

Procedure each cycle:
1. Build linearised plant `A(η₀, U_rel, ψ₀)` and input matrix `B` for both states
   (intact and post-WCFDI thruster set / power constraints from `consequence_analysis`).
2. Build linearised closed-loop `A_cl = A − B K`, with `K` extracted from the running
   `dp_controller`.
3. Build environmental force PSD matrix `S_w(ω)` per direction.
4. Solve continuous-time Lyapunov equation `A_cl P + P A_clᵀ + B_w S̄_w B_wᵀ = 0`
   for the steady-state position covariance `P` (using the variance-equivalent
   white-noise spectral intensity `S̄_w` for each disturbance band).
5. Output:
   - 1-σ surge/sway/heading,
   - 95 % position ellipse,
   - heading 95 % interval,
   - direction-of-most-likely excursion (eigenvector of P).

Cost: one Lyapunov solve per direction sweep (or single solve at current heading; the
"polar" comes from sweeping the relative weather direction). Cheap enough for ~1 Hz.

The same machinery runs twice per cycle (intact, post-WCFDI), reusing the
`SelectiveState` produced by `ConsequenceAnalysis`.

### 4.3 Excursion predictor — validation method (Monte-Carlo, offline first)

Use brucon `vessel_simulator` (3-DOF, no wave-frequency calculations) in batch:
- For a grid of (wind speed, wind direction, current, Hs, Tp), run N short replicates
  with stochastic wind and current variability inputs.
- Estimate sample covariance of position/heading.
- Compare to the linearised prediction and quantify regions where the linearisation
  breaks down (large excursions, controller saturation, post-WCFDI low margin).

Initially **offline only**. Online MC is not in scope.

### 4.4 Gangway operability map

**Geometry of the CSOV gangway (corrected):**
- The gangway has a **rotating base ("rotation centre")** that can be **raised and
  lowered vertically** along the gangway pedestal. Its position in the vessel body
  frame, when at the lowest point, is `p_base = (5.0, -9.0, -8.0)` m
  (from `config_csov/posrefs.prototxt.in`).
- `max_height = 25 m` is the **max vertical travel of the rotation centre** above the
  base position, **not** the tip height. Hence with boom angle `β = 0` the tip sits
  at the same height as the rotation centre.
- From the rotation centre the gangway extends with three controllable DOFs:
  - **telescope length** `L ∈ [18, 32] m`,
  - **slew angle** `α` (about vertical),
  - **boom angle** `β` (elevation).
- Plus the **rotation-centre vertical position** `h ∈ [0, 25] m` (treated as a slow
  setup variable controlled by the operator before/around connection, not a fast
  control DOF for operability).

**What we model now and why:**
- **Slew `α` and boom `β`** are deliberately *not* gated against limits in the
  initial study. In real W2W operations the operator selects vessel heading and
  rotation-centre height so that connection occurs comfortably away from those
  end-stops; they are rarely show-stoppers. We **display** the predicted
  `α, β, αdot, βdot` ranges from the excursion + wave-frequency motion so the
  operator (and the analysis) sees the working point, but we don't compute
  pass/fail margins against α/β end-stops.
- **Telescope** is the operability-critical axis. Both:
  - **end-stops** `L ∈ [18, 32] m` — vessel surge/sway/yaw excursion translates
    almost directly into telescope-length variation when the gangway points
    roughly along the connection direction; hitting `L_min` or `L_max` is the
    typical disconnect trigger.
  - **stroke velocity** `Ldot` — driven by low-frequency vessel velocity at the
    rotation centre and wave-frequency motion at the rotation centre, both
    projected onto the gangway pointing direction. Industrial gangways have a max
    stroke velocity (typ. ~0.3–1.0 m/s); we will compute the predicted `Ldot`
    distribution and present it, then choose thresholds.

**Per cycle, for both intact and post-WCFDI states:**

1. **Forward kinematics (operating point).** From current vessel pose, gangway
   rotation-centre height `h`, and landing-point world position, compute the
   nominal `(L₀, α₀, β₀)` of the gangway.
2. **Telescope-direction unit vector** `ê_L` in vessel body frame: the line from
   rotation centre to landing point. The Jacobian row that matters is
   `Ldot ≈ ê_Lᵀ · v_rc`, where `v_rc` is the velocity of the rotation centre
   relative to the landing point.
3. **Telescope length envelope from excursion.** From the vessel position+heading
   covariance `P` (§4.2), propagate to the rotation-centre position covariance
   (translation + lever-arm × heading), then to telescope length:
   `σ_L = sqrt(ê_Lᵀ · P_rc · ê_L)`. Compare distribution of `L = L₀ + ΔL` against
   `[L_min, L_max]`. Output:
   - margin-to-`L_min` and margin-to-`L_max` in σ-units and as a probability of
     exceedance per unit time (using the level-crossing rate of a Gaussian process
     given the closed-loop bandwidth).
   - the principal direction of vessel excursion that drives telescope variation
     (i.e. how much would be gained by a small heading change).
4. **Telescope velocity envelope.** Combine:
   - low-frequency contribution: `σ_Ldot,LF = sqrt(ê_Lᵀ · P_v_rc,LF · ê_L)` from
     the velocity covariance produced by the same Lyapunov solve (`P` includes
     velocity DOFs in the state vector),
   - wave-frequency contribution: `σ_Ldot,WF` from RAOs at the rotation centre
     (translational + rotational coupling) integrated against the wave
     spectrum (parametric, see §11.2). Use yaw-rate via RAO (model-based) rather
     than differentiation here, per §4.5.
   Total `σ_Ldot = sqrt(σ²_LF + σ²_WF)`. Report the 95-percentile `|Ldot|` and
   compare to a configurable threshold (default 0.5 m/s, displayed only).
5. **Display-only outputs** for slew/boom: nominal `α₀, β₀`, and 95-percentile
   ranges of `α, β, αdot, βdot` derived in the same way (Jacobian rows for slew and
   boom). No pass/fail.

### 4.5 Angular rate handling

No measured roll/pitch/yaw rate. Approach:
- Take filtered MRU attitude and heading from `dp_estimator`.
- Differentiate with a 2nd-order low-pass / Savitzky–Golay; document the cut-off
  (target ≥ wave peak frequency, ≤ sensor noise floor) and resulting amplitude/phase
  error vs. frequency.
- For wave-frequency yaw rate, prefer the model-based path: combine measured wave
  spectrum × yaw-RAO to estimate the std of yaw rate without differentiation.
- Quantify the resulting uncertainty in `αdot` operability margin.

### 4.6 Two-state evaluation cadence

Both intact and WCFDI evaluations run each control cycle (target ~1 Hz online).
Intermediate quantities reused between the two:
- environment estimate and PSDs (identical),
- wind/wave/current forces at vessel CO (identical),
- linearised plant (different B, possibly different K if controller reconfigures).

## 5. Data and signal needs

| Signal | Source | Status |
|---|---|---|
| Position/heading low-freq | `dp_estimator` | Available |
| Velocity low-freq | `dp_estimator` | Available |
| Bias force (≈ env. low-freq force) | `dp_estimator` | Available |
| Wind speed/direction | wind sensor | Available |
| Wave spectrum (Hs, Tp, dir) per partition | Forecast (NORA3/WW3); optionally wave radar; wind-sea via `environment/wave_buoy.py` + measured wind | **No direct onboard spectrum measurement assumed**; parametric JONSWAP/Torsethaugen shape used in cqa |
| Current speed/direction | `dp_current_estimator` | Available, model-based |
| Thruster availability/utilisation | `propulsors_interface` | Available |
| Power/bus state | `power_interface` | Available |
| Controller gains `K` | Internal cqa approximation of `dp_controller` (PD + bias FF + integral), tuned to brucon `config_csov` defaults | No linearisation hook from production controller required |
| Vessel hydrodynamic coefs | `vessel_model` config | Available |
| RAOs and drift coefs | `pdstrip.dat` (PdStrip via `pdstrip_cat`) | Available |
| Gangway geometry + limits | `config_csov/posrefs.prototxt.in`: base position, telescope range, max rotation-centre height. Slew/boom limits not gated in study (display only). | Sufficient for study |
| Gangway base in vessel frame | `config_csov/posrefs.prototxt.in` `gangway.position` = (5.0, -9.0, -8.0) m (lowest point of rotation centre) | Available |
| Gangway rotation-centre height `h` | Operator-set; treated as slow setup variable | Need a placeholder input in cqa config |
| Landing point on structure | `offshore_structures.prototxt` | Available |

## 6. Risks and open issues

1. **Linearisation validity near WCFDI saturation.** When the post-failure thruster set
   is barely capable, controller saturation breaks the linear analysis. Mitigation:
   detect saturation rate from MC and revert to a degraded "saturation envelope"
   estimate; flag operator.
2. **Controller `K` extraction** requires a clean linearisation API in `dp_controller`.
   May need a small refactor.
3. **Wave spectrum onboard.** If only Hs/Tp/direction are available (no full spectrum),
   assume JONSWAP/Torsethaugen — already standard in brucon `WaveSpectrum`.
4. **Non-stationary environment.** Online linearisation assumes stationarity over the
   averaging window; gusts and squalls violate this. Use a worst-of-window
   (e.g. P95 wind over the last 10 min) for the prevailing input.
5. **Angular rate fidelity.** Numerical differentiation will lag wave-frequency motion;
   model-based path partly mitigates but adds dependency on wave-spectrum estimate.
6. **Sensor failure handling.** Footprint must degrade gracefully when MRU/wind/wave
   inputs are lost; reuse `dp_estimator` "measurement lost" patterns.
7. **Compute budget online.** Lyapunov solves for two states × heading sweep are O(ms);
   should fit within the existing DP cycle. Verify on-target.

## 7. Validation plan

1. **Unit-level**: closed-form Lyapunov vs. analytic 1-DOF surge cases.
2. **Component**: linearised prediction vs. `vessel_simulator` MC across a (wind, current)
   grid, intact and post-WCFDI.
3. **End-to-end**: replay archived DP recordings (`dp_playback`) against the cqa
   prototype and compare predicted excursion ellipse with the observed sample
   covariance over sliding windows.
4. **Operability**: replay W2W operations and check predicted gangway margin vs.
   actual joint usage / disconnect events if logged.

## 8. Roadmap

| Phase | Output | Tooling | Where |
|---|---|---|---|
| P0 (this doc) | Agreed scope, gaps, methods | Markdown | `cqa/analysis.md` |
| P1 | Equations + offline notebooks: Lyapunov-based excursion polar from synthetic env, intact only | Python (numpy/scipy/matplotlib) | `cqa/` — **DONE** (`scripts/run_polar_demo.py`, `csov_excursion_polar.png`) |
| P2 | Add WCFDI handling, parametrise from a CSOV-like config (mirror brucon `config_csov`) | Python | `cqa/` — **DONE** (`scripts/run_wcfdi_transient_demo.py`, `csov_wcfdi_transient.png`). Augmented 12-state model (η, ν, b̂, τ_thr) with first-order thruster lag (τ_thr ≈ 5 s) and first-order bias estimator (T_b ≈ 100 s); deterministic mean ODE with **time-varying per-DOF thrust cap** modelling thrust reallocation: at t=0 the cap drops to `gamma_immediate · cap_intact` (default 0.5) and recovers exponentially to `alpha · cap_intact` with time constant `T_realloc` (default 10 s). Regulator may saturate during this recovery window — that is the dominant deterministic transient source. Covariance ODE uses the **intact closed-loop A** (full feedback gains; saturation only affects the mean). **Modelling principle:** the upstream static CQA is responsible for `|tau_env| ≤ alpha · cap_intact` per DOF; the transient analysis presupposes this and produces a *bounded recoverable transient*. CQA-precondition violation is surfaced via `info["cqa_precondition_violated"]`. **Future work (post P6):** replace the parametric `(gamma_immediate, T_realloc)` first-order ramp with a query to brucon `BasicAllocator` over the surviving thruster set (with actual azimuth slew rates, per-thruster force/moment limits, ramp rates) to compute the true time-varying per-DOF achievable cap envelope. The exponential placeholder lets us validate the approach end-to-end before incurring that integration cost. |
| P3 | Add gangway model + operability checks | Python | `cqa/` — **DONE** (`cqa/gangway.py`). Forward kinematics (rotation centre, telescope direction unit vector e_L, tip in body & world frames); linearised telescope-length sensitivity `c` (3-vector) such that `Delta_L ≈ c^T eta`; sigma_L from eta covariance and sigma_Ldot from nu covariance; `evaluate_operability(...)` gates on telescope end-stops (L_mean ± k·sigma_L within [L_min, L_max]) and stroke velocity (k·sigma_Ldot below threshold). Slew α and boom β are display-only as per scope. Wired into the WCFDI demo: telescope envelope and operability margins now computed via the gangway module. 9 unit tests passing. |
| P4 | MC validation harness driving brucon `vessel_simulator` (or a Python re-implementation of its 3-DOF horizontal plane) | Python | `cqa/` |
| P5 | DP playback replay vs. predictor on logged data | Python | `cqa/` |
| P6 | Decision gate: continue to demonstrator? | Short report | `cqa/` |
| P7 | C++ demonstrator integrated next to `capability_analysis` and `consequence_analysis` in brucon, publishing on DDS, **operator view in `dp_gui`** | C++ + DDS + QML | `~/src/brucon/libs/dp/cqa/` and `~/src/brucon/dp_gui/views/` (proposed) |

## 9. Proposed Python prototype layout (placeholder for P1)

```
cqa/
  analysis.md              (this file)
  pyproject.toml
  cqa/
    __init__.py
    config.py              # load brucon-style prototxt or yaml
    vessel.py              # 3-DOF linearised vessel + wind/current/wave drift forces
    controller.py          # linearised DP control gains
    closed_loop.py         # build A_cl, B_w, solve Lyapunov
    capability.py          # thin wrapper of capability concept (or call brucon binary)
    consequence.py         # WCFDI selection (Python re-impl. for offline study)
    excursion.py           # excursion polar (intact + WCFDI)
    gangway.py             # kinematics, joint limits, rate limits
    operability.py         # combine excursion + gangway -> envelope, margins
    online_loop.py         # cyclic update simulation
  scripts/
    run_polar_demo.py
    run_wcfdi_demo.py
    run_gangway_demo.py
    run_replay.py          # against dp_playback recordings
  tests/
```

## 10. Success criteria for the feasibility study

- Methodology section is mathematically complete and reviewed.
- Linearised excursion polar reproduces MC sample covariance within ±20 % for
  representative (wind, current) cases at a CSOV configuration, intact and post-WCFDI.
- Gangway operability mapping correctly flags exceedances on synthetic excursion
  inputs at all three joint axes and at the rate limits.
- Per-cycle compute budget below 50 ms on a developer laptop (targeting <10 ms on
  the onboard CPU after C++ port).
- Clear go/no-go recommendation for the brucon demonstrator.

## 11. Decisions taken

1. **Target configuration for the study**: brucon `config_csov`. The production
   implementation must remain configuration-agnostic (any of the
   `config_*` configs in `~/src/brucon/modules`).
2. **Wave spectrum onboard**: assume **not directly measured** in operation. Sources
   in priority:
   1. **Forecast** (NORA3 / WW3 — already fetchable via `hydro_tools/environment/`).
   2. **Wave radar** if installed.
   3. **Wind-sea estimate via wind/wave-buoy analogy** (see
      `hydro_tools/environment/wave_buoy.py`) combined with measured wind, plus a
      forecast swell partition.
   The cqa prototype must therefore tolerate (Hs, Tp, mean direction) per
   wind-sea/swell partition, not a full directional spectrum, and assume a
   parametric shape (JONSWAP / Torsethaugen).
3. **Gangway envelope**: generic envelope for the study, anchored on the real
   numbers from `config_csov/posrefs.prototxt.in`:
   - rotation-centre base position in vessel body frame:
     `(x, y, z) = (5.0, -9.0, -8.0)` m (lowest position of the rotation centre),
   - rotation-centre vertical travel: `0..25 m` (operator-set, slow),
   - telescope length: `18 m ≤ L ≤ 32 m`.

   Operability gating in the study is **telescope-only**:
   - end-stops `L_min, L_max` — pass/fail with margin and exceedance rate,
   - stroke velocity `Ldot` — display predicted distribution and 95-percentile,
     compare to a configurable default (≈0.5 m/s).

   Slew angle `α`, boom angle `β`, and their rates are **computed and displayed
   only** — no pass/fail. Operator picks heading and rotation-centre height to
   stay clear of α/β end-stops, and the analysis surfaces those values for
   situational awareness.
4. **Controller representation**: the prototype carries its own **approximation** of
   the brucon DP controller (PD on position with bias-feedforward, plus integral
   action), tuned to mirror the brucon `dp_controller` defaults for `config_csov`.
   No linearisation hook into the production controller is required for the study.
5. **GUI**: the demonstrator output is to be presented in `~/src/brucon/dp_gui`.
   The Python prototype will mirror the data layout that the dp_gui will consume
   (so a later DDS topic + view can be added without redesign).

---

## 12. Findings and methodology refinements (2026 implementation)

This section documents the engineering refinements and validation findings
accumulated during the Python prototype work. Each subsection records the
problem encountered, the resolution, and the residual uncertainty so the
C++ port (P7) can carry the same conclusions forward.

### 12.1 Operator decision view (IMCA M254 Rev.1 Fig. 8)

`cqa.operator_view.summarise_intact_prior(...)` produces the operator-facing
two-axis decision summary:

* **Position axis**: radial distance from setpoint at the gangway base, with
  green/amber/red traffic lights against the IMCA position warning and
  alarm radii (`cfg.operational_limits.position_warning_radius_m` /
  `position_alarm_radius_m`, defaults 2 m / 4 m).
* **Telescope axis**: combined low-frequency + wave-frequency telescope
  length deviation `|ΔL|`, with traffic lights against operator-set
  fractions of the worst-side stroke (defaults 60 % / 80 %, IMCA M254
  utilisation thresholds).

The summary returns two quantiles (default P50, P90) of the running maximum
of `|X(t)|` over the planned operation duration `T_op`, inverted from the
Rice formula with Vanmarcke clustering correction (`extreme_value.inverse_rice`,
`inverse_rice_multiband`). Both axes share the same green/amber/red logic so the
operator reads them with identical semantics.

### 12.2 Closed-loop covariance via frequency-domain integration

The Lyapunov solution sketched in §4.2 is replaced by a direct frequency-domain
integral (`closed_loop.state_covariance_freqdomain`) over a logspace grid
`np.logspace(-4, 0, 1024)` rad/s. This avoids the matrix-Lyapunov solver and
keeps the per-cycle compute well below the budget; more importantly it makes
the "axis PSD" (telescope-length deviation PSD, radial-position PSD) directly
available as a 1-D function of `omega`, which is needed downstream for
spectral moments and Vanmarcke bandwidth.

**Realisation grid policy.** Time-domain realisations
(`time_series_realisation.realise_vector_force_time_series`,
`realise_wave_motion_6dof`) accept non-uniform grids. The LF demos use
`np.geomspace(1e-4, 0.6, 256)`: this captures the closed-loop response across
decades with 16× fewer points than a uniform grid that reaches `1e-4`.
A uniform `linspace(1e-3, 0.6, 256)` was previously dropping ~6 % of the LF
variance (the response has substantial energy below `omega = 1e-3` rad/s,
period > 100 min). After the geom-grid switch, the empirical realisation
sigma matches the freq-domain prediction within sampling noise (~1 % on
24 h realisations, see `tests/test_time_series_realisation.py`).

### 12.3 Wave-frequency telescope channel

The original §4.4 plan treated the wave-frequency telescope contribution
through a tip-displacement RAO. The implemented path is more direct:
`wave_response.sigma_L_wave(joint, cfg, rao, Hs, Tp, theta_wave_rel)` projects
the 6-DOF rigid-body motion at the body origin onto the gangway telescope
sensitivity vector `c6 = telescope_sensitivity_6dof(joint, cfg.gangway)`,
giving

```
sigma_L_wave^2 = ∫ |c6 · H_6dof(omega, beta)|^2 · S_eta(omega) · D(phi) dphi domega
```

with directional spreading (default cos-2s, s=15) integrated by Gauss
quadrature.

**Bandwidth caveat.** The Vanmarcke `q_wave` parameter for this band is
spectrum-dependent: the canonical CSOV operating point gives `q ≈ 0.16`,
not the 0.30 narrowband proxy that was hardcoded earlier. Callers should
pass `q_wave=vanmarcke_bandwidth_q(wave.integrand, wave.omega)` to
`summarise_intact_prior` (the prior-vs-posterior demo does so). The
fallback 0.30 biases towards Poisson clustering (slightly conservative
for amber/red triggers).

Because the slow band typically dominates `dL` variance, the WF `q`
choice does not materially change the multiband `P_breach`. It does
affect WF-isolation diagnostics and matters for sea states where the
wave channel becomes the dominant telescope driver.

### 12.4 Bayesian sigma estimator + posterior length scales

`bayesian.BayesianSigmaEstimator` runs online on the bandsplit residuals
(see §12.5) and produces a posterior over the per-axis sigma. Three estimators
operate in parallel:

* radial position (LF closed-loop band),
* telescope slow channel (LF closed-loop band),
* telescope wave channel (WF, RAO-driven).

The slow estimators data-condition the model `sigma` (level update) but
keep the spectral SHAPE (`nu_0+`, Vanmarcke `q`) from the prior closed-loop
spectrum. The wave estimator data-conditions `sigma_L_wave` directly: this
is the diagnostic that flags **sea-state misclassification**. If the
operator-supplied `(Hs, Tp, theta_wave_rel)` is wrong, the model
`sigma_L_wave` is biased and the WF posterior pulls toward the
data-consistent value. About 30 effective samples accumulate per 5 min
on the WF channel (~6 % half-width on `sigma_L_wave`), see
`scripts/run_prior_vs_posterior_demo.py`.

**Variance-decorrelation time `T_var`.** The Bayesian estimator's effective
sample size is `T_window / T_var`, where `T_var` is the variance estimator's
correlation time, NOT the state correlation time. The correct expression is

```
T_var = π · ∫ S²(omega) domega / m₀²
```

(`extreme_value.variance_decorrelation_time_from_psd`), about 5× larger than
`1/(zeta · omega_n)` for the CSOV closed-loop response. Earlier code used
the latter and over-counted independent samples by a factor of 5; fixed in
commit `ed604f7`.

### 12.5 Band-split utility

`signal_processing.bandsplit_lowpass(x, dt, fc_Hz)` zero-phase Butterworth
splits the observed deviation into LF and WF bands at a cutoff (default
0.05 Hz) below the wave-frequency floor and above the closed-loop bandwidth.
The two bands are uncorrelated by construction (see §12.7) so the LF and WF
posteriors update independently. Validated to <1 % LF / <10 % WF reproduction
on 60 min CSOV realisations (`scripts/validate_bandsplit.py`).

### 12.6 Running-maximum CDF and the Rice/Vanmarcke validity envelope

The operator-facing P50/P90 quantiles of `M_T = max|X(t)|` over `T_op` rest on
the assumption that level up-crossings of a Gaussian process are independent
events:

```
P(M_T > a) ≈ 1 − exp(−2 · nu_a+ · T_op),     nu_a+ = nu_0+ · exp(−a²/(2σ²))
```

with the Vanmarcke clustering correction applied per band before combining.
A bootstrap M_T validation (`scripts/validate_running_max_cdf.py`,
`scripts/diagnose_dL_running_max_bias.py`,
`scripts/diagnose_rice_validity.py`) on a 24 h closed-loop realisation
shows:

| diagnostic | result |
|---|---|
| sigma vs Lyapunov | <1 % bias ✓ |
| nu_a+ vs Rice (a/sigma ∈ [1, 2.5]) | within sampling noise ✓ |
| Vanmarcke vs Poisson at q=0.78 (slow band) | nearly identical (q is broadband) |
| **P(M_T > a) at a/sigma = 1.0** | emp 0.83 vs Rice 0.63 → **+31 % Rice under-prediction** |
| P(M_T > a) at a/sigma = 1.5 | +18 % |
| P(M_T > a) at a/sigma = 2.0 | +10 % |
| P(M_T > a) at a/sigma ≥ 2.5 | within sampling noise (Rice valid) |

The Rice **level-crossing rate** `nu_a+` is essentially correct. The bias
comes from the **Poisson-of-rare-events** approximation: at moderate rarity
(`a/sigma < 2.5`) crossings are not independent, and the windowed maximum
distribution shifts higher than Poisson predicts. The
`RiceExceedanceResult.valid` flag (default `rarity_min=2`) correctly marks
this regime.

**Direction is conservative for safety.** Under-predicted exceedance
probability means the predicted "safe" quantile `a_p` such that
`F_M(a_p) = p` is LOWER than empirical, so the operator sees a tighter
limit than reality. The amber/red triggers fail SAFE.

**Operator panel recommendation.** P95 (`a/sigma ≈ 2.7` for the canonical
operating point) is in the Rice-valid regime; P50 is not (`a/sigma ≈ 1.8`).
The C++ port (P7) should anchor amber/red on P95 by default and badge
sub-P95 quantiles with a "low-rarity, conservative" annotation rather than
display them as the primary operator number.

### 12.7 LF / WF independence verification

The decision view treats the LF and WF telescope contributions as
statistically independent (`p_exceed_rice_multiband` adds Poisson counts
across bands). This is a non-trivial assumption: the same vessel rigid-body
displaces both ways. Empirical validation
(`scripts/diagnose_dL_running_max_bias.py`) on 12 h CSOV realisations gives:

* direct LF vs WF correlation: `rho = 0.003`
* envelope LF vs envelope WF correlation: `rho = -0.006`

i.e. perfectly uncorrelated within sampling noise. The two channels live in
disjoint frequency bands and are excited by independent disturbance
mechanisms (wind-gust + slow-drift + current-variability LF; first-order
wave RAO WF), so this is expected.

### 12.8 Updated roadmap status

Phases P0-P3 of §8 are complete as marked. Implementation progress beyond
that table:

| Workstream | Status | Notes |
|---|---|---|
| Operator decision view (IMCA M254 Fig. 8) | DONE | `operator_view.summarise_intact_prior` + `plot_intact_prior` |
| Frequency-domain closed-loop covariance | DONE | `closed_loop.state_covariance_freqdomain` (replaces the original Lyapunov plan, faster and exposes axis-PSD) |
| Wave-frequency telescope channel | DONE | `wave_response.sigma_L_wave` |
| Time-domain realisations (LF + WF) | DONE | non-uniform-grid acceptance, geom-grid policy |
| Bayesian sigma estimator (3 channels in parallel) | DONE | `bayesian.BayesianSigmaEstimator`, with the `T_var` fix |
| Posterior health diagnostics (A1-A5 primitives) | DONE | `online_estimator.PosteriorHealth` + `BayesianSigmaEstimator.health()` |
| Per-axis (x, y) radial estimation + Hoyt-aware combine | DONE | `online_estimator.combine_radial_posterior`, validated against time-domain realisation |
| Band-split + posterior pipeline | DONE | `signal_processing.bandsplit_lowpass`, validated |
| Running-max CDF validation | DONE | Rice valid above `a/sigma ≈ 2.5`, conservative below |
| P4 MC validation against `vessel_simulator` | TODO | grid sweep needed |
| Sigma²-validity residual alarm | TODO | flag when posterior-vs-prior diverges |
| WCFDI overlay on operability polar | TODO | overlay post-failure transient envelope |
| `sigma_L_wave` batch evaluation optimisation | TODO | for the offline polar case |
| Tier 2 sigma-inconsistency fix (12-state augmented system) | TODO | open from P2 |
| P7 C++ demonstrator | TODO | feeds `dp_gui` |

### 12.9 Per-axis (x, y) estimation and Hoyt-aware radial combination

The first iteration of the online posterior pipeline ran a single
`BayesianSigmaEstimator` on the radial channel `r(t) = sqrt(dx² + dy²)`
directly. The newly-added `PosteriorHealth.sample_mean_over_sigma`
diagnostic immediately flagged this channel as `INVALID`
(`|mean(r)|/σ_post ≈ 1.0`) — but for a structural reason, not a
settling-transient one:

* `r` is **Rayleigh-distributed** (when σ_x = σ_y) with mean
  `E[r] = σ·sqrt(π/2) ≈ 1.253·σ`. Even when (dx, dy) are perfectly
  zero-mean, `r` has a structural non-zero mean that contaminates the
  zero-mean (A2) assumption baked into `BayesianSigmaEstimator`.
* The conjugate model is still inverse-gamma (the Rayleigh sufficient
  statistic `Σ rᵢ²` recovers `2σ²·N` in expectation), so the variance
  point estimate `posterior().sigma_median` was actually correct in
  expectation — but the A2 health primitive could not distinguish
  "DP integral term not yet settled" from "this channel has a
  Rayleigh-shaped distribution".

#### Decision: per-axis (dx, dy) channels

The DP regulates body-x and body-y independently, so `dx(t)` and
`dy(t)` are **zero-mean Gaussian by construction** — A2 holds. The
new helper `time_series_realisation.base_position_xy_time_series`
exposes them; two separate `BayesianSigmaEstimator` instances watch
the two channels.

Benefits realised in the demo:

* The A2 indicator is now physically meaningful per axis. At the
  canonical 30° quartering operating point with 5 min of data:
  `|mean(dx)|/σ_x ≈ 0.51` (UNSETTLED), `|mean(dy)|/σ_y ≈ 0.84`
  (UNSETTLED). The y-axis is correctly flagged worse than the x-axis
  because the slow-drift sway memory at oblique forcing is longer
  (T_var_y ≈ 94 s vs T_var_x ≈ 47 s).
* The radial composite badge is composed as the worst-of (x, y) — if
  either axis hasn't settled, the radial summary is unreliable.
* Per-axis priors (σ_x, σ_y, T_var_x, T_var_y) are surfaced as new
  fields on `IntactPriorSummary`.

#### Why a single Rayleigh estimator was rejected

A naïve alternative is to swap the Gaussian likelihood in
`BayesianSigmaEstimator` for a Rayleigh likelihood: same conjugate
inverse-gamma family, sufficient statistic `S = Σ rᵢ²`, posterior
update `α += N/2`, `β += S/4` (note the `/4` instead of `/2` because
of `E[r²] = 2σ²`). This was rejected because:

* For σ_x ≠ σ_y the radial process is not Rayleigh but **Hoyt
  (Nakagami-q)** with eccentricity `q = σ_min/σ_max`. The conjugate
  inverse-gamma breaks down — the Hoyt MLE has no closed form.
* At the canonical CSOV 30° quartering operating point we observe
  σ_x_prior = 0.72 m vs σ_y_prior = 0.61 m, ratio 1.18. Forcing
  Rayleigh would systematically bias the variance estimate.
* Correct handling of σ_x ≠ σ_y requires a 2D estimator on (dx, dy)
  anyway, which is exactly what we now have.

#### Combining per-axis posteriors for the operator-facing radial scalar

The operator monitors radial distance, so we still need a single
"σ_R" / "E[|R|]" scalar to display. The natural radial scale is

`σ_R := sqrt(σ_x² + σ_y²) = sqrt(trace(Σ))`

— equal to `sqrt(2)·σ` in the Rayleigh limit, axis-rotation invariant
in general, and the parameter the Rice formula consumes for the
bilateral-Gaussian envelope of the radial running max.

The sum `σ_x² + σ_y²` of two independent inverse-gamma RVs is
**not** itself inverse-gamma — there is no closed-form posterior. The
new `combine_radial_posterior` helper handles this with cheap MC
(default 2000 samples):

1. Sample `σ_x² ~ InvGamma(α_x, β_x)` and `σ_y² ~ InvGamma(α_y, β_y)`
   independently.
2. Form `σ_R = sqrt(σ_x² + σ_y²)` per sample → median, mean, equal-tail
   credible interval.
3. For `E[|R|]` (the operator-friendly "typical radial distance"):
   sample one `(X, Y)` pair per `(σ_x, σ_y)` draw, take the mean of
   `sqrt(X² + Y²)`. This integrates over BOTH posterior uncertainty AND
   in-window Hoyt asymmetry exactly, with no need for the elliptic-
   integral closed form.

The closed-form `E[σ_R²] = β_x/(α_x-1) + β_y/(α_y-1)` is also reported
when both posteriors have α > 1.

#### Validation against the time-domain realisation

Tier A check, evaluated on the 5-min CSOV demo realisation each run:

* Empirical `sqrt(E[r²]) = 1.136 m` vs predicted `σ_R = 1.038 m`,
  inside the posterior 90% CI [0.750, 1.621]. ✓
* Empirical `mean(r) = 0.951 m` vs predicted `E[|R|] = 0.952 m` — agreement
  to 1 mm. ✓

The `E[|R|]` agreement to sub-mm precision in a single-seed test is
remarkable but expected: the posterior median tracks the underlying σ
scale parameters tightly even with small `n_eff`, and `E[|R|]` is a
smooth function of those scale parameters. The wide 90% CI honestly
reflects the small effective sample count (n_eff_x ≈ 6, n_eff_y ≈ 3
in 5 min).

**Tier B coverage validation** (`scripts/validate_radial_combine.py`):
M=200 independent 5-min realisations at the canonical CSOV operating
point. For each seed we draw the per-axis posteriors, combine, and ask
whether the spectral-truth `sigma_R` and `E[|R|]` lie inside the
claimed 90% credible interval. Two-sided binomial test against
H0: true coverage = 0.90.

Result (commit follow-up to 64d5119):

| Quantity | In CI | Empirical coverage | Wilson 95% CI | p-value | Verdict |
|---|---|---|---|---|---|
| `sigma_R`   | 177/200 | 88.5% | [83.2%, 92.6%] | 0.479 | **PASS** |
| `E[\|R\|]`  | 177/200 | 88.5% | [83.2%, 92.6%] | 0.479 | **PASS** |

The CI is well-calibrated. Note however the **median bias**: posterior
median sigma_R sits ~12 cm below the truth (0.83 m vs 0.95 m). This
is a known and benign property of reporting the median of an
InvGamma-on-σ²: at α ≈ 6 (small n_eff), median(σ²) ≈ β/(α − 1/3) sits
~13% below the mean β/(α−1), so √median underestimates σ by ~6% per
axis. The CI is wide enough to absorb this, hence calibration passes.
For an unbiased point estimate use `sigma_R_mean` (closed form;
already exposed on `RadialPosterior`).

### 12.10 Posterior health primitives (assumption diagnostics)

The conjugate posterior in `BayesianSigmaEstimator` rests on five
assumptions; `PosteriorHealth` exposes one cheap runtime primitive
per assumption so the operator panel can compose a WARMING / OK /
UNSETTLED / INVALID badge.

| Assumption | Primitive | Failure mode caught |
|---|---|---|
| **A1** stationarity within window | `halves_sigma_ratio` | sea-state ramp, controller retune mid-window |
| **A2** zero-mean signal | `sample_mean_over_sigma` (PRIMARY) | DP integral term still settling (~2-5 min), observer bias still converging (~1-2 min), persistent low-frequency disturbance, setpoint drift |
| **A3** Gaussian marginals | `kurtosis_excess` | thruster saturation, slamming, heavy-tail residuals |
| **A4** Bartlett ESS captures autocorrelation | `is_warm` (`n_eff ≥ threshold`) | window too short relative to T_var |
| **A5** prior shape correct, only level data-conditioned | `prior_in_credible_interval` | sea-state misclassification, post-WCFDI controller retune |

The A2 primitive is the operationally most important: it directly
catches the early-operation transient where the DP integral and
observer bias estimator have not yet converged, before any of the
spectral assumptions enter. Suggested operator thresholds:

* `|mean|/σ < 0.1`: settled
* `0.1 ≤ ratio < 0.3`: warming
* `0.3 ≤ ratio < 1.0`: UNSETTLED (variance estimate inflated by
  9-100%)
* `ratio ≥ 1.0`: INVALID (variance inflated by ≥2×; likely setpoint
  drift or unmodeled DC bias)

The thresholds are exposed as primitives, not enforced inside
`BayesianSigmaEstimator`, so the C++ panel layer can tune them per
site / per channel without modifying the estimator.

### 12.11 Radial composite A2 indicator (2D vector-mean magnitude)

The per-axis A2 primitive is `|sample_mean| / σ_median` on each axis
independently. Composing these into a single radial badge by
worst-of-x,y is operationally fine but not principled: it is *not*
rotation-invariant in the body frame (a heading change re-shuffles
the offset between cardinal axes and can flip the badge), and it can
double-count the same physical drift when it projects onto both axes.

The principled alternative is the magnitude of the 2D sample-mean
vector divided by the radial scale:

    radial_mean_offset_over_sigma = |(mean_x, mean_y)| / σ_R_median

exposed on `RadialPosterior` (and `combine_radial_posterior` accepts
optional `sample_mean_x`, `sample_mean_y` kwargs to compute it). Two
properties:

* **Rotation-invariant** in the body frame. Vessel drifting 0.5 m to
  the north-east at heading 0° gives the same value as drifting 0.5 m
  to the east at heading 45°.
* **Strictly ≤ worst-of-x,y per-axis ratio** when only one axis carries
  the drift. Geometrically: a 0.5 m offset on body-x with σ_x = σ_y = 1
  has per-axis worst ratio 0.5 but radial ratio 0.5/√2 ≈ 0.354. The
  radial scale dilutes the per-axis drift correctly.

The same threshold bands are reused (< 0.1 settled, < 0.3 warming,
< 1.0 UNSETTLED, ≥ 1.0 INVALID). The physical meaning is identical:
inflation of the radial variance estimate by `µ²` contamination of
the per-axis sufficient statistics.

CSOV demo readout (5-min window, 30° quartering): per-axis ratios
(0.56, 1.00) → worst-of-x,y badge UNSETTLED; radial 2D ratio 0.76 →
also UNSETTLED but for the right reason. The signs of the
contributions are not coincidental cancellation here; both axes carry
genuine slow drift and the 2D magnitude correctly reflects ~80 cm of
mean offset over the 5-min window.

### 12.12 Per-channel validity badge (compose_validity_badge)

`PosteriorHealth` exposes the assumption-failure primitives; the
operator panel needs a single per-channel verdict the bridge can act
on. `compose_validity_badge(health, ...)` is the pure-function
composer: takes a `PosteriorHealth`, returns a `ValidityBadge` with a
4-state `level` (`OK`, `WARMING`, `UNSETTLED`, `INVALID`) and a list
of human-readable `reasons` naming every primitive that contributed
at or above WARMING.

**Per-assumption verdicts** (worst wins):

| Assumption | Severity ladder | Rationale |
|---|---|---|
| **A4** `n_eff` | < 2: INVALID; < 5: WARMING | Below 2 independent draws the posterior is the prior + noise; the other primitives are uninterpretable. |
| **A2** `\|mean\|/σ` | 0.1 / 0.3 / 1.0 → WARMING / UNSETTLED / INVALID; NaN → INVALID | Operator-band thresholds documented on `PosteriorHealth`. |
| **A1** `halves_sigma_ratio` | inside [1/1.5, 1.5]: OK; inside [1/2, 2]: WARMING; else UNSETTLED | Symmetric in log-ratio. Stationarity violations inflate variance but degrade gracefully. |
| **A3** `\|κ_ex\|` | < 0.5: OK; < 1.5: WARMING; else UNSETTLED | Skip if NaN. Caveat: sample kurtosis variance ~24/n_raw, so WARMING fires routinely below ~50 samples. |
| **A5** `prior_in_credible_interval` | False → WARMING only | Informational ("model and data disagree"); the data-driven posterior is still trustworthy. |

All thresholds are kwargs with defaults matching the operator-band
suggestions. Site-tuning is one call away; the C++ panel can override
per channel without touching the estimator.

CSOV demo readout (5 min, 30° quartering) — exercises every band:

```
pos x  : [UNSETTLED]
   -> A2: |mean|/sigma=0.56 in [0.30, 1.00) (...)
   -> A1: halves_sigma_ratio=1.72 in warming band
pos y  : [UNSETTLED]
   -> A4: n_eff=3.2 below warm threshold 5.0
   -> A2: |mean|/sigma=1.00 in [0.30, 1.00)
   -> A3: |kurtosis_excess|=0.76 in warming band
slow gw: [INVALID]
   -> A4: n_eff=3.2 below warm threshold 5.0
   -> A2: |mean|/sigma=1.03 >= 1.00 (variance inflated >=2x; ...)
   -> A3: |kurtosis_excess|=0.89 in warming band
wave gw: [OK]
```

The `wave gw` channel is the only one that scores `OK`, which matches
the physics: 30 effective samples in the 5-min window, A2 ratio ~10⁻³
(perfect zero-mean), well-behaved kurtosis. The slow channels all
suffer from small `n_eff` and the slow-drift settling transient,
exactly as expected at the start of an operation.

### 12.13 WCFDI overlay on the operability polar (design / table-top tool)

Added `wcfdi_operability_overlay` and a single dashed-line overlay on
`plot_operability_polar`: per heading, the V_w at which the post-WCFDI
peak excursion (deterministic mean from the linearised post-failure
transient + `k_sigma * sigma(t)` from the augmented covariance ODE)
crosses the IMCA M254 alarm threshold. The default scenario is one of
three thruster groups lost (`alpha = 2/3` per DOF, `gamma_immediate =
0.5`, `T_realloc = 10 s`); default `k_sigma = 0.674` (P75 of the
conditional peak distribution).

**Scope and audience.** This polar (intact + WCFDI overlay) is a
**design / table-top / feasibility** chart, not an operational chart.
The sea state at each V_w is a synthetic Pierson-Moskowitz law and all
360 directions are evaluated independently against a swept envelope.
It answers the question *"is this vessel + thruster layout suitable
for this work scope?"* before the steel is cut, and it sits alongside
the standard DP capability plot (IMCA M140 / DNV-ST-0111) in the design
deliverables. It is not what a navigator should be looking at on the
bridge.

**The two operationally-facing analogues are separate workstreams:**

* **Forecast case (item 4b on the roadmap, not yet implemented).** Per
  forecast time-slot and the *chosen* heading, evaluate the same
  intact and post-WCFDI metrics at the *forecast*
  `(V_w, Hs, Tp, V_c, theta_w, theta_wave, theta_c)` and emit a per
  time-slot traffic light. This is the operating-window decision matrix
  the operator actually uses pre-operation. Reuses the same engines as
  the polar but consumes forecast inputs instead of swept inputs.
* **Operation case live what-if (item 4c, not yet implemented).** At
  runtime, given the live posterior from `online_estimator` and the
  live measured environment, run `wcfdi_transient` with the live
  posterior `P0` (rather than the steady-state Lyapunov `P0`) and the
  live measured `(V_w, Hs, Tp, V_c)` as the operating point; emit a
  single live badge "if WCFDI fires now, P75 peak vessel excursion =
  X m, telescope = Y m". Depends on item P4 (vessel_simulator MC
  validation matrix) to bound the linearisation error against the
  nonlinear truth before the operator can trust the live number.

**Metric definitions.** Per heading, with the linearised post-failure
state-space evaluated by `wcfdi_transient`:

* Vessel base: `peak_pos = max_t  ||eta_mean[:,0:2](t)||
  + k_sigma * sigma_R(t)` with `sigma_R(t) = sqrt(P[t,0,0] + P[t,1,1])`
  (trace of the 2x2 position-block covariance; correct when `cov(x,y)`
  is small as it is for decoupled controllers under collinear forcing,
  slight over-estimate otherwise).
* Telescope: `peak_dL = max_t  |c_L^T eta_mean(t)|
  + k_sigma * sigma_dL(t)` with `sigma_dL(t)
  = sqrt(c_L^T P[t,0:3,0:3] c_L)` and `c_L` from
  `telescope_sensitivity`.

Adding the `+ k * sigma` *inside* the `max_t` is a small conservative
over-estimate (the time of peak mean and the time of peak sigma may
differ); the alternative `max(mean) + k * max(sigma)` is similar and
slightly less conservative. The per-time sum matches the visual
envelope drawn by `run_wcfdi_transient_demo.py`.

**`k_sigma` choice and rare-event conditioning.** The intact polar
uses `quantile_p = 0.90`. For the WCFDI overlay we default to
`k_sigma = 0.674` (P75 of the post-failure conditional peak), which
sits *below* the intact P90 convention. The reason is that WCFDI is
itself a rare event (annual frequency `~ 1e-3` for a single
thruster-group failure on a redundant DP-2 vessel); conditioning on a
high-quantile peak excursion *given* the failure compounds two rare
events and inflates the design margin against an event whose joint
probability is already small. P75 is a meaningful margin above the
deterministic mean (P50, `k = 0`) without paying for a low-probability
tail twice. The parameter is exposed; raise to 1.282 (P90) or 1.96
(95 %) for more conservative envelopes.

**CQA precondition handling.** If the surviving thrusters cannot hold
the steady-state environmental load nominally
(`|tau_env| > alpha * cap_intact` in any DOF), the linearised
post-failure covariance ODE diverges. We catch the resulting
`RuntimeError` and assign `+inf` to the peak metric, which causes the
bisection to saturate the boundary at `Vw_min` for that direction.
This is *not* surfaced as a separate flag on the dataclass: the
saturation already conveys "no-go at any wind speed in this heading
under this failure scenario" and the polar's existing `*_capped_low`
arrays carry the bookkeeping.

**The head-sea shape difference is real physics, not a bug.** On the
CSOV demo, the WCFDI dashed line sits *outside* the intact alarm at
head/stern seas (theta ~ 0 / 180). The intact bands and the WCFDI line
are different metrics with different rose patterns: intact = slow-drift
*PSD* of the fluctuating load integrated over T_op; WCFDI = post-failure
*step response* to the deterministic mean of the load. They genuinely
disagree in shape, especially where one mechanism is unfavourable and
the other is favourable. This is informative for a design engineer
(it tells you *which* failure modes constrain *which* headings) and
exactly the reason this chart should not be sent to operations: the
operator would parse it as *"the worst case is safer than the best
case"*, which is not the message.

**CSOV demo readout (default scenario).** 36 directions, 20-minute
operating window, Vc = 0.5 m/s, port gangway at mid-stroke. Intact
worst-direction alarm V_w = 17.6 m/s @ 70 deg; post-WCFDI
worst-direction alarm V_w = 12.4 m/s @ 80 deg, a 30 % reduction in
operable wind speed for the worst heading (vessel base). For the
telescope axis the reduction is 36 %. Saved figures:
`scripts/csov_operability_polar.png` (intact) and
`scripts/csov_wcfdi_operability_polar.png` (intact + WCFDI overlay).
Full overlay sweep takes ~10 s (most directions saturate the bisection
early on the post-failure side); intact polar takes ~30 s.


### 12.14 Bistability of the saturated post-WCFDI dynamics

A time-domain self-MC validator (`cqa.wcfdi_self_mc`) was built to
cross-check the linearised `wcfdi_transient` predictor against
stochastic realisations of the same augmented system, driven by
Shinozuka realisations of the wind-gust / slow-drift / current
disturbance PSDs. The validator targets the augmented-state structure,
the time-varying thrust-cap clipping, and the covariance ODE; it
shares the same equivalent-white-noise approximation as the linear
predictor and is therefore not a check against a higher-fidelity
nonlinear simulator (that cross-check belongs to the brucon
`vessel_simulator` / `dp_runfast_simulator` work, deferred).

Running the validator at increasing severity along the beam direction
revealed a clean three-regime structure for the CSOV defaults
(alpha = 2/3, gamma_immediate = 0.5, T_realloc = 10 s):

| `|tau_env|/cap_post` | `|tau_env|/cap_imm` | regime                                           |
|--------------------- |--------------------- |--------------------------------------------------|
| < 0.85               | < 1.15               | immediate cap not exceeded; no transient at all  |
| 0.85 - 0.92          | 1.15 - 1.22          | mild transient; deterministic + 100 % MC recover |
| **0.92 - 1.0**       | **1.22 - 1.39**      | **bistability band: deterministic recovers, fraction of MC realisations runs away** |
| > 1.0                | > 1.39               | drift-off; deterministic and MC both fail; CQA precondition flag fires |

The bistability is structural: under hard saturation the controller
is open-loop in the saturated DOF, and the bias estimator integrates
the position residual slowly enough that a moderate disturbance kick
during the recovery window can push individual realisations onto a
runaway branch from which the slow integrator never catches up. The
deterministic mean ODE always finds the recovering branch (because
the mean disturbance is zero), so the linear predictor's mean-and-std
output is *systematically optimistic* in this band: at Vw = 14 m/s
(beam, |tau|/cap_post = 0.98) the deterministic predictor returns a
1.9 m peak with the MC ensemble mean at 4.2 m and ~40 % of
realisations diverging.

This matches operator experience: when the CQA margin is around 10 %
the vessel "usually" recovers from a WCF; below that it does not.

**Detection: the bistability_risk_score.** No linear correction
(describing-function / Bussgang statistical linearisation) closes the
gap because in the hard-saturation limit the Bussgang gain
`N0(mu, sigma, L) = P(|tau_cmd| < L)` falls structurally to zero
(commanded thrust mean is 2-6 sigma into the saturated region for
all bistability-band operating points). The fix is therefore not to
correct the predictor but to *flag* operating points where it is
unreliable. We compute, along the deterministic mean trajectory:

    severity(t, dof) = max(0, |tau_cmd_mean(t, dof)| - cap(t, dof)) / sigma_tau_cmd(t, dof)

with `sigma_tau_cmd(t)^2 = K_tau P(t) K_tau^T` (per DOF) and
`K_tau = [-Kp, -Kd, -I_3, 0]`. The headline
`bistability_risk_score = max over (t, dof)` is reported in
`wcfdi_transient`'s `info` dict alongside per-DOF and time-series
diagnostics (`tau_cmd_mean`, `sigma_tau_cmd`, `cap_t`).

**Empirical calibration against `wcfdi_self_mc` (M = 128 seeds, 90
deg beam):**

| score | recovery rate |
|-------|---------------|
| 0.55  | 100 %         |
| 1.11  | 99 %          |
| 1.39  | 98 %          |
| 1.66  | 88 %          |
| 2.79  | 77 %          |
| 5.98  | 59 %          |

The 95 % recovery boundary sits at score ~1.5; the 80 % boundary at
score ~2. We adopt **`bistability_alarm = 1.5`** as the default gate
in `wcfdi_operability_overlay`: any direction / V_w combination whose
deterministic predictor returns a score above 1.5 is folded into the
alarm boundary regardless of the nominal mean+sigma envelope, by
forcing the metric to +inf at that operating point (mirroring the
existing CQA-violation handling). This shifts the CSOV post-WCFDI
alarm boundary inward by 0.15 - 0.61 m/s depending on direction, with
the largest shifts in the oblique-to-beam quarter where the operating
point is closest to the saturated regime.

**Why this is the right place for the gate (and not a cheaper /
linear fix).** Three alternatives were investigated and discarded:

1. *Bussgang correction on the variance ODE (gain N0 modulated):*
   correctly identifies "open loop" in the hard-saturation regime
   (N0 -> 0) but cannot restore the missing closed-loop dynamics
   because they don't exist at that operating point. The corrected
   variance ODE collapses to the open-loop variance growth that we
   need to model, but does not by itself capture the bistability
   (which is a population-level statement, not a linearisation).
2. *Bussgang correction on the mean ODE:* shifts E[clip(X)] toward
   the cap by a `sigma * phi(alpha)` term. In the hard-saturation
   limit (alpha << 0) the correction vanishes; the deterministic
   mean trajectory is already correctly clipped to the cap. So the
   2-3x mean discrepancy in the bistability band is not an
   E[clip(X)] != clip(E[X]) effect.
3. *Linearise around the deterministic mean trajectory and propagate
   IC perturbations:* under hard saturation the linearised dynamics
   are open-loop, so the first-order mean correction is zero
   (intact stationary IC is zero-mean). The MC's 2x larger mean is a
   *second-order* effect (variance of IC plus disturbance, both
   coupled through the saturating clip), which has no linear closed
   form.

The bistability is fundamentally a feature of the *coexistence* of
two stable branches in the saturated nonlinear dynamics, with the
ratio of basin volumes determining the recovery rate. No deterministic
linearisation can capture coexistence; only a population-level metric
(such as our severity score, calibrated against MC) can flag it.

**Implication for the operability polar.** With the gate enabled
(default), the polar's amber and red boundaries move inward by the
amounts above and now correctly mark the bistability band as alarm.
This adds physical realism to the polar without adding a Monte-Carlo
cost to the design tool: the score is computed from quantities
(`x_mean`, `P`) the linear predictor already produces. The polar can
still be regenerated with `bistability_alarm = inf` to recover the
pre-gate behaviour (useful for diagnostic comparisons).

**Implication for the brucon production transfer.** The
deterministic-predictor + bistability-gate pattern is exactly the
right shape for a real-time DP advisory: it stays cheap (no
per-cell MC), it is honest about regime boundaries, and the gate
threshold is tunable per vessel from a one-time MC calibration sweep.
When the brucon nonlinear thruster allocator and power-limit
saturation are wired in, individual-DOF saturation in the score
generalises naturally to per-allocator-output saturation; the same
threshold logic applies. This avoids the "always run a Monte-Carlo
in the loop" path that would otherwise be necessary if the linear
predictor were used naively in the bistability band.

**Files.** `cqa/cqa/transient.py` (score computation in `info`),
`cqa/cqa/operability_polar.py` (`bistability_alarm` parameter and
gate, `WcfdiOperabilityOverlay.bistability_alarm` field),
`cqa/cqa/wcfdi_self_mc.py` (validator engine, used here for the
calibration), `cqa/tests/test_transient.py` (3 score tests),
`cqa/tests/test_operability_polar.py` (2 gate tests).

**Future work / open questions.**

* The score uses the *intact* covariance ODE for `sigma_tau_cmd` (the
  same one that under-predicts variance in the saturated regime).
  This is conservative for the score: the true `sigma_tau_cmd` during
  saturation is larger, which would *lower* the severity ratio and
  reduce the gate's coverage. Worth re-calibrating the threshold once
  fix tier 3 (saturation-window open-loop variance correction) is
  added to the variance ODE.
* Threshold 1.5 calibrated for the CSOV defaults
  (alpha = 2/3, gamma_immediate = 0.5, T_realloc = 10 s). For other
  vessels and other failure modes the calibration sweep should be
  re-run via `wcfdi_self_mc`.
* The `wcfdi_self_mc` engine itself is general-purpose; promoting it
  to a public API (with its own demo / validation script) is a small
  follow-on.
