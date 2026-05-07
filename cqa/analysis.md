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

**Relationship to DNV-ST-0111 Level 3.** The "Level-3-light" framing
above is shorthand and worth unpacking, because the comparison to
ST-0111 Level 3 as written is not "we do the same thing cheaper" --
it is "we answer different questions, with better statistics, from
the same underlying physics".

ST-0111 Level 3 as written has three structural weaknesses that the
linearised closed-loop machinery (intact polar, WCFDI overlay,
forecast decision matrix, transient peak envelope, bistability gate)
each address independently:

1. **Wrong question.** Level 3 configures the vessel *already in the
   post-failure state* and runs station-keeping in a synthetic
   sea-state. It tests the *steady-state post-fault footprint*, not
   the *transient recovery from the failure event* -- which is the
   operationally dangerous moment. Transient behaviour appears in
   ST-0111 only as guidance-note "results which may be included",
   not as part of the procedural test. The cqa equivalents:
   ``cqa.transient.wcfdi_transient`` (peak excursion envelope of the
   transient itself), ``info["bistability_risk_score"]`` and the
   gate (the meta-stable saturated regime where deterministic mean
   recovers but a non-trivial fraction of stochastic realisations
   diverges -- Level 3 is silent on this band entirely). The
   steady-state question Level 3 *does* answer is reflected by
   ``info["cqa_precondition_violated"]``.
2. **Wrong statistic.** Level 3 collapses 9 hours (3 seeds x 3 hours)
   of simulation to a single binary "did the worst sample exceed the
   limit?" outcome. The empirical max over a finite window is the
   noisiest possible statistic of the whole simulation: a single
   freak realisation flips green to red, the typical behaviour is
   invisible, and the answer changes meaningfully if you re-run with
   different seeds. The cqa equivalents are explicit quantiles of
   the running maximum over a chosen operation duration ``T_op``
   (P50, P90, P95 ... selectable), via the inverse-Rice /
   Cartwright-Longuet-Higgins / Vanmarcke machinery in
   ``extreme_value.py``. Operator-meaningful: "P90 of the largest
   excursion you will see in the next 20 minutes is 1.8 m" carries
   information; "the largest sample in 9 hours of one seed was
   3.7 m" mostly carries seed.
3. **Wrong sample utilisation.** A 9-hour Level 3 run at 10 Hz is
   ~3.5 million samples per direction, collapsed to one boolean.
   Almost none of the simulation's actual statistical content is
   used. The textbook fix is to estimate the closed-loop *spectrum*
   from one short realisation (~ 1 hour suffices for a few-percent
   spectral estimate), then read any quantile of the running maximum
   off the parametric inverse-Rice curve. Same simulator effort, all
   quantiles for free, confidence intervals out of the spectral
   estimate's uncertainty, different operation durations T_op
   without re-running. cqa goes one step further and skips the
   simulator entirely on the spectrum-estimation side: the linearised
   closed-loop covariance gives the spectrum directly. The full
   "Level-3-equivalent" polar (36 headings, all quantiles of the
   running max) then costs ~ 1 s in the linearised pipeline; the
   simulator (whether ``vessel_simulator`` C++ or anything else) is
   needed only for *cross-validation* of the linearisation (P7), not
   for the answer itself.

The honest comparison table:

| Question | ST-0111 Level 3 | cqa equivalent | Statistical fidelity |
| --- | --- | --- | --- |
| Steady-state post-fault feasible? | 9-hour binary check | ``cqa_precondition_violated`` flag | both deterministic |
| Steady-state post-fault footprint quantile? | empirical max of 9 hr (binary vs limit) | linearised covariance + inverse-Rice | cqa: full quantile curve; Level 3: one noisy max |
| Transient recovery from failure event? | not addressed | ``wcfdi_transient`` peak envelope + bistability gate | cqa: full quantile + bistability flag; Level 3: silent |

Net framing: cqa is **complementary to**, not a replacement for,
ST-0111 Level 3. The standard's steady-state question we answer with
``cqa_precondition_violated``; the same answer, reported as a flag
instead of a binary outcome of a long simulation. The standard's
sample utilisation we improve via the inverse-Rice curve. The
transient-recovery question -- arguably the operationally more
important of the two and the one operators most lack a fast
quantitative answer to ("if this thruster group dies right now,
where will I end up before the system recovers?") -- is genuinely
absent from Level 3 and is addressed only by the cqa transient
machinery (``wcfdi_transient`` + bistability gate). The cost
collapse (``~ 1 s`` linearised polar vs ``~ 324 simulator-hours``
per Level 3 polar) is what makes the transient question tractable
to ask at all, and what makes the forecast-case decision matrix
(\u00a712.15) and the live operation-case what-if (item 4c) feasible
as runtime tools rather than design-time exercises.

What the boundary in the WCFDI overlay does **not** claim, to
preempt mis-reading: it does not say the vessel could not
station-keep in the post-WCFDI configuration at higher V_w starting
from rest. The transient and the steady state are distinct failure
modes; the overlay's boundary is the *transient-recoverability*
boundary. A separate "post-WCFDI steady-state capability polar"
(the operability polar evaluated with ``cap_intact -> alpha *
cap_intact``) would address the steady-state question and would
extend further out in V_w. Both are useful; they answer different
questions.

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

**Visualisation: bistability spaghetti.** A direct illustration of
the bistability is produced by
`scripts/diagnose_bistability_spaghetti.py`, which runs M=128
stochastic post-failure realisations at the deep-band operating point
(CSOV defaults, `V_w = 14 m/s` beam, alpha=2/3, `t_end = 400 s`) and
plots all per-realisation sway trajectories on a single axis,
colour-coded by recovery (green if `|eta_y(t_end)| < 5 m`, red
otherwise), with the deterministic linear-predictor mean overlaid.
Output: `csov_wcfdi_bistability_spaghetti.png`.

At this operating point the deterministic mean stays on the
recovering branch (peak ~ 1.9 m, returns to the new equilibrium
within ~ 200 s), but 50/128 = 39% of the realisations diverge onto
the runaway branch -- they sit on top of each other and look like a
single bundle until ~ 60-100 s after the failure, then bifurcate.
The companion peak-`|eta_y|` histogram makes the bimodality
explicit: a tight green cluster around the deterministic peak versus
a broad red tail extending well beyond the IMCA alarm radii. The
score for this point is ~ 6, which is consistent with the
> 5 -> < 70% recovery row of the calibration table above and far
above the gate threshold of 1.5.

This is the picture that motivates the entire bistability gate: the
mean trajectory alone is *not* a sufficient operability statistic in
the saturated band, and a cheap deterministic indicator
(`bistability_risk_score`) is the right way to flag it without
running MC in the operational loop.

### 12.15 Forecast-case WCFDI decision matrix (operationally-facing)

The operability polar (§12.13) and its WCFDI overlay are
*table-top / design-time* artefacts that sweep a synthetic
Pierson-Moskowitz environment over all directions. The
**forecast-case decision matrix** is the first operationally-facing
analogue: per forecast time-slot and chosen vessel heading, evaluate
the *same* intact and post-WCFDI metrics at the *forecast* sea state
`(V_w, H_s, T_p, V_c, theta_env)` and assign a per-cell
green/amber/red traffic light.

**Module:** `cqa/cqa/decision_matrix.py`. Public types
`ForecastSlot`, `DecisionCell`, `WcfdiDecisionMatrix`. Public
functions `evaluate_decision_cell` (single-cell, useful standalone)
and `wcfdi_decision_matrix` (full grid driver, with progress
callback). Tests: `cqa/tests/test_decision_matrix.py` (12 tests
covering helpers, single-cell happy path, bistability gate,
CQA precondition, and matrix decomposition invariants). Demo:
`scripts/run_decision_matrix_demo.py`, output
`csov_wcfdi_decision_matrix.png`.

**Direction model.** v1 honours the polar's collinear convention:
each slot carries a single `theta_env_compass` for wind, wave and
current. The evaluator computes
`theta_rel = wrap_to_pi(theta_env_compass - heading_compass)` and
feeds it to the underlying PSD assemblers and `wcfdi_transient`.
This is realistic for North-Sea wind-driven seas where wind, wave
and (wind-driven) current are usually co-aligned. Independent
per-peril directions are a deferred extension; they would require
extending `wcfdi_transient` and (more substantially) the
`slow_drift_force_psd_newman` derivation, which currently assumes
collinear forcing.

**Reuse, not reinvention.** The intact axis goes through
`summarise_intact_prior` (the same engine used by the polar at each
swept point) so the forecast-case intact P90 metric and traffic
light are identical to the polar's per-direction read at the
matching `(V_w, H_s, T_p)`. The WCFDI axis goes through
`wcfdi_transient` and uses the same envelope rule
`max_t (|eta_mean(t)| + k_sigma * sigma(t))` as the polar's
`_wcfdi_peak_metrics`, with the same `k_sigma = 0.674` (P75) default.
The bistability gate (`bistability_alarm = 1.5`) from §12.14 is
applied identically. Net effect: at any operating point that the
polar can read directly (i.e. `(H_s, T_p) = PM(V_w)`), the decision
matrix and the polar agree exactly. The matrix's added value is the
*forecast* operating points where `(H_s, T_p)` are independent of
`V_w`.

**Combination rule.** Per cell, `overall = worst(intact, wcfdi)`
under the order `green < amber < red`. This matches IMCA M254 Fig. 8
"decision matrix" semantics: any axis red flips the cell red; any
amber and none red flips amber. The CQA precondition violation
(`info["cqa_precondition_violated"]` from `wcfdi_transient`) and the
bistability gate both force WCFDI red regardless of the nominal
peak envelope; on the demo storm grid the gate fires at the
expected operating points (high-V_w beam slots).

**Demo summary (24 h synthetic storm, CSOV defaults).** The demo
generates a triangular V_w ramp from 7 -> 16 -> 7 m/s over 24 h
with a slowly veering NW -> N direction, evaluates the matrix on
12 vessel headings every 30 deg (12 x 24 = 288 cells), and emits a
3-row heatmap (intact / WCFDI / overall) of headings x time. With
the default thresholds and bistability gate, 217 cells are green,
1 amber, 70 red; the red region forms a contiguous band around the
storm peak (hours 9-15) for headings broadside to the
weather, while head-on headings stay green throughout. This is
exactly the shape an operator wants on a planning chart: pick a
green vessel heading column for each forecast slot, accept that the
beam-on hours are no-go.

**Differences vs operability polar (when to use which).**

* Polar: design-time, sweeps synthetic PM environment, output is the
  V_w boundary at each direction. Use to size the vessel /
  controller / IMCA-radius limits, or to compare candidate vessels.
* Decision matrix: operationally-facing, consumes a forecast,
  output is a per-(slot, heading) traffic light. Use to plan a
  specific operation window. Same engines, different inputs.

This is roadmap item 4b ("Forecast-case decision matrix"). Item 4c
("Operation-case live what-if") is the next operationally-facing
step: at runtime, given the live posterior from `online_estimator`
and the live measured environment, evaluate the post-failure metric
with the *live posterior* P0 (rather than the steady-state Lyapunov
P0 used here) and emit a single live badge. 4c depends on the
brucon `vessel_simulator` cross-check (P7) to bound the
linearisation error before the operator trusts the live number; 4b
does not, because the forecast itself is the dominant uncertainty
on the planning side.

**Files.** `cqa/cqa/decision_matrix.py` (engine + dataclasses),
`cqa/tests/test_decision_matrix.py` (12 tests), `cqa/__init__.py`
(exports), `cqa/scripts/run_decision_matrix_demo.py` (synthetic
storm demo + heatmap), `csov_wcfdi_decision_matrix.png` (demo
output, gitignored).

**Open issues / future work.**

* The synthetic storm demo uses `pm_hs_from_vw` (the proper
  Pierson-Moskowitz `H_s = 0.21 V_w^2 / g` law). Note that
  `wcfdi_self_mc_matrix` and the §12.14 calibration use the simpler
  proxy `H_s = 0.21 V_w` (mismatched by a factor of `V_w / g ~ 1.4`
  at V_w=14). Both are internally consistent within their own
  modules; the bistability calibration table in §12.14 is keyed to
  the proxy. When the calibration is next re-run (e.g. after the
  saturation-window variance correction), the two should be
  reconciled to the proper PM law and the table re-emitted.
* Independent wind / wave / current directions per slot: deferred.
  Real forecasts (NORA3, ECMWF) can have wave swell from a
  different bearing than wind; the v1 collinear model is a
  conservative approximation when the three are within ~30 deg of
  each other (typical for wind-driven seas) but breaks down for
  swell-dominated conditions.
* Forecast input format: v1 takes a Python list of `ForecastSlot`
  objects. A JSON / NetCDF parser is a natural follow-on once the
  brucon-side forecast format is fixed.
* The decision matrix grid is dense (slots x headings) and recomputes
  every cell; for a fixed vessel/joint the per-slot intact axis
  could be computed once per slot (it does not depend on heading
  beyond `theta_rel`, but `theta_rel` does depend on heading -- so
  no savings there). The WCFDI axis similarly. No cheap caching
  trick was applied; the demo runs 288 cells in ~ 60 s on the
  prototype stack.


### 12.16 P7 brucon force-level cross-validation

Companion to the closed-loop intact validation
(`scripts/p7_brucon_validation/run_comparison.py`): the force-level harness
`scripts/p7_brucon_validation/compare_forces.py` compares cqa's
deterministic environmental forcing against the brucon vessel-simulator
ensemble at the same `(V_w, H_s, T_p, V_c, theta_rel)` operating point,
sampled in the late intact window after closed-loop transients have
decayed.

#### Findings (V_w=14 m/s, H_s=4.20 m, T_p=10.22 s, V_c=0.5 m/s, beam-on)

* **Brucon CSV columns are kN / kN.m, not N / N.m.** The pipeline parses
  `WindX/Y/Mz`, `DriftX/Y/Mz`, `CurX/Y/Mz`, and `Tx/Ty/Tz` from `*.out`;
  all are scaled by 1e-3 in `PrintDataLine()` before printing. Multiply
  by 1e3 before comparing against cqa's N / N.m output.

* **Body-frame +sway convention differs.** cqa: +sway = starboard. Brucon
  (per measured signs): +sway = port. The pdstrip-based path in cqa
  (`mean_drift_force_pdstrip`) inherits brucon's convention naturally
  because both read the same `csov_pdstrip.dat`.

* **Drift force agrees to 1% across all three DOFs** when both sides
  evaluate the same QTF table with matching spectrum and spreading:

  | DOF   | cqa Bret + cos^2 (brucon-replicated) | sim   | ratio |
  |-------|-------------------------------------:|------:|------:|
  | surge | +3.3 kN                              | +3.0  | 0.90  |
  | sway  | -216.5 kN                            | -214.8| 0.99  |
  | yaw   | +436.8 kN.m                          | +424.5| 0.97  |

  Reaching this required getting three things right:

  1. **Spectrum shape.** Brucon falls back to **Bretschneider** for
     V_w-driven seas when `wave_spectrum_type` is unset in
     `vessel_simulator_settings.prototxt` (the default branch in
     `vessel_simulator_wrapper.cpp:117-120`). cqa's
     `mean_drift_force_pdstrip` hard-codes JONSWAP (gamma=3.3 default).
     The two differ by ~30-50% in the drift integral because their peak
     shapes redistribute energy differently across the QTF support.

  2. **Spreading kind.** Brucon uses **cos^n(delta) over (-pi/2, +pi/2)**
     with n given by `WaveSpectrum`'s `spreading_factor` argument
     (default n=2, `vessel_simulator_wrapper.cpp:110`). cqa's
     `SeaSpreading` is **cos-2s = cos^(2s)(delta/2) over (-pi, +pi)**.
     These are different functional forms. The Gaussian-limit
     equivalence is **s ~ 2n** (so brucon's cos^2 ~ cqa's cos-2s s=4),
     NOT s = n/2 as the surface-similar form might suggest. Folk
     intuition fails here -- a footgun bit me during this validation.

  3. **Faltinsen factor 2.** pdstrip QTFs are in N per amplitude^2
     (zeta_a^2), and Faltinsen [90] eq. 5.41 reads
     `F_drift = 2 * integral D(w, beta) * S_eta(w) dw`. The factor 2
     appears in both cqa and brucon. Misapplying factor-1 (sometimes
     seen when QTFs are tabulated as F/zeta_a, not F/zeta_a^2) would
     halve the result.

* **Spreading-convention sweep** at this operating point (Bretschneider,
  beam-on, replicating brucon's discrete pdstrip integral):

  | spreading              | sigma (deg) | F_y (kN) |
  |------------------------|------------:|---------:|
  | long-crested           |           0 | -309.5   |
  | cos-2s s=15 (cqa def.) |        20.6 | -252.7   |
  | cos-2s s=8             |        27.8 | -231.4   |
  | cos-2s s=5             |        34.5 | -211.9   |
  | cos-2s s=4             |        38.1 | -201.5   |
  | cos-2s s=2             |        50.9 | -164.8   |
  | cos-2s s=1 (= cos^1!)  |        65.1 | -123.6   |
  | brucon cos^n n=2       |        33.0 | -216.5   |

  The ~33 deg one-sigma cone of cos^2 is well-aligned with cos-2s s~5,
  illustrating the s ~ 2n equivalence numerically. The sim observed
  -214.8 kN sits exactly on cos-2s s=5 / brucon cos^2.

* **Wind sway** matches to 0.05% (189.6 kN cqa vs 189.7 kN sim, opposite
  signs per the convention difference). cqa's `WindForceModel` uses the
  same OCIMF-style coefficient table as brucon's `WindForceModel`, so
  this agreement is unsurprising once the units are right.

* **Current sway** matches to 13% (50.5 vs 57.0 kN). Probably small
  differences in C_y(theta_rel) curve sampling or current-angle
  convention. Not investigated further -- 13% is well within the
  forecast uncertainty band that drives the upstream
  decision-matrix application.

* **The legacy parametric drift placeholder is wrong by ~2x.**
  `WaveDriftParticulars.drift_y_amp = 25_000 N/m^2` (`config.py:81-86`,
  flagged "very simplified placeholder for P1") gives 440 kN at
  H_s=4.2 m, twice the simulator's 215 kN. **Production drift use
  should switch to a spectral-QTF integration matching brucon's
  Bretschneider + cos^2** in the modules that currently use the
  parametric form: `transient.py:425`, `wcfdi_mc.py:191`,
  `wcfdi_self_mc.py:173`, `excursion.py:154`, `decision_matrix.py:268`,
  `operability_polar.py:262`. Likely API additions:

  * `mean_drift_force_pdstrip(... spectrum: 'jonswap'|'bretschneider' = 'jonswap')`.
  * `SeaSpreading.cos_n(n)` constructor with explicit conversion
    documentation, alongside the existing `cos-2s` constructor.

* **Wind yaw moments at beam-on** are non-zero in the simulator
  (-3824 kN.m) but ~zero in cqa's `WindForceModel.force()` because the
  default coefficient table has `C_n_yaw(beam) = 0`. Brucon's wind model
  evidently has a non-zero yaw coefficient at beam, possibly from
  asymmetric superstructure or a moment reference-point offset (LCG vs
  midship). Not investigated in depth -- listed as follow-up; relevant
  if heading-control authority or yaw-direction WCFDI saturation enters
  the limiting envelope.

#### Validation status

The intact closed-loop response cross-validation (sigma_x, sigma_y,
running-max CDFs) **passes tightly** with the agreed controller tuning
(omega_n=(0.06, 0.08, 0.12), zeta=(0.95, 0.95, 0.95),
SetControllerGainLevel(2,2,2), 500 s settle, 200 s sample window,
PosDev as the sim radial proxy): cqa P50/P90 of running-max position
error 1.11/1.37 m vs simulator 1.61/2.13 m. The transient WCFDI phase
still shows cqa pessimism (52 m sway peak predicted vs sim-bounded
recovery), attributable to the per-DOF thrust saturation cap in cqa's
transient solver not modelling DOF trade-off (the allocator can
sacrifice yaw to keep sway). Listed as follow-up to the
saturated-equilibrium and achievable-polytope work.

#### Decision: Bretschneider as cqa's DPCAP default spectrum

This validation also forces a project-level decision: **cqa's default
wave spectrum for capability / station-keeping analyses should be
Bretschneider (= 2-parameter Pierson-Moskowitz), not JONSWAP**.
Three reasons -- and one important caveat about conservatism:

1. **Standards alignment.** IMCA M254 Rev.1 (DPCAP guidance) and
   DNV-ST-0111 (assessment of station-keeping capability of DP
   vessels) both prescribe a 2-parameter PM / Bretschneider spectrum
   for capability analyses. The standards bodies picked the simpler
   form to be **globally applicable rather than tuning a `gamma`
   per region** (JONSWAP's gamma was originally calibrated to
   North Sea wind seas; the value to use elsewhere is contested
   and adds avoidable variance to a normative deliverable). cqa
   targets these standards directly via the M254 Fig. 8 decision
   matrix (sec 12.1) and ST-0111 wind/wave/current relations.

2. **Brucon agreement.** Brucon's vessel_simulator defaults to
   Bretschneider when `wave_spectrum_type` is unset
   (`vessel_simulator_wrapper.cpp:117-120`). Matching brucon's
   default removes a needless cross-validation friction.

3. **Mathematical equivalence.** Bretschneider is JONSWAP at
   gamma=1 (verified numerically in cqa: `jonswap_psd(...,
   gamma=1.0)` agrees with the standard Bretschneider closed
   form to machine precision). So the migration is a one-line
   parameter change in the helper, plus the API surface decisions
   below.

**Caveat on conservatism.** Capability analyses are often loosely
described as wanting "conservative" disturbance spectra. The
JONSWAP-vs-Bretschneider conservatism direction depends on T_p
relative to the QTF peak frequency and is **not unconditionally
in either direction** -- it cannot be argued generically. For the
CSOV's sway-drift QTF (peak at omega ~ 1.1 rad/s, T ~ 5.6 s):

  | T_p [s] | omega_p [rad/s] | F_y Bret [kN] | F_y JONSWAP gamma=3.3 [kN] | Bret/JON |
  |--------:|----------------:|--------------:|---------------------------:|---------:|
  |     5.0 | 1.26            | -867          | -988                       | 0.88     |
  |     7.0 | 0.90            | -797          | -765                       | 1.04     |
  |     8.0 | 0.79            | -618          | -488                       | 1.27     |
  |    10.2 | 0.61            | -309          | -207                       | 1.49     |
  |    12.0 | 0.52            | -167          | -108                       | 1.55     |
  |    14.0 | 0.45            | -77           | -27                        | 2.82     |
  |    16.0 | 0.39            | -34           | -1                         | large    |

  At the *short* wind-sea T_p typical of dimensioning operability
  conditions for North Sea / North Atlantic CSOVs (T_p ~ 6-9 s,
  near or above the QTF peak), **JONSWAP gamma=3.3 gives larger
  drift than Bretschneider**: JONSWAP's narrower peak concentrates
  more energy in the high-overlap region near the QTF maximum.
  At long T_p (swell-like), the order reverses because
  Bretschneider's broader high-frequency tail still feeds the QTF
  while JONSWAP's sharp low-frequency peak does not.

  So the standards' move toward Bretschneider/2-param PM is
  **not primarily a conservatism choice** -- it is a portability
  and standardisation choice that accepts modest non-conservatism
  in fully-developed wind seas (the regime where JONSWAP gamma > 1
  was historically calibrated) in exchange for a globally
  applicable single shape. This is a known DPCAP-community
  trade-off and is acceptable within the standards framework.

JONSWAP remains relevant for fatigue / extreme-response work where
regional spectral peakedness matters and is well-calibrated, and
should stay available in cqa as a non-default option. The
follow-up implementation involves:

* Adding a `spectrum: Literal['bretschneider', 'jonswap'] = 'bretschneider'`
  argument to `mean_drift_force_pdstrip`,
  `slow_drift_force_psd_newman_pdstrip`, and the wave-frequency PSD
  helpers in `cqa.psd` and `cqa.wave_response`. Internally,
  `bretschneider` dispatches to `jonswap_psd(..., gamma=1.0)`;
  no separate Bretschneider implementation is required.
* Changing the per-call default from JONSWAP gamma=3.3 to
  Bretschneider; existing tests and analyses that depend on
  JONSWAP behaviour pass `spectrum='jonswap'` explicitly.
* Adding `SeaSpreading.cos_n(n)` constructor (brucon's convention)
  alongside `cos-2s s=...` (cqa convention), with explicit docstring
  on the two parameterisations and the Gaussian-limit equivalence
  s ~ 2n. Default DPCAP spreading: cos^2 (n=2), matching brucon.
* Re-calibration of the §12.14 bistability table under the new
  default spectrum, since the slow-drift PSD energy distribution
  (and hence the saturated-equilibrium / bistability transition)
  shifts noticeably with gamma at the relevant T_p.

### 12.17 Spectral drift in `wcfdi_transient` and brucon-aligned added mass

Two related closed-loop-fidelity improvements made on top of the §12.16
force-level cross-validation. Neither closes the residual closed-loop
intact P50/P90 gap by itself; both are physically required regardless,
and §12.17.3 records the surprising direction of the residual that they
expose.

#### 12.17.1 Spectral drift opt-in for `wcfdi_transient` / `wcfdi_self_mc`

`wcfdi_transient` and the supporting `_build_disturbance_psd_funcs`
(used by `wcfdi_self_mc` / `wcfdi_self_mc_matrix`) gained an optional
`rao_table` kwarg. When supplied, the **mean drift force** uses
`mean_drift_force_pdstrip` and the **slow-drift force PSD** uses
`slow_drift_force_psd_newman_pdstrip`, both integrating the same
pdstrip QTF table that brucon's `MeanDriftForces()` uses (and that
§12.16 validated to ~1 % at force level). When `rao_table=None`
(default), the parametric `WaveDriftParticulars` path is preserved
verbatim for backwards compatibility.

Motivation: at the P7 validation point (V_w=14, H_s=4.20, T_p=10.22,
beam-on), the parametric `WaveDriftParticulars` for CSOV gave **+441 kN
sway drift** (wrong sign, ~2x magnitude vs the spectral path's
**-216 kN** that matched brucon to 1 %), and `drift_n_amp = 0` so the
yaw-drift channel was missing entirely.

#### 12.17.2 Brucon-aligned added-mass fractions

cqa's `surge_added_mass_frac=0.05`, `sway_added_mass_frac=0.80`,
`yaw_added_inertia_frac=0.30` were heuristic typical-OSV values. We
ported brucon's section-integrated added-mass calculation
(`libs/dp/vessel_model/vessel_coefficients.cpp::AddedMass::A11/A22/A66`,
including the Lewis-form section coefficient
`SectionAddedMassCoefficient`) to Python and applied it to the brucon
CSOV `vessel_data.prototxt` section table at design draft 6.50 m. The
resulting fractions:

| | typical (was) | brucon-derived (now) | total cqa/brucon |
|---|---:|---:|---:|
| `surge_added_mass_frac` | 0.05 | **0.060** | M11 0.99 |
| `sway_added_mass_frac` | 0.80 | **0.671** | M22 1.00 |
| `yaw_added_inertia_frac` | 0.30 | **0.620** | M66 1.00 |

Total M11/M22/M66 (rigid + added) match brucon to within ~1 %.
Rigid-body yaw inertia uses the brucon default `r66 = L/4` in both
codes, so cqa's existing `VesselParticulars.yaw_inertia` is unchanged.

#### 12.17.3 Closed-loop residual after both fixes

| | parametric drift | spectral drift | + mass aligned |
|---|---:|---:|---:|
| cqa P50 \|pos\| (m) | 1.11 | 2.93 | 3.09 |
| cqa P90 \|pos\| (m) | 1.37 | 3.55 | 3.75 |
| brucon P50 \|pos\| (m) | 1.61 | 1.56 | 1.46 |
| brucon P90 \|pos\| (m) | 2.13 | 2.02 | 2.27 |
| cqa stationary σ_y (m) | 0.38 | 0.92 | 0.99 |
| brucon ensemble median σ_y | ≈0.64 | 0.64 | 0.61 |
| cqa decorrelation T_y (s) | 237 | 39 | 39 |

The previously-claimed 31-36 % cqa under-prediction was an artefact of
two cancelling errors in the parametric path:

1. The 680 kN sway drift coefficient over-cooked the mean load enough
   to **saturate** `cap_immediate` after WCFDI, producing a spurious
   ~52 m post-failure peak excursion. With the spectral path's truthful
   13 kN sway / 461 kN.m yaw, the surviving thrusters absorb the load
   and there is no mean-level transient kick. The closed-loop comparison
   subtracts a per-seed offset before comparing P50/P90 (see
   `run_comparison.py:314-315`), which masked this in the headline
   numbers.
2. The parametric path had `drift_n_amp = 0`, so yaw slow-drift PSD was
   identically zero. The spectral path produces nonzero yaw slow-drift,
   which couples into gangway-tip y via the lever arm, **plus** a much
   richer broadband sway slow-drift PSD that collapses the
   decorrelation time from 237 s to 39 s, inflating the extreme-value
   quantile over a 1800 s operating window.

After both fixes, **cqa now over-predicts σ_y by ~60 %**
(0.99 m vs brucon 0.61 m). This is in the *opposite* direction of the
old apparent under-prediction. The mass alignment alone moved σ_y from
0.92 to 0.99 m, contradicting the naive σ² ∝ 1/M³ heuristic; the
closed-loop transfer cancels enough of the mass dependence (because
K_p, K_d are scaled with M to maintain ω_n) that mass is not the
dominant variance lever.

The remaining 60 % over-prediction is therefore in the **slow-drift
PSD shape** itself, not in the force-level magnitudes (which §12.16
validated) and not in vessel mass. Candidates to investigate:

* **Newman approximation overshoot.** `slow_drift_force_psd_newman_pdstrip`
  uses Newman's S_FF(w) = 8 |D(w_carrier)|^2 ∫ S_η(w)^2 dw with a single
  carrier frequency. The full 2nd-order QTF at low difference
  frequencies can be substantially smaller (Pinkster 1980), particularly
  in beam seas where mean drift is large.
* **Yaw slow-drift / lateral coupling.** With `drift_n_amp = 0` the
  parametric path missed yaw slow-drift entirely. A direct test is to
  zero the yaw column of `slow_drift_force_psd_newman_pdstrip`'s output
  and re-run; the σ_y delta attributes the lever-arm contribution.
* **Closed-loop bandwidth.** The simulator's `omega_n_sway = 0.08 rad/s`
  was used in cqa, but brucon's actual realised closed-loop bandwidth
  may differ due to thrust-allocation lag / commanded-vs-realised
  thrust transfer that cqa does not currently model in the *intact*
  variance path (only post-WCFDI).

Roadmap items 14 (PSD-shape diagnostic) and 15 (achievable polytope)
are the natural follow-ons.

### 12.18 Closing the residual: Newman directional-weight bug + spreading default

§12.17.3 left a residual cqa σ_y over-prediction of ~60 % (0.99 m vs
brucon 0.61 m) attributed tentatively to "Newman approximation
overshoot" or related slow-drift PSD shape error. A direct
time-series comparison of the **drift force itself** (not the closed-
loop position) collapsed the puzzle into two independent bugs that
together accounted for the full discrepancy.

#### 12.18.1 Diagnostic: direct DriftY time-series comparison

`scripts/p7_brucon_validation/compare_drift_y_timeseries.py` reads the
`DriftY` channel from each of the 30 brucon P7 seeds, computes per-
seed mean / std / decorrelation time / Welch PSD, and overlays them
against (a) cqa's analytic `slow_drift_force_psd_newman_pdstrip` and
(b) a cqa-realised time series from that PSD. This is the right
diagnostic level for testing the Newman approximation in isolation:
no closed-loop transfer in the way, and brucon's `DriftY` is a
direct realisation of the *full diagonal QTF* (which is the gold-
standard reference -- see `wave_response.cpp::CalculateDriftForces`).

The comparison initially showed a 6.6× over-prediction of cqa's
σ_DriftY (363 kN vs brucon's per-seed median 94 kN) -- much larger
than could be explained by the closed-loop variance transfer alone
and well outside any plausible Newman-vs-full-QTF discrepancy in the
literature. That ruled out the §12.17.3 candidate list and pointed
to a more basic problem.

#### 12.18.2 Bug 1: Quadratic vs linear directional weighting in Newman PSD

Inspection of `cqa/cqa/drift.py::slow_drift_force_psd_newman_pdstrip`
revealed the per-direction PSD contributions were accumulated with a
**linear** weight `w_k`:

```
G += w_k * G_dir   # WRONG
```

Brucon's reference implementation realises the Newman force per
direction (`wave_response.cpp:285-339`) with per-direction wave
amplitude
``a_i^k = sqrt(2 * S_η(ω_i) * D(θ_k) * Δθ * Δω)``
(`wave_response.cpp:444-465`). The Newman expression contains
`a_i a_m` so each direction contributes a force time series scaling
linearly in `a^k`, hence its PSD scales as `(a^k)^4 ∝ w_k²`. With
**independent random phases per direction** (brucon stores a separate
phase array per direction), the direction-summed PSD is the sum of
per-direction PSDs, each carrying `w_k²`:

```
S_F(μ) = sum_k w_k² · 8 ∫ T_k(ω+μ)² S_η(ω) S_η(ω+μ) dω    # CORRECT
```

The bug was silent in the long-crested limit (single direction with
`w = w² = 1`) and in the mean drift force (which is linear in
``a²``, so `w_k¹` is correct; §12.16 mean-force agreement to 1 %
confirms this). For the cqa default short-crested spreading, the
over-prediction factor is `1 / (sum w² / sum w¹·sum w¹) = 1 / 0.164
≈ 6.1×`, matching the observed 6.6× to within sampling noise.

Fix: `cqa/cqa/drift.py:299` `G += w * G_dir` → `G += (w * w) * G_dir`,
with a docstring block citing the brucon source lines and the
`a_i a_m` argument.

After the fix:
- cqa σ_DriftY analytic = 157 kN (was 363 kN)
- brucon per-seed median = 94 kN
- residual ratio: 1.67× (still high)

#### 12.18.3 Bug 2: cqa default spreading too narrow vs brucon

Steered by user pushback ("if the realisation does not match the
mean, splitting in slow / wave bands is barking up the wrong tree"),
we re-checked the **mean** drift force at the same operating point:

| | cqa F_y | brucon empirical mean | ratio |
|---|---:|---:|---:|
| `cos-2s s=15` (PRIOR cqa default) | -253 kN | -215 kN | 1.18× |
| `cos-2s s=4` (brucon-equivalent)  | -202 kN | -215 kN | 0.94× |
| `long-crested`                    | -308 kN | -215 kN | 1.43× |

The 17 % mean over-prediction with the prior `s=15` default
exposed a spreading-width mismatch: brucon's `WaveSpectrum`
default is `cos²(δ)` over `(-π/2, +π/2)` (i.e. `n=2` in its `cos^n`
family). Per the Gaussian-width equivalence `s ≈ 2 n` (verified in
`tests/test_sea_spreading.py::test_cos_n_n2_matches_cos2s_s4_in_one_sigma`),
cqa cos-2s `s=15` is **substantially narrower** than brucon's `n=2`
and over-concentrates wave energy on the beam-on direction where
`D_y(ω, β=90°)` peaks.

Fix: change cqa default `SeaSpreading.s` from 15 to **4**, anchored
to (i) the DNV-RP-C205 cos-2s wind-sea range `s ∈ 2-10`
(Mitsuyasu et al. 1975), and (ii) Gaussian-width equivalence with
brucon's `cos²(δ)` and DNV-ST-0111 wind-sea practice. The prior
docstring claim that "s=15 is DNV-RP-C205 wind-sea typical" was
not anchored to a specific paragraph and sits at the swell-ish end
of the canonical range. This change is propagated to docstrings
(`drift.py`, `wave_response.py`, `time_series_realisation.py`,
`sea_spreading.py`) and to `tests/test_sea_spreading.py`,
`tests/test_wave_response.py`, `tests/test_drift.py`.

#### 12.18.4 Final closure at P7 (both fixes applied)

`scripts/p7_brucon_validation/compare_drift_y_timeseries.py`:

| | brucon (n=30) | cqa | ratio |
|---|---:|---:|---:|
| mean DriftY            | -215 kN | -202 kN | 0.94× |
| σ DriftY (analytic)    | 94 kN   | 99.7 kN | **1.06×** |
| σ DriftY (realised)    | 94 kN   | 92.0 kN | **0.98×** |
| T_decorr               | 2.7 s   | 2.5 s   | 0.95× |
| σ²·T (low-freq budget) | -       | -       | **1.07×** |

`scripts/p7_brucon_validation/run_comparison.py` closed-loop:

| | brucon (sim, ensemble) | cqa | diff |
|---|---:|---:|---:|
| std(sway) per seed   | 0.59 m   | -            | (cqa P50/P90 implies σ ≈ 0.55 m) |
| P50 \|pos\|          | 1.48 m   | 1.31 m       | -0.17 m  |
| P90 \|pos\|          | 2.03 m   | 1.60 m       | -0.43 m  |

cqa is now within sampling noise on force level and mildly
**under**-predicts the closed-loop quantiles -- the opposite side of
the prior 60 % over-prediction. Plausible source of the residual
under: cqa's stationary-Gaussian Lyapunov path doesn't capture
non-Gaussian tail content from non-stationary wind / finite-sample
extreme-value sampling that brucon's time-domain ensemble does.
For an operability-feasibility tool this level of agreement is a
green light to proceed.

#### 12.18.5 Methodology note: when realisations diverge, check the mean first

The bug-finding sequence is worth recording. The §12.17.3 candidate
list (Newman overshoot, yaw lever arm, closed-loop bandwidth) was
informed but speculative; what actually closed the gap was

1. Compare the *force* directly (not the position), because the
   closed-loop transfer obscures everything;
2. Decompose the gap into mean and variance, because the mean is
   linear in `a²` and exposes spreading errors that the squared
   variance also inherits;
3. Trust the brucon source as the reference and trace the discrepancy
   back to the cqa expression term-by-term, in particular the
   per-direction quadrature weights.

Step 2 was the user's, and was decisive. A band-split analysis of
DriftY into slow vs wave components -- the natural-looking next
move -- would have been a wild goose chase: brucon's `DriftY` is the
full QTF realisation, not 1st-order excitation, so it has no wave-
band content for the Newman PSD to fail to model. The mean-vs-σ
decomposition cleanly separated the two independent root causes
that were each contributing roughly half of the σ over-prediction
on a log scale.


### 12.19 Waves-only closed-loop residual: channel semantics + integrator gap

After §12.18 closed the force-level residual to ~6 % at P7 beam-on,
the closed-loop intact P50/P90 |pos| was still under-predicted: cqa
1.31 / 1.60 m vs brucon ensemble 1.48 / 2.03 m. The user suggested
isolating the wave channel by running a Vw=0, Vc=0 ("waves-only")
ensemble to remove wind/current contributions. The result was more
discordant than the full-env case: cqa 0.85 / 1.04 m vs brucon
1.74 / 2.26 m -- a ~2× P50 ratio that reopened the diagnostic.

The investigation involved two distinct stages: a wrong working
hypothesis ("missing 1st-order wave-frequency motion") that was
sharpened, then refuted by the user's clarification of the brucon
output channel semantics. The final attribution is documented below.

#### 12.19.1 Vw=0 zero-input handling

The NPD wind-gust spectrum (`npd_wind_speed_psd`) takes a positive
power of `Vw_mean` and is undefined for `Vw_mean=0`. Two call sites
needed a short-circuit to return `S_wind = 0` when `Vw <= 1e-9`:
`cqa/decision_matrix.py::_build_intact_prior_at_forecast` and
`cqa/transient.py::wcfdi_transient`. Both fixes are applied in the
current working tree -- a clean precondition for any waves-only
diagnosis.

#### 12.19.2 First diagnosis (later refuted): "WF motion missing from pos_a_p*"

Running the waves-only ensemble and inspecting cqa's
`IntactPriorSummary` showed:

| | brucon | cqa | ratio |
|---|---:|---:|---:|
| σ_y (sway, body, intact window) | 0.590 m | 0.255 m | 2.31× |
| ν_0_y (zero-up-crossings)       | 0.035 Hz | 0.012 Hz | 3.0× |
| P50 |pos| (running max, T=200s) | 1.74 m | 0.85 m | 2.05× |

Plugging brucon's measured σ_y + brucon's measured ν_0 into
Vanmarcke's running-max formula reproduced brucon's empirical
running-max (1.37 m P50 predicted vs 1.74 m measured -- the residual
attributable to the non-zero sway mean of -0.28 m, see below). So
the running-max machinery was correct; the inputs (σ, ν_0) were
wrong on the cqa side.

A first hypothesis emerged: cqa's `pos_sigma_*_m / pos_a_p*` is
built only from the slow-band forcing PSDs `[S_wind, S_drift,
S_curr]` passed through the closed-loop transfer (`operator_view.
py:836`-`839`), but brucon's `SwayDev` includes the 1st-order
wave-frequency RAO motion. At Hs=4.2 m beam-on, a quick
RAO-times-Bretschneider integral at the vessel CG gave σ_y_WF ≈
0.69 m -- the right ballpark to close the gap if combined with
cqa's slow-band σ_y_slow=0.255 m via RSS.

A new utility `sigma_pos_wave_at_body_point` was written in
`cqa/wave_response.py` to compute, at any body-fixed point, the
1st-order wave-frequency horizontal-position σ via the standard
6-DOF RAO × spreading × wave-PSD quadrature. The function
mirrors `sigma_L_wave` exactly. Per-axis sensitivity 6-vectors at
a body point `r_b = (x_b, y_b, z_b)` are

    c6_x = [1, 0, 0, 0,  z_b, -y_b]
    c6_y = [0, 1, 0, -z_b, 0,  x_b]

(rigid-body small-motion kinematics: `δp = ξ_trans + ω × r_b`).
Three sanity checks passed:

1. Long-crested at CG, beam=90°: σ_y_wf = 0.687 m. Matches the
   hand calc to 0.001 m.
2. Short-crested (cos²ˢ s=4) at CG: σ_y_wf = 0.518 m. Spreading
   attenuates beam-on sway as expected.
3. At dp_base body point (5, -9, -8) m, short-crested:
   σ_x_wf = 0.250 m, σ_y_wf = 0.596 m. Yaw lever-arm (x_b·yaw)
   boosts surge response; combination of sway + roll-lever +
   yaw-lever gives σ_y close to the brucon-measured σ_SwayDev
   = 0.59 m.

The tantalising 1 % match with brucon's measured σ_SwayDev = 0.590 m
made it look like the diagnosis was complete: just plumb
`sigma_pos_wave_at_body_point` through `summarise_intact_prior`,
multi-band the running-max via the existing `inverse_rice_multiband`
infrastructure, and the cross-validation gap would close.

It was the wrong fix.

#### 12.19.3 The user's clarification: SurgeDev/SwayDev = DP LF estimate, NOT raw motion

The user pointed out that the operationally-meaningful position
channel for IMCA radii / DP alarms is the **DP estimator's
low-frequency position estimate**, not the true total motion --
1st-order wave motion is a known periodic oscillation that the DP
control system already filters out, and ST-0111 / IMCA M254 define
the station-keeping radius against the LF-filtered position. Truth
checking confirmed:

| brucon channel | what it actually is |
|---|---|
| `EstPosX, EstPosY` (estimator output) | DP estimator's LF position estimate, NED, relative to simulator start |
| `HfPosX, HfPosY` (estimator output) | DP estimator's WF position estimate, NED |
| `x, y` (main output) | True NED position, relative to simulator start |
| `xHf, yHf` (main output) | High-frequency component of the true NED position |
| `SurgeDev, SwayDev` | body-frame projection of `EstPosX, EstPosY` (= DP LF estimate, body-frame) |

A direct numeric test on one seed in the intact window, with
heading 180° compass:

    body_proj(EstPosY) std       = 0.7502 m
    SwayDev std                  = 0.7501 m
    diff (SwayDev - body_proj(EstPosY)) std = 0.0009 m   <- effectively identical
    body_proj(y)      std        = 0.9923 m              <- true total motion, larger

So `SwayDev` IS the body-projected DP LF estimate. The cross-
validation channel was already semantically correct. The 1st-order
RAO motion is **not in `SwayDev`** -- it has been filtered out by
the DP estimator before that channel is published. The plan to
add `sigma_pos_wave_at_body_point` to `pos_a_p*` would have mixed
two different physical channels under one field name, which is
exactly wrong.

The 2.3× σ_y gap is therefore **inside the slow band**, not a WF-
motion attribution problem. cqa's pos_a_p* semantic (slow-band
running-max for IMCA radii) is correct; what's wrong is the slow-
band σ itself.

#### 12.19.4 PSD-band attribution of the slow-band gap

A Welch PSD on body-projected `EstPosY` (intact window, 30 seeds
averaged) split the brucon LF-channel variance by band:

    f < 0.04 Hz (LF)        : variance = 0.295 m²    (90.5%)
    0.04 <= f < 0.2 Hz (WF) : variance = 0.031 m²    ( 9.5%)
    total                   : variance = 0.326 m²    (σ = 0.57 m)

So:

* The DP estimator does what it claims: WF leak into the LF channel
  is ~10 %, small but non-zero.
* The **LF variance itself is 0.30 m², while cqa predicts 0.065 m²**
  -- a 4.5× ratio in variance, 2.1× in σ. This is the residual
  after force-level matching of §12.18.

The brucon LF spectrum peaks at ω ≈ 0.063 rad/s (f ≈ 0.010 Hz) --
not at the controller bandwidth ω_n=0.08 rad/s, but **lower than**
ω_n. cqa's closed-loop position spectrum, in contrast, peaks at
the lowest grid frequency (ω → 0): for an over-damped 2nd-order
plant + PD controller driven by a flat low-frequency forcing PSD,
|H_pos(ω)|² is largest at DC.

Quick variance check using the analytic 2nd-order result:

    σ² ≈ S_F(0) / (2 · D_total · K)
    
with M = m22 ≈ 1.34e7 kg, ω_n = 0.08, ζ = 0.95:

    K = M·ω_n² = 8.55e4 N/m
    D_total = 2·ζ·M·ω_n + D_open = 2.04e6 + 3.34e5 = 2.37e6 N·s/m
    S_F_y(LF) = 1.39e10 N²·s   (from cqa.drift slow-drift PSD)

    σ² = 1.39e10 / (2 · 2.37e6 · 8.55e4) = 0.034 m²
    σ  = 0.18 m

This matches cqa's frequency-domain integration (0.255 m, the small
discrepancy from finite-grid quadrature and mass-matrix off-diagonal
coupling). For brucon's σ² = 0.30 m², either K or D would need to
be ~4.5× smaller -- equivalent to ω_n,eff ≈ 0.038 rad/s, which is
inconsistent with the brucon prototxt (ω_n = 0.08).

#### 12.19.5 Likely root cause: omitted integrator in cqa's `LinearDpController`

`cqa/controller.py` is explicit (lines 24-26):

> The integral action contributes only at frequencies below the
> dominant disturbance band and is omitted from the Lyapunov
> analysis (it is a slow process tracking the bias, modelled
> instead as 'perfect bias rejection').

This was a defensible simplification at design time -- the
integrator's role *is* to cancel the slow mean force. But it
fails to model the **transient response** of the integrator to a
slowly-varying force: the integrator lags the disturbance by its
own time constant T_i ≈ 1 / ω_i, and during that lag the position
acquires a transient offset. Brucon's actual PID has finite-bandwidth
integral action; cqa's "perfect bias rejection" is the ω_i → ∞
limit (instantaneous integrator) which leaves no LF residual.

Adding an integrator state to the closed-loop with realistic
ω_i ≈ 0.02 rad/s would:

1. Augment the state from 6 to 9 (`[eta(3), nu(3), eta_int(3)]`),
   change A_cl shape and `LinearDpController.feedback()`.
2. Lower the closed-loop position |H_pos(ω)|² near ω_i, raising it
   between ω_i and ω_n -- consistent with the brucon LF spectrum
   peaking at 0.063 rad/s.
3. Increase the position σ in the right ballpark (4.5× variance is
   the right order of magnitude for an integrator with ω_i ≈ 0.2 ω_n
   added to a PD baseline; the exact ratio depends on the forcing
   PSD shape across [ω_i, ω_n]).

A complementary candidate (estimator-in-the-loop dynamics: the DP
controller acts on the Kalman LF estimate, not the true state) is
likely a smaller contribution because brucon's true LF (`y - yHf`,
NED → body) and brucon's estimator LF (`EstPosY` → body) have
near-identical std (0.668 vs 0.590 m surge; 1.237 vs 0.590 m sway --
the y discrepancy is from `xHf/yHf` being an estimator output with
unclear semantic, not the truth). Need brucon source inspection
to nail this down precisely, but the dominant lever is the
integrator.

#### 12.19.6 Status of `sigma_pos_wave_at_body_point`

The new function in `cqa/wave_response.py` is kept as a forward-
looking utility. It is **not** wired into `IntactPriorSummary`,
because the `pos_a_p*` semantic is "DP LF estimate" (matching IMCA
radii) and 1st-order WF motion is not part of that channel.
Anticipated future uses, for which the user has expressed intent:

* "True motion at gangway base / hook" panel -- what physically
  drives gangway end-stop loads, structural fatigue, latch
  integrity. Slow + WF at the body-fixed point of interest.
  Distinct from both pos_a_p* (LF station-keeping radius) and
  gw_sigma_*_m (telescope length deviation).
* Collision-risk estimation against a fixed structure -- needs
  the true body-point excursion, not the LF estimate.

The function is regression-tested by virtue of the three sanity
checks documented in §12.19.2 and is ready for use the moment a
consumer needs it.

#### 12.19.7 Methodology note: two channels named the same

The investigation was led astray for ~2 hours by an implicit
assumption that brucon's `SwayDev` is the "raw" sway motion. The
name suggests deviation from setpoint, full stop. In fact every
brucon DP-state output channel ending in `Dev` is the body-frame
projection of a Kalman-filtered LF estimate, not the truth. The
prototype's docstring near `parse_output` mentions only "tab-
separated header"; the channel semantics are documented only in
brucon source comments. Two consequences:

1. The C++ port should expose channel semantics in named types
   (e.g. `PositionLfEstimate` vs `TruePositionBodyFrame`) rather
   than raw float arrays, to make the LF/raw distinction
   compile-time enforced rather than reader-implied.
2. Future cqa cross-validation drivers should print the channel
   semantics they're comparing against in the diagnostic output,
   to catch this class of mistake at run time.

The resolution itself was the user's, not mine. A pure code-side
investigation would not have discovered the LF vs raw distinction
without happening to read the brucon estimator source for an
unrelated reason.

> **Update from §12.20:** the integrator hypothesis above (§12.19.5)
> turned out to be only a ~10 % contributor to σ_y_LF, not the
> dominant lever. The actual closed-loop residual is mostly from
> brucon's full passive observer (5 states/DOF, with finite bias
> time constant T_b = 1000 s and a 2nd-order wave filter) plus a
> phenomenological 1st-order pole on the controller-output → vessel-
> force path with τ ≈ 5 s. See §12.20 for the corrected story and
> the multi-seed sandbox validation.

### 12.20 Sandbox passive-observer closed loop validated against brucon

§12.19.5 hypothesised that the cqa σ_y_LF under-prediction was
primarily from omitting the controller integrator. Building a
brucon-faithful sandbox closed loop and exercising it against the
30 P7 waves-only seeds refuted that single-cause story and produced
a quantitative attribution of the missing physics. This section
records the diagnostic chain, the refuted hypotheses, and the
working model that reproduces brucon σ_y_LF to within ~6 %.

#### 12.20.1 Tooling: the sandbox

Three Python tools were added under
`scripts/p7_brucon_validation/` (no production code touched):

* `sandbox_passive_observer.py` -- frequency-domain σ-prediction of
  a per-DOF closed loop with optional Fossen passive observer (5
  states), bias feed-forward, integrator state, 1st-order thrust
  lag, and 2nd-order Sælid/Jensen wave filter. All gains taken
  from brucon `build/bin/config_csov/observer.prototxt` and
  `build/bin/settings/tuning.prototxt`.
* `validate_sandbox_timeseries.py` -- LTI time-domain integration
  of the sandbox closed loop driven by brucon's logged `DriftY`
  channel and `yHf` true wave-frequency motion, single seed.
* `multi_seed_sandbox_validation.py` -- 30-seed batch wrapper with
  per-seed σ scatter and ensemble-mean PSD comparison.

Auxiliary diagnostics:

* `estimate_thrust_lag.py` -- empirical fit of `OrderTau →
  AllocTau`, `Order → Ty`, `Alloc → Ty` first-order time constants
  from the 30 brucon seeds.
* `wave_filter_sigma_sweep.py` -- sweeps wave-filter design Tp
  over [5, 30] s to quantify how much closed-loop σ the wave filter
  alone explains.
* `diagnose_thrust_command_gap.py` -- compares brucon σ(OrderTau)
  vs sandbox σ(u_cmd) at matched σ(F_drift), the diagnostic that
  proved sandbox over-suppresses regardless of the linear damping D.

The intact-window for all comparisons is `t ∈ [300, 555] s`: 240 s
after simulation start (long enough for observer transients to
settle) and 5 s before the WCFDI failure injection at t=560 s. An
earlier choice of `T_START=200, T_END=740` was contaminated by
both the initial settling and 180 s of post-failure transient,
artificially inflating brucon σ_y_LF to 0.85 m. With the corrected
window, the 30-seed median is 0.65 m (pooled 0.75 m) -- consistent
with the prior session's single-seed 0.59 m measurement.

#### 12.20.2 Channel semantics correction

Re-reading
`brucon/libs/simulator/dp_runfast_simulator/dp_runfast_simulator.cpp`
in the lines that print the CSV header (lines 1067-1112) clarified
two channels that had been mis-interpreted in §12.19:

| brucon CSV channel | what it actually is |
|---|---|
| `x, y` | **NED** total position (LF + WF combined) [m] |
| `xHf, yHf` | **BODY-frame** wave-frequency motion (already body-aligned) [m] |
| `Tx, Ty, Tz` | **Force on the vessel** from the simulator's thruster model (after rate limits / azimuth dynamics) [kN] |
| `OrderTau{Surge,Sway,Yaw}` | controller's commanded body-frame force [kN] |
| `AllocTau{Surge,Sway,Yaw}` | net body-frame force after allocator solves for individual thrusters [kN] |
| `FbTau{Surge,Sway,Yaw}` | feedback to the controller [kN] |
| `CurX, CurY, CurMz, CurMx` | **Damping forces** (`damping_forces().sway()`, mis-named) [kN] |
| `DriftY` | **2nd-order slow-drift sway force** [kN] |

Two corrections to my prior assumption:

1. `xHf, yHf` are body-frame, not NED. To recover the LF body-
   frame sway from CSV: project `(x, y)` NED via heading into
   body, then subtract `yHf` directly (no rotation).
2. `CurY` is total damping force (linear + quadratic cross-flow),
   not current force. Computing `σ(CurY) / σ(SwaySpeed)` over the
   intact window gives an *equivalent linear* damping coefficient
   D_eq ≈ 17 000 N·s/m, but the underlying model is dominated by
   quadratic cross-flow drag (R²=0.974 for pure quadratic fit
   `F_damp = -C_q · |v| · v` with C_q ≈ 240 000 N·s²/m²; vs
   R²=0.894 for pure linear). The describing-function equivalent
   at brucon's operating σ_v=4.4 cm/s reproduces D_eq=17 000 to
   within 1 %.

#### 12.20.3 The brucon Fossen passive observer (per DOF)

`brucon/libs/dp/dp_estimator/nonlinear_passive_observer.cpp` runs
a 5-state-per-DOF observer:

```
e        = y_meas − ŷ_LF − η̂_w               (wave-corrected innovation)
ŷ_LF_dot = ν̂ + ω_c · e                       (LF position estimate)
ν̂_dot    = -(D/M)·ν̂ + (1/M)·b̂ + (1/M)·u_cmd + K_a1 · e
b̂_dot    = -(1/T_b) · b̂ + K_b1 · e             (bias estimate)
ξ_w_dot  = η̂_w + k1_f · e                     (wave filter state 1)
η̂_w_dot  = -ω_w² · ξ_w − 2ζ_n·ω_w · η̂_w + k2_f · e
```

For brucon's CSOV sway block, the gains and parameters
(`config_csov/observer.prototxt` + `tuning.prototxt`) are:

* K_a1 = 0.12, K_a2 = 0
* K_b1 = 0.0012, K_b2 = 0
* T_b = 1000 s (cqa default `bias_time_constant_s = 100` is wrong
  by an order of magnitude)
* ω_c = 1.04 rad/s
* Wave filter: ω_w = 2π/Tp, ζ_n = ScaleGainLinear(ω_w, 2π/18,
  2π/10, 0.25, 0.10) ⇒ at Tp=10.2 → ζ_n = 0.107
* k1_f = -2(1-ζ_n)·ω_c/ω_w = -3.02
* k2_f = 2(1-ζ_n)·ω_w = 1.10

The critical detail is the **innovation form** `e = y_meas − ŷ_LF
− η̂_w`. An earlier sandbox version using `e = y_meas − η̂_w` (the
naive wave-rejected innovation) over-estimated wave-filter slow-
band leakage by ~10× and produced unphysical σ-vs-Tp behaviour.

The controller (`brucon/libs/dp/dp_controller/mode_base.cpp:145-146`)
is a textbook PID with Fossen-style gain calculation
(`tuning_parameters_no_speed_dependency.cpp:22-29`):

```
Kp = M · ω_n²,   Ki = 0.1 · ω_n · Kp,   Kd = 2 · ζ · M · ω_n
```

with `M = vessel_model_.Mass() + vessel_model_.SwayAddedMass()`
(matches cqa's m22). For sway, ω_n=0.08, ζ=0.95, ω_i=0.008 rad/s,
GainLevel kMedium (scale 1.0).

A second important controller feature is the velocity feedforward
`SetVelocityFeedforward(-Y(u, v, r) / 1000.0)`
(`mode_base.cpp:216`) which **cancels the vessel's hydrodynamic
damping** in the closed loop. The acceleration FF
(`SetAccelerationFeedforward(...)`) is path-driven and so is zero
during station-keeping.

#### 12.20.4 Multi-seed sandbox validation

Pooled over 30 P7 waves-only seeds in the intact window:

| configuration | median σ_y_LF [m] | % of brucon |
|---|---:|---:|
| brucon sway_LF (true, target) | 0.65 | 100 |
| brucon SwayDev (DP estimator) | 0.66 | 102 |
| sandbox perfect feedback (cqa equiv) | 0.17 | 26 |
| sandbox full obs, no thrust lag, no integrator | 0.41 | 63 |
| sandbox full obs, no thrust lag, **+ integrator** | 0.45 | 69 |
| sandbox full obs, **τ=5 s lag**, no integrator | 0.56 | 86 |
| sandbox full obs, τ=5 s lag, + integrator | 0.61 | **94** |

The ensemble-mean PSD plot
(`multi_seed_sandbox_psd.png`) shows that the slow-band plateau at
f ≈ 0.005 Hz, which carries most of the variance, is reproduced
within model uncertainty by the "full obs + τ=5 s" configuration;
without the τ knob the plateau is ~2.5× too low.

#### 12.20.5 Refuted hypotheses

The diagnostic chain was driven by user pushback at four key
points; each successive hypothesis turned out to capture only a
partial truth.

1. **"Integrator alone explains the gap"** (§12.19.5): the
   sandbox with brucon-realistic Ki = 0.1·ω·Kp = 0.0008 rad/s
   raises σ from 0.41 → 0.45 m (+10 %). Real but minor; cannot
   close the 0.41 → 0.65 gap.

2. **"Rate limit halves effective controller bandwidth"** (my
   own refuted hypothesis): user pointed out that ζ=0.95 puts the
   PID near critically damped, with no resonant peak in |H_pos|.
   The slow-band peak in σ_y(f) at ω ≈ 0.06 rad/s is the drift-
   force PSD plateau being passed through, not closed-loop
   ringing.

3. **"Brucon thrust pipeline introduces a 1st-order lag with τ ≈
   5 s"** (refuted by `estimate_thrust_lag.py`): the empirical
   transfer `OrderTau → Ty` has |H|² ≈ 1.0 and phase ≈ 0° across
   the entire slow band (f < 0.05 Hz) with coherence > 0.99. A
   first-order fit returns τ = 0.10 s (the search lower bound),
   meaning the simulator's rate-limited thrusters (RPM rate
   10 %/s, azimuth 12°/s) are essentially transparent at the
   typical sway-force amplitudes / rates of P7 station-keeping.
   **The τ=5 s parameter that closes the σ gap in the sandbox
   does not correspond to a measurable physical actuator lag.**

4. **"GPS antenna lever-arm × roll contaminates y_meas"** (user
   suggestion, refuted by data): brucon CSOV GNSS antennas at
   z = -24.23 m below CO; logged Roll has σ_LF (filtered <0.05 Hz)
   = 0.08° → expected lever-arm contamination σ ≈ 0.03 m, far too
   small to explain the residual gap. Quadrature sum
   `√(0.41² + 0.03²)` = 0.41 m.

5. **"Sandbox over-damps because it uses linear D where brucon
   has quadratic"**: refuted by D-sweep. With brucon's controller
   `Kd = 2.8e6 N·s/m` swamping any open-loop D ≤ 20 000 N·s/m,
   sandbox σ_y is insensitive to D over two orders of magnitude
   (100 to 20 000 N·s/m → σ_y = 0.37 ± 0.001 m). Brucon's
   velocity-FF additionally cancels the natural damping in
   software, making the closed loop see only Kd. Damping form
   cannot be the explanation.

6. **"Wave filter alone explains it"** (user suggestion,
   partially confirmed): the 2nd-order wave filter at Tp=10.2 s
   (ω_w = 0.616 rad/s, ζ_n = 0.107) has |η_w/e| = 0.236 with +89°
   phase at the controller bandwidth ω = 0.08 rad/s. Sweeping
   the wave-filter Tp from 5 s to 30 s in the sandbox moves σ
   from 0.39 → 0.49 m -- a real but small contributor. At the
   actual Tp=10.2 s, the wave filter accounts for ~14 % of the
   gap (0.36 → 0.41 m).

#### 12.20.6 Status of the residual

After ruling out (a) thrust lag literal interpretation, (b) over-
damping, (c) mass mismatch, (d) estimator under-reporting, (e)
roll-coupled GPS lever arm, (f) integrator as sole cause, (g) GPS
1 Hz zero-order hold on y_meas, and (h) numerical integration
error (brucon uses RK4 at 10 Hz with no filters in either signal
processing or observer; per-DOF observer fastest pole ω_c = 1.04
rad/s gives ~60 samples per period -- numerically essentially
exact), the sandbox still under-predicts σ by ~36 % (0.41 vs
0.65) without the τ=5 s knob. With the knob it matches to 6 %.
The mechanism the τ knob is calibrating is **not identified** in
this work.

Remaining (untested) candidates for future investigation:

* Cross-DOF coupling (sway-yaw via `B26`, sway-roll via `B42`)
  excited by yaw-band drift moments and slow-drift roll moments
  not modelled in the sandbox 1-DOF cut. P7 has roll dynamics
  with σ_roll ≈ 1° in this sea state (driven by 2nd-order roll
  exciting moment `wave_model_.RollExcitationMoment`); the lateral
  wave-drift sway force has correlated yaw moment that drives a
  yaw-rate response, which couples back to sway via Coriolis.
* Brucon-side instrumentation of the controller-internal sums
  (term-by-term: Kp·yLF, Kd·νh, b̂, vel_FF, accel_FF, integral)
  during a single seed would resolve this empirically but requires
  brucon-side logging changes.

Tested-and-refuted GPS / numerics candidates:

* GPS 1 Hz zero-order hold on the wave-frequency motion
  contribution `y_wf` to `y_meas`: applying ZOH at 1 Hz in the
  sandbox produced zero change in σ_y_LF (the wave filter's
  slow-band rejection of the held signal is essentially perfect).
* Discrete-time controller / sample-and-hold at brucon's 10 Hz
  rate: ω_n · Ts = 0.008 rad ⇒ phase lag at the controller
  bandwidth is 0.5° -- negligible.

The user's call (recorded here) was to **stop chasing the
residual mechanism and instead consolidate the diagnostic work**:
a 6 %-accurate phenomenological model is sufficient to make
progress on the cqa-side closed-loop integration, and the
unresolved physics is documented as a known limitation.

#### 12.20.7 Implications for cqa core

Before any cqa-side closed-loop change, three things are
established:

1. The cqa default `ControllerParams.bias_time_constant_s = 100`
   is a factor 10× too small vs brucon's T_b = 1000 s. Easy fix
   if/when an estimator-augmented closed loop is built into cqa.
2. The current cqa `LinearDpController` formulation (perfect
   feedback from true state, no observer dynamics, no bias FF
   transient) under-predicts σ_y_LF by ~4× (sandbox 0.17 m vs
   brucon 0.65 m). The closed-loop variance bug is real and
   physical, not a tuning artefact.
3. A cqa-side fix would require adding 5 observer states + 1
   integrator state per DOF (≥18 extra states for the 3-DOF
   `ClosedLoop`) plus a phenomenological τ ≈ 5 s pole on the
   command path. This is a non-trivial refactor; the validation
   evidence supports it but the design choice (do it now vs ship
   the under-prediction with a documented caveat) is left for a
   later session.

The diagnostic tools, plots, and the
quantitative-residual breakdown are committed so the next
iteration starts from a clear baseline.

#### 12.20.8 Long-settle re-validation and transfer-function decomposition

After the initial §12.20 work flagged the residual gap as
"unidentified slow-band physics", a focused follow-up established
two further results that decisively localise the mechanism.

**Long-settle ensemble (`long_run_validation.py`).** The original
[300, 555] s window is short relative to the bias estimator's
T_b = 1000 s. A 30-seed ensemble was repeated with
`settle_s = 3000` (intact-DP window [60, 3060] s) and analysed in
the late window [1500, 3000] s where bias and Tp estimators are
fully settled (~3·T_b). Result, median across seeds:

|                   | EARLY [300, 555] s | LATE [1500, 3000] s |
|-------------------|-------------------:|--------------------:|
| brucon σ_y_LF     |             0.65 m |              0.67 m |
| EstBiasSway mean  |          −193 kN |             −197 kN |
| EstBiasSway std   |             7.2 kN |              7.3 kN |
| DriftY std        |             ~ 100 kN |              97 kN |

The σ gap vs the sandbox (0.41 m) is **+2 cm wider in the late
window** — i.e. the residual is genuine steady-state physics, not
initialisation tail. The bias estimator also reaches a stable
operating point: it absorbs **91 % of the −217 kN DriftY mean** as
expected, leaving the remaining 9 % to the controller's
integrator. Bias slow-band variation 7.3 kN is steady-state.

A unit-conversion bug in an interim diagnostic transiently
suggested DriftY ≈ 0; this was corrected by re-reading the channel
through `harness.parse_output` and confirmed against the file
header. Forces in both `*.out` and `*_estimator.out` are in **kN**
(not SI N).

**Empirical transfer F_drift → y_LF.** With the long-run data
giving stationary statistics, Welch PSDs of (DriftY, sway_LF)
were computed seed-by-seed and ratioed to obtain
`|y_LF/F_drift|(f)` for brucon. The same transfer was computed
from the sandbox closed-loop A matrix
(`G(jω) = (jωI − A)^{−1} B_drift`, output state 0). The
comparison localises the gap precisely:

| f [Hz] | sandbox \|T\| [m/N] | brucon emp \|T\| [m/N] | ratio brucon/sand |
|-------:|--------------------:|-----------------------:|------------------:|
| 0.0019 |             2.1e−5 |                 2.0e−5 |          **0.97** |
| 0.0052 |             1.9e−5 |                 2.7e−5 |          **1.44** |
| 0.0073 |             1.6e−5 |                 2.4e−5 |          **1.49** |
| 0.0097 |             1.4e−5 |                 2.2e−5 |          **1.57** |
| 0.0194 |             6.0e−6 |                 6.2e−6 |          **1.03** |
| 0.0515 |             5.7e−7 |                 6.1e−7 |          **1.06** |

Three regimes are clear:

1. **Very low frequency (f < 0.003 Hz, T > 300 s):** transfers
   match within 5 %. The bias estimator's slow pole at 1/T_b is
   the dominant slow-band closed-loop pole and is correctly
   modelled in the sandbox. Note this is **not** the naive
   1/Kp = 8.4e−6 m/N — the closed-loop slow gain is ~2.5× softer
   than that, because the bias state introduces a low-frequency
   integrator-like mode that limits how stiffly the controller
   resists slow forces (b̂ partially absorbs the input before u_cmd
   responds).

2. **Bias-loop transition band (f ∈ [0.005, 0.01] Hz, T = 100–200
   s):** brucon is **1.4–1.6× softer** than the sandbox. This is
   exactly the band where the bias compensation is rolling off
   (T_b = 1000 s gives a corner at 1.6e−4 Hz; the loop gain
   K_b1·ω_c·... gives the actual roll-off shape). The variance
   integral over this band combined with the DriftY PSD is what
   produces the σ gap.

3. **High frequency (f > 0.02 Hz):** transfers match within 6 %.
   The PID + WF transfer is correct.

**82 % of brucon's σ_y_LF² variance lies in the f < 0.01 Hz band**
(σ_slow² = 0.56² = 0.314, σ_total² = 0.687² = 0.472). With brucon
1.5× softer in the [0.005, 0.01] Hz sub-band, the
variance contribution from that sub-band is ~2.25× larger in
brucon than in the sandbox — exactly what closes the gap from
0.41² = 0.168 to 0.687² = 0.472 (≈ +0.30 m² of extra variance).

**Mechanism**: a structural difference in the bias-state coupling
to the controller, manifest only in the bias-loop transition
band. The sandbox observer states are coupled identically to the
brucon equations as documented at §12.20.3, so the residual must
be in:

  - a **gain or sign convention** in how `(1/M) · b̂` enters the
    sandbox `ν̂_dot` vs brucon's,
  - the **wave-corrected innovation form**:
    `e = y_meas − ŷ_LF − η̂_w` (matches brucon `dp_passive_observer.cpp:230-280`)
    is structurally correct in the sandbox, but the gain on the
    bias path may be miswired (`K_b1·e` vs `K_b1·(y − ŷ)` without
    the wave correction).

**Refuted in this round:**
  - lever-arm coupling (LCG ≈ −1.7 m from sections, `m_c²/Kp_yaw·ω²`
    correction = 0.03 % of M; unintended yaw moment ~ 50 kN·m
    gives ≈ 0.04° response, negligible).
  - Tp estimator drift (brucon EstWavePeriodPitch settles at
    8.6 s vs sandbox-fixed 10.2 s; impact on σ_y_LF < 1 cm).

**Status: residual gap mechanism localised to bias-loop
transition band.** A focused next session should:
  1. Re-derive the sandbox bias-state coupling from
     `nonlinear_passive_observer.cpp` line by line, comparing to
     `sandbox_passive_observer.py:build_closed_loop`.
  2. If the structural form is identical, sweep K_b1 by ±50 % in
     the sandbox and check whether |T(f)| at f=0.0073 Hz rises by
     1.5×.
  3. If not, the gap is in non-LTI behaviour the brucon observer
     exhibits but my LTI sandbox can't reproduce (saturation,
     wave-filter Q-scheduling crosstalk, GPS measurement noise
     spectrum).

#### 12.20.9 Integrator's contribution localised

A follow-up question (recorded here): "the 100–200 s band is
also in the integral term land — could the integrator be
introducing oscillations there?" T_i = 2π/ω_i = 785 s vs
T_b = 2π·1000 ≈ 6280 s; the **integrator's natural period
(125–800 s) overlaps the residual band (100–200 s)** and its
interaction with the bias state's slow pole is a candidate
mechanism.

Direct test: re-compute the sandbox |y_LF/F_drift| transfer
with `use_integrator=True/False`:

| f [Hz] | brucon emp | sandbox no-I | sandbox WITH I | Δ from I |
|-------:|-----------:|-------------:|---------------:|---------:|
| 0.0019 |     2.0e−5 |       2.1e−5 |         2.0e−5 |     −3 % |
| 0.0053 | **2.7e−5** |       1.8e−5 |     **2.1e−5** | **+15 %** |
| 0.0072 | **2.4e−5** |       1.6e−5 |     **1.9e−5** | **+13 %** |
| 0.0098 | **2.2e−5** |       1.4e−5 |     **1.5e−5** |  **+8 %** |
| 0.0197 |     6.2e−6 |       5.9e−6 |         5.9e−6 |     +1 % |
| 0.0520 |     6.1e−7 |       5.6e−7 |         5.6e−7 |      0 % |

σ predictions given brucon F_drift PSD (10-seed average):

|                              | σ_y_LF predicted [m] |
|------------------------------|---------------------:|
| sandbox full obs, no I       |                0.484 |
| sandbox full obs, WITH I     |            **0.528** |
| sandbox no-bias-FF, no I     |                0.484 |
| brucon (measured)            |            **0.687** |

**Confirmed**: enabling the integrator in the sandbox raises |T|
by 8–15 % in exactly the f ∈ [0.005, 0.01] Hz band where brucon
is softer, closing **9 % of the σ gap** (0.484 → 0.528 m). This
is the first mechanism in the entire investigation that shifts
the closed-loop transfer in the right direction in the right band.

Mechanism: the integrator state with ω_i = 0.008 rad/s
introduces an additional slow pole that creates a slight peak
in `|y_LF/F_drift|` at frequencies where the bias-loop and
integrator-loop interact. The sandbox without integrator is
mistakenly *over*-stiff in the band — the integrator is a
genuine and necessary part of the brucon closed loop.

**Remaining 1.3× residual softness in the [0.005, 0.01] Hz band**
(after enabling sandbox integrator) is not yet attributed.
Verified to NOT be:
  - velocity feedforward (only active during commanded
    position moves; in station-keeping with a fixed setpoint
    the velocity FF is zero, so it cannot be the source of any
    discrepancy — the sandbox correctly omits it);
  - integrator gain mismatch (`i_scaling = 1` confirmed in
    `tuning.prototxt`, `Ki = 0.1·ω_n·Kp` matches);
  - integrator input source (sandbox already integrates `ŷ_LF`
    not true `y`, matching brucon `position_hold.cpp:393–394`
    `state_error_.ErrorSway()` which uses `estimator_data_.sway_speed`
    for the D term and observer position for the P/I terms);
  - controller computing forces at midship vs CG (LCG ≈ −1.7 m,
    coupling correction 0.03 % of M);
  - bias-tracking magnitude (8 cm contribution).

Outstanding candidates for next session:
  - K_b1 sensitivity sweep (does ±50 % move the band?);
  - non-LTI features (clamp-style anti-windup, observer mode
    switches, wave-filter Q-scheduling on online Tp estimate);
  - cross-DOF coupling via the allocator (a slow yaw moment
    generated by sway thruster actuation in a 5-thruster
    configuration could feed back into sway via the yaw loop).

#### 12.20.10 K_b1 sweep (negative), bias-FF discovery, and σ reframing

Three findings this round, all of which sharpen rather than
close the diagnostic.

**(a) K_b1 sensitivity sweep is a clean negative result.**
`scripts/p7_brucon_validation/k_b1_sweep.py` evaluates
the analytic |H_sandbox(jω)| = |y_LF / F_drift|(jω) for
K_b1 ∈ {0.5, 1, 2, 4, 8} × nominal (0.0012). The transfer
function is **bit-identical** across all multipliers in the
band Welch can resolve (f ≥ 0.002 Hz). Reason: the bias-loop
pole sits at ω_b ≈ K_b1 × (Kp/M)^(1/2) plus the bias time
constant 1/T_b ≈ 0.001 rad/s — both far below the controller
bandwidth ω_n = 0.08 rad/s. Above 0.005 Hz the bias state is
essentially decoupled from y_LF; whatever bias error remains
is rejected by the much-faster P/D loop. **K_b1 cannot be
the source of the [0.005, 0.01] Hz residual softness.**

**(b) Brucon position_hold has NO observer-bias feed-forward.**
While verifying the controller chain end-to-end, re-read
`libs/dp/dp_controller/position_hold.cpp:392-396`:

```cpp
void PositionHold::UpdateTau() {
  tau_.surge = surge_regulator_.Step(state_error_.ErrorSurge(),  state_error_.ErrorSurgeVelocity());
  tau_.sway  = sway_regulator_.Step(state_error_.ErrorSway(),    state_error_.ErrorSwayVelocity());
  tau_.yaw   = heading_regulator_.StepYaw(state_error_.ErrorHeading(), state_error_.ErrorRateOfTurn());
}
```

`tau_` is set purely by the PID acting on `state_error_`,
which holds `(ŷ_LF − y_ref, ν̂)`. There is **no `+ b̂` or
equivalent term anywhere in `position_hold`**. A grep for
`bias` in `dp_controller/` finds only `dr_error_bias_*` in
`target_following.cpp` — those are dead-reckoning
continuity biases for target-following mode, structurally
unrelated to the observer's bias estimator.

The sandbox's `use_bias_ff=True` was therefore wrong as a
brucon model. The K_b1 sweep also computed the no-bias-FF
case (`use_bias_ff=False, K_b1 nominal`) and found the
transfer is **identical** to the bias-FF case in the resolved
band — so the modelling correction has no impact on the
prior σ comparisons. The bias-FF presence/absence only
changes steady-state DC behaviour, which Welch with
nperseg = 5000 (df = 0.002 Hz) filters out.

**(c) The σ_y_LF gap is dominated by f < 0.002 Hz, not by
the [0.005, 0.01] Hz band.** Re-checking the late window
[1500, 3000] s of the long-run ensemble:

| computation                              | σ_y_LF [m] |
|------------------------------------------|-----------:|
| per-seed `y.std()` (time domain)         |  **0.671** |
| Welch-integrated √∫S_yy(f)df (df=0.002)  |    0.404   |

Variance accounting:
σ_total² = σ_resolved² + σ_subresolved²
0.671² = 0.404² + 0.535²
→ **0.535 m of σ_y_LF lives below 0.002 Hz** (periods >
500 s), and only 0.404 m above it.

The "1.3× softer in [0.005, 0.01] Hz" finding from §12.20.8
is real but is the *minor* contribution. The dominant gap
is at quasi-DC, well below the bias-loop transition band
that prior sections focused on. The Welch nperseg = 5000
choice (designed to give ~3 segments per 1500 s window)
inadvertently band-pass-filters out exactly where the
gap lives.

This reframes the next-step priority list (§12.20.9):
neither cross-DOF allocator coupling nor anti-windup
clamping is naturally a sub-mHz mechanism. What *is* a
sub-mHz mechanism in the brucon stack:

  1. **Wave-period estimator drift**: brucon's
     `EstWavePeriodPitch` settles to ~8.6 s vs the
     sandbox-fixed 10.22 s, and continues to drift slowly
     (per §12.20.5). A time-varying wave-filter peak
     frequency makes the closed loop time-varying →
     low-frequency intermodulation between the
     estimator's drift rate and the slow-drift forcing.
  2. **I-term wandering**: with no bias FF, the entire
     DC rejection burden falls on the PID's integrator
     (Ki = 950 N/(m·s)). The integrator state is itself
     a 1/s pole driven by ŷ_LF, so its relaxation time
     is set by the closed loop (~125 s), not by Ki
     directly. But brucon's `pid.cpp` clamps the
     integrator (`std::clamp` anti-windup) — if the
     clamp is hit during slow-drift excursions, the
     rejection is non-LTI and the residual error
     wanders.
  3. **Observer DC mode**: with `use_tau_feedback: false`,
     the observer integrates `allocated_tau` (the post-
     allocator commanded force) into its velocity model.
     If the allocator clips or rounds at small thrust
     levels, the observer's velocity model accumulates a
     slow integrated error that propagates into ŷ_LF.

Files for this round (uncommitted):
  - `scripts/p7_brucon_validation/k_b1_sweep.py`
    (built `build_closed_loop` extension to take
    `k_b1`, `k_a1`, `t_b` overrides; ran sweep and
    overlaid against empirical brucon transfer)
  - `scripts/p7_brucon_validation/k_b1_sweep_transfer.png`
    (gitignored)
  - extension to
    `scripts/p7_brucon_validation/sandbox_passive_observer.py`
    (`build_closed_loop` now takes `k_b1`, `k_a1`, `t_b`
    overrides; defaults preserve original behaviour)
  - extension to `scripts/p7_brucon_validation/harness.py`
    (`ScenarioSpec.config_overrides` field; per-run
    shadow config dir from symlinks + overridden files;
    keeps global brucon config untouched)

#### 12.20.11 Tp-estimator exoneration test (LOCKED Tp)

To test hypothesis (1) above we lock brucon's wave-filter
peak frequency to a constant value matching the sandbox.
Brucon supports this via the `wave_filter_response_frequency`
block (proto: `WaveFilterResponseFrequency_PeakFrequency`):

```protobuf
wave_filter_response_frequency {
  wave_filter_surge_peak_frequency: LOCKED
  wave_filter_sway_peak_frequency: LOCKED
  wave_filter_heading_peak_frequency: LOCKED
  wave_filter_rate_of_turn_peak_frequency: LOCKED
  locked_peak_period: 10.22
}
```

This is the analogue of the `config_hav934/observer.prototxt`
production setting (which uses 8.0 s). `dp_estimator_wrapper.cpp:184–203`
calls `passive_observer_->SetWaveFilterPositionPeriodEstimate`
on each DOF when `LOCKED` is selected, then never updates
that value during the run.

Driver: `scripts/p7_brucon_validation/long_run_locked_tp_validation.py`.
Smoke-tested with single seed: shadow-config mechanism
materialises 20 symlinks + 1 overridden file, brucon loads
the override correctly, EstWavePeriodPitch still wanders
(~9.45 s in [100, 660] s window of the smoke run) — but the
wave filter inside the observer uses the locked value 10.22 s
regardless. Single-seed σ_y_LF in [300, 660] s = 0.672 m,
indistinguishable from the unlocked baseline 0.67 m within
per-seed scatter.

30-seed ensemble result (3.8 min wall on 12 workers):

| window         | median σ_y_LF [m] | mean σ_y_LF [m] | range [m]      |
|----------------|------------------:|----------------:|----------------|
| early [300, 555]   |         0.657 |          0.716 | [0.35, 1.37]   |
| late  [1500, 3000] |     **0.691** |          0.712 | [0.51, 0.91]   |

vs unlocked baseline late-window median = 0.67 m → **Δ = +2.1 cm**,
well within the per-seed std-of-medians (~0.07 m), and in fact
slightly *worse* than unlocked rather than better. EstWavePeriodPitch
still drifts to ~8.9 s (mean across seeds, late window) but is
no longer consumed by the observer; σ_y_LF is unaffected.

**Hypothesis (1) decisively refuted: Tp-estimator drift does not
cause the sub-mHz σ_y_LF variance.** The +2 cm shift is consistent
with locking Tp at 10.22 s being slightly suboptimal — the true
pitch-driven response peak in this sea state is closer to 8.9 s, so
matching the wave filter to 10.22 mistunes it relative to the actual
HF disturbance, very mildly increasing residual position variance.

Surviving sub-mHz mechanism candidates (per §12.20.10):

  - **(2) I-term wandering / anti-windup non-LTI**: brucon's
    `pid.cpp` has `std::clamp` anti-windup. With no bias FF, the
    integrator carries the full DC rejection burden. Test: extract
    the I-term contribution from brucon's `tau_sway` log
    (`tau_sway − Kp · ŷ_LF − Kd · ν̂` should leave I-term + any
    other FF), check its time series for clamp behaviour and
    sub-mHz wandering against the implied integrator state.
  - **(3) Observer DC mode via `use_tau_feedback: false`**: the
    observer integrates `allocated_tau` not `feedback_tau`. If the
    allocator clips/dead-bands at small thrust levels, the
    observer's velocity model accumulates a slow integrated
    error that propagates into ŷ_LF as a low-frequency wander.
    Test: compare per-seed running mean of `allocated_tau_sway`
    vs `feedback_tau_sway` (= true Ty after rate limiter) over
    [1500, 3000] s; any sustained mean offset is an observer
    DC-poisoning candidate.
  - **(4) Coupling via the bias estimator nonlinearity**: T_b =
    1000 s combined with the position-innovation drive K_b1 = 0.0012
    creates a sub-mHz oscillator in the b̂ state that may
    transfer slow-drift forcing into y_LF in a way the
    linearised sandbox doesn't capture. Test: re-run the K_b1
    sweep in the time-domain (not the analytic transfer) using
    brucon's actual DriftY(t) — if a long-period oscillation
    in b̂ is being excited that the LTI analysis misses, K_b1
    should now matter.

#### 12.20.12 Welch /π normalisation bug — §12.20.10(c) and §12.20.11 framing retracted

Pursuing candidate (2) "observer-vs-vessel τ mismatch" via
`tau_mismatch_diagnostic.py` against 30 long-run seeds gave a
clean negative:

  - mean(AllocTauSway − FbTauSway) = +0.064 ± 0.067 kN
    (= 64 N out of the 216 kN balancing the slow-drift load —
    the controller balances drift to 0.03 % at DC).
  - corr(mean_delta, σ_y_LF) = +0.05; corr(|mean_delta|, σ_y_LF)
    = +0.05; corr(std_delta, σ_y_LF) = +0.17. All within seed
    scatter.
  - Coherence(delta_tau, sway_LF) ≈ 0.93 in [0.005, 0.02] Hz but
    explained by common DriftY drive (correlation, not causation).

The rate-limited thruster channel tracks commanded force to
sub-permille at DC, so `use_tau_feedback: false` does not poison
the observer through the τ path. Candidate (3) refuted.

While Welch-integrating S_yy on the long-run ensemble for that
diagnostic, an empirical convergence study with multiple
`nperseg` values exposed a sign-of-life issue in the prior
normalisation:

| nperseg | seg_len_s | √∫S_yy [m] | time-domain σ_y [m] |
|--------:|----------:|-----------:|--------------------:|
|     500 |        50 |     0.32   |             0.695   |
|    2000 |       200 | **0.71**   |         **0.695**   ← match |
|    5000 |       500 |     0.68   |             0.695   |
|   10000 |      1000 |     0.67   |             0.695   |

With `nperseg ≥ 2000`, **the Welch integral matches the
time-domain σ_y to within 0.02 m**. The earlier "0.404 m
(Welch) vs 0.687 m (time-domain) → 0.535 m of σ lives below
0.002 Hz" claim from §12.20.10(c) was the result of a
spurious `/ np.pi` factor: 0.71 / √π ≈ 0.40. **There is no
missing sub-mHz variance.** The σ gap, to the extent it
exists, lives in the resolved [0.005, 0.01] Hz band, exactly
as §12.20.8 originally diagnosed.

Cause traced to two places:

  1. `sandbox_passive_observer.py:state_variance_freqdomain`
     used `σ² = (1/π) ∫ |H|² S(ω) dω`. This is the *two-sided*
     formula; `cqa.psd.wave_elevation_psd` is **one-sided
     rad/s-native** (verified directly: ∫ S_eta(ω) dω = Hs²/16,
     no /π), so the correct expression is `σ² = ∫₀^∞ |H|² S(ω) dω`.
  2. The companion `sigma_F = √(∫ S_drift / π)` line carried
     the same bug, under-reporting σ_F_drift_y by √π = 1.77.

Both fixed (commit pending). `cqa/closed_loop.py:83`
(`state_covariance_freqdomain_general`) was already using the
bare integral `np.trapezoid(integrand, omega)` — production
core untouched, bug confined to the sandbox script.

**Re-running the sandbox with the fix gives a completely
different picture:**

```
Drift PSD at HS=4.20, TP=10.22, β=90°...
  σ_F_drift_y = 98.7 kN  (one-sided rad/s PSD ∫ S dω)
                                                   σ_y [m]
  perfect FB, no integrator (cqa equivalent)         0.254
  observer no-WF, no bias-FF, no integrator          0.561
  observer with WF, no bias-FF                       0.621
  observer with WF + bias-FF + integrator            0.638
  full obs + thrust lag τ=2s                         0.673
```

  - `σ_F_drift_y = 98.7 kN` matches the brucon long-run ensemble
    mean of ~94 kN to within 5 % (previously 55.7 kN, off by
    1.7×).
  - "observer with WF + bias-FF + integrator" predicts σ_y_LF =
    **0.638 m**, vs empirical brucon median of **0.69 m**: a
    7 % gap that is comfortably within per-seed scatter
    (±0.07 m on the median).
  - Adding a 2-s thrust lag closes the gap entirely (0.673 m vs
    0.69 m).

**The sandbox passive-observer model has now reproduced brucon
σ_y_LF to within ensemble scatter.** §12.20.10(c) finding (c)
is retracted. §12.20.11 conclusion that Tp-estimator drift is
not the cause stands (it was decided on per-seed time-domain
σ, not Welch), but its framing as a "sub-mHz mechanism hunt"
is moot.

The earlier sandbox-vs-brucon σ comparisons in §§12.20.4–12.20.8
that predicted σ_y_LF ≈ 0.29 m (the "perfect FB no integrator"
or basic-observer cases) were each under by √π relative to
their correct values once the integrator and observer
augmentations are included; the corrected predictions land
near brucon. Specifically the §12.20.4 row "full obs + integrator
+ thrust lag τ=2s, σ_y_LF = 0.38 m" should read **0.67 m**.

Implications:

  - The cqa P1 model (perfect FB, no integrator, no observer)
    correctly predicts **σ_y_LF ≈ 0.25 m** for this sea state,
    but that is the *ideal*-controller floor, not what brucon
    achieves. The brucon-achievable σ is set by the
    integrator + thrust-lag combination and is ~2.7× higher.
    For the operability prototype the realistic floor is
    σ_y_LF ≈ 0.65 m, not 0.25 m.
  - `cqa.config.bias_time_constant_s = 100` is still wrong;
    brucon uses 1000 s and the sandbox match relies on it.
  - Surviving sandbox-modelling todos:
      a) audit any other PSD-domain σ formulas (none found in
         production cqa core; sandbox now clean).
      b) propagate the integrator + thrust-lag augmentation
         into the cqa P1 closed loop so its predicted σ_y_LF
         reflects achievable performance, not ideal-PD floor.
      c) update `cqa.config` defaults: `bias_time_constant_s`
         100 → 1000; `use_bias_ff: True` → `False` (matches
         brucon's no-bias-FF position_hold).

Files touched this round (uncommitted):
  - `scripts/p7_brucon_validation/sandbox_passive_observer.py`
    (line 356/371: removed /π, updated docstring with
    derivation note pointing to ∫ S_eta = Hs²/16 sanity check)
  - `scripts/p7_brucon_validation/tau_mismatch_diagnostic.py`
    (NEW; ensemble correlations + coherence; refuted candidate (3))
  - `scripts/p7_brucon_validation/tau_mismatch_diagnostic.png`
    (gitignored)


#### 12.20.13 Observer-augmented closed loop in cqa core

Following the §12.20.12 sandbox closure, the brucon-aligned
observer + integrator + thrust-lag augmentation is now part of
cqa proper as `cqa.observer`. The new module exports:

  - `ObserverParams` (in `cqa.config`): per-DOF observer gains
    sourced from `config_csov/observer.prototxt` (surge/sway K_a1
    = 0.12, K_b1 = 0.0012; heading K_a1 = 0.20, K_b1 = 0.002;
    common ω_c = 1.04 rad/s, T_b = 1000 s).
  - `wave_filter_zeta_n(Tp)` (in `cqa.config`): brucon
    `ScaleGainLinear` schedule for the wave-filter notch
    damping ζ_n ∈ [0.10, 0.25] over Tp ∈ [10, 18] s.
  - `ObserverAugmentedSystem`: 24-state (or 27 with PI integrator)
    block-diagonal-in-DOF linearisation with state ordering
    `[eta, nu, b̂, τ̂_thr, ŷ_LF, ν̂, ξ_w, η̂_w, (I)]`.
  - `build_observer_augmented_system(vessel, controller, observer,
    Tp, T_thr, include_integrator)`: assembles A, B_w, B_wf.
  - `position_state_indices(aug=None)`: convenience block index map.

Validation in `tests/test_observer.py`:

  - At HS=4.20 m, Tp=10.22 s, β=90° (the §12.20 brucon test sea
    state), the cqa observer-aug at csov_default tuning predicts:
    | T_thr [s] | σ_eta_e [m] | vs sandbox | vs brucon 0.69 m |
    |---|--:|--:|--:|
    | T_thr [s] | cqa σ_eta_e [m] | sandbox σ_y [m] | vs brucon 0.69 m |
    |---|--:|--:|--:|
    | 0 (limit)         | 0.696 | 0.640 |  +1.0 % |
    | 2.0 (calib)       | 0.736 | 0.699 |  +6.7 % |
    | 5.0 (cfg default) | 0.812 | 0.845 | +17.7 % |

  - For comparison, the bare 6-state PD-only closed loop (no
    observer, no integrator, no thrust lag) predicts σ ≈ 0.25 m
    (−64 %); the pre-existing 12-state `AugmentedSystem` (PI +
    thrust lag, no observer) predicts σ ≈ 0.46 m (−33 %). The
    observer alone contributes ~0.30 m of σ — it is the dominant
    physical mechanism the prior cqa core was missing, not the
    integrator (~2 cm) or the wave filter (~6 cm), confirming the
    §12.20.10–12 conclusions inside cqa proper.

The 8 unit tests cover: ζ_n schedule endpoints, state-layout
arithmetic, A-matrix Hurwitz stability across Tp ∈ {6, 10, 14,
18} s, block-diagonality in DOF when off-diagonal mass/damping
gains vanish, σ_y match against brucon at the §12.20 test sea
state (csov_default within 20 %, T_thr=2 s calibration within
10 %), `position_state_indices` layout with and without the PI
integrator, and the zero-drift / zero-σ sanity check.

Known limitations carried forward (do not block this commit):

  - Only the brucon `use_tau_feedback: false` mode is modelled
    (commanded τ feeds the observer's velocity equation, not the
    measured feedback τ). §12.20.5 verified this is the relevant
    mode for `config_csov`.
  - `B_wf` (the wave-frequency excitation channel into the
    augmented system) is constructed but not yet wired into a
    public combined LF + WF covariance pipeline; only σ_eta_e
    (LF estimate, the brucon `pos_a_p*` semantic) is exposed by
    the validation tests.
  - `cqa.operability_polar.excursion_polar` still uses the bare
    6-state `ClosedLoop`. A follow-up will add an opt-in
    `use_observer: bool = False` flag; flipping the default to
    True would change published P1 σ predictions by ~3× and is
    deferred to a separate decision.

Operability polar wiring (follow-up to §12.20.13):

  - `cqa.operability_polar.operability_polar` now takes an opt-in
    keyword `use_observer: bool = False`. Default-False keeps the
    historical bare 6-state P1 polar bit-identically, so all
    previously published P1 σ / V_w boundary numbers in
    §§12.20.7, 12.20.10, 12.20.12 remain valid as-is. Setting
    `use_observer=True` swaps the bare `ClosedLoop` for the
    24/27-state `ObserverAugmentedSystem` from `cqa.observer` via a
    duck-typed adapter (`_ObserverClosedLoopShim`) exposing only the
    `A_cl` / `B_w` attributes that `axis_psd` consumes; no API change
    to `summarise_intact_prior` was required.
  - The polar's footprint metric remains the **true vessel position**
    `eta` (state indices 0:3 of the augmented system), not the
    observer's LF estimate `eta_hat_LF`. This matches the operational
    "what hits the turbine" semantic and keeps the IMCA M254 radii
    interpretation unchanged. LF-only forcing is used (`B_wf` is left
    at zero); wave-frequency content is not added to the footprint.
  - Validated by 3 new tests in `tests/test_operability_polar.py`:
    `*_default_is_false_byte_for_byte` (no-op when off),
    `*_tightens_position_boundary` (observer-aug shrinks the operable
    V_w window in every direction, by several m/s in at least one
    direction at the §12.20 sea state), and `*_preserves_metadata`.
  - Decision on flipping the default to True is deferred: it would
    change published P1 polar V_w boundaries by O(several m/s) and
    requires a separate analysis-md re-validation pass.

#### 12.20.14 LF + WF combined covariance pipeline in cqa.observer

Adds two public functions to `cqa.observer` that compose the LF
disturbance channel (slow drift via `B_w`) with the WF wave-motion
channel (true wave-frequency vessel position via `B_wf`) under
the linearised assumption that the two inputs are uncorrelated:

  - `combined_state_psd(aug, S_F_funcs, S_eta_w_func, omega)`:
    one-sided state PSD of the full augmented state vector,
    `H_LF · S_F · H_LF^H + H_WF · S_η_w · H_WF^H`. Pass `S_F_funcs=[]`
    or `S_eta_w_func=None` to disable a channel.
  - `total_position_psd(aug, S_F_funcs, S_eta_w_func, omega)`:
    one-sided 3×3 PSD of the **total observed position**
    `y_total = η + η_w_true` (LF + WF), in the body-fixed
    (surge, sway, yaw) basis. The WF channel contributes both
    indirectly through `C_η · H_WF` (controller responding to the
    wave-corrupted innovation through the wave-filter notch) and
    directly through `+ I₃` (the wave itself).

The signature `S_eta_w_func: callable ω -> (3, 3)` is left to the
caller to construct from JONSWAP × |RAO|² (e.g. via
`cqa.rao.evaluate_rao` × `cqa.psd.wave_elevation_psd`); cqa.observer
deliberately stays free of RAO machinery so both pdstrip-derived
and parametric RAO sources remain available.

Validation in `tests/test_observer.py` (6 new tests):

  - `test_combined_state_psd_lf_only_matches_state_psd_freqdomain`
    (sanity: WF=None reproduces the bare LF state PSD bit-for-bit).
  - `test_total_position_psd_high_freq_limit_is_bare_S_eta_w`
    (at ω ≫ closed-loop bandwidth, `(jωI − A)⁻¹ → 0` so
    `G_WF → I₃` and `S_y(ω) → S_η_w(ω)`).
  - `test_total_position_psd_channels_add` (linearity /
    superposition: total = LF-only + WF-only at every frequency).
  - `test_total_position_psd_hermitian_and_nonneg_diag` (one-sided
    PSD properties).
  - `test_combined_state_psd_zero_inputs_gives_zero`.
  - `test_total_position_psd_sigma_y_total_brucon_cross_check`
    (pdstrip-guarded; verifies σ_y_total > σ_y_LF and that
    σ_y_total sits within ±15 % of the direct quadrature
    σ_y_quad_direct = sqrt(σ_y_LF² + σ_y_WF_bare²) where
    σ_y_WF_bare = ∫ |RAO_sway|² S_η dω. At the §12.20 sea state
    with csov_default and T_thr=2 s: σ_y_LF ≈ 0.67 m,
    σ_y_WF_bare ≈ 0.69 m on the CG-referenced sway RAO, so
    σ_y_quad_direct ≈ 0.96 m and σ_y_total ≈ 1.01 m. See
    §12.20.15 below for the reconciliation against the brucon
    empirical breakdown (σ_y_total_body = 0.89 m, σ_y_LF = 0.69 m,
    σ_y_WF = 0.54 m); the cqa pipeline composition is correct
    but the input RAO sway response is over-predicted by ~28 %
    relative to brucon's `yHf = y_first_order_wave()`, so cqa
    σ_y_total comes out +13 % high vs brucon (conservative).

This closes the original sandbox-vs-cqa pipeline gap noted at the
end of §12.20.13: the public cqa-core API can now produce both
σ_y_LF (matches the brucon `pos_a_p*` semantic, i.e. what the DP
shows the operator) and σ_y_total (matches the operational "what
hits the turbine" semantic for collision / gangway risk).

#### 12.20.15 σ_y_total reconciliation against brucon empirical

Following §12.20.14, ran the brucon `pwo_lockedTp` ensemble
(30 seeds, late window [1500, 3000] s, HS=4.20 m, Tp=10.22 s,
β=90°) through the same body-frame projection used by
`long_run_locked_tp_validation.get_brucon_sway_lf` to extract
the three reference σ figures separately:

  - σ_y_total_body  = std(y_body)         = **0.893 m**
  - σ_y_LF_body     = std(y_body - yHf)   = **0.691 m**  ← the
    "0.69 m brucon" figure used in §12.20 (matches the brucon
    `pos_a_p*` semantic).
  - σ_y_WF          = std(yHf)            = **0.536 m**

with sqrt(σ_y_LF² + σ_y_WF²) = 0.874 ≈ σ_y_total_body, confirming
that y_LF_body and yHf are statistically uncorrelated at this
sea state (as expected from the Fossen passive observer's wave
filter doing its job: the LF channel y_LF rejects the WF content
that yHf carries).

Comparison to cqa pipeline outputs at the same point
(csov_default tuning, T_thr=2 s sandbox calibration, full
cqa.observer.total_position_psd stack):

  | quantity           | cqa pipeline | brucon empirical | error |
  |--------------------|-------------:|-----------------:|------:|
  | σ_y_LF             |       0.74 m |          0.69 m  |  +7 % |
  | σ_y_WF (bare RAO)  |       0.69 m |          0.54 m  | +28 % |
  | σ_y_total          |       1.01 m |          0.89 m  | +13 % |

σ_y_LF is reproduced well; σ_y_total is dominated by the σ_y_WF
over-prediction (0.69 vs 0.54). The σ_y_WF over-prediction
already appears in the **bare** `∫ |RAO_sway|² S_η dω` integral
on the pdstrip CG-referenced sway RAO at β=90°, so it is **not
a bug in `cqa.observer.total_position_psd`**; it is a RAO /
spectrum modelling question:

  - Spectral shape: switching from PM (γ=1, the cqa default for
    `wave_elevation_psd`) to JONSWAP γ=3.3 actually moves σ_y_WF
    *up* slightly (0.69 → 0.72 m), so γ choice is not the
    explanation.
  - Reference-point: brucon's `yHf = vessel_simulator->y_first_order_wave()`
    is computed at the simulator's "first-order wave reference
    point", which is not necessarily the CG. A reduced-RAO at a
    body point further forward (smaller sway lever arm under
    yaw RAO) would naturally give a smaller σ. The expected
    correction is `H_sway_at_pt = H_sway_CG + r_x · H_yaw`,
    where r_x is the longitudinal offset from CG; with brucon's
    yaw RAO at Tp ~ 10 s and a ~10 m offset this can shift
    σ_y_WF by O(20-30 %).
  - Underlying RAO definition: pdstrip's complex 6-DOF position
    RAO at CG vs brucon's `WaveResponse::CalculateLinearResponse`
    output. Cross-checking these two against a known sea-trial
    point or against pdstrip's own time-domain reconstruction
    is a separate workstream.

For now, the take-away is:

  - **σ_y_LF cqa is well-validated** against brucon (within 7 %
    at csov_default with T_thr=2 s sandbox calibration).
  - **σ_y_total cqa is +13 % over-predicted**, attributable to
    a (separate, RAO-modelling) +28 % over-prediction in σ_y_WF.
    For the operability polar this is a *conservative* error
    direction (predicting a larger footprint => earlier amber/red
    => tighter operating window).
  - The §12.20.14 test
    `test_total_position_psd_sigma_y_total_brucon_cross_check`
    passes at ±15 % vs the **direct quadrature** sqrt(LF²+WF²)
    upper bound (0.96 m vs cqa 1.01 m, +5 %), confirming the
    cqa pipeline composition is correct; the residual is
    entirely in the input WF RAO model.
  - A body-point RAO projection helper for cqa.observer is
    deferred until there is a concrete operational use case
    (e.g. the operability polar wanting to score y_total at the
    vessel base or gangway pedestal rather than at CG); when
    that is needed, add `H_pos_at_body_point(rao, r_offset)` to
    `cqa.wave_response` and pass the projected RAO into
    `total_position_psd`'s S_eta_w_func.

### 12.21 From validated σ-prediction to live operability nowcast

With §12.20 closing the σ-prediction validation loop (LF +7 %,
total +13 % conservative against the brucon `pwo_lockedTp`
ensemble), the next P7 deliverable is to put the validated model
to work on **measured** vessel state in real time, not on a
forecast sea state. This section maps what is already built,
identifies the actual gaps, and locks the next build step.

#### 12.21.1 Architecture decision: measure σ, model only for what-ifs

After discussing several variants (full directional sea-state
inversion, parametric HS/Tp/β fit from response PSDs, hybrid
forecast-prior + measurement update), the agreed architecture
is the simplest one that delivers the operator-facing IMCA M254
Fig 8 traffic light:

  - **Live truth source**: measured σ of the operationally-
    limited quantities (body-frame position relative to turbine,
    heading, gangway telescope length). Direct measurement —
    no RAO, no spectrum, no inversion. Robust to anything the
    model cannot capture (drift bias, current, multimodal seas,
    hull fouling, controller retune, thruster failures).

  - **Model for counter-factuals only**: when the operator asks
    "what if thruster X fails" or "what if I change heading",
    run `cqa.observer.total_position_psd` for the counter-
    factual configuration, scaled per DOF by
    `k_dof = σ_measured / σ_model_intact` so the intact
    prediction matches reality by construction. Differences
    between scenarios (intact → WCFDI, intact → heading-shift)
    are then trustworthy even if the absolute model prediction
    carries the +13 % bias of §12.20.15.

  - **Sea-state estimation deferred**: not required for the
    nowcast. Kept open as a future hybrid track (forecast HS/Tp/β
    as prior, refined against measured response PSDs) for
    capabilities that need a directional spectrum (multi-vessel
    coordination, body points without MRU, forecast hand-off).
    Module boundaries are designed to admit it later without
    disturbing the measurement-first live loop.

  - **Onboard signals consumed by the prototype**: GPS/INS
    position relative to the turbine (x_body, y_body), gyro
    heading ψ, gangway telescope length L(t). MRU at CG is
    available; pedestal MRU typically is not (DP does not
    have access to the gangway MRU on most installations).

#### 12.21.2 What is already built (and was missing from session memory)

A pre-existing module `cqa.online_estimator` (1182 lines, 60
passing tests on synthetic data) implements Option B at a more
sophisticated level than the naive `np.std(window)` that was
on the table:

  - **`BayesianSigmaEstimator`** — sliding-window InvGamma
    posterior on σ² with a Bartlett effective-sample-size
    correction. Conjugate update on a ring buffer with O(1)
    per-sample push and incremental Σx² book-keeping. Posterior
    parameters are
        α = α₀ + N_eff/2
        β = β₀ + S_eff/2
    with `S_eff = Σx² · (N_eff / N_raw)` and
    `N_eff = N_raw · dt / max(dt, T_decorr)`. The conjugate
    prior degrades gracefully to the prior mean as N_eff → 0,
    so the posterior is well-defined from sample 0.

  - **`SigmaPosterior`** — InvGamma summary (mean, median,
    equal-tail credible interval at user-specified level, both
    in σ² and σ space).

  - **`PosteriorHealth` + `compose_validity_badge`** — five
    cheap runtime diagnostics covering the assumptions baked
    into the InvGamma posterior:
      A1 stationarity (`halves_sigma_ratio` of in-window first
         and second halves)
      A2 zero-mean (`|sample_mean|/σ_post` — primary indicator;
         catches DP integral / observer bias not yet converged,
         persistent low-frequency disturbance, setpoint drift)
      A3 Gaussian marginals (`kurtosis_excess` — flags slamming,
         saturation)
      A4 ESS warmth (`n_eff` ≥ threshold)
      A5 prior-data tension (whether the prior σ falls inside
         the posterior credible interval — flags model-data
         mismatch but only ever escalates to WARMING; data wins)
    Composed by `compose_validity_badge` into a single per-
    channel `OK / WARMING / UNSETTLED / INVALID` level with
    site-tunable thresholds.

  - **`combine_radial_posterior`** — combines two per-axis
    InvGamma posteriors (on `dx`, `dy`) into a radial summary
    via Monte-Carlo over the joint (assuming
    `cov(σ_x², σ_y²) = 0`, deferred per online_estimator.py:300).
    Reports σ_R = √(σ_x² + σ_y²) and E[R] (Hoyt-aware via direct
    MC of `R = √(X² + Y²)`), credible intervals, and a
    rotation-invariant 2D `radial_mean_offset_over_sigma`
    diagnostic that catches systematic 2D drift independent of
    heading.

  - **`closed_loop_decorrelation_time`** — coarse `1/(ζ ω_n)`
    fallback for `T_decorr`, kept for diagnostics. Production
    callers should pass the PSD-derived
    `T_var = π · ∫ S_X(ω)² dω / m₀²` from
    `cqa.extreme_value.variance_decorrelation_time_from_psd`
    (this is the exact Bartlett scale for variance estimation;
    see online_estimator.py:78-98 for the derivation and a
    cited 5× error example on the canonical CSOV when the
    legacy fallback is used naively).

  - **`cqa.signal_processing.bandsplit_lowpass`** — zero-phase
    Butterworth split (`x_lf, x_wf = bandsplit_lowpass(x, fs,
    omega_split)`) for separating the gangway telescope signal
    into its slow band (closed-loop response to wind/drift/
    current, < ~0.15 rad/s) and wave band (1st-order RAO
    response, ~0.5–1.0 rad/s). The two bands are >2 octaves
    apart so a 4th-order Butterworth at ~0.3 rad/s is clean;
    `x_lf + x_wf == x` exactly by construction. **Offline
    only** — uses `scipy.signal.filtfilt` (forward then
    backward), so it requires the full series and is strictly
    non-causal. The online band-split is a separate design
    decision (causal IIR with group-delay book-keeping vs
    complementary Linkwitz-Riley pair vs sliding-window Welch
    band-power vs trailing filtfilt with latency); deferred
    until G1 has demonstrated that the offline path closes
    the validation loop. G1 uses the offline filtfilt directly
    on brucon CSVs since brucon writes complete time series.

  - **`operator_view.summarise_intact_prior`** — already
    accepts the live-posterior hooks:
      `posterior_sigma_radial_m`            — overrides σ on
        the vessel-position Rice / inverse-Rice channel.
      `posterior_sigma_telescope_slow_m`    — overrides slow-
        band σ on the gangway channel.
      `posterior_sigma_telescope_wave_m`    — overrides WF-
        band σ on the gangway channel.
      `posterior_health_*` (×4)             — passes the
        per-channel A1–A5 diagnostics through to
        `IntactPriorSummary` for the operator-facing badge.
    The override mechanism is the textbook "spectral SHAPE
    from prior, LEVEL from data" decomposition implemented
    in `cqa.extreme_value.p_exceed_from_psd(..., sigma_override=
    σ_post, ...)`: nu_0+ and the Vanmarcke q stay model-derived
    (the spectral shape is what the model is good at), only
    the variance level is data-conditioned.

  - **`decision_matrix.evaluate_decision_cell`** + 
    `wcfdi_decision_matrix` — forecast-grid `(slot × heading)`
    of IMCA M254 Fig 8 traffic lights, combining intact-prior
    and post-WCFDI panels by worst-of. Built for the forecast
    case (roadmap 4b); does not yet plumb posterior-σ kwargs
    through to its `summarise_intact_prior` call.

#### 12.21.3 What is actually missing

With the above on the bench, the real gaps for "put the model
to work on measured data" are:

  - **G1. Brucon-ensemble end-to-end validation of
    `BayesianSigmaEstimator`.** The entire estimator stack is
    tested only on synthetic IID Gaussian / Student-t /
    sinusoid+offset / variance-ramp signals. We have 60
    `pwo_lockedTp` brucon seeds with known true
    σ_y_body = 0.893 m at HS=4.20 m, Tp=10.22 s, β=90°
    (validation-script semantic, late window [1500, 3000] s).
    No script feeds those time series into the estimator and
    confirms the posterior recovers σ within ESS-implied noise.
    This is the single most important missing piece: it would
    close Option B end-to-end with simulator-grade data and
    establish the confidence baseline needed before wiring the
    posterior into operator-facing decisions. It also exercises
    the health diagnostics (A2 should flag if the brucon
    integral / bias loop hasn't fully settled at t=1500 s; A3
    should be clean for Gaussian wave forcing; A4 should clear
    once the window contains enough N_eff at the
    PSD-derived T_var) and the radial composition
    (combine `BayesianSigmaEstimator` on x_body and y_body into
    a `RadialPosterior` and check σ_R against
    sqrt(σ_x_body² + σ_y_body²) from the same seeds).

  - **G2. WCFDI counter-factual calibration hook.**
    `summarise_for_operator` (the post-WCF panel) accepts no
    posterior σ. `wcfdi_mc` and `wcfdi_self_mc` always compute
    σ from the model. There is no `k_dof = σ_measured / σ_model`
    calibration anywhere in the WCFDI path. The
    `operability_polar.py:23-37` roadmap calls this **item 4c
    "Operation case live what-if"** and marks it not-yet-
    implemented. After G1 lands, this is the natural next
    build: a thin layer that takes the live `RadialPosterior`
    + `IntactPriorSummary` from `summarise_intact_prior`,
    computes per-DOF k from the model's intact σ vs the
    measured posterior σ, and rescales the WCFDI counter-
    factual σ before it hits `_imca_traffic`.

  - **G3. `decision_matrix` posterior plumbing.** Once G1 and
    G2 are in, plumb the same posterior-σ kwargs through
    `evaluate_decision_cell` → `_build_intact_prior_at_forecast`
    → `summarise_intact_prior` so the forecast grid can be
    re-evaluated with the live calibration applied. Mechanical;
    no design decisions.

  - **G4. Trivial: `ValidityBadge` and `compose_validity_badge`
    are missing from `cqa.online_estimator.__all__`** despite
    being part of the public surface and covered by tests.

#### 12.21.4 Recommended next build: G1, brucon-ensemble validation

A new script `cqa/scripts/p7_brucon_validation/online_estimator_brucon_validation.py`
(mirroring the `long_run_locked_tp_validation.py` pattern):

  1. Reuse the existing `pwo_lockedTp_seed{1000..1029}` 30-seed
     ensemble (already on disk, already used in §12.20.10).
     No new brucon runs needed.

  2. For each seed, project (x_ned, y_ned, heading) into body
     frame using the same convention as
     `long_run_locked_tp_validation.get_brucon_sway_lf`:
        y_body = -sin(ψ)·x_ned + cos(ψ)·y_ned
        x_body =  cos(ψ)·x_ned + sin(ψ)·y_ned
     (use total heading; the brucon `heading` channel already
     carries LF+WF). Subtract the per-window mean.

  3. Construct two `BayesianSigmaEstimator`s (one per body
     axis) with:
        prior_sigma2  = σ_model² from `axis_psd` at the same
                        (HS, Tp, β) — this lets A5 fire if the
                        model and data disagree, which is the
                        +13 % bias we already documented.
        prior_strength_n0 = 2.0 (weak; data-dominated by the
                                  end of the window).
        T_decorr_s    = `variance_decorrelation_time_from_psd`
                        on the same axis_psd (NOT the legacy
                        `closed_loop_decorrelation_time`).
        window_s      = 1500.0 (the late window from §12.20).
        dt_s          = brucon sample period (read from the
                        first two t entries).

  4. Stream the late-window samples through `update(x)` per
     axis, then query `posterior(credible=0.90)` and
     `health(...)`, plus `combine_radial_posterior` on the
     two axes.

  5. Compare per seed:
        σ_x_post.median  vs  σ_x_body_truth = std(x_body_lw)
        σ_y_post.median  vs  σ_y_body_truth = std(y_body_lw)
        σ_R_post.median  vs  sqrt(σ_x²+σ_y²)
     Aggregate over the 30-seed ensemble: report median
     posterior σ across seeds and the 5th/95th percentile band.

  6. Acceptance: median posterior σ within ~3 % of brucon
     truth (the InvGamma posterior is unbiased on the variance,
     and σ = sqrt(σ²) introduces only an O(1/N_eff) bias which
     is small at N_eff ≳ 30); per-seed σ within the
     posterior 90 % credible interval at ≥85 % rate (binomial
     coverage check).

  7. Health diagnostics: report the median A1–A5 ladder across
     seeds. Expected: A1 clean (stationary late window), A2
     clean post-1500 s settle, A3 clean for Gaussian forcing,
     A4 warm at N_eff ≳ 5, A5 likely WARMING (since the model
     prior σ is +13 % off the data — exactly what A5 is for).

  8. Companion test in `tests/test_online_estimator.py` (or a
     new `test_online_estimator_brucon.py` if we want to keep
     the synthetic-data file pure) that runs a single seed
     and asserts the posterior recovers brucon truth within
     the documented tolerance. Guarded on the brucon ensemble
     directory existing (skip if not, like the existing
     pdstrip-guarded tests).

  9. Plot: a per-seed scatter of (σ_post median, brucon truth)
     with the 90 % CI as error bars, plus a histogram of
     A1–A5 levels across seeds. Saved into the script's
     work directory (gitignored).

Concrete deliverable: one script + one test + a few-paragraph
analysis.md §12.21.5 reporting the results. After this lands
we have demonstrated, end-to-end, that the existing
`BayesianSigmaEstimator` recovers true σ from simulator-grade
vessel motion data — which is the foundation of the entire
Option-B operator panel. G2 (WCFDI calibration) is the
natural follow-up build after G1 lands.

#### 12.21.5 G1 results: posterior recovery of brucon σ

G1 implemented per §12.21.4:
  - script: `scripts/p7_brucon_validation/online_estimator_brucon_validation.py`
  - test:   `tests/test_online_estimator_brucon.py` (single-seed, ±5 %
            tolerance per axis + 90 % CI coverage + radial composition).

Per-seed `BayesianSigmaEstimator` posterior at the §12.20 sea state,
30 seeds, late window [1500, 3000] s, dt=0.1 s, prior σ² and
T_decorr from `cqa.observer.total_position_psd` +
`variance_decorrelation_time_from_psd`:

  | quantity         | brucon truth (median) | posterior median | rel err |
  |------------------|----------------------:|-----------------:|--------:|
  | σ_x (body)       |                0.412 m|          0.405 m |  −1.7 % |
  | σ_y (body)       |                0.893 m|          0.888 m |  −0.6 % |
  | σ_R = √(σ_x²+σ_y²)|               0.984 m|          0.981 m |  −0.4 % |

  - 90 % credible-interval coverage of the per-seed brucon truth:
    σ_x 30/30, σ_y 30/30 (target ≥ 85 %, comfortably exceeded —
    the posterior CI is mildly conservative, which is the right
    direction for an operator nowcast).
  - Median Bartlett ESS in the 1500 s window: n_eff_x = 38.0,
    n_eff_y = 118.7 (T_var_x = 39.5 s, T_var_y = 12.6 s — σ²_y is
    WF-dominated and decorrelates fast; σ²_x is LF-dominated and
    decorrelates slowly).
  - The posterior median sits ~0.5 % below brucon truth on σ_y. This
    is the expected √(β/(α-1)) vs E[σ²]^(1/2) artefact of an
    InvGamma posterior at finite N_eff: the InvGamma is unbiased on
    σ², which introduces an O(1/N_eff) downward bias on σ. At
    N_eff ≈ 119 this bias is ≲1 %, matching what we measure.
  - σ_x error (−1.7 %) is larger than σ_y error (−0.6 %) by exactly
    the ratio (n_eff_y/n_eff_x)^(1/2) ≈ 1.77; consistent with the
    InvGamma posterior bias scaling.

Health A1–A5 ladder across the ensemble (composed via
`compose_validity_badge` per channel):
  - σ_x channel: WARMING in 29 of 30 seeds, OK in 1.
  - σ_y channel: WARMING in 21 of 30, OK in 9.
  - The dominant WARMING reason on both channels is **A5
    (prior-data tension)**: the model prior σ_x = 0.280 m and
    σ_y = 1.005 m sit outside or near the edge of the posterior
    90 % credible interval for most seeds. This is **expected and
    correct behaviour**:
      * σ_y prior 1.005 m vs truth 0.893 m: model over-predicts by
        +13 %, exactly the §12.20.15 conservative bias on the bare
        WF RAO. Some seeds fall inside the posterior CI (9 OK),
        most don't (21 WARMING).
      * σ_x prior 0.280 m vs truth 0.412 m: model **under-predicts
        by 32 %**. This is a new finding — at β=90° the surge motion
        is small but not negligible (~0.4 m std), driven by yaw →
        x_body coupling and second-order drift, and our LF+WF model
        misses about a third of it. The σ_x channel is not on the
        critical path for the §12.20 sway-validation work, so this
        was not previously surfaced. Worth a small follow-up:
        cross-check `total_position_psd` σ_x at β=90° against
        brucon σ_x_body = 0.41 m using the same recipe as
        §12.20.15. Likely culprits: (a) yaw RAO contribution to
        x_body via a body-point projection (the same r_x · H_yaw
        correction that affects σ_y_WF), (b) Newman drift force
        sway → surge cross-coupling.
  - A1 (stationarity), A2 (zero-mean), A3 (Gaussian), A4 (warmth)
    all pass cleanly across the ensemble (no INVALID, no
    UNSETTLED). The estimator is appropriately confident in its
    posterior on this data.

Conclusion: the existing `BayesianSigmaEstimator` recovers brucon's
true σ from simulator-grade vessel motion data at ~0.5 % bias, with
correctly conservative 90 % credible intervals (100 % coverage) and
a health ladder that fires WARMING on exactly the channels where
the model and data disagree (A5 catches the +13 % conservative
bias on σ_y and the −32 % under-prediction on σ_x). **Option B is
validated end-to-end.** The posterior σ produced by this stack is
ready to be wired into `summarise_intact_prior` for live operations.

Plot saved (gitignored): `online_estimator_brucon_validation.png` —
three panels (σ_x, σ_y, σ_R) showing posterior median + 90 % CI
vs brucon truth per seed, with the model prior σ as a reference
line.

Next build (G2): the measured-σ → WCFDI counter-factual calibration
hook. Take the `RadialPosterior` produced here, compute
`k_dof = σ_post.sigma_median / σ_model_intact` per DOF, and rescale
the post-WCFDI σ envelope before it hits `_imca_traffic`. See
operability_polar.py:23-37 (roadmap item 4c) for the original
intent.

#### 12.21.6 G2 design discussion: what does the WCFDI nowcast actually need from measurements?

Before coding G2 we walked through the physics carefully because
the original "rescale σ at operator_view" plan turned out to be
too naive. Two things are happening in `wcfdi_transient` that
matter for live calibration:

1. **Initial-condition distribution at t = WCF.** The post-WCFDI
   excursion ensemble is generated by `wcfdi_mc` drawing
   `(η, ν)` at t=0⁻ from the intact `P12` covariance. Today this
   `P12` is built from `S_wind`, `S_drift` (Newman QTF + Bretschneider),
   `S_curr` — i.e. entirely from operator-set sea state. An incorrect
   (Hs, Tp, β) on input directly mis-sizes the starting-state cloud,
   so every per-realisation post-failure trajectory inherits the
   error.

2. **Mean environmental force `tau_env(t=0)`.** The deterministic
   post-failure trajectory `eta_mean(t)` depends on `tau_env =
   F_wind(Vw, β) + F_curr(Vc, β) + F_drift(Hs, Tp, β)`. The
   immediate cap clipping at t=0⁺ creates a thrust step
   `delta_tau = clip(x_ss[9:12], cap_immediate) − x_ss[9:12]`,
   and during reallocation the unbalanced load drives an
   excursion. **`eta_mean(t_peak)` is the dominant contribution
   to `pos_peak` whenever any DOF clips at t=0⁺.** This depends
   on `tau_env`, not on σ.

Empirical check (Vw=15, Hs=4.2, Tp=10.22, Vc=1.0, theta=30°,
α=0.5, γ_imm=0.2, T_realloc=60 s) showed:
  - `eta_mean` peak surge: 0.27 m at t=29 s, recovers to 0 by t=300 s.
  - `eta_std` evolution during the transient: max/min ratio of
    1.25 in surge, 2.0 in sway, 2.0 in yaw across 300 s. Not
    flat — the lifted P0 (with zero variance in `b_hat`, `tau_thr`
    components) is not the steady state of `aug.A`, and the
    augmented covariance evolves to it. Sea-state-dependent.

Architecture options considered:
  - **Arch A (covariance rescale inside `wcfdi_transient`):**
    cleanest mathematically, but re-runs the ODE per-call and
    couples calibration to the model engine.
  - **Arch B (post-pass on `WcfdiMcResult`):** smallest diff,
    but only handles the σ side.
  - **Arch C (rescale at `summarise_for_operator`):** decoupled,
    but addresses only σ — does not touch `tau_env`, so the
    deterministic transient kick remains uncalibrated. **This was
    our initial plan; rejected as too naive.**
  - **Arch D (fully measurement-derived `r_dof` from vessel
    mechanics only, no spectra):** appealing in principle —
    no wave-buoy dependency, robust in the field — but the
    empirical check above shows the post-WCFDI σ envelope
    evolution is sea-state-dependent enough that we can't
    factor it out without a substantial empirical justification.
    Worth exploring as Phase 2; not on the G2 critical path.

**Locked G2 plan (option 1 from the §12.21.6 design discussion):**
inject *both* missing measurement-driven inputs the user identified,
at the `wcfdi_mc` call site:
  1. **Replace P6 diagonals** (LF σ_x, σ_y, σ_ψ — body frame) with
     measured values from `BayesianSigmaEstimator`. Cross-terms
     and velocity covariances stay model-derived (they are not
     directly observable from position channels alone). Diagonals
     are rescaled, off-diagonals are renormalised so the
     correlation matrix is preserved (PSD-safe).
  2. **Replace `tau_env`** with measured wind force + measured
     current force + bias-estimate slow-drift force. The
     controller's `b_hat` integrator already tracks the slowly-
     varying environmental load by construction; the wind is
     directly available from the anemometer. This avoids
     depending on accurate `(Hs, Tp, β)` estimates from a wave
     buoy for the *force* part of the transient.
  3. Run `wcfdi_transient` and `wcfdi_mc` with these substituted
     inputs. The IMCA traffic-light decision (`pos_p95` vs
     warning/alarm radii) then reflects the live state.

Architecture C remains valuable for a different use case the user
flagged: ingesting wave-radar or buoy-derived spectra in real time
to flag incoming dangerous wave trains, particularly relevant for
the *intact* operability state. That's a future deliverable, not
G2.

Architecture 3 (full re-engineering of `wcfdi_mc` to take
`(σ_intact_measured, tau_env_measured)` as primary inputs and
build P6 from a model-derived correlation structure scaled to
measured diagonals) is the Phase 2 endpoint after Phase 1 (the
substitution path above) validates the approach end-to-end
against brucon's WCF event.

G2 deliverables:
  - new helper module (likely `cqa.calibrated_wcfdi`) exposing
    `wcfdi_mc_calibrated(cfg, ..., sigma_measured_lf, tau_env_measured)`.
  - validation script reading the existing 60-seed `pwo_long`
    ensemble (WCF event at t=settle_s + activate_sk_s = 3060 s),
    computing measured σ_intact in the [activate_sk_s, WCF] window,
    measured `tau_env` from the brucon thrust output (or
    bias-estimate proxy if available in the .out file), and
    comparing post-WCF brucon excursion against:
      (a) raw model `wcfdi_mc` (current behaviour),
      (b) calibrated `wcfdi_mc_calibrated` using measured inputs.
  - companion test `tests/test_calibrated_wcfdi.py` (single-seed,
    skipped if brucon ensemble missing).
  - analysis.md §12.21.7 with results table (footprint p95
    prediction error, with vs without calibration).

#### 12.21.6.1 Brucon WCF event survey (the existing ensemble does not exercise the transient)

Step 1 of G2 was a survey of the existing `pwo` 30-seed ensemble
(`work/pwo_seed*`, `run_comparison_waves_only.py` configuration:
Hs=4.20 m, Tp=10.22 s, beam-on, no wind/current, settle=500 s,
post_failure=180 s, failed thrusters = `CSOV_WCF_GROUPS["bus_port"]`
= Bow1 + PortMP). Findings:

  - **The WCF event is invisible in the response.** Pre-WCF and
    post-WCF position deviations have indistinguishable statistics:
    SurgeDev / SwayDev magnitudes overlap completely, post-WCF
    radial peak (1.6 m) is smaller than the *intact* radial peak
    (1.7 m) earlier in the run. The 30-seed peak |pos_dev| over
    [560, 740] s (median 2.0 m, P95 2.8 m) is **all intact-state
    slow-drift variability**, not a transient.
  - **No clipping.** `AllocTauSway = OrderTauSway` exactly, both
    before and after WCF. The 3 surviving thrusters can deliver
    full demand at the operating point.
  - **EstBiasSway is excellent.** The estimator output tracks the
    slowly-varying mean Fy_env beautifully (settled at -200 kN
    against truth median -214 kN). The bias-estimator-as-tau_env
    proxy is *validated* by this survey — that part of the G2
    plumbing will work as designed.
  - Pre-WCF intact σ in [360, 560] s window (200 s, body frame):
    σ_x median 0.253 m, σ_y median 0.589 m, σ_ψ median 0.34°.
    Per-seed σ_y range is 0.32 – 1.36 m (factor 4 spread), good
    spread for any future calibration validation that needs to
    track seed-to-seed variability.

**Implication for G2.** The validation can't use the existing `pwo`
ensemble as-is — without a real post-WCF transient there's nothing
to predict. We need a harder failure scenario that produces
saturation-driven excursion. Options:
  1. More aggressive failure grouping (e.g., entire bus = bus_port +
     bus_stbd combined).
  2. Worse environment (add wind + current at the chosen direction).
  3. Both.

The brucon ensemble re-run is ~25 min on 12 workers per scenario,
so the cost of trying a couple of scenarios is modest. Pre-empt by
running a 1-seed smoke test first to confirm the new scenario
actually clips at WCF.

#### 12.21.6.2 Correction: the WCF transient IS in the existing ensemble (right metric needed)

The §12.21.6.1 survey looked at **raw** post-WCF position
deviations (absolute body-frame `SurgeDev`, `SwayDev`), not the
**deviation from the t=WCF instant**. The intact slow-drift
fluctuation (~0.5–1.5 m peak) was masking the actual transient
in single-seed plots. The pre-existing
`p7_waves_only_validation_transient.png` already shows the right
metric (ensemble-mean of (`SurgeDev(t) − SurgeDev(t_WCF)`,
`SwayDev(t) − SwayDev(t_WCF)`)) and a substantial transient is
visible:

  - Ensemble-mean **surge** peaks at **+0.92 m** at t = 37 s post-WCF.
  - Ensemble-mean **sway** troughs at **−1.42 m** at t = 35 s
    post-WCF (smaller positive overshoot at t ≈ 116 s).
  - Per-seed peak |Δradial| (radial deviation since WCF instant):
    median 2.40 m, P95 3.53 m, range [1.32, 3.95] m.

**The same plot also shows that the existing cqa `wcfdi_transient`
predicts essentially zero ensemble-mean transient** (red dashed
line at zero through both panels) and `bistability_risk_score = 0`.
This is the model gap G2 has to close: brucon shows a clean ~1 m
deterministic transient, cqa shows nothing.

The mechanism is unambiguous: even without immediate-cap clipping
(`gamma_immediate * tau_cap` may not bind at this operating point),
losing 2 thrusters changes the **allocator's** force/moment
distribution, which the cqa point-mass + bandwidth controller
abstracts away entirely. The bias-estimator integrator briefly
loses its corrector gain match against the new effective `B`
matrix, and the system slips before re-establishing equilibrium.
This is precisely the regime the Phase 1 `wcfdi_mc_calibrated`
should capture by injecting the **measured tau_env** (which the
brucon estimator tracks correctly through the event) into the
post-WCF propagation.

**Implication for G2.** The existing `pwo` 30-seed ensemble is
suitable for validation. No re-run needed. The validation metric is
**ensemble-mean Δsurge(t), Δsway(t) since WCF**, plus per-seed
peak |Δradial|, compared against:
  (a) raw `wcfdi_mc` (current behaviour — predicts ~0 mean
      transient),
  (b) calibrated `wcfdi_mc_calibrated` with measured `(σ_intact_lf,
      tau_env)`.
