# cqa — Continuation prompt

Snapshot of the cqa (combined capability + excursion / consequence
analysis) prototype as of the commit that introduces this file. Copy
the contents below verbatim into the next chat session under a
"Recap" or "Continuation" heading and the assistant will resume
cleanly.

---

## Goal

Build a feasibility study (and progressively a Python prototype, eventually leading to a C++ demonstrator in `~/src/brucon`) for **online estimation of vessel station-keeping footprint and walk-to-work (W2W) gangway operability**, evaluated for both **current intact state** and **post-WCFDI state** (worst-case failure design intent under DP-class-2 redundancy). Continuously running while on DP, giving operator situational awareness of the post-failure outcome envelope at the *current* operating point.

**Dual deliverable** (formalised in `analysis.md` §1.1):
- **A. Online "better consequence analysis"** (IMO MSC.645 / DNVGL-OS-DPS-1 / IMCA M103, M220).
- **B. Offline footprint extension to static capability analysis** (IMCA M140 rev.2 / DNV-ST-0111).

Operator-facing semantics follow **IMCA M254 Rev.1 Figure 8 "Decision matrix"**: two independent capability axes with green/amber/red traffic lights:
- **Vessel capability (footprint)**: green <2 m, amber >2 m, red >4 m.
- **Gangway capability (utilisation)**: green <60 %, amber >60 %, red >80 %.

**Operator-framework structure (4 questions):**
1. **Intact, prior to operation**: P50/P90 length-scale quantiles of running max from Rice/Cartwright-Longuet-Higgins inversion. Default T_op=20 min. **DONE.**
2. **Intact, during operation**: same Rice formula but with **Bayesian-updated σ²** from a 60 s sliding window. **DONE (estimator wired into `summarise_intact_prior` via `posterior_sigma_*` kwargs).**
3. **Post-WCF, conditional**: existing IMCA M254 Fig.8 panel ("if WCF now: ..."). Time horizon = transient settling, NOT op duration. **DONE.**
4. **σ prior vs σ posterior**: shown side-by-side once the online estimator is wired. **NOT YET IN DEMO** (estimator + integration done; no side-by-side panel yet).

## Instructions

- **Target vessel:** brucon `config_csov` (Norwind SOV: Lpp=101.1 m, B=22.4 m, T=6.50 m, Cb=0.737, lateral wind area 1755 m², frontal 700 m², gangway base body-frame `(5.0, -9.0, -8.0) m`, telescope `L ∈ [18, 32] m`, max rotation-centre travel 25 m, default mid-stroke h=12.5 m). Production must remain config-agnostic.
- **Roadmap status:** P0→P3 done; P3.5/P3.7/P3.8 done; P3.9 Rice + Vanmarcke clustering DONE; P3.9b length-scale (P50/P90) operator panel DONE; σ-inconsistency Tier 1 fix DONE (controller tuning unified via `cfg.controller`); P3.10 online Bayesian σ² estimator DONE (module + tests + wiring + 3 integration tests); **operability polar DONE** (Level-3-light intact-prior capability sweep over wind direction with PM wind-wave law); P4 vessel_simulator validation, P5 DP-playback validation, P6 go/no-go gate, P7 C++ demonstrator + GUI.
- **Reuse from brucon:** `CapabilityAnalysis`, `ConsequenceAnalysis`, `BasicAllocator`, `VesselModel`, `WaveResponse`, `WindForceModel`, `dp_estimator`, `vessel_simulator`, `dp_runfast_simulator`. brucon `LandingPoint` only models the structure-side anchor; gangway *mechanism* kinematics built fresh in `cqa/gangway.py`.
- **Wave spectrum:** assume **not directly measured onboard**; sources by priority — forecast (NORA3/WW3), wave radar, or wind-sea wind/wave-buoy analogy. Use parametric JONSWAP/Torsethaugen.
- **Gangway operability gating:** **telescope only** (end-stops + stroke velocity). Slew α and boom β are display-only.
- **WCFDI dominant transient source = thrust reallocation lag.** Per-DOF cap drops at t=0 to `gamma_immediate · cap_intact` (default 0.5), recovers exponentially with `T_realloc` (default 10 s) toward `alpha · cap_intact`.
- **Operator-facing presentation philosophy:** "boil down to almost 1 number" — operators have rarely seen WCFs and need actionable info. **IMCA M254 Fig.8 framing**: traffic lights, units in meters per the user's preference.
- **IMCA M254 thresholds** match cqa defaults: `pos_warning_radius_m=2.0`, `pos_alarm_radius_m=4.0`, `gw_imca_warning_frac=0.60`, `gw_imca_alarm_frac=0.80`.
- **Drift-PSD convention:** **arithmetic-mean Newman variant** `T(ω₁,ω₂) ≈ ½(D(ω₁)+D(ω₂))`, matching brucon `wave_response.cpp:333`.
- **Short-crested default:** cos-2s s=15 (DNV-RP-C205 wind sea, ~21° one-sigma).
- **Stack:** Python (numpy/scipy/matplotlib). Dedicated venv at `cqa/.venv`. Use `/home/blofro/src/hydro_tools/cqa/.venv/bin/python` for tests/demo. **The venv is NOT committed; reproduce from `pyproject.toml`.**
- **Validation strategy:** brucon `vessel_simulator` time-domain runs at later stage.

## Discoveries

### Online estimator (P3.10)
- **Conjugate inverse-gamma:** `σ² ~ InvGamma(α₀, β₀)` with prior parameterised as `prior_sigma2 = β₀/(α₀-1)` and `prior_strength_n0 = 2α₀`. Default `n0=2.0` (very weak; soft-clipped to α₀=1+1e-6 internally so prior mean is finite).
- **Effective sample size (Bartlett):** `N_eff = N_raw · dt / max(dt, T_decorr)`. Sufficient statistic rescaled `S_eff = S · N_eff/N_raw`.
- **Posterior:** `α = α₀ + N_eff/2`, `β = β₀ + S_eff/2`. Mean `β/(α-1)` for α>1, median + equal-tail credible interval via `scipy.stats.invgamma`.
- **T_decorr:** `1/(ζ·ωₙ)` per axis. For position channel, take max of surge/sway (slowest contributor).
- **Zero-mean assumption** by default (DP regulates to setpoint). `assume_zero_mean=False` available (subtracts windowed sample mean, loses 1 dof).
- **Ring buffer** with O(1) incremental sum-of-squares update.
- **`is_warm`** is a method (not a property): `est.is_warm(n_eff_threshold=1.0)`.
- **Spectral SHAPE kept from prior, only LEVEL updated from data** — `summarise_intact_prior(..., posterior_sigma_radial_m=..., posterior_sigma_telescope_slow_m=...)` substitutes σ but keeps ν₀⁺ and q from the prior closed-loop spectrum. Wave-frequency telescope band is intentionally NOT data-updated (it's RAO-driven, not DP-driven).

### σ-inconsistency root cause (FIXED Tier 1)
The intact-prior σ_radial = 1.31 m vs WCFDI MC σ ≈ 1.0 m gap was due to:
- **Tier 1 (FIXED):** demo and `wcfdi_mc` used different controller tuning. Now unified through `cfg.controller` (`ControllerParams` dataclass with `omega_n_*`, `zeta_*`, `T_b`, `T_thr`).
- **Tier 2 (NOT FIXED, ~5% residual):** WCFDI MC uses 12-state augmented system (`aug.A`); intact-prior uses 6-state direct PD closed loop. Different transfer functions.
- **Tier 3 (NOT FIXED, doesn't fire under current config):** when pdstrip data missing, WCFDI MC falls back to parametric Newman drift PSD; intact-prior falls back to **zero**. Cosmetic for the demo since pdstrip file is present.

After Tier 1 fix at canonical OP (CSOV, bow quartering 30°, Hs=2.8, Tp=9, Vw=14): both panels read AMBER on vessel axis, GREEN on gangway axis. Intact-prior σ_radial dropped from 1.31→0.95 m, P50 from 3.26→2.37 m, P90 from 4.14→3.00 m.

### Operability polar (this turn)
- **Goal:** "Level-3-light" capability plot. Sweep relative weather direction; per axis (vessel base position / gangway telescope) bisect on V_w until intact-prior P90 hits the IMCA M254 warn (amber) and alarm (red) thresholds. Sea state at each V_w from **Pierson-Moskowitz fully-developed** (DNV-RP-C205 §3.5.5.4 / DNV-ST-0111 App.F): `Hs = 0.21 · Vw² / g`, `ω_p = 0.877 · g / Vw`.
- **Module:** `cqa/sea_state_relations.py` (PM law) + `cqa/operability_polar.py` (driver + plot). 16 tests pass.
- **Plot:** 2-panel side-by-side polar (vessel | gangway), bow-up clockwise (DP-cap convention), green/amber/red traffic-light shading. `scripts/csov_operability_polar.png` shows the demo output.
- **CSOV demo (parametric drift, no RAO, mid-stroke port gangway):**
  - Vessel **(binding)**: worst-direction warn at **13.0 m/s** (θ=80°, beam-quartering); alarm at 17.6 m/s (θ=70°). Best (head/stern): alarm at 19.8 m/s.
  - Gangway: warn at 18.1 m/s; alarm at 19.4 m/s; best 24.0 m/s.
- **Performance gotcha:** with `rao_table=` enabled each intact-prior eval is **5 s** (sigma_L_wave is the bottleneck — 256-pt × spreading-quadrature × 6-DOF complex interp), making the polar ~100 min. Without RAO each eval is 37 ms → polar in ~50 s. **Demo currently runs without RAO.** Production needs vectorised / pre-tabulated `sigma_L_wave` over (Vw, θ).

### Earlier (preserved) discoveries
- Rice / CLH formula: `ν_a+ = ν_0+ exp(-a²/(2σ²))`, `P_breach = 1 - exp(-2 ν_a+ T)`.
- Vanmarcke (1975) clustering: `q = √(1 - λ_1²/(λ_0 λ_2))`. q<0.3 narrowband, 0.3-0.6 transition, q>0.6 wideband. CSOV intact-prior position channel is wideband (q=0.78).
- Inverse Rice formula (closed-form Poisson, bisection Vanmarcke): `a(p) = σ √(-2 ln(-ln(1-p)/(2 ν_0+ T)))`.
- Bias estimator stochastic-variance bug fixed earlier — A_aug matches brucon `dp_estimator` passive-observer form.
- pdstrip rotational columns per unit wave **slope**; `evaluate_rao` applies `k = ω²/g` for `dof > 3`.
- Cos-2s s=15 has **~21° one-sigma**, not 28°.

## Accomplished

### Done in the most recent sessions (this commit)
**P3.10 online Bayesian σ² estimator + integration** — fully wired:
- `cqa/online_estimator.py` (~340 lines): `BayesianSigmaEstimator`, `SigmaPosterior`, `closed_loop_decorrelation_time`. Conjugate IG with Bartlett ESS, ring-buffer O(1) update, soft-clipped α₀.
- `cqa/operator_view.py::summarise_intact_prior` accepts `posterior_sigma_radial_m` and `posterior_sigma_telescope_slow_m`. Spectral shape preserved; only level overridden. Wave band untouched.
- `tests/test_online_estimator.py` — 26 tests.
- `tests/test_operator_view.py` — 3 new override-integration tests.

**Operability polar (Level-3-light intact-prior capability sweep):**
- `cqa/sea_state_relations.py` — PM `pm_hs_from_vw`, `pm_tp_from_vw`, `pm_sea_state`, `WindSeaState`.
- `cqa/operability_polar.py` — `operability_polar(cfg, joint, ...)` per-direction bisection finder, `OperabilityPolar` dataclass with cap-flags, `plot_operability_polar` 2-panel polar.
- `tests/test_operability_polar.py` — 16 tests (PM closed-form, bisection root-finding & cap-flags, polar shape/metadata/monotonicity/range, plot smoke).
- `scripts/run_operability_polar_demo.py` — CSOV demo at mid-stroke, port gangway, 36 directions, 2-30 m/s.

**Earlier (committed in this same commit but described in prior recaps):**
- σ-inconsistency Tier 1 fix (controller params unified via `cfg.controller`).
- `ControllerParams` dataclass.

### Test totals
**171 tests pass** with `/home/blofro/src/hydro_tools/cqa/.venv/bin/python -m pytest tests/ -q` (~2 min).

### Long-term roadmap (still ahead)
- **σ²-validity residual alarm** — windowed-max vs Rice prediction; flag if disagreement >3× for >2 windows.
- **Side-by-side prior-σ vs posterior-σ panel** in `run_operator_summary_demo.py` (operator-facing question 4).
- **Optimise `sigma_L_wave`** for batch evaluation (pre-tabulate over Vw at each θ) and re-run polar with RAO.
- **WCFDI overlay** on the operability polar (dashed lines), needs MC at each (θ, Vw); expensive but high-value.
- **Update `analysis.md`** for P3.5/P3.7/P3.8/P3.9/P3.9b/Tier1-fix/P3.10/operability polar.
- **P4** validation matrix vs brucon `vessel_simulator` (time-domain runs vs frequency-domain predictions).
- **P5** DP-playback validation against logged sea trials.
- **P6** decision report / go-no-go gate.
- **P7** C++ demonstrator + `dp_gui` integration. Operator-selectable quantile to be added here.
- **(Defer) Tier 2 σ-inconsistency:** plumb `aug.A / aug.B_w` into `summarise_intact_prior` to use 12-state augmented system.

## Relevant files / directories

**cqa package (latest, all in this commit):**
- `cqa/config.py` — `CqaConfig`, `ControllerParams`, `csov_default_config()`.
- `cqa/vessel.py`, `cqa/controller.py`, `cqa/closed_loop.py` — DP plant + linearised closed loop.
- `cqa/psd.py` — wind/drift/current force PSDs.
- `cqa/extreme_value.py` — Rice + Vanmarcke + inverse Rice (single + multiband).
- `cqa/transient.py` — `wcfdi_transient` deterministic.
- `cqa/wcfdi_mc.py` — Monte Carlo full12 mode.
- `cqa/gangway.py` — kinematics, sensitivities, operability evaluation.
- `cqa/excursion.py` — original excursion polar (sigma-only).
- `cqa/operator_view.py` — `summarise_for_operator` (WCFDI panel) + `summarise_intact_prior` (length-scale P50/P90 panel) + plotters. **Now accepts `posterior_sigma_*` overrides.**
- `cqa/online_estimator.py` — Bayesian σ² estimator.
- `cqa/sea_state_relations.py` — PM wind-wave law.
- `cqa/operability_polar.py` — per-direction Vw boundary, polar plot.
- `cqa/rao.py`, `cqa/wave_response.py`, `cqa/drift.py`, `cqa/sea_spreading.py` — pdstrip RAO loading + 1st-order wave telescope contribution + drift PSDs.
- `cqa/__init__.py` — exports everything above.

**Tests (171 total):**
- `tests/test_extreme_value.py` — 29.
- `tests/test_closed_loop.py` — 8.
- `tests/test_operator_view.py` — 20 (16 base + 1 plot smoke + 3 posterior-override).
- `tests/test_online_estimator.py` — 26.
- `tests/test_operability_polar.py` — 16.
- (Other modules' tests cover gangway, vessel, wcfdi_mc, transient, psd, rao, wave_response, drift, sea_spreading.)

**Demo scripts:** (output PNGs are gitignored)
- `scripts/run_operator_summary_demo.py` — IMCA M254 Fig.8 panel + intact-prior length-scale panel.
- `scripts/run_operability_polar_demo.py` — operability polar (no RAO; ~50 s).
- `scripts/run_wcfdi_mc_demo.py`, `run_wcfdi_transient_demo.py`, `run_polar_demo.py`, `run_rao_smoke.py`.

**Source-of-truth doc (outdated, update pending):**
- `cqa/analysis.md` — last comprehensive update was around P3.5/P3.7. Needs a sweep through P3.8/P3.9/P3.9b/Tier1-fix/P3.10/operability polar.

**brucon files referenced (read-only):**
- `~/src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat` — RAO data.
- `~/src/brucon/libs/dp/vessel_model/wave_response.cpp` — drift PSD reference.

## Reproducibility

```bash
cd ~/src/hydro_tools/cqa
python3 -m venv .venv
.venv/bin/pip install -e .   # uses pyproject.toml
.venv/bin/python -m pytest tests/ -q   # ~2 min, 171 tests
.venv/bin/python scripts/run_operability_polar_demo.py   # ~50 s, generates polar PNG
.venv/bin/python scripts/run_operator_summary_demo.py    # ~30 s, generates two PNGs
```

## Next-step menu (pick when resuming)

1. **σ²-validity residual alarm** — close the loop on the online estimator: detect when the Rice prediction is being violated by the observed running max.
2. **Side-by-side prior-σ vs posterior-σ panel** in the operator-summary demo (question 4 of the operator framework).
3. **Optimise `sigma_L_wave`** for batch evaluation; re-run operability polar with RAO.
4. **WCFDI overlay** on the operability polar (dashed lines).
5. **Update `analysis.md`** comprehensively.
6. **P4 vessel_simulator validation matrix.**

Recommended immediate priority: (2) side-by-side panel — directly closes the operator-framework loop and reuses everything already built; or (5) `analysis.md` update — useful for stakeholders before pushing further code.
