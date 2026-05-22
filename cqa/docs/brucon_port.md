# Brucon C++ Port Handoff: Live Operator CQA Panel

**Audience:** brucon C++ developer porting the cqa **live operator pipeline** (not the planning/forecast pipeline) into brucon for online execution.

**Status of cqa side (as of commit `7a9aee5`):** all four traffic-light gates implemented and validated against brucon truth on a 14-cell roll-up matrix. 414/414 unit + integration tests green. Further cqa work will be driven by porting needs (validation/extension).

This document is a **concise overview + pointer index**. It does not duplicate derivations. For derivations, read `cqa/analysis.md` sections referenced below. For algorithms, read the cqa source at the file:line refs given.

---

## 1. Scope

**In scope for the port:** the *live* operator panel, computed online from observer state alone.

- Entry point: `cqa/cqa/live_operator_view.py:831` `summarise_for_operator_live`
- Output: `LiveOperatorSummary` (`cqa/cqa/live_operator_view.py:204`) — four traffic lights + headline numbers
- Inputs at runtime (already present in brucon — no weather model, no QTF lookup, no Tp_obs needed):
  - `eta_hat` (3-vector, body NED-equivalent), `nu_hat` (3-vector body), `b_hat` (3-vector body, **N and N·m** at the cqa boundary — see §3 bear-trap)
  - `eta_wave` (3-vector wave-frequency position from second-order wave filter)
  - `heading_compass` (deg)
  - `tau_buffer` (recent commanded thruster forces/moments, body frame, shape `(N, 3)` in N/N·m at boundary; for sigma estimation)
  - `tau_buffer_fs_hz` (sampling rate)

**Out of scope:**
- The forecast/planning pipeline (`cqa/cqa/decision_matrix.py`, `cqa/cqa/transient.py::wcfdi_transient`) — stays in Python for offline use.
- Catastrophic loss-of-station regime modelling (sec.12.21.21.30b). Documented as known limitation; current panel correctly red-flags via the `regime_b_traffic` `p_sat` gate.
- Gangway operability bar visualisation. The numbers are computed (`_sigma_dL_intact`, `_sigma_dL_wcf`); render however brucon prefers.

---

## 2. Pinned conventions (BEAR TRAPS — read before coding)

1. **Body frame** (Fossen 2011 §2.1): +surge = AHEAD, +sway = STARBOARD, +yaw = CW viewed from above. cqa uses this everywhere internally.
2. **Brucon `wave_direction`** = compass bearing waves come FROM (not "going to"). Convert at boundary if needed.
3. **One-sided rad/s-native PSD convention:** `σ² = ∫₀^∞ S(ω) dω` with NO `/π` factor. All `cqa/cqa/live_regime_b.py` PSD functions (`gauss_markov_psd:649`, `eta_psd_from_dtau:753`) follow this. Match exactly or moments will be wrong by π.
4. **K_b_pos units bug — historical (fixed cqa-side `de0bda7`):** brucon stores `bias_estimate_` in **acceleration units** (`m/s²`, `rad/s²`) in `nonlinear_passive_observer.cpp:178-184, 254-259`. cqa converts at boundary by `K_b_pos_force = M @ K_b_pos` inside `build_observer_augmented_system_full` (`cqa/cqa/transient_obs.py:257-273`, see in-place comments). The cqa input contract takes `b_hat` in **N/N·m** (already mass-multiplied). The C++ port can either keep brucon's native acceleration units throughout and skip the conversion, OR convert at the panel boundary; choose one and document.
5. **Force units at C++↔Python boundary:** brucon `.out` files use kN and kN·m. cqa internals are SI (N, N·m). The port runs entirely inside brucon → use SI consistently and there is no boundary, but the analysis.md numbers and tests are SI.
6. **Target vessel:** brucon `config_csov` (CSOV: Lpp=101.1 m, B=22.4 m, T=6.50 m). Polytope constants in §6.

---

## 3. Input contract

Read `cqa/cqa/live_decision.py:170-219` — `LiveObserverState` dataclass. Every field maps 1:1 to a brucon observer output. Nothing here requires sea-state knowledge.

Sigma posteriors (online noise-floor estimation): `cqa/cqa/online_estimator.py::BayesianSigmaEstimator`. Sea-state-agnostic; updates from `tau_buffer` residuals. Produces `LiveSigmaPosterior` (`cqa/cqa/live_decision.py:222-302`). The port should mirror the recursive Bayesian update; algorithm is small and self-contained.

---

## 4. Pipeline overview — four traffic-light gates

`summarise_for_operator_live` (`cqa/cqa/live_operator_view.py:831`) produces four independent gates, then takes the worst as `overall`:

| gate | meaning | implementation |
|---|---|---|
| `intact_traffic` | Position-keeping under nominal sea-state (no WCF) | radial p95 of intact closed-loop excursion; IMCA position bands |
| `wcf_traffic` | Post-WCF station-keeping | **quadrature** of two channels: non-saturated transient + saturated regime-B excursion (sec.21.30) |
| `regime_b_traffic` | Probability of thrust-allocation saturation given a WCF event | `p_sat` via clipped-Gaussian on the polytope; IMCA bands {green<0.01, amber<0.10, red≥0.10} |
| `gangway_traffic` (optional) | Gangway tip excursion bar | derived from intact + WCF sigmas projected at gangway attachment point |

All four use the same `LiveSigmaPosterior` and the same observer-augmented dynamics; differences are which input excitation (wave-frequency vs WCF tau-step) drives which closed-loop transfer.

---

## 5. Module-by-module port work items

### 5.1 Augmented closed-loop system (the core LTI model)

**cqa source:** `cqa/cqa/transient_obs.py`

- State indices: `:99-108` (`IDX_ETA=0:3 … N_STATE=27`). Note Option 2 uses the 21-state subset (drops the explicit `XI`/`ETA_W` integrators where appropriate; `eta_frequency_response` handles slicing internally).
- Observer gains: `:114-143` `ObserverGains`, `csov_observer_gains(Tp_s=10.0, zeta_w=0.1)` returns CSOV defaults `K_b_pos=[0.0012,0.0012,0.002]`, `K_a_pos=[0.12,0.12,0.20]`, `omega_c=[1.04,1.04,1.04]`, `T_b=[1000,1000,1000]`. These match `~/src/brucon/build/bin/config_csov/observer.prototxt`. Read from the proto, don't hard-code.
- **Canonical builder:** `:166-311` `build_observer_augmented_system_full`. Mirrors brucon `nonlinear_passive_observer.cpp` and `second_order_wave_filter.cpp:64-67`. **K_b_pos unit-conversion bear-trap documented at lines 257-273** — do not skip those comments.

**Port:** a single function `BuildAugmentedSystem(vessel, controller_gains, observer_gains) -> (A, B_w, B_tau, C)` returning the state-space matrices. All four gates re-use this.

### 5.2 Sigma posteriors (online noise floor)

**cqa source:** `cqa/cqa/online_estimator.py::BayesianSigmaEstimator` — recursive Bayesian update from `tau_buffer` residuals. Produces wave-frequency and low-frequency sigma posteriors (intact and WCF axes via `_sigmas_intact_axis:679` and `_sigmas_wcf_axis:691`).

**Port:** small, standalone; mirror the math. Sea-state-agnostic by design.

### 5.3 Intact axis

**cqa source:**
- `cqa/cqa/live_operator_view.py:548` `_radial_quantiles` — projects 3-DoF Gaussian onto radial excursion, returns p50/p95 with Gumbel peak-factor.
- `:514` `_gumbel_peak_factor` — peak-factor for finite-window maxima.
- `:679` `_sigmas_intact_axis` — picks the right sigma posterior.

**Port:** straight numerical translation. Acceptance: matches cqa to <1% on any test case from `tests/test_live_operator_view.py`.

### 5.4 WCF non-saturated channel

**cqa source:**
- `cqa/cqa/live_operator_view.py:576` `_radial_window_max_quantiles` — radial window-max for the transient.
- `:691` `_sigmas_wcf_axis` — WCF-axis sigma posteriors.
- Deterministic transient: `cqa/cqa/transient_obs.py::pulse_response` and `pulse_response_saturated` (`:376+`).
- Stored in `LiveOperatorSummary.wcf_R_p95_nonsat_m`.

**Port acceptance:** within ±26% of brucon truth on 11/14 roll-up cells (the pre-sec.21.30 baseline; see `analysis.md` sec.12.21.21.28b).

### 5.5 Regime-B saturation gate + Option 2 (post-WCF excursion under saturation)

**cqa source:** `cqa/cqa/live_regime_b.py` (entire file; ~920 lines but small functions).

- Polytope geometry: `:143-280`. Yaw-priority conditional cap (sec.21.24) — `sway_cap_given_yaw:175`, `yaw_cap_given_sway:220`, `operational_cap_at:251`.
- Saturation severity: `:284-540` — `RegimeBSeverity`, `lf_filter:317`, `saturation_probability_gaussian:351`, `estimate_regime_b_severity:379` (canonical entry, two cap modes; use yaw-priority).
- **Option 2** (saturated-channel excursion p95): `:543-922`
  - `clipped_gaussian_moments:543` — μ, σ of `max(τ, cap)`.
  - `gauss_markov_psd:649` — AR(1) PSD of the clipped LF tau process.
  - `eta_frequency_response:693` — closed-loop |H(jω)|² from `dtau → eta` using the augmented system.
  - `eta_psd_from_dtau:753` — convolve PSD through |H|².
  - `estimate_post_wcf_excursion_distribution:821` — Rice/Vanmarcke peak distribution → p95 excursion.

Extreme-value primitives: `cqa/cqa/extreme_value.py` (`zero_upcrossing_rate`, `vanmarcke_bandwidth_q`, `inverse_rice`, `inverse_rice_multiband`).

**Port acceptance:** matches synthetic ramp from `scripts/p7_brucon_validation/synthetic_amber_demo_option2.py` (sec.21.29b): monotone 0.4 cm @ +4σ headroom → 2.69 m @ −1σ. Brucon real-data point: bf8.5_q10_w45 should produce Option-2 p95 ≈ 0.85 m (sec.21.30b).

### 5.6 Sec.21.30 quadrature (the headline WCF number)

**cqa source:** `cqa/cqa/live_operator_view.py:1227-1257`.

```
wcf_R_p95_nonsat_m = <non-sat channel result>
wcf_R_p95          = hypot(wcf_R_p95_nonsat_m, post_wcf_excur_R_xy_p95)
wcf_traffic        = _imca_traffic(wcf_R_p95, pos_warning, pos_alarm)
overall            = _worst([intact, wcf, regime_b, gangway])
```

Combination is approximately conservative-but-tight: channels are near-independent (different time scales). Rationale and pre/post-WCF moment invariance in `analysis.md` sec.12.21.21.20-21, 21.30.

### 5.7 Gangway bar (optional)

**cqa source:** `cqa/cqa/live_operator_view.py:757` `_sigma_dL_intact`, `:786` `_sigma_dL_wcf`, rendered by `_draw_abs_dL_bar:1410`.

Pure post-processing of the upstream sigmas + attachment-point arm. Port last or defer.

---

## 6. CSOV polytope geometry (validation reference values)

From `analysis.md` sec.12.21.21.16-24 and the 14-cell roll-up:

- Intact cap (sway-x, sway-y, yaw): `(1.36e6 N, 1.695e6 N, 8.64e7 N·m)`
- Residual cap (post-WCF, decoupled): `(8.38e5, 1.104e6, 4.79e7)`
- α per DOF: `(0.616, 0.651, 0.555)`
- Yaw-priority polytope vertices: bow ±629 kN @ arm +36.29 m, stern ±522 kN @ arm −48.09 m
- IMCA position bands: `pos_warning = 2.0 m`, `pos_alarm = 4.0 m`
- IMCA p_sat bands: green `<0.01`, amber `[0.01, 0.10)`, red `≥0.10`
- Option 2 defaults: `tau_decorr_lf_s=15.0`, `t_horizon_s=200.0`, `omega_max_rad_s=3.0`, `n_omega=512`, `clustering="vanmarcke"`

For a different vessel: regenerate from `cqa/cqa/config.py:341+ csov_default_config()` and the polytope-construction helpers; do not hand-edit.

---

## 7. Validation: acceptance criteria for the port

**Primary scoreboard:** the 14-cell brucon validation roll-up.

```bash
cd /home/blofro/src/hydro_tools/cqa
PYTHONPATH=. .venv/bin/python scripts/p7_brucon_validation/roll_up_live_operator_panel.py
```

Cells: 12 baseline (BF6/BF7/BF8 × {q10, q90} × {w0, w45}) + 2 headroom-gradient (BF7.5_q10_w45, BF8.5_q10_w45). See `scripts/p7_brucon_validation/run_validation_matrix.py:119-135` for the cell dict.

**Recommended bar:** C++ panel must reproduce cqa `wcf_R_p95` to within **5% relative or 5 cm absolute (whichever larger)** on all 14 cells, and produce identical traffic-light colours on all 14.

**Secondary:**
- All four `tests/test_live_*.py` files (414 tests) should have C++ equivalents for the numerical primitives.
- `synthetic_amber_demo_option2.py` ramp reproduces monotonically.
- bf8.5_q10_w45 panel goes RED via both `wcf_R_p95 > 4 m` AND `p_sat > 0.10` independently (defence-in-depth, sec.21.30b finding).

---

## 8. Known limitations (deliberately accepted for v1)

- **Catastrophic loss-of-station tail** (bf8.5_q10_w45 truth p95 ≈ 30 m vs Option-2 ≈ 0.85 m): Option 2 assumes stationary-Gaussian saturated tau; this breaks down once the controller diverges. The panel still correctly red-flags via the `regime_b_traffic` `p_sat=0.257` gate. Modelling the catastrophic regime explicitly is **out of scope** for v1. See `analysis.md` sec.12.21.21.30b for full discussion.
- **No projection of pre-WCF (μ, σ) onto residual polytope:** measured invariant on bf8_q10_w45 (sec.21.20-21); reallocation is transient and already captured in the non-sat channel. Don't add a projection step.
- **`Tp_obs_s` is not an input to the live panel.** The forecast pipeline needs it; the live pipeline does not.

---

## 9. Pointers into `cqa/analysis.md`

Read in this order for the live-panel design rationale:

- sec.12.21.21.16–22 — Regime-B v1 design
- sec.12.21.21.20–21 — pre/post-WCF moment invariance + 6.24σ headroom caveat
- sec.12.21.21.24 — yaw-priority conditional cap
- sec.12.21.21.28b — non-sat channel −38% bf8 acceptance
- sec.12.21.21.29 — Option 2 build + brucon zero-result
- sec.12.21.21.29b — synthetic amber demo
- sec.12.21.21.30 — quadrature wiring
- sec.12.21.21.30b — BF7.5/BF8.5 headroom-gradient sweep and catastrophic-regime out-of-scope decision
