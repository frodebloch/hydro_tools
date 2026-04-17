# Optimiser — Session Notes

## Voyage Comparison Tool (`voyage_comparison.py`)

### Overview

Hindcast-based voyage comparison tool that simulates repeated daily departures
(one per day, full year 2024) on the route **Rotterdam Europoort -> Gothenburg
(509 nm)** using NORA3 wave/wind hindcast data. Compares fuel consumption
between a **factory combinator** (fixed pitch/RPM schedule for a CPP vessel)
and an **optimised pitch/RPM selection**, including a **Flettner rotor**
wind-assist model and **Blendermann wind resistance** on the hull.

### Route and Vessel

- **Route:** 7 legs, 509 nm, weighted mean heading ~63 deg (NE-bound outbound)
- **Engine:** MAN L27/38, 800 RPM max, 480 RPM min, 2920 kW MCR, 8-cylinder
- **Gear ratio:** 6.803 (800 RPM engine / 117.6 RPM shaft)
- **Propeller:** D=4.80m, P/D_design=0.771, BAR=0.432 (C4-series Fourier model)
- **Flettner:** H=28m, D=4m, max 220 RPM, target spin ratio 3.0, k_rough=3.0
- **Wind areas:** AV=320 m2 frontal (from GA drawing), AL=1100 m2 lateral, Loa=100 m
- **Hull data:** Model test 25-0461/25-0288, draught 7.600 m, displacement 11165 m3
- **Hull at 10 kn:** R_calm=110 kN, w=0.227, t=0.171, eta_R=1.010, T_calm=132.7 kN
- **Fuel price:** MGO @ EUR 650/tonne

### Key Architecture

- `evaluate_voyage()` -- evaluates one voyage (outbound or return), returns `VoyageResult`
- `_voyage_worker()` -- parallel worker; evaluates outbound, optionally return leg, combines via `_combine_round_trip()`
- `run_annual_comparison()` -- orchestrates 366 voyages using `ProcessPoolExecutor` (16 workers)
- Operating point caches (optimiser + factory) pre-computed for T = 10-350 kN in 0.5 kN steps
- Thrust demand: `T = (R_calm + R_aw + R_wind + R_roughness - F_flettner) / (1 - t)`

### Implementation History

#### Phase 1: Map & Route
- Natural Earth 10m land shapefiles for coastline (via `pyshp`, Cartopy broken on NumPy 2.x)
- Auto-computing map extent from waypoints
- Skagerrak entrance waypoint moved to (58.00, 9.50) to clear Denmark

#### Phase 2: Initial Annual Run
- Full 2024 at 10 kn, 364 voyages (Dec 30-31 need Jan 2025 data)

#### Phase 3: Shaft Generator (SG) Mode
- Factory combinator accepts `sg_allowance_kw`, `engine_rpm_min/max`
- Schedule built so total engine load (propulsion + SG) sits on propeller curve
- SG RPM constraint: `engine_rpm_min = max_rpm * freq_min / freq_max`
- Lever 0 = zero thrust point (T=0, pitch=0, RPM=min) to avoid infeasibility
- Optimiser accepts `auxiliary_power_kw`, `engine_rpm_min/max`
- CLI: `--sg-load`, `--sg-factory-allowance`, `--sg-freq-min/max`, `--idle-pct`, `--compare-sg`

#### Phase 4: Normalized Reporting
- `print_summary()` with `idle_pct` and `fuel_price_eur_per_t`
- Transit time in hours and days
- Per-voyage stats table (mean/median/P10/P90/min/max)
- Annualized fuel and cost sections
- Seasonal breakdown by quarters

#### Phase 5: Comparison Plots (Standard vs SG)
- 5 publication-quality figures (`comparison_*.png`)
- Colour scheme: Standard = lighter (#3574a3, #e8774a), SG = darker (#1a4a70, #b84520)

#### Phase 6: Speed Sensitivity Sweep
- `--speed-sweep 8 10 12 14` CLI flag
- `SpeedSweepResult` dataclass, `run_speed_sweep()`, compact summary tables
- 2 figures: `speed_sweep_fuel.png`, `speed_sweep_savings.png`

#### Phase 7: Blendermann Wind Resistance Model
- `wind_resistance_kN()` -- full Blendermann (1994) for general cargo vessel (Type 7)
- Tabulated CX(angle) at 10 deg intervals, 0-180 deg (19 points)
- Returns increment over still-air drag (avoids double-counting with calm-water data)
- Positive = resistance, negative = drive (following wind)
- Integrated into thrust demand alongside R_aw and F_flettner
- Tracked in `HourlyResult.R_wind_kN`, `VoyageResult.mean_R_wind_kN`

#### Phase 8: Fuel Cost Estimates
- `--fuel-price` CLI (default EUR 650/t MGO, ECA zone)

#### Phase 9: Round-Trip Voyages
- Round-trip is now the default for all modes (standard, SG, speed sweep)
- Return leg departs when outbound arrives, experiencing different weather hours
- `_combine_round_trip()` sums fuel, averages weather means, sums feasibility counters
- `--one-way` CLI flag for backwards compatibility

#### Phase 10: Plot Baseline Consistency Fix
- All saving series use factory NF baseline (`sav_total_pct = sav_pr_pct + sav_fl_pct`)
- Y-axis labels read "% of standard baseline" for clarity

#### Phase 11: Hull Data Fix
- Replaced wrong thrust formula and wrong wake/thrust-deduction values
- Now uses model test data (test 25-0461/25-0288, draught 7.600 m, disp 11165 m3)
- `HULL_RESISTANCE_KN` array with `T = (R_calm + R_added)/(1-t)`
- Cross-validated against RT BruCon simulator at Point C (Hs=2.77m, T=152.7kN):
  our optimiser gives 171.8 kg/h vs RT's ~176 kg/h (-2.4%, consistent with expected
  dynamic overhead from Jensen's inequality + governor transients)

#### Phase 12: SG Evaluation Bug Fix
- Split `sg_allowance_kw` (200 kW, schedule construction) from `sg_load_kw` (150 kW, fuel evaluation)
- SG effective SFOC: 225 g/kWh (includes opportunity cost of RPM floor constraint)

#### Phase 13: Wind Area & Stopped-Rotor Drag Fix
- **`WIND_AREA_FRONTAL_M2`**: 280 -> 320 m2 (from GA drawing)
- **`FlettnerRotor.stopped_surge_force_kN()`**: new method computing surge force of
  non-spinning cylinder (SR=0, CD=0.6)
- **Smart spinning decision**: only spin rotor if spinning improves surge vs stopped.
  Otherwise use stopped-rotor force and zero rotor power.
- **"No Flettner" baseline now includes stopped-rotor drag**: the rotor (H=28m, D=4m,
  A_proj=112 m2) is permanently mounted. When not spinning it acts as a bluff cylinder
  with CD~0.6, adding ~4 kN drag at 10 m/s wind. Previously this was ignored.
- Removed `max(0, ...)` clamp on F_flettner_kN that silently discarded rotor drag

#### Phase 14: Relative Rotative Efficiency (eta_R)
- **`HULL_ETA_R`**: speed-dependent array (15 values) from model test ETAR column.
  At 10 kn: eta_R = 1.010. Range: 1.000 (8 kn) to 1.019 (14 kn).
- **Physics**: `Q_behind = Q_open / eta_R`, so `P_shaft_behind = P_shaft_open / eta_R`.
  When eta_R > 1, the propeller needs less shaft power in behind condition.
- **Applied in**: `find_min_fuel_operating_point()`, `find_optimal_operating_point()`,
  `FactoryCombinator._build_schedule()` (per-speed interpolation),
  `FactoryCombinator.evaluate()`, cache build, engine fallback paths.
- **Impact**: ~1% reduction in absolute fuel (1286.4 -> 1270.8 t/yr for factory NF
  at 10 kn). Saving percentages barely change (13.1% -> 13.0%).
- **Cross-validation**: With correct eta_R=1.01 on both sides, at 174 kN:
  Our model Factory ~204.2 kg/h, Optimiser ~197.1 kg/h.
  RT corrected (eta_R=1.01): Factory ~205.0, Optimiser ~199.7.
  Agreement within 0.4% (factory) and 1.3% (optimiser).
- **Note**: RT simulator was using eta_R=1.07 (ETAH column, hull efficiency) instead of
  ETAR (1.01). This was the wrong column from the model test report.

---

## Current Results (Round-Trip, 10 kn, Standard Mode)

| Configuration                     | Fuel [t/yr] |
|-----------------------------------|-------------|
| Standard control, no rotor        |      1270.8 |
| Standard control + Flettner       |      1167.4 |
| Optimiser, no rotor               |      1208.5 |
| Optimiser + Flettner              |      1105.6 |

| Saving component          | Tonnes/year | Percentage |
|---------------------------|-------------|------------|
| Propeller optimisation    |        62.3 |       4.9% |
| Wind-assist (Flettner)    |       102.9 |       8.1% |
| **Combined**              |   **165.2** |  **13.0%** |

Annual cost saving: EUR 107k/yr (13.0%) at MGO EUR 650/t.

### Per-Voyage Statistics

| Metric                         |  Mean | Median |   P10 |   P90 |   Min |   Max |
|--------------------------------|-------|--------|-------|-------|-------|-------|
| Mean Hs [m]                    |   1.4 |    1.3 |   0.7 |   2.3 |   0.4 |   4.0 |
| Mean wind speed [m/s]          |   7.8 |    7.6 |   5.3 |  10.6 |   3.5 |  14.5 |
| Mean R_aw [kN]                 |   4.7 |    2.1 |   0.2 |  12.4 |  -0.9 |  45.2 |
| Mean R_wind [kN]               |   5.2 |    4.5 |  -0.7 |  12.0 |  -5.1 |  30.7 |
| Mean F_flettner [kN]           |  12.3 |    9.6 |   0.5 |  29.4 |  -5.3 |  54.0 |
| Mean rotor power [kW]          |  19.0 |   15.5 |   4.7 |  40.5 |   0.7 |  69.1 |
| Pitch/RPM saving [%]           |   4.9 |    5.1 |   3.9 |   5.7 |   2.6 |   6.4 |
| Flettner saving [%]            |   8.3 |    7.1 |   2.0 |  16.6 |   0.0 |  28.6 |

### Seasonal Breakdown (Standard)

| Season        | Voyages | Std [kg/voy] | Opt+Fl [kg/voy] | P/R   | Flettner |
|---------------|---------|--------------|-----------------|-------|----------|
| Q1 (Jan-Mar)  |      91 |       17,420 |          14,835 |  4.7% |    10.4% |
| Q2 (Apr-Jun)  |      91 |       17,519 |          15,607 |  5.1% |     6.0% |
| Q3 (Jul-Sep)  |      92 |       17,348 |          15,377 |  5.1% |     6.3% |
| Q4 (Oct-Dec)  |      88 |       17,220 |          14,642 |  4.9% |    10.4% |

P/R saving stable at 4.7-5.1% year-round.

## Current Results (Round-Trip, 10 kn, SG Mode)

SG load: 150 kW actual, 200 kW factory allowance, freq 48-60 Hz,
RPM floor 640 eng / 94.1 shaft.

| Configuration                     | Fuel [t/yr] |
|-----------------------------------|-------------|
| Standard control, no rotor        |      1535.7 |
| Standard control + Flettner       |      1452.3 |
| Optimiser, no rotor               |      1535.4 |
| Optimiser + Flettner              |      1453.3 |

| Saving component          | Tonnes/year | Percentage |
|---------------------------|-------------|------------|
| Propeller optimisation    |         0.3 |       0.0% |
| Wind-assist (Flettner)    |        82.1 |       5.3% |
| **Combined**              |    **82.4** |   **5.4%** |

Annual cost saving: EUR 54k/yr (5.4%).

Note: P/R saving is ~0% because the SG RPM floor (94.1 shaft) pins both
factory and optimiser at the same operating point for calm-weather thrust
demands (~133 kN at 10 kn). The constrained RPM band eliminates the
optimiser's ability to run "high pitch, low RPM". The main benefit in SG
mode is Flettner wind-assist only.

### Seasonal Breakdown (SG)

| Season        | Voyages | Std [kg/voy] | Opt+Fl [kg/voy] | P/R   | Flettner |
|---------------|---------|--------------|-----------------|-------|----------|
| Q1 (Jan-Mar)  |      91 |       20,991 |          19,554 |  0.1% |     6.9% |
| Q2 (Apr-Jun)  |      91 |       21,148 |          20,330 | -0.0% |     3.9% |
| Q3 (Jul-Sep)  |      92 |       20,990 |          20,135 | -0.1% |     4.1% |
| Q4 (Oct-Dec)  |      88 |       20,870 |          19,460 |  0.0% |     6.8% |

---

## RT Simulator Comparison Data

### Factory Combinator Schedule (Standard, no SG)

14 points. This is the lever -> pitch/RPM lookup table for the RT.

| Lever% |  T [kN] |   P/D | Shaft RPM | Engine RPM |
|--------|---------|-------|-----------|------------|
|    0.0 |     0.0 | 0.000 |      70.6 |        480 |
|    7.7 |    97.5 | 0.679 |      75.0 |        510 |
|   15.4 |   109.9 | 0.698 |      78.0 |        530 |
|   23.1 |   123.8 | 0.711 |      81.4 |        553 |
|   30.8 |   137.5 | 0.720 |      84.9 |        577 |
|   38.5 |   152.6 | 0.734 |      88.1 |        600 |
|   46.2 |   167.7 | 0.742 |      91.6 |        623 |
|   53.8 |   184.1 | 0.752 |      95.0 |        647 |
|   61.5 |   202.0 | 0.763 |      98.4 |        669 |
|   69.2 |   221.2 | 0.773 |     101.8 |        693 |
|   76.9 |   242.8 | 0.782 |     105.6 |        718 |
|   84.6 |   267.6 | 0.788 |     109.8 |        747 |
|   92.3 |   288.5 | 0.800 |     112.7 |        766 |
|  100.0 |   309.6 | 0.800 |     116.6 |        793 |

### Factory vs Optimiser Operating Points

At 10 kn: Va=3.977 m/s (w=0.227). The optimiser selects higher pitch, lower RPM,
better eta0 -- especially at low-to-mid thrust demands.

| T [kN] | Fac P/D | Fac sRPM | Fac eRPM | Fac g/h  | Fac eta0 | Opt P/D | Opt sRPM | Opt eRPM | Opt g/h  | Opt eta0 | Save% |
|--------|---------|----------|----------|----------|----------|---------|----------|----------|----------|----------|-------|
|     80 |   0.703 |     79.4 |      540 |   98,905 |    0.620 |   0.810 |     71.0 |      483 |   90,712 |    0.683 |  8.3% |
|     90 |   0.708 |     80.7 |      549 |  109,051 |    0.629 |   0.840 |     70.9 |      482 |   99,502 |    0.698 |  8.8% |
|    100 |   0.713 |     81.9 |      557 |  119,405 |    0.635 |   0.870 |     70.8 |      482 |  109,277 |    0.703 |  8.5% |
|    110 |   0.716 |     83.2 |      566 |  130,091 |    0.638 |   0.900 |     70.8 |      481 |  119,941 |    0.702 |  7.8% |
|    120 |   0.719 |     84.5 |      575 |  141,150 |    0.639 |   0.900 |     72.3 |      492 |  131,262 |    0.697 |  7.0% |
|    130 |   0.724 |     85.7 |      583 |  151,955 |    0.640 |   0.880 |     74.8 |      509 |  142,818 |    0.689 |  6.0% |
|  **133** | **0.725** | **86.0** | **585** | **155,517** | **0.639** | **0.880** | **75.2** | **512** | **146,279** | **0.688** | **5.9%** |
|    140 |   0.728 |     86.8 |      590 |  163,419 |    0.639 |   0.870 |     76.8 |      522 |  154,607 |    0.683 |  5.4% |
|    150 |   0.733 |     87.9 |      598 |  174,875 |    0.638 |   0.850 |     79.3 |      540 |  166,894 |    0.675 |  4.6% |
|    160 |   0.736 |     89.0 |      606 |  186,938 |    0.636 |   0.840 |     81.3 |      553 |  179,247 |    0.668 |  4.1% |
|    170 |   0.739 |     90.2 |      614 |  199,106 |    0.633 |   0.830 |     83.3 |      566 |  191,818 |    0.660 |  3.7% |
|    180 |   0.742 |     91.4 |      622 |  211,460 |    0.630 |   0.820 |     85.3 |      580 |  204,622 |    0.653 |  3.2% |
|    200 |   0.748 |     93.5 |      636 |  236,463 |    0.623 |   0.810 |     88.6 |      602 |  230,374 |    0.640 |  2.6% |
|    220 |   0.754 |     95.6 |      650 |  262,764 |    0.616 |   0.790 |     92.6 |      630 |  258,693 |    0.625 |  1.5% |
|    250 |   0.763 |     98.5 |      670 |  304,519 |    0.604 |   0.770 |     97.9 |      666 |  303,646 |    0.606 |  0.3% |

T=133 kN is the calm-water operating point at 10 kn: T_calm = R_calm/(1-t) = 110/0.829 = 132.7 kN.

### RT Simulator Parameters to Update

- **Wind area frontal (AV):** 320 m2 (was 120 in RT, should be 320 from GA)
- **Wind area lateral (AL):** 1100 m2
- **Propeller:** D=4.80 m, P/D_design=0.771, BAR=0.432
- **At 10 kn:** w=0.227, t=0.171, R_calm=110 kN, T_calm=132.7 kN
- **Flettner (candidates 2 and 4):** H=28m, D=4m, max 220 RPM, SR_target=3, k_rough=3.0, CD_stopped=0.6
- **Stopped rotor drag:** CD=0.6, A_proj=28*4=112 m2 (included in "no Flettner" baseline)

### The 4 Comparison Candidates

1. **Standard control, no rotor** -- Factory combinator schedule (above), Flettner disabled but stopped-rotor drag included. Annual: 1270.8 t/yr.
2. **Standard control + Flettner** -- Same combinator, Flettner spinning. Annual: 1167.4 t/yr.
3. **Optimiser, no rotor** -- Optimiser P/D-RPM schedule (above), Flettner disabled but stopped-rotor drag included. Annual: 1208.5 t/yr.
4. **Optimiser + Flettner** -- Optimiser schedule, Flettner spinning. Annual: 1105.6 t/yr.

#### Phase 15: Fixed-Pitch Propeller (FPP) Baseline Comparison
- **`FixedPitchCombinator`** class in `models/combinator.py`: models a conventional FPP
  designed for maximum achievable speed with 15% sea margin at propeller curve power.
- **Design procedure**: scans hull speeds high-to-low, at each speed bisects on P/D
  to find the pitch where the propeller at max shaft RPM delivers the margined thrust,
  checks if engine power fits within propeller curve.
- **Design point**: P/D = 0.790, design speed = 14.0 kn, T = 309.6 kN, P_shaft = 2565 kW.
- **At off-design speeds**: RPM is the only free variable (fixed pitch). Bisects on RPM.
- **CLI**: `--fpp-baseline` flag swaps the factory CPP combinator for the FPP baseline.
- **Results (FPP baseline at 10 kn)**:

| Configuration                     | Fuel [t/yr] |
|-----------------------------------|-------------|
| FPP no rotor                      |      1122.3 |
| CPP Opt no rotor                  |      1099.1 |
| CPP Opt + Flettner                |      1030.3 |

| Saving (vs FPP baseline)  | Tonnes/year | Percentage |
|---------------------------|-------------|------------|
| CPP pitch/RPM flexibility |        23.2 |       2.1% |
| Flettner wind-assist      |        68.7 |       6.1% |
| **Combined**              |    **91.9** |   **8.2%** |

- **Note**: FPP baseline fuel (1122.3 t/yr) is lower than CPP factory (1270.8) because
  the FPP at P/D=0.790 is closer to the efficiency optimum than the factory combinator's
  propeller-curve-following schedule. However, the FPP has 9% infeasible hours (can't
  handle high thrust demands in rough weather), which are excluded from the comparison.
  The CPP factory has only 1% infeasible hours due to its variable pitch capability.

---

## Open Items

1. **RT wind area:** RT has AV=120 m2 (too low). Needs updating to 320 m2.
2. **RT thrust discrepancy:** "Simulator Status" shows 153-154 kN but mean `force_thrust_kN` is ~124 kN. At t=0.171, 153.5*0.829=127.3, not 124. To investigate.
3. **Dynamic overhead:** RT shows ~2-4% fuel penalty from shaft dynamics/governor even with `sea_state_margin=0`. Expected: Jensen's inequality on SFOC convexity + governor transients.
4. **SG speed sweep:** Only 10 kn re-run under SG. Could re-run 12/14 kn: `python3 voyage_comparison.py --speed-sweep 10 12 14 --sg-load 150 --sg-factory-allowance 200 --quiet`
5. **Lateral wind area:** AL=1100 m2 not reviewed for Flettner lateral projection (28*4=112 m2). Only affects sway, low priority.
6. **RT eta_R fix:** RT simulator uses eta_R=1.07 (ETAH column, hull efficiency). Should use ETAR=1.010 at 10 kn.
7. **Standalone tools need eta_R:** `compare_combinators.py`, `combinator_generator.py`, `diagnose_diameter.py`, `demo.py` all use bare `Q * 2*pi*n` without `/ eta_R`. Low priority (not used for reporting).
8. **FPP infeasibility:** 9% of hours infeasible for FPP (can't deliver thrust in rough weather). Could investigate adaptive speed reduction or alternative FPP design points.
9. **FPP report section:** FPP baseline results not yet written into the `.org` reports.

## Usage Examples

```bash
# Standard round-trip mode (default)
python3 voyage_comparison.py --year 2024 --speed 10 --quiet

# With plots
python3 voyage_comparison.py --speed 10 --quiet --plot

# SG comparison
python3 voyage_comparison.py --compare-sg --sg-load 150 --sg-factory-allowance 200 \
    --sg-freq-min 48 --sg-freq-max 60 --quiet

# Speed sweep
python3 voyage_comparison.py --speed-sweep 8 10 12 14 --quiet

# FPP baseline comparison (fixed-pitch propeller vs CPP optimiser)
python3 voyage_comparison.py --year 2024 --fpp-baseline --quiet --plot

# One-way mode (backwards compatibility)
python3 voyage_comparison.py --year 2024 --speed 10 --quiet --one-way
```

## Key Files

- `voyage_comparison.py` -- main CLI tool
- `optimiser.py` -- `find_min_fuel_operating_point()`, engine models
- `propeller_model.py` -- C-series Fourier coefficient model
- `models/combinator.py` -- `FactoryCombinator` schedule builder, `FixedPitchCombinator` FPP baseline
- `models/constants.py` -- hull data, propeller params, wind areas
- `models/flettner.py` -- Flettner rotor model (incl. stopped-rotor drag)
- `simulation/engine.py` -- hourly evaluation loop
- `simulation/cache.py` -- pre-computed operating point caches
- `simulation/orchestrator.py` -- annual comparison orchestrator
- `plotting/comparison.py` -- all plot generation
- `data/ne_10m/ne_10m_land.shp` -- Natural Earth coastline

## Known Issues / Notes

- Dec 28-31 round-trip voyages fail (need Jan 2025 NORA3 data for return leg)
- `FactoryCombinator._build_schedule()` takes ~25 seconds
- Cartopy broken (NumPy 2.x) -- using pyshp + Natural Earth shapefiles
- NORA3 data: `data/nora3_route/nora3_wave_2024{01-12}.nc`
