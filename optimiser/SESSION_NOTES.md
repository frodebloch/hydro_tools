# Optimiser — Session Notes

## Voyage Comparison Tool (`voyage_comparison.py`)

### Overview

Hindcast-based voyage comparison tool that simulates repeated daily departures
(one per day, full year 2024) on the route **Rotterdam Europoort → Gothenburg
(509 nm)** using NORA3 wave/wind hindcast data. Compares fuel consumption
between a **factory combinator** (fixed pitch/RPM schedule for a CPP vessel)
and an **optimised pitch/RPM selection**, including a **Flettner rotor**
wind-assist model and **Blendermann wind resistance** on the hull.

### Route and Vessel

- **Route:** 7 legs, 509 nm, weighted mean heading ~63° (NE-bound outbound)
- **Engine:** MAN L27/38, 800 RPM max, 480 RPM min, 2920 kW MCR
- **Gear ratio:** 6.803 (800 RPM engine / 117.6 RPM shaft)
- **Propeller:** D=4.66m, P/D_design=0.771, BAR=0.432 (C-series Fourier model)
- **Flettner:** H=28m, D=4m, max 220 RPM, target spin ratio 3.0
- **Wind areas:** A_frontal=280 m², A_lateral=1100 m², Loa=100 m

### Key Architecture

- `evaluate_voyage()` — evaluates one voyage (outbound or return), returns `VoyageResult`
- `_voyage_worker()` — parallel worker; evaluates outbound, optionally return leg, combines via `_combine_round_trip()`
- `run_annual_comparison()` — orchestrates 366 voyages using `ProcessPoolExecutor` (16 workers)
- Operating point caches (optimiser + factory) pre-computed for T = 10-350 kN in 0.5 kN steps
- Thrust demand: `T = T_calm + (R_aw + R_wind - F_flettner) / (1 - t_deduction)`

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

**Business case note:** SG mode increases P/R saving (~32→63 t/yr) because the
factory combinator runs further below torque limit (SG allowance creates extra
headroom). Flettner saving decreases (~99→83 t/yr) due to RPM constraint limiting
optimiser freedom. A frequency converter SG would recover the lost Flettner benefit
(~15 t/yr, €10k/yr) but at ~€100k capex + 3-5% efficiency loss (~€5-7k/yr in
extra fuel), giving 18-33 year payback — not viable.

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
- `wind_resistance_kN()` — full Blendermann (1994) for general cargo vessel (Type 7)
- Tabulated CX(angle) at 10° intervals, 0-180° (19 points)
- Returns increment over still-air drag (avoids double-counting with calm-water data)
- Positive = resistance, negative = drive (following wind)
- Integrated into thrust demand alongside R_aw and F_flettner
- Tracked in `HourlyResult.R_wind_kN`, `VoyageResult.mean_R_wind_kN`

#### Phase 8: Fuel Cost Estimates
- `--fuel-price` CLI (default €650/t MGO, ECA zone)

#### Phase 9: Round-Trip Voyages
- **Problem:** One-way NE-bound route creates directional wind bias from prevailing
  SW winds — wind resistance, Flettner effectiveness, and overall savings are biased.
- **Solution:** Round-trip voyages (outbound + return on reversed waypoints).
  Return leg departs when outbound arrives, experiencing different weather hours.
- `ROUTE_GOTHENBURG_ROTTERDAM = list(reversed(ROUTE_ROTTERDAM_GOTHENBURG))`
- `_combine_round_trip()` — sums fuel, averages weather means (weighted by hours),
  sums feasibility counters, concatenates hourly lists
- `_voyage_worker()` evaluates both legs when `return_route` is provided
- `run_annual_comparison()` accepts `round_trip=True` (default)
- `--one-way` CLI flag for backwards compatibility
- Round-trip is now the default for all modes (standard, SG, speed sweep)

**Impact of round-trip (10 kn):**

| Metric              | One-way | Round-trip | Change |
|---------------------|---------|------------|--------|
| Transit time        | 50.9 h  | 101.9 h    | 2x     |
| Voyages/year        | 146     | 73         | 0.5x   |
| Fac NF/voyage       | 7,306 kg| 14,110 kg  | ~2x    |
| Annual fac NF       | 1,069 t | 1,032 t    | -3%    |
| Pitch/RPM saving    | 2.7%    | 2.9%       | +0.2pp |
| Flettner saving     | 8.9%    | 10.9%      | +2.0pp |
| Total saving        | 11.6%   | 13.9%      | +2.3pp |
| Mean R_wind         | 7.3 kN  | 4.5 kN     | -38%   |
| Mean F_flettner     | 11.8 kN | 14.3 kN    | +21%   |

The one-way bias slightly underestimated total savings. Mean wind resistance
dropped significantly as outbound/return average out the directional effect.

#### Phase 10: Plot Baseline Consistency Fix
- Upper panel of `voyage_comparison_results.png` had mismatched baselines:
  stacked areas used factory NF denominator, but Total line used factory+Fl.
  Fixed so all three series use factory NF baseline (`sav_total_pct = sav_pr_pct + sav_fl_pct`).
- Scatter plot (`voyage_comparison_scatter.png`) also switched to factory NF
  baseline with explicit title indicating the baseline.
- Y-axis labels now read "% of factory NF" for clarity.

### Current Results (Round-Trip, 10 kn)

| Metric | Value |
|--------|-------|
| Per-voyage factory NF | 14,110 kg |
| Per-voyage opt+Fl | 12,176 kg |
| Annual factory NF | 1,032 t/yr |
| Annual opt+Fl | 890 t/yr |
| **Pitch/RPM saving** | **2.9% (30 t/yr)** |
| **Flettner saving** | **10.9% (111 t/yr)** |
| **Total saving** | **13.9% (141 t/yr, €92k/yr)** |
| Mean R_aw | 4.7 kN |
| Mean R_wind | 4.5 kN |
| Mean F_flettner | 14.3 kN |
| Factory infeasible | 1% |

### Usage Examples
```bash
# Standard round-trip mode (default)
python3 voyage_comparison.py --year 2024 --speed 10 --quiet

# One-way mode (backwards compatibility)
python3 voyage_comparison.py --year 2024 --speed 10 --quiet --one-way

# Speed sweep
python3 voyage_comparison.py --speed-sweep 8 10 12 14 --quiet

# SG comparison
python3 voyage_comparison.py --compare-sg --sg-load 150 --sg-factory-allowance 200 \
    --sg-freq-min 48 --sg-freq-max 60 --quiet

# With plots
python3 voyage_comparison.py --speed 10 --quiet --plot
```

### Known Issues / Notes
- Dec 28-31 round-trip voyages fail (need Jan 2025 data for return leg)
- `FactoryCombinator._build_schedule()` takes ~25 seconds
- Cartopy broken (NumPy 2.x) — using pyshp + Natural Earth shapefiles
- NORA3 data: `data/nora3_route/nora3_wave_2024{01-12}.nc`

### Key Files
- `voyage_comparison.py` — main tool (~3300 lines)
- `optimiser.py` — `find_min_fuel_operating_point()` with aux power and RPM constraints
- `propeller_model.py` — C-series Fourier coefficient model
- `data/ne_10m/ne_10m_land.shp` — Natural Earth coastline
