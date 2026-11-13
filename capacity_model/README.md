# Control-system capacity model

Monte Carlo model of the capacity for control-system projects sold (a) bundled with our own
mechanical units (1–3 years lead time) and (b) as retrofit (3–12 months, lead-time sensitive). It models
engineering, production and test/FAT capacity, hardware and sensor lead times, class approval and
learning, and compares levers by ΔNPV. A calculator-level **hand check** is run automatically against
the model.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib
.venv/bin/python run_scenarios.py 200                     # ~5 min, includes the hand check
.venv/bin/python hand_check.py                            # hand check only (uses the latest model results)
.venv/bin/python target_advanced.py 40                    # 300 MNOK DP2 case (~3 min each)
.venv/bin/python target_advanced.py 40 --no-learning
.venv/bin/python target_advanced.py 40 --dp2-weight=6
.venv/bin/python sensitivity.py 40                        # rework / learning sensitivities (~3 min)
```

Output in `results/`: `summary.md` (model table + learning sensitivity), `hand_check.md`,
`target_advanced*.md/png`, `scenarios.png`, `demand_sweep.png`, `calibration.json`, `model_means.json`.

| File | Content |
|---|---|
| `config.py` | All inputs (USER = given, PLACEHOLDER = guess): sub types, classes, hours calibration, prices, learning, scenarios |
| `sim.py` | Weekly simulation, retrofit quoting, approval/FAT rebooking, learning, working capital, NPV, calibration |
| `run_scenarios.py` | Calibration, Monte Carlo over scenarios (common random numbers), tables, plots, hand check |
| `hand_check.py` | Calculator-level formulas with the same inputs, compared with the model |
| `target_advanced.py` | What it takes to reach 30 retrofit DP2 projects (300 MNOK) per year |
| `sensitivity.py` | One-at-a-time sensitivities on rework share, rework half-life and learning rate |

## Inputs

### Product sub types (USER, 2026 projects = to date + planned)

| Sub type | 2026 | Class |
|---|---:|---|
| PTC – Thruster | 36 | simple |
| CMS | 26 | simple (assumed) |
| PTC – Other | 6 | simple (assumed) |
| PTC – Shaft propulsion | 18 | medium |
| BruCon-5 DP0 | 7 | medium |
| PTC – Azimuth propulsion | 5 | medium |
| BruCon-5 JS | 3 | medium |
| BruCon-5 AC | 2 | medium |
| BruCon-5 DP2 | 3 | advanced |
| BruCon-4 | 1 | excluded (outgoing); its share of the reported hours is removed |
| **Total modelled** | **106** | 68 simple, 35 medium, 3 advanced |

These give **5 project types (family × class)**: PTC simple (42), CMS (26), PTC medium (23),
BruCon-5 medium (12) and BruCon-5 DP2 (3).

### Per project class

| Class | Price | Traded sensors | Weight (hours vs simple) | Eng / project | Prod / project | FAT / project | Hardware |
|---|---:|---:|---:|---:|---:|---:|---|
| Simple | 0.3 MNOK | – | 1 | 154 h (4.1 wk) | 119 h | 46 h | stock |
| Medium | 1 MNOK | – | 2 (PLACEHOLDER) | 308 h | 239 h | 92 h | stock |
| Advanced (DP2) | 10 MNOK | 75 % of price | 3.5 (PLACEHOLDER, test 6) | 539 h | 418 h | 161 h | sensors 9–13 wk |

Hours per project = class weight × hours per simple-equivalent. The hours per simple-equivalent are
calibrated so that the 2026 volume (148.5 simple-equivalents) reproduces the hours **booked on the
projects** in 2026 (USER), minus the BruCon-4 share:

| Hours 2026 | Hours | In the model |
|---|---:|---|
| Engineering | 23,400 | yes (breakdown being checked) |
| Production control systems | 17,599 | production |
| Assembly bridge systems (integration part of packages) | 545 | production |
| Production Service / Retrofit | 2,111 | **left out for now** (unclear if it belongs) |
| Test and FAT control systems | 6,972 | FAT |

### Other inputs

| Input | Value | Source |
|---|---|---|
| Bundled / retrofit split | 60 % / 40 % for all classes | **PLACEHOLDER (highest priority)** |
| Capacity | 24 engineers × 80 % × 45 wk; production load 85 %; FAT load 92 % (bottleneck); capacity = hours / load | USER / **PLACEHOLDER loads** |
| Cost of extra capacity | Engineer 1.3 MNOK/yr; production 0.8 kNOK/h; FAT +25 %: 2 MNOK + 1.5 MNOK/yr | PLACEHOLDER |
| FAT hours | 6,972 h/yr (0.38 × production hours) | USER |
| Class approval | Case-by-case 13 wk, 50 % comments +4 wk; type approval 6.5 wk (25 %, +2 wk) | USER / PLACEHOLDER |
| Class share | DNV 80 %, ABS 10 %, BV 7 %, LR 3 % | USER |
| Margin | 20 % on own content, 20 % on traded sensors (separate variable) | USER |
| Working capital | Traded sensors paid on arrival, invoiced at delivery; 8 %/yr | assumption |
| LDs | Orgalim, 0.5 % per full week, max 7.5 %; bundled on mech + CS value (mech 20 MNOK) | USER / PLACEHOLDER |
| Retrofit market | Win probability flat up to ~6 months, then falling; express (≤ 4 months) +15 % price | PLACEHOLDER |
| Bundled scheduling | As late as possible (current practice) | USER |
| Rework / waiting | 30 % of today's production and FAT hours (software not ready, drawing errors, re-tests); half gone within a year, floor 5 %; varies strongly between projects (CV 1.0) | USER / PLACEHOLDER floor, CV |
| Repetition learning | On the remaining ("clean") hours, per family (PTC, CMS, BruCon-5), **85 %** per doubling, floor 50 %, 30 % transfer between families | Literature (see below) / USER |
| Experience today | Deliveries since 2023, volume doubled to 2026: PTC ≈ 239, CMS ≈ 72, BruCon-5 ≈ 93 simple-equivalents | USER (CMS history PLACEHOLDER) |
| Comment learning | Comment rate × 0.6 per doubling of approvals, down to 7.5 % | USER (target 5–10 %) |
| Today's quote (anchor) | Medium retrofit quoted 29.5 wk today (= model baseline) | **PLACEHOLDER: replace with real figure** |

## Model logic

```
order → engineering ─┬→ class approval (comments?) ───────┐
                     └→ production (needs hardware) ──────┴→ FAT → shipping
```

- Engineering, production and FAT are **capacity calendars**. Each project books capacity in specific
  weeks, up to a maximum rate per project.
- Production starts when engineering is finished and the hardware has arrived. Hardware is in stock for
  simple and medium; for DP2, traded sensors take 9–13 weeks unless a stock set is available.
  Production does not wait for approval. **FAT waits for approval**. FAT is booked using the nominal
  approval time. If comments delay approval beyond the booked slot, the slot is **released and FAT is
  rebooked** when the comments arrive.
- **Bundled projects** are booked as late as possible, allowing time for approval comments (current
  practice). With the *pull forward* lever, capacity that would otherwise be idle in the current week
  is used for bundled work, so bundled work acts as a buffer around retrofit projects.
- **Retrofit** opportunities are quoted the earliest feasible delivery. Customers accept with probability
  p(L). The model includes the express premium, the discount for > 6 months, and LDs if delivery is
  later than quoted.
- **Contribution** = own content × own margin + traded sensors × traded margin. Working capital on
  sensors is charged from arrival to delivery.
- **Rework**: production and FAT hours = clean hours × (1 + r/(1−r) × g), with r the rework share and
  g a per-project draw (mean 1, CV 1), so some projects are hit much harder than others. r falls from 30 %
  with a half-life of one year; every 4 weeks the remaining booked production/FAT work is rescaled. Less
  rework gives both fewer hours and less spread, so queues shrink twice.
- **Repetition learning per family** (on clean hours): hours per project = today's hours × max(floor, (n_eff / n_eff,0)^−b), with
  b = log₂(1 / learning rate) and n_eff = own experience + 0.3 × the other families' experience
  (simple-equivalents completed). New projects are booked with the family's current factor; when the
  factor improves, the remaining booked work of the family's active projects is scaled down. So more
  volume in one family (e.g. DP2) makes that family learn faster. The class comment rate falls with the
  number of approvals.

## Results (placeholders, see `results/summary.md`)

Production and FAT hours fall to about 0.6× in 5 years (rework 30 % → 6 %, repetition learning 85 %).
Turnover about 100 MNOK a year in the baseline. ΔNPV over 5 years against the baseline (as late as
possible, case-by-case approval):

| Lever | Lever value | ΔNPV vs baseline | What happens |
|---|---:|---:|---|
| 1c Rework *not* reduced | | −5 | FAT stays at ~85 %, quotes ~2 weeks longer |
| 0 Bundled as early as possible | | −1 | Blocks near-term capacity |
| 1b Quote incl. approval comments | | −9 | Every quote 4 weeks longer although comments become rare |
| 2 Pull bundled work into idle capacity | +3 | +3 | Less valuable now that rework reduction frees capacity |
| **3 + Type approval DNV** | **+15** | **+18** | **The most valuable lever**: ~40 express (≤ 4 months) wins a year |
| 4 + 1 sensor stock set | −3 | +15 | Too few DP2 |
| 5 + Type approval ABS | 0 | +15 | Only 10 % of volume |
| 5b + 10 % production instead | −7 | +8 | Not needed |
| 6 + 25 % FAT capacity | −8 | +7 | Not needed if rework halves |
| 7 + 10 % production as well | −7 | 0 | Not needed |

### Sensitivities (`results/sensitivity.md`, 40 runs each)

ΔNPV of each lever vs the baseline of the same variant [MNOK]:

| Variant | Baseline NPV vs base | Load prod / FAT | Pull fwd | + TA DNV | + 25 % FAT | FAT load with TA |
|---|---:|---|---:|---:|---:|---:|
| **Base: rework 30 %, half-life 1 yr, learning 85 %** | 0 | 66 % / 72 % | +3 | +19 | +8 | 80 % |
| Rework 20 % today | −2 | 72 % / 78 % | +5 | +19 | +9 | 88 % |
| Rework 40 % today | +2 | 59 % / 65 % | +2 | +18 | +7 | 72 % |
| Rework half-life 6 months | +1 | 64 % / 70 % | +3 | +18 | +7 | 77 % |
| Rework half-life 2 years | −1 | 69 % / 75 % | +4 | +19 | +9 | 85 % |
| **No rework reduction** | **−6** | 78 % / 85 % | +8 | +20 | +11 | **97 %** |
| Learning 80 % | +1 | 63 % / 69 % | +2 | +18 | +8 | 76 % |
| Learning 90 % | −1 | 69 % / 75 % | +4 | +19 | +8 | 85 % |
| No learning, no rework reduction | −18 | 81 % / 88 % | +15 | +20 | +14 | 100 % |

Main messages at today's volume:

1. **Type approval for DNV is robust**: +18 to +20 MNOK in every variant.
2. **Whether FAT capacity must rise depends on rework.** If rework halves within a year, FAT falls to
   ~70–80 % and extra FAT capacity is not worth it. If rework stays, FAT is at ~98 % once type approval
   brings more express orders. **Track rework hours and FAT re-tests**; this decides the FAT investment.
3. **The exact learning rate (80–90 %) matters little** at today's volume; the rework assumption matters
   more. The biggest effect is whether improvement happens at all (−18 MNOK if neither).
4. The value of halving rework at today's volume is about +6 MNOK in contribution. Much larger value
   comes from the capacity it frees for growth (see the DP2 case).

### Sources for learning rates

| Source | Finding |
|---|---|
| Wright (1936), *J. Aeronautical Sciences*, "Factors affecting the cost of airplanes" | Original learning curve, ~80 % in aircraft assembly |
| Dutton & Thomas (1984), *Academy of Management Review*, "Treating progress functions as a managerial opportunity" | ~200 studies: mostly 70–90 %, centred ~80–82 % |
| Argote & Epple (1990), *Science*, "Learning curves in manufacturing" | Overview; experience depreciates (turnover, gaps), which pulls the rate up |
| NASA Cost Estimating Handbook / learning curve calculator | Aerospace ~85 %, electronics 90–95 %, repetitive electrical 75–85 % |
| Argote (2013), *Organizational Learning* (book) | Transfer of learning between product lines |

A historically fitted curve also contains rework removal. Here the two are separated: rework (driven by
fixing causes, falls fast) and repetition learning (driven by volume, slow). The best calibration is
your own data: hours per project per quarter against cumulative projects, on log-log axes.

## Case: 300 MNOK/yr retrofit on DP2

`target_advanced.py` → `results/target_advanced*.md/png`.

300 MNOK / 10 MNOK = **30 retrofit DP2 projects per year**, reached by year 3 and held in years 3–5.
Today: 3 DP2 per year in total, ≈ 1.2 retrofit (placeholder split). The script grows the DP2 retrofit
market (opportunities × M, linear to year 3), and finds the M each lever set needs.

Results with rework 30 % → half in a year and learning 85 % (DP2 = 3.5× a simple project):

| Lever set | DP2 won/yr at M = 30 | Market needed for 30 | Win rate | DP2 quote | Loads eng / prod / FAT | ΔNPV at same market |
|---|---:|---:|---:|---:|---|---:|
| A Baseline | 36 | 60 opps (M ≈ 23) | 51 % | 27 wk ⚠ | 96 % / 86 % / 92 % | 0 |
| B + Pull forward | 44 | 47 (M ≈ 18) | 65 % | 22 wk | **98 %** / 88 % / 96 % | +46 |
| **C + TA DNV** | 45 | **45 (M ≈ 17)** | **68 %** | **18 wk** | **99 %** / 89 % / 98 % | **+113** |
| D + 4 sensor stock sets | 45 | 46 | 68 % | 17 wk | 99 % / 90 % / 98 % | +113 |
| E D + 25 % FAT | 48 | 44 | 69 % | 17 wk | 98 % / 89 % / 77 % | +114 |
| G D + 25 % FAT + 25 % production | 48 | 44 | 69 % | 17 wk | 98 % / 71 % / 77 % | +97 |
| **I G + 25 % engineers** | **54** | 44 | 69 % | 16 wk | **81 %** / 74 % / 80 % | +65 |
| J/K + 50 % FAT/prod, +25/50 % engineers | 54 | 44 | 69 % | 16 wk | ≤ 81 % / 61 % / 66 % | +50 / +16 |

What it takes (with rework halved):

1. **The target is reachable without extra production or FAT capacity**, if rework halves within a year.
   Pull forward + type approval DNV (+ sensor stock) are enough, at ~44 DP2 opportunities a year.
2. **Engineering becomes the bottleneck** (98–99 %), since rework reduction frees production and FAT but
   not engineering. For a robust plan: **+25 % engineers (~6) and +25 % FAT**. Note that drawing errors
   probably also cost engineering hours today; if engineering rework is reduced too, fewer engineers are
   needed. This is not modelled yet.
3. **The market must grow about 15×**, from ~3 to ~44 DP2 retrofit opportunities a year.
4. **About 180 MNOK of sensor purchases** a year, with 9–13 weeks lead time.
5. The earlier runs (`target_advanced_no_learning.md`, `_w6.md`) are **from before the rework model**
   and show the case without improvement: then +50 % FAT and production and +25 % engineers are needed.

**Hand check: capacity needed for 30 DP2 a year** (≈ 27 more than today), years 3–5:

    need = today's need + 27 × hours per DP2,   × hours factor (engineering ~0.88; production/FAT ~0.62)

| Resource | Total need without improvement | Factor | Need | Available (98 %) | Result |
|---|---:|---:|---:|---:|---|
| Engineering | 610 + 27 × 14.4 = 999 eng-wk | 0.88 (learning only) | 879 | 847 | **+4 %, at the limit** |
| Production | 17,726 + 27 × 418 = 29,012 h | 0.62 (rework + learning) | 17,990 | 20,437 | fits (86 %) |
| FAT | 6,811 + 27 × 161 = 11,158 h | 0.62 | 6,918 | 7,256 | fits, but at 93 % |

Matches the model: engineering is binding, FAT tight, production has room.

## Hand check

The model is verified by calculator-level formulas with the **same inputs** (`hand_check.py`, run
automatically after the model). The check compares **each lever's own value** (step vs previous
scenario) by hand and by model. They must agree within 3 MNOK or 50 %. Otherwise the step is flagged
with **CHECK** in `results/hand_check.md`.

### Formulas

The worked numbers below illustrate the formulas with today's inputs. The current numbers from the
latest run are in `results/hand_check.md`, and the result table at the end of this section is
updated automatically.

**1. Load (which resource is the bottleneck)**

    ρ = hours needed per year / hours available per year

Volume in simple-equivalents (simple = 1, medium = 2, DP2 = 3.5): 68 + 35 × 2 + 3 × 3.5 = 148.5 per year.

| Resource | Need | Available | ρ |
|---|---:|---:|---:|
| Engineering | 610 eng-weeks | 24 × 0.8 × 45 = 864 | 71 % |
| Production | 17,726 h | 20,854 h | 85 % |
| FAT | 6,811 h | 7,404 h | **92 %** |

**2. Minimum lead time (critical path, no queues)**

    L_min = max( max(eng, hardware) + production ,  eng + approval ) + FAT + shipping

| Class | Case-by-case | Type approved | Type approved + sensor stock |
|---|---:|---:|---:|
| Simple | 19.7 | 13.2 | 13.2 |
| Medium | 21.3 | 14.8 | 14.8 |
| DP2 | 21.6 | 18.1 | 15.1 |

Express means ≤ 17 weeks, so it is impossible with case-by-case approval. For DP2 with type approval,
the sensor lead time (9–13 weeks) is on the critical path. Comments add 0.5 × 4 = 2 weeks on average.

**3. Queue: anchored on today's quote, scaled with the queueing factor**

    W_today = L_quoted_today − L_min − safety − 0.5          (PTC medium: 29.5 − 20.6 − 1 − 0.5 = 7.4 weeks)
    W_new   = W_today × [ρ / (1 − ρ_seen)]_new / [ρ / (1 − ρ)]_today

- ρ_seen = total load when served first come first served (today).
- ρ_seen = retrofit load only when bundled work is used as a buffer (retrofit gets priority).
- The factor is summed over engineering, production and FAT, weighted by mean job time.
- The 0.5 is rounding, since quotes are given in whole weeks.

This is the standard M/G/1 waiting time (Pollaczek–Khinchine), and Cobham's formula for priority.

| ρ | 0.80 | 0.85 | 0.90 | 0.92 | 0.95 |
|---|---|---|---|---|---|
| ρ / (1 − ρ) | 4.0 | 5.7 | 9.0 | 11.5 | 19.0 |

The last few percent of load cost a disproportionate amount of lead time.

Quoted lead time:

    L = L_min + W + safety + 0.5

**4. Contribution per project**

    price × [(1 − traded share) × own margin + traded share × traded margin]
      + price × (express share × 15 % − long share × 5 %) − working capital on sensors

| Class | Price | Traded sensors (cost) | Contribution | Express extra |
|---|---:|---:|---:|---:|
| Simple | 300 kNOK | – | 60 | 45 |
| Medium | 1,000 kNOK | – | 200 | 150 |
| DP2 | 10,000 kNOK | 6,000 | 2,000 | 1,500 |

Express and long shares assume quotes spread ±3 weeks around the average.

**5. Win rate**

    won = opportunities × p(L)        opportunities = today's wins / p(today's L)    (≈ 98 per year)

This uses the same p(L) curve as the model, so the hand check verifies the model's mechanics, not the market assumption.

**6. NPV**

    ΔNPV = Σ_years (Δcontribution − lever costs) / 1.08^(y + 0.5) − investment

Type approval takes effect from year 2.

**7. Rework**

    rework share r_t = 5 % + (30 % − 5 %) × 0.5^(t / 1 year)
    production/FAT hours factor = (1 − r_0) / (1 − r_t)            year 1–5: 0.91, 0.81, 0.77, 0.75, 0.75

**8. Repetition learning (per family)**

    hours factor = max(50 %, (n_eff,y / n_eff,0)^−0.234)          (85 %: b = log₂(1/0.85) = 0.234)       n_eff = own + 0.3 × other families' experience
    n_y = experience today + simple-equivalents per year × (y + 0.5)

Hours per project are multiplied by the factor. Experience today (from deliveries since 2023, volume
doubled to 2026): PTC 239, CMS 72, BruCon-5 93 simple-equivalents. At today's volume all families
give (at 85 %) about 0.96, 0.90, 0.86, 0.82, 0.80 for years 1–5. Combined with rework, production and FAT:
0.87, 0.73, 0.66, 0.62, 0.59.

Steps 1–5 depend on each other (more wins mean higher load and longer lead times). By hand, repeat them
2–3 times. The script iterates until the numbers settle, and caps load at 98 %.

### Result of the check

<!-- hand-check:begin (generated by hand_check.py, do not edit by hand) -->
Latest run (with learning), ΔNPV vs baseline over 5 years [MNOK]:

| Scenario | Lever value hand / model | ΔNPV hand | ΔNPV model | Check |
|---|---:|---:|---:|---|
| 2 +pull bundled fwd | 6.4 / 3.0 | 6 | 3 | **CHECK** |
| 3 +TA DNV | 12.7 / 14.9 | 19 | 18 | OK |
| 4 +1 sensor stock set | -2.1 / -2.6 | 17 | 15 | OK |
| 5 +TA ABS | -0.5 / -0.2 | 17 | 15 | OK |
| 5b +10% prod. (not FAT) | -6.7 / -7.2 (vs 5) | 10 | 8 | OK |
| 6 +25% FAT capacity | -7.4 / -8.0 | 9 | 7 | OK |
| 7 +FAT +production | -6.5 / -7.2 | 3 | -0 | OK |

Check: lever values agree within 3 MNOK or 50%. Steps outside: 1.
<!-- hand-check:end -->

The FAT and production loads (92 % / 85 %) are still placeholders: capacity = reported hours / assumed
load. Today's queue of ~7 weeks matches what is seen in practice (USER), which supports the combination
of load and variability, but not where the queue sits. Next calibration: FAT benches/hours, FAT waiting
time, rework share.

The hand calculation agrees with the model on load, lead times, retrofit volume and most lever values.
**Pull forward** is flagged in the latest run (hand ≈ 6, model ≈ 3 MNOK): the hand formula gives retrofit
full priority over bundled work at every resource, while in the model bundled work only gives way in idle
weeks. With rework halving and loads falling, the model value is the more realistic one.
It **overestimates express** volume (≈ 29 vs 22 a year with type approval), because its queue estimate
is about 1 week shorter than the model's. Levers whose value comes mainly from express pricing
(type approval, sensor stock) are therefore higher by hand. Read them as a range: hand = optimistic end,
model = conservative end.

## Still to calibrate

- **Bundled / retrofit split per class (highest priority)**: currently 60 / 40 for all.
- **DP2 own hours**: hours booked on the 3 DP2 projects (decides 3.5× vs 6×), and medium's weight (2×).
- **FAT**: test hours per project, number of benches and real load. FAT is the bottleneck.
- **Today's quoted lead time** for a retrofit (anchors the hand check).
- **Rework**: actual share of production/FAT hours (re-tests, NCRs, waiting for software/drawings), and
  whether engineering also has significant rework from drawing errors.
- **Learning**: hours per project over time per family; CMS delivery history (assumed as PTC).
- Classification of CMS and PTC – Other (assumed simple).
- Win probability vs quoted lead time, and the size of the express premium. No quote history is
  available, so a sales-team estimate would help.
- Market for DP2 retrofit: how many opportunities could realistically be generated.
- Sensor suppliers' capacity, and the margin on traded sensors (20 % for now).
- Costs of type approval, extra FAT capacity, extra production capacity and extra engineers.
