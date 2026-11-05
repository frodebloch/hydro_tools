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
| Simple | 0.3 MNOK | – | 1 | 154 h (4.1 wk) | 212 h | 61 h | stock |
| Medium | 1 MNOK | – | 2 (PLACEHOLDER) | 308 h | 425 h | 122 h | stock |
| Advanced (DP2) | 10 MNOK | 75 % of price | 3.5 (PLACEHOLDER, test 6) | 539 h | 744 h | 213 h | sensors 9–13 wk |

Hours per project = class weight × hours per simple-equivalent. The hours per simple-equivalent are
calibrated so that the 2026 volume (148.5 simple-equivalents) reproduces the hours **booked on the
projects**: 23,400 h engineering and 32,300 h production (USER), minus the BruCon-4 share.

### Other inputs

| Input | Value | Source |
|---|---|---|
| Bundled / retrofit split | 60 % / 40 % for all classes | **PLACEHOLDER (highest priority)** |
| Capacity | 24 engineers × 80 % × 45 wk; production load 85 %; FAT load 92 % (bottleneck) | USER / PLACEHOLDER loads |
| Cost of extra capacity | Engineer 1.3 MNOK/yr; production 0.8 kNOK/h; FAT +25 %: 2 MNOK + 1.5 MNOK/yr | PLACEHOLDER |
| FAT hours | 0.286 × production hours | PLACEHOLDER |
| Class approval | Case-by-case 13 wk, 50 % comments +4 wk; type approval 6.5 wk (25 %, +2 wk) | USER / PLACEHOLDER |
| Class share | DNV 80 %, ABS 10 %, BV 7 %, LR 3 % | USER |
| Margin | 20 % on own content, 20 % on traded sensors (separate variable) | USER |
| Working capital | Traded sensors paid on arrival, invoiced at delivery; 8 %/yr | assumption |
| LDs | Orgalim, 0.5 % per full week, max 7.5 %; bundled on mech + CS value (mech 20 MNOK) | USER / PLACEHOLDER |
| Retrofit market | Win probability flat up to ~6 months, then falling; express (≤ 4 months) +15 % price | PLACEHOLDER |
| Bundled scheduling | As late as possible (current practice) | USER |
| Learning | Per family (PTC, CMS, BruCon-5), 90 % per doubling, floor 50 %, 30 % transfer between families | USER |
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
- **Learning per family**: hours per project = today's hours × max(floor, (n_eff / n_eff,0)^−b), with
  b = log₂(1 / learning rate) and n_eff = own experience + 0.3 × the other families' experience
  (simple-equivalents completed). New projects are booked with the family's current factor; when the
  factor improves, the remaining booked work of the family's active projects is scaled down. So more
  volume in one family (e.g. DP2) makes that family learn faster. The class comment rate falls with the
  number of approvals.

## Results (placeholders, see `results/summary.md`)

Turnover about 95 MNOK a year in the baseline (about 110 MNOK with all levers). ΔNPV over 5 years
against the baseline (as late as possible, case-by-case approval):

| Lever | Lever value, with learning | ΔNPV vs baseline, with learning | ΔNPV vs baseline, without learning | What happens |
|---|---:|---:|---:|---|
| Learning itself | | +10 | | Hours fall to ~0.85× in 5 years (all families), comments to ~7 % |
| 0 Bundled as early as possible | | −1 | −2 | Blocks near-term capacity |
| 1b Quote incl. approval comments | | −9 | −3 | Every quote 4 weeks longer although comments become rare |
| **2 Pull bundled work into idle capacity** | **+12** | +12 | +15 | Nearly free; retrofit wins 52 → 63 a year |
| **3 + Type approval DNV** | **+8** | +19 | +17 | Express (≤ 4 months) becomes possible: ~31 express wins a year |
| 4 + 1 sensor stock set | −3 | +16 | +12 | 6 MNOK stock for ~1 retrofit DP2 a year |
| 5 + Type approval ABS | −1 | +15 | +10 | Only 10 % of volume |
| 5b + 10 % production instead | −13 | +1 | −4 | Not the bottleneck |
| 6 + 25 % FAT capacity | −3 | +11 | +12 | FAT is at 100 % after step 2, but the gain does not cover the cost |
| 7 + 10 % production as well | −13 | −1 | −1 | Not needed |

Main messages at today's volume:

1. **The economics are modest at today's volume** (contribution about 20 MNOK a year). Lever values are
   single- to low double-digit MNOK.
2. **Pull forward (+12) and type approval for DNV (+8)** pay off in all cases tested.
3. **Learning is slower than first assumed**, because PTC and BruCon-5 have been delivered since 2023
   (about 3 years of experience). Hours fall only to ~0.85× in 5 years. Learning alone does not remove the
   FAT bottleneck.
4. **Sensor stock does not pay off at 3 DP2 a year.**

## Case: 300 MNOK/yr retrofit on DP2

`target_advanced.py` → `results/target_advanced*.md/png`.

300 MNOK / 10 MNOK = **30 retrofit DP2 projects per year**, reached by year 3 and held in years 3–5.
Today: 3 DP2 per year in total, ≈ 1.2 retrofit (placeholder split). The script grows the DP2 retrofit
market (opportunities × M, linear to year 3), and finds the M each lever set needs.

Retrofit DP2 won per year in years 3–5, at the largest market tested (M = 40, ~120 opportunities a year):

| Lever set | With learning, DP2 = 3.5× | Without learning | With learning, DP2 = 6× |
|---|---:|---:|---:|
| A Baseline | 22 | 16 | 14 |
| B–D + pull forward, TA DNV, 4 sensor stock sets | 23 | 17 | 15 |
| E D + 25 % FAT | 30 (just) | 22 | 19 |
| G D + 25 % FAT + 25 % production | **38** ✓ (engineering 100 %) | 28 | 24 |
| H D + 50 % FAT + 50 % production | **40** ✓ (engineering 100 %) | 30 | 24 |
| I G + 25 % engineers | 38 ✓ | 28 | 25 |
| **J H + 25 % engineers** | **53** ✓ | **39** ✓ | **34** ✓ (quotes > 6 mo) |
| K H + 50 % engineers | 53 | 39 | 34 |

At the market needed to hit 30 a year:

| Case | Lever set | Market needed (DP2 opportunities/yr) | Win rate | DP2 quote | Loads eng / prod / FAT | ΔNPV vs baseline at same market |
|---|---|---:|---:|---:|---|---:|
| With learning | G: +25 % FAT, +25 % production (minimum) | 58 (M ≈ 18) | 55 % | 24 wk | **100 % / 95 % / 100 %** | +61 MNOK |
| With learning | H: +50 % FAT, +50 % production | 52 (M ≈ 16) | 64 % | 21 wk | **100 %** / 80 % / 86 % | +64 MNOK |
| With learning | **J: H + 25 % engineers (robust)** | **47 (M ≈ 14)** | **68 %** | **19 wk** | 92 % / 90 % / 96 % | +41 MNOK |
| Without learning | J: H + 25 % engineers | 55 (M ≈ 17) | 58 % | 23 wk | 99 % / 97 % / 99 % | +19 MNOK |
| DP2 = 6× | J: H + 25 % engineers | 63 (M ≈ 19) | 48 % | 27 wk ⚠ | 100 % / 98 % / 100 % | −2 MNOK |

What it takes:

1. **Capacity: about +50 % FAT, +50 % production and +25 % engineering (~6 engineers).** With only
   +25 % FAT and production the target is just reached, but engineering and FAT run at 100 %, which is
   fragile. Without learning, or if a DP2 is heavier than assumed, the full package is needed.
2. **The market must grow about 14–19×**, from ~3 to ~50–60 DP2 retrofit opportunities a year, and we
   win only 1 in 2 to 2 in 3 of them.
3. **Without capacity expansion** the extra DP2 projects only crowd out other retrofit, at quotes over
   6 months. The maximum is then ~23 DP2 a year.
4. **Supply chain and money:** 30 DP2 a year means about **180 MNOK of sensor purchases** a year with
   9–13 weeks lead time. Supplier capacity and working capital have to be secured.
5. **How heavy a DP2 is in own hours** (3.5× vs 6×) decides whether the case pays off (+41 vs −2 MNOK
   for the robust lever set). Measure the hours booked on the 3 DP2 projects.

**Hand check: capacity needed for 30 DP2 a year** (≈ 27 more than today):

    need = today's need + 27 × hours per DP2,   then × the learning factor (~0.85 in years 3–5)

| Resource | Today's need | + 27 DP2 | Total | Available (98 %) | Without learning | With learning (× 0.85) |
|---|---:|---:|---:|---:|---|---|
| Engineering | 610 eng-wk | 27 × 14.4 = 389 | 999 | 847 | **+18 %** | 849: **at the limit** |
| Production | 31,556 h | 27 × 744 = 20,088 | 51,644 | 36,382 | **+42 %** | 43,900: **+21 %** |
| FAT | 9,025 h | 27 × 213 = 5,751 | 14,776 | 9,614 | **+54 %** | 12,560: **+31 %** |

This matches the model. With learning, +25 % production and FAT is just enough, with engineering at its
limit; with margin for queues (loads ≤ ~90 %), roughly +50 % FAT and production and +25 % engineering
are needed.

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
| Production | 31,556 h | 37,125 h | 85 % |
| FAT | 9,025 h | 9,810 h | **92 %** |

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

    W_today = L_quoted_today − L_min − safety − 0.5          (medium: 29.5 − 21.3 − 1 − 0.5 = 6.7 weeks)
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

**7. Learning (per family)**

    hours factor = max(50 %, (n_eff,y / n_eff,0)^−0.152)       n_eff = own + 0.3 × other families' experience
    n_y = experience today + simple-equivalents per year × (y + 0.5)

Hours per project are multiplied by the factor. Experience today (from deliveries since 2023, volume
doubled to 2026): PTC 239, CMS 72, BruCon-5 93 simple-equivalents. At today's volume all families
give 0.97, 0.94, 0.91, 0.88, 0.86 for years 1–5.

Steps 1–5 depend on each other (more wins mean higher load and longer lead times). By hand, repeat them
2–3 times. The script iterates until the numbers settle, and caps load at 98 %.

### Result of the check

<!-- hand-check:begin (generated by hand_check.py, do not edit by hand) -->
Latest run (with learning), ΔNPV vs baseline over 5 years [MNOK]:

| Scenario | Lever value hand / model | ΔNPV hand | ΔNPV model | Check |
|---|---:|---:|---:|---|
| 2 +pull bundled fwd | 13.1 / 11.8 | 13 | 12 | OK |
| 3 +TA DNV | 5.9 / 7.6 | 19 | 19 | OK |
| 4 +1 sensor stock set | -0.4 / -3.4 | 19 | 16 | OK |
| 5 +TA ABS | -1.2 / -1.4 | 17 | 15 | OK |
| 5b +10% prod. (not FAT) | -11.9 / -13.1 (vs 5) | 5 | 1 | OK |
| 6 +25% FAT capacity | -0.9 / -3.1 | 16 | 11 | OK |
| 7 +FAT +production | -10.9 / -12.8 | 5 | -1 | OK |

Check: lever values agree within 3 MNOK or 50%. Steps outside: 0.
<!-- hand-check:end -->

The hand calculation agrees with the model on load, lead times, retrofit volume and most lever values.
It **overestimates express** volume (≈ 29 vs 22 a year with type approval), because its queue estimate
is about 1 week shorter than the model's. Levers whose value comes mainly from express pricing
(type approval, sensor stock) are therefore higher by hand. Read them as a range: hand = optimistic end,
model = conservative end.

## Still to calibrate

- **Bundled / retrofit split per class (highest priority)**: currently 60 / 40 for all.
- **DP2 own hours**: hours booked on the 3 DP2 projects (decides 3.5× vs 6×), and medium's weight (2×).
- **FAT**: test hours per project, number of benches and real load. FAT is the bottleneck.
- **Today's quoted lead time** for a retrofit (anchors the hand check).
- **Learning**: hours per project over time per family; CMS delivery history (assumed as PTC).
- Classification of CMS and PTC – Other (assumed simple).
- Win probability vs quoted lead time, and the size of the express premium. No quote history is
  available, so a sales-team estimate would help.
- Market for DP2 retrofit: how many opportunities could realistically be generated.
- Sensor suppliers' capacity, and the margin on traded sensors (20 % for now).
- Costs of type approval, extra FAT capacity, extra production capacity and extra engineers.
