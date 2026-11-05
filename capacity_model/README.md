# Control-system capacity model

Monte Carlo model of the capacity for control systems sold (a) bundled with our own mechanical units
(1–3 years lead time) and (b) as retrofit (3–12 months, lead-time sensitive). It models engineering,
production and test/FAT capacity, hardware lead times, class approval and learning, and compares levers
by ΔNPV. A calculator-level **hand check** is run automatically against the model.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib
.venv/bin/python run_scenarios.py 200     # ~6 min, includes the hand check
.venv/bin/python hand_check.py            # hand check only (uses the latest model results)
```

Output in `results/`: `summary.md` (model table + learning sensitivity), `hand_check.md`
(hand formulas vs model), `scenarios.png`, `demand_sweep.png`, `calibration.json`, `model_means.json`.

| File | Content |
|---|---|
| `config.py` | All inputs (USER = given, PLACEHOLDER = guess), derived capacities, learning curve, scenarios |
| `sim.py` | Weekly simulation, retrofit quoting, approval/FAT rebooking, learning, NPV, retrofit calibration |
| `run_scenarios.py` | Calibration, Monte Carlo over scenarios (common random numbers), tables, plots, hand check |
| `hand_check.py` | Calculator-level formulas with the same inputs, compared with the model |

## Inputs

| Input | Value | Source |
|---|---|---|
| Order types | single (1 simple), quad (4 simple + simple top-level), package (8 simple + 1 advanced) | USER |
| Bundled volume | 50 single, 20 quad, 10 package per year; 1–3 years lead time | USER |
| Retrofit volume | 20 won per type per year today; opportunities back-calculated (≈ 37 / 43 / 46) | USER / calibrated |
| Engineering | 24 engineers, 80 %, 45 weeks/yr; 1 / 4 weeks per simple / advanced, scaled to 23,400 h/yr | USER |
| Production | 32,300 h/yr; advanced 3.5 × simple (≈ 53 h); lognormal spread CV 0.3; load 85 % | USER / PLACEHOLDER load |
| Test / FAT | Current bottleneck; 15 h per simple unit, load 92 % | PLACEHOLDER |
| Hardware | Simple in stock; advanced 9–13 weeks long-lead (75 % of cost) | USER |
| Class approval | Case-by-case 13 weeks, 50 % comments +4 weeks; type approval 6.5 weeks (25 % comments, +2 weeks) | USER / PLACEHOLDER |
| Class share | DNV 80 %, ABS 10 %, BV 7 %, LR 3 % | USER |
| Prices | Simple 300 kNOK, advanced 10 MNOK, margin 20 %; mechanical unit 20 MNOK | USER / PLACEHOLDER mech |
| LDs | Orgalim, 0.5 % per full week, max 7.5 %; bundled on mech + CS value | USER |
| Retrofit market | Win probability flat up to ~6 months, then falling; express (≤ 4 months) +15 % price | PLACEHOLDER |
| Bundled scheduling | As late as possible (current practice) | USER |
| Learning | 90 % per doubling (eng, production, FAT), floor 50 %, ~225 units built so far | USER |
| Comment learning | Comment rate × 0.6 per doubling of approvals, down to 7.5 % | USER (target 5–10 %) |
| Today's quote (anchor) | Quad retrofit quoted ~27 weeks today (anchors the hand check) | PLACEHOLDER |

## Model logic

```
order → engineering ─┬→ class approval (comments?) ───────┐
                     └→ production (needs hardware) ──────┴→ FAT → shipping
```

- Engineering, production and FAT are **capacity calendars**. Each order books capacity in specific
  weeks, up to a maximum rate per order.
- Production starts when engineering is finished and the hardware has arrived; it does not wait for
  approval. **FAT waits for approval**. FAT is booked using the nominal approval time. If comments delay
  approval beyond the booked slot, the slot is **released and FAT is rebooked** when the comments arrive.
- **Bundled orders** are booked as late as possible, allowing time for approval comments (current
  practice). With the *pull forward* lever, capacity that would otherwise be idle in the current week
  is used for bundled work, so bundled work acts as a buffer around retrofit orders.
- **Retrofit** opportunities are quoted the earliest feasible delivery. Customers accept with probability
  p(L). The model includes the express premium, the discount for > 6 months, and LDs if delivery is
  later than quoted.
- **Learning**: hours per unit = max(floor, (n / n₀)^−b), with b = log₂(1 / learning rate) and n the
  cumulative units completed. It is modelled as capacity growing from the current week onwards, so
  quotes reflect the performance reached so far and more volume gives faster learning. The class
  comment rate falls the same way with the number of approvals. Learning starts at "today" (end of
  warm-up); the reported hours are taken as today's level.

## Results (with placeholders, see `results/summary.md`)

ΔNPV over 5 years against the baseline (as late as possible, case-by-case approval), main case **with learning**:

| Step | ΔNPV with learning | Step value | Without learning | What happens |
|---|---:|---:|---:|---|
| Learning itself (baseline) | +120 | | | Hours fall to ~0.65× in 5 years, comments to ~7 % |
| 0 Bundled as early as possible | −1 | | −3 | Blocks near-term capacity |
| 1b Quote incl. approval comments | −93 | | −8 | Adds 4 weeks to every quote although comments become rare |
| **2 Pull bundled work into idle capacity** | +32 | **+32** | +68 | Cheap, but learning reduces the need |
| **3 + Type approval DNV** | +80 | **+48** | +85 | Most valuable lever with learning: express for DNV ships |
| 4 + 1 advanced stock set | +91 | +11 | +82 | Hardware becomes the critical path for packages once FAT eases |
| 5 + Type approval ABS | +97 | +6 | +83 | Small (10 % of volume) |
| 5b + 10 % production instead | +84 | −13 | +72 | Not needed |
| 6 + 25 % FAT capacity | +98 | **+1** | +173 | Learning frees FAT capacity by itself |
| 7 + 10 % production as well | +86 | −12 | +176 | Not needed |

Main messages:

1. **Learning changes the investment picture.** Without learning, extra FAT capacity is the largest
   lever (+90 MNOK). With learning at 90 % per doubling, FAT eases by itself within 1–2 years and the
   extra capacity is worth about zero. Before investing in fixed capacity, **check whether hours per
   unit are actually falling** (track hours per simple-equivalent, quarter by quarter).
2. **Type approval (DNV) is robust.** It is valuable with and without learning, because it shortens the
   critical path itself (approval 13 → 6.5 weeks). Learning does not remove it.
3. **Using bundled work as a buffer** costs almost nothing and is worth +30 to +70 MNOK.
4. **Advanced stock** turns slightly positive with learning: once FAT is no longer the bottleneck, the
   9–13 week hardware lead time limits packages.
5. **Quoting with allowance for comments** gets worse as comments become rare. Quote nominal approval and
   manage the remaining risk.

**Caveats**
- The gains come from winning more of the ~127 retrofit opportunities per year that the calibration
  implies (≈ 47 % win rate today). If there are fewer real opportunities, or customers are less
  sensitive to lead time, the gains are smaller.
- The learning inputs (rate, floor, units built) have a large effect on the result. They should be
  checked against the hours booked per unit over time.

## Hand check

The model is verified by calculator-level formulas with the **same inputs** (`hand_check.py`, run
automatically after the model). Each scenario's hand ΔNPV must be within **0.67–1.5×** the model's (or
both below 10 MNOK). Otherwise it is flagged with **CHECK** in `results/hand_check.md`.

### Formulas

The worked numbers below illustrate the formulas with today's inputs. The current numbers from the
latest run are in `results/hand_check.md`, and the result table at the end of this section is
updated automatically.

**1. Load (which resource is the bottleneck)**

    ρ = hours needed per year / hours available per year

Count volume in simple-equivalents (simple = 1, advanced = 3.5): single = 1, quad = 5, package = 11.5.
Today: 265 bundled + 350 retrofit = 615 per year.

| Resource | Need | Available | ρ |
|---|---|---|---|
| Engineering | 624 eng-weeks | 24 × 0.8 × 45 = 864 | 72 % |
| Production | 32,300 h | 38,000 h | 85 % |
| FAT | 615 × 15 h = 9,225 h | 10,027 h | **92 %** |

**2. Minimum lead time (critical path, no queues)**

    L_min = max( max(eng, hardware) + production ,  eng + approval ) + FAT + shipping

| Order | Case-by-case | Type approved | Type approved + adv. stock |
|---|---:|---:|---:|
| Single | 15.4 | 8.9 | 8.9 |
| Quad | 18.5 | 12.0 | 12.0 |
| Package | 20.3 | 16.3 | 13.8 |

Express means ≤ 17 weeks, so it is impossible with case-by-case approval. Comments add 0.5 × 4 = 2 weeks
on average.

**3. Queue: anchored on today's quote, scaled with the queueing factor**

    W_today = L_quoted_today − L_min − safety               (quad: 27 − 18.5 − 1 = 7.5 weeks)
    W_new   = W_today × [ρ / (1 − ρ_seen)]_new / [ρ / (1 − ρ)]_today

- ρ_seen = total load when orders are served first come first served (today).
- ρ_seen = retrofit load only when bundled work is used as a buffer (retrofit gets priority).
- The factor is summed over engineering, production and FAT, weighted by mean job time.

This is the standard M/G/1 waiting time (Pollaczek–Khinchine), and Cobham's formula for priority.

| ρ | 0.80 | 0.85 | 0.90 | 0.92 | 0.95 |
|---|---|---|---|---|---|
| ρ / (1 − ρ) | 4.0 | 5.7 | 9.0 | 11.5 | 19.0 |

The last few percent of load cost a disproportionate amount of lead time.

Quoted lead time:

    L = L_min + W + safety

**4. Contribution per order**

    price × (20 % margin + express share × 15 % − long share × 5 %)

| Type | Price | Contribution | Express extra |
|---|---:|---:|---:|
| Single | 300 kNOK | 60 | 45 |
| Quad | 1,500 kNOK | 300 | 225 |
| Package | 12,400 kNOK | 2,480 | 1,860 |

Express and long shares assume quotes spread ±3 weeks around the average.

**5. Win rate**

    won = opportunities × p(L)        opportunities = today's wins / p(today's L)    (≈ 119 per year)

This uses the same p(L) curve as the model, so the hand check verifies the model's mechanics, not the market assumption.

**6. NPV**

    ΔNPV = Σ_years (Δcontribution − lever costs) / 1.08^(y + 0.5) − investment

Type approval takes effect from year 2.

**7. Learning**

    hours factor in year y = max(50 %, (n_y / 225)^−0.152)       n_y = 225 + units per year × (y + 0.5)

Capacity is divided by the factor. At today's volume: 0.88, 0.78, 0.73, 0.70, 0.67 for years 1–5.

Steps 1–5 depend on each other (more wins mean higher load and longer lead times). By hand, repeat them
2–3 times. The script iterates until the numbers settle, and caps load at 98 %.

### Result of the check

<!-- hand-check:begin (generated by hand_check.py, do not edit by hand) -->
Latest run (with learning), ΔNPV vs baseline over 5 years [MNOK]:

| Scenario | Hand | Model | Ratio | Check |
|---|---:|---:|---:|---|
| 2 +pull bundled fwd | 46 | 32 | 1.42 | OK |
| 3 +TA DNV | 117 | 80 | 1.47 | OK |
| 4 +1 adv. stock set | 121 | 91 | 1.33 | OK |
| 5 +TA ABS | 129 | 97 | 1.33 | OK |
| 5b +10% prod. (not FAT) | 120 | 84 | 1.42 | OK |
| 6 +25% FAT capacity | 133 | 98 | 1.36 | OK |
| 7 +FAT +production | 126 | 86 | 1.47 | OK |

Scenarios outside the band: 0.
<!-- hand-check:end -->

With learning, the hand calculation is systematically about 35–45 % higher than the model, with the same
ranking and the same conclusions. This is because it ignores late-delivery penalties (LDs) and FAT
rebooking, and uses averages instead of spread.

## Case: 300 MNOK/yr retrofit on advanced units

`target_advanced.py [n_runs] [--no-learning]` → `results/target_advanced[_no_learning].md/.png`.

300 MNOK / 10 MNOK per advanced unit means **30 packages (8 simple + 1 advanced) per year**, reached by
year 3 and held in years 3–5 (≈ 372 MNOK including the simple units). Today the figure is 20 per year,
from ~46 package opportunities.

The script grows the package market (opportunities × M, linear to year 3). For each lever set it finds
the M needed, and reports the market, win rate, lead time, loads, side effects and ΔNPV at that point.

Packages won per year in years 3–5 (60 runs per point):

| Lever set | With learning, today's market | With learning, max (M = 2.5) | Without learning, today's market | Without learning, max (M = 2.5) |
|---|---:|---:|---:|---:|
| A Baseline | 28.5 | 51 | 20 | 29 (quotes > 6 mo) |
| B Pull bundled forward | 29.4 | 58 | 25 | 30 (quotes > 6 mo) |
| C B + TA DNV | **31.0** ✓ | 57 | 25 | 30 |
| D C + 2 advanced stock sets | 31.1 ✓ | 57 | 25 | 30 |
| E D + 25 % FAT | 31.1 ✓ | 70 | **30.3** ✓ | 37 |
| G D + 25 % FAT + 10 % production | 31.1 ✓ | 74 | 31.1 ✓ | 44 |

What it takes:

1. **With learning as assumed (90 %/doubling), the target is reached without market growth.** Pull
   forward plus **type approval for DNV** is enough: quotes fall to ~19 weeks and the package win rate
   rises from ~44 % to ~68 %. The baseline needs about +6 % more package opportunities.
2. **Without learning, capacity is the constraint, not the market.** Without extra FAT, packages plateau
   at ~30 a year even if the market grows 2.5×. Quotes exceed 6 months, and the extra packages crowd out
   single and quad retrofit (41 → 27 a year). **+25 % FAT reaches 30 a year at today's market.** Production
   is then at 100 %, so **+10 % production** is needed as well to have headroom (ceiling ~44 a year).
3. **Decision rule:**
   - Do type approval for DNV and pull forward now; they pay off in both cases.
   - Track hours per simple-equivalent. If year-1 hours are not ≈ 0.88× today's, start the FAT (+25 %)
     and production (+10 %) expansion, since without learning the target cannot be reached otherwise.
4. Bundled deliveries are not affected (0 % late) in any of these cases.

**Hand check of the capacity ceiling** for packages, per resource:

    max packages/yr = (0.98 × capacity − bundled − single/quad retrofit) / hours per package

Hours per package are 11.5 simple-equivalents × 15 h = 172.5 h of FAT, and 11.5 × 52.5 h = 604 h of
production. Divide the hours by the learning factor (~0.72 in years 3–5) when learning applies.

| Without learning | FAT | Production | Ceiling | Model |
|---|---|---|---|---|
| Today | (10,027 − 3,975 − 1,800) / 172.5 = 25 | (38,000 − 13,900 − 6,300) / 604 = 29 | **≈ 25** | 25 (B, M = 1) |
| + 25 % FAT | (12,534 − 5,775) / 172.5 = 39 | 29 | **≈ 29–30** | 30.5 (E, M = 1, production 100 %) |
| + FAT + 10 % production | 39 | (41,800 − 20,200) / 604 = 36 | **≈ 36** | 31 at M = 1 (market-limited); 44 at M = 2.5 by crowding out single/quad |

## Still to calibrate

- **FAT**: test hours per unit, number of benches and real load. This is the most important input.
- **Learning**: hours per simple-equivalent over time for engineering, production and FAT.
- **Today's quoted lead time** for a retrofit (anchors the hand check).
- Win probability vs quoted lead time, and the size of the express premium. No quote history is
  available, so a sales-team estimate would help.
- How many retrofit opportunities are really out there, including proactive offers.
- Price of a mechanical unit (drives bundled LDs). Bundled orders are never late in the model so far.
- Costs of type approval, extra FAT capacity and extra production capacity.
