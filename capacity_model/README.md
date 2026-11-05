# Control-system capacity model

Monte Carlo model of the capacity for control systems sold (a) bundled with our own mechanical units
(1–3 years lead time) and (b) as retrofit (3–12 months, lead-time sensitive). It models engineering,
production and test/FAT capacity, hardware lead times and class approval, and compares levers by ΔNPV.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib
.venv/bin/python run_scenarios.py 200     # ~5 min
```

Output in `results/`: `summary.md` (table), `scenarios.png`, `demand_sweep.png`, `calibration.json`.

| File | Content |
|---|---|
| `config.py` | All inputs (USER = given, PLACEHOLDER = guess), derived capacities, scenario definitions |
| `sim.py` | Weekly simulation, retrofit quoting, approval/FAT rebooking, NPV, retrofit calibration |
| `run_scenarios.py` | Calibration, Monte Carlo over scenarios (common random numbers), table and plots |

## Inputs

| Input | Value | Source |
|---|---|---|
| Order types | single (1 simple), quad (4 simple + simple top-level), package (8 simple + 1 advanced) | USER |
| Bundled volume | 50 single, 20 quad, 10 package per year; 1–3 years lead time | USER |
| Retrofit volume | 20 won per type per year in the baseline; opportunities back-calculated (≈ 37 / 43 / 46) | USER / calibrated |
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

## Results (with placeholders, see `results/summary.md`)

| Step | ΔNPV vs baseline, 5 yr | What happens |
|---|---|---|
| 0 Bundled as early as possible | ≈ −2 MNOK | Blocks near-term capacity |
| 1b Quote including approval comments | ≈ −7 MNOK | No LDs, but loses more orders than the LDs cost |
| **2 Pull bundled work into idle capacity** | **≈ +69 MNOK** | Retrofit wins rise from 60 to 73 per year, quotes drop to ~22–25 weeks |
| 3 + Type approval DNV | +19 | Express becomes possible, but FAT limits it |
| 4 + 1 advanced long-lead set in stock | −2 | Stock worth 6 MNOK, rarely on the critical path |
| 5 + Type approval ABS | ≈ 0 | Only 10 % of volume |
| 5b + 10 % production instead of FAT | −10 | Wrong bottleneck |
| **6 + 25 % FAT capacity** | **+88** | Express wins rise from 8 to 32 per year, quotes drop to 15–21 weeks |
| 7 + 10 % production as well | +3 | Production becomes the next limit |

Main messages:

1. **FAT gates everything.** Type approval and stock only pay off when FAT has slack. Extra
   production capacity does not help while FAT is the bottleneck.
2. **Using bundled work as a buffer is the cheapest lever.** Today, bundled work booked as late as
   possible leaves capacity idle while the retrofit queue grows.
3. **Advanced hardware stock does not pay off** at 6 MNOK per set with the current FAT and approval
   constraints.
4. Retrofit deliveries are often later than quoted (13–30 %), mostly because of approval comments
   and FAT rebooking. Quoting with an allowance for comments removes the LDs but loses more orders.
   This depends on the assumed win curve.

**Caveat:** the large gains come from winning more of the ~127 retrofit opportunities per year that the
calibration implies, at a win rate of ≈ 47 %. If there are fewer real opportunities, or customers are
less sensitive to lead time, the gains are smaller. The demand sweep shows how the results depend on
retrofit volume.

## Still to calibrate

- FAT: test hours per unit, number of benches, and real load. This is the most important input,
  because FAT is the bottleneck.
- Win probability vs quoted lead time, and the size of the express premium. No quote history is
  available, so a sales-team estimate would help.
- How many retrofit opportunities are really out there, including proactive offers.
- Price of a mechanical unit (drives bundled LDs). Bundled orders are never late in the model so far.
- Comment rate and comment-loop duration under type approval.
- Costs of type approval, extra FAT capacity and extra production capacity.
