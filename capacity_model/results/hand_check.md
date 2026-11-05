# Hand check

Calculator-level formulas (see README, 'Hand check') with the same inputs as the model.

## 1. Load at baseline volumes

| Resource | Need/yr | Available/yr | Load ρ |
|---|---:|---:|---:|
| engineering | 610 eng-wk | 864 eng-wk | 71% |
| production | 31,556 h | 37,125 h | 85% |
| FAT | 9,025 h | 9,810 h | 92% |

## 2. Minimum lead time without queues (weeks)

| Class | Case-by-case | Type approved | Type approved + sensor stock |
|---|---:|---:|---:|
| simple | 19.7 | 13.2 | 13.2 |
| medium | 21.3 | 14.8 | 14.8 |
| advanced | 21.6 | 18.1 | 15.1 |

Express limit: 17 weeks. Comments add on average 2.0 weeks (case-by-case).

## 3. Queue factor ρ/(1−ρ)

| ρ | 0.80 | 0.85 | 0.90 | 0.92 | 0.95 |
|---|---|---|---|---|---|
| ρ/(1−ρ) | 4.0 | 5.7 | 9.0 | 11.5 | 19.0 |

## 4. Contribution per project (kNOK)

| Class | Price | Traded sensors (cost) | Contribution | Express extra |
|---|---:|---:|---:|---:|
| simple | 300 | 0 | 60 | 45 |
| medium | 1,000 | 0 | 200 | 150 |
| advanced | 10,000 | 6,000 | 2,000 | 1,500 |

## 7. Learning: hours factor per year (baseline volume)

| Year | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Hours factor | 0.88 | 0.79 | 0.74 | 0.71 | 0.68 |

Learning rate 90% per doubling, 60 units built today, floor 50%.

Today's queue (anchor): quoted medium lead time 29.5 wk − L_min 21.3 − safety 1 − rounding 0.5 = **6.7 weeks** of queue.

Retrofit opportunities by hand (today's wins / p(today's lead time)): simple 58, medium 37, advanced 3 = 98 per year.

## 5–6. Scenarios: hand calculation vs model

| Scenario | FAT load ρ (retrofit part), 5-yr avg. | Queue [wk] hand | Quote anchor class [wk] hand / model | Retrofit won/yr hand / model | Express/yr hand / model | ΔNPV [MNOK] hand / model | Ratio | Step = lever value [MNOK] hand / model | Check |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 baseline | 82% (40%) | 2.6 | 25 / 25 | 58.6 / 58.3 | 1.0 / 0.0 | 0.0 / 0.0 | – | – | – |
| 2 +pull bundled fwd | 84% (42%) | 1.0 | 24 / 25 | 62.8 / 62.7 | 2.7 / 0.0 | 6.1 / 5.8 | 1.04 | 6.1 / 5.8 | OK |
| 3 +TA DNV | 86% (45%) | 1.0 | 20 / 19 | 66.0 / 67.2 | 28.6 / 21.7 | 14.8 / 9.9 | 1.49 | 8.7 / 4.1 | **CHECK** |
| 4 +1 sensor stock set | 87% (45%) | 1.0 | 20 / 19 | 66.0 / 67.3 | 29.1 / 21.8 | 14.5 / 6.3 | 2.31 | -0.2 / -3.6 | **CHECK** |
| 5 +TA ABS | 87% (45%) | 1.0 | 19 / 19 | 66.4 / 67.8 | 32.3 / 24.5 | 13.8 / 4.6 | 3.02 | -0.7 / -1.7 | OK |
| 5b +10% prod. (not FAT) | 87% (45%) | 0.9 | 19 / 19 | 66.5 / 67.8 | 32.8 / 24.5 | 1.8 / -8.3 | -0.22 | -12.0 / -12.9 (vs 5) | OK |
| 6 +25% FAT capacity | 69% (36%) | 0.8 | 19 / 19 | 66.6 / 67.8 | 33.6 / 25.2 | 6.5 / -3.1 | -2.13 | -7.2 / -7.6 | OK |
| 7 +FAT +production | 69% (36%) | 0.7 | 19 / 19 | 66.7 / 67.8 | 34.1 / 25.2 | -5.4 / -15.9 | 0.34 | -11.9 / -12.8 | OK |

Check: each lever's own value (step vs previous) by hand and by model differ by at most 3 MNOK or 50% of the model value.
Steps outside: 2.

Hand-check values are averages over the 5 years. Known simplifications: average lead time per order type (spread only assumed as
±3 weeks for express share), queue anchored on today's quote and scaled, no late-delivery penalties (LDs), no approval
comments in the quote (as in the model), type approval from year 2.
