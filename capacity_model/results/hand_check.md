# Hand check

Calculator-level formulas (see README, 'Hand check') with the same inputs as the model.

## 1. Load at baseline volumes

| Resource | Need/yr | Available/yr | Load ρ |
|---|---:|---:|---:|
| engineering | 624 eng-wk | 864 eng-wk | 72% |
| production | 32,300 h | 38,000 h | 85% |
| FAT | 9,225 h | 10,027 h | 92% |

## 2. Minimum lead time without queues (weeks)

| Order type | Case-by-case | Type approved | Type approved + adv. stock |
|---|---:|---:|---:|
| single | 15.4 | 8.9 | 8.9 |
| quad | 18.5 | 12.0 | 12.0 |
| package | 20.3 | 16.3 | 13.8 |

Express limit: 17 weeks. Comments add on average 2.0 weeks (case-by-case).

## 3. Queue factor ρ/(1−ρ)

| ρ | 0.80 | 0.85 | 0.90 | 0.92 | 0.95 |
|---|---|---|---|---|---|
| ρ/(1−ρ) | 4.0 | 5.7 | 9.0 | 11.5 | 19.0 |

## 4. Contribution per order (kNOK)

| Type | Price | Contribution | Express extra |
|---|---:|---:|---:|
| single | 300 | 60 | 45 |
| quad | 1,500 | 300 | 225 |
| package | 12,400 | 2,480 | 1,860 |

## 7. Learning: hours factor per year (baseline volume)

| Year | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Hours factor | 0.88 | 0.78 | 0.73 | 0.70 | 0.67 |

Learning rate 90% per doubling, 225 units built today, floor 50%.

Today's queue (anchor): quoted quad lead time 27 wk − L_min 18.5 − safety 1 = **7.5 weeks** of queue.

Retrofit opportunities by hand (today's wins / p(today's lead time)): single 32, quad 39, package 48 = 119 per year.

## 5–6. Scenarios: hand calculation vs model

| Scenario | FAT load ρ (retrofit part), 5-yr avg. | Queue [wk] hand | Quote quad [wk] hand / model | Retrofit won/yr hand / model | Express/yr hand / model | ΔNPV [MNOK] hand / model | Ratio | Check | Step vs previous [MNOK] hand / model |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 baseline | 83% (53%) | 3.2 | 23 / 23 | 75.3 / 78.6 | 7.6 / 0.0 | 0.0 / 0.0 | – | – | – |
| 2 +pull bundled fwd | 86% (56%) | 1.4 | 21 / 22 | 78.7 / 82.8 | 14.9 / 0.0 | 45.5 / 32.1 | 1.42 | OK | 45.5 / 32.1 |
| 3 +TA DNV | 87% (58%) | 1.5 | 17 / 16 | 80.7 / 85.3 | 40.9 / 36.5 | 116.9 / 79.7 | 1.47 | OK | 71.3 / 47.5 |
| 4 +1 adv. stock set | 88% (58%) | 1.5 | 17 / 16 | 80.7 / 85.4 | 42.0 / 38.5 | 121.1 / 90.9 | 1.33 | OK | 4.2 / 11.3 |
| 5 +TA ABS | 88% (58%) | 1.5 | 16 / 16 | 81.0 / 85.7 | 45.3 / 43.3 | 128.6 / 96.6 | 1.33 | OK | 7.5 / 5.7 |
| 5b +10% prod. (not FAT) | 88% (58%) | 1.4 | 16 / 16 | 81.0 / 85.7 | 46.1 / 43.5 | 119.7 / 84.1 | 1.42 | OK | -8.9 / -12.5 (vs 5) |
| 6 +25% FAT capacity | 70% (46%) | 1.2 | 16 / 16 | 81.5 / 86.2 | 47.5 / 45.6 | 133.3 / 97.9 | 1.36 | OK | 4.7 / 1.3 |
| 7 +FAT +production | 69% (46%) | 1.0 | 16 / 16 | 81.6 / 86.2 | 48.5 / 45.7 | 125.6 / 85.6 | 1.47 | OK | -7.7 / -12.4 |

Ballpark agreement: hand/model ΔNPV ratio within 0.67–1.5, or both below 10 MNOK.
Scenarios outside: 0.

Hand-check values are averages over the 5 years. Known simplifications: average lead time per order type (spread only assumed as
±3 weeks for express share), queue anchored on today's quote and scaled, no late-delivery penalties (LDs), no approval
comments in the quote (as in the model), type approval from year 2.
