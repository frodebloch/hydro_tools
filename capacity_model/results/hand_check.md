# Hand check

Calculator-level formulas (see README, 'Hand check') with the same inputs as the model.

## 1. Load at baseline volumes

| Resource | Need/yr | Available/yr | Load ρ |
|---|---:|---:|---:|
| engineering | 610 eng-wk | 864 eng-wk | 71% |
| production | 17,726 h | 20,854 h | 85% |
| FAT | 6,811 h | 7,404 h | 92% |

## 2. Minimum lead time without queues (weeks)

| Class | Case-by-case | Type approved | Type approved + sensor stock |
|---|---:|---:|---:|
| ptc_simple | 19.3 | 12.8 | 12.8 |
| cms | 19.3 | 12.8 | 12.8 |
| ptc_medium | 20.6 | 14.1 | 14.1 |
| b5_medium | 20.6 | 14.1 | 14.1 |
| b5_dp2 | 20.9 | 16.0 | 14.4 |

Express limit: 17 weeks. Comments add on average 2.0 weeks (case-by-case).

## 3. Queue factor ρ/(1−ρ)

| ρ | 0.80 | 0.85 | 0.90 | 0.92 | 0.95 |
|---|---|---|---|---|---|
| ρ/(1−ρ) | 4.0 | 5.7 | 9.0 | 11.5 | 19.0 |

## 4. Contribution per project (kNOK)

| Class | Price | Traded sensors (cost) | Contribution | Express extra |
|---|---:|---:|---:|---:|
| ptc_simple | 300 | 0 | 60 | 45 |
| cms | 300 | 0 | 60 | 45 |
| ptc_medium | 1,000 | 0 | 200 | 150 |
| b5_medium | 1,000 | 0 | 200 | 150 |
| b5_dp2 | 10,000 | 6,000 | 2,000 | 1,500 |

## 7. Rework: share of production/FAT hours and resulting hours factor

| | Year 1 | Year 2 | Year 3 | Year 4 | Year 5 |
|---|---:|---:|---:|---:|---:|
| Rework share (mid-year) | 23% | 14% | 9% | 7% | 6% |
| Hours factor from rework | 0.91 | 0.81 | 0.77 | 0.75 | 0.75 |

Rework today 30%, half-life 52 weeks, floor 5%.

## 8. Learning + rework: production hours factor per family and year (baseline volume)

| Family | Experience today [simple-eq.] | Year 1 | Year 2 | Year 3 | Year 4 | Year 5 |
|---|---:|---:|---:|---:|---:|---:|
| PTC | 205 | 0.86 | 0.72 | 0.65 | 0.60 | 0.58 |
| CMS | 26 | 0.85 | 0.70 | 0.63 | 0.58 | 0.55 |
| BruCon-5 | 78 | 0.86 | 0.72 | 0.64 | 0.60 | 0.57 |

Learning rate 85% per doubling, transfer between families 30%, floor 50%. Experience from deliveries since 2023 (volume doubled to 2026).

Today's queue (anchor): quoted ptc_medium lead time 29.5 wk − L_min 20.6 − safety 1 − rounding 0.5 = **7.4 weeks** of queue.

Retrofit opportunities by hand (today's wins / p(today's lead time)): ptc_simple 38, cms 23, ptc_medium 24, b5_medium 13, b5_dp2 3 = 101 per year.

## 5–6. Scenarios: hand calculation vs model

| Scenario | FAT load ρ (retrofit part), 5-yr avg. | Queue [wk] hand | Quote anchor class [wk] hand / model | Retrofit won/yr hand / model | Express/yr hand / model | ΔNPV [MNOK] hand / model | Ratio | Step = lever value [MNOK] hand / model | Check |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 baseline | 75% (38%) | 2.9 | 24 / 23 | 63.0 / 57.1 | 2.8 / 0.0 | 0.0 / 0.0 | – | – | – |
| 2 +pull bundled fwd | 77% (40%) | 1.1 | 22 / 22 | 66.7 / 60.0 | 6.6 / 0.0 | 6.4 / 3.0 | 2.14 | 6.4 / 3.0 | **CHECK** |
| 3 +TA DNV | 78% (41%) | 1.1 | 18 / 16 | 68.7 / 62.1 | 38.2 / 39.6 | 19.2 / 17.9 | 1.07 | 12.7 / 14.9 | OK |
| 4 +1 sensor stock set | 79% (41%) | 1.1 | 18 / 16 | 68.7 / 62.1 | 38.5 / 39.9 | 17.0 / 15.3 | 1.11 | -2.1 / -2.6 | OK |
| 5 +TA ABS | 79% (41%) | 1.1 | 17 / 16 | 69.0 / 62.3 | 42.5 / 44.8 | 16.6 / 15.1 | 1.10 | -0.5 / -0.2 | OK |
| 5b +10% prod. (not FAT) | 79% (42%) | 1.1 | 17 / 16 | 69.0 / 62.3 | 42.8 / 44.8 | 9.9 / 7.9 | 1.26 | -6.7 / -7.2 (vs 5) | OK |
| 6 +25% FAT capacity | 62% (33%) | 0.9 | 17 / 16 | 69.1 / 62.3 | 43.5 / 45.1 | 9.2 / 7.1 | 1.30 | -7.4 / -8.0 | OK |
| 7 +FAT +production | 62% (33%) | 0.9 | 17 / 16 | 69.2 / 62.3 | 44.0 / 45.1 | 2.7 / -0.1 | – | -6.5 / -7.2 | OK |

Check: each lever's own value (step vs previous) by hand and by model differ by at most 3 MNOK or 50% of the model value.
Steps outside: 1.

Hand-check values are averages over the 5 years. Known simplifications: average lead time per order type (spread only assumed as
±3 weeks for express share), queue anchored on today's quote and scaled, no late-delivery penalties (LDs), no approval
comments in the quote (as in the model), type approval from year 2.
