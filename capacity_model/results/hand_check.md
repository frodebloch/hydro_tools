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
| ptc_simple | 19.7 | 13.2 | 13.2 |
| cms | 19.7 | 13.2 | 13.2 |
| ptc_medium | 21.3 | 14.8 | 14.8 |
| b5_medium | 21.3 | 14.8 | 14.8 |
| b5_dp2 | 21.6 | 18.1 | 15.1 |

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
| PTC | 239 | 0.87 | 0.73 | 0.66 | 0.62 | 0.59 |
| CMS | 72 | 0.87 | 0.73 | 0.66 | 0.62 | 0.59 |
| BruCon-5 | 93 | 0.87 | 0.73 | 0.66 | 0.62 | 0.59 |

Learning rate 85% per doubling, transfer between families 30%, floor 50%. Experience from deliveries since 2023 (volume doubled to 2026).

Today's queue (anchor): quoted ptc_medium lead time 29.5 wk − L_min 21.3 − safety 1 − rounding 0.5 = **6.7 weeks** of queue.

Retrofit opportunities by hand (today's wins / p(today's lead time)): ptc_simple 36, cms 22, ptc_medium 24, b5_medium 13, b5_dp2 3 = 98 per year.

## 5–6. Scenarios: hand calculation vs model

| Scenario | FAT load ρ (retrofit part), 5-yr avg. | Queue [wk] hand | Quote anchor class [wk] hand / model | Retrofit won/yr hand / model | Express/yr hand / model | ΔNPV [MNOK] hand / model | Ratio | Step = lever value [MNOK] hand / model | Check |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 baseline | 76% (38%) | 2.4 | 24 / 23 | 61.6 / 58.5 | 3.3 / 0.0 | 0.0 / 0.0 | – | – | – |
| 2 +pull bundled fwd | 78% (40%) | 0.9 | 22 / 23 | 64.7 / 61.3 | 6.4 / 0.0 | 5.5 / 3.1 | 1.78 | 5.5 / 3.1 | OK |
| 3 +TA DNV | 80% (41%) | 1.0 | 18 / 17 | 66.7 / 63.7 | 36.7 / 39.3 | 16.3 / 16.0 | 1.02 | 10.8 / 12.9 | OK |
| 4 +1 sensor stock set | 80% (41%) | 1.0 | 18 / 17 | 66.7 / 63.7 | 37.2 / 39.8 | 15.8 / 14.4 | 1.09 | -0.5 / -1.6 | OK |
| 5 +TA ABS | 80% (42%) | 1.0 | 18 / 17 | 67.0 / 64.0 | 41.0 / 44.7 | 15.3 / 13.9 | 1.09 | -0.5 / -0.4 | OK |
| 5b +10% prod. (not FAT) | 80% (42%) | 0.9 | 18 / 17 | 67.0 / 64.0 | 41.3 / 44.7 | 3.2 / 1.1 | – | -12.1 / -12.8 (vs 5) | OK |
| 6 +25% FAT capacity | 63% (33%) | 0.8 | 17 / 17 | 67.1 / 64.0 | 42.1 / 45.0 | 7.9 / 5.9 | 1.32 | -7.4 / -8.0 | OK |
| 7 +FAT +production | 63% (33%) | 0.7 | 17 / 17 | 67.2 / 64.0 | 42.5 / 45.0 | -4.1 / -6.9 | 0.59 | -12.0 / -12.8 | OK |

Check: each lever's own value (step vs previous) by hand and by model differ by at most 3 MNOK or 50% of the model value.
Steps outside: 0.

Hand-check values are averages over the 5 years. Known simplifications: average lead time per order type (spread only assumed as
±3 weeks for express share), queue anchored on today's quote and scaled, no late-delivery penalties (LDs), no approval
comments in the quote (as in the model), type approval from year 2.
