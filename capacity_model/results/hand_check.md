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
| PTC | 239 | 0.87 | 0.73 | 0.66 | 0.62 | 0.59 |
| CMS | 72 | 0.87 | 0.73 | 0.66 | 0.62 | 0.59 |
| BruCon-5 | 93 | 0.87 | 0.73 | 0.66 | 0.62 | 0.59 |

Learning rate 85% per doubling, transfer between families 30%, floor 50%. Experience from deliveries since 2023 (volume doubled to 2026).

Today's queue (anchor): quoted ptc_medium lead time 29.5 wk − L_min 20.6 − safety 1 − rounding 0.5 = **7.4 weeks** of queue.

Retrofit opportunities by hand (today's wins / p(today's lead time)): ptc_simple 38, cms 23, ptc_medium 24, b5_medium 13, b5_dp2 3 = 101 per year.

## 5–6. Scenarios: hand calculation vs model

| Scenario | FAT load ρ (retrofit part), 5-yr avg. | Queue [wk] hand | Quote anchor class [wk] hand / model | Retrofit won/yr hand / model | Express/yr hand / model | ΔNPV [MNOK] hand / model | Ratio | Step = lever value [MNOK] hand / model | Check |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 baseline | 77% (39%) | 2.8 | 24 / 23 | 63.3 / 57.3 | 3.5 / 0.0 | 0.0 / 0.0 | – | – | – |
| 2 +pull bundled fwd | 79% (41%) | 1.1 | 22 / 22 | 66.8 / 60.0 | 7.5 / 0.0 | 6.4 / 3.0 | – | 6.4 / 3.0 | **CHECK** |
| 3 +TA DNV | 81% (42%) | 1.1 | 18 / 16 | 68.7 / 62.1 | 39.2 / 39.5 | 18.8 / 17.6 | 1.07 | 12.4 / 14.6 | OK |
| 4 +1 sensor stock set | 81% (42%) | 1.1 | 18 / 16 | 68.7 / 62.1 | 39.5 / 39.8 | 17.1 / 15.1 | 1.13 | -1.8 / -2.5 | OK |
| 5 +TA ABS | 81% (43%) | 1.1 | 17 / 16 | 68.9 / 62.3 | 43.5 / 44.7 | 16.6 / 14.8 | 1.12 | -0.4 / -0.2 | OK |
| 5b +10% prod. (not FAT) | 81% (43%) | 1.0 | 17 / 16 | 68.9 / 62.3 | 43.9 / 44.7 | 10.0 / 7.6 | 1.30 | -6.7 / -7.2 (vs 5) | OK |
| 6 +25% FAT capacity | 64% (34%) | 0.9 | 17 / 16 | 69.2 / 62.4 | 44.6 / 45.0 | 9.4 / 6.9 | 1.36 | -7.3 / -7.9 | OK |
| 7 +FAT +production | 64% (34%) | 0.8 | 17 / 16 | 69.2 / 62.4 | 45.2 / 45.0 | 2.9 / -0.3 | – | -6.5 / -7.2 | OK |

Check: each lever's own value (step vs previous) by hand and by model differ by at most 3 MNOK or 50% of the model value.
Steps outside: 1.

Hand-check values are averages over the 5 years. Known simplifications: average lead time per order type (spread only assumed as
±3 weeks for express share), queue anchored on today's quote and scaled, no late-delivery penalties (LDs), no approval
comments in the quote (as in the model), type approval from year 2.
