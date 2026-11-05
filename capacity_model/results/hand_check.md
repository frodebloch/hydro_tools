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

## 7. Learning: hours factor per family and year (baseline volume)

| Family | Experience today [simple-eq.] | Year 1 | Year 2 | Year 3 | Year 4 | Year 5 |
|---|---:|---:|---:|---:|---:|---:|
| PTC | 239 | 0.97 | 0.94 | 0.91 | 0.88 | 0.86 |
| CMS | 72 | 0.97 | 0.94 | 0.91 | 0.88 | 0.86 |
| BruCon-5 | 93 | 0.97 | 0.94 | 0.91 | 0.88 | 0.86 |

Learning rate 90% per doubling, transfer between families 30%, floor 50%. Experience from deliveries since 2023 (volume doubled to 2026).

Today's queue (anchor): quoted ptc_medium lead time 29.5 wk − L_min 21.3 − safety 1 − rounding 0.5 = **6.7 weeks** of queue.

Retrofit opportunities by hand (today's wins / p(today's lead time)): ptc_simple 36, cms 22, ptc_medium 24, b5_medium 13, b5_dp2 3 = 98 per year.

## 5–6. Scenarios: hand calculation vs model

| Scenario | FAT load ρ (retrofit part), 5-yr avg. | Queue [wk] hand | Quote anchor class [wk] hand / model | Retrofit won/yr hand / model | Express/yr hand / model | ΔNPV [MNOK] hand / model | Ratio | Step = lever value [MNOK] hand / model | Check |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 baseline | 90% (40%) | 5.8 | 28 / 27 | 50.0 / 52.2 | 0.1 / 0.0 | 0.0 / 0.0 | – | – | – |
| 2 +pull bundled fwd | 98% (48%) | 1.4 | 24 / 23 | 58.6 / 63.3 | 2.8 / 0.0 | 13.1 / 11.8 | 1.11 | 13.1 / 11.8 | OK |
| 3 +TA DNV | 98% (48%) | 1.4 | 19 / 19 | 58.8 / 66.3 | 26.9 / 31.2 | 18.9 / 19.4 | 0.98 | 5.9 / 7.6 | OK |
| 4 +1 sensor stock set | 98% (48%) | 1.4 | 19 / 19 | 58.7 / 66.3 | 27.3 / 31.3 | 18.5 / 15.9 | 1.16 | -0.4 / -3.4 | OK |
| 5 +TA ABS | 98% (48%) | 1.4 | 19 / 19 | 58.7 / 66.6 | 30.2 / 34.8 | 17.3 / 14.5 | 1.19 | -1.2 / -1.4 | OK |
| 5b +10% prod. (not FAT) | 98% (48%) | 1.3 | 19 / 19 | 58.7 / 66.6 | 30.8 / 34.9 | 5.3 / 1.4 | – | -11.9 / -13.1 (vs 5) | OK |
| 6 +25% FAT capacity | 83% (43%) | 1.3 | 19 / 18 | 65.8 / 67.2 | 34.4 / 40.4 | 16.4 / 11.4 | 1.43 | -0.9 / -3.1 | OK |
| 7 +FAT +production | 83% (43%) | 1.2 | 19 / 18 | 66.7 / 67.2 | 35.6 / 40.4 | 5.5 / -1.4 | – | -10.9 / -12.8 | OK |

Check: each lever's own value (step vs previous) by hand and by model differ by at most 3 MNOK or 50% of the model value.
Steps outside: 0.

Hand-check values are averages over the 5 years. Known simplifications: average lead time per order type (spread only assumed as
±3 weeks for express share), queue anchored on today's quote and scaled, no late-delivery penalties (LDs), no approval
comments in the quote (as in the model), type approval from year 2.
