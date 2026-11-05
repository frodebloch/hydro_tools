# Scenario results (200 Monte Carlo runs, means)

Retrofit opportunities calibrated so the baseline wins 20 per type per year: single 37.5, quad 43.3, package 45.8 per year.

| Metric | 0 bundled asap (ref.) | 1 baseline | 1b quote incl. comments | 2 +pull bundled fwd | 2b +adv. stock, no TA | 3 +TA DNV | 4 +1 adv. stock set | 5 +TA ABS | 5b +10% prod. (not FAT) | 6 +25% FAT capacity | 7 +FAT +production |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Retrofit opportunities/yr | 127 | 127 | 127 | 127 | 127 | 127 | 127 | 127 | 127 | 127 | 127 |
| Retrofit won/yr | 78.4 | 78.6 | 69.1 | 82.8 | 82.8 | 85.3 | 85.4 | 85.7 | 85.7 | 86.2 | 86.2 |
|   express (≤4 mo) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 36.5 | 38.5 | 43.3 | 43.5 | 45.6 | 45.7 |
|   standard (≤6 mo) | 72.4 | 73.3 | 41.1 | 82.4 | 82.4 | 48.5 | 46.5 | 42.0 | 41.9 | 40.6 | 40.5 |
|   long (>6 mo) | 5.9 | 5.2 | 28.0 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.0 | 0.0 |
| Median quote single [wk] | 20 | 20 | 22 | 18 | 18 | 12 | 12 | 12 | 12 | 12 | 12 |
| Median quote quad [wk] | 23 | 23 | 26 | 22 | 22 | 16 | 16 | 16 | 16 | 16 | 16 |
| Median quote package [wk] | 23 | 24 | 27 | 23 | 23 | 20 | 20 | 20 | 20 | 20 | 20 |
| Retrofit late vs quote | 9% | 10% | 0% | 12% | 12% | 9% | 9% | 9% | 9% | 9% | 9% |
| Orders with FAT rebooked | 10% | 6% | 0% | 6% | 6% | 5% | 5% | 5% | 5% | 5% | 5% |
| Bundled late (≥1 wk) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| LDs retrofit [kNOK/yr] | 2082 | 1955 | 0 | 1301 | 1302 | 911 | 933 | 889 | 893 | 573 | 571 |
| LDs bundled [kNOK/yr] | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Hours per unit after 5 yr (learning) | 0.65 | 0.65 | 0.66 | 0.64 | 0.64 | 0.64 | 0.64 | 0.64 | 0.64 | 0.64 | 0.64 |
| Class comment rate after 5 yr | 7% | 7% | 7% | 7% | 7% | 7% | 7% | 7% | 7% | 7% | 7% |
| Engineering load | 64% | 63% | 58% | 65% | 65% | 67% | 67% | 68% | 68% | 68% | 68% |
| Production load | 75% | 73% | 67% | 78% | 78% | 80% | 80% | 80% | 72% | 81% | 72% |
| FAT load | 82% | 80% | 72% | 87% | 87% | 90% | 90% | 91% | 91% | 72% | 71% |
| **ΔNPV vs baseline [MNOK]** | **-1.0** | **0.0** | **-93.2** | **32.1** | **28.3** | **79.7** | **90.9** | **96.6** | **84.1** | **97.9** | **85.6** |
| ΔNPV P10 / P90 [MNOK] | -12.9 / 11.1 | 0.0 / 0.0 | -107.0 / -80.6 | 14.8 / 50.3 | 10.9 / 46.4 | 62.7 / 100.3 | 73.7 / 110.1 | 79.2 / 117.1 | 66.2 / 104.4 | 77.6 / 121.3 | 64.9 / 110.2 |
| ΔNPV vs previous step [MNOK] | – | – | -93.2 (vs 1) | 32.1 | -3.8 (vs 2) | 47.5 | 11.3 | 5.7 | -12.5 (vs 5) | 1.3 | -12.4 |

NPV over 5 years after a 2-year warm-up; contribution = 20 % margin + express premium / long-lead discount, minus Orgalim LDs and lever costs. Placeholder values: see config.py.
Main table includes learning (config.learning = True); retrofit opportunities are calibrated on today's situation without learning.

## Sensitivity: with vs without learning

| Scenario | ΔNPV with learning [MNOK] | ΔNPV without learning [MNOK] | Retrofit won/yr with / without |
|---|---:|---:|---:|
| 0 bundled asap (ref.) | -1.0 | -3.4 | 78.4 / 59.2 |
| 1 baseline | 0.0 | 0.0 | 78.6 / 60.6 |
| 1b quote incl. comments | -93.2 | -7.5 | 69.1 / 60.1 |
| 2 +pull bundled fwd | 32.1 | 67.6 | 82.8 / 73.0 |
| 2b +adv. stock, no TA | 28.3 | 63.5 | 82.8 / 73.0 |
| 3 +TA DNV | 79.7 | 85.1 | 85.3 / 73.5 |
| 4 +1 adv. stock set | 90.9 | 82.4 | 85.4 / 73.5 |
| 5 +TA ABS | 96.6 | 82.6 | 85.7 / 73.5 |
| 5b +10% prod. (not FAT) | 84.1 | 72.1 | 85.7 / 73.8 |
| 6 +25% FAT capacity | 97.9 | 172.5 | 86.2 / 84.7 |
| 7 +FAT +production | 85.6 | 176.4 | 86.2 / 86.0 |

Learning itself (baseline with vs without): 120.4 MNOK.
