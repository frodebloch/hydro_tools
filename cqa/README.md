# cqa — Combined Capability + Excursion Analysis (prototype)

See `analysis.md` for the full feasibility study.

This Python package is the P1 prototype: linearised closed-loop covariance
prediction of vessel position/heading excursion under DP, used to build an
**excursion polar** vs. relative weather direction. Intact state only in P1.

## Layout

```
cqa/
  analysis.md            # feasibility study (read this first)
  cqa/                   # python package
    config.py            # CSOV-like configuration container
    vessel.py            # 3-DOF linearised vessel model + force models
    controller.py        # PD + integral DP controller approximation
    closed_loop.py       # build A_cl, B_w, solve Lyapunov
    psd.py               # environmental force PSDs (wind gust, slow drift, current)
    excursion.py         # excursion polar over relative weather direction
  scripts/
    run_polar_demo.py    # end-to-end demo: CSOV polar plot
  tests/
```

## Quick start

```bash
pip install -e .
python scripts/run_polar_demo.py
```
