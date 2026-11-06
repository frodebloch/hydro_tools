"""One-at-a-time sensitivities on rework and learning assumptions.

For each variant: baseline and key lever sets, with retrofit won, quote, loads and ΔNPV of the levers
(vs the baseline of the same variant). Opportunities from results/calibration.json (today's market).

Usage:  .venv/bin/python sensitivity.py [n_runs]
Output: results/sensitivity.md
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import SCENARIOS
from sim import simulate

OUT = Path(__file__).parent / "results"
LEVERS = ["1 baseline", "2 +pull bundled fwd", "3 +TA DNV", "6 +25% FAT capacity"]
LR = lambda x: {"learning_rates": {"eng": x, "prod": x, "fat": x}}
VARIANTS = {
    "Base: rework 30 %, half-life 1 yr, learning 85 %": {},
    "Rework 20 % today": {"rework_share": 0.20},
    "Rework 40 % today": {"rework_share": 0.40},
    "Rework half-life 6 months": {"rework_halflife_weeks": 26.0},
    "Rework half-life 2 years": {"rework_halflife_weeks": 104.0},
    "No rework reduction": {"rework_halflife_weeks": None},
    "Learning 80 %": LR(0.80),
    "Learning 90 %": LR(0.90),
    "No learning, no rework reduction": {"learning": False},
}


def main(n):
    rates = json.loads((OUT / "calibration.json").read_text())
    L = ["# Sensitivity: rework and learning", "",
         f"{n} runs per point, today's market. ΔNPV of each lever set vs the baseline **of the same variant** "
         "(5 years, MNOK). Baseline NPV change vs the base case shows the value of the assumption itself.", "",
         "| Variant | Baseline NPV vs base case | Retrofit won/yr (baseline) | Quote PTC medium [wk] | "
         "Load prod / FAT, baseline | Pull fwd | + TA DNV | + 25 % FAT | FAT load with TA |",
         "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    ref = None
    for vname, over in VARIANTS.items():
        res = {}
        for lv in LEVERS:
            cfg = replace(SCENARIOS[lv], **over)
            rr = [simulate(cfg, 300_000 + i, rates) for i in range(n)]
            res[lv] = {k: float(np.nanmean([r[k] for r in rr])) for k in
                       ("npv", "retrofit_won", "quote_ptc_medium", "prod_load", "fat_load")}
        b = res["1 baseline"]
        ref = b["npv"] if ref is None else ref
        d = lambda lv: (res[lv]["npv"] - b["npv"]) / 1000
        L.append(f"| {vname} | {(b['npv'] - ref) / 1000:+.0f} | {b['retrofit_won']:.0f} | {b['quote_ptc_medium']:.0f} | "
                 f"{b['prod_load']:.0%} / {b['fat_load']:.0%} | {d('2 +pull bundled fwd'):+.0f} | {d('3 +TA DNV'):+.0f} | "
                 f"{d('6 +25% FAT capacity'):+.0f} | {res['3 +TA DNV']['fat_load']:.0%} |")
        print(L[-1], flush=True)
    L += ["", "Lever columns are cumulative: '+ TA DNV' includes pull forward; '+ 25 % FAT' includes pull forward, "
          "TA DNV/ABS and 1 sensor stock set (scenario 6).", ""]
    (OUT / "sensitivity.md").write_text("\n".join(L))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
