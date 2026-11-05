"""What does it take to reach a retrofit turnover of 300 MNOK/yr on advanced (DP2) projects?

300 MNOK / 10 MNOK per DP2 = 30 retrofit DP2 projects per year, reached by year 3 and held in
years 3-5. Today: 3 DP2 projects per year in total, of which retrofit_share are retrofit.

For each lever set the script finds the market growth (DP2 retrofit opportunities x M, reached
linearly by year 3) needed to win 30 per year in years 3-5, and reports at that point:
market, win rate, quoted lead time, load per resource, side effects, sensor purchases and ΔNPV.

Usage:  .venv/bin/python target_advanced.py [n_runs] [--no-learning] [--dp2-weight=6]
Output: results/target_advanced[_no_learning][_w6].md / .png
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import BASE, ORDER_TYPES, cs_price, retrofit_won_target, traded_cost
from sim import simulate

OUT = Path(__file__).parent / "results"
T = "b5_dp2"
TARGET = 30.0                      # retrofit DP2 projects per year (300 MNOK / 10 MNOK)
YEARS = (3, 4, 5)
GRID = [1, 5, 10, 20, 30, 40]

PULL = replace(BASE, pull_forward=True)
TA = replace(PULL, type_approved_societies=("DNV",))
STOCK = replace(TA, advanced_stock_sets=4)
LEVERS = {
    "A baseline": BASE,
    "B pull bundled fwd": PULL,
    "C B + TA DNV": TA,
    "D C + 4 sensor stock sets": STOCK,
    "E D + 25% FAT": replace(STOCK, fat_capacity_factor=1.25),
    "F D + 25% production": replace(STOCK, prod_capacity_factor=1.25),
    "G D + 25% FAT + 25% production": replace(STOCK, fat_capacity_factor=1.25, prod_capacity_factor=1.25),
    "H G + 50% FAT + 50% production": replace(STOCK, fat_capacity_factor=1.5, prod_capacity_factor=1.5),
    "I G + 25% engineers": replace(STOCK, fat_capacity_factor=1.25, prod_capacity_factor=1.25, eng_capacity_factor=1.25),
    "J H + 25% engineers": replace(STOCK, fat_capacity_factor=1.5, prod_capacity_factor=1.5, eng_capacity_factor=1.25),
    "K H + 50% engineers": replace(STOCK, fat_capacity_factor=1.5, prod_capacity_factor=1.5, eng_capacity_factor=1.5),
}


def avg_y(r, key):
    return float(np.mean([r[f"y{y}_{key}"] for y in YEARS]))


def run(cfg, M, rates, n_runs, seed0=500_000):
    c = replace(cfg, retrofit_ramp={T: M})
    rr = [simulate(c, seed0 + i, rates) for i in range(n_runs)]
    out = {k: float(np.nanmean([r[k] for r in rr])) for k in rr[0]}
    out["won_y35"] = float(np.mean([avg_y(r, f"won_{T}") for r in rr]))
    out["opps_y35"] = float(np.mean([avg_y(r, f"opps_{T}") for r in rr]))
    out["quote_y35"] = float(np.nanmean([avg_y(r, f"quote_{T}") for r in rr]))
    out["other_won_y35"] = float(np.mean([sum(avg_y(r, f"won_{o}") for o in ORDER_TYPES if o != T) for r in rr]))
    out["medium_quote_y35"] = float(np.nanmean([avg_y(r, "quote_ptc_medium") for r in rr]))
    for res in ("eng", "prod", "fat"):
        out[f"{res}_load_y35"] = float(np.mean([avg_y(r, f"{res}_load") for r in rr]))
    out["npv_runs"] = [r["npv"] for r in rr]
    return out


def solve_M(curve):
    """Linear interpolation of the market multiplier where DP2 won (years 3-5) = TARGET."""
    Ms = sorted(curve)
    ys = [curve[m]["won_y35"] for m in Ms]
    for (m0, y0), (m1, y1) in zip(zip(Ms, ys), zip(Ms[1:], ys[1:])):
        if y0 <= TARGET <= y1 and y1 > y0:
            return m0 + (TARGET - y0) * (m1 - m0) / (y1 - y0)
    return None if ys[-1] < TARGET else Ms[0]


def main(n_runs, learning=True, dp2_weight=None):
    global LEVERS
    suffix = ("" if learning else "_no_learning") + (f"_w{dp2_weight:g}" if dp2_weight else "")
    over = dict(learning=learning)
    if dp2_weight:
        over["class_weight"] = {**BASE.class_weight, "advanced": dp2_weight}
    LEVERS = {k: replace(v, **over) for k, v in LEVERS.items()}
    base = LEVERS["A baseline"]
    rates = json.loads((OUT / "calibration.json").read_text())
    ot = ORDER_TYPES[T]
    price = cs_price(base, ot) / 1000
    print(f"Target {TARGET:.0f} retrofit DP2/yr = {TARGET * price:.0f} MNOK  ({suffix or 'base case'})")

    curves = {}
    for name, cfg in LEVERS.items():
        curves[name] = {M: run(cfg, M, rates, n_runs) for M in GRID}
        print(f"  {name:32s} " + "  ".join(f"M{M}: {curves[name][M]['won_y35']:.1f}" for M in GRID), flush=True)

    at = {}
    for name, cfg in LEVERS.items():
        M = solve_M(curves[name])
        if M is not None:
            at[name] = (M, run(cfg, M, rates, n_runs))
    base_at = {name: run(base, M, rates, n_runs) for name, (M, _) in at.items()}

    today = retrofit_won_target(base)[T]
    w = base.class_weight[ORDER_TYPES[T].cls]
    sensors = TARGET * traded_cost(base, ot) / 1000
    L = ["# What it takes: 300 MNOK/yr retrofit on advanced (DP2) projects", "",
         f"Target: **{TARGET:.0f} retrofit DP2 projects per year in years 3–5** (= {TARGET * price:.0f} MNOK). "
         f"Today: {base.projects_per_year[T]} DP2 per year in total, ≈ {today:.1f} retrofit "
         f"(retrofit share {base.retrofit_share[T]:.0%}, placeholder), from ≈ {rates[T]:.1f} opportunities.", "",
         f"Assumptions: DP2 = {w:g} × a simple project in own hours; traded sensors "
         f"{base.traded_share[ORDER_TYPES[T].cls]:.0%} of the price ({sensors:.0f} MNOK/yr of sensor purchases at the target), "
         f"{base.sensor_hw_weeks[0]:.0f}–{base.sensor_hw_weeks[1]:.0f} weeks lead time. "
         + ("With learning." if learning else "**Without learning.**")
         + f" DP2 market × M reached linearly by year 3; other volumes unchanged. {n_runs} runs per point.", "",
         "## Retrofit DP2 won per year (years 3–5) vs market growth", "",
         "| Lever set | " + " | ".join(f"M = {M}" for M in GRID) + " |",
         "|---|" + "---:|" * len(GRID)]
    for name in LEVERS:
        L.append(f"| {name} | " + " | ".join(f"{curves[name][M]['won_y35']:.1f}" for M in GRID) + " |")

    L += ["", "## At the market needed to reach the target", "",
          "| Lever set | Market M | DP2 opps/yr | Win rate | DP2 quote [wk] | PTC medium quote [wk] | "
          "Load eng / prod / FAT | Other retrofit won/yr | Bundled late | ΔNPV vs baseline at same market [MNOK] |",
          "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    ref_other = curves["A baseline"][GRID[0]]["other_won_y35"]
    for name, (M, o) in at.items():
        b = base_at[name]
        dnpv = (np.array(o["npv_runs"]) - np.array(b["npv_runs"])).mean() / 1000
        flag = " ⚠" if o["quote_y35"] > base.standard_max_weeks else ""
        L.append(f"| {name} | {M:.1f} | {o['opps_y35']:.0f} | {o['won_y35'] / o['opps_y35']:.0%} | "
                 f"{o['quote_y35']:.0f}{flag} | {o['medium_quote_y35']:.0f} | "
                 f"{o['eng_load_y35']:.0%} / {o['prod_load_y35']:.0%} / {o['fat_load_y35']:.0%} | "
                 f"{o['other_won_y35']:.0f} (at M=1: {ref_other:.0f}) | {o['bundled_late_share_y3_5']:.1%} | {dnpv:.0f} |")
    missing = [n for n in LEVERS if n not in at]
    if missing:
        L.append(f"\nNot reached within M ≤ {GRID[-1]}: {', '.join(missing)}.")
    L += ["", "⚠ = quoted DP2 lead time above 6 months (win rate falls steeply).",
          "ΔNPV compares each lever set with the baseline **at the same market**: the value of the levers given that "
          "the market has been grown. The cost of growing the market (sales effort) is not included.", ""]
    (OUT / f"target_advanced{suffix}.md").write_text("\n".join(L))
    print("\n".join(L))
    plot(curves, suffix)


def plot(curves, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for name, c in curves.items():
        Ms = sorted(c)
        ax[0].plot(Ms, [c[m]["won_y35"] for m in Ms], "o-", label=name)
        ax[1].plot(Ms, [c[m]["quote_y35"] for m in Ms], "o-", label=name)
        ax[2].plot(Ms, [max(c[m]["fat_load_y35"], c[m]["prod_load_y35"]) for m in Ms], "o-", label=name)
    ax[0].axhline(TARGET, color="k", ls="--")
    ax[0].text(GRID[0], TARGET + 0.5, "target 30 DP2/yr (300 MNOK)")
    ax[0].set_ylabel("Retrofit DP2 won per year, years 3–5")
    ax[1].axhline(26, color="grey", ls="--")
    ax[1].text(GRID[0], 26.3, "6 months", color="grey")
    ax[1].set_ylabel("Median quoted DP2 lead time, years 3–5 [wk]")
    ax[2].set_ylabel("Highest of production / FAT load, years 3–5")
    for a in ax:
        a.set_xlabel("DP2 retrofit market multiplier M (reached by year 3)")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / f"target_advanced{suffix}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    wt = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--dp2-weight=")), None)
    main(int(args[0]) if args else 60, learning="--no-learning" not in sys.argv, dp2_weight=wt)
