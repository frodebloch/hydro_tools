"""What does it take to reach a retrofit turnover of 300 MNOK/yr on advanced units?

300 MNOK / 10 MNOK per advanced unit = 30 advanced (i.e. 30 packages of 8 simple + 1 advanced) per
year, reached by year 3 and held in years 3-5. Today: ~20 per year.

For each lever set the script finds the market growth (package opportunities, multiplier M reached
linearly by year 3) needed to win 30 packages per year in years 3-5, and reports at that point:
market, win rate, quoted lead time, load per resource, side effects and ΔNPV.

Usage:  .venv/bin/python target_advanced.py [n_runs] [--no-learning]
Output: results/target_advanced[_no_learning].md / .png
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import BASE, ORDER_TYPES, cs_price
from sim import simulate

OUT = Path(__file__).parent / "results"
TARGET = 30.0                      # advanced retrofit units per year (300 MNOK / 10 MNOK)
YEARS = (3, 4, 5)
GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]

PULL = replace(BASE, pull_forward=True)
TA = replace(PULL, type_approved_societies=("DNV",))
STOCK = replace(TA, advanced_stock_sets=2)
LEVERS = {
    "A baseline": BASE,
    "B pull bundled fwd": PULL,
    "C B + TA DNV": TA,
    "D C + 2 adv. stock sets": STOCK,
    "E D + 25% FAT": replace(STOCK, fat_capacity_factor=1.25),
    "F D + 10% production": replace(STOCK, prod_capacity_factor=1.10),
    "G D + FAT + production": replace(STOCK, fat_capacity_factor=1.25, prod_capacity_factor=1.10),
}


def avg_y(r, key):
    return float(np.mean([r[f"y{y}_{key}"] for y in YEARS]))


def run(cfg, M, rates, n_runs, seed0=500_000):
    c = replace(cfg, retrofit_ramp={"package": M})
    rr = [simulate(c, seed0 + i, rates) for i in range(n_runs)]
    out = {k: float(np.nanmean([r[k] for r in rr])) for k in rr[0]}
    out["pkg_won_y35"] = float(np.mean([avg_y(r, "won_package") for r in rr]))
    out["pkg_opps_y35"] = float(np.mean([avg_y(r, "opps_package") for r in rr]))
    out["pkg_quote_y35"] = float(np.nanmean([avg_y(r, "quote_package") for r in rr]))
    out["simple_won_y35"] = float(np.mean([avg_y(r, "won_single") + avg_y(r, "won_quad") for r in rr]))
    out["quad_quote_y35"] = float(np.nanmean([avg_y(r, "quote_quad") for r in rr]))
    for res in ("eng", "prod", "fat"):
        out[f"{res}_load_y35"] = float(np.mean([avg_y(r, f"{res}_load") for r in rr]))
    out["npv_runs"] = [r["npv"] for r in rr]
    return out


def solve_M(curve):
    """Linear interpolation of the market multiplier where packages won (years 3-5) = TARGET."""
    Ms = sorted(curve)
    ys = [curve[m]["pkg_won_y35"] for m in Ms]
    for (m0, y0), (m1, y1) in zip(zip(Ms, ys), zip(Ms[1:], ys[1:])):
        if y0 <= TARGET <= y1 and y1 > y0:
            return m0 + (TARGET - y0) * (m1 - m0) / (y1 - y0)
    return None if ys[-1] < TARGET else Ms[0]


def main(n_runs, learning=True):
    global LEVERS
    suffix = "" if learning else "_no_learning"
    LEVERS = {k: replace(v, learning=learning) for k, v in LEVERS.items()}
    rates = json.loads((OUT / "calibration.json").read_text())
    pkg_rev = cs_price(BASE, ORDER_TYPES["package"]) / 1000
    print(f"Target {TARGET:.0f} advanced/yr (= {TARGET * 10:.0f} MNOK advanced, "
          f"{TARGET * pkg_rev:.0f} MNOK incl. the simple units in the packages)")

    curves = {}
    for name, cfg in LEVERS.items():
        curves[name] = {M: run(cfg, M, rates, n_runs) for M in GRID}
        print(f"  {name:26s} " + "  ".join(f"M{M}: {curves[name][M]['pkg_won_y35']:.1f}" for M in GRID), flush=True)

    # ---- at the market each lever set needs
    at = {}
    for name, cfg in LEVERS.items():
        M = solve_M(curves[name])
        if M is not None:
            at[name] = (M, run(cfg, M, rates, n_runs))
    # ΔNPV vs baseline at the same market (common random numbers)
    base_at = {name: run(LEVERS["A baseline"], M, rates, n_runs) for name, (M, _) in at.items()}

    today_opps = rates["package"]
    L = ["# What it takes: 300 MNOK/yr retrofit on advanced units", "",
         f"Target: **{TARGET:.0f} advanced units (packages) per year in years 3–5** "
         f"(= {TARGET * 10:.0f} MNOK advanced; {TARGET * pkg_rev:.0f} MNOK incl. the 8 simple units per package). "
         f"Today: 20 per year from ~{today_opps:.0f} package opportunities.",
         "", f"Market growth: package opportunities × M, reached linearly by year 3. {n_runs} runs per point, "
         + ("with learning." if learning else "**without learning** (today's hours and comment rate).")
         + " Single / quad retrofit and bundled volumes unchanged.", "",
         "## Packages won per year (years 3–5) vs market growth", "",
         "| Lever set | " + " | ".join(f"M = {M}" for M in GRID) + " |",
         "|---|" + "---:|" * len(GRID)]
    for name in LEVERS:
        L.append(f"| {name} | " + " | ".join(f"{curves[name][M]['pkg_won_y35']:.1f}" for M in GRID) + " |")

    L += ["", "## At the market needed to reach the target", "",
          "| Lever set | Market M | Package opps/yr | Win rate | Package quote [wk] | Quad quote [wk] | "
          "Load eng / prod / FAT | Simple+quad retrofit won/yr | Bundled late | ΔNPV vs baseline at same market [MNOK] |",
          "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    ref_simple = curves["A baseline"][1.0]["simple_won_y35"]
    for name, (M, o) in at.items():
        b = base_at[name]
        dnpv = (np.array(o["npv_runs"]) - np.array(b["npv_runs"])).mean() / 1000
        flag = " ⚠" if o["pkg_quote_y35"] > BASE.standard_max_weeks else ""
        L.append(f"| {name} | {M:.2f} | {o['pkg_opps_y35']:.0f} | {o['pkg_won_y35'] / o['pkg_opps_y35']:.0%} | "
                 f"{o['pkg_quote_y35']:.0f}{flag} | {o['quad_quote_y35']:.0f} | "
                 f"{o['eng_load_y35']:.0%} / {o['prod_load_y35']:.0%} / {o['fat_load_y35']:.0%} | "
                 f"{o['simple_won_y35']:.0f} (today-level {ref_simple:.0f}) | {o['bundled_late_share_y3_5']:.1%} | "
                 f"{dnpv:.0f} |")
    missing = [n for n in LEVERS if n not in at]
    if missing:
        L.append(f"\nNot reached within M ≤ {GRID[-1]}: {', '.join(missing)}.")
    L += ["", "⚠ = quoted package lead time above 6 months (win rate falls steeply).",
          "ΔNPV compares each lever set with the baseline **at the same market**, so it is the value of the levers "
          "given that the market has been grown. The cost of growing the market (sales effort) is not included.", ""]
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
        ax[0].plot(Ms, [c[m]["pkg_won_y35"] for m in Ms], "o-", label=name)
        ax[1].plot(Ms, [c[m]["pkg_quote_y35"] for m in Ms], "o-", label=name)
        ax[2].plot(Ms, [max(c[m]["fat_load_y35"], c[m]["prod_load_y35"]) for m in Ms], "o-", label=name)
    ax[0].axhline(TARGET, color="k", ls="--")
    ax[0].text(GRID[0], TARGET + 0.5, "target 30/yr (300 MNOK advanced)")
    ax[0].set_ylabel("Packages won per year, years 3–5")
    ax[1].axhline(26, color="grey", ls="--")
    ax[1].text(GRID[0], 26.3, "6 months", color="grey")
    ax[1].set_ylabel("Median quoted package lead time, years 3–5 [wk]")
    ax[2].set_ylabel("Highest of production / FAT load, years 3–5")
    for a in ax:
        a.set_xlabel("Package market multiplier M (reached by year 3)")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / f"target_advanced{suffix}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(int(args[0]) if args else 60, learning="--no-learning" not in sys.argv)
