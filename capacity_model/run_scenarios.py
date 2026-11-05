"""Calibrate, run all lever scenarios with common random numbers, write results.

Usage:  .venv/bin/python run_scenarios.py [n_runs]
Output: results/summary.md, results/scenarios.png, results/demand_sweep.png
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import FAMILIES, ORDER_TYPES, SCENARIOS
import hand_check
from sim import calibrate_retrofit, simulate

OUT = Path(__file__).parent / "results"
BASELINE = "1 baseline"

COLUMNS = [
    ("retrofit_opps", "Retrofit opportunities/yr", "{:.0f}"),
    ("retrofit_won", "Retrofit won/yr", "{:.1f}"),
    ("won_express", "  express (≤4 mo)", "{:.1f}"),
    ("won_standard", "  standard (≤6 mo)", "{:.1f}"),
    ("won_long", "  long (>6 mo)", "{:.1f}"),
    *[(f"won_{t}", f"  {t} won/yr", "{:.1f}") for t in ORDER_TYPES],
    *[(f"quote_{t}", f"Median quote {t} [wk]", "{:.0f}") for t in ORDER_TYPES],
    ("revenue", "Turnover, all projects [kNOK/yr]", "{:,.0f}"),
    ("wc_cost", "Working capital, traded sensors [kNOK/yr]", "{:.0f}"),
    ("retrofit_late_share", "Retrofit late vs quote", "{:.0%}"),
    ("fat_rebook_share", "Orders with FAT rebooked", "{:.0%}"),
    ("bundled_late_share", "Bundled late (≥1 wk)", "{:.1%}"),
    ("ld_retrofit", "LDs retrofit [kNOK/yr]", "{:.0f}"),
    ("ld_bundled", "LDs bundled [kNOK/yr]", "{:.0f}"),
    ("hours_factor_end", "Hours per project after 5 yr (learning, avg.)", "{:.2f}"),
    *[(f"hours_factor_end_{f}", f"  {f}", "{:.2f}") for f in FAMILIES],
    ("comment_rate_end", "Class comment rate after 5 yr", "{:.0%}"),
    ("eng_load", "Engineering load", "{:.0%}"),
    ("prod_load", "Production load", "{:.0%}"),
    ("fat_load", "FAT load", "{:.0%}"),
]


def run_all(n_runs, rates, **override):
    res = {}
    for name, cfg in SCENARIOS.items():
        cfg = replace(cfg, **override)
        runs = [simulate(cfg, cfg.seed * 100000 + i, rates) for i in range(n_runs)]
        res[name] = {k: np.array([r[k] for r in runs]) for k in runs[0]}
        print(f"  {name:28s} done", flush=True)
    return res


def incremental(res, names):
    """ΔNPV along the main chain; side branches ('2b ...') vs their parent."""
    out, prev = [], None
    for n in names:
        tag = n.split()[0]
        if tag[-1].isalpha():
            parent = next(m for m in names if m.split()[0] == tag[:-1])
            out.append(f"{(res[n]['npv'] - res[parent]['npv']).mean() / 1000:.1f} (vs {tag[:-1]})")
            continue
        if prev is None or n == BASELINE:
            out.append("–")
        else:
            out.append(f"{(res[n]['npv'] - res[prev]['npv']).mean() / 1000:.1f}")
        prev = n if tag != "0" else None
    return out


def learning_table(res, res_nl):
    names = list(res)
    L = ["", "## Sensitivity: with vs without learning", "",
         "| Scenario | ΔNPV with learning [MNOK] | ΔNPV without learning [MNOK] | Retrofit won/yr with / without |",
         "|---|---:|---:|---:|"]
    b, bn = res[BASELINE]["npv"], res_nl[BASELINE]["npv"]
    for n in names:
        L.append(f"| {n} | {(res[n]['npv'] - b).mean() / 1000:.1f} | {(res_nl[n]['npv'] - bn).mean() / 1000:.1f} | "
                 f"{res[n]['retrofit_won'].mean():.1f} / {res_nl[n]['retrofit_won'].mean():.1f} |")
    L.append(f"\nLearning itself (baseline with vs without): "
             f"{(res[BASELINE]['npv'].mean() - res_nl[BASELINE]['npv'].mean()) / 1000:.1f} MNOK.")
    return L


def write_summary(res, n_runs, rates, extra=()):
    names = list(res)
    base = res[BASELINE]["npv"]
    L = [f"# Scenario results ({n_runs} Monte Carlo runs, means)", "",
         "Retrofit opportunities calibrated so the baseline (without learning) wins today's retrofit volume: "
         + ", ".join(f"{t} {v:.1f}" for t, v in rates.items()) + " per year.", ""]
    L.append("| Metric | " + " | ".join(names) + " |")
    L.append("|---|" + "---:|" * len(names))
    for key, label, fmt in COLUMNS:
        L.append(f"| {label} | " + " | ".join(fmt.format(np.nanmean(res[n][key])) for n in names) + " |")
    d = {n: (res[n]["npv"] - base) / 1000 for n in names}
    L.append("| **ΔNPV vs baseline [MNOK]** | " + " | ".join(f"**{d[n].mean():.1f}**" for n in names) + " |")
    L.append("| ΔNPV P10 / P90 [MNOK] | " + " | ".join(
        f"{np.percentile(d[n], 10):.1f} / {np.percentile(d[n], 90):.1f}" for n in names) + " |")
    L.append("| ΔNPV vs previous step [MNOK] | " + " | ".join(incremental(res, names)) + " |")
    L += ["", "NPV over 5 years after a 2-year warm-up; contribution = 20 % margin + express premium / "
          "long-lead discount, minus Orgalim LDs and lever costs. Placeholder values: see config.py.",
          "Main table includes learning (config.learning = True); retrofit opportunities are calibrated on "
          "today's situation without learning."]
    L += list(extra) + [""]
    (OUT / "summary.md").write_text("\n".join(L))
    means = {n: {k: float(np.nanmean(v)) for k, v in res[n].items()} for n in names}
    for n in names:
        means[n]["dnpv"] = float(d[n].mean())
    (OUT / "model_means.json").write_text(json.dumps(means, indent=1))
    print("\n".join(L))


def plots(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(res)
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    bottom = np.zeros(len(names))
    for key, lab, col in [("won_express", "express ≤4 mo", "tab:green"),
                          ("won_standard", "standard ≤6 mo", "tab:blue"),
                          ("won_long", "long >6 mo", "tab:orange")]:
        v = np.array([res[n][key].mean() for n in names])
        ax[0].bar(x, v, bottom=bottom, label=lab, color=col)
        bottom += v
    ax[0].set_xticks(x, names, rotation=30, ha="right")
    ax[0].set_ylabel("Retrofit orders won per year")
    ax[0].legend(loc="upper left")
    ax[0].set_ylim(0, 105)
    ax[0].set_title("Retrofit wins by quoted lead time")

    base = res[BASELINE]["npv"]
    others = [n for n in names if n != BASELINE]
    ax[1].boxplot([(res[n]["npv"] - base) / 1000 for n in others], whis=(10, 90), showfliers=False)
    ax[1].set_xticks(range(1, len(others) + 1), others, rotation=30, ha="right")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("ΔNPV vs baseline [MNOK]\n(box P25–P75, whiskers P10–P90)")
    ax[1].set_title("Value of levers")
    fig.tight_layout()
    fig.savefig(OUT / "scenarios.png", dpi=130)
    plt.close(fig)


def sweep_demand(n_runs, rates):
    """Retrofit win share and quoted lead time vs retrofit demand."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mults = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for name in [BASELINE, "3 +TA DNV", "5 +TA ABS", "6 +25% FAT capacity"]:
        won, lead, fat = [], [], []
        for m in mults:
            rr = [simulate(SCENARIOS[name], 7_000_000 + i, {t: v * m for t, v in rates.items()})
                  for i in range(n_runs)]
            won.append(np.mean([r["retrofit_won"] / r["retrofit_opps"] for r in rr]))
            lead.append(np.nanmean([r["quote_ptc_medium"] for r in rr]))
            fat.append(np.mean([r["fat_load"] for r in rr]))
        ax[0].plot(mults, won, "o-", label=name)
        ax[1].plot(mults, lead, "o-", label=name)
        ax[2].plot(mults, fat, "o-", label=name)
    ax[0].set_ylabel("Share of retrofit opportunities won")
    ax[1].set_ylabel("Median quoted lead time, PTC medium [weeks]")
    ax[1].axhline(26, ls="--", color="grey")
    ax[1].axhline(17, ls=":", color="grey")
    ax[2].set_ylabel("FAT load")
    for a in ax:
        a.set_xlabel("Retrofit opportunities (× calibrated level)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "demand_sweep.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    OUT.mkdir(exist_ok=True)
    print("Calibrating retrofit opportunities on the baseline ...")
    rates = calibrate_retrofit(replace(SCENARIOS[BASELINE], learning=False))
    (OUT / "calibration.json").write_text(json.dumps(rates, indent=2))
    res = run_all(n, rates)
    print("Without learning ...")
    res_nl = run_all(max(50, n // 2), rates, learning=False)
    write_summary(res, n, rates, extra=learning_table(res, res_nl))
    plots(res)
    print("Demand sweep ...")
    sweep_demand(max(20, n // 5), rates)
    print("Hand check ...")
    hand_check.main()
