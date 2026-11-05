"""Hand-calculation check of the simulation model.

Calculator-level formulas, using the same inputs as the model (config.py):

  1. Load            rho = hours needed / hours available            (per resource)
  2. Min lead time   L_min = max(max(eng, hardware) + production, eng + approval) + FAT + shipping
  3. Queue           anchored on today's quoted lead time:
                       W_today = L_quoted_today - L_min - safety
                     and scaled with the standard queueing factor (Pollaczek-Khinchine / Cobham):
                       W = W_today x [rho / (1 - rho_seen)]_new / [rho / (1 - rho)]_today
                     rho_seen = total load (first come first served, today),
                              = retrofit load only when bundled work is used as a buffer (pull forward).
                     Summed over engineering, production and FAT, weighted by mean job time.
  4. Contribution    price x (margin + express share x premium - long share x discount);
                     shares from a +-3 week spread of quotes around the average lead time
  5. Win rate        won = opportunities x p(L), same p(L) curve as the model
  6. NPV             sum of yearly contribution x discount factor - lever costs
  7. Learning        hours factor in year y = max(floor, (n_y / n_0)^-b), b = log2(1 / learning rate),
                     n_y = units built today + units per year x (y + 0.5); capacity / factor

More wins -> higher load -> longer lead time, so steps 1-5 are repeated until they settle
(a few rounds by hand; the script does it automatically).

Usage:  .venv/bin/python hand_check.py      (reads results/model_means.json if present)
Output: results/hand_check.md
"""

import json
import math
from dataclasses import replace
from pathlib import Path

from config import (CASE_BY_CASE, CLASS_SHARE, ORDER_TYPES, SCENARIOS, TYPE_APPROVED, cs_price,
                    eng_capacity_per_week, eng_weeks, fat_capacity_per_week, learning_factor,
                    prod_capacity_per_week, prod_simple_hours)
from sim import win_probability

OUT = Path(__file__).parent / "results"
README = Path(__file__).parent / "README.md"
MARK_BEGIN = "<!-- hand-check:begin"
MARK_END = "<!-- hand-check:end -->"
L_SPREAD = 3.0      # weeks: spread of actual quotes around the average, for express / long shares
RHO_MAX = 0.98      # practical maximum load of a resource
TOL = (0.67, 1.5)   # hand / model ratio regarded as "ballpark agreement"
CHECKED = ["1 baseline", "2 +pull bundled fwd", "3 +TA DNV", "4 +1 adv. stock set", "5 +TA ABS",
           "5b +10% prod. (not FAT)", "6 +25% FAT capacity", "7 +FAT +production"]


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def class_shares(cfg, L):
    """Share of quotes that are express / long, assuming quotes spread +-L_SPREAD around L."""
    express = phi((cfg.express_max_weeks + 0.5 - L) / L_SPREAD)
    long_ = 1 - phi((cfg.standard_max_weeks + 0.5 - L) / L_SPREAD)
    return express, long_


def simple_equiv(cfg, ot):
    return ot.n_simple + cfg.advanced_to_simple * ot.n_advanced


def resources(cfg):
    """(name, job size per order type in resource units, capacity per week, max rate per order type)."""
    return [
        ("engineering", {t: eng_weeks(cfg, ot) for t, ot in ORDER_TYPES.items()},
         eng_capacity_per_week(cfg), {t: ot.eng_max_rate for t, ot in ORDER_TYPES.items()}),
        ("production", {t: simple_equiv(cfg, ot) * prod_simple_hours(cfg) for t, ot in ORDER_TYPES.items()},
         prod_capacity_per_week(cfg), {t: ot.prod_max_rate for t, ot in ORDER_TYPES.items()}),
        ("FAT", {t: simple_equiv(cfg, ot) * cfg.fat_simple_hours for t, ot in ORDER_TYPES.items()},
         fat_capacity_per_week(cfg), {t: ot.fat_max_rate for t, ot in ORDER_TYPES.items()}),
    ]


def l_min(cfg, t, appr, hw):
    """Formula 2: critical path without queues, nominal approval."""
    res = {name: job[t] / rate[t] for name, job, _, rate in resources(cfg)}
    eng, prod, fat = res["engineering"], res["production"], res["FAT"]
    return max(max(eng, hw) + prod, eng + appr.nominal_weeks) + fat + cfg.shipping_weeks


def loads(cfg, won):
    """Formula 1 per resource: total load, retrofit load, and weight k = E[S^2] / (2 E[S])."""
    out = {}
    for name, job, cap, _ in resources(cfg):
        lam = {t: (cfg.bundled_per_year[t] + won[t]) / 52.0 for t in ORDER_TYPES}
        s = {t: job[t] / cap for t in ORDER_TYPES}            # weeks of the whole resource
        rho = sum(lam[t] * s[t] for t in ORDER_TYPES)
        rho_r = sum(won[t] / 52.0 * s[t] for t in ORDER_TYPES)
        k = sum(lam[t] * s[t] ** 2 for t in ORDER_TYPES) / (2 * rho)
        out[name] = dict(rho=rho, rho_r=rho_r, k=k)
    return out


def queue_factor(ld, pull):
    tot = 0.0
    for v in ld.values():
        rho = min(v["rho"], 0.99)
        seen = min(v["rho_r"] if pull else v["rho"], 0.99)
        tot += v["k"] * rho / (1 - seen)
    return tot


def queue_weeks(cfg, won, base):
    """Formula 3: today's queue scaled by the change in the queueing factor."""
    w_today = base.today_quote_quad_weeks - l_min(base, "quad", CASE_BY_CASE, base.simple_hw_weeks) \
        - base.quote_safety_weeks
    f_today = queue_factor(loads(base, base.retrofit_won_target), pull=False)
    return max(w_today, 0.0) * queue_factor(loads(cfg, won), cfg.pull_forward) / f_today


def steady_state(cfg, opps, ta_active=True, base=None):
    """Formulas 1-5 iterated to a consistent volume / lead time."""
    ta_share = sum(CLASS_SHARE[s] for s in cfg.type_approved_societies) if ta_active else 0.0
    hw_adv = sum(cfg.advanced_hw_weeks) / 2
    base = base or SCENARIOS["1 baseline"]
    won = dict(cfg.retrofit_won_target)
    for _ in range(40):
        Wq = queue_weeks(cfg, won, base)
        new_won, contrib, express, detail = {}, 0.0, 0.0, {}
        for t, ot in ORDER_TYPES.items():
            cases = []   # (share of opportunities, approval, hardware weeks)
            stock_share = 0.0
            if ot.n_advanced:
                if cfg.advanced_stock_sets:
                    per_year = cfg.advanced_stock_sets * 52.0 / hw_adv
                    stock_share = min(1.0, per_year / max(won[t], 1e-9))
            for a_share, appr in ((ta_share, TYPE_APPROVED), (1 - ta_share, CASE_BY_CASE)):
                if a_share <= 0:
                    continue
                if ot.n_advanced:
                    cases.append((a_share * stock_share, appr, cfg.simple_hw_weeks))
                    cases.append((a_share * (1 - stock_share), appr, hw_adv))
                else:
                    cases.append((a_share, appr, cfg.simple_hw_weeks))
            w_t = 0.0
            Ls = []
            for share, appr, hw in cases:
                if share <= 0:
                    continue
                L = l_min(cfg, t, appr, hw) + Wq + cfg.quote_safety_weeks
                n = opps[t] * share * win_probability(cfg, L)
                w_t += n
                ex, lg = class_shares(cfg, L)
                contrib += n * cs_price(cfg, ot) * (cfg.margin + ex * cfg.express_price_premium
                                                    - lg * cfg.long_price_discount)
                express += n * ex
                Ls.append((share, L))
            new_won[t] = w_t
            detail[t] = sum(s * L for s, L in Ls) / sum(s for s, _ in Ls)
        # capacity cap: retrofit wins cannot push any resource above RHO_MAX
        ld = loads(cfg, new_won)
        scale = 1.0
        for v in ld.values():
            bundled = v["rho"] - v["rho_r"]
            if v["rho"] > RHO_MAX and v["rho_r"] > 0:
                scale = min(scale, max(RHO_MAX - bundled, 0.0) / v["rho_r"])
        if scale < 1.0:
            contrib *= scale
            express *= scale
            new_won = {t: n * scale for t, n in new_won.items()}
        won = {t: 0.5 * won[t] + 0.5 * new_won[t] for t in won}   # damped update
    return dict(won=won, contrib=contrib, express=express, L=detail, queue=Wq,
                loads=loads(cfg, won))


def units_per_year(cfg, won):
    return sum((cfg.bundled_per_year[t] + won[t]) * simple_equiv(cfg, ot) for t, ot in ORDER_TYPES.items())


def year_factors(cfg, units_rate, years):
    """Formula 7: hours factor per year at mid-year cumulative volume."""
    if not cfg.learning:
        return [1.0] * years
    rate = min(cfg.learning_rates.values())
    return [learning_factor(rate, cfg.units_built_today + units_rate * (y + 0.5), cfg.units_built_today,
                            cfg.learning_floor) for y in range(years)]


def with_learning(cfg, f):
    """Same config, capacities divided by the hours factor (lever costs still taken from cfg)."""
    return replace(cfg, engineers=cfg.engineers / f, prod_capacity_factor=cfg.prod_capacity_factor / f,
                   fat_capacity_factor=cfg.fat_capacity_factor / f)


def npv(cfg, opps):
    """Formula 6: 5 years, discounted mid-year; type approval active from year 2; learning per year."""
    years = int(cfg.horizon_weeks / 52)
    first_year_ta = cfg.type_approval_ready_week // 52
    first = steady_state(cfg, opps, ta_active=False)
    factors = year_factors(cfg, units_per_year(cfg, first["won"]), years)
    n_ta = len(cfg.type_approved_societies)
    stock_value = cfg.advanced_stock_sets * cfg.price_advanced * (1 - cfg.margin) * cfg.long_lead_share_of_cost
    extra_prod_h = (cfg.prod_capacity_factor - 1.0) * cfg.prod_hours_reported / cfg.prod_load
    total, states = 0.0, []
    for y in range(years):
        d = (1 + cfg.discount_rate) ** -(y + 0.5)
        ss = steady_state(with_learning(cfg, factors[y]), opps, ta_active=y >= first_year_ta)
        states.append(ss)
        cost = stock_value * cfg.stock_carrying_rate + extra_prod_h * cfg.prod_cost_per_hour
        cost += cfg.fat_extra_cost_per_year if cfg.fat_capacity_factor > 1 else 0.0
        cost += cfg.ta_cost_per_year * n_ta if y >= first_year_ta else 0.0
        total += (ss["contrib"] - cost) * d
    if n_ta:
        total -= cfg.ta_cost_first + cfg.ta_cost_additional * (n_ta - 1)
    if cfg.fat_capacity_factor > 1:
        total -= cfg.fat_extra_cost_upfront
    # report averages over the years
    avg = dict(states[-1])
    for k in ("contrib", "express", "queue"):
        avg[k] = sum(s[k] for s in states) / years
    avg["won"] = {t: sum(s["won"][t] for s in states) / years for t in ORDER_TYPES}
    avg["L"] = {t: sum(s["L"][t] for s in states) / years for t in ORDER_TYPES}
    avg["loads"] = {k: {kk: sum(s["loads"][k][kk] for s in states) / years for kk in ("rho", "rho_r")}
                    for k in states[0]["loads"]}
    avg["factors"] = factors
    return total, avg


def hand_opportunities(base):
    """Opportunities that give today's wins at today's quoted lead time (by hand: won / p(L))."""
    opps = {}
    for t, ot in ORDER_TYPES.items():
        hw = sum(base.advanced_hw_weeks) / 2 if ot.n_advanced else base.simple_hw_weeks
        w_today = base.today_quote_quad_weeks - l_min(base, "quad", CASE_BY_CASE, base.simple_hw_weeks) \
            - base.quote_safety_weeks
        L = l_min(base, t, CASE_BY_CASE, hw) + w_today + base.quote_safety_weeks
        opps[t] = base.retrofit_won_target[t] / win_probability(base, L)
    return opps


def update_readme(lines):
    """Replace the generated block between the markers in README.md (if the markers exist)."""
    if not README.exists():
        return
    text = README.read_text()
    i, j = text.find(MARK_BEGIN), text.find(MARK_END)
    if i < 0 or j < 0:
        return
    i = text.index("\n", i) + 1
    README.write_text(text[:i] + "\n".join(lines) + "\n" + text[j:])


def main():
    model_file = OUT / "model_means.json"
    model = json.loads(model_file.read_text()) if model_file.exists() else {}
    opps = hand_opportunities(SCENARIOS["1 baseline"])
    base = SCENARIOS["1 baseline"]
    L = ["# Hand check", "",
         "Calculator-level formulas (see README, 'Hand check') with the same inputs as the model.", ""]

    # ---- 1. load table at baseline volumes
    L += ["## 1. Load at baseline volumes", "", "| Resource | Need/yr | Available/yr | Load ρ |", "|---|---:|---:|---:|"]
    won0 = base.retrofit_won_target
    for name, job, cap, _ in resources(base):
        need = sum((base.bundled_per_year[t] + won0[t]) * job[t] for t in ORDER_TYPES)
        unit = "eng-wk" if name == "engineering" else "h"
        L.append(f"| {name} | {need:,.0f} {unit} | {cap * 52:,.0f} {unit} | {need / (cap * 52):.0%} |")

    # ---- 2. minimum lead times
    hw_adv = sum(base.advanced_hw_weeks) / 2
    L += ["", "## 2. Minimum lead time without queues (weeks)", "",
          "| Order type | Case-by-case | Type approved | Type approved + adv. stock |", "|---|---:|---:|---:|"]
    for t, ot in ORDER_TYPES.items():
        hw = hw_adv if ot.n_advanced else base.simple_hw_weeks
        L.append(f"| {t} | {l_min(base, t, CASE_BY_CASE, hw):.1f} | {l_min(base, t, TYPE_APPROVED, hw):.1f} | "
                 f"{l_min(base, t, TYPE_APPROVED, base.simple_hw_weeks):.1f} |")
    L.append(f"\nExpress limit: {base.express_max_weeks:.0f} weeks. Comments add on average "
             f"{CASE_BY_CASE.p_comments * CASE_BY_CASE.comment_weeks:.1f} weeks (case-by-case).")

    # ---- 3. queue factor table
    L += ["", "## 3. Queue factor ρ/(1−ρ)", "", "| ρ | 0.80 | 0.85 | 0.90 | 0.92 | 0.95 |", "|---|---|---|---|---|---|",
          "| ρ/(1−ρ) | " + " | ".join(f"{r / (1 - r):.1f}" for r in (0.8, 0.85, 0.9, 0.92, 0.95)) + " |"]

    # ---- 4. contribution per order
    L += ["", "## 4. Contribution per order (kNOK)", "", "| Type | Price | Contribution | Express extra |", "|---|---:|---:|---:|"]
    for t, ot in ORDER_TYPES.items():
        p = cs_price(base, ot)
        L.append(f"| {t} | {p:,.0f} | {p * base.margin:,.0f} | {p * base.express_price_premium:,.0f} |")

    # ---- 5/6. scenarios: hand vs model
    base_npv, base_ss = npv(base, opps)
    if base.learning:
        L += ["", "## 7. Learning: hours factor per year (baseline volume)", "",
              "| Year | " + " | ".join(str(y + 1) for y in range(len(base_ss["factors"]))) + " |",
              "|---|" + "---:|" * len(base_ss["factors"]),
              "| Hours factor | " + " | ".join(f"{f:.2f}" for f in base_ss["factors"]) + " |",
              "", f"Learning rate {min(base.learning_rates.values()):.0%} per doubling, "
              f"{base.units_built_today:.0f} units built today, floor {base.learning_floor:.0%}."]
    w_today = base.today_quote_quad_weeks - l_min(base, "quad", CASE_BY_CASE, base.simple_hw_weeks) \
        - base.quote_safety_weeks
    L += ["", f"Today's queue (anchor): quoted quad lead time {base.today_quote_quad_weeks:.0f} wk − L_min "
          f"{l_min(base, 'quad', CASE_BY_CASE, base.simple_hw_weeks):.1f} − safety {base.quote_safety_weeks:.0f} "
          f"= **{w_today:.1f} weeks** of queue.",
          "", "Retrofit opportunities by hand (today's wins / p(today's lead time)): "
          + ", ".join(f"{t} {v:.0f}" for t, v in opps.items()) + f" = {sum(opps.values()):.0f} per year."]
    L += ["", "## 5–6. Scenarios: hand calculation vs model", "",
          "| Scenario | FAT load ρ (retrofit part), 5-yr avg. | Queue [wk] hand | Quote quad [wk] hand / model | Retrofit won/yr hand / model | "
          "Express/yr hand / model | ΔNPV [MNOK] hand / model | Ratio | Check | Step vs previous [MNOK] hand / model |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    prev = None
    hand_dn = {}
    readme_rows = []
    n_fail = 0
    for name in CHECKED:
        cfg = SCENARIOS[name]
        v, ss = npv(cfg, opps)
        dn = (v - base_npv) / 1000
        m = model.get(name, {})
        q = ss["queue"]
        won = sum(ss["won"].values())
        mdn = m.get("dnpv")
        if mdn is None or name == "1 baseline":
            ratio, check = "–", "–"
        elif abs(mdn) < 10 and abs(dn) < 10:
            ratio, check = "–", "OK (both small)"
        else:
            r = dn / mdn if mdn else float("inf")
            ok = TOL[0] <= r <= TOL[1]
            n_fail += not ok
            ratio, check = f"{r:.2f}", "OK" if ok else "**CHECK**"
        fm = lambda k, f="{:.1f}": f.format(m[k]) if k in m else "–"
        hand_dn[name] = dn
        parent = "5 +TA ABS" if name.startswith("5b") else prev
        if parent is None:
            step = "–"
        else:
            mstep = (f"{m['dnpv'] - model[parent]['dnpv']:.1f}" if "dnpv" in m and parent in model else "–")
            step = f"{dn - hand_dn[parent]:.1f} / {mstep}" + (" (vs 5)" if name.startswith("5b") else "")
        if not name.startswith("5b"):
            prev = name
        if name != "1 baseline":
            readme_rows.append(f"| {name} | {dn:.0f} | {fm('dnpv', '{:.0f}')} | {ratio} | {check} |")
        fl = ss["loads"]["FAT"]
        L.append(f"| {name} | {fl['rho']:.0%} ({fl['rho_r']:.0%}) | {q:.1f} | {ss['L']['quad']:.0f} / {fm('quote_quad', '{:.0f}')} | "
                 f"{won:.1f} / {fm('retrofit_won')} | {ss['express']:.1f} / {fm('won_express')} | "
                 f"{dn:.1f} / {fm('dnpv')} | {ratio} | {check} | {step} |")
    L += ["", f"Ballpark agreement: hand/model ΔNPV ratio within {TOL[0]}–{TOL[1]}, or both below 10 MNOK.",
          f"Scenarios outside: {n_fail}.", "",
          "Hand-check values are averages over the 5 years. Known simplifications: average lead time per order type (spread only assumed as",
          "±3 weeks for express share), queue anchored on today's quote and scaled, no late-delivery penalties (LDs), no approval",
          "comments in the quote (as in the model), type approval from year 2.", ""]
    OUT.mkdir(exist_ok=True)
    (OUT / "hand_check.md").write_text("\n".join(L))
    learn = "with learning" if base.learning else "without learning"
    update_readme([f"Latest run ({learn}), ΔNPV vs baseline over 5 years [MNOK]:", "",
                   "| Scenario | Hand | Model | Ratio | Check |", "|---|---:|---:|---:|---|",
                   *readme_rows, "", f"Scenarios outside the band: {n_fail}."])
    print("\n".join(L))
    return n_fail


if __name__ == "__main__":
    main()
