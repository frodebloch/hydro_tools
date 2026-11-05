"""Weekly Monte Carlo model of control-system engineering, production and test/FAT capacity,
retrofit quoting, hardware lead times and class approval.

Flow of one order
-----------------
  order --> engineering --+--> class approval (comments?) ----------+
                          |                                         +--> FAT --> shipping --> delivered
                          +--> production (needs hardware) ---------+
* Engineering, production and FAT are capacity calendars (engineer-weeks/week, hours/week).
  Orders reserve capacity in specific weeks, limited by a max rate per order.
* Production starts when engineering is done and hardware has arrived; it does not wait for
  approval. FAT waits for production and approval (class attends only when all comments are
  closed). FAT is planned with the nominal approval time; if comments push approval past the
  planned FAT slot, the slot is released and FAT is rebooked when the comments arrive.
* Bundled orders: design inputs and hardware ordering at contract. Scheduling policies:
    asap         - everything as early as possible
    fixed_offset - engineering from a fixed time before delivery, the rest right after
    alap         - everything as late as possible (with approval contingency) = current practice
  With `pull_forward`, idle capacity in the current week pulls bundled work forward
  (bundled work used as a buffer that fills gaps between retrofit orders).
* Retrofit: each opportunity is quoted the earliest feasible delivery; win probability p(L);
  price premium <= 4 months, discount > 6 months. Advanced hardware from stock if a set is
  available (replenished after the normal lead time), otherwise ordered at award.
* Late delivery: Orgalim LDs 0.5 %/full week, max 7.5 %; bundled on the mech + CS value.
* Learning (from the start of the horizon): hours per unit = max(floor, (n/n0)^-b) with n the
  cumulative simple-equivalent units completed. Implemented as capacity growing by 1/factor from
  the current week onwards, so quotes use the performance reached so far. The class comment
  probability falls the same way with the number of approvals.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from config import (CASE_BY_CASE, CLASS_SHARE, ORDER_TYPES, TYPE_APPROVED, Config, comment_probability,
                    cs_price, eng_capacity_per_week, eng_weeks, fat_capacity_per_week, learning_factor,
                    prod_capacity_per_week, prod_simple_hours)

DRAIN_WEEKS = 170
SOCIETIES = list(CLASS_SHARE)
SOC_CUM = np.cumsum([CLASS_SHARE[s] for s in SOCIETIES])
MAX_UNITS = 9


@dataclass
class Task:
    kind: str
    otype: str
    arrival: int
    due: int
    hw_ready: float
    approval: object
    price: float
    contract_value: float
    u: object
    eng: dict
    prod: dict
    fat: dict
    quoted_L: float = 0.0
    appr_end: float = None
    rebooked: bool = False
    fat_rebooks: int = 0


class Calendar:
    def __init__(self, n, cap):
        self.cap = np.full(n, cap)
        self.used = np.zeros(n)

    def free(self, w):
        return self.cap[w] - self.used[w]

    def forward(self, start, amount, rate):
        plan, w = {}, int(math.ceil(start - 1e-9))
        while amount > 1e-9:
            take = min(self.free(w), rate, amount)
            if take > 1e-9:
                plan[w] = take
                amount -= take
            w += 1
        return plan

    def backward(self, latest, earliest, amount, rate):
        """Allocate backwards from week `latest`; None if it does not fit after `earliest`."""
        plan, w, earliest = {}, int(math.floor(latest)), int(math.ceil(earliest - 1e-9))
        while amount > 1e-9 and w >= earliest:
            take = min(self.free(w), rate, amount)
            if take > 1e-9:
                plan[w] = take
                amount -= take
            w -= 1
        return plan if amount <= 1e-9 else None

    def commit(self, plan, sign=1.0):
        for w, v in plan.items():
            self.used[w] += sign * v

    def pull(self, task_plan, w, max_amount):
        moved = 0.0
        for k in sorted((k for k in task_plan if k > w), reverse=True):
            if moved >= max_amount - 1e-9:
                break
            m = min(max_amount - moved, task_plan[k])
            task_plan[k] -= m
            self.used[k] -= m
            if task_plan[k] <= 1e-9:
                del task_plan[k]
            moved += m
        if moved > 0:
            task_plan[w] = task_plan.get(w, 0.0) + moved
            self.used[w] += moved
        return moved


def end_of(plan):
    return max(plan) + 1


def lead_class(cfg, L):
    if L <= cfg.express_max_weeks:
        return "express"
    if L <= cfg.standard_max_weeks:
        return "standard"
    return "long"


def win_probability(cfg, L):
    return cfg.win_p_max / (1.0 + math.exp((L - cfg.win_L50) / cfg.win_width))


def price_factor(cfg, L):
    c = lead_class(cfg, L)
    if c == "express":
        return 1.0 + cfg.express_price_premium
    return 1.0 if c == "standard" else 1.0 - cfg.long_price_discount


def ld(cfg, late_weeks, value):
    return min(cfg.ld_per_week * math.floor(late_weeks + 1e-9), cfg.ld_cap) * value


def ramp(cfg: Config, t: str, w: int) -> float:
    """Market multiplier for retrofit type t in week w (1 before the horizon starts)."""
    m = cfg.retrofit_ramp.get(t, 1.0)
    x = min(max((w - cfg.warmup_weeks) / cfg.ramp_weeks, 0.0), 1.0)
    return 1.0 + (m - 1.0) * x


def generate_arrivals(cfg: Config, seed: int, n_weeks: int, retro_rates: dict):
    """Pre-draw all randomness (common random numbers across scenarios with the same market)."""
    rng = np.random.default_rng(seed)
    out = []
    for w in range(n_weeks):
        b, r = [], []
        for t, n in cfg.bundled_per_year.items():
            for _ in range(rng.poisson(n / 52.0)):
                b.append(dict(otype=t, lead=int(round(rng.uniform(*cfg.bundled_lead_weeks))),
                              u=rng.random(4), z=rng.standard_normal(MAX_UNITS)))
        for t, n in retro_rates.items():
            for _ in range(rng.poisson(n * ramp(cfg, t, w) / 52.0)):
                r.append(dict(otype=t, u=rng.random(4), z=rng.standard_normal(MAX_UNITS)))
        rng.shuffle(r)
        out.append((b, r))
    return out


def unit_hours(cfg, ot, z, simple_hours):
    """Hours for all units in an order: lognormal spread per unit, advanced = ratio x simple."""
    s = math.sqrt(math.log(1 + cfg.unit_cv ** 2))
    f = np.exp(s * z - s * s / 2)
    n = ot.n_simple
    return float(simple_hours * (f[:n].sum() + cfg.advanced_to_simple * f[n:n + ot.n_advanced].sum()))


def society(u):
    return SOCIETIES[int(np.searchsorted(SOC_CUM, u * SOC_CUM[-1]))]


def simulate(cfg: Config, seed: int, retro_rates: dict):
    t0, t1 = cfg.warmup_weeks, cfg.warmup_weeks + cfg.horizon_weeks
    n = t1 + DRAIN_WEEKS + 600
    E = Calendar(n, eng_capacity_per_week(cfg))
    P = Calendar(n, prod_capacity_per_week(cfg))
    F = Calendar(n, fat_capacity_per_week(cfg))
    psh = prod_simple_hours(cfg)
    arrivals = generate_arrivals(cfg, seed, t1 + DRAIN_WEEKS, retro_rates)
    ta_ready = t0 + cfg.type_approval_ready_week
    ship = cfg.shipping_weeks
    stock = [0.0] * cfg.advanced_stock_sets
    lo, hi = cfg.advanced_hw_weeks

    for i, _ in enumerate(cfg.type_approved_societies):
        E.commit(E.forward(t0 + 8 * i, cfg.type_approval_eng_weeks, 0.5))

    def approval_for(u, w):
        ok = society(u) in cfg.type_approved_societies and w >= ta_ready
        return TYPE_APPROVED if ok else CASE_BY_CASE

    def forward_chain(ot, start, hw, ew, ph, fh, appr_weeks):
        ep = E.forward(start, ew, ot.eng_max_rate)
        E.commit(ep)
        pp = P.forward(max(end_of(ep), hw), ph, ot.prod_max_rate)
        P.commit(pp)
        fp = F.forward(max(end_of(pp), end_of(ep) + appr_weeks), fh, ot.fat_max_rate)
        F.commit(fp)
        return ep, pp, fp

    active, finished, quotes = [], [], []
    base_cap = {"eng": E.cap[0], "prod": P.cap[0], "fat": F.cap[0]}
    cals = {"eng": E, "prod": P, "fat": F}
    n_units, n_appr = cfg.units_built_today, cfg.approvals_done_today
    last_f = {k: 1.0 for k in cals}

    for w in range(t1 + DRAIN_WEEKS):
        bundled_arr, retro_arr = arrivals[w]

        # ---------------- learning: capacity from this week on ----------------
        if cfg.learning and w >= t0:
            for k, cal in cals.items():
                f = learning_factor(cfg.learning_rates[k], n_units, cfg.units_built_today, cfg.learning_floor)
                if f < last_f[k] - 1e-4:
                    cal.cap[w:] = base_cap[k] / f
                    last_f[k] = f
        if w == t1 - 1:
            f_end = min(last_f.values()) if cfg.learning else 1.0
            c_end = comment_probability(cfg, CASE_BY_CASE.p_comments, n_appr)

        # ---------------- bundled arrivals ----------------
        for a in bundled_arr:
            ot, u = ORDER_TYPES[a["otype"]], a["u"]
            due = w + a["lead"]
            appr = approval_for(u[0], w)
            hw = w + (lo + u[3] * (hi - lo) if ot.n_advanced else cfg.simple_hw_weeks)
            ew = eng_weeks(cfg, ot)
            ph = unit_hours(cfg, ot, a["z"], psh)
            fh = unit_hours(cfg, ot, a["z"], cfg.fat_simple_hours)
            plans = None
            if cfg.bundled_policy == "alap":
                fp = F.backward(due - ship - 1, w + 2, fh, ot.fat_max_rate)
                if fp:
                    F.commit(fp)
                    pp = P.backward(min(fp) - 1, max(w + 1, hw), ph, ot.prod_max_rate)
                    if pp:
                        P.commit(pp)
                        latest_eng_end = min(min(pp), min(fp) - appr.nominal_weeks - appr.comment_weeks)
                        ep = E.backward(latest_eng_end - 1, w + 1, ew, ot.eng_max_rate)
                        if ep:
                            E.commit(ep)
                            plans = (ep, pp, fp)
                        else:
                            P.commit(pp, -1)
                            F.commit(fp, -1)
                    else:
                        F.commit(fp, -1)
                if plans is None:          # does not fit late -> as early as possible
                    plans = forward_chain(ot, w + 1, hw, ew, ph, fh, appr.nominal_weeks)
            else:
                s = w + 1 if cfg.bundled_policy == "asap" else max(w + 1, due - cfg.bundled_offset_start)
                plans = forward_chain(ot, s, hw, ew, ph, fh, appr.nominal_weeks)
            value = cs_price(cfg, ot) + ot.n_mech * cfg.price_mech_unit
            active.append(Task("bundled", ot.name, w, due, hw, appr, cs_price(cfg, ot), value, u, *plans))

        # ---------------- retrofit opportunities ----------------
        for a in retro_arr:
            ot, u = ORDER_TYPES[a["otype"]], a["u"]
            appr = approval_for(u[0], w)
            stock_idx = None
            hw = w + cfg.simple_hw_weeks
            if ot.n_advanced:
                avail = [k for k, t_av in enumerate(stock) if t_av <= w]
                if avail:
                    stock_idx = avail[0]
                else:
                    hw = w + lo + u[3] * (hi - lo)
            ew = eng_weeks(cfg, ot)
            ph = unit_hours(cfg, ot, a["z"], psh)
            fh = unit_hours(cfg, ot, a["z"], cfg.fat_simple_hours)
            appr_q = appr.nominal_weeks + (appr.comment_weeks if cfg.quote_comment_allowance else 0.0)
            ep, pp, fp = forward_chain(ot, w + 1, hw, ew, ph, fh, appr_q)
            L = math.ceil(end_of(fp) + ship - w + cfg.quote_safety_weeks)
            quoted = L <= cfg.max_quote_weeks
            won = quoted and u[1] < win_probability(cfg, L)
            quotes.append((w, ot.name, L, won, lead_class(cfg, L) if quoted else "none"))
            if not won:
                E.commit(ep, -1)
                P.commit(pp, -1)
                F.commit(fp, -1)
                continue
            if stock_idx is not None:
                stock[stock_idx] = w + lo + u[3] * (hi - lo)
            price = cs_price(cfg, ot) * price_factor(cfg, L)
            active.append(Task("retrofit", ot.name, w, w + L, hw, appr, price, price, u,
                               ep, pp, fp, quoted_L=L))

        # ---------------- approval comments -> rebook FAT ----------------
        for t in active:
            if t.appr_end is None or t.rebooked:
                continue
            if w >= math.ceil(end_of(t.eng) + t.approval.nominal_weeks) and min(t.fat) >= w:
                t.rebooked = True
                earliest = max(end_of(t.prod), t.appr_end)
                if min(t.fat) < earliest:
                    F.commit(t.fat, -1)
                    fh = sum(t.fat.values())
                    t.fat = F.forward(earliest, fh, ORDER_TYPES[t.otype].fat_max_rate)
                    F.commit(t.fat)
                    t.fat_rebooks += 1

        # ---------------- pull bundled work into idle capacity ----------------
        bund = sorted((t for t in active if t.kind == "bundled"), key=lambda t: t.due) if cfg.pull_forward else []
        for cal, attr, rate_attr, ready in (
                (E, "eng", "eng_max_rate", lambda t: True),
                (P, "prod", "prod_max_rate", lambda t: t.hw_ready <= w and max(t.eng) < w),
                (F, "fat", "fat_max_rate", lambda t: t.appr_end is not None and t.appr_end <= w
                 and max(t.prod) < w)):
            free = cal.free(w)
            for t in bund:
                if free <= 1e-9:
                    break
                plan = getattr(t, attr)
                if any(k > w for k in plan) and ready(t):
                    room = getattr(ORDER_TYPES[t.otype], rate_attr) - plan.get(w, 0.0)
                    free -= cal.pull(plan, w, min(free, room))

        # ---------------- state updates / completions ----------------
        still = []
        for t in active:
            if t.appr_end is None and max(t.eng) <= w:
                p_c = comment_probability(cfg, t.approval.p_comments, n_appr) if w >= t0 else t.approval.p_comments
                if w >= t0:
                    n_appr += 1
                extra = t.approval.comment_weeks if t.u[2] < p_c else 0.0
                t.appr_end = end_of(t.eng) + t.approval.nominal_weeks + extra
            if max(t.fat) <= w and t.appr_end is not None:
                t.ready = end_of(t.fat) + ship
                t.late = max(0.0, t.ready - t.due)
                finished.append(t)
                if w >= t0:
                    ot = ORDER_TYPES[t.otype]
                    n_units += ot.n_simple + cfg.advanced_to_simple * ot.n_advanced
            else:
                still.append(t)
        active = still

    # ---------------- results within [t0, t1) ----------------
    years = cfg.horizon_weeks / 52.0
    inw = lambda wk: t0 <= wk < t1
    disc = lambda wk: (1.0 + cfg.discount_rate) ** (-(wk - t0) / 52.0)
    r = {}
    q = [x for x in quotes if inw(x[0])]
    r["retrofit_opps"] = len(q) / years
    r["retrofit_won"] = sum(x[3] for x in q) / years
    for t in ORDER_TYPES:
        r[f"won_{t}"] = sum(1 for x in q if x[3] and x[1] == t) / years
        Ls = [x[2] for x in q if x[1] == t]
        r[f"quote_{t}"] = float(np.median(Ls)) if Ls else float("nan")
    for c in ("express", "standard", "long"):
        r[f"won_{c}"] = sum(1 for x in q if x[3] and x[4] == c) / years
    r["no_quote"] = sum(1 for x in q if x[4] == "none") / years

    for y in range(int(years)):
        a, b = t0 + 52 * y, t0 + 52 * (y + 1)
        qy = [x for x in quotes if a <= x[0] < b]
        for t in ORDER_TYPES:
            r[f"y{y + 1}_won_{t}"] = sum(1 for x in qy if x[3] and x[1] == t)
            r[f"y{y + 1}_opps_{t}"] = sum(1 for x in qy if x[1] == t)
            Ls = [x[2] for x in qy if x[1] == t]
            r[f"y{y + 1}_quote_{t}"] = float(np.median(Ls)) if Ls else float("nan")
        for name, cal in (("eng", E), ("prod", P), ("fat", F)):
            r[f"y{y + 1}_{name}_load"] = cal.used[a:b].sum() / cal.cap[a:b].sum()

    retro = [t for t in finished if t.kind == "retrofit" and inw(t.arrival)]
    bund = [t for t in finished if t.kind == "bundled" and inw(t.arrival)]
    r["retrofit_revenue"] = sum(t.price for t in retro) / years
    r["retrofit_late_share"] = float(np.mean([t.late >= 1 for t in retro])) if retro else 0.0
    r["bundled_late_share"] = float(np.mean([t.late >= 1 for t in bund])) if bund else 0.0
    r["bundled_late_weeks"] = float(np.mean([t.late for t in bund])) if bund else 0.0
    late3 = [t for t in bund if t.arrival >= t0 + 104]
    r["bundled_late_share_y3_5"] = float(np.mean([t.late >= 1 for t in late3])) if late3 else 0.0
    r["fat_rebook_share"] = float(np.mean([t.fat_rebooks > 0 for t in retro + bund])) if finished else 0.0
    r["ld_retrofit"] = sum(ld(cfg, t.late, t.contract_value) for t in retro) / years
    r["ld_bundled"] = sum(ld(cfg, t.late, t.contract_value) for t in bund) / years
    for name, cal in (("eng", E), ("prod", P), ("fat", F)):
        r[f"{name}_load"] = cal.used[t0:t1].sum() / cal.cap[t0:t1].sum()
    r["eng_hours"] = E.used[t0:t1].sum() * 37.5 / years
    r["hours_factor_end"] = f_end
    r["comment_rate_end"] = c_end
    r["prod_hours"] = P.used[t0:t1].sum() / years

    # NPV: contribution at award (20 % margin + premium/discount), LDs at delivery, lever costs
    npv = 0.0
    for t in retro + bund:
        base = cs_price(cfg, ORDER_TYPES[t.otype])
        npv += (cfg.margin * base + (t.price - base)) * disc(t.arrival)
        npv -= ld(cfg, t.late, t.contract_value) * disc(t.ready)
    n_ta = len(cfg.type_approved_societies)
    if n_ta:
        npv -= cfg.ta_cost_first + cfg.ta_cost_additional * (n_ta - 1)
    if cfg.fat_capacity_factor > 1.0:
        npv -= cfg.fat_extra_cost_upfront
    stock_value = cfg.advanced_stock_sets * cfg.price_advanced * (1 - cfg.margin) * cfg.long_lead_share_of_cost
    extra_prod_h = (cfg.prod_capacity_factor - 1.0) * cfg.prod_hours_reported / cfg.prod_load
    for y in range(int(years)):
        wk = t0 + 52 * y
        yearly = stock_value * cfg.stock_carrying_rate + extra_prod_h * cfg.prod_cost_per_hour
        if cfg.fat_capacity_factor > 1.0:
            yearly += cfg.fat_extra_cost_per_year
        if wk + 52 > ta_ready:
            yearly += cfg.ta_cost_per_year * n_ta
        npv -= yearly * disc(wk)
    r["npv"] = npv
    return r


def calibrate_retrofit(cfg: Config, n_runs=40, iters=5, verbose=True):
    """Retrofit opportunities per type so that `cfg` wins `retrofit_won_target` per type."""
    rates = {t: n / 0.6 for t, n in cfg.retrofit_won_target.items()}
    for it in range(iters):
        runs = [simulate(cfg, 9_000_000 + i, rates) for i in range(n_runs)]
        won = {t: np.mean([r[f"won_{t}"] for r in runs]) for t in rates}
        rates = {t: rates[t] * cfg.retrofit_won_target[t] / max(won[t], 0.1) for t in rates}
        if verbose:
            print(f"  calibration {it}: won " + ", ".join(f"{t} {won[t]:.1f}" for t in won)
                  + " -> opportunities " + ", ".join(f"{t} {rates[t]:.1f}" for t in rates), flush=True)
    return rates
