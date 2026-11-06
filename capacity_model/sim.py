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
* Learning per product family (from the start of the horizon): hours per project =
  max(floor, (n_eff/n_eff0)^-b) x today's hours, with n_eff = own experience + transfer x the
  other families' experience (simple-equivalents completed). New projects are booked with the
  current factor; when a family's factor improves, the remaining booked work of its active
  projects is scaled down. The class comment probability falls with the number of approvals.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from config import (CASE_BY_CASE, CLASS_SHARE, ORDER_TYPES, TYPE_APPROVED, Config, bundled_per_year,
                    comment_probability, contribution, cs_price, eng_capacity_per_week, eng_weeks,
                    experience_today, family_factors, fat_capacity_per_week, lever_cost_per_year,
                    prod_capacity_per_week,
                    prod_hours, retrofit_won_target, rework_share_at, traded_cost, FAMILIES)

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
    wc: float = 0.0
    g: float = 1.0


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
        for t, n in bundled_per_year(cfg).items():
            for _ in range(rng.poisson(n / 52.0)):
                b.append(dict(otype=t, lead=int(round(rng.uniform(*cfg.bundled_lead_weeks))),
                              u=rng.random(4), z=rng.standard_normal(MAX_UNITS)))
        for t, n in retro_rates.items():
            for _ in range(rng.poisson(n * ramp(cfg, t, w) / 52.0)):
                r.append(dict(otype=t, u=rng.random(4), z=rng.standard_normal(MAX_UNITS)))
        rng.shuffle(r)
        out.append((b, r))
    return out


def rework_draw(cfg, z):
    """Per-project rework multiplier g (mean 1, lognormal with cv rework_cv)."""
    s = math.sqrt(math.log(1 + cfg.rework_cv ** 2))
    return math.exp(s * z[1] - s * s / 2)


def unit_hours(cfg, mean_hours, z):
    """Hours for one project: mean x lognormal factor (same draw for production and FAT)."""
    s = math.sqrt(math.log(1 + cfg.unit_cv ** 2))
    return float(mean_hours * math.exp(s * z[0] - s * s / 2))


def society(u):
    return SOCIETIES[int(np.searchsorted(SOC_CUM, u * SOC_CUM[-1]))]


def simulate(cfg: Config, seed: int, retro_rates: dict):
    t0, t1 = cfg.warmup_weeks, cfg.warmup_weeks + cfg.horizon_weeks
    n = t1 + DRAIN_WEEKS + 600
    E = Calendar(n, eng_capacity_per_week(cfg))
    P = Calendar(n, prod_capacity_per_week(cfg))
    F = Calendar(n, fat_capacity_per_week(cfg))
    arrivals = generate_arrivals(cfg, seed, t1 + DRAIN_WEEKS, retro_rates)
    ta_ready = t0 + cfg.type_approval_ready_week
    ship = cfg.shipping_weeks
    stock = [0.0] * cfg.advanced_stock_sets
    lo, hi = cfg.sensor_hw_weeks

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
    rework_end = cfg.rework_share
    cals = {"eng": E, "prod": P, "fat": F}
    n_fam = dict(experience_today(cfg))
    n_appr = cfg.approvals_done_today
    f_cur = {fam: {r: 1.0 for r in cals} for fam in FAMILIES}
    r_now = rework_share_at(cfg, 0)
    rr_cur = r_now / (1 - r_now)
    se_today = {fam: 0.0 for fam in FAMILIES}
    for t, n in cfg.projects_per_year.items():
        se_today[ORDER_TYPES[t].family] += n * cfg.class_weight[ORDER_TYPES[t].cls]

    for w in range(t1 + DRAIN_WEEKS):
        bundled_arr, retro_arr = arrivals[w]

        # ---------------- learning: hours per project per family ----------------
        if cfg.learning and w >= t0:
            f_new = family_factors(cfg, n_fam)
            for fam in FAMILIES:
                for r, cal in cals.items():
                    if f_new[fam][r] < f_cur[fam][r] - 1e-4:
                        ratio = f_new[fam][r] / f_cur[fam][r]
                        for t in active:
                            if ORDER_TYPES[t.otype].family != fam:
                                continue
                            plan = getattr(t, r)
                            for k in [k for k in plan if k > w]:
                                cut = plan[k] * (1 - ratio)
                                plan[k] -= cut
                                cal.used[k] -= cut
                        f_cur[fam][r] = f_new[fam][r]
        # ---------------- rework reduction (every 4 weeks): rescale remaining prod/FAT work ----------------
        if w > t0 and (w - t0) % 4 == 0:
            r_new = rework_share_at(cfg, w - t0)
            rr_new = r_new / (1 - r_new)
            if rr_new < rr_cur - 1e-5:
                for t in active:
                    ratio = (1 + rr_new * t.g) / (1 + rr_cur * t.g)
                    for cal, plan in ((P, t.prod), (F, t.fat)):
                        for k in [k for k in plan if k > w]:
                            cut = plan[k] * (1 - ratio)
                            plan[k] -= cut
                            cal.used[k] -= cut
                rr_cur = rr_new
        if w == t1 - 1:
            rework_end = rr_cur / (1 + rr_cur)
            f_end = {fam: f_cur[fam]["prod"] for fam in FAMILIES}
            c_end = comment_probability(cfg, CASE_BY_CASE.p_comments, n_appr)

        # ---------------- bundled arrivals ----------------
        for a in bundled_arr:
            ot, u = ORDER_TYPES[a["otype"]], a["u"]
            due = w + a["lead"]
            appr = approval_for(u[0], w)
            hw = w + (lo + u[3] * (hi - lo) if ot.long_lead else cfg.stock_hw_weeks)
            fc = f_cur[ot.family]
            ew = eng_weeks(cfg, ot) * fc["eng"]
            g = rework_draw(cfg, a["z"])
            clean = unit_hours(cfg, prod_hours(cfg, ot), a["z"]) * (1 - cfg.rework_share)
            rw = 1 + rr_cur * g
            ph, fh = clean * rw * fc["prod"], clean * rw * cfg.fat_to_prod * fc["fat"]
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
            active[-1].g = g

        # ---------------- retrofit opportunities ----------------
        for a in retro_arr:
            ot, u = ORDER_TYPES[a["otype"]], a["u"]
            appr = approval_for(u[0], w)
            stock_idx = None
            hw = w + cfg.stock_hw_weeks
            if ot.long_lead:
                avail = [k for k, t_av in enumerate(stock) if t_av <= w]
                if avail:
                    stock_idx = avail[0]
                else:
                    hw = w + lo + u[3] * (hi - lo)
            fc = f_cur[ot.family]
            ew = eng_weeks(cfg, ot) * fc["eng"]
            g = rework_draw(cfg, a["z"])
            clean = unit_hours(cfg, prod_hours(cfg, ot), a["z"]) * (1 - cfg.rework_share)
            rw = 1 + rr_cur * g
            ph, fh = clean * rw * fc["prod"], clean * rw * cfg.fat_to_prod * fc["fat"]
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
            active[-1].g = g

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
                # working capital: traded sensors paid on arrival (bundled: just in time for
                # production), invoiced at delivery
                ot = ORDER_TYPES[t.otype]
                start = min(t.prod) if t.kind == "bundled" else min(t.hw_ready, min(t.prod))
                t.wc = traded_cost(cfg, ot) * cfg.discount_rate * max(t.ready - start, 0.0) / 52.0
                finished.append(t)
                if w >= t0:
                    n_fam[ot.family] += cfg.class_weight[ot.cls]
            else:
                still.append(t)
        active = still

    # ---------------- results within [t0, t1) ----------------
    years = cfg.horizon_weeks / 52.0
    inw = lambda wk: t0 <= wk < t1
    disc = lambda wk: (1.0 + cfg.discount_rate) ** (-(wk - t0) / 52.0)
    r = {"rework_end": rework_end}
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
    r["wc_cost"] = sum(t.wc for t in retro + bund) / years
    r["revenue"] = sum(t.price for t in retro + bund) / years
    for name, cal in (("eng", E), ("prod", P), ("fat", F)):
        r[f"{name}_load"] = cal.used[t0:t1].sum() / cal.cap[t0:t1].sum()
    r["eng_hours"] = E.used[t0:t1].sum() * 37.5 / years
    tot = sum(se_today.values())
    r["hours_factor_end"] = sum(f_end[f] * se_today[f] for f in FAMILIES) / tot
    for fam in FAMILIES:
        r[f"hours_factor_end_{fam}"] = f_end[fam]
    r["comment_rate_end"] = c_end
    r["prod_hours"] = P.used[t0:t1].sum() / years

    # NPV: contribution at award (own + traded margin, + premium/discount), LDs and working
    # capital at delivery, lever costs yearly
    npv = 0.0
    for t in retro + bund:
        ot = ORDER_TYPES[t.otype]
        npv += (contribution(cfg, ot) + (t.price - cs_price(cfg, ot))) * disc(t.arrival)
        npv -= (ld(cfg, t.late, t.contract_value) + t.wc) * disc(t.ready)
    n_ta = len(cfg.type_approved_societies)
    if n_ta:
        npv -= cfg.ta_cost_first + cfg.ta_cost_additional * (n_ta - 1)
    if cfg.fat_capacity_factor > 1.0:
        npv -= cfg.fat_extra_cost_upfront
    stock_value = cfg.advanced_stock_sets * traded_cost(cfg, ORDER_TYPES["b5_dp2"])
    for y in range(int(years)):
        wk = t0 + 52 * y
        yearly = stock_value * cfg.stock_carrying_rate + lever_cost_per_year(cfg)
        if cfg.fat_capacity_factor > 1.0:
            yearly += cfg.fat_extra_cost_per_year
        if wk + 52 > ta_ready:
            yearly += cfg.ta_cost_per_year * n_ta
        npv -= yearly * disc(wk)
    r["npv"] = npv
    return r


def calibrate_retrofit(cfg: Config, n_runs=40, iters=5, verbose=True):
    """Retrofit opportunities per type so that `cfg` wins `retrofit_won_target` per type."""
    target = retrofit_won_target(cfg)
    rates = {t: n / 0.6 for t, n in target.items()}
    for it in range(iters):
        runs = [simulate(cfg, 9_000_000 + i, rates) for i in range(n_runs)]
        won = {t: np.mean([r[f"won_{t}"] for r in runs]) for t in rates}
        rates = {t: rates[t] * target[t] / max(won[t], 0.05) for t in rates}
        if verbose:
            print(f"  calibration {it}: won " + ", ".join(f"{t} {won[t]:.1f}" for t in won)
                  + " -> opportunities " + ", ".join(f"{t} {rates[t]:.1f}" for t in rates), flush=True)
    return rates
