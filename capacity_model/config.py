"""Model assumptions for the control-system capacity model.

Values marked USER are from the product owner; PLACEHOLDER values are guesses to be calibrated.
Units: time in weeks, engineering in engineer-weeks (37.5 h), production/FAT in hours, money in kNOK.
"""

import math
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class OrderType:
    name: str
    n_simple: int             # simple control systems (incl. a simple top-level system)
    n_advanced: int
    n_mech: int               # mechanical units in the same (bundled) contract
    eng_max_rate: float       # max engineers in parallel on the order
    prod_max_rate: float      # max production hours/week on the order (parallel cabinets)
    fat_max_rate: float       # max FAT hours/week on the order (test benches in parallel)


# USER: single simple; 4 simple + a simple top-level system; package of 8 simple + 1 advanced.
ORDER_TYPES = {
    "single": OrderType("single", 1, 0, 1, eng_max_rate=1.0, prod_max_rate=75.0, fat_max_rate=37.5),
    "quad": OrderType("quad", 5, 0, 4, eng_max_rate=2.0, prod_max_rate=225.0, fat_max_rate=37.5),
    "package": OrderType("package", 8, 1, 8, eng_max_rate=3.0, prod_max_rate=300.0, fat_max_rate=75.0),
}


@dataclass(frozen=True)
class Approval:
    nominal_weeks: float
    p_comments: float
    comment_weeks: float


# USER: case-by-case ~3 months, 50 % get comments, ~1 month extra.
CASE_BY_CASE = Approval(nominal_weeks=13.0, p_comments=0.5, comment_weeks=4.0)
# USER: type approval ~half; PLACEHOLDER: fewer and shorter comment loops.
TYPE_APPROVED = Approval(nominal_weeks=6.5, p_comments=0.25, comment_weeks=2.0)

# USER: retrofit share by class society (assumed the same for bundled).
CLASS_SHARE = {"DNV": 0.80, "ABS": 0.10, "BV": 0.07, "LR": 0.03}


@dataclass(frozen=True)
class Config:
    # --- simulation ---
    horizon_weeks: int = 5 * 52
    warmup_weeks: int = 104
    n_runs: int = 200
    seed: int = 1

    # --- volumes per year ---
    bundled_per_year: dict = field(default_factory=lambda: {"single": 50, "quad": 20, "package": 10})  # USER
    retrofit_won_target: dict = field(default_factory=lambda: {"single": 20, "quad": 20, "package": 20})  # USER
    bundled_lead_weeks: tuple = (52.0, 156.0)   # USER: 1-3 years

    # --- engineering ---
    engineers: float = 24.0                     # USER
    availability: float = 0.80                  # USER
    working_weeks: float = 45.0                 # per year after holidays (assumption)
    eng_hours_reported: float = 23400.0         # USER: scales per-unit effort
    eng_simple_weeks: float = 1.0               # USER (before scaling)
    eng_advanced_weeks: float = 4.0             # USER (before scaling)

    # --- production ---
    prod_hours_reported: float = 32300.0        # USER
    advanced_to_simple: float = 3.5             # USER: advanced 3-4x simple (also used for FAT)
    unit_cv: float = 0.30                       # spread of hours per unit (lognormal), PLACEHOLDER
    prod_load: float = 0.85                     # PLACEHOLDER: reported / available hours
    prod_capacity_factor: float = 1.0           # lever

    # --- test / FAT (USER: currently the limiting factor) ---
    fat_simple_hours: float = 15.0              # PLACEHOLDER bench hours per simple unit
    fat_load: float = 0.92                      # PLACEHOLDER: highest load of the three
    fat_capacity_factor: float = 1.0            # lever

    # --- hardware ---
    simple_hw_weeks: float = 1.0                # USER: simple mostly in stock
    advanced_hw_weeks: tuple = (9.0, 13.0)      # USER: long-lead items 2-3 months
    advanced_stock_sets: int = 0                # lever: long-lead sets in stock for retrofit

    shipping_weeks: float = 1.0

    # --- bundled scheduling ---
    bundled_policy: str = "alap"                # USER: current practice ~as late as possible
    bundled_offset_start: float = 45.0          # 'fixed_offset': engineering starts this long before delivery
    pull_forward: bool = False                  # lever: fill idle capacity with bundled work

    # --- retrofit market (PLACEHOLDER: no quote history) ---
    win_p_max: float = 0.70
    win_L50: float = 30.0
    win_width: float = 3.0
    express_max_weeks: float = 17.0             # <= ~4 months
    standard_max_weeks: float = 26.0            # <= 6 months
    express_price_premium: float = 0.15         # PLACEHOLDER ("high")
    long_price_discount: float = 0.05
    max_quote_weeks: float = 52.0
    quote_safety_weeks: float = 1.0
    quote_comment_allowance: bool = False       # quote approval incl. the comment loop
    today_quote_quad_weeks: float = 27.0        # PLACEHOLDER: lead time quoted today for a quad retrofit
                                                # (anchors the hand check; replace with the real figure)

    # --- class approval ---
    type_approved_societies: tuple = ()
    type_approval_ready_week: int = 52
    type_approval_eng_weeks: float = 20.0       # per society, PLACEHOLDER
    ta_cost_first: float = 3000.0               # PLACEHOLDER kNOK
    ta_cost_additional: float = 1500.0          # PLACEHOLDER kNOK per further society
    ta_cost_per_year: float = 400.0             # PLACEHOLDER kNOK per society

    # --- learning (new product, wrinkles being ironed out) ---
    learning: bool = True
    learning_rates: dict = field(default_factory=lambda: {"eng": 0.90, "prod": 0.90, "fat": 0.90})  # USER
    learning_floor: float = 0.50                # USER: hours can fall to 50 % of today at best
    units_built_today: float = 225.0            # USER: ~200-250 simple-equivalents built so far
    comment_learning_rate: float = 0.60         # comment rate x0.6 per doubling of approvals (USER: "fast")
    comment_floor: float = 0.075                # USER: 5-10 % realistic target
    approvals_done_today: float = 50.0          # PLACEHOLDER: ~225 units / ~4.4 units per order

    # --- economics (kNOK) ---
    price_simple: float = 300.0                 # USER
    price_advanced: float = 10000.0             # USER
    price_mech_unit: float = 20000.0            # PLACEHOLDER
    margin: float = 0.20                        # USER
    ld_per_week: float = 0.005                  # Orgalim: 0.5 % per full week
    ld_cap: float = 0.075                       # Orgalim: max 7.5 %
    long_lead_share_of_cost: float = 0.75       # USER: 70-80 % of advanced cost
    stock_carrying_rate: float = 0.15           # PLACEHOLDER: capital + obsolescence per year
    prod_cost_per_hour: float = 0.8             # PLACEHOLDER kNOK/h for extra production capacity
    fat_extra_cost_upfront: float = 2000.0      # PLACEHOLDER: extra test bench
    fat_extra_cost_per_year: float = 1500.0     # PLACEHOLDER: ~1.5 FTE test staff
    discount_rate: float = 0.08


# ---------------------------------------------------------------- derived quantities
def _units(cfg, adv_weight):
    tot = 0.0
    for vol in (cfg.bundled_per_year, cfg.retrofit_won_target):
        for t, n in vol.items():
            ot = ORDER_TYPES[t]
            tot += n * (ot.n_simple + adv_weight * ot.n_advanced)
    return tot


def eng_scale(cfg: Config) -> float:
    """Scale per-unit engineering so the expected volume reproduces the reported hours."""
    base = _units(cfg, cfg.eng_advanced_weeks / cfg.eng_simple_weeks) * cfg.eng_simple_weeks
    return cfg.eng_hours_reported / 37.5 / base


def eng_weeks(cfg: Config, ot: OrderType) -> float:
    return eng_scale(cfg) * (ot.n_simple * cfg.eng_simple_weeks + ot.n_advanced * cfg.eng_advanced_weeks)


def eng_capacity_per_week(cfg: Config) -> float:
    return cfg.engineers * cfg.availability * cfg.working_weeks / 52.0


def prod_simple_hours(cfg: Config) -> float:
    return cfg.prod_hours_reported / _units(cfg, cfg.advanced_to_simple)


def prod_capacity_per_week(cfg: Config) -> float:
    return cfg.prod_hours_reported / cfg.prod_load / 52.0 * cfg.prod_capacity_factor


def fat_capacity_per_week(cfg: Config) -> float:
    hours = cfg.fat_simple_hours * _units(cfg, cfg.advanced_to_simple)
    return hours / cfg.fat_load / 52.0 * cfg.fat_capacity_factor


def learning_factor(rate: float, n: float, n0: float, floor: float) -> float:
    """Hours per unit relative to today after n cumulative units (n0 today): max(floor, (n/n0)^-b)."""
    b = -math.log2(rate)
    return max(floor, (max(n, n0) / n0) ** (-b))


def comment_probability(cfg: Config, p0: float, n_approvals: float) -> float:
    if not cfg.learning:
        return p0
    f = learning_factor(cfg.comment_learning_rate, n_approvals, cfg.approvals_done_today, 0.0)
    return max(min(cfg.comment_floor, p0), p0 * f)


def cs_price(cfg: Config, ot: OrderType) -> float:
    return ot.n_simple * cfg.price_simple + ot.n_advanced * cfg.price_advanced


# ---------------------------------------------------------------- scenarios
BASE = Config()                                  # current practice: bundled ALAP, case-by-case approval
BUF = replace(BASE, pull_forward=True)
L3 = replace(BUF, type_approved_societies=("DNV",))
L4 = replace(L3, advanced_stock_sets=1)
L5 = replace(L4, type_approved_societies=("DNV", "ABS"))
L6 = replace(L5, fat_capacity_factor=1.25)

SCENARIOS = {
    "0 bundled asap (ref.)": replace(BASE, bundled_policy="asap"),
    "1 baseline": BASE,
    "1b quote incl. comments": replace(BASE, quote_comment_allowance=True),
    "2 +pull bundled fwd": BUF,
    "2b +adv. stock, no TA": replace(BUF, advanced_stock_sets=1),
    "3 +TA DNV": L3,
    "4 +1 adv. stock set": L4,
    "5 +TA ABS": L5,
    "5b +10% prod. (not FAT)": replace(L5, prod_capacity_factor=1.10),
    "6 +25% FAT capacity": L6,
    "7 +FAT +production": replace(L6, prod_capacity_factor=1.10),
}
