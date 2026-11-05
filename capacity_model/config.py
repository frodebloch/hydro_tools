"""Model assumptions for the control-system capacity model.

Values marked USER are from the product owner; PLACEHOLDER values are guesses to be calibrated.
Units: time in weeks, engineering in engineer-weeks (37.5 h), production/FAT in hours, money in kNOK.

Structure
---------
* SUBTYPES: the real product sub types with their 2026 volume (projects) and class.
* Project classes (ORDER_TYPES): simple / medium / advanced. A project's hours are
  class weight x hours per "simple-equivalent"; the hours per simple-equivalent are calibrated
  so that the 2026 volume reproduces the hours reported on the projects.
* Price per project is split into own content and traded (bought-in) sensors, each with its own
  margin. Traded sensors have a long lead time and tie up working capital until delivery.
"""

import math
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class OrderType:
    name: str
    eng_max_rate: float       # max engineers in parallel on one project
    prod_max_rate: float      # max production hours/week on one project
    fat_max_rate: float       # max FAT hours/week on one project
    long_lead: bool           # needs long-lead traded sensors (else hardware from stock)
    n_mech: int = 1           # mechanical units in the same (bundled) contract, PLACEHOLDER


ORDER_TYPES = {
    "simple": OrderType("simple", eng_max_rate=1.0, prod_max_rate=75.0, fat_max_rate=37.5, long_lead=False),
    "medium": OrderType("medium", eng_max_rate=2.0, prod_max_rate=150.0, fat_max_rate=37.5, long_lead=False),
    "advanced": OrderType("advanced", eng_max_rate=3.0, prod_max_rate=225.0, fat_max_rate=75.0, long_lead=True),
}

# USER: 2026 projects (to date + planned) per sub type and class. BruCon-4 is outgoing -> excluded.
SUBTYPES = {
    "PTC - Thruster": ("simple", 36),
    "CMS": ("simple", 26),                    # class assumed
    "PTC - Other": ("simple", 6),             # class assumed
    "PTC - Shaft propulsion": ("medium", 18),
    "BruCon-5 DP0": ("medium", 7),
    "PTC - Azimuth propulsion": ("medium", 5),
    "BruCon-5 JS": ("medium", 3),
    "BruCon-5 AC": ("medium", 2),
    "BruCon-5 DP2": ("advanced", 3),
}
EXCLUDED_FROM_HOURS = {"BruCon-4": ("advanced", 1)}   # hours booked in 2026, removed from calibration


def class_volumes():
    vol = {c: 0 for c in ORDER_TYPES}
    for cls, n in SUBTYPES.values():
        vol[cls] += n
    return vol


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

    # --- volumes per year (projects) ---
    projects_per_year: dict = field(default_factory=class_volumes)                 # USER (2026)
    retrofit_share: dict = field(default_factory=lambda: {"simple": 0.4, "medium": 0.4, "advanced": 0.4})  # PLACEHOLDER
    bundled_lead_weeks: tuple = (52.0, 156.0)   # USER: 1-3 years
    # market growth: retrofit opportunities per class x multiplier, ramped linearly over ramp_weeks
    retrofit_ramp: dict = field(default_factory=dict)   # e.g. {"advanced": 10}
    ramp_weeks: int = 156                       # reached by year 3

    # --- hours per project: class weight x hours per simple-equivalent (calibrated) ---
    class_weight: dict = field(default_factory=lambda: {"simple": 1.0, "medium": 2.0, "advanced": 3.5})  # USER/PLACEHOLDER
    eng_hours_reported: float = 23400.0         # USER: booked on the 2026 projects (incl. BruCon-4)
    prod_hours_reported: float = 32300.0        # USER
    fat_to_prod: float = 0.286                  # PLACEHOLDER: FAT hours per production hour
    unit_cv: float = 0.30                       # spread of hours per project (lognormal), PLACEHOLDER

    # --- capacity ---
    engineers: float = 24.0                     # USER
    availability: float = 0.80                  # USER
    working_weeks: float = 45.0                 # per year after holidays (assumption)
    prod_load: float = 0.85                     # PLACEHOLDER: reported / available hours
    prod_capacity_factor: float = 1.0           # lever
    fat_load: float = 0.92                      # PLACEHOLDER: FAT is the bottleneck (USER)
    fat_capacity_factor: float = 1.0            # lever

    # --- hardware ---
    stock_hw_weeks: float = 1.0                 # USER: simple/medium hardware mostly in stock
    sensor_hw_weeks: tuple = (9.0, 13.0)        # USER: traded sensors 9-13 weeks (varies)
    advanced_stock_sets: int = 0                # lever: sensor sets in stock for retrofit

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
    express_price_premium: float = 0.15         # PLACEHOLDER ("high"), on the whole price
    long_price_discount: float = 0.05
    max_quote_weeks: float = 52.0
    quote_safety_weeks: float = 1.0
    quote_comment_allowance: bool = False       # quote approval incl. the comment loop
    anchor_type: str = "medium"                 # hand check anchor
    today_quote_weeks: float = 29.5             # PLACEHOLDER (= model baseline today): replace with the real quoted lead time, medium retrofit

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
    units_built_today: float = 60.0             # PLACEHOLDER: simple-equivalents built so far (~0.4 yr)
    comment_learning_rate: float = 0.60         # comment rate x0.6 per doubling of approvals (USER: "fast")
    comment_floor: float = 0.075                # USER: 5-10 % realistic target
    approvals_done_today: float = 40.0          # PLACEHOLDER: projects approved so far

    # --- economics (kNOK) ---
    price: dict = field(default_factory=lambda: {"simple": 300.0, "medium": 1000.0, "advanced": 10000.0})  # USER
    traded_share: dict = field(default_factory=lambda: {"simple": 0.0, "medium": 0.0, "advanced": 0.75})  # share of price; USER ~70-80 % for DP2
    margin_own: float = 0.20                    # USER
    margin_traded: float = 0.20                 # USER: own variable, 20 % for now
    price_mech_unit: float = 20000.0            # PLACEHOLDER (LD basis for bundled)
    ld_per_week: float = 0.005                  # Orgalim: 0.5 % per full week
    ld_cap: float = 0.075                       # Orgalim: max 7.5 %
    stock_carrying_rate: float = 0.15           # PLACEHOLDER: capital + obsolescence per year
    prod_cost_per_hour: float = 0.8             # PLACEHOLDER kNOK/h for extra production capacity
    fat_extra_cost_upfront: float = 2000.0      # PLACEHOLDER: extra test bench
    fat_extra_cost_per_year: float = 1500.0     # PLACEHOLDER: ~1.5 FTE test staff
    discount_rate: float = 0.08                 # also used as cost of working capital


# ---------------------------------------------------------------- derived quantities
def bundled_per_year(cfg: Config) -> dict:
    return {t: n * (1 - cfg.retrofit_share[t]) for t, n in cfg.projects_per_year.items()}


def retrofit_won_target(cfg: Config) -> dict:
    return {t: n * cfg.retrofit_share[t] for t, n in cfg.projects_per_year.items()}


def simple_equiv(cfg: Config, t: str) -> float:
    return cfg.class_weight[t]


def workload_se(cfg: Config) -> float:
    """Simple-equivalents per year at today's volume."""
    return sum(n * cfg.class_weight[t] for t, n in cfg.projects_per_year.items())


def _scope(cfg: Config) -> float:
    """Share of the reported hours that belongs to the modelled projects (BruCon-4 removed)."""
    w = workload_se(cfg)
    excl = sum(n * cfg.class_weight[c] for c, n in EXCLUDED_FROM_HOURS.values())
    return w / (w + excl)


def eng_hours_per_se(cfg: Config) -> float:
    return cfg.eng_hours_reported * _scope(cfg) / workload_se(cfg)


def prod_hours_per_se(cfg: Config) -> float:
    return cfg.prod_hours_reported * _scope(cfg) / workload_se(cfg)


def eng_weeks(cfg: Config, ot: OrderType) -> float:
    return eng_hours_per_se(cfg) * cfg.class_weight[ot.name] / 37.5


def prod_hours(cfg: Config, ot: OrderType) -> float:
    """Mean production hours per project."""
    return prod_hours_per_se(cfg) * cfg.class_weight[ot.name]


def fat_hours(cfg: Config, ot: OrderType) -> float:
    return prod_hours(cfg, ot) * cfg.fat_to_prod


def eng_capacity_per_week(cfg: Config) -> float:
    return cfg.engineers * cfg.availability * cfg.working_weeks / 52.0


def prod_capacity_per_week(cfg: Config) -> float:
    return cfg.prod_hours_reported * _scope(cfg) / cfg.prod_load / 52.0 * cfg.prod_capacity_factor


def fat_capacity_per_week(cfg: Config) -> float:
    return cfg.prod_hours_reported * _scope(cfg) * cfg.fat_to_prod / cfg.fat_load / 52.0 * cfg.fat_capacity_factor


def cs_price(cfg: Config, ot: OrderType) -> float:
    return cfg.price[ot.name]


def contribution(cfg: Config, ot: OrderType, price: float = None) -> float:
    """Contribution of a project: own part x own margin + traded part x traded margin."""
    p = cfg.price[ot.name] if price is None else price
    s = cfg.traded_share[ot.name]
    return p * ((1 - s) * cfg.margin_own + s * cfg.margin_traded)


def traded_cost(cfg: Config, ot: OrderType) -> float:
    return cfg.price[ot.name] * cfg.traded_share[ot.name] * (1 - cfg.margin_traded)


def learning_factor(rate: float, n: float, n0: float, floor: float) -> float:
    """Hours per unit relative to today after n cumulative units (n0 today): max(floor, (n/n0)^-b)."""
    b = -math.log2(rate)
    return max(floor, (max(n, n0) / n0) ** (-b))


def comment_probability(cfg: Config, p0: float, n_approvals: float) -> float:
    if not cfg.learning:
        return p0
    f = learning_factor(cfg.comment_learning_rate, n_approvals, cfg.approvals_done_today, 0.0)
    return max(min(cfg.comment_floor, p0), p0 * f)


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
    "2b +sensor stock, no TA": replace(BUF, advanced_stock_sets=1),
    "3 +TA DNV": L3,
    "4 +1 sensor stock set": L4,
    "5 +TA ABS": L5,
    "5b +10% prod. (not FAT)": replace(L5, prod_capacity_factor=1.10),
    "6 +25% FAT capacity": L6,
    "7 +FAT +production": replace(L6, prod_capacity_factor=1.10),
}
