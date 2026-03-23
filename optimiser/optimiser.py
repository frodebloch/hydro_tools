"""
Propulsion optimiser.

Two optimisation modes:

1. Minimum shaft power:  Given required thrust T and advance speed Va,
   find the (pitch, n) combination that minimises shaft power, subject
   to constraints on max rpm, max torque, and max power.

2. Minimum fuel consumption:  Given required thrust T, advance speed Va,
   and an engine model (with auxiliary load), find the (pitch, n)
   combination that minimises total fuel consumption rate.

Both sweep a fine pitch grid (using the PCHIP-interpolated propeller
model). For each pitch, the rpm that delivers the required thrust is
found by bisection. The operating point is then evaluated against the
objective and constraints.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from propeller_model import CSeriesPropeller, PchipInterpolator


# ============================================================
# Engine model interface
# ============================================================

class EngineModel(Protocol):
    """Protocol for engine fuel consumption models.

    An engine model maps (power, rpm) to fuel consumption rate.
    """

    def fuel_rate(self, power_kw: float, rpm: float) -> float:
        """Fuel consumption rate [kg/h or g/kWh -- be consistent].

        Parameters
        ----------
        power_kw : float
            Total engine brake power [kW] (propeller + auxiliaries).
        rpm : float
            Engine speed [rpm]. May be relevant for geared installations
            or if SFOC depends on engine speed.
        """
        ...

    def max_power(self, rpm: float) -> float:
        """Maximum available brake power [kW] at given engine rpm."""
        ...

    def max_rpm(self) -> float:
        """Maximum engine speed [rpm]."""
        ...

    def min_rpm(self) -> float:
        """Minimum engine speed [rpm]."""
        ...


@dataclass
class SimpleDieselEngine:
    """Simple diesel engine model with a quadratic SFOC curve.

    SFOC (Specific Fuel Oil Consumption) varies with load fraction:
        sfoc(load) = sfoc_min * (1 + k * (load - load_opt)^2)

    where load = P / P_max.

    This gives the classic "bathtub" curve with minimum SFOC at
    the optimal load fraction (typically 75-85% MCR).

    Parameters
    ----------
    max_power_kw : float
        Maximum continuous rating (MCR) [kW].
    max_engine_rpm : float
        Maximum engine rpm.
    min_engine_rpm : float
        Minimum stable engine rpm.
    sfoc_min : float
        Minimum SFOC [g/kWh] at optimal load.
    load_optimal : float
        Load fraction at minimum SFOC (0-1), default 0.80.
    sfoc_curvature : float
        Curvature parameter k for the SFOC parabola, default 0.5.
    """

    max_power_kw: float
    max_engine_rpm: float
    min_engine_rpm: float = 0.0
    sfoc_min: float = 185.0        # g/kWh -- typical modern marine diesel
    load_optimal: float = 0.80
    sfoc_curvature: float = 0.5

    def fuel_rate(self, power_kw: float, rpm: float) -> float:
        """Fuel consumption rate [g/h].

        Returns the total fuel rate (not specific), so that the optimiser
        can directly compare operating points at different power levels.
        """
        if power_kw <= 0:
            return 0.0
        load = power_kw / self.max_power_kw
        sfoc = self.sfoc_min * (1.0 + self.sfoc_curvature * (load - self.load_optimal) ** 2)
        return sfoc * power_kw  # g/h

    def max_power(self, rpm: float) -> float:
        """Maximum power at given rpm (simple: constant up to max_rpm)."""
        if rpm > self.max_engine_rpm:
            return 0.0
        return self.max_power_kw

    def max_rpm(self) -> float:
        return self.max_engine_rpm

    def min_rpm(self) -> float:
        return self.min_engine_rpm


@dataclass
class MuzzleDiagramEngine:
    """Engine model based on a digitised muzzle diagram.

    Uses a 2D lookup table of SFOC(engine_rpm, brake_power) with
    bilinear interpolation, plus a power limit envelope as a function
    of engine rpm.

    The SFOC field is defined on an irregular grid of (rpm, power) points
    with associated SFOC values.  For interpolation, the data is organised
    onto a regular grid using the rpm and power breakpoints.

    Parameters
    ----------
    name : str
        Engine name / description.
    max_power_kw : float
        Maximum continuous rating [kW].
    max_engine_rpm : float
        Maximum engine speed [rpm].
    min_engine_rpm : float
        Minimum stable engine speed [rpm].
    power_limit_rpm : list of float
        Engine RPM breakpoints for the power limit envelope.
    power_limit_kw : list of float
        Maximum brake power [kW] at each RPM breakpoint.
    sfoc_rpm : list of float
        RPM breakpoints for the SFOC grid.
    sfoc_power_kw : list of float
        Power breakpoints for the SFOC grid [kW].
    sfoc_table : 2D array-like, shape (n_rpm, n_power)
        SFOC values [g/kWh] at each (rpm, power) grid point.
        Use 0 or NaN for points outside the operating envelope.
    """

    name: str = ""
    max_power_kw: float = 0.0
    max_engine_rpm: float = 0.0
    min_engine_rpm: float = 0.0

    # Power limit envelope
    power_limit_rpm: list = field(default_factory=list)
    power_limit_kw: list = field(default_factory=list)

    # SFOC grid
    sfoc_rpm: list = field(default_factory=list)
    sfoc_power_kw: list = field(default_factory=list)
    sfoc_table: list = field(default_factory=list)

    def __post_init__(self):
        self._pl_rpm = np.array(self.power_limit_rpm, dtype=float)
        self._pl_kw = np.array(self.power_limit_kw, dtype=float)
        self._sfoc_rpm = np.array(self.sfoc_rpm, dtype=float)
        self._sfoc_power = np.array(self.sfoc_power_kw, dtype=float)
        self._sfoc_grid = np.array(self.sfoc_table, dtype=float)

    def max_power(self, rpm: float) -> float:
        """Maximum brake power at given engine RPM, from the limit envelope."""
        if rpm < self._pl_rpm[0] or rpm > self._pl_rpm[-1]:
            return 0.0
        return float(np.interp(rpm, self._pl_rpm, self._pl_kw))

    def max_rpm(self) -> float:
        return self.max_engine_rpm

    def min_rpm(self) -> float:
        return self.min_engine_rpm

    def sfoc(self, power_kw: float, rpm: float) -> float:
        """Interpolate SFOC [g/kWh] from the muzzle diagram grid.

        Uses bilinear interpolation on the (rpm, power) grid.
        Clamps to grid boundaries.
        """
        # Find position in rpm axis
        rpm_arr = self._sfoc_rpm
        pwr_arr = self._sfoc_power
        grid = self._sfoc_grid

        # Clamp
        rpm_c = max(rpm_arr[0], min(rpm_arr[-1], rpm))
        pwr_c = max(pwr_arr[0], min(pwr_arr[-1], power_kw))

        # Find bracketing indices for rpm
        i = int(np.searchsorted(rpm_arr, rpm_c, side='right')) - 1
        i = max(0, min(i, len(rpm_arr) - 2))

        # Find bracketing indices for power
        j = int(np.searchsorted(pwr_arr, pwr_c, side='right')) - 1
        j = max(0, min(j, len(pwr_arr) - 2))

        # Bilinear weights
        rpm_frac = (rpm_c - rpm_arr[i]) / (rpm_arr[i + 1] - rpm_arr[i])
        pwr_frac = (pwr_c - pwr_arr[j]) / (pwr_arr[j + 1] - pwr_arr[j])

        # Bilinear interpolation
        s00 = grid[i, j]
        s10 = grid[i + 1, j]
        s01 = grid[i, j + 1]
        s11 = grid[i + 1, j + 1]

        s0 = s00 + (s10 - s00) * rpm_frac
        s1 = s01 + (s11 - s01) * rpm_frac
        return float(s0 + (s1 - s0) * pwr_frac)

    def fuel_rate(self, power_kw: float, rpm: float) -> float:
        """Total fuel consumption rate [g/h]."""
        if power_kw <= 0:
            return 0.0
        return self.sfoc(power_kw, rpm) * power_kw


def make_man_l27_38() -> MuzzleDiagramEngine:
    """Create engine model for MAN L27/38-800rpm 365kW/cyl.

    Digitised from muzzle diagram 2591/P37206 (Tm=7.6m) (BF2).
    Per-propeller values (the diagram shows per-propeller power).

    Engine: 8-cylinder L27/38, MCR 2920 kW at 800 rpm.
    Gearbox ratio: 800/117.6 = 6.803.
    """
    # Power limit envelope (red curve on the diagram)
    # Digitised from the MCR line. At full rpm it's 2920 kW, falling
    # at lower rpm roughly following a propeller-law shape with a floor.
    power_limit_rpm = [
        480, 500, 520, 540, 560, 580, 600, 620, 640, 660,
        680, 700, 720, 740, 760, 780, 800, 820,
    ]
    power_limit_kw = [
        1750, 1850, 1960, 2070, 2180, 2290, 2400, 2510, 2610, 2700,
        2770, 2820, 2860, 2880, 2900, 2910, 2920, 2920,
    ]

    # SFOC grid digitised from the blue contour curves.
    #
    # The contours on the diagram are:
    #   181-182 g/kWh  -- innermost (sweet spot)
    #   182-183 g/kWh
    #   183-185 g/kWh
    #   185-187 g/kWh  -- low-load region
    #   187-190 g/kWh  -- corners / edges
    #
    # I build a regular grid and assign SFOC values by reading where
    # each grid point falls relative to the contours.
    #
    # RPM breakpoints (engine rpm):
    sfoc_rpm = [480, 520, 560, 600, 640, 680, 720, 760, 800]
    # Power breakpoints (kW):
    sfoc_power = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800,
                  2000, 2200, 2400, 2600, 2800, 2920]

    # SFOC grid [rpm_index, power_index] in g/kWh
    # Read from the contour positions on the diagram.
    # Rows = rpm (480..800), Cols = power (200..2920)
    sfoc_table = [
        # 480 rpm
        [195, 192, 189, 187, 186, 185, 184, 184, 185, 186, 188, 190, 192, 195, 197],
        # 520 rpm
        [194, 191, 188, 186, 185, 184, 183, 183, 183, 184, 186, 188, 190, 193, 195],
        # 560 rpm
        [193, 190, 187, 185, 184, 183, 182, 182, 182, 183, 184, 186, 188, 191, 193],
        # 600 rpm
        [193, 189, 187, 184, 183, 182, 181, 181, 181, 182, 183, 185, 187, 189, 192],
        # 640 rpm
        [193, 189, 187, 184, 183, 182, 181, 181, 181, 182, 183, 184, 186, 189, 191],
        # 680 rpm
        [194, 190, 187, 185, 183, 182, 182, 181, 182, 182, 183, 185, 187, 189, 191],
        # 720 rpm
        [195, 191, 188, 186, 184, 183, 182, 182, 182, 183, 184, 185, 187, 189, 191],
        # 760 rpm
        [196, 192, 189, 187, 185, 184, 183, 183, 183, 184, 185, 186, 188, 190, 192],
        # 800 rpm
        [197, 193, 190, 188, 186, 185, 184, 184, 184, 185, 186, 187, 189, 190, 192],
    ]

    return MuzzleDiagramEngine(
        name="MAN L27/38-800rpm 8cyl (per propeller)",
        max_power_kw=2920.0,
        max_engine_rpm=800.0,
        min_engine_rpm=480.0,
        power_limit_rpm=power_limit_rpm,
        power_limit_kw=power_limit_kw,
        sfoc_rpm=sfoc_rpm,
        sfoc_power_kw=sfoc_power,
        sfoc_table=sfoc_table,
    )


# ============================================================
# Operating point result
# ============================================================

@dataclass
class OperatingPoint:
    """Result of an optimisation query."""
    pitch: float         # P/D ratio
    n: float             # shaft speed [rev/s]
    rpm: float           # shaft speed [rpm]
    Va: float            # advance velocity [m/s]
    thrust: float        # thrust [N]
    torque: float        # torque [Nm]
    power: float         # shaft power [W]
    power_kw: float      # shaft power [kW]
    eta0: float          # open-water efficiency
    fuel_rate: Optional[float] = None  # fuel consumption [g/h], if engine model provided
    engine_power_kw: Optional[float] = None  # total engine power including aux [kW]
    engine_rpm: Optional[float] = None  # engine speed [rpm], if geared
    sfoc: Optional[float] = None  # specific fuel consumption [g/kWh]
    found: bool = True


# ============================================================
# Solvers
# ============================================================

def _find_n_for_thrust(
    prop: CSeriesPropeller,
    pitch: float,
    Va: float,
    T_required: float,
    n_min: float = 0.1,
    n_max: float = 40.0,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> Optional[float]:
    """Find shaft speed n [rev/s] that produces the required thrust.

    Uses bisection, matching the approach in the brucon C++ code.
    Returns None if the required thrust cannot be achieved within bounds.
    """
    # Check that the thrust at n_max exceeds required (monotonic in n for fixed pitch)
    T_at_max = prop.thrust(pitch, n_max, Va)
    T_at_min = prop.thrust(pitch, n_min, Va)

    # For positive thrust requirement, thrust should increase with n
    if T_required > 0:
        if T_at_max < T_required:
            return None  # cannot achieve required thrust even at max rpm
        if T_at_min > T_required:
            return n_min  # already exceeds at minimum
    else:
        # Negative thrust (reversing) -- thrust becomes more negative with higher |pitch|
        # This is less common for the power optimiser, handle simply
        if T_at_max > T_required and T_at_min > T_required:
            return None

    lo, hi = n_min, n_max
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        T_mid = prop.thrust(pitch, mid, Va)
        if abs(T_mid - T_required) < tol * abs(T_required + 1e-10):
            return mid
        if T_mid < T_required:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def _pitch_sweep_range(
    prop: CSeriesPropeller,
    T_required: float,
    pitch_range: Optional[tuple[float, float]] = None,
    pitch_step: float = 0.005,
) -> np.ndarray:
    """Generate pitch values for the optimiser sweep.

    Uses a fine uniform grid over the relevant range of the pitch table,
    exploiting the PCHIP-interpolated propeller model.
    """
    pitches = prop.pitch_table

    if pitch_range is not None:
        p_min, p_max = pitch_range
    elif T_required > 0:
        p_min = 0.0
        p_max = float(pitches[-1])
    else:
        p_min = float(pitches[0])
        p_max = 0.0

    # Clamp to tabulated range (PCHIP clamps, but no point sweeping outside)
    p_min = max(p_min, float(pitches[0]))
    p_max = min(p_max, float(pitches[-1]))

    return np.arange(p_min, p_max + pitch_step * 0.5, pitch_step)


def find_optimal_operating_point(
    prop: CSeriesPropeller,
    Va: float,
    T_required: float,
    max_rpm: float = 300.0,
    max_torque: float = float("inf"),
    max_power: float = float("inf"),
    min_rpm: float = 0.0,
    pitch_range: Optional[tuple[float, float]] = None,
    pitch_step: float = 0.005,
) -> OperatingPoint:
    """Find the pitch/rpm combination that minimises shaft power.

    Uses a two-pass coarse-then-fine search.  The coarse pass sweeps at
    10× the pitch step, then a fine pass refines around the best region.
    For each pitch, the rpm needed to deliver T_required is found by
    bisection. The combination with the lowest power that satisfies all
    constraints is returned.

    Parameters
    ----------
    prop : CSeriesPropeller
        Propeller model.
    Va : float
        Advance velocity [m/s].
    T_required : float
        Required thrust [N].
    max_rpm : float
        Maximum shaft speed [rpm].
    max_torque : float
        Maximum shaft torque [Nm].
    max_power : float
        Maximum shaft power [W].
    min_rpm : float
        Minimum shaft speed [rpm].
    pitch_range : tuple, optional
        (min_pitch, max_pitch) P/D limits. If None, uses [0, max] for
        positive thrust or [min, 0] for negative thrust.
    pitch_step : float
        Pitch grid spacing for the fine sweep, default 0.005.

    Returns
    -------
    OperatingPoint
        The optimal operating point, or one with found=False if no
        feasible point exists.
    """
    n_max = max_rpm / 60.0
    n_min = min_rpm / 60.0

    def _evaluate(pitch):
        """Evaluate power at a given pitch, or return None."""
        n = _find_n_for_thrust(prop, pitch, Va, T_required,
                               n_min=max(n_min, 0.1), n_max=n_max)
        if n is None:
            return None

        rpm = n * 60.0
        if rpm > max_rpm or rpm < min_rpm:
            return None

        Q = prop.torque(pitch, n, Va)
        P = Q * 2.0 * math.pi * n

        if abs(Q) > max_torque:
            return None
        if abs(P) > max_power:
            return None
        if n < 1e-9:
            return None

        T_actual = prop.thrust(pitch, n, Va)
        return (P, pitch, n, rpm, Q, T_actual)

    # --- Coarse pass: 10× step ---
    coarse_step = pitch_step * 10.0
    coarse_grid = _pitch_sweep_range(prop, T_required, pitch_range, coarse_step)

    best_power = float("inf")
    best_coarse_pitch = None
    for pitch in coarse_grid:
        result = _evaluate(pitch)
        if result is not None and result[0] < best_power:
            best_power = result[0]
            best_coarse_pitch = pitch

    if best_coarse_pitch is None:
        fine_grid = _pitch_sweep_range(prop, T_required, pitch_range, pitch_step)
    else:
        full_range = _pitch_sweep_range(prop, T_required, pitch_range, pitch_step)
        p_lo = best_coarse_pitch - coarse_step
        p_hi = best_coarse_pitch + coarse_step
        fine_grid = full_range[(full_range >= p_lo - 1e-9) &
                               (full_range <= p_hi + 1e-9)]

    best_power = float("inf")
    best_point = None
    for pitch in fine_grid:
        result = _evaluate(pitch)
        if result is None:
            continue
        P, p, n, rpm, Q, T_actual = result
        if P < best_power:
            best_power = P
            best_point = OperatingPoint(
                pitch=p,
                n=n,
                rpm=rpm,
                Va=Va,
                thrust=T_actual,
                torque=Q,
                power=P,
                power_kw=P / 1000.0,
                eta0=prop.eta0(p, n, Va),
            )

    if best_point is None:
        return OperatingPoint(
            pitch=0.0, n=0.0, rpm=0.0, Va=Va,
            thrust=0.0, torque=0.0, power=0.0, power_kw=0.0,
            eta0=0.0, found=False,
        )
    return best_point


def find_min_fuel_operating_point(
    prop: CSeriesPropeller,
    Va: float,
    T_required: float,
    engine: EngineModel,
    gear_ratio: float = 1.0,
    shaft_efficiency: float = 0.97,
    auxiliary_power_kw: float = 0.0,
    max_torque: float = float("inf"),
    pitch_range: Optional[tuple[float, float]] = None,
    pitch_step: float = 0.005,
    engine_rpm_min: Optional[float] = None,
    engine_rpm_max: Optional[float] = None,
) -> OperatingPoint:
    """Find the pitch/rpm combination that minimises fuel consumption.

    Uses a two-pass coarse-then-fine search.  The coarse pass sweeps at
    10× the pitch step to locate the region of minimum fuel, then a fine
    pass refines within ±1 coarse step at the requested resolution.

    For each candidate pitch, finds the rpm to deliver T_required via
    bisection, then computes the total engine load (propeller shaft power /
    shaft_eff + auxiliary load) and queries the engine model for fuel rate.

    Parameters
    ----------
    prop : CSeriesPropeller
        Propeller model.
    Va : float
        Advance velocity [m/s].
    T_required : float
        Required thrust [N].
    engine : EngineModel
        Engine fuel consumption model.
    gear_ratio : float
        Gearbox ratio (engine_rpm = gear_ratio * shaft_rpm), default 1.0.
    shaft_efficiency : float
        Mechanical efficiency of the shaft line, default 0.97.
    auxiliary_power_kw : float
        Additional electrical/auxiliary load on the engine [kW], default 0.
    max_torque : float
        Maximum shaft torque [Nm].
    pitch_range : tuple, optional
        (min_pitch, max_pitch) P/D limits.
    pitch_step : float
        Pitch grid spacing for the fine sweep, default 0.005.
    engine_rpm_min : float, optional
        Override minimum engine RPM (e.g. for shaft generator frequency
        constraint).  If None, uses engine.min_rpm().
    engine_rpm_max : float, optional
        Override maximum engine RPM.  If None, uses engine.max_rpm().

    Returns
    -------
    OperatingPoint
        The operating point with minimum fuel consumption.
    """
    eff_min_engine_rpm = engine_rpm_min if engine_rpm_min is not None else engine.min_rpm()
    eff_max_engine_rpm = engine_rpm_max if engine_rpm_max is not None else engine.max_rpm()
    max_shaft_rpm = eff_max_engine_rpm / gear_ratio
    min_shaft_rpm = eff_min_engine_rpm / gear_ratio
    n_max = max_shaft_rpm / 60.0
    n_min = min_shaft_rpm / 60.0

    def _evaluate(pitch):
        """Evaluate fuel rate at a given pitch, or return None."""
        n = _find_n_for_thrust(prop, pitch, Va, T_required,
                               n_min=max(n_min, 0.1), n_max=n_max)
        if n is None:
            return None

        shaft_rpm = n * 60.0
        engine_rpm = shaft_rpm * gear_ratio
        if engine_rpm > eff_max_engine_rpm or engine_rpm < eff_min_engine_rpm:
            return None

        Q = prop.torque(pitch, n, Va)
        if abs(Q) > max_torque:
            return None

        P_shaft = Q * 2.0 * math.pi * n  # W
        P_shaft_kw = P_shaft / 1000.0

        P_engine_kw = P_shaft_kw / shaft_efficiency + auxiliary_power_kw

        if P_engine_kw > engine.max_power(engine_rpm):
            return None
        if P_engine_kw <= 0:
            return None

        fuel = engine.fuel_rate(P_engine_kw, engine_rpm)
        return (fuel, pitch, n, shaft_rpm, engine_rpm, Q, P_shaft, P_shaft_kw,
                P_engine_kw)

    # --- Coarse pass: 10× step ---
    coarse_step = pitch_step * 10.0
    coarse_grid = _pitch_sweep_range(prop, T_required, pitch_range, coarse_step)

    best_fuel = float("inf")
    best_coarse_pitch = None
    for pitch in coarse_grid:
        result = _evaluate(pitch)
        if result is not None and result[0] < best_fuel:
            best_fuel = result[0]
            best_coarse_pitch = pitch

    if best_coarse_pitch is None:
        # Coarse pass found nothing — try a full fine sweep as fallback
        fine_grid = _pitch_sweep_range(prop, T_required, pitch_range, pitch_step)
    else:
        # --- Fine pass: refine around the coarse best ---
        full_range = _pitch_sweep_range(prop, T_required, pitch_range, pitch_step)
        p_lo = best_coarse_pitch - coarse_step
        p_hi = best_coarse_pitch + coarse_step
        fine_grid = full_range[(full_range >= p_lo - 1e-9) &
                               (full_range <= p_hi + 1e-9)]

    best_fuel = float("inf")
    best_point = None
    for pitch in fine_grid:
        result = _evaluate(pitch)
        if result is None:
            continue
        fuel, p, n, shaft_rpm, engine_rpm, Q, P_shaft, P_shaft_kw, P_engine_kw = result
        if fuel < best_fuel:
            best_fuel = fuel
            best_point = OperatingPoint(
                pitch=p,
                n=n,
                rpm=shaft_rpm,
                Va=Va,
                thrust=prop.thrust(p, n, Va),
                torque=Q,
                power=P_shaft,
                power_kw=P_shaft_kw,
                eta0=prop.eta0(p, n, Va),
                fuel_rate=fuel,
                engine_power_kw=P_engine_kw,
                engine_rpm=engine_rpm,
                sfoc=fuel / P_engine_kw if P_engine_kw > 0 else 0.0,
            )

    if best_point is None:
        return OperatingPoint(
            pitch=0.0, n=0.0, rpm=0.0, Va=Va,
            thrust=0.0, torque=0.0, power=0.0, power_kw=0.0,
            eta0=0.0, found=False,
        )
    return best_point


# ============================================================
# Combinator curve generation
# ============================================================

@dataclass
class CombinatorPoint:
    """Single operating point on a combinator curve."""
    Vs_kn: float          # ship speed [kn]
    Va: float             # advance velocity [m/s]
    T_required: float     # required thrust [N]
    pitch: float          # optimal P/D ratio
    n: float              # shaft speed [rev/s]
    rpm: float            # shaft speed [rpm]
    engine_rpm: float     # engine speed [rpm]
    power_kw: float       # shaft power [kW]
    engine_power_kw: float  # total engine brake power [kW]
    eta0: float           # open-water efficiency
    fuel_rate: float      # fuel consumption [g/h]
    sfoc: float           # specific fuel consumption [g/kWh]
    found: bool = True


def generate_optimal_combinator(
    prop: CSeriesPropeller,
    engine: EngineModel,
    speeds_kn: np.ndarray,
    resistance_kn: np.ndarray,
    wake_fraction: np.ndarray,
    thrust_deduction: np.ndarray,
    gear_ratio: float = 1.0,
    shaft_efficiency: float = 0.97,
    auxiliary_power_kw: float = 0.0,
    sea_margin: float = 0.0,
    pitch_step: float = 0.005,
) -> list[CombinatorPoint]:
    """Generate an optimal combinator curve for a range of ship speeds.

    For each speed, finds the (pitch, RPM) combination that minimises
    fuel consumption, subject to engine constraints.

    Parameters
    ----------
    prop : CSeriesPropeller
        Propeller model.
    engine : EngineModel
        Engine fuel consumption model.
    speeds_kn : array
        Ship speeds [knots].
    resistance_kn : array
        Total resistance [kN] at each speed (in service/trial conditions).
    wake_fraction : array
        Wake fraction w at each speed.
    thrust_deduction : array
        Thrust deduction factor t at each speed.
    gear_ratio : float
        Engine RPM / prop RPM.
    shaft_efficiency : float
        Shaft line mechanical efficiency.
    auxiliary_power_kw : float
        Auxiliary electrical load on the engine [kW].
    sea_margin : float
        Additional power margin as a fraction (e.g. 0.15 for 15%).
        Applied as a multiplier on the required thrust: T = RT/(1-t) * (1+margin).
    pitch_step : float
        Pitch sweep resolution.

    Returns
    -------
    list of CombinatorPoint
        Optimal operating point at each speed.
    """
    results = []

    for i in range(len(speeds_kn)):
        Vs_kn = speeds_kn[i]
        Vs = Vs_kn * 0.5144  # m/s
        w = wake_fraction[i]
        t = thrust_deduction[i]
        Va = Vs * (1.0 - w)
        RT = resistance_kn[i] * 1000.0  # N
        T_required = RT / (1.0 - t) * (1.0 + sea_margin)

        op = find_min_fuel_operating_point(
            prop, Va, T_required, engine,
            gear_ratio=gear_ratio,
            shaft_efficiency=shaft_efficiency,
            auxiliary_power_kw=auxiliary_power_kw,
            pitch_step=pitch_step,
        )

        if op.found:
            results.append(CombinatorPoint(
                Vs_kn=Vs_kn,
                Va=Va,
                T_required=T_required,
                pitch=op.pitch,
                n=op.n,
                rpm=op.rpm,
                engine_rpm=op.engine_rpm,
                power_kw=op.power_kw,
                engine_power_kw=op.engine_power_kw,
                eta0=op.eta0,
                fuel_rate=op.fuel_rate,
                sfoc=op.sfoc,
            ))
        else:
            results.append(CombinatorPoint(
                Vs_kn=Vs_kn, Va=Va, T_required=T_required,
                pitch=0.0, n=0.0, rpm=0.0, engine_rpm=0.0,
                power_kw=0.0, engine_power_kw=0.0, eta0=0.0,
                fuel_rate=0.0, sfoc=0.0, found=False,
            ))

    return results


def fit_combinator_curve(
    points: list[CombinatorPoint],
) -> PchipInterpolator:
    """Fit a smooth combinator curve P/D = f(shaft_rpm) through the optimal points.

    Only uses points where a feasible solution was found.
    Returns a PchipInterpolator that maps shaft RPM -> pitch.
    """
    feasible = [p for p in points if p.found]
    if len(feasible) < 2:
        raise ValueError("Need at least 2 feasible points to fit a combinator curve")

    rpms = np.array([p.rpm for p in feasible])
    pitches = np.array([p.pitch for p in feasible])

    # Sort by RPM (should already be, but ensure)
    order = np.argsort(rpms)
    rpms = rpms[order]
    pitches = pitches[order]

    return PchipInterpolator(rpms, pitches)
