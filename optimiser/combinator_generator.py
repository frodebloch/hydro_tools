"""
Offline combinator generator and diagram plotter.

Generates combinator curves for a CPP (controllable pitch propeller) vessel
and produces four key diagrams:

1. Lever diagram:  Lever command (x-axis) vs RPM command & Pitch command
                   (dual y-axis).

2. Lever power/speed diagram:  Lever command (x-axis) vs engine power &
                               ship speed (dual y-axis).

3. P/D - N power diagram:  Shaft RPM (x-axis) vs Engine power (y-axis),
                           with iso-pitch and iso-speed lines, engine
                           envelope, propeller curve, and combinator
                           operating points.

4. P/D - N fuel diagram:  Same layout as the power diagram but with fuel
                          consumption [kg/h] on the y-axis.

Both the factory combinator (built from the muzzle diagram propeller curve)
and the fuel-optimal combinator (minimum SFOC at each speed) are plotted
for comparison.

Usage:
    python combinator_generator.py [--speeds 8 9 10 11 12 13 14 15]
                                   [--sea-margin 0.15]
                                   [--sg-power 0]
                                   [--output-dir .]
"""

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from propeller_model import CSeriesPropeller, load_c_series_data
from optimiser import (
    find_min_fuel_operating_point,
    _find_n_for_thrust,
    make_man_l27_38,
    MuzzleDiagramEngine,
)
from models.constants import (
    DATA_PATH_C440,
    DATA_PATH_C455,
    DATA_PATH_C470,
    GEAR_RATIO,
    HULL_RESISTANCE_KN,
    HULL_SPEEDS_KN,
    HULL_THRUST_CALM_KN,
    HULL_WAKE,
    HULL_T_DEDUCTION,
    PROP_BAR,
    PROP_DESIGN_PITCH,
    PROP_DIAMETER,
    SHAFT_EFF,
)
from models.combinator import FactoryCombinator


# ============================================================
# Propeller & engine setup
# ============================================================

def make_propeller() -> CSeriesPropeller:
    """Create the vessel 206 propeller model (C-series, multi-BAR)."""
    data_40 = load_c_series_data(DATA_PATH_C440)
    data_55 = load_c_series_data(DATA_PATH_C455)
    data_70 = load_c_series_data(DATA_PATH_C470)
    bar_data = {0.40: data_40, 0.55: data_55, 0.70: data_70}
    return CSeriesPropeller(
        bar_data,
        design_pitch=PROP_DESIGN_PITCH,
        diameter=PROP_DIAMETER,
        area_ratio=PROP_BAR,
        rho=1025.0,
    )


# ============================================================
# Optimal combinator generation (per operating point)
# ============================================================

def generate_optimal_schedule(
    prop: CSeriesPropeller,
    engine,
    speeds_kn: np.ndarray,
    sea_margin: float = 0.15,
    auxiliary_power_kw: float = 0.0,
    engine_rpm_min: float | None = None,
    engine_rpm_max: float | None = None,
) -> list[dict]:
    """Compute fuel-optimal (pitch, RPM) for each design speed.

    Returns list of dicts with keys:
        speed_kn, Va, T_kN, pitch, shaft_rpm, engine_rpm,
        power_kw, engine_power_kw, eta0, fuel_rate, sfoc, found
    """
    eff_min = engine_rpm_min if engine_rpm_min is not None else engine.min_rpm()
    eff_max = engine_rpm_max if engine_rpm_max is not None else engine.max_rpm()

    results = []
    for speed_kn in speeds_kn:
        w = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_WAKE))
        t_ded = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_T_DEDUCTION))
        R_calm_kN = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_RESISTANCE_KN))
        Va = speed_kn * 0.5144 * (1.0 - w)
        T_req_N = R_calm_kN * (1.0 + sea_margin) / (1.0 - t_ded) * 1000.0

        op = find_min_fuel_operating_point(
            prop, Va, T_req_N, engine,
            gear_ratio=GEAR_RATIO,
            shaft_efficiency=SHAFT_EFF,
            auxiliary_power_kw=auxiliary_power_kw,
            pitch_step=0.005,
            engine_rpm_min=eff_min,
            engine_rpm_max=eff_max,
        )

        results.append({
            "speed_kn": speed_kn,
            "Va": Va,
            "T_kN": T_req_N / 1000.0,
            "pitch": op.pitch if op.found else float("nan"),
            "shaft_rpm": op.rpm if op.found else float("nan"),
            "engine_rpm": op.engine_rpm if op.found else float("nan"),
            "power_kw": op.power_kw if op.found else float("nan"),
            "engine_power_kw": op.engine_power_kw if op.found else float("nan"),
            "eta0": op.eta0 if op.found else float("nan"),
            "fuel_rate": op.fuel_rate if op.found else float("nan"),
            "sfoc": op.sfoc if op.found else float("nan"),
            "found": op.found,
        })

    return results


# ============================================================
# Build lever-based schedule arrays
# ============================================================

def build_lever_schedule(
    combo_pitch: np.ndarray,
    combo_rpm: np.ndarray,
    combo_speed_kn: np.ndarray | None = None,
    combo_power_kw: np.ndarray | None = None,
    n_points: int = 201,
) -> dict[str, np.ndarray]:
    """Interpolate a combinator schedule onto a uniform lever grid.

    Parameters
    ----------
    combo_pitch : array
        Pitch (P/D) at each schedule breakpoint.
    combo_rpm : array
        Shaft RPM at each schedule breakpoint.
    combo_speed_kn : array, optional
        Ship speed [kn] at each breakpoint.
    combo_power_kw : array, optional
        Engine brake power [kW] at each breakpoint.
    n_points : int
        Number of points on the lever axis (0-100%).

    Returns
    -------
    dict with keys "lever", "pitch", "rpm", and optionally
    "speed_kn", "power_kw" -- each an array of shape (n_points,).
    """
    N = len(combo_pitch)
    breakpoint_lever = np.linspace(0.0, 100.0, N)
    lever = np.linspace(0.0, 100.0, n_points)
    result = {
        "lever": lever,
        "pitch": np.interp(lever, breakpoint_lever, combo_pitch),
        "rpm": np.interp(lever, breakpoint_lever, combo_rpm),
    }
    if combo_speed_kn is not None:
        result["speed_kn"] = np.interp(lever, breakpoint_lever, combo_speed_kn)
    if combo_power_kw is not None:
        result["power_kw"] = np.interp(lever, breakpoint_lever, combo_power_kw)
    return result


# ============================================================
# Plotting
# ============================================================

def plot_lever_diagram(
    factory_lever, factory_pitch, factory_rpm,
    optimal_lever, optimal_pitch, optimal_rpm,
    title_suffix: str = "",
    output_path: Path | None = None,
):
    """Plot lever command (x) vs RPM command and Pitch command (dual y-axis).

    Shows both factory and fuel-optimal combinator curves.
    """
    fig, ax_rpm = plt.subplots(figsize=(10, 6))
    ax_pitch = ax_rpm.twinx()

    # RPM curves (left y-axis)
    ax_rpm.plot(factory_lever, factory_rpm, "b-", linewidth=2,
                label="Standard RPM")
    ax_rpm.plot(optimal_lever, optimal_rpm, "b--", linewidth=2,
                label="Optimiser RPM")
    ax_rpm.set_xlabel("Lever command [%]", fontsize=12)
    ax_rpm.set_ylabel("Shaft RPM", fontsize=12, color="b")
    ax_rpm.tick_params(axis="y", labelcolor="b")
    ax_rpm.set_xlim(0, 100)

    # Pitch curves (right y-axis)
    ax_pitch.plot(factory_lever, factory_pitch, "r-", linewidth=2,
                  label="Standard P/D")
    ax_pitch.plot(optimal_lever, optimal_pitch, "r--", linewidth=2,
                  label="Optimiser P/D")
    ax_pitch.set_ylabel("Pitch ratio P/D", fontsize=12, color="r")
    ax_pitch.tick_params(axis="y", labelcolor="r")

    # Combined legend
    lines_rpm, labels_rpm = ax_rpm.get_legend_handles_labels()
    lines_pitch, labels_pitch = ax_pitch.get_legend_handles_labels()
    ax_rpm.legend(lines_rpm + lines_pitch, labels_rpm + labels_pitch,
                  loc="upper left", fontsize=10)

    ax_rpm.grid(True, alpha=0.3)
    ax_rpm.set_title(f"Combinator: Lever vs RPM & Pitch{title_suffix}",
                     fontsize=13)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"  Saved lever diagram: {output_path}")
    plt.close(fig)


def plot_lever_power_speed(
    factory_sched: dict[str, np.ndarray],
    optimal_sched: dict[str, np.ndarray],
    title_suffix: str = "",
    output_path: Path | None = None,
):
    """Plot lever command (x) vs engine power and ship speed (dual y-axis).

    Shows both factory and fuel-optimal combinator curves.
    """
    fig, ax_pwr = plt.subplots(figsize=(10, 6))
    ax_spd = ax_pwr.twinx()

    fl = factory_sched["lever"]
    ol = optimal_sched["lever"]

    # Power curves (left y-axis)
    ax_pwr.plot(fl, factory_sched["power_kw"], "b-", linewidth=2,
                label="Standard power")
    ax_pwr.plot(ol, optimal_sched["power_kw"], "b--", linewidth=2,
                label="Optimiser power")
    ax_pwr.set_xlabel("Lever command [%]", fontsize=12)
    ax_pwr.set_ylabel("Engine brake power [kW]", fontsize=12, color="b")
    ax_pwr.tick_params(axis="y", labelcolor="b")
    ax_pwr.set_xlim(0, 100)

    # Speed curves (right y-axis)
    ax_spd.plot(fl, factory_sched["speed_kn"], "r-", linewidth=2,
                label="Standard speed")
    ax_spd.plot(ol, optimal_sched["speed_kn"], "r--", linewidth=2,
                label="Optimiser speed")
    ax_spd.set_ylabel("Ship speed [kn]", fontsize=12, color="r")
    ax_spd.tick_params(axis="y", labelcolor="r")

    # Combined legend
    lines_pwr, labels_pwr = ax_pwr.get_legend_handles_labels()
    lines_spd, labels_spd = ax_spd.get_legend_handles_labels()
    ax_pwr.legend(lines_pwr + lines_spd, labels_pwr + labels_spd,
                  loc="upper left", fontsize=10)

    ax_pwr.grid(True, alpha=0.3)
    ax_pwr.set_title(f"Combinator: Lever vs Power & Speed{title_suffix}",
                     fontsize=13)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"  Saved lever power/speed diagram: {output_path}")
    plt.close(fig)


# ============================================================
# Iso-line computation (shared by power and fuel diagrams)
# ============================================================

@dataclass
class IsoLinePoint:
    """Single point on an iso-line with all derived quantities."""
    shaft_rpm: float
    engine_rpm: float
    pitch: float           # P/D at this point (= constant for iso-pitch)
    shaft_power_kw: float  # propeller shaft power
    engine_power_kw: float # total engine brake power (shaft/eff + sg)
    fuel_rate_gph: float   # fuel consumption [g/h]


@dataclass
class IsoLine:
    """A complete iso-line (constant pitch or constant speed)."""
    label: str
    points: list[IsoLinePoint]

    def shaft_rpm_arr(self) -> np.ndarray:
        return np.array([p.shaft_rpm for p in self.points])

    def engine_power_arr(self) -> np.ndarray:
        return np.array([p.engine_power_kw for p in self.points])

    def fuel_rate_arr(self) -> np.ndarray:
        return np.array([p.fuel_rate_gph for p in self.points])


def compute_iso_pitch_lines(
    prop: CSeriesPropeller,
    engine: MuzzleDiagramEngine,
    pitch_values: list[float],
    sg_power_kw: float = 0.0,
    rpm_range: tuple[float, float] = (65, 122),
    n_rpm: int = 60,
) -> dict[float, IsoLine]:
    """Compute iso-pitch lines (constant P/D, bollard condition Va=0).

    For each pitch setting, sweeps shaft RPM and computes the propeller
    absorbed power at bollard pull (Va=0).  This is the standard muzzle
    diagram convention showing the maximum power demand for each pitch.

    The SG load is added to the engine brake power at each point so both
    diagrams correctly reflect the total engine loading.

    Returns dict mapping pitch -> IsoLine.
    """
    rpm_arr = np.linspace(rpm_range[0], rpm_range[1], n_rpm)
    result = {}

    for pitch in pitch_values:
        points = []
        for shaft_rpm in rpm_arr:
            n = shaft_rpm / 60.0
            eng_rpm = shaft_rpm * GEAR_RATIO
            Q = prop.torque(pitch, n, 0.0)  # bollard (Va=0)
            P_shaft_kw = 2.0 * math.pi * n * Q / 1000.0
            P_eng_kw = P_shaft_kw / SHAFT_EFF + sg_power_kw

            if P_eng_kw <= 0 or eng_rpm < engine.min_rpm():
                continue

            # Clamp to engine envelope for fuel calc; skip if way over
            if P_eng_kw > engine.max_power(eng_rpm) * 1.2:
                continue

            P_for_fuel = min(P_eng_kw, engine.max_power(eng_rpm))
            fuel = engine.fuel_rate(P_for_fuel, eng_rpm)

            points.append(IsoLinePoint(
                shaft_rpm=shaft_rpm,
                engine_rpm=eng_rpm,
                pitch=pitch,
                shaft_power_kw=P_shaft_kw,
                engine_power_kw=P_eng_kw,
                fuel_rate_gph=fuel,
            ))

        result[pitch] = IsoLine(label=f"P/D={pitch:.1f}", points=points)

    return result


def compute_iso_speed_lines(
    prop: CSeriesPropeller,
    engine: MuzzleDiagramEngine,
    speeds_kn: list[float],
    sg_power_kw: float = 0.0,
    rpm_range: tuple[float, float] = (65, 122),
    n_rpm: int = 60,
    sea_margin: float = 0.15,
) -> dict[float, IsoLine]:
    """Compute iso-speed lines (constant ship speed).

    For a given ship speed, Va and required thrust T are fixed.  As we
    vary shaft RPM, we bisect on pitch to deliver T, then compute the
    resulting engine power and fuel rate.

    Returns dict mapping speed_kn -> IsoLine.
    """
    rpm_arr = np.linspace(rpm_range[0], rpm_range[1], n_rpm)
    result = {}

    for speed_kn in speeds_kn:
        w = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_WAKE))
        t_ded = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_T_DEDUCTION))
        R_calm_kN = float(np.interp(speed_kn, HULL_SPEEDS_KN,
                                     HULL_RESISTANCE_KN))
        Va = speed_kn * 0.5144 * (1.0 - w)
        T_req_N = R_calm_kN * (1.0 + sea_margin) / (1.0 - t_ded) * 1000.0

        points = []
        for shaft_rpm in rpm_arr:
            n = shaft_rpm / 60.0
            eng_rpm = shaft_rpm * GEAR_RATIO
            if n < 0.1 or eng_rpm < engine.min_rpm():
                continue

            # Bisect on pitch to find the pitch that delivers T_req_N
            p_lo, p_hi = 0.0, 1.5
            found = False
            for _ in range(50):
                p_mid = (p_lo + p_hi) / 2.0
                T_mid = prop.thrust(p_mid, n, Va)
                if abs(T_mid - T_req_N) < 50.0:  # 50 N tolerance
                    found = True
                    break
                if T_mid < T_req_N:
                    p_lo = p_mid
                else:
                    p_hi = p_mid

            if not found:
                T_max = prop.thrust(1.5, n, Va)
                if T_max < T_req_N:
                    continue
                p_mid = (p_lo + p_hi) / 2.0

            Q = prop.torque(p_mid, n, Va)
            P_shaft_kw = 2.0 * math.pi * n * Q / 1000.0
            P_eng_kw = P_shaft_kw / SHAFT_EFF + sg_power_kw

            if P_eng_kw <= 0:
                continue

            P_for_fuel = min(P_eng_kw, engine.max_power(eng_rpm))
            fuel = engine.fuel_rate(P_for_fuel, eng_rpm) if P_for_fuel > 0 else 0.0

            points.append(IsoLinePoint(
                shaft_rpm=shaft_rpm,
                engine_rpm=eng_rpm,
                pitch=p_mid,
                shaft_power_kw=P_shaft_kw,
                engine_power_kw=P_eng_kw,
                fuel_rate_gph=fuel,
            ))

        result[speed_kn] = IsoLine(label=f"{speed_kn:.0f} kn", points=points)

    return result


# ============================================================
# P/D - N diagram plotting (shared core)
# ============================================================

def _plot_pd_n_core(
    ax,
    prop: CSeriesPropeller,
    engine: MuzzleDiagramEngine,
    factory_schedule: list[dict],
    optimal_schedule: list[dict],
    iso_pitch: dict[float, IsoLine],
    iso_speed: dict[float, IsoLine],
    y_field: str,
    y_label: str,
    y_max: float,
    y_scale: float = 1.0,
    sg_power: float = 0.0,
):
    """Shared plotting logic for power and fuel variants of the P/D-N diagram.

    Parameters
    ----------
    y_field : str
        Which IsoLinePoint field to plot on y-axis.
        "engine_power_kw" for power diagram, "fuel_rate_gph" for fuel diagram.
    y_label : str
        Y-axis label string.
    y_max : float
        Upper y-axis limit (after scaling).
    y_scale : float
        Multiplier applied to raw y values before plotting (e.g. 0.001
        to convert g/h to kg/h).
    """
    # ---- Axis ranges ----
    min_shaft_rpm = engine.min_rpm() / GEAR_RATIO
    max_shaft_rpm = engine.max_rpm() / GEAR_RATIO
    rpm_lo = min_shaft_rpm - 5
    rpm_hi = max_shaft_rpm + 5

    # ---- Engine envelope (only on power diagram) ----
    if y_field == "engine_power_kw":
        eng_limit_shaft_rpm = np.array(engine.power_limit_rpm) / GEAR_RATIO
        eng_limit_kw = np.array(engine.power_limit_kw)
        ax.plot(eng_limit_shaft_rpm, eng_limit_kw, "k-", linewidth=2.5,
                label="Engine power limit", zorder=5)

        if len(engine.prop_curve_rpm) > 0:
            pc_shaft_rpm = np.array(engine.prop_curve_rpm) / GEAR_RATIO
            pc_kw = np.array(engine.prop_curve_kw)
            ax.plot(pc_shaft_rpm, pc_kw, "g-", linewidth=2, alpha=0.9,
                    label="Propeller curve", zorder=5)

    # ---- Iso-pitch lines ----
    pitch_colors = plt.cm.coolwarm(
        np.linspace(0.1, 0.9, len(iso_pitch)))
    for (pitch, iso), color in zip(iso_pitch.items(), pitch_colors):
        if not iso.points:
            continue
        x = iso.shaft_rpm_arr()
        y = np.array([getattr(p, y_field) for p in iso.points]) * y_scale

        mask = y <= y_max * 1.05
        if mask.any():
            ax.plot(x[mask], y[mask], "-", color=color,
                    linewidth=1.0, alpha=0.6, zorder=2)
            # Label at right end
            valid_idx = np.where(mask)[0]
            li = valid_idx[-1]
            ax.annotate(
                f"P/D={pitch:.1f}",
                (x[li], y[li]),
                textcoords="offset points", xytext=(3, 0),
                fontsize=7, color=color, alpha=0.8, va="center",
            )

    # ---- Iso-speed lines ----
    speed_colors = plt.cm.viridis(
        np.linspace(0.15, 0.85, len(iso_speed)))
    for (spd, iso), color in zip(iso_speed.items(), speed_colors):
        if not iso.points:
            continue
        x = iso.shaft_rpm_arr()
        y = np.array([getattr(p, y_field) for p in iso.points]) * y_scale

        valid = y > 0
        if valid.any():
            ax.plot(x[valid], y[valid], "--", color=color,
                    linewidth=1.5, alpha=0.7, zorder=2)
            valid_idx = np.where(valid)[0]
            mid = valid_idx[len(valid_idx) // 2]
            ax.annotate(
                f"{spd:.0f} kn",
                (x[mid], y[mid]),
                textcoords="offset points", xytext=(0, 8),
                fontsize=8, color=color, fontweight="bold", alpha=0.9,
                ha="center",
            )

    # ---- Factory combinator operating points ----
    fact_x = [p["shaft_rpm"] for p in factory_schedule if p["found"]]
    fact_y = [p[y_field] * y_scale for p in factory_schedule if p["found"]]
    fact_spd = [p["speed_kn"] for p in factory_schedule if p["found"]]
    ax.plot(fact_x, fact_y, "bs-", markersize=8, linewidth=2,
            markeredgecolor="navy", label="Standard control", zorder=6)
    for r, val, s in zip(fact_x, fact_y, fact_spd):
        ax.annotate(f"{s:.0f}", (r, val),
                    textcoords="offset points", xytext=(-12, 8),
                    fontsize=7, color="navy", fontweight="bold")

    # ---- Optimal combinator operating points ----
    opt_x = [p["shaft_rpm"] for p in optimal_schedule if p["found"]]
    opt_y = [p[y_field] * y_scale for p in optimal_schedule if p["found"]]
    opt_spd = [p["speed_kn"] for p in optimal_schedule if p["found"]]
    ax.plot(opt_x, opt_y, "ro-", markersize=8, linewidth=2,
            markeredgecolor="darkred", label="Optimiser",
            zorder=6)
    for r, val, s in zip(opt_x, opt_y, opt_spd):
        ax.annotate(f"{s:.0f}", (r, val),
                    textcoords="offset points", xytext=(8, -10),
                    fontsize=7, color="darkred", fontweight="bold")

    # ---- Formatting ----
    ax.set_xlabel("Shaft RPM (N)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(rpm_lo, rpm_hi)
    ax.set_ylim(0, y_max)


def plot_pd_n_power(
    prop: CSeriesPropeller,
    engine: MuzzleDiagramEngine,
    factory_schedule: list[dict],
    optimal_schedule: list[dict],
    iso_pitch: dict[float, IsoLine],
    iso_speed: dict[float, IsoLine],
    sg_power: float = 0.0,
    title_suffix: str = "",
    output_path: Path | None = None,
):
    """P/D - N diagram with engine brake power on the y-axis."""
    fig, ax = plt.subplots(figsize=(12, 8))

    _plot_pd_n_core(
        ax, prop, engine,
        factory_schedule, optimal_schedule,
        iso_pitch, iso_speed,
        y_field="engine_power_kw",
        y_label="Engine brake power [kW]",
        y_max=engine.max_power_kw * 1.1,
        sg_power=sg_power,
    )
    ax.set_title(f"P/D - N Diagram (power){title_suffix}", fontsize=14)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"  Saved P/D-N power diagram: {output_path}")
    plt.close(fig)


def plot_pd_n_fuel(
    prop: CSeriesPropeller,
    engine: MuzzleDiagramEngine,
    factory_schedule: list[dict],
    optimal_schedule: list[dict],
    iso_pitch: dict[float, IsoLine],
    iso_speed: dict[float, IsoLine],
    sg_power: float = 0.0,
    title_suffix: str = "",
    output_path: Path | None = None,
):
    """P/D - N diagram with fuel consumption rate [kg/h] on the y-axis."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Determine y-max from operating points (convert g/h -> kg/h)
    all_fuel = ([p["fuel_rate"] for p in factory_schedule if p["found"]]
                + [p["fuel_rate"] for p in optimal_schedule if p["found"]])
    if all_fuel:
        y_max = max(all_fuel) / 1000.0 * 1.3
    else:
        y_max = 600.0  # fallback kg/h

    # Add fuel_rate_gph alias so _plot_pd_n_core can find the field
    for sched in (factory_schedule, optimal_schedule):
        for pt in sched:
            pt["fuel_rate_gph"] = pt.get("fuel_rate", float("nan"))

    _plot_pd_n_core(
        ax, prop, engine,
        factory_schedule, optimal_schedule,
        iso_pitch, iso_speed,
        y_field="fuel_rate_gph",
        y_label="Fuel consumption [kg/h]",
        y_max=y_max,
        y_scale=0.001,  # g/h -> kg/h
        sg_power=sg_power,
    )
    ax.set_title(f"P/D - N Diagram (fuel){title_suffix}", fontsize=14)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"  Saved P/D-N fuel diagram: {output_path}")
    plt.close(fig)


# ============================================================
# Tabular output
# ============================================================

def print_schedule(label: str, schedule: list[dict]):
    """Print a combinator schedule as a formatted table."""
    print(f"\n{'=' * 90}")
    print(f"  {label}")
    print(f"{'=' * 90}")
    print(f"  {'Speed':>6s}  {'Va':>6s}  {'T':>7s}  {'P/D':>6s}  "
          f"{'ShRPM':>7s}  {'EngRPM':>7s}  {'Pshaft':>7s}  "
          f"{'Peng':>7s}  {'eta0':>6s}  {'SFOC':>6s}  {'Fuel':>8s}")
    print(f"  {'[kn]':>6s}  {'[m/s]':>6s}  {'[kN]':>7s}  {'[-]':>6s}  "
          f"{'[rpm]':>7s}  {'[rpm]':>7s}  {'[kW]':>7s}  "
          f"{'[kW]':>7s}  {'[-]':>6s}  {'g/kWh':>6s}  {'[kg/h]':>8s}")
    print(f"  {'-' * 86}")

    for pt in schedule:
        if not pt["found"]:
            print(f"  {pt['speed_kn']:6.1f}  {pt['Va']:6.2f}  "
                  f"{pt['T_kN']:7.1f}  {'--':>6s}  {'--':>7s}  "
                  f"{'--':>7s}  {'--':>7s}  {'--':>7s}  {'--':>6s}  "
                  f"{'--':>6s}  {'--':>8s}")
            continue
        print(f"  {pt['speed_kn']:6.1f}  {pt['Va']:6.2f}  {pt['T_kN']:7.1f}  "
              f"{pt['pitch']:6.3f}  {pt['shaft_rpm']:7.1f}  "
              f"{pt['engine_rpm']:7.0f}  {pt['power_kw']:7.0f}  "
              f"{pt['engine_power_kw']:7.0f}  {pt['eta0']:6.4f}  "
              f"{pt['sfoc']:6.1f}  {pt['fuel_rate'] / 1000:8.1f}")


# ============================================================
# Main entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Offline combinator generator and diagram plotter.")
    parser.add_argument(
        "--speeds", type=float, nargs="+",
        default=[8, 9, 10, 11, 12, 13, 14, 15],
        help="Design speeds [kn] for the combinator schedule.")
    parser.add_argument(
        "--sea-margin", type=float, default=0.15,
        help="Sea margin fraction (default: 0.15 = 15%%).")
    parser.add_argument(
        "--sg-power", type=float, default=0.0,
        help="Shaft generator auxiliary power [kW] (default: 0).")
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory for output PNG files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    speeds = np.array(args.speeds)
    sea_margin = args.sea_margin
    sg_power = args.sg_power

    print("Combinator Generator")
    print("=" * 50)
    print(f"  Propeller:  D={PROP_DIAMETER}m, BAR={PROP_BAR}, "
          f"design P/D={PROP_DESIGN_PITCH}")
    print(f"  Gear ratio: {GEAR_RATIO:.3f}")
    print(f"  Sea margin: {sea_margin * 100:.0f}%")
    print(f"  SG power:   {sg_power:.0f} kW")
    print(f"  Speeds:     {speeds} kn")
    print()

    # ---- Build models ----
    print("Building propeller model ...")
    prop = make_propeller()
    engine = make_man_l27_38()
    print(f"  Engine: {engine.name}")
    print(f"  Engine RPM: {engine.min_rpm():.0f} - {engine.max_rpm():.0f}")

    # ---- Factory combinator ----
    print("\nBuilding factory combinator ...")
    factory = FactoryCombinator(
        engine, prop,
        sea_margin=sea_margin,
        sg_allowance_kw=sg_power,
    )

    # Build factory schedule data (evaluate at each design speed)
    factory_schedule = []
    for speed_kn in speeds:
        w = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_WAKE))
        t_ded = float(np.interp(speed_kn, HULL_SPEEDS_KN, HULL_T_DEDUCTION))
        R_calm_kN = float(np.interp(speed_kn, HULL_SPEEDS_KN,
                                     HULL_RESISTANCE_KN))
        Va = speed_kn * 0.5144 * (1.0 - w)
        T_req_N = R_calm_kN * (1.0 + sea_margin) / (1.0 - t_ded) * 1000.0

        result = factory.evaluate(T_req_N, Va)
        if result is not None:
            eng_rpm = result["engine_rpm"]
            factory_schedule.append({
                "speed_kn": speed_kn,
                "Va": Va,
                "T_kN": T_req_N / 1000.0,
                "pitch": result["pitch"],
                "shaft_rpm": result["rpm"],
                "engine_rpm": eng_rpm,
                "power_kw": result["power_kw"],
                "engine_power_kw": result["power_kw"] / SHAFT_EFF + sg_power,
                "eta0": result["eta0"],
                "fuel_rate": result["fuel_rate"],
                "sfoc": (result["fuel_rate"]
                         / (result["power_kw"] / SHAFT_EFF + sg_power)),
                "found": True,
            })
        else:
            factory_schedule.append({
                "speed_kn": speed_kn, "Va": Va,
                "T_kN": T_req_N / 1000.0,
                "pitch": float("nan"), "shaft_rpm": float("nan"),
                "engine_rpm": float("nan"), "power_kw": float("nan"),
                "engine_power_kw": float("nan"), "eta0": float("nan"),
                "fuel_rate": float("nan"), "sfoc": float("nan"),
                "found": False,
            })

    print_schedule("Factory Combinator Schedule", factory_schedule)

    # ---- Fuel-optimal combinator ----
    print("\nComputing fuel-optimal combinator ...")
    optimal_schedule = generate_optimal_schedule(
        prop, engine, speeds,
        sea_margin=sea_margin,
        auxiliary_power_kw=sg_power,
    )
    print_schedule("Fuel-Optimal Combinator Schedule", optimal_schedule)

    # ---- Extract arrays for plotting ----
    fact_sched_pitch = np.array([p["pitch"] for p in factory_schedule
                                 if p["found"]])
    fact_sched_rpm = np.array([p["shaft_rpm"] for p in factory_schedule
                               if p["found"]])
    fact_sched_speeds = np.array([p["speed_kn"] for p in factory_schedule
                                  if p["found"]])
    fact_sched_power = np.array([p["engine_power_kw"] for p in factory_schedule
                                 if p["found"]])

    opt_sched_pitch = np.array([p["pitch"] for p in optimal_schedule
                                if p["found"]])
    opt_sched_rpm = np.array([p["shaft_rpm"] for p in optimal_schedule
                              if p["found"]])
    opt_sched_speeds = np.array([p["speed_kn"] for p in optimal_schedule
                                 if p["found"]])
    opt_sched_power = np.array([p["engine_power_kw"] for p in optimal_schedule
                                if p["found"]])

    # Build lever-based arrays for the lever diagram
    # Prepend zero-thrust point (lever=0: pitch=0, rpm=min, speed=0, power=0)
    min_shaft_rpm = engine.min_rpm() / GEAR_RATIO
    fact_full_pitch = np.concatenate([[0.0], fact_sched_pitch])
    fact_full_rpm = np.concatenate([[min_shaft_rpm], fact_sched_rpm])
    fact_full_speed = np.concatenate([[0.0], fact_sched_speeds])
    fact_full_power = np.concatenate([[0.0], fact_sched_power])

    opt_full_pitch = np.concatenate([[0.0], opt_sched_pitch])
    opt_full_rpm = np.concatenate([[min_shaft_rpm], opt_sched_rpm])
    opt_full_speed = np.concatenate([[0.0], opt_sched_speeds])
    opt_full_power = np.concatenate([[0.0], opt_sched_power])

    fact_lever_sched = build_lever_schedule(
        fact_full_pitch, fact_full_rpm,
        combo_speed_kn=fact_full_speed,
        combo_power_kw=fact_full_power,
    )
    opt_lever_sched = build_lever_schedule(
        opt_full_pitch, opt_full_rpm,
        combo_speed_kn=opt_full_speed,
        combo_power_kw=opt_full_power,
    )

    # ---- Title suffix ----
    suffix = ""
    if sg_power > 0:
        suffix = f" (SG {sg_power:.0f} kW)"

    # ---- Plot 1: Lever diagram ----
    print("\nGenerating lever diagram ...")
    plot_lever_diagram(
        fact_lever_sched["lever"], fact_lever_sched["pitch"],
        fact_lever_sched["rpm"],
        opt_lever_sched["lever"], opt_lever_sched["pitch"],
        opt_lever_sched["rpm"],
        title_suffix=suffix,
        output_path=output_dir / "combinator_lever.png",
    )

    # ---- Plot 2: Lever power/speed diagram ----
    print("Generating lever power/speed diagram ...")
    plot_lever_power_speed(
        fact_lever_sched,
        opt_lever_sched,
        title_suffix=suffix,
        output_path=output_dir / "combinator_lever_power_speed.png",
    )

    # ---- Plot 3 & 4: P/D - N diagrams ----
    # Compute iso-lines once, reuse for both power and fuel variants
    print("Computing iso-lines for P/D-N diagrams ...")
    min_shaft_rpm_plot = engine.min_rpm() / GEAR_RATIO
    max_shaft_rpm_plot = engine.max_rpm() / GEAR_RATIO
    rpm_range = (min_shaft_rpm_plot - 5, max_shaft_rpm_plot + 5)

    pitch_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    speed_values = [float(s) for s in speeds]

    iso_pitch = compute_iso_pitch_lines(
        prop, engine, pitch_values,
        sg_power_kw=sg_power,
        rpm_range=rpm_range,
    )
    iso_speed = compute_iso_speed_lines(
        prop, engine, speed_values,
        sg_power_kw=sg_power,
        rpm_range=rpm_range,
        sea_margin=sea_margin,
    )

    print("Generating P/D - N power diagram ...")
    plot_pd_n_power(
        prop, engine,
        factory_schedule, optimal_schedule,
        iso_pitch, iso_speed,
        sg_power=sg_power,
        title_suffix=suffix,
        output_path=output_dir / "combinator_pd_n_power.png",
    )

    print("Generating P/D - N fuel diagram ...")
    plot_pd_n_fuel(
        prop, engine,
        factory_schedule, optimal_schedule,
        iso_pitch, iso_speed,
        sg_power=sg_power,
        title_suffix=suffix,
        output_path=output_dir / "combinator_pd_n_fuel.png",
    )

    # ---- Summary comparison ----
    print(f"\n{'=' * 90}")
    print("  Factory vs Optimal Comparison")
    print(f"{'=' * 90}")
    print(f"  {'Speed':>6s}  {'F.P/D':>6s}  {'F.RPM':>7s}  {'F.Fuel':>8s}  "
          f"{'O.P/D':>6s}  {'O.RPM':>7s}  {'O.Fuel':>8s}  {'Saving':>7s}")
    print(f"  {'[kn]':>6s}  {'[-]':>6s}  {'[rpm]':>7s}  {'[kg/h]':>8s}  "
          f"{'[-]':>6s}  {'[rpm]':>7s}  {'[kg/h]':>8s}  {'[%]':>7s}")
    print(f"  {'-' * 86}")

    for f, o in zip(factory_schedule, optimal_schedule):
        spd = f["speed_kn"]
        if f["found"] and o["found"]:
            saving = 100.0 * (f["fuel_rate"] - o["fuel_rate"]) / f["fuel_rate"]
            print(f"  {spd:6.1f}  {f['pitch']:6.3f}  {f['shaft_rpm']:7.1f}  "
                  f"{f['fuel_rate'] / 1000:8.1f}  "
                  f"{o['pitch']:6.3f}  {o['shaft_rpm']:7.1f}  "
                  f"{o['fuel_rate'] / 1000:8.1f}  {saving:+6.1f}%")
        else:
            f_str = (f"{f['pitch']:6.3f}  {f['shaft_rpm']:7.1f}  "
                     f"{f['fuel_rate'] / 1000:8.1f}"
                     if f["found"] else f"{'--':>6s}  {'--':>7s}  {'--':>8s}")
            o_str = (f"{o['pitch']:6.3f}  {o['shaft_rpm']:7.1f}  "
                     f"{o['fuel_rate'] / 1000:8.1f}"
                     if o["found"] else f"{'--':>6s}  {'--':>7s}  {'--':>8s}")
            print(f"  {spd:6.1f}  {f_str}  {o_str}  {'--':>7s}")

    print(f"\nOutput files in: {output_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
