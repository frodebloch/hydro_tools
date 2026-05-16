"""Regime-B saturation screening for cqa post-WCFDI residual thruster set.

Per sec.12.21.21.12-14 of analysis.md:

  Regime B = sustained saturation in (T_A, T_END] s post-WCF, driven by the
             slow-varying environmental load distribution exceeding the
             residual thruster polytope per-DOF maxima.

The screening calculator answers: given (a) the pre-WCF demand distribution
on (tau_x, tau_y, tau_z) and (b) the residual thruster set after the WCFDI,
what is the probability that demand exceeds the residual polytope at any
instant, and how often does that happen over the operator's t_horizon?

Key components:

1. ``compute_residual_polytope(thrusters, surviving_indices)``
   Per-DOF max/min available thrust, ported from brucon's
   ``BasicAllocator::CalculateAvailableThrust`` (basic_allocation.cpp:1089).
   Returns dict {max_surge, min_surge, max_sway, min_sway, max_yaw, min_yaw}.

2. ``saturation_probability(mu, sigma, t_max)``
   For each DOF: P(|tau_i| > t_i_max) under a Gaussian assumption.

3. ``saturation_event_rate(mu, sigma, omega_decorr, t_max)``
   Expected upcrossings per second of the threshold |tau_i|=t_i_max under
   a band-limited Gaussian process. Ric's formula:
     rate = (omega / (2 pi)) * exp(-(t_max - mu)^2 / (2 sigma^2))
   for each side, summed.

4. ``traffic_light(N_sat)``
   Three-tier classification for operator display.

This module takes the pre-WCF (mu, sigma, omega_decorr) as input. Two
sources are supported:

  (a) Empirical: estimated from the pre-WCF window of brucon delivered Tx/Ty/Tz.
      Used here for validation against the brucon ensemble.
  (b) Theoretical: derived from cqa's intact-state PSD prediction (the
      eventual operational target). To be wired in once (a) is validated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.special import erfc


# ---------------------------------------------------------------------------
# Vessel data: CSOV thruster geometry from brucon
#   build/bin/config_csov/propulsors.prototxt
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Thruster:
    name: str
    x_position: float       # m, body-fixed (positive = forward of midships)
    y_position: float       # m, body-fixed (positive = starboard)
    kind: str               # "tunnel" or "azimuth"
    max_pos_thrust: float   # kN
    max_neg_thrust: float   # kN, signed (negative for pullable; 0 for one-way)


# CSOV from build/bin/config_csov/propulsors.prototxt.  Indices match brucon's
# ordering used in the WCFDI lua scripts:
#   bus_port lost = thrusters {0 (Bow1), 3 (PortMP)}
#   surviving    = {1 (Bow2), 2 (BowAz), 4 (StbdMP)}
CSOV_THRUSTERS: tuple[Thruster, ...] = (
    Thruster("Bow1",   x_position=42.813, y_position=0.0, kind="tunnel",
             max_pos_thrust=313.0, max_neg_thrust=-313.0),
    Thruster("Bow2",   x_position=38.612, y_position=0.0, kind="tunnel",
             max_pos_thrust=313.0, max_neg_thrust=-313.0),
    Thruster("BowAz",  x_position=33.99,  y_position=0.0, kind="azimuth",
             max_pos_thrust=316.0, max_neg_thrust=0.0),
    Thruster("PortMP", x_position=-48.09, y_position=-5.4, kind="azimuth",
             max_pos_thrust=522.0, max_neg_thrust=0.0),
    Thruster("StbdMP", x_position=-48.09, y_position=+5.4, kind="azimuth",
             max_pos_thrust=522.0, max_neg_thrust=0.0),
)

CSOV_BUS_PORT_LOST = (0, 3)  # Bow1 + PortMP


# ---------------------------------------------------------------------------
# Per-thruster max forces at zero surge speed (DP-mode operating point)
# ---------------------------------------------------------------------------

def _thr_max_pos_y(t: Thruster) -> float:
    """Max +Y force the thruster can deliver in DP-mode (zero surge speed,
    no fixed-angle, no transit). Mirrors brucon Azimuth::MaxPositiveYForce
    and Tunnel::MaxPositiveYForce."""
    if t.kind == "tunnel":
        return t.max_pos_thrust
    if t.kind == "azimuth":
        return t.max_pos_thrust  # rotate to +90deg
    raise ValueError(f"unknown kind {t.kind!r}")


def _thr_max_neg_y(t: Thruster) -> float:
    if t.kind == "tunnel":
        return t.max_neg_thrust  # already negative for tunnel (-313)
    if t.kind == "azimuth":
        return -t.max_pos_thrust  # rotate to -90deg
    raise ValueError(f"unknown kind {t.kind!r}")


def _thr_max_pos_x(t: Thruster) -> float:
    if t.kind == "tunnel":
        return 0.0
    if t.kind == "azimuth":
        return t.max_pos_thrust
    raise ValueError(f"unknown kind {t.kind!r}")


def _thr_max_neg_x(t: Thruster) -> float:
    if t.kind == "tunnel":
        return 0.0
    if t.kind == "azimuth":
        return -t.max_pos_thrust  # azimuth can reverse via 180deg rotation
    raise ValueError(f"unknown kind {t.kind!r}")


# ---------------------------------------------------------------------------
# Polytope: BasicAllocator::CalculateAvailableThrust port
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaxTau:
    max_surge: float
    min_surge: float
    max_sway: float
    min_sway: float
    max_yaw: float
    min_yaw: float
    # Direct contributions for diagnostics
    max_sway_bow: float
    min_sway_bow: float
    arm_bow: float
    max_sway_stern: float
    min_sway_stern: float
    arm_stern: float


def compute_residual_polytope(
    thrusters: Sequence[Thruster] = CSOV_THRUSTERS,
    surviving_indices: Sequence[int] | None = None,
    sway_active: bool = True,
    bow_x_threshold_m: float = 0.0,
) -> MaxTau:
    """Port of brucon BasicAllocator::CalculateAvailableThrust at surge_speed=0.

    Splits surviving thrusters into bow group (x_position > bow_x_threshold)
    and stern group (x_position <= bow_x_threshold). Main propellers (azimuth
    at the stern) get aggregated with stern thrusters for sway/yaw, and their
    arm is the y-weighted average -- but since both port/stbd MPs have the
    same x_position, this collapses to just x_position_stern.

    Returns ``MaxTau`` with per-DOF max/min available thrust in kN and
    kN.m. Yaw uses the brucon convention from basic_allocation.cpp:1126-1132
    where active sway changes the yaw bound formula.
    """
    if surviving_indices is None:
        surviving_indices = tuple(range(len(thrusters)))
    surv = [thrusters[i] for i in surviving_indices]

    bow = [t for t in surv if t.x_position > bow_x_threshold_m]
    stern_tunnel = [t for t in surv if t.x_position <= bow_x_threshold_m and t.kind == "tunnel"]
    stern_azimuth = [t for t in surv if t.x_position <= bow_x_threshold_m and t.kind == "azimuth"]
    # In CSOV, "stern thrusters" group is empty (no tunnel at stern) and
    # main propellers are PortMP/StbdMP.  We follow brucon's split:
    #   bow_thrusters  = bow thruster group
    #   stern_thrusters = stern *non-azimuth* thrusters (empty for CSOV)
    #   main_propellers = stern azimuth thrusters (PortMP, StbdMP)

    # ---- Bow group ----
    max_sway_bow = sum(_thr_max_pos_y(t) for t in bow)
    min_sway_bow = sum(_thr_max_neg_y(t) for t in bow)
    if bow:
        # Force-weighted x position used as moment arm (matches brucon group)
        # In brucon, ThrusterGroup::x_position() is the *capability-weighted*
        # x-position of the group's contributing thrusters. We approximate by
        # equally weighting (each thruster's nominal x with weight 1).
        # For CSOV bow: Bow1 (42.81), Bow2 (38.61), BowAz (33.99) -> mean ~38.5 m.
        # If only Bow2+BowAz survive: mean ~36.3 m.
        arm_bow = sum(t.x_position * _thr_max_pos_y(t) for t in bow) / max(max_sway_bow, 1e-9)
    else:
        arm_bow = 0.0

    # ---- Stern group (stern tunnel + main propellers) ----
    max_sway_stern_tunnel = sum(_thr_max_pos_y(t) for t in stern_tunnel)
    min_sway_stern_tunnel = sum(_thr_max_neg_y(t) for t in stern_tunnel)
    max_sway_stern_mp = sum(_thr_max_pos_y(t) for t in stern_azimuth)
    min_sway_stern_mp = sum(_thr_max_neg_y(t) for t in stern_azimuth)
    max_sway_stern = max_sway_stern_tunnel + max_sway_stern_mp
    min_sway_stern = min_sway_stern_tunnel + min_sway_stern_mp
    if max_sway_stern > 1e-9:
        arm_stern_pos = (
            sum(t.x_position * _thr_max_pos_y(t) for t in stern_tunnel) +
            sum(t.x_position * _thr_max_pos_y(t) for t in stern_azimuth)
        ) / max_sway_stern
    else:
        arm_stern_pos = 0.0
    arm_stern = arm_stern_pos  # brucon uses single arm_stern

    EPS = 1e-9

    # ---- Sway maxima/minima (brucon basic_allocation.cpp:1110-1124) ----
    if max_sway_bow < EPS or max_sway_stern < EPS:
        max_sway = 0.0
    elif max_sway_bow * arm_bow > -max_sway_stern * arm_stern and arm_bow > EPS:
        # 1.35 boost factor "to account for moment offloading stern"
        max_sway = 1.35 * (-max_sway_stern * arm_stern / arm_bow + max_sway_stern)
    elif arm_stern < EPS:
        max_sway = -max_sway_bow * arm_bow / arm_stern + max_sway_bow
    else:
        max_sway = 0.0

    if min_sway_bow > -EPS or min_sway_stern > -EPS:
        min_sway = 0.0
    elif min_sway_bow * arm_bow < -min_sway_stern * arm_stern and arm_bow > EPS:
        min_sway = 1.35 * (-min_sway_stern * arm_stern / arm_bow + min_sway_stern)
    elif arm_stern < -EPS:
        min_sway = -min_sway_bow * arm_bow / arm_stern + min_sway_bow
    else:
        min_sway = 0.0

    # ---- Yaw (brucon basic_allocation.cpp:1126-1132) ----
    if sway_active and max_sway > EPS and min_sway < -EPS:
        max_yaw = max_sway_bow * arm_bow + min_sway_stern * arm_stern
        min_yaw = min_sway_bow * arm_bow + max_sway_stern * arm_stern
    else:
        max_yaw = max(max_sway_bow * arm_bow, min_sway_stern * arm_stern)
        min_yaw = min(min_sway_bow * arm_bow, max_sway_stern * arm_stern)

    # ---- Surge ----
    max_surge = sum(_thr_max_pos_x(t) for t in surv)
    min_surge = sum(_thr_max_neg_x(t) for t in surv)

    return MaxTau(
        max_surge=max_surge, min_surge=min_surge,
        max_sway=max_sway, min_sway=min_sway,
        max_yaw=max_yaw, min_yaw=min_yaw,
        max_sway_bow=max_sway_bow, min_sway_bow=min_sway_bow, arm_bow=arm_bow,
        max_sway_stern=max_sway_stern, min_sway_stern=min_sway_stern,
        arm_stern=arm_stern,
    )


# ---------------------------------------------------------------------------
# Saturation probability and event rate (Gaussian assumption)
# ---------------------------------------------------------------------------

def _gaussian_tail(threshold: float, mu: float, sigma: float) -> float:
    """P(X > threshold) for X ~ N(mu, sigma^2)."""
    if sigma <= 0:
        return 1.0 if mu > threshold else 0.0
    z = (threshold - mu) / sigma
    return 0.5 * erfc(z / np.sqrt(2.0))


def saturation_probability(mu: float, sigma: float,
                           t_max_pos: float, t_max_neg: float) -> dict:
    """Probability of being above the +threshold OR below the -threshold
    for a Gaussian demand X ~ N(mu, sigma^2)."""
    p_high = _gaussian_tail(t_max_pos, mu, sigma)
    p_low = _gaussian_tail(-t_max_neg, -mu, sigma)  # = P(X < t_max_neg)
    return {
        "p_high": p_high,
        "p_low": p_low,
        "p_either": p_high + p_low,
        "z_high": (t_max_pos - mu) / sigma if sigma > 0 else np.inf,
        "z_low":  (-t_max_neg - (-mu)) / sigma if sigma > 0 else np.inf,
    }


def saturation_event_rate(mu: float, sigma: float, omega_c: float,
                          t_max_pos: float, t_max_neg: float) -> dict:
    """Expected number of upcrossings per second of |X|=threshold under a
    band-limited Gaussian process X(t) with mean mu, std sigma, and
    characteristic radian frequency omega_c (the spectral peak of the
    relevant slow-varying load).  Rice's formula:

      rate_+ = (omega_c / (2 pi)) * exp(-(t_max_pos - mu)^2 / (2 sigma^2))
      rate_- = (omega_c / (2 pi)) * exp(-(-(t_max_neg) - mu)^2 / (2 sigma^2))
                                                # i.e. downcrossing of t_max_neg

    For a narrow-band slow-varying process the characteristic frequency is
    1/T_decorr -- pass omega_c = 2 pi / T_decorr_lf_s.

    Returns rates in events per second on each side.
    """
    if sigma <= 0:
        return {"rate_high": 0.0, "rate_low": 0.0, "rate_either": 0.0}
    base = omega_c / (2.0 * np.pi)
    rate_high = base * np.exp(-((t_max_pos - mu) ** 2) / (2.0 * sigma ** 2))
    rate_low = base * np.exp(-((-(t_max_neg) - (-mu)) ** 2) / (2.0 * sigma ** 2))
    return {
        "rate_high": rate_high,
        "rate_low": rate_low,
        "rate_either": rate_high + rate_low,
    }


def traffic_light(n_sat_horizon: float) -> str:
    """Three-tier traffic light from expected number of saturation events
    over the operator's t_horizon."""
    if n_sat_horizon < 0.1:
        return "GREEN"
    if n_sat_horizon < 1.0:
        return "AMBER"
    return "RED"


# ---------------------------------------------------------------------------
# Empirical (mu, sigma) estimation from brucon pre-WCF window
# ---------------------------------------------------------------------------

T_WCF_S = 1560.0
T_DECORR_LF_S_DEFAULT = 15.0  # heuristic, matches cqa/_constants

# Column indices used in saturation_regime_scan.py
COL_T = 0
COL_TX = 7
COL_TY = 8
COL_TZ = 9


def estimate_pre_wcf_demand(seed_dir: Path, t_pre_window_s: float = 290.0,
                            t_pre_skip_s: float = 10.0) -> dict:
    """Fit Gaussian (mu, sigma) and effective decorrelation freq omega_c
    to the delivered (Tx, Ty, Tz) pre-WCF window
    [t in (T_WCF - t_pre_window_s - t_pre_skip_s, T_WCF - t_pre_skip_s)].
    Decorr freq estimated from autocovariance integral time scale."""
    f = seed_dir / f"{seed_dir.name}.out"
    if not f.exists():
        return {}
    d = np.loadtxt(f, skiprows=1, usecols=(COL_T, COL_TX, COL_TY, COL_TZ))
    t = d[:, 0]
    mask = (t >= T_WCF_S - t_pre_window_s - t_pre_skip_s) & (t <= T_WCF_S - t_pre_skip_s)
    if mask.sum() < 50:
        return {}
    Tx, Ty, Tz = d[mask, 1], d[mask, 2], d[mask, 3]
    dt = float(np.median(np.diff(t[mask])))

    def _decorr(x: np.ndarray) -> float:
        """Integral time scale via integral of normalized autocovariance to
        first zero crossing."""
        x0 = x - x.mean()
        n = len(x0)
        # Fast autocovariance via FFT
        s = np.fft.rfft(x0, n=2 * n)
        ac = np.fft.irfft(s * np.conj(s), n=2 * n)[:n].real
        ac /= ac[0]
        # First zero crossing
        zc = np.where(ac < 0)[0]
        end = zc[0] if len(zc) > 0 else min(n, int(60.0 / dt))
        T_int = max(np.trapezoid(ac[:end], dx=dt), 1e-3)
        return T_int

    return {
        "Tx": {"mu": float(Tx.mean()), "sigma": float(Tx.std()),
               "T_decorr": _decorr(Tx)},
        "Ty": {"mu": float(Ty.mean()), "sigma": float(Ty.std()),
               "T_decorr": _decorr(Ty)},
        "Tz": {"mu": float(Tz.mean()), "sigma": float(Tz.std()),
               "T_decorr": _decorr(Tz)},
    }


# ---------------------------------------------------------------------------
# CLI: print residual polytope + screening for each cell vs ensemble
# ---------------------------------------------------------------------------

CELLS = [
    "bf4_c1_h0", "bf4_c1_q10",
    "bf6_h0", "bf6_h0_w45", "bf6_q10", "bf6_q10_w45",
    "bf8_h0", "bf8_h0_w45", "bf8_q10", "bf8_q10_w45",
    "pwq30",
]
HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
T_HORIZON_S = 200.0  # match the regime-B classifier window


def _ensemble_demand(cell: str, seeds: range = range(1000, 1051)) -> dict | None:
    """Aggregate (mu, sigma, T_decorr) across all available seeds for a cell
    by pooling the pre-WCF samples."""
    Tx_all, Ty_all, Tz_all, dt_list = [], [], [], []
    for s in seeds:
        d_dir = WORK / f"{cell}_seed{s}"
        if not d_dir.exists():
            continue
        f = d_dir / f"{d_dir.name}.out"
        if not f.exists():
            continue
        d = np.loadtxt(f, skiprows=1, usecols=(COL_T, COL_TX, COL_TY, COL_TZ))
        t = d[:, 0]
        mask = (t >= T_WCF_S - 300.0) & (t <= T_WCF_S - 10.0)
        if mask.sum() < 50:
            continue
        Tx_all.append(d[mask, 1] - d[mask, 1].mean())
        Ty_all.append(d[mask, 2] - d[mask, 2].mean())
        Tz_all.append(d[mask, 3] - d[mask, 3].mean())
        # for mean estimation use raw samples
        dt_list.append(float(np.median(np.diff(t[mask]))))
    if not Tx_all:
        return None
    # Re-load means from each seed (since we de-meaned for variance/decorr
    # estimation -- now compute global mean across seeds)
    Tx_means, Ty_means, Tz_means = [], [], []
    for s in seeds:
        d_dir = WORK / f"{cell}_seed{s}"
        if not d_dir.exists():
            continue
        f = d_dir / f"{d_dir.name}.out"
        if not f.exists():
            continue
        d = np.loadtxt(f, skiprows=1, usecols=(COL_T, COL_TX, COL_TY, COL_TZ))
        t = d[:, 0]
        mask = (t >= T_WCF_S - 300.0) & (t <= T_WCF_S - 10.0)
        if mask.sum() < 50:
            continue
        Tx_means.append(d[mask, 1].mean())
        Ty_means.append(d[mask, 2].mean())
        Tz_means.append(d[mask, 3].mean())

    Tx_cat = np.concatenate(Tx_all)
    Ty_cat = np.concatenate(Ty_all)
    Tz_cat = np.concatenate(Tz_all)
    dt = float(np.mean(dt_list))

    def _decorr(x: np.ndarray) -> float:
        n = len(x)
        s = np.fft.rfft(x, n=2 * n)
        ac = np.fft.irfft(s * np.conj(s), n=2 * n)[:n].real
        ac /= ac[0]
        zc = np.where(ac < 0)[0]
        end = zc[0] if len(zc) > 0 else min(n, int(60.0 / dt))
        return max(np.trapezoid(ac[:end], dx=dt), 1e-3)

    return {
        "Tx": {"mu": float(np.mean(Tx_means)),
               "sigma": float(Tx_cat.std()),
               "T_decorr": _decorr(Tx_cat),
               "n_seeds": len(Tx_means)},
        "Ty": {"mu": float(np.mean(Ty_means)),
               "sigma": float(Ty_cat.std()),
               "T_decorr": _decorr(Ty_cat),
               "n_seeds": len(Ty_means)},
        "Tz": {"mu": float(np.mean(Tz_means)),
               "sigma": float(Tz_cat.std()),
               "T_decorr": _decorr(Tz_cat),
               "n_seeds": len(Tz_means)},
    }


def main() -> None:
    # ---- Residual polytope ----
    intact = compute_residual_polytope(CSOV_THRUSTERS, surviving_indices=None)
    surviving = tuple(i for i in range(len(CSOV_THRUSTERS)) if i not in CSOV_BUS_PORT_LOST)
    residual = compute_residual_polytope(CSOV_THRUSTERS, surviving_indices=surviving)

    print("=" * 78)
    print("CSOV residual polytope (bus_port lost: Bow1 + PortMP)")
    print("=" * 78)
    print(f"{'DOF':<10s} {'INTACT (max/min)':<28s} {'RESIDUAL (max/min)':<28s}")
    print("-" * 78)
    print(f"{'Surge [kN]':<10s} {intact.max_surge:8.1f} / {intact.min_surge:8.1f}      "
          f"{residual.max_surge:8.1f} / {residual.min_surge:8.1f}")
    print(f"{'Sway  [kN]':<10s} {intact.max_sway:8.1f} / {intact.min_sway:8.1f}      "
          f"{residual.max_sway:8.1f} / {residual.min_sway:8.1f}")
    print(f"{'Yaw [kNm]':<10s} {intact.max_yaw:8.0f} / {intact.min_yaw:8.0f}      "
          f"{residual.max_yaw:8.0f} / {residual.min_yaw:8.0f}")
    print(f"\nresidual diagnostics: max_sway_bow={residual.max_sway_bow:.0f} kN, "
          f"arm_bow={residual.arm_bow:.2f} m, "
          f"max_sway_stern={residual.max_sway_stern:.0f} kN, "
          f"arm_stern={residual.arm_stern:.2f} m")

    # ---- Per-cell screening ----
    print("\n" + "=" * 110)
    print("REGIME-B SCREENING per cell (using brucon pre-WCF Tx/Ty/Tz as demand proxy)")
    print(f"t_horizon = {T_HORIZON_S:.0f} s")
    print("=" * 110)
    hdr = (f"{'cell':<14s} | "
           f"{'mu_y':>6s} {'sig_y':>6s} {'Td_y':>5s} | "
           f"{'p_sat_y':>8s} {'rate_y':>7s} {'N_sat_y':>8s} {'TL_y':>5s} | "
           f"{'p_sat_z':>8s} {'N_sat_z':>8s} {'TL_z':>5s}")
    print(hdr)
    print("-" * 110)

    for cell in CELLS:
        agg = _ensemble_demand(cell)
        if agg is None:
            print(f"{cell:<14s} | (no data)")
            continue

        ty = agg["Ty"]
        tz = agg["Tz"]

        sat_y = saturation_probability(ty["mu"], ty["sigma"],
                                       residual.max_sway, residual.min_sway)
        sat_z = saturation_probability(tz["mu"], tz["sigma"],
                                       residual.max_yaw, residual.min_yaw)
        omega_y = 2 * np.pi / max(ty["T_decorr"], 1.0)
        omega_z = 2 * np.pi / max(tz["T_decorr"], 1.0)
        rate_y = saturation_event_rate(ty["mu"], ty["sigma"], omega_y,
                                       residual.max_sway, residual.min_sway)
        rate_z = saturation_event_rate(tz["mu"], tz["sigma"], omega_z,
                                       residual.max_yaw, residual.min_yaw)
        N_sat_y = rate_y["rate_either"] * T_HORIZON_S
        N_sat_z = rate_z["rate_either"] * T_HORIZON_S

        print(f"{cell:<14s} | "
              f"{ty['mu']:6.0f} {ty['sigma']:6.0f} {ty['T_decorr']:5.1f} | "
              f"{sat_y['p_either']:8.4f} {rate_y['rate_either']:7.4f} "
              f"{N_sat_y:8.3f} {traffic_light(N_sat_y):>5s} | "
              f"{sat_z['p_either']:8.4f} {N_sat_z:8.3f} {traffic_light(N_sat_z):>5s}")


if __name__ == "__main__":
    main()
