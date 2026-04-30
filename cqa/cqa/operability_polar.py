"""Operability polar: V_w at which the intact-prior traffic light flips,
swept over relative weather direction. A "Level-3-light" capability plot
in the spirit of DNV-ST-0111, but with operability (footprint vs IMCA
radii, telescope vs end-stops) replacing pure thrust capability.

Concept
-------
Standard DP capability plots (IMCA M140 / DNV-ST-0111) sweep weather
direction and report the maximum sustainable wind speed at which the
thrusters can still hold station. That is a *thrust* limit.

This polar instead sweeps direction and reports, per axis (vessel base
position OR gangway telescope), the V_w at which the intact-prior P90
quantile of the running-max excursion crosses the IMCA M254 warn (amber)
and alarm (red) thresholds:

  Vessel:  warn = position_warning_radius_m  (default 2 m)
           alarm = position_alarm_radius_m   (default 4 m)

  Gangway: warn = gw_warn_frac * worst_side_stroke   (default 0.60 * stroke)
           alarm = gw_alarm_frac * worst_side_stroke (default 0.80 * stroke)

The sea state at each V_w is collapsed to (Hs, Tp) via the
Pierson-Moskowitz fully-developed wind-wave law (DNV-ST-0111 App.F /
DNV-RP-C205 sec 3.5.5.4) so the polar has a single radial axis,
operator-familiar wind speed.

Why "Level-3-light"
-------------------
* Level 1 = static thrust capability (IMCA M140 rev.2).
* Level 2 = dynamic capability (time-domain / Lyapunov in DP-cl-2).
* Level 3 = consequence-aware capability (post-fault excursion).
This polar is intact-only (Level 2 in spirit) but reads the operability
limit instead of the thrust limit, so it sits between the two.
WCFDI overlay is a follow-up turn.

Bisection vs grid
-----------------
For a fixed direction the P90 footprint is monotonically increasing in
V_w (more energetic disturbance -> larger excursion variance ->
larger inverse-Rice quantile). We bisect V_w in [Vw_min, Vw_max] until
the chosen threshold is hit. Tolerance is set in V_w directly (default
0.1 m/s) which translates to <2% error on Hs.

If the threshold is never breached at Vw_max the boundary is capped to
Vw_max (and flagged in the result's `*_capped_high`); if it is already
breached at Vw_min the boundary is set to Vw_min (and flagged in
`*_capped_low`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .config import CqaConfig
from .gangway import GangwayJointState
from .sea_state_relations import pm_hs_from_vw, pm_tp_from_vw


@dataclass(frozen=True)
class OperabilityPolar:
    """Per-direction wind-speed boundary at which each operability axis
    flips amber (warn) and red (alarm).

    Fields
    ------
    theta_rel_rad : (N,) array of relative weather directions [rad].
    pos_warn_Vw   : (N,) wind speed [m/s] at which vessel-base footprint
                    P90 hits the IMCA warn radius. NaN-safe; capped to
                    [Vw_min, Vw_max].
    pos_alarm_Vw  : (N,) same for the alarm radius.
    gw_warn_Vw    : (N,) wind speed at which telescope P90 hits 60% stroke.
    gw_alarm_Vw   : (N,) wind speed at which telescope P90 hits 80% stroke.
    pos_warn_capped_low / _high  : (N,) bool. True if the warn boundary
                    saturated against Vw_min / Vw_max, respectively.
    pos_alarm_capped_low / _high : ditto for the alarm boundary.
    gw_warn_capped_low / _high   : ditto.
    gw_alarm_capped_low / _high  : ditto.
    Vw_min, Vw_max               : sweep range used [m/s].
    Vc_m_s                       : current speed used [m/s] (constant).
    pos_warn_radius_m, pos_alarm_radius_m : IMCA M254 radii [m].
    gw_warn_m, gw_alarm_m        : IMCA M254 telescope thresholds [m].
    quantile_p                   : the running-max quantile used (default 0.90).
    T_op_s                       : planned operation duration [s].
    """

    theta_rel_rad: np.ndarray
    pos_warn_Vw: np.ndarray
    pos_alarm_Vw: np.ndarray
    gw_warn_Vw: np.ndarray
    gw_alarm_Vw: np.ndarray
    pos_warn_capped_low: np.ndarray
    pos_warn_capped_high: np.ndarray
    pos_alarm_capped_low: np.ndarray
    pos_alarm_capped_high: np.ndarray
    gw_warn_capped_low: np.ndarray
    gw_warn_capped_high: np.ndarray
    gw_alarm_capped_low: np.ndarray
    gw_alarm_capped_high: np.ndarray
    Vw_min: float
    Vw_max: float
    Vc_m_s: float
    pos_warn_radius_m: float
    pos_alarm_radius_m: float
    gw_warn_m: float
    gw_alarm_m: float
    quantile_p: float
    T_op_s: float


# ---------------------------------------------------------------------------
# Boundary search
# ---------------------------------------------------------------------------


def _bisect_boundary(
    metric: Callable[[float], float],
    threshold: float,
    Vw_lo: float,
    Vw_hi: float,
    tol: float = 0.1,
    max_iter: int = 64,
) -> tuple[float, bool, bool]:
    """Bisect on V_w to find the smallest V_w with metric(V_w) >= threshold.

    Returns
    -------
    (V_w_boundary, capped_low, capped_high) where:
      capped_low  = True if metric(Vw_lo) >= threshold already (boundary
                    saturated to Vw_lo).
      capped_high = True if metric(Vw_hi) <  threshold still (boundary
                    saturated to Vw_hi).

    Notes
    -----
    Assumes metric is non-decreasing in V_w (PSD energy and thus
    sigma_radial both grow with wind speed; nu_0+ and q are weak
    functions of V_w; the inverse-Rice quantile a_p90 = sigma * f(nu0+ T)
    is dominated by the sigma factor, so monotonicity holds in practice).
    A non-monotone metric would still return a valid bracket but might
    miss earlier crossings.
    """
    m_lo = metric(Vw_lo)
    if m_lo >= threshold:
        return float(Vw_lo), True, False
    m_hi = metric(Vw_hi)
    if m_hi < threshold:
        return float(Vw_hi), False, True
    a, b = Vw_lo, Vw_hi
    for _ in range(max_iter):
        if (b - a) <= tol:
            break
        mid = 0.5 * (a + b)
        if metric(mid) >= threshold:
            b = mid
        else:
            a = mid
    return float(0.5 * (a + b)), False, False


# ---------------------------------------------------------------------------
# Per-direction operating-point evaluator
# ---------------------------------------------------------------------------


def _evaluate_intact_prior_at(
    cfg: CqaConfig,
    joint: GangwayJointState,
    Vw: float,
    Vc: float,
    theta_rel: float,
    *,
    rao_table=None,
    omega_n: tuple[float, float, float],
    zeta: tuple[float, float, float],
    sigma_Vc: float,
    tau_Vc: float,
    T_op_s: float,
    quantile_p: float,
    quantile_lo: float,
    omega_grid: np.ndarray,
    use_pm_for_drift: bool,
):
    """Build the closed loop + disturbance PSDs for one operating point
    and call summarise_intact_prior. Returns the IntactPriorSummary.

    Sea state Hs, Tp derived from V_w via PM (pm_hs_from_vw / pm_tp_from_vw).
    """
    from .vessel import LinearVesselModel, CurrentForceModel
    from .controller import LinearDpController
    from .closed_loop import ClosedLoop
    from .psd import (
        npd_wind_gust_force_psd,
        slow_drift_force_psd_newman,
        current_variability_force_psd,
    )
    from .operator_view import summarise_intact_prior
    from .wave_response import sigma_L_wave as _sigma_L_wave_fn
    from .drift import slow_drift_force_psd_newman_pdstrip

    Hs = pm_hs_from_vw(Vw)
    Tp = pm_tp_from_vw(Vw)

    vp = cfg.vessel
    wp = cfg.wind
    cp = cfg.current
    wd = cfg.wave_drift

    vessel = LinearVesselModel.from_config(vp)
    controller = LinearDpController.from_bandwidth(
        vessel.M, vessel.D, omega_n=omega_n, zeta=zeta,
    )
    cl = ClosedLoop.build(vessel, controller)

    from .psd import WindForceModel
    wind_model = WindForceModel(wp=wp, loa=vp.loa)
    S_wind = npd_wind_gust_force_psd(wind_model, Vw, theta_rel)

    if rao_table is not None:
        S_drift = slow_drift_force_psd_newman_pdstrip(
            rao_table=rao_table, Hs=Hs, Tp=Tp, theta_wave_rel=theta_rel,
        )
    elif use_pm_for_drift:
        S_drift = slow_drift_force_psd_newman(
            (wd.drift_x_amp, wd.drift_y_amp, wd.drift_n_amp),
            Hs, Tp, theta_rel,
        )
    else:
        def S_drift(_w):
            return np.zeros((3, 3))

    current_model = CurrentForceModel(
        cp=cp,
        lateral_area_underwater=vp.lpp * vp.draft,
        frontal_area_underwater=vp.beam * vp.draft,
        loa=vp.loa,
    )
    if Vc > 1e-9:
        F0 = current_model.force(Vc, theta_rel)
        dFdVc = 2.0 * F0 / Vc
    else:
        dFdVc = np.zeros(3)
    S_curr = current_variability_force_psd(dFdVc, sigma_Vc=sigma_Vc, tau=tau_Vc)

    if rao_table is not None and Tp > 0 and Hs > 0:
        wave = _sigma_L_wave_fn(joint, cfg, rao_table, Hs=Hs, Tp=Tp,
                                theta_wave_rel=theta_rel)
        sigma_L_wave_m = wave.sigma_L_wave
    else:
        sigma_L_wave_m = 0.0

    return summarise_intact_prior(
        cl, [S_wind, S_drift, S_curr], cfg, joint,
        T_op_s=T_op_s,
        sigma_L_wave=sigma_L_wave_m,
        Tp_wave_s=Tp if Tp > 0 else 8.0,
        quantiles=(quantile_lo, quantile_p),
        omega_grid=omega_grid,
    )


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def operability_polar(
    cfg: CqaConfig,
    joint: GangwayJointState,
    *,
    n_directions: int = 36,
    Vw_min: float = 0.5,
    Vw_max: float = 30.0,
    Vc_m_s: float = 0.5,
    rao_table=None,
    sigma_Vc: float = 0.1,
    tau_Vc: float = 600.0,
    T_op_s: float = 20.0 * 60.0,
    quantile_p: float = 0.90,
    bisect_tol_m_s: float = 0.1,
    omega_grid: Optional[np.ndarray] = None,
) -> OperabilityPolar:
    """Sweep relative weather direction and find, per axis, the V_w at
    which the intact-prior amber and red operability thresholds are hit.

    The sea state at each V_w follows DNV-ST-0111 / Pierson-Moskowitz:
    Hs = 0.21 * V_w^2 / g, T_p = 2*pi*V_w / (0.877*g). Wind, wave and
    current are taken collinear (same theta_rel), matching the
    DP capability convention.

    Parameters
    ----------
    cfg : CqaConfig. Vessel + controller + IMCA radii. The controller
        tuning omega_n / zeta come from cfg.controller.
    joint : current gangway joint state. Telescope length L sets the
        worst-side stroke and therefore the gangway warn / alarm
        thresholds.
    n_directions : int. Number of relative directions sampled in
        [0, 2*pi). Default 36 (every 10 deg).
    Vw_min, Vw_max : float. Sweep range [m/s]. Default 0.5 ... 30.
    Vc_m_s : float. Constant current speed [m/s]. Default 0.5.
    rao_table : optional pdstrip-derived RAO table. If provided,
        wave-frequency telescope contribution is included via
        sigma_L_wave AND slow-drift PSD uses pdstrip-Newman; otherwise
        the parametric PM-Newman drift PSD from cfg.wave_drift is used
        and sigma_L_wave defaults to 0.
    sigma_Vc, tau_Vc : current-variability std [m/s] and correlation
        time [s]. Defaults match the demo.
    T_op_s : planned operation duration [s] feeding the inverse Rice.
        Default 1200 s (20 min).
    quantile_p : the running-max quantile used as the boundary metric.
        Default 0.90 (P90).
    bisect_tol_m_s : V_w bisection tolerance [m/s]. Default 0.1.
    omega_grid : optional integration grid for axis_psd. None = default.

    Returns
    -------
    OperabilityPolar. See the dataclass docstring for fields.

    Notes
    -----
    Cost: ~ n_directions * 4 * O(log2((Vw_max - Vw_min) / tol)) closed-loop
    + intact-prior evaluations. With defaults (36 dirs, 0.1 m/s tol over
    30 m/s span), that is ~36 * 4 * 9 ~= 1300 evaluations -- a few seconds
    on the prototype stack.
    """
    if Vw_min <= 0.0:
        raise ValueError(f"Vw_min must be > 0, got {Vw_min}")
    if Vw_max <= Vw_min:
        raise ValueError(
            f"Vw_max must be > Vw_min; got Vw_min={Vw_min}, Vw_max={Vw_max}"
        )
    if not (0.0 < quantile_p < 1.0):
        raise ValueError(f"quantile_p must be in (0, 1), got {quantile_p}")
    if n_directions < 4:
        raise ValueError(f"n_directions must be >= 4, got {n_directions}")

    omega_n = cfg.controller.omega_n
    zeta = cfg.controller.zeta

    # Quantile_lo is required by summarise_intact_prior's (lo, hi) tuple
    # but only the "hi" boundary is used here; pick a small valid value.
    quantile_lo = min(0.10, 0.5 * quantile_p)

    use_pm_for_drift = True

    # Pull the IMCA radii once (constant across the sweep).
    pos_warn_r = float(cfg.operational_limits.position_warning_radius_m)
    pos_alarm_r = float(cfg.operational_limits.position_alarm_radius_m)

    # Gangway thresholds depend on the telescope geometry and the
    # current setpoint L0 -- constant across the sweep too.
    L0 = float(joint.L)
    L_min = float(cfg.gangway.telescope_min)
    L_max = float(cfg.gangway.telescope_max)
    stroke = min(max(L0 - L_min, 0.0), max(L_max - L0, 0.0))
    gw_warn_m = 0.60 * stroke
    gw_alarm_m = 0.80 * stroke

    thetas = np.linspace(0.0, 2.0 * np.pi, n_directions, endpoint=False)

    pos_warn_Vw = np.zeros(n_directions)
    pos_alarm_Vw = np.zeros(n_directions)
    gw_warn_Vw = np.zeros(n_directions)
    gw_alarm_Vw = np.zeros(n_directions)
    pos_warn_lo = np.zeros(n_directions, dtype=bool)
    pos_warn_hi = np.zeros(n_directions, dtype=bool)
    pos_alarm_lo = np.zeros(n_directions, dtype=bool)
    pos_alarm_hi = np.zeros(n_directions, dtype=bool)
    gw_warn_lo = np.zeros(n_directions, dtype=bool)
    gw_warn_hi = np.zeros(n_directions, dtype=bool)
    gw_alarm_lo = np.zeros(n_directions, dtype=bool)
    gw_alarm_hi = np.zeros(n_directions, dtype=bool)

    for i, theta in enumerate(thetas):
        # Per-direction caches for the four bisections so we don't pay
        # for the same Vw twice.
        cache_pos: dict[float, float] = {}
        cache_gw: dict[float, float] = {}

        def _eval(Vw: float):
            # Cache key rounded to 1e-6 m/s.
            key = round(float(Vw), 6)
            if key in cache_pos:
                return cache_pos[key], cache_gw[key]
            res = _evaluate_intact_prior_at(
                cfg, joint, Vw, Vc_m_s, theta,
                rao_table=rao_table,
                omega_n=omega_n, zeta=zeta,
                sigma_Vc=sigma_Vc, tau_Vc=tau_Vc,
                T_op_s=T_op_s,
                quantile_p=quantile_p,
                quantile_lo=quantile_lo,
                omega_grid=omega_grid,
                use_pm_for_drift=use_pm_for_drift,
            )
            cache_pos[key] = float(res.pos_a_p90)
            cache_gw[key] = float(res.gw_a_p90)
            return cache_pos[key], cache_gw[key]

        def metric_pos(Vw: float) -> float:
            return _eval(Vw)[0]

        def metric_gw(Vw: float) -> float:
            return _eval(Vw)[1]

        v, lo, hi = _bisect_boundary(metric_pos, pos_warn_r,
                                     Vw_min, Vw_max, tol=bisect_tol_m_s)
        pos_warn_Vw[i], pos_warn_lo[i], pos_warn_hi[i] = v, lo, hi

        v, lo, hi = _bisect_boundary(metric_pos, pos_alarm_r,
                                     Vw_min, Vw_max, tol=bisect_tol_m_s)
        pos_alarm_Vw[i], pos_alarm_lo[i], pos_alarm_hi[i] = v, lo, hi

        if stroke > 0.0:
            v, lo, hi = _bisect_boundary(metric_gw, gw_warn_m,
                                         Vw_min, Vw_max, tol=bisect_tol_m_s)
            gw_warn_Vw[i], gw_warn_lo[i], gw_warn_hi[i] = v, lo, hi
            v, lo, hi = _bisect_boundary(metric_gw, gw_alarm_m,
                                         Vw_min, Vw_max, tol=bisect_tol_m_s)
            gw_alarm_Vw[i], gw_alarm_lo[i], gw_alarm_hi[i] = v, lo, hi
        else:
            # Telescope at an end-stop: every V_w is "red" immediately.
            gw_warn_Vw[i] = Vw_min
            gw_alarm_Vw[i] = Vw_min
            gw_warn_lo[i] = True
            gw_alarm_lo[i] = True

    return OperabilityPolar(
        theta_rel_rad=thetas,
        pos_warn_Vw=pos_warn_Vw,
        pos_alarm_Vw=pos_alarm_Vw,
        gw_warn_Vw=gw_warn_Vw,
        gw_alarm_Vw=gw_alarm_Vw,
        pos_warn_capped_low=pos_warn_lo,
        pos_warn_capped_high=pos_warn_hi,
        pos_alarm_capped_low=pos_alarm_lo,
        pos_alarm_capped_high=pos_alarm_hi,
        gw_warn_capped_low=gw_warn_lo,
        gw_warn_capped_high=gw_warn_hi,
        gw_alarm_capped_low=gw_alarm_lo,
        gw_alarm_capped_high=gw_alarm_hi,
        Vw_min=float(Vw_min),
        Vw_max=float(Vw_max),
        Vc_m_s=float(Vc_m_s),
        pos_warn_radius_m=pos_warn_r,
        pos_alarm_radius_m=pos_alarm_r,
        gw_warn_m=float(gw_warn_m),
        gw_alarm_m=float(gw_alarm_m),
        quantile_p=float(quantile_p),
        T_op_s=float(T_op_s),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_operability_polar(
    polar: OperabilityPolar,
    title: Optional[str] = None,
):
    """Two-panel polar (vessel | gangway) of the V_w boundaries.

    Convention: theta = 0 along the +x body axis (bow); polar plot is
    drawn with theta=0 to the right and counter-clockwise CCW (matplotlib
    default). For a "bow points up" orientation set
    ``ax.set_theta_zero_location('N')`` and ``ax.set_theta_direction(-1)``
    -- done below so the polar reads like a standard DP capability plot.

    The amber boundary is drawn as a yellow ring, the red boundary as a
    red ring; the operable region is shaded green INSIDE the amber ring
    and amber between amber/red, mirroring the IMCA M254 traffic-light
    semantics.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6), subplot_kw=dict(projection="polar"),
    )

    # Close the polygons.
    th = np.concatenate([polar.theta_rel_rad, polar.theta_rel_rad[:1]])

    def _close(arr: np.ndarray) -> np.ndarray:
        return np.concatenate([arr, arr[:1]])

    for ax, name, warn_Vw, alarm_Vw, threshold_label in [
        (axes[0], "Vessel base position",
         polar.pos_warn_Vw, polar.pos_alarm_Vw,
         f"warn {polar.pos_warn_radius_m:.1f} m / alarm {polar.pos_alarm_radius_m:.1f} m"),
        (axes[1], "Gangway telescope",
         polar.gw_warn_Vw, polar.gw_alarm_Vw,
         f"warn {polar.gw_warn_m:.2f} m / alarm {polar.gw_alarm_m:.2f} m"),
    ]:
        warn_c = _close(warn_Vw)
        alarm_c = _close(alarm_Vw)

        # Green = always operable (inside the warn ring).
        ax.fill(th, warn_c, color="#bce6b0", alpha=0.7, label="green (operable)")
        # Amber annulus (warn .. alarm).
        ax.fill_between(th, warn_c, alarm_c,
                        color="#f6d57a", alpha=0.7, label="amber (caution)")
        # Red region (outside alarm) -- shade out to Vw_max so the
        # operator immediately sees "no go" sectors.
        ax.fill_between(th, alarm_c, np.full_like(alarm_c, polar.Vw_max),
                        color="#f29c9c", alpha=0.5, label="red (no go)")

        # Boundary lines on top.
        ax.plot(th, warn_c, color="#cc8800", lw=2.0)
        ax.plot(th, alarm_c, color="#990000", lw=2.0)

        # Make the polar feel like a DP capability plot.
        ax.set_theta_zero_location("N")  # bow up
        ax.set_theta_direction(-1)       # clockwise
        ax.set_rlim(0.0, polar.Vw_max)
        ax.set_rlabel_position(135.0)
        ax.set_title(f"{name}\n({threshold_label})", fontsize=11, pad=15)

    # Single shared legend below.
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#bce6b0", alpha=0.7,
                      label=f"P{polar.quantile_p*100:.0f} < warn  (operable)"),
        plt.Rectangle((0, 0), 1, 1, fc="#f6d57a", alpha=0.7,
                      label=f"warn <= P{polar.quantile_p*100:.0f} < alarm  (caution)"),
        plt.Rectangle((0, 0), 1, 1, fc="#f29c9c", alpha=0.5,
                      label=f"P{polar.quantile_p*100:.0f} >= alarm  (no go)"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)

    if title is None:
        title = (f"Intact-prior operability polar  (T_op = "
                 f"{polar.T_op_s/60:.0f} min, Vc = {polar.Vc_m_s:.1f} m/s, "
                 f"Hs/Tp from PM(V_w))")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


__all__ = [
    "OperabilityPolar",
    "operability_polar",
    "plot_operability_polar",
]
