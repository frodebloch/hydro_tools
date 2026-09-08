"""Forecast-conditioned operability polar (MTS DP Houston 2026 paper).

Unlike the design-sweep polar (run_operability_polar_demo.py), which
grows the sea state with wind speed via the DNV-ST-0111 /
Pierson-Moskowitz fully-developed law, this chart pins the wave
condition and current to the *forecast* values and varies only the
wind. For each vessel heading it finds the wind speed at which the
motion limit is reached, and marks the forecast wind speed against
that boundary so the operator reads off the remaining margin per
heading directly.

The question answered is therefore operational rather than
design-stage: "given the sea state we expect during this operation,
which headings work, and how much more wind can each of them take
before we run out of motion envelope?"

Run:
    python scripts/run_forecast_polar_demo.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cqa import csov_default_config, GangwayJointState  # noqa: E402
from cqa.operability_polar import (  # noqa: E402
    operability_polar,
    wcfdi_operability_overlay,
)


# --- Forecast operating point ------------------------------------------
# Beaufort 6, the nominal DNV-ST-0111 condition also used in the
# simulator validation grid. The forecast is deliberately NOT
# collinear: the wind has veered 90 deg ahead of the sea, which has
# not yet turned to follow it, and the tidal stream runs with the
# wind. This is an ordinary forecast situation and it is what makes
# the operable envelope genuinely asymmetric in vessel heading -- the
# heading that is best for the waves is not the one that is best for
# the wind and current. Only the wind speed varies in the sweep; Hs,
# Tp, Vc and the directional layout are pinned to the forecast.
FORECAST_HS = 3.1       # m   (DNV-ST-0111 Bf 6)
FORECAST_TP = 8.5       # s
FORECAST_VC = 0.75      # m/s
FORECAST_VW = 13.8      # m/s -- the forecast wind speed to mark
WIND_OFFSET_DEG = 90.0    # wind veered this far from the waves
CURRENT_OFFSET_DEG = 90.0  # tidal stream runs with the wind


def main() -> None:
    cfg = csov_default_config()
    gw = cfg.gangway
    L0 = 0.5 * (gw.telescope_min + gw.telescope_max)
    joint = GangwayJointState(
        h=gw.rotation_centre_height_above_base,
        alpha_g=-np.pi / 2.0,  # port
        beta_g=0.0,
        L=L0,
    )

    print("Forecast-conditioned polar")
    print(f"  fixed sea state : Hs={FORECAST_HS} m, Tp={FORECAST_TP} s")
    print(f"  fixed current   : Vc={FORECAST_VC} m/s")
    print(f"  forecast wind   : {FORECAST_VW} m/s, "
          f"veered {WIND_OFFSET_DEG:.0f} deg from the waves")
    print("  sweeping vessel heading, bisecting on wind speed ...")

    polar = operability_polar(
        cfg, joint,
        n_directions=36,
        Vw_min=0.5,
        Vw_max=70.0,
        Vc_m_s=FORECAST_VC,
        T_op_s=20.0 * 60.0,
        quantile_p=0.90,
        Hs_fixed=FORECAST_HS,
        Tp_fixed=FORECAST_TP,
        wind_offset_rad=np.deg2rad(WIND_OFFSET_DEG),
        current_offset_rad=np.deg2rad(CURRENT_OFFSET_DEG),
    )

    print("  evaluating post-WCFDI overlay ...")
    overlay = wcfdi_operability_overlay(
        cfg, joint,
        n_directions=36,
        Vw_min=0.5, Vw_max=70.0, Vc_m_s=FORECAST_VC,
        Hs_fixed=FORECAST_HS, Tp_fixed=FORECAST_TP,
        wind_offset_rad=np.deg2rad(WIND_OFFSET_DEG),
        current_offset_rad=np.deg2rad(CURRENT_OFFSET_DEG),
    )

    th = polar.theta_rel_rad
    th_closed = np.concatenate([th, th[:1]])

    def close(a):
        return np.concatenate([a, a[:1]])

    R_MAX = 40.0   # common radial scale for both panels

    # Threshold labels follow plot_operability_polar so both charts
    # state their limits the same way.
    panels = [
        ("Vessel base position",
         f"warn {polar.pos_warn_radius_m:.1f} m / "
         f"alarm {polar.pos_alarm_radius_m:.1f} m",
         close(polar.pos_warn_Vw), close(polar.pos_alarm_Vw),
         close(overlay.pos_alarm_Vw)),
        ("Gangway telescope",
         f"warn {polar.gw_warn_m:.2f} m / alarm {polar.gw_alarm_m:.2f} m",
         close(polar.gw_warn_Vw), close(polar.gw_alarm_Vw),
         close(overlay.gw_alarm_Vw)),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, 6.0),
        subplot_kw={"projection": "polar"},
    )

    for ax, (title, thr_label, warn, alarm, wcf_alarm) in zip(axes, panels):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # Clip to the plotted range; boundaries beyond R_MAX mean the
        # limit is never reached at any wind speed of interest.
        warn_c = np.minimum(warn, R_MAX)
        alarm_c = np.minimum(alarm, R_MAX)

        # Palette matches plot_operability_polar so the forecast chart
        # reads the same way as the design polars: green operable,
        # amber caution, light red outside the alarm boundary.
        ax.fill(th_closed, warn_c, color="#bce6b0", alpha=0.7,
                label="green (operable)")
        ax.fill_between(th_closed, warn_c, alarm_c,
                        color="#f6d57a", alpha=0.7, label="amber (caution)")
        ax.fill_between(th_closed, alarm_c,
                        np.full_like(alarm_c, R_MAX),
                        color="#f29c9c", alpha=0.5, label="red (no go)")
        ax.plot(th_closed, warn_c, color="#cc8800", lw=2.0)
        ax.plot(th_closed, alarm_c, color="#990000", lw=2.0)
        # Post-WCFDI no-go boundary: the wind speed at which the
        # post-failure motion would cross the alarm limit.
        ax.plot(th_closed, np.minimum(wcf_alarm, R_MAX),
                color="#1a1a1a", lw=2.0, ls="--",
                label="post-WCF alarm boundary")
        # Validity boundary: beyond this the linear post-WCF predictor
        # is in the meta-stable regime and its motion figure should be
        # read as a lower bound. Same role as p_sat on the live panel.
        ax.plot(th_closed, np.minimum(close(overlay.bist_Vw), R_MAX),
                color="#6a3d9a", lw=1.8, ls=":",
                label="post-WCF prediction not trustworthy")
        ax.plot(th_closed, np.full_like(th_closed, FORECAST_VW),
                color="#1f77b4", lw=2.4, ls="--",
                label=f"forecast wind {FORECAST_VW:.1f} m/s")

        ax.set_rmax(R_MAX)
        ax.set_rlabel_position(135.0)
        ax.set_title(f"{title}\n({thr_label})", fontsize=11, pad=15)
        ax.grid(True, alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=9,
               frameon=False)

    fig.suptitle(
        "Forecast-conditioned operability polar  -  "
        f"Hs={FORECAST_HS} m, Tp={FORECAST_TP} s, Vc={FORECAST_VC} m/s, "
        f"wind and current veered {WIND_OFFSET_DEG:.0f}\u00b0 from the waves\n"
        "radius = wind speed [m/s]; angle = vessel heading",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))

    out_dir = Path.home() / "Documents" / "MTS_26" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "csov_forecast_polar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    # Report the margin per heading for the paper text. Margin to the
    # warn boundary is the operationally useful number: it is what the
    # operator has in hand before the operation goes to caution.
    margin = polar.pos_warn_Vw - FORECAST_VW
    worst = int(np.argmin(margin))
    best = int(np.argmax(margin))
    print(f"  warn boundary  : {polar.pos_warn_Vw.min():.1f} to "
          f"{polar.pos_warn_Vw.max():.1f} m/s")
    print(f"  alarm boundary : {polar.pos_alarm_Vw.min():.1f} to "
          f"{polar.pos_alarm_Vw.max():.1f} m/s")
    print(f"  wind margin to caution: {margin.min():+.1f} to "
          f"{margin.max():+.1f} m/s")
    print(f"  tightest heading {np.rad2deg(th[worst]):.0f} deg "
          f"(margin {margin[worst]:+.1f} m/s); "
          f"most open {np.rad2deg(th[best]):.0f} deg "
          f"(margin {margin[best]:+.1f} m/s)")
    bst = overlay.bist_Vw
    print(f"  post-WCF validity boundary: {np.nanmin(bst):.1f} to "
          f"{np.nanmax(bst):.1f} m/s")
    wcf = overlay.pos_alarm_Vw
    n_nogo = int(np.sum(wcf < FORECAST_VW))
    print(f"  post-WCF alarm boundary: {np.nanmin(wcf):.1f} to "
          f"{np.nanmax(wcf):.1f} m/s")
    print(f"  headings where post-WCF alarm is already breached at the "
          f"forecast wind: {n_nogo} of {len(th)}")
    if n_nogo:
        bad = np.rad2deg(th[wcf < FORECAST_VW])
        print(f"    no-go headings: {bad.min():.0f} to {bad.max():.0f} deg")
    gw_binds = int(np.sum(polar.gw_warn_Vw < polar.pos_warn_Vw))
    print(f"  gangway warn boundary: {polar.gw_warn_Vw.min():.1f} to "
          f"{polar.gw_warn_Vw.max():.1f} m/s "
          f"(binds at {gw_binds} of {len(th)} headings)")


if __name__ == "__main__":
    main()
