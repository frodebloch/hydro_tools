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
from cqa.operability_polar import operability_polar  # noqa: E402


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
        Vw_max=45.0,
        Vc_m_s=FORECAST_VC,
        T_op_s=20.0 * 60.0,
        quantile_p=0.90,
        Hs_fixed=FORECAST_HS,
        Tp_fixed=FORECAST_TP,
        wind_offset_rad=np.deg2rad(WIND_OFFSET_DEG),
        current_offset_rad=np.deg2rad(CURRENT_OFFSET_DEG),
    )

    th = polar.theta_rel_rad
    th_closed = np.concatenate([th, th[:1]])

    def close(a):
        return np.concatenate([a, a[:1]])

    warn = close(polar.pos_warn_Vw)
    alarm = close(polar.pos_alarm_Vw)

    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Operable region: inside the warn boundary.
    ax.fill(th_closed, warn, color="#2ca02c", alpha=0.22,
            label="operable (below warn)")
    # Caution band between warn and alarm.
    ax.fill_between(th_closed, warn, alarm, color="#ff9900", alpha=0.28,
                    label="caution (warn to alarm)")

    ax.plot(th_closed, warn, color="#2ca02c", lw=2.0)
    ax.plot(th_closed, alarm, color="#d62728", lw=2.0,
            label="alarm boundary")

    # The forecast wind speed as a ring: where the ring sits outside the
    # alarm boundary, that heading is not operable in this forecast.
    ax.plot(th_closed, np.full_like(th_closed, FORECAST_VW),
            color="#1f77b4", lw=2.4, ls="--",
            label=f"forecast wind {FORECAST_VW:.0f} m/s")

    # Shade the headings where the forecast wind already sits outside
    # the alarm boundary - the no-go sector for this forecast.
    nogo = alarm < FORECAST_VW
    if nogo.any():
        ax.fill_between(th_closed, 0.0, ax.get_rmax(), where=nogo,
                        color="#d62728", alpha=0.13, zorder=0,
                        label="no-go at forecast wind")

    ax.set_rmax(float(np.nanmax(alarm)) * 1.08)
    ax.set_rlabel_position(112.5)
    ax.set_title(
        "Forecast-conditioned operability polar\n"
        f"forecast sea state Hs={FORECAST_HS} m, Tp={FORECAST_TP} s, "
        f"Vc={FORECAST_VC} m/s; wind and current veered "
        f"{WIND_OFFSET_DEG:.0f}\u00b0 from the waves\n"
        f"radius = wind speed [m/s]; heading = vessel heading",
        fontsize=10, pad=18,
    )
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.10), fontsize=8)
    fig.tight_layout()

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


if __name__ == "__main__":
    main()
