"""Operability polar demo: V_w boundary at which the intact-prior
amber/red traffic light flips, swept over relative weather direction.

This is a "Level-3-light" capability plot in the spirit of
DNV-ST-0111 / IMCA M140, but reading operability (footprint vs IMCA
M254 radii, telescope vs end-stops) instead of pure thrust capability.
Sea state at each V_w is collapsed to (Hs, Tp) via the
Pierson-Moskowitz fully-developed wind-wave law (DNV-RP-C205 sec
3.5.5.4).

Run:
    python scripts/run_operability_polar_demo.py
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

from cqa import (
    csov_default_config,
    GangwayJointState,
    operability_polar,
    plot_operability_polar,
    load_pdstrip_rao,
)

PDSTRIP_PATH = Path.home() / "src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"


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

    # The pdstrip-driven sigma_L_wave is currently O(5 s) per call --
    # too slow for the inner loop of a polar sweep (~1300 evals).
    # For the prototype demo we run the polar with the parametric
    # PM-Newman drift PSD (cfg.wave_drift) and sigma_L_wave = 0,
    # which captures the slow-drift / wind-gust DP content (the dominant
    # contribution to the position footprint). The wave-frequency
    # telescope band is omitted; this is documented in the printout
    # and to be revisited once sigma_L_wave is optimised for batch
    # evaluation.
    use_rao = False
    rao_table = None
    if use_rao and PDSTRIP_PATH.exists():
        rao_table = load_pdstrip_rao(PDSTRIP_PATH)
    else:
        print("NOTE: running with parametric drift PSD only "
              "(sigma_L_wave omitted) for tractable runtime.")

    print("\nSweeping operability polar (36 directions x 2 axes x 2 thresholds)...")
    polar = operability_polar(
        cfg, joint,
        n_directions=36,
        Vw_min=2.0,
        Vw_max=30.0,
        Vc_m_s=0.5,
        rao_table=rao_table,
        T_op_s=20.0 * 60.0,
        quantile_p=0.90,
        bisect_tol_m_s=0.25,
    )

    # Console summary: worst direction for each axis / threshold.
    def _worst_dir_deg(arr: np.ndarray) -> tuple[float, float]:
        i = int(np.argmin(arr))
        return float(np.degrees(polar.theta_rel_rad[i])), float(arr[i])

    th_pwarn, vw_pwarn = _worst_dir_deg(polar.pos_warn_Vw)
    th_palarm, vw_palarm = _worst_dir_deg(polar.pos_alarm_Vw)
    th_gwarn, vw_gwarn = _worst_dir_deg(polar.gw_warn_Vw)
    th_galarm, vw_galarm = _worst_dir_deg(polar.gw_alarm_Vw)

    print(f"\n=== Operability polar summary "
          f"(T_op={polar.T_op_s/60:.0f} min, Vc={polar.Vc_m_s:.1f} m/s, "
          f"P{polar.quantile_p*100:.0f}) ===")
    print(f"Vessel base position  (warn {polar.pos_warn_radius_m:.1f} m / "
          f"alarm {polar.pos_alarm_radius_m:.1f} m IMCA M254)")
    print(f"  worst-direction warn boundary:  {vw_pwarn:5.1f} m/s "
          f"at theta_rel = {th_pwarn:5.0f} deg")
    print(f"  worst-direction alarm boundary: {vw_palarm:5.1f} m/s "
          f"at theta_rel = {th_palarm:5.0f} deg")
    print(f"  best-direction alarm boundary:  "
          f"{float(np.max(polar.pos_alarm_Vw)):5.1f} m/s")
    print()
    print(f"Gangway telescope     (warn {polar.gw_warn_m:.2f} m / "
          f"alarm {polar.gw_alarm_m:.2f} m, 60%/80% of stroke)")
    print(f"  worst-direction warn boundary:  {vw_gwarn:5.1f} m/s "
          f"at theta_rel = {th_gwarn:5.0f} deg")
    print(f"  worst-direction alarm boundary: {vw_galarm:5.1f} m/s "
          f"at theta_rel = {th_galarm:5.0f} deg")
    print(f"  best-direction alarm boundary:  "
          f"{float(np.max(polar.gw_alarm_Vw)):5.1f} m/s")

    # Capping flags worth surfacing.
    if np.any(polar.pos_alarm_capped_high):
        n = int(np.sum(polar.pos_alarm_capped_high))
        print(f"  NOTE: vessel alarm boundary not reached within "
              f"V_w <= {polar.Vw_max:.0f} m/s in {n} directions.")
    if np.any(polar.gw_alarm_capped_high):
        n = int(np.sum(polar.gw_alarm_capped_high))
        print(f"  NOTE: gangway alarm boundary not reached within "
              f"V_w <= {polar.Vw_max:.0f} m/s in {n} directions.")

    fig = plot_operability_polar(polar)
    out = os.path.join(HERE, "csov_operability_polar.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
