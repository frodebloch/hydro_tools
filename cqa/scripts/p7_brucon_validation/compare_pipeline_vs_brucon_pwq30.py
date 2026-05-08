"""Compare the existing CQA decision-matrix WCFDI peak prediction (15-state
linearised ``transient.wcfdi_transient`` driven) against:
    (a) brucon ground truth at pwq30 (this session)
    (b) the validated 27-state observer-augmented model (transient_obs.py)

Result (recorded 2026-05-08):
    pipeline (15-state):       0.176 m   -> traffic green
    cqa-27 transient:         -0.504 m   (mean of body-sway peak)
    brucon ensemble mean:     -0.683 m   (Delta-sway peak)
    brucon per-seed mean:     -1.35  m   (mean depth across seeds)
    IMCA radii:    warn=2.0 m, alarm=4.0 m

The 15-state pipeline under-predicts brucon by ~4x. cqa-27 closes that gap
to ~25 percent. This script is the regression target for the upcoming
'wire transient_obs into decision_matrix' work.

Run:
    PYTHONPATH=. .venv/bin/python scripts/p7_brucon_validation/compare_pipeline_vs_brucon_pwq30.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cqa import csov_default_config, GangwayJointState, ForecastSlot  # noqa: E402
from cqa.decision_matrix import evaluate_decision_cell  # noqa: E402


# pwq30 conditions (from cqa/scripts/p7_brucon_validation/work/pwq30_seed*.lua)
HS = 4.19571865443425
TP = 10.22443464601827
THETA_ENV_COMPASS = np.deg2rad(210.0)   # wave from 210 deg
HEADING_COMPASS = np.deg2rad(180.0)     # vessel bow south -> beam port to weather
# theta_rel = 210 - 180 = +30 deg (bow-quartering port)


def main():
    cfg = csov_default_config()
    joint = GangwayJointState(L=25.0, h=12.0, alpha_g=0.0, beta_g=np.deg2rad(15.0))
    slot = ForecastSlot(
        label="pwq30", Vw=14.0, Hs=HS, Tp=TP, Vc=0.0,
        theta_env_compass=THETA_ENV_COMPASS,
    )
    cell = evaluate_decision_cell(cfg, joint, slot, HEADING_COMPASS)

    print("=== CQA decision-matrix cell at pwq30 conditions ===")
    print(f"theta_rel = {np.rad2deg(cell.theta_rel):.1f} deg (expect +30)")
    print()
    print(f"INTACT axis:")
    print(f"  pos_a_p90  = {cell.intact_pos_a_p90_m:.3f} m   traffic={cell.intact_pos_traffic}")
    print(f"  gw_a_p90   = {cell.intact_gw_a_p90_m:.3f} m    traffic={cell.intact_gw_traffic}")
    print()
    print(f"WCFDI axis (15-state linearised transient + 0.674-sigma envelope):")
    print(f"  pos_peak   = {cell.wcfdi_pos_peak_m:.3f} m   traffic={cell.wcfdi_pos_traffic}")
    print(f"  gw_peak    = {cell.wcfdi_gw_peak_m:.3f} m    traffic={cell.wcfdi_gw_traffic}")
    print(f"  bistability= {cell.wcfdi_bistability_score:.3f}")
    print(f"  cqa_violated = {cell.wcfdi_cqa_violated}")
    print(f"OVERALL: {cell.overall_traffic}")

    print()
    print("=== Brucon ground truth at pwq30 (this session, 30 seeds) ===")
    print(f"  intact LF sigma_y      = 0.765 m  (mean {-0.61:.3f} m)")
    print(f"  intact WF sigma_y      = 0.339 m")
    print(f"  intact total sigma_y   = 0.840 m")
    print(f"  WCFDI ensemble-mean Delta-sway peak = -0.683 m at +34 s")
    print(f"  WCFDI per-seed mean depth          = -1.35  m (std 0.56)")
    print(f"  cqa-27 transient (validated)        = -0.504 m at +23 s")
    print()
    print("=== Gap ===")
    pipe = cell.wcfdi_pos_peak_m
    print(f"  pipeline (15-state) pos_peak = {pipe:.3f} m")
    print(f"  vs brucon ensemble mean      = {pipe / 0.683:.3f} of brucon")
    print(f"  vs cqa-27 (validated)        = {pipe / 0.504:.3f} of cqa-27")
    print()
    print("Action: wire cqa.transient_obs into decision_matrix._wcfdi_peak_at_forecast.")


if __name__ == "__main__":
    main()
