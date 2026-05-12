"""Compare the existing CQA decision-matrix WCFDI peak prediction (15-state
linearised ``transient.wcfdi_transient`` driven) against:
    (a) brucon ground truth at pwq30 (this session)
    (b) the validated 27-state observer-augmented model (transient_obs.py)

Result (recorded 2026-05-08, post-wiring):
    pipeline 15-state legacy:    0.176 m   -> traffic green
    pipeline 27-state NEW:       1.124 m   -> traffic green (mean only: 0.895 m)
    cqa-27 transient (direct):  -0.504 m   (sway only, brucon-fed tau_lost)
    brucon ensemble mean R:      0.692 m   at +34 s (radial)
    brucon ensemble mean dy:    -0.641 m   at +34 s
    brucon ensemble mean dx:     0.402 m   at +28 s
    IMCA radii:    warn=2.0 m, alarm=4.0 m

Per-component shape match (cqa-27 forecast pipeline, mean only):
    cqa-27: |eta_x|=0.36 at t=28s, |eta_y|=0.82 at t=25s, R=0.90 at t=28s
    brucon: |dx|  =0.40 at t=28s, |dy|  =0.64 at t=34s, R=0.69 at t=34s
Surge match: 0.36 vs 0.40 (excellent). Sway: 0.82 vs 0.64 (overshoot 28%).

The 27-state pipeline closes the gap from 4x low (15-state) to 1.6x high
on the radial peak. The remaining overshoot is a consequence of the
brucon-empirical ``gamma_immediate=0.5`` leading to slightly larger
tau_lost than brucon's measured allocator deficit. Tunable via
``WcfdiScenario.gamma_immediate``.

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
    # CSOV forward gangway: port-pointing (alpha_g = -pi/2). See
    # analysis.md sec.12.21.16. beta_g=15 deg kept from the original
    # comparison setup.
    joint = GangwayJointState(L=25.0, h=12.0, alpha_g=-np.pi / 2.0,
                              beta_g=np.deg2rad(15.0))
    slot = ForecastSlot(
        label="pwq30", Vw=14.0, Hs=HS, Tp=TP, Vc=0.0,
        theta_env_compass=THETA_ENV_COMPASS,
    )
    cell15 = evaluate_decision_cell(cfg, joint, slot, HEADING_COMPASS)
    cell27 = evaluate_decision_cell(
        cfg, joint, slot, HEADING_COMPASS, use_obs_transient=True,
    )

    print("=== CQA decision-matrix cell at pwq30 conditions ===")
    print(f"theta_rel = {np.rad2deg(cell15.theta_rel):.1f} deg (expect +30)")
    print()
    print(f"INTACT axis (same in both pipelines):")
    print(f"  pos_a_p90  = {cell15.intact_pos_a_p90_m:.3f} m   traffic={cell15.intact_pos_traffic}")
    print(f"  gw_a_p90   = {cell15.intact_gw_a_p90_m:.3f} m    traffic={cell15.intact_gw_traffic}")
    print()
    print(f"WCFDI axis -- 15-state linearised transient (legacy):")
    print(f"  pos_peak   = {cell15.wcfdi_pos_peak_m:.3f} m   traffic={cell15.wcfdi_pos_traffic}")
    print(f"  gw_peak    = {cell15.wcfdi_gw_peak_m:.3f} m    traffic={cell15.wcfdi_gw_traffic}")
    print(f"  bistability= {cell15.wcfdi_bistability_score:.3f}")
    print(f"  cqa_violated = {cell15.wcfdi_cqa_violated}")
    print(f"  OVERALL: {cell15.overall_traffic}")
    print()
    print(f"WCFDI axis -- 27-state observer-augmented (NEW, validated):")
    print(f"  pos_peak   = {cell27.wcfdi_pos_peak_m:.3f} m   traffic={cell27.wcfdi_pos_traffic}")
    print(f"  gw_peak    = {cell27.wcfdi_gw_peak_m:.3f} m    traffic={cell27.wcfdi_gw_traffic}")
    print(f"  bistability= {cell27.wcfdi_bistability_score:.3f}")
    print(f"  cqa_violated = {cell27.wcfdi_cqa_violated}")
    print(f"  OVERALL: {cell27.overall_traffic}")

    print()
    print("=== Brucon ground truth at pwq30 (this session, 30 seeds) ===")
    print(f"  intact LF sigma_y      = 0.765 m  (mean {-0.61:.3f} m)")
    print(f"  intact WF sigma_y      = 0.339 m")
    print(f"  intact total sigma_y   = 0.840 m")
    print(f"  WCFDI ensemble-mean radial peak    = 0.692 m at +34 s")
    print(f"  WCFDI ensemble-mean |dx_body| peak = 0.402 m at +28 s (surge)")
    print(f"  WCFDI ensemble-mean |dy_body| peak = 0.641 m at +34 s (sway)")
    print(f"  cqa-27 transient (validated direct, sway only) = -0.504 m at +23 s")
    print()
    print("=== Gap vs brucon ensemble-mean radial peak (0.692 m) ===")
    print(f"  pipeline 15-state:  pos_peak = {cell15.wcfdi_pos_peak_m:.3f} m  "
          f"({cell15.wcfdi_pos_peak_m / 0.692:+.3f}x)")
    print(f"  pipeline 27-state:  pos_peak = {cell27.wcfdi_pos_peak_m:.3f} m  "
          f"({cell27.wcfdi_pos_peak_m / 0.692:+.3f}x)")
    print()
    print("Per-component (mean only, k_sigma=0):")
    print("  cqa-27 forecast:  |eta_x|=0.36  |eta_y|=0.82  R=0.90  at t=28 s")
    print("  brucon ensemble:  |dx|  =0.40  |dy|  =0.64  R=0.69  at t=28-34 s")
    print("  Surge match excellent. Sway overshoots 28%.")


if __name__ == "__main__":
    main()
