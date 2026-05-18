"""Forecast-pipeline regime-B saturation validation (sec.12.21.21.16).

For each brucon-validation cell, run ``cqa.transient.wcfdi_transient`` with
the brucon-derived intact + residual thrust polytope (sec.12.21.21.15) and
the calibrated lost-bus deficit + per-DOF reallocation time constant
(sec.12.21.20). Extract the headline ``bistability_risk_score`` and the
per-DOF ``bistability_per_dof``, then cross-tabulate them against the
empirical regime-B clip fractions from ``saturation_regime_scan.py``.

Hypothesis (sec.12.21.21.12):
  * Regime B (sustained saturation in (30, 200] s post-WCF) is the
    dominant failure mode for CSOV bus_port WCF.
  * The ``severity_t = max(0, |mu_tau| - cap(t)) / sigma_tau_cmd(t)``
    construct in ``wcfdi_transient`` -- once the cap envelope is set
    correctly via the brucon polytope -- should predict which cells are
    at risk (score >> 1) and which are safe (score ~ 0).
  * Empirically only ``bf8_h0_w45`` (3/30 seeds clipped) and
    ``bf8_q10_w45`` (15/30, severe) show regime-B saturation.

Usage::

    PYTHONPATH=. .venv/bin/python \\
        scripts/p7_brucon_validation/wcfdi_transient_regime_b_check.py

Prints a per-cell table comparing predicted bistability score against
empirical regime-B clip-fraction. No plots written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))
sys.path.insert(0, str(THIS))

from cqa.transient import wcfdi_transient, WcfdiScenario      # noqa: E402
from cqa.rao import load_pdstrip_rao                          # noqa: E402

from run_comparison import setup_cqa                          # noqa: E402
from saturation_screening import (                            # noqa: E402
    CSOV_THRUSTERS, CSOV_BUS_PORT_LOST, compute_residual_polytope,
)
from saturation_regime_scan import scan_cell                  # noqa: E402

PDSTRIP_PATH = (
    "/home/blofro/src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"
)
_LOST_BUS_CAL_PATH = THIS / "wcfdi_lost_bus_calibration.json"

# Brucon polytope (sec.12.21.21.15): port of basic_allocation.cpp
# ``BasicAllocator::CalculateAvailableThrust`` for the CSOV thruster set.
_INTACT_POLYTOPE = compute_residual_polytope(CSOV_THRUSTERS)
_BUS_PORT_SURV = tuple(
    i for i in range(len(CSOV_THRUSTERS)) if i not in CSOV_BUS_PORT_LOST
)
_RESIDUAL_POLYTOPE = compute_residual_polytope(
    CSOV_THRUSTERS, surviving_indices=_BUS_PORT_SURV,
)
_INTACT_CAP_N_NM = (
    1e3 * min(abs(_INTACT_POLYTOPE.max_surge), abs(_INTACT_POLYTOPE.min_surge)),
    1e3 * min(abs(_INTACT_POLYTOPE.max_sway),  abs(_INTACT_POLYTOPE.min_sway)),
    1e3 * min(abs(_INTACT_POLYTOPE.max_yaw),   abs(_INTACT_POLYTOPE.min_yaw)),
)
_RESIDUAL_CAP_N_NM = (
    1e3 * min(abs(_RESIDUAL_POLYTOPE.max_surge), abs(_RESIDUAL_POLYTOPE.min_surge)),
    1e3 * min(abs(_RESIDUAL_POLYTOPE.max_sway),  abs(_RESIDUAL_POLYTOPE.min_sway)),
    1e3 * min(abs(_RESIDUAL_POLYTOPE.max_yaw),   abs(_RESIDUAL_POLYTOPE.min_yaw)),
)
_ALPHA_FROM_POLYTOPE = tuple(
    _RESIDUAL_CAP_N_NM[i] / _INTACT_CAP_N_NM[i] for i in range(3)
)

# Vessel heading mirrors run_validation_matrix.py.
VESSEL_HEADING_COMPASS = 180.0

# Per-cell environment definitions (mirror run_validation_matrix.py CELLS).
BF6 = dict(Vw=13.8, Hs=3.1, Tp=8.5,  Vc=0.75)
BF8 = dict(Vw=20.7, Hs=5.7, Tp=10.0, Vc=0.75)
BF4 = dict(Vw=7.0,  Hs=1.5, Tp=6.0,  Vc=0.75)
# Half-step BF (sec.12.21.21.30b): see run_validation_matrix.py for rationale.
BF7P5 = dict(Vw=18.975, Hs=5.05, Tp=9.625,  Vc=0.75)
BF8P5 = dict(Vw=22.425, Hs=6.35, Tp=10.375, Vc=0.75)
_BF4_CURR_COMPASS = (VESSEL_HEADING_COMPASS + 45.0) % 360.0  # 225
_BF4_CURR_SPEED = 1.0

# (theta_rel_deg, env, wind_offset_deg, current_compass_abs, current_speed_override)
CELLS: dict[str, tuple[float, dict, float, float | None, float | None]] = {
    "bf4_c1_h0":   (0.0,  BF4, 0.0,  _BF4_CURR_COMPASS, _BF4_CURR_SPEED),
    "bf4_c1_q10":  (10.0, BF4, 0.0,  _BF4_CURR_COMPASS, _BF4_CURR_SPEED),
    "bf6_h0":      (0.0,  BF6, 0.0,  None, None),
    "bf6_h0_w45":  (0.0,  BF6, 45.0, None, None),
    "bf6_q10":     (10.0, BF6, 0.0,  None, None),
    "bf6_q10_w45": (10.0, BF6, 45.0, None, None),
    "bf8_h0":      (0.0,  BF8, 0.0,  None, None),
    "bf8_h0_w45":  (0.0,  BF8, 45.0, None, None),
    "bf8_q10":     (10.0, BF8, 0.0,  None, None),
    "bf8_q10_w45": (10.0, BF8, 45.0, None, None),
    "bf7p5_q10_w45": (10.0, BF7P5, 45.0, None, None),
    "bf8p5_q10_w45": (10.0, BF8P5, 45.0, None, None),
    "pwq30":       (30.0, BF8, 0.0,  None, None),  # waves-only, theta=quartering
}

POST_FAILURE_S = 200.0
N_T = 401


def _scenario_for_cell(tag: str) -> WcfdiScenario:
    """Build the WcfdiScenario with brucon polytope + calibrated lost-bus
    parameters when available, otherwise just the polytope (uncalibrated
    cells default tau_lost_pre_wcf=None, T_realloc_lost=None).
    """
    tau_lost = None
    T_realloc_lost: tuple[float, float, float] | None = None
    if _LOST_BUS_CAL_PATH.exists():
        data = json.loads(_LOST_BUS_CAL_PATH.read_text())
        row = data.get(tag)
        if row is not None:
            tau_lost = tuple(float(v) for v in row["tau_lost_pre_wcf_N_Nm"])
            tau_dof = row.get("T_realloc_lost_s")
            r2_dof = row.get("T_realloc_lost_R2", [1.0, 1.0, 1.0])
            if tau_dof is not None:
                T_realloc_lost = tuple(
                    float(tau_dof[i]) if r2_dof[i] >= 0.7 else 5.0
                    for i in range(3)
                )
    return WcfdiScenario(
        alpha=_ALPHA_FROM_POLYTOPE,
        tau_cap_intact=_INTACT_CAP_N_NM,
        gamma_immediate=0.5,
        T_realloc=10.0,
        tau_lost_pre_wcf=tau_lost,
        T_realloc_lost=T_realloc_lost,
    )


def _theta_rel_for_cell(tag: str) -> float:
    """Wave-relative direction (rad) as consumed by cqa.

    Mirrors the sign convention boundary diagnosed in sec.12.21.19:
    rel_deg below is the compass-CW bearing of the wave source from the
    bow (+ = source on starboard); cqa internals expect the opposite
    (+ = source on port), hence the negation.
    """
    theta_rel_deg = CELLS[tag][0]
    wave_compass = (VESSEL_HEADING_COMPASS + theta_rel_deg) % 360.0
    rel_deg = (wave_compass - VESSEL_HEADING_COMPASS + 540) % 360 - 180
    return float(np.radians(-rel_deg))


def run_one(cfg, rao, tag: str) -> dict:
    theta_rel_deg, env, _wind_off, _curr_abs, _curr_spd = CELLS[tag]
    theta_rel = _theta_rel_for_cell(tag)
    scenario = _scenario_for_cell(tag)
    # For pwq30 we run waves-only (Vw=0, Vc=0) to mirror the actual
    # ensemble, since that cell's calibration was generated that way.
    if tag == "pwq30":
        Vw_in, Vc_in = 0.0, 0.0
    else:
        Vw_in, Vc_in = float(env["Vw"]), float(env["Vc"])
    tr = wcfdi_transient(
        cfg=cfg,
        Vw_mean=Vw_in,
        Hs=float(env["Hs"]),
        Tp=float(env["Tp"]),
        Vc=Vc_in,
        theta_rel=theta_rel,
        scenario=scenario,
        t_end=POST_FAILURE_S,
        n_t=N_T,
        rao_table=rao,
    )
    info = tr.info
    return {
        "tag": tag,
        "bistability_per_dof": info["bistability_per_dof"],
        "bistability_risk_score": info["bistability_risk_score"],
        "tau_env": info["tau_env"],
        "tau_cap_post": info["tau_cap_post"],
        "cqa_precondition_violated": info["cqa_precondition_violated"],
    }


def empirical_regime_b(tag: str) -> dict:
    """Compute the per-cell empirical regime-B clip fraction (max over DOF
    of mean over seeds of fB_<DOF>) from the brucon ensemble."""
    rows = scan_cell(tag)
    if not rows:
        return {"n": 0, "fB_S_mean": np.nan, "fB_Y_mean": np.nan,
                "fB_Z_mean": np.nan, "fB_any_max": np.nan,
                "n_B_active": 0}
    fB_S = np.array([r["fB_S"] for r in rows])
    fB_Y = np.array([r["fB_Y"] for r in rows])
    fB_Z = np.array([r["fB_Z"] for r in rows])
    fB_any = np.maximum.reduce([fB_S, fB_Y, fB_Z])
    return {
        "n": len(rows),
        "fB_S_mean": float(fB_S.mean()),
        "fB_Y_mean": float(fB_Y.mean()),
        "fB_Z_mean": float(fB_Z.mean()),
        "fB_any_max": float(fB_any.max()),
        "n_B_active": int((fB_any > 0.05).sum()),
    }


def main() -> None:
    print(f"[polytope] intact_cap   (N,N,Nm) = "
          f"({_INTACT_CAP_N_NM[0]:.0f}, {_INTACT_CAP_N_NM[1]:.0f}, "
          f"{_INTACT_CAP_N_NM[2]:.0f})")
    print(f"[polytope] residual_cap (N,N,Nm) = "
          f"({_RESIDUAL_CAP_N_NM[0]:.0f}, {_RESIDUAL_CAP_N_NM[1]:.0f}, "
          f"{_RESIDUAL_CAP_N_NM[2]:.0f})")
    print(f"[polytope] alpha per DOF         = "
          f"({_ALPHA_FROM_POLYTOPE[0]:.3f}, {_ALPHA_FROM_POLYTOPE[1]:.3f}, "
          f"{_ALPHA_FROM_POLYTOPE[2]:.3f})")
    print()

    cfg, _ = setup_cqa()
    print(f"[rao] loading {PDSTRIP_PATH}")
    rao = load_pdstrip_rao(PDSTRIP_PATH)
    print()

    results = []
    for tag in CELLS:
        try:
            r = run_one(cfg, rao, tag)
        except Exception as e:
            print(f"[{tag}] FAILED: {e}")
            continue
        emp = empirical_regime_b(tag)
        r.update(emp)
        results.append(r)

    print("=" * 112)
    print("WCFDI-TRANSIENT bistability (predicted) vs empirical regime-B "
          "clip fraction (sec.12.21.21.16)")
    print("=" * 112)
    hdr = (f"{'cell':<14s} | {'bist_S':>7s} {'bist_Y':>7s} {'bist_Z':>7s} "
           f"{'bist_max':>9s} | "
           f"{'fB_S%':>6s} {'fB_Y%':>6s} {'fB_Z%':>6s} "
           f"{'n_B':>4s}/{'n':<3s} | {'cqa_viol':>9s}")
    print(hdr)
    print("-" * 112)
    for r in results:
        bd = r["bistability_per_dof"]
        viol = r["cqa_precondition_violated"]
        viol_str = "".join("X" if v else "." for v in viol)
        print(
            f"{r['tag']:<14s} | "
            f"{bd[0]:7.2f} {bd[1]:7.2f} {bd[2]:7.2f} "
            f"{r['bistability_risk_score']:9.2f} | "
            f"{r['fB_S_mean']*100:6.2f} {r['fB_Y_mean']*100:6.2f} "
            f"{r['fB_Z_mean']*100:6.2f} "
            f"{r['n_B_active']:>4d}/{r['n']:<3d} | "
            f"{viol_str:>9s}"
        )

    print()
    print("Interpretation:")
    print("  bist_X     = max over t in [0,200]s of severity_t in DOF X")
    print("               (0 = comfortable, >1 = saturation likely)")
    print("  bist_max   = headline bistability_risk_score = max over DOFs")
    print("  fB_X%      = mean over seeds of fraction of regime-B window")
    print("               with |Order|>|Alloc|+1%maxOrder in DOF X")
    print("  n_B / n    = seeds with any-DOF regime-B clipping >5%  /  total")
    print("  cqa_viol   = per-DOF |tau_env| > cap_post (S,Y,Z)")
    print()
    print("Expected: bist_max should be ~0 for the 9 safe cells, positive")
    print("for bf8_h0_w45 (mild) and clearly >1 for bf8_q10_w45 (severe).")


if __name__ == "__main__":
    main()
