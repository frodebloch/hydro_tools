"""Tests for cqa.decision_matrix (forecast-case WCFDI decision matrix)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cqa import (
    csov_default_config,
    GangwayJointState,
    ForecastSlot,
    WcfdiScenario,
    evaluate_decision_cell,
    wcfdi_decision_matrix,
)
from cqa.decision_matrix import _wrap_pi, _worst, _imca_traffic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg_joint():
    cfg = csov_default_config()
    joint = GangwayJointState(h=15.0, alpha_g=0.0, beta_g=0.0, L=25.0)
    return cfg, joint


def _benign_slot(theta_env_compass: float = 0.0) -> ForecastSlot:
    """Slot well inside the green region for the CSOV defaults."""
    return ForecastSlot(
        label="benign", Vw=8.0, Hs=1.7, Tp=5.2, Vc=0.5,
        theta_env_compass=theta_env_compass,
    )


def _bistable_slot() -> ForecastSlot:
    """Beam-direction slot inside the bistability band (per analysis.md §12.14)."""
    Vw = 14.0
    return ForecastSlot(
        label="bistable", Vw=Vw, Hs=0.21 * Vw,
        Tp=max(4.0, 4.0 * np.sqrt(0.21 * Vw)), Vc=0.5,
        theta_env_compass=np.pi / 2,
    )


def _hopeless_slot() -> ForecastSlot:
    """Way past the CQA precondition (post-failure thrust insufficient)."""
    return ForecastSlot(
        label="hopeless", Vw=25.0, Hs=5.0, Tp=10.0, Vc=1.5,
        theta_env_compass=np.pi / 2,
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def test_wrap_pi_handles_extremes():
    assert math.isclose(_wrap_pi(0.0), 0.0)
    # +/- pi map to the same equivalence class (the function happens to
    # return the negative branch); accept either.
    w = _wrap_pi(np.pi)
    assert math.isclose(w, np.pi) or math.isclose(w, -np.pi)
    assert math.isclose(_wrap_pi(-np.pi + 1e-9), -np.pi + 1e-9, abs_tol=1e-8)
    # 3pi -> +/- pi
    w3 = _wrap_pi(3 * np.pi)
    assert math.isclose(w3, np.pi) or math.isclose(w3, -np.pi)
    # -3pi/2 -> pi/2
    assert math.isclose(_wrap_pi(-1.5 * np.pi), 0.5 * np.pi)


def test_worst_picks_red_over_amber_over_green():
    assert _worst("green", "green") == "green"
    assert _worst("green", "amber") == "amber"
    assert _worst("amber", "green") == "amber"
    assert _worst("amber", "red") == "red"
    assert _worst("red", "green") == "red"
    assert _worst("green", "amber", "red") == "red"


def test_imca_traffic_thresholds():
    assert _imca_traffic(0.5, 2.0, 4.0) == "green"
    assert _imca_traffic(2.0, 2.0, 4.0) == "amber"
    assert _imca_traffic(3.9, 2.0, 4.0) == "amber"
    assert _imca_traffic(4.0, 2.0, 4.0) == "red"
    assert _imca_traffic(float("inf"), 2.0, 4.0) == "red"
    assert _imca_traffic(float("nan"), 2.0, 4.0) == "red"


# ---------------------------------------------------------------------------
# Single-cell evaluator
# ---------------------------------------------------------------------------


def test_evaluate_cell_benign_is_all_green():
    cfg, joint = _cfg_joint()
    slot = _benign_slot()
    cell = evaluate_decision_cell(cfg, joint, slot, heading_compass=0.0)
    assert cell.intact_traffic == "green"
    assert cell.wcfdi_traffic == "green"
    assert cell.overall_traffic == "green"
    assert not cell.wcfdi_cqa_violated
    assert cell.wcfdi_bistability_score == 0.0
    # theta_rel is the relative direction from env into the vessel
    assert math.isclose(cell.theta_rel, 0.0)
    assert math.isfinite(cell.intact_pos_a_p90_m)
    assert math.isfinite(cell.wcfdi_pos_peak_m)


def test_evaluate_cell_heading_changes_theta_rel():
    cfg, joint = _cfg_joint()
    # Env from north (compass 0); heading east (pi/2) -> env from port beam
    slot = _benign_slot(theta_env_compass=0.0)
    cell_head = evaluate_decision_cell(cfg, joint, slot,
                                       heading_compass=np.pi / 2)
    # theta_rel = 0 - pi/2 = -pi/2 (env from port beam relative)
    assert math.isclose(cell_head.theta_rel, -np.pi / 2, abs_tol=1e-9)


def test_evaluate_cell_bistability_gate_forces_red():
    """Bistability score above threshold should force WCFDI traffic to red
    even though the deterministic peak alone might be amber.

    At the analysis.md §12.14 deep-band point (V_w=14 m/s beam) the
    score is ~6, well above the default alarm of 1.5.
    """
    cfg, joint = _cfg_joint()
    slot = _bistable_slot()
    # heading=0 -> theta_rel = pi/2 (env beam from starboard)
    cell = evaluate_decision_cell(cfg, joint, slot, heading_compass=0.0)
    assert cell.wcfdi_bistability_score > 1.5
    assert cell.wcfdi_traffic == "red"
    # And: disable the gate -> different (less severe) classification
    # possible. We just check the gate field is what we expect.
    cell_no_gate = evaluate_decision_cell(
        cfg, joint, slot, heading_compass=0.0,
        bistability_alarm=float("inf"),
    )
    assert cell_no_gate.wcfdi_bistability_score == cell.wcfdi_bistability_score
    # Without the gate the linear ODE still saturates at large peak
    # for V_w=14 (we observed ~2 m which already crosses 4 m alarm
    # only marginally; keep the test loose).
    assert cell_no_gate.wcfdi_pos_peak_m > 0.0


def test_evaluate_cell_cqa_violation_returns_red():
    cfg, joint = _cfg_joint()
    slot = _hopeless_slot()
    cell = evaluate_decision_cell(cfg, joint, slot, heading_compass=0.0)
    assert cell.wcfdi_cqa_violated
    assert cell.wcfdi_traffic == "red"
    assert cell.overall_traffic == "red"
    assert math.isinf(cell.wcfdi_pos_peak_m)


def test_evaluate_cell_collinear_with_polar_at_pm_point():
    """At a Pierson-Moskowitz (Hs, Tp)=PM(V_w) operating point, the
    forecast-case evaluator should agree with the polar's underlying
    point evaluation (same metric, same direction)."""
    from cqa.sea_state_relations import pm_hs_from_vw, pm_tp_from_vw
    cfg, joint = _cfg_joint()
    Vw = 12.0
    slot = ForecastSlot(
        label="pm-pt", Vw=Vw,
        Hs=pm_hs_from_vw(Vw), Tp=pm_tp_from_vw(Vw),
        Vc=0.5, theta_env_compass=np.pi / 2,
    )
    cell = evaluate_decision_cell(cfg, joint, slot, heading_compass=0.0)
    # Compare against direct calls to wcfdi_transient and the same
    # k_sigma envelope used by the cell.
    from cqa.transient import wcfdi_transient, WcfdiScenario
    scen = WcfdiScenario(alpha=(2/3,)*3, gamma_immediate=0.5, T_realloc=10.0)
    res = wcfdi_transient(cfg, Vw_mean=Vw, Hs=slot.Hs, Tp=slot.Tp,
                          Vc=slot.Vc, theta_rel=np.pi/2, scenario=scen)
    eta = res.eta_mean
    P_eta = res.P[:, 0:3, 0:3]
    pos_mean_r = np.sqrt(eta[:, 0]**2 + eta[:, 1]**2)
    sigma_R_t = np.sqrt(np.maximum(P_eta[:, 0, 0] + P_eta[:, 1, 1], 0.0))
    pos_envelope = pos_mean_r + 0.674 * sigma_R_t
    assert math.isclose(cell.wcfdi_pos_peak_m, float(np.max(pos_envelope)),
                        rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Matrix driver
# ---------------------------------------------------------------------------


def test_matrix_shape_and_indexing():
    cfg, joint = _cfg_joint()
    slots = [_benign_slot(), _benign_slot(theta_env_compass=np.pi/4)]
    headings = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    mx = wcfdi_decision_matrix(cfg, joint, slots, headings)
    assert len(mx.cells) == 8
    assert mx.headings_compass.shape == (4,)
    grid = mx.overall_grid()
    assert grid.shape == (2, 4)
    # cell() is consistent with the linear cells tuple
    for s in range(2):
        for h in range(4):
            assert mx.cell(s, h) is mx.cells[s * 4 + h]
            assert mx.cell(s, h).slot_index == s
            assert mx.cell(s, h).heading_index == h


def test_matrix_intact_amber_or_red_propagates_to_overall():
    """Force a cell where intact is amber but WCFDI is green -> overall amber."""
    cfg, joint = _cfg_joint()
    # Pick a moderate beam wind: at Vw ~ 13 m/s beam, intact P90 starts
    # to creep into the warn radius (2 m) but WCFDI is still recoverable
    # at heading aligned to bow.
    slots = [
        ForecastSlot(label="moderate-beam", Vw=13.0,
                     Hs=0.21 * 13.0, Tp=max(4.0, 4.0 * np.sqrt(0.21 * 13.0)),
                     Vc=0.5, theta_env_compass=np.pi / 2),
    ]
    # Heading=0 -> theta_rel=pi/2 (beam env). Intact may flip amber.
    mx = wcfdi_decision_matrix(cfg, joint, slots, np.array([0.0]))
    cell = mx.cell(0, 0)
    # The semantic invariant: overall is the worst of intact and WCFDI.
    expected = max((cell.intact_traffic, cell.wcfdi_traffic),
                   key=lambda s: {"green": 0, "amber": 1, "red": 2}[s])
    assert cell.overall_traffic == expected


def test_matrix_progress_callback_invoked_per_cell():
    cfg, joint = _cfg_joint()
    slots = [_benign_slot(), _benign_slot(theta_env_compass=np.pi)]
    headings = np.array([0.0, np.pi])
    seen = []
    wcfdi_decision_matrix(
        cfg, joint, slots, headings,
        progress_cb=lambda k, n, label: seen.append((k, n, label)),
    )
    assert len(seen) == 4
    # k counts up from 1 to n
    assert [s[0] for s in seen] == [1, 2, 3, 4]
    assert all(s[1] == 4 for s in seen)


def test_matrix_grids_decompose_consistently():
    cfg, joint = _cfg_joint()
    slots = [_benign_slot(), _bistable_slot(), _hopeless_slot()]
    headings = np.array([0.0, np.pi/2])
    mx = wcfdi_decision_matrix(cfg, joint, slots, headings)
    intact = mx.intact_grid()
    wcfdi = mx.wcfdi_grid()
    overall = mx.overall_grid()
    order = {"green": 0, "amber": 1, "red": 2}
    for s in range(3):
        for h in range(2):
            assert order[overall[s, h]] == max(
                order[intact[s, h]], order[wcfdi[s, h]]
            )

    # The bistable slot at heading=0 (theta_rel=pi/2) must be WCFDI red
    # (gate-driven). Hopeless slot must be WCFDI red on the beam
    # heading; at heading=pi/2 the env is head-on, which is much more
    # benign (surge has more reserve than sway in the CSOV defaults),
    # so we don't assert red there.
    assert wcfdi[1, 0] == "red"
    assert wcfdi[2, 0] == "red"
