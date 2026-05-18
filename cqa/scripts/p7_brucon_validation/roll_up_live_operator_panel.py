"""Roll up the live operator panel validation across all brucon cells.

Runs ``validate_live_operator_panel.py --tag <cell>`` for every cell
that has a calibration npz and brucon work-dirs, captures the key
quality metrics (intact P95 coverage, |err|/truth on intact P95, WCF
P95 coverage, WCF P95 bias vs realised LF peak), and prints a summary
table that lets us compare across sea states.

Run with::

    PYTHONPATH=. .venv/bin/python \\
        scripts/p7_brucon_validation/roll_up_live_operator_panel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))
sys.path.insert(0, str(THIS))

import live_cell_per_seed_pwq30 as live_cell                     # noqa: E402
from live_cell_per_seed_pwq30 import (                            # noqa: E402
    _load_tsv,
    load_seed,
    build_live_sigma_posterior,
)

from cqa.config import csov_default_config                       # noqa: E402
from cqa.live_decision import LiveObserverState                  # noqa: E402
from cqa.live_operator_view import summarise_for_operator_live   # noqa: E402
from cqa.live_regime_b import (                                  # noqa: E402
    DEFAULT_WINDOW_S as REGB_WIN_S,
    OperationalCapGeometry,
)
from cqa.transient import WcfdiScenario                          # noqa: E402

import json                                                       # noqa: E402

from saturation_screening import (                                # noqa: E402
    CSOV_THRUSTERS, CSOV_BUS_PORT_LOST, compute_residual_polytope,
)

_LOST_BUS_CAL_PATH = THIS / "wcfdi_lost_bus_calibration.json"


# Per sec.12.21.21.15 -- replace the heuristic alpha=(2/3,)*3 with the
# brucon-derived intact and residual thrust polytopes ported from
# BasicAllocator::CalculateAvailableThrust. This makes the cap_at_time(t)
# envelope used by sigma_tau_cmd and the bistability score match the
# actual feasibility region of the surviving thruster set.
_INTACT_POLYTOPE = compute_residual_polytope(CSOV_THRUSTERS)
_BUS_PORT_SURV = tuple(
    i for i in range(len(CSOV_THRUSTERS)) if i not in CSOV_BUS_PORT_LOST
)
_RESIDUAL_POLYTOPE = compute_residual_polytope(
    CSOV_THRUSTERS, surviving_indices=_BUS_PORT_SURV,
)
# Symmetrise: the cap_at_time machinery applies a single scalar per DOF
# that gets used as both upper and lower bound. Take min(|max|, |min|).
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
print(f"[polytope] intact_cap (N, N, Nm) = {_INTACT_CAP_N_NM}")
print(f"[polytope] residual_cap          = {_RESIDUAL_CAP_N_NM}")
print(f"[polytope] alpha (per DOF)       = {_ALPHA_FROM_POLYTOPE}")

# Per sec.12.21.21.24, the decoupled residual cap is unreachable in
# practice (yaw consumes part of the bow/stern sway capacity). The
# yaw-priority operational cap uses the bow+stern force decomposition
# to compute a tighter cap conditional on the LF mean of the other DOFs.
_REGB_GEOMETRY = OperationalCapGeometry(
    F_bow_max  = 1e3 * _RESIDUAL_POLYTOPE.max_sway_bow,
    F_bow_min  = 1e3 * _RESIDUAL_POLYTOPE.min_sway_bow,
    F_stern_max= 1e3 * _RESIDUAL_POLYTOPE.max_sway_stern,
    F_stern_min= 1e3 * _RESIDUAL_POLYTOPE.min_sway_stern,
    arm_bow    = _RESIDUAL_POLYTOPE.arm_bow,
    arm_stern  = _RESIDUAL_POLYTOPE.arm_stern,
)
_REGB_SURGE_CAP_N = _RESIDUAL_CAP_N_NM[0]
print(f"[regb-geom] bow ({_REGB_GEOMETRY.F_bow_min/1e3:+.0f}, "
      f"{_REGB_GEOMETRY.F_bow_max/1e3:+.0f}) kN @ arm {_REGB_GEOMETRY.arm_bow:+.2f} m")
print(f"[regb-geom] stern ({_REGB_GEOMETRY.F_stern_min/1e3:+.0f}, "
      f"{_REGB_GEOMETRY.F_stern_max/1e3:+.0f}) kN @ arm {_REGB_GEOMETRY.arm_stern:+.2f} m")


def _scenario_for_cell(tag: str) -> WcfdiScenario | None:
    """Return a WcfdiScenario seeded with brucon-calibrated tau_lost_pre_wcf
    and per-DOF T_realloc_lost when a JSON entry exists; otherwise None,
    which makes ``summarise_for_operator_live`` fall back to its default
    parametric placeholder. See sec.12.21.20.
    """
    if not _LOST_BUS_CAL_PATH.exists():
        return None
    data = json.loads(_LOST_BUS_CAL_PATH.read_text())
    row = data.get(tag)
    if row is None:
        return None
    tau_lost = tuple(float(v) for v in row["tau_lost_pre_wcf_N_Nm"])
    # R2 < 0.7 -> single-exponential is structurally wrong (e.g. yaw
    # closed-loop ringing); fall back to scalar T_realloc on those DOFs.
    tau_dof = row.get("T_realloc_lost_s")
    r2_dof = row.get("T_realloc_lost_R2", [1.0, 1.0, 1.0])
    if tau_dof is not None:
        T_realloc_lost = tuple(
            float(tau_dof[i]) if r2_dof[i] >= 0.7 else 5.0
            for i in range(3)
        )
    else:
        T_realloc_lost = None
    # Keep the scalar transient kinematics (alpha, gamma_imm, T_realloc)
    # at the live-cell defaults so the cap_at_time(t) reallocation
    # envelope is unchanged. Only the deficit pulse + IC re-init are
    # calibrated.
    # Use the brucon-derived polytope for both intact and residual cap
    # (sec.12.21.21.15). This replaces the heuristic alpha=(2/3,)*3 with
    # per-DOF residual/intact ratios derived from the surviving thruster
    # geometry (CSOV bus_port lost = Bow1 + PortMP).
    return WcfdiScenario(
        alpha=_ALPHA_FROM_POLYTOPE,
        tau_cap_intact=_INTACT_CAP_N_NM,
        gamma_immediate=0.5,
        T_realloc=10.0,
        tau_lost_pre_wcf=tau_lost,
        T_realloc_lost=T_realloc_lost,
    )


CELLS = [
    "bf4_c1_h0", "bf4_c1_q10",
    "bf6_h0",    "bf6_q10",    "bf6_h0_w45", "bf6_q10_w45",
    "bf8_h0",    "bf8_q10",    "bf8_h0_w45", "bf8_q10_w45",
    "bf7p5_q10_w45", "bf8p5_q10_w45",   # sec.12.21.21.30b half-BF probes
    "pwo",       "pwq30",
]
from _constants import T_WCF_S as T_WCF  # noqa: E402  # active script: refresh sec.12.21.16
SEED_LO, SEED_HI = 1000, 1030


def _set_cell(tag: str) -> None:
    live_cell.TAG = tag
    live_cell.T_WCF = T_WCF
    live_cell.T_EVAL = T_WCF - 5.0
    live_cell.WIN_END = T_WCF - 1.0
    live_cell.WIN_START = live_cell.WIN_END - live_cell.WIN_S
    live_cell.SEEDS = list(range(SEED_LO, SEED_HI))
    live_cell.CALIB_NPZ = THIS / f"scenario_{tag}_calibration.npz"


def _validate_cell(tag: str) -> dict | None:
    """Run validation for one cell. Returns a metrics dict, or None
    if the cell can't be loaded."""
    _set_cell(tag)
    calib_npz = live_cell.CALIB_NPZ
    if not calib_npz.exists():
        return None
    sigma_R_b_hat_m = float(np.load(calib_npz, allow_pickle=True)["sigma_R_b_hat_m"])

    cfg = csov_default_config()

    rows = []
    for seed in live_cell.SEEDS:
        d = load_seed(seed)
        if d is None:
            continue
        sigma_post = build_live_sigma_posterior(d, sigma_R_b_hat_m=sigma_R_b_hat_m)

        seed_dir = live_cell.WORK_ROOT / f"{tag}_seed{seed:04d}"
        main_p = next((p for p in seed_dir.glob("*.out")
                       if "estimator" not in p.name), None)
        M = _load_tsv(main_p)
        t_main = M["t"]
        win_m = (t_main >= live_cell.WIN_START) & (t_main <= live_cell.WIN_END)

        # Regime-B buffer: last REGB_WIN_S (=300s) of pre-WCF delivered
        # thrust Tx/Ty/Tz, ending at T_EVAL = T_WCF - 5. brucon stores
        # forces in kN, moments in kNm -> convert to N, Nm.
        regb_lo = live_cell.T_EVAL - REGB_WIN_S
        regb_hi = live_cell.T_EVAL
        regb_m = (t_main >= regb_lo) & (t_main <= regb_hi)
        tau_buffer = 1e3 * np.column_stack(
            [M["Tx"][regb_m], M["Ty"][regb_m], M["Tz"][regb_m]]
        )
        dt = float(np.median(np.diff(t_main[regb_m])))
        regb_fs_hz = 1.0 / dt if dt > 0 else 10.0

        obs = LiveObserverState(
            eta_hat=d["eta_hat"], nu_hat=d["nu_hat"], b_hat=d["b_hat"],
            eta_wave=d["eta_wave"],
            heading_compass=float(d.get("heading_compass", 0.0)),
            tau_buffer=tau_buffer,
            tau_buffer_fs_hz=regb_fs_hz,
        )
        scenario = _scenario_for_cell(tag)
        s = summarise_for_operator_live(
            cfg, obs, sigma_post, scenario=scenario,
            regime_b_geometry=_REGB_GEOMETRY,
            regime_b_surge_cap_N=_REGB_SURGE_CAP_N,
        )
        # Intact-axis truth: actual radial distance from setpoint (do NOT
        # demean). The operator panel reports |eta_hat_LF + nu|, i.e.
        # the actual offset+noise distance from the DP setpoint, so the
        # apples-to-apples truth is |SurgeDev, SwayDev| directly --
        # SurgeDev/SwayDev are already setpoint-relative LF deviations
        # in body frame.
        # WCF-axis truth still uses the demeaned LF radius because the
        # WCF transient is measured *relative to the pre-WCF mean
        # position* (the deterministic part of the WCF prediction is
        # eta_hat + delta_eta_mean(t), and we already include eta_hat
        # in the pred -- the realised peak is the post-WCF excursion
        # *from the same eta_hat reference*).
        sd_pre = float(M["SurgeDev"][win_m].mean())
        wd_pre = float(M["SwayDev"][win_m].mean())
        R_lf_intact_truth = np.hypot(M["SurgeDev"], M["SwayDev"])
        R_lf_wcf = np.hypot(M["SurgeDev"] - sd_pre, M["SwayDev"] - wd_pre)
        post_mask = ((t_main > live_cell.T_EVAL + 5.0)
                     & (t_main <= live_cell.T_EVAL + 120.0))
        intact_R = R_lf_intact_truth[win_m]
        post_R = R_lf_wcf[post_mask]

        rows.append(dict(
            intact_p95_pred=s.intact_R_p95,
            intact_p95_truth=float(np.quantile(intact_R, 0.95)),
            intact_p50_pred=s.intact_R_p50,
            intact_p50_truth=float(np.quantile(intact_R, 0.50)),
            wcf_p95_pred=s.wcf_R_p95,
            wcf_p50_pred=s.wcf_R_p50,
            wcf_peak_truth=float(np.max(post_R)),
            intact_traffic=s.intact_traffic,
            wcf_traffic=s.wcf_traffic,
            overall=s.overall_traffic,
            regb_severity=s.regime_b_severity,
            regb_traffic=s.regime_b_traffic,
            regb_p_sat_sway=(
                s.regime_b_p_sat[1] if s.regime_b_p_sat is not None else 0.0
            ),
            # Option-2 (sec.12.21.21.29) analytical post-WCF excursion
            # P95. Surfaced when Regime-B is present (same gating). 0
            # otherwise.
            wcf_excur_p95=(
                float(s.wcf_excur_R_xy_p95_m)
                if s.wcf_excur_present else 0.0
            ),
        ))
    if not rows:
        return None

    int_p95_pred = np.array([r["intact_p95_pred"] for r in rows])
    int_p95_tr = np.array([r["intact_p95_truth"] for r in rows])
    int_p50_pred = np.array([r["intact_p50_pred"] for r in rows])
    int_p50_tr = np.array([r["intact_p50_truth"] for r in rows])
    wcf_p95_pred = np.array([r["wcf_p95_pred"] for r in rows])
    wcf_p50_pred = np.array([r["wcf_p50_pred"] for r in rows])
    wcf_truth = np.array([r["wcf_peak_truth"] for r in rows])
    regb_sev = np.array([r["regb_severity"] for r in rows])
    regb_psat = np.array([r["regb_p_sat_sway"] for r in rows])
    excur_p95 = np.array([r["wcf_excur_p95"] for r in rows])

    # Apples-to-apples WCF truth: the panel pred is a *distribution*
    # (P50/P95 of the post-WCF radial peak under noise), so the truth
    # to compare against is the *distribution* of single-realisation
    # post-WCF peaks across the ensemble of seeds, NOT the mean of
    # those peaks. Mean(realised_peak) > P50(realised_peak) by ~10-20%
    # for a noisy LF channel; the panel pred is calibrated to predict
    # P50, not mean, of the post-WCF peak.
    wcf_truth_p50 = float(np.quantile(wcf_truth, 0.50))
    wcf_truth_p95 = float(np.quantile(wcf_truth, 0.95))
    wcf_p50_pred_mean = float(wcf_p50_pred.mean())
    wcf_p95_pred_mean = float(wcf_p95_pred.mean())

    return dict(
        tag=tag,
        n_seeds=len(rows),
        intact_p95_pred_mean=int_p95_pred.mean(),
        intact_p95_truth_mean=int_p95_tr.mean(),
        intact_p95_cov=float(np.mean(int_p95_tr <= int_p95_pred)),
        intact_p95_err_pct=100 * np.mean(np.abs(int_p95_pred - int_p95_tr) / int_p95_tr),
        intact_p95_bias_pct=100 * (int_p95_pred - int_p95_tr).mean() / int_p95_tr.mean(),
        intact_p50_bias=float((int_p50_pred - int_p50_tr).mean()),
        intact_p50_bias_pct=100 * (int_p50_pred - int_p50_tr).mean() / int_p50_tr.mean(),
        wcf_p95_pred_mean=wcf_p95_pred_mean,
        wcf_p50_pred_mean=wcf_p50_pred_mean,
        wcf_truth_mean=float(wcf_truth.mean()),
        wcf_truth_p50=wcf_truth_p50,
        wcf_truth_p95=wcf_truth_p95,
        # Pred-vs-truth bias on matched quantiles.
        wcf_p50_bias_pct=100 * (wcf_p50_pred_mean - wcf_truth_p50) / wcf_truth_p50,
        wcf_p95_bias_pct=100 * (wcf_p95_pred_mean - wcf_truth_p95) / wcf_truth_p95,
        # Coverage of the pred-P95 by the truth distribution: how
        # many seeds had their realised peak <= the predicted P95.
        # For a calibrated P95 this should be ~95%.
        wcf_p95_cov=float(np.mean(wcf_truth <= wcf_p95_pred)),
        # Composition of per-seed traffic-light verdicts (how often green/amber/red).
        n_red=sum(r["overall"] == "red" for r in rows),
        n_amber=sum(r["overall"] == "amber" for r in rows),
        n_green=sum(r["overall"] == "green" for r in rows),
        # Regime-B (sustained mean-thrust saturation risk on residual
        # polytope). See cqa.live_regime_b. severity = max_i p_sat,
        # IMCA thresholds green<0.01, amber [0.01,0.10), red>=0.10.
        regb_sev_mean=float(regb_sev.mean()),
        regb_sev_p90=float(np.quantile(regb_sev, 0.90)),
        regb_sev_max=float(regb_sev.max()),
        regb_psat_sway_max=float(regb_psat.max()),
        regb_n_green=sum(r["regb_traffic"] == "green" for r in rows),
        regb_n_amber=sum(r["regb_traffic"] == "amber" for r in rows),
        regb_n_red=sum(r["regb_traffic"] == "red" for r in rows),
        # Option-2 (sec.12.21.21.29) ensemble metrics.
        # NB: excur_p95 is an analytical *predicted* P95 of the post-WCF
        # |eta_xy| running max; the truth comparator is the same
        # wcf_truth_p95 (ensemble quantile of single-realisation peaks)
        # the legacy wcf path uses. Coverage = fraction of seeds with
        # realised peak <= predicted P95. The fraction of seeds where
        # the analytical estimator is even *active* (Regime-B present)
        # is regb_n_amber + regb_n_red; on green cells excur_p95=0 so
        # we just track the bias on the active subset to avoid
        # diluting with structurally-zero predictions.
        wcf_excur_p95_pred_mean_all=float(excur_p95.mean()),
        wcf_excur_p95_pred_mean_active=(
            float(excur_p95[excur_p95 > 0].mean()) if np.any(excur_p95 > 0) else 0.0
        ),
        wcf_excur_n_active=int(np.sum(excur_p95 > 0)),
        wcf_excur_p95_bias_pct=(
            100 * (excur_p95.mean() - wcf_truth_p95) / wcf_truth_p95
        ),
        wcf_excur_p95_cov=float(np.mean(wcf_truth <= excur_p95)),
    )


def main() -> int:
    results = []
    for tag in CELLS:
        m = _validate_cell(tag)
        if m is None:
            print(f"  {tag}: skipped (no data)")
            continue
        results.append(m)
        print(f"  {tag}: validated {m['n_seeds']} seeds")
    print()

    # ---- Table ----
    print("Live operator panel vs brucon LF truth -- 12-cell roll-up")
    print("(Intact: pred = P50/P95 of |eta_hat + nu_LF|; truth = quantiles of")
    print(" hypot(SurgeDev, SwayDev) over the 60-s pre-WCF window.)")
    print("(WCF:    pred = P50/P95 of |eta_hat + delta_eta_mean(t_peak) + nu|;")
    print(" truth = quantiles across seeds of single-realisation post-WCF max")
    print(" of vector-demeaned hypot(SurgeDev-pre, SwayDev-pre).)\n")
    hdr = (f"{'cell':<14} {'N':>3}   "
           f"{'iP95.pr':>7} {'iP95.tr':>7} {'P95bs%':>7} {'iP50bs%':>8} {'cov%':>5}   "
           f"{'wP50.pr':>7} {'wP50.tr':>7} {'P50bs%':>7} "
           f"{'wP95.pr':>7} {'wP95.tr':>7} {'P95bs%':>7} {'cov%':>5}   "
           f"{'g/a/r':>9}   "
           f"{'rgB.mn':>8} {'rgB.p90':>8} {'rgB.mx':>8} {'rgB g/a/r':>11}")
    print(hdr)
    print("-" * len(hdr))
    for m in results:
        print(f"{m['tag']:<14} {m['n_seeds']:>3}   "
              f"{m['intact_p95_pred_mean']:>7.3f} {m['intact_p95_truth_mean']:>7.3f} "
              f"{m['intact_p95_bias_pct']:>+6.0f}% {m['intact_p50_bias_pct']:>+7.0f}% "
              f"{100*m['intact_p95_cov']:>4.0f}%   "
              f"{m['wcf_p50_pred_mean']:>7.3f} {m['wcf_truth_p50']:>7.3f} "
              f"{m['wcf_p50_bias_pct']:>+6.0f}% "
              f"{m['wcf_p95_pred_mean']:>7.3f} {m['wcf_truth_p95']:>7.3f} "
              f"{m['wcf_p95_bias_pct']:>+6.0f}% {100*m['wcf_p95_cov']:>4.0f}%   "
              f"{m['n_green']}/{m['n_amber']}/{m['n_red']:<5}   "
              f"{m['regb_sev_mean']:>8.2e} {m['regb_sev_p90']:>8.2e} "
              f"{m['regb_sev_max']:>8.2e} "
              f"{m['regb_n_green']}/{m['regb_n_amber']}/{m['regb_n_red']:<7}")

    # ---- Option-2 (sec.12.21.21.29) head-to-head vs legacy wcf P95 ----
    print()
    print("Option-2 post-WCF excursion-distribution P95 vs brucon LF peak P95")
    print("(excur P95 is the analytical estimator from")
    print(" estimate_post_wcf_excursion_distribution; 0 when Regime-B inactive.)")
    print("(Comparator: same wcf_truth_p95 the legacy column uses.)\n")
    hdr2 = (f"{'cell':<14} {'N':>3}   "
            f"{'wTr.P95':>7} "
            f"{'lgcy.pr':>7} {'lgcy.bs%':>8} {'lgcy.cov':>8}   "
            f"{'opt2.pr':>7} {'opt2.bs%':>8} {'opt2.cov':>8}   "
            f"{'opt2.act':>8} {'opt2.act.pr':>11}")
    print(hdr2)
    print("-" * len(hdr2))
    for m in results:
        print(f"{m['tag']:<14} {m['n_seeds']:>3}   "
              f"{m['wcf_truth_p95']:>7.3f} "
              f"{m['wcf_p95_pred_mean']:>7.3f} "
              f"{m['wcf_p95_bias_pct']:>+7.0f}% {100*m['wcf_p95_cov']:>7.0f}%   "
              f"{m['wcf_excur_p95_pred_mean_all']:>7.3f} "
              f"{m['wcf_excur_p95_bias_pct']:>+7.0f}% {100*m['wcf_excur_p95_cov']:>7.0f}%   "
              f"{m['wcf_excur_n_active']:>4}/{m['n_seeds']:<3} "
              f"{m['wcf_excur_p95_pred_mean_active']:>11.3f}")

    # ---- Plot: intact, WCF, regime-B severity, Option-2 head-to-head ----
    fig, axes = plt.subplots(1, 4, figsize=(26, 5))
    tags = [m["tag"] for m in results]
    x = np.arange(len(tags))

    ax = axes[0]
    ax.bar(x - 0.20, [m["intact_p95_pred_mean"] for m in results],
           width=0.4, color="#1f77b4", alpha=0.8, label="pred P95")
    ax.bar(x + 0.20, [m["intact_p95_truth_mean"] for m in results],
           width=0.4, color="#444444", alpha=0.8, label="brucon P95")
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.axhline(2.0, color="#ff9900", ls="--", lw=1, label="IMCA amber 2 m")
    ax.axhline(4.0, color="#d62728", ls="--", lw=1, label="IMCA red 4 m")
    ax.set_ylabel("intact P95 of |R_LF| [m]")
    ax.set_title("INTACT axis  -  pred vs brucon LF P95 across cells")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - 0.20, [m["wcf_p95_pred_mean"] for m in results],
           width=0.4, color="#1f77b4", alpha=0.8, label="pred P95 (mean)")
    ax.bar(x + 0.20, [m["wcf_truth_p95"] for m in results],
           width=0.4, color="#444444", alpha=0.8, label="brucon LF peak P95")
    ax.plot(x, [m["wcf_truth_mean"] for m in results], "o",
            color="#888888", label="brucon LF peak mean")
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.axhline(2.0, color="#ff9900", ls="--", lw=1, label="IMCA amber 2 m")
    ax.axhline(4.0, color="#d62728", ls="--", lw=1, label="IMCA red 4 m")
    ax.set_ylabel("WCF post-fault peak [m]")
    ax.set_title("WCF axis  -  pred vs brucon LF peak across cells")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    sev_mean = np.array([m["regb_sev_mean"] for m in results])
    sev_p90 = np.array([m["regb_sev_p90"] for m in results])
    sev_max = np.array([m["regb_sev_max"] for m in results])
    # Floor for log axis: tiny but non-zero so log10 doesn't blow up.
    floor = 1e-10
    ax.bar(x - 0.25, np.maximum(sev_mean, floor), width=0.25,
           color="#1f77b4", alpha=0.8, label="ensemble mean")
    ax.bar(x + 0.00, np.maximum(sev_p90,  floor), width=0.25,
           color="#888888", alpha=0.8, label="ensemble P90")
    ax.bar(x + 0.25, np.maximum(sev_max,  floor), width=0.25,
           color="#444444", alpha=0.8, label="ensemble max")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.axhline(0.01, color="#ff9900", ls="--", lw=1, label="IMCA amber 0.01")
    ax.axhline(0.10, color="#d62728", ls="--", lw=1, label="IMCA red 0.10")
    ax.set_ylabel("regime-B severity = max p_sat (residual polytope)")
    ax.set_title("REGIME-B  -  sustained mean-thrust saturation risk")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", which="both", alpha=0.3)

    # Panel 4: Option-2 head-to-head with legacy and truth.
    ax = axes[3]
    legacy = np.array([m["wcf_p95_pred_mean"] for m in results])
    opt2 = np.array([m["wcf_excur_p95_pred_mean_all"] for m in results])
    truth = np.array([m["wcf_truth_p95"] for m in results])
    ax.bar(x - 0.25, legacy, width=0.25, color="#1f77b4", alpha=0.8,
           label="legacy pred P95")
    ax.bar(x + 0.00, opt2,   width=0.25, color="#9467bd", alpha=0.8,
           label="Option-2 pred P95")
    ax.bar(x + 0.25, truth,  width=0.25, color="#444444", alpha=0.8,
           label="brucon LF peak P95")
    # Mark cells where Option-2 was inactive across all seeds.
    inactive_mask = opt2 == 0.0
    for k in np.where(inactive_mask)[0]:
        ax.text(x[k] + 0.0, max(legacy[k], truth[k]) * 1.05, "inact",
                ha="center", fontsize=7, color="#9467bd", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.axhline(2.0, color="#ff9900", ls="--", lw=1, label="IMCA amber 2 m")
    ax.axhline(4.0, color="#d62728", ls="--", lw=1, label="IMCA red 4 m")
    ax.set_ylabel("post-WCF peak [m]")
    ax.set_title("Option-2 (sec.21.29) vs legacy WCF P95 vs brucon truth")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Live operator panel  -  12-cell brucon roll-up", fontsize=12)
    fig.tight_layout()
    out = THIS / "roll_up_live_operator_panel.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
