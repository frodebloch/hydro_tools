"""Calibrate WcfdiScenario.tau_lost_pre_wcf per cell from brucon truth.

Background
----------
The cqa scenario `tau_lost_fn` previously used the parametric proxy

    tau_lost(t) = (1 - beta(t)) * tau_env,
    beta(t)    = 1 + (gamma_imm - 1) * exp(-t / T_realloc)

with `tau_env = +b_hat`. This proxy under-represents the true lost-bus
contribution on cells where the failed bus carries internally-cancelled
forces. Concretely for CSOV `bus_port` (Bow1 tunnel + PortMP stern
azimuth) on `pwq30`: each thruster carries large yaw-trim moments
(sway from each combines with their lever arms to oppose) that cancel
at the global tau_env level. Brucon-measured per-DOF deficit is

    Delta(delivered) = T_post(plateau) - T_pre(SS)
                     ~  -50 kN surge,  -139 kN sway,  +3873 kN.m yaw

vs cqa parametric `(1 - 0.2) * (-1183 kN.m) = -237 kN.m yaw` -- wrong
sign for yaw because the failed bus's actual yaw contribution at SS is
opposite-signed to tau_env_yaw.

The fix per analysis.md sec.12.21.20: calibrate `tau_lost_pre_wcf` per
cell from brucon `(Tx, Ty, Tz)` ensembles and supply it on
`WcfdiScenario`. When set, `wcfdi_transient` uses

    tau_lost(t) = tau_lost_pre_wcf * exp(-t / T_realloc)

instead of the parametric placeholder. This is the analogue of the
calibrated path's `tau_lost_pre_wcf` field (sec.12.21.7) but plumbed
into the scenario path used by the validation launchers.

Usage
-----
.venv/bin/python scripts/p7_brucon_validation/calibrate_lost_bus.py

Outputs ``wcfdi_lost_bus_calibration.json`` in the same directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent.parent))

from harness import parse_output  # noqa: E402

WORK = THIS / "work"

# Per-cell calibration spec. Extend as new validation cells are added.
# t_wcf_s and the seed range come from the cell's lua / harness config.
CELLS = [
    {
        "tag": "pwq30",
        "bus": "bus_port",
        "seeds": list(range(1000, 1030)),
        "t_wcf_s": 1560.0,
        "t_pre_window": (-30.0, -5.0),     # SS averaging window pre-WCF (s rel)
        "t_post_window": (0.5, 2.0),       # plateau window post-WCF (s rel)
        "t_decay_window": (0.5, 15.0),     # window for per-DOF tau exponential fit
        "t_decay_grid_dt": 0.1,            # resampling step for the fit (s)
    },
]


def _fit_tau(t: np.ndarray, d: np.ndarray, default_tau: float) -> tuple[float, float]:
    """Fit `d(t) = A * exp(-t / tau)` and return (tau_s, R2).

    If the fit is degenerate (very small amplitude or curve_fit failure),
    returns (default_tau, 0.0). The amplitude itself is discarded -- the
    JSON's tau_lost_pre_wcf field is the snapshot-based magnitude, not
    the extrapolated A here.
    """
    if d.size < 5 or np.std(d) < 1.0:
        return default_tau, 0.0
    try:
        p0 = (d[0] if abs(d[0]) > 1.0 else np.sign(d.mean()) * max(abs(d.mean()), 1.0),
              max(default_tau, 1.0))
        popt, _ = curve_fit(lambda x, A, tau: A * np.exp(-x / tau), t, d,
                            p0=p0, maxfev=5000,
                            bounds=([-np.inf, 0.5], [np.inf, 60.0]))
        tau = float(popt[1])
        pred = popt[0] * np.exp(-t / popt[1])
        ss_res = float(np.sum((d - pred) ** 2))
        ss_tot = float(np.sum((d - d.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return tau, r2
    except Exception:
        return default_tau, 0.0


def calibrate_cell(cell: dict) -> dict:
    tag = cell["tag"]
    t_wcf = cell["t_wcf_s"]
    pre_lo, pre_hi = cell["t_pre_window"]
    post_lo, post_hi = cell["t_post_window"]
    pre_abs = (t_wcf + pre_lo, t_wcf + pre_hi)
    post_abs = (t_wcf + post_lo, t_wcf + post_hi)
    decay_lo, decay_hi = cell["t_decay_window"]
    decay_dt = cell["t_decay_grid_dt"]
    t_grid = np.arange(0.0, decay_hi + decay_dt, decay_dt)

    pre_v, post_v, drop_v = [], [], []
    # Per-seed deficit time series d_dof(t_grid) = T_pre_dof - T_dof(t_grid)
    # Stacked then ensemble-meaned for the per-DOF tau fit.
    deficits = {"surge": [], "sway": [], "yaw": []}
    for seed in cell["seeds"]:
        p = WORK / f"{tag}_seed{seed:04d}" / f"{tag}_seed{seed:04d}.out"
        if not p.exists():
            continue
        m = parse_output(p)
        t = m.columns["t"]
        pre_mask = (t >= pre_abs[0]) & (t <= pre_abs[1])
        post_mask = (t >= post_abs[0]) & (t <= post_abs[1])
        if pre_mask.sum() < 5 or post_mask.sum() < 5:
            continue
        # brucon Tx/Ty/Tz are in kN, kN, kNm; convert to N, N, Nm.
        tx_pre = float(m.columns["Tx"][pre_mask].mean()) * 1e3
        ty_pre = float(m.columns["Ty"][pre_mask].mean()) * 1e3
        tz_pre = float(m.columns["Tz"][pre_mask].mean()) * 1e3
        tx_post = float(m.columns["Tx"][post_mask].mean()) * 1e3
        ty_post = float(m.columns["Ty"][post_mask].mean()) * 1e3
        tz_post = float(m.columns["Tz"][post_mask].mean()) * 1e3
        pre_v.append(np.array([tx_pre, ty_pre, tz_pre]))
        post_v.append(np.array([tx_post, ty_post, tz_post]))
        # Sign convention: tau_lost_pre_wcf = T_pre - T_post (the
        # MAGNITUDE of lost positive thrust, body-frame). This matches
        # the calibrated path's `tau_thr_post_init_delta` convention
        # (calibrated_wcfdi_brucon_validation.py:290) so the same value
        # can drive both the open-loop tau_lost_fn pulse (with a sign
        # flip in wcfdi_transient -- the hull experiences the OPPOSITE
        # direction during the deficit) and the IC re-init.
        drop_v.append(np.array([tx_pre - tx_post,
                                ty_pre - ty_post,
                                tz_pre - tz_post]))
        # Per-seed deficit time series for the per-DOF tau fit.
        rel = t - t_wcf
        deficits["surge"].append(tx_pre - np.interp(t_grid, rel,
                                                    m.columns["Tx"]) * 1e3)
        deficits["sway"].append(ty_pre - np.interp(t_grid, rel,
                                                   m.columns["Ty"]) * 1e3)
        deficits["yaw"].append(tz_pre - np.interp(t_grid, rel,
                                                  m.columns["Tz"]) * 1e3)

    n = len(drop_v)
    drop = np.array(drop_v)
    drop_mean = drop.mean(axis=0)
    drop_std = drop.std(axis=0)

    # Per-DOF tau fit on ensemble-mean deficit curve restricted to
    # [decay_lo, decay_hi]. Default fallback is the scalar T_realloc=5 s
    # (same as the historical scenario default).
    fit_mask = (t_grid >= decay_lo) & (t_grid <= decay_hi)
    tau_fit = {}
    r2_fit = {}
    for k in ("surge", "sway", "yaw"):
        d_mean = np.mean(deficits[k], axis=0)
        tau, r2 = _fit_tau(t_grid[fit_mask], d_mean[fit_mask], default_tau=5.0)
        tau_fit[k] = tau
        r2_fit[k] = r2

    return {
        "n_seeds": n,
        "t_wcf_s": t_wcf,
        "t_pre_window_s": list(cell["t_pre_window"]),
        "t_post_window_s": list(cell["t_post_window"]),
        "t_decay_window_s": list(cell["t_decay_window"]),
        "tau_lost_pre_wcf_N_Nm": drop_mean.tolist(),
        "tau_lost_pre_wcf_std_N_Nm": drop_std.tolist(),
        "T_realloc_lost_s": [tau_fit["surge"], tau_fit["sway"],
                             tau_fit["yaw"]],
        "T_realloc_lost_R2": [r2_fit["surge"], r2_fit["sway"],
                              r2_fit["yaw"]],
        "interpretation": "tau_lost_pre_wcf = T_pre(SS) - T_post(plateau), "
                          "ensemble-mean across n_seeds. Hull-frame body "
                          "(surge, sway, yaw). POSITIVE sway means the "
                          "failed bus was carrying positive (starboard) "
                          "thrust at SS (typical when env pushes port). "
                          "Same sign convention as the calibrated path's "
                          "tau_thr_post_init_delta in calibrated_wcfdi.py. "
                          "T_realloc_lost is the per-DOF exponential decay "
                          "time constant of the deficit T_pre - T(t) fitted "
                          "on the ensemble-mean over t_decay_window. R2 < "
                          "0.7 indicates poor single-exponential fit "
                          "(e.g. closed-loop ringing in yaw); fall back to "
                          "scalar T_realloc in that case.",
        "bus": cell["bus"],
    }


def main() -> None:
    out = {}
    for cell in CELLS:
        cal = calibrate_cell(cell)
        out[cell["tag"]] = cal
        d = cal["tau_lost_pre_wcf_N_Nm"]
        s = cal["tau_lost_pre_wcf_std_N_Nm"]
        tau = cal["T_realloc_lost_s"]
        r2 = cal["T_realloc_lost_R2"]
        print(f"[{cell['tag']}] n={cal['n_seeds']}  bus={cell['bus']}")
        print(f"  tau_lost_pre_wcf [kN, kN, kN.m]  = "
              f"({d[0]/1e3:+8.1f}, {d[1]/1e3:+8.1f}, {d[2]/1e3:+9.1f})")
        print(f"  std                              = "
              f"({s[0]/1e3:8.1f}, {s[1]/1e3:8.1f}, {s[2]/1e3:9.1f})")
        print(f"  T_realloc_lost [s]               = "
              f"({tau[0]:8.2f}, {tau[1]:8.2f}, {tau[2]:9.2f})")
        print(f"  R2 of single-exp fit             = "
              f"({r2[0]:8.2f}, {r2[1]:8.2f}, {r2[2]:9.2f})")
    out_path = THIS / "wcfdi_lost_bus_calibration.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nSaved {out_path.name}")


if __name__ == "__main__":
    main()
