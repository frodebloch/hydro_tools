"""Survey post-WCF yaw tau_lost sign across all 12 validation cells.

Tests the hypothesis (user 2026-05) that the cqa WcfdiScenario's symmetric
per-DOF cap reduction (alpha_yaw) cannot represent the bf8-oblique yaw
loss because the actual hull-experienced yaw tau_lost sign depends on
which way the surviving stern azimuth (StbdMP) reorients to take up the
load lost when PortMP is killed:

  - Low aft side-force demand pre-WCF: the two stern azimuths are biased
    inwards (anti-bias for redundancy). Killing PortMP forces StbdMP to
    swing ~180 deg to take over forward thrust; during that swing the
    realised yaw torque overshoots Order with the opposite sign of the
    env yaw moment.
  - High aft side-force demand pre-WCF: both stern azimuths point the
    same way (toward the aft side-force direction). Killing PortMP just
    removes one stbd-pushing stern thruster; sway tau_lost and yaw
    tau_lost are both in the natural (cqa-predicted) direction.

The survey just looks at the ensemble-mean of (Tz - OrderTauYaw) in the
first 5 s post-WCF and compares its sign to b_hat_yaw at t_eval. If
sign(tau_lost_yaw) != sign(-b_hat_yaw) for some cells, the stern-azimuth
bias mechanism is confirmed (cqa scenario formula tau_lost = (1-beta)*b
predicts sign(tau_lost) = sign(b), but brucon can show the opposite when
StbdMP reorients).

Usage:
  .venv/bin/python scripts/p7_brucon_validation/survey_tau_lost_yaw_sign.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from harness import parse_output  # noqa: E402

ENSEMBLE_DIR = THIS / "work"
SEEDS = list(range(1000, 1030))
T_WCF = 560.0
T_EVAL = T_WCF - 5.0
T_PEAK_WINDOW = (0.0, 5.0)  # seconds post-WCF
DT = 0.1

CELLS = [
    "bf4_c1_h0", "bf4_c1_q10",
    "bf6_h0", "bf6_q10", "bf6_h0_w45", "bf6_q10_w45",
    "bf8_h0", "bf8_q10", "bf8_h0_w45", "bf8_q10_w45",
    "pwo", "pwq30",
]


def load_seed(tag: str, seed: int):
    seed_dir = ENSEMBLE_DIR / f"{tag}_seed{seed:04d}"
    out_path = seed_dir / f"{tag}_seed{seed:04d}.out"
    est_path = seed_dir / f"{tag}_seed{seed:04d}_estimator.out"
    if not out_path.exists() or not est_path.exists():
        return None
    main = parse_output(out_path)
    est = parse_output(est_path)
    t = main.columns["t"]
    if t[-1] < T_WCF + T_PEAK_WINDOW[1] + 5.0:
        return None

    # Triplet of tau_lost definitions at post-WCF interpolated grid.
    t_rel = t - T_WCF
    t_grid = np.arange(T_PEAK_WINDOW[0], T_PEAK_WINDOW[1] + DT / 2, DT)
    yaw_T  = np.interp(t_grid, t_rel,
                       main.columns["Tz"] - main.columns["OrderTauYaw"])
    yaw_Fb = np.interp(t_grid, t_rel,
                       main.columns["FbTauYaw"] - main.columns["OrderTauYaw"])
    sway_T = np.interp(t_grid, t_rel,
                       main.columns["Ty"] - main.columns["OrderTauSway"])
    surge_T = np.interp(t_grid, t_rel,
                        main.columns["Tx"] - main.columns["OrderTauSurge"])

    # b_hat at t_eval (estimator: EstBiasYaw in kNm, EstBiasSurge/Sway in kN).
    t_est = est.columns["Time"]
    i_eval = int(np.argmin(np.abs(t_est - T_EVAL)))
    b_hat = np.array([
        est.columns["EstBiasSurge"][i_eval],
        est.columns["EstBiasSway"][i_eval],
        est.columns["EstBiasYaw"][i_eval],
    ])
    return dict(t_grid=t_grid, surge=surge_T, sway=sway_T,
                yaw=yaw_T, yaw_fb=yaw_Fb, b_hat=b_hat)


def survey():
    print()
    print(f"{'cell':<14} | {'n':>3} | "
          f"{'b_hat_yaw [kNm]':>16} | "
          f"{'tau_lost_yaw_peak [kNm]':>26} | "
          f"{'sign(cqa)':>10} | {'sign(brucon)':>13} | match")
    print("-" * 110)
    for tag in CELLS:
        data = []
        for s in SEEDS:
            d = load_seed(tag, s)
            if d is not None:
                data.append(d)
        if not data:
            print(f"{tag:<14} | no data")
            continue
        n = len(data)

        b_yaw_mean = np.mean([d["b_hat"][2] for d in data])
        yaw_arr = np.array([d["yaw"] for d in data])      # (n, Nt)
        yaw_ens = yaw_arr.mean(axis=0)                     # (Nt,)
        # Peak signed value (largest magnitude post-WCF in window)
        idx_peak = int(np.argmax(np.abs(yaw_ens)))
        peak = yaw_ens[idx_peak]
        t_peak = data[0]["t_grid"][idx_peak]

        # cqa scenario formula: tau_lost = (1 - beta) * b_hat,
        # at t=0+ with gamma_imm=0.5 this is +0.5 * b_hat.
        # So sign(cqa scenario tau_lost) = sign(b_hat).
        sign_cqa = "+" if b_yaw_mean > 0 else "-"
        sign_brucon = "+" if peak > 0 else "-"
        match = "OK" if sign_cqa == sign_brucon else "**FLIP**"

        print(f"{tag:<14} | {n:>3} | "
              f"{b_yaw_mean:>16.1f} | "
              f"{peak:>+18.1f} (t={t_peak:4.1f}s) | "
              f"{sign_cqa:>10} | {sign_brucon:>13} | {match}")

    # Also report sway tau_lost peak sign comparison.
    print()
    print(f"{'cell':<14} | {'n':>3} | "
          f"{'b_hat_sway [kN]':>16} | "
          f"{'tau_lost_sway_peak [kN]':>26} | "
          f"{'sign(cqa)':>10} | {'sign(brucon)':>13} | match")
    print("-" * 110)
    for tag in CELLS:
        data = []
        for s in SEEDS:
            d = load_seed(tag, s)
            if d is not None:
                data.append(d)
        if not data:
            continue
        n = len(data)
        b_sway_mean = np.mean([d["b_hat"][1] for d in data])
        sway_arr = np.array([d["sway"] for d in data])
        sway_ens = sway_arr.mean(axis=0)
        idx_peak = int(np.argmax(np.abs(sway_ens)))
        peak = sway_ens[idx_peak]
        t_peak = data[0]["t_grid"][idx_peak]
        sign_cqa = "+" if b_sway_mean > 0 else "-"
        sign_brucon = "+" if peak > 0 else "-"
        match = "OK" if sign_cqa == sign_brucon else "**FLIP**"
        print(f"{tag:<14} | {n:>3} | "
              f"{b_sway_mean:>16.1f} | "
              f"{peak:>+18.1f} (t={t_peak:4.1f}s) | "
              f"{sign_cqa:>10} | {sign_brucon:>13} | {match}")


if __name__ == "__main__":
    survey()
