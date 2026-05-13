"""Single-seed proof-of-concept: lift-coupling correction in pulse_response.

Procedure (one cell at a time):
  1. Load one seed's brucon trajectory (e.g. bf8_h0_seed1000).
  2. Build its tau_lost(t) using the same logic as peak_R_b_hat_sigma_pwq30.py
     (tau_pre = -b_hat snapshot, tau_lost = T_post - T_pre = tau_thr - tau_pre).
     [Authoritative convention; sec.12.21.17 sign fix.]
  3. Compute the brucon-truth peak |R_x|, |R_y| for that seed.
  4. Run pulse_response (no coupling) -> peak_A
  5. Run pulse_response_with_lift_coupling(K) -> peak_B
  6. Report all three for surge and sway.

If the coupling implementation is correct, peak_B should be CLOSER to truth
than peak_A on the off-axis (sway for head-on cells, surge for beam-on).
Coupling correction at b_hat_x=0 (e.g. pwo) should be ~0 -> peak_B ~ peak_A.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
CQA_ROOT = THIS.parent.parent
sys.path.insert(0, str(CQA_ROOT))
sys.path.insert(0, str(THIS))

from cqa.config import csov_default_config  # noqa: E402
from cqa.vessel import LinearVesselModel  # noqa: E402
from cqa.controller import LinearDpController  # noqa: E402
from cqa.transient_obs import (  # noqa: E402
    build_observer_augmented_system_full,
    csov_observer_gains,
    pulse_response,
    pulse_response_with_lift_coupling,
    N_STATE,
    IDX_ETA,
)

WORK_DIR = THIS / "work"
T_WCF = 560.0
T_HORIZON = 60.0
DT = 0.05
B_HAT_SNAPSHOT_T = T_WCF - 5.0


def _load_seed(tag: str, seed: int):
    seed_dir = WORK_DIR / f"{tag}_seed{seed}"
    out_p = seed_dir / f"{tag}_seed{seed}.out"
    est_p = seed_dir / f"{tag}_seed{seed}_estimator.out"
    with open(out_p) as f:
        hdr_m = f.readline().strip().split("\t")
    M = {h: data for h, data in zip(hdr_m, np.loadtxt(out_p, skiprows=1, delimiter="\t").T)}
    with open(est_p) as f:
        hdr_e = f.readline().strip().split("\t")
    E = {h: data for h, data in zip(hdr_e, np.loadtxt(est_p, skiprows=1, delimiter="\t").T)}
    return M, E


def _per_seed_tau_lost(M, E, t_grid):
    k_bh = int(np.argmin(np.abs(E["Time"] - B_HAT_SNAPSHOT_T)))
    b_hat = np.array([
        float(E["EstBiasSurge"][k_bh]),
        float(E["EstBiasSway"][k_bh]),
        float(E["EstBiasYaw"][k_bh]),
    ]) * 1e3
    tau_pre = -b_hat
    t = M["t"]
    post = (t >= T_WCF) & (t <= T_WCF + T_HORIZON + 1.0)
    tau_thr_post = np.column_stack([
        M["Tx"][post], M["Ty"][post], M["Tz"][post]
    ]) * 1e3
    t_post = t[post]
    tau_thr_grid = np.column_stack([
        np.interp(t_grid + T_WCF, t_post, tau_thr_post[:, k]) for k in range(3)
    ])
    # tau_lost = T_post - T_pre (authoritative; sec.12.21.17 sign fix)
    tau_lost_grid = tau_thr_grid - tau_pre[None, :]
    return tau_lost_grid, tau_pre, b_hat


def _truth_peaks(M, t_grid):
    t = M["t"]
    pre = (t >= T_WCF - 60.0) & (t < T_WCF)
    post = (t >= T_WCF) & (t <= T_WCF + T_HORIZON)
    surge_pre = float(np.mean(M["SurgeDev"][pre]))
    sway_pre = float(np.mean(M["SwayDev"][pre]))
    Rx = M["SurgeDev"][post] - surge_pre
    Ry = M["SwayDev"][post] - sway_pre
    return float(np.max(np.abs(Rx))), float(np.max(np.abs(Ry)))


def main():
    K = json.loads((THIS / "lift_coupling_K.json").read_text())["K_per_rad"]
    print(f"Loaded K = {K:.3f} per rad")

    cfg = csov_default_config()
    vessel = LinearVesselModel.from_config(cfg.vessel)
    cp = cfg.controller
    ctrl = LinearDpController.from_bandwidth(
        vessel.M, vessel.D, omega_n=cp.omega_n, zeta=cp.zeta
    )
    obs = csov_observer_gains(Tp_s=10.0)
    aug = build_observer_augmented_system_full(
        vessel, ctrl, obs_gains=obs, T_thr=cp.thruster_time_constant_s
    )
    t_grid = np.arange(0.0, T_HORIZON + 1e-9, DT)

    test_cases = [
        ("bf6_h0", 1000),
        ("bf6_h0", 1001),
        ("bf8_h0", 1000),
        ("bf8_h0", 1001),
        ("bf8_q10", 1000),
        ("pwq30", 1000),
        ("pwo", 1000),
        ("bf4_c1_h0", 1000),
    ]

    print()
    print(f"{'cell':<12} {'seed':>5} {'b_hat[N]':>32}  "
          f"{'truth_x':>8} {'A_x':>7} {'B_x':>7} {'gA%':>6} {'gB%':>6} | "
          f"{'truth_y':>8} {'A_y':>7} {'B_y':>7} {'gA%':>6} {'gB%':>6}")
    print("-" * 142)

    for tag, seed in test_cases:
        try:
            M, E = _load_seed(tag, seed)
        except FileNotFoundError:
            print(f"{tag} seed {seed}: not found")
            continue
        tau_lost, tau_pre, b_hat = _per_seed_tau_lost(M, E, t_grid)
        truth_x, truth_y = _truth_peaks(M, t_grid)

        # A: no coupling
        X_A = pulse_response(aug, t_grid, tau_lost, x0=np.zeros(N_STATE))
        A_x = float(np.max(np.abs(X_A[:, IDX_ETA.start + 0])))
        A_y = float(np.max(np.abs(X_A[:, IDX_ETA.start + 1])))

        # B: with coupling. b_hat0 = -tau_pre = b_hat (the env force estimate).
        X_B = pulse_response_with_lift_coupling(
            aug, t_grid, tau_lost, b_hat0=b_hat, K_lift=K,
            x0=np.zeros(N_STATE),
        )
        B_x = float(np.max(np.abs(X_B[:, IDX_ETA.start + 0])))
        B_y = float(np.max(np.abs(X_B[:, IDX_ETA.start + 1])))

        gA_x = (A_x - truth_x) / max(truth_x, 1e-3) * 100
        gA_y = (A_y - truth_y) / max(truth_y, 1e-3) * 100
        gB_x = (B_x - truth_x) / max(truth_x, 1e-3) * 100
        gB_y = (B_y - truth_y) / max(truth_y, 1e-3) * 100

        bh_str = f"({b_hat[0]/1e3:>+5.0f},{b_hat[1]/1e3:>+5.0f},{b_hat[2]/1e3:>+6.0f})k"
        print(f"{tag:<12} {seed:>5} {bh_str:>32}  "
              f"{truth_x:>8.3f} {A_x:>7.3f} {B_x:>7.3f} {gA_x:>+5.0f}% {gB_x:>+5.0f}% | "
              f"{truth_y:>8.3f} {A_y:>7.3f} {B_y:>7.3f} {gA_y:>+5.0f}% {gB_y:>+5.0f}%")


if __name__ == "__main__":
    main()
