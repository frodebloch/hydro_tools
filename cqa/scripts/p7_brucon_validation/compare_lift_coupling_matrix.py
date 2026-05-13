"""Per-cell A/B coverage comparison: pulse_response vs lift-coupled.

Across the full validation matrix, for each cell:
  - load all seeds' brucon trajectories
  - reconstruct per-seed tau_lost from (-b_hat snapshot - tau_thr)
  - run V3 (b_hat ensemble-mean tau_pre) under both:
        A: pulse_response               (no coupling)
        B: pulse_response_with_lift_coupling(K)
  - report per-axis bias for surge / sway and overall envelope coverage.

Mirrors the V3 logic in where_we_are_now_<tag>.py. Sigma halo is
recomputed from the cell's calibration npz + pre-WCF brucon residuals.

Run:
    PYTHONPATH=. .venv/bin/python \\
        scripts/p7_brucon_validation/compare_lift_coupling_matrix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))

from cqa.config import csov_default_config  # noqa: E402
from cqa.controller import LinearDpController  # noqa: E402
from cqa.transient_obs import (  # noqa: E402
    build_observer_augmented_system_full,
    csov_observer_gains,
    pulse_response,
    pulse_response_with_lift_coupling,
    IDX_ETA_HAT,
    N_STATE,
)
from cqa.vessel import LinearVesselModel  # noqa: E402

WORK_ROOT = THIS / "work"
T_WCF = 560.0
T_HORIZON = 60.0
DT = 0.05
T_PRE_LO = T_WCF - 30.0
T_PRE_HI = T_WCF - 5.0
B_HAT_SNAPSHOT_T = T_WCF - 5.0
SEEDS = list(range(1000, 1030))
K_SIGMA = 0.674

CELLS = [
    "pwq30", "pwo",
    "bf6_h0", "bf6_q10",
    "bf8_h0", "bf8_q10",
    "bf6_h0_w45", "bf6_q10_w45",
    "bf8_h0_w45", "bf8_q10_w45",
    "bf4_c1_h0", "bf4_c1_q10",
]


def _load_seed(tag, seed):
    seed_dir = WORK_ROOT / f"{tag}_seed{seed:04d}"
    main_p = next((p for p in seed_dir.glob("*.out") if "estimator" not in p.name), None)
    est_p = next(seed_dir.glob("*estimator*.out"), None)
    if main_p is None or est_p is None:
        return None
    with open(main_p) as f:
        hdr = f.readline().strip().split("\t")
    M = {h: c for h, c in zip(hdr, np.loadtxt(main_p, skiprows=1, delimiter="\t").T)}
    with open(est_p) as f:
        hdr_e = f.readline().strip().split("\t")
    E = {h: c for h, c in zip(hdr_e, np.loadtxt(est_p, skiprows=1, delimiter="\t").T)}
    if M["t"][-1] < T_WCF + T_HORIZON:
        return None
    return M, E


def _per_cell_stats(tag, aug, t_grid, K_lift):
    eta_lf = []
    tau_pre_bh = []
    tau_thr = []
    truth_peak = []
    truth_peak_x = []
    truth_peak_y = []
    pre_lf_dx = []; pre_lf_dy = []
    pre_wf_dx = []; pre_wf_dy = []

    for seed in SEEDS:
        out = _load_seed(tag, seed)
        if out is None:
            continue
        M, E = out
        post = (M["t"] >= T_WCF) & (M["t"] <= T_WCF + T_HORIZON + 1)
        t_post = M["t"][post]
        eta_lf.append([M["SurgeDev"][post][0], M["SwayDev"][post][0]])
        k_bh = int(np.argmin(np.abs(E["Time"] - B_HAT_SNAPSHOT_T)))
        tau_pre_bh.append([
            -float(E["EstBiasSurge"][k_bh]),
            -float(E["EstBiasSway"][k_bh]),
            -float(E["EstBiasYaw"][k_bh]),
        ])
        tau_thr.append(np.column_stack([
            np.interp(t_grid + T_WCF, t_post, M["Tx"][post]),
            np.interp(t_grid + T_WCF, t_post, M["Ty"][post]),
            np.interp(t_grid + T_WCF, t_post, M["Tz"][post]),
        ]))
        Rx = M["SurgeDev"][post]; Ry = M["SwayDev"][post]
        truth_peak.append(float(np.hypot(Rx, Ry).max()))
        # Per-axis truth: peak of |dev - pre-WCF mean|
        pre = (M["t"] >= T_PRE_LO) & (M["t"] <= T_PRE_HI)
        truth_peak_x.append(float(np.max(np.abs(Rx - M["SurgeDev"][pre].mean()))))
        truth_peak_y.append(float(np.max(np.abs(Ry - M["SwayDev"][pre].mean()))))
        # sigma proxies
        sx = M["SurgeDev"][pre]; sy = M["SwayDev"][pre]
        pre_lf_dx.append(sx - sx.mean()); pre_lf_dy.append(sy - sy.mean())
        pre_e = (E["Time"] >= T_PRE_LO) & (E["Time"] <= T_PRE_HI)
        wx = E["HfPosX"][pre_e]; wy = E["HfPosY"][pre_e]
        pre_wf_dx.append(wx - wx.mean()); pre_wf_dy.append(wy - wy.mean())

    n = len(eta_lf)
    if n == 0:
        return None
    eta_lf = np.asarray(eta_lf)
    tau_pre_bh = np.asarray(tau_pre_bh) * 1e3
    tau_thr = np.asarray(tau_thr) * 1e3
    truth_peak = np.asarray(truth_peak)
    truth_peak_x = np.asarray(truth_peak_x)
    truth_peak_y = np.asarray(truth_peak_y)
    tau_pre_ens = tau_pre_bh.mean(axis=0)
    b_hat0 = -tau_pre_ens   # ensemble env-force estimate

    def _peaks(use_coupling: bool):
        peaks = np.zeros(n)
        peaks_x = np.zeros(n)
        peaks_y = np.zeros(n)
        for i in range(n):
            # tau_lost = T_post - T_pre (authoritative; sec.12.21.17 sign fix)
            tau_lost = tau_thr[i] - tau_pre_ens[None, :]
            if use_coupling:
                X = pulse_response_with_lift_coupling(
                    aug, t_grid, tau_lost, b_hat0=b_hat0, K_lift=K_lift,
                    x0=np.zeros(N_STATE),
                )
            else:
                X = pulse_response(aug, t_grid, tau_lost, x0=np.zeros(N_STATE))
            de = X[:, IDX_ETA_HAT][:, 0:2]
            R = np.hypot(eta_lf[i, 0] + de[:, 0], eta_lf[i, 1] + de[:, 1])
            peaks[i] = R.max()
            # per-axis peak: max |delta_eta| (frame-aligned, no eta_lf IC)
            peaks_x[i] = float(np.max(np.abs(de[:, 0])))
            peaks_y[i] = float(np.max(np.abs(de[:, 1])))
        return peaks, peaks_x, peaks_y

    pk_A, px_A, py_A = _peaks(False)
    pk_B, px_B, py_B = _peaks(True)

    # halo
    calib = THIS / f"scenario_{tag}_calibration.npz"
    sig_bh = float(np.load(calib)["sigma_R_b_hat_m"]) if calib.exists() else 0.073
    sig_lf = float(np.hypot(np.std(np.concatenate(pre_lf_dx)),
                             np.std(np.concatenate(pre_lf_dy))))
    sig_wf = float(np.hypot(np.std(np.concatenate(pre_wf_dx)),
                             np.std(np.concatenate(pre_wf_dy))))
    halo = K_SIGMA * float(np.sqrt(sig_lf**2 + sig_wf**2 + sig_bh**2))

    env_A = pk_A + halo
    env_B = pk_B + halo
    cov_A = float((env_A >= truth_peak).mean())
    cov_B = float((env_B >= truth_peak).mean())

    def _gap(pred, truth):
        return float((pred.mean() - truth.mean()) / max(truth.mean(), 1e-3) * 100)

    return {
        "tag": tag, "n": n, "halo": halo,
        "truth": float(truth_peak.mean()),
        "truth_x": float(truth_peak_x.mean()),
        "truth_y": float(truth_peak_y.mean()),
        "A": float(pk_A.mean()), "B": float(pk_B.mean()),
        "A_x": float(px_A.mean()), "B_x": float(px_B.mean()),
        "A_y": float(py_A.mean()), "B_y": float(py_B.mean()),
        "gap_A": _gap(pk_A, truth_peak), "gap_B": _gap(pk_B, truth_peak),
        "gap_A_x": _gap(px_A, truth_peak_x), "gap_B_x": _gap(px_B, truth_peak_x),
        "gap_A_y": _gap(py_A, truth_peak_y), "gap_B_y": _gap(py_B, truth_peak_y),
        "cov_A": cov_A, "cov_B": cov_B,
        "env_A": float(env_A.mean()), "env_B": float(env_B.mean()),
    }


def main():
    K = json.loads((THIS / "lift_coupling_K.json").read_text())["K_per_rad"]
    print(f"K = {K:.3f} per rad\n")

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

    print(f"{'cell':<13} {'n':>3} | {'truth_x':>7} {'A_x':>6} {'B_x':>6} {'gA%':>5} {'gB%':>5} | "
          f"{'truth_y':>7} {'A_y':>6} {'B_y':>6} {'gA%':>5} {'gB%':>5} | "
          f"{'covA':>5} {'covB':>5}")
    print("-" * 110)

    rows = []
    for tag in CELLS:
        try:
            r = _per_cell_stats(tag, aug, t_grid, K)
        except FileNotFoundError as e:
            print(f"{tag:<13}  missing: {e}")
            continue
        if r is None:
            print(f"{tag:<13}  no seeds loaded")
            continue
        rows.append(r)
        print(f"{r['tag']:<13} {r['n']:>3} | "
              f"{r['truth_x']:>7.3f} {r['A_x']:>6.3f} {r['B_x']:>6.3f} "
              f"{r['gap_A_x']:>+4.0f}% {r['gap_B_x']:>+4.0f}% | "
              f"{r['truth_y']:>7.3f} {r['A_y']:>6.3f} {r['B_y']:>6.3f} "
              f"{r['gap_A_y']:>+4.0f}% {r['gap_B_y']:>+4.0f}% | "
              f"{r['cov_A']*100:>4.0f}% {r['cov_B']*100:>4.0f}%")

    if rows:
        print()
        cov_a = np.mean([r["cov_A"] for r in rows]) * 100
        cov_b = np.mean([r["cov_B"] for r in rows]) * 100
        gax = np.mean([abs(r["gap_A_x"]) for r in rows])
        gbx = np.mean([abs(r["gap_B_x"]) for r in rows])
        gay = np.mean([abs(r["gap_A_y"]) for r in rows])
        gby = np.mean([abs(r["gap_B_y"]) for r in rows])
        print(f"=== Matrix mean ===")
        print(f"  mean |gap_x|:  A {gax:.1f}%   B {gbx:.1f}%")
        print(f"  mean |gap_y|:  A {gay:.1f}%   B {gby:.1f}%")
        print(f"  mean coverage: A {cov_a:.0f}%   B {cov_b:.0f}%")


if __name__ == "__main__":
    main()
