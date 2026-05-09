"""Per-seed spread of brucon truth peaks vs V3 predictor.

For each cell, report:
  - mean and std of brucon truth peak |R_y| (sway)
  - mean and std of V3 (b_hat ensemble) prediction peak |R_y|
  - coefficient of variation (std / mean) for both
  - same for surge

Question: how much of the 36% under-prediction on bf6_h0 sway is
deterministic core defect vs irreducible per-realisation scatter?
If the brucon spread is comparable to the bias, the defect is
already inside the wave-realisation noise envelope.
"""
from __future__ import annotations

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
    pulse_response_with_lift_coupling,
    IDX_ETA_HAT,
    N_STATE,
)
from cqa.vessel import LinearVesselModel  # noqa: E402
import json

WORK_ROOT = THIS / "work"
T_WCF = 560.0
T_HORIZON = 60.0
DT = 0.05
T_PRE_LO = T_WCF - 30.0
T_PRE_HI = T_WCF - 5.0
B_HAT_SNAPSHOT_T = T_WCF - 5.0
SEEDS = list(range(1000, 1030))

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


def _per_cell(tag, aug, t_grid, K_lift):
    eta_lf = []
    tau_pre_bh = []
    tau_thr = []
    truth_x = []; truth_y = []
    for seed in SEEDS:
        out = _load_seed(tag, seed)
        if out is None:
            continue
        M, E = out
        post = (M["t"] >= T_WCF) & (M["t"] <= T_WCF + T_HORIZON + 1)
        pre = (M["t"] >= T_PRE_LO) & (M["t"] <= T_PRE_HI)
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
        truth_x.append(float(np.max(np.abs(Rx - M["SurgeDev"][pre].mean()))))
        truth_y.append(float(np.max(np.abs(Ry - M["SwayDev"][pre].mean()))))

    n = len(eta_lf)
    if n == 0:
        return None
    eta_lf = np.asarray(eta_lf)
    tau_pre_bh = np.asarray(tau_pre_bh) * 1e3
    tau_thr = np.asarray(tau_thr) * 1e3
    truth_x = np.asarray(truth_x); truth_y = np.asarray(truth_y)
    tau_pre_ens = tau_pre_bh.mean(axis=0)
    b_hat0 = -tau_pre_ens

    pred_x = np.zeros(n); pred_y = np.zeros(n)
    for i in range(n):
        tau_lost = tau_pre_ens[None, :] - tau_thr[i]
        X = pulse_response_with_lift_coupling(
            aug, t_grid, tau_lost, b_hat0=b_hat0, K_lift=K_lift,
            x0=np.zeros(N_STATE),
        )
        de = X[:, IDX_ETA_HAT][:, 0:2]
        pred_x[i] = float(np.max(np.abs(de[:, 0])))
        pred_y[i] = float(np.max(np.abs(de[:, 1])))

    return {
        "tag": tag, "n": n,
        "tx_mean": truth_x.mean(), "tx_std": truth_x.std(),
        "ty_mean": truth_y.mean(), "ty_std": truth_y.std(),
        "px_mean": pred_x.mean(), "px_std": pred_x.std(),
        "py_mean": pred_y.mean(), "py_std": pred_y.std(),
        "truth_x": truth_x, "truth_y": truth_y,
        "pred_x": pred_x, "pred_y": pred_y,
    }


def main():
    K = json.loads((THIS / "lift_coupling_K.json").read_text())["K_per_rad"]
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

    print("Per-seed spread analysis (mean +/- std, CoV = std/mean)")
    print("Truth = brucon, Pred = V3 b_hat-ensemble + lift coupling\n")
    print(f"{'cell':<13}     SURGE truth    pred    CoV_t  CoV_p  | "
          f"   SWAY  truth    pred    CoV_t  CoV_p  | bias_y  bias/sigma_t")
    print("-" * 120)

    rows = []
    for tag in CELLS:
        try:
            r = _per_cell(tag, aug, t_grid, K)
        except FileNotFoundError:
            continue
        if r is None:
            continue
        rows.append(r)
        cov_tx = r["tx_std"] / max(r["tx_mean"], 1e-3)
        cov_ty = r["ty_std"] / max(r["ty_mean"], 1e-3)
        cov_px = r["px_std"] / max(r["px_mean"], 1e-3)
        cov_py = r["py_std"] / max(r["py_mean"], 1e-3)
        bias_y = r["py_mean"] - r["ty_mean"]
        # how many truth-stds is the bias?
        bias_y_norm = bias_y / max(r["ty_std"], 1e-3)
        print(f"{r['tag']:<13}  "
              f"{r['tx_mean']:5.2f}+/-{r['tx_std']:4.2f}  "
              f"{r['px_mean']:5.2f}+/-{r['px_std']:4.2f}  "
              f"{cov_tx*100:4.0f}%  {cov_px*100:4.0f}%  | "
              f"{r['ty_mean']:5.2f}+/-{r['ty_std']:4.2f}  "
              f"{r['py_mean']:5.2f}+/-{r['py_std']:4.2f}  "
              f"{cov_ty*100:4.0f}%  {cov_py*100:4.0f}%  | "
              f"{bias_y:+5.2f}m  {bias_y_norm:+4.1f} sigma")

    print()
    print("Interpretation:")
    print("  CoV_t (truth) shows seed-to-seed scatter from wave realisation alone.")
    print("  bias/sigma_t < 1 means the deterministic gap is inside one truth-std.")


if __name__ == "__main__":
    main()
