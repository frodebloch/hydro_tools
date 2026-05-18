"""Controller demand audit: compare cqa implicit tau_cmd vs brucon OrderTau.

Question (sec.12.21.21.28b, follow-up to sec.12.21.21.28 K_b_pos fix):
  cqa b_hat(t) now matches brucon EstBias(t) well (RMS 22.7 vs 19.0 kN,
  lag 0, xcorr 0.99 on calm seed 1001). But Test E sway prediction is
  still 1.55 (calm) and 0.68 (outlier) vs brucon truth. Where is the
  remaining gap?

This script:
  1) integrates Test E (linear closed loop + dtau + env force);
  2) extracts cqa's deviation tau_cmd via implicit_tau_cmd(aug, x_dev);
  3) loads brucon's OrderTau(t) and demeans by [T_WCF-30, T_WCF-5] mean;
  4) plots both per axis (surge / sway / yaw) on the same panel.

Interpretation:
  - If cqa tau_cmd_dev matches brucon OrderTau_dev closely, the
    controller's linear demand response is correct -> any sway gap
    is downstream (plant lag, saturation feedback, lift coupling).
  - If cqa tau_cmd_dev differs (e.g. over-shoots), the controller's
    closed-loop demand is amplified, likely due to integrator
    anti-windup behaviour brucon implements but cqa does not, or
    a different Kp/Kd convention than we have audited.

Run:
  PYTHONPATH=. .venv/bin/python scripts/p7_brucon_validation/tau_cmd_audit_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
ROOT = THIS.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqa.live_decision import _build_aug_for_live  # noqa: E402
from cqa.config import csov_default_config  # noqa: E402
from cqa.transient_obs import (  # noqa: E402
    IDX_ETA, IDX_B_HAT, implicit_tau_cmd,
)

from scripts.p7_brucon_validation.linearity_reconstruction_diagnostic import (  # noqa: E402
    load_seed,
    extract_delta_tau,
    extract_brucon_sway,
    extract_env_force_perturbation,
    reconstruct_with_env,
    T_WCF,
    T_HORIZON,
    DT_INT,
    COL_T,
    COL_ORDER_SURGE, COL_ORDER_SWAY, COL_ORDER_YAW,
)

SEEDS = {
    "outlier_1012": 1012,
    "calm_1001": 1001,
}
DOF_LABELS = ["surge", "sway", "yaw"]


def extract_brucon_order_tau(data: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """OrderTau(t) on grid, demeaned by [T_WCF-30, T_WCF-5] mean.

    Returns (N, 3) in N, N, N*m (converted from kN, kN*m). This is
    brucon's pre-saturation controller demand.
    """
    t_b = data[:, COL_T]
    pre = (t_b >= T_WCF - 30.0) & (t_b <= T_WCF - 5.0)
    cols = [COL_ORDER_SURGE, COL_ORDER_SWAY, COL_ORDER_YAW]
    t_abs = T_WCF + t_grid
    out = np.zeros((len(t_grid), 3))
    for i, c in enumerate(cols):
        sig = data[:, c]
        sig_pre = float(sig[pre].mean())
        out[:, i] = (np.interp(t_abs, t_b, sig) - sig_pre) * 1e3  # kN -> N
    return out


def run() -> None:
    cfg = csov_default_config()
    aug = _build_aug_for_live(cfg, Tp_obs_s=10.0)

    t_grid = np.arange(0.0, T_HORIZON + DT_INT * 0.5, DT_INT)
    N = len(t_grid)

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)

    for col, (label, seed) in enumerate(SEEDS.items()):
        data = load_seed(seed)
        dtau = extract_delta_tau(data, t_grid)
        df_env = extract_env_force_perturbation(data, t_grid)
        brucon_order_tau = extract_brucon_order_tau(data, t_grid)
        brucon_sway = extract_brucon_sway(data, t_grid)

        # cqa Test E linear closed loop
        X_E = reconstruct_with_env(aug, t_grid, dtau, df_env)
        cqa_tau_cmd = np.array([implicit_tau_cmd(aug, X_E[k]) for k in range(N)])
        sway_E = X_E[:, IDX_ETA][:, 1]

        def p95(x):
            return float(np.percentile(np.abs(x), 95))

        print(f"\n=== {label} (seed {seed}) ===")
        print(f"  brucon sway P95: {p95(brucon_sway):.2f} m  / Test E: {p95(sway_E):.2f} m  ratio {p95(sway_E)/p95(brucon_sway):.2f}")
        for i, dof in enumerate(DOF_LABELS):
            unit = "kN" if i < 2 else "kN.m"
            print(f"  {dof:5s}: brucon OrderTau_dev RMS / peak: "
                  f"{np.std(brucon_order_tau[:, i])/1e3:7.1f} / {np.abs(brucon_order_tau[:, i]).max()/1e3:7.1f} {unit}"
                  f"   |   cqa tau_cmd_dev RMS / peak: "
                  f"{np.std(cqa_tau_cmd[:, i])/1e3:7.1f} / {np.abs(cqa_tau_cmd[:, i]).max()/1e3:7.1f} {unit}")

            # Cross-correlation
            x = cqa_tau_cmd[:, i] - cqa_tau_cmd[:, i].mean()
            y = brucon_order_tau[:, i] - brucon_order_tau[:, i].mean()
            if np.std(x) > 1e-6 and np.std(y) > 1e-6:
                xc = np.correlate(x, y, mode="full") / (np.std(x) * np.std(y) * len(x))
                lags = np.arange(-len(x) + 1, len(x)) * DT_INT
                ilag = np.argmax(xc)
                print(f"           xcorr peak {xc[ilag]:+.3f} at lag {lags[ilag]:+.1f} s")

            ax = axes[i, col]
            ax.plot(t_grid, brucon_order_tau[:, i] / 1e3, color="C0", lw=1.6,
                    label="brucon OrderTau_dev")
            ax.plot(t_grid, cqa_tau_cmd[:, i] / 1e3, color="C3", lw=1.2, ls="--",
                    label="cqa implicit tau_cmd_dev")
            ax.axhline(0, color="black", lw=0.4)
            ax.set_ylabel(f"{dof} tau_cmd_dev [{unit}]")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"{label}")
                ax.legend(fontsize=8)
            if i == 2:
                ax.set_xlabel("t - T_WCF [s]")

    fig.suptitle(
        "bf8_q10_w45 controller-demand audit: cqa implicit tau_cmd vs brucon OrderTau\n"
        "Test E reconstruction (post-K_b_pos-fix). Both in pre-WCF-demeaned form. "
        "Disagreement -> closed-loop controller dynamics differ.",
        fontsize=11,
    )
    fig.tight_layout()
    out = THIS / "tau_cmd_audit_diagnostic.png"
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    run()
