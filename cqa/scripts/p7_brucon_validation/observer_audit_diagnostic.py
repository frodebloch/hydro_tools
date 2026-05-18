"""Observer audit: compare cqa linear b_hat(t) vs brucon EstBias(t) post-WCF.

Question (sec.12.21.21.28 follow-up to Test E/F result):
  Test E (closed-loop linear + brucon delta_tau + brucon env-force via B_d)
  gives ratio P95 sway = 0.78 on outlier (1012) and 1.29 on calm (1001).
  The calm-vs-outlier asymmetry suggests the observer's bias estimator
  b_hat(t) tracks low-frequency env force in brucon but not in our
  linear reconstruction (or tracks too aggressively).

This script:
  1) integrates Test E (linear closed loop + dtau + env force) and
     extracts b_hat_y(t) from the augmented state;
  2) loads brucon EstBiasSway(t) from the estimator log;
  3) plots both, plus the env-force trajectory dF_env_y(t), to see
     whether brucon's bias estimator is absorbing more of the env force
     than ours does;
  4) reports mean / RMS / lag between the two b_hat traces.

Interpretation:
  - If brucon b_hat_y closely tracks dF_env_y while cqa b_hat_y lags,
    the cqa observer bandwidth is too slow -> calm-seed over-prediction
    explained (env force is uncancelled in cqa, so the rigid body
    responds; in brucon it's largely absorbed by b_hat and fed forward
    by the controller).
  - If brucon b_hat_y is *less* responsive than cqa's, the gap is
    elsewhere (controller saturation, integrator anti-windup).
  - If both b_hat traces agree but cqa sway still over-shoots, the
    issue is downstream (controller feedforward of b_hat is different,
    or the b_hat -> tau_thr cancellation in cqa A-matrix is wrong).

Run:
  PYTHONPATH=. .venv/bin/python scripts/p7_brucon_validation/observer_audit_diagnostic.py
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
from cqa.transient_obs import IDX_ETA, IDX_B_HAT  # noqa: E402

# Reuse extractors and integrators from the linearity diagnostic
from scripts.p7_brucon_validation.linearity_reconstruction_diagnostic import (  # noqa: E402
    load_seed,
    extract_delta_tau,
    extract_brucon_sway,
    extract_env_force_perturbation,
    reconstruct_with_env,
    T_WCF,
    T_HORIZON,
    DT_INT,
)

SEEDS = {
    "outlier_1012": 1012,
    "calm_1001": 1001,
}


def extract_brucon_b_hat_y(seed: int, t_grid: np.ndarray) -> np.ndarray:
    """Brucon EstBiasSway(t) on grid, demeaned by pre-WCF mean. Units: N."""
    f = THIS / "work" / f"bf8_q10_w45_seed{seed}" / f"bf8_q10_w45_seed{seed}_estimator.out"
    d = np.genfromtxt(f, names=True, delimiter="\t")
    t_b = d["Time"]
    bsway = d["EstBiasSway"]  # in kN
    pre = (t_b >= T_WCF - 30.0) & (t_b <= T_WCF - 5.0)
    bsway_pre = float(bsway[pre].mean())
    t_abs = T_WCF + t_grid
    return (np.interp(t_abs, t_b, bsway) - bsway_pre) * 1e3  # kN -> N


def run() -> None:
    cfg = csov_default_config()
    aug = _build_aug_for_live(cfg, Tp_obs_s=10.0)
    print(f"Built aug: n_state={aug.n_state}, T_thr={aug.T_thr:.3f}, K_b_pos diag = {np.diag(aug.A[IDX_B_HAT, IDX_ETA.start:IDX_ETA.stop])}")

    t_grid = np.arange(0.0, T_HORIZON + DT_INT * 0.5, DT_INT)

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)

    for col, (label, seed) in enumerate(SEEDS.items()):
        data = load_seed(seed)
        dtau = extract_delta_tau(data, t_grid)
        df_env = extract_env_force_perturbation(data, t_grid)
        brucon_sway = extract_brucon_sway(data, t_grid)
        brucon_bhat_y = extract_brucon_b_hat_y(seed, t_grid)

        # cqa linear closed loop, Test E
        X_E = reconstruct_with_env(aug, t_grid, dtau, df_env)
        sway_E = X_E[:, IDX_ETA][:, 1]
        cqa_bhat_y = X_E[:, IDX_B_HAT][:, 1]  # demeaned (x0=0 already pre-WCF reference)

        def p95(x):
            return float(np.percentile(np.abs(x), 95))

        print(f"\n=== {label} (seed {seed}) ===")
        print(f"  brucon sway P95:        {p95(brucon_sway):.2f} m")
        print(f"  Test E   sway P95:       {p95(sway_E):.2f} m  (ratio {p95(sway_E)/p95(brucon_sway):.2f})")
        print(f"  dF_env_y RMS / peak:   {np.std(df_env[:, 1])/1e3:.1f} / {np.abs(df_env[:, 1]).max()/1e3:.1f} kN")
        print(f"  brucon b_hat_y RMS / peak / final: "
              f"{np.std(brucon_bhat_y)/1e3:.1f} / {np.abs(brucon_bhat_y).max()/1e3:.1f} / {brucon_bhat_y[-1]/1e3:.1f} kN")
        print(f"  cqa    b_hat_y RMS / peak / final: "
              f"{np.std(cqa_bhat_y)/1e3:.1f} / {np.abs(cqa_bhat_y).max()/1e3:.1f} / {cqa_bhat_y[-1]/1e3:.1f} kN")

        # Cross-correlation lag (cqa vs brucon b_hat)
        x = cqa_bhat_y - cqa_bhat_y.mean()
        y = brucon_bhat_y - brucon_bhat_y.mean()
        if np.std(x) > 1e-6 and np.std(y) > 1e-6:
            xc = np.correlate(x, y, mode="full") / (np.std(x) * np.std(y) * len(x))
            lags = np.arange(-len(x) + 1, len(x)) * DT_INT
            ilag = np.argmax(xc)
            print(f"  b_hat cross-corr peak: {xc[ilag]:.3f} at lag {lags[ilag]:+.1f} s "
                  f"(positive = cqa lags brucon)")

        # Row 0: sway truth vs Test E reconstruction
        ax = axes[0, col]
        ax.plot(t_grid, brucon_sway, color="C0", lw=1.8, label="brucon truth")
        ax.plot(t_grid, sway_E, color="C2", lw=1.2, label="Test E (cqa)")
        ax.axhline(0, color="black", lw=0.4)
        ax.set_title(f"{label}: sway")
        ax.set_ylabel("sway [m]")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        # Row 1: b_hat_y, brucon vs cqa, with dF_env_y on twinx
        ax = axes[1, col]
        ax2 = ax.twinx()
        ax.plot(t_grid, brucon_bhat_y / 1e3, color="C0", lw=1.6, label="brucon EstBiasSway")
        ax.plot(t_grid, cqa_bhat_y / 1e3, color="C3", lw=1.4, ls="--", label="cqa b_hat_y (Test E)")
        ax2.plot(t_grid, df_env[:, 1] / 1e3, color="C7", lw=0.7, alpha=0.7, label="dF_env_y")
        ax.axhline(0, color="black", lw=0.4)
        ax.set_ylabel("b_hat_y [kN]")
        ax2.set_ylabel("dF_env_y [kN]")
        ax.set_title("observer bias estimate vs env force")
        ax.grid(alpha=0.3)
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, loc="best", fontsize=7)

        # Row 2: residual env force (after subtracting b_hat as feedforward)
        # Plot dF_env_y - b_hat_y for both (positive = env force NOT absorbed by observer)
        ax = axes[2, col]
        ax.plot(t_grid, (df_env[:, 1] - brucon_bhat_y) / 1e3, color="C0", lw=1.4,
                label="dF_env - brucon b_hat")
        ax.plot(t_grid, (df_env[:, 1] - cqa_bhat_y) / 1e3, color="C3", lw=1.4, ls="--",
                label="dF_env - cqa b_hat")
        ax.axhline(0, color="black", lw=0.4)
        ax.set_ylabel("uncancelled forcing [kN]")
        ax.set_xlabel("t - T_WCF [s]")
        ax.set_title("env force NOT absorbed by observer")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(
        "bf8_q10_w45 observer audit: cqa b_hat(t) vs brucon EstBias(t)\n"
        "K_b_pos = (0.0012, 0.0012, 0.002), tau_b = 1000 s (cqa config). "
        "Sway P95 ratio: outlier 0.78, calm 1.29 (Test E).",
        fontsize=11,
    )
    fig.tight_layout()
    out = THIS / "observer_audit_diagnostic.png"
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    run()
