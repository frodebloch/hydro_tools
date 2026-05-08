"""Brucon EstBias slewing post-WCF — diagnose late-time +0.2 m offset.

Question: in the brucon truth ensemble (pwq30, use_tau_feedback=false
baseline), how does the observer's bias estimate b_hat evolve over the
[0, 90 s] post-WCF window?

cqa's Fix-3 model FREEZES b_hat at its pre-WCF value (the b_hat block
of A is zero; b_hat_dot = 0). The PI integrator is supposed to absorb
any post-WCF mismatch. But brucon's truth ensemble shows a sustained
+0.2 m offset in Δsway by t > 100 s that cqa's model does not
reproduce. If brucon's b_hat slews substantially during the post-WCF
window, that mechanism is missing from cqa.

Plot: per-seed EstBias{Surge,Sway,Yaw}(t) over [-30 s, +90 s] (i.e.
straddling the WCF event at t=560 s) for the 30 pwq30 seeds, plus the
ensemble mean. Show cqa's frozen-b_hat reference (the per-seed
pre-WCF value as a horizontal line through t=0).

Decision
--------
- if EstBias slews by O(50 kN) within [0, 90 s], the frozen-b_hat
  assumption is the dominant late-time offset mechanism. The fix is
  either (a) un-freeze b_hat post-WCF in the cqa model, or (b)
  acknowledge the limitation and use the integrator alone.
- if EstBias is essentially flat post-WCF, the +0.2 m offset is
  driven by something else (mean-drift force re-balance under
  heading deviation; secondary loop dynamics).
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

# Match calibrated_wcfdi_brucon_validation.py:
ENSEMBLE_DIR = THIS / "work"
TAG = "pwq30"
SEEDS = list(range(1000, 1030))
T_WCF_S = 560.0
T_PRE_WIN = 30.0   # show 30 s before WCF
T_POST_WIN = 120.0  # show 120 s after WCF

# Estimator column indices (per dp_cms_export.cpp + analysis.md):
COL_T = 0
COL_BIAS_SURGE_KN = 19
COL_BIAS_SWAY_KN = 20
COL_BIAS_YAW_KNM = 21


def load_seed_bias(seed: int):
    seed_dir = ENSEMBLE_DIR / f"{TAG}_seed{seed:04d}"
    est_path = seed_dir / f"{TAG}_seed{seed:04d}_estimator.out"
    if not est_path.exists():
        return None
    data = np.loadtxt(est_path, skiprows=1)
    t = data[:, COL_T]
    mask = (t >= T_WCF_S - T_PRE_WIN) & (t <= T_WCF_S + T_POST_WIN)
    t_rel = t[mask] - T_WCF_S
    # kN -> N, kNm -> Nm for consistency with cqa
    b_surge = 1e3 * data[mask, COL_BIAS_SURGE_KN]
    b_sway = 1e3 * data[mask, COL_BIAS_SWAY_KN]
    b_yaw = 1e3 * data[mask, COL_BIAS_YAW_KNM]
    # cqa's "frozen b_hat" reference = b at t = T_WCF - DT (last intact sample)
    idx_pre = int(np.searchsorted(t[mask], -0.05, side="right") - 1)
    if idx_pre < 0:
        idx_pre = 0
    b_pre = np.array([b_surge[idx_pre], b_sway[idx_pre], b_yaw[idx_pre]])
    return t_rel, b_surge, b_sway, b_yaw, b_pre


def main():
    seeds_data = []
    for s in SEEDS:
        d = load_seed_bias(s)
        if d is not None:
            seeds_data.append((s, *d))
    print(f"Loaded {len(seeds_data)}/{len(SEEDS)} seeds")
    if not seeds_data:
        raise RuntimeError("no seed data found")

    t = seeds_data[0][1]
    n_t = len(t)
    surge_all = np.array([s[2][:n_t] for s in seeds_data])
    sway_all = np.array([s[3][:n_t] for s in seeds_data])
    yaw_all = np.array([s[4][:n_t] for s in seeds_data])
    pre_all = np.array([s[5] for s in seeds_data])  # (n_seeds, 3)

    mean_surge = surge_all.mean(axis=0)
    mean_sway = sway_all.mean(axis=0)
    mean_yaw = yaw_all.mean(axis=0)

    # Slew between pre-WCF (t=0-) and quasi-steady-state at t=90 s:
    idx_pre = int(np.argmin(np.abs(t + 0.05)))      # last sample before t=0
    idx_post60 = int(np.argmin(np.abs(t - 60.0)))
    idx_post120 = int(np.argmin(np.abs(t - 120.0)))
    slew_surge_60 = surge_all[:, idx_post60] - surge_all[:, idx_pre]
    slew_sway_60 = sway_all[:, idx_post60] - sway_all[:, idx_pre]
    slew_yaw_60 = yaw_all[:, idx_post60] - yaw_all[:, idx_pre]
    slew_surge_120 = surge_all[:, idx_post120] - surge_all[:, idx_pre]
    slew_sway_120 = sway_all[:, idx_post120] - sway_all[:, idx_pre]
    slew_yaw_120 = yaw_all[:, idx_post120] - yaw_all[:, idx_pre]

    # --- plot ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    titles_units = [
        ("EstBiasSurge", surge_all, mean_surge, pre_all[:, 0], "kN", 1e3),
        ("EstBiasSway", sway_all, mean_sway, pre_all[:, 1], "kN", 1e3),
        ("EstBiasYaw", yaw_all, mean_yaw, pre_all[:, 2], "kNm", 1e3),
    ]
    for ax, (name, allarr, m, pre_arr, unit, scale) in zip(axes, titles_units):
        for s in allarr:
            ax.plot(t, s / scale, color="grey", alpha=0.25, lw=0.6)
        ax.plot(t, m / scale, "C0-", lw=2.0,
                label=f"brucon ensemble mean (n={len(allarr)})")
        # cqa frozen-b_hat reference: ensemble-mean pre-WCF value held flat
        ax.axhline(pre_arr.mean() / scale, color="r", ls="--", lw=1.5,
                   label=f"cqa frozen b_hat (= ensemble pre-WCF mean = {pre_arr.mean()/scale:+.0f} {unit})")
        ax.axvline(0, color="k", lw=0.4, alpha=0.5)
        ax.set_title(f"{name} [{unit}] over [-{T_PRE_WIN:.0f}, +{T_POST_WIN:.0f}] s around WCF")
        ax.set_ylabel(unit)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t since WCF [s]")
    fig.suptitle("Brucon observer bias estimate vs cqa frozen-b_hat assumption "
                 "(pwq30, use_tau_feedback=false baseline)", y=1.00)
    plt.tight_layout()
    out_png = THIS / "brucon_estbias_post_wcf.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"saved {out_png}")

    # --- summary ---
    print()
    print("--- ensemble-mean slew of b_hat (post-WCF minus pre-WCF) ---")
    print(f"  surge  @ t=60s:  {slew_surge_60.mean()/1e3:+7.1f} kN  "
          f"(per-seed std {slew_surge_60.std()/1e3:.1f} kN, "
          f"range [{slew_surge_60.min()/1e3:+.1f}, {slew_surge_60.max()/1e3:+.1f}] kN)")
    print(f"  sway   @ t=60s:  {slew_sway_60.mean()/1e3:+7.1f} kN  "
          f"(per-seed std {slew_sway_60.std()/1e3:.1f} kN, "
          f"range [{slew_sway_60.min()/1e3:+.1f}, {slew_sway_60.max()/1e3:+.1f}] kN)")
    print(f"  yaw    @ t=60s:  {slew_yaw_60.mean()/1e3:+7.1f} kNm "
          f"(per-seed std {slew_yaw_60.std()/1e3:.1f} kNm, "
          f"range [{slew_yaw_60.min()/1e3:+.1f}, {slew_yaw_60.max()/1e3:+.1f}] kNm)")
    print(f"  surge  @ t=120s: {slew_surge_120.mean()/1e3:+7.1f} kN")
    print(f"  sway   @ t=120s: {slew_sway_120.mean()/1e3:+7.1f} kN")
    print(f"  yaw    @ t=120s: {slew_yaw_120.mean()/1e3:+7.1f} kNm")
    print()
    pre_mean_kn = pre_all.mean(axis=0) / 1e3
    print(f"  ensemble-mean PRE-WCF b_hat: surge={pre_mean_kn[0]:+.1f} kN, "
          f"sway={pre_mean_kn[1]:+.1f} kN, yaw={pre_mean_kn[2]:+.1f} kNm")
    print(f"  -> if |slew| << |pre|, frozen-b_hat is a good approximation")
    print(f"  -> if |slew| ~ |pre| or larger, brucon observer is finding a new "
          f"equilibrium that cqa's frozen-b_hat misses")


if __name__ == "__main__":
    main()
