"""Brucon (Order - Delivered) thrust deficit over [0, 60s] post-WCF.

Question: does the per-DOF thrust deficit (commanded minus delivered)
really collapse to ~0 within ~9 s as our cqa tau_lost(t) pulse model
assumes, or does it have a sustained tail that explains the larger /
later sway peak and the late-time +0.2 m offset in the brucon truth
ensemble mean?

Plot per-seed deficit traces (light) + ensemble mean (bold) for surge,
sway, yaw over [0, 90 s] post-WCF. Also overlay the cqa pulse model
(linear-decay over 9 s with peak = per-seed median) for comparison.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from harness import parse_output

# Match the validation script:
ENSEMBLE_DIR = Path(__file__).parent / "work"
TAG = "pwq30"
T_WCF_S = 560.0
T_POST_S = 90.0  # extend window vs the 30 s used for tau_lost extraction
SEEDS = list(range(1000, 1030))
T_PULSE_CQA = 9.0


def load_seed_deficit(seed: int):
    seed_dir = ENSEMBLE_DIR / f"{TAG}_seed{seed:04d}"
    out_path = seed_dir / f"{TAG}_seed{seed:04d}.out"
    if not out_path.exists():
        return None
    res = parse_output(out_path)
    t = res.columns["t"]
    Tx = res.columns["Tx"]
    Ty = res.columns["Ty"]
    Tz = res.columns["Tz"]
    OrderSurge = res.columns["OrderTauSurge"]
    OrderSway = res.columns["OrderTauSway"]
    OrderYaw = res.columns["OrderTauYaw"]
    mask = (t >= T_WCF_S) & (t <= T_WCF_S + T_POST_S)
    t_post = t[mask] - T_WCF_S
    # All in kN / kNm in the .out file. Convert to N / Nm for consistency.
    def_surge = 1e3 * (OrderSurge[mask] - Tx[mask])
    def_sway = 1e3 * (OrderSway[mask] - Ty[mask])
    def_yaw = 1e3 * (OrderYaw[mask] - Tz[mask])
    return t_post, def_surge, def_sway, def_yaw


def main():
    seeds_data = []
    for s in SEEDS:
        d = load_seed_deficit(s)
        if d is not None:
            seeds_data.append((s, *d))
    print(f"Loaded {len(seeds_data)}/{len(SEEDS)} seeds")
    if not seeds_data:
        raise RuntimeError("no seed data found")

    # Common time grid (assume identical from brucon).
    t = seeds_data[0][1]
    n_t = len(t)
    def_surge_all = np.array([d[2][:n_t] for d in seeds_data])
    def_sway_all = np.array([d[3][:n_t] for d in seeds_data])
    def_yaw_all = np.array([d[4][:n_t] for d in seeds_data])

    mean_surge = def_surge_all.mean(axis=0)
    mean_sway = def_sway_all.mean(axis=0)
    mean_yaw = def_yaw_all.mean(axis=0)

    # cqa pulse model: linear decay 0 -> T_PULSE_CQA s, peak = ensemble peak
    # of |def| within [0, 30 s] (matches extraction window in the
    # validation script).
    mask_30 = t <= 30.0
    peak_surge = float(np.median([
        s[np.argmax(np.abs(s[mask_30]))] for s in def_surge_all
    ]))
    peak_sway = float(np.median([
        s[np.argmax(np.abs(s[mask_30]))] for s in def_sway_all
    ]))
    peak_yaw = float(np.median([
        y[np.argmax(np.abs(y[mask_30]))] for y in def_yaw_all
    ]))

    def cqa_pulse(t_arr, peak):
        out = np.where(t_arr < T_PULSE_CQA, peak * (1.0 - t_arr / T_PULSE_CQA), 0.0)
        out = np.where(t_arr < 0, 0.0, out)
        return out

    pulse_surge = cqa_pulse(t, peak_surge)
    pulse_sway = cqa_pulse(t, peak_sway)
    pulse_yaw = cqa_pulse(t, peak_yaw)

    # --- plot ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    titles = ["Surge deficit (Order - Delivered) [kN]",
              "Sway deficit (Order - Delivered) [kN]",
              "Yaw deficit (Order - Delivered) [kNm]"]
    means = [mean_surge, mean_sway, mean_yaw]
    alls = [def_surge_all, def_sway_all, def_yaw_all]
    pulses = [pulse_surge, pulse_sway, pulse_yaw]
    peaks = [peak_surge, peak_sway, peak_yaw]
    for ax, ttl, m, allarr, pulse, peak in zip(axes, titles, means, alls, pulses, peaks):
        for s in allarr:
            ax.plot(t, s / 1e3, color="grey", alpha=0.25, lw=0.6)
        ax.plot(t, m / 1e3, "C0-", lw=2.0,
                label=f"brucon ensemble mean (n={len(allarr)})")
        ax.plot(t, pulse / 1e3, "k--", lw=1.8,
                label=f"cqa pulse model (peak={peak/1e3:+.0f}, T={T_PULSE_CQA:.0f}s lin-decay)")
        ax.axhline(0, color="k", lw=0.4, alpha=0.5)
        ax.axvline(T_PULSE_CQA, color="k", lw=0.4, alpha=0.5, ls=":")
        ax.set_title(ttl)
        ax.set_ylabel("kN" if "Sway" in ttl or "Surge" in ttl else "kNm")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t since WCF [s]")
    fig.suptitle("Brucon thrust deficit vs cqa pulse model", y=1.00)
    plt.tight_layout()

    out_path = "scripts/p7_brucon_validation/brucon_thrust_deficit.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")

    # --- print summary ---
    print()
    print("--- summary at key times (ensemble mean) ---")
    for label, m in zip(["surge [kN]", "sway [kN]", "yaw [kNm]"], means):
        print(f"{label:>14s}:  "
              f"t=0+: {m[1]/1e3:+7.1f}  "
              f"t=9s: {m[int(9.0/(t[1]-t[0]))]/1e3:+7.1f}  "
              f"t=30s: {m[int(30.0/(t[1]-t[0]))]/1e3:+7.1f}  "
              f"t=60s: {m[int(60.0/(t[1]-t[0]))]/1e3:+7.1f}  "
              f"t=85s: {m[int(85.0/(t[1]-t[0]))]/1e3:+7.1f}")


if __name__ == "__main__":
    main()
