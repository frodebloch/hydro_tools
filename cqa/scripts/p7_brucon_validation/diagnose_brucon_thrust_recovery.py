"""Plot brucon delivered thrust (Tx,Ty,Tz) vs commanded (OrderTau*)
around t_WCF for several seeds in the pwq30 ensemble.

Goal: see how long the post-WCF thrust deficit (Order - Delivered)
actually persists.  This bears on the choice of T_eff for the cqa
calibrated open-loop tau_lost pulse.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent.parent))

from harness import parse_output  # noqa: E402


HERE = THIS
WORK = HERE / "work"
TAG = "pwq30"
T_WCF = 560.0
WINDOW_BEFORE = 30.0
WINDOW_AFTER = 60.0
SEEDS = [1000, 1001, 1002, 1003, 1018, 1028]  # mix of low/high truth peaks


def load_seed(seed: int):
    p = WORK / f"{TAG}_seed{seed}" / f"{TAG}_seed{seed}.out"
    res = parse_output(p)
    c = res.columns
    return {
        "seed": seed,
        "t": c["t"],
        "Tx": c["Tx"],
        "Ty": c["Ty"],
        "Tz": c["Tz"],
        "OrdSurge": c["OrderTauSurge"],
        "OrdSway": c["OrderTauSway"],
        "OrdYaw": c["OrderTauYaw"],
        "SurgeDev": c["SurgeDev"],
        "SwayDev": c["SwayDev"],
        "HeadingDev": c["HeadingDev"],
    }


def deficit_recovery_time(t, order, delivered, t_wcf,
                           settle_window=(120.0, 180.0), tol_frac=0.20):
    """Time after t_WCF for the deficit |Order - Delivered| to settle
    within tol_frac of its peak transient value.

    'Settled' = decayed to <= tol_frac * peak_transient_deficit AND stays
    below thereafter.  The peak is computed over [t_wcf, t_wcf+30]s; the
    settle reference is the deficit's median over settle_window."""
    post = t >= t_wcf
    t_post = t[post] - t_wcf
    deficit = order[post] - delivered[post]  # signed
    abs_def = np.abs(deficit)
    early = (t_post >= 0) & (t_post <= 30.0)
    if not early.any():
        return float("nan"), float("nan")
    peak = float(np.max(abs_def[early]))
    # tolerance band relative to peak transient excursion
    tol = tol_frac * peak
    settled = (t_post >= 0) & (abs_def <= tol)
    # Find first time after the peak when we're settled and stay settled
    # for at least 5 s.
    idx_peak = int(np.argmax(abs_def[early]))
    t_peak = float(t_post[early][idx_peak])
    after_peak = t_post >= t_peak
    abs_after = abs_def[after_peak]
    t_after = t_post[after_peak]
    # walk forward, find first sustained-settled instant
    win_samples = 50  # ~5 s if dt=0.1
    settle_t = float("nan")
    for k in range(len(t_after) - win_samples):
        if np.all(abs_after[k:k + win_samples] <= tol):
            settle_t = float(t_after[k] - t_peak)
            break
    return peak, settle_t


def main():
    seeds = [load_seed(s) for s in SEEDS]
    # 6 columns: per-DOF (Order, Delivered) overlay + deficit subpanel
    # Layout: rows = seeds, columns = (surge order/deliv, surge deficit,
    # sway order/deliv, sway deficit, yaw order/deliv, yaw deficit)
    fig, axes = plt.subplots(len(seeds), 6, figsize=(20, 2.4 * len(seeds)),
                              sharex=True)
    dof_specs = [
        ("surge", "Tx", "OrdSurge", "kN"),
        ("sway",  "Ty", "OrdSway",  "kN"),
        ("yaw",   "Tz", "OrdYaw",   "kNm"),
    ]
    summary = []
    for i, s in enumerate(seeds):
        t = s["t"]
        mask = (t > T_WCF - WINDOW_BEFORE) & (t < T_WCF + WINDOW_AFTER)
        t_rel = t[mask] - T_WCF
        seed_summary = {"seed": s["seed"]}
        for j, (dof, deliv_key, order_key, unit) in enumerate(dof_specs):
            deliv = s[deliv_key]
            order = s[order_key]
            ax_overlay = axes[i, 2 * j]
            ax_deficit = axes[i, 2 * j + 1]
            ax_overlay.plot(t_rel, order[mask], color="tab:red", lw=1.2,
                            label="Order")
            ax_overlay.plot(t_rel, deliv[mask], color="black", lw=1.2,
                            label="Delivered")
            ax_overlay.axvline(0, color="gray", ls=":", lw=0.8)
            ax_overlay.axhline(0, color="k", lw=0.4, alpha=0.3)
            ax_overlay.set_title(f"seed {s['seed']} {dof}", fontsize=9)
            ax_overlay.set_ylabel(f"[{unit}]")
            ax_overlay.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax_overlay.legend(fontsize=7, loc="best")

            deficit = (order[mask] - deliv[mask])
            ax_deficit.plot(t_rel, deficit, color="tab:purple", lw=1.2)
            ax_deficit.axvline(0, color="gray", ls=":", lw=0.8)
            ax_deficit.axhline(0, color="k", lw=0.4, alpha=0.3)
            peak, settle_t = deficit_recovery_time(t, order, deliv, T_WCF, tol_frac=0.20)
            ax_deficit.set_title(
                f"deficit (Ord-Del)  pk={peak:+.0f}  settle≈{settle_t:.1f}s",
                fontsize=9,
            )
            ax_deficit.set_ylabel(f"deficit [{unit}]")
            ax_deficit.grid(alpha=0.3)
            seed_summary[f"{dof}_pk"] = peak
            seed_summary[f"{dof}_settle"] = settle_t
        summary.append(seed_summary)
    for ax in axes[-1]:
        ax.set_xlabel("time since WCF [s]")
    fig.suptitle(
        f"brucon {TAG} (β=30°, t_WCF={T_WCF} s): commanded vs delivered thrust "
        f"and post-WCF deficit (Ord-Del). Settle = recovery to 20% of peak deficit.",
        fontsize=11,
    )
    fig.tight_layout()
    out = HERE / "diagnose_brucon_thrust_recovery.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")

    print("\n--- per-seed deficit peak and 20%-settle time after peak ---")
    print(f"  {'seed':>5} | "
          f"{'surge pk[kN]':>12} {'settle[s]':>10} | "
          f"{'sway pk[kN]':>12} {'settle[s]':>10} | "
          f"{'yaw pk[kNm]':>12} {'settle[s]':>10}")
    for s in summary:
        print(
            f"  {s['seed']:>5} | "
            f"{s['surge_pk']:>+12.0f} {s['surge_settle']:>10.1f} | "
            f"{s['sway_pk']:>+12.0f} {s['sway_settle']:>10.1f} | "
            f"{s['yaw_pk']:>+12.0f} {s['yaw_settle']:>10.1f}"
        )


if __name__ == "__main__":
    main()
