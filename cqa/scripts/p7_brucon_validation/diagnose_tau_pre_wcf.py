"""Pre-WCF Tx/Ty/Tz diagnostic for regime-B severity estimator design
(sec.12.21.21.19).

For one outlier seed of bf8_q10_w45, examine:

  1. The identity Tx/y/z ?= FbTauSurge/Sway/Yaw (cols 8/9/10 vs 41/42/43).
     The brucon C++ implementation will expose ``FeedbackThrust`` per
     DOF, which should be the same quantity. If they diverge, we need
     to understand the difference before assuming equivalence.

  2. The (mean, std) of pre-WCF Ty at several window lengths
     {60, 300, 900} s, to characterise the sampling-error vs
     stationarity trade-off for the live estimator.

  3. The PSD of pre-WCF Ty over the longest available window, to choose
     an LF-band cutoff frequency for separating LF from WF demand.

  4. Compare ``(mu, sigma)`` of raw Ty vs LF-filtered Ty at several
     candidate cutoffs (0.05, 0.10, 0.20 rad/s).

Outputs a single multi-panel PNG. No commits; this is purely diagnostic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, butter, filtfilt

THIS = Path(__file__).resolve().parent

T_WCF_S = 1560.0
SEED = 1027
TAG = "bf8_q10_w45"
SEED_PATH = THIS / "work" / f"{TAG}_seed{SEED}" / f"{TAG}_seed{SEED}.out"

# Column indices (0-based) per the .out header inspection (sec.12.21.21.18).
COL_T = 0
COL_TX = 7   # delivered surge thrust [kN]
COL_TY = 8   # delivered sway thrust  [kN]
COL_TZ = 9   # delivered yaw moment   [kN*m? or kN? -- check below]
COL_FBX = 40 # FbTauSurge [kN]
COL_FBY = 41 # FbTauSway  [kN]
COL_FBZ = 42 # FbTauYaw

WINDOWS_S = [60.0, 300.0, 900.0]
CUTOFFS_RAD = [0.05, 0.10, 0.20]  # candidate LF cutoffs

# Sample rate from SIM_DT
SIM_DT = 0.1
FS = 1.0 / SIM_DT


def butter_lp(x: np.ndarray, w_c: float) -> np.ndarray:
    """Zero-phase 4th-order Butterworth low-pass, cutoff in rad/s."""
    f_c = w_c / (2.0 * np.pi)
    b, a = butter(4, f_c / (FS / 2.0), btype="low")
    return filtfilt(b, a, x)


def main() -> None:
    d = np.loadtxt(SEED_PATH, skiprows=1)
    t = d[:, COL_T]
    Tx = d[:, COL_TX]
    Ty = d[:, COL_TY]
    Tz = d[:, COL_TZ]
    FbX = d[:, COL_FBX]
    FbY = d[:, COL_FBY]
    FbZ = d[:, COL_FBZ]

    # Pre-WCF window: settle has finished, WCF not yet triggered.
    pre = (t >= 600.0) & (t <= T_WCF_S - 5.0)
    t_pre = t[pre]
    Ty_pre = Ty[pre]
    FbY_pre = FbY[pre]
    print(f"Pre-WCF window {t_pre[0]:.0f}..{t_pre[-1]:.0f} s, "
          f"N = {len(t_pre)} samples ({(t_pre[-1]-t_pre[0])/60:.1f} min)")

    # ----- 1. Tx/y/z vs Fb identity -----
    print("\n[1] Identity check: T (delivered) vs Fb (feedback)")
    for label, T_, Fb_ in [("surge", Tx[pre], FbX[pre]),
                           ("sway",  Ty[pre], FbY[pre]),
                           ("yaw",   Tz[pre], FbZ[pre])]:
        diff = T_ - Fb_
        rel = np.std(diff) / (np.std(T_) + 1e-9)
        print(f"  {label:5s}: T mean={T_.mean():+8.2f}  Fb mean={Fb_.mean():+8.2f}  "
              f"|diff| std={np.std(diff):.3f}  rel={rel*100:.2f}% of T std")

    # ----- 2. Window-length dependence of (mu, sigma) of Ty -----
    print("\n[2] Pre-WCF Ty (mu, sigma) vs window length")
    print(f"  {'window':>8s}  {'mu (kN)':>10s}  {'sigma (kN)':>11s}  "
          f"{'N_eff~':>8s}  {'rel err sigma':>14s}")
    rows_window = []
    for w_s in WINDOWS_S:
        # take last w_s seconds of pre window
        mask = t_pre >= (t_pre[-1] - w_s)
        Ty_w = Ty_pre[mask]
        mu = Ty_w.mean()
        sig = Ty_w.std()
        # Effective N at LF: T_decorr_LF ~ 15 s
        N_eff = w_s / 15.0
        rel_err = 1.0 / np.sqrt(2.0 * N_eff)  # sigma estimator rel error
        rows_window.append((w_s, mu, sig, N_eff, rel_err))
        print(f"  {w_s:8.0f}  {mu:+10.2f}  {sig:11.2f}  "
              f"{N_eff:8.1f}  {rel_err*100:13.1f}%")

    # ----- 3. PSD of pre-WCF Ty -----
    fY, Pxx = welch(Ty_pre - Ty_pre.mean(), fs=FS, nperseg=4096)
    omega = 2.0 * np.pi * fY

    # ----- 4. mu/sigma vs LF cutoff -----
    print("\n[3] (mu, sigma) of LF-filtered Ty vs cutoff frequency")
    print(f"  {'w_c (rad/s)':>12s}  {'T_c (s)':>8s}  {'mu (kN)':>10s}  "
          f"{'sigma_LF (kN)':>14s}  {'sigma_LF/sigma_full':>20s}")
    sigma_full = Ty_pre.std()
    rows_lp = []
    for w_c in CUTOFFS_RAD:
        Ty_LF = butter_lp(Ty_pre, w_c)
        mu_LF = Ty_LF.mean()
        sig_LF = Ty_LF.std()
        rows_lp.append((w_c, Ty_LF))
        print(f"  {w_c:12.3f}  {2*np.pi/w_c:8.1f}  {mu_LF:+10.2f}  "
              f"{sig_LF:14.2f}  {sig_LF/sigma_full:20.3f}")

    # ----- Plot -----
    fig, axes = plt.subplots(4, 1, figsize=(11, 12))

    # Panel 1: T vs Fb time series, sway
    ax = axes[0]
    ax.plot(t_pre, Ty_pre, "C0-", lw=0.6, label="Ty (delivered)")
    ax.plot(t_pre, FbY_pre, "C1--", lw=0.6, alpha=0.7, label="FbTauSway")
    ax.set_title(f"{TAG} seed {SEED}: pre-WCF sway demand "
                 f"(T_delivered vs FbTauSway)")
    ax.set_ylabel("tau_y [kN]")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: PSD of pre-WCF Ty (log-log) + cutoff markers
    ax = axes[1]
    ax.loglog(omega, Pxx, "k-", lw=1.0)
    for w_c in CUTOFFS_RAD:
        ax.axvline(w_c, ls="--", lw=0.8, alpha=0.6,
                   label=f"w_c = {w_c:.2f} rad/s (T={2*np.pi/w_c:.0f}s)")
    ax.set_title("PSD of pre-WCF Ty (mean removed)")
    ax.set_xlabel("omega [rad/s]")
    ax.set_ylabel("S_Ty(omega) [kN^2 / (rad/s)]")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.set_xlim(1e-3, FS * np.pi)

    # Panel 3: LF-filtered Ty time series for each cutoff
    ax = axes[2]
    ax.plot(t_pre, Ty_pre, "0.7", lw=0.4, label="raw Ty")
    colors = ["C2", "C3", "C4"]
    for (w_c, Ty_LF), c in zip(rows_lp, colors):
        ax.plot(t_pre, Ty_LF, color=c, lw=1.2,
                label=f"LF, w_c={w_c:.2f} rad/s")
    ax.set_title("Raw vs LF-filtered Ty (last 600 s shown)")
    ax.set_xlim(t_pre[-1] - 600, t_pre[-1])
    ax.set_ylabel("tau_y [kN]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 4: rolling (mu, sigma) of Ty at three window lengths
    ax = axes[3]
    for w_s, color in zip([60.0, 300.0, 900.0], ["C0", "C1", "C2"]):
        n_w = int(w_s / SIM_DT)
        if n_w >= len(Ty_pre):
            continue
        mu_roll = np.array([Ty_pre[max(0, k-n_w):k].mean()
                            for k in range(n_w, len(Ty_pre))])
        sig_roll = np.array([Ty_pre[max(0, k-n_w):k].std()
                             for k in range(n_w, len(Ty_pre))])
        t_roll = t_pre[n_w:]
        ax.plot(t_roll, mu_roll, color=color, lw=1.0,
                label=f"window {w_s:.0f}s mean")
        ax.fill_between(t_roll, mu_roll - sig_roll, mu_roll + sig_roll,
                        color=color, alpha=0.15)
    ax.set_title("Rolling (mu +/- 1*sigma) of Ty at three window lengths")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("tau_y [kN]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    out = THIS / f"diagnose_tau_pre_wcf_{TAG}_seed{SEED}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
