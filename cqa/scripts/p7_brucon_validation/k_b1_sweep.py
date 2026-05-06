"""K_b1 sensitivity sweep: does the bias-loop gain explain the residual
1.3x softness in |y_LF/F_drift|(f) over f in [0.005, 0.01] Hz?

Background
----------
analysis.md sect. 12.20.8 localised the brucon-vs-sandbox sigma_y_LF gap
to the bias-loop transition band. sect. 12.20.9 showed that turning on
the PI integrator in the sandbox closes ~9 % of the gap (sigma 0.484 ->
0.528 m vs brucon 0.687 m). Per the user's framing: same observer + same
innovation + same PID + same vessel SHOULD give matching transfers, so
where in the chain is it breaking?

This script tests the hypothesis that the brucon T_b = 1000 s bias loop
is somehow effectively "softer" in the band by sweeping the bias gain
K_b1 in the sandbox over {0.5, 1, 2, 4, 8} x nominal and overlaying the
sandbox |H_sandbox(f)| = |y_LF / F_drift|(f) against the empirical brucon
|H_brucon(f)| computed from the 30 long-run seeds (settle_s=3000, window
[1500, 3000] s).

Empirical transfer protocol (per seed)
-------------------------------------
  - input  u(t) = brucon DriftY(t)  [N]      (slow-drift sway force)
  - output y(t) = brucon sway_LF(t) [m]      (body-frame, demeaned)
  - Welch S_uu(f), S_yy(f), CSD S_uy(f) with nperseg covering ~5 cycles
    of the slowest band of interest (~200 s), nperseg = 5000 samples
    at 10 Hz = 500 s per segment, ~3 segments per 1500 s window
  - per-seed |H(f)| = |S_uy(f)| / S_uu(f)  (estimator H1, biased low
    when output noise is uncorrelated; for narrow-band coherent input
    this is fine)
  - ensemble average across seeds in S_uu / S_uy / S_yy domain
  - report ensemble |H| and coherence gamma^2 = |S_uy|^2 / (S_uu S_yy)

Sandbox transfer
----------------
  - For each K_b1 multiplier:
    A, B_w, _, C_y, C_yLF = build_closed_loop(use_observer=True,
                                              use_bias_ff=True,
                                              use_wave_filter=True,
                                              use_integrator=True,
                                              k_b1=KB1*mult)
    H(j*omega) = C_yLF . (j*omega*I - A)^-1 . B_w   [m / N]
  - Also compute predicted sigma_y_LF given empirical S_uu(f):
    sigma_pred^2 = (1/pi) integral |H|^2 S_uu d_omega

Outputs
-------
  - PNG: k_b1_sweep_transfer.png    (gitignored)
  - Console: integrated sigma per K_b1 vs brucon target
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.signal

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
_REPO_ROOT = str(THIS.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sandbox_passive_observer import (  # noqa: E402
    KB1,
    build_closed_loop,
)
from validate_sandbox_timeseries import (  # noqa: E402
    load_seed_csv,
    project_ned_to_body,
)

# -------- analysis settings -------------------------------------------
WORK_DIR = THIS / "work_long"
TAG = "pwo_long"
SEEDS = list(range(1000, 1030))
WINDOW = (1500.0, 3000.0)  # late window: bias and Tp settled
DT = 0.1                    # brucon sample period [s]
NPERSEG = 5000              # 500 s segment -> ~3 segments per seed
KB1_MULTIPLIERS = [0.5, 1.0, 2.0, 4.0, 8.0]
OUT_PNG = THIS / "k_b1_sweep_transfer.png"


def get_brucon_traces(seed: int):
    """Return (t, F_drift_N, sway_LF_body_m) for seed in WINDOW, demeaned."""
    seed_dir = WORK_DIR / f"{TAG}_seed{seed}"
    cols = load_seed_csv(seed_dir)
    t = cols["t"]
    mask = (t >= WINDOW[0]) & (t <= WINDOW[1])
    if not mask.any():
        raise RuntimeError(f"seed {seed}: no samples in {WINDOW}")
    t_w = t[mask]
    # body-frame sway_LF: project NED (x,y) through heading then subtract
    # body-frame WF channel yHf
    _, sway_total = project_ned_to_body(cols["x"][mask], cols["y"][mask],
                                        cols["heading"][mask])
    sway_lf = sway_total - cols["yHf"][mask]
    sway_lf -= sway_lf.mean()
    # DriftY is in kN -> N
    F = cols["DriftY"][mask] * 1.0e3
    F -= F.mean()
    return t_w, F, sway_lf


def empirical_transfer_ensemble(seeds: list[int]):
    """Welch-based H1 estimator, ensemble-averaged in PSD domain.

    Returns (f, |H|, gamma2, S_uu_avg, S_yy_avg).
    """
    fs = 1.0 / DT
    sums_uu = None
    sums_uy = None  # complex
    sums_yy = None
    n_used = 0
    for s in seeds:
        try:
            _, F, y = get_brucon_traces(s)
        except Exception as e:
            print(f"  seed {s}: skip ({e})")
            continue
        f, S_uu = scipy.signal.welch(F, fs=fs, nperseg=NPERSEG,
                                     return_onesided=True, scaling="density")
        _, S_yy = scipy.signal.welch(y, fs=fs, nperseg=NPERSEG,
                                     return_onesided=True, scaling="density")
        _, S_uy = scipy.signal.csd(F, y, fs=fs, nperseg=NPERSEG,
                                   return_onesided=True, scaling="density")
        if sums_uu is None:
            sums_uu = np.zeros_like(S_uu)
            sums_uy = np.zeros_like(S_uy, dtype=complex)
            sums_yy = np.zeros_like(S_yy)
            f_out = f
        sums_uu += S_uu
        sums_uy += S_uy
        sums_yy += S_yy
        n_used += 1
    if n_used == 0:
        raise RuntimeError("no seeds processed")
    S_uu_avg = sums_uu / n_used
    S_uy_avg = sums_uy / n_used
    S_yy_avg = sums_yy / n_used
    H_emp = np.abs(S_uy_avg) / S_uu_avg
    gamma2 = (np.abs(S_uy_avg) ** 2) / (S_uu_avg * S_yy_avg + 1e-30)
    print(f"  ensemble: {n_used} seeds, {len(f_out)} freqs, df = {f_out[1]:.5f} Hz")
    return f_out, H_emp, gamma2, S_uu_avg, S_yy_avg


def sandbox_transfer(omega: np.ndarray, k_b1_mult: float,
                     use_bias_ff: bool = True):
    """Return |H(j*omega)| = |y_LF / F_drift| for sandbox with K_b1 scaled."""
    A, B_w, _, C_y, C_yLF = build_closed_loop(
        use_observer=True, use_bias_ff=use_bias_ff,
        use_wave_filter=True, use_integrator=True,
        k_b1=KB1 * k_b1_mult,
    )
    n = A.shape[0]
    I = np.eye(n)
    Hmag = np.zeros_like(omega)
    for k, w in enumerate(omega):
        H = C_yLF @ np.linalg.solve(1j * w * I - A, B_w)
        Hmag[k] = abs(H)
    return Hmag


def integrated_sigma(omega: np.ndarray, S_F_omega: np.ndarray,
                      Hmag: np.ndarray) -> float:
    """sigma_y_LF^2 = integral |H(omega)|^2 S_F(omega) d_omega.

    S_F_omega is one-sided PSD vs angular frequency [N^2 / (rad/s)] -- i.e.
    the conversion S(omega) = S(f) / (2*pi) has already been applied. With
    that normalisation, sigma^2 is the bare integral over omega; do NOT
    divide by pi (that's the two-sided convention).
    """
    integrand = (Hmag ** 2) * S_F_omega
    return float(np.sqrt(np.trapezoid(integrand, omega)))


def main() -> None:
    print("=" * 78)
    print(f"K_b1 sensitivity sweep + empirical transfer overlay")
    print(f"  window: {WINDOW} s,  seeds: {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})")
    print(f"  K_b1 multipliers: {KB1_MULTIPLIERS}")
    print("=" * 78)
    print()

    # --- empirical transfer ---
    print("[1/3] computing empirical brucon |H_brucon|(f) from long-run seeds")
    f, H_emp, gamma2, S_uu_f, S_yy_f = empirical_transfer_ensemble(SEEDS)
    omega = 2.0 * np.pi * f                 # rad/s
    # convert one-sided PSD vs Hz -> vs rad/s: S(omega) = S(f) / (2*pi)
    S_uu_w = S_uu_f / (2.0 * np.pi)
    S_yy_w = S_yy_f / (2.0 * np.pi)

    sigma_F_emp = float(np.sqrt(np.trapezoid(S_uu_w, omega)))
    sigma_y_emp_check = float(np.sqrt(np.trapezoid(S_yy_w, omega)))
    sigma_y_pred_via_emp_H = integrated_sigma(omega, S_uu_w, H_emp)
    print(f"  sigma(F_drift)   from empirical S_uu = {sigma_F_emp/1e3:.1f} kN")
    print(f"  sigma(sway_LF)   from empirical S_yy = {sigma_y_emp_check:.3f} m")
    print(f"  sigma_y_LF predicted by emp |H|*emp S_uu = {sigma_y_pred_via_emp_H:.3f} m"
          f" (lower bound; H1 biases low when output noise present)")
    print()

    # --- sandbox transfers + predicted sigmas ---
    print("[2/3] sandbox |H_sandbox|(omega) for each K_b1 multiplier")
    # use the same omega grid as the empirical PSD
    Hmag_by_mult = {}
    sigma_by_mult = {}
    for mult in KB1_MULTIPLIERS:
        Hmag = sandbox_transfer(omega, mult, use_bias_ff=True)
        sig = integrated_sigma(omega, S_uu_w, Hmag)
        Hmag_by_mult[mult] = Hmag
        sigma_by_mult[mult] = sig
        print(f"  K_b1 * {mult:4.1f} (+bias FF): sigma_y_LF = {sig:.3f} m")
    # CRITICAL: brucon position_hold.cpp does NOT add bias FF!
    # It is pure PID on observer estimates. Verify by also computing the
    # sandbox WITHOUT bias FF (relying on PI integrator alone for DC rejection).
    Hmag_nobiasff = sandbox_transfer(omega, 1.0, use_bias_ff=False)
    sig_nobiasff = integrated_sigma(omega, S_uu_w, Hmag_nobiasff)
    print(f"  NO bias FF, K_b1 = nominal: sigma_y_LF = {sig_nobiasff:.3f} m"
          f"   <-- matches brucon controller actual structure")
    Hmag_by_mult["no_bff"] = Hmag_nobiasff
    sigma_by_mult["no_bff"] = sig_nobiasff
    print()

    # --- plot ---
    print("[3/3] plotting")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax = axes[0]
    # plot in Hz, log-log
    f_plot = f.copy()
    f_plot[0] = f_plot[1]                  # avoid log(0)
    ax.loglog(f_plot, H_emp, "k-", lw=2.0,
              label=f"brucon empirical (n={len(SEEDS)} seeds, window {WINDOW})")
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(KB1_MULTIPLIERS)))
    for c, mult in zip(colors, KB1_MULTIPLIERS):
        H = Hmag_by_mult[mult]
        H_plot = H.copy()
        H_plot[0] = H_plot[1]
        ax.loglog(f_plot, H_plot, color=c, lw=1.2,
                  label=f"sandbox K_b1*{mult:g} +biasFF (sigma={sigma_by_mult[mult]:.3f} m)")
    H_nbf = Hmag_by_mult["no_bff"].copy(); H_nbf[0] = H_nbf[1]
    ax.loglog(f_plot, H_nbf, color="red", lw=2.0, ls="--",
              label=f"sandbox NO bias FF (sigma={sigma_by_mult['no_bff']:.3f} m)  <- matches brucon controller")
    ax.axvspan(0.005, 0.01, color="orange", alpha=0.10,
               label="band where brucon is 1.3x softer (sect. 12.20.8)")
    ax.set_ylabel("|H(f)| = |y_LF / F_drift|  [m / N]")
    ax.set_title(
        f"Sandbox transfer vs brucon empirical, K_b1 sweep\n"
        f"empirical sigma_F_drift = {sigma_F_emp/1e3:.1f} kN, "
        f"empirical sigma_y_LF = {sigma_y_emp_check:.3f} m"
    )
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1e-3, 0.5)

    ax = axes[1]
    ax.semilogx(f_plot, gamma2, "k-", lw=1.5)
    ax.axvspan(0.005, 0.01, color="orange", alpha=0.10)
    ax.set_xlabel("frequency f [Hz]")
    ax.set_ylabel("coherence gamma^2(f)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Coherence between DriftY and sway_LF")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120)
    print(f"  saved: {OUT_PNG}")

    # --- band-localised numerical comparison ---
    label_keys = list(KB1_MULTIPLIERS) + ["no_bff"]

    def hdr(label):
        return f"K_b1*{label:g}" if isinstance(label, (int, float)) else "no_bff"

    print()
    print("Band-averaged |H| comparison (geometric mean over band):")
    bands = [(0.001, 0.003), (0.003, 0.005), (0.005, 0.01),
             (0.01, 0.02), (0.02, 0.05)]
    print(f"  {'band [Hz]':<14} {'brucon emp':>12} "
          + " ".join(f"{hdr(m):>12}" for m in label_keys))
    for lo, hi in bands:
        m = (f >= lo) & (f <= hi)
        if not m.any():
            continue
        H_b = float(np.exp(np.mean(np.log(H_emp[m] + 1e-30))))
        line = f"  [{lo:.3f}-{hi:.3f}] {H_b:>12.3e}"
        for key in label_keys:
            H_s = float(np.exp(np.mean(np.log(Hmag_by_mult[key][m] + 1e-30))))
            line += f" {H_s:>12.3e}"
        print(line)

    print()
    print("Ratio brucon / sandbox (>1 means sandbox is stiffer than brucon):")
    print(f"  {'band [Hz]':<14} " + " ".join(f"{hdr(m):>10}" for m in label_keys))
    for lo, hi in bands:
        m = (f >= lo) & (f <= hi)
        if not m.any():
            continue
        H_b = float(np.exp(np.mean(np.log(H_emp[m] + 1e-30))))
        line = f"  [{lo:.3f}-{hi:.3f}]"
        for key in label_keys:
            H_s = float(np.exp(np.mean(np.log(Hmag_by_mult[key][m] + 1e-30))))
            ratio = H_b / H_s if H_s > 0 else np.nan
            line += f" {ratio:>10.2f}"
        print(line)


if __name__ == "__main__":
    main()
