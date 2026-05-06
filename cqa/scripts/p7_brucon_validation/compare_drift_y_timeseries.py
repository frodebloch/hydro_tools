"""Compare brucon's realised DriftY time series to cqa's spectral slow-drift PSD model.

Purpose
-------
Section 12.17.3 of analysis.md identified a residual closed-loop variance
gap (cqa over-predicts intact sigma_y by ~60 %) after both spectral-drift
migration and brucon-aligned added-mass alignment. The residual must
live in the slow-drift PSD shape itself (not in mean-drift magnitude
which sec.12.16 validated to 1 %). This script tests that hypothesis
*directly* by:

  1. extracting DriftY(t) from each brucon-ensemble seed in the late
     intact window (t in [failure - 200 s, failure - 1 s]),
  2. computing per-seed std(DriftY) and an autocorrelation-based
     decorrelation time, plus the ensemble-averaged empirical PSD,
  3. building cqa's `slow_drift_force_psd_newman_pdstrip` for the same
     point and overlaying it on the empirical PSD, and
  4. realising a same-length cqa time series via Shinozuka and
     computing matching diagnostics for direct visual comparison.

The output is a single PDF with three panels (overlay PSD, time series
sample, autocorrelation) and a console table of the headline numbers.

Usage
-----
    .venv/bin/python scripts/p7_brucon_validation/compare_drift_y_timeseries.py

Requires that `run_comparison.py` has already populated
`scripts/p7_brucon_validation/work/p7v1_seed*` (the brucon ensemble).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_REPO_ROOT = str(HERE.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from harness import parse_output  # noqa: E402

from cqa.drift import slow_drift_force_psd_newman_pdstrip  # noqa: E402
from cqa.rao import load_pdstrip_rao  # noqa: E402
from cqa.time_series_realisation import realise_vector_force_time_series  # noqa: E402

# ---------------- operating point (same as run_comparison.py) ---------
VW = 14.0
HS = 4.196
TP = 10.224
THETA_REL = np.radians(90.0)        # beam-on
SETTLE_S = 500.0
ACTIVATE_SK_S = 60.0
FAILURE_TIME_S = ACTIVATE_SK_S + SETTLE_S
INTACT_SAMPLE_S = 200.0

PDSTRIP_PATH = "/home/blofro/src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"
WORK_DIR = HERE / "work"
OUT_PDF = HERE / "p7_drift_y_timeseries_compare.pdf"


def empirical_psd_one_sided(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided PSD via Welch with default Hann + 50 % overlap.

    Convention matches cqa's: integral S(omega) d_omega over [0, +inf)
    equals var(x). We use scipy.signal.welch and convert from S(f)
    (one-sided in Hz, integral over [0, +inf) Hz = var) to S(omega)
    by dividing by 2*pi.
    """
    from scipy.signal import welch
    nperseg = min(len(x) // 4, 2048)
    f, Sxx_f = welch(x, fs=1.0/dt, nperseg=nperseg, return_onesided=True,
                     scaling="density")
    omega = 2.0 * np.pi * f
    Sxx_omega = Sxx_f / (2.0 * np.pi)
    return omega, Sxx_omega


def empirical_decorr_time(x: np.ndarray, dt: float) -> float:
    """First time the (positive) autocorrelation drops below 1/e.

    x is centred internally. Linear-bias correction so AC(0) = var.
    """
    x = x - x.mean()
    n = len(x)
    # Biased autocovariance via FFT (np.correlate is O(n^2))
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(X * np.conj(X), n=nfft)[:n] / n
    if ac[0] <= 0:
        return float("nan")
    rho = ac / ac[0]
    # First crossing of 1/e from above
    below = np.where(rho < np.exp(-1.0))[0]
    if len(below) == 0:
        return float("nan")
    k = below[0]
    if k == 0:
        return 0.0
    # Linear interpolate the crossing
    a, b = rho[k-1], rho[k]
    frac = (np.exp(-1.0) - a) / (b - a)
    return (k - 1 + frac) * dt


def main():
    seed_dirs = sorted(p for p in WORK_DIR.iterdir()
                       if p.is_dir() and p.name.startswith("p7v1_seed"))
    if not seed_dirs:
        raise SystemExit(
            f"No p7v1_seed* dirs under {WORK_DIR}. Run run_comparison.py first."
        )
    print(f"[brucon] reading DriftY from {len(seed_dirs)} seeds in {WORK_DIR}")

    # ---- assemble DriftY in the late intact window across all seeds ----
    drift_y_seeds = []
    t_window = None
    dt_sim = None
    for sd in seed_dirs:
        out = sd / f"{sd.name}.out"
        sim = parse_output(out)
        t = sim.columns["t"]
        # Intact-stats window: identical to run_comparison.py's intact_mask.
        mask = ((t >= FAILURE_TIME_S - INTACT_SAMPLE_S)
                & (t < FAILURE_TIME_S - 1.0))
        # Brucon outputs DriftY in kN, scale to N for direct comparison
        # with cqa (sec. 12.16 documents the kN/kN.m unit convention).
        dy = sim.columns["DriftY"][mask] * 1.0e3   # N
        drift_y_seeds.append(dy)
        if t_window is None:
            t_window = t[mask] - (FAILURE_TIME_S - INTACT_SAMPLE_S)
            dt_sim = float(np.median(np.diff(t[mask])))

    L_min = min(len(s) for s in drift_y_seeds)
    drift_y_arr = np.stack([s[:L_min] for s in drift_y_seeds])  # (n_seed, L)
    t_window = t_window[:L_min]
    print(f"  per-seed window: {L_min} samples at dt={dt_sim:.3f} s "
          f"= {L_min*dt_sim:.1f} s")

    # ---- per-seed empirical std and decorr time ----
    sigma_emp = drift_y_arr.std(axis=1, ddof=1)
    mean_emp = drift_y_arr.mean(axis=1)
    Tdec_emp = np.array([empirical_decorr_time(s, dt_sim)
                          for s in drift_y_arr])

    # ---- ensemble-averaged empirical PSD ----
    omega_emp = None
    S_emp_acc = None
    for s in drift_y_arr:
        s_centered = s - s.mean()
        omega, S = empirical_psd_one_sided(s_centered, dt_sim)
        if omega_emp is None:
            omega_emp = omega
            S_emp_acc = np.zeros_like(S)
        S_emp_acc += S
    S_emp = S_emp_acc / len(drift_y_arr)
    var_check_emp = float(np.trapezoid(S_emp, omega_emp))
    sigma_check_emp = np.sqrt(var_check_emp)

    # ---- cqa spectral slow-drift PSD ----
    rao = load_pdstrip_rao(PDSTRIP_PATH)
    # Use a dense omega grid covering the same band as the empirical PSD,
    # plus a low-frequency floor below the closed-loop knee.
    omega_cqa = np.geomspace(1e-3, max(omega_emp[-1], 6.0), 401)
    S_cqa_diag_3x3 = np.array([
        slow_drift_force_psd_newman_pdstrip(rao, Hs=HS, Tp=TP,
                                            theta_wave_rel=THETA_REL)(w)
        for w in omega_cqa
    ])  # (N, 3, 3)
    S_cqa_yy = S_cqa_diag_3x3[:, 1, 1]  # sway component
    var_cqa = float(np.trapezoid(S_cqa_yy, omega_cqa))
    sigma_cqa = np.sqrt(var_cqa)

    # ---- realise cqa time series and compute matching diagnostics ----
    T_real = float(L_min * dt_sim)
    n_t = L_min
    t_cqa = np.linspace(0.0, T_real, n_t)
    rng = np.random.default_rng(7)
    psd_func = slow_drift_force_psd_newman_pdstrip(
        rao, Hs=HS, Tp=TP, theta_wave_rel=THETA_REL,
    )
    F_cqa = realise_vector_force_time_series(
        S_F_funcs=[psd_func],
        omega_grid=omega_cqa,
        t=t_cqa,
        rng=rng,
    )
    drift_y_cqa = F_cqa[1]
    sigma_cqa_real = float(drift_y_cqa.std(ddof=1))
    Tdec_cqa_real = empirical_decorr_time(drift_y_cqa, dt_sim)
    omega_cqa_real, S_cqa_real = empirical_psd_one_sided(drift_y_cqa, dt_sim)

    # ---- console summary ----
    print("\n=== sway-drift slow-variation comparison @ P7 (V_w=14, Hs=4.20, Tp=10.22, beam) ===")
    print(f"brucon (n={len(drift_y_arr)} seeds, {T_real:.0f} s window):")
    print(f"  per-seed mean DriftY:  median = {np.median(mean_emp)/1e3:+.1f} kN, "
          f"range [{mean_emp.min()/1e3:+.1f}, {mean_emp.max()/1e3:+.1f}] kN")
    print(f"  per-seed std (DriftY): median = {np.median(sigma_emp)/1e3:.1f} kN, "
          f"range [{sigma_emp.min()/1e3:.1f}, {sigma_emp.max()/1e3:.1f}] kN")
    print(f"  per-seed T_decorr   : median = {np.nanmedian(Tdec_emp):.1f} s")
    print(f"  ensemble-avg Welch sigma_check = {sigma_check_emp/1e3:.1f} kN "
          f"(should be close to per-seed std)")
    print()
    print(f"cqa Newman+pdstrip (analytic):")
    print(f"  sqrt(int S_yy d_omega) = {sigma_cqa/1e3:.1f} kN")
    print(f"cqa realisation (matched n_t, dt):")
    print(f"  empirical std         = {sigma_cqa_real/1e3:.1f} kN")
    print(f"  empirical T_decorr    = {Tdec_cqa_real:.1f} s")
    print()
    print(f"ratios (cqa / brucon):")
    print(f"  sigma:       {sigma_cqa / np.median(sigma_emp):.2f}")
    print(f"  T_decorr:    {Tdec_cqa_real / np.nanmedian(Tdec_emp):.2f}")
    print(f"  sigma^2 * T: {(sigma_cqa**2 * Tdec_cqa_real) / (np.median(sigma_emp)**2 * np.nanmedian(Tdec_emp)):.2f}  "
          f"(low-freq quasi-DC variance budget after closed-loop filtering)")

    # ---- plot ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)

    # (a) one-sided PSD overlay
    ax = axes[0]
    ax.loglog(omega_emp, S_emp/1e6, label=f"brucon ensemble avg (n={len(drift_y_arr)})",
              lw=1.6, color="tab:red")
    ax.loglog(omega_cqa, S_cqa_yy/1e6,
              label="cqa Newman + pdstrip (analytic)", lw=1.6, color="tab:blue")
    ax.loglog(omega_cqa_real, S_cqa_real/1e6,
              label="cqa realisation (Welch, same dt/T)",
              lw=1.0, ls="--", color="tab:cyan")
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel(r"$S_{F_y F_y}(\omega)\;[\mathrm{kN}^2/(\mathrm{rad/s})]$")
    ax.set_title("Sway slow-drift force PSD (one-sided)")
    ax.set_xlim(1e-3, 5.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # (b) representative time series
    ax = axes[1]
    ax.plot(t_window, drift_y_arr[0]/1e3, color="tab:red", lw=1.0,
            label=f"brucon seed 0 (DriftY)")
    if len(drift_y_arr) > 5:
        for s in drift_y_arr[1:5]:
            ax.plot(t_window, s/1e3, color="tab:red", lw=0.5, alpha=0.4)
    ax.plot(t_cqa, drift_y_cqa/1e3, color="tab:blue", lw=1.0,
            label="cqa realisation seed 7")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$F_y$ [kN]")
    ax.set_title("Sway drift force time series (centred? brucon shows non-zero mean)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # (c) autocorrelation
    ax = axes[2]
    def normalised_ac(x):
        x = x - x.mean()
        n = len(x)
        nfft = 1 << int(np.ceil(np.log2(2*n)))
        X = np.fft.rfft(x, n=nfft)
        ac = np.fft.irfft(X * np.conj(X), n=nfft)[:n] / n
        return ac / ac[0]
    lags = np.arange(L_min) * dt_sim
    ac_emp_avg = np.mean([normalised_ac(s) for s in drift_y_arr], axis=0)
    ac_cqa = normalised_ac(drift_y_cqa)
    ax.plot(lags, ac_emp_avg, color="tab:red", lw=1.6, label="brucon ensemble-avg")
    ax.plot(lags, ac_cqa, color="tab:blue", lw=1.6, label="cqa realisation")
    ax.axhline(np.exp(-1.0), color="grey", ls=":", label=r"$1/e$")
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xlim(0.0, min(300.0, lags[-1]))
    ax.set_ylim(-0.5, 1.05)
    ax.set_xlabel("lag [s]")
    ax.set_ylabel(r"$\rho(\tau)$")
    ax.set_title(
        f"Sway-drift autocorrelation -- T_decorr brucon ~ {np.nanmedian(Tdec_emp):.0f} s, "
        f"cqa ~ {Tdec_cqa_real:.0f} s"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.savefig(OUT_PDF, dpi=110)
    print(f"\nwrote {OUT_PDF}")


if __name__ == "__main__":
    main()
