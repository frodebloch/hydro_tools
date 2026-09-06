"""Estimator convergence figure for MTS DP Houston 2026 paper (Sec. 4.6).

Simulates a stationary AR(1) surrogate of the DP position-deviation
process with known true sigma and decorrelation time, feeds it into
BayesianSigmaEstimator with a sliding 10-min window and Bartlett ESS
correction, and plots the posterior sigma (median + 5/95 credible band)
as a function of elapsed time.

The point of the figure is to show, in operator-friendly terms, how
quickly the online estimate settles onto the true process sigma once
data starts flowing in.

Run:
    python scripts/run_estimator_convergence.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cqa import BayesianSigmaEstimator  # noqa: E402


def simulate_ar1(sigma_true: float, T_decorr_s: float, dt_s: float,
                 T_total_s: float, seed: int) -> np.ndarray:
    """Zero-mean AR(1) process with prescribed steady-state sigma and
    autocorrelation time. tau = -dt / ln(phi) so phi = exp(-dt/tau).
    """
    rng = np.random.default_rng(seed)
    n = int(np.round(T_total_s / dt_s))
    phi = float(np.exp(-dt_s / T_decorr_s))
    # Innovation std so that steady-state var(x) = sigma_true^2.
    eps_std = sigma_true * np.sqrt(1.0 - phi * phi)
    x = np.zeros(n, dtype=float)
    # Seed initial state at the stationary distribution so we do not
    # confound estimator warm-up with process warm-up.
    x[0] = rng.normal(0.0, sigma_true)
    eps = rng.normal(0.0, eps_std, size=n)
    for k in range(1, n):
        x[k] = phi * x[k - 1] + eps[k]
    return x


def run_estimator_track(x: np.ndarray, dt_s: float, window_s: float,
                        T_decorr_s: float, prior_sigma: float,
                        sample_every_s: float = 5.0):
    """Feed x[] into BayesianSigmaEstimator sample-by-sample and record
    posterior stats every sample_every_s of simulated time.
    """
    est = BayesianSigmaEstimator(
        prior_sigma2=float(prior_sigma) ** 2,
        T_decorr_s=float(T_decorr_s),
        dt_s=float(dt_s),
        window_s=float(window_s),
    )
    stride = max(1, int(round(sample_every_s / dt_s)))
    ts, med, lo, hi, n_eff = [], [], [], [], []
    for k, xk in enumerate(x):
        est.update(float(xk))
        if (k + 1) % stride == 0:
            post = est.posterior(credible=0.90)
            ts.append((k + 1) * dt_s)
            med.append(post.sigma_median)
            lo.append(post.sigma_lo)
            hi.append(post.sigma_hi)
            n_eff.append(post.n_eff)
    return (np.asarray(ts), np.asarray(med), np.asarray(lo),
            np.asarray(hi), np.asarray(n_eff))


def main() -> None:
    # Process settings. Values are illustrative of DP-class LF position
    # deviation on a mid-size vessel in a moderate sea state.
    sigma_true = 0.60          # m
    T_decorr_s = 45.0          # s (variance-estimator decorrelation time)
    dt_s = 0.5                 # s (typical DP model output rate)
    window_s = 600.0           # 10 min sliding window (Sec. 4.2)
    T_total_s = 1800.0         # 30 min run: warm-up + steady state
    prior_sigma = 0.90         # deliberately biased high to make warm-up visible
    seeds = [11, 23, 47]       # a few realisations to show run-to-run variability

    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)

    for i, seed in enumerate(seeds):
        x = simulate_ar1(sigma_true, T_decorr_s, dt_s, T_total_s, seed)
        ts, med, lo, hi, _ = run_estimator_track(
            x, dt_s, window_s, T_decorr_s, prior_sigma
        )
        ts_min = ts / 60.0
        label_med = 'posterior median' if i == 0 else None
        label_band = 'posterior 5-95% credible band' if i == 0 else None
        ax.fill_between(ts_min, lo, hi, color='C0', alpha=0.18,
                        linewidth=0, label=label_band)
        ax.plot(ts_min, med, color='C0', lw=1.4, alpha=0.85,
                label=label_med)

    ax.axhline(sigma_true, color='k', ls='--', lw=1.1,
               label=f'true \u03c3 = {sigma_true:.2f} m')
    ax.axhline(prior_sigma, color='0.55', ls=':', lw=1.0,
               label=f'prior \u03c3 = {prior_sigma:.2f} m')
    # Mark window-fill time: once t >= window_s the sliding window is
    # fully populated and the estimator is no longer padding with prior.
    ax.axvline(window_s / 60.0, color='0.4', ls='-.', lw=0.9)
    ax.text(window_s / 60.0 + 0.15, ax.get_ylim()[1] * 0.98 if False else prior_sigma + 0.02,
            'window full', fontsize=8, color='0.35', va='bottom')

    ax.set_xlabel('elapsed time [min]')
    ax.set_ylabel('posterior \u03c3 of position deviation [m]')
    ax.set_title('Online estimator convergence '
                 '(10 min sliding window, three realisations)')
    ax.set_xlim(0.0, T_total_s / 60.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    out_dir = Path.home() / 'Documents' / 'MTS_26' / 'images'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'estimator_convergence.png'
    fig.savefig(out_path, dpi=200)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
