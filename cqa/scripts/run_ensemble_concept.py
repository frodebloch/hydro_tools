"""Conceptual illustration of the ensemble evaluator for the post-WCF
motion prediction (MTS DP Houston 2026 paper, Sec. 4.3).

This figure is deliberately conceptual — no realistic vessel model,
no true units. It shows N post-WCF trajectories of |eta_xy(t)| starting
from a spread of initial conditions drawn from the pre-WCF buffer,
with a common WCF event at t = 0, overlaid with the P50 and P95
envelopes of the ensemble and their peak markers.

Run:
    python scripts/run_ensemble_concept.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def synth_trajectory(rng: np.random.Generator, t: np.ndarray,
                     eta0: float, v0: float, peak_scale: float,
                     peak_time: float, tail_scale: float) -> np.ndarray:
    """Synthesise a plausible-looking post-WCF |eta_xy(t)| trajectory.

    Purely for illustration: a critically-damped-ish rise to a peak
    around ``peak_time``, then a slow decay to a long-term level, with
    a small stochastic wave-frequency ripple superimposed.
    """
    # Deterministic envelope: quick rise then relaxation to tail level.
    rise = 1.0 - np.exp(-t / (0.35 * peak_time))
    decay = np.exp(-(t - peak_time) / (2.5 * peak_time))
    decay = np.where(t < peak_time, 1.0, decay)
    env = eta0 + v0 * t * np.exp(-t / peak_time) \
        + peak_scale * rise * decay \
        + tail_scale * (1.0 - np.exp(-t / (4.0 * peak_time)))
    # Wave-frequency ripple.
    freq = rng.uniform(0.35, 0.65)   # cycles per sec of synthetic time
    phase = rng.uniform(0.0, 2 * np.pi)
    ripple = 0.06 * peak_scale * np.sin(2 * np.pi * freq * t + phase)
    ripple *= (1.0 - np.exp(-t / (0.3 * peak_time)))  # taper in
    return np.maximum(env + ripple, 0.0)


def main() -> None:
    rng = np.random.default_rng(7)
    n_traj = 24
    t = np.linspace(0.0, 120.0, 601)   # 120 s post-WCF window

    trajectories = np.zeros((n_traj, t.size), dtype=float)
    for i in range(n_traj):
        eta0 = rng.normal(0.35, 0.12)       # initial deviation
        v0 = rng.normal(0.0, 0.015)         # initial rate
        peak_scale = rng.normal(1.0, 0.22)  # per-trajectory peak scaling
        peak_time = rng.normal(18.0, 3.0)   # s to peak
        tail_scale = rng.normal(0.35, 0.10) # long-term settling level
        trajectories[i] = synth_trajectory(
            rng, t, eta0, v0, max(peak_scale, 0.3),
            max(peak_time, 8.0), tail_scale
        )

    # Envelopes across the ensemble at each t.
    p50 = np.quantile(trajectories, 0.50, axis=0)
    p95 = np.quantile(trajectories, 0.95, axis=0)

    # Peak-of-trajectory statistics (what the panel actually reports).
    peaks = trajectories.max(axis=1)
    peak_p50 = float(np.quantile(peaks, 0.50))
    peak_p95 = float(np.quantile(peaks, 0.95))

    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)

    # Individual trajectories, thin and pale.
    for i in range(n_traj):
        ax.plot(t, trajectories[i], color='0.55', lw=0.6, alpha=0.55)
    # Add one darker representative for legend.
    ax.plot([], [], color='0.55', lw=0.9,
            label=f'ensemble members ({n_traj} initial conditions)')

    # Envelopes.
    ax.plot(t, p50, color='C0', lw=2.0, label='ensemble $P_{50}$ of $|\\eta_{xy}(t)|$')
    ax.plot(t, p95, color='C3', lw=2.0, label='ensemble $P_{95}$ of $|\\eta_{xy}(t)|$')

    # Peak markers on the right side.
    ax.axhline(peak_p50, color='C0', ls=':', lw=1.0, alpha=0.7)
    ax.axhline(peak_p95, color='C3', ls=':', lw=1.0, alpha=0.7)
    ax.text(t[-1] * 0.995, peak_p50, ' $P_{50}$ peak',
            color='C0', va='center', ha='right',
            fontsize=9, backgroundcolor='white')
    ax.text(t[-1] * 0.995, peak_p95, ' $P_{95}$ peak',
            color='C3', va='center', ha='right',
            fontsize=9, backgroundcolor='white')

    # WCF marker at t = 0.
    ax.axvline(0.0, color='k', ls='--', lw=1.0, alpha=0.7)
    ax.text(1.5, ax.get_ylim()[1] * 0.02, 'WCF', fontsize=9,
            color='k', va='bottom')

    ax.set_xlabel('time after WCF [s]')
    ax.set_ylabel('$|\\eta_{xy}(t)|$  [conceptual units]')
    ax.set_title('Ensemble post-WCF motion prediction (concept)')
    ax.set_xlim(0.0, t[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_yticks([])   # concept figure: hide numeric ticks on y
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    out_dir = Path.home() / 'Documents' / 'MTS_26' / 'images'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'ensemble_concept.png'
    fig.savefig(out_path, dpi=200)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
