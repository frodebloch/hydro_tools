"""Plotting helpers for coupled-simulation results.

Uses matplotlib. All functions accept an optional ``ax`` to plot into so
multiple results can be overlaid by the caller.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .simulation import SimulationResults


def plot_roll_time_history(
    results: SimulationResults,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    **plot_kwargs,
) -> plt.Axes:
    """Plot roll angle (deg) vs. time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    ax.plot(results.t, results.phi_deg, label=label, **plot_kwargs)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("roll [deg]")
    ax.grid(True, alpha=0.3)
    if label is not None:
        ax.legend()
    return ax


def plot_moments(
    results: SimulationResults,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot wave moment and tank-on-vessel moment vs. time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    ax.plot(results.t, results.M_wave * 1e-6, label="M_wave [MN·m]")
    ax.plot(results.t, results.M_tank * 1e-6, label="M_tank [MN·m]")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("roll moment [MN·m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_tank_state(
    results: SimulationResults,
    tank_index: int = 0,
    component: int = 0,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    scale: float = 1.0,
    **plot_kwargs,
) -> plt.Axes:
    """Plot one component of one tank's state history."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    s = results.tank_states[tank_index][:, component] * scale
    ax.plot(results.t, s, label=label, **plot_kwargs)
    ax.set_xlabel("time [s]")
    ax.grid(True, alpha=0.3)
    if label is not None:
        ax.legend()
    return ax


def overlay_roll_histories(
    runs: Sequence[tuple[str, SimulationResults]],
    title: Optional[str] = None,
) -> plt.Figure:
    """Overlay multiple labelled roll histories in a single figure."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, res in runs:
        ax.plot(res.t, res.phi_deg, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("roll [deg]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
