"""Smoke test for cqa.rao: load csov pdstrip data and plot a few RAOs.

Generates ``csov_rao_smoke.png`` with magnitude curves for surge, sway,
heave, roll, pitch, yaw at three representative wave directions
(head, beam, following sea, in pdstrip convention 180°/90°/0°).

evaluate_rao() returns physical per-wave-amplitude values:
translations in m/m, rotations in rad/m. The wavenumber factor
k = omega^2 / g is applied internally for rotational DOFs (matches
brucon WaveResponse::CalculateLinearResponse).

Sanity checks (printed to stdout):

* head sea (180°): sway and yaw RAO magnitudes should be near zero
  (port/starboard symmetric); surge, heave, pitch should dominate.
* beam sea (90°): sway and roll should dominate; surge and pitch
  should be near zero.
* following sea (0°): mirror of head; sway/yaw small.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cqa"))

from cqa.rao import evaluate_rao, load_pdstrip_rao  # noqa: E402

PDSTRIP_PATH = Path.home() / "src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"
OUT_PATH = Path(__file__).with_name("csov_rao_smoke.png")


def main() -> int:
    table = load_pdstrip_rao(PDSTRIP_PATH, speed=0.0)
    print(f"Loaded {table.source_path}")
    print(f"  n_omega={table.n_omega}, omega range = [{table.omega[0]:.3f}, {table.omega[-1]:.3f}] rad/s")
    print(f"  n_beta={table.n_beta}, beta range  = [{table.beta_deg[0]:.1f}, {table.beta_deg[-1]:.1f}] deg")

    omega = np.linspace(0.2, 1.8, 200)
    dirs = {"head (180°)": 180.0, "beam from port (90°)": 90.0, "following (0°)": 0.0}

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    dof_names = ["surge [m/m]", "sway [m/m]", "heave [m/m]", "roll [rad/m]", "pitch [rad/m]", "yaw [rad/m]"]

    for label, beta in dirs.items():
        H = evaluate_rao(table, omega, beta)
        mag = np.abs(H)
        for k, ax in enumerate(axes.flat):
            ax.plot(omega, mag[:, k], label=label, lw=1.5)

    for k, ax in enumerate(axes.flat):
        ax.set_title(dof_names[k])
        ax.set_xlabel("omega [rad/s]")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"csov pdstrip RAOs at speed=0 (loaded {table.n_omega}×{table.n_beta} grid)")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=120)
    print(f"Wrote {OUT_PATH}")

    # Symmetry sanity numbers at omega=0.6 rad/s (~10 s wave).
    omega_probe = np.array([0.6])
    Hh = np.abs(evaluate_rao(table, omega_probe, 180.0))[0]
    Hb = np.abs(evaluate_rao(table, omega_probe, 90.0))[0]
    Hf = np.abs(evaluate_rao(table, omega_probe, 0.0))[0]
    print("\nMagnitudes at omega=0.6 rad/s [surge, sway, heave, roll, pitch, yaw]:")
    print(f"  head  (180°): {Hh}")
    print(f"  beam  ( 90°): {Hb}")
    print(f"  follo (  0°): {Hf}")
    print("\nExpected at head/following: |sway|, |yaw|, |roll| << |surge|, |heave|, |pitch|.")
    print("Expected at beam:            |surge|, |pitch|, |yaw| << |sway|, |heave|, |roll|.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
