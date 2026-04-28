"""GM sweep: bare-vessel roll response in beam seas at several GM values.

Demonstrates that the wave-moment back-out from pdstrip is GM-decoupled:
the same RAO file can be re-used at any GM by adjusting only the
hydrostatic stiffness ``c44 = rho * g * displacement * GM``.

Outputs:
  examples/output/csov_gm_sweep_time_history.png
  examples/output/csov_gm_sweep_steady_amplitudes.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from roll_reduction_tanks.coupling import CoupledSystem
from roll_reduction_tanks.pdstrip_io import load_csov
from roll_reduction_tanks.simulation import run_simulation
from roll_reduction_tanks.vessel import RollVessel, RollVesselConfig
from roll_reduction_tanks.waves import RegularWave, roll_moment_from_pdstrip

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "csov"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


def main():
    pd = load_csov(DATA / "csov_pdstrip.dat", DATA / "csov_pdstrip.inp")

    # Typical North Sea wind-sea period.
    wave = RegularWave(omega=2 * np.pi / 10.0, amplitude=1.0,
                       heading_deg=90.0, speed=0.0)
    M_wave = roll_moment_from_pdstrip(wave, pd)

    # Bracket realistic operational GM values (CSOV: 1.8 - 3.5 m typical).
    GM_values = [1.787, 2.20, 2.60, 3.00, 3.50]
    dt = 0.025
    t_end = 300.0
    n_tail = int(0.3 * t_end / dt)

    fig, ax = plt.subplots(figsize=(10, 5))
    print(f"{'GM [m]':>8s} {'T_n [s]':>10s} {'amp [deg]':>12s}")
    for GM in GM_values:
        cfg = RollVesselConfig(
            I44=pd.I44, a44=pd.a44_assumed, b44_lin=pd.b44_assumed,
            GM=GM, displacement=pd.displacement, rho=pd.rho, g=pd.g,
        )
        v = RollVessel(cfg)
        sys = CoupledSystem(v, tanks=[], M_wave_func=M_wave)
        res = run_simulation(sys, dt=dt, t_end=t_end)
        amp = float(np.max(np.abs(res.phi_deg[-n_tail:])))
        T_n = cfg.natural_period
        print(f"{GM:>8.3f} {T_n:>10.2f} {amp:>12.2f}")
        ax.plot(res.t, res.phi_deg,
                label=f"GM = {GM:.2f} m  (T_n = {T_n:.1f} s)")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("roll [deg]")
    ax.set_title(
        "CSOV bare-vessel roll, beam seas, T_w = 10 s, zeta_a = 1 m\n"
        "GM swept; pdstrip RAO back-out unchanged (GM-decoupling)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUT / "csov_gm_sweep_time_history.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
