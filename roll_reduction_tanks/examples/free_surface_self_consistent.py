"""Self-consistent free-surface tank tuning vs naive (bare-vessel) tuning.

A free-surface (Lloyd) tank's static surface-tilt stiffness ``dc44_extra``
acts on the hull as ``+dc44_extra * phi`` and therefore reduces the
effective roll restoring stiffness from ``c44`` to ``c44 - dc44_extra``.
The vessel's effective natural frequency drops correspondingly. Tuning
the tank to the *bare* vessel period leaves it mistuned in operation;
the apparent reduction is then partly an artefact of the vessel
detuning away from the wave excitation, rather than honest absorber
action.

This example contrasts the two designs across a width sweep, with the
along-beam length pinned at ``L = beam`` and the depth ``h`` chosen to
match the relevant target frequency:

  * Naive tuning:        ``omega_n_tank = omega_bare``  (fixed h ~ 1.6 m).
  * Self-consistent:     ``omega_n_tank = omega_eff``   (h decreases with W).

For the CSOV at GM = 3.0 m:
  - the bare vessel period is ``T_n = 11.4 s``;
  - widths in the Faltinsen 1990 §5.4 GM-loss range
    ``Delta_GM/GM in [0.15, 0.30]`` are roughly W in [5.5, 11] m;
  - self-consistent depths land in ``h ~ 1.0-1.5 m``;
  - effective vessel periods land in ``T_eff ~ 12.0-13.5 s``.

Output:
  examples/output/free_surface_self_consistent.png
  Console table of W, h_naive, h_self, T_eff, GM-loss ratio, reductions.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from roll_reduction_tanks.coupling import CoupledSystem
from roll_reduction_tanks.pdstrip_io import load_csov
from roll_reduction_tanks.simulation import run_simulation
from roll_reduction_tanks.tanks.free_surface import (
    FreeSurfaceConfig, FreeSurfaceTank, _depth_for_omega, tune_self_consistent,
)
from roll_reduction_tanks.vessel import RollVessel, RollVesselConfig
from roll_reduction_tanks.waves import RegularWave, roll_moment_from_pdstrip

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "csov"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

BEAM = 22.4   # m, geometric maximum tank length
GM = 3.0      # m


def build_vessel(pd):
    return RollVessel(RollVesselConfig(
        I44=pd.I44, a44=pd.a44_assumed, b44_lin=pd.b44_assumed,
        GM=GM, displacement=pd.displacement, rho=pd.rho, g=pd.g,
    ))


def main():
    pd = load_csov(DATA / "csov_pdstrip.dat", DATA / "csov_pdstrip.inp")
    c44 = pd.rho * pd.g * pd.displacement * GM
    I_tot = pd.I44 + pd.a44_assumed
    omega_bare = float(np.sqrt(c44 / I_tot))
    T_bare = 2 * np.pi / omega_bare
    print(f"bare vessel: T_n = {T_bare:.3f} s")

    # Wave excitation at the bare-vessel period (the historical operating
    # point used in compare_tank_types.py).
    wave = RegularWave(omega=omega_bare, amplitude=1.0,
                       heading_deg=90.0, speed=0.0)
    M_wave = roll_moment_from_pdstrip(wave, pd)

    dt, t_end = 0.025, 300.0
    n_tail = int(0.3 * t_end / dt)

    # Bare-vessel reference.
    sys0 = CoupledSystem(build_vessel(pd), tanks=[], M_wave_func=M_wave)
    res0 = run_simulation(sys0, dt=dt, t_end=t_end)
    amp_bare = float(np.max(np.abs(res0.phi_deg[-n_tail:])))
    print(f"bare amp = {amp_bare:.3f} deg")

    widths = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    h_naive = np.full_like(widths, _depth_for_omega(omega_bare, BEAM, pd.g))
    rows = []  # (W, h_naive, h_self, T_eff, GM_loss/GM, amp_naive, amp_self)

    print()
    print(f"{'W':>5} {'h_naive':>8} {'h_self':>8} {'T_eff':>7} "
          f"{'GMloss/GM':>10} {'amp_naive':>10} {'amp_self':>10}")
    for W, h_n in zip(widths, h_naive):
        # Naive: tank tuned to bare period (h = h_naive). Even though the
        # vessel still detunes in operation, the *tank* is left at the
        # naive frequency.
        cfg_naive = FreeSurfaceConfig(
            length=BEAM, width=float(W), fluid_depth=float(h_n),
            z_tank=8.5, z_cog=2.5, damping_ratio=0.10, warn_fill_ratio=10.0,
        )
        tank_naive = FreeSurfaceTank(cfg_naive)
        sys = CoupledSystem(build_vessel(pd), tanks=[tank_naive], M_wave_func=M_wave)
        res = run_simulation(sys, dt=dt, t_end=t_end)
        amp_naive = float(np.max(np.abs(res.phi_deg[-n_tail:])))

        # Self-consistent: depth iterated so omega_n_tank == omega_eff.
        cfg_self, info = tune_self_consistent(
            length=BEAM, width=float(W),
            z_tank=8.5, z_cog=2.5, damping_ratio=0.10,
            vessel_c44=c44, vessel_inertia_total=I_tot,
            warn_fill_ratio=10.0,
        )
        tank_self = FreeSurfaceTank(cfg_self)
        sys = CoupledSystem(build_vessel(pd), tanks=[tank_self], M_wave_func=M_wave)
        res = run_simulation(sys, dt=dt, t_end=t_end)
        amp_self = float(np.max(np.abs(res.phi_deg[-n_tail:])))
        T_eff = 2 * np.pi / info["omega_eff"]

        rows.append((W, h_n, info["fluid_depth"], T_eff,
                     info["gm_loss_ratio"], amp_naive, amp_self))
        print(f"{W:5.1f} {h_n:8.3f} {info['fluid_depth']:8.3f} {T_eff:7.2f} "
              f"{info['gm_loss_ratio']:10.3f} {amp_naive:10.3f} {amp_self:10.3f}")

    rows = np.array(rows)
    W_arr = rows[:, 0]
    amp_n = rows[:, 5]
    amp_s = rows[:, 6]
    red_n = (amp_bare - amp_n) / amp_bare * 100.0
    red_s = (amp_bare - amp_s) / amp_bare * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(W_arr, amp_n, "o-", label="naive (tank tuned to bare T_n)")
    ax.plot(W_arr, amp_s, "s-", label="self-consistent (tank tuned to T_eff)")
    ax.axhline(amp_bare, color="k", linestyle=":", alpha=0.5,
               label=f"bare = {amp_bare:.2f} deg")
    ax.set_xlabel("tank width W [m]")
    ax.set_ylabel("steady-state roll amplitude [deg]")
    ax.set_title("Free-surface tank, CSOV beam seas T = 11.4 s, GM = 3.0 m")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ax.plot(W_arr, red_n, "o-", label="naive")
    ax.plot(W_arr, red_s, "s-", label="self-consistent")
    ax.set_xlabel("tank width W [m]")
    ax.set_ylabel("roll reduction [%]")
    ax.set_title("Reduction vs bare vessel")
    # Annotate Faltinsen GM-loss range.
    gm_loss = rows[:, 4]
    in_band = (gm_loss >= 0.15) & (gm_loss <= 0.30)
    if in_band.any():
        W_lo = float(W_arr[in_band].min())
        W_hi = float(W_arr[in_band].max())
        ax.axvspan(W_lo, W_hi, color="green", alpha=0.10,
                   label=f"Faltinsen GM-loss in [0.15, 0.30]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    out = OUT / "free_surface_self_consistent.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
