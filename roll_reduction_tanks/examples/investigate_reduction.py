"""Investigate roll reduction sensitivity (post-Holden-revision in utube_open.py).

Originally written to diagnose why the headline reduction was only ~22 %.
That diagnosis succeeded: the cross-coupling formula in
:mod:`tanks.utube_open` was reconciled with Holden, Perez & Fossen (2011)
(Lagrangian derivation, validated against 44 model-scale experiments).
With the corrected algebra and a sensible placement, the headline
at-resonance reduction is in the literature range for properly tuned
passive U-tubes.

This script remains useful as a parameter-sensitivity tool. Four
experiments:

  0.  Vertical placement sweep. With ``a_phi = Q (z_d + h_0)`` (Holden
      eq. 14, ``z_d = duct_below_waterline``, z-down WL-referenced), the
      kinematic cross-coupling vanishes at ``z_d = -h_0`` (duct one
      fluid-depth *above* the waterline) and changes sign across that
      height. Both extreme-low (``z_d >> 0``, near keel) and
      extreme-high (``z_d << -h_0``, deep in the superstructure)
      placements give large ``|a_phi|``; the relative phase of
      ``+a_phi*phi_ddot`` and ``+c_phi*phi`` on the tank RHS sets which
      placements convert that into useful absorber action.

  1.  Tank size sweep. At each tank thickness the duct height is
      retuned so ``T_tau == T_n``. Damping is set to the Den Hartog
      optimum

          zeta_opt = sqrt( 3 mu / (8 (1+mu)^3) )

      where ``mu = a_phi^2 / (a_tau (I44+a44))``. Caveat: Den Hartog's
      formula assumes a single cross-coupling term; the U-tube has two
      (``+a_phi phi_ddot`` AND ``+c_phi phi`` on the tank RHS), so the
      optimum is approximate. Empirical optimum from experiment 2 is
      typically slightly different.

  2.  Damping sweep at fixed thickness. Compares the empirical optimum
      to the Den Hartog prediction.

  3.  Hull-damping sensitivity. TMDs work best against lightly damped
      primary systems; halving the assumed bare-hull damping
      dramatically increases the at-resonance reduction.

Outputs to ``examples/output/investigate_reduction.png``.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from roll_reduction_tanks.coupling import CoupledSystem
from roll_reduction_tanks.pdstrip_io import load_csov
from roll_reduction_tanks.simulation import run_simulation
from roll_reduction_tanks.tanks.utube_open import OpenUtubeConfig, OpenUtubeTank
from roll_reduction_tanks.vessel import RollVessel, RollVesselConfig
from roll_reduction_tanks.waves import RegularWave, roll_moment_from_pdstrip

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "csov"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


# ----------------------------------------------------------------------- helpers


def make_vessel(pd, GM=3.0):
    cfg = RollVesselConfig(
        I44=pd.I44, a44=pd.a44_assumed, b44_lin=pd.b44_assumed,
        GM=GM, displacement=pd.displacement, rho=pd.rho, g=pd.g,
    )
    return RollVessel(cfg)


def base_tank_cfg(thickness=5.0, mu=0.05, h_d=0.6, z_d=6.5) -> OpenUtubeConfig:
    """Default configuration. ``z_d`` is ``duct_below_waterline``,
    z-down (positive = duct below WL).

    Default 6.5 m matches the headline ``csov_passive_utube.py`` example
    (duct at keel level for the CSOV at draft T = 6.5 m). Raising the
    tank reduces ``z_d``; the kinematic cross-coupling has a sign change
    at ``z_d = -h_0`` (one fluid-depth above the WL).
    """
    return OpenUtubeConfig(
        duct_below_waterline=z_d,
        undisturbed_fluid_height=2.5,
        utube_duct_height=h_d,
        resevoir_duct_width=2.0,
        utube_duct_width=16.0,
        tank_thickness=thickness,
        tank_to_xcog=0.0,
        tank_wall_friction_coef=mu,
        tank_height=5.0,
    )


def fluid_mass(cfg: OpenUtubeConfig) -> float:
    """Total fluid mass: 2 reservoirs (h_0 deep) + duct."""
    duct_volume = cfg.utube_duct_width * cfg.tank_thickness * cfg.utube_duct_height
    res_volume = (
        2.0 * cfg.resevoir_duct_width * cfg.tank_thickness * cfg.undisturbed_fluid_height
    )
    return cfg.rho * (duct_volume + res_volume)


def tune_h_d_to_period(cfg: OpenUtubeConfig, target_period: float, tol=1e-3) -> float:
    """Bisect on h_d so the tank natural period matches `target_period`.

    From the tank EOM,
        omega_n^2 = c_tau / a_tau
                  = g / [ h_0 + W*b_r/(2 h_d) ]
    so larger h_d -> larger omega -> shorter period; we search h_d in
    [0.05, 5.0] m.
    """
    target_omega = 2 * np.pi / target_period

    def omega_of(h_d):
        return OpenUtubeTank(replace(cfg, utube_duct_height=h_d)).natural_frequency

    lo, hi = 0.05, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if omega_of(mid) < target_omega:  # mid too small => omega too low
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def steady_amplitude(res, frac=0.3):
    n = int(frac * len(res.t))
    return float(np.max(np.abs(res.phi_deg[-n:])))


def simulate_amp(pd, vessel_GM, tank_cfg, T_w, t_end=300.0, dt=0.025):
    vessel = make_vessel(pd, GM=vessel_GM)
    tank = OpenUtubeTank(tank_cfg) if tank_cfg is not None else None
    wave = RegularWave(omega=2 * np.pi / T_w, amplitude=1.0, heading_deg=90.0, speed=0.0)
    M_wave = roll_moment_from_pdstrip(wave, pd)
    tanks = [tank] if tank else []
    sys = CoupledSystem(vessel, tanks=tanks, M_wave_func=M_wave)
    res = run_simulation(sys, dt=dt, t_end=t_end)
    return steady_amplitude(res), tank


def den_hartog_opt(mu_inertia: float) -> tuple[float, float]:
    """Den Hartog optimum tuning ratio and damping ratio for given mu."""
    f_opt = 1.0 / (1.0 + mu_inertia)
    zeta_opt = np.sqrt(3.0 * mu_inertia / (8.0 * (1.0 + mu_inertia) ** 3))
    return f_opt, zeta_opt


# ----------------------------------------------------------------------- experiments


def main():
    pd = load_csov(DATA / "csov_pdstrip.dat", DATA / "csov_pdstrip.inp")
    vessel = make_vessel(pd, GM=3.0)
    T_n = vessel.config.natural_period
    omega_n = vessel.config.natural_frequency
    M_eff = vessel.config.total_inertia

    print(f"Vessel: T_n = {T_n:.3f} s, omega_n = {omega_n:.3f} rad/s, "
          f"I44+a44 = {M_eff:.3e} kg.m^2, "
          f"bare-hull zeta = {vessel.config.damping_ratio:.4f}")

    # ----- bare vessel headline: amplitude at exact resonance ----------
    amp_bare, _ = simulate_amp(pd, 3.0, None, T_w=T_n)
    print(f"\nBare vessel @ T_n: amplitude = {amp_bare:.3f} deg")

    # =========================================================== experiment 0
    # Vertical placement sweep. Keep size fixed (thk=5), retune h_d at each
    # placement (a_phi changes nothing for self-coefficients but the friction
    # is fixed at user-trial mu=0.05 to match earlier examples).
    print("\n--- experiment 0: vertical placement sweep (mu=0.05, thk=5 m) ---")
    print("    z_d = duct_below_waterline (positive = below WL; CSOV keel at +6.5 m)")
    z_d_grid = np.array([-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 6.5])
    placement_rows = []
    for z_d in z_d_grid:
        cfg = base_tank_cfg(thickness=5.0, mu=0.05, z_d=z_d)
        h_d_t = tune_h_d_to_period(cfg, T_n)
        cfg = replace(cfg, utube_duct_height=h_d_t)
        amp, tnk = simulate_amp(pd, 3.0, cfg, T_w=T_n)
        red = 100*(1 - amp/amp_bare)
        lever = z_d + cfg.undisturbed_fluid_height
        placement_rows.append(dict(z_d=z_d, lever=lever, a_phi=tnk.a_phi,
                                    amp=amp, red=red))
        print(f"  z_d={z_d:+5.1f} m | (z_d+h0)={lever:+5.1f} m | "
              f"a_phi={tnk.a_phi:+.3e} | amp={amp:.3f} deg | "
              f"reduction = {red:5.1f} %")

    # =========================================================== experiment 1
    # Sweep tank size (thickness). At each point: re-tune h_d to match T_n,
    # find inertia ratio mu, set mu to Den Hartog optimum, run sim.

    thicknesses = np.array([2.5, 5.0, 7.5, 10.0, 12.5, 15.0])
    rows = []
    for thk in thicknesses:
        cfg = base_tank_cfg(thickness=thk, mu=0.05, h_d=0.6)
        h_d_tuned = tune_h_d_to_period(cfg, T_n)
        cfg = replace(cfg, utube_duct_height=h_d_tuned)
        m_fluid = fluid_mass(cfg)

        # Inertia ratio for Den Hartog: a_phi^2 / (a_tau * (I44+a44))
        tnk_probe = OpenUtubeTank(cfg)
        mu_inertia = tnk_probe.a_phi ** 2 / (tnk_probe.a_tau * M_eff)

        f_opt, zeta_opt = den_hartog_opt(mu_inertia)

        # Convert zeta_opt -> tank_wall_friction_coef.
        # b_tau = Q * mu_friction * res * (...)  and  zeta = b_tau / (2*sqrt(c_tau*a_tau))
        # So mu_friction = zeta_opt * 2*sqrt(c_tau*a_tau) / [Q*res*(...)]
        c = cfg
        bracket_b = (
            (c.utube_duct_width + c.resevoir_duct_width)
            / (2.0 * c.utube_duct_height ** 2)
            + c.undisturbed_fluid_height / c.resevoir_duct_width
        )
        b_tau_opt = zeta_opt * 2.0 * np.sqrt(tnk_probe.c_tau * tnk_probe.a_tau)
        mu_friction = b_tau_opt / (tnk_probe.Q * c.resevoir_duct_width * bracket_b)

        cfg_opt = replace(cfg, tank_wall_friction_coef=mu_friction)

        amp_tnk, tnk = simulate_amp(pd, 3.0, cfg_opt, T_w=T_n)
        red = 100.0 * (1 - amp_tnk / amp_bare)
        rows.append(dict(
            thk=thk, m_fluid=m_fluid,
            m_ratio=m_fluid / pd.displacement / pd.rho,
            mu_inertia=mu_inertia, zeta_opt=zeta_opt, mu_friction=mu_friction,
            h_d=h_d_tuned, amp=amp_tnk, red=red,
            T_tnk=tnk.natural_period, zeta_actual=tnk.damping_ratio,
        ))
        print(
            f"thk={thk:5.1f} m | m_fluid={m_fluid/1e3:6.1f} t "
            f"({100*m_fluid/(pd.displacement*pd.rho):4.2f}% disp) | "
            f"mu_I={mu_inertia:.4f} | zeta_opt={zeta_opt:.3f} -> "
            f"mu_fric={mu_friction:.3f} | "
            f"amp={amp_tnk:.3f} deg | reduction = {red:5.1f} %"
        )

    # =========================================================== experiment 2
    # At default size (thk=5), sweep friction coefficient. Find the empirical
    # optimum and compare to Den Hartog prediction.

    print("\n--- experiment 2: friction sweep at thickness = 5 m ---")
    cfg5 = base_tank_cfg(thickness=5.0)
    h_d_5 = tune_h_d_to_period(cfg5, T_n)
    cfg5 = replace(cfg5, utube_duct_height=h_d_5)
    tnk5 = OpenUtubeTank(cfg5)
    mu_I_5 = tnk5.a_phi ** 2 / (tnk5.a_tau * M_eff)
    _, zeta_opt_5 = den_hartog_opt(mu_I_5)
    print(f"mu_inertia = {mu_I_5:.4f}, Den Hartog zeta_opt = {zeta_opt_5:.3f}")

    mu_grid = np.array([0.005, 0.01, 0.02, 0.04, 0.06, 0.10, 0.15, 0.25, 0.40])
    sweep_rows = []
    for mu_f in mu_grid:
        cfg_i = replace(cfg5, tank_wall_friction_coef=mu_f)
        amp_i, tnk_i = simulate_amp(pd, 3.0, cfg_i, T_w=T_n)
        sweep_rows.append(dict(mu=mu_f, zeta=tnk_i.damping_ratio,
                               amp=amp_i, red=100*(1-amp_i/amp_bare)))
        print(f"  mu_friction={mu_f:5.3f} | zeta={tnk_i.damping_ratio:.3f} | "
              f"amp={amp_i:.3f} deg | reduction = {sweep_rows[-1]['red']:5.1f}%")

    best = min(sweep_rows, key=lambda r: r["amp"])
    print(f"\nEmpirical optimum: mu_friction={best['mu']:.3f} (zeta={best['zeta']:.3f}), "
          f"reduction = {best['red']:.1f} %")
    print(f"Den Hartog predicts: zeta_opt = {zeta_opt_5:.3f}")

    # =========================================================== experiment 3
    # Bare-hull damping sensitivity. Halve b44_lin and re-do experiment 1 row at thk=10.

    print("\n--- experiment 3: hull damping sensitivity (thk=10 m, optimal tank) ---")
    for damp_factor in (1.0, 0.5, 0.25):
        pd_copy = pd
        cfg10 = base_tank_cfg(thickness=10.0)
        h_d_10 = tune_h_d_to_period(cfg10, T_n)
        cfg10 = replace(cfg10, utube_duct_height=h_d_10)
        tnk10 = OpenUtubeTank(cfg10)
        mu_I = tnk10.a_phi ** 2 / (tnk10.a_tau * M_eff)
        _, zeta_o = den_hartog_opt(mu_I)
        c = cfg10
        bracket_b = (
            (c.utube_duct_width + c.resevoir_duct_width) / (2.0 * c.utube_duct_height ** 2)
            + c.undisturbed_fluid_height / c.resevoir_duct_width
        )
        b_tau_opt = zeta_o * 2.0 * np.sqrt(tnk10.c_tau * tnk10.a_tau)
        mu_fric = b_tau_opt / (tnk10.Q * c.resevoir_duct_width * bracket_b)
        cfg10 = replace(cfg10, tank_wall_friction_coef=mu_fric)

        # Modified vessel
        cfg_v = RollVesselConfig(
            I44=pd.I44, a44=pd.a44_assumed,
            b44_lin=pd.b44_assumed * damp_factor,
            GM=3.0, displacement=pd.displacement, rho=pd.rho, g=pd.g,
        )
        v = RollVessel(cfg_v)
        wave = RegularWave(omega=omega_n, amplitude=1.0, heading_deg=90.0, speed=0.0)
        M_w = roll_moment_from_pdstrip(wave, pd)
        sys_b = CoupledSystem(v, tanks=[], M_wave_func=M_w)
        res_b = run_simulation(sys_b, dt=0.025, t_end=400.0)
        ab = steady_amplitude(res_b)

        v2 = RollVessel(cfg_v)
        sys_t = CoupledSystem(v2, tanks=[OpenUtubeTank(cfg10)], M_wave_func=M_w)
        res_t = run_simulation(sys_t, dt=0.025, t_end=400.0)
        at = steady_amplitude(res_t)
        zeta_h = cfg_v.b44_lin / (2 * np.sqrt(cfg_v.c44 * cfg_v.total_inertia))
        print(f"hull zeta={zeta_h:.4f} | bare amp={ab:.3f} deg | with tank={at:.3f} deg | "
              f"reduction = {100*(1-at/ab):5.1f} %")

    # ----- plots -------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))

    ax = axs[0]
    z_arr = np.array([r["z_d"] for r in placement_rows])
    rd_arr = np.array([r["red"] for r in placement_rows])
    ax.plot(z_arr, rd_arr, "o-", color="C2")
    ax.set_xlabel("duct_below_waterline [m] (positive = below WL)")
    ax.set_ylabel("Roll-amplitude reduction at resonance [%]")
    ax.set_title("Experiment 0: vertical placement\n(thk=5 m, mu=0.05, retuned h_d)")
    ax.grid(alpha=0.3)
    ax.axvline(6.5, ls=":", color="g", alpha=0.6, label="keel (CSOV T=6.5 m)")
    ax.axvline(-2.5, ls=":", color="r", alpha=0.6, label="dead zone z_d=-h_0")
    ax.legend(fontsize=8)

    thk_arr = np.array([r["thk"] for r in rows])
    red_arr = np.array([r["red"] for r in rows])
    mr_arr = np.array([r["m_ratio"] for r in rows]) * 100  # percent of disp
    muI_arr = np.array([r["mu_inertia"] for r in rows])

    ax = axs[1]
    ax.plot(mr_arr, red_arr, "o-", color="C0")
    ax.set_xlabel("Tank fluid mass / vessel displacement [%]")
    ax.set_ylabel("Roll-amplitude reduction at resonance [%]")
    ax.set_title("Experiment 1: tank size at Den Hartog optimum\n(z_d = 6.5 m (keel), retuned h_d each step)")
    ax.grid(alpha=0.3)
    for x, y, t in zip(mr_arr, red_arr, thk_arr):
        ax.annotate(f"{t:.1f} m", (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)

    ax = axs[2]
    mu_a = np.array([r["mu"] for r in sweep_rows])
    z_a = np.array([r["zeta"] for r in sweep_rows])
    red_a = np.array([r["red"] for r in sweep_rows])
    ax.plot(z_a, red_a, "o-", color="C1", label="simulated")
    ax.axvline(zeta_opt_5, ls=":", color="k", label=f"Den Hartog opt zeta = {zeta_opt_5:.3f}")
    ax.set_xlabel("Tank damping ratio zeta")
    ax.set_ylabel("Roll-amplitude reduction [%]")
    ax.set_title(f"Experiment 2: friction sweep\n(thk=5 m, mu_I={mu_I_5:.4f}, z_d=6.5 m)")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out = OUT / "investigate_reduction.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
