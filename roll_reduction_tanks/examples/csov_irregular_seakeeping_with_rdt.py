"""Irregular-seas comparison including the active RDT-driven U-tube.

Extends ``csov_irregular_seakeeping.py`` with two RDT-active tank
designs:

  * **Ideal active (inverse-dynamics)**: assumes perfect knowledge of
    M_wave(t) and inverts the coupled dynamics to produce a thrust
    that exactly cancels the wave moment subject to actuator
    amplitude saturation (F_max = 200 kN, the bow-tunnel-thruster
    class identified in README sec. 4.y).
  * **State-feedback PD**: pure rate damping on phi_dot, no wave
    knowledge -- the honest signal-only baseline.

The same JONSWAP realisation (Hs = 3 m, Tp = 8.5 s, beam seas) is run
through bare vessel + the standard passive devices + both RDT-active
configurations. Reports phi_1/3 = 4*sigma_phi and the time-averaged
mechanical shaft power |F_RDT * v_duct| demanded from the RDT drive.
Note this is the *reactive* power swing the drive must source/sink, not
the dissipated power -- most of it cycles in and out of the fluid
kinetic energy each half-period (see README sec. 4.y, RDT sizing).

Output:
  examples/output/csov_irregular_seakeeping_with_rdt.png
"""
from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

from roll_reduction_tanks.controllers.constant import FullyOpenValve
from roll_reduction_tanks.coupling import CoupledSystem
from roll_reduction_tanks.pdstrip_io import load_csov
from roll_reduction_tanks.simulation import run_simulation
from roll_reduction_tanks.tanks.free_surface import (
    FreeSurfaceTank, tune_self_consistent,
)
from roll_reduction_tanks.tanks.tuned_mass_damper import (
    TunedMassDamperConfig, TunedMassDamperTank, den_hartog_optimal,
)
from roll_reduction_tanks.tanks.utube_open import (
    OpenUtubeConfig, OpenUtubeTank,
)
from roll_reduction_tanks.tanks.utube_rdt import (
    RDTUtubeConfig, RDTUtubeTank,
    InverseDynamicsRDTController, StateFeedbackRDTController,
    ResonatorObserverRDTController,
)
from roll_reduction_tanks.controllers.luenberger_wave_observer import (
    LuenbergerWaveObserver, LuenbergerWaveObserverConfig,
)
from roll_reduction_tanks.vessel import RollVessel, RollVesselConfig
from roll_reduction_tanks.waves import (
    IrregularWave, roll_moment_from_irregular,
)

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "csov"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

GM = 3.0

# -------------------------------------------------------- seaway

HS = 3.0
TP = 8.5
GAMMA = 3.3
HEADING = 90.0
SEED = 20260428

T_WARMUP = 60.0
T_SIM = 600.0
DT = 0.05

# -------------------------------------------------------- factories


def _vessel(pd):
    return RollVessel(RollVesselConfig(
        I44=pd.I44, a44=pd.a44_assumed, b44_lin=pd.b44_assumed,
        GM=GM, displacement=pd.displacement, rho=pd.rho, g=pd.g,
    ))


# Use the same passive U-tube geometry across all variants so the only
# variable across the active/passive comparison is the controller.
_UTUBE_GEOM = dict(
    duct_below_waterline=6.5,
    undisturbed_fluid_height=2.5,
    utube_duct_height=0.6,
    resevoir_duct_width=2.0,
    utube_duct_width=16.0,
    tank_thickness=5.0,
    tank_to_xcog=0.0,
    tank_wall_friction_coef=0.05,
    tank_height=5.0,
)

F_MAX = 200_000.0  # N, ~bow-tunnel-thruster class


def _passive_utube():
    return OpenUtubeTank(OpenUtubeConfig(**_UTUBE_GEOM))


def _free_surface(c44, I_tot):
    cfg, _ = tune_self_consistent(
        length=22.4, width=8.0,
        z_tank=8.5, z_cog=2.5, damping_ratio=0.10,
        vessel_c44=c44, vessel_inertia_total=I_tot,
        warn_fill_ratio=10.0,
    )
    return FreeSurfaceTank(cfg)


def _tmd(I_tot, omega_p):
    h_arm = 6.0
    mu = 0.05
    mass = mu * I_tot / h_arm**2
    omega_t, zeta_t = den_hartog_optimal(mass, h_arm, I_tot, omega_p)
    return TunedMassDamperTank(TunedMassDamperConfig(
        mass=mass, natural_frequency=omega_t,
        z_mount=2.5 + h_arm, z_cog=2.5, damping_ratio=zeta_t,
    ))


def _rdt_ideal(M_wave_func):
    cfg = RDTUtubeConfig(F_max=F_MAX, **_UTUBE_GEOM)
    ctrl = InverseDynamicsRDTController(M_wave_func=M_wave_func)
    return RDTUtubeTank(cfg, controller=ctrl)


def _rdt_pd():
    """Pure rate-feedback PD controller. K_phidot tuned to give roughly
    critical damping addition to the bare vessel at resonance.

    The tank's contribution to the *closed-loop* damping coefficient
    (via the active force) at peak is roughly K_phidot times the
    vessel-to-tank coupling. Tuning by hand: K_phidot ~ 5e7 N/(rad/s)
    gives a useful effect without saturating immediately at the F_max
    we picked. Larger K saturates more aggressively and degrades
    performance.
    """
    cfg = RDTUtubeConfig(F_max=F_MAX, **_UTUBE_GEOM)
    ctrl = StateFeedbackRDTController(K_phi=0.0, K_phidot=5e7)
    return RDTUtubeTank(cfg, controller=ctrl)


def _rdt_observer(I_total, b44, c44, omega_e):
    """Realistic active controller: Sælid resonator + Luenberger
    observer reconstructs M_wave from phi(t) and the known applied
    M_tank, then drives the same algebraic inversion as the ideal
    controller.

    Parameters are the *vessel-only* effective roll EOM coefficients
    (I_total = I44 + a44, b44 = linear damping, c44 = restoring stiffness)
    and the dominant encounter frequency omega_e of the seastate.
    """
    cfg = RDTUtubeConfig(F_max=F_MAX, **_UTUBE_GEOM)
    obs = LuenbergerWaveObserver(LuenbergerWaveObserverConfig(
        I_total=I_total, b44=b44, c44=c44, omega_e=omega_e,
    ))
    ctrl = ResonatorObserverRDTController(observer=obs)
    return RDTUtubeTank(cfg, controller=ctrl)


# -------------------------------------------------------- run


def _run_one(pd, M_wave, tank_factory):
    v = _vessel(pd)
    if tank_factory is None:
        tanks = []
    else:
        tank = tank_factory(M_wave) if "M_wave_func" in tank_factory.__code__.co_varnames else tank_factory()
        # tank_factory accepting M_wave is the inverse-dynamics one
        tanks = [tank]
    sys = CoupledSystem(v, tanks=tanks, M_wave_func=M_wave)
    return sys, run_simulation(sys, dt=DT, t_end=T_WARMUP + T_SIM)


def _phi_significant(results) -> float:
    n_skip = int(T_WARMUP / DT)
    return float(4.0 * np.std(results.phi[n_skip:]))


def main():
    pd = load_csov(DATA / "csov_pdstrip.dat", DATA / "csov_pdstrip.inp")
    c44 = pd.rho * pd.g * pd.displacement * GM
    I_tot = pd.I44 + pd.a44_assumed
    omega_n = float(np.sqrt(c44 / I_tot))
    T_n = 2 * np.pi / omega_n

    print(f"Vessel T_n = {T_n:.2f} s, sea Tp = {TP:.2f} s, "
          f"Hs = {HS:.1f} m, gamma = {GAMMA:.1f}")
    print(f"RDT actuator F_max = {F_MAX/1e3:.0f} kN")

    wave = IrregularWave(
        Hs=HS, Tp=TP, gamma=GAMMA, heading_deg=HEADING,
        omega_min=0.15, omega_max=1.5, n_components=512, seed=SEED,
    )
    M_wave = roll_moment_from_irregular(wave, pd)

    # Each entry: (label, factory, color, ls). factory is either None
    # (bare vessel), a no-arg callable (passive tanks), or a callable
    # taking M_wave_func (the inverse-dynamics RDT).
    cases = [
        ("bare vessel",                 None,                                "k",  "-"),
        ("passive open U-tube",         _passive_utube,                      "C0", "-"),
        ("free-surface (self-cons.)",   lambda: _free_surface(c44, I_tot),   "C2", "-"),
        ("TMD (mu=5%, opt)",            lambda: _tmd(I_tot, omega_n),        "C3", "-"),
        ("RDT, PD (signals only)",      _rdt_pd,                             "C5", "-"),
        ("RDT, observer (Sælid+Luenb)", lambda: _rdt_observer(
                                            I_tot, pd.b44_assumed, c44,
                                            omega_e=omega_n),                "C1", "--"),
        ("RDT, ideal (knows M_wave)",   _rdt_ideal,                          "C4", "-"),
    ]

    fig, (ax_phi, ax_thrust, ax_bar) = plt.subplots(
        3, 1, figsize=(11, 9.5),
        gridspec_kw={"height_ratios": [2.0, 1.0, 1.2]},
    )

    bare_sig = None
    labels, sigs, sat_pcts, p_avgs = [], [], [], []
    rdt_traces = {}   # for thrust subplot

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for label, factory, color, ls in cases:
            print(f"  running {label}...")
            sys, res = _run_one(pd, M_wave, factory)
            sig = _phi_significant(res)
            if bare_sig is None:
                bare_sig = sig
            sig_deg = float(np.rad2deg(sig))
            reduction = 100.0 * (1.0 - sig / bare_sig)

            # If this is an RDT tank, sweep the recorded actuator history
            # and compute time-average dissipated power.
            sat_pct = None
            p_avg_kw = None
            if sys.tanks and isinstance(sys.tanks[0], RDTUtubeTank):
                tank = sys.tanks[0]
                # Re-run with thrust logging by rerunning with a wrapper.
                # (Cheaper: rerun is fine; ~6 s each.)
                v = _vessel(pd)
                if factory is _rdt_ideal:
                    ctrl_local = InverseDynamicsRDTController(M_wave_func=M_wave)
                    tank2 = RDTUtubeTank(
                        RDTUtubeConfig(F_max=F_MAX, **_UTUBE_GEOM),
                        controller=ctrl_local,
                    )
                elif "observer" in label:
                    obs_local = LuenbergerWaveObserver(LuenbergerWaveObserverConfig(
                        I_total=I_tot, b44=pd.b44_assumed, c44=c44,
                        omega_e=omega_n,
                    ))
                    ctrl_local = ResonatorObserverRDTController(observer=obs_local)
                    tank2 = RDTUtubeTank(
                        RDTUtubeConfig(F_max=F_MAX, **_UTUBE_GEOM),
                        controller=ctrl_local,
                    )
                else:
                    ctrl_local = StateFeedbackRDTController(K_phi=0.0, K_phidot=5e7)
                    tank2 = RDTUtubeTank(
                        RDTUtubeConfig(F_max=F_MAX, **_UTUBE_GEOM),
                        controller=ctrl_local,
                    )
                sys2 = CoupledSystem(v, tanks=[tank2], M_wave_func=M_wave)
                F_log = []
                v_duct_log = []
                t_log = []
                A_res = _UTUBE_GEOM["resevoir_duct_width"] * _UTUBE_GEOM["tank_thickness"]
                A_duct = _UTUBE_GEOM["utube_duct_height"] * _UTUBE_GEOM["tank_thickness"]
                W = _UTUBE_GEOM["utube_duct_width"] + _UTUBE_GEOM["resevoir_duct_width"]
                # Manual integration loop for logging
                t = 0.0
                t_end = T_WARMUP + T_SIM
                while t < t_end - 1e-9:
                    sys2.step(t, DT)
                    t += DT
                    F_log.append(tank2.last_F_applied)
                    v_duct_log.append(
                        (A_res / A_duct) * (W / 2.0) * float(tank2.state[1])
                    )
                    t_log.append(t)
                F_log = np.asarray(F_log)
                v_duct_log = np.asarray(v_duct_log)
                t_log = np.asarray(t_log)
                # Saturation fraction (post-warmup)
                mask_post = t_log > T_WARMUP
                sat_pct = 100.0 * np.mean(np.abs(F_log[mask_post]) >= F_MAX * 0.999)
                # Two power figures (post-warmup):
                #   * Apparent (|F*v|): envelope of reactive swings.
                #     Sets motor/drive electronics rating.
                #   * Net dissipated (signed F*v, with positive = drive
                #     does work on the fluid): mean energy the battery
                #     actually drains. With a regenerative drive most of
                #     the apparent power is recovered.
                P_app_inst = np.abs(F_log[mask_post] * v_duct_log[mask_post])
                P_net_inst = F_log[mask_post] * v_duct_log[mask_post]
                p_avg_kw = float(np.mean(P_app_inst) / 1e3)
                p_net_kw = float(np.mean(P_net_inst) / 1e3)
                rdt_traces[label] = (t_log, F_log, color, ls)

            print(f"    phi_1/3 = {sig_deg:6.3f} deg "
                  f"({reduction:+5.1f} % vs bare)"
                  + (f"  | sat {sat_pct:5.1f}% | "
                     f"<|P|>={p_avg_kw:5.1f} kW (apparent), "
                     f"<P>={p_net_kw:5.1f} kW (net)"
                     if sat_pct is not None else ""))
            labels.append(label)
            sigs.append(sig_deg)
            sat_pcts.append(sat_pct)
            p_avgs.append(p_avg_kw)

            ax_phi.plot(res.t, np.rad2deg(res.phi), ls, color=color,
                        lw=0.9, alpha=0.85, label=label)

    ax_phi.axvline(T_WARMUP, color="grey", linestyle=":",
                   alpha=0.5, label=f"warmup ends ({T_WARMUP:.0f} s)")
    ax_phi.set_ylabel("roll [deg]")
    ax_phi.set_title(
        f"CSOV roll in JONSWAP, Hs = {HS:.1f} m, Tp = {TP:.1f} s, "
        f"gamma = {GAMMA:.1f}, beam seas (seed = {SEED})"
    )
    ax_phi.grid(True, alpha=0.3)
    ax_phi.legend(loc="upper right", fontsize=8, ncol=2)

    # Thrust subplot
    for lbl, (tlog, Flog, color, ls) in rdt_traces.items():
        ax_thrust.plot(tlog, Flog / 1e3, ls, color=color,
                       lw=0.7, alpha=0.85, label=lbl)
    ax_thrust.axhline(+F_MAX/1e3, color="k", linestyle=":",
                      alpha=0.4, label=f"F_max = +/- {F_MAX/1e3:.0f} kN")
    ax_thrust.axhline(-F_MAX/1e3, color="k", linestyle=":", alpha=0.4)
    ax_thrust.axvline(T_WARMUP, color="grey", linestyle=":", alpha=0.5)
    ax_thrust.set_xlabel("time [s]")
    ax_thrust.set_ylabel("RDT thrust [kN]")
    ax_thrust.grid(True, alpha=0.3)
    ax_thrust.legend(loc="upper right", fontsize=8)

    # Bar chart
    colours = [c for (_, _, c, _) in cases]
    bars = ax_bar.bar(range(len(labels)), sigs, color=colours)
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax_bar.set_ylabel("phi_1/3 [deg]")
    ax_bar.grid(True, axis="y", alpha=0.3)
    for b, val, base, sat, p_avg in zip(
        bars, sigs, [sigs[0]] * len(sigs), sat_pcts, p_avgs,
    ):
        red = 100.0 * (1.0 - val / base)
        annot = f"{val:.2f}\n({red:+.0f}%)" if base != val else f"{val:.2f}"
        if p_avg is not None:
            annot += f"\nP={p_avg:.0f} kW\nsat={sat:.0f}%"
        ax_bar.text(b.get_x() + b.get_width() / 2, val,
                    annot, ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    out = OUT / "csov_irregular_seakeeping_with_rdt.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
