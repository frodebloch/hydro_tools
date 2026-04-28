"""Heading sweep: passive vs RDT-active anti-roll, JONSWAP irregular seas.

Extends ``csov_irregular_seakeeping.py`` to compare the bare vessel,
the standard passive devices, and three RDT-active U-tube
configurations across a sweep of wave headings (90 deg = beam,
120 deg = 30 deg off bow, 150 deg = 60 deg off bow). The single-
heading version of this example was misleading because real
station-keeping CSOVs choose a favourable heading whenever they can,
so beam seas is a worst-case design point, not the operational one.

Active controllers compared
---------------------------
  * **Ideal active (inverse-dynamics)** -- assumes perfect knowledge
    of M_wave(t) and inverts the coupled dynamics. Performance ceiling.
  * **Observer-based active** -- realistic shipboard counterpart of
    the ideal: a Saelid 2nd-order resonator + Luenberger observer
    reconstructs M_wave from the roll signal and the known applied
    tank moment, then drives the same algebraic inversion. See
    README sec. 4.z.
  * **State-feedback PD** -- pure rate damping on phi_dot, no wave
    knowledge. Honest signal-only baseline.

Actuator sizing
---------------
F_max = 80 kN, the realistic single-RDT fluid-thrust ceiling for a
1.0 MW / 1.8 m shaft after rim friction (~30-35 % of P_shaft) and
propeller losses are subtracted. See README sec. 4.y for the full
hardware-sizing derivation; vendor "system thrust" figures for hull-
installed tunnel thrusters are ~2x larger but include hull-suction
reactions absent in the closed-duct application here.

Authority check
---------------
At F_max = 80 kN with a wing-tank arm of ~8 m the direct counter-
moment is at most ~0.7 MN*m. The wave-exciting roll moment in beam
seas at Hs = 3 m peaks at ~8-12 MN*m. Even with U-tube resonance
amplification (~3-5x) the available tank moment is well short of the
forcing -- the actuator is roughly an order of magnitude under-
authority for outright wave cancellation. This is reflected in the
results below: 80 kN is sufficient to *trim* the resonance peak in
favourable headings, but in beam seas it is dominated by the simple
passive free-surface tank.

Output:
  examples/output/csov_irregular_seakeeping_with_rdt.png
"""
from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

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
HEADINGS = [90.0, 120.0, 150.0]   # deg: beam / 30-off-bow / 60-off-bow
SEED = 20260428

T_WARMUP = 60.0
T_SIM = 600.0
DT = 0.05

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

F_MAX = 80_000.0  # N, realistic single-RDT fluid thrust; see docstring.

# -------------------------------------------------------- factories


def _vessel(pd):
    return RollVessel(RollVesselConfig(
        I44=pd.I44, a44=pd.a44_assumed, b44_lin=pd.b44_assumed,
        GM=GM, displacement=pd.displacement, rho=pd.rho, g=pd.g,
    ))


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
    cfg = RDTUtubeConfig(F_max=F_MAX, **_UTUBE_GEOM)
    ctrl = StateFeedbackRDTController(K_phi=0.0, K_phidot=5e7)
    return RDTUtubeTank(cfg, controller=ctrl)


def _rdt_observer(I_total, b44, c44, omega_e):
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
        tank = (
            tank_factory(M_wave)
            if "M_wave_func" in tank_factory.__code__.co_varnames
            else tank_factory()
        )
        tanks = [tank]
    sys = CoupledSystem(v, tanks=tanks, M_wave_func=M_wave)
    return sys, run_simulation(sys, dt=DT, t_end=T_WARMUP + T_SIM)


def _phi_significant(results) -> float:
    n_skip = int(T_WARMUP / DT)
    return float(4.0 * np.std(results.phi[n_skip:]))


def _make_cases():
    """Return list of (label, factory, color, marker, needs_M_wave).

    factory takes either zero args, or M_wave (for inverse-dynamics).
    needs_M_wave just controls the inner-loop dispatch sugar.
    """
    return [
        ("bare vessel",                 None,             "k",  "o"),
        ("passive open U-tube",         "_passive_utube", "C0", "s"),
        ("free-surface (self-cons.)",   "_free_surface",  "C2", "^"),
        ("TMD (mu=5%, opt)",            "_tmd",           "C3", "v"),
        ("RDT, PD (signals only)",      "_rdt_pd",        "C5", "D"),
        ("RDT, observer (Sælid+Luenb)", "_rdt_observer",  "C1", "P"),
        ("RDT, ideal (knows M_wave)",   "_rdt_ideal",     "C4", "*"),
    ]


def main():
    pd = load_csov(DATA / "csov_pdstrip.dat", DATA / "csov_pdstrip.inp")
    c44 = pd.rho * pd.g * pd.displacement * GM
    I_tot = pd.I44 + pd.a44_assumed
    omega_n = float(np.sqrt(c44 / I_tot))
    T_n = 2 * np.pi / omega_n

    print(f"Vessel T_n = {T_n:.2f} s, sea Tp = {TP:.2f} s, "
          f"Hs = {HS:.1f} m, gamma = {GAMMA:.1f}")
    print(f"RDT actuator F_max = {F_MAX/1e3:.0f} kN")
    print(f"Heading sweep: {HEADINGS} deg "
          f"(90 = beam, larger = nearer head seas)\n")

    cases = _make_cases()

    # results[case_label] = list of (heading_deg, phi_sig_deg, reduction_pct, sat_pct, p_avg_kw)
    results = {label: [] for (label, *_rest) in cases}

    for heading in HEADINGS:
        print(f"=== heading = {heading:.0f} deg ===")
        wave = IrregularWave(
            Hs=HS, Tp=TP, gamma=GAMMA, heading_deg=heading,
            omega_min=0.15, omega_max=1.5, n_components=512, seed=SEED,
        )
        M_wave = roll_moment_from_irregular(wave, pd)

        bare_sig = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for label, fac_name, color, marker in cases:
                # Resolve factory by name (lets us re-bind c44/I_tot per heading
                # cleanly; constants here, but keeps the dispatch uniform).
                if fac_name is None:
                    factory = None
                elif fac_name == "_passive_utube":
                    factory = _passive_utube
                elif fac_name == "_free_surface":
                    factory = lambda: _free_surface(c44, I_tot)
                elif fac_name == "_tmd":
                    factory = lambda: _tmd(I_tot, omega_n)
                elif fac_name == "_rdt_pd":
                    factory = _rdt_pd
                elif fac_name == "_rdt_observer":
                    factory = lambda: _rdt_observer(
                        I_tot, pd.b44_assumed, c44, omega_e=omega_n,
                    )
                elif fac_name == "_rdt_ideal":
                    factory = _rdt_ideal
                else:
                    raise ValueError(fac_name)

                sys, res = _run_one(pd, M_wave, factory)
                sig = _phi_significant(res)
                if bare_sig is None:
                    bare_sig = sig
                sig_deg = float(np.rad2deg(sig))
                reduction = 100.0 * (1.0 - sig / bare_sig)

                # RDT diagnostics: rerun with logging to harvest
                # saturation fraction and apparent power.
                sat_pct = None
                p_avg_kw = None
                if sys.tanks and isinstance(sys.tanks[0], RDTUtubeTank):
                    sat_pct, p_avg_kw = _rdt_diagnostic_run(
                        pd, M_wave, fac_name, c44, I_tot, omega_n,
                    )

                results[label].append(
                    (heading, sig_deg, reduction, sat_pct, p_avg_kw)
                )
                tag = ""
                if sat_pct is not None:
                    tag = (f"  | sat {sat_pct:5.1f}%"
                           f" | <|P|>={p_avg_kw:5.1f} kW")
                print(f"  {label:35s}  phi_1/3 = {sig_deg:5.2f} deg "
                      f"({reduction:+5.1f} %){tag}")
        print()

    _plot(results, cases)


def _rdt_diagnostic_run(pd, M_wave, fac_name, c44, I_tot, omega_n):
    """Re-run an RDT case with manual stepping to log thrust history.

    Returns (sat_pct, p_apparent_kw).
    """
    v = _vessel(pd)
    if fac_name == "_rdt_ideal":
        ctrl = InverseDynamicsRDTController(M_wave_func=M_wave)
    elif fac_name == "_rdt_observer":
        obs = LuenbergerWaveObserver(LuenbergerWaveObserverConfig(
            I_total=I_tot, b44=pd.b44_assumed, c44=c44, omega_e=omega_n,
        ))
        ctrl = ResonatorObserverRDTController(observer=obs)
    elif fac_name == "_rdt_pd":
        ctrl = StateFeedbackRDTController(K_phi=0.0, K_phidot=5e7)
    else:
        raise ValueError(fac_name)
    tank = RDTUtubeTank(RDTUtubeConfig(F_max=F_MAX, **_UTUBE_GEOM),
                        controller=ctrl)
    sys2 = CoupledSystem(v, tanks=[tank], M_wave_func=M_wave)

    A_res = _UTUBE_GEOM["resevoir_duct_width"] * _UTUBE_GEOM["tank_thickness"]
    A_duct = _UTUBE_GEOM["utube_duct_height"] * _UTUBE_GEOM["tank_thickness"]
    W = _UTUBE_GEOM["utube_duct_width"] + _UTUBE_GEOM["resevoir_duct_width"]

    F_log, v_log, t_log = [], [], []
    t = 0.0
    t_end = T_WARMUP + T_SIM
    while t < t_end - 1e-9:
        sys2.step(t, DT)
        t += DT
        F_log.append(tank.last_F_applied)
        v_log.append((A_res / A_duct) * (W / 2.0) * float(tank.state[1]))
        t_log.append(t)
    F_log = np.asarray(F_log)
    v_log = np.asarray(v_log)
    t_log = np.asarray(t_log)
    mask = t_log > T_WARMUP
    sat_pct = 100.0 * np.mean(np.abs(F_log[mask]) >= F_MAX * 0.999)
    p_avg_kw = float(np.mean(np.abs(F_log[mask] * v_log[mask])) / 1e3)
    return sat_pct, p_avg_kw


def _plot(results, cases):
    """Two-panel summary: top = phi_1/3 vs heading, bottom = % reduction."""
    fig, (ax_abs, ax_red) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for label, _fac, color, marker in cases:
        rows = results[label]
        h = [r[0] for r in rows]
        sig = [r[1] for r in rows]
        red = [r[2] for r in rows]
        ax_abs.plot(h, sig, marker=marker, color=color,
                    lw=1.4, ms=8, label=label)
        if label != "bare vessel":
            ax_red.plot(h, red, marker=marker, color=color,
                        lw=1.4, ms=8, label=label)

    ax_abs.set_ylabel(r"$\phi_{1/3}$  [deg]")
    ax_abs.set_title(
        f"CSOV roll across heading sweep, JONSWAP "
        f"Hs = {HS:.1f} m, Tp = {TP:.1f} s, gamma = {GAMMA:.1f}, "
        f"F_max = {F_MAX/1e3:.0f} kN"
    )
    ax_abs.grid(True, alpha=0.3)
    ax_abs.legend(loc="upper right", fontsize=8, ncol=2)

    ax_red.set_xlabel("wave heading [deg]  (90 = beam, 180 = head)")
    ax_red.set_ylabel("reduction vs bare [%]")
    ax_red.axhline(0.0, color="k", lw=0.5)
    ax_red.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUT / "csov_irregular_seakeeping_with_rdt.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
