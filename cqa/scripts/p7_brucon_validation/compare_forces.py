"""P7 cross-validation: compare cqa-predicted vs brucon-simulated mean
environmental forces (wind, wave drift, current) per DOF.

Motivation
----------
Before iterating on the closed-loop transient model (which currently
predicts a 52 m sway runaway whereas the simulator stays bounded), we
need to know whether the *forcing* itself agrees. Disagreement here
would mean the saturation analysis is built on wrong inputs.

The simulator emits SurgeWindForce / SwayWindForce / YawWindMoment
(named WindX, WindY, WindMz in the output file) and equivalent for
drift and current, all in body-frame (surge, sway, yaw moment) -- see
brucon dp_runfast_simulator.cpp:1100-1106 PrintDataLine() and the
*Force() / *Moment() accessor names. cqa computes the same with
WindForceModel.force(Vw, theta_rel),
CurrentForceModel.force(Vc, theta_rel),
F_drift = (drift_x_amp, drift_y_amp, drift_n_amp) * Hs^2 * (cos, sin, sin(2)).

Procedure
---------
1. Load all 30 ensemble seeds from work/p7v1_seed*/p7v1_seedNNNN.out.
2. Restrict to the late intact window (last 200 s before failure) so the
   vessel is at steady state, mean position ~ 0, mean heading at setpoint.
3. Per seed, compute time-mean of each force component.
4. Across seeds, compute median + IQR -> ensemble force distribution.
5. Compute cqa's deterministic mean force at the same (Vw, Hs, Tp, Vc,
   theta_rel) and overlay.
6. Plot per-DOF (surge, sway, yaw) time series of the three force
   components for one representative seed plus the cqa value as a
   horizontal line, so we can see both mean agreement and amplitude /
   spectral character.

No new simulator runs are performed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from harness import parse_output

from cqa.config import csov_default_config
from cqa.vessel import WindForceModel, CurrentForceModel
from cqa.rao import load_pdstrip_rao
from cqa.drift import mean_drift_force_pdstrip
from cqa.sea_spreading import SeaSpreading

PDSTRIP_PATH = "/home/blofro/src/brucon/build/bin/vessel_simulator_config/csov_pdstrip.dat"


# --- Mirror canonical scenario windows (must match run_comparison.py) ---
ACTIVATE_SK_S = 60.0
SETTLE_S = 500.0
POST_FAILURE_S = 180.0
FAILURE_TIME_S = ACTIVATE_SK_S + SETTLE_S
TOTAL_S = FAILURE_TIME_S + POST_FAILURE_S
SIM_DT = 0.1
INTACT_SAMPLE_S = 200.0

# Scenario (must match run_comparison.py)
VW = 14.0
HS = 4.196
TP = 10.224
VC = 0.5
WAVE_DIR_COMPASS = 270.0
VESSEL_HEADING_COMPASS = 180.0
# Compass-relative weather direction (into vessel): +90 deg = beam from starboard.
THETA_REL = np.radians((WAVE_DIR_COMPASS - VESSEL_HEADING_COMPASS + 540) % 360 - 180)


def _bretschneider(omega: np.ndarray, Hs: float, Tp: float) -> np.ndarray:
    """Standard Bretschneider one-sided angular PSD (m^2 s/rad).

    Brucon `BretschneiderSpectralValue` (`wave_spectrum.cpp:47-54`).
    """
    wp = 2.0 * np.pi / Tp
    return (5.0 / 16.0) * Hs ** 2 * wp ** 4 / omega ** 5 * np.exp(-1.25 * (wp / omega) ** 4)


def _brucon_drift_replica(table, Hs: float, Tp: float, theta_rel_rad: float,
                          spreading_factor: float | None) -> np.ndarray:
    """Replicate brucon's discrete drift integral exactly.

    * Bretschneider spectrum (brucon's default for V_w-driven seas, no
      explicit `wave_spectrum_type` in `vessel_simulator_settings.prototxt`).
    * Pdstrip omega and beta grid (10 deg step), no resampling.
    * Directional weight: cos^n(delta) / scale where delta is angle to
      principal direction and scale = integral cos^n(delta) over (-pi/2, pi/2)
      (brucon `WaveSpectrum::DirectionWeight` with n = `spreading_factor`).
      Pass ``spreading_factor=None`` for long-crested.
    * Faltinsen [90] eq. 5.41 with factor 2: F = sum 2 wave_amp^2 transf.
    """
    from cqa.wave_response import cqa_theta_rel_to_pdstrip_beta_deg
    from cqa.rao import evaluate_drift

    omega = table.omega
    domega = np.gradient(omega)
    S = _bretschneider(omega, Hs, Tp)

    principal_beta = cqa_theta_rel_to_pdstrip_beta_deg(theta_rel_rad)
    betas = table.beta_deg

    if spreading_factor is None:
        # Long-crested: only principal direction, weight 1.
        D = evaluate_drift(table, omega, principal_beta)            # (n_omega, 3)
        F = 2.0 * np.sum(S[:, None] * domega[:, None] * D, axis=0)  # (3,)
        return F

    # Short-crested cos^n weight, normalised over (-pi/2, +pi/2).
    n = float(spreading_factor)
    delta_norm = np.linspace(-np.pi/2 + 1e-9, np.pi/2 - 1e-9, 1001)
    scale = np.trapezoid(np.cos(delta_norm) ** n, delta_norm)
    dir_step_rad = np.radians(betas[1] - betas[0])

    F = np.zeros(3)
    for b in betas:
        delta_deg = (b - principal_beta + 180.0) % 360.0 - 180.0
        if abs(delta_deg) >= 90.0:
            continue
        w_dir = (np.cos(np.radians(delta_deg)) ** n) / scale * dir_step_rad
        D = evaluate_drift(table, omega, b)                                  # (n_omega, 3)
        F += 2.0 * w_dir * np.sum(S[:, None] * domega[:, None] * D, axis=0)
    return F


def cqa_mean_forces() -> dict[str, np.ndarray]:
    """Return cqa's deterministic mean (wind, drift, current) force per DOF.

    The drift variants below trace the model-fidelity sensitivity that
    matters for cross-validating brucon vessel_simulator. Three knobs
    matter, in order of impact at this operating point:

    1. **Spectrum shape** -- JONSWAP gamma=3.3 vs Bretschneider (= JONSWAP
       gamma=1) differ by ~50% in their drift-integral overlap with the
       QTF. Brucon defaults to Bretschneider for V_w-driven seas (the
       only built-in spectra; selectable via
       `vessel_simulator_settings.prototxt: wave_spectrum_type`).
    2. **Spreading kind** -- brucon uses cos^n(delta) over (-pi/2, +pi/2);
       cqa's SeaSpreading is cos-2s (cos^2s(delta/2)) over (-pi, +pi).
       The Gaussian-limit equivalence is s ~ 2n, NOT s = n/2.
    3. **Faltinsen factor 2** vs factor 1 -- depends on whether the QTF
       table is in N/m^2 per unit amplitude^2 or per unit amplitude.
       pdstrip QTFs are per amplitude^2, factor 2 is correct.
    """
    cfg = csov_default_config()
    vp = cfg.vessel
    wp = cfg.wind
    cp = cfg.current
    wd = cfg.wave_drift

    wind_model = WindForceModel(wp=wp, loa=vp.loa)
    current_model = CurrentForceModel(
        cp=cp,
        lateral_area_underwater=vp.lpp * vp.draft,
        frontal_area_underwater=vp.beam * vp.draft,
        loa=vp.loa,
    )

    F_wind = wind_model.force(VW, THETA_REL)
    F_curr = current_model.force(VC, THETA_REL)

    # Parametric placeholder (config.WaveDriftParticulars defaults).
    F_drift_param = np.array([
        wd.drift_x_amp * HS ** 2 * np.cos(THETA_REL),
        wd.drift_y_amp * HS ** 2 * np.sin(THETA_REL),
        wd.drift_n_amp * HS ** 2 * np.sin(2.0 * THETA_REL),
    ])

    # pdstrip-QTF variants (Bretschneider, three spreading models)
    table = load_pdstrip_rao(PDSTRIP_PATH)
    F_drift_bret_long = _brucon_drift_replica(table, HS, TP, THETA_REL, spreading_factor=None)
    F_drift_bret_cos2 = _brucon_drift_replica(table, HS, TP, THETA_REL, spreading_factor=2.0)
    # cqa's existing JONSWAP integrator with default cos-2s s=15, for reference.
    F_drift_jons_s15 = mean_drift_force_pdstrip(
        table, HS, TP, THETA_REL,
        spreading=SeaSpreading(kind="cos_2s", s=15.0, n_dir=37),
    )

    return {
        "wind": F_wind,
        "drift_param": F_drift_param,
        "drift_bret_long": F_drift_bret_long,
        "drift_bret_cos2": F_drift_bret_cos2,
        "drift_jons_cos2s15": F_drift_jons_s15,
        "current": F_curr,
        "total": F_wind + F_drift_bret_cos2 + F_curr,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    work_dir = out_dir / "work"

    seed_dirs = sorted(p for p in work_dir.iterdir()
                       if p.is_dir() and p.name.startswith("p7v1_seed"))
    if not seed_dirs:
        raise SystemExit(f"No p7v1_seed* dirs under {work_dir}")
    print(f"Loading {len(seed_dirs)} seeds ...")

    results = [parse_output(d / f"{d.name}.out") for d in seed_dirs]
    n_min = min(r.n_rows for r in results)
    n_seeds = len(results)
    t = results[0]["t"][:n_min]

    # Late intact window mask
    intact_mask = (t >= FAILURE_TIME_S - INTACT_SAMPLE_S) & (t < FAILURE_TIME_S - 1.0)
    intact_n = intact_mask.sum()
    print(f"  intact-window: {intact_n * SIM_DT:.0f} s, {intact_n} samples per seed")

    # Sim force arrays: shape (N_seeds, n_t)
    components = {
        "wind":    {"x": "WindX",  "y": "WindY",  "n": "WindMz"},
        "drift":   {"x": "DriftX", "y": "DriftY", "n": "DriftMz"},
        "current": {"x": "CurX",   "y": "CurY",   "n": "CurMz"},
    }
    # IMPORTANT: brucon CSV emits all forces/moments in kN and kN.m
    # (see dp_runfast_simulator.cpp PrintDataLine -- values are divided by 1e3
    # before printing). cqa works internally in N and N.m. Convert here.
    KN_TO_N = 1.0e3
    sim = {}    # sim[component][dof] = (N_seeds, n_intact) np.ndarray, in N or N.m
    for comp, cols in components.items():
        sim[comp] = {}
        for dof, colname in cols.items():
            sim[comp][dof] = np.array(
                [r[colname][:n_min][intact_mask] for r in results]
            ) * KN_TO_N

    # cqa side
    F_cqa = cqa_mean_forces()

    # ------------------------------------------------------------------
    # Print per-DOF mean comparison
    # ------------------------------------------------------------------
    print(f"\nMean force comparison (theta_rel = {np.degrees(THETA_REL):+.1f} deg, "
          f"Vw={VW} m/s, Hs={HS:.2f} m, Tp={TP:.2f} s, Vc={VC} m/s)")
    print("=" * 110)
    print(f"{'component':<32}{'DOF':<6}{'cqa mean':>12}  "
          f"{'sim ensemble (median, IQR-lo, IQR-hi)':>40}  {'unit':<8}")
    print("-" * 110)
    for comp_name, sim_key, cqa_vec in [
        ("wind",                            "wind",    F_cqa["wind"]),
        ("drift (param placeholder)",       "drift",   F_cqa["drift_param"]),
        ("drift (Bret long-crested)",       "drift",   F_cqa["drift_bret_long"]),
        ("drift (Bret cos\u00b2, brucon repl.)", "drift",   F_cqa["drift_bret_cos2"]),
        ("drift (JONSWAP cos-2s s=15)",     "drift",   F_cqa["drift_jons_cos2s15"]),
        ("current",                         "current", F_cqa["current"]),
        ("total (Bret cos\u00b2)",              "total",   F_cqa["total"]),
    ]:
        for i, (dof, unit) in enumerate([("surge", "kN"), ("sway", "kN"), ("yaw", "kN.m")]):
            scale = 1e-3
            cqa_val = cqa_vec[i] * scale
            if sim_key == "total":
                # Sum the three sim components per seed-and-time, then time-mean per seed
                per_seed_mean = (
                    sim["wind"][["x", "y", "n"][i]].mean(axis=1)
                    + sim["drift"][["x", "y", "n"][i]].mean(axis=1)
                    + sim["current"][["x", "y", "n"][i]].mean(axis=1)
                ) * scale
            else:
                per_seed_mean = sim[sim_key][["x", "y", "n"][i]].mean(axis=1) * scale
            sim_med = np.median(per_seed_mean)
            sim_lo = np.quantile(per_seed_mean, 0.25)
            sim_hi = np.quantile(per_seed_mean, 0.75)
            ratio = sim_med / cqa_val if abs(cqa_val) > 1e-6 else float("nan")
            print(f"{comp_name:<32}{dof:<6}{cqa_val:>+12.1f}  "
                  f"{sim_med:>+10.1f}  [{sim_lo:>+8.1f}, {sim_hi:>+8.1f}]  "
                  f"{unit:<8} (sim/cqa = {ratio:+.2f})")
        print()

    # ------------------------------------------------------------------
    # Plot: per-DOF time series, one representative seed, all 3 components
    # ------------------------------------------------------------------
    F_cqa_plot = {
        "wind":    F_cqa["wind"],
        "drift":   F_cqa["drift_bret_cos2"],
        "current": F_cqa["current"],
    }
    rep_seed = 0
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    t_intact = t[intact_mask] - FAILURE_TIME_S  # shift so failure = 0
    dof_labels = ["surge", "sway", "yaw"]
    units = ["kN", "kN", "kN.m"]
    sim_keys = ["x", "y", "n"]
    colors = {"wind": "tab:orange", "drift": "tab:blue", "current": "tab:green"}

    for i, ax in enumerate(axes):
        scale = 1e-3
        for comp, color in colors.items():
            trace = sim[comp][sim_keys[i]][rep_seed] * scale
            ax.plot(t_intact, trace, color=color, alpha=0.8, lw=0.7,
                    label=f"sim {comp}")
            cqa_val = F_cqa_plot[comp][i] * scale
            ax.axhline(cqa_val, color=color, ls="--", alpha=0.9, lw=1.2,
                       label=f"cqa {comp} = {cqa_val:+.1f}")
        ax.set_ylabel(f"{dof_labels[i]} [{units[i]}]")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.axhline(0.0, color="black", lw=0.5, alpha=0.5)

    axes[2].set_xlabel("time relative to failure [s]  (negative = pre-failure intact window)")
    fig.suptitle(
        f"P7 force comparison -- seed #{rep_seed} late intact window\n"
        f"Vw={VW} m/s, Hs={HS:.2f} m, Tp={TP:.2f} s, Vc={VC} m/s, "
        f"theta_rel={np.degrees(THETA_REL):+.0f} deg (beam)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out1 = out_dir / "p7_validation_forces_timeseries.png"
    fig.savefig(out1, dpi=130)
    plt.close(fig)
    print(f"\n  wrote {out1}")

    # ------------------------------------------------------------------
    # Plot: ensemble distribution of per-seed mean force, per DOF
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for i, dof in enumerate(dof_labels):
        for j, comp in enumerate(["wind", "drift", "current"]):
            ax = axes[i, j]
            scale = 1e-3
            per_seed_mean = sim[comp][sim_keys[i]].mean(axis=1) * scale
            cqa_val = F_cqa_plot[comp][i] * scale
            ax.hist(per_seed_mean, bins=12, color=colors[comp], alpha=0.6,
                    edgecolor="black", lw=0.5)
            ax.axvline(np.median(per_seed_mean), color=colors[comp], lw=2.0,
                       label=f"sim median = {np.median(per_seed_mean):+.1f}")
            ax.axvline(cqa_val, color="black", ls="--", lw=1.5,
                       label=f"cqa = {cqa_val:+.1f}")
            ax.set_title(f"{comp} {dof} [{units[i]}]", fontsize=10)
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(True, alpha=0.4)
    fig.suptitle(
        f"P7 force comparison -- ensemble distribution of per-seed mean force\n"
        f"(N={n_seeds} seeds, late intact window {intact_n * SIM_DT:.0f} s, "
        f"Vw={VW}, Hs={HS:.2f}, Tp={TP:.2f}, Vc={VC}, theta_rel={np.degrees(THETA_REL):+.0f} deg)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out2 = out_dir / "p7_validation_forces_distribution.png"
    fig.savefig(out2, dpi=130)
    plt.close(fig)
    print(f"  wrote {out2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
