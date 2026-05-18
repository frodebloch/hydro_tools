"""Generate the brucon validation-matrix ensembles.

Co-linear cells {head-on, Q10} x {Bf 6 (nominal), Bf 8 (extreme)},
plus 45-deg wind/wave split cells with the same {theta_rel, Bf} grid,
all per DNV-ST-0111 Table (provided by the user 2026-05):

    Bf 6  (nominal): Vw = 13.8 m/s, Hs = 3.1 m, Tp = 8.5 s, Vc = 0.75 m/s
    Bf 8 (extreme): Vw = 20.7 m/s, Hs = 5.7 m, Tp = 10.0 s, Vc = 0.75 m/s

WCFDI: bus_port (Bow1 + PortMP), matching pwq30 / pwo.

Tags written under work/<TAG>_seed{1000..1029}/:

  Co-linear (wind = waves = current):
    bf6_h0       head-on,   Bf 6
    bf6_q10      theta=+10, Bf 6
    bf8_h0       head-on,   Bf 8
    bf8_q10      theta=+10, Bf 8

  45-deg split (wind veers +45 deg right of waves; current co-linear with waves):
    bf6_h0_w45   head-on,   Bf 6
    bf6_q10_w45  theta=+10, Bf 6
    bf8_h0_w45   head-on,   Bf 8
    bf8_q10_w45  theta=+10, Bf 8

Once a cell is generated, validate it with::

    .venv/bin/python scripts/p7_brucon_validation/peak_R_b_hat_sigma_pwq30.py --tag bf6_h0
    .venv/bin/python scripts/p7_brucon_validation/where_we_are_now_pwq30.py --tag bf6_h0
    .venv/bin/python scripts/p7_brucon_validation/live_cell_per_seed_pwq30.py --tag bf6_h0

Wall-clock: ~30 s/cell on 12 workers (4 cells x 30 seeds x 1 s each / 12 = 10 s
  if perfectly parallel; observed ~30 s with overheads).

Run from cqa root::

    .venv/bin/python scripts/p7_brucon_validation/run_validation_matrix.py [--tags bf6_h0,...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from harness import ScenarioSpec, run_ensemble, CSOV_WCF_GROUPS  # noqa: E402
from _constants import ACTIVATE_SK_S, SETTLE_S, POST_FAILURE_S, T_WCF_S  # noqa: E402

WORK_DIR = THIS / "work"
N_SEEDS = 30
BASE_SEED = 1000

# Vessel heading: bow into 180 deg compass (matches pwq30 / pwo).
VESSEL_HEADING_COMPASS = 180.0

# WCFDI failure: bus_port (Bow1 + PortMP). Matches pwq30 / pwo.
FAILED_THRUSTERS = CSOV_WCF_GROUPS["bus_port"]

# Lua run timings come from _constants.py (single source of truth for the
# brucon-validation matrix). settle_s=1500 chosen so the bias estimator
# (tau_b = 1000 s) has > 1*tau_b free DP after station-keeping activation,
# eliminating the pre-WCF demean-window contamination diagnosed in
# analysis.md sec.12.21.15. -> failure_time = 60 + 1500 = 1560 s,
# total = 1740 s. Existing data generated with the older
# settle_s=500/T_WCF=560 layout will no longer match these timings; the
# downstream validation scripts will loudly fail rather than silently
# read off the wrong sim time, which is intentional.

# DNV-ST-0111 Beaufort table (image provided 2026-05).
BF6 = dict(Vw=13.8, Hs=3.1, Tp=8.5, Vc=0.75)
BF8 = dict(Vw=20.7, Hs=5.7, Tp=10.0, Vc=0.75)
# Bf 4 benign sea state for low-variability current-dominated check
# (DNV-ish standard values, current overridden per-cell):
BF4 = dict(Vw=7.0, Hs=1.5, Tp=6.0, Vc=0.75)
# Half-step BF interpolation (sec.12.21.21.30b): linear in Beaufort
# index between BF6 and BF8 anchors. Per-Bf step is (+3.45 m/s Vw,
# +1.3 m Hs, +0.75 s Tp). BF7.5 sits halfway between BF7 and BF8;
# BF8.5 extrapolates the same slope halfway to BF9. Designed to
# probe the sigma-headroom gradient at the operational boundary --
# BF8.5 is the most likely cell to land in the Regime-B amber band
# and give the first real brucon validation point for Option 2.
BF7P5 = dict(Vw=18.975, Hs=5.05, Tp=9.625,  Vc=0.75)
BF8P5 = dict(Vw=22.425, Hs=6.35, Tp=10.375, Vc=0.75)


def _make_spec(env: dict, theta_rel_deg: float,
               wind_offset_deg: float = 0.0,
               current_compass_abs: float | None = None,
               current_speed_override: float | None = None) -> ScenarioSpec:
    """Build a ScenarioSpec.

    theta_rel_deg = (wave_compass - vessel_heading_compass), i.e. waves and
    current come from compass = 180 + theta_rel_deg. wind comes from
    (wave_compass + wind_offset_deg).

    If current_compass_abs is given, current direction is taken absolute
    (independent of waves) — used for current-dominated cells where the
    current loads the vessel asymmetrically. current_speed_override
    likewise replaces env["Vc"] when supplied.
    """
    wave_compass = (VESSEL_HEADING_COMPASS + theta_rel_deg) % 360.0
    wind_compass = (wave_compass + wind_offset_deg) % 360.0
    current_compass = (current_compass_abs % 360.0
                       if current_compass_abs is not None else wave_compass)
    current_speed = (current_speed_override
                     if current_speed_override is not None else env["Vc"])
    return ScenarioSpec(
        Hs=env["Hs"], Tp=env["Tp"], wave_dir_compass=wave_compass,
        wind_speed=env["Vw"], wind_dir_compass=wind_compass,
        current_speed=current_speed, current_dir_compass=current_compass,
        vessel_heading_compass=VESSEL_HEADING_COMPASS,
        failed_thruster_indices=FAILED_THRUSTERS,
        activate_sk_s=ACTIVATE_SK_S,
        settle_s=SETTLE_S,
        post_failure_s=POST_FAILURE_S,
        print_every_steps=1,
    )


# Heavy-current cells: current from vessel+45 deg compass = 225, Vc = 1.0 m/s.
_BF4_CURR_COMPASS = (VESSEL_HEADING_COMPASS + 45.0) % 360.0  # = 225
_BF4_CURR_SPEED = 1.0

# CELLS[tag] = (theta_rel_deg, env_dict, wind_offset_deg,
#               current_compass_abs_or_None, current_speed_override_or_None).
CELLS: dict[str, tuple[float, dict, float, float | None, float | None]] = {
    # Co-linear matrix.
    "bf6_h0":      (0.0,  BF6, 0.0,  None, None),
    "bf6_q10":     (10.0, BF6, 0.0,  None, None),
    "bf8_h0":      (0.0,  BF8, 0.0,  None, None),
    "bf8_q10":     (10.0, BF8, 0.0,  None, None),
    # 45-deg wind/wave split (wind veers right of waves; current co-linear).
    "bf6_h0_w45":  (0.0,  BF6, 45.0, None, None),
    "bf6_q10_w45": (10.0, BF6, 45.0, None, None),
    "bf8_h0_w45":  (0.0,  BF8, 45.0, None, None),
    "bf8_q10_w45": (10.0, BF8, 45.0, None, None),
    # Half-step BF (sec.12.21.21.30b): probe the headroom gradient
    # at the operational boundary. Both cells inherit the worst-
    # direction-combo geometry (q10_w45) from bf8_q10_w45.
    "bf7p5_q10_w45": (10.0, BF7P5, 45.0, None, None),
    "bf8p5_q10_w45": (10.0, BF8P5, 45.0, None, None),
    # Bf 4 + heavy current from vessel+45 deg (low-variability check).
    "bf4_c1_h0":   (0.0,  BF4, 0.0,  _BF4_CURR_COMPASS, _BF4_CURR_SPEED),
    "bf4_c1_q10":  (10.0, BF4, 0.0,  _BF4_CURR_COMPASS, _BF4_CURR_SPEED),
}


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags",
                   default="bf6_h0_w45,bf6_q10_w45,bf8_h0_w45,bf8_q10_w45,"
                           "bf4_c1_h0,bf4_c1_q10",
                   help="Comma-separated list of cell tags to (re)generate. "
                        f"Known: {','.join(CELLS.keys())}. "
                        "Default: the 4 split-direction cells plus the 2 "
                        "Bf 4 + heavy-current cells.")
    p.add_argument("--n-seeds", type=int, default=N_SEEDS,
                   help=f"Seeds per cell (default {N_SEEDS}).")
    p.add_argument("--n-workers", type=int, default=None,
                   help="Process pool size (default = cpu_count // 2).")
    return p.parse_args()


def main():
    args = _parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    bad = [t for t in tags if t not in CELLS]
    if bad:
        sys.exit(f"Unknown tags: {bad}. Known: {list(CELLS.keys())}")

    print("=" * 72)
    print("brucon validation-matrix ensemble runs")
    print("=" * 72)
    print(f"  vessel heading (compass) = {VESSEL_HEADING_COMPASS:.1f} deg")
    print(f"  WCFDI                    = bus_port (Bow1 + PortMP)")
    print(f"  N seeds per cell         = {args.n_seeds}")
    print(f"  cells                    = {tags}")
    print()

    for tag in tags:
        theta_rel, env, wind_off, curr_compass_abs, curr_speed_ovr = CELLS[tag]
        spec = _make_spec(env, theta_rel, wind_offset_deg=wind_off,
                          current_compass_abs=curr_compass_abs,
                          current_speed_override=curr_speed_ovr)
        wave_compass = (VESSEL_HEADING_COMPASS + theta_rel) % 360.0
        wind_compass = (wave_compass + wind_off) % 360.0
        curr_compass = (curr_compass_abs if curr_compass_abs is not None
                        else wave_compass)
        curr_speed = (curr_speed_ovr if curr_speed_ovr is not None
                      else env["Vc"])
        print("-" * 72)
        print(f"[{tag}]  theta_rel = {theta_rel:+.1f} deg  "
              f"(wave from {wave_compass:.1f}, wind from {wind_compass:.1f}, "
              f"current from {curr_compass:.1f} compass)")
        print(f"  Hs = {env['Hs']:.2f} m, Tp = {env['Tp']:.2f} s, "
              f"Vw = {env['Vw']:.2f} m/s, Vc = {curr_speed:.2f} m/s")
        t0 = time.time()
        run_ensemble(
            spec, n_seeds=args.n_seeds, work_dir=WORK_DIR, tag=tag,
            base_seed=BASE_SEED, n_workers=args.n_workers,
        )
        dt = time.time() - t0
        sim_seconds = args.n_seeds * spec.total_seconds
        print(f"  -> {args.n_seeds} seeds in {dt:.1f} s wall  "
              f"({sim_seconds / dt:.0f}x realtime aggregate)")

    print()
    print("Done. Validate any cell with e.g.:")
    print(f"  .venv/bin/python scripts/p7_brucon_validation/"
          f"peak_R_b_hat_sigma_pwq30.py --tag {tags[0]}")
    print(f"  .venv/bin/python scripts/p7_brucon_validation/"
          f"where_we_are_now_pwq30.py --tag {tags[0]}")


if __name__ == "__main__":
    main()
