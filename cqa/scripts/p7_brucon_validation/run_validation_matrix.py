"""Generate the brucon validation-matrix ensembles.

Runs four cells of {head-on, Q10} x {Bf 6 (nominal), Bf 8 (extreme)},
all with co-linear wind/wave/current per DNV-ST-0111 Table (provided
by the user 2026-05):

    Bf 6  (nominal): Vw = 13.8 m/s, Hs = 3.1 m, Tp = 8.5 s, Vc = 0.75 m/s
    Bf 8 (extreme): Vw = 20.7 m/s, Hs = 5.7 m, Tp = 10.0 s, Vc = 0.75 m/s

WCFDI: bus_port (Bow1 + PortMP), matching pwq30 / pwo.

Tags written under work/<TAG>_seed{1000..1029}/:

    bf6_h0    head-on,   Bf 6
    bf6_q10   theta=+10, Bf 6
    bf8_h0    head-on,   Bf 8
    bf8_q10   theta=+10, Bf 8

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

WORK_DIR = THIS / "work"
N_SEEDS = 30
BASE_SEED = 1000

# Vessel heading: bow into 180 deg compass (matches pwq30 / pwo).
VESSEL_HEADING_COMPASS = 180.0

# WCFDI failure: bus_port (Bow1 + PortMP). Matches pwq30 / pwo.
FAILED_THRUSTERS = CSOV_WCF_GROUPS["bus_port"]

# Lua run timings (must match pwq30 / pwo so T_WCF = 560 s in all cells,
# and the validation scripts pick up the data correctly without further
# parameterisation):
ACTIVATE_SK_S = 60.0      # precondition window before SK activates
SETTLE_S = 500.0          # intact-DP window before WCFDI
POST_FAILURE_S = 180.0    # WCFDI transient window
# -> failure_time = 60 + 500 = 560 s, total = 740 s (matches pwq30 lua).

# DNV-ST-0111 Beaufort table (image provided 2026-05).
BF6 = dict(Vw=13.8, Hs=3.1, Tp=8.5, Vc=0.75)
BF8 = dict(Vw=20.7, Hs=5.7, Tp=10.0, Vc=0.75)


def _make_spec(env: dict, theta_rel_deg: float) -> ScenarioSpec:
    """Build a ScenarioSpec with co-linear wind/wave/current.

    theta_rel_deg = (wave_compass - vessel_heading_compass) wrapped to
    [-180, 180]. We hold vessel heading at 180 deg, so wave_compass =
    180 + theta_rel_deg.
    """
    wave_compass = (VESSEL_HEADING_COMPASS + theta_rel_deg) % 360.0
    return ScenarioSpec(
        Hs=env["Hs"], Tp=env["Tp"], wave_dir_compass=wave_compass,
        wind_speed=env["Vw"], wind_dir_compass=wave_compass,
        current_speed=env["Vc"], current_dir_compass=wave_compass,
        vessel_heading_compass=VESSEL_HEADING_COMPASS,
        failed_thruster_indices=FAILED_THRUSTERS,
        activate_sk_s=ACTIVATE_SK_S,
        settle_s=SETTLE_S,
        post_failure_s=POST_FAILURE_S,
        print_every_steps=1,
    )


# (tag, theta_rel_deg, env_dict).
CELLS: dict[str, tuple[float, dict]] = {
    "bf6_h0":  (0.0,  BF6),
    "bf6_q10": (10.0, BF6),
    "bf8_h0":  (0.0,  BF8),
    "bf8_q10": (10.0, BF8),
}


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", default=",".join(CELLS.keys()),
                   help="Comma-separated list of cell tags to (re)generate. "
                        "Default: all four.")
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
        theta_rel, env = CELLS[tag]
        spec = _make_spec(env, theta_rel)
        wave_compass = (VESSEL_HEADING_COMPASS + theta_rel) % 360.0
        print("-" * 72)
        print(f"[{tag}]  theta_rel = {theta_rel:+.1f} deg  "
              f"(wave from {wave_compass:.1f} compass)")
        print(f"  Hs = {env['Hs']:.2f} m, Tp = {env['Tp']:.2f} s, "
              f"Vw = {env['Vw']:.2f} m/s, Vc = {env['Vc']:.2f} m/s "
              f"(co-linear)")
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
