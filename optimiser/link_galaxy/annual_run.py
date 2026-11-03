"""Annual voyage-comparison for MV Link Galaxy.

Mirrors ../voyage_comparison.py but swaps the vessel-parameter overlay
``models.vessel_link_galaxy`` in place of ``models.constants`` before
any downstream module loads.  See ``models/vessel_link_galaxy.py``
for the trick.

Steps
-----
1. Ensure NORA3 data is downloaded (link_galaxy/download_hindcast.py).
2. Build the propeller open-water file if needed (defaults use the
   Aas206 open-water .dat -- adequate for a C4/40 proxy at EAR 0.435).
3. Run the annual sweep.

Usage::

    python link_galaxy/download_hindcast.py --year 2024
    python link_galaxy/annual_run.py --year 2024 [--plot]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the top-level `models`, `simulation`, `reporting`, `plotting`
# packages importable.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _install_link_galaxy_constants() -> None:
    """Swap models.constants -> models.vessel_link_galaxy.

    Must be called before any module that does
    ``from models.constants import ...`` is imported.
    """
    import models
    import models.vessel_link_galaxy as lg          # noqa: E402
    sys.modules["models.constants"] = lg
    # models.__init__ already ran and set models.constants as an
    # attribute of the package (pointing at the original module).
    # Patch that too so downstream ``import models.constants`` and
    # ``from models import constants`` also see the overlay.
    models.constants = lg

    # Verify the swap: reload key downstream modules only if they got
    # imported earlier.  For the driver's cold-start case this is a no-op.
    stale = [name for name in list(sys.modules)
             if name.startswith("simulation.") or name in {
                 "models.combinator", "models.roughness",
                 "models.wind_resistance", "models.flettner"}]
    for name in stale:
        del sys.modules[name]


_install_link_galaxy_constants()


# Only NOW may we import anything that pulls constants transitively.
from models.route import ROUTE_LINK_GALAXY_ROUNDTRIP     # noqa: E402
from models import constants as _c                       # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--speed", type=float, default=12.5,
                   help="Nominal transit speed [kn] (default: 12.5).")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--as-commissioned", metavar="PATH", default=None,
                   help="Use operator combinator table from a brucon "
                        "gear_control_tables_4.prototxt.in as baseline "
                        "(instead of synthesised design combinator).")
    p.add_argument("--as-commissioned-block", default="main",
                   help="Block name inside the prototxt (default: main; "
                        "first combinator table = main.combinator.positive).")
    p.add_argument("--sg-load", type=float, default=200.0,
                   help="Shaft-generator electrical load [kW] (default 200).")
    p.add_argument("--no-sg-floor", action="store_true",
                   help="Do not apply the SG minimum engine speed "
                        "(gear_control generator.min_speed, 69 %% of 720 rpm).")
    args = p.parse_args()

    print("Link Galaxy annual voyage comparison")
    print(f"  route waypoints : {len(ROUTE_LINK_GALAXY_ROUNDTRIP)}")
    print(f"  pdstrip.dat     : {_c.PDSTRIP_DAT}")
    print(f"  hindcast dir    : {_c.NORA3_DATA_DIR}")
    print(f"  design speed    : {args.speed} kn")
    print(f"  Lpp / B / T     : {_c.HULL_LWL} / {_c.HULL_B} / "
          f"{_c.HULL_T_MEAN} m")
    print(f"  displacement    : {_c.HULL_DISPLACEMENT_M3:.0f} m^3")
    print(f"  D_prop / P/D    : {_c.PROP_DIAMETER} / {_c.PROP_DESIGN_PITCH}")
    print(f"  gear ratio      : {_c.GEAR_RATIO}")

    # -- Sanity: confirm the overlay is active ---------------------------
    assert abs(_c.PROP_DIAMETER - 4.30) < 1e-6, (
        "constants overlay not active -- got PROP_DIAMETER="
        f"{_c.PROP_DIAMETER}")

    # -- Annual sweep ---------------------------------------------------
    # Delayed imports: everything below transitively depends on
    # models.constants, which we swapped for the LG overlay above.
    #
    # Engine: models.constants.ENGINE_FACTORY -> make_wartsila_vasa32_16v
    # (port of the brucon Link Galaxy engine config).
    from simulation.orchestrator import run_annual_comparison  # noqa: E402
    from reporting.summary import print_summary                # noqa: E402

    results = run_annual_comparison(
        year=args.year,
        speed_kn=args.speed,
        waypoints=ROUTE_LINK_GALAXY_ROUNDTRIP,
        data_dir=_c.NORA3_DATA_DIR,
        pdstrip_path=_c.PDSTRIP_DAT,
        flettner_enabled=False,     # no rotor fitted on LG
        sg_load_kw=args.sg_load,
        sg_factory_allowance_kw=args.sg_load,
        # gear_control holds the engine at >= generator.min_speed (69 %
        # of 720 rpm = 497 rpm) while the SG is engaged -- confirmed by the
        # commissioning log (shaft floor 79.1 rpm).  Applies to both sides.
        sg_freq_min=0.0 if args.no_sg_floor else _c.SG_MIN_SPEED_PCT,
        sg_freq_max=0.0 if args.no_sg_floor else 100.0,
        verbose=not args.quiet,
        round_trip=False,           # LG route already returns to origin
        as_commissioned_prototxt=args.as_commissioned,
        as_commissioned_block=args.as_commissioned_block,
    )
    print_summary(results, args.speed, round_trip=False)

    if args.plot:
        from plotting.comparison import plot_results        # noqa: E402
        plot_results(results)


if __name__ == "__main__":
    main()
