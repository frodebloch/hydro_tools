"""P7 smoke test: drive RunFastStandalone for one short scenario and sanity-
check the resulting position trace.

Run from cqa/scripts/p7_brucon_validation/:
    /home/blofro/src/hydro_tools/cqa/.venv/bin/python smoke_test.py

Expected behaviour:
- Simulator runs without error.
- Output DataFrame has the expected columns (t, x, y, ...).
- During the settle phase position deviation stays small (< ~5 m).
- After failure (Bus port: Bow1 + PortMP) position drifts.
- Drift direction: with weather coming from 270 deg (west) onto a vessel
  heading 180 deg (bow south, beam port to weather), the wave pushes the
  vessel east. With Bow1 + PortMP lost, transverse + yaw capacity is
  asymmetrically reduced and we expect a *non-zero* y excursion (sway).
  We don't enforce a sign or magnitude here -- just that the post-failure
  excursion is visibly larger than the pre-failure noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import ScenarioSpec, run_simulation, CSOV_WCF_GROUPS, SIM_DT


def main() -> None:
    spec = ScenarioSpec(
        Hs=4.0, Tp=8.5, wave_dir_compass=270.0,
        wind_speed=14.0, wind_dir_compass=270.0,
        current_speed=0.5, current_dir_compass=270.0,
        vessel_heading_compass=180.0,
        failed_thruster_indices=CSOV_WCF_GROUPS["bus_port"],
        # Short scenario for the smoke test (production runs use longer settle).
        # 60 s precondition (vessel held still while sensors settle), then
        # 60 s of free intact-DP, then 60 s post-WCFDI.
        activate_sk_s=60.0,
        settle_s=60.0,
        post_failure_s=60.0,
        print_every_steps=10,        # 1 Hz output
    )

    work_dir = Path(__file__).resolve().parent / "work"
    print(f"Smoke test: scenario total = {spec.total_seconds:.0f} s")
    print(f"  precondition: 0 -> {spec.activate_sk_s:.0f} s (vessel held still)")
    print(f"  intact DP:    {spec.activate_sk_s:.0f} -> {spec.failure_time_s:.0f} s")
    print(f"  post-WCFDI:   {spec.failure_time_s:.0f} -> {spec.total_seconds:.0f} s")
    print(f"  Work dir: {work_dir}")

    df = run_simulation(spec, seed=1, work_dir=work_dir, tag="smoke")

    print(f"  Output rows: {df.n_rows}")
    print(f"  Columns ({len(df.column_names())}): "
          f"{', '.join(df.column_names()[:10])} ...")

    # Sanity checks on basic columns.
    for required in ("t", "x", "y", "heading"):
        if required not in df:
            raise SystemExit(f"FAIL: column '{required}' not in simulator output")

    t = df["t"]
    x = df["x"]
    y = df["y"]

    # Three windows: precondition, free-intact-DP, post-WCFDI.
    # Skip a small margin around each transition (DP-engagement transient,
    # immediate WCFDI shock).
    pre_mask = t < spec.activate_sk_s - 1.0
    intact_mask = (t > spec.activate_sk_s + 10.0) & (t < spec.failure_time_s - 1.0)
    post_mask = t > spec.failure_time_s + 10.0

    pre_pos = np.hypot(x[pre_mask], y[pre_mask])
    intact_pos = np.hypot(x[intact_mask], y[intact_mask])
    post_pos = np.hypot(x[post_mask], y[post_mask])

    print(f"  Precondition  |pos|: max = {pre_pos.max() if pre_pos.size else float('nan'):.2f} m, "
          f"mean = {pre_pos.mean() if pre_pos.size else float('nan'):.2f} m  "
          f"(expect ~0 with SetFixedCourseAndSpeed)")
    print(f"  Free intact-DP |pos|: max = {intact_pos.max() if intact_pos.size else float('nan'):.2f} m, "
          f"mean = {intact_pos.mean() if intact_pos.size else float('nan'):.2f} m")
    print(f"  Post-failure   |pos|: max = {post_pos.max():.2f} m, "
          f"mean = {post_pos.mean():.2f} m")
    print(f"  Final position: x = {x[-1]:.2f} m, y = {y[-1]:.2f} m")

    if pre_pos.size and pre_pos.max() > 2.0:
        print("  WARN: vessel drifted >2 m during precondition window. "
              "Check SetFixedCourseAndSpeed timing.")
    else:
        print("  OK: precondition holds vessel still (initial 1-step "
              "transient < 2 m is a sample-timing artefact).")

    if not (post_pos.max() > intact_pos.max() + 0.5):
        print("  WARN: post-failure excursion not visibly larger than intact. "
              "Check thruster indexing / failure injection.")
    else:
        print("  OK: WCFDI produces visible position drift.")


if __name__ == "__main__":
    main()
