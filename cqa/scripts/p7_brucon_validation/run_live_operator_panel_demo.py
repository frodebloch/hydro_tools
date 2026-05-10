"""Operator-panel demo: build a `LiveOperatorSummary` from a single
brucon seed at the pwq30 cell and render the two-bar panel as a PNG.

This is the first end-to-end visual of the LIVE operational CQA
pipeline driving the operator-facing summary in
``cqa.live_operator_view``. It mirrors the per-seed setup from
``live_cell_per_seed_pwq30.py`` (load brucon log, snapshot the live
observer state at ``T_WCF - 5 s``, build a LiveSigmaPosterior from
the pre-WCF window) but instead of computing a per-seed envelope it
calls ``summarise_for_operator_live`` and saves the operator panel.

Run with::

    PYTHONPATH=. .venv/bin/python \\
        scripts/p7_brucon_validation/run_live_operator_panel_demo.py

Output: ``run_live_operator_panel_demo.png`` in this directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))

# Reuse the pwq30 per-seed pipeline helpers verbatim.
sys.path.insert(0, str(THIS))
from live_cell_per_seed_pwq30 import (   # noqa: E402  (path tweak above)
    load_seed,
    build_live_sigma_posterior,
)

from cqa.config import csov_default_config                       # noqa: E402
from cqa.live_decision import LiveObserverState                  # noqa: E402
from cqa.live_operator_view import (                             # noqa: E402
    summarise_for_operator_live,
    plot_live_operator_summary,
)


SEED = 1000
CALIB_NPZ = THIS / "scenario_pwq30_calibration.npz"


def main() -> int:
    if not CALIB_NPZ.exists():
        print(f"Missing calibration artefact: {CALIB_NPZ}", file=sys.stderr)
        print("Run scripts/p7_brucon_validation/peak_R_b_hat_sigma_pwq30.py "
              "first.", file=sys.stderr)
        return 1

    import numpy as np
    sigma_R_b_hat_m = float(np.load(CALIB_NPZ, allow_pickle=True)["sigma_R_b_hat_m"])

    d = load_seed(SEED)
    if d is None:
        print(f"Seed {SEED} log missing or truncated.", file=sys.stderr)
        return 1

    cfg = csov_default_config()
    sigma_post = build_live_sigma_posterior(d, sigma_R_b_hat_m=sigma_R_b_hat_m)

    obs = LiveObserverState(
        eta_hat=d["eta_hat"],
        nu_hat=d["nu_hat"],
        b_hat=d["b_hat"],
        eta_wave=d["eta_wave"],
        heading_compass=float(d.get("heading_compass", 0.0)),
    )

    summary = summarise_for_operator_live(cfg, obs, sigma_post)

    print(f"Live operator summary @ pwq30 seed {SEED}")
    print(f"  intact: P50 = {summary.intact_R_p50:.2f} m, "
          f"P95 = {summary.intact_R_p95:.2f} m, "
          f"offset = {summary.intact_R_offset_m:.2f} m, "
          f"traffic = {summary.intact_traffic}")
    print(f"  WCF   : P50 = {summary.wcf_R_p50:.2f} m, "
          f"P95 = {summary.wcf_R_p95:.2f} m, "
          f"peak in {summary.wcf_t_peak_s:.0f} s, "
          f"offset@peak = {summary.wcf_R_offset_at_peak_m:.2f} m, "
          f"traffic = {summary.wcf_traffic}")
    print(f"  sigma_R_intact = {summary.sigma_R_intact_m:.2f} m   "
          f"sigma_R_wcf = {summary.sigma_R_wcf_m:.2f} m   "
          f"overall = {summary.overall_traffic.upper()}")

    fig = plot_live_operator_summary(summary)
    out = THIS / f"run_live_operator_panel_demo_seed{SEED}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
