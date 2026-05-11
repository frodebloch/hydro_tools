"""Read brucon's logged environmental force time-series and report the
ensemble-mean change post-WCF for wind / drift / current.

The previous diagnostic (brucon_drift_heading_coupling.py) reconstructed
ΔF from the heading deviation through a pdstrip QTF LUT. That is an
approximation -- and it only models wave-drift sensitivity to heading,
not wind or current angular sensitivity.

Brucon already logs the actual forces and moments per timestep:
  WindX, WindY, WindMz       (wind, body-frame)
  DriftX, DriftY, DriftMz    (mean wave-drift, body-frame)
  CurX,  CurY,  CurMz        (current, body-frame)

So the cleanest test of the user's hypothesis is:
  for each cell, report ensemble-mean Δ(WindX), Δ(WindY), Δ(DriftX),
  Δ(DriftY), Δ(CurX), Δ(CurY) post-WCF, baseline = pre-WCF window mean.
  If the SUM ΔF body-frame is comparable to b̂ magnitudes (~50-200 kN)
  then the heading-induced env-force change IS a real channel that the
  cqa frozen-tau_env model misses. If not, hypothesis is falsified.

Usage:
  .venv/bin/python scripts/p7_brucon_validation/brucon_force_change_post_wcf.py \
      --tags bf8_h0,bf8_q10,bf8_h0_w45,bf8_q10_w45,bf6_h0,bf6_q10,pwo,pwq30
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
_REPO_ROOT = str(THIS.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from harness import parse_output  # noqa: E402

ENSEMBLE_DIR = THIS / "work"
SEEDS = list(range(1000, 1030))
T_WCF_S = 560.0
WIN_PRE_END = T_WCF_S - 1.0
WIN_PRE_START = WIN_PRE_END - 60.0
WIN_POST_START = T_WCF_S + 5.0
WIN_POST_END = T_WCF_S + 120.0


def load_seed(tag: str, seed: int):
    seed_dir = ENSEMBLE_DIR / f"{tag}_seed{seed:04d}"
    out_path = seed_dir / f"{tag}_seed{seed:04d}.out"
    if not out_path.exists():
        return None
    res = parse_output(out_path)
    t = res.columns["t"]
    if t[-1] < WIN_POST_END:
        return None
    win_pre = (t >= WIN_PRE_START) & (t <= WIN_PRE_END)
    win_post = (t >= WIN_POST_START) & (t <= WIN_POST_END)

    def chan_delta(name: str) -> tuple[float, float]:
        """Return (mean_pre, mean_post) for a column."""
        v = res.columns[name]
        return float(np.mean(v[win_pre])), float(np.mean(v[win_post]))

    pre_d, post_d = {}, {}
    for c in ("WindX", "WindY", "WindMz",
              "DriftX", "DriftY", "DriftMz",
              "CurX", "CurY", "CurMz",
              "HeadingDev"):
        pre_d[c], post_d[c] = chan_delta(c)
    return pre_d, post_d


def cell_table(tag: str):
    seeds_data = []
    for s in SEEDS:
        d = load_seed(tag, s)
        if d is not None:
            seeds_data.append(d)
    if not seeds_data:
        print(f"[{tag}] no seed data")
        return
    n = len(seeds_data)
    chans = ("WindX", "WindY", "WindMz",
             "DriftX", "DriftY", "DriftMz",
             "CurX", "CurY", "CurMz",
             "HeadingDev")
    delta_mean = {}
    delta_p95 = {}
    for c in chans:
        d = np.array([post[c] - pre[c] for pre, post in seeds_data])
        delta_mean[c] = float(np.mean(d))
        delta_p95[c] = float(np.quantile(np.abs(d), 0.95))

    print(f"\n[{tag}]  n_seeds = {n}")
    print(f"  Window pre  = [{WIN_PRE_START:.1f}, {WIN_PRE_END:.1f}] s")
    print(f"  Window post = [{WIN_POST_START:.1f}, {WIN_POST_END:.1f}] s")
    print(f"  HeadingDev change (deg): mean = {delta_mean['HeadingDev']:+.2f}  "
          f"per-seed |Δ| P95 = {delta_p95['HeadingDev']:.2f}")
    print(f"  ---  Wind  ---")
    print(f"    ΔWindX  : mean = {delta_mean['WindX']:+8.2f} kN   "
          f"|Δ|P95 = {delta_p95['WindX']:6.2f} kN")
    print(f"    ΔWindY  : mean = {delta_mean['WindY']:+8.2f} kN   "
          f"|Δ|P95 = {delta_p95['WindY']:6.2f} kN")
    print(f"    ΔWindMz : mean = {delta_mean['WindMz']:+8.2f} kNm  "
          f"|Δ|P95 = {delta_p95['WindMz']:6.2f} kNm")
    print(f"  ---  Drift  ---")
    print(f"    ΔDriftX : mean = {delta_mean['DriftX']:+8.2f} kN   "
          f"|Δ|P95 = {delta_p95['DriftX']:6.2f} kN")
    print(f"    ΔDriftY : mean = {delta_mean['DriftY']:+8.2f} kN   "
          f"|Δ|P95 = {delta_p95['DriftY']:6.2f} kN")
    print(f"    ΔDriftMz: mean = {delta_mean['DriftMz']:+8.2f} kNm  "
          f"|Δ|P95 = {delta_p95['DriftMz']:6.2f} kNm")
    print(f"  ---  Current  ---")
    print(f"    ΔCurX   : mean = {delta_mean['CurX']:+8.2f} kN   "
          f"|Δ|P95 = {delta_p95['CurX']:6.2f} kN")
    print(f"    ΔCurY   : mean = {delta_mean['CurY']:+8.2f} kN   "
          f"|Δ|P95 = {delta_p95['CurY']:6.2f} kN")
    print(f"    ΔCurMz  : mean = {delta_mean['CurMz']:+8.2f} kNm  "
          f"|Δ|P95 = {delta_p95['CurMz']:6.2f} kNm")
    sumX = (delta_mean['WindX'] + delta_mean['DriftX'] + delta_mean['CurX'])
    sumY = (delta_mean['WindY'] + delta_mean['DriftY'] + delta_mean['CurY'])
    sumMz = (delta_mean['WindMz'] + delta_mean['DriftMz'] + delta_mean['CurMz'])
    print(f"  ---  Sum (mean)  ---")
    print(f"    ΣΔF body-frame  : surge = {sumX:+8.2f} kN   "
          f"sway = {sumY:+8.2f} kN   yaw = {sumMz:+8.2f} kNm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="bf8_h0,bf8_q10,bf8_h0_w45,bf8_q10_w45,"
                    "bf6_h0,bf6_q10,pwo,pwq30")
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    for tag in tags:
        cell_table(tag)


if __name__ == "__main__":
    main()
