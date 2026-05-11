"""Plot brucon's logged environmental forces across the full 740 s run.

For each cell, plot per-seed WindX/Y, DriftX/Y, CurX/Y (kN) and
HeadingDev (deg) vs time, with a vertical line at T_WCF = 560 s.

Goal: visually answer whether there is a STEP or sustained DRIFT in any
env-force channel at the WCF event, separate from the natural slow
fluctuation that goes on across the whole 740 s.

Usage:
  .venv/bin/python scripts/p7_brucon_validation/plot_env_forces_full_run.py \
      --tags bf8_h0,bf8_h0_w45,bf8_q10,pwo
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent.parent))

from harness import parse_output  # noqa: E402

ENSEMBLE_DIR = THIS / "work"
SEEDS = list(range(1000, 1030))
T_WCF_S = 560.0


def load_seed(tag: str, seed: int):
    seed_dir = ENSEMBLE_DIR / f"{tag}_seed{seed:04d}"
    out_path = seed_dir / f"{tag}_seed{seed:04d}.out"
    if not out_path.exists():
        return None
    res = parse_output(out_path)
    cols = res.columns
    return dict(
        t=cols["t"],
        WindX=cols["WindX"], WindY=cols["WindY"], WindMz=cols["WindMz"],
        DriftX=cols["DriftX"], DriftY=cols["DriftY"], DriftMz=cols["DriftMz"],
        CurX=cols["CurX"], CurY=cols["CurY"], CurMz=cols["CurMz"],
        HeadingDev=cols["HeadingDev"],
    )


def plot_cell(tag: str):
    data = []
    for s in SEEDS:
        d = load_seed(tag, s)
        if d is not None:
            data.append(d)
    if not data:
        print(f"[{tag}] no data")
        return
    n = len(data)
    print(f"[{tag}] loaded {n} seeds")

    # Common time grid (use seed-0 t).
    t = data[0]["t"]
    t_lo, t_hi = float(t[0]), float(t[-1])

    # Resample everything onto seed-0 t (most seeds match exactly).
    def stack(key):
        return np.array([np.interp(t, d["t"], d[key]) for d in data])

    chans = [
        ("WindX",  "WindX [kN]"),
        ("WindY",  "WindY [kN]"),
        ("WindMz", "WindMz [kNm]"),
        ("DriftX", "DriftX [kN]"),
        ("DriftY", "DriftY [kN]"),
        ("DriftMz", "DriftMz [kNm]"),
        ("CurX",   "CurX [kN]"),
        ("CurY",   "CurY [kN]"),
        ("HeadingDev", "HeadingDev [deg]"),
    ]
    fig, axes = plt.subplots(len(chans), 1, figsize=(11, 14), sharex=True)
    for ax, (key, label) in zip(axes, chans):
        Y = stack(key)
        # per-seed grey lines
        for k in range(n):
            ax.plot(t, Y[k], color="grey", alpha=0.18, lw=0.5)
        # ensemble mean thick
        ax.plot(t, Y.mean(axis=0), "C0-", lw=1.5, label=f"mean (n={n})")
        ax.axvline(T_WCF_S, color="r", lw=0.8, alpha=0.6, label="T_WCF")
        ax.axhline(0, color="k", lw=0.3)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
        if key == chans[0][0]:
            ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("t [s]")
    fig.suptitle(f"Full-run env-force time series  [{tag}]", y=1.00)
    plt.tight_layout()
    out_png = THIS / f"plot_env_forces_full_run_{tag}.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_png}")

    # Print summary: ensemble-mean value at a few key time samples.
    def smean(key, t0, t1):
        Y = stack(key)
        m = (t >= t0) & (t <= t1)
        return float(Y[:, m].mean())

    windows = [(380, 440), (440, 500), (499, 559), (565, 625), (625, 680)]
    print(f"  ensemble-mean window values (kN, kNm, deg):")
    print(f"  {'channel':12s}  " +
          "  ".join(f"[{a:3d},{b:3d}]" for a, b in windows))
    for key, _ in chans:
        vals = [smean(key, a, b) for a, b in windows]
        print(f"  {key:12s}  " +
              "  ".join(f"{v:+9.2f}" for v in vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="bf8_h0,bf8_h0_w45,bf8_q10,pwo")
    args = ap.parse_args()
    for tag in [t.strip() for t in args.tags.split(",") if t.strip()]:
        plot_cell(tag)


if __name__ == "__main__":
    main()
