"""Single-seed smoke test for the LOCKED-Tp shadow-config mechanism."""
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from harness import CSOV_CONFIG, ScenarioSpec, run_simulation
from long_run_locked_tp_validation import (
    HS, TP, WAVE_DIR_COMPASS, VESSEL_HEADING_COMPASS, CSOV_WCF_BUS_PORT,
    LOCKED_TP_S, build_observer_override, get_brucon_sway_lf,
    read_estimator_tp,
)
import numpy as np

# Short run: settle 600s + post 60s + activate 60s = 720s sim ~ 8s wall
spec = ScenarioSpec(
    Hs=HS, Tp=TP, wave_dir_compass=WAVE_DIR_COMPASS,
    wind_speed=0.0, wind_dir_compass=WAVE_DIR_COMPASS,
    current_speed=0.0, current_dir_compass=WAVE_DIR_COMPASS,
    vessel_heading_compass=VESSEL_HEADING_COMPASS,
    failed_thruster_indices=CSOV_WCF_BUS_PORT,
    activate_sk_s=60.0, settle_s=600.0, post_failure_s=60.0,
    print_every_steps=1,
    config_overrides={"observer.prototxt": build_observer_override()},
)

work = THIS / "work_smoke_lockedTp"
print(f"running 1 seed in {work} ...")
t0 = time.time()
result = run_simulation(spec, seed=1000, work_dir=work, tag="smoke")
print(f"  done in {time.time() - t0:.1f}s wall")

# Verify shadow dir exists with override file
shadow = work / "smoke_seed1000" / "config_shadow"
assert shadow.exists(), f"shadow dir missing: {shadow}"
assert (shadow / "observer.prototxt").is_file() and not (shadow / "observer.prototxt").is_symlink(), \
    "observer.prototxt should be an overridden file, not symlink"
n_links = sum(1 for p in shadow.iterdir() if p.is_symlink())
n_files = sum(1 for p in shadow.iterdir() if p.is_file() and not p.is_symlink())
n_orig = sum(1 for p in CSOV_CONFIG.iterdir())
print(f"  shadow contents: {n_links} symlinks + {n_files} files (orig has {n_orig} entries)")
assert n_links + n_files == n_orig, "shadow dir entry count mismatch"

# Verify the override file actually contains the LOCKED block
ovr = (shadow / "observer.prototxt").read_text()
assert "LOCKED" in ovr and f"locked_peak_period: {LOCKED_TP_S}" in ovr, \
    "override file is missing the LOCKED block"
print("  shadow override content OK")

# Verify Tp readback
mean_tp, std_tp = read_estimator_tp(work / "smoke_seed1000", 100.0, 660.0)
print(f"  EstWavePeriodPitch in window [100, 660] s: mean={mean_tp:.3f} s, std={std_tp:.4f} s")
print(f"  (This is the pitch ESTIMATOR output; the wave filter is using {LOCKED_TP_S} s regardless.)")

sigma = get_brucon_sway_lf(result, 300.0, 660.0)
print(f"  sigma_y_LF in [300, 660] s: {sigma:.3f} m")
print()
print("smoke OK -- shadow-config mechanism works")
