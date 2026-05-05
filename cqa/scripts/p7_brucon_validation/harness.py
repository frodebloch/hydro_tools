"""Harness for driving brucon RunFastStandalone from Python.

Usage outline (high level)::

    from harness import (
        ScenarioSpec, render_lua, run_simulation, parse_output, run_ensemble
    )

    spec = ScenarioSpec(
        Hs=4.0, Tp=8.5, wave_dir_compass=270.0,
        wind_speed=14.0, wind_dir_compass=270.0,
        current_speed=0.5, current_dir_compass=270.0,
        vessel_heading_compass=180.0,            # bow into 180 -> beam to wave
        failed_thruster_indices=(0, 3),          # Bus 1 = Bow1 + PortMP
        settle_s=120.0, post_failure_s=120.0,
    )
    result = run_simulation(spec, seed=42, work_dir="work")
    ensemble = run_ensemble(spec, n_seeds=30, work_dir="work")

The harness writes one .lua + one .out per seed under ``work_dir``, then
parses the tab-separated output produced by RunFastStandalone into a
``SimResult`` (column-name -> 1-D numpy array). We deliberately avoid pandas
to keep dependency parity with the rest of cqa (numpy + scipy + matplotlib
only).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Hard-coded paths to brucon. Kept here, not in cqa, because they are part of
# this harness's external dependency.
# ---------------------------------------------------------------------------
BRUCON_BIN = Path.home() / "src" / "brucon" / "build" / "bin"
RUNFAST = BRUCON_BIN / "RunFastStandalone"
CSOV_CONFIG = BRUCON_BIN / "config_csov"

# Vessel-response (pdstrip) data lives in a separate directory in bin/.
# RunFastStandalone takes:
#   -r <response_data_file_name>          (default: "pdstrip.dat", a generic stub)
#   --vessel-simulator-config <dir>       (default: relative "vessel_simulator_config")
# and joins them. We must pass *both* explicitly, otherwise the simulator
# silently logs an ERROR and falls back to whatever it can find -- which is
# the behaviour we want to avoid for cross-validation.
CSOV_RESPONSE_FILE = "csov_pdstrip.dat"
SIMULATOR_CONFIG_DIR = BRUCON_BIN / "vessel_simulator_config"

LUA_TEMPLATE = (
    Path(__file__).resolve().parent / "lua_templates" / "wcfdi_validation.lua.tpl"
)

# Simulator step is 0.1 s (per dp_runfast_simulator README).
SIM_DT = 0.1

# Thruster index map for config_csov (verified from propulsors.prototxt order).
# This is the configuration-as-coded; see the proto file for ground truth.
CSOV_THRUSTER_INDEX = {
    "Bow1": 0,
    "Bow2": 1,
    "BowAz": 2,
    "PortMP": 3,
    "StbdMP": 4,
}

# Worst-case failure groupings on the CSOV (per user input):
# 1. Bus port: Bow1 + PortMP
# 2. Bus stbd: Bow2 + StbdMP
# 3. BowAz alone
CSOV_WCF_GROUPS = {
    "bus_port": (CSOV_THRUSTER_INDEX["Bow1"], CSOV_THRUSTER_INDEX["PortMP"]),
    "bus_stbd": (CSOV_THRUSTER_INDEX["Bow2"], CSOV_THRUSTER_INDEX["StbdMP"]),
    "bow_az":   (CSOV_THRUSTER_INDEX["BowAz"],),
}


@dataclass
class ScenarioSpec:
    """Single (sea state, heading, WCFDI) scenario, deterministic apart from seed."""

    Hs: float                                      # m
    Tp: float                                      # s
    wave_dir_compass: float                        # deg, "from"
    wind_speed: float                              # m/s
    wind_dir_compass: float                        # deg, "from"
    current_speed: float                           # m/s
    current_dir_compass: float                     # deg, "from"
    vessel_heading_compass: float                  # deg
    failed_thruster_indices: Sequence[int]
    settle_s: float = 300.0                        # free intact-DP window (sampled for steady-state stats)
    post_failure_s: float = 180.0                  # post-WCFDI transient window
    activate_sk_s: float = 60.0                    # vessel held still via SetFixedCourseAndSpeed for this many s
    print_every_steps: int = 1                     # PrintDataLine every N sim steps -> 10 Hz output

    @property
    def total_seconds(self) -> float:
        return self.activate_sk_s + self.settle_s + self.post_failure_s

    @property
    def failure_time_s(self) -> float:
        return self.activate_sk_s + self.settle_s


def render_lua(spec: ScenarioSpec, seed: int, output_file: Path,
               estimator_output_file: Path) -> str:
    """Materialise the lua template for one (spec, seed)."""
    template = LUA_TEMPLATE.read_text()

    # Build the failure block: one SetThrusterActive(idx, false) per failed unit.
    failure_lines = "\n".join(
        f"        SetThrusterActive({idx}, false)"
        for idx in spec.failed_thruster_indices
    )

    n_total = int(round(spec.total_seconds / SIM_DT))
    i_release = int(round(spec.activate_sk_s / SIM_DT))    # release SetFixedCourseAndSpeed at end of precondition
    i_sk = i_release + 1                                   # SetStationKeeping immediately on the next step
    i_fail = int(round(spec.failure_time_s / SIM_DT))

    return template.format(
        output_file=str(output_file),
        estimator_output_file=str(estimator_output_file),
        wave_seed=int(seed),
        Hs=spec.Hs,
        Tp=spec.Tp,
        wave_dir_compass=spec.wave_dir_compass,
        wind_speed=spec.wind_speed,
        wind_dir_compass=spec.wind_dir_compass,
        current_speed=spec.current_speed,
        current_dir_compass=spec.current_dir_compass,
        vessel_heading_compass=spec.vessel_heading_compass,
        n_total_steps=n_total,
        i_release_fixed=i_release,
        i_activate_sk=i_sk,
        i_failure=i_fail,
        print_every=spec.print_every_steps,
        failure_lines=failure_lines,
    )


@dataclass
class SimResult:
    """Parsed output of one RunFastStandalone run.

    columns
        Mapping column-name -> 1-D float array, in row order. ``__getitem__``
        and ``__contains__`` are forwarded to ``columns``.
    """
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    source_path: Path | None = None

    def __getitem__(self, key: str) -> np.ndarray:
        return self.columns[key]

    def __contains__(self, key: str) -> bool:
        return key in self.columns

    @property
    def n_rows(self) -> int:
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))

    def column_names(self) -> list[str]:
        return list(self.columns.keys())


def parse_output(out_path: Path) -> SimResult:
    """Parse a RunfastTesting-style tab-separated output file.

    First line is the tab-separated header; subsequent lines are numeric.
    Returns a SimResult with a numpy array per named column.
    """
    if not out_path.exists():
        raise FileNotFoundError(f"Simulator did not produce output: {out_path}")
    with out_path.open("r") as fh:
        header_line = fh.readline().strip()
    headers = header_line.split("\t")
    data = np.loadtxt(out_path, skiprows=1, delimiter="\t", ndmin=2)
    if data.shape[1] != len(headers):
        raise ValueError(
            f"Column-count mismatch in {out_path}: header has "
            f"{len(headers)} fields, data has {data.shape[1]} columns"
        )
    columns = {h: data[:, i] for i, h in enumerate(headers)}
    return SimResult(columns=columns, source_path=out_path)


def run_simulation(spec: ScenarioSpec, seed: int, work_dir: Path,
                   tag: str = "run") -> SimResult:
    """Render lua, invoke RunFastStandalone, parse output. Returns SimResult.

    work_dir
        Per-run artefacts are written to ``work_dir / f"{tag}_seed{seed:04d}/"``.
        The directory is created if it doesn't exist; previous contents are wiped.

    The simulator is invoked from the CSOV config dir as cwd, because the lua
    script writes its output file as a *relative* path -- this matches how the
    bundled dp_operation.lua and dp_runfast_simulator README examples behave.
    """
    work_dir = Path(work_dir).resolve()
    run_dir = work_dir / f"{tag}_seed{seed:04d}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    out_name = f"{tag}_seed{seed:04d}.out"
    est_name = f"{tag}_seed{seed:04d}_estimator.out"
    lua_path = run_dir / f"{tag}_seed{seed:04d}.lua"

    lua_source = render_lua(
        spec, seed=seed,
        output_file=Path(out_name),               # relative -> written into cwd
        estimator_output_file=Path(est_name),
    )
    lua_path.write_text(lua_source)

    # Run with cwd = run_dir so relative output paths in the lua script land here.
    cmd = [
        str(RUNFAST),
        "-c", str(CSOV_CONFIG),
        "-l", str(lua_path),
        "-r", CSOV_RESPONSE_FILE,
        "--vessel-simulator-config", str(SIMULATOR_CONFIG_DIR),
    ]
    proc = subprocess.run(
        cmd, cwd=run_dir, capture_output=True, text=True, check=False,
    )

    # Persist BOTH stdout and stderr. brucon's Qt logger writes its
    # timestamped Log() lines to *stdout* when running non-interactive,
    # so the most diagnostic content (file loading, sea-state setpoints,
    # alerts) is in stdout.
    log_path = run_dir / f"{tag}_seed{seed:04d}.log"
    log_path.write_text(
        "===== STDOUT =====\n" + proc.stdout
        + "\n===== STDERR =====\n" + proc.stderr
    )

    full_log = proc.stdout + "\n" + proc.stderr

    # Hard fail on the silent-loader symptom: brucon logs ERROR but exits
    # cleanly when the response data file is empty or missing. Catch it
    # here so we never compare against a stub-vessel run.
    if "is empty or missing" in full_log:
        bad_lines = [ln for ln in full_log.splitlines() if "is empty or missing" in ln]
        raise RuntimeError(
            f"RunFastStandalone reported missing data file(s) for seed {seed}:\n"
            + "\n".join(bad_lines)
            + f"\nFull log: {log_path}"
        )

    # Verify the response data file we asked for was the one actually loaded.
    # Catches typos in CSOV_RESPONSE_FILE / SIMULATOR_CONFIG_DIR before we
    # waste compute on the wrong vessel.
    expected_response_path = str(SIMULATOR_CONFIG_DIR / CSOV_RESPONSE_FILE)
    if expected_response_path not in full_log:
        raise RuntimeError(
            f"RunFastStandalone did not log the expected response-data path.\n"
            f"  Expected to see: {expected_response_path}\n"
            f"  Full log: {log_path}"
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"RunFastStandalone failed (rc={proc.returncode}) for seed {seed}.\n"
            f"--- stdout (tail) ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr (tail) ---\n{proc.stderr[-2000:]}"
        )

    return parse_output(run_dir / out_name)


def run_ensemble(spec: ScenarioSpec, n_seeds: int, work_dir: Path,
                 tag: str = "run", base_seed: int = 1000,
                 n_workers: int | None = None) -> list[SimResult]:
    """Drive the simulator with n_seeds different wave seeds.

    RunFastStandalone is single-threaded, so multiple seeds are run in
    parallel via a process pool. Each subprocess has its own cwd
    (``work_dir / f"{tag}_seed{seed:04d}"``) so there is no file-write
    contention.

    Parameters
    ----------
    n_workers
        How many simulators to run concurrently. Default: half of
        ``os.cpu_count()`` (so on a 24-thread box we use 12, leaving
        headroom and avoiding HT oversubscription on a single-threaded
        binary). Pass 1 to force sequential execution (useful for
        debugging stack traces).

    Returns
    -------
    Per-seed SimResults in seed order (not completion order).
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) // 2)
    n_workers = min(n_workers, n_seeds)

    seeds = [base_seed + k for k in range(n_seeds)]
    results: list[SimResult | None] = [None] * n_seeds

    if n_workers == 1:
        for k, seed in enumerate(seeds):
            results[k] = run_simulation(spec, seed=seed, work_dir=work_dir, tag=tag)
        return results  # type: ignore[return-value]

    # Process pool: each worker imports this module fresh, so spec is
    # pickled and sent across.
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_to_idx = {
            pool.submit(run_simulation, spec, seed, work_dir, tag): k
            for k, seed in enumerate(seeds)
        }
        for fut in as_completed(future_to_idx):
            k = future_to_idx[fut]
            results[k] = fut.result()        # re-raises subprocess errors

    return results  # type: ignore[return-value]


__all__ = [
    "ScenarioSpec", "SimResult",
    "CSOV_THRUSTER_INDEX", "CSOV_WCF_GROUPS",
    "BRUCON_BIN", "RUNFAST", "CSOV_CONFIG",
    "CSOV_RESPONSE_FILE", "SIMULATOR_CONFIG_DIR",
    "render_lua", "run_simulation", "run_ensemble", "parse_output",
    "SIM_DT",
]
