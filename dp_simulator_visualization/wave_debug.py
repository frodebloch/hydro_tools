#!/usr/bin/env python3
"""Offline diagnostic script for wave elevation mismatch.

Listens on UDP, captures one set of wave parameters + a few vessel data
messages, then computes Python elevation at various positions and compares
with the C++ reported elevation.

Usage:
    python3 wave_debug.py [--port 9000]

Run the dp_simulator + VisualisationInterface, then start this script.
It will collect data for a few seconds, then print a detailed comparison.
"""

import argparse
import json
import socket
import sys
import time

import numpy as np

# Add the package to path
sys.path.insert(0, ".")

from dp_sim_vis.wave_model import WaveElevation, WaveSpectrum, _generate_phases_cpp

PI = np.pi
G = 9.81


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Max seconds to collect data (stops early once wave params + vessel data received)")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(1.0)

    print(f"Listening on UDP port {args.port} (up to {args.duration:.0f}s)...")
    print(f"Waiting for wave parameters... (restart the scenario if needed to trigger resend)")

    wave_params = None
    vessel_samples = []
    sim_times = []

    t_start = time.time()
    while time.time() - t_start < args.duration:
        try:
            data, _ = sock.recvfrom(65536)
            msg = json.loads(data.decode("utf-8"))
        except socket.timeout:
            continue
        except Exception:
            continue

        if "frequencies" in msg and "spectrums" in msg:
            wave_params = msg
            print(f"  Got wave params: {len(msg.get('frequencies',[]))} freqs, "
                  f"{len(msg.get('directions',[]))} dirs, "
                  f"seed={msg.get('randomSeed', 0)}")
            # Reset vessel samples to collect fresh data after wave params
            vessel_samples.clear()
            sim_times.clear()
            t_start = time.time()  # collect for a few more seconds

        if "OceanSimulationTime" in msg:
            sim_times.append(msg["OceanSimulationTime"])

        if "latlon" in msg:
            vessel_samples.append({
                "ned_north": msg.get("ned_north", 0.0),
                "ned_east": msg.get("ned_east", 0.0),
                "heading": msg.get("yaw", 0.0),
                "waveElevation": msg.get("waveElevation", 0.0),
                "sim_time": sim_times[-1] if sim_times else 0.0,
            })

        # Stop early: once we have wave params + enough vessel data
        if wave_params and len(vessel_samples) >= 30:
            print(f"  Collected {len(vessel_samples)} vessel samples. Analyzing...")
            break

    sock.close()

    if wave_params is None:
        print("\nERROR: No wave parameters received! Make sure the simulator is "
              "running and wave_sync_delay has elapsed.")
        return

    if not vessel_samples:
        print("\nERROR: No vessel data received!")
        return

    # ── Parse wave parameters ──────────────────────────────────────
    freqs = np.array(wave_params["frequencies"])
    dirs = np.array(wave_params["directions"])
    n_freq = len(freqs)
    n_dir = len(dirs)
    seed = wave_params.get("randomSeed", 0)

    print(f"\n{'='*60}")
    print(f"WAVE PARAMETERS")
    print(f"{'='*60}")
    print(f"  Frequencies: {n_freq} values")
    print(f"    first 5: {freqs[:5]}")
    print(f"    last 5:  {freqs[-5:]}")
    print(f"    range:   [{freqs.min():.4f}, {freqs.max():.4f}] rad/s")
    print(f"    descending: {np.all(np.diff(freqs) <= 0)}")
    print(f"  Directions: {n_dir} values")
    print(f"    first 5: {dirs[:5]}")
    print(f"    last 5:  {dirs[-5:]}")
    print(f"    step:    {dirs[1]-dirs[0]:.4f} deg")
    print(f"  Random seed: {seed}")

    spectrums = wave_params["spectrums"]
    for i, s in enumerate(spectrums):
        print(f"  Spectrum {i}: Hs={s['significantWaveHeight']:.2f}m, "
              f"Tp={s['peakPeriod']:.2f}s, "
              f"Dir={s['dominantDirection']:.1f}deg, "
              f"Spreading={s['spreadingFactor']}")

    # ── Build wave model from seed + spectrum params ───────────────
    # Generate phases from seed (matching C++ MT19937 + generate_canonical)
    phases = _generate_phases_cpp(seed, n_freq, n_dir)
    print(f"\n  Phases generated from seed {seed}")
    print(f"    first 5: {phases.ravel()[:5]}")

    # Build wave elevation model and add spectra
    wave_elev = WaveElevation(freqs, dirs, random_seed=seed)
    for s in spectrums:
        if s["significantWaveHeight"] > 0:
            wave_elev.add_spectrum(WaveSpectrum(
                hs=s["significantWaveHeight"],
                tp=s["peakPeriod"],
                direction_deg=s["dominantDirection"],
                spreading_factor=s["spreadingFactor"],
            ))

    # Get amplitude info after spectra are set
    # Force amplitude computation
    wave_elev._compute_amplitudes()
    amps = wave_elev.amplitudes
    print(f"\n  Amplitudes computed from spectra")
    print(f"    max amplitude: {amps.max():.6f} m")
    print(f"    nonzero count: {np.count_nonzero(amps > 1e-10)}")
    active_dir = np.any(amps > 1e-10, axis=0)
    active_indices = np.nonzero(active_dir)[0]
    print(f"    active directions: {len(active_indices)} indices: {active_indices}")
    print(f"    active dir values: {dirs[active_indices]}")

    # ── Compute Python elevation using seed-based params ──────────
    active_idx = active_indices
    dir_rad = np.deg2rad(dirs[active_idx])
    cos_dir = np.cos(dir_rad)
    sin_dir = np.sin(dir_rad)
    k = freqs**2 / G
    amps_active = amps[:, active_idx]
    phases_active = phases[:, active_idx]

    def elevation_f64(t, north, east):
        """Compute elevation at (north, east) using float64."""
        spatial_proj = cos_dir * north + sin_dir * east  # (n_active,)
        elev = 0.0
        for i in range(n_freq):
            phase = freqs[i] * t + k[i] * spatial_proj + phases_active[i]
            elev += np.dot(amps_active[i], np.cos(phase))
        return elev

    def elevation_f32(t, north, east):
        """Compute elevation using float32 (matching the viz code)."""
        north_f32 = np.float32(north)
        east_f32 = np.float32(east)
        cos_dir_f32 = cos_dir.astype(np.float32)
        sin_dir_f32 = sin_dir.astype(np.float32)
        k_f32 = k.astype(np.float32)
        freqs_f32 = freqs.astype(np.float32)
        amps_f32 = amps_active.astype(np.float32)
        phases_f32 = phases_active.astype(np.float32)

        spatial_proj = cos_dir_f32 * north_f32 + sin_dir_f32 * east_f32
        elev = np.float32(0.0)
        for i in range(n_freq):
            phase = freqs_f32[i] * np.float32(t) + k_f32[i] * spatial_proj + phases_f32[i]
            elev += np.dot(amps_f32[i], np.cos(phase))
        return float(elev)

    # Take the last vessel sample and corresponding time
    last_sample = vessel_samples[-1]
    last_time = last_sample["sim_time"]
    north = last_sample["ned_north"]
    east = last_sample["ned_east"]
    cpp_elev = last_sample["waveElevation"]

    print(f"\n{'='*60}")
    print(f"ELEVATION COMPARISON")
    print(f"{'='*60}")
    print(f"  Vessel position: N={north:.4f}m, E={east:.4f}m")
    print(f"  Vessel heading: {last_sample['heading']:.2f} deg")
    print(f"  Sim time (OceanSimulationTime): {last_time:.4f}s")
    print(f"  C++ reported wave elevation: {cpp_elev:+.6f}m")

    # Try different time offsets
    for dt_label, dt in [("t", 0.0), ("t-0.1", -0.1), ("t+0.1", 0.1)]:
        t = last_time + dt
        py_f64 = elevation_f64(t, north, east)
        py_f32 = elevation_f32(t, north, east)
        py_f64_origin = elevation_f64(t, 0.0, 0.0)
        diff_f64 = py_f64 - cpp_elev
        diff_f32 = py_f32 - cpp_elev
        diff_origin = py_f64_origin - cpp_elev
        print(f"\n  Time = {dt_label} = {t:.4f}s:")
        print(f"    Py f64 at ({north:.1f}, {east:.1f}): {py_f64:+.6f}m  diff={diff_f64:+.6f}m")
        print(f"    Py f32 at ({north:.1f}, {east:.1f}): {py_f32:+.6f}m  diff={diff_f32:+.6f}m")
        print(f"    Py f64 at (0, 0):         {py_f64_origin:+.6f}m  diff={diff_origin:+.6f}m")

    # Also check elevation at several positions along north axis
    print(f"\n  Elevation along north axis (t={last_time - 0.1:.2f}s, east=0):")
    t_compare = last_time - 0.1
    for n_pos in [0, 1, 2, 5, 10, 50, 100, 500]:
        py_val = elevation_f64(t_compare, float(n_pos), 0.0)
        print(f"    N={n_pos:>4d}m: Py={py_val:+.6f}m")

    # Sanity check: compute at (0,0) with t=0 — should be sum of amp*cos(phase)
    elev_00_t0 = elevation_f64(0.0, 0.0, 0.0)
    manual = np.sum(amps_active * np.cos(phases_active))
    print(f"\n  Sanity check: elevation(0, 0, 0):")
    print(f"    Computed:  {elev_00_t0:+.6f}m")
    print(f"    Manual:    {manual:+.6f}m")
    print(f"    Match: {abs(elev_00_t0 - manual) < 1e-10}")

    # ── Multi-sample comparison ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"MULTI-SAMPLE COMPARISON (best time offset: t-0.1)")
    print(f"{'='*60}")
    print(f"  {'Sample':>6} {'Time':>8} {'North':>8} {'East':>8} "
          f"{'C++':>10} {'Py_f64':>10} {'Diff':>10} {'|Diff|':>8}")
    for idx, s in enumerate(vessel_samples):
        t = s["sim_time"] - 0.1
        n, e = s["ned_north"], s["ned_east"]
        cpp_e = s["waveElevation"]
        py_e = elevation_f64(t, n, e)
        diff = py_e - cpp_e
        print(f"  {idx:>6d} {s['sim_time']:>8.2f} {n:>+8.2f} {e:>+8.2f} "
              f"{cpp_e:>+10.4f} {py_e:>+10.4f} {diff:>+10.4f} {abs(diff):>8.4f}")

    # ── Additional: check if vessel_samples have consistent positions ──
    if len(vessel_samples) > 3:
        norths = [s["ned_north"] for s in vessel_samples]
        easts = [s["ned_east"] for s in vessel_samples]
        print(f"\n  Vessel position range over {len(vessel_samples)} samples:")
        print(f"    North: [{min(norths):.2f}, {max(norths):.2f}]")
        print(f"    East:  [{min(easts):.2f}, {max(easts):.2f}]")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
